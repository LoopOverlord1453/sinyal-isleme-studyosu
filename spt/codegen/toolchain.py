"""Derleyici araç zinciri keşfi; çapraz derleme, ELF bölüm boyutları, bağlama ve statik yığın analizi.

Genel amaçlıdır (füzyon dışındaki C üreteçleri de kullanabilir): verilen C kaynaklarını bir MCU profili, derleyici
ve iyileştirme düzeyiyle derler, nesne / imaj boyutlarını döndürür. Araç bulunamazsa işlevler hata fırlatmaz;
`None` ya da `ok=False` döner (çağıran tahmine düşer).

Keşif sırası (her araç için): ortam değişkeni (SPT_ARM_GCC, SPT_CLANG, SPT_HOST_CC, SPT_HOST_CXX — exe ya da bin
klasörü), PATH, yaygın kurulum klasörleri (STM32CubeIDE / CubeCLT eklentileri, Arm GNU Toolchain, LLVM, MinGW-w64).
Birden çok aday varsa en yeni sürüm seçilir. Derleme çıktıları `%TEMP%/spt_codegen/build/<özet>` altında içerik
özetine göre önbelleklenir (aynı kaynak + bayrak + araç → yeniden derlenmez).
"""
from __future__ import annotations

import functools
import glob
import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

from spt.codegen.mcu import McuProfile

WORK_ROOT = Path(os.environ.get("SPT_CODEGEN_DIR") or Path(tempfile.gettempdir()) / "spt_codegen")
WARN_FLAGS = ("-Wall", "-Wextra", "-pedantic", "-Werror")
_EXE = ".exe" if os.name == "nt" else ""
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


# ============================================================================ araç keşfi
@dataclass(frozen=True)
class Tool:
    kind: str           # "arm-gcc" | "clang" | "host-cc" | "host-cxx"
    path: str
    version: str
    machine: str = ""   # -dumpmachine

    @property
    def bin_dir(self) -> Path:
        return Path(self.path).parent

    @property
    def label(self) -> str:
        return f"{Path(self.path).stem} {self.version}"

    def sibling(self, name: str) -> str | None:
        """Aynı bin klasöründeki yardımcı araç (ör. arm-none-eabi-objdump, llvm-objdump)."""
        p = self.bin_dir / (name + _EXE)
        return str(p) if p.exists() else None


def _run(cmd: Sequence[str], cwd: Path | str | None = None, timeout: float = 180.0,
         path_prefix: Iterable[Path] = ()) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    prefix = os.pathsep.join(str(p) for p in path_prefix)
    if prefix:
        env["PATH"] = prefix + os.pathsep + env.get("PATH", "")
    return subprocess.run([str(c) for c in cmd], cwd=None if cwd is None else str(cwd), capture_output=True,
                          text=True, encoding="utf-8", errors="replace", timeout=timeout, env=env,
                          creationflags=_NO_WINDOW)


def _probe(path: str, kind: str) -> Tool | None:
    try:
        ver = _run([path, "-dumpversion"], timeout=30, path_prefix=[Path(path).parent])
        mach = _run([path, "-dumpmachine"], timeout=30, path_prefix=[Path(path).parent])
    except (OSError, subprocess.SubprocessError):
        return None
    if ver.returncode != 0:
        return None
    return Tool(kind, str(Path(path)), ver.stdout.strip(), mach.stdout.strip())


def _version_key(tool: Tool) -> tuple:
    nums = tuple(int(n) for n in re.findall(r"\d+", tool.version)[:4])
    natural = tuple(int(t) if t.isdigit() else t for t in re.split(r"(\d+)", tool.path.lower()))
    return nums, natural


def _env_candidates(var: str, names: Sequence[str]) -> list[str]:
    value = os.environ.get(var, "").strip().strip('"')
    if not value:
        return []
    p = Path(value)
    if p.is_file():
        return [str(p)]
    out = []
    for d in (p, p / "bin"):
        for n in names:
            if (d / (n + _EXE)).is_file():
                out.append(str(d / (n + _EXE)))
    return out


def _which(names: Sequence[str]) -> list[str]:
    return [w for n in names if (w := shutil.which(n))]


def _globs(patterns: Sequence[str]) -> list[str]:
    out: list[str] = []
    for pat in patterns:
        out.extend(glob.glob(pat))
    return out


def _unique(paths: Iterable[str]) -> list[str]:
    seen, out = set(), []
    for p in paths:
        key = os.path.normcase(os.path.abspath(p))
        if key not in seen and os.path.isfile(p):
            seen.add(key)
            out.append(p)
    return out


_ARM_GCC_GLOBS = (
    "C:/ST/STM32CubeIDE*/STM32CubeIDE/plugins/*gnu-tools-for-stm32*/tools/bin/arm-none-eabi-gcc.exe",
    "C:/ST/STM32CubeCLT*/GNU-tools-for-STM32/bin/arm-none-eabi-gcc.exe",
    "C:/Program Files*/Arm GNU Toolchain*/*/bin/arm-none-eabi-gcc.exe",
    "C:/Program Files*/Arm/GNU Toolchain*/bin/arm-none-eabi-gcc.exe",
    "C:/Program Files*/GNU Arm Embedded Toolchain/*/bin/arm-none-eabi-gcc.exe",
    "C:/Program Files*/GNU Tools ARM Embedded/*/bin/arm-none-eabi-gcc.exe",
    "/usr/bin/arm-none-eabi-gcc", "/usr/local/bin/arm-none-eabi-gcc", "/opt/*/bin/arm-none-eabi-gcc",
    "/Applications/ArmGNUToolchain/*/*/bin/arm-none-eabi-gcc",
)
_CLANG_GLOBS = ("C:/mingw64/bin/clang.exe", "C:/Program Files/LLVM/bin/clang.exe", "C:/msys64/*/bin/clang.exe",
                "/usr/bin/clang", "/usr/local/opt/llvm/bin/clang", "/opt/homebrew/opt/llvm/bin/clang")
_HOST_CC_GLOBS = ("C:/mingw64/bin/gcc.exe", "C:/msys64/ucrt64/bin/gcc.exe", "C:/msys64/mingw64/bin/gcc.exe",
                  "C:/mingw64/bin/clang.exe")
_HOST_CXX_GLOBS = ("C:/mingw64/bin/g++.exe", "C:/msys64/ucrt64/bin/g++.exe", "C:/msys64/mingw64/bin/g++.exe",
                   "C:/mingw64/bin/clang++.exe")


@functools.lru_cache(maxsize=None)
def find_arm_gcc() -> Tool | None:
    """arm-none-eabi-gcc (en yeni sürüm; eşitlikte en yeni eklenti klasörü)."""
    names = ("arm-none-eabi-gcc",)
    env = _env_candidates("SPT_ARM_GCC", names)
    for cands in (env, _unique(_which(names) + _globs(_ARM_GCC_GLOBS))):
        tools = [t for c in cands if (t := _probe(c, "arm-gcc"))]
        if tools:
            return max(tools, key=_version_key)
    return None


@functools.lru_cache(maxsize=None)
def _clang_has_arm(path: str) -> bool:
    try:
        out = _run([path, "--print-targets"], timeout=30, path_prefix=[Path(path).parent])
    except (OSError, subprocess.SubprocessError):
        return False
    return out.returncode == 0 and re.search(r"^\s*thumb\s", out.stdout, re.M) is not None


@functools.lru_cache(maxsize=None)
def find_clang() -> Tool | None:
    """ARM / Thumb hedefi derlenmiş clang (Arm Compiler 6 = clang tabanlı; boyutlar benzerdir)."""
    names = ("clang",)
    env = _env_candidates("SPT_CLANG", names)
    for cands in (env, _unique(_which(names) + _globs(_CLANG_GLOBS))):
        tools = [t for c in cands if _clang_has_arm(c) and (t := _probe(c, "clang"))]
        if tools:
            return max(tools, key=_version_key)
    return None


def _host_ok(tool: Tool, cxx: bool) -> bool:
    """Gerçekten çalışan bir program üretebiliyor mu (64 bit tercih)."""
    d = WORK_ROOT / "probe" / hashlib.sha1(f"{tool.path}{tool.version}{cxx}".encode()).hexdigest()[:12]
    marker = d / "ok"
    if marker.exists():
        return True
    d.mkdir(parents=True, exist_ok=True)
    src = d / ("t.cpp" if cxx else "t.c")
    src.write_text("#include <stdio.h>\nint main(void){printf(\"ok\");return 0;}\n", encoding="utf-8")
    exe = d / ("t" + _EXE)
    try:
        r = _run([tool.path, str(src), "-o", str(exe), "-static"], cwd=d, timeout=120, path_prefix=[tool.bin_dir])
        if r.returncode != 0:
            return False
        out = _run([str(exe)], cwd=d, timeout=30, path_prefix=[tool.bin_dir])
    except (OSError, subprocess.SubprocessError):
        return False
    if out.returncode == 0 and out.stdout == "ok":
        marker.write_text("ok")
        return True
    return False


def _find_host(var: str, names: Sequence[str], globs: Sequence[str], kind: str) -> Tool | None:
    env = _env_candidates(var, names)
    for cands in (env, _unique(_globs(globs) + _which(names))):
        tools = [t for c in cands if (t := _probe(c, kind))]
        tools.sort(key=lambda t: 0 if re.match(r"(x86_64|amd64|aarch64|arm64)", t.machine) else 1)  # 64 bit önce
        for t in tools:
            if _host_ok(t, kind == "host-cxx"):
                return t
    return None


@functools.lru_cache(maxsize=None)
def find_host_cc() -> Tool | None:
    return _find_host("SPT_HOST_CC", ("gcc", "cc", "clang"), _HOST_CC_GLOBS, "host-cc")


@functools.lru_cache(maxsize=None)
def find_host_cxx() -> Tool | None:
    return _find_host("SPT_HOST_CXX", ("g++", "c++", "clang++"), _HOST_CXX_GLOBS, "host-cxx")


def clear_cache() -> None:
    """Araç keşfi önbelleğini boşaltır (ortam değişkeni değişince)."""
    for f in (find_arm_gcc, find_clang, find_host_cc, find_host_cxx, _clang_has_arm):
        f.cache_clear()


def cross_tool(compiler: str) -> Tool | None:
    return {"gcc": find_arm_gcc, "clang": find_clang}.get(compiler, lambda: None)()


def arm_sysroot() -> Path | None:
    """newlib başlıkları / kütüphaneleri (arm-none-eabi-gcc'nin yanındaki arm-none-eabi klasörü)."""
    gcc = find_arm_gcc()
    if gcc is None:
        return None
    root = gcc.bin_dir.parent / "arm-none-eabi"
    return root if (root / "include").is_dir() else None


def report() -> dict[str, str]:
    """Bulunan araçların özeti (arayüz / tanılama için)."""
    out = {}
    for key, fn in (("arm-gcc", find_arm_gcc), ("clang", find_clang), ("host-cc", find_host_cc),
                    ("host-cxx", find_host_cxx)):
        t = fn()
        out[key] = f"{t.label} — {t.path}" if t else "bulunamadı"
    return out


# ============================================================================ ELF okuma
_SHF_WRITE, _SHF_ALLOC, _SHF_EXEC = 0x1, 0x2, 0x4
_SHT_SYMTAB, _SHT_NOBITS = 2, 8


@dataclass(frozen=True)
class Section:
    name: str
    type: int
    flags: int
    addr: int
    offset: int
    size: int
    link: int


@dataclass(frozen=True)
class Symbol:
    name: str
    value: int
    size: int
    shndx: int
    kind: int     # STT_*: 1 nesne, 2 fonksiyon


def _elf(data: bytes) -> tuple[str, int, list[Section]]:
    if data[:4] != b"\x7fELF":
        raise ValueError("ELF değil")
    e = "<" if data[5] == 1 else ">"
    if data[4] == 1:
        shoff = struct.unpack_from(e + "I", data, 0x20)[0]
        shentsize, shnum, shstrndx = struct.unpack_from(e + "HHH", data, 0x2E)
        fmt, idx = e + "IIIIIIIIII", (0, 1, 2, 3, 4, 5, 6)
    else:
        shoff = struct.unpack_from(e + "Q", data, 0x28)[0]
        shentsize, shnum, shstrndx = struct.unpack_from(e + "HHH", data, 0x3A)
        fmt, idx = e + "IIQQQQIIQQ", (0, 1, 2, 3, 4, 5, 6)
    raw = [struct.unpack_from(fmt, data, shoff + i * shentsize) for i in range(shnum)]
    strtab = raw[shstrndx]
    names_off, names_size = strtab[4], strtab[5]
    names = data[names_off:names_off + names_size]

    def name(o: int) -> str:
        end = names.find(b"\0", o)
        return names[o:end].decode("ascii", "replace")

    secs = [Section(name(r[idx[0]]), r[idx[1]], r[idx[2]], r[idx[3]], r[idx[4]], r[idx[5]], r[idx[6]])
            for r in raw]
    return e, data[4], secs


@dataclass(frozen=True)
class Sizes:
    """Bayt: text (kod), rodata (sabitler, .ARM.exidx dahil), data (ilk değerli RAM), bss (sıfırlanan RAM)."""
    text: int = 0
    rodata: int = 0
    data: int = 0
    bss: int = 0

    @property
    def flash(self) -> int:
        return self.text + self.rodata + self.data

    @property
    def ram(self) -> int:
        return self.data + self.bss

    def __add__(self, o: Sizes) -> Sizes:
        return Sizes(self.text + o.text, self.rodata + o.rodata, self.data + o.data, self.bss + o.bss)

    def __sub__(self, o: Sizes) -> Sizes:
        return Sizes(self.text - o.text, self.rodata - o.rodata, self.data - o.data, self.bss - o.bss)


def elf_sizes(path: Path | str) -> Sizes:
    """Nesne ya da bağlanmış imajın ALLOC bölümleri sınıflanarak toplanır (araçtan bağımsız)."""
    _, _, secs = _elf(Path(path).read_bytes())
    acc = {"text": 0, "rodata": 0, "data": 0, "bss": 0}
    for s in secs:
        if not s.flags & _SHF_ALLOC:
            continue
        if s.type == _SHT_NOBITS:
            acc["bss"] += s.size
        elif s.flags & _SHF_EXEC:
            acc["text"] += s.size
        elif s.flags & _SHF_WRITE:
            acc["data"] += s.size
        else:
            acc["rodata"] += s.size
    return Sizes(**acc)


def elf_symbols(path: Path | str) -> dict[str, Symbol]:
    data = Path(path).read_bytes()
    e, cls, secs = _elf(data)
    out: dict[str, Symbol] = {}
    for s in secs:
        if s.type != _SHT_SYMTAB:
            continue
        strsec = secs[s.link]
        strs = data[strsec.offset:strsec.offset + strsec.size]
        ent = 16 if cls == 1 else 24
        for off in range(s.offset, s.offset + s.size, ent):
            if cls == 1:
                st_name, value, size, info, _, shndx = struct.unpack_from(e + "IIIBBH", data, off)
            else:
                st_name, info, _, shndx, value, size = struct.unpack_from(e + "IBBHQQ", data, off)
            if not st_name:
                continue
            end = strs.find(b"\0", st_name)
            nm = strs[st_name:end].decode("ascii", "replace")
            out[nm] = Symbol(nm, value, size, shndx, info & 0xF)
    return out


# ============================================================================ derleme
@dataclass
class Build:
    ok: bool
    log: str
    workdir: Path
    tool: Tool | None
    objects: dict[str, Path] = field(default_factory=dict)   # kaynak adı → nesne dosyası
    output: Path | None = None                                # bağlanmış imaj / çalıştırılabilir
    command: list[str] = field(default_factory=list)

    def sizes(self, source: str) -> Sizes:
        return elf_sizes(self.objects[source])

    def stack_usage(self) -> dict[str, int]:
        """-fstack-usage (.su) çıktısı: fonksiyon → kendi çerçevesi (bayt; çağrılanlar hariç)."""
        out: dict[str, int] = {}
        for su in self.workdir.glob("*.su"):
            for line in su.read_text(encoding="utf-8", errors="replace").splitlines():
                parts = line.split("\t")
                if len(parts) >= 2 and parts[1].strip().isdigit():
                    out[parts[0].rsplit(":", 1)[-1]] = int(parts[1])
        return out


_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def _key_lock(key: str) -> threading.Lock:
    with _locks_guard:
        return _locks.setdefault(key, threading.Lock())


def _cached_build(kind: str, tool: Tool, files: Mapping[str, str], recipe: list, runner) -> Build:
    """İçerik özetine göre önbellekli derleme: aynı girdide önceki sonuç (başarısız olsa da) döner."""
    blob = json.dumps([kind, tool.path, tool.version, sorted(files.items()), recipe], ensure_ascii=False)
    key = hashlib.sha256(blob.encode("utf-8")).hexdigest()[:24]
    workdir = WORK_ROOT / "build" / key
    meta = workdir / "build.json"
    with _key_lock(key):
        if meta.exists():
            try:
                m = json.loads(meta.read_text(encoding="utf-8"))
                b = Build(m["ok"], m["log"], workdir, tool, {k: workdir / v for k, v in m["objects"].items()},
                          workdir / m["output"] if m.get("output") else None, m.get("command", []))
                if all(p.exists() for p in b.objects.values()) and (b.output is None or b.output.exists()):
                    return b
            except (OSError, ValueError, KeyError):
                pass
        workdir.mkdir(parents=True, exist_ok=True)
        for name, text in files.items():
            p = workdir / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text, encoding="utf-8", newline="\n")
        try:
            b = runner(workdir)
        except (OSError, subprocess.SubprocessError) as exc:
            b = Build(False, f"{type(exc).__name__}: {exc}", workdir, tool)
        meta.write_text(json.dumps({"ok": b.ok, "log": b.log,
                                    "objects": {k: v.name for k, v in b.objects.items()},
                                    "output": b.output.name if b.output else None, "command": b.command},
                                   ensure_ascii=False), encoding="utf-8")
        return b


_MATH_SHIM = """/* Asgari math.h (newlib başlıkları bulunamadığında yalnız boyut ölçümü için) */
#ifndef SPT_MATH_SHIM_H
#define SPT_MATH_SHIM_H
float sqrtf(float); float sinf(float); float cosf(float); float atan2f(float, float); float asinf(float);
float expf(float); float logf(float); float fabsf(float); float fmodf(float, float); float hypotf(float, float);
double sqrt(double); double sin(double); double cos(double); double atan2(double, double); double asin(double);
double exp(double); double log(double); double fabs(double); double fmod(double, double); double hypot(double, double);
#endif
"""


def cross_flags(profile: McuProfile, compiler: str) -> list[str] | None:
    """Profil + derleyici için hedef bayrakları; derleyici ya da çekirdek desteği yoksa None."""
    if compiler == "gcc":
        return list(profile.gcc_flags) if profile.gcc_flags is not None and find_arm_gcc() else None
    if compiler == "clang":
        if profile.clang_flags is None or find_clang() is None:
            return None
        flags = list(profile.clang_flags) + ["-fshort-enums"]   # newlib (GCC) ile aynı enum ABI'si
        root = arm_sysroot()
        flags += [f"--sysroot={root}"] if root else ["-isystem", "spt_shim"]
        return flags
    return None


def compile_cross(files: Mapping[str, str], sources: Sequence[str], profile: McuProfile, compiler: str,
                  opt: str = "-Os", *, defines: Sequence[str] = (), extra: Sequence[str] = (), std: str = "c99",
                  warnings: Sequence[str] = WARN_FLAGS, stack_usage: bool = True) -> Build:
    """Kaynakları nesneye derler (-c, bölüm başına fonksiyon / veri). Uyarılar -Werror ile hatadır."""
    tool = cross_tool(compiler)
    flags = cross_flags(profile, compiler)
    if tool is None or flags is None:
        return Build(False, f"{compiler}: {profile.key} için derleyici yok", WORK_ROOT, None)
    files = dict(files)
    if "spt_shim" in flags:
        files["spt_shim/math.h"] = _MATH_SHIM
    cmd = [tool.path, *flags, opt, f"-std={std}", *warnings, "-ffunction-sections", "-fdata-sections",
           *(["-fstack-usage"] if stack_usage else []), *(f"-D{d}" for d in defines), "-I.", *extra, "-c",
           *sources]

    def runner(wd: Path) -> Build:
        r = _run(cmd, cwd=wd, path_prefix=[tool.bin_dir])
        objs = {s: wd / (Path(s).stem + ".o") for s in sources}
        ok = r.returncode == 0 and all(p.exists() for p in objs.values())
        return Build(ok, (r.stdout + r.stderr).strip(), wd, tool, objs if ok else {}, None, cmd)

    return _cached_build("cross", tool, files, cmd, runner)


def link_cross(files: Mapping[str, str], sources: Sequence[str], profile: McuProfile, compiler: str,
               opt: str = "-Os", *, defines: Sequence[str] = (), extra: Sequence[str] = (), std: str = "c99",
               libs: Sequence[str] = ("-lm",), output: str = "image.elf") -> Build:
    """Derle (compiler) + bağla (arm-none-eabi-gcc, newlib-nano, --gc-sections, giriş = main, başlangıç kodu yok).

    Bağlanmış imaj, kullanılan libm / libgcc (yazılım kayan nokta) parçalarını içerir; boş bir temel imajla farkı
    alınarak kütüphane dahil gerçek flash maliyeti bulunur."""
    tool, gcc = cross_tool(compiler), find_arm_gcc()
    flags = cross_flags(profile, compiler)
    if tool is None or gcc is None or flags is None or profile.gcc_flags is None:
        return Build(False, f"{compiler}: {profile.key} için bağlayıcı / derleyici yok", WORK_ROOT, None)
    files = dict(files)
    if "spt_shim" in flags:
        files["spt_shim/math.h"] = _MATH_SHIM
    ccmd = [tool.path, *flags, opt, f"-std={std}", "-ffunction-sections", "-fdata-sections",
            *(f"-D{d}" for d in defines), "-I.", *extra, "-c", *sources]
    objs = [Path(s).stem + ".o" for s in sources]
    lcmd = [gcc.path, *profile.gcc_flags, opt, "-nostartfiles", "-Wl,--gc-sections", "-Wl,-e,main",
            "--specs=nano.specs", "--specs=nosys.specs", *objs, *libs, "-o", output]

    def runner(wd: Path) -> Build:
        r = _run(ccmd, cwd=wd, path_prefix=[tool.bin_dir])
        log = r.stdout + r.stderr
        if r.returncode != 0:
            return Build(False, log.strip(), wd, tool, {}, None, ccmd)
        r2 = _run(lcmd, cwd=wd, path_prefix=[gcc.bin_dir])
        ok = r2.returncode == 0 and (wd / output).exists()
        return Build(ok, (log + r2.stdout + r2.stderr).strip(), wd, tool,
                     {s: wd / o for s, o in zip(sources, objs)} if ok else {}, wd / output if ok else None,
                     ccmd + ["&&"] + lcmd)

    return _cached_build("link", tool, files, [ccmd, lcmd, gcc.path, gcc.version], runner)


def host_build(files: Mapping[str, str], sources: Sequence[str], *, cxx: bool = False,
               defines: Sequence[str] = (), extra: Sequence[str] = (), opt: str = "-O2",
               output: str = "prog") -> Build:
    """Ana makine için çalıştırılabilir (statik bağlı: PATH'teki farklı MinGW DLL'leri karışmaz).

    cxx=True: .c kaynakları C++ olarak derlenir (ör. işlem sayacı sınıfıyla)."""
    tool = find_host_cxx() if cxx else find_host_cc()
    if tool is None:
        return Build(False, "ana makine derleyicisi yok", WORK_ROOT, None)
    exe = output + _EXE
    lang = ["-x", "c++"] if cxx else []
    cmd = [tool.path, opt, "-ffp-contract=off", *(f"-D{d}" for d in defines), "-I.", *extra, *lang, *sources,
           *(["-x", "none"] if cxx else []), "-o", exe, "-static"]

    def runner(wd: Path) -> Build:
        r = _run(cmd, cwd=wd, path_prefix=[tool.bin_dir], timeout=300)
        ok = r.returncode == 0 and (wd / exe).exists()
        return Build(ok, (r.stdout + r.stderr).strip(), wd, tool, {}, wd / exe if ok else None, cmd)

    return _cached_build("host", tool, files, cmd, runner)


def run_program(build: Build, args: Sequence[str], timeout: float = 300.0) -> subprocess.CompletedProcess:
    if not build.ok or build.output is None:
        raise RuntimeError("program derlenemedi:\n" + build.log)
    prefix = [build.tool.bin_dir] if build.tool else []
    return _run([str(build.output), *args], cwd=build.workdir, timeout=timeout, path_prefix=prefix)


# ============================================================================ statik yığın analizi
@dataclass(frozen=True)
class StackInfo:
    bytes: int                    # en kötü çağrı yolu boyunca toplam yığın (bayt)
    path: tuple[str, ...]         # o yol
    unknown: tuple[str, ...]      # dolaylı çağrı / özyineleme / çözülemeyen yığın ayarı olan fonksiyonlar
    frame: int = 0                # giriş fonksiyonunun kendi çerçevesi (bayt; -fstack-usage ile karşılaştırılabilir)

    @property
    def exact(self) -> bool:
        return not self.unknown


STACK_ANALYSIS_VERSION = 4   # çözümleyici değişince ölçüm önbellekleri geçersiz olsun (anahtarlara girer)

_FUNC_RE = re.compile(r"^([0-9a-fA-F]+) <([^>]+)>:\s*$")
# Sabit havuzu: gcc "  90:\t.word\t0xfffffc14", llvm "  d0: 68 c1 ff ff \t.word\t0xffffc168"
_WORD_RE = re.compile(r"^\s*([0-9a-fA-F]+):\s+(?:[0-9a-fA-F]{2}\s+){0,4}\.word\s+0x([0-9a-fA-F]+)")
# ldr rN, [pc, #imm] → yorumdaki sabit adresi: gcc "@ (120 <f+0x88>)", llvm "@ 0xd0 <f+0x6c>"
_LDR_LIT_RE = re.compile(r"(r\d+|ip),\s*\[pc,\s*#(?:0x)?[0-9a-fA-F]+\]")
_LIT_ADDR_RE = re.compile(r"@\s*\(?\s*(?:0x)?([0-9a-fA-F]+)")
_INS_RE = re.compile(r"^\s*([0-9a-fA-F]+):\s+([a-z][\w.]*)\s*(.*)$")
_TARGET_RE = re.compile(r"<([^>+]+)(\+0x[0-9a-fA-F]+)?>")
_BRANCH_RE = re.compile(r"^b(eq|ne|cs|hs|cc|lo|mi|pl|vs|vc|hi|ls|ge|lt|gt|le|al)?(\.n|\.w)?$")


def _reg_count(operands: str) -> tuple[int, int]:
    """{r4-r7, lr} → (kayıt sayısı, bayt/kayıt); d kayıtları 8, s kayıtları 4, çekirdek kayıtları 4 bayt."""
    m = re.search(r"\{([^}]*)\}", operands)
    if not m:
        return 0, 4
    count, width = 0, 4
    for part in m.group(1).split(","):
        part = part.strip()
        if not part:
            continue
        width = 8 if part.startswith("d") else 4
        r = re.match(r"[rsd](\d+)\s*-\s*[rsd](\d+)", part)
        count += int(r.group(2)) - int(r.group(1)) + 1 if r else 1
    return count, width


_NO_DEST = ("str", "stm", "push", "pop", "cmp", "cmn", "tst", "teq", "nop", "dmb", "dsb", "isb", "it", "cb", "bl", "bx",
            "svc", "bkpt", "wfi", "wfe", "cps")


def _frame(ins: list[tuple[str, str]], literals: dict[int, int] | None = None) -> tuple[int, bool]:
    """Fonksiyonun tüm yığın azaltmalarının toplamı (yol duyarsız üst sınır) ve çözülemeyen ayar var mı.

    ARMv6-M (Cortex-M0+) 508 bayttan büyük çerçeveyi `ldr rN, [pc, #..]` + `add sp, rN` (negatif sabit sabit havuzunda)
    ya da `movs rN, #k; lsls rN, rN, #s; add sp, rN` ile ayırır / geri verir: yazmaçlardaki sabitler izlenir (sabit
    havuzu dahil), üzerine bilinmeyen komutla yazılan yazmaç unutulur; çözülemeyen `add/sub sp, rN` "bilinmiyor"."""
    literals = literals or {}
    total, unknown = 0, False
    regs: dict[str, int] = {}   # yazmaçtaki bilinen (işaretli) sabit
    for mn, ops in ins:
        ops_l = ops.split("@")[0].strip()
        base = mn.split(".")[0]
        dest = re.match(r"(r\d+|ip|fp|sl|sb)\b", ops_l)
        written = None              # bu komutun bilinen değerle yazdığı yazmaç
        if base == "push" or (base in ("stmdb", "stmfd") and ops_l.startswith("sp!")):
            n, w = _reg_count(ops_l)
            total += n * w
        elif base in ("vpush", "vstmdb") and (base == "vpush" or ops_l.startswith("sp!")):
            n, w = _reg_count(ops_l)
            total += n * w
        elif base in ("sub", "subw", "subs") and re.match(r"sp,\s*(sp,\s*)?", ops_l):
            m = re.match(r"sp,\s*(?:sp,\s*)?#(0x[0-9a-fA-F]+|\d+)", ops_l)
            if m:
                total += int(m.group(1), 0)
            else:
                r = re.match(r"sp,\s*(?:sp,\s*)?(r\d+|ip)", ops_l)
                if r and r.group(1) in regs:
                    total += max(regs[r.group(1)], 0)
                else:
                    unknown = True
        elif base in ("add", "adds") and (m := re.match(r"sp,\s*(?:sp,\s*)?(r\d+|ip)\s*$", ops_l)):
            value = regs.get(m.group(1))
            if value is None:
                unknown = True             # sp'ye bilinmeyen değer eklendi
            elif value < 0:
                total += -value            # negatif sabit: yığın ayırma (pozitif olan bitişte geri verme)
        elif base in ("str", "strd") and (m := re.search(r"\[sp,\s*#-(0x[0-9a-fA-F]+|\d+)\]!", ops_l)):
            total += int(m.group(1), 0)
        elif dest and base == "ldr" and _LDR_LIT_RE.match(ops_l):
            a = _LIT_ADDR_RE.search(ops)
            word = literals.get(int(a.group(1), 16)) if a else None
            if word is not None:
                regs[dest.group(1)] = word - (1 << 32) if word & 0x80000000 else word
                written = dest.group(1)
        elif dest and base in ("mov", "movw", "movs") and (m := re.match(r"\w+,\s*#(0x[0-9a-fA-F]+|\d+)$", ops_l)):
            regs[dest.group(1)] = int(m.group(1), 0)
            written = dest.group(1)
        elif dest and base in ("lsls", "lsl") and (m := re.match(r"\w+,\s*(?:(r\d+|ip),\s*)?#(\d+)$", ops_l)):
            src = m.group(1) or dest.group(1)
            if src in regs:
                regs[dest.group(1)] = regs[src] << int(m.group(2))
                written = dest.group(1)
        elif dest and base in ("negs", "rsbs", "rsb") and (m := re.match(r"\w+,\s*(r\d+|ip)(?:,\s*#0)?$", ops_l)):
            if m.group(1) in regs:
                regs[dest.group(1)] = -regs[m.group(1)]
                written = dest.group(1)
        elif dest and base in ("adds", "add", "subs", "sub") and (
                m := re.match(r"\w+,\s*(?:(r\d+|ip),\s*)?#(0x[0-9a-fA-F]+|\d+)$", ops_l)):
            src = m.group(1) or dest.group(1)
            if src in regs:
                k = int(m.group(2), 0)
                regs[dest.group(1)] = regs[src] + (k if base.startswith("add") else -k)
                written = dest.group(1)
        if dest and written != dest.group(1) and not base.startswith(_NO_DEST) and not _BRANCH_RE.match(mn):
            regs.pop(dest.group(1), None)  # bilinmeyen değer yazıldı: eski sabit artık geçerli değil
    return total, unknown


def stack_analysis(elf: Path | str, entries: Iterable[str], objdump: str | None = None,
                   include: Callable[[str], bool] | None = None) -> dict[str, StackInfo]:
    """Bağlanmış imajın disassembly'sinden en kötü durum yığın kullanımı (kütüphaneler dahil).

    Her fonksiyonun çerçevesi = push / vpush / sub sp toplamı (yol duyarsız üst sınır); çağrı grafiği bl ve kuyruk
    dallanmalarından (b.w <başka_fonksiyon>) çıkarılır. Dolaylı çağrı / özyineleme `unknown` olarak bildirilir.
    include verilirse yalnız include(ad) doğru olan fonksiyonlar sayılır (diğerleri 0 baytlık yaprak)."""
    if objdump is None:
        gcc, clang = find_arm_gcc(), find_clang()
        objdump = (gcc.sibling("arm-none-eabi-objdump") if gcc else None) or \
                  (clang.sibling("llvm-objdump") if clang else None)
    if objdump is None:
        return {}
    r = _run([objdump, "-d", "--no-show-raw-insn", str(elf)], timeout=120)
    if r.returncode != 0:
        return {}
    funcs: dict[str, list[tuple[str, str]]] = {}
    literals: dict[int, int] = {}
    current = None
    for line in r.stdout.splitlines():
        w = _WORD_RE.match(line)
        if w:
            literals[int(w.group(1), 16)] = int(w.group(2), 16)
            continue
        m = _FUNC_RE.match(line)
        if m:
            current = m.group(2)
            funcs.setdefault(current, [])
            continue
        if current is None:
            continue
        m = _INS_RE.match(line)
        if m:
            funcs[current].append((m.group(2), m.group(3)))
    frames: dict[str, int] = {}
    calls: dict[str, set[str]] = {}
    unknown: set[str] = set()
    for name, ins in funcs.items():
        frames[name], unk = _frame(ins, literals)
        if unk:
            unknown.add(name)
        callees: set[str] = set()
        for mn, ops in ins:
            t = _TARGET_RE.search(ops)
            if mn in ("bl", "blx"):
                if t and t.group(1) == name and t.group(2):
                    continue                   # Thumb-1'de uzak dallanma `bl <kendi+ofset>`: çağrı değil
                if t:
                    callees.add(t.group(1))
                else:
                    unknown.add(name)          # dolaylı çağrı (blx rN)
            elif _BRANCH_RE.match(mn) and t and not t.group(2) and t.group(1) != name:
                callees.add(t.group(1))        # kuyruk çağrısı
            elif mn == "bx" and not ops.strip().startswith("lr"):
                unknown.add(name)
        calls[name] = callees

    memo: dict[str, tuple[int, tuple[str, ...]]] = {}
    visiting: set[str] = set()

    def worst(f: str) -> tuple[int, tuple[str, ...]]:
        if f in memo:
            return memo[f]
        if f in visiting:                       # özyineleme: sınır yok
            unknown.add(f)
            return 0, (f,)
        visiting.add(f)
        best = (0, ())
        for c in calls.get(f, ()):
            if c in funcs and (include is None or include(c)):
                best = max(best, worst(c))
        visiting.discard(f)
        memo[f] = (frames.get(f, 0) + best[0], (f, *best[1]))
        return memo[f]

    out = {}
    for e in entries:
        if e in funcs:
            total, path = worst(e)
            out[e] = StackInfo(total, path, tuple(sorted(u for u in unknown if u in path)), frames.get(e, 0))
    return out
