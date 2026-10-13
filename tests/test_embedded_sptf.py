"""Gömülü SPTF-UDP C kütüphanesi (embedded/sptf_udp): birim testleri, arayüzle çapraz doğrulama, STM32 derlemesi.

- C birim testleri (test/sptf_test.c) lwIP saplamasıyla, sıkı uyarılar ve `-Werror` ile; çıplak metal (NO_SYS 1),
  FreeRTOS çekirdek kilitli (NO_SYS 0), jumbo datagram / tek akış ve kilitsiz RTOS yapılandırmalarında çalışır.
- `test/sptf_dump` her örnek tipi × yerleşim × CRC için kütüphanenin gerçek API'siyle datagram üretir. Python referans
  kodlayıcısı (`spt.protocol.builder`) aynı senaryodan bayt bayt aynı datagramları üretmelidir (bölme, seq,
  first_sample, zaman damgası, DESCRIPTOR tekrarı, MARKER). Datagramlar arayüzün gerçek alıcısına (`StreamStore`)
  verilir; bütün örnekler kayıpsız ve hatasız çözülmelidir.
- STM32: kütüphane ve örnekler arm-none-eabi-gcc ile, gerçek STM32CubeMX lwIP / HAL / CMSIS-RTOS2 başlıklarına karşı
  `-Werror` ile derlenir (araç zinciri ve başlıklar bulunamazsa atlanır).
"""
from __future__ import annotations

import glob
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from spt.codegen import toolchain
from spt.core.stream_store import StreamStore
from spt.protocol.builder import build_data, build_descriptor, build_marker
from spt.protocol.format import SAMPLE_TYPES, ChannelInfo, Descriptor, Layout

ROOT = Path(__file__).resolve().parents[1] / "embedded" / "sptf_udp"
LIBRARY = ["Src/sptf_crc32.c", "Src/sptf_frame.c", "Src/sptf_udp.c"]
HOST_INCLUDES = ["-IInc", "-Itest/stub"]
STRICT = ["-std=c99", "-Wall", "-Wextra", "-Wpedantic", "-Werror", "-Wconversion", "-Wsign-conversion", "-Wshadow",
          "-Wcast-qual", "-Wcast-align", "-Wstrict-prototypes", "-Wmissing-prototypes", "-Wundef",
          "-Wdouble-promotion"]

STM32_PROJECT = Path(__file__).resolve().parents[1] / "stm32f4_examples" / "000_spt_udp_integration"
CORTEX_M4 = ["-mcpu=cortex-m4", "-mthumb", "-mfpu=fpv4-sp-d16", "-mfloat-abi=hard", "-DSTM32F429xx",
             "-DUSE_HAL_DRIVER"]

DESCRIPTOR_INTERVAL_MS = 500
MARKER_TEXT = "event \u03bcs"
MARKER_CODE = 0xC0DE
MARKER_TIME_NS = 777


def _compile(tool, out: Path, sources: list[str], extra: list[str] = (), includes: list[str] = HOST_INCLUDES,
             flags: list[str] = STRICT) -> Path:
    cmd = [tool.path, *flags, "-O2", *includes, *extra, *sources, "-o", str(out)]
    r = toolchain._run(cmd, cwd=ROOT, path_prefix=[tool.bin_dir])
    assert r.returncode == 0, f"derleme hatası ({' '.join(extra)}):\n{r.stdout}\n{r.stderr}"
    return out


def _run(exe: Path, *args, bin_dir=None, timeout=60) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    if bin_dir:
        env["PATH"] = str(bin_dir) + os.pathsep + env.get("PATH", "")
    return subprocess.run([str(exe), *map(str, args)], capture_output=True, timeout=timeout, env=env)


@pytest.fixture(scope="module")
def cc():
    tool = toolchain.find_host_cc()
    if tool is None:
        pytest.skip("ana makine C derleyicisi yok")
    return tool


@pytest.fixture(scope="module")
def dump(cc, tmp_path_factory):
    d = tmp_path_factory.mktemp("sptf")
    # Senaryolarda jumbo datagram da (8972 B) olduğu için akış tamponu o boyuta göre derlenir.
    return _compile(cc, d / "sptf_dump.exe", LIBRARY + ["test/stub/lwip_stub.c", "test/sptf_dump.c"],
                    ["-DSPTF_CFG_MAX_DATAGRAM_SIZE=8972U"])


# ------------------------------------------------------------------------------------------- C birim testleri
@pytest.mark.parametrize("variant", [
    [],
    ["-DNO_SYS=0"],
    ["-O0", "-DSPTF_CFG_MAX_STREAMS=1U", "-DSPTF_CFG_MAX_DATAGRAM_SIZE=8972U"],
    ["-Os", "-DNO_SYS=0", "-DSPTF_CFG_LWIP_CORE_LOCKING=0U"],
], ids=["ciplak-metal", "freertos-kilitli", "jumbo-tek-akis", "rtos-kilitsiz"])
def test_c_unit_tests(cc, tmp_path, variant):
    exe = _compile(cc, tmp_path / "t.exe", LIBRARY + ["test/stub/lwip_stub.c", "test/sptf_test.c"], variant)
    r = _run(exe, bin_dir=cc.bin_dir)
    out = r.stdout.decode("utf-8", "replace")
    assert r.returncode == 0, out
    assert out.startswith("PASSED")


def test_clang_also_builds_warning_free(tmp_path):
    clang = Path("C:/mingw64/bin/clang.exe")
    if not clang.exists():
        pytest.skip("clang yok")
    tool = toolchain.Tool("host-cc", str(clang), "clang", "x86_64-w64-mingw32")
    exe = _compile(tool, tmp_path / "t.exe", LIBRARY + ["test/stub/lwip_stub.c", "test/sptf_test.c"])
    assert _run(exe, bin_dir=clang.parent).returncode == 0


# ------------------------------------------------------------------------- Python referansıyla bayt bayt aynı
def _datagrams(path: Path) -> list[bytes]:
    raw = path.read_bytes()
    out, off = [], 0
    while off < len(raw):
        n = int.from_bytes(raw[off:off + 4], "little")
        out.append(raw[off + 4:off + 4 + n])
        off += 4 + n
    return out


def _expected(code, layout, crc, ch, n, blocks, maxdg, fs, ts0, seed) -> tuple[Descriptor, list[bytes], int]:
    """C göndericisinin kurallarının bağımsız Python ifadesi: bölme, seq, DESCRIPTOR periyodu (örnek sayısı ve süre),
    zaman damgası ve MARKER."""
    st = SAMPLE_TYPES[code]
    desc = Descriptor(7, 1, fs, st, Layout(layout), "dump stream",
                      tuple(ChannelInfo(f"c{i}", "V" if i % 2 == 0 else "\u00b0C", 1.0 + 0.5 * i, -float(i))
                            for i in range(ch)))
    per = (maxdg - 48 - (4 if crc else 0)) // (st.size * ch)
    period_f = (fs * DESCRIPTOR_INTERVAL_MS) / 1000.0
    period = int(period_f) if period_f >= 1.0 else 1
    interval_ns = DESCRIPTOR_INTERVAL_MS * 1_000_000
    ns_per = 1.0e9 / fs
    frames: list[bytes] = []
    state = {"seq": 0, "next": 0, "streaming": False, "due": 0, "last_ns": 0}

    def maybe_descriptor(ts: int) -> None:
        due = (not state["streaming"]) or state["next"] >= state["due"]
        if not due and state["last_ns"] != 0 and ts > state["last_ns"]:
            due = ts - state["last_ns"] >= interval_ns
        if due:
            frames.append(build_descriptor(desc, state["seq"], ts, crc))
            state.update(seq=state["seq"] + 1, streaming=True, due=state["next"] + period, last_ns=ts)

    nbytes = n * st.size * ch
    for b in range(blocks):
        raw = ((np.arange(nbytes, dtype=np.uint64) * 131 + seed + 17 * b) & 0xFF).astype(np.uint8).tobytes()
        arr = np.frombuffer(raw, st.dtype)
        arr = arr.reshape(n, ch) if layout == Layout.INTERLEAVED else arr.reshape(ch, n)
        base = ts0 + b * 1000 if ts0 else 0
        done = 0
        while done < n:
            k = min(n - done, per)
            ts = base + int(done * ns_per + 0.5) if base else 0
            maybe_descriptor(ts)
            part = arr[done:done + k] if layout == Layout.INTERLEAVED else arr[:, done:done + k].T
            frames.append(build_data(desc, state["seq"], ts, state["next"], part, crc))
            state["seq"] += 1
            state["next"] += k
            done += k
        if b == 0:
            maybe_descriptor(MARKER_TIME_NS)
            frames.append(build_marker(7, 1, state["seq"], MARKER_TIME_NS, state["next"], MARKER_CODE, MARKER_TEXT,
                                       crc))
            state["seq"] += 1
    return desc, frames, per


def _scenarios():
    out = []
    for i, code in enumerate(sorted(SAMPLE_TYPES)):
        for layout in (0, 1):
            for crc in (False, True):
                j = i + layout + 2 * crc
                ch = (1, 3, 5)[j % 3]
                maxdg = (1472, 600, 8972)[(j // 3) % 3]
                fs = (1000.0, 48000.0, 1.0e6)[i % 3]
                out.append(pytest.param(code, layout, crc, ch, maxdg, fs, 123456789 if j % 2 else 0,
                                        id=f"{SAMPLE_TYPES[code].name}-{'planar' if layout else 'interleaved'}"
                                           f"-{'crc' if crc else 'nocrc'}"))
    return out


@pytest.mark.parametrize("code,layout,crc,ch,maxdg,fs,ts0", _scenarios())
def test_c_frames_match_python_builder_and_receiver(cc, dump, tmp_path, code, layout, crc, ch, maxdg, fs, ts0):
    st = SAMPLE_TYPES[code]
    per = (maxdg - 48 - (4 if crc else 0)) // (st.size * ch)
    n = 2 * per + 13                     # her blok 3 çerçeveye bölünür (son çerçeve kısmi)
    blocks, seed = 3, 11 + code
    out = tmp_path / "dg.bin"
    r = _run(dump, out, code, layout, int(crc), ch, n, blocks, maxdg, repr(fs), ts0, seed, bin_dir=cc.bin_dir)
    assert r.returncode == 0, r.stdout + r.stderr
    got = _datagrams(out)
    desc, expected, per_py = _expected(code, layout, crc, ch, n, blocks, maxdg, fs, ts0, seed)
    assert f"ok {len(expected)} {per_py}" in r.stdout.decode()
    assert len(got) == len(expected)
    for i, (g, e) in enumerate(zip(got, expected)):
        assert g == e, f"{i}. datagram farklı (tip {g[5]}, {len(g)} / {len(e)} bayt)"
        assert len(g) <= maxdg

    events = []
    store = StreamStore(events.append, session_start_ns=0, ring_seconds=1.0, max_ring_bytes=1 << 20)
    try:
        for i, g in enumerate(got):
            store.handle_datagram(memoryview(g), len(g), "10.0.0.9", i)
        assert store.frame_errors == 0 and store.no_descriptor == 0
        (s,) = store.stats(None)
        assert (s.samples, s.lost_samples, s.duplicates, s.restarts) == (blocks * n, 0, 0, 0)
        markers = [e for e in events if e[0] == "marker"]
        assert [(m[2], m[3], m[4], m[5]) for m in markers] == [(n, MARKER_CODE, MARKER_TEXT, n)]
        info = next(e[1] for e in events if e[0] == "stream")
        assert info.descriptor == desc
    finally:
        store.close()


# ------------------------------------------------------------------------------------------ STM32 derlemesi
def _stm32_includes(hal_inc: Path | None, conf_dir: Path | None) -> list[str]:
    p = STM32_PROJECT
    dirs = [p / "LWIP" / "Target", p / "Middlewares/Third_Party/LwIP/src/include", p / "Middlewares/Third_Party/LwIP/system",
            p / "Core/Inc", p / "Drivers/STM32F4xx_HAL_Driver/Inc", p / "Middlewares/Third_Party/FreeRTOS/Source/include",
            p / "Middlewares/Third_Party/FreeRTOS/Source/CMSIS_RTOS_V2",
            p / "Middlewares/Third_Party/FreeRTOS/Source/portable/GCC/ARM_CM4F", p / "Drivers/CMSIS/Include",
            p / "Drivers/CMSIS/Device/ST/STM32F4xx/Include"]
    if hal_inc is not None:
        dirs.insert(0, hal_inc)
    if conf_dir is not None:
        dirs.insert(0, conf_dir)
    out = []
    for d in dirs:
        out += ["-isystem", str(d)]
    return out


@pytest.fixture(scope="module")
def arm():
    tool = toolchain.find_arm_gcc()
    if tool is None:
        pytest.skip("arm-none-eabi-gcc yok")
    if not (STM32_PROJECT / "Middlewares/Third_Party/LwIP/src/include/lwip/udp.h").exists():
        pytest.skip("STM32CubeMX lwIP projesi (stm32f4_examples/000_spt_udp_integration) yok")
    return tool


@pytest.mark.parametrize("locking", ["1U", "0U"], ids=["cekirdek-kilitli", "kilitsiz"])
def test_library_builds_for_stm32(arm, tmp_path, locking):
    for src in LIBRARY:
        _compile(arm, tmp_path / (Path(src).stem + ".o"), [src],
                 ["-c", *CORTEX_M4, f"-DSPTF_CFG_LWIP_CORE_LOCKING={locking}"],
                 includes=["-IInc", *_stm32_includes(None, None)])


def test_examples_build_for_stm32(arm, tmp_path):
    """Örnekler HAL ADC sürücüsünü kullanır; CubeMX projesinde ADC kapalı olduğu için başlıklar STM32Cube deposundan
    alınır ve ADC modülü açık bir hal_conf kopyası kullanılır (proje değiştirilmez)."""
    found = sorted(glob.glob(str(Path.home() / "STM32Cube/Repository/STM32Cube_FW_F4_V*/Drivers/STM32F4xx_HAL_Driver/Inc")))
    if not found:
        pytest.skip("STM32Cube F4 HAL deposu yok")
    conf = (STM32_PROJECT / "Core/Inc/stm32f4xx_hal_conf.h").read_text(encoding="utf-8", errors="replace")
    (tmp_path / "stm32f4xx_hal_conf.h").write_text(
        conf.replace("/* #define HAL_ADC_MODULE_ENABLED */", "#define HAL_ADC_MODULE_ENABLED"), encoding="utf-8")
    for example in ("stm32_baremetal_adc", "stm32_freertos_adc"):
        _compile(arm, tmp_path / f"{example}.o", [f"examples/{example}/sptf_app.c"], ["-c", *CORTEX_M4],
                 includes=["-IInc", *_stm32_includes(Path(found[-1]), tmp_path)])
