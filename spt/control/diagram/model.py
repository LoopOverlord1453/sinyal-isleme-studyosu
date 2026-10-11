"""Blok diyagram modeli: bloklar, bağlantılar, sinyal adları, alt sistemler, doğrulama ve `.sptd` (JSON) dosyası.

Port numaraları Python API'sinde **0 tabanlıdır** (`connect(a, b, src_port=0, dst_port=1)`); kullanıcıya gösterilen
iletiler 1 tabanlı ("2. giriş") yazar. Sinyal, bir bloğun bir çıkış portudur (dallar aynı sinyali paylaşır); ad ve
kayıt bayrağı sinyale aittir.

Dosya biçimi (`.sptd`, UTF-8 JSON, katı: sonsuzlar "inf" / "-inf" metni):
    {"format": "spt-diagram", "version": 1, "name": …,
     "blocks": [{"name", "type", "params": {…}, "meta": {…}, ["system": {…iç model…}]}],
     "connections": [{"src": [blok, port], "dst": [blok, port], "meta": {…}}],
     "signals": [{"block", "port", "name", "log"}], "settings": {…}, "meta": {…}}
`meta` alanları arayüzün serbest kullanımı içindir (konum, boyut, yön, kıvrım noktaları …) ve aynen korunur.
"""
from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from spt.control.diagram.block import Block, create_block, encode_value
from spt.control.errors import ControlError

FORMAT = "spt-diagram"
VERSION = 1
FILE_SUFFIX = ".sptd"

DEFAULT_SETTINGS: dict[str, Any] = {
    "solver": "ode4",        # ode1 | ode4 | ode45
    "stop_time": 10.0,
    "start_time": 0.0,
    "step": None,            # sabit adım (None: otomatik)
    "rtol": 1e-3,
    "atol": "auto",
    "max_step": None,
    "zero_crossing": True,
}


@dataclass
class Connection:
    """src bloğunun src_port çıkışından dst bloğunun dst_port girişine (0 tabanlı) bağlantı."""
    src: str
    src_port: int
    dst: str
    dst_port: int
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Issue:
    level: str                 # "error" | "warning"
    message: str
    block: str | None = None
    port: int | None = None


@dataclass
class ValidationReport:
    issues: list[Issue] = field(default_factory=list)

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.level == "error"]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.level == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def __str__(self) -> str:
        if not self.issues:
            return "Model geçerli."
        return "\n".join(("Hata: " if i.level == "error" else "Uyarı: ") + i.message for i in self.issues)


_NAME_RE = re.compile(r"[^/\n\r\t]+")


class Model:
    """Blok diyagram modeli.

    Örnek:
        m = Model("hız denetimi")
        r = m.add("Step", "Referans", step_time=0.0)
        s = m.add("Sum", signs="+-")
        c = m.add("PIDController", "PID", P=2.0, I=1.0)
        g = m.add("TransferFcn", "Bitki", num=[1], den=[1, 2, 1])
        m.connect(r, s); m.connect(s, c, name="e"); m.connect(c, g, name="u")
        m.connect(g, s, dst_port=1); m.log_signal(g, name="y")
    """

    def __init__(self, name: str = "model"):
        self.name = str(name)
        self.blocks: dict[str, Block] = {}
        self.connections: list[Connection] = []
        self.signals: dict[tuple[str, int], dict[str, Any]] = {}
        self.settings: dict[str, Any] = dict(DEFAULT_SETTINGS)
        self.meta: dict[str, Any] = {}

    # ------------------------------------------------------------------ bloklar
    def _unique(self, base: str) -> str:
        if base not in self.blocks:
            return base
        m = re.match(r"^(.*?)(\d+)$", base)
        stem = m.group(1) if m else base
        k = int(m.group(2)) + 1 if m else 1
        while f"{stem}{k}" in self.blocks:
            k += 1
        return f"{stem}{k}"

    def add(self, block: Block | str, name: str | None = None, **params: Any) -> Block:
        """Blok ekler: tür adı ("Gain") ve parametrelerle ya da hazır `Block` nesnesiyle. Ad verilmezse ya da
        çakışıyorsa benzersiz ad üretilir (Gain, Gain1, …); verilen ad çakışırsa hata."""
        if isinstance(block, str):
            block = create_block(block, name, **params)
            if name is None:
                block.name = self._unique(block.name)
        elif params:
            raise ControlError("Hazır blok nesnesiyle birlikte parametre verilemez; set_params kullanın.")
        if name is not None:
            block.name = str(name)
        if not block.name or not _NAME_RE.fullmatch(block.name):
            raise ControlError(f"Geçersiz blok adı '{block.name}' (boş olamaz, '/' ve satır sonu içeremez).")
        if block.name in self.blocks:
            if name is not None:
                raise ControlError(f"'{block.name}' adlı blok zaten var.")
            block.name = self._unique(block.name)
        self.blocks[block.name] = block
        return block

    def block(self, ref: Block | str) -> Block:
        """Ad (ya da "Alt/İç" yolu) veya blok nesnesinden blok."""
        if isinstance(ref, Block):
            if self.blocks.get(ref.name) is not ref:
                raise ControlError(f"'{ref.name}' bloğu bu modelde değil.")
            return ref
        if ref in self.blocks:
            return self.blocks[ref]
        if "/" in ref:
            head, rest = ref.split("/", 1)
            sub = self.blocks.get(head)
            if sub is not None and hasattr(sub, "system"):
                return sub.system.block(rest)
        raise ControlError(f"'{ref}' adlı blok bulunamadı.")

    def _name(self, ref: Block | str) -> str:
        return self.block(ref).name

    def remove(self, ref: Block | str) -> None:
        """Bloğu, bağlantılarını ve sinyal kayıtlarını siler."""
        name = self._name(ref)
        del self.blocks[name]
        self.connections = [c for c in self.connections if c.src != name and c.dst != name]
        self.signals = {k: v for k, v in self.signals.items() if k[0] != name}

    def rename(self, ref: Block | str, new_name: str) -> None:
        old = self._name(ref)
        if new_name == old:
            return
        if not new_name or not _NAME_RE.fullmatch(new_name):
            raise ControlError(f"Geçersiz blok adı '{new_name}'.")
        if new_name in self.blocks:
            raise ControlError(f"'{new_name}' adlı blok zaten var.")
        b = self.blocks.pop(old)
        b.name = new_name
        self.blocks = {**self.blocks, new_name: b}
        for c in self.connections:
            if c.src == old:
                c.src = new_name
            if c.dst == old:
                c.dst = new_name
        self.signals = {((new_name if k[0] == old else k[0]), k[1]): v for k, v in self.signals.items()}

    # ------------------------------------------------------------------ bağlantılar
    def connect(self, src: Block | str, dst: Block | str, src_port: int = 0, dst_port: int = 0, *,
                name: str | None = None, log: bool | None = None, meta: dict | None = None) -> Connection:
        """src bloğunun src_port çıkışını dst bloğunun dst_port girişine bağlar (0 tabanlı). `name` / `log`
        sinyale ad / kayıt bayrağı verir."""
        s, d = self.block(src), self.block(dst)
        if s.name not in self.blocks or d.name not in self.blocks:
            raise ControlError("Alt sistem içindeki bloklar alt sistemin kendi modelinde bağlanır.")
        no, ni = s.n_outputs(), d.n_inputs()
        if not 0 <= src_port < no:
            raise ControlError(f"'{s.name}' bloğunun {src_port + 1}. çıkışı yok (çıkış sayısı {no}).")
        if not 0 <= dst_port < ni:
            raise ControlError(f"'{d.name}' bloğunun {dst_port + 1}. girişi yok (giriş sayısı {ni}).")
        for c in self.connections:
            if c.dst == d.name and c.dst_port == dst_port:
                raise ControlError(f"'{d.name}' bloğunun {dst_port + 1}. girişi zaten '{c.src}' bloğuna bağlı.")
        conn = Connection(s.name, int(src_port), d.name, int(dst_port), dict(meta or {}))
        self.connections.append(conn)
        if name is not None or log is not None:
            self.name_signal(s, src_port, name, log=log)
        return conn

    def disconnect(self, dst: Block | str, dst_port: int = 0) -> None:
        name = self._name(dst)
        before = len(self.connections)
        self.connections = [c for c in self.connections if not (c.dst == name and c.dst_port == dst_port)]
        if len(self.connections) == before:
            raise ControlError(f"'{name}' bloğunun {dst_port + 1}. girişinde bağlantı yok.")

    def source_of(self, dst: Block | str, dst_port: int = 0) -> tuple[str, int] | None:
        name = self._name(dst)
        for c in self.connections:
            if c.dst == name and c.dst_port == dst_port:
                return c.src, c.src_port
        return None

    # ------------------------------------------------------------------ sinyaller
    def name_signal(self, block: Block | str, port: int = 0, name: str | None = None, *,
                    log: bool | None = None) -> None:
        """Sinyale (blok çıkış portu) ad ve / veya kayıt bayrağı verir."""
        b = self.block(block)
        if not 0 <= port < b.n_outputs():
            raise ControlError(f"'{b.name}' bloğunun {port + 1}. çıkışı yok.")
        key = (b.name, int(port))
        entry = self.signals.setdefault(key, {"name": None, "log": False})
        if name is not None:
            name = str(name).strip()
            for k, v in self.signals.items():
                if k != key and v.get("name") == name:
                    raise ControlError(f"'{name}' sinyal adı zaten '{k[0]}' bloğunun çıkışında kullanılıyor.")
            entry["name"] = name or None
        if log is not None:
            entry["log"] = bool(log)

    def log_signal(self, block: Block | str, port: int = 0, name: str | None = None) -> None:
        """Sinyali benzetim sonucuna kaydettirir (isteğe bağlı adla)."""
        self.name_signal(block, port, name, log=True)

    def signal_key(self, block: str, port: int) -> str:
        """Sinyalin sonuçtaki anahtarı: adı varsa ad, yoksa "Blok" (tek çıkış) ya da "Blok:2" (1 tabanlı)."""
        entry = self.signals.get((block, port))
        if entry and entry.get("name"):
            return entry["name"]
        b = self.blocks.get(block)
        if b is not None and b.n_outputs() == 1:
            return block
        return f"{block}:{port + 1}"

    # ------------------------------------------------------------------ alt sistem
    def add_subsystem(self, name: str, system: "Model | None" = None) -> Block:
        """Alt sistem bloğu ekler (içi `block.system` modelidir; Giriş / Çıkış portlarıyla bağlanır)."""
        from spt.control.diagram.blocks.ports import Subsystem
        return self.add(Subsystem(name, system=system if system is not None else Model(name)), name)

    # ------------------------------------------------------------------ kopya / doğrulama
    def copy(self) -> "Model":
        return copy.deepcopy(self)

    def validate(self, *, compile: bool = True) -> ValidationReport:
        """Yapısal denetim (bağlanmamış / çift bağlı giriş, geçersiz port, bağlanmamış çıkış uyarısı) ve derleme
        denetimi (genişlik uyumu, örnekleme zamanı çatışması, cebirsel döngü)."""
        rep = ValidationReport()
        self._structural(rep, "")
        if compile and rep.ok:
            from spt.control.diagram.engine import compile_model
            try:
                compile_model(self)
            except ControlError as exc:
                rep.issues.append(Issue("error", str(exc)))
        return rep

    def check(self) -> None:
        """Doğrulama hatası varsa ilkini `ControlError` olarak yükseltir."""
        rep = self.validate()
        if not rep.ok:
            raise ControlError(rep.errors[0].message)

    def _structural(self, rep: ValidationReport, prefix: str) -> None:
        conns_in: dict[tuple[str, int], list[Connection]] = {}
        used_out: set[tuple[str, int]] = set()
        for c in self.connections:
            s, d = self.blocks.get(c.src), self.blocks.get(c.dst)
            if s is None or d is None:
                rep.issues.append(Issue("error", f"Bağlantı var olmayan bloğa işaret ediyor ({c.src} → {c.dst})."))
                continue
            if not 0 <= c.src_port < s.n_outputs():
                rep.issues.append(Issue("error", f"'{prefix}{s.name}' bloğunun {c.src_port + 1}. çıkışı artık yok "
                                                 f"(bağlantı {prefix}{c.src} → {prefix}{c.dst}).", prefix + s.name))
                continue
            if not 0 <= c.dst_port < d.n_inputs():
                rep.issues.append(Issue("error", f"'{prefix}{d.name}' bloğunun {c.dst_port + 1}. girişi artık yok "
                                                 f"(bağlantı {prefix}{c.src} → {prefix}{c.dst}).", prefix + d.name))
                continue
            conns_in.setdefault((c.dst, c.dst_port), []).append(c)
            used_out.add((c.src, c.src_port))
        for name, b in self.blocks.items():
            for p in range(b.n_inputs()):
                lst = conns_in.get((name, p), [])
                if not lst:
                    rep.issues.append(Issue("error", f"'{prefix}{name}' bloğunun {p + 1}. girişi bağlanmamış.",
                                            prefix + name, p))
                elif len(lst) > 1:
                    rep.issues.append(Issue("error", f"'{prefix}{name}' bloğunun {p + 1}. girişine birden çok "
                                                     "sinyal bağlı.", prefix + name, p))
            for p in range(b.n_outputs()):
                if (name, p) not in used_out:
                    rep.issues.append(Issue("warning", f"'{prefix}{name}' bloğunun {p + 1}. çıkışı bağlanmamış.",
                                            prefix + name, p))
            if hasattr(b, "system"):
                try:
                    b.inports()
                    b.outports()
                except ControlError as exc:
                    rep.issues.append(Issue("error", str(exc), prefix + name))
                    continue
                b.system._structural(rep, f"{prefix}{name}/")
        names = [v["name"] for v in self.signals.values() if v.get("name")]
        dup = {n for n in names if names.count(n) > 1}
        for n in sorted(dup):
            rep.issues.append(Issue("error", f"'{n}' sinyal adı birden çok sinyalde kullanılmış."))

    # ------------------------------------------------------------------ dosya
    def to_dict(self) -> dict[str, Any]:
        blocks = []
        for name, b in self.blocks.items():
            d: dict[str, Any] = {"name": name, "type": b.type_name, "params": b.encoded_params(),
                                 "meta": encode_value(copy.deepcopy(b.meta))}
            if hasattr(b, "system"):
                d["system"] = b.system.to_dict()
            blocks.append(d)
        return {
            "format": FORMAT, "version": VERSION, "name": self.name, "blocks": blocks,
            "connections": [{"src": [c.src, c.src_port], "dst": [c.dst, c.dst_port],
                             "meta": encode_value(copy.deepcopy(c.meta))} for c in self.connections],
            "signals": [{"block": k[0], "port": k[1], "name": v.get("name"), "log": bool(v.get("log"))}
                        for k, v in self.signals.items()],
            "settings": encode_value(dict(self.settings)),
            "meta": encode_value(copy.deepcopy(self.meta)),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Model":
        if not isinstance(data, dict) or data.get("format") != FORMAT:
            raise ControlError("Dosya bir blok diyagram (.sptd) dosyası değil.")
        ver = data.get("version")
        if not isinstance(ver, int) or ver < 1:
            raise ControlError("Blok diyagram dosyasının sürüm alanı geçersiz.")
        if ver > VERSION:
            raise ControlError(f"Dosya daha yeni bir sürümle kaydedilmiş (sürüm {ver}; desteklenen {VERSION}).")
        m = cls(data.get("name", "model"))
        for bd in data.get("blocks", []):
            typ = bd.get("type")
            try:
                if typ == "Subsystem":
                    from spt.control.diagram.blocks.ports import Subsystem
                    blk: Block = Subsystem(bd["name"], system=cls.from_dict(bd["system"]))
                else:
                    blk = create_block(typ, bd["name"], **bd.get("params", {}))
            except ControlError as exc:
                raise ControlError(f"'{bd.get('name')}' bloğu ({typ}) yüklenemedi: {exc}") from None
            blk.meta = dict(bd.get("meta", {}))
            m.add(blk, bd["name"])
        for cd in data.get("connections", []):
            (s, sp), (d, dp) = cd["src"], cd["dst"]
            m.connections.append(Connection(str(s), int(sp), str(d), int(dp), dict(cd.get("meta", {}))))
        for sd in data.get("signals", []):
            m.signals[(str(sd["block"]), int(sd["port"]))] = {"name": sd.get("name"), "log": bool(sd.get("log"))}
        st = dict(DEFAULT_SETTINGS)
        for k, v in dict(data.get("settings", {})).items():
            if isinstance(v, str) and v in ("inf", "-inf"):
                v = float(v)
            st[k] = v
        m.settings = st
        m.meta = dict(data.get("meta", {}))
        return m

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        if p.suffix.lower() != FILE_SUFFIX:
            p = p.with_suffix(FILE_SUFFIX)
        text = json.dumps(self.to_dict(), ensure_ascii=False, indent=1, allow_nan=False)
        p.write_text(text, encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: str | Path) -> "Model":
        p = Path(path)
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ControlError(f"Blok diyagram dosyası okunamadı: {exc}") from None
        return cls.from_dict(data)

    # ------------------------------------------------------------------ bilgi
    def iter_blocks(self, prefix: str = "") -> Iterable[tuple[str, Block]]:
        """Tüm bloklar (alt sistemler dahil) tam adlarıyla."""
        for name, b in self.blocks.items():
            yield prefix + name, b
            if hasattr(b, "system"):
                yield from b.system.iter_blocks(f"{prefix}{name}/")

    def __repr__(self) -> str:
        return f"<Model '{self.name}': {len(self.blocks)} blok, {len(self.connections)} bağlantı>"


__all__ = ["Model", "Connection", "Issue", "ValidationReport", "FORMAT", "VERSION", "FILE_SUFFIX",
           "DEFAULT_SETTINGS"]
