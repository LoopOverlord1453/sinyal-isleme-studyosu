"""Blok sözleşmesi: parametreler, portlar, örnekleme zamanı, durumlar ve blok kaydı (registry).

Bir blok (Simulink S-fonksiyonu anlamlarıyla) şunları tanımlar:

- **Parametreler** (`PARAMS`): türlü, doğrulanan, Türkçe etiket / ipucu / birimli, varsayılanlı `Param` listesi.
  Değerler `block.params[ad]` içinde normalize edilmiş biçimde (float, int, bool, str, `np.ndarray`) durur.
- **Portlar**: giriş / çıkış sayısı (`n_inputs`, `n_outputs`; parametrelere bağlı olabilir, ör. Toplam'ın işaretleri)
  ve genişlikleri (vektör sinyaller). Genişlikler derlemede yayılır: `propagate_widths`.
- **Örnekleme zamanı** (`sample_time()` → `SampleTime`): sürekli `(0, 0)`, ayrık `(Ts, ofset)`, miras `(-1, 0)`,
  sabit `(inf, 0)`. Miras bloklar derlemede girişlerinden çözülür; çözülen değer `block.ts`'dedir.
- **Durumlar**: `nx` sürekli (türev `derivatives`), `nxd` ayrık (güncelleme `update`). Durum vektörleri motorda
  tutulur ve fonksiyonlara açıkça verilir (yan etkisiz değerlendirme → doğrulusallaştırma / trim mümkündür).
- **Doğrudan geçiş** (`feedthrough()`): her giriş için, çıkış o girişe aynı anda bağlı mı (yürütme sırası ve cebirsel
  döngü tespiti buna göre).
- **İş belleği** (work): durum sayılmayan iç bellek (taşıma gecikmesi tamponu, hız sınırlayıcının önceki çıkışı,
  rölenin konumu …). `reset()` ile başlatılır; `major()` (çıkıştan önce) ve `commit()` (çıkıştan sonra) yalnız ana
  adımlarda çağrılır.

Yürütme anlamları (Simulink ile aynı)
------------------------------------
Her **ana adımda** (major step) t anında: (1) her blok için sıralı olarak `project` (yalnız durum kırpma),
`major` ve `output`; (2) kayıt; (3) örnekleme anı gelen ayrık bloklar için `update` (yeni ayrık durum) ve
`commit`; sürekli bloklar için her ana adımda `commit`; (4) sürekli durumlar çözücüyle ilerletilir — çözücünün **ara
adımlarında** (minor step) yalnız sürekli (çözülmüş Ts = 0) blokların `output`'u ve durumlu blokların `derivatives`'i
çağrılır; ayrık blokların çıkışı bir sonraki örnekleme anına kadar tutulur (ZOH).

Yeni blok yazmak (eklenti sözleşmesi) için `Block` alt sınıfı tanımlayıp `@register_block` ile kaydedin; ayrıntı
`spt/control/diagram/__init__.py` belgesinde.
"""
from __future__ import annotations

import importlib
import math
from dataclasses import dataclass
from typing import Any, Callable, ClassVar, Sequence

import numpy as np

from spt.control.errors import ControlError
from spt.control.diagram.expr import parse_numeric

# ============================================================================ örnekleme zamanı
CONTINUOUS = 0.0
INHERITED = -1.0
CONSTANT = math.inf


@dataclass(frozen=True)
class SampleTime:
    """Örnekleme zamanı (Simulink [Ts, ofset]): period 0 sürekli, > 0 ayrık, −1 miras, inf sabit."""
    period: float = 0.0
    offset: float = 0.0

    @property
    def is_continuous(self) -> bool:
        return self.period == 0.0

    @property
    def is_discrete(self) -> bool:
        return 0.0 < self.period < math.inf

    @property
    def is_inherited(self) -> bool:
        return self.period < 0.0

    @property
    def is_constant(self) -> bool:
        return self.period == math.inf

    def describe(self) -> str:
        if self.is_continuous:
            return "sürekli"
        if self.is_constant:
            return "sabit"
        if self.is_inherited:
            return "miras"
        return f"ayrık Ts = {self.period:g} s" + (f", ofset {self.offset:g} s" if self.offset else "")

    def hit(self, t: float) -> bool:
        """t bu ayrık hızın örnekleme anı mı? (göreli 1e-9 toleransla)"""
        if not self.is_discrete:
            return False
        n = (t - self.offset) / self.period
        r = round(n)
        return r >= 0 and abs(n - r) <= 1e-9 * max(1.0, abs(n))

    def next_hit(self, t: float) -> float:
        """t'den kesin büyük ilk örnekleme anı."""
        n = (t - self.offset) / self.period
        k = math.floor(n + 1e-9 * max(1.0, abs(n))) + 1
        k = max(k, 0)
        return self.offset + k * self.period


CONTINUOUS_TS = SampleTime(0.0, 0.0)
INHERITED_TS = SampleTime(-1.0, 0.0)
CONSTANT_TS = SampleTime(math.inf, 0.0)


def make_sample_time(value: Any, owner: str = "") -> SampleTime:
    """0 / −1 / inf / Ts / [Ts, ofset] → SampleTime (doğrulamalı)."""
    pre = f"'{owner}' bloğu: " if owner else ""
    if isinstance(value, SampleTime):
        return value
    arr = np.atleast_1d(np.asarray(value, dtype=np.float64)).ravel()
    if arr.size not in (1, 2):
        raise ControlError(f"{pre}örnekleme zamanı tek sayı ya da [Ts, ofset] olmalı.")
    ts = float(arr[0])
    off = float(arr[1]) if arr.size == 2 else 0.0
    if math.isnan(ts) or math.isnan(off):
        raise ControlError(f"{pre}örnekleme zamanı sayı olmalı.")
    if ts == math.inf:
        return CONSTANT_TS
    if ts == 0.0:
        return CONTINUOUS_TS
    if ts == -1.0:
        return INHERITED_TS
    if ts < 0:
        raise ControlError(f"{pre}örnekleme zamanı 0 (sürekli), −1 (miras), inf (sabit) ya da pozitif olmalı "
                           f"(verilen {ts:g}).")
    if not (0.0 <= off < ts) or not math.isfinite(off):
        raise ControlError(f"{pre}örnekleme ofseti 0 ≤ ofset < Ts olmalı (verilen {off:g}).")
    return SampleTime(ts, off)


# ============================================================================ parametreler
PARAM_KINDS = ("float", "int", "bool", "choice", "vector", "cvector", "matrix", "str", "expr", "signs",
               "sample_time")


def _num_encode(v: float) -> Any:
    if math.isnan(v):
        return "nan"
    if math.isinf(v):
        return "inf" if v > 0 else "-inf"
    return v


def encode_value(value: Any) -> Any:
    """Parametre değerini katı JSON'a uygun biçime çevirir (sonsuzlar "inf" / "-inf" metni olur)."""
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _num_encode(float(value))
        return [encode_value(x) for x in value]
    if isinstance(value, (list, tuple)):
        return [encode_value(x) for x in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return _num_encode(float(value))
    return value


def _decode_numeric_nested(value: Any) -> Any:
    if isinstance(value, str):
        low = value.strip().lower()
        if low in ("inf", "+inf"):
            return math.inf
        if low == "-inf":
            return -math.inf
        if low == "nan":
            return math.nan
        return value
    if isinstance(value, (list, tuple)):
        return [_decode_numeric_nested(x) for x in value]
    return value


@dataclass(frozen=True)
class Param:
    """Blok parametresi tanımı.

    kind: "float" | "int" | "bool" | "choice" | "vector" | "matrix" | "str" | "expr" | "signs" | "sample_time".
    - vector: 1-B float dizi (skaler → 1 eleman); matrix: 2-B (1-B değer `orient`'e göre satır / sütun olur).
    - choice: `choices` = {değer: Türkçe etiket}.
    - Sayılar için `min` / `max` (dahil; `strict_min` ile kesin), `allow_inf`, `allow_nan` yok.
    - Metin değerler (arayüzden) MATLAB tarzı çözülür: "[1 2; 3 4]", "2*pi", "inf".
    """
    name: str
    kind: str
    default: Any
    label: str
    hint: str = ""
    unit: str = ""
    choices: dict[str, str] | None = None
    min: float | None = None
    max: float | None = None
    strict_min: bool = False
    allow_inf: bool = False
    orient: str = "row"
    min_len: int = 0
    group: str = ""

    def __post_init__(self) -> None:
        if self.kind not in PARAM_KINDS:
            raise ValueError(f"bilinmeyen parametre türü {self.kind}")

    # ------------------------------------------------------------------ doğrulama
    def _fail(self, owner: str, msg: str) -> None:
        raise ControlError(f"'{owner}' bloğu: '{self.label}' {msg}")

    def _check_range(self, arr: np.ndarray, owner: str) -> None:
        if arr.size == 0:
            return
        if np.isnan(arr).any():
            self._fail(owner, "sayı olmalı (NaN verildi).")
        if not self.allow_inf and not np.isfinite(arr).all():
            self._fail(owner, "sonlu olmalı.")
        if self.min is not None:
            bad = arr <= self.min if self.strict_min else arr < self.min
            if bad.any():
                op = ">" if self.strict_min else "≥"
                self._fail(owner, f"{op} {self.min:g} olmalı (verilen {arr[bad].ravel()[0]:g}).")
        if self.max is not None and (arr > self.max).any():
            self._fail(owner, f"≤ {self.max:g} olmalı (verilen {arr[arr > self.max].ravel()[0]:g}).")

    def _array(self, value: Any, owner: str) -> np.ndarray:
        try:
            if isinstance(value, str):
                arr = parse_numeric(value)
            else:
                arr = np.asarray(_decode_numeric_nested(value), dtype=np.float64)
        except ControlError as exc:
            self._fail(owner, f"okunamadı: {exc}")
        except (TypeError, ValueError):
            self._fail(owner, f"sayısal olmalı (verilen {value!r}).")
        return arr

    def validate(self, value: Any, owner: str) -> Any:
        k = self.kind
        if k == "float":
            arr = self._array(value, owner)
            if arr.size != 1:
                self._fail(owner, f"tek sayı olmalı ({arr.size} eleman verildi).")
            v = float(arr.ravel()[0])
            self._check_range(np.array([v]), owner)
            return v
        if k == "int":
            arr = self._array(value, owner)
            if arr.size != 1 or not np.isfinite(arr).all() or not float(arr.ravel()[0]).is_integer():
                self._fail(owner, f"tam sayı olmalı (verilen {value!r}).")
            v = int(arr.ravel()[0])
            self._check_range(np.array([float(v)]), owner)
            return v
        if k == "bool":
            if isinstance(value, (bool, np.bool_)):
                return bool(value)
            if isinstance(value, (int, float, np.integer, np.floating)) and value in (0, 1):
                return bool(value)
            if isinstance(value, str) and value.strip().lower() in ("on", "true", "1", "evet", "açık"):
                return True
            if isinstance(value, str) and value.strip().lower() in ("off", "false", "0", "hayır", "kapalı"):
                return False
            self._fail(owner, f"açık / kapalı olmalı (verilen {value!r}).")
        if k == "choice":
            assert self.choices is not None
            if value in self.choices:
                return value
            for key, lab in self.choices.items():       # Türkçe etiketle de kabul et
                if value == lab:
                    return key
            self._fail(owner, f"için geçersiz seçim {value!r}; geçerliler: {', '.join(self.choices)}.")
        if k == "vector":
            arr = self._array(value, owner)
            if arr.ndim > 2 or (arr.ndim == 2 and min(arr.shape) > 1):
                self._fail(owner, "vektör olmalı (matris verildi).")
            arr = np.array(arr.ravel(), dtype=np.float64)
            if arr.size < self.min_len:
                self._fail(owner, f"en az {self.min_len} elemanlı olmalı.")
            self._check_range(arr, owner)
            return arr
        if k == "cvector":
            return self._cvector(value, owner)
        if k == "matrix":
            arr = self._array(value, owner)
            if arr.ndim == 0:
                arr = arr.reshape(1, 1)
            elif arr.ndim == 1:
                arr = arr.reshape(1, -1) if self.orient == "row" else arr.reshape(-1, 1)
            elif arr.ndim > 2:
                self._fail(owner, "2 boyutlu matris olmalı.")
            arr = np.array(arr, dtype=np.float64)
            self._check_range(arr, owner)
            return arr
        if k in ("str", "expr"):
            if not isinstance(value, str):
                self._fail(owner, "metin olmalı.")
            return value
        if k == "signs":
            if not isinstance(value, str):
                arr = self._array(value, owner)
                if arr.size != 1 or not float(arr.ravel()[0]).is_integer() or arr.ravel()[0] < 1:
                    self._fail(owner, "'+-' gibi işaret dizisi ya da pozitif tam sayı olmalı.")
                return "+" * int(arr.ravel()[0])
            s = value.replace(" ", "")
            if not s or any(c not in "+-|" for c in s) or not s.replace("|", ""):
                self._fail(owner, f"yalnız '+', '-' ve '|' içermeli (verilen {value!r}).")
            return s
        if k == "sample_time":
            if isinstance(value, SampleTime):
                return make_sample_time([value.period, value.offset], owner)
            arr = self._array(value, owner)
            return make_sample_time(arr, owner)
        raise AssertionError(k)  # pragma: no cover

    def _cvector(self, value: Any, owner: str) -> np.ndarray:
        """Karmaşık vektör: karmaşık sayılar, [gerçel, sanal] çiftleri (JSON) ya da "[-1+2i -1-2i]" metni."""
        try:
            if isinstance(value, str):
                s = value.strip()
                if s.startswith("[") and s.endswith("]"):
                    s = s[1:-1]
                parts = [q for q in s.replace(",", " ").split() if q]
                vals = [complex(q.replace("i", "j")) for q in parts]
                arr = np.array(vals, dtype=np.complex128)
            elif isinstance(value, np.ndarray) and np.iscomplexobj(value):
                arr = np.array(value.ravel(), dtype=np.complex128)
            else:
                seq = list(value) if isinstance(value, (list, tuple, np.ndarray)) else [value]
                vals = []
                for q in seq:
                    if isinstance(q, (list, tuple, np.ndarray)) and len(q) == 2:
                        re_, im_ = _decode_numeric_nested(list(q))
                        vals.append(complex(float(re_), float(im_)))
                    else:
                        vals.append(complex(_decode_numeric_nested(q)))
                arr = np.array(vals, dtype=np.complex128)
        except (TypeError, ValueError):
            self._fail(owner, f"karmaşık sayı listesi olmalı (verilen {value!r}).")
        if not np.isfinite(arr).all():
            self._fail(owner, "sonlu olmalı.")
        return arr

    # ------------------------------------------------------------------ JSON
    def encode(self, value: Any) -> Any:
        if self.kind == "cvector":
            return [[float(z.real), float(z.imag)] for z in np.asarray(value, dtype=np.complex128)]
        if self.kind == "sample_time":
            st: SampleTime = value
            if st.offset:
                return [_num_encode(st.period), st.offset]
            return _num_encode(st.period)
        return encode_value(value)

    def info(self) -> dict[str, Any]:
        """Arayüz için tanım (JSON uyumlu)."""
        d: dict[str, Any] = {"name": self.name, "kind": self.kind, "label": self.label, "hint": self.hint,
                             "unit": self.unit, "default": self.encode(self.validate(self.default, "?")),
                             "group": self.group}
        if self.choices:
            d["choices"] = dict(self.choices)
        if self.min is not None:
            d["min"] = self.min
        if self.max is not None:
            d["max"] = self.max
        return d


def p_ts(default: Any = -1.0, label: str = "Örnekleme zamanı", hint: str = "") -> Param:
    """Ortak örnekleme zamanı parametresi."""
    return Param("Ts", "sample_time", default, label,
                 hint or "0: sürekli, −1: miras (girişten), inf: sabit, Ts > 0: ayrık (s); [Ts, ofset] verilebilir.",
                 unit="s")


# ============================================================================ yürütme bağlamı
@dataclass
class SimContext:
    """Bloklar için paylaşılan benzetim bağlamı (motor her aşamada günceller).

    major: ana adımda mı (False: çözücü ara adımı). zc: sıfır geçiş (kip dondurma) etkin mi (değişken adımlı ode45).
    freeze_events: değişken adımlı çözücüde zaman olaylı kaynaklar (Basamak, darbe, ZOH veri) ara adımda son ana
    adımdaki değerini korur (olay anları zaten ana adımdır; adım sonu ara değerlendirmesi olayın sağ limitini görmez).
    linearizing: doğrusallaştırma / trim değerlendirmesi (iş belleği değiştirilmez, kipler kullanılmaz).
    """
    major: bool = True
    zc: bool = False
    freeze_events: bool = False
    linearizing: bool = False
    t_start: float = 0.0
    solver: str = ""


DEFAULT_CONTEXT = SimContext()


# ============================================================================ blok tabanı
class Block:
    """Tüm blokların tabanı (eklenti sözleşmesi).

    Sınıf öznitelikleri (alt sınıf doldurur):
        type_name     — kalıcı tür adı (dosyada saklanır; Simulink adına yakın İngilizce, ör. "TransferFcn")
        display_name  — Türkçe görünen ad ("Transfer Fonksiyonu")
        category      — kütüphane kategorisi (Türkçe; `CATEGORIES`)
        description   — kısa Türkçe açıklama (ipucu)
        simulink      — Simulink karşılığı (bilgi)
        PARAMS        — `Param` demeti
    Örnek düzeyinde: `name`, `params`, `meta` (arayüz: konum, boyut, yön …), derlemede `ts`, `in_widths`,
    `out_widths`, `nx`, `nxd`, `ctx`.
    """
    type_name: ClassVar[str] = ""
    display_name: ClassVar[str] = ""
    category: ClassVar[str] = ""
    description: ClassVar[str] = ""
    simulink: ClassVar[str] = ""
    PARAMS: ClassVar[tuple[Param, ...]] = ()
    is_virtual: ClassVar[bool] = False       # Inport / Outport / alt sistem (yalnız yapı)

    def __init__(self, name: str | None = None, **params: Any):
        self.name: str = str(name) if name else (self.type_name or type(self).__name__)
        self.meta: dict[str, Any] = {}
        self.params: dict[str, Any] = {}
        self._param_defs: dict[str, Param] = {p.name: p for p in self.PARAMS}
        unknown = set(params) - set(self._param_defs)
        if unknown:
            raise ControlError(f"'{self.name}' bloğu ({self.display_name}): bilinmeyen parametre(ler) "
                               f"{', '.join(sorted(unknown))}; geçerliler: {', '.join(self._param_defs)}.")
        values = {p.name: p.default for p in self.PARAMS}
        values.update(params)
        for p in self.PARAMS:
            self.params[p.name] = p.validate(values[p.name], self.name)
        self.check_params()
        # derleme sonucu
        self.ts: SampleTime = INHERITED_TS
        self.in_widths: list[int] = []
        self.out_widths: list[int] = []
        self.nx: int = 0
        self.nxd: int = 0
        self.ctx: SimContext = DEFAULT_CONTEXT

    # ------------------------------------------------------------------ parametreler
    def set_params(self, **params: Any) -> None:
        """Parametreleri doğrulayarak değiştirir (hatalıysa hiçbiri değişmez)."""
        unknown = set(params) - set(self._param_defs)
        if unknown:
            raise ControlError(f"'{self.name}' bloğu: bilinmeyen parametre(ler) {', '.join(sorted(unknown))}.")
        old = dict(self.params)
        try:
            for k, v in params.items():
                self.params[k] = self._param_defs[k].validate(v, self.name)
            self.check_params()
        except ControlError:
            self.params = old
            raise

    def check_params(self) -> None:
        """Parametreler arası tutarlılık denetimi (alt sınıf; hata için `self.error`)."""

    def error(self, msg: str) -> None:
        raise ControlError(f"'{self.name}' bloğu ({self.display_name}): {msg}")

    def p(self, name: str) -> Any:
        return self.params[name]

    def encoded_params(self) -> dict[str, Any]:
        return {k: self._param_defs[k].encode(v) for k, v in self.params.items()}

    # ------------------------------------------------------------------ yapı
    def n_inputs(self) -> int:
        return 1

    def n_outputs(self) -> int:
        return 1

    def input_labels(self) -> list[str]:
        n = self.n_inputs()
        return ["u"] if n == 1 else [f"u{i + 1}" for i in range(n)]

    def output_labels(self) -> list[str]:
        n = self.n_outputs()
        return ["y"] if n == 1 else [f"y{i + 1}" for i in range(n)]

    def sample_time(self) -> SampleTime:
        """Parametrelerden örnekleme zamanı (miras olabilir). Varsayılan: 'Ts' parametresi varsa o, yoksa miras."""
        st = self.params.get("Ts")
        return st if isinstance(st, SampleTime) else INHERITED_TS

    def allowed_sample_times(self) -> str:
        """"any" | "continuous" | "discrete" — çözülmüş Ts için kısıt (ör. Birim Gecikme yalnız ayrık)."""
        return "any"

    def feedthrough(self) -> Sequence[bool]:
        """Her giriş için doğrudan geçiş bayrağı (varsayılan: hepsi True)."""
        return [True] * self.n_inputs()

    def propagate_widths(self, in_widths: list[int | None], guess: bool = False) -> list[int | None]:
        """Giriş genişliklerinden çıkış genişlikleri (bilinmiyorsa None). `guess=True` iken eksik bilgiyle tahmin
        yapılabilir (geri besleme döngüsünde yayılımı başlatmak için); son denetim `setup`'tadır.
        Varsayılan: tüm çıkışlar ilk girişin genişliğinde."""
        w = in_widths[0] if in_widths else 1
        return [w] * self.n_outputs()

    def setup(self, in_widths: list[int], out_widths: list[int]) -> None:
        """Kesin genişliklerle son denetim ve hazırlık: `nx`, `nxd` burada belirlenir (alt sınıf)."""

    # ------------------------------------------------------------------ durumlar
    def x0(self) -> np.ndarray:
        return np.zeros(self.nx)

    def xd0(self) -> np.ndarray:
        return np.zeros(self.nxd)

    def reset(self) -> None:
        """İş belleğini başlatır (benzetim başında)."""

    # ------------------------------------------------------------------ yürütme
    def output(self, t: float, x: np.ndarray, xd: np.ndarray, u: list[np.ndarray]) -> list[np.ndarray]:
        raise NotImplementedError

    def derivatives(self, t: float, x: np.ndarray, xd: np.ndarray, u: list[np.ndarray]) -> np.ndarray:
        return np.zeros(self.nx)

    def update(self, t: float, x: np.ndarray, xd: np.ndarray, u: list[np.ndarray]) -> np.ndarray:
        return xd

    def major(self, t: float, x: np.ndarray, xd: np.ndarray, u: list[np.ndarray]) -> None:
        """Ana adımda çıkıştan hemen önce (kip güncelleme; ör. röle konumu)."""

    def commit(self, t: float, x: np.ndarray, xd: np.ndarray, u: list[np.ndarray], y: list[np.ndarray]) -> None:
        """Ana adımda çıkışlardan sonra iş belleği güncellemesi (ör. gecikme tamponu)."""

    def project(self, x: np.ndarray) -> np.ndarray | None:
        """Ana adımda sürekli durum kırpma (ör. sınırlı integratör); değişiklik yoksa None."""
        return None

    def next_event(self, t: float) -> float:
        """t'den sonraki zaman olayı (değişken adımlı çözücü bu ana tam basar; ör. Basamak anı)."""
        return math.inf

    def zero_crossings(self, t: float, x: np.ndarray, xd: np.ndarray, u: list[np.ndarray]) -> np.ndarray | None:
        """Sıfır geçiş sinyalleri (değişken adımlı çözücüde kip değişimi anını bulmak için); yoksa None."""
        return None

    # ------------------------------------------------------------------ doğrusallaştırma
    def linearize(self, t: float, x: np.ndarray, xd: np.ndarray, u: list[np.ndarray]) \
            -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
        """Analitik yerel doğrusallaştırma (A, B, C, D): durumlar (sürekli ya da ayrık, bloğun hızında), girişler
        (tüm giriş portları art arda) ve çıkışlar (tüm çıkış portları art arda). None → sayısal merkezi fark."""
        return None

    def trim_output(self, t: float, x: np.ndarray, xd: np.ndarray, u: list[np.ndarray]) -> list[np.ndarray] | None:
        """Çalışma noktası (kararlı hâl) değerlendirmesinde farklı çıkış gerekiyorsa (ör. hız sınırlayıcı: y = u)."""
        return None

    # ------------------------------------------------------------------ bilgi
    def __repr__(self) -> str:
        return f"<{type(self).__name__} '{self.name}'>"

    def clone(self, name: str | None = None) -> "Block":
        b = type(self)(name or self.name, **self.params)
        b.meta = dict(self.meta)
        return b


# ============================================================================ yardımcı alt sınıflar
class SourceBlock(Block):
    """Girişsiz kaynak bloklar."""

    def n_inputs(self) -> int:
        return 0

    def feedthrough(self) -> Sequence[bool]:
        return []

    def propagate_widths(self, in_widths, guess=False):
        return [self.source_width()]

    def source_width(self) -> int:
        return 1


# ============================================================================ kayıt (registry)
CATEGORIES: dict[str, str] = {
    "sources": "Kaynaklar",
    "continuous": "Doğrusal sürekli",
    "discrete": "Ayrık",
    "nonlinear": "Doğrusal olmayan",
    "controllers": "Kontrolcüler",
    "routing": "Sinyal yönlendirme",
    "sinks": "Gözlem",
    "ports": "Portlar ve alt sistemler",
    "plants": "Bitki eklentileri",
}

_REGISTRY: dict[str, type[Block]] = {}
_LAZY: dict[str, dict[str, str]] = {}


def register_block(cls: type[Block] | None = None, *, override: bool = False) -> Any:
    """Blok sınıfını kaydeder (dekoratör olarak da kullanılır). Aynı tür adı ikinci kez kaydedilirse hata
    (`override=True` ile bilinçli değiştirme)."""
    def deco(c: type[Block]) -> type[Block]:
        if not (isinstance(c, type) and issubclass(c, Block)):
            raise TypeError("register_block yalnız Block alt sınıfı kabul eder")
        name = c.type_name
        if not name or not name.replace("_", "").isalnum():
            raise ValueError(f"geçersiz type_name {name!r}")
        if not c.display_name or not c.category:
            raise ValueError(f"{name}: display_name ve category zorunlu")
        if name in _REGISTRY and _REGISTRY[name] is not c and not override:
            raise ValueError(f"'{name}' blok türü zaten kayıtlı")
        _REGISTRY[name] = c
        _LAZY.pop(name, None)
        return c
    return deco(cls) if cls is not None else deco


def register_lazy(type_name: str, target: str, *, display_name: str, category: str = "plants",
                  description: str = "") -> None:
    """Modülü ilk kullanımda içe aktarılacak blok kaydı: target = "paket.modul:Sinif".

    Başka ajanların (MPC, İHA) blokları için: kütüphane listesinde görünür, modül yalnız blok oluşturulurken yüklenir.
    """
    if ":" not in target:
        raise ValueError("target 'modul:Sinif' biçiminde olmalı")
    if type_name in _REGISTRY:
        return
    _LAZY[type_name] = {"target": target, "display_name": display_name, "category": category,
                        "description": description}


def block_class(type_name: str) -> type[Block]:
    """Tür adından blok sınıfı (gerekirse tembel kaydı yükler)."""
    _ensure_builtin()
    if type_name in _REGISTRY:
        return _REGISTRY[type_name]
    lazy = _LAZY.get(type_name)
    if lazy is not None:
        mod_name, cls_name = lazy["target"].split(":", 1)
        try:
            mod = importlib.import_module(mod_name)
            cls = getattr(mod, cls_name)
        except (ImportError, AttributeError) as exc:
            raise ControlError(f"'{lazy['display_name']}' bloğu yüklenemedi ({exc}); ilgili modül henüz kurulu "
                               "değil.") from None
        if type_name not in _REGISTRY:
            register_block(cls)
        return _REGISTRY[type_name]
    raise ControlError(f"Bilinmeyen blok türü '{type_name}'.")


def create_block(type_name: str, name: str | None = None, **params: Any) -> Block:
    """Tür adı ve parametrelerle blok oluşturur."""
    return block_class(type_name)(name, **params)


def block_types() -> list[str]:
    _ensure_builtin()
    return sorted(set(_REGISTRY) | set(_LAZY))


def block_catalog() -> list[dict[str, Any]]:
    """Arayüz kütüphanesi: her blok türü için tür adı, Türkçe ad, kategori, açıklama, parametre tanımları ve
    varsayılan port etiketleri (JSON uyumlu)."""
    _ensure_builtin()
    out: list[dict[str, Any]] = []
    for name in sorted(_REGISTRY):
        cls = _REGISTRY[name]
        try:
            inst = cls()
            ins, outs = inst.input_labels(), inst.output_labels()
        except ControlError:
            ins, outs = [], []
        out.append({"type": name, "display_name": cls.display_name, "category": cls.category,
                    "category_label": CATEGORIES.get(cls.category, cls.category), "description": cls.description,
                    "simulink": cls.simulink, "params": [p.info() for p in cls.PARAMS],
                    "inputs": ins, "outputs": outs, "lazy": False})
    for name, info in sorted(_LAZY.items()):
        out.append({"type": name, "display_name": info["display_name"], "category": info["category"],
                    "category_label": CATEGORIES.get(info["category"], info["category"]),
                    "description": info["description"], "simulink": "", "params": [], "inputs": [], "outputs": [],
                    "lazy": True})
    return out


_BUILTIN_LOADED = False


def _ensure_builtin() -> None:
    global _BUILTIN_LOADED
    if not _BUILTIN_LOADED:
        _BUILTIN_LOADED = True
        importlib.import_module("spt.control.diagram.blocks")


# ============================================================================ küçük yardımcılar
def as_vec(v: Any) -> np.ndarray:
    return np.atleast_1d(np.asarray(v, dtype=np.float64)).ravel()


def expand(value: np.ndarray, width: int) -> np.ndarray:
    """Skaler parametreyi genişliğe yayar; aksi hâlde aynen döndürür."""
    v = as_vec(value)
    return np.full(width, v[0]) if v.size == 1 and width != 1 else v


def width_join(widths: list[int | None], guess: bool) -> int | None:
    """Skaler genişletmeli ortak genişlik: herhangi biri > 1 ise o; hepsi biliniyor ve 1 ise 1; aksi hâlde None
    (tahminde bilinenlerin en büyüğü)."""
    known = [w for w in widths if w is not None]
    big = [w for w in known if w != 1]
    if big:
        return max(big)
    if len(known) == len(widths) and known:
        return 1
    if guess and known:
        return max(known)
    return None


Fn = Callable[..., Any]
__all__ = ["Block", "SourceBlock", "Param", "SampleTime", "SimContext", "CONTINUOUS_TS", "INHERITED_TS",
           "CONSTANT_TS", "CATEGORIES", "register_block", "register_lazy", "block_class", "create_block",
           "block_types", "block_catalog", "make_sample_time", "p_ts", "expand", "as_vec", "width_join",
           "encode_value"]
