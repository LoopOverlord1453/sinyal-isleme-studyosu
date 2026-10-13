"""Sinyal üreteci (DAC çıkışı için): dalga şekli sentezi, özel tasarımlar, DAC modeli, periyodik tablo,
ölçümler ve STM32 tarzı zamanlayıcı ayarı.

Kurallar (tüm API aynı tanımları kullanır)
------------------------------------------
- Birimler: zaman s, frekans Hz, faz derece, çıkış "volt" (DAC çıkışının fiziksel birimi; `DacSpec` ile aynı).
- Çıkış = offset + amplitude · şekil(…). `amplitude` tepe değeridir (offset'ten en büyük sapma):
    · Çift kutuplu şekiller [−1, +1]: sine, square, triangle, sawtooth, trapezoid, chirp, am, fm, damped_sine;
      noise_white / noise_pink: Gauss için σ = 1/3 (±1 ≈ 3σ), tekdüze (uniform) için [−1, 1].
    · Tek kutuplu şekiller [0, 1] (taban = offset): pulse, staircase, half_rect_sine, full_rect_sine,
      gaussian_pulse, exp_decay; sinc ≈ [−0.22, 1]; ecg ≈ [−0.25, 1] (R tepesi = 1).
    · dc: yalnızca offset kullanılır (şekil = 0).
    · multitone / harmonics / expr / points / samples: şekil kullanıcı birimindedir (amplitude = 1, offset = 0
      verilirse değerler doğrudan volttur).
- Faz: `phase_deg`, temel periyodun derecesi cinsinden başlangıç kaydırmasıdır: ph = (ph0 + phase_deg/360) mod 1.
  İstisnalar: multitone / am / fm / chirp'te taşıyıcıya (her tona) eklenir; damped_sine'da salınımın başlangıç
  fazıdır (patlama yine tekrar periyodunun başında başlar).
- Periyodik türlerde `ph` ∈ [0, 1) periyot içi faz, `t` = n / fs mutlak zamandır.
- `synthesize` faz-sürekli, sürekli zamanlı idealin fs'te örneklenmiş hâlidir; frekans ve fs tamsayıysa faz
  tamsayı aritmetiğiyle hatasız hesaplanır (bkz. `spt.dsp.waveforms`).

DAC modeli
----------
Seviye L ∈ [0, 2^b − 1], 1 LSB = (out_max − out_min) / 2^b, V(L) = out_min + L·LSB (endüstri standardı: en üst kod
out_max − 1 LSB verir; STM32 RM: DAC_OUT = VREF · DOR / 4096). Nicemleme: L = ⌊x + ½⌋, x = (v − out_min)/LSB
(yarım LSB yukarı yuvarlama; negatif olmayan seviyelerde "sıfırdan uzağa" ile aynıdır). x < −½ ya da
x ≥ 2^b − ½ olan örnekler kırpılır ve sayılır (asla sessizce değil). Kodlama: unsigned ve offset_binary sayısal
olarak aynıdır (offset_binary = çift kutuplu çıkış aralığında orta kod = orta gerilim); twos_complement:
kod = L − 2^(b−1). Sola hizalamada kod, kelimenin üst bitlerine kaydırılır: kod · 2^(word_bits − bits).
"""
from __future__ import annotations

import ast
import functools
import math
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import windows as sig_windows

from spt.dsp import waveforms as wf

ACC_BITS = 32                     # DDS faz akümülatörü genişliği (C kodu da 32 bit kullanır)
MAX_TABLE_LEN = 1 << 20           # tablo uzunluğu üst sınırı (örnek)
TABLE_MODES = {"exact": "Tam periyot (DMA dairesel)", "dds": "DDS (2^k tablo + faz akümülatörü)"}


class SignalError(ValueError):
    """Kullanıcıya gösterilecek Türkçe hata. `position`: formül hatalarında 1 tabanlı sütun (yoksa None)."""

    def __init__(self, message: str, position: int | None = None):
        super().__init__(message)
        self.position = position


# ============================================================================ dalga türleri ve parametreler
WAVEFORMS: dict[str, str] = {
    "sine": "Sinüs",
    "square": "Kare",
    "triangle": "Üçgen",
    "sawtooth": "Testere",
    "pulse": "Darbe",
    "trapezoid": "Yamuk",
    "dc": "DC (sabit)",
    "chirp": "Frekans taraması (chirp)",
    "noise_white": "Beyaz gürültü",
    "noise_pink": "Pembe gürültü",
    "multitone": "Çoklu ton",
    "am": "Genlik modülasyonu (AM)",
    "fm": "Frekans modülasyonu (FM)",
    "damped_sine": "Sönümlü sinüs",
    "ecg": "EKG (sentetik kalp atımı)",
    "staircase": "Merdiven",
    "half_rect_sine": "Yarım dalga doğrultulmuş sinüs",
    "full_rect_sine": "Tam dalga doğrultulmuş sinüs",
    "sinc": "Sinc",
    "gaussian_pulse": "Gauss darbesi",
    "exp_decay": "Üstel sönüm",
    "expr": "Formül (özel)",
    "points": "Noktalar / eğri (özel)",
    "harmonics": "Harmonikler / Fourier serisi (özel)",
    "samples": "Örnek dizisi (CSV)",
}

# Temel frekansı `freq` olan tek temel frekanslı periyodik türler
_FREQ_KINDS = frozenset({"sine", "square", "triangle", "sawtooth", "pulse", "trapezoid", "staircase",
                         "half_rect_sine", "full_rect_sine", "sinc", "gaussian_pulse", "exp_decay", "expr",
                         "points", "harmonics"})
_MULTI_KINDS = frozenset({"multitone", "am", "fm"})
NONPERIODIC_KINDS = frozenset({"chirp", "noise_white", "noise_pink"})   # + samples (samples_mode = "play")
UNIPOLAR_KINDS = frozenset({"pulse", "staircase", "half_rect_sine", "full_rect_sine", "sinc", "gaussian_pulse",
                            "exp_decay", "ecg"})

# Her türde anlamlı WaveSpec alanları (amplitude ve offset hepsinde ortaktır; dc yalnızca offset kullanır)
PARAMS: dict[str, tuple[str, ...]] = {
    "sine": ("freq", "phase_deg"),
    "square": ("freq", "phase_deg", "duty"),
    "triangle": ("freq", "phase_deg", "symmetry"),
    "sawtooth": ("freq", "phase_deg", "direction"),
    "pulse": ("freq", "phase_deg", "width"),
    "trapezoid": ("freq", "phase_deg", "rise", "high", "fall", "low"),
    "dc": (),
    "chirp": ("freq", "f1", "sweep_time", "chirp_method", "phase_deg"),
    "noise_white": ("distribution", "seed", "duration"),
    "noise_pink": ("seed", "duration"),
    "multitone": ("tones", "phase_deg"),
    "am": ("freq", "mod_freq", "depth", "phase_deg"),
    "fm": ("freq", "mod_freq", "deviation", "phase_deg"),
    "damped_sine": ("freq", "tau", "repeat_period", "phase_deg"),
    "ecg": ("bpm", "phase_deg"),
    "staircase": ("freq", "steps", "phase_deg"),
    "half_rect_sine": ("freq", "phase_deg"),
    "full_rect_sine": ("freq", "phase_deg"),   # freq = tümsek tekrar frekansı (= 2 × kaynak sinüs frekansı)
    "sinc": ("freq", "width", "phase_deg"),
    "gaussian_pulse": ("freq", "width", "phase_deg"),
    "exp_decay": ("freq", "tau", "phase_deg"),
    "expr": ("freq", "expr", "phase_deg"),
    "points": ("freq", "points", "interp", "phase_deg"),
    "harmonics": ("freq", "harmonics", "phase_deg"),
    "samples": ("samples", "samples_mode", "freq", "interp"),
}

# alan → (Türkçe etiket, birim)
FIELD_INFO: dict[str, tuple[str, str]] = {
    "kind": ("Dalga türü", ""),
    "freq": ("Frekans (chirp: başlangıç, AM/FM: taşıyıcı, sönümlü sinüs: salınım)", "Hz"),
    "amplitude": ("Genlik (tepe, ofsetten sapma)", "V"),
    "offset": ("Ofset / taban", "V"),
    "phase_deg": ("Faz", "°"),
    "duty": ("Doluluk oranı", "0…1"),
    "symmetry": ("Simetri (yükselme payı)", "0…1"),
    "direction": ("Yön", ""),
    "width": ("Genişlik (darbe: üst süre, sinc: ilk sıfır, Gauss: FWHM)", "s"),
    "rise": ("Yükselme payı", "periyot oranı"),
    "high": ("Üst seviye payı", "periyot oranı"),
    "fall": ("Düşme payı", "periyot oranı"),
    "low": ("Alt seviye payı", "periyot oranı"),
    "f1": ("Bitiş frekansı", "Hz"),
    "sweep_time": ("Tarama süresi", "s"),
    "chirp_method": ("Tarama türü", ""),
    "mod_freq": ("Modülasyon frekansı", "Hz"),
    "depth": ("Modülasyon derinliği", "0…1"),
    "deviation": ("Tepe frekans sapması", "Hz"),
    "tau": ("Zaman sabiti τ", "s"),
    "repeat_period": ("Tekrar periyodu", "s"),
    "bpm": ("Kalp hızı", "atım/dk"),
    "steps": ("Basamak sayısı", ""),
    "distribution": ("Dağılım", ""),
    "seed": ("Rastgele tohum", ""),
    "duration": ("Tablo süresi", "s"),
    "tones": ("Tonlar [(f, genlik, faz°)]", "Hz, V, °"),
    "expr": ("Formül", ""),
    "points": ("Noktalar [(faz 0…1, değer)]", ""),
    "interp": ("Ara değerleme", ""),
    "harmonics": ("Harmonikler [(n, genlik, faz°)]", ""),
    "samples": ("Örnekler", ""),
    "samples_mode": ("Örnek kipi", ""),
}

CHOICES: dict[str, dict[str, str]] = {
    "direction": {"rising": "Yükselen", "falling": "Düşen"},
    "chirp_method": {"linear": "Doğrusal", "log": "Logaritmik"},
    "distribution": {"gaussian": "Gauss (σ = 1/3)", "uniform": "Tekdüze [−1, 1]"},
    "interp": {"step": "Basamak", "linear": "Doğrusal", "cubic": "Kübik (periyodik spline)"},
    "samples_mode": {"play": "fs'te sırayla çal", "period": "Bir periyot kabul et (freq'te çal)"},
}

_STR_FIELDS = frozenset({"kind", "direction", "chirp_method", "distribution", "expr", "interp", "samples_mode"})
_INT_FIELDS = frozenset({"steps", "seed"})
_ROW_FIELDS = {"tones": 3, "harmonics": 3, "points": 2}


def _num(value: Any, name: str) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise SignalError(f"'{name}' sayı olmalı (verilen: {value!r}).") from None
    return v


@dataclass(frozen=True)
class WaveSpec:
    """Bir dalga şeklinin tüm tanımı (değişmez). Listeler / numpy dizileri otomatik olarak demete çevrilir.

    Her alanın anlamı ve birimi `FIELD_INFO`'da, türlere göre kullanılan alanlar `PARAMS`'ta. Kaydetmek için
    `to_dict()` (JSON uyumlu), geri yüklemek için `WaveSpec.from_dict(d)`.
    """
    kind: str = "sine"
    freq: float = 1000.0
    amplitude: float = 1.0
    offset: float = 0.0
    phase_deg: float = 0.0
    duty: float = 0.5
    symmetry: float = 0.5
    direction: str = "rising"
    width: float = 1e-4
    rise: float = 0.1
    high: float = 0.4
    fall: float = 0.1
    low: float = 0.4
    f1: float = 10000.0
    sweep_time: float = 0.01
    chirp_method: str = "linear"
    mod_freq: float = 100.0
    depth: float = 0.5
    deviation: float = 500.0
    tau: float = 1e-3
    repeat_period: float = 0.01
    bpm: float = 60.0
    steps: int = 8
    distribution: str = "gaussian"
    seed: int = 0
    duration: float = 0.01
    tones: tuple[tuple[float, float, float], ...] = ((1000.0, 0.5, 0.0), (3000.0, 0.25, 0.0))
    expr: str = "sin(2*pi*ph)"
    points: tuple[tuple[float, float], ...] = ((0.0, -1.0), (0.25, 1.0), (0.5, 0.0), (0.75, 0.5))
    interp: str = "linear"
    harmonics: tuple[tuple[float, float, float], ...] = ((1.0, 1.0, 0.0), (3.0, 1.0 / 3.0, 0.0))
    samples: tuple[float, ...] = ()
    samples_mode: str = "play"

    def __post_init__(self) -> None:
        for f in fields(self):
            name, v = f.name, getattr(self, f.name)
            if name in _STR_FIELDS:
                if not isinstance(v, str):
                    raise SignalError(f"'{name}' metin olmalı (verilen: {v!r}).")
                continue
            if name in _INT_FIELDS:
                fv = _num(v, name)
                if not fv.is_integer():
                    raise SignalError(f"'{name}' tam sayı olmalı (verilen: {v!r}).")
                object.__setattr__(self, name, int(fv))
            elif name in _ROW_FIELDS:
                width = _ROW_FIELDS[name]
                rows = []
                for row in (v if v is not None else ()):
                    r = tuple(_num(x, name) for x in row)
                    if len(r) != width:
                        raise SignalError(f"'{name}' satırları {width} elemanlı olmalı (verilen: {row!r}).")
                    rows.append(r)
                object.__setattr__(self, name, tuple(rows))
            elif name == "samples":
                arr = np.asarray(v if v is not None else (), dtype=np.float64).ravel()
                object.__setattr__(self, name, tuple(arr.tolist()))
            else:
                object.__setattr__(self, name, _num(v, name))
        if self.kind not in WAVEFORMS:
            raise SignalError(f"Bilinmeyen dalga türü '{self.kind}'. Geçerli türler: {', '.join(WAVEFORMS)}.")

    def to_dict(self) -> dict[str, Any]:
        """JSON'a doğrudan yazılabilir sözlük (demetler → listeler)."""
        out: dict[str, Any] = {}
        for f in fields(self):
            v = getattr(self, f.name)
            if isinstance(v, tuple):
                v = [list(x) if isinstance(x, tuple) else x for x in v]
            out[f.name] = v
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WaveSpec:
        """`to_dict` çıktısından geri yükler; bilinmeyen anahtarlar (ileri sürümler) yok sayılır."""
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in dict(data).items() if k in known})


def fundamental(spec: WaveSpec) -> float | None:
    """Tek temel frekanslı periyodik türlerin tekrar frekansı (Hz); diğerlerinde None."""
    k = spec.kind
    if k == "damped_sine":
        return 1.0 / spec.repeat_period if spec.repeat_period > 0 else None
    if k == "ecg":
        return spec.bpm / 60.0
    if k in _FREQ_KINDS or (k == "samples" and spec.samples_mode == "period"):
        return spec.freq
    return None


def is_periodic(spec: WaveSpec) -> bool:
    """Tablonun tam sayıda periyot tutabileceği tür mü (dc dahil)?"""
    return _components(spec) is not None


def _with_fundamental(spec: WaveSpec, f: float) -> WaveSpec:
    if spec.kind == "damped_sine":
        return replace(spec, repeat_period=1.0 / f)
    if spec.kind == "ecg":
        return replace(spec, bpm=60.0 * f)
    return replace(spec, freq=f)


def _components(spec: WaveSpec) -> list[float] | None:
    """Tabloda her biri tam sayıda periyot tamamlaması gereken frekanslar; periyodik değilse None."""
    k = spec.kind
    if k == "dc":
        return []
    if k in ("am", "fm"):
        return [spec.freq, spec.mod_freq]
    if k == "multitone":
        return [t[0] for t in spec.tones]
    f = fundamental(spec)
    return None if f is None else [f]


def _apply_components(spec: WaveSpec, freqs: list[float]) -> WaveSpec:
    k = spec.kind
    if k == "dc":
        return spec
    if k in ("am", "fm"):
        return replace(spec, freq=freqs[0], mod_freq=freqs[1])
    if k == "multitone":
        return replace(spec, tones=tuple((fr, a, p) for fr, (_, a, p) in zip(freqs, spec.tones)))
    return _with_fundamental(spec, freqs[0])


def _fmt_hz(f: float) -> str:
    a = abs(f)
    if a >= 1e6:
        return f"{f / 1e6:.6g} MHz"
    if a >= 1e3:
        return f"{f / 1e3:.6g} kHz"
    return f"{f:.6g} Hz"


def describe(spec: WaveSpec) -> str:
    """Kısa Türkçe açıklama, ör. "Sinüs, 1 kHz, genlik 1.5 V, ofset 1.65 V"."""
    k = spec.kind
    parts = [WAVEFORMS[k]]
    if k in ("am", "fm"):
        extra = f"derinlik {spec.depth:g}" if k == "am" else f"sapma {_fmt_hz(spec.deviation)}"
        parts.append(f"taşıyıcı {_fmt_hz(spec.freq)}, modülasyon {_fmt_hz(spec.mod_freq)}, {extra}")
    elif k == "chirp":
        parts.append(f"{_fmt_hz(spec.freq)} → {_fmt_hz(spec.f1)} / {spec.sweep_time:g} s ({spec.chirp_method})")
    elif k == "multitone":
        parts.append(" + ".join(_fmt_hz(t[0]) for t in spec.tones))
    elif k == "ecg":
        parts.append(f"{spec.bpm:g} atım/dk")
    elif k == "samples":
        parts.append(f"{len(spec.samples)} örnek ({spec.samples_mode})")
    elif k == "expr":
        parts.append(f"'{spec.expr}', {_fmt_hz(spec.freq)}")
    else:
        f = fundamental(spec)
        if f:
            parts.append(_fmt_hz(f))
    if k == "square":
        parts.append(f"doluluk {spec.duty:g}")
    if k != "dc":
        parts.append(f"genlik {spec.amplitude:g} V")
    parts.append(f"ofset {spec.offset:g} V")
    return ", ".join(parts)


# ============================================================================ formül (güvenli ifade)
MAX_EXPR_LEN = 1000
MAX_EXPONENT = 100.0              # sabit üs sınırı (ör. 9**9**9 reddedilir)
_MAX_NODES = 500
_MAX_DEPTH = 200     # sol-ilişkili uzun toplamlar (a + b + c …) her terimde bir düzey derinleşir

_CONSTS: dict[str, float] = {"pi": math.pi, "e": math.e}
EXPR_VARIABLES: dict[str, str] = {
    "t": "zaman (s), t = n / fs",
    "ph": "periyot içi faz, 0 … 1 (faz kaydırması dahil)",
    "f": "temel frekans (Hz)",
    "fs": "örnekleme hızı (Hz)",
    "n": "örnek indisi",
    "pi": "π sabiti",
    "e": "e sabiti",
}


def _f_square(x, duty=0.5):
    return np.where(np.mod(x, 1.0) < duty, 1.0, -1.0)


def _f_saw(x):
    return 2.0 * np.mod(x, 1.0) - 1.0


def _f_tri(x):
    return 1.0 - 4.0 * np.abs(np.mod(x, 1.0) - 0.5)


def _f_pulse(x, width=0.5):
    return np.where(np.mod(x, 1.0) < width, 1.0, 0.0)


def _f_step(x):
    return np.where(np.asarray(x) >= 0, 1.0, 0.0)


def _f_round(x):
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


def _f_min(*args):
    return functools.reduce(np.minimum, args)


def _f_max(*args):
    return functools.reduce(np.maximum, args)


def _f_where(cond, a, b):
    return np.where(np.asarray(cond) != 0, a, b)


def _f_gauss(x):
    return np.exp(-0.5 * np.square(x))


# ad → (fonksiyon, en az argüman, en çok argüman, açıklama)
_FUNCS: dict[str, tuple[Callable[..., Any], int, int, str]] = {
    "sin": (np.sin, 1, 1, "sinüs (radyan)"),
    "cos": (np.cos, 1, 1, "kosinüs (radyan)"),
    "tan": (np.tan, 1, 1, "tanjant (radyan)"),
    "asin": (np.arcsin, 1, 1, "ters sinüs"),
    "acos": (np.arccos, 1, 1, "ters kosinüs"),
    "atan": (np.arctan, 1, 1, "ters tanjant"),
    "atan2": (np.arctan2, 2, 2, "atan2(y, x)"),
    "sinh": (np.sinh, 1, 1, "hiperbolik sinüs"),
    "cosh": (np.cosh, 1, 1, "hiperbolik kosinüs"),
    "tanh": (np.tanh, 1, 1, "hiperbolik tanjant"),
    "exp": (np.exp, 1, 1, "e üssü"),
    "log": (np.log, 1, 1, "doğal logaritma"),
    "log10": (np.log10, 1, 1, "10 tabanında logaritma"),
    "log2": (np.log2, 1, 1, "2 tabanında logaritma"),
    "sqrt": (np.sqrt, 1, 1, "karekök"),
    "abs": (np.abs, 1, 1, "mutlak değer"),
    "sign": (np.sign, 1, 1, "işaret (−1, 0, 1)"),
    "floor": (np.floor, 1, 1, "aşağı yuvarla"),
    "ceil": (np.ceil, 1, 1, "yukarı yuvarla"),
    "round": (_f_round, 1, 1, "en yakına yuvarla (yarım sıfırdan uzağa)"),
    "mod": (np.mod, 2, 2, "mod(x, y), sonuç y'nin işaretinde"),
    "min": (_f_min, 2, 8, "eleman bazında en küçük"),
    "max": (_f_max, 2, 8, "eleman bazında en büyük"),
    "clip": (np.clip, 3, 3, "clip(x, alt, üst)"),
    "where": (_f_where, 3, 3, "where(koşul, a, b): koşul ≠ 0 ise a, değilse b"),
    "hypot": (np.hypot, 2, 2, "√(x² + y²)"),
    "square": (_f_square, 1, 2, "square(ph, doluluk=0.5): ±1 kare dalga"),
    "saw": (_f_saw, 1, 1, "saw(ph): −1 → +1 testere"),
    "tri": (_f_tri, 1, 1, "tri(ph): 0'da −1, 0.5'te +1 üçgen"),
    "pulse": (_f_pulse, 1, 2, "pulse(ph, genişlik=0.5): 0/1 darbe"),
    "step": (_f_step, 1, 1, "step(x): x ≥ 0 ise 1, değilse 0"),
    "sinc": (np.sinc, 1, 1, "sin(πx)/(πx)"),
    "gauss": (_f_gauss, 1, 1, "exp(−x²/2)"),
}
EXPR_FUNCTIONS: dict[str, str] = {k: v[3] for k, v in _FUNCS.items()}

_BINOPS: dict[type, Callable[[Any, Any], Any]] = {
    ast.Add: np.add, ast.Sub: np.subtract, ast.Mult: np.multiply, ast.Div: np.true_divide,
    ast.FloorDiv: np.floor_divide, ast.Mod: np.mod, ast.Pow: np.power,
}
_UNARY: dict[type, Callable[[Any], Any]] = {
    ast.USub: np.negative, ast.UAdd: np.positive,
    ast.Not: lambda a: (np.asarray(a) == 0).astype(np.float64),
}
_CMP: dict[type, Callable[[Any, Any], Any]] = {
    ast.Lt: np.less, ast.LtE: np.less_equal, ast.Gt: np.greater, ast.GtE: np.greater_equal,
    ast.Eq: np.equal, ast.NotEq: np.not_equal,
}
_FORBIDDEN: dict[type, str] = {
    ast.Attribute: "öznitelik erişimi (nokta '.') kullanılamaz",
    ast.Subscript: "indeksleme ('[...]') kullanılamaz",
    ast.Slice: "dilimleme kullanılamaz",
    ast.Lambda: "lambda kullanılamaz",
    ast.ListComp: "liste üreteci kullanılamaz",
    ast.SetComp: "küme üreteci kullanılamaz",
    ast.DictComp: "sözlük üreteci kullanılamaz",
    ast.GeneratorExp: "üreteç ifadesi kullanılamaz",
    ast.Tuple: "demet / virgülle ayrılmış ifade kullanılamaz",
    ast.List: "liste kullanılamaz",
    ast.Dict: "sözlük kullanılamaz",
    ast.Set: "küme kullanılamaz",
    ast.JoinedStr: "metin kullanılamaz",
    ast.NamedExpr: "':=' ataması kullanılamaz",
    ast.Starred: "'*' açma kullanılamaz",
    ast.Await: "await kullanılamaz",
    ast.Yield: "yield kullanılamaz",
    ast.YieldFrom: "yield kullanılamaz",
}


def _at(pos: int | None) -> str:
    return f" (sütun {pos})" if pos else ""


def _eval_node(node: ast.AST, env: dict[str, Any]) -> Any:
    """Doğrulanmış ağacı numpy ile vektörel değerlendirir (eval/compile kullanılmaz)."""
    if isinstance(node, ast.Expression):
        return _eval_node(node.body, env)
    if isinstance(node, ast.Constant):
        return np.float64(node.value)
    if isinstance(node, ast.Name):
        if node.id in _CONSTS:
            return np.float64(_CONSTS[node.id])
        return env[node.id]
    if isinstance(node, ast.BinOp):
        return _BINOPS[type(node.op)](_eval_node(node.left, env), _eval_node(node.right, env))
    if isinstance(node, ast.UnaryOp):
        return _UNARY[type(node.op)](_eval_node(node.operand, env))
    if isinstance(node, ast.Compare):
        left = _eval_node(node.left, env)
        result = None
        for op, comp in zip(node.ops, node.comparators):
            right = _eval_node(comp, env)
            r = _CMP[type(op)](left, right)
            result = r if result is None else np.logical_and(result, r)
            left = right
        return np.asarray(result, dtype=np.float64)
    if isinstance(node, ast.BoolOp):
        vals = [np.asarray(_eval_node(v, env)) != 0 for v in node.values]
        fn = np.logical_and if isinstance(node.op, ast.And) else np.logical_or
        return functools.reduce(fn, vals).astype(np.float64)
    if isinstance(node, ast.IfExp):
        return np.where(np.asarray(_eval_node(node.test, env)) != 0,
                        _eval_node(node.body, env), _eval_node(node.orelse, env))
    if isinstance(node, ast.Call):
        assert isinstance(node.func, ast.Name)
        return _FUNCS[node.func.id][0](*[_eval_node(a, env) for a in node.args])
    raise SignalError("İç hata: doğrulanmamış formül öğesi.")


class _Validator:
    """Beyaz liste denetimi: yalnızca sayı sabitleri, izinli adlar/fonksiyonlar ve aritmetik/karşılaştırma."""

    def __init__(self, shift: int, body: str = ""):
        self.shift = shift
        self.count = 0
        self._utf8 = body.encode("utf-8")

    def pos(self, node: ast.AST) -> int | None:
        col = getattr(node, "col_offset", None)  # UTF-8 bayt konumu → karakter konumu
        if col is None:
            return None
        chars = len(self._utf8[:col].decode("utf-8", errors="ignore")) if self._utf8 else col
        return chars + 1 + self.shift

    def fail(self, node: ast.AST, msg: str) -> None:
        p = self.pos(node)
        raise SignalError(f"{msg}{_at(p)}.", p)

    def const_value(self, node: ast.AST) -> float | None:
        """Değişken içermeyen alt ağacın değeri (float); değişken varsa None."""
        try:
            with np.errstate(all="ignore"):
                return float(_eval_node(node, {}))
        except KeyError:
            return None
        except (ArithmeticError, ValueError, TypeError):
            return math.inf

    def check(self, node: ast.AST, depth: int = 0) -> None:
        self.count += 1
        if self.count > _MAX_NODES:
            raise SignalError(f"Formül çok karmaşık (en fazla {_MAX_NODES} öğe).")
        if depth > _MAX_DEPTH:
            self.fail(node, "Formül çok derin iç içe")
        d = depth + 1
        if isinstance(node, ast.Expression):
            self.check(node.body, d)
        elif isinstance(node, ast.Constant):
            v = node.value
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                self.fail(node, "Yalnızca gerçel sayı sabitleri kullanılabilir")
            try:
                fv = float(v)
            except OverflowError:
                fv = math.inf
            if not math.isfinite(fv):
                self.fail(node, "Sayı çok büyük")
        elif isinstance(node, ast.Name):
            if node.id in EXPR_VARIABLES:
                return
            if node.id in _FUNCS:
                self.fail(node, f"'{node.id}' bir fonksiyondur; {node.id}(…) biçiminde çağırın")
            self.fail(node, f"Bilinmeyen ad '{node.id}'; kullanılabilir adlar: {', '.join(EXPR_VARIABLES)}")
        elif isinstance(node, ast.BinOp):
            if type(node.op) not in _BINOPS:
                self.fail(node, "Bu işleç kullanılamaz (yalnızca + − * / // % **)")
            self.check(node.left, d)
            self.check(node.right, d)
            if isinstance(node.op, ast.Pow):
                val = self.const_value(node.right)
                if val is not None and (not math.isfinite(val) or abs(val) > MAX_EXPONENT):
                    self.fail(node.right, f"Üs çok büyük (sabit üs için |üs| ≤ {MAX_EXPONENT:g})")
        elif isinstance(node, ast.UnaryOp):
            if type(node.op) not in _UNARY:
                self.fail(node, "Bu tekli işleç kullanılamaz (yalnızca −, +, not)")
            self.check(node.operand, d)
        elif isinstance(node, ast.Compare):
            for op in node.ops:
                if type(op) not in _CMP:
                    self.fail(node, "Bu karşılaştırma kullanılamaz (yalnızca < <= > >= == !=)")
            self.check(node.left, d)
            for c in node.comparators:
                self.check(c, d)
        elif isinstance(node, ast.BoolOp):
            for v in node.values:
                self.check(v, d)
        elif isinstance(node, ast.IfExp):
            self.check(node.test, d)
            self.check(node.body, d)
            self.check(node.orelse, d)
        elif isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                self.fail(node, "Yalnızca izinli fonksiyonlar doğrudan adıyla çağrılabilir")
            name = node.func.id
            if name not in _FUNCS:
                self.fail(node, f"İzin verilmeyen fonksiyon '{name}'; izinliler: {', '.join(_FUNCS)}")
            if node.keywords:
                self.fail(node, f"'{name}' çağrısında anahtar sözcüklü argüman kullanılamaz")
            _, lo, hi, _ = _FUNCS[name]
            if not lo <= len(node.args) <= hi:
                rng = f"{lo}" if lo == hi else f"{lo}–{hi}"
                self.fail(node, f"'{name}' {rng} argüman alır, {len(node.args)} verildi")
            for a in node.args:
                self.check(a, d)
        else:
            self.fail(node, _FORBIDDEN.get(type(node), f"'{type(node).__name__}' öğesi formülde kullanılamaz"))


@functools.lru_cache(maxsize=128)
def _parse_expr(text: str) -> ast.Expression:
    if len(text) > MAX_EXPR_LEN:
        raise SignalError(f"Formül çok uzun ({len(text)} karakter; en fazla {MAX_EXPR_LEN}).")
    src = text.replace("\r", " ").replace("\n", " ")
    body = src.lstrip()
    shift = len(src) - len(body)
    if not body.strip():
        raise SignalError("Formül boş.")
    try:
        tree = ast.parse(body, mode="eval")
    except SyntaxError as exc:
        pos = exc.offset + shift if exc.offset else None
        raise SignalError(f"Sözdizimi hatası{_at(pos)}: {exc.msg}.", pos) from None
    except (ValueError, RecursionError, MemoryError) as exc:
        raise SignalError(f"Formül çözümlenemedi: {exc}.") from None
    _Validator(shift, body).check(tree)
    return tree


def check_expr(text: str) -> None:
    """Formülü değerlendirmeden doğrular; hatalıysa `SignalError` (konumlu) yükseltir."""
    if not isinstance(text, str):
        raise SignalError("Formül metin olmalı.")
    _parse_expr(text)


def eval_expr(text: str, **variables: Any) -> np.ndarray:
    """Güvenli formül değerlendirme. Değişkenler: t, ph, f, fs, n (skaler ya da dizi; yayınlanır).

    Sonuç, değişkenlerin ortak biçiminde float64 dizidir. NaN / sonsuz üreten noktalarda `SignalError`.
    """
    check_expr(text)
    tree = _parse_expr(text)
    env = {k: np.asarray(v, dtype=np.float64) for k, v in variables.items()}
    try:
        with np.errstate(all="ignore"):
            out = _eval_node(tree, env)
    except KeyError as exc:
        raise SignalError(f"'{exc.args[0]}' bu bağlamda tanımlı değil.") from None
    except (ValueError, TypeError) as exc:
        raise SignalError(f"Formül değerlendirilemedi: {exc}.") from None
    shape = np.broadcast_shapes(*(a.shape for a in env.values())) if env else ()
    res = np.array(np.broadcast_to(np.asarray(out, dtype=np.float64), shape), dtype=np.float64)
    bad = ~np.isfinite(res)
    if bad.any():
        i = int(np.flatnonzero(bad.ravel())[0])
        where = ""
        t = env.get("t")
        if t is not None and t.ndim == 1 and t.size == res.size:
            where = f", t = {t[i]:.6g} s"
        ph = env.get("ph")
        if ph is not None and ph.ndim == 1 and ph.size == res.size:
            where += f", ph = {ph[i]:.6g}"
        raise SignalError(f"Formül {i}. örnekte tanımsız ya da sonsuz değer üretti{where} "
                          "(ör. sıfıra bölme, log(0), negatif sayının karekökü).")
    return res


# ============================================================================ doğrulama
def _finite(v: float, label: str) -> None:
    if not math.isfinite(v):
        raise SignalError(f"{label} sonlu bir sayı olmalı.")


def _positive(v: float, label: str) -> None:
    if not (math.isfinite(v) and v > 0):
        raise SignalError(f"{label} pozitif olmalı (verilen {v:g}).")


def _check_choice(spec: WaveSpec, name: str) -> None:
    v = getattr(spec, name)
    if v not in CHOICES[name]:
        raise SignalError(f"'{name}' için geçersiz seçim '{v}'; geçerliler: {', '.join(CHOICES[name])}.")


def _check_points(points: tuple[tuple[float, float], ...], interp: str) -> None:
    if not points:
        raise SignalError("En az bir nokta gerekli.")
    if len(points) > 100_000:
        raise SignalError("Çok fazla nokta (en fazla 100000).")
    p = np.array([q[0] for q in points])
    v = np.array([q[1] for q in points])
    if not (np.all(np.isfinite(p)) and np.all(np.isfinite(v))):
        raise SignalError("Noktalar sonlu sayılar olmalı.")
    if np.any(p < 0) or np.any(p > 1):
        raise SignalError("Nokta fazları 0 … 1 aralığında olmalı.")
    if interp == "cubic" and len(np.unique(p)) != len(p):
        raise SignalError("Kübik eğride aynı fazda iki nokta olamaz (sıçrama için 'linear' ya da 'step' kullanın).")


def validate(spec: WaveSpec, fs: float) -> None:
    """Parametreleri fs'e göre denetler; hatalıysa Türkçe `SignalError`."""
    _positive(fs, "Örnekleme hızı fs")
    k = spec.kind
    _finite(spec.amplitude, "Genlik")
    _finite(spec.offset, "Ofset")
    _finite(spec.phase_deg, "Faz")
    nyq = fs / 2.0

    def need_freq(f: float, label: str) -> None:
        if not (math.isfinite(f) and 0 < f < nyq):
            raise SignalError(f"{label} 0 < f < fs/2 = {nyq:g} Hz aralığında olmalı (verilen {f:g} Hz).")

    if k in _FREQ_KINDS:
        need_freq(spec.freq, "Frekans")
    if k == "square" and not 0.0 <= spec.duty <= 1.0:
        raise SignalError("Doluluk oranı 0 … 1 aralığında olmalı.")
    if k == "triangle" and not 0.0 <= spec.symmetry <= 1.0:
        raise SignalError("Simetri 0 … 1 aralığında olmalı.")
    if k == "sawtooth":
        _check_choice(spec, "direction")
    if k == "pulse":
        _positive(spec.width, "Darbe genişliği")
        if spec.width >= 1.0 / spec.freq:
            raise SignalError(f"Darbe genişliği ({spec.width:g} s) periyottan ({1.0 / spec.freq:g} s) küçük olmalı.")
    if k == "trapezoid":
        parts = (spec.rise, spec.high, spec.fall, spec.low)
        if any((not math.isfinite(x)) or x < 0 for x in parts) or sum(parts) <= 0:
            raise SignalError("Yamuk payları negatif olmayan sayılar olmalı ve toplamı > 0 olmalı.")
    if k == "staircase" and not 2 <= spec.steps <= 65536:
        raise SignalError("Basamak sayısı 2 … 65536 aralığında olmalı.")
    if k in ("sinc", "gaussian_pulse"):
        _positive(spec.width, "Genişlik")
    if k in ("exp_decay", "damped_sine"):
        _positive(spec.tau, "Zaman sabiti τ")
    if k == "damped_sine":
        need_freq(spec.freq, "Salınım frekansı")
        _positive(spec.repeat_period, "Tekrar periyodu")
        need_freq(1.0 / spec.repeat_period, "Tekrar frekansı (1/periyot)")
    if k == "ecg":
        if not (math.isfinite(spec.bpm) and 20.0 <= spec.bpm <= 300.0):
            raise SignalError("Kalp hızı 20 … 300 atım/dk aralığında olmalı.")
        need_freq(spec.bpm / 60.0, "Atım frekansı")
    if k == "chirp":
        _check_choice(spec, "chirp_method")
        _positive(spec.sweep_time, "Tarama süresi")
        lo = 0.0 if spec.chirp_method == "linear" else None
        for f, label in ((spec.freq, "Başlangıç frekansı"), (spec.f1, "Bitiş frekansı")):
            ok = math.isfinite(f) and f <= nyq and (f >= 0 if lo is not None else f > 0)
            if not ok:
                rng = "0 ≤ f" if lo is not None else "0 < f"
                raise SignalError(f"{label} {rng} ≤ fs/2 = {nyq:g} Hz olmalı (verilen {f:g} Hz).")
    if k == "noise_white":
        _check_choice(spec, "distribution")
    if k in ("noise_white", "noise_pink"):
        _positive(spec.duration, "Tablo süresi")
        if spec.seed < 0:
            raise SignalError("Tohum negatif olamaz.")
    if k == "multitone":
        if not spec.tones:
            raise SignalError("En az bir ton gerekli.")
        for f, a, p in spec.tones:
            need_freq(f, "Ton frekansı")
            _finite(a, "Ton genliği")
            _finite(p, "Ton fazı")
    if k in ("am", "fm"):
        need_freq(spec.freq, "Taşıyıcı frekansı")
        need_freq(spec.mod_freq, "Modülasyon frekansı")
    if k == "am" and not 0.0 <= spec.depth <= 1.0:
        raise SignalError("Modülasyon derinliği 0 … 1 aralığında olmalı.")
    if k == "fm" and not (math.isfinite(spec.deviation) and spec.deviation >= 0):
        raise SignalError("Frekans sapması negatif olmayan bir sayı olmalı.")
    if k == "expr":
        check_expr(spec.expr)
    if k == "points":
        _check_choice(spec, "interp")
        _check_points(spec.points, spec.interp)
    if k == "harmonics":
        if not spec.harmonics:
            raise SignalError("En az bir harmonik gerekli.")
        for n, a, p in spec.harmonics:
            if not (math.isfinite(n) and n >= 0 and float(n).is_integer()):
                raise SignalError(f"Harmonik numarası negatif olmayan tam sayı olmalı (verilen {n:g}).")
            _finite(a, "Harmonik genliği")
            _finite(p, "Harmonik fazı")
    if k == "samples":
        _check_choice(spec, "samples_mode")
        if not spec.samples:
            raise SignalError("Örnek dizisi boş: “CSV / metin dosyası aç…” ile örnek yükleyin.")
        if not np.all(np.isfinite(spec.samples)):
            raise SignalError("Örnek dizisinde NaN / sonsuz değer var.")
        if spec.samples_mode == "period":
            need_freq(spec.freq, "Frekans")
            _check_choice(spec, "interp")


# ============================================================================ sentez
class _Clock:
    """Örnek indisleri ve faz üreteci. `table_len` verilirse, tabloda tam sayıda periyot tamamlayan frekansların
    fazı tamsayı aritmetiğiyle (n·M mod N)/N hesaplanır: tablo tam periyodik olur."""

    def __init__(self, count: int, fs: float, table_len: int | None = None, dac_fs: float | None = None):
        self.fs = float(fs)
        self.dac_fs = float(dac_fs) if dac_fs else self.fs  # formüldeki `fs`: DAC güncelleme hızı (DDS'te ≠ tablo saati)
        self.size = int(count)
        self.n = np.arange(self.size, dtype=np.int64)
        self.t = self.n / self.fs
        self.table_len = table_len

    def phase(self, f: float) -> np.ndarray:
        if self.table_len:
            x = f * self.table_len / self.fs
            m = round(x)
            if m > 0 and abs(x - m) <= 1e-9 * x:
                return wf.cycle_phase(self.n, int(m), self.table_len)
        return wf.phase(self.n, f, self.fs)


_ECG_WAVES = (  # (R'ye göre merkez s, genlik, σ s, RR ile ölçeklenir mi) — P, Q, R, S, T
    (-0.20, 0.12, 0.025, True),
    (-0.035, -0.10, 0.010, False),
    (0.0, 1.00, 0.012, False),
    (0.035, -0.22, 0.012, False),
    (0.30, 0.30, 0.050, True),
)


def _ecg(ph: np.ndarray, period: float) -> np.ndarray:
    """Gauss toplamıyla sentetik EKG; P ve T dalgaları √RR ile ölçeklenir (Bazett benzeri), QRS sabit."""
    s = math.sqrt(period)
    t_r = 0.35 * period

    def at(tl: np.ndarray) -> np.ndarray:
        out = np.zeros_like(tl)
        for c, a, sig, scaled in _ECG_WAVES:
            cc = t_r + (c * s if scaled else c)
            sg = sig * s if scaled else sig
            d = np.mod(tl - cc + period / 2, period) - period / 2
            out += a * np.exp(-0.5 * (d / sg) ** 2)
        return out

    return at(ph * period) / float(at(np.array([t_r]))[0])


def _pink(n: int, seed: int) -> np.ndarray:
    """1/f güç yoğunluklu (−3.01 dB/oktav) Gauss gürültü, FFT ile şekillendirilmiş; dairesel olduğundan tablo
    döngüsünde dikiş yeri yoktur. σ = 1/3'e normalize edilir."""
    if n < 4:
        return np.zeros(n)
    rng = np.random.default_rng(seed)
    m = n // 2 + 1
    spec = rng.standard_normal(m) + 1j * rng.standard_normal(m)
    k = np.arange(m, dtype=np.float64)
    k[0] = 1.0
    spec /= np.sqrt(k)
    spec[0] = 0.0
    if n % 2 == 0:
        spec[-1] = spec[-1].real
    x = np.fft.irfft(spec, n)
    sd = x.std()
    return x * ((1.0 / 3.0) / sd) if sd > 0 else x


def _points_shape(p: np.ndarray, v: np.ndarray, interp: str, ph: np.ndarray) -> np.ndarray:
    """Periyodik kırılma noktası eğrisi. 1.0 fazındaki nokta bir sonraki periyodun 0 fazıdır; 0 ve 1'de farklı
    değer verilirse periyot sınırında bilinçli sıçrama olur."""
    order = np.argsort(p, kind="stable")
    p, v = np.asarray(p, dtype=np.float64)[order], np.asarray(v, dtype=np.float64)[order]
    if len(p) == 1:
        return np.full(ph.shape, v[0])
    if interp == "step":
        idx = np.searchsorted(p, ph, side="right") - 1      # −1 → son nokta (sarma)
        return v[idx]
    if interp == "linear":
        xp = np.concatenate(([p[-1] - 1.0], p, [p[0] + 1.0]))
        fp = np.concatenate(([v[-1]], v, [v[0]]))
        return np.interp(ph, xp, fp)
    # kübik
    if p[-1] == 1.0:
        if p[0] == 0.0:
            if v[0] != v[-1]:                    # bilinçli sıçrama: periyodik olmayan spline [0, 1]
                cs = CubicSpline(p, v, bc_type="not-a-knot" if len(p) >= 4 else "natural")
                return cs(ph)
            p, v = p[:-1], v[:-1]                # 1'deki nokta 0'dakinin tekrarı
        else:                                    # 1'deki nokta = 0 fazı
            p = np.concatenate(([0.0], p[:-1]))
            v = np.concatenate(([v[-1]], v[:-1]))
    if len(p) == 1:
        return np.full(ph.shape, v[0])
    x = np.concatenate((p, [p[0] + 1.0]))
    y = np.concatenate((v, [v[0]]))
    cs = CubicSpline(x, y, bc_type="periodic")
    return cs(p[0] + np.mod(ph - p[0], 1.0))


def _chirp(spec: WaveSpec, clk: _Clock) -> np.ndarray:
    T = spec.sweep_time
    ns = T * clk.fs
    if abs(ns - round(ns)) < 1e-9 and round(ns) > 0:
        tl = (clk.n % int(round(ns))) / clk.fs
    else:
        tl = np.mod(clk.t, T)
    f0, f1 = spec.freq, spec.f1
    if spec.chirp_method == "log" and f1 != f0:
        k = f1 / f0
        cyc = f0 * T / math.log(k) * (np.power(k, tl / T) - 1.0)
    else:
        cyc = f0 * tl + (f1 - f0) * tl * tl / (2.0 * T)
    return np.sin(2 * np.pi * (cyc + spec.phase_deg / 360.0))


def _shape(spec: WaveSpec, clk: _Clock) -> np.ndarray:
    k = spec.kind
    two_pi = 2.0 * np.pi
    if k == "noise_white":
        rng = np.random.default_rng(spec.seed)
        if spec.distribution == "uniform":
            return rng.uniform(-1.0, 1.0, clk.size)
        return rng.standard_normal(clk.size) / 3.0
    if k == "noise_pink":
        return _pink(clk.size, spec.seed)
    if k == "chirp":
        return _chirp(spec, clk)
    if k == "multitone":
        out = np.zeros(clk.size)
        for f, a, p in spec.tones:
            out += a * np.sin(two_pi * (clk.phase(f) + (p + spec.phase_deg) / 360.0))
        return out
    if k == "am":
        env = 1.0 + spec.depth * np.sin(two_pi * clk.phase(spec.mod_freq))
        return env * np.sin(two_pi * (clk.phase(spec.freq) + spec.phase_deg / 360.0)) / (1.0 + spec.depth)
    if k == "fm":
        beta = spec.deviation / spec.mod_freq
        return np.sin(two_pi * (clk.phase(spec.freq) + spec.phase_deg / 360.0)
                      + beta * np.sin(two_pi * clk.phase(spec.mod_freq)))
    F = fundamental(spec)
    assert F is not None
    ph = clk.phase(F)
    if k == "damped_sine":
        tl = ph / F
        return np.sin(two_pi * spec.freq * tl + math.radians(spec.phase_deg)) * np.exp(-tl / spec.tau)
    if spec.phase_deg:
        ph = np.mod(ph + spec.phase_deg / 360.0, 1.0)
    T = 1.0 / F
    if k == "sine":
        return wf.sine(ph)
    if k == "square":
        return wf.square(ph, spec.duty)
    if k == "triangle":
        s = spec.symmetry
        if s <= 0.0:
            return 1.0 - 2.0 * ph
        if s >= 1.0:
            return 2.0 * ph - 1.0
        if s == 0.5:
            return wf.triangle(ph)
        return np.where(ph < s, -1.0 + 2.0 * ph / s, 1.0 - 2.0 * (ph - s) / (1.0 - s))
    if k == "sawtooth":
        return wf.sawtooth(ph) if spec.direction == "rising" else -wf.sawtooth(ph)
    if k == "pulse":
        return np.where(ph < spec.width * F, 1.0, 0.0)
    if k == "trapezoid":
        tot = spec.rise + spec.high + spec.fall + spec.low
        a, b, c = spec.rise / tot, (spec.rise + spec.high) / tot, (spec.rise + spec.high + spec.fall) / tot
        up = -1.0 + 2.0 * ph / max(a, 1e-300)
        down = 1.0 - 2.0 * (ph - b) / max(c - b, 1e-300)
        return np.select([ph < a, ph < b, ph < c], [up, 1.0, down], default=-1.0)
    if k == "staircase":
        n = spec.steps
        return np.minimum(np.floor(ph * n), n - 1) / (n - 1)
    if k == "half_rect_sine":
        return np.maximum(wf.sine(ph), 0.0)
    if k == "full_rect_sine":   # freq = tümsek tekrar frekansı (50 Hz şebekenin tam dalga doğrultması: 100 Hz)
        return np.abs(np.sin(np.pi * ph))
    if k == "sinc":
        return np.sinc((ph - 0.5) * T / spec.width)
    if k == "gaussian_pulse":
        sigma = spec.width / (2.0 * math.sqrt(2.0 * math.log(2.0)))
        return np.exp(-0.5 * ((ph - 0.5) * T / sigma) ** 2)
    if k == "exp_decay":
        return np.exp(-ph * T / spec.tau)
    if k == "ecg":
        return _ecg(ph, T)
    if k == "expr":
        return eval_expr(spec.expr, t=clk.t, ph=ph, f=F, fs=clk.dac_fs, n=clk.n.astype(np.float64))
    if k == "points":
        return _points_shape(np.array([q[0] for q in spec.points]), np.array([q[1] for q in spec.points]),
                             spec.interp, ph)
    if k == "harmonics":
        out = np.zeros(clk.size)
        for n, a, p in spec.harmonics:
            out += a if n == 0 else a * np.sin(two_pi * np.mod(n * ph, 1.0) + math.radians(p))
        return out
    if k == "samples":  # bir periyot kipi
        s = np.asarray(spec.samples, dtype=np.float64)
        return _points_shape(np.arange(len(s)) / len(s), s, spec.interp, ph)
    raise SignalError(f"İç hata: '{k}' için şekil tanımlı değil.")


def _render(spec: WaveSpec, clk: _Clock) -> np.ndarray:
    if spec.kind == "dc":
        return np.full(clk.size, spec.offset)
    if spec.kind == "samples" and spec.samples_mode == "play":
        s = np.resize(np.asarray(spec.samples, dtype=np.float64), clk.size)
        return spec.offset + spec.amplitude * s
    return spec.offset + spec.amplitude * _shape(spec, clk)


def synthesize(spec: WaveSpec, fs: float, n: int) -> np.ndarray:
    """`n` örnek, t = 0'dan başlayarak fs'te örneklenmiş ideal (nicemlenmemiş) değerler, float64 volt.

    Faz-süreklidir. chirp her `sweep_time`'da baştan başlar; samples ("play") döngüyle tekrarlanır; gürültü
    `seed` ile belirlenimlidir (pembe gürültü n'ye bağlı FFT ile üretildiğinden n değişince gerçekleme değişir).
    """
    fs = _num(fs, "fs")
    validate(spec, fs)
    n = int(n)
    if n < 0:
        raise SignalError("Örnek sayısı negatif olamaz.")
    return _render(spec, _Clock(n, fs))


# ============================================================================ DAC modeli
DAC_CODINGS: dict[str, str] = {
    "unsigned": "İşaretsiz (0 … 2^b−1)",
    "offset_binary": "Ofsetli ikili (çift kutuplu; sayısal olarak işaretsizle aynı)",
    "twos_complement": "İkiye tümleyen (ses kodekleri / I2S)",
}
DAC_ALIGNS: dict[str, str] = {
    "right": "Sağa hizalı",
    "left": "Sola hizalı (kelimenin üst bitleri, ör. STM32 DHR12L)",
}


@dataclass(frozen=True)
class DacSpec:
    """DAC tanımı. `out_max` None ise vref kullanılır. word_bits: C dizisinin eleman genişliği (8/16/32);
    channels = 2 ise tablo serpiştirilir (L, R, L, R …). Ayrıntılar modül başlığında."""
    bits: int = 12
    vref: float = 3.3
    out_min: float = 0.0
    out_max: float | None = None
    coding: str = "unsigned"
    align: str = "right"
    word_bits: int = 16
    channels: int = 1
    name: str = "DAC"

    def __post_init__(self) -> None:
        for nm in ("bits", "word_bits", "channels"):
            v = _num(getattr(self, nm), nm)
            if not v.is_integer():
                raise SignalError(f"'{nm}' tam sayı olmalı.")
            object.__setattr__(self, nm, int(v))
        object.__setattr__(self, "vref", _num(self.vref, "vref"))
        object.__setattr__(self, "out_min", _num(self.out_min, "out_min"))
        object.__setattr__(self, "out_max", self.vref if self.out_max is None else _num(self.out_max, "out_max"))
        if not 6 <= self.bits <= 24:
            raise SignalError("DAC bit sayısı 6 … 24 aralığında olmalı.")
        if self.word_bits not in (8, 16, 32):
            raise SignalError("Kelime genişliği 8, 16 ya da 32 bit olmalı.")
        if self.bits > self.word_bits:
            raise SignalError(f"{self.bits} bitlik kod {self.word_bits} bitlik kelimeye sığmaz.")
        if self.channels not in (1, 2):
            raise SignalError("Kanal sayısı 1 ya da 2 olmalı.")
        if self.coding not in DAC_CODINGS:
            raise SignalError(f"Geçersiz kodlama '{self.coding}'; geçerliler: {', '.join(DAC_CODINGS)}.")
        if self.align not in DAC_ALIGNS:
            raise SignalError(f"Geçersiz hizalama '{self.align}'; geçerliler: right, left.")
        assert self.out_max is not None
        if not (math.isfinite(self.out_min) and math.isfinite(self.out_max) and self.out_max > self.out_min):
            raise SignalError("DAC çıkış aralığı geçersiz: out_max > out_min olmalı.")

    @property
    def full_scale(self) -> float:
        assert self.out_max is not None
        return self.out_max - self.out_min

    @property
    def lsb(self) -> float:
        """1 LSB (V) = (out_max − out_min) / 2^bits."""
        return self.full_scale / (1 << self.bits)

    @property
    def max_level(self) -> int:
        return (1 << self.bits) - 1

    @property
    def shift(self) -> int:
        """Sola hizalama kaydırması (bit)."""
        return self.word_bits - self.bits if self.align == "left" else 0

    @property
    def v_lo(self) -> float:
        """Ulaşılabilen en düşük çıkış (kod 0)."""
        return self.out_min

    @property
    def v_hi(self) -> float:
        """Ulaşılabilen en yüksek çıkış (en üst kod) = out_max − 1 LSB."""
        return self.out_min + self.max_level * self.lsb

    @property
    def signed(self) -> bool:
        return self.coding == "twos_complement"

    @property
    def word_bytes(self) -> int:
        return self.word_bits // 8

    @property
    def c_type(self) -> str:
        """Tablo elemanının C tipi (uint8_t / uint16_t / int16_t / uint32_t …)."""
        return f"{'int' if self.signed else 'uint'}{self.word_bits}_t"

    @property
    def level_bytes(self) -> int:
        """DDS tablosundaki işaretsiz seviyenin bayt sayısı."""
        return 1 if self.bits <= 8 else 2 if self.bits <= 16 else 4

    @property
    def level_c_type(self) -> str:
        return f"uint{8 * self.level_bytes}_t"

    @property
    def code_min(self) -> int:
        return int(levels_to_codes(np.array([0]), self)[0])

    @property
    def code_max(self) -> int:
        return int(levels_to_codes(np.array([self.max_level]), self)[0])

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DacSpec:
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in dict(data).items() if k in known})


DAC_PRESETS: dict[str, DacSpec] = {
    "STM32 DAC 12 bit, sağa hizalı (0–3.3 V)": DacSpec(12, 3.3, name="STM32 DAC 12R"),
    "STM32 DAC 12 bit, sola hizalı (0–3.3 V)": DacSpec(12, 3.3, align="left", name="STM32 DAC 12L"),
    "STM32 DAC 8 bit (0–3.3 V)": DacSpec(8, 3.3, word_bits=8, name="STM32 DAC 8R"),
    "MCP4725 / MCP4921 12 bit (0–VREF)": DacSpec(12, 3.3, name="MCP4725"),
    "I2S kodek 16 bit (±1 tam ölçek)": DacSpec(16, 1.0, out_min=-1.0, out_max=1.0, coding="twos_complement",
                                             name="I2S 16"),
    "I2S kodek 24 bit / 32 bit kelime, sola hizalı (±1)": DacSpec(24, 1.0, out_min=-1.0, out_max=1.0,
                                                                  coding="twos_complement", align="left",
                                                                  word_bits=32, name="I2S 24L"),
}


def levels_to_codes(levels: np.ndarray, dac: DacSpec) -> np.ndarray:
    """İşaretsiz seviyeler (0 … 2^b − 1) → DAC kelimeleri (kodlama + hizalama), int64."""
    lv = np.asarray(levels, dtype=np.int64)
    c = lv - (1 << (dac.bits - 1)) if dac.signed else lv
    return c * (1 << dac.shift)


def codes_to_levels(codes: np.ndarray, dac: DacSpec) -> np.ndarray:
    """DAC kelimeleri → işaretsiz seviyeler (levels_to_codes'un tersi)."""
    c = np.asarray(codes, dtype=np.int64) >> dac.shift
    return c + (1 << (dac.bits - 1)) if dac.signed else c


def codes_to_volts(codes: np.ndarray, dac: DacSpec) -> np.ndarray:
    """DAC kelimelerinin ideal çıkış gerilimi (V)."""
    return dac.out_min + codes_to_levels(codes, dac) * dac.lsb


def _quantize_levels(values: np.ndarray, dac: DacSpec) -> tuple[np.ndarray, int]:
    v = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(v)):
        raise SignalError("Nicemlenecek değerlerde NaN / sonsuz var.")
    lv = np.floor((v - dac.out_min) / dac.lsb + 0.5)
    clipped = int(np.count_nonzero((lv < 0) | (lv > dac.max_level)))
    return np.clip(lv, 0, dac.max_level).astype(np.int64), clipped


def quantize(values: np.ndarray, dac: DacSpec) -> tuple[np.ndarray, np.ndarray, int]:
    """(kodlar int64, kodlardan geri hesaplanan çıkış V, kırpılan örnek sayısı).

    Yuvarlama: seviye = ⌊(v − out_min)/LSB + ½⌋ (yarım LSB yukarı). Aralık dışı örnekler en uç koda kırpılır ve
    sayılır. Kodlar `levels_to_codes` ile kodlanmış ve hizalanmıştır.
    """
    lv, clipped = _quantize_levels(values, dac)
    return levels_to_codes(lv, dac), dac.out_min + lv * dac.lsb, clipped


def shape_range(spec: WaveSpec) -> tuple[float, float]:
    """Şeklin (amplitude = 1, offset = 0) en küçük ve en büyük değeri. Gauss gürültüde ±1 (3σ) nominaldir;
    multitone'da Σ|genlik| üst sınırı kullanılır."""
    k = spec.kind
    if k == "dc":
        return 0.0, 0.0
    if k in ("sine", "square", "triangle", "sawtooth", "trapezoid", "chirp", "am", "fm", "noise_white",
             "noise_pink"):
        return -1.0, 1.0
    if k in ("pulse", "staircase", "half_rect_sine", "full_rect_sine"):
        return 0.0, 1.0
    # gaussian_pulse / exp_decay periyot içinde 0'a inmez (τ ya da genişliğe bağlı): aşağıda sayısal örneklenir
    if k == "multitone":
        s = sum(abs(t[1]) for t in spec.tones)
        return -s, s
    if k == "samples" and spec.samples_mode == "play":
        s = np.asarray(spec.samples, dtype=np.float64)
        return (float(s.min()), float(s.max())) if s.size else (0.0, 0.0)
    F = fundamental(spec)
    assert F is not None
    count = 8192
    unit = replace(spec, amplitude=1.0, offset=0.0)
    if k == "damped_sine":   # salınım periyodun içinde kalmalı: örnekleme hızını salınıma göre seç
        count = int(min(max(8192, 64 * spec.freq / F), 1 << 20))
    clk = _Clock(count, F * count, table_len=count)
    y = _shape(unit, clk)
    return float(y.min()), float(y.max())


def fit_to_dac(spec: WaveSpec, dac: DacSpec, margin: float = 0.0) -> WaveSpec:
    """amplitude ve offset'i, şekil DAC'ın ulaşılabilen aralığını [v_lo, v_hi] dolduracak biçimde ayarlar.
    `margin`: her iki uçta bırakılacak pay (tam ölçeğin oranı, 0 … 0.5)."""
    if not 0.0 <= margin < 0.5:
        raise SignalError("Pay 0 … 0.5 aralığında olmalı.")
    lo_v, hi_v = dac.v_lo, dac.v_hi
    span = (hi_v - lo_v) * (1.0 - 2.0 * margin)
    mid = 0.5 * (lo_v + hi_v)
    smin, smax = shape_range(spec)
    if smax - smin < 1e-15:
        return replace(spec, offset=mid - smin * spec.amplitude) if spec.kind != "dc" else replace(spec, offset=mid)
    amp = span / (smax - smin)
    return replace(spec, amplitude=amp, offset=mid - amp * 0.5 * (smin + smax))


# ============================================================================ tablo
@dataclass(frozen=True, eq=False)
class Table:
    """DAC'ın döngüyle çalacağı tablo.

    values: ideal değerler (V), (N,) ya da stereo (N, 2) · codes: DAC kelimeleri (kodlama + hizalama), int64 ·
    levels: işaretsiz seviyeler 0 … 2^b − 1 · volts_out: kodlardan geri hesaplanan çıkış (V) ·
    N: kanal başına örnek · periods (M): tablodaki temel periyot sayısı · f_target / f_real: istenen /
    gerçekleşen temel frekans (exact: M·fs/N; dds: W·fs/2^32; periyodik olmayanda f_real = fs/N tekrar hızı) ·
    freqs_target / freqs_real: tüm bileşenler (AM/FM: taşıyıcı, modülasyon; multitone: tonlar) ·
    tuning_word / table_bits / freq_resolution: yalnızca DDS · warnings: Türkçe uyarılar.
    """
    values: np.ndarray
    codes: np.ndarray
    levels: np.ndarray
    volts_out: np.ndarray
    N: int
    periods: int
    f_target: float
    f_real: float
    fs: float
    mode: str
    periodic: bool
    clipped: int
    warnings: tuple[str, ...]
    dac: DacSpec
    spec: WaveSpec
    spec2: WaveSpec | None = None
    freqs_target: tuple[float, ...] = ()
    freqs_real: tuple[float, ...] = ()
    tuning_word: int | None = None
    table_bits: int | None = None
    freq_resolution: float | None = None

    @property
    def channels(self) -> int:
        return 1 if self.codes.ndim == 1 else int(self.codes.shape[1])

    @property
    def freq_error_hz(self) -> float:
        return self.f_real - self.f_target

    @property
    def freq_error_ppm(self) -> float:
        return 1e6 * self.freq_error_hz / self.f_target if self.f_target else 0.0

    @property
    def max_freq_error_ppm(self) -> float:
        """Tüm bileşenlerde en büyük göreli frekans hatası (ppm)."""
        errs = [abs(r - t) / t for t, r in zip(self.freqs_target, self.freqs_real) if t]
        return 1e6 * max(errs) if errs else abs(self.freq_error_ppm)

    @property
    def samples_per_period(self) -> float:
        return self.fs / self.f_real if self.f_real else math.inf

    @property
    def memory_bytes(self) -> int:
        """C dizisinin bellek boyutu: exact → N·kanal·kelime baytı; dds → 2^k·seviye baytı."""
        if self.mode == "dds":
            return self.N * self.dac.level_bytes
        return self.N * self.channels * self.dac.word_bytes

    @property
    def flat_codes(self) -> np.ndarray:
        """C dizisindeki sırayla kodlar (stereo: serpiştirilmiş L, R, L, R …)."""
        return self.codes.reshape(-1)


def best_table_length(freqs: list[float] | tuple[float, ...], fs: float, max_len: int = 4096, *,
                      min_len: int = 2, tol_ppm: float = 0.0) -> tuple[int, tuple[int, ...]]:
    """Her bileşenin tam sayıda periyot (M_i ≥ 1, M_i < N/2) tamamladığı tablo uzunluğu N ≤ max_len.

    En büyük göreli frekans hatası max_i |M_i·fs/N − f_i| / f_i en küçük olan N seçilir (eşitlikte en kısa N;
    böylece M/N indirgenmiş kesir olur). tol_ppm > 0 ise hatası bu sınırın altındaki en kısa N seçilir.
    Dönüş: (N, (M_1, M_2, …)).
    """
    f = np.asarray(freqs, dtype=np.float64).ravel()
    fs = float(fs)
    if f.size == 0:
        raise SignalError("En az bir frekans gerekli.")
    if np.any(~np.isfinite(f)) or np.any(f <= 0) or np.any(f >= fs / 2):
        raise SignalError(f"Frekanslar 0 < f < fs/2 = {fs / 2:g} Hz aralığında olmalı.")
    lo, hi = max(int(min_len), 2), int(max_len)
    if hi > MAX_TABLE_LEN:
        raise SignalError(f"En uzun tablo en fazla {MAX_TABLE_LEN} örnek olabilir.")
    if hi < lo:
        raise SignalError(f"En uzun tablo ({hi} örnek) en az {lo} örnek olmalı.")
    ns = np.arange(lo, hi + 1, dtype=np.float64)[:, None]
    m = np.rint(ns * (f / fs))
    ok = np.all((m >= 1) & (2 * m < ns), axis=1)
    rel = np.max(np.abs(m * fs / ns - f) / f, axis=1)
    rel[~ok] = np.inf
    best = float(rel.min())
    if not math.isfinite(best):
        need = int(math.ceil(fs / f.min()))
        raise SignalError(f"En uzun tablo ({hi} örnek), en düşük frekans bileşeninin ({f.min():g} Hz) bir periyodu için "
                          f"yetmiyor: en az ≈ {need} örnek gerekir (“En uzun tablo”yu artırın ya da örnekleme "
                          "hızını düşürün).")
    thresh = max(best, tol_ppm * 1e-6) if tol_ppm > 0 else best
    i = int(np.argmax(rel <= thresh * (1.0 + 1e-9) + 1e-15))
    return int(ns[i, 0]), tuple(int(x) for x in m[i])


def _natural_length(spec: WaveSpec, fs: float, max_len: int, warns: list[str]) -> int:
    k = spec.kind
    if k == "chirp":
        n = int(round(spec.sweep_time * fs))
        if n > max_len:
            raise SignalError(f"Tarama {spec.sweep_time:g} s × fs = {n} örnek; en uzun tablo ({max_len} örnek) "
                              "aşılıyor (tarama süresini kısaltın, örnekleme hızını düşürün ya da “En uzun tablo”yu "
                              "artırın).")
        if n < 4:
            raise SignalError("Tarama süresi çok kısa (en az 4 örnek).")
        return n
    if k in ("noise_white", "noise_pink"):
        n = int(round(spec.duration * fs))
        if n > max_len:
            warns.append(f"Gürültü süresi tabloya sığmadı: {n} → {max_len} örnek ({max_len / fs:.6g} s).")
            n = max_len
        if n < 16:
            n = min(16, max_len)
            warns.append(f"Gürültü tablosu çok kısaydı; {n} örneğe uzatıldı.")
        return n
    n = len(spec.samples)
    if n > max_len:
        raise SignalError(f"Örnek dizisi ({n} örnek) en uzun tablodan ({max_len}) uzun: “En uzun tablo”yu artırın.")
    return n


def _content_warnings(spec: WaveSpec, fs: float, warns: list[str]) -> None:
    nyq = fs / 2.0
    k = spec.kind
    if k == "harmonics":
        bad = [int(n) for n, _, _ in spec.harmonics if n * spec.freq >= nyq]
        if bad:
            warns.append(f"Harmonik(ler) {bad} fs/2'nin üstünde: örtüşme (aliasing) ile başka frekanslara katlanır.")
    if k == "am" and spec.freq + spec.mod_freq >= nyq:
        warns.append("AM üst yan bandı (fc + fm) fs/2'nin üstünde: örtüşme olur.")
    if k == "fm" and spec.freq + spec.deviation + spec.mod_freq >= nyq:
        warns.append("FM bant genişliği (Carson: fc + Δf + fm) fs/2'yi aşıyor: örtüşme olur.")
    f = fundamental(spec)
    if f and fs / f < 8 and k != "sine":
        warns.append(f"Periyot başına yalnızca {fs / f:.3g} örnek: dalga şekli kaba olur (fs'yi artırın).")


def make_table(spec: WaveSpec, dac: DacSpec, fs: float, max_len: int = 4096, mode: str = "exact", *,
               min_len: int = 1, tol_ppm: float = 0.0, table_bits: int | None = None,
               spec2: WaveSpec | None = None) -> Table:
    """DAC döngü tablosu.

    mode = "exact" (DMA dairesel): N ≤ max_len ve tabloda tam M periyot; f_real = M·fs/N istenen frekansa en
      yakın (bkz. `best_table_length`). Periyodik olmayan türlerde (gürültü, chirp, samples "play") tablo
      istenen süre / örneklerdir ve `periodic=False`. dc → N = max(min_len, 1).
    mode = "dds": 2^k uzunlukta (k = table_bits ya da ⌊log2 max_len⌋) tek periyotluk tablo, 32 bit faz
      akümülatörü; tuning_word = round(f·2^32/fs), çözünürlük fs/2^32. Yalnızca tek temel frekanslı periyodik
      türler ve tek kanal.
    Stereo (dac.channels = 2): ikinci kanal `spec2` (verilmezse spec); exact modda iki kanalın tüm bileşenleri
    birlikte tam periyoda oturtulur.
    """
    fs = _num(fs, "fs")
    _positive(fs, "Örnekleme hızı fs")
    max_len = int(max_len)
    if not 1 <= max_len <= MAX_TABLE_LEN:
        raise SignalError(f"En uzun tablo 1 … {MAX_TABLE_LEN} örnek aralığında olmalı.")
    if mode not in TABLE_MODES:
        raise SignalError(f"Geçersiz tablo modu '{mode}'; geçerliler: {', '.join(TABLE_MODES)}.")
    validate(spec, fs)
    specs = [spec]
    if dac.channels == 2:
        spec2 = spec if spec2 is None else spec2
        validate(spec2, fs)
        specs.append(spec2)
    elif spec2 is not None:
        raise SignalError("İkinci kanal dalgası yalnızca 2 kanallı DAC için kullanılabilir.")
    warns: list[str] = []
    for s in specs:
        _content_warnings(s, fs, warns)

    tuning = bits = res = None
    if mode == "dds":
        if dac.channels != 1:
            raise SignalError("DDS kipi tek kanal destekler; iki kanal için “Tablo kipi”ni “Tam periyot” seçin.")
        F = fundamental(spec)
        if F is None:
            raise SignalError(f"DDS modu yalnızca tek temel frekanslı periyodik dalgalar içindir; "
                              f"'{WAVEFORMS[spec.kind]}' için mode='exact' kullanın.")
        k = int(math.floor(math.log2(max_len))) if table_bits is None else int(table_bits)
        if not 2 <= k <= 20:
            raise SignalError("DDS tablo bit sayısı 2 … 20 aralığında olmalı (en uzun tablo ≥ 4 örnek).")
        n_tab = 1 << k
        tuning, f_real, res = dds_tuning(F, fs)
        if tuning == 0:
            raise SignalError(f"Frekans DDS çözünürlüğünün ({fs / 2 ** ACC_BITS:.3g} Hz) altında.")
        clk = _Clock(n_tab, f_real * n_tab, table_len=n_tab, dac_fs=fs)
        values = _render(_with_fundamental(spec, f_real), clk)
        bits, n, m, f_target, periodic = k, n_tab, 1, F, True
        freqs_t, freqs_r = (F,), (f_real,)
        if fs / F < 8:
            warns.append(f"Periyot başına {fs / F:.3g} örnek: DDS faz kesme / titreşim bozulması belirgin olur.")
    else:
        comps = [_components(s) for s in specs]
        if all(c is not None for c in comps):
            allf = [f for c in comps for f in (c or [])]
            if allf:
                n, cycles = best_table_length(allf, fs, max_len, min_len=max(min_len, 2), tol_ppm=tol_ppm)
                real = [mm * fs / n for mm in cycles]
            else:
                n, cycles, real = max(int(min_len), 1), (), []
            chans, pos = [], 0
            for s, c in zip(specs, comps):
                assert c is not None
                r = real[pos:pos + len(c)]
                pos += len(c)
                v = _render(_apply_components(s, r) if c else s, _Clock(n + 1, fs, table_len=n))
                jump = abs(v[n] - v[0])
                if jump > 1e-9 * max(1.0, float(np.max(np.abs(v)))):
                    hint = ("; formülde mutlak zaman t yerine ph ya da f*t kullanın" if s.kind == "expr" else "")
                    warns.append(f"Tablonun sonu ile başı arasında süreksizlik var (Δ = {jump:.3g} V){hint}.")
                chans.append(v[:n])
            c1 = comps[0] or []
            if c1:
                f_target, f_real, m = c1[0], real[0], cycles[0]
            else:
                f_target = f_real = 0.0
                m = 0
            freqs_t, freqs_r, periodic = tuple(allf), tuple(real), True
            values = chans[0] if len(chans) == 1 else np.column_stack(chans)
            err = max((abs(r - t) / t for t, r in zip(allf, real)), default=0.0)
            if err > 100e-6:
                warns.append(f"Frekans hatası yüksek ({err * 1e6:.0f} ppm): “En uzun tablo”yu artırın ya da örnekleme "
                             "hızını ayarlayın.")
        elif not any(c is not None for c in comps):
            n = _natural_length(spec, fs, max_len, warns)
            chans = [synthesize(s, fs, n) for s in specs]
            values = chans[0] if len(chans) == 1 else np.column_stack(chans)
            m, periodic = 1, False
            f_target = f_real = fs / n
            freqs_t = freqs_r = ()
        else:
            raise SignalError("Stereo tabloda iki kanal da periyodik ya da ikisi de periyodik olmayan türde olmalı.")

    levels, clipped = _quantize_levels(values, dac)
    codes = levels_to_codes(levels, dac)
    volts = dac.out_min + levels * dac.lsb
    if clipped:
        warns.append(f"{clipped} örnek DAC aralığı ({dac.v_lo:.4g} … {dac.v_hi:.4g} V) dışında kaldı ve kırpıldı.")
    return Table(values=np.asarray(values, dtype=np.float64), codes=codes, levels=levels, volts_out=volts,
                 N=int(n), periods=int(m), f_target=float(f_target), f_real=float(f_real), fs=fs, mode=mode,
                 periodic=periodic, clipped=clipped, warnings=tuple(warns), dac=dac, spec=spec,
                 spec2=spec2 if dac.channels == 2 else None, freqs_target=tuple(freqs_t),
                 freqs_real=tuple(freqs_r), tuning_word=tuning, table_bits=bits, freq_resolution=res)


# ============================================================================ DDS modeli
def dds_tuning(freq: float, fs: float) -> tuple[int, float, float]:
    """(ayar kelimesi W, gerçekleşen frekans W·fs/2^32, çözünürlük fs/2^32).

    W = ⌊f · (2^32/fs) + ½⌋, f önce [0, fs/2] aralığına kırpılır. Üretilen C kodundaki `*_dds_set_freq` aynı
    işlemleri aynı sırayla (double) yapar: sonuç bit düzeyinde aynıdır.
    """
    fs = float(fs)
    scale = float(2 ** ACC_BITS) / fs
    f = float(freq)
    f = 0.0 if not f > 0.0 else min(f, fs / 2.0)      # C: if (!(hz > 0.0)) hz = 0.0; (NaN → 0)
    w = int(math.floor(f * scale + 0.5))
    return w, w * fs / 2 ** ACC_BITS, fs / 2 ** ACC_BITS


def dds_frac_bits(table_bits: int) -> int:
    """Doğrusal ara değerlemede kullanılan kesir biti: min(16, 32 − k)."""
    return min(16, ACC_BITS - int(table_bits))


def dds_sequence(table: Table, n: int, *, step: int | None = None, interpolate: bool = False,
                 phase: int = 0) -> np.ndarray:
    """DDS çıkışının bit düzeyinde modeli (üretilen C kodu ile birebir): n DAC kelimesi (int64).

    acc_i = (phase + i·W) mod 2^32; indeks = acc >> (32 − k). Ara değerlemede F = min(16, 32 − k) kesir biti:
    seviye = (a·(2^F − kesir) + b·kesir + 2^(F−1)) >> F, b = sonraki tablo elemanı (sarmalı).
    """
    if table.mode != "dds" or table.table_bits is None:
        raise SignalError("dds_sequence yalnızca mode='dds' tablolar içindir.")
    k = table.table_bits
    size = 1 << k
    w = table.tuning_word if step is None else int(step)
    assert w is not None
    mask = np.uint64((1 << ACC_BITS) - 1)
    acc = (np.uint64(int(phase) & int(mask)) + np.arange(int(n), dtype=np.uint64) * np.uint64(w & int(mask))) & mask
    idx = (acc >> np.uint64(ACC_BITS - k)).astype(np.int64)
    lv = np.asarray(table.levels, dtype=np.int64).reshape(-1)
    out = lv[idx]
    if interpolate:
        fb = dds_frac_bits(k)
        frac = ((acc >> np.uint64(ACC_BITS - k - fb)) & np.uint64((1 << fb) - 1)).astype(np.int64)
        nxt = lv[(idx + 1) & (size - 1)]
        out = (out * ((1 << fb) - frac) + nxt * frac + (1 << (fb - 1))) >> fb
    return levels_to_codes(out, table.dac)


# ============================================================================ ölçümler
@dataclass(frozen=True)
class Metrics:
    """Tablonun (DDS'te benzetilen çıkış dizisinin) ölçümleri. Uygulanamayan değerler NaN.

    Zaman alanı (V): vmin, vmax, vpp, mean (DC), rms, ac_rms, crest_factor (= max|x − ort| / AC RMS).
    Spektrum (tek ton tanımları, temel bileşene göre): thd_pct / thd_db (n_harmonics harmonik, örtüşmüş
    konumlarıyla), sfdr_dbc, snr_db (harmonikler hariç), sinad_db, enob = (SINAD − 1.76)/6.02; harmonics: ilk
    10 harmonik (k, görünen frekans Hz, dBc). Çok bileşenli (multitone / AM / FM), gürültü ve tarama
    sinyallerinde bunlar NaN'dır. quant_snr_db: dalga şeklinden bağımsız nicemleme SNR'si
    = 10·log10(var(ideal) / ort((nicemli − ideal)²)), tablo değerleri üzerinden (kırpma dahil).
    max_step_v / max_slew_v_per_s: ardışık örnekler arası en büyük sıçrama ve bunun fs ile çarpımı (döngü sarması
    dahil). image_hz = fs − f_real; content_max_hz: content_db eşiğini aşan en yüksek bileşen; first_image_hz =
    fs − content_max_hz; filter_cutoff_hz = √(content_max · first_image) (geometrik orta), filter_order:
    ilk görüntüyü image_atten_db kadar bastıran Butterworth derecesi (tahmin); zoh_droop_db: sıfır dereceli
    tutmanın content_max'taki sinc düşüşü.
    """
    vmin: float
    vmax: float
    vpp: float
    mean: float
    rms: float
    ac_rms: float
    crest_factor: float
    f_real: float
    fs: float
    update_interval: float
    thd_pct: float
    thd_db: float
    sfdr_dbc: float
    snr_db: float
    sinad_db: float
    enob: float
    quant_snr_db: float
    harmonics: tuple[tuple[int, float, float], ...]
    max_step_v: float
    max_slew_v_per_s: float
    image_hz: float
    content_max_hz: float
    first_image_hz: float
    filter_cutoff_hz: float
    filter_order: int | None
    image_atten_db: float
    zoh_droop_db: float
    memory_bytes: int
    lsb_v: float
    clipped: int
    analysis: str
    notes: tuple[str, ...]


@dataclass
class _Spec:
    freqs: np.ndarray
    power: np.ndarray            # tek taraflı güç / kutu (V²)
    fund: float = math.nan
    harm_total: float = math.nan
    noise: float = math.nan
    spur_ratio: float = math.nan  # temel tepe / en büyük istenmeyen tepe
    harmonics: tuple[tuple[int, float, float], ...] = ()


def _one_sided(X: np.ndarray, n: int, scale: float) -> np.ndarray:
    p = np.abs(X) ** 2 * scale
    p[1:] *= 2.0
    if n % 2 == 0:
        p[-1] /= 2.0
    return p


def _db(x: float) -> float:
    return 10.0 * math.log10(x) if x > 0 else (-math.inf if x == 0 else math.nan)


def _analyze_periodic(x: np.ndarray, fs: float, m: int, n_harm: int | None) -> _Spec:
    """Tam periyot içeren tablo: pencere yok, her bileşen tam bir kutuya düşer (sızıntı yok)."""
    n = len(x)
    p = _one_sided(np.fft.rfft(x), n, 1.0 / n ** 2)
    freqs = np.arange(len(p)) * fs / n
    out = _Spec(freqs, p)
    if not 1 <= m < n / 2:
        return out
    fund = float(p[m])
    kmax = (n // 2) // m if n_harm is None else n_harm + 1
    ks = np.arange(2, max(kmax, 1) + 1)
    b = (ks * m) % n
    b = np.where(b > n // 2, n - b, b)
    uniq, first = np.unique(b, return_index=True)
    keep = (uniq != 0) & (uniq != m)
    harm_total = float(p[uniq[keep]].sum())
    ac = float(p[1:].sum())
    mask = np.ones(len(p), dtype=bool)
    mask[0] = mask[m] = False
    spur = float(p[mask].max()) if mask.any() else 0.0
    shown = []
    for k in range(2, 12):
        bk = (k * m) % n
        bk = n - bk if bk > n // 2 else bk
        if bk not in (0, m):
            shown.append((k, float(freqs[bk]), _db(float(p[bk]) / fund) if fund > 0 else math.nan))
    out.fund, out.harm_total = fund, harm_total
    out.noise = max(ac - fund - harm_total, 0.0)
    out.spur_ratio = fund / spur if spur > 0 else math.inf
    out.harmonics = tuple(shown)
    return out


_BH_HALF = 4  # 4 terimli Blackman-Harris ana lob yarı genişliği (kutu)


_KAISER_BETA, _KAISER_HALF = 38.0, 13  # yan loblar ≈ −300 dB; ana lob yarı genişliği ≈ √(1+(β/π)²) ≈ 12.2 kutu


def _analyze_windowed(x: np.ndarray, fs: float, f0: float, n_harm: int | None, window: str = "bh") -> _Spec:
    """Periyodik olmayan / DDS dizisi: pencereli FFT; bileşen gücü ana lob kutularının toplamı, gürültü kalan
    kutulardan (dışlanan kutular ortalama yoğunlukla tamamlanır). window="kaiser": β = 38 Kaiser — DDS'te 16–24
    bit DAC'ın nicemleme tabanını ölçebilmek için (4 terimli Blackman-Harris'in −92 dB yan lobu tavan oluşturur)."""
    n = len(x)
    if window == "kaiser":
        w = sig_windows.kaiser(n, _KAISER_BETA, sym=False)
        _BH_HALF = _KAISER_HALF  # noqa: N806 — aşağıdaki kutu genişliği hesapları bu pencereye göre
    else:
        w = sig_windows.blackmanharris(n, sym=False)
        _BH_HALF = 4  # noqa: N806
    p = _one_sided(np.fft.rfft((x - x.mean()) * w), n, 1.0 / (n * float(np.sum(w * w))))
    freqs = np.arange(len(p)) * fs / n
    out = _Spec(freqs, p)
    nb = len(p)
    if not (math.isfinite(f0) and 0 < f0 < fs / 2) or nb < 4 * _BH_HALF:
        return out
    used = np.zeros(nb, dtype=bool)
    used[:_BH_HALF + 1] = True
    n_dc = int(used.sum())

    def peak(f: float) -> int:
        b = int(round(f * n / fs))
        lo, hi = max(b - 2, 1), min(b + 3, nb)
        return lo + int(np.argmax(p[lo:hi])) if hi > lo else min(max(b, 0), nb - 1)

    def band(b: int) -> slice:
        return slice(max(b - _BH_HALF, 0), min(b + _BH_HALF + 1, nb))

    kf = peak(f0)
    fb = band(kf)
    fund_peak = float(p[kf])
    fund_bins = used.copy()
    fund_bins[:] = False
    fund_bins[fb] = True
    used[fb] = True
    kmax = min(int((fs / 2) // f0), 1000) if n_harm is None else n_harm + 1
    harm_sums: list[tuple[int, float, float, int]] = []
    for k in range(2, kmax + 1):
        fk = (k * f0) % fs
        fk = fs - fk if fk > fs / 2 else fk
        hb = peak(fk)
        sl = band(hb)
        if used[sl].any():
            continue
        harm_sums.append((k, float(freqs[hb]), float(p[sl].sum()), sl.stop - sl.start))
        used[sl] = True
    noise_mask = ~used
    noise_sum = float(p[noise_mask].sum())
    noise_mean = noise_sum / max(int(noise_mask.sum()), 1)
    n_sig = int(used.sum()) - n_dc
    fund = max(float(p[fb].sum()) - noise_mean * (fb.stop - fb.start), 0.0)
    harm_total = sum(max(s - noise_mean * nbins, 0.0) for _, _, s, nbins in harm_sums)
    spur_mask = ~fund_bins
    spur_mask[:_BH_HALF + 1] = False
    spur = float(p[spur_mask].max()) if spur_mask.any() else 0.0
    out.fund, out.harm_total = fund, harm_total
    out.noise = noise_sum + noise_mean * n_sig
    out.spur_ratio = fund_peak / spur if spur > 0 else math.inf
    out.harmonics = tuple((k, f, _db(max(s - noise_mean * nbins, 0.0) / fund) if fund > 0 else math.nan)
                          for k, f, s, nbins in harm_sums if k <= 11)
    return out


def metrics(table: Table, n_harmonics: int | None = 10, *, channel: int = 0, image_atten_db: float = 40.0,
            content_db: float = -40.0, dds_len: int = 32768, dds_interpolate: bool = False) -> Metrics:
    """Tablonun DAC çıkışı (nicemlenmiş) üzerinden ölçümler.

    exact + periyodik: tek periyotluk tablo üzerinde pencere kullanmadan tam FFT (tam sayıda periyot ⇒ sızıntı
    yok). Periyodik olmayan tablolar ve DDS (benzetilen `dds_len` örnek) Blackman-Harris pencereli.
    n_harmonics = None: örtüşmesiz tüm harmonikler (k·f < fs/2) THD'ye katılır. content_db: filtre önerisi için
    "anlamlı içerik" eşiği (temel bileşene göre dBc).
    """
    dac, fs = table.dac, table.fs
    notes: list[str] = []
    if not 0 <= channel < table.channels:
        raise SignalError(f"Kanal 0 … {table.channels - 1} olmalı.")
    kind = table.spec.kind if channel == 0 or table.spec2 is None else table.spec2.kind
    f_ch, m_ch = table.f_real, table.periods
    if channel == 1 and table.spec2 is not None and table.mode == "exact" and table.periodic:
        n1 = len(_components(table.spec) or [])
        c2 = _components(table.spec2) or []
        f_ch = table.freqs_real[n1] if c2 else 0.0
        m_ch = int(round(f_ch * table.N / fs))
    if table.mode == "dds":
        x = codes_to_volts(dds_sequence(table, dds_len, interpolate=dds_interpolate), dac)
        wrap = False
    else:
        vo = table.volts_out
        x = vo if vo.ndim == 1 else vo[:, channel]
        wrap = True
    x = np.asarray(x, dtype=np.float64)
    mean = float(x.mean())
    ac = float(x.std())
    peak_dev = float(np.max(np.abs(x - mean))) if x.size else 0.0
    steps = np.abs(np.diff(np.concatenate((x, x[:1])) if wrap else x))
    max_step = float(steps.max()) if steps.size else 0.0

    f0 = math.nan
    if ac <= 0 or x.size < 4:
        sp = _Spec(np.zeros(1), np.zeros(1))
        analysis = "sabit sinyal (spektrum yok)"
    elif table.mode == "exact" and table.periodic and m_ch >= 1:
        f0 = f_ch
        sp = _analyze_periodic(x, fs, m_ch, n_harmonics)
        analysis = "tam periyot, pencere yok (kesin FFT)"
    else:
        if table.mode == "dds":
            f0 = table.f_real
            analysis = f"DDS çıkışı ({dds_len} örnek benzetim), Kaiser (β = {_KAISER_BETA:g}) pencereli"
        else:
            analysis = "periyodik olmayan tablo, Blackman-Harris pencereli"
            if kind == "samples":
                sp0 = _analyze_windowed(x, fs, math.nan, None)
                if sp0.power.size > _BH_HALF + 2:
                    i = _BH_HALF + 1 + int(np.argmax(sp0.power[_BH_HALF + 1:]))
                    f0 = float(sp0.freqs[i])
            else:
                notes.append("Gürültü / tarama sinyalinde THD, SNR, SFDR tanımlı değil.")
        sp = _analyze_windowed(x, fs, f0, n_harmonics, "kaiser" if table.mode == "dds" else "bh")
    multi = kind in _MULTI_KINDS
    if multi:
        notes.append("Çok bileşenli sinyalde (multitone / AM / FM) tek ton ölçümleri (THD, SNR, SINAD, ENOB, "
                     "SFDR) tanımsız; nicemleme kalitesi için quant_snr_db'ye bakın.")
    elif kind != "sine" and math.isfinite(f0):
        notes.append("THD / SINAD / ENOB tek sinüs içindir; bu dalga şeklinde harmonikler sinyalin parçasıdır. "
                     "Nicemleme kalitesi için quant_snr_db'ye bakın.")
    fund = math.nan if multi else sp.fund
    has_f = math.isfinite(fund) and fund > 0
    thd_r = sp.harm_total / fund if has_f else math.nan
    thd_pct = 100.0 * math.sqrt(thd_r) if has_f else math.nan
    thd_db = _db(thd_r) if has_f else math.nan
    snr = _db(fund / sp.noise) if has_f and sp.noise > 0 else (math.inf if has_f else math.nan)
    sinad_den = sp.noise + sp.harm_total if has_f else math.nan
    sinad = _db(fund / sinad_den) if has_f and sinad_den > 0 else (math.inf if has_f else math.nan)
    enob = (sinad - 1.76) / 6.02 if math.isfinite(sinad) else (math.inf if has_f else math.nan)
    sfdr = _db(sp.spur_ratio) if has_f else math.nan

    # anlamlı içerik ve yeniden yapılandırma filtresi önerisi
    p, fr = sp.power, sp.freqs
    content = 0.0
    if p.size > 1 and np.any(p[1:] > 0):
        ref = fund if has_f else float(p[1:].max())
        sig = (p >= ref * 10 ** (content_db / 10.0))
        sig[0] = False
        if table.mode != "exact" or not table.periodic:
            sig[:_BH_HALF + 1] = False
        content = float(fr[sig].max()) if sig.any() else (f0 if math.isfinite(f0) else 0.0)
    first_img = fs - content if content > 0 else math.nan
    cutoff, order = math.nan, None
    if content > 0 and first_img > 1.2 * content:
        cutoff = math.sqrt(content * first_img)
        need = math.log10(10 ** (image_atten_db / 10.0) - 1.0) / (2.0 * math.log10(first_img / cutoff))
        order = max(1, int(math.ceil(need - 1e-9)))
    elif content > 0:
        notes.append("İçerik fs/2'ye kadar uzanıyor (keskin kenarlar / gürültü): görüntüler içerikle örtüşür; "
                     "fs'yi artırın ya da filtre köşesini korunacak en yüksek harmoniğe göre seçin.")
    droop = 20.0 * math.log10(abs(float(np.sinc(content / fs)))) if 0 < content < fs else math.nan
    if table.clipped:
        notes.append(f"{table.clipped} örnek kırpıldı: ölçümler kırpılmış çıkışa aittir.")
    if table.mode == "dds":
        notes.append(f"DDS faz kesme spurları ≈ −{6.02 * (table.table_bits or 0):.0f} dBc beklenir "
                     "(ara değerleme ile belirgin biçimde daha düşük).")
    image = fs - f0 if math.isfinite(f0) and f0 > 0 else math.nan
    ideal = table.values if table.values.ndim == 1 else table.values[:, channel]
    quant = table.volts_out if table.volts_out.ndim == 1 else table.volts_out[:, channel]
    err_p = float(np.mean((quant - ideal) ** 2)) if ideal.size else 0.0
    sig_p = float(np.var(ideal)) if ideal.size else 0.0
    quant_snr = _db(sig_p / err_p) if err_p > 0 and sig_p > 0 else (math.inf if sig_p > 0 else math.nan)
    return Metrics(
        vmin=float(x.min()), vmax=float(x.max()), vpp=float(x.max() - x.min()), mean=mean,
        rms=float(math.sqrt(np.mean(x * x))), ac_rms=ac, crest_factor=peak_dev / ac if ac > 0 else math.nan,
        f_real=f_ch, fs=fs, update_interval=1.0 / fs,
        thd_pct=thd_pct, thd_db=thd_db, sfdr_dbc=sfdr, snr_db=snr, sinad_db=sinad, enob=enob,
        quant_snr_db=quant_snr, harmonics=() if multi else sp.harmonics, max_step_v=max_step, max_slew_v_per_s=max_step * fs,
        image_hz=image, content_max_hz=content, first_image_hz=first_img, filter_cutoff_hz=cutoff,
        filter_order=order, image_atten_db=float(image_atten_db), zoh_droop_db=droop,
        memory_bytes=table.memory_bytes, lsb_v=dac.lsb, clipped=table.clipped, analysis=analysis,
        notes=tuple(notes))


# ============================================================================ zamanlayıcı
@dataclass(frozen=True)
class TimerConfig:
    """STM32 tarzı zamanlayıcı: güncelleme hızı = clk / ((PSC + 1)·(ARR + 1))."""
    psc: int
    arr: int
    timer_clock_hz: float
    fs_target: float
    fs_real: float
    error_hz: float
    error_ppm: float
    bits: int
    psc_bits: int

    @property
    def divisor(self) -> int:
        return (self.psc + 1) * (self.arr + 1)


def timer_config(fs_target: float, timer_clock_hz: float, bits: int = 16, *, psc_bits: int = 16) -> TimerConfig:
    """Frekans hatasını en küçükleyen (PSC, ARR) çifti; eşitlikte en küçük PSC.

    bits: ARR genişliği (16 ya da 32 bit zamanlayıcı). psc_bits: PSC genişliği — STM32'de 32 bit zamanlayıcılarda
    da (TIM2/TIM5) PSC 16 bittir, bu yüzden varsayılan 16. 16 bit zamanlayıcıda PSC, ARR ≤ 65535. ARR ≥ 1
    zorunludur (STM32'de ARR = 0 sayacı durdurur), bu yüzden fs ≤ clk/2.
    """
    fs = _num(fs_target, "fs_target")
    clk = _num(timer_clock_hz, "timer_clock_hz")
    _positive(fs, "Hedef örnekleme hızı")
    _positive(clk, "Zamanlayıcı saati")
    if bits not in (16, 32) or psc_bits not in (16, 32):
        raise SignalError("Zamanlayıcı genişliği 16 ya da 32 bit olmalı.")
    if fs > clk / 2:
        raise SignalError(f"Örnekleme hızı en fazla zamanlayıcı saatinin yarısı ({clk / 2:g} Hz) olabilir (ARR ≥ 1).")
    pmax = 1 << 16   # PSC araması 2^16 adayla sınırlı (32 bit PSC yalnızca çok düşük fs için aşağıda)
    amax = float(1 << bits)
    d = clk / fs
    lowest = clk / (float(1 << psc_bits) * amax)
    if fs < lowest * (1 - 1e-12):
        raise SignalError(f"Örnekleme hızı bu zamanlayıcı için çok düşük (en az {lowest:.6g} Hz); 32 bit zamanlayıcı "
                          "kullanın ya da zamanlayıcı saatini düşürün.")
    pp = np.arange(1, pmax + 1, dtype=np.float64)
    q = d / pp
    cand = np.clip(np.stack((np.floor(q), np.ceil(q)), axis=1), 2.0, amax)
    err = np.abs(clk / (pp[:, None] * cand) - fs)
    j = np.argmin(err, axis=1)
    e = err[np.arange(len(pp)), j]
    best = float(e.min())
    i = int(np.argmax(e <= best + max(best, fs) * 1e-12))
    psc, arr = int(pp[i]) - 1, int(cand[i, j[i]]) - 1
    if psc_bits == 32 and d / amax > pmax:
        # çok düşük fs (yalnızca 32 bit PSC'de): PSC'yi doğrudan hesapla
        psc = int(math.ceil(d / amax)) - 1
        arr = min(max(int(round(d / (psc + 1))), 2), int(amax)) - 1
    f_real = clk / ((psc + 1) * (arr + 1))
    return TimerConfig(psc=psc, arr=arr, timer_clock_hz=clk, fs_target=fs, fs_real=f_real,
                       error_hz=f_real - fs, error_ppm=1e6 * (f_real - fs) / fs, bits=bits, psc_bits=psc_bits)


# ============================================================================ CSV örnek yükleme
def _to_float(token: str, decimal: str) -> float:
    s = token.strip().strip('"').strip("'").replace("−", "-")
    if decimal == ",":
        s = s.replace(",", ".")
    return float(s)


def parse_samples_text(text: str, column: int | str | None = None, *, delimiter: str | None = None,
                       decimal: str | None = None) -> np.ndarray:
    """CSV / düz metinden örnek dizisi. Ayırıcı otomatik: ';' (Türkçe Excel, ondalık virgül), sekme, ',' ya da
    boşluk. Başlık satırı (sayı olmayan ilk satır) varsa sütun adıyla seçilebilir. column: 0 tabanlı indis ya da
    başlık adı; None → tek sütunda o sütun, çok sütunda SON sütun (tipik "zaman;değer"). '#' ile başlayan satırlar
    yorum sayılır."""
    lines = [(i + 1, ln.strip()) for i, ln in enumerate(text.splitlines())]
    lines = [(i, ln) for i, ln in lines if ln and not ln.startswith("#")]
    if not lines:
        raise SignalError("Dosyada veri yok.")
    sample = [ln for _, ln in lines[:50]]
    if delimiter is None:
        body = [ln for ln in sample if any(ch.isdigit() for ch in ln)]  # başlık / metin satırları hariç
        if any(";" in ln for ln in sample):
            delimiter = ";"
        elif any("\t" in ln for ln in sample):
            delimiter = "\t"
        elif any(len(ln.split()) > 1 for ln in body):
            delimiter = " "                       # boşluk ayrılmış (ondalık virgül olabilir: "0,5 1,2")
        elif any("," in ln for ln in sample) and decimal != ",":
            if (decimal is None and body and not any("." in ln for ln in body)
                    and all(ln.count(",") == 1 for ln in body)):
                # Türkçe Excel tek sütunu ";" yazmaz: her satırda tek virgül ve hiç nokta yok → ondalık virgül
                delimiter, decimal = " ", ","
            else:
                delimiter = ","
        else:
            delimiter = " "
    if decimal is None:
        decimal = "," if delimiter != "," and any("," in ln for ln in sample) else "."

    def split(ln: str) -> list[str]:
        cells = ln.split() if delimiter == " " else [c.strip() for c in ln.split(delimiter)]
        while len(cells) > 1 and cells[-1] == "":       # satır sonundaki fazla ayırıcılar
            cells.pop()
        return cells

    header: list[str] | None = None
    first_no, first = lines[0]
    try:
        [_to_float(c, decimal) for c in split(first) if c != ""]
    except ValueError:
        header = [c.strip().strip('"') for c in split(first)]
        lines = lines[1:]
    if not lines:
        raise SignalError("Başlıktan sonra veri yok.")
    ncols = len(split(lines[0][1]))
    if isinstance(column, str):
        names = [h.lower() for h in (header or [])]
        if column.lower() not in names:
            raise SignalError(f"'{column}' adlı sütun bulunamadı (başlık: {header}).")
        col = names.index(column.lower())
    elif column is None:
        col = ncols - 1
    else:
        col = int(column)
    out = np.empty(len(lines))
    for j, (no, ln) in enumerate(lines):
        cells = split(ln)
        if col >= len(cells) or col < -len(cells):
            raise SignalError(f"{no}. satırda {col + 1}. sütun yok.")
        try:
            out[j] = _to_float(cells[col], decimal)
        except ValueError:
            raise SignalError(f"{no}. satır, {col + 1}. sütun sayı değil: '{cells[col]}'.") from None
    if not np.all(np.isfinite(out)):
        raise SignalError("Örneklerde NaN / sonsuz değer var.")
    return out


def load_samples_csv(path: str | Path, column: int | str | None = None, *, delimiter: str | None = None,
                     decimal: str | None = None) -> np.ndarray:
    """CSV dosyasından örnek dizisi (bkz. `parse_samples_text`). Kodlama: UTF-8 (BOM'lu da), yoksa cp1254."""
    raw = Path(path).read_bytes()
    for enc in ("utf-8-sig", "cp1254", "latin-1"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    return parse_samples_text(text, column, delimiter=delimiter, decimal=decimal)
