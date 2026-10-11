"""Filtre tasarım motoru: FIR / IIR / özel filtreler, cevap analizi ve blok (durum taşıyan) uygulama.

- IIR filtreler ikinci derece bölümler (SOS) olarak tasarlanır ve uygulanır. 12. derece Eliptik gibi yüksek
  dereceli tasarımlar bu yapıda kararlıdır; tek polinom (b, a) biçiminde katsayılar sayısal olarak bozulur.
- FIR filtreler katsayı dizisi (b) olarak tutulur; uzun filtreler FFT ile (örtüşmeli konvolüsyon) uygulanır.
- Medyan filtre doğrusal değildir: frekans cevabı ve katsayısı yoktur, yalnızca uygulanır.
- Canlı uygulamada her kanal kendi durumunu (zi / geçmiş örnekler) taşır; sinyali bloklar halinde filtrelemek,
  tek seferde filtrelemekle aynı sonucu verir. Başlangıç durumu ilk örneğe göre kararlı hâldir (adım cevabının
  oturmuş hâli), böylece DC içeren sinyallerde açılışta sıçrama olmaz.

Tüm frekanslar Hz cinsindendir; tasarımlar örnekleme hızıyla (fs) birlikte yapılır.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import asdict, dataclass, fields
from functools import cached_property

import numpy as np
from scipy import ndimage, signal

IIR_METHODS = {
    "butter": "Butterworth",
    "cheby1": "Chebyshev I",
    "cheby2": "Chebyshev II",
    "ellip": "Eliptik",
    "bessel": "Bessel",
}
FIR_WINDOWS = {"kaiser": "Kaiser", "hamming": "Hamming", "hann": "Hann", "blackman": "Blackman",
               "boxcar": "Dikdörtgen"}
FIR_METHODS = {
    **{key: f"Pencere ({name})" for key, name in FIR_WINDOWS.items()},
    "remez": "Parks-McClellan (eşdalgalı)",
    "firls": "En küçük kareler",
}
SPECIAL_METHODS = {
    "notch": "Notch (çentik)",
    "comb": "Tarak (comb)",
    "dc_block": "DC engelleyici",
    "ema": "Üstel ortalama (EMA)",
    "kalman": "Kalman (kararlı hâl)",
    "alpha_beta": "Alfa-beta izleyici",
    "moving_avg": "Hareketli ortalama",
    "gaussian": "Gauss (FIR)",
    "median": "Medyan",
    "savgol": "Savitzky-Golay",
    "differentiator": "Türev alıcı (FIR)",
    "integrator": "Sızdıran integratör",
    "hilbert": "Hilbert dönüştürücü (FIR)",
    "allpass": "Tüm geçiren (faz)",
}
KALMAN_MODELS = {"cv": "Sabit hız (konum + hız)", "rw": "Sabit seviye (rastgele yürüyüş)"}
COMB_TYPES = {"notch": "Çentik (f₀ ve katlarını bastır)", "peak": "Tepe (yalnız f₀ ve katlarını geçir)"}
FAMILIES = {"iir": "IIR", "fir": "FIR", "special": "Özel"}
METHODS = {"iir": IIR_METHODS, "fir": FIR_METHODS, "special": SPECIAL_METHODS}
BANDS = {"lowpass": "Alçak geçiren", "highpass": "Yüksek geçiren", "bandpass": "Bant geçiren",
         "bandstop": "Bant durduran"}
TWO_EDGE = ("bandpass", "bandstop")
# Faz / türev / integral için tasarlanan filtreler: sıfır fazlı (ileri + geri) uygulama amaçlarını bozar
PHASE_DEFINED = ("differentiator", "hilbert", "allpass", "integrator")

MAX_IIR_ORDER = 40        # prototip derecesi; bant geçiren / durduranda gerçek derece bunun 2 katıdır
MAX_FIR_TAPS = 16_385
MAX_LS_TAPS = 4_097       # Parks-McClellan / en küçük kareler: süre ~N² (4001 katsayıda ~1 s), bellek ~N²
MAX_LENGTH = 100_001      # hareketli ortalama / medyan / Savitzky-Golay pencere boyu
MAX_ROOT_TAPS = 256       # bundan uzun FIR'de sıfırlar (polinom kökleri) hesaplanmaz (401 katsayıda ~0.2 s)
FIR_FFT_POINTS = 1 << 15  # FIR cevabı FFT ile bu kadar (+1) noktada hesaplanır
MASK_TOL_DB = 0.01

# Pencere yöntemi: geçiş bandı genişliği ≈ K·fs/N ve pencerenin sağlayabildiği yaklaşık en yüksek bastırma
_WINDOW_WIDTH = {"hamming": 3.3, "hann": 3.1, "blackman": 5.5, "boxcar": 0.92}
_WINDOW_ATTEN = {"hamming": 53.0, "hann": 44.0, "blackman": 74.0, "boxcar": 21.0}


class FilterError(ValueError):
    """Kullanıcıya gösterilecek tasarım hatası (geçersiz parametre, karşılanamayan spesifikasyon…)."""


def format_hz(hz: float) -> str:
    if abs(hz) >= 1e6:
        return f"{hz / 1e6:.6g} MHz"
    if abs(hz) >= 1e3:
        return f"{hz / 1e3:.6g} kHz"
    return f"{hz:.6g} Hz"


@dataclass(frozen=True)
class FilterSpec:
    """Bir filtrenin tam tanımı. Alanların hangisinin kullanıldığı aileye ve yönteme bağlıdır.

    f1 / f2 anlamı:
      - Derece elle verildiğinde: Butterworth / Bessel'de −3 dB kesim, Chebyshev I / Eliptik'te geçirme bandı
        kenarı, Chebyshev II'de durdurma bandının başladığı frekans (scipy / MATLAB tanımı), pencere FIR'de −6 dB
        kesim (geçiş bandının ortası).
      - Derece otomatikken ve Parks-McClellan / en küçük karelerde: geçirme bandı kenarları. Durdurma bandı
        `transition` (Δf) kadar dışarıda başlar (bant durduranda içeride).
    """
    family: str = "iir"
    method: str = "butter"
    band: str = "lowpass"
    fs: float = 10_000.0
    order: int = 4
    auto_order: bool = False
    f1: float = 1000.0
    f2: float = 2000.0
    transition: float = 200.0
    ripple_db: float = 1.0
    atten_db: float = 60.0
    kaiser_beta: float = 8.6
    notch_f0: float = 50.0
    notch_q: float = 30.0
    harmonics: int = 1
    dc_cutoff: float = 1.0
    length: int = 11
    polyorder: int = 3
    comb_type: str = "notch"
    kalman_model: str = "cv"
    kalman_q: float = 10.0      # süreç gürültüsü σ: sabit hızda sürekli beyaz ivme yoğunluğu (birim/s^1.5),
    #                             sabit seviyede birim/√s
    kalman_r: float = 0.1       # ölçüm gürültüsü σ (birim)
    ab_alpha: float = 0.5
    ab_beta: float = 0.1
    gauss_sigma: float = 2.0    # örnek cinsinden

    @property
    def nyquist(self) -> float:
        return self.fs / 2.0

    @property
    def uses_band_edges(self) -> bool:
        """Tasarım geçirme/durdurma bandı kenarlarıyla (Δf) mı yapılıyor? (Maske de bu durumda çizilir.)"""
        return self.family in ("iir", "fir") and (self.auto_order or self.method in ("remez", "firls"))

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> FilterSpec:
        defaults = cls()
        kwargs = {}
        for f in fields(cls):
            if f.name not in data:
                continue
            kind = type(getattr(defaults, f.name))
            try:
                value = data[f.name]
                kwargs[f.name] = (value in (True, "true", "True", 1, "1")) if kind is bool else kind(value)
            except (TypeError, ValueError):
                continue
        spec = cls(**kwargs)
        if (spec.family not in METHODS or spec.method not in METHODS[spec.family] or spec.band not in BANDS
                or spec.kalman_model not in KALMAN_MODELS or spec.comb_type not in COMB_TYPES):
            raise FilterError("Tanınmayan filtre ailesi / yöntemi / bant tipi")
        return spec

    def label(self) -> str:
        """Kısa insan okunur tanım (derece elle verilmişse onu da içerir)."""
        m = self.method
        if self.family == "special":
            if m == "notch":
                text = f"Notch · {format_hz(self.notch_f0)} · Q {self.notch_q:g}"
                return text + (f" · {self.harmonics} harmonik" if self.harmonics > 1 else "")
            if m == "dc_block":
                return f"DC engelleyici · {format_hz(self.dc_cutoff)}"
            if m == "savgol":
                return f"Savitzky-Golay · {self.length} örnek · {self.polyorder}. derece"
            if m == "comb":
                return f"Tarak ({'çentik' if self.comb_type == 'notch' else 'tepe'}) · {format_hz(self.notch_f0)} · " \
                       f"Q {self.notch_q:g}"
            if m in ("ema", "integrator"):
                return f"{SPECIAL_METHODS[m]} · {format_hz(self.dc_cutoff)}"
            if m == "kalman":
                model = "sabit hız" if self.kalman_model == "cv" else "sabit seviye"
                return f"Kalman ({model}) · σ_süreç {self.kalman_q:g} · σ_ölçüm {self.kalman_r:g}"
            if m == "alpha_beta":
                return f"Alfa-beta · α {self.ab_alpha:g} · β {self.ab_beta:g}"
            if m == "gaussian":
                return f"Gauss · {self.length} örnek · σ {self.gauss_sigma:g}"
            if m in ("differentiator", "hilbert"):
                return f"{SPECIAL_METHODS[m]} · {self.order}. derece · {format_hz(self.f1)}"
            if m == "allpass":
                return f"Tüm geçiren · {format_hz(self.notch_f0)} · Q {self.notch_q:g}"
            return f"{SPECIAL_METHODS[m]} · {self.length} örnek"
        edges = (f"{format_hz(self.f1)} – {format_hz(self.f2)}" if self.band in TWO_EDGE
                 else format_hz(self.f1))
        name = IIR_METHODS[m] if self.family == "iir" else FIR_METHODS[m]
        return f"{name} · {BANDS[self.band]} · {edges}"


@dataclass(frozen=True)
class Mask:
    """Tasarım maskesi: geçirme bantlarında kazanç [lo, hi] dB içinde, durdurma bantlarında −atten dB'nin altında."""
    passbands: tuple[tuple[float, float], ...]
    stopbands: tuple[tuple[float, float], ...]
    pass_lo_db: float
    pass_hi_db: float
    stop_db: float


@dataclass(frozen=True, eq=False)
class Design:
    spec: FilterSpec
    sos: np.ndarray | None = None
    b: np.ndarray | None = None
    median_length: int = 0
    order: int = 0
    warnings: tuple[str, ...] = ()
    comb: tuple[int, float, float, float] | None = None  # tarak: (N, b₀, b_N, a_N) — kapalı biçim cevap için

    @property
    def fs(self) -> float:
        return self.spec.fs

    @property
    def kind(self) -> str:
        if self.sos is not None:
            return "iir"
        return "fir" if self.b is not None else "median"

    @property
    def is_linear(self) -> bool:
        return self.kind != "median"

    @cached_property
    def zpk(self) -> tuple[np.ndarray | None, np.ndarray, float]:
        """(sıfırlar, kutuplar, kazanç). Çok uzun FIR'de sıfırlar None (kök bulma pahalı ve anlamsız derecede yavaş)."""
        if self.kind == "iir":
            return _sos_zpk(self.sos)
        if self.kind == "fir":
            b = np.trim_zeros(self.b, "f")
            poles = np.zeros(max(len(self.b) - 1, 0))
            if len(b) > MAX_ROOT_TAPS:
                return None, poles, float(self.b[0])
            return np.roots(b) if len(b) > 1 else np.zeros(0), poles, float(b[0]) if len(b) else 0.0
        return np.zeros(0), np.zeros(0), 1.0

    @cached_property
    def max_pole_radius(self) -> float:
        p = self.zpk[1]
        return float(np.abs(p).max()) if len(p) else 0.0

    @property
    def stable(self) -> bool:
        return self.max_pole_radius < 1.0

    @cached_property
    def linear_phase(self) -> bool:
        if self.kind != "fir":
            return False
        b = self.b
        return bool(np.allclose(b, b[::-1], rtol=0, atol=1e-12 * max(np.abs(b).max(), 1e-300))
                    or np.allclose(b, -b[::-1], rtol=0, atol=1e-12 * max(np.abs(b).max(), 1e-300)))

    @property
    def delay_samples(self) -> float | None:
        """Sabit gecikme (doğrusal fazlı FIR ve medyan için); diğerlerinde frekansa bağlıdır (None)."""
        if self.kind == "fir" and self.linear_phase:
            return (len(self.b) - 1) / 2.0
        if self.kind == "median":
            return (self.median_length - 1) / 2.0
        return None

    def describe(self) -> str:
        s = self.spec
        if s.family == "special":
            if s.method in ("differentiator", "hilbert"):  # derece tip III / IV için ayarlanmış olabilir
                return f"{SPECIAL_METHODS[s.method]} · {self.order}. derece · {format_hz(s.f1)}"
            return s.label()
        return f"{s.label()} · {self.order}. derece"

    def structure_text(self) -> str:
        if self.kind == "iir":
            n = len(self.sos)
            return f"{self.order}. derece · {n} ikinci derece bölüm (SOS)"
        if self.kind == "fir":
            return f"{self.order}. derece · {len(self.b)} katsayı" + (" · doğrusal faz" if self.linear_phase else "")
        return f"{self.median_length} örnek pencere · doğrusal olmayan"


def _sos_zpk(sos: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Bölüm bölüm kökler. scipy.sos2zpk, kazancın toplandığı ilk bölümde katsayılar çok küçükse (ör. 39. derece
    Butterworth'te ~1e-20) polinomu "kötü koşullu" sayıp baştaki terimleri atabiliyor; burada her bölüm kendi
    ölçeğinde çözülür."""
    zeros, poles, gain = [], [], 1.0
    for sec in sos:
        b, a = sec[:3].astype(float), sec[3:].astype(float)
        trailing = 0  # 1. derece bölüm: b2 = a2 = 0 → ortak z⁻¹ çarpanı sadeleşir
        while trailing < 2 and b[2 - trailing] == 0 and a[2 - trailing] == 0:
            trailing += 1
        b, a = b[:3 - trailing], a[:3 - trailing]
        nz = np.flatnonzero(b)
        if len(nz):
            lead = b[nz[0]]
            gain *= lead / a[0]
            zeros.extend(np.roots(b[nz[0]:] / lead))  # baştaki sıfır katsayı = sonsuzdaki sıfır (çizilmez)
        else:
            gain = 0.0
        poles.extend(np.roots(a / a[0]))
    zeros, poles = np.asarray(zeros, dtype=complex), np.asarray(poles, dtype=complex)
    # Tek dereceli tasarımlarda bölümleri eşitlemek için eklenen orijindeki kutup/sıfır çiftleri birbirini götürür
    at_origin_z, at_origin_p = np.abs(zeros) < 1e-12, np.abs(poles) < 1e-12
    cancel = min(at_origin_z.sum(), at_origin_p.sum())
    if cancel:
        zeros = np.delete(zeros, np.flatnonzero(at_origin_z)[:cancel])
        poles = np.delete(poles, np.flatnonzero(at_origin_p)[:cancel])
    return zeros, poles, float(gain)


# ============================================================================ tasarım
def design(spec: FilterSpec) -> Design:
    """Spesifikasyondan filtre tasarlar. Geçersiz girdide `FilterError` fırlatır."""
    _validate(spec)
    try:
        if spec.family == "iir":
            return _design_iir(spec)
        if spec.family == "fir":
            return _design_fir(spec)
        return _design_special(spec)
    except FilterError:
        raise
    except (ValueError, ArithmeticError, np.linalg.LinAlgError) as exc:
        # ör. Parks-McClellan yakınsamadı, bant kenarları çok yakın, Riccati çözülemedi
        raise FilterError(f"Bu parametrelerle tasarım yapılamadı ({exc}). Dereceyi, kesimi ya da Δf'yi "
                          "değiştirin.") from exc


def _validate(spec: FilterSpec) -> None:
    if spec.family not in METHODS or spec.method not in METHODS[spec.family]:
        raise FilterError("Bilinmeyen filtre yöntemi")
    if not (math.isfinite(spec.fs) and spec.fs > 0):
        raise FilterError("Örnekleme hızı pozitif olmalı")
    nyq = spec.nyquist
    if spec.family == "special":
        m = spec.method
        if m == "notch":
            if not 0 < spec.notch_f0 < nyq:
                raise FilterError(f"Notch frekansı 0 ile Nyquist ({format_hz(nyq)}) arasında olmalı")
            if spec.notch_q <= 0:
                raise FilterError("Q pozitif olmalı")
            if spec.harmonics < 1:
                raise FilterError("Harmonik sayısı en az 1 olmalı")
        elif m in ("dc_block", "ema", "integrator"):
            if not 0 < spec.dc_cutoff < nyq:
                raise FilterError(f"Kesim frekansı 0 ile Nyquist ({format_hz(nyq)}) arasında olmalı")
        elif m in ("comb", "allpass"):
            if not 0 < spec.notch_f0 < nyq:
                raise FilterError(f"f₀ 0 ile Nyquist ({format_hz(nyq)}) arasında olmalı")
            if spec.notch_q <= 0:
                raise FilterError("Q pozitif olmalı")
            if m == "comb":
                ratio = spec.fs / spec.notch_f0
                if abs(ratio - round(ratio)) > 1e-9 * ratio:
                    raise FilterError(f"Tarak filtresinde fs / f₀ tam sayı olmalı (şu an {ratio:.4g}); "
                                      f"ör. fs = {spec.fs:g} Hz için f₀ = {spec.fs / max(round(ratio), 1):g} Hz")
                if round(ratio) > 4096:
                    raise FilterError("Tarak filtresinde fs / f₀ en fazla 4096 olabilir")
        elif m == "kalman":
            if spec.kalman_q <= 0 or spec.kalman_r <= 0:
                raise FilterError("Kalman gürültü σ değerleri pozitif olmalı")
        elif m == "alpha_beta":
            a, b = spec.ab_alpha, spec.ab_beta
            if not (0 < a < 2 and 0 < b < 4 - 2 * a):  # kararlılık bölgesi (Jury): 0<α<2, 0<β<4−2α
                raise FilterError("Alfa-beta kararsız: 0 < α < 2 ve 0 < β < 4 − 2α olmalı (tipik: α 0.1 … 0.8)")
        elif m in ("differentiator", "hilbert"):
            if not 2 <= spec.order <= MAX_LS_TAPS - 1:
                raise FilterError(f"Derece 2 … {MAX_LS_TAPS - 1} olmalı")
            if not 0 < spec.f1 < nyq:
                raise FilterError(f"f₁ 0 ile Nyquist ({format_hz(nyq)}) arasında olmalı")
            if m == "differentiator" and not spec.transition > 0:
                raise FilterError("Türev alıcıda geçiş bandı Δf pozitif olmalı")
            if m == "differentiator" and not spec.f1 + spec.transition < nyq:
                raise FilterError("Türev alıcıda f₁ + Δf Nyquist'ten küçük olmalı")
            if m == "hilbert" and not spec.f1 < 0.45 * nyq:
                raise FilterError("Hilbert dönüştürücüde alt kenar f₁, 0.225·fs'den küçük olmalı "
                                  "(bant f₁ … fs/2 − f₁ yeterince geniş kalsın)")
        else:
            if not 1 <= spec.length <= MAX_LENGTH:
                raise FilterError(f"Pencere boyu 1 … {MAX_LENGTH} olmalı")
            if m == "savgol":
                if spec.length % 2 == 0 or spec.length < 3:
                    raise FilterError("Savitzky-Golay pencere boyu 3 veya daha büyük tek sayı olmalı")
                if not 0 <= spec.polyorder < spec.length:
                    raise FilterError("Polinom derecesi pencere boyundan küçük olmalı")
            if m == "gaussian" and spec.gauss_sigma <= 0:
                raise FilterError("Gauss σ pozitif olmalı")
        return

    if spec.band not in BANDS:
        raise FilterError("Bilinmeyen bant tipi")
    if not 0 < spec.f1 < nyq:
        raise FilterError(f"f₁ 0 ile Nyquist ({format_hz(nyq)}) arasında olmalı")
    if spec.band in TWO_EDGE:
        if not spec.f1 < spec.f2:
            raise FilterError("f₁ < f₂ olmalı")
        if not spec.f2 < nyq:
            raise FilterError(f"f₂ Nyquist frekansından ({format_hz(nyq)}) küçük olmalı")
    if not spec.auto_order:
        limit = MAX_IIR_ORDER if spec.family == "iir" else MAX_FIR_TAPS - 1
        if not 1 <= spec.order <= limit:
            raise FilterError(f"Derece 1 … {limit} olmalı")
    if spec.uses_band_edges:
        if spec.transition <= 0:
            raise FilterError("Geçiş bandı genişliği (Δf) pozitif olmalı")
        _edges(spec)  # kenarların aralıkta olduğunu doğrular
    if uses_ripple(spec) and spec.ripple_db <= 0:
        raise FilterError("Geçirme bandı dalgalanması pozitif olmalı")
    if uses_atten(spec) and spec.atten_db <= 0:
        raise FilterError("Durdurma bandı bastırması pozitif olmalı")
    if spec.method == "ellip" and spec.atten_db <= spec.ripple_db:
        raise FilterError("Eliptik filtrede durdurma bastırması geçirme dalgalanmasından büyük olmalı")


def uses_ripple(spec: FilterSpec) -> bool:
    """Geçirme bandı ripple'ı tasarıma giriyor mu?"""
    if spec.family == "iir":
        return spec.method in ("cheby1", "ellip") or (spec.auto_order and spec.method != "bessel")
    return spec.family == "fir" and spec.method in ("remez", "firls")


def uses_atten(spec: FilterSpec) -> bool:
    """Durdurma bandı bastırması tasarıma giriyor mu?"""
    if spec.family == "iir":
        return spec.method in ("cheby2", "ellip") or spec.auto_order
    return spec.family == "fir" and (spec.method in ("remez", "firls") or spec.auto_order)


def _edges(spec: FilterSpec) -> tuple[tuple[tuple[float, float], ...], tuple[tuple[float, float], ...]]:
    """Geçirme ve durdurma bantları (Hz). Kenarlar 0 … Nyquist dışına taşarsa FilterError."""
    f1, f2, tw, nyq = spec.f1, spec.f2, spec.transition, spec.nyquist
    if spec.band == "lowpass":
        passbands, stopbands = ((0.0, f1),), ((f1 + tw, nyq),)
    elif spec.band == "highpass":
        passbands, stopbands = ((f1, nyq),), ((0.0, f1 - tw),)
    elif spec.band == "bandpass":
        passbands, stopbands = ((f1, f2),), ((0.0, f1 - tw), (f2 + tw, nyq))
    else:
        passbands, stopbands = ((0.0, f1), (f2, nyq)), ((f1 + tw, f2 - tw),)
    for lo, hi in stopbands:
        if not 0 <= lo < hi <= nyq:
            raise FilterError(f"Δf = {format_hz(tw)} ile durdurma bandı 0 … {format_hz(nyq)} aralığına sığmıyor; "
                              "Δf'yi küçültün ya da kesim frekanslarını değiştirin")
    return passbands, stopbands


def _ord_edges(spec: FilterSpec) -> tuple[float | list[float], float | list[float]]:
    f1, f2, tw = spec.f1, spec.f2, spec.transition
    if spec.band == "lowpass":
        return f1, f1 + tw
    if spec.band == "highpass":
        return f1, f1 - tw
    if spec.band == "bandpass":
        return [f1, f2], [f1 - tw, f2 + tw]
    return [f1, f2], [f1 + tw, f2 - tw]


def _cutoff(spec: FilterSpec):
    return [spec.f1, spec.f2] if spec.band in TWO_EDGE else spec.f1


def mask_for(spec: FilterSpec) -> Mask | None:
    """Spesifikasyonun gerektirdiği maske; kenarlarla (Δf) tasarlanmayan filtrelerde None."""
    if not spec.uses_band_edges:
        return None
    try:
        passbands, stopbands = _edges(spec)
    except FilterError:
        return None
    rp, rs = spec.ripple_db, spec.atten_db
    if spec.family == "iir":
        lo, hi = (-3.0103 if spec.method == "bessel" else -rp), 0.0
    elif spec.method in ("remez", "firls"):
        delta = _ripple_delta(rp)
        lo, hi = 20 * math.log10(1 - delta), 20 * math.log10(1 + delta)
    else:  # pencere yöntemi: geçirme ve durdurma dalgalılığı aynı (δ = 10^(−atten/20))
        delta = min(10 ** (-rs / 20), 0.5)
        lo, hi = 20 * math.log10(1 - delta), 20 * math.log10(1 + delta)
    return Mask(passbands, stopbands, lo, hi, -rs)


def _ripple_delta(ripple_db: float) -> float:
    """Tepe-tepe ripple (dB) → geçirme bandı sapması δp (|H| ∈ [1−δp, 1+δp])."""
    g = 10 ** (ripple_db / 20)
    return (g - 1) / (g + 1)


def check_mask(d: Design, mask: Mask, points: int = 2048) -> tuple[bool, float, float]:
    """Maske karşılanıyor mu? Dönüş: (uygun, geçirme bandındaki en kötü aşım dB, durdurma bandı tepesi dB)."""
    if d.kind == "fir":  # uzun FIR'de nokta nokta hesap yavaş: yoğun FFT ızgarası + bant kenarları
        nfft = max(4 * points, 1 << int(math.ceil(math.log2(8 * len(d.b)))))
        grid = np.fft.rfftfreq(nfft, 1.0 / d.fs)
        grid_db = _db(np.fft.rfft(d.b, nfft))

        def band_db(lo: float, hi: float) -> np.ndarray:
            inside = grid_db[(grid >= lo) & (grid <= hi)]
            return np.concatenate((inside, _db(_response_at(d, np.array([lo, hi])))))
    else:
        def band_db(lo: float, hi: float) -> np.ndarray:
            return _db(_response_at(d, np.linspace(lo, hi, points)))
    worst_pass, stop_peak = 0.0, -np.inf
    for lo, hi in mask.passbands:
        mag = band_db(lo, hi)
        worst_pass = max(worst_pass, mask.pass_lo_db - mag.min(), mag.max() - mask.pass_hi_db)
    for lo, hi in mask.stopbands:
        stop_peak = max(stop_peak, float(band_db(lo, hi).max()))
    ok = bool(worst_pass <= MASK_TOL_DB and stop_peak <= mask.stop_db + MASK_TOL_DB)
    return ok, float(worst_pass), float(stop_peak)


def _db(h: np.ndarray) -> np.ndarray:
    return 20 * np.log10(np.maximum(np.abs(h), 1e-15))


# ---------------------------------------------------------------------------- IIR
def _design_iir(spec: FilterSpec) -> Design:
    fs, rp, rs, band, m = spec.fs, spec.ripple_db, spec.atten_db, spec.band, spec.method
    warns: list[str] = []
    if spec.auto_order:
        wp, ws = _ord_edges(spec)
        if m == "bessel":
            n, wn = _bessel_order(spec, ws), _cutoff(spec)
            warns.append("Bessel'de f₁ (ve f₂) −3 dB noktasıdır; derece durdurma bastırmasına göre seçildi")
        else:
            order_fn = {"butter": signal.buttord, "cheby1": signal.cheb1ord, "cheby2": signal.cheb2ord,
                        "ellip": signal.ellipord}[m]
            n, wn = order_fn(wp, ws, rp, rs, fs=fs)
        n = int(n)
        if n > MAX_IIR_ORDER:
            raise FilterError(f"Bu spesifikasyon {n}. derece gerektiriyor (en fazla {MAX_IIR_ORDER}); "
                              "Δf'yi genişletin ya da bastırmayı azaltın")
    else:
        n, wn = spec.order, _cutoff(spec)
    sos = _iir_sos(m, n, wn, band, rp, rs, fs)
    d = Design(spec, sos=sos, order=n * (2 if band in TWO_EDGE else 1), warnings=tuple(warns))
    return _with_stability(d)


def _iir_sos(method: str, n: int, wn, band: str, rp: float, rs: float, fs: float) -> np.ndarray:
    if method == "butter":
        return signal.butter(n, wn, band, fs=fs, output="sos")
    if method == "cheby1":
        return signal.cheby1(n, rp, wn, band, fs=fs, output="sos")
    if method == "cheby2":
        return signal.cheby2(n, rs, wn, band, fs=fs, output="sos")
    if method == "ellip":
        return signal.ellip(n, rp, rs, wn, band, fs=fs, output="sos")
    return signal.bessel(n, wn, band, norm="mag", fs=fs, output="sos")


def _bessel_order(spec: FilterSpec, ws) -> int:
    """Bessel için derece tahmini fonksiyonu yok: durdurma kenarında bastırma sağlanana kadar dereceyi artır."""
    ws = np.atleast_1d(ws).astype(float)
    for n in range(1, MAX_IIR_ORDER + 1):
        sos = _iir_sos("bessel", n, _cutoff(spec), spec.band, 0, 0, spec.fs)
        _, h = signal.freqz_sos(sos, worN=ws, fs=spec.fs)
        if np.all(20 * np.log10(np.maximum(np.abs(h), 1e-15)) <= -spec.atten_db):
            return n
    raise FilterError(f"Bessel ile bu bastırma {MAX_IIR_ORDER}. dereceye kadar sağlanamıyor; Δf'yi genişletin")


def _with_stability(d: Design) -> Design:
    r = d.max_pole_radius
    extra = []
    if r >= 1.0:
        extra.append(f"Kararsız: en büyük kutup |p| = {r:.6f} ≥ 1")
    elif r > 0.99999:
        extra.append(f"Kutuplar birim çembere çok yakın (|p| = {r:.7f}); uzun oturma süresi ve sayısal hassasiyet "
                     "sorunları olabilir")
    if not extra:
        return d
    return Design(d.spec, sos=d.sos, b=d.b, median_length=d.median_length, order=d.order,
                  warnings=d.warnings + tuple(extra))


# ---------------------------------------------------------------------------- FIR
def _fix_parity(numtaps: int, spec: FilterSpec, warns: list[str]) -> int:
    """Yüksek geçiren / bant durduran (Nyquist'te kazanç ≠ 0) ve en küçük kareler tek sayıda katsayı ister."""
    need_odd = spec.band in ("highpass", "bandstop") or spec.method == "firls"
    if need_odd and numtaps % 2 == 0:
        numtaps += 1
        reason = "En küçük kareler" if spec.method == "firls" else BANDS[spec.band]
        warns.append(f"{reason} FIR tek sayıda katsayı ister; derece {numtaps - 1} yapıldı")
    return numtaps


def _design_fir(spec: FilterSpec) -> Design:
    warns: list[str] = []
    m = spec.method
    mask = mask_for(spec)
    if m in FIR_WINDOWS:
        if spec.auto_order:
            tw = spec.transition
            if m == "kaiser":
                numtaps, beta = signal.kaiserord(spec.atten_db, tw / spec.nyquist)
            else:
                numtaps, beta = math.ceil(_WINDOW_WIDTH[m] * spec.fs / tw) + 1, spec.kaiser_beta
                if spec.atten_db > _WINDOW_ATTEN[m]:
                    warns.append(f"{FIR_WINDOWS[m]} penceresi en fazla ~{_WINDOW_ATTEN[m]:.0f} dB bastırma sağlar; "
                                 "daha yüksek bastırma için Kaiser veya Parks-McClellan seçin")
            cutoff = _mid_cutoff(spec)
        else:
            numtaps, beta, cutoff = spec.order + 1, spec.kaiser_beta, _cutoff(spec)
        window = ("kaiser", beta) if m == "kaiser" else m

        def build(n: int) -> np.ndarray:
            return signal.firwin(n, cutoff, window=window, pass_zero=spec.band, fs=spec.fs)

        numtaps = _fix_parity(int(numtaps), spec, warns)
        if spec.auto_order and m == "kaiser":
            numtaps, b = _search_taps(spec, build, numtaps, mask)
        else:
            _check_taps(numtaps, spec)
            b = build(numtaps)
    else:
        bands, desired, weight = _ls_bands(spec)
        if m == "remez":
            def build(n: int) -> np.ndarray:
                return signal.remez(n, bands, desired, weight=weight, fs=spec.fs, maxiter=60)
        else:
            pairs = [d for d in desired for _ in range(2)]

            def build(n: int) -> np.ndarray:
                return signal.firls(n, bands, pairs, weight=weight, fs=spec.fs)
        if spec.auto_order:
            numtaps = _fix_parity(_estimate_taps(spec), spec, [])
            numtaps, b = _search_taps(spec, build, numtaps, mask)
        else:
            numtaps = _fix_parity(spec.order + 1, spec, warns)
            _check_taps(numtaps, spec)
            b = build(numtaps)
    return Design(spec, b=np.asarray(b, dtype=np.float64), order=len(b) - 1, warnings=tuple(warns))


def _tap_limit(spec: FilterSpec) -> int:
    return MAX_LS_TAPS if spec.method in ("remez", "firls") else MAX_FIR_TAPS


def _check_taps(numtaps: int, spec: FilterSpec) -> None:
    limit = _tap_limit(spec)
    if numtaps > limit:
        hint = " ya da daha uzun filtreler için Kaiser penceresini seçin" if limit == MAX_LS_TAPS else ""
        raise FilterError(f"{numtaps} katsayı gerekiyor (bu yöntemde en fazla {limit}); Δf'yi genişletin{hint}")


def _mid_cutoff(spec: FilterSpec):
    """Otomatik dereceli pencere tasarımı: −6 dB kesim, geçiş bandının ortasına konur."""
    h = spec.transition / 2
    f1, f2 = spec.f1, spec.f2
    return {"lowpass": f1 + h, "highpass": f1 - h, "bandpass": [f1 - h, f2 + h], "bandstop": [f1 + h, f2 - h]}[spec.band]


def _ls_bands(spec: FilterSpec) -> tuple[list[float], list[float], list[float]]:
    """Parks-McClellan / en küçük kareler için bant kenarları, hedef kazançlar ve ağırlıklar."""
    passbands, stopbands = _edges(spec)
    w_stop = _ripple_delta(spec.ripple_db) / 10 ** (-spec.atten_db / 20)
    tagged = sorted([(lo, hi, 1.0, 1.0) for lo, hi in passbands] + [(lo, hi, 0.0, w_stop) for lo, hi in stopbands])
    bands = [edge for lo, hi, _, _ in tagged for edge in (lo, hi)]
    return bands, [t[2] for t in tagged], [t[3] for t in tagged]


def _estimate_taps(spec: FilterSpec) -> int:
    """Eşdalgalı FIR için Kaiser derece tahmini."""
    dp, ds = _ripple_delta(spec.ripple_db), 10 ** (-spec.atten_db / 20)
    n = (-20 * math.log10(math.sqrt(dp * ds)) - 13) / (14.6 * spec.transition / spec.fs)
    return max(3, math.ceil(n) + 1)


def _search_taps(spec: FilterSpec, build, numtaps: int, mask: Mask) -> tuple[int, np.ndarray]:
    """Maskeyi karşılayan en kısa filtreyi arar (tahminden başlayıp büyüterek, sonra küçültmeyi deneyerek)."""
    step = 2 if (spec.band in ("highpass", "bandstop") or spec.method == "firls") else 1

    def ok(n: int) -> tuple[bool, np.ndarray]:
        b = build(n)
        d = Design(spec, b=np.asarray(b, dtype=np.float64), order=n - 1)
        return check_mask(d, mask)[0], b

    _check_taps(numtaps, spec)
    good, b = ok(numtaps)
    too_short = 0
    limit = _tap_limit(spec)
    while not good:  # tahmin yetmedi: büyüt (adım ~%4, eşlik korunur)
        too_short = numtaps
        numtaps += max(step, (numtaps // 25) // step * step)
        if numtaps > limit:
            raise FilterError(f"Spesifikasyon {limit} katsayıya kadar karşılanamadı; Δf'yi genişletin ya da "
                              "bastırma / dalgalanma hedefini gevşetin")
        good, b = ok(numtaps)
    while numtaps - step > max(too_short, 2):  # tahmin ya da son adım fazla olabilir: en kısasına in
        fine, shorter = ok(numtaps - step)
        if not fine:
            break
        numtaps, b = numtaps - step, shorter
    return numtaps, b


# ---------------------------------------------------------------------------- özel filtreler
def _design_special(spec: FilterSpec) -> Design:
    m, fs, warns = spec.method, spec.fs, []
    if m == "notch":
        bandwidth = spec.notch_f0 / spec.notch_q  # tüm harmoniklerde aynı bant genişliği
        sections = []
        for k in range(1, spec.harmonics + 1):
            fk = k * spec.notch_f0
            if fk >= spec.nyquist:
                warns.append(f"{k}. harmonik ({format_hz(fk)}) Nyquist'in üstünde; atlandı")
                break
            b, a = signal.iirnotch(fk, fk / bandwidth, fs=fs)
            sections.append(np.concatenate((b, a)))
        return _with_stability(Design(spec, sos=np.vstack(sections), order=2 * len(sections), warnings=tuple(warns)))
    if m == "dc_block":
        # y[n] = g·(x[n] − x[n−1]) + r·y[n−1]; 1. derece Butterworth yüksek geçiren ile aynı yapı, f_c'de tam −3 dB
        sos = signal.butter(1, spec.dc_cutoff, "highpass", fs=fs, output="sos")
        return _with_stability(Design(spec, sos=sos, order=1))
    if m == "moving_avg":
        n = spec.length
        return Design(spec, b=np.full(n, 1.0 / n), order=n - 1)
    if m == "median":
        n = spec.length
        if n % 2 == 0:
            n += 1
            warns.append(f"Medyan pencere boyu tek olmalı; {n} yapıldı")
        return Design(spec, median_length=n, order=n - 1, warnings=tuple(warns))
    if m == "savgol":
        b = signal.savgol_coeffs(spec.length, spec.polyorder)
        return Design(spec, b=np.asarray(b, dtype=np.float64), order=spec.length - 1)
    if m == "gaussian":
        n = spec.length
        b = signal.windows.gaussian(n, spec.gauss_sigma)
        if n < 6 * spec.gauss_sigma:
            warns.append(f"Pencere boyu 6σ'dan ({6 * spec.gauss_sigma:g}) kısa: Gauss kuyrukları kesiliyor")
        return Design(spec, b=b / b.sum(), order=n - 1, warnings=tuple(warns))
    if m == "comb":
        # scipy'de pass_zero tarağı yarım f₀ kaydırır: çentikte False (DC ve k·f₀ bastırılır), tepede True
        # (DC ve k·f₀ geçer, aradakiler bastırılır)
        b, a = signal.iircomb(spec.notch_f0, spec.notch_q, ftype=spec.comb_type, fs=fs,
                              pass_zero=spec.comb_type == "peak")
        n = len(a) - 1
        # H(z) = (b₀ + b_N·z⁻ᴺ) / (1 + a_N·z⁻ᴺ): kökler zᴺ = −b_N/b₀ ve zᴺ = −a_N
        sos = _comb_sos(n, -b[-1] / b[0], -a[-1], b[0])
        return _with_stability(Design(spec, sos=sos, order=n, comb=(n, float(b[0]), float(b[-1]), float(a[-1]))))
    if m == "ema":
        # y[n] = y[n−1] + α·(x[n] − y[n−1]); α, |H| = 1/√2 olacak şekilde kesimden tam çözülür
        e = 2 * math.sin(math.pi * spec.dc_cutoff / fs) ** 2  # 1 − cos w, iptalsiz
        alpha = math.sqrt(e * e + 2 * e) - e
        pole = 1 - alpha
        sos = np.array([[alpha, 0.0, 0.0, 1.0, -pole, 0.0]])
        warns.append(f"α = {alpha:.6g} (y[n] = y[n−1] + α·(x[n] − y[n−1]))")
        return _with_stability(Design(spec, sos=sos, order=1, warnings=tuple(warns)))
    if m == "integrator":
        # y[n] = λ·y[n−1] + T·x[n]: f ≫ f_c'de 1/(2πf) (integral), DC'de sınırlı kazanç (sürüklenmez)
        lam = math.exp(-2 * math.pi * spec.dc_cutoff / fs)
        sos = np.array([[1.0 / fs, 0.0, 0.0, 1.0, -lam, 0.0]])
        warns.append(f"λ = {lam:.6g}; çıkış birimi: giriş birimi × s")
        return _with_stability(Design(spec, sos=sos, order=1, warnings=tuple(warns)))
    if m in ("kalman", "alpha_beta"):
        return _design_tracker(spec)
    if m == "allpass":
        # RBJ ikinci derece tüm geçiren: |H| = 1, f₀'da faz −180°
        w0 = 2 * math.pi * spec.notch_f0 / fs
        alpha = math.sin(w0) / (2 * spec.notch_q)
        b = np.array([1 - alpha, -2 * math.cos(w0), 1 + alpha])
        a = np.array([1 + alpha, -2 * math.cos(w0), 1 - alpha])
        return _with_stability(Design(spec, sos=np.concatenate((b, a))[None, :] / a[0], order=2))
    if m == "differentiator":
        numtaps = spec.order + 1
        if numtaps % 2:  # tip IV (çift katsayı): Nyquist'te zorunlu sıfır yok
            numtaps += 1
            warns.append(f"Türev alıcı çift sayıda katsayıyla daha iyi: derece {numtaps - 1} yapıldı")
        b = signal.remez(numtaps, [0, spec.f1, spec.f1 + spec.transition, spec.nyquist], [1, 0],
                         type="differentiator", fs=fs)
        b = b * 2 * math.pi * fs  # remez eğimi f/fs; birim/s için H(f) = j·2πf
        warns.append("Çıkış birimi: giriş birimi / s")
        return Design(spec, b=b, order=numtaps - 1, warnings=tuple(warns))
    numtaps = spec.order + 1  # hilbert
    if numtaps % 2 == 0:  # tip III (tek katsayı, ters simetrik): tam örnek gecikmesi
        numtaps += 1
        warns.append(f"Hilbert dönüştürücü tek sayıda katsayı ister: derece {numtaps - 1} yapıldı")
    b = -signal.remez(numtaps, [spec.f1, spec.nyquist - spec.f1], [1], type="hilbert", fs=fs)
    return Design(spec, b=b, order=numtaps - 1, warnings=tuple(warns))


def _comb_sos(n: int, zero_c: float, pole_c: float, gain: float) -> np.ndarray:
    """zᴺ = zero_c sıfırları ve zᴺ = pole_c kutuplarıyla tarak filtresinin SOS'u, O(N log N).

    Her kök kümesi |c|^(1/N)·e^{j(φ + 2πk)/N}; eşlenik çiftler açı sırasıyla ikinci derece bölümlere, gerçel
    kökler (0 ya da π açısında) birinci / ikinci derece bölümlere toplanır."""
    def factors(c: float) -> tuple[list[tuple[float, list[float]]], list[float]]:
        radius = abs(c) ** (1.0 / n)
        phase0 = 0.0 if c >= 0 else math.pi
        quads, reals = [], []
        for k in range(n):
            ang = (phase0 + 2 * math.pi * k) / n
            ang = math.atan2(math.sin(ang), math.cos(ang))
            if abs(ang) < 1e-12 or abs(abs(ang) - math.pi) < 1e-12:
                reals.append(radius * math.copysign(1.0, math.cos(ang)))
            elif ang > 0:
                quads.append((ang, [1.0, -2 * radius * math.cos(ang), radius * radius]))
        quads.sort()
        return quads, reals

    def to_quads(quads, reals):
        polys = [q for _, q in quads]
        while len(reals) >= 2:  # iki gerçel kök tek ikinci derece çarpan
            r1, r2 = reals.pop(), reals.pop()
            polys.append([1.0, -(r1 + r2), r1 * r2])
        single = [1.0, -reals[0], 0.0] if reals else None
        return polys, single

    zq, zs = to_quads(*factors(zero_c))
    pq, ps = to_quads(*factors(pole_c))
    sections = [np.r_[z, p] for z, p in zip(zq, pq)]
    if zs is not None and ps is not None:
        sections.append(np.r_[zs, ps])
    elif zs is not None or ps is not None or len(zq) != len(pq):
        raise FilterError("Tarak filtresinin kökleri eşlenemedi")
    sos = np.array(sections, dtype=np.float64)
    sos[0, :3] *= gain
    return sos


def _design_tracker(spec: FilterSpec) -> Design:
    """Kararlı hâl Kalman ve alfa-beta izleyici: x̂ₖ = (I − K·H)·F·x̂ₖ₋₁ + K·zₖ, çıkış H·x̂ₖ.

    Kalman kazancı ayrık Riccati denkleminden (DARE) bulunur; kazanç sabitlendiği için filtre doğrusal ve zamanla
    değişmez olur, frekans cevabı / katsayıları diğer IIR filtreler gibi incelenip dışa aktarılabilir. Zamanla
    değişen Kalman, ilk birkaç örnekten sonra bununla aynı çıktıyı verir.
    """
    from scipy.linalg import solve_discrete_are

    T = 1.0 / spec.fs
    warns = []
    if spec.method == "kalman" and spec.kalman_model == "rw":
        F, H = np.array([[1.0]]), np.array([[1.0]])
        q, r = spec.kalman_q ** 2 * T, spec.kalman_r ** 2
        p = (q + math.sqrt(q * q + 4 * q * r)) / 2  # skaler Riccati'nin kapalı çözümü (öngörü kovaryansı)
        K = np.array([[p / (p + r)]])
    else:
        F, H = np.array([[1.0, T], [0.0, 1.0]]), np.array([[1.0, 0.0]])
        if spec.method == "kalman":
            # Durum [konum, hız·T] (örnek başına hız) ve R = 1'e normalize: Q' = λ²·[[1/3, 1/2], [1/2, 1]],
            # λ = σ_süreç·T^1.5/σ_ölçüm. Kazanç yalnız λ'ya bağlıdır; mutlak ölçek koşul sayısını bozmaz.
            lam2 = (spec.kalman_q * T ** 1.5 / spec.kalman_r) ** 2
            Fn = np.array([[1.0, 1.0], [0.0, 1.0]])
            Qn = lam2 * np.array([[1 / 3, 1 / 2], [1 / 2, 1.0]])
            with warnings.catch_warnings(), np.errstate(all="ignore"):  # aşırı oranda sonuç aşağıda denetlenir
                warnings.simplefilter("ignore")
                try:
                    P = solve_discrete_are(Fn.T, H.T, Qn, np.array([[1.0]]))
                except (ValueError, np.linalg.LinAlgError):
                    P = np.full((2, 2), np.nan)
            Kn = P @ H.T / (P[0, 0] + 1.0)
            K = np.array([[Kn[0, 0]], [Kn[1, 0] / T]])
        else:
            K = np.array([[spec.ab_alpha], [spec.ab_beta / T]])
    if spec.method == "kalman":
        if not (np.all(np.isfinite(K)) and 0 < K[0, 0] < 1):
            raise FilterError("Kalman kazancı hesaplanamadı (σ_süreç / σ_ölçüm oranı aşırı küçük ya da büyük); "
                              "değerleri birbirine yaklaştırın")
        gains = ", ".join(f"{k:.6g}" for k in K.ravel())
        warns.append(f"Kararlı hâl kazancı K = [{gains}]")
    A = (np.eye(len(F)) - K @ H) @ F
    b, a = signal.ss2tf(A, K, H @ A, H @ K)
    b = np.atleast_1d(np.squeeze(b))
    sos = signal.tf2sos(b, a)
    return _with_stability(Design(spec, sos=sos, order=len(F), warnings=tuple(warns)))


# ============================================================================ analiz
@dataclass(frozen=True)
class Response:
    freqs: np.ndarray        # Hz
    h: np.ndarray            # karmaşık frekans cevabı
    magnitude_db: np.ndarray
    phase_deg: np.ndarray    # sarılmamış (unwrap)
    group_delay: np.ndarray  # örnek; tanımsız noktalarda (|H| ≈ 0) NaN


def _response_at(d: Design, freqs: np.ndarray) -> np.ndarray:
    if d.comb is not None:  # tarak: H = (b₀ + b_N·e) / (1 + a_N·e), e = e^(−jωN) — yüzlerce bölüm yerine kapalı biçim
        n, b0, bn, an = d.comb
        e = np.exp(-1j * 2 * np.pi * np.asarray(freqs, dtype=np.float64) / d.fs * n)
        return (b0 + bn * e) / (1 + an * e)
    if d.kind == "iir":
        return signal.freqz_sos(d.sos, worN=freqs, fs=d.fs)[1]
    if d.kind == "fir":
        return signal.freqz(d.b, 1.0, worN=freqs, fs=d.fs)[1]
    raise FilterError("Medyan filtre doğrusal değildir; frekans cevabı tanımsız")


def frequency_grid(d: Design, points: int = 8192) -> np.ndarray:
    """IIR grafiği için frekans ızgarası: doğrusal + logaritmik noktalar ve dar özelliklerin (notch, kenarlar)
    çevresinde sıklaştırılmış örnekleme. Böylece 50 Hz'lik dar bir çentik de 25 kHz'lik eksende görünür."""
    nyq = d.fs / 2
    parts = [np.linspace(0, nyq, points), np.geomspace(nyq * 1e-5, nyq, points // 2)]
    s = d.spec
    focus = []
    if s.family == "special" and s.method == "notch":
        bw = s.notch_f0 / s.notch_q
        focus = [(k * s.notch_f0, bw) for k in range(1, s.harmonics + 1) if k * s.notch_f0 < nyq]
    elif s.family == "special" and s.method == "comb":
        parts.append(np.linspace(0, nyq, 65_537))  # çok sayıda dar çentik: yoğun düzgün ızgara
    elif s.family == "special" and s.method == "allpass":
        focus = [(s.notch_f0, s.notch_f0 / s.notch_q)]
    elif s.family == "special" and s.method in ("dc_block", "ema", "integrator"):
        focus = [(s.dc_cutoff, s.dc_cutoff)]
    elif s.family in ("iir", "fir"):
        width = max(s.transition if s.uses_band_edges else 0, nyq / points * 4)
        focus = [(s.f1, width)] + ([(s.f2, width)] if s.band in TWO_EDGE else [])
    for fc, width in focus:
        parts.append(np.clip(np.linspace(fc - 6 * width, fc + 6 * width, 801), 0, nyq))
    return np.unique(np.concatenate(parts))


def analyze(d: Design, freqs: np.ndarray | None = None) -> Response:
    """Frekans cevabı, faz ve grup gecikmesi. `freqs` verilmezse FIR'de FFT ızgarası, IIR'de `frequency_grid`."""
    if not d.is_linear:
        raise FilterError("Medyan filtre doğrusal değildir; frekans cevabı tanımsız")
    with np.errstate(divide="ignore", invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if d.kind == "fir" and freqs is None:
            # FFT ile: H = DFT(b), grup gecikmesi = Re{DFT(n·b) / DFT(b)}; 16k katsayıda bile milisaniyeler
            nfft = max(2 * FIR_FFT_POINTS, 1 << int(math.ceil(math.log2(2 * len(d.b)))))
            freqs = np.fft.rfftfreq(nfft, 1.0 / d.fs)
            h = np.fft.rfft(d.b, nfft)
            gd = np.real(np.fft.rfft(np.arange(len(d.b)) * d.b, nfft) / h)
        else:
            if freqs is None:
                freqs = frequency_grid(d)
            h = _response_at(d, freqs)
            if d.kind == "fir":
                gd = signal.group_delay((d.b, [1.0]), w=freqs, fs=d.fs)[1]
            elif d.comb is not None:
                n, b0, bn, an = d.comb
                e = np.exp(-1j * 2 * np.pi * freqs / d.fs * n)
                gd = n * (np.real(bn * e / (b0 + bn * e)) - np.real(an * e / (1 + an * e)))
            else:
                gd = _sos_group_delay(d.sos, 2 * np.pi * freqs / d.fs)
    mag = np.abs(h)
    gd = np.asarray(gd, dtype=np.float64)
    gd[mag < 1e-6 * max(mag.max(), 1e-300)] = np.nan  # sıfırlarda grup gecikmesi tanımsız
    return Response(freqs, h, 20 * np.log10(np.maximum(mag, 1e-15)), np.degrees(np.unwrap(np.angle(h))), gd)


def _sos_group_delay(sos: np.ndarray, w: np.ndarray, block: int = 16) -> np.ndarray:
    """Kaskadın grup gecikmesi (örnek): her bölüm için τ = Re{Σk·b_k·e^(−jωk) / B} − Re{Σk·a_k·e^(−jωk) / A},
    bölümler toplanır. Vektörel (bölüm başına scipy çağrısından onlarca kat hızlı)."""
    e1 = np.exp(-1j * w)
    e2 = e1 * e1
    gd = np.zeros(len(w))
    for i in range(0, len(sos), block):
        s = np.asarray(sos[i:i + block], dtype=np.float64)
        b0, b1, b2 = s[:, 0:1], s[:, 1:2], s[:, 2:3]
        a0, a1, a2 = s[:, 3:4], s[:, 4:5], s[:, 5:6]
        num = (b1 * e1 + 2 * b2 * e2) / (b0 + b1 * e1 + b2 * e2)
        den = (a1 * e1 + 2 * a2 * e2) / (a0 + a1 * e1 + a2 * e2)
        gd += np.real(num - den).sum(axis=0)
    return gd


def impulse_response(d: Design, n: int | None = None) -> np.ndarray | None:
    if d.kind == "median":
        return None
    if d.kind == "fir":
        n = n or len(d.b) + max(8, len(d.b) // 8)
        out = np.zeros(n)
        out[:min(n, len(d.b))] = d.b[:n]
        return out
    if n is None:
        r = d.max_pole_radius
        n = 2048 if r >= 1 else int(np.clip(math.ceil(math.log(1e-5) / math.log(max(r, 1e-9))) + 8, 64, 20_000))
    return signal.sosfilt(d.sos, signal.unit_impulse(n))


def step_response(d: Design, n: int | None = None) -> np.ndarray | None:
    h = impulse_response(d, n)
    return None if h is None else np.cumsum(h)


# ============================================================================ uygulama
def _median_valid(buf: np.ndarray, size: int) -> np.ndarray:
    """buf üzerinde kayan medyan; yalnızca pencerenin tamamen içeride olduğu len(buf) − size + 1 çıktı."""
    half = size // 2
    count = len(buf) - size + 1
    if np.iscomplexobj(buf):
        return _median_valid(buf.real, size) + 1j * _median_valid(buf.imag, size)
    return ndimage.median_filter(buf, size=size, mode="nearest")[half:half + count]


def finite_runs(x: np.ndarray) -> list[tuple[int, int, bool]]:
    """Diziyi sonlu / sonlu olmayan ardışık bölgelere ayırır: [(başlangıç, bitiş, sonlu_mu), …]."""
    finite = np.isfinite(x)
    if finite.all():
        return [(0, len(x), True)] if len(x) else []
    edges = np.flatnonzero(np.diff(finite.astype(np.int8))) + 1
    bounds = np.r_[0, edges, len(x)]
    return [(int(a), int(b), bool(finite[a])) for a, b in zip(bounds[:-1], bounds[1:])]


class FilterRunner:
    """Bir tasarımı tek kanala blok blok uygular; durum (zi / geçmiş örnekler) bloklar arasında taşınır."""

    def __init__(self, d: Design):
        self.design = d
        self._state = None

    def reset(self) -> None:
        self._state = None

    def _initial(self, x0) -> np.ndarray:
        d = self.design
        if d.kind == "iir":
            return signal.sosfilt_zi(d.sos) * x0
        m = (len(d.b) if d.kind == "fir" else d.median_length) - 1
        return np.full(m, x0, dtype=np.result_type(x0, np.float64))

    def process(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        x = x.astype(np.complex128 if np.iscomplexobj(x) else np.float64, copy=False)
        if x.size == 0:
            return x.copy()
        if self._state is None:
            self._state = self._initial(x[0])
        d = self.design
        if d.kind == "iir":
            y, self._state = signal.sosfilt(d.sos, x, zi=self._state)
            return y
        hist = self._state
        buf = np.concatenate((hist, x))
        y = signal.convolve(buf, d.b, mode="valid") if d.kind == "fir" else _median_valid(buf, d.median_length)
        m = len(hist)
        self._state = buf[len(buf) - m:].copy() if m else hist
        return y


class ChainRunner:
    """Sıralı filtreler (işlem zinciri) — her biri kendi durumunu taşır."""

    def __init__(self, designs: list[Design]):
        self.runners = [FilterRunner(d) for d in designs]

    def reset(self) -> None:
        for r in self.runners:
            r.reset()

    def process(self, x: np.ndarray) -> np.ndarray:
        for r in self.runners:
            x = r.process(x)
        return np.asarray(x)


def _odd_extend(x: np.ndarray, n: int) -> np.ndarray:
    if n < 1:
        return x
    left = 2 * x[0] - x[n:0:-1]
    right = 2 * x[-1] - x[-2:-n - 2:-1]
    return np.concatenate((left, x, right))


def zero_phase(d: Design, x: np.ndarray) -> np.ndarray:
    """Sıfır fazlı (ileri + geri) uygulama. Doğrusal filtrelerde genlik cevabının karesi uygulanır
    (dB cinsinden bastırma iki katına çıkar), faz kayması ve gecikme olmaz. Medyanda merkezli pencere."""
    x = np.asarray(x)
    x = x.astype(np.complex128 if np.iscomplexobj(x) else np.float64, copy=False)
    if d.spec.family == "special" and d.spec.method in PHASE_DEFINED:
        raise FilterError(f"{SPECIAL_METHODS[d.spec.method]} sıfır fazlı uygulanamaz: ileri + geri uygulama "
                          "genlik cevabının karesini verir ve bu filtrenin amacını (faz / türev / integral) bozar. "
                          "Nedensel uygulayın.")
    n = len(x)
    taps = len(d.b) if d.kind == "fir" else d.median_length if d.kind == "median" else 2
    if n < max(2, taps):  # bölüt filtreden kısa: kenar etkisiz bir çıktı yok
        return np.full(n, np.nan, dtype=x.dtype)
    if d.kind == "iir":
        pad = min(3 * (2 * len(d.sos) + 1), n - 1)
        return signal.sosfiltfilt(d.sos, x, padlen=pad)
    if d.kind == "fir":
        pad = min(3 * len(d.b), n - 1)
        kernel = np.convolve(d.b, d.b[::-1])  # simetrik, tek uzunluk: "same" modu gecikmesiz
        return signal.convolve(_odd_extend(x, pad), kernel, mode="same")[pad:pad + n]
    if np.iscomplexobj(x):
        return (ndimage.median_filter(x.real, size=d.median_length, mode="nearest")
                + 1j * ndimage.median_filter(x.imag, size=d.median_length, mode="nearest"))
    return ndimage.median_filter(x, size=d.median_length, mode="nearest")


def apply_offline(designs: list[Design], x: np.ndarray, zero_phase_mode: bool) -> np.ndarray:
    """Tek parça (kayıpsız) sinyale zinciri uygular: nedensel (canlıyla aynı) veya sıfır fazlı."""
    if zero_phase_mode:
        for d in designs:
            x = zero_phase(d, x)
        return np.asarray(x)
    return ChainRunner(designs).process(x)
