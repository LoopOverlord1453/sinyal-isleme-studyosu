"""Genlik spektrumu (pencereli FFT)."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.signal import get_window

WINDOWS = {
    "Hann": "hann",
    "Hamming": "hamming",
    "Blackman": "blackman",
    "Blackman-Harris": "blackmanharris",
    "Flat-top": "flattop",
    "Kaiser": "kaiser",
    "Dikdörtgen": "boxcar",
}


@lru_cache(maxsize=16)
def _window(name: str, n: int, beta: float) -> np.ndarray:
    spec = (name, beta) if name == "kaiser" else name
    return get_window(spec, n)


ONE_SIDED = "one"
TWO_SIDED = "two"


def amplitude_spectrum(x: np.ndarray, fs: float, window: str = "Hann", beta: float = 8.6,
                       sides: str = ONE_SIDED) -> tuple[np.ndarray, np.ndarray]:
    """Genlik düzeltmeli spektrum; dB (1 birime göre).

    - Çift taraflı (−fs/2 … +fs/2): her bileşen kendi frekansında |X|/Σw. Genliği A olan gerçel bir sinüs,
      +f ve −f'de A/2 olarak görünür (tek taraflıya göre −6.02 dB).
    - Tek taraflı (0 … fs/2), gerçel sinyal: negatif frekans yarısı pozitife katlanır (×2, DC ve Nyquist hariç);
      genliği A olan sinüs A olarak okunur.
    - Tek taraflı, karmaşık sinyal: negatif frekanslar bağımsız bilgi taşıdığı için katlanmaz; yalnızca
      0 … +fs/2 yarısı gösterilir.
    """
    n = len(x)
    w = _window(WINDOWS.get(window, "hann"), n, beta)
    gain = w.sum()
    x = np.nan_to_num(x)
    if np.iscomplexobj(x) or sides == TWO_SIDED:
        spec = np.fft.fftshift(np.fft.fft(x * w))
        freqs = np.fft.fftshift(np.fft.fftfreq(n, 1.0 / fs))
        amp = np.abs(spec) / gain
        if sides == ONE_SIDED:  # karmaşık sinyalin yalnızca pozitif yarısı
            keep = freqs >= 0
            freqs, amp = freqs[keep], amp[keep]
    else:
        spec = np.fft.rfft(x * w)
        freqs = np.fft.rfftfreq(n, 1.0 / fs)
        amp = 2.0 * np.abs(spec) / gain
        amp[0] /= 2.0
        if n % 2 == 0:
            amp[-1] /= 2.0
    return freqs, 20.0 * np.log10(np.maximum(amp, 1e-15))


@dataclass(frozen=True)
class SpectralPeak:
    freq: float     # Hz, bin altı (parabolik interpolasyon)
    level: float    # dB, aynı interpolasyonla (pencerenin bin arası kaybı büyük ölçüde giderilir)
    index: int      # tepenin bulunduğu bin


def mainlobe_bins(window: str, beta: float = 8.6) -> int:
    """Pencere ana lobunun yarı genişliği (ilk sıfıra kadar, bin): aynı tonun komşu binleri ayrı tepe sayılmasın."""
    if window == "Kaiser":
        return int(np.ceil(np.sqrt(1.0 + (beta / np.pi) ** 2)))
    return {"Dikdörtgen": 1, "Hann": 2, "Hamming": 2, "Blackman": 3, "Blackman-Harris": 4,
            "Flat-top": 5}.get(window, 2)


def spectral_peaks(freqs: np.ndarray, mag_db: np.ndarray, count: int = 5, window: str = "Hann",
                   beta: float = 8.6, above_floor: float = 15.0, prominence: float = 10.0,
                   dynamic_range: float = 160.0) -> list[SpectralPeak]:
    """Spektrumun en yüksek `count` tepesi, yüksekten alçağa.

    Tepe: gürültü tabanının (medyan) en az `above_floor` dB üstünde ve çevresinden en az `prominence` dB yükselen
    yerel en büyük. Gürültünün kendi dalgalanması (Rayleigh) 15 dB'yi binde birden az aşar; bu yüzden yalnız gerçek
    bileşenler raporlanır. Taban en güçlü bileşenin en çok `dynamic_range` dB altında sayılır: gürültüsüz (sentetik)
    sinyalde medyan sayısal yuvarlama düzeyine (≈ −300 dB) iner ve yuvarlama kırıntıları tepe sayılırdı; 160 dB,
    24 bit ADC'nin dinamik aralığının da üstündedir. Komşu tepeler arası en az pencerenin ana lob yarı genişliği kadar bin bulunur (aynı tonun
    yan binleri tek tepe). Güçlü bir tonun yan lobları (sızıntı) gürültüyle birlikte yerel tepe oluşturabilir (ör.
    Hamming'de −43 dB düzeyinde): daha güçlü bir tepenin pencere yan lob zarfının (pencerenin DTFT'sinden) 6 dB
    üstüne çıkmayan aday sızıntı sayılır, raporlanmaz. DC ve ana lobu dışarıda bırakılır.

    Frekans: dB üzerinde parabolik interpolasyon (bin altı). Seviye: tepe bininin değeri, bu bin altı konumda
    pencerenin bin arası kaybı (scalloping, pencerenin DTFT'sinden) kadar düzeltilir. Bin arasına düşen tek tonda
    ölçülen en kötü hata (tests/test_spectrum.py): Hann 0.02 bin / 0.1 dB; dikdörtgende çok daha büyüktür."""
    from scipy.signal import find_peaks

    f = np.asarray(freqs, dtype=np.float64)
    m = np.asarray(mag_db, dtype=np.float64)
    if count <= 0 or len(m) < 5 or len(f) != len(m):
        return []
    ok = np.isfinite(m)
    if not ok.all():
        m = np.where(ok, m, np.min(m[ok]) if ok.any() else -300.0)
    df = float(f[1] - f[0])
    half = mainlobe_bins(window, beta)
    floor = max(float(np.median(m)), float(np.max(m)) - dynamic_range)
    idx, _ = find_peaks(m, height=floor + above_floor, prominence=prominence, distance=half + 1)
    idx = idx[np.abs(f[idx]) > (half + 0.5) * abs(df)]       # DC ve ana lobu
    envelope = _sidelobe_envelope(window, beta)
    accepted: list[int] = []
    for i in idx[np.argsort(m[idx])[::-1]]:                  # güçlüden zayıfa: sızıntıyı ele
        if all(m[i] > m[j] + envelope[min(abs(int(i) - j), len(envelope) - 1)] + LEAKAGE_MARGIN_DB
               for j in accepted):
            accepted.append(int(i))
            if len(accepted) == count:
                break
    return [_refine(f, m, i, window, beta) for i in accepted]


def _refine(f: np.ndarray, m: np.ndarray, i: int, window: str, beta: float) -> SpectralPeak:
    """i. binin tepesi bin altı konumda: frekans dB üzerinde parabolik interpolasyonla, seviye pencerenin bu konumdaki
    bin arası kaybı (scalloping) kadar düzeltilerek. Kenardaki ya da yerel en büyük olmayan binde düzeltme yapılmaz."""
    p = 0.0
    if 0 < i < len(m) - 1:
        a, b, c = m[i - 1], m[i], m[i + 1]
        den = a - 2 * b + c
        if den < 0 and b >= max(a, c):
            p = float(np.clip(0.5 * (a - c) / den, -0.5, 0.5))
    grid, loss = _scallop_loss(window, beta)
    level = float(m[i]) - (float(np.interp(p, grid, loss)) if p else 0.0)
    return SpectralPeak(float(f[0] + (i + p) * (f[1] - f[0])), level, int(i))


def spectral_extrema(freqs: np.ndarray, mag_db: np.ndarray, window: str = "Hann",
                     beta: float = 8.6) -> tuple[SpectralPeak | None, SpectralPeak | None]:
    """Spektrumun (verilen aralığın) en büyük ve en küçük noktası: (max, min). Tepe aramasından farklı olarak eşik
    yoktur, DC dahildir; gürültüde de her zaman bir değer döner. Max, tepelerle aynı bin altı düzeltmeyi alır (aralığın
    kenarına düşerse almaz); min, binin kendi değeridir. Max DC'nin ana lobuna düşerse DC bininin kendisi raporlanır:
    tek taraflı spektrumda DC'nin yan binleri ×2 katlandığından (DC bini katlanmaz) ofsetli sinyalde 1. bin DC'den
    yüksek okunur (Hann'da 2 V DC → 0.5 Hz'te 7.4 dB; doğrusu 0 Hz'te 6.0 dB). Sonlu değer yoksa (None, None)."""
    f = np.asarray(freqs, dtype=np.float64)
    m = np.asarray(mag_db, dtype=np.float64)
    ok = np.isfinite(m)
    if len(m) < 2 or len(f) != len(m) or not ok.any():
        return None, None
    hi = int(np.argmax(np.where(ok, m, -np.inf)))
    lo = int(np.argmin(np.where(ok, m, np.inf)))
    dc = np.flatnonzero(f == 0.0)
    if dc.size and ok[dc[0]] and abs(hi - int(dc[0])) <= mainlobe_bins(window, beta):
        top = SpectralPeak(0.0, float(m[dc[0]]), int(dc[0]))
    else:
        top = _refine(f, np.where(ok, m, m[lo]), hi, window, beta)
    return top, SpectralPeak(float(f[lo]), float(m[lo]), lo)


class SpectrumHold:
    """Max-hold / min-hold: her frekans bininde o ana kadar görülen en büyük ve en küçük seviye (dB). Frekans ızgarası
    değişince (FFT boyu, örnekleme hızı, tek / çift taraf, incelemede kısalan aralık) baştan başlar."""

    def __init__(self) -> None:
        self.freqs: np.ndarray | None = None
        self.max: np.ndarray | None = None
        self.min: np.ndarray | None = None
        self.count = 0                                   # biriktirilen spektrum sayısı

    def reset(self) -> None:
        self.freqs = self.max = self.min = None
        self.count = 0

    def update(self, freqs: np.ndarray, mag_db: np.ndarray) -> None:
        f = np.asarray(freqs, dtype=np.float64)
        m = np.asarray(mag_db, dtype=np.float64)
        same = (self.freqs is not None and len(f) == len(self.freqs) and len(f) > 0
                and f[0] == self.freqs[0] and f[-1] == self.freqs[-1])
        if not same:
            self.freqs, self.max, self.min, self.count = f.copy(), m.copy(), m.copy(), 1
            return
        np.fmax(self.max, m, out=self.max)               # fmax / fmin: NaN biriktirilmez
        np.fmin(self.min, m, out=self.min)
        self.count += 1


LEAKAGE_MARGIN_DB = 6.0   # yan lob zarfının bu kadar üstündeki aday gerçek bileşen sayılır
ENVELOPE_BINS = 256       # zarf bu uzaklığa kadar hesaplanır; ötesinde son değer


@lru_cache(maxsize=16)
def _sidelobe_envelope(window: str, beta: float) -> np.ndarray:
    """env[k]: ana lob tepesinden en az k bin uzakta pencere cevabının en büyük değeri (dB, ≤ 0), k = 0..ENVELOPE_BINS.
    512 noktalı pencerenin 32 kat sıfır dolgulu DTFT'sinden (eğrinin bin cinsinden biçimi uzunluktan bağımsıza yakın)."""
    n, pad = 512, 32
    w = _window(WINDOWS.get(window, "hann"), n, beta)
    resp = np.abs(np.fft.rfft(w, n * pad))
    db = 20.0 * np.log10(np.maximum(resp / resp[0], 1e-15))
    tail = np.maximum.accumulate(db[::-1])[::-1]               # i'den sona en büyük
    k = np.minimum(np.arange(ENVELOPE_BINS + 1) * pad, len(tail) - 1)
    return tail[k]


@lru_cache(maxsize=16)
def _scallop_loss(window: str, beta: float) -> tuple[np.ndarray, np.ndarray]:
    """Bin merkezinden d bin uzaktaki tonun tepe bininde okunan kayıp (dB, ≤ 0), d ∈ [−0.5, 0.5]. Pencerenin DTFT'si
    1024 noktalı pencereyle hesaplanır (eğrinin biçimi pencere uzunluğundan bağımsıza yakındır)."""
    n = 1024
    w = _window(WINDOWS.get(window, "hann"), n, beta)
    d = np.linspace(-0.5, 0.5, 201)
    resp = np.abs(np.exp(-2j * np.pi * np.outer(d, np.arange(n)) / n) @ w) / w.sum()
    return d, 20.0 * np.log10(resp)
