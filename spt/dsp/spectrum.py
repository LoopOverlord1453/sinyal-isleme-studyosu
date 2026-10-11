"""Genlik spektrumu (pencereli FFT)."""
from __future__ import annotations

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
