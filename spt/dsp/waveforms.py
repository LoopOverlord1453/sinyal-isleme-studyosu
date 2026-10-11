"""Periyodik test dalga şekilleri, faz tamsayı aritmetiğiyle hesaplanır.

`scipy.signal.square(2π·f·t)` gibi kayan noktalı faz kullanımında, geçişin tam bir örneğe denk geldiği
durumlarda (ör. fs = 50·f) `mod(2π·k, 2π)` bazen 2π'nin hemen altında çıkar ve örnek yanlış yarıya düşer.
Bu tek örneklik hatalar spektrumda geniş bantlı sahte bir gürültü tabanı oluşturur. Burada faz, frekans ve
örnekleme hızı tamsayıysa tam (hatasız) hesaplanır.
"""
from __future__ import annotations

import numpy as np


def phase(n: np.ndarray, freq: float, fs: float) -> np.ndarray:
    """n. örnekteki faz, [0, 1) aralığında (1 = bir periyot)."""
    if float(freq).is_integer() and float(fs).is_integer():
        return (np.asarray(n, dtype=np.int64) * int(freq) % int(fs)) / int(fs)
    return np.mod(np.asarray(n, dtype=np.float64) * (freq / fs), 1.0)


def cycle_phase(n: np.ndarray, cycles: int, length: int) -> np.ndarray:
    """`length` örneklik tabloda tam `cycles` periyot için faz, [0, 1)."""
    return (np.asarray(n, dtype=np.int64) * cycles % length) / length


def sine(ph: np.ndarray) -> np.ndarray:
    return np.sin(2 * np.pi * ph)


def square(ph: np.ndarray, duty: float = 0.5) -> np.ndarray:
    return np.where(ph < duty, 1.0, -1.0)


def triangle(ph: np.ndarray) -> np.ndarray:
    """Fazda 0'da −1, 0.5'te +1 (scipy `sawtooth(w, 0.5)` ile aynı biçim)."""
    return 1.0 - 4.0 * np.abs(ph - 0.5)


def sawtooth(ph: np.ndarray) -> np.ndarray:
    """−1'den +1'e yükselen testere."""
    return 2.0 * ph - 1.0
