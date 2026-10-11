"""Ekran için seyreltme: milyonlarca örneği piksel sayısı kadar noktaya indirir.

Tepe korumalı (min/max) seyreltme, ham tamsayı veri üzerinde kopyasız çalışır; fiziksel birime çevirme
yalnızca kalan birkaç bin nokta için yapılır. Böylece gigabit hızlarındaki akışlar da akıcı çizilir.
"""
from __future__ import annotations

import numpy as np

MAX_PEAK_SAMPLES = 4_000_000  # bundan uzun pencerelerde min/max yerine adım atlayarak seyreltilir


def decimate(segments: list[tuple[int, np.ndarray]], columns: int, peak: bool = True
             ) -> tuple[np.ndarray, np.ndarray, int]:
    """Bitişik parçalardan (mutlak indeks, 1-B değer görünümü) ekran noktaları üretir.

    Dönüş: (x: mutlak örnek indeksleri, y: değerler, blok boyu k).
    `peak=True` iken her blok için min ve max iki nokta olarak verilir (k > 1 ise); noktalar bloğun başına / ortasına
    değil, min ve max örneklerin GERÇEK indekslerine (zaman sırasıyla) konur — böylece çizgi doğru yerden geçer ve
    grafikte tıklanan tepe, tampondaki o örneğin kendisidir.
    """
    total = sum(len(v) for _, v in segments)
    if total == 0:
        return np.empty(0), np.empty(0), 1
    k = max(1, -(-total // max(columns, 1)))
    xs, ys = [], []
    for start, values in segments:
        n = len(values)
        if k == 1:
            xs.append(start + np.arange(n, dtype=np.float64))
            ys.append(values)
            continue
        if not peak or total > MAX_PEAK_SAMPLES:
            idx = np.arange(0, n, k)
            xs.append(start + idx.astype(np.float64))
            ys.append(values[::k])
            continue
        m = n // k
        if m:
            x, y = _block_extrema(values[:m * k].reshape(m, k))
            xs.append(start + np.arange(m, dtype=np.float64).repeat(2) * k + x)
            ys.append(y)
        if n - m * k:
            x, y = _block_extrema(values[m * k:][None, :])
            xs.append(start + m * k + x)
            ys.append(y)
    return np.concatenate(xs), np.concatenate(ys), k


def _block_extrema(blocks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Her satırın min ve max'ı, blok içi konumlarıyla, zaman sırasında: (konumlar, değerler) — 2·m eleman."""
    rows = np.arange(len(blocks))
    lo_i, hi_i = blocks.argmin(axis=1), blocks.argmax(axis=1)
    lo, hi = blocks[rows, lo_i], blocks[rows, hi_i]
    first = lo_i <= hi_i
    x = np.column_stack((np.where(first, lo_i, hi_i), np.where(first, hi_i, lo_i))).ravel().astype(np.float64)
    y = np.column_stack((np.where(first, lo, hi), np.where(first, hi, lo))).ravel()
    return x, y


def mask_gaps(x: np.ndarray, y: np.ndarray, gaps: list[tuple[int, int]], k: int, valid_from: int) -> np.ndarray:
    """Kayıp aralıklarına ve üzerine yazılmış bölgeye denk gelen noktaları NaN yapar (çizgi kesilir)."""
    y = y.astype(np.float64, copy=True)
    for gs, ge in gaps:
        y[(x < ge) & (x + k > gs)] = np.nan
    if valid_from:
        y[x < valid_from] = np.nan
    return y
