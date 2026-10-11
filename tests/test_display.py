"""Ekran seyreltmesi ve analiz istatistikleri (arayüzden bağımsız)."""
import time

import numpy as np

from spt.dsp.display import decimate
from spt.gui.widgets.inspector import dominant_frequency, time_stats


def test_decimated_extrema_sit_at_their_true_sample_indices():
    # Denetim: min / max blok başına ve ortasına çiziliyordu; tıklanan tepe başka bir örneği gösteriyordu
    rng = np.random.default_rng(1)
    y = rng.standard_normal(100_003)
    x, v, k = decimate([(5000, y)], 1000)
    assert k > 1 and len(x) == len(v)
    idx = x.astype(np.int64) - 5000
    assert np.array_equal(x, idx + 5000.0)          # tam sayı örnek indeksleri
    assert np.array_equal(v, y[idx])                # her nokta gerçekten o örneğin değeri
    assert np.all(np.diff(x) > 0)                   # zaman sırasında
    assert v.max() == y.max() and v.min() == y.min()  # tepeler korunur


def test_decimate_without_peak_and_short_input():
    y = np.arange(10.0)
    x, v, k = decimate([(0, y)], 1000)
    assert k == 1 and np.array_equal(x, np.arange(10.0)) and np.array_equal(v, y)
    x, v, k = decimate([(0, np.arange(1000.0))], 10, peak=False)
    assert k == 100 and np.array_equal(v, np.arange(0, 1000, 100.0))


def test_dominant_frequency_requires_two_periods_and_is_fast_on_prime_lengths():
    fs = 10_000.0
    t = np.arange(200) / fs                        # 20 ms: 50 Hz'in 1 periyodu
    assert dominant_frequency(np.sin(2 * np.pi * 50 * t), fs) is None
    t = np.arange(4_999_999) / fs                  # asal uzunluk: eskiden 5–8 s sürüyordu
    start = time.perf_counter()
    f = dominant_frequency(np.sin(2 * np.pi * 50 * t), fs)
    assert time.perf_counter() - start < 1.0 and abs(f - 50.0) < 0.01
    stats = dict(time_stats(np.sin(2 * np.pi * 50 * np.arange(200) / fs), 0.0, fs, "V"))
    assert stats["Baskın frekans"].startswith("—")
