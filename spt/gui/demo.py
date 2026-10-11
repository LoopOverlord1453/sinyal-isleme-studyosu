"""İskelet aşamasında grafikleri dolduran örnek sinyaller.

Gerçek veri akışı (Aşama 1: UDP alıcı) bağlandığında bu modülün kullanımı kalkacak.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal

FS = 10_000.0  # Hz


@dataclass(frozen=True)
class DemoChannel:
    name: str
    unit: str
    data: np.ndarray


def time_signals(duration: float = 1.0) -> tuple[np.ndarray, list[DemoChannel]]:
    """İki kanal: 50 Hz şebeke + yüksek frekans bileşen + gürültü."""
    t = np.arange(int(FS * duration)) / FS
    rng = np.random.default_rng(7)
    ch0 = 1.2 * np.sin(2 * np.pi * 50 * t) + 0.35 * np.sin(2 * np.pi * 1200 * t) + 0.01 * rng.standard_normal(t.size)
    ch1 = (0.8 * np.sin(2 * np.pi * 50 * t + np.pi / 3) + 0.2 * np.sin(2 * np.pi * 3100 * t)
           + 0.004 * rng.standard_normal(t.size))
    return t, [DemoChannel("ch0", "V", ch0), DemoChannel("ch1", "V", ch1)]


def amplitude_spectrum_dbv(x: np.ndarray, n_fft: int = 8192) -> tuple[np.ndarray, np.ndarray]:
    """Hann pencereli, genlik düzeltmeli tek taraflı spektrum (dBV)."""
    seg = x[:n_fft]
    win = np.hanning(seg.size)
    spec = np.fft.rfft(seg * win)
    amp = 2.0 * np.abs(spec) / win.sum()
    freqs = np.fft.rfftfreq(seg.size, d=1.0 / FS)
    return freqs, 20.0 * np.log10(np.maximum(amp, 1e-9))


@dataclass(frozen=True)
class FilterResponse:
    freqs: np.ndarray
    magnitude_db: np.ndarray
    phase_deg: np.ndarray
    group_delay: np.ndarray
    zeros: np.ndarray
    poles: np.ndarray


def butterworth_response(order: int = 4, cutoff: float = 1000.0) -> FilterResponse:
    sos = signal.butter(order, cutoff, btype="low", fs=FS, output="sos")
    w, h = signal.sosfreqz(sos, worN=2048, fs=FS)
    b, a = signal.sos2tf(sos)
    _, gd = signal.group_delay((b, a), w=w, fs=FS)
    z, p, _ = signal.sos2zpk(sos)
    return FilterResponse(
        freqs=w,
        magnitude_db=20 * np.log10(np.maximum(np.abs(h), 1e-12)),
        phase_deg=np.degrees(np.unwrap(np.angle(h))),
        group_delay=gd,
        zeros=z,
        poles=p,
    )


def measurements(channels: list[DemoChannel]) -> dict[str, list[str]]:
    """Ölçüm paneli için biçimlendirilmiş değerler (satır adı → kanal başına metin)."""
    rows: dict[str, list[str]] = {"RMS": [], "Tepe-tepe": [], "Ortalama": [], "Baskın frekans": [], "Crest faktörü": []}
    for ch in channels:
        x = ch.data
        rms = float(np.sqrt(np.mean(x ** 2)))
        freqs, mag = amplitude_spectrum_dbv(x)
        peak_f = float(freqs[np.argmax(mag[1:]) + 1])
        rows["RMS"].append(f"{rms:.3f} {ch.unit}")
        rows["Tepe-tepe"].append(f"{np.ptp(x):.3f} {ch.unit}")
        rows["Ortalama"].append(f"{np.mean(x) * 1e3:+.2f} m{ch.unit}")
        rows["Baskın frekans"].append(f"{peak_f:.1f} Hz")
        rows["Crest faktörü"].append(f"{np.max(np.abs(x)) / rms:.2f}")
    return rows
