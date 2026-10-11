"""FFT doğrulaması: örnek kayıtlardaki bilinen bileşenler, arayüzle aynı yoldan ölçülüp teoriyle karşılaştırılır.

Kullanım:  .venv\\Scripts\\python tools\\verify_fft_samples.py

Yol (arayüzdeki Canlı İzleme ile aynı): SPTF çerçeveleri → StreamStore → paylaşımlı tampon →
kayıpsız aralık → fiziksel birim → pencereli FFT (`amplitude_spectrum`).
Çerçeveler, samples_signal_data/ dosyalarını üreten fonksiyonlarla bellekte üretilir (içerik dosyalarla aynıdır).

Genlik ölçümleri Flat-top penceresiyle yapılır (tepe kaybı ≤ 0.01 dB); Hann sonucu da bilgi olarak verilir.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import make_sample_recordings as samples  # noqa: E402

from spt.console import utf8_stdio  # noqa: E402
from spt.core.ring import SharedRing  # noqa: E402
from spt.core.stream_store import StreamStore  # noqa: E402
from spt.dsp import waveforms as wf  # noqa: E402
from spt.dsp.convert import physical_column  # noqa: E402
from spt.dsp.spectrum import amplitude_spectrum  # noqa: E402

AMPLITUDE_TOL_DB = 0.1
N_FFT = 8192


@dataclass
class Check:
    sample: str
    channel: str
    what: str
    expected: float
    measured: float
    hann: float | None
    unit: str
    tol: float

    @property
    def ok(self) -> bool:
        return abs(self.measured - self.expected) <= self.tol


class Stream:
    def __init__(self, info, ring: SharedRing):
        self.desc = info.descriptor
        self.ring = ring
        self.fs = self.desc.sample_rate

    def channel(self, name: str) -> int:
        return next(i for i, c in enumerate(self.desc.channels) if c.name == name)

    def window(self, ch: str, n: int = N_FFT, end: int | None = None) -> np.ndarray:
        rng = self.ring.latest_clean_range(n, end)
        assert rng is not None, "kayıpsız pencere bulunamadı"
        _, raw = self.ring.read(*rng)
        i = self.channel(ch)
        return physical_column(raw[:, i], self.desc.sample_type, self.desc.channels[i])


def load(sample_fn, seed: int) -> tuple[dict[str, Stream], StreamStore]:
    """Örnek kaydı bellekte üretip arayüzle aynı akış katmanından geçirir."""
    rng = np.random.default_rng(seed)
    frames = sorted(f for spec in sample_fn(np.random.default_rng(seed)) for f in samples.stream_frames(spec, rng))
    events = []
    store = StreamStore(events.append, frames[0][0], ring_seconds=120.0)
    for rx, frame in frames:
        store.handle_frame(memoryview(frame), 0, len(frame), "örnek", rx)
    streams = {e[1].descriptor.name: Stream(e[1], SharedRing.attach(e[1].shm_name, e[1].descriptor.sample_type.dtype))
               for e in events if e[0] == "stream"}
    return streams, store


def level(x: np.ndarray, fs: float, freq: float, window: str, sides: str = "one") -> float:
    """Bileşenin genlik okuması (dB). DC yalnızca DC bininden okunur: tek taraflı spektrumda DC'nin komşu
    binlere sızan kısmı ×2 ölçeklendiği için (özellikle geniş ana loblu Flat-top'ta) 1–4. binler DC'yi
    ~6 dB yüksek gösterir; DC'nin doğru tahmini 0 Hz binidir."""
    freqs, mag = amplitude_spectrum(x, fs, window, sides=sides)
    i = int(np.argmin(np.abs(freqs - freq)))
    if freq == 0:
        return float(mag[i])
    return float(mag[max(i - 3, 0):i + 4].max())


def db(v: float) -> float:
    return 20 * np.log10(v)


def amplitude_checks(sample: str, stream: Stream, channel: str, components, end=None, unit="dB",
                     sides: str = "one") -> list[Check]:
    x = stream.window(channel, end=end)
    out = []
    for freq, amp in components:
        label = "DC" if freq == 0 else f"{freq:g} Hz"
        if sides == "two":
            label += " (çift t.)"
        out.append(Check(sample, channel, label, db(abs(amp)), level(x, stream.fs, freq, "Flat-top", sides),
                         level(x, stream.fs, freq, "Hann", sides), unit, AMPLITUDE_TOL_DB))
    return out


def discrete_harmonics(wave: np.ndarray, ks) -> list[float]:
    """Tek periyotluk ayrık dalganın k. harmonik genliği (sızıntısız gerçek değer)."""
    spec = np.fft.fft(wave)
    return [2 * abs(spec[k]) / len(wave) for k in ks]


def peak_frequency(x: np.ndarray, fs: float, lo: float, hi: float, count: int) -> list[float]:
    """[lo, hi] aralığındaki en güçlü `count` yerel tepenin frekansı (parabolik interpolasyonla)."""
    freqs, mag = amplitude_spectrum(x, fs, "Hann")
    idx = np.flatnonzero((freqs >= lo) & (freqs <= hi))
    local = [i for i in idx if mag[i] >= mag[i - 1] and mag[i] >= mag[i + 1]]
    best = sorted(local, key=lambda i: mag[i], reverse=True)[:count]
    out = []
    for i in sorted(best):
        a, b, c = mag[i - 1], mag[i], mag[i + 1]
        shift = 0.5 * (a - c) / (a - 2 * b + c)
        out.append(freqs[i] + shift * (freqs[1] - freqs[0]))
    return out


def run_checks() -> list[Check]:
    checks: list[Check] = []
    fns = dict((name[:2], fn) for name, fn in samples.SAMPLES)

    s, store = load(fns["01"], 1)
    st = s["sebeke-olcum"]
    checks += amplitude_checks("01 şebeke", st, "faz_A", [(50, 1.2), (150, 0.18), (1200, 0.35)])
    checks += amplitude_checks("01 şebeke", st, "faz_B", [(50, 0.8), (3100, 0.2)])
    # Çift taraflı gösterim: gerçel sinüs ±f'de A/2 olarak görünür
    checks += amplitude_checks("01 şebeke", st, "faz_A", [(50, 0.6), (-50, 0.6), (1200, 0.175), (-1200, 0.175)],
                               sides="two")
    store.close()

    s, store = load(fns["02"], 2)
    st = s["frekans-tarama"]
    # Logaritmik tarama: f(t) = 20·400^(t/4). Son 1024 örneklik pencerenin ortasındaki anlık frekans.
    end = st.ring.write_count
    t_mid = ((end - 512) / st.fs) % 4.0
    f_inst = 20 * 400 ** (t_mid / 4)
    (f_peak,) = peak_frequency(st.window("chirp", 1024), st.fs, 10, 9000, 1)
    checks.append(Check("02 chirp", "chirp", "anlık frekans", f_inst, f_peak, None, "Hz", 0.05 * f_inst))
    store.close()

    s, store = load(fns["03"], 3)
    st = s["dalga-sekilleri"]
    n = np.arange(int(st.fs))
    for ch, freq, wave, ks in (
            ("kare_1kHz", 1000, 3.0 * wf.square(wf.phase(n[:50], 1000, st.fs)), (1, 3, 5, 7)),
            ("ucgen_500Hz", 500, 2.0 * wf.triangle(wf.phase(n[:100], 500, st.fs)), (1, 3, 5)),
            ("testere_250Hz", 250, 1.0 * wf.sawtooth(wf.phase(n[:200], 250, st.fs)), (1, 2, 3, 4))):
        truth = discrete_harmonics(wave, ks)
        checks += amplitude_checks("03 dalga şekilleri", st, ch, [(k * freq, a) for k, a in zip(ks, truth)])
    store.close()

    s, store = load(fns["04"], 4)
    st = s["imu"]
    steady_end = int(21.0 * st.fs)  # motor 11–22 s arası tam hızda
    checks += amplitude_checks("04 IMU", st, "ivme_x", [(120, 0.35), (240, 0.12)], end=steady_end)
    checks += amplitude_checks("04 IMU", st, "ivme_y", [(120, 0.21), (240, 0.072)], end=steady_end)
    checks += amplitude_checks("04 IMU", st, "ivme_z", [(0, 1.0), (120, 0.14)], end=steady_end)
    checks += amplitude_checks("04 IMU", st, "jiro_z", [(0, 180.0)], end=steady_end)
    store.close()

    s, store = load(fns["05"], 5)
    st = s["ses-24bit"]
    notes = [440.0, 554.37, 659.25, 880.0]
    found = peak_frequency(st.window("sol", 8192, end=int(5.0 * st.fs)), st.fs, 300, 1000, 4)
    bin_hz = st.fs / 8192
    for f_true, f_meas in zip(notes, found):
        checks.append(Check("05 ses int24", "sol", f"nota {f_true:g} Hz", f_true, f_meas, None, "Hz", bin_hz / 2))
    store.close()

    s, store = load(fns["06"], 6)
    iq = s["iq-alici"]
    checks += amplitude_checks("06 I/Q", iq, "iq", [(125_000, 0.5), (-300_000, 0.05)], sides="two")
    # Tek taraflı: karmaşık sinyalde yalnızca pozitif yarı; +125 kHz tam genlik, −300 kHz gösterilmez
    checks += amplitude_checks("06 I/Q", iq, "iq", [(125_000, 0.5)], sides="one")
    freqs, _ = amplitude_spectrum(iq.window("iq"), iq.fs, "Hann", sides="one")
    checks.append(Check("06 I/Q", "iq", "tek t. en düşük f", 0.0, float(freqs.min()), None, "Hz", 0.0))
    store.close()

    s, store = load(fns["07"], 7)
    checks += amplitude_checks("07 EKG", s["ekg-12bit"], "ekg", [(50, 0.25)])
    store.close()

    s, store = load(fns["08"], 8)
    checks += amplitude_checks("08 kayıplı akış", s["kayipli-akis"], "sinyal", [(100, 1.0)])
    store.close()

    s, store = load(fns["09"], 9)
    checks += amplitude_checks("09 çoklu akış", s["titresim"], "rulman_x", [(29.5, 0.2)])
    checks += amplitude_checks("09 çoklu akış", s["titresim"], "rulman_y", [(29.5, 0.14)])
    store.close()
    return checks


def gap_effect() -> tuple[float, float]:
    """08 kaydında, içinde kayıp olan pencere ile kayıpsız pencerenin 100 Hz okuması (dB)."""
    s, store = load(dict((n[:2], f) for n, f in samples.SAMPLES)["08"], 8)
    st = s["kayipli-akis"]
    end = st.ring.write_count
    gaps = st.ring.gaps_between(0, end)
    gs, ge = gaps[-1]
    _, raw = st.ring.read(gs - N_FFT // 2, gs + N_FFT // 2)  # kaybın tam ortasında
    i = st.channel("sinyal")
    dirty = physical_column(raw[:, i], st.desc.sample_type, st.desc.channels[i])
    result = level(dirty, st.fs, 100, "Flat-top"), level(st.window("sinyal"), st.fs, 100, "Flat-top")
    store.close()
    return result


def main() -> int:
    utf8_stdio()
    checks = run_checks()
    print(f"{'Kayıt':20s} {'Kanal':14s} {'Bileşen':16s} {'Beklenen':>10s} {'Flat-top':>10s} {'Fark':>8s} "
          f"{'Hann':>9s}  Sonuç")
    for c in checks:
        hann = f"{c.hann:+8.2f}" if c.hann is not None else "       —"
        print(f"{c.sample:20s} {c.channel:14s} {c.what:16s} {c.expected:+10.2f} {c.measured:+10.2f} "
              f"{c.measured - c.expected:+8.3f} {hann}  {'✓' if c.ok else '✗'}  ({c.unit})")
    failed = [c for c in checks if not c.ok]
    print(f"\n{len(checks) - len(failed)} / {len(checks)} kontrol geçti "
          f"(genlik toleransı ±{AMPLITUDE_TOL_DB} dB, frekans toleransı yarım bin / %5).")
    dirty, clean = gap_effect()
    print(f"\nKayıp etkisi (08, 100 Hz, 0 dB beklenir): kayıp içeren pencere {dirty:+.2f} dB, "
          f"kayıpsız pencere {clean:+.2f} dB")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
