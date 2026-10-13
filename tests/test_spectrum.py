"""Genlik spektrumu doğruluğu ve test dalga şekillerinin temizliği."""
import numpy as np
import pytest

from spt.dsp import waveforms as wf
from spt.dsp.spectrum import WINDOWS, amplitude_spectrum

FS, N, AMP = 50_000.0, 4096, 3.0
# İki bin arasındaki tonda pencere başına beklenen en kötü genlik kaybı (scalloping), dB.
# Dikdörtgende teorik sınır −3.92 dB; gerçel sinüsün negatif frekans görüntüsü yüksek yan loblardan
# ~0.02 dB sızdığı için N = 4096'da −3.90 dB ölçülür.
SCALLOP_DB = {"Hann": -1.42, "Hamming": -1.75, "Blackman": -1.10, "Blackman-Harris": -0.83,
              "Flat-top": -0.01, "Kaiser": -1.11, "Dikdörtgen": -3.90}
# Bin ortasındaki gerçel sinüste negatif frekans görüntüsünün yan lob sızıntısı (Kaiser'de en büyük)
EXACT_TOL_DB = 0.005


def _tone(bin_position: float, complex_: bool = False) -> np.ndarray:
    t = np.arange(N) / FS
    w = 2 * np.pi * bin_position * FS / N * t + 0.3
    return AMP * (np.exp(1j * w) if complex_ else np.sin(w))


@pytest.mark.parametrize("window", list(WINDOWS))
def test_bin_centered_tone_reads_exact_amplitude(window):
    _, mag = amplitude_spectrum(_tone(82.0), FS, window)
    assert mag.max() == pytest.approx(20 * np.log10(AMP), abs=EXACT_TOL_DB)


@pytest.mark.parametrize("window", list(WINDOWS))
def test_worst_case_scalloping_matches_window_theory(window):
    _, mag = amplitude_spectrum(_tone(82.5), FS, window)
    assert mag.max() - 20 * np.log10(AMP) == pytest.approx(SCALLOP_DB[window], abs=0.02)


def test_complex_tone_two_sided_is_exact():
    freqs, mag = amplitude_spectrum(_tone(-100.0, complex_=True), FS, "Hann", sides="two")
    assert freqs[np.argmax(mag)] == pytest.approx(-100 * FS / N)
    assert mag.max() == pytest.approx(20 * np.log10(AMP), abs=1e-6)


def test_real_tone_two_sided_shows_half_amplitude_on_both_sides():
    """Gerçel sinüs A·sin(ωt) = (A/2)e^{jωt} + (A/2)e^{−jωt}: her iki tarafta A/2 (−6.02 dB)."""
    freqs, mag = amplitude_spectrum(_tone(82.0), FS, "Hann", sides="two")
    f0 = 82.0 * FS / N
    for f in (f0, -f0):
        i = np.argmin(abs(freqs - f))
        assert mag[i] == pytest.approx(20 * np.log10(AMP / 2), abs=EXACT_TOL_DB)
    assert freqs.min() < 0 < freqs.max()


def test_real_tone_one_sided_folds_to_full_amplitude():
    freqs, mag = amplitude_spectrum(_tone(82.0), FS, "Hann", sides="one")
    assert freqs.min() == 0
    assert mag.max() == pytest.approx(20 * np.log10(AMP), abs=EXACT_TOL_DB)


def test_complex_one_sided_hides_negative_frequencies_without_folding():
    """Karmaşık sinyalde tek taraflı gösterim yalnızca pozitif yarıdır; −f bileşeni +f'ye eklenmez."""
    x = _tone(100.0, complex_=True) + 0.1 * _tone(-300.0, complex_=True)
    freqs, mag = amplitude_spectrum(x, FS, "Hann", sides="one")
    assert freqs.min() >= 0
    assert mag.max() == pytest.approx(20 * np.log10(AMP), abs=EXACT_TOL_DB)
    i = np.argmin(abs(freqs - 300 * FS / N))
    assert mag[i] < 20 * np.log10(0.1 * AMP) - 60  # −300 bileşeni pozitif tarafa taşınmadı


def test_square_wave_fundamental_is_4_over_pi():
    """Genliği A olan kare dalganın temel bileşeni 4A/π'dir (tepe değerinden büyük)."""
    x = AMP * wf.square(wf.phase(np.arange(N), 1000, FS))
    freqs, mag = amplitude_spectrum(x, FS, "Flat-top")
    i = np.argmin(abs(freqs - 1000))
    assert mag[i - 2:i + 3].max() == pytest.approx(20 * np.log10(4 * AMP / np.pi), abs=0.05)


@pytest.mark.parametrize("freq", [1000, 500, 250])
def test_square_wave_matches_integer_reference(freq):
    """Periyot tam sayı örnekse her periyodun ilk yarısı +1, ikinci yarısı −1 olmalı (hatalı örnek yok)."""
    period = int(FS // freq)
    n = np.arange(period * 200)
    reference = np.where((n % period) < period / 2, 1.0, -1.0)
    assert np.array_equal(wf.square(wf.phase(n, freq, FS)), reference)


def test_square_wave_duty_is_exact_for_non_integer_period():
    """1234 Hz / 50 kHz: 25000 örneklik tam döngüde (617 periyot) yüksek ve alçak örnek sayısı eşittir."""
    x = wf.square(wf.phase(np.arange(25_000), 1234, FS))
    assert x.sum() == 0


# ============================================================================ canlı tepe bulma (spectral_peaks)
# Bin arasına düşen tek tonda ölçülen en kötü hatalar (frekans: bin, seviye: dB) + pay. Seviye, parabolik konumda
# pencerenin bin arası kaybıyla düzeltilir; dikdörtgende parabol ana lobu kötü tanımlar (en kötü ~1.5 dB).
PEAK_TOL = {"Hann": (0.03, 0.10), "Hamming": (0.03, 0.10), "Blackman": (0.02, 0.05), "Blackman-Harris": (0.01, 0.02),
            "Flat-top": (0.20, 0.02), "Kaiser": (0.02, 0.05), "Dikdörtgen": (0.25, 1.6)}


@pytest.mark.parametrize("window", list(WINDOWS))
def test_peak_frequency_and_level_between_bins(window):
    from spt.dsp.spectrum import spectral_peaks

    fs, n, amp = 10_000.0, 8192, 0.35
    df = fs / n
    t = np.arange(n) / fs
    rng = np.random.default_rng(3)
    f_tol, l_tol = PEAK_TOL[window]
    for frac in np.linspace(0.0, 0.5, 11):
        f0 = (1007 + frac) * df
        x = amp * np.sin(2 * np.pi * f0 * t + 0.7) + 1e-4 * rng.standard_normal(n)
        peaks = spectral_peaks(*amplitude_spectrum(x, fs, window), 3, window)
        assert len(peaks) == 1, (window, frac, peaks)
        assert abs(peaks[0].freq - f0) <= f_tol * df
        assert abs(peaks[0].level - 20 * np.log10(amp)) <= l_tol


def test_peaks_sorted_limited_and_dc_excluded():
    """Şebeke benzeri sinyal: DC ofset raporlanmaz, tepeler seviyeye göre sıralı, sayı sınırı uygulanır."""
    from spt.dsp.spectrum import spectral_peaks

    fs, n = 10_000.0, 8192
    t = np.arange(n) / fs
    x = (0.5 + np.sin(2 * np.pi * 50 * t) + 0.08 * np.sin(2 * np.pi * 150 * t)
         + 0.35 * np.sin(2 * np.pi * 1230 * t) + 0.003 * np.random.default_rng(1).standard_normal(n))
    freqs, mag = amplitude_spectrum(x, fs, "Hann")
    peaks = spectral_peaks(freqs, mag, 5, "Hann")
    assert [round(p.freq) for p in peaks] == [50, 1230, 150]
    for p, amp in zip(peaks, (1.0, 0.35, 0.08)):
        assert p.level == pytest.approx(20 * np.log10(amp), abs=0.1)
    assert [round(p.freq) for p in spectral_peaks(freqs, mag, 2, "Hann")] == [50, 1230]
    assert spectral_peaks(freqs, mag, 0, "Hann") == []


def test_no_peaks_in_pure_noise_and_no_roundoff_peaks():
    from spt.dsp.spectrum import spectral_peaks

    fs, n = 10_000.0, 8192
    for seed in range(20):     # gürültünün kendi dalgalanması tepe sayılmaz (taban + 15 dB)
        x = 0.003 * np.random.default_rng(seed).standard_normal(n)
        assert spectral_peaks(*amplitude_spectrum(x, fs, "Hann"), 10, "Hann") == []
    t = np.arange(4000) / 1000.0  # gürültüsüz: yuvarlama kırıntıları (≈ −280 dB) tepe sayılmaz
    x = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 120 * t)
    for window in ("Hann", "Dikdörtgen", "Flat-top"):
        peaks = spectral_peaks(*amplitude_spectrum(x, 1000.0, window), 10, window)
        assert [(round(p.freq), round(p.level, 1)) for p in peaks] == [(50, 0.0), (120, -6.0)]


def test_two_sided_real_tone_peaks_at_both_sides():
    from spt.dsp.spectrum import TWO_SIDED, spectral_peaks

    t = np.arange(N) / FS
    x = AMP * np.sin(2 * np.pi * 3000.0 * t)
    peaks = spectral_peaks(*amplitude_spectrum(x, FS, "Hann", sides=TWO_SIDED), 4, "Hann")
    assert sorted(round(p.freq) for p in peaks) == [-3000, 3000]
    assert all(abs(p.level - (20 * np.log10(AMP) - 6.02)) < 0.1 for p in peaks)


# ============================================================================ anlık max / min ve max / min-hold
def test_extrema_always_reported_and_max_refined_like_peaks():
    """Max / min eşiksizdir: saf gürültüde de döner. Max, tepeyle aynı bin altı düzeltmeyi alır; min binin kendisi."""
    from spt.dsp.spectrum import spectral_extrema, spectral_peaks

    fs, n = 10_000.0, 8192
    df = fs / n
    t = np.arange(n) / fs
    rng = np.random.default_rng(5)
    noise = 0.003 * rng.standard_normal(n)
    freqs, mag = amplitude_spectrum(noise, fs, "Hann")
    hi, lo = spectral_extrema(freqs, mag)
    assert hi.level == pytest.approx(mag.max(), abs=1.5) and lo.level == mag.min()
    assert lo.freq == freqs[lo.index] and spectral_peaks(freqs, mag, 5) == []
    f0 = (1007 + 0.37) * df                                    # bin arası ton
    x = 0.35 * np.sin(2 * np.pi * f0 * t) + noise
    freqs, mag = amplitude_spectrum(x, fs, "Hann")
    hi, lo = spectral_extrema(freqs, mag)
    peak = spectral_peaks(freqs, mag, 1)[0]
    assert (hi.freq, hi.level, hi.index) == (peak.freq, peak.level, peak.index)
    assert abs(hi.freq - f0) <= 0.03 * df and hi.level == pytest.approx(20 * np.log10(0.35), abs=0.1)
    sl = slice(0, 500)                                         # tondan uzak aralık: aralığın kendi max / min'i
    hi2, lo2 = spectral_extrema(freqs[sl], mag[sl])
    assert hi2.index == int(np.argmax(mag[sl])) and lo2.level == mag[sl].min()


def test_extrema_includes_dc_and_handles_edges():
    from spt.dsp.spectrum import spectral_extrema

    fs, n = 1000.0, 1024
    x = 2.0 + 0.1 * np.sin(2 * np.pi * 100 * np.arange(n) / fs)
    hi, _ = spectral_extrema(*amplitude_spectrum(x, fs, "Hann"))
    assert hi.index == 0 and hi.freq == 0.0 and hi.level == pytest.approx(20 * np.log10(2.0), abs=1e-6)
    assert spectral_extrema(np.array([1.0]), np.array([0.0])) == (None, None)
    assert spectral_extrema(np.arange(4.0), np.full(4, np.nan)) == (None, None)
    hi, lo = spectral_extrema(np.arange(4.0), np.array([np.nan, -3.0, -1.0, np.nan]))
    assert (hi.index, lo.index) == (2, 1)


def test_spectrum_hold_tracks_max_and_min_and_restarts_on_new_grid():
    from spt.dsp.spectrum import SpectrumHold

    hold = SpectrumHold()
    f = np.linspace(0, 500, 6)
    hold.update(f, np.array([0.0, -10, -20, -30, -40, -50]))
    hold.update(f, np.array([-5.0, 0, -25, np.nan, -35, -60]))
    assert hold.count == 2
    assert hold.max.tolist() == [0.0, 0, -20, -30, -35, -50]
    assert hold.min.tolist() == [-5.0, -10, -25, -30, -40, -60]   # NaN birikime girmez
    hold.update(np.linspace(0, 500, 11), np.zeros(11))            # FFT boyu değişti: baştan
    assert hold.count == 1 and len(hold.max) == 11 and hold.min.tolist() == [0.0] * 11
    hold.reset()
    assert hold.max is None and hold.count == 0
    # Genliği değişen ton: max-hold en büyük, min-hold en küçük genliği okur
    fs, n = 10_000.0, 4096
    t = np.arange(n) / fs
    for amp in (0.2, 1.0, 0.5):
        hold.update(*amplitude_spectrum(amp * np.sin(2 * np.pi * 1000 * t), fs, "Flat-top"))
    i = int(np.argmin(np.abs(hold.freqs - 1000)))
    assert hold.max[i] == pytest.approx(0.0, abs=0.02) and hold.min[i] == pytest.approx(20 * np.log10(0.2), abs=0.02)
