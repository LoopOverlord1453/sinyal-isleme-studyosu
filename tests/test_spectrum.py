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
