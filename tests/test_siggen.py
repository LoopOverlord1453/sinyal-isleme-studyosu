"""Sinyal üreteci: dalga şekli doğruluğu, formül güvenliği, tablo/DDS/nicemleme, ölçüm teorisi, zamanlayıcı ve
üretilen C kodunun Python modeliyle bit düzeyinde aynı çıktı verdiği (host derleyiciyle) doğrulanır."""
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path

import numpy as np
import pytest
from scipy import signal

from spt.codegen import dac_c
from spt.dsp import siggen as sg
from spt.dsp.siggen import DacSpec, SignalError, WaveSpec

DAC12 = DacSpec(12, 3.3)
SCRATCH_ROOT = Path(tempfile.gettempdir()) / "spt_siggen"


@pytest.fixture
def scratch():
    d = SCRATCH_ROOT / f"t_{uuid.uuid4().hex[:10]}"
    d.mkdir(parents=True)
    yield d
    shutil.rmtree(d, ignore_errors=True)


def _tone_amp(x: np.ndarray, fs: float, f: float) -> float:
    """Tam kutuya düşen tonun genliği (pencere yok)."""
    n = len(x)
    X = np.fft.rfft(x) / n
    return 2 * abs(X[int(round(f * n / fs))])


def _inst_freq(x: np.ndarray, fs: float) -> np.ndarray:
    ph = np.unwrap(np.angle(signal.hilbert(x)))
    return np.diff(ph) * fs / (2 * np.pi)


# ============================================================================ dalga şekilleri
def test_sine_frequency_amplitude_offset():
    fs, f = 48_000.0, 1000.0
    x = sg.synthesize(WaveSpec("sine", freq=f, amplitude=1.2, offset=1.65), fs, 4800)
    spec = np.abs(np.fft.rfft(x - x.mean()))
    assert np.argmax(spec) * fs / len(x) == pytest.approx(f)
    assert _tone_amp(x, fs, f) == pytest.approx(1.2, abs=1e-9)
    assert x.mean() == pytest.approx(1.65, abs=1e-12)
    assert x[0] == pytest.approx(1.65)                      # sinüs fazı 0'da başlar


def test_phase_offset_shifts_waveform():
    fs = 48_000.0
    a = sg.synthesize(WaveSpec("sine", freq=1000.0, phase_deg=90.0), fs, 48)
    assert a[0] == pytest.approx(1.0)
    b = sg.synthesize(WaveSpec("square", freq=1000.0, phase_deg=180.0), fs, 48)
    assert np.all(b[:24] == -1.0) and np.all(b[24:] == 1.0)


@pytest.mark.parametrize("duty", [0.1, 0.3, 0.5, 0.75])
def test_square_duty(duty):
    x = sg.synthesize(WaveSpec("square", freq=1000.0, duty=duty), 100_000.0, 10_000)
    assert set(np.unique(x)) <= {-1.0, 1.0}
    assert np.mean(x > 0) == pytest.approx(duty, abs=1e-12)


@pytest.mark.parametrize("sym", [0.2, 0.5, 0.9])
def test_triangle_symmetry(sym):
    x = sg.synthesize(WaveSpec("triangle", freq=100.0, symmetry=sym), 100_000.0, 1000)
    assert np.argmax(x) == pytest.approx(sym * 1000, abs=1)
    assert x[0] == pytest.approx(-1.0) and x.max() == pytest.approx(1.0, abs=1e-12)
    rising = np.diff(x[: int(sym * 1000)])
    assert np.all(rising > 0)


def test_sawtooth_direction():
    up = sg.synthesize(WaveSpec("sawtooth", freq=100.0, direction="rising"), 10_000.0, 100)
    down = sg.synthesize(WaveSpec("sawtooth", freq=100.0, direction="falling"), 10_000.0, 100)
    assert np.all(np.diff(up) > 0) and np.all(np.diff(down) < 0)
    assert up[0] == -1.0 and down[0] == 1.0


def test_pulse_width_and_baseline():
    x = sg.synthesize(WaveSpec("pulse", freq=1000.0, width=1e-4, amplitude=3.0, offset=0.5), 1e6, 10_000)
    assert set(np.unique(x)) == {0.5, 3.5}
    assert np.sum(x[:1000] > 1) == 100
    with pytest.raises(SignalError):
        sg.synthesize(WaveSpec("pulse", freq=1000.0, width=2e-3), 1e6, 10)


def test_trapezoid_fractions():
    x = sg.synthesize(WaveSpec("trapezoid", freq=10.0, rise=0.1, high=0.4, fall=0.1, low=0.4), 10_000.0, 1000)
    assert np.sum(np.isclose(x, 1.0)) == pytest.approx(400, abs=2)
    assert np.sum(np.isclose(x, -1.0)) == pytest.approx(400, abs=2)
    assert x[50] == pytest.approx(0.0, abs=1e-12)            # yükselme ortası


def test_chirp_instantaneous_frequency():
    fs = 20_000.0
    lin = sg.synthesize(WaveSpec("chirp", freq=100.0, f1=1000.0, sweep_time=1.0), fs, 20_000)
    fi = _inst_freq(lin, fs)
    assert np.median(fi[9_900:10_100]) == pytest.approx(550.0, abs=2.0)
    assert np.median(fi[4_900:5_100]) == pytest.approx(325.0, abs=2.0)
    log = sg.synthesize(WaveSpec("chirp", freq=100.0, f1=1000.0, sweep_time=1.0, chirp_method="log"), fs, 20_000)
    fi = _inst_freq(log, fs)
    assert np.median(fi[9_900:10_100]) == pytest.approx(math.sqrt(100 * 1000), rel=0.01)


def _slope_db_per_octave(x: np.ndarray, fs: float) -> float:
    f, p = signal.welch(x, fs, nperseg=4096)
    sel = (f > 100) & (f < 10_000)
    return float(np.polyfit(np.log2(f[sel]), 10 * np.log10(p[sel]), 1)[0])


def test_noise_spectra_and_seed():
    fs = 48_000.0
    pink = sg.synthesize(WaveSpec("noise_pink", seed=3), fs, 1 << 17)
    white = sg.synthesize(WaveSpec("noise_white", seed=3), fs, 1 << 17)
    assert _slope_db_per_octave(pink, fs) == pytest.approx(-3.01, abs=0.3)
    assert _slope_db_per_octave(white, fs) == pytest.approx(0.0, abs=0.3)
    assert pink.std() == pytest.approx(1 / 3, rel=1e-9) and white.std() == pytest.approx(1 / 3, rel=0.02)
    again = sg.synthesize(WaveSpec("noise_pink", seed=3), fs, 1 << 17)
    other = sg.synthesize(WaveSpec("noise_pink", seed=4), fs, 1 << 17)
    assert np.array_equal(pink, again) and not np.allclose(pink, other)
    uni = sg.synthesize(WaveSpec("noise_white", distribution="uniform", seed=1), fs, 10_000)
    assert uni.min() >= -1 and uni.max() <= 1


def test_multitone_am_fm():
    fs = 100_000.0
    mt = sg.synthesize(WaveSpec("multitone", tones=((1000.0, 0.5, 0.0), (3000.0, 0.25, 90.0))), fs, 10_000)
    assert _tone_amp(mt, fs, 1000) == pytest.approx(0.5) and _tone_amp(mt, fs, 3000) == pytest.approx(0.25)
    am = sg.synthesize(WaveSpec("am", freq=10_000.0, mod_freq=1000.0, depth=0.5), fs, 10_000)
    carrier = _tone_amp(am, fs, 10_000)
    assert carrier == pytest.approx(1 / 1.5)                  # zarf tepesi = amplitude
    assert _tone_amp(am, fs, 11_000) / carrier == pytest.approx(0.25)
    assert _tone_amp(am, fs, 9_000) / carrier == pytest.approx(0.25)
    assert np.abs(am).max() <= 1.0 + 1e-12
    fm = sg.synthesize(WaveSpec("fm", freq=10_000.0, mod_freq=100.0, deviation=2000.0), 200_000.0, 40_000)
    fi = _inst_freq(fm, 200_000.0)[1000:-1000]
    assert fi.max() == pytest.approx(12_000.0, rel=0.01) and fi.min() == pytest.approx(8_000.0, rel=0.01)


def test_ecg_rate_staircase_damped_rect():
    fs = 1000.0
    ecg = sg.synthesize(WaveSpec("ecg", bpm=75.0), fs, 8000)
    peaks, _ = signal.find_peaks(ecg, height=0.8)
    assert len(peaks) == 10 and ecg.max() == pytest.approx(1.0, abs=1e-3)
    assert np.diff(peaks) == pytest.approx(np.full(9, 800), abs=1)
    st = sg.synthesize(WaveSpec("staircase", freq=10.0, steps=5), fs, 100)
    assert np.allclose(np.unique(st), [0, 0.25, 0.5, 0.75, 1.0])
    ds = sg.synthesize(WaveSpec("damped_sine", freq=1000.0, tau=2e-3, repeat_period=0.01), 100_000.0, 1000)
    assert np.abs(ds[:200]).max() > 3 * np.abs(ds[800:]).max()
    fr = sg.synthesize(WaveSpec("full_rect_sine", freq=100.0), 10_000.0, 10_000)
    assert _tone_amp(fr - fr.mean(), 10_000.0, 100.0) > 0.1     # tümsek tekrar frekansı = freq
    hr = sg.synthesize(WaveSpec("half_rect_sine", freq=100.0), 10_000.0, 100)
    assert hr.min() == 0.0 and np.all(hr[51:] == 0.0) and hr[50] == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("kind", list(sg.WAVEFORMS))
def test_every_kind_builds_table_and_metrics(kind):
    samples = np.sin(np.linspace(0, 2 * np.pi, 200, endpoint=False)) if kind == "samples" else ()
    spec = sg.fit_to_dac(WaveSpec(kind, samples=samples), DAC12)
    x = sg.synthesize(spec, 48_000.0, 2000)
    assert x.shape == (2000,) and np.all(np.isfinite(x))
    tab = sg.make_table(spec, DAC12, 48_000.0, 65_536)
    m = sg.metrics(tab)
    assert tab.codes.min() >= 0 and tab.codes.max() <= 4095
    if kind not in ("noise_white", "noise_pink"):
        assert tab.clipped == 0, tab.warnings
    if sg.is_periodic(spec) and kind != "dc":
        assert tab.periodic and tab.periods >= 1
        assert not any("süreksizlik" in w for w in tab.warnings)
    assert m.memory_bytes == tab.N * 2 and math.isfinite(m.vpp)
    assert WaveSpec.from_dict(json.loads(json.dumps(spec.to_dict()))) == spec


# ============================================================================ formül
@pytest.mark.parametrize("text", [
    "__import__('os')", "().__class__", "ph.real", "lambda: 1", "9**9**9", "2**1000", "[x for x in ph]",
    "sin(ph)[0]", "open('x')", "eval('1')", "sin", "foo(1)", "sin(1, 2)", "'abc'", "1e999", "(t := 1)",
    "sin(x=1)", "*ph", "{1: 2}", "sin(ph) +", "x" * (sg.MAX_EXPR_LEN + 1), "(" * 300 + "1" + ")" * 300,
    "1" + "+1" * 600, "globals()", "ph if 1 else __builtins__",
])
def test_expression_rejects_unsafe_or_invalid(text):
    with pytest.raises(SignalError):
        sg.check_expr(text)


def test_expression_errors_have_positions_and_nan_detection():
    with pytest.raises(SignalError) as exc:
        sg.check_expr("sin(ph) + foo(1)")
    assert exc.value.position == 11 and "sütun 11" in str(exc.value)
    with pytest.raises(SignalError) as exc:
        sg.check_expr("  2**(9**9)")
    assert exc.value.position == 7                        # üssün başladığı sütun (baştaki boşluklar dahil)
    with pytest.raises(SignalError, match="tanımsız"):
        sg.eval_expr("log(ph)", ph=np.linspace(0, 1, 5))
    with pytest.raises(SignalError):
        sg.synthesize(WaveSpec("expr", expr="1/(ph-0.5)"), 1000.0, 1000)


def test_expression_evaluates_vectorized():
    ph = np.linspace(0, 1, 101, endpoint=False)
    sq = sg.eval_expr("where(ph < 0.25, 1, -1)", ph=ph)
    assert np.array_equal(sq, np.where(ph < 0.25, 1.0, -1.0))
    assert np.array_equal(sg.eval_expr("square(ph, 0.25)", ph=ph), sq)
    assert np.array_equal(sg.eval_expr("1 if 0.1 < ph < 0.5 else 0", ph=ph), ((ph > 0.1) & (ph < 0.5)) * 1.0)
    assert np.allclose(sg.eval_expr("clip(2*tri(ph), -1, 1) + min(ph, 0.5, 0.2) - max(ph, ph)", ph=ph),
                       np.clip(2 * (1 - 4 * np.abs(ph - 0.5)), -1, 1) + np.minimum(ph, 0.2) - ph)
    assert sg.eval_expr("pi + e + 2**3 - 7 % 4 + 7 // 2", t=np.zeros(3)) == pytest.approx(
        np.full(3, math.pi + math.e + 8 - 3 + 3))
    expr = sg.synthesize(WaveSpec("expr", expr="sin(2*pi*f*t)", freq=1000.0), 48_000.0, 480)
    ref = sg.synthesize(WaveSpec("sine", freq=1000.0), 48_000.0, 480)
    assert np.allclose(expr, ref, atol=1e-12)


def test_expression_with_absolute_time_warns_discontinuity():
    spec = WaveSpec("expr", expr="sin(2*pi*1000.3*t)", freq=1000.0)
    tab = sg.make_table(spec, DacSpec(12, 2.0, out_min=-1.0, out_max=1.0), 48_000.0, 512)
    assert any("süreksizlik" in w for w in tab.warnings)


# ============================================================================ noktalar ve harmonikler
PTS = ((0.1, 0.0), (0.4, 1.0), (0.7, -0.5))


def _points(interp, points=PTS):
    return WaveSpec("points", freq=1.0, points=points, interp=interp)


def test_points_step_and_linear():
    fs, n = 1000.0, 1000
    st = sg.synthesize(_points("step"), fs, n)
    assert st[0] == -0.5 and st[100] == 0.0 and st[399] == 0.0 and st[400] == 1.0 and st[999] == -0.5
    li = sg.synthesize(_points("linear"), fs, n)
    assert li[100] == pytest.approx(0.0) and li[250] == pytest.approx(0.5) and li[400] == pytest.approx(1.0)
    # sarma: 0.7 → 1.1 arası doğrusal (−0.5 → 0), sürekli
    assert li[0] == pytest.approx(-0.125) and li[999] == pytest.approx(-0.125 - 0.5 / 400 * 1, abs=1e-12)
    jump = sg.synthesize(_points("linear", ((0.0, -1.0), (1.0, 1.0))), fs, n)   # bilinçli sıçrama (testere)
    assert np.allclose(jump, np.linspace(-1, 1, n, endpoint=False))


def test_points_cubic_periodic_continuity():
    fs, n = 100_000.0, 100_000
    cu = sg.synthesize(_points("cubic"), fs, n)
    assert cu[10_000] == pytest.approx(0.0, abs=1e-12) and cu[40_000] == pytest.approx(1.0, abs=1e-12)
    d = np.diff(cu)
    wrap_step = cu[0] - cu[-1]
    assert abs(wrap_step - d[-1]) < 1e-6 and abs(wrap_step - d[0]) < 1e-6   # değer ve eğim sürekli
    assert np.abs(np.diff(d)).max() < 1e-6                                   # eğri düzgün (C2)
    with pytest.raises(SignalError):
        sg.synthesize(_points("cubic", ((0.2, 0.0), (0.2, 1.0))), fs, 10)
    nonper = sg.synthesize(_points("cubic", ((0.0, -1.0), (0.5, 0.2), (1.0, 1.0))), 1000.0, 1000)
    assert nonper[0] == pytest.approx(-1.0) and nonper[999] == pytest.approx(1.0, abs=0.01)


def test_harmonics_equal_expression():
    harm = ((1.0, 1.0, 0.0), (3.0, 1 / 3, 30.0), (5.0, 0.2, -45.0), (0.0, 0.1, 0.0))
    expr = "sin(2*pi*ph) + sin(2*pi*3*ph + 30*pi/180)/3 + 0.2*sin(2*pi*5*ph - 45*pi/180) + 0.1"
    for phase in (0.0, 37.0):
        a = sg.synthesize(WaveSpec("harmonics", freq=440.0, harmonics=harm, phase_deg=phase, amplitude=0.8), 44_100.0,
                          5000)
        b = sg.synthesize(WaveSpec("expr", freq=440.0, expr=expr, phase_deg=phase, amplitude=0.8), 44_100.0, 5000)
        assert np.allclose(a, b, atol=1e-12)
    with pytest.raises(SignalError):
        sg.synthesize(WaveSpec("harmonics", harmonics=((1.5, 1.0, 0.0),)), 1000.0, 10)


def test_samples_modes():
    s = np.array([0.0, 1.0, 0.0, -1.0])
    play = sg.synthesize(WaveSpec("samples", samples=s, amplitude=2.0, offset=1.0), 1000.0, 10)
    assert np.array_equal(play, 1.0 + 2.0 * np.resize(s, 10))
    per = sg.synthesize(WaveSpec("samples", samples=s, samples_mode="period", freq=10.0, interp="linear"), 1000.0, 100)
    assert per[0] == 0.0 and per[25] == pytest.approx(1.0) and per[12] == pytest.approx(0.48)


# ============================================================================ tablo
def test_exact_table_frequency_is_optimal():
    fs, f, max_len = 48_000.0, 1234.5, 4096
    spec = sg.fit_to_dac(WaveSpec("sine", freq=f), DAC12)
    tab = sg.make_table(spec, DAC12, fs, max_len)
    assert tab.N <= max_len and tab.f_real == pytest.approx(tab.periods * fs / tab.N)
    ns = np.arange(2, max_len + 1)
    best = np.min(np.abs(np.maximum(np.rint(ns * f / fs), 1) * fs / ns - f))
    assert abs(tab.freq_error_hz) <= best + 1e-12
    assert abs(tab.freq_error_ppm) < 10.0 and math.gcd(tab.N, tab.periods) == 1   # tam çözüm N = 32000 ister
    # tolerans: sınırın altındaki en kısa N
    tol = sg.make_table(spec, DAC12, fs, max_len, tol_ppm=1000.0)
    rel = np.abs(np.maximum(np.rint(ns * f / fs), 1) * fs / ns - f) / f
    assert abs(tol.freq_error_ppm) <= 1000.0 and tol.N < tab.N
    assert tol.N == ns[np.argmax(rel <= 1e-3)]
    # tam bölünen durum: tek periyot
    t2 = sg.make_table(WaveSpec("sine", freq=1000.0), DAC12, fs, 4096)
    assert (t2.N, t2.periods, t2.freq_error_hz) == (48, 1, 0.0)
    # tablo tam periyodik: 2 tur arka arkaya çalınınca tek uzun sentezle aynı
    ideal = sg.synthesize(sg.fit_to_dac(WaveSpec("sine", freq=tab.f_real), DAC12), fs, 2 * tab.N)
    assert np.allclose(np.tile(tab.values, 2), ideal, atol=1e-9)
    with pytest.raises(SignalError, match="En uzun tablo"):
        sg.make_table(WaveSpec("sine", freq=5.0), DAC12, fs, 1024)


def test_exact_table_multi_component_and_nonperiodic():
    fs = 48_000.0
    am = sg.make_table(WaveSpec("am", freq=10_000.0, mod_freq=300.0, depth=0.3, amplitude=1, offset=1.65), DAC12, fs)
    for fr, ft in zip(am.freqs_real, am.freqs_target):
        assert (fr * am.N / fs) == pytest.approx(round(fr * am.N / fs), abs=1e-9)
        assert abs(fr - ft) / ft < 1e-3
    noise = sg.make_table(WaveSpec("noise_pink", duration=0.02, offset=1.65), DAC12, fs)
    assert not noise.periodic and noise.N == 960 and noise.f_real == pytest.approx(fs / 960)
    ch = sg.make_table(WaveSpec("chirp", freq=100, f1=5000, sweep_time=0.05, offset=1.65), DAC12, fs, 4096)
    assert ch.N == 2400 and not ch.periodic
    with pytest.raises(SignalError):
        sg.make_table(WaveSpec("chirp", sweep_time=1.0), DAC12, fs, 4096)
    st = DacSpec(12, 3.3, channels=2)
    stereo = sg.make_table(WaveSpec("sine", freq=1000.0, offset=1.65), st, fs,
                           spec2=WaveSpec("square", freq=1500.0, offset=1.65))
    assert stereo.codes.shape == (stereo.N, 2) and stereo.flat_codes[1] == stereo.codes[0, 1]
    assert stereo.N == 96 and stereo.freqs_real == (1000.0, 1500.0)
    m1, m2 = sg.metrics(stereo, channel=0), sg.metrics(stereo, channel=1)
    assert m1.f_real == 1000.0 and m2.f_real == 1500.0
    assert m1.thd_pct < 0.1 and 43 < m2.thd_pct < 48      # kare: ideal 43.8 % + 32 örnek/periyotta örtüşme
    with pytest.raises(SignalError):
        sg.metrics(stereo, channel=2)


def test_dds_tuning_word_and_resolution():
    fs = 48_000.0
    w, f_real, res = sg.dds_tuning(1000.0, fs)
    assert w == round(1000.0 * 2 ** 32 / fs) == 89_478_485
    assert res == fs / 2 ** 32 and f_real == w * fs / 2 ** 32
    assert abs(f_real - 1000.0) <= res / 2
    tab = sg.make_table(sg.fit_to_dac(WaveSpec("sine", freq=1234.567), DAC12), DAC12, fs, 1000, mode="dds")
    assert tab.N == 512 and tab.table_bits == 9 and tab.periods == 1
    assert tab.tuning_word == sg.dds_tuning(1234.567, fs)[0]
    seq = sg.dds_sequence(tab, 8)
    acc = (np.arange(8, dtype=np.uint64) * np.uint64(tab.tuning_word)) & np.uint64(0xFFFFFFFF)
    assert np.array_equal(seq, tab.codes[(acc >> np.uint64(23)).astype(int)])
    m = sg.metrics(tab)
    assert 45 < m.sfdr_dbc < 70 and m.f_real == tab.f_real
    mi = sg.metrics(tab, dds_interpolate=True)
    assert mi.sfdr_dbc > m.sfdr_dbc + 10
    with pytest.raises(SignalError):
        sg.make_table(WaveSpec("noise_white"), DAC12, fs, mode="dds")


# ============================================================================ nicemleme
def test_quantization_lsb_rounding_and_clipping():
    dac = DacSpec(12, 3.3)
    assert dac.lsb == 3.3 / 4096 and dac.v_hi == pytest.approx(3.3 - dac.lsb)
    k = np.array([0, 1, 2047, 4095])
    codes, volts, clipped = sg.quantize(dac.out_min + k * dac.lsb, dac)
    assert np.array_equal(codes, k) and np.allclose(volts, k * dac.lsb) and clipped == 0
    codes, _, _ = sg.quantize(np.array([0.49, 0.5, 0.51]) * dac.lsb, dac)
    assert list(codes) == [0, 1, 1]                        # yarım LSB yukarı
    _, _, clipped = sg.quantize(np.array([dac.v_hi + 0.49 * dac.lsb, dac.v_hi + 0.51 * dac.lsb, -0.6 * dac.lsb,
                                          -0.4 * dac.lsb, 3.3]), dac)
    assert clipped == 3
    tab = sg.make_table(WaveSpec("sine", amplitude=2.0, offset=1.65), dac, 48_000.0)
    assert tab.clipped > 0 and any("kırpıldı" in w for w in tab.warnings)
    assert tab.codes.min() == 0 and tab.codes.max() == 4095
    with pytest.raises(SignalError):
        sg.quantize(np.array([np.nan]), dac)


def test_alignment_and_twos_complement():
    left = DacSpec(12, 3.3, align="left")
    tab = sg.make_table(sg.fit_to_dac(WaveSpec("sine"), left), left, 48_000.0)
    assert np.array_equal(tab.codes, tab.levels << 4) and np.all(tab.codes & 0xF == 0)
    assert left.c_type == "uint16_t" and left.code_max == 0xFFF0
    tc = DacSpec(16, 1.0, out_min=-1.0, out_max=1.0, coding="twos_complement")
    codes, volts, _ = sg.quantize(np.array([-1.0, 0.0, 1.0 - 2 / 65536, 0.5]), tc)
    assert list(codes) == [-32768, 0, 32767, 16384] and tc.c_type == "int16_t"
    assert np.allclose(sg.codes_to_volts(codes, tc), volts)
    c24 = DacSpec(24, 1.0, out_min=-1.0, out_max=1.0, coding="twos_complement", align="left", word_bits=32)
    codes, _, _ = sg.quantize(np.array([-1.0, 0.0, 0.25]), c24)
    assert list(codes) == [-(1 << 31), 0, (1 << 21) << 8] and c24.c_type == "int32_t"
    assert np.array_equal(sg.codes_to_levels(sg.levels_to_codes(np.arange(0, 1 << 24, 4097), c24), c24),
                          np.arange(0, 1 << 24, 4097))
    ob = DacSpec(12, 10.0, out_min=-5.0, out_max=5.0, coding="offset_binary")
    assert list(sg.quantize(np.array([-5.0, 0.0]), ob)[0]) == [0, 2048]
    with pytest.raises(SignalError):
        DacSpec(16, word_bits=8)


# ============================================================================ ölçümler
@pytest.mark.parametrize("bits", [8, 12, 16])
def test_quantization_snr_matches_theory(bits):
    dac = DacSpec(bits, 1.0)
    tab = sg.make_table(sg.fit_to_dac(WaveSpec("sine", freq=997.0), dac), dac, 48_000.0, 16_384)
    assert math.gcd(tab.N, tab.periods) == 1 and tab.N > 4000   # örtüşmeyen M/N: nicemleme hatası beyaz
    m = sg.metrics(tab)
    theory = 6.02 * bits + 1.76
    assert m.snr_db == pytest.approx(theory, abs=1.0)
    assert m.sinad_db == pytest.approx(theory, abs=1.0)
    assert m.enob == pytest.approx(bits, abs=0.2)
    assert m.quant_snr_db == pytest.approx(theory, abs=1.0)
    assert m.sfdr_dbc > theory


def test_square_wave_thd():
    dac = DacSpec(16, 1.0)
    tab = sg.make_table(sg.fit_to_dac(WaveSpec("square", freq=1000.0), dac), dac, 1e6, 4096)
    assert (tab.N, tab.periods) == (1000, 1)
    ideal_all = 100 * math.sqrt(math.pi ** 2 / 8 - 1)                       # 48.34 %
    assert sg.metrics(tab, None).thd_pct == pytest.approx(ideal_all, abs=0.2)
    first10 = 100 * math.sqrt(sum(1 / k ** 2 for k in (3, 5, 7, 9, 11)))       # 43.83 %
    m = sg.metrics(tab)
    assert m.thd_pct == pytest.approx(first10, abs=0.2)
    assert [h[0] for h in m.harmonics][:3] == [2, 3, 4] and m.harmonics[1][2] == pytest.approx(-9.54, abs=0.05)


def test_metrics_time_domain_and_filter():
    tab = sg.make_table(sg.fit_to_dac(WaveSpec("sine", freq=1000.0), DAC12), DAC12, 48_000.0)
    m = sg.metrics(tab)
    assert m.vmin == 0.0 and m.vmax == pytest.approx(DAC12.v_hi)
    assert m.crest_factor == pytest.approx(math.sqrt(2), rel=1e-3)
    assert m.ac_rms == pytest.approx(DAC12.v_hi / 2 / math.sqrt(2), rel=1e-3)
    assert m.update_interval == pytest.approx(1 / 48_000) and m.image_hz == pytest.approx(47_000.0)
    assert m.content_max_hz == pytest.approx(1000.0)
    assert m.first_image_hz < 48_000 and 1000 < m.filter_cutoff_hz < m.first_image_hz
    att = 10 * math.log10(1 + (m.first_image_hz / m.filter_cutoff_hz) ** (2 * m.filter_order))
    assert att >= 40.0
    assert m.max_slew_v_per_s == pytest.approx(m.max_step_v * 48_000.0)
    assert m.max_step_v == pytest.approx(DAC12.v_hi / 2 * 2 * math.sin(math.pi / 48), rel=0.01)
    assert m.memory_bytes == 96
    am = sg.metrics(sg.make_table(WaveSpec("am", offset=1.65, amplitude=1.0), DAC12, 48_000.0))
    assert math.isnan(am.thd_pct) and am.quant_snr_db > 60


# ============================================================================ zamanlayıcı
def test_timer_config():
    tc = sg.timer_config(48_000.0, 84e6)
    assert (tc.psc, tc.arr, tc.fs_real, tc.error_ppm) == (0, 1749, 48_000.0, 0.0)
    one = sg.timer_config(1.0, 84e6)
    assert one.fs_real == 1.0 and one.psc <= 65535 and one.arr <= 65535
    smaller = [p for p in range(1, one.psc + 1) if 84_000_000 % p == 0 and 84_000_000 // p <= 65536]
    assert smaller == [] and one.psc == 1343                     # tam çözümlü en küçük PSC
    t441 = sg.timer_config(44_100.0, 84e6)
    assert t441.divisor == 1905 and t441.psc == 0 and t441.error_ppm == pytest.approx(-124.98, abs=0.01)
    t32 = sg.timer_config(1.0, 84e6, bits=32)
    assert (t32.psc, t32.arr) == (0, 83_999_999)
    half = sg.timer_config(42e6, 84e6)
    assert (half.psc, half.arr) == (0, 1)                      # ARR ≥ 1 (ARR = 0 sayacı durdurur)
    low = sg.timer_config(1e-3, 84e6, psc_bits=32)
    assert low.arr <= 65535 and abs(low.error_ppm) < 1.0
    for bad in ((1e9, 84e6), (50e6, 84e6), (1e-3, 84e6)):
        with pytest.raises(SignalError):
            sg.timer_config(*bad)


# ============================================================================ kayıt / CSV / yardımcılar
def test_spec_roundtrip_and_validation():
    spec = WaveSpec("points", points=[(0, 1), (0.5, -1)], interp="cubic", harmonics=[[1, 1, 0]],
                    samples=np.arange(3.0), tones=[(100, 1, 0)])
    assert isinstance(spec.points, tuple) and spec.points[1] == (0.5, -1.0)
    d = json.loads(json.dumps(spec.to_dict()))
    assert WaveSpec.from_dict({**d, "unknown_future_key": 1}) == spec
    dac = DacSpec(16, 1.0, out_min=-1, out_max=1, coding="twos_complement", channels=2)
    assert DacSpec.from_dict(json.loads(json.dumps(dac.to_dict()))) == dac
    with pytest.raises(SignalError):
        WaveSpec("nope")
    with pytest.raises(SignalError):
        sg.synthesize(WaveSpec("sine", freq=30_000.0), 48_000.0, 10)
    assert set(sg.PARAMS) == set(sg.WAVEFORMS)
    assert all(p in sg.FIELD_INFO for ps in sg.PARAMS.values() for p in ps)


def test_fit_to_dac_fills_range():
    for kind in ("sine", "pulse", "ecg", "sinc", "harmonics", "expr"):
        spec = sg.fit_to_dac(WaveSpec(kind, freq=100.0), DAC12)
        x = sg.synthesize(spec, 2000.0 * sg.fundamental(spec), 2000)     # tam bir periyot
        assert x.min() >= DAC12.v_lo - 1e-9 and x.max() <= DAC12.v_hi + 1e-9
        assert x.max() - x.min() == pytest.approx(DAC12.v_hi - DAC12.v_lo, rel=0.01)


def test_load_samples_csv(scratch):
    tr = scratch / "tr.csv"
    tr.write_bytes("Zaman;Değer\n0,000;1,5\n0,001;-2,25\n0,002;3\n".encode("cp1254"))
    assert np.allclose(sg.load_samples_csv(tr), [1.5, -2.25, 3.0])
    assert np.allclose(sg.load_samples_csv(tr, "zaman"), [0.0, 0.001, 0.002])
    one = scratch / "one.csv"
    one.write_text("\ufeff# yorum\n0.5\n-1e-3\n\n2\n", encoding="utf-8")
    assert np.allclose(sg.load_samples_csv(one), [0.5, -0.001, 2.0])
    en = scratch / "en.csv"
    en.write_text("t,v,w\n0,1,2\n1,3,4\n", encoding="utf-8")
    assert np.allclose(sg.load_samples_csv(en, 1), [1, 3]) and np.allclose(sg.load_samples_csv(en), [2, 4])
    bad = scratch / "bad.csv"
    bad.write_text("1;2\n3;x\n", encoding="utf-8")
    with pytest.raises(SignalError, match="2. satır"):
        sg.load_samples_csv(bad)


# ============================================================================ C üreteci
def _host_cc() -> str | None:
    cands = [os.environ.get("SPT_HOST_CC"), r"C:\mingw64\bin\gcc.exe", shutil.which("gcc")]
    for c in cands:
        if c and Path(c).is_file():
            return c
    return None


CFLAGS = ["-std=c99", "-Wall", "-Wextra", "-pedantic", "-Werror", "-O2"]


@pytest.fixture
def cc():
    c = _host_cc()
    if c is None:
        pytest.skip("Host C derleyicisi yok (SPT_HOST_CC ya da C:\\mingw64\\bin\\gcc.exe)")
    return c


def _write(files, d: Path) -> None:
    for f in files:
        (d / f.name).write_text(f.text, encoding="utf-8")


def _build_run(cc: str, d: Path, main: str, extra=()) -> list[str]:
    (d / "main_test.c").write_text(main, encoding="utf-8")
    exe = d / "t.exe"
    r = subprocess.run([cc, *CFLAGS, *extra, "-static", "main_test.c", "wave.c", "-o", str(exe)], cwd=d,
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    out = subprocess.run([str(exe)], cwd=d, capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    return out.stdout.split()


def _fmt(dac: DacSpec) -> tuple[str, str]:
    return ("%ld", "(long)") if dac.signed else ("%lu", "(unsigned long)")


@pytest.mark.parametrize("dac", [
    DacSpec(12, 3.3),
    DacSpec(12, 3.3, align="left"),
    DacSpec(16, 1.0, out_min=-1.0, out_max=1.0, coding="twos_complement"),
    DacSpec(24, 1.0, out_min=-1.0, out_max=1.0, coding="twos_complement", align="left", word_bits=32),
    DacSpec(8, 3.3, word_bits=8),
])
def test_c_generic_isr_matches_python(cc, scratch, dac):
    spec = sg.fit_to_dac(WaveSpec("sine", freq=1234.5), dac)
    tab = sg.make_table(spec, dac, 48_000.0, 512)
    _write(dac_c.generate(spec, dac, tab, "generic_isr"), scratch)
    fmt, cast = _fmt(dac)
    count = 2 * tab.N + 7
    main = f"""#include <stdio.h>
#include "wave.h"
void dac_write(wave_word_t code) {{ printf("{fmt}\\n", {cast}code); }}
int main(void)
{{
    unsigned long i;
    for (i = 0; i < {count}ul; i++) {{
        wave_isr_tick();
    }}
    printf("%lu\\n", (unsigned long)wave_arr_for_hz(WAVE_F_HZ));
    printf("%lu\\n", (unsigned long)WAVE_TIM_ARR);
    return 0;
}}
"""
    out = [int(v) for v in _build_run(cc, scratch, main)]
    expect = tab.codes[np.arange(count) % tab.N]
    assert np.array_equal(np.array(out[:count]), expect)
    assert out[count] == out[count + 1] == 1749


def test_c_generic_isr_stereo_matches_python(cc, scratch):
    dac = DacSpec(12, 3.3, channels=2)
    spec = WaveSpec("sine", freq=1000.0, offset=1.65, amplitude=1.5)
    tab = sg.make_table(spec, dac, 48_000.0, spec2=WaveSpec("triangle", freq=1500.0, offset=1.65, amplitude=1.0))
    _write(dac_c.generate(spec, dac, tab, "generic_isr"), scratch)
    main = """#include <stdio.h>
#include "wave.h"
void dac_write_stereo(wave_word_t a, wave_word_t b) { printf("%lu %lu\\n", (unsigned long)a, (unsigned long)b); }
int main(void)
{
    unsigned long i;
    for (i = 0; i < 2ul * WAVE_N; i++) {
        wave_isr_tick();
    }
    return 0;
}
"""
    out = np.array([int(v) for v in _build_run(cc, scratch, main)]).reshape(-1, 2)
    assert np.array_equal(out, np.tile(tab.codes, (2, 1)))


@pytest.mark.parametrize("interp", [False, True])
@pytest.mark.parametrize("dac", [
    DacSpec(12, 3.3),
    DacSpec(24, 1.0, out_min=-1.0, out_max=1.0, coding="twos_complement", align="left", word_bits=32),
])
def test_c_dds_matches_python_model(cc, scratch, dac, interp):
    fs = 48_000.0
    spec = sg.fit_to_dac(WaveSpec("triangle", freq=1234.567, symmetry=0.3), dac)
    tab = sg.make_table(spec, dac, fs, 1024, mode="dds")
    _write(dac_c.generate(spec, dac, tab, "dds", interpolate=interp), scratch)
    fmt, cast = _fmt(dac)
    k1, k2, hz2 = 3000, 2000, 3210.987
    main = f"""#include <stdio.h>
#include "wave.h"
int main(void)
{{
    wave_dds_t g;
    unsigned long i;
    wave_dds_init(&g);
    for (i = 0; i < {k1}ul; i++) {{
        printf("{fmt}\\n", {cast}wave_dds_next(&g));
    }}
    wave_dds_set_freq(&g, {hz2!r});
    printf("%lu\\n", (unsigned long)g.step);
    for (i = 0; i < {k2}ul; i++) {{
        printf("{fmt}\\n", {cast}wave_dds_next(&g));
    }}
    wave_dds_set_freq(&g, 1e9);
    printf("%lu\\n", (unsigned long)g.step);
    return 0;
}}
"""
    out = [int(v) for v in _build_run(cc, scratch, main)]
    seq1 = sg.dds_sequence(tab, k1, interpolate=interp)
    w2 = sg.dds_tuning(hz2, fs)[0]
    seq2 = sg.dds_sequence(tab, k2, step=w2, phase=(k1 * tab.tuning_word) % 2 ** 32, interpolate=interp)
    assert np.array_equal(np.array(out[:k1]), seq1)
    assert out[k1] == w2
    assert np.array_equal(np.array(out[k1 + 1:k1 + 1 + k2]), seq2)
    assert out[-1] == sg.dds_tuning(1e9, fs)[0] == 2 ** 31
    # varsayılanı -D ile ters çevirmek de derlenir
    r = subprocess.run([cc, *CFLAGS, f"-DWAVE_DDS_INTERP={0 if interp else 1}", "-c", "wave.c", "wave_ornek.c"],
                       cwd=scratch, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr


@pytest.mark.parametrize("target,dac,mode", [
    ("stm32_hal_dma", DacSpec(12, 3.3), "exact"),
    ("stm32_hal_dma", DacSpec(12, 3.3, align="left"), "exact"),
    ("stm32_hal_dma", DacSpec(8, 3.3, word_bits=8), "exact"),
    ("stm32_hal_dma", DacSpec(12, 3.3, channels=2), "exact"),
    ("generic_isr", DacSpec(12, 3.3), "exact"),
    ("generic_isr", DacSpec(12, 3.3, channels=2), "exact"),
    ("dds", DacSpec(16, 1.0, out_min=-1, out_max=1, coding="twos_complement"), "dds"),
])
def test_c_core_compiles_for_every_target(cc, scratch, target, dac, mode):
    spec = sg.fit_to_dac(WaveSpec("sine", freq=1000.0), dac)
    tab = sg.make_table(spec, dac, 48_000.0, 1024, mode=mode)
    files = dac_c.generate(spec, dac, tab, target, "wave")
    names = [f.name for f in files]
    assert names[:2] == ["wave.h", "wave.c"] and names[-1] == "KULLANIM.md"
    assert names[2] == "wave_ornek.c"
    _write(files, scratch)
    srcs = ["wave.c"] + (["wave_ornek.c"] if target in ("generic_isr", "dds") else [])
    r = subprocess.run([cc, *CFLAGS, "-c", *srcs], cwd=scratch, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    text = files[1].text.replace("wave_word_t", dac.c_type)
    if target != "dds" and dac.channels == 2:  # stereo: 4 bayt hizalı birleşim içinde (ikili DMA)
        assert "static WAVE_TABLE_QUAL union {" in text and f"    {dac.c_type} w[WAVE_TABLE_LEN];" in text
    elif target != "dds":
        assert f"static WAVE_TABLE_QUAL {dac.c_type} " in text


@pytest.mark.parametrize("dac", [DacSpec(12, 3.3, channels=2), DacSpec(8, 3.3, word_bits=8, channels=2)])
def test_stm32_stereo_table_is_word_aligned_for_dual_dma(dac):
    """İkili DAC'a tek DMA aktarımı (12 bit: Word, 8 bit: Half Word): tablo her derleyici / optimizasyonda 4 bayt
    hizalı olmalı (C kodu doğrulayıcısı: uint16/uint8 dizi gcc -Os ve clang'da 2 / 1 bayt hizaya düşüyordu)."""
    from spt.codegen import toolchain as tc
    from spt.codegen.mcu import MCU_PROFILES
    spec = sg.fit_to_dac(WaveSpec("sine", freq=1000.0), dac)
    tab = sg.make_table(spec, dac, 48_000.0, spec2=sg.fit_to_dac(WaveSpec("triangle", freq=1500.0), dac))
    files = {f.name: f.text for f in dac_c.generate(spec, dac, tab, "stm32_hal_dma", "wave")}
    objdump = None
    gcc = tc.find_arm_gcc()
    if gcc is not None:
        objdump = gcc.sibling("arm-none-eabi-objdump")
    if objdump is None:
        pytest.skip("arm-none-eabi-objdump yok")
    checked = 0
    for compiler in ("gcc", "clang"):
        for opt in ("-O0", "-Os", "-O2"):
            build = tc.compile_cross({"wave.h": files["wave.h"], "wave.c": files["wave.c"]}, ["wave.c"],
                                     MCU_PROFILES["cortex-m4f"], compiler, opt, warnings=("-Wall", "-Werror"))
            if build.tool is None:
                continue
            assert build.ok, build.log
            r = subprocess.run([objdump, "-h", str(build.objects["wave.c"])], capture_output=True, text=True)
            line = next(ln for ln in r.stdout.splitlines() if "wave_table_store" in ln)
            align = int(line.split()[-1].split("**")[1])
            assert align >= 2, (compiler, opt, line)  # 2**2 = 4 bayt
            checked += 1
    assert checked


def test_generate_content_and_validation():
    spec = sg.fit_to_dac(WaveSpec("sine", freq=1000.0), DacSpec(12, 3.3, align="left"))
    dac = DacSpec(12, 3.3, align="left")
    tab = sg.make_table(spec, dac, 48_000.0)
    files = {f.name: f.text for f in dac_c.generate(spec, dac, tab, "stm32_hal_dma", "Sine 1k!")}
    # ad C tanımlayıcısına çevrilir; çift / sondaki alt çizgi (C / C++'ta saklı: SINE_1K__…) üretilmez
    assert set(files) == {"Sine_1k.h", "Sine_1k.c", "Sine_1k_ornek.c", "KULLANIM.md"}
    h, md = files["Sine_1k.h"], files["KULLANIM.md"]
    assert "DAC_ALIGN_12B_L" in h and "#define SINE_1K_TIM_ARR             1749u" in h
    assert "__" not in "".join(re.findall(r"\b\w*SINE_1K\w*", h))
    assert "HAL_DAC_Start_DMA" in files["Sine_1k.c"] and "USE_HAL_DRIVER" in files["Sine_1k.c"]
    assert "PSC = 0" in md and "ARR = 1749" in md and "Circular" in md and "Update Event" in md
    assert "96 bayt" in md and "Butterworth" in md
    assert set(dac_c.TARGETS) == {"stm32_hal_dma", "generic_isr", "dds"}   # yalnız STM32 (+ taşınabilir C)
    for removed in ("esp32", "arduino"):
        with pytest.raises(SignalError, match="Bilinmeyen hedef"):
            dac_c.generate(spec, dac, tab, removed)
    with pytest.raises(SignalError, match="DDS"):  # kullanıcı diliyle: “Tablo kipi”ni DDS seçin
        dac_c.generate(spec, dac, tab, "dds")
    with pytest.raises(SignalError):
        dac_c.generate(WaveSpec("sine"), dac, tab, "generic_isr")
    with pytest.raises(SignalError):
        dac_c.generate(spec, dac, tab, "generic_isr", fs=44_100.0)
    with pytest.raises(SignalError):
        dac_c.generate(spec, dac, tab, "nope")
    odd = dac_c.generate(spec, dac, sg.make_table(spec, dac, 44_100.0), "generic_isr")
    assert "Zamanlayıcı fs'yi tam üretemiyor" in odd[-1].text


def test_memory_report():
    spec = sg.fit_to_dac(WaveSpec("sine", freq=1000.0), DAC12)
    tab = sg.make_table(spec, DAC12, 48_000.0)
    rep = dac_c.memory_report(tab, DAC12, "generic_isr", cpu_clock_hz=168e6)
    assert rep["table_flash_bytes"] == 96 and rep["code_size_is_estimate"] is True
    assert rep["cpu_load_pct_estimate"] == pytest.approx(40 * 48_000 / 168e6 * 100)
    dma = dac_c.memory_report(tab, DAC12, "stm32_hal_dma")
    assert dma["cpu_cycles_per_sample_estimate"] == 0 and dma["ram_bytes"] == 0
    dds = sg.make_table(spec, DAC12, 48_000.0, 256, mode="dds")
    assert dac_c.memory_report(dds, DAC12, "dds", interpolate=True)["table_flash_bytes"] == 512
    for target in dac_c.TARGETS:   # STM32 DMA tabloyu yerinde okur: ek tampon yok, CPU tahmini her hedefte var
        rep = dac_c.memory_report(tab, DAC12, target)
        assert rep["dma_buffer_ram_bytes"] == 0 and rep["cpu_load_pct_estimate"] is not None
    with pytest.raises(SignalError, match="Bilinmeyen hedef"):
        dac_c.memory_report(tab, DAC12, "arduino")


# ============================================================================ denetim regresyonları
def test_turkish_single_column_decimal_comma_csv():
    # Denetim: ";" içermeyen Türkçe tek sütun "0,5" değerleri 5.0 okunuyor, işaret kayboluyordu
    assert np.allclose(sg.parse_samples_text("0,5\n0,25\n-0,75"), [0.5, 0.25, -0.75])
    assert np.allclose(sg.parse_samples_text("Gerilim\n1,5\n-2,25"), [1.5, -2.25])
    assert np.allclose(sg.parse_samples_text("0,5 1,2\n0,6 2,4"), [1.2, 2.4])      # boşluk ayrılmış, son sütun
    assert np.allclose(sg.parse_samples_text("t,v\n0,1.5\n1,2.5"), [1.5, 2.5])    # İngilizce CSV aynen


def test_dds_metrics_reach_quantization_floor():
    # Denetim: Blackman-Harris yan lobları DDS SINAD'ını ~86 dB'de tavanlıyordu (16 bit için teori 98 dB)
    spec = sg.WaveSpec(kind="sine", freq=1234.567, amplitude=1.0 - 1e-6, offset=0.0)
    dac = sg.DacSpec(16, 1.0, out_min=-1.0, out_max=1.0, coding="twos_complement")
    t = sg.make_table(spec, dac, 48000.0, max_len=1 << 12, mode="dds")
    m = sg.metrics(t, dds_interpolate=True)
    assert m.sinad_db > 93.0 and m.enob > 15.0 and m.sfdr_dbc > 100.0


def test_fit_to_dac_uses_real_range_of_decaying_shapes():
    for spec in (sg.WaveSpec(kind="exp_decay", tau=0.01), sg.WaveSpec(kind="gaussian_pulse", width=0.002)):
        lo, hi = sg.shape_range(spec)
        assert lo > 0.3  # periyot içinde 0'a inmez (eskiden sabit 0 kabul ediliyordu → DAC'ın %9–60'ı kullanılıyordu)
        dac = sg.DacSpec(12, 3.3)
        fitted = sg.fit_to_dac(spec, dac, margin=0.02)
        y = sg.synthesize(fitted, 100_000.0, 20_000)
        assert (y.max() - y.min()) > 0.95 * (dac.v_hi - dac.v_lo) * 0.96


def test_expression_position_in_characters_and_long_sums():
    with pytest.raises(sg.SignalError) as exc:
        sg.check_expr("ｐｈ + foo")      # tam genişlikli karakterler: konum karakter cinsinden
    assert exc.value.position == 6
    sg.check_expr(" + ".join(["ph"] * 80))  # 80 terimli düz toplam "çok derin" sayılmaz


def test_dds_expression_fs_is_dac_rate():
    spec = sg.WaveSpec(kind="expr", expr="sin(2*pi*ph)*(fs < 100000)", freq=1000.0)
    dac = sg.DacSpec(12, 3.3, out_min=-1.65, out_max=1.65)
    exact = sg.make_table(spec, dac, 48000.0, mode="exact")
    dds = sg.make_table(spec, dac, 48000.0, max_len=1 << 10, mode="dds")
    assert np.ptp(exact.values) > 1.9 and np.ptp(dds.values) > 1.9  # iki kipte de fs = 48 kHz


def test_error_messages_speak_the_users_language():
    """Arayüzde olduğu gibi gösterilen iletiler iç API adlarını (make_table, DacSpec, max_len…) içermez; arayüzdeki
    alan / seçenek adlarını söyler (UI denetimi 2. tur, bulgu 14)."""
    dac = DacSpec(12, 3.3)
    spec = sg.fit_to_dac(WaveSpec("sine", freq=1000.0), dac)
    exact = sg.make_table(spec, dac, 48_000.0)
    dds = sg.make_table(spec, dac, 48_000.0, mode="dds")
    d10 = DacSpec(10, 3.3)
    dsig = DacSpec(12, 3.3, coding="twos_complement")
    cases = [
        lambda: sg.make_table(WaveSpec("samples"), dac, 48_000.0),
        lambda: sg.make_table(WaveSpec("ecg", bpm=30.0), dac, 48_000.0, 4096),
        lambda: sg.make_table(spec, dac, 48_000.0, 1 << 30),
        lambda: dac_c.generate(spec, dac, exact, "dds"),
        lambda: dac_c.generate(spec, dac, dds, "stm32_hal_dma"),
        lambda: dac_c.generate(spec, dsig, sg.make_table(spec, dsig, 48_000.0), "stm32_hal_dma"),
        lambda: dac_c.generate(spec, d10, sg.make_table(spec, d10, 48_000.0), "stm32_hal_dma"),
    ]
    for case in cases:
        with pytest.raises(SignalError) as info:
            case()
        msg = str(info.value)
        assert not any(api in msg for api in ("make_table", "DacSpec", "WaveSpec", "load_samples_csv", "mode=",
                                              "max_len", "word_bits", "coding=")), msg
