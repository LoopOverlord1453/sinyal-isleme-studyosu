"""Aşama 3 filtre motoru: scipy referansıyla birebirlik, spesifikasyon (maske), blok = tek sefer, kararlılık."""
import re
from dataclasses import replace

import numpy as np
import pytest
from scipy import signal

from spt.dsp import coefficients as C
from spt.dsp import filters as F
from spt.dsp.chain import ChainStage, ProcessingChain

FS = 10_000.0
BASE = F.FilterSpec(fs=FS, f1=1000.0, f2=2000.0, transition=200.0, ripple_db=1.0, atten_db=60.0)
IIR_CALLS = {
    "butter": lambda n, wn, band: signal.butter(n, wn, band, fs=FS, output="sos"),
    "cheby1": lambda n, wn, band: signal.cheby1(n, 1.0, wn, band, fs=FS, output="sos"),
    "cheby2": lambda n, wn, band: signal.cheby2(n, 60.0, wn, band, fs=FS, output="sos"),
    "ellip": lambda n, wn, band: signal.ellip(n, 1.0, 60.0, wn, band, fs=FS, output="sos"),
    "bessel": lambda n, wn, band: signal.bessel(n, wn, band, norm="mag", fs=FS, output="sos"),
}
ORD_CALLS = {"butter": signal.buttord, "cheby1": signal.cheb1ord, "cheby2": signal.cheb2ord, "ellip": signal.ellipord}


def _wn(band):
    return [1000.0, 2000.0] if band in F.TWO_EDGE else 1000.0


# ============================================================================ scipy referansıyla birebir
@pytest.mark.parametrize("band", list(F.BANDS))
@pytest.mark.parametrize("method", list(F.IIR_METHODS))
def test_iir_designs_equal_scipy_reference(method, band):
    d = F.design(replace(BASE, family="iir", method=method, band=band, order=6))
    assert np.array_equal(d.sos, IIR_CALLS[method](6, _wn(band), band))
    assert d.order == (12 if band in F.TWO_EDGE else 6)
    assert d.stable


@pytest.mark.parametrize("band", list(F.BANDS))
@pytest.mark.parametrize("method", list(ORD_CALLS))
def test_iir_auto_order_equals_scipy_ord_and_meets_mask(method, band):
    spec = replace(BASE, family="iir", method=method, band=band, auto_order=True)
    wp, ws = F._ord_edges(spec)
    n, wn = ORD_CALLS[method](wp, ws, 1.0, 60.0, fs=FS)
    d = F.design(spec)
    assert np.array_equal(d.sos, IIR_CALLS[method](n, wn, band))
    ok, pass_excess, stop_peak = F.check_mask(d, F.mask_for(spec))
    assert ok, (pass_excess, stop_peak)


@pytest.mark.parametrize("band", list(F.BANDS))
@pytest.mark.parametrize("window", list(F.FIR_WINDOWS))
def test_fir_window_designs_equal_firwin(window, band):
    d = F.design(replace(BASE, family="fir", method=window, band=band, order=64, kaiser_beta=7.0))
    numtaps = 65  # 64. derece; yüksek geçiren / bant durduran zaten tek sayı ister
    win = ("kaiser", 7.0) if window == "kaiser" else window
    assert np.array_equal(d.b, signal.firwin(numtaps, _wn(band), window=win, pass_zero=band, fs=FS))


def test_fir_even_taps_highpass_is_made_odd_with_warning():
    d = F.design(replace(BASE, family="fir", method="hamming", band="highpass", order=63))
    assert len(d.b) == 65 and d.order == 64
    assert any("tek sayıda" in w for w in d.warnings)


@pytest.mark.parametrize("band", list(F.BANDS))
def test_remez_and_firls_equal_scipy(band):
    spec = replace(BASE, family="fir", band=band, order=80)
    bands, desired, weight = F._ls_bands(spec)
    remez = F.design(replace(spec, method="remez"))
    assert np.array_equal(remez.b, signal.remez(81, bands, desired, weight=weight, fs=FS, maxiter=60))
    firls = F.design(replace(spec, method="firls"))
    pairs = [d for d in desired for _ in range(2)]
    assert np.array_equal(firls.b, signal.firls(81, bands, pairs, weight=weight, fs=FS))


@pytest.mark.parametrize("band", list(F.BANDS))
@pytest.mark.parametrize("method", ["kaiser", "remez", "firls", "blackman"])
def test_fir_auto_order_meets_mask(method, band):
    spec = replace(BASE, family="fir", method=method, band=band, auto_order=True)
    d = F.design(spec)
    ok, pass_excess, stop_peak = F.check_mask(d, F.mask_for(spec))
    assert ok, (d.order, pass_excess, stop_peak)


def test_remez_auto_order_is_minimal():
    spec = replace(BASE, family="fir", method="remez", auto_order=True)
    d = F.design(spec)
    shorter = F.design(replace(spec, auto_order=False, order=d.order - 1))
    assert not F.check_mask(shorter, F.mask_for(spec))[0]


# ============================================================================ bilinen değerler
def test_butterworth_is_minus_3db_at_cutoff_and_bessel_too():
    for method in ("butter", "bessel"):
        d = F.design(replace(BASE, method=method, order=5))
        r = F.analyze(d, np.array([1000.0]))
        assert r.magnitude_db[0] == pytest.approx(-3.0103, abs=1e-3)


def test_cheby1_and_ellip_passband_ripple_is_exact():
    for method in ("cheby1", "ellip"):
        d = F.design(replace(BASE, method=method, order=6, ripple_db=0.5))
        mag = F.analyze(d, np.linspace(0, 1000, 20001)).magnitude_db
        assert mag.min() == pytest.approx(-0.5, abs=1e-3)
        assert mag.max() == pytest.approx(0.0, abs=1e-6)


def test_cheby2_and_ellip_stopband_reaches_attenuation():
    for method in ("cheby2", "ellip"):
        d = F.design(replace(BASE, method=method, order=6, ripple_db=0.5, atten_db=70))
        f = np.linspace(1000, 5000, 40001)
        mag = F.analyze(d, f).magnitude_db
        stop = f >= f[np.argmax(mag <= -70.0 + 1e-6)]  # durdurma bandı: kazancın ilk kez −70 dB'ye indiği yerden
        assert mag[stop].max() == pytest.approx(-70.0, abs=0.01)


def test_linear_phase_fir_group_delay_is_half_length():
    d = F.design(replace(BASE, family="fir", method="hamming", order=100))
    r = F.analyze(d)
    passband = r.freqs < 800
    assert d.linear_phase and d.delay_samples == 50.0
    assert np.nanmax(np.abs(r.group_delay[passband] - 50.0)) < 1e-6


def test_iir_group_delay_matches_tf_group_delay():
    d = F.design(replace(BASE, method="butter", order=4))
    w = np.linspace(10, 4000, 200)
    expected = signal.group_delay(signal.sos2tf(d.sos), w=w, fs=FS)[1]
    assert np.allclose(F.analyze(d, w).group_delay, expected, atol=1e-6)


def test_step_response_settles_to_dc_gain():
    for spec in (replace(BASE, method="cheby1", order=5), replace(BASE, family="fir", method="kaiser", order=80)):
        d = F.design(spec)
        step = F.step_response(d)
        assert step[-1] == pytest.approx(1.0, abs=2e-3)
        assert np.allclose(np.cumsum(F.impulse_response(d)), step)


# ============================================================================ doğrulama
@pytest.mark.parametrize("changes, text", [
    ({"f1": 6000.0}, "Nyquist"),
    ({"band": "bandpass", "f1": 2000.0, "f2": 1000.0}, "f₁ < f₂"),
    ({"band": "bandpass", "f2": 5000.0}, "f₂"),
    ({"order": 0}, "Derece"),
    ({"fs": 0.0}, "Örnekleme"),
    ({"auto_order": True, "f1": 4900.0, "transition": 200.0}, "Δf"),
    ({"method": "ellip", "ripple_db": 5.0, "atten_db": 3.0}, "Eliptik"),
    ({"family": "special", "method": "savgol", "length": 10}, "tek sayı"),
    ({"family": "special", "method": "notch", "notch_f0": 0.0}, "Notch"),
])
def test_invalid_specs_raise_readable_errors(changes, text):
    with pytest.raises(F.FilterError, match=re.escape(text)):
        F.design(replace(BASE, **changes))


def test_impossible_spec_reports_order_limit():
    with pytest.raises(F.FilterError, match="derece"):
        F.design(replace(BASE, method="butter", auto_order=True, transition=1.0, atten_db=120))


# ============================================================================ kararlılık (kabul kriteri 3)
def test_12th_order_elliptic_is_stable_in_sos_but_not_as_single_polynomial():
    fs = 48_000.0
    spec = F.FilterSpec(fs=fs, family="iir", method="ellip", order=12, f1=100.0, ripple_db=0.5, atten_db=80.0)
    d = F.design(spec)
    assert d.stable and d.max_pole_radius > 0.9999  # kutuplar birim çembere çok yakın
    t = np.arange(400_000) / fs
    x = np.sin(2 * np.pi * 25.0 * t)
    y = signal.sosfilt(d.sos, x)
    gain = abs(F.analyze(d, np.array([25.0])).h[0])
    assert np.all(np.isfinite(y))
    assert np.abs(y[-20_000:]).max() == pytest.approx(gain, rel=1e-3)
    b, a = signal.sos2tf(d.sos)  # aynı filtre tek polinom olarak: katsayı yuvarlaması kararsız yapar
    assert np.abs(np.roots(a)).max() > 1.0
    with np.errstate(all="ignore"):
        y_tf = signal.lfilter(b, a, x)
    assert not np.all(np.isfinite(y_tf)) or np.abs(y_tf[-20_000:]).max() > 10


# ============================================================================ blok = tek sefer (kabul kriteri 2)
BLOCK_SPECS = {
    "ellip12": replace(BASE, method="ellip", order=12, ripple_db=0.5, atten_db=80),
    "butter_bp": replace(BASE, method="butter", band="bandpass", order=4),
    "fir_short": replace(BASE, family="fir", method="hamming", order=20),
    "fir_long_fft": replace(BASE, family="fir", method="kaiser", order=2000),
    "notch3": replace(BASE, family="special", method="notch", notch_f0=50.0, harmonics=3),
    "dc_block": replace(BASE, family="special", method="dc_block", dc_cutoff=2.0),
    "moving_avg": replace(BASE, family="special", method="moving_avg", length=16),
    "median": replace(BASE, family="special", method="median", length=9),
    "savgol": replace(BASE, family="special", method="savgol", length=21, polyorder=3),
}


def _random_blocks(x, rng):
    out, pos = [], 0
    while pos < len(x):
        n = int(rng.choice([1, 2, 7, 64, 1000, 4096]))
        out.append(x[pos:pos + n])
        pos += n
    return out


@pytest.mark.parametrize("complex_", [False, True])
@pytest.mark.parametrize("name", list(BLOCK_SPECS))
def test_block_filtering_equals_one_shot(name, complex_):
    rng = np.random.default_rng(1)
    x = 0.7 + rng.standard_normal(20_000)
    if complex_:
        x = x + 1j * rng.standard_normal(20_000)
    d = F.design(BLOCK_SPECS[name])
    one_shot = F.FilterRunner(d).process(x)
    runner = F.FilterRunner(d)
    blocks = np.concatenate([runner.process(b) for b in _random_blocks(x, rng)])
    scale = np.abs(one_shot).max()
    assert blocks.shape == one_shot.shape
    assert np.max(np.abs(blocks - one_shot)) <= 1e-9 * scale


def test_chain_block_equals_one_shot():
    rng = np.random.default_rng(2)
    x = rng.standard_normal(30_000) + 2.0
    designs = [F.design(BLOCK_SPECS[k]) for k in ("dc_block", "butter_bp", "notch3", "median")]
    one_shot = F.ChainRunner(designs).process(x)
    runner = F.ChainRunner(designs)
    blocks = np.concatenate([runner.process(b) for b in _random_blocks(x, rng)])
    assert np.max(np.abs(blocks - one_shot)) <= 1e-9 * np.abs(one_shot).max()


def test_start_is_steady_state_no_transient():
    x = np.full(5000, 3.0)
    low = F.FilterRunner(F.design(BLOCK_SPECS["ellip12"])).process(x)
    assert np.max(np.abs(low - 3.0 * 10 ** (-0.5 / 20))) < 1e-6  # Eliptik çift derecede DC'de −ripple
    high = F.FilterRunner(F.design(replace(BASE, method="butter", band="highpass", order=6))).process(x)
    assert np.max(np.abs(high)) < 1e-9
    fir = F.FilterRunner(F.design(BLOCK_SPECS["fir_short"])).process(x)
    assert np.max(np.abs(fir - 3.0)) < 1e-9


def test_median_matches_scipy_medfilt_with_delay():
    rng = np.random.default_rng(3)
    x = rng.standard_normal(2000)
    d = F.design(BLOCK_SPECS["median"])
    causal = F.FilterRunner(d).process(x)
    centered = signal.medfilt(x, 9)
    assert np.array_equal(causal[8:], centered[4:-4])  # nedensel = merkezli, (L−1)/2 örnek gecikmeli
    assert np.array_equal(F.zero_phase(d, x)[4:-4], centered[4:-4])


def test_savgol_and_moving_average_match_scipy():
    rng = np.random.default_rng(4)
    x = rng.standard_normal(3000)
    sg = F.FilterRunner(F.design(BLOCK_SPECS["savgol"])).process(x)
    assert np.allclose(sg[20:], signal.savgol_filter(x, 21, 3)[10:-10], atol=1e-12)
    ma = F.FilterRunner(F.design(BLOCK_SPECS["moving_avg"])).process(x)
    assert np.allclose(ma[15:], np.convolve(x, np.ones(16) / 16, mode="valid"), atol=1e-12)


# ============================================================================ sıfır fazlı ve özel filtreler
def test_zero_phase_has_no_delay_and_squared_magnitude():
    t = np.arange(40_000) / FS
    x = np.sin(2 * np.pi * 1500 * t)
    for spec in (replace(BASE, method="butter", band="bandpass", order=4),
                 replace(BASE, family="fir", method="hamming", band="bandpass", order=200)):
        d = F.design(spec)
        y = F.zero_phase(d, x)
        g = abs(F.analyze(d, np.array([1500.0])).h[0])
        mid = slice(10_000, 30_000)
        assert np.max(np.abs(y[mid] - g ** 2 * x[mid])) < 1e-6


def test_notch_removes_mains_and_keeps_signal():
    d = F.design(replace(BASE, family="special", method="notch", notch_f0=50.0, notch_q=30.0, harmonics=2))
    r = F.analyze(d, np.array([50.0, 100.0, 1000.0, 0.0]))
    assert r.magnitude_db[0] < -100 and r.magnitude_db[1] < -100
    assert abs(r.magnitude_db[2]) < 0.01 and abs(r.magnitude_db[3]) < 0.01
    t = np.arange(50_000) / FS
    y = F.FilterRunner(d).process(np.sin(2 * np.pi * 50 * t) + 0.2 * np.sin(2 * np.pi * 1000 * t))
    tail = y[-10_000:]
    assert np.abs(tail - 0.2 * np.sin(2 * np.pi * 1000 * t[-10_000:])).max() < 0.01


def test_dc_blocker_removes_dc_and_keeps_high_frequencies():
    d = F.design(replace(BASE, family="special", method="dc_block", dc_cutoff=5.0))
    r = F.analyze(d, np.array([0.0, 5.0, FS / 2]))
    assert r.magnitude_db[0] < -200
    assert r.magnitude_db[1] == pytest.approx(-3.0103, abs=1e-3)
    assert r.magnitude_db[2] == pytest.approx(0.0, abs=1e-9)


# ============================================================================ katsayılar (kabul kriteri 4)
ELLIP = F.design(replace(BASE, method="ellip", order=8, ripple_db=0.5, atten_db=60))


def _c_arrays(text):
    body = re.search(r"=\s*\{(.*?)\};", text, re.S).group(1)
    return [float(v.rstrip("f")) for v in re.findall(r"-?\d[\d.eE+-]*f?", re.sub(r"/\*.*?\*/", "", body))]


def test_cmsis_float_biquads_have_negated_feedback_coefficients():
    values = np.array(_c_arrays(C.to_c(ELLIP, "float32", cmsis=True)), dtype=np.float64).reshape(-1, 5)
    expected = np.column_stack((ELLIP.sos[:, :3], -ELLIP.sos[:, 4:]))
    assert np.allclose(values, expected, rtol=1e-6, atol=1e-9)


def test_cmsis_q15_layout_postshift_and_reconstruction():
    text = C.to_c(ELLIP, "q15", cmsis=True)
    shift = int(re.search(r"POST_SHIFT (\d+)", text).group(1))
    codes = np.array(_c_arrays(text), dtype=np.int64).reshape(-1, 6)
    assert np.all(codes[:, 1] == 0)                       # {b0, 0, b1, b2, a1, a2}
    assert codes.min() >= -32768 and codes.max() <= 32767
    q = C.quantize(ELLIP, "q15")
    assert shift == q.post_shift >= 1                       # a1 ≈ −1.9 → [−1, 1)'e sığmaz
    real = codes[:, [0, 2, 3, 4, 5]] * 2.0 ** shift / 32768
    rebuilt = np.column_stack((real[:, :3], np.ones(len(real)), -real[:, 3:]))
    assert np.allclose(rebuilt, q.sos)
    ideal = C.scale_sections(ELLIP.sos)
    assert np.max(np.abs(rebuilt - ideal)) <= 2.0 ** shift / 32768 / 2 + 1e-12


def test_section_scaling_preserves_response_and_bounds_internal_gain():
    scaled = C.scale_sections(ELLIP.sos)
    w = np.linspace(0, np.pi, 4096)
    assert np.allclose(signal.freqz_sos(scaled, worN=w)[1], signal.freqz_sos(ELLIP.sos, worN=w)[1], atol=1e-10)
    dense = np.linspace(0, np.pi, 1 << 20)  # gerçek tepe (ölçekleme kaba ızgarayla sınırlı değil)
    cumulative = np.ones(len(dense), complex)
    for sec in scaled[:-1]:
        cumulative *= signal.freqz(sec[:3], sec[3:], worN=dense)[1]
        assert np.abs(cumulative).max() == pytest.approx(1.0, abs=1e-6)
        assert np.abs(cumulative).max() <= 1.0 + 1e-9


def test_quantization_deviation_is_reported():
    q15 = C.quantization_report(ELLIP, C.quantize(ELLIP, "q15"))
    q31 = C.quantization_report(ELLIP, C.quantize(ELLIP, "q31"))
    f32 = C.quantization_report(ELLIP, C.quantize(ELLIP, "float32"))
    assert q15.dev_db > 1e-3 and q15.stable                 # Q15'te sapma grafikte görülebilir düzeyde
    assert q31.dev_db < 1e-4 and f32.dev_db < 1e-3
    notch = F.design(replace(BASE, family="special", method="notch"))  # 10 kHz'de 50 Hz: Q15 için çok dar
    assert C.quantization_report(notch, C.quantize(notch, "q15")).pass_dev_db > 1.0


def test_cmsis_fir_is_time_reversed_and_q15_padded_even():
    d = F.design(replace(BASE, family="fir", method="remez", order=30))  # 31 katsayı
    b = d.b + np.linspace(0, 1e-3, len(d.b))  # simetriyi boz: ters sıra ancak simetrik olmayan filtrede görünür
    asym = F.Design(d.spec, b=b, order=len(b) - 1)
    f32 = np.array(_c_arrays(C.to_c(asym, "float32", cmsis=True)))
    assert np.allclose(f32, b[::-1], rtol=1e-6, atol=1e-9)
    plain = np.array(_c_arrays(C.to_c(asym, "float64", cmsis=False)))
    assert np.allclose(plain, b, rtol=0, atol=1e-15)
    q15 = np.array(_c_arrays(C.to_c(asym, "q15", cmsis=True)))
    assert len(q15) == 32 and q15[0] == 0                   # sona eklenen sıfır, ters sırada başta
    assert np.array_equal(q15[1:], np.round(b * 32768)[::-1])


def test_csv_export_round_trips_float64():
    lines = [ln for ln in C.to_csv(ELLIP).splitlines() if not ln.startswith("#")]
    assert lines[0] == "b0,b1,b2,a0,a1,a2"
    rows = np.array([[float(v) for v in ln.split(",")] for ln in lines[1:]])
    assert np.array_equal(rows, ELLIP.sos)


def test_median_has_no_coefficients():
    with pytest.raises(F.FilterError):
        C.quantize(F.design(BLOCK_SPECS["median"]), "q15")


# ============================================================================ işlem zinciri
def test_chain_json_round_trip_and_immutability():
    chain = ProcessingChain().added(ChainStage(BLOCK_SPECS["dc_block"])).added(ChainStage(BLOCK_SPECS["notch3"], False))
    again = ProcessingChain.from_dict(chain.to_dict())
    assert again == chain and again.enabled_count == 1
    moved = chain.moved(1, 0)
    assert moved.stages[0].spec == BLOCK_SPECS["notch3"] and chain.stages[0].spec == BLOCK_SPECS["dc_block"]


def test_chain_is_redesigned_for_each_sample_rate():
    chain = ProcessingChain([ChainStage(replace(BASE, method="butter", f1=3000.0, order=4))])
    designs, errors = chain.designs(48_000.0)
    assert not errors and designs[0].fs == 48_000.0
    r = F.analyze(designs[0], np.array([3000.0]))
    assert r.magnitude_db[0] == pytest.approx(-3.0103, abs=1e-3)  # kesim Hz olarak korunur
    designs, errors = chain.designs(4_000.0)                       # 3 kHz > Nyquist (2 kHz)
    assert not designs and "Nyquist" in errors[0]


# ============================================================================ özel filtreler (spesifik tasarımlar)
def _special(method, **kw):
    return F.design(F.FilterSpec(fs=FS, family="special", method=method, **kw))


def test_ema_is_exact_recursion_with_minus_3db_at_cutoff():
    d = _special("ema", dc_cutoff=40.0)
    assert F.analyze(d, np.array([40.0])).magnitude_db[0] == pytest.approx(-3.0103, abs=1e-6)
    alpha = d.sos[0, 0]
    x = np.random.default_rng(20).standard_normal(3000)
    y, acc = np.empty_like(x), x[0]
    for i, v in enumerate(x):
        acc += alpha * (v - acc)
        y[i] = acc
    assert np.allclose(F.FilterRunner(d).process(x), y, atol=1e-12)


@pytest.mark.parametrize("model", ["cv", "rw"])
def test_steady_state_kalman_equals_time_varying_kalman_after_convergence(model):
    q, r = 5.0, 0.2
    d = _special("kalman", kalman_model=model, kalman_q=q, kalman_r=r)
    T = 1 / FS
    if model == "cv":
        Fm, H = np.array([[1, T], [0, 1]]), np.array([[1.0, 0.0]])
        Q = q ** 2 * np.array([[T ** 3 / 3, T ** 2 / 2], [T ** 2 / 2, T]])
    else:
        Fm, H, Q = np.array([[1.0]]), np.array([[1.0]]), np.array([[q ** 2 * T]])
    R = np.array([[r ** 2]])
    rng = np.random.default_rng(21)
    z = np.cumsum(rng.standard_normal(20_000)) * 0.01 + r * rng.standard_normal(20_000)
    x = np.zeros((len(Fm), 1))
    x[0, 0] = z[0]
    P = np.eye(len(Fm)) * 10.0
    out = np.empty_like(z)
    for i, zi in enumerate(z):  # klasik (zamanla değişen) Kalman
        x, P = Fm @ x, Fm @ P @ Fm.T + Q
        K = P @ H.T @ np.linalg.inv(H @ P @ H.T + R)
        x = x + K @ (zi - H @ x)
        P = (np.eye(len(Fm)) - K @ H) @ P
        out[i] = (H @ x)[0, 0]
    lti = F.FilterRunner(d).process(z)
    tail = slice(15_000, None)  # kazanç oturduktan sonra birebir aynı olmalı
    assert np.max(np.abs(lti[tail] - out[tail])) < 1e-6 * np.abs(out).max()


def test_alpha_beta_equals_explicit_recursion():
    a, b = 0.4, 0.08
    d = _special("alpha_beta", ab_alpha=a, ab_beta=b)
    T = 1 / FS
    z = np.cumsum(np.random.default_rng(22).standard_normal(4000))
    xh, vh, out = z[0], 0.0, []
    for zi in z:
        xp = xh + T * vh
        res = zi - xp
        xh, vh = xp + a * res, vh + b / T * res
        out.append(xh)
    lti = F.FilterRunner(d).process(z)
    assert np.max(np.abs(lti[200:] - np.array(out)[200:])) < 1e-6 * np.abs(z).max()
    with pytest.raises(F.FilterError, match="kararsız"):
        _special("alpha_beta", ab_alpha=1.5, ab_beta=1.5)


def test_comb_notches_every_harmonic_and_peak_passes_only_harmonics():
    notch = _special("comb", notch_f0=50.0, notch_q=30.0, comb_type="notch")
    k = np.arange(1, 99)
    r = F.analyze(notch, np.concatenate((k * 50.0, (k + 0.5) * 50.0)))
    assert r.magnitude_db[:len(k)].max() < -100
    assert np.abs(r.magnitude_db[len(k):]).max() < 0.1
    peak = _special("comb", notch_f0=50.0, notch_q=30.0, comb_type="peak")
    r = F.analyze(peak, np.concatenate((k * 50.0, (k + 0.5) * 50.0)))
    assert np.abs(r.magnitude_db[:len(k)]).max() < 1e-6 and r.magnitude_db[len(k):].max() < -20
    with pytest.raises(F.FilterError, match="tam sayı"):
        _special("comb", notch_f0=33.0)


def test_allpass_has_unit_gain_and_minus_180_degrees_at_f0():
    d = _special("allpass", notch_f0=700.0, notch_q=2.0)
    f = np.linspace(1, 4999, 5000)
    r = F.analyze(d, f)
    assert np.abs(r.magnitude_db).max() < 1e-9
    ph = np.degrees(np.angle(F.analyze(d, np.array([700.0])).h[0]))
    assert abs(abs(ph) - 180.0) < 1e-6


def test_differentiator_gives_true_derivative_in_passband():
    d = _special("differentiator", order=101, f1=500.0, transition=500.0)
    f = np.array([50.0, 150.0, 300.0, 450.0])
    gain = np.abs(F.analyze(d, f).h)
    assert np.allclose(gain, 2 * np.pi * f, rtol=0.01)
    t = np.arange(20_000) / FS
    x = 2.0 * np.sin(2 * np.pi * 120 * t)
    y = F.FilterRunner(d).process(x)
    # d/dt [2·sin(2π·120·t)] = 2·2π·120·cos(…), doğrusal fazlı FIR'in (N−1)/2 örnek gecikmesiyle
    expected = 2.0 * 2 * np.pi * 120 * np.cos(2 * np.pi * 120 * (t - d.delay_samples / FS))
    assert np.max(np.abs(y[2000:] - expected[2000:])) < 0.01 * 2.0 * 2 * np.pi * 120


def test_leaky_integrator_integrates_above_corner():
    d = _special("integrator", dc_cutoff=1.0)
    t = np.arange(50_000) / FS
    f0 = 100.0
    y = F.FilterRunner(d).process(np.sin(2 * np.pi * f0 * t))
    amplitude = (y[-20_000:].max() - y[-20_000:].min()) / 2
    assert amplitude == pytest.approx(1 / (2 * np.pi * f0), rel=0.02)  # ∫sin = −cos/(2πf)


def test_hilbert_matches_analytic_signal_imaginary_part():
    d = _special("hilbert", order=200, f1=100.0)
    t = np.arange(30_000) / FS
    x = np.sin(2 * np.pi * 800 * t) + 0.5 * np.cos(2 * np.pi * 2300 * t)
    y = F.FilterRunner(d).process(x)
    delay = int(d.delay_samples)
    ideal = np.imag(signal.hilbert(x))
    mid = slice(5000, 25_000)
    err = y[mid.start + delay:mid.stop + delay] - ideal[mid]
    assert np.max(np.abs(err)) < 0.01 * np.abs(ideal[mid]).max()


def test_gaussian_has_unit_dc_gain_and_expected_bandwidth():
    d = _special("gaussian", length=61, gauss_sigma=6.0)
    assert d.b.sum() == pytest.approx(1.0) and np.allclose(d.b, d.b[::-1])
    f = np.linspace(0, FS / 2, 200_001)
    mag = F.analyze(d, f).magnitude_db
    f3 = f[np.argmax(mag <= -3.0103)]
    assert f3 == pytest.approx(0.1325 * FS / 6.0, rel=0.01)


@pytest.mark.parametrize("method", ["ema", "kalman", "alpha_beta", "comb", "allpass", "integrator",
                                    "differentiator", "hilbert", "gaussian"])
def test_special_filters_run_in_blocks_and_export(method):
    d = _special(method, order=64, f1=200.0, transition=500.0, length=31, notch_f0=50.0)
    x = np.random.default_rng(23).standard_normal(8000) + 1.0
    one = F.FilterRunner(d).process(x)
    runner = F.FilterRunner(d)
    blocks = np.concatenate([runner.process(x[i:i + 333]) for i in range(0, len(x), 333)])
    assert np.max(np.abs(blocks - one)) <= 1e-9 * max(np.abs(one).max(), 1e-12)
    assert "static const" in C.to_c(d, "float32") and C.to_csv(d).count("\n") > 3


# ============================================================================ DSP denetimi regresyonları
def test_numeric_design_failures_become_filter_errors():
    # remez yakınsamazsa / geçersiz parametrede kullanıcı ham scipy istisnası değil anlaşılır bir FilterError görür
    with pytest.raises(F.FilterError, match="Δf pozitif"):
        _special("differentiator", order=40, f1=400.0, transition=0.0)
    with pytest.raises(F.FilterError):
        _special("differentiator", order=400, f1=4000.0, transition=500.0)  # f₁ + Δf ≥ Nyquist
    with pytest.raises(F.FilterError):
        _special("hilbert", order=64, f1=2400.0)  # f₁ ≥ 0.45·Nyquist


def test_differentiator_description_reports_adjusted_order():
    d = _special("differentiator", order=40, f1=300.0, transition=100.0)
    assert len(d.b) == 42 and d.order == 41  # tip IV için çift katsayı
    assert "41. derece" in d.describe() and "40. derece" not in d.describe()
    assert any("41" in w for w in d.warnings)


@pytest.mark.parametrize("model", ["rw", "cv"])
def test_kalman_gain_depends_only_on_noise_ratio(model):
    base = _special("kalman", kalman_model=model, kalman_q=2.0, kalman_r=0.5)
    for scale in (1e-3, 1e3, 1e6):
        d = _special("kalman", kalman_model=model, kalman_q=2.0 * scale, kalman_r=0.5 * scale)
        assert np.allclose(d.sos, base.sos, rtol=1e-9, atol=1e-15)


def test_kalman_extreme_ratio_is_a_clean_error():
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # scipy'nin ara uyarıları kullanıcıya sızmamalı
        with pytest.raises(F.FilterError, match="Kalman kazancı"):
            F.design(F.FilterSpec(fs=1e8, family="special", method="kalman", kalman_model="cv",
                                  kalman_q=1e-6, kalman_r=1e3))


@pytest.mark.parametrize("ftype", ["notch", "peak"])
@pytest.mark.parametrize("fs", [48_000.0, 204_800.0])
def test_comb_closed_form_matches_scipy(ftype, fs):
    import time
    t = time.perf_counter()
    d = F.design(F.FilterSpec(fs=fs, family="special", method="comb", notch_f0=50.0, notch_q=30.0, comb_type=ftype))
    assert time.perf_counter() - t < 10.0  # N = fs/f₀ = 4096'ya kadar kapalı biçim (eski yol dakikalar sürüyordu)
    b, a = signal.iircomb(50.0, 30.0, ftype=ftype, fs=fs, pass_zero=ftype == "peak")
    f = np.linspace(1.0, fs / 2 - 1.0, 4001)
    ref = signal.freqz(b, a, worN=f, fs=fs)[1]
    got = signal.freqz_sos(d.sos, worN=f, fs=fs)[1]
    assert np.max(np.abs(got - ref)) < 1e-7 and d.stable


def test_linf_scaling_holds_for_very_narrow_band():
    d = F.design(F.FilterSpec(fs=48_000.0, family="iir", method="butter", band="bandpass", order=6,
                              f1=1000.0, f2=1010.0))
    sc = C.scale_sections(d.sos)
    w = np.linspace(0, np.pi, 1 << 21)
    cum = np.ones(len(w), complex)
    for sec in sc[:-1]:
        cum *= signal.freqz(sec[:3], sec[3:], worN=w)[1]
        assert 0.999 < np.abs(cum).max() <= 1.0 + 1e-6  # her ara düğümde tepe tam 1 (dar tepe kaçmıyor)
    assert np.allclose(signal.freqz_sos(sc, worN=w[::997])[1], signal.freqz_sos(d.sos, worN=w[::997])[1])


@pytest.mark.parametrize("fs,fc", [(1e6, 0.01), (1e6, 0.001), (1e9, 1.0), (1000.0, 400.0)])
def test_ema_cutoff_exact_at_extreme_ratios(fs, fc):
    d = F.design(F.FilterSpec(fs=fs, family="special", method="ema", dc_cutoff=fc))
    assert F.analyze(d, np.array([fc])).magnitude_db[0] == pytest.approx(-3.0103, abs=1e-4)


@pytest.mark.parametrize("method", list(F.PHASE_DEFINED))
def test_zero_phase_refuses_phase_defined_filters(method):
    d = _special(method, order=64, f1=200.0, transition=500.0, notch_f0=700.0, dc_cutoff=5.0)
    with pytest.raises(F.FilterError, match="sıfır fazlı"):
        F.zero_phase(d, np.ones(1000))


def test_zero_phase_on_segment_shorter_than_filter_is_nan():
    d = F.design(replace(BASE, family="fir", method="hamming", band="lowpass", order=100))
    assert np.all(np.isnan(F.zero_phase(d, np.ones(50))))
    iir = F.design(replace(BASE, family="iir", method="butter", band="lowpass", order=4))
    assert np.all(np.isnan(F.zero_phase(iir, np.ones(1))))
    assert np.all(np.isfinite(F.zero_phase(iir, np.ones(5))))


def test_finite_runs_splits_on_nan_and_inf():
    x = np.array([1.0, 2.0, np.nan, np.nan, 3.0, np.inf, 4.0])
    assert F.finite_runs(x) == [(0, 2, True), (2, 4, False), (4, 5, True), (5, 6, False), (6, 7, True)]
    assert F.finite_runs(np.ones(3)) == [(0, 3, True)] and F.finite_runs(np.ones(0)) == []


@pytest.mark.parametrize("fmt", ["q15", "q31"])
def test_fixed_point_fir_with_large_taps_uses_post_shift(fmt):
    d = _special("differentiator", order=40, f1=300.0, transition=100.0)
    assert np.abs(d.b).max() > 1.0  # birim/s ölçekli türev alıcı: katsayılar ≫ 1
    q = C.quantize(d, fmt)
    assert q.post_shift > 0 and q.clipped == 0
    one = 2.0 ** (C.FIXED_BITS[fmt] - 1)
    assert np.abs(q.codes).max() <= one and np.array_equal(q.b, q.codes * 2.0 ** q.post_shift / one)
    rep = C.quantization_report(d, q)
    assert rep.pass_dev_db < (0.01 if fmt == "q15" else 1e-6)
    header = C.to_c(d, fmt)
    assert f"FLT_POST_SHIFT {q.post_shift}" in header and f"arm_shift_{fmt}(dst, FLT_POST_SHIFT" in header
    assert "/*" not in header.split("/* Kullanım:")[1].split("*/")[0]  # kullanım yorumunun içinde iç içe yorum yok


def test_fixed_point_fir_within_unit_range_needs_no_shift():
    d = F.design(replace(BASE, family="fir", method="hamming", band="lowpass", order=64))
    q = C.quantize(d, "q15")
    assert q.post_shift == 0 and "arm_shift" not in C.to_c(d, "q15")


# ============================================================================ performans denetimi regresyonları
def test_comb_analysis_is_closed_form_and_fast():
    import time
    d = F.design(F.FilterSpec(fs=48_000.0, family="special", method="comb", notch_f0=50.0, notch_q=30.0))
    start = time.perf_counter()
    r = F.analyze(d)
    assert time.perf_counter() - start < 1.0  # eskiden 480 bölümde dakikalar
    f = np.linspace(10.0, 23_990.0, 3001)
    h_sos = signal.freqz_sos(d.sos, worN=f, fs=48_000.0)[1]
    assert np.max(np.abs(F.analyze(d, f).h - h_sos)) < 1e-8
    assert np.isfinite(r.group_delay).sum() > 0.9 * len(r.freqs)


def test_vectorized_group_delay_matches_scipy():
    d = F.design(F.FilterSpec(fs=FS, family="iir", method="ellip", band="bandpass", order=5, f1=1000.0, f2=2000.0,
                              ripple_db=0.5, atten_db=60.0))
    f = np.linspace(500.0, 2500.0, 400)
    ref = signal.group_delay(signal.sos2tf(d.sos), w=f, fs=FS)[1]
    assert np.max(np.abs(F.analyze(d, f).group_delay - ref)) < 1e-3


def test_section_scaling_is_linear_in_section_count():
    import time
    d = F.design(F.FilterSpec(fs=10_000.0, family="special", method="comb", notch_f0=50.0, notch_q=30.0))
    start = time.perf_counter()
    C.quantize.cache_clear()
    q = C.quantize(d, "q15")
    C.quantization_report(d, q)
    assert time.perf_counter() - start < 5.0  # 100 bölüm: eskiden 25 s+
    assert C.quantize(d, "q15") is q           # aynı tasarım / biçim tekrar kuantize edilmez
