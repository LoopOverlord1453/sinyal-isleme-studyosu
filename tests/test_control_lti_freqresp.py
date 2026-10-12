"""Frekans cevabı ve paylar: freqresp, evalfr, bode (MATLAB faz açma kuralı), nyquist, sigma, margin, allmargin,
bandwidth, dcgain, norm (H2 / H∞), getPeakGain.

Referans: MATLAB R2025b (tests/data/matlab/lti_freqresp.json). Genlik / faz / tekil değer / H2 / H∞ (tol 1e-12
ile alınmış MATLAB değeri) / bant genişliği göreli 1e-9. MATLAB'in margin / allmargin geçiş frekansları yaklaşıktır
(ölçülen göreli hata ≤ 5e-4; ör. 2/(s−1) için Wcp = 1.7320003, kesin √3): bu yüzden MATLAB karşılaştırması 1e-3 ile,
kesinlik ise analitik durumlar (1e-10) ve her geçişte artık (|G| = 1 ya da ∠G = −180°) kontrolüyle (1e-10) yapılır.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from spt.control.lti import ss, tf, zpk
from spt.control.freqresp import (allmargin, bandwidth, bode, dcgain, evalfr, freqresp, getPeakGain, margin, norm,
                                  nyquist, sigma)
from tools.matlab_oracle import OracleUnavailable, fixture


def _m(x) -> str:
    if isinstance(x, complex):
        return f"complex({float(x.real)!r},{float(x.imag)!r})"
    return repr(float(x))


def _mv(v) -> str:
    return "[" + " ".join(_m(x) for x in np.atleast_1d(v)) + "]"


def _mm(M) -> str:
    M = np.atleast_2d(np.asarray(M, float))
    return "[" + ";".join(" ".join(_m(x) for x in row) for row in M) + "]"


def _mcell(grid) -> str:
    return "{" + ";".join(",".join(_mv(p) for p in row) for row in grid) + "}"


def _nv(d) -> str:
    s = ""
    for key, mk in (("ind", "InputDelay"), ("outd", "OutputDelay"), ("io", "IODelay")):
        if key in d:
            v = d[key]
            s += f",'{mk}'," + (_mm(v) if np.ndim(v) == 2 else _mv(v))
    return s


def _kw(d) -> dict:
    return {pk: d[k] for k, pk in (("ind", "input_delay"), ("outd", "output_delay"), ("io", "io_delay")) if k in d}


def model(spec):
    kind = spec[0]
    if kind == "tf":
        _, num, den, Ts, d = spec
        grid = isinstance(num, list) and num and isinstance(num[0], list) and isinstance(num[0][0], list)
        mx = f"tf({_mcell(num) if grid else _mv(num)},{_mcell(den) if grid else _mv(den)},{_m(Ts)}{_nv(d)})"
        return tf(num, den, Ts, **_kw(d)), mx
    if kind == "zpk":
        _, z, p, k, Ts, d = spec
        zs = _mv([complex(v) for v in z]) if len(z) else "[]"
        ps = _mv([complex(v) for v in p]) if len(p) else "[]"
        return zpk(z, p, k, Ts, **_kw(d)), f"zpk({zs},{ps},{_m(k)},{_m(Ts)}{_nv(d)})"
    _, A, B, C, D, Ts, d = spec
    return ss(A, B, C, D, Ts, **_kw(d)), f"ss({_mm(A)},{_mm(B)},{_mm(C)},{_mm(D)},{_m(Ts)}{_nv(d)})"


def TF(num, den, Ts=0.0, **d):
    return ("tf", num, den, Ts, d)


def ZK(z, p, k, Ts=0.0, **d):
    return ("zpk", z, p, k, Ts, d)


def SS(A, B, C, D, Ts=0.0, **d):
    return ("ss", A, B, C, D, Ts, d)


A3 = [[0, 1, 0], [0, 0, 1], [-6, -11, -6]]
W = np.array([0.001, 0.01, 0.1, 0.5, 1.0, 1.7, 3.0, 7.0, 20.0, 100.0, 1000.0])
BODE = {
    "b_int3": TF([1], [1, 0, 0, 0]),
    "b_neg": TF([-1], [1, 1]),
    "b_rhp": TF([1], [1, -1]),
    "b_nmp": TF([1, -1], [1, 1]),
    "b_dly": TF([1], [1, 1], ind=1.0),
    "b_res": ZK([0], [-0.05 + 1j, -0.05 - 1j, -0.05 + 1j, -0.05 - 1j], -1.0),
    "b_und": TF([1], [1, 0, 1]),
    "b_h8": ZK([-1.5, 2 + 1j, 2 - 1j], [-0.5 + 3j, -0.5 - 3j, -1 + 1j, -1 - 1j, -2, -3, -4 + 0.5j, -4 - 0.5j], 7.0),
    "b_mimo": TF([[[1], [-1]], [[1], [1, -2]]], [[[1, 1], [1, 0]], [[1, 0, 0], [1, 2]]], io=[[0.0, 0.2], [0.1, 0.0]]),
    "b_ss": SS(A3, [[0, 1], [1, 0], [0, 2]], [[1, 0, 0], [0, 1, 1]], [[0.3, 0.5], [0, 0]], ind=[0.25, 0.0]),
    "b_z1": TF([1], [1, -1.5, 0.7], 0.1),
    "b_z2": ZK([-0.5, 0.9], [0.95, -0.3 + 0.6j, -0.3 - 0.6j], -2.0, 0.05, io=3),
    "b_z3": TF([1, 0.5], [1, -1], 0.2),
    "b_up": ZK([-0.3], [0.5 + 2j, 0.5 - 2j, -1], 3.0),
    "b_rz": ZK([1 + 1j, 1 - 1j, 0.5], [-1, -2, -3, -0.2 + 4j, -0.2 - 4j], -2.0),
    "b_zo": ZK([1.2 + 0.5j, 1.2 - 0.5j], [0.5, -0.2, 1.1 + 0.6j, 1.1 - 0.6j], 1.0, 0.1),
}
SIGMA = {"s_ss": BODE["b_ss"], "s_tf": BODE["b_mimo"],
         "s_z": SS([[0.5, 0.1], [-0.2, 0.7]], [[1, 0], [0, 1]], [[1, 0], [0.5, 1], [0, 1]], [[0, 0.2], [0, 0], [1, 0]],
                   0.2)}
MARG = {
    "m_cube": ZK([], [-1, -1, -1], 1.0),
    "m_rhp": TF([2], [1, -1]),
    "m_dint": TF([10, 10], [1, 10, 0, 0]),
    "m_int": TF([4], [1, 2, 0]),
    "m_res": TF([100], [1, 2, 100]),
    "m_h8": ZK([-1.5, -2 + 1j, -2 - 1j], [-0.5 + 3j, -0.5 - 3j, -1 + 1j, -1 - 1j, -2, -3, -4 + 0.5j, -4 - 0.5j],
               150.0),
    "m_cond": ZK([-0.2, -0.3], [0, -0.01, -5, -6, -7], 200.0),
    "m_ucl": ZK([-3], [0, 1], 2.5),
    "m_lag": ZK([-0.1], [0, -0.01, -2], 5.0),
    "m_d1": TF([2], [1, 1], ind=0.5),
    "m_d2": TF([5], [1, 2, 5], ind=0.3),
    "m_d3": TF([1], [1, 0], ind=1.0),
    "m_d4": ZK([-1], [-0.2, -3, -5], 12.0, ind=0.2),
    "m_z1": TF([0.5], [1, -0.8], 0.1),
    "m_z2": TF([0.5], [1, -0.8], 0.1, io=2),
    "m_z3": TF([0.2, 0.1], [1, -1.2, 0.5], 0.05),
    "m_z4": TF([1, 0.5], [1, -0.5, 0.06], 0.2),
    "m_z5": TF([0.8], [1, -1], 0.1),
    "m_z6": TF([1.5], [1, 0.2], 0.1),
    "m_z7": ZK([-0.5], [1, 0.6, 0.2], 0.3, 0.05),
}
BW = {"w1": (TF([1], [1, 1]), -3.0), "w2": (TF([1], [1, 1]), -6.0), "w3": (TF([10], [1, 0.4, 4]), -3.0),
      "w4": (TF([1], [1, -0.5], 0.1), -3.0), "w5": (TF([0.01], [1, -0.99], 0.1), -3.0),
      "w6": (TF([2, 1], [1, 3, 2], ind=0.4), -3.0), "w7": (TF([1, 0], [1, 1]), -3.0), "w8": (TF([1], [1, 0]), -3.0),
      "w9": (BODE["b_h8"], -10.0)}
DC = {"g1": TF([2, 1], [1, 3, 2]), "g2": TF([1], [1, 0]), "g3": TF([-1], [1, 0, 0]), "g4": TF([1, 0], [1, 1]),
      "g5": TF([1], [1, -0.5], 0.1), "g6": TF([1], [1, -1], 0.1), "g7": BODE["b_ss"], "g8": BODE["b_mimo"],
      "g9": ZK([0], [0, -1], 2.0), "g10": SS([[0, 1], [0, -1]], [[0], [1]], [[1, 0], [0, 1]], [[0], [0]])}
NORM = {"n1": TF([1], [1, 0.2, 1]), "n2": TF([2, 1], [1, 3, 2]), "n3": BODE["b_ss"], "n4": BODE["b_h8"],
        "n5": TF([1], [1, -1]), "n6": TF([1, 0], [1, -0.5], 0.1), "n7": SIGMA["s_z"],
        "n8": TF([1, 0, 1], [1, 2, 1]), "n9": TF([0.2, 0.1], [1, -1.2, 0.5], 0.05),
        "n10": SS([[-0.01, 1], [-1, -0.01]], [[0, 1], [1, 0]], [[1, 0], [0, 1]], [[0, 0], [0, 0.5]]),
        "n11": TF([1], [1, 0, 1]), "n12": TF([1], [1, 0])}
EVAL = {"e1": (BODE["b_mimo"], [0.3 + 1j, -2 + 0.5j]), "e2": (BODE["b_z2"], [1.2 + 0.3j, 0.5j])}


def _wfor(sys):
    return W[W < math.pi / sys.Ts] if sys.is_discrete else W


def _script() -> str:
    L = []
    for name, spec in {**BODE, **SIGMA}.items():
        py, mx = model(spec)
        w = _wfor(py)
        L.append(f"G = {mx}; w = {_mv(w)}; [m_, p_] = bode(G, w); out.{name}.mag = m_; out.{name}.ph = p_; "
                 f"[r_, i_] = nyquist(G, w); out.{name}.re = r_; out.{name}.im = i_; out.{name}.sv = sigma(G, w);")
    for name, spec in MARG.items():
        L.append(f"G = {model(spec)[1]}; [gm_, pm_, wg_, wp_] = margin(G); out.{name}.r = [gm_ pm_ wg_ wp_]; "
                 f"out.{name}.a = allmargin(G);")
    for name, (spec, db) in BW.items():
        L.append(f"out.{name} = bandwidth({model(spec)[1]}, {_m(db)});")
    for name, spec in DC.items():
        L.append(f"out.{name} = dcgain({model(spec)[1]});")
    for name, spec in NORM.items():
        L.append(f"G = {model(spec)[1]}; out.{name}.h2 = norm(G, 2); [ni_, fp_] = norm(G, inf, 1e-12); "
                 f"out.{name}.hi = [ni_ fp_]; [gp_, wp_] = getPeakGain(G, 1e-12); out.{name}.pk = [gp_ wp_];")
    for name, (spec, pts) in EVAL.items():
        L.append(f"G = {model(spec)[1]}; pts = {_mv([complex(p) for p in pts])}; for q = 1:numel(pts), "
                 f"H_ = evalfr(G, pts(q)); out.{name}.re(:,:,q) = real(H_); out.{name}.im(:,:,q) = imag(H_); end")
    out = ["out.errs = {};"]
    for k, ln in enumerate(L):
        out.append(f"try; {ln}; catch e_; out.errs{{end+1}} = sprintf('satir {k}: %s', e_.message); end")
    return "\n".join(out)


SCRIPT = _script()


@pytest.fixture(scope="module")
def M():
    try:
        return fixture("lti_freqresp", SCRIPT)
    except OracleUnavailable as e:
        pytest.skip(str(e))


def test_matlab_script_ran_cleanly(M):
    assert not M.get("errs"), M.get("errs")


def _arr(v, shape):
    return np.asarray(v, float).reshape(shape)


@pytest.mark.parametrize("name", list(BODE))
def test_bode_nyquist(M, name):
    G = model(BODE[name])[0]
    w = _wfor(G)
    mag, ph, _ = bode(G, w)
    shape = G.shape + (w.size,)
    rm = _arr(M[name]["mag"], shape)
    fin = np.isfinite(rm)
    assert np.array_equal(np.isfinite(mag), fin)                         # tam kutup üstünde sonsuz genlik
    assert np.all(np.abs(mag[fin] - rm[fin]) <= 1e-9 * rm[fin] + 1e-300)
    assert np.max(np.abs(ph[fin] - _arr(M[name]["ph"], shape)[fin])) <= 1e-7      # derece
    re, im, _ = nyquist(G, w)
    H = _arr(M[name]["re"], shape) + 1j * _arr(M[name]["im"], shape)
    assert np.all(np.abs(re + 1j * im - H)[fin] <= 1e-9 * np.abs(H)[fin] + 1e-300)
    assert np.all(np.abs(freqresp(G, w) - H)[fin] <= 1e-9 * np.abs(H)[fin] + 1e-300)


@pytest.mark.parametrize("name", ["b_mimo", "b_ss", "s_z"])
def test_sigma(M, name):
    G = model({**BODE, **SIGMA}[name])[0]
    w = _wfor(G)
    sv, _ = sigma(G, w)
    ref = _arr(M[name]["sv"], sv.shape)
    assert np.max(np.abs(sv - ref) / ref) <= 1e-10


def _list(v):
    if v is None:
        return np.zeros(0)
    return np.atleast_1d(np.asarray(v, float))


TOL_M = 1e-3     # MATLAB'in kendi yaklaşıklığı (bkz. modül belgesi)


def _residuals(G, a):
    """Kesinlik: her kazanç geçişinde |G| = 1, her faz geçişinde ∠G ≡ −180° (gecikme dahil)."""
    for w in a["PMFrequency"]:
        if 0 < w < math.inf:
            assert abs(abs(freqresp(G, [w])[0, 0, 0]) - 1.0) <= 1e-11
    for w in a["GMFrequency"]:
        if 0 < w < math.inf:
            h = freqresp(G, [w])[0, 0, 0]
            assert abs(h.imag) <= 1e-10 * abs(h) and h.real < 0


@pytest.mark.parametrize("name", list(MARG))
def test_allmargin(M, name):
    G = model(MARG[name])[0]
    a = allmargin(G)
    ref = M[name]["a"]
    _residuals(G, a)
    assert a["Stable"] == bool(ref["Stable"])
    delay_c = (not G.is_discrete) and G.has_delay()
    for kv, kf in (("GainMargin", "GMFrequency"), ("PhaseMargin", "PMFrequency"), ("DelayMargin", "DMFrequency")):
        rv, rf = _list(ref[kv]), _list(ref[kf])
        mv, mf = a[kv], a[kf]
        if delay_c and kv == "GainMargin":
            k = min(rf.size, 6)            # MATLAB'in gecikme listesi düzensiz / sınırlı: ilk geçişler
            rv, rf = rv[:k], rf[:k]
            mv, mf = mv[:k], mf[:k]
        assert mf.size == rf.size, (kv, mf, rf)
        for x, y in ((mv, rv), (mf, rf)):
            fin = np.isfinite(y)
            assert np.array_equal(np.isfinite(x), fin) and np.array_equal(np.isinf(x), np.isinf(y)), (kv, x, y)
            assert np.allclose(x[fin], y[fin], rtol=TOL_M, atol=1e-6), (kv, x, y)
    r = margin(G)
    rr = _list(M[name]["r"])
    for x, y in zip(r, rr):
        if math.isnan(y):
            assert math.isnan(x)
        elif math.isinf(y):
            assert x == y
        else:
            assert x == pytest.approx(y, rel=TOL_M, abs=1e-6)


def test_margin_exact_analytic():
    Gm, Pm, Wcg, Wcp = margin(zpk([], [-1, -1, -1], 1))
    assert Gm == pytest.approx(8.0, rel=1e-12) and Wcg == pytest.approx(math.sqrt(3), rel=1e-12)
    Gm, Pm, Wcg, Wcp = margin(tf([2], [1, -1]))
    assert Gm == pytest.approx(0.5, rel=1e-14) and Wcg == 0
    assert Pm == pytest.approx(60.0, rel=1e-12) and Wcp == pytest.approx(math.sqrt(3), rel=1e-12)
    _, Pm, _, Wcp = margin(tf([100], [1, 2, 100]))
    assert Wcp == pytest.approx(14.0, rel=1e-13) and Pm == pytest.approx(math.degrees(math.atan2(28, 96)), rel=1e-12)
    # gecikmeli: 2e^{-0.5s}/(s+1): ω_c = √3, Pm = 180 − 60 − 0.5·√3 rad
    a = allmargin(tf([2], [1, 1], input_delay=0.5))
    assert a["PMFrequency"][0] == pytest.approx(math.sqrt(3), rel=1e-13)
    assert a["PhaseMargin"][0] == pytest.approx(120 - math.degrees(0.5 * math.sqrt(3)), rel=1e-12)
    assert a["DelayMargin"][0] == pytest.approx(math.radians(a["PhaseMargin"][0]) / math.sqrt(3), rel=1e-12)
    # ayrık: 0.3/(z²−1.5z+0.7): kapalı çevrim kutupları birim çemberde → Gm = 1, Pm = 0 tam aynı frekansta
    a = allmargin(tf([0.3], [1, -1.5, 0.7], 0.1))
    w0 = math.acos(0.75) / 0.1
    assert a["GainMargin"][0] == pytest.approx(1.0, abs=1e-12) and a["GMFrequency"][0] == pytest.approx(w0, rel=1e-12)
    assert abs(a["PhaseMargin"][0]) <= 1e-9 and not a["Stable"]


@pytest.mark.parametrize("name", list(BW))
def test_bandwidth(M, name):
    spec, db = BW[name]
    G = model(spec)[0]
    r = M[name]
    b = bandwidth(G, db)
    if r is None or (isinstance(r, float) and math.isnan(r)):
        assert math.isnan(b)
    elif math.isinf(r):
        assert math.isinf(b)
    else:
        assert b == pytest.approx(r, rel=1e-9)


@pytest.mark.parametrize("name", list(DC))
def test_dcgain(M, name):
    G = model(DC[name])[0]
    g = np.asarray(dcgain(G), float)
    r = np.asarray(M[name], float).reshape(g.shape)
    fin = np.isfinite(r)
    assert np.array_equal(np.isinf(g), np.isinf(r)) and np.all(np.sign(g[~fin]) == np.sign(r[~fin]))
    assert np.allclose(g[fin], r[fin], rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("name", list(NORM))
def test_norms(M, name):
    G = model(NORM[name])[0]
    ref = M[name]
    h2 = norm(G, 2)
    if math.isinf(ref["h2"]):
        assert math.isinf(h2)
    else:
        assert h2 == pytest.approx(ref["h2"], rel=1e-10)
    for key, (v, f) in (("hi", norm(G, math.inf)), ("pk", getPeakGain(G))):
        rv, rf = _list(ref[key])
        if math.isinf(rv):
            assert math.isinf(v) and f == pytest.approx(rf, rel=1e-9, abs=1e-12)
            continue
        assert v == pytest.approx(rv, rel=1e-9)
        # tepe düz: frekans SISO'da türev kökü olarak 1e-8, MIMO'da 1e-6 (≈ √eps) doğrulukta
        assert f == pytest.approx(rf, rel=1e-8 if G.is_siso else 1e-5, abs=1e-9)


def test_hinf_analytic():
    z = 0.1
    g, w = norm(tf([1], [1, 2 * z, 1]), math.inf)
    assert g == pytest.approx(1 / (2 * z * math.sqrt(1 - z * z)), rel=1e-13)
    assert w == pytest.approx(math.sqrt(1 - 2 * z * z), rel=1e-12)
    assert norm(tf([1], [1, 1]), 2) == pytest.approx(math.sqrt(0.5), rel=1e-14)


@pytest.mark.parametrize("name", list(EVAL))
def test_evalfr(M, name):
    G, pts = model(EVAL[name][0])[0], EVAL[name][1]
    ref = M[name]
    shape = G.shape + (len(pts),)
    H = _arr(ref["re"], shape) + 1j * _arr(ref["im"], shape)
    for q, p in enumerate(pts):
        v = np.asarray(evalfr(G, p)).reshape(G.shape)
        assert np.all(np.abs(v - H[:, :, q]) <= 1e-10 * np.abs(H[:, :, q]) + 1e-300)


@pytest.mark.parametrize("Ts", [0.01, 0.001])
def test_discrete_margin_small_wTs_crossovers_are_found(Ts):
    """Regresyon: ayrık çevrimde ωTs küçükken cos θ polinomunun kökleri 1 çevresine kümelenip yuvarlamayla atılıyordu;
    margin kazanç geçişini kaçırıp Pm = inf, Wcp = nan veriyordu (ör. Ts = 1 ms, ωc = 0.01…3 rad/s, PI/PID)."""
    from spt.control.classical import pidtune
    from spt.control.discretize import c2d
    G = c2d(zpk([], [0, -2, -5], 10), Ts)
    for typ, wc in (("PI", 0.01), ("PI", 0.3), ("PID", 0.1), ("PIDF", 1.0)):
        C = pidtune(G, typ, wc)
        C = C[0] if isinstance(C, tuple) else C
        L = C * G
        Gm, Pm, Wcg, Wcp = margin(L)
        assert math.isfinite(Pm) and math.isfinite(Wcp), (typ, wc)
        h = freqresp(L, [Wcp])[0, 0, 0]
        assert abs(abs(h) - 1.0) < 1e-9
        assert abs(((math.degrees(np.angle(h)) + 180.0 - Pm) + 180.0) % 360.0 - 180.0) < 1e-6


# ---------------------------------------------------------------------------------------------------------------
# Bağımsız denetim regresyonları (MATLAB R2025b değerleri denetçi oturumundan: spt_denetim/kisa_lti/ml1.json)
# ---------------------------------------------------------------------------------------------------------------
def test_delay_margin_sign_rule():
    """Kararsız kapalı çevrim: çevrimde gecikme varsa DM = Pm/ω (negatif), yoksa (Pm mod 360°)/ω (MATLAB)."""
    from spt.control.discretize import c2d
    a = allmargin(tf([0.5], [1, -0.9], 0.1, io_delay=3))            # MATLAB −0.120422 (yaklaşık); kesin −0.1203571
    assert not a["Stable"]
    assert a["DelayMargin"][0] == pytest.approx(-0.12035708045983422, rel=1e-9)
    assert a["DelayMargin"][0] == pytest.approx(-0.12042241262823783, rel=1e-3)
    a = allmargin(c2d(tf([5], [1, 1]) * tf([1], [0.01, 0.001, 1]), 0.01))
    assert a["DelayMargin"][0] == pytest.approx(40.01476178214942, rel=1e-3)
    a = allmargin(tf([0.1, 0.05], [1, -1.98, 0.9801], 0.01))
    assert a["DelayMargin"][0] == pytest.approx(15.981720146246426, rel=1e-3)


def test_no_spurious_crossings_near_tustin_zeros():
    """Tustin'in z = −1 sıfırları civarında |G| ~ 1e-17: faz gürültüsü geçiş sayılmaz (MATLAB yalnız gerçek geçiş)."""
    from spt.control.discretize import c2d
    G = c2d(tf([5], [1, 1]) * tf([1], [0.01, 0.001, 1]), 0.0005, "tustin")
    a = allmargin(G)
    assert a["GainMargin"].size == 1
    # MATLAB'in frekansı yaklaşık (∠G artığı 1.3e-4); bizimki kesin (artık ~1e-9, keskin rezonans):
    assert a["GainMargin"][0] == pytest.approx(0.02022029934691279, rel=1e-4)
    assert a["GMFrequency"][0] == pytest.approx(10.0049843687426, rel=1e-5)
    h = freqresp(G, a["GMFrequency"])[0, 0, 0]       # Ts = 0.5 ms'de kümelenmiş kutuplu tf: polyval ~1e-10 doğruluk
    assert abs(h.imag) <= 1e-8 * abs(h) and h.real < 0
    # Tustin sonrası z = 1 + 5e-10'daki integratör tanınır: ω = 0'da sahte kazanç payı yok
    a = allmargin(c2d(tf([10], [1, 2, 10, 0]), 0.0005, "tustin"))
    assert a["GainMargin"].size == 1 and a["GainMargin"][0] == pytest.approx(2.0, rel=1e-5)
    assert math.isinf(dcgain(c2d(tf([10], [1, 2, 10, 0]), 0.0005, "tustin")))


def test_degenerate_static_and_allpass_loops():
    a = allmargin(tf(-1.0))                                           # MATLAB: GM 1 @ 1 rad/s, PM NaN, DM 0 @ NaN
    assert list(a["GainMargin"]) == [1.0] and list(a["GMFrequency"]) == [1.0]
    assert math.isnan(a["PhaseMargin"][0]) and math.isnan(a["DMFrequency"][0]) and not a["Stable"]
    Gm, Pm, Wcg, Wcp = margin(tf(-1.0))
    assert Gm == 1.0 and Wcg == 1.0 and math.isnan(Pm) and math.isnan(Wcp)
    a = allmargin(tf([1, -1], [1, 1]))                                # MATLAB: DM yalnız 0 @ 0
    assert list(a["DelayMargin"]) == [0.0] and list(a["DMFrequency"]) == [0.0]
    a = allmargin(tf([1], [1], 0.1))                                  # MATLAB: DM [∞, 1] @ [0, ∞]
    assert list(a["DelayMargin"]) == [math.inf, 1.0] and list(a["DMFrequency"]) == [0.0, math.inf]
