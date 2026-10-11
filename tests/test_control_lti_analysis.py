"""Analiz: pole, zero, tzero (MIMO, kare olmayan), damp, pzmap, rlocus, ctrb, obsv, gram, covar, isstable, dsort,
esort ve kök yer eğrisi yardımcıları (kırılma noktaları, asimptotlar).

Referans: MATLAB R2025b (tests/data/matlab/lti_analysis.json). Kutup / sıfır kümeleri ve Gramyen / kovaryans göreli
1e-8 (çoklu köklerde koşul sayısına göre gerekçeli gevşetme yok: test durumları basit köklü). rlocus kolları
sütun sütun küme olarak ve süreklilik olarak karşılaştırılır (kol sırası MATLAB'in kutup sırasına bağlıdır).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from spt.control.errors import ControlError
from spt.control.lti import ss, tf, zpk
from spt.control.analysis import (asymptotes, breakaway, covar, ctrb, damp, dsort, esort, gram, isstable, obsv,
                                  pole, pzmap, rlocus, tzero, zero)
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


def model(spec):
    kind = spec[0]
    if kind == "tf":
        _, num, den, Ts = spec
        grid = isinstance(num, list) and num and isinstance(num[0], list) and isinstance(num[0][0], list)
        return tf(num, den, Ts), f"tf({_mcell(num) if grid else _mv(num)},{_mcell(den) if grid else _mv(den)},{_m(Ts)})"
    if kind == "zpk":
        _, z, p, k, Ts = spec
        zs = _mv([complex(v) for v in z]) if len(z) else "[]"
        return zpk(z, p, k, Ts), f"zpk({zs},{_mv([complex(v) for v in p])},{_m(k)},{_m(Ts)})"
    _, A, B, C, D, Ts = spec
    return ss(A, B, C, D, Ts), f"ss({_mm(A)},{_mm(B)},{_mm(C)},{_mm(D)},{_m(Ts)})"


def TF(num, den, Ts=0.0):
    return ("tf", num, den, Ts)


def ZK(z, p, k, Ts=0.0):
    return ("zpk", z, p, k, Ts)


def SS(A, B, C, D, Ts=0.0):
    return ("ss", A, B, C, D, Ts)


A3 = [[0, 1, 0], [0, 0, 1], [-6, -11, -6]]
PZ = {
    "p1": TF([1, 2], [1, 3, 5, 7]),
    "p2": ZK([-0.5 + 2j, -0.5 - 2j], [-1, -0.3 + 3j, -0.3 - 3j, -4], 6.0),
    "p3": SS(A3, [[0, 1], [1, 0], [0, 2]], [[1, 0, 0], [0, 1, 1]], [[0, 0.5], [0, 0]]),
    "p4": SS(A3, [[0, 1], [1, 0], [0, 2]], [[1, 0, 0], [0, 1, 1], [1, 1, 1]], [[0, 0], [0, 0], [0, 1]]),
    "p5": SS(A3, [[0], [1], [0]], [[1, 0, 0], [0, 1, 1]], [[0], [0]]),
    "p6": TF([[[1], [1]], [[1], [1]]], [[[1, 1], [1, 2]], [[1, 3], [1, 4]]]),
    "p7": TF([[[1, 1]], [[1]]], [[[1, 2, 1]], [[1, 1]]]),
    "p8": SS([[0.5, 0.1, 0, 0], [0, 0.8, 0.2, 0], [0, 0, -0.3, 1], [0, 0, -0.2, 0.1]],
             [[1, 0], [0, 1], [0.5, 0], [0, 0.3]], [[1, 0, 1, 0], [0, 1, 0, 1]], [[0, 0.1], [0.2, 0]], 0.2),
    "p9": TF([1, -0.5, 0.06], [1, -1.2, 0.5, -0.1], 0.1),
    "p10": SS([[-1, 1, 0, 0], [0, -1, 0, 0], [0, 0, -2, 1], [0, 0, 0, -5]], [[0, 1], [1, 0], [0, 0], [1, 1]],
              [[1, 0, 1, 0], [0, 0, 0, 1]], [[0, 0], [0, 0]]),
}
DAMP = {"d1": TF([1], [1, 1, 0]), "d2": ZK([], [0, 0.5, -0.5, 0.3 + 0.4j, 0.3 - 0.4j], 1.0, 0.1),
        "d3": ZK([], [-1, -2 + 3j, -2 - 3j, 2, -0.1], 1.0), "d4": ZK([], [1, 1.2], 1.0, 0.1), "d5": PZ["p3"]}
RLOC = {"r1": (TF([1, 2], [1, 3, 4, 0]), [0, 0.5, 1, 2, 5, 10, 100]),
        "r2": (ZK([-1], [0, -2, -4 + 1j, -4 - 1j], 1.0), [0, 1, 3, 10, 30, 100]),
        "r3": (TF([0.5, -0.2], [1, -1.4, 0.45], 0.1), [0, 0.1, 0.5, 1, 2]),
        "r4": (TF([1, 3], [1, 5, 6, 0]), [0, 0.2, 0.4, 0.42, 1, 4])}
GRAM = {"q1": SS(A3, [[0, 1], [1, 0], [0, 2]], [[1, 0, 0], [0, 1, 1]], [[0, 0.5], [0, 0]]),
        "q2": PZ["p8"],
        "q3": SS([[-1, 2], [-3, -4]], [[1], [0.5]], [[1, -1]], [[0]])}
COVAR = {"v1": (GRAM["q3"], [[2.0]]), "v2": (PZ["p8"], [[1.0, 0.2], [0.2, 0.5]]),
         "v3": (SS([[-1, 2], [-3, -4]], [[1], [0.5]], [[1, -1]], [[0.3]]), [[1.0]])}


def _script() -> str:
    L = []
    for name, spec in PZ.items():
        L.append(f"G = {model(spec)[1]}; out.{name}.p = spt_c(pole(G)); out.{name}.tz = spt_c(tzero(G)); "
                 f"out.{name}.st = isstable(G); if isequal(size(G), [1 1]), out.{name}.z = spt_c(zero(G)); end")
    for name, spec in DAMP.items():
        L.append(f"[wn_, z_, p_] = damp({model(spec)[1]}); out.{name}.wn = wn_; out.{name}.z = z_; "
                 f"out.{name}.p = spt_c(p_);")
    for name, (spec, k) in RLOC.items():
        L.append(f"r_ = rlocus({model(spec)[1]}, {_mv(k)}); out.{name}.re = real(r_); out.{name}.im = imag(r_);")
    for name, spec in GRAM.items():
        L.append(f"G = {model(spec)[1]}; out.{name}.wc = gram(G, 'c'); out.{name}.wo = gram(G, 'o'); "
                 f"out.{name}.cm = ctrb(G); out.{name}.om = obsv(G);")
    for name, (spec, Wn) in COVAR.items():
        L.append(f"[P_, Q_] = covar({model(spec)[1]}, {_mm(Wn)}); out.{name}.P = P_; out.{name}.Q = Q_;")
    L.append("p_ = [0.5; -0.9; 0.3+0.95i; 0.3-0.95i; -0.2; 1.1]; [s_, i_] = dsort(p_); out.ds.s = spt_c(s_); "
             "out.ds.i = i_;")
    L.append("p_ = [-1; 2; -0.5+3i; -0.5-3i; 0; -7]; [s_, i_] = esort(p_); out.es.s = spt_c(s_); out.es.i = i_;")
    out = ["out.errs = {};"]
    for k, ln in enumerate(L):
        out.append(f"try; {ln}; catch e_; out.errs{{end+1}} = sprintf('satir {k}: %s', e_.message); end")
    return "\n".join(out)


SCRIPT = _script()


@pytest.fixture(scope="module")
def M():
    try:
        return fixture("lti_analysis", SCRIPT)
    except OracleUnavailable as e:
        pytest.skip(str(e))


def test_matlab_script_ran_cleanly(M):
    assert not M.get("errs"), M.get("errs")


def _cplx(v):
    a = np.asarray(v, float)
    if a.size == 0:
        return np.zeros(0, complex)
    a = a.reshape(-1, 2)
    return a[:, 0] + 1j * a[:, 1]


def _match(a, b, rtol=1e-8):
    a, b = np.asarray(a, complex).ravel(), np.asarray(b, complex).ravel()
    assert a.size == b.size, (a, b)
    used = np.zeros(b.size, bool)
    for x in a:
        d = np.abs(b - x)
        d[used] = np.inf
        k = int(np.argmin(d))
        assert d[k] <= rtol * max(1.0, abs(x)), (x, b)
        used[k] = True


@pytest.mark.parametrize("name", list(PZ))
def test_poles_zeros(M, name):
    G = model(PZ[name])[0]
    ref = M[name]
    _match(pole(G), _cplx(ref["p"]))
    _match(tzero(G), _cplx(ref["tz"]), 1e-7)
    assert isstable(G) == bool(ref["st"])
    if "z" in ref:
        _match(zero(G), _cplx(ref["z"]), 1e-7)
    p, z = pzmap(G)
    _match(p, pole(G), 1e-14)


@pytest.mark.parametrize("name", list(DAMP))
def test_damp(M, name):
    wn, z, p = damp(model(DAMP[name])[0])
    ref = M[name]
    rwn = np.atleast_1d(np.asarray(ref["wn"], float))
    assert np.array_equal(np.isinf(wn), np.isinf(rwn))
    fin = np.isfinite(rwn)
    assert np.allclose(wn[fin], rwn[fin], rtol=1e-10)
    assert np.allclose(z, np.atleast_1d(np.asarray(ref["z"], float)), rtol=1e-10, atol=1e-14)
    rp = _cplx(ref["p"])
    # aynı ω_n'li çiftler içinde sıra serbest; ω_n sırası ve küme eşleşmeli
    _match(p, rp, 1e-10)


@pytest.mark.parametrize("name", list(RLOC))
def test_rlocus(M, name):
    spec, k = RLOC[name]
    r, kk = rlocus(model(spec)[0], k)
    ref = np.asarray(M[name]["re"], float) + 1j * np.asarray(M[name]["im"], float)
    ref = ref.reshape(r.shape)
    for j in range(r.shape[1]):
        _match(r[:, j], ref[:, j], 1e-8)
    # süreklilik: her kol bir sonraki sütunda en yakın köke gider (Macar eşleme)
    for j in range(1, r.shape[1]):
        step = np.abs(r[:, j] - r[:, j - 1])
        alt = np.abs(r[:, j][::-1] - r[:, j - 1])
        assert np.sum(step) <= np.sum(alt) + 1e-12


def test_rlocus_auto_and_helpers():
    G = tf([1], [1, 3, 2, 0])
    r, k = rlocus(G)
    assert r.shape[0] == 3 and k[0] == 0 and np.all(np.diff(k) > 0)
    pts, gains = breakaway(G)
    # 1/(s(s+1)(s+2)): kırılma s = −1 + 1/√3, K = 2/(3√3)
    assert pts.size == 1 and pts[0].real == pytest.approx(-1 + 1 / math.sqrt(3), rel=1e-12)
    assert gains[0] == pytest.approx(2 / (3 * math.sqrt(3)), rel=1e-12)
    sig, ang = asymptotes(G)
    assert sig == pytest.approx(-1.0) and np.allclose(ang, [math.pi / 3, math.pi, 5 * math.pi / 3])
    with pytest.raises(ControlError):
        rlocus(tf([1], [1, 1], io_delay=0.3))


@pytest.mark.parametrize("name", list(GRAM))
def test_gram_ctrb_obsv(M, name):
    G = model(GRAM[name])[0]
    ref = M[name]
    for key, val in (("wc", gram(G, "c")), ("wo", gram(G, "o")), ("cm", ctrb(G)), ("om", obsv(G))):
        r = np.asarray(ref[key], float).reshape(val.shape)
        assert np.max(np.abs(val - r)) <= 1e-9 * max(1.0, float(np.max(np.abs(r)))), key
    with pytest.raises(ControlError):
        gram(tf([1], [1, -1]), "c")


@pytest.mark.parametrize("name", list(COVAR))
def test_covar(M, name):
    spec, Wn = COVAR[name]
    P, Q = covar(model(spec)[0], Wn)
    ref = M[name]
    rP = np.asarray(ref["P"], float).reshape(P.shape)
    if np.all(np.isinf(rP)):
        assert np.all(np.isinf(P))
    else:
        assert np.allclose(P, rP, rtol=1e-10)
    rQ = np.asarray(ref["Q"], float).reshape(Q.shape)
    assert np.allclose(Q, rQ, rtol=1e-10, atol=1e-14)


def test_dsort_esort(M):
    p = np.array([0.5, -0.9, 0.3 + 0.95j, 0.3 - 0.95j, -0.2, 1.1])
    s, i = dsort(p)
    _match(s, _cplx(M["ds"]["s"]), 1e-14)
    assert np.allclose(np.abs(s), np.abs(_cplx(M["ds"]["s"])))
    p = np.array([-1, 2, -0.5 + 3j, -0.5 - 3j, 0, -7])
    s, i = esort(p)
    assert np.allclose(s.real, _cplx(M["es"]["s"]).real)


def test_isstable_boundaries():
    assert not isstable(tf([1], [1, 0]))
    assert not isstable(tf([1], [1, 0, 1]))
    assert isstable(tf([1], [1, 2, 1]))
    assert not isstable(tf([1], [1, -1], 0.1))
    assert isstable(tf([1], [1, 0.99], 0.1))
