"""Ayrıklaştırma: c2d (zoh / foh / impulse / tustin + prewarp / matched; kesirli gecikmeler), d2c, d2d, pade,
absorb_delay.

Referans: MATLAB R2025b (tests/data/matlab/lti_discretize.json). Karşılaştırma: ayrık modelin frekans cevabı
(gecikmeler dahil; gösterimden bağımsız) göreli 1e-9, SISO tf katsayıları ve toplam gecikme MATLAB'in gösterimiyle
aynı olduğu yerde 1e-9. Ayrıca kuramsal kontroller (zoh = expm, Tustin eşlemesi, c2d∘d2c = özdeşlik).
"""
from __future__ import annotations

import math
import warnings

import numpy as np
import pytest
import scipy.linalg as sla

from spt.control.errors import ControlError, ControlWarning
from spt.control.lti import StateSpace, TransferFunction, ZPK, ss, tf, zpk
from spt.control import convert
from spt.control.discretize import absorb_delay, c2d, d2c, d2d, pade
from spt.control.freqresp import freqresp
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
        mx = f"zpk({_mv(z) if len(z) else '[]'},{_mv(p) if len(p) else '[]'},{_m(k)},{_m(Ts)}{_nv(d)})"
        return zpk(z, p, k, Ts, **_kw(d)), mx
    _, A, B, C, D, Ts, d = spec
    return ss(A, B, C, D, Ts, **_kw(d)), f"ss({_mm(A)},{_mm(B)},{_mm(C)},{_mm(D)},{_m(Ts)}{_nv(d)})"


def TF(num, den, Ts=0.0, **d):
    return ("tf", num, den, Ts, d)


def ZK(z, p, k, Ts=0.0, **d):
    return ("zpk", z, p, k, Ts, d)


def SS(A, B, C, D, Ts=0.0, **d):
    return ("ss", A, B, C, D, Ts, d)


W_D = np.array([0.05, 0.31, 0.9, 1.7, 2.6, 3.1])        # × 1/Ts
W_C = np.array([0.07, 0.4, 1.3, 3.7, 9.1])

A3 = [[0, 1, 0], [0, 0, 1], [-6, -11, -6]]
CONT = {
    "g1": TF([1, 2], [1, 3, 5, 7]),
    "g2": ZK([-0.5 + 2j, -0.5 - 2j], [-1, -0.3 + 3j, -0.3 - 3j, -4], 6.0),
    "g8": ZK([-1.5, 2 + 1j, 2 - 1j], [-0.5 + 3j, -0.5 - 3j, -1 + 1j, -1 - 1j, -2, -3, -4 + 0.5j, -4 - 0.5j], 7.0),
    "gi": TF([2, 1], [1, 3, 0]),
    "gu": TF([1, 1], [1, -0.5, 2]),
    "gb": TF([2, 3, 1], [1, 5, 6]),
    "gm": SS(A3, [[0, 1], [1, 0], [0, 2]], [[1, 0, 0], [0, 1, 1]], [[0, 0.5], [0, 0]]),
    "dt1": TF([1], [1, 1], io=0.25),
    "dt2": TF([1, 3], [1, 2, 5], io=0.537),
    "ds1": SS(A3, [[0, 1], [1, 0], [0, 2]], [[1, 0, 0], [0, 1, 1]], [[0.3, 0.5], [0, 0]], ind=[0.25, 0.13]),
    "ds2": SS(A3, [[0], [1], [0]], [[1, 0, 0], [0, 1, 1]], [[0.2], [0]], outd=[0.25, 0.04]),
    "ds3": SS(A3, [[0, 1], [1, 0], [0, 2]], [[1, 0, 0], [0, 1, 1]], [[0.3, 0.5], [0, 0.1]], ind=[0.15, 0.0],
              outd=[0.07, 0.2]),
    "dmt": TF([[[1], [1, 0.5]], [[2], [1]]], [[[1, 1], [1, 2, 2]], [[1, 3], [1, 0.5]]], io=[[0.13, 0.0], [0.25, 0.4]]),
}
CMETHODS = {
    "g1": ["zoh", "foh", "impulse", "tustin", "tustinw", "matched"],
    "g2": ["zoh", "foh", "impulse", "tustin", "matched"],
    "g8": ["zoh", "foh", "tustin", "matched"],
    "gi": ["zoh", "foh", "impulse", "tustin", "matched"],
    "gu": ["zoh", "foh", "tustin", "matched"],
    "gb": ["zoh", "foh", "impulse", "tustin", "matched"],
    "gm": ["zoh", "foh", "impulse", "tustin"],
    "dt1": ["zoh", "foh", "impulse", "tustin", "matched"],
    "dt2": ["zoh", "foh", "impulse"],
    "ds1": ["zoh", "foh", "impulse"],
    "ds2": ["zoh", "foh"],
    "ds3": ["zoh", "foh"],
    "dmt": ["zoh", "foh"],
}
TS = 0.1
PREWARP = 7.0

DISC = {
    "z1": TF([0.2, 0.1], [1, -1.2, 0.5], 0.05),
    "z2": ZK([-0.5], [0.9, 0.3 + 0.4j, 0.3 - 0.4j], 2.0, 0.1),
    "zi": TF([0.1], [1, -1], 0.1),
    "zn": ZK([], [-0.5], 1.0, 0.1),
    "zm": ZK([-1], [0.5, 0.8], 1.0, 0.1),
    "zs": SS([[0.5, 0.1], [-0.2, 0.7]], [[1, 0], [0, 1]], [[1, 0], [0.5, 1]], [[0, 0.2], [0, 0]], 0.2),
    "zd": TF([0.3], [1, -0.7], 0.1, io=3),
}
DMETHODS = {"z1": ["zoh", "tustin", "matched"], "z2": ["zoh", "tustin", "matched"], "zi": ["zoh", "tustin"],
            "zn": ["zoh", "tustin"], "zm": ["zoh", "tustin", "matched"], "zs": ["zoh", "tustin"], "zd": ["zoh"]}
D2D = {"z1": [0.025, 0.15], "z2": [0.05, 0.3], "zs": [0.1, 0.5]}
PADE = [(0.1, 3), (2.0, 5), (0.5, 1), (1.3, 8), (0.02, 4)]
PADE_SYS = {"p1": TF([1, 3], [1, 2, 5], io=0.4), "p2": SS(A3, [[0, 1], [1, 0], [0, 2]], [[1, 0, 0], [0, 1, 1]],
                                                           [[0, 0.5], [0, 0]], ind=[0.2, 0.0], outd=[0.0, 0.3]),
            "p3": TF([1], [1, 1], ind=0.3, outd=0.2, io=0.1)}
ABSORB = {"b1": TF([0.3], [1, -0.7], 0.1, io=3, ind=1),
          "b2": SS([[0.5, 0.1], [-0.2, 0.7]], [[1, 0], [0, 1]], [[1, 0], [0.5, 1]], [[0, 0.2], [0, 0]], 0.2,
                   ind=[2, 0], outd=[1, 3]),
          "b3": ZK([-0.5], [0.9, 0.2], 2.0, 0.1, outd=2)}


def _mcall(method: str) -> str:
    if method == "tustinw":
        return f"c2d(G0, {_m(TS)}, c2dOptions('Method','tustin','PrewarpFrequency',{_m(PREWARP)}))"
    return f"c2d(G0, {_m(TS)}, '{method}')"


def _script() -> str:
    L = []
    rec = "out.{n}.re = real(freqresp(G, w)); out.{n}.im = imag(freqresp(G, w)); out.{n}.cls = class(G);"
    tfrec = ("if isequal(size(G), [1 1]), [n_, d_] = tfdata(tf(G), 'v'); out.{n}.tn = n_; out.{n}.td = d_; "
             "Gt_ = tf(G); out.{n}.tot = Gt_.IODelay + Gt_.InputDelay + Gt_.OutputDelay; end;")
    for name, spec in CONT.items():
        L.append(f"G0 = {model(spec)[1]}; w = {_mv(W_D / TS)};")
        for meth in CMETHODS[name]:
            n = f"{name}_{meth}"
            L.append(f"G = {_mcall(meth)}; " + rec.format(n=n) + " " + tfrec.format(n=n))
    for name, spec in DISC.items():
        L.append(f"G0 = {model(spec)[1]}; w = {_mv(W_C)};")
        for meth in DMETHODS[name]:
            n = f"{name}_{meth}"
            L.append(f"G = d2c(G0, '{meth}'); " + rec.format(n=n) + " " + tfrec.format(n=n) +
                     f" out.{n}.p = spt_c(pole(G)); out.{n}.z = spt_c(zero(G));")
        for k, ts2 in enumerate(D2D.get(name, [])):
            n = f"{name}_dd{k}"
            L.append(f"G = d2d(G0, {_m(ts2)}); w = {_mv(W_D / ts2)}; " + rec.format(n=n))
    for k, (T, N) in enumerate(PADE):
        L.append(f"[n_, d_] = pade({_m(T)}, {N}); out.pade{k}.n = n_; out.pade{k}.d = d_;")
    for name, spec in PADE_SYS.items():
        L.append(f"G = pade({model(spec)[1]}, 3); w = {_mv(W_C)}; " + rec.format(n=name))
    for name, spec in ABSORB.items():
        py = model(spec)[0]
        L.append(f"G = absorbDelay({model(spec)[1]}); w = {_mv(W_D / py.Ts)}; " + rec.format(n=name) +
                 f" out.{name}.order = order(G);")
    out = ["out.errs = {};"]
    for k, ln in enumerate(L):
        out.append(f"try; {ln}; catch e_; out.errs{{end+1}} = sprintf('satir {k}: %s', e_.message); end")
    return "\n".join(out)


SCRIPT = _script()


@pytest.fixture(scope="module")
def M():
    try:
        return fixture("lti_discretize", SCRIPT)
    except OracleUnavailable as e:
        pytest.skip(str(e))


# MATLAB'in de reddettiği durumlar: 'matched' d2c, tek katlı negatif gerçel sıfırda tanımsız (z1, z2 → sıfır −0.5)
MATLAB_REJECTS = {("z1", "matched"), ("z2", "matched")}


def test_matlab_script_ran_cleanly(M):
    errs = [e for e in M.get("errs", []) if "negative real zeros" not in e]
    assert not errs, errs
    assert len(M.get("errs", [])) == len(MATLAB_REJECTS)


def _H(rec):
    H = np.asarray(rec["re"], float) + 1j * np.asarray(rec["im"], float)
    return H.reshape(1, 1, -1) if H.ndim == 1 else H


def _close(a, b, rtol=1e-9, scale=None):
    a, b = np.asarray(a, complex), np.asarray(b, complex)
    assert a.shape == b.shape, (a.shape, b.shape)
    sc = (float(np.max(np.abs(b))) or 1.0) if scale is None else scale
    err = float(np.max(np.abs(a - b)))
    assert err <= rtol * sc, f"fark {err:.3e} > {rtol:.0e}·{sc:.3e}"


def _cplx(v):
    a = np.asarray(v, float)
    if a.size == 0:
        return np.zeros(0, complex)
    a = a.reshape(-1, 2)
    return a[:, 0] + 1j * a[:, 1]


def _c2d(G, meth):
    if meth == "tustinw":
        return c2d(G, TS, "tustin", prewarp=PREWARP)
    return c2d(G, TS, meth)


CPAIRS = [(n, m) for n, ms in CMETHODS.items() for m in ms]


@pytest.mark.parametrize("name,meth", CPAIRS)
def test_c2d(M, name, meth):
    G0 = model(CONT[name])[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ControlWarning)
        G = _c2d(G0, meth)
    ref = M[f"{name}_{meth}"]
    assert G.Ts == TS and type(G).__name__ == {"tf": "TransferFunction", "zpk": "ZPK", "ss": "StateSpace"}[ref["cls"]]
    _close(freqresp(G, W_D / TS), _H(ref), 1e-9)
    if "tn" in ref and G.is_siso:
        tot = float(G.total_delay()[0, 0])
        Gt = convert.to_tf(G)
        n, d = np.trim_zeros(Gt.num[0][0], "f"), Gt.den[0][0]
        rn = np.trim_zeros(np.atleast_1d(np.asarray(ref["tn"], float)), "f")
        rd = np.trim_zeros(np.atleast_1d(np.asarray(ref["td"], float)), "f")
        if n.size == rn.size and d.size == rd.size and tot == pytest.approx(float(ref["tot"])):
            _close(n / d[0], rn / rd[0], 1e-9, scale=max(1.0, float(np.max(np.abs(rn / rd[0])))))
            _close(d / d[0], rd / rd[0], 1e-9, scale=max(1.0, float(np.max(np.abs(rd / rd[0])))))
        else:
            # eşdeğer gösterim (z = 0'daki kutup / sıfırla gecikme alışverişi): toplam gecikme + derece farkı tutarlı
            assert (tot + d.size - n.size) == pytest.approx(float(ref["tot"]) + rd.size - rn.size)


def test_c2d_delay_rounding_warns():
    with pytest.warns(ControlWarning):
        G = c2d(tf([1], [1, 1], io_delay=0.25), 0.1, "tustin")
    assert G.io_delay[0, 0] == 3                       # MATLAB gibi yarım yukarı yuvarlama (2.5 → 3)


def test_c2d_zoh_equals_expm():
    A = np.array([[0.0, 1.0], [-2.0, -0.3]])
    B = np.array([[0.0], [1.0]])
    G = c2d(ss(A, B, [[1, 0]], 0), 0.2)
    Ad = sla.expm(A * 0.2)
    Bd = np.linalg.solve(A, (Ad - np.eye(2)) @ B)
    _close(G.A, Ad, 1e-14, 1.0)
    _close(G.B, Bd, 1e-13, 1.0)
    # birinci derece analitik: 1/(s+a) → (1−e^{−aT})/a / (z − e^{−aT})
    g = c2d(tf([1], [1, 2.0]), 0.3)
    n, d = g.tfdata(True)
    assert d[1] == pytest.approx(-math.exp(-0.6), rel=1e-15)
    assert n[0] == pytest.approx((1 - math.exp(-0.6)) / 2.0, rel=1e-14)


def test_tustin_mapping_is_bilinear():
    G = tf([1, 2], [1, 3, 5, 7])
    T = 0.05
    Gd = c2d(G, T, "tustin")
    for w in (0.3, 2.0, 15.0):
        z = np.exp(1j * w * T)
        s = 2 / T * (z - 1) / (z + 1)
        assert freqresp(Gd, [w])[0, 0, 0] == pytest.approx(np.polyval([1, 2], s) / np.polyval([1, 3, 5, 7], s),
                                                           rel=1e-12)
    w0 = 9.0
    Gw = c2d(G, T, "tustin", prewarp=w0)
    assert freqresp(Gw, [w0])[0, 0, 0] == pytest.approx(freqresp(G, [w0])[0, 0, 0], rel=1e-12)


def test_fractional_delay_zoh_is_exact_in_time():
    """Kesirli gecikmeli zoh ayrıklaştırma, gecikmeli sürekli sistemin örneklerdeki adım cevabıyla aynı."""
    from spt.control.timeresp import step
    G = tf([1, 3], [1, 2, 5], io_delay=0.537)
    Gd = c2d(G, 0.1)
    t = np.arange(60) * 0.1
    yc, _ = step(G, t)
    yd, _ = step(Gd, t)
    _close(yd, yc, 1e-12, 1.0)


DPAIRS = [(n, m) for n, ms in DMETHODS.items() for m in ms]


@pytest.mark.parametrize("name,meth", DPAIRS)
def test_d2c(M, name, meth):
    G0 = model(DISC[name])[0]
    if (name, meth) in MATLAB_REJECTS:
        with pytest.raises(ControlError):
            d2c(G0, meth)
        return
    G = d2c(G0, meth)
    ref = M[f"{name}_{meth}"]
    assert G.Ts == 0
    if name == "zn" and meth == "zoh":
        # Negatif gerçel kutup: MATLAB derece artırır; MATLAB'in kutbu yaklaşık (Im = 31.4159250, kesin π/Ts =
        # 31.4159265): burada kesin. Karşılaştırma 1e-6; kesinlik c2d∘d2c = özdeşlik ile.
        _close(freqresp(G, W_C), _H(ref), 1e-6)
        back = c2d(G, G0.Ts)
        _close(freqresp(back, W_D / G0.Ts), freqresp(G0, W_D / G0.Ts), 1e-12)
        return
    _close(freqresp(G, W_C), _H(ref), 1e-9)
    if meth in ("zoh", "tustin"):
        back = c2d(G, G0.Ts, meth)
        _close(freqresp(back, W_D / G0.Ts), freqresp(G0, W_D / G0.Ts), 1e-11)


def test_d2c_zoh_pole_at_origin_errors():
    with pytest.raises(ControlError):
        d2c(zpk([], [0, 0.5], 1, 0.1))


@pytest.mark.parametrize("name,k", [(n, k) for n, v in D2D.items() for k in range(len(v))])
def test_d2d(M, name, k):
    G0 = model(DISC[name])[0]
    ts2 = D2D[name][k]
    G = d2d(G0, ts2)
    assert G.Ts == ts2
    _close(freqresp(G, W_D / ts2), _H(M[f"{name}_dd{k}"]), 1e-9)


@pytest.mark.parametrize("k", range(len(PADE)))
def test_pade_coefficients(M, k):
    T, N = PADE[k]
    n, d = pade(T, N)
    rn = np.atleast_1d(np.asarray(M[f"pade{k}"]["n"], float))
    rd = np.atleast_1d(np.asarray(M[f"pade{k}"]["d"], float))
    _close(n, rn, 1e-13)
    _close(d, rd, 1e-13)
    # tüm geçiren: |num(jω)| = |den(jω)|
    for w in (0.1, 1.0, 10.0):
        assert abs(np.polyval(n, 1j * w)) == pytest.approx(abs(np.polyval(d, 1j * w)), rel=1e-12)


@pytest.mark.parametrize("name", list(PADE_SYS))
def test_pade_system(M, name):
    G = pade(model(PADE_SYS[name])[0], 3)
    assert not G.has_delay()
    _close(freqresp(G, W_C), _H(M[name]), 1e-9)


@pytest.mark.parametrize("name", list(ABSORB))
def test_absorb_delay(M, name):
    G0 = model(ABSORB[name])[0]
    G = absorb_delay(G0)
    assert not G.has_delay() and type(G) is type(G0)
    _close(freqresp(G, W_D / G0.Ts), _H(M[name]), 1e-12)
    _close(freqresp(G, W_D / G0.Ts), freqresp(G0, W_D / G0.Ts), 1e-12)
    if isinstance(G, StateSpace):
        assert G.nx == M[name]["order"]


def test_continuous_absorb_and_pade_errors():
    with pytest.raises(ControlError):
        absorb_delay(tf([1], [1, 1], io_delay=0.2))
    with pytest.raises(ControlError):
        pade(tf([1], [1, -0.5], 0.1, io_delay=2), 3)
    with pytest.raises(ControlError):
        c2d(tf([1], [1, 1], 0.1), 0.1)
    with pytest.raises(ControlError):
        d2c(tf([1], [1, 1]))
    with pytest.raises(ControlError):
        c2d(tf([1], [1, 1]), 0.1, "bilinear")
    assert isinstance(c2d(zpk([], [-1], 1), 0.1), ZPK) and isinstance(c2d(tf([1], [1, 1]), 0.1), TransferFunction)
