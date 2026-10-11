"""LTI çekirdeği: model nesneleri, tf ↔ zpk ↔ ss, aritmetik (gecikmeli, MIMO), bağlantılar, minreal, dengeli
indirgeme, kanonik biçimler, JSON, gösterim.

Referans: MATLAB R2025b (tests/data/matlab/lti_core.json, tools/matlab_oracle.py). Gerçeklemeler MATLAB'inkiyle aynı
matris olmak zorunda değildir: karşılaştırma değişmezlerle (transfer katsayıları, Markov parametreleri, özdeğerler,
frekans cevabı) yapılır. Kapalı biçimli hesaplarda tolerans göreli 1e-8 (tf katsayıları / frekans cevabı 1e-9).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from spt.control.errors import ControlError
from spt.control.lti import LTI, StateSpace, TransferFunction, ZPK, inv, ss, tf, zpk
from spt.control import convert
from spt.control.connect import append, connect, feedback, lft, loopsens, parallel, series, sumblk
from spt.control.freqresp import freqresp
from tools.matlab_oracle import OracleUnavailable, fixture

W_C = [0.13, 0.7, 1.9, 4.3, 11.0]
W_D = [0.13, 0.7, 1.9, 3.1]          # ×(1/Ts) ile ölçeklenir (ω < π/Ts)


# ---------------------------------------------------------------------------------------------------------------
# MATLAB / Python model tanımları (tek kaynaktan)
# ---------------------------------------------------------------------------------------------------------------
def _m(x) -> str:
    if isinstance(x, complex):
        return f"complex({float(x.real)!r},{float(x.imag)!r})"
    x = float(x)
    if math.isinf(x):
        return "Inf" if x > 0 else "-Inf"
    return repr(x)


def _mv(v) -> str:
    return "[" + " ".join(_m(x) for x in np.atleast_1d(v)) + "]"


def _mm(M) -> str:
    M = np.atleast_2d(np.asarray(M, float))
    if M.size == 0:
        return f"zeros({M.shape[0]},{M.shape[1]})"
    return "[" + ";".join(" ".join(_m(x) for x in row) for row in M) + "]"


def _mcell(grid) -> str:
    return "{" + ";".join(",".join(_mv(p) for p in row) for row in grid) + "}"


def _is_grid(x) -> bool:
    return isinstance(x, list) and x and isinstance(x[0], list) and x[0] and isinstance(x[0][0], (list, tuple))


def _nv(d) -> str:
    s = ""
    for key, mk in (("ind", "InputDelay"), ("outd", "OutputDelay"), ("io", "IODelay")):
        if key in d:
            v = d[key]
            s += f",'{mk}'," + (_mm(v) if np.ndim(v) == 2 else _mv(v))
    return s


def _py_kw(d) -> dict:
    kw = {}
    for key, pk in (("ind", "input_delay"), ("outd", "output_delay"), ("io", "io_delay")):
        if key in d:
            kw[pk] = d[key]
    return kw


def model(spec):
    """spec → (Python modeli, MATLAB ifadesi)."""
    kind = spec[0]
    if kind == "tf":
        _, num, den, Ts, d = spec
        if _is_grid(num):
            mexpr = f"tf({_mcell(num)},{_mcell(den)},{_m(Ts)}{_nv(d)})"
        else:
            mexpr = f"tf({_mv(num)},{_mv(den)},{_m(Ts)}{_nv(d)})"
        return tf(num, den, Ts, **_py_kw(d)), mexpr
    if kind == "zpk":
        _, z, p, k, Ts, d = spec
        if np.ndim(k) == 2:
            zc = "{" + ";".join(",".join(_mv(v) if len(v) else "[]" for v in row) for row in z) + "}"
            pc = "{" + ";".join(",".join(_mv(v) if len(v) else "[]" for v in row) for row in p) + "}"
            mexpr = f"zpk({zc},{pc},{_mm(k)},{_m(Ts)}{_nv(d)})"
        else:
            mexpr = f"zpk({_mv(z) if len(z) else '[]'},{_mv(p) if len(p) else '[]'},{_m(k)},{_m(Ts)}{_nv(d)})"
        return zpk(z, p, k, Ts, **_py_kw(d)), mexpr
    _, A, B, C, D, Ts, d = spec
    return ss(A, B, C, D, Ts, **_py_kw(d)), f"ss({_mm(A)},{_mm(B)},{_mm(C)},{_mm(D)},{_m(Ts)}{_nv(d)})"


def TF(num, den, Ts=0.0, **d):
    return ("tf", num, den, Ts, d)


def ZK(z, p, k, Ts=0.0, **d):
    return ("zpk", z, p, k, Ts, d)


def SS(A, B, C, D, Ts=0.0, **d):
    return ("ss", A, B, C, D, Ts, d)


def _wgrid(sys) -> list[float]:
    return [w / sys.Ts for w in W_D] if sys.is_discrete else W_C


# ---------------------------------------------------------------------------------------------------------------
# Durumlar
# ---------------------------------------------------------------------------------------------------------------
P8 = [-0.5 + 3j, -0.5 - 3j, -1 + 1j, -1 - 1j, -2, -3, -4 + 0.5j, -4 - 0.5j]
CONV = {
    "c1": TF([1, 2], [1, 3, 5, 7]),
    "c2": TF([2, 0, -1], [1, 0.5, 3, 1.2, 0.4]),
    "c3": TF([3, 1, 2], [1, 4, 6, 4]),
    "c4": ZK([-1.5, 2 + 1j, 2 - 1j], P8, 7.0),
    "c5": TF([0.2, -0.1, 0.05], [1, -1.3, 0.8, -0.2], 0.05),
    "c6": TF([[[1], [1, 2]], [[2, 0], [1]]], [[[1, 1], [1, 3, 2]], [[1, 2, 5], [1, 0]]]),
    "c7": TF([[[0.5], [1, -0.3], [0.2]], [[1, 0], [0.1], [1, 0.5]]],
             [[[1, -0.5], [1, -0.9, 0.2], [1, -0.1]], [[1, -0.7], [1, 0.4], [1, -0.5]]], 0.1),
    "c8": SS([[0, 1, 0], [0, 0, 1], [-6, -11, -6]], [[0, 1], [1, 0], [0, 2]], [[1, 0, 0], [0, 1, 1]],
             [[0, 0.5], [0, 0]]),
    "c9": ZK([[[], [-3]]], [[[-1], [-1]]], [[1.0, 2.0]]),
    "c10": TF([1, 1.0000001], [1, 3, 2]),
    "c11": TF([2, 3, 1], [1, 5, 6]),
    "c12": TF([1, -1], [1, -2, 5]),
    "c13": SS([[0.5, 0.1, 0, 0], [0, 0.8, 0.2, 0], [0, 0, -0.3, 1], [0, 0, -0.2, 0.1]],
              [[1, 0], [0, 1], [0.5, 0], [0, 0.3]], [[1, 0, 1, 0]], [[0, 0.1]], 0.2),
    "c14": ZK([-0.05 + 2j, -0.05 - 2j], [-1, -0.2 + 2.2j, -0.2 - 2.2j], -3.0),
    "c15": TF([[[1], [2]], [[3], [4]]], [[[1, 1], [1, 1]], [[1, 1], [1, 1]]]),
    "c16": TF([[[1, 1]], [[1]]], [[[1, 2, 1]], [[1, 1]]]),
}

# Aritmetik ifadeleri: ad → (python lambda, MATLAB ifadesi); temel modeller BASE'de
BASE = {
    "G1": TF([1, 2], [1, 3, 5, 7]),
    "G2": TF([1], [1, 1]),
    "G3": TF([2, 3, 1], [1, 5, 6]),
    "Z1": ZK([-2], [-1, -3 + 4j, -3 - 4j], 5.0),
    "Z2": ZK([], [-1, -2], 1.0),
    "S1": SS([[-1, 2], [-3, -4]], [[1], [0.5]], [[1, -1]], [[0.2]]),
    "T2": TF([[[1], [1, 2]], [[2, 0], [1]]], [[[1, 1], [1, 3, 2]], [[1, 2, 5], [1, 4]]]),
    "M2": SS([[-1, 0.5, 0], [0, -2, 1], [0.3, 0, -3]], [[1, 0], [0, 1], [1, 1]], [[1, 0, 1], [0, 1, 0]],
             [[0, 0.1], [0, 0]]),
    "D1": TF([0.5, 0.1], [1, -0.6, 0.08], 0.1),
    "D2": TF([1], [1, -0.5], 0.1, io=2),
    "D3": TF([0.3], [1, -0.2], 0.1, ind=1),
    "L1": TF([1], [1, 1], ind=0.1),
    "L2": TF([1], [1, 2], outd=0.2),
    "L3": TF([1], [1, 1], io=0.1),
    "L4": TF([1], [1, 2], io=0.2),
    "SL": SS([[-1, 0], [0, -2]], [[1, 0], [0, 1]], [[1, 1], [0, 1]], [[0, 0], [0, 0]], ind=[0.3, 0.3]),
}

ARITH = {
    "a_add": ("G1 + G2", "G1 + G2"),
    "a_mul": ("G1 * G2", "G1 * G2"),
    "a_div": ("G1 / G2", "G1 / G2"),
    "a_sub0": ("G1 - G1", "G1 - G1"),
    "a_lin": ("3 * G1 + 2", "3 * G1 + 2"),
    "a_pow3": ("G2 ** 3", "G2^3"),
    "a_powm2": ("G2 ** -2", "G2^-2"),
    "a_inv": ("inv(G3)", "inv(G3)"),
    "a_zadd": ("Z1 + Z2", "Z1 + Z2"),
    "a_zmul": ("Z1 * G2", "Z1 * G2"),
    "a_zdiv": ("Z1 / Z2", "Z1 / Z2"),
    "a_ss_tf": ("S1 + G1", "S1 + G1"),
    "a_ss_mul": ("S1 * G1 * S1", "S1 * G1 * S1"),
    "a_mimo_mul": ("T2 * T2", "T2 * T2"),
    "a_mimo_add": ("T2 + M2", "T2 + M2"),
    "a_mimo_ss": ("M2 * T2 - 2 * M2", "M2 * T2 - 2 * M2"),
    "a_scal": ("G2 * T2", "G2 * T2"),
    "a_scal2": ("T2 + G2", "T2 + G2"),
    "a_sub": ("T2[1, 0]", "T2(2, 1)"),
    "a_neg": ("-Z1", "-Z1"),
    "a_mat": ("np.array([[1.0, 2.0], [0.0, 1.0]]) * T2", "[1 2; 0 1] * T2"),
    "a_fb": ("feedback(G1, G2)", "feedback(G1, G2)"),
    "a_fbpos": ("feedback(G1, G3, sign=+1)", "feedback(G1, G3, +1)"),
    "a_fbz": ("feedback(Z1, 2.0)", "feedback(Z1, 2)"),
    "a_fbss": ("feedback(M2, S1, feedin=[1], feedout=[0])", "feedback(M2, S1, 2, 1)"),
    "a_fbmimo": ("feedback(T2, np.eye(2))", "feedback(T2, eye(2))"),
    "a_series": ("series(M2, T2, [1, 0], [0, 1])", "series(M2, T2, [2 1], [1 2])"),
    "a_par": ("parallel(M2, T2, [0], [1], [1], [0])", "parallel(M2, T2, 1, 2, 2, 1)"),
    "a_app": ("append(G1, Z1, S1)", "append(G1, Z1, S1)"),
    "a_d1": ("D1 * D2 + D3", "D1 * D2 + D3"),
    "a_d2": ("D2 * D3", "D2 * D3"),
    "a_d3": ("D2 + D3", "D2 + D3"),
    "a_dfb": ("feedback(D1 * D2, 1)", "feedback(D1 * D2, 1)"),
    "a_l1": ("L1 * L2", "L1 * L2"),
    "a_l2": ("L3 * L4", "L3 * L4"),
    "a_l3": ("SL * M2", "SL * M2"),
    "a_l4": ("M2 * SL", "M2 * SL"),
    "a_lft": ("lft(M2, S1)", "lft(M2, S1)"),
    "a_lft2": ("lft(S1, M2, 1, 1)", "lft(S1, M2, 1, 1)"),
}

MINREAL = {
    "m_tf7": TF([1, 1.0000001], [1, 3, 2]),
    "m_tf8": TF([1, 1.00000001], [1, 3, 2]),
    "m_tf100": TF([1, 100.000001], np.convolve([1, 100], [1, 3]).tolist()),
    "m_z": ZK([-1, -2], [-1, -3], 5.0),
    "m_ss": SS([[-1, 0, 0], [0, -2, 0], [0, 0, -3]], [[1], [0], [1]], [[1, 1, 0]], [[0]]),
    "m_ss2": SS([[-1, 1, 0, 0], [0, -1, 0, 0], [0, 0, -2, 1], [0, 0, 0, -5]], [[0, 1], [1, 0], [0, 0], [1, 1]],
                [[1, 0, 1, 0], [0, 0, 0, 1]], [[0, 0], [0, 0]]),
}

RED = {
    "r4": TF([1], np.convolve([1, 1], np.convolve([1, 2], [1, 3, 5])).tolist()),
    "r6": ZK([-0.7, -4], [-0.5, -1 + 2j, -1 - 2j, -3, -6, -10], 20.0),
    "ru": ZK([], [1, -1, -2, -10], 1.0),
    "rd": TF([0.1, 0.05, 0.02], [1, -1.6, 0.89, -0.21, 0.018], 0.1),
    "rm": SS([[-1, 0.5, 0, 0], [0, -2, 0.3, 0], [0, 0, -4, 1], [0.1, 0, 0, -8]], [[1, 0], [0, 1], [1, 0], [0, 1]],
             [[1, 0, 0.5, 0], [0, 1, 0, 0.2]], [[0, 0], [0.1, 0]]),
}

CANON = {
    "k1": TF([1, 2], [1, 3, 5, 7]),
    "k2": ZK([], [-1, -2 + 3j, -2 - 3j, -5], 4.0),
    "k3": ZK([], [-1, -0.5 + 5j, -0.5 - 5j, 2, -3], 4.0),
    "k4": ZK([-1], [-0.1 + 1j, -0.1 - 1j, -0.2 + 0.5j, -0.2 - 0.5j], 4.0),
    "k5": TF([1, 0.5], [1, -1.2, 0.5], 0.1),
}


def _script() -> str:
    L = ["w_c = " + _mv(W_C) + ";"]
    rec = "out.{n}.re = real(freqresp(G, w)); out.{n}.im = imag(freqresp(G, w));"
    for name, spec in CONV.items():
        py, mx = model(spec)
        w = _wgrid(py)
        L.append(f"G = {mx}; w = {_mv(w)};")
        L.append(rec.format(n=name))
        ny, nu = py.shape
        for i in range(ny):
            for j in range(nu):
                L.append(f"[n_, d_] = tfdata(tf(G({i + 1},{j + 1})), 'v'); out.{name}.tf{i}{j}.n = n_; "
                         f"out.{name}.tf{i}{j}.d = d_;")
                L.append(f"[z_, p_, k_] = zpkdata(zpk(G({i + 1},{j + 1})), 'v'); out.{name}.zpk{i}{j}.z = spt_c(z_);"
                         f" out.{name}.zpk{i}{j}.p = spt_c(p_); out.{name}.zpk{i}{j}.k = k_;")
        L.append(f"S = ss(G); out.{name}.n = order(S); mk = zeros(size(S.D,1), size(S.D,2), 7); "
                 f"mk(:,:,1) = S.D; for q = 1:6, mk(:,:,q+1) = S.C*S.A^(q-1)*S.B; end; out.{name}.mk = mk; "
                 f"out.{name}.eig = spt_c(eig(S.A)); out.{name}.pole = spt_c(pole(G)); "
                 f"out.{name}.tzero = spt_c(tzero(G));")
    for name, spec in BASE.items():
        L.append(f"{name} = {model(spec)[1]};")
    for name, (_, mx) in ARITH.items():
        L.append(f"G = {mx}; w = w_c; if G.Ts > 0, w = {_mv(W_D)} / G.Ts; end; " + rec.format(n=name) +
                 f" out.{name}.cls = class(G); out.{name}.n = order(G); out.{name}.size = size(G);")
        L.append(f"if isa(G, 'tf') && isequal(size(G), [1 1]), [n_, d_] = tfdata(G, 'v'); out.{name}.tn = n_; "
                 f"out.{name}.td = d_; out.{name}.io = G.IODelay; out.{name}.ind = G.InputDelay; "
                 f"out.{name}.outd = G.OutputDelay; end")
        L.append(f"if isa(G, 'zpk') && isequal(size(G), [1 1]), [z_, p_, k_] = zpkdata(G, 'v'); "
                 f"out.{name}.zz = spt_c(z_); out.{name}.zp = spt_c(p_); out.{name}.zk = k_; end")
    for name, spec in MINREAL.items():
        L.append(f"G = minreal({model(spec)[1]}); w = w_c; " + rec.format(n=name) + f" out.{name}.n = order(G);")
        L.append(f"if isa(G, 'tf'), [n_, d_] = tfdata(G, 'v'); out.{name}.tn = n_; out.{name}.td = d_; end")
    for name, spec in RED.items():
        py, mx = model(spec)
        w = _wgrid(py)
        L.append(f"G0 = {mx}; w = {_mv(w)}; out.{name}.hsv = hsvd(G0); [gb_, g_] = balreal(ss(G0)); "
                 f"out.{name}.g = g_; out.{name}.nb = order(gb_);")
        L.append("G = gb_; " + rec.format(n=name + "_bal"))
        L.append("G = balred(G0, 2); " + rec.format(n=name + "_dc") + f" out.{name}_dc.p = spt_c(pole(G));")
        L.append("G = balred(G0, 2, balredOptions('StateElimMethod','Truncate')); " + rec.format(n=name + "_tr"))
        L.append("G = modred(gb_, [3 4]); " + rec.format(n=name + "_mr"))
        L.append("G = modred(gb_, [3 4], 'Truncate'); " + rec.format(n=name + "_mt"))
    L.append(f"G = modred({model(RED['rm'])[1]}, [1 3]); w = w_c; " + rec.format(n="rm_raw"))
    for name, spec in CANON.items():
        L.append(f"G0 = {model(spec)[1]}; [c_, T_] = canon(ss(G0), 'modal'); out.{name}.modal = c_.A;")
        L.append(f"[c_, T_] = canon(ss(G0), 'companion'); out.{name}.ca = c_.A; out.{name}.cb = c_.B; "
                 f"out.{name}.cc = c_.C; out.{name}.cd = c_.D;")
    # connect / loopsens / sumblk
    L += [
        "C = tf(2,1); C.InputName = 'e'; C.OutputName = 'u'; P = tf(1,[1 1]); P.InputName = 'u'; P.OutputName = 'y';",
        "S = sumblk('e = r - y'); w = w_c;",
        "G = connect(P,C,S,'r','y'); " + rec.format(n="n_cl") + " out.n_cl.cls = class(G);",
        "G = connect(P,C,S,{'r','u'},{'y','u','e'}); " + rec.format(n="n_ap"),
        "G = connect(P,C,S,'u','y'); " + rec.format(n="n_u"),
        "P2 = ss([-1 0.5;0 -2],[1 0;0 1],[1 0;0.5 1],0); P2.InputName = 'u'; P2.OutputName = 'y';",
        "K2 = ss(-0.5,[1 0.2],[1;0.3],[2 0;0 1.5]); K2.InputName = 'e'; K2.OutputName = 'u';",
        "S2 = sumblk('e = r - y', 2); G = connect(P2,K2,S2,'r','y'); " + rec.format(n="n_mimo"),
        "L = loopsens(P2, ss(-0.5,[1 0.2],[1;0.3],[2 0;0 1.5]));",
        "G = L.Si; " + rec.format(n="ls_Si") + " G = L.So; " + rec.format(n="ls_So"),
        "G = L.Ti; " + rec.format(n="ls_Ti") + " G = L.To; " + rec.format(n="ls_To"),
        "G = L.PSi; " + rec.format(n="ls_PSi") + " G = L.CSo; " + rec.format(n="ls_CSo"),
        "G = L.Li; " + rec.format(n="ls_Li") + " G = L.Lo; " + rec.format(n="ls_Lo"),
        "out.ls.poles = spt_c(L.Poles); out.ls.stable = L.Stable;",
        "S3 = sumblk('v = a + b - c'); out.sb.in = S3.InputName; out.sb.D = S3.Numerator;",
    ]
    return _wrap_lines(L)


def _wrap_lines(lines: list[str]) -> str:
    """Her satırı try/catch ile sarar: bir durumun hatası diğerlerini durdurmaz (hatalar out.errs'e)."""
    out = ["out.errs = {};"]
    for k, ln in enumerate(lines):
        out.append(f"try; {ln}; catch e_; out.errs{{end+1}} = sprintf('satir {k}: %s', e_.message); end")
    return "\n".join(out)


SCRIPT = _script()


@pytest.fixture(scope="module")
def M():
    try:
        return fixture("lti_core", SCRIPT)
    except OracleUnavailable as e:
        pytest.skip(str(e))


def test_matlab_script_ran_cleanly(M):
    assert not M.get("errs"), M.get("errs")


def _cplx(v) -> np.ndarray:
    a = np.asarray(v, float)
    if a.size == 0:
        return np.zeros(0, complex)
    a = a.reshape(-1, 2)
    return a[:, 0] + 1j * a[:, 1]


def _H(rec) -> np.ndarray:
    re, im = np.asarray(rec["re"], float), np.asarray(rec["im"], float)
    H = re + 1j * im
    if H.ndim == 1:
        H = H.reshape(1, 1, -1)
    return H


def _close(a, b, rtol=1e-9, scale=None):
    a = np.asarray(a, complex)
    b = np.asarray(b, complex)
    assert a.shape == b.shape, (a.shape, b.shape)
    sc = max(float(np.max(np.abs(b))) if b.size else 0.0, 1e-300) if scale is None else scale
    err = float(np.max(np.abs(a - b))) if a.size else 0.0
    assert err <= rtol * sc, f"fark {err:.3e} > {rtol:.0e}·{sc:.3e}"


def _sorted(v):
    v = np.asarray(v, complex)
    return v[np.lexsort((np.round(v.imag, 8), np.round(v.real, 8)))]


def _match_roots(a, b, rtol=1e-8):
    a, b = np.asarray(a, complex), np.asarray(b, complex)
    assert a.size == b.size, (a, b)
    used = np.zeros(b.size, bool)
    for x in a:
        d = np.abs(b - x)
        d[used] = np.inf
        k = int(np.argmin(d))
        assert d[k] <= rtol * max(1.0, abs(x)), (x, b)
        used[k] = True


def _norm_tf(n, d):
    n = np.trim_zeros(np.atleast_1d(np.asarray(n, float)), "f")
    d = np.trim_zeros(np.atleast_1d(np.asarray(d, float)), "f")
    if n.size == 0:
        return np.zeros(1), np.ones(1)
    return n / d[0], d / d[0]


def _env():
    env = {"np": np, "feedback": feedback, "series": series, "parallel": parallel, "append": append, "lft": lft,
           "inv": inv}
    for k, spec in BASE.items():
        env[k] = model(spec)[0]
    return env


# ---------------------------------------------------------------------------------------------------------------
# Dönüşümler
# ---------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("name", list(CONV))
def test_conversion_invariants(M, name):
    ref = M[name]
    py = model(CONV[name])[0]
    w = _wgrid(py)
    _close(freqresp(py, w), _H(ref))
    ny, nu = py.shape
    T = convert.to_tf(py)
    Z = convert.to_zpk(py)
    for i in range(ny):
        for j in range(nu):
            rn, rd = _norm_tf(ref[f"tf{i}{j}"]["n"], ref[f"tf{i}{j}"]["d"])
            mn, md = _norm_tf(T._num[i][j], T._den[i][j])
            assert md.size == rd.size, (name, i, j, md, rd)
            _close(md, rd, 1e-8)
            if np.any(rn):
                assert mn.size == rn.size, (mn, rn)
                _close(mn, rn, 1e-8)
            _match_roots(Z._p[i][j], _cplx(ref[f"zpk{i}{j}"]["p"]), 1e-6 if name in ("c3",) else 1e-8)
            _match_roots(Z._z[i][j], _cplx(ref[f"zpk{i}{j}"]["z"]), 1e-8)
            assert Z._k[i, j] == pytest.approx(ref[f"zpk{i}{j}"]["k"], rel=1e-9)
    S = convert.to_ss(py)
    assert S.nx == ref["n"], "ss(tf) derecesi MATLAB'inkiyle aynı olmalı"
    mk = np.asarray(ref["mk"], float).reshape(ny, nu, 7)
    mine = np.stack([S.D] + [S.C @ np.linalg.matrix_power(S.A, q) @ S.B for q in range(6)], axis=2)
    _close(mine, mk, 1e-9)
    _match_roots(np.linalg.eigvals(S.A), _cplx(ref["eig"]), 1e-6 if name in ("c3",) else 1e-8)
    from spt.control.analysis import pole, tzero
    _match_roots(pole(py), _cplx(ref["pole"]), 1e-6 if name in ("c3",) else 1e-8)
    _match_roots(tzero(py), _cplx(ref["tzero"]), 1e-7)


def test_ss_tf_realization_matches_matlab_siso():
    """SISO tf → ss: MATLAB'in matrisleriyle birebir (yoldaş biçimi + dengeleme)."""
    S = ss(tf([1, 2], [1, 3, 5, 7]))
    assert np.allclose(S.A, [[-3, -2.5, -1.75], [2, 0, 0], [0, 2, 0]], rtol=0, atol=1e-15)
    assert np.allclose(S.B.ravel(), [1, 0, 0]) and np.allclose(S.C.ravel(), [0, 0.5, 0.5]) and S.D[0, 0] == 0


def test_tf_construction_rules():
    G = tf([0, 0, 1, 2], [0, 2, 4])
    assert np.array_equal(G.num[0][0], [1, 2]) and np.array_equal(G.den[0][0], [2, 4])
    Z = tf([0], [1, 2, 3])
    assert np.array_equal(Z.den[0][0], [1.0])               # sıfır pay → payda 1 (MATLAB gibi)
    with pytest.raises(ControlError):
        tf([1], [0, 0])
    with pytest.raises(ControlError):
        tf([1], [1, 1], -1)
    with pytest.raises(ControlError):
        tf([1], [1, 1], 0.1, io_delay=0.5)                   # ayrıkta gecikme tamsayı örnek
    s = tf("s")
    assert isinstance(s, TransferFunction) and s.Ts == 0
    z = tf("z", 0.1)
    assert z.Ts == 0.1
    with pytest.raises(ControlError):
        tf("z")
    K = tf(np.array([[1.0, 2.0], [3.0, 4.0]]))
    assert K.shape == (2, 2) and K.is_static()
    with pytest.raises(ControlError):
        tf([1], [1, 1]) + tf([1], [1, 1], 0.1)               # Ts uyumsuz
    assert (tf(2) * tf([1], [1, -0.5], 0.1)).Ts == 0.1        # statik kazanç her Ts'ye uyar


# ---------------------------------------------------------------------------------------------------------------
# Aritmetik
# ---------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("name", list(ARITH))
def test_arithmetic(M, name):
    ref = M[name]
    G = eval(ARITH[name][0], _env())          # noqa: S307 (test içi sabit ifade)
    cls = {"tf": TransferFunction, "zpk": ZPK, "ss": StateSpace}[ref["cls"]]
    assert isinstance(G, cls), (type(G).__name__, ref["cls"])
    assert list(G.shape) == list(np.atleast_1d(ref["size"]))
    w = [x / G.Ts for x in W_D] if G.is_discrete else W_C
    _close(freqresp(G, w), _H(ref), 1e-9)
    if "tn" in ref and isinstance(G, TransferFunction) and G.is_siso:
        rn = np.trim_zeros(np.atleast_1d(np.asarray(ref["tn"], float)), "f")
        rd = np.trim_zeros(np.atleast_1d(np.asarray(ref["td"], float)), "f")
        mn, md = G.num[0][0], G.den[0][0]
        if name not in ("a_d1", "a_d3", "a_dfb"):               # ayrık gecikme katılımı: farklı (eşdeğer) gösterim
            assert md.size == rd.size and mn.size == max(rn.size, 1), (mn, md, rn, rd)
            _close(md, rd, 1e-12)
            _close(mn if np.any(mn) else np.zeros(1), rn if rn.size else np.zeros(1), 1e-12, scale=np.max(np.abs(rd)))
            tot = float(G.total_delay()[0, 0])
            assert tot == pytest.approx(float(np.sum(ref["io"]) + np.sum(ref["ind"]) + np.sum(ref["outd"])))
    if "zz" in ref and isinstance(G, ZPK) and G.is_siso:
        _match_roots(G.z[0][0], _cplx(ref["zz"]), 1e-8)
        _match_roots(G.p[0][0], _cplx(ref["zp"]), 1e-8)
        assert G.k[0, 0] == pytest.approx(ref["zk"], rel=1e-10)
    if ref["cls"] != "ss" and G.is_proper():
        from spt.control.analysis import pole
        n_mine = convert.to_ss(_nodelay(G)).nx if not G.is_siso else pole(G).size
        assert n_mine == ref["n"] or name in ("a_d1", "a_d3", "a_dfb"), (n_mine, ref["n"])


def _nodelay(G):
    out = G.copy()
    out.input_delay = np.zeros(G.ninputs)
    out.output_delay = np.zeros(G.noutputs)
    if hasattr(out, "io_delay"):
        out.io_delay = np.zeros(G.shape)
    return out


def test_delay_bookkeeping():
    """MATLAB'den ölçülen gecikme dağılımı (SISO tf çarpımında iç gecikmeler io'ya katlanır)."""
    r = tf([1], [1, 1], input_delay=0.1) * tf([1], [1, 2], output_delay=0.2)
    assert r.input_delay[0] == 0 and r.output_delay[0] == 0 and r.io_delay[0, 0] == pytest.approx(0.3)
    r = tf([1], [1, -0.5], 0.1, io_delay=2) * tf([1], [1, -0.3], 0.1, input_delay=1)
    assert r.io_delay[0, 0] == 2 and r.input_delay[0] == 1
    with pytest.raises(ControlError):
        tf([1], [1, 1], io_delay=0.1) + tf([1], [1, 2])      # sürekli: iç gecikme gerekir
    with pytest.raises(ControlError):
        feedback(tf([1], [1, 1], io_delay=0.5), 1)
    with pytest.raises(ControlError):
        inv(tf([1], [1, 1], io_delay=0.5))
    # ss: düzgün iç gecikme girişe taşınır, düzgün olmayan hata
    a = ss(-1, 1, 1, 0, input_delay=0.2)
    b = ss(-2, 1, 1, 0, output_delay=0.1)
    assert (a * b).input_delay[0] == pytest.approx(0.3)
    m = ss([[-1, 0], [0, -2]], [[1, 0], [0, 1]], [[1, 1]], [[0, 0]], input_delay=[0.1, 0.2])
    with pytest.raises(ControlError):
        m * ss([[-1]], [[1]], [[1], [1]], [[0], [0]])
    # ayrık: aynı yapı hata vermez; gecikme z⁻¹ durumu olarak tam katılır (frekans cevabı çarpıma eşit)
    md = ss([[0.5, 0], [0, 0.2]], [[1, 0], [0, 1]], [[1, 1]], [[0, 0]], 0.1, input_delay=[1, 2])
    b = ss([[0.1]], [[1]], [[1], [1]], [[0], [0]], 0.1)
    r = md * b
    w = np.array([0.3, 1.1, 2.9, 9.0, 27.0])
    Hm, Hb = freqresp(md, w), freqresp(b, w)
    _close(freqresp(r, w), np.einsum("ikw,kjw->ijw", Hm, Hb), 1e-12)


def test_mimo_indexing_and_names():
    G = tf([[[1], [2]], [[3], [1, 0]]], [[[1, 1], [1, 2]], [[1, 3], [1, 4]]], input_names=["u1", "u2"],
           output_names=["y1", "y2"], io_delay=[[0.1, 0.0], [0.3, 0.2]])
    sub = G[1, :]
    assert sub.shape == (1, 2) and sub.output_names == ("y2",) and np.allclose(sub.io_delay, [[0.3, 0.2]])
    assert G[:, 1].input_names == ("u2",)
    S = ss(G)                                   # io gecikmesi giriş [0.1, 0] + çıkış [0, 0.2] olarak ayrışır
    assert np.allclose(S.input_delay, [0.1, 0.0]) and np.allclose(S.output_delay, [0.0, 0.2])
    with pytest.raises(ControlError):           # ayrışmayan io gecikmesi: sürekli zamanda iç gecikme gerekir
        ss(tf([[[1], [2]], [[3], [1]]], [[[1, 1], [1, 2]], [[1, 3], [1, 4]]], io_delay=[[0.1, 0.0], [0.2, 0.3]]))
    assert tf(np.eye(2), input_names="u").input_names == ("u(1)", "u(2)")


# ---------------------------------------------------------------------------------------------------------------
# minreal / indirgeme / kanonik
# ---------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("name", list(MINREAL))
def test_minreal(M, name):
    ref = M[name]
    G = convert.minreal(model(MINREAL[name])[0])
    _close(freqresp(G, W_C), _H(ref), 1e-9)
    n = G.nx if isinstance(G, StateSpace) else len(G.p[0][0]) if isinstance(G, ZPK) else G.den[0][0].size - 1
    assert n == ref["n"]
    if "tn" in ref and isinstance(G, TransferFunction):
        _close(G.den[0][0], np.atleast_1d(ref["td"]), 1e-9)


@pytest.mark.parametrize("name", list(RED))
def test_balanced_reduction(M, name):
    G0 = model(RED[name])[0]
    w = _wgrid(G0)
    ref = M[name]
    hsv = convert.hsvd(G0)
    rh = np.asarray(ref["hsv"], float)
    assert hsv.size == rh.size
    fin = np.isfinite(rh)
    assert np.array_equal(np.isinf(hsv), ~fin)
    _close(hsv[fin], rh[fin], 1e-8)
    sb, g, T, Ti = convert.balreal(G0)
    _close(g[np.isfinite(g)], np.asarray(ref["g"], float)[np.isfinite(np.asarray(ref["g"], float))], 1e-8)
    _close(freqresp(sb, w), _H(M[name + "_bal"]), 1e-9)
    if np.all(np.isfinite(g)):
        from spt.control.analysis import gram
        _close(gram(sb, "c"), np.diag(g), 1e-8, scale=g[0])
        _close(gram(sb, "o"), np.diag(g), 1e-8, scale=g[0])
    S0 = convert.to_ss(G0)
    _close(T @ S0.A @ Ti, sb.A, 1e-8, scale=np.max(np.abs(sb.A)))
    _close(freqresp(convert.balred(G0, 2), w), _H(M[name + "_dc"]), 1e-8)
    _close(freqresp(convert.balred(G0, 2, "Truncate"), w), _H(M[name + "_tr"]), 1e-8)
    # modred dengeli koordinatlarda (durum işaretleri / ölçekleri bölmeyle değiştiği için sonuç gerçeklemeden bağımsız)
    _close(freqresp(convert.modred(sb, [2, 3]), w), _H(M[name + "_mr"]), 1e-8)
    _close(freqresp(convert.modred(sb, [2, 3], "Truncate"), w), _H(M[name + "_mt"]), 1e-8)


def test_modred_plain_realization(M):
    G = convert.modred(model(RED["rm"])[0], [0, 2])
    _close(freqresp(G, W_C), _H(M["rm_raw"]), 1e-9)
    with pytest.raises(ControlError):          # A22 tekil: MatchDC tanımsız (MATLAB tanımlayıcı modele geçer)
        convert.modred(ss(tf([1], [1, 3, 3, 1, 0.2])), [0, 2])


def _blocks(A):
    """Modal biçimin blokları: gerçel özdeğer (λ, 0) ya da [[σ, ω], [−ω, σ]] → (σ, ω)."""
    out, i, n = [], 0, A.shape[0]
    while i < n:
        if i + 1 < n and abs(A[i, i + 1]) > 1e-12:
            assert A[i, i] == pytest.approx(A[i + 1, i + 1]) and A[i, i + 1] == pytest.approx(-A[i + 1, i])
            out.append((A[i, i], abs(A[i, i + 1])))
            i += 2
        else:
            out.append((A[i, i], 0.0))
            i += 1
    return out


@pytest.mark.parametrize("name", list(CANON))
def test_canon(M, name):
    ref = M[name]
    G0 = model(CANON[name])[0]
    cm, T = convert.canon(G0, "modal")
    # Blok kümesi MATLAB'inkiyle aynı (σ, ω); blok SIRASI MATLAB'de Schur sırasına bağlı olduğundan karşılaştırılmaz
    ra = np.asarray(ref["modal"], float)
    mine, theirs = _blocks(cm.A), _blocks(ra)
    assert len(mine) == len(theirs)
    for b in theirs:
        k = int(np.argmin([abs(complex(*b) - complex(*x)) for x in mine]))
        assert abs(complex(*b) - complex(*mine[k])) <= 1e-9 * max(1.0, abs(complex(*b))), (b, mine)
        mine.pop(k)
    assert np.count_nonzero(np.abs(cm.A) > 1e-12) == np.count_nonzero(np.abs(ra) > 1e-12)
    w = _wgrid(G0)
    _close(freqresp(cm, w), freqresp(G0, w), 1e-9)
    cc, T = convert.canon(G0, "companion")
    for key, mat in (("ca", cc.A), ("cb", cc.B), ("cc", cc.C), ("cd", cc.D)):
        r = np.asarray(ref[key], float).reshape(mat.shape)
        _close(mat, r, 1e-9, scale=max(1.0, float(np.max(np.abs(r)))))
    S = convert.to_ss(G0)
    _close(T @ S.A @ np.linalg.inv(T), cc.A, 1e-8, scale=np.max(np.abs(cc.A)))


# ---------------------------------------------------------------------------------------------------------------
# Adlarla bağlantı / loopsens / sumblk
# ---------------------------------------------------------------------------------------------------------------
def test_connect_and_loopsens(M):
    C = tf(2, 1, input_names="e", output_names="u")
    P = tf([1], [1, 1], input_names="u", output_names="y")
    S = sumblk("e = r - y")
    G = connect(P, C, S, "r", "y")
    assert isinstance(G, StateSpace) and G.input_names == ("r",) and G.output_names == ("y",)
    _close(freqresp(G, W_C), _H(M["n_cl"]))
    _close(freqresp(connect(P, C, S, ["r", "u"], ["y", "u", "e"]), W_C), _H(M["n_ap"]))
    _close(freqresp(connect(P, C, S, "u", "y"), W_C), _H(M["n_u"]))
    P2 = ss([[-1, 0.5], [0, -2]], [[1, 0], [0, 1]], [[1, 0], [0.5, 1]], 0, input_names="u", output_names="y")
    K2 = ss(-0.5, [[1, 0.2]], [[1], [0.3]], [[2, 0], [0, 1.5]], input_names="e", output_names="u")
    S2 = sumblk("e = r - y", 2)
    _close(freqresp(connect(P2, K2, S2, "r", "y"), W_C), _H(M["n_mimo"]))
    L = loopsens(P2, ss(-0.5, [[1, 0.2]], [[1], [0.3]], [[2, 0], [0, 1.5]]))
    for key in ("Si", "So", "Ti", "To", "PSi", "CSo", "Li", "Lo"):
        _close(freqresp(L[key], W_C), _H(M["ls_" + key]))
    _match_roots(L["Poles"], _cplx(M["ls"]["poles"]), 1e-8)
    assert L["Stable"] == bool(M["ls"]["stable"])
    s3 = sumblk("v = a + b - c")
    assert s3.input_names == tuple(M["sb"]["in"])
    assert np.allclose([float(n[0]) for n in s3.num[0]], [1, 1, -1])
    with pytest.raises(ControlError):
        connect(tf([1], [1, 1], input_names="u", output_names="y"), tf([1], [1, 2], input_names="u",
                                                                       output_names="y"), "u", "y")


# ---------------------------------------------------------------------------------------------------------------
# JSON, gösterim, hata iletileri
# ---------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["c1", "c4", "c6", "c7", "c8", "c13"])
def test_to_from_dict_roundtrip(name):
    import json
    G = model(CONV[name])[0]
    G.input_delay = G.input_delay + (0 if G.is_discrete else 0.0)
    d = json.loads(json.dumps(G.to_dict()))
    H = LTI.from_dict(d)
    assert type(H) is type(G) and H.shape == G.shape and H.Ts == G.Ts
    w = _wgrid(G)
    _close(freqresp(H, w), freqresp(G, w), 1e-14)


def test_display_matlab_like():
    s = str(tf([1, 2], [1, 3, 5]))
    assert "s + 2" in s and "s^2 + 3 s + 5" in s and "-------------" in s and "Sürekli zamanlı" in s
    s = str(zpk([-1 + 2j, -1 - 2j, 0], [-3, 0.5], -2.5))
    assert "-2.5 s (s^2 + 2s + 5)" in s and "(s+3) (s-0.5)" in s
    s = str(tf([1, -0.5], [1, -1.2, 0.4], 0.1, io_delay=2))
    assert "z^(-2) *" in s and "z^2 - 1.2 z + 0.4" in s and "Örnekleme zamanı: 0.1 s" in s
    s = str(tf([1, 0, 0], [1, 1e-5, 123456.789]))
    assert "1e-05 s + 1.235e05" in s
    s = str(ss([[0, 1], [-2, -3]], [[0], [1]], [[1, 0]], 0))
    assert "A = " in s and "x1" in s and "durum uzayı" in s
    s = str(zpk([], [-1, -1], 1))
    assert "(s+1)^2" in s


def test_errors_are_turkish_control_errors():
    with pytest.raises(ControlError, match="boyut"):
        tf([[[1], [1]]], [[[1, 1], [1, 2]]]) * tf([[[1], [1]]], [[[1, 1], [1, 2]]])
    with pytest.raises(ControlError, match="Uygun olmayan"):
        ss(tf([1, 0, 0], [1, 1]))
    with pytest.raises(ControlError):
        tf([1], [1, 1]) ** 0.5
