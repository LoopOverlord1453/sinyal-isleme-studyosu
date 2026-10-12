"""Zaman cevabı: step, impulse, initial, lsim (zoh / foh, kesirli gecikmeler, MIMO, ayrık), stepinfo, lsiminfo.

Referans: MATLAB R2025b (tests/data/matlab/lti_timeresp.json). Açık zaman ızgaralarında y (nt, ny, nu) göreli 1e-9
(ölçek: max|y|). stepinfo dizi girişinde MATLAB'in tanımlarıyla 1e-10; model girişinde MATLAB ızgara tabanlı
(yaklaşık) olduğundan kesin analitik değerlerle (2. derece aşım, tepe anı, 1. derece yükselme / yerleşme) 1e-10.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from spt.control.errors import ControlError
from spt.control.lti import ss, tf, zpk
from spt.control.timeresp import impulse, initial, lsim, lsiminfo, step, stepinfo
from tools.matlab_oracle import OracleUnavailable, fixture


def _m(x) -> str:
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
        def cv(vals):
            items = [f"complex({float(v.real)!r},{float(v.imag)!r})" for v in np.asarray(vals, complex)]
            return "[" + " ".join(items) + "]"

        zs = cv(z) if len(z) else "[]"
        ps = cv(p)
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
SYS = {
    "u2": (TF([1], [1, 0.6, 1]), 0.1, 30),
    "nm": (TF([-1, 1], [1, 2, 1]), 0.1, 15),
    "bp": (TF([2, 3, 1], [1, 5, 6]), 0.05, 8),
    "in": (TF([1], [1, 1, 0]), 0.1, 10),
    "un": (TF([1], [1, -0.5]), 0.1, 6),
    "ud": (TF([1], [1, 0, 4]), 0.05, 10),
    "h8": (ZK([-1.5, 2 + 1j, 2 - 1j], [-0.5 + 3j, -0.5 - 3j, -1 + 1j, -1 - 1j, -2, -3, -4 + 0.5j, -4 - 0.5j], 7.0),
           0.05, 12),
    "dl": (TF([1, 3], [1, 2, 5], io=0.537), 0.1, 8),
    "dm": (TF([[[1], [1, 0.5]], [[2], [1]]], [[[1, 1], [1, 2, 2]], [[1, 3], [1, 0.5]]],
              io=[[0.13, 0.0], [0.25, 0.4]]), 0.1, 8),
    "sm": (SS(A3, [[0, 1], [1, 0], [0, 2]], [[1, 0, 0], [0, 1, 1]], [[0.3, 0.5], [0, 0]], ind=[0.25, 0.13],
              outd=[0.0, 0.07]), 0.05, 8),
    "z1": (TF([0.2, 0.1], [1, -1.2, 0.5], 0.05), 0.05, 4),
    "zd": (SS([[0.5, 0.1], [-0.2, 0.7]], [[1, 0], [0, 1]], [[1, 0], [0.5, 1]], [[0, 0.2], [0, 0]], 0.2,
              ind=[2, 0], outd=[1, 3]), 0.2, 6),
}
INIT = {
    "i1": (SS([[0, 1], [-2, -0.4]], [[0], [1]], [[1, 0], [0.5, 1]], [[0], [0]]), [1.0, -0.5], 0.1, 15),
    "i2": (SS([[0, 1], [-2, -0.4]], [[0], [1]], [[1, 0], [0.5, 1]], [[0], [0]], outd=[0.33, 0.0]), [1.0, -0.5],
           0.1, 8),
    "i3": (SS([[0.9, 0.2], [-0.1, 0.7]], [[1], [0]], [[1, 1]], [[0]], 0.1, outd=[2]), [1.0, 2.0], 0.1, 3),
}
LSIM = {
    "l1": ("u2", "zoh"), "l2": ("u2", "foh"), "l3": ("dl", "zoh"), "l4": ("dl", "foh"), "l5": ("sm", "zoh"),
    "l6": ("sm", "foh"), "l7": ("dm", "foh"), "l8": ("z1", "zoh"), "l9": ("zd", "zoh"), "l10": ("h8", "foh"),
}
LSIM_X0 = {"lx": (SS(A3, [[0, 1], [1, 0], [0, 2]], [[1, 0, 0], [0, 1, 1]], [[0.3, 0.5], [0, 0]], ind=[0.25, 0.0]),
                  [0.5, -1.0, 0.2], "foh")}
# stepinfo dizi durumları (MATLAB'de üretilir): ad → (MATLAB y ifadesi, t ifadesi, ek argümanlar)
SI = {
    "s_u": ("step(tf(1,[1 0.6 1]), t)", "(0:0.37:25)'", ""),
    "s_yf": ("step(tf(1,[1 0.6 1]), t)", "(0:0.37:25)'", ", 1"),
    "s_thr": ("step(tf(1,[1 0.6 1]), t)", "(0:0.37:25)'",
              ", 'SettlingTimeThreshold', 0.05, 'RiseTimeLimits', [0.05 0.95]"),
    "s_nm": ("step(tf([-1 1],[1 2 1]), t)", "(0:0.21:15)'", ""),
    "s_off": ("0.5 + step(tf(1,[1 0.6 1]), t)", "(0:0.37:25)'", ""),
    "s_off2": ("0.5 + step(tf(1,[1 0.6 1]), t)", "(0:0.37:25)'", ", 1.5, 0.5"),
    "s_neg": ("-2*step(tf(1,[1 1.2 1]), t)", "(0:0.13:15)'", ""),
    "s_osc": ("step(tf(4,[1 0.4 4]), t)", "(0:0.05:40)'", ", 'SettlingTimeThreshold', 0.01"),
    "s_ramp": ("t/10 + 0.1*sin(3*t)", "(0:0.1:10)'", ""),
}
LSI = {
    "li1": ("exp(-t).*cos(3*t) + 0.2", "(0:0.07:8)'", ""),
    "li2": ("1 - exp(-0.5*t) + 0.3*exp(-t).*sin(5*t)", "(0:0.1:15)'", ", 1"),
    "li3": ("-t.*exp(-t)", "(0:0.05:12)'", ""),
}
STEPINFO_SYS = {"q1": TF([1], [1, 0.6, 1], 0.0), "q2": TF([0.02, 0.01], [1, -1.6, 0.67], 0.1),
                "q3": TF([0.3], [1, -0.7], 0.1, io=3), "q4": TF([-0.05, 0.1], [1, -1.5, 0.6], 0.2),
                "q5": TF([2, 3, 1], [1, 5, 6])}


def _tvec(dt, T):
    return np.arange(int(round(T / dt)) + 1) * dt


def _uin(t, nu):
    cols = [np.sin(1.3 * t) + 0.5 * np.cos(0.4 * t), np.where(t > 1.0, 1.0, 0.0) - 0.3 * np.sin(2.1 * t)]
    return np.column_stack(cols[:nu])


def _script() -> str:
    L = []
    for name, (spec, dt, T) in SYS.items():
        mx = model(spec)[1]
        L.append(f"G = {mx}; t = (0:{int(round(T / dt))})'*{_m(dt)}; out.{name}.y = step(G, t); "
                 f"out.{name}.h = impulse(G, t);")
    for name, (spec, x0, dt, T) in INIT.items():
        L.append(f"G = {model(spec)[1]}; t = (0:{int(round(T / dt))})'*{_m(dt)}; out.{name}.y = initial(G, "
                 f"{_mv(x0)}', t);")
    for name, (base, meth) in LSIM.items():
        spec, dt, T = SYS[base]
        py, mx = model(spec)
        n = int(round(T / dt))
        L.append(f"G = {mx}; t = (0:{n})'*{_m(dt)}; u = [sin(1.3*t) + 0.5*cos(0.4*t), (t > 1.0) - 0.3*sin(2.1*t)];"
                 f" u = u(:, 1:size(G, 2)); out.{name}.y = lsim(G, u, t, '{meth}');")
    for name, (spec, x0, meth) in LSIM_X0.items():
        L.append(f"G = {model(spec)[1]}; t = (0:80)'*0.05; u = [sin(1.3*t) + 0.5*cos(0.4*t), (t > 1.0) - "
                 f"0.3*sin(2.1*t)]; out.{name}.y = lsim(G, u, t, {_mv(x0)}', '{meth}');")
    for name, (yexpr, texpr, extra) in SI.items():
        L.append(f"t = {texpr}; y = {yexpr}; out.{name}.y = y; out.{name}.t = t; out.{name}.S = stepinfo(y, t{extra});")
    L.append("y = [0 .2 .5 .9 1.2 1.1 .95 1.02 .99 1.0 1.0]'; out.s_idx.S = stepinfo(y); out.s_idx.y = y;")
    for name, (yexpr, texpr, extra) in LSI.items():
        L.append(f"t = {texpr}; y = {yexpr}; out.{name}.y = y; out.{name}.t = t; out.{name}.S = lsiminfo(y, t{extra});")
    L.append("y = [1 3 2 2.5 2.2 2.3 2.25 2.25]'; out.li_idx.S = lsiminfo(y); out.li_idx.y = y;")
    for name, spec in STEPINFO_SYS.items():
        L.append(f"out.{name}.S = stepinfo({model(spec)[1]});")
    out = ["out.errs = {};"]
    for k, ln in enumerate(L):
        out.append(f"try; {ln}; catch e_; out.errs{{end+1}} = sprintf('satir {k}: %s', e_.message); end")
    return "\n".join(out)


SCRIPT = _script()


@pytest.fixture(scope="module")
def M():
    try:
        return fixture("lti_timeresp", SCRIPT)
    except OracleUnavailable as e:
        pytest.skip(str(e))


def test_matlab_script_ran_cleanly(M):
    assert not M.get("errs"), M.get("errs")


def _close(a, b, rtol=1e-9, scale=None):
    a, b = np.asarray(a, float), np.asarray(b, float)
    assert a.shape == b.shape, (a.shape, b.shape)
    sc = (float(np.max(np.abs(b))) or 1.0) if scale is None else scale
    err = float(np.max(np.abs(a - b)))
    assert err <= rtol * sc, f"fark {err:.3e} > {rtol:.0e}·{sc:.3e}"


def _y3(ref, shape):
    return np.asarray(ref, float).reshape(shape)


@pytest.mark.parametrize("name", list(SYS))
def test_step_impulse(M, name):
    spec, dt, T = SYS[name]
    G = model(spec)[0]
    t = _tvec(dt, T)
    y, tt, *_ = step(G, t)
    ny, nu = G.shape
    _close(y, _y3(M[name]["y"], (t.size, ny, nu)), 1e-9)
    h, _ = impulse(G, t)
    _close(h, _y3(M[name]["h"], (t.size, ny, nu)), 1e-9)


@pytest.mark.parametrize("name", list(INIT))
def test_initial(M, name):
    spec, x0, dt, T = INIT[name]
    G = model(spec)[0]
    t = _tvec(dt, T)
    y, _ = initial(G, x0, t)
    _close(y, np.asarray(M[name]["y"], float).reshape(t.size, G.noutputs), 1e-9)


@pytest.mark.parametrize("name", list(LSIM))
def test_lsim(M, name):
    base, meth = LSIM[name]
    spec, dt, T = SYS[base]
    G = model(spec)[0]
    t = _tvec(dt, T)
    u = _uin(t, G.ninputs)
    y, _ = lsim(G, u, t, method=meth)
    _close(y, np.asarray(M[name]["y"], float).reshape(t.size, G.noutputs), 1e-9)


def test_lsim_x0(M):
    spec, x0, meth = LSIM_X0["lx"]
    G = model(spec)[0]
    t = np.arange(81) * 0.05
    y, _, x = lsim(G, _uin(t, 2), t, x0, meth, return_x=True)
    _close(y, np.asarray(M["lx"]["y"], float).reshape(t.size, 2), 1e-9)
    assert np.allclose(x[0], x0)


def _cmp_info(mine: dict, ref: dict, rtol=1e-10):
    for k, v in ref.items():
        a = mine[k]
        if v is None or (isinstance(v, float) and math.isnan(v)):
            assert math.isnan(a), (k, a)
        elif isinstance(v, float) and math.isinf(v):
            assert a == v, (k, a, v)
        else:
            assert a == pytest.approx(v, rel=rtol, abs=1e-12), (k, a, v)


def _si_args(extra: str) -> dict:
    if not extra:
        return {}
    if "SettlingTimeThreshold" in extra:
        kw = {"SettlingTimeThreshold": float(extra.split("'SettlingTimeThreshold', ")[1].split(",")[0])}
        if "RiseTimeLimits" in extra:
            lo, hi = extra.split("[")[1].split("]")[0].split()
            kw["RiseTimeLimits"] = (float(lo), float(hi))
        return kw
    vals = [float(v) for v in extra.strip(", ").split(",")]
    return {"yfinal": vals[0], **({"yinit": vals[1]} if len(vals) > 1 else {})}


@pytest.mark.parametrize("name", list(SI))
def test_stepinfo_array(M, name):
    ref = M[name]
    y = np.asarray(ref["y"], float).ravel()
    t = np.asarray(ref["t"], float).ravel()
    _cmp_info(stepinfo(y, t, **_si_args(SI[name][2])), ref["S"])


def test_stepinfo_index_time(M):
    _cmp_info(stepinfo(np.asarray(M["s_idx"]["y"], float)), M["s_idx"]["S"])


@pytest.mark.parametrize("name", list(LSI))
def test_lsiminfo(M, name):
    ref = M[name]
    y = np.asarray(ref["y"], float).ravel()
    t = np.asarray(ref["t"], float).ravel()
    extra = LSI[name][2]
    kw = {"yfinal": float(extra.strip(", "))} if extra else {}
    _cmp_info(lsiminfo(y, t, **kw), ref["S"])


def test_lsiminfo_index_time(M):
    _cmp_info(lsiminfo(np.asarray(M["li_idx"]["y"], float)), M["li_idx"]["S"])


@pytest.mark.parametrize("name", ["q2", "q3", "q4"])
def test_stepinfo_discrete_system(M, name):
    """Ayrık modelde stepinfo örnek anlarında (aradeğersiz): MATLAB ile birebir."""
    G = model(STEPINFO_SYS[name])[0]
    mine = stepinfo(G)
    ref = M[name]["S"]
    if math.isinf(mine["PeakTime"]):            # tekdüze cevap: tepe sonsuzdaki sınır (MATLAB ızgara sonunu verir)
        ref = {k: v for k, v in ref.items() if k not in ("Peak", "PeakTime")}
    _cmp_info(mine, ref)


def test_stepinfo_continuous_exact():
    for zeta in (0.05, 0.3, 0.7):
        wn = 2.0
        S = stepinfo(tf([wn * wn], [1, 2 * zeta * wn, wn * wn]))
        os_ = 100 * math.exp(-math.pi * zeta / math.sqrt(1 - zeta * zeta))
        tp = math.pi / (wn * math.sqrt(1 - zeta * zeta))
        assert S["Overshoot"] == pytest.approx(os_, rel=1e-10)
        assert S["PeakTime"] == pytest.approx(tp, rel=1e-12)
        assert S["Peak"] == pytest.approx(1 + os_ / 100, rel=1e-12)
    a = 3.0
    S = stepinfo(tf([a], [1, a]))
    assert S["RiseTime"] == pytest.approx(math.log(9) / a, rel=1e-12)
    assert S["SettlingTime"] == pytest.approx(math.log(50) / a, rel=1e-12)
    assert S["Overshoot"] == 0 and math.isinf(S["PeakTime"]) and S["Peak"] == pytest.approx(1.0)
    S = stepinfo(tf([a], [1, a], input_delay=0.4))
    assert S["SettlingTime"] == pytest.approx(0.4 + math.log(50) / a, rel=1e-12)
    for G in (tf([1], [1, 0]), tf([1], [1, 0, 1]), tf([1], [1, -1])):
        S = stepinfo(G)
        assert math.isnan(S["RiseTime"]) and S["Peak"] == math.inf


def test_stepinfo_continuous_vs_matlab_grid(M):
    """MATLAB stepinfo(sys) ızgara tabanlı: kesin değerimize göre hatası birkaç 1e-3 içinde (bilinen fark)."""
    mine = stepinfo(model(STEPINFO_SYS["q1"])[0])
    ref = M["q1"]["S"]
    for k in ("RiseTime", "SettlingTime", "Overshoot", "Peak", "PeakTime"):
        assert mine[k] == pytest.approx(ref[k], rel=3e-2)


def test_step_shapes_and_errors():
    G = ss([[0, 1], [-2, -3]], [[0, 1], [1, 0]], [[1, 0]], [[0, 0]])
    y, t, x = step(G, 5.0, return_x=True)
    assert y.shape == (t.size, 1, 2) and x.shape == (t.size, 2, 2) and t[-1] == pytest.approx(5.0)
    y, t = step(tf([1], [1, -0.5], 0.1))
    assert np.allclose(np.diff(t), 0.1)
    with pytest.raises(ControlError):
        step(tf([1], [1, 1], 0.1), [0, 0.05, 0.1])
    with pytest.raises(ControlError):
        lsim(tf([1], [1, 1]), np.ones(3), [0, 0.1, 0.3])
    with pytest.raises(ControlError):
        initial(tf([1], [1, 1]), [1.0])
    y, _ = step(tf([1], [1, 1]))
    assert y[-1, 0, 0] == pytest.approx(1.0, abs=1e-3)


def test_stepinfo_zero_final_value_matches_matlab_semantics():
    """Son değer = başlangıç değeri (ör. S = feedback(1, L)): önceden tüm ölçütler 0 dönüyordu. MATLAB R2025b
    (ölçülen): RiseTime 0, SettlingTime NaN, Overshoot = Undershoot = ∞, TransientTime / SettlingMin / SettlingMax /
    Peak / PeakTime olağan tanımla."""
    from spt.control.connect import feedback
    S = feedback(1, tf([4], [1, 2, 0]))
    info = stepinfo(S)
    assert info["RiseTime"] == 0 and math.isnan(info["SettlingTime"])
    assert math.isinf(info["Overshoot"]) and math.isinf(info["Undershoot"])
    assert info["Peak"] == pytest.approx(1.0) and info["PeakTime"] == 0
    assert info["TransientTime"] == pytest.approx(4.037934116881348, rel=1e-3)   # MATLAB (ızgaralı)
    assert info["SettlingMin"] == pytest.approx(-0.1629291986994923, rel=2e-3)
    G = stepinfo(tf([1, 0], [1, 3, 2]))                                           # tepe kesin: 1/4, t = ln 2
    assert G["Peak"] == pytest.approx(0.25, rel=1e-6) and G["PeakTime"] == pytest.approx(math.log(2), rel=1e-3)
    t = np.arange(0, 5.0001, 0.01)
    d = stepinfo(-np.exp(-2 * t), t, 0)                                           # dizi yolu: MATLAB ile birebir
    assert d["TransientTime"] == pytest.approx(1.956035463216449, rel=1e-12)
    assert d["SettlingMax"] == pytest.approx(-4.539992976248485e-05, rel=1e-9) and d["SettlingMin"] == -1.0
    assert stepinfo(tf([0], [1, 1]))["Peak"] == 0                                 # sıfır sistem


def test_stepinfo_peak_ties_and_zero_final():
    """MATLAB: (s²−3s+2)/(s²+3s+2) tepesi t = 0'da da erişilir → PeakTime 0; ayrık son değer 0 → SettlingMin 0."""
    S = stepinfo(tf([1, -3, 2], [1, 3, 2]))
    assert S["PeakTime"] == 0.0 and S["Peak"] == pytest.approx(1.0)
    assert S["Undershoot"] == pytest.approx(50.0, rel=1e-12)          # kesin (MATLAB ızgarada 49.9992)
    S = stepinfo(tf([1, -1], [1, -0.5], 0.1))
    assert S["SettlingMin"] == 0.0 and S["SettlingMax"] == 1.0 and S["PeakTime"] == 0.0
    assert S["TransientTime"] == pytest.approx(0.6)
