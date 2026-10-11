"""Blok diyagram doğrusallaştırması / çalışma noktası / Kalman ↔ MATLAB R2025b (Simulink Control Design `linearize`,
`findop`, Control System Toolbox `kalman`).

Model aynı dönüştürücüyle (test_control_diagram_simulink.build_lines) Simulink'te kurulur; doğrusallaştırma
noktaları `linio` ile aynı sinyallere konur. Karşılaştırma gerçeklemeden bağımsızdır: örnekleme zamanı, durum
sayısı ve frekans cevabı (göreli 1e-8).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spt.control.diagram import Model, create_block, findop, linearize, linio, operspec  # noqa: E402
from spt.control.diagram.engine import OperatingPoint  # noqa: E402
from test_control_diagram_simulink import _mat, _num, build_lines  # noqa: E402
from tools.matlab_oracle import OracleUnavailable, fixture  # noqa: E402

# ============================================================================ vakalar
# her vaka: (model, noktalar [(blok, port, tür)], kök giriş değerleri {port: değer} | None, seçenekler)
LCASES: dict[str, Callable[[], tuple]] = {}


def lcase(fn):
    LCASES[fn.__name__] = fn
    return fn


def _loop(m, ref, ctrl, plant, name="E"):
    s = m.add("Sum", name, signs="+-")
    m.connect(ref, s)
    m.connect(s, ctrl)
    m.connect(ctrl, plant)
    m.connect(plant, s, dst_port=1)


def _pid_plant_model(**pid):
    m = Model()
    r = m.add("Step", "R")
    c = m.add("PIDController", "C", **pid)
    g = m.add("TransferFcn", "G", num=[1.0, 2.0], den=[1.0, 3.0, 4.0, 1.0])
    _loop(m, r, c, g)
    return m


@lcase
def lin_pid_closed():
    m = _pid_plant_model(P=2.0, I=1.5, D=0.3, N=40.0)
    return m, [("R", 0, "input"), ("G", 0, "output")], None, {}


@lcase
def lin_pid_looptransfer():
    m = _pid_plant_model(P=2.0, I=1.5, D=0.3, N=40.0, form="ideal")
    return m, [("C", 0, "looptransfer")], None, {}


@lcase
def lin_pid_sat_inout():
    m = _pid_plant_model(P=2.0, I=1.5, D=0.3, N=40.0, limit_output=True, upper=1.0, lower=-1.0,
                         anti_windup="back_calculation", Kb=2.0)
    return m, [("E", 0, "inout"), ("C", 0, "outin")], None, {}


@lcase
def lin_disc_pid_sampled():
    m = _pid_plant_model(P=2.0, I=1.5, D=0.3, N=20.0, Ts=0.1, integrator_method="backward_euler",
                         filter_method="trapezoidal")
    return m, [("R", 0, "input"), ("G", 0, "output")], None, {}


@lcase
def lin_two_rates():
    m = Model()
    r = m.add("Step", "R")
    c1 = m.add("DiscreteTransferFcn", "C1", num=[2.0, -1.6], den=[1.0, -1.0], Ts=0.1)
    c2 = m.add("DiscreteTransferFcn", "C2", num=[1.0, -0.5], den=[1.0, 0.2], Ts=0.02)
    g = m.add("TransferFcn", "G", num=[1.0], den=[1.0, 1.2, 1.0])
    s = m.add("Sum", "E", signs="+-")
    m.connect(r, s)
    m.connect(s, c1)
    m.connect(c1, c2)
    m.connect(c2, g)
    m.connect(g, s, dst_port=1)
    return m, [("R", 0, "input"), ("G", 0, "output")], None, {}


@lcase
def lin_series_zoh_chain():
    m = Model()
    i = m.add("Inport", "In")
    z = m.add("ZeroOrderHold", "Z", Ts=0.1)
    g1 = m.add("TransferFcn", "G1", num=[1.0], den=[1.0, 1.0])
    k = m.add("Gain", "K", K=3.0)
    g2 = m.add("TransferFcn", "G2", num=[1.0, 0.5], den=[1.0, 2.0, 5.0])
    o = m.add("Outport", "Out")
    m.connect(i, z)
    m.connect(z, g1)
    m.connect(g1, k)
    m.connect(k, g2)
    m.connect(g2, o)
    return m, [("In", 0, "input"), ("G2", 0, "output")], None, {}


@lcase
def lin_disc_to_cont_zoh():
    m = Model()
    i = m.add("Inport", "In")
    d1 = m.add("DiscreteTransferFcn", "D1", num=[0.3, 0.1], den=[1.0, -0.6], Ts=0.05)
    d2 = m.add("DiscreteStateSpace", "D2", A=[[0.8, 0.1], [0.0, 0.7]], B=[[1.0], [0.5]], C=[[1.0, 1.0]],
               D=[[0.0]], Ts=0.05)
    o = m.add("Outport", "Out")
    m.connect(i, d1)
    m.connect(d1, d2)
    m.connect(d2, o)
    return m, [("In", 0, "input"), ("D2", 0, "output")], None, {"Ts": 0.0}


@lcase
def lin_disc_to_cont_tustin():
    m, pts, _, _ = lin_disc_to_cont_zoh()
    return m, pts, None, {"Ts": 0.0, "rate_conversion": "tustin"}


@lcase
def lin_upsample_zoh():
    """Hızlı ayrık bloğun yavaş hedef hıza dönüşümü + yavaş hız 0.1 (tam kat 5)."""
    m = Model()
    i = m.add("Inport", "In")
    d1 = m.add("DiscreteTransferFcn", "D1", num=[0.2], den=[1.0, -0.8], Ts=0.02)
    d2 = m.add("DiscreteTransferFcn", "D2", num=[1.0, 0.3], den=[1.0, -0.5], Ts=0.1)
    o = m.add("Outport", "Out")
    m.connect(i, d1)
    m.connect(d1, d2)
    m.connect(d2, o)
    return m, [("In", 0, "input"), ("D2", 0, "output")], None, {}


@lcase
def lin_delay_pade():
    m = Model()
    r = m.add("Step", "R")
    c = m.add("PIDController", "C", controller="PI", P=0.8, I=0.5)
    td = m.add("TransportDelay", "TD", delay=0.37, pade_order=3)
    g = m.add("TransferFcn", "G", num=[1.0], den=[2.0, 1.0])
    s = m.add("Sum", "E", signs="+-")
    m.connect(r, s)
    m.connect(s, c)
    m.connect(c, td)
    m.connect(td, g)
    m.connect(g, s, dst_port=1)
    return m, [("R", 0, "input"), ("G", 0, "output")], None, {}


def _nl_model():
    m = Model()
    i = m.add("Inport", "In")
    sat = m.add("Saturation", "SAT", upper=0.5, lower=-0.5, linearize_as_gain=False)
    dz = m.add("DeadZone", "DZ", start=-0.5, end=0.5, linearize_as_gain=False)
    lut = m.add("Lookup1D", "LUT", breakpoints=[-1.0, 0.0, 0.5, 1.2], table=[2.0, 0.0, 1.0, -1.0])
    sw = m.add("Switch", "SW", criteria="ge", threshold=0.3)
    mm = m.add("MinMax", "MM", function="max", inputs=2)
    c = m.add("Constant", "C", value=0.25)
    pr = m.add("Product", "PR", ops="**")
    f = m.add("Fcn", "F", expr="u^2*cos(u) + atan2(u, 2)")
    for b in (sat, dz, lut, f):
        m.connect(i, b)
    m.connect(i, sw, dst_port=0)
    m.connect(i, sw, dst_port=1)
    m.connect(c, sw, dst_port=2)
    m.connect(i, mm, dst_port=0)
    m.connect(c, mm, dst_port=1)
    m.connect(i, pr, dst_port=0)
    m.connect(f, pr, dst_port=1)
    for k, b in enumerate((sat, dz, lut, sw, mm, pr), start=1):
        o = m.add("Outport", f"O{k}", port=k)
        m.connect(b, o)
    pts = [("In", 0, "input")] + [(n, 0, "output") for n in ("SAT", "DZ", "LUT", "SW", "MM", "PR")]
    return m, pts


@lcase
def lin_nl_op_high():
    m, pts = _nl_model()
    return m, pts, {1: 0.7}, {}


@lcase
def lin_nl_op_low():
    m, pts = _nl_model()
    return m, pts, {1: 0.2}, {}


@lcase
def lin_rl_quant_gain_off():
    m = Model()
    i = m.add("Inport", "In")
    rl = m.add("RateLimiter", "RL", rising=1.0, falling=-1.0, linearize_as_gain=False)
    q = m.add("Quantizer", "Q", interval=0.25, linearize_as_gain=False)
    o1 = m.add("Outport", "O1", port=1)
    o2 = m.add("Outport", "O2", port=2)
    m.connect(i, rl)
    m.connect(i, q)
    m.connect(rl, o1)
    m.connect(q, o2)
    return m, [("In", 0, "input"), ("RL", 0, "output"), ("Q", 0, "output")], {1: 0.3}, {}


@lcase
def lin_2dof_pid():
    m = Model()
    r = m.add("Step", "R")
    c = m.add("PIDController2DOF", "C", P=3.0, I=2.0, D=0.4, N=30.0, b=0.6, c=0.2)
    g = m.add("TransferFcn", "G", num=[1.0], den=[1.0, 1.0, 0.0])
    m.connect(r, c, 0, 0)
    m.connect(c, g)
    m.connect(g, c, 0, 1)
    return m, [("R", 0, "input"), ("G", 0, "output"), ("C", 0, "output")], None, {}


def _pendulum():
    m = Model()
    u = m.add("Inport", "U")
    s = m.add("Sum", "S", signs="+--")
    w = m.add("Integrator", "W", x0=0.1)
    th = m.add("Integrator", "TH", x0=0.2)
    f = m.add("Fcn", "F", expr="9.81*sin(u)")
    c = m.add("Gain", "C", K=0.4)
    o = m.add("Outport", "O")
    m.connect(u, s, 0, 0)
    m.connect(f, s, 0, 1)
    m.connect(c, s, 0, 2)
    m.connect(s, w)
    m.connect(w, th)
    m.connect(w, c)
    m.connect(th, f)
    m.connect(th, o)
    return m


# ============================================================================ MATLAB betiği
def _lin_script() -> str:
    parts = ["warning('off', 'all');"]
    for name, fn in LCASES.items():
        m, pts, inputs, opts = fn()
        lines, sl = build_lines(m, name)
        for k, (b, p, typ) in enumerate(pts):
            lines.append(f"io({k + 1}) = linio([mdl '/{sl[b]}'], {p + 1}, '{typ}');")
        ts = opts.get("Ts")
        rc = opts.get("rate_conversion", "zoh")
        lines.append(f"opt = linearizeOptions('SampleTime', {_num(-1 if ts is None else ts)}, "
                     f"'RateConversionMethod', '{rc}');")
        if inputs:
            lines.append("op = operpoint(mdl);")
            for port, v in inputs.items():
                lines.append(f"op.Inputs({port}).u = {_mat(v)};")
            lines.append("L = linearize(mdl, io, op, opt);")
        else:
            lines.append("L = linearize(mdl, io, opt);")
        lines.append(f"out.{name}_A = L.A; out.{name}_B = L.B; out.{name}_C = L.C; out.{name}_D = L.D;")
        lines.append(f"out.{name}_Ts = L.Ts; out.{name}_dims = [size(L.A, 1), size(L.B, 2), size(L.C, 1)];")
        lines.append("clear io; close_system(mdl, 0);")
        body = "\n".join("  " + ln for ln in lines)
        parts.append(f"try\n{body}\ncatch e\n  out.{name}_err = e.message;\n  try close_system('{name}', 0); catch; end\n"
                     "  clear io;\nend")
    # çalışma noktası (findop) + o noktada doğrusallaştırma: sarkaç
    m = _pendulum()
    lines, sl = build_lines(m, "trim_pend")
    lines += ["spec = operspec(mdl);", "spec.Outputs(1).y = 0.5;", "spec.Outputs(1).Known = true;",
              "fo = findopOptions('DisplayReport', 'off');", "[op, rep] = findop(mdl, spec, fo);",
              "out.trim_u = op.Inputs(1).u;", "out.trim_blocks = {op.States.Block};",
              "out.trim_x = [op.States.x];",
              f"io(1) = linio([mdl '/{sl['U']}'], 1, 'input'); io(2) = linio([mdl '/{sl['TH']}'], 1, 'output');",
              "L = linearize(mdl, io, op);",
              "out.trim_A = L.A; out.trim_B = L.B; out.trim_C = L.C; out.trim_D = L.D;",
              "clear io; close_system(mdl, 0);"]
    body = "\n".join("  " + ln for ln in lines)
    parts.append(f"try\n{body}\ncatch e\n  out.trim_err = e.message;\n  try close_system('trim_pend', 0); catch; end\n"
                 "end")
    return "\n".join(parts)


_KF = {"A": [[1.0, 0.1, 0.0], [0.0, 1.0, 0.1], [0.0, 0.0, 0.95]], "B": [[0.0], [0.005], [0.1]],
       "C": [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], "D": [[0.0], [0.0]], "G": [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
       "Q": [[0.02, 0.005], [0.005, 0.04]], "R": [[0.1, 0.01], [0.01, 0.2]], "N": [[0.01, 0.0], [0.0, 0.02]],
       "Ts": 0.1}


def _kf_script() -> str:
    k = _KF
    return "\n".join([
        f"A = {_mat(k['A'])}; B = {_mat(k['B'])}; C = {_mat(k['C'])}; D = {_mat(k['D'])}; G = {_mat(k['G'])};",
        f"Q = {_mat(k['Q'])}; R = {_mat(k['R'])}; N = {_mat(k['N'])}; Ts = {_num(k['Ts'])};",
        "sys = ss(A, [B G], C, [D zeros(2, 2)], Ts);",
        "[kest, L, P, M] = kalman(sys, Q, R, N, 1:2, 1);",
        "out.L = L; out.P = P; out.M = M;",
    ])


_REF: dict[str, dict] = {}


def _ref(key: str, script_fn) -> dict:
    if key not in _REF:
        try:
            _REF[key] = fixture(key, script_fn())
        except OracleUnavailable as exc:
            pytest.skip(f"MATLAB referansı yok: {exc}")
    return _REF[key]


def _mat2(v: Any, r: int, c: int) -> np.ndarray:
    a = np.asarray(v, dtype=float)
    if r == 0 or c == 0:
        return np.zeros((r, c))
    return a.reshape(r, c)


def _fr(A, B, C, D, Ts, w):
    out = []
    n = A.shape[0]
    for wk in w:
        s = np.exp(1j * wk * Ts) if Ts > 0 else 1j * wk
        out.append(C @ np.linalg.solve(s * np.eye(n) - A, B) + D if n else D.astype(complex))
    return np.array(out)


# ============================================================================ testler
@pytest.mark.parametrize("name", list(LCASES))
def test_linearize_vs_matlab(name):
    ref = _ref("control_diagram_linearize", _lin_script)
    assert ref.get(f"{name}_err") is None, f"MATLAB hatası: {ref.get(name + '_err')}"
    nx, nu, ny = (int(v) for v in np.atleast_1d(ref[f"{name}_dims"]))
    A = _mat2(ref[f"{name}_A"], nx, nx)
    B = _mat2(ref[f"{name}_B"], nx, nu)
    C = _mat2(ref[f"{name}_C"], ny, nx)
    D = _mat2(ref[f"{name}_D"], ny, nu)
    Ts = float(ref[f"{name}_Ts"])
    m, pts, inputs, opts = LCASES[name]()
    op = OperatingPoint(0.0, inputs={p: np.atleast_1d(float(v)) for p, v in inputs.items()}) if inputs else None
    L = linearize(m, [linio(b, p, typ) for b, p, typ in pts], op, **opts)
    assert L.Ts == pytest.approx(Ts, abs=1e-15)
    assert L.D.shape == D.shape
    nyq = math.pi / Ts if Ts > 0 else 100.0
    w = np.concatenate([[1e-3], np.logspace(-2, math.log10(0.95 * nyq), 25)])
    H1 = _fr(L.A, L.B, L.C, L.D, L.Ts, w)
    H2 = _fr(A, B, C, D, Ts, w)
    scale = max(1.0, float(np.max(np.abs(H2))))
    np.testing.assert_allclose(H1, H2, rtol=0, atol=1e-8 * scale)
    # kutuplar (kontrol edilebilir ve gözlenebilir kısım; durum sayısı aynıysa doğrudan)
    if L.A.shape == A.shape and nx:
        np.testing.assert_allclose(np.sort_complex(np.linalg.eigvals(L.A)), np.sort_complex(np.linalg.eigvals(A)),
                                   rtol=1e-7, atol=1e-9)


def test_findop_and_linearize_at_op_vs_matlab():
    ref = _ref("control_diagram_linearize", _lin_script)
    assert ref.get("trim_err") is None, f"MATLAB hatası: {ref.get('trim_err')}"
    m = _pendulum()
    sp = operspec(m)
    sp.outputs[1] = {"y": 0.5, "known": True}
    res = findop(m, sp)
    assert res.success, res.message
    assert res.op.inputs[1][0] == pytest.approx(float(ref["trim_u"]), rel=1e-6)
    # MATLAB durum sırası blok yollarıyla verilir
    _, sl = build_lines(m, "trim_pend")
    inv = {f"trim_pend/{v}": k for k, v in sl.items()}
    xm = np.atleast_1d(np.asarray(ref["trim_x"], dtype=float))
    for blk, xv in zip(ref["trim_blocks"], xm):
        assert res.op.x[inv[blk]][0] == pytest.approx(xv, abs=1e-6)
    L = linearize(m, [linio("U", 0, "input"), linio("TH", 0, "output")], res.op)
    A = _mat2(ref["trim_A"], 2, 2)
    B = _mat2(ref["trim_B"], 2, 1)
    C = _mat2(ref["trim_C"], 1, 2)
    D = _mat2(ref["trim_D"], 1, 1)
    w = np.logspace(-2, 2, 20)
    np.testing.assert_allclose(_fr(L.A, L.B, L.C, L.D, 0.0, w), _fr(A, B, C, D, 0.0, w), rtol=1e-6)
    np.testing.assert_allclose(np.sort_complex(np.linalg.eigvals(L.A)), np.sort_complex(np.linalg.eigvals(A)),
                               rtol=1e-6)


def test_kalman_gains_vs_matlab():
    ref = _ref("control_diagram_kalman", _kf_script)
    k = _KF
    kf = create_block("KalmanFilter", "KF", A=k["A"], B=k["B"], C=k["C"], D=k["D"], G=k["G"], Q=k["Q"], R=k["R"],
                      N=k["N"], Ts=k["Ts"])
    kf.reset()
    g = kf.gains()
    np.testing.assert_allclose(g["P"], np.asarray(ref["P"], dtype=float), rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(g["L"], np.asarray(ref["L"], dtype=float).reshape(3, 2), rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(g["M"], np.asarray(ref["M"], dtype=float).reshape(3, 2), rtol=1e-9, atol=1e-12)
