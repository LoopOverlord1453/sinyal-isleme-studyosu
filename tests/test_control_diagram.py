"""Blok diyagram motoru: model / doğrulama / dosya / motor / doğrusallaştırma / trim / gerçek veri birim testleri.

MATLAB gerektirmez; doğruluk analitik çözümlerle (matris üsteli ile tam ayrıklaştırma, kapalı biçimli cevaplar,
scipy lfilter / lsim) karşılaştırılır. Simulink karşılaştırmaları: test_control_diagram_simulink.py, MATLAB
doğrusallaştırma / findop / kalman karşılaştırmaları: test_control_diagram_linearize.py.
"""
from __future__ import annotations

import json
import math
import sys
import threading
import warnings
from pathlib import Path

import numpy as np
import pytest
import scipy.linalg as sla
import scipy.signal as ss

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spt.control.diagram import (Block, LinearSystem, Model, OperSpec, Param, block_catalog, compile_model,  # noqa
                                 create_block, findop, fit_percent, linearize, linio, operspec, p_ts, register_block,
                                 register_lazy, simulate, simulate_with_data)
from spt.control.diagram.blocks.plants import ContinuousPlant, DiscreteController  # noqa: E402
from spt.control.diagram.expr import Expression, parse_numeric  # noqa: E402
from spt.control.errors import ControlError, ControlWarning  # noqa: E402


# ============================================================================ yardımcılar
def unity_loop(ctrl_type="Gain", ctrl=None, plant=None, ref="Step", **ref_kw):
    m = Model("döngü")
    r = m.add(ref, "R", **ref_kw)
    s = m.add("Sum", "E", signs="+-")
    c = m.add(ctrl_type, "C", **(ctrl or {"K": 2.0}))
    g = m.add("TransferFcn", "G", **(plant or {"num": [1.0], "den": [1.0, 1.0]}))
    m.connect(r, s)
    m.connect(s, c, name="e")
    m.connect(c, g, name="u")
    m.connect(g, s, dst_port=1)
    m.log_signal(g, name="y")
    return m


def ss_of_tf(num, den):
    A, B, C, D = ss.tf2ss(num, den)
    return A, B, C, D


def exact_step_cl(num_ol, den_ol, t):
    """Birim geri beslemeli kapalı çevrimin birim basamak cevabı (matris üsteliyle, tam)."""
    num_cl = np.polyadd(np.zeros(len(den_ol)), num_ol)
    den_cl = np.polyadd(den_ol, num_cl)
    A, B, C, D = ss.tf2ss(num_ol if False else np.atleast_1d(num_ol), den_cl)
    n = A.shape[0]
    y = np.empty(t.size)
    M = np.zeros((n + 1, n + 1))
    M[:n, :n] = A
    M[:n, n:] = B
    for k, tk in enumerate(t):
        E = sla.expm(M * tk)
        x = E[:n, n:].ravel()
        y[k] = (C @ x).item() + D.item()
    return y


# ============================================================================ ifade değerlendirici
def test_expression_matlab_syntax_and_safety():
    e = Expression("u(1)^2 + 2*sin(u(2)) - (u1 ~= 0) + (t > 1 && u(2) < 3)", scalars=("t",), vectors={"u": 2})
    env = {"u": np.array([3.0, 0.5]), "t": 2.0}
    assert e(env) == pytest.approx(9 + 2 * math.sin(0.5) - 1 + 1, abs=1e-15)
    assert Expression("mod(-1, 3) + rem(-1, 3) + round(-2.5)")({}) == pytest.approx(2 - 1 - 3)
    for bad in ["__import__('os')", "u.real", "[1, 2]", "open('x')", "lambda: 1", "u[0]", "x", "'a'", "2**10000"]:
        with pytest.raises(ControlError):
            e2 = Expression(bad, scalars=("t",), vectors={"u": 2})
            e2({"u": np.zeros(2), "t": 0.0})
    with pytest.raises(ControlError, match="sıfıra"):
        Expression("1/u", vectors={"u": 1})({"u": np.zeros(1)})
    with pytest.raises(ControlError):
        Expression("sqrt(u)", vectors={"u": 1})({"u": np.array([-1.0])})
    with pytest.raises(ControlError):
        Expression("u^0.5", vectors={"u": 1})({"u": np.array([-8.0])})    # karmaşık sonuç asla
    ex = Expression("u(3)", vectors={"u": None})
    with pytest.raises(ControlError, match="genişliği 2"):
        ex.check_widths({"u": 2}, "F")


def test_parse_numeric():
    np.testing.assert_array_equal(parse_numeric("[1 2; 3 4]"), [[1, 2], [3, 4]])
    np.testing.assert_allclose(parse_numeric("[0, 2*pi, -inf]"), [[0, 2 * math.pi, -math.inf]])
    with pytest.raises(ControlError):
        parse_numeric("[1 2; 3]")


# ============================================================================ model ve doğrulama
def test_model_api_names_and_errors():
    m = Model()
    a = m.add("Gain")
    b = m.add("Gain")
    assert (a.name, b.name) == ("Gain", "Gain1")
    with pytest.raises(ControlError, match="zaten var"):
        m.add("Gain", "Gain")
    with pytest.raises(ControlError, match="bilinmeyen parametre"):
        m.add("Gain", "X", KK=2)
    with pytest.raises(ControlError, match="Kazanç K"):
        m.add("Gain", "Y", K="abc")
    s = m.add("Step", "S")
    m.connect(s, a)
    with pytest.raises(ControlError, match="zaten 'S'"):
        m.connect(s, a)
    with pytest.raises(ControlError, match="2. girişi yok"):
        m.connect(s, b, dst_port=1)
    m.rename(a, "K1")
    assert m.source_of("K1") == ("S", 0)
    m.remove("S")
    assert m.source_of("K1") is None


def test_validation_messages():
    m = Model()
    s = m.add("Sum", "Toplam", signs="++")
    g = m.add("Gain", "G")
    m.connect(s, g)
    rep = m.validate()
    assert not rep.ok
    msgs = " ".join(i.message for i in rep.errors)
    assert "'Toplam' bloğunun 1. girişi bağlanmamış" in msgs and "2. girişi" in msgs
    assert any("'G' bloğunun 1. çıkışı bağlanmamış" in i.message for i in rep.warnings)


def test_algebraic_loop_detected_with_block_names():
    m = Model()
    r = m.add("Constant", "R")
    s = m.add("Sum", "Topla", signs="+-")
    k = m.add("Gain", "Kazanc", K=2.0)
    k2 = m.add("Gain", "Kazanc2", K=0.5)
    m.connect(r, s)
    m.connect(s, k)
    m.connect(k, k2)
    m.connect(k2, s, dst_port=1)
    with pytest.raises(ControlError) as ei:
        compile_model(m)
    msg = str(ei.value)
    assert "Cebirsel döngü" in msg and "'Topla'" in msg and "'Kazanc'" in msg and "'Kazanc2'" in msg
    assert "(3 blok)" in msg
    # Bellek bloğu döngüyü kırar
    m.disconnect(s, 1)
    mem = m.add("Memory", "Bellek")
    m.connect(k2, mem)
    m.connect(mem, s, dst_port=1)
    compile_model(m)


def test_sample_time_errors():
    m = Model()
    s = m.add("SineWave", "S")
    a = m.add("ZeroOrderHold", "A", Ts=0.1)
    b = m.add("ZeroOrderHold", "B", Ts=0.15)
    t = m.add("Sum", "T", signs="++")
    m.connect(s, a)
    m.connect(s, b)
    m.connect(a, t)
    m.connect(b, t, dst_port=1)
    with pytest.raises(ControlError, match="çatışması.*'T'"):
        compile_model(m)
    m2 = Model()
    s2 = m2.add("SineWave", "S")
    ud = m2.add("UnitDelay", "Gecikme")
    m2.connect(s2, ud)
    with pytest.raises(ControlError, match="'Gecikme'.*ayrık bir blok"):
        compile_model(m2)
    # sabit adım tam katı değil
    m3 = Model()
    s3 = m3.add("SineWave", "S")
    z = m3.add("ZeroOrderHold", "Z", Ts=0.03)
    m3.connect(s3, z)
    with pytest.raises(ControlError, match="tam katı"):
        simulate(m3, 1.0, solver="ode4", step=0.02)
    # tam katta miras: en hızlı hız
    a.set_params(Ts=0.1)
    b.set_params(Ts=0.2)
    cm = compile_model(m)
    assert cm.blocks[cm.index["T"]].ts.period == pytest.approx(0.1)


def test_width_propagation_vectors():
    m = Model()
    c = m.add("Constant", "C", value=[1.0, 2.0, 3.0])
    k = m.add("Gain", "K", K=[[1, 0, 0], [0, 2, 0]], mode="matrix")
    mx = m.add("Mux", "M", inputs=2)
    one = m.add("Constant", "One", value=5.0)
    s = m.add("Sum", "S", signs="+-")
    dm = m.add("Demux", "D", outputs=3)
    for blk in ("T1", "T2", "T3"):
        m.add("Terminator", blk)
    m.connect(c, k)
    m.connect(k, mx)
    m.connect(one, mx, dst_port=1)
    m.connect(mx, s)
    m.connect(one, s, dst_port=1)
    m.connect(s, dm)
    for i, blk in enumerate(("T1", "T2", "T3")):
        m.connect(dm, blk, src_port=i)
    m.log_signal(s)
    res = simulate(m, 0.1, solver="ode4", step=0.1)
    np.testing.assert_allclose(res.signals["S"][-1], [1 - 5, 4 - 5, 5 - 5])
    k.set_params(K=[[1, 0], [0, 1]])
    with pytest.raises(ControlError, match="giriş genişliği 2"):
        compile_model(m)


def test_signal_name_unique_and_keys():
    m = unity_loop()
    with pytest.raises(ControlError, match="zaten"):
        m.name_signal("C", 0, "e")
    res = simulate(m, 0.5, solver="ode4", step=0.01, log_all=True)
    assert {"y", "e", "u", "R"} <= set(res.signals)


# ============================================================================ dosya
def test_sptd_roundtrip(tmp_path):
    m = unity_loop("PIDController", {"P": 2.0, "I": 1.0, "D": 0.1, "limit_output": True, "upper": 3.0,
                                     "anti_windup": "clamping"})
    m.blocks["C"].meta = {"pos": [120, 40], "size": [60, 40], "flip": False}
    m.connections[0].meta = {"points": [[10, 20], [30, 20]]}
    m.settings["solver"] = "ode45"
    sub = m.add_subsystem("Alt")
    sub.system.add("Inport", "In1")
    sub.system.add("Gain", "G", K=3.0)
    sub.system.add("Outport", "Out1")
    sub.system.connect("In1", "G")
    sub.system.connect("G", "Out1")
    m.connect("G", sub)
    m.add("Scope", "Skop")
    m.connect(sub, "Skop")
    p = m.save(tmp_path / "model")
    assert p.suffix == ".sptd"
    raw = json.loads(p.read_text(encoding="utf-8"))
    assert raw["format"] == "spt-diagram" and raw["version"] == 1
    assert raw["blocks"][2]["params"]["lower"] == "-inf"       # katı JSON
    m2 = Model.load(p)
    assert m2.to_dict() == m.to_dict()
    r1 = simulate(m, 3.0, solver="ode4", step=0.01)
    r2 = simulate(m2, 3.0, solver="ode4", step=0.01)
    np.testing.assert_array_equal(r1.signals["y"], r2.signals["y"])
    np.testing.assert_array_equal(r1.scopes["Skop"].data[0], r2.scopes["Skop"].data[0])
    raw["version"] = 99
    (tmp_path / "yeni.sptd").write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ControlError, match="daha yeni"):
        Model.load(tmp_path / "yeni.sptd")


def test_subsystem_equals_flat():
    flat = unity_loop(plant={"num": [1.0], "den": [1.0, 2.0, 1.0]})
    m = Model()
    r = m.add("Step", "R")
    s = m.add("Sum", "E", signs="+-")
    sub = m.add_subsystem("Kontrol")
    inner = sub.system
    inner.add("Inport", "e", port=1)
    inner.add("Gain", "C", K=2.0)
    inner.add("Outport", "u", port=1)
    inner.connect("e", "C")
    inner.connect("C", "u")
    g = m.add("TransferFcn", "G", num=[1.0], den=[1.0, 2.0, 1.0])
    m.connect(r, s)
    m.connect(s, sub)
    m.connect(sub, g)
    m.connect(g, s, dst_port=1)
    m.log_signal(g, name="y")
    a = simulate(flat, 4.0, solver="ode4", step=0.01).signals["y"]
    b = simulate(m, 4.0, solver="ode4", step=0.01).signals["y"]
    np.testing.assert_array_equal(a, b)
    cm = compile_model(m)
    assert "Kontrol/C" in cm.index


# ============================================================================ motor: analitik karşılaştırmalar
@pytest.mark.parametrize("solver,opts,tol", [("ode4", {"step": 0.005}, 1e-10), ("ode45", {"rtol": 1e-11,
                                                                                            "atol": 1e-13}, 1e-9)])
def test_closed_loop_lti_exact(solver, opts, tol):
    # G = (s + 3)/(s³ + 3s² + 4s + 2), C = 2  → kapalı çevrim tam cevabı matris üsteliyle
    m = unity_loop(plant={"num": [1.0, 3.0], "den": [1.0, 3.0, 4.0, 2.0]}, step_time=0.0)
    t_out = np.linspace(0, 6, 61)
    res = simulate(m, 6.0, solver=solver, output_times=t_out, **opts)
    y_exact = exact_step_cl(np.array([2.0, 6.0]), np.array([1.0, 3.0, 4.0, 2.0]), t_out)
    np.testing.assert_allclose(res.signals["y"][:, 0], y_exact, atol=tol)


def test_sampled_data_exact_recurrence():
    """Ayrık PI (Ts) + ZOH'lu sürekli bitki: örnekleme anlarında tam ayrık özyineleme ile birebir (ode45 tam
    örnekleme anlarına basar)."""
    Ts = 0.1
    m = Model()
    r = m.add("Constant", "R", value=1.0)
    s = m.add("Sum", "E", signs="+-")
    c = m.add("DiscreteTransferFcn", "C", num=[1.2, -0.9], den=[1.0, -1.0], Ts=Ts)
    g = m.add("TransferFcn", "G", num=[2.0], den=[1.0, 1.5, 0.5])
    m.connect(r, s)
    m.connect(s, c)
    m.connect(c, g)
    m.connect(g, s, dst_port=1)
    m.log_signal(g, name="y")
    m.log_signal(c, name="u")
    tk = np.arange(0, 41) * Ts
    res = simulate(m, 4.0, solver="ode45", rtol=1e-12, atol=1e-14, output_times=tk)
    A, B, C, D = ss.tf2ss([2.0], [1.0, 1.5, 0.5])
    Ad, Bd, _, _, _ = ss.cont2discrete((A, B, C, D), Ts, "zoh")
    x = np.zeros((2, 1))
    xc = 0.0
    yk, uk = [], []
    for k in range(tk.size):
        y = (C @ x).item()
        e = 1.0 - y
        u = 1.2 * e + xc            # DTF (1.2 z − 0.9)/(z − 1) denetlenebilir kanonik: u = 1.2 e + s
        xc = xc + 0.3 * e           # s[k+1] = s[k] + (−0.9 + 1.2)·e
        yk.append(y)
        uk.append(u)
        x = Ad @ x + Bd * u
    np.testing.assert_allclose(res.signals["y"][:, 0], yk, atol=2e-11)
    np.testing.assert_allclose(res.signals["u"][:, 0], uk, atol=2e-11)


def test_discrete_filters_vs_lfilter():
    rng = np.random.default_rng(3)
    t = np.arange(0, 200) * 0.01
    u = rng.standard_normal(t.size)
    m = Model()
    d = m.add("DataSource", "D", time=t, values=u, interp="zoh")
    f1 = m.add("DiscreteFilter", "F1", num=[0.2, 0.3, 0.1], den=[1.0, -0.5, 0.06], Ts=0.01)
    f2 = m.add("DiscreteTransferFcn", "F2", num=[0.4, 0.1], den=[1.0, -0.3, 0.02], Ts=0.01)
    m.connect(d, f1)
    m.connect(d, f2)
    res = simulate(m, t[-1], solver="ode4", step=0.01, log=["F1", "F2"])
    np.testing.assert_allclose(res.signals["F1"][:, 0], ss.lfilter([0.2, 0.3, 0.1], [1.0, -0.5, 0.06], u),
                               atol=1e-13)
    # z'nin azalan kuvvetleri: (0.4 z + 0.1)/(z² − 0.3 z + 0.02) = (0.4 z⁻¹ + 0.1 z⁻²)/(1 − 0.3 z⁻¹ + 0.02 z⁻²)
    np.testing.assert_allclose(res.signals["F2"][:, 0], ss.lfilter([0, 0.4, 0.1], [1.0, -0.3, 0.02], u), atol=1e-13)


def test_zero_crossing_saturation_kink_ode45():
    """Rampa → doyum (0.7) → integratör: tam çözüm t²/2 (t ≤ 0.7), 0.245 + 0.7(t − 0.7) sonra."""
    m = Model()
    r = m.add("Ramp", "R", slope=1.0)
    s = m.add("Saturation", "S", upper=0.7, lower=-1.0)
    i = m.add("Integrator", "I")
    m.connect(r, s)
    m.connect(s, i)
    m.log_signal(i)
    t = np.linspace(0, 2, 21)
    res = simulate(m, 2.0, solver="ode45", rtol=1e-8, atol=1e-12, output_times=t)
    exact = np.where(t <= 0.7, t ** 2 / 2, 0.245 + 0.7 * (t - 0.7))
    np.testing.assert_allclose(res.signals["I"][:, 0], exact, atol=1e-12)
    assert res.stats["zc_events"] >= 1


def test_relay_event_time_located():
    """İntegratör (eğim 1) → röle (açma 0.5): röle tam t = 0.5'te açılır; ikinci integratör (t − 0.5)."""
    m = Model()
    c = m.add("Constant", "C", value=1.0)
    i1 = m.add("Integrator", "I1")
    rel = m.add("Relay", "Rel", on_point=0.5, off_point=0.2, on_value=1.0, off_value=0.0)
    i2 = m.add("Integrator", "I2")
    m.connect(c, i1)
    m.connect(i1, rel)
    m.connect(rel, i2)
    m.log_signal(i2)
    res = simulate(m, 1.0, solver="ode45", rtol=1e-6, output_times=[1.0])
    assert res.signals["I2"][-1, 0] == pytest.approx(0.5, abs=1e-12)


def test_integrator_limits_and_transport_delay_analytic():
    m = Model()
    c = m.add("Step", "S", step_time=0.2)
    td = m.add("TransportDelay", "TD", delay=0.3)
    i = m.add("Integrator", "I", upper=0.25)
    m.connect(c, td)
    m.connect(td, i)
    m.log_signal(i)
    t = np.linspace(0, 1.5, 16)
    res = simulate(m, 1.5, solver="ode4", step=0.01, output_times=t)
    # gecikme tamponu ana adım örnekleri arasında doğrusal ara değerler (Simulink gibi): 0.19 → 0.2'deki basamak
    # 0.49 → 0.5 arasında rampa olur; integrali t ≥ 0.5 için (t − 0.5) + 0.005, sonra 0.25'te doyum
    exact = np.clip(np.where(t <= 0.49, 0.0, np.where(t <= 0.5, (t - 0.49) ** 2 / 0.02, t - 0.495)), 0, 0.25)
    np.testing.assert_allclose(res.signals["I"][:, 0], exact, atol=1e-12)


def test_pid_vector_equals_scalars():
    def build(width2: bool):
        m = Model()
        r = m.add("Constant", "R", value=[1.0, -0.5] if width2 else 1.0)
        c = m.add("PIDController", "C", P=2.0, I=1.5, D=0.2, N=25.0, limit_output=True, upper=1.2, lower=-0.8,
                  anti_windup="clamping")
        g = m.add("Integrator", "G", x0=[0.0, 0.0] if width2 else 0.0)
        s = m.add("Sum", "E", signs="+-")
        m.connect(r, s)
        m.connect(s, c)
        m.connect(c, g)
        m.connect(g, s, dst_port=1)
        m.log_signal(g)
        return m
    yv = simulate(build(True), 3.0, solver="ode4", step=0.01).signals["G"]
    m1 = build(False)
    y1 = simulate(m1, 3.0, solver="ode4", step=0.01).signals["G"][:, 0]
    m1.blocks["R"].set_params(value=-0.5)
    y2 = simulate(m1, 3.0, solver="ode4", step=0.01).signals["G"][:, 0]
    np.testing.assert_allclose(yv[:, 0], y1, atol=1e-14)
    np.testing.assert_allclose(yv[:, 1], y2, atol=1e-14)


def test_fixed_step_grid_and_partial_last_step():
    m = unity_loop()
    res = simulate(m, 1.005, solver="ode4", step=0.01)
    assert res.t[-1] == pytest.approx(1.005) and res.t.size == 102
    res2 = simulate(m, 1.0, solver="ode4", step=0.01, decimation=10)
    np.testing.assert_allclose(res2.t, np.linspace(0, 1, 11), atol=1e-12)


def test_progress_and_cancel():
    m = unity_loop()
    ev = threading.Event()
    seen = []

    def prog(frac, t):
        seen.append(frac)
        ev.set()
    res = simulate(m, 2000.0, solver="ode4", step=0.001, progress=prog, cancel=ev)
    assert res.cancelled and res.t[-1] < 2000.0


def test_noise_deterministic():
    def run(seed):
        m = Model()
        n = m.add("BandLimitedWhiteNoise", "N", power=0.2, seed=seed, Ts=0.01)
        g = m.add("TransferFcn", "G", num=[1.0], den=[0.1, 1.0])
        m.connect(n, g)
        m.log_signal(g)
        m.log_signal(n)
        return simulate(m, 5.0, solver="ode4", step=0.01)
    a, b, c = run(1), run(1), run(2)
    np.testing.assert_array_equal(a.signals["G"], b.signals["G"])
    assert not np.allclose(a.signals["G"], c.signals["G"])
    # varyans = güç / Ts
    assert np.var(run(5).signals["N"][:-1]) == pytest.approx(0.2 / 0.01, rel=0.1)


def test_unstable_reports_blowup():
    m = Model()
    c = m.add("Constant", "C", value=0.0)
    p = m.add("StateSpace", "P", A=[[50.0]], B=[[1.0]], C=[[1.0]], D=[[0.0]], x0=[1.0])
    m.connect(c, p)
    with pytest.raises(ControlError, match="ıraksadı"):
        simulate(m, 100.0, solver="ode4", step=0.1)


# ============================================================================ doğrusallaştırma (analitik)
def _tf_eval(num, den, s):
    return np.polyval(num, s) / np.polyval(den, s)


def test_linearize_closed_and_loop_transfer_exact():
    plant = {"num": [1.0, 2.0], "den": [1.0, 3.0, 4.0, 1.0]}
    m = unity_loop("PIDController", {"P": 2.0, "I": 1.5, "D": 0.3, "N": 40.0}, plant=plant)
    w = np.logspace(-2, 2, 15)
    s = 1j * w
    C = 2.0 + 1.5 / s + 0.3 * 40.0 / (1 + 40.0 / s)
    G = _tf_eval(plant["num"], plant["den"], s)
    L1 = linearize(m, [linio("R", 0, "input"), linio("G", 0, "output")])
    np.testing.assert_allclose(L1.freqresp(w)[0, 0], C * G / (1 + C * G), rtol=1e-10)
    L2 = linearize(m, [linio("C", 0, "looptransfer")])
    np.testing.assert_allclose(L2.freqresp(w)[0, 0], -C * G, rtol=1e-10)
    L3 = linearize(m, [linio("e", 0, "inout") if False else linio("E", 0, "inout")])
    np.testing.assert_allclose(L3.freqresp(w)[0, 0], 1 / (1 + C * G), rtol=1e-10)
    assert L1.Ts == 0.0 and L1.A.shape == (5, 5)


def test_linearize_sampled_data_exact():
    """Ayrık denetleyici + sürekli bitki: hedef Ts = 0.1; sürekli kısım bir bütün olarak ZOH ile ayrıklaşır."""
    Ts = 0.1
    m = Model()
    r = m.add("Constant", "R", value=0.0)
    s = m.add("Sum", "E", signs="+-")
    c = m.add("DiscreteTransferFcn", "C", num=[1.2, -0.9], den=[1.0, -1.0], Ts=Ts)
    g1 = m.add("TransferFcn", "G1", num=[1.0], den=[1.0, 1.0])
    g2 = m.add("TransferFcn", "G2", num=[2.0], den=[1.0, 0.5])
    m.connect(r, s)
    m.connect(s, c)
    m.connect(c, g1)
    m.connect(g1, g2)
    m.connect(g2, s, dst_port=1)
    L = linearize(m, [linio("C", 0, "looptransfer")])
    assert L.Ts == Ts
    num_g, den_g = np.polymul([1.0], [2.0]), np.polymul([1.0, 1.0], [1.0, 0.5])
    Gd = ss.cont2discrete(ss.tf2ss(num_g, den_g), Ts, "zoh")
    w = np.array([0.1, 1.0, 5.0, 20.0])
    z = np.exp(1j * w * Ts)
    Gz = np.array([(Gd[2] @ np.linalg.solve(zk * np.eye(2) - Gd[0], Gd[1]) + Gd[3]).item() for zk in z])
    Cz = (1.2 * z - 0.9) / (z - 1.0)
    np.testing.assert_allclose(L.freqresp(w)[0, 0], -Cz * Gz, rtol=1e-10)
    Lc = linearize(m, [linio("C", 0, "looptransfer")], Ts=0.0)
    assert Lc.Ts == 0.0


def test_linearize_vector_signals_and_pruning():
    m = Model()
    i = m.add("Inport", "In", width=2)
    k = m.add("Gain", "K", K=[[1.0, 2.0], [3.0, 4.0]], mode="matrix")
    g = m.add("StateSpace", "P", A=[[-1.0, 0.0], [0.0, -2.0]], B=[[1.0, 0.0], [0.0, 1.0]],
              C=[[1.0, 0.0], [0.0, 1.0]], D=[[0.0, 0.0], [0.0, 0.0]], x0=[0.0, 0.0])
    extra = m.add("TransferFcn", "Yan", num=[1.0], den=[1.0, 7.0])     # yol dışı
    o = m.add("Outport", "Out")
    m.connect(i, k)
    m.connect(k, g)
    m.connect(g, o)
    m.add("Sum", "S2", signs="+")
    m.connect(g, "S2")
    m.connect("S2", extra)
    m.add("Terminator", "T")
    m.connect(extra, "T")
    L = linearize(m, [linio("In", 0, "input"), linio("P", 0, "output")])
    assert L.A.shape == (2, 2) and L.B.shape == (2, 2)
    np.testing.assert_allclose(L.dcgain(), np.diag([1.0, 0.5]) @ np.array([[1.0, 2.0], [3.0, 4.0]]), atol=1e-12)


def test_linearize_nonlinear_numeric_matches_analytic():
    """Fcn bloğu (sayısal merkezi fark) — analitik türevle."""
    m = Model()
    i = m.add("Inport", "In")
    f = m.add("Fcn", "F", expr="u^3 + exp(0.5*u)")
    o = m.add("Outport", "O")
    m.connect(i, f)
    m.connect(f, o)
    from spt.control.diagram.engine import OperatingPoint
    op = OperatingPoint(0.0, inputs={1: np.array([0.7])})
    L = linearize(m, [linio("In", 0, "input"), linio("F", 0, "output")], op)
    assert L.D[0, 0] == pytest.approx(3 * 0.49 + 0.5 * math.exp(0.35), rel=1e-9)


# ============================================================================ trim
def test_findop_pendulum_analytic():
    """Sönümlü sarkaç (ifade bloğu): θ = 0.5 rad'da kalması için gereken tork u = g·sin(θ)/... analitik."""
    m = Model()
    u = m.add("Inport", "Tork")
    p = m.add("ExprStateSpace", "Sarkac", f="x(2); -9.81*sin(x(1)) - 0.4*x(2) + 2*u(1)", g="x(1)", x0=[0.0, 0.0])
    o = m.add("Outport", "Aci")
    m.connect(u, p)
    m.connect(p, o)
    sp = operspec(m)
    sp.outputs[1] = {"y": 0.5, "known": True}
    sp.states["Sarkac"]["x"] = [0.3, 0.1]
    res = findop(m, sp)
    assert res.success, res.message
    assert res.op.inputs[1][0] == pytest.approx(9.81 * math.sin(0.5) / 2, rel=1e-10)
    np.testing.assert_allclose(res.op.x["Sarkac"], [0.5, 0.0], atol=1e-10)
    L = linearize(m, [linio("Tork", 0, "input"), linio("Sarkac", 0, "output")], res.op)
    A_exact = np.array([[0, 1], [-9.81 * math.cos(0.5), -0.4]])
    np.testing.assert_allclose(L.A, A_exact, atol=1e-7)
    # çalışma noktasından benzetim kararlı hâlde kalır
    sim = simulate(m, 2.0, solver="ode45", rtol=1e-10, atol=1e-12, initial=res.op,
                   inputs={1: res.op.inputs[1]})
    assert sim.outputs[1][-1, 0] == pytest.approx(0.5, abs=1e-9)


def test_findop_with_delay_and_discrete_steady():
    m = Model()
    u = m.add("Inport", "U")
    td = m.add("TransportDelay", "TD", delay=0.2)
    g = m.add("TransferFcn", "G", num=[3.0], den=[1.0, 2.0])
    ud = m.add("UnitDelay", "UD", Ts=0.1)
    o = m.add("Outport", "Y")
    m.connect(u, td)
    m.connect(td, g)
    m.connect(g, ud)
    m.connect(ud, o)
    sp = operspec(m)
    sp.outputs[1] = {"y": 3.0}
    res = findop(m, sp)
    assert res.success
    assert res.op.inputs[1][0] == pytest.approx(2.0, rel=1e-10)
    assert res.op.xd["UD"][0] == pytest.approx(3.0, rel=1e-10)


# ============================================================================ gerçek veri
def test_simulate_with_data_matches_lsim_and_fit():
    t = np.arange(0, 1001) * 0.01
    rng = np.random.default_rng(0)
    u = np.cumsum(rng.standard_normal(t.size)) * 0.05
    m = Model()
    d = m.add("DataSource", "Olcum", time=[0, 1], values=[0, 0])
    g = m.add("TransferFcn", "G", num=[4.0], den=[1.0, 1.2, 4.0])
    m.connect(d, g)
    m.log_signal(g, name="y")
    _, y_true, _ = ss.lsim(([4.0], [1.0, 1.2, 4.0]), u, t, interp=True)
    cmp = simulate_with_data(m, "Olcum", t, u, output="y", y_meas=y_true, solver="ode45", rtol=1e-10, atol=1e-12)
    assert cmp.fit[0] > 99.9999
    assert cmp.max_error[0] < 1e-7
    y_noisy = y_true + 0.01 * rng.standard_normal(t.size)
    cmp2 = simulate_with_data(m, "Olcum", t, u, output="y", y_meas=y_noisy, solver="ode4")
    expected = 100 * (1 - np.linalg.norm(y_noisy - cmp2.y_sim[:, 0]) / np.linalg.norm(y_noisy - y_noisy.mean()))
    assert cmp2.fit[0] == pytest.approx(expected)
    assert "uyum" in cmp2.summary()


def test_fit_percent_definition():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert fit_percent(y, y)[0] == 100.0
    assert fit_percent(y, np.full(4, 2.5))[0] == pytest.approx(0.0)


# ============================================================================ katalog ve eklenti sözleşmesi
def test_catalog_all_blocks_instantiate_and_json():
    cat = block_catalog()
    json.dumps(cat)
    types = {c["type"] for c in cat}
    for needed in ("PIDController", "PIDController2DOF", "TransportDelay", "KalmanFilter", "DataSource", "Relay",
                   "Subsystem", "ExprStateSpace", "Fcn", "Observer", "StateFeedback", "Notch", "LeadLag"):
        assert needed in types
    for c in cat:
        if not c["lazy"]:
            create_block(c["type"])
            assert c["display_name"] and c["category_label"]


def test_plugin_register_custom_block():
    @register_block
    class Kare(Block):
        type_name = "TestKare"
        display_name = "Kare (test)"
        category = "nonlinear"
        description = "y = k·u²"
        PARAMS = (Param("k", "float", 1.0, "Kazanç"), p_ts(-1.0))

        def output(self, t, x, xd, u):
            return [self.params["k"] * u[0] ** 2]

    m = Model()
    r = m.add("Ramp", "R")
    q = m.add("TestKare", "Q", k=3.0)
    i = m.add("Integrator", "I")
    m.connect(r, q)
    m.connect(q, i)
    m.log_signal(i)
    res = simulate(m, 1.0, solver="ode4", step=0.01)
    assert res.signals["I"][-1, 0] == pytest.approx(1.0, abs=1e-12)     # ∫3t² = t³
    L = linearize(m, [linio("R", 0, "input"), linio("Q", 0, "output")],
                  simulate(m, 0.5, solver="ode4", step=0.01))
    assert L.D[0, 0] == pytest.approx(2 * 3.0 * 0.5, rel=1e-8)
    register_lazy("TestYokModul", "spt.control.yok_boyle_modul:Sinif", display_name="Yok")
    with pytest.raises(ControlError, match="yüklenemedi"):
        create_block("TestYokModul")


def test_plugin_bases_continuous_plant_and_discrete_controller():
    class Tank(ContinuousPlant):
        type_name = "TestTank"
        display_name = "Tank (test)"
        PARAMS = (Param("a", "float", 0.5, "Çıkış katsayısı"),)

        def dims(self):
            return (1, 1, 1)

        def f(self, t, x, u):
            return np.array([u[0] - self.params["a"] * math.sqrt(max(x[0], 0.0))])

        def g(self, t, x, u):
            return x.copy()

    class SayacPI(DiscreteController):
        type_name = "TestPI"
        display_name = "PI (test)"
        PARAMS = (p_ts(0.1),)

        def port_widths_in(self):
            return [1]

        def output_width(self):
            return 1

        def initial_memory(self):
            return {"s": 0.0}

        def compute(self, t, u, mem):
            e = float(u[0][0])
            return np.array([2.0 * e + mem["s"]]), {"s": mem["s"] + 0.1 * e}

    register_block(Tank)
    register_block(SayacPI)
    m = Model()
    r = m.add("Constant", "R", value=1.0)
    s = m.add("Sum", "E", signs="+-")
    c = m.add("TestPI", "C")
    p = m.add("TestTank", "P")
    m.connect(r, s)
    m.connect(s, c)
    m.connect(c, p)
    m.connect(p, s, dst_port=1)
    m.log_signal(p)
    res = simulate(m, 30.0, solver="ode4", step=0.01)
    assert res.signals["P"][-1, 0] == pytest.approx(1.0, abs=1e-3)


def test_kalman_steady_state_gains_definition():
    A = np.array([[1.0, 0.1], [0.0, 1.0]])
    C = np.array([[1.0, 0.0]])
    Q = np.diag([1e-3, 1e-2])
    R = np.array([[0.1]])
    kf = create_block("KalmanFilter", "KF", A=A, B=[[0.005], [0.1]], C=C, D=[[0.0]], Q=Q, R=R, Ts=0.1)
    P = kf._Pss
    # DARE: P = A P A' + Q − A P C'(C P C' + R)⁻¹ C P A'
    S = C @ P @ C.T + R
    np.testing.assert_allclose(A @ P @ A.T + Q - A @ P @ C.T @ np.linalg.solve(S, C @ P @ A.T), P, atol=1e-12)


def test_time_varying_kalman_matches_manual_recursion():
    A = np.array([[1.0, 0.1], [0.0, 1.0]])
    B = np.array([[0.005], [0.1]])
    C = np.array([[1.0, 0.0]])
    Q = np.diag([1e-4, 1e-3])
    R = np.array([[0.05]])
    t = np.arange(0, 51) * 0.1
    rng = np.random.default_rng(1)
    u = np.sin(t)
    ymeas = np.cumsum(rng.standard_normal(t.size)) * 0.1
    m = Model()
    du = m.add("DataSource", "U", time=t, values=u, interp="zoh", Ts=0.1)
    dy = m.add("DataSource", "Y", time=t, values=ymeas, interp="zoh", Ts=0.1)
    kf = m.add("KalmanFilter", "KF", A=A, B=B, C=C, D=[[0.0]], Q=Q, R=R, x0=[0.2, -0.1], P0=[[1.0, 0.0], [0.0, 2.0]],
               steady_state=False, Ts=0.1)
    m.connect(du, kf, 0, 0)
    m.connect(dy, kf, 0, 1)
    m.log_signal(kf)
    res = simulate(m, t[-1], solver="ode4", step=0.1)
    x = np.array([0.2, -0.1])
    P = np.diag([1.0, 2.0])
    out = []
    for k in range(t.size):
        S = C @ P @ C.T + R
        M = P @ C.T @ np.linalg.inv(S)
        e = ymeas[k] - (C @ x).item()
        out.append(x + (M * e).ravel())
        L = A @ P @ C.T @ np.linalg.inv(S)
        x = A @ x + (B * u[k]).ravel() + (L * e).ravel()
        P = A @ P @ A.T + Q - L @ S @ L.T
    np.testing.assert_allclose(res.signals["KF"], np.array(out), atol=1e-12)


def test_warning_on_chattering_does_not_crash():
    m = Model()
    c = m.add("Constant", "R", value=0.0)
    s = m.add("Sum", "E", signs="+-")
    sg = m.add("Sign", "SG")
    i = m.add("Integrator", "I", x0=0.3)
    m.connect(c, s)
    m.connect(s, sg)
    m.connect(sg, i)
    m.connect(i, s, dst_port=1)
    m.log_signal(i)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        res = simulate(m, 1.0, solver="ode45", rtol=1e-4)
    assert abs(res.signals["I"][-1, 0]) < 1e-3
    assert any(issubclass(x.category, ControlWarning) for x in w)


def test_operspec_defaults_and_linear_system_api():
    m = unity_loop()
    sp = operspec(m)
    assert isinstance(sp, OperSpec) and "G" in sp.states
    L = linearize(m, [linio("R", 0, "input"), linio("G", 0, "output")])
    assert isinstance(L, LinearSystem)
    assert L.input_names == ["R"] and L.output_names == ["y"]
    np.testing.assert_allclose(L.poles(), [-3.0])


# ============================================================================ kompanzatörler ve durum geri beslemesi
def _single_block_lin(type_name, **params):
    m = Model()
    i = m.add("Inport", "In")
    b = m.add(type_name, "B", **params)
    o = m.add("Outport", "O")
    m.connect(i, b)
    m.connect(b, o)
    return linearize(m, [linio("In", 0, "input"), linio("B", 0, "output")])


def test_leadlag_and_tustin_prewarp_exact():
    w = np.array([0.3, 2.0, 7.0, 25.0])
    L = _single_block_lin("LeadLag", K=2.0, wz=1.5, wp=12.0)
    Hc = lambda s: 2.0 * (s / 1.5 + 1) / (s / 12.0 + 1)  # noqa: E731
    np.testing.assert_allclose(L.freqresp(w)[0, 0], Hc(1j * w), rtol=1e-12)
    T = 0.02
    Ld = _single_block_lin("LeadLag", K=2.0, wz=1.5, wp=12.0, Ts=T, method="tustin")
    assert Ld.Ts == T
    # Tustin: H_d(e^{jωT}) = H_c(j·(2/T)·tan(ωT/2))
    np.testing.assert_allclose(Ld.freqresp(w)[0, 0], Hc(1j * (2 / T) * np.tan(w * T / 2)), rtol=1e-10)


def test_notch_prewarped_depth_exact():
    T, w0 = 0.01, 60.0
    L = _single_block_lin("Notch", w0=w0, zeta_z=0.02, zeta_p=0.7, Ts=T)
    H = L.freqresp([w0])[0, 0, 0]
    assert abs(H) == pytest.approx(0.02 / 0.7, rel=1e-10)        # ön bükme: çentik derinliği ω0'da tam
    Lc = _single_block_lin("Notch", w0=w0, zeta_z=0.02, zeta_p=0.7)
    assert abs(Lc.freqresp([w0])[0, 0, 0]) == pytest.approx(0.02 / 0.7, rel=1e-12)


def test_feedforward_discrete_and_static():
    L = _single_block_lin("Feedforward", num=[0.5, 0.1], den=[1.0, -0.4], Ts=0.1)
    z = np.exp(1j * 3.0 * 0.1)
    assert L.freqresp([3.0])[0, 0, 0] == pytest.approx((0.5 * z + 0.1) / (z - 0.4), rel=1e-12)
    assert _single_block_lin("Feedforward", num=[3.0], den=[2.0], Ts=-1.0).D[0, 0] == pytest.approx(1.5)
    with pytest.raises(ControlError):
        create_block("Feedforward", num=[1.0, 1.0], den=[1.0, 2.0], Ts=-1.0)


def test_observer_state_feedback_separation():
    """Gözlemci + durum geri beslemesi kapalı çevriminin kutupları = eig(A − BK) ∪ eig(A − LC)."""
    A = np.array([[0.0, 1.0], [-2.0, -0.5]])
    B = np.array([[0.0], [1.0]])
    C = np.array([[1.0, 0.0]])
    K = np.array([[4.0, 2.5]])
    Lg = np.array([[9.0], [20.0]])
    m = Model()
    r = m.add("Constant", "R", value=0.0)
    p = m.add("StateSpace", "P", A=A, B=B, C=C, D=[[0.0]])
    ob = m.add("Observer", "Obs", A=A, B=B, C=C, D=[[0.0]], L=Lg)
    sf = m.add("StateFeedback", "SF", K=K, use_reference=True, Nr=1.0)
    m.connect(sf, p)
    m.connect(sf, ob, dst_port=0)
    m.connect(p, ob, dst_port=1)
    m.connect(ob, sf, dst_port=0)
    m.connect(r, sf, dst_port=1)
    Lsys = linearize(m, [linio("R", 0, "input"), linio("P", 0, "output")])
    expected = np.concatenate([np.linalg.eigvals(A - B @ K), np.linalg.eigvals(A - Lg @ C)])
    np.testing.assert_allclose(np.sort_complex(Lsys.poles()), np.sort_complex(expected), atol=1e-10)
    # ayrık gözlemci (öngörücü) benzetimi: kestirim hatası (A − LC)^k ile söner
    Ad = np.array([[1.0, 0.1], [0.0, 1.0]])
    Bd = np.array([[0.005], [0.1]])
    Ld = np.array([[0.9], [2.0]])
    m2 = Model()
    u = m2.add("SineWave", "U", Ts=0.1)
    pl = m2.add("DiscreteStateSpace", "P", A=Ad, B=Bd, C=C, D=[[0.0]], x0=[1.0, -0.5], Ts=0.1)
    ob2 = m2.add("Observer", "Obs", A=Ad, B=Bd, C=C, D=[[0.0]], L=Ld, Ts=0.1)
    m2.connect(u, pl)
    m2.connect(u, ob2, dst_port=0)
    m2.connect(pl, ob2, dst_port=1)
    m2.log_signal(ob2)
    cm = compile_model(m2)
    res = simulate(m2, 3.0, solver="ode4", step=0.1, log_all=True)
    e0 = np.array([1.0, -0.5])
    Acl = Ad - Ld @ C
    xs = []
    x = e0.copy()
    uk = np.sin(np.arange(31) * 0.1)
    for k in range(31):
        xs.append(x.copy())
        x = Ad @ x + (Bd * uk[k]).ravel()
    err = np.array(xs) - res.signals["Obs"]
    for k in range(31):
        np.testing.assert_allclose(err[k], np.linalg.matrix_power(Acl, k) @ e0, atol=1e-12)
    assert cm.blocks[cm.index["Obs"]].nxd == 2


def test_lti_block_duck_typing():
    from spt.control.diagram import lti_block
    b1 = lti_block(ss.TransferFunction([1.0, 2.0], [1.0, 3.0, 2.0]), "G")
    b2 = lti_block(ss.StateSpace([[0.5]], [[1.0]], [[2.0]], [[0.0]], dt=0.1), "H")
    assert b1.type_name == "TransferFcn" and b2.type_name == "DiscreteStateSpace"
    assert b2.params["Ts"].period == 0.1
    L = _single_block_lin("TransferFcn", num=b1.params["num"], den=b1.params["den"])
    assert L.dcgain()[0, 0] == pytest.approx(1.0)


def test_encoded_params_roundtrip_all_blocks():
    """Arayüzün kullandığı gidiş-dönüş: create_block(tür, ad, **blok.encoded_params()) aynı parametreleri verir."""
    special = {"ZeroPole": {"zeros": [-1 + 2j, -1 - 2j], "poles": [-3.0, -0.5 + 1j, -0.5 - 1j]},
               "Integrator": {"upper": 2.0}, "PIDController": {"Ts": [0.1, 0.05], "limit_output": True},
               "Saturation": {"upper": [1.0, float("inf")], "lower": [-1.0, -2.0]}}
    for c in block_catalog():
        if c["lazy"]:
            continue
        b = create_block(c["type"], "B", **special.get(c["type"], {}))
        enc = b.encoded_params()
        json.dumps(enc, allow_nan=False)
        b2 = create_block(c["type"], "B", **json.loads(json.dumps(enc)))
        for k, v in b.params.items():
            v2 = b2.params[k]
            if isinstance(v, np.ndarray):
                np.testing.assert_array_equal(v, v2)
            else:
                assert v == v2, (c["type"], k)
