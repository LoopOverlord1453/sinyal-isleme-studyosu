"""İHA otopilot şablonları ve blok diyagram blokları (kısa): blok kaydı / Jacobian'lar, tasarım kazançlarının
kütüphaneyle eşliği, kademeli otopilot ve roll öğreticisinin motorla kararlı benzetimi, LQR şablonunun kapalı çevrim
kutupları (diyagram doğrusallaştırması ⊇ dlqr tasarımı)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from spt.control.diagram import Model, block_types, create_block, linearize, linio, simulate
from spt.control.errors import ControlError
from spt.control.uav.autopilot import design_quad_cascade, design_quad_lqr_attitude, step_metrics
from spt.control.uav.common import numjac, rotmat_from_euler
from spt.control.uav.diagram_templates import TEMPLATES, build_quad_cascade, build_quad_lqr, build_roll_axis_tutorial
from spt.control.uav.quadrotor import Quadrotor, QuadrotorParams

UAV_TYPES = {"UAVQuadrotor", "UAVFixedWing", "UAVMixer", "UAVMotorsToVirtual", "UAVAccelToAttitude", "UAVLinearMPC",
             "UAVIMU", "UAVBarometer", "UAVGPS"}


def test_blocks_registered_and_templates_listed():
    assert UAV_TYPES <= set(block_types())
    assert len(TEMPLATES) == 5
    for t in TEMPLATES:
        assert {"name", "build", "description", "category"} <= set(t) and callable(t["build"])


def test_quadrotor_block_analytic_jacobian():
    blk = create_block("UAVQuadrotor", "Q", gyro=True, drag=[0.1, 0.12, 0.2], euler0=[0.2, -0.1, 0.5],
                       rates0=[0.3, -0.2, 0.1], motors0=[500, 560, 600, 540])
    q, x = blk.model, blk.x0()
    u = [np.array([510.0, 570.0, 590.0, 545.0])]
    A, B, C, D = blk.linearize(0.0, x, None, u)
    np.testing.assert_allclose(A, numjac(lambda xx: q.derivatives(xx, u[0]), x), rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(B, numjac(lambda uu: q.derivatives(x, uu), u[0]), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(C, numjac(lambda xx: np.concatenate(blk.output(0.0, xx, None, u)), x), rtol=1e-8,
                               atol=1e-8)
    with pytest.raises(ControlError):
        create_block("UAVQuadrotor", "Q2", inertia=[1.0, 2.0])


def test_mixer_inverse_mixer_and_thrust_vectoring():
    mix, inv = create_block("UAVMixer", "M"), create_block("UAVMotorsToVirtual", "I")
    v = np.array([14.0, 0.2, -0.1, 0.01])
    om = mix.output(0.0, None, None, [v])[0]
    np.testing.assert_allclose(om, Quadrotor(input_mode="virtual").motor_reference(v), rtol=1e-14)
    np.testing.assert_allclose(inv.output(0.0, None, None, [om])[0], v, rtol=1e-12, atol=1e-14)
    a2a = create_block("UAVAccelToAttitude", "A", mass=1.4, max_tilt=math.radians(40))
    g = a2a.params["g"]
    a, psi = np.array([2.0, -1.0, 0.5]), 0.7
    (phi, th), T = a2a.output(0.0, None, None, [a, np.array([psi])])
    R = rotmat_from_euler(phi, th, psi)
    np.testing.assert_allclose(np.array([0, 0, g]) - T[0] / 1.4 * R[:, 2], a, atol=1e-12)   # v̇ = g·e3 − T/m·R·e3


def test_cascade_design_equals_library_and_template():
    from spt.control.classical import pidtune
    from spt.control.lti import tf
    p = QuadrotorParams()
    d = design_quad_cascade(p)
    C, info = pidtune(tf([1 / 0.0211], [p.tau_m, 1, 0]), "PIDF", 30.0, PhaseMargin=60.0)
    assert (d.rate["roll"]["P"], d.rate["roll"]["I"], d.rate["roll"]["D"]) == (C.Kp, C.Ki, C.Kd)
    assert info["Stable"] and info["PhaseMargin"] == pytest.approx(60.0, abs=1e-6)
    m = build_quad_cascade(p)
    assert m.blocks["Hız PID (roll)"].params["P"] == d.rate["roll"]["P"]
    assert m.blocks["Açı P (pitch)"].params["P"] == d.angle["pitch"]
    assert m.blocks["Dikey hız PI"].params["I"] == d.alt_velocity["I"]
    assert Model.from_dict(m.to_dict()).validate().ok


def test_cascade_autopilot_tracks_step():
    m = build_quad_cascade(stop_time=7.0)
    assert m.validate().ok
    r = simulate(m)
    t, pos = r.signal("konum")
    _, eu = r.signal("Euler")
    _, om = r.signal("motor hızı")
    for k, (y0, y1, ts) in enumerate(((0, 2, 4.5), (0, 1, 5.8), (-10, -11, 4.5))):
        mm = step_metrics(t, pos[:, k], t_step=1.0, y0=y0, y1=y1)
        assert mm["overshoot"] < 5.0 and mm["settling_time"] < ts, (k, mm)
    my = step_metrics(t, eu[:, 2], t_step=1.0, y0=0, y1=0.5)
    assert my["overshoot"] < 2.0 and my["settling_time"] < 2.5
    assert np.max(np.abs(eu[:, :2])) < math.radians(35) and np.all((om > 0) & (om < 1006.6))


def test_roll_tutorial_stable_and_saturating():
    m = build_roll_axis_tutorial(step=0.05)
    L = linearize(m, [linio("φ referansı", 0, "input"), linio("∫ → φ", 0, "output")])
    assert L.Ts == pytest.approx(0.002) and np.max(np.abs(L.poles())) < 1.0
    assert L.dcgain()[0, 0] == pytest.approx(1.0, abs=1e-9)
    t, phi = simulate(m).signal("φ")
    mm = step_metrics(t, phi, t_step=0.1, y0=0, y1=0.05)
    assert mm["overshoot"] < 2.0 and mm["settling_time"] < 0.6
    big = simulate(build_roll_axis_tutorial(step=0.6))             # moment doyumu + anti-windup
    t, phi = big.signal("φ")
    assert np.max(np.abs(big.signal("Mx_c")[1])) == pytest.approx(1.5, rel=1e-9)
    mb = step_metrics(t, phi, t_step=0.1, y0=0, y1=0.6)
    assert mb["overshoot"] < 5.0 and mb["settling_time"] < 0.9


def test_lqr_template_closed_loop_contains_dlqr_poles():
    lq = design_quad_lqr_attitude()
    L = linearize(build_quad_lqr(), [linio("Tutum referansı [φ θ ψ]", 0, "input"), linio("Dörtrotor", 2, "output")])
    assert L.Ts == pytest.approx(0.004)
    full = list(L.poles())
    for z in lq["poles"]:
        k = int(np.argmin([abs(z - f) for f in full]))
        assert abs(z - full.pop(k)) < 1e-9
    assert np.max(np.abs(L.poles())) < 1.0 + 1e-12          # kalan: irtifa çevrimi, motor, z = 1 (konum durumları)
