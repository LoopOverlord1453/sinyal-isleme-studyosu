"""İHA otopilot şablonları ve blok diyagram blokları — motorla benzetim, tasarım kazançlarının kütüphaneyle eşliği ve
MATLAB R2025b karşılaştırmaları:

- LQR tutum: c2d + dlqr (MATLAB) ↔ design_quad_lqr_attitude; şablonun (diyagramın) askıda doğrusallaştırılmış kapalı
  çevrim kutupları dlqr'nin kapalı çevrim kutuplarını içerir.
- Roll ekseni öğreticisi: MATLAB'de aynı örneklenmiş veri sistemi (c2d(G, 'zoh') + pid(…, 'ForwardEuler') +
  connect / sumblk) → kutuplar ve step ↔ diyagramın doğrusallaştırması ve doğrusal olmayan benzetimi.
- MPC: MATLAB mpc / mpcstate / mpcmove ↔ diyagramdaki MPC bloğunun hamleleri (aynı ölçüm dizisiyle).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from spt.control.diagram import Model, block_catalog, block_types, create_block, linearize, linio, simulate
from spt.control.errors import ControlError
from spt.control.uav.autopilot import (
    design_fixedwing_slc, design_quad_cascade, design_quad_lqr_attitude, quad_mpc_model, step_metrics,
)
from spt.control.uav.common import numjac, rotmat_from_euler
from spt.control.uav.diagram_templates import (
    MPC_SETTINGS, TEMPLATES, build_fixedwing_slc, build_quad_cascade, build_quad_lqr, build_quad_mpc,
    build_roll_axis_tutorial,
)
from spt.control.uav.fixedwing import FixedWing
from spt.control.uav.quadrotor import Quadrotor, QuadrotorParams

UAV_TYPES = {"UAVQuadrotor", "UAVFixedWing", "UAVMixer", "UAVMotorsToVirtual", "UAVAccelToAttitude", "UAVLinearMPC",
             "UAVIMU", "UAVBarometer", "UAVGPS"}


def _m(M) -> str:
    M = np.atleast_2d(np.asarray(M, float))
    return "[" + "; ".join(" ".join(repr(float(v)) for v in row) for row in M) + "]"


def _script() -> str:
    d = design_quad_cascade()
    r = d.rate["roll"]
    s = MPC_SETTINGS
    A, B, C, D = quad_mpc_model(1.0 / d.wc["angle_rp"], QuadrotorParams().tau_m)
    return f"""
Jx = 0.0211; Jy = 0.0219; Jz = 0.0366; tau = 0.0136;
Ts = 0.004;
A = zeros(9); A(1:3,4:6) = eye(3); A(4:6,7:9) = diag(1./[Jx Jy Jz]); A(7:9,7:9) = -eye(3)/tau;
B = zeros(9,3); B(7:9,:) = eye(3)/tau;
sysd = c2d(ss(A, B, eye(9), zeros(9,3)), Ts, 'zoh');
Q = diag(1./[0.1 0.1 0.2 1 1 1 1 1 0.1].^2); R = diag(1./[0.5 0.5 0.08].^2);
[K, S, e] = dlqr(sysd.A, sysd.B, Q, R);
out.lqr.K = K; out.lqr.e = spt_c(e); out.lqr.Ad = sysd.A; out.lqr.Bd = sysd.B;
Ts2 = 0.002;
G = ss([-1/tau 0 0; 1/Jx 0 0; 0 1 0], [1/tau; 0; 0], [0 1 0; 0 0 1], [0; 0]);
Gd = c2d(G, Ts2, 'zoh'); Gd.InputName = {{'u'}}; Gd.OutputName = {{'p'; 'phi'}};
Cr = pid({r['P']!r}, {r['I']!r}, {r['D']!r}, {1.0 / r['N']!r}, Ts2, 'IFormula', 'ForwardEuler', 'DFormula', 'ForwardEuler');
Cr.InputName = {{'ep'}}; Cr.OutputName = {{'u'}};
Ca = pid({d.angle['roll']!r}, 0, 0, 0, Ts2); Ca.InputName = {{'ephi'}}; Ca.OutputName = {{'pref'}};
S1 = sumblk('ep = pref - p'); S2 = sumblk('ephi = phiref - phi');
T = connect(Gd, Cr, Ca, S1, S2, 'phiref', 'phi');
out.roll.poles = spt_c(pole(T));
[y, ~] = step(T, (0:Ts2:1.0)');
out.roll.y = y(:);
si = stepinfo(T); out.roll.overshoot = si.Overshoot; out.roll.settling = si.SettlingTime;
mpcverbosity off;
obj = mpc(ss({_m(A)}, {_m(B)}, {_m(C)}, {_m(D)}), {s['Ts']!r}, {s['p']}, {s['m']});
obj.Weights.OutputVariables = {_m(s['w_y'])}; obj.Weights.ManipulatedVariables = {_m(s['w_u'])};
obj.Weights.ManipulatedVariablesRate = {_m(s['w_du'])};
umin = {_m(s['u_min'])}; umax = {_m(s['u_max'])};
for i = 1:3, obj.MV(i).Min = umin(i); obj.MV(i).Max = umax(i); end
x = mpcstate(obj, [0 0 0 0 0 0 -10 0 0]');
ym = [0 0 -10 0 0 0; 0.002 0.001 -10.01 0.05 0.02 -0.2; 0.01 0.004 -10.03 0.15 0.07 -0.45; 0.03 0.01 -10.07 0.3 0.12 -0.7];
r = [2 1 -11 0 0 0];
U = zeros(4, 3);
for k = 1:4, U(k,:) = mpcmove(obj, x, ym(k,:)', r)'; end
out.mpc.U = U;
"""


SCRIPT = _script()
YM = np.array([[0, 0, -10, 0, 0, 0], [0.002, 0.001, -10.01, 0.05, 0.02, -0.2],
               [0.01, 0.004, -10.03, 0.15, 0.07, -0.45], [0.03, 0.01, -10.07, 0.3, 0.12, -0.7]])


@pytest.fixture(scope="module")
def ml():
    from tools.matlab_oracle import OracleUnavailable, fixture
    try:
        return fixture("uav_autopilot", SCRIPT)
    except OracleUnavailable as exc:  # pragma: no cover
        pytest.skip(str(exc))


def _cplx(v):
    a = np.atleast_2d(np.asarray(v, float))
    return a[:, 0] + 1j * a[:, 1]


def _match(sub, full, tol):
    """Her `sub` kutbu için `full` içinde en yakın kutup (birebir eşleme) ve en büyük fark."""
    full = list(full)
    worst = 0.0
    for z in sub:
        k = int(np.argmin([abs(z - f) for f in full]))
        worst = max(worst, abs(z - full.pop(k)))
    return worst


# ============================================================================ bloklar
def test_blocks_registered_with_turkish_catalog():
    assert UAV_TYPES <= set(block_types())
    cat = {c["type"]: c for c in block_catalog()}
    for t in UAV_TYPES:
        c = cat[t]
        assert c["display_name"] and c["description"] and not c["lazy"]
        assert all(p["label"] for p in c["params"])
    assert cat["UAVQuadrotor"]["outputs"] == ["konum", "hız", "Euler", "pqr", "Ω", "f_ivme"]


def test_quadrotor_block_matches_model_and_linearizes_analytically():
    blk = create_block("UAVQuadrotor", "Q", gyro=True, drag=[0.1, 0.12, 0.2], euler0=[0.2, -0.1, 0.5],
                       rates0=[0.3, -0.2, 0.1], motors0=[500, 560, 600, 540])
    q = blk.model
    x = blk.x0()
    u = [np.array([510.0, 570.0, 590.0, 545.0])]
    np.testing.assert_allclose(blk.derivatives(0.0, x, None, u), q.derivatives(x, u[0]), atol=0)
    y = blk.output(0.0, x, None, u)
    np.testing.assert_allclose(np.concatenate(y[:5]), q.output(x), atol=0)
    np.testing.assert_allclose(y[5], q.specific_force(x, u[0]), atol=0)
    A, B, C, D = blk.linearize(0.0, x, None, u)
    f = lambda xx: np.concatenate(blk.output(0.0, xx, None, u))  # noqa: E731
    np.testing.assert_allclose(A, numjac(lambda xx: q.derivatives(xx, u[0]), x), rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(B, numjac(lambda uu: q.derivatives(x, uu), u[0]), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(C, numjac(f, x), rtol=1e-8, atol=1e-8)
    assert D.shape == (19, 4) and np.allclose(D, 0)


def test_fixedwing_block_trim_start_and_linearization():
    blk = create_block("UAVFixedWing", "FW", Va0=25.0, h0=100.0)
    blk.reset()
    x = blk.x0()
    tr = FixedWing().trim(25.0)
    np.testing.assert_allclose(x, tr.x, atol=1e-12)
    assert np.max(np.abs(blk.derivatives(0.0, x, None, [tr.u])[3:])) < 1e-10
    A, B, C, D = blk.linearize(0.0, x, None, [tr.u])
    assert A.shape == (12, 12) and C.shape == (18, 12) and np.allclose(D, 0)
    y = np.concatenate(blk.output(0.0, x, None, [tr.u]))
    np.testing.assert_allclose(y[12:15], [25.0, tr.alpha, 0.0], atol=1e-12)


def test_mixer_and_inverse_mixer_blocks():
    mix = create_block("UAVMixer", "M", method="px4", output="omega")
    inv = create_block("UAVMotorsToVirtual", "I")
    q = Quadrotor(QuadrotorParams(), input_mode="virtual")
    v = np.array([14.0, 0.2, -0.1, 0.01])
    om = mix.output(0.0, None, None, [v])[0]
    np.testing.assert_allclose(om, q.motor_reference(v), rtol=1e-14)
    np.testing.assert_allclose(inv.output(0.0, None, None, [om])[0], v, rtol=1e-12, atol=1e-14)
    _, _, _, D = mix.linearize(0.0, None, None, [v])
    np.testing.assert_allclose(D, numjac(lambda vv: mix.output(0.0, None, None, [vv])[0], v), rtol=1e-8)
    _, _, _, Di = inv.linearize(0.0, None, None, [om])
    np.testing.assert_allclose(Di @ D, np.eye(4), atol=1e-10)
    pwm = create_block("UAVMixer", "P", output="pwm").output(0.0, None, None, [v])[0]
    np.testing.assert_allclose(1148.0 * pwm - 141.4, om, rtol=1e-12)
    sat = mix.output(0.0, None, None, [np.array([13.7, 0.3, -0.2, 0.8])])[0]
    assert np.all(sat >= 0) and np.all(sat <= 1006.6 + 1e-9)
    assert mix.linearize(0.0, None, None, [np.array([13.7, 0.3, -0.2, 0.8])]) is None   # doygun: sayısal


def test_accel_to_attitude_exact_inversion():
    blk = create_block("UAVAccelToAttitude", "A", mass=1.4, max_tilt=math.radians(40))
    g = blk.params["g"]
    att, T = blk.output(0.0, None, None, [np.zeros(3), np.array([0.3])])
    np.testing.assert_allclose(att, 0, atol=1e-15)
    assert T[0] == pytest.approx(1.4 * g)
    for a, psi in (([2.0, -1.0, 0.5], 0.7), ([-3.0, 1.5, -2.0], -2.0)):
        a = np.array(a)
        (phi, th), T = blk.output(0.0, None, None, [a, np.array([psi])])
        R = rotmat_from_euler(phi, th, psi)
        # v̇ = g·e3 − (T/m)·R·e3 = a  (tam)
        np.testing.assert_allclose(np.array([0, 0, g]) - T[0] / 1.4 * R[:, 2], a, atol=1e-12)
    (phi, th), _ = blk.output(0.0, None, None, [np.array([30.0, 0, 0]), np.array([0.0])])
    tilt = math.acos(math.cos(phi) * math.cos(th))
    assert tilt == pytest.approx(math.radians(40), rel=1e-12)               # eğim sınırı


def test_sensor_blocks():
    imu = create_block("UAVIMU", "IMU", gyro_noise=0.01, accel_noise=0.1, gyro_bias=[0.01, 0, 0], seed=5)
    imu.reset()
    mem = imu.initial_memory()
    w, f = np.array([0.1, 0.2, 0.3]), np.array([0.0, 0.0, -9.8])
    ys = []
    for k in range(2000):
        y, mem = imu.compute(k * 0.002, [w, f], mem)
        ys.append(y)
    ys = np.array(ys)
    np.testing.assert_allclose(ys[:, :3].mean(0), w + [0.01, 0, 0], atol=4 * 0.01 / math.sqrt(0.002) / math.sqrt(2000))
    np.testing.assert_allclose(ys[:, 3:].std(0), 0.1 / math.sqrt(0.002), rtol=0.08)
    y_again, _ = imu.compute(7 * 0.002, [w, f], imu.initial_memory())
    y7, _ = imu.compute(7 * 0.002, [w, f], imu.initial_memory())
    np.testing.assert_array_equal(y_again, y7)                                  # sayaç tabanlı: tekrarlanabilir
    baro = create_block("UAVBarometer", "B")
    assert baro.compute(0.0, [np.array([1.0, 2.0, -12.5])], None)[0][0] == pytest.approx(12.5)
    gps = create_block("UAVGPS", "G", sigma_pos=[0, 0, 0], sigma_vel=0.0, delay=2)
    mem = gps.initial_memory()
    outs = []
    for k in range(5):
        y, mem = gps.compute(k * 0.2, [np.array([k, 0.0, 0.0]), np.zeros(3)], mem)
        outs.append(y[0])
    np.testing.assert_allclose(outs, [0, 0, 0, 1, 2])                          # 2 örnek gecikme


# ============================================================================ tasarım kazançları = kütüphane
def test_cascade_design_meets_targets_and_templates_use_it():
    from spt.control.classical import pidtune
    from spt.control.lti import tf
    p = QuadrotorParams()
    d = design_quad_cascade(p)
    for axis in ("roll", "pitch", "yaw"):
        info = d.info[axis]["rate"]
        assert info["Stable"] and info["CrossoverFrequency"] == pytest.approx(d.wc["rate_yaw" if axis == "yaw"
                                                                                   else "rate_rp"], rel=1e-9)
        assert info["PhaseMargin"] == pytest.approx(60.0, abs=1e-6)
    C, _ = pidtune(tf([1 / 0.0211], [p.tau_m, 1, 0]), "PIDF", 30.0, PhaseMargin=60.0)
    assert d.rate["roll"]["P"] == C.Kp and d.rate["roll"]["I"] == C.Ki and d.rate["roll"]["D"] == C.Kd
    m = build_quad_cascade(p)
    b = m.blocks["Hız PID (roll)"].params
    assert (b["P"], b["I"], b["D"], b["N"]) == (d.rate["roll"]["P"], d.rate["roll"]["I"], d.rate["roll"]["D"],
                                                d.rate["roll"]["N"])
    assert m.blocks["Açı P (pitch)"].params["P"] == d.angle["pitch"]
    assert m.blocks["Hız PI (N)"].params["I"] == d.velocity["I"]
    assert m.blocks["Dikey hız PI"].params["P"] == d.alt_velocity["P"]
    assert m.blocks["Hız PID (roll)"].params["Ts"].period == 0.002
    assert m.blocks["Açı P (roll)"].params["Ts"].period == 0.004
    assert m.blocks["Konum P (N)"].params["Ts"].period == 0.02


def test_fixedwing_book_gain_formulas():
    fw = FixedWing()
    tr = fw.trim(25.0)
    g = design_fixedwing_slc(fw, tr)
    c = g["coef"]
    # kapalı çevrim karakteristik denklemleri kitabın hedeflerini verir
    wn, z = g["wn"]["roll"], g["specs"]["roll"]["zeta"]
    assert np.allclose(np.sort(np.roots([1, c["a_phi1"] + c["a_phi2"] * g["kd_phi"], c["a_phi2"] * g["kp_phi"]])),
                       np.sort(np.roots([1, 2 * z * wn, wn ** 2])), rtol=1e-12)
    wn, z = g["wn"]["pitch"], g["specs"]["pitch"]["zeta"]
    assert np.allclose(np.sort(np.roots([1, c["a_theta1"] + c["a_theta3"] * g["kd_theta"],
                                         c["a_theta2"] + c["a_theta3"] * g["kp_theta"]])),
                       np.sort(np.roots([1, 2 * z * wn, wn ** 2])), rtol=1e-12)
    wv = g["specs"]["airspeed"]["wn"]
    assert np.allclose(np.roots([1, c["a_V1"] + c["a_V2"] * g["kp_V"], c["a_V2"] * g["ki_V"]]),
                       np.roots([1, 2 * g["specs"]["airspeed"]["zeta"] * wv, wv ** 2]), rtol=1e-12)
    assert g["wn"]["course"] == pytest.approx(g["wn"]["roll"] / g["specs"]["course"]["W"])
    m = build_fixedwing_slc()
    assert m.blocks["Roll P (kp_φ)"].params["P"] == g["kp_phi"]
    assert m.blocks["Pitch sönümü (kd_θ)"].params["P"] == g["kd_theta"]


# ============================================================================ şablon benzetimleri
@pytest.fixture(scope="module")
def sim_cascade():
    m = build_quad_cascade(stop_time=7.0)
    return m, simulate(m)


def test_template_list_format():
    assert len(TEMPLATES) == 5
    for t in TEMPLATES:
        assert {"name", "build", "description", "category"} <= set(t) and t["category"] == "Otopilot"
        assert callable(t["build"])


def test_cascade_tracks_position_step(sim_cascade):
    m, r = sim_cascade
    assert m.validate().ok
    t, pos = r.signal("konum")
    _, eu = r.signal("Euler")
    _, om = r.signal("motor hızı")
    for k, (y0, y1, ts) in enumerate(((0, 2, 4.5), (0, 1, 5.8), (-10, -11, 4.5))):
        mm = step_metrics(t, pos[:, k], t_step=1.0, y0=y0, y1=y1)
        assert mm["overshoot"] < 5.0 and mm["settling_time"] < ts, (k, mm)
    my = step_metrics(t, eu[:, 2], t_step=1.0, y0=0, y1=0.5)
    assert my["overshoot"] < 2.0 and my["settling_time"] < 2.5
    assert np.max(np.abs(eu[:, :2])) < math.radians(35)
    assert np.all((om > 0) & (om < 1006.6))


def test_cascade_survives_motor_saturation():
    p = QuadrotorParams(omega_max=760.0)                 # itki/ağırlık ≈ 1.9: tırmanışta üst doygunluk
    m = build_quad_cascade(p, step=(8.0, -6.0, -6.0, 1.5), stop_time=11.0)
    r = simulate(m)
    t, pos = r.signal("konum")
    _, cmd = r.signal("motor komutu")
    assert np.all(np.isfinite(pos))
    assert np.mean(cmd >= 760.0 - 1e-9) > 0.005                               # gerçekten doygun
    assert np.linalg.norm(pos[-1] - [8.0, -6.0, -16.0]) < 0.6


def test_lqr_template_and_matlab_dlqr(ml):
    lq = design_quad_lqr_attitude()
    np.testing.assert_allclose(lq["Ad"], ml["lqr"]["Ad"], rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(lq["Bd"], ml["lqr"]["Bd"], rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(lq["K"], ml["lqr"]["K"], rtol=1e-9, atol=1e-12)
    e_ml = _cplx(ml["lqr"]["e"])
    assert _match(e_ml, lq["poles"], 1e-9) < 1e-9
    m = build_quad_lqr()
    L = linearize(m, [linio("Tutum referansı [φ θ ψ]", 0, "input"), linio("Dörtrotor", 2, "output")])
    assert L.Ts == pytest.approx(0.004)
    assert _match(e_ml, L.poles(), 1e-6) < 1e-6              # diyagramın kapalı çevrimi dlqr kutuplarını içerir
    assert np.max(np.abs(L.poles())) < 1.0 + 1e-9            # (konum durumları G/Ç yolunda değil: atılır)
    np.testing.assert_allclose(L.dcgain(), np.eye(3), atol=1e-6)   # açı referansında kalıcı hatasız
    r = simulate(m)
    t, eu = r.signal("Euler")
    for k, y1 in enumerate((0.3, -0.2, 0.5)):
        mm = step_metrics(t, eu[:, k], t_step=0.5, y0=0.0, y1=y1)
        assert mm["settling_time"] < 2.2 and mm["overshoot"] < 15.0, (k, mm)
    _, pos = r.signal("konum")
    assert np.max(np.abs(pos[:, 2] + 10.0)) < 0.3


def test_mpc_template_tracks_and_matches_matlab(ml):
    m = build_quad_mpc(stop_time=6.0)
    blk = m.blocks["MPC (konum)"]
    mem = blk.initial_memory()
    U = []
    for k in range(4):
        u, mem = blk.compute(k * 0.05, [YM[k], np.array([2.0, 1.0, -11.0, 0, 0, 0])], mem)
        U.append(u)
    np.testing.assert_allclose(np.array(U), ml["mpc"]["U"], rtol=1e-7, atol=1e-8)
    r = simulate(m)
    t, pos = r.signal("konum")
    _, a = r.signal("ivme komutu")
    for k, (y0, y1) in enumerate(((0, 2), (0, 1), (-10, -11))):
        mm = step_metrics(t, pos[:, k], t_step=1.0, y0=y0, y1=y1)
        assert mm["overshoot"] < 15.0 and mm["settling_time"] < 4.8, (k, mm)
    lo, hi = np.array(MPC_SETTINGS["u_min"]), np.array(MPC_SETTINGS["u_max"])
    assert np.all(a >= lo - 1e-6) and np.all(a <= hi + 1e-6)                 # MV kısıtları


def test_fixedwing_slc_template():
    m = build_fixedwing_slc(times=(2.0, 12.0, 22.0), stop_time=32.0)
    assert m.validate().ok
    r = simulate(m)
    t, h = r.signal("h")
    _, ad = r.signal("hava verisi")
    pre = t < 12.0                                               # irtifa adımı, dönüşten önce
    mh = step_metrics(t[pre], h[pre], t_step=2.0, y0=100.0, y1=115.0)
    assert mh["settling_time"] < 8.0 and mh["overshoot"] < 20.0, mh
    assert np.max(np.abs(h[t >= 12.0] - 115.0)) < 1.5                        # koordineli dönüşte irtifa kaybı küçük
    mc = step_metrics(t, ad[:, 4], t_step=12.0, y0=0.0, y1=math.radians(60))
    assert mc["settling_time"] < 5.0 and mc["overshoot"] < 5.0, mc
    mv = step_metrics(t, ad[:, 0], t_step=22.0, y0=25.0, y1=28.0)
    assert mv["settling_time"] < 5.0 and mv["overshoot"] < 5.0, mv
    assert np.max(np.abs(ad[:, 2])) < math.radians(3)                         # yan kayma küçük
    _, dl = r.signal("dümenler")
    assert np.all(dl[:, 3] >= 0) and np.all(dl[:, 3] <= 1)


def test_roll_tutorial_matches_matlab(ml):
    m = build_roll_axis_tutorial(step=0.05)
    L = linearize(m, [linio("φ referansı", 0, "input"), linio("∫ → φ", 0, "output")])
    assert L.Ts == pytest.approx(0.002)
    p_ml = _cplx(ml["roll"]["poles"])
    assert len(L.poles()) == len(p_ml) and _match(p_ml, L.poles(), 1e-9) < 1e-9
    y_ml = np.asarray(ml["roll"]["y"], float)
    x = np.zeros(L.A.shape[0])
    y = []
    for _ in range(y_ml.size):
        y.append((L.C @ x + L.D @ [1.0])[0])
        x = L.A @ x + L.B @ [1.0]
    np.testing.assert_allclose(y, y_ml, atol=1e-9)
    # doğrusal olmayan benzetim (doyum etkin değil, 0.05 rad adım) aynı örneklenmiş veri cevabını verir
    r = simulate(m)
    t, phi = r.signal("φ")
    k0 = int(round(0.1 / 0.002))
    n = min(y_ml.size, phi.size - k0)
    np.testing.assert_allclose(phi[k0:k0 + n], 0.05 * y_ml[:n], atol=2e-6)
    mm = step_metrics(t, phi, t_step=0.1, y0=0, y1=0.05)
    assert mm["overshoot"] == pytest.approx(max(0.0, ml["roll"]["overshoot"]), abs=0.2)
    # büyük adımda moment doyumu + anti-windup: aşımsız, hızlı
    big = simulate(build_roll_axis_tutorial(step=0.6))
    t, phi = big.signal("φ")
    _, mx = big.signal("Mx_c")
    assert np.max(np.abs(mx)) == pytest.approx(1.5, rel=1e-9)               # doyuma ulaştı
    mb = step_metrics(t, phi, t_step=0.1, y0=0, y1=0.6)
    assert mb["overshoot"] < 5.0 and mb["settling_time"] < 0.9


def test_templates_save_and_reload():
    for t in TEMPLATES:
        m = t["build"]()
        d = m.to_dict()
        m2 = Model.from_dict(d)
        assert set(m2.blocks) == set(m.blocks)
        assert m2.validate().ok, t["name"]


def test_block_parameter_validation():
    with pytest.raises(ControlError):
        create_block("UAVQuadrotor", "Q", inertia=[1.0, 2.0])
    with pytest.raises(ControlError):
        create_block("UAVQuadrotor", "Q", motors0=[1.0, 2.0])
    with pytest.raises(ControlError):
        create_block("UAVFixedWing", "F", trim_init=False, x0=[1.0])
    with pytest.raises(ControlError):
        create_block("UAVLinearMPC", "M", A=[[1, 0], [0, 1]], B=[[1]], C=[[1, 0]], D=[[0]])


def test_mpc_block_refuses_silent_linearization():
    m = build_quad_mpc(stop_time=1.0)
    with pytest.raises(ControlError):
        linearize(m, [linio("Konum referansı [pn pe pd]", 0, "input"), linio("Dörtrotor", 0, "output")])
