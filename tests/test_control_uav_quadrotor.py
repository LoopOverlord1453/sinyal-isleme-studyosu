"""Dörtrotor 6-DOF modeli — analitik çözümler, korunum yasaları, Jacobian tutarlılığı, karıştırıcı ve MATLAB R2025b
referansları (Aerospace / UAV Toolbox döndürme kuralları, aynı denklemlerin ode45 çözümü, Symbolic Math Toolbox ile
türetilmiş Jacobian).

MATLAB tarafı Python kodundan bağımsız yazılmıştır: döndürme quat2dcm, kinematik quatmultiply, momentler rotor başına
cross(r_i, F_i) ile; Jacobian sembolik türetmeyle (jacobian) elde edilir.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.integrate import quad

from spt.control.errors import ControlError
from spt.control.uav.common import (
    euler_from_quat, numjac, quat_from_euler, quat_mul, rotmat_from_euler, rotmat_from_quat,
)
from spt.control.uav.quadrotor import (
    Mixer, Quadrotor, QuadrotorParams, axis_transfer_functions, hover_small_angle_model,
)


def _mfile(name: str, code: str) -> str:
    """MATLAB betiğinde yardımcı fonksiyon dosyası yazan komut (oracle betiği tek fonksiyon gövdesidir)."""
    lines = [ln.replace('"', '""') for ln in code.strip("\n").splitlines()]
    body = ";\n    ".join(f'"{ln}"' for ln in lines)
    return f"writelines([\n    {body}], fullfile(pwd, '{name}.m'));\n"


QR_FQ = r"""
function dx = qr_fq(t, x, P, ufun)
v = x(4:6); q = x(7:10); w = x(11:13); Om = x(14:17);
Omref = min(max(ufun(t), P.ommin), P.ommax);
Omdot = (Omref - Om)/P.tau;
qn = q/norm(q);
Rt = quat2dcm(qn.');
R = Rt.';
Ti = P.kT*Om.^2;
M = zeros(3,1);
for i = 1:4
    M = M + cross(P.pos(:,i), [0; 0; -Ti(i)]) + [0; 0; P.spin(i)*P.kQ*Om(i)^2];
end
h = P.gyro*[0; 0; -P.Jr*sum(P.spin(:).*Om)];
hdot = P.gyro*[0; 0; -P.Jr*sum(P.spin(:).*Omdot)];
Fb = [0; 0; -sum(Ti)] - P.Dt*(Rt*(v - P.wind));
vdot = [0; 0; P.g] + R*Fb/P.m;
qdot = 0.5*quatmultiply(q.', [0 w.']).' + P.ks*(1 - q.'*q)*q;
wdot = P.J\(M - P.Dr*w - cross(w, P.J*w + h) - hdot);
dx = [v; vdot; qdot; wdot; Omdot];
end
"""

QR_FE = r"""
function dx = qr_fe(x, u, P)
v = x(4:6); ph = x(7); th = x(8); ps = x(9); w = x(10:12); Om = x(13:16);
Omdot = (u - Om)/P.tau;
Rx = [1 0 0; 0 cos(ph) -sin(ph); 0 sin(ph) cos(ph)];
Ry = [cos(th) 0 sin(th); 0 1 0; -sin(th) 0 cos(th)];
Rz = [cos(ps) -sin(ps) 0; sin(ps) cos(ps) 0; 0 0 1];
R = Rz*Ry*Rx;
Ti = P.kT*Om.^2;
M = sym(zeros(3,1));
for i = 1:4
    M = M + cross(P.pos(:,i), [0; 0; -Ti(i)]) + [0; 0; P.spin(i)*P.kQ*Om(i)^2];
end
h = P.gyro*[0; 0; -P.Jr*(P.spin*Om)];
hdot = P.gyro*[0; 0; -P.Jr*(P.spin*Omdot)];
Fb = [0; 0; -sum(Ti)] - P.Dt*(R.'*(v - P.wind));
vdot = [0; 0; P.g] + R*Fb/P.m;
W = [1 sin(ph)*tan(th) cos(ph)*tan(th); 0 cos(ph) -sin(ph); 0 sin(ph)/cos(th) cos(ph)/cos(th)];
wdot = P.J\(M - P.Dr*w - cross(w, P.J*w + h) - hdot);
dx = [v; vdot; W*w; wdot; Omdot];
end
"""

QUAD_SCRIPT = _mfile("qr_fq", QR_FQ) + _mfile("qr_fe", QR_FE) + r"""
rehash;
E = [0.2 -0.1 0.5; -1.0 0.7 2.5; 0.4 1.2 -2.9];
for k = 1:3
    c = struct();
    c.dcm = angle2dcm(E(k,3), E(k,2), E(k,1), 'ZYX');
    c.q = angle2quat(E(k,3), E(k,2), E(k,1), 'ZYX');
    c.quat2dcm = quat2dcm(c.q);
    c.eul2rotm = eul2rotm([E(k,3) E(k,2) E(k,1)], 'ZYX');
    c.eul2quat = eul2quat([E(k,3) E(k,2) E(k,1)], 'ZYX');
    [y, p, r] = quat2angle(c.q, 'ZYX');
    c.quat2angle = [r p y];
    out.conv{k} = c;
end
out.qmul = quatmultiply([0.3 0.5 -0.2 0.7], [0.9 -0.1 0.3 0.2]);
a = 0.225/sqrt(2);
PA = struct('m', 1.4, 'g', 9.80665, 'J', diag([0.0211 0.0219 0.0366]), 'kT', 1.105e-5, 'kQ', 1.779e-7, ...
    'tau', 0.0136, 'ommin', 0, 'ommax', 1006.6, 'Jr', 1.287e-4, 'gyro', 1, 'Dt', diag([0.1 0.12 0.2]), ...
    'Dr', diag([0.002 0.003 0.004]), 'ks', 1, 'wind', [1; -0.5; 0]);
PA.pos = [a -a a -a; a -a -a a; 0 0 0 0]; PA.spin = [1 1 -1 -1];
omA = sqrt(PA.m*PA.g/(4*PA.kT));
uA = @(t) omA + [30*sin(2*pi*1.3*t); -20*sin(2*pi*0.7*t + 0.5); 25*cos(2*pi*0.9*t) - 25; -15*sin(2*pi*1.1*t)];
x0A = [1; -2; -10; 1; -0.5; 0.2; angle2quat(0.5, -0.1, 0.2).'; 0.3; -0.2; 0.1; omA + [10; -5; 8; -3]];
opts = odeset('RelTol', 1e-12, 'AbsTol', 1e-12);
[tt, X] = ode45(@(t, x) qr_fq(t, x, PA, uA), 0:0.05:2, x0A, opts);
out.simA.t = tt; out.simA.x = X; out.simA.x0 = x0A;
L = 0.225;
PB = PA; PB.J = [0.03 0.001 -0.002; 0.001 0.035 0.0005; -0.002 0.0005 0.06];
PB.Dt = zeros(3); PB.Dr = zeros(3); PB.wind = [0; 0; 0];
PB.pos = [0 0 L -L; L -L 0 0; 0 0 0 0];
omB = sqrt(PB.m*PB.g/(4*PB.kT));
uB = @(t) omB + [40*sin(3*t); 40*sin(3*t + 1); -30*sin(2*t); 20*cos(4*t) - 20];
x0B = [0; 0; -5; 0; 0; 0; angle2quat(-1.0, 0.3, 0.6).'; -0.5; 0.4; 0.8; omB*ones(4,1)];
[tt, X] = ode45(@(t, x) qr_fq(t, x, PB, uB), 0:0.05:2, x0B, opts);
out.simB.t = tt; out.simB.x = X; out.simB.x0 = x0B;
xs = sym('x', [16 1], 'real'); us = sym('u', [4 1], 'real');
fe = qr_fe(xs, us, PA);
JA = jacobian(fe, xs); JB = jacobian(fe, us);
xh = [0; 0; 0; 0; 0; 0; 0; 0; 0.3; 0; 0; 0; omA*ones(4,1)]; uh = omA*ones(4,1);
out.jh.A = double(subs(JA, [xs; us], [xh; uh])); out.jh.B = double(subs(JB, [xs; us], [xh; uh]));
out.jh.f = double(subs(fe, [xs; us], [xh; uh]));
xg = [1; 2; -3; 1; -2; 0.5; 0.3; -0.2; 1.0; 0.5; -0.4; 0.3; 500; 560; 600; 540]; ug = [510; 570; 590; 545];
out.jg.A = double(subs(JA, [xs; us], [xg; ug])); out.jg.B = double(subs(JB, [xs; us], [xg; ug]));
out.jg.f = double(subs(fe, [xs; us], [xg; ug]));
out.omA = omA;
"""


@pytest.fixture(scope="module")
def ml():
    from tools.matlab_oracle import OracleUnavailable, fixture
    try:
        return fixture("uav_quadrotor", QUAD_SCRIPT)
    except OracleUnavailable as exc:  # pragma: no cover - MATLAB yoksa ve kayıt yoksa
        pytest.skip(str(exc))


def _params_A(**kw) -> QuadrotorParams:
    base = dict(gyro=True, drag=(0.1, 0.12, 0.2), rot_drag=(0.002, 0.003, 0.004))
    base.update(kw)
    return QuadrotorParams(**base)


def _params_B() -> QuadrotorParams:
    return QuadrotorParams(config="+", gyro=True,
                           inertia=np.array([[0.03, 0.001, -0.002], [0.001, 0.035, 0.0005], [-0.002, 0.0005, 0.06]]))


def _rel(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.max(np.abs(a - b)) / max(1.0, float(np.max(np.abs(b)))))


# ============================================================================ MATLAB: döndürme kuralları
def test_rotation_conventions_match_aerospace_and_uav_toolbox(ml):
    E = [(0.2, -0.1, 0.5), (-1.0, 0.7, 2.5), (0.4, 1.2, -2.9)]
    for (roll, pitch, yaw), c in zip(E, ml["conv"]):
        R = rotmat_from_euler(roll, pitch, yaw)
        q = quat_from_euler(roll, pitch, yaw)
        np.testing.assert_allclose(np.array(c["dcm"]), R.T, atol=1e-14)          # angle2dcm = NED→gövde = Rᵀ
        np.testing.assert_allclose(np.array(c["eul2rotm"]), R, atol=1e-14)      # eul2rotm = gövde→NED = R
        qm = np.array(c["q"]) * np.sign(np.array(c["q"])[0]) * np.sign(q[0])
        np.testing.assert_allclose(qm, q, atol=1e-14)                            # angle2quat ile aynı q
        qu = np.array(c["eul2quat"]) * np.sign(np.array(c["eul2quat"])[0]) * np.sign(q[0])
        np.testing.assert_allclose(qu, q, atol=1e-14)
        np.testing.assert_allclose(np.array(c["quat2dcm"]), rotmat_from_quat(q).T, atol=1e-14)
        np.testing.assert_allclose(c["quat2angle"], euler_from_quat(q), atol=1e-12)
    np.testing.assert_allclose(ml["qmul"], quat_mul([0.3, 0.5, -0.2, 0.7], [0.9, -0.1, 0.3, 0.2]), atol=1e-15)


# ============================================================================ MATLAB: ode45 ile aynı denklemler
@pytest.mark.parametrize("case", ["A", "B"])
def test_simulation_matches_matlab_ode45(ml, case):
    sim = ml[f"sim{case}"]
    p = _params_A() if case == "A" else _params_B()
    quad_ = Quadrotor(p, wind=(1.0, -0.5, 0.0) if case == "A" else None)
    om0 = quad_.hover_motor_speeds()[0]
    assert om0 == pytest.approx(ml["omA"], rel=1e-13)
    if case == "A":
        uf = lambda t, x: om0 + np.array([30 * math.sin(2 * math.pi * 1.3 * t),  # noqa: E731
                                          -20 * math.sin(2 * math.pi * 0.7 * t + 0.5),
                                          25 * math.cos(2 * math.pi * 0.9 * t) - 25,
                                          -15 * math.sin(2 * math.pi * 1.1 * t)])
        x0 = quad_.initial_state((1, -2, -10), (1, -0.5, 0.2), (0.2, -0.1, 0.5), (0.3, -0.2, 0.1),
                                 om0 + np.array([10, -5, 8, -3]))
    else:
        uf = lambda t, x: om0 + np.array([40 * math.sin(3 * t), 40 * math.sin(3 * t + 1),  # noqa: E731
                                          -30 * math.sin(2 * t), 20 * math.cos(4 * t) - 20])
        x0 = quad_.initial_state((0, 0, -5), (0, 0, 0), (0.6, 0.3, -1.0), (-0.5, 0.4, 0.8), om0)
    np.testing.assert_allclose(x0, sim["x0"], rtol=1e-13, atol=1e-14)
    t = np.array(sim["t"])
    res = quad_.simulate(x0, uf, t, method="DOP853", rtol=1e-12, atol=1e-12)
    X = np.array(sim["x"])
    scale = np.maximum(1.0, np.max(np.abs(X), axis=0))
    err = np.max(np.abs(res.x - X) / scale)
    assert err < 1e-10, err
    # gerçekten hareketli bir senaryo (önemsiz olmayan karşılaştırma)
    assert np.ptp(res.info["euler"][:, 0]) > 0.1


# ============================================================================ MATLAB: sembolik Jacobian
@pytest.mark.parametrize("point", ["jh", "jg"])
def test_analytic_jacobian_matches_matlab_symbolic(ml, point):
    p = _params_A()
    quad_ = Quadrotor(p, wind=(1.0, -0.5, 0.0))
    om0 = quad_.hover_motor_speeds()
    if point == "jh":
        x0 = quad_.initial_state(euler=(0, 0, 0.3), motor_speeds=om0)
        u0 = om0.copy()
    else:
        x0 = quad_.initial_state((1, 2, -3), (1, -2, 0.5), (0.3, -0.2, 1.0), (0.5, -0.4, 0.3), [500, 560, 600, 540])
        u0 = np.array([510.0, 570.0, 590.0, 545.0])
    lin = quad_.linearize(x0, u0, attitude="euler")
    ref = ml[point]
    assert _rel(lin.A, ref["A"]) < 1e-12
    assert _rel(lin.B, ref["B"]) < 1e-12
    assert _rel(lin.xdot0, ref["f"]) < 1e-12


# ============================================================================ askıda kalma (trim)
@pytest.mark.parametrize("mode", ["omega", "pwm", "virtual"])
@pytest.mark.parametrize("config", ["x", "+"])
def test_hover_trim(mode, config):
    p = QuadrotorParams(config=config, gyro=True)
    quad_ = Quadrotor(p, input_mode=mode)
    x0, u0 = quad_.trim_hover(position=(3, -1, -20), yaw=1.1)
    om = math.sqrt(p.mass * p.g / (4 * p.kT))
    np.testing.assert_allclose(quad_.hover_motor_speeds(), om, rtol=1e-14)
    np.testing.assert_allclose(p.kT * quad_.hover_motor_speeds() ** 2, p.mass * p.g / 4, rtol=1e-14)
    assert np.max(np.abs(quad_.derivatives(x0, u0))) < 1e-10
    np.testing.assert_allclose(quad_.euler(x0), (0, 0, 1.1), atol=1e-15)
    f = quad_.specific_force(x0, u0)
    np.testing.assert_allclose(f, [0, 0, -p.g], atol=1e-12)            # ivmeölçer askıda −g·ẑ okur
    if mode == "pwm":
        np.testing.assert_allclose(p.cr * u0 + p.wb, om, rtol=1e-14)


def test_hover_impossible_raises():
    with pytest.raises(ControlError):
        Quadrotor(QuadrotorParams(mass=6.0)).trim_hover()


# ============================================================================ analitik çözümler
def test_free_fall():
    p = QuadrotorParams()
    quad_ = Quadrotor(p)
    x0 = quad_.initial_state((1, 2, -50), (2, -1, -3), (0.1, 0.2, 0.3), (0, 0, 0), np.zeros(4))
    t = np.linspace(0, 2.0, 11)
    res = quad_.simulate(x0, np.zeros(4), t, method="DOP853", rtol=1e-12, atol=1e-12)
    pos = x0[0:3] + np.outer(t, x0[3:6]) + 0.5 * np.outer(t ** 2, [0, 0, p.g])
    np.testing.assert_allclose(res.x[:, 0:3], pos, atol=1e-9)
    np.testing.assert_allclose(res.x[:, 6:10], np.tile(x0[6:10], (t.size, 1)), atol=1e-12)


def test_pure_yaw_moment():
    p = QuadrotorParams(gyro=True)                       # Ω sabit ve ω ∥ h_r: jiroskopik terimler sıfır olmalı
    quad_ = Quadrotor(p, input_mode="virtual")
    Mz = 0.02
    u = np.array([p.mass * p.g, 0, 0, Mz])
    om = np.sqrt(quad_.mixer.allocate(u)[0] / p.kT)
    x0 = quad_.initial_state((0, 0, -10), euler=(0, 0, 0.4), motor_speeds=om)
    t = np.linspace(0, 1.5, 7)
    res = quad_.simulate(x0, u, t, method="DOP853", rtol=1e-12, atol=1e-12)
    a = Mz / p.inertia[2, 2]
    np.testing.assert_allclose(res["r"], a * t, atol=1e-10)
    psi = 0.4 + 0.5 * a * t ** 2
    np.testing.assert_allclose(np.unwrap(res.info["euler"][:, 2]), psi, atol=1e-9)
    np.testing.assert_allclose(res.x[:, 0:3], np.tile(x0[0:3], (t.size, 1)), atol=1e-9)
    np.testing.assert_allclose(res.info["euler"][:, :2], 0, atol=1e-10)


def test_pure_roll_moment():
    p = QuadrotorParams()
    quad_ = Quadrotor(p, input_mode="virtual")
    Mx = 0.05
    u = np.array([p.mass * p.g, Mx, 0, 0])
    om = np.sqrt(quad_.mixer.allocate(u)[0] / p.kT)
    x0 = quad_.initial_state(motor_speeds=om)
    t = np.linspace(0, 1.0, 6)
    res = quad_.simulate(x0, u, t, method="DOP853", rtol=1e-12, atol=1e-12)
    a = Mx / p.inertia[0, 0]
    np.testing.assert_allclose(res["p"], a * t, atol=1e-10)
    np.testing.assert_allclose(res.info["euler"][:, 0], 0.5 * a * t ** 2, atol=1e-9)
    # öteleme: v̇e = g·sinφ(t), v̇d = g·(1 − cosφ(t)) (T = m·g) → konum = ∫₀ᵗ (t − s)·ivme(s) ds
    for k, tk in enumerate(t):
        pe = quad(lambda s: (tk - s) * p.g * math.sin(0.5 * a * s * s), 0, tk, epsabs=1e-13, epsrel=1e-13)[0]
        pd = quad(lambda s: (tk - s) * p.g * (1 - math.cos(0.5 * a * s * s)), 0, tk, epsabs=1e-13,
                  epsrel=1e-13)[0]
        assert res["pe"][k] == pytest.approx(pe, abs=1e-9)
        assert res["pd"][k] == pytest.approx(pd, abs=1e-9)
        assert res["pn"][k] == pytest.approx(0.0, abs=1e-12)


def test_energy_and_angular_momentum_conservation():
    J = np.array([[0.03, 0.001, -0.002], [0.001, 0.035, 0.0005], [-0.002, 0.0005, 0.06]])
    p = QuadrotorParams(inertia=J)
    quad_ = Quadrotor(p)
    x0 = quad_.initial_state((0, 0, -100), (3, -2, 1), (0.3, -0.4, 1.0), (4.0, -2.0, 3.0), np.zeros(4))
    t = np.linspace(0, 10, 101)
    res = quad_.simulate(x0, np.zeros(4), t, method="DOP853", rtol=1e-12, atol=1e-12)
    E = np.array([quad_.kinetic_energy(x) + p.mass * p.g * x[2] * -1 for x in res.x])
    H = np.array([quad_.angular_momentum_world(x) for x in res.x])
    Er = np.array([0.5 * x[10:13] @ J @ x[10:13] for x in res.x])
    assert np.max(np.abs(E - E[0])) / abs(E[0]) < 1e-10
    assert np.max(np.abs(Er - Er[0])) / Er[0] < 1e-10
    assert np.max(np.abs(H - H[0])) / np.linalg.norm(H[0]) < 1e-10
    assert np.max(np.abs(np.linalg.norm(res.x[:, 6:10], axis=1) - 1)) < 1e-11
    assert np.ptp(res["p"]) > 1.0                     # gerçekten yalpalayan (asimetrik J) hareket


def test_quaternion_norm_stabilization_logistic():
    """|q|² = s için ṡ = 2·k·(1 − s)·s (lojistik): s(t) = 1 / (1 + (1/s0 − 1)·e^{−2kt}); yönelim (q̂) etkilenmez."""
    p = QuadrotorParams(quat_stab=2.0)
    quad_ = Quadrotor(p)
    x0 = quad_.initial_state((0, 0, 0), (0, 0, 0), (0.2, 0.1, -0.3), (1.0, -0.5, 0.7), np.zeros(4))
    x0b = x0.copy()
    x0b[6:10] *= 1.1
    t = np.linspace(0, 2, 21)
    ra = quad_.simulate(x0, np.zeros(4), t, method="DOP853", rtol=1e-12, atol=1e-12)
    rb = quad_.simulate(x0b, np.zeros(4), t, method="DOP853", rtol=1e-12, atol=1e-12)
    s0 = 1.21
    s = 1.0 / (1.0 + (1.0 / s0 - 1.0) * np.exp(-2 * p.quat_stab * t))
    np.testing.assert_allclose(np.sum(rb.x[:, 6:10] ** 2, axis=1), s, atol=1e-10)
    np.testing.assert_allclose(rb.info["euler"], ra.info["euler"], atol=1e-10)
    # sabit adımlı RK4 her adımda normlar
    r4 = quad_.simulate(x0b, np.zeros(4), t, method="rk4", substeps=20)
    np.testing.assert_allclose(np.linalg.norm(r4.x[1:, 6:10], axis=1), 1.0, atol=1e-15)
    np.testing.assert_allclose(r4.info["euler"], ra.info["euler"], atol=1e-7)


# ============================================================================ doğrusallaştırma
@pytest.mark.parametrize("mode", ["omega", "pwm", "virtual"])
@pytest.mark.parametrize("attitude", ["euler", "quaternion"])
@pytest.mark.parametrize("motor_dynamics", [True, False])
def test_analytic_vs_numeric_jacobian(mode, attitude, motor_dynamics):
    p = _params_A()
    quad_ = Quadrotor(p, input_mode=mode, motor_dynamics=motor_dynamics, wind=(1.0, -0.5, 0.3))
    x0, u0 = quad_.trim_hover(yaw=0.3)
    x1 = quad_.initial_state((1, 2, -3), (1, -2, 0.5), (0.3, -0.2, 1.0), (0.5, -0.4, 0.3),
                             [500, 560, 600, 540] if motor_dynamics else None)
    u1 = {"omega": np.array([510.0, 570, 590, 545]), "pwm": np.array([0.55, 0.6, 0.62, 0.58]),
          "virtual": np.array([14.0, 0.1, -0.05, 0.01])}[mode]
    for x, u in ((x0, u0), (x1, u1)):
        la = quad_.linearize(x, u, attitude=attitude)
        ln = quad_.linearize_numeric(x, u, attitude=attitude)
        assert _rel(la.A, ln.A) < 1e-9
        assert _rel(la.B, ln.B) < 1e-9
        assert _rel(la.C, ln.C) < 1e-9
        assert la.state_names == ln.state_names and la.input_names == quad_.input_names


@pytest.mark.parametrize("mode", ["omega", "virtual"])
def test_quaternion_and_euler_linearizations_agree(mode):
    """Dengede (f = 0) doğrusallaştırma koordinat seçiminden bağımsızdır: iki gösterimin giriş → çıkış transfer
    matrisleri aynı olmalı. (Denge dışı noktada dönüşümün türevi f ile çarpılan ek terim getirdiğinden aynı olmaz.)"""
    p = _params_A(config="+")
    quad_ = Quadrotor(p, input_mode=mode)
    x0, u0 = quad_.trim_hover(position=(1, 2, -3), yaw=0.7)
    le = quad_.linearize(x0, u0, attitude="euler")
    lq = quad_.linearize(x0, u0, attitude="quaternion")
    assert np.max(np.abs(lq.xdot0)) < 1e-10
    for s in (0.3j, 2.0 + 5j, 40j):
        G = le.freqresp(s)
        np.testing.assert_allclose(lq.freqresp(s), G, rtol=1e-10, atol=1e-12 * np.max(np.abs(G)))
    # norm yönündeki kip: özdeğer −2·k_s (çıkışlardan gözlenemez)
    ev = np.linalg.eigvals(lq.A)
    assert np.min(np.abs(ev + 2 * p.quat_stab)) < 1e-10


@pytest.mark.parametrize("gyro", [False, True])
@pytest.mark.parametrize("config", ["x", "+"])
def test_small_angle_model_equals_full_jacobian(gyro, config):
    p = QuadrotorParams(config=config, gyro=gyro, drag=(0.1, 0.12, 0.2), rot_drag=(0.002, 0.003, 0.004))
    quad_ = Quadrotor(p, input_mode="virtual")
    for yaw in (0.0, 0.7):
        x0, u0 = quad_.trim_hover(yaw=yaw)
        lin = quad_.linearize(x0, u0)
        G = quad_.mixer.E @ np.diag(2 * p.kT * quad_.hover_motor_speeds())
        S = np.eye(16)
        S[12:, 12:] = G
        sm = hover_small_angle_model(p, yaw=yaw)
        assert _rel(S @ lin.A @ np.linalg.inv(S), sm.A) < 1e-12
        assert _rel(S @ lin.B, sm.B) < 1e-12
    # motor dinamiksiz model de tam modelle aynı
    q2 = Quadrotor(QuadrotorParams(config=config), input_mode="virtual", motor_dynamics=False)
    x0, u0 = q2.trim_hover()
    lin = q2.linearize(x0, u0)
    sm = hover_small_angle_model(QuadrotorParams(config=config), motor_dynamics=False)
    assert _rel(lin.A, sm.A) < 1e-12 and _rel(lin.B, sm.B) < 1e-12


def test_axis_transfer_functions_match_small_angle_model():
    p = QuadrotorParams()
    sm = hover_small_angle_model(p)
    tfs = axis_transfer_functions(p)
    pairs = {"roll": ("phi", "Mx_c", 1), "pitch": ("theta", "My_c", 1), "yaw": ("psi", "Mz_c", 1),
             "altitude": ("pd", "T_c", -1)}
    for key, (yname, uname, sgn) in pairs.items():
        num, den = tfs[key]
        yi, ui = sm.output_names.index(yname), sm.input_names.index(uname)
        for s in (0.5j, 1 + 3j, 20j):
            g = sm.freqresp(s)[yi, ui] * sgn
            assert g == pytest.approx(np.polyval(num, s) / np.polyval(den, s), rel=1e-12)
    # x/θ ve y/φ: ẍ = −g·θ, ÿ = g·φ
    assert sm.A[3, 7] == pytest.approx(-p.g) and sm.A[4, 6] == pytest.approx(p.g)
    # eksen başına çift tümleyici + motor gecikmesi: özdeğerler 0 (12 adet) ve −1/τ (4 adet)
    ev = np.sort(np.linalg.eigvals(sm.A).real)
    np.testing.assert_allclose(ev[:4], -1 / p.tau_m, rtol=1e-12)
    np.testing.assert_allclose(ev[4:], 0.0, atol=1e-12)


def test_custom_hexacopter_geometry():
    ang = np.deg2rad(np.arange(6) * 60.0 + 30.0)
    pos = 0.3 * np.column_stack((np.cos(ang), np.sin(ang)))
    p = QuadrotorParams(mass=2.5, rotor_positions=pos, rotor_spins=[1, -1, 1, -1, 1, -1], gyro=True)
    quad_ = Quadrotor(p, input_mode="omega")
    assert quad_.nx == 19 and quad_.nu == 6
    x0, u0 = quad_.trim_hover()
    np.testing.assert_allclose(u0, math.sqrt(p.mass * p.g / (6 * p.kT)), rtol=1e-12)
    assert np.max(np.abs(quad_.derivatives(x0, u0))) < 1e-10
    x1 = quad_.initial_state((0, 0, 0), (1, 2, 3), (0.2, 0.1, 0.5), (0.3, 0.2, -0.1), u0 * 1.05)
    la, ln = quad_.linearize(x1, u0), quad_.linearize_numeric(x1, u0)
    assert _rel(la.A, ln.A) < 1e-9 and _rel(la.B, ln.B) < 1e-9


# ============================================================================ karıştırıcı
def _quad_mixer():
    quad_ = Quadrotor(QuadrotorParams(), input_mode="virtual")
    return quad_, quad_.mixer


def test_mixer_inverse_and_effectiveness():
    quad_, mx = _quad_mixer()
    np.testing.assert_allclose(mx.E @ mx.P, np.eye(4), atol=1e-12)
    v = np.array([15.0, 0.2, -0.1, 0.01])
    for method in ("px4", "px4_airmode", "scale", "clip", "none"):
        t, info = mx.allocate(v, method)
        assert not info["saturated"]
        np.testing.assert_allclose(mx.virtual(t), v, rtol=1e-12, atol=1e-14)
    # ters karıştırıcı fiziksel momentlerle aynı (r_i × F_i)
    t = np.array([3.0, 4.0, 5.0, 2.0])
    M = sum(np.cross(quad_.pos[i], [0, 0, -t[i]]) for i in range(4))
    np.testing.assert_allclose(mx.virtual(t)[1:3], M[:2], atol=1e-15)
    om = np.sqrt(t / quad_.params.kT)
    assert mx.virtual(t)[3] == pytest.approx(quad_.params.kQ * float(quad_.spins @ om ** 2), rel=1e-12)


def test_px4_yaw_is_cut_first():
    quad_, mx = _quad_mixer()
    p = quad_.params
    v = np.array([p.mass * p.g, 0.3, -0.2, 0.8])          # yaw talebi motor aralığını aşar
    t, info = mx.allocate(v, "px4")
    assert info["saturated"]
    a = mx.virtual(t)
    assert np.all(t >= mx.t_min - 1e-12) and np.all(t <= mx.t_max + 1e-12)
    np.testing.assert_allclose(a[1:3], v[1:3], rtol=1e-12)     # roll / pitch korunur
    assert 0 < a[3] < v[3]                                     # yaw kesilir (aynı işaret)
    assert a[0] == pytest.approx(v[0], rel=1e-12)              # itki korunur (bu durumda gerek yok)


def test_px4_thrust_reduced_before_roll_at_top():
    quad_, mx = _quad_mixer()
    tmax = float(mx.t_max[0])
    v = np.array([4 * tmax * 0.98, 0.8, 0.0, 0.0])         # tam gaz + roll: üst doygunluk
    t, info = mx.allocate(v, "px4")
    a = mx.virtual(t)
    assert info["saturated"]
    assert a[1] == pytest.approx(v[1], rel=1e-12) and a[0] < v[0]      # roll korunur, itki azalır
    assert np.max(t) == pytest.approx(tmax, rel=1e-12)


def test_px4_airmode_raises_thrust_at_bottom():
    quad_, mx = _quad_mixer()
    v = np.array([0.5, 0.4, 0.0, 0.0])                      # düşük gaz + roll: alt doygunluk
    t0, _ = mx.allocate(v, "px4")
    t1, _ = mx.allocate(v, "px4_airmode")
    a0, a1 = mx.virtual(t0), mx.virtual(t1)
    assert a0[0] <= v[0] + 1e-12 and abs(a0[1]) < v[1]      # airmode kapalı: itki artırılmaz, roll kısılır
    assert a1[1] == pytest.approx(v[1], rel=1e-12) and a1[0] > v[0]   # airmode: roll korunur, itki artar
    for t in (t0, t1):
        assert np.all(t >= mx.t_min - 1e-12) and np.all(t <= mx.t_max + 1e-12)


def test_scale_keeps_moment_direction():
    _, mx = _quad_mixer()
    v = np.array([10.0, 3.0, -2.0, 0.5])
    t, info = mx.allocate(v, "scale")
    a = mx.virtual(t)
    assert info["saturated"]
    k = a[1:] / v[1:]
    np.testing.assert_allclose(k, k[0], rtol=1e-10)
    assert 0 < k[0] < 1
    assert np.all(t >= mx.t_min - 1e-12) and np.all(t <= mx.t_max + 1e-12)


def test_mixer_rejects_bad_input():
    _, mx = _quad_mixer()
    with pytest.raises(ControlError):
        mx.allocate([1.0, 2.0], "px4")
    with pytest.raises(ControlError):
        mx.allocate([1.0, 0, 0, 0], "bilinmeyen")
    with pytest.raises(ControlError):
        Mixer(np.ones((4, 4)), 0.0, 1.0)


def test_input_validation():
    quad_ = Quadrotor()
    with pytest.raises(ControlError):
        quad_.derivatives(np.zeros(5), np.zeros(4))
    with pytest.raises(ControlError):
        quad_.derivatives(quad_.initial_state(), np.zeros(3))
    with pytest.raises(ControlError):
        Quadrotor(input_mode="rpm")
    with pytest.raises(ControlError):
        Quadrotor(QuadrotorParams(inertia=np.diag([1.0, -1.0, 1.0])))


def test_numjac_helper_accuracy():
    f = lambda x: np.array([math.sin(x[0]) * math.exp(x[1]), x[0] ** 3 * x[1]])  # noqa: E731
    x = np.array([0.7, -0.3])
    J = np.array([[math.cos(0.7) * math.exp(-0.3), math.sin(0.7) * math.exp(-0.3)], [3 * 0.49 * -0.3, 0.343]])
    np.testing.assert_allclose(numjac(f, x), J, rtol=1e-10)


def test_output_vector_interface():
    quad_ = Quadrotor(QuadrotorParams())
    x = quad_.initial_state((1, 2, 3), (0.1, 0.2, 0.3), (0.3, -0.2, 1.0), (0.5, -0.4, 0.3), [500, 560, 600, 540])
    y = quad_.output(x)
    assert quad_.output_names == quad_.euler_state_names and y.shape == (16,)
    np.testing.assert_allclose(y[6:9], (0.3, -0.2, 1.0), atol=1e-14)
    np.testing.assert_allclose(quad_.body_velocity(x), rotmat_from_euler(0.3, -0.2, 1.0).T @ [0.1, 0.2, 0.3],
                               atol=1e-15)


def test_fast_derivatives_equal_reference_implementation():
    rng = np.random.default_rng(0)
    J = np.array([[0.03, 0.001, -0.002], [0.001, 0.035, 0.0005], [-0.002, 0.0005, 0.06]])
    for gyro in (False, True):
        for md in (True, False):
            for mode in ("omega", "virtual"):
                q = Quadrotor(QuadrotorParams(gyro=gyro, inertia=J, drag=(0.1, 0.12, 0.2), rot_drag=(0.002, 0.003,
                                                                                                    0.004)),
                              input_mode=mode, motor_dynamics=md, wind=(1.0, -0.5, 0.3))
                for _ in range(10):
                    x = q.initial_state(rng.normal(size=3), rng.normal(size=3), rng.normal(size=3), rng.normal(size=3),
                                        500 + 50 * rng.normal(size=4) if md else None)
                    x[6:10] *= 1 + 0.1 * rng.normal()
                    u = 500 + 50 * rng.normal(size=4) if mode == "omega" else \
                        np.array([14.0, 0.1, -0.05, 0.01]) + 0.1 * rng.normal(size=4)
                    assert _rel(q.derivatives(x, u, 0.3), q.derivatives_reference(x, u, 0.3)) < 1e-14
