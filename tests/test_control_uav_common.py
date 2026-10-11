"""İHA ortak araçları: döndürme / kuaterniyon kuralları (NED, ZYX), analitik türev yardımcıları, benzetim, doğrusal
model kabı, motor / pervane yardımcıları ve algılayıcı modelleri. (MATLAB Aerospace / UAV Toolbox karşılaştırması
test_control_uav_quadrotor.py içinde.)"""
from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.linalg import expm

from spt.control.errors import ControlError
from spt.control.uav import common as cm


EULERS = [(0.2, -0.1, 0.5), (-1.0, 0.7, 2.5), (0.4, 1.2, -2.9), (3.0, -1.4, 0.1)]


def test_rotation_matrix_conventions():
    for roll, pitch, yaw in EULERS:
        R = cm.rotmat_from_euler(roll, pitch, yaw)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-15)
        assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-14)
        np.testing.assert_allclose(R, cm.rot_z(yaw) @ cm.rot_y(pitch) @ cm.rot_x(roll), atol=0)
        q = cm.quat_from_euler(roll, pitch, yaw)
        np.testing.assert_allclose(cm.rotmat_from_quat(q), R, atol=1e-15)
        np.testing.assert_allclose(cm.rotmat_from_quat(3.7 * q), R, atol=1e-15)        # normdan bağımsız
        np.testing.assert_allclose(cm.euler_from_quat(q), (roll, pitch, yaw), atol=1e-12)
        np.testing.assert_allclose(cm.euler_from_rotmat(R), (roll, pitch, yaw), atol=1e-12)
        np.testing.assert_allclose(cm.euler_from_quat_array(np.array([q, 2 * q]))[1], (roll, pitch, yaw), atol=1e-12)
    # NED: pozitif yaw burnu doğuya (sağa) çevirir, pozitif pitch burnu kaldırır (gövde x → yukarı = −z)
    np.testing.assert_allclose(cm.rotmat_from_euler(0, 0, math.pi / 2) @ [1, 0, 0], [0, 1, 0], atol=1e-15)
    np.testing.assert_allclose(cm.rotmat_from_euler(0, 0.3, 0) @ [1, 0, 0], [math.cos(0.3), 0, -math.sin(0.3)],
                               atol=1e-15)
    # pozitif roll sağ kanadı (gövde +y) aşağı (+z) indirir
    assert (cm.rotmat_from_euler(0.3, 0, 0) @ [0, 1, 0])[2] > 0


def test_consistency_with_fusion_module():
    from spt.dsp import fusion
    for roll, pitch, yaw in EULERS:
        q = cm.quat_from_euler(roll, pitch, yaw)
        np.testing.assert_allclose(q, fusion.quat_from_euler(roll, pitch, yaw), atol=0)
        np.testing.assert_allclose(cm.rotmat_from_quat(q), fusion.rotmat(q), atol=1e-15)
        np.testing.assert_allclose(cm.euler_from_quat(q), np.array(fusion.euler_from_quat(q)), atol=1e-15)
    a, b = np.array([0.3, 0.5, -0.2, 0.7]), np.array([0.9, -0.1, 0.3, 0.2])
    np.testing.assert_allclose(cm.quat_mul(a, b), fusion.quat_mul(a, b), atol=0)


def test_kinematics_euler_rates_and_quaternion():
    """Ṙ = R·[ω×] (ω gövdede): hem η̇ = W(η)·ω hem q̇ = ½·q⊗[0, ω] aynı Ṙ'yi vermeli."""
    w = np.array([0.4, -0.7, 1.1])
    h = 1e-6
    for eta in EULERS[:3]:
        eta = np.array(eta)
        R = cm.rotmat_from_euler(*eta)
        etad = cm.euler_rate_matrix(eta[0], eta[1]) @ w
        Rd = (cm.rotmat_from_euler(*(eta + h * etad)) - cm.rotmat_from_euler(*(eta - h * etad))) / (2 * h)
        np.testing.assert_allclose(R.T @ Rd, cm.skew(w), atol=1e-8)
        q = cm.quat_from_euler(*eta)
        qd = 0.5 * cm.quat_mul(q, np.concatenate(([0.0], w)))
        np.testing.assert_allclose(qd, 0.5 * cm.quat_omega_matrix(w) @ q, atol=1e-15)
        np.testing.assert_allclose(qd, 0.5 * cm.quat_xi_matrix(q) @ w, atol=1e-15)
        Rq = (cm.rotmat_from_quat(q + h * qd) - cm.rotmat_from_quat(q - h * qd)) / (2 * h)
        np.testing.assert_allclose(R.T @ Rq, cm.skew(w), atol=1e-8)


def test_analytic_derivative_helpers():
    a = np.array([0.3, -1.2, 2.0])
    for q in (cm.quat_from_euler(0.2, -0.1, 0.5), np.array([0.9, 0.3, -0.4, 0.2]), np.array([-0.5, 1.1, 0.2, 0.7])):
        J1 = cm.numjac(lambda qq: cm.rotmat_from_quat(qq) @ a, q)
        np.testing.assert_allclose(cm.d_rotmat_quat_times(q, a), J1, atol=1e-10)
        J2 = cm.numjac(lambda qq: cm.rotmat_from_quat(qq).T @ a, q)
        np.testing.assert_allclose(cm.d_rotmat_T_quat_times(q, a), J2, atol=1e-10)
        J3 = cm.numjac(lambda qq: np.array(cm.euler_from_quat(qq)), q)
        np.testing.assert_allclose(cm.d_euler_d_quat(q), J3, atol=1e-10)
        np.testing.assert_allclose(cm.d_euler_d_quat(q) @ q, 0, atol=1e-14)        # norm yönünde türev yok


def test_small_helpers():
    v, w = np.array([1.0, -2.0, 0.5]), np.array([0.3, 0.7, -1.1])
    np.testing.assert_allclose(cm.skew(v) @ w, np.cross(v, w), atol=1e-15)
    np.testing.assert_allclose(cm.wrap_angle([3 * math.pi, -math.pi, 0.5]), [math.pi, math.pi, 0.5], atol=1e-15)
    assert cm.wrap_angle(-3 * math.pi / 2) == pytest.approx(math.pi / 2)
    with pytest.raises(ControlError):
        cm.quat_normalize([0, 0, 0, 0])
    kT, kQ = cm.prop_coefficients(0.1, 0.005, 0.254, 1.225)
    assert kT == pytest.approx(0.1 * 1.225 * 0.254 ** 4 / (4 * math.pi ** 2), rel=1e-15)
    assert kQ == pytest.approx(0.005 * 1.225 * 0.254 ** 5 / (4 * math.pi ** 2), rel=1e-15)
    om = cm.hover_speed(1.4, 1.105e-5, 4)
    assert 4 * 1.105e-5 * om ** 2 == pytest.approx(1.4 * cm.GRAVITY, rel=1e-15)
    u = np.array([0.2, 0.5, 0.9])
    np.testing.assert_allclose(cm.speed_to_throttle(cm.throttle_to_speed(u, 1148, -141.4), 1148, -141.4), u)
    np.testing.assert_allclose(cm.throttle_to_speed([0.0, 1.5], 1148, -141.4, 0, 1006.6), [0, 1006.6])


# ============================================================================ benzetim
class _Linear:
    state_names = ("x1", "x2")
    input_names = ("u",)
    nx, nu = 2, 1
    A = np.array([[0.0, 1.0], [-4.0, -0.6]])
    B = np.array([[0.0], [2.0]])

    def derivatives(self, x, u, t=0.0):
        return self.A @ x + self.B @ np.atleast_1d(u)


def test_simulate_against_exact_solutions():
    m = _Linear()
    x0 = np.array([1.0, -0.5])
    t = np.linspace(0, 3, 31)
    exact = np.array([expm(m.A * tk) @ x0 for tk in t])
    for method, kw in (("RK45", dict(rtol=1e-11, atol=1e-12)), ("DOP853", dict(rtol=1e-12, atol=1e-12)),
                       ("rk4", dict(substeps=10))):
        res = cm.simulate(m, x0, [0.0], t, method=method, **kw)
        np.testing.assert_allclose(res.x, exact, atol=1e-8 if method != "rk4" else 1e-6)
        np.testing.assert_allclose(res["x1"], res.x[:, 0])
    # dizi giriş = sıfırıncı derece tutucu: ayrık tam çözüm (Ad, Bd) ile aynı
    rng = np.random.default_rng(0)
    U = rng.standard_normal((t.size, 1))
    dt = t[1] - t[0]
    M = expm(np.block([[m.A, m.B], [np.zeros((1, 3))]]) * dt)
    Ad, Bd = M[:2, :2], M[:2, 2:]
    xd = [x0]
    for k in range(t.size - 1):
        xd.append(Ad @ xd[-1] + Bd @ U[k])
    for method in ("RK45", "rk4"):
        res = cm.simulate(m, x0, U, t, method=method, rtol=1e-11, atol=1e-12, substeps=40)
        np.testing.assert_allclose(res.x, np.array(xd), atol=1e-9 if method == "RK45" else 1e-7)
    # çağrılabilir giriş ve ZOH tutucu (rk4)
    res = cm.simulate(m, x0, lambda tt, xx: [math.sin(tt)], t, method="rk4", substeps=20)
    res_z = cm.simulate(m, x0, lambda tt, xx: [math.sin(tt)], t, method="rk4", substeps=1, hold="zoh")
    assert np.max(np.abs(res.x - res_z.x)) > 1e-4                      # ZOH farklı (beklenen)
    with pytest.raises(ControlError):
        cm.simulate(m, x0, [0.0], [0.0])
    with pytest.raises(ControlError):
        cm.simulate(m, [1.0], [0.0], t)
    with pytest.raises(ControlError):
        cm.simulate(m, x0, np.zeros((3, 1)), t)


def test_linear_model_container():
    A = np.array([[0, 1, 0], [-2, -3, 1], [0, 0, -5.0]])
    B = np.array([[0.0], [1.0], [2.0]])
    lm = cm.LinearModel(A, B, np.eye(3), np.zeros((3, 1)), ("a", "b", "c"), ("u",), ("a", "b", "c"))
    sub = lm.subsystem(["a", "b"])
    np.testing.assert_array_equal(sub.A, A[:2, :2])
    assert sub.state_names == ("a", "b") and sub.output_names == ("a", "b")
    s = 1.5j
    np.testing.assert_allclose(lm.freqresp(s), np.linalg.solve(s * np.eye(3) - A, B))
    with pytest.raises(ControlError):
        lm.subsystem(["z"])
    import importlib.util
    if importlib.util.find_spec("spt.control.lti") is None:       # LTI çekirdeği başka ajanca yazılıyor
        with pytest.raises(ControlError):
            lm.to_ss()
    else:
        sys = lm.to_ss()
        np.testing.assert_allclose(np.asarray(sys.A), A)
        np.testing.assert_allclose(np.asarray(sys.B), B)
        assert tuple(sys.state_names) == ("a", "b", "c") and tuple(sys.input_names) == ("u",)
        assert sys.Ts == 0.0


# ============================================================================ algılayıcılar
def test_imu_model():
    t = np.arange(0, 20, 0.005)
    n = t.size
    w = np.column_stack((np.sin(t), np.cos(t), 0.1 * t))
    f = np.tile([0.0, 0.0, -cm.GRAVITY], (n, 1))
    g0, a0 = cm.ImuModel().measure(t, w, f)
    np.testing.assert_array_equal(g0, w)
    np.testing.assert_array_equal(a0, f)
    imu = cm.ImuModel(gyro_noise_density=0.01, accel_noise_density=0.2, gyro_bias=(0.01, -0.02, 0.03), seed=4)
    g1, a1 = imu.measure(t, w, f)
    g2, a2 = cm.ImuModel(gyro_noise_density=0.01, accel_noise_density=0.2, gyro_bias=(0.01, -0.02, 0.03),
                         seed=4).measure(t, w, f)
    np.testing.assert_array_equal(g1, g2)
    np.testing.assert_allclose((g1 - w).mean(axis=0), [0.01, -0.02, 0.03], atol=0.01)   # 4σ (σ_ort ≈ 0.0022)
    np.testing.assert_allclose((g1 - w).std(axis=0), 0.01 / math.sqrt(0.005), rtol=0.05)
    np.testing.assert_allclose((a1 - f).std(axis=0), 0.2 / math.sqrt(0.005), rtol=0.05)
    gw, _ = cm.ImuModel(gyro_bias_walk=0.002, seed=1).measure(t, w, f)
    inc = np.diff(gw - w, axis=0)
    np.testing.assert_allclose(inc.std(axis=0), 0.002 * math.sqrt(0.005), rtol=0.05)
    with pytest.raises(ControlError):
        imu.measure(np.array([0.0, 0.1, 0.3]), w[:3], f[:3])


def test_gps_barometer_magnetometer():
    t = np.arange(0, 10.0001, 0.01)
    pos = np.column_stack((3 * t, -t ** 2, -100 + 0.5 * t))
    vel = np.column_stack((3 + 0 * t, -2 * t, 0.5 + 0 * t))
    tg, pg, vg = cm.gps(t, pos, vel, rate=5.0, delay=0.2, sigma_pos=(0, 0, 0), sigma_vel=0.0, seed=1)
    np.testing.assert_allclose(np.diff(tg), 0.2, atol=1e-12)
    src = tg - 0.2
    np.testing.assert_allclose(pg, np.column_stack((3 * src, -src ** 2, -100 + 0.5 * src)), atol=1e-3)
    np.testing.assert_allclose(vg[:, 1], -2 * src, atol=1e-9)
    tg, pg, _ = cm.gps(t, pos, vel, rate=1.0, sigma_pos=(0.21, 0.21, 0.4), seed=2)
    err = pg - np.column_stack([np.interp(tg, t, pos[:, i]) for i in range(3)])
    assert np.all(err[0] == 0)                                     # Gauss–Markov sıfırdan başlar
    a = math.exp(-1.0 / 1100.0)
    eta = err[1:] - a * err[:-1]
    assert eta.shape == (10, 3) and np.all(np.abs(eta) < 5 * np.array([0.21, 0.21, 0.4]))
    h = -pos[:, 2]
    np.testing.assert_array_equal(cm.barometer(t, h), h)
    hb = cm.barometer(t, h, sigma=0.5, bias=1.0, seed=3)
    assert (hb - h).mean() == pytest.approx(1.0, abs=0.05) and (hb - h).std() == pytest.approx(0.5, rel=0.06)
    mf = np.array([0.22, 0.02, 0.43])
    np.testing.assert_allclose(cm.magnetometer(cm.quat_from_euler(0, 0, 0), mf)[0], mf, atol=1e-15)
    m90 = cm.magnetometer(cm.quat_from_euler(0, 0, math.pi / 2), mf)[0]
    np.testing.assert_allclose(m90, [0.02, -0.22, 0.43], atol=1e-15)              # burun doğuda: kuzey solda
