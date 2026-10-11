"""Sensör füzyonu: kurallar (kuaterniyon / Euler / yerçekimi), algoritmaların doğruluğu, blok = tek sefer,
füzyon motoru (ayrı süreçler) ve örnek kayıt."""
import functools
import math
import time
from pathlib import Path

import numpy as np
import pytest

from spt.core.fusion_engine import FusionEngine
from spt.dsp import fusion_generic as fg
from spt.core.local_ring import StaticRing
from spt.core.snapshot import Snapshot
from spt.dsp import fusion as fu
from spt.protocol.format import SAMPLE_TYPES_BY_NAME, ChannelInfo, Descriptor, Layout

SIM = fu.simulate_imu(duration=40.0, fs=200.0, seed=7)
DT = 1 / SIM.fs
SKIP = int(5 * SIM.fs)   # ilk 5 s: yakınsama
RNG = np.random.default_rng(0)


# ============================================================================ kurallar
def test_quaternion_euler_round_trip_and_rotation_matrix():
    angles = RNG.uniform([-np.pi, -1.4, -np.pi], [np.pi, 1.4, np.pi], (500, 3))
    q = fu.quat_from_euler(*angles.T)
    assert np.allclose(np.linalg.norm(q, axis=1), 1.0)
    back = np.column_stack(fu.euler_from_quat(q))
    assert np.allclose(fu.wrap(back - angles), 0, atol=1e-9)
    # R = Rz(yaw)·Ry(pitch)·Rx(roll)
    r, p, y = angles[0]
    rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    assert np.allclose(fu.rotmat(q[0]), rz @ ry @ rx)
    assert np.allclose(fu.gravity_body(q), fu.rotmat(q)[:, 2, :])  # Rᵀ·ẑ = R'nin 3. satırı


def test_quaternion_exp_log_and_kinematics():
    v = RNG.normal(0, 0.5, (100, 3))
    assert np.allclose(fu.quat_log(fu.quat_exp(v)), v, atol=1e-12)
    assert np.allclose(fu.quat_exp(np.zeros(3)), [1, 0, 0, 0])
    # Sabit gövde hızıyla döndürme: q ⊗ exp(ω·t), dünya çerçevesinde gövde ekseni etrafında dönme
    q0 = fu.quat_from_euler(0.2, -0.1, 0.5)
    omega = np.array([0.0, 0.0, 0.3])
    q1 = fu.quat_mul(q0, fu.quat_exp(omega * 2.0))
    assert np.allclose(fu.rotmat(q1), fu.rotmat(q0) @ fu.rotmat(fu.quat_exp(omega * 2.0)))


def test_accelerometer_and_heading_formulas_match_truth_without_noise():
    sim = fu.simulate_imu(duration=20.0, fs=100.0, seed=2, gyro_noise=0, acc_noise=0, mag_noise=0, linear_acc=0)
    roll, pitch = fu.accel_roll_pitch(sim.acc)
    assert np.allclose(roll, sim.truth[:, 0], atol=1e-9) and np.allclose(pitch, sim.truth[:, 1], atol=1e-9)
    heading = fu.tilt_compensated_heading(sim.mag, sim.truth[:, 0], sim.truth[:, 1])
    assert np.allclose(fu.wrap(heading - sim.truth[:, 2]), 0, atol=1e-9)


def test_simulated_gyro_matches_attitude_kinematics():
    """Gürültüsüz jiroskopu tam integral almak gerçek yönelimi vermeli (simülasyonun kendisi doğru mu?)."""
    sim = fu.simulate_imu(duration=10.0, fs=2000.0, seed=3, gyro_bias=(0, 0, 0), gyro_noise=0)
    q = fu.quat_from_euler(*sim.truth[0])
    for k in range(len(sim.gyro) - 1):  # orta nokta kuralı
        q = fu.normalize(fu.quat_mul(q, fu.quat_exp(0.5 * (sim.gyro[k] + sim.gyro[k + 1]) / sim.fs)))
    final = np.array(fu.euler_from_quat(q))
    assert np.max(np.abs(fu.wrap(final - sim.truth[-1]))) < 1e-4


def test_madgwick_corrective_step_is_gradient_of_objective():
    """Madgwick'in kapalı biçim düzeltme adımı = amaç fonksiyonunun (½‖f‖²) sayısal gradyanı."""
    est = fu.Madgwick(beta=1.0)
    q0 = fu.normalize(np.array([0.9, 0.2, -0.3, 0.25]))
    a = np.array([0.3, -0.2, 9.5])
    m = np.array([20.0, 5.0, -40.0])

    def objective(q):
        w, x, y, z = q
        an, mn = a / np.linalg.norm(a), m / np.linalg.norm(m)
        rq = fu.rotmat(fu.normalize(q))
        h = rq @ mn
        b = np.array([math.hypot(h[0], h[1]), 0.0, h[2]])
        f = np.r_[2 * (x * z - w * y) - an[0], 2 * (w * x + y * z) - an[1], 2 * (0.5 - x * x - y * y) - an[2],
                  2 * b[0] * (0.5 - y * y - z * z) + 2 * b[2] * (x * z - w * y) - mn[0],
                  2 * b[0] * (x * y - w * z) + 2 * b[2] * (w * x + y * z) - mn[1],
                  2 * b[0] * (w * y + x * z) + 2 * b[2] * (0.5 - x * x - y * y) - mn[2]]
        return 0.5 * f @ f, b

    _, b = objective(q0)
    eps = 1e-7
    grad = np.zeros(4)
    for i in range(4):  # b'yi sabit tutarak (Madgwick'te referans her adımda tahminden alınır)
        def f_fixed_b(q):
            w, x, y, z = q
            an, mn = a / np.linalg.norm(a), m / np.linalg.norm(m)
            f = np.r_[2 * (x * z - w * y) - an[0], 2 * (w * x + y * z) - an[1], 2 * (0.5 - x * x - y * y) - an[2],
                      2 * b[0] * (0.5 - y * y - z * z) + 2 * b[2] * (x * z - w * y) - mn[0],
                      2 * b[0] * (x * y - w * z) + 2 * b[2] * (w * x + y * z) - mn[1],
                      2 * b[0] * (w * y + x * z) + 2 * b[2] * (0.5 - x * x - y * y) - mn[2]]
            return 0.5 * f @ f
        d = np.zeros(4)
        d[i] = eps
        grad[i] = (f_fixed_b(q0 + d) - f_fixed_b(q0 - d)) / (2 * eps)
    est.reset(q0)
    dt = 1e-3
    q1 = est.run(np.zeros((1, 3)), a[None], m[None], dt)[0]
    step = (q1 - q0) / dt  # normalizasyondan önceki adım ≈ −β·∇f/‖∇f‖ (küçük dt'de)
    expected = -grad / np.linalg.norm(grad)
    tangent = expected - (expected @ q0) * q0  # birim küre üzerindeki bileşenler karşılaştırılır
    measured = step - (step @ q0) * q0
    assert np.allclose(measured, tangent, atol=2e-3)


# ============================================================================ doğruluk
# Derece cinsinden RMSE üst sınırları (5 s yakınsamadan sonra, doğrusal ivme ve gürültü varken)
LIMITS_MAG = {"mahony": (1.0, 1.0, 1.5), "madgwick": (1.2, 1.2, 1.5), "kalman": (1.2, 1.2, 1.6),
              "ekf": (0.8, 0.8, 0.9), "ukf": (0.8, 0.8, 0.9), "pf": (1.2, 1.2, 1.6)}


@functools.lru_cache(maxsize=None)
def _run(key, mag=True, block=50, samples=None):
    """Aynı çalıştırma birden çok testte kullanılır (UKF / parçacık filtresi pahalı)."""
    n = samples or len(SIM.gyro)
    est = fu.make_estimator(key)
    parts = [est.run(SIM.gyro[i:min(i + block, n)], SIM.acc[i:min(i + block, n)],
                     SIM.mag[i:min(i + block, n)] if mag else None, DT) for i in range(0, n, block)]
    return est, np.concatenate(parts)


@pytest.mark.parametrize("key", list(fu.ALGORITHMS))
def test_algorithms_track_true_orientation_with_magnetometer(key):
    est, q = _run(key)
    angles = fu.euler_from_quat(q)
    errors = [fu.rms(fu.angle_error_deg(a[SKIP:], SIM.truth[SKIP:, i])) for i, a in enumerate(angles)]
    assert all(e < lim for e, lim in zip(errors, LIMITS_MAG[key])), errors
    assert np.allclose(np.linalg.norm(q, axis=1), 1.0, atol=1e-9)


@pytest.mark.parametrize("key", list(fu.ALGORITHMS))
def test_algorithms_without_magnetometer_keep_roll_and_pitch(key):
    _, q = _run(key, mag=False)
    roll, pitch, _ = fu.euler_from_quat(q)
    assert fu.rms(fu.angle_error_deg(roll[SKIP:], SIM.truth[SKIP:, 0])) < 1.3
    assert fu.rms(fu.angle_error_deg(pitch[SKIP:], SIM.truth[SKIP:, 1])) < 1.3


@pytest.mark.parametrize("key", ["mahony", "kalman", "ekf", "ukf"])
def test_gyro_bias_is_estimated(key):
    est, _ = _run(key)
    tol = 0.003 if key in ("ekf", "ukf") else 0.006  # rad/s
    assert np.allclose(est.bias, SIM.gyro_bias, atol=tol), est.bias


@pytest.mark.parametrize("key", list(fu.ALGORITHMS))
def test_block_processing_equals_one_shot(key):
    _, whole = _run(key, block=2000, samples=2000)
    _, blocks = _run(key, block=37, samples=2000)
    assert np.max(np.abs(blocks - whole)) < 1e-9


def test_fusion_beats_raw_accelerometer_angle():
    """Füzyon, ham (filtresiz) ivmeölçer açısından daha doğru olmalı (doğrusal ivme ve gürültü varken)."""
    raw_roll, _ = fu.accel_roll_pitch(SIM.acc)
    raw = fu.rms(fu.angle_error_deg(raw_roll[SKIP:], SIM.truth[SKIP:, 0]))
    _, q = _run("ekf")
    fused = fu.rms(fu.angle_error_deg(fu.euler_from_quat(q)[0][SKIP:], SIM.truth[SKIP:, 0]))
    assert fused < 0.6 * raw


# ============================================================================ eşleme ve motor
def _imu_snapshot(sim, gaps=()):
    deg = 180 / np.pi
    ch = ([ChannelInfo(f"jiro_{a}", "°/s") for a in "xyz"] + [ChannelInfo(f"ivme_{a}", "m/s²") for a in "xyz"]
          + [ChannelInfo(f"mag_{a}", "µT") for a in "xyz"] + [ChannelInfo(f"ref_{n}", "°") for n in ("roll", "pitch", "yaw")])
    data = np.column_stack((sim.gyro * deg, sim.acc, sim.mag, sim.truth * deg)).astype(np.float64)
    desc = Descriptor(1, 1, sim.fs, SAMPLE_TYPES_BY_NAME["float64"], Layout.INTERLEAVED, "imu", tuple(ch))
    return Snapshot(1, "imu", "test", desc, 0.0, StaticRing(data, 0, list(gaps)))


def test_auto_map_recognises_turkish_and_english_names():
    desc = _imu_snapshot(fu.simulate_imu(duration=1.0)).desc
    m = fg.auto_map("orientation", desc)
    assert m.inputs == {"gyro": (0, 1, 2), "acc": (3, 4, 5), "mag": (6, 7, 8)}
    assert m.reference == {"roll": 9, "pitch": 10, "yaw": 11} and m.units["gyro"] == "°/s"
    eng = Descriptor(1, 1, 100.0, SAMPLE_TYPES_BY_NAME["float32"], Layout.INTERLEAVED, "x",
                     tuple(ChannelInfo(n, u) for n, u in [("gx", "rad/s"), ("gy", "rad/s"), ("gz", "rad/s"),
                                                          ("acc_x", "g"), ("acc_y", "g"), ("acc_z", "g")]))
    m = fg.auto_map("orientation", eng)
    assert m.inputs == {"gyro": (0, 1, 2), "acc": (3, 4, 5)} and m.units["gyro"] == "rad/s"
    assert fg.auto_map("orientation", Descriptor(1, 1, 1.0, SAMPLE_TYPES_BY_NAME["float32"], Layout.INTERLEAVED, "x",
                               (ChannelInfo("sinyal"),))) is None


def _wait(engine, timeout=60.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        engine.pump()
        if engine.progress() >= 1.0 and not engine.busy:
            return True
        time.sleep(0.01)
    return False


def test_engine_runs_algorithms_in_processes_and_matches_direct_run():
    sim = fu.simulate_imu(duration=12.0, fs=200.0, seed=11)
    snap = _imu_snapshot(sim)
    handle = snap.handle()
    mapping = fg.auto_map("orientation", snap.desc)
    engine = FusionEngine(handle, "orientation", mapping, {}, {"mahony": {}, "ekf": {}})
    try:
        assert _wait(engine)
        n = snap.samples
        direct = fu.make_estimator("ekf").run(sim.gyro, sim.acc, sim.mag, 1 / sim.fs)
        expected = np.column_stack(fu.euler_from_quat(direct))
        got = engine.read("ekf", 0, n)
        assert np.max(np.abs(fu.wrap(got - expected))) < 1e-5  # float32 sonuç
        ref = engine.reference(0, n)
        assert np.allclose(ref, sim.truth, atol=1e-6)
        assert engine.algos["ekf"].cost_us > 0
    finally:
        engine.stop()


def test_engine_restarts_after_gap_and_leaves_nan():
    sim = fu.simulate_imu(duration=10.0, fs=200.0, seed=12)
    snap = _imu_snapshot(sim, gaps=[(800, 900)])
    snap.ring.data[800:900] = np.nan
    engine = FusionEngine(snap.handle(), "orientation", fg.auto_map("orientation", snap.desc), {}, {"madgwick": {}})
    try:
        assert _wait(engine)
        got = engine.read("madgwick", 0, snap.samples)
        assert np.all(np.isnan(got[800:900])) and np.all(np.isfinite(got[:800])) and np.all(np.isfinite(got[900:]))
        err = fu.angle_error_deg(got[905:1000, 0], sim.truth[905:1000, 0])
        assert np.max(np.abs(err)) < 3.0  # kayıptan sonra ivmeölçerden temiz başlangıç
    finally:
        engine.stop()


def test_reference_sample_recording_exists_and_maps():
    path = Path(__file__).resolve().parents[1] / "samples_signal_data" / "10_imu_9eksen_yonelim_referansli_200Hz.sptr"
    assert path.exists()
    from spt.protocol.recording import SptrReader
    from spt.protocol.parser import check_frame, unpack_descriptor
    from spt.protocol.format import FrameType, HEADER
    with SptrReader(path) as reader:
        offsets, _, lengths = reader.index()
        for off, n in zip(offsets.tolist(), lengths.tolist()):
            ftype, flags, flen, sid, rev, _, _ = check_frame(reader.buffer, off, off + n)
            if ftype == FrameType.DESCRIPTOR:
                desc = unpack_descriptor(reader.buffer, off + HEADER.size, off + flen, sid, rev)
                break
    m = fg.auto_map("orientation", desc)
    assert m is not None and "mag" in m.inputs and m.reference and m.units["gyro"] == "°/s"


# ============================================================================ füzyon denetimi regresyonları
def _static_imu(roll, pitch, yaw, seconds=30.0, fs=100.0, disturb=None, seed=3):
    """Sabit yönelim; isteğe bağlı olarak [t0, t1) aralığında dünyaya bağlı 40 µT manyetik bozulma."""
    rng = np.random.default_rng(seed)
    d2r = np.pi / 180
    q = fu.quat_from_euler(roll * d2r, pitch * d2r, yaw * d2r)
    Rt = fu.rotmat(q).T
    n = int(seconds * fs)
    gyro = np.array([0.01, -0.008, 0.006]) + 0.003 * rng.standard_normal((n, 3))
    acc = Rt @ np.array([0.0, 0.0, fu.GRAVITY]) + 0.03 * rng.standard_normal((n, 3))
    inc = 60 * d2r
    mag = Rt @ (50 * np.array([np.cos(inc), 0.0, -np.sin(inc)])) + 0.3 * rng.standard_normal((n, 3))
    t = np.arange(n) / fs
    if disturb:
        mag[(t >= disturb[0]) & (t < disturb[1])] += Rt @ np.array([0.0, 40.0, 0.0])
    return q, gyro, acc, mag, t, fs


def _tilt_heading_errors(q_est, q_true):
    g_e, g_t = fu.gravity_body(q_est), fu.gravity_body(q_true)
    tilt = np.degrees(np.arccos(np.clip((g_e * g_t).sum(axis=1), -1.0, 1.0)))
    r = fu.quat_mul(q_est, fu.quat_conj(q_true))
    return tilt, np.degrees(fu.wrap(2 * np.arctan2(r[:, 3], r[:, 0])))  # başlık: dünya z etrafındaki fark


@pytest.mark.parametrize("key,limit", [("ekf", 0.2), ("ukf", 0.2), ("pf", 2.5), ("kalman", 0.2)])
def test_magnetic_disturbance_does_not_leak_into_roll_pitch(key, limit):
    # Denetim: UKF ve parçacık filtresi başlığı sigma / parçacık ZYX yaw'ı ile ölçünce bozulma eğime ~15° sızıyordu
    q, gyro, acc, mag, t, fs = _static_imu(-35, -75, 30, disturb=(12, 22))
    tilt, _ = _tilt_heading_errors(fu.make_estimator(key).run(gyro, acc, mag, 1 / fs), np.broadcast_to(q, (len(t), 4)))
    window = (t >= 12) & (t < 22)
    assert np.sqrt(np.mean(tilt[window] ** 2)) < limit


@pytest.mark.parametrize("key,limit", [("ekf", 0.3), ("ukf", 0.3), ("pf", 1.5)])
def test_heading_is_corrected_near_gimbal_lock(key, limit):
    # Denetim: pitch −89°'de UKF / PF başlık hatası 30°+ idi (ZYX yaw'ın kazancı sıfıra iniyordu)
    q, gyro, acc, mag, t, fs = _static_imu(20, -89, 50)
    _, head = _tilt_heading_errors(fu.make_estimator(key).run(gyro, acc, mag, 1 / fs), np.broadcast_to(q, (len(t), 4)))
    assert np.sqrt(np.mean(head[t >= 5] ** 2)) < limit


@pytest.mark.parametrize("key", list(fu.ALGORITHMS))
def test_invalid_samples_are_skipped_and_filters_recover(key):
    # Denetim: tek bir NaN jiroskop örneği tüm algoritmaları kalıcı NaN yapıyordu; Kalman sıfır ivme / mag'i ölçüm sanıyordu
    sim = fu.simulate_imu(duration=20, fs=200, seed=4)
    g, a, m = sim.gyro.copy(), sim.acc.copy(), sim.mag.copy()
    g[1000] = np.nan
    a[0] = np.nan                 # ilk örnek geçersiz: ilk geçerli ivmeden başlar
    a[2000:2003] = np.nan
    m[2500] = np.inf
    m[3000:3100] = 0.0            # manyetometre kanalı sıfırla doldurulmuş
    a[3500:3600] = 0.0
    q = fu.make_estimator(key).run(g, a, m, 1 / 200)
    assert np.flatnonzero(~np.isfinite(q).all(axis=1)).tolist() == [0, 1000]
    angles = fu.euler_from_quat(q)
    tail = slice(1500, None)
    errors = [fu.rms(fu.angle_error_deg(v[tail], sim.truth[tail, i])) for i, v in enumerate(angles)]
    assert max(errors) < 2.0, errors
    # bloklar halinde işlem aynı sonucu verir (geçersiz örnek bölge sınırına denk gelse de)
    est = fu.make_estimator(key)
    parts = np.concatenate([est.run(g[i:i + 333], a[i:i + 333], m[i:i + 333], 1 / 200) for i in range(0, 4000, 333)])
    assert np.allclose(parts, q, equal_nan=True, atol=1e-12)


# ============================================================================ motor denetimi regresyonları
_EXIT_SCRIPT = r'''
import time, numpy as np
from spt.core.fusion_engine import FusionEngine
from spt.dsp import fusion_generic as fg
from spt.core.local_ring import StaticRing
from spt.core.snapshot import Snapshot
from spt.dsp import fusion as fu
from spt.protocol.format import SAMPLE_TYPES_BY_NAME, ChannelInfo, Descriptor, Layout
if __name__ == "__main__":
    sim = fu.simulate_imu(duration=60.0, fs=200.0, seed=3)
    deg = 180 / np.pi
    ch = ([ChannelInfo(f"jiro_{a}", "°/s") for a in "xyz"] + [ChannelInfo(f"ivme_{a}", "m/s²") for a in "xyz"]
          + [ChannelInfo(f"mag_{a}", "µT") for a in "xyz"])
    data = np.column_stack((sim.gyro * deg, sim.acc, sim.mag))
    desc = Descriptor(1, 1, sim.fs, SAMPLE_TYPES_BY_NAME["float64"], Layout.INTERLEAVED, "imu", tuple(ch))
    snap = Snapshot(1, "imu", "test", desc, 0.0, StaticRing(data, 0, []))
    engine = FusionEngine(snap.handle(), "orientation", fg.auto_map("orientation", desc), {},
                          {k: {} for k in fu.ALGORITHMS})
    end = time.monotonic() + 2.0
    while time.monotonic() < end:
        engine.pump()
        time.sleep(0.02)
    t = time.perf_counter()
    engine.stop()
    print(f"STOP {time.perf_counter() - t:.3f}", flush=True)
'''


def test_engine_stop_is_non_blocking_and_interpreter_exits():
    # Denetim (KRİTİK): ağır algoritmalar (UKF / PF) parça işlerken durdurulunca stop() arayüzü ~3 s donduruyor ve
    # kuyruk besleyici iş parçacıkları yüzünden süreç hiç kapanmıyordu (pencere kapansa da arkada kalıyordu).
    import subprocess
    import sys
    from spt.console import CHILD_ENV
    root = Path(__file__).resolve().parents[1]
    proc = subprocess.run([sys.executable, "-c", _EXIT_SCRIPT], cwd=root, capture_output=True, text=True,
                          encoding="utf-8", timeout=90, env={**__import__("os").environ, **CHILD_ENV})
    assert proc.returncode == 0, proc.stderr
    stop_s = float(proc.stdout.split("STOP")[1].split()[0])
    assert stop_s < 0.5  # engellemesiz


def test_engine_progress_completes_when_recording_ends_with_gap():
    sim = fu.simulate_imu(duration=10.0, fs=200.0, seed=13)
    snap = _imu_snapshot(sim, gaps=[(1500, 2000)])
    snap.ring.data[1500:2000] = np.nan
    engine = FusionEngine(snap.handle(), "orientation", fg.auto_map("orientation", snap.desc), {}, {"mahony": {}})
    try:
        assert _wait(engine, timeout=60)  # eskiden %75'te takılıyordu
        assert engine.progress() == 1.0
        got = engine.read("mahony", 0, snap.samples)
        assert np.all(np.isnan(got[1500:])) and np.all(np.isfinite(got[:1500]))
    finally:
        engine.stop()


def test_engine_metrics_cover_whole_run_and_history_outlives_ring():
    # Denetim: RMSE yalnız halka tampondaki son 10 s'yi kapsıyordu; "30 s / 60 s" pencereleri de etkisizdi
    sim = fu.simulate_imu(duration=30.0, fs=200.0, seed=14)
    snap = _imu_snapshot(sim)
    engine = FusionEngine(snap.handle(), "orientation", fg.auto_map("orientation", snap.desc), {}, {"ekf": {}})
    try:
        assert _wait(engine)
        n = snap.samples
        rmse, worst, covered = engine.metrics("ekf")
        est = engine.read("ekf", 0, n)
        skip = int(2.0 * sim.fs)
        err = np.abs(fu.wrap(est[skip:] - sim.truth[skip:]))
        assert covered == pytest.approx((n - skip) / sim.fs)
        assert np.allclose(rmse, np.sqrt(np.mean(err ** 2, axis=0)), rtol=1e-4)
        assert np.allclose(worst, err.max(axis=0), rtol=1e-4)  # çıktı başına en büyük |hata|
        assert engine.capacity >= int(600 * sim.fs) or engine.capacity == 1 << 20
        raw = engine.raw_estimate(0, n)
        assert np.isfinite(raw).all() and np.allclose(engine.reference(0, n), sim.truth, atol=1e-6)
    finally:
        engine.stop()


def test_engine_chunks_adapt_to_algorithm_cost():
    sim = fu.simulate_imu(duration=20.0, fs=200.0, seed=15)
    snap = _imu_snapshot(sim)
    engine = FusionEngine(snap.handle(), "orientation", fg.auto_map("orientation", snap.desc), {}, {"ukf": {}})
    try:
        assert engine._chunk(engine.algos["ukf"]) == 200  # maliyet bilinmeden küçük ilk parça
        assert _wait(engine, timeout=120)
        cost = engine.algos["ukf"].cost_us
        chunk = engine._chunk(engine.algos["ukf"])
        assert cost > 0 and chunk * cost * 1e-6 <= 0.25  # parça ~0.2 s: durdurma bu kadar gecikir
    finally:
        engine.stop()
