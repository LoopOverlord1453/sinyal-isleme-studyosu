"""Genel (senaryo tabanlı) sensör füzyonu: API, doğruluk, blok = tek sefer, NaN / çok hızlı veri, birimler,
otomatik eşleme, EKF ↔ UKF uyumu, KF'nin kapalı biçim / matris referansıyla eşitliği, yönelim uyarlayıcısı ve
örnek kayıtlar (11–13)."""
import functools
import math
import pickle
from pathlib import Path

import numpy as np
import pytest

from spt.core.ring import SharedRing
from spt.core.stream_store import StreamStore
from spt.dsp import fusion as fu
from spt.dsp import fusion_generic as fg
from spt.dsp.convert import physical_column
from spt.dsp.fusion_scenarios import altitude, linear_multi, position2d
from spt.protocol.format import SAMPLE_TYPES_BY_NAME, ChannelInfo, Descriptor, Layout
from spt.protocol.recording import SptrReader

SAMPLES = Path(__file__).resolve().parents[1] / "samples_signal_data"
D2R = math.pi / 180


@functools.lru_cache(maxsize=None)
def _sim(key: str, duration: float, seed: int = 3, **kw):
    return fg.SCENARIOS[key].simulate(duration, None, seed, **kw)


def _sim_kw(key, duration, seed=3, **kw):
    return _sim(key, duration, seed, **{k: (tuple(v) if isinstance(v, list) else v) for k, v in kw.items()})


@functools.lru_cache(maxsize=None)
def _run(key: str, algo: str, duration: float, seed: int = 3):
    sim = _sim(key, duration, seed)
    est = fg.make_estimator(key, algo, sim.values, {})
    return est, est.run(sim.inputs, sim.dt)


def _rmse(key, y, truth, skip):
    return fg.rmse(fg.errors(key, y[skip:], truth[skip:]))


def _desc(channels, fs=100.0):
    return Descriptor(1, 1, fs, SAMPLE_TYPES_BY_NAME["float32"], Layout.INTERLEAVED, "test",
                      tuple(ChannelInfo(n, u) for n, u in channels))


def _slice(inputs, a, b):
    return {k: v[a:b] for k, v in inputs.items()}


def _run_blocks(est, inputs, dt, n, rng):
    cuts = np.unique(np.r_[0, np.sort(rng.integers(1, n, 25)), 1, 2, n])
    return np.concatenate([est.run(_slice(inputs, a, b), dt) for a, b in zip(cuts[:-1], cuts[1:])])


# ============================================================================ API ve tanımlar
def test_scenarios_definitions_are_consistent():
    assert list(fg.SCENARIOS) == ["orientation", "altitude", "position2d", "linear_multi"]
    for scen in fg.SCENARIOS.values():
        assert scen.name and scen.description
        for group in (scen.inputs, scen.outputs, scen.params):
            keys = [g.key for g in group]
            assert len(keys) == len(set(keys))
        assert scen.resolve({}) == {p.key: p.default if isinstance(p, fg.Choice) else float(p.default)
                                    for p in scen.params}
        for inp in scen.inputs:
            assert inp.axes and inp.units and inp.words and all(w == w.lower() for w in inp.words)
        assert scen.algorithms, scen.key
        for key, cls in scen.algorithms.items():
            assert cls.key == key and cls.name
            assert cls.outputs == scen.outputs and cls.inputs == scen.inputs
            est = cls({})
            assert est.cost_us == 0.0 and isinstance(est.diagnostics(), dict)
    assert list(fg.SCENARIOS["altitude"].algorithms) == ["complementary", "kf", "ekf", "ukf", "pf"]
    assert list(fg.SCENARIOS["position2d"].algorithms) == ["dead_reckoning", "ekf", "ukf", "pf"]
    assert list(fg.SCENARIOS["linear_multi"].algorithms) == ["weighted_mean", "kf", "pf"]
    assert list(fg.SCENARIOS["orientation"].algorithms) == list(fu.ALGORITHMS)
    assert [o.angle for o in fg.SCENARIOS["position2d"].outputs] == [False, False, True, False]


def test_estimators_are_picklable_for_worker_processes():
    for scen in fg.SCENARIOS.values():
        for key in scen.algorithms:
            est = fg.make_estimator(scen.key, key)
            clone = pickle.loads(pickle.dumps(est))
            assert type(clone) is type(est)


def test_simulators_produce_consistent_multirate_data():
    alt = _sim("altitude", 120.0)
    assert alt.truth.shape == (24000, 2) and set(alt.inputs) == {"baro_p", "acc_z"}
    assert np.isfinite(alt.inputs["baro_p"]).sum() == 3000                      # 25 Hz / 200 Hz
    assert np.allclose(np.gradient(alt.truth[:, 0], alt.dt)[1:-1], alt.truth[1:-1, 1], atol=1e-3)
    pos = _sim("position2d", 120.0)
    gx = pos.inputs["gnss"][:, 0]
    t = np.arange(pos.n) / pos.fs
    assert not np.isfinite(gx[(t >= 60) & (t < 70)]).any() and np.isfinite(gx).sum() == 550
    x, y, psi, v = pos.truth.T
    assert np.allclose(np.gradient(x, pos.dt)[1:-1], (v * np.cos(psi))[1:-1], atol=2e-3)
    assert np.allclose(np.gradient(y, pos.dt)[1:-1], (v * np.sin(psi))[1:-1], atol=2e-3)
    lin = _sim("linear_multi", 60.0)
    assert [np.isfinite(lin.inputs[f"s{i}"]).sum() for i in range(1, 5)] == [6000, 1200, 3000, 6000]
    assert lin.values["s4_kind"] == "hiz" and lin.values["s2_sigma"] == 0.10


# ============================================================================ doğruluk (simülasyon, gerçeğe göre)
ALT_LIMITS = {"complementary": (0.10, 0.06), "kf": (0.08, 0.04), "ekf": (0.08, 0.04), "ukf": (0.08, 0.04),
              "pf": (0.09, 0.045)}


@pytest.mark.parametrize("algo", list(ALT_LIMITS))
def test_altitude_accuracy(algo):
    sim = _sim("altitude", 120.0)
    est, y = _run("altitude", algo, 120.0)
    h, v = _rmse("altitude", y, sim.truth, 2000)
    assert h < ALT_LIMITS[algo][0] and v < ALT_LIMITS[algo][1], (h, v)
    p = sim.inputs["baro_p"]
    ok = np.isfinite(p)
    raw = fg.rmse(altitude.isa_altitude(p[ok], altitude.P_STD) - sim.truth[ok, 0])[0]
    assert h < 0.35 * raw                                     # ham barometreden (0.34 m) çok daha iyi
    assert np.isfinite(y).all()
    bias = float(est.diagnostics()["İvme sapması (m/s²)"])
    assert abs(bias - 0.12) < 0.05                            # gerçek sapma 0.12 ± 0.02 (yavaş sinüs)


def test_altitude_kalman_beats_complementary():
    """Doğru modelli KF, sabit kazançlı tamamlayıcı filtreden hem irtifada hem dikey hızda iyidir."""
    sim = _sim("altitude", 120.0)
    cf = _rmse("altitude", _run("altitude", "complementary", 120.0)[1], sim.truth, 2000)
    kf = _rmse("altitude", _run("altitude", "kf", 120.0)[1], sim.truth, 2000)
    assert kf[0] < 0.95 * cf[0] and kf[1] < 0.8 * cf[1], (kf, cf)


POS_LIMITS = {"ekf": (1.0, 0.8, 0.1), "ukf": (1.0, 0.8, 0.1), "pf": (1.0, 1.0, 0.1)}


def _pos_metrics(y, truth, mask):
    err = fg.errors("position2d", y, truth)
    pos = np.hypot(err[:, 0], err[:, 1])
    return (float(np.sqrt(np.mean(pos[mask] ** 2))), float(np.degrees(fg.rmse(err[mask, 2])[0])),
            float(fg.rmse(err[mask, 3])[0]))


@pytest.mark.parametrize("algo", list(POS_LIMITS))
def test_position2d_accuracy_and_gnss_outage(algo):
    sim = _sim("position2d", 120.0)
    t = np.arange(sim.n) / sim.fs
    est, y = _run("position2d", algo, 120.0)
    _, dr = _run("position2d", "dead_reckoning", 120.0)
    pos, hdg, spd = _pos_metrics(y, sim.truth, t > 10)
    assert pos < POS_LIMITS[algo][0] and hdg < POS_LIMITS[algo][1] and spd < POS_LIMITS[algo][2], (pos, hdg, spd)
    g = sim.inputs["gnss"]
    ok = np.isfinite(g[:, 0])
    raw = float(np.sqrt(np.mean(np.sum((g[ok] - sim.truth[ok, :2]) ** 2, axis=1))))
    assert pos < 0.3 * raw                                    # ham GNSS ≈ 3.5 m
    dr_pos = _pos_metrics(dr, sim.truth, t > 10)[0]
    assert pos < 0.05 * dr_pos                                # ölü hesap kayar (sapma + ölçek hatası)
    outage = (t >= 60) & (t < 70)                             # 10 s GNSS kesintisi (dönüş dahil)
    out_pos = _pos_metrics(y, sim.truth, outage)[0]
    assert out_pos < 1.5 and out_pos < 0.1 * _pos_metrics(dr, sim.truth, outage)[0]
    d = est.diagnostics()
    assert abs(float(d["Jiroskop sapması (°/s)"]) - 0.4) < 0.05
    assert abs(float(d["Odometri ölçeği"]) - 1.03) < 0.01


def test_position2d_dead_reckoning_initialises_from_gnss_track():
    sim = _sim("position2d", 120.0)
    est, y = _run("position2d", "dead_reckoning", 120.0)
    k0 = int(est.diagnostics()["Başlatma örneği"])
    assert 50 < k0 < 400                                      # 20 m yol ≈ 2 s (10 m/s)
    assert np.isnan(y[:k0, 2]).all() and np.isfinite(y[k0:]).all()
    err = fg.errors("position2d", y[k0:k0 + 1], sim.truth[k0:k0 + 1])[0]
    assert math.hypot(err[0], err[1]) < 8.0 and abs(err[2]) < 15 * D2R


LIN_SKIP = 200


@pytest.mark.parametrize("algo", ["weighted_mean", "kf", "pf"])
def test_linear_multi_accuracy(algo):
    sim = _sim("linear_multi", 60.0)
    _, y = _run("linear_multi", algo, 60.0)
    r = _rmse("linear_multi", y, sim.truth, LIN_SKIP)
    _, wm = _run("linear_multi", "weighted_mean", 60.0)
    rw = _rmse("linear_multi", wm, sim.truth, LIN_SKIP)
    assert rw[0] < 0.035 and rw[1] < 0.021 and np.isnan(rw[2])   # ivme sensörü yok → NaN
    if algo == "kf":
        assert r[0] < 0.004 and r[0] < 0.2 * rw[0]           # zamanda füzyon: anlık ortalamadan ≫ iyi
        assert r[1] < 0.95 * rw[1]                            # hız: enkoder + konum türevi
        assert np.all(y[LIN_SKIP:, 2] == 0.0)                 # sabit hız modeli: ivme = 0 (model türevi)
    if algo == "pf":
        _, kf = _run("linear_multi", "kf", 60.0)
        rk = _rmse("linear_multi", kf, sim.truth, LIN_SKIP)
        assert r[0] < 2.0 * rk[0] and r[0] < 0.3 * rw[0] and r[1] < 1.2 * rk[1]


def test_linear_multi_constant_acceleration_model_estimates_acceleration():
    sim = _sim_kw("linear_multi", 30.0, order="ivme")
    y = fg.make_estimator("linear_multi", "kf", sim.values).run(sim.inputs, sim.dt)
    r = _rmse("linear_multi", y, sim.truth, LIN_SKIP)
    assert r[0] < 0.004 and r[1] < 0.012 and r[2] < 0.25      # gerçek ivme RMS ≈ 0.7


def test_orientation_adapter_accuracy():
    sim = _sim("orientation", 20.0, seed=7)
    for algo in ("mahony", "kalman", "ekf"):
        y = fg.make_estimator("orientation", algo).run(sim.inputs, sim.dt)
        r = np.degrees(_rmse("orientation", y, sim.truth, 1000))
        assert np.all(r < 2.5), (algo, r)


# ============================================================================ blok = tek sefer
BLOCK_CASES = [("altitude", a, 20.0) for a in ("complementary", "kf", "ekf", "ukf", "pf")] + \
    [("position2d", a, 40.0) for a in ("dead_reckoning", "ekf", "ukf", "pf")] + \
    [("linear_multi", a, 20.0) for a in ("weighted_mean", "kf", "pf")]


@pytest.mark.parametrize("key,algo,duration", BLOCK_CASES, ids=lambda v: str(v))
def test_block_processing_equals_one_shot_exactly(key, algo, duration):
    sim = _sim(key, duration)
    whole = fg.make_estimator(key, algo, sim.values).run(sim.inputs, sim.dt)
    blocks = _run_blocks(fg.make_estimator(key, algo, sim.values), sim.inputs, sim.dt, sim.n,
                         np.random.default_rng(5))
    assert np.array_equal(whole, blocks, equal_nan=True)
    assert np.isfinite(whole[-1, :2 if algo == "weighted_mean" else None]).all()   # ivme sensörü yok


@pytest.mark.parametrize("algo", list(fu.ALGORITHMS))
def test_orientation_block_processing_equals_one_shot(algo):
    sim = _sim("orientation", 1.5, seed=7)
    whole = fg.make_estimator("orientation", algo).run(sim.inputs, sim.dt)
    blocks = _run_blocks(fg.make_estimator("orientation", algo), sim.inputs, sim.dt, sim.n,
                         np.random.default_rng(6))
    assert np.max(np.abs(fg.errors("orientation", blocks, whole))) < 1e-9


def test_particle_filter_seed_controls_randomness():
    sim = _sim("altitude", 20.0)
    a = fg.make_estimator("altitude", "pf", sim.values, {"seed": 1}).run(sim.inputs, sim.dt)
    b = fg.make_estimator("altitude", "pf", sim.values, {"seed": 1}).run(sim.inputs, sim.dt)
    c = fg.make_estimator("altitude", "pf", sim.values, {"seed": 2}).run(sim.inputs, sim.dt)
    assert np.array_equal(a, b) and not np.array_equal(a, c)


# ============================================================================ NaN / çok hızlı veri / eksik giriş
def test_altitude_survives_missing_accelerometer_and_inf():
    sim = _sim_kw("altitude", 60.0, acc_gap=(30.0, 1.0))
    inputs = dict(sim.inputs)
    acc = inputs["acc_z"].copy()
    acc[[1000, 1001, 5000]] = np.inf
    inputs["acc_z"] = acc
    t = np.arange(sim.n) / sim.fs
    for algo in ("complementary", "kf", "ekf", "ukf", "pf"):
        y = fg.make_estimator("altitude", algo, sim.values).run(inputs, sim.dt)
        assert np.isfinite(y).all(), algo
        err = fg.errors("altitude", y, sim.truth)
        gap = (t >= 30) & (t < 32)
        assert np.max(np.abs(err[gap, 0])) < 1.0, algo        # ivmesiz 1 s: barometre taşır
        assert fg.rmse(err[t > 35, 0])[0] < 0.12, algo        # sonra toparlanır


def test_altitude_waits_for_first_barometer_sample():
    sim = _sim("altitude", 5.0)
    inputs = dict(sim.inputs)
    p = inputs["baro_p"].copy()
    p[:100] = np.nan                                          # ilk baro 100. örnekten sonra (104)
    inputs["baro_p"] = p
    y = fg.make_estimator("altitude", "kf", sim.values).run(inputs, sim.dt)
    first = int(np.flatnonzero(np.isfinite(p))[0])
    assert np.isnan(y[:first]).all() and np.isfinite(y[first:]).all()


def test_position2d_survives_gyro_dropout():
    sim = _sim("position2d", 60.0)
    inputs = dict(sim.inputs)
    g = inputs["gyro_z"].copy()
    g[3000:3050] = np.nan                                     # 0.5 s jiroskop yok (dönüş sırasında değil)
    inputs["gyro_z"] = g
    t = np.arange(sim.n) / sim.fs
    for algo in ("dead_reckoning", "ekf", "ukf", "pf"):
        y = fg.make_estimator("position2d", algo, sim.values).run(inputs, sim.dt)
        k0 = int(np.flatnonzero(np.isfinite(y[:, 2]))[0])
        assert np.isfinite(y[k0:]).all(), algo
        if algo != "dead_reckoning":
            assert _pos_metrics(y, sim.truth, t > 10)[0] < 1.0, algo


def test_position2d_latlon_input_matches_local_metres():
    sim_m = _sim("position2d", 40.0)
    sim_ll = _sim_kw("position2d", 40.0, latlon=True)
    assert "gnss_ll" in sim_ll.inputs and "gnss" not in sim_ll.inputs
    a = fg.make_estimator("position2d", "ekf", sim_m.values).run(sim_m.inputs, sim_m.dt)
    b = fg.make_estimator("position2d", "ekf", sim_ll.values).run(sim_ll.inputs, sim_ll.dt)
    # enlem/boylam ilk düzeltme etrafında açılır: aradaki fark ilk düzeltmenin (gürültülü) konumu kadar öteleme
    first = np.flatnonzero(np.isfinite(sim_m.inputs["gnss"][:, 0]))[0]
    shift = sim_m.inputs["gnss"][first]
    ok = np.isfinite(a).all(axis=1)
    assert np.allclose(b[ok, :2] + shift, a[ok, :2], atol=2e-3)
    assert np.allclose(fg.errors("position2d", a[ok], b[ok])[:, 2:], 0, atol=1e-4)
    x, y = position2d.ll_to_local(*position2d.local_to_ll(np.array([0.0, 1000.0]), np.array([0.0, -500.0]),
                                                          0.7, 0.5), 0.7, 0.5)
    assert np.allclose(x, [0, 1000], atol=1e-9) and np.allclose(y, [0, -500], atol=1e-9)


def test_linear_multi_all_nan_slot_equals_unmapped_slot():
    sim = _sim("linear_multi", 20.0)
    with_nan = dict(sim.inputs, s3=np.full(sim.n, np.nan))
    without = {k: v for k, v in sim.inputs.items() if k != "s3"}
    for algo in ("weighted_mean", "kf", "pf"):
        a = fg.make_estimator("linear_multi", algo, sim.values).run(with_nan, sim.dt)
        b = fg.make_estimator("linear_multi", algo, sim.values).run(without, sim.dt)
        assert np.array_equal(a, b, equal_nan=True), algo


def test_orientation_adapter_handles_gaps_and_sparse_magnetometer():
    sim = _sim("orientation", 20.0, seed=7)
    gyro, acc, mag = sim.inputs["gyro"].copy(), sim.inputs["acc"], sim.inputs["mag"].copy()
    gyro[2000:2010] = np.nan                                  # 50 ms kayıp → NaN, temiz yeniden başlatma
    sparse = np.full_like(mag, np.nan)
    sparse[1::4] = mag[1::4]                                  # 50 Hz manyetometre, ilk satırda yok
    est = fg.make_estimator("orientation", "ekf")
    y = est.run({"gyro": gyro, "acc": acc, "mag": sparse}, sim.dt)
    assert np.isnan(y[0]).all() and np.isfinite(y[1:2000]).all()     # ilk manyetometre beklenir
    nxt = 2010 + (1 - 2010) % 4                               # yeniden başlatmadan sonraki ilk manyetometre: 2013
    assert np.isnan(y[2000:nxt]).all()                        # kayıp + yeniden başlatmada manyetometre beklenir
    assert np.isfinite(y[nxt:]).all() and est.restarts == 1
    err = np.degrees(fg.errors("orientation", y, sim.truth))
    assert np.max(np.abs(err[nxt:2100])) < 3.0                # ivmeölçer + manyetometreden temiz başlangıç
    assert np.all(np.degrees(_rmse("orientation", y[:2000], sim.truth[:2000], 1000)) < 2.0)


# ============================================================================ birimler ve eşleme
def test_unit_tables_and_conversions():
    rate = fg.SCENARIOS["orientation"].input("gyro").units
    assert rate["rad/s"] == 1.0 and math.isclose(rate["°/s"], D2R) and math.isclose(rate["mdps"], D2R / 1000)
    press = fg.SCENARIOS["altitude"].input("baro_p").units
    assert press["hPa"] == 100.0 and press["mbar"] == 100.0 and press["kPa"] == 1000.0
    assert math.isclose(fg.SCENARIOS["altitude"].input("acc_z").units["g"], fu.GRAVITY)
    # Kanal birimi → giriş birim anahtarı
    from spt.dsp.fusion_scenarios.base import unit_key
    assert unit_key(rate, "deg/s") == "°/s" and unit_key(rate, "dps") == "°/s" and unit_key(rate, "rad/s") == "rad/s"
    assert unit_key(rate, "mdps") == "mdps" and unit_key(rate, "V") is None
    assert unit_key(press, "hPa") == "hPa" and unit_key(press, "mbar") == "mbar" and unit_key(press, "kPa") == "kPa"
    acc = fg.SCENARIOS["altitude"].input("acc_z").units
    assert unit_key(acc, "m/s^2") == "m/s²" and unit_key(acc, "m/s2") == "m/s²" and unit_key(acc, "g") == "g"
    assert unit_key(fg.SCENARIOS["orientation"].input("mag").units, "uT") == "µT"
    heading = fg.SCENARIOS["position2d"].outputs[2]
    assert math.isclose(fg.reference_scale(heading, "°"), D2R) and math.isclose(fg.reference_scale(heading, "deg"), D2R)
    assert fg.reference_scale(heading, "rad") == 1.0 and fg.reference_scale(fg.SCENARIOS["position2d"].outputs[0], "m") == 1.0


def test_gather_inputs_converts_units_to_si_and_matches_si_run():
    sim = _sim("altitude", 20.0)
    desc = _desc([("Basınç", "hPa"), ("AccZ", "g"), ("ref_irtifa", "m"), ("ref_vz", "m/s")], fs=sim.fs)
    cols = np.column_stack((sim.inputs["baro_p"] / 100.0, sim.inputs["acc_z"] / fu.GRAVITY, sim.truth))
    m = fg.auto_map("altitude", desc)
    assert m.inputs == {"baro_p": (0,), "acc_z": (1,)} and m.units == {"baro_p": "hPa", "acc_z": "g"}
    assert m.reference == {"alt": 2, "vz": 3}
    x = fg.gather_inputs("altitude", m, cols)
    assert np.allclose(x["baro_p"], sim.inputs["baro_p"], rtol=1e-12, equal_nan=True)
    assert np.allclose(x["acc_z"], sim.inputs["acc_z"], rtol=1e-12)
    a = fg.make_estimator("altitude", "ekf", sim.values).run(x, sim.dt)
    b = fg.make_estimator("altitude", "ekf", sim.values).run(sim.inputs, sim.dt)
    assert np.allclose(a, b, atol=1e-6)
    ref = fg.gather_reference("altitude", m, desc, cols)
    assert np.array_equal(ref, sim.truth)
    # açısal referans derece → radyan; eşlenmeyen çıktı NaN
    pdesc = _desc([("gps_x", "m"), ("gps_y", "m"), ("jiro_z", "°/s"), ("ref_baslik", "°")])
    pm = fg.auto_map("position2d", pdesc)
    pref = fg.gather_reference("position2d", pm, pdesc, np.array([[1.0, 2.0, 3.0, 90.0]]))
    assert np.isnan(pref[0, [0, 1, 3]]).all() and math.isclose(pref[0, 2], math.pi / 2)
    assert np.allclose(fg.to_display("position2d", pref)[0, 2], 90.0)


def test_auto_map_turkish_and_english_names():
    # Yönelim: CamelCase İngilizce + Türkçe büyük İ ve boşluk, referans "true_*"
    desc = _desc([("GyroX", "deg/s"), ("GyroY", "deg/s"), ("GyroZ", "deg/s"), ("İvme X", "g"), ("İvme Y", "g"),
                  ("İvme Z", "g"), ("MagX", "uT"), ("MagY", "uT"), ("MagZ", "uT"), ("true_roll", "deg"),
                  ("true_pitch", "deg"), ("true_yaw", "deg")])
    m = fg.auto_map("orientation", desc)
    assert m.inputs == {"gyro": (0, 1, 2), "acc": (3, 4, 5), "mag": (6, 7, 8)}
    assert m.units == {"gyro": "°/s", "acc": "g", "mag": "µT"}
    assert m.reference == {"roll": 9, "pitch": 10, "yaw": 11}
    # Kısa adlar, manyetometresiz
    m = fg.auto_map("orientation", _desc([(n, u) for n, u in zip(("ax", "ay", "az", "gx", "gy", "gz"),
                                                                 ("m/s²",) * 3 + ("rad/s",) * 3)]))
    assert m.inputs == {"acc": (0, 1, 2), "gyro": (3, 4, 5)} and m.units["gyro"] == "rad/s" and not m.reference
    # İrtifa: baro irtifası (m) basınçtan ayırt edilir; ivme 3 eksenden z seçilir
    m = fg.auto_map("altitude", _desc([("baro_irtifa", "m"), ("ivme_x", "m/s²"), ("ivme_y", "m/s²"),
                                       ("ivme_z", "m/s²"), ("ref_irtifa", "m")]))
    assert m.inputs == {"baro_alt": (0,), "acc_z": (3,)} and m.reference == {"alt": 4}
    m = fg.auto_map("altitude", _desc([("pressure", "mbar"), ("acc_x", "g"), ("acc_y", "g"), ("acc_z", "g"),
                                       ("true_alt", "m"), ("ref_dikey_hiz", "m/s")]))
    assert m.inputs == {"baro_p": (0,), "acc_z": (3,)} and m.units["baro_p"] == "mbar"
    assert m.reference == {"alt": 4, "vz": 5}
    # 2B konum: enlem / boylam (°), dps, km/h; Türkçe adlar
    m = fg.auto_map("position2d", _desc([("lat", "deg"), ("lon", "deg"), ("GyroZ", "dps"), ("speed", "km/h"),
                                         ("true_heading", "deg"), ("ref_speed", "m/s")]))
    assert m.inputs == {"gnss_ll": (0, 1), "gyro_z": (2,), "speed": (3,)}
    assert m.units == {"gnss_ll": "°", "gyro_z": "°/s", "speed": "km/h"}
    assert m.reference == {"heading": 4, "speed": 5}
    m = fg.auto_map("position2d", _desc([("Boylam", "°"), ("Enlem", "°"), ("jiro_z", "°/s"), ("odom", "m/s")]))
    assert m.inputs == {"gnss_ll": (1, 0), "gyro_z": (2,), "speed": (3,)}
    # Doğrusal: yedekli sıcaklık sensörleri; enkoder konumu + ivme → tür ve derece önerisi
    m = fg.auto_map("linear_multi", _desc([("sicaklik_1", "°C"), ("Sıcaklık 2", "°C"), ("temp3", "°C")]))
    assert m.inputs == {"s1": (0,), "s2": (1,), "s3": (2,)} and m.units["s1"] == "°C"
    assert m.values == {"s1_kind": "konum", "s2_kind": "konum", "s3_kind": "konum", "order": "hiz"}
    m = fg.auto_map("linear_multi", _desc([("encoder_pos", "m"), ("acc", "m/s²"), ("ref_konum", "m")]))
    assert m.values["s1_kind"] == "konum" and m.values["s2_kind"] == "ivme" and m.values["order"] == "ivme"
    assert m.reference == {"konum": 2}
    # Eşlenemeyen
    for key in fg.SCENARIOS:
        assert fg.auto_map(key, _desc([("sinyal", "V")])) is None
    assert fg.auto_map("altitude", _desc([("ivme_z", "m/s²")])) is None      # baro yok
    assert fg.auto_map("position2d", _desc([("gps_x", "m"), ("gps_y", "m")])) is None   # jiroskop yok


def _descriptor_of(path):
    from spt.protocol.format import HEADER, FrameType
    from spt.protocol.parser import check_frame, unpack_descriptor
    with SptrReader(path) as reader:
        offsets, _, lengths = reader.index()
        for off, n in zip(offsets.tolist(), lengths.tolist()):
            ftype, _, flen, sid, rev, _, _ = check_frame(reader.buffer, off, off + n)
            if ftype == FrameType.DESCRIPTOR:
                return unpack_descriptor(reader.buffer, off + HEADER.size, off + flen, sid, rev)


def _replay(path):
    """Kaydı alıcının akış katmanından geçirir; (descriptor, (n, kanal) fiziksel değerler)."""
    events = []
    with SptrReader(path) as reader:
        offsets, rx, lengths = reader.index()
        store = StreamStore(events.append, int(rx[0]), ring_seconds=300.0)
        for off, t, n in zip(offsets.tolist(), rx.tolist(), lengths.tolist()):
            store.handle_frame(reader.buffer, off, off + n, "kayit", t)
        (info,) = [e[1] for e in events if e[0] == "stream"]
        ring = SharedRing.attach(info.shm_name, info.descriptor.sample_type.dtype)
        try:
            desc = info.descriptor
            _, raw = ring.read(ring.valid_from(), ring.write_count)
            cols = np.column_stack([physical_column(raw[:, i], desc.sample_type, desc.channels[i]).real
                                    for i in range(desc.channel_count)])
        finally:
            ring.close()
            store.close()
    return desc, cols


SAMPLE_CASES = [("11_irtifa_baro_25Hz_ivme_200Hz_referansli.sptr", "altitude", "kf", (0.10, 0.06)),
                ("12_arac_gnss_5Hz_jiro_odometri_100Hz_referansli.sptr", "position2d", "ekf", (1.0, 1.0, 1.0, 0.1)),
                ("13_coklu_mesafe_sensoru_enkoder_100Hz_referansli.sptr", "linear_multi", "kf", (0.005, 0.03))]


@pytest.mark.parametrize("name,key,algo,limits", SAMPLE_CASES, ids=lambda v: str(v)[:2])
def test_sample_recordings_map_fully_and_fuse(name, key, algo, limits):
    path = SAMPLES / name
    if not path.exists():
        pytest.skip("örnek kayıt üretilmemiş (tools/make_fusion_samples.py)")
    scen = fg.SCENARIOS[key]
    m = fg.auto_map(scen, _descriptor_of(path))
    assert m is not None and len(m.reference) >= 2
    desc, cols = _replay(path)
    m = fg.auto_map(scen, desc)
    mapped = {ch for idx in m.inputs.values() for ch in idx} | set(m.reference.values())
    assert mapped == set(range(desc.channel_count))           # tüm kanallar eşlendi
    values = dict(m.values)
    if key == "linear_multi":                                 # MARKER'daki σ'lar
        values.update(s1_sigma=0.03, s2_sigma=0.10, s3_sigma=0.20, s4_sigma=0.02)
    if key == "altitude":
        values.update(sigma_p=4.0, sigma_acc=0.08)
    est = fg.make_estimator(key, algo, values)
    y = est.run(fg.gather_inputs(scen, m, cols), 1.0 / desc.sample_rate)
    ref = fg.gather_reference(scen, m, desc, cols)
    skip = int(10 * desc.sample_rate)
    r = fg.rmse(fg.errors(scen, y[skip:], ref[skip:]))
    if key == "position2d":
        r[2] = np.degrees(r[2])
    mapped_r = [r[scen.output_index(k)] for k in m.reference]
    assert all(v < lim for v, lim in zip(mapped_r, limits)), mapped_r


# ============================================================================ EKF ↔ UKF, kapalı biçim referanslar
@pytest.mark.parametrize("h0", [120.0, 3000.0])
def test_altitude_ekf_and_ukf_agree_on_nonlinear_baro_model(h0):
    sim = _sim_kw("altitude", 60.0, h0=h0)
    e = fg.make_estimator("altitude", "ekf", sim.values).run(sim.inputs, sim.dt)
    u = fg.make_estimator("altitude", "ukf", sim.values).run(sim.inputs, sim.dt)
    k = fg.make_estimator("altitude", "kf", sim.values).run(sim.inputs, sim.dt)
    assert np.max(np.abs(e - u)) < 1e-4                      # basınç modeli yumuşak: UT ≈ doğrusallaştırma
    assert 0 < np.max(np.abs(e - k)) < 1e-2                   # KF (basınç → irtifa) da yakın ama aynı değil


def test_ukf_unscented_update_matches_monte_carlo_moments():
    """UKF'nin basınç güncellemesi: z ortalaması ve kovaryansları, ayrıntılı Monte Carlo ile ±%2 içinde."""
    est = altitude.AltitudeUKF({"p0": 101325.0})
    est.p0 = 101325.0
    P = (400.0, 30.0, 0.5, 9.0, 0.2, 0.04)                    # σ_h = 20 m: belirgin doğrusal olmama
    h, v, b = 1000.0, 2.0, 0.1
    z = altitude.isa_pressure(1010.0, 101325.0)
    h1, v1, b1, P1 = est._update(h, v, b, P, (True, z))
    rng = np.random.default_rng(0)
    cov = np.array([[P[0], P[1], P[2]], [P[1], P[3], P[4]], [P[2], P[4], P[5]]])
    xs = rng.multivariate_normal([h, v, b], cov, 400_000)
    zs = altitude.isa_pressure(xs[:, 0], 101325.0)
    r = est.sv["sigma_p"] ** 2
    pzz = zs.var() + r
    pxz = ((xs - xs.mean(0)) * (zs - zs.mean())[:, None]).mean(0)
    k_mc = pxz / pzz
    k_ukf = np.array([h1 - h, v1 - v, b1 - b]) / (z - zs.mean())
    assert np.allclose(k_ukf, k_mc, rtol=0.02)


def test_linear_kf_equals_running_mean_for_constant_level():
    """Sabit seviye, σ_q = 0, tek sensör: KF tahmini = o ana kadarki ölçümlerin ortalaması (kapalı biçim)."""
    rng = np.random.default_rng(1)
    z = 5.0 + 0.3 * rng.standard_normal(500)
    z[::7] = np.nan                                           # seyrek kayıplar
    est = fg.make_estimator("linear_multi", "kf", {"order": "seviye", "sigma_q": 0.0, "s1_sigma": 0.3})
    y = est.run({"s1": z}, 0.01)
    ok = np.isfinite(z)
    mean = np.cumsum(np.where(ok, z, 0.0)) / np.maximum(np.cumsum(ok), 1)
    first = int(np.flatnonzero(ok)[0])
    assert np.isnan(y[:first, 0]).all()
    assert np.allclose(y[first:, 0], mean[first:], atol=1e-12)
    assert np.all(y[first:, 1:] == 0.0)
    assert math.isclose(float(est.diagnostics()["σ konum"]), 0.3 / math.sqrt(ok.sum()), rel_tol=1e-3)


def _matrix_kf(F, Q, x0, P0, meas):
    """Ders kitabı matris Kalman filtresi (numpy): her örnekte öngörü + sıralı skaler güncellemeler."""
    x, P = np.array(x0, dtype=float), np.array(P0, dtype=float)
    out = [x.copy()]
    for zs in meas[1:]:
        x, P = F @ x, F @ P @ F.T + Q
        for H, z, r in zs:
            S = H @ P @ H + r
            K = P @ H / S
            x = x + K * (z - H @ x)
            P = (np.eye(len(x)) - np.outer(K, H)) @ P
        out.append(x.copy())
    return np.array(out)


def test_linear_kf_equals_textbook_matrix_kf_multirate():
    sim = _sim_kw("linear_multi", 10.0, sensors=(("konum", 0.05, 100.0), ("hiz", 0.03, 25.0), ("ivme", 0.2, 50.0)),
                  order="ivme")
    est = fg.make_estimator("linear_multi", "kf", sim.values)
    y = est.run(sim.inputs, sim.dt)
    F, Q = (np.array(m) for m in linear_multi.model_matrices(3, sim.dt, sim.values["sigma_q"]))
    meas = []
    for k in range(sim.n):
        zs = []
        for i, lvl in ((1, 0), (2, 1), (3, 2)):
            z = sim.inputs[f"s{i}"][k]
            if np.isfinite(z):
                zs.append((np.eye(3)[lvl], z, sim.values[f"s{i}_sigma"] ** 2))
        meas.append(zs)
    # başlatma (belgelenen kural): ilk örnekte konum ölçümü; ölçülmeyen türevler 0 ± σ0
    first = meas[0]
    x0 = [0.0, 0.0, 0.0]
    p0 = [1.0, 1.0, 1.0]
    for H, z, r in first:
        lvl = int(np.argmax(H))
        x0[lvl], p0[lvl] = z, r
    ref = _matrix_kf(F, Q, x0, np.diag(p0), meas)
    assert np.allclose(y, ref, atol=1e-9)
    assert np.allclose(np.array(est.P), est.P[0][0] * 0 + np.array(est.P).T)   # simetrik


def test_altitude_kf_equals_textbook_matrix_kf():
    sim = _sim_kw("altitude", 10.0, baro="altitude")
    v = dict(sim.values)
    est = fg.make_estimator("altitude", "kf", v)
    y = est.run(sim.inputs, sim.dt)
    T, g = sim.dt, fu.GRAVITY
    F = np.array([[1, T, -T * T / 2], [0, 1, -T], [0, 0, 1]])
    G = np.array([T * T / 2, T, 0.0])
    sa, sb = est.sv["sigma_acc"], est.sv["sigma_bias"]
    Q = sa ** 2 * np.outer(G, G) + np.diag([0, 0, sb ** 2 * T])
    zh, f = sim.inputs["baro_alt"], sim.inputs["acc_z"]
    x = np.array([zh[0], 0.0, 0.0])
    P = np.diag([v["sigma_h"] ** 2, 1.0, est.sv["bias0"] ** 2])
    out = [x[:2].copy()]
    H = np.array([1.0, 0.0, 0.0])
    for k in range(1, sim.n):
        x = F @ x + np.array([T * T / 2, T, 0.0]) * (f[k] - g)
        P = F @ P @ F.T + Q
        if np.isfinite(zh[k]):
            S = H @ P @ H + v["sigma_h"] ** 2
            K = P @ H / S
            x = x + K * (zh[k] - x[0])
            P = P - np.outer(K, H @ P)
        out.append(x[:2].copy())
    assert np.allclose(y, np.array(out), atol=1e-9)


# ============================================================================ yönelim uyarlayıcısı
@pytest.mark.parametrize("algo", list(fu.ALGORITHMS))
def test_orientation_adapter_equals_direct_fusion_call(algo):
    sim = _sim("orientation", 2.0, seed=7)
    y = fg.make_estimator("orientation", algo).run(sim.inputs, sim.dt)
    direct = fu.make_estimator(algo).run(sim.inputs["gyro"], sim.inputs["acc"], sim.inputs["mag"], sim.dt)
    assert np.array_equal(y, np.column_stack(fu.euler_from_quat(direct)))
    params = {p.key: p.default for p in fu.ALGORITHMS[algo].params}
    est = fg.make_estimator("orientation", algo, {}, params)
    assert set(est.diagnostics()) >= {"Jiroskop sapması (°/s)"}
    no_mag = {k: v for k, v in sim.inputs.items() if k != "mag"}
    y2 = fg.make_estimator("orientation", algo).run(no_mag, sim.dt)
    d2 = fu.make_estimator(algo).run(sim.inputs["gyro"], sim.inputs["acc"], None, sim.dt)
    assert np.array_equal(y2, np.column_stack(fu.euler_from_quat(d2)))


# ============================================================================ füzyonsuz (ham) kestirim
def _raw_chunked(scen, inputs, values, n, rng):
    cuts = np.unique(np.r_[0, np.sort(rng.integers(1, n, 15)), n])
    return np.concatenate([scen.raw(_slice(inputs, a, b), values) for a, b in zip(cuts[:-1], cuts[1:])])


def test_raw_orientation_from_single_sensors():
    scen = fg.SCENARIOS["orientation"]
    sim = _sim("orientation", 5.0, seed=7)
    r = scen.raw(sim.inputs)
    assert r.shape == (sim.n, 3)
    roll, pitch = fu.accel_roll_pitch(sim.inputs["acc"])
    assert np.array_equal(r[:, 0], roll) and np.array_equal(r[:, 1], pitch)
    assert np.allclose(r[:, 2], fu.tilt_compensated_heading(sim.inputs["mag"], roll, pitch))
    assert np.isnan(scen.raw({k: v for k, v in sim.inputs.items() if k != "mag"})[:, 2]).all()
    sparse = np.full_like(sim.inputs["mag"], np.nan)
    sparse[::4] = sim.inputs["mag"][::4]
    r2 = scen.raw(dict(sim.inputs, mag=sparse))
    assert np.isfinite(r2[::4, 2]).all() and np.isnan(r2[1::4, 2]).all()
    # ham açı doğrusal ivme yüzünden füzyondan kötüdür
    fused = fg.make_estimator("orientation", "ekf").run(sim.inputs, sim.dt)
    assert _rmse("orientation", fused, sim.truth, 400)[0] < _rmse("orientation", r, sim.truth, 400)[0]


def test_raw_altitude_baro_only_and_auto_p0_state():
    scen = fg.SCENARIOS["altitude"]
    sim = _sim("altitude", 20.0)
    r = scen.raw(sim.inputs, sim.values)
    p = sim.inputs["baro_p"]
    ok = np.isfinite(p)
    assert r.shape == (sim.n, 2) and np.isnan(r[:, 1]).all()
    assert np.isnan(r[~ok, 0]).all()
    assert np.allclose(r[ok, 0], altitude.isa_altitude(p[ok], 101325.0))
    assert fg.rmse(r[ok, 0] - sim.truth[ok, 0])[0] < 0.45        # σ_p = 4 Pa ≈ 0.34 m
    # birim: hPa girişi → aynı irtifa (çağıran SI'ya çevirir)
    desc = _desc([("basinc", "hPa"), ("ivme_z", "m/s²")], fs=sim.fs)
    m = fg.auto_map("altitude", desc)
    cols = np.column_stack((p / 100.0, sim.inputs["acc_z"]))
    assert np.allclose(scen.raw(fg.gather_inputs("altitude", m, cols), sim.values), r, equal_nan=True)
    # p0 = 0 (otomatik): ilk basınç sözlüğe yazılır; parça parça = tek sefer; kestiricinin referansıyla aynı
    state = dict(sim.values, p0=0.0)
    whole = scen.raw(sim.inputs, dict(state))
    chunked = _raw_chunked(scen, sim.inputs, state, sim.n, np.random.default_rng(3))
    assert state["_p0"] == p[ok][0] and np.array_equal(whole, chunked, equal_nan=True)
    assert whole[0, 0] == 0.0
    est = fg.make_estimator("altitude", "kf", state)            # "_p0" doğrulamada yok sayılır
    est.run(sim.inputs, sim.dt)
    assert est.p0 == state["_p0"]
    # baro irtifa girişi doğrudan
    sim_h = _sim_kw("altitude", 5.0, baro="altitude")
    rh = scen.raw(sim_h.inputs, sim_h.values)
    assert np.array_equal(rh[:, 0], sim_h.inputs["baro_alt"], equal_nan=True)


def test_raw_position2d_gnss_and_odometry():
    scen = fg.SCENARIOS["position2d"]
    sim = _sim("position2d", 40.0)
    r = scen.raw(sim.inputs, sim.values)
    g = sim.inputs["gnss"]
    assert r.shape == (sim.n, 4) and np.isnan(r[:, 2]).all()
    assert np.array_equal(r[:, :2], g, equal_nan=True) and np.array_equal(r[:, 3], sim.inputs["speed"])
    no_speed = {k: v for k, v in sim.inputs.items() if k != "speed"}
    assert np.isnan(scen.raw(no_speed)[:, 3]).all()
    # enlem / boylam: ilk düzeltme etrafında (kestiricilerle aynı), referans sözlükte; parça parça = tek sefer
    sim_ll = _sim_kw("position2d", 40.0, latlon=True)
    state: dict = {}
    chunked = _raw_chunked(scen, sim_ll.inputs, state, sim_ll.n, np.random.default_rng(4))
    first = np.flatnonzero(np.isfinite(g[:, 0]))[0]
    assert np.allclose(chunked[:, :2] + g[first], g, atol=2e-3, equal_nan=True)
    assert np.array_equal(np.isnan(chunked[:, 0]), np.isnan(g[:, 0]))
    assert np.allclose(state["_origin"], sim_ll.inputs["gnss_ll"][first])
    assert np.array_equal(chunked, scen.raw(sim_ll.inputs, {}), equal_nan=True)


def test_raw_linear_multi_inverse_variance_mean():
    scen = fg.SCENARIOS["linear_multi"]
    sim = _sim("linear_multi", 10.0)
    r = scen.raw(sim.inputs, sim.values)
    assert r.shape == (sim.n, 3) and np.isnan(r[:, 2]).all()     # ivme türünde sensör yok
    assert np.allclose(r[:, 1], sim.inputs["s4"], rtol=1e-12, equal_nan=True)
    zs = np.column_stack([sim.inputs[f"s{i}"] for i in (1, 2, 3)])
    w = 1.0 / np.array([0.03, 0.10, 0.20]) ** 2
    ok = np.isfinite(zs)
    expected = np.where(ok, zs, 0) @ w / (ok @ w)
    assert np.allclose(r[:, 0], expected)
    # o örnekte konum ölçümü yoksa NaN (durumsuz: son değer tutulmaz)
    only_s2 = {"s2": sim.inputs["s2"]}
    r2 = scen.raw(only_s2, sim.values)
    assert np.allclose(r2[:, 0], sim.inputs["s2"], rtol=1e-12, equal_nan=True) and np.isnan(r2[:, 1:]).all()
    # anlık taban çizgisiyle uyum (her örnekte s1 olduğundan tutma devreye girmez)
    wm = fg.make_estimator("linear_multi", "weighted_mean", sim.values).run(sim.inputs, sim.dt)
    assert np.allclose(wm[:, :2], r[:, :2], rtol=1e-12)
    with pytest.raises(fg.FusionError, match="Bilinmeyen giriş"):
        scen.raw({"s9": sim.inputs["s1"]}, sim.values)


# ============================================================================ hatalar ve doğrulama
def test_errors_wraps_angles_and_keeps_nan():
    est = np.array([[0.0, 0.0, math.radians(179.0), 5.0], [400.0, 0.0, math.radians(-170.0), np.nan]])
    ref = np.array([[1.0, 0.0, math.radians(-179.0), 4.0], [0.0, np.nan, math.radians(170.0), 1.0]])
    err = fg.errors("position2d", est, ref)
    assert np.allclose(np.degrees(err[:, 2]), [-2.0, 20.0])
    assert err[0, 0] == -1.0 and err[1, 0] == 400.0           # açısal olmayan sütun sarılmaz
    assert np.isnan(err[1, 1]) and np.isnan(err[1, 3])
    r = fg.rmse(err)
    assert math.isclose(r[3], 1.0) and math.isclose(r[1], 0.0)
    assert np.isnan(fg.rmse(np.full((3, 1), np.nan)))[0]
    with pytest.raises(fg.FusionError):
        fg.errors("position2d", est[:, :3], ref[:, :3])


def test_validation_raises_clear_turkish_errors():
    E = fg.FusionError
    with pytest.raises(E, match="geçersiz"):
        fg.make_estimator("altitude", "kf", {"sigma_p": -1.0})
    with pytest.raises(E, match="geçersiz"):
        fg.make_estimator("altitude", "kf", {"sigma_acc": float("nan")})
    with pytest.raises(E, match="bilinmeyen parametre"):
        fg.make_estimator("altitude", "kf", {"sigma_x": 1.0})
    with pytest.raises(E, match="seçim"):
        fg.make_estimator("linear_multi", "kf", {"order": "kubik"})
    with pytest.raises(E, match="p0"):
        fg.make_estimator("altitude", "kf", {"p0": 500.0})
    with pytest.raises(E, match="sayı"):
        fg.make_estimator("altitude", "complementary", {}, {"tau": "hızlı"})
    with pytest.raises(E, match="senaryo"):
        fg.make_estimator("nope", "kf")
    with pytest.raises(E, match="algoritma"):
        fg.make_estimator("altitude", "lqr")
    est = fg.make_estimator("altitude", "kf")
    acc = np.full(10, 9.8)
    with pytest.raises(E, match="en az biri"):
        est.run({"acc_z": acc}, 0.01)
    with pytest.raises(E, match="Zorunlu"):
        est.run({"baro_p": acc}, 0.01)
    with pytest.raises(E, match="Bilinmeyen giriş"):
        est.run({"acc_z": acc, "baro_p": acc, "gps": acc}, 0.01)
    with pytest.raises(E, match="aralığı"):
        est.run({"acc_z": acc, "baro_p": acc}, 0.0)
    with pytest.raises(E, match="aynı uzunlukta"):
        est.run({"acc_z": acc, "baro_p": acc[:5]}, 0.01)
    with pytest.raises(E, match="skaler"):
        est.run({"acc_z": np.zeros((10, 3)), "baro_p": acc}, 0.01)
    est.run({"acc_z": acc, "baro_p": np.full(10, 101325.0)}, 0.01)
    with pytest.raises(E, match="değişemez"):
        est.run({"acc_z": acc, "baro_alt": acc}, 0.01)
    with pytest.raises(E, match="\\(n, 3\\)"):
        fg.make_estimator("orientation", "mahony").run({"gyro": np.zeros((5, 2)), "acc": np.zeros((5, 3))}, 0.01)
    # Yuva türü ↔ model derecesi: hem erken denetim (Scenario.check) hem ilk run
    values = {"order": "hiz", "s2_kind": "ivme"}
    with pytest.raises(E, match="Sensör 2"):
        fg.SCENARIOS["linear_multi"].check(["s1", "s2"], values)
    with pytest.raises(E, match="Sensör 2"):
        fg.make_estimator("linear_multi", "kf", values).run({"s1": acc, "s2": acc}, 0.01)
    fg.SCENARIOS["linear_multi"].check(["s1"], values)       # s2 eşlenmemişse sorun yok
    with pytest.raises(E, match="Zorunlu"):
        fg.SCENARIOS["position2d"].check(["gnss"])
    assert fg.make_estimator("linear_multi", "kf").run({"s1": acc}, 0.01).shape == (10, 3)
    assert fg.make_estimator("linear_multi", "kf").run({"s1": acc[:0]}, 0.01).shape == (0, 3)


def test_diagnostics_and_cost_are_reported():
    for key, algo, dur in (("altitude", "pf", 120.0), ("position2d", "ekf", 120.0), ("linear_multi", "kf", 60.0)):
        est, _ = _run(key, algo, dur)
        d = est.diagnostics()
        assert d and all(isinstance(k, str) and isinstance(v, str) for k, v in d.items())
        assert est.cost_us > 0
    est, _ = _run("altitude", "pf", 120.0)
    assert 0 < float(est.diagnostics()["ESS"]) <= 500
    est.reset()
    assert est.samples > 0 and not est.inited
