"""Genel (senaryo tabanlı) füzyon C kod üreteci: Python ile birebirlik (double; eksik / NaN ölçümlü akışlar ve farklı
giriş eşlemeleri dahil), float doğruluğu, parçacık filtresi istatistiği, UKF özdeğer yedeği, parametre denetimi,
sizeof = yapı düzeni, çapraz derleme uyarısızlığı (Cortex-M × GCC / Clang × float / double), ayak izi ve maliyet.

Derleyici gerektiren testler, derleyici yoksa atlanır. Derleme çıktıları içerik özetiyle önbelleklenir (ilk
çalıştırma birkaç dakika, sonrakiler hızlı)."""
import functools
import shutil
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from spt.codegen import fusion_generic_c as gc
from spt.codegen import toolchain as tc
from spt.codegen.mcu import MCU_PROFILES, OPS
from spt.dsp import fusion_generic as fg
from spt.dsp.fusion_scenarios import altitude as alt_mod

HOST_CC = tc.find_host_cc()
HOST_CXX = tc.find_host_cxx()
ARM_GCC = tc.find_arm_gcc()
CLANG = tc.find_clang()
needs_host = pytest.mark.skipif(HOST_CC is None, reason="ana makine C derleyicisi yok")
needs_cxx = pytest.mark.skipif(HOST_CXX is None, reason="ana makine C++ derleyicisi yok")
needs_cross = pytest.mark.skipif(ARM_GCC is None and CLANG is None, reason="ARM çapraz derleyici yok")

DURATION = {"altitude": 40.0, "position2d": 75.0, "linear_multi": 20.0}
DEFAULT_INPUTS = {"altitude": ("baro_p", "acc_z"), "position2d": ("gnss", "gyro_z", "speed"),
                  "linear_multi": ("s1", "s2", "s3", "s4")}


# ============================================================================ veri: simülasyon + eksik örnekler
@functools.lru_cache(maxsize=None)
def _sim(scen: str, seed: int = 3, **kw):
    return fg.SCENARIOS[scen].simulate(DURATION[scen], None, seed, **kw)


def _altitude_both(sim):
    """Basınç + ayrı hızda baro irtifa (araya serpiştirilmiş); basınç kesintisinde baro irtifa taşır."""
    rng = np.random.default_rng(11)
    n = sim.n
    h = np.full(n, np.nan)
    idx = np.arange(4, n, 8)
    h[idx] = sim.truth[idx, 0] + 0.35 * rng.standard_normal(len(idx))
    inputs = dict(sim.inputs, baro_alt=h)
    inputs["baro_p"] = inputs["baro_p"].copy()
    inputs["baro_p"][3000:4200] = np.nan                       # basınç kesintisi: baro irtifa taşır
    return inputs


def _inject(scen: str, inputs: dict) -> dict:
    """Eksik / geçersiz örnekler: başlatmayı geciktiren baştaki NaN, kesintiler, sonsuz, ≤ 0 basınç, tek eksenli GNSS."""
    x = {k: np.array(v, dtype=float, copy=True) for k, v in inputs.items()}
    if scen == "altitude":
        for key in ("baro_p", "baro_alt"):
            if key in x:
                x[key][:300] = np.nan                         # ilk baro ölçümü gecikir (çıktı NaN)
                x[key][5000:5400] = np.nan
        if "baro_p" in x:
            x["baro_p"][6000] = -5.0                          # geçersiz basınç → yok
            x["baro_p"][6008] = 0.0
        if "baro_alt" in x:
            x["baro_alt"][6004] = np.inf
        x["acc_z"][:150] = np.nan                             # başta ivme yok: a = 0
        x["acc_z"][1000:1100] = np.nan                        # kesinti: son değer + hız yürüyüşü
        x["acc_z"][2000] = np.inf
    elif scen == "position2d":
        x["gyro_z"][2000:2050] = np.nan
        x["gyro_z"][3000] = np.inf
        for key in ("gnss", "gnss_ll"):
            if key in x:
                x[key][:100] = np.nan                         # başlatma gecikir
                x[key][1500, 0] = np.nan                      # tek eksen eksik → düzeltme yok
                x[key][2500:2700] = np.nan
        if "speed" in x:
            x["speed"][4000:4100] = np.nan
        if "acc_fwd" in x:
            x["acc_fwd"][4200:4300] = np.nan
    else:
        first = next(k for k in ("s1", "s2", "s3", "s4", "s5", "s6") if k in x)
        for key in x:
            x[key][:50] = np.nan                              # hiç ölçüm yok: henüz kestirim yok
        x[first][:200] = np.nan
        for key in list(x)[1:]:
            x[key][700:760] = np.nan
        x[list(x)[-1]][900] = np.inf
    return x


# (kimlik, senaryo, simülasyon kwarg'ları, eşlenen girişler, ek senaryo değerleri, girişleri üreten fonksiyon)
CONFIGS = {
    "alt_pressure": ("altitude", {}, ("baro_p", "acc_z"), {}, None),
    "alt_baroalt": ("altitude", {"baro": "altitude"}, ("baro_alt", "acc_z"), {}, None),
    "alt_both_autop0": ("altitude", {}, ("baro_p", "baro_alt", "acc_z"), {"p0": 0.0, "sigma_h": 0.35},
                        _altitude_both),
    "pos_local": ("position2d", {}, ("gnss", "gyro_z", "speed"), {}, None),
    "pos_latlon_acc": ("position2d", {"latlon": True, "speed": False, "acc": True}, ("gnss_ll", "gyro_z", "acc_fwd"),
                       {}, None),
    "pos_both_gnss": ("position2d", {"acc": True}, ("gnss", "gnss_ll", "gyro_z", "speed", "acc_fwd"), {}, "both_gnss"),
    # 10 Hz GNSS + uzun kiriş: başlatma geçmişi 160 düzeltmeyle 16 s'ye düşüyordu (C 29.8 s'de, Python 19 s'de
    # başlıyordu); artık ikisi de seyreltilmiş 30 s geçmişle aynı örnekte başlar
    "pos_fast_gnss": ("position2d", {"gnss_rate": 10.0}, ("gnss", "gyro_z", "speed"), {"d_init": 200.0}, None),
    "lin_default": ("linear_multi", {}, ("s1", "s2", "s3", "s4"), {}, None),
    "lin_accel": ("linear_multi", {"sensors": (("konum", 0.05, 50.0), ("ivme", 0.1, 100.0)), "order": "ivme"},
                  ("s1", "s2"), {}, None),
    "lin_level": ("linear_multi", {"sensors": (("konum", 0.03, 100.0), ("konum", 0.2, 20.0)), "order": "seviye"},
                  ("s1", "s2"), {}, None),
    "lin_relative": ("linear_multi", {"sensors": (("hiz", 0.02, 100.0), ("hiz", 0.05, 50.0)), "order": "hiz"},
                     ("s1", "s2"), {}, None),
}


@functools.lru_cache(maxsize=None)
def _case(cid: str):
    """(seçenekler, temiz girişler, eksik örnekli girişler, simülasyon)."""
    scen, kw, inputs, extra, make = CONFIGS[cid]
    sim = _sim(scen, **kw)
    values = dict(sim.values, **extra)
    if make == "both_gnss":
        ll = fg.SCENARIOS[scen].simulate(DURATION[scen], None, 3, latlon=True, acc=True).inputs["gnss_ll"]
        data = dict(sim.inputs, gnss_ll=ll)
    elif make is not None:
        data = make(sim)
    else:
        data = dict(sim.inputs)
    data = {k: v for k, v in data.items() if k in inputs}
    opts = gc.GenericCodegenOptions(scen, gc.SUPPORTED[scen], values, {}, inputs, fs_hz=sim.fs)
    return opts, data, _inject(scen, data), sim


@functools.lru_cache(maxsize=None)
def _python(cid: str, algo: str, dropped: bool, seed: int | None = None):
    opts, clean, drop, sim = _case(cid)
    params = {"seed": seed} if seed is not None else {}
    return fg.make_estimator(opts.scenario, algo, opts.scenario_values, params).run(drop if dropped else clean, sim.dt)


def _c(cid: str, algo: str, dropped: bool, real: str, seed: int | None = None) -> np.ndarray:
    opts, clean, drop, sim = _case(cid)
    if seed is not None:
        opts = gc.GenericCodegenOptions(opts.scenario, ("pf",), opts.scenario_values, {"pf": {"seed": seed}},
                                        opts.inputs, fs_hz=opts.fs_hz)
    return gc.run_generated(opts, algo, drop if dropped else clean, sim.dt, real=real).out


def _rmse(scen, y, truth, skip):
    return fg.rmse(fg.errors(scen, y[skip:], truth[skip:]))


DET_CASES = [(cid, algo) for cid, cfg in CONFIGS.items() for algo in gc.DETERMINISTIC[cfg[0]]]


# ============================================================================ üretim ve seçenekler
def test_generated_files_parameters_and_prefix():
    opts = gc.GenericCodegenOptions("altitude", ("kf", "pf", "complementary"), {"sigma_p": 3.0},
                                    {"pf": {"particles": 200, "seed": 9}, "complementary": {"bias": "off"}},
                                    ("baro_p", "acc_z"))
    files = {f.name: f.text for f in gc.generate(opts)}
    assert set(files) == {"spt_fusion_generic_common.h", "spt_alt_common.h", "spt_alt_complementary.h",
                          "spt_alt_complementary.c", "spt_alt_kf.h", "spt_alt_kf.c", "spt_alt_pf.h", "spt_alt_pf.c",
                          "ornek_kullanim_alt.c", "KULLANIM_alt.md"}  # yönelim üreticisinin adlarıyla çakışmaz
    assert all("{{" not in t for t in files.values())
    assert "#define SPT_ALT_SIGMA_P 3.0" in files["spt_alt_common.h"]
    assert "#define SPT_ALT_P0 101325.0" in files["spt_alt_common.h"]
    assert "#define SPT_ALT_HAS_BARO_P 1" in files["spt_alt_common.h"]
    assert "#define SPT_ALT_HAS_BARO_ALT 0" in files["spt_alt_common.h"]
    assert "#define SPT_ALT_PF_PARTICLES 200" in files["spt_alt_pf.h"]
    assert "#define SPT_ALT_PF_SEED 9u" in files["spt_alt_pf.h"]
    assert "#define SPT_ALT_COMPLEMENTARY_BIAS 0" in files["spt_alt_complementary.h"]
    # API yalnız eşlenen girişleri alır (Scenario.inputs sırasıyla), çıktı alıcıları Scenario.outputs sırasında
    assert "int spt_alt_kf_step(spt_alt_kf_t *s, SPT_REAL baro_p, SPT_REAL acc_z, SPT_REAL dt);" in files["spt_alt_kf.h"]
    assert "SPT_REAL spt_alt_kf_alt(const spt_alt_kf_t *s);" in files["spt_alt_kf.h"]
    assert "SPT_REAL spt_alt_kf_vz(const spt_alt_kf_t *s);" in files["spt_alt_kf.h"]
    assert "return spt_alt_kf_core(s, baro_p, SPT_NAN, acc_z, dt);" in files["spt_alt_kf.c"]
    # 2B konum: vektör girişler dizi (NULL = yok), enlem / boylam her zaman double
    opts = gc.GenericCodegenOptions("position2d", ("ekf",), {}, {}, ("gyro_z", "gnss_ll", "speed"), prefix="nav")
    files = {f.name: f.text for f in gc.generate(opts)}
    assert set(files) == {"nav_fusion_generic_common.h", "nav_pos_common.h", "nav_pos_ekf.h", "nav_pos_ekf.c",
                          "ornek_kullanim_pos.c", "KULLANIM_pos.md"}
    assert "@file ornek_kullanim_pos.c" in files["ornek_kullanim_pos.c"]
    assert "kullanım: KULLANIM_pos.md" in files["nav_pos_ekf.h"] and "KULLANIM.md" not in files["nav_pos_ekf.h"]
    assert ("int nav_pos_ekf_step(nav_pos_ekf_t *s, const double gnss_ll[2], NAV_REAL gyro_z, NAV_REAL speed, "
            "NAV_REAL dt);") in files["nav_pos_ekf.h"]
    assert "NAV_REAL nav_pos_ekf_heading(const nav_pos_ekf_t *s);" in files["nav_pos_ekf.h"]
    for name, text in files.items():
        if name.endswith((".c", ".h")):
            assert "spt_" not in text and "SPT_" not in text, name
    # doğrusal: seçimler tamsayı makro, eşlenen yuvalar maskesi
    opts = gc.GenericCodegenOptions("linear_multi", ("kf",), {"order": "ivme", "s3_kind": "ivme"}, {}, ("s1", "s3"))
    files = {f.name: f.text for f in gc.generate(opts)}
    assert "#define SPT_LIN_ORDER 2" in files["spt_lin_common.h"] and "#define SPT_LIN_S3_KIND 2" in files[
        "spt_lin_common.h"]
    assert "#define SPT_LIN_MAPPED 0x05u" in files["spt_lin_common.h"]
    assert "const SPT_REAL z[6] = {s1, SPT_NAN, s3, SPT_NAN, SPT_NAN, SPT_NAN};" in files["spt_lin_kf.c"]


def test_invalid_options_are_rejected():
    ok = dict(scenario="altitude", algorithms=("kf",), scenario_values={}, params={}, inputs=("baro_p", "acc_z"))
    bad = [dict(scenario="orientation"), dict(scenario="yok"), dict(algorithms=()), dict(algorithms=("madgwick",)),
           dict(inputs=("acc_z",)),                                   # baro_p / baro_alt en az biri
           dict(inputs=("baro_p",)),                                  # acc_z zorunlu
           dict(inputs=("baro_p", "acc_z", "gyro_z")),                # bilinmeyen giriş
           dict(scenario_values={"p0": 500.0}),                       # 0 < p0 < 1000 (Python FusionError)
           dict(scenario_values={"sigma_x": 1.0}), dict(params={"kf": {"tau": 1.0}}),
           dict(params={"pf": {"particles": 10}}), dict(real="half"), dict(prefix="1a"), dict(mcu="z80"),
           dict(compiler="msvc"), dict(opt="-O9")]
    for b in bad:
        with pytest.raises(ValueError):
            gc.generate(gc.GenericCodegenOptions(**dict(ok, **b)))
    with pytest.raises(ValueError):                                   # yuva türü > model derecesi
        gc.generate(gc.GenericCodegenOptions("linear_multi", ("kf",), {"order": "seviye", "s2_kind": "hiz"}, {},
                                             ("s1", "s2")))
    gc.generate(gc.GenericCodegenOptions(**dict(ok, scenario_values={"_p0": 1.0, "p0": 0.0})))   # "_" yok sayılır
    assert set(gc.SUPPORTED) == {"altitude", "position2d", "linear_multi"}
    for scen, algos in gc.SUPPORTED.items():
        assert algos == tuple(fg.SCENARIOS[scen].algorithms)


# ============================================================================ Python ile birebirlik (double)
@needs_host
@pytest.mark.parametrize("dropped", [False, True], ids=["clean", "dropouts"])
@pytest.mark.parametrize("cid,algo", DET_CASES, ids=[f"{c}-{a}" for c, a in DET_CASES])
def test_double_matches_python(cid, algo, dropped):
    """Çift hassasiyette C = Python: aynı örneklerde NaN, en büyük fark < 1e-6 (çıktı biriminde, açılar sarılı)."""
    opts, _, _, sim = _case(cid)
    y_py = _python(cid, algo, dropped)
    y_c = _c(cid, algo, dropped, "double")
    assert np.array_equal(np.isnan(y_py), np.isnan(y_c))
    ok = ~np.isnan(y_py)
    assert ok.any(axis=0).any()
    diff = np.abs(fg.errors(opts.scenario, y_c, y_py))
    assert np.nanmax(diff) < 1e-6, np.nanmax(diff, axis=0)
    if dropped:
        assert np.isnan(y_py).any()                                   # baştaki eksik ölçümler: henüz kestirim yok


def test_injected_dropouts_exercise_python_rules():
    """Eksik örnek setlerinin gerçekten ilgili kuralları tetiklediği (başlatma gecikmesi, kesinti) Python'da görülür."""
    y = _python("alt_pressure", "kf", True)
    assert np.isnan(y[:300]).all() and np.isfinite(y[310:]).all()
    y = _python("pos_local", "ekf", True)
    assert np.isnan(y[:100]).all() and np.isfinite(y[-1]).all()
    y = _python("lin_default", "kf", True)
    assert np.isnan(y[:50]).all() and np.isfinite(y[200:]).all()


# ============================================================================ parçacık filtreleri (istatistik)
PF_CASES = ("alt_pressure", "alt_baroalt", "pos_local", "lin_default", "lin_accel")


@needs_host
@pytest.mark.parametrize("cid", PF_CASES)
def test_particle_filter_statistically_equivalent(cid):
    """Farklı rastgele sayı üreteci: üç tohumun medyan RMSE'si ≤ Python medyanı × 1.5 + küçük pay; başlatmadan sonra NaN
    yok, eksik örnekli akışta NaN konumları Python ile aynı."""
    opts, _, _, sim = _case(cid)
    scen = opts.scenario
    skip = int(5 * sim.fs)
    seeds = (1, 2, 3)
    e_py = np.median([_rmse(scen, _python(cid, "pf", False, s), sim.truth, skip) for s in seeds], axis=0)
    runs = [_c(cid, "pf", False, "double", s) for s in seeds]
    e_c = np.median([_rmse(scen, y, sim.truth, skip) for y in runs], axis=0)
    margin = {"altitude": 0.02, "position2d": 0.1, "linear_multi": 0.005}[scen]
    for y, y_py in zip(runs, (_python(cid, "pf", False, s) for s in seeds)):
        assert np.array_equal(np.isnan(y), np.isnan(y_py))
    valid = np.isfinite(e_py)
    assert np.all(e_c[valid] <= e_py[valid] * 1.5 + margin), (e_c, e_py)
    y_drop, y_py_drop = _c(cid, "pf", True, "double"), _python(cid, "pf", True)
    assert np.array_equal(np.isnan(y_drop), np.isnan(y_py_drop))
    started = np.isfinite(y_drop).all(axis=1)
    assert started.any() and started[int(np.argmax(started)):].all()      # başlatmadan sonra NaN yok


# ============================================================================ float doğruluğu
FLOAT_CASES = [(cid, algo) for cid in ("alt_pressure", "alt_baroalt", "pos_local", "lin_default", "lin_accel")
               for algo in gc.DETERMINISTIC[CONFIGS[cid][0]]]


@needs_host
@pytest.mark.parametrize("cid,algo", FLOAT_CASES, ids=[f"{c}-{a}" for c, a in FLOAT_CASES])
def test_float_accuracy_matches_python(cid, algo):
    """float: çıktı başına RMSE ≤ Python (double) RMSE × 1.05 + 1e-4 (birim: m, m/s, rad …)."""
    opts, _, _, sim = _case(cid)
    skip = int(5 * sim.fs)
    y_py = _python(cid, algo, False)
    y_f = _c(cid, algo, False, "float")
    assert np.array_equal(np.isnan(y_py), np.isnan(y_f))
    e_py = _rmse(opts.scenario, y_py, sim.truth, skip)
    e_f = _rmse(opts.scenario, y_f, sim.truth, skip)
    valid = np.isfinite(e_py)
    assert np.all(e_f[valid] <= e_py[valid] * 1.05 + 1e-4), (e_f, e_py)


@needs_host
def test_float_particle_filters_finite_and_accurate():
    for cid in ("alt_pressure", "pos_local", "lin_default"):
        opts, _, _, sim = _case(cid)
        skip = int(5 * sim.fs)
        y_f = _c(cid, "pf", False, "float")
        y_py = _python(cid, "pf", False)
        assert np.array_equal(np.isnan(y_py), np.isnan(y_f))
        e_py, e_f = _rmse(opts.scenario, y_py, sim.truth, skip), _rmse(opts.scenario, y_f, sim.truth, skip)
        valid = np.isfinite(e_py)
        assert np.all(e_f[valid] <= e_py[valid] * 1.5 + 0.1), (cid, e_f, e_py)


# ============================================================================ özel yollar: UKF yedeği, parametreler
def _arr(v) -> str:
    return ", ".join(float(x).hex() for x in np.ravel(v))


@needs_host
def test_position2d_ukf_eigen_fallback_matches_python():
    """Cholesky başarısız (belirsiz P) → Jacobi özdeğer yedeği; Python eigh yedeğiyle aynı adım."""
    opts, clean, _, sim = _case("pos_local")
    est = fg.make_estimator("position2d", "ukf", opts.scenario_values, {})
    k0 = 1000
    est.run({k: v[:k0] for k, v in clean.items()}, sim.dt)
    rng = np.random.default_rng(3)
    vecs = np.linalg.qr(rng.normal(size=(6, 6)))[0]
    vals = np.array([4.0, 1.0, 1e-3, -2e-4, 1e-5, 1e-7])            # ayrık özdeğerler, biri negatif
    p_bad = vecs @ np.diag(vals) @ vecs.T
    p_bad = 0.5 * (p_bad + p_bad.T)
    with pytest.raises(np.linalg.LinAlgError):
        np.linalg.cholesky(6 * p_bad)
    x0 = est.x.copy()
    est.P = p_bad.copy()
    k = k0                                                             # GNSS + hız olan örnek
    while not np.isfinite(clean["gnss"][k]).all():
        k += 1
    est.run({key: v[k0:k + 1] for key, v in clean.items()}, sim.dt)
    y_py = np.array([est.x[0], est.x[1], est.x[2], est.x[3]])
    # C: aynı durumdan (x0, belirsiz P) aynı k0…k örnekleri (ilk adımda Cholesky başarısız → özdeğer yedeği)
    rows = []
    for i in range(k0, k + 1):
        gn = clean["gnss"][i]
        rows.append((float(clean["gyro_z"][i]), float(clean["speed"][i]), float(gn[0]), float(gn[1])))
    data = ", ".join(_arr(r) for r in rows)
    main = f"""#include <stdio.h>
#include "spt_pos_ukf.h"
int main(void)
{{
    static spt_pos_ukf_t s;
    const double x0[6] = {{{_arr(x0)}}}, p[36] = {{{_arr(p_bad)}}}, d[{4 * len(rows)}] = {{{data}}};
    double out[4];
    int i;
    spt_pos_ukf_init(&s);
    s.c.ready = 1;
    s.c.has_w = 1;
    for (i = 0; i < 6; ++i) s.x[i] = x0[i];
    for (i = 0; i < 36; ++i) s.P[i / 6][i % 6] = p[i];
    for (i = 0; i < {len(rows)}; ++i) {{
        const double gn[2] = {{d[4 * i + 2], d[4 * i + 3]}};
        (void)spt_pos_ukf_step(&s, gn, d[4 * i], d[4 * i + 1], {sim.dt!r});
    }}
    spt_pos_ukf_output(&s, out);
    printf("%a %a %a %a %u\\n", out[0], out[1], out[2], out[3], (unsigned)s.fallbacks);
    return 0;
}}
"""
    o = gc.GenericCodegenOptions("position2d", ("ukf",), opts.scenario_values, {}, ("gnss", "gyro_z", "speed"),
                                 real="double")
    files = {f.name: f.text for f in gc.generate(o)}
    files["fallback_main.c"] = main
    build = tc.host_build(files, ["spt_pos_ukf.c", "fallback_main.c"], extra=["-std=c99"], output="pos_ukf_fallback")
    assert build.ok, build.log
    out = tc.run_program(build, []).stdout.split()
    y_c = np.array([float.fromhex(t) for t in out[:4]])
    assert int(out[4]) >= 1
    assert np.max(np.abs(fg.errors("position2d", y_c[None], y_py[None]))) < 1e-6, (y_c, y_py)


@needs_host
def test_runtime_parameter_checks_and_null_inputs():
    """init / reset Python'un FusionError kurallarını denetler; geçersizken step 0 döner ve çıktı NaN kalır."""
    alt = gc.GenericCodegenOptions("altitude", ("kf",), {}, {}, ("baro_p", "acc_z"), real="double")
    lin = gc.GenericCodegenOptions("linear_multi", ("kf", "pf"), {"order": "hiz", "s2_kind": "hiz"}, {"pf": {"particles": 60}},
                                   ("s1", "s2"), real="double")
    pos = gc.GenericCodegenOptions("position2d", ("ekf",), {}, {}, ("gnss", "gyro_z"), real="double")
    files = {}
    for o in (alt, lin, pos):
        files.update({f.name: f.text for f in gc.generate(o) if f.name.endswith((".c", ".h"))})
    for name in [n for n in files if n.startswith("ornek_kullanim")]:
        files.pop(name)
    files["check_main.c"] = r"""#include <stdio.h>
#include "spt_alt_kf.h"
#include "spt_lin_kf.h"
#include "spt_lin_pf.h"
#include "spt_pos_ekf.h"
int main(void)
{
    static spt_alt_kf_t a;
    static spt_lin_kf_t l;
    static spt_lin_pf_t q;
    static spt_pos_ekf_t p;
    double out[4];
    int r1 = spt_alt_kf_init(&a), r2, r3, r4, r5, r6, r7, r8, k, ok = 0;
    a.cfg.p0 = 500.0;
    r2 = spt_alt_kf_reset(&a);
    r3 = spt_alt_kf_step(&a, 101000.0, 9.8, 0.01);
    spt_alt_kf_output(&a, out);
    a.cfg.p0 = 0.0;
    r4 = spt_alt_kf_reset(&a) && spt_alt_kf_step(&a, 101000.0, 9.8, 0.01) && spt_alt_kf_alt(&a) == 0.0;
    r5 = spt_lin_kf_init(&l);
    l.cfg.order = 0;                               /* s2 hız ölçüyor: derece en az "sabit hız" olmalı */
    r6 = spt_lin_kf_reset(&l);
    r7 = spt_lin_pf_init(&q);
    q.cfg.order = 2;                               /* PF durum dizisi derece + 1 = 2 boyutlu derlendi */
    r7 = r7 && !spt_lin_pf_reset(&q);
    r8 = spt_pos_ekf_init(&p);
    for (k = 0; k < 400; ++k) {                    /* GNSS hep NULL: başlatma yok, x / y NaN */
        ok += spt_pos_ekf_step(&p, NULL, 0.0, 0.01);
    }
    printf("%d %d %d %d %d %d %d %d %d %d\n", r1, r2, r3, out[0] != out[0], r4, r5, r6, r7, r8,
           ok == 0 && spt_pos_ekf_x(&p) != spt_pos_ekf_x(&p));
    return 0;
}
"""
    srcs = ["spt_alt_kf.c", "spt_lin_kf.c", "spt_lin_pf.c", "spt_pos_ekf.c", "check_main.c"]
    build = tc.host_build(files, srcs, defines=["SPT_USE_DOUBLE=1"], extra=["-std=c99"], output="param_check")
    assert build.ok, build.log
    out = tc.run_program(build, []).stdout.split()
    assert out == ["1", "0", "0", "1", "1", "1", "0", "1", "1", "1"], out


@needs_host
def test_state_size_matches_sizeof_double_and_float():
    """state_size (yapı düzeninden) = sizeof, double ve float (ana makine x86-64 düzeni = ARM AAPCS: işaretçi yok)."""
    for real in ("double", "float"):
        for scen in gc.SUPPORTED:
            values = {"order": "ivme"} if scen == "linear_multi" else {}
            o = gc.GenericCodegenOptions(scen, gc.SUPPORTED[scen], values, {"pf": {"particles": 77}},
                                         DEFAULT_INPUTS[scen], real=real)
            files = {f.name: f.text for f in gc.generate(o) if f.name.endswith(".h")}
            sab = {"altitude": "alt", "position2d": "pos", "linear_multi": "lin"}[scen]
            incs = "".join(f'#include "spt_{sab}_{a}.h"\n' for a in o.algorithms)
            prints = "".join(f'    printf("%u ", (unsigned)sizeof(spt_{sab}_{a}_t));\n' for a in o.algorithms)
            files["sz.c"] = f"#include <stdio.h>\n{incs}int main(void)\n{{\n{prints}    return 0;\n}}\n"
            build = tc.host_build(files, ["sz.c"], defines=[f"SPT_USE_DOUBLE={int(real == 'double')}"],
                                  extra=["-std=c99"], output=f"sz_{sab}_{real}")
            assert build.ok, build.log
            sizes = [int(t) for t in tc.run_program(build, []).stdout.split()]
            assert sizes == [gc.state_size(o, a) for a in o.algorithms], (scen, real, sizes)


# ============================================================================ çapraz derleme
def _compile_job(name, files, sources, prof, compiler, defines, extra):
    b = tc.compile_cross(files, sources, prof, compiler, "-Os", defines=defines, extra=extra, stack_usage=False)
    return name, b


@needs_cross
def test_cross_compile_warning_free_all_cortex_m():
    """Her senaryo (tüm algoritmalar + örnek dosya) × her yerel Cortex-M profili × GCC / Clang × float / double:
    -std=c99 -Wall -Wextra -pedantic -Werror temiz (float'ta ayrıca -Wdouble-promotion). Farklı giriş eşlemeleri ve
    önekli üretim de Cortex-M4F'de derlenir."""
    jobs = []
    for scen in gc.SUPPORTED:
        o = gc.GenericCodegenOptions(scen, gc.SUPPORTED[scen], {}, {}, DEFAULT_INPUTS[scen])
        files = {f.name: f.text for f in gc.generate(o)}
        sources = [n for n in files if n.endswith(".c")]
        for prof in MCU_PROFILES.values():
            for comp in ("gcc", "clang"):
                if tc.cross_flags(prof, comp) is None:
                    continue
                for real in ("0", "1"):
                    extra = ["-Wdouble-promotion"] if real == "0" else []
                    jobs.append((f"{scen}/{prof.key}/{comp}/{real}", files, sources, prof, comp,
                                 [f"SPT_USE_DOUBLE={real}"], extra))
    m4f = MCU_PROFILES["cortex-m4f"]
    variants = [("altitude", ("baro_p", "baro_alt", "acc_z"), {"p0": 0.0}),
                ("position2d", ("gnss", "gnss_ll", "gyro_z", "speed", "acc_fwd"), {}),
                ("position2d", ("gnss_ll", "gyro_z"), {}),
                ("linear_multi", ("s2", "s6"), {"order": "ivme", "s6_kind": "ivme"})]
    for scen, inputs, values in variants:
        o = gc.GenericCodegenOptions(scen, gc.SUPPORTED[scen], values, {}, inputs, prefix="gf")
        files = {f.name: f.text for f in gc.generate(o)}
        for comp in ("gcc", "clang"):
            if tc.cross_flags(m4f, comp) is not None:
                for real in ("0", "1"):
                    jobs.append((f"{scen}/{'+'.join(inputs)}/{comp}/{real}", files,
                                 [n for n in files if n.endswith(".c")], m4f, comp, [f"GF_USE_DOUBLE={real}"],
                                 ["-Wdouble-promotion"] if real == "0" else []))
    if not jobs:
        pytest.skip("çapraz derleyici yok")
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda j: _compile_job(*j), jobs))
    failures = [f"{name}:\n{b.log}" for name, b in results if not b.ok or b.log.strip()]
    assert not failures, "\n\n".join(failures[:5])
    assert len(results) >= 6


@needs_cxx
def test_generated_code_is_valid_cpp():
    """C++ projelerinden de derlenebilir: -x c++ -Wall -Wextra -pedantic -Werror temiz."""
    for scen in gc.SUPPORTED:
        o = gc.GenericCodegenOptions(scen, gc.SUPPORTED[scen], {}, {}, DEFAULT_INPUTS[scen])
        folder = gc.GEN_ROOT / "cxx_check" / scen
        shutil.rmtree(folder, ignore_errors=True)
        gc.write_files(gc.generate(o), folder)
        sources = sorted(p.name for p in folder.glob("*.c"))
        for real in ("0", "1"):
            res = tc._run([HOST_CXX.path, "-x", "c++", "-std=c++11", *tc.WARN_FLAGS, f"-DSPT_USE_DOUBLE={real}",
                           "-fsyntax-only", *sources], cwd=folder, path_prefix=[HOST_CXX.bin_dir])
            assert res.returncode == 0 and not res.stderr.strip(), res.stderr


# ============================================================================ ayak izi
@needs_cross
def test_footprint_measured_on_cortex_m4f():
    def one(scen):
        o = gc.GenericCodegenOptions(scen, gc.SUPPORTED[scen], {}, {}, DEFAULT_INPUTS[scen], real="float",
                                     mcu="cortex-m4f", opt="-Os")
        return scen, o, gc.footprint(o, ("gcc", "clang", "iar"))

    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(one, gc.SUPPORTED))
    for scen, o, rows in results:
        by = {(r.algorithm, r.compiler): r for r in rows}
        for comp, tool in (("gcc", ARM_GCC), ("clang", CLANG)):
            if tool is None:
                continue
            for a in gc.SUPPORTED[scen]:
                r = by[(a, comp)]
                assert r.measured and "ölçüldü" in r.method, r
                assert r.ram_state == gc.state_size(o, a)              # sizeof = yapı düzeni
                assert 200 < r.flash_code < 20_000, r
                if ARM_GCC is not None:
                    assert r.flash_with_libm >= r.flash_code, r
                    assert r.stack >= r.stack_own > 0, r
            assert by[("pf", comp)].ram_state > 500 * 4 * 3                # parçacık dizileri (500 × d × float)
            det = [a for a in gc.SUPPORTED[scen] if a != "pf"]
            assert by[(det[0], comp)].flash_code < by[("pf", comp)].flash_code
        if ARM_GCC is not None:
            for a in gc.SUPPORTED[scen]:
                iar, gcc = by[(a, "iar")], by[(a, "gcc")]
                assert not iar.measured and "tahmin" in iar.method
                assert iar.flash_code == round(gcc.flash_code * 0.9) and iar.ram_state == gcc.ram_state


def test_footprint_degrades_without_compilers(monkeypatch):
    monkeypatch.setattr(tc, "find_arm_gcc", lambda: None)
    monkeypatch.setattr(tc, "find_clang", lambda: None)
    for scen in gc.SUPPORTED:
        o = gc.GenericCodegenOptions(scen, gc.SUPPORTED[scen], {}, {}, DEFAULT_INPUTS[scen], real="double",
                                     mcu="cortex-m3")
        rows = gc.footprint(o, ("gcc", "clang", "iar"))
        assert len(rows) == 3 * len(gc.SUPPORTED[scen])
        for r in rows:
            assert not r.measured and "tahmin" in r.method and r.flash_code > 0
            assert r.ram_state == gc.state_size(o, r.algorithm)


# ============================================================================ maliyet
@needs_cxx
def test_cost_ops_counted_and_load_formula():
    for scen in gc.SUPPORTED:
        o = gc.GenericCodegenOptions(scen, gc.SUPPORTED[scen], {}, {}, DEFAULT_INPUTS[scen], mcu="cortex-m4f",
                                     clock_mhz=168.0, fs_hz=100.0)
        rows = gc.cost(o)
        assert [r.algorithm for r in rows] == list(gc.SUPPORTED[scen])
        by = {r.algorithm: r for r in rows}
        for r in rows:
            assert r.us_per_update == pytest.approx(r.cycles / o.clock_mhz)
            assert r.cpu_load_percent == pytest.approx(r.us_per_update * o.fs_hz / 1e4)
            assert r.max_rate_hz == pytest.approx(1e6 / r.us_per_update)
            assert r.cycles_max >= r.cycles > 0 and "ölçüldü" in r.method and "tahmin" in r.method
            assert set(r.ops) == set(OPS)
        assert by["pf"].ops["rng"] > 500                                # parçacık başına ≥ 1 normal sayı
        assert by["pf"].cycles > 20 * max(by[a].cycles for a in gc.DETERMINISTIC[scen])
        if scen == "altitude":
            assert by["complementary"].cycles < by["kf"].cycles < by["ukf"].cycles
            assert by["kf"].ops["sin"] == 0 and by["kf"].ops["sqrt"] == 0
        elif scen == "position2d":
            assert by["dead_reckoning"].cycles < by["ekf"].cycles < by["ukf"].cycles
            assert by["ekf"].ops["sin"] == pytest.approx(1.0) and by["ekf"].ops["cos"] == pytest.approx(1.0)
            assert by["ukf"].ops["sin"] >= 13
        else:
            assert by["weighted_mean"].cycles < by["kf"].cycles
    # işlem sayıları belirleyici: koşu önbelleği silinip yeniden çalıştırılınca aynı
    o = gc.GenericCodegenOptions("altitude", ("pf",), {}, {}, ("baro_p", "acc_z"))
    sim = fg.SCENARIOS["altitude"].simulate(1.0, 100.0, 2)
    first = gc.run_generated(o, "pf", sim.inputs, sim.dt, count_ops=True).ops
    shutil.rmtree(gc.GEN_ROOT / "runs", ignore_errors=True)
    second = gc.run_generated(o, "pf", sim.inputs, sim.dt, count_ops=True).ops
    assert np.array_equal(first, second)


@needs_cxx
def test_cost_soft_double_is_much_slower_on_single_precision_core():
    f = gc.cost(gc.GenericCodegenOptions("altitude", ("kf",), {}, {}, ("baro_p", "acc_z")))[0]
    d = gc.cost(gc.GenericCodegenOptions("altitude", ("kf",), {}, {}, ("baro_p", "acc_z"), real="double"))[0]
    assert d.ops == f.ops and d.cycles > 10 * f.cycles


# ============================================================================ kılavuz
def test_usage_markdown_contents():
    fp = [gc.FootprintRow("kf", "gcc", 1000, 5000, 120, 300, True, "ölçüm yöntemi", 70)]
    cost = [gc.CostRow("kf", {op: 1.0 for op in OPS}, 800.0, 4.76, 0.095, 210000.0, "yöntem", False, 900.0, 5.4)]
    o = gc.GenericCodegenOptions("altitude", ("kf", "pf"), {"p0": 0.0}, {}, ("baro_p", "acc_z"))
    text = gc.usage_markdown(o, fp, cost)
    for needle in ("## Dosyalar", "## Girişler", "## Çıktılar", "## Eksik ölçüm (NaN)", "SPT_NAN", "STM32CubeIDE",
                   "Keil MDK", "## Parçacık filtresi", "SPT_ALT_PF_PARTICLES", "## Ölçülen ayak izi",
                   "## Hesap maliyeti", "Python referansından", "| KF (doğrusal) | GCC", "1 000 B", "210 000 Hz",
                   "spt_alt_kf_step(&f, baro_p, acc_z", "p0 = 0 (otomatik)", "-ffast-math"):
        assert needle in text, needle
    plain = gc.usage_markdown(gc.GenericCodegenOptions("position2d", ("ukf",), {}, {}, ("gnss_ll", "gyro_z"),
                                                       prefix="nav"))
    assert "nav_pos_ukf_step" in plain and "ölçümsüz" in plain and "NAV_POS_FIX_HISTORY" in plain
    assert "spt_" not in plain
    gen = {f.name: f.text for f in gc.generate(o, fp, cost)}
    assert "| KF (doğrusal) | GCC" in gen["KULLANIM_alt.md"]
    assert alt_mod.ISA_H == 44330.77                                   # C sabitleri Python ile aynı kaynaktan


# ============================================================================ C kodu doğrulayıcısı bulguları
_DT_CASES = {  # senaryo: (eşlenen girişler, i. örnekteki argümanlar — dt hariç)
    "altitude": (("baro_p", "acc_z"), "101325.0 - 0.6 * i, 0.2"),
    "position2d": (("gnss", "gyro_z", "speed"), "(i % 20 == 0) ? gnss : NULL, 0.05, 2.0"),
    "linear_multi": (("s1", "s2"), "0.01 * i, 0.01 * i + 0.003"),
}


@needs_host
@pytest.mark.parametrize("scenario", list(_DT_CASES))
def test_step_skips_invalid_dt_without_touching_state(scenario):
    """C kodu doğrulayıcısı: dt ≤ 0 / NaN / sonsuz parçacık filtrelerini (NaN / sonsuzda tümünü) kalıcı bozuyordu;
    Python bu dt'yi FusionError ile reddeder. Artık step() 0 döner ve durum değişmez (sonraki çıktılar bit düzeyinde
    o örnek hiç gelmemiş gibi)."""
    inputs, args = _DT_CASES[scenario]
    opts = gc.GenericCodegenOptions(scenario, gc.SUPPORTED[scenario], {}, {}, inputs, real="double")
    files = {f.name: f.text for f in gc.generate(opts) if f.name.endswith((".c", ".h"))
             and not f.name.startswith("ornek")}
    r = gc._resolve(opts)
    k = len(fg.SCENARIOS[scenario].outputs)
    runs = []
    for a in r.algorithms:
        t = f"spt_{r.sab}_{a}"
        runs.append(f"""    for (b = 0; b < 4; ++b) {{
        static {t}_t s1, s2;
        int ret = -1, same = 1, any = 0;
        double o1[{k}], o2[{k}];
        (void){t}_init(&s1); (void){t}_init(&s2);
        for (i = 0; i < 400; ++i) {{
            (void){t}_step(&s1, {args}, 0.01);
            if (i == 200) ret = {t}_step(&s2, {args}, BAD[b]);
            (void){t}_step(&s2, {args}, 0.01);
        }}
        {t}_output(&s1, o1); {t}_output(&s2, o2);
        for (j = 0; j < {k}; ++j) {{  /* NaN = bu çıktı kestirilmiyor: iki çalıştırmada da NaN olmalı */
            same = same && (o1[j] == o2[j] || (isnan(o1[j]) && isnan(o2[j])));
            any = any || isfinite(o1[j]);
        }}
        printf("{a} %d %d %d %d\\n", b, ret, same, any);
    }}""")
    files["dt_main.c"] = "\n".join(f'#include "spt_{r.sab}_{a}.h"' for a in r.algorithms) + f"""
#include <math.h>
#include <stdio.h>
static const double BAD[4] = {{0.0, -0.01, NAN, INFINITY}};
int main(void)
{{
    const double gnss[2] = {{3.0, -2.0}};
    int b, i, j;
    (void)gnss;
{chr(10).join(runs)}
    return 0;
}}
"""
    build = tc.host_build(files, [f"spt_{r.sab}_{a}.c" for a in r.algorithms] + ["dt_main.c"],
                          extra=["-std=c99"], output=f"dt_{r.sab}")
    assert build.ok, build.log
    rows = [line.split() for line in tc.run_program(build, []).stdout.splitlines()]
    assert len(rows) == 4 * len(r.algorithms)
    for algo, kind, ret, same, any_finite in rows:
        assert ret == "0" and same == "1" and any_finite == "1", (scenario, algo, kind, ret, same, any_finite)
