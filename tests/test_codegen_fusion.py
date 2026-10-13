"""Füzyon C kod üreteci: Python ile birebirlik (double), float doğruluğu, parçacık filtresi istatistiği, UKF özdeğer
yedeği, çapraz derleme uyarısızlığı (Cortex-M × GCC / Clang × float / double), ayak izi ve maliyet.

Derleyici gerektiren testler, derleyici yoksa atlanır. Derleme çıktıları %TEMP%/spt_codegen altında içerik
özetiyle önbelleklenir (ilk çalıştırma ~2 dk, sonrakiler hızlı)."""
import functools
import shutil
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from spt.codegen import fusion_c as fc
from spt.codegen import toolchain as tc
from spt.codegen.mcu import COMPILERS, MCU_PROFILES, OPS, cycle_key
from spt.dsp import fusion as fu

SIM = fu.simulate_imu(duration=20.0, fs=200.0, seed=5)
DT = 1 / SIM.fs
SKIP = int(5 * SIM.fs)   # ilk 5 s: yakınsama
ALL = tuple(fu.ALGORITHMS)
DET = ("mahony", "madgwick", "kalman", "ekf", "ukf")
OPTS = fc.CodegenOptions(algorithms=ALL)

HOST_CC = tc.find_host_cc()
HOST_CXX = tc.find_host_cxx()
ARM_GCC = tc.find_arm_gcc()
CLANG = tc.find_clang()
needs_host = pytest.mark.skipif(HOST_CC is None, reason="ana makine C derleyicisi yok")
needs_cxx = pytest.mark.skipif(HOST_CXX is None, reason="ana makine C++ derleyicisi yok")
needs_cross = pytest.mark.skipif(ARM_GCC is None and CLANG is None, reason="ARM çapraz derleyici yok")


def _dropout_data():
    """Kayıp ve geçersiz örnekler: sonlu olmayan jiroskop (durum unutulur), sıfır / sonsuz / NaN ivme ve manyetometre."""
    g, a, m = SIM.gyro.copy(), SIM.acc.copy(), SIM.mag.copy()
    g[500:520] = np.nan
    a[518:530] = 0.0          # kayıptan sonra ivme geçersiz: başlatma ertelenir
    a[531, 1] = np.inf
    m[700:710] = np.nan
    m[711] = 0.0
    a[1500:1510] = np.nan
    g[3000, 2] = np.inf
    return g, a, m


DROP = _dropout_data()


@functools.lru_cache(maxsize=None)
def _python(key: str, mag: bool, data: str):
    g, a, m = (SIM.gyro, SIM.acc, SIM.mag) if data == "clean" else DROP
    est = fu.make_estimator(key)
    q = est.run(g, a, m if mag else None, DT)
    return q, est.bias.copy()


def _c(key: str, mag: bool, data: str, real: str) -> fc.HostRun:
    g, a, m = (SIM.gyro, SIM.acc, SIM.mag) if data == "clean" else DROP
    return fc.run_generated(OPTS, key, g, a, m if mag else None, DT, real=real)


def _angle(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """İki kuaterniyon arasındaki dönme açısı (rad)."""
    v = fu.quat_mul(fu.quat_conj(q1), q2)[:, 1:]
    return 2 * np.arcsin(np.clip(np.linalg.norm(v, axis=1), 0.0, 1.0))


def _rmse(q: np.ndarray, axes=(0, 1, 2)) -> list[float]:
    angles = fu.euler_from_quat(q)
    return [fu.rms(fu.angle_error_deg(angles[i][SKIP:], SIM.truth[SKIP:, i])) for i in axes]


# ============================================================================ üretim ve seçenekler
def test_generated_files_parameters_and_prefix():
    files = {f.name: f.text for f in fc.generate(OPTS)}
    expected = {"spt_fusion_common.h", "ornek_kullanim.c", "KULLANIM.md"} | \
        {f"spt_{a}.{ext}" for a in ALL for ext in ("h", "c")}
    assert set(files) == expected
    assert all("{{" not in t for t in files.values())
    assert "#define SPT_MAHONY_KP 1.0" in files["spt_mahony.h"]
    assert "#define SPT_PF_PARTICLES 600" in files["spt_pf.h"]
    assert "spt_kalman_euler_bias" in files["spt_kalman.h"] and "spt_kalman_gyro_bias" not in files["spt_kalman.h"]
    assert "spt_madgwick_gyro_bias" not in files["spt_madgwick.h"]
    # parametreler gömülür, önek her tanımlayıcıya uygulanır
    opts = fc.CodegenOptions(("mahony", "pf"), {"mahony": {"kp": 2.5}, "pf": {"particles": 200, "seed": 9}},
                             real="double", use_mag=False, prefix="imu")
    files = {f.name: f.text for f in fc.generate(opts)}
    assert set(files) == {"imu_fusion_common.h", "imu_mahony.h", "imu_mahony.c", "imu_pf.h", "imu_pf.c",
                          "ornek_kullanim.c", "KULLANIM.md"}
    assert "#define IMU_MAHONY_KP 2.5" in files["imu_mahony.h"]
    assert "#define IMU_PF_PARTICLES 200" in files["imu_pf.h"] and "#define IMU_PF_SEED 9u" in files["imu_pf.h"]
    assert "#define IMU_USE_DOUBLE 1" in files["imu_fusion_common.h"]
    assert "#define IMU_USE_MAG 0" in files["imu_fusion_common.h"]
    for name, text in files.items():
        if name.endswith((".c", ".h")):
            assert "spt_" not in text and "SPT_" not in text, name
    for bad in (dict(algorithms=()), dict(algorithms=("yok",)), dict(algorithms=("mahony",), real="half"),
                dict(algorithms=("mahony",), params={"mahony": {"kx": 1.0}}), dict(algorithms=("mahony",), prefix="1a"),
                dict(algorithms=("pf",), params={"pf": {"particles": 70000}}), dict(algorithms=("ekf",), mcu="z80"),
                # C kodu doğrulayıcısı: Python aralığı dışı değerler (σ = 0 → kalıcı NaN) ve saklı tanımlayıcı üreten önek
                dict(algorithms=("pf",), params={"pf": {"sigma_heading": 0.0}}),
                dict(algorithms=("ekf",), params={"ekf": {"sigma_acc": 0.0}}),
                dict(algorithms=("pf",), params={"pf": {"particles": 30000}}),
                dict(algorithms=("mahony",), params={"mahony": {"kp": -5.0}}),
                dict(algorithms=("mahony",), prefix="_x"), dict(algorithms=("mahony",), prefix="a__b"),
                dict(algorithms=("mahony",), prefix="nav_")):
        with pytest.raises(ValueError):
            fc.generate(fc.CodegenOptions(**bad))


def test_mcu_profiles_and_cycle_tables():
    assert set(MCU_PROFILES) == {"cortex-m0plus", "cortex-m3", "cortex-m4f", "cortex-m7", "cortex-m33f"}
    for p in MCU_PROFILES.values():
        for op in OPS:
            assert p.cycles[cycle_key(op, "float")] > 0 and p.cycles[cycle_key(op, "double")] > 0
        assert p.cycles["mem"] >= 1 and p.cycles["call"] >= 1
        if p.fpu == "sp":   # tek hassasiyet FPU'da double yazılım kayan noktadır
            assert p.cycles["dadd"] > 10 * p.cycles["fadd"] and p.cycles["ddiv"] > 5 * p.cycles["fdiv"]
        assert p.gcc_flags and p.clang_flags
        assert all(part.strip().startswith("STM32") for part in p.examples.split(","))   # yalnız STM32 aileleri
    assert MCU_PROFILES["cortex-m7"].cycles["dadd"] <= 2
    assert MCU_PROFILES["cortex-m0plus"].cycles["fmul"] > MCU_PROFILES["cortex-m3"].cycles["fmul"] > 10
    assert set(COMPILERS) == {"gcc", "clang", "iar"}


# ============================================================================ Python ile birebirlik
@needs_host
@pytest.mark.parametrize("mag", [True, False], ids=["marg", "imu"])
@pytest.mark.parametrize("key", DET)
def test_double_matches_python_exactly(key, mag):
    """Çift hassasiyette C = Python (kayıp / geçersiz örnek akışı dahil): en büyük açı farkı < 1e-6 rad."""
    q_py, bias_py = _python(key, mag, "drop")
    run = _c(key, mag, "drop", "double")
    nan_py, nan_c = np.isnan(q_py).any(axis=1), np.isnan(run.q).any(axis=1)
    assert np.array_equal(nan_py, nan_c) and nan_py.sum() > 20     # aynı örneklerde çıktı yok
    ok = ~nan_py
    assert np.max(_angle(q_py[ok], run.q[ok])) < 1e-6
    assert np.max(np.abs(bias_py - run.bias)) < 1e-9
    assert np.allclose(np.linalg.norm(run.q[ok], axis=1), 1.0, atol=1e-12)


@needs_host
@pytest.mark.parametrize("mag", [True, False], ids=["marg", "imu"])
def test_particle_filter_statistically_equivalent(mag):
    """Farklı rastgele sayı üreteci: RMSE ≤ Python RMSE × 1.5 + 0.5° (IMU'da yaw gözlemlenemez → yalnız roll / pitch)."""
    axes = (0, 1, 2) if mag else (0, 1)
    q_py, _ = _python("pf", mag, "clean")
    run = _c("pf", mag, "clean", "double")
    assert np.all(np.isfinite(run.q))
    for e_c, e_py in zip(_rmse(run.q, axes), _rmse(q_py, axes)):
        assert e_c <= e_py * 1.5 + 0.5, (e_c, e_py)
    # kayıp akışı: aynı örneklerde çıktı yok, geri kalan sonlu
    q_py, _ = _python("pf", mag, "drop")
    run = _c("pf", mag, "drop", "double")
    assert np.array_equal(np.isnan(q_py).any(axis=1), np.isnan(run.q).any(axis=1))
    assert np.all(np.isfinite(run.q[~np.isnan(q_py).any(axis=1)]))


@needs_host
@pytest.mark.parametrize("key", DET)
def test_float_accuracy_matches_python(key):
    """float: eksen başına RMSE ≤ Python (double) RMSE + 0.1°."""
    q_py, _ = _python(key, True, "clean")
    run = _c(key, True, "clean", "float")
    assert np.all(np.isfinite(run.q))
    for e_c, e_py in zip(_rmse(run.q), _rmse(q_py)):
        assert e_c <= e_py + 0.1, (e_c, e_py)


@needs_host
def test_ukf_eigen_fallback_matches_python():
    """Cholesky başarısız (belirsiz P) → Jacobi özdeğer yedeği; Python eigh yedeğiyle aynı adım."""
    rng = np.random.default_rng(3)
    a = rng.normal(size=(6, 6))
    vecs = np.linalg.qr(a)[0]
    vals = np.array([4e-4, 2e-4, 1e-4, -5e-5, 3e-6, 2e-6])       # ayrık özdeğerler, biri negatif
    p_bad = vecs @ np.diag(vals) @ vecs.T
    p_bad = 0.5 * (p_bad + p_bad.T)
    g, acc, mag = SIM.gyro[0], SIM.acc[0], SIM.mag[0]
    est = fu.UKF()
    est.reset(fu.initial_quaternion(acc, mag))
    est.P = p_bad.copy()
    q_py = est._run(g[None], acc[None], mag[None], DT)[0]
    with pytest.raises(np.linalg.LinAlgError):
        np.linalg.cholesky(6 * p_bad)

    def arr(v):
        return ", ".join(float(x).hex() for x in np.ravel(v))

    main = f"""#include <stdio.h>
#include "spt_ukf.h"
int main(void)
{{
    static spt_ukf_t s;
    const double g[3] = {{{arr(g)}}}, a[3] = {{{arr(acc)}}}, m[3] = {{{arr(mag)}}}, p[36] = {{{arr(p_bad)}}};
    int i;
    spt_ukf_init(&s, a, m);
    for (i = 0; i < 36; ++i) s.P[i / 6][i % 6] = p[i];
    spt_ukf_update(&s, g, a, m, {DT!r});
    printf("%a %a %a %a %u\\n", s.q[0], s.q[1], s.q[2], s.q[3], (unsigned)s.fallbacks);
    return 0;
}}
"""
    files = {f.name: f.text for f in fc.generate(fc.CodegenOptions(("ukf",), real="double"))}
    files["fallback_main.c"] = main
    build = tc.host_build(files, ["spt_ukf.c", "fallback_main.c"], extra=["-std=c99"], output="ukf_fallback")
    assert build.ok, build.log
    out = tc.run_program(build, []).stdout.split()
    q_c = np.array([float.fromhex(t) for t in out[:4]])
    assert int(out[4]) >= 1
    assert _angle(q_py[None], q_c[None])[0] < 1e-9


# ============================================================================ çapraz derleme
def _compile_job(files, sources, prof, compiler, defines, extra):
    b = tc.compile_cross(files, sources, prof, compiler, "-Os", defines=defines, extra=extra, stack_usage=False)
    return f"{prof.key}/{compiler}/{' '.join(defines)}", b


@needs_cross
def test_cross_compile_warning_free_all_cortex_m():
    """Her yerel Cortex-M profili × GCC / Clang × float / double: -std=c99 -Wall -Wextra -pedantic -Werror temiz
    (float'ta ayrıca -Wdouble-promotion: istemsiz double işlemi yok). Örnek dosya ve manyetometresiz / önekli üretim
    de derlenir."""
    files = {f.name: f.text for f in fc.generate(OPTS)}
    sources = [n for n in files if n.endswith(".c")]
    jobs = []
    for prof in MCU_PROFILES.values():
        for comp in ("gcc", "clang"):
            if tc.cross_flags(prof, comp) is None:
                continue
            for real in ("0", "1"):
                extra = ["-Wdouble-promotion"] if real == "0" else []
                jobs.append((files, sources, prof, comp, [f"SPT_USE_DOUBLE={real}"], extra))
    m4f = MCU_PROFILES["cortex-m4f"]
    nomag = {f.name: f.text for f in fc.generate(fc.CodegenOptions(ALL, use_mag=False, prefix="imu"))}
    for comp in ("gcc", "clang"):
        if tc.cross_flags(m4f, comp) is not None:
            jobs.append((nomag, [n for n in nomag if n.endswith(".c")], m4f, comp, ["IMU_USE_DOUBLE=0"],
                         ["-Wdouble-promotion"]))
    if not jobs:
        pytest.skip("çapraz derleyici yok")
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda j: _compile_job(*j), jobs))
    failures = [f"{name}:\n{b.log}" for name, b in results if not b.ok or b.log.strip()]
    assert not failures, "\n\n".join(failures)
    assert len(results) >= 4


@needs_cxx
@pytest.mark.parametrize("real", ["0", "1"])
def test_generated_code_is_valid_cpp(real):
    """C++ projelerinden (STM32CubeIDE C++ …) de derlenebilir: -x c++ -Wall -Wextra -pedantic -Werror temiz."""
    folder = tc.WORK_ROOT / "cxx_check" / real
    shutil.rmtree(folder, ignore_errors=True)
    fc.write_files(fc.generate(OPTS), folder)
    sources = sorted(p.name for p in folder.glob("*.c"))
    res = tc._run([HOST_CXX.path, "-x", "c++", "-std=c++11", *tc.WARN_FLAGS, f"-DSPT_USE_DOUBLE={real}",
                   "-fsyntax-only", *sources], cwd=folder, path_prefix=[HOST_CXX.bin_dir])
    assert res.returncode == 0 and not res.stderr.strip(), res.stderr


# ============================================================================ ayak izi
@needs_cross
def test_footprint_measured_on_cortex_m4f():
    opts = fc.CodegenOptions(ALL, real="float", mcu="cortex-m4f", opt="-Os")
    rows = fc.footprint(opts, ("gcc", "clang", "iar"))
    by = {(r.algorithm, r.compiler): r for r in rows}
    n = 600
    for comp, tool in (("gcc", ARM_GCC), ("clang", CLANG)):
        if tool is None:
            continue
        for a in ALL:
            r = by[(a, comp)]
            assert r.measured and "ölçüldü" in r.method
            assert r.ram_state == fc.state_size(a, "float", n)            # sizeof = yapı düzeni formülü
            assert 300 < r.flash_code < 20_000
            if ARM_GCC is not None:
                assert r.flash_with_libm > r.flash_code
                assert r.stack >= r.stack_own > 0
        assert by[("mahony", comp)].flash_code < by[("ekf", comp)].flash_code < by[("ukf", comp)].flash_code
        assert by[("madgwick", comp)].ram_state < by[("mahony", comp)].ram_state < by[("ekf", comp)].ram_state
        pf = by[("pf", comp)].ram_state
        assert n * 7 * 4 < pf < n * 9.5 * 4 + 128                        # q + b + logw (+ uint16 sayaç)
    if ARM_GCC is not None:
        for a in ALL:
            iar, gcc = by[(a, "iar")], by[(a, "gcc")]
            assert not iar.measured and "tahmin" in iar.method
            assert iar.flash_code == round(gcc.flash_code * 0.9) and iar.ram_state == gcc.ram_state
        # objdump tabanlı çerçeve analizi = GCC -fstack-usage (update fonksiyonunun kendi çerçevesi)
        raw = fc._measure(opts, MCU_PROFILES["cortex-m4f"], "gcc")
        for d in raw:
            assert d["frame_update"] == d["su_update"], d


@pytest.mark.parametrize("key", ["esp32", "rv32imac", "rv32imfc"])
def test_removed_non_stm32_profiles_are_rejected(key):
    """Yalnız STM32 (Cortex-M) desteklenir: eski ESP32 / RISC-V profilleri açık bir hatayla reddedilir."""
    opts = fc.CodegenOptions(("mahony",), mcu=key)
    with pytest.raises(ValueError, match="Bilinmeyen MCU profili"):
        fc.generate(opts)
    with pytest.raises(ValueError, match="Bilinmeyen MCU profili"):
        fc.footprint(opts, ("gcc",))


def test_footprint_degrades_without_compilers(monkeypatch):
    monkeypatch.setattr(tc, "find_arm_gcc", lambda: None)
    monkeypatch.setattr(tc, "find_clang", lambda: None)
    rows = fc.footprint(fc.CodegenOptions(ALL, real="double", mcu="cortex-m3"), ("gcc", "clang", "iar"))
    assert len(rows) == 3 * len(ALL)
    for r in rows:
        assert not r.measured and "tahmin" in r.method and r.flash_code > 0
        assert r.ram_state == fc.state_size(r.algorithm, "double", 600)


# ============================================================================ maliyet
@needs_cxx
def test_cost_ops_deterministic_ordering_and_load_formula():
    opts = fc.CodegenOptions(ALL, mcu="cortex-m4f", clock_mhz=168.0, fs_hz=200.0)
    rows = fc.cost(opts)
    assert len(rows) == 2 * len(ALL)
    by = {(r.algorithm, r.mag): r for r in rows}
    for r in rows:
        assert r.us_per_update == pytest.approx(r.cycles / opts.clock_mhz)
        assert r.cpu_load_percent == pytest.approx(r.us_per_update * opts.fs_hz / 1e4)
        assert r.max_rate_hz == pytest.approx(1e6 / r.us_per_update)
        assert r.cycles_max >= r.cycles > 0 and "ölçüldü" in r.method and "tahmin" in r.method
        assert by[(r.algorithm, True)].cycles >= by[(r.algorithm, False)].cycles
    for mag in (True, False):
        c = {a: by[(a, mag)].cycles for a in ALL}
        assert c["mahony"] < c["madgwick"] < c["ekf"] < c["ukf"] < c["pf"]
        assert c["kalman"] < c["ekf"]
    # Mahony MARG adımı: ivme / manyetometre normu, bx, kuaterniyon normu = 4 karekök, aşkın fonksiyon yok
    mahony = by[("mahony", True)].ops
    assert mahony["sqrt"] == 4 and mahony["atan2"] == 0 and mahony["sin"] == 0
    assert by[("pf", True)].ops["rng"] > 6 * 600                         # parçacık başına ≥ 6 normal sayı (çift)
    # işlem sayıları belirleyici: önbellek silinip yeniden çalıştırılınca aynı
    build = fc.host_program(opts, count_ops=True)
    sim = fu.simulate_imu(duration=1.0, fs=200.0, seed=2)
    first = fc.run_generated(opts, "pf", sim.gyro, sim.acc, sim.mag, DT, count_ops=True).ops
    shutil.rmtree(build.workdir / "runs")
    second = fc.run_generated(opts, "pf", sim.gyro, sim.acc, sim.mag, DT, count_ops=True).ops
    assert np.array_equal(first, second)


@needs_cxx
def test_cost_soft_double_is_much_slower_on_single_precision_core():
    f = {r.algorithm: r for r in fc.cost(fc.CodegenOptions(("ekf",), use_mag=False))}
    d = {r.algorithm: r for r in fc.cost(fc.CodegenOptions(("ekf",), real="double", use_mag=False))}
    assert d["ekf"].ops == f["ekf"].ops                                   # aynı kod yolu
    assert d["ekf"].cycles > 10 * f["ekf"].cycles


def test_usage_markdown_contents():
    fp = [fc.FootprintRow("mahony", "gcc", 1000, 5000, 40, 500, True, "ölçüm yöntemi", 70)]
    cost = [fc.CostRow("mahony", {op: 1.0 for op in OPS}, 800.0, 4.76, 0.095, 210000.0, "yöntem", True, 900.0, 5.4)]
    text = fc.usage_markdown(fc.CodegenOptions(("mahony", "kalman")), fp, cost)
    for needle in ("## Dosyalar", "STM32CubeIDE", "Keil MDK", "IAR EWARM", "DWT->CYCCNT", "## Sensör eksen hizalaması",
                   "## Kalibrasyon", "## İş parçacığı güvenliği", "## Ölçülen ayak izi", "## Hesap maliyeti",
                   "spt_kalman_euler_bias", "-mfpu=fpv4-sp-d16", "| Mahony | GCC", "1 000 B", "Python referansından",
                   "210 000 Hz"):
        assert needle in text, needle
    plain = fc.usage_markdown(fc.CodegenOptions(("pf",), prefix="imu"))
    assert "imu_pf_step" in plain and "ölçümsüz" in plain


# ============================================================================ C kodu doğrulayıcısı bulguları
_BAD_DT_MAIN = r"""#include <math.h>
#include <stdio.h>
{includes}
static const double BAD[4] = {{0.0, -0.005, NAN, INFINITY}};
#define RUN(A) do {{ \
    int b, i, ret = -1, fin = 1; double q1[4], q2[4], d = 0.0; \
    const double g[3] = {{0.1, -0.05, 0.2}}, a[3] = {{0.5, -0.3, 9.7}}, m[3] = {{0.2, 0.0, -0.4}}; \
    for (b = 0; b < 4; ++b) {{ \
        static spt_##A##_t s1, s2; \
        spt_##A##_init(&s1, a, m); spt_##A##_init(&s2, a, m); \
        for (i = 0; i < 300; ++i) {{ \
            (void)spt_##A##_step(&s1, g, a, m, 0.005); \
            if (i == 150) ret = spt_##A##_step(&s2, g, a, m, BAD[b]); \
            (void)spt_##A##_step(&s2, g, a, m, 0.005); \
        }} \
        spt_##A##_quat(&s1, q1); spt_##A##_quat(&s2, q2); \
        for (i = 0; i < 4; ++i) {{ d = fmax(d, fabs(q1[i] - q2[i])); fin = fin && isfinite(q2[i]); }} \
        printf("%s %d %d %.17g %d\n", #A, b, ret, d, fin); \
    }} \
}} while (0)
int main(void)
{{
{runs}
    return 0;
}}
"""


@needs_host
def test_step_skips_invalid_dt_without_touching_state():
    """C kodu doğrulayıcısı: tek bir dt = 0 / < 0 / NaN / sonsuz örneği parçacık filtresinin (ve NaN / sonsuzda tüm
    algoritmaların) durumunu kalıcı NaN yapıyor, step() yine 1 döndürüyordu. Artık örnek atlanır (0 döner) ve durum
    değişmez: sonraki çıktı, o örnek hiç gelmemiş gibi bit düzeyinde aynıdır (PF'nin rastgele dizisi dahil)."""
    files = {f.name: f.text for f in fc.generate(fc.CodegenOptions(ALL, real="double"))}
    files["bad_dt_main.c"] = _BAD_DT_MAIN.format(includes="\n".join(f'#include "spt_{a}.h"' for a in ALL),
                                                  runs="\n".join(f"    RUN({a});" for a in ALL))
    build = tc.host_build(files, [f"spt_{a}.c" for a in ALL] + ["bad_dt_main.c"], extra=["-std=c99"],
                          output="bad_dt")
    assert build.ok, build.log
    rows = [line.split() for line in tc.run_program(build, []).stdout.splitlines()]
    assert len(rows) == 4 * len(ALL)
    for algo, kind, ret, diff, finite in rows:
        assert ret == "0" and float(diff) == 0.0 and finite == "1", (algo, kind, ret, diff, finite)


@needs_host
def test_orientation_and_generic_headers_in_any_order():
    """Yönelim ve genel füzyon başlıkları aynı derleme biriminde, hangi sırayla eklenirse eklensin derlenir
    (önceden genel başlık önce gelirse spt_finite yeniden tanımlanıyordu); dosya adları çakışmaz."""
    from spt.codegen import fusion_generic_c as gc
    ori = {f.name: f.text for f in fc.generate(fc.CodegenOptions(("mahony",), real="float"))}
    gen = {f.name: f.text for f in gc.generate(gc.GenericCodegenOptions("altitude", ("kf",), {}, {},
                                                                         ("baro_p", "acc_z"), real="float"))}
    assert not set(ori) & set(gen)  # aynı klasöre kaydedilebilir
    files = {n: t for n, t in {**ori, **gen}.items() if n.endswith((".h", ".c")) and not n.startswith("ornek")}
    for order in (("spt_alt_kf.h", "spt_mahony.h"), ("spt_mahony.h", "spt_alt_kf.h")):
        files["both_main.c"] = "".join(f'#include "{h}"\n' for h in order) + r"""
#include <stdio.h>
int main(void)
{
    static spt_mahony_t m;
    static spt_alt_kf_t k;
    const float a[3] = {0.0f, 0.0f, 9.81f};
    (void)spt_mahony_init(&m, a, NULL);
    (void)spt_alt_kf_init(&k);
    printf("%d\n", spt_alt_kf_step(&k, 101325.0f, 0.0f, 0.01f) + spt_mahony_step(&m, a, a, NULL, 0.01f));
    return 0;
}
"""
        build = tc.host_build(files, ["spt_mahony.c", "spt_alt_kf.c", "both_main.c"],
                              extra=["-std=c99", "-Wall", "-Wextra", "-Werror"], output="both_" + order[0][4:7])
        assert build.ok, (order, build.log)


@needs_cross
@pytest.mark.parametrize("compiler,opt,real", [("gcc", "-O0", "double"), ("gcc", "-O3", "double"),
                                               ("clang", "-O0", "double"), ("clang", "-Os", "float")])
def test_stack_analysis_exact_on_cortex_m0plus(compiler, opt, real):
    """C kodu doğrulayıcısı: ARMv6-M 508 bayttan büyük çerçeveyi `ldr rN, [pc, #..]` + `add sp, rN` (sabit havuzunda
    negatif sabit) ya da `movs; lsls; add sp, rN` ile ayırır; çözümleyici bunu atlıyor ve "kesin" diyordu (EKF update
    gcc -O0: 984 B yerine 0; clang -O0: 1832 B yerine 20). Ayrıca Thumb-1'de fonksiyon içi uzak dallanma `bl <kendi+ofs>`
    özyineleme sanılıyordu. Artık her çerçeve -fstack-usage ile aynı ve yol kesin."""
    if tc.cross_tool(compiler) is None:
        pytest.skip(f"{compiler} yok")
    opts = fc.CodegenOptions(ALL, mcu="cortex-m0plus", opt=opt, real=real)
    rows = fc._measure(opts, MCU_PROFILES["cortex-m0plus"], compiler)
    assert rows
    for d in rows:
        assert d["frame_update"] == d["su_update"] > 0, d["algorithm"]
        assert d["stack_exact"] and d["stack"] >= d["stack_own"] >= d["frame_update"], d["algorithm"]
    if real == "double":  # büyük çerçeve (> 508 B) yolu gerçekten sınandı
        assert next(d for d in rows if d["algorithm"] == "ekf")["frame_update"] > 508


def test_stack_frame_parser_understands_thumb1_idioms():
    """Çözümleyicinin birim testi (derleyici gerekmez): gcc / llvm-objdump biçimleri."""
    lits = {0x120: 0xFFFFC16C, 0x124: 0x00003E8C}
    gcc = [("push", "{r4, r7, lr}"), ("ldr", "r4, [pc, #132]\t@ (120 <huge+0x88>)"), ("add", "sp, r4"),
           ("ldr", "r3, [pc, #132]\t@ (124 <huge+0x8c>)"), ("add", "sp, r3"), ("pop", "{r4, r7, pc}")]
    assert tc._frame(gcc, lits) == (12 + 0x3E94, False)
    llvm = [("push", "{r4, r6, r7, lr}"), ("ldr", "r6, [pc, #0x64]         @ 0xd0 <huge+0x6c>"),
            ("add", "sp, r6")]
    assert tc._frame(llvm, {0xD0: 0xFFFFC168}) == (16 + 0x3E98, False)
    shifted = [("push", "{r4, lr}"), ("movs", "r3, #167"), ("lsls", "r3, r3, #2"), ("negs", "r3, r3"),
               ("add", "sp, r3")]
    assert tc._frame(shifted) == (8 + 668, False)
    # üzerine bilinmeyen değer yazılan yazmaç unutulur: bayat sabitle yanlış "kesin" sonuç verilmez
    stale = [("movs", "r3, #16"), ("adds", "r3, r3, r2"), ("sub", "sp, r3")]
    assert tc._frame(stale) == (0, True)
