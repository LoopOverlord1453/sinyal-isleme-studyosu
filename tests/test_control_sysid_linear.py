"""Parametrik doğrusal polinom modeller ve değerlendirme fonksiyonları — MATLAB R2025b referanslı.

Ölçülen MATLAB kuralları (bkz. spt/control/sysid/linear.py ve models.py belgeleri):
- arx ilk nx satırı atar (nx = max(na, nb + min(nk,1) − 1)), LossFcn = Σe²/N_toplam, d = parametre + nx;
- compare / predict / pe / resid varsayılan başlangıç koşulu 'e' (öngörücü hatasında en küçük kareler);
- PEM ölçütü 'zero' / 'estimate' başlangıç koşullarıyla birebir; MATLAB sıkı toleransta bile optimumdan önce
  durabildiği için ('zero' durumunda) ölçüt tanımı MATLAB parametrelerinde kaybı yeniden üreterek doğrulanır ve
  bizim optimumun kaybının MATLAB'inkinden büyük olmadığı test edilir.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy import signal
from test_control_sysid_common import dataset, oracle

from spt.control.errors import ControlError
from spt.control.sysid import (IDData, ar, armax, arx, bj, compare, merge, oe, pe, polyest, predict, resid, sim)
from spt.control.sysid.linear import _PEM, _exps, _PolyStructure

SCRIPT = r"""
f = @(m) struct('A', {m.A}, 'B', {m.B}, 'C', m.C, 'D', m.D, 'F', {m.F}, 'nv', m.NoiseVariance, 'fit', m.Report.Fit, 'cov', getcov(m), 'ic', m.Report.InitialCondition);
D = readmatrix('{DATA}/arx1.csv'); z = iddata(D(:,1), D(:,2), 0.1);
out.arx221 = f(arx(z, [2 2 1]));
out.arx232 = f(arx(z, [2 3 2]));
out.arx310 = f(arx(z, [3 1 0]));
out.arx123 = f(arx(z, [1 2 3]));
out.arx_merge = f(arx(merge(z(1:300), z(301:600)), [2 2 1]));
o0 = armaxOptions('InitialCondition','zero'); o0.SearchOptions.Tolerance = 1e-12; o0.SearchOptions.MaxIterations = 300;
oe_ = armaxOptions('InitialCondition','estimate'); oe_.SearchOptions.Tolerance = 1e-12; oe_.SearchOptions.MaxIterations = 300;
out.armax_z = f(armax(z, [2 2 2 1], o0));
mae = armax(z, [2 2 2 1], oe_); out.armax_e = f(mae);
out.armax_def = f(armax(z, [2 2 2 1]));
out.armax_def_tail = f(armax(z(201:600), [2 2 2 1]));
[yc, fit, x0] = compare(z, mae); out.cmp_fit = fit; out.cmp_y = yc.y;
[yc, fit] = compare(z, mae, 2); out.cmp2_fit = fit; out.cmp2_y = yc.y;
yp = predict(mae, z, 3); out.pred3 = yp.y;
e = pe(z, mae); out.pe = e.y;
[E, R] = resid(z, mae); out.resid_R = R;
ys = sim(mae, z); out.sim = ys.y;
D = readmatrix('{DATA}/oe1.csv'); z = iddata(D(:,1), D(:,2), 1);
out.oe_z = f(oe(z, [2 2 2], o0));
out.oe_e = f(oe(z, [2 2 2], oe_));
out.oe_def = f(oe(z, [2 2 2]));
D = readmatrix('{DATA}/bj1.csv'); z = iddata(D(:,1), D(:,2), 1);
out.bj_z = f(bj(z, [2 1 1 2 1], o0));
out.bj_e = f(bj(z, [2 1 1 2 1], oe_));
out.poly_z = f(polyest(z, [1 2 1 0 1 1], o0));
D = readmatrix('{DATA}/miso1.csv'); z = iddata(D(:,1), D(:,2:3), 1);
out.miso_arx = f(arx(z, [1 2 1 1 2]));
out.miso_armax = f(armax(z, [1 2 1 1 1 2], oe_));
out.miso_oe = f(oe(z, [2 1 1 1 1 2], oe_));
D = readmatrix('{DATA}/ts1.csv'); z = iddata(D(:,1), [], 1);
out.ts_arx = f(arx(z, 2));
g = @(m) struct('A', m.A, 'nv', m.NoiseVariance, 'fit', m.Report.Fit);
out.ts_ar_fb = g(ar(z, 2));
out.ts_ar_ls = g(ar(z, 2, 'ls'));
out.ts_ar_yw = g(ar(z, 2, 'yw'));
out.ts_ar_burg = g(ar(z, 2, 'burg'));
out.ts_armax_e = f(armax(z, [2 1], oe_));
mts = armax(z, [2 1], oe_); yp = predict(mts, z, 2); out.ts_pred2 = yp.y;
D = readmatrix('{DATA}/mimo1.csv'); z = iddata(D(:,1:2), D(:,3:4), 1);
mm = arx(z, [2*ones(2) 2*ones(2) ones(2)]);
out.mimo_A = mm.A; out.mimo_B = mm.B; out.mimo_nv = mm.NoiseVariance; out.mimo_fit = mm.Report.Fit; out.mimo_cov = getcov(mm);
[yc, fit] = compare(z, mm); out.mimo_cmp_fit = fit; out.mimo_cmp_y = yc.y;
yp = predict(mm, z, 1); out.mimo_pred1 = yp.y;
"""


@pytest.fixture(scope="module")
def ml():
    return oracle("sysid_linear", SCRIPT, ("arx1", "oe1", "bj1", "miso1", "ts1", "mimo1"))


def _z(name, Ts, ny=1):
    D = dataset(name)
    return IDData(D[:, :ny], D[:, ny:] if D.shape[1] > ny else None, Ts)


def rel(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-300))


def _cells(x, n=1):
    """MATLAB hücre dizisi (liste) ya da tek vektör → n elemanlı 1-B dizi listesi (skalerlerden oluşan hücre JSON'da
    düz listedir: {1, 1} → [1, 1])."""
    if isinstance(x, list) and x and isinstance(x[0], list):
        return [np.atleast_1d(np.asarray(v, float)) for v in x]
    if n > 1 and isinstance(x, list) and len(x) == n:
        return [np.atleast_1d(np.asarray(v, float)) for v in x]
    return [np.atleast_1d(np.asarray(x, float))]


def _check_fit(m, ref, rtol):
    for k in ("LossFcn", "MSE", "FPE", "AIC", "AICc", "nAIC", "BIC"):
        assert rel(getattr(m.report.fit, k), ref["fit"][k]) < rtol, k
    assert rel(np.ravel(m.report.fit.FitPercent), np.ravel(ref["fit"]["FitPercent"])) < rtol


# ============================================================================================ ARX
@pytest.mark.parametrize("key,orders", [("arx221", [2, 2, 1]), ("arx232", [2, 3, 2]), ("arx310", [3, 1, 0]),
                                        ("arx123", [1, 2, 3])])
def test_arx_matches_matlab(ml, key, orders):
    m = arx(_z("arx1", 0.1), orders)
    ref = ml[key]
    assert rel(m.A, ref["A"]) < 1e-10
    assert rel(m.B, ref["B"]) < 1e-10
    assert m.NoiseVariance == pytest.approx(ref["nv"], rel=1e-10)
    _check_fit(m, ref, 1e-10)
    assert rel(m.getcov(), ref["cov"]) < 1e-8


def test_arx_multi_experiment(ml):
    z = _z("arx1", 0.1)
    m = arx(merge(z[:300], z[300:]), [2, 2, 1])
    ref = ml["arx_merge"]
    assert rel(m.A, ref["A"]) < 1e-10 and rel(m.B, ref["B"]) < 1e-10
    assert m.NoiseVariance == pytest.approx(ref["nv"], rel=1e-10)
    _check_fit(m, ref, 1e-10)


def test_arx_miso(ml):
    m = arx(_z("miso1", 1.0), [1, 2, 1, 1, 2])
    ref = ml["miso_arx"]
    assert rel(m.A, ref["A"]) < 1e-10
    for b, rb in zip(m.B, _cells(ref["B"])):
        assert rel(b, rb) < 1e-10
    assert m.NoiseVariance == pytest.approx(ref["nv"], rel=1e-10)
    _check_fit(m, ref, 1e-10)
    assert rel(m.getcov(), ref["cov"]) < 1e-8


def test_arx_mimo(ml):
    z = _z("mimo1", 1.0, ny=2)
    m = arx(z, ([[2, 2], [2, 2]], [[2, 2], [2, 2]], [[1, 1], [1, 1]]))
    for i in range(2):
        for j in range(2):
            # MATLAB hücre dizisi JSON'da sütun sıralı düz listedir
            assert rel(m.A[i][j], ml["mimo_A"][i + 2 * j]) < 1e-10
            assert rel(m.B[i][j], ml["mimo_B"][i + 2 * j]) < 1e-10
    assert rel(m.noise_variance, ml["mimo_nv"]) < 1e-10
    assert rel(m.report.fit.FitPercent, ml["mimo_fit"]["FitPercent"]) < 1e-10
    assert m.report.fit.LossFcn == pytest.approx(ml["mimo_fit"]["LossFcn"], rel=1e-10)
    assert m.report.fit.FPE == pytest.approx(ml["mimo_fit"]["FPE"], rel=1e-10)
    assert rel(m.getcov(), ml["mimo_cov"]) < 1e-8
    r = compare(z, m)
    assert rel(r.fit[0], ml["mimo_cmp_fit"]) < 1e-9
    assert rel(r.yhat[0].y, ml["mimo_cmp_y"]) < 1e-9
    assert rel(predict(m, z, 1).y, ml["mimo_pred1"]) < 1e-9


def test_arx_time_series_and_ar(ml):
    z = _z("ts1", 1.0)
    m = arx(z, [2])
    assert rel(m.A, ml["ts_arx"]["A"]) < 1e-10
    assert m.NoiseVariance == pytest.approx(ml["ts_arx"]["nv"], rel=1e-10)
    for app, key in (("fb", "ts_ar_fb"), ("ls", "ts_ar_ls"), ("yw", "ts_ar_yw"), ("burg", "ts_ar_burg")):
        assert rel(ar(z, 2, app).A, ml[key]["A"]) < 1e-9, app


# ============================================================================================ PEM
def _criterion(st, data, ic, th):
    p = _PEM(st, _exps(data), ic)
    if ic == "estimate":
        s = p.best_s(th)
        r = p.resid(np.r_[th, s])
    else:
        r = p.resid(th)
    return float(r @ r) / len(r)


def _pack_ref(st, ref):
    B = _cells(ref["B"], st.nu) if st.nu else []
    F = _cells(ref["F"], st.nu) if st.nu else []
    return st.pack(np.atleast_1d(ref["A"]), B, np.atleast_1d(ref["C"]), np.atleast_1d(ref["D"]), F)


PEM_CASES = [
    ("armax", "arx1", 0.1, [2, 2, 2, 1], (2, [2], 2, 0, [0], [1])),
    ("oe", "oe1", 1.0, [2, 2, 2], (0, [2], 0, 0, [2], [2])),
    ("bj", "bj1", 1.0, [2, 1, 1, 2, 1], (0, [2], 1, 1, [2], [1])),
]
# Kovaryans toleransı: gürültü modelli yapılarda birebir; OE'de MATLAB'in renkli artık düzeltmesi (n4sid AR(5)
# gürültü modeliyle süzülmüş gradyan) uygulanmaz → ~%1–2.
COV_TOL = {"armax": 1e-6, "oe": 3e-2, "bj": 1e-6}
_EST = {"armax": armax, "oe": oe, "bj": bj}


@pytest.mark.parametrize("name,ds,Ts,orders,struct", PEM_CASES)
def test_pem_estimate_ic_matches_matlab(ml, name, ds, Ts, orders, struct):
    """'estimate' başlangıç koşulu, sıkı MATLAB toleransı: aynı optimum (~1e-6), aynı ölçütler ve kovaryans.

    OE [2 2 2]'de (nk = 2) MATLAB'in ölçülen kuralı: kestirimde gecikme durumları da serbest başlangıç durumudur
    (LossFcn bununla), diğer ölçütler gecikmesi ayrılmış gerçeklemede yeniden kestirilen x0 ile."""
    z = _z(ds, Ts)
    m = _EST[name](z, orders, initial_condition="estimate")
    ref = ml[f"{name}_e"]
    assert rel(m.getpvec(), _pack_ref(_PolyStructure(*struct), ref)) < 2e-6
    assert m.NoiseVariance == pytest.approx(ref["nv"], rel=1e-7)
    _check_fit(m, ref, 1e-7)
    assert rel(m.getcov(), ref["cov"]) < COV_TOL[name]


@pytest.mark.parametrize("name,ds,Ts,orders,struct", PEM_CASES)
def test_pem_zero_ic_same_criterion_and_not_worse(ml, name, ds, Ts, orders, struct):
    """'zero' başlangıç: ölçüt MATLAB parametrelerinde MATLAB kaybını 1e-12 ile verir; bizim optimum en az onun kadar
    iyi ve parametreler yakın (MATLAB'in erken durması nedeniyle 1e-3)."""
    z = _z(ds, Ts)
    st = _PolyStructure(*struct)
    ref = ml[f"{name}_z"]
    thM = _pack_ref(st, ref)
    assert _criterion(st, z, "zero", thM) == pytest.approx(ref["fit"]["LossFcn"], rel=1e-11)
    m = _EST[name](z, orders, initial_condition="zero")
    assert m.report.fit.LossFcn <= ref["fit"]["LossFcn"] * (1 + 1e-12)
    assert rel(m.getpvec(), thM) < 2e-3


def test_polyest_general_structure(ml):
    z = _z("bj1", 1.0)
    st = _PolyStructure(1, [2], 1, 0, [1], [1])
    ref = ml["poly_z"]
    thM = _pack_ref(st, ref)
    assert _criterion(st, z, "zero", thM) == pytest.approx(ref["fit"]["LossFcn"], rel=1e-11)
    m = polyest(z, 1, [2], 1, 0, [1], [1], initial_condition="zero")
    assert m.report.fit.LossFcn <= ref["fit"]["LossFcn"] * (1 + 1e-12)
    assert rel(m.getpvec(), thM) < 2e-3


def test_pem_default_options(ml):
    """Varsayılan seçenekler: 'auto' başlangıç koşulu seçimi MATLAB'le aynı; MATLAB varsayılan durdurmada (Tolerance
    0.01, 20 yineleme) optimuma yakın durur → parametreler 1e-2 içinde, bizim kayıp ≤ MATLAB'inki."""
    z = _z("arx1", 0.1)
    for key, data in (("armax_def", z), ("armax_def_tail", z[200:])):
        m = armax(data, [2, 2, 2, 1])
        ref = ml[key]
        assert m.report.initial_condition == ref["ic"]
        st = _PolyStructure(2, [2], 2, 0, [0], [1])
        assert rel(m.getpvec(), _pack_ref(st, ref)) < 1e-2
        assert m.report.fit.LossFcn <= ref["fit"]["LossFcn"] * (1 + 1e-9)
    m = oe(_z("oe1", 1.0), [2, 2, 2])
    assert m.report.initial_condition == ml["oe_def"]["ic"]
    assert rel(m.getpvec(), _pack_ref(_PolyStructure(0, [2], 0, 0, [2], [2]), ml["oe_def"])) < 1e-2


def test_miso_pem(ml):
    z = _z("miso1", 1.0)
    m = armax(z, [1, 2, 1, 1, 1, 2], initial_condition="estimate")
    st = _PolyStructure(1, [2, 1], 1, 0, [0, 0], [1, 2])
    assert rel(m.getpvec(), _pack_ref(st, ml["miso_armax"])) < 2e-6
    assert m.NoiseVariance == pytest.approx(ml["miso_armax"]["nv"], rel=1e-8)
    m = oe(z, [2, 1, 1, 1, 1, 2], initial_condition="estimate")
    st = _PolyStructure(0, [2, 1], 0, 0, [1, 1], [1, 2])
    assert rel(m.getpvec(), _pack_ref(st, ml["miso_oe"])) < 2e-6
    _check_fit(m, ml["miso_oe"], 1e-7)


def test_arma_time_series(ml):
    z = _z("ts1", 1.0)
    m = armax(z, [2, 1], initial_condition="estimate")
    ref = ml["ts_armax_e"]
    assert rel(m.A, ref["A"]) < 2e-6 and rel(m.C, ref["C"]) < 2e-6
    assert m.NoiseVariance == pytest.approx(ref["nv"], rel=1e-8)
    assert rel(predict(m, z, 2).y[:, 0], ml["ts_pred2"]) < 1e-6


# ============================================================================================ değerlendirme
def test_evaluation_functions_match_matlab(ml):
    """compare / predict / pe / resid / sim, gürültü modelli (C ≠ 1) modelde — MATLAB'in kendi modeliyle."""
    z = _z("arx1", 0.1)
    from spt.control.sysid import idpoly
    ref = ml["armax_e"]
    m = idpoly(ref["A"], ref["B"], ref["C"], Ts=0.1, noise_variance=ref["nv"])
    r = compare(z, m)
    assert r.fit[0][0] == pytest.approx(ml["cmp_fit"], rel=1e-10)
    assert rel(r.yhat[0].y[:, 0], ml["cmp_y"]) < 1e-10
    r = compare(z, m, k=2)
    assert r.fit[0][0] == pytest.approx(ml["cmp2_fit"], rel=1e-10)
    assert rel(r.yhat[0].y[:, 0], ml["cmp2_y"]) < 1e-10
    assert rel(predict(m, z, 3).y[:, 0], ml["pred3"]) < 1e-10
    assert rel(pe(m, z).y[:, 0], ml["pe"]) < 1e-9
    assert rel(resid(z, m).R, ml["resid_R"]) < 1e-9
    assert rel(sim(m, z).y[:, 0], ml["sim"]) < 1e-10


def test_resid_confidence_definition():
    """Güven sınırları: özilinti 2.58/√N, çapraz ilinti 2.58·√(1 + 2Σ r_e r_u)/√N (MATLAB residCorr tanımı)."""
    z = _z("arx1", 0.1)
    m = arx(z, [2, 2, 1])
    r = resid(z, m, max_lag=10)
    N = 600
    assert r.conf_auto[0] == pytest.approx(2.58 / np.sqrt(N))
    R = r.R
    re = R[:, 0, 0] / R[0, 0, 0]
    ru = R[:, 1, 1] / R[0, 1, 1]
    assert r.conf_cross[0, 0] == pytest.approx(2.58 * np.sqrt(1 + 2 * np.sum(re[1:] * ru[1:])) / np.sqrt(N))
    assert r.auto[0, 0] == 1.0
    # ARX artığı regresörlerine dik: gecikme 1 ve 2'de giriş–artık çapraz ilintisi ~0
    i1 = list(r.lags_cross).index(1)
    assert abs(r.cross[i1, 0, 0]) < 1e-12 and abs(r.cross[i1 + 1, 0, 0]) < 1e-12


# ============================================================================================ kuramsal
def test_noiseless_recovery_exact():
    rng = np.random.default_rng(5)
    u = rng.standard_normal(400)
    A, B = [1, -1.3, 0.5], [0, 0, 0.7, -0.2]
    y = signal.lfilter(B, A, u)
    z = IDData(y, u, 1.0)
    m = arx(z, [2, 2, 2])
    assert np.allclose(m.A, A, atol=1e-11) and np.allclose(m.B, B, atol=1e-11)
    m = oe(z, [2, 2, 2], initial_condition="zero")
    assert np.allclose(m.F, A, atol=1e-8) and np.allclose(m.B, B, atol=1e-8)
    C = [1, 0.4]
    e = 0.0 * u
    y2 = signal.lfilter(B, A, u) + signal.lfilter(C, A, e)
    m = armax(IDData(y2, u, 1.0), [2, 2, 1, 2], initial_condition="zero")
    assert np.allclose(m.A, A, atol=1e-7) and np.allclose(m.B, B, atol=1e-7)


def test_covariance_monte_carlo():
    """Rapor edilen standart sapma Monte Carlo dağılımıyla uyumlu (ARX 300, OE 60 gerçekleşme)."""
    rng = np.random.default_rng(11)
    A, B = [1, -1.2, 0.5], [0, 1.0, 0.4]
    est, sds = [], []
    for _ in range(300):
        u = rng.standard_normal(300)
        y = signal.lfilter(B, A, u) + signal.lfilter([1], A, 0.3 * rng.standard_normal(300))
        m = arx(IDData(y, u, 1.0), [2, 2, 1])
        est.append(m.getpvec())
        sds.append(m.std())
    ratio = np.std(est, axis=0) / np.mean(sds, axis=0)
    assert np.all(np.abs(ratio - 1) < 0.15), ratio
    est, sds = [], []
    for _ in range(60):
        u = rng.standard_normal(300)
        y = signal.lfilter(B, A, u) + 0.3 * rng.standard_normal(300)
        m = oe(IDData(y, u, 1.0), [2, 2, 1], initial_condition="zero")
        est.append(m.getpvec())
        sds.append(m.std())
    ratio = np.std(est, axis=0) / np.mean(sds, axis=0)
    assert np.all(np.abs(ratio - 1) < 0.3), ratio


def test_errors_are_turkish_and_explicit():
    z = _z("arx1", 0.1)
    y = z.y.copy()
    y[10] = np.nan
    with pytest.raises(ControlError, match="eksik"):
        arx(IDData(y, z.u, 0.1), [2, 2, 1])
    with pytest.raises(ControlError):
        arx(z, [2, 2])
    with pytest.raises(ControlError):
        oe(_z("ts1", 1.0), [2, 2, 1])
    with pytest.raises(ControlError):
        armax(_z("mimo1", 1.0, ny=2), [2, 2, 2, 1])
