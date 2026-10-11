"""Sürekli zamanlı kestirim (tfest, procest) — MATLAB R2025b referanslı ve kuramsal testler.

Ölçüt: kesirli gecikme dahil tam ZOH ayrıklaştırmayla benzetim hatası (OE). MATLAB sıkı toleransla çalıştırılır;
ölçüt tanımı MATLAB parametrelerinde MATLAB kaybını yeniden üreterek doğrulanır.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy import signal
from test_control_sysid_common import _zoh_ct, dataset, oracle

from spt.control.errors import ControlError
from spt.control.sysid import IDData, compare
from spt.control.sysid.ctime import _CTOE, _ct_to_dtf, procest, tfest
from spt.control.sysid.models import IDProc, IDTF

SCRIPT = r"""
tf_ = @(m) struct('num', m.Numerator, 'den', m.Denominator, 'iod', m.IODelay, 'nv', m.NoiseVariance, 'fit', m.Report.Fit, 'cov', getcov(m), 'pvec', getpvec(m), 'ic', m.Report.InitialCondition);
D = readmatrix('{DATA}/ct1.csv'); z = iddata(D(:,1), D(:,2), 0.05);
o = tfestOptions('InitialCondition','zero'); o.SearchOptions.Tolerance = 1e-12; o.SearchOptions.MaxIterations = 300;
out.tf_z = tf_(tfest(z, 2, 1, 0.12, o));
out.tf_zd = tf_(tfest(z, 2, 1, NaN, o));
oe_ = o; oe_.InitialCondition = 'estimate';
out.tf_e = tf_(tfest(z, 2, 1, 0.12, oe_));
out.tf_def = tf_(tfest(z, 2));
md = tfest(z, 2, 1, 2, 'Ts', 0.05); out.tf_disc = struct('num', md.Numerator, 'den', md.Denominator, 'iod', md.IODelay, 'Ts', md.Ts);
po = procestOptions('InitialCondition','zero','DisturbanceModel','none'); po.SearchOptions.Tolerance = 1e-12; po.SearchOptions.MaxIterations = 300;
pe_ = po; pe_.InitialCondition = 'estimate';
pr = @(p) struct('Kp', p.Kp, 'Tp1', p.Tp1, 'Tp2', p.Tp2, 'Tp3', p.Tp3, 'Tw', p.Tw, 'Zeta', p.Zeta, 'Td', p.Td, 'Tz', p.Tz, 'nv', p.NoiseVariance, 'fit', p.Report.Fit, 'cov', getcov(p), 'pvec', getpvec(p), 'ic', p.Report.InitialCondition);
out.p2dzu = pr(procest(z, 'P2DZU', po));
D = readmatrix('{DATA}/proc1.csv'); z1 = iddata(D(:,1), D(:,3), 0.1); z2 = iddata(D(:,2), D(:,3), 0.1);
out.p1d = pr(procest(z1, 'P1D', po));
out.p1d_e = pr(procest(z1, 'P1D', pe_));
out.p2d = pr(procest(z2, 'P2D', po));
p = procest(z1, 'P1D'); out.p1d_def = pr(p); out.p1d_def_noise = evalc('disp(p.NoiseTF)');
"""


@pytest.fixture(scope="module")
def ml():
    return oracle("sysid_ct", SCRIPT, ("ct1", "proc1"))


def _zct():
    D = dataset("ct1")
    return IDData(D[:, 0], D[:, 1], 0.05)


def _zp(i):
    D = dataset("proc1")
    return IDData(D[:, i], D[:, 2], 0.1)


def _cost_tf(z, num, den, delay, ic="zero"):
    exps = [(z.y[:, 0], z.u)]

    def build(th):
        return [_ct_to_dtf(num, den, delay, z.Ts)]

    p = _CTOE(exps, build, 0, ic).setup(np.zeros(0))
    r = p.resid(np.r_[np.zeros(0), p.best_s(np.zeros(0))] if ic == "estimate" else np.zeros(0))
    return float(r @ r) / len(r)


def rel(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-300))


# ============================================================================================ tfest
def test_tfest_zero_ic_matches_matlab(ml):
    z = _zct()
    ref = ml["tf_z"]
    assert _cost_tf(z, ref["num"], ref["den"], ref["iod"]) == pytest.approx(ref["fit"]["LossFcn"], rel=1e-10)
    m = tfest(z, 2, 1, 0.12, initial_condition="zero")
    assert m.report.fit.LossFcn <= ref["fit"]["LossFcn"] * (1 + 1e-10)
    assert rel(m.num[0], ref["num"]) < 1e-4 and rel(m.den[0], ref["den"]) < 1e-4
    assert m.NoiseVariance == pytest.approx(ref["nv"], rel=1e-6)  # sürekli modelde Ts·λ
    assert m.report.fit.FPE == pytest.approx(ref["fit"]["FPE"], rel=1e-6)
    assert rel(m.getpvec(), ref["pvec"][:4]) < 1e-4  # MATLAB getpvec sabit gecikmeyi de içerir
    # OE yapısı: MATLAB n4sid AR(5) gürültü modeliyle renkli artık düzeltmesi yapar (yayımlanmamış) → %5–10 fark;
    # bizimki beyaz artık için asimptotik kovaryans λ(JᵀJ)⁻¹ (Ljung 9.18; Monte Carlo testi linear dosyasında)
    assert rel(np.diag(m.getcov()), np.diag(np.asarray(ref["cov"]))[:4]) < 0.15


def test_tfest_estimated_delay(ml):
    """Bilinçli fark: MATLAB tfest gecikmeyi tamsayı örnek ızgarasında belirliyor (ölçüldü: 0.1 s = 2 örnek);
    biz gecikmeyi sürekli parametre olarak kestiririz → kayıp MATLAB'inkinden düşük, gecikme gerçeğe (0.12 s)
    daha yakın."""
    z = _zct()
    ref = ml["tf_zd"]
    assert abs(ref["iod"] / z.Ts - round(ref["iod"] / z.Ts)) < 1e-9
    m = tfest(z, 2, 1, np.nan, initial_condition="zero")
    assert m.report.fit.LossFcn < ref["fit"]["LossFcn"]
    assert abs(m.io_delay[0] - 0.12) < abs(ref["iod"] - 0.12)
    # MATLAB'in gecikmesi sabitken aynı optimum
    m2 = tfest(z, 2, 1, ref["iod"], initial_condition="zero")
    assert rel(m2.den[0], ref["den"]) < 1e-4 and rel(m2.num[0], ref["num"]) < 1e-4


def test_tfest_estimate_ic(ml):
    z = _zct()
    ref = ml["tf_e"]
    assert _cost_tf(z, ref["num"], ref["den"], ref["iod"], "estimate") == pytest.approx(ref["fit"]["LossFcn"],
                                                                                       rel=1e-9)
    m = tfest(z, 2, 1, 0.12, initial_condition="estimate")
    assert m.report.fit.LossFcn <= ref["fit"]["LossFcn"] * (1 + 1e-10)
    assert rel(m.den[0], ref["den"]) < 1e-4 and rel(m.num[0], ref["num"]) < 1e-4
    assert m.report.fit.FPE == pytest.approx(ref["fit"]["FPE"], rel=1e-5)


def test_tfest_defaults(ml):
    """tfest(z, 2): nz = np − 1 (MATLAB pay uzunluğu), gecikme 0. MATLAB'in 'auto' seçimi başlangıç modeline bağlıdır
    (burada 'estimate'); aynı başlangıç koşuluyla optimum MATLAB'in varsayılan sonucuna yakın (MATLAB varsayılan
    durdurma ölçütüyle erken durur)."""
    z = _zct()
    ref = ml["tf_def"]
    m = tfest(z, 2)
    assert len(m.num[0]) == len(np.atleast_1d(ref["num"]))
    m = tfest(z, 2, initial_condition=ref["ic"])
    assert rel(m.den[0], ref["den"]) < 1e-2 and rel(m.num[0], ref["num"]) < 1e-2
    assert m.report.fit.FitPercent[0] >= ref["fit"]["FitPercent"] - 1e-3


def test_tfest_discrete_format(ml):
    """Ayrık tfest: pay [0, b1 … b_nz] (bir örnek örtük gecikme) · z^(−iodelay), MATLAB ile aynı yapı."""
    ref = ml["tf_disc"]
    m = tfest(_zct(), 2, 1, 2, Ts=0.05)
    assert m.Ts == ref["Ts"] and m.io_delay[0] == ref["iod"]
    assert len(m.num[0]) == len(ref["num"]) and m.num[0][0] == 0.0 and len(m.den[0]) == len(ref["den"])
    assert rel(m.den[0], ref["den"]) < 5e-3 and rel(m.num[0], ref["num"]) < 5e-3


# ============================================================================================ procest
def _proc_params(ref, names):
    return {k: float(ref[k]) for k in names}


@pytest.mark.parametrize("key,ds,typ,names", [
    ("p1d", 0, "P1D", ["Kp", "Tp1", "Td"]),
    ("p2d", 1, "P2D", ["Kp", "Tp1", "Tp2", "Td"]),
    ("p2dzu", None, "P2DZU", ["Kp", "Tw", "Zeta", "Td", "Tz"]),
])
def test_procest_matches_matlab(ml, key, ds, typ, names):
    """Ölçüt MATLAB parametrelerinde MATLAB'in MSE'sini 1e-10 ile verir (MATLAB'in LossFcn'i iç gecikme gösterimi
    nedeniyle MSE'den ~1e-8 farklı). MATLAB Td'yi Ts/1000 ızgarasında tutar (ölçüldü: 0.3467, 0.246, 0.11955);
    biz sürekli tahmin ederiz → Td farkı ≤ Ts/1000, diğer parametreler 1e-4."""
    z = _zct() if ds is None else _zp(ds)
    ref = ml[key]
    pm = IDProc(typ, _proc_params(ref, names))
    n, d, td = pm.tf_coeffs()
    assert _cost_tf(z, n / d[0], d / d[0], td) == pytest.approx(ref["fit"]["MSE"], rel=1e-10)
    p = procest(z, typ, initial_condition="zero")
    assert p.report.fit.LossFcn <= ref["fit"]["MSE"] * (1 + 1e-10)
    for k in names:
        if k == "Td":
            assert abs(p.params[0][k] - ref[k]) <= z.Ts / 1000 + 1e-12
        else:
            assert p.params[0][k] == pytest.approx(ref[k], rel=1e-4), k
    assert p.NoiseVariance == pytest.approx(ref["nv"], rel=1e-4)


def test_procest_estimate_ic_and_default(ml):
    z = _zp(0)
    ref = ml["p1d_e"]
    p = procest(z, "P1D", initial_condition="estimate")
    for k in ("Kp", "Tp1"):
        assert p.params[0][k] == pytest.approx(ref[k], rel=1e-4), k
    assert abs(p.params[0]["Td"] - ref["Td"]) <= z.Ts / 1000 + 1e-12
    assert p.report.fit.FPE == pytest.approx(ref["fit"]["FPE"], rel=1e-5)
    # MATLAB varsayılanı (DisturbanceModel 'estimate'): biz gürültü modelsiz kestiririz; ölçüm modeli yakın
    d = ml["p1d_def"]
    p = procest(z, "P1D")
    for k in ("Kp", "Tp1", "Td"):
        assert p.params[0][k] == pytest.approx(d[k], rel=2e-2), k


# ============================================================================================ kuramsal
def test_zoh_fractional_delay_discretization_exact():
    """Kesirli gecikmeli ZOH ayrıklaştırması bağımsız ince ızgara benzetimiyle aynı (1e-9)."""
    rng = np.random.default_rng(3)
    u = np.repeat(rng.standard_normal(40), 5)
    Ts = 0.1
    num, den, tau = [1.0, 3.0], [1.0, 1.2, 2.0], 0.27
    yref = _zoh_ct(num, den, u, Ts, delay=tau, sub=100)
    b, a, d = _ct_to_dtf(num, den, tau, Ts)
    y = signal.lfilter(b, a, np.r_[np.zeros(d), u][:len(u)])
    assert np.max(np.abs(y - yref)) < 1e-9 * max(1, np.max(np.abs(yref)))


def test_tfest_noiseless_exact_recovery():
    rng = np.random.default_rng(8)
    Ts = 0.05
    u = np.repeat(rng.standard_normal(150), 4)
    y = _zoh_ct([2.0, 8.0], [1.0, 1.6, 4.0], u, Ts, delay=0.15, sub=10)
    m = tfest(IDData(y, u, Ts), 2, 1, 0.15, initial_condition="zero")
    assert np.allclose(m.num[0], [2, 8], rtol=1e-6) and np.allclose(m.den[0], [1, 1.6, 4], rtol=1e-6)
    m = tfest(IDData(y, u, Ts), 2, 1, np.nan, initial_condition="zero")
    assert m.io_delay[0] == pytest.approx(0.15, abs=1e-6)


def test_procest_integrating_noiseless():
    rng = np.random.default_rng(9)
    Ts = 0.1
    u = np.repeat(rng.standard_normal(60), 10)
    u -= u.mean()
    y = _zoh_ct([0.7], np.polymul([0.8, 1.0], [1.0, 0.0]), u, Ts, delay=0.3, sub=10)
    p = procest(IDData(y, u, Ts), "P1DI", initial_condition="zero")
    assert p.params[0]["Kp"] == pytest.approx(0.7, rel=1e-5)
    assert p.params[0]["Tp1"] == pytest.approx(0.8, rel=1e-5)
    assert p.params[0]["Td"] == pytest.approx(0.3, abs=1e-6)


def test_compare_with_continuous_model_uses_exact_zoh():
    z = _zct()
    m = IDTF([2.0, 8.0], [1.0, 1.6, 4.0], 0.0, io_delay=0.12)
    f = compare(z, m, initial_condition="z").fit[0][0]
    # gürültü 0.05, çıkış std ~1.4 → uyum ≈ 100(1 − 0.05/1.4) ≈ 96.4
    assert 95.5 < f < 97.5


def test_tfest_delay_estimate_at_zero_bound_regression():
    """Arayüz hatası (gecikmeyi tahmin et): gecikme optimumu 0 sınırındayken sayısal türev τ < 0 noktasını
    değerlendirip çöküyordu. Şimdi sınırda tek yönlü fark + etkin küme: çökmez, τ ≥ 0, yakınsar."""
    D = dataset("arx1")  # ayrık sistem, 1 örnek gecikme → sürekli modelde τ = 0 sınırı
    z = IDData(D[:, 0], D[:, 1], 0.1)
    from spt.control.sysid import split
    est, _ = split(z, 0.5)
    for d in (est, z):
        m = tfest(d, 2, 1, np.nan)
        assert m.io_delay[0] >= 0.0
        assert m.report.why_stop in ("yakınsadı", "adım çok küçük", "iyileşme yok (yerel en küçük)")
        assert np.all(np.real(np.roots(m.den[0])) < 0)
    p = procest(z, "P1D")  # Td de 0 sınırına yakın
    assert p.params[0]["Td"] >= 0.0


def test_procest_type_errors():
    with pytest.raises(ControlError):
        procest(_zp(0), "P1U")
    with pytest.raises(ControlError):
        procest(_zp(0), "X2")
