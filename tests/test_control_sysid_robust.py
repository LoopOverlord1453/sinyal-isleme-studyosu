"""Denetim bulguları için regresyon testleri: FOH verisi, sabit giriş (yetersiz uyarım), kısa veri, iş katmanı."""
from __future__ import annotations

import warnings

import numpy as np
import pytest
from scipy import signal

from spt.control.errors import ControlError, ControlWarning
from spt.control.sysid import (IDData, arx, bj, compare, n4sid, oe, procest, resample, spa, split, ssest, tfest)


def _foh_data():
    rng = np.random.default_rng(7)
    N, Ts = 1000, 0.01
    u = np.sign(rng.standard_normal(N))
    y = signal.lfilter([0, 0.5, 0.3], [1, -1.2, 0.5], u) + signal.lfilter([1, 0.6], [1, -0.7],
                                                                          0.05 * rng.standard_normal(N))
    z = resample(IDData(y, u, Ts), 1, 2)  # Q ≠ 1 → giriş FOH (MATLAB iddata/resample)
    assert z.intersample == ("foh",)
    return split(z, 0.5)


def test_continuous_ssest_on_foh_data_consistent():
    """Bulgu 1: FOH verisinde sürekli ssest artık FOH ayrıklaştırmasıyla kestirilir: rapor uyumu aynı başlangıç
    koşuluyla compare(k=1) ile birebir; benzetim uyumu ayrık modelinkiyle tutarlı (önceden %55.7)."""
    est, val = _foh_data()
    for ic in ("zero", "estimate"):
        m = ssest(est, 2, initial_state=ic)
        assert m.Ts == 0.0
        k1 = compare(est, m, k=1, initial_condition="z" if ic == "zero" else "e").fit[0][0]
        assert k1 == pytest.approx(m.report.fit.FitPercent[0], abs=1e-9)
    md = ssest(est, 2, Ts=est.Ts)
    assert compare(val, m).fit[0][0] >= compare(val, md).fit[0][0] - 1.0
    assert compare(val, m).fit[0][0] > 85.0


def test_foh_fractional_delay_clear_message():
    """Bulgu 3: FOH'ta kesirli gecikme: anlaşılır Türkçe neden + çözüm (iş başlamadan)."""
    est, _ = _foh_data()
    for f in (lambda: tfest(est, 2, 1, np.nan), lambda: procest(est, "P2D"),
              lambda: tfest(est, 2, 1, 0.5 * est.Ts)):
        with pytest.raises(ControlError, match="FOH.*Yeniden örneklemeyi kapatın"):
            f()
    tfest(est, 2, 1, 2 * est.Ts)  # örnek katı gecikme FOH'ta tam


def test_spa_constant_input_turkish_error():
    """Bulgu 2: sabit girişte spa LinAlgError yerine Türkçe ControlError."""
    z = IDData(np.random.default_rng(1).standard_normal(200), np.zeros(200), 0.1)
    with pytest.raises(ControlError, match="yetersiz uyarım"):
        spa(z)


def test_short_data_no_zero_division():
    """Bulgu 4: kısa veride ZeroDivisionError yok; N ≤ d ise MATLAB gibi NoiseVariance/FPE NaN + uyarı."""
    rng = np.random.default_rng(1)
    u = rng.standard_normal(6)
    y = signal.lfilter([0, 0.5, 0.3], [1, -1.2, 0.5], u) + 0.01 * rng.standard_normal(6)
    z = IDData(y, u, 0.1)
    with pytest.warns(ControlWarning, match="Gürültü varyansı"):
        m = arx(z, [2, 2, 1])
    assert np.isnan(m.NoiseVariance) and np.isnan(m.report.fit.FPE)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ControlWarning)
        oe(z, [2, 2, 1])
        tfest(z, 2, 1)
    with pytest.raises(ControlError, match="çok kısa"):
        n4sid(z, 2)


def test_constant_input_warns_and_parameters_stay_bounded():
    """Bulgu 6: sabit girişte uyarı; tanımlanamayan parametreler adımdan çıkarıldığı için anlamsız büyümez."""
    rng = np.random.default_rng(2)
    e = rng.standard_normal(200)
    y = signal.lfilter([1, 0.5], [1, -0.7], e)
    z = IDData(y, np.zeros(200), 0.1)
    with pytest.warns(ControlWarning, match="yetersiz uyarım"):
        m = bj(z, [2, 2, 2, 2, 1])
    assert np.max(np.abs(m.getpvec())) < 10.0
    assert any("yetersiz uyarım" in w for w in m.report.warnings)


def test_jobs_layer_robustness():
    """İş katmanı: spa hatası işi düşürmez (model korunur), ayrık + gecikme tahmini baştan reddedilir, bölme sonrası
    kısa veri ve negatif doğrulama uyumu uyarı olarak sınıflanır."""
    J = pytest.importorskip("spt.gui.control.sysid_jobs")
    rng = np.random.default_rng(3)
    N, Ts = 300, 0.1
    y = signal.lfilter([1, 0.5], [1, -0.7], rng.standard_normal(N))
    tab = J.SourceTable("k", "sabit", ["u", "y"], ["", ""], np.column_stack([np.ones(N), y]), Ts)
    with pytest.warns(ControlWarning):
        work, est, val = J.build_data(tab, [0], [1], Ts, J.Prep(detrend=0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ControlWarning)
        r = J.run_job({"est": est, "val": val, "work": work,
                       "new": {"method": "arx", "opts": dict(na=2, nb=2, nk=1), "name": "a"}, "models": []})
    assert r["new"] is not None and r["spa"] is None and "uyarım" in r["spa_error"]
    assert r["evals"]["a"].errors  # yetersiz uyarım uyarısı → durum "iyi" değil
    with pytest.raises(ControlError, match="Ayrık zamanlı"):
        J.plan("tfest", dict(np=2, nz=1, discrete=True, estimate_delay=True), 1, 1, 0.1)
    tab2 = J.SourceTable("k", "kısa", ["u", "y"], ["", ""], np.column_stack([rng.standard_normal(12), y[:12]]), Ts)
    with pytest.raises(ControlError, match="Bölme sonrası"):
        J.build_data(tab2, [0], [1], Ts, J.Prep(fraction=0.5))
