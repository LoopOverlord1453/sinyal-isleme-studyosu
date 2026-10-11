"""Doğrusal olmayan modeller (nlarx, nlhw) — MATLAB R2025b referanslı.

Doğrusal-parametreli durumlar (idLinear + polinom / özel regresörler) en küçük karelerdir: öngörüler MATLAB'le
birebir karşılaştırılır. Ağ eşlemelerinde (sigmoid, dalgacık) ve Hammerstein–Wiener'de MATLAB'in eğitim ayrıntısı
yayımlanmamıştır: aynı model sınıfı ve ölçütle doğrulama verisindeki uyum yüzdesi MATLAB'inkinden belirgin kötü
olmamalıdır.
"""
from __future__ import annotations

import numpy as np
import pytest
from test_control_sysid_common import dataset, oracle

from spt.control.sysid import IDData, compare, predict, sim
from spt.control.sysid.nonlinear import (CustomRegressor, DeadZone, Linear, Poly1D, PolyRegressor, Saturation, SigmoidNet,
                                         nlarx, nlhw)

SCRIPT = r"""
D = readmatrix('{DATA}/nl1.csv'); z = iddata(D(:,1), D(:,2), 1); ze = z(1:1000); zv = z(1001:1500);
m = nlarx(ze, [2 2 1], idLinear); yp = predict(m, zv, 1, predictOptions('InitialCondition','z')); out.lin_p1 = yp.y;
[~, f1] = compare(zv, m, 1); [~, fs] = compare(zv, m); out.lin_fit = [f1 fs];
r = linearRegressor({'y1','u1'}, {[1 2], [1 2]});
p = polynomialRegressor({'y1','u1'}, {1, 1}, 2);
c1 = customRegressor({'u1','y1'}, {1, 1}, @(a,b) a.*b);
c2 = customRegressor({'u1'}, {1}, @(x) tanh(x));
m = nlarx(ze, [r; p; c1; c2], idLinear); yp = predict(m, zv, 1, predictOptions('InitialCondition','z')); out.poly_p1 = yp.y;
ys = sim(m, zv); out.poly_sim = ys.y;
[~, f1] = compare(zv, m, 1); [~, fs] = compare(zv, m); out.poly_fit = [f1 fs];
m = nlarx(ze, [2 2 1], idSigmoidNetwork(10)); [~, f1] = compare(zv, m, 1); [~, fs] = compare(zv, m); out.sig_fit = [f1 fs];
m = nlarx(ze, [2 2 1]); [~, f1] = compare(zv, m, 1); [~, fs] = compare(zv, m); out.wav_fit = [f1 fs];
D = readmatrix('{DATA}/hw1.csv'); zh = iddata(D(:,1), D(:,2), 1); zhe = zh(1:1000); zhv = zh(1001:1500);
mh = nlhw(zhe, [2 2 1], idSaturation, idDeadZone); [~, f] = compare(zhv, mh); out.hw_fit = f;
out.hw_sat = mh.InputNonlinearity.LinearInterval; out.hw_dz = mh.OutputNonlinearity.ZeroInterval;
mh = nlhw(zhe, [2 2 1], idPiecewiseLinear, idPiecewiseLinear); [~, f] = compare(zhv, mh); out.hw_pwl_fit = f;
mh = nlhw(zhe, [2 2 1], idPolynomial1D(3), idUnitGain); [~, f] = compare(zhv, mh); out.hw_poly_fit = f;
"""


@pytest.fixture(scope="module")
def ml():
    return oracle("sysid_nonlinear", SCRIPT, ("nl1", "hw1"))


def _nl():
    D = dataset("nl1")
    z = IDData(D[:, 0], D[:, 1], 1.0)
    return z[:1000], z[1000:]


def _hw():
    D = dataset("hw1")
    z = IDData(D[:, 0], D[:, 1], 1.0)
    return z[:1000], z[1000:]


def rel(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-300))


def test_nlarx_linear_matches_matlab(ml):
    """idLinear: MATLAB nlarx en küçük karelerle aynı (sıfır başlangıçlı bir adım öngörü 1e-8)."""
    ze, zv = _nl()
    m = nlarx(ze, [2, 2, 1], Linear())
    assert rel(predict(m, zv, 1, initial_condition="z").y[:, 0], ml["lin_p1"]) < 1e-8
    r = compare(zv, m, k=1).fit[0][0]
    assert r == pytest.approx(ml["lin_fit"][0], abs=0.05)


def test_nlarx_polynomial_custom_regressors_match_matlab(ml):
    ze, zv = _nl()
    m = nlarx(ze, [2, 2, 1], Linear(),
              custom_regressors=[PolyRegressor(["y1", "u1"], [[1], [1]], 2),
                                 CustomRegressor(["u1", "y1"], [1, 1], lambda a, b: a * b, "u1*y1"),
                                 CustomRegressor(["u1"], [1], np.tanh, "tanh(u1)")])
    assert rel(predict(m, zv, 1, initial_condition="z").y[:, 0], ml["poly_p1"]) < 1e-8
    assert rel(sim(m, zv).y[:, 0], ml["poly_sim"]) < 1e-7
    assert compare(zv, m, k=1).fit[0][0] == pytest.approx(ml["poly_fit"][0], abs=0.05)


@pytest.mark.parametrize("key,nl", [("sig_fit", SigmoidNet(10)), ("wav_fit", "wavenet")])
def test_nlarx_networks_not_worse_than_matlab(ml, key, nl):
    """Ağ eşlemeleri: doğrulama verisinde öngörü ve benzetim uyumu MATLAB'inkinden en çok 2 puan düşük."""
    ze, zv = _nl()
    m = nlarx(ze, [2, 2, 1], nl)
    f1 = compare(zv, m, k=1).fit[0][0]
    fs = compare(zv, m).fit[0][0]
    assert f1 >= ml[key][0] - 2.0
    assert fs >= ml[key][1] - 2.0


def test_nlhw_saturation_deadzone(ml):
    """Doyum + ölü bölge: uyum MATLAB'inkinden belirgin kötü değil; sınırlar gerçeğe (−0.8, 1.0 / −0.2, 0.3) ve
    MATLAB'e yakın."""
    ze, zv = _hw()
    m = nlhw(ze, [2, 2, 1], Saturation(), DeadZone())
    f = compare(zv, m).fit[0][0]
    assert f >= ml["hw_fit"] - 1.0
    lo, hi = sorted(m.input_nl[0].LinearInterval)
    assert lo == pytest.approx(-0.8, abs=0.05) and hi == pytest.approx(1.0, abs=0.05)
    dlo, dhi = sorted(m.output_nl.ZeroInterval)
    # doyum ile doğrusal blok kazancı arasında ölçek belirsizliği yok (doyum eğimi 1); ölü bölge gerçeğe yakın
    assert dlo == pytest.approx(-0.2, abs=0.05) and dhi == pytest.approx(0.3, abs=0.05)


@pytest.mark.parametrize("key,inl,onl", [("hw_pwl_fit", "pwlinear", "pwlinear"), ("hw_poly_fit", Poly1D(3), "unitgain")])
def test_nlhw_other_nonlinearities(ml, key, inl, onl):
    ze, zv = _hw()
    m = nlhw(ze, [2, 2, 1], inl, onl)
    assert compare(zv, m).fit[0][0] >= ml[key] - 2.0


def test_nlarx_simulation_definition():
    """Benzetimde geçmiş çıkışlar yerine model çıktıları geri beslenir; doğrusal eşlemede ARX benzetimiyle aynı."""
    from spt.control.sysid import arx
    ze, zv = _nl()
    m = nlarx(ze, [2, 2, 1], Linear())
    a = arx(ze, [2, 2, 1])
    ys = sim(m, zv).y[:, 0]
    # nlarx doğrusal eşlemesi sabit terim içerir: ARX + sabit ile eşdeğer (aynı LS); burada yalnız tutarlılık
    assert np.all(np.isfinite(ys)) and np.corrcoef(ys, sim(a, zv).y[:, 0])[0, 1] > 0.999
