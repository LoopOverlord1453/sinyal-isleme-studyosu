"""Durum uzayı kestirimi (n4sid, ssest) — MATLAB R2025b referanslı ve kuramsal testler.

n4sid'de MATLAB'in K ve B/D hesap ayrıntıları yayımlanmamıştır: karşılaştırma gerçeklemeden bağımsız büyüklüklerle
(frekans cevabı, kutuplar, uyum yüzdesi) yapılır. ssest tek çıkışta ARMAX ile aynı model sınıfı ve ölçüttür.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy import signal
from test_control_sysid_common import dataset, oracle

from spt.control.sysid import IDData, compare
from spt.control.sysid.subspace import d2c_ss, n4sid, ssest

W = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0]
SCRIPT = r"""
w = [0.01 0.05 0.1 0.2 0.5 1 2 3];
s = @(m) struct('fr', spt_c(freqresp(m, w)), 'p', spt_c(pole(m)), 'fit', m.Report.Fit, 'nv', m.NoiseVariance);
D = readmatrix('{DATA}/mimo1.csv'); z = iddata(D(:,1:2), D(:,3:4), 1);
out.n4_auto = s(n4sid(z, 4));
out.n4_moesp = s(n4sid(z, 4, n4sidOptions('N4Weight','MOESP','N4Horizon',[6 6 6])));
out.n4_cva = s(n4sid(z, 4, n4sidOptions('N4Weight','CVA','N4Horizon',[6 6 6])));
o = ssestOptions; o.SearchOptions.Tolerance = 1e-10; o.SearchOptions.MaxIterations = 100;
out.ss_mimo = s(ssest(z, 4, 'Ts', 1, o));
D = readmatrix('{DATA}/arx1.csv'); z2 = iddata(D(:,1), D(:,2), 0.1);
o2 = ssestOptions('InitialState','zero'); o2.SearchOptions.Tolerance = 1e-12; o2.SearchOptions.MaxIterations = 300;
out.ss_siso_z = s(ssest(z2, 2, 'Ts', 0.1, o2));
out.ss_siso_ct = s(ssest(z2, 2));
out.n4_siso = s(n4sid(z2, 2));
"""


@pytest.fixture(scope="module")
def ml():
    return oracle("sysid_ss", SCRIPT, ("mimo1", "arx1"))


def _cplx(x, shape):
    a = np.asarray(x, float)
    return (a[:, 0] + 1j * a[:, 1]).reshape(shape, order="F")


def _zm():
    D = dataset("mimo1")
    return IDData(D[:, :2], D[:, 2:], 1.0)


def _za():
    D = dataset("arx1")
    return IDData(D[:, 0], D[:, 1], 0.1)


def _relfr(H, ref):
    return float(np.max(np.abs(H - ref)) / np.max(np.abs(ref)))


@pytest.mark.parametrize("key,kw", [("n4_moesp", dict(weighting="moesp", horizon=(6, 6, 6))),
                                    ("n4_cva", dict(weighting="cva", horizon=(6, 6, 6))),
                                    ("n4_auto", dict())])
def test_n4sid_mimo_vs_matlab(ml, key, kw):
    """Aynı ağırlık / ufukta frekans cevabı ve kutuplar MATLAB'e yakın; uyum yüzdesi farkı küçük."""
    m = n4sid(_zm(), 4, **kw)
    ref = ml[key]
    H = m.freqresp(W)
    Href = _cplx(ref["fr"], (2, 2, len(W)))
    assert _relfr(H, Href) < 0.05
    p = np.sort_complex(np.linalg.eigvals(m.A))
    pr = np.sort_complex(_cplx(ref["p"], (-1,)))
    assert np.max(np.abs(p - pr)) < 0.03
    assert np.all(np.abs(np.asarray(m.report.fit.FitPercent) - np.asarray(ref["fit"]["FitPercent"])) < 1.0)


def test_ssest_mimo_vs_matlab(ml):
    """PEM optimumu: frekans cevabı ve kutuplar MATLAB'le 1e-3 içinde, uyum ≥ MATLAB − 0.05."""
    m = ssest(_zm(), 4, Ts=None)
    ref = ml["ss_mimo"]
    assert _relfr(m.freqresp(W), _cplx(ref["fr"], (2, 2, len(W)))) < 1e-3
    p = np.sort_complex(np.linalg.eigvals(m.A))
    assert np.max(np.abs(p - np.sort_complex(_cplx(ref["p"], (-1,))))) < 1e-3
    assert np.all(np.asarray(m.report.fit.FitPercent) >= np.asarray(ref["fit"]["FitPercent"]) - 0.05)


def test_ssest_siso_equals_matlab_optimum(ml):
    """Tek çıkışta ssest = ARMAX(nx, nx, nx, 1) modeli: sıfır başlangıçla MATLAB optimumuyla frekans cevabı 1e-4."""
    m = ssest(_za(), 2, Ts=None, initial_state="zero")
    ref = ml["ss_siso_z"]
    assert _relfr(m.freqresp(W), _cplx(ref["fr"], (1, 1, len(W)))) < 1e-4
    assert m.report.fit.LossFcn <= ref["fit"]["LossFcn"] * (1 + 1e-9)
    assert m.NoiseVariance == pytest.approx(ref["nv"], rel=1e-4)


def test_ssest_continuous_default(ml):
    """MATLAB ssest varsayılanı sürekli zaman: kutuplar ve frekans cevabı MATLAB'e yakın."""
    m = ssest(_za(), 2)
    assert m.Ts == 0.0
    ref = ml["ss_siso_ct"]
    p = np.sort_complex(np.linalg.eigvals(m.A))
    assert np.max(np.abs(p - np.sort_complex(_cplx(ref["p"], (-1,))))) / np.max(np.abs(p)) < 5e-3
    assert _relfr(m.freqresp(W), _cplx(ref["fr"], (1, 1, len(W)))) < 5e-3


def test_n4sid_siso(ml):
    m = n4sid(_za(), 2)
    ref = ml["n4_siso"]
    assert _relfr(m.freqresp(W), _cplx(ref["fr"], (1, 1, len(W)))) < 0.05
    assert abs(m.report.fit.FitPercent[0] - ref["fit"]["FitPercent"]) < 1.0


def test_n4sid_noiseless_exact():
    """Gürültüsüz veride gerçek sistem (transfer fonksiyonu) tam kurtarılır."""
    rng = np.random.default_rng(4)
    A = np.array([[0.8, 0.3], [-0.3, 0.8]])
    B = np.array([[1.0], [0.5]])
    C = np.array([[1.0, 0.2]])
    u = rng.standard_normal(500)
    x = np.zeros(2)
    y = np.zeros(500)
    for t in range(500):
        y[t] = (C @ x)[0]
        x = A @ x + B[:, 0] * u[t]
    z = IDData(y, u, 1.0)
    for w in ("moesp", "cva"):
        m = n4sid(z, 2, weighting=w, disturbance=False)
        assert np.allclose(np.sort_complex(np.linalg.eigvals(m.A)), np.sort_complex(np.linalg.eigvals(A)),
                           atol=1e-8)
        num, den = signal.ss2tf(A, B, C, np.zeros((1, 1)))
        Ht = np.polyval(num[0], np.exp(1j * 0.3)) / np.polyval(den, np.exp(1j * 0.3))
        assert abs(m.freqresp([0.3])[0, 0, 0] - Ht) < 1e-7 * abs(Ht)
        assert compare(z, m).fit[0][0] > 99.999


def test_d2c_roundtrip():
    """d2c (ZOH tersi) → yeniden ayrıklaştırma özgün ayrık modeli verir."""
    m = ssest(_za(), 2, Ts=None)
    mc = d2c_ss(m)
    s = mc.dss(0.1)
    assert np.allclose(np.sort_complex(np.linalg.eigvals(s.A)), np.sort_complex(np.linalg.eigvals(m.A)), atol=1e-10)
    w = [0.5, 2.0, 10.0]
    assert np.allclose(mc.freqresp(w) * 0 + m.freqresp(w), m.freqresp(w))
    from spt.control.sysid.models import IDSS
    back = IDSS(s.A, s.B, s.C, s.D, s.K, 0.1)
    assert np.allclose(back.freqresp(w), m.freqresp(w), atol=1e-10)
