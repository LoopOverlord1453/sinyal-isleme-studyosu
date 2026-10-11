"""Veri (iddata) ve ön işleme + parametrik olmayan yöntemler (spa, etfe, impulseest, delayest, arxstruc, selstruc)
— MATLAB R2025b referanslı ve kuramsal testler."""
from __future__ import annotations

import numpy as np
import pytest
from scipy import io as sio
from scipy import signal
from test_control_sysid_common import dataset, oracle

from spt.control.errors import ControlError
from spt.control.sysid import (IDData, arx, detrend, fill_missing, iddata_from_handle, idfilt, mat_variables, merge,
                               read_csv, read_mat, resample, retrend, split)
from spt.control.sysid.nonparam import arxstruc, covf, delayest, etfe, impulseest, selstruc, spa, struc

SCRIPT = r"""
D = readmatrix('{DATA}/arx1.csv'); z = iddata(D(:,1), D(:,2), 0.1);
g = spa(z); out.spa_G = spt_c(squeeze(g.ResponseData)); out.spa_S = squeeze(g.SpectrumData); out.spa_w = g.Frequency;
g = spa(z, 20, [0.1 0.5 1 2 10]); out.spa2_G = spt_c(squeeze(g.ResponseData)); out.spa2_S = squeeze(g.SpectrumData);
D3 = readmatrix('{DATA}/miso1.csv'); zm = iddata(D3(:,1), D3(:,2:3), 0.5);
g = spa(zm, 25); out.spam_G = spt_c(g.ResponseData); out.spam_S = real(squeeze(g.SpectrumData));
g = etfe(z); out.etfe_G = spt_c(squeeze(g.ResponseData));
g = etfe(z, 30); out.etfe30_G = spt_c(squeeze(g.ResponseData));
g = etfe(z, 40, 64); out.etfe40_G = spt_c(squeeze(g.ResponseData)); out.etfe40_w = g.Frequency;
zt = iddata(D(:,1), [], 0.1);
g = etfe(zt, 30); out.etfe_ts_S = real(squeeze(g.SpectrumData));
g = spa(zt); out.spa_ts_S = real(squeeze(g.SpectrumData));
D4 = readmatrix('{DATA}/fir1.csv'); zf = iddata(D4(:,1), D4(:,2), 1);
o = impulseestOptions('RegularizationKernel','TC'); o.Advanced.AROrder = 0;
m = impulseest(zf, 40, 0, o); out.imp_tc = m.Numerator; out.imp_tc_fit = m.Report.Fit; out.imp_tc_nv = m.NoiseVariance;
o.RegularizationKernel = 'DC'; m = impulseest(zf, 40, 0, o); out.imp_dc = m.Numerator;
m = impulseest(zf, 40); out.imp_def = m.Numerator;
m = impulseest(zf); out.imp_auto = m.Numerator;
D5 = readmatrix('{DATA}/oe1.csv'); zo = iddata(D5(:,1), D5(:,2), 1);
out.dly_arx1 = delayest(z); out.dly_oe1 = delayest(zo); out.dly_miso = delayest(zm);
NN = struc(1:3, 1:3, 1:2); out.NN = NN;
V = arxstruc(z(1:300), z(301:600), NN); out.V = V;
out.nn_aic = selstruc(V, 'aic'); out.nn_mdl = selstruc(V, 'mdl'); out.nn_0 = selstruc(V, 0);
out.V_same = arxstruc(z, z, NN);
[zd, T] = detrend(zm, 1); out.d1_y = zd.y; out.d1_u = zd.u;
out.T1 = struct('io', T.InputOffset, 'oo', T.OutputOffset, 'is', T.InputSlope, 'os', T.OutputSlope);
zf1 = idfilt(zm, {[0.2 0.3], [1 -0.5]}); out.f1_y = zf1.y; out.f1_u = zf1.u;
zf2 = idfilt(zm, {[0.2 0.3], [1 -0.5]}, 'noncausal'); out.f2_y = zf2.y;
try, zf3 = idfilt(zm, 4, 0.3); out.f3_y = zf3.y; catch e, out.f3_err = e.message; end
try, zf4 = idfilt(zm, 3, [0.1 0.4]); out.f4_y = zf4.y; catch e, out.f4_err = e.message; end
try, zf5 = idfilt(zm, 4, 0.3, 'high'); out.f5_y = zf5.y; catch e, out.f5_err = e.message; end
try, zr = resample(z, 3, 2); out.r1_y = zr.y; out.r1_u = zr.u; out.r1_Ts = zr.Ts; catch e, out.r1_err = e.message; end
try, zr = resample(z, 1, 2); out.r2_y = zr.y; catch e, out.r2_err = e.message; end
try
y = D(:,1); idx = [50 51 52 200 333 334 480]; y(idx) = NaN; zz = iddata(y, D(:,2), 0.1);
zmis = misdata(zz); out.mis = zmis.y(idx);
catch e, out.mis_err = e.message; end
"""


@pytest.fixture(scope="module")
def ml():
    return oracle("sysid_data_nonparam", SCRIPT, ("arx1", "miso1", "fir1", "oe1"))


def rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-300))


def _c(x):
    a = np.asarray(x, float)
    return a[:, 0] + 1j * a[:, 1]


def _z():
    D = dataset("arx1")
    return IDData(D[:, 0], D[:, 1], 0.1)


def _zm():
    D = dataset("miso1")
    return IDData(D[:, 0], D[:, 1:], 0.5)


# ============================================================================================ iddata
def test_iddata_basics_and_selection():
    z = _z()
    assert z.tstart == pytest.approx(0.1)  # MATLAB: Tstart varsayılanı Ts
    assert z.size() == (600, 1, 1, 1)
    s = z[100:200]
    assert s.N == 100 and s.tstart == pytest.approx(0.1 + 100 * 0.1)
    assert np.allclose(s.time, 0.1 + 0.1 * np.arange(100, 200))
    zm = _zm()
    assert zm.select(inputs=["u2"]).input_names == ("u2",)
    zz = merge(z[:300], z[300:])
    assert zz.ne == 2 and zz.N == [300, 300] and zz.getexp(1).tstart == pytest.approx(0.1 + 30.0)
    e, v = split(z, 0.5)
    assert e.N == 300 and v.N == 300
    with pytest.raises(ControlError):
        IDData(np.ones(10), np.ones(9), 1.0)
    with pytest.raises(ControlError):
        IDData(np.ones(10), None, 0.0)


def test_detrend_retrend_matches_matlab(ml):
    zm = _zm()
    zd, T = detrend(zm, 1, return_trend=True)
    assert rel(zd.y[:, 0], ml["d1_y"]) < 1e-10
    assert rel(zd.u, ml["d1_u"]) < 1e-10
    assert rel(T.OutputOffset, ml["T1"]["oo"]) < 1e-10 and rel(T.OutputSlope, ml["T1"]["os"]) < 1e-10
    assert rel(T.InputOffset, ml["T1"]["io"]) < 1e-10 and rel(T.InputSlope, ml["T1"]["is"]) < 1e-10
    back = retrend(zd, T)
    assert np.allclose(back.y, zm.y, atol=1e-12) and np.allclose(back.u, zm.u, atol=1e-12)
    z0 = detrend(zm, 0)
    assert np.allclose(z0.y.mean(axis=0), 0, atol=1e-12)


def test_idfilt_matches_matlab(ml):
    zm = _zm()
    f1 = idfilt(zm, ([0.2, 0.3], [1, -0.5]))
    assert rel(f1.y[:, 0], ml["f1_y"]) < 1e-12 and rel(f1.u, ml["f1_u"]) < 1e-12
    f2 = idfilt(zm, ([0.2, 0.3], [1, -0.5]), causal=False)
    assert rel(f2.y[:, 0], ml["f2_y"]) < 1e-10
    for key, kw in (("f3_y", dict(order=4, band=0.3)), ("f4_y", dict(order=3, band=[0.1, 0.4])),
                    ("f5_y", dict(order=4, band=0.3, kind="high"))):
        if key in ml:
            assert rel(idfilt(zm, **kw).y[:, 0], ml[key]) < 1e-8, key


def test_resample_matches_matlab(ml):
    z = _z()
    if "r1_y" in ml:
        r = resample(z, 3, 2)
        assert r.Ts == pytest.approx(ml["r1_Ts"])
        assert rel(r.y[:, 0], ml["r1_y"]) < 1e-10 and rel(r.u[:, 0], ml["r1_u"]) < 1e-10
        assert r.intersample == ("foh",)
    if "r2_y" in ml:
        assert rel(resample(z, 1, 2).y[:, 0], ml["r2_y"]) < 1e-10


def test_fill_missing(ml):
    D = dataset("arx1")
    y = D[:, 0].copy()
    idx = np.array([50, 51, 52, 200, 333, 334, 480]) - 1
    truth = y[idx].copy()
    y[idx] = np.nan
    zz = IDData(y, D[:, 1], 0.1)
    with pytest.raises(ControlError, match="eksik"):
        arx(zz, [2, 2, 1])
    lin = fill_missing(zz, "linear")
    mod = fill_missing(zz, "model")  # varsayılan ARX [4 4 1]
    err_lin = np.linalg.norm(lin.y[idx, 0] - truth)
    err_mod = np.linalg.norm(mod.y[idx, 0] - truth)
    assert err_mod < err_lin
    if "mis" in ml:  # MATLAB misdata ile benzer doğruluk (farklı algoritma)
        err_ml = np.linalg.norm(np.asarray(ml["mis"]) - truth)
        assert err_mod <= 1.5 * err_ml + 1e-9
    assert not mod.has_missing and np.allclose(np.delete(mod.y[:, 0], idx), np.delete(D[:, 0], idx))


def test_readers(tmp_path):
    D = dataset("arx1")[:50]
    p = tmp_path / "veri.csv"
    t = 0.2 + 0.1 * np.arange(50)
    np.savetxt(p, np.column_stack([t, D]), delimiter=";", header="zaman;cikis;giris", comments="", fmt="%.15g")
    z = read_csv(p, outputs=["cikis"], inputs=["giris"], time_column="zaman")
    assert z.Ts == pytest.approx(0.1) and z.tstart == pytest.approx(0.2)
    assert np.allclose(z.y[:, 0], D[:, 0]) and z.output_names == ("cikis",)
    m = tmp_path / "veri.mat"
    sio.savemat(m, {"y": D[:, 0], "u": D[:, 1], "Ts": 0.1})
    assert "y" in mat_variables(m)
    z2 = read_mat(m, y="y", u="u")
    assert z2.Ts == 0.1 and np.allclose(z2.u[:, 0], D[:, 1])
    with pytest.raises(ControlError):
        read_csv(p, outputs=["yok"], Ts=1.0)


def test_iddata_from_snapshot_handle():
    from spt.core.local_ring import StaticRing
    from spt.core.snapshot import Snapshot
    from spt.protocol.format import SAMPLE_TYPES_BY_NAME, ChannelInfo, Descriptor, Layout

    raw = np.column_stack([np.arange(100, dtype=np.int16), 2 * np.arange(100, dtype=np.int16)])
    desc = Descriptor(1, 1, 50.0, SAMPLE_TYPES_BY_NAME["int16"], Layout.INTERLEAVED, "deney",
                      (ChannelInfo("u", "V", 0.5, 1.0), ChannelInfo("y", "m", 0.1, 0.0)))
    snap = Snapshot(1, "deney", "test", desc, 2.0, StaticRing(raw, 1000, [(1040, 1043)]))
    h = snap.handle()
    z = iddata_from_handle(h, inputs=["u"], outputs=[1], n0=1010, n1=1060)
    assert z.Ts == pytest.approx(0.02) and z.tstart == pytest.approx(2.0 + 1010 / 50)
    assert z.input_units == ("V",) and z.output_units == ("m",)
    assert z.u[0, 0] == pytest.approx(10 * 0.5 + 1.0) and z.y[5, 0] == pytest.approx(2 * 15 * 0.1)
    assert np.isnan(z.y[30:33]).all() and z.has_missing
    with pytest.raises(ControlError):
        iddata_from_handle(h, inputs=[0], outputs=[0])


# ============================================================================================ spa / etfe
def test_spa_matches_matlab(ml):
    z = _z()
    g = spa(z)
    assert rel(g.frequency, ml["spa_w"]) < 1e-14
    assert rel(g.response[0, 0], _c(ml["spa_G"])) < 1e-10
    assert rel(g.spectrum[0, 0], ml["spa_S"]) < 1e-10
    g = spa(z, 20, [0.1, 0.5, 1, 2, 10])
    assert rel(g.response[0, 0], _c(ml["spa2_G"])) < 1e-10
    assert rel(g.spectrum[0, 0], ml["spa2_S"]) < 1e-10
    gm = spa(_zm(), 25)
    Gm = _c(ml["spam_G"]).reshape((1, 2, 128), order="F")
    assert rel(gm.response, Gm) < 1e-9
    assert rel(gm.spectrum[0, 0], ml["spam_S"]) < 1e-9
    zt = IDData(dataset("arx1")[:, 0], None, 0.1)
    assert rel(spa(zt).spectrum[0, 0], ml["spa_ts_S"]) < 1e-10


def test_etfe_matches_matlab(ml):
    z = _z()
    assert rel(etfe(z).response[0, 0], _c(ml["etfe_G"])) < 1e-10
    assert rel(etfe(z, 30).response[0, 0], _c(ml["etfe30_G"])) < 1e-10
    g = etfe(z, 40, 64)
    assert rel(g.frequency, ml["etfe40_w"]) < 1e-14 and rel(g.response[0, 0], _c(ml["etfe40_G"])) < 1e-10
    zt = IDData(dataset("arx1")[:, 0], None, 0.1)
    assert rel(etfe(zt, 30).spectrum[0, 0], ml["etfe_ts_S"]) < 1e-10
    # w ile (bizim eklentimiz): yumuşatmasız tanım = DTFT oranı
    w = np.array([0.3, 1.7])
    gw = etfe(z, w=w)
    y, u = z.y[:, 0], z.u[:, 0]
    E = np.exp(-1j * np.outer(w * 0.1, np.arange(len(y))))
    assert np.allclose(gw.response[0, 0], (E @ y) / (E @ u))


def test_covf_definition():
    z = _z()
    R = covf(z, 5)
    y, u = z.y[:, 0], z.u[:, 0]
    assert R[2, 0, 1] == pytest.approx(np.dot(y[:-2], u[2:]) / 600)  # E y(t) u(t+τ)


# ============================================================================================ impulseest
def test_impulseest_matches_matlab(ml):
    zf = IDData(dataset("fir1")[:, 0], dataset("fir1")[:, 1], 1.0)
    g = impulseest(zf, 40, 0, kernel="TC", ar_order=0)
    assert rel(g.B, ml["imp_tc"]) < 1e-4
    g = impulseest(zf, 40, 0, kernel="DC", ar_order=0)
    assert rel(g.B, ml["imp_dc"]) < 1e-3
    g = impulseest(zf, 40)
    assert rel(g.B, ml["imp_def"]) < 1e-3
    g = impulseest(zf)
    assert len(g.B) == len(ml["imp_auto"])
    # hiperparametre en iyileyicisi farklı (MATLAB fmincon / burada L-BFGS-B + Nelder–Mead): tepe değerin 2e-3'ü
    assert rel(g.B, ml["imp_auto"]) < 2e-3


def test_impulseest_noiseless_fir_exact():
    rng = np.random.default_rng(2)
    u = rng.standard_normal(400)
    h = 0.8 ** np.arange(15) * np.cos(0.5 * np.arange(15))
    y = signal.lfilter(h, [1.0], u)
    g = impulseest(IDData(y, u, 1.0), 15, 0, kernel="none", ar_order=0)
    assert np.allclose(g.B, h, atol=1e-10)


# ============================================================================================ yapı seçimi
def test_delayest_matches_matlab(ml):
    assert delayest(_z()) == ml["dly_arx1"]
    D = dataset("oe1")
    assert delayest(IDData(D[:, 0], D[:, 1], 1.0)) == ml["dly_oe1"]
    assert np.array_equal(np.atleast_1d(delayest(_zm())), np.atleast_1d(ml["dly_miso"]))


def test_arxstruc_selstruc_match_matlab(ml):
    z = _z()
    NN = struc(range(1, 4), range(1, 4), range(1, 3))
    assert np.array_equal(NN, np.asarray(ml["NN"], int))
    V = arxstruc(z[:300], z[300:], NN)
    assert rel(V, ml["V"]) < 1e-10
    assert rel(arxstruc(z, z, NN), ml["V_same"]) < 1e-10
    for c, key in (("aic", "nn_aic"), ("mdl", "nn_mdl"), (0, "nn_0")):
        nn, _ = selstruc(V, c)
        assert np.array_equal(nn, np.asarray(ml[key], int)), c
