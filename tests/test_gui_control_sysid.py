"""Kontrol Tasarımı → Sistem Tanımlama sekmesi (ekransız).

- İş katmanı (`sysid_jobs`): ön işleme zinciri ve tahmin / değerlendirme sonuçları doğrudan sysid API'siyle birebir.
- Sekme: CSV / MAT içe aktarma, kanal tahmini, Ts, aralık, ARX / OE tahmini arka planda; model tablosu (uyum %,
  FPE, AIC) ve karşılaştırma grafiği verisi API ile birebir; frekans ve artık grafikleri; çalışma alanına ekleme.
- Arka plan işi: iptal süreci gerçekten durdurur, arayüz olay döngüsü iş sürerken döner.
- Geçersiz seçimde Türkçe ileti; ölçülmüş paletin her çifti ayırt edilebilir.

Ayarlar geçici dosyada (SPT_SETTINGS_FILE); kullanıcının ayarlarına dokunulmaz.
"""
from __future__ import annotations

import math
import os
import time

import numpy as np
import pytest
from scipy import signal

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QTimer  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

from spt.control import sysid as si  # noqa: E402
from spt.gui.control import sysid_jobs as J  # noqa: E402

TS = 0.01
B_TRUE = [0.0, 0.5, 0.3]
A_TRUE = [1.0, -1.2, 0.5]


# ------------------------------------------------------------------------------------------- veri
def _system_data(n=1200, seed=7, noise=0.05):
    """İkinci derece ayrık sistem (OE yapısı: ölçüm gürültüsü beyaz) — u: ±1 rastgele ikili dizi."""
    rng = np.random.default_rng(seed)
    u = np.sign(rng.standard_normal(n))
    u[u == 0] = 1.0
    y = signal.lfilter(B_TRUE, A_TRUE, u) + noise * rng.standard_normal(n)
    return u, y


def _write_csv(path, u, y, *, sep=",", decimal=".", header=True, time_col=True):
    with open(path, "w", encoding="utf-8") as f:
        if header:
            f.write(sep.join((["zaman"] if time_col else []) + ["giris_pwm", "cikis_aci"]) + "\n")
        for k in range(len(u)):
            cells = ([f"{k * TS:.6f}"] if time_col else []) + [f"{u[k]:.17g}", f"{y[k]:.17g}"]
            if decimal != ".":
                cells = [c.replace(".", decimal) for c in cells]
            f.write(sep.join(cells) + "\n")


# ------------------------------------------------------------------------------------------- Qt
@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def theme(app):
    from spt.gui.theme import ThemeManager

    t = ThemeManager(app)
    t.apply()
    return t


@pytest.fixture(scope="module")
def shared_tab(app, theme, tmp_path_factory):
    """Tek sekme (yardımcı hesap süreci bir kez açılır: scipy içe aktarımı yavaş); her test öncesi sıfırlanır."""
    old = os.environ.get("SPT_SETTINGS_FILE")
    os.environ["SPT_SETTINGS_FILE"] = str(tmp_path_factory.mktemp("ayar") / "settings.ini")
    from spt.gui.control.sysid_tab import SysIdTab

    w = SysIdTab(theme)
    w.resize(1326, 600)
    w.show()
    pump(app, 0.05)
    yield w
    w.shutdown()
    w.close()
    w.deleteLater()
    pump(app, 0.05)
    if old is None:
        os.environ.pop("SPT_SETTINGS_FILE", None)
    else:
        os.environ["SPT_SETTINGS_FILE"] = old


@pytest.fixture
def tab(app, shared_tab):
    w = shared_tab
    w.cancel()
    w.remove_models([e.name for e in w.entries])
    w.workspace = None
    w._files.clear()
    w.set_sources([])
    w.context = None
    w.redraw_results()
    w.set_prep(detrend=0, missing="error", filter_kind=None, zero_phase=False, P=1, Q=1, fraction=0.5)
    w.set_method("arx", na=2, nb=2, nk=1)
    w._set_status("")
    pump(app, 0.02)
    assert w.table is None and w.entries == []
    return w


def pump(app, seconds):
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        app.processEvents()
        time.sleep(0.005)


def wait_idle(app, w, timeout=90.0):
    end = time.perf_counter() + timeout
    while w.busy and time.perf_counter() < end:
        app.processEvents()
        time.sleep(0.005)
    app.processEvents()
    assert not w.busy, "arka plan işi zamanında bitmedi"


def _estimate(app, w, method, **opts):
    w.set_method(method, **opts)
    assert w.estimate(), w.status_text
    wait_idle(app, w)
    assert w.status_state == "good", w.status_text


# =========================================================================================== iş katmanı
def test_build_data_matches_api_chain():
    u, y = _system_data()
    tab = J.SourceTable("k", "deney", ["u", "y"], ["%", "°"], np.column_stack([u, y]), TS, 0.0)
    prep = J.Prep(t0=1.0, t1=10.0, detrend=1, filter_kind="low", filter_order=3, filter_f1=20.0, P=1, Q=2,
                  fraction=0.6)
    work, est, val = J.build_data(tab, [0], [1], TS, prep)
    # aynı zincir doğrudan API ile: aralık (t0 ≤ t ≤ t1) → detrend → idfilt → resample → split
    k0, k1 = int(round(1.0 / TS)), int(round(10.0 / TS)) + 1
    d = si.iddata(y[k0:k1], u[k0:k1], TS, tstart=k0 * TS)
    d = si.resample(si.idfilt(si.detrend(d, 1), order=3, band=20.0 / (0.5 / TS), kind="low"), 1, 2)
    e2, v2 = si.split(d, 0.6)
    for a, b in ((work, d), (est, e2), (val, v2)):
        np.testing.assert_array_equal(a.y, b.y)
        np.testing.assert_array_equal(a.u, b.u)
        assert a.Ts == b.Ts and a.tstart == b.tstart
    assert work.Ts == pytest.approx(2 * TS)
    assert est.input_names == ("u",) and est.output_names == ("y",) and est.output_units == ("°",)


def test_run_job_equals_api_and_recovers_system():
    u, y = _system_data()
    data = si.iddata(y, u, TS)
    data = si.detrend(data, 0)
    est, val = si.split(data, 0.5)
    res = J.run_job({"est": est, "val": val, "work": data,
                     "new": {"method": "oe", "opts": {"nb": [2], "nf": [2], "nk": [1]}, "name": "oe221"},
                     "models": [("arx221", si.arx(est, [2, 2, 1]))]})
    name, m = res["new"]
    ref = si.oe(est, [2, 2, 1])
    np.testing.assert_array_equal(m.getpvec(), ref.getpvec())
    # bilinen sistem (doğru giriş / çıkış bağlantısı): OE verisinde parametreler gerçeğe yakın
    np.testing.assert_allclose(m.B, B_TRUE, atol=0.03)
    np.testing.assert_allclose(m.F, A_TRUE, atol=0.03)
    for nm, model in (("oe221", m), ("arx221", si.arx(est, [2, 2, 1]))):
        ev = res["evals"][nm]
        assert not ev.errors
        np.testing.assert_array_equal(ev.fit_est, si.compare(est, model).fit[0])
        cv = si.compare(val, model)
        np.testing.assert_array_equal(ev.fit_val, cv.fit[0])
        np.testing.assert_array_equal(ev.yhat_val, cv.yhat[0].y)
        assert ev.fpe == model.report.fit.FPE and ev.aic == model.report.fit.AIC
        r = si.resid(val, model, max_lag=25)
        np.testing.assert_array_equal(ev.resid.auto, r.auto)
        np.testing.assert_array_equal(ev.resid.cross, r.cross)
        np.testing.assert_array_equal(ev.H, model.freqresp(res["w"]))
    assert res["evals"]["oe221"].fit_val[0] > 95.0
    w = res["w"]
    assert w[-1] == pytest.approx(np.pi / TS) and np.all(np.diff(w) > 0)


def test_estimate_errors_are_turkish():
    u, y = _system_data(200)
    tab = J.SourceTable("k", "deney", ["u", "y", "y2"], ["", "", ""], np.column_stack([u, y, y]), TS, 0.0)
    from spt.control.errors import ControlError

    cases = [
        (dict(inputs=[0], outputs=[]), "En az bir çıkış"),
        (dict(inputs=[], outputs=[1]), "En az bir giriş"),
        (dict(inputs=[0], outputs=[0]), "hem giriş hem çıkış"),
    ]
    for kw, text in cases:
        with pytest.raises(ControlError, match=text):
            J.build_data(tab, kw["inputs"], kw["outputs"], TS, J.Prep())
    with pytest.raises(ControlError, match="Ts"):
        J.build_data(tab, [0], [1], 0.0, J.Prep())
    with pytest.raises(ControlError, match="en az 8"):
        J.build_data(tab, [0], [1], TS, J.Prep(t0=0.0, t1=0.03))
    with pytest.raises(ControlError, match="Nyquist"):
        J.build_data(tab, [0], [1], TS, J.Prep(filter_kind="low", filter_f1=80.0))
    work, est, val = J.build_data(tab, [0], [1, 2], TS, J.Prep())
    with pytest.raises(ControlError, match="tek çıkışlı"):
        J.estimate("oe", est, {"nb": [2], "nf": [2], "nk": [1]})
    with pytest.raises(ControlError, match="giriş başına"):
        J.estimate("arx", est, {"na": 2, "nb": [2, 2], "nk": [1]})
    with pytest.raises(ControlError, match="Geçersiz süreç modeli"):
        J.estimate("procest", J.build_data(tab, [0], [1], TS, J.Prep())[1], {"type": "P2X"})


# =========================================================================================== sekme
def test_csv_import_arx_oe_table_and_compare_match_api(app, tab, tmp_path):
    u, y = _system_data()
    path = tmp_path / "deney.csv"
    _write_csv(path, u, y)
    assert tab.open_file(path), tab.status_text
    t = tab.table
    assert t.names == ["zaman", "giris_pwm", "cikis_aci"] and t.time_column == 0
    assert t.Ts == pytest.approx(TS, rel=1e-9)
    # kanal tahmini: adlardan giriş / çıkış; zaman sütunu seçilemez
    assert tab.inputs == [1] and tab.outputs == [2]
    assert tab.inputs_list.count() == 2
    assert tab.ts_auto.isChecked() and tab.current_Ts() == t.Ts
    tab.set_prep(detrend=0, fraction=0.5)

    _estimate(app, tab, "arx", na=2, nb=2, nk=1)
    _estimate(app, tab, "oe", nb=2, nf=2, nk=1)
    assert [e.name for e in tab.entries] == ["arx221", "oe221"]

    work, est, val = J.build_data(t, tab.inputs, tab.outputs, tab.current_Ts(), tab.prep())
    np.testing.assert_array_equal(tab.context.val.y, val.y)
    models = {"arx221": si.arx(est, [2, 2, 1]), "oe221": si.oe(est, [2, 2, 1])}
    rows = {tab.model_table.item(r, 1).text(): r for r in range(tab.model_table.rowCount())}
    for name, ref in models.items():
        e = tab.entry(name)
        np.testing.assert_array_equal(e.model.getpvec(), ref.getpvec())
        fit_e = si.compare(est, ref).fit[0]
        cv = si.compare(val, ref)
        r = rows[name]
        assert tab.model_table.item(r, 5).text() == f"{fit_e[0]:.2f}"
        assert tab.model_table.item(r, 6).text() == f"{cv.fit[0][0]:.2f}"
        assert tab.model_table.item(r, 7).text() == f"{ref.report.fit.FPE:.4g}"
        assert tab.model_table.item(r, 8).text() == f"{ref.report.fit.AIC:.4g}"
        assert tab.model_table.item(r, 3).text().startswith("n")
        # karşılaştırma grafiğinin verisi = compare çıktısı (zaman ekseni doğrulama verisinin örnekleme anları)
        tr = next(tr for tr in tab._cmp_traces if tr.name.startswith(name))
        np.testing.assert_array_equal(tr.item.yData, cv.yhat[0].y[:, 0])
        np.testing.assert_array_equal(tr.item.xData, val.time)
        assert f"{name}: %{cv.fit[0][0]:.2f}" in tab.legend_texts("compare")
    measured = tab._cmp_traces[0]
    np.testing.assert_array_equal(measured.item.yData, val.y[:, 0])
    assert tab.legend_texts("compare")[0] == "Ölçülen cikis_aci"
    # artık ve frekans grafikleri her model için
    assert tab.legend_texts("auto")[:2] == ["arx221", "oe221"]
    mag = {tr.name: tr for tr in tab._freq_traces["mag"]}
    assert set(mag) >= {"arx221", "oe221"}
    w = tab.context.w
    np.testing.assert_allclose(mag["oe221"].item.xData, w)
    np.testing.assert_allclose(mag["oe221"].item.yData, 20 * np.log10(np.abs(models["oe221"].freqresp(w)[0, 0])))
    if J.method_available("spa"):
        assert "spa (çalışma verisi)" in mag
    # tahmin / doğrulama sınırı: doğrulama verisinin ilk örneği
    assert tab._split_time() == pytest.approx(val.tstart)
    # hiçbir renk iki seride yok (veri kanalları, modeller, spa)
    from spt.gui.control.sysid_tab import spa_color

    colors = [tr.color for tr in tab._data_curves] + [e.color for e in tab.entries] + [spa_color()]
    assert len(colors) == len(set(colors))


def test_add_to_workspace_and_signal(app, tab, tmp_path):
    from spt.gui.control.workspace import ModelWorkspace

    u, y = _system_data(600)
    path = tmp_path / "d.csv"
    _write_csv(path, u, y)
    tab.open_file(path)
    _estimate(app, tab, "arx", na=2, nb=2, nk=1)
    got = []
    tab.models_ready.connect(lambda items: got.extend(items))
    ws = ModelWorkspace()
    tab.workspace = ws
    names = tab.add_to_workspace()
    assert names == ["arx221"] and ws.names() == ["arx221"]
    assert ws.get("arx221") is tab.entry("arx221").model
    ref = tab.entry("arx221").model.to_lti()
    lti = ws.lti("arx221")
    np.testing.assert_allclose(lti.num[0][0], ref.num[0][0])
    np.testing.assert_allclose(lti.den[0][0], ref.den[0][0])
    assert got and got[0][0] == "arx221"
    # ikinci kez: çalışma alanı benzersiz ad seçer
    assert tab.add_to_workspace(["arx221"]) == ["arx2212"]
    if J.method_available("spa"):
        _estimate(app, tab, "spa", M=0)
        assert tab.add_to_workspace(["spa"]) == []
        assert "parametrik olmayan" in tab.status_text and tab.status_state == "critical"


def test_invalid_selection_messages_in_tab(app, tab, tmp_path):
    assert not tab.estimate()
    assert "veri kaynağı" in tab.status_text and tab.status_state == "critical"
    u, y = _system_data(400)
    path = tmp_path / "d.csv"
    _write_csv(path, u, y)
    tab.open_file(path)
    tab.select_channels(inputs=[1], outputs=[])
    assert not tab.estimate() and "En az bir çıkış kanalı seçin" in tab.status_text
    tab.select_channels(inputs=["giris_pwm"], outputs=["cikis_aci"])
    # aynı kanal ikinci listede işaretlenince ilkinden kalkar
    it = tab.outputs_list.item(0)  # giris_pwm
    from PyQt6.QtCore import Qt
    it.setCheckState(Qt.CheckState.Checked)
    assert tab.inputs == [] and tab.outputs == [1, 2]
    tab.select_channels(inputs=[1], outputs=[2])
    tab.set_range(0.0, 0.03)
    assert not tab.estimate() and "en az 8 gerekli" in tab.status_text
    tab._full_range()
    tab.set_prep(filter_kind="bandpass", filter_f1=10.0, filter_f2=5.0)
    assert not tab.estimate() and "alt kenar" in tab.status_text
    tab.set_prep(filter_kind=None)
    tab.set_method("arx", nb=[2, 3])
    assert not tab.estimate() and "giriş başına" in tab.status_text
    # iç API adları iletiye sızmaz
    for word in ("ControlError", "IDData", "build_data", "Traceback"):
        assert word not in tab.status_text
    # tek çıkışlı yöntem + iki çıkış: iş başlamadan Türkçe ileti
    path2 = tmp_path / "iki_cikis.csv"
    with open(path2, "w", encoding="utf-8") as f:
        f.write("t,u,y1,y2\n")
        for k in range(len(u)):
            f.write(f"{k * TS:.6f},{u[k]:.17g},{y[k]:.17g},{-y[k]:.17g}\n")
    tab.open_file(path2)
    tab.select_channels(inputs=["u"], outputs=["y1", "y2"])
    tab.set_method("oe", nb=2, nf=2, nk=1)
    assert not tab.estimate() and "tek çıkışlı veri ister" in tab.status_text
    # hesabın içinden gelen hata (veri dereceye göre çok kısa) arka plandan Türkçe döner
    tab.select_channels(inputs=["u"], outputs=["y1"])
    tab.set_range(0.0, 0.2)
    tab.set_method("arx", na=20, nb=2, nk=1)
    assert tab.estimate()
    wait_idle(app, tab)
    assert tab.status_state == "critical" and "çok kısa" in tab.status_text, tab.status_text
    assert tab.entries == []


def test_csv_semicolon_decimal_comma_manual_ts_and_mat(app, tab, tmp_path):
    from scipy import io as sio

    u, y = _system_data(300)
    path = tmp_path / "virgul.txt"
    _write_csv(path, u, y, sep=";", decimal=",", header=False, time_col=False)
    assert tab.open_file(path), tab.status_text
    assert tab.table.names == ["col0", "col1"] and tab.table.Ts is None
    np.testing.assert_allclose(tab.table.X[:, 0], u)
    assert not tab.ts_auto.isChecked()
    assert not tab.estimate() and "Ts girin" in tab.status_text
    tab.set_Ts(0.05)
    assert tab.t1_spin.value() == pytest.approx(0.05 * 299)
    _estimate(app, tab, "arx", na=2, nb=2, nk=1)
    assert tab.entries[-1].model.Ts == 0.05

    mpath = tmp_path / "olcum.mat"
    sio.savemat(mpath, {"u": u, "y": y.reshape(-1, 1), "Ts": 0.02, "not_used": np.arange(3.0)})
    assert tab.open_file(mpath), tab.status_text
    t = tab.table
    assert set(t.names) == {"u", "y"} and t.Ts == 0.02
    assert tab.inputs == [t.names.index("u")] and tab.outputs == [t.names.index("y")]
    assert tab.source.count() == 2  # iki dosya da kaynak listesinde
    assert not tab.open_file(tmp_path / "yok.csv") and "Dosya açılamadı" in tab.status_text


def test_snapshot_source_with_gap(app, tab):
    from spt.core.local_ring import StaticRing
    from spt.core.snapshot import Snapshot
    from spt.protocol.format import SAMPLE_TYPES_BY_NAME, ChannelInfo, Descriptor, Layout

    u, y = _system_data(1000)
    data = np.column_stack([u, y])
    desc = Descriptor(1, 1, 1 / TS, SAMPLE_TYPES_BY_NAME["float64"], Layout.INTERLEAVED, "kayit",
                      (ChannelInfo("motor_cmd", "%"), ChannelInfo("aci", "°")))
    snap = Snapshot(3, "kayit", "test", desc, 0.0, StaticRing(data, 0, [(500, 510)]))
    tab.set_sources([("ws3", "Çalışma alanı · kayit", snap.handle())])
    assert tab.table is not None and tab.table.kind == "snapshot"
    assert tab.inputs == [0] and tab.outputs == [1] and tab.current_Ts() == pytest.approx(TS)
    assert "10 eksik örnek" in tab.source_info.text()
    tab.set_method("arx", na=2, nb=2, nk=1)
    assert not tab.estimate() and "eksik (NaN)" in tab.status_text
    tab.set_prep(missing="linear")
    _estimate(app, tab, "arx", na=2, nb=2, nk=1)
    # kaynak listesi yenilenince seçili kaynak korunur
    tab.set_sources([("ws3", "Çalışma alanı · kayit", snap.handle()), ("ws4", "başka", snap.handle())])
    assert tab.source.currentData() == "ws3" and tab.outputs == [1]


def test_runner_cancel_stops_process_and_ui_stays_responsive(app):
    from spt.gui.control.sysid_runner import JobRunner

    r = JobRunner()
    done, failed = [], []
    r.finished.connect(lambda k, res: done.append((k, res)))
    r.failed.connect(lambda k, msg: failed.append((k, msg)))
    r.start("uyku", time.sleep, 30)
    proc = r.process
    assert r.busy and proc is not None
    ticks = []
    timer = QTimer()
    timer.timeout.connect(lambda: ticks.append(1))
    timer.start(20)
    pump(app, 1.0)
    timer.stop()
    assert len(ticks) >= 10            # iş sürerken olay döngüsü dönüyor (arayüz donmuyor)
    assert proc.is_alive() and r.busy
    t0 = time.perf_counter()
    assert r.cancel()
    assert not proc.is_alive() and time.perf_counter() - t0 < 3.0
    assert not r.busy and r.process is None
    pump(app, 0.2)
    assert done == [] and failed == []
    # sonraki iş yeni süreçte çalışır
    r.start("hipot", math.hypot, 3.0, 4.0)
    end = time.perf_counter() + 60
    while r.busy and time.perf_counter() < end:
        app.processEvents()
        time.sleep(0.005)
    assert done == [("hipot", 5.0)]
    # hata iletisi Türkçe ve tür bilgisiyle döner
    r.start("hata", J.estimate, "yok", None, {})
    end = time.perf_counter() + 60
    while r.busy and time.perf_counter() < end:
        app.processEvents()
        time.sleep(0.005)
    assert failed == [("hata", "Bilinmeyen yöntem")]
    r.shutdown()


def test_tab_cancel_keeps_previous_state(app, tab, tmp_path):
    u, y = _system_data(800)
    path = tmp_path / "d.csv"
    _write_csv(path, u, y)
    tab.open_file(path)
    _estimate(app, tab, "arx", na=2, nb=2, nk=1)
    before = tab.model_table.rowCount()
    tab.set_method("bj", nb=2, nc=2, nd=2, nf=2, nk=1)
    assert tab.estimate() and tab.busy
    assert tab.cancel_button.isEnabled() and not tab.estimate_button.isEnabled()
    assert tab.cancel()
    assert not tab.busy and tab.status_state == "warning" and "iptal" in tab.status_text
    pump(app, 0.5)  # geç sonuç gelmez
    assert tab.model_table.rowCount() == before and [e.name for e in tab.entries] == ["arx221"]
    assert tab.estimate_button.isEnabled()
    # iptalden sonra yeni tahmin çalışır
    _estimate(app, tab, "oe", nb=2, nf=2, nk=1)
    assert [e.name for e in tab.entries] == ["arx221", "oe221"]


def test_mimo_arx_and_model_management(app, tab, tmp_path):
    rng = np.random.default_rng(3)
    n = 1500
    U = np.sign(rng.standard_normal((n, 2)))
    y1 = signal.lfilter([0, 0.4], [1, -0.7], U[:, 0]) + signal.lfilter([0, 0.2], [1, -0.5], U[:, 1])
    y2 = signal.lfilter([0, 0.3], [1, -0.6], U[:, 1])
    Y = np.column_stack([y1, y2]) + 0.01 * rng.standard_normal((n, 2))
    path = tmp_path / "mimo.csv"
    with open(path, "w", encoding="utf-8") as f:
        f.write("t,u1,u2,y1,y2\n")
        for k in range(n):
            f.write(f"{k * TS:.6f},{U[k, 0]:.17g},{U[k, 1]:.17g},{Y[k, 0]:.17g},{Y[k, 1]:.17g}\n")
    tab.open_file(path)
    tab.select_channels(inputs=["u1", "u2"], outputs=["y1", "y2"])
    _estimate(app, tab, "arx", na=1, nb=1, nk=1)
    e = tab.entries[-1]
    assert e.name == "arx111" and e.ev.fit_val.shape == (2,)
    work, est, val = J.build_data(tab.table, tab.inputs, tab.outputs, tab.current_Ts(), tab.prep())
    cv = si.compare(val, si.arx(est, [1, 1, 1, 1, 1]))
    assert tab.model_table.item(0, 6).text() == " / ".join(f"{v:.2f}" for v in cv.fit[0])
    tab.cmp_output.setCurrentIndex(1)
    tr = next(tr for tr in tab._cmp_traces if tr.name.startswith("arx"))
    np.testing.assert_array_equal(tr.item.yData, cv.yhat[0].y[:, 1])
    assert tab.legend_texts("compare")[0] == "Ölçülen y2"
    # gizle / sil: renk serbest kalır
    color = e.color
    from PyQt6.QtCore import Qt

    tab.model_table.item(0, 0).setCheckState(Qt.CheckState.Unchecked)
    assert not e.visible and all(not tr.name.startswith("arx") for tr in tab._cmp_traces)
    tab.remove_models(["arx111"])
    assert tab.entries == [] and tab._free_color() == color


def test_suggest_orders_and_other_methods(app, tab, tmp_path):
    u, y = _system_data(600)
    path = tmp_path / "d.csv"
    _write_csv(path, u, y)
    tab.open_file(path)
    assert tab.suggest_orders()
    wait_idle(app, tab)
    assert tab.status_state == "good", tab.status_text
    work, est, val = J.build_data(tab.table, tab.inputs, tab.outputs, tab.current_Ts(), tab.prep())
    assert J.suggest_arx_orders(est, val) == (tab.method_opts("arx")["na"], tab.method_opts("arx")["nb"][0],
                                             tab.method_opts("arx")["nk"][0])
    from spt.control.sysid.nonparam import arxstruc, selstruc

    NN = np.array([[a, b, k] for a in range(1, 11) for b in range(1, 11) for k in range(1, 6)])
    nn, _ = selstruc(arxstruc(est, val, NN), "aic")
    assert tab.status_text.startswith(f"Önerilen ARX derecesi: na={nn[0]}, nb={nn[1]}, nk={nn[2]}")
    # süreç modeli formu: tür kutulardan
    tab.set_method("procest", type="P2DU")
    assert tab.method_opts() == {"type": "P2DU"}
    if J.method_available("procest"):
        _estimate(app, tab, "procest", type="P1D")
        assert tab.entries[-1].name == "P1D" and tab.entries[-1].model.Ts == 0.0
        assert tab.model_table.item(len(tab.entries) - 1, 4).text() == "sürekli"


# =========================================================================================== MATLAB
def _lcg_data(n=1000):
    """MATLAB'de aynen üretilebilen deterministik veri: çoklu sinüs giriş, tamsayı LCG (Park–Miller) gürültü."""
    k = np.arange(n, dtype=float)
    u = np.sin(0.37 * k) + 0.6 * np.sin(1.31 * k + 0.4) + 0.3 * np.sin(2.9 * k)
    x = 12345.0
    e = np.empty(n)
    for i in range(n):
        x = (16807.0 * x) % 2147483647.0
        e[i] = (x / 2147483647.0 - 0.5) * 0.1
    y = signal.lfilter(B_TRUE, A_TRUE, u) + e
    return u, y


MATLAB_E2E = r"""
n = 1000; k = (0:n-1)';
u = sin(0.37*k) + 0.6*sin(1.31*k + 0.4) + 0.3*sin(2.9*k);
x = 12345; e = zeros(n, 1);
for i = 1:n
    x = mod(16807*x, 2147483647);
    e(i) = (x/2147483647 - 0.5)*0.1;
end
y = filter([0 0.5 0.3], [1 -1.2 0.5], u) + e;
data = detrend(iddata(y, u, 0.01), 0);
ze = data(1:500); zv = data(501:1000);
m1 = arx(ze, [2 2 1]);
opt = oeOptions('SearchMethod', 'lm');
opt.SearchOptions.MaxIterations = 500; opt.SearchOptions.Tolerance = 1e-12;
m2 = oe(ze, [2 2 1], opt);
[yv1, fv1] = compare(zv, m1); [~, fe1] = compare(ze, m1);
[yv2, fv2] = compare(zv, m2); [~, fe2] = compare(ze, m2);
out.arx_a = m1.A; out.arx_b = m1.B; out.oe_b = m2.B; out.oe_f = m2.F;
out.arx_fit_val = fv1; out.arx_fit_est = fe1; out.oe_fit_val = fv2; out.oe_fit_est = fe2;
out.arx_yhat = yv1.y; out.oe_yhat = yv2.y;
out.arx_fpe = m1.Report.Fit.FPE; out.arx_aic = m1.Report.Fit.AIC;
out.oe_fpe = m2.Report.Fit.FPE; out.oe_aic = m2.Report.Fit.AIC;
"""


def test_tab_end_to_end_matches_matlab(app, tab, tmp_path):
    """Sekmenin tüm akışı (CSV → elle Ts → ortalama çıkar → %50 bölme → ARX / OE → tablo ve karşılaştırma) MATLAB
    System Identification Toolbox (R2025b) ile: ARX göreli 1e-8, OE (yinelemeli) 1e-4 + kayıp MATLAB'inkinden
    büyük değil."""
    import sys

    sys.path.insert(0, str(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from tools.matlab_oracle import OracleUnavailable, fixture

    try:
        ref = fixture("sysid_tab_end_to_end", MATLAB_E2E)
    except OracleUnavailable as exc:
        pytest.skip(str(exc))
    u, y = _lcg_data()
    path = tmp_path / "lcg.csv"
    with open(path, "w", encoding="utf-8") as f:
        f.write("u,y\n")
        for a, b in zip(u, y):
            f.write(f"{a:.17g},{b:.17g}\n")
    assert tab.open_file(path)
    assert tab.inputs == [0] and tab.outputs == [1]
    tab.set_Ts(0.01)
    tab.set_prep(detrend=0, fraction=0.5)
    _estimate(app, tab, "arx", na=2, nb=2, nk=1)
    _estimate(app, tab, "oe", nb=2, nf=2, nk=1)
    a, o = tab.entry("arx221"), tab.entry("oe221")
    np.testing.assert_allclose(a.model.A, ref["arx_a"], rtol=1e-8)
    np.testing.assert_allclose(a.model.B, ref["arx_b"], rtol=1e-8, atol=1e-14)
    # OE: aynı ölçüt (sıfır başlangıçlı çıkış hatası kareleri) — MATLAB 'lm' sıkı seçeneklerle de optimumdan
    # ~1e-7 göreli kayıp önce duruyor (ölçüldü); parametreler 1e-4 içinde, bizim kaybımız MATLAB'inkinden büyük değil
    np.testing.assert_allclose(o.model.B, ref["oe_b"], rtol=1e-4, atol=1e-12)
    np.testing.assert_allclose(o.model.F, ref["oe_f"], rtol=1e-4)
    est = tab.context.est

    def loss(B, F):
        r = est.y[:, 0] - signal.lfilter(B, F, est.u[:, 0])
        return float(r @ r) / len(r)
    assert loss(o.model.B, o.model.F) <= loss(ref["oe_b"], ref["oe_f"]) * (1 + 1e-12)
    for e, key, rtol in ((a, "arx", 1e-8), (o, "oe", 1e-4)):
        np.testing.assert_allclose(e.ev.fit_val, ref[f"{key}_fit_val"], rtol=rtol)
        np.testing.assert_allclose(e.ev.fit_est, ref[f"{key}_fit_est"], rtol=rtol)
        np.testing.assert_allclose(e.ev.fpe, ref[f"{key}_fpe"], rtol=max(rtol, 1e-7))
        np.testing.assert_allclose(e.ev.aic, ref[f"{key}_aic"], rtol=max(rtol, 1e-7))
        tr = next(tr for tr in tab._cmp_traces if tr.name.startswith(e.name))
        yh = np.asarray(ref[f"{key}_yhat"], float)
        np.testing.assert_allclose(tr.item.yData, yh, rtol=0, atol=max(rtol, 1e-7) * np.max(np.abs(yh)))


@pytest.mark.parametrize("name, surface_hex, normal, colorblind, min_contrast", [
    ("dark", "#121110", 10.0, 6.4, 4.9),
    ("light", "#fffdfa", 11.5, 7.4, 3.09),
])
def test_palette_measured_distinct(name, surface_hex, normal, colorblind, min_contrast, monkeypatch):
    """Modül belgesindeki ölçüm (tema başına): 16 rengin her çifti OKLab ΔE ≥ normal / renk körlüğü eşiği."""
    from spt.gui import theme
    from spt.gui.control.sysid_tab import CHANNEL_COLORS, MODEL_COLORS, spa_color

    monkeypatch.setattr(theme, "_active", theme.THEMES[name])
    colors = list(MODEL_COLORS) + list(CHANNEL_COLORS) + [spa_color()]
    assert len(set(colors)) == 16
    prot = np.array([[0.152286, 1.052583, -0.204868], [0.114503, 0.786281, 0.099216],
                     [-0.003882, -0.048116, 1.051998]])
    deut = np.array([[0.367322, 0.860646, -0.227968], [0.280085, 0.672501, 0.047413],
                     [-0.011820, 0.042940, 0.968881]])
    m1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929], [0.2119034982, 0.6806995451, 0.1073969566],
                   [0.0883024619, 0.2817188376, 0.6299787005]])
    m2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468], [1.9779984951, -2.4285922050, 0.4505937099],
                   [0.0259040371, 0.7827717662, -0.8086757660]])

    def lin(h):
        c = np.array([int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)])
        return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)

    def lab(h, M=None):
        c = lin(h) if M is None else np.clip(M @ lin(h), 0, 1)
        return 100 * (m2 @ np.cbrt(m1 @ c))

    def lum(h):
        return float(lin(h) @ [0.2126, 0.7152, 0.0722])

    for M, limit in ((None, normal), (prot, colorblind), (deut, colorblind)):
        pts = [lab(c, M) for c in colors]
        dmin = min(np.linalg.norm(a - b) for i, a in enumerate(pts) for b in pts[i + 1:])
        assert dmin >= limit - 0.05
    surface = lum(surface_hex)
    ratios = [(max(lum(c), surface) + 0.05) / (min(lum(c), surface) + 0.05) for c in colors]
    assert min(ratios) >= min_contrast
