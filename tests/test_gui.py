"""Arayüz uçtan uca testleri (ekransız): Filtre Tasarımcısı, işlem zinciri, canlı filtre, Dondur, inceleme,
geri al / yinele, katsayı dışa aktarma ve ayrı pencereler.

Ayarlar geçici bir INI dosyasında tutulur (SPT_SETTINGS_FILE); kullanıcının kayıtlı yerleşimine dokunulmaz.
"""
import os
import time
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import Qt  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

from spt.dsp import filters as F  # noqa: E402
from spt.dsp.convert import physical_column  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "samples_signal_data" / "07_ekg_50Hz_parazitli_uint16_500Hz.sptr"


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def theme(app):
    from spt.gui.theme import ThemeManager

    t = ThemeManager(app)
    t.apply()  # bir kez: stil sayfası her uygulamada tüm pencereleri yeniden biçimlendirir
    return t


@pytest.fixture
def win(app, theme, tmp_path, monkeypatch):
    monkeypatch.setenv("SPT_SETTINGS_FILE", str(tmp_path / "settings.ini"))
    from spt.gui.main_window import MainWindow

    w = MainWindow(theme)
    w.resize(1600, 960)
    w.show()
    pump(app, 0.2)
    yield w
    w.close()
    w.deleteLater()
    pump(app, 0.2)


def pump(app, seconds):
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        app.processEvents()
        time.sleep(0.005)


def wait_for(app, condition, timeout=10.0):
    end = time.perf_counter() + timeout
    while time.perf_counter() < end:
        app.processEvents()
        if condition():
            return True
        time.sleep(0.01)
    return False


def visible_rows(view):
    return {key for key, (lbl, _w) in view._rows.items() if view._form.isRowVisible(lbl)}


# ============================================================================ tasarımcı
def test_designer_hides_irrelevant_fields_and_redesigns(app, win):
    fv = win.filter_view
    fv.set_spec(F.FilterSpec(family="iir", method="butter"))
    assert {"order", "f1"} <= visible_rows(fv) and not {"ripple", "atten", "f2", "transition"} & visible_rows(fv)
    fv.set_spec(F.FilterSpec(family="iir", method="ellip", band="bandpass"))
    assert {"ripple", "atten", "f2"} <= visible_rows(fv)
    fv.set_spec(F.FilterSpec(family="fir", method="kaiser"))
    assert "beta" in visible_rows(fv)
    fv.set_spec(F.FilterSpec(family="special", method="notch"))
    assert {"notch_f0", "notch_q", "harmonics"} <= visible_rows(fv) and "order" not in visible_rows(fv)

    # Kullanıcı bir alanı değiştirince grafik kısa gecikmeyle yeniden çizilir
    fv.set_spec(F.FilterSpec(family="iir", method="butter", order=4, f1=1000.0))
    before = fv._mag_curve.getData()[1].copy()
    fv.f1.setValue(2000.0)
    assert wait_for(app, lambda: fv.design is not None and fv.design.spec.f1 == 2000.0, 2.0)
    after = fv._mag_curve.getData()[1]
    assert not np.array_equal(before, after)
    r = F.analyze(fv.design, np.array([2000.0]))
    assert r.magnitude_db[0] == pytest.approx(-3.0103, abs=1e-3)


def test_invalid_design_shows_error_and_blocks_apply(app, win):
    fv = win.filter_view
    fv.set_spec(F.FilterSpec(family="iir", method="butter", f1=6000.0, fs=10_000.0))
    assert fv.design is None and "Nyquist" in fv.error
    assert not fv.apply_button.isEnabled() and "hatası" in fv.info.text()
    fv.request_apply()
    assert len(win.chain_panel.chain) == 0


def test_auto_order_updates_order_field_and_mask(app, win):
    fv = win.filter_view
    fv.set_spec(F.FilterSpec(family="iir", method="ellip", auto_order=True, f1=1000.0, transition=200.0,
                             ripple_db=0.5, atten_db=60.0))
    assert fv.order.value() == fv.design.order and not fv.order.isEnabled()
    assert "Maske karşılanıyor" in fv.info.text()
    assert fv._mask_items  # maske grafikte çizili


def test_quantized_response_is_overlaid(app, win):
    fv = win.filter_view
    fv.set_spec(F.FilterSpec(family="iir", method="ellip", order=8, ripple_db=0.5, atten_db=60.0))
    fv.set_quant("q15")
    x, y = fv._quant_curve.getData()
    assert x is not None and len(x) > 100
    assert "Q15" in fv.quant_label.text() and not fv.quant_label.isHidden()
    fv.set_quant(None)
    assert fv.quant_label.isHidden()


# ============================================================================ zincir + canlı filtre
def test_apply_filters_live_replay_and_matches_offline(app, win):
    fv = win.filter_view
    win.start_replay(SAMPLE)
    assert wait_for(app, lambda: win.sources.streams and next(iter(win.sources.streams.values())).latest_index() > 600)
    fv.set_spec(F.FilterSpec(family="special", method="notch", notch_f0=50.0, notch_q=20.0, fs=500.0))
    fv.request_apply()
    fv.set_spec(F.FilterSpec(family="iir", method="butter", order=4, f1=40.0, fs=500.0))
    fv.request_apply()
    chain = win.chain_panel.chain
    assert len(chain) == 2 and win.live_filter.active and win.chain_panel.live.isChecked()
    win._show_view(0)
    pump(app, 1.0)
    lv = win.live_view
    handle = next(iter(win.sources.streams.values()))
    key = (handle.key, 0, 0)
    x, y = lv._filt_curves[key].getData()
    assert lv._filt_curves[key].isVisible() and x is not None and len(x) > 10
    assert lv.filter_chip.isVisible() and "2 aşama" in lv.filter_chip.text()

    # Canlı (blok blok) çıktı, aynı verinin tek seferde filtrelenmesiyle aynı olmalı
    win.registry["stream.run"].trigger()  # görüntüyü dondur
    lv.refresh(force=True)
    end = lv.end_index(handle.key, handle)
    start = win.live_filter._tracks[(handle.key, 0)].out.origin
    first, raw = handle.ring.read(start, end)
    live = win.live_filter.read_clean(handle.key, 0, first, end)
    designs, _ = chain.designs(handle.fs)
    x = physical_column(raw[:, 0], handle.desc.sample_type, handle.desc.channels[0])
    assert first == start  # 500 Hz'lik akışta 10 s'lik tampon taşmaz
    offline = F.ChainRunner(designs).process(x)
    assert live is not None and len(live) > 500 and np.max(np.abs(live - offline)) < 1e-9
    win.registry["stream.run"].trigger()

    # Yalnız filtreli gösterim: ham eğri gizlenir
    win.chain_panel.display.setCurrentIndex(1)
    assert not lv._time_curves[key].isVisible() and lv._filt_curves[key].isVisible()
    win.chain_panel.display.setCurrentIndex(0)
    # Canlı uygulama kapatılınca filtre eğrileri kalkar
    win.chain_panel.live.setChecked(False)
    assert not lv._filt_curves[key].isVisible() and not lv.filter_chip.isVisible()
    win.sources.stop()


def test_chain_edit_reorder_toggle_and_save_load(app, win, tmp_path):
    fv, cp = win.filter_view, win.chain_panel
    for spec in (F.FilterSpec(family="special", method="dc_block"),
                 F.FilterSpec(family="iir", method="butter", order=2),
                 F.FilterSpec(family="special", method="moving_avg", length=5)):
        fv.set_spec(spec)
        fv.request_apply()
    assert [s.spec.method for s in cp.chain.stages] == ["dc_block", "butter", "moving_avg"]
    cp.list.setCurrentRow(2)
    cp.up.click()
    assert [s.spec.method for s in cp.chain.stages] == ["dc_block", "moving_avg", "butter"]
    cp.list.item(0).setCheckState(Qt.CheckState.Unchecked)
    assert not cp.chain.stages[0].enabled and win.live_filter.chain.enabled_count == 2

    # Düzenle: aşama tasarımcıya yüklenir, "Sinyale Uygula" o aşamayı günceller
    cp._edit(2)
    assert fv.editing == 2 and fv.spec().method == "butter" and win.views.currentIndex() == 2
    fv.order.setValue(6)
    fv.request_apply()
    assert cp.chain.stages[2].spec.order == 6 and len(cp.chain) == 3 and fv.editing is None

    path = tmp_path / "zincir.sptchain"
    cp.chain.save(path)
    from spt.dsp.chain import ProcessingChain
    assert ProcessingChain.load(path) == cp.chain


def test_undo_redo_covers_designer_and_chain(app, win):
    fv = win.filter_view
    fv.set_spec(F.FilterSpec(family="iir", method="butter", order=4, f1=1000.0), emit=True)
    win._push_history()
    fv.f1.setValue(1500.0)
    win.undo()  # bekleyen değişiklik önce kaydedilir, sonra geri alınır
    assert fv.spec().f1 == 1000.0
    win.redo()
    assert fv.spec().f1 == 1500.0
    fv.request_apply()
    assert len(win.chain_panel.chain) == 1
    win.undo()
    assert len(win.chain_panel.chain) == 0 and win.live_filter.chain.enabled_count == 0
    win.redo()
    assert len(win.chain_panel.chain) == 1
    assert win.registry["edit.undo"].isEnabled()


# ============================================================================ dondur / inceleme
def test_freeze_offline_filter_and_review(app, win):
    win.start_replay(SAMPLE)
    assert wait_for(app, lambda: win.sources.streams and next(iter(win.sources.streams.values())).latest_index() > 1500)
    snaps = win.freeze()
    assert len(snaps) == 1 and snaps[0].samples > 1500
    assert win.workspace_panel.selected() == snaps
    frozen = snaps[0].ring.data.copy()
    pump(app, 0.5)  # canlı akış devam ediyor; kopya değişmemeli
    assert np.array_equal(snaps[0].ring.data, frozen)

    assert win.filter_snapshots(zero_phase=True) == []  # zincir boş: filtre uygulanmaz
    win.filter_view.set_spec(F.FilterSpec(family="special", method="notch", notch_f0=50.0, fs=500.0))
    win.filter_view.request_apply()
    results = win.filter_snapshots(zero_phase=True)
    assert len(results) == 1 and results[0].parent == snaps[0].id
    assert win.views.currentIndex() == 3
    rv = win.review_view
    assert len(rv._streams) == 2 and rv.pages.currentIndex() == 1
    x, y = rv._time_curves[(f"ws{results[0].id}", 0, 0)].getData()
    assert x is not None and len(x) > 100
    win.workspace_panel.list.selectAll()
    win.workspace_panel.remove_selected()
    assert not win.workspace_panel.snapshots and rv.pages.currentIndex() == 0
    win.sources.stop()


def test_freeze_without_streams_is_harmless(app, win):
    assert win.freeze() == []
    assert not win.workspace_panel.snapshots


# ============================================================================ dışa aktarma ve ayrı pencereler
def test_export_dialog_follows_designer(app, win):
    fv = win.filter_view
    fv.set_spec(F.FilterSpec(family="iir", method="ellip", order=6, ripple_db=0.5, atten_db=60.0))
    win.show_export()
    dlg = win._export
    dlg.number.setCurrentIndex(dlg.number.findData("q15"))
    assert "arm_biquad_cascade_df1_init_q15" in dlg.text and "POST_SHIFT" in dlg.text
    assert fv.quant.currentData() == "q15"  # seçilen biçim genlik grafiğine yansır
    fv.order.setValue(4)
    assert wait_for(app, lambda: "NUM_STAGES 2" in dlg.text, 2.0)
    dlg.kind.setCurrentIndex(dlg.kind.findData("csv"))
    assert dlg.text.splitlines()[-3].startswith("b0,b1,b2,a1,a2")
    fv.set_spec(F.FilterSpec(family="special", method="median"))
    assert not dlg.save_button.isEnabled()
    dlg.close()


def test_new_cards_and_chain_panel_can_be_detached(app, win):
    d = win.detacher
    for key in ("filter_mag", "filter_analysis", "filter_pz", "review_time", "dock_chain"):
        d.detach(key)
        assert d.is_detached(key)
    d.attach_all()
    assert not d.detached_keys()
    assert win.filter_view.analysis_card.window() is win


def test_filter_state_persists_across_sessions(app, theme, tmp_path, monkeypatch):
    monkeypatch.setenv("SPT_SETTINGS_FILE", str(tmp_path / "settings.ini"))
    from spt.gui.main_window import MainWindow

    first = MainWindow(theme)
    spec = F.FilterSpec(family="fir", method="remez", band="bandpass", order=120, f1=300.0, f2=900.0)
    first.filter_view.set_spec(spec)
    first.filter_view.request_apply()
    first.chain_panel.display.setCurrentIndex(1)
    first.filter_view.set_log_freq(True)
    first.close()
    first.deleteLater()
    pump(app, 0.1)

    second = MainWindow(theme)
    try:
        assert second.filter_view.spec() == spec
        assert [s.spec for s in second.chain_panel.chain.stages] == [spec]
        assert second.live_filter.chain == second.chain_panel.chain
        assert not second.chain_panel.live.isChecked()  # canlı uygulama her açılışta kapalı başlar
        assert second.chain_panel.display_mode == "filtered" and second.filter_view.log_freq
        assert not second.registry["edit.undo"].isEnabled()  # geri alma geçmişi oturuma özel
    finally:
        second.close()
        second.deleteLater()
        pump(app, 0.1)


# ============================================================================ veri ipuçları ve analiz
def _tip_at(tips, plot, x, y):
    from PyQt6.QtCore import QPointF
    return tips.add_at_scene(plot.getPlotItem().getViewBox().mapViewToScene(QPointF(x, y)))


def test_data_tips_read_exact_samples_and_analysis_matches_numpy(app, win):
    win.start_replay(SAMPLE)
    assert wait_for(app, lambda: win.sources.streams and next(iter(win.sources.streams.values())).latest_index() > 600)
    win.properties_panel.time_window.setCurrentText("1 s")
    win._show_view(0)
    pump(app, 0.5)
    win.registry["stream.run"].trigger()  # görüntüyü dondur: değerler sabit kalsın
    lv = win.live_view
    lv.refresh(force=True)
    handle = next(iter(win.sources.streams.values()))
    x, y = lv._time_curves[(handle.key, 0, 0)].getData()
    i = len(x) // 3
    assert _tip_at(lv.time_tips, lv.time_plot, x[i], y[i])
    text = lv.time_tips.texts()[-1]
    n = int(text.split("n = ")[1].split("\n")[0].replace(" ", ""))  # binlik ayırıcı: ince boşluk
    info = handle.desc.channels[0]
    expected = float(handle.ring.read(n, n + 1)[1][0, 0]) * info.scale + info.offset
    shown = float(text.split("değer = ")[1].split(" ")[0])
    assert shown == pytest.approx(expected, rel=1e-4, abs=1e-6)
    # İşarete tıklamak onu kaldırır (gerçek tıklama yolu: _commit_click)
    from PyQt6.QtCore import QPointF
    lv.time_tips._pending = lv.time_plot.getPlotItem().getViewBox().mapViewToScene(QPointF(x[i], y[i]))
    lv.time_tips._commit_click()
    assert lv.time_tips.count == 0
    assert _tip_at(lv.time_tips, lv.time_plot, x[i], y[i])
    lv.time_tips.clear()
    assert lv.time_tips.count == 0

    # Analiz tablosu: görünen aralığın RMS'i numpy ile aynı
    lv.time_analysis.button.setChecked(True)
    x0, x1 = lv.time_analysis.visible_range()
    columns = lv._time_analysis(x0, x1)
    rms_text = dict(columns[0][2])["RMS"]
    n0 = max(handle.ring.valid_from(), int(np.ceil((x0 - handle.t0) * handle.fs)))
    n1 = min(lv.end_index(handle.key, handle), int(np.floor((x1 - handle.t0) * handle.fs)) + 1)
    values = handle.ring.read(n0, n1)[1][:, 0].astype(np.float64) * info.scale + info.offset
    assert float(rms_text.split()[0]) == pytest.approx(float(np.sqrt(np.mean(values ** 2))), rel=1e-3)
    win.registry["stream.run"].trigger()
    win.sources.stop()


def test_designer_tips_and_pole_zero_analysis(app, win):
    fv = win.filter_view
    fv.set_spec(F.FilterSpec(family="iir", method="butter", order=4, f1=1000.0, fs=10_000.0))
    assert _tip_at(fv.mag_tips, fv.mag_plot, 1000.0, -3.0)
    assert "-3.01" in fv.mag_tips.texts()[0] or "-3.00" in fv.mag_tips.texts()[0]
    stats = dict(fv._pz_stats(0, 0)[0][2])
    assert float(stats["En büyük |p|"]) == pytest.approx(fv.design.max_pole_radius, abs=1e-6)
    fv.f1.setValue(1500.0)
    assert wait_for(app, lambda: fv.design.spec.f1 == 1500.0, 2.0)
    assert fv.mag_tips.count == 0  # yeniden tasarımda eski ipuçları silinir


@pytest.mark.parametrize("method", list(F.SPECIAL_METHODS))
def test_every_special_filter_designs_in_the_designer(app, win, method):
    fv = win.filter_view
    fv.select("special", method)
    fv.redesign()
    assert fv.design is not None, fv.error
    assert fv.apply_button.isEnabled()
    assert fv.export_button.isEnabled() == (method != "median")
    if method != "median":
        x, _ = fv._mag_curve.getData()
        assert x is not None and len(x) > 100


# ============================================================================ sensör füzyonu
def test_fusion_view_runs_algorithms_on_frozen_imu_recording(app, win):
    controls = win.sources_panel.replay
    controls.speed.setCurrentIndex(controls.speed.count() - 1)
    win.start_replay(ROOT / "samples_signal_data" / "10_imu_9eksen_yonelim_referansli_200Hz.sptr")
    assert wait_for(app, lambda: win.sources.streams and next(iter(win.sources.streams.values())).latest_index() > 1500)
    snap = win.freeze()[0]
    win.sources.stop()
    fv = win.fusion_view
    fv.source.setCurrentIndex(fv.source.findData(f"ws{snap.id}"))
    mapping = fv.mapping()
    assert mapping is not None and "mag" in mapping.inputs and mapping.reference and mapping.units["gyro"] == "°/s"
    for key, box in fv._enabled.items():
        box.setChecked(key in ("mahony", "ekf"))
    assert fv.run()
    assert wait_for(app, lambda: fv.engine is not None and not fv._timer.isActive(), 90.0)
    table = fv.table
    headers = [table.horizontalHeaderItem(c).text() for c in range(table.columnCount())]
    rows = {table.item(r, 0).text(): [table.item(r, c).text() for c in range(table.columnCount())]
            for r in range(table.rowCount())}
    assert set(rows) == {"Mahony", "EKF (çarpımsal)"}
    roll = float(rows["EKF (çarpımsal)"][headers.index("Roll RMSE (°)")])
    assert roll < 3.0
    fv.stop()
    assert fv.engine is None


# ============================================================================ denetim bulgularının regresyon testleri
def test_filtered_spectrum_is_drawn_on_every_live_refresh(app, win):
    """UX denetimi Y2: canlı akışta filtreli spektrum yenilemelerin çoğunda boş kalıyordu (uç indeks yarışı)."""
    win.start_replay(ROOT / "samples_signal_data" / "01_sinus_50Hz_harmonik_2kanal_int16_10kHz.sptr")
    assert wait_for(app, lambda: win.sources.streams and next(iter(win.sources.streams.values())).latest_index() > 9000)
    win.filter_view.set_spec(F.FilterSpec(family="iir", method="butter", order=4, f1=500.0, fs=10_000.0))
    win.filter_view.request_apply()
    win._show_view(0)
    lv = win.live_view
    handle = next(iter(win.sources.streams.values()))
    curve = lv._filt_spec[(handle.key, 0)]
    drawn = 0
    for _ in range(15):
        pump(app, 0.05)
        curve.setData([], [])
        lv.refresh(force=True)
        x, _ = curve.getData()
        drawn += x is not None and len(x) > 0
    assert drawn == 15
    win.sources.stop()


def test_unimplemented_commands_are_disabled_with_reason(app, win):
    for key in ("file.new_project", "export.csv", "an.psd", "edit.preferences"):
        act = win.registry[key]
        assert not act.isEnabled() and "bu sürümde yok" in act.toolTip()
    for key in ("view.fusion", "flt.export", "stream.freeze", "flt.comb", "edit.undo"):
        assert win.registry[key].isEnabled() or key == "edit.undo"


def test_time_window_follows_sample_rate_of_first_stream(app, win):
    win.start_replay(SAMPLE)  # 500 Hz EKG → ~2000 örnek ≈ 5 s
    assert wait_for(app, lambda: bool(win.sources.streams))
    pump(app, 0.2)
    assert win.properties_panel.time_window.currentText() == "5 s"
    assert win.properties_panel.y_min.suffix() == " mV"  # Y sınırlarının eki kanalın birimi
    win.sources.stop()


def test_repeated_channel_colors_are_dashed_and_raw_gray_adapts(app, win, theme):
    lv = win.live_view
    from spt.core.local_ring import StaticRing
    from spt.core.snapshot import Snapshot
    from spt.protocol.format import SAMPLE_TYPES_BY_NAME, ChannelInfo, Descriptor, Layout
    from PyQt6.QtCore import Qt as _Qt
    desc = Descriptor(1, 1, 100.0, SAMPLE_TYPES_BY_NAME["float64"], Layout.INTERLEAVED, "çok",
                      tuple(ChannelInfo(f"k{i}") for i in range(10)))
    snap = Snapshot(1, "çok", "test", desc, 0.0, StaticRing(np.zeros((200, 10)), 0))
    colors = [theme.series_color(i) for i in range(10)]
    lv.add_stream(snap.handle(), colors, [True] * 10)
    key = snap.handle().key
    styles = [lv._time_curves[(key, ch, 0)].opts["pen"].style() for ch in range(10)]
    assert styles[:8] == [_Qt.PenStyle.SolidLine] * 8 and styles[8:] == [_Qt.PenStyle.DashLine] * 2
    assert theme.raw_trace_for(["#d95926"]) == "#8a837c" and theme.raw_trace_for(colors) == "#b3aca5"
    lv.remove_stream(key)
    assert not any(k[0] == key for k in lv._colors)


# ============================================================================ denetim regresyonları (arayüz)
def test_designer_readouts_are_exact_in_log_frequency_mode(app, win):
    # Denetim: Log f açıkken imleç okuması her noktada −300 dB / −359.9° gösteriyordu; hareketli ortalamada da
    # ekran için seyreltilmiş eğriden okunduğu için −19.6 dB (gerçek −34.8 dB) görünüyordu
    fv = win.filter_view
    for spec, freqs in ((F.FilterSpec(family="iir", method="butter", band="lowpass", order=4, f1=1000.0, fs=10_000.0),
                         (500.0, 1000.0, 3000.0)),
                        (F.FilterSpec(family="special", method="moving_avg", length=21, fs=10_000.0), (612.0, 1234.5))):
        fv.set_spec(spec)
        fv.redesign()
        fv.set_log_freq(True)
        for f in freqs:
            fv._on_mag_hover(np.log10(f), 0.0)
            exact = F.analyze(fv.design, np.array([f])).magnitude_db[0]
            assert f"{exact:.2f} dB" in fv.mag_readout.text()
        fv.set_analysis_mode("phase")
        fv._on_analysis_hover(np.log10(freqs[0]), 0.0)
        exact = F.analyze(fv.design, np.array([freqs[0]])).phase_deg[0]
        shown = float(fv.analysis_readout.text().split()[-1].rstrip("°"))
        assert abs((shown - exact + 180) % 360 - 180) < 0.01
    fv.set_log_freq(False)


def test_step_table_and_minus_3db_rows(app, win):
    fv = win.filter_view
    fv.set_spec(F.FilterSpec(family="iir", method="butter", band="highpass", order=4, f1=1000.0, fs=10_000.0))
    fv.redesign()
    fv.set_analysis_mode("step")
    items = dict(fv._analysis_stats(-1e9, 1e9)[0][2])
    assert items["Aşma"] == "—" and "DC kazancı sıfır" in items["Son değer"]  # eskiden %1.65e+08
    rows = dict(fv._mag_stats(0, 5000)[0][2])
    assert "−3 dB noktası" in rows  # yüksek geçirende de var (eskiden yoktu)
    fv.set_spec(F.FilterSpec(family="iir", method="butter", band="bandpass", order=3, f1=1000.0, f2=2000.0,
                             fs=10_000.0))
    fv.redesign()
    edges = dict(fv._mag_stats(0, 5000)[0][2])["−3 dB noktaları"].split(", ")
    assert len(edges) == 2  # alt ve üst kenar (eskiden yalnız üst)


def test_editing_updates_the_same_stage_after_chain_changes(app, win):
    # Denetim: düzenleme indeksle tutuluyordu; başka adım silinince yanlış adımın üzerine yazılıyordu
    cp = win.chain_panel
    from spt.dsp.chain import ChainStage, ProcessingChain
    specs = [F.FilterSpec(family="special", method="dc_block", dc_cutoff=1.0),
             F.FilterSpec(family="iir", method="butter", order=2, f1=100.0),
             F.FilterSpec(family="special", method="moving_avg", length=5)]
    cp.set_chain(ProcessingChain(ChainStage(s) for s in specs))
    win._on_chain_changed(cp.chain)
    win._edit_stage(1, specs[1])                       # butter düzenleniyor
    cp.list.setCurrentRow(0)
    cp._remove()                                       # önce dc_block silindi
    cp.list.item(0).setCheckState(Qt.CheckState.Unchecked)  # düzenlenen adım kapatıldı (uid korunur)
    win.filter_view.order.setValue(6)
    win.filter_view.request_apply()
    stages = cp.chain.stages
    assert [s.spec.method for s in stages] == ["butter", "moving_avg"]
    assert stages[0].spec.order == 6 and not stages[0].enabled
    assert stages[1].spec == specs[2]                   # moving_avg'e dokunulmadı


def test_display_preferences_persist_and_auto_window_respects_user(app, theme, tmp_path, monkeypatch):
    monkeypatch.setenv("SPT_SETTINGS_FILE", str(tmp_path / "settings.ini"))
    from spt.gui.main_window import MainWindow

    first = MainWindow(theme)
    box = first.properties_panel.time_window
    box.setCurrentIndex(box.findText("500 ms"))
    box.activated.emit(box.currentIndex())             # kullanıcı seçimi
    first.sources_panel.replay.loop.setChecked(True)
    first.close()
    first.deleteLater()
    pump(app, 0.1)
    second = MainWindow(theme)
    try:
        assert second.properties_panel.time_window.currentText() == "500 ms"
        assert second.sources_panel.replay.loop.isChecked()
        second._suggest_time_window(500.0)              # örnekleme hızına göre öneri kullanıcı seçimini ezmez
        assert second.properties_panel.time_window.currentText() == "500 ms"
    finally:
        second.close()
        second.deleteLater()
        pump(app, 0.1)


def test_placeholder_in_horizontal_splitter_is_compact(app, win):
    win._show_view(2)
    pump(app, 0.2)
    pz = win.filter_view.pz_card
    splitter = pz.parentWidget()
    before, card_width = splitter.sizes()[:2]
    win.detacher.detach("filter_pz")
    pump(app, 0.2)
    placeholder = splitter.widget(1)
    assert placeholder is not pz
    assert all(not b.text() for b, _ in placeholder._buttons)  # yan yana bölücüde yalnız ikon (metin kesilmez)
    assert placeholder.minimumSizeHint().width() <= 300
    sizes = splitter.sizes()
    extent = max(placeholder.minimumSizeHint().width(), 160)
    # boşalan yer komşuya geçer (kart yer tutucudan darsa yer tutucu en küçük genişliğini alır, komşu o kadar daralır)
    assert sizes[1] <= extent + 2 and sizes[0] >= before - max(0, extent - card_width) - 2
    win.detacher.attach_all()
    assert splitter.widget(1) is pz


def test_unimplemented_shortcut_gives_feedback(app, win):
    from PyQt6.QtGui import QShortcut
    act = win.registry["file.new_project"]
    assert not act.isEnabled() and act.shortcuts()
    shortcuts = [s for s in win.findChildren(QShortcut) if s.key() == act.shortcuts()[0]]
    assert shortcuts
    shortcuts[0].activated.emit()
    assert "Aşama" in win.status.currentMessage()


# ============================================================================ sinyal üreteci, genel füzyon, 3B, kod
def test_spectrogram_removed_and_signal_generator_on_ctrl3(app, win):
    from spt.gui.main_window import VIEW_KEYS
    assert "view.spectrogram" not in VIEW_KEYS and not hasattr(win, "spectrogram_view")
    act = win.registry["view.siggen"]
    assert act.shortcut().toString() == "Ctrl+3"
    act.trigger()
    assert win.views.currentWidget() is win.siggen_view


def test_signal_generator_computes_dac_output_and_metrics(app, win):
    from spt.dsp import siggen as sg
    v = win.siggen_view
    v.kind.setCurrentIndex(v.kind.findData("square"))
    v.recompute()
    assert v.table is not None and v.metrics is not None
    assert 30.0 < v.metrics.thd_pct < 60.0  # kare dalga: ideal 10 harmonikte %43.8 (örtüşme / nicemleme payıyla)
    # nicemlenmiş çıkış gerçekten DAC kodlarından: her değer LSB ızgarasında
    lsb = v.table.dac.lsb
    levels = (v._time_out - v.table.dac.out_min) / lsb
    assert np.allclose(levels, np.round(levels), atol=1e-6)
    # formül hatası konumuyla gösterilir, kod düğmesi kapanır
    v.kind.setCurrentIndex(v.kind.findData("expr"))
    v.expr.setText("sin(2*pi*ph) + __import__('os')")
    v.recompute()
    assert v.table is None and not v.code_button.isEnabled()
    v.expr.setText("0.6*sin(2*pi*ph) + 0.3*square(ph, 0.2)")
    v.recompute()
    assert v.table is not None and v.code_button.isEnabled()
    # sığdır: genlik / ofset DAC aralığını doldurur, kırpma yok
    v._fit_to_dac()
    v.recompute()
    assert v.table.clipped == 0
    assert isinstance(v.wave_spec(), sg.WaveSpec)


def test_signal_generator_code_dialog_and_simulator_file(app, win, tmp_path):
    v = win.siggen_view
    v.kind.setCurrentIndex(v.kind.findData("sine"))
    v.recompute()
    v.show_code()
    dlg = v._code_dialog
    names = [f.name for f in dlg.files]
    assert {"wave.h", "wave.c", "KULLANIM.md"} <= set(names)
    written = dlg.save_to(tmp_path, ask=False)
    assert len(written) == len(names) and (tmp_path / "wave.c").read_text(encoding="utf-8").startswith("/*")
    dlg.close()
    paths = []
    v.simulate_requested.disconnect()
    v.simulate_requested.connect(lambda path, fs: paths.append((path, fs)))
    v._simulate()
    assert paths and paths[0][1] == v.table.fs
    data = np.load(paths[0][0])
    assert len(data) >= v.table.N and np.isfinite(data).all()


@pytest.mark.parametrize("scenario,sample,algos", [
    ("altitude", "11_irtifa_baro_25Hz_ivme_200Hz_referansli.sptr", ("complementary", "kf")),
    ("linear_multi", "13_coklu_mesafe_sensoru_enkoder_100Hz_referansli.sptr", ("weighted_mean", "kf")),
])
def test_generic_fusion_scenarios_run_from_recordings(app, win, scenario, sample, algos):
    controls = win.sources_panel.replay
    controls.speed.setCurrentIndex(controls.speed.count() - 1)
    win.start_replay(ROOT / "samples_signal_data" / sample)
    assert wait_for(app, lambda: win.sources.streams and next(iter(win.sources.streams.values())).latest_index() > 1500)
    snap = win.freeze()[0]
    win.sources.stop()
    fv = win.fusion_view
    fv.scenario_combo.setCurrentIndex(fv.scenario_combo.findData(scenario))
    fv.source.setCurrentIndex(fv.source.findData(f"ws{snap.id}"))
    m = fv.mapping()
    assert m is not None and m.reference  # ad / birimden otomatik eşlendi
    for key, box in fv._enabled.items():
        box.setChecked(key in algos)
    assert fv.run()
    assert wait_for(app, lambda: not fv._timer.isActive(), 90.0)
    headers = [fv.table.horizontalHeaderItem(c).text() for c in range(fv.table.columnCount())]
    rows = {fv.table.item(r, 0).text(): fv.table.item(r, 1).text() for r in range(fv.table.rowCount())}
    assert len(rows) == 2 and "RMSE" in headers[1]
    assert all(np.isfinite(float(v)) for v in rows.values())
    # her çıktının kendi kartı; lejantta seçili algoritmalar + referans + ham
    assert len(fv._cards) == len(fv.scenario.outputs)
    assert fv._legend_lay.count() == len(algos) + 2
    colors = {fv.algo_color(k) for k in fv.scenario.algorithms}
    assert len(colors) == len(fv.scenario.algorithms)  # aynı renk iki algoritmada yok
    fv.stop()


def test_orientation_3d_view_and_quaternion_mode(app, win):
    controls = win.sources_panel.replay
    controls.speed.setCurrentIndex(controls.speed.count() - 1)
    win.start_replay(ROOT / "samples_signal_data" / "10_imu_9eksen_yonelim_referansli_200Hz.sptr")
    assert wait_for(app, lambda: win.sources.streams and next(iter(win.sources.streams.values())).latest_index() > 1500)
    snap = win.freeze()[0]
    win.sources.stop()
    fv = win.fusion_view
    fv.scenario_combo.setCurrentIndex(fv.scenario_combo.findData("orientation"))
    fv.source.setCurrentIndex(fv.source.findData(f"ws{snap.id}"))
    for key, box in fv._enabled.items():
        box.setChecked(key in ("mahony", "ekf"))
    assert fv.run()
    assert wait_for(app, lambda: not fv._timer.isActive(), 90.0)
    fv._update_3d()
    bodies = fv.view3d.bodies
    assert any(b.solid and b.key == "ekf" for b in bodies) and any(b.key == "ref" for b in bodies)
    solid = next(b for b in bodies if b.solid)
    assert abs(np.linalg.norm(solid.q) - 1.0) < 1e-6
    info = "\n".join(fv.view3d.overlay)
    assert "q w" in info and "R " in info and "hata" in info and fv.view3d.gauges is not None
    fv.mode_combo.setCurrentIndex(2)  # kuaterniyon bileşenleri
    assert len(fv._cards) == 4
    fv.redraw()
    x, y = fv._curves[("ekf", 0)].getData()
    assert y is not None and np.nanmax(np.abs(y)) <= 1.0 + 1e-6
    fv.mode_combo.setCurrentIndex(0)
    fv.stop()


def test_fusion_code_dialog_generates_and_measures(app, win):
    fv = win.fusion_view
    fv.scenario_combo.setCurrentIndex(fv.scenario_combo.findData("orientation"))
    for key, box in fv._enabled.items():
        box.setChecked(key in ("mahony",))
    fv.show_code()
    dlg = fv._code_dialog
    assert {"spt_mahony.h", "spt_mahony.c", "KULLANIM.md"} <= {f.name for f in dlg.files}
    assert wait_for(app, lambda: dlg.generate_button.isEnabled(), 240.0)  # ölçüm arka planda bitti
    text = dlg.summary.toPlainText()
    assert "Bellek" in text and "CPU yükü" in text
    dlg.close()


def test_orientation3d_math_matches_fusion_conventions():
    from spt.dsp import fusion as fu
    from spt.gui.widgets.orientation3d import euler_deg, quat_from_euler_deg, rotation
    rng = np.random.default_rng(5)
    for _ in range(20):
        r, p, y = rng.uniform(-170, 170), rng.uniform(-80, 80), rng.uniform(-170, 170)
        q = quat_from_euler_deg(r, p, y)
        ref = fu.quat_from_euler(np.radians(r), np.radians(p), np.radians(y))
        assert np.allclose(q, ref, atol=1e-12)
        assert np.allclose(rotation(q), fu.rotmat(ref), atol=1e-12)
        assert np.allclose(euler_deg(q), (r, p, y), atol=1e-9)


def test_signal_generator_audit_regressions(app, win):
    v = win.siggen_view
    # Arduino hedefinde CPU yükü tahmini yok: yeniden hesap çökmeden sürmeli ve tablo güncellenmeli
    v.target.setCurrentIndex(v.target.findData("arduino"))
    v.kind.setCurrentIndex(v.kind.findData("sine"))
    v._spec_fields["freq"].setValue(2000.0)
    v.recompute()
    assert v.table is not None and abs(v.table.f_real - 2000.0) < 1.0
    texts = [v.metrics_table.item(r, 1).text() for r in range(v.metrics_table.rowCount())]
    assert any("2 kHz" in t for t in texts)
    # formüldeki "<" durum satırını yutmamalı
    v.kind.setCurrentIndex(v.kind.findData("expr"))
    v.expr.setText("where(ph<0.5, 1, -1)")
    v.recompute()
    assert "&lt;" in v.status.text() and "Hz" in v.status.text()
    # simülatör dosyası döngüde kesintisiz (tek atımlık gürültü / tarama tam tekrar)
    paths = []
    v.simulate_requested.disconnect()
    v.simulate_requested.connect(lambda path, fs: paths.append(path))
    v.kind.setCurrentIndex(v.kind.findData("chirp"))
    v.recompute()
    v._simulate()
    data = np.load(paths[-1])
    assert len(data) % v.table.N == 0
    v.target.setCurrentIndex(v.target.findData("stm32_hal_dma"))


def test_fusion_code_dialog_for_generic_scenario(app, win):
    fv = win.fusion_view
    fv.scenario_combo.setCurrentIndex(fv.scenario_combo.findData("altitude"))
    for key, box in fv._enabled.items():
        box.setChecked(key in ("kf",))
    fv.show_code()
    dlg = fv._code_dialog
    names = {f.name for f in dlg.files}
    assert any(n.endswith(".c") and "kf" in n for n in names) and "KULLANIM_alt.md" in names
    assert wait_for(app, lambda: dlg.generate_button.isEnabled(), 240.0)
    text = dlg.summary.toPlainText()
    assert "Bellek" in text and "CPU yükü" in text and "ölçüldü" in text
    dlg.close()


# ============================================================================ UI denetimi 2. tur regresyon testleri
def _frozen(app, win, sample):
    """Kaydı en hızlı oynatıp dondurur; oynatmayı durdurur ve anlık görüntüyü döndürür."""
    controls = win.sources_panel.replay
    controls.speed.setCurrentIndex(controls.speed.count() - 1)
    win.start_replay(ROOT / "samples_signal_data" / sample)
    assert wait_for(app, lambda: win.sources.streams and next(iter(win.sources.streams.values())).latest_index() > 1500)
    snap = win.freeze()[0]
    win.sources.stop()
    return snap


def test_narrow_window_focus_layout_hides_and_restores_panels(app, win):
    from spt.gui.main_window import FOCUS_WIDTH
    d = win.docks

    def shown():
        return {k: dock.toggleViewAction().isChecked() for k, dock in d.items()}

    assert win.width() < FOCUS_WIDTH
    win._show_view(0)
    pump(app, 0.2)
    before = shown()
    win._show_view(1)  # sinyal üreteci akış kullanmaz: tüm paneller gizlenir, kullanıcıya söylenir
    pump(app, 0.2)
    assert not any(shown().values()) and "gizlendi" in win.status.currentMessage()
    win._show_view(0)
    pump(app, 0.2)
    assert shown() == before
    # sekme grubunda önde olan yine önde (arkadaki sekme öne geçmez)
    assert not d["chain"].visibleRegion().isEmpty() and d["measurements"].visibleRegion().isEmpty()
    win._show_view(2)  # filtre tasarımcısı: tasarımın uygulandığı İşlem Zinciri açık kalır
    pump(app, 0.2)
    assert d["chain"].toggleViewAction().isChecked() and not d["properties"].toggleViewAction().isChecked()
    assert win.filter_view.width() >= win.filter_view.minimumSizeHint().width()  # görünüm sıkışmaz
    win._show_view(0)
    pump(app, 0.2)
    assert shown() == before
    # kullanıcı üreteçteyken menüden açtığı paneli o görünümde bir daha gizlemeyiz
    win._show_view(1)
    pump(app, 0.2)
    d["sources"].toggleViewAction().trigger()
    win._show_view(0)
    win._show_view(1)
    pump(app, 0.2)
    assert d["sources"].toggleViewAction().isChecked() and not d["chain"].toggleViewAction().isChecked()
    # geniş pencerede gizleme yok; daralınca yeniden uygulanır
    win.resize(FOCUS_WIDTH + 100, 960)
    pump(app, 0.3)
    assert all(shown()[k] for k in before if before[k])
    win.resize(1600, 960)
    win._show_view(4)
    pump(app, 0.3)
    assert not d["chain"].toggleViewAction().isChecked()
    win._restore_focus_docks()  # kapanışta yerleşim kaydından önce çağrılır
    assert all(shown()[k] for k in before if before[k])
    win._show_view(0)


def test_setup_panels_fit_and_fusion_layout_adapts(app, win):
    from spt.dsp import fusion_generic as fg
    from spt.dsp import siggen as sg
    fv, sv = win.fusion_view, win.siggen_view

    def fits(host):  # ayar içeriği, kartın en dar hâlinde bile görünür alana sığar (yatayda kırpılmaz)
        scroll = host.parentWidget().parentWidget()
        card = scroll.parentWidget()
        chrome = 2 * 16 + scroll.verticalScrollBar().sizeHint().width() + 2  # kart kenarları + kaydırma çubuğu
        return host.minimumSizeHint().width() <= card.minimumWidth() - chrome

    for key in fg.SCENARIOS:
        fv._set_scenario(key)
        pump(app, 0.05)
        assert fits(fv._setup_col.parentWidget()), key
    for kind in sg.WAVEFORMS:
        sv.kind.setCurrentIndex(sv.kind.findData(kind))
        pump(app, 0.05)
        assert fits(sv.wave_section.parentWidget()), kind
    sv.kind.setCurrentIndex(sv.kind.findData("sine"))
    fv._set_scenario("orientation")
    # geniş alanda 3B kartı + tablo grafiklerin sağında (grafikler tam boy); dar alanda altında
    win._show_view(4)
    pump(app, 0.4)
    assert fv.side_by_side
    assert fv._cards_scroll.height() >= 2 * 150  # en az iki çıktı kartı tamamen görünür
    # ayrık penceredeki kart düzen değişirken geri çalınmaz; geri takılınca doğru bölücüye döner
    win.detacher.detach("fusion_side")
    win.resize(1000, 960)
    pump(app, 0.4)
    assert not fv.side_by_side
    assert win.detacher.is_detached("fusion_side") and fv.side_card.parentWidget() is not fv._bottom
    win.detacher.attach("fusion_side")
    pump(app, 0.1)
    assert fv.side_card.parentWidget() is fv._bottom
    win.resize(1600, 960)
    pump(app, 0.4)
    assert fv.side_by_side and fv._bottom.widget(0) is fv.side_card  # yan yanada 3B üstte
    win._show_view(0)


def test_fusion_keeps_user_mapping_when_source_disappears_and_returns(app, win):
    snap = _frozen(app, win, "10_imu_9eksen_yonelim_referansli_200Hz.sptr")
    fv = win.fusion_view
    fv.scenario_combo.setCurrentIndex(fv.scenario_combo.findData("orientation"))
    sid = f"ws{snap.id}"
    fv.source.setCurrentIndex(fv.source.findData(sid))
    for box in fv._map_inputs["mag"][0]:  # kullanıcı manyetometreyi ve yaw referansını kaldırır
        box.setCurrentIndex(0)
    fv._map_refs["yaw"].setCurrentIndex(0)

    def kept():
        m = fv._selection()
        return "mag" not in m.inputs and "yaw" not in m.reference and "gyro" in m.inputs and bool(m.reference)

    assert kept()
    sources = list(fv._sources)
    others = [s for s in sources if s[0] != sid]
    # durmuşken kaynak gidip gelir (ör. döngülü oynatma başa sardı): otomatik eşleme üzerine yazmaz
    fv.set_sources(others)
    fv.set_sources(sources)
    assert fv.source.currentData() == sid and kept()
    # çalışırken kaynak kaybolur: kutular olduğu gibi kalır; geri gelince aynı eşlemeyle yeniden başlar
    for key, box in fv._enabled.items():
        box.setChecked(key == "mahony")
    assert fv.run()
    fv.set_sources(others)
    assert fv.engine is None and fv._pending_restart == sid and kept()
    fv.set_sources(sources)
    assert fv.engine is not None and fv._pending_restart is None and kept()
    assert "mag" not in fv.engine.mapping.inputs
    fv.stop()


def test_fusion_processes_up_to_run_time_when_live_processing_is_off(app, win):
    win.start_replay(ROOT / "samples_signal_data" / "10_imu_9eksen_yonelim_referansli_200Hz.sptr")
    assert wait_for(app, lambda: win.sources.streams and next(iter(win.sources.streams.values())).latest_index() > 300)
    handle = next(iter(win.sources.streams.values()))
    fv = win.fusion_view
    fv.scenario_combo.setCurrentIndex(fv.scenario_combo.findData("orientation"))
    fv.source.setCurrentIndex(fv.source.findData(handle.key))
    for key, box in fv._enabled.items():
        box.setChecked(key == "mahony")
    fv.live.setChecked(False)
    assert fv.run()
    frozen = fv._frozen
    assert frozen is not None and frozen > fv.engine.start
    assert wait_for(app, lambda: fv.engine.processed("mahony") >= frozen, 30.0)  # “İşlenen %0”da kalmaz
    pump(app, 0.8)
    assert fv.engine.processed("mahony") <= frozen and handle.latest_index() > frozen  # yeni örnekler işlenmez
    fv.live.setChecked(True)
    assert wait_for(app, lambda: fv.engine.processed("mahony") > frozen, 30.0)
    fv.stop()
    win.sources.stop()


def test_trajectory_is_cleared_on_rerun_and_follows_legend(app, win):
    snap = _frozen(app, win, "12_arac_gnss_5Hz_jiro_odometri_100Hz_referansli.sptr")
    fv = win.fusion_view
    fv.scenario_combo.setCurrentIndex(fv.scenario_combo.findData("position2d"))
    fv.source.setCurrentIndex(fv.source.findData(f"ws{snap.id}"))
    for key, box in fv._enabled.items():
        box.setChecked(key in ("dead_reckoning", "ekf"))
    assert fv.run()
    assert wait_for(app, lambda: not fv._timer.isActive(), 90.0)
    assert {"dead_reckoning", "ekf"} <= set(fv._traj_curves)
    fv._set_trace_visible("ekf", False)  # lejanttan gizlenen algoritma yörüngede de gizlenir
    assert not fv._traj_curves["ekf"].isVisible()
    fv._set_trace_visible("ekf", True)
    for key, box in fv._enabled.items():
        box.setChecked(key == "dead_reckoning")
    assert fv.run()
    assert wait_for(app, lambda: not fv._timer.isActive(), 90.0)
    assert "ekf" not in fv._traj_curves  # önceki çalıştırmanın eğrisi kalmaz
    drawn = [i for i in fv.traj_plot.listDataItems() if i.xData is not None and len(i.xData)]
    assert len(drawn) == len(fv._traj_curves)
    fv.stop()


def test_fusion_code_dialog_mcu_defaults_and_non_blocking_measurement(app, win):
    from spt.codegen import mcu
    fv = win.fusion_view
    fv.scenario_combo.setCurrentIndex(fv.scenario_combo.findData("orientation"))
    for key, box in fv._enabled.items():
        box.setChecked(key == "mahony")
    fv.show_code()
    dlg = fv._code_dialog
    # MCU değişince Flash / RAM o profilin tipik değerine gelir (Flash % / RAM % doğru çıksın)
    for key in ("cortex-m0plus", "cortex-m7"):
        dlg.mcu.setCurrentIndex(dlg.mcu.findData(key))
        prof = mcu.MCU_PROFILES[key]
        assert dlg.flash.value() == prof.flash_kb and dlg.ram.value() == prof.ram_kb
    # ölçüm sürerken yeniden üretmek arayüzü bekletmez; eski ölçümün sonucu yok sayılır
    dlg.mcu.setCurrentIndex(dlg.mcu.findData("cortex-m3"))
    t0 = time.perf_counter()
    dlg.refresh()
    dlg.mcu.setCurrentIndex(dlg.mcu.findData("cortex-m4f"))
    dlg.refresh()
    assert time.perf_counter() - t0 < 3.0
    assert wait_for(app, lambda: dlg.generate_button.isEnabled(), 240.0)
    text = dlg.summary.toPlainText()
    assert mcu.MCU_PROFILES["cortex-m4f"].name in text and mcu.MCU_PROFILES["cortex-m3"].name not in text
    dlg.close()


def test_signal_generator_ui_audit_round2_regressions(app, win, theme):
    from PyQt6.QtGui import QColor

    from spt.gui.widgets.inspector import fmt, visible_slice
    v = win.siggen_view
    win._show_view(1)
    pump(app, 0.2)
    # noktalar: çizgiyi sürüklemek tüm noktaları kaydırmaz (ROI ötelenemez)
    v.kind.setCurrentIndex(v.kind.findData("points"))
    v.recompute()
    assert v._roi is not None and not v._roi.translatable
    # harmonik tablosunda sayı olmayan hücre: kırmızı işaretlenir, durum satırında uyarılır
    v.kind.setCurrentIndex(v.kind.findData("harmonics"))
    v.recompute()
    v.rows_table.item(0, 1).setText("abc")
    v.recompute()
    assert "1. satır sayı değil" in v.status.text()
    assert v.rows_table.item(0, 1).background().color().name() == QColor(theme.tokens.critical).name()
    v.rows_table.item(0, 1).setText("1")
    v.recompute()
    assert "sayı değil" not in v.status.text()
    # basamağın sağ ucu kendi örneğini gösterir (sonrakini değil)
    v.kind.setCurrentIndex(v.kind.findData("sine"))
    v.recompute()
    xs, _ = v._out_curve.getData()
    out = v._time_traces()[0]
    for n in (0, 5, 17):
        t, value = out.exact(float(xs[2 * n + 1]))
        assert t == pytest.approx(n / v.table.fs) and value == pytest.approx(float(v._time_out[n]))
    # “İdeal” sütununun zamanları da pencerenin başından sayılır (0'dan değil)
    x0, x1 = 0.5e-3, 1.0e-3
    cols = v._time_stats(x0, x1)
    ideal = dict(cols[1][2])
    sl = visible_slice(v._time_x, x0, x1)
    t0 = float(v._time_x[sl][0])
    assert ideal["En küçük zamanı"] == fmt(t0 + int(np.argmin(v._time_ideal[sl])) / v.table.fs, "s", 7)
    assert ideal["En büyük zamanı"] == fmt(t0 + int(np.argmax(v._time_ideal[sl])) / v.table.fs, "s", 7)
    # EKG 30 bpm'de bir periyot 2 s: “En uzun tablo” kendiliğinden yeterli boya çıkar
    v.max_len.setValue(4096)
    v.kind.setCurrentIndex(v.kind.findData("ecg"))
    v._spec_fields["bpm"].setValue(30.0)
    v.recompute()
    assert v.table is not None and v.error is None and v.max_len.value() >= 2 * v.fs.value()
    v.kind.setCurrentIndex(v.kind.findData("sine"))
    v.max_len.setValue(4096)
    v.recompute()
    win._show_view(0)
