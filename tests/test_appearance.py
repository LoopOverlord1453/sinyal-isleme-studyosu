"""Grafik görünümü (ekransız): kanal renkleri, spektrum / max-min-hold renkleri ve grafik yazı tipi; kalıcılık ve
"Varsayılana dön".

Ayarlar geçici bir INI dosyasında tutulur (SPT_SETTINGS_FILE); kullanıcının kayıtlı ayarlarına dokunulmaz.
"""
import os
import time

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def theme(app):
    from spt.gui.theme import ThemeManager

    t = ThemeManager(app)
    t.apply()
    return t


@pytest.fixture
def settings_file(tmp_path, monkeypatch):
    path = tmp_path / "settings.ini"
    monkeypatch.setenv("SPT_SETTINGS_FILE", str(path))
    return path


def make_window(app, theme):
    from spt.gui.main_window import MainWindow

    w = MainWindow(theme)
    w.resize(1600, 960)
    w.show()
    pump(app, 0.2)
    return w


def close_window(app, w):
    w.close()
    w.deleteLater()
    pump(app, 0.2)


def pump(app, seconds):
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        app.processEvents()
        time.sleep(0.005)


def add_stream(win, name="kart"):
    """İki kanallı sabit akış; renk kaydı akış adını SourceManager'dan okuduğu için oraya da eklenir."""
    from spt.core.local_ring import StaticRing
    from spt.core.snapshot import Snapshot
    from spt.protocol.format import SAMPLE_TYPES_BY_NAME, ChannelInfo, Descriptor, Layout

    fs = 1000.0
    t = np.arange(4000) / fs
    data = np.stack([np.sin(2 * np.pi * 50 * t), 0.5 * np.sin(2 * np.pi * 120 * t)], axis=1)
    desc = Descriptor(1, 1, fs, SAMPLE_TYPES_BY_NAME["float64"], Layout.INTERLEAVED, name,
                      (ChannelInfo("a", "V"), ChannelInfo("b", "V")))
    handle = Snapshot(1, name, "test", desc, 0.0, StaticRing(data, 0)).handle()
    win.sources.streams[handle.key] = handle
    win._on_stream_added(handle)
    win.live_view.refresh(force=True)
    return handle


def pen_color(item) -> str:
    return item.opts["pen"].color().name()


def test_channel_color_from_sources_panel_and_legend(app, theme, settings_file, monkeypatch):
    from spt.gui import main_window
    from spt.gui.views import live_view

    win = make_window(app, theme)
    try:
        lv = win.live_view
        h = add_stream(win)
        monkeypatch.setattr(main_window, "pick_color", lambda *_: "#7a1fa2")
        win.sources_panel.color_requested.emit(h.key, 0)          # Kaynaklar: sağ tık → Rengi değiştir…
        assert pen_color(lv._time_curves[(h.key, 0, 0)]) == "#7a1fa2"
        assert [b.color for b in lv.legend.toggles()][0] == "#7a1fa2"
        assert win._color_overrides == {"kart/a": "#7a1fa2"}
        spectra = {pen_color(c) for c in lv._spec_curves.values()}
        assert spectra == {theme.tokens.accent}                    # spektrum kanal renginden bağımsız

        monkeypatch.setattr(live_view, "pick_color", lambda *_: "#118811")
        lv._choose_channel_color(h.key, 1, "kart · b")             # zaman lejantından
        assert pen_color(lv._time_curves[(h.key, 1, 0)]) == "#118811"
        assert win._color_overrides["kart/b"] == "#118811"
        monkeypatch.setattr(live_view, "pick_color", lambda *_: None)   # vazgeçildi: değişmez
        lv._choose_channel_color(h.key, 1, "kart · b")
        assert pen_color(lv._time_curves[(h.key, 1, 0)]) == "#118811"
        # lejant öğesi kanalı adıyla bulur (ham gri öğesi kanal değildir)
        assert lv._legend_name(h, 0) == "kart · a"
    finally:
        close_window(app, win)


def test_spectrum_colors_and_font_apply_to_both_views(app, theme, settings_file):
    win = make_window(app, theme)
    try:
        lv, rv, p = win.live_view, win.review_view, win.properties_panel
        h = add_stream(win)
        p.color_buttons["spectrum"].set_color("#1f7a3a")
        p.color_buttons["spectrum"].color_changed.emit("#1f7a3a")
        p.color_buttons["min"].set_color("#000080")
        p.color_buttons["min"].color_changed.emit("#000080")
        assert lv.spectrum_colors == {"spectrum": "#1f7a3a", "max": None, "min": "#000080"}
        assert rv.spectrum_colors == lv.spectrum_colors
        assert pen_color(lv._spec_curves[(h.key, 0)]) == "#1f7a3a"
        assert pen_color(lv._hold_curves[(h.key, 0, "min")]) == "#000080"
        assert pen_color(lv._hold_curves[(h.key, 0, "max")]) == theme.hold_color("max")
        assert [b.color for b in lv.spec_legend.toggles()] == [theme.hold_color("max"), "#000080"]

        lv._set_spectrum_color("max", "#aa0000")                   # spektrum lejantından: panel de izler
        assert p.spectrum_colors()["max"] == "#aa0000" and rv.spectrum_colors["max"] == "#aa0000"

        p.font_px.setValue(16)
        family = next(f for f in ("Consolas", "Arial", "Segoe UI") if p.font_family.findText(f) >= 0)
        p.font_family.setCurrentIndex(p.font_family.findText(family))
        for view in (lv, rv):
            assert (view.font_family, view.font_px) == (family, 16)
            for plot in (view.time_plot, view.spec_plot):
                for name in ("left", "bottom"):
                    ax = plot.getPlotItem().getAxis(name)
                    assert ax.style["tickFont"].family() == family and ax.style["tickFont"].pixelSize() == 16
                    assert ax.labelStyle["font-size"] == "16px" and family in ax.labelStyle["font-family"]
        # SI ön eki ayarı yazı tipi değişince bozulmaz (spektrumun dB ekseni ön eksiz kalır)
        assert not lv.spec_plot.getPlotItem().getAxis("left").autoSIPrefix
        lv._update_peaks()
        assert "font-size:15px" in lv._label_css() and family in lv._label_css()
    finally:
        close_window(app, win)


def test_appearance_persists_and_resets(app, theme, settings_file, monkeypatch):
    from spt.gui import main_window

    win = make_window(app, theme)
    try:
        h = add_stream(win)
        monkeypatch.setattr(main_window, "pick_color", lambda *_: "#7a1fa2")
        win._choose_channel_color(h.key, 0)
        p = win.properties_panel
        p.color_buttons["spectrum"].set_color("#1f7a3a")
        p.color_buttons["spectrum"].color_changed.emit("#1f7a3a")
        p.font_px.setValue(14)
    finally:
        close_window(app, win)                                     # kapanışta kaydedilir

    win = make_window(app, theme)
    try:
        lv, p = win.live_view, win.properties_panel
        h = add_stream(win)
        assert lv._colors[(h.key, 0)] == "#7a1fa2"                  # aynı ad: kayıtlı renk
        assert lv._colors[(h.key, 1)] == theme.series_color(1)
        assert lv.spectrum_colors["spectrum"] == "#1f7a3a" and lv.font_px == 14
        assert p.spectrum_colors()["spectrum"] == "#1f7a3a" and p.plot_font() == (None, 14)
        assert p.color_buttons["max"].current() == theme.hold_color("max")   # varsayılan tema rengini gösterir
        p.reset_appearance()
        assert lv._colors[(h.key, 0)] == theme.series_color(0) and win._color_overrides == {}
        assert lv.spectrum_colors == {"spectrum": None, "max": None, "min": None}
        assert (lv.font_family, lv.font_px) == (None, 12)
        assert pen_color(lv._spec_curves[(h.key, 0)]) == theme.tokens.accent
    finally:
        close_window(app, win)


def test_corrupt_appearance_settings_fall_back(app, theme, settings_file):
    from spt.gui.settings import app_settings

    s = app_settings()
    s.setValue("display/spectrum_colors", "{bozuk")
    s.setValue("display/channel_colors", '{"kart/a": "renk-değil", "kart/b": "#123456"}')
    s.setValue("display/plot_font_px", 999)
    s.sync()
    win = make_window(app, theme)
    try:
        assert win.live_view.spectrum_colors == {"spectrum": None, "max": None, "min": None}
        assert win._color_overrides == {"kart/b": "#123456"}
        assert win.live_view.font_px == 24
    finally:
        close_window(app, win)


def test_legend_names_separate_streams_with_same_channel_names(app, theme):
    """Canlıda iki akış aynı kanal adlarını taşıyorsa (ör. kart + simülatör) lejant akış adını, aynı adlı iki akışta
    kaynak adresini de gösterir; tek akışta yalnız kanal adı."""
    from spt.core.handle import RingHandle
    from spt.core.local_ring import StaticRing
    from spt.gui.views.live_view import LiveView
    from spt.protocol.format import SAMPLE_TYPES_BY_NAME, ChannelInfo, Descriptor, Layout

    def live(key, source, name):
        h = RingHandle()
        h.key, h.source, h.t0 = key, source, 0.0
        h.desc = Descriptor(1, 1, 1000.0, SAMPLE_TYPES_BY_NAME["float64"], Layout.INTERLEAVED, name,
                            (ChannelInfo("ch0"),))
        h.ring = StaticRing(np.zeros((2000, 1)), 0)
        return h

    lv = LiveView(theme)
    try:
        a = live("192.168.1.10#1", "192.168.1.10", "kart")
        lv.add_stream(a, [theme.series_color(0)], [True])
        assert lv.legend.texts() == ["ch0"]
        lv.add_stream(live("127.0.0.1#1", "127.0.0.1", "simulator"), [theme.series_color(1)], [True])
        assert lv.legend.texts() == ["kart · ch0", "simulator · ch0"]
        lv.add_stream(live("192.168.1.11#1", "192.168.1.11", "kart"), [theme.series_color(2)], [True])
        assert lv.legend.texts() == ["kart (192.168.1.10) · ch0", "simulator · ch0", "kart (192.168.1.11) · ch0"]
        lv.remove_stream("127.0.0.1#1")
        lv.remove_stream("192.168.1.11#1")
        assert lv.legend.texts() == ["ch0"]
    finally:
        lv.clear()
        lv.deleteLater()
