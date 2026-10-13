"""Arayüz (ekransız): panelleri kenara gizleme şeritleri, gizli sekmeli panellerin sekme çubuğu ve üst çubuğun dar
pencere kademeleri.

Ayarlar geçici INI dosyasındadır (SPT_SETTINGS_FILE).
"""
import os
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import Qt  # noqa: E402
from PyQt6.QtTest import QTest  # noqa: E402
from PyQt6.QtWidgets import QApplication, QTabBar, QToolBar, QToolButton  # noqa: E402


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
def win(app, theme, tmp_path, monkeypatch):
    monkeypatch.setenv("SPT_SETTINGS_FILE", str(tmp_path / "settings.ini"))
    from spt.gui.main_window import MainWindow

    w = MainWindow(theme)
    w.resize(1600, 960)
    w.show()
    pump(app, 0.3)
    yield w
    w.close()
    w.deleteLater()
    pump(app, 0.2)


def pump(app, seconds):
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        app.processEvents()
        time.sleep(0.005)


def test_every_panel_has_a_stripe_button_on_its_side(app, win):
    stripes, d = win.stripes, win.docks
    sides = {key: stripes.side_of(dock) for key, dock in d.items()}
    assert sides == {"sources": "left", "workspace": "left", "properties": "right", "chain": "right",
                     "measurements": "right", "stats": "bottom", "log": "right"}
    for side in ("left", "right"):
        bar = stripes.bars[side]
        assert bar.isVisible() and bar.orientation() == Qt.Orientation.Vertical and bar.property("stripe")
        assert win.toolBarArea(bar) == (Qt.ToolBarArea.LeftToolBarArea if side == "left"
                                        else Qt.ToolBarArea.RightToolBarArea)
    btn = stripes.buttons[d["sources"]]
    assert btn.sizeHint().height() > btn.sizeHint().width()       # yan şeritte dikey yazı
    assert "sola" in d["sources"].titleBarWidget().close_btn.toolTip()
    assert "sağa" in d["properties"].titleBarWidget().close_btn.toolTip()


def test_hide_to_side_and_restore_from_stripe(app, win):
    dock = win.docks["properties"]
    btn = win.stripes.buttons[dock]
    assert dock.isVisible() and btn.isChecked()
    dock.titleBarWidget().close_btn.click()                       # başlıktaki » : sağa gizle
    pump(app, 0.1)
    assert not dock.isVisible() and not btn.isChecked() and btn.isVisible()
    btn.click()                                                   # şeritten geri aç
    pump(app, 0.1)
    assert dock.isVisible() and btn.isChecked()


def test_stripe_button_follows_moved_panel(app, win):
    dock = win.docks["sources"]
    win.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
    pump(app, 0.1)
    assert win.stripes.side_of(dock) == "right"
    assert win.stripes.buttons[dock].parent() is win.stripes.bars["right"]
    assert "sağa" in dock.titleBarWidget().close_btn.toolTip()
    win.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)
    pump(app, 0.1)
    assert win.stripes.side_of(dock) == "left" and "sola" in dock.titleBarWidget().close_btn.toolTip()


def test_hidden_tabbed_panels_leave_no_stray_tab_bar(app, win):
    """Sekmeli iki panel birlikte gizlenince Qt'nin sekme çubuğu eski konumunda görünür kalmasın."""
    d = win.docks
    win.tabifyDockWidget(d["stats"], d["log"])
    d["stats"].show()
    d["log"].show()
    pump(app, 0.1)
    d["stats"].hide()
    d["log"].hide()
    pump(app, 0.2)
    shown = {dock.windowTitle() for dock in d.values() if dock.isVisible()}
    for bar in win.findChildren(QTabBar, options=Qt.FindChildOption.FindDirectChildrenOnly):
        if bar.isVisible() and bar.count():                       # sekmesiz olanlar Qt'nin ekran dışı yedekleri
            assert {bar.tabText(i) for i in range(bar.count())} & shown


@pytest.mark.parametrize("width, level, texts", [(1920, 0, True), (1600, 1, True), (1366, 2, False)])
def test_header_fits_without_overflow(app, win, width, level, texts):
    """Yedi görünüm düğmesi (Bağlantı dahil) üst çubuğa sığar; sağdaki akış düğmeleri taşma menüsüne düşmez."""
    win.resize(width, 900)
    pump(app, 0.2)
    bar = win.findChild(QToolBar, "HeaderBar")
    ext = bar.findChild(QToolButton, "qt_toolbar_ext_button")
    assert win._compact == level and not (ext is not None and ext.isVisible())
    style = win._nav_pills[0].toolButtonStyle()
    assert (style == Qt.ToolButtonStyle.ToolButtonTextBesideIcon) is texts


def test_log_panel_hides_to_right_stripe(app, win):
    """Günlük alt panel değil: sağ kenarda (İşlem Zinciri / Ölçümler ile sekmeli), sağ şeritten gizlenir / açılır."""
    log = win.docks["log"]
    assert win.dockWidgetArea(log) == Qt.DockWidgetArea.RightDockWidgetArea
    assert log in win.tabifiedDockWidgets(win.docks["chain"])
    btn = win.stripes.buttons[log]
    assert btn.parent() is win.stripes.bars["right"] and "sağa" in log.titleBarWidget().close_btn.toolTip()
    win.docks["chain"].raise_()                                   # Günlük sekmede arkada (açık ama görünmüyor)
    pump(app, 0.1)
    from spt.gui.widgets.side_stripe import behind_tab
    assert not log.isHidden() and behind_tab(log)
    QTest.mouseClick(btn, Qt.MouseButton.LeftButton)              # arkadaki sekme: gizlenmez, öne gelir
    pump(app, 0.1)
    assert not behind_tab(log) and not log.isHidden() and btn.isChecked()
    log.titleBarWidget().close_btn.click()
    pump(app, 0.1)
    assert not log.isVisible() and not btn.isChecked()


def test_middle_button_drag_pans_in_zoom_mode(app, win):
    """Orta tuşla basılı sürükleme her grafikte kaydırır; Canlı İzleme büyüteç (dikdörtgen yakınlaştırma) modundayken de.
    Sol tuşla sürükleme o modda yakınlaştırma olarak kalır."""
    import pyqtgraph as pg
    from PyQt6.QtCore import QEvent, QPointF
    from PyQt6.QtGui import QMouseEvent

    pw = pg.PlotWidget()
    pw.resize(400, 300)
    pw.show()
    pw.plot([0, 1, 2, 3], [0, 1, 0, 1])
    vb = pw.getPlotItem().getViewBox()
    vb.setMouseMode(pg.ViewBox.RectMode)
    vb.setRange(xRange=(0, 3), yRange=(0, 1), padding=0)
    vb.disableAutoRange()
    pump(app, 0.1)
    vp = pw.viewport()

    def drag(button):
        start = QPointF(pw.mapFromScene(vb.sceneBoundingRect().center()))

        def send(kind, pos, buttons, btn):
            QApplication.sendEvent(vp, QMouseEvent(kind, pos, QPointF(vp.mapToGlobal(pos.toPoint())), btn, buttons,
                                                   Qt.KeyboardModifier.NoModifier))
        send(QEvent.Type.MouseButtonPress, start, button, button)
        for i in range(1, 6):
            send(QEvent.Type.MouseMove, start + QPointF(10 * i, 6 * i), button, Qt.MouseButton.NoButton)
        send(QEvent.Type.MouseButtonRelease, start + QPointF(50, 30), Qt.MouseButton.NoButton, button)
        pump(app, 0.05)

    (x0, x1), (y0, y1) = vb.viewRange()
    drag(Qt.MouseButton.MiddleButton)
    (a0, a1), (b0, b1) = vb.viewRange()
    assert a0 < x0 and b0 > y0                                    # sağa-aşağı sürükle: görünüm sola-yukarı kayar
    assert a1 - a0 == pytest.approx(x1 - x0) and b1 - b0 == pytest.approx(y1 - y0)   # yakınlaşma yok
    assert vb.state["mouseMode"] == pg.ViewBox.RectMode           # mod korunur
    drag(Qt.MouseButton.LeftButton)
    (c0, c1), _ = vb.viewRange()
    assert c1 - c0 < (a1 - a0) * 0.5                              # sol tuş: dikdörtgen yakınlaştırma
    pw.close()
    pw.deleteLater()
    lv = win.live_view
    assert lv.time_plot.getPlotItem().getViewBox().state["mouseMode"] == pg.ViewBox.RectMode
