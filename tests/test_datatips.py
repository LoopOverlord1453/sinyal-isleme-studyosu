"""Veri ipucu yönü (MATLAB "Location" benzeri): dört çeyrek, kenarda otomatik çevirme, sürükleyerek ve sağ tık
menüsüyle yön değiştirme, piksel olarak sabit uzaklık ve son seçilen yönün varsayılan olması.

Ayarlar geçici bir INI dosyasında tutulur (SPT_SETTINGS_FILE); kullanıcının kayıtlı ayarlarına dokunulmaz.
"""
import os
import time

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QPoint, QPointF, Qt  # noqa: E402
from PyQt6.QtTest import QTest  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

GAP_TOL = 1.5   # px


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
def settings(tmp_path, monkeypatch):
    monkeypatch.setenv("SPT_SETTINGS_FILE", str(tmp_path / "settings.ini"))
    from spt.gui.settings import app_settings
    return app_settings()


@pytest.fixture
def plot(app, theme, settings):
    """600×400 grafik, x 0 … 10, y −1.2 … 1.2, tek sinüs eğrisi ve veri ipuçları."""
    from spt.gui.widgets.inspector import DataTips, Trace
    from spt.gui.widgets.plot import make_plot

    pw = make_plot(theme, "Zaman", "Genlik", x_units="s")
    pw.resize(600, 400)
    pw.show()
    x = np.linspace(0, 10, 2001)
    curve = pw.plot(x, np.sin(x), pen="#2877d3")
    pw.setXRange(0, 10, padding=0)
    pw.setYRange(-1.2, 1.2, padding=0)
    pump(app, 0.1)
    trace = Trace(curve, "sin", "#2877d3", lambda x, y: [f"x = {x:.3f}", f"y = {y:.3f}"])
    tips = DataTips(pw, theme, lambda: [trace])
    yield pw, tips, trace
    tips.clear()
    pw.close()
    pw.deleteLater()
    pump(app, 0.05)


def pump(app, seconds):
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        app.processEvents()
        time.sleep(0.005)


def geometry(pw, tips, i=0):
    """(işaretin sahne konumu, etiketin sahne dikdörtgeni)."""
    tip = tips._tips[i]
    return pw.getPlotItem().getViewBox().mapViewToScene(tip.point), tip.label.sceneBoundingRect()


def assert_quadrant(marker, rect, orientation):
    """Etiket kutusu işaretin doğru çeyreğinde ve köşesi işaretten TIP_GAP_PX uzakta."""
    from spt.gui.widgets.inspector import TIP_GAP_PX
    north, east = orientation[0] == "n", orientation[1] == "e"
    near_x = rect.left() - marker.x() if east else marker.x() - rect.right()
    near_y = marker.y() - rect.bottom() if north else rect.top() - marker.y()
    assert abs(near_x - TIP_GAP_PX) <= GAP_TOL, (orientation, near_x)
    assert abs(near_y - TIP_GAP_PX) <= GAP_TOL, (orientation, near_y)


def scene_of(pw, x, y):
    return pw.getPlotItem().getViewBox().mapViewToScene(QPointF(x, y))


def test_four_orientations_and_pixel_constant_offset(app, plot):
    pw, tips, trace = plot
    tips.add(trace, 5.0, 0.0)                                  # grafiğin ortası: her yöne yer var
    assert tips.orientations() == ["ne"] and tips.shown_orientations() == ["ne"]   # varsayılan: sağ üst
    for key in ("ne", "nw", "se", "sw"):
        tips.set_orientation(0, key)
        assert tips.orientations() == [key] and tips.shown_orientations() == [key]
        assert_quadrant(*geometry(pw, tips), key)
    # yakınlaştırma / kaydırma: etiket işaretten piksel olarak aynı uzaklıkta kalır
    pw.setXRange(4.0, 6.0, padding=0)
    pw.setYRange(-1.5, 0.5, padding=0)
    pump(app, 0.05)
    assert_quadrant(*geometry(pw, tips), "sw")
    pw.setXRange(4.5, 5.5, padding=0)
    pump(app, 0.05)
    assert_quadrant(*geometry(pw, tips), "sw")


@pytest.mark.parametrize("x, y, preferred, shown", [
    (9.95, 0.0, "ne", "nw"),     # sağ kenar: sola çevrilir
    (0.05, 0.0, "sw", "se"),     # sol kenar: sağa
    (5.0, 1.15, "ne", "se"),     # üst kenar: alta
    (5.0, -1.15, "sw", "nw"),    # alt kenar: üste
    (9.95, 1.15, "ne", "sw"),    # sağ üst köşe: iki eksende
])
def test_edge_flips_shown_orientation_but_keeps_preference(app, plot, x, y, preferred, shown):
    pw, tips, trace = plot
    tips.add(trace, x, y, orientation=preferred)
    assert tips.orientations() == [preferred] and tips.shown_orientations() == [shown]
    marker, rect = geometry(pw, tips)
    assert_quadrant(marker, rect, shown)
    area = pw.getPlotItem().getViewBox().sceneBoundingRect()
    assert area.contains(rect), (area, rect)   # etiket grafiğin içinde
    # nokta ortaya gelince tercih edilen yön geri döner
    pw.setXRange(x - 5, x + 5, padding=0)
    pw.setYRange(y - 1.2, y + 1.2, padding=0)
    pump(app, 0.05)
    assert tips.shown_orientations() == [preferred]


def test_drag_snaps_to_cursor_quadrant_and_becomes_default(app, plot, settings):
    from spt.gui.widgets.inspector import ORIENTATION_KEY

    pw, tips, trace = plot
    tips.add(trace, 5.0, 0.0)
    marker, _ = geometry(pw, tips)
    label = tips._tips[0].label
    for dx, dy, key in ((-40, -30, "nw"), (-40, 30, "sw"), (40, 30, "se"), (40, -30, "ne")):
        tips._drag(label, QPointF(marker.x() + dx, marker.y() + dy), finish=False)
        assert tips.orientations() == [key]
        assert_quadrant(*geometry(pw, tips), key)
    assert settings.value(ORIENTATION_KEY) is None            # sürükleme bitmeden kaydedilmez
    tips._drag(label, QPointF(marker.x() - 40, marker.y() + 30), finish=True)
    assert tips.orientations() == ["sw"] and settings.value(ORIENTATION_KEY) == "sw"
    tips.add(trace, 3.0, float(np.sin(3.0)))                  # yeni ipucu son seçilen yönle
    assert tips.orientations() == ["sw", "sw"]


def test_mouse_drag_on_label_changes_orientation_without_panning_or_adding(app, plot):
    pw, tips, trace = plot
    tips.add(trace, 5.0, 0.0)
    marker, rect = geometry(pw, tips)
    view = pw.viewport()
    start = pw.mapFromScene(rect.center())
    target = pw.mapFromScene(QPointF(marker.x() - 60, marker.y() + 40))   # işaretin sol altı
    before = pw.getPlotItem().getViewBox().viewRange()
    QTest.mouseMove(view, start)
    pump(app, 0.02)
    QTest.mousePress(view, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, start)
    for k in range(1, 11):
        p = start + (target - start) * k / 10
        QTest.mouseMove(view, QPoint(int(p.x()), int(p.y())))
        pump(app, 0.01)
    QTest.mouseRelease(view, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, target)
    pump(app, QApplication.doubleClickInterval() / 1000 + 0.1)
    assert tips.orientations() == ["sw"]
    after = pw.getPlotItem().getViewBox().viewRange()
    assert np.allclose(before, after)                          # grafik kaydırılmadı
    assert tips.count == 1                                     # sürükleme yeni ipucu eklemedi


def test_clicks_on_label_and_marker(app, plot):
    pw, tips, trace = plot
    tips.add(trace, 5.0, 0.0)
    marker, rect = geometry(pw, tips)
    tips._pending = QPointF(rect.center())                     # etikete tık: ne ekler ne kaldırır
    tips._commit_click()
    assert tips.count == 1
    tips._pending = QPointF(marker)                            # işarete tık: kaldırır
    tips._commit_click()
    assert tips.count == 0
    tips._pending = scene_of(pw, 2.0, float(np.sin(2.0)))      # boş yere (eğri yakınına) tık: ekler
    tips._commit_click()
    assert tips.count == 1


def test_context_menu_orientation_remove_and_clear(app, plot, settings):
    from spt.gui.widgets.inspector import ORIENTATION_KEY

    pw, tips, trace = plot
    tips.add(trace, 4.0, float(np.sin(4.0)))
    tips.add(trace, 6.0, float(np.sin(6.0)))
    menu = tips.menu_for(1)
    texts = [a.text() for a in menu.actions() if a.text()]
    assert texts == ["Konum", "Bu ipucunu kaldır", "Tüm ipuçlarını temizle"]
    where = menu.actions()[0].menu()
    options = {a.data(): a for a in where.actions()}
    assert list(options) == ["ne", "nw", "se", "sw"] and options["ne"].isChecked()
    assert [a.text() for a in where.actions()] == ["Sağ üst", "Sol üst", "Sağ alt", "Sol alt"]
    options["se"].trigger()
    assert tips.orientations() == ["ne", "se"] and settings.value(ORIENTATION_KEY) == "se"
    assert_quadrant(*geometry(pw, tips, 1), "se")
    assert tips.menu_for(1).actions()[0].menu().actions()[2].isChecked()   # yeni menüde "Sağ alt" seçili
    next(a for a in tips.menu_for(0).actions() if a.text() == "Bu ipucunu kaldır").trigger()
    assert tips.count == 1 and tips.orientations() == ["se"]
    next(a for a in tips.menu_for(0).actions() if a.text() == "Tüm ipuçlarını temizle").trigger()
    assert tips.count == 0


def test_right_click_on_label_opens_tip_menu_not_view_menu(app, plot):
    pw, tips, trace = plot
    tips.add(trace, 5.0, 0.0)
    _, rect = geometry(pw, tips)
    pos = pw.mapFromScene(rect.center())
    QTest.mouseMove(pw.viewport(), pos)
    pump(app, 0.02)
    QTest.mouseClick(pw.viewport(), Qt.MouseButton.RightButton, Qt.KeyboardModifier.NoModifier, pos)
    pump(app, 0.1)
    assert tips._menu is not None and tips._menu.isVisible()
    assert not pw.getPlotItem().getViewBox().menu.isVisible()
    tips._menu.close()
    assert tips.count == 1


def test_saved_orientation_validates_setting(settings):
    from spt.gui.widgets.inspector import ORIENTATION_KEY, saved_orientation

    assert saved_orientation() == "ne"
    settings.setValue(ORIENTATION_KEY, "nw")
    assert saved_orientation() == "nw"
    settings.setValue(ORIENTATION_KEY, "yukarı")               # bozuk değer: varsayılan
    assert saved_orientation() == "ne"


def test_log_x_plot_keeps_quadrant(app, plot):
    pw, tips, trace = plot
    pw.getPlotItem().setLogMode(x=True)
    pw.setXRange(-1, 1, padding=0)
    pump(app, 0.05)
    tips.add(trace, 3.0, float(np.sin(3.0)), orientation="nw")
    assert tips._tips[0].point.x() == pytest.approx(np.log10(3.0))   # işaret log10 x'te
    assert_quadrant(*geometry(pw, tips), "nw")
