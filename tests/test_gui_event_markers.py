"""Canlı İzleme: cihazdan gelen SPTF MARKER'ları zaman grafiğinde etiketli dikey çizgi olarak gösterilir."""
import os
from types import SimpleNamespace

import pyqtgraph as pg
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication  # noqa: E402


@pytest.fixture(scope="module")
def view():
    app = QApplication.instance() or QApplication([])
    from spt.gui.theme import ThemeManager
    from spt.gui.views.live_view import LiveView

    theme = ThemeManager(app)
    theme.apply()
    v = LiveView(theme)
    yield v
    v.deleteLater()


def _lines(v):
    return [i for i in v.time_plot.getPlotItem().items if isinstance(i, pg.InfiniteLine) and i.angle == 90
            and i in v._event_lines.get("k", [])]


def test_marker_line_position_label_and_cleanup(view):
    from spt.gui.views import live_view as lv

    view._streams["k"] = SimpleNamespace(t0=1.0, fs=100.0)
    view.add_event_marker("k", 1050, 7, "durum = RUN", 50)
    view.add_event_marker("k", 0, 9, "", None)              # konumu bilinmeyen işaret çizilmez
    view.add_event_marker("yok", 0, 1, "x", 3)               # bilinmeyen akış yok sayılır
    (line,) = _lines(view)
    assert line.value() == pytest.approx(1.5)                # t0 + 50 / fs
    assert line.label.toPlainText() == "durum = RUN"
    assert "kod 7" in line.toolTip() and "örnek 1050" in line.toolTip()
    view.add_event_marker("k", 1, 3, "", 60)
    assert view._event_lines["k"][-1].label.toPlainText() == "#3"   # metinsiz işarette kod gösterilir
    for i in range(lv.MAX_EVENT_LINES + 5):
        view.add_event_marker("k", i, 1, "x" * 60, i)
    assert len(view._event_lines["k"]) == lv.MAX_EVENT_LINES        # eskiler silinir
    assert len(view._event_lines["k"][-1].label.toPlainText()) == lv.EVENT_LABEL_CHARS
    view.remove_stream("k")
    assert "k" not in view._event_lines
    assert not [i for i in view.time_plot.getPlotItem().items if isinstance(i, pg.InfiniteLine)
                and getattr(i, "label", None) is not None and i.label.toPlainText().startswith("x")]
