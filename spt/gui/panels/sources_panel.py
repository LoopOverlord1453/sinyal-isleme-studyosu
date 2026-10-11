"""Kaynaklar paneli: UDP dinleme ayarları, kayıt oynatma kontrolleri ve akış / kanal ağacı."""
from __future__ import annotations

from PyQt6.QtCore import QSize, Qt, pyqtSignal
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QCheckBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QSlider,
    QSpinBox,
    QStackedLayout,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from spt.gui.icons import icon, swatch_icon
from spt.gui.theme import ThemeManager
from spt.gui.widgets.common import combo, icon_button, label
from spt.protocol.format import UDP_PORT

SPEEDS = [("0.25×", 0.25), ("0.5×", 0.5), ("1×", 1.0), ("2×", 2.0), ("5×", 5.0), ("10×", 10.0), ("Maksimum", 0.0)]
_KEY_ROLE = Qt.ItemDataRole.UserRole


def format_clock(seconds: float) -> str:
    minutes, sec = divmod(max(seconds, 0.0), 60)
    return f"{int(minutes):02d}:{sec:04.1f}"


class ReplayControls(QFrame):
    command = pyqtSignal(str, object)  # ("pause"|"resume"|"seek"|"speed"|"loop"|"stop", değer)

    def __init__(self):
        super().__init__()
        self.setObjectName("Inset")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 8, 10, 8)
        lay.setSpacing(6)

        self.title = label("", "Readout")
        lay.addWidget(self.title)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 1000)
        self.slider.setCursor(Qt.CursorShape.PointingHandCursor)
        self.slider.sliderReleased.connect(lambda: self.command.emit("seek", self.slider.value() / 1000))
        lay.addWidget(self.slider)

        row = QHBoxLayout()
        row.setSpacing(6)
        self.play = icon_button("ph.pause", "Duraklat / devam et", object_name="FlatIcon", size=16)
        self.play.clicked.connect(self._toggle_pause)
        stop = icon_button("ph.stop", "Oynatmayı durdur", object_name="FlatIcon", size=16)
        stop.clicked.connect(lambda: self.command.emit("stop", None))
        self.clock = label("00:00.0 / 00:00.0", "Muted")
        for w in (self.play, stop, self.clock):
            row.addWidget(w)
        row.addStretch(1)
        lay.addLayout(row)

        row2 = QHBoxLayout()
        row2.setSpacing(8)
        self.speed = combo([s for s, _ in SPEEDS], 2)
        self.speed.setToolTip("Oynatma hızı")
        self.speed.currentIndexChanged.connect(lambda i: self.command.emit("speed", SPEEDS[i][1]))
        self.loop = QCheckBox("Döngü")
        self.loop.toggled.connect(lambda on: self.command.emit("loop", on))
        row2.addWidget(label("Hız", "FormLabel"))
        row2.addWidget(self.speed, 1)
        row2.addWidget(self.loop)
        lay.addLayout(row2)
        self._paused = False

    def start(self, name: str, speed: float, loop: bool) -> None:
        self.title.setText(f"▶  {name}")
        self.title.setToolTip(name)
        self.speed.blockSignals(True)
        self.speed.setCurrentIndex(next((i for i, (_, v) in enumerate(SPEEDS) if v == speed), 2))
        self.speed.blockSignals(False)
        self.loop.blockSignals(True)
        self.loop.setChecked(loop)
        self.loop.blockSignals(False)
        self.update_progress(0.0, 0.0, False, False)
        self.show()

    def update_progress(self, position: float, duration: float, paused: bool, finished: bool) -> None:
        if not self.slider.isSliderDown() and duration > 0:
            self.slider.setValue(int(1000 * position / duration))
        self.clock.setText(f"{format_clock(position)} / {format_clock(duration)}")
        self._paused = paused or finished
        self.play.setIcon(icon("ph.play" if self._paused else "ph.pause"))

    def _toggle_pause(self) -> None:
        self.command.emit("resume" if self._paused else "pause", None)

    def selected_speed(self) -> float:
        return SPEEDS[self.speed.currentIndex()][1]


class SourcesPanel(QWidget):
    channel_toggled = pyqtSignal(str, int, bool)  # akış anahtarı, kanal, görünür

    def __init__(self, theme: ThemeManager, listen_action: QAction, open_action: QAction, simulator_action: QAction):
        super().__init__()
        self._theme = theme
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(10)

        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(6)
        self.protocol = combo(["UDP", "TCP"])
        self.protocol.model().item(1).setEnabled(False)
        self.protocol.setToolTip("TCP desteği Aşama 5'te eklenecek")
        self.address = QLineEdit("0.0.0.0")
        self.address.setToolTip("Dinlenecek yerel adres (0.0.0.0 = tüm ağ arayüzleri, 127.0.0.1 = yalnızca bu bilgisayar)")
        self.port = QSpinBox()
        self.port.setRange(1, 65535)
        self.port.setValue(UDP_PORT)
        grid.addWidget(label("Protokol", "FormLabel"), 0, 0)
        grid.addWidget(label("Adres", "FormLabel"), 0, 1)
        grid.addWidget(label("Port", "FormLabel"), 0, 2)
        grid.addWidget(self.protocol, 1, 0)
        grid.addWidget(self.address, 1, 1)
        grid.addWidget(self.port, 1, 2)
        grid.setColumnStretch(1, 1)
        lay.addLayout(grid)

        self.listen_button = QPushButton()
        self.listen_button.setObjectName("Primary")
        self.listen_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.listen_button.clicked.connect(listen_action.trigger)
        lay.addWidget(self.listen_button)
        self.set_listening(False)

        row = QHBoxLayout()
        row.setSpacing(8)
        for action, text in ((open_action, "Kayıt Aç…"), (simulator_action, "Simülatör…")):
            btn = QPushButton(action.icon(), f"  {text}")
            btn.setIconSize(QSize(16, 16))
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.clicked.connect(action.trigger)
            row.addWidget(btn)
        lay.addLayout(row)

        self.replay = ReplayControls()
        self.replay.hide()
        lay.addWidget(self.replay)

        lay.addSpacing(4)
        self.streams_label = label("AKIŞLAR", "SectionLabel")
        lay.addWidget(self.streams_label)

        holder = QWidget()
        self._stack = QStackedLayout(holder)
        empty = label("Henüz akış yok.", "EmptyState")
        empty.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop)
        self._stack.addWidget(empty)
        self.tree = QTreeWidget()
        self.tree.setColumnCount(2)
        self.tree.setHeaderHidden(True)
        self.tree.setIndentation(14)
        self.tree.setUniformRowHeights(True)
        self.tree.itemChanged.connect(self._item_changed)
        self._stack.addWidget(self.tree)
        lay.addWidget(holder, 1)

        self._sources: dict[str, QTreeWidgetItem] = {}
        self._streams: dict[str, QTreeWidgetItem] = {}

    # ---------------------------------------------------------------- bağlantı
    def set_listening(self, active: bool) -> None:
        self.listen_button.setText("  Dinlemeyi Durdur" if active else "  Dinlemeyi Başlat")
        self.listen_button.setIcon(icon("ph.stop" if active else "ph.broadcast"))
        for w in (self.address, self.port):
            w.setEnabled(not active)

    # ---------------------------------------------------------------- akış ağacı
    def add_stream(self, handle, colors: list[str], visible: list[bool]) -> None:
        self.remove_stream(handle.key)
        muted = self.tree.palette().placeholderText()
        src = self._sources.get(handle.source)
        if src is None:
            src = QTreeWidgetItem([handle.source, ""])
            src.setIcon(0, icon("ph.film-strip" if handle.source.startswith("Kayıt") else "ph.broadcast",
                                self._theme.tokens.text_muted))
            self.tree.addTopLevelItem(src)
            self._sources[handle.source] = src
        desc = handle.desc
        self.tree.blockSignals(True)
        stream = QTreeWidgetItem([f"{desc.stream_id:02d} · {desc.name or 'akış'}", desc.format_text()])
        stream.setIcon(0, icon("mdi6.sine-wave", self._theme.tokens.text_muted))
        stream.setData(0, _KEY_ROLE, handle.key)
        stream.setForeground(1, muted)
        src.addChild(stream)
        for i, ch in enumerate(desc.channels):
            item = QTreeWidgetItem([ch.name or f"ch{i}", ch.unit])
            item.setIcon(0, swatch_icon(colors[i]))
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(0, Qt.CheckState.Checked if visible[i] else Qt.CheckState.Unchecked)
            item.setData(0, _KEY_ROLE, (handle.key, i))
            item.setForeground(1, muted)
            stream.addChild(item)
        self.tree.blockSignals(False)
        self._streams[handle.key] = stream
        src.setExpanded(True)
        stream.setExpanded(desc.channel_count <= 16)
        self.tree.resizeColumnToContents(0)
        self._update_empty()

    def remove_stream(self, key: str) -> None:
        stream = self._streams.pop(key, None)
        if stream is None:
            return
        src = stream.parent()
        src.removeChild(stream)
        if src.childCount() == 0:
            self.tree.takeTopLevelItem(self.tree.indexOfTopLevelItem(src))
            self._sources = {k: v for k, v in self._sources.items() if v is not src}
        self._update_empty()

    def clear_streams(self) -> None:
        self.tree.clear()
        self._sources.clear()
        self._streams.clear()
        self._update_empty()

    def _update_empty(self) -> None:
        count = len(self._streams)
        self._stack.setCurrentIndex(1 if count else 0)
        self.streams_label.setText(f"AKIŞLAR · {count}" if count else "AKIŞLAR")

    def _item_changed(self, item: QTreeWidgetItem, column: int) -> None:
        data = item.data(0, _KEY_ROLE)
        if column == 0 and isinstance(data, tuple):
            self.channel_toggled.emit(data[0], data[1], item.checkState(0) == Qt.CheckState.Checked)
