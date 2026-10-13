"""Kaynaklar paneli: bağlantı seçimi (Ethernet / UDP, USB sanal COM, CAN-bus), kayıt oynatma kontrolleri ve akış /
kanal ağacı."""
from __future__ import annotations

from PyQt6.QtCore import QRegularExpression, QSize, Qt, pyqtSignal
from PyQt6.QtGui import QAction, QRegularExpressionValidator
from PyQt6.QtWidgets import (
    QCheckBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLineEdit,
    QMenu,
    QPushButton,
    QSlider,
    QSpinBox,
    QStackedLayout,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from spt.gui.icons import icon, primary_icon, swatch_icon
from spt.gui.theme import ThemeManager
from spt.gui.widgets.common import combo, icon_button, label
from spt.protocol.format import UDP_PORT
from spt.sources.stream_receiver import (
    CAN_BITRATES,
    CAN_DATA_BITRATES,
    CAN_INTERFACES,
    SERIAL_BAUDS,
    CanConfig,
    SerialConfig,
    list_serial_ports,
)

TRANSPORTS = (("udp", "Ethernet (UDP)", "mdi6.lan"), ("usb", "USB (sanal COM)", "mdi6.usb-port"),
              ("can", "CAN-bus", "mdi6.transit-connection-horizontal"))
SOURCE_ICONS = {"USB": "mdi6.usb-port", "CAN": "mdi6.transit-connection-horizontal"}
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
    color_requested = pyqtSignal(str, int)        # kanal satırında sağ tık → "Rengi değiştir…"

    def __init__(self, theme: ThemeManager, listen_action: QAction, open_action: QAction, simulator_action: QAction):
        super().__init__()
        self._theme = theme
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(10)

        self.transport_combo = combo([])
        for key, text, name in TRANSPORTS:
            self.transport_combo.addItem(icon(name, theme.tokens.text_muted), text, key)
        self.transport_combo.setToolTip("Kartın SPTF çerçevelerini gönderdiği bağlantı: Ethernet (UDP), USB sanal COM "
                                        "portu (STM32 USB CDC ya da ST-LINK sanal COM) veya CAN-bus adaptörü")
        self.transport_combo.currentIndexChanged.connect(self._on_transport)
        top = QGridLayout()
        top.setHorizontalSpacing(8)
        top.addWidget(label("Bağlantı", "FormLabel"), 0, 0)
        top.addWidget(self.transport_combo, 0, 1)
        top.setColumnStretch(1, 1)
        lay.addLayout(top)

        self._pages = {"udp": self._udp_page(), "usb": self._usb_page(), "can": self._can_page()}
        for page in self._pages.values():
            lay.addWidget(page)
        self._on_transport()

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
        self.tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._tree_menu)
        self._stack.addWidget(self.tree)
        lay.addWidget(holder, 1)

        self._sources: dict[str, QTreeWidgetItem] = {}
        self._streams: dict[str, QTreeWidgetItem] = {}

    # ---------------------------------------------------------------- bağlantı sayfaları
    @staticmethod
    def _page() -> tuple[QWidget, QGridLayout]:
        page = QWidget()
        grid = QGridLayout(page)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(6)
        return page, grid

    def _udp_page(self) -> QWidget:
        page, grid = self._page()
        self.address = QLineEdit("0.0.0.0")
        self.address.setToolTip("Dinlenecek yerel adres (0.0.0.0 = tüm ağ arayüzleri, 127.0.0.1 = yalnızca bu bilgisayar)."
                                "\nCihaz SPTF çerçevelerini bu bilgisayarın IP adresine UDP ile gönderir.")
        self.port = QSpinBox()
        self.port.setRange(1, 65535)
        self.port.setValue(UDP_PORT)
        self.port.setToolTip(f"UDP dinleme portu (varsayılan {UDP_PORT})")
        grid.addWidget(label("UDP adresi", "FormLabel"), 0, 0)
        grid.addWidget(label("Port", "FormLabel"), 0, 1)
        grid.addWidget(self.address, 1, 0)
        grid.addWidget(self.port, 1, 1)
        grid.setColumnStretch(0, 1)
        return page

    def _usb_page(self) -> QWidget:
        page, grid = self._page()
        self.serial_port = combo([])
        self.serial_port.setEditable(True)
        self.serial_port.setToolTip("Kartın sanal COM portu. STM32 USB CDC ve ST-LINK sanal COM aygıtları listenin "
                                    "başındadır. Listede yoksa adı yazılabilir (ör. COM12).")
        self.serial_port.currentIndexChanged.connect(lambda _i: self.serial_port.lineEdit().setCursorPosition(0))
        self.refresh_button = icon_button("ph.arrows-clockwise", "Port listesini yenile", object_name="FlatIcon", size=16)
        self.refresh_button.clicked.connect(self.refresh_ports)
        self.baud = combo([str(b) for b in SERIAL_BAUDS], SERIAL_BAUDS.index(115200))
        self.baud.setEditable(True)
        self.baud.lineEdit().setValidator(QRegularExpressionValidator(QRegularExpression("[0-9]{1,8}")))
        self.baud.setToolTip("Seri hız (bit/s). STM32 USB CDC'de önemsizdir (USB hızında akar); ST-LINK sanal COM ya da "
                             "USB-UART köprüsünde kartın UART hızıyla aynı olmalıdır.")
        grid.addWidget(label("USB portu", "FormLabel"), 0, 0, 1, 2)
        grid.addWidget(self.serial_port, 1, 0)
        grid.addWidget(self.refresh_button, 1, 1)
        grid.addWidget(label("Hız (baud)", "FormLabel"), 2, 0, 1, 2)
        grid.addWidget(self.baud, 3, 0, 1, 2)
        grid.setColumnStretch(0, 1)
        return page

    def _can_page(self) -> QWidget:
        page, grid = self._page()
        self.can_interface = combo([])
        for key, text, _example in CAN_INTERFACES:
            self.can_interface.addItem(text, key)
        self.can_interface.setToolTip("PC'deki CAN adaptörü (sürücüsü kurulu olmalı): PCAN = PEAK PCAN-USB, "
                                      "slcan = CANable / USBtin gibi seri hat adaptörleri")
        self.can_channel = QLineEdit()
        self.can_channel.setToolTip("Adaptörün kanalı: PCAN'de PCAN_USBBUS1, Kvaser / Vector / IXXAT'ta 0, 1 …, "
                                    "slcan'da COM portu")
        self.can_interface.currentIndexChanged.connect(self._on_can_interface)
        self.can_bitrate = combo([f"{b // 1000} kbit/s" for b in CAN_BITRATES], CAN_BITRATES.index(500_000))
        self.can_bitrate.setToolTip("Hattın (nominal) bit hızı; karttaki CAN ayarıyla aynı olmalı")
        self.can_fd = QCheckBox("CAN FD")
        self.can_fd.setToolTip("CAN FD: mesaj başına 64 bayta kadar veri ve daha yüksek veri hızı")
        self.can_data_bitrate = combo([f"{b / 1e6:g} Mbit/s" for b in CAN_DATA_BITRATES], 1)
        self.can_data_bitrate.setToolTip("CAN FD veri fazı bit hızı (bit hızı anahtarlaması, BRS)")
        self.can_fd.toggled.connect(self.can_data_bitrate.setEnabled)
        self.can_data_bitrate.setEnabled(False)
        hexa = QRegularExpressionValidator(QRegularExpression("(0[xX])?[0-9A-Fa-f]{1,8}"))
        self.can_id = QLineEdit("0x6F0")
        self.can_mask = QLineEdit("0x7F0")
        for w in (self.can_id, self.can_mask):
            w.setValidator(hexa)
        self.can_id.setToolTip("Kartın SPTF gönderdiği CAN kimliği. Kabul: (kimlik & maske) = (bu değer & maske); "
                               "her kimlik ayrı bir gönderici sayılır")
        self.can_mask.setToolTip("Kimlik maskesi: 0x7F0 ile 0x6F0 … 0x6FF arası 16 kimlik kabul edilir; 0x7FF tek kimlik")
        self.can_extended = QCheckBox("29 bit kimlik")
        self.can_extended.setToolTip("Genişletilmiş (29 bit) CAN kimliği; kapalıyken standart 11 bit")
        grid.addWidget(label("Adaptör", "FormLabel"), 0, 0)
        grid.addWidget(label("Kanal", "FormLabel"), 0, 1)
        grid.addWidget(self.can_interface, 1, 0)
        grid.addWidget(self.can_channel, 1, 1)
        grid.addWidget(label("Bit hızı", "FormLabel"), 2, 0)
        grid.addWidget(self.can_fd, 2, 1)
        grid.addWidget(self.can_bitrate, 3, 0)
        grid.addWidget(self.can_data_bitrate, 3, 1)
        grid.addWidget(label("Kimlik (ID)", "FormLabel"), 4, 0)
        grid.addWidget(label("Maske", "FormLabel"), 4, 1)
        grid.addWidget(self.can_id, 5, 0)
        grid.addWidget(self.can_mask, 5, 1)
        grid.addWidget(self.can_extended, 6, 0, 1, 2)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        self._on_can_interface()
        return page

    def _on_transport(self, *_args) -> None:
        kind = self.transport()
        for key, page in self._pages.items():
            page.setVisible(key == kind)
        if kind == "usb" and not self.serial_port.count():
            self.refresh_ports()

    def _on_can_interface(self, *_args) -> None:
        example = CAN_INTERFACES[max(self.can_interface.currentIndex(), 0)][2]
        if self.can_channel.text().strip() in ("", *(item[2] for item in CAN_INTERFACES)):
            self.can_channel.setText(example)
        self.can_channel.setPlaceholderText(example)

    def refresh_ports(self) -> None:
        current = self.serial_port_name()
        self.serial_port.blockSignals(True)
        self.serial_port.clear()
        for device, text, stm32 in list_serial_ports():
            self.serial_port.addItem(text if not stm32 or "STM" in text.upper() else f"{text} (STM32)", device)
        index = self.serial_port.findData(current) if current else -1
        if index >= 0:
            self.serial_port.setCurrentIndex(index)
        elif current:
            self.serial_port.setEditText(current)
        self.serial_port.blockSignals(False)
        self.serial_port.lineEdit().setCursorPosition(0)

    # ---------------------------------------------------------------- bağlantı seçimi
    def transport(self) -> str:
        return self.transport_combo.currentData() or "udp"

    def set_transport(self, kind: str) -> None:
        index = self.transport_combo.findData(kind)
        if index >= 0:
            self.transport_combo.setCurrentIndex(index)

    def serial_port_name(self) -> str:
        """Seçilen ya da yazılan port (liste öğesinde aygıt adı, ör. "COM7 — STM32 Virtual ComPort" → "COM7")."""
        text = self.serial_port.currentText().strip()
        index = self.serial_port.currentIndex()
        if index >= 0 and self.serial_port.itemText(index) == text:
            return self.serial_port.itemData(index) or text
        return text

    def serial_config(self, ring_seconds: float, max_ring_mb: int) -> SerialConfig:
        port = self.serial_port_name()
        if not port:
            raise ValueError("USB portu seçilmedi. Kartı takıp port listesini yenileyin ya da port adını yazın.")
        try:
            baud = int(self.baud.currentText())
        except ValueError:
            raise ValueError("Geçersiz seri hız") from None
        return SerialConfig(port, baud, ring_seconds, max_ring_mb)

    def set_serial(self, port: str, baud: int) -> None:
        index = self.serial_port.findData(port) if port else -1
        if index >= 0:
            self.serial_port.setCurrentIndex(index)
        elif port:
            self.serial_port.setEditText(port)
        self.baud.setEditText(str(baud))

    def can_config(self, ring_seconds: float, max_ring_mb: int) -> CanConfig:
        extended = self.can_extended.isChecked()
        limit = 0x1FFFFFFF if extended else 0x7FF
        try:
            can_id, mask = int(self.can_id.text().strip(), 16), int(self.can_mask.text().strip(), 16)
        except ValueError:
            raise ValueError("CAN kimliği ve maskesi onaltılık sayı olmalı (ör. 0x6F0)") from None
        if can_id > limit or mask > limit:
            raise ValueError(f"CAN kimliği / maskesi {'29' if extended else '11'} bit sınırını aşıyor "
                             f"(en fazla 0x{limit:X})")
        channel = self.can_channel.text().strip()
        if not channel:
            raise ValueError("CAN adaptörünün kanalı boş")
        return CanConfig(self.can_interface.currentData(), channel, CAN_BITRATES[self.can_bitrate.currentIndex()],
                         self.can_fd.isChecked(), CAN_DATA_BITRATES[self.can_data_bitrate.currentIndex()],
                         can_id, mask, extended, ring_seconds, max_ring_mb)

    def set_can(self, cfg: CanConfig) -> None:
        index = self.can_interface.findData(cfg.interface)
        self.can_interface.setCurrentIndex(max(index, 0))
        self.can_channel.setText(cfg.channel)
        if cfg.bitrate in CAN_BITRATES:
            self.can_bitrate.setCurrentIndex(CAN_BITRATES.index(cfg.bitrate))
        if cfg.data_bitrate in CAN_DATA_BITRATES:
            self.can_data_bitrate.setCurrentIndex(CAN_DATA_BITRATES.index(cfg.data_bitrate))
        self.can_fd.setChecked(cfg.fd)
        self.can_extended.setChecked(cfg.extended)
        width = 8 if cfg.extended else 3
        self.can_id.setText(f"0x{cfg.can_id:0{width}X}")
        self.can_mask.setText(f"0x{cfg.can_mask:0{width}X}")

    def set_listening(self, active: bool) -> None:
        self.listen_button.setText("  Dinlemeyi Durdur" if active else "  Dinlemeyi Başlat")
        self.listen_button.setIcon(primary_icon("ph.stop" if active else "ph.broadcast"))
        self.transport_combo.setEnabled(not active)
        for page in self._pages.values():
            page.setEnabled(not active)

    # ---------------------------------------------------------------- akış ağacı
    def add_stream(self, handle, colors: list[str], visible: list[bool]) -> None:
        self.remove_stream(handle.key)
        muted = self.tree.palette().placeholderText()
        src = self._sources.get(handle.source)
        if src is None:
            src = QTreeWidgetItem([handle.source, ""])
            name = "ph.film-strip" if handle.source.startswith("Kayıt") else SOURCE_ICONS.get(
                handle.source.split(" ", 1)[0], "ph.broadcast")
            src.setIcon(0, icon(name, self._theme.tokens.text_muted))
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

    def set_channel_color(self, key: str, channel: int, color: str) -> None:
        stream = self._streams.get(key)
        if stream is not None and 0 <= channel < stream.childCount():
            stream.child(channel).setIcon(0, swatch_icon(color))

    def _tree_menu(self, pos) -> None:
        item = self.tree.itemAt(pos)
        data = item.data(0, _KEY_ROLE) if item is not None else None
        if not isinstance(data, tuple):
            return
        menu = QMenu(self.tree)
        menu.addAction("Rengi değiştir…", lambda: self.color_requested.emit(data[0], data[1]))
        menu.exec(self.tree.viewport().mapToGlobal(pos))

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
