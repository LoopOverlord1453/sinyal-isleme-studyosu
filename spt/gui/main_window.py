"""Ana pencere: menüler, üst çubuk, paneller ve merkezi görünümler."""
from __future__ import annotations

import json
import logging
import math
from dataclasses import replace
from datetime import datetime
from functools import partial
from pathlib import Path

from PyQt6.QtCore import QSettings, QSize, Qt, QTimer, QUrl
from PyQt6.QtGui import (
    QActionGroup,
    QCloseEvent,
    QDesktopServices,
    QGuiApplication,
    QResizeEvent,
    QShortcut,
    QShowEvent,
)
from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QDockWidget,
    QFileDialog,
    QFrame,
    QHeaderView,
    QMainWindow,
    QMenu,
    QMessageBox,
    QSizePolicy,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QToolBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from spt import APP_NAME, __version__
from spt.core.live_filter import LiveFilter
from spt.core.snapshot import Snapshot, filter_snapshot, take_snapshot
from spt.dsp.chain import ChainStage, ProcessingChain
from spt.dsp.filters import SPECIAL_METHODS, FilterError, FilterSpec, format_hz
from spt.dsp.spectrum import ONE_SIDED, TWO_SIDED
from spt.gui.actions import ActionRegistry
from spt.gui.detach import DetachManager
from spt.gui.dialogs.connection_dialog import ConnectionDialog
from spt.gui.dialogs.export_dialog import ExportDialog
from spt.gui.dialogs.simulator_dialog import SimulatorDialog
from spt.gui.history import History
from spt.gui.icons import app_icon, icon, logo_pixmap
from spt.gui.panels.chain_panel import ChainPanel
from spt.gui.panels.log_panel import LogPanel
from spt.gui.panels.measurements_panel import MeasurementsPanel
from spt.gui.panels.properties_panel import PropertiesPanel
from spt.gui.panels.sources_panel import SourcesPanel
from spt.gui.panels.stats_panel import StatsPanel
from spt.gui.panels.workspace_panel import WorkspacePanel
from spt.gui.settings import app_settings
from spt.gui.theme import ThemeManager, style_native_titlebar
from spt.gui.views.control_view import ControlView
from spt.gui.views.filter_designer_view import FilterDesignerView
from spt.gui.views.fusion_view import FusionView
from spt.gui.views.live_view import LiveView
from spt.gui.views.siggen_view import SignalGeneratorView
from spt.gui.widgets.command_box import CommandBox
from spt.gui.widgets.common import DockTitleBar, demo_chip, hbox, icon_button, label
from spt.gui.widgets.status_bar import StatusBar
from spt.sources.manager import SourceManager
from spt.sources.receiver import ReceiverConfig

log = logging.getLogger("spt.gui")

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_DOC = ROOT / "docs" / "SPTF_Protokol_v1.md"
EMBEDDED_DOC = ROOT / "docs" / "SPTF_UDP_Kullanim_Kilavuzu.md"
SAMPLES_DIR = ROOT / "samples_signal_data"
LAYOUT_VERSION = 2   # 2: İşlem Zinciri paneli eklendi (eski kayıtlı yerleşim yerine varsayılan kurulur)
REFRESH_MS = 33      # canlı grafik ~30 FPS
HISTORY_MS = 400     # tasarımcıdaki ardışık değişiklikler tek geri alma adımında birleşir

# Menü düzeni: str = komut anahtarı, None = ayırıcı, (başlık, ikon, [...]) = alt menü,
# "@panels" = panel aç/kapa listesi.
MENU_LAYOUT = [
    ("&Dosya", [
        "file.new_project", "file.open_project", "file.save_project", None,
        ("İçe Aktar", "ph.download-simple", ["import.sptr", None, "import.csv", "import.wav", "import.npy"]),
        ("Dışa Aktar", "ph.upload-simple", ["export.csv", "export.wav", None, "export.png", None, "flt.export"]),
        None, "app.quit",
    ]),
    ("Dü&zen", ["edit.undo", "edit.redo", None, "edit.copy_plot", None, "edit.preferences"]),
    ("&Görünüm", [
        "view.live", "view.review", "view.siggen", "view.filter", "view.fusion", "view.control", None,
        ("Paneller", "ph.layout", ["@panels"]),
        ("Ayrı Pencerede Aç", "ph.app-window", ["@detach"]), None,
        "view.command_palette", "view.reset_layout", "view.fullscreen",
    ]),
    ("&Bağlantı", [
        "conn.listen", "conn.settings", None,
        "stream.run", "stream.freeze", None,
        "stream.record", "stream.replay",
    ]),
    ("&Analiz", [
        "an.fft", "an.psd", None, "view.fusion", "view.control", None,
        "an.measure", "an.cursors", None,
        "an.corr", "an.hilbert", "an.resample", "an.adc",
    ]),
    ("&Filtre", [
        "view.filter", None, "flt.iir", "flt.fir",
        ("Özel Filtreler", "ph.sliders", [
            "flt.notch", "flt.comb", "flt.dc_block", None,
            "flt.ema", "flt.kalman", "flt.alpha_beta", "flt.moving_avg", "flt.gaussian", "flt.median", "flt.savgol",
            None, "flt.differentiator", "flt.integrator", "flt.hilbert", "flt.allpass",
        ]),
        None, "flt.design", "flt.apply", None, "flt.chain", "flt.export",
    ]),
    ("A&raçlar", ["tools.generator", "tools.simulator", "tools.packet_inspector", None, "edit.preferences"]),
    ("&Yardım", ["help.docs", "help.protocol", "help.embedded", "help.shortcuts", None, "help.about"]),
]

# Sıra kayıtlı ayarlardaki görünüm indeksidir (view/index): yeni görünümler SONA eklenir
VIEW_KEYS = ("view.live", "view.siggen", "view.filter", "view.review", "view.fusion", "view.control")

# Dar pencerede (FOCUS_WIDTH altı) bu görünümlerde ilgisiz paneller geçici olarak gizlenir: 1366×768'de paneller
# genişliğin yarısını, 1600×960'ta alt paneller yüksekliğin üçte birini alıyor, grafiklere yer kalmıyordu. Görünümden
# çıkınca (ya da pencere genişleyince) geri gelirler; kullanıcı bu görünümdeyken panel menüsünden açtığı paneli o
# görünümde bir daha gizlemeyiz.
_ALL_PANELS = ("sources", "workspace", "properties", "chain", "measurements", "stats", "log")
FOCUS_DOCKS = {
    "view.siggen": _ALL_PANELS,  # üreteç akış kullanmaz
    # füzyon kaynağını (akış / anlık görüntü) kendi listesinden seçer; dinleme / oynatma üst çubukta ve Dosya menüsünde
    "view.fusion": _ALL_PANELS,
    # kontrol tasarımı kaynağını (akış / anlık görüntü) Sistem Tanımlama ve lsim girişinde kendisi seçer
    "view.control": _ALL_PANELS,
    # tasarımcı örnekleme hızını kendi "Akış" düğmesiyle alır; tasarımın uygulandığı İşlem Zinciri açık kalır
    "view.filter": ("sources", "workspace", "properties", "measurements", "stats", "log"),
}
FOCUS_WIDTH = 1800
NAV_ORDER = ("view.live", "view.review", "view.siggen", "view.filter", "view.fusion", "view.control")

# Aşama numarasına gelen bulunma eki: "Aşama 1'de", "Aşama 3'te"
_LOCATIVE = {1: "'de", 2: "'de", 3: "'te", 4: "'te", 5: "'te"}


class MainWindow(QMainWindow):
    def __init__(self, theme: ThemeManager):
        super().__init__()
        self.theme = theme
        self._titlebar_styled = False
        self.setWindowTitle(APP_NAME)
        self.setWindowIcon(app_icon())
        screen = QGuiApplication.primaryScreen()
        area = screen.availableGeometry() if screen is not None else None
        width = min(1600, int(area.width() * 0.96)) if area else 1600
        height = min(960, int(area.height() * 0.94)) if area else 960
        self.resize(width, height)
        self.setMinimumSize(1000, 640)
        self._compact = None  # dar ekran düzeni (üst çubuk ikon modunda)
        self._focus_hidden: set[str] = set()              # odak düzeni için gizlediğimiz paneller
        self._focus_raised: set[str] = set()              # gizlenirken sekme grubunda önde olanlar
        self._focus_kept: set[tuple[str, str]] = set()    # (görünüm, panel): kullanıcı açık tuttu, gizleme
        self._focus_narrow: bool | None = None
        self._seen_streams: set[str] = set()
        self._replay_ended = False
        self.setContentsMargins(8, 0, 8, 0)
        self.setDockOptions(
            QMainWindow.DockOption.AnimatedDocks
            | QMainWindow.DockOption.AllowNestedDocks
            | QMainWindow.DockOption.AllowTabbedDocks
        )
        self.setCorner(Qt.Corner.BottomLeftCorner, Qt.DockWidgetArea.LeftDockWidgetArea)
        self.setCorner(Qt.Corner.BottomRightCorner, Qt.DockWidgetArea.RightDockWidgetArea)

        self.registry = ActionRegistry(self)
        self.log_panel = LogPanel(theme.tokens)  # erken: kurulum iletileri de görünsün
        self.sources = SourceManager(self)
        self.live_filter = LiveFilter()
        self.history = History(self)
        self._restoring = False
        self._snap_seq = 0
        self._export: ExportDialog | None = None
        self._receiver_cfg = self._load_receiver_config()
        self._channel_colors: dict[tuple[str, int], str] = {}
        self._channel_visible: dict[tuple[str, int], bool] = {}
        self._simulator: SimulatorDialog | None = None
        self._editing_uid: int | None = None  # tasarımcıda düzenlenen zincir adımının kimliği
        self._window_user_set = False  # zaman penceresi kullanıcı tarafından seçildi (otomatik öneri yapılmaz)
        self.detacher = DetachManager(theme.tokens, self._settings,
                                      lambda: [a for _, a in self.registry.items()], self)

        self._build_central()
        self._build_docks()
        self._register_detachables()
        self._build_menus()
        self._build_header()
        self.status = StatusBar(theme.tokens)
        self.setStatusBar(self.status)
        self._wire_actions()
        self._wire_sources()
        self._wire_properties()
        self._wire_filters()
        self._place_docks()
        self.registry["stream.run"].setChecked(True)

        self._refresh_timer = QTimer(self)
        self._refresh_timer.setInterval(REFRESH_MS)
        self._refresh_timer.timeout.connect(self._refresh)
        self._refresh_timer.start()

        self._load_control_state()
        if not self._restore_settings():
            QTimer.singleShot(0, self._size_docks)
        QTimer.singleShot(0, self.detacher.restore_open)  # önceki oturumda ayrı pencerede olanlar
        log.info("Hazır (v%s). UDP dinlemeyi başlatın ya da samples_signal_data/ altındaki bir kaydı açın.",
                 __version__)

    # ------------------------------------------------------------------ kurulum
    def _build_central(self) -> None:
        self.views = QStackedWidget()
        self.live_view = LiveView(self.theme)
        self.siggen_view = SignalGeneratorView(self.theme)
        self.siggen_view.simulate_requested.connect(self._send_generated_signal)
        self.filter_view = FilterDesignerView(self.theme, self.registry)
        self.review_view = LiveView(self.theme, review=True)
        self.fusion_view = FusionView(self.theme)
        self.control_view = ControlView(self.theme)
        for v in (self.live_view, self.siggen_view, self.filter_view, self.review_view, self.fusion_view,
                  self.control_view):
            self.views.addWidget(v)

        host = QFrame()
        host.setObjectName("Canvas")
        lay = QVBoxLayout(host)
        lay.setContentsMargins(4, 8, 4, 0)
        lay.addWidget(self.views)
        self.setCentralWidget(host)

    def _make_dock(self, key: str, title: str, icon_name: str, panel: QWidget, demo: bool = False) -> QDockWidget:
        dock = QDockWidget(title, self)
        dock.setObjectName(f"dock_{key}")
        dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetClosable
            | QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )
        bar = DockTitleBar(dock, icon_name, on_detach=lambda: self.detacher.toggle(f"dock_{key}"))
        if demo:
            bar.add_widget(demo_chip())
        dock.setTitleBarWidget(bar)

        body = QFrame()
        body.setObjectName("DockBody")
        lay = QVBoxLayout(body)
        lay.setContentsMargins(12, 4, 12, 12)
        lay.addWidget(panel)
        dock.setWidget(body)

        toggle = dock.toggleViewAction()
        toggle.setIcon(icon(icon_name))
        toggle.triggered.connect(lambda on, k=key: self._on_dock_toggled(k, on))
        return dock

    def _build_docks(self) -> None:
        r = self.registry
        self.sources_panel = SourcesPanel(self.theme, r["conn.listen"], r["stream.replay"], r["tools.simulator"])
        self.sources_panel.address.setText(self._receiver_cfg.address)
        self.sources_panel.port.setValue(self._receiver_cfg.port)
        self.properties_panel = PropertiesPanel()
        self.stats_panel = StatsPanel()
        self.workspace_panel = WorkspacePanel(r["stream.freeze"])
        self.chain_panel = ChainPanel(self.theme.tokens)
        self.docks = {
            "sources": self._make_dock("sources", "Kaynaklar", "ph.broadcast", self.sources_panel),
            "workspace": self._make_dock("workspace", "Çalışma Alanı", "ph.stack", self.workspace_panel),
            "properties": self._make_dock("properties", "Özellikler", "ph.sliders-horizontal", self.properties_panel),
            "chain": self._make_dock("chain", "İşlem Zinciri", "ph.flow-arrow", self.chain_panel),
            "measurements": self._make_dock("measurements", "Ölçümler", "ph.ruler",
                                            MeasurementsPanel(), demo=True),
            "stats": self._make_dock("stats", "Paket İstatistikleri", "ph.table", self.stats_panel),
            "log": self._make_dock("log", "Günlük", "ph.terminal-window", self.log_panel),
        }

    def _register_detachables(self) -> None:
        """Ayrı pencerede açılabilen her şey: canlı grafikler, diğer görünümlerin grafikleri ve tüm paneller."""
        d = self.detacher
        cards = (
            ("time", "Zaman Domeni", self.live_view.time_card),
            ("spectrum", "Spektrum", self.live_view.spec_card),
            ("review_time", "İnceleme · Zaman Domeni", self.review_view.time_card),
            ("review_spectrum", "İnceleme · Spektrum", self.review_view.spec_card),
            ("siggen_time", "Sinyal Üreteci · DAC Çıkışı", self.siggen_view.time_card),
            ("siggen_spectrum", "Sinyal Üreteci · Spektrum", self.siggen_view.spec_card),
            ("siggen_metrics", "Sinyal Üreteci · Ölçümler", self.siggen_view.metrics_card),
            ("filter_mag", "Filtre · Genlik Cevabı", self.filter_view.mag_card),
            ("filter_analysis", "Filtre · Faz / Gecikme / Zaman Cevabı", self.filter_view.analysis_card),
            ("filter_pz", "Filtre · Kutup-Sıfır", self.filter_view.pz_card),
            ("fusion_angles", "Füzyon · Kestirimler", self.fusion_view.angles_card),
            ("fusion_side", "Füzyon · 3B Yönelim / Yörünge", self.fusion_view.side_card),
            ("fusion_metrics", "Füzyon · Doğruluk", self.fusion_view.metrics_card),
        )
        cards += tuple((key, f"Kontrol · {title}", card)
                       for key, title, card in self.control_view.detachable_cards())
        for key, title, card in cards:
            d.register_card(key, title, card)
        for key, dock in self.docks.items():
            d.register_dock(f"dock_{key}", dock)

    def _place_docks(self) -> None:
        d = self.docks
        for dock in d.values():
            dock.setFloating(False)
            dock.show()
        left, right, bottom = (Qt.DockWidgetArea.LeftDockWidgetArea, Qt.DockWidgetArea.RightDockWidgetArea,
                               Qt.DockWidgetArea.BottomDockWidgetArea)
        self.addDockWidget(left, d["sources"])
        self.splitDockWidget(d["sources"], d["workspace"], Qt.Orientation.Vertical)
        self.addDockWidget(right, d["properties"])
        self.splitDockWidget(d["properties"], d["chain"], Qt.Orientation.Vertical)
        self.tabifyDockWidget(d["chain"], d["measurements"])
        d["chain"].raise_()
        self.addDockWidget(bottom, d["stats"])
        if self._narrow_screen():
            self.tabifyDockWidget(d["stats"], d["log"])
            d["stats"].raise_()
        else:
            self.splitDockWidget(d["stats"], d["log"], Qt.Orientation.Horizontal)
        if self._short_screen():  # ≤ 900 px yükseklikte grafiklere yer kalsın: alt paneller kapalı başlar
            d["stats"].hide()
            d["log"].hide()

    def _narrow_screen(self) -> bool:
        screen = QGuiApplication.primaryScreen()
        return screen is not None and screen.availableGeometry().width() < 1600

    def _short_screen(self) -> bool:
        screen = QGuiApplication.primaryScreen()
        return screen is not None and screen.availableGeometry().height() < 900

    def _size_docks(self) -> None:
        d = self.docks
        h, v = Qt.Orientation.Horizontal, Qt.Orientation.Vertical
        narrow = self._narrow_screen()
        self.resizeDocks([d["sources"], d["properties"]], [260, 280] if narrow else [310, 340], h)
        self.resizeDocks([d["sources"], d["workspace"]], [520, 340], v)
        self.resizeDocks([d["properties"], d["chain"]], [470, 390], v)
        self.resizeDocks([d["stats"]], [180 if narrow else 230], v)
        if not narrow:
            self.resizeDocks([d["stats"], d["log"]], [600, 320], h)

    def _build_menus(self) -> None:
        bar = self.menuBar()
        for title, items in MENU_LAYOUT:
            self._fill_menu(bar.addMenu(title), items)

    def _fill_menu(self, menu: QMenu, items: list) -> None:
        for item in items:
            if item is None:
                menu.addSeparator()
            elif item == "@panels":
                for dock in self.docks.values():
                    menu.addAction(dock.toggleViewAction())
            elif item == "@detach":
                for kind in ("card", "dock"):  # önce grafikler, sonra paneller
                    for act in self.detacher.actions(kind):
                        menu.addAction(act)
                    menu.addSeparator()
                menu.addAction(icon("ph.arrow-square-in"), "Tümünü ana pencereye geri al", self.detacher.attach_all)
            elif isinstance(item, tuple):
                title, icon_name, sub_items = item
                sub = menu.addMenu(icon(icon_name), title)
                self._fill_menu(sub, sub_items)
            else:
                menu.addAction(self.registry[item])

    def _build_header(self) -> None:
        bar = QToolBar("Üst Çubuk")
        bar.setObjectName("HeaderBar")
        bar.setMovable(False)
        bar.setFloatable(False)
        bar.setIconSize(QSize(18, 18))
        bar.toggleViewAction().setEnabled(False)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, bar)

        logo = label("")
        logo.setPixmap(logo_pixmap(28))
        bar.addWidget(logo)
        self._wordmark = label("Signal Processing Toolbox", "Wordmark")
        self._wordmark_action = bar.addWidget(self._wordmark)
        self._nav_pills = []
        bar.addSeparator()

        self.command_box = CommandBox([a for _, a in self.registry.items()])
        bar.addWidget(self.command_box)
        bar.addSeparator()

        self._view_group = QActionGroup(self)
        self._view_group.setExclusive(True)
        for key in NAV_ORDER:
            act = self.registry[key]
            self._view_group.addAction(act)
            pill = icon_button(action=act, object_name="NavPill", size=16)
            pill.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
            self._nav_pills.append(pill)
            bar.addWidget(pill)
        self.registry["view.live"].setChecked(True)

        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        bar.addWidget(spacer)

        stream_keys = ("conn.listen", "stream.run", "stream.freeze", "stream.record")
        bar.addWidget(hbox(*(icon_button(action=self.registry[k]) for k in stream_keys), spacing=6))
        bar.addSeparator()
        bar.addWidget(icon_button(action=self.registry["edit.preferences"]))

        export_menu = QMenu(self)
        for key in ("export.csv", "export.wav", None, "export.png", "flt.export"):
            if key is None:
                export_menu.addSeparator()
            else:
                export_menu.addAction(self.registry[key])
        export = QToolButton()
        export.setObjectName("MenuButton")
        export.setText("Dışa Aktar")
        export.setIcon(icon("ph.upload-simple"))
        export.setIconSize(QSize(16, 16))
        export.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        export.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        export.setMenu(export_menu)
        export.setCursor(Qt.CursorShape.PointingHandCursor)
        bar.addWidget(export)

    def _wire_actions(self) -> None:
        handlers = {
            "app.quit": self.close,
            "view.live": lambda *_: self._show_view(0),
            "view.siggen": lambda *_: self._show_view(1),
            "tools.generator": lambda *_: self._show_view(1),
            "view.filter": lambda *_: self._show_view(2),
            "view.review": lambda *_: self._show_view(3),
            "view.fusion": lambda *_: self._show_view(4),
            "view.control": lambda *_: self._show_view(5),
            "view.command_palette": lambda *_: self.command_box.focus(),
            "view.reset_layout": lambda *_: self._reset_layout(),
            "view.fullscreen": self._set_fullscreen,
            "help.protocol": lambda *_: self._open_protocol_doc(),
            "help.embedded": lambda *_: self._open_protocol_doc(EMBEDDED_DOC),
            "help.shortcuts": lambda *_: self._show_shortcuts(),
            "help.about": lambda *_: self._show_about(),
            "conn.listen": self._toggle_listen,
            "conn.settings": lambda *_: self._edit_connection(),
            "stream.run": self._set_running,
            "stream.freeze": lambda *_: self.freeze(),
            "stream.record": self._toggle_record,
            "stream.replay": lambda *_: self._open_recording(),
            "import.sptr": lambda *_: self._open_recording(),
            "tools.simulator": lambda *_: self._show_simulator(),
            "edit.undo": lambda *_: self.undo(),
            "edit.redo": lambda *_: self.redo(),
            "flt.iir": lambda *_: self._open_designer("iir"),
            "flt.fir": lambda *_: self._open_designer("fir"),
            **{f"flt.{m}": partial(lambda m, *_: self._open_designer("special", m), m) for m in SPECIAL_METHODS},
            "flt.design": lambda *_: self._design_now(),
            "flt.apply": lambda *_: self.filter_view.request_apply(),
            "flt.chain": lambda *_: self._show_dock("chain"),
            "flt.export": lambda *_: self.show_export(),
        }
        for key, act in self.registry.items():
            handler = handlers.get(key)
            if handler is None:  # henüz uygulanmadı: menüde devre dışı görünsün, ipucu nedenini söylesin
                spec = self.registry.spec(key)
                when = f"Aşama {spec.phase}{_LOCATIVE[spec.phase]} planlandı" if spec.phase else "henüz bağlanmadı"
                act.setEnabled(False)
                act.setToolTip(f"{spec.text.rstrip('…')} — bu sürümde yok ({when})")
                act.setStatusTip(f"Bu sürümde yok ({when})")
                act.triggered.connect(partial(self._not_implemented, key))
                # Devre dışı eylemin kısayolu tetiklenmez; tuşa basan kullanıcı yine de geri bildirim alsın
                for seq in act.shortcuts():
                    QShortcut(seq, self, activated=partial(self._not_implemented, key))
            else:
                act.triggered.connect(handler)

    def _wire_sources(self) -> None:
        s = self.sources
        s.stream_added.connect(self._on_stream_added)
        s.stream_removed.connect(self._on_stream_removed)
        s.streams_cleared.connect(self._on_streams_cleared)
        s.stats_updated.connect(self._on_stats)
        s.state_changed.connect(self._on_state)
        s.replay_progress.connect(self.sources_panel.replay.update_progress)
        s.replay_progress.connect(self._on_replay_progress)
        s.recording_changed.connect(self._on_recording)
        s.marker_received.connect(self.live_view.add_event_marker)
        self.sources_panel.channel_toggled.connect(self._on_channel_toggled)
        self.sources_panel.replay.command.connect(self._on_replay_command)
        self.live_view.listen_requested.connect(self.registry["conn.listen"].trigger)
        self.live_view.open_requested.connect(self._open_recording)
        self.live_view.simulator_requested.connect(self._show_simulator)
        self.live_view.unit_changed.connect(self.properties_panel.set_unit)
        self.review_view.freeze_requested.connect(self.freeze)

    def _wire_properties(self) -> None:
        p, views = self.properties_panel, (self.live_view, self.review_view)

        def window(*_):
            value, unit = p.time_window.currentText().split()
            self.live_view.set_window(float(value) / (1000 if unit == "ms" else 1))

        def y_axis(*_):
            for v in views:
                v.set_y(p.y_mode.currentText() == "Otomatik", p.y_min.value(), p.y_max.value())

        def fft(*_):
            for v in views:
                v.set_fft(int(p.fft_size.currentText()), p.window.currentText(), p.kaiser_beta.value())

        p.time_window.currentTextChanged.connect(window)
        for w in (p.y_min, p.y_max):
            w.valueChanged.connect(y_axis)
        p.y_mode.currentTextChanged.connect(y_axis)
        for v in views:
            p.grid.toggled.connect(v.set_grid)
            p.peak_decimation.toggled.connect(v.set_peak)
        p.fft_size.currentTextChanged.connect(fft)
        p.window.currentTextChanged.connect(fft)
        p.kaiser_beta.valueChanged.connect(fft)

        # Spektrum gösterimi: Özellikler'deki seçim ile spektrum kartlarındaki düğmeler senkron
        sides = (ONE_SIDED, TWO_SIDED)

        def set_sides(value: str) -> None:
            for v in views:
                v.set_spectrum_sides(value)
            p.sides.blockSignals(True)
            p.sides.setCurrentIndex(sides.index(value))
            p.sides.blockSignals(False)

        p.sides.currentIndexChanged.connect(lambda i: set_sides(sides[i]))
        for v in views:
            v.sides_changed.connect(set_sides)
        window()
        y_axis()
        fft()

    def _wire_filters(self) -> None:
        fv, cp, ws = self.filter_view, self.chain_panel, self.workspace_panel
        self.live_view.set_processor(self.live_filter)
        fv.apply_requested.connect(self._apply_design)
        fv.spec_changed.connect(lambda: self._history_timer.start())
        fv.editing_cancelled.connect(self._on_editing_cancelled)
        cp.chain_changed.connect(self._on_chain_changed)
        cp.live_toggled.connect(self._on_live_toggled)
        cp.display_changed.connect(self.live_view.set_filter_display)
        cp.edit_requested.connect(self._edit_stage)
        ws.selection_changed.connect(self._show_snapshots)
        ws.review_requested.connect(lambda: self._show_view(3))
        ws.filter_requested.connect(self.filter_snapshots)
        ws.removed.connect(lambda *_: self._update_fusion_sources())

        self._history_timer = QTimer(self)
        self._history_timer.setSingleShot(True)
        self._history_timer.setInterval(HISTORY_MS)
        self._history_timer.timeout.connect(self._push_history)
        self.history.changed.connect(self._update_undo_actions)
        self._load_filter_state()
        self._load_display_state()
        self.history.reset(self._ui_state())

    # ------------------------------------------------------------------ kaynaklar
    def _toggle_listen(self, checked: bool) -> None:
        if not checked:
            self.sources.stop()
            return
        cfg = replace(self._receiver_cfg, address=self.sources_panel.address.text().strip() or "0.0.0.0",
                      port=self.sources_panel.port.value())
        self._receiver_cfg = cfg
        self._reset_channels()
        self.sources_panel.replay.hide()
        self.sources.start_udp(cfg)
        self.sources_panel.set_listening(True)

    def _open_recording(self) -> None:
        s = self._settings()
        start_dir = s.value("replay/dir", str(SAMPLES_DIR))
        path, _ = QFileDialog.getOpenFileName(self, "Kayıt aç", start_dir, "SPTF kayıtları (*.sptr);;Tüm dosyalar (*)")
        if not path:
            return
        s.setValue("replay/dir", str(Path(path).parent))
        self.start_replay(Path(path))

    def start_replay(self, path: Path) -> None:
        cfg = self._receiver_cfg
        controls = self.sources_panel.replay
        speed, loop = controls.selected_speed(), controls.loop.isChecked()
        self._reset_channels()
        self._replay_ended = False
        self.sources.start_replay(path, speed, loop, cfg.ring_seconds, cfg.max_ring_mb)
        self.registry["conn.listen"].setChecked(False)
        self.sources_panel.set_listening(False)
        controls.start(path.name, speed, loop)
        self._show_view(0)

    def _on_replay_command(self, name: str, value) -> None:
        if name == "stop":
            self.sources.stop()
        elif value is None:
            self.sources.send(name)
        else:
            self.sources.send(name, value)

    def _toggle_record(self, checked: bool) -> None:
        act = self.registry["stream.record"]
        if not checked:
            self.sources.send("record_stop")
            return
        if self.sources.mode != "udp":
            act.setChecked(False)
            log.warning("Kayıt yalnızca UDP dinlenirken yapılabilir; önce dinlemeyi başlatın.")
            self.status.showMessage("Kayıt için önce UDP dinlemeyi başlatın", 4000)
            return
        default = SAMPLES_DIR / f"kayit_{datetime.now():%Y%m%d_%H%M%S}.sptr"
        path, _ = QFileDialog.getSaveFileName(self, "Kaydı kaydet", str(default), "SPTF kayıtları (*.sptr)")
        if not path:
            act.setChecked(False)
            return
        self.sources.send("record_start", path)

    def _set_running(self, running: bool) -> None:
        self.live_view.set_running(running)

    def _refresh(self) -> None:
        # Canlı grafikler ayrı pencerede açıksa, ana pencerede başka görünüm seçiliyken de güncellenir.
        live_visible = (self.views.currentIndex() == 0 or self.detacher.is_detached("time")
                        or self.detacher.is_detached("spectrum"))
        if self.sources.streams and live_visible:
            self.live_view.refresh()

    def _reset_channels(self) -> None:
        self._channel_colors.clear()
        self._channel_visible.clear()

    def _on_stream_added(self, handle) -> None:
        colors, visible = [], []
        for ch in range(handle.desc.channel_count):
            key = (handle.key, ch)
            if key not in self._channel_colors:  # renk kanala bağlıdır; yeniden bağlanınca değişmez
                self._channel_colors[key] = self.theme.series_color(len(self._channel_colors))
            colors.append(self._channel_colors[key])
            visible.append(self._channel_visible.setdefault(key, True))
        first = not self.live_view._streams and handle.key not in self._seen_streams
        self._seen_streams.add(handle.key)
        self.sources_panel.add_stream(handle, colors, visible)
        if first:  # yeni bir oturumun ilk akışı (oynatma başa sarınca kullanıcının seçimi korunur)
            self._suggest_time_window(handle.fs)
        if self.live_filter.active:
            log.info("İşlem zinciri (%s) yeni akışa da uygulanıyor: %s · %s", self.live_filter.chain.describe(),
                     handle.desc.name or handle.key, format_hz(handle.fs))
        self.live_view.add_stream(handle, colors, visible)
        self._update_stream_rates()

    def _suggest_time_window(self, fs: float) -> None:
        """İlk akışta zaman penceresi ~2000 örnek gösterecek şekilde (0.1 … 5 s) seçilir: 500 Hz EKG'de 5 s
        (kalp atımları görünür), 10 kHz'de 100 ms. Kullanıcı pencereyi kendisi seçtiyse (kaydedilir) dokunulmaz."""
        if self._window_user_set:
            return
        box = self.properties_panel.time_window
        best, best_err = None, None
        for i in range(box.count()):
            value, unit = box.itemText(i).split()
            seconds = float(value) / (1000 if unit == "ms" else 1)
            if not 0.1 <= seconds <= 5.0:
                continue
            err = abs(math.log(seconds * fs / 2000))
            if best_err is None or err < best_err:
                best, best_err = i, err
        if best is not None and best != box.currentIndex():
            box.setCurrentIndex(best)

    def _on_replay_progress(self, position: float, duration: float, paused: bool, finished: bool) -> None:
        if self.sources.mode != "replay" or finished == self._replay_ended:
            return
        self._replay_ended = finished
        if finished:
            self.status.set_connection("replay_end", "Kayıt sonuna gelindi — baştan oynatmak için ▶")
        else:
            self.status.set_connection("replay", self.sources.detail)

    def _on_stream_removed(self, key: str) -> None:
        self.sources_panel.remove_stream(key)
        self.live_view.remove_stream(key)
        self.stats_panel.remove(key)
        self.live_filter.forget(key)
        self._update_stream_rates()

    def _on_streams_cleared(self) -> None:
        self.sources_panel.clear_streams()
        self.live_view.clear()
        self.stats_panel.clear()
        self.live_filter.clear()
        self._update_stream_rates()

    def _prune_filter_errors(self) -> None:
        """Kalan akışlarda karşılığı olmayan örnekleme hızlarının uyarılarını zincir durumundan kaldır."""
        before = len(self.live_filter.errors)
        self.live_filter.prune_errors(h.fs for h in self.sources.streams.values())
        if len(self.live_filter.errors) != before:
            self._update_chain_status()

    def _update_stream_rates(self) -> None:
        self.filter_view.set_stream_rates([(h.desc.name or h.key, h.fs) for h in self.sources.streams.values()])
        self._prune_filter_errors()
        self._update_fusion_sources()

    def _update_fusion_sources(self) -> None:
        """Sensör Füzyonu kaynakları: canlı / oynatılan akışlar ve Çalışma Alanı'ndaki anlık görüntüler."""
        sources = [(h.key, f"Akış · {h.desc.name or h.key}", h) for h in self.sources.streams.values()]
        sources += [(f"ws{s.id}", f"Çalışma alanı · {s.name}", s.handle())
                    for s in self.workspace_panel.snapshots.values()]
        self.fusion_view.set_sources(sources)
        self.control_view.set_sources(sources)

    def _on_channel_toggled(self, key: str, channel: int, visible: bool) -> None:
        self._channel_visible[(key, channel)] = visible
        self.live_view.set_channel_visible(key, channel, visible)

    def _on_stats(self, session: dict, rates: dict) -> None:
        self.stats_panel.update_rates(rates)
        self.status.set_traffic(sum(r.packets_per_s for r in rates.values()),
                                sum(r.bytes_per_s for r in rates.values()),
                                sum(r.stats.lost_samples for r in rates.values() if r.stats))
        if self.registry["stream.record"].isChecked():
            self.status.set_recording(True, session.get("recorded_bytes", 0))
        self._update_chain_status()

    def _on_state(self, state: str, detail: str) -> None:
        self.status.set_connection(state, detail)
        udp = self.sources.mode == "udp" and state in ("starting", "listening", "receiving")
        self.registry["conn.listen"].setChecked(udp)
        self.sources_panel.set_listening(udp)
        subtitles = {
            "idle": "Akış bağlı değil",
            "starting": "Başlatılıyor…",
            "listening": f"{detail} dinleniyor · veri bekleniyor",
            "receiving": f"{detail} · veri alınıyor",
            "replay": f"Kayıt · {detail}",
            "error": "Bağlantı hatası — ayrıntılar Günlük panelinde",
        }
        self.live_view.set_subtitle(subtitles.get(state, ""))
        if state in ("idle", "error") and not self.sources.active:
            self.sources_panel.replay.hide()
            self.registry["stream.record"].setChecked(False)
            self.status.set_recording(False)
            self.status.set_traffic(0.0, 0.0, 0)
            if not self.sources.streams:
                self.live_view.clear(show_empty=True)

    def _on_recording(self, active: bool, path: str, size: int) -> None:
        self.registry["stream.record"].setChecked(active)
        self.status.set_recording(active, size)

    def _edit_connection(self) -> None:
        cfg = replace(self._receiver_cfg, address=self.sources_panel.address.text().strip() or "0.0.0.0",
                      port=self.sources_panel.port.value())
        dlg = ConnectionDialog(cfg, self.theme.tokens, self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        self._receiver_cfg = dlg.config()
        self.sources_panel.address.setText(self._receiver_cfg.address)
        self.sources_panel.port.setValue(self._receiver_cfg.port)
        self._save_receiver_config()
        log.info("Bağlantı ayarları kaydedildi; bir sonraki dinlemede geçerli olacak.")

    def _show_simulator(self) -> None:
        if self._simulator is None:
            self._simulator = SimulatorDialog(self.theme.tokens, self)
            self._simulator.started.connect(self._on_simulator_started)
        self._simulator.show()
        self._simulator.raise_()
        self._simulator.activateWindow()

    def _send_generated_signal(self, path: str, fs: float) -> None:
        """Sinyal Üreteci'nin nicemlenmiş çıkışını simülatörle UDP akışı olarak gönder (döngülü) ve dinlemeyi aç."""
        self._show_simulator()
        dlg = self._simulator
        if dlg.running:
            dlg.stop()
        dlg.file.setText(path)
        dlg.fs.setValue(fs)
        dlg.loop.setChecked(True)
        dlg.auto_listen.setChecked(True)
        dlg.toggle()
        self.status.showMessage("Üretilen sinyal simülatörle gönderiliyor — Canlı İzleme'de (Ctrl+1) görünür", 6000)

    def _on_simulator_started(self, host: str, port: int) -> None:
        if self.sources.mode == "udp" and self._receiver_cfg.port == port:
            return
        self.sources_panel.port.setValue(port)
        if self.sources.mode == "udp":
            self.sources.stop()
        self.registry["conn.listen"].setChecked(True)
        self._toggle_listen(True)

    def _load_receiver_config(self) -> ReceiverConfig:
        s, d = self._settings(), ReceiverConfig()
        return ReceiverConfig(str(s.value("conn/address", d.address)), int(s.value("conn/port", d.port)),
                              int(s.value("conn/socket_mb", d.socket_buffer_mb)),
                              float(s.value("buffer/seconds", d.ring_seconds)),
                              int(s.value("buffer/max_mb", d.max_ring_mb)))

    def _save_receiver_config(self) -> None:
        s, c = self._settings(), self._receiver_cfg
        s.setValue("conn/address", c.address)
        s.setValue("conn/port", c.port)
        s.setValue("conn/socket_mb", c.socket_buffer_mb)
        s.setValue("buffer/seconds", c.ring_seconds)
        s.setValue("buffer/max_mb", c.max_ring_mb)

    # ------------------------------------------------------------------ filtre ve işlem zinciri
    def _open_designer(self, family: str, method: str | None = None) -> None:
        self._show_view(2)
        self.filter_view.select(family, method)

    def _design_now(self) -> None:
        self._show_view(2)
        self.filter_view.redesign()

    def _show_dock(self, key: str) -> None:
        if self.detacher.is_detached(f"dock_{key}"):
            self.detacher.show_window(f"dock_{key}")
            return
        dock = self.docks[key]
        dock.show()
        dock.raise_()

    def _apply_design(self, spec: FilterSpec, _index: int | None) -> None:
        """Tasarımı zincire ekler (ya da düzenlenen aşamayı günceller) ve canlı uygulamayı açar.

        Düzenlenen adım indeksle değil kimliğiyle (uid) bulunur: düzenleme sırasında başka bir adım silinse,
        sıralama değişse ya da adım kapatılıp açılsa da doğru adım güncellenir; adım artık yoksa eklenir."""
        cp = self.chain_panel
        uid = self._editing_uid
        index = cp.chain.index_of(uid) if uid is not None else None
        if index is not None:
            old = cp.chain.stages[index]
            cp.replace_stage(index, ChainStage(spec, old.enabled))
            what = f"{index + 1}. adım güncellendi"
        else:
            cp.add_stage(ChainStage(spec))
            what = f"zincire {len(cp.chain)}. adım olarak eklendi"
            if uid is not None:
                what += " (düzenlenen adım zincirde artık yoktu)"
        self.filter_view.stop_editing()
        cp.set_live(True)
        self._show_dock("chain")
        message = f"{spec.label()} — {what}; canlı akışa uygulanıyor"
        log.info(message)
        self.status.showMessage(message, 5000)

    def _on_editing_cancelled(self) -> None:
        self._editing_uid = None
        self.status.showMessage("Aşama düzenleme bırakıldı", 3000)

    def _edit_stage(self, index: int, spec: FilterSpec) -> None:
        self._editing_uid = self.chain_panel.chain.stages[index].uid
        self._show_view(2)
        self.filter_view.start_editing(index, spec)

    def _on_chain_changed(self, chain: ProcessingChain) -> None:
        self.live_filter.set_chain(chain)
        self.live_view.filter_state_changed()
        self._update_chain_status()
        self._push_history()

    def _on_live_toggled(self, on: bool) -> None:
        self.live_filter.set_enabled(on)
        self.live_view.filter_state_changed()
        self._update_chain_status()
        if on and not self.live_filter.chain.enabled_count:
            self.status.showMessage("İşlem zincirinde etkin aşama yok", 4000)

    def _update_chain_status(self) -> None:
        t = self.theme.tokens
        lf = self.live_filter
        chain = lf.chain
        if not len(chain):
            self.chain_panel.set_status("")
            return
        lines = [f"<b>{chain.enabled_count} / {len(chain)}</b> adım etkin · {chain.describe()}"]
        if lf.active:
            lines.append(f'<span style="color:{t.good}">●</span> Canlı akışa uygulanıyor (nedensel)')
        else:
            lines.append(f'<span style="color:{t.text_muted}">●</span> Canlı uygulama kapalı')
        for fs, errors in sorted(lf.errors.items()):
            for error in errors[:3]:
                lines.append(f'<span style="color:{t.warning}">⚠</span> {format_hz(fs)} akışı: {error}')
        lines.append("Adımlar her akışa kendi örnekleme hızıyla yeniden tasarlanarak uygulanır.")
        self.chain_panel.set_status("<br>".join(lines))

    def show_export(self) -> None:
        if self._export is None:
            self._export = ExportDialog(self.theme.tokens, self)
            self._export.format_changed.connect(self.filter_view.set_quant)
            self._export.closed.connect(lambda: self.filter_view.set_quant(None))
            self.filter_view.design_changed.connect(self._export.set_design)
        self._export.set_design(self.filter_view.design)
        self.filter_view.set_quant(self._export.number.currentData())
        self._export.show()
        self._export.raise_()
        self._export.activateWindow()

    # ---------------------------------------------------------------- geri al / yinele
    def _ui_state(self) -> dict:
        return {"spec": self.filter_view.spec().to_dict(), "chain": self.chain_panel.chain.to_dict()}

    def _push_history(self) -> None:
        if not self._restoring:
            self.history.push(self._ui_state())

    def _restore_ui_state(self, state: dict) -> None:
        self._restoring = True
        try:
            spec = FilterSpec.from_dict(state["spec"])
            if spec != self.filter_view.spec():
                self.filter_view.set_spec(spec)
            chain = ProcessingChain.from_dict(state["chain"])
            if chain != self.chain_panel.chain:
                self.filter_view.stop_editing()  # geri alınan zincirde düzenlenen adımın kimliği yok
                self.chain_panel.set_chain(chain)
                self._on_chain_changed(chain)
        finally:
            self._restoring = False

    def undo(self) -> None:
        if self._history_timer.isActive():  # bekleyen tasarımcı değişikliğini önce adım olarak kaydet
            self._history_timer.stop()
            self._push_history()
        state = self.history.undo()
        if state is not None:
            self._restore_ui_state(state)
            self.status.showMessage("Geri alındı", 2000)

    def redo(self) -> None:
        state = self.history.redo()
        if state is not None:
            self._restore_ui_state(state)
            self.status.showMessage("Yinelendi", 2000)

    def _update_undo_actions(self) -> None:
        self.registry["edit.undo"].setEnabled(self.history.can_undo)
        self.registry["edit.redo"].setEnabled(self.history.can_redo)

    # ---------------------------------------------------------------- dondur / çalışma alanı
    def freeze(self) -> list[Snapshot]:
        """Tüm akışların tampondaki verisini Çalışma Alanı'na kopyalar (durdurulmuşsa o ana kadar)."""
        streams = list(self.sources.streams.values())
        if not streams:
            self.status.showMessage("Dondurulacak akış yok — önce dinlemeyi başlatın ya da bir kayıt açın", 4000)
            log.info("Dondur: bağlı akış yok.")
            return []
        snaps = []
        for handle in streams:
            self._snap_seq += 1
            name = f"Anlık {self._snap_seq} · {handle.desc.name or handle.key}"
            try:
                snap = take_snapshot(handle, self.live_view.end_index(handle.key, handle), self._snap_seq, name)
            except MemoryError:
                log.error("Dondur: %s için bellek yetersiz (tampon %s). Kullanılmayan anlık görüntüleri silin ya da "
                          "Ayarlar'da tampon süresini kısaltın.", handle.desc.name or handle.key,
                          f"{handle.ring.capacity:,} örnek")
                continue
            if snap is not None:
                self.workspace_panel.add(snap)
                snaps.append(snap)
        if snaps:
            self.workspace_panel.select_ids([s.id for s in snaps])
            self._update_fusion_sources()
            text = ", ".join(f"{s.name} ({s.duration:.3g} s)" for s in snaps)
            log.info("Anlık görüntü alındı: %s", text)
            self.status.showMessage(f"Çalışma Alanı'na alındı: {text} — İnceleme: Ctrl+2", 5000)
        return snaps

    def filter_snapshots(self, zero_phase: bool) -> list[Snapshot]:
        snaps = self.workspace_panel.selected()
        chain = self.chain_panel.chain
        if not snaps:
            return []
        if not chain.enabled_count:
            self.status.showMessage("İşlem zincirinde etkin aşama yok — önce bir filtre tasarlayıp zincire ekleyin",
                                    5000)
            log.warning("Çevrimdışı filtre: işlem zinciri boş.")
            return []
        results, show = [], []
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            for snap in snaps:
                self._snap_seq += 1
                try:
                    out = filter_snapshot(snap, chain, zero_phase, self._snap_seq)
                except FilterError as exc:
                    log.warning("%s filtrelenemedi: %s", snap.name, exc)
                    continue
                except MemoryError:
                    log.error("%s filtrelenemedi: bellek yetersiz (%s örnek × %d kanal). Daha kısa bir anlık görüntü "
                              "alın ya da kullanılmayan anlık görüntüleri silin.", snap.name, f"{snap.samples:,}",
                              snap.desc.channel_count)
                    continue
                self.workspace_panel.add(out)
                results.append(out)
                show += [snap.id, out.id]
        finally:
            QApplication.restoreOverrideCursor()
        if results:
            self._update_fusion_sources()
            mode = "sıfır fazlı" if zero_phase else "nedensel"
            log.info("Çevrimdışı filtre (%s, %s): %d anlık görüntü", mode, chain.describe(), len(results))
            self.workspace_panel.select_ids(show)
            self._show_view(3)
        return results

    def _show_snapshots(self, snaps: list[Snapshot]) -> None:
        rv = self.review_view
        rv.clear(show_empty=not snaps)
        # Canlı görünümle aynı kural: filtrelenmiş sonucu da seçili olan ham anlık görüntü nötr gri çizilir,
        # filtrelenmiş sonuçlar ve tek başına seçilenler kanal renklerini (sabit sırayla) alır.
        selected = {s.id for s in snaps}
        parents = {s.parent for s in snaps if s.parent in selected}
        index = 0
        for snap in snaps:
            n = snap.desc.channel_count
            if snap.id in parents:
                colors = [self.theme.raw_trace] * n
            else:
                colors = [self.theme.series_color(index + ch) for ch in range(n)]
                index += n
            rv.add_stream(snap.handle(), colors, [True] * n)
        if snaps:
            rv.set_subtitle(" + ".join(s.name for s in snaps))
            rv.return_to_live()
        else:
            rv.set_subtitle("Çalışma Alanı'ndan bir anlık görüntü seçin")

    # ------------------------------------------------------------------ komutlar
    def _show_view(self, index: int) -> None:
        self.views.setCurrentIndex(index)
        self.registry[VIEW_KEYS[index]].setChecked(True)
        self._fit_docks_to_view()

    def _fit_docks_to_view(self) -> None:
        """Dar pencerede geçerli görünüm için ilgisiz yan panelleri gizle, önceki görünüm için gizlenenleri geri aç."""
        if not hasattr(self, "docks") or not self.isVisible():
            return  # pencere gösterilince (showEvent) uygulanır: sekmede önde olan panel ancak o zaman bilinir
        view = VIEW_KEYS[self.views.currentIndex()]
        narrow = self.width() < FOCUS_WIDTH
        hide = {k for k in FOCUS_DOCKS.get(view, ()) if (view, k) not in self._focus_kept} if narrow else set()
        restored = sorted(self._focus_hidden - hide)
        for key in restored:
            self._focus_hidden.discard(key)
            if not self.detacher.is_detached(f"dock_{key}"):
                self.docks[key].show()
        for key in restored:
            if key in self._focus_raised:  # sekme grubunda önceki önde olan yeniden önde
                self._focus_raised.discard(key)
                self.docks[key].raise_()
        hidden_now = []
        targets = [k for k in sorted(hide) if self.docks[k].toggleViewAction().isChecked()
                   and not self.docks[k].isFloating()]
        # gizlemeden önce: sekme grubunda önde olan (arkadaki sekme de isVisible() döner, görünür bölgesi boştur)
        self._focus_raised |= {k for k in targets if not self.docks[k].visibleRegion().isEmpty()}
        for key in targets:
            dock = self.docks[key]
            dock.hide()
            self._focus_hidden.add(key)
            hidden_now.append(dock.windowTitle())
        if hidden_now and hasattr(self, "status"):
            noun = "paneli" if len(hidden_now) == 1 else "panelleri"
            self.status.showMessage(f"Dar pencere: görünüme yer açmak için {', '.join(hidden_now)} {noun} gizlendi "
                                    "(Görünüm menüsünden açılabilir; görünümden çıkınca geri gelir).", 8000)

    def _on_dock_toggled(self, key: str, on: bool) -> None:
        """Kullanıcı paneli menüden açtı: bu görünümde odak düzeni o paneli bir daha gizlemesin."""
        view = VIEW_KEYS[self.views.currentIndex()]
        if on and key in FOCUS_DOCKS.get(view, ()):
            self._focus_kept.add((view, key))
            self._focus_hidden.discard(key)
            self._focus_raised.discard(key)

    def _restore_focus_docks(self) -> None:
        """Odak düzeninin gizlediği panelleri geri aç (yerleşim kaydı kullanıcının düzenini içersin)."""
        for key in sorted(self._focus_hidden):
            if not self.detacher.is_detached(f"dock_{key}"):
                self.docks[key].show()
        for key in sorted(self._focus_raised):
            self.docks[key].raise_()
        self._focus_hidden.clear()
        self._focus_raised.clear()

    def _not_implemented(self, key: str, *_) -> None:
        act = self.registry[key]
        spec = self.registry.spec(key)
        if act.isCheckable():
            act.blockSignals(True)
            act.setChecked(False)
            act.blockSignals(False)
        name = spec.text.rstrip("…")
        when = f"Aşama {spec.phase}{_LOCATIVE[spec.phase]} eklenecek" if spec.phase else "henüz bağlanmadı"
        log.info("“%s” henüz uygulanmadı — %s.", name, when)
        self.status.showMessage(f"{name}: {when}", 4000)

    def _reset_layout(self) -> None:
        self._focus_hidden.clear()
        self._focus_raised.clear()
        self._focus_kept.clear()
        self._place_docks()
        self._fit_docks_to_view()
        QTimer.singleShot(0, self._size_docks)
        log.info("Panel yerleşimi varsayılana döndürüldü.")

    def _set_fullscreen(self, on: bool) -> None:
        if on:
            self._was_maximized = self.isMaximized()
            self.showFullScreen()
        elif getattr(self, "_was_maximized", False):
            self.showMaximized()
        else:
            self.showNormal()

    def _open_protocol_doc(self, path: Path | None = None) -> None:
        path = path or PROTOCOL_DOC
        if path.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))
        else:
            log.warning("Doküman bulunamadı: %s", path)

    def _show_shortcuts(self) -> None:
        rows = [(a.text().replace("&", ""), a.shortcut().toString())
                for _, a in self.registry.items() if not a.shortcut().isEmpty()]
        dlg = QDialog(self)
        dlg.setWindowTitle("Klavye Kısayolları")
        dlg.resize(440, 520)
        table = QTableWidget(len(rows), 2)
        table.setHorizontalHeaderLabels(["Komut", "Kısayol"])
        table.verticalHeader().setVisible(False)
        table.setShowGrid(False)
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        table.horizontalHeader().setDefaultAlignment(Qt.AlignmentFlag.AlignLeft)
        table.verticalHeader().setDefaultSectionSize(34)
        for r, (name, keys) in enumerate(rows):
            table.setItem(r, 0, QTableWidgetItem(name))
            table.setItem(r, 1, QTableWidgetItem(keys))
        lay = QVBoxLayout(dlg)
        lay.setContentsMargins(16, 16, 16, 16)
        lay.addWidget(table)
        style_native_titlebar(dlg, self.theme.tokens)
        dlg.exec()

    def _show_about(self) -> None:
        QMessageBox.about(
            self,
            "Hakkında",
            f"<h3>{APP_NAME}</h3>"
            f"<p>Sürüm {__version__} — Aşama 1 (altyapı ve canlı akış) ve Aşama 3 (filtre tasarımı ve işleme)</p>"
            "<p>Ethernet (UDP) üzerinden SPTF formatında gelen sinyalleri gerçek zamanlı "
            "görselleştirme, filtreleme ve analiz aracı.</p>"
            "<p>PyQt6 · pyqtgraph · NumPy · SciPy</p>",
        )

    # ------------------------------------------------------------------ ayarlar
    def _settings(self) -> QSettings:
        return app_settings()

    def _load_filter_state(self) -> None:
        s = self._settings()
        try:
            spec = s.value("filter/spec")
            if spec:
                self.filter_view.set_spec(FilterSpec.from_dict(json.loads(spec)))
            chain = s.value("filter/chain")
            if chain:
                restored = ProcessingChain.from_dict(json.loads(chain))
                self.chain_panel.set_chain(restored)
                self.live_filter.set_chain(restored)
        except (ValueError, TypeError, FilterError) as exc:
            log.warning("Kayıtlı filtre ayarları okunamadı (%s); varsayılanlar kullanılıyor.", exc)
        display = str(s.value("filter/display", "both"))
        self.chain_panel.display.setCurrentIndex(1 if display == "filtered" else 0)
        self.filter_view.set_log_freq(s.value("filter/log_freq", False, type=bool))
        self._update_chain_status()

    def _load_display_state(self) -> None:
        """Zaman penceresi (kullanıcı seçtiyse) ve oynatma hızı / döngü tercihleri."""
        s = self._settings()
        p = self.properties_panel
        self._window_user_set = s.value("display/window_user", False, type=bool)
        index = p.time_window.findText(str(s.value("display/time_window", "")))
        if self._window_user_set and index >= 0:
            p.time_window.setCurrentIndex(index)
        p.time_window.activated.connect(lambda *_: setattr(self, "_window_user_set", True))  # yalnız elle seçim
        controls = self.sources_panel.replay
        speed = controls.speed.findText(str(s.value("replay/speed", "")))
        if speed >= 0:
            controls.speed.setCurrentIndex(speed)
        controls.loop.blockSignals(True)
        controls.loop.setChecked(s.value("replay/loop", False, type=bool))
        controls.loop.blockSignals(False)

    def _save_display_state(self) -> None:
        s = self._settings()
        s.setValue("display/time_window", self.properties_panel.time_window.currentText())
        s.setValue("display/window_user", self._window_user_set)
        s.setValue("replay/speed", self.sources_panel.replay.speed.currentText())
        s.setValue("replay/loop", self.sources_panel.replay.loop.isChecked())

    def _save_filter_state(self) -> None:
        s = self._settings()
        s.setValue("filter/spec", json.dumps(self.filter_view.spec().to_dict()))
        s.setValue("filter/chain", json.dumps(self.chain_panel.chain.to_dict(), ensure_ascii=False))
        s.setValue("filter/display", self.chain_panel.display_mode)
        s.setValue("filter/log_freq", self.filter_view.log_freq)

    def _restore_settings(self) -> bool:
        s = self._settings()
        geometry, state = s.value("window/geometry"), s.value("window/state")
        if geometry is None or state is None:
            return False
        self.restoreGeometry(geometry)
        restored = self.restoreState(state, LAYOUT_VERSION)
        view = int(s.value("view/index", 0))
        self._show_view(view if 0 <= view < len(VIEW_KEYS) else 0)
        return restored

    def _load_control_state(self) -> None:
        try:
            self.control_view.restore_state(self._settings())
        except Exception:  # noqa: BLE001 - bozuk ayar uygulamanın açılmasını engellemesin
            log.exception("Kontrol Tasarımı ayarları okunamadı; varsayılanlar kullanılıyor.")

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        compact = self.width() < 1500
        if compact != self._compact and hasattr(self, "_nav_pills"):
            self._compact = compact
            style = (Qt.ToolButtonStyle.ToolButtonIconOnly if compact
                     else Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
            for pill in self._nav_pills:
                pill.setToolButtonStyle(style)
            self._wordmark_action.setVisible(not compact)
            self.command_box.setFixedWidth(200 if compact else 300)
        narrow = self.width() < FOCUS_WIDTH
        if narrow != self._focus_narrow and self.isVisible():
            self._focus_narrow = narrow
            self._fit_docks_to_view()

    def showEvent(self, event: QShowEvent) -> None:
        super().showEvent(event)
        if not self._titlebar_styled:
            style_native_titlebar(self, self.theme.tokens)
            self._titlebar_styled = True
        QTimer.singleShot(0, self._fit_docks_to_view)

    def closeEvent(self, event: QCloseEvent) -> None:
        # Ayrık pencereler hatırlanır ve geri alınır; böylece panel yerleşimi tam hâliyle kaydedilir.
        self.detacher.save_open()
        self.detacher.attach_all()
        self._restore_focus_docks()
        if self._export is not None:
            self._export.close()
        s = self._settings()
        s.setValue("window/geometry", self.saveGeometry())
        s.setValue("window/state", self.saveState(LAYOUT_VERSION))
        s.setValue("view/index", self.views.currentIndex())
        self._save_filter_state()
        self._save_display_state()
        try:
            self.control_view.save_state(s)
        except Exception:  # noqa: BLE001
            log.exception("Kontrol Tasarımı ayarları kaydedilemedi.")
        self._receiver_cfg = replace(self._receiver_cfg, address=self.sources_panel.address.text().strip() or "0.0.0.0",
                                     port=self.sources_panel.port.value())
        self._save_receiver_config()
        self._refresh_timer.stop()
        self.fusion_view.shutdown()
        self.siggen_view.shutdown()
        self.control_view.shutdown()
        if self._simulator is not None:
            self._simulator.stop()
        self.sources.shutdown()
        self.log_panel.detach()
        super().closeEvent(event)
