"""Sinyal Test Uygulaması penceresi: akış / kanal düzenleme, senaryo seçimi, ağ bozulmaları, canlı durum ve önizleme.

Gönderim motoru (spt.sender.engine) ayrı süreçte çalışır; pencere yalnız yapılandırmayı düzenler, motoru başlatır /
durdurur ve istatistik olaylarını gösterir. Gönderim sürerken akış / kanal düzenlemeleri kısa bir gecikmeyle motoru
yeniden başlatarak uygulanır (tanımlayıcısı değişen akışın desc_rev'i artar; alıcı yeniden yapılandırır), ağ bozulmaları
ise motor durmadan uygulanır.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import queue
from dataclasses import replace
from pathlib import Path

import numpy as np
from PyQt6.QtCore import QObject, QSettings, QSize, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QBrush, QColor
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from spt import ORG_NAME
from spt.gui.icons import app_icon, icon, primary_icon
from spt.gui.theme import ThemeManager, style_native_titlebar
from spt.gui.widgets.common import Card, LegendBar, PageHeader, StatusBadge, chip, combo, form_layout, icon_button, label
from spt.gui.widgets.plot import make_plot, pen
from spt.protocol.format import SAMPLE_TYPES_BY_NAME, format_rate
from spt.sender import signals as sg
from spt.sender.engine import run_engine
from spt.sources import netinfo

APP_TITLE = "Sinyal Test Uygulaması"
SETTINGS_APP = "SignalTestSender"     # Stüdyo'nun ayarlarından ayrı
RESTART_MS = 700                     # düzenleme bitince motorun yeniden başlatılması için bekleme
PREVIEW_POINTS = 4000
PREVIEW_CHANNELS = 8
POLL_MS = 100
STREAM_COLS = ["Açık", "ID", "Ad", "Örnekleme", "Tip", "Kanal", "Paket", "Veri hızı", "Durum"]
CHANNEL_COLS = ["Ad", "Birim", "Sinyal", "Frekans (Hz)", "Genlik", "Ofset", "Faz (°)", "Gürültü σ"]
CHANNEL_WIDTHS = (100, 56, 150, 104, 90, 90, 72, 90)
KIND_KEYS = list(sg.SIGNAL_KINDS)


def sender_settings() -> QSettings:
    """Ayarlar: `SPT_SENDER_SETTINGS_FILE` verilirse o INI dosyası (testler), yoksa kullanıcı kaydı (Stüdyo'dan ayrı)."""
    path = os.environ.get("SPT_SENDER_SETTINGS_FILE")
    if path:
        return QSettings(path, QSettings.Format.IniFormat)
    return QSettings(ORG_NAME, SETTINGS_APP)


def load_config(settings: QSettings) -> sg.SenderConfig:
    text = settings.value("config", "")
    if text:
        try:
            return sg.config_from_dict(json.loads(str(text)))
        except (TypeError, ValueError):
            pass
    return sg.default_config()


def _rate(bps: float) -> str:
    if bps >= 1e6:
        return f"{bps / 1e6:.2f} MB/s"
    if bps >= 1e3:
        return f"{bps / 1e3:.1f} kB/s"
    return f"{bps:.0f} B/s"


def _dspin(lo: float, hi: float, value: float, decimals: int = 3, suffix: str = "", step: float = 1.0) -> QDoubleSpinBox:
    w = QDoubleSpinBox()
    w.setRange(lo, hi)
    w.setDecimals(decimals)
    w.setSingleStep(step)
    w.setValue(value)
    w.setSuffix(suffix)
    w.setKeyboardTracking(False)       # her tuşta değil, değer girilince
    return w


class EngineController(QObject):
    """Motor sürecini başlatır / durdurur, olay kuyruğunu Qt sinyallerine çevirir."""

    started = pyqtSignal(object)
    stats = pyqtSignal(object)
    stopped = pyqtSignal()
    failed = pyqtSignal(str)

    def __init__(self, parent: QObject | None = None):
        super().__init__(parent)
        self._ctx = mp.get_context("spawn")
        self._proc = None
        self._commands = self._events = None
        self._timer = QTimer(self)
        self._timer.setInterval(POLL_MS)
        self._timer.timeout.connect(self._poll)

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.is_alive()

    def start(self, config: dict) -> None:
        self.stop()
        self._commands, self._events = self._ctx.Queue(), self._ctx.Queue()
        self._proc = self._ctx.Process(target=run_engine, args=(config, self._commands, self._events),
                                       name="sinyal-test-motoru", daemon=True)
        self._proc.start()
        self._timer.start()

    def stop(self) -> None:
        proc = self._proc
        if proc is None:
            return
        if proc.is_alive():
            try:
                self._commands.put(("stop",))
            except (OSError, ValueError):
                pass
            proc.join(3)
            if proc.is_alive():
                proc.kill()
                proc.join(2)
        self._poll()
        self._timer.stop()
        self._proc = None
        self.stopped.emit()

    def set_impairments(self, imp: sg.Impairments) -> None:
        if self.running:
            self._commands.put(("impairments", vars(imp).copy()))

    def _poll(self) -> None:
        if self._events is None:
            return
        for _ in range(200):
            try:
                ev = self._events.get_nowait()
            except (queue.Empty, OSError, ValueError, EOFError):
                break
            kind = ev[0]
            if kind == "started":
                self.started.emit(ev[1])
            elif kind == "stats":
                self.stats.emit(ev[1])
            elif kind == "error":
                self.failed.emit(ev[1])
        if self._proc is not None and not self._proc.is_alive() and self._timer.isActive():
            self._timer.stop()
            self._proc = None
            self.stopped.emit()


class SenderWindow(QMainWindow):
    def __init__(self, theme: ThemeManager, config: sg.SenderConfig | None = None,
                 settings: QSettings | None = None, studio_target: tuple[str, int] | None = None):
        super().__init__()
        self.theme = theme
        self.settings = settings or sender_settings()
        self.cfg = config or load_config(self.settings)
        if studio_target is not None:
            self.cfg.host, self.cfg.port = studio_target
        self._sent_keys: dict[int, tuple] = {}         # gönderilen tanımlayıcılar (rev artırımı için)
        self._live: dict[int, dict] = {}               # akış kimliği → son istatistik satırı
        self._syncing = False
        self.setWindowTitle(APP_TITLE)
        self.setWindowIcon(app_icon())
        self._titlebar_styled = False

        self.engine = EngineController(self)
        self.engine.started.connect(self._on_started)
        self.engine.stats.connect(self._on_stats)
        self.engine.stopped.connect(self._on_stopped)
        self.engine.failed.connect(self._on_failed)
        self._restart = QTimer(self)
        self._restart.setSingleShot(True)
        self._restart.setInterval(RESTART_MS)
        self._restart.timeout.connect(self._apply_restart)
        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(800)
        self._save_timer.timeout.connect(self.save_settings)
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(120)
        self._preview_timer.timeout.connect(self._draw_preview)

        central = QWidget()
        central.setObjectName("Canvas")
        outer = QVBoxLayout(central)
        outer.setContentsMargins(12, 8, 12, 8)
        outer.setSpacing(10)
        self.header = PageHeader(APP_TITLE, "Birden çok SPTF akışını UDP ile, örnekleme hızına göre zamanlayarak "
                                            "gönderir — Stüdyo'da Canlı İzleme ve Bağlantı sayfasında görünür")
        self.studio_chip = chip(f"Stüdyo · {self.cfg.host}:{self.cfg.port}")
        self.studio_chip.setVisible(studio_target is not None)
        self.badge = StatusBadge("Durduruldu", "idle")
        self.start_btn = QPushButton(primary_icon("ph.play"), "  Başlat")
        self.start_btn.setObjectName("Primary")
        self.start_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.start_btn.clicked.connect(self.toggle)
        for w in (self.studio_chip, self.badge, self.start_btn):
            self.header.add_right(w)
        outer.addWidget(self.header)

        body = QSplitter(Qt.Orientation.Horizontal)
        body.setChildrenCollapsible(False)
        left = QWidget()
        lcol = QVBoxLayout(left)
        lcol.setContentsMargins(0, 0, 0, 0)
        lcol.setSpacing(10)
        lcol.addWidget(self._build_target())
        lcol.addWidget(self._build_scenario())
        lcol.addWidget(self._build_impairments())
        lcol.addStretch(1)
        left.setMinimumWidth(270)
        left.setMaximumWidth(360)
        body.addWidget(left)

        right = QSplitter(Qt.Orientation.Vertical)
        right.setChildrenCollapsible(False)
        right.addWidget(self._build_streams())
        middle = QSplitter(Qt.Orientation.Horizontal)
        middle.setChildrenCollapsible(False)
        middle.addWidget(self._build_stream_form())
        middle.addWidget(self._build_channels())
        middle.setStretchFactor(0, 0)
        middle.setStretchFactor(1, 1)
        middle.setSizes([300, 700])
        right.addWidget(middle)
        right.addWidget(self._build_preview())
        right.setSizes([230, 300, 220])
        body.addWidget(right)
        body.setStretchFactor(1, 1)
        body.setSizes([300, 1000])
        outer.addWidget(body, 1)
        self.setCentralWidget(central)

        bar = QStatusBar()
        self.total_label = QLabel("Gönderim yok")
        self.link_label = QLabel("")
        self.error_label = QLabel("")
        bar.addWidget(self.total_label, 1)
        bar.addPermanentWidget(self.error_label)
        bar.addPermanentWidget(self.link_label)
        self.setStatusBar(bar)

        self._load_into_widgets()
        self.resize(1320, 820)
        geometry = self.settings.value("geometry")
        if geometry is not None:
            self.restoreGeometry(geometry)

    # ================================================================ kurulum
    def _build_target(self) -> Card:
        card = Card("Hedef")
        form = form_layout()
        self.host = QLineEdit()
        self.host.setPlaceholderText("Stüdyo'nun çalıştığı bilgisayarın IP'si")
        self.host.setToolTip("Stüdyo'nun dinlediği bilgisayar: aynı PC'de 127.0.0.1, ağdaki başka PC'de onun IP'si")
        self.port = QSpinBox()
        self.port.setRange(1, 65535)
        self.port.setToolTip("Stüdyo'nun UDP dinleme portu (Kaynaklar paneli; varsayılan 50505)")
        self.bind = combo(["Otomatik (yönlendirme seçer)"])
        self.bind.setToolTip("Gönderimin çıkacağı yerel ağ arayüzü (birden çok kart / ağ varsa)")
        for item in netinfo.interfaces():
            for ip, plen in item.addresses:
                self.bind.addItem(f"{ip} · {item.name}", ip)
        self.route = label("", "Muted")
        self.route.setWordWrap(True)
        form.addRow(label("IP adresi", "FormLabel"), self.host)
        form.addRow(label("Port", "FormLabel"), self.port)
        form.addRow(label("Yerel arayüz", "FormLabel"), self.bind)
        card.body.addLayout(form)
        card.body.addWidget(self.route)
        self.host.editingFinished.connect(self._target_changed)
        self.port.valueChanged.connect(lambda *_: self._target_changed())
        self.bind.currentIndexChanged.connect(lambda *_: self._target_changed())
        return card

    def _build_scenario(self) -> Card:
        card = Card("Senaryo")
        self.scenario = combo(sg.scenario_names())
        self.scenario.setToolTip("Hazır akış setleri; seçip “Uygula” ile akış listesine yüklenir, sonra düzenlenebilir")
        apply = QPushButton(icon("ph.arrows-clockwise"), " Uygula")
        apply.setToolTip("Seçili senaryonun akışlarını yükler (mevcut akışların yerine)")
        apply.clicked.connect(self.apply_scenario)
        row = QHBoxLayout()
        row.setSpacing(6)
        row.addWidget(self.scenario, 1)
        row.addWidget(apply)
        card.body.addLayout(row)
        files = QHBoxLayout()
        files.setSpacing(6)
        save = QPushButton(icon("ph.floppy-disk"), " Kaydet…")
        save.setToolTip("Akışlar, kanallar, hedef ve bozulmalar JSON dosyasına")
        save.clicked.connect(self._save_json)
        load = QPushButton(icon("ph.folder-open"), " Aç…")
        load.clicked.connect(self._load_json)
        files.addWidget(save)
        files.addWidget(load)
        card.body.addLayout(files)
        return card

    def _build_impairments(self) -> Card:
        card = Card("Ağ bozulmaları")
        form = form_layout()
        self.loss = _dspin(0, 100, 0, 2, " %", 0.5)
        self.loss.setToolTip("Atılan DATA çerçevesi oranı (Stüdyo'da kayıp olarak görünür)")
        self.jitter = _dspin(0, 10_000, 0, 2, " ms", 0.5)
        self.jitter.setToolTip("Her çerçevenin gönderimine eklenen |N(0, σ)| gecikme; σ büyükse çerçeveler sıra dışı da "
                               "gelebilir (Bağlantı sayfasında titreşim / gecikme değişimi)")
        self.reorder = _dspin(0, 100, 0, 2, " %", 0.5)
        self.reorder.setToolTip("Bir sonraki çerçeveyle yer değiştiren çerçeve oranı (Stüdyo'da 'geç' çerçeve)")
        self.duplicate = _dspin(0, 100, 0, 2, " %", 0.5)
        self.duplicate.setToolTip("İki kez gönderilen çerçeve oranı (Stüdyo'da 'yineleme')")
        for text, w in (("Paket kaybı", self.loss), ("Titreşim σ", self.jitter), ("Yeniden sıralama", self.reorder),
                        ("Yinelenen paket", self.duplicate)):
            form.addRow(label(text, "FormLabel"), w)
            w.valueChanged.connect(lambda *_: self._impairments_changed())
        card.body.addLayout(form)
        note = label("Gönderim sürerken hemen uygulanır.", "Muted")
        note.setWordWrap(True)
        card.body.addWidget(note)
        return card

    def _build_streams(self) -> Card:
        card = Card("Akışlar")
        for name, tip, slot in (("ph.plus", "Akış ekle", self.add_stream), ("ph.copy", "Seçili akışı çoğalt",
                                                                             self.duplicate_stream),
                                ("ph.trash", "Seçili akışı sil", self.remove_stream)):
            btn = icon_button(name, tip, object_name="FlatIcon", size=16)
            btn.clicked.connect(slot)
            card.add_right(btn)
        self.stream_table = QTableWidget(0, len(STREAM_COLS))
        self.stream_table.setHorizontalHeaderLabels(STREAM_COLS)
        self.stream_table.verticalHeader().setVisible(False)
        self.stream_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.stream_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.stream_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        head = self.stream_table.horizontalHeader()
        head.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        head.setStretchLastSection(True)
        self.stream_table.itemSelectionChanged.connect(self._stream_selected)
        self.stream_table.itemChanged.connect(self._stream_item_changed)
        card.body.addWidget(self.stream_table)
        return card

    def _build_stream_form(self) -> Card:
        card = Card("Akış ayarları")
        form = form_layout()
        self.s_name = QLineEdit()
        self.s_name.setMaxLength(31)
        self.s_id = QSpinBox()
        self.s_id.setRange(0, 0xFFFF)
        self.s_id.setToolTip("SPTF stream_id: aynı göndericiden gelen akışlar bununla ayrılır (benzersiz olmalı)")
        self.s_fs = _dspin(0.001, sg.MAX_FS, 1000.0, 3, " Hz", 100.0)
        self.s_type = combo(list(SAMPLE_TYPES_BY_NAME))
        self.s_layout = combo(["Interleaved", "Planar"])
        self.s_datagram = combo([f"{d} B" for d in sg.DATAGRAMS])
        self.s_datagram.setToolTip("1472 B standart MTU, 8972 B jumbo frame, 65507 B en büyük (IP parçalanır)")
        self.s_frame = _dspin(0.1, 10_000, 10.0, 1, " ms", 1.0)
        self.s_frame.setToolTip("En uzun paket aralığı: düşük hızda datagram dolmadan gönderilir (gerçek zamanlı izleme "
                                "gecikmesi). Yüksek hızda datagram boyu sınırlar.")
        self.s_crc = QCheckBox("CRC32")
        self.s_marker = _dspin(0, 3600, 0, 2, " s", 1.0)
        self.s_marker.setToolTip("MARKER çerçevesi aralığı (0 = kapalı); Stüdyo'da zaman grafiğinde dikey çizgi")
        for text, w in (("Ad", self.s_name), ("Akış kimliği", self.s_id), ("Örnekleme hızı", self.s_fs),
                        ("Örnek tipi", self.s_type), ("Yerleşim", self.s_layout), ("Datagram", self.s_datagram),
                        ("Paket aralığı ≤", self.s_frame), ("MARKER aralığı", self.s_marker)):
            form.addRow(label(text, "FormLabel"), w)
        form.addRow(self.s_crc)
        # Kaydırılabilir: 768 px yükseklikte formun en küçük boyu akış tablosunu ve önizlemeyi eziyordu
        holder = QWidget()
        holder.setLayout(form)
        scroll = QScrollArea()
        scroll.setWidget(holder)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setMinimumHeight(80)
        card.body.addWidget(scroll, 1)
        self.s_name.editingFinished.connect(self._stream_form_changed)
        for w in (self.s_id, self.s_fs, self.s_frame, self.s_marker):
            w.valueChanged.connect(lambda *_: self._stream_form_changed())
        for w in (self.s_type, self.s_layout, self.s_datagram):
            w.currentIndexChanged.connect(lambda *_: self._stream_form_changed())
        self.s_crc.toggled.connect(lambda *_: self._stream_form_changed())
        self.stream_form = card
        return card

    def _build_channels(self) -> Card:
        card = Card("Kanallar")
        for name, tip, slot in (("ph.plus", "Kanal ekle", self.add_channel),
                                ("ph.trash", "Seçili kanalı sil", self.remove_channel)):
            btn = icon_button(name, tip, object_name="FlatIcon", size=16)
            btn.clicked.connect(slot)
            card.add_right(btn)
        self.ch_table = QTableWidget(0, len(CHANNEL_COLS))
        self.ch_table.setHorizontalHeaderLabels(CHANNEL_COLS)
        self.ch_table.verticalHeader().setVisible(False)
        self.ch_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.ch_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        head = self.ch_table.horizontalHeader()
        head.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        head.setStretchLastSection(True)
        for c, width in enumerate(CHANNEL_WIDTHS):
            self.ch_table.setColumnWidth(c, width)
        self.ch_table.verticalHeader().setDefaultSectionSize(30)
        self.ch_table.itemSelectionChanged.connect(self._channel_selected)
        card.body.addWidget(self.ch_table, 1)
        extra = QHBoxLayout()
        extra.setSpacing(8)
        self.extra_title = label("", "FormLabel")
        extra.addWidget(self.extra_title)
        self.extra_fields: dict[str, tuple[QLabel, QDoubleSpinBox]] = {}
        specs = {"duty": (0, 1, 3, 0.05), "f1": (0, sg.MAX_FS, 3, 100.0), "sweep_s": (0.001, 3600, 3, 0.5),
                 "harmonics": (1, 64, 0, 1), "mod_freq": (0, sg.MAX_FS, 3, 1.0), "depth": (0, 1, 3, 0.05),
                 "deviation": (0, sg.MAX_FS, 3, 10.0)}
        for key, (lo, hi, dec, step) in specs.items():
            text, suffix = sg.PARAM_LABELS[key]
            lab = label(text, "FormLabel")
            spin = _dspin(lo, hi, 0, dec, suffix, step)
            spin.valueChanged.connect(lambda _v, k=key: self._extra_changed(k))
            # Baştan gizli: yedi alan birden görünürken hesaplanan en küçük genişlik (≈ 1800 px) pencere ilk
            # gösterildiğinde boyunu büyütüyordu; seçili kanalın türüne göre yalnız ilgili olanlar açılır
            lab.hide()
            spin.hide()
            extra.addWidget(lab)
            extra.addWidget(spin)
            self.extra_fields[key] = (lab, spin)
        extra.addStretch(1)
        card.body.addLayout(extra)
        return card

    def _build_preview(self) -> Card:
        card = Card("Önizleme")
        self.preview_legend = LegendBar(lambda: [self.plot])
        card.add_left(self.preview_legend, optional=True)
        self.preview_info = label("", "Readout")
        card.add_right(self.preview_info, optional=True)
        self.plot = make_plot(self.theme, "Zaman", "Değer", x_units="s")
        self.plot.setMinimumHeight(110)
        card.body.addWidget(self.plot)
        return card

    # ================================================================ yapılandırma ↔ arayüz
    def _load_into_widgets(self) -> None:
        self._syncing = True
        cfg = self.cfg
        self.host.setText(cfg.host)
        self.port.setValue(cfg.port)
        idx = self.bind.findData(cfg.bind) if cfg.bind else 0
        self.bind.setCurrentIndex(max(idx, 0))
        if cfg.scenario in sg.scenario_names():
            self.scenario.setCurrentText(cfg.scenario)
        imp = cfg.impairments
        for w, v in ((self.loss, imp.loss_pct), (self.jitter, imp.jitter_ms), (self.reorder, imp.reorder_pct),
                     (self.duplicate, imp.duplicate_pct)):
            w.setValue(v)
        self._syncing = False
        self._fill_streams(select=0)
        self._update_route()

    def _fill_streams(self, select: int | None = None) -> None:
        current = self.current_row() if select is None else select
        self._syncing = True
        t = self.stream_table
        t.setRowCount(len(self.cfg.streams))
        for r, st in enumerate(self.cfg.streams):
            items = self._stream_row(st)
            for c, item in enumerate(items):
                t.setItem(r, c, item)
        self._syncing = False
        if self.cfg.streams:
            t.selectRow(min(max(current, 0), len(self.cfg.streams) - 1))
        # Seçili satır numarası değişmediyse (ör. senaryo değişti, yine 0. satır) seçim sinyali gelmez: kanal
        # tablosu eski akışta kalmasın
        self._show_stream(self.current_stream())
        self._refresh_live()

    def _stream_row(self, st: sg.StreamSpec) -> list[QTableWidgetItem]:
        on = QTableWidgetItem()
        on.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
        on.setCheckState(Qt.CheckState.Checked if st.enabled else Qt.CheckState.Unchecked)
        on.setToolTip("Akışı gönder / gönderme")
        spf = sg.frame_samples(st)
        texts = [str(st.stream_id), st.name, format_rate(st.fs), st.sample_type, str(len(st.channels)),
                 f"{spf} örn · {spf / st.fs * 1e3:.1f} ms", _rate(st.bytes_per_second()), "—"]
        return [on] + [QTableWidgetItem(x) for x in texts]

    def current_row(self) -> int:
        rows = self.stream_table.selectionModel().selectedRows() if self.stream_table.selectionModel() else []
        return rows[0].row() if rows else -1

    def current_stream(self) -> sg.StreamSpec | None:
        r = self.current_row()
        return self.cfg.streams[r] if 0 <= r < len(self.cfg.streams) else None

    def _stream_selected(self) -> None:
        self._show_stream(self.current_stream())

    def _show_stream(self, st: sg.StreamSpec | None) -> None:
        self._syncing = True
        self.stream_form.setEnabled(st is not None)
        if st is not None:
            self.s_name.setText(st.name)
            self.s_id.setValue(st.stream_id)
            self.s_fs.setValue(st.fs)
            self.s_type.setCurrentText(st.sample_type)
            self.s_layout.setCurrentIndex(1 if st.layout == "planar" else 0)
            self.s_datagram.setCurrentIndex(sg.DATAGRAMS.index(st.datagram) if st.datagram in sg.DATAGRAMS else 0)
            self.s_frame.setValue(st.frame_ms)
            self.s_marker.setValue(st.marker_every)
            self.s_crc.setChecked(st.crc)
        self._syncing = False
        self._fill_channels(st)
        self._preview_timer.start()

    def _fill_channels(self, st: sg.StreamSpec | None, select: int = 0) -> None:
        self._syncing = True
        t = self.ch_table
        t.setRowCount(0 if st is None else len(st.channels))
        for r, ch in enumerate(st.channels if st is not None else []):
            name = QLineEdit(ch.name)
            name.setMaxLength(31)
            unit = QLineEdit(ch.unit)
            unit.setMaxLength(15)
            kind = combo([sg.SIGNAL_KINDS[k][0] for k in KIND_KEYS], KIND_KEYS.index(ch.kind))
            freq = _dspin(0, sg.MAX_FS, ch.freq, 3, "", 1.0)
            amp = _dspin(-1e9, 1e9, ch.amp, 3, "", 0.1)
            off = _dspin(-1e9, 1e9, ch.offset, 3, "", 0.1)
            ph = _dspin(-360, 360, ch.phase_deg, 1, "", 15.0)
            noise = _dspin(0, 1e9, ch.noise, 3, "", 0.01)
            widgets = (name, unit, kind, freq, amp, off, ph, noise)
            for c, w in enumerate(widgets):
                if isinstance(w, QDoubleSpinBox):   # dar sütun: oklar yerine yazma / tekerlek
                    w.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.NoButtons)
                t.setCellWidget(r, c, w)
            name.editingFinished.connect(lambda r=r: self._channel_changed(r))
            unit.editingFinished.connect(lambda r=r: self._channel_changed(r))
            kind.currentIndexChanged.connect(lambda _i, r=r: self._channel_changed(r))
            for w in (freq, amp, off, ph, noise):
                w.valueChanged.connect(lambda _v, r=r: self._channel_changed(r))
        self._syncing = False
        if st is not None and st.channels:
            t.selectRow(min(select, len(st.channels) - 1))
        self._channel_selected()

    def _channel_widgets(self, r: int) -> list[QWidget]:
        return [self.ch_table.cellWidget(r, c) for c in range(len(CHANNEL_COLS))]

    def current_channel(self) -> int:
        rows = self.ch_table.selectionModel().selectedRows() if self.ch_table.selectionModel() else []
        return rows[0].row() if rows else (0 if self.ch_table.rowCount() else -1)

    def _channel_selected(self) -> None:
        st, r = self.current_stream(), self.current_channel()
        ch = st.channels[r] if st is not None and 0 <= r < len(st.channels) else None
        params = sg.SIGNAL_KINDS[ch.kind][1] if ch is not None else ()
        self._syncing = True
        self.extra_title.setText(f"{ch.name} ·" if ch is not None and params else "")
        for key, (lab, spin) in self.extra_fields.items():
            show = key in params
            lab.setVisible(show)
            spin.setVisible(show)
            if show:
                spin.setValue(float(getattr(ch, key)))
        self._syncing = False

    # ================================================================ düzenleme
    def _target_changed(self) -> None:
        if self._syncing:
            return
        self.cfg.host = self.host.text().strip() or "127.0.0.1"
        self.cfg.port = self.port.value()
        self.cfg.bind = self.bind.currentData() or ""
        self.studio_chip.setText(f"Stüdyo · {self.cfg.host}:{self.cfg.port}")
        self._update_route()
        self._edited()

    def _impairments_changed(self) -> None:
        if self._syncing:
            return
        self.cfg.impairments = sg.Impairments(self.loss.value(), self.jitter.value(), self.reorder.value(),
                                              self.duplicate.value())
        self.engine.set_impairments(self.cfg.impairments)
        self._save_timer.start()

    def _stream_form_changed(self) -> None:
        st = self.current_stream()
        if self._syncing or st is None:
            return
        idx = self.current_row()
        layout = "planar" if self.s_layout.currentIndex() == 1 else "interleaved"
        new = replace(st, name=self.s_name.text().strip() or "akış", stream_id=self.s_id.value(),
                      fs=self.s_fs.value(), sample_type=self.s_type.currentText(), layout=layout,
                      datagram=sg.DATAGRAMS[self.s_datagram.currentIndex()], frame_ms=self.s_frame.value(),
                      marker_every=self.s_marker.value(), crc=self.s_crc.isChecked())
        self.cfg.streams[idx] = new
        self._update_stream_row(idx)
        self._preview_timer.start()
        self._edited()

    def _stream_item_changed(self, item: QTableWidgetItem) -> None:
        if self._syncing or item.column() != 0:
            return
        st = self.cfg.streams[item.row()]
        st.enabled = item.checkState() == Qt.CheckState.Checked
        self._edited()

    def _channel_changed(self, r: int) -> None:
        st = self.current_stream()
        if self._syncing or st is None or r >= len(st.channels):
            return
        name, unit, kind, freq, amp, off, ph, noise = self._channel_widgets(r)
        old = st.channels[r]
        new_kind = KIND_KEYS[kind.currentIndex()]
        st.channels[r] = replace(old, name=name.text().strip() or f"ch{r}", unit=unit.text().strip(), kind=new_kind,
                                 freq=freq.value(), amp=amp.value(), offset=off.value(), phase_deg=ph.value(),
                                 noise=noise.value())
        if new_kind != old.kind and r == self.current_channel():
            self._channel_selected()
        self._update_stream_row(self.current_row())
        self._preview_timer.start()
        self._edited()

    def _extra_changed(self, key: str) -> None:
        st, r = self.current_stream(), self.current_channel()
        if self._syncing or st is None or not 0 <= r < len(st.channels):
            return
        value = self.extra_fields[key][1].value()
        st.channels[r] = replace(st.channels[r], **{key: int(value) if key == "harmonics" else value})
        self._preview_timer.start()
        self._edited()

    def _update_stream_row(self, r: int) -> None:
        if not 0 <= r < len(self.cfg.streams):
            return
        self._syncing = True
        for c, item in enumerate(self._stream_row(self.cfg.streams[r])):
            if c:   # onay kutusu kendi durumunu taşır
                self.stream_table.setItem(r, c, item)
        self._syncing = False
        self._refresh_live()

    def _edited(self) -> None:
        """Yapılandırma değişti: kaydet; gönderim sürüyorsa kısa gecikmeyle motoru yeniden başlat."""
        self.cfg.scenario = ""
        self._save_timer.start()
        if self.engine.running:
            self._restart.start()

    # ---------------------------------------------------------------- akış / kanal ekle-sil
    def _free_id(self) -> int:
        used = {s.stream_id for s in self.cfg.streams}
        return next(i for i in range(1, 0x10000) if i not in used)

    def add_stream(self) -> None:
        if len(self.cfg.streams) >= sg.MAX_STREAMS:
            return
        sid = self._free_id()
        self.cfg.streams.append(sg.StreamSpec(sid, f"akis-{sid}", 1000.0, channels=[sg.ChannelSpec("ch0")]))
        self._fill_streams(select=len(self.cfg.streams) - 1)
        self._edited()

    def duplicate_stream(self) -> None:
        st = self.current_stream()
        if st is None or len(self.cfg.streams) >= sg.MAX_STREAMS:
            return
        copy = sg.stream_from_dict(sg.config_to_dict(sg.SenderConfig(streams=[st]))["streams"][0])
        copy.stream_id = self._free_id()
        copy.name = f"{st.name}-kopya"[:31]
        copy.rev = 1
        self.cfg.streams.append(copy)
        self._fill_streams(select=len(self.cfg.streams) - 1)
        self._edited()

    def remove_stream(self) -> None:
        r = self.current_row()
        if 0 <= r < len(self.cfg.streams):
            del self.cfg.streams[r]
            self._fill_streams(select=max(r - 1, 0))
            self._edited()

    def add_channel(self) -> None:
        st = self.current_stream()
        if st is None or len(st.channels) >= 64:
            return
        st.channels.append(sg.ChannelSpec(f"ch{len(st.channels)}"))
        self._fill_channels(st, select=len(st.channels) - 1)
        self._update_stream_row(self.current_row())
        self._preview_timer.start()
        self._edited()

    def remove_channel(self) -> None:
        st, r = self.current_stream(), self.current_channel()
        if st is None or len(st.channels) <= 1 or not 0 <= r < len(st.channels):
            return
        del st.channels[r]
        self._fill_channels(st, select=max(r - 1, 0))
        self._update_stream_row(self.current_row())
        self._preview_timer.start()
        self._edited()

    # ---------------------------------------------------------------- senaryo / dosya
    def apply_scenario(self, name: str | None = None) -> None:
        name = name or self.scenario.currentText()
        self.scenario.setCurrentText(name)
        self.cfg.streams = sg.scenario(name)
        self._fill_streams(select=0)
        self._edited()
        self.cfg.scenario = name

    def _save_json(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Ayarları kaydet", str(Path.home() / "sinyal_test.json"),
                                              "JSON (*.json)")
        if path:
            self.save_json(Path(path))

    def save_json(self, path: Path) -> None:
        path.write_text(json.dumps(sg.config_to_dict(self.cfg), ensure_ascii=False, indent=2), encoding="utf-8")
        self.statusBar().showMessage(f"Kaydedildi: {path.name}", 4000)

    def _load_json(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Ayarları aç", str(Path.home()), "JSON (*.json)")
        if path:
            self.load_json(Path(path))

    def load_json(self, path: Path) -> None:
        try:
            cfg = sg.config_from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, TypeError, ValueError) as exc:
            QMessageBox.warning(self, APP_TITLE, f"Dosya okunamadı:\n{exc}")
            return
        self.cfg = cfg
        self._load_into_widgets()
        self._edited()

    def save_settings(self) -> None:
        self.settings.setValue("config", json.dumps(sg.config_to_dict(self.cfg), ensure_ascii=False))
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.sync()

    # ================================================================ gönderim
    def toggle(self) -> None:
        if self.engine.running:
            self.stop()
        else:
            self.start()

    def _bump_revisions(self) -> None:
        """Tanımlayıcısı son gönderimden beri değişen akışın desc_rev'i artar (protokol: yeniden yapılandırma)."""
        for st in self.cfg.streams:
            key = sg.descriptor_key(st)
            old = self._sent_keys.get(st.stream_id)
            if old is not None and old != key:
                st.rev = st.rev % 0xFFFF + 1
            self._sent_keys[st.stream_id] = key

    def start(self) -> None:
        if not self.cfg.enabled_streams():
            self.badge.set_state("Gönderilecek akış yok", "warning")
            return
        ids = [s.stream_id for s in self.cfg.enabled_streams()]
        if len(ids) != len(set(ids)):
            self.badge.set_state("Akış kimlikleri benzersiz olmalı", "critical")
            return
        self._bump_revisions()
        self._live.clear()
        self.engine.start(sg.config_to_dict(self.cfg))
        self.badge.set_state("Başlatılıyor…", "idle")
        self.start_btn.setText("  Durdur")
        self.start_btn.setIcon(primary_icon("ph.stop"))
        self.save_settings()

    def stop(self) -> None:
        self._restart.stop()
        self.engine.stop()

    def _apply_restart(self) -> None:
        if self.engine.running:
            self.start()

    def _on_started(self, info: dict) -> None:
        local = info.get("local") or ("", 0)
        self.badge.set_state(f"Gönderiliyor → {info['host']}:{info['port']}", "good")
        self.error_label.setText("")
        self.statusBar().showMessage(f"Kaynak {local[0]}:{local[1]} · {len(info['streams'])} akış", 5000)

    def _on_stopped(self) -> None:
        if not self.engine.running:
            self.start_btn.setText("  Başlat")
            self.start_btn.setIcon(primary_icon("ph.play"))
            if not self.badge.text().endswith(("Hata", "olmalı", "yok")):
                self.badge.set_state("Durduruldu", "idle")
            self._live.clear()
            self._refresh_live()

    def _on_failed(self, text: str) -> None:
        self.badge.set_state("Hata", "critical")
        self.error_label.setText(text.strip().splitlines()[-1][:120])
        self.error_label.setToolTip(text)

    def _on_stats(self, stats: dict) -> None:
        self._live = {row["stream_id"]: row for row in stats["streams"]}
        self._refresh_live()
        pps = sum(r["pps"] for r in stats["streams"])
        late = [r["name"] for r in stats["streams"] if r["late"]]
        text = f"Gönderilen {_rate(stats['bytes_per_s'])} · {pps:,.0f} paket/s".replace(",", " ")
        if late:
            text += f" · ⚠ gerçek zamana yetişemiyor: {', '.join(late)}"
        self.total_label.setText(text)
        wire = stats["wire_bits_per_s"]
        iface = getattr(self, "_iface", None)
        usage = f" · hat %{100 * wire / iface.speed_bps:.1f}" if iface is not None and iface.speed_bps else ""
        self.link_label.setText(f"Hat {wire / 1e6:.2f} Mbit/s{usage}")
        if stats["errors"]:
            self.error_label.setText(f"Gönderim hatası ×{stats['errors']}: {stats['last_error']}")
        if late:
            self.badge.set_state("Yetişemiyor", "warning")
        elif self.badge.text().endswith("Yetişemiyor"):
            self.badge.set_state(f"Gönderiliyor → {self.cfg.host}:{self.cfg.port}", "good")

    def _refresh_live(self) -> None:
        t = self.stream_table
        self._syncing = True
        for r, st in enumerate(self.cfg.streams):
            row = self._live.get(st.stream_id)
            item = t.item(r, len(STREAM_COLS) - 1)
            if item is None:
                continue
            if row is None:
                item.setText("—" if not self.engine.running or not st.enabled else "bekliyor")
                item.setForeground(self.palette().placeholderText())
                continue
            text = (f"{row['pps']:,.0f} paket/s · {_rate(row['bytes_per_s'])} · gecikme {row['lag_ms']:.0f} ms"
                    .replace(",", " "))
            extra = [f"atılan {row['lost']}" if row["lost"] else "", f"yinelenen {row['duplicated']}"
                     if row["duplicated"] else "", f"sırası değişen {row['reordered']}" if row["reordered"] else ""]
            extra = [e for e in extra if e]
            if extra:
                text += " · " + ", ".join(extra)
            item.setText(text)
            color = self.theme.tokens.warning if row["late"] else self.theme.tokens.text
            item.setForeground(QBrush(QColor(color)))
        self._syncing = False

    def _update_route(self) -> None:
        host = self.cfg.host
        self._iface = netinfo.interface_for(host) if host else None
        iface = self._iface
        if iface is None:
            self.route.setText("Yol: bilinmiyor (hedef ağda değil ya da ulaşılamıyor)")
        else:
            self.route.setText(f"Yol: {iface.name} · {iface.kind} · {iface.speed_text} · MTU {iface.mtu or '?'}")

    # ================================================================ önizleme
    def _draw_preview(self) -> None:
        st = self.current_stream()
        self.plot.clear()
        if st is None:
            self.preview_legend.set_items([])
            self.preview_info.setText("")
            return
        # Pencere: kanalların ortanca frekansının ≈ 3 periyodu. En yavaşa göre seçilirse hızlı kanallar dolu bant
        # olur; seyreltilmiş örnekler de katlanır (aliasing). Bu yüzden nokta sınırı içinde her örnek çizilir.
        freqs = sorted(c.freq for c in st.channels[:PREVIEW_CHANNELS] if c.freq > 0 and c.kind not in ("dc", "noise"))
        span = 3.0 / freqs[len(freqs) // 2] if freqs else 0.05
        span = min(max(span, 2.0 / st.fs, 0.002), 2.0, PREVIEW_POINTS / st.fs)
        n = np.arange(max(2, int(span * st.fs)), dtype=np.int64)
        t = n / st.fs
        items = []
        for i, ch in enumerate(st.channels[:PREVIEW_CHANNELS]):
            rng = np.random.default_rng(i)
            y = sg.channel_values(ch, n, st.fs, rng, complex_out=False)
            color = self.theme.series_color(i)
            self.plot.plot(t, y, pen=pen(color, 1.5))
            items.append((color, ch.name))
        self.preview_legend.set_items(items)
        self.preview_info.setText(f"{format_rate(st.fs)} · ilk {span * 1e3:.0f} ms")

    def sizeHint(self) -> QSize:  # noqa: N802 — tablolar tüm içeriklerini önerir (≈ 2600 px); makul ilk boy
        return QSize(1320, 820)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if not self._titlebar_styled:   # yerel pencere gösterimden önce oluşturulursa ilk boy kayboluyordu
            style_native_titlebar(self, self.theme.tokens)
            self._titlebar_styled = True

    def closeEvent(self, event) -> None:
        self.engine.stop()
        self.save_settings()
        super().closeEvent(event)
