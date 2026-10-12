"""Canlı İzleme: paylaşımlı tampondan zaman domeni ve spektrum grafikleri.

Yakınlaştırma (MATLAB benzeri): sol tıkla kutu çizerek yakınlaşma, tekerlekle iki eksende yakınlaşma,
orta tuşla (veya Kaydır aracıyla) kaydırma, çift tıkla sıfırlama. Kullanıcı yakınlaştırınca görünüm canlı
takibi bırakır ("inceleme") ve görünen aralığın verisi tampondan tam çözünürlükte yeniden okunur; böylece
zamanda yakınlaştıkça sinyalin ayrıntısı, en sonunda tek tek örnekler görünür.

İşlem zinciri canlı uygulanıyorsa (Aşama 3) her kanalın filtrelenmiş hâli aynı zaman ekseninde çizilir: ham sinyal
nötr gri, filtrelenmiş sinyal kanalın kendi renginde (ya da yalnız filtrelenmiş). Kanal renkleri filtre açılıp
kapanınca değişmez. Spektrumda da iki eğri üst üste görünür.

Aynı sınıf `review=True` ile İnceleme görünümü olarak kullanılır: Çalışma Alanı'ndaki dondurulmuş anlık
görüntüler canlı akış gibi çizilir, başlangıçta tamamı gösterilir ve spektrum görünen aralıktan hesaplanır.
"""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QPen
from PyQt6.QtWidgets import (
    QButtonGroup,
    QHBoxLayout,
    QPushButton,
    QSplitter,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from spt.dsp.convert import physical_column
from spt.dsp.display import decimate, mask_gaps
from spt.dsp.spectrum import ONE_SIDED, TWO_SIDED, amplitude_spectrum
from spt.gui.icons import icon
from spt.gui.theme import ThemeManager
from spt.gui.widgets.common import Card, PageHeader, chip, hbox, icon_button, label, legend_chip
from spt.gui.widgets.inspector import (
    MAX_STATS_SAMPLES,
    DataTips,
    PlotAnalysis,
    Trace,
    fmt,
    group,
    spectrum_stats,
    time_stats,
    visible_slice,
)
from spt.gui.widgets.plot import Crosshair, make_plot, pen, set_si_prefix

SPECTRUM_EVERY = 3     # spektrum her 3 karede bir (~10 Hz) güncellenir
MAX_LEGEND = 8
# Yoğun canlı verilerde Qt'nin kalın + yumuşatmalı çizgi çizimi ~100 kat yavaş (ölçüm: 2 px 3–12 s,
# 1 px 30–90 ms / kare). Canlı grafikler bu yüzden 1 px ve yumuşatmasız çizilir.
LIVE_LINE_WIDTH = 1
MARKER_MIN_PX = 6      # örnekler arası en az bu kadar piksel varsa her örnek nokta olarak gösterilir
MAX_EVENT_LINES = 100  # akış başına gösterilen en fazla SPTF MARKER çizgisi (eskiler silinir)
EVENT_LABEL_CHARS = 28


def format_seconds(seconds: float) -> str:
    if seconds < 1e-3:
        return f"{seconds * 1e6:.3g} µs"
    return f"{seconds * 1e3:.3g} ms" if seconds < 1 else f"{seconds:.3g} s"


class LiveView(QWidget):
    listen_requested = pyqtSignal()
    open_requested = pyqtSignal()
    simulator_requested = pyqtSignal()
    sides_changed = pyqtSignal(str)  # spektrum tek/çift taraflı (kullanıcı kart başlığından değiştirdiğinde)
    unit_changed = pyqtSignal(str)   # görünen kanalların ortak birimi ("" = karışık / yok)
    freeze_requested = pyqtSignal()

    def __init__(self, theme: ThemeManager, review: bool = False):
        super().__init__()
        self._theme = theme
        self.review = review
        self.spectrum_sides = ONE_SIDED
        self.processor = None            # LiveFilter: işlem zinciri canlı uygulanıyorsa
        self.filter_display = "both"     # "both": ham soluk + filtreli, "filtered": yalnız filtreli
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(10)

        if review:
            self.header = PageHeader("İnceleme", "Çalışma Alanı'ndan bir anlık görüntü seçin")
        else:
            self.header = PageHeader("Canlı İzleme", "Akış bağlı değil")
        self.window_chip = chip("")
        self.summary_chip = chip("")
        self.header.add_right(self._build_tools())
        self.header.add_right(self.summary_chip)
        self.header.add_right(self.window_chip)
        lay.addWidget(self.header)

        self.pages = QStackedWidget()
        self.pages.addWidget(self._build_empty())
        self.pages.addWidget(self._build_plots(theme))
        lay.addWidget(self.pages, 1)

        self._streams: dict[str, object] = {}         # anahtar → StreamHandle
        self._colors: dict[tuple[str, int], str] = {}
        self._visible: dict[tuple[str, int], bool] = {}
        self._dashed: dict[tuple[str, int], bool] = {}  # rengi başka kanalla aynı (9+. kanal): kesikli çizilir
        self._unit = ""
        self._time_curves: dict[tuple[str, int, int], pg.PlotDataItem] = {}
        self._markers_on: dict[tuple[str, int, int], bool] = {}
        self._event_lines: dict[str, list[pg.InfiniteLine]] = {}  # akış → SPTF MARKER çizgileri
        self._spec_curves: dict[tuple[str, int], pg.PlotDataItem] = {}
        self._filt_curves: dict[tuple[str, int, int], pg.PlotDataItem] = {}
        self._filt_spec: dict[tuple[str, int], pg.PlotDataItem] = {}
        self._frozen: dict[str, int] = {}             # durdurulduğunda akış başına son örnek
        self._pinned: dict[str, int] = {}             # bir yenileme boyunca sabit tutulan uç indeks
        self.running = True
        self.follow = True                             # False: kullanıcı yakınlaştırdı (inceleme)
        self.window_s = 0.1
        self.y_auto, self.y_range = True, (-2.0, 2.0)
        self.peak = True
        self.fft_size, self.fft_window, self.kaiser_beta = 8192, "Hann", 8.6
        self._tick = 0
        self._redraw = QTimer(self)
        self._redraw.setSingleShot(True)
        self._redraw.setInterval(0)
        self._redraw.timeout.connect(self._draw_visible)
        self._update_chips()

    # ---------------------------------------------------------------- kurulum
    def _build_tools(self) -> QWidget:
        self.zoom_tool = icon_button("ph.magnifying-glass-plus", "Kutu yakınlaştırma: sol tıkla sürükle "
                                     "(tekerlek: yakınlaş/uzaklaş, çift tık: sıfırla)", size=16)
        self.pan_tool = icon_button("ph.hand", "Kaydırma: sol tıkla sürükle (orta tuş her zaman kaydırır)", size=16)
        group = QButtonGroup(self)
        for btn in (self.zoom_tool, self.pan_tool):
            btn.setCheckable(True)
            group.addButton(btn)
        self.zoom_tool.setChecked(True)
        group.buttonToggled.connect(lambda *_: self._apply_mouse_mode())
        return hbox(self.zoom_tool, self.pan_tool, spacing=4)

    def _build_empty(self) -> QWidget:
        card = Card("")
        card.title.hide()
        box = QVBoxLayout()
        box.setSpacing(10)
        box.addStretch(1)
        glyph = label("")
        glyph.setPixmap(icon("ph.snowflake" if self.review else "ph.chart-line",
                             self._theme.tokens.accent).pixmap(40, 40))
        glyph.setAlignment(Qt.AlignmentFlag.AlignCenter)
        box.addWidget(glyph)
        title = label("İncelenecek sinyal yok" if self.review else "Henüz veri yok", "EmptyTitle")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        box.addWidget(title)
        if self.review:
            message = ("Canlı akışı Dondur (F6) ile Çalışma Alanı'na alın, ardından listeden seçin.<br>"
                       "Birden çok seçim (ör. ham + filtreli) aynı zaman ekseninde üst üste çizilir.")
            buttons = (("Dondur (F6)", "ph.snowflake", self.freeze_requested, True),)
        else:
            message = ("Ethernet üzerinden canlı veri almak için UDP dinlemeyi başlatın.<br>"
                       "Ağ bağlantısı yoksa daha önce kaydedilmiş bir .sptr dosyasını açıp analiz edebilir<br>"
                       "ya da simülatörle test sinyali gönderebilirsiniz.")
            buttons = (("Dinlemeyi Başlat", "ph.broadcast", self.listen_requested, True),
                       ("Kayıt Aç…", "ph.film-strip", self.open_requested, False),
                       ("Simülatör…", "ph.cpu", self.simulator_requested, False))
        text = label(message, "Muted")
        text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        box.addWidget(text)
        row = QHBoxLayout()
        row.addStretch(1)
        for text, icon_name, signal, primary in buttons:
            btn = QPushButton(icon(icon_name), f"  {text}")
            if primary:
                btn.setObjectName("Primary")
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.clicked.connect(signal.emit)
            row.addWidget(btn)
        row.addStretch(1)
        box.addSpacing(6)
        box.addLayout(row)
        box.addStretch(2)
        card.body.addLayout(box)
        return card

    def _build_plots(self, theme: ThemeManager) -> QWidget:
        split = QSplitter(Qt.Orientation.Vertical)
        split.setChildrenCollapsible(False)

        self.time_card = Card("Zaman Domeni")
        self.legend = QWidget()
        self._legend_lay = QHBoxLayout(self.legend)
        self._legend_lay.setContentsMargins(0, 0, 0, 0)
        self._legend_lay.setSpacing(10)
        self.time_card.add_left(self.legend)
        self.filter_chip = label("", "DemoChip")
        self.filter_chip.hide()
        self.time_card.add_left(self.filter_chip)
        self.time_readout = label("", "Readout")
        self.time_plot = make_plot(theme, "Zaman", "Genlik", x_units="s", si_prefix=False)
        set_si_prefix(self.time_plot.getPlotItem().getAxis("bottom"), True)
        self.live_button = QPushButton(icon("ph.broadcast", theme.tokens.accent), " Canlıya dön")
        self.live_button.setObjectName("LiveButton")
        self.live_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.live_button.setToolTip("Yakınlaştırmayı bırak, en son veriyi takip et (çift tık da aynı işi yapar)")
        self.live_button.clicked.connect(self.return_to_live)
        self.live_button.hide()
        fit = icon_button("ph.arrows-out", "Tümünü göster" if self.review else "Görünümü sıfırla ve canlı takibe dön",
                          object_name="FlatIcon", size=16)
        fit.clicked.connect(self.return_to_live)
        self.time_card.add_right(self.time_readout, optional=True)
        self.time_card.add_right(self.live_button)
        self.time_card.add_right(fit)
        self.time_card.body.addWidget(self.time_plot)
        split.addWidget(self.time_card)

        self.spec_card = Card("Spektrum")
        self.spec_card.add_left(self._build_sides_toggle())
        self.spec_chip = chip("")
        self.spec_card.add_left(self.spec_chip)
        self.spec_readout = label("", "Readout")
        self.spec_plot = make_plot(theme, "Frekans", "Genlik (dB)", x_units="Hz")
        set_si_prefix(self.spec_plot.getPlotItem().getAxis("left"), False)
        fit_spec = icon_button("ph.arrows-out", "Spektrum görünümünü sıfırla", object_name="FlatIcon", size=16)
        fit_spec.clicked.connect(self._reset_spectrum_view)
        self.spec_card.add_right(self.spec_readout, optional=True)
        self.spec_card.add_right(fit_spec)
        self.spec_card.body.addWidget(self.spec_plot)
        split.addWidget(self.spec_card)
        split.setSizes([560, 420])

        t = theme.tokens
        tint = QColor(t.accent)
        tint.setAlpha(40)
        for plot in (self.time_plot, self.spec_plot):
            vb = plot.getPlotItem().getViewBox()
            vb.rbScaleBox.setPen(pg.mkPen(t.accent, width=1))
            vb.rbScaleBox.setBrush(pg.mkBrush(tint))
        time_vb = self.time_plot.getPlotItem().getViewBox()
        time_vb.setAutoVisible(y=True)
        time_vb.sigRangeChangedManually.connect(self._on_manual_range)
        time_vb.sigXRangeChanged.connect(self._on_x_range)
        self.time_plot.scene().sigMouseClicked.connect(self._on_time_click)
        self.spec_plot.scene().sigMouseClicked.connect(
            lambda ev: self._reset_spectrum_view() if ev.double() else None)
        self._apply_mouse_mode()

        Crosshair(self.time_plot, theme, self._on_time_hover)
        Crosshair(self.spec_plot, theme, self._on_spec_hover)
        # Seçilen noktanın değerleri (veri ipuçları) ve görünen aralığın ayrıntılı analizi
        self.time_tips = DataTips(self.time_plot, theme, self._time_traces)
        self.spec_tips = DataTips(self.spec_plot, theme, self._spec_traces)
        self.time_analysis = PlotAnalysis(self.time_card, self.time_plot, theme, self._time_analysis)
        self.spec_analysis = PlotAnalysis(self.spec_card, self.spec_plot, theme, self._spec_analysis)
        return split

    def _build_sides_toggle(self) -> QWidget:
        self._sides_buttons = {}
        group = QButtonGroup(self)
        for sides, text, tip in (
                (ONE_SIDED, "Tek taraflı", "0 … fs/2. Gerçel sinüs A genliğinde okunur."),
                (TWO_SIDED, "Çift taraflı", "−fs/2 … +fs/2. Gerçel sinüs ±f'de A/2 (−6 dB) olarak iki lobda görünür; "
                                            "karmaşık (I/Q) sinyallerde negatif frekanslar gerçek bilgidir.")):
            btn = QToolButton()
            btn.setObjectName("Segment")
            btn.setText(text)
            btn.setToolTip(tip)
            btn.setCheckable(True)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.clicked.connect(lambda _=False, s=sides: self.set_spectrum_sides(s, notify=True))
            group.addButton(btn)
            self._sides_buttons[sides] = btn
        self._sides_buttons[ONE_SIDED].setChecked(True)
        return hbox(*self._sides_buttons.values(), spacing=2)

    def set_spectrum_sides(self, sides: str, notify: bool = False) -> None:
        if sides == self.spectrum_sides:
            return
        self.spectrum_sides = sides
        self._sides_buttons[sides].setChecked(True)
        self._update_chips()
        self.refresh(force=True)
        self._reset_spectrum_view()
        if notify:
            self.sides_changed.emit(sides)

    def _apply_mouse_mode(self) -> None:
        mode = pg.ViewBox.RectMode if self.zoom_tool.isChecked() else pg.ViewBox.PanMode
        for plot in (self.time_plot, self.spec_plot):
            plot.getPlotItem().getViewBox().setMouseMode(mode)

    # ---------------------------------------------------------------- yakınlaştırma / takip
    def _on_manual_range(self, *_) -> None:
        """Kullanıcı fareyle yakınlaştırdı/kaydırdı: canlı takibi bırak, görünen aralığı yeniden oku."""
        if self.follow:
            self.follow = False
            self.live_button.setVisible(not self.review)
        self._update_chips()
        self._redraw.start()

    def _on_x_range(self, *_) -> None:
        if not self.follow:
            self._update_chips()
            self._redraw.start()

    def _on_time_click(self, event) -> None:
        if event.double():
            self.return_to_live()

    def return_to_live(self) -> None:
        self.follow = True
        self.live_button.hide()
        vb = self.time_plot.getPlotItem().getViewBox()
        if self.y_auto:
            vb.enableAutoRange(axis=pg.ViewBox.YAxis)
        else:
            self.time_plot.setYRange(*self.y_range, padding=0)
        self._update_chips()
        self.refresh(force=True)

    def _reset_spectrum_view(self) -> None:
        self.spec_plot.getPlotItem().enableAutoRange()

    def set_running(self, running: bool) -> None:
        """Durdurulunca görüntü o andaki son örnekte dondurulur; yakınlaştırma bu veriyle çalışır."""
        self.running = running
        if running:
            self._frozen.clear()
            self.return_to_live()
        else:
            self._frozen = {key: h.latest_index() for key, h in self._streams.items()}

    def end_index(self, key: str, handle) -> int:
        """Çizilecek son örnek: durdurulduysa o an, bir yenileme sırasındaysa yenilemenin başındaki an (filtre,
        zaman ve spektrum aynı uç indeksi kullansın), aksi halde en son örnek."""
        if key in self._frozen:
            return self._frozen[key]
        if key in self._pinned:
            return self._pinned[key]
        return handle.latest_index()

    # ---------------------------------------------------------------- akışlar
    def add_stream(self, handle, colors: list[str], visible: list[bool]) -> None:
        self.remove_stream(handle.key)
        self._streams[handle.key] = handle
        for ch in range(handle.desc.channel_count):
            key = (handle.key, ch)
            self._dashed[key] = colors[ch] in self._colors.values()
            self._colors[key] = colors[ch]
            self._visible[key] = visible[ch]
            parts = 2 if handle.desc.sample_type.is_complex else 1
            for part in range(parts):
                p = self._pen(key, part, colors[ch])
                curve = self.time_plot.plot(pen=p, connect="finite", antialias=False)
                curve.setVisible(visible[ch])
                self._time_curves[(handle.key, ch, part)] = curve
                filtered = self.time_plot.plot(pen=QPen(p), connect="finite", antialias=False)
                filtered.setVisible(False)
                self._filt_curves[(handle.key, ch, part)] = filtered
            spec = self.spec_plot.plot(pen=self._pen(key, 0, colors[ch]), antialias=False)
            spec.setVisible(visible[ch])
            self._spec_curves[key] = spec
            filtered_spec = self.spec_plot.plot(pen=self._pen(key, 0, colors[ch]), antialias=False)
            filtered_spec.setVisible(False)
            self._filt_spec[key] = filtered_spec
        self.pages.setCurrentIndex(1)
        self._apply_filter_style()
        self._update_chips()
        self._rebuild_legend()

    def add_event_marker(self, key: str, sample_index: int, code: int, text: str, pos) -> None:
        """Cihazdan gelen SPTF MARKER: akışın `pos` tampon indeksindeki örneğine etiketli, kesikli dikey çizgi."""
        handle = self._streams.get(key)
        if handle is None or pos is None:
            return
        tok = self._theme.tokens
        caption = text or f"#{code}"
        if len(caption) > EVENT_LABEL_CHARS:
            caption = caption[:EVENT_LABEL_CHARS - 1] + "…"
        line = pg.InfiniteLine(pos=handle.t0 + pos / handle.fs, angle=90, movable=False,
                               pen=pg.mkPen(tok.text_secondary, width=1, style=Qt.PenStyle.DashLine),
                               label=caption, labelOpts={"position": 0.94, "color": tok.text, "fill": tok.surface})
        line.setToolTip(f"İşaret · kod {code} · örnek {sample_index}\n{text}")
        line.setZValue(5)
        self.time_plot.addItem(line)
        lines = self._event_lines.setdefault(key, [])
        lines.append(line)
        while len(lines) > MAX_EVENT_LINES:
            self.time_plot.removeItem(lines.pop(0))

    def remove_stream(self, key: str) -> None:
        for line in self._event_lines.pop(key, []):
            self.time_plot.removeItem(line)
        if self._streams.pop(key, None) is None:
            return
        self.time_tips.clear()  # ipuçları kaldırılan akışın eğrilerine bağlıydı
        self.spec_tips.clear()
        for k in [k for k in self._time_curves if k[0] == key]:
            self.time_plot.removeItem(self._time_curves.pop(k))
            self.time_plot.removeItem(self._filt_curves.pop(k))
            self._markers_on.pop(k, None)
            self._markers_on.pop(("f", *k), None)
        for k in [k for k in self._spec_curves if k[0] == key]:
            self.spec_plot.removeItem(self._spec_curves.pop(k))
            self.spec_plot.removeItem(self._filt_spec.pop(k))
        self._frozen.pop(key, None)
        for k in [k for k in self._colors if k[0] == key]:
            self._colors.pop(k, None)
            self._visible.pop(k, None)
            self._dashed.pop(k, None)
        if self.processor is not None:
            self.processor.forget(key)
        self._update_chips()
        self._rebuild_legend()

    def clear(self, show_empty: bool = False) -> None:
        for key in list(self._streams):
            self.remove_stream(key)
        self.time_tips.clear()
        self.spec_tips.clear()
        if show_empty:
            self.pages.setCurrentIndex(0)
            if not self.follow:
                self.return_to_live()

    def set_channel_visible(self, key: str, channel: int, visible: bool) -> None:
        self._visible[(key, channel)] = visible
        self._apply_filter_style()
        self._rebuild_legend()
        if visible:
            self.refresh(force=True)

    # ---------------------------------------------------------------- canlı filtre
    @property
    def filtering(self) -> bool:
        return self.processor is not None and self.processor.active

    def set_processor(self, processor) -> None:
        self.processor = processor
        self.filter_state_changed()

    def set_filter_display(self, mode: str) -> None:
        self.filter_display = mode
        self.filter_state_changed()

    def filter_state_changed(self) -> None:
        """Zincir, canlı uygulama ya da gösterim değişti: eğri stillerini yenile ve yeniden çiz."""
        self.time_tips.clear()  # filtreli değerler değişti: eski ipucu yanlış değer göstermesin
        self.spec_tips.clear()
        self._apply_filter_style()
        self._rebuild_legend()
        on = self.filtering
        if on:
            n = self.processor.chain.enabled_count
            self.filter_chip.setText(f"FİLTRE · {n} aşama")
            self.filter_chip.setToolTip("İşlem zinciri canlı uygulanıyor: " + self.processor.chain.describe())
        self.filter_chip.setVisible(on)
        self.refresh(force=True)

    def _pen(self, key: tuple[str, int], part: int, color) -> QPen:
        """Kanal kalemi: 9+. kanal rengi tekrar ettiğinde kesikli; karmaşık sinyalde Q bileşeni ayrıca farklı."""
        p = pen(color, LIVE_LINE_WIDTH)
        dashed = self._dashed.get(key, False)
        if part:
            p.setStyle(Qt.PenStyle.DotLine if dashed else Qt.PenStyle.DashLine)
        elif dashed:
            p.setStyle(Qt.PenStyle.DashLine)
        return p

    def _raw_color(self, color: str) -> str:
        """Ham eğrinin rengi: filtre açık ve ikisi birlikte gösteriliyorsa nötr gri (görünen kanal renklerinden
        ayrışan en sakin ton), değilse kanal rengi."""
        if self.filtering and self.filter_display == "both":
            visible = [self._colors[k] for k, v in self._visible.items() if v and k in self._colors]
            return self._theme.raw_trace_for(visible)
        return color

    def _apply_filter_style(self) -> None:
        on = self.filtering
        self._markers_on.clear()  # nokta renkleri (soluk / tam) bir sonraki çizimde yeniden ayarlanır
        only_filtered = on and self.filter_display == "filtered"
        for (key, ch, part), curve in self._time_curves.items():
            visible = self._visible.get((key, ch), True)
            color = self._colors[(key, ch)]
            curve.setPen(self._pen((key, ch), part, self._raw_color(color)))
            curve.setVisible(visible and not only_filtered)
            filtered = self._filt_curves[(key, ch, part)]
            filtered.setVisible(visible and on)
            if not on:
                filtered.setData([], [])
        for key, spec in self._spec_curves.items():
            visible = self._visible.get(key, True)
            color = self._colors[key]
            spec.setPen(self._pen(key, 0, self._raw_color(color)))
            spec.setVisible(visible and not only_filtered)
            filtered = self._filt_spec[key]
            filtered.setVisible(visible and on)
            if not on:
                filtered.setData([], [])

    def _visible_channels(self):
        for key, handle in self._streams.items():
            for ch in range(handle.desc.channel_count):
                if self._visible.get((key, ch), True):
                    yield key, handle, ch

    def _rebuild_legend(self) -> None:
        while self._legend_lay.count():
            self._legend_lay.takeAt(0).widget().deleteLater()
        items = [(self._colors[(k, ch)], self._display_name(h, ch)) for k, h, ch in self._visible_channels()]
        filtered = self.filtering
        for color, name in items[:MAX_LEGEND]:
            self._legend_lay.addWidget(legend_chip(color, f"{name} · filtreli" if filtered else name))
        if len(items) > MAX_LEGEND:
            more = label(f"+{len(items) - MAX_LEGEND}", "Readout")
            more.setToolTip("\n".join(name for _, name in items[MAX_LEGEND:]))
            self._legend_lay.addWidget(more)
        if filtered and self.filter_display == "both" and items:
            self._legend_lay.addWidget(legend_chip(self._raw_color(items[0][0]), "ham (filtresiz)"))
        units = {h.desc.channels[ch].unit for _, h, ch in self._visible_channels()}
        unit = units.pop() if len(units) == 1 else ""
        self.time_plot.getPlotItem().setLabel("left", f"Genlik ({unit})" if unit else "Genlik")
        if unit != self._unit:
            self._unit = unit
            self._update_chips()
            self.unit_changed.emit(unit)

    def _display_name(self, handle, ch: int) -> str:
        name = handle.desc.channels[ch].name or f"ch{ch}"
        snapshot = getattr(handle, "snapshot", None)
        return f"{snapshot.name.split(' · ')[0]} · {name}" if snapshot is not None else name

    # ---------------------------------------------------------------- ayarlar
    def set_subtitle(self, text: str) -> None:
        self.header.subtitle.setText(text)

    def set_window(self, seconds: float) -> None:
        self.window_s = seconds
        if self.follow:
            self.refresh(force=True)
        else:
            self.return_to_live()  # pencere seçimi, yakınlaştırmayı bırakıp yeni pencereyle takibe döner
        self._update_chips()

    def set_y(self, auto: bool, lo: float, hi: float) -> None:
        self.y_auto, self.y_range = auto, (lo, hi)
        if auto:
            self.time_plot.getPlotItem().enableAutoRange(axis="y")
        elif lo < hi:
            self.time_plot.setYRange(lo, hi, padding=0)

    def set_grid(self, on: bool) -> None:
        for plot in (self.time_plot, self.spec_plot):
            plot.showGrid(x=on, y=on, alpha=0.35)

    def set_peak(self, on: bool) -> None:
        self.peak = on
        self.refresh(force=True)

    def set_fft(self, size: int, window: str, beta: float) -> None:
        self.fft_size, self.fft_window, self.kaiser_beta = size, window, beta
        self._update_chips()
        self.refresh(force=True)

    def _update_chips(self) -> None:
        if self.follow and self.review:
            spans = [h.latest_index() - h.ring.valid_from() for h in self._streams.values()]
            fs = [h.fs for h in self._streams.values()]
            total = max((n / f for n, f in zip(spans, fs)), default=0.0)
            self.window_chip.setText(f"Süre: {format_seconds(max(total, 1e-9))}")
        elif self.follow:
            self.window_chip.setText(f"Pencere: {format_seconds(self.window_s)}")
        else:
            x0, x1 = self.time_plot.getPlotItem().getViewBox().viewRange()[0]
            self.window_chip.setText(f"İnceleme: {format_seconds(max(x1 - x0, 1e-9))}")
        n_ch = sum(h.desc.channel_count for h in self._streams.values())
        what = "anlık görüntü" if self.review else "akış"
        self.summary_chip.setText(f"{len(self._streams)} {what} · {n_ch} kanal")
        self.summary_chip.setVisible(bool(self._streams))
        reference = f"dB re 1 {self._unit}" if self._unit else "dB (1 birime göre)"
        self.spec_chip.setText(f"{self.fft_window} · {self.fft_size} nokta · {reference}")
        self.spec_plot.getPlotItem().setLabel(
            "bottom", "Frekans · çift taraflı" if self.spectrum_sides == TWO_SIDED else "Frekans", units="Hz")

    # ---------------------------------------------------------------- güncelleme
    def refresh(self, force: bool = False) -> None:
        if not self._streams or not (self.running or force):
            return
        # Alıcı süreç bu sırada yazmaya devam eder; uç indeks yenileme başında bir kez alınır ki filtre çıktısı,
        # zaman grafiği ve spektrum aynı örneğe kadar olsun (aksi halde filtreli spektrum sık sık boş kalıyordu).
        self._pinned = {key: self.end_index(key, h) for key, h in self._streams.items()}
        try:
            if self.filtering:  # yalnızca yeni gelen örnekler filtrelenir (durum taşıyarak)
                for key, handle, ch in self._visible_channels():
                    # yeni iz yalnız görünen pencere + spektrum kadar geriden başlar (tüm tampon değil)
                    history = int(self.window_s * handle.fs * 2) + self.fft_size + 1024
                    self.processor.update(handle, ch, self.end_index(key, handle), history=history)
            if self.follow:
                self._draw_following()
            else:
                self._draw_visible()
            self._tick += 1
            if force or self._tick % SPECTRUM_EVERY == 1:
                for key, handle, ch in self._visible_channels():
                    self._draw_spectrum(key, handle, ch)
                self.spec_analysis.schedule()
        finally:
            self._pinned = {}
        self.time_analysis.schedule()

    def _draw_following(self) -> None:
        columns = max(400, self.time_plot.width())
        if self.review:  # incelemede başlangıçta tüm anlık görüntü gösterilir
            first = last = None
            for key, handle, ch in self._visible_channels():
                start, end = handle.ring.valid_from(), handle.latest_index()
                self._draw_time(key, handle, ch, start, end, columns)
                t_start, t_end = handle.time_of(start), handle.time_of(end)
                first = t_start if first is None else min(first, t_start)
                last = t_end if last is None else max(last, t_end)
            if first is not None and last > first:
                self.time_plot.setXRange(first, last, padding=0)
            return
        latest = None
        for key, handle, ch in self._visible_channels():
            end = self.end_index(key, handle)
            if end == 0:
                continue
            start = max(0, end - int(np.ceil(self.window_s * handle.fs)) - 1)
            self._draw_time(key, handle, ch, start, end, columns)
            t_end = handle.time_of(end)
            latest = t_end if latest is None else max(latest, t_end)
        if latest is not None:
            self.time_plot.setXRange(latest - self.window_s, latest, padding=0)

    def _draw_visible(self) -> None:
        """İnceleme modu: yalnızca görünen zaman aralığını, ekran çözünürlüğünde yeniden oku."""
        if self.follow or not self._streams:
            return
        x0, x1 = self.time_plot.getPlotItem().getViewBox().viewRange()[0]
        columns = max(400, self.time_plot.width())
        for key, handle, ch in self._visible_channels():
            end_limit = self.end_index(key, handle)
            start = max(0, int(np.floor((x0 - handle.t0) * handle.fs)) - 1)
            end = min(end_limit, int(np.ceil((x1 - handle.t0) * handle.fs)) + 2)
            self._draw_time(key, handle, ch, start, end, columns)
        if self.review:  # incelemede spektrum görünen aralıktan hesaplanır
            for key, handle, ch in self._visible_channels():
                self._draw_spectrum(key, handle, ch)

    def _draw_time(self, key: str, handle, ch: int, start: int, end: int, columns: int) -> None:
        info = handle.desc.channels[ch]
        parts = 2 if handle.desc.sample_type.is_complex else 1
        for part in range(parts):
            curve = self._time_curves[(key, ch, part)]
            segs = handle.column_segments(ch, start, end, part) if end > start else []
            if not segs:
                curve.setData([], [])
                continue
            x, y, k = decimate(segs, columns, self.peak)
            y = y.astype(np.float64) * info.scale + info.offset
            y = mask_gaps(x, y, handle.ring.gaps_between(start, end), k, handle.ring.valid_from())
            curve.setData(handle.t0 + x / handle.fs, y, connect="finite")
            # Yeterince yakınlaşınca her örneği nokta olarak göster (MATLAB'daki gibi örnekler seçilebilir)
            self._set_markers(curve, (key, ch, part), k == 1 and len(x) * MARKER_MIN_PX <= columns, key, ch)
            if self.filtering:
                self._draw_filtered(key, handle, ch, part, start, end, columns)

    def _draw_filtered(self, key: str, handle, ch: int, part: int, start: int, end: int, columns: int) -> None:
        curve = self._filt_curves[(key, ch, part)]
        segs = self.processor.segments(key, ch, start, end, part)
        if not segs:
            curve.setData([], [])
            return
        x, y, k = decimate(segs, columns, self.peak)  # filtre çıktısı zaten fiziksel birimde; kayıplar NaN
        curve.setData(handle.t0 + x / handle.fs, np.asarray(y, dtype=np.float64), connect="finite")
        self._set_markers(curve, ("f", key, ch, part), k == 1 and len(x) * MARKER_MIN_PX <= columns, key, ch)

    def _set_markers(self, curve: pg.PlotDataItem, marker_key: tuple, show: bool, key: str, ch: int) -> None:
        if self._markers_on.get(marker_key) == show:
            return
        self._markers_on[marker_key] = show
        curve.setSymbol("o" if show else None)
        curve.setSymbolSize(5)
        curve.setSymbolPen(None)
        color = self._colors[(key, ch)]
        curve.setSymbolBrush(pg.mkBrush(color if marker_key[0] == "f" else self._raw_color(color)))

    def _draw_spectrum(self, key: str, handle, ch: int) -> None:
        """Spektrum, içinde kayıp olmayan en son pencereden hesaplanır (kayıp bölgeler sıfırla dolu olduğundan
        FFT'ye girerse genliği düşürür; 08 örneğinde −2.7 dB ölçüldü)."""
        end = self._spectrum_end(key, handle)
        n = min(self.fft_size, end - handle.ring.valid_from())
        if self.review and not self.follow:  # görünen aralıktan (en fazla FFT boyu kadar)
            x0 = self.time_plot.getPlotItem().getViewBox().viewRange()[0][0]
            n = min(n, end - max(handle.ring.valid_from(), int(np.ceil((x0 - handle.t0) * handle.fs))))
        if n < 16:
            return
        rng = handle.ring.latest_clean_range(n, end)
        if rng is None:
            return
        _, raw = handle.ring.read(*rng)
        x = physical_column(raw[:, ch], handle.desc.sample_type, handle.desc.channels[ch])
        freqs, mag = amplitude_spectrum(x, handle.fs, self.fft_window, self.kaiser_beta, self.spectrum_sides)
        self._spec_curves[(key, ch)].setData(freqs, mag)
        if self.filtering:
            y = self.processor.read_clean(key, ch, *rng)
            if y is not None:  # yoksa (ör. kayıp yeni geçtiyse) bir önceki filtreli spektrum ekranda kalır
                f2, m2 = amplitude_spectrum(y, handle.fs, self.fft_window, self.kaiser_beta, self.spectrum_sides)
                self._filt_spec[(key, ch)].setData(f2, m2)

    def _spectrum_end(self, key: str, handle) -> int:
        end = self.end_index(key, handle)
        if self.review and not self.follow:
            x1 = self.time_plot.getPlotItem().getViewBox().viewRange()[0][1]
            visible_end = int(np.floor((x1 - handle.t0) * handle.fs))
            end = int(np.clip(visible_end, handle.ring.valid_from(), end))
        return end

    # ---------------------------------------------------------------- veri ipuçları ve analiz
    def _curve_name(self, key: str, ch: int, part: int, filtered: bool) -> str:
        handle = self._streams[key]
        name = self._display_name(handle, ch)
        if handle.desc.sample_type.is_complex:
            name += ".Q" if part else ".I"
        if filtered:
            return f"{name} · filtreli"
        return f"{name} · ham" if self.filtering else name

    def _sample_value(self, key: str, ch: int, part: int, n: int, filtered: bool) -> float | None:
        handle = self._streams.get(key)
        if handle is None:
            return None
        if filtered:
            segs = self.processor.segments(key, ch, n, n + 1, part) if self.filtering else []
            value = float(segs[0][1][0]) if segs else float("nan")
        else:
            if handle.ring.gaps_between(n, n + 1):
                return None
            segs = handle.column_segments(ch, n, n + 1, part)
            if not segs:
                return None
            info = handle.desc.channels[ch]
            value = float(segs[0][1][0]) * info.scale + info.offset
        return value if np.isfinite(value) else None

    def _time_traces(self) -> list[Trace]:
        traces = []
        for (key, ch, part), curve in list(self._time_curves.items()):
            for filtered, item in ((False, curve), (True, self._filt_curves[(key, ch, part)])):
                if not item.isVisible() or (filtered and not self.filtering):
                    continue
                handle = self._streams[key]
                unit = handle.desc.channels[ch].unit
                color = self._colors[(key, ch)] if filtered else self._raw_color(self._colors[(key, ch)])

                def exact(x, key=key, ch=ch, part=part, filtered=filtered, handle=handle):
                    n = int(round((x - handle.t0) * handle.fs))
                    n = min(max(n, handle.ring.valid_from()), self.end_index(key, handle) - 1)
                    value = self._sample_value(key, ch, part, n, filtered)
                    return None if value is None else (handle.time_of(n), value)

                def describe(x, y, handle=handle, unit=unit):
                    n = int(round((x - handle.t0) * handle.fs))
                    return [f"t = {x:.7f} s", f"n = {group(n)}", f"değer = {fmt(y, unit, 6)}"]

                traces.append(Trace(item, self._curve_name(key, ch, part, filtered), color, describe, exact))
        return traces

    def _spec_traces(self) -> list[Trace]:
        traces = []
        for (key, ch), curve in list(self._spec_curves.items()):
            for filtered, item in ((False, curve), (True, self._filt_spec[(key, ch)])):
                if not item.isVisible() or (filtered and not self.filtering):
                    continue
                handle = self._streams[key]
                unit = handle.desc.channels[ch].unit
                color = self._colors[(key, ch)] if filtered else self._raw_color(self._colors[(key, ch)])

                def describe(x, y, item=item, unit=unit):
                    xs = item.xData
                    df = float(xs[1] - xs[0]) if xs is not None and len(xs) > 1 else 0.0
                    lines = [f"f = {fmt(x, 'Hz', 7)}"]
                    if df:
                        lines.append(f"bin {int(round((x - xs[0]) / df))} · Δf = {fmt(df, 'Hz')}")
                    lines += [f"{y:.2f} dB", f"genlik = {fmt(10 ** (y / 20), unit, 5)}"]
                    return lines

                traces.append(Trace(item, self._curve_name(key, ch, 0, filtered), color, describe))
        return traces

    def _time_analysis(self, x0: float, x1: float) -> list:
        columns = []
        for key, handle, ch in self._visible_channels():
            n0 = max(handle.ring.valid_from(), int(np.ceil((x0 - handle.t0) * handle.fs)))
            n1 = min(self.end_index(key, handle), int(np.floor((x1 - handle.t0) * handle.fs)) + 1)
            if n1 - n0 > MAX_STATS_SAMPLES:  # çok uzun aralıkta en yeni kısım (bellek / süre sınırı)
                n0 = n1 - MAX_STATS_SAMPLES
            if n1 <= n0:
                continue
            info = handle.desc.channels[ch]
            parts = 2 if handle.desc.sample_type.is_complex else 1
            for part in range(parts):
                if self.filter_display != "filtered" or not self.filtering:
                    y = np.full(n1 - n0, np.nan)
                    for start, values in handle.column_segments(ch, n0, n1, part):
                        y[start - n0:start - n0 + len(values)] = values.astype(np.float64) * info.scale + info.offset
                    for gs, ge in handle.ring.gaps_between(n0, n1):
                        y[max(gs, n0) - n0:min(ge, n1) - n0] = np.nan
                    columns.append((self._curve_name(key, ch, part, False),
                                    self._raw_color(self._colors[(key, ch)]),
                                    time_stats(y, handle.time_of(n0), handle.fs, info.unit)))
                if self.filtering:
                    y = np.full(n1 - n0, np.nan)
                    for start, values in self.processor.segments(key, ch, n0, n1, part):
                        y[start - n0:start - n0 + len(values)] = values
                    columns.append((self._curve_name(key, ch, part, True), self._colors[(key, ch)],
                                    time_stats(y, handle.time_of(n0), handle.fs, info.unit)))
        return columns

    def _spec_analysis(self, x0: float, x1: float) -> list:
        columns = []
        for trace in self._spec_traces():
            xs, ys = trace.item.xData, trace.item.yData
            if xs is None or not len(xs):
                continue
            sl = visible_slice(np.asarray(xs), x0, x1)
            columns.append((trace.name, trace.color, spectrum_stats(xs[sl], ys[sl])))
        return columns

    # ---------------------------------------------------------------- imleç okuması
    def _channel_label(self, key: str, ch: int) -> tuple[str, str]:
        handle = self._streams.get(key)
        if handle is None:
            return f"ch{ch}", ""
        info = handle.desc.channels[ch]
        return info.name or f"ch{ch}", info.unit

    @staticmethod
    def _value_at(curve: pg.PlotDataItem, x: float) -> float | None:
        """Spektrum eğrisinde x'e en yakın bin (özgün dizi; ekran seyreltmesi değil)."""
        if not curve.isVisible():
            return None
        xs, ys = curve.xData, curve.yData
        if xs is None or len(xs) == 0:
            return None
        i = int(np.clip(np.searchsorted(xs, x), 1, len(xs) - 1))
        i = i - 1 if abs(xs[i - 1] - x) <= abs(xs[i] - x) else i
        return float(ys[i]) if np.isfinite(ys[i]) else None

    def _on_time_hover(self, x: float | None, _y: float | None) -> None:
        if x is None:
            self.time_readout.setText("")
            return
        parts = []
        curves = [(k, c, "") for k, c in self._time_curves.items()]
        if self.filtering:
            curves = [(k, c, " · filtreli") for k, c in self._filt_curves.items()] + curves
        for (key, ch, part), curve, mark in curves:
            handle = self._streams.get(key)
            if handle is None or not curve.isVisible():
                continue
            # ekran için seyreltilmiş eğriden değil, tampondaki gerçek örnekten
            n = int(round((x - handle.t0) * handle.fs))
            if not handle.ring.valid_from() <= n < self.end_index(key, handle):
                continue
            value = self._sample_value(key, ch, part, n, bool(mark))
            if value is None:
                continue
            name, unit = self._channel_label(key, ch)
            if self._streams[key].desc.sample_type.is_complex:
                name += ".Q" if part else ".I"
            parts.append(f"{name}{mark} {value:+.4g} {unit}")
            if len(parts) == 4:
                break
        self.time_readout.setText(f"t = {x:.6f} s   " + "   ".join(parts))

    def _on_spec_hover(self, x: float | None, _y: float | None) -> None:
        if x is None:
            self.spec_readout.setText("")
            return
        parts = []
        curves = [(k, c, "") for k, c in self._spec_curves.items()]
        if self.filtering:
            curves = [(k, c, " · filtreli") for k, c in self._filt_spec.items()] + curves
        for (key, ch), curve, mark in curves:
            value = self._value_at(curve, x)
            if value is not None:
                parts.append(f"{self._channel_label(key, ch)[0]}{mark} {value:.2f} dB")
            if len(parts) == 4:
                break
        self.spec_readout.setText(f"f = {fmt(x, 'Hz', 6)}   " + "   ".join(parts))
