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
from PyQt6.QtCore import QRectF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QActionGroup, QColor, QPen
from PyQt6.QtWidgets import (
    QButtonGroup,
    QHBoxLayout,
    QMenu,
    QPushButton,
    QSplitter,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from spt.dsp.convert import physical_column
from spt.dsp.display import decimate, mask_gaps
from spt.dsp.spectrum import (
    ONE_SIDED,
    TWO_SIDED,
    SpectrumHold,
    amplitude_spectrum,
    spectral_extrema,
    spectral_peaks,
)
from spt.gui.icons import icon, primary_icon
from spt.gui.theme import ThemeManager
from spt.gui.widgets.common import Card, LegendBar, PageHeader, chip, hbox, icon_button, label, pick_color
from spt.gui.widgets.kpi import KpiStrip
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
from spt.gui.widgets.plot import AXIS_FONT_PX, Crosshair, make_plot, pen, plot_font, set_plot_font, set_si_prefix

SPECTRUM_EVERY = 3     # spektrum her 3 karede bir (~10 Hz) güncellenir
MAX_LEGEND = 8
# Yoğun canlı verilerde Qt'nin kalın (> 1 px) + yumuşatmalı çizgi çizimi ~100 kat yavaş (ölçüm: 2 px 3–12 s / kare).
# 1 px yumuşatmalı çizgi ise neredeyse yumuşatmasız kadar hızlı (920×360 grafik, iki eğri, kare başına: zaman 1800
# nokta 1.5 → 1.6 ms, spektrum 4097 nokta 4.1 → 7.6 ms) ve tırtıksız görünür. Canlı grafikler bu yüzden 1 px ve
# yumuşatmalı çizilir; çok noktalı spektrumda (32769 nokta 13 → 37 ms) yumuşatma kapatılır.
LIVE_LINE_WIDTH = 1
SMOOTH_SPECTRUM_MAX = 16385   # bu kadar spektrum noktasına (FFT ≤ 32768) kadar yumuşatmalı
PEAK_CHOICES = (0, 1, 3, 5, 10)   # spektrumda işaretlenen en yüksek tepe sayısı (0: kapalı)
DEFAULT_PEAKS = 5
# Max-hold / min-hold izlerinin renkleri temadan gelir (Theme.hold): spektrum tema vurgusunda çizildiğinden her tema
# kendi vurgusuna, ham grisine ve birbirine karşı ölçülmüş iki renk taşır (bkz. theme.HOLD_FLUENT, tests/test_theme).
# Min-hold ayrıca kesikli.
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
    window_chosen = pyqtSignal(str)  # başlıktaki pencere çipinden seçilen zaman penceresi (ör. "100 ms")
    channel_color_changed = pyqtSignal(str, int, str)   # kullanıcı lejanttan kanal rengini değiştirdi
    spectrum_colors_changed = pyqtSignal(object)        # {"spectrum" | "max" | "min": renk ya da None}

    def __init__(self, theme: ThemeManager, review: bool = False):
        super().__init__()
        self._theme = theme
        self.review = review
        self.spectrum_sides = ONE_SIDED
        self.processor = None            # LiveFilter: işlem zinciri canlı uygulanıyorsa
        self.filter_display = "both"     # "both": ham soluk + filtreli, "filtered": yalnız filtreli
        self.peak_count = DEFAULT_PEAKS  # spektrumda canlı işaretlenen en yüksek tepe sayısı
        self._peaks: list = []           # (SpectralPeak, kanal adı, eğri anahtarı) — yüksekten alçağa
        self.show_extrema = True         # görünen aralığın en büyük / en küçük noktası her yenilemede işaretli
        self.hold = {"max": True, "min": True}  # max-hold / min-hold izleri
        self._extrema: list = []         # ("Max" | "Min", SpectralPeak, kanal adı, eğri anahtarı)
        # Kullanıcının seçtiği spektrum / iz renkleri (None: tema varsayılanı) ve grafik yazı tipi
        self.spectrum_colors: dict[str, str | None] = {"spectrum": None, "max": None, "min": None}
        self.font_family: str | None = None
        self.font_px = AXIS_FONT_PX
        self._holds: dict[tuple[str, int], SpectrumHold] = {}                # kanal → max / min birikimi
        self._hold_curves: dict[tuple[str, int, str], pg.PlotDataItem] = {}  # (akış, kanal, "max" | "min")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(10)

        if review:
            self.header = PageHeader("İnceleme", "Çalışma Alanı'ndan bir anlık görüntü seçin")
        else:
            self.header = PageHeader("Canlı İzleme", "Akış bağlı değil")
        # Canlı durum rozeti + pencere seçimi + özet + fare araçları (İnceleme'de pencere yalnız bilgi çipi)
        self.live_badge = label("", "LiveBadge")
        self.live_badge.hide()
        self.window_chip = chip("") if review else self._build_window_menu()
        self.summary_chip = chip("")
        if not review:
            self.header.add_right(self.live_badge)
        self.header.add_right(self.window_chip)
        self.header.add_right(self.summary_chip)
        self.header.add_right(self._build_tools())
        lay.addWidget(self.header)
        # Canlı paket istatistikleri (paket hızı, veri hızı, paket sayısı, hatalı paket, kayıp); İnceleme'de yok
        self.kpi = None if review else KpiStrip(theme.tokens.accent)
        if self.kpi is not None:
            lay.addWidget(self.kpi)

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

    def _build_window_menu(self) -> QToolButton:
        """Zaman penceresi: başlıkta açılır çip. Seçenekler Özellikler › Görüntü'dekilerle aynıdır
        (`set_window_options`); seçim `window_chosen` ile bildirilir."""
        btn = QToolButton()
        btn.setObjectName("ChipMenu")
        btn.setToolTip("Zaman penceresi — Özellikler › Görüntü ile aynı ayar")
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self._window_menu = QMenu(btn)
        self._window_actions = QActionGroup(self)
        self._window_actions.setExclusive(True)
        btn.setMenu(self._window_menu)
        return btn

    def set_window_options(self, texts: list[str]) -> None:
        if self.review:
            return
        self._window_menu.clear()
        for act in self._window_actions.actions():
            self._window_actions.removeAction(act)
        for text in texts:
            act = self._window_menu.addAction(text)
            act.setCheckable(True)
            self._window_actions.addAction(act)
            act.triggered.connect(lambda _=False, t=text: self.window_chosen.emit(t))
        self._update_chips()

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
            message = ("Canlı veri için Kaynaklar panelinde bağlantıyı (Ethernet / UDP, USB ya da CAN-bus) seçip "
                       "dinlemeyi başlatın.<br>"
                       "Kart bağlı değilse daha önce kaydedilmiş bir .sptr dosyasını açıp analiz edebilir<br>"
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
            btn = QPushButton((primary_icon if primary else icon)(icon_name), f"  {text}")
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
        # Tıklanabilir lejant: kanalı yalnız zaman grafiğinde gizler / gösterir (spektrum tek renk, etkilenmez)
        self.legend = LegendBar(lambda: [self.time_plot], limit=MAX_LEGEND + 1)
        self.legend.context_requested.connect(self._time_legend_menu)
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
        # Bilgi çipi (FFT ayarları Özellikler'de de var): yan panel açıkken dar kartta önce lejant, sonra bu gizlenir
        self.spec_card.add_left(self.spec_chip, optional=True)
        self.peaks_button = self._build_peaks_menu()
        self.spec_card.add_left(self.peaks_button)
        # Max / min-hold lejantı (spektrumun kendisi tek renk, lejantsız); tıklayınca izi gizler / gösterir
        self.spec_legend = LegendBar(lambda: [self.spec_plot])
        self.spec_legend.context_requested.connect(self._spec_legend_menu)
        self.spec_card.add_left(self.spec_legend, optional=True)
        self.spec_readout = label("", "Readout")
        self.spec_plot = make_plot(theme, "Frekans", "Genlik (dB)", x_units="Hz")
        set_si_prefix(self.spec_plot.getPlotItem().getAxis("left"), False)
        self.hold_reset = icon_button("ph.arrow-counter-clockwise", "Max / min-hold izlerini sıfırla",
                                      object_name="FlatIcon", size=16)
        self.hold_reset.clicked.connect(self.reset_holds)
        fit_spec = icon_button("ph.arrows-out", "Spektrum görünümünü sıfırla", object_name="FlatIcon", size=16)
        fit_spec.clicked.connect(self._reset_spectrum_view)
        self.spec_card.add_right(self.spec_readout, optional=True)
        self.spec_card.add_right(self.hold_reset)
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

        self._build_peak_overlay(theme)
        Crosshair(self.time_plot, theme, self._on_time_hover)
        Crosshair(self.spec_plot, theme, self._on_spec_hover)
        # Seçilen noktanın değerleri (veri ipuçları) ve görünen aralığın ayrıntılı analizi
        self.time_tips = DataTips(self.time_plot, theme, self._time_traces)
        self.spec_tips = DataTips(self.spec_plot, theme, self._spec_traces)
        self.time_analysis = PlotAnalysis(self.time_card, self.time_plot, theme, self._time_analysis)
        self.spec_analysis = PlotAnalysis(self.spec_card, self.spec_plot, theme, self._spec_analysis)
        return split

    def _build_peaks_menu(self) -> QToolButton:
        """Spektrum işaretleri: tepe sayısı (Kapalı / 1 / 3 / 5 / 10), anlık max / min, max / min-hold izleri."""
        btn = QToolButton()
        btn.setObjectName("ChipMenu")
        btn.setToolTip("Spektrumda işaretlenen en yüksek tepeler: frekans ve seviye (dB) her spektrum yenilemesinde "
                       "hesaplanır. Gürültü tabanının 15 dB üstündeki belirgin tepeler sayılır. Aynı menüden anlık "
                       "max / min işaretleri ve max / min-hold izleri açılıp kapatılır.")
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        menu = QMenu(btn)
        menu.setToolTipsVisible(True)
        menu.addSection("Tepeler")
        self._peak_actions = QActionGroup(self)
        self._peak_actions.setExclusive(True)
        for n in PEAK_CHOICES:
            act = menu.addAction("Kapalı" if n == 0 else f"{n} tepe")
            act.setCheckable(True)
            act.setData(n)
            self._peak_actions.addAction(act)
            act.triggered.connect(lambda _=False, n=n: self.set_peak_count(n))
        menu.addSection("Sürekli göstergeler")
        self._extrema_action = menu.addAction("Anlık max / min")
        self._extrema_action.setCheckable(True)
        self._extrema_action.setChecked(self.show_extrema)
        self._extrema_action.setToolTip("Görünen frekans aralığının en yüksek ve en düşük noktası, tepe eşiğinden "
                                        "bağımsız olarak her yenilemede işaretlenir (DC dahil).")
        self._extrema_action.toggled.connect(self.set_show_extrema)
        self._hold_actions = {}
        for mode, text, tip in (
                ("max", "Max-hold izi", "Her frekansta şimdiye kadar görülen en yüksek seviye."),
                ("min", "Min-hold izi", "Her frekansta şimdiye kadar görülen en düşük seviye (kesikli).")):
            act = menu.addAction(text)
            act.setCheckable(True)
            act.setChecked(self.hold[mode])
            act.setToolTip(tip + " FFT boyu, pencere, tek / çift taraf ya da filtre değişince baştan başlar.")
            act.toggled.connect(lambda on, mode=mode: self.set_hold(mode, on))
            self._hold_actions[mode] = act
        menu.addAction(icon("ph.arrow-counter-clockwise"), "İzleri sıfırla", self.reset_holds)
        btn.setMenu(menu)
        return btn

    def _hold_color(self, mode: str) -> str:
        return self.spectrum_colors[mode] or self._theme.hold_color(mode)

    # ---------------------------------------------------------------- kullanıcı renkleri / yazı tipi
    def default_spectrum_colors(self) -> dict[str, str]:
        """Tema varsayılanları (Özellikler panelindeki renk düğmeleri bunları gösterir)."""
        return {"spectrum": self._theme.tokens.accent, "max": self._theme.hold_color("max"),
                "min": self._theme.hold_color("min")}

    def set_spectrum_colors(self, colors: dict) -> None:
        """Spektrum ve max / min-hold renkleri; None ya da eksik anahtar tema varsayılanına döner."""
        self.spectrum_colors = {k: (QColor(colors[k]).name() if colors.get(k) else None) for k in self.spectrum_colors}
        for (key, ch, mode), curve in self._hold_curves.items():
            hold_pen = pen(self._hold_color(mode), LIVE_LINE_WIDTH)
            if mode == "min":
                hold_pen.setStyle(Qt.PenStyle.DashLine)
            curve.setPen(hold_pen)
        self._apply_filter_style()   # spektrum eğrilerinin kalemleri
        self._rebuild_spec_legend()

    def set_channel_color(self, key: str, channel: int, color: str) -> None:
        """Kanalın zaman grafiğindeki rengi (spektrum tek renk; etkilenmez)."""
        if (key, channel) not in self._colors:
            return
        color = QColor(color).name()
        self._colors[(key, channel)] = color
        self._dashed[(key, channel)] = color in [c for k, c in self._colors.items() if k != (key, channel)]
        self._apply_filter_style()
        self._rebuild_legend()

    def set_plot_font(self, family: str | None, px: int) -> None:
        """Zaman ve spektrum grafiklerinin yazı tipi: eksen sayıları / başlıkları, tepe ve işaret etiketleri."""
        self.font_family, self.font_px = family or None, int(px)
        for plot in (self.time_plot, self.spec_plot):
            set_plot_font(plot, self.font_family, self.font_px)
        font = plot_font(self.font_family, self.font_px)
        for lab in self._peak_labels:
            lab.textItem.setFont(font)
        for lines in self._event_lines.values():
            for line in lines:
                if line.label is not None:
                    line.label.textItem.setFont(font)
        self._place_peak_labels()

    def _label_css(self) -> str:
        family = f" font-family:'{self.font_family}';" if self.font_family else ""
        return f"font-size:{max(self.font_px - 1, 8)}px;{family}"

    def _time_legend_menu(self, color: str, name: str, pos) -> None:
        """Zaman lejantında sağ tık: kanalın rengini değiştir (ham gri öğesi kanal değildir)."""
        target = next(((k, ch) for k, h, ch in self._visible_channels()
                       if self._colors.get((k, ch)) == color and self._legend_name(h, ch) == name), None)
        if target is None:
            return
        menu = QMenu(self)
        menu.addAction("Rengi değiştir…", lambda: self._choose_channel_color(*target, name))
        menu.exec(pos)

    def _choose_channel_color(self, key: str, channel: int, name: str) -> None:
        chosen = pick_color(self, self._colors[(key, channel)], f"{name} — kanal rengi")
        if chosen is not None:
            self.set_channel_color(key, channel, chosen)
            self.channel_color_changed.emit(key, channel, chosen)

    def _spec_legend_menu(self, color: str, name: str, pos) -> None:
        mode = {"Max-hold": "max", "Min-hold": "min"}.get(name)
        if mode is None:
            return
        menu = QMenu(self)
        menu.addAction("Rengi değiştir…", lambda: self._choose_spectrum_color(mode, name))
        if self.spectrum_colors[mode]:
            menu.addAction("Varsayılan renk", lambda: self._set_spectrum_color(mode, None))
        menu.exec(pos)

    def _choose_spectrum_color(self, mode: str, name: str) -> None:
        chosen = pick_color(self, self._hold_color(mode), f"{name} rengi")
        if chosen is not None:
            self._set_spectrum_color(mode, chosen)

    def _set_spectrum_color(self, mode: str, color: str | None) -> None:
        self.set_spectrum_colors({**self.spectrum_colors, mode: color})
        self.spectrum_colors_changed.emit(dict(self.spectrum_colors))

    def set_hold(self, mode: str, on: bool) -> None:
        on = bool(on)
        if self.hold[mode] != on:
            self.hold[mode] = on
            if not any(self.hold.values()):
                self.reset_holds()                 # ikisi de kapalı: açılınca baştan biriksin
            self._apply_filter_style()
            self._rebuild_spec_legend()
            for key in self._holds:
                self._draw_hold(key)
        self._sync_hold_menu()

    def reset_holds(self) -> None:
        """Biriken max / min izlerini siler; sonraki spektrumdan yeniden başlar."""
        for hold in self._holds.values():
            hold.reset()
        for curve in self._hold_curves.values():
            curve.setData([], [])

    def _sync_hold_menu(self) -> None:
        for mode, act in self._hold_actions.items():
            act.blockSignals(True)
            act.setChecked(self.hold[mode])
            act.blockSignals(False)
        self.hold_reset.setVisible(any(self.hold.values()))

    def _update_hold(self, key: tuple[str, int], freqs: np.ndarray, mag: np.ndarray) -> None:
        if not any(self.hold.values()):
            return
        self._holds.setdefault(key, SpectrumHold()).update(freqs, mag)
        self._draw_hold(key)

    def _draw_hold(self, key: tuple[str, int]) -> None:
        hold = self._holds.get(key)
        for mode in ("max", "min"):
            curve = self._hold_curves.get((*key, mode))
            if curve is None:
                continue
            data = None if hold is None else (hold.max if mode == "max" else hold.min)
            if data is None or not self.hold[mode]:
                curve.setData([], [])
            else:
                curve.setData(hold.freqs, data, antialias=len(data) <= SMOOTH_SPECTRUM_MAX)

    def _rebuild_spec_legend(self) -> None:
        """Açık izlerin lejantı (akış yoksa boş). Başlıkta yer yoksa kart lejantı gizler (bkz. Card.refit)."""
        items = []
        if self._streams:
            if self.hold["max"]:
                items.append((self._hold_color("max"), "Max-hold"))
            if self.hold["min"]:
                items.append((self._hold_color("min"), "Min-hold", Qt.PenStyle.DashLine))
        self.spec_legend.set_items(items)
        self.spec_card.refit()

    def set_show_extrema(self, on: bool) -> None:
        self.show_extrema = bool(on)
        self._extrema_action.blockSignals(True)
        self._extrema_action.setChecked(self.show_extrema)
        self._extrema_action.blockSignals(False)
        self._update_peaks()

    def _build_peak_overlay(self, theme: ThemeManager) -> None:
        """Tepe işaretleri (▼), anlık max (büyük ▼) / min (▲) ve etiketleri (frekans · seviye); eksen aralığını
        etkilemez."""
        t = theme.tokens
        self._peak_marks = pg.ScatterPlotItem(symbol="t", size=9, pen=pg.mkPen(t.surface, width=1),
                                              brush=pg.mkBrush(t.text))
        self._peak_marks.setZValue(30)
        self.spec_plot.addItem(self._peak_marks, ignoreBounds=True)
        fill = QColor(t.surface)
        fill.setAlpha(235)
        self._peak_labels = []
        for _ in range(max(PEAK_CHOICES) + 2):     # + max ve min
            lab = pg.TextItem(anchor=(0, 0), fill=pg.mkBrush(fill), border=pg.mkPen(t.border_strong))
            lab.setZValue(31)
            lab.hide()
            self.spec_plot.addItem(lab, ignoreBounds=True)
            self._peak_labels.append(lab)
        # yakınlaştırınca / eksen veriye oturunca görünen aralıkta yeniden aranır (iki kanalda ~1 ms)
        self.spec_plot.getPlotItem().getViewBox().sigRangeChanged.connect(lambda *_: self._update_peaks())
        self._sync_peak_menu()
        self._sync_hold_menu()

    def set_peak_count(self, n: int) -> None:
        self.peak_count = n if n in PEAK_CHOICES else DEFAULT_PEAKS
        self._sync_peak_menu()
        self._update_peaks()

    def _sync_peak_menu(self) -> None:
        for act in self._peak_actions.actions():
            act.setChecked(act.data() == self.peak_count)
        self.peaks_button.setText(f"Tepeler: {self.peak_count}" if self.peak_count else "Tepeler: kapalı")

    def _update_peaks(self) -> None:
        """Görünen spektrum eğrilerinin (filtre açıksa filtreli olanların) en yüksek tepeleri ve anlık max / min
        noktası; görünen frekans aralığında aranır. Birden çok kanal görünüyorsa etikette kanal adı da yazılır."""
        found, extrema = [], []
        if self.peak_count or self.show_extrema:
            x0, x1 = self.spec_plot.getPlotItem().getViewBox().viewRange()[0]
            curves = self._filt_spec if self.filtering else self._spec_curves
            shown = [(k, c) for k, c in curves.items() if c.isVisible() and c.xData is not None and len(c.xData) > 4]
            top = bottom = None
            for (key, ch), curve in shown:
                xs, ys = np.asarray(curve.xData), np.asarray(curve.yData)
                sl = visible_slice(xs, x0, x1)
                name = self._channel_label(key, ch)[0] if len(shown) > 1 else ""
                for peak in spectral_peaks(xs[sl], ys[sl], self.peak_count, self.fft_window, self.kaiser_beta):
                    found.append((peak, name, (key, ch)))
                if self.show_extrema:
                    hi, lo = spectral_extrema(xs[sl], ys[sl], self.fft_window, self.kaiser_beta)
                    if hi is not None and (top is None or hi.level > top[1].level):
                        top = ("Max", hi, name, (key, ch))
                    if lo is not None and (bottom is None or lo.level < bottom[1].level):
                        bottom = ("Min", lo, name, (key, ch))
            found.sort(key=lambda item: item[0].level, reverse=True)
            extrema = [e for e in (top, bottom) if e is not None]
        self._peaks = found[:self.peak_count]
        self._extrema = extrema
        self._place_peak_labels()

    def _place_peak_labels(self) -> None:
        """Etiket işaretin sağında (sağ kenara yakınsa solunda), grafiğin içinde kalacak biçimde; başka bir etiketle
        çakışan etiket gösterilmez (işareti kalır). Öncelik: max, min, tepeler (yüksekten alçağa). Max bir tepeyle
        aynı bindeyse tek işaret ve "Max" etiketi."""
        marks = [(title, peak, name) for title, peak, name, _ in self._extrema]
        taken = {(peak.index, curve) for title, peak, _, curve in self._extrema if title == "Max"}
        marks += [("", peak, name) for peak, name, curve in self._peaks if (peak.index, curve) not in taken]
        self._peak_marks.setData([p.freq for _, p, _ in marks], [p.level for _, p, _ in marks],
                                 symbol=["t1" if title == "Min" else "t" for title, _, _ in marks],
                                 size=[11 if title else 9 for title, _, _ in marks])
        vb = self.spec_plot.getPlotItem().getViewBox()
        (x0, x1), (y0, y1) = vb.viewRange()
        sx, sy = vb.viewPixelSize()
        t = self._theme.tokens
        placed: list[QRectF] = []
        for i, lab in enumerate(self._peak_labels):
            if i >= len(marks) or not sx or not sy:
                lab.hide()
                continue
            title, peak, name = marks[i]
            extra = f" · {name}" if name else ""
            head = f'<span style="color:{t.accent_text};">{title}</span> ' if title else ""
            lab.setHtml(f'<div style="color:{t.text}; {self._label_css()}">{head}<b>{fmt(peak.freq, "Hz", 4)}</b><br>'
                        f'<span style="color:{t.text_secondary};">{peak.level:.1f} dB{extra}</span></div>')
            br = lab.textItem.boundingRect()
            w, h = br.width(), br.height()
            width_px, height_px = (x1 - x0) / sx, (y1 - y0) / sy
            mx, my = (peak.freq - x0) / sx, (y1 - peak.level) / sy
            left = mx + 8 if mx + 8 + w <= width_px else mx - 8 - w
            top = min(max(my - h / 2, 0.0), max(height_px - h, 0.0))
            rect = QRectF(left, top, w, h)
            if any(rect.intersects(r) for r in placed):
                lab.hide()
                continue
            placed.append(rect)
            lab.setPos(x0 + left * sx, y1 - top * sy)
            lab.show()

    def peak_readout(self) -> list[tuple[float, float, str]]:
        """Şu an işaretli tepeler: (frekans Hz, seviye dB, kanal adı)."""
        return [(p.freq, p.level, name) for p, name, _ in self._peaks]

    def extrema_readout(self) -> dict[str, tuple[float, float, str]]:
        """Şu an işaretli anlık max / min: {"Max" | "Min": (frekans Hz, seviye dB, kanal adı)}."""
        return {title: (p.freq, p.level, name) for title, p, name, _ in self._extrema}

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
        self.reset_holds()
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
        self._update_chips()

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
                curve = self.time_plot.plot(pen=p, connect="finite", antialias=True)
                curve.setVisible(visible[ch])
                self._time_curves[(handle.key, ch, part)] = curve
                filtered = self.time_plot.plot(pen=QPen(p), connect="finite", antialias=True)
                filtered.setVisible(False)
                self._filt_curves[(handle.key, ch, part)] = filtered
            spec = self.spec_plot.plot(pen=self._pen(key, 0, self._spec_color(False)), antialias=True)
            spec.setVisible(visible[ch])
            self._spec_curves[key] = spec
            filtered_spec = self.spec_plot.plot(pen=self._pen(key, 0, self._spec_color(True)), antialias=True)
            filtered_spec.setVisible(False)
            self._filt_spec[key] = filtered_spec
            for mode in ("max", "min"):
                hold_pen = pen(self._hold_color(mode), LIVE_LINE_WIDTH)
                if mode == "min":
                    hold_pen.setStyle(Qt.PenStyle.DashLine)
                curve = pg.PlotDataItem(pen=hold_pen, antialias=True)
                curve.setZValue(-1)            # canlı spektrum izlerin üstünde
                # min-hold derin çukurlara iner: otomatik eksen canlı spektrum + max-hold'a göre kalsın
                self.spec_plot.addItem(curve, ignoreBounds=mode == "min")
                self._hold_curves[(handle.key, ch, mode)] = curve
        self.pages.setCurrentIndex(1)
        self._apply_filter_style()
        self._update_chips()
        self._rebuild_legend()
        self._rebuild_spec_legend()

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
        if line.label is not None and (self.font_family or self.font_px != AXIS_FONT_PX):
            line.label.textItem.setFont(plot_font(self.font_family, self.font_px))
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
            self._holds.pop(k, None)
        for k in [k for k in self._hold_curves if k[0] == key]:
            self.spec_plot.removeItem(self._hold_curves.pop(k))
        self._frozen.pop(key, None)
        for k in [k for k in self._colors if k[0] == key]:
            self._colors.pop(k, None)
            self._visible.pop(k, None)
            self._dashed.pop(k, None)
        if self.processor is not None:
            self.processor.forget(key)
        self._update_chips()
        self._rebuild_legend()
        self._rebuild_spec_legend()
        self._update_peaks()

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
        self.reset_holds()      # izler filtreli spektrumu izler: zincir değişince eski birikim geçersiz
        self._apply_filter_style()
        self._rebuild_legend()
        self._rebuild_spec_legend()
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

    def _legend_name(self, handle, ch: int) -> str:
        """Kanalın lejanttaki adı (_rebuild_legend ile aynı)."""
        name = self._display_name(handle, ch)
        return f"{name} · filtreli" if self.filtering else name

    def _spec_color(self, filtered: bool) -> str:
        """Spektrum kanala göre renklenmez: bütün kanallar tek renkte (vurgu). Filtre açık ve ham + filtreli birlikte
        gösteriliyorsa ham spektrum, zaman grafiğindeki gibi nötr gri. Kullanıcı spektrum rengi seçtiyse o renk."""
        if not filtered and self.filtering and self.filter_display == "both":
            return self._theme.raw_trace
        return self.spectrum_colors["spectrum"] or self._theme.tokens.accent

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
            spec.setPen(self._pen(key, 0, self._spec_color(False)))
            spec.setVisible(visible and not only_filtered)
            filtered = self._filt_spec[key]
            filtered.setPen(self._pen(key, 0, self._spec_color(True)))
            filtered.setVisible(visible and on)
            if not on:
                filtered.setData([], [])
        for (key, ch, mode), curve in self._hold_curves.items():
            curve.setVisible(self._visible.get((key, ch), True) and self.hold[mode])
        self.legend.apply()  # lejantta kapatılan kanallar görünür yapılan eğrilerde de kapalı kalsın
        self.spec_legend.apply()

    def _visible_channels(self):
        for key, handle in self._streams.items():
            for ch in range(handle.desc.channel_count):
                if self._visible.get((key, ch), True):
                    yield key, handle, ch

    def _rebuild_legend(self) -> None:
        items = [(self._colors[(k, ch)], self._display_name(h, ch)) for k, h, ch in self._visible_channels()]
        filtered = self.filtering
        legend = [(color, f"{name} · filtreli" if filtered else name) for color, name in items]
        if filtered and self.filter_display == "both" and items:
            legend.insert(min(len(legend), MAX_LEGEND), (self._raw_color(items[0][0]), "ham (filtresiz)"))
        self.legend.set_items(legend)
        units = {h.desc.channels[ch].unit for _, h, ch in self._visible_channels()}
        unit = units.pop() if len(units) == 1 else ""
        self.time_plot.getPlotItem().setLabel("left", f"Genlik ({unit})" if unit else "Genlik")
        if unit != self._unit:
            self._unit = unit
            self._update_chips()
            self.unit_changed.emit(unit)

    def _display_name(self, handle, ch: int) -> str:
        """Kanalın lejant adı. Birden çok canlı akış varsa akış adı eklenir; aynı adlı iki akış (ör. kart ve
        simülatör ya da başka IP'den gelen aynı kart) kaynak adresiyle ayrılır — yoksa üst üste çizilen iki sinyal
        aynı "ch0" adını taşır ve ayırt edilemez."""
        name = handle.desc.channels[ch].name or f"ch{ch}"
        snapshot = getattr(handle, "snapshot", None)
        if snapshot is not None:
            return f"{snapshot.name.split(' · ')[0]} · {name}"
        if len(self._streams) < 2:
            return name
        stream = handle.desc.name or f"akış {handle.desc.stream_id}"
        if sum(h.desc.name == handle.desc.name for h in self._streams.values()) > 1:
            stream += f" ({handle.source})"
        return f"{stream} · {name}"

    # ---------------------------------------------------------------- ayarlar
    def update_kpis(self, session: dict, rates: dict) -> None:
        if self.kpi is not None:
            self.kpi.update_stats(session, rates)

    def clear_kpis(self) -> None:
        if self.kpi is not None:
            self.kpi.clear()

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
        if (size, window, beta) != (self.fft_size, self.fft_window, self.kaiser_beta):
            self.reset_holds()  # pencere seviyeleri (ENBW, sızıntı) değiştirir: eski birikimle karıştırılmaz
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
        if not self.review:
            self._update_live_badge()
            current = format_seconds(self.window_s)
            for act in self._window_actions.actions():
                act.setChecked(act.text() == current)
        n_ch = sum(h.desc.channel_count for h in self._streams.values())
        what = "anlık görüntü" if self.review else "akış"
        self.summary_chip.setText(f"{len(self._streams)} {what} · {n_ch} kanal")
        self.summary_chip.setVisible(bool(self._streams))
        reference = f"dB re 1 {self._unit}" if self._unit else "dB (1 birime göre)"
        self.spec_chip.setText(f"{self.fft_window} · {self.fft_size} nokta · {reference}")
        self.spec_plot.getPlotItem().setLabel(
            "bottom", "Frekans · çift taraflı" if self.spectrum_sides == TWO_SIDED else "Frekans", units="Hz")

    def _update_live_badge(self) -> None:
        """Canlı: en son veri izleniyor · Durduruldu: görüntü güncellemesi kapalı (F7) · Takip kapalı: yakınlaştırıldı."""
        if not self.follow:
            state, text = "zoom", "Takip kapalı"
            tip = "Yakınlaştırıldı; en son veriye dönmek için “Canlıya dön” ya da çift tık"
        elif self.running:
            state, text, tip = "live", "Canlı", "En son veri izleniyor"
        else:
            state, text, tip = "paused", "Durduruldu", "Görüntü güncellemesi durduruldu (Çalıştır / Durdur: F7)"
        t = self._theme.tokens
        dot = t.accent if state == "live" else t.text_muted
        self.live_badge.setText(f'<span style="color:{dot}">●</span>&nbsp;{text}')
        self.live_badge.setToolTip(tip)
        if self.live_badge.property("state") != state:
            self.live_badge.setProperty("state", state)
            self.live_badge.style().unpolish(self.live_badge)
            self.live_badge.style().polish(self.live_badge)
        self.live_badge.setVisible(bool(self._streams))

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
                self._draw_spectra()
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
            self._draw_spectra()

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

    def _draw_spectra(self) -> None:
        for key, handle, ch in self._visible_channels():
            self._draw_spectrum(key, handle, ch)
        self._update_peaks()

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
        self._spec_curves[(key, ch)].setData(freqs, mag, antialias=len(freqs) <= SMOOTH_SPECTRUM_MAX)
        if self.filtering:
            y = self.processor.read_clean(key, ch, *rng)
            if y is not None:  # yoksa (ör. kayıp yeni geçtiyse) bir önceki filtreli spektrum ekranda kalır
                f2, m2 = amplitude_spectrum(y, handle.fs, self.fft_window, self.kaiser_beta, self.spectrum_sides)
                self._filt_spec[(key, ch)].setData(f2, m2, antialias=len(f2) <= SMOOTH_SPECTRUM_MAX)
                self._update_hold((key, ch), f2, m2)   # izler tepelerle aynı eğriyi (filtreli) izler
        else:
            self._update_hold((key, ch), freqs, mag)

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
                color = self._spec_color(filtered)

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
