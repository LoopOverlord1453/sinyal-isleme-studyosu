"""TerraFlux esinli koyu tema.

Tüm renkler tek bir `Tokens` nesnesinden gelir; QSS, QPalette, pyqtgraph ayarları
ve Windows pencere çerçevesi bu token'lardan üretilir.
"""
from __future__ import annotations

import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from string import Template

import pyqtgraph as pg
from PyQt6.QtCore import QObject, Qt
from PyQt6.QtGui import QColor, QFont, QFontDatabase, QPalette
from PyQt6.QtWidgets import QApplication, QStyleFactory, QWidget

FONT_DIR = Path(__file__).parent / "resources" / "fonts"
UI_FONT_FAMILIES = ["Inter", "Segoe UI Variable Text", "Segoe UI", "Arial"]
MONO_FONT_FAMILIES = ["Cascadia Mono", "Consolas", "Courier New"]


@dataclass(frozen=True)
class Tokens:
    bg: str              # pencere zemini
    surface: str         # kart / panel yüzeyi
    surface_alt: str     # kart içi ikincil yüzey
    elevated: str        # buton, girdi
    hover: str
    pressed: str
    border: str          # ince ayırıcı
    border_strong: str
    text: str
    text_secondary: str
    text_muted: str
    accent: str
    accent_dark: str
    accent_tint: str     # seçili öğe zemini
    good: str            # durum renkleri (seri renkleriyle karıştırılmaz)
    warning: str
    critical: str
    grid: str
    axis: str


TERRAFLUX_DARK = Tokens(
    bg="#080808",
    surface="#121110",
    surface_alt="#171513",
    elevated="#1b1917",
    hover="#24211e",
    pressed="#2c2825",
    border="#25221f",
    border_strong="#38332f",
    text="#fffdfb",
    text_secondary="#b3aca5",
    text_muted="#8a837c",
    accent="#ec7c22",
    accent_dark="#a4510e",
    accent_tint="#2c1e12",
    good="#0ca30c",
    warning="#fab219",
    critical="#d03b3b",
    grid="#1e1c1a",
    axis="#3a3531",
)

# Kanal renkleri: sabit sıra, döngü yok. #121110 yüzeyinde doğrulandı
# (açıklık bandı, kroma, komşu çiftlerde renk körlüğü ΔE ≥ 8.4, kontrast ≥ 3:1).
SERIES_DARK = ("#d95926", "#3987e5", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767")
# Filtre açıkken ham (filtresiz) sinyalin rengi: nötr açık gri. 8 kanal renginin hepsinden normal görüşte ve
# protan/deutan simülasyonunda ayrışır (en kötü OKLab ΔE 9.1 ≥ 8, #e66767 ile deutan); yüzeyde 8.4:1 kontrast.
# Daha koyu gri (#8a837c) deutan'da #d55181 ile karışıyordu (ΔE 1.6).
RAW_TRACE_DARK = "#b3aca5"
# Yalnızca ilk iki kanal rengi görünürken daha koyu gri yeterli ve filtreli eğriyi bastırmaz:
# #8a837c ↔ #d95926 normal 16.1 / CVD 11.1, ↔ #3987e5 de ≥ 8; yüzeye 5.05:1. Diğer renklerle (#199e70, #d55181,
# #e66767) renk körlüğünde karıştığından (ΔE 1.6–3.0) genel durumda açık gri kullanılır.
RAW_TRACE_SOFT = "#8a837c"
_SOFT_RAW_OK = {"#d95926", "#3987e5"}

def ui_font(px: int = 13, weight: QFont.Weight = QFont.Weight.Normal) -> QFont:
    font = QFont()
    font.setFamilies(UI_FONT_FAMILIES)
    font.setPixelSize(px)
    font.setWeight(weight)
    return font


def mono_font(px: int = 12) -> QFont:
    font = QFont()
    font.setFamilies(MONO_FONT_FAMILIES)
    font.setPixelSize(px)
    font.setStyleHint(QFont.StyleHint.Monospace)
    return font


_SVG_SHAPES = {
    "caret_down": '<path d="M4 6l4 4 4-4"/>',
    "caret_up": '<path d="M4 10l4-4 4 4"/>',
    "caret_right": '<path d="M6 4l4 4-4 4"/>',
    "check": '<path d="M3.5 8.5l3 3 6-7"/>',
}


def _write_svg_assets(t: Tokens) -> dict[str, str]:
    """QSS'in url() ile kullandığı küçük ok/onay ikonlarını tema renginde üretir."""
    out_dir = Path(tempfile.gettempdir()) / "spt_theme"
    out_dir.mkdir(exist_ok=True)
    paths = {}
    for name, shape in _SVG_SHAPES.items():
        color = t.bg if name == "check" else t.text_muted
        svg = (
            '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 16 16" '
            f'fill="none" stroke="{color}" stroke-width="1.8" stroke-linecap="round" '
            f'stroke-linejoin="round">{shape}</svg>'
        )
        path = out_dir / f"{name}.svg"
        path.write_text(svg, encoding="utf-8")
        paths[name] = path.as_posix()
    return paths


def _palette(t: Tokens) -> QPalette:
    p = QPalette()
    roles = {
        QPalette.ColorRole.Window: t.bg,
        QPalette.ColorRole.WindowText: t.text,
        QPalette.ColorRole.Base: t.elevated,
        QPalette.ColorRole.AlternateBase: t.surface,
        QPalette.ColorRole.Text: t.text,
        QPalette.ColorRole.Button: t.elevated,
        QPalette.ColorRole.ButtonText: t.text,
        QPalette.ColorRole.Highlight: t.accent_dark,
        QPalette.ColorRole.HighlightedText: t.text,
        QPalette.ColorRole.ToolTipBase: t.elevated,
        QPalette.ColorRole.ToolTipText: t.text,
        QPalette.ColorRole.PlaceholderText: t.text_muted,
        QPalette.ColorRole.Link: t.accent,
        QPalette.ColorRole.BrightText: t.critical,
        QPalette.ColorRole.Mid: t.border_strong,
        QPalette.ColorRole.Dark: t.bg,
        QPalette.ColorRole.Light: t.hover,
    }
    for role, color in roles.items():
        p.setColor(role, QColor(color))
    for role in (QPalette.ColorRole.Text, QPalette.ColorRole.ButtonText, QPalette.ColorRole.WindowText):
        p.setColor(QPalette.ColorGroup.Disabled, role, QColor(t.text_muted))
    return p


QSS = """
* { outline: 0; }
QWidget { color: $text; selection-background-color: $accent_tint; selection-color: $text; }
QMainWindow, QDialog { background: $bg; }
QMainWindow::separator { background: $bg; width: 8px; height: 8px; }
QMainWindow::separator:hover { background: $accent_tint; }
QToolTip { background: $elevated; color: $text; border: 1px solid $border_strong; border-radius: 6px; padding: 6px 8px; }

/* ---- Menü çubuğu ve menüler ---- */
QMenuBar { background: $bg; padding: 4px 8px 0 8px; }
QMenuBar::item { background: transparent; color: $text_secondary; padding: 5px 10px; border-radius: 6px; }
QMenuBar::item:selected { background: $hover; color: $text; }
QMenuBar::item:pressed { background: $pressed; color: $text; }
QMenu { background: $surface; border: 1px solid $border_strong; border-radius: 10px; padding: 6px; }
QMenu::item { color: $text; padding: 6px 28px 6px 12px; border-radius: 6px; }
QMenu::item:selected { background: $hover; }
QMenu::item:disabled { color: $text_muted; }
QMenu::separator { height: 1px; background: $border; margin: 5px 8px; }
QMenu::icon { padding-left: 10px; }
QMenu::right-arrow { image: url("$caret_right"); width: 12px; height: 12px; margin-right: 6px; }
QMenu::indicator { width: 14px; height: 14px; margin-left: 8px; border: 1px solid $border_strong; border-radius: 4px; background: $elevated; }
QMenu::indicator:checked { background: $accent; border-color: $accent; image: url("$check"); }

/* ---- Üst çubuk (header) ---- */
QToolBar#HeaderBar { background: $bg; border: none; border-bottom: 1px solid $border; padding: 8px 12px; spacing: 6px; }
QToolBar#HeaderBar::separator { background: $border; width: 1px; margin: 6px 6px; }
QLabel#Wordmark { font-size: 14px; font-weight: 600; color: $text; padding: 0 6px; }

QToolButton { background: transparent; border: 1px solid transparent; border-radius: 8px; padding: 6px; color: $text_secondary; }
QToolButton:hover { background: $hover; color: $text; }
QToolButton:pressed { background: $pressed; }
QToolButton:checked { background: $accent_tint; border-color: $accent_dark; color: $text; }
QToolButton#IconButton { background: $elevated; border: 1px solid $border; padding: 7px; }
QToolButton#IconButton:hover { background: $hover; border-color: $border_strong; }
QToolButton#IconButton:checked { background: $accent_tint; border-color: $accent_dark; }
QToolButton#FlatIcon { padding: 3px; border-radius: 6px; }
QToolButton#NavPill { padding: 6px 12px; color: $text_secondary; }
QToolButton#NavPill:checked { background: $elevated; border: 1px solid $accent_dark; color: $text; }
QToolButton#MenuButton { background: $elevated; border: 1px solid $border; padding: 6px 26px 6px 10px; color: $text; }
QToolButton#MenuButton:hover { background: $hover; border-color: $border_strong; }
QToolButton#MenuButton::menu-indicator { image: url("$caret_down"); subcontrol-position: right center; subcontrol-origin: padding; right: 8px; width: 12px; height: 12px; }

QFrame#SearchBox { background: $elevated; border: 1px solid $border; border-radius: 8px; }
QFrame#SearchBox:hover { border-color: $border_strong; }
QFrame#SearchBox QLineEdit { background: transparent; border: none; padding: 6px 2px; }
QLabel#Kbd { background: $hover; border: 1px solid $border_strong; border-radius: 5px; padding: 1px 6px; font-size: 11px; color: $text_secondary; }

/* ---- Butonlar ---- */
QPushButton { background: $elevated; border: 1px solid $border_strong; border-radius: 8px; padding: 7px 14px; color: $text; }
QPushButton:hover { background: $hover; }
QPushButton:pressed { background: $pressed; }
QPushButton:disabled { color: $text_muted; border-color: $border; }
QPushButton#Primary { background: $accent_tint; border: 1px solid $accent_dark; }
QPushButton#Primary:hover { border-color: $accent; }

/* ---- Girdiler ---- */
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
    background: $elevated; border: 1px solid $border; border-radius: 8px; padding: 5px 8px; color: $text; min-height: 18px;
}
QLineEdit:hover, QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover { border-color: $border_strong; }
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus { border-color: $accent_dark; }
QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled { color: $text_muted; background: $surface; }
QComboBox { combobox-popup: 0; padding-right: 24px; }
QComboBox::drop-down { border: none; width: 24px; subcontrol-origin: padding; subcontrol-position: center right; }
QComboBox::down-arrow { image: url("$caret_down"); width: 12px; height: 12px; }
QComboBox QAbstractItemView { background: $surface; border: 1px solid $border_strong; border-radius: 8px; padding: 4px; outline: 0; }
QComboBox QAbstractItemView::item { padding: 6px 8px; border-radius: 6px; min-height: 20px; }
QComboBox QAbstractItemView::item:selected { background: $hover; color: $text; }
QSpinBox, QDoubleSpinBox { padding-right: 22px; }
QSpinBox::up-button, QDoubleSpinBox::up-button { subcontrol-origin: border; subcontrol-position: top right; width: 20px; border: none; background: transparent; }
QSpinBox::down-button, QDoubleSpinBox::down-button { subcontrol-origin: border; subcontrol-position: bottom right; width: 20px; border: none; background: transparent; }
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow { image: url("$caret_up"); width: 10px; height: 10px; }
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow { image: url("$caret_down"); width: 10px; height: 10px; }

QCheckBox { spacing: 8px; color: $text_secondary; }
QCheckBox::indicator, QTreeView::indicator, QTableView::indicator, QListView::indicator {
    width: 14px; height: 14px; border: 1px solid $border_strong; border-radius: 4px; background: $elevated;
}
QCheckBox::indicator:hover { border-color: $text_muted; }
QCheckBox::indicator:checked, QTreeView::indicator:checked, QTableView::indicator:checked, QListView::indicator:checked {
    background: $accent; border-color: $accent; image: url("$check");
}

/* ---- Kartlar, paneller ---- */
QFrame#Canvas {
    background: qradialgradient(cx: 0.25, cy: 0, radius: 0.9, fx: 0.25, fy: 0, stop: 0 #1c140c, stop: 1 $bg);
    border-radius: 12px;
}
QFrame#Card { background: $surface; border: 1px solid $border; border-radius: 12px; }
QFrame#DockTitle { background: $surface; border: 1px solid $border; border-bottom: none; border-top-left-radius: 12px; border-top-right-radius: 12px; }
QFrame#DockBody { background: $surface; border: 1px solid $border; border-top: none; border-bottom-left-radius: 12px; border-bottom-right-radius: 12px; }
QLabel#DockTitleLabel, QLabel#CardTitle { font-size: 13px; font-weight: 600; color: $text; }
QLabel#PageTitle { font-size: 22px; font-weight: 500; color: $text; }
QLabel#Muted, QLabel#FormLabel { color: $text_muted; }
QLabel#Readout { color: $text_secondary; font-size: 12px; }
QLabel#SectionLabel { color: $text_muted; font-size: 11px; font-weight: 600; }
QLabel#Chip { background: $elevated; border: 1px solid $border; border-radius: 8px; padding: 3px 9px; color: $text_secondary; font-size: 12px; }
QLabel#DemoChip { background: $accent_tint; border: 1px solid $accent_dark; border-radius: 6px; padding: 1px 7px; color: $accent; font-size: 11px; font-weight: 600; }
QLabel#NumChip { background: $elevated; border: 1px solid $border; border-radius: 11px; color: $text_secondary; font-size: 11px; }
QLabel#LinkText { color: $accent; }
QLabel#EmptyState { color: $text_muted; padding: 24px 12px; }
QLabel#Badge { background: $elevated; border-radius: 6px; padding: 3px 9px; border-left: 2px solid $border_strong; color: $text; }
QLabel#Badge[state="good"] { border-left-color: $good; }
QLabel#Badge[state="warning"] { border-left-color: $warning; }
QLabel#Badge[state="critical"] { border-left-color: $critical; }
QLabel#StatusChip { background: $surface; border: 1px solid $border; border-radius: 7px; padding: 2px 9px; color: $text_secondary; }

QFrame#Inset { background: $surface_alt; border: 1px solid $border; border-radius: 10px; }
QWidget#DetachedWindow { background: $bg; }
QToolButton#LegendToggle { background: transparent; border: 1px solid transparent; border-radius: 6px; padding: 1px 6px; color: $text_secondary; font-size: 12px; }
QToolButton#LegendToggle:hover { background: $hover; }
QToolButton#LegendToggle:checked { background: transparent; border-color: transparent; color: $text_secondary; }
QToolButton#LegendToggle:!checked { color: $text_muted; border: 1px dashed $border_strong; }
QToolButton#qt_toolbar_ext_button { background: $elevated; border: 1px solid $border_strong; border-radius: 6px; min-width: 18px; padding: 2px; }
QToolButton#Segment { background: $elevated; border: 1px solid $border; border-radius: 6px; padding: 2px 9px; color: $text_secondary; font-size: 12px; }
QToolButton#Segment:hover { border-color: $border_strong; color: $text; }
QToolButton#Segment:checked { background: $accent_tint; border-color: $accent_dark; color: $text; }
QPushButton#LiveButton { background: $accent_tint; border: 1px solid $accent_dark; border-radius: 7px; padding: 2px 10px; color: $accent; font-size: 12px; font-weight: 600; }
QPushButton#LiveButton:hover { border-color: $accent; }
QLabel#EmptyTitle { font-size: 16px; font-weight: 600; color: $text; }
QSlider::groove:horizontal { height: 4px; background: $border_strong; border-radius: 2px; }
QSlider::sub-page:horizontal { background: $accent; border-radius: 2px; }
QSlider::handle:horizontal { width: 12px; height: 12px; margin: -5px 0; border-radius: 7px; background: $text; border: 1px solid $accent; }
QSlider::handle:horizontal:hover { background: $accent; }

QToolButton#SectionHeader { background: transparent; border: none; border-radius: 6px; padding: 6px 4px; color: $text; font-weight: 600; text-align: left; }
QToolButton#SectionHeader:hover { background: $hover; }

/* ---- Ağaç, liste, tablo ---- */
QTreeView, QListView, QTableView { background: transparent; border: none; alternate-background-color: transparent; gridline-color: $border; }
QTreeView::item, QListView::item { padding: 5px 4px; border-radius: 6px; }
QTreeView::item:hover, QListView::item:hover { background: $hover; }
QTreeView::item:selected, QListView::item:selected { background: $accent_tint; color: $text; }
QTreeView::branch { background: transparent; }
QTreeView::branch:has-children:closed { image: url("$caret_right"); }
QTreeView::branch:has-children:open { image: url("$caret_down"); }
QTableView::item { padding: 0 8px; border-bottom: 1px solid $border; }
QTableView::item:selected { background: $accent_tint; color: $text; }
QHeaderView { background: transparent; }
QHeaderView::section { background: transparent; color: $text_muted; border: none; border-bottom: 1px solid $border; padding: 8px; }
QTableCornerButton::section { background: transparent; border: none; }
QListView#CompleterPopup { background: $surface; border: 1px solid $border_strong; border-radius: 8px; padding: 4px; }

QScrollArea { background: transparent; border: none; }
QScrollArea > QWidget > QWidget { background: transparent; }
QScrollBar:vertical { background: transparent; width: 10px; margin: 2px; }
QScrollBar:horizontal { background: transparent; height: 10px; margin: 2px; }
QScrollBar::handle { background: $border_strong; border-radius: 3px; }
QScrollBar::handle:vertical { min-height: 30px; }
QScrollBar::handle:horizontal { min-width: 30px; }
QScrollBar::handle:hover { background: $text_muted; }
QScrollBar::add-line, QScrollBar::sub-line { width: 0; height: 0; }
QScrollBar::add-page, QScrollBar::sub-page { background: none; }

QSplitter::handle { background: transparent; }
QSplitter::handle:vertical { height: 12px; }
QSplitter::handle:horizontal { width: 12px; }

QPlainTextEdit#Log { background: transparent; border: none; color: $text_secondary; }

QStatusBar { background: $bg; border-top: 1px solid $border; color: $text_muted; }
QStatusBar::item { border: none; }
QStatusBar QLabel { color: $text_muted; padding: 0 6px; }
QTabBar::tab { background: transparent; color: $text_muted; padding: 6px 12px; border: 1px solid transparent; border-radius: 8px; margin: 4px 2px; }
QTabBar::tab:selected { background: $elevated; color: $text; border-color: $border_strong; }
QTabBar::tab:hover:!selected { background: $hover; color: $text_secondary; }
"""


def _load_bundled_fonts() -> None:
    """resources/fonts altındaki .ttf/.otf dosyalarını (ör. Inter) uygulamaya yükler."""
    if not FONT_DIR.is_dir():
        return
    for path in FONT_DIR.iterdir():
        if path.suffix.lower() in (".ttf", ".otf"):
            QFontDatabase.addApplicationFont(str(path))


class ThemeManager(QObject):
    def __init__(self, app: QApplication):
        super().__init__(app)
        self.app = app
        self.tokens = TERRAFLUX_DARK
        self.series = SERIES_DARK
        self.raw_trace = RAW_TRACE_DARK

    def apply(self) -> None:
        t = self.tokens
        _load_bundled_fonts()
        qss = Template(QSS).substitute({**asdict(t), **_write_svg_assets(t)})
        # Aynı tema zaten uygulanmışsa stil / yazı tipi / palet yeniden kurulmaz: Qt aksi hâlde canlı tüm
        # pencereleri yeniden cilalar (ana pencere başına saniyeler)
        if self.app.property("spt_theme") != repr(t) or self.app.styleSheet() != qss:
            self.app.setStyle(QStyleFactory.create("Fusion"))
            self.app.setFont(ui_font(13))
            self.app.setPalette(_palette(t))
            self.app.setStyleSheet(qss)
            self.app.setProperty("spt_theme", repr(t))
        hints = self.app.styleHints()
        if hasattr(hints, "setColorScheme"):
            hints.setColorScheme(Qt.ColorScheme.Dark)
        pg.setConfigOptions(
            antialias=True,
            background=t.surface,
            foreground=t.text_muted,
            imageAxisOrder="row-major",
        )

    def series_color(self, index: int) -> str:
        """Kanal rengi, sabit sırada. 8'den sonraki kanallar aynı renkleri tekrar eder; çizimde kesikli çizgiyle
        ayrılır (bkz. LiveView)."""
        return self.series[max(index, 0) % len(self.series)]

    def raw_trace_for(self, colors) -> str:
        """Ham (filtresiz) sinyalin grisi: görünen kanal renklerine göre ayrışan en sakin gri."""
        used = {c.lower() for c in colors}
        return RAW_TRACE_SOFT if used and used <= _SOFT_RAW_OK else self.raw_trace


def style_native_titlebar(widget: QWidget, t: Tokens) -> None:
    """Windows 11'de pencere başlık çubuğunu tema renklerine boyar (diğer sistemlerde etkisiz)."""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        from ctypes import wintypes

        hwnd = wintypes.HWND(int(widget.winId()))
        dwm = ctypes.windll.dwmapi

        def colorref(hex_color: str) -> int:
            r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
            return r | (g << 8) | (b << 16)

        for attr, value in (
            (20, 1),                           # DWMWA_USE_IMMERSIVE_DARK_MODE
            (34, colorref(t.border)),          # DWMWA_BORDER_COLOR
            (35, colorref(t.bg)),              # DWMWA_CAPTION_COLOR
            (36, colorref(t.text_secondary)),  # DWMWA_TEXT_COLOR
        ):
            v = ctypes.c_int(value)
            dwm.DwmSetWindowAttribute(hwnd, attr, ctypes.byref(v), ctypes.sizeof(v))
    except (AttributeError, OSError):
        pass
