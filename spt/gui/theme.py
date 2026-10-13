"""Temalar: açık (varsayılan, Windows 11 Fluent / WPF Gallery görünümü), TerraFlux açık (sıcak kâğıt tonları) ve
TerraFlux koyu.

Tüm renkler bir `Tokens` nesnesinden gelir; QSS, QPalette, pyqtgraph ayarları ve Windows pencere çerçevesi bu
token'lardan üretilir. Etkin tema uygulama açılırken bir kez seçilir (`ThemeManager`, kayıtlı tercih ya da açık tema).
Widget'lar renklerini kurulurken aldığı için tema değişikliği uygulama yeniden başlatılınca geçerli olur. Fluent tema
ortak QSS'in üstüne kendi katmanını (`FLUENT_QSS`: 4 / 8 px köşeler, alt çizgili girdiler, mavi birincil düğme,
gösterge çubuklu seçim) ve kendi yazı tipini (Segoe UI Variable) ekler.

Grafik paletleri her tema için ayrı ölçülür (koyu: #121110, açık: #ffffff, TerraFlux açık: #fffdfa kart yüzeyinde).
Modüller tema başına farklı paletlerini `ThemedColors` / `ThemedMap` / `themed` ile tanımlar (açık / koyu); değerler
kullanıldıkları anda etkin temaya göre çözülür (modül içe aktarılırken değil). İki açık tema aynı açık paletleri
kullanır: #ffffff ile #fffdfa arasındaki kontrast farkı %1'in altındadır.
"""
from __future__ import annotations

import sys
import tempfile
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from string import Template

import pyqtgraph as pg
from PyQt6.QtCore import QObject, Qt
from PyQt6.QtGui import QColor, QFont, QFontDatabase, QPalette
from PyQt6.QtWidgets import QApplication, QStyleFactory, QWidget

from spt.gui.settings import app_settings

FONT_DIR = Path(__file__).parent / "resources" / "fonts"
UI_FONT_FAMILIES = ["Inter", "Segoe UI Variable Text", "Segoe UI", "Arial"]
# Fluent: Windows 11 sistem yazı tipi (değişken Segoe; Windows 10'da Segoe UI'ye düşer)
FLUENT_FONT_FAMILIES = ["Segoe UI Variable Text", "Segoe UI", "Inter", "Arial"]
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
    accent: str          # vurgu: çizgi, simge, dolgu (yüzeye ≥ 3:1)
    accent_dark: str     # seçili öğe kenarı, metin seçimi
    accent_tint: str     # seçili öğe zemini
    accent_text: str     # vurgu renginde metin (yüzeye ≥ 4.5:1)
    good: str            # durum renkleri (seri renkleriyle karıştırılmaz)
    warning: str
    critical: str
    grid: str
    axis: str            # grafik eksen çizgisi (yüzeye ≥ 2.4:1)
    glow: str            # orta alanın sol üstündeki sıcak ışıma (radyal gradyan)


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
    accent_text="#f59a4d",
    good="#0ca30c",
    warning="#fab219",
    critical="#d03b3b",
    grid="#1e1c1a",
    axis="#57504a",
    glow="#1c140c",
)

# Sıcak kâğıt tonları. Metin kontrastı (WCAG): text 15:1+, text_secondary 7.2:1+, text_muted 4.7:1+, accent_text
# 5.2:1+ (zemin, yüzey, girdi ve seçili öğe zemininin hepsinde); accent yüzeye 3.1:1 (yalnız çizgi / simge / dolgu).
TERRAFLUX_LIGHT = Tokens(
    bg="#f3f0eb",
    surface="#fffdfa",
    surface_alt="#f8f5f0",
    elevated="#ffffff",
    hover="#f2ede6",
    pressed="#e9e2d9",
    border="#e7e0d6",
    border_strong="#d4cabd",
    text="#1d1a16",
    text_secondary="#554d44",
    text_muted="#72685d",
    accent="#e2731b",
    accent_dark="#e0904f",
    accent_tint="#fcecdd",
    accent_text="#a14b06",
    good="#0a7f0a",
    warning="#986300",
    critical="#c43333",
    grid="#efe9e1",
    axis="#a89b8b",
    glow="#fbe5cf",
)

# Windows 11 Fluent (WPF Gallery) açık: Mica zemini #f3f3f3, beyaz kartlar, ince nötr kenarlar, sistem vurgusu #005fb8
# (SystemAccentColorDark1), vurgu metni #003e92 (Dark2). Seçim nötr dolgu + mavi gösterge çubuğuyla (FLUENT_QSS);
# accent_tint açık mavi yalnız rozetlerde ve işaretli simge düğmelerinde. Ölçülen WCAG kontrastı (zemin / kart / girdi /
# seçili zemin): text 15.7 / 17.4 / 16.8 / 15.3, text_secondary 5.9 / 6.6 / 6.4 / 5.8, text_muted 4.8 / 5.3 / 5.2,
# accent_text 9.0 / 10.0 / 9.6 / 8.7; accent karta 6.3:1, eksen grisi karta 3.45:1.
FLUENT_LIGHT = Tokens(
    bg="#f3f3f3",
    surface="#ffffff",
    surface_alt="#f9f9f9",
    elevated="#fbfbfb",
    hover="#ececec",
    pressed="#e3e3e3",
    border="#e5e5e5",
    border_strong="#d1d1d1",
    text="#1a1a1a",
    text_secondary="#5d5d5d",
    text_muted="#6b6b6b",
    accent="#005fb8",
    accent_dark="#7fb0e3",
    accent_tint="#e8f1fb",
    accent_text="#003e92",
    good="#0f7b0f",
    warning="#9d5d00",
    critical="#c42b1c",
    grid="#ededed",
    axis="#8a8a8a",
    glow="#f3f3f3",
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
RAW_TRACE_SOFT_DARK = "#8a837c"

# Açık tema kanal renkleri: koyu paletle aynı tonlar ve sıra, #fffdfa yüzeyi için yeniden adımlandı. Ölçüm (OKLab
# ΔE ×100, Machado 2009 şiddet 1.0): komşu çiftlerde normal görüş ≥ 19.0, protan ≥ 9.3, deutan ≥ 13.3; yüzeye
# kontrast ≥ 3.14:1 (açık paletteki camgöbeği / sarı / pembe 3:1'in altında kaldığı için koyulaştırıldı).
SERIES_LIGHT = ("#e56022", "#2877d3", "#1c8f6e", "#c68206", "#d25188", "#208209", "#6051d9", "#de4547")
# Ham sinyal: koyu sıcak gri; 8 kanal renginin hepsinden protan/deutan ΔE ≥ 9.3, normal ≥ 18.5; yüzeye 7.8:1.
RAW_TRACE_LIGHT = "#55504c"
# Yalnız ilk iki kanal görünürken daha açık gri: ↔ #e56022 / #2877d3 renk körlüğünde ΔE ≥ 10.6; yüzeye 3.9:1.
RAW_TRACE_SOFT_LIGHT = "#857f7a"
# Fluent (nötr griler, #ffffff): ham #4f4f4f 8 kanal renginden protan/deutan ΔE ≥ 10.0, karta 8.2:1; ilk iki kanal
# görünürken #858585 (↔ #e56022 / #2877d3 CVD ΔE ≥ 12.1, karta 3.7:1).
RAW_TRACE_FLUENT = "#4f4f4f"
RAW_TRACE_SOFT_FLUENT = "#858585"

# Spektrum max-hold / min-hold izleri (Canlı İzleme spektrumu tema vurgusunda çizilir). Ölçüm (OKLab ΔE ×100,
# Machado 2009 protan/deutan en kötüsü, kendi kart yüzeyinde): TerraFlux temalarında pembe / mavi kanal renkleri
# turuncu vurgudan ≥ 14.5, ham griden ≥ 11.8, birbirinden ≥ 11.9 ayrışır. Fluent'te vurgu mavi olduğundan mavi
# min-hold spektrumla karışır (CVD 7.6): turuncu #ca5010 / yeşil #1c8f6e — vurgudan 24.7 / 20.4, ham griden
# 13.1 / 15.3, birbirinden 11.0; karta 4.5 / 4.0:1.
HOLD_DARK = ("#d55181", "#3987e5")
HOLD_LIGHT = ("#d25188", "#2877d3")
HOLD_FLUENT = ("#ca5010", "#1c8f6e")


@dataclass(frozen=True)
class Theme:
    name: str                   # ayarlarda saklanan ad
    label: str                  # menüde görünen ad
    dark: bool
    tokens: Tokens
    series: tuple[str, ...]
    raw_trace: str
    raw_trace_soft: str         # yalnız ilk iki kanal rengi görünürken
    hold: tuple[str, str] = HOLD_LIGHT          # spektrum max-hold, min-hold
    style: str = "terraflux"                    # QSS katmanı: "terraflux" | "fluent"
    fonts: tuple[str, ...] = tuple(UI_FONT_FAMILIES)


# "light" adı kayıtlı tercihlerde yaşar: varsayılan açık tema artık Fluent; eski sıcak açık tema "terraflux"
LIGHT = Theme("light", "Açık", False, FLUENT_LIGHT, SERIES_LIGHT, RAW_TRACE_FLUENT, RAW_TRACE_SOFT_FLUENT,
              HOLD_FLUENT, "fluent", tuple(FLUENT_FONT_FAMILIES))
TERRAFLUX = Theme("terraflux", "TerraFlux açık", False, TERRAFLUX_LIGHT, SERIES_LIGHT, RAW_TRACE_LIGHT,
                  RAW_TRACE_SOFT_LIGHT, HOLD_LIGHT)
DARK = Theme("dark", "Koyu", True, TERRAFLUX_DARK, SERIES_DARK, RAW_TRACE_DARK, RAW_TRACE_SOFT_DARK, HOLD_DARK)
THEMES = {t.name: t for t in (LIGHT, TERRAFLUX, DARK)}
DEFAULT_THEME = LIGHT.name
SETTINGS_KEY = "appearance/theme"

_active: Theme = LIGHT


def active() -> Theme:
    return _active


def tokens() -> Tokens:
    """Etkin temanın token'ları (çağrıldığı anda)."""
    return _active.tokens


def is_dark() -> bool:
    return _active.dark


def themed(dark: str, light: str) -> str:
    """Etkin temaya göre iki renkten biri."""
    return dark if _active.dark else light


def saved_theme() -> str:
    """Kayıtlı tema tercihi; yoksa ya da tanınmıyorsa açık tema."""
    name = str(app_settings().value(SETTINGS_KEY, DEFAULT_THEME))
    return name if name in THEMES else DEFAULT_THEME


def save_theme(name: str) -> None:
    if name not in THEMES:
        raise ValueError(f"Bilinmeyen tema: {name}")
    app_settings().setValue(SETTINGS_KEY, name)


class ThemedColors(Sequence):
    """Tema başına ayrı ölçülmüş renk dizisi; tuple gibi okunur (indeks, dilim, `in`, döngü, `==`) ve her
    erişimde etkin temanın değerini verir. `dark` / `light` öznitelikleri iki paleti doğrudan verir (testler)."""

    def __init__(self, dark: Iterable[str], light: Iterable[str]):
        self.dark, self.light = tuple(dark), tuple(light)
        if len(self.dark) != len(self.light):
            raise ValueError("Açık ve koyu paletler aynı uzunlukta olmalı")

    def _current(self) -> tuple[str, ...]:
        return self.dark if _active.dark else self.light

    def __getitem__(self, index):
        return self._current()[index]

    def __len__(self) -> int:
        return len(self.dark)

    def __iter__(self) -> Iterator[str]:
        return iter(self._current())

    def __contains__(self, color) -> bool:
        return color in self._current()

    def __eq__(self, other) -> bool:
        if isinstance(other, (tuple, list, ThemedColors)):
            return self._current() == tuple(other)
        return NotImplemented

    __hash__ = None

    def __repr__(self) -> str:
        return f"ThemedColors({self._current()!r})"


class ThemedMap(Mapping):
    """Tema başına renk eşlemesi. `ThemedMap(koyu, açık)` iki sözlükle, `ThemedMap(üretici)` her erişimde etkin
    temanın renkleriyle kurulan sözlükle (ör. `ThemedColors`'tan türetilen rol → renk eşlemeleri) çalışır."""

    def __init__(self, dark: Mapping[str, str] | Callable[[], Mapping[str, str]],
                 light: Mapping[str, str] | None = None):
        self._factory = dark if callable(dark) and light is None else None
        self.dark = None if self._factory else dict(dark)
        self.light = None if self._factory else dict(light or {})

    def _current(self) -> Mapping[str, str]:
        if self._factory is not None:
            return self._factory()
        return self.dark if _active.dark else self.light

    def __getitem__(self, key):
        return self._current()[key]

    def __iter__(self):
        return iter(self._current())

    def __len__(self) -> int:
        return len(self._current())


def ui_font(px: int = 13, weight: QFont.Weight = QFont.Weight.Normal) -> QFont:
    """Etkin temanın arayüz yazı tipi (Fluent: Segoe UI Variable, TerraFlux: Inter)."""
    font = QFont()
    font.setFamilies(list(_active.fonts))
    font.setPixelSize(px)
    font.setWeight(weight)
    return font


def mono_font(px: int = 12) -> QFont:
    font = QFont()
    font.setFamilies(MONO_FONT_FAMILIES)
    font.setPixelSize(px)
    font.setStyleHint(QFont.StyleHint.Monospace)
    return font


def _is_dark_tokens(t: Tokens) -> bool:
    return QColor(t.bg).lightnessF() < 0.5


_SVG_SHAPES = {
    "caret_down": '<path d="M4 6l4 4 4-4"/>',
    "caret_up": '<path d="M4 10l4-4 4 4"/>',
    "caret_right": '<path d="M6 4l4 4-4 4"/>',
    "check": '<path d="M3.5 8.5l3 3 6-7"/>',
}


def _write_svg_assets(t: Tokens) -> dict[str, str]:
    """QSS'in url() ile kullandığı küçük ok/onay ikonlarını tema renginde üretir (dosya adında tema rengi: aynı anda
    açık iki farklı temalı uygulama birbirinin ikonunu ezmez)."""
    out_dir = Path(tempfile.gettempdir()) / "spt_theme"
    out_dir.mkdir(exist_ok=True)
    paths = {}
    for name, shape in _SVG_SHAPES.items():
        color = (t.bg if _is_dark_tokens(t) else t.elevated) if name == "check" else t.text_muted
        svg = (
            '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 16 16" '
            f'fill="none" stroke="{color}" stroke-width="1.8" stroke-linecap="round" '
            f'stroke-linejoin="round">{shape}</svg>'
        )
        path = out_dir / f"{name}_{color.lstrip('#')}.svg"
        path.write_text(svg, encoding="utf-8")
        paths[name] = path.as_posix()
    # Fluent seçim göstergeleri: yatay (üst gezinme, sekme) ve dikey (liste, yan şerit) yuvarlatılmış vurgu çubuğu
    for name, (w, h) in (("indicator_h", (16, 3)), ("indicator_v", (3, 16))):
        svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">'
               f'<rect width="{w}" height="{h}" rx="1.5" fill="{t.accent}"/></svg>')
        path = out_dir / f"{name}_{t.accent.lstrip('#')}.svg"
        path.write_text(svg, encoding="utf-8")
        paths[name] = path.as_posix()
    return paths


def _accent_states(t: Tokens) -> dict[str, str]:
    """Dolgulu vurgu düğmesinin üzerine gelme / basılı tonları ve üstündeki metin rengi. Üzerine gelme Fluent'teki gibi
    vurgu %90 opaklık (AccentFillColorSecondary); basılı ton açılmaz, koyulaşır: %80 opaklık beyaz metinle 4.2:1'e
    düşüyordu (#005fb8 → hover #196fbf 5.2:1, basılı #0053a0 7.7:1)."""
    base, surface = QColor(t.accent), QColor(t.surface)

    def mix(alpha: float) -> str:
        return QColor(*(round(getattr(base, c)() * alpha + getattr(surface, c)() * (1 - alpha))
                        for c in ("red", "green", "blue"))).name()

    return {"accent_hover": mix(0.9), "accent_pressed": base.darker(115).name(),
            "on_accent": "#ffffff" if QColor(t.accent).lightnessF() < 0.6 else t.text}


def primary_text() -> str:
    """Birincil (Primary) düğmenin metin rengi: Fluent'te dolgulu vurgu üstünde beyaz, diğer temalarda normal metin.
    Düğme simgeleri de bu renkte çizilmeli (ikincil gri simge mavi dolgu üstünde görünmüyordu)."""
    t = _active.tokens
    return _accent_states(t)["on_accent"] if _active.style == "fluent" else t.text


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
        QPalette.ColorRole.Link: t.accent_text,
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
/* Görünüm düğmeleri ve akış denetimi birer segment kabında */
QFrame#NavGroup, QFrame#ToolGroup { background: $surface; border: 1px solid $border; border-radius: 10px; }

QToolButton { background: transparent; border: 1px solid transparent; border-radius: 8px; padding: 6px; color: $text_secondary; }
QToolButton:hover { background: $hover; color: $text; }
QToolButton:pressed { background: $pressed; }
QToolButton:checked { background: $accent_tint; border-color: $accent_dark; color: $text; }
QToolButton#IconButton { background: $elevated; border: 1px solid $border; padding: 7px; }
QToolButton#IconButton:hover { background: $hover; border-color: $border_strong; }
QToolButton#IconButton:checked { background: $accent_tint; border-color: $accent_dark; }
QToolButton#GroupIcon { padding: 6px; border-radius: 7px; }
QToolButton#GroupIcon:checked { background: $accent_tint; border-color: $accent_dark; }
QToolButton#FlatIcon { padding: 3px; border-radius: 6px; }
QToolButton#NavPill { padding: 5px 12px; border-radius: 7px; color: $text_secondary; }
QToolButton#NavPill:checked { background: $accent_tint; border: 1px solid $accent_dark; color: $text; }
QToolButton#MenuButton { background: $elevated; border: 1px solid $border; padding: 6px 26px 6px 10px; color: $text; }
QToolButton#MenuButton:hover { background: $hover; border-color: $border_strong; }
QToolButton#MenuButton::menu-indicator { image: url("$caret_down"); subcontrol-position: right center; subcontrol-origin: padding; right: 8px; width: 12px; height: 12px; }
QToolButton#ChipMenu { background: $elevated; border: 1px solid $border; border-radius: 8px; padding: 3px 24px 3px 9px; color: $text_secondary; font-size: 12px; }
QToolButton#ChipMenu:hover { border-color: $border_strong; color: $text; }
QToolButton#ChipMenu::menu-indicator { image: url("$caret_down"); subcontrol-position: right center; subcontrol-origin: padding; right: 7px; width: 11px; height: 11px; }

QFrame#SearchBox { background: $elevated; border: 1px solid $border; border-radius: 8px; }
QFrame#SearchBox:hover { border-color: $border_strong; }
QFrame#SearchBox QLineEdit { background: transparent; border: none; padding: 6px 2px; }
QLabel#Kbd { background: $hover; border: 1px solid $border_strong; border-radius: 5px; padding: 1px 6px; font-size: 11px; color: $text_secondary; }

/* ---- Butonlar ---- */
QPushButton { background: $elevated; border: 1px solid $border_strong; border-radius: 8px; padding: 7px 14px; color: $text; }
QPushButton:hover { background: $hover; }
QPushButton:pressed { background: $pressed; }
QPushButton:disabled { color: $text_muted; border-color: $border; }
QPushButton#Primary { background: $accent_tint; border: 1px solid $accent_dark; font-weight: 600; }
QPushButton#Primary:hover { border-color: $accent; }

/* ---- Girdiler ---- */
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
    background: $elevated; border: 1px solid $border; border-radius: 8px; padding: 5px 8px; color: $text; min-height: 18px;
}
QLineEdit:hover, QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover { border-color: $border_strong; }
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus { border-color: $accent; }
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
    background: qradialgradient(cx: 0.25, cy: 0, radius: 0.9, fx: 0.25, fy: 0, stop: 0 $glow, stop: 1 $bg);
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
QLabel#DemoChip { background: $accent_tint; border: 1px solid $accent_dark; border-radius: 6px; padding: 1px 7px; color: $accent_text; font-size: 11px; font-weight: 600; }
QLabel#LiveBadge { background: $accent_tint; border: 1px solid $accent_dark; border-radius: 8px; padding: 3px 10px; color: $accent_text; font-size: 12px; font-weight: 600; }
QLabel#LiveBadge[state="paused"], QLabel#LiveBadge[state="zoom"] { background: $elevated; border-color: $border_strong; color: $text_secondary; }
QLabel#NumChip { background: $elevated; border: 1px solid $border; border-radius: 11px; color: $text_secondary; font-size: 11px; }
QLabel#LinkText { color: $accent_text; }
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
/* Segment kabı: düğmeler tek bir rayın içinde, yalnız seçili olan çerçeveli */
QFrame#SegmentGroup { background: $elevated; border: 1px solid $border; border-radius: 8px; }
QFrame#SegmentGroup QToolButton#Segment { background: transparent; border-color: transparent; }
QFrame#SegmentGroup QToolButton#Segment:hover { background: $hover; }
QFrame#SegmentGroup QToolButton#Segment:checked { background: $accent_tint; border-color: $accent_dark; color: $text; }
QPushButton#LiveButton { background: $accent_tint; border: 1px solid $accent_dark; border-radius: 7px; padding: 2px 10px; color: $accent_text; font-size: 12px; font-weight: 600; }
QPushButton#LiveButton:hover { border-color: $accent; }
QLabel#EmptyTitle { font-size: 16px; font-weight: 600; color: $text; }
QSlider::groove:horizontal { height: 4px; background: $border_strong; border-radius: 2px; }
QSlider::sub-page:horizontal { background: $accent; border-radius: 2px; }
QSlider::handle:horizontal { width: 10px; height: 10px; margin: -5px 0; border-radius: 7px; background: $elevated; border: 2px solid $accent; }
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

/* ---- Canlı İzleme KPI kutuları ---- */
QFrame#KpiTile { background: $surface; border: 1px solid $border; border-radius: 12px; }
QLabel#KpiTitle { color: $text_secondary; font-size: 12px; }
QLabel#KpiValue { color: $text; font-size: 22px; font-weight: 600; }
QLabel#KpiUnit { color: $text_muted; font-size: 12px; }
QLabel#KpiSub { color: $text_muted; font-size: 11px; }

/* ---- Panelleri kenara gizleme şeritleri ---- */
QToolBar[stripe="true"] { background: $bg; border: none; padding: 6px 2px; spacing: 4px; }
QToolButton#StripeButton { background: transparent; border: 1px solid transparent; border-radius: 8px; padding: 6px 2px; color: $text_secondary; }
QToolButton#StripeButton:hover { background: $hover; color: $text; }
QToolButton#StripeButton:checked { background: $accent_tint; border-color: $accent_dark; color: $text; }
"""

# Windows 11 Fluent (WPF Gallery) katmanı: ortak QSS'ten sonra eklenir, aynı seçicileri ezer. Kartlar 8 px, kontroller
# 4 px köşe; seçim nötr dolgu + mavi gösterge çubuğu; girdilerde alt kenar koyu, odakta 2 px vurgu; birincil düğme
# dolgulu mavi. Gölge yok (canlı grafiklerde QGraphicsDropShadowEffect pahalı): ince nötr kenar yeter.
FLUENT_QSS = """
QToolTip { background: $surface_alt; border: 1px solid $border_strong; border-radius: 4px; padding: 5px 8px; }
QMainWindow::separator:hover { background: $border_strong; }

QMenuBar { background: $bg; }
QMenuBar::item { border-radius: 4px; padding: 5px 10px; }
QMenu { background: $surface_alt; border: 1px solid $border_strong; border-radius: 8px; padding: 4px; }
QMenu::item { padding: 6px 28px 6px 12px; border-radius: 4px; margin: 1px 0; }
QMenu::item:selected { background: $hover; }
QMenu::separator { background: $border; margin: 4px 0; }
QMenu::indicator { border: 1px solid $axis; border-radius: 3px; background: $surface; }
QMenu::indicator:checked { background: $accent; border-color: $accent; }

/* Üst çubuk Mica zemininde, ayraçsız; görünümler WinUI üst gezinmesi gibi altta kısa gösterge çubuğuyla */
QToolBar#HeaderBar { border-bottom: none; }
QFrame#NavGroup, QFrame#ToolGroup { background: transparent; border: none; }
QToolButton { border-radius: 4px; }
QToolButton:checked { background: $accent_tint; border-color: transparent; }
QToolButton#NavPill { padding: 6px 12px 8px 12px; border: none; border-radius: 4px; color: $text_secondary; }
QToolButton#NavPill:hover { background: $hover; color: $text; }
QToolButton#NavPill:checked {
    background-color: transparent; color: $text; border: none;
    background-image: url("$indicator_h"); background-repeat: no-repeat; background-position: bottom center;
}
QToolButton#NavPill:checked:hover { background-color: $hover; }
QToolButton#IconButton { background: $elevated; border: 1px solid $border; border-bottom-color: $border_strong; border-radius: 4px; }
QToolButton#IconButton:hover { background: $surface_alt; border-color: $border_strong; }
QToolButton#IconButton:checked { background: $accent_tint; border-color: $accent_dark; }
QToolButton#GroupIcon { border-radius: 4px; }
QToolButton#GroupIcon:checked { background: $accent_tint; border-color: transparent; }
QToolButton#FlatIcon { border-radius: 4px; }
QToolButton#MenuButton { border-radius: 4px; border-bottom-color: $border_strong; }
QToolButton#ChipMenu { border-radius: 4px; border-bottom-color: $border_strong; }
QFrame#SearchBox { border-radius: 4px; border-bottom-color: $axis; }
QLabel#Kbd { border-radius: 3px; }
QLabel#Wordmark { font-weight: 600; }

QPushButton { background: $elevated; border: 1px solid $border; border-bottom-color: $border_strong; border-radius: 4px; padding: 6px 14px; }
QPushButton:hover { background: $surface_alt; }
QPushButton:pressed { background: $hover; color: $text_secondary; border-bottom-color: $border; }
QPushButton#Primary { background: $accent; border: 1px solid $accent; color: $on_accent; font-weight: 600; }
QPushButton#Primary:hover { background: $accent_hover; border-color: $accent_hover; }
QPushButton#Primary:pressed { background: $accent_pressed; border-color: $accent_pressed; }
QPushButton#Primary:disabled { background: $hover; border-color: $border; color: $text_muted; }

QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox { border-radius: 4px; border: 1px solid $border; border-bottom-color: $axis; }
QLineEdit:hover, QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover { background: $surface_alt; border-color: $border_strong; border-bottom-color: $axis; }
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus { background: $surface; border-color: $border; border-bottom: 2px solid $accent; padding-bottom: 4px; }
QComboBox:focus { border-color: $border_strong; border-bottom-color: $axis; }
QComboBox QAbstractItemView { background: $surface_alt; border-radius: 8px; }
QComboBox QAbstractItemView::item { border-radius: 4px; }
QComboBox QAbstractItemView::item:selected { background: $hover; }

QCheckBox::indicator, QTreeView::indicator, QTableView::indicator, QListView::indicator {
    width: 16px; height: 16px; border: 1px solid $axis; border-radius: 4px; background: $surface_alt;
}
QCheckBox::indicator:hover { border-color: $text_secondary; background: $hover; }
QCheckBox::indicator:checked, QTreeView::indicator:checked, QTableView::indicator:checked, QListView::indicator:checked {
    background: $accent; border-color: $accent;
}

/* Kartlar ve paneller: Mica zemin üstünde beyaz katmanlar; içerik alanı hafif açık katman (WPF Gallery) */
QFrame#Canvas { background: $surface_alt; border: 1px solid $border; border-radius: 8px; }
QFrame#Card, QFrame#KpiTile { border-radius: 8px; }
QFrame#DockTitle { border-top-left-radius: 8px; border-top-right-radius: 8px; }
QFrame#DockBody { border-bottom-left-radius: 8px; border-bottom-right-radius: 8px; }
QFrame#Inset { border-radius: 6px; }
QLabel#PageTitle { font-family: "Segoe UI Variable Display"; font-size: 24px; font-weight: 600; }
QLabel#DockTitleLabel, QLabel#CardTitle { font-weight: 600; }
QLabel#Chip, QLabel#StatusChip { border-radius: 4px; }
QLabel#DemoChip, QLabel#LiveBadge { border-radius: 4px; border-color: transparent; }
QLabel#NumChip { border-radius: 10px; }
QLabel#Badge { border-radius: 4px; }
QToolButton#LegendToggle { border-radius: 4px; }
QToolButton#Segment { border-radius: 4px; }
QToolButton#Segment:checked { background: $surface; border: 1px solid $border; border-bottom-color: $border_strong; color: $text; }
QFrame#SegmentGroup { background: $surface_alt; border-radius: 4px; }
QFrame#SegmentGroup QToolButton#Segment:checked { background: $surface; border: 1px solid $border; border-bottom-color: $border_strong; }
QPushButton#LiveButton { border-radius: 4px; border-color: transparent; }
QSlider::groove:horizontal { height: 4px; background: $axis; border-radius: 2px; }
QSlider::handle:horizontal {
    width: 18px; height: 18px; margin: -7px 0; border-radius: 9px; border: 1px solid $border_strong;
    background: qradialgradient(cx: 0.5, cy: 0.5, radius: 0.5, fx: 0.5, fy: 0.5, stop: 0 $accent, stop: 0.5 $accent, stop: 0.56 $surface, stop: 1 $surface);
}
QSlider::handle:horizontal:hover {
    background: qradialgradient(cx: 0.5, cy: 0.5, radius: 0.5, fx: 0.5, fy: 0.5, stop: 0 $accent, stop: 0.62 $accent, stop: 0.68 $surface, stop: 1 $surface);
}
QToolButton#SectionHeader { border-radius: 4px; }

/* Liste / ağaç / tablo: nötr seçim dolgusu + solda gösterge çubuğu */
QTreeView::item, QListView::item { border-radius: 4px; }
QTreeView::item:selected, QListView::item:selected, QTableView::item:selected {
    background-color: $hover; color: $text;
}
QListView::item:selected {
    background-image: url("$indicator_v"); background-repeat: no-repeat; background-position: left center;
}
QHeaderView::section { color: $text_secondary; }
QListView#CompleterPopup { background: $surface_alt; border-radius: 8px; }
QScrollBar::handle { background: $axis; border-radius: 3px; }
QScrollBar::handle:hover { background: $text_muted; }
QScrollBar:vertical { width: 8px; }
QScrollBar:horizontal { height: 8px; }

QTabBar::tab { border-radius: 4px; padding: 6px 12px 8px 12px; }
QTabBar::tab:selected {
    background-color: transparent; color: $text; border-color: transparent;
    background-image: url("$indicator_h"); background-repeat: no-repeat; background-position: bottom center;
}

QToolBar[stripe="true"] { background: $bg; }
QToolButton#StripeButton { border-radius: 4px; }
QToolButton#StripeButton:checked {
    background-color: $hover; border-color: transparent; color: $text;
    background-image: url("$indicator_v"); background-repeat: no-repeat; background-position: left center;
}
QToolBar[stripe="true"][side="right"] QToolButton#StripeButton:checked { background-position: right center; }
"""


def _load_bundled_fonts() -> None:
    """resources/fonts altındaki .ttf/.otf dosyalarını (ör. Inter) uygulamaya yükler."""
    if not FONT_DIR.is_dir():
        return
    for path in FONT_DIR.iterdir():
        if path.suffix.lower() in (".ttf", ".otf"):
            QFontDatabase.addApplicationFont(str(path))


class ThemeManager(QObject):
    """Etkin temayı seçer ve uygular. `name` verilmezse kayıtlı tercih (yoksa açık tema) kullanılır.

    Etkin tema kurulurken modül genelinde de ayarlanır: `tokens()`, `ThemedColors` gibi çağrı anında çözülen
    değerler bu temayı verir. Pencereler bu nesne kurulduktan sonra oluşturulmalıdır."""

    def __init__(self, app: QApplication, name: str | None = None):
        super().__init__(app)
        self.app = app
        self.theme = THEMES.get(name or saved_theme(), LIGHT)
        self._activate()
        self.tokens = self.theme.tokens
        self.series = self.theme.series
        self.raw_trace = self.theme.raw_trace

    @property
    def name(self) -> str:
        return self.theme.name

    @property
    def dark(self) -> bool:
        return self.theme.dark

    def _activate(self) -> None:
        global _active
        _active = self.theme

    def apply(self) -> None:
        self._activate()
        t = self.tokens
        _load_bundled_fonts()
        layers = QSS + (FLUENT_QSS if self.theme.style == "fluent" else "")
        qss = Template(layers).substitute({**asdict(t), **_write_svg_assets(t), **_accent_states(t)})
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
            hints.setColorScheme(Qt.ColorScheme.Dark if self.dark else Qt.ColorScheme.Light)
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

    def hold_color(self, mode: str) -> str:
        """Spektrum max-hold ("max") / min-hold ("min") izinin tema rengi (bkz. HOLD_FLUENT)."""
        return self.theme.hold[0 if mode == "max" else 1]

    def raw_trace_for(self, colors) -> str:
        """Ham (filtresiz) sinyalin grisi: görünen kanal renklerine göre ayrışan en sakin gri."""
        used = {c.lower() for c in colors}
        soft_ok = {c.lower() for c in self.series[:2]}
        return self.theme.raw_trace_soft if used and used <= soft_ok else self.raw_trace


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
            (20, int(_is_dark_tokens(t))),     # DWMWA_USE_IMMERSIVE_DARK_MODE
            (34, colorref(t.border)),          # DWMWA_BORDER_COLOR
            (35, colorref(t.bg)),              # DWMWA_CAPTION_COLOR
            (36, colorref(t.text_secondary)),  # DWMWA_TEXT_COLOR
        ):
            v = ctypes.c_int(value)
            dwm.DwmSetWindowAttribute(hwnd, attr, ctypes.byref(v), ctypes.sizeof(v))
    except (AttributeError, OSError):
        pass
