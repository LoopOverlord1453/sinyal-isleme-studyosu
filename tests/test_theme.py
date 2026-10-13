"""Tema: varsayılan açık (Fluent) tema, kayıtlı tercih, tema başına çözülen paletler ve her temanın ölçülmüş renkleri.

Ölçüm modül belgelerindekiyle aynı: OKLab ΔE (×100), protan / deutan benzetimi Machado 2009 (şiddet 1.0), WCAG
kontrastı. Her palet kendi temasının kart yüzeyinde ölçülür (koyu #121110, açık / Fluent #ffffff, TerraFlux açık
#fffdfa). Açık paletler (ThemedColors.light) iki açık temada da ölçülür.
"""
import itertools

import numpy as np
import pytest
from PyQt6.QtWidgets import QApplication

PROT = np.array([[0.152286, 1.052583, -0.204868], [0.114503, 0.786281, 0.099216], [-0.003882, -0.048116, 1.051998]])
DEUT = np.array([[0.367322, 0.860646, -0.227968], [0.280085, 0.672501, 0.047413], [-0.011820, 0.042940, 0.968881]])
M1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929], [0.2119034982, 0.6806995451, 0.1073969566],
               [0.0883024619, 0.2817188376, 0.6299787005]])
M2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468], [1.9779984951, -2.4285922050, 0.4505937099],
               [0.0259040371, 0.7827717662, -0.8086757660]])
SURFACE = {"dark": "#121110", "light": "#ffffff", "terraflux": "#fffdfa"}
THEME_NAMES = ["dark", "light", "terraflux"]


def _lin(h):
    c = np.array([int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)])
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _lab(h, m=None):
    c = _lin(h) if m is None else np.clip(m @ _lin(h), 0, 1)
    return 100 * (M2 @ np.cbrt(M1 @ c))


def de(a, b, m=None) -> float:
    return float(np.linalg.norm(_lab(a, m) - _lab(b, m)))


def cvd(a, b) -> float:
    """Protan ve deutan benzetiminde en küçük ΔE."""
    return min(de(a, b, PROT), de(a, b, DEUT))


def contrast(a, b) -> float:
    la, lb = (float(_lin(h) @ [0.2126, 0.7152, 0.0722]) for h in (a, b))
    return (max(la, lb) + 0.05) / (min(la, lb) + 0.05)


def worst(colors, pairs):
    return min(de(colors[i], colors[j]) for i, j in pairs), min(cvd(colors[i], colors[j]) for i, j in pairs)


def adjacent(n):
    return [(i, i + 1) for i in range(n - 1)]


def every(n):
    return list(itertools.combinations(range(n), 2))


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def settings(tmp_path, monkeypatch):
    monkeypatch.setenv("SPT_SETTINGS_FILE", str(tmp_path / "ayar.ini"))


@pytest.fixture
def restore_theme():
    """Testin değiştirdiği etkin temayı geri alır (aynı süreçteki diğer testler etkilenmesin)."""
    from spt.gui import theme
    before = theme._active
    yield theme
    theme._active = before


# ============================================================================ tercih ve etkin tema
def test_default_theme_is_light(settings):
    from spt.gui.theme import saved_theme
    assert saved_theme() == "light"


def test_saved_theme_round_trip_and_unknown_value(settings):
    from spt.gui.settings import app_settings
    from spt.gui.theme import SETTINGS_KEY, save_theme, saved_theme

    save_theme("dark")
    assert saved_theme() == "dark"
    app_settings().setValue(SETTINGS_KEY, "mor")
    assert saved_theme() == "light"          # tanınmayan değer: varsayılan
    with pytest.raises(ValueError):
        save_theme("mor")


def test_theme_manager_reads_preference_and_activates(app, settings, restore_theme):
    from spt.gui.control import control_plots as cp
    from spt.gui.theme import FLUENT_LIGHT, TERRAFLUX_DARK, TERRAFLUX_LIGHT, ThemeManager, save_theme, tokens
    from spt.gui.views import fusion_view as fv

    save_theme("dark")
    tm = ThemeManager(app)
    assert tm.name == "dark" and tm.dark and tokens() is TERRAFLUX_DARK
    assert cp.PALETTE[0] == "#f54748" and fv.ALGO_COLORS["madgwick"] == "#488bfb" and cp.guide() == "#8a837c"

    tm = ThemeManager(app, "light")          # açıkça verilen ad tercihi geçersiz kılar
    assert tm.name == "light" and not tm.dark and tokens() is FLUENT_LIGHT
    assert cp.PALETTE[0] == "#b31319" and fv.ALGO_COLORS["madgwick"] == "#1a4eda" and cp.guide() == "#7c7670"
    assert list(cp.PALETTE) == list(cp.PALETTE.light) and cp.PALETTE == cp.PALETTE.light
    assert tm.series_color(9) == tm.series[1]
    assert (tm.hold_color("max"), tm.hold_color("min")) == ("#ca5010", "#1c8f6e")

    tm = ThemeManager(app, "terraflux")      # eski sıcak açık tema ayrı adla duruyor, açık paletleri kullanır
    assert not tm.dark and tokens() is TERRAFLUX_LIGHT and cp.PALETTE == cp.PALETTE.light
    assert (tm.hold_color("max"), tm.hold_color("min")) == (tm.series[4], tm.series[1])


def test_fluent_layer_font_and_indicators(app, settings, restore_theme):
    """Fluent tema ortak QSS'in üstüne kendi katmanını ve Segoe UI Variable yazı tipini kullanır; diğer temalar
    kullanmaz. Gösterge çubukları vurgu renginde üretilir; birincil düğme metni dolgulu vurgunun üstünde beyaz."""
    from string import Template

    from spt.gui import theme

    fluent, terra = theme.THEMES["light"], theme.THEMES["terraflux"]
    assert fluent.style == "fluent" and terra.style == "terraflux" and theme.DARK.style == "terraflux"
    assert fluent.fonts[0] == "Segoe UI Variable Text" and terra.fonts[0] == "Inter"
    theme._active = fluent
    assert theme.ui_font(13).families()[0] == "Segoe UI Variable Text"
    theme._active = terra
    assert theme.ui_font(13).families()[0] == "Inter"
    assets = theme._write_svg_assets(fluent.tokens)
    assert fluent.tokens.accent in open(assets["indicator_h"], encoding="utf-8").read()
    states = theme._accent_states(fluent.tokens)
    assert states["on_accent"] == "#ffffff" and contrast(states["on_accent"], fluent.tokens.accent) >= 4.5
    for key in ("accent_hover", "accent_pressed"):    # üzerine gelince / basılı ton beyaz metinle okunur kalır
        assert contrast("#ffffff", states[key]) >= 4.5
    values = {**theme.asdict(fluent.tokens), **assets, **states}
    Template(theme.QSS + theme.FLUENT_QSS).substitute(values)   # bütün değişkenler tanımlı
    tm = theme.ThemeManager(app, "light")
    tm.apply()
    assert "indicator_h" in app.styleSheet() and "Segoe UI Variable Display" in app.styleSheet()
    tm = theme.ThemeManager(app, "terraflux")
    tm.apply()
    assert "indicator_h" not in app.styleSheet()
    theme.ThemeManager(app, "light").apply()

def test_themed_containers_resolve_at_access(restore_theme):
    theme = restore_theme
    seq = theme.ThemedColors(dark=("#111111", "#222222"), light=("#aaaaaa", "#bbbbbb"))
    derived = theme.ThemedMap(lambda: {"ilk": seq[0]})
    fixed = theme.ThemedMap({"x": "#000000"}, {"x": "#ffffff"})
    theme._active = theme.DARK
    assert seq[0] == "#111111" and seq[-1:] == ("#222222",) and "#222222" in seq and derived["ilk"] == "#111111"
    assert fixed["x"] == "#000000" and theme.themed("k", "a") == "k"
    theme._active = theme.LIGHT
    assert tuple(seq) == ("#aaaaaa", "#bbbbbb") and len(seq) == 2 and derived["ilk"] == "#aaaaaa"
    assert fixed.get("x") == "#ffffff" and theme.themed("k", "a") == "a"
    with pytest.raises(ValueError):
        theme.ThemedColors(dark=("#111111",), light=())


# ============================================================================ ölçülmüş renkler
@pytest.mark.parametrize("name", THEME_NAMES)
def test_text_tokens_are_readable(name):
    from spt.gui.theme import THEMES
    t = THEMES[name].tokens
    for fg in (t.text, t.text_secondary, t.accent_text):
        for bg in (t.bg, t.surface, t.elevated, t.accent_tint):
            assert contrast(fg, bg) >= 4.5, (fg, bg)
    for bg in (t.bg, t.surface, t.elevated):  # soluk metin seçili öğe zemininde kullanılmaz
        assert contrast(t.text_muted, bg) >= 4.5, bg
    assert contrast(t.accent, t.surface) >= 3.0          # çizgi / simge / dolgu


@pytest.mark.parametrize("name", THEME_NAMES)
def test_channel_series_and_raw_trace(name):
    """Canlı izleme kanal renkleri: komşu çiftler ayrışır; ham gri her kanaldan ayrışır (bkz. theme.py)."""
    from spt.gui.theme import THEMES
    th = THEMES[name]
    s, surface = list(th.series), SURFACE[name]
    normal, colorblind = worst(s, adjacent(len(s)))
    assert normal >= 19.0 - 0.05 and colorblind >= 8.4 - 0.05
    assert min(contrast(c, surface) for c in s) >= 3.0
    assert min(cvd(th.raw_trace, c) for c in s) >= 9.1 - 0.05 and contrast(th.raw_trace, surface) >= 3.0
    assert min(cvd(th.raw_trace_soft, c) for c in s[:2]) >= 8.0 and contrast(th.raw_trace_soft, surface) >= 3.0


# Tema başına ölçülen en kötü değerler (renk körlüğü ΔE): vurguya, ham griye, iki iz arası (bkz. theme.HOLD_*)
HOLD_FLOORS = {"dark": (14.9, 11.8, 15.9), "light": (20.4, 13.1, 11.0), "terraflux": (14.5, 12.5, 11.9)}


@pytest.mark.parametrize("name", THEME_NAMES)
def test_spectrum_hold_traces_stand_apart(name):
    """Spektrum: max-hold / min-hold izleri tek renk spektrumdan (tema vurgusu), filtre açıkken ham griden ve
    birbirinden renk körlüğünde de ayrışır (bkz. theme.HOLD_FLUENT; Fluent'te vurgu mavi olduğundan mavi iz yok)."""
    from spt.gui.theme import THEMES
    th = THEMES[name]
    hmax, hmin = th.hold
    to_accent, to_raw, between = HOLD_FLOORS[name]
    for hold in (hmax, hmin):
        assert cvd(hold, th.tokens.accent) >= to_accent - 0.05 and cvd(hold, th.raw_trace) >= to_raw - 0.05
        assert de(hold, th.tokens.accent) >= 15.0 and contrast(hold, SURFACE[name]) >= 3.0
    assert cvd(hmax, hmin) >= between - 0.05 and de(hmax, hmin) >= 15.0


@pytest.mark.parametrize("name, on, contrast_floor", [("dark", "dark", 5.1), ("light", "light", 3.1),
                                                     ("light", "terraflux", 3.09)])
def test_control_palette(name, on, contrast_floor):
    """Kontrol paleti: 8 rengin her çifti normal ΔE ≥ 19, renk körlüğünde ≥ 9.8; kılavuz grisi paletten ≥ 13."""
    from spt.gui.control import control_plots as cp
    from spt.gui.views import fusion_view as fv
    palette = list(getattr(cp.PALETTE, name))
    normal, colorblind = worst(palette, every(len(palette)))
    assert normal >= 19.0 - 0.05 and colorblind >= 9.8 - 0.05
    assert min(contrast(c, SURFACE[on]) for c in palette) >= contrast_floor
    guide = {"dark": "#8a837c", "light": "#7c7670"}[name]
    assert min(de(guide, c) for c in palette) >= 12.95
    # Füzyon paleti kontrol paletinin ilk altı rengidir (sıra farklı)
    assert set(getattr(fv.PALETTE, name)) == set(palette[:6])


@pytest.mark.parametrize("name", THEME_NAMES)
def test_fusion_reference_and_raw_stand_apart(name, restore_theme):
    theme = restore_theme
    from spt.gui.views import fusion_view as fv
    theme._active = theme.THEMES[name]
    algos = sorted(set(fv.ALGO_COLORS.values()))
    assert set(algos) <= set(fv.PALETTE)
    for special in (fv.ref_color(), fv.raw_color()):
        assert min(cvd(special, c) for c in algos) >= 8.0 and contrast(special, SURFACE[name]) >= 3.0


@pytest.mark.parametrize("name", THEME_NAMES)
def test_orientation_axes_visible(name, restore_theme):
    theme = restore_theme
    from spt.gui.widgets import orientation3d as o3d
    theme._active = theme.THEMES[name]
    assert min(contrast(c, SURFACE[name]) for c in o3d.AXIS_COLORS) >= 3.0
    assert contrast(o3d.hull_color(), SURFACE[name]) >= 1.8   # dolu gövde zeminden ayrılır (kenarı algoritma renginde)
