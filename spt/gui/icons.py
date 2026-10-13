"""İkonlar (qtawesome / Phosphor) ve uygulama logosu."""
from __future__ import annotations

import math

import qtawesome as qta
from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QIcon, QLinearGradient, QPainter, QPainterPath, QPen, QPixmap

from spt.gui.theme import TERRAFLUX_DARK, Tokens, primary_text, tokens


def icon(name: str, color: str | None = None, t: Tokens | None = None) -> QIcon:
    """Etkin temanın renklerinde ikon. İşaretli (checked) durumda vurgu rengine döner."""
    t = t or tokens()
    return qta.icon(
        name,
        color=color or t.text_secondary,
        color_active=color or t.text,
        color_disabled=t.text_muted,
        color_on=t.accent,
        color_on_active=t.accent,
    )


def primary_icon(name: str) -> QIcon:
    """Birincil (Primary) düğme simgesi: düğme metniyle aynı renk (bkz. theme.primary_text)."""
    return icon(name, primary_text())


def swatch_icon(color: str, size: int = 12) -> QIcon:
    """Kanal rengini gösteren küçük yuvarlatılmış kare."""
    pix = QPixmap(size * 2, size * 2)
    pix.setDevicePixelRatio(2.0)
    pix.fill(Qt.GlobalColor.transparent)
    p = QPainter(pix)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor(color))
    p.drawRoundedRect(QRectF(1, 1, size - 2, size - 2), 3, 3)
    p.end()
    return QIcon(pix)


def line_swatch(color: str, style: Qt.PenStyle = Qt.PenStyle.SolidLine, width: float = 2.0,
                w: int = 26, h: int = 12) -> QIcon:
    """Lejant için çizgi örneği (renk + çizgi stili): renk körlüğünde stil de kimliği taşır."""
    scale = 2.0
    pix = QPixmap(int(w * scale), int(h * scale))
    pix.setDevicePixelRatio(scale)
    pix.fill(Qt.GlobalColor.transparent)
    p = QPainter(pix)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    pen = QPen(QColor(color), width)
    pen.setStyle(style)
    pen.setCapStyle(Qt.PenCapStyle.FlatCap)
    p.setPen(pen)
    p.drawLine(QPointF(1, h / 2), QPointF(w - 1, h / 2))
    p.end()
    return QIcon(pix)


def logo_pixmap(size: int, t: Tokens = TERRAFLUX_DARK) -> QPixmap:
    """Turuncu gradyanlı kare üzerinde beyaz sinüs dalgası. Marka işaretidir: her temada aynı (koyu temanın vurgu
    renkleri) — görev çubuğu ve pencere ikonu temadan bağımsızdır."""
    scale = 2.0
    pix = QPixmap(int(size * scale), int(size * scale))
    pix.setDevicePixelRatio(scale)
    pix.fill(Qt.GlobalColor.transparent)
    p = QPainter(pix)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)

    grad = QLinearGradient(0, 0, size, size)
    grad.setColorAt(0.0, QColor(t.accent))
    grad.setColorAt(1.0, QColor(t.accent_dark))
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(grad)
    p.drawRoundedRect(QRectF(0, 0, size, size), size * 0.26, size * 0.26)

    path = QPainterPath()
    margin = size * 0.2
    width = size - 2 * margin
    amp = size * 0.18
    steps = 48
    for i in range(steps + 1):
        x = margin + width * i / steps
        y = size / 2 - amp * math.sin(2 * math.pi * i / steps)
        if i == 0:
            path.moveTo(QPointF(x, y))
        else:
            path.lineTo(QPointF(x, y))
    pen = QPen(QColor("#fffdfb"), max(1.5, size * 0.08))
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    p.setPen(pen)
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawPath(path)
    p.end()
    return pix


def app_icon() -> QIcon:
    ico = QIcon()
    for s in (16, 24, 32, 48, 64, 128, 256):
        ico.addPixmap(logo_pixmap(s))
    return ico
