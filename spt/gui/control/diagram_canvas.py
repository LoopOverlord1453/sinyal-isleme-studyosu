"""Blok diyagram tuvali: bloklar, portlar, dik açılı teller, etkileşim ve geri al / yinele komutları.

Tuval yalnız *görünüm ve düzenleme* katmanıdır: her blok öğesi motorun `Block` nesnesini taşır (parametreler,
port sayıları, Türkçe adlar oradan okunur); teller (kaynak blok, çıkış portu) → (hedef blok, giriş portu)
bağlantılarıdır. Benzetim / doğrusallaştırma matematiği motorda (`spt.control.diagram`), burada yoktur.

Etkileşim (Simulink alışkanlıklarıyla)
--------------------------------------
- Çıkış portundan giriş portuna (ya da tersine) sürükle → bağlantı. Bir çıkıştan birden çok girişe bağlanabilir
  (dallanma); var olan telden Ctrl + sürükle o telin noktasından dal çeker. Bir giriş yalnız bir tel alır.
- Teller dik açılıdır; ara bölümler sürüklenerek kaydırılır (geri alınabilir). Bloklar taşınınca yol korunur.
- Seç / taşı (ızgaraya oturur), lastik bant seçimi, Sil, Kopyala / Yapıştır, Çevir (giriş-çıkış yer değiştirir).
- Görünüm: tekerlekle yakınlaştırma (imleç altında), orta tuş ya da Boşluk + sol tuşla kaydırma, "Sığdır".
"""
from __future__ import annotations

import json
import math
import re
from collections.abc import Callable
from typing import Any

import numpy as np
from PyQt6.QtCore import QLineF, QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QFontMetricsF,
    QPainter,
    QPainterPath,
    QPainterPathStroker,
    QPen,
    QPolygonF,
    QUndoCommand,
    QUndoStack,
)
from PyQt6.QtWidgets import (
    QApplication,
    QGraphicsItem,
    QGraphicsObject,
    QGraphicsPathItem,
    QGraphicsScene,
    QGraphicsView,
)

from spt.control.errors import ControlError
from spt.gui.theme import ui_font

GRID = 10
STUB = 20            # portun önünde / arkasında düz kalan tel uzunluğu
PORT = 7             # port üçgeninin boyu
HIT = 14             # port yakalama yarıçapı (sahne birimi)
PORT_PITCH = 20.0    # portlar arası uzaklık
MIME_CLIP = "application/x-spt-diagram"
# Doğrusallaştırma noktası rozetleri (sinyalin kaynak portunun yanında; tam adlar bağlam menüsünde)
LIN_BADGES = {"input": "G", "output": "Ç", "inout": "G+Ç", "outin": "Ç+G", "openinput": "AçG", "openoutput": "AçÇ",
              "looptransfer": "ÇT", "loopbreak": "Aç"}


def snap(v: float, step: float = GRID) -> float:
    return round(v / step) * step


def snap_point(p: QPointF) -> QPointF:
    return QPointF(snap(p.x()), snap(p.y()))


# ============================================================================ blok özeti (simge / metin)
def _fmt(v: float) -> str:
    v = float(v)
    if math.isinf(v):
        return "∞" if v > 0 else "−∞"
    if v.is_integer() and abs(v) < 1e6:
        return str(int(v))
    return f"{v:.4g}"


def poly_text(c, var: str = "s") -> str:
    """Azalan kuvvetli katsayılar → okunur polinom ("s² + 2s + 1")."""
    sup = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")
    c = np.atleast_1d(np.asarray(c, dtype=np.float64)).ravel()
    nz = np.flatnonzero(c != 0)
    if nz.size == 0:
        return "0"
    c = c[nz[0]:]
    n = c.size - 1
    parts: list[str] = []
    for i, a in enumerate(c):
        k = n - i
        if a == 0:
            continue
        mag = abs(a)
        coef = "" if (mag == 1 and k > 0) else _fmt(mag)
        term = coef + ("" if k == 0 else (var if k == 1 else var + str(k).translate(sup)))
        sign = "−" if a < 0 else "+"
        if not parts:
            parts.append(("−" if a < 0 else "") + term)
        else:
            parts.append(f" {sign} {term}")
    return "".join(parts)


def inv_poly_text(c, var: str = "z") -> str:
    """Artan z⁻¹ kuvvetli katsayılar (ayrık filtre) → "1 − 0.5z⁻¹"."""
    sup = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")
    c = np.atleast_1d(np.asarray(c, dtype=np.float64)).ravel()
    parts: list[str] = []
    for k, a in enumerate(c):
        if a == 0:
            continue
        mag = abs(a)
        coef = "" if (mag == 1 and k > 0) else _fmt(mag)
        term = coef + ("" if k == 0 else var + ("-" + str(k)).translate(sup))
        sign = "−" if a < 0 else "+"
        parts.append((("−" if a < 0 else "") + term) if not parts else f" {sign} {term}")
    return "".join(parts) or "0"


def _roots_text(roots, var: str) -> str:
    """Kökler → çarpanlar: gerçel kök (s + 1), eşlenik çift (s² + 2s + 5)."""
    r = list(np.atleast_1d(np.asarray(roots, dtype=np.complex128)))
    out: list[str] = []
    used = [False] * len(r)
    for i, z in enumerate(r):
        if used[i]:
            continue
        used[i] = True
        if abs(z.imag) <= 1e-12 * max(1.0, abs(z)):
            out.append(f"({poly_text([1.0, -z.real], var)})" if z.real != 0 else var)
            continue
        for j in range(i + 1, len(r)):
            if not used[j] and abs(r[j] - z.conjugate()) <= 1e-9 * max(1.0, abs(z)):
                used[j] = True
                break
        out.append(f"({poly_text([1.0, -2 * z.real, abs(z) ** 2], var)})")
    return "".join(out) or "1"


def block_summary(block) -> tuple[str, Any]:
    """Blok içinde gösterilecek özet: ("frac", (pay, payda)) | ("text", [satırlar]) | ("glyph", ad)."""
    t = block.type_name
    p = block.params
    try:
        if t == "TransferFcn":
            num = np.asarray(p["num"])
            n = poly_text(num[0] if num.ndim == 2 else num, "s") + (" …" if num.ndim == 2 and num.shape[0] > 1 else "")
            return "frac", (n, poly_text(p["den"], "s"))
        if t == "DiscreteTransferFcn":
            num = np.asarray(p["num"])
            n = poly_text(num[0] if num.ndim == 2 else num, "z") + (" …" if num.ndim == 2 and num.shape[0] > 1 else "")
            return "frac", (n, poly_text(p["den"], "z"))
        if t == "DiscreteFilter":
            return "frac", (inv_poly_text(p["num"]), inv_poly_text(p["den"]))
        if t == "ZeroPole":
            k = float(p["gain"])
            num = _roots_text(p["zeros"], "s")
            numer = ("" if k == 1 else _fmt(k)) + (num if num != "1" or k == 1 else "")    # 3(s + 1), 3, 1
            return "frac", (numer, _roots_text(p["poles"], "s"))
        if t in ("StateSpace", "Observer"):
            return "text", ["ẋ = Ax + Bu", "y = Cx + Du"]
        if t == "DiscreteStateSpace":
            return "text", ["x⁺ = Ax + Bu", "y = Cx + Du"]
        if t == "Gain":
            k = np.asarray(p["K"])
            return "text", [_fmt(k.ravel()[0]) if k.size == 1 else "K"]
        if t == "Sum":
            return "text", ["Σ"]
        if t == "Product":
            return "text", ["×" if "/" not in p["ops"] else "× ÷"]
        if t == "Integrator":
            return "frac", ("1", "s")
        if t == "DiscreteIntegrator":
            m = p["method"]
            num = {"forward_euler": "K·Ts", "backward_euler": "K·Ts·z", "trapezoidal": "K·Ts(z + 1)"}.get(m, "K·Ts")
            den = "2(z − 1)" if m == "trapezoidal" else "z − 1"
            return "frac", (num, den)
        if t == "UnitDelay":
            return "frac", ("1", "z")
        if t == "DiscreteDerivative":
            return "frac", ("K(z − 1)", "Ts·z")
        if t == "FilteredDerivative":
            return "text", ["du/dt", "filtreli"]
        if t == "TransportDelay":
            return "text", [f"e^(−{_fmt(p['delay'])}s)"]
        if t in ("PIDController", "PIDController2DOF"):
            ts = p["Ts"].period
            var = "(s)" if ts == 0 else ("(z)" if 0 < ts < math.inf else "")
            lines = [p["controller"] + var]
            if t == "PIDController2DOF":
                lines.append("2-DOF")
            return "text", lines
        if t == "Constant":
            v = np.asarray(p["value"]).ravel()
            return "text", [_fmt(v[0]) if v.size == 1 else f"[{v.size}]"]
        if t == "Fcn":
            e = str(p["expr"])
            return "text", ["f(u)", e if len(e) <= 16 else e[:15] + "…"]
        if t == "MinMax":
            return "text", [p["function"]]
        if t == "Memory":
            return "text", ["Bellek"]
        if t == "ZeroOrderHold":
            return "glyph", "zoh"
        if t in ("Step", "SineWave", "Ramp", "PulseGenerator", "Saturation", "DeadZone", "Relay", "RateLimiter",
                 "Quantizer", "Sign", "Abs", "Scope", "Lookup1D", "Switch", "RandomNumber", "UniformRandomNumber",
                 "BandLimitedWhiteNoise", "Ground", "Terminator", "Clock", "Mux", "Demux", "VectorConcatenate"):
            return "glyph", t
        if t == "DataSource":
            return "text", ["Veri", f"{np.asarray(p['values']).shape[1]} kanal"]
        if t in ("Inport", "Outport"):
            return "text", [str(p["port"])]
        if t == "Selector":
            return "text", ["Seç"]
    except (KeyError, IndexError, TypeError, ValueError, AttributeError):
        pass
    words = block.display_name.split()
    lines, cur = [], ""
    for w in words:
        if len(cur) + len(w) + 1 > 14 and cur:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return "text", lines[:3]


_GLYPHS: dict[str, list[list[tuple[float, float]]]] = {
    "Step": [[(0.1, 0.8), (0.45, 0.8), (0.45, 0.2), (0.9, 0.2)]],
    "Ramp": [[(0.1, 0.8), (0.35, 0.8), (0.9, 0.2)]],
    "PulseGenerator": [[(0.08, 0.8), (0.2, 0.8), (0.2, 0.2), (0.4, 0.2), (0.4, 0.8), (0.6, 0.8), (0.6, 0.2),
                        (0.8, 0.2), (0.8, 0.8), (0.92, 0.8)]],
    "Saturation": [[(0.1, 0.75), (0.35, 0.75), (0.65, 0.25), (0.9, 0.25)], [(0.5, 0.1), (0.5, 0.9)],
                   [(0.1, 0.5), (0.9, 0.5)]],
    "DeadZone": [[(0.1, 0.85), (0.38, 0.5), (0.62, 0.5), (0.9, 0.15)], [(0.5, 0.1), (0.5, 0.9)]],
    "Relay": [[(0.15, 0.75), (0.65, 0.75), (0.65, 0.25), (0.88, 0.25)], [(0.12, 0.75), (0.35, 0.75), (0.35, 0.25),
                                                                          (0.85, 0.25)]],
    "RateLimiter": [[(0.12, 0.8), (0.35, 0.8), (0.65, 0.2), (0.88, 0.2)], [(0.45, 0.35), (0.6, 0.35)]],
    "Quantizer": [[(0.1, 0.85), (0.3, 0.85), (0.3, 0.65), (0.5, 0.65), (0.5, 0.45), (0.7, 0.45), (0.7, 0.25),
                   (0.9, 0.25)]],
    "Sign": [[(0.1, 0.75), (0.5, 0.75), (0.5, 0.25), (0.9, 0.25)], [(0.1, 0.5), (0.9, 0.5)]],
    "Abs": [[(0.12, 0.2), (0.5, 0.8), (0.88, 0.2)]],
    "Lookup1D": [[(0.1, 0.8), (0.3, 0.55), (0.55, 0.5), (0.9, 0.15)], [(0.1, 0.9), (0.1, 0.1)]],
    "Switch": [[(0.1, 0.25), (0.35, 0.25)], [(0.1, 0.75), (0.35, 0.75)], [(0.35, 0.25), (0.75, 0.5), (0.9, 0.5)]],
    "Ground": [[(0.5, 0.15), (0.5, 0.5)], [(0.2, 0.5), (0.8, 0.5)], [(0.3, 0.65), (0.7, 0.65)],
               [(0.4, 0.8), (0.6, 0.8)]],
    "Terminator": [[(0.2, 0.5), (0.55, 0.5)], [(0.55, 0.25), (0.55, 0.75)], [(0.7, 0.3), (0.7, 0.7)]],
    "zoh": [[(0.1, 0.75), (0.3, 0.75), (0.3, 0.5), (0.5, 0.5), (0.5, 0.3), (0.7, 0.3), (0.7, 0.55), (0.9, 0.55)]],
    "RandomNumber": [[(0.08, 0.5), (0.2, 0.3), (0.3, 0.7), (0.42, 0.2), (0.55, 0.75), (0.68, 0.35), (0.8, 0.6),
                      (0.92, 0.45)]],
}
_GLYPHS["UniformRandomNumber"] = _GLYPHS["RandomNumber"]
_GLYPHS["BandLimitedWhiteNoise"] = _GLYPHS["RandomNumber"]


def glyph_path(name: str, rect: QRectF) -> QPainterPath:
    path = QPainterPath()

    def P(x: float, y: float) -> QPointF:
        return QPointF(rect.left() + x * rect.width(), rect.top() + y * rect.height())

    if name in ("SineWave",):
        n = 40
        for i in range(n + 1):
            x = 0.1 + 0.8 * i / n
            y = 0.5 - 0.32 * math.sin(2 * math.pi * i / n)
            (path.moveTo if i == 0 else path.lineTo)(P(x, y))
        return path
    if name == "Scope":
        path.addRoundedRect(QRectF(P(0.15, 0.2), P(0.85, 0.72)), 3, 3)
        n = 24
        for i in range(n + 1):
            x = 0.22 + 0.56 * i / n
            y = 0.46 - 0.15 * math.sin(2 * math.pi * i / n)
            (path.moveTo if i == 0 else path.lineTo)(P(x, y))
        path.moveTo(P(0.4, 0.84))
        path.lineTo(P(0.6, 0.84))
        return path
    if name == "Clock":
        path.addEllipse(QRectF(P(0.25, 0.2), P(0.75, 0.8)))
        path.moveTo(P(0.5, 0.5))
        path.lineTo(P(0.5, 0.3))
        path.moveTo(P(0.5, 0.5))
        path.lineTo(P(0.64, 0.58))
        return path
    for line in _GLYPHS.get(name, []):
        path.moveTo(P(*line[0]))
        for q in line[1:]:
            path.lineTo(P(*q))
    return path


# ============================================================================ port
class PortItem(QGraphicsItem):
    """Bloğun giriş / çıkış portu (bloğun alt öğesi). `kind`: "in" | "out"."""

    def __init__(self, block_item: "BlockItem", kind: str, index: int):
        super().__init__(block_item)
        self.block_item = block_item
        self.kind = kind
        self.index = index
        self.hover = False
        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setZValue(2)

    def boundingRect(self) -> QRectF:  # noqa: N802
        return QRectF(-HIT / 2, -HIT / 2, HIT, HIT)

    def facing(self) -> int:
        """Telin bu porttan çıkış (çıkış portu) ya da porta giriş (giriş portu) yönü: +1 sağa, −1 sola."""
        return -1 if self.block_item.flipped else 1

    def paint(self, painter: QPainter, option, widget=None) -> None:
        t = self.block_item.tokens
        connected = self.block_item.scene_port_connected(self.kind, self.index)
        color = QColor(t.accent if self.hover else (t.text_secondary if connected else t.text_muted))
        d = self.facing()
        if self.kind == "in":     # blok kenarına doğru (içeri) bakan üçgen, kenarın dışında
            tri = QPolygonF([QPointF(-d * PORT, -PORT / 2), QPointF(0, 0), QPointF(-d * PORT, PORT / 2)])
        else:                     # dışarı bakan üçgen, kenarın dışında
            tri = QPolygonF([QPointF(0, -PORT / 2), QPointF(d * PORT, 0), QPointF(0, PORT / 2)])
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(color, 1.4))
        painter.setBrush(QBrush(color) if (self.hover or not connected) else Qt.BrushStyle.NoBrush)
        if connected and not self.hover:
            painter.drawPolyline(tri)
        else:
            painter.drawPolygon(tri)

    def hoverEnterEvent(self, event) -> None:  # noqa: N802
        self.hover = True
        self.update()

    def hoverLeaveEvent(self, event) -> None:  # noqa: N802
        self.hover = False
        self.update()

    def scene_point(self) -> QPointF:
        return self.mapToScene(QPointF(0, 0))


# ============================================================================ blok
class BlockItem(QGraphicsObject):
    """Tuvaldeki blok. `block`: motorun Block nesnesi; GUI bilgisi `block.meta`'da (pos, flip, lin_io)."""

    def __init__(self, block, tokens):
        super().__init__()
        self.block = block
        self.tokens = tokens
        self.error = False
        self.in_ports: list[PortItem] = []
        self.out_ports: list[PortItem] = []
        self._w = 80.0
        self._h = 40.0
        self._label_w = 80.0
        self.setFlags(QGraphicsItem.GraphicsItemFlag.ItemIsMovable | QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
                      | QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges)
        self.setZValue(1)
        self.setAcceptHoverEvents(True)
        pos = block.meta.get("pos")
        if pos is not None:
            self.setPos(QPointF(float(pos[0]), float(pos[1])))
        self.rebuild()

    # ---------------------------------------------------------------- yapı
    @property
    def name(self) -> str:
        return self.block.name

    @property
    def flipped(self) -> bool:
        return bool(self.block.meta.get("flip", False))

    def lin_io(self) -> dict[int, str]:
        raw = self.block.meta.get("lin_io") or {}
        return {int(k): v for k, v in raw.items()}

    def n_in(self) -> int:
        try:
            return int(self.block.n_inputs())
        except ControlError:
            return 0

    def n_out(self) -> int:
        try:
            return int(self.block.n_outputs())
        except ControlError:
            return 0

    def _labels(self, kind: str) -> list[str]:
        try:
            return list(self.block.input_labels() if kind == "in" else self.block.output_labels())
        except ControlError:
            return []

    def rebuild(self) -> None:
        """Parametre / çevirme değişince boyutu ve portları yeniden kurar."""
        self.prepareGeometryChange()
        ni, no = self.n_in(), self.n_out()
        n = max(ni, no, 1)
        kind, data = block_summary(self.block)
        fm = QFontMetricsF(ui_font(12))
        if kind == "frac":
            tw = max(fm.horizontalAdvance(data[0]), fm.horizontalAdvance(data[1]))
            th = 2 * fm.height() + 8
        elif kind == "text":
            tw = max((fm.horizontalAdvance(s) for s in data), default=0.0)
            th = len(data) * fm.height()
        else:
            tw, th = 30.0, 24.0
        t = self.block.type_name
        if t in ("Mux", "Demux", "VectorConcatenate"):
            w = 10.0
        elif t == "Sum":
            w = 40.0
        elif t in ("Gain",):
            w = max(50.0, snap(tw + 34, GRID))
        else:
            w = max(60.0 if kind == "glyph" else 70.0, math.ceil((tw + 24) / GRID) * GRID)
        fl = QFontMetricsF(ui_font(10))
        lab_in = max((fl.horizontalAdvance(s) for s in self._labels("in")), default=0) if ni > 1 else 0
        lab_out = max((fl.horizontalAdvance(s) for s in self._labels("out")), default=0) if no > 1 else 0
        self._lab_in, self._lab_out = 0.0, 0.0
        if t not in ("Mux", "Demux", "VectorConcatenate", "Sum") and (lab_in or lab_out):
            w = max(w, math.ceil((tw + lab_in + lab_out + 30) / GRID) * GRID)
            self._lab_in, self._lab_out = (lab_out, lab_in) if self.flipped else (lab_in, lab_out)
        # yükseklik 20'nin katı: portlar 20 aralıkla ortaya dizilir, tek portlu blokların portu ızgaradaki aynı
        # satıra düşer (bloklar aynı y'deyse teller düz gider)
        h = max(40.0, PORT_PITCH * n + (PORT_PITCH if n > 2 else 0.0),
                math.ceil((th + 14) / PORT_PITCH) * PORT_PITCH)
        self._w, self._h = w, h
        self._label_w = QFontMetricsF(ui_font(11)).horizontalAdvance(self.block.name) + 12
        for p in self.in_ports + self.out_ports:
            p.setParentItem(None)
            if p.scene() is not None:
                p.scene().removeItem(p)
        self.in_ports = [PortItem(self, "in", i) for i in range(ni)]
        self.out_ports = [PortItem(self, "out", i) for i in range(no)]
        self._place_ports()
        self.update()

    def _place_ports(self) -> None:
        f = self.flipped
        for kind, ports in (("in", self.in_ports), ("out", self.out_ports)):
            n = len(ports)
            left = (kind == "in") != f
            x = 0.0 if left else self._w
            for i, p in enumerate(ports):
                y = self._h / 2 + (i - (n - 1) / 2) * PORT_PITCH
                p.setPos(QPointF(x, y))

    def body_rect(self) -> QRectF:
        return QRectF(0, 0, self._w, self._h)

    def boundingRect(self) -> QRectF:  # noqa: N802
        half = max(self._w / 2 + PORT + 64, self._label_w / 2)
        return QRectF(self._w / 2 - half, -20, 2 * half, self._h + 42)

    def shape(self) -> QPainterPath:
        path = QPainterPath()
        path.addRect(self.body_rect().adjusted(-2, -2, 2, 2))
        return path

    def port(self, kind: str, index: int) -> PortItem:
        ports = self.in_ports if kind == "in" else self.out_ports
        if not 0 <= index < len(ports):
            raise ControlError(f"'{self.name}' bloğunun {index + 1}. {'girişi' if kind == 'in' else 'çıkışı'} yok.")
        return ports[index]

    def port_scene_pos(self, kind: str, index: int) -> QPointF:
        return self.port(kind, index).scene_point()

    def scene_port_connected(self, kind: str, index: int) -> bool:
        sc = self.scene()
        return bool(sc is not None and hasattr(sc, "port_connected") and sc.port_connected(self, kind, index))

    # ---------------------------------------------------------------- çizim
    def paint(self, painter: QPainter, option, widget=None) -> None:
        t = self.tokens
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = self.body_rect()
        sel = self.isSelected()
        border = QColor(t.critical if self.error else (t.accent if sel else t.border_strong))
        typ = self.block.type_name
        painter.setPen(QPen(border, 2.0 if (sel or self.error) else 1.3))
        painter.setBrush(QColor(t.elevated))
        if typ in ("Mux", "Demux", "VectorConcatenate"):
            painter.setBrush(QColor(t.text_secondary))
            painter.drawRect(r)
        elif typ == "Gain":
            if self.flipped:
                tri = QPolygonF([r.topRight(), QPointF(r.left(), r.center().y()), r.bottomRight()])
            else:
                tri = QPolygonF([r.topLeft(), QPointF(r.right(), r.center().y()), r.bottomLeft()])
            painter.drawPolygon(tri)
        elif typ == "Sum" and max(self.n_in(), 1) <= 3:
            painter.drawEllipse(r.adjusted(2, 2, -2, -2) if r.height() <= r.width() + 1 else
                                QRectF(r.left(), r.center().y() - r.width() / 2, r.width(), r.width()))
        else:
            painter.drawRoundedRect(r, 6, 6)
        self._paint_content(painter, r)
        self._paint_port_labels(painter, r)
        # ad (altta)
        painter.setFont(ui_font(11))
        painter.setPen(QColor(t.accent_text if sel else t.text_secondary))
        painter.drawText(QRectF(r.center().x() - self._label_w / 2, r.bottom() + 3, self._label_w, 16),
                         int(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop), self.block.name)
        self._paint_lin_marks(painter)

    def _paint_content(self, painter: QPainter, r: QRectF) -> None:
        t = self.tokens
        kind, data = block_summary(self.block)
        typ = self.block.type_name
        painter.setPen(QColor(t.text))
        painter.setFont(ui_font(12))
        inner = r.adjusted(6 + getattr(self, "_lab_in", 0.0), 4, -6 - getattr(self, "_lab_out", 0.0), -4)
        if typ == "Gain":
            inner = r.adjusted(4 if not self.flipped else 18, 4, -18 if not self.flipped else -4, -4)
        if typ == "Sum":
            painter.setFont(ui_font(15))
            painter.drawText(r, int(Qt.AlignmentFlag.AlignCenter), "Σ")
            return
        if typ in ("Mux", "Demux", "VectorConcatenate"):
            return
        if kind == "frac":
            fm = QFontMetricsF(painter.font())
            num, den = data
            cy = inner.center().y()
            painter.drawText(QRectF(inner.left(), cy - fm.height() - 3, inner.width(), fm.height()),
                             int(Qt.AlignmentFlag.AlignCenter), num)
            painter.drawText(QRectF(inner.left(), cy + 3, inner.width(), fm.height()),
                             int(Qt.AlignmentFlag.AlignCenter), den)
            wline = max(fm.horizontalAdvance(num), fm.horizontalAdvance(den)) + 6
            painter.setPen(QPen(QColor(t.text_secondary), 1.0))
            painter.drawLine(QPointF(inner.center().x() - wline / 2, cy), QPointF(inner.center().x() + wline / 2, cy))
        elif kind == "text":
            painter.drawText(inner, int(Qt.AlignmentFlag.AlignCenter), "\n".join(data))
        else:
            side = min(inner.width(), inner.height(), 34.0)
            g = QRectF(inner.center().x() - side / 2, inner.center().y() - side / 2, side, side)
            pen = QPen(QColor(t.text), 1.6)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            if self.flipped and data not in ("Scope", "Clock"):
                painter.save()
                painter.translate(2 * g.center().x(), 0)
                painter.scale(-1, 1)
                painter.drawPath(glyph_path(data, g))
                painter.restore()
            else:
                painter.drawPath(glyph_path(data, g))

    def _paint_port_labels(self, painter: QPainter, r: QRectF) -> None:
        t = self.tokens
        typ = self.block.type_name
        painter.setFont(ui_font(10))
        painter.setPen(QColor(t.text_muted))
        if typ == "Sum":
            signs = str(self.block.params.get("signs", "")).replace("|", "")
            painter.setPen(QColor(t.text_secondary))
            for i, p in enumerate(self.in_ports):
                s = signs[i] if i < len(signs) else "+"
                pos = p.pos()
                dx = 5 if not self.flipped else -15
                painter.drawText(QRectF(pos.x() + dx, pos.y() - 6, 10, 12), int(Qt.AlignmentFlag.AlignCenter),
                                 "−" if s == "-" else "+")
            return
        if typ in ("Mux", "Demux", "VectorConcatenate"):
            return
        for kind, ports in (("in", self.in_ports), ("out", self.out_ports)):
            labels = self._labels(kind)
            if len(ports) <= 1:
                continue
            fm = QFontMetricsF(painter.font())
            for p, text in zip(ports, labels):
                pos = p.pos()
                at_left = pos.x() < r.center().x()
                tw = fm.horizontalAdvance(text) + 2
                x = pos.x() + 3 if at_left else pos.x() - 3 - tw
                rect = QRectF(x, pos.y() - 7, tw, 14)
                align = Qt.AlignmentFlag.AlignLeft if at_left else Qt.AlignmentFlag.AlignRight
                painter.drawText(rect, int(align | Qt.AlignmentFlag.AlignVCenter), text)

    def _paint_lin_marks(self, painter: QPainter) -> None:
        marks = self.lin_io()
        if not marks:
            return
        t = self.tokens
        painter.setFont(ui_font(10))
        for port, kind in marks.items():
            if not 0 <= port < len(self.out_ports):
                continue
            pos = self.out_ports[port].pos()
            d = -1 if self.flipped else 1
            text = LIN_BADGES.get(kind, kind)
            fm = QFontMetricsF(painter.font())
            w = fm.horizontalAdvance(text) + 8
            x = pos.x() + d * (PORT + 4) - (w if d < 0 else 0)
            rect = QRectF(x, pos.y() - 18, w, 14)
            painter.setPen(QPen(QColor(t.accent), 1.0))
            painter.setBrush(QColor(t.accent_tint))
            painter.drawRoundedRect(rect, 4, 4)
            painter.setPen(QColor(t.accent_text))
            painter.drawText(rect, int(Qt.AlignmentFlag.AlignCenter), text)

    # ---------------------------------------------------------------- konum
    def itemChange(self, change, value):  # noqa: N802
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionChange:
            return snap_point(value)
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            self.block.meta["pos"] = [float(self.pos().x()), float(self.pos().y())]
            sc = self.scene()
            if sc is not None and hasattr(sc, "block_moved"):
                sc.block_moved(self)
        return super().itemChange(change, value)

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        sc = self.scene()
        if sc is not None and hasattr(sc, "block_double_clicked"):
            sc.block_double_clicked.emit(self.name)
        event.accept()


# ============================================================================ tel
def _simplify(pts: list[QPointF]) -> list[QPointF]:
    """Ardışık aynı noktaları ve doğrusal ara noktaları atar (uçlar korunur)."""
    out: list[QPointF] = []
    for p in pts:
        if out and abs(out[-1].x() - p.x()) < 1e-6 and abs(out[-1].y() - p.y()) < 1e-6:
            continue
        out.append(QPointF(p))
    i = 1
    while i < len(out) - 1:
        a, b, c = out[i - 1], out[i], out[i + 1]
        if (abs(a.x() - b.x()) < 1e-6 and abs(b.x() - c.x()) < 1e-6) or \
           (abs(a.y() - b.y()) < 1e-6 and abs(b.y() - c.y()) < 1e-6):
            out.pop(i)
        else:
            i += 1
    return out


def _horizontal(a: QPointF, b: QPointF) -> bool:
    return abs(a.y() - b.y()) < 1e-6


class WireItem(QGraphicsPathItem):
    """Dik açılı bağlantı: (src, src_port) → (dst, dst_port). Dal ise `parent_wire` üzerindeki `anchor` (yol
    uzunluğunun kesri) noktasından başlar. `vertices`: elle düzenlenmiş ara köşeler (None: otomatik yol)."""

    def __init__(self, src: BlockItem, src_port: int, dst: BlockItem, dst_port: int, tokens,
                 parent_wire: "WireItem | None" = None, anchor: float = 0.5,
                 vertices: list[QPointF] | None = None):
        super().__init__()
        self.src, self.src_port, self.dst, self.dst_port = src, src_port, dst, dst_port
        self.tokens = tokens
        self.parent_wire = parent_wire
        self.anchor = float(anchor)
        self.vertices: list[QPointF] | None = [QPointF(v) for v in vertices] if vertices else None
        self._h_first = True
        self._h_last = True
        self._last_ends: tuple[QPointF, QPointF] | None = None
        self._drag: tuple[int, bool, list[QPointF], list[QPointF] | None] | None = None
        self.setFlags(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
        self.setAcceptHoverEvents(True)
        self.setZValue(0)
        if self.vertices:
            self._record_orientation(self.route())

    # ---------------------------------------------------------------- geometri
    def start_point(self) -> tuple[QPointF, int | None]:
        """Başlangıç noktası ve çıkış yönü (dal başlangıcında None: yön üst telin bölümünden)."""
        if self.parent_wire is not None and self.parent_wire.scene() is not None:
            return self.parent_wire.point_at(self.anchor)[0], None
        port = self.src.port("out", self.src_port)
        return port.scene_point(), port.facing()

    def end_point(self) -> tuple[QPointF, int]:
        port = self.dst.port("in", self.dst_port)
        return port.scene_point(), port.facing()

    def _lane(self, S: QPointF, E: QPointF) -> float:
        bottom = max(self.src.sceneBoundingRect().bottom(), self.dst.sceneBoundingRect().bottom())
        return snap(bottom + GRID)

    def auto_route(self) -> list[QPointF]:
        S, dS = self.start_point()
        E, dE = self.end_point()
        b = E.x() - dE * STUB
        if dS is None:                                     # dal: üst telin bölüm yönüne göre
            _, seg_h = self.parent_wire.point_at(self.anchor)
            if seg_h:
                if (E.x() - S.x()) * dE > STUB / 2:
                    return _simplify([S, QPointF(S.x(), E.y()), E])
                lane = self._lane(S, E)
                return _simplify([S, QPointF(S.x(), lane), QPointF(b, lane), QPointF(b, E.y()), E])
            return _simplify([S, QPointF(b, S.y()), QPointF(b, E.y()), E])
        if dS == dE:
            if abs(S.y() - E.y()) < 1e-6 and (E.x() - S.x()) * dS > 0:
                return [S, E]
            if (E.x() - S.x()) * dS >= 2 * STUB:
                mx = snap((S.x() + E.x()) / 2)
                return _simplify([S, QPointF(mx, S.y()), QPointF(mx, E.y()), E])
            a = S.x() + dS * STUB
            lane = self._lane(S, E)
            return _simplify([S, QPointF(a, S.y()), QPointF(a, lane), QPointF(b, lane), QPointF(b, E.y()), E])
        a = S.x() + dS * STUB
        x = max(a, b) if dS > 0 else min(a, b)
        return _simplify([S, QPointF(x, S.y()), QPointF(x, E.y()), E])

    def route(self) -> list[QPointF]:
        if not self.vertices:
            return self.auto_route()
        S, _ = self.start_point()
        E, _ = self.end_point()
        return [S] + [QPointF(v) for v in self.vertices] + [E]

    def _record_orientation(self, pts: list[QPointF]) -> None:
        if len(pts) >= 2:
            self._h_first = _horizontal(pts[0], pts[1])
            self._h_last = _horizontal(pts[-2], pts[-1])

    def _follow_ends(self) -> None:
        """Elle düzenlenmiş yolda uçlar hareket edince köşeleri dik açıyı koruyacak biçimde uyarlar."""
        if not self.vertices:
            return
        S, _ = self.start_point()
        E, _ = self.end_point()
        if self._last_ends is not None:
            dS = S - self._last_ends[0]
            dE = E - self._last_ends[1]
            if (dS.x() or dS.y()) and abs(dS.x() - dE.x()) < 1e-6 and abs(dS.y() - dE.y()) < 1e-6:
                self.vertices = [v + dS for v in self.vertices]      # iki uç birlikte taşındı: yolu da taşı
        v = self.vertices
        if self._h_first:
            v[0].setY(S.y())
        else:
            v[0].setX(S.x())
        if self._h_last:
            v[-1].setY(E.y())
        else:
            v[-1].setX(E.x())

    def update_path(self) -> None:
        if self.scene() is None:
            return
        self._follow_ends()
        pts = _simplify(self.route())
        if len(pts) < 2:
            pts = [pts[0], QPointF(pts[0])]
        S, _ = self.start_point()
        E, _ = self.end_point()
        self._last_ends = (QPointF(S), QPointF(E))
        path = QPainterPath(pts[0])
        for p in pts[1:]:
            path.lineTo(p)
        self.prepareGeometryChange()
        self.setPath(path)
        self._pts = pts
        sc = self.scene()
        if hasattr(sc, "children_of"):
            for child in sc.children_of(self):
                child.update_path()
            for w in sc.wires:                     # kardeş tellerin ayrılma noktası değişmiş olabilir
                if w is not self and w.src is self.src and w.src_port == self.src_port:
                    w.update()

    def points(self) -> list[QPointF]:
        return list(getattr(self, "_pts", None) or self.route())

    def length(self) -> float:
        pts = self.points()
        return sum(QLineF(a, b).length() for a, b in zip(pts, pts[1:]))

    def point_at(self, frac: float) -> tuple[QPointF, bool]:
        """Yol uzunluğunun `frac` kesrindeki nokta ve o bölümün yatay olup olmadığı."""
        pts = self.points()
        total = self.length()
        if total <= 0 or len(pts) < 2:
            return QPointF(pts[0]), True
        target = min(max(frac, 0.0), 1.0) * total
        acc = 0.0
        for a, b in zip(pts, pts[1:]):
            seg = QLineF(a, b).length()
            if acc + seg >= target and seg > 0:
                s = (target - acc) / seg
                p = QPointF(a.x() + (b.x() - a.x()) * s, a.y() + (b.y() - a.y()) * s)
                return p, _horizontal(a, b)
            acc += seg
        return QPointF(pts[-1]), _horizontal(pts[-2], pts[-1])

    def fraction_of(self, p: QPointF) -> float:
        """Noktanın yola izdüşümünün yol uzunluğu kesri."""
        pts = self.points()
        total = self.length()
        if total <= 0:
            return 0.0
        best, best_d, acc = 0.0, math.inf, 0.0
        for a, b in zip(pts, pts[1:]):
            seg = QLineF(a, b).length()
            if seg <= 0:
                continue
            ux, uy = (b.x() - a.x()) / seg, (b.y() - a.y()) / seg
            s = min(max((p.x() - a.x()) * ux + (p.y() - a.y()) * uy, 0.0), seg)
            q = QPointF(a.x() + ux * s, a.y() + uy * s)
            d = QLineF(p, q).length()
            if d < best_d:
                best_d, best = d, (acc + s) / total
            acc += seg
        return best

    def segment_at(self, p: QPointF, tol: float = 6.0) -> int | None:
        pts = self.points()
        for k, (a, b) in enumerate(zip(pts, pts[1:])):
            r = QRectF(a, b).normalized().adjusted(-tol, -tol, tol, tol)
            if r.contains(p):
                return k
        return None

    def draggable(self, k: int | None) -> bool:
        n = len(self.points()) - 1           # bölüm sayısı
        return k is not None and 1 <= k <= n - 2

    # ---------------------------------------------------------------- çizim
    def shape(self) -> QPainterPath:
        s = QPainterPathStroker()
        s.setWidth(10)
        return s.createStroke(self.path())

    def boundingRect(self) -> QRectF:  # noqa: N802
        return super().boundingRect().adjusted(-8, -8, 8, 8)

    def paint(self, painter: QPainter, option, widget=None) -> None:
        t = self.tokens
        sel = self.isSelected()
        color = QColor(t.accent if sel else t.text_secondary)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        pen = QPen(color, 2.2 if sel else 1.5)
        pen.setJoinStyle(Qt.PenJoinStyle.MiterJoin)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(self.path())
        pts = self.points()
        if len(pts) >= 2:                         # varış oku
            E = pts[-1]
            _, dE = self.end_point()
            tip = QPolygonF([E, QPointF(E.x() - dE * 8, E.y() - 4), QPointF(E.x() - dE * 8, E.y() + 4)])
            painter.setBrush(color)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawPolygon(tip)
        split = pts[0] if self.parent_wire is not None else None
        sc = self.scene()
        if split is None and sc is not None and hasattr(sc, "wires"):
            for other in sc.wires:                 # aynı porttan önce çıkan telle ayrıldığı nokta
                if other is self:
                    break
                if other.parent_wire is None and other.src is self.src and other.src_port == self.src_port:
                    split = _split_point(pts, other.points())
                    if split is not None:
                        break
        if split is not None:                     # dallanma noktası
            painter.setBrush(color)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(split, 3.5, 3.5)

    # ---------------------------------------------------------------- bölüm sürükleme
    def hoverMoveEvent(self, event) -> None:  # noqa: N802
        k = self.segment_at(event.scenePos())
        if self.draggable(k):
            pts = self.points()
            horiz = _horizontal(pts[k], pts[k + 1])
            self.setCursor(Qt.CursorShape.SizeVerCursor if horiz else Qt.CursorShape.SizeHorCursor)
        else:
            self.unsetCursor()
        super().hoverMoveEvent(event)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        sc = self.scene()
        if event.button() == Qt.MouseButton.LeftButton and \
                event.modifiers() & Qt.KeyboardModifier.ControlModifier and hasattr(sc, "begin_branch"):
            sc.begin_branch(self, event.scenePos())
            event.accept()
            return
        super().mousePressEvent(event)
        if event.button() != Qt.MouseButton.LeftButton:
            return
        k = self.segment_at(event.scenePos())
        if self.draggable(k):
            pts = self.points()                  # çizilen (sadeleştirilmiş) yol
            old = [QPointF(v) for v in self.vertices] if self.vertices else None
            self._drag = (k, _horizontal(pts[k], pts[k + 1]), pts, old)
            event.accept()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._drag is None:
            super().mouseMoveEvent(event)
            return
        k, horiz, base, _old = self._drag
        pts = [QPointF(p) for p in base]
        p = event.scenePos()
        if horiz:
            y = snap(p.y())
            pts[k].setY(y)
            pts[k + 1].setY(y)
        else:
            x = snap(p.x())
            pts[k].setX(x)
            pts[k + 1].setX(x)
        self.set_route(pts)

    def set_route(self, pts: list[QPointF]) -> None:
        """Tam yoldan (uçlar dahil) elle düzenlenmiş köşeleri kurar."""
        pts = _simplify(pts)
        self.vertices = [QPointF(v) for v in pts[1:-1]] or None
        if self.vertices:
            self._record_orientation(pts)
        self._last_ends = None
        self.update_path()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if self._drag is None:
            super().mouseReleaseEvent(event)
            return
        _k, _h, _base, old = self._drag
        self._drag = None
        new = [QPointF(v) for v in self.vertices] if self.vertices else None
        sc = self.scene()
        if sc is not None and hasattr(sc, "route_edited") and not _same_route(old, new):
            sc.route_edited(self, old, new)
        event.accept()


def _dir(a: QPointF, b: QPointF) -> tuple[int, int]:
    dx, dy = b.x() - a.x(), b.y() - a.y()
    return (int(dx > 1e-6) - int(dx < -1e-6), int(dy > 1e-6) - int(dy < -1e-6))


def _split_point(a: list[QPointF], b: list[QPointF]) -> QPointF | None:
    """Aynı noktadan başlayan iki dik açılı yolun ayrıldığı nokta (ortak kısım yoksa None)."""
    if len(a) < 2 or len(b) < 2 or QLineF(a[0], b[0]).length() > 1e-6:
        return None
    i = j = 0
    pa, pb = QPointF(a[0]), QPointF(b[0])
    moved = False
    while i < len(a) - 1 and j < len(b) - 1:
        la, lb = QLineF(pa, a[i + 1]).length(), QLineF(pb, b[j + 1]).length()
        if la < 1e-6:
            i += 1
            pa = QPointF(a[i])
            continue
        if lb < 1e-6:
            j += 1
            pb = QPointF(b[j])
            continue
        da, db = _dir(pa, a[i + 1]), _dir(pb, b[j + 1])
        if da != db:
            break
        step = min(la, lb)
        pa = QPointF(pa.x() + da[0] * step, pa.y() + da[1] * step)
        pb = QPointF(pb.x() + db[0] * step, pb.y() + db[1] * step)
        moved = True
    if not moved or i >= len(a) - 1 or j >= len(b) - 1:
        return None
    return pa


def _same_route(a: list[QPointF] | None, b: list[QPointF] | None) -> bool:
    if not a and not b:
        return True
    if not a or not b or len(a) != len(b):
        return False
    return all(abs(p.x() - q.x()) < 1e-9 and abs(p.y() - q.y()) < 1e-9 for p, q in zip(a, b))


# ============================================================================ geri al komutları
class _Cmd(QUndoCommand):
    def __init__(self, scene: "DiagramScene", text: str, applied: bool = False):
        super().__init__(text)
        self.scene = scene
        self._skip = applied          # işlem zaten uygulandıysa ilk redo atlanır (sürükleme sonuçları)

    def _skip_once(self) -> bool:
        if self._skip:
            self._skip = False
            return True
        return False


class AddBlockCmd(_Cmd):
    def __init__(self, scene, item: BlockItem):
        super().__init__(scene, f"Blok ekle: {item.name}")
        self.item = item

    def redo(self) -> None:
        self.scene._insert_block(self.item)

    def undo(self) -> None:
        self.scene._detach_block(self.item)


class AddWireCmd(_Cmd):
    def __init__(self, scene, wire: WireItem):
        super().__init__(scene, "Bağlantı")
        self.wire = wire

    def redo(self) -> None:
        self.scene._insert_wire(self.wire)

    def undo(self) -> None:
        self.scene._detach_wire(self.wire)


class RemoveCmd(_Cmd):
    """Blok ve telleri siler. Silinen tellerden dal alan (silinmeyen) teller kaynak porta bağlanır."""

    def __init__(self, scene, blocks: list[BlockItem], wires: list[WireItem], text: str = "Sil"):
        super().__init__(scene, text)
        self.blocks = list(blocks)
        bset = set(self.blocks)
        ws = list(wires)
        for w in scene.wires:
            if (w.src in bset or w.dst in bset) and w not in ws:
                ws.append(w)
        wset = set(ws)
        self.wires = [w for w in scene.wires if w in wset]              # sahnedeki sırayla (üst teller önce)
        self.reanchor = [(c, c.parent_wire, c.anchor, [QPointF(v) for v in c.vertices] if c.vertices else None)
                         for c in scene.wires if c not in wset and c.parent_wire in wset]
        self.block_order = list(scene.blocks)          # geri almada sıra aynen döner (model / yürütme sırası)
        self.wire_order = list(scene.wires)

    def redo(self) -> None:
        for c, *_ in self.reanchor:
            c.parent_wire = None
            c.vertices = None
        for w in reversed(self.wires):
            self.scene._detach_wire(w)
        for b in self.blocks:
            self.scene._detach_block(b)
        for c, *_ in self.reanchor:
            c.update_path()

    def undo(self) -> None:
        for b in self.blocks:
            self.scene._insert_block(b)
        for w in self.wires:
            self.scene._insert_wire(w)
        for c, parent, anchor, verts in self.reanchor:
            c.parent_wire, c.anchor, c.vertices = parent, anchor, verts
            c._last_ends = None
            c.update_path()
        sc = self.scene
        rank = {n: i for i, n in enumerate(self.block_order)}
        sc.blocks = dict(sorted(sc.blocks.items(), key=lambda kv: rank.get(kv[0], len(rank))))
        wrank = {w: i for i, w in enumerate(self.wire_order)}
        sc.wires.sort(key=lambda w: wrank.get(w, len(wrank)))
        sc.refresh_all()


class MoveCmd(_Cmd):
    def __init__(self, scene, moves: list[tuple[BlockItem, QPointF, QPointF]]):
        super().__init__(scene, "Taşı", applied=True)
        self.moves = moves

    def redo(self) -> None:
        if self._skip_once():
            return
        for item, _old, new in self.moves:
            item.setPos(new)

    def undo(self) -> None:
        for item, old, _new in self.moves:
            item.setPos(old)


class RouteCmd(_Cmd):
    def __init__(self, scene, wire: WireItem, old, new):
        super().__init__(scene, "Tel yolu", applied=True)
        self.wire, self.old, self.new = wire, old, new

    def _set(self, verts) -> None:
        w = self.wire
        w.vertices = [QPointF(v) for v in verts] if verts else None
        w._last_ends = None
        if w.vertices:
            w._record_orientation(w.route())
        w.update_path()

    def redo(self) -> None:
        if self._skip_once():
            return
        self._set(self.new)

    def undo(self) -> None:
        self._set(self.old)


class ParamsCmd(_Cmd):
    """Parametre / ad / meta değişikliği. Port sayısı azalırsa artık olmayan portlardaki teller silinir."""

    def __init__(self, scene, item: BlockItem, new_params: dict, new_name: str | None, meta: dict | None = None):
        super().__init__(scene, f"Parametreler: {item.name}")
        self.item = item
        self.old_params = dict(item.block.params)
        self.new_params = dict(new_params)
        self.old_name = item.name
        self.new_name = new_name or item.name
        self.old_meta = {k: item.block.meta.get(k) for k in (meta or {})}
        self.new_meta = dict(meta or {})
        self.removed: RemoveCmd | None = None

    def _apply(self, params: dict, name: str, meta: dict) -> None:
        b = self.item.block
        b.params = dict(params)
        if name != b.name:
            self.scene._rename(self.item, name)
        for k, v in meta.items():
            if v is None:
                b.meta.pop(k, None)
            else:
                b.meta[k] = v
        self.item.rebuild()

    def redo(self) -> None:
        self._apply(self.new_params, self.new_name, self.new_meta)
        if self.removed is None:
            ni, no = self.item.n_in(), self.item.n_out()
            bad = [w for w in self.scene.wires if (w.dst is self.item and w.dst_port >= ni)
                   or (w.src is self.item and w.src_port >= no)]
            self.removed = RemoveCmd(self.scene, [], bad)
        self.removed.redo()
        self.scene.refresh_all()

    def undo(self) -> None:
        self._apply(self.old_params, self.old_name, self.old_meta)
        if self.removed is not None:
            self.removed.undo()
        self.scene.refresh_all()


class FlipCmd(_Cmd):
    def __init__(self, scene, items: list[BlockItem]):
        super().__init__(scene, "Çevir")
        self.items = list(items)
        self.routes = {w: ([QPointF(v) for v in w.vertices] if w.vertices else None, w._h_first, w._h_last)
                       for w in scene.wires if w.src in self.items or w.dst in self.items}

    def _toggle(self) -> None:
        for it in self.items:
            if it.flipped:
                it.block.meta.pop("flip", None)        # varsayılan yön: anahtar yok (dosya gidiş-dönüşü aynı)
            else:
                it.block.meta["flip"] = True
            it.rebuild()

    def redo(self) -> None:
        self._toggle()
        for w in self.routes:
            w.vertices = None                      # çevrilen bloğa bağlı teller otomatik yola döner
        self.scene.refresh_all()

    def undo(self) -> None:
        self._toggle()
        for w, (verts, h0, h1) in self.routes.items():
            w.vertices = [QPointF(v) for v in verts] if verts else None
            w._h_first, w._h_last = h0, h1
            w._last_ends = None
        self.scene.refresh_all()


# ============================================================================ sahne
class DiagramScene(QGraphicsScene):
    """Diyagram sahnesi: blok / tel kayıtları, bağlama etkileşimi ve geri alınabilir düzenleme API'si."""

    changed_doc = pyqtSignal()                 # diyagram değişti (kaydedilmemiş)
    message = pyqtSignal(str, str)             # (metin, durum: good / warning / critical / idle)
    block_double_clicked = pyqtSignal(str)
    context_requested = pyqtSignal(object, object, object)   # (öğe | None, ekran konumu, sahne konumu)
    block_renamed = pyqtSignal(str, str)                      # (eski ad, yeni ad)

    def __init__(self, theme, undo: QUndoStack | None = None, parent=None):
        super().__init__(parent)
        self.theme = theme
        self.tokens = theme.tokens
        self.undo = undo if undo is not None else QUndoStack(self)
        self.blocks: dict[str, BlockItem] = {}
        self.wires: list[WireItem] = []
        self._wiring: dict | None = None
        self._press: dict[BlockItem, QPointF] = {}
        self.setSceneRect(QRectF(-4000, -3000, 8000, 6000))
        self.setItemIndexMethod(QGraphicsScene.ItemIndexMethod.NoIndex)
        self.undo.indexChanged.connect(lambda *_: self.changed_doc.emit())

    # ---------------------------------------------------------------- arka plan (ızgara)
    def drawBackground(self, painter: QPainter, rect: QRectF) -> None:  # noqa: N802
        t = self.tokens
        painter.fillRect(rect, QColor(t.surface))
        view = self.views()[0] if self.views() else None
        scale = view.transform().m11() if view is not None else 1.0
        step = GRID * 2
        while step * scale < 9:
            step *= 2
        left = math.floor(rect.left() / step) * step
        top = math.floor(rect.top() / step) * step
        minor = QPen(QColor(t.grid), 0)
        major = QPen(QColor(t.border), 0)
        x = left
        lines_minor, lines_major = [], []
        while x < rect.right():
            (lines_major if abs(x % (step * 5)) < 1e-6 else lines_minor).append(QLineF(x, rect.top(), x, rect.bottom()))
            x += step
        y = top
        while y < rect.bottom():
            (lines_major if abs(y % (step * 5)) < 1e-6 else lines_minor).append(QLineF(rect.left(), y, rect.right(), y))
            y += step
        painter.setPen(minor)
        painter.drawLines(lines_minor)
        painter.setPen(major)
        painter.drawLines(lines_major)

    # ---------------------------------------------------------------- kayıt (komutların iç işleri)
    def _insert_block(self, item: BlockItem) -> None:
        if item.name in self.blocks and self.blocks[item.name] is not item:
            raise ControlError(f"'{item.name}' adlı blok zaten var.")
        self.blocks[item.name] = item
        if item.scene() is not self:
            self.addItem(item)
        item.block.meta["pos"] = [float(item.pos().x()), float(item.pos().y())]

    def _detach_block(self, item: BlockItem) -> None:
        self.blocks.pop(item.name, None)
        if item.scene() is self:
            item.setSelected(False)
            self.removeItem(item)

    def _insert_wire(self, wire: WireItem) -> None:
        if wire not in self.wires:
            self.wires.append(wire)
        if wire.scene() is not self:
            self.addItem(wire)
        wire._last_ends = None
        wire.update_path()
        wire.src.update()
        wire.dst.update()

    def _detach_wire(self, wire: WireItem) -> None:
        if wire in self.wires:
            self.wires.remove(wire)
        if wire.scene() is self:
            wire.setSelected(False)
            self.removeItem(wire)
        wire.src.update()
        wire.dst.update()

    def _rename(self, item: BlockItem, new: str) -> None:
        old = item.name
        order = list(self.blocks)
        item.block.name = new
        self.blocks = {(new if n == old else n): self.blocks[n] for n in order}     # sıra korunur
        item.rebuild()
        self.block_renamed.emit(old, new)

    # ---------------------------------------------------------------- sorgular
    def port_connected(self, item: BlockItem, kind: str, index: int) -> bool:
        if kind == "in":
            return any(w.dst is item and w.dst_port == index for w in self.wires)
        return any(w.src is item and w.src_port == index for w in self.wires)

    def children_of(self, wire: WireItem) -> list[WireItem]:
        return [w for w in self.wires if w.parent_wire is wire]

    def block_moved(self, item: BlockItem) -> None:
        """Blok taşındı: ona bağlı teller (dallar üst telle birlikte) yeniden çizilir."""
        for w in self.wires:
            if w.dst is item or (w.src is item and w.parent_wire is None):
                w.update_path()

    def refresh_all(self) -> None:
        for w in self.wires:
            if w.parent_wire is None:
                w.update_path()
        for w in self.wires:
            if w.parent_wire is not None:
                w.update_path()
        for b in self.blocks.values():
            b.update()

    def unique_name(self, base: str) -> str:
        # blok adında '/' (alt sistem yol ayıracı), satır sonu ve sekme olamaz: varsayılan adlar da buna uysun
        base = re.sub(r"\s*[/\n\t\r]+\s*", "-", base).strip(" -") or "Blok"
        if base not in self.blocks:
            return base
        m = re.match(r"^(.*?)(\d+)$", base)
        stem = m.group(1).rstrip() if m else base
        i = int(m.group(2)) + 1 if m else 2
        while f"{stem} {i}" in self.blocks:
            i += 1
        return f"{stem} {i}"

    def selected_blocks(self) -> list[BlockItem]:
        return [i for i in self.selectedItems() if isinstance(i, BlockItem)]

    def selected_wires(self) -> list[WireItem]:
        return [i for i in self.selectedItems() if isinstance(i, WireItem)]

    def find_port(self, p: QPointF, kind: str | None = None, radius: float = HIT) -> PortItem | None:
        best, best_d = None, radius
        for item in self.items(QRectF(p.x() - radius, p.y() - radius, 2 * radius, 2 * radius)):
            if isinstance(item, PortItem) and (kind is None or item.kind == kind):
                d = QLineF(item.scene_point(), p).length()
                if d <= best_d:
                    best, best_d = item, d
        return best

    def wire_into(self, item: BlockItem, port: int) -> WireItem | None:
        return next((w for w in self.wires if w.dst is item and w.dst_port == port), None)

    # ---------------------------------------------------------------- düzenleme API'si (geri alınabilir)
    def add_block(self, block, pos: QPointF | tuple[float, float] | None = None) -> BlockItem:
        if block.name in self.blocks:
            raise ControlError(f"'{block.name}' adlı blok zaten var.")
        if pos is not None:
            p = pos if isinstance(pos, QPointF) else QPointF(float(pos[0]), float(pos[1]))
            block.meta["pos"] = [snap(p.x()), snap(p.y())]
        item = BlockItem(block, self.tokens)
        self.undo.push(AddBlockCmd(self, item))
        return item

    def connect(self, src: str | BlockItem, src_port: int, dst: str | BlockItem, dst_port: int,
                vertices: list | None = None, parent: WireItem | None = None, anchor: float = 0.5) -> WireItem:
        """(src, çıkış portu) → (dst, giriş portu) bağlantısı; portlar 0 tabanlı."""
        s = self.blocks[src] if isinstance(src, str) else src
        d = self.blocks[dst] if isinstance(dst, str) else dst
        s.port("out", src_port)
        d.port("in", dst_port)
        if self.wire_into(d, dst_port) is not None:
            labels = d._labels("in")
            lab = labels[dst_port] if dst_port < len(labels) else str(dst_port + 1)
            raise ControlError(f"'{d.name}' bloğunun '{lab}' girişi zaten bağlı (bir girişe tek tel bağlanır).")
        if parent is not None and (parent.src is not s or parent.src_port != src_port):
            parent = None
        verts = [QPointF(*v) if not isinstance(v, QPointF) else v for v in vertices] if vertices else None
        wire = WireItem(s, src_port, d, dst_port, self.tokens, parent_wire=parent, anchor=anchor, vertices=verts)
        self.undo.push(AddWireCmd(self, wire))
        return wire

    def remove_items(self, blocks: list[BlockItem], wires: list[WireItem], text: str = "Sil") -> None:
        if not blocks and not wires:
            return
        self.undo.push(RemoveCmd(self, blocks, wires, text))

    def delete_selection(self) -> None:
        self.remove_items(self.selected_blocks(), self.selected_wires())

    def flip(self, items: list[BlockItem]) -> None:
        if items:
            self.undo.push(FlipCmd(self, items))

    def set_params(self, item: BlockItem, new_name: str | None, changes: dict) -> None:
        """Parametreleri doğrulayıp geri alınabilir biçimde uygular; hatada `ControlError` (blok değişmez)."""
        meta = {}
        changes = dict(changes)
        if "__meta_binding__" in changes:
            meta["binding"] = changes.pop("__meta_binding__")
        if new_name is not None:
            new_name = new_name.strip()
            if not new_name:
                raise ControlError("Blok adı boş olamaz.")
            if new_name != item.name and new_name in self.blocks:
                raise ControlError(f"'{new_name}' adlı başka bir blok var; adlar tek olmalı.")
            if any(c in new_name for c in "/\n\t"):
                raise ControlError("Blok adında '/', satır sonu ya da sekme olamaz.")
        tmp = clone_block(item.block)
        tmp.set_params(**changes)                  # hata → ControlError, blok değişmez
        if new_name is None and not changes and not meta:
            return
        self.undo.push(ParamsCmd(self, item, tmp.params, new_name, meta))

    def set_meta(self, item: BlockItem, key: str, value: Any, text: str) -> None:
        cmd = ParamsCmd(self, item, dict(item.block.params), None, {key: value})
        cmd.setText(text)
        self.undo.push(cmd)

    def route_edited(self, wire: WireItem, old, new) -> None:
        self.undo.push(RouteCmd(self, wire, old, new))

    def clear_all(self) -> None:
        self._wiring = None
        for w in list(self.wires):
            self._detach_wire(w)
        for b in list(self.blocks.values()):
            self._detach_block(b)
        self.undo.clear()

    def highlight(self, names: set[str]) -> None:
        for name, item in self.blocks.items():
            on = name in names
            if item.error != on:
                item.error = on
                item.update()

    # ---------------------------------------------------------------- pano (kopyala / yapıştır)
    def snapshot(self, blocks: list[BlockItem] | None = None) -> dict:
        """Seçili (ya da tüm) blokların ve aralarındaki tellerin JSON uyumlu kopyası."""
        from spt.control.diagram.block import encode_value

        items = list(self.blocks.values()) if blocks is None else list(blocks)
        bset = set(items)
        out_b = []
        for b in items:
            e = {"type": b.block.type_name, "name": b.name, "params": b.block.encoded_params(),
                 "meta": json.loads(json.dumps(encode_value_meta(b.block.meta, encode_value)))}
            if hasattr(b.block, "system"):                 # alt sistemin içi de kopyalanır
                e["system"] = b.block.system.to_dict()
            out_b.append(e)
        wires = [w for w in self.wires if w.src in bset and w.dst in bset]
        index = {w: i for i, w in enumerate(wires)}
        out_w = []
        for w in wires:
            out_w.append({"src": w.src.name, "src_port": w.src_port, "dst": w.dst.name, "dst_port": w.dst_port,
                          "vertices": [[v.x(), v.y()] for v in w.vertices] if w.vertices else None,
                          "parent": index.get(w.parent_wire) if w.parent_wire is not None else None,
                          "anchor": w.anchor})
        return {"blocks": out_b, "wires": out_w}

    def copy_selection(self) -> bool:
        items = self.selected_blocks()
        if not items:
            return False
        data = json.dumps(self.snapshot(items))
        from PyQt6.QtCore import QMimeData

        mime = QMimeData()
        mime.setData(MIME_CLIP, data.encode("utf-8"))
        mime.setText(data)
        QApplication.clipboard().setMimeData(mime)
        return True

    def paste(self, data: dict | None = None, offset: QPointF | None = None, text: str = "Yapıştır") -> list[BlockItem]:
        """Kopyayı yeni adlarla ekler (tek geri al adımı); eklenen blokları döndürür."""
        if data is None:
            mime = QApplication.clipboard().mimeData()
            if mime is None or not mime.hasFormat(MIME_CLIP):
                return []
            data = json.loads(bytes(mime.data(MIME_CLIP)).decode("utf-8"))
        from spt.control.diagram.block import create_block

        off = offset if offset is not None else QPointF(2 * GRID, 2 * GRID)
        rename: dict[str, str] = {}
        made: list[BlockItem] = []
        self.undo.beginMacro(text)
        try:
            taken = set(self.blocks)
            for e in data.get("blocks", []):
                name = e["name"]
                if name in taken:
                    name = self._free_name(name, taken)
                taken.add(name)
                rename[e["name"]] = name
                if "system" in e:
                    from spt.control.diagram.blocks.ports import Subsystem
                    from spt.control.diagram.model import Model

                    blk = Subsystem(name, system=Model.from_dict(e["system"]))
                else:
                    blk = create_block(e["type"], name, **e.get("params", {}))
                blk.meta.update(e.get("meta") or {})
                pos = blk.meta.get("pos", [0.0, 0.0])
                made.append(self.add_block(blk, QPointF(float(pos[0]) + off.x(), float(pos[1]) + off.y())))
            created: list[WireItem] = []
            for e in data.get("wires", []):
                verts = [QPointF(v[0] + off.x(), v[1] + off.y()) for v in e["vertices"]] if e.get("vertices") else None
                par = e.get("parent")
                parent = created[par] if par is not None and par < len(created) else None
                created.append(self.connect(rename[e["src"]], e["src_port"], rename[e["dst"]], e["dst_port"],
                                            vertices=verts, parent=parent, anchor=e.get("anchor", 0.5)))
        finally:
            self.undo.endMacro()
        self.clearSelection()
        for it in made:
            it.setSelected(True)
        return made

    def _free_name(self, name: str, taken: set[str]) -> str:
        m = re.match(r"^(.*?)(\d+)$", name)
        stem = m.group(1).rstrip() if m else name
        i = int(m.group(2)) + 1 if m else 2
        while f"{stem} {i}" in taken:
            i += 1
        return f"{stem} {i}"

    # ---------------------------------------------------------------- fare: bağlama ve taşıma
    def _temp_path(self, a: QPointF, b: QPointF, d: int) -> QPainterPath:
        path = QPainterPath(a)
        if abs(a.y() - b.y()) < 1e-6:
            path.lineTo(b)
            return path
        mx = snap((a.x() + b.x()) / 2) if (b.x() - a.x()) * d > 2 * STUB else a.x() + d * STUB
        path.lineTo(QPointF(mx, a.y()))
        path.lineTo(QPointF(mx, b.y()))
        path.lineTo(b)
        return path

    def _start_wiring(self, origin: dict) -> None:
        temp = QGraphicsPathItem()
        pen = QPen(QColor(self.tokens.accent), 1.5, Qt.PenStyle.DashLine)
        temp.setPen(pen)
        temp.setZValue(5)
        self.addItem(temp)
        origin["temp"] = temp
        self._wiring = origin

    def begin_branch(self, wire: WireItem, p: QPointF) -> None:
        frac = wire.fraction_of(p)
        a, _ = wire.point_at(frac)
        self._start_wiring({"kind": "branch", "wire": wire, "anchor": frac, "point": a, "dir": 1})

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            port = self.find_port(event.scenePos(), radius=9)   # bloğu kenarından tutmak taşıma olarak kalsın
            if port is not None:
                if port.kind == "in" and self.wire_into(port.block_item, port.index) is not None:
                    self.message.emit("Bu giriş zaten bağlı; önce teli silin.", "warning")
                    event.accept()
                    return
                self._start_wiring({"kind": port.kind, "port": port, "point": port.scene_point(),
                                    "dir": port.facing() if port.kind == "out" else -port.facing()})
                event.accept()
                return
        super().mousePressEvent(event)
        self._press = {b: QPointF(b.pos()) for b in self.selected_blocks()}

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._wiring is not None:
            w = self._wiring
            target_kind = "out" if w["kind"] == "in" else "in"
            port = self.find_port(event.scenePos(), target_kind)
            end = port.scene_point() if port is not None else event.scenePos()
            w["temp"].setPath(self._temp_path(w["point"], end, w["dir"]))
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if self._wiring is not None and event.button() == Qt.MouseButton.LeftButton:
            w, self._wiring = self._wiring, None
            self.removeItem(w["temp"])
            grabber = self.mouseGrabberItem()           # dal çekmede tel fareyi tutuyordu
            if grabber is not None:
                grabber.ungrabMouse()
            target_kind = "out" if w["kind"] == "in" else "in"
            port = self.find_port(event.scenePos(), target_kind)
            if port is None:
                event.accept()
                return
            try:
                if w["kind"] == "out":
                    src = w["port"]
                    self.connect(src.block_item, src.index, port.block_item, port.index)
                elif w["kind"] == "in":
                    dst = w["port"]
                    self.connect(port.block_item, port.index, dst.block_item, dst.index)
                else:
                    pw: WireItem = w["wire"]
                    self.connect(pw.src, pw.src_port, port.block_item, port.index, parent=pw, anchor=w["anchor"])
            except ControlError as exc:
                self.message.emit(str(exc), "warning")
            event.accept()
            return
        super().mouseReleaseEvent(event)
        if self._press:
            moves = [(b, old, QPointF(b.pos())) for b, old in self._press.items()
                     if b.scene() is self and (abs(b.pos().x() - old.x()) > 1e-9 or abs(b.pos().y() - old.y()) > 1e-9)]
            self._press = {}
            if moves:
                self.undo.push(MoveCmd(self, moves))

    def contextMenuEvent(self, event) -> None:  # noqa: N802
        item = None
        for it in self.items(event.scenePos()):
            if isinstance(it, PortItem):
                item = it.block_item
                break
            if isinstance(it, (BlockItem, WireItem)):
                item = it
                break
        if item is not None and not item.isSelected():
            self.clearSelection()
            item.setSelected(True)
        self.context_requested.emit(item, event.screenPos(), event.scenePos())
        event.accept()


def clone_block(block):
    """Bloğun bağımsız kopyası (parametreler kalıcı biçimlerinden yeniden doğrulanır; değerler bit düzeyinde aynı).
    Alt sistem kendi `clone`'uyla kopyalanır."""
    if hasattr(block, "system"):
        return block.clone()
    from spt.control.diagram.block import create_block

    b = create_block(block.type_name, block.name, **block.encoded_params())
    b.meta = json.loads(json.dumps(block.meta, default=str))
    return b


def encode_value_meta(meta: dict, enc: Callable[[Any], Any]) -> dict:
    """Blok meta sözlüğünü JSON'a uygun hâle getirir (numpy değerleri listeye)."""
    out = {}
    for k, v in meta.items():
        if isinstance(v, dict):
            out[str(k)] = encode_value_meta(v, enc)
        else:
            out[str(k)] = enc(v)
    return out


# ============================================================================ görünüm
class DiagramView(QGraphicsView):
    """Tuval görünümü: tekerlekle yakınlaştırma, orta tuş / Boşluk + sürükle ile kaydırma, paletten bırakma."""

    block_dropped = pyqtSignal(str, QPointF)
    MIN_ZOOM, MAX_ZOOM = 0.15, 4.0

    def __init__(self, scene: DiagramScene, parent=None):
        super().__init__(scene, parent)
        self.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.TextAntialiasing)
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.BoundingRectViewportUpdate)
        self.setAcceptDrops(True)
        self.setFrameShape(QGraphicsView.Shape.NoFrame)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._pan: QPointF | None = None
        self._space = False
        self.reserve_bottom = 0                     # altta üst katman (durum satırı) için ayrılan piksel
        self.setStyleSheet(f"QGraphicsView {{ background: {scene.tokens.surface}; border: none; }}")

    # ---------------------------------------------------------------- yakınlaştırma
    def zoom(self) -> float:
        return self.transform().m11()

    def zoom_by(self, factor: float) -> None:
        z = self.zoom()
        factor = min(max(factor, self.MIN_ZOOM / z), self.MAX_ZOOM / z)
        self.scale(factor, factor)

    def wheelEvent(self, event) -> None:  # noqa: N802
        delta = event.angleDelta().y()
        if delta == 0:
            return
        self.zoom_by(1.15 ** (delta / 120.0))
        event.accept()

    def fit_all(self) -> None:
        sc = self.scene()
        items = [i for i in sc.items() if isinstance(i, (BlockItem, WireItem))]
        if not items:
            self.resetTransform()
            self.centerOn(0, 0)
            return
        rect = QRectF()
        for i in items:
            rect = rect.united(i.sceneBoundingRect())
        rect = rect.adjusted(-40, -40, 40, 40)
        vh = self.viewport().height()
        if 0 < self.reserve_bottom < vh - 40:       # alt şeridi boş bırak
            rect = rect.adjusted(0, 0, 0, rect.height() * self.reserve_bottom / (vh - self.reserve_bottom))
        self.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)
        if self.zoom() > 1.5:
            self.resetTransform()
            self.scale(1.5, 1.5)
            self.centerOn(rect.center())

    # ---------------------------------------------------------------- kaydırma
    def _start_pan(self, pos) -> None:
        self._pan = QPointF(pos)
        self.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.MiddleButton or \
                (event.button() == Qt.MouseButton.LeftButton and self._space):
            self._start_pan(event.position())
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._pan is not None:
            d = event.position() - self._pan
            self._pan = QPointF(event.position())
            self.horizontalScrollBar().setValue(int(self.horizontalScrollBar().value() - d.x()))
            self.verticalScrollBar().setValue(int(self.verticalScrollBar().value() - d.y()))
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if self._pan is not None and event.button() in (Qt.MouseButton.MiddleButton, Qt.MouseButton.LeftButton):
            self._pan = None
            self.viewport().setCursor(Qt.CursorShape.OpenHandCursor if self._space else Qt.CursorShape.ArrowCursor)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self._space = True
            self.viewport().setCursor(Qt.CursorShape.OpenHandCursor)
            event.accept()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event) -> None:  # noqa: N802
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self._space = False
            self.viewport().setCursor(Qt.CursorShape.ArrowCursor)
            event.accept()
            return
        super().keyReleaseEvent(event)

    # ---------------------------------------------------------------- paletten bırakma
    def _accepts(self, event) -> bool:
        from spt.gui.control.diagram_palette import MIME_BLOCK

        return event.mimeData().hasFormat(MIME_BLOCK)

    def dragEnterEvent(self, event) -> None:  # noqa: N802
        if self._accepts(event):
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event) -> None:  # noqa: N802
        if self._accepts(event):
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event) -> None:  # noqa: N802
        from spt.gui.control.diagram_palette import MIME_BLOCK

        if self._accepts(event):
            type_name = bytes(event.mimeData().data(MIME_BLOCK)).decode("utf-8")
            self.block_dropped.emit(type_name, self.mapToScene(event.position().toPoint()))
            event.acceptProposedAction()
            return
        super().dropEvent(event)


__all__ = ["DiagramScene", "DiagramView", "BlockItem", "WireItem", "PortItem", "block_summary", "poly_text",
           "snap", "GRID", "MIME_CLIP"]
