"""Kuaterniyon / Euler yönelimini görselleştiren bileşenler (QPainter; OpenGL gerektirmez).

- `Orientation3D`: dünya çerçevesinde (x kuzey, y batı, z yukarı) bir uçak modeli. Seçili algoritmanın yönelimi dolu
  gövdeyle, referans ve diğer algoritmalar tel kafes "hayalet" olarak çizilir. Gövde eksenleri: x ileri (kırmızı),
  y sol (yeşil), z yukarı (mavi) — füzyon modülünün z-yukarı, sağ el kuralı tanımıyla aynı. Fareyle sürükleyerek
  kamera döndürülür, tekerlekle yakınlaştırılır, çift tıkla sıfırlanır.
- `AttitudeIndicator`: yapay ufuk (roll / pitch), `HeadingIndicator`: pusula (yaw) — Euler açılarının uçuş
  göstergeleriyle okunuşu.

Kuaterniyon q = [w, x, y, z] gövde → dünya döndürmesidir; çizim R(q)·p_gövde ile yapılır.

İşaret kuralı (z yukarı, y sol, ZYX): pozitif roll sağa yatış (sol kanat yukarı), pozitif PITCH BURUN AŞAĞI,
pozitif yaw kuzeyden batıya (sola) dönüş. Yapay ufuk fiziksel duruşu çizer (burun yukarı → ufuk aşağı kayar).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QBrush, QColor, QMouseEvent, QPainter, QPainterPath, QPen, QPolygonF, QWheelEvent
from PyQt6.QtWidgets import QSizePolicy, QWidget

from spt.gui.theme import ThemedColors, mono_font, themed, ui_font

# Gövde x, y, z (geleneksel RGB); açık temada zemine ≥ 3.5:1 kontrast için koyulaştırıldı
AXIS_COLORS = ThemedColors(dark=("#f25555", "#48d16b", "#4d8dfa"), light=("#d63a3a", "#1f9a4a", "#2f6fe0"))


def hull_color() -> str:
    """Dolu gövdenin nötr rengi: algoritma rengi kenarda; RGB gövde eksenleri her algoritmada seçilir."""
    return themed("#b9b2ab", "#a39b93")


def rotation(q) -> np.ndarray:
    w, x, y, z = (float(v) for v in q)
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def euler_deg(q) -> tuple[float, float, float]:
    w, x, y, z = (float(v) for v in q)
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


def quat_from_euler_deg(roll: float, pitch: float, yaw: float) -> np.ndarray:
    r, p, y = (math.radians(v) / 2 for v in (roll, pitch, yaw))
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
                     cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy])


def _box(x0, x1, y0, y1, z0, z1) -> list[np.ndarray]:
    """Kutunun 6 yüzü (dışa bakan normal sırasıyla köşeler)."""
    c = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                  [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]], dtype=float)
    faces = ((0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (2, 3, 7, 6), (1, 2, 6, 5), (0, 4, 7, 3))
    return [c[list(f)] for f in faces]


def _airplane() -> list[tuple[np.ndarray, str]]:
    """Gövde çerçevesinde basit uçak: (yüz köşeleri, rol). Rol gölgelemede kullanılır (üst yüzler daha açık)."""
    parts: list[tuple[np.ndarray, str]] = []
    for face in _box(-0.95, 0.75, -0.11, 0.11, -0.10, 0.10):          # gövde
        parts.append((face, "body"))
    nose_base = np.array([[0.75, -0.11, -0.10], [0.75, 0.11, -0.10], [0.75, 0.11, 0.10], [0.75, -0.11, 0.10]])
    tip = np.array([1.15, 0.0, 0.0])
    for i in range(4):                                                 # burun (piramit) — ileri yön belirgin
        parts.append((np.array([nose_base[i], nose_base[(i + 1) % 4], tip]), "nose"))
    for face in _box(-0.25, 0.22, -1.15, 1.15, -0.02, 0.02):          # kanat
        parts.append((face, "wing"))
    for face in _box(-0.95, -0.72, -0.45, 0.45, -0.015, 0.015):       # yatay kuyruk
        parts.append((face, "wing"))
    fin = np.array([[-0.95, 0.0, 0.10], [-0.72, 0.0, 0.10], [-0.88, 0.0, 0.50], [-0.98, 0.0, 0.50]])
    parts.append((fin, "fin"))                                         # dikey kuyruk (yukarı yön belirgin)
    parts.append((fin[::-1].copy(), "fin"))
    return parts


MODEL = _airplane()


@dataclass
class Body:
    key: str
    label: str
    color: str
    q: np.ndarray            # [w, x, y, z]
    solid: bool = False      # dolu gövde (seçili) ya da tel kafes hayalet


class Orientation3D(QWidget):
    def __init__(self, tokens, parent: QWidget | None = None):
        super().__init__(parent)
        self._t = tokens
        self.bodies: list[Body] = []
        self.overlay: list[str] = []
        self.warning = ""
        self.gauges: tuple[float, float, float] | None = None  # (roll, pitch, yaw) ° — alt köşelerde göstergeler
        self.compact: list[str] = []  # küçük boyutta gösterilecek kısa bilgi satırları
        self.reset_camera()
        self._drag: QPointF | None = None
        self.setMinimumSize(220, 200)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(False)
        self.setToolTip("Sürükle: kamerayı döndür · Tekerlek: yakınlaştır · Çift tık: varsayılan görünüm")

    # ---------------------------------------------------------------- dış arayüz
    def reset_camera(self) -> None:
        self.azimuth, self.elevation, self.zoom = -55.0, 22.0, 1.0
        self.update()

    def set_scene(self, bodies: list[Body], overlay: list[str] | None = None, warning: str = "",
                  gauges: tuple[float, float, float] | None = None, compact: list[str] | None = None) -> None:
        """overlay: tam bilgi satırları; compact: küçük pencerede (yükseklik < 320 px) bunun yerine gösterilenler."""
        self.bodies = bodies
        self.overlay = overlay or []
        self.warning = warning
        self.gauges = gauges
        self.compact = compact or []
        self.update()

    # ---------------------------------------------------------------- kamera
    def _basis(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        a, e = math.radians(self.azimuth), math.radians(self.elevation)
        c = np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])   # kameraya doğru
        r = np.array([-math.sin(a), math.cos(a), 0.0])                                        # ekran sağı
        u = np.array([-math.sin(e) * math.cos(a), -math.sin(e) * math.sin(a), math.cos(e)])   # ekran yukarısı
        return r, u, c

    def _project(self, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Dünya noktaları (n, 3) → ekran (n, 2) ve derinlik (büyük = yakın). Hafif perspektif."""
        r, u, c = self._basis()
        depth = pts @ c
        persp = 6.0 / (6.0 - np.clip(depth, -5.0, 5.0))
        scale = 0.36 * min(self.width(), self.height()) * self.zoom
        sx = self.width() / 2 + (pts @ r) * persp * scale
        sy = self.height() / 2 - (pts @ u) * persp * scale
        return np.column_stack((sx, sy)), depth

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag = event.position()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._drag is None:
            return
        d = event.position() - self._drag
        self._drag = event.position()
        self.azimuth = (self.azimuth - d.x() * 0.5 + 180) % 360 - 180
        self.elevation = max(-89.0, min(89.0, self.elevation + d.y() * 0.5))
        self.update()

    def mouseReleaseEvent(self, _event: QMouseEvent) -> None:
        self._drag = None

    def mouseDoubleClickEvent(self, _event: QMouseEvent) -> None:
        self.reset_camera()

    def wheelEvent(self, event: QWheelEvent) -> None:
        self.zoom = max(0.4, min(3.0, self.zoom * (1.0 + event.angleDelta().y() / 1200)))
        self.update()

    # ---------------------------------------------------------------- çizim
    def paintEvent(self, _event) -> None:
        t = self._t
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor(t.surface))
        self._draw_ground(p)
        self._draw_world_axes(p)
        faces = []  # (derinlik, poligon, renk)
        light = np.array([0.35, -0.45, 0.82])
        light /= np.linalg.norm(light)
        for body in self.bodies:
            R = rotation(body.q)
            if body.solid:
                base = QColor(hull_color())
                tint = QColor(body.color)
                for verts, role in MODEL:
                    world = verts @ R.T
                    scr, depth = self._project(world)
                    normal = np.cross(world[1] - world[0], world[2] - world[0])
                    norm = np.linalg.norm(normal)
                    normal = normal / norm if norm > 0 else normal
                    _, _, cam = self._basis()
                    if float(normal @ cam) < 0 and role != "fin":
                        continue  # arka yüz
                    shade = 0.45 + 0.55 * max(0.0, float(normal @ light))
                    color = QColor(base)
                    if role == "body":
                        color = color.darker(125)
                    elif role == "nose":
                        color = QColor(tint)            # burun algoritmanın renginde: hangi kestirim olduğu belli
                    elif role == "fin":
                        color = QColor(AXIS_COLORS[2])  # dikey kuyruk gövde +z yönünde: "yukarı" belli
                    elif role == "wing":
                        color = QColor.fromRgbF(0.65 * base.redF() + 0.35 * tint.redF(),
                                                0.65 * base.greenF() + 0.35 * tint.greenF(),
                                                0.65 * base.blueF() + 0.35 * tint.blueF())
                    # gövdenin üst yüzleri (gövde +z) biraz açık: ters dönünce fark edilir
                    up = float((verts.mean(axis=0) @ np.array([0, 0, 1.0])))
                    factor = (1.15 if up > 0.0 else 0.8) * shade
                    color = QColor.fromHsvF(color.hsvHueF() if color.hsvHueF() >= 0 else 0.0, color.hsvSaturationF(),
                                            max(0.0, min(1.0, color.valueF() * factor)))
                    faces.append((float(depth.mean()), scr, color, tint))
            else:
                self._draw_ghost(p, body, R)
        faces.sort(key=lambda f: f[0])
        for _, scr, color, tint in faces:
            edge = QColor(tint)
            edge.setAlpha(150)
            p.setPen(QPen(edge, 1))
            p.setBrush(QBrush(color))
            p.drawPolygon(QPolygonF([QPointF(float(x), float(y)) for x, y in scr]))
        for body in self.bodies:
            if body.solid:
                self._draw_body_axes(p, rotation(body.q), 2.4)
        self._draw_overlay(p)
        if self.gauges is not None:  # Euler göstergeleri: alt köşelerde, boyuta göre ölçekli
            g = max(70.0, min(150.0, 0.30 * min(self.width(), self.height())))
            bottom = self.height() - g - 16 - (30 if self.warning else 0)
            paint_attitude(p, QRectF(6, bottom, g, g), self.gauges[0], self.gauges[1], self._t, caption=False)
            paint_heading(p, QRectF(self.width() - g - 6, bottom, g, g), self.gauges[2], self._t, caption=False)
            p.setPen(QColor(self._t.text))
            p.setFont(ui_font(11))
            text_y = bottom + g - 2
            p.drawText(QRectF(6, text_y, self.width() / 2, 16), Qt.AlignmentFlag.AlignLeft,
                       f"R {self.gauges[0]:+.1f}°  P {self.gauges[1]:+.1f}°")
            p.drawText(QRectF(self.width() / 2, text_y, self.width() / 2 - 6, 16), Qt.AlignmentFlag.AlignRight,
                       f"Y {self.gauges[2]:+.1f}°")
        p.end()

    def _draw_ground(self, p: QPainter) -> None:
        pen = QPen(QColor(self._t.grid if hasattr(self._t, "grid") else "#222"), 1)
        p.setPen(pen)
        z = -0.9
        for v in np.linspace(-1.5, 1.5, 7):
            for a, b in (((v, -1.5, z), (v, 1.5, z)), ((-1.5, v, z), (1.5, v, z))):
                scr, _ = self._project(np.array([a, b], dtype=float))
                p.drawLine(QPointF(*scr[0]), QPointF(*scr[1]))

    def _draw_world_axes(self, p: QPainter) -> None:
        t = self._t
        p.setFont(ui_font(11))
        pen = QPen(QColor(t.text_muted), 1, Qt.PenStyle.DashLine)
        z = -0.9
        labels = (((1.7, 0, z), "K"), ((-1.7, 0, z), "G"), ((0, 1.7, z), "B"), ((0, -1.7, z), "D"))
        origin, _ = self._project(np.array([[0, 0, z]], dtype=float))
        for (pt, text) in labels:
            scr, _ = self._project(np.array([pt], dtype=float))
            p.setPen(pen)
            p.drawLine(QPointF(*origin[0]), QPointF(*scr[0]))
            p.setPen(QColor(t.text_secondary))
            p.drawText(QPointF(scr[0][0] - 4, scr[0][1] + 4), text)

    def _draw_ghost(self, p: QPainter, body: Body, R: np.ndarray) -> None:
        color = QColor(body.color)
        color.setAlpha(170)
        pen = QPen(color, 1.2, Qt.PenStyle.DashLine if body.key == "ref" else Qt.PenStyle.SolidLine)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        for verts, role in MODEL:
            if role == "body":
                continue
            scr, _ = self._project(verts @ R.T)
            p.drawPolygon(QPolygonF([QPointF(float(x), float(y)) for x, y in scr]))
        # gövde ekseni (burun yönü) belirgin
        scr, _ = self._project(np.array([[-0.95, 0, 0], [1.15, 0, 0]]) @ R.T)
        p.setPen(QPen(color, 2))
        p.drawLine(QPointF(*scr[0]), QPointF(*scr[1]))

    def _draw_body_axes(self, p: QPainter, R: np.ndarray, width: float) -> None:
        names = ("x", "y", "z")
        p.setFont(ui_font(11, ui_font().Weight.Bold))
        origin, _ = self._project(np.zeros((1, 3)))
        for i in range(3):
            tip = np.zeros(3)
            tip[i] = 1.55 if i == 0 else 1.35 if i == 1 else 0.95
            scr, _ = self._project((tip @ R.T)[None, :])
            color = QColor(AXIS_COLORS[i])
            p.setPen(QPen(color, width))
            p.drawLine(QPointF(*origin[0]), QPointF(*scr[0]))
            p.setPen(color)
            p.drawText(QPointF(scr[0][0] + 4, scr[0][1] - 4), names[i])

    def _draw_overlay(self, p: QPainter) -> None:
        t = self._t
        p.setFont(mono_font(10 if self.width() < 420 else 11))
        lines = self.compact if self.compact and self.height() < 320 else self.overlay
        if lines:  # yarı saydam zemin: sayılar modelin üstünde de okunur
            metrics = p.fontMetrics()
            width = max(metrics.horizontalAdvance(line) for line in lines) + 26
            back = QColor(t.surface)
            back.setAlpha(205)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(back)
            p.drawRoundedRect(QRectF(4, 3, min(width, self.width() - 8), 15 * len(lines) + 6), 6, 6)
        y = 16
        for line in lines:
            if line.startswith("●"):  # renk işaretli satır: "●#rrggbb metin"
                color, text = line[1:8], line[8:]
                p.setPen(QColor(color))
                p.drawText(QPointF(10, y), "●")
                p.setPen(QColor(t.text_secondary))
                p.drawText(QPointF(24, y), text)
            else:
                p.setPen(QColor(t.text_secondary))
                p.drawText(QPointF(10, y), line)
            y += 15
        if self.warning:
            p.setPen(QColor(t.warning))
            p.setFont(ui_font(11))
            p.drawText(QRectF(10, self.height() - 34, self.width() - 20, 30),
                       Qt.AlignmentFlag.AlignLeft | Qt.TextFlag.TextWordWrap, self.warning)


# ============================================================================ Euler göstergeleri
def paint_attitude(p: QPainter, rect: QRectF, roll: float, pitch: float, t, caption: bool = True) -> None:
    """Yapay ufuk çizimi (bileşen ve 3B köşe göstergesi ortak). Bu projenin kuralında pozitif pitch burun aşağıdır;
    çizim fiziksel duruşu gösterir (burun yukarı = −pitch → ufuk aşağı kayar)."""
    text_h = 14 if caption else 0
    side = min(rect.width(), rect.height() - text_h) - 4
    if side < 20:
        return
    r = side / 2
    cx, cy = rect.center().x(), rect.top() + 2 + r
    p.save()
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    clip = QPainterPath()
    clip.addEllipse(QPointF(cx, cy), r, r)
    p.setClipPath(clip)
    p.translate(cx, cy)
    p.rotate(-roll)
    px_per_deg = r / 45.0
    shift = -pitch * px_per_deg
    p.fillRect(QRectF(-2 * r, -3 * r + shift, 4 * r, 3 * r), QColor("#2f5d8a"))   # gök
    p.fillRect(QRectF(-2 * r, shift, 4 * r, 3 * r), QColor("#6b4a2b"))            # yer
    p.setPen(QPen(QColor("#f0ece8"), 1.5))
    p.drawLine(QPointF(-2 * r, shift), QPointF(2 * r, shift))
    p.setFont(ui_font(9))
    for deg in range(-60, 61, 10):
        if deg == 0:
            continue
        y = shift - deg * px_per_deg
        w = r * (0.35 if deg % 20 == 0 else 0.18)
        p.drawLine(QPointF(-w, y), QPointF(w, y))
        if deg % 20 == 0 and r > 40:
            p.drawText(QPointF(w + 3, y + 3), f"{abs(deg)}")
    p.restore()
    p.save()
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setPen(QPen(QColor(t.accent), 3))  # sabit uçak simgesi
    p.drawLine(QPointF(cx - r * 0.5, cy), QPointF(cx - r * 0.15, cy))
    p.drawLine(QPointF(cx + r * 0.15, cy), QPointF(cx + r * 0.5, cy))
    p.drawPoint(QPointF(cx, cy))
    p.setPen(QPen(QColor(t.border_strong), 2))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawEllipse(QPointF(cx, cy), r, r)
    if caption:
        p.setPen(QColor(t.text))
        p.setFont(ui_font(11))
        p.drawText(QRectF(rect.left(), cy + r + 1, rect.width(), 14), Qt.AlignmentFlag.AlignCenter,
                   f"R {roll:+.1f}°  P {pitch:+.1f}°")
    p.restore()


def paint_heading(p: QPainter, rect: QRectF, yaw: float, t, caption: bool = True) -> None:
    """Pusula çizimi: kadran, burun yukarıyı gösterecek şekilde döner (yaw kuzeyden batıya doğru artar)."""
    text_h = 14 if caption else 0
    side = min(rect.width(), rect.height() - text_h) - 4
    if side < 20:
        return
    r = side / 2
    cx, cy = rect.center().x(), rect.top() + 2 + r
    p.save()
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setPen(QPen(QColor(t.border_strong), 2))
    p.setBrush(QColor(t.surface_alt))
    p.drawEllipse(QPointF(cx, cy), r, r)
    p.translate(cx, cy)
    p.rotate(yaw)
    p.setFont(ui_font(11 if r > 40 else 9, ui_font().Weight.Bold))
    for deg in range(0, 360, 30):
        p.save()
        p.rotate(-deg)  # dünya +y batı: kuzeyden saat yönünün tersine
        p.setPen(QPen(QColor(t.text_secondary), 1.2))
        p.drawLine(QPointF(0, -r + 2), QPointF(0, -r + (9 if deg % 90 == 0 else 5)))
        if deg % 90 == 0:
            text = {0: "K", 90: "B", 180: "G", 270: "D"}[deg]
            p.setPen(QColor(t.accent if deg == 0 else t.text))
            p.drawText(QRectF(-8, -r + 9, 16, 14), Qt.AlignmentFlag.AlignCenter, text)
        p.restore()
    p.restore()
    p.save()
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setPen(QPen(QColor(t.accent), 3))
    p.drawLine(QPointF(cx, cy + r * 0.35), QPointF(cx, cy - r * 0.55))
    p.drawLine(QPointF(cx - r * 0.25, cy), QPointF(cx + r * 0.25, cy))
    if caption:
        p.setPen(QColor(t.text))
        p.setFont(ui_font(11))
        p.drawText(QRectF(rect.left(), cy + r + 1, rect.width(), 14), Qt.AlignmentFlag.AlignCenter, f"Y {yaw:+.1f}°")
    p.restore()


class AttitudeIndicator(QWidget):
    """Yapay ufuk: gökyüzü / yer, roll'a göre döner, burun açısına göre kayar (10°'lik merdiven)."""

    def __init__(self, tokens, parent: QWidget | None = None):
        super().__init__(parent)
        self._t = tokens
        self.roll = self.pitch = 0.0
        self.setMinimumSize(110, 110)
        self.setToolTip("Yapay ufuk: roll (yatış) ve pitch (yunuslama). z-yukarı / y-sol kuralında pozitif roll sağa "
                        "yatış, pozitif pitch burun aşağıdır; gösterge fiziksel duruşu çizer.")

    def set_angles(self, roll: float, pitch: float) -> None:
        self.roll, self.pitch = roll, pitch
        self.update()

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        paint_attitude(p, QRectF(self.rect()), self.roll, self.pitch, self._t)
        p.end()


class HeadingIndicator(QWidget):
    """Pusula: yaw (manyetik kuzeyden, dünya +x) — sağ el kuralında yaw saat yönünün tersine artar."""

    def __init__(self, tokens, parent: QWidget | None = None):
        super().__init__(parent)
        self._t = tokens
        self.yaw = 0.0
        self.setMinimumSize(110, 110)
        self.setToolTip("Pusula: yaw (sapma / başlık). z-yukarı sağ el kuralında yaw kuzeyden batıya doğru artar.")

    def set_yaw(self, yaw: float) -> None:
        self.yaw = yaw
        self.update()

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        paint_heading(p, QRectF(self.rect()), self.yaw, self._t)
        p.end()
