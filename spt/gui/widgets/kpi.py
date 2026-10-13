"""Canlı İzleme'nin üstündeki KPI kutuları: paket hızı, veri hızı, paket sayısı, hatalı paket ve kayıp.

Alıcının istatistik olaylarıyla (≈ 4 Hz) beslenir; değerler durum çubuğuyla aynı kaynaktan gelir (akış başına hızların
toplamı, oturum sayaçları). Hız kutularında son ≈ 30 saniyenin küçük çizgi grafiği vardır; çizim pyqtgraph'sız,
doğrudan QPainter ile yapılır (kutu başına birkaç yüz nokta, ihmal edilebilir maliyet).
"""
from __future__ import annotations

from collections import deque

from PyQt6.QtCore import QPointF, QRectF, QSize, Qt
from PyQt6.QtGui import QColor, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import QFrame, QGridLayout, QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget

from spt.gui.icons import icon
from spt.gui.widgets.common import StatusBadge, label

SPARK_POINTS = 120          # ≈ 30 s (istatistik olayları ≈ 4 Hz)
TILE_MIN_WIDTH = 118        # bunun altında kutular iki satıra kırılır
BADGE_TEXT_TILE = 190       # bundan dar kutuda rozet yalnız renkli nokta (metni ipucunda)
SPARK_MIN_TILE = 210        # çizgi grafik bu genişlikten dar kutuda gizlenir


def _group(n: float) -> str:
    """Binlik ayraçlı tam sayı (dar boşluk)."""
    return f"{n:,.0f}".replace(",", " ")


def _rate_parts(bytes_per_s: float) -> tuple[str, str]:
    if bytes_per_s >= 1e6:
        return f"{bytes_per_s / 1e6:.2f}", "MB/s"
    if bytes_per_s >= 1e3:
        return f"{bytes_per_s / 1e3:.1f}", "kB/s"
    return f"{bytes_per_s:.0f}", "B/s"


class Sparkline(QWidget):
    """Son değerlerin küçük çizgi grafiği (alt kısmı soluk dolgulu); eksen yok, ölçek kendi en büyüğüne göre."""

    def __init__(self, color: str, parent: QWidget | None = None):
        super().__init__(parent)
        self._values: deque[float] = deque(maxlen=SPARK_POINTS)
        self._color = QColor(color)
        self.setMinimumSize(60, 26)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setFixedHeight(30)

    def set_color(self, color: str) -> None:
        self._color = QColor(color)
        self.update()

    def push(self, value: float) -> None:
        self._values.append(max(float(value), 0.0))
        self.update()

    def clear(self) -> None:
        self._values.clear()
        self.update()

    def values(self) -> list[float]:
        return list(self._values)

    def paintEvent(self, event) -> None:  # noqa: N802
        if len(self._values) < 2:
            return
        top = max(self._values) or 1.0
        w, h = self.width() - 2, self.height() - 3
        step = w / (SPARK_POINTS - 1)
        x0 = 1 + w - step * (len(self._values) - 1)   # en yeni değer sağda
        points = [QPointF(x0 + i * step, 1.5 + h * (1 - v / top)) for i, v in enumerate(self._values)]
        line = QPainterPath(points[0])
        for pt in points[1:]:
            line.lineTo(pt)
        area = QPainterPath(line)
        area.lineTo(points[-1].x(), h + 1.5)
        area.lineTo(points[0].x(), h + 1.5)
        area.closeSubpath()
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        fill = QColor(self._color)
        fill.setAlpha(40)
        p.fillPath(area, fill)
        pen = QPen(self._color, 1.5)
        pen.setCosmetic(True)
        p.setPen(pen)
        p.drawPath(line)
        p.end()


class KpiTile(QFrame):
    """Tek KPI kutusu: simge + başlık, büyük değer + birim, alt satır; isteğe bağlı rozet ve çizgi grafik."""

    def __init__(self, icon_name: str, title: str, unit: str = "", tip: str = "", spark_color: str | None = None,
                 badge: bool = False):
        super().__init__()
        self.setObjectName("KpiTile")
        self.setMinimumWidth(TILE_MIN_WIDTH)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setToolTip(tip)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 8, 12, 8)
        lay.setSpacing(2)
        head = QHBoxLayout()
        head.setSpacing(6)
        ico = QLabel()
        ico.setPixmap(icon(icon_name).pixmap(14, 14))
        head.addWidget(ico)
        head.addWidget(label(title, "KpiTitle"))
        head.addStretch(1)
        lay.addLayout(head)
        row = QHBoxLayout()
        row.setSpacing(8)
        self.value = label("–", "KpiValue")
        self.unit = label(unit, "KpiUnit")
        row.addWidget(self.value, 0, Qt.AlignmentFlag.AlignBottom)
        row.addWidget(self.unit, 0, Qt.AlignmentFlag.AlignBottom)
        row.addStretch(1)
        # Rozet değer satırının sağında: başlık satırında dar kutuda başlığı ve kendisini kesiyordu
        self.badge = StatusBadge("", "idle") if badge else None
        if self.badge is not None:
            row.addWidget(self.badge, 0, Qt.AlignmentFlag.AlignBottom)
        self.spark = Sparkline(spark_color) if spark_color else None
        if self.spark is not None:
            self.spark.setFixedWidth(84)
            row.addWidget(self.spark, 0, Qt.AlignmentFlag.AlignBottom)
        lay.addLayout(row)
        self.sub = label("", "KpiSub")
        lay.addWidget(self.sub)
        self.compact = False

    def set_badge(self, text: str, state: str) -> None:
        """Rozet; dar kutuda yalnız durum noktası, metin ipucunda (kesik "Hat…" yerine)."""
        self._badge_state = (text, state)
        self.badge.set_state("" if self.compact else text, state)
        self.badge.setToolTip(text)

    def set_compact(self, compact: bool) -> None:
        if compact != self.compact:
            self.compact = compact
            if self.badge is not None and hasattr(self, "_badge_state"):
                self.set_badge(*self._badge_state)

    def set(self, value: str, unit: str | None = None, sub: str | None = None) -> None:
        if self.value.text() != value:
            self.value.setText(value)
        if unit is not None and self.unit.text() != unit:
            self.unit.setText(unit)
        if sub is not None and self.sub.text() != sub:
            self.sub.setText(sub)


class KpiStrip(QWidget):
    """Beş KPI kutusu yan yana (dar alanda iki satıra kırılır). `update_stats(session, rates)` ile beslenir."""

    def __init__(self, accent: str):
        super().__init__()
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self._grid = QGridLayout(self)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setHorizontalSpacing(10)
        self._grid.setVerticalSpacing(8)
        self.pps = KpiTile("ph.broadcast", "Paket hızı", "paket/s",
                           "Saniyede alınan DATA paketi (tüm akışlar, son ≈ 0.25 s); grafik son ≈ 30 s", accent)
        self.rate = KpiTile("ph.download-simple", "Veri hızı", "kB/s",
                            "Örnek verisi (başlıklar hariç), tüm akışlar; alt satırda Mbit/s", accent)
        self.count = KpiTile("ph.stack", "Paket sayısı", "paket",
                             "Bu oturumda alınan DATA paketi (tüm akışlar); alt satırda toplam datagram ve bayt")
        # Rozetli kutularda birim başlıkta: değer satırında rozete yer kalsın (dar kutuda "Kayıpsız" kesiliyordu)
        self.errors = KpiTile("ph.warning-circle", "Hatalı paket", "",
                              "Doğrulamadan geçemeyen çerçeve (CRC, boyut, biçim) ve tanımlayıcısı gelmemiş akışa ait "
                              "çerçeve (protokol gereği atılır)", badge=True)
        self.loss = KpiTile("ph.link-break", "Kayıp (örnek)", "",
                            "Sıra numarasıyla (first_sample) tespit edilen eksik örnekler; oran ve kayıp olayı sayısı",
                            badge=True)
        self.tiles = [self.pps, self.rate, self.count, self.errors, self.loss]
        self._columns = 0
        self._relayout(len(self.tiles))
        self.clear()

    def set_accent(self, color: str) -> None:
        for tile in (self.pps, self.rate):
            tile.spark.set_color(color)

    def clear(self) -> None:
        self.pps.set("0", sub="")
        self.rate.set("0", "B/s", "0 Mbit/s")
        self.count.set("0", sub="0 datagram")
        self.errors.set("0", sub="CRC/biçim 0 · tanımsız 0")
        self.errors.set_badge("Bekliyor", "idle")
        self.loss.set("0", sub="% 0")
        self.loss.set_badge("Bekliyor", "idle")
        for tile in (self.pps, self.rate):
            tile.spark.clear()

    def update_stats(self, session: dict, rates: dict) -> None:
        stats = [r.stats for r in rates.values() if r.stats is not None]
        pps = sum(r.packets_per_s for r in rates.values())
        bps = sum(r.bytes_per_s for r in rates.values())
        self.pps.set(_group(pps))
        self.pps.spark.push(pps)
        value, unit = _rate_parts(bps)
        self.rate.set(value, unit, f"{bps * 8 / 1e6:.2f} Mbit/s")
        self.rate.spark.push(bps)
        packets = sum(st.packets for st in stats)
        unit = "dgram" if session.get("transport", "udp") == "udp" else "çerçeve"
        self.count.set(_group(packets), sub=f"{_group(session.get('datagrams', 0))} {unit} · "
                                            f"{session.get('bytes', 0) / 1e6:.1f} MB")
        frame_errors, no_desc = int(session.get("frame_errors", 0)), int(session.get("no_descriptor", 0))
        bad = frame_errors + no_desc
        self.errors.set(_group(bad), sub=f"CRC/biçim {_group(frame_errors)} · tanımsız {_group(no_desc)}")
        lost = sum(st.lost_samples for st in stats)
        total = sum(st.samples for st in stats) + lost
        events = sum(getattr(st, "loss_events", 0) for st in stats)
        ratio = 100.0 * lost / total if total else 0.0
        self.loss.set(_group(lost), sub=f"% {ratio:.3g} · {_group(events)} olay" if lost else "% 0")
        if not stats:
            self.errors.set_badge("Bekliyor", "idle")
            self.loss.set_badge("Bekliyor", "idle")
            return
        self.errors.set_badge("Hatalı" if frame_errors else ("Atlanan" if no_desc else "Hatasız"),
                              "critical" if frame_errors else ("warning" if no_desc else "good"))
        self.loss.set_badge("Kayıplı" if lost else "Kayıpsız", "warning" if lost else "good")

    # ---------------------------------------------------------------- yerleşim
    def resizeEvent(self, event) -> None:
        """Tek satır korunur (dar ekranda dikey yer değerli): kutu daralınca önce çizgi grafikler gizlenir, kutular
        en dar boyun altına inince iki satıra (3 + 2) kırılır."""
        super().resizeEvent(event)
        n, spacing = len(self.tiles), self._grid.horizontalSpacing()
        per = (self.width() - spacing * (n - 1)) / n
        columns = n if per >= TILE_MIN_WIDTH else 3
        if columns != self._columns:
            self._relayout(columns)
        tile_width = (self.width() - spacing * (columns - 1)) / columns
        show = tile_width >= SPARK_MIN_TILE
        for tile in (self.pps, self.rate):
            if tile.spark.isVisibleTo(tile) != show:
                tile.spark.setVisible(show)
        for tile in (self.errors, self.loss):
            tile.set_compact(tile_width < BADGE_TEXT_TILE)

    def _relayout(self, columns: int) -> None:
        self._columns = columns
        for tile in self.tiles:
            self._grid.removeWidget(tile)
        for i, tile in enumerate(self.tiles):
            self._grid.addWidget(tile, i // columns, i % columns)
        for c in range(len(self.tiles)):
            self._grid.setColumnStretch(c, 1 if c < columns else 0)

    def sizeHint(self) -> QSize:  # noqa: N802
        return QSize(super().sizeHint().width(), self._grid.sizeHint().height())

    def tile_rects(self) -> list[QRectF]:
        """Testler için: kutuların yerleşimi (satır kırılması doğrulanır)."""
        return [QRectF(t.geometry()) for t in self.tiles]
