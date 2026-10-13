"""Canlı İzleme'nin yanındaki "Bağlantı" paneli: akış başına anlık paket / zamanlama istatistikleri, oturum
sayaçları ve verinin geldiği PC ağ arayüzü (ad, tür, bağlantı hızı, MTU, IP) ile hat kullanımı.

Değerler alıcının ≈ 4 Hz istatistik olayından (SourceManager.stats_updated) gelir; ağ arayüzü bilgisi
(GetAdaptersAddresses) yalnız panel görünürken 2 s'de bir yenilenir.
"""
from __future__ import annotations

import math
import time

from PyQt6.QtCore import QSize, Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from spt.gui.theme import tokens
from spt.gui.widgets.common import StatusBadge, combo, icon_button, label
from spt.sources import netinfo
from spt.sources.link_report import fmt_ppm, fmt_us

NET_REFRESH_MS = 2000
LOSS_HOLD_S = 2.0          # kayıp görüldükten sonra "Kayıplı" durumunun kalma süresi (StatsPanel ile aynı)
IDLE_AFTER_S = 1.0
PANEL_WIDTH = 300         # varsayılan genişlik (bölücüyle değiştirilir)
# DATA çerçevesi başına hatta giden ek bayt: SPTF başlığı + DATA alt başlığı (48) + UDP / IPv4 (28) + Ethernet
# başlığı, FCS, önsöz ve çerçeveler arası boşluk (38). Datagram başına tek çerçeve varsayılır (yaklaşık).
WIRE_OVERHEAD = 48 + 28 + 38


def fmt_count(n: float) -> str:
    """Binlik ayraçlı tamsayı (dar boşluk)."""
    return f"{n:,.0f}".replace(",", " ")


def fmt_bits(bytes_per_s: float) -> str:
    """Veri hızı: Mbit/s (ağ birimi) ve MB/s birlikte."""
    bits = bytes_per_s * 8
    if bits >= 1e6:
        main = f"{bits / 1e6:.2f} Mbit/s"
    elif bits >= 1e3:
        main = f"{bits / 1e3:.1f} kbit/s"
    else:
        main = f"{bits:.0f} bit/s"
    return f"{main} · {bytes_per_s / 1e6:.3f} MB/s" if bytes_per_s >= 1e4 else main


def fmt_bytes(n: float) -> str:
    for unit, scale in (("GB", 1e9), ("MB", 1e6), ("kB", 1e3)):
        if n >= scale:
            return f"{n / scale:.2f} {unit}"
    return f"{n:.0f} B"


def fmt_ago(seconds: float) -> str:
    if not math.isfinite(seconds):
        return "yok"
    if seconds < 60:
        return f"{seconds:.1f} s önce"
    return f"{seconds / 60:.0f} dk önce" if seconds < 3600 else f"{seconds / 3600:.1f} sa önce"


class _Section(QWidget):
    """Başlık + (ad, değer) satırları. Değer etiketleri anahtarla güncellenir."""

    def __init__(self, title: str, rows: list[tuple[str, str, str]]):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        lay.addWidget(label(title.upper(), "SectionLabel"))
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(3)
        grid.setColumnStretch(1, 1)
        self.values: dict[str, QLabel] = {}
        self.names: dict[str, QLabel] = {}
        for i, (key, name, tip) in enumerate(rows):
            name_lbl = label(name, "FormLabel")
            self.names[key] = name_lbl
            name_lbl.setToolTip(tip)
            value = label("—")
            value.setToolTip(tip)
            value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            value.setWordWrap(True)
            value.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
            grid.addWidget(name_lbl, i, 0, Qt.AlignmentFlag.AlignTop)
            grid.addWidget(value, i, 1)
            self.values[key] = value
        lay.addLayout(grid)

    def set(self, key: str, text: str, color: str | None = None, tip: str | None = None) -> None:
        lbl = self.values[key]
        html = f'<span style="color:{color}">{text}</span>' if color else text
        if lbl.text() != html:
            lbl.setText(html)
        if tip is not None:
            lbl.setToolTip(tip)

    def reset(self) -> None:
        for lbl in self.values.values():
            lbl.setText("—")

    def show_row(self, key: str, visible: bool) -> None:
        self.names[key].setVisible(visible)
        self.values[key].setVisible(visible)


class LinkStatsPanel(QFrame):
    """Kompakt bağlantı paneli. `update_stats(session, rates)` ile beslenir."""

    link_test_requested = pyqtSignal(str)   # seçili akışın anahtarı ("" = akış yok)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("Card")
        self.setMinimumWidth(PANEL_WIDTH - 50)
        self.setMaximumWidth(PANEL_WIDTH + 160)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(14, 12, 8, 12)
        outer.setSpacing(8)

        title = label("Bağlantı", "CardTitle")
        self.test_button = icon_button("ph.gauge", "UDP link testi: hız, titreşim, kayıp, RTT (Araçlar menüsünde de)",
                                       object_name="FlatIcon", size=16)
        self.test_button.clicked.connect(lambda: self.link_test_requested.emit(self.current_key()))
        top = QHBoxLayout()
        top.setSpacing(6)
        top.addWidget(title)
        top.addStretch(1)
        top.addWidget(self.test_button)
        outer.addLayout(top)
        self.badge = StatusBadge("Akış yok", "idle")
        self.stream_combo = combo([])
        self.stream_combo.setToolTip("İstatistikleri gösterilen akış")
        self.stream_combo.currentIndexChanged.connect(lambda *_: self._render())
        self.stream_combo.hide()
        head = QVBoxLayout()
        head.setSpacing(6)
        head.addWidget(self.badge, 0, Qt.AlignmentFlag.AlignLeft)
        head.addWidget(self.stream_combo)
        outer.addLayout(head)

        body = QWidget()
        col = QVBoxLayout(body)
        col.setContentsMargins(0, 0, 6, 0)
        col.setSpacing(14)
        self.stream = _Section("Akış", [
            ("source", "Kaynak", "Gönderici: IP adresi ve UDP kaynak portu, USB portu ya da CAN kimliği"),
            ("conflict", "Çakışma", "Aynı IP ve akış kimliğiyle (stream_id) ikinci bir gönderici: başka kaynak "
                                    "portu ya da aynı porttan karışık first_sample dizisi. Örnekler iki sinyal "
                                    "karışmış gibi görünür; kartta stream_id'leri ayırın ya da simülatörü durdurun."),
            ("pps", "Paket hızı", "Saniyedeki DATA çerçevesi (son ≈ 0.25 s)"),
            ("rate", "Veri hızı", "Örnek verisi (başlıklar hariç), son ≈ 0.25 s"),
            ("loss", "Kayıp", "Kaybolan örnek ve oranı (oturum); kayıp olayı sayısı ve en uzun kayıp"),
            ("last_loss", "Son kayıp", "En son kayıp ne zaman görüldü"),
            ("dup", "Sıra", "Yinelenen çerçeve · sıra dışı (geç gelen; zamanında gelmediği için kayıp sayılan, "
                            "çizilmeyen) çerçeve · cihaz yeniden başlama (reset)"),
        ])
        self.timing = _Section("Zamanlama", [
            ("interval", "Paket aralığı", "Ardışık DATA çerçeveleri arası süre (alıcı saati), son ≈ 0.25 s "
                                          "ortalaması · beklenen = çerçevedeki örnek sayısı / fs"),
            ("jitter", "Titreşim", "Paket arası sürenin standart sapması ve p99'u, son ≈ 0.25 s"),
            ("gap", "En büyük boşluk", "Son ≈ 0.25 s'deki ve oturumdaki en uzun paket arası süre"),
            ("fs", "Etkin fs", "first_sample ilerlemesi / PC saatiyle geçen süre ve nominal fs'e göre sapma "
                               "(+: cihaz hızlı). Gerçek zamanda kalması için sapma küçük olmalı; ölçüm 2 s'den "
                               "sonra başlar, süre uzadıkça belirsizlik azalır."),
            ("pdv", "Gecikme değ.", "Gönderici zaman damgasına göre tek yön gecikmenin değişimi (son 2048 paket, "
                                        "p99 − en küçük; saat kayması çıkarılmış). Gönderici zaman damgası 0 ise "
                                        "ölçülemez."),
            ("clock", "Gönderici saati", "Gönderici saatinin PC saatine göre hız farkı (alt zarf eğimi)"),
        ])
        self.session = _Section("Oturum", [
            ("address", "Dinlenen", "Alıcının dinlediği bağlantı: UDP adres ve portu, USB portu ya da CAN adaptörü"),
            ("total", "Toplam", "Alınan datagram (USB / CAN'de çerçeve) ve bayt (tüm akışlar)"),
            ("errors", "Hatalı", "Çerçeve hatası (CRC, biçim) · tanımlayıcısı gelmemiş akışa ait çerçeve"),
            ("buffer", "Soket tamponu", "İşletim sisteminin verdiği UDP alım tamponu (istenen)"),
            ("stream", "Bayt akışı", "USB / CAN bayt akışı: alınan bayt · çerçeve dışında atlanan bayt (sıfır dolgu "
                                     "hariç) · senkron kaybı (bozuk çerçeveden sonra yeniden \"SPTF\" imzası arandı)"),
            ("can", "CAN mesajı", "Süzgeçten geçen veri mesajı · hata çerçevesi · süzgeç dışı (başka kimlik ya da "
                                  "uzak çerçeve) · kimlik / maske"),
        ])
        self.net = _Section("PC ağ arayüzü", [
            ("name", "Arayüz", "Verinin geldiği yerel ağ arayüzü (göndericiyle aynı alt ağ ya da yönlendirme)"),
            ("kind", "Tür / durum", "Bağlantı türü ve durumu"),
            ("speed", "Bağlantı hızı", "Arayüzün anlaşılan bağlantı hızı (alım yönü)"),
            ("mtu", "MTU", "En büyük IP paketi (bayt); SPTF standart datagram 1472 B için 1500 gerekir"),
            ("ip", "IP adresi", "Arayüzün IPv4 adresi / önek uzunluğu"),
            ("usage", "Hat kullanımı", "Bu arayüzden gelen akışların toplam hızı (başlıklar dahil yaklaşık) / "
                                       "bağlantı hızı"),
        ])
        for sec in (self.stream, self.timing, self.session, self.net):
            col.addWidget(sec)
        col.addStretch(1)
        self._show_transport("udp")
        scroll = QScrollArea()
        scroll.setWidget(body)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.viewport().setAutoFillBackground(False)
        body.setAutoFillBackground(False)
        outer.addWidget(scroll, 1)

        self._session: dict = {}
        self._rates: dict = {}
        self._iface = None
        self._net_timer = QTimer(self)
        self._net_timer.setInterval(NET_REFRESH_MS)
        self._net_timer.timeout.connect(self.refresh_network)
        self.refresh_network()

    def sizeHint(self) -> QSize:  # noqa: N802 — ilk yerleşimde bölücü bu genişliği verir
        return QSize(PANEL_WIDTH, super().sizeHint().height())

    # ---------------------------------------------------------------- besleme
    def current_key(self) -> str:
        return self.stream_combo.currentData() or ""

    def update_stats(self, session: dict, rates: dict) -> None:
        self._session, self._rates = session, rates
        self._sync_streams()
        self._render()

    def clear(self) -> None:
        self._rates = {}
        self._session = {}
        self._sync_streams()
        for sec in (self.stream, self.timing, self.session):
            sec.reset()
        self.badge.set_state("Akış yok", "idle")
        self._render_network()

    def _sync_streams(self) -> None:
        items = [(key, r.stats) for key, r in self._rates.items() if r.stats is not None]
        keys = [k for k, _ in items]
        current = self.current_key()
        if keys != [self.stream_combo.itemData(i) for i in range(self.stream_combo.count())]:
            self.stream_combo.blockSignals(True)
            self.stream_combo.clear()
            for key, st in items:
                self.stream_combo.addItem(f"{st.stream_id:02d} · {st.name or key}", key)
            if current in keys:
                self.stream_combo.setCurrentIndex(keys.index(current))
            elif keys:   # en yoğun akış
                busiest = max(items, key=lambda kv: self._rates[kv[0]].bytes_per_s)[0]
                self.stream_combo.setCurrentIndex(keys.index(busiest))
            self.stream_combo.blockSignals(False)
        self.stream_combo.setVisible(len(keys) > 1)

    # ---------------------------------------------------------------- çizim
    def _render(self) -> None:
        t = tokens()
        rate = self._rates.get(self.current_key())
        s = self._session
        transport = s.get("transport", "udp")
        self._show_transport(transport)
        self.session.set("address", s.get("address", "—") if s.get("connected", True)
                         else f"{s.get('address', '—')} · bağlı değil", None if s.get("connected", True) else t.warning)
        if "datagrams" in s:
            unit = "dgram" if transport == "udp" else "çerçeve"
            self.session.set("total", f"{fmt_count(s['datagrams'])} {unit} · {fmt_bytes(s.get('bytes', 0))}")
        if transport != "udp" and "stream_bytes" in s:
            skipped, resyncs = s.get("skipped_bytes", 0), s.get("resyncs", 0)
            self.session.set("stream", f"{fmt_bytes(s['stream_bytes'])} · {fmt_count(skipped)} B atlandı · "
                                       f"{fmt_count(resyncs)} senkron kaybı", t.warning if resyncs else None)
        if transport == "can" and "can_messages" in s:
            errors = s.get("can_error_frames", 0)
            self.session.set("can", f"{fmt_count(s['can_messages'])} · {fmt_count(errors)} hata çerçevesi · "
                                    f"{fmt_count(s.get('can_ignored', 0))} süzgeç dışı · {s.get('can_filter', '')}",
                             t.warning if errors else None)
        errors, nodesc = s.get("frame_errors", 0), s.get("no_descriptor", 0)
        self.session.set("errors", f"{fmt_count(errors)} · {fmt_count(nodesc)} tanımsız",
                         t.warning if errors else None)
        if s.get("socket_buffer"):
            want = s.get("socket_buffer_requested")
            extra = f" (istenen {fmt_bytes(want)})" if want and want != s["socket_buffer"] else ""
            self.session.set("buffer", fmt_bytes(s["socket_buffer"]) + extra)
        if rate is None or rate.stats is None:
            self.badge.set_state("Akış yok", "idle")
            self._render_network()
            return
        st = rate.stats
        now = time.monotonic()
        conflict = getattr(st, "conflict", "")
        if conflict:
            self.badge.set_state(f"Kaynak çakışması · {st.name or st.key}", "critical")
        elif st.idle_s >= IDLE_AFTER_S:
            self.badge.set_state(f"Bekliyor · {st.name or st.key}", "idle")
        elif now - rate.last_loss <= LOSS_HOLD_S:
            self.badge.set_state(f"Kayıplı · {st.name or st.key}", "warning")
        else:
            self.badge.set_state(f"Aktif · {st.name or st.key}", "good")
        self.stream.set("source", f"{st.source}:{st.port}" if st.port else st.source)
        if conflict == "port":
            ports = ", ".join(f":{p}" for p in st.source_ports)
            self.stream.set("conflict", f"{len(st.source_ports)} kaynak portu ({ports}) · "
                                        f"{fmt_count(st.collisions)} yabancı datagram", t.critical)
        elif conflict == "sıra":
            self.stream.set("conflict", "first_sample dizisi karışık (aynı stream_id?)", t.critical)
        else:
            self.stream.set("conflict", "yok" if not st.collisions else
                            f"yok (önceden {fmt_count(st.collisions)} yabancı datagram)")
        self.stream.set("pps", f"{fmt_count(rate.packets_per_s)} /s")
        self.stream.set("rate", fmt_bits(rate.bytes_per_s))
        total = st.samples + st.lost_samples
        pct = 100.0 * st.lost_samples / total if total else 0.0
        loss = f"{fmt_count(st.lost_samples)} örnek ({pct:.3g} %)"
        if st.loss_events:
            loss += f" · {fmt_count(st.loss_events)} olay, en uzun {fmt_count(st.max_loss_burst)}"
        self.stream.set("loss", loss, t.warning if now - rate.last_loss <= LOSS_HOLD_S else None)
        self.stream.set("last_loss", fmt_ago(now - rate.last_loss) if st.lost_samples else "yok")
        self.stream.set("dup", f"{fmt_count(st.duplicates)} yineleme · {fmt_count(st.late_packets)} geç · "
                               f"{fmt_count(st.restarts)} reset",
                        t.warning if st.late_packets or st.restarts else None)
        tm = st.timing
        if tm is not None:
            self.timing.set("interval", f"{fmt_us(tm.recent_mean_us)} · beklenen {fmt_us(tm.expected_us)}")
            self.timing.set("jitter", f"σ {fmt_us(tm.recent_std_us)} · p99 {fmt_us(tm.recent_p99_us)}")
            self.timing.set("gap", f"{fmt_us(tm.recent_max_us)} · oturum {fmt_us(tm.max_us)}")
            if math.isfinite(tm.rate_hz):
                hz = f"{tm.rate_hz:,.7g} Hz".replace(",", " ")
                self.timing.set("fs", f"{hz}<br>{fmt_ppm(tm.rate_ppm, tm.rate_ppm_err)}")
            else:
                self.timing.set("fs", "ölçülüyor… (≥ 2 s)")
            self.timing.set("pdv", fmt_us(tm.pdv_us) if math.isfinite(tm.pdv_us) else "— (zaman damgası yok)")
            self.timing.set("clock", fmt_ppm(tm.sender_ppm) if math.isfinite(tm.sender_ppm) else "—")
        self._render_network()

    def _show_transport(self, transport: str) -> None:
        """UDP'ye özgü satırlar (soket tamponu, PC ağ arayüzü) USB / CAN'de gizlenir; bayt akışı satırları gösterilir."""
        udp = transport == "udp"
        self.session.show_row("buffer", udp)
        self.session.show_row("stream", not udp)
        self.session.show_row("can", transport == "can")
        self.net.setVisible(udp)

    # ---------------------------------------------------------------- ağ arayüzü
    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._net_timer.start()
        self.refresh_network()

    def hideEvent(self, event) -> None:
        super().hideEvent(event)
        self._net_timer.stop()

    def refresh_network(self) -> None:
        rate = self._rates.get(self.current_key())
        source = rate.stats.source if rate is not None and rate.stats is not None else None
        self._iface = netinfo.interface_for(source)
        self._render_network()

    def _render_network(self) -> None:
        t = tokens()
        iface = self._iface
        if iface is None:
            self.net.reset()
            self.net.set("name", "bilinmiyor")
            return
        self.net.set("name", iface.name, tip=iface.description or iface.name)
        self.net.set("kind", f"{iface.kind} · {'bağlı' if iface.up else 'bağlı değil'}", None if iface.up else t.warning)
        self.net.set("speed", iface.speed_text)
        self.net.set("mtu", str(iface.mtu) if iface.mtu else "—",
                     t.warning if iface.mtu and iface.mtu < 1500 else None)
        self.net.set("ip", iface.address_text())
        # Bu arayüzden gelen akışların hattaki hızı: veri + çerçeve başına WIRE_OVERHEAD
        bps = 0.0
        for r in self._rates.values():
            if r.stats is not None and (iface.contains(r.stats.source) or
                                        (iface.kind == netinfo.LOOPBACK and r.stats.source.startswith("127."))):
                bps += (r.bytes_per_s + r.packets_per_s * WIRE_OVERHEAD) * 8
        if iface.speed_bps:
            usage = 100.0 * bps / iface.speed_bps
            self.net.set("usage", f"{usage:.2f} % ({bps / 1e6:.2f} / {iface.speed_bps / 1e6:g} Mbit/s)",
                         t.warning if usage > 70 else None)
        else:
            self.net.set("usage", f"{bps / 1e6:.2f} Mbit/s (bağlantı hızı bilinmiyor)")
