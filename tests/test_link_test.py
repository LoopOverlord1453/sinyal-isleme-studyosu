"""Arayüz (ekransız): üst gezinmedeki Bağlantı görünümü (istatistikler ve UDP Link Testi: pasif ve yankı) ve
Canlı İzleme'nin KPI satırı.

Pasif test, alıcı süreci olmadan sentetik istatistik olaylarıyla beslenir; yankı testi gerçek UDP ile, yerel yankı
sunucusu ve test ayrı süreçlerde çalışır. Ayarlar geçici INI dosyasındadır.
"""
import os
import socket
import time

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication  # noqa: E402

from spt.core.stream_store import StreamStore  # noqa: E402
from spt.protocol.builder import build_data, build_descriptor  # noqa: E402
from spt.protocol.format import SAMPLE_TYPES_BY_NAME, ChannelInfo, Descriptor, Layout  # noqa: E402
from spt.sources.manager import StreamRate  # noqa: E402

SRC = "192.168.172.5"


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def theme(app):
    from spt.gui.theme import ThemeManager

    t = ThemeManager(app)
    t.apply()
    return t


@pytest.fixture
def win(app, theme, tmp_path, monkeypatch):
    monkeypatch.setenv("SPT_SETTINGS_FILE", str(tmp_path / "settings.ini"))
    from spt.gui.main_window import MainWindow

    w = MainWindow(theme)
    w.resize(1600, 960)
    w.show()
    pump(app, 0.2)
    yield w
    w.close()
    w.deleteLater()
    pump(app, 0.2)


def pump(app, seconds):
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        app.processEvents()
        time.sleep(0.005)


class _Feed:
    """Sentetik akış: StreamStore'a çerçeve verip SourceManager'ın yaydığı (oturum, hızlar) olaylarını üretir."""

    def __init__(self, gap_at: int | None = None):
        self.events = []
        self.store = StreamStore(self.events.append, 0, ring_seconds=2.0)
        self.desc = Descriptor(1, 1, 100_000.0, SAMPLE_TYPES_BY_NAME["int16"], Layout.INTERLEAVED, "kart",
                               (ChannelInfo("a"),))
        d = build_descriptor(self.desc)
        self.store.handle_datagram(memoryview(d), len(d), SRC, 1, 40000)
        self.data = np.zeros((100, 1), np.int16)
        self.i = 0
        self.rate = StreamRate()
        self.gap_at = gap_at
        self.rng = np.random.default_rng(3)

    def step(self, packets: int = 250):
        """`packets` çerçeve (1 ms aralık, ±30 µs titreşim) ve bir istatistik olayı; (oturum, {anahtar: hız})."""
        for _ in range(packets):
            i = self.i
            self.i += 1
            if self.gap_at is not None and i == self.gap_at:
                continue                                       # kayıp datagram
            rx = 1_000_000_000 + i * 1_000_000 + int(self.rng.normal(0, 30e3))
            f = build_data(self.desc, i, 5_000_000 + i * 1_000_000, i * 100, self.data)
            self.store.handle_datagram(memoryview(f), len(f), SRC, rx, 40000)
        t = (1_000_000_000 + self.i * 1_000_000) / 1e9
        (st,) = self.store.stats(int(t * 1e9))
        r = self.rate
        if r._prev is not None:
            packets0, bytes0, lost0, t0 = r._prev
            r.packets_per_s = (st.packets - packets0) / (t - t0)
            r.bytes_per_s = (st.bytes - bytes0) / (t - t0)
            if st.lost_samples > lost0:
                r.last_loss = time.monotonic()
        r._prev = (st.packets, st.bytes, st.lost_samples, t)
        r.stats = st
        session = {"t": t, "datagrams": self.i, "bytes": st.bytes, "frame_errors": 0, "no_descriptor": 0,
                   "address": "0.0.0.0:50505", "socket_buffer": 64 << 20, "socket_buffer_requested": 64 << 20}
        return session, {st.key: r}

    def close(self):
        self.store.close()


def test_connection_view_in_top_navigation_shows_stats(app, win):
    """Bağlantı ayrı bir yan panel değil: üst gezinmede bir görünüm. Canlı İzleme'de yan panel yok."""
    from spt.gui.main_window import NAV_ORDER, VIEW_KEYS

    assert NAV_ORDER[1] == "view.connection" and VIEW_KEYS[-1] == "view.connection"
    assert [p.text() for p in win._nav_pills][1] == "Bağlantı"
    assert not hasattr(win.live_view, "body") and not hasattr(win, "link_panel")
    win.registry["view.connection"].trigger()
    assert win.views.currentWidget() is win.connection_view and win.registry["view.connection"].isChecked()
    panel = win.connection_view.stats

    feed = _Feed(gap_at=600)
    for _ in range(4):                                         # 1 s veri, 600. datagram kayıp
        win._on_stats(*feed.step())
    pump(app, 0.05)
    values = {k: v.text() for sec in (panel.stream, panel.timing, panel.session) for k, v in sec.values.items()}
    assert values["source"] == f"{SRC}:40000"
    assert "/s" in values["pps"] and "Mbit/s" in values["rate"] or "kbit/s" in values["rate"]
    assert "100 örnek" in values["loss"] and "1 olay" in values["loss"]   # renkli (kayıp yeni): HTML
    assert "beklenen 1 ms" in values["interval"] and "σ" in values["jitter"]
    assert values["address"] == "0.0.0.0:50505" and "MB" in values["buffer"]
    assert panel.badge.text().endswith("Kayıplı · kart")
    assert panel.net.values["name"].text()                     # arayüz adı ya da "bilinmiyor"
    # istatistik sütunu ilk gösterimde geniş (dar sütunda satırlar kırılıyordu)
    pump(app, 0.1)
    assert win.connection_view.split.sizes()[0] >= 380
    win._on_streams_cleared()
    assert panel.badge.text().endswith("Akış yok")
    feed.close()


def test_live_kpi_strip_shows_rates_counts_errors_and_loss(app, win):
    kpi = win.live_view.kpi
    assert [t.findChild(type(t.value), "KpiTitle").text() for t in kpi.tiles] == \
        ["Paket hızı", "Veri hızı", "Paket sayısı", "Hatalı paket", "Kayıp (örnek)"]
    assert win.review_view.kpi is None                         # İnceleme'de canlı istatistik yok
    feed = _Feed(gap_at=300)
    for _ in range(3):
        session, rates = feed.step()
        session["frame_errors"], session["no_descriptor"] = 2, 1
        win._on_stats(session, rates)
    pump(app, 0.05)
    assert kpi.pps.value.text() == "1\u202f000" and kpi.pps.unit.text() == "paket/s"   # 1 ms aralık
    assert kpi.rate.unit.text() == "kB/s" and "Mbit/s" in kpi.rate.sub.text()
    assert kpi.count.value.text() == "749" and "750 dgram" in kpi.count.sub.text()
    assert kpi.errors.value.text() == "3" and kpi.errors.sub.text() == "CRC/biçim 2 · tanımsız 1"
    assert kpi.errors.badge.toolTip() == "Hatalı" and kpi.errors.badge.property("state") == "critical"
    assert kpi.loss.value.text() == "100" and "1 olay" in kpi.loss.sub.text() and kpi.loss.badge.toolTip() == "Kayıplı"
    assert len(kpi.pps.spark.values()) == 3                    # çizgi grafik her olayda bir nokta
    win._on_streams_cleared()
    assert kpi.pps.value.text() == "0" and kpi.loss.badge.toolTip() == "Bekliyor" and not kpi.pps.spark.values()
    feed.close()


def test_passive_link_test_report_and_export(app, win, tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QFileDialog

    feed = _Feed(gap_at=1200)
    win._on_stats(*feed.step())                               # akış listede görünsün
    win.sources.rates = feed.step()[1]
    win.registry["tools.link_test"].trigger()                  # Araçlar / Bağlantı menüsü: Bağlantı görünümü
    view = win.connection_view
    tab = view.passive
    assert win.views.currentWidget() is view and view.tabs.currentWidget() is tab
    assert tab.stream.count() == 1 and tab.start_btn.isEnabled()
    tab.duration.setValue(2)
    tab.loss.setValue(0.5)                                     # 100 / ~200 000 örnek = %0.05 → uygun
    tab.toggle()
    assert tab.running
    for _ in range(12):                                        # 12 × 0.25 s olay: süre dolunca biter
        win.sources.stats_updated.emit(*feed.step())
    pump(app, 0.05)
    assert not tab.running
    rep = tab._report
    assert rep is not None and rep.duration_s == pytest.approx(2.0, abs=0.01)
    assert rep.lost_samples == 100 and rep.loss_events == 1 and rep.passed is True
    assert tab.verdict.text().endswith(f"Gerçek zamana uygun ({rep.duration_s:.1f} s)")
    assert tab.results.rowCount() == len(rep.rows()) and tab.checks.rowCount() == len(rep.checks)
    tab.loss.setValue(0.01)                                    # eşik değişince sonuç yeniden değerlendirilir
    assert tab._report.passed is False and "Kayıp oranı" in tab.verdict.text()
    out = tmp_path / "rapor.md"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(out), ""))
    tab.export_md.click()
    assert "UYGUN DEĞİL" in out.read_text(encoding="utf-8")
    feed.close()


def test_source_collision_shown_in_panel_stats_dock_and_report(app, win):
    from spt.sources.link_report import PassiveTest

    events = []
    store = StreamStore(events.append, 0, ring_seconds=2.0)
    desc = Descriptor(1, 1, 100_000.0, SAMPLE_TYPES_BY_NAME["int16"], Layout.INTERLEAVED, "kart", (ChannelInfo("a"),))
    d = build_descriptor(desc)
    store.handle_datagram(memoryview(d), len(d), SRC, 1, 50001)
    data = np.zeros((100, 1), np.int16)
    rate, test, rx = StreamRate(), None, 1_000_000_000
    for i in range(400):                                       # iki gönderici, aynı stream_id, farklı kaynak portu
        for port, first in ((50001, i * 100), (50002, 50_000 + i * 100)):
            rx += 500_000
            f = build_data(desc, i, rx, first, data)
            store.handle_datagram(memoryview(f), len(f), SRC, rx, port)
        if i % 100 == 99:
            (st,) = store.stats(rx)
            rate.stats, rate.packets_per_s, rate.bytes_per_s = st, 1000.0, 2e5
            win._on_stats({"t": rx / 1e9}, {st.key: rate})
            test = test or PassiveTest(st.key)
            test.feed(rx / 1e9, st)
    panel = win.connection_view.stats
    assert panel.badge.text().endswith("Kaynak çakışması · kart")
    assert ":50001, :50002" in panel.stream.values["conflict"].text()
    badge = win.stats_panel._rows[st.key][1]
    assert badge.text().endswith("Çakışma")
    rep = test.report()
    check = rep.checks[0]
    assert check.name.startswith("Kaynak çakışması") and check.ok is False and rep.passed is False
    assert "50002" in check.value and "| Kaynak çakışması |" in rep.to_markdown()
    store.close()


def _wait(app, cond, timeout=30.0):
    end = time.perf_counter() + timeout
    while time.perf_counter() < end:
        app.processEvents()
        if cond():
            return True
        time.sleep(0.01)
    return False


def test_echo_tab_with_local_server_and_missing_server(app, win):
    win._show_link_test(echo=True)
    tab = win.connection_view.echo
    assert win.connection_view.tabs.currentWidget() is tab
    tab.local.setChecked(True)                                 # yerel yankı sunucusu ayrı süreçte
    assert _wait(app, lambda: tab.start_btn.isEnabled() and tab.port.value() != 7)
    tab.rate.setValue(1000)
    tab.count.setValue(500)
    tab.payload.setValue(256)
    tab.toggle()
    assert _wait(app, lambda: tab._result is not None and tab._result.finished)
    snap = tab._result
    assert (snap.sent, snap.received, snap.lost) == (500, 500, 0)
    assert 0 < snap.stats["p50"] < 100_000 and "kayıpsız" in tab.verdict.text()
    assert tab.results.item(1, 1).text() == "500 / 500" and len(snap.plot_seq) == 500
    tab.local.setChecked(False)
    assert tab._server is None and tab.host.isEnabled()

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:   # kapalı port: kartta yankı sunucusu yok
        s.bind(("127.0.0.1", 0))
        closed = s.getsockname()[1]
    tab.host.setText("127.0.0.1")
    tab.port.setValue(closed)
    tab.count.setValue(10)
    tab.rate.setValue(100)
    tab.timeout.setValue(200)
    tab.toggle()
    assert _wait(app, lambda: tab._result is not None and tab._result.finished and tab._result.sent == 10)
    assert tab.verdict.text().endswith("Yanıt yok")
    assert "yankı sunucusu gerekiyor" in tab.diagnosis.text() and not tab.diagnosis.isHidden()
    tab.host.setText("")
    tab.toggle()                                               # geçersiz ayar: süreç başlamaz, açık ileti
    assert tab._proc is None and "Hedef IP" in tab.verdict.text()
    win.connection_view.shutdown()
