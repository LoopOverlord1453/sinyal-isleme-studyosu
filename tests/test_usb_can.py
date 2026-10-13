"""USB (sanal COM) ve CAN-bus taşımaları: bayt akışı çerçeve ayırıcı, alıcı süreçleri ve arayüz.

USB alıcısı pyserial'ın `socket://` adresiyle (yerel bir TCP sunucusu gerçek bir COM portu yerine), CAN alıcısı
python-can'in süreç içi sanal veri yoluyla (interface="virtual") uçtan uca sınanır; donanım gerekmez.
"""
import os
import queue
import socket
import threading
import time

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from spt.protocol.builder import build_data, build_descriptor, build_marker  # noqa: E402
from spt.protocol.format import SAMPLE_TYPES_BY_NAME, ChannelInfo, Descriptor, Layout  # noqa: E402
from spt.protocol.stream import StreamDeframer  # noqa: E402


def _desc(sid: int = 1, channels: int = 2) -> Descriptor:
    return Descriptor(sid, 1, 1000.0, SAMPLE_TYPES_BY_NAME["int16"], Layout.INTERLEAVED, f"akış{sid}",
                      tuple(ChannelInfo(f"c{i}", "V", 0.5, 0.0) for i in range(channels)))


def _frames(desc: Descriptor, blocks: int = 5, n: int = 8, crc: bool = True) -> list[bytes]:
    out = [build_descriptor(desc, 0, 0, crc=crc)]
    for i in range(blocks):
        samples = (np.arange(n * desc.channel_count).reshape(n, desc.channel_count) + i * 1000).astype("<i2")
        out.append(build_data(desc, i + 1, 1_000 * (i + 1), i * n, samples, crc=crc))
    out.append(build_marker(desc.stream_id, 1, blocks + 1, 0, 3, 7, "RUN", crc=crc))
    return out


# ------------------------------------------------------------------------------------------------ çerçeve ayırıcı
def test_deframer_any_split_gives_same_frames():
    frames = _frames(_desc())
    stream = b"".join(frames)
    for size in (1, 2, 3, 7, 8, 31, 32, 33, 64, 500, len(stream)):
        errors = []
        d = StreamDeframer(errors.append)
        out = []
        for i in range(0, len(stream), size):
            out += d.feed(stream[i:i + size])
        assert out == frames, size
        assert not errors and d.resyncs == 0 and d.skipped == 0


def test_deframer_zero_padding_is_not_an_error():
    frames = _frames(_desc())
    errors = []
    d = StreamDeframer(errors.append)
    out = []
    for f in frames:   # CAN FD gibi: her çerçeve 64 baytın katına sıfırla tamamlanır
        out += d.feed(f + bytes(-len(f) % 64))
    assert out == frames and not errors and d.skipped == 0


def test_deframer_starts_mid_frame_silently_and_recovers_from_garbage():
    frames = _frames(_desc())
    errors = []
    d = StreamDeframer(errors.append)
    out = d.feed(frames[1][10:])           # bağlantı akışın ortasında açıldı: sessizce atlanır
    assert out == [] and errors == []
    out = d.feed(b"".join(frames[2:4]))
    assert out == frames[2:4]
    before = d.skipped                       # açılıştaki yarım çerçevenin sıfır olmayan baytları
    out = d.feed(b"\x55\xAA garbage" + frames[4])   # senkron sonrası çerçeve dışı veri: tek hata
    assert out == [frames[4]]
    assert len(errors) == 1 and "senkron" in errors[0] and d.skipped - before == 10


def test_deframer_crc_error_resyncs_on_next_frame():
    frames = _frames(_desc())
    bad = bytearray(frames[2])
    bad[40] ^= 0xFF                         # örnek baytı bozuldu (ör. kaybolan / bozulan CAN mesajı)
    errors = []
    d = StreamDeframer(errors.append)
    out = d.feed(frames[0] + frames[1] + bytes(bad) + frames[3] + frames[4])
    assert out == [frames[0], frames[1], frames[3], frames[4]]
    assert len(errors) == 1 and errors[0].startswith("CRC hatası")


def test_deframer_lost_bytes_resync():
    frames = _frames(_desc())
    cut = frames[2][:20] + frames[2][28:]   # 8 bayt (bir CAN mesajı) kayboldu: frame_len artık tutmaz
    errors = []
    d = StreamDeframer(errors.append)
    out = d.feed(frames[1] + cut + frames[3] + frames[4])
    assert out == [frames[1], frames[3], frames[4]] and len(errors) == 1


def test_deframer_requires_crc():
    frames = _frames(_desc(), crc=False)
    errors = []
    d = StreamDeframer(errors.append)
    assert d.feed(b"".join(frames)) == []
    assert errors and "CRC'siz" in errors[0]


def test_deframer_magic_inside_samples_is_harmless():
    desc = Descriptor(1, 1, 1000.0, SAMPLE_TYPES_BY_NAME["uint8"], Layout.INTERLEAVED, "m", (ChannelInfo("a"),))
    payload = np.frombuffer(b"xxSPTF\x01\x02\x01\x00\xff\xff\x00\x00yy", np.uint8).reshape(-1, 1)
    frames = [build_descriptor(desc, crc=True), build_data(desc, 1, 0, 0, payload, crc=True),
              build_data(desc, 2, 0, len(payload), payload, crc=True)]
    errors = []
    d = StreamDeframer(errors.append)
    assert d.feed(b"".join(frames)) == frames and not errors


# ------------------------------------------------------------------------------------------------ USB alıcısı
def _events_until(events: queue.Queue, pred, timeout: float = 10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            ev = events.get(timeout=0.5)
        except queue.Empty:
            continue
        if pred(ev):
            return ev
    return None


def _ring_values(info, count: int) -> np.ndarray:
    from spt.core.ring import SharedRing

    ring = SharedRing.attach(info.shm_name, info.descriptor.sample_type.dtype)
    try:
        end = ring.write_count
        return ring.read(end - count, end)[1]
    finally:
        ring.close()


def test_serial_receiver_end_to_end_with_reconnect():
    from spt.sources.stream_receiver import SerialConfig, _SerialLink, serve_link

    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]
    commands, events = queue.Queue(), queue.Queue()
    cfg = SerialConfig(f"socket://127.0.0.1:{port}", 115200, ring_seconds=2.0)
    th = threading.Thread(target=serve_link, args=(_SerialLink(cfg), cfg, commands, events.put), daemon=True)
    th.start()
    desc = _desc()
    frames = _frames(desc, blocks=20)
    opened = lambda e: e[0] == "state" and e[1] == "listening" and "bekleniyor" not in e[2]  # noqa: E731
    try:
        conn, _ = server.accept()
        # pyserial portu açarken giriş tamponunu boşaltır (eski veri atılır): port açıldıktan sonra gönder
        assert _events_until(events, opened) is not None
        stream = b"\x07\x07" + b"".join(frames)        # açılışta yarım çerçeve artığı
        for i in range(0, len(stream), 13):            # USB paketleri çerçeve sınırına uymaz
            conn.sendall(stream[i:i + 13])
        ev = _events_until(events, lambda e: e[0] == "stream")
        assert ev is not None and ev[1].source == f"USB {cfg.port}" and ev[1].descriptor == desc
        stats = _events_until(events, lambda e: e[0] == "stats" and e[2] and e[2][0].packets == 20)
        assert stats is not None
        session, (st,) = stats[1], stats[2]
        assert session["transport"] == "usb" and session["connected"] and session["frame_errors"] == 0
        assert st.samples == 160 and st.lost_samples == 0 and st.source == f"USB {cfg.port}"
        values = _ring_values(ev[1], 160)
        assert values[-1].tolist() == [19014, 19015]

        conn.close()                                     # kart resetlendi / kablo çıktı
        assert _events_until(events, lambda e: e[0] == "log" and "koptu" in e[2]) is not None
        conn, _ = server.accept()                        # yeniden bağlandı
        assert _events_until(events, opened) is not None
        conn.sendall(b"".join(_frames(desc, blocks=2)))
        restart = _events_until(events, lambda e: e[0] == "log" and "yeniden başladı" in e[2])
        assert restart is not None                       # first_sample 0'a döndü: reset olarak tanındı
        conn.close()
    finally:
        commands.put(("stop",))
        th.join(5)
        server.close()
    assert not th.is_alive()


def test_serial_receiver_missing_port_waits_without_error():
    from spt.sources.stream_receiver import SerialConfig, _SerialLink, serve_link

    commands, events = queue.Queue(), queue.Queue()
    cfg = SerialConfig("COM250", 115200)
    th = threading.Thread(target=serve_link, args=(_SerialLink(cfg), cfg, commands, events.put), daemon=True)
    th.start()
    ev = _events_until(events, lambda e: e[0] == "state" and e[1] == "listening")
    commands.put(("stop",))
    th.join(5)
    assert ev is not None and "cihaz bekleniyor" in ev[2]
    assert not th.is_alive()


# ------------------------------------------------------------------------------------------------ CAN alıcısı
can = pytest.importorskip("can")


def test_can_receiver_two_senders_interleaved_and_filtered():
    from spt.sources.stream_receiver import CanConfig, _CanLink, serve_link

    channel = f"spt-test-{time.perf_counter_ns()}"
    commands, events, seen = queue.Queue(), queue.Queue(), []

    def emit(ev):
        seen.append(ev)
        events.put(ev)

    cfg = CanConfig("virtual", channel, 500_000, ring_seconds=2.0)
    th = threading.Thread(target=serve_link, args=(_CanLink(cfg), cfg, commands, emit), daemon=True)
    th.start()
    assert _events_until(events, lambda e: e[0] == "state" and e[1] == "listening") is not None
    tx = can.Bus(interface="virtual", channel=channel)
    a, b = _frames(_desc(1), blocks=6), _frames(_desc(2, channels=1), blocks=6)
    try:
        # İki gönderici (0x6F1, 0x6F2) mesaj mesaj araya girer; 0x100 başka bir düğümün trafiği
        sa, sb = b"".join(a), b"".join(b)
        for i in range(0, max(len(sa), len(sb)), 8):
            if i < len(sa):
                tx.send(can.Message(arbitration_id=0x6F1, data=sa[i:i + 8], is_extended_id=False))
            if i < len(sb):
                tx.send(can.Message(arbitration_id=0x6F2, data=sb[i:i + 8], is_extended_id=False))
            tx.send(can.Message(arbitration_id=0x100, data=b"\x11" * 8, is_extended_id=False))
        found = _events_until(events, lambda e: e[0] == "stats" and len(e[2]) == 2
                              and all(st.packets == 6 for st in e[2]))
        assert found is not None
        session, stats = found[1], {st.source: st for st in found[2]}
        assert set(stats) == {"CAN 0x6F1", "CAN 0x6F2"}
        assert session["transport"] == "can" and session["frame_errors"] == 0 and session["resyncs"] == 0
        assert session["can_messages"] > 0 and session["can_filter"] == "0x6F0 / 0x7F0"
        assert stats["CAN 0x6F1"].samples == 48 and stats["CAN 0x6F2"].samples == 48
        assert _events_until(events, lambda e: e[0] == "stats" and e[1]["can_messages"]
                             == -(-len(sa) // 8) - (-len(sb) // 8)) is not None    # 0x100 mesajları sayılmaz
        markers = sorted(ev[1] for ev in seen if ev[0] == "marker")
        assert markers == ["CAN 0x6F1#1", "CAN 0x6F2#2"]
    finally:
        commands.put(("stop",))
        th.join(5)
        tx.shutdown()
    assert not th.is_alive()


def test_can_fd_padded_messages():
    from spt.sources.stream_receiver import CanConfig, _CanLink, serve_link

    channel = f"spt-fd-{time.perf_counter_ns()}"
    commands, events = queue.Queue(), queue.Queue()
    cfg = CanConfig("virtual", channel, 500_000, fd=True, ring_seconds=2.0)
    th = threading.Thread(target=serve_link, args=(_CanLink(cfg), cfg, commands, events.put), daemon=True)
    th.start()
    assert _events_until(events, lambda e: e[0] == "state" and e[1] == "listening") is not None
    tx = can.Bus(interface="virtual", channel=channel)
    try:
        for frame in _frames(_desc(), blocks=4, n=40):    # her SPTF çerçevesi yeni mesajda başlar, son parça dolgulu
            for i in range(0, len(frame), 64):
                chunk = frame[i:i + 64]
                dlc = next(n for n in (8, 12, 16, 20, 24, 32, 48, 64) if n >= len(chunk))
                tx.send(can.Message(arbitration_id=0x6F0, data=chunk + bytes(dlc - len(chunk)),
                                    is_extended_id=False, is_fd=True))
        found = _events_until(events, lambda e: e[0] == "stats" and e[2] and e[2][0].packets == 4)
        assert found is not None
        assert found[1]["frame_errors"] == 0 and found[1]["skipped_bytes"] == 0
        assert found[2][0].samples == 160 and found[2][0].lost_samples == 0
    finally:
        commands.put(("stop",))
        th.join(5)
        tx.shutdown()


def test_can_config_texts():
    from spt.sources.stream_receiver import CanConfig, can_source

    cfg = CanConfig("pcan", "PCAN_USBBUS1", 500_000, True, 2_000_000)
    assert cfg.describe() == "CAN PCAN_USBBUS1 · 500 kbit/s · FD 2 Mbit/s"
    assert cfg.id_text() == "0x6F0 / 0x7F0"
    assert can_source(0x6F1, False) == "CAN 0x6F1" and can_source(0x18FF0001, True) == "CAN 0x18FF0001"


# ------------------------------------------------------------------------------------------------ arayüz
@pytest.fixture(scope="module")
def app():
    from PyQt6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def theme(app):
    from spt.gui.theme import ThemeManager

    t = ThemeManager(app)
    t.apply()
    return t


def pump(app, seconds):
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        app.processEvents()
        time.sleep(0.005)


def wait_for(app, condition, timeout=15.0):
    end = time.perf_counter() + timeout
    while time.perf_counter() < end:
        app.processEvents()
        if condition():
            return True
        time.sleep(0.01)
    return False


def test_sources_panel_transport_pages_and_validation(app, theme):
    from PyQt6.QtGui import QAction

    from spt.gui.panels.sources_panel import SourcesPanel

    panel = SourcesPanel(theme, QAction("dinle"), QAction("aç"), QAction("sim"))
    panel.resize(320, 700)
    panel.show()
    pump(app, 0.05)
    assert panel.transport() == "udp" and panel.address.isVisible() and not panel.serial_port.isVisible()
    panel.set_transport("usb")
    assert panel.serial_port.isVisible() and not panel.address.isVisible()
    panel.serial_port.setEditText("")
    with pytest.raises(ValueError, match="USB portu"):
        panel.serial_config(10.0, 256)
    panel.serial_port.setEditText("COM42")
    panel.baud.setEditText("921600")
    cfg = panel.serial_config(10.0, 256)
    assert (cfg.port, cfg.baudrate) == ("COM42", 921600)

    panel.set_transport("can")
    assert panel.can_id.isVisible() and not panel.serial_port.isVisible()
    assert panel.can_channel.text() == "PCAN_USBBUS1"
    panel.can_interface.setCurrentIndex(panel.can_interface.findData("kvaser"))
    assert panel.can_channel.text() == "0"                  # örnek kanal adaptöre göre değişir
    panel.can_id.setText("0x800")
    with pytest.raises(ValueError, match="11 bit"):
        panel.can_config(10.0, 256)
    panel.can_extended.setChecked(True)
    panel.can_fd.setChecked(True)
    cfg = panel.can_config(10.0, 256)
    assert cfg.can_id == 0x800 and cfg.extended and cfg.fd and cfg.interface == "kvaser"
    panel.set_listening(True)
    assert not panel.transport_combo.isEnabled() and not panel.can_id.isEnabled()
    panel.set_listening(False)
    panel.close()
    panel.deleteLater()


def test_main_window_listens_usb_and_reports_can_driver_error(app, theme, tmp_path, monkeypatch):
    monkeypatch.setenv("SPT_SETTINGS_FILE", str(tmp_path / "settings.ini"))
    from spt.gui.main_window import MainWindow

    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    server.settimeout(15)
    port = server.getsockname()[1]
    w = MainWindow(theme)
    w.resize(1600, 960)
    w.show()
    pump(app, 0.2)
    try:
        panel = w.sources_panel
        panel.set_transport("usb")
        panel.serial_port.setEditText(f"socket://127.0.0.1:{port}")
        w.registry["conn.listen"].trigger()               # F5 / Dinlemeyi Başlat
        assert w.sources.mode == "usb"
        conn, _ = server.accept()
        assert wait_for(app, lambda: w.sources.state == "listening" and "bekleniyor" not in w.sources.detail)
        desc = _desc()
        conn.sendall(b"".join(_frames(desc, blocks=30)))
        assert wait_for(app, lambda: len(w.sources.streams) == 1)
        handle = next(iter(w.sources.streams.values()))
        assert handle.source.startswith("USB socket://") and handle.desc == desc
        assert wait_for(app, lambda: w.sources.state == "receiving" or w.sources.state == "listening")
        assert panel._sources and not panel.transport_combo.isEnabled()
        assert wait_for(app, lambda: w.connection_view.stats.session.values["total"].text().endswith("B")
                        and "çerçeve" in w.connection_view.stats.session.values["total"].text())
        assert not w.connection_view.stats.net.isVisibleTo(w.connection_view.stats)
        w.registry["conn.listen"].trigger()               # durdur
        assert wait_for(app, lambda: w.sources.mode is None)
        conn.close()

        # CAN: sürücüsü kurulu olmayan adaptör (bu makinede PCAN-Basic yok) → hata durumu, dinleme kapanır
        panel.set_transport("can")
        w.registry["conn.listen"].trigger()
        assert wait_for(app, lambda: w.sources.state == "error" and not w.sources.active, timeout=30)
        assert not w.registry["conn.listen"].isChecked() and panel.transport_combo.isEnabled()
        w._save_link_settings()
        assert w._settings().value("conn/transport") == "can"
    finally:
        w.close()
        w.deleteLater()
        pump(app, 0.2)
        server.close()
