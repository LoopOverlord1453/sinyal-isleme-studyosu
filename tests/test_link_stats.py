"""Bağlantı istatistikleri: alıcıdaki varış zamanlaması, hız / saat sapması, gecikme değişimi, sıra dışı çerçeve,
pasif link testi raporu, UDP yankı (RTT) testi ve ağ arayüzü bilgisi. Qt gerekmez."""
import math
import queue
import socket
import threading
import time

import numpy as np
import pytest

from spt.core.stream_store import TIMING_EDGES_US, StreamStats, StreamStore, hist_percentile
from spt.protocol.builder import build_data, build_descriptor
from spt.protocol.format import SAMPLE_TYPES_BY_NAME, ChannelInfo, Descriptor, Layout

SRC = "10.0.0.2"


def _desc(fs: float = 100_000.0) -> Descriptor:
    return Descriptor(1, 1, fs, SAMPLE_TYPES_BY_NAME["int16"], Layout.INTERLEAVED, "link", (ChannelInfo("a"),))


def _feed(store: StreamStore, frames, port: int = 0) -> None:
    for frame, rx in frames:
        store.handle_datagram(memoryview(frame), len(frame), SRC, rx, port)


def _stream(rx_ns, ts_ns=None, count=100, fs=100_000.0, firsts=None, port=0):
    """Düzenli çerçeve dizisi: i. çerçeve rx_ns[i]'de gelir; gönderici zamanı ts_ns[i]."""
    events = []
    store = StreamStore(events.append, 0, ring_seconds=2.0)
    desc = _desc(fs)
    d = build_descriptor(desc)
    store.handle_datagram(memoryview(d), len(d), SRC, int(rx_ns[0]) - 1000, port)
    data = np.zeros((count, 1), np.int16)
    firsts = range(0, len(rx_ns) * count, count) if firsts is None else firsts
    ts = ts_ns if ts_ns is not None else np.zeros(len(rx_ns), np.int64)
    frames = [(build_data(desc, i, int(t), int(f), data), int(r)) for i, (f, r, t) in enumerate(zip(firsts, rx_ns, ts))]
    return store, frames


def test_arrival_timing_mean_jitter_histogram():
    rng = np.random.default_rng(1)
    n = 4000
    jitter = rng.normal(0, 20e3, n)                                       # σ 20 µs
    rx = (1_000_000_000 + np.arange(n) * 1_000_000 + jitter).astype(np.int64)  # 1 ms aralık
    store, frames = _stream(rx)
    _feed(store, frames[:1500])
    store.stats(int(rx[1499]))                                            # arada boşaltma: birleşim doğru olmalı
    _feed(store, frames[1500:])
    st = store.stats(int(rx[-1]))[0]
    tm = st.timing
    d = np.diff(rx) / 1e3
    assert tm.intervals == n - 1 == sum(tm.hist)
    assert tm.mean_us == pytest.approx(d.mean(), rel=1e-9) and tm.std_us == pytest.approx(d.std(ddof=1), rel=1e-6)
    assert tm.max_us == pytest.approx(d.max()) and tm.expected_us == pytest.approx(1000.0)
    assert hist_percentile(tm.hist, 99) == pytest.approx(np.percentile(d, 99), rel=0.13)   # kutu ≈ %12
    assert math.isnan(tm.pdv_us) and math.isnan(tm.sender_ppm)          # gönderici zamanı 0: ölçülmez
    assert st.sample_rate == 100_000.0 and (st.loss_events, st.late_packets) == (0, 0)
    store.close()


def test_effective_rate_and_sender_clock_ppm():
    """Cihaz PC saatine göre %0.005 hızlı: örnekler ve gönderici saati +50 ppm okunur."""
    n = 5000
    rx = (1_000_000_000 + np.arange(n) * 1_000_000 * (1 - 50e-6)).astype(np.int64)
    ts = (7_000_000 + np.arange(n) * 1_000_000).astype(np.int64)          # gönderici saati: tam 1 ms
    store, frames = _stream(rx, ts)
    for k in range(0, n, 250):                                            # ≈ 4 Hz boşaltma gibi
        _feed(store, frames[k:k + 250])
        store.stats(int(rx[min(k + 249, n - 1)]))
    tm = store.stats(int(rx[-1]))[0].timing
    assert tm.rate_ppm == pytest.approx(50.0, abs=1.0) and tm.rate_hz == pytest.approx(100_005.0, abs=0.1)
    assert tm.sender_ppm == pytest.approx(50.0, abs=1.0)
    assert tm.pdv_us == pytest.approx(0.0, abs=0.05)                      # saat kayması gecikme değişimi değildir
    store.close()


def test_packet_delay_variation_from_sender_timestamps():
    rng = np.random.default_rng(4)
    n = 3000
    ts = (5_000_000 + np.arange(n) * 1_000_000).astype(np.int64)
    delay = rng.uniform(0, 300e3, n)                                      # 0 … 300 µs kuyruk gecikmesi
    rx = (2_000_000_000 + ts + delay).astype(np.int64)
    store, frames = _stream(rx, ts)
    _feed(store, frames)
    tm = store.stats(int(rx[-1]))[0].timing
    window = delay[-2048:] / 1e3
    assert tm.pdv_us == pytest.approx(np.percentile(window, 99) - window.min(), abs=3.0)   # eğim düzeltmesi payı
    store.close()


def test_reordered_frame_is_late_not_duplicate_and_port_recorded():
    rx = 1_000_000_000 + np.arange(5) * 1_000_000
    store, frames = _stream(rx, port=40123)
    order = [0, 1, 3, 2, 4]                                               # 2 ile 3 yer değiştirdi
    _feed(store, [frames[i] for i in order], port=40123)
    _feed(store, [frames[1]], port=40123)                                 # gerçek yinelenen
    st = store.stats(int(rx[-1]))[0]
    assert (st.lost_samples, st.loss_events, st.max_loss_burst) == (100, 1, 100)
    assert (st.late_packets, st.late_samples, st.duplicates) == (1, 100, 1)
    assert st.port == 40123
    store.close()


def test_stream_stats_positional_construction_still_works():
    st = StreamStats("k", 1, "a", SRC, "f", 1, 2, 3, 4, 5, 6, 0.5)       # eski 12 alanlı biçim
    assert st.port == 0 and st.timing is None and st.late_packets == 0


def test_hist_percentile_bounds():
    hist = np.zeros(len(TIMING_EDGES_US) + 1)
    assert math.isnan(hist_percentile(hist, 50))
    i = int(np.searchsorted(TIMING_EDGES_US, 1000.0, side="right"))      # 1 ms'nin kutusu
    hist[i] = 10
    assert TIMING_EDGES_US[i - 1] <= hist_percentile(hist, 50) <= TIMING_EDGES_US[i]


# ------------------------------------------------------------------------------------------------ kaynak çakışması
def _two_senders(port_a: int, port_b: int, n: int = 400, offset_b: int = 37_000):
    """İki gönderici aynı IP + stream_id: A 0'dan, B `offset_b`'den başlayan kendi first_sample dizisiyle, iç içe."""
    events = []
    store = StreamStore(events.append, 0, ring_seconds=2.0)
    desc = _desc()
    d = build_descriptor(desc)
    store.handle_datagram(memoryview(d), len(d), SRC, 1_000_000, port_a)
    data = np.zeros((100, 1), np.int16)
    rx = 1_000_000_000
    for i in range(n):
        for port, first in ((port_a, i * 100), (port_b, offset_b + i * 100)):
            rx += 500_000
            f = build_data(desc, i, rx, first, data)
            store.handle_datagram(memoryview(f), len(f), SRC, rx, port)
        if i % 100 == 99:
            store.stats(rx)
    return store, events, rx


def _warnings(events):
    return [e[2] for e in events if e[0] == "log" and e[1] == 30]


def test_two_source_ports_same_stream_id_is_a_collision():
    store, events, rx = _two_senders(50001, 50002)
    st = store.stats(rx)[0]
    assert st.conflict == "port" and st.collisions == 400 and st.source_ports == (50001, 50002)
    warn = [w for w in _warnings(events) if "iki kaynaktan" in w]
    assert len(warn) == 1                                                  # tekrar oranla sınırlı
    assert f"{SRC}:50001 ve {SRC}:50002" in warn[0] and "stream_id'leri ayırın" in warn[0]
    assert st.timing.intervals < st.packets                                # yabancı kaynağın varışları zamanlamaya girmez
    # Yabancı kaynağın örnekleri tampona yazılmaz: tamponda yalnız asıl kaynağın kesintisiz dizisi (iki sinyal aynı
    # kanalda parça parça karışmaz; ne kayıp boşluğu ne reset)
    ring = store._streams[(SRC, 1)].ring
    assert (st.packets, st.lost_samples, st.restarts, st.duplicates) == (400, 0, 0, 0)
    assert ring.write_count == 400 * 100 and ring.gaps_between(0, ring.write_count) == []
    assert store._streams[(SRC, 1)].foreign_packets == 400
    st2 = store.stats(rx + 6_000_000_000)[0]                               # 5 s sonra çakışma söner
    assert st2.conflict == "" and st2.source_ports == (50001,)
    store.close()


def test_new_source_port_after_silence_is_a_reconnect_not_a_collision():
    events = []
    store = StreamStore(events.append, 0, ring_seconds=2.0)
    desc = _desc()
    d = build_descriptor(desc)
    store.handle_datagram(memoryview(d), len(d), SRC, 1, 50001)
    data = np.zeros((100, 1), np.int16)
    for i, (rx, port) in enumerate([(10**9, 50001), (10**9 + 10**6, 50001), (3 * 10**9, 50009), (3 * 10**9 + 10**6, 50009)]):
        f = build_data(desc, i, rx, i * 100, data)
        store.handle_datagram(memoryview(f), len(f), SRC, rx, port)
    st = store.stats(3 * 10**9 + 10**6)[0]
    assert st.port == 50009 and st.collisions == 0 and st.conflict == "" and st.packets == 4
    assert any("kaynak portu değişti 50001 → 50009" in e[2] for e in events if e[0] == "log")
    store.close()


def test_foreign_descriptor_does_not_reconfigure_active_stream():
    store, events, rx = _two_senders(50001, 50002, n=50)
    other = Descriptor(1, 1, 48_000.0, SAMPLE_TYPES_BY_NAME["int16"], Layout.INTERLEAVED, "baska", (ChannelInfo("x"),))
    d = build_descriptor(other)
    store.handle_datagram(memoryview(d), len(d), SRC, rx + 1000, 50002)
    assert not [e for e in events if e[0] == "stream_removed"]
    st = store.stats(rx + 2000)[0]
    assert st.sample_rate == 100_000.0 and st.conflict == "port"
    store.close()


def test_interleaved_sequences_from_one_port_flag_sequence_conflict():
    store, events, rx = _two_senders(50001, 50001)
    st = store.stats(rx)[0]
    assert st.conflict == "sıra" and st.collisions == 0 and (st.restarts > 2 or st.duplicates > 20)
    assert len([w for w in _warnings(events) if "first_sample dizisi tutarsız" in w]) == 1
    store.close()


# ------------------------------------------------------------------------------------------------ pasif rapor
def test_passive_report_window_and_verdict():
    from spt.sources.link_report import PassiveTest, Thresholds

    rng = np.random.default_rng(7)
    n = 6000
    rx = (1_000_000_000 + np.arange(n) * 1_000_000 + rng.normal(0, 30e3, n)).astype(np.int64)
    ts = (np.arange(n) * 1_000_000).astype(np.int64) + 1
    store, frames = _stream(rx, ts)
    del frames[4000:4003]                                                 # 3 datagram (300 örnek) kayıp
    rx = np.delete(rx, range(4000, 4003))
    n = len(rx)
    test = PassiveTest("10.0.0.2#1")
    for k in range(0, n, 250):
        _feed(store, frames[k:k + 250])
        last = min(k + 249, n - 1)
        if k >= 1000:                                                     # test 1000. çerçeveden sonra başlar
            test.feed(rx[last] / 1e9, store.stats(int(rx[last]))[0])
        else:
            store.stats(int(rx[last]))
    rep = test.report(Thresholds(max_gap_ms=20, loss_pct=0.01, jitter_p99_us=200, rate_ppm=500))
    win = np.diff(rx[1249:]) / 1e3                                        # ilk beslemeden sonraki aralıklar
    assert rep.packets == n - 1250 and rep.lost_samples == 300 and rep.loss_events == 1
    assert rep.mean_us == pytest.approx(win.mean(), rel=1e-6) and rep.std_us == pytest.approx(win.std(ddof=1), rel=1e-3)
    assert rep.hist.sum() == len(win)
    assert rep.loss_pct == pytest.approx(100 * 300 / (rep.samples + 300))
    checks = {c.name: c.ok for c in rep.checks}
    assert checks["Kayıp oranı"] is False and checks["En büyük boşluk"] is True
    assert rep.passed is False
    md, csv = rep.to_markdown(), rep.to_csv()
    assert "UYGUN DEĞİL" in md and "| Kayıp oranı |" in md and csv.startswith("ölçü;değer")
    assert test.report(Thresholds(loss_pct=1.0, jitter_p99_us=500)).passed is True
    store.close()


# ------------------------------------------------------------------------------------------------ yankı testi
def test_echo_against_local_server():
    from spt.sources.echo_test import EchoConfig, EchoServer, EchoTest

    srv = EchoServer().start()
    try:
        progress = []
        res = EchoTest(EchoConfig("127.0.0.1", srv.port, 200, 2000.0, 400, 0.5), progress.append).run()
    finally:
        srv.stop()
    assert res.finished and res.sent == 400 and res.received == 400 and res.lost == 0 and res.loss_pct == 0
    st = res.stats()
    assert 0 < st["min"] <= st["p50"] <= st["p99"] <= st["max"] and np.isfinite(st["jitter"])
    assert res.throughput_mbps()[0] == pytest.approx(400 * 200 * 8 / res.elapsed_s / 1e6)
    assert progress and progress[-1] is res and res.diagnosis() == ""


def test_echo_reports_missing_echo_server():
    from spt.sources.echo_test import EchoConfig, EchoTest

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:           # boş port bul, sonra kapat
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    res = EchoTest(EchoConfig("127.0.0.1", port, 64, 200.0, 10, 0.2)).run()
    assert res.received == 0 and res.loss_pct == 100.0
    assert "yankı sunucusu gerekiyor" in res.diagnosis()


def test_echo_config_validation():
    from spt.sources.echo_test import EchoConfig

    for bad in (EchoConfig(""), EchoConfig("1.2.3.4", payload=4), EchoConfig("1.2.3.4", rate_hz=0)):
        with pytest.raises(ValueError):
            bad.validate()


# ------------------------------------------------------------------------------------------------ alıcı + ağ
def test_receiver_reports_port_timing_and_socket_buffer():
    from spt.sources.receiver import ReceiverConfig, _serve

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    commands, events = queue.Queue(), queue.Queue()
    th = threading.Thread(target=_serve, args=(ReceiverConfig("127.0.0.1", port, socket_buffer_mb=4, ring_seconds=1.0),
                                               commands, events.put), daemon=True)
    th.start()
    deadline = time.time() + 10
    while time.time() < deadline and events.get(timeout=10)[0] != "state":
        pass
    desc = _desc()
    tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    tx.bind(("127.0.0.1", 0))
    tx_port = tx.getsockname()[1]
    data = np.zeros((100, 1), np.int16)
    tx.sendto(build_descriptor(desc), ("127.0.0.1", port))
    for i in range(200):
        tx.sendto(build_data(desc, i, time.perf_counter_ns(), i * 100, data), ("127.0.0.1", port))
        time.sleep(0.001)
    found = None
    deadline = time.time() + 10
    while time.time() < deadline and found is None:
        ev = events.get(timeout=10)
        if ev[0] == "stats" and ev[2] and ev[2][0].packets == 200:
            found = ev
    commands.put(("stop",))
    th.join(5)
    tx.close()
    assert found is not None
    session, (st,) = found[1], found[2]
    assert session["socket_buffer"] > 0 and session["address"] == f"127.0.0.1:{port}"
    assert st.port == tx_port and st.timing.intervals == 199
    assert 500 < st.timing.mean_us < 50_000                               # ≈ 1 ms + Windows uyku adımı
    assert math.isfinite(st.timing.pdv_us)


def test_netinfo_is_safe_and_matches_subnets():
    from spt.sources import netinfo

    items = netinfo.interfaces()
    assert isinstance(items, list)
    lo = netinfo.interface_for("127.0.0.1", items)
    assert lo is None or lo.kind == netinfo.LOOPBACK
    nic = netinfo.NetInterface("Ethernet 2", "Realtek GbE", netinfo.ETHERNET, True, 100_000_000, 1500,
                               (("192.168.172.99", 24),))
    assert nic.contains("192.168.172.5") and not nic.contains("192.168.173.5") and not nic.contains("x")
    assert netinfo.interface_for("192.168.172.5", [nic]) is nic
    assert nic.speed_text == "100 Mbit/s" and netinfo.format_link_speed(None) == "bilinmiyor"
    virt = netinfo.NetInterface("vEthernet", "Hyper-V Virtual Ethernet Adapter", netinfo.ETHERNET, True,
                                10_000_000_000, 1500, (("172.24.112.1", 20),))
    assert virt.virtual and netinfo.interface_for(None, [virt, nic]) is nic
