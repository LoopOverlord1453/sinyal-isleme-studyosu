"""Sinyal Test Uygulaması: üreteçlerin doğruluğu, SPTF çerçeveleri, gerçek UDP ile uçtan uca gönderim ve bozulmalar,
arayüz (ekransız) ve Stüdyo'dan açılış.

Uçtan uca testlerde motor bu süreçte (ayrı süreç değil) çalışır, alıcı tarafı spt.core.stream_store'dur; bir test
ayrıca motoru gerçek alt süreçte (spawn) çalıştırır.
"""
import multiprocessing as mp
import os
import queue
import socket
import threading
import time

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import Qt  # noqa: E402

from spt.core.stream_store import StreamStore  # noqa: E402
from spt.dsp.convert import physical_column  # noqa: E402
from spt.protocol.parser import DataFrame, DescriptorFrame, MarkerFrame, decode_frames, samples_view  # noqa: E402
from spt.sender import signals as sg  # noqa: E402
from spt.sender.engine import Engine, run_engine  # noqa: E402


def _tone_freq(x: np.ndarray, fs: float) -> tuple[float, float]:
    """En güçlü bileşenin frekansı (Hann, parabolik) ve genliği (flat-top: bin arası kayıp ≈ 0)."""
    from scipy.signal import get_window

    n = len(x)
    x = x - x.mean()
    w = np.hanning(n)
    spec = np.abs(np.fft.rfft(x * w)) * 2 / w.sum()
    i = int(np.argmax(spec[1:])) + 1
    a, b, c = np.log(spec[i - 1:i + 2] + 1e-30)
    p = 0.5 * (a - c) / (a - 2 * b + c)
    ft = get_window("flattop", n)
    amp = np.abs(np.fft.rfft(x * ft)) * 2 / ft.sum()
    return (i + p) * fs / n, float(amp[i - 3:i + 4].max())


# ============================================================================ üreteçler
@pytest.mark.parametrize("kind", list(sg.SIGNAL_KINDS))
def test_blocks_are_continuous(kind):
    """Bloklar hâlinde üretim tek seferlik üretimle birebir aynı (gürültü dahil): dikiş yok."""
    st = sg.StreamSpec(1, "a", 8000.0, "float64", channels=[sg.ChannelSpec("x", kind=kind, freq=123.4, amp=1.0,
                                                                           noise=0.01)])
    one = sg.StreamGen(st).physical(1000, 3000)
    gen = sg.StreamGen(st)
    parts = np.concatenate([gen.physical(1000, 700), gen.physical(1700, 1100), gen.physical(2800, 1200)])
    np.testing.assert_allclose(parts, one, rtol=0, atol=1e-12)
    assert np.all(np.isfinite(one))


@pytest.mark.parametrize("freq", [50.0, 123.4, 997.0])
def test_sine_frequency_amplitude_offset_and_phase(freq):
    fs = 10_000.0
    st = sg.StreamSpec(1, "a", fs, "float64", channels=[
        sg.ChannelSpec("a", freq=freq, amp=2.0, offset=0.5),
        sg.ChannelSpec("b", freq=freq, amp=2.0, phase_deg=90.0)])
    x = sg.StreamGen(st).physical(0, 20_000)
    f, a = _tone_freq(x[:, 0], fs)
    assert f == pytest.approx(freq, abs=0.05) and a == pytest.approx(2.0, abs=0.02)
    assert x[:, 0].mean() == pytest.approx(0.5, abs=0.01)
    assert x[0, 1] == pytest.approx(2.0, abs=1e-9)                # 90° faz: cos ile başlar


def test_kind_specific_shapes():
    fs, n = 10_000.0, 10_000
    mk = sg.ChannelSpec
    st = sg.StreamSpec(1, "a", fs, "float64", channels=[
        mk("kare", kind="square", freq=10.0, amp=1.0, duty=0.3),
        mk("darbe", kind="pulse", freq=10.0, amp=2.0, duty=0.1),
        mk("dc", kind="dc", amp=5.0, offset=1.65),
        mk("chirp", kind="chirp", freq=100.0, f1=1000.0, sweep_s=1.0, amp=1.0),
        mk("am", kind="am", freq=1000.0, mod_freq=10.0, depth=0.5, amp=1.0),
        mk("ekg", kind="ecg", freq=1.0, amp=1.0),
        mk("rampa", kind="ramp", freq=2.0, amp=3.0),
        mk("ton", kind="multitone", freq=50.0, harmonics=3, amp=1.0),
    ])
    x = sg.StreamGen(st).physical(0, n)
    assert np.mean(x[:, 0] > 0) == pytest.approx(0.3, abs=1e-3)              # görev oranı
    assert set(np.unique(x[:, 1])) == {0.0, 2.0} and np.mean(x[:, 1] > 0) == pytest.approx(0.1, abs=1e-3)
    assert np.all(x[:, 2] == 1.65)                                           # DC: yalnız ofset
    inst = np.diff(np.unwrap(np.angle(np.fft.ifft(np.fft.fft(x[:, 3]) * (np.arange(n) < n // 2) * 2))))
    assert inst[1000] * fs / (2 * np.pi) == pytest.approx(100 + 900 * 0.1, rel=0.02)   # 0.1 s'de anlık frekans
    assert 0.9 < x[:, 4].max() <= 1.0 + 1e-12                                # AM tepe ≤ genlik (örneklenmiş)
    assert x[:, 5].max() == pytest.approx(1.0, abs=1e-3)                     # EKG R tepesi = genlik
    assert x[:, 6].min() == 0.0 and x[:, 6].max() < 3.0
    f, _ = _tone_freq(x[:, 7], fs)
    assert f == pytest.approx(50.0, abs=0.05)


def test_integer_types_round_trip_through_descriptor_scaling():
    """Tamsayı tipte ölçek / ofset tanımlayıcıda taşınır: alıcı fiziksel değeri nicemleme hatası içinde geri bulur."""
    for tname in ("int16", "uint16", "int32", "int24", "uint8"):
        st = sg.StreamSpec(1, "a", 1000.0, tname, channels=[
            sg.ChannelSpec("acc_z", kind="sine", freq=3.0, amp=0.5, offset=9.81, noise=0.01),
            sg.ChannelSpec("dc", kind="dc", offset=-3.3)])
        gen = sg.StreamGen(st)
        phys = sg.StreamGen(st).physical(0, 500)
        wire = gen.wire(0, 500)
        for c, info in enumerate(gen.desc.channels):
            back = physical_column(wire[:, c], gen.desc.sample_type, info)
            lsb = info.scale
            assert np.max(np.abs(back - phys[:, c])) <= lsb * 0.51 + 1e-9, tname


def test_complex_iq_stream():
    st = sg.StreamSpec(7, "iq", 250_000.0, "cint16", channels=[sg.ChannelSpec("rx", kind="iq", freq=12_500.0,
                                                                             amp=0.8)])
    gen = sg.StreamGen(st)
    x = gen.physical(0, 4096)
    assert np.iscomplexobj(x) and np.allclose(np.abs(x[:, 0]), 0.8)
    w = gen.wire(0, 4096)
    assert w.dtype == gen.desc.sample_type.dtype


def test_scenarios_and_json_round_trip():
    names = sg.scenario_names()
    assert names[-1] == sg.ALL and "IMU 6 eksen 1 kHz" in names
    allspec = sg.scenario(sg.ALL)
    ids = [s.stream_id for s in allspec]
    assert len(ids) == len(set(ids)) >= 7                                    # akış kimlikleri benzersiz
    for name in names:
        for st in sg.scenario(name):
            sg.StreamGen(st).wire(0, 64)                                     # her senaryo üretilebilir
    cfg = sg.default_config()
    cfg.impairments.loss_pct = 2.5
    back = sg.config_from_dict(sg.config_to_dict(cfg))
    assert back == cfg
    junk = {"host": " ", "port": 99999, "streams": [{"sample_type": "yok", "fs": -5, "layout": "?",
                                                      "channels": [{"kind": "bilinmeyen", "duty": 7}]}],
            "impairments": {"loss_pct": 300}, "fazla": 1}
    cfg2 = sg.config_from_dict(junk)
    st = cfg2.streams[0]
    assert (cfg2.host, cfg2.port, cfg2.impairments.loss_pct) == ("127.0.0.1", 65535, 100.0)
    assert (st.sample_type, st.layout, st.channels[0].kind, st.channels[0].duty) == ("int16", "interleaved",
                                                                                   "sine", 1.0)
    assert st.fs > 0


# ============================================================================ çerçeveler
class _Capture:
    """Yerel UDP soketi: gelen datagramları toplar (ayrı iş parçacığında)."""

    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 32 << 20)
        self.sock.bind(("127.0.0.1", 0))
        self.sock.settimeout(0.05)
        self.port = self.sock.getsockname()[1]
        self.datagrams: list[tuple[bytes, int, int]] = []    # (veri, kaynak portu, alım ns)
        self._stop = False
        self._th = threading.Thread(target=self._run, daemon=True)
        self._th.start()

    def _run(self):
        while not self._stop:
            try:
                data, addr = self.sock.recvfrom(65536)
            except (socket.timeout, ConnectionResetError):
                continue
            self.datagrams.append((data, addr[1], time.time_ns()))

    def close(self):
        self._stop = True
        self._th.join(1)
        self.sock.close()


def _config(port, streams, **imp) -> sg.SenderConfig:
    return sg.SenderConfig("127.0.0.1", port, streams=streams, impairments=sg.Impairments(**imp))


def test_frames_are_valid_sptf_and_samples_match_generator():
    cap = _Capture()
    try:
        st_i = sg.StreamSpec(1, "il", 2000.0, "int16", channels=[sg.ChannelSpec("a", freq=10.0),
                                                                sg.ChannelSpec("b", kind="square", freq=5.0)])
        st_p = sg.StreamSpec(2, "pl", 1000.0, "float32", layout="planar", crc=True, marker_every=0.1,
                             channels=[sg.ChannelSpec("x", kind="sawtooth", freq=2.0),
                                       sg.ChannelSpec("y", kind="dc", offset=1.5)])
        eng = Engine(_config(cap.port, [st_i, st_p]))
        eng.run(duration=0.6)
        eng.close()
        time.sleep(0.1)
    finally:
        cap.close()
    descs, data, markers = {}, {1: [], 2: []}, []
    for raw, _, _ in cap.datagrams:
        for fr in decode_frames(raw):                    # CRC ve biçim burada doğrulanır
            if isinstance(fr, DescriptorFrame):
                descs[fr.header.stream_id] = fr.descriptor
            elif isinstance(fr, DataFrame):
                data[fr.header.stream_id].append(fr)
            elif isinstance(fr, MarkerFrame):
                markers.append(fr)
    assert set(descs) == {1, 2} and descs[2].layout.name == "PLANAR" and descs[1].channels[1].name == "b"
    assert len(markers) >= 2 and markers[0].text.startswith("işaret")
    for sid, spec in ((1, st_i), (2, st_p)):
        frames = sorted(data[sid], key=lambda f: f.first_sample)
        firsts = [f.first_sample for f in frames]
        assert firsts[0] == 0 and all(b - a == frames[0].sample_count for a, b in zip(firsts, firsts[1:]))
        desc = descs[sid]
        wire = np.concatenate([samples_view(f.payload, 0, f.sample_count, desc) for f in frames])
        expected = sg.StreamGen(spec).wire(0, len(wire))
        np.testing.assert_array_equal(wire, expected)
        assert 0.5 * spec.fs <= len(wire) <= 0.75 * spec.fs                # ≈ 0.6 s gerçek zaman
        assert frames[0].sample_count == sg.frame_samples(spec)            # en çok 10 ms'lik paket
        assert frames[0].sample_count == round(spec.fs * 0.01)


def _receive(cap: _Capture) -> StreamStore:
    events = []
    store = StreamStore(events.append, time.time_ns(), ring_seconds=5.0)
    for raw, port, rx in cap.datagrams:
        mv = memoryview(raw)
        store.handle_datagram(mv, len(raw), "127.0.0.1", rx, port)
    return store


def test_end_to_end_real_time_rate_and_no_loss():
    cap = _Capture()
    try:
        streams = sg.scenario("IMU 6 eksen 1 kHz") + sg.scenario("Ses stereo 48 kHz")
        eng = Engine(_config(cap.port, streams))
        t0 = time.perf_counter()
        eng.run(duration=1.5)
        elapsed = time.perf_counter() - t0
        stats = eng.stats()
        eng.close()
        time.sleep(0.1)
    finally:
        cap.close()
    store = _receive(cap)
    got = {s.stream_id: s for s in store.stats(time.time_ns())}
    for spec in streams:
        s = got[spec.stream_id]
        assert s.lost_samples == 0 and s.duplicates == 0
        # gerçek zamanlı hız: gönderilen ≈ fs · süre (en çok bir paket eksik)
        assert abs(s.samples - spec.fs * elapsed) <= sg.frame_samples(spec) + 0.02 * spec.fs * elapsed
    assert all(not r["late"] for r in stats["streams"]) and stats["errors"] == 0
    store.close()


def test_impairments_are_seen_by_receiver():
    cap = _Capture()
    try:
        spec = sg.StreamSpec(1, "boz", 200_000.0, "int16", channels=[sg.ChannelSpec("a")])
        eng = Engine(_config(cap.port, [spec], loss_pct=10.0, duplicate_pct=5.0, reorder_pct=5.0, jitter_ms=1.0))
        eng.run(duration=1.5)
        st = eng.streams[0]
        lost, dup, reo = st.lost, st.duplicated, st.reordered
        eng.close()
        time.sleep(0.1)
    finally:
        cap.close()
    store = _receive(cap)
    (s,) = store.stats(time.time_ns())
    frames = st.sent // st.spf
    assert frames > 300 and lost / frames == pytest.approx(0.10, abs=0.04) and dup > 0 and reo > 0
    # Alıcının kayıp sayacı: atılan çerçeveler (sondakiler görünmez: alıcı son çerçeveden sonraki kaybı bilemez);
    # sıra dışı gelenler önce kayıp, sonra "geç" sayılır
    assert (lost - 3) * st.spf <= s.lost_samples <= (lost + reo + 3) * st.spf
    assert s.duplicates > 0 and s.late_packets > 0
    store.close()


def test_engine_subprocess_start_stats_and_stop():
    cap = _Capture()
    ctx = mp.get_context("spawn")
    commands, events = ctx.Queue(), ctx.Queue()
    cfg = sg.config_to_dict(_config(cap.port, sg.scenario("EKG 500 Hz")))
    proc = ctx.Process(target=run_engine, args=(cfg, commands, events), daemon=True)
    proc.start()
    seen = []
    try:
        end = time.time() + 20
        while time.time() < end:
            try:
                ev = events.get(timeout=0.5)
            except queue.Empty:
                continue
            seen.append(ev[0])
            if ev[0] == "stats" and ev[1]["streams"][0]["packets"] > 3:
                break
        commands.put(("impairments", {"loss_pct": 50.0}))
        commands.put(("stop",))
        end = time.time() + 10
        while "stopped" not in seen and time.time() < end:
            try:
                seen.append(events.get(timeout=0.5)[0])
            except queue.Empty:
                pass
        proc.join(5)
    finally:
        if proc.is_alive():
            proc.kill()
        cap.close()
    assert seen[0] == "started" and "stats" in seen and seen[-1] == "stopped" and "error" not in seen
    assert any(decode_frames(d) for d, _, _ in cap.datagrams)


# ============================================================================ arayüz (ekransız)
def _pump(app, seconds):
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        app.processEvents()
        time.sleep(0.005)


def _wait(app, cond, timeout=20.0):
    end = time.perf_counter() + timeout
    while time.perf_counter() < end:
        app.processEvents()
        if cond():
            return True
        time.sleep(0.01)
    return False


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


@pytest.fixture
def sender_env(tmp_path, monkeypatch):
    monkeypatch.setenv("SPT_SENDER_SETTINGS_FILE", str(tmp_path / "sender.ini"))
    monkeypatch.setenv("SPT_SETTINGS_FILE", str(tmp_path / "studio.ini"))
    return tmp_path


def test_window_scenario_edit_save_load_and_send(app, theme, sender_env):
    from spt.sender.window import SenderWindow

    cap = _Capture()
    win = SenderWindow(theme)
    try:
        win.show()
        win.port.setValue(cap.port)
        win.host.setText("127.0.0.1")
        win.host.editingFinished.emit()
        win.apply_scenario("IMU 6 eksen 1 kHz")
        assert win.stream_table.rowCount() == 1 and win.ch_table.rowCount() == 6
        assert win.stream_table.item(0, 2).text() == "imu-6eksen"
        # kanal düzenle: frekans ve tür → yapılandırma ve önizleme
        freq = win.ch_table.cellWidget(0, 3)
        freq.setValue(7.0)
        win.ch_table.cellWidget(0, 2).setCurrentIndex(list(sg.SIGNAL_KINDS).index("square"))
        ch0 = win.cfg.streams[0].channels[0]
        assert (ch0.freq, ch0.kind) == (7.0, "square")
        assert win.extra_fields["duty"][1].isVisibleTo(win)            # karenin ek parametresi görünür
        assert not win.extra_fields["f1"][1].isVisibleTo(win)
        # akış ekle / çoğalt / sil; kimlikler benzersiz
        win.add_stream()
        win.duplicate_stream()
        ids = [s.stream_id for s in win.cfg.streams]
        assert len(ids) == 3 == len(set(ids))
        win.remove_stream()
        assert len(win.cfg.streams) == 2
        win.stream_table.item(1, 0).setCheckState(Qt.CheckState.Unchecked)   # ikinci akışı kapat
        assert [s.enabled for s in win.cfg.streams] == [True, False]
        # JSON kaydet / aç
        path = sender_env / "ayar.json"
        win.save_json(path)
        win.apply_scenario("EKG 500 Hz")
        win.load_json(path)
        assert [s.name for s in win.cfg.streams][0] == "imu-6eksen" and win.cfg.streams[0].channels[0].freq == 7.0
        # gönder: canlı durum dolar; bozulma canlı uygulanır; dur
        win.start()
        assert _wait(app, lambda: "paket/s" in win.stream_table.item(0, 8).text())
        assert win.badge.text().endswith(f"127.0.0.1:{cap.port}")
        win.loss.setValue(30.0)
        assert _wait(app, lambda: "atılan" in win.stream_table.item(0, 8).text())
        # gönderim sürerken akış düzenle: motor yeniden başlar, tanımlayıcı değiştiği için desc_rev artar
        rev = win.cfg.streams[0].rev
        win.s_fs.setValue(2000.0)
        assert _wait(app, lambda: win.cfg.streams[0].rev == rev + 1 and win.engine.running, timeout=10)

        def rates():
            return {fr.descriptor.sample_rate for raw, _, _ in list(cap.datagrams) for fr in decode_frames(raw)
                    if isinstance(fr, DescriptorFrame)}

        assert _wait(app, lambda: 2000.0 in rates(), timeout=15)    # yeni tanımlayıcı alıcıya ulaştı
        win.stop()
        assert not win.engine.running and win.badge.text().endswith("Durduruldu")
    finally:
        win.close()
        cap.close()
    assert {1000.0, 2000.0} <= rates()                              # yeniden yapılandırma alıcıya ulaştı
    # ayarlar kalıcı: yeni pencere son yapılandırmayla açılır
    again = SenderWindow(theme)
    try:
        assert again.cfg.port == cap.port and again.cfg.streams[0].fs == 2000.0 and again.loss.value() == 30.0
    finally:
        again.close()


def test_studio_opens_sender_and_receives_streams(app, theme, sender_env, monkeypatch):
    """Stüdyo: Araçlar → Sinyal Test Uygulaması kendi portuyla açar (dinleme kapalıysa açar); gönderilen akışlar
    Canlı İzleme'de ve KPI kutularında görünür."""
    from PyQt6.QtCore import QProcess

    from spt.gui import main_window as mw

    calls = []
    monkeypatch.setattr(QProcess, "startDetached", staticmethod(lambda *a: calls.append(a) or (True, 1)))
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    win = mw.MainWindow(theme)
    try:
        win.show()
        menu_items = [a.text() for a in win.menuBar().actions()]
        assert any("raçlar" in t for t in menu_items)
        win.sources_panel.address.setText("127.0.0.1")
        win.sources_panel.port.setValue(port)
        win.registry["tools.signal_sender"].trigger()
        assert win.sources.mode == "udp"                              # dinleme açıldı
        exe, args, cwd = calls[-1]
        assert args[:3] == ["-m", "spt.sender", "--studio"] and args[args.index("--port") + 1] == str(port)
        assert args[args.index("--host") + 1] == "127.0.0.1" and cwd == str(mw.ROOT)
        assert _wait(app, lambda: win.sources.state == "listening", timeout=15)
        streams = sg.scenario("IMU 6 eksen 1 kHz") + sg.scenario("EKG 500 Hz") + sg.scenario("Ses stereo 48 kHz")
        eng = Engine(_config(port, streams))
        th = threading.Thread(target=eng.run, kwargs={"duration": 3.0}, daemon=True)
        th.start()
        assert _wait(app, lambda: len(win.sources.streams) == 3, timeout=15)
        assert _wait(app, lambda: win.live_view.kpi.pps.value.text() not in ("", "0"), timeout=10)
        th.join(10)
        eng.close()
        names = sorted(h.desc.name for h in win.sources.streams.values())
        assert names == ["ekg", "imu-6eksen", "ses-stereo"]
    finally:
        win.close()
        win.deleteLater()
        _pump(app, 0.3)
