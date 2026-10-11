"""Aşama 1 + 3 uçtan uca: simülatör → UDP → alıcı süreci → arayüz; kayıt → oynatma birebirliği; canlı filtre.

Gerçek UDP soketi (localhost) ve gerçek alt süreçler kullanılır. Ayarlar geçici INI dosyasındadır.
"""
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication  # noqa: E402

from spt.dsp import filters as F  # noqa: E402
from spt.dsp.convert import physical_column  # noqa: E402
from spt.dsp.spectrum import amplitude_spectrum  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = re.compile(r"BİTTİ · (\d+) çerçeve · (\d+) örnek · .* yapay kayıp (\d+) örnek")


@pytest.fixture(scope="module")
def app():
    a = QApplication.instance() or QApplication([])
    from spt.gui.theme import ThemeManager
    ThemeManager(a).apply()
    return a


@pytest.fixture
def win(app, tmp_path, monkeypatch):
    monkeypatch.setenv("SPT_SETTINGS_FILE", str(tmp_path / "settings.ini"))
    from spt.gui.main_window import MainWindow
    from spt.gui.theme import ThemeManager

    w = MainWindow(ThemeManager(app))
    w.resize(1600, 960)
    w.show()
    yield w
    w.close()
    w.deleteLater()
    pump(app, 0.2)


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


def free_udp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def run_simulator(port: int, *args: str) -> tuple[int, int]:
    """Simülatörü çalıştırır; (gönderilen örnek, yapay kayıp örnek) döndürür."""
    out = subprocess.run([sys.executable, "-m", "spt.sources.simulator", "--port", str(port), *args],
                         cwd=ROOT, capture_output=True, text=True, encoding="utf-8", timeout=60)
    match = SUMMARY.search(out.stdout)
    assert match, out.stdout + out.stderr
    return int(match.group(2)), int(match.group(3))


def test_udp_live_loss_record_and_replay_are_identical(app, win, tmp_path):
    port = free_udp_port()
    win.sources_panel.address.setText("127.0.0.1")
    win.sources_panel.port.setValue(port)
    win.registry["conn.listen"].setChecked(True)
    win._toggle_listen(True)
    assert wait_for(app, lambda: win.sources.state == "listening")

    recording = tmp_path / "kayit.sptr"
    win.sources.send("record_start", str(recording))
    assert wait_for(app, lambda: win.registry["stream.record"].isChecked())

    sent, dropped = run_simulator(port, "--signal", "sine", "--freq", "1000", "--amp", "1.5", "--fs", "20000",
                                  "--channels", "2", "--type", "int16", "--duration", "2", "--drop", "0.02",
                                  "--crc", "--name", "e2e")
    assert dropped > 0
    assert wait_for(app, lambda: win.sources.streams and
                    next(iter(win.sources.streams.values())).latest_index() == sent + dropped)
    handle = next(iter(win.sources.streams.values()))
    assert wait_for(app, lambda: (r := win.sources.rates.get(handle.key)) is not None and r.stats is not None
                    and r.stats.samples == sent)
    stats = win.sources.rates[handle.key].stats
    assert stats.lost_samples == dropped and stats.duplicates == 0 and stats.restarts == 0

    # Arayüz canlı grafikleri çiziyor; 1 kHz, 1.5 birim sinüs
    win._show_view(0)
    pump(app, 0.3)
    x, _ = win.live_view._time_curves[(handle.key, 0, 0)].getData()
    assert x is not None and len(x) > 100
    rng = handle.ring.latest_clean_range(4096)
    _, raw = handle.ring.read(*rng)
    freqs, mag = amplitude_spectrum(physical_column(raw[:, 0], handle.desc.sample_type, handle.desc.channels[0]),
                                    handle.fs, "Flat-top")
    assert freqs[np.argmax(mag)] == pytest.approx(1000.0, abs=handle.fs / 4096)
    assert mag.max() == pytest.approx(20 * np.log10(1.5), abs=0.1)

    live = win.freeze()[0]
    win.sources.send("record_stop")
    assert wait_for(app, lambda: not win.registry["stream.record"].isChecked())
    win.sources.stop()

    # Kaydı en yüksek hızda oynat: tampondaki veri canlı akıştakiyle birebir aynı olmalı (kayıplar dahil)
    finished = []
    win.sources.replay_progress.connect(lambda pos, dur, paused, done: done and finished.append(True))
    win.sources_panel.replay.speed.setCurrentIndex(win.sources_panel.replay.speed.count() - 1)  # Maksimum
    win.start_replay(recording)
    assert wait_for(app, lambda: bool(finished))
    assert wait_for(app, lambda: win.sources.streams and
                    next(iter(win.sources.streams.values())).latest_index() == sent + dropped)
    replayed = win.freeze()[0]
    assert np.array_equal(replayed.ring.data, live.ring.data)
    assert replayed.ring.gaps == live.ring.gaps and replayed.ring.base == live.ring.base
    assert sum(e - s for s, e in live.ring.gaps) == dropped
    win.sources.stop()


def test_live_filter_on_udp_stream_attenuates_out_of_band_tone(app, win):
    port = free_udp_port()
    win.sources_panel.address.setText("127.0.0.1")
    win.sources_panel.port.setValue(port)
    win._toggle_listen(True)
    assert wait_for(app, lambda: win.sources.state == "listening")
    fv = win.filter_view
    fv.set_spec(F.FilterSpec(family="iir", method="ellip", band="lowpass", order=6, f1=500.0, ripple_db=0.5,
                             atten_db=70.0, fs=10_000.0))
    fv.request_apply()
    win._show_view(0)
    proc = subprocess.Popen([sys.executable, "-m", "spt.sources.simulator", "--port", str(port), "--signal", "sine",
                             "--freq", "2000", "--amp", "1.0", "--fs", "10000", "--channels", "1", "--type",
                             "float32", "--duration", "3"], cwd=ROOT, stdout=subprocess.DEVNULL)
    try:
        assert wait_for(app, lambda: win.sources.streams and
                        next(iter(win.sources.streams.values())).latest_index() > 15_000)
        handle = next(iter(win.sources.streams.values()))
        lv = win.live_view
        lv.refresh(force=True)
        end = handle.latest_index()
        y = win.live_filter.read_clean(handle.key, 0, end - 8192, end)
        assert y is not None
        steady = y[-4096:]
        _, raw = handle.ring.read(end - 4096, end)
        assert np.std(raw[:, 0]) == pytest.approx(1 / np.sqrt(2), rel=0.01)  # filtre gerçek 2 kHz sinüsü gördü
        attenuation = 20 * np.log10(max(np.sqrt(2) * np.std(steady), 1e-15))  # 2 kHz durdurma bandında: ≤ −70 dB
        assert -120.0 < attenuation < -69.0, attenuation  # tam sıfır çıkış da hata (sinyal filtreye ulaşmamış)
        x, _ = lv._filt_curves[(handle.key, 0, 0)].getData()
        assert x is not None and len(x) > 100
    finally:
        proc.wait(timeout=30)
        win.sources.stop()


def test_simulator_dialog_streams_into_receiver(app, win):
    # Arayüzdeki Simülatör penceresi alt süreci boru üzerinden okur: Windows kod sayfasında (cp1254) "→" gibi
    # karakterler simülatörü çökertmemeli; otomatik dinleme ile veri alıcıya ulaşmalı
    win._show_simulator()
    dlg = win._simulator
    dlg.port.setValue(free_udp_port())
    dlg.auto_listen.setChecked(True)
    dlg.toggle()
    try:
        assert wait_for(app, lambda: win.sources.streams and
                        next(iter(win.sources.streams.values())).latest_index() > 5_000, timeout=60)
        assert dlg.running
        assert wait_for(app, lambda: "MB/s" in dlg.status.text(), timeout=30)
        assert "Hata" not in dlg.status.text()
    finally:
        dlg.stop()
        win.sources.stop()
        dlg.close()
