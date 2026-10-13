"""SPTG istemcisi (spt.protocol.sptg) ve "Karta Gönder" penceresi: yerel UDP'de çalışan sahte kartla sınanır.

Sahte kart protokolün kart tarafını (stm32f4_examples/001_signal_generator, signal_remote.c) Python'da taklit eder:
BEGIN doğrulaması, DATA yerleştirme, START'ta CRC denetimi ve yinelenen isteğe önceki yanıtı verme.
"""
import os
import socket
import threading
import time
import zlib
from dataclasses import replace

import pytest

from spt.dsp import siggen as sg
from spt.protocol import sptg

CLOCK_HZ = 90_000_000
CAPACITY = 32768
MAX_CHUNK = 1460


class FakeBoard:
    """Tek iş parçacıklı sahte kart. `drop_replies`: atılacak yanıt sayısı; `corrupt_data`: bozulacak DATA sayısı."""

    def __init__(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind(("127.0.0.1", 0))
        self.socket.settimeout(0.05)
        self.port = self.socket.getsockname()[1]
        self.table = bytearray(CAPACITY)
        self.begin = None
        self.loading = False
        self.state, self.source, self.active_crc = 0, 0, 0
        self.played: list[tuple[tuple, bytes]] = []
        self.requests: list[int] = []
        self.drop_replies = 0
        self.corrupt_data = 0
        self._cache: tuple[int, int, bytes] | None = None
        self._running = True
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def close(self):
        self._running = False
        self._thread.join(timeout=2)
        self.socket.close()

    def _reply(self, message, sequence, status):
        return (sptg.HEADER.pack(sptg.MAGIC, sptg.VERSION, message | sptg.REPLY_FLAG, sequence)
                + sptg.REPLY.pack(status, self.state, self.source, CLOCK_HZ, CAPACITY, MAX_CHUNK, 0, self.active_crc))

    def _handle(self, message, body):
        if message == sptg.Message.INFO:
            return sptg.Status.OK
        if message == sptg.Message.BEGIN:
            if len(body) != sptg.BEGIN.size:
                return sptg.Status.MESSAGE_ERR
            output, channels, bits, align, _psc, arr, clock, count, crc = fields = sptg.BEGIN.unpack(body)
            if output not in (1, 2, 3) or channels not in (1, 2) or (channels == 2 and output != 3) \
                    or bits not in (8, 12) or align > 1 or (bits == 8 and align) or not 1 <= count <= 65535 or arr == 0:
                return sptg.Status.ARG_ERR
            if clock != CLOCK_HZ:
                return sptg.Status.CLOCK_ERR
            size = count * channels * (1 if bits == 8 else 2)
            if size > CAPACITY:
                return sptg.Status.SIZE_ERR
            self.begin, self.loading = (fields, size, crc), True
            self.state, self.source, self.active_crc = 5, 0, 0
            return sptg.Status.OK
        if message == sptg.Message.DATA:
            if not self.loading:
                return sptg.Status.STATE_ERR
            (offset,) = sptg.DATA_PREFIX.unpack_from(body)
            chunk = bytearray(body[sptg.DATA_PREFIX.size:])
            if offset + len(chunk) > self.begin[1]:
                return sptg.Status.SIZE_ERR
            if self.corrupt_data:
                self.corrupt_data -= 1
                chunk[0] ^= 0x40
            self.table[offset:offset + len(chunk)] = chunk
            return sptg.Status.OK
        if message == sptg.Message.START:
            if not self.loading:
                return sptg.Status.STATE_ERR
            fields, size, crc = self.begin
            data = bytes(self.table[:size])
            if zlib.crc32(data) & 0xFFFFFFFF != crc:
                return sptg.Status.CRC_ERR
            self.loading = False
            self.state, self.source, self.active_crc = 1, 2, crc
            self.played.append((fields, data))
            return sptg.Status.OK
        if message == sptg.Message.STOP:
            self.loading, self.state, self.source, self.active_crc = False, 5, 0, 0
            return sptg.Status.OK
        return sptg.Status.MESSAGE_ERR

    def _serve(self):
        while self._running:
            try:
                datagram, address = self.socket.recvfrom(4096)
            except (TimeoutError, OSError):
                continue
            if len(datagram) < sptg.HEADER.size or datagram[:4] != sptg.MAGIC:
                continue
            _magic, _version, message, sequence = sptg.HEADER.unpack_from(datagram)
            self.requests.append(message)
            if self._cache and message != sptg.Message.INFO and self._cache[:2] == (message, sequence):
                reply = self._cache[2]
            else:
                reply = self._reply(message, sequence, self._handle(message, datagram[sptg.HEADER.size:]))
                if message != sptg.Message.INFO:
                    self._cache = (message, sequence, reply)
            if self.drop_replies:
                self.drop_replies -= 1
                continue
            self.socket.sendto(reply, address)


@pytest.fixture
def board():
    fake = FakeBoard()
    yield fake
    fake.close()


def _table(bits=12, align="right", kind="sine", freq=1000.0, fs=48_000.0, kind2=None, max_len=4096):
    dac = sg.DacSpec(8, 3.3, word_bits=8) if bits == 8 else sg.DacSpec(12, 3.3, align=align)
    spec = sg.WaveSpec(kind=kind, freq=freq, amplitude=1.5, offset=1.65)
    if kind2:
        return sg.make_table(spec, replace(dac, channels=2), fs, max_len=max_len, spec2=replace(spec, kind=kind2))
    return sg.make_table(spec, dac, fs, max_len=max_len)


def test_payload_is_little_endian_dac_words():
    table = _table()
    payload = sptg.payload_from_table(table)
    assert payload.sample_count == 48 and payload.channel_count == 1 and payload.dac_bits == 12
    assert len(payload.data) == 96 and not payload.left_aligned
    assert int.from_bytes(payload.data[:2], "little") == int(table.flat_codes[0]) == 2048
    stereo = sptg.payload_from_table(_table(kind2="sawtooth"))
    assert stereo.channel_count == 2 and len(stereo.data) == 48 * 2 * 2
    assert int.from_bytes(stereo.data[2:4], "little") == int(_table(kind2="sawtooth").flat_codes[1])
    eight = sptg.payload_from_table(_table(bits=8))
    assert eight.dac_bits == 8 and len(eight.data) == 48
    assert sptg.payload_from_table(_table(align="left")).left_aligned


@pytest.mark.parametrize("make", [
    lambda: sg.make_table(sg.WaveSpec(), sg.DacSpec(12, 3.3), 48_000.0, mode="dds"),
    lambda: sg.make_table(sg.WaveSpec(amplitude=0.5), sg.DacSpec(16, 1.0, out_min=-1.0, out_max=1.0,
                                                                 coding="twos_complement"), 48_000.0),
    lambda: sg.make_table(sg.WaveSpec(offset=1.65), sg.DacSpec(12, 3.3, word_bits=32), 48_000.0),
])
def test_payload_rejects_tables_the_internal_dac_cannot_play(make):
    with pytest.raises(sptg.SptgError):
        sptg.payload_from_table(make())


def test_play_sends_table_and_reports_realised_rate(board):
    table = _table(freq=997.0)           # 3996 örnek: birden fazla DATA isteği
    result = sptg.send_table("127.0.0.1", table, sptg.Output.DAC2, port=board.port)
    (fields, data), = board.played
    timer = sg.timer_config(table.fs, CLOCK_HZ, 16)
    assert fields[:8] == (2, 1, 12, 0, timer.psc, timer.arr, CLOCK_HZ, table.N)
    assert data == sptg.payload_from_table(table).data
    assert board.requests.count(sptg.Message.DATA) == -(-len(data) // 1400)
    assert result.output is sptg.Output.DAC2 and result.info.player_state == sptg.PLAYER_RUNNING
    assert result.fs == pytest.approx(timer.fs_real) and result.f == pytest.approx(table.f_real * timer.fs_real / table.fs)


def test_stereo_table_always_uses_both_outputs(board):
    result = sptg.send_table("127.0.0.1", _table(kind2="triangle"), sptg.Output.DAC1, port=board.port)
    assert result.output is sptg.Output.BOTH and board.played[0][0][:2] == (3, 2)


def test_lost_replies_are_retried_without_replaying(board):
    board.drop_replies = 3
    with sptg.SptgClient("127.0.0.1", board.port, timeout=0.05, retries=5) as client:
        client.play(sptg.payload_from_table(_table()), sptg.Output.BOTH)
    assert len(board.played) == 1                       # yinelenen START yeniden işlenmedi
    assert len(board.requests) > 4                      # INFO, BEGIN, DATA, START + yinelemeler


def test_corrupted_data_is_resent_once(board):
    board.corrupt_data = 1
    payload = sptg.payload_from_table(_table())
    with sptg.SptgClient("127.0.0.1", board.port) as client:
        result = client.play(payload, sptg.Output.DAC1)
    assert board.played[0][1] == payload.data and result.info.table_crc32 == payload.crc32
    assert board.requests.count(sptg.Message.BEGIN) == 2


def test_errors_are_reported_in_turkish(board):
    with sptg.SptgClient("127.0.0.1", board.port) as client:
        with pytest.raises(sptg.SptgError, match="yükleme başlatılmamış") as excinfo:
            client.request(sptg.Message.START)
        assert excinfo.value.status == sptg.Status.STATE_ERR
        big = sptg.TablePayload(bytes(CAPACITY + 2), CAPACITY // 2 + 1, 1, 12, False, 48_000.0, 0.0)
        with pytest.raises(sptg.SptgError, match="ayrılmış"):
            client.play(big)
        assert client.stop().state_text == "durduruldu"
    with sptg.SptgClient("127.0.0.1", board.port, timeout=0.02, retries=1) as client:
        board.drop_replies = 10
        with pytest.raises(sptg.SptgError, match="yanıt vermiyor"):
            client.info()


def test_board_send_dialog_plays_current_table(board, tmp_path, monkeypatch):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setenv("SPT_SETTINGS_FILE", str(tmp_path / "settings.ini"))
    from PyQt6.QtWidgets import QApplication

    from spt.gui.dialogs.board_send_dialog import BoardSendDialog
    from spt.gui.theme import ThemeManager

    app = QApplication.instance() or QApplication([])
    table = _table(kind="square")
    dialog = BoardSendDialog(ThemeManager(app).tokens, lambda: table)
    assert dialog.host.text() == "192.168.172.98" and dialog.port.value() == sptg.UDP_PORT
    dialog.host.setText("127.0.0.1")
    dialog.port.setValue(board.port)
    dialog.output.setCurrentIndex(dialog.output.findData(int(sptg.Output.BOTH)))

    def wait(text):
        end = time.perf_counter() + 5
        while text not in dialog.status.text() and time.perf_counter() < end:
            app.processEvents()
            time.sleep(0.005)
        assert text in dialog.status.text(), dialog.status.text()

    dialog.send()
    assert not dialog.send_button.isEnabled()
    wait("Çalıyor")
    assert dialog.send_button.isEnabled() and board.played[0][0][0] == 3
    assert board.played[0][1] == sptg.payload_from_table(table).data
    dialog.stop()
    wait("durduruldu")
    dialog.close()
    dialog.deleteLater()
