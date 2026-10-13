"""UDP alıcı süreci.

Arayüzden ayrı bir süreçte çalışır; böylece grafik çizimi alımı yavaşlatmaz (Python GIL'i paylaşılmaz).
Örnekler paylaşımlı belleğe yazılır, durum bilgisi olay kuyruğuyla arayüze gönderilir.
Bu modül Qt import etmez.
"""
from __future__ import annotations

import logging
import queue
import select
import socket
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from spt.core.stream_store import StreamStore
from spt.protocol.format import UDP_PORT
from spt.protocol.recording import SptrWriter

COMMAND_INTERVAL = 0.1   # s
STATS_INTERVAL = 0.25    # s


@dataclass(frozen=True)
class ReceiverConfig:
    address: str = "0.0.0.0"
    port: int = UDP_PORT
    socket_buffer_mb: int = 64
    ring_seconds: float = 10.0
    max_ring_mb: int = 256


def run_udp_receiver(cfg: ReceiverConfig, commands, events) -> None:
    """Süreç giriş noktası. `commands`/`events`: multiprocessing kuyrukları."""
    emit = events.put
    try:
        _serve(cfg, commands, emit)
    except Exception:  # noqa: BLE001 — süreç içindeki her hata arayüze iletilmeli
        emit(("error", "UDP alıcı beklenmedik şekilde durdu:\n" + traceback.format_exc()))
    finally:
        emit(("stopped",))


def _open_socket(cfg: ReceiverConfig, emit) -> socket.socket | None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, cfg.socket_buffer_mb << 20)
    try:
        sock.bind((cfg.address, cfg.port))
    except OSError as exc:
        sock.close()
        emit(("error", f"UDP {cfg.address}:{cfg.port} açılamadı: {exc.strerror or exc}. "
                       "Port başka bir program tarafından kullanılıyor olabilir."))
        return None
    sock.setblocking(False)
    return sock


def rx_clock() -> tuple[int, Callable[[], int]]:
    """Alım saati: duvar saatine bir kez eşlenmiş yüksek çözünürlüklü monoton sayaç (ns). `time.time_ns` Python
    3.12 / Windows'ta ≈ 1 ms adımlıdır (GetSystemTimeAsFileTime); paket arası süre ölçümü µs çözünürlük ister.
    Dönüş: (taban, perf_counter_ns) — zaman = taban + perf_counter_ns()."""
    perf = time.perf_counter_ns
    return time.time_ns() - perf(), perf


def _serve(cfg: ReceiverConfig, commands, emit) -> None:
    sock = _open_socket(cfg, emit)
    if sock is None:
        return
    actual_buffer = sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
    base, perf_ns = rx_clock()
    store = StreamStore(emit, base + perf_ns(), cfg.ring_seconds, cfg.max_ring_mb << 20)
    emit(("state", "listening", f"UDP {cfg.address}:{cfg.port}"))
    emit(("log", logging.INFO, f"UDP {cfg.address}:{cfg.port} dinleniyor (soket tamponu {actual_buffer >> 20} MB)"))

    buf = bytearray(65536)
    mv = memoryview(buf)
    recv_into = sock.recvfrom_into
    handle = store.handle_datagram
    clock = time.perf_counter
    next_cmd = next_stats = clock()
    prev = (0, 0, clock())
    recorder: SptrWriter | None = None
    received = 0
    link = {"transport": "udp", "address": f"{cfg.address}:{cfg.port}", "socket_buffer": actual_buffer,
            "socket_buffer_requested": cfg.socket_buffer_mb << 20}
    try:
        while True:
            try:
                n, addr = recv_into(buf)
            except BlockingIOError:
                select.select([sock], [], [], COMMAND_INTERVAL)
                n = 0
            except ConnectionResetError:
                n = 0  # Windows: önceki bir ICMP hatası; alıcı için önemsiz
            if n:
                handle(mv, n, addr[0], base + perf_ns(), addr[1])
                received += 1
                if received & 255:
                    continue  # yoğun akışta saati her 256 datagramda bir kontrol et

            now = clock()
            if now < next_cmd:
                continue
            next_cmd = now + COMMAND_INTERVAL
            stop, recorder = _commands(commands, store, recorder, emit)
            if stop:
                break
            if now >= next_stats:
                next_stats = now + STATS_INTERVAL
                prev = _send_stats(store, recorder, prev, now, emit, base + perf_ns(), link)
    finally:
        if recorder is not None:
            recorder.close()
            emit(("record", False, str(recorder.path), recorder.bytes_written))
        store.close()
        sock.close()


def _commands(commands, store: StreamStore, recorder: SptrWriter | None, emit):
    while True:
        try:
            cmd = commands.get_nowait()
        except queue.Empty:
            return False, recorder
        name = cmd[0]
        if name == "stop":
            return True, recorder
        if name == "record_start" and recorder is None:
            try:
                recorder = SptrWriter(cmd[1])
            except OSError as exc:
                emit(("error", f"Kayıt dosyası açılamadı: {exc}"))
                continue
            now_ns = time.time_ns()
            for frame in store.descriptor_frames():
                recorder.write(memoryview(frame), now_ns)
            store.recorder = recorder
            emit(("record", True, str(recorder.path), recorder.bytes_written))
            emit(("log", logging.INFO, f"Kayıt başladı: {Path(cmd[1]).name}"))
        elif name == "record_stop" and recorder is not None:
            store.recorder = None
            recorder.close()
            emit(("record", False, str(recorder.path), recorder.bytes_written))
            emit(("log", logging.INFO,
                  f"Kayıt durdu: {recorder.path.name} ({recorder.bytes_written / 1e6:.1f} MB, "
                  f"{recorder.frames_written} çerçeve)"))
            recorder = None


def _send_stats(store: StreamStore, recorder, prev, now: float, emit, now_ns: int | None = None,
                extra: dict | None = None):
    """Oturum ve akış istatistikleri olayı. `extra`: oturuma eklenecek sabit bilgiler (dinlenen adres, soket
    tamponu); `now_ns`: alım saatiyle aynı saatte şimdiki zaman (boşta süresi için)."""
    datagrams, nbytes, t_prev = prev
    dt = max(now - t_prev, 1e-6)
    session = {
        "t": now,
        "datagrams_per_s": (store.datagrams - datagrams) / dt,
        "bytes_per_s": (store.bytes - nbytes) / dt,
        "datagrams": store.datagrams,
        "bytes": store.bytes,
        "frame_errors": store.frame_errors,
        "no_descriptor": store.no_descriptor,
        "recorded_bytes": recorder.bytes_written if recorder else 0,
        "unknown_frames": store.unknown_frames,
        **(extra or {}),
    }
    emit(("stats", session, store.stats(time.time_ns() if now_ns is None else now_ns)))
    return store.datagrams, store.bytes, now
