"""Arayüz tarafı kaynak yöneticisi: alıcı / oynatıcı süreçlerini başlatır, olayları Qt sinyallerine çevirir."""
from __future__ import annotations

import logging
import multiprocessing as mp
import queue
import time
from dataclasses import dataclass, field
from pathlib import Path

from PyQt6.QtCore import QObject, QTimer, pyqtSignal

from spt.core.handle import RingHandle
from spt.core.ring import SharedRing
from spt.core.stream_store import StreamInfo, StreamStats
from spt.protocol.format import Descriptor
from spt.sources.receiver import ReceiverConfig, run_udp_receiver
from spt.sources.replay import run_replay

log = logging.getLogger("spt.sources")
POLL_MS = 30
MAX_EVENTS_PER_POLL = 2000


class StreamHandle(RingHandle):
    """Canlı akış: tanımlayıcı + alıcı / oynatıcı sürecinin paylaşımlı tamponuna kopyasız okuma erişimi."""

    def __init__(self, info: StreamInfo):
        self.key = info.key
        self.source = info.source
        self.desc: Descriptor = info.descriptor
        self.t0 = info.t0
        self.ring = SharedRing.attach(info.shm_name, info.descriptor.sample_type.dtype)


@dataclass
class StreamRate:
    packets_per_s: float = 0.0
    bytes_per_s: float = 0.0
    lost_recent: int = 0
    last_loss: float = float("-inf")
    stats: StreamStats | None = None
    _prev: tuple[int, int, int, float] | None = field(default=None, repr=False)


class SourceManager(QObject):
    stream_added = pyqtSignal(object)          # StreamHandle
    stream_removed = pyqtSignal(str)           # key
    streams_cleared = pyqtSignal()
    stats_updated = pyqtSignal(object, object)  # oturum sözlüğü, {key: StreamRate}
    state_changed = pyqtSignal(str, str)       # durum, ayrıntı
    replay_progress = pyqtSignal(float, float, bool, bool)  # konum, süre, duraklatıldı, bitti
    recording_changed = pyqtSignal(bool, str, int)
    marker_received = pyqtSignal(str, int, int, str)

    def __init__(self, parent: QObject | None = None):
        super().__init__(parent)
        self._ctx = mp.get_context("spawn")
        self._proc = None
        self._commands = None
        self._events = None
        self.mode: str | None = None   # "udp" | "replay" | None
        self.state = "idle"
        self._detail = ""
        self._emitted = ("idle", "")
        self.streams: dict[str, StreamHandle] = {}
        self.rates: dict[str, StreamRate] = {}
        self._timer = QTimer(self)
        self._timer.setInterval(POLL_MS)
        self._timer.timeout.connect(self._poll)

    # ---------------------------------------------------------------- başlat / durdur
    def start_udp(self, cfg: ReceiverConfig) -> None:
        self.stop()
        self._spawn(run_udp_receiver, (cfg,), "udp")

    def start_replay(self, path: Path, speed: float, loop: bool, ring_seconds: float, max_ring_mb: int) -> None:
        self.stop()
        self._spawn(run_replay, (str(path), speed, loop, ring_seconds, max_ring_mb), "replay")

    def _spawn(self, target, args: tuple, mode: str) -> None:
        self._commands = self._ctx.Queue()
        self._events = self._ctx.Queue()
        self._proc = self._ctx.Process(target=target, args=(*args, self._commands, self._events),
                                       name=f"spt-{mode}", daemon=True)
        self._proc.start()
        self.mode = mode
        self._set_state("starting", "")
        self._timer.start()

    def stop(self) -> None:
        if self._proc is None:
            return
        self.send("stop")
        # Alt süreç, kuyruğundaki olaylar okunmadan kapanamaz; beklerken kuyruğu boşaltmaya devam et.
        deadline = time.monotonic() + 3.0
        while self._proc.is_alive() and time.monotonic() < deadline:
            self._drain(final=True)
            self._proc.join(timeout=0.05)
        if self._proc.is_alive():
            self._proc.terminate()
            self._proc.join(timeout=1.0)
        self._drain(final=True)
        self._timer.stop()
        self._proc = self._commands = self._events = None
        self.mode = None
        self._clear_streams()
        self._set_state("idle", "")

    def send(self, *command) -> None:
        if self._commands is not None:
            self._commands.put(command)

    @property
    def detail(self) -> str:
        """Son durumun ayrıntısı (ör. dinlenen adres, oynatılan dosya adı)."""
        return self._detail

    @property
    def active(self) -> bool:
        return self._proc is not None

    # ---------------------------------------------------------------- olaylar
    def _poll(self) -> None:
        self._drain()
        if self._proc is not None and not self._proc.is_alive():
            self._drain(final=True)
            self._timer.stop()
            self._proc = self._commands = self._events = None
            self.mode = None
            if self.state not in ("error",):
                self._set_state("idle", "")

    def _drain(self, final: bool = False) -> None:
        if self._events is None:
            return
        for _ in range(MAX_EVENTS_PER_POLL if not final else 100_000):
            try:
                event = self._events.get_nowait()
            except (queue.Empty, OSError, ValueError):
                return
            self._handle(event)

    def _handle(self, event: tuple) -> None:
        kind = event[0]
        if kind == "stream":
            self._add_stream(event[1])
        elif kind == "stream_removed":
            self._remove_stream(event[1])
        elif kind == "streams_reset":
            self._clear_streams()
        elif kind == "stats":
            self._update_stats(event[1], event[2])
        elif kind == "state":
            self._set_state(event[1], event[2])
        elif kind == "replay":
            self.replay_progress.emit(*event[1:])
        elif kind == "record":
            self.recording_changed.emit(*event[1:])
        elif kind == "marker":
            self.marker_received.emit(*event[1:])
            log.info("MARKER %s · örnek %d · kod %d · %s", event[1].split("#")[-1], event[2], event[3], event[4])
        elif kind == "log":
            log.log(event[1], event[2])
        elif kind == "error":
            log.error(event[1])
            self._set_state("error", event[1].splitlines()[0])
        elif kind == "stopped":
            pass

    def _add_stream(self, info: StreamInfo) -> None:
        self._remove_stream(info.key)
        try:
            handle = StreamHandle(info)
        except FileNotFoundError:
            return  # tampon bu arada kapatılmış (yapılandırma hızla değişti)
        self.streams[info.key] = handle
        self.rates[info.key] = StreamRate()
        self.stream_added.emit(handle)

    def _remove_stream(self, key: str) -> None:
        handle = self.streams.pop(key, None)
        self.rates.pop(key, None)
        if handle is not None:
            self.stream_removed.emit(key)
            handle.close()

    def _clear_streams(self) -> None:
        for key in list(self.streams):
            self._remove_stream(key)
        self.streams_cleared.emit()

    def _update_stats(self, session: dict, items: list[StreamStats]) -> None:
        # Hızlar, olayı üreten sürecin saatine göre hesaplanır; arayüz meşgulken biriken olaylar
        # aynı anda işlense bile süre farkı doğru kalır.
        now = session.get("t", time.monotonic())
        for st in items:
            rate = self.rates.get(st.key)
            if rate is None:
                continue
            if rate._prev is not None:
                packets, nbytes, lost, then = rate._prev
                dt = max(now - then, 1e-3)
                rate.packets_per_s = (st.packets - packets) / dt
                rate.bytes_per_s = (st.bytes - nbytes) / dt
                rate.lost_recent = st.lost_samples - lost
                if rate.lost_recent:
                    rate.last_loss = now
            rate._prev = (st.packets, st.bytes, st.lost_samples, now)
            rate.stats = st
        if self.mode == "udp" and self.state in ("listening", "receiving"):
            receiving = any(r.stats and r.stats.idle_s < 1.0 for r in self.rates.values())
            self._set_state("receiving" if receiving else "listening")
        self.stats_updated.emit(session, dict(self.rates))

    def _set_state(self, state: str, detail: str | None = None) -> None:
        if detail is not None:
            self._detail = detail
        if (state, self._detail) == self._emitted:
            return
        self.state = state
        self._emitted = (state, self._detail)
        self.state_changed.emit(state, self._detail)

    def shutdown(self) -> None:
        self.stop()
