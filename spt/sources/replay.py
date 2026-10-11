"""`.sptr` kayıt oynatma süreci.

Kaydı, UDP alıcısıyla aynı akış katmanından (StreamStore) geçirir; böylece ağ bağlantısı olmadan
kaydedilmiş veri canlı akışla birebir aynı şekilde analiz edilir. Zaman ekseni kayıt zamanını gösterir.
Bu modül Qt import etmez.
"""
from __future__ import annotations

import logging
import queue
import time
import traceback
from pathlib import Path

import numpy as np

from spt.core.stream_store import StreamStore
from spt.protocol.parser import FrameError
from spt.protocol.recording import RecordingError, SptrReader

PROGRESS_INTERVAL = 0.1
STATS_INTERVAL = 0.25
MAX_FRAMES_PER_STEP = 4000
MIN_LOOP_SECONDS = 0.25  # döngülü oynatmada iki başa sarma arası en kısa süre
MAX_SPEED = 0.0  # hız = 0: beklemeden, olabildiğince hızlı


def run_replay(path: str, speed: float, loop: bool, ring_seconds: float, max_ring_mb: int,
               commands, events) -> None:
    emit = events.put
    try:
        _play(Path(path), speed, loop, ring_seconds, max_ring_mb, commands, emit)
    except Exception:  # noqa: BLE001
        emit(("error", "Kayıt oynatma beklenmedik şekilde durdu:\n" + traceback.format_exc()))
    finally:
        emit(("stopped",))


def _play(path: Path, speed: float, loop: bool, ring_seconds: float, max_ring_mb: int, commands, emit) -> None:
    try:
        reader = SptrReader(path)
        offsets, rx, lengths = reader.index()
    except (OSError, RecordingError) as exc:
        emit(("error", f"Kayıt açılamadı ({path.name}): {exc}"))
        return
    if len(offsets) == 0:
        emit(("error", f"Kayıt boş: {path.name}"))
        reader.close()
        return
    if reader.resyncs:
        emit(("log", logging.WARNING, f"{path.name}: {reader.resyncs} bozuk bölge atlandı "
                                      f"({reader.skipped_bytes} bayt); geri kalan kayıtlar oynatılıyor"))
    if reader.truncated and not reader.resyncs:
        emit(("log", logging.WARNING, f"{path.name}: dosya sonu yarım kalmış, son kayıt atlandı"))

    mv = reader.buffer
    file_start = int(rx[0])
    duration = (int(rx[-1]) - file_start) / 1e9
    source = f"Kayıt · {path.name}"
    store = StreamStore(emit, file_start, ring_seconds, max_ring_mb << 20)
    emit(("state", "replay", path.name))
    emit(("log", logging.INFO, f"Kayıt açıldı: {path.name} — {len(offsets)} çerçeve, {duration:.1f} s"))

    clock = time.perf_counter
    n = len(offsets)
    pos = 0
    paused = finished = False
    anchor_wall, anchor_file = clock(), file_start
    next_progress = next_stats = 0.0
    off_l, rx_l, len_l = offsets.tolist(), rx.tolist(), lengths.tolist()

    def seek(fraction: float) -> None:
        nonlocal pos, anchor_wall, anchor_file, finished
        target = file_start + fraction * duration * 1e9
        pos = int(np.searchsorted(rx, target))
        pos = min(pos, n - 1)
        anchor_wall, anchor_file = clock(), rx_l[pos]
        finished = False
        store.reset()

    last_loop = clock()
    try:
        while True:
            # ---- komutlar
            try:
                while True:
                    cmd = commands.get_nowait()
                    name = cmd[0]
                    if name == "stop":
                        return
                    if name == "pause":
                        paused = True
                    elif name == "resume":
                        if finished:
                            seek(0.0)
                        paused = False
                        anchor_wall, anchor_file = clock(), rx_l[min(pos, n - 1)]
                    elif name == "seek":
                        seek(cmd[1])
                    elif name == "speed":
                        speed = cmd[1]
                        anchor_wall, anchor_file = clock(), rx_l[min(pos, n - 1)]
                    elif name == "loop":
                        loop = cmd[1]
            except queue.Empty:
                pass

            now = clock()
            if now >= next_progress:
                next_progress = now + PROGRESS_INTERVAL
                position = (rx_l[min(pos, n - 1)] - file_start) / 1e9
                emit(("replay", position, duration, paused, finished))
            if now >= next_stats:
                next_stats = now + STATS_INTERVAL
                playing = not (paused or finished)
                emit(("stats", {"t": now, "frame_errors": store.frame_errors, "no_descriptor": store.no_descriptor},
                      store.stats(rx_l[min(pos, n - 1)] if playing else None)))

            if paused or finished:
                time.sleep(0.02)
                continue

            target = float("inf") if speed == MAX_SPEED else anchor_file + (now - anchor_wall) * speed * 1e9
            done = 0
            while pos < n and rx_l[pos] <= target and done < MAX_FRAMES_PER_STEP:
                off = off_l[pos]
                try:
                    store.handle_frame(mv, off, off + len_l[pos], source, rx_l[pos])
                except FrameError as exc:
                    store.report_frame_error(source, exc)
                except Exception as exc:  # noqa: BLE001 — tek bir bozuk kayıt oynatmayı durdurmamalı
                    store.report_frame_error(source, FrameError(f"beklenmeyen hata ({type(exc).__name__}): {exc}"))
                pos += 1
                done += 1

            if pos >= n:
                if loop:
                    # Başa sarma tamponları yeniden kurar; arayüz yeni tampona ~30 ms'de bir bağlanır. Çok kısa kayıt
                    # "Maksimum" hızda daha sık sarılırsa akış arayüzde hiç görünemez: döngüler arası en az 250 ms.
                    wait = MIN_LOOP_SECONDS - (clock() - last_loop)
                    if wait > 0:
                        time.sleep(min(wait, 0.05))
                        continue
                    last_loop = clock()
                    seek(0.0)
                else:
                    finished = True
                    emit(("replay", duration, duration, paused, True))
                    emit(("log", logging.INFO, f"Kayıt sonuna gelindi: {path.name}"))
            elif done == 0:
                wait = (rx_l[pos] - target) / 1e9 / max(speed, 1e-9)
                time.sleep(min(max(wait, 0.0005), 0.01))
    finally:
        store.close()
        del mv
        reader.close()
