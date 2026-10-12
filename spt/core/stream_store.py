"""Alıcı / oynatıcı sürecindeki akış durumu: çerçeveleri doğrular, tamponlara yazar, kayıpları sayar.

Bu modül Qt kullanmaz; arayüze olaylar `emit(tuple)` ile (ör. multiprocessing kuyruğu) gönderilir.

Olaylar:
    ("stream", StreamInfo)                    yeni akış veya yeniden yapılandırılmış akış
    ("stream_removed", key)
    ("streams_reset",)                        oynatmada konum değişti; tüm tamponlar yeniden kuruluyor
    ("marker", key, sample_index, code, text)
    ("log", seviye, ileti)
"""
from __future__ import annotations

import logging
import struct
import zlib
from collections.abc import Callable
from dataclasses import dataclass

from spt.core.ring import SharedRing
import numpy as np

from spt.protocol.format import (
    CRC,
    DATA,
    FLAG_CRC32,
    HEADER,
    MAGIC,
    MAX_FIRST_SAMPLE,
    MAX_SAMPLE_JUMP,
    Descriptor,
    FrameType,
    Layout,
)
from spt.protocol.parser import FrameError, check_frame, samples_view, unpack_descriptor, unpack_marker

MAX_MARKER_EVENTS_PER_TICK = 50
MAX_PENDING_MARKERS = 32   # tampon açılmadan (ilk DATA'dan önce) gelen işaretler, açılınca konumlanır
MAX_ERROR_LOGS = 20
RESTART_RUN = 4  # geriye düşen ama kendi içinde ardışık bu kadar çerçeve: reset çerçevesi kaybolmuş, yeniden başla

# Hızlı yol: başlığın gereken alanları + DATA alt başlığı tek okumada (seq, reserved, timestamp atlanır)
_FAST_UNPACK = struct.Struct("<IBBHIHH16xQI").unpack_from
_MAGIC_U32 = struct.unpack("<I", MAGIC)[0]
_FAST_MIN = HEADER.size + DATA.size
_CRC_UNPACK = CRC.unpack_from


@dataclass(frozen=True)
class StreamInfo:
    """Arayüzün bir akışa bağlanması için gereken her şey (süreçler arası taşınır)."""
    key: str
    source: str
    descriptor: Descriptor
    shm_name: str
    t0: float  # tampondaki ilk örneğin oturum başlangıcına göre zamanı (s)


@dataclass
class StreamStats:
    key: str
    stream_id: int
    name: str
    source: str
    format: str
    packets: int
    bytes: int
    samples: int
    lost_samples: int
    duplicates: int
    restarts: int
    idle_s: float


class _Stream:
    def __init__(self, key: str, source: str, desc: Descriptor, ring_seconds: float, max_ring_bytes: int):
        self.key = key
        self.source = source
        self.desc = desc
        self.rev = desc.desc_rev
        self.ring: SharedRing | None = None
        self.dtype = desc.sample_type.dtype
        self.channels = desc.channel_count
        self.bytes_per_sample = desc.bytes_per_sample
        by_time = int(desc.sample_rate * ring_seconds)
        by_memory = max_ring_bytes // desc.bytes_per_sample
        self.capacity = max(1024, min(by_time, by_memory))
        self.next_sample: int | None = None
        self.planar = desc.layout == Layout.PLANAR
        self.raw_descriptor: bytes | None = None  # kayıt ortada başlarsa dosyaya ilk yazılır
        self._back_next: int | None = None        # geriye düşen çerçevelerin beklenen sonraki first_sample'ı
        self._back_run = 0
        self.packets = self.bytes = self.samples = 0
        self.lost = self.duplicates = self.restarts = 0
        self.last_rx_ns = 0
        self.pending_markers: list[tuple[int, int, str]] = []  # (örnek indeksi, kod, metin)

    def close(self) -> None:
        if self.ring is not None:
            self.ring.close()
            self.ring = None


class StreamStore:
    def __init__(self, emit: Callable[[tuple], None], session_start_ns: int, ring_seconds: float = 10.0,
                 max_ring_bytes: int = 256 << 20):
        self._emit = emit
        self.session_start_ns = session_start_ns
        self.ring_seconds = ring_seconds
        self.max_ring_bytes = max_ring_bytes
        self._streams: dict[tuple[str, int], _Stream] = {}
        self._fast: dict[int, _Stream] = {}  # stream_id → tamponu açık, interleaved akış (kaynak ayrıca denetlenir)
        self.recorder = None  # SptrWriter; ayarlanırsa geçerli her çerçeve kaydedilir
        self.datagrams = 0
        self.bytes = 0
        self.frame_errors = 0
        self.unknown_frames = 0
        self.no_descriptor = 0
        self._marker_budget = MAX_MARKER_EVENTS_PER_TICK
        self._dropped_markers = 0

    # ---------------------------------------------------------------- giriş noktaları
    def handle_datagram(self, mv: memoryview, n: int, source: str, rx_ns: int) -> None:
        """Bir UDP datagramındaki tüm çerçeveleri işler.

        Gigabit hızlarında saniyede ~80 bin datagram gelir; bu yüzden en sık durum olan "tanımlı,
        interleaved akışın sıradaki DATA çerçevesi" burada numpy'ye uğramadan işlenir. Geri kalan her
        şey (tanımlayıcı, işaret, kayıp, CRC hatası…) genel yola (`handle_frame`) düşer.
        """
        self.datagrams += 1
        self.bytes += n
        fast = self._fast
        off = 0
        while off < n:
            if n - off >= _FAST_MIN:
                magic, version, ftype, flags, flen, sid, rev, first, count = _FAST_UNPACK(mv, off)
                if (ftype == 2 and flags <= FLAG_CRC32 and magic == _MAGIC_U32 and version == 1
                        and off + flen <= n):
                    st = fast.get(sid)
                    if st is not None and st.rev == rev and st.source == source:
                        crc = flags * 4
                        nbytes = flen - _FAST_MIN - crc
                        if (first == st.next_sample and nbytes == count * st.bytes_per_sample
                                and (not crc or zlib.crc32(mv[off:off + flen - crc])
                                     == _CRC_UNPACK(mv, off + flen - crc)[0])):
                            start = off + _FAST_MIN
                            if st.planar:  # kanal blokları: (kanal, n) görünümünün devriği tampona yazılır
                                st.ring.write(np.frombuffer(mv, st.dtype, count * st.channels, start)
                                              .reshape(st.channels, count).T)
                            else:
                                st.ring.write_bytes(mv[start:start + nbytes], count)
                            st._back_run = 0
                            st.next_sample = first + count
                            st.packets += 1
                            st.bytes += nbytes
                            st.samples += count
                            st.last_rx_ns = rx_ns
                            if self.recorder is not None:
                                self.recorder.write(mv[off:off + flen], rx_ns)
                            off += flen
                            continue
            if mv[off] == 0 and not any(mv[off:n]):
                return  # datagram sonundaki sıfır dolgu: hata değil
            try:
                off += self.handle_frame(mv, off, n, source, rx_ns)
            except FrameError as exc:
                self.report_frame_error(source, exc)
                return  # datagramın geri kalanı güvenilir değil
            except Exception as exc:  # noqa: BLE001 — tek bir bozuk datagram alıcıyı asla durdurmamalı
                self.report_frame_error(source, FrameError(f"beklenmeyen hata ({type(exc).__name__}): {exc}"))
                return

    def handle_frame(self, mv, off: int, end: int, source: str, rx_ns: int) -> int:
        """Tek bir çerçeveyi işler, çerçeve uzunluğunu döndürür. Hatalıysa FrameError fırlatır."""
        ftype, flags, flen, sid, rev, seq, ts = check_frame(mv, off, end)
        pstart = off + HEADER.size
        pend = off + flen - (CRC.size if flags & FLAG_CRC32 else 0)
        if ftype == FrameType.DATA:
            self._data(mv, pstart, pend, source, sid, rev, rx_ns)
        elif ftype == FrameType.DESCRIPTOR:
            self._descriptor(unpack_descriptor(mv, pstart, pend, sid, rev), source, rx_ns)
            st = self._streams.get((source, sid))
            if st is not None:
                st.raw_descriptor = bytes(mv[off:off + flen])
        elif ftype == FrameType.MARKER:
            self._marker(mv, pstart, pend, source, sid)
        else:  # ileri uyumluluk: bilinmeyen tip atlanır, datagramın geri kalanı işlenir
            self.unknown_frames += 1
            return flen
        if self.recorder is not None:
            self.recorder.write(mv[off:off + flen], rx_ns)
        return flen

    # ---------------------------------------------------------------- çerçeve tipleri
    def _data(self, mv, pstart: int, pend: int, source: str, sid: int, rev: int, rx_ns: int) -> None:
        st = self._streams.get((source, sid))
        if st is None or st.desc.desc_rev != rev:
            self.no_descriptor += 1  # tanımlayıcı henüz gelmedi; protokol gereği atılır
            return
        if pend - pstart < DATA.size:
            raise FrameError("DATA payload'ı çok kısa")
        first, count, _res = DATA.unpack_from(mv, pstart)
        if first >= MAX_FIRST_SAMPLE:
            raise FrameError(f"first_sample çok büyük: {first} (en fazla 2^62)")
        nbytes = pend - pstart - DATA.size
        if nbytes != count * st.bytes_per_sample:
            raise FrameError(f"DATA boyutu {nbytes} bayt, {count} örnek için {count * st.bytes_per_sample} olmalı")

        if st.ring is None:
            self._open_ring(st, rx_ns, count)
        expected = st.next_sample
        if expected is not None and first != expected:
            if first > expected:
                if first - expected > MAX_SAMPLE_JUMP:
                    raise FrameError(f"first_sample sıçraması çok büyük: {expected} → {first}")
                st.lost += first - expected
                st.ring.skip(first - expected)
                st._back_run = 0
            elif first == 0 or self._backward_run(st, first, count):
                st.restarts += 1
                st._back_run = 0
                why = "" if first == 0 else " (reset çerçevesi kaybolmuş; ardışık geri çerçevelerden anlaşıldı)"
                self._emit(("log", logging.WARNING, f"{st.desc.name}: akış yeniden başladı (cihaz reset?){why}"))
            else:
                st.duplicates += 1
                return
        else:
            st._back_run = 0
        st.ring.write(samples_view(mv, pstart + DATA.size, count, st.desc))
        st.next_sample = first + count
        if st.pending_markers:
            for sample_index, code, text in st.pending_markers:
                self._emit_marker(st, sample_index, code, text)
            st.pending_markers.clear()
        st.packets += 1
        st.bytes += nbytes
        st.samples += count
        st.last_rx_ns = rx_ns

    @staticmethod
    def _backward_run(st: _Stream, first: int, count: int) -> bool:
        """Geriye düşen çerçeve: tek bir eski kopya mı (yinelenen), yoksa cihaz resetlendi ve first_sample=0
        çerçevesi kayboldu mu? Kendi içinde ardışık RESTART_RUN geri çerçeve gelirse reset kabul edilir."""
        if st._back_next == first:
            st._back_run += 1
        else:
            st._back_run = 1
        st._back_next = first + count
        return st._back_run >= RESTART_RUN

    def descriptor_frames(self) -> list[bytes]:
        """Bilinen akışların son tanımlayıcı çerçeveleri (kayıt ortada başladığında dosyanın başına)."""
        return [st.raw_descriptor for st in self._streams.values() if st.raw_descriptor]

    def _descriptor(self, desc: Descriptor, source: str, rx_ns: int) -> None:
        key = (source, desc.stream_id)
        st = self._streams.get(key)
        if st is not None:
            if st.desc == desc:
                return  # periyodik tekrar
            if self._fast.get(desc.stream_id) is st:
                del self._fast[desc.stream_id]
            st.close()
            self._emit(("stream_removed", st.key))
            self._emit(("log", logging.INFO, f"{desc.name}: yapılandırma değişti (rev {desc.desc_rev})"))
        st = _Stream(f"{source}#{desc.stream_id}", source, desc, self.ring_seconds, self.max_ring_bytes)
        self._streams[key] = st
        self._emit(("log", logging.INFO, f"Yeni akış: {desc.name or desc.stream_id} — {desc.format_text()} ({source})"))

    def _marker(self, mv, pstart: int, pend: int, source: str, sid: int) -> None:
        sample_index, code, text = unpack_marker(mv, pstart, pend)
        st = self._streams.get((source, sid))
        if st is None:
            return
        if st.ring is None or st.next_sample is None:
            if len(st.pending_markers) < MAX_PENDING_MARKERS:
                st.pending_markers.append((sample_index, code, text))
            return
        self._emit_marker(st, sample_index, code, text)

    def _emit_marker(self, st: _Stream, sample_index: int, code: int, text: str) -> None:
        """İşareti olay olarak iletir. Konum, akışın `first_sample` ekseninden halka tampon indeksine çevrilir
        (tampon indeksi = yazılan + atlanan örnek; arayüzde zaman = t0 + indeks / fs)."""
        if self._marker_budget > 0:
            self._marker_budget -= 1
            pos = st.ring.write_count + (sample_index - st.next_sample)
            self._emit(("marker", st.key, sample_index, code, text, pos))
        else:
            self._dropped_markers += 1

    def _open_ring(self, st: _Stream, rx_ns: int, first_count: int) -> None:
        st.ring = SharedRing.create(st.capacity, st.desc.channel_count, st.dtype)
        self._fast[st.desc.stream_id] = st  # interleaved: bayt kopyası; planar: devrik görünüm (hızlı yolda)
        duration_ns = first_count / st.desc.sample_rate * 1e9
        t0 = (rx_ns - duration_ns - self.session_start_ns) / 1e9
        self._emit(("stream", StreamInfo(st.key, st.source, st.desc, st.ring.name, t0)))

    def report_frame_error(self, source: str, exc: FrameError) -> None:
        self.frame_errors += 1
        if self.frame_errors <= MAX_ERROR_LOGS:
            suffix = " (sonraki hatalar yalnızca sayılacak)" if self.frame_errors == MAX_ERROR_LOGS else ""
            self._emit(("log", logging.WARNING, f"Hatalı çerçeve ({source}): {exc}{suffix}"))

    # ---------------------------------------------------------------- yönetim
    def stats(self, now_ns: int | None) -> list[StreamStats]:
        """Akış istatistikleri. `now_ns=None`: akış duraklatılmış sayılır (boşta süresi sonsuz)."""
        if self._dropped_markers:
            self._emit(("log", logging.WARNING, f"{self._dropped_markers} MARKER çok sık geldiği için gösterilmedi"))
            self._dropped_markers = 0
        self._marker_budget = MAX_MARKER_EVENTS_PER_TICK
        out = []
        for st in self._streams.values():
            idle = (now_ns - st.last_rx_ns) / 1e9 if st.last_rx_ns and now_ns is not None else float("inf")
            out.append(StreamStats(st.key, st.desc.stream_id, st.desc.name, st.source, st.desc.format_text(),
                                   st.packets, st.bytes, st.samples, st.lost, st.duplicates, st.restarts, idle))
        return out

    def reset(self, session_start_ns: int | None = None) -> None:
        """Tüm tamponları kapatır; tanımlayıcılar korunur, sonraki veriyle tamponlar yeniden açılır."""
        if session_start_ns is not None:
            self.session_start_ns = session_start_ns
        self._fast.clear()
        for st in self._streams.values():
            st.close()
            st.next_sample = None
        self._emit(("streams_reset",))

    def close(self) -> None:
        self._fast.clear()
        for st in self._streams.values():
            st.close()
        self._streams.clear()
