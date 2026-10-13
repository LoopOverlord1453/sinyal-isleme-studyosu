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
import math
import struct
import zlib
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from spt.core.ring import SharedRing

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
GAP_MEMORY = 64  # geç gelen (sıra dışı) çerçeveyi tanımak için hatırlanan son kayıp aralıkları
# Kaynak çakışması: aynı (IP, stream_id) iki göndericiden (ör. kartta iki görev / soket, ya da simülatör ve kart).
PORT_HANDOVER_NS = 1_000_000_000      # asıl kaynak portu bu kadar sustuysa yeni port aynı akışın devamıdır (yeniden bağlanma)
CONFLICT_HOLD_NS = 5_000_000_000      # çakışma son görüldükten bu kadar sonra temizlenir
CONFLICT_WARN_NS = 30_000_000_000     # aynı akış için uyarı günlüğü en sık bu aralıkla

# Varış zamanlaması (alıcı saati). Paket arası süre histogramı kenarları, µs: 1 µs … 10 s, dekad başına 20 kutu
# (kutu genişliği ≈ %12). Kutu 0: < 1 µs, son kutu: ≥ 10 s.
TIMING_EDGES_US = np.logspace(0, 7, 141)
PDV_WINDOW = 2048        # gecikme değişimi bu kadar son paketten
RATE_MIN_S = 2.0         # etkin örnekleme hızı / saat farkı bu kadar gözlem süresinden sonra raporlanır

# Hızlı yol: başlığın gereken alanları + DATA alt başlığı tek okumada (seq ve reserved atlanır)
_FAST_UNPACK = struct.Struct("<IBBHIHH8xQQI").unpack_from
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


@dataclass(frozen=True)
class LinkTiming:
    """Akışın varış zamanlaması ve hız doğruluğu, alıcının saatine göre. Süreler µs; ölçülemeyen değer NaN."""
    intervals: int              # oturumda ölçülen paket arası sayısı
    mean_us: float              # paket arası süre, oturum ortalaması
    std_us: float               # titreşim (standart sapma), oturum
    max_us: float               # en büyük boşluk, oturum
    recent_mean_us: float       # son istatistik aralığındaki (≈ 0.25 s) paketler
    recent_std_us: float
    recent_p99_us: float
    recent_max_us: float
    expected_us: float          # çerçeve başına ortalama örnek / fs: düzenli gönderimde beklenen aralık
    hist: tuple[int, ...]       # oturum histogramı; kutular TIMING_EDGES_US (ilk: < 1 µs, son: ≥ 10 s)
    rate_hz: float              # etkin örnekleme hızı: first_sample ilerlemesi / alıcı süresi
    rate_ppm: float             # nominal fs'e göre sapma (+: cihaz hızlı)
    rate_ppm_err: float         # ± belirsizlik (varış titreşimi / gözlem süresi)
    pdv_us: float               # gecikme değişimi: son PDV_WINDOW pakette (alım − gönderici zamanı) p99 − en küçük
    sender_ppm: float           # gönderici saati − PC saati (alım − gönderim farkının alt zarf eğimi)


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
    port: int = 0               # göndericinin UDP kaynak portu (bilinmiyorsa 0)
    sample_rate: float = 0.0    # nominal fs
    loss_events: int = 0        # kayıp olayı (first_sample boşluğu) sayısı
    max_loss_burst: int = 0     # tek seferde kaybolan en çok örnek
    late_packets: int = 0       # sıra dışı: daha önce kayıp sayılan aralığa sonradan düşen çerçeve (çizilmez)
    late_samples: int = 0
    timing: LinkTiming | None = None
    collisions: int = 0         # asıl kaynak etkinken başka kaynak portundan gelen datagram (oturum)
    source_ports: tuple[int, ...] = ()   # asıl port + son CONFLICT_HOLD_NS içinde görülen yabancı portlar
    conflict: str = ""          # "" | "port" (iki kaynak portu) | "sıra" (aynı porttan tutarsız first_sample dizisi)


def hist_percentile(hist, q: float) -> float:
    """TIMING_EDGES_US histogramından q. yüzdelik (0–100), µs; kutu içinde log-doğrusal ara değerleme. Boşsa NaN."""
    h = np.asarray(hist, dtype=np.float64)
    total = h.sum()
    if total <= 0:
        return math.nan
    cum = np.cumsum(h)
    target = q / 100.0 * total
    i = int(np.searchsorted(cum, target, side="left"))
    i = min(i, len(h) - 1)
    edges = TIMING_EDGES_US
    if i == 0:
        return float(edges[0])
    if i >= len(edges):
        return float(edges[-1])
    lo, hi = math.log10(edges[i - 1]), math.log10(edges[i])
    below = cum[i - 1] if i > 0 else 0.0
    frac = (target - below) / h[i] if h[i] else 0.0
    return float(10 ** (lo + (hi - lo) * min(max(frac, 0.0), 1.0)))


def _pdv(t_s: np.ndarray, owd: np.ndarray) -> float:
    """Gecikme değişimi (ns): (alım − gönderim) farkının p99'u − en küçüğü, saat kayması (iki saatin hız farkından
    doğan doğrusal eğim) çıkarılarak. Eğim alt zarftan: pencerenin iki yarısının en küçük noktaları arası (kuyruk
    gecikmesi yalnız ekler, en küçük değerler kaymayı izler)."""
    h = len(owd) // 2
    resid = owd
    if h >= 8:
        i, j = int(np.argmin(owd[:h])), h + int(np.argmin(owd[h:]))
        if t_s[j] > t_s[i]:
            resid = owd - (owd[j] - owd[i]) / (t_s[j] - t_s[i]) * (t_s - t_s[i])
    return float(np.percentile(resid, 99) - resid.min())


class _Timing:
    """Akış başına varış günlüğü ve birikmiş zamanlama istatistiği. Sıcak yol yalnız iki listeye ekler (alım ve
    gönderici zamanı, ns; ölçüm: datagram başına ≈ 0.1 µs, array('q')'dan hızlı); hesap `drain` / `summary` içinde
    (≈ 4 Hz) numpy ile yapılır."""

    def __init__(self) -> None:
        self.rx: list[int] = []
        self.ts: list[int] = []
        self.prev_rx = 0
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0
        self.max = 0.0
        self.hist = np.zeros(len(TIMING_EDGES_US) + 1, np.int64)
        self.recent = (math.nan,) * 4
        self.owd_ref: int | None = None          # alım − gönderim farkının başlangıç değeri (ns, taşma önlemi)
        self.t_ref = 0                           # owd_rx'in sıfırı (alım ns)
        self.owd = np.empty(0)                   # son PDV_WINDOW fark, ns (owd_ref'e göre)
        self.owd_rx = np.empty(0)                # aynı paketlerin alım zamanı, s (eğim düzeltmesi için)
        self.env_ref: tuple[int, float] | None = None   # (alım ns, en küçük fark): saat farkının başlangıcı
        self.env_now: tuple[int, float] | None = None
        self.rate_ref: tuple[int, int] | None = None    # (son alım ns, next_sample): etkin hızın başlangıcı
        self.rate_now: tuple[int, int] | None = None
        self.restarts = 0

    def restart(self) -> None:
        """Cihaz resetlendi: gönderici saati ve örnek sayacı yeniden başladı; hız / saat referansları silinir."""
        self.owd_ref = self.env_ref = self.env_now = self.rate_ref = self.rate_now = None
        self.owd = self.owd_rx = np.empty(0)
        self.prev_rx = 0

    def drain(self, next_sample: int | None) -> None:
        if not self.rx:
            self.recent = (math.nan,) * 4
            return
        rx = np.array(self.rx, dtype=np.int64)
        ts = np.array(self.ts, dtype=np.uint64).astype(np.int64)   # protokolde u64
        self.rx.clear()
        self.ts.clear()
        d = np.diff(rx, prepend=self.prev_rx) if self.prev_rx else np.diff(rx)
        self.prev_rx = int(rx[-1])
        if len(d):
            d_us = d.astype(np.float64) / 1e3
            nb, mb = len(d_us), float(d_us.mean())
            m2b = float(((d_us - mb) ** 2).sum())
            n = self.n + nb                                  # Chan: iki grubun ortalama / varyans birleşimi
            delta = mb - self.mean
            self.mean += delta * nb / n
            self.m2 += m2b + delta * delta * self.n * nb / n
            self.n = n
            self.max = max(self.max, float(d_us.max()))
            self.hist += np.bincount(np.searchsorted(TIMING_EDGES_US, d_us, side="right"),
                                     minlength=len(self.hist))[:len(self.hist)]
            self.recent = (mb, float(d_us.std()), float(np.percentile(d_us, 99)), float(d_us.max()))
        else:
            self.recent = (math.nan,) * 4
        if next_sample is not None:
            now = (self.prev_rx, next_sample)
            self.rate_now = now
            if self.rate_ref is None:
                self.rate_ref = now
        if ts.all():                                         # gönderici zamanı biliniyor (0 değil)
            if self.owd_ref is None:
                self.owd_ref = int(rx[0] - ts[0])
                self.t_ref = int(rx[0])
            owd = ((rx - ts) - self.owd_ref).astype(np.float64)
            self.owd = np.concatenate((self.owd, owd))[-PDV_WINDOW:]
            self.owd_rx = np.concatenate((self.owd_rx, (rx - self.t_ref) / 1e9))[-PDV_WINDOW:]
            k = int(np.argmin(owd))
            point = (int(rx[k]), float(owd[k]))
            self.env_now = point
            if self.env_ref is None:
                self.env_ref = point

    def summary(self, fs: float, expected_us: float) -> LinkTiming:
        std = math.sqrt(self.m2 / (self.n - 1)) if self.n > 1 else math.nan
        rate = ppm = err = math.nan
        if self.rate_ref is not None and self.rate_now is not None:
            elapsed = self.rate_now[0] - self.rate_ref[0]
            if elapsed >= RATE_MIN_S * 1e9:
                rate = (self.rate_now[1] - self.rate_ref[1]) / elapsed * 1e9
                ppm = (rate / fs - 1.0) * 1e6
                jitter_ns = (self.recent[2] if math.isfinite(self.recent[2]) else self.mean) * 1e3
                err = jitter_ns / elapsed * 1e6
        pdv = sender = math.nan
        if len(self.owd) > 1:
            pdv = _pdv(self.owd_rx, self.owd) / 1e3
        if self.env_ref is not None and self.env_now is not None:
            elapsed = self.env_now[0] - self.env_ref[0]
            if elapsed >= RATE_MIN_S * 1e9:
                sender = -(self.env_now[1] - self.env_ref[1]) / elapsed * 1e6
        mean = self.mean if self.n else math.nan
        return LinkTiming(self.n, mean, std, self.max if self.n else math.nan, *self.recent, expected_us,
                          tuple(int(v) for v in self.hist), rate, ppm, err, pdv, sender)


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
        self.loss_events = self.max_loss_burst = 0
        self.late_packets = self.late_samples = 0
        self.gaps: deque[tuple[int, int]] = deque(maxlen=GAP_MEMORY)  # son kayıp aralıkları [başlangıç, bitiş)
        self.port = 0
        self.collisions = 0
        self.foreign_packets = 0                    # çakışan kaynaktan gelip atılan DATA çerçeveleri
        self.foreign_ports: dict[int, int] = {}     # yabancı kaynak portu → son görülme (alım ns)
        self.conflict = ""
        self.conflict_ns = 0
        self.warned_ns: int | None = None
        self.prev_counts = (0, 0, 0)                # (paket, yineleme, reset): önceki istatistik anında
        self.timing = _Timing()
        self.rx_log = self.timing.rx                # sıcak yol: bağlı liste nesneleri
        self.ts_log = self.timing.ts
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
    def handle_datagram(self, mv: memoryview, n: int, source: str, rx_ns: int, port: int = 0) -> None:
        """Bir UDP datagramındaki tüm çerçeveleri işler. `port`: göndericinin kaynak portu (yalnız bilgi).

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
                magic, version, ftype, flags, flen, sid, rev, ts, first, count = _FAST_UNPACK(mv, off)
                if (ftype == 2 and flags <= FLAG_CRC32 and magic == _MAGIC_U32 and version == 1
                        and off + flen <= n):
                    st = fast.get(sid)
                    if st is not None and st.rev == rev and st.source == source and st.port == port:
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
                            st.rx_log.append(rx_ns)
                            st.ts_log.append(ts)
                            if self.recorder is not None:
                                self.recorder.write(mv[off:off + flen], rx_ns)
                            off += flen
                            continue
            if mv[off] == 0 and not any(mv[off:n]):
                return  # datagram sonundaki sıfır dolgu: hata değil
            try:
                off += self.handle_frame(mv, off, n, source, rx_ns, port)
            except FrameError as exc:
                self.report_frame_error(source, exc)
                return  # datagramın geri kalanı güvenilir değil
            except Exception as exc:  # noqa: BLE001 — tek bir bozuk datagram alıcıyı asla durdurmamalı
                self.report_frame_error(source, FrameError(f"beklenmeyen hata ({type(exc).__name__}): {exc}"))
                return

    def handle_frame(self, mv, off: int, end: int, source: str, rx_ns: int, port: int = 0) -> int:
        """Tek bir çerçeveyi işler, çerçeve uzunluğunu döndürür. Hatalıysa FrameError fırlatır."""
        ftype, flags, flen, sid, rev, seq, ts = check_frame(mv, off, end)
        pstart = off + HEADER.size
        pend = off + flen - (CRC.size if flags & FLAG_CRC32 else 0)
        if ftype == FrameType.DATA:
            self._data(mv, pstart, pend, source, sid, rev, rx_ns, ts, port)
        elif ftype == FrameType.DESCRIPTOR:
            st = self._descriptor(unpack_descriptor(mv, pstart, pend, sid, rev), source, rx_ns, port)
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
    def _data(self, mv, pstart: int, pend: int, source: str, sid: int, rev: int, rx_ns: int, ts: int = 0,
              port: int = 0) -> None:
        st = self._streams.get((source, sid))
        if st is None or st.desc.desc_rev != rev:
            self.no_descriptor += 1  # tanımlayıcı henüz gelmedi; protokol gereği atılır
            return
        own = self._own_port(st, port, rx_ns)
        if pend - pstart < DATA.size:
            raise FrameError("DATA payload'ı çok kısa")
        first, count, _res = DATA.unpack_from(mv, pstart)
        if first >= MAX_FIRST_SAMPLE:
            raise FrameError(f"first_sample çok büyük: {first} (en fazla 2^62)")
        nbytes = pend - pstart - DATA.size
        if nbytes != count * st.bytes_per_sample:
            raise FrameError(f"DATA boyutu {nbytes} bayt, {count} örnek için {count * st.bytes_per_sample} olmalı")
        if not own:
            # Kaynak çakışması: ikinci göndericinin örnekleri tampona yazılırsa iki farklı sinyal aynı kanalda
            # parça parça karışır ("üst üste birkaç sinyal"). Sayılır ve uyarılır (_own_port), çizilmez.
            st.foreign_packets += 1
            return

        if st.ring is None:
            self._open_ring(st, rx_ns, count)
        expected = st.next_sample
        if expected is not None and first != expected:
            if first > expected:
                if first - expected > MAX_SAMPLE_JUMP:
                    raise FrameError(f"first_sample sıçraması çok büyük: {expected} → {first}")
                st.lost += first - expected
                st.loss_events += 1
                st.max_loss_burst = max(st.max_loss_burst, first - expected)
                st.gaps.append((expected, first))
                st.ring.skip(first - expected)
                st._back_run = 0
            elif first != 0 and self._late(st, first, count):
                st.late_packets += 1   # kayıp sayılan aralığa sonradan düştü: sıra dışı varış (tampon geçti)
                st.late_samples += count
                return
            elif first == 0 or self._backward_run(st, first, count):
                st.restarts += 1
                st._back_run = 0
                st.gaps.clear()
                st.timing.restart()
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
        if own:   # yabancı kaynağın varışları zamanlamaya ve boşta süresine karışmaz (asıl kaynağın susması görülsün)
            st.last_rx_ns = rx_ns
            st.rx_log.append(rx_ns)
            st.ts_log.append(ts)

    def _own_port(self, st: _Stream, port: int, rx_ns: int) -> bool:
        """Çerçeve akışın asıl kaynak portundan mı? Asıl port bir süredir susmuşsa yeni port devralır (cihaz yeniden
        bağlandı, yeni yerel port); asıl kaynak etkinken başka porttan gelen çerçeve kaynak çakışmasıdır: sayılır,
        bir kez uyarılır; verisi tampona yazılmaz (bkz. _data)."""
        if not port or port == st.port:
            return True
        if not st.port or rx_ns - st.last_rx_ns > PORT_HANDOVER_NS:
            if st.port:
                self._emit(("log", logging.INFO, f"{st.desc.name}: kaynak portu değişti {st.port} → {port} "
                                                 "(cihaz yeniden bağlandı?)"))
            st.port = port
            st.foreign_ports.pop(port, None)
            return True
        st.collisions += 1
        st.foreign_ports[port] = rx_ns
        self._conflict(st, "port", rx_ns,
                       f"{st.desc.name}: aynı akış kimliği (sid {st.desc.stream_id}) iki kaynaktan geliyor: "
                       f"{st.source}:{st.port} ve {st.source}:{port} — kartta stream_id'leri ayırın ya da simülatörü "
                       "durdurun")
        return False

    def _conflict(self, st: _Stream, kind: str, now_ns: int, message: str) -> None:
        st.conflict = kind
        st.conflict_ns = now_ns
        if st.warned_ns is None or now_ns - st.warned_ns >= CONFLICT_WARN_NS:
            st.warned_ns = now_ns
            self._emit(("log", logging.WARNING, message))

    @staticmethod
    def _late(st: _Stream, first: int, count: int) -> bool:
        """Geriye düşen çerçeve, yakın zamanda kayıp sayılan bir aralığın içinde mi (yeniden sıralanmış datagram)?"""
        end = first + count
        return any(g0 <= first and end <= g1 for g0, g1 in st.gaps)

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

    def _descriptor(self, desc: Descriptor, source: str, rx_ns: int, port: int = 0) -> _Stream | None:
        """Tanımlayıcıyı uygular; kabul edilen akışı döndürür. Asıl kaynak etkinken başka porttan gelen farklı
        tanımlayıcı (kaynak çakışması) akışı yeniden kurmaz (aksi halde iki gönderici akışı sürekli sıfırlardı)."""
        key = (source, desc.stream_id)
        st = self._streams.get(key)
        if st is not None:
            if port and st.port and port != st.port and not self._own_port(st, port, rx_ns):
                return None
            if st.desc == desc:
                return st  # periyodik tekrar
            if self._fast.get(desc.stream_id) is st:
                del self._fast[desc.stream_id]
            st.close()
            self._emit(("stream_removed", st.key))
            self._emit(("log", logging.INFO, f"{desc.name}: yapılandırma değişti (rev {desc.desc_rev})"))
        st = _Stream(f"{source}#{desc.stream_id}", source, desc, self.ring_seconds, self.max_ring_bytes)
        st.port = port
        self._streams[key] = st
        self._emit(("log", logging.INFO, f"Yeni akış: {desc.name or desc.stream_id} — {desc.format_text()} ({source})"))
        return st

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
            if now_ns is not None:
                self._check_sequence(st, now_ns)
            idle = (now_ns - st.last_rx_ns) / 1e9 if st.last_rx_ns and now_ns is not None else float("inf")
            fs = st.desc.sample_rate
            st.timing.drain(st.next_sample)
            expected = st.samples / st.packets / fs * 1e6 if st.packets else math.nan
            out.append(StreamStats(st.key, st.desc.stream_id, st.desc.name, st.source, st.desc.format_text(),
                                   st.packets, st.bytes, st.samples, st.lost, st.duplicates, st.restarts, idle,
                                   st.port, fs, st.loss_events, st.max_loss_burst, st.late_packets,
                                   st.late_samples, st.timing.summary(fs, expected), st.collisions,
                                   (st.port, *sorted(st.foreign_ports)) if st.port else tuple(sorted(st.foreign_ports)),
                                   st.conflict))
        return out

    def _check_sequence(self, st: _Stream, now_ns: int) -> None:
        """Aynı kaynak portundan iki göndericinin karışık first_sample dizisi: aralıkta sık reset ya da çerçevelerin
        ≥ %5'i yineleme. Çakışma CONFLICT_HOLD_NS boyunca görülmezse temizlenir."""
        for port, seen in list(st.foreign_ports.items()):
            if now_ns - seen > CONFLICT_HOLD_NS:
                del st.foreign_ports[port]
        packets, dups, restarts = st.prev_counts
        d_pk, d_dup, d_rs = st.packets - packets, st.duplicates - dups, st.restarts - restarts
        st.prev_counts = (st.packets, st.duplicates, st.restarts)
        if d_rs >= 2 or (d_dup >= 5 and d_dup * 20 >= d_pk):
            if st.foreign_ports:
                st.conflict_ns = now_ns          # nedeni zaten görülen ikinci kaynak portu
            else:
                self._conflict(st, "sıra", now_ns,
                               f"{st.desc.name}: first_sample dizisi tutarsız ({d_dup} yineleme, {d_rs} yeniden "
                               f"başlama / {d_pk} çerçeve) — aynı akış kimliğiyle (sid {st.desc.stream_id}) iki "
                               "kaynak gönderiyor olabilir; kartta stream_id'leri ayırın ya da simülatörü durdurun")
        if st.conflict and now_ns - st.conflict_ns > CONFLICT_HOLD_NS:
            st.conflict = ""

    def reset(self, session_start_ns: int | None = None) -> None:
        """Tüm tamponları kapatır; tanımlayıcılar korunur, sonraki veriyle tamponlar yeniden açılır."""
        if session_start_ns is not None:
            self.session_start_ns = session_start_ns
        self._fast.clear()
        for st in self._streams.values():
            st.close()
            st.next_sample = None
            st.gaps.clear()
            st.timing.restart()
            st.timing.rx.clear()
            st.timing.ts.clear()
        self._emit(("streams_reset",))

    def close(self) -> None:
        self._fast.clear()
        for st in self._streams.values():
            st.close()
        self._streams.clear()
