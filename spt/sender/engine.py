"""Sinyal Test Uygulaması gönderim motoru: tüm akışları tek süreçte, örnekleme hızına göre zamanlayıp SPTF/UDP ile
gönderir.

Tasarım:
- Arayüzden ayrı süreçte çalışır (multiprocessing "spawn"): arayüz çizimi ve Python'un genel kilidi zamanlamayı
  bozmaz. Komutlar ve istatistikler kuyruklarla taşınır.
- Akış başına ayrı süreç yerine tek süreç + tek soket: akışlar aynı kaynak adres/porttan gelir (Stüdyo IP + akış
  kimliğiyle ayırır), süreç başına ≈ 30–50 MB bellek ve başlatma süresi tasarruf edilir. Darboğaz çerçeve başına
  Python maliyetidir (≈ 5 µs); tek süreç ≈ 100 bin çerçeve/s'e (1472 B'lık datagramla ≈ 140 MB/s) yetişir; bu,
  gigabit hattın ve alıcının kapasitesinin üstündedir.
- Zamanlama: her döngüde akışın "olması gereken" örnek sayısı = (şimdi − başlangıç) · fs; aradaki fark tam
  çerçeveler hâlinde gönderilir. Windows'ta varsayılan uyku çözünürlüğü ≈ 15.6 ms olduğundan motor süreci
  `timeBeginPeriod(1)` ile 1 ms'ye çeker; çerçeveler böylece ≈ 1 ms içinde, düzgün aralıklarla çıkar.
- Örnekler bloklar hâlinde üretilir (akış başına ≈ 10 ms ya da en az 8 çerçeve): numpy maliyeti örnek başına küçük
  kalır; çerçeve, bloğun bayt görünümünden kopyasız dilimlenir.
- Gerçek zamandan sapma ("gecikme"): olması gereken ile gönderilen arasındaki fark. Motor yetişemezse (ör. çok yüksek
  hız) örnek atlamaz, geride kalanı ara vermeden gönderir ve uyarır (gerçek cihaz gibi veri kaybettirmez).
- Bozulmalar (test için) yalnız DATA çerçevelerine uygulanır; akışın ilk çerçevesi atılmaz (alıcı ilk çerçeveden
  önceki kaybı göremez).

Bu modül Qt kullanmaz.
"""
from __future__ import annotations

import heapq
import queue
import socket
import sys
import time
import traceback
from dataclasses import asdict

import numpy as np

from spt.protocol.builder import DataFrameWriter, build_descriptor, build_marker
from spt.protocol.format import Layout
from spt.sender.signals import Impairments, SenderConfig, StreamGen, StreamSpec, config_from_dict, frame_samples

DESCRIPTOR_INTERVAL = 1.0      # s — tanımlayıcı tekrarı (protokol: en az 1 Hz)
BLOCK_SECONDS = 0.01           # üretim bloğu
MIN_BLOCK_FRAMES = 8
MAX_BURST_FRAMES = 256         # akış başına döngüde en çok (diğer akışlar aç kalmasın)
STATS_INTERVAL = 0.25          # s
COMMAND_INTERVAL = 0.05        # s
LAG_WARN_S = 0.1               # bu kadar geride kalınca "yetişemiyor"
UDP_IP_OVERHEAD = 28           # IPv4 + UDP başlığı (hat kullanımı tahmini)
ETH_OVERHEAD = 38              # Ethernet çerçevesi + ön ek + çerçeveler arası boşluk


class _StreamState:
    """Bir akışın gönderim durumu: üreteç, çerçeve yazıcı, sayaçlar."""

    def __init__(self, spec: StreamSpec, seed: int):
        self.spec = spec
        self.gen = StreamGen(spec, seed)
        desc = self.gen.desc
        self.desc = desc
        self.fs = float(spec.fs)
        self.spf = frame_samples(spec, desc)
        self.writer = DataFrameWriter(desc, self.spf, spec.crc)
        self.bps = desc.bytes_per_sample
        self.planar = desc.layout == Layout.PLANAR
        self.block_frames = max(MIN_BLOCK_FRAMES, int(BLOCK_SECONDS * self.fs / self.spf))
        self._block = None            # (başlangıç indeksi, bayt görünümü / düzlemsel dizi, uzunluk)
        self.sent = 0                 # gönderilen (ya da bozulmayla atılan) örnek sayısı
        self.seq = 0
        self.next_desc = 0
        self.marker_step = int(spec.marker_every * self.fs) if spec.marker_every > 0 else 0
        self.next_marker = self.marker_step or 0
        self.marker_no = 0
        self.packets = self.bytes = self.wire_bytes = 0
        self.lost = self.duplicated = self.reordered = self.delayed = 0
        self.held = None              # yeniden sıralama için bekletilen çerçeve
        self.lag_s = 0.0
        self.max_lag_s = 0.0
        self._prev = (0, 0, 0.0)

    def sample_bytes(self, i: int, k: int):
        """[i, i+k) örneklerinin kablodaki baytları (yerleşime göre)."""
        blk = self._block
        if blk is None or not (blk[0] <= i and i + k <= blk[0] + blk[2]):
            n = max(k, self.block_frames * self.spf)
            wire = self.gen.wire(i, n)
            if self.planar:
                blk = (i, wire, n)
            else:
                blk = (i, memoryview(wire.view(np.uint8).reshape(-1)), n)
            self._block = blk
        start, data, _ = blk
        off = i - start
        if self.planar:
            return np.ascontiguousarray(data[off:off + k].T).tobytes()
        return data[off * self.bps:(off + k) * self.bps]


class Engine:
    """Gönderim motoru. Süreç dışında da (testler) kullanılabilir: `start()`, sonra `pump()` döngüsü."""

    def __init__(self, cfg: SenderConfig, emit=None, seed: int = 1):
        self.cfg = cfg
        self._emit = emit or (lambda ev: None)
        self.imp = cfg.impairments
        self.rng = np.random.default_rng(seed + 1000)
        self.streams = [_StreamState(s, seed) for s in cfg.enabled_streams()]
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 8 << 20)
        # Açık bağlama: kaynak portu baştan belli olur (alıcı akışları IP + akış kimliği + port ile izler)
        self.sock.bind((cfg.bind or "0.0.0.0", 0))
        self.addr = (cfg.host, cfg.port)
        self._heap: list[tuple[float, int, bytes, _StreamState]] = []   # (gönderim zamanı, sıra, çerçeve, akış)
        self._heap_seq = 0
        self.errors = 0
        self.last_error = ""
        self.t0 = 0.0
        self._next_stats = 0.0

    # ---------------------------------------------------------------- gönderim
    def start(self) -> None:
        self.t0 = time.perf_counter()
        self._next_stats = self.t0 + STATS_INTERVAL

    def set_impairments(self, imp: Impairments) -> None:
        self.imp = imp

    def _sendto(self, frame, st: _StreamState | None) -> None:
        try:
            self.sock.sendto(frame, self.addr)
        except OSError as exc:      # ağ yok / hedef ulaşılmaz: sayılır, gönderim sürer
            self.errors += 1
            self.last_error = f"{type(exc).__name__}: {exc.strerror or exc}"
            return
        if st is not None:
            st.packets += 1
            st.bytes += len(frame)
            st.wire_bytes += len(frame) + UDP_IP_OVERHEAD + ETH_OVERHEAD

    def _control_frame(self, st: _StreamState, frame: bytes) -> None:
        """Tanımlayıcı / işaret: bozulmasız, hemen (sırası DATA'dan önce gelmeli)."""
        self._sendto(frame, None)
        st.seq += 1

    def _data_frame(self, st: _StreamState, frame, now: float, first: bool) -> None:
        imp = self.imp
        if not imp.active:
            self._sendto(frame, st)
            return
        rng = self.rng
        if not first and imp.loss_pct and rng.random() * 100 < imp.loss_pct:
            st.lost += 1
            return
        frame = bytes(frame)                      # yazıcının tamponu yeniden kullanılır: kopya
        copies = 2 if imp.duplicate_pct and rng.random() * 100 < imp.duplicate_pct else 1
        st.duplicated += copies - 1
        if not first and imp.reorder_pct and st.held is None and rng.random() * 100 < imp.reorder_pct:
            st.held = frame                       # bir sonraki çerçeveden sonra gönderilir
            st.reordered += 1
            return
        frames = [frame] * copies
        if st.held is not None:
            frames.append(st.held)
            st.held = None
        for f in frames:
            if imp.jitter_ms > 0:
                delay = abs(rng.normal(0.0, imp.jitter_ms / 1000.0))
                st.delayed += 1
                heapq.heappush(self._heap, (now + delay, self._heap_seq, f, st))
                self._heap_seq += 1
            else:
                self._sendto(f, st)

    def pump(self, now: float | None = None) -> float:
        """Vadesi gelen çerçeveleri gönderir; bir sonraki işin zamanını (perf_counter) döndürür."""
        now = time.perf_counter() if now is None else now
        elapsed = now - self.t0
        soonest = now + 0.005
        for st in self.streams:
            due = int(elapsed * st.fs)
            burst = 0
            while due - st.sent >= st.spf and burst < MAX_BURST_FRAMES:
                ts = int(st.sent / st.fs * 1e9)
                if st.sent >= st.next_desc:
                    self._control_frame(st, build_descriptor(st.desc, st.seq, ts, st.spec.crc))
                    st.next_desc = st.sent + max(1, int(DESCRIPTOR_INTERVAL * st.fs))
                if st.marker_step and st.sent >= st.next_marker:
                    st.marker_no += 1
                    self._control_frame(st, build_marker(st.desc.stream_id, st.desc.desc_rev, st.seq, ts, st.sent,
                                                         st.marker_no, f"işaret {st.marker_no}", st.spec.crc))
                    st.next_marker = st.sent + st.marker_step
                frame = st.writer.write(st.seq, ts, st.sent, st.spf, st.sample_bytes(st.sent, st.spf))
                st.seq += 1
                self._data_frame(st, frame, now, first=st.sent == 0)
                st.sent += st.spf
                burst += 1
            st.lag_s = max(0.0, (due - st.sent) / st.fs - st.spf / st.fs)
            st.max_lag_s = max(st.max_lag_s, st.lag_s)
            soonest = min(soonest, self.t0 + (st.sent + st.spf) / st.fs)
        while self._heap and self._heap[0][0] <= now:
            _, _, f, st = heapq.heappop(self._heap)
            self._sendto(f, st)
        if self._heap:
            soonest = min(soonest, self._heap[0][0])
        if now >= self._next_stats:
            self._next_stats = now + STATS_INTERVAL
            self._emit(("stats", self.stats(now)))
        return soonest

    def stats(self, now: float | None = None) -> dict:
        now = time.perf_counter() if now is None else now
        rows = []
        total_bps = total_wire = 0.0
        for st in self.streams:
            packets0, bytes0, t_prev = st._prev
            dt = max(now - t_prev, 1e-6) if t_prev else max(now - self.t0, 1e-6)
            pps = (st.packets - packets0) / dt
            bps = (st.bytes - bytes0) / dt
            st._prev = (st.packets, st.bytes, now)
            total_bps += bps
            total_wire += bps + pps * (UDP_IP_OVERHEAD + ETH_OVERHEAD)
            rows.append({"stream_id": st.desc.stream_id, "name": st.desc.name, "packets": st.packets,
                         "bytes": st.bytes, "samples": st.sent, "pps": pps, "bytes_per_s": bps,
                         "lag_ms": st.lag_s * 1e3, "max_lag_ms": st.max_lag_s * 1e3, "late": st.lag_s > LAG_WARN_S,
                         "lost": st.lost, "duplicated": st.duplicated, "reordered": st.reordered,
                         "nominal_bytes_per_s": st.fs * st.bps})
        return {"t": now - self.t0, "streams": rows, "bytes_per_s": total_bps, "wire_bits_per_s": total_wire * 8,
                "errors": self.errors, "last_error": self.last_error}

    def flush(self) -> None:
        """Bekleyen (titreşimle geciktirilmiş / yeniden sıralanmış) çerçeveleri hemen gönderir."""
        for st in self.streams:
            if st.held is not None:
                self._sendto(st.held, st)
                st.held = None
        while self._heap:
            _, _, f, st = heapq.heappop(self._heap)
            self._sendto(f, st)

    def close(self) -> None:
        self.sock.close()

    # ---------------------------------------------------------------- döngü
    def run(self, commands=None, duration: float = 0.0) -> None:
        """Durdurulana (komut "stop") ya da `duration` saniye dolana kadar gönderir."""
        self.start()
        next_cmd = self.t0
        while True:
            soonest = self.pump()
            now = time.perf_counter()
            if duration and now - self.t0 >= duration:
                break
            if commands is not None and now >= next_cmd:
                next_cmd = now + COMMAND_INTERVAL
                if self._commands(commands):
                    break
            wait = soonest - time.perf_counter()
            if wait > 0.0005:
                time.sleep(min(wait, 0.002))
        self.flush()
        self._emit(("stats", self.stats()))

    def _commands(self, commands) -> bool:
        while True:
            try:
                cmd = commands.get_nowait()
            except queue.Empty:
                return False
            if cmd[0] == "stop":
                return True
            if cmd[0] == "impairments":
                self.set_impairments(Impairments(**cmd[1]))


def _fine_timer(on: bool) -> None:
    """Windows: zamanlayıcı çözünürlüğü 1 ms (varsayılan ≈ 15.6 ms; time.sleep(0.001) 15 ms uyurdu)."""
    if sys.platform != "win32":
        return
    try:
        import ctypes

        (ctypes.windll.winmm.timeBeginPeriod if on else ctypes.windll.winmm.timeEndPeriod)(1)
    except (AttributeError, OSError):
        pass


def run_engine(config: dict, commands, events, duration: float = 0.0) -> None:
    """Süreç giriş noktası: `config` = config_to_dict(SenderConfig). Olaylar: ("started", bilgi), ("stats", dict),
    ("error", ileti), ("stopped",)."""
    emit = events.put
    _fine_timer(True)
    engine = None
    try:
        cfg = config_from_dict(config)
        engine = Engine(cfg, emit)
        emit(("started", {"host": cfg.host, "port": cfg.port, "local": engine.sock.getsockname(),
                          "streams": [asdict(st.spec) | {"spf": st.spf} for st in engine.streams]}))
        engine.run(commands, duration)
    except Exception:  # noqa: BLE001 — her hata arayüze iletilmeli
        emit(("error", "Gönderim motoru durdu:\n" + traceback.format_exc()))
    finally:
        if engine is not None:
            engine.close()
        _fine_timer(False)
        emit(("stopped",))
