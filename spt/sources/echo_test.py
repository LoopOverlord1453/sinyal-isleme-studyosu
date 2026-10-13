"""UDP yankı (RTT) testi: hedefe sıra numaralı, zaman damgalı problar gönderir, geri dönenleri eşler.

Ölçülenler: gidiş-dönüş süresi (RTT) dağılımı, titreşim (RFC 3550 tarzı ardışık fark ortalaması ve standart
sapma), kayıp, geç gelen / yinelenen / bozuk yanıt, ICMP "port kapalı" yanıtları ve yük altında verim. Hedefin
gelen datagramı aynen geri göndermesi (UDP echo, RFC 862; ör. lwIP UDP echo örneği, port 7) yeterlidir.

Prob: [b"SPTE"][u32 sıra][u64 gönderim zamanı, ns (perf_counter)] + sıfır dolgu (toplam `payload` bayt).
Test kendi geçici yerel portunu kullanır; aynı makinede dinleyen SPTF alıcısıyla çakışmaz.

Arayüz testi ve yerel yankı sunucusunu AYRI SÜREÇLERDE çalıştırır (`run_echo_process`, `run_echo_server`): aynı
süreçte Python'un genel kilidi (GIL, 5 ms geçiş aralığı) arayüz çizimi sırasında yanıtın alınmasını geciktirir ve
RTT'yi milisaniyelerce şişirir (ölçüm: iş parçacığıyla p99 ≈ 97 ms, ayrı süreçle µs düzeyi). Süreçlere yalnız
küçük özetler (`EchoSnapshot`) taşınır. Bu modül Qt kullanmaz.
"""
from __future__ import annotations

import math
import select
import socket
import struct
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from spt.core.stream_store import TIMING_EDGES_US

ECHO_MAGIC = b"SPTE"
PROBE = struct.Struct("<4sIQ")
MIN_PAYLOAD = PROBE.size
MAX_PAYLOAD = 65507
DEFAULT_ECHO_PORT = 7
UDP_IP_OVERHEAD = 28            # IPv4 (20) + UDP (8) başlığı: hat verimi için
ETHERNET_OVERHEAD = 38          # Ethernet çerçeve başlığı + FCS + önsöz + çerçeveler arası boşluk (bayt)
SPIN_BELOW_S = 0.02             # bundan kısa beklemeler meşgul döngüyle (Windows'ta select ≈ 15 ms adımlı)
MAX_LAG_S = 0.05                # gönderim bu kadar geride kalırsa yetişmek için patlama yapılmaz, takvim kayar
NO_REPLY_HINT = ("Kartta UDP yankı sunucusu gerekiyor (ör. lwIP UDP echo, port 7); kart yazılımı bu sürümde "
                 "değiştirilmedi.")
MAX_PLOT_POINTS = 20_000        # özetle taşınan RTT noktası (seyreltilmiş)


@dataclass(frozen=True)
class EchoConfig:
    host: str
    port: int = DEFAULT_ECHO_PORT
    payload: int = 64           # UDP yük boyutu (bayt)
    rate_hz: float = 100.0      # prob gönderim hızı
    count: int = 1000
    timeout_s: float = 1.0      # bir yanıtın en geç bu sürede gelmesi beklenir; son probdan sonra da bu kadar beklenir

    def validate(self) -> None:
        if not self.host:
            raise ValueError("Hedef IP adresi girin.")
        if not 1 <= self.port <= 65535:
            raise ValueError("Port 1 … 65535 olmalı.")
        if not MIN_PAYLOAD <= self.payload <= MAX_PAYLOAD:
            raise ValueError(f"Yük {MIN_PAYLOAD} … {MAX_PAYLOAD} bayt olmalı.")
        if not 0.1 <= self.rate_hz <= 200_000:
            raise ValueError("Hız 0.1 … 200 000 paket/s olmalı.")
        if not 1 <= self.count <= 10_000_000:
            raise ValueError("Paket sayısı 1 … 10 000 000 olmalı.")
        if not 0.001 <= self.timeout_s <= 60:
            raise ValueError("Zaman aşımı 1 ms … 60 s olmalı.")


@dataclass(frozen=True)
class EchoSnapshot:
    """Süreçler arası taşınan küçük ara / son sonuç (RTT dizisinin tamamı taşınmaz)."""
    config: EchoConfig
    sent: int
    received: int
    late: int
    duplicates: int
    invalid: int
    unreachable: int
    send_slips: int
    lost: int
    loss_pct: float
    elapsed_s: float
    stats: dict
    throughput: tuple[float, float]       # (UDP yükü, hat) Mbit/s
    plot_seq: np.ndarray                  # seyreltilmiş (sıra no, RTT µs) noktaları
    plot_rtt: np.ndarray
    hist: np.ndarray                      # RTT histogramı, TIMING_EDGES_US kutuları
    finished: bool
    stopped: bool
    error: str
    diagnosis: str


@dataclass
class EchoResult:
    config: EchoConfig
    rtt_us: np.ndarray                      # sıra numarasına göre RTT (µs); yanıt gelmeyen NaN
    sent: int = 0
    received: int = 0                       # zaman aşımından önce gelen tekil yanıt
    late: int = 0                           # zaman aşımından sonra gelen (kayıp sayılır)
    duplicates: int = 0
    invalid: int = 0                        # tanınmayan / bozuk yanıt
    unreachable: int = 0                    # ICMP port kapalı (Windows: WSAECONNRESET)
    send_slips: int = 0                     # gönderim takvimi yetişemeyip kaydırıldı (PC yükü)
    elapsed_s: float = 0.0
    finished: bool = False
    stopped: bool = False
    error: str = ""
    extra: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ özet
    @property
    def valid(self) -> np.ndarray:
        r = self.rtt_us[:self.sent]
        return r[np.isfinite(r)]

    @property
    def lost(self) -> int:
        """Yanıtı beklenen süre içinde gelmeyen prob (bitmemiş testte son `timeout` içinde gönderilenler hariç)."""
        return max(self.extra.get("due", self.sent) - self.received, 0)

    @property
    def loss_pct(self) -> float:
        due = self.extra.get("due", self.sent)
        return 100.0 * self.lost / due if due else math.nan

    def stats(self) -> dict[str, float]:
        r = self.valid
        if not len(r):
            nan = math.nan
            return {"min": nan, "mean": nan, "p50": nan, "p95": nan, "p99": nan, "max": nan, "std": nan,
                    "jitter": nan}
        ordered = self.rtt_us[:self.sent]
        ordered = ordered[np.isfinite(ordered)]
        jitter = float(np.mean(np.abs(np.diff(ordered)))) if len(ordered) > 1 else math.nan
        return {"min": float(r.min()), "mean": float(r.mean()), "p50": float(np.percentile(r, 50)),
                "p95": float(np.percentile(r, 95)), "p99": float(np.percentile(r, 99)), "max": float(r.max()),
                "std": float(r.std()), "jitter": jitter}

    def throughput_mbps(self) -> tuple[float, float]:
        """(UDP yükü, hat) verimi, Mbit/s: zamanında dönen yanıtların tek yön verisi / test süresi."""
        if self.elapsed_s <= 0:
            return math.nan, math.nan
        p = self.config.payload
        payload = self.received * p * 8 / self.elapsed_s / 1e6
        wire = self.received * (p + UDP_IP_OVERHEAD + ETHERNET_OVERHEAD) * 8 / self.elapsed_s / 1e6
        return payload, wire

    def snapshot(self, max_points: int = MAX_PLOT_POINTS) -> EchoSnapshot:
        n = self.sent
        step = max(1, n // max_points)
        idx = np.arange(0, n, step)
        sel = self.rtt_us[idx]
        ok = np.isfinite(sel)
        valid = self.valid
        hist = np.bincount(np.searchsorted(TIMING_EDGES_US, valid, side="right"),
                           minlength=len(TIMING_EDGES_US) + 1)[:len(TIMING_EDGES_US) + 1]
        return EchoSnapshot(self.config, self.sent, self.received, self.late, self.duplicates, self.invalid,
                            self.unreachable, self.send_slips, self.lost, self.loss_pct, self.elapsed_s, self.stats(),
                            self.throughput_mbps(), idx[ok], sel[ok], hist, self.finished, self.stopped, self.error,
                            self.diagnosis())

    def diagnosis(self) -> str:
        """Yanıt yoksa ya da port kapalıysa kullanıcıya açıklama; sorun yoksa boş."""
        if self.error:
            return self.error
        if self.received == 0 and self.sent:
            if self.unreachable:
                return f"Hedef portu kapalı (ICMP: port ulaşılamaz). {NO_REPLY_HINT}"
            return f"Hiç yanıt gelmedi (zaman aşımı). {NO_REPLY_HINT}"
        return ""


class EchoTest:
    """`stop_event`: is_set() / set() olan herhangi bir olay (threading.Event ya da süreçler arası Event)."""

    def __init__(self, cfg: EchoConfig, progress: Callable[[EchoResult], None] | None = None,
                 progress_interval_s: float = 0.1, stop_event=None):
        cfg.validate()
        self.cfg = cfg
        self._progress = progress
        self._interval = progress_interval_s
        self._stop = stop_event if stop_event is not None else threading.Event()
        self.result = EchoResult(cfg, np.full(cfg.count, np.nan))

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> EchoResult:
        cfg, res = self.cfg, self.result
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        except OSError as exc:
            res.error = f"Soket açılamadı: {exc}"
            res.finished = True
            return res
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 << 20)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 8 << 20)
            sock.bind(("0.0.0.0", 0))
            sock.setblocking(False)
            target = (socket.gethostbyname(cfg.host), cfg.port)
            self._loop(sock, target)
        except OSError as exc:
            res.error = f"Ağ hatası: {exc.strerror or exc}"
        finally:
            sock.close()
            res.finished = True
            if self._progress is not None:
                self._progress(res)
        return res

    def _loop(self, sock: socket.socket, target: tuple[str, int]) -> None:
        cfg, res = self.cfg, self.result
        perf = time.perf_counter_ns
        period = 1e9 / cfg.rate_hz
        timeout_ns = int(cfg.timeout_s * 1e9)
        packet = bytearray(cfg.payload)
        rbuf = bytearray(max(cfg.payload, 2048))
        rtt = res.rtt_us
        sent_at = np.zeros(cfg.count, np.int64)
        start = perf()
        next_send = float(start)
        next_report = start + int(self._interval * 1e9)
        last_send = start
        while not self._stop.is_set():
            now = perf()
            while res.sent < cfg.count and now >= next_send:
                if now - next_send > MAX_LAG_S * 1e9:
                    next_send = float(now)         # PC yetişemedi: patlama yapma, takvimi kaydır
                    res.send_slips += 1
                PROBE.pack_into(packet, 0, ECHO_MAGIC, res.sent, now)
                try:
                    sock.sendto(packet, target)
                except BlockingIOError:
                    break                          # gönderim tamponu dolu: sonraki turda yeniden dene
                except (ConnectionResetError, ConnectionRefusedError):
                    res.unreachable += 1
                sent_at[res.sent] = now
                res.sent += 1
                last_send = now
                next_send += period
                now = perf()
            # Yanıtları boşalt
            while True:
                try:
                    n, addr = sock.recvfrom_into(rbuf)
                except BlockingIOError:
                    break
                except (ConnectionResetError, ConnectionRefusedError):   # önceki probun ICMP "port ulaşılamaz" yanıtı
                    res.unreachable += 1
                    continue
                t = perf()
                if n < PROBE.size or addr[0] != target[0]:
                    res.invalid += 1
                    continue
                magic, seq, t_send = PROBE.unpack_from(rbuf, 0)
                if magic != ECHO_MAGIC or seq >= res.sent or sent_at[seq] != t_send or n != cfg.payload:
                    res.invalid += 1
                    continue
                if not math.isnan(rtt[seq]) or (seq in res.extra.get("late_seqs", ())):
                    res.duplicates += 1
                    continue
                d = t - t_send
                if d > timeout_ns:
                    res.late += 1
                    res.extra.setdefault("late_seqs", set()).add(seq)
                    continue
                rtt[seq] = d / 1e3
                res.received += 1
            now = perf()
            res.elapsed_s = (now - start) / 1e9
            # Zamanı dolmuş problar: gönderimden `timeout` geçmiş olanlar kayıp hesabına girer
            res.extra["due"] = int(np.searchsorted(sent_at[:res.sent], now - timeout_ns, side="right"))
            if now >= next_report and self._progress is not None:
                next_report = now + int(self._interval * 1e9)
                self._progress(res)
            if res.sent >= cfg.count and (now - last_send > timeout_ns or res.received + res.late >= res.sent):
                break
            wait = (next_send - now) / 1e9 if res.sent < cfg.count else 0.01
            if wait > SPIN_BELOW_S:
                select.select([sock], [], [], wait - SPIN_BELOW_S)
            elif res.sent >= cfg.count:
                select.select([sock], [], [], 0.002)
            # kısa bekleme: meşgul döngü (Windows'ta select'in ms altı çözünürlüğü yok)
        res.stopped = self._stop.is_set()
        if not res.stopped:
            res.extra["due"] = res.sent
        res.elapsed_s = (perf() - start) / 1e9


class EchoServer:
    """Yerel UDP yankı sunucusu (test ve gösterim için): gelen her datagramı göndericisine geri yollar."""

    def __init__(self, host: str = "127.0.0.1", port: int = 0, stop_event=None):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 << 20)
        self._sock.bind((host, port))
        self._sock.setblocking(False)
        self.host, self.port = self._sock.getsockname()
        self.echoed = 0
        self._stop = stop_event if stop_event is not None else threading.Event()
        self._thread = threading.Thread(target=self._serve, name="spt-echo", daemon=True)

    def start(self) -> EchoServer:
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._sock.close()

    def _serve(self) -> None:
        buf = bytearray(MAX_PAYLOAD)
        sock = self._sock
        while not self._stop.is_set():
            try:
                select.select([sock], [], [], 0.1)
            except (OSError, ValueError):
                return
            while True:
                try:
                    n, addr = sock.recvfrom_into(buf)
                except BlockingIOError:
                    break
                except (ConnectionResetError, ConnectionRefusedError):
                    continue
                except OSError:
                    return
                try:
                    sock.sendto(memoryview(buf)[:n], addr)
                    self.echoed += 1
                except OSError:
                    pass


# ------------------------------------------------------------------------------------------------ süreç girişleri
def run_echo_process(cfg: EchoConfig, stop_event, events, interval_s: float = 0.1) -> None:
    """Yankı testi süreci: ara özetleri ("echo", EchoSnapshot) olarak `events` kuyruğuna koyar; sonuncusu
    `finished=True`. Hata da bitmiş özet olarak bildirilir."""
    try:
        EchoTest(cfg, lambda r: events.put(("echo", r.snapshot())), interval_s, stop_event).run()
    except Exception as exc:  # noqa: BLE001 — süreç içindeki her hata arayüze bitmiş sonuç olarak dönmeli
        res = EchoResult(cfg, np.full(1, np.nan), finished=True, error=f"Yankı testi durdu: {exc}")
        events.put(("echo", res.snapshot()))


def run_echo_server(host: str, port: int, stop_event, events) -> None:
    """Yerel yankı sunucusu süreci: açılınca ("ready", adres, port), açılamazsa ("error", ileti)."""
    try:
        srv = EchoServer(host, port, stop_event)
    except OSError as exc:
        events.put(("error", f"Yerel yankı sunucusu açılamadı: {exc.strerror or exc}"))
        return
    events.put(("ready", srv.host, srv.port))
    try:
        srv._serve()
    finally:
        srv._sock.close()
