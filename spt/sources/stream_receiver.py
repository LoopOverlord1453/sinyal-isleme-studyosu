"""USB (sanal COM portu) ve CAN-bus alıcı süreçleri.

UDP alıcısı gibi arayüzden ayrı bir süreçte çalışır ve aynı olayları üretir (bkz. receiver.py, stream_store.py).
Fark, verinin datagram değil bayt akışı olarak gelmesidir: her gönderici (USB portu ya da CAN kimliği) kendi
`StreamDeframer`'ına beslenir, ayrılan çerçeveler `StreamStore`'a UDP datagramı gibi verilir.

Bağlantı koparsa (kart resetlendi, USB kablosu çıktı, CAN adaptörü takılı değil) alıcı durmaz; RETRY_S aralıkla
yeniden bağlanmayı dener. Yalnızca ayarın kendisi hatalıysa (ör. adaptörün sürücüsü kurulu değil) süreç durur.
Bu modül Qt import etmez.
"""
from __future__ import annotations

import logging
import time
import traceback
from dataclasses import dataclass

from spt.core.stream_store import StreamStore
from spt.protocol.parser import FrameError
from spt.protocol.stream import StreamDeframer
from spt.sources.receiver import COMMAND_INTERVAL, STATS_INTERVAL, _commands, _send_stats, rx_clock

RETRY_S = 1.0             # kopan / açılamayan bağlantıyı yeniden deneme aralığı
MAX_CAN_BATCH = 4096      # bir okumada en çok CAN mesajı (komut ve istatistik aralığı aksamasın)
STM32_VID = 0x0483        # STMicroelectronics USB üretici kimliği (CDC sanal COM, ST-LINK VCP)

SERIAL_BAUDS = (9600, 115200, 230400, 460800, 921600, 1_000_000, 2_000_000, 3_000_000)
CAN_BITRATES = (125_000, 250_000, 500_000, 800_000, 1_000_000)
CAN_DATA_BITRATES = (1_000_000, 2_000_000, 4_000_000, 5_000_000, 8_000_000)
CAN_DEFAULT_ID = 0x6F0
CAN_DEFAULT_MASK = 0x7F0
# python-can arayüz adı, gösterilen ad, örnek kanal
CAN_INTERFACES = (
    ("pcan", "PCAN", "PCAN_USBBUS1"),           # PEAK PCAN-USB (PCAN-Basic sürücüsü)
    ("kvaser", "Kvaser", "0"),                  # Kvaser CANlib
    ("vector", "Vector", "0"),                  # Vector XL sürücüsü
    ("ixxat", "IXXAT", "0"),                    # HMS IXXAT VCI
    ("slcan", "slcan", "COM7"),                 # CANable, USBtin … (seri hat üzerinden CAN)
)


class LinkError(Exception):
    """Bağlantı şu an kurulamadı / koptu; yeniden denenebilir."""


class FatalLinkError(Exception):
    """Ayar ya da sürücü hatası; yeniden denemek işe yaramaz."""


@dataclass(frozen=True)
class SerialConfig:
    port: str = ""               # COM5, /dev/ttyACM0 ya da pyserial adresi (socket://…, rfc2217://…)
    baudrate: int = 115200       # USB CDC'de önemsiz; ST-LINK sanal COM / USB-UART köprüsünde kartın UART hızı
    ring_seconds: float = 10.0
    max_ring_mb: int = 256


@dataclass(frozen=True)
class CanConfig:
    interface: str = "pcan"
    channel: str = "PCAN_USBBUS1"
    bitrate: int = 500_000
    fd: bool = False
    data_bitrate: int = 2_000_000
    can_id: int = CAN_DEFAULT_ID     # kabul edilen kimlikler: (kimlik & maske) == (can_id & maske)
    can_mask: int = CAN_DEFAULT_MASK
    extended: bool = False           # 29 bit kimlik
    ring_seconds: float = 10.0
    max_ring_mb: int = 256

    def id_text(self) -> str:
        width = 8 if self.extended else 3
        return f"0x{self.can_id:0{width}X} / 0x{self.can_mask:0{width}X}"

    def describe(self) -> str:
        rate = f"{self.bitrate // 1000} kbit/s"
        if self.fd:
            rate += f" · FD {self.data_bitrate / 1e6:g} Mbit/s"
        return f"CAN {self.channel} · {rate}"


def can_source(can_id: int, extended: bool) -> str:
    """CAN göndericisinin kaynak adı (akış anahtarı ve Kaynaklar ağacı): her kimlik ayrı bir göndericidir."""
    return f"CAN 0x{can_id:08X}" if extended else f"CAN 0x{can_id:03X}"


def list_serial_ports() -> list[tuple[str, str, bool]]:
    """Bilgisayardaki seri portlar: (aygıt, açıklama, STM32 mi). STM32 aygıtları başta."""
    try:
        from serial.tools import list_ports
    except ImportError:
        return []
    out = []
    for p in list_ports.comports():
        stm32 = p.vid == STM32_VID
        desc = (p.description or "").replace(f"({p.device})", "").strip()
        text = p.device if not desc or desc == "n/a" else f"{p.device} · {desc}"
        out.append((p.device, text, stm32))
    out.sort(key=lambda item: (not item[2], _port_order(item[0])))
    return out


def _port_order(device: str) -> tuple:
    digits = "".join(ch for ch in device if ch.isdigit())
    return (device.rstrip("0123456789"), int(digits) if digits else 0)


# ---------------------------------------------------------------------------------------------------- bağlantılar
class _SerialLink:
    """USB CDC sanal COM portu (STM32 USB aygıtı ya da ST-LINK sanal COM). pyserial adresleri de kabul edilir."""
    kind = "usb"

    def __init__(self, cfg: SerialConfig):
        self.cfg = cfg
        self.title = f"USB {cfg.port}"
        self.source = self.title
        self._ser = None

    @property
    def is_open(self) -> bool:
        return self._ser is not None

    def open(self) -> None:
        try:
            import serial
        except ImportError as exc:
            raise FatalLinkError("USB için pyserial paketi gerekli: pip install pyserial") from exc
        if not self.cfg.port:
            raise FatalLinkError("USB portu seçilmedi")
        try:
            self._ser = serial.serial_for_url(self.cfg.port, baudrate=self.cfg.baudrate, timeout=COMMAND_INTERVAL)
        except ValueError as exc:
            raise FatalLinkError(f"{self.title}: geçersiz ayar: {exc}") from exc
        except (serial.SerialException, OSError) as exc:
            raise LinkError(f"{self.title} açılamadı: {exc}") from exc
        try:   # STM32 CDC örnekleri çoğunlukla PC portu açınca (DTR) göndermeye başlar
            self._ser.dtr = True
        except (serial.SerialException, OSError, ValueError, AttributeError):
            pass

    def read(self) -> list[tuple[str, bytes]]:
        import serial
        ser = self._ser
        try:
            data = ser.read(max(1, ser.in_waiting))   # veri yoksa en fazla COMMAND_INTERVAL bekler
            waiting = ser.in_waiting
            if waiting:
                data += ser.read(waiting)
        except (serial.SerialException, OSError) as exc:
            raise LinkError(f"{self.title} bağlantısı koptu: {exc}") from exc
        return [(self.source, data)] if data else []

    def close(self) -> None:
        if self._ser is not None:
            try:
                self._ser.close()
            except Exception:  # noqa: BLE001 — kopmuş portun kapatılması hata verebilir
                pass
            self._ser = None

    def info(self) -> dict:
        return {"address": self.title, "baudrate": self.cfg.baudrate}


class _CanLink:
    """CAN / CAN FD adaptörü (python-can). Her CAN kimliği ayrı bir gönderici (kaynak) sayılır."""
    kind = "can"

    def __init__(self, cfg: CanConfig):
        self.cfg = cfg
        self.title = cfg.describe()
        self._bus = None
        self._mask = cfg.can_mask
        self._match = cfg.can_id & cfg.can_mask
        self.messages = 0          # kabul edilen (süzgeçten geçen) veri mesajı
        self.error_frames = 0
        self.ignored = 0           # süzgeç dışı kimlik, uzak çerçeve ya da farklı kimlik biçimi
        self._names: dict[int, str] = {}

    @property
    def is_open(self) -> bool:
        return self._bus is not None

    def open(self) -> None:
        try:
            import can
        except ImportError as exc:
            raise FatalLinkError("CAN-bus için python-can paketi gerekli: pip install python-can") from exc
        cfg = self.cfg
        kwargs: dict = {
            "interface": cfg.interface, "channel": cfg.channel or None, "receive_own_messages": False,
            "can_filters": [{"can_id": cfg.can_id, "can_mask": cfg.can_mask, "extended": cfg.extended}],
        }
        if cfg.fd and cfg.interface == "pcan":    # PCAN FD bit zamanlaması saat frekansıyla verilir
            kwargs["timing"] = can.BitTimingFd.from_sample_point(
                f_clock=80_000_000, nom_bitrate=cfg.bitrate, nom_sample_point=80.0,
                data_bitrate=cfg.data_bitrate, data_sample_point=80.0)
        else:
            kwargs["bitrate"] = cfg.bitrate
            if cfg.fd:
                kwargs.update(fd=True, data_bitrate=cfg.data_bitrate)
        try:
            self._bus = can.Bus(**kwargs)
        except (can.CanInterfaceNotImplementedError, ImportError, NotImplementedError, ValueError, TypeError) as exc:
            raise FatalLinkError(self._driver_message(exc)) from exc
        except (can.CanError, OSError) as exc:
            text = str(exc).lower()
            if "library" in text or ".dll" in text:   # ör. PCAN: "PCANBasic library not found"
                raise FatalLinkError(self._driver_message(exc)) from exc
            raise LinkError(f"{self.title} açılamadı: {exc}") from exc   # adaptör takılı değil / kanal meşgul
        except Exception as exc:  # noqa: BLE001 — bazı arayüzler sürücü yokken başka hata verir (ör. Kvaser)
            raise FatalLinkError(self._driver_message(exc)) from exc

    def _driver_message(self, exc: Exception) -> str:
        return (f"{self.title}: adaptör açılamadı ({type(exc).__name__}: {exc}). Adaptörün sürücüsü kurulu mu, "
                "arayüz ve kanal doğru mu?")

    def read(self) -> list[tuple[str, bytes]]:
        import can
        bus = self._bus
        chunks: dict[str, bytearray] = {}
        try:
            msg = bus.recv(COMMAND_INTERVAL)
            count = 0
            while msg is not None:
                if msg.is_error_frame:
                    self.error_frames += 1
                elif (msg.is_remote_frame or msg.is_extended_id != self.cfg.extended
                      or msg.arbitration_id & self._mask != self._match):
                    self.ignored += 1   # bazı adaptörler donanım süzgecini uygulamaz
                else:
                    self.messages += 1
                    name = self._names.get(msg.arbitration_id)
                    if name is None:
                        name = self._names[msg.arbitration_id] = can_source(msg.arbitration_id, self.cfg.extended)
                    chunks.setdefault(name, bytearray()).extend(msg.data)
                count += 1
                if count >= MAX_CAN_BATCH:
                    break
                msg = bus.recv(0)
        except (can.CanError, OSError) as exc:
            raise LinkError(f"{self.title} bağlantısı koptu: {exc}") from exc
        return [(name, bytes(data)) for name, data in chunks.items()]

    def close(self) -> None:
        if self._bus is not None:
            try:
                self._bus.shutdown()
            except Exception:  # noqa: BLE001
                pass
            self._bus = None

    def info(self) -> dict:
        return {"address": self.title, "can_filter": self.cfg.id_text(), "can_messages": self.messages,
                "can_error_frames": self.error_frames, "can_ignored": self.ignored}


# ---------------------------------------------------------------------------------------------------- süreçler
def run_serial_receiver(cfg: SerialConfig, commands, events) -> None:
    """USB alıcı süreci giriş noktası. `commands`/`events`: multiprocessing kuyrukları."""
    _run(_SerialLink(cfg), cfg, commands, events)


def run_can_receiver(cfg: CanConfig, commands, events) -> None:
    """CAN-bus alıcı süreci giriş noktası."""
    _run(_CanLink(cfg), cfg, commands, events)


def _run(link, cfg, commands, events) -> None:
    emit = events.put
    try:
        serve_link(link, cfg, commands, emit)
    except Exception:  # noqa: BLE001 — süreç içindeki her hata arayüze iletilmeli
        emit(("error", f"{link.title} alıcısı beklenmedik şekilde durdu:\n" + traceback.format_exc()))
    finally:
        emit(("stopped",))


def serve_link(link, cfg, commands, emit) -> None:
    """Bağlantıdan okur, çerçeveleri ayırıp akış deposuna verir; komutları ve istatistikleri UDP alıcısıyla aynı
    aralıklarla işler. Bağlantı koparsa RETRY_S aralıkla yeniden dener."""
    base, perf_ns = rx_clock()
    store = StreamStore(emit, base + perf_ns(), cfg.ring_seconds, cfg.max_ring_mb << 20)
    deframers: dict[str, StreamDeframer] = {}
    clock = time.perf_counter
    next_cmd = next_stats = retry_at = clock()
    prev = (0, 0, clock())
    recorder = None
    waiting_logged = False
    emit(("state", "starting", link.title))
    try:
        while True:
            now = clock()
            if not link.is_open and now >= retry_at:
                try:
                    link.open()
                except FatalLinkError as exc:
                    emit(("error", str(exc)))
                    return
                except LinkError as exc:
                    retry_at = now + RETRY_S
                    if not waiting_logged:
                        waiting_logged = True
                        emit(("log", logging.WARNING, f"{exc} — {RETRY_S:g} s aralıkla yeniden denenecek"))
                        emit(("state", "listening", f"{link.title} · cihaz bekleniyor"))
                else:
                    waiting_logged = False
                    for d in deframers.values():
                        d.reset()
                    emit(("state", "listening", link.title))
                    emit(("log", logging.INFO, f"{link.title} dinleniyor"))
            if link.is_open:
                try:
                    chunks = link.read()
                except LinkError as exc:
                    link.close()
                    retry_at = clock() + RETRY_S
                    emit(("log", logging.WARNING, f"{exc} — yeniden bağlanılacak"))
                    emit(("state", "listening", f"{link.title} · cihaz bekleniyor"))
                    waiting_logged = True
                    chunks = []
                if chunks:
                    rx_ns = base + perf_ns()
                    for source, data in chunks:
                        d = deframers.get(source)
                        if d is None:
                            d = deframers[source] = StreamDeframer(
                                lambda msg, s=source: store.report_frame_error(s, FrameError(msg)))
                        for frame in d.feed(data):
                            store.handle_datagram(memoryview(frame), len(frame), source, rx_ns, 0)
            else:
                time.sleep(COMMAND_INTERVAL)

            now = clock()
            if now < next_cmd:
                continue
            next_cmd = now + COMMAND_INTERVAL
            stop, recorder = _commands(commands, store, recorder, emit)
            if stop:
                break
            if now >= next_stats:
                next_stats = now + STATS_INTERVAL
                extra = {"transport": link.kind, "connected": link.is_open,
                         "stream_bytes": sum(d.bytes for d in deframers.values()),
                         "skipped_bytes": sum(d.skipped for d in deframers.values()),
                         "resyncs": sum(d.resyncs for d in deframers.values()), **link.info()}
                prev = _send_stats(store, recorder, prev, now, emit, base + perf_ns(), extra)
    finally:
        if recorder is not None:
            recorder.close()
            emit(("record", False, str(recorder.path), recorder.bytes_written))
        store.close()
        link.close()
