"""SPTG v1 istemcisi: Sinyal Üreteci'nde tasarlanan DAC tablosunu çalışan karta UDP ile gönderir; kart tabloyu
RAM'e alır ve DAC'tan çalar (derleme ve yükleme gerekmez).

Kart tarafı ve çerçeve biçimi: stm32f4_examples/001_signal_generator (SignalGenerator/Inc/signal_remote.h);
doküman: docs/SPT-EX-001_STM32F4_Sinyal_Ureteci_Ornegi.docx.

Protokol istek - yanıt biçimindedir (küçük uçlu). Her isteğe bir yanıt beklenir; gelmezse aynı istek aynı sıra
numarasıyla yeniden gönderilir, kart yinelenen isteği yeniden işlemez.

    Başlık (8 B)   : magic "SPTG" · version u8 · message_type u8 (yanıtta | 0x80) · sequence_number u16
    INFO / START / STOP : gövde yok
    BEGIN (20 B)   : output u8 · channel_count u8 · dac_bits u8 · alignment u8 · timer_prescaler u16 ·
                     timer_autoreload u16 · timer_clock_hz u32 · sample_count u32 · table_crc32 u32
    DATA           : byte_offset u32 · tablo baytları
    Yanıt (20 B)   : status u16 · player_state u8 · signal_source u8 · timer_clock_hz u32 ·
                     table_capacity_bytes u32 · max_chunk_bytes u16 · underrun_count u16 · table_crc32 u32

Gönderim sırası: INFO (kartın zamanlayıcı saati ve sınırları) → BEGIN → DATA … → START. Zamanlayıcı ön bölücüsü ve
sayma sınırı kartın bildirdiği saatten hesaplanır; arayüzdeki "Zamanlayıcı saati" alanına bağlı değildir.

Komut satırı:
    python -m spt.protocol.sptg info  --host 192.168.172.98
    python -m spt.protocol.sptg play  --host 192.168.172.98 --kind sine --freq 1000 --output both
    python -m spt.protocol.sptg stop  --host 192.168.172.98

Bu modül Qt kullanmaz.
"""
from __future__ import annotations

import argparse
import random
import socket
import struct
import sys
import zlib
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

MAGIC = b"SPTG"
VERSION = 1
UDP_PORT = 50506            # SPTF akış portunun (50505) bir fazlası
REPLY_FLAG = 0x80
MAX_SAMPLE_COUNT = 65535    # kartta DMA aktarım sayacı 16 bittir
TIMER_BITS = 16             # TIM6

HEADER = struct.Struct("<4sBBH")
BEGIN = struct.Struct("<BBBBHHIII")
DATA_PREFIX = struct.Struct("<I")
REPLY = struct.Struct("<HBBIIHHI")


class Message(IntEnum):
    INFO = 1
    BEGIN = 2
    DATA = 3
    START = 4
    STOP = 5


class Status(IntEnum):
    OK = 0
    MESSAGE_ERR = 1
    VERSION_ERR = 2
    ARG_ERR = 3
    CLOCK_ERR = 4
    SIZE_ERR = 5
    STATE_ERR = 6
    CRC_ERR = 7
    PLAYER_ERR = 8


class Output(IntEnum):
    DAC1 = 1
    DAC2 = 2
    BOTH = 3


OUTPUT_LABELS: dict[Output, str] = {Output.DAC1: "DAC1 (PA4)", Output.DAC2: "DAC2 (PA5)",
                                    Output.BOTH: "DAC1 (PA4) + DAC2 (PA5)"}

STATUS_TEXT: dict[int, str] = {
    Status.OK: "tamam",
    Status.MESSAGE_ERR: "kart isteği tanımadı (istek türü ya da gövde uzunluğu)",
    Status.VERSION_ERR: "kart bu protokol sürümünü desteklemiyor",
    Status.ARG_ERR: "tablo tanımı geçersiz (çıkış, kanal, çözünürlük, hizalama, örnek sayısı ya da ARR)",
    Status.CLOCK_ERR: "zamanlayıcı saati karttakiyle uyuşmuyor",
    Status.SIZE_ERR: "tablo kartta ayrılan belleğe sığmıyor",
    Status.STATE_ERR: "kartta yükleme başlatılmamış",
    Status.CRC_ERR: "kartta toplanan tablonun CRC'si uyuşmuyor",
    Status.PLAYER_ERR: "kart DAC / DMA / zamanlayıcıyı başlatamadı",
}

PLAYER_STATES: dict[int, str] = {0: "sinyal yok", 1: "çalışıyor", 2: "zamanlayıcı saati hatası",
                                 3: "başlatma hatası", 4: "DAC DMA alt akışı", 5: "durduruldu"}
SIGNAL_SOURCES: dict[int, str] = {0: "yok", 1: "derlenmiş sinyal", 2: "çalışma zamanı tablosu"}
PLAYER_RUNNING = 1


class SptgError(Exception):
    """Kartla konuşulamadı ya da kart isteği reddetti. `status`: kartın bildirdiği durum (yanıt yoksa None)."""

    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class BoardInfo:
    """Kartın her yanıtta bildirdiği durum."""
    status: int
    player_state: int
    signal_source: int
    timer_clock_hz: int
    table_capacity_bytes: int
    max_chunk_bytes: int
    underrun_count: int
    table_crc32: int

    @property
    def state_text(self) -> str:
        return PLAYER_STATES.get(self.player_state, f"bilinmeyen durum {self.player_state}")

    @property
    def source_text(self) -> str:
        return SIGNAL_SOURCES.get(self.signal_source, f"bilinmeyen kaynak {self.signal_source}")


@dataclass(frozen=True)
class TablePayload:
    """Karta gönderilecek tablo: DAC kelimeleri (küçük uçlu; iki kanalda serpiştirilmiş) ve tanımı."""
    data: bytes
    sample_count: int
    channel_count: int
    dac_bits: int
    left_aligned: bool
    fs: float           # tablonun tasarlandığı DAC güncelleme hızı (Hz)
    f_real: float       # o hızdaki çıkış frekansı (Hz); periyodik değilse 0

    @property
    def crc32(self) -> int:
        return zlib.crc32(self.data) & 0xFFFFFFFF


@dataclass(frozen=True)
class PlayResult:
    """Başarılı bir gönderimin sonucu. fs / f: kartın zamanlayıcısının gerçekleştirdiği değerler."""
    info: BoardInfo
    output: Output
    sample_count: int
    channel_count: int
    timer_prescaler: int
    timer_autoreload: int
    fs: float
    fs_error_ppm: float
    f: float


def payload_from_table(table: Any) -> TablePayload:
    """`spt.dsp.siggen.Table` → gönderilecek tablo. Kartın dahili DAC'ına uymayan tablo SptgError verir."""
    dac = table.dac
    if table.mode != "exact":
        raise SptgError("Karta yalnızca “Tam periyot” tablosu gönderilir; “Tablo kipi”ni “Tam periyot” seçin.")
    if dac.signed or dac.bits not in (8, 12):
        raise SptgError("Kartın dahili DAC'ı 8 ya da 12 bit, işaretsiz kod kullanır: bir “STM32 DAC” hazır ayarı seçin.")
    if dac.bits == 8 and (dac.align != "right" or dac.word_bits != 8):
        raise SptgError("STM32 8 bit modu sağa hizalıdır ve 8 bit kelime kullanır: “STM32 DAC 8 bit” hazır ayarını seçin.")
    if dac.bits == 12 and dac.word_bits != 16:
        raise SptgError("STM32 12 bit modu 16 bitlik kelime kullanır: Kelime'yi 16 bit seçin.")
    if not 1 <= table.N <= MAX_SAMPLE_COUNT:
        raise SptgError(f"Tablo {table.N} örnek; kart en fazla {MAX_SAMPLE_COUNT} örnek çalar.")
    import numpy as np

    words = np.asarray(table.flat_codes).reshape(-1)
    data = words.astype("<u2" if dac.bits == 12 else "u1").tobytes()
    return TablePayload(data=data, sample_count=int(table.N), channel_count=int(table.channels), dac_bits=int(dac.bits),
                        left_aligned=dac.align == "left", fs=float(table.fs),
                        f_real=float(table.f_real) if table.periodic else 0.0)


def encode_request(message: Message, sequence_number: int, body: bytes = b"") -> bytes:
    return HEADER.pack(MAGIC, VERSION, int(message), sequence_number & 0xFFFF) + body


def decode_reply(datagram: bytes) -> tuple[int, int, BoardInfo] | None:
    """Yanıt datagramı → (istek türü, sıra numarası, BoardInfo); bu protokolün yanıtı değilse None."""
    if len(datagram) != HEADER.size + REPLY.size:
        return None
    magic, version, message_type, sequence_number = HEADER.unpack_from(datagram)
    if magic != MAGIC or version != VERSION or not message_type & REPLY_FLAG:
        return None
    return message_type & ~REPLY_FLAG, sequence_number, BoardInfo(*REPLY.unpack_from(datagram, HEADER.size))


class SptgClient:
    """Tek karta bağlı istemci. `with SptgClient(host) as client: client.play(payload, Output.DAC1)`."""

    def __init__(self, host: str, port: int = UDP_PORT, *, timeout: float = 0.25, retries: int = 4):
        self.host, self.port = host, int(port)
        self.timeout, self.retries = float(timeout), int(retries)
        self._sequence = random.randrange(0x10000)
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self._socket.connect((host, self.port))   # yalnız kartın yanıtları alınır
        except OSError as exc:
            self._socket.close()
            raise SptgError(f"{host}:{self.port} adresi açılamadı: {exc}") from exc

    def __enter__(self) -> SptgClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        self._socket.close()

    # ---------------------------------------------------------------- istek / yanıt
    def request(self, message: Message, body: bytes = b"", *, accept: tuple[int, ...] = (Status.OK,)) -> BoardInfo:
        """İsteği gönderir, eşleşen yanıtı bekler; zaman aşımında aynı sıra numarasıyla yineler."""
        self._sequence = (self._sequence + 1) & 0xFFFF
        datagram = encode_request(message, self._sequence, body)
        for _attempt in range(self.retries + 1):
            try:
                self._socket.send(datagram)
            except OSError as exc:
                raise SptgError(f"{self.host}:{self.port} adresine gönderilemedi: {exc}") from exc
            info = self._wait_reply(int(message))
            if info is not None:
                if info.status not in accept:
                    text = STATUS_TEXT.get(info.status, f"bilinmeyen durum {info.status}")
                    raise SptgError(f"Kart {message.name} isteğini reddetti: {text}.", info.status)
                return info
        raise SptgError(f"Kart yanıt vermiyor ({self.host}:{self.port}, {self.retries + 1} deneme). Kartın açık, "
                        "Ethernet kablosunun takılı ve bilgisayarın aynı alt ağda olduğunu denetleyin.")

    def _wait_reply(self, message_type: int) -> BoardInfo | None:
        self._socket.settimeout(self.timeout)
        while True:
            try:
                datagram = self._socket.recv(2048)
            except TimeoutError:
                return None
            except ConnectionError:      # Windows: ICMP "port kapalı" yanıtı alındı; yeniden denenir
                return None
            except OSError as exc:
                raise SptgError(f"Karttan yanıt alınamadı: {exc}") from exc
            reply = decode_reply(datagram)
            if reply is not None and reply[0] == message_type and reply[1] == self._sequence:
                return reply[2]
            # önceki isteğin gecikmiş yanıtı ya da yabancı datagram: beklemeye devam

    # ---------------------------------------------------------------- işlemler
    def info(self) -> BoardInfo:
        return self.request(Message.INFO)

    def stop(self) -> BoardInfo:
        return self.request(Message.STOP)

    def play(self, payload: TablePayload, output: Output = Output.DAC1) -> PlayResult:
        """Tabloyu gönderir ve çalmaya başlatır. İki kanallı tablo her zaman iki çıkışa verilir."""
        from spt.dsp.siggen import timer_config

        output = Output.BOTH if payload.channel_count == 2 else Output(output)
        info = self.info()
        if len(payload.data) > info.table_capacity_bytes:
            raise SptgError(f"Tablo {len(payload.data)} bayt; kartta {info.table_capacity_bytes} bayt ayrılmış. "
                            "“En uzun tablo”yu küçültün.", Status.SIZE_ERR)
        timer = timer_config(payload.fs, float(info.timer_clock_hz), TIMER_BITS)
        if timer.arr < 1:
            raise SptgError(f"Örnekleme hızı {payload.fs:g} Hz kartın zamanlayıcısı için çok yüksek.", Status.ARG_ERR)
        begin = BEGIN.pack(int(output), payload.channel_count, payload.dac_bits, 1 if payload.left_aligned else 0,
                           timer.psc, timer.arr, info.timer_clock_hz, payload.sample_count, payload.crc32)
        chunk = max(1, min(info.max_chunk_bytes, 1400))
        for attempt in range(2):
            self.request(Message.BEGIN, begin)
            for offset in range(0, len(payload.data), chunk):
                self.request(Message.DATA, DATA_PREFIX.pack(offset) + payload.data[offset:offset + chunk])
            info = self.request(Message.START, accept=(Status.OK, Status.CRC_ERR))
            if info.status == Status.OK:
                break
            if attempt == 1:
                raise SptgError("Kartta toplanan tablonun CRC'si iki denemede de uyuşmadı.", Status.CRC_ERR)
        if info.player_state != PLAYER_RUNNING:
            raise SptgError(f"Kart tabloyu aldı ama çalmıyor: {info.state_text}.", info.status)
        scale = timer.fs_real / payload.fs
        return PlayResult(info=info, output=output, sample_count=payload.sample_count,
                          channel_count=payload.channel_count, timer_prescaler=timer.psc,
                          timer_autoreload=timer.arr, fs=timer.fs_real, fs_error_ppm=timer.error_ppm,
                          f=payload.f_real * scale)


def send_table(host: str, table: Any, output: Output = Output.DAC1, *, port: int = UDP_PORT, timeout: float = 0.25,
               retries: int = 4) -> PlayResult:
    """`spt.dsp.siggen.Table`'ı karta gönderip çaldırır (tek çağrı)."""
    payload = payload_from_table(table)
    with SptgClient(host, port, timeout=timeout, retries=retries) as client:
        return client.play(payload, output)


# ============================================================================ komut satırı
def _print_info(info: BoardInfo) -> None:
    print(f"durum            : {info.state_text}")
    print(f"kaynak           : {info.source_text}")
    print(f"zamanlayıcı saati: {info.timer_clock_hz} Hz")
    print(f"tablo belleği    : {info.table_capacity_bytes} bayt (DATA başına {info.max_chunk_bytes} bayt)")
    print(f"DMA alt akışı    : {info.underrun_count}")
    if info.table_crc32:
        print(f"çalan tablo CRC  : 0x{info.table_crc32:08X}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m spt.protocol.sptg", description="Karta çalışma zamanında DAC "
                                     "tablosu gönderir (SPTG).")
    parser.add_argument("command", choices=("info", "play", "stop"))
    parser.add_argument("--host", default="192.168.172.98", help="kartın IP adresi")
    parser.add_argument("--port", type=int, default=UDP_PORT)
    parser.add_argument("--output", choices=("dac1", "dac2", "both"), default="dac1")
    parser.add_argument("--kind", default="sine", help="dalga türü (spt.dsp.siggen: sine, square, triangle …)")
    parser.add_argument("--freq", type=float, default=1000.0, help="frekans (Hz)")
    parser.add_argument("--amplitude", type=float, default=1.5, help="genlik (V)")
    parser.add_argument("--offset", type=float, default=1.65, help="ofset (V)")
    parser.add_argument("--fs", type=float, default=48_000.0, help="DAC güncelleme hızı (Hz)")
    parser.add_argument("--bits", type=int, choices=(8, 12), default=12)
    parser.add_argument("--max-len", type=int, default=4096, help="en uzun tablo (örnek)")
    args = parser.parse_args(argv)
    try:
        if args.command == "play":
            from spt.dsp import siggen as sg

            dac = sg.DacSpec(8, 3.3, word_bits=8) if args.bits == 8 else sg.DacSpec(12, 3.3)
            spec = sg.WaveSpec(kind=args.kind, freq=args.freq, amplitude=args.amplitude, offset=args.offset)
            table = sg.make_table(spec, dac, args.fs, max_len=args.max_len)
            result = send_table(args.host, table, Output[args.output.upper()], port=args.port)
            print(f"çalıyor: {sg.describe(spec)} → {OUTPUT_LABELS[result.output]}")
            print(f"fs = {result.fs:.9g} Hz ({result.fs_error_ppm:+.3f} ppm), f = {result.f:.9g} Hz, "
                  f"N = {result.sample_count}, PSC = {result.timer_prescaler}, ARR = {result.timer_autoreload}")
        else:
            with SptgClient(args.host, args.port) as client:
                _print_info(client.stop() if args.command == "stop" else client.info())
    except (SptgError, ValueError) as exc:
        print(f"hata: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
