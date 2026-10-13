"""Bayt akışı taşımaları (USB sanal COM portu, CAN-bus) için SPTF çerçeve ayırıcı.

UDP'de her datagram çerçeve sınırını kendiliğinden verir; USB CDC ve CAN ise sınırı olmayan bir bayt akışı taşır.
Ayırıcı akışta "SPTF" imzasını arar, `frame_len` kadar bayt biriktirir ve CRC-32 ile doğrular. Doğrulanamayan
imzada bir bayt ilerleyip yeniden arar (senkron); kayıp ya da bozuk bayttan sonra sonraki sağlam çerçevede toparlanır.

Bayt akışında CRC zorunludur: CRC'siz bir "çerçevenin" gerçek çerçeve başı mı, yoksa örnek verisinin içinde
rastlantıyla geçen "SPTF" baytları mı olduğu ayırt edilemez. Çerçeveler arasındaki sıfır baytlar dolgudur (ör. CAN
FD'de son parçanın DLC'ye tamamlanması) ve hata sayılmaz. Bu modül Qt kullanmaz.
"""
from __future__ import annotations

import struct
import zlib
from collections.abc import Callable

from spt.protocol.format import CRC, FLAG_CRC32, HEADER, MAGIC, VERSION

MAX_STREAM_FRAME = 65535   # bayt akışında kabul edilen en büyük çerçeve (UDP'deki datagram sınırına denk)
_PEEK = struct.Struct("<BBHI").unpack_from   # imzadan sonra: version, frame_type, flags, frame_len
_MIN_FRAME = HEADER.size + CRC.size


class StreamDeframer:
    """Bir bayt akışını (tek gönderici) SPTF çerçevelerine ayırır. `feed` her çağrıda tamamlanan çerçeveleri
    döndürür; yarım kalan çerçeve sonraki çağrıya saklanır. Hatalar `on_error(ileti)` ile bildirilir."""

    def __init__(self, on_error: Callable[[str], None] | None = None, max_frame: int = MAX_STREAM_FRAME):
        self._buf = bytearray()
        self._on_error = on_error
        self.max_frame = max_frame
        self.synced = False        # son işlenen bayt geçerli bir çerçevenin sonu mu (ilk bağlantıda akışın ortası)
        self.frames = 0            # doğrulanan çerçeve
        self.skipped = 0           # çerçeve dışında atlanan bayt (sıfır dolgu hariç)
        self.resyncs = 0           # senkron kaybı olayı (bozuk çerçeve ya da çerçeve dışı veri)
        self.bytes = 0             # akıştan gelen toplam bayt

    def reset(self) -> None:
        """Bağlantı yeniden kuruldu: yarım çerçeve atılır, ilk imzaya kadar gelen veri sessizce atlanır."""
        self._buf.clear()
        self.synced = False

    def feed(self, data) -> list[bytes]:
        buf = self._buf
        buf += data
        self.bytes += len(data)
        out: list[bytes] = []
        pos, n = 0, len(buf)
        while n - pos >= len(MAGIC):
            if buf[pos:pos + 4] != MAGIC:
                found = buf.find(MAGIC, pos)
                end = found if found >= 0 else n - (len(MAGIC) - 1)   # yarım imza sonraki parçayla tamamlanabilir
                self._skip(buf, pos, end)
                pos = end
                if found < 0:
                    break
            if n - pos < HEADER.size:
                break
            version, _ftype, flags, flen = _PEEK(buf, pos + 4)
            if version != VERSION:
                problem = f"desteklenmeyen sürüm: {version}"
            elif not flags & FLAG_CRC32:
                problem = ("CRC'siz çerçeve: USB ve CAN-bus üzerinden CRC-32 zorunludur "
                           "(kartta akışın crc_enabled alanını açın)")
            elif not _MIN_FRAME <= flen <= self.max_frame:
                problem = f"frame_len geçersiz: {flen}"
            else:
                if n - pos < flen:
                    break   # çerçevenin geri kalanı bekleniyor
                frame = bytes(buf[pos:pos + flen])
                actual = zlib.crc32(memoryview(frame)[:-CRC.size])
                (expected,) = CRC.unpack_from(frame, flen - CRC.size)
                if actual == expected:
                    out.append(frame)
                    pos += flen
                    self.frames += 1
                    self.synced = True
                    continue
                problem = f"CRC hatası: hesaplanan 0x{actual:08X}, çerçevede 0x{expected:08X}"
            self._lose_sync(problem)
            pos += 1   # imzanın bir bayt ötesinden yeniden ara
        if pos:
            del buf[:pos]
        return out

    def _skip(self, buf: bytearray, start: int, end: int) -> None:
        if end <= start:
            return
        chunk = buf[start:end]
        zeros = chunk.count(0)
        if zeros == len(chunk):
            return   # çerçeveler arası sıfır dolgu
        self.skipped += len(chunk) - zeros
        if self.synced:
            self._lose_sync("çerçeve sınırı dışında veri (senkron kaybı); sonraki \"SPTF\" imzası aranıyor")

    def _lose_sync(self, problem: str) -> None:
        self.synced = False
        self.resyncs += 1
        if self._on_error is not None:
            self._on_error(problem)
