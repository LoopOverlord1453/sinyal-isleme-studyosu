"""`.sptr` kayıt dosyası: `[u64 rx_time_ns][SPTF çerçevesi]` kayıtları. Bkz. protokol dokümanı §11."""
from __future__ import annotations

import mmap
import struct
import time
from pathlib import Path

import numpy as np

from spt.protocol.format import HEADER, MAGIC

FILE_MAGIC = b"SPTR"
FILE_VERSION = 1
FILE_HEADER = struct.Struct("<4sHHQ16s")  # 32 B
RECORD = struct.Struct("<Q")              # rx_time_ns
_FRAME_LEN = struct.Struct("<I")          # çerçeve başlığındaki frame_len (ofset 8)


class RecordingError(ValueError):
    pass


class SptrWriter:
    def __init__(self, path: str | Path, created_unix_ns: int | None = None, buffering: int = 8 << 20):
        self.path = Path(path)
        self._f = open(self.path, "wb", buffering=buffering)
        created = time.time_ns() if created_unix_ns is None else created_unix_ns
        self._f.write(FILE_HEADER.pack(FILE_MAGIC, FILE_VERSION, FILE_HEADER.size, created, b"\0" * 16))
        self.bytes_written = FILE_HEADER.size
        self.frames_written = 0

    def write(self, frame, rx_time_ns: int) -> None:
        self._f.write(RECORD.pack(rx_time_ns))
        self._f.write(frame)
        self.bytes_written += RECORD.size + len(frame)
        self.frames_written += 1

    def close(self) -> None:
        if not self._f.closed:
            self._f.close()

    def __enter__(self) -> SptrWriter:
        return self

    def __exit__(self, *_) -> None:
        self.close()


class SptrReader:
    """Kayıt dosyasını belleğe eşleyerek (mmap) okur; büyük dosyalar da hızlıca açılır."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._f = open(self.path, "rb")
        size = self.path.stat().st_size
        if size < FILE_HEADER.size:
            self._f.close()
            raise RecordingError("dosya .sptr başlığından kısa")
        self._mm = mmap.mmap(self._f.fileno(), 0, access=mmap.ACCESS_READ)
        self.buffer = memoryview(self._mm)
        magic, version, header_len, created, _ = FILE_HEADER.unpack_from(self.buffer, 0)
        if magic != FILE_MAGIC:
            self.close()
            raise RecordingError("SPTR dosyası değil (magic hatalı)")
        if version != FILE_VERSION:
            self.close()
            raise RecordingError(f"desteklenmeyen .sptr sürümü: {version}")
        self.created_unix_ns = created
        self.data_start = header_len
        self.size = size
        self.truncated = False
        self.skipped_bytes = 0
        self.resyncs = 0

    def index(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Tüm kayıtların (çerçeve ofseti, rx_time_ns, çerçeve uzunluğu) dizilerini döndürür.

        Dosya sonundaki yarım kalmış kayıt atlanır ve `truncated` işaretlenir.
        """
        offsets, times, lengths = [], [], []
        mv, pos, size = self.buffer, self.data_start, self.size
        unpack_rx, unpack_len = RECORD.unpack_from, _FRAME_LEN.unpack_from
        self.skipped_bytes = 0
        self.resyncs = 0
        while pos + RECORD.size + HEADER.size <= size:
            frame = pos + RECORD.size
            flen = unpack_len(mv, frame + 8)[0] if mv[frame:frame + 4] == MAGIC else 0
            if flen >= HEADER.size and frame + flen <= size:
                offsets.append(frame)
                times.append(unpack_rx(mv, pos)[0])
                lengths.append(flen)
                pos = frame + flen
                continue
            if mv[frame:frame + 4] == MAGIC and flen >= HEADER.size:
                break  # dosya sonunda yarım kalmış kayıt
            # Bozuk kayıt: bir sonraki geçerli kaydı ("SPTF" + makul uzunluk) ara, aradakini atla
            found = self._resync(frame + 1)
            if found is None:
                self.skipped_bytes += size - pos
                pos = size
                break
            self.skipped_bytes += found - pos
            self.resyncs += 1
            pos = found
        self.truncated = pos != size
        return (np.asarray(offsets, dtype=np.int64), np.asarray(times, dtype=np.int64),
                np.asarray(lengths, dtype=np.int64))

    def _resync(self, start: int) -> int | None:
        """`start`'tan sonra geçerli görünen ilk kaydın (rx_time + çerçeve) başlangıcı."""
        data = self._mm
        pos = start
        while True:
            hit = data.find(MAGIC, pos)
            if hit < 0:
                return None
            record = hit - RECORD.size
            if record >= self.data_start and hit + HEADER.size <= self.size:
                flen = _FRAME_LEN.unpack_from(self.buffer, hit + 8)[0]
                if HEADER.size <= flen and hit + flen <= self.size:
                    return record
            pos = hit + 1

    def __iter__(self):
        offsets, times, lengths = self.index()
        for off, rx, n in zip(offsets.tolist(), times.tolist(), lengths.tolist()):
            yield rx, self.buffer[off:off + n]

    def close(self) -> None:
        try:
            self.buffer.release()
            self._mm.close()
        except (BufferError, AttributeError):
            pass  # dışarıda hâlâ görünüm tutuluyorsa çöp toplayıcı kapatır
        self._f.close()

    def __enter__(self) -> SptrReader:
        return self

    def __exit__(self, *_) -> None:
        self.close()
