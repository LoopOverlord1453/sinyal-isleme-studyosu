"""Cihaz simülatörü: SPTF formatında UDP üzerinden sinyal gönderir.

Örnekler:
    python -m spt.sources.simulator --signal sine --freq 50 --fs 10000 --channels 2 --type int16
    python -m spt.sources.simulator --file samples_signal_data/01_sinus_50Hz_harmonik_2kanal_int16_10kHz.sptr --loop
    python -m spt.sources.simulator --file kayit.wav --loop
    python -m spt.sources.simulator --fs 14.4e6 --channels 4 --type int16 --datagram 8972   (≈115 MB/s)

Bu modül Qt import etmez.
"""
from __future__ import annotations

import argparse
import csv
import socket
import struct
import sys
import time
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import signal as sps
from scipy.io import wavfile

from spt.console import utf8_stdio
from spt.dsp import waveforms as wf
from spt.protocol.builder import DataFrameWriter, build_descriptor, build_marker, to_wire
from spt.protocol.format import (
    CRC,
    FLAG_CRC32,
    HEADER,
    MAX_DATAGRAM,
    SAMPLE_TYPES_BY_NAME,
    STANDARD_DATAGRAM,
    UDP_PORT,
    ChannelInfo,
    Descriptor,
    FrameType,
    Layout,
    SampleType,
    samples_per_frame,
)
from spt.protocol.recording import SptrReader

SIGNALS = ("sine", "square", "triangle", "sawtooth", "chirp", "noise", "mix")
MAX_TABLE = 1 << 20
DESCRIPTOR_INTERVAL = 1.0  # s


# ---------------------------------------------------------------- sinyal tablosu
@dataclass
class Table:
    desc: Descriptor
    wire: np.ndarray        # (L, kanal) kablo tipinde, bitişik
    loop: bool
    note: str = ""


def scale_for(stype: SampleType, amplitude: float) -> tuple[float, float]:
    """Fiziksel ±1.25·genlik aralığını tamsayı tipin tam ölçeğine oturtan (scale, offset)."""
    if stype.is_float:
        return 1.0, 0.0
    full = (1 << (stype.bits - 1)) - 1
    scale = amplitude * 1.25 / full
    offset = 0.0 if stype.signed else -(full + 1) * scale
    return scale, offset


def physical_to_wire(phys: np.ndarray, stype: SampleType, scale: float, offset: float) -> np.ndarray:
    if stype.is_float:
        raw = phys
    elif stype.is_complex:
        raw = phys / scale
    else:
        raw = (phys - offset) / scale
    return np.ascontiguousarray(to_wire(raw, stype))


def generate_table(kind: str, freq: float, amplitude: float, fs: float, channels: int, stype: SampleType,
                   layout: Layout, spf: int, name: str, unit: str, stream_id: int, seed: int = 1) -> Table:
    length = max(spf, (min(int(fs), MAX_TABLE) // spf) * spf)
    cycles = max(1, round(freq * length / fs))
    base = cycles * fs / length
    n = np.arange(length)
    t = n / fs
    rng = np.random.default_rng(seed)
    scale, offset = scale_for(stype, amplitude)
    cols, infos = [], []
    for k in range(channels):
        cycles_k = cycles * (k + 1)
        if base * (k + 1) > fs / 2.2:
            cycles_k = cycles
        f = cycles_k * fs / length
        w = 2 * np.pi * f * t
        ph = wf.cycle_phase(n, cycles_k, length)  # kare/üçgen/testere için tam faz (geçişte hatalı örnek yok)
        if stype.is_complex:
            x = np.exp(1j * w) if kind != "noise" else (rng.standard_normal(length) + 1j * rng.standard_normal(length)) / 3
        elif kind == "sine":
            x = np.sin(w)
        elif kind == "square":
            x = wf.square(ph)
        elif kind == "triangle":
            x = wf.triangle(ph)
        elif kind == "sawtooth":
            x = wf.sawtooth(ph)
        elif kind == "chirp":
            x = sps.chirp(t, f0=f / 10, t1=t[-1], f1=f)
        elif kind == "noise":
            x = np.clip(rng.standard_normal(length) / 3, -1, 1)
        else:  # mix
            x = 0.6 * np.sin(w) + 0.25 * np.sin(3.7 * w) + 0.1 * rng.standard_normal(length)
        cols.append(amplitude * x)
        infos.append(ChannelInfo(f"ch{k}", unit, scale, offset))
    phys = np.stack(cols, axis=1)
    desc = Descriptor(stream_id, 1, fs, stype, layout, name, tuple(infos))
    note = f"temel frekans {base:g} Hz (tablo {length} örnek)"
    return Table(desc, physical_to_wire(phys, stype, scale, offset), loop=True, note=note)


# ---------------------------------------------------------------- dosyadan tablo
_NUMPY_TO_SPTF = {
    "int8": "int8", "uint8": "uint8", "int16": "int16", "uint16": "uint16", "int32": "int32",
    "uint32": "uint32", "int64": "int64", "uint64": "uint64", "float16": "float16", "float32": "float32",
    "float64": "float64", "complex64": "cfloat32", "complex128": "cfloat64",
}


def _table_from_array(data: np.ndarray, fs: float, name: str, unit: str, stream_id: int, loop: bool,
                      layout: Layout, full_scale: bool) -> Table:
    if data.ndim == 1:
        data = data[:, None]
    if data.ndim != 2:
        raise ValueError(f"dizi 1 ya da 2 boyutlu olmalı (örnek × kanal); verilen biçim {data.shape}")
    if data.shape[0] < data.shape[1] and data.shape[0] <= 256:
        data = data.T  # (kanal, N) verilmiş
    stype = SAMPLE_TYPES_BY_NAME.get(_NUMPY_TO_SPTF.get(data.dtype.name, ""), SAMPLE_TYPES_BY_NAME["float64"])
    if stype.is_float or stype.is_complex:
        scale, offset = 1.0, 0.0
    elif full_scale:  # WAV: tam ölçek = ±1
        half = 1 << (stype.bits - 1)
        scale, offset = 1.0 / half, (-1.0 if not stype.signed else 0.0)
    else:
        scale, offset = 1.0, 0.0
    infos = tuple(ChannelInfo(f"ch{k}", unit, scale, offset) for k in range(data.shape[1]))
    desc = Descriptor(stream_id, 1, fs, stype, layout, name, infos)
    return Table(desc, np.ascontiguousarray(to_wire(data, stype)), loop)


def _read_csv(path: Path) -> np.ndarray:
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    sample = text[:4096]
    try:
        delimiter = csv.Sniffer().sniff(sample, delimiters=",;\t ").delimiter
    except csv.Error:
        delimiter = ";" if sample.count(";") > sample.count(",") else ","
    if delimiter == ";":  # Türkçe / Avrupa Excel: ondalık ayırıcı virgül
        text = text.replace(",", ".")
    lines = text.splitlines()
    for skip in (0, 1):
        try:
            return np.loadtxt(lines[skip:], delimiter=delimiter, ndmin=2, dtype=np.float64)
        except ValueError:
            continue
    raise ValueError("CSV sayısal sütunlar içermiyor (ayraç: , ; sekme ya da boşluk)")


def table_from_file(path: Path, fs: float | None, stream_id: int, loop: bool, layout: Layout) -> Table:
    suffix = path.suffix.lower()
    if suffix == ".wav":
        rate, data = wavfile.read(path)
        return _table_from_array(data, float(rate), path.stem, "FS", stream_id, loop, layout, full_scale=True)
    if suffix == ".npy":
        data = np.load(path, allow_pickle=False)
    elif suffix in (".csv", ".txt"):
        data = _read_csv(path)
    else:
        raise ValueError(f"desteklenmeyen dosya türü: {suffix}")
    if not fs:
        raise ValueError(f"{suffix} dosyası için --fs ile örnekleme hızı verilmeli")
    return _table_from_array(data, fs, path.stem, "", stream_id, loop, layout, full_scale=False)


# ---------------------------------------------------------------- gönderici
class Sender:
    def __init__(self, host: str, port: int, drop: float, seed: int = 2):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 8 << 20)
        self.addr = (host, port)
        self.drop = drop
        self.rng = np.random.default_rng(seed)
        self.frames = self.bytes = self.samples = self.dropped_frames = self.dropped_samples = 0
        self._t0 = self._last_t = time.perf_counter()
        self._last_bytes = self._last_frames = 0

    def send(self, frame, samples: int = 0, droppable: bool = True) -> None:
        if self.drop and samples and droppable and self.rng.random() < self.drop:
            self.dropped_frames += 1
            self.dropped_samples += samples
            return
        self.sock.sendto(frame, self.addr)
        self.frames += 1
        self.bytes += len(frame)
        self.samples += samples

    def report(self, force: bool = False) -> None:
        now = time.perf_counter()
        if not force and now - self._last_t < 1.0:
            return
        dt = now - self._last_t
        mbps = (self.bytes - self._last_bytes) / dt / 1e6
        pps = (self.frames - self._last_frames) / dt
        print(f"{now - self._t0:6.1f} s · {mbps:8.2f} MB/s · {pps:9.0f} paket/s · "
              f"yapay kayıp {self.dropped_samples} örnek", flush=True)
        self._last_t, self._last_bytes, self._last_frames = now, self.bytes, self.frames

    def summary(self) -> None:
        elapsed = time.perf_counter() - self._t0
        print(f"BİTTİ · {self.frames} çerçeve · {self.samples} örnek · {self.bytes / 1e6:.1f} MB · ortalama "
              f"{self.bytes / max(elapsed, 1e-9) / 1e6:.2f} MB/s · yapay kayıp {self.dropped_samples} örnek "
              f"({self.dropped_frames} çerçeve)", flush=True)


def stream_table(sender: Sender, table: Table, datagram: int, crc: bool, marker_every: float,
                 duration: float, max_rate: bool) -> None:
    desc = table.desc
    spf = samples_per_frame(desc, datagram, crc)
    writer = DataFrameWriter(desc, spf, crc)
    wire = table.wire
    length = wire.shape[0]
    flat = memoryview(np.ascontiguousarray(wire).view(np.uint8).reshape(-1))
    bps = desc.bytes_per_sample
    planar = desc.layout == Layout.PLANAR
    fs = desc.sample_rate
    seq = sent = 0
    next_desc = next_marker = 0
    marker_step = int(marker_every * fs) if marker_every > 0 else 0
    marker_no = 0
    t_start = time.perf_counter()
    end_samples = int(duration * fs) if duration > 0 else None

    wire_planar = np.ascontiguousarray(wire.T) if planar else None

    def frame_bytes(i: int, k: int):
        if i + k <= length:
            return flat[i * bps:(i + k) * bps] if not planar else wire_planar[:, i:i + k].tobytes()
        idx = (np.arange(i, i + k) % length)
        block = wire[idx]
        return (block.T if planar else block).tobytes()

    while True:
        if max_rate:
            due = sent + spf * 64
        else:
            due = int((time.perf_counter() - t_start) * fs)
        if end_samples is not None:
            due = min(due, end_samples)
        # süre sonunda kalan (< 1 çerçeve) örnekler kısa bir son çerçeveyle gönderilir: --duration tam karşılanır
        while due - sent >= spf or (not table.loop and sent < length and due >= length) or \
                (end_samples is not None and sent < due == end_samples):
            if sent >= next_desc:
                sender.send(build_descriptor(desc, seq, int(sent / fs * 1e9), crc))
                seq += 1
                next_desc = sent + int(DESCRIPTOR_INTERVAL * fs)
            if marker_step and sent >= next_marker:
                marker_no += 1
                sender.send(build_marker(desc.stream_id, desc.desc_rev, seq, int(sent / fs * 1e9), sent,
                                         marker_no, f"işaret {marker_no}", crc))
                seq += 1
                next_marker = sent + marker_step
            i = sent % length if table.loop else sent
            k = spf if table.loop else min(spf, length - sent)
            if end_samples is not None:
                k = min(k, end_samples - sent)
            frame = writer.write(seq, int(sent / fs * 1e9), sent, k, frame_bytes(i, k))
            # İlk ve son çerçeve yapay kayba uğramaz: alıcı, ilk çerçeveden önceki ve son çerçeveden
            # sonraki kaybı protokol gereği göremez; böylece iki taraftaki kayıp sayısı birebir karşılaştırılabilir.
            last = (end_samples is not None and sent + k >= end_samples) or \
                (not table.loop and sent + k >= length)
            sender.send(frame, k, droppable=sent > 0 and not last)
            seq += 1
            sent += k
            if not table.loop and sent >= length:
                return
        sender.report()
        if end_samples is not None and sent >= end_samples:
            return
        if not max_rate:
            time.sleep(0.001)


def stream_sptr(sender: Sender, path: Path, speed: float, loop: bool, duration: float) -> None:
    """Kaydı, kaydedildiği zamanlamayla gönderir. Döngüde örnek indeksleri kaydırılarak devam eder."""
    reader = SptrReader(path)
    offsets, rx, lengths = reader.index()
    if len(offsets) == 0:
        raise ValueError("kayıt boş")
    mv = reader.buffer
    spans: dict[int, tuple[int, int]] = {}
    for off in offsets.tolist():
        if mv[off + 5] == FrameType.DATA:
            sid = struct.unpack_from("<H", mv, off + 12)[0]
            first, count = struct.unpack_from("<QI", mv, off + HEADER.size)
            lo, hi = spans.get(sid, (first, first + count))
            spans[sid] = (min(lo, first), max(hi, first + count))
    rx0 = int(rx[0])
    t_start = time.perf_counter()
    loop_no = 0
    while True:
        for off, t, n in zip(offsets.tolist(), rx.tolist(), lengths.tolist()):
            wall = (t - rx0) / 1e9 / speed + loop_no * ((int(rx[-1]) - rx0) / 1e9 / speed + 0.01)
            if duration and wall > duration:
                return
            delay = wall - (time.perf_counter() - t_start)
            if delay > 0.001:
                time.sleep(delay)
            frame = mv[off:off + n]
            ftype = frame[5]
            if loop_no and ftype in (FrameType.DATA, FrameType.MARKER):
                sid = struct.unpack_from("<H", frame, 12)[0]
                lo, hi = spans.get(sid, (0, 0))
                patched = bytearray(frame)
                value = struct.unpack_from("<Q", patched, HEADER.size)[0] + loop_no * (hi - lo)
                struct.pack_into("<Q", patched, HEADER.size, value)
                if patched[6] & FLAG_CRC32:
                    CRC.pack_into(patched, n - CRC.size, zlib.crc32(patched[:n - CRC.size]))
                frame = patched
            samples = struct.unpack_from("<I", frame, HEADER.size + 8)[0] if ftype == FrameType.DATA else 0
            sender.send(frame, samples)
            sender.report()
        if not loop:
            return
        loop_no += 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m spt.sources.simulator",
                                description="SPTF cihaz simülatörü — UDP üzerinden sinyal gönderir.")
    p.add_argument("--host", default="127.0.0.1", help="hedef IP (varsayılan 127.0.0.1)")
    p.add_argument("--port", type=int, default=UDP_PORT, help=f"hedef UDP portu (varsayılan {UDP_PORT})")
    p.add_argument("--signal", choices=SIGNALS, default="mix", help="üretilecek sinyal")
    p.add_argument("--freq", type=float, default=50.0, help="temel frekans, Hz")
    p.add_argument("--amp", type=float, default=1.0, help="genlik (fiziksel birim)")
    p.add_argument("--unit", default="V", help="kanal birimi")
    p.add_argument("--fs", type=float, default=None, help="örnekleme hızı, Hz (üreteç için varsayılan 10000)")
    p.add_argument("--channels", type=int, default=2, help="kanal sayısı")
    p.add_argument("--type", default="int16", choices=sorted(SAMPLE_TYPES_BY_NAME), help="örnek veri tipi")
    p.add_argument("--layout", choices=("interleaved", "planar"), default="interleaved")
    p.add_argument("--datagram", type=int, default=STANDARD_DATAGRAM,
                   help="datagram boyutu, bayt (1472 standart, 8972 jumbo, 65507 en büyük)")
    p.add_argument("--crc", action="store_true", help="çerçevelere CRC32 ekle")
    p.add_argument("--drop", type=float, default=0.0, help="yapay paket kaybı oranı (0.01 = %%1)")
    p.add_argument("--marker-every", type=float, default=0.0, help="MARKER aralığı, s (0 = kapalı)")
    p.add_argument("--duration", type=float, default=0.0, help="süre, s (0 = durdurulana kadar)")
    p.add_argument("--file", type=Path, help=".sptr, .wav, .csv, .txt veya .npy dosyasından gönder")
    p.add_argument("--loop", action="store_true", help="dosya bitince başa dön")
    p.add_argument("--speed", type=float, default=1.0, help=".sptr oynatma hızı çarpanı")
    p.add_argument("--stream-id", type=int, default=1)
    p.add_argument("--name", default="simulator")
    p.add_argument("--max-rate", action="store_true", help="gerçek zamanı bekleme, olabildiğince hızlı gönder")
    return p


def main(argv: list[str] | None = None) -> int:
    utf8_stdio()
    args = build_parser().parse_args(argv)
    if not 64 <= args.datagram <= MAX_DATAGRAM:
        print(f"Hata: --datagram 64 ile {MAX_DATAGRAM} arasında olmalı", file=sys.stderr)
        return 2
    if not args.speed > 0:
        print("Hata: --speed pozitif olmalı (ör. 1, 2, 0.5)", file=sys.stderr)
        return 2
    sender = Sender(args.host, args.port, args.drop)
    layout = Layout.PLANAR if args.layout == "planar" else Layout.INTERLEAVED
    try:
        if args.file and args.file.suffix.lower() == ".sptr":
            print(f"Kayıt gönderiliyor: {args.file.name} → {args.host}:{args.port}", flush=True)
            stream_sptr(sender, args.file, args.speed, args.loop, args.duration)
        else:
            stype = SAMPLE_TYPES_BY_NAME[args.type]
            if args.file:
                table = table_from_file(args.file, args.fs, args.stream_id, args.loop, layout)
            else:
                fs = args.fs or 10_000.0
                probe = Descriptor(args.stream_id, 1, fs, stype, layout, args.name,
                                   tuple(ChannelInfo(f"ch{k}") for k in range(args.channels)))
                spf = samples_per_frame(probe, args.datagram, args.crc)
                table = generate_table(args.signal, args.freq, args.amp, fs, args.channels, stype, layout,
                                       spf, args.name, args.unit, args.stream_id)
            d = table.desc
            rate = d.sample_rate * d.bytes_per_sample / 1e6
            print(f"Gönderiliyor: {d.name} · {d.format_text()} · {rate:.2f} MB/s hedef → "
                  f"{args.host}:{args.port} {table.note}", flush=True)
            stream_table(sender, table, args.datagram, args.crc, args.marker_every, args.duration, args.max_rate)
    except KeyboardInterrupt:
        pass
    except (OSError, ValueError) as exc:
        print(f"Hata: {exc}", file=sys.stderr, flush=True)
        return 1
    finally:
        sender.report(force=True)
        sender.summary()
    return 0


if __name__ == "__main__":
    sys.exit(main())
