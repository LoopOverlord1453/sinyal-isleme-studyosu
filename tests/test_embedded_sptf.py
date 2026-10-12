"""Gömülü SPTF-UDP C kütüphanesi (embedded/sptf_udp): derleme, birim testleri ve arayüzle çapraz doğrulama.

- C birim testleri gcc / clang ile, sıkı uyarılar `-Werror` ile; küçük uçlu, büyük uçlu (bayt çevirme yolu), bit bit
  CRC ve CRC'siz yapılandırmalarda çalışır.
- `sptf_dump` her örnek tipi × yerleşim × CRC için çerçeveleri üretir; Python referans kodlayıcısı
  (`spt.protocol.builder`) aynı senaryodan bayt bayt aynı datagramları üretmelidir (bölme, seq, first_sample,
  zaman damgası, DESCRIPTOR tekrarı, MARKER dahil). Datagramlar arayüzün gerçek alıcısına (`StreamStore`) verilir.
- Uçtan uca: C göndericisi (Winsock / POSIX soket portu) gerçek UDP ile gönderir; alıcıda tüm örnekler tam değerle,
  kayıpsız ve hatasız bulunur.
"""
from __future__ import annotations

import os
import socket
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest

from spt.codegen import toolchain
from spt.core.ring import SharedRing
from spt.core.stream_store import StreamStore
from spt.protocol.builder import build_data, build_descriptor, build_marker
from spt.protocol.format import SAMPLE_TYPES, ChannelInfo, Descriptor, Layout

ROOT = Path(__file__).resolve().parents[1] / "embedded" / "sptf_udp"
CORE = ["src/sptf_frame.c", "src/sptf_link.c"]
STRICT = ["-std=c99", "-Wall", "-Wextra", "-Wpedantic", "-Werror", "-Wconversion", "-Wsign-conversion", "-Wshadow",
          "-Wcast-qual", "-Wcast-align", "-Wstrict-prototypes", "-Wmissing-prototypes", "-Wundef",
          "-Wdouble-promotion"]


def _compile(tool, out: Path, sources: list[str], extra: list[str] = (), libs: list[str] = ()) -> Path:
    cmd = [tool.path, *STRICT, "-O2", "-Iinclude", "-Isrc", "-Iport/socket", *extra, *sources, *libs, "-o", str(out)]
    r = toolchain._run(cmd, cwd=ROOT, path_prefix=[tool.bin_dir])
    assert r.returncode == 0, f"derleme hatası ({' '.join(extra)}):\n{r.stdout}\n{r.stderr}"
    return out


@pytest.fixture(scope="module")
def cc():
    tool = toolchain.find_host_cc()
    if tool is None:
        pytest.skip("ana makine C derleyicisi yok")
    return tool


@pytest.fixture(scope="module")
def bins(cc, tmp_path_factory):
    d = tmp_path_factory.mktemp("sptf")
    net = ["-lws2_32"] if os.name == "nt" else []
    return {
        "dump": _compile(cc, d / "sptf_dump.exe", CORE + ["test/sptf_dump.c"]),
        "demo": _compile(cc, d / "sptf_host_demo.exe",
                         CORE + ["port/socket/sptf_port_socket.c", "examples/host/sptf_host_demo.c"], libs=net),
        "bin_dir": cc.bin_dir,
    }


def _run(exe: Path, *args, bin_dir=None, timeout=60) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    if bin_dir:
        env["PATH"] = str(bin_dir) + os.pathsep + env.get("PATH", "")
    return subprocess.run([str(exe), *map(str, args)], capture_output=True, timeout=timeout, env=env)


# ------------------------------------------------------------------------------------------- C birim testleri
@pytest.mark.parametrize("variant", [[], ["-DSPTF_CFG_HOST_BIG_ENDIAN=1"], ["-DSPTF_CFG_CRC_TABLE=0", "-O0"],
                                     ["-DSPTF_CFG_CRC=0", "-Os"]],
                         ids=["kucuk-uclu", "buyuk-uclu", "bit-crc", "crc-yok"])
def test_c_unit_tests(cc, tmp_path, variant):
    exe = _compile(cc, tmp_path / "t.exe", CORE + ["test/sptf_test.c"], variant)
    r = _run(exe, bin_dir=cc.bin_dir)
    out = r.stdout.decode("utf-8", "replace")
    assert r.returncode == 0, out
    assert "GECTI" in out and ", 0 hata" in out


def test_clang_also_builds_warning_free(tmp_path):
    clang = Path("C:/mingw64/bin/clang.exe")
    if not clang.exists():
        pytest.skip("clang yok")
    tool = toolchain.Tool("host-cc", str(clang), "clang", "x86_64-w64-mingw32")
    exe = _compile(tool, tmp_path / "t.exe", CORE + ["test/sptf_test.c"])
    assert _run(exe, bin_dir=clang.parent).returncode == 0


# ------------------------------------------------------------------------- Python referansıyla bayt bayt aynı
def _datagrams(path: Path) -> list[bytes]:
    raw = path.read_bytes()
    out, off = [], 0
    while off < len(raw):
        n = int.from_bytes(raw[off:off + 4], "little")
        out.append(raw[off + 4:off + 4 + n])
        off += 4 + n
    return out


def _expected(code, layout, crc, ch, n, blocks, maxdg, fs, ts0, seed) -> tuple[Descriptor, list[bytes], int]:
    """C kütüphanesinin kurallarının bağımsız Python ifadesi (bölme, seq, DESCRIPTOR periyodu, zaman damgası)."""
    st = SAMPLE_TYPES[code]
    desc = Descriptor(7, 1, fs, st, Layout(layout), "dump akisi",
                      tuple(ChannelInfo(f"k{i}", "V" if i % 2 == 0 else "°C", 1.0 + 0.5 * i, -float(i))
                            for i in range(ch)))
    per = (maxdg - 48 - (4 if crc else 0)) // (st.size * ch)
    period_f = (fs * 500.0) / 1000.0
    period = int(period_f) if period_f >= 1.0 else 1
    ns_per = 1.0e9 / fs
    frames: list[bytes] = []
    state = {"seq": 0, "next": 0, "pending": True, "due": 0}

    def maybe_desc():
        if state["pending"] or state["next"] >= state["due"]:
            frames.append(build_descriptor(desc, state["seq"], 0, crc))
            state["seq"] += 1
            state["pending"] = False
            state["due"] = state["next"] + period

    nbytes = n * st.size * ch
    for b in range(blocks):
        raw = ((np.arange(nbytes, dtype=np.uint64) * 131 + seed + 17 * b) & 0xFF).astype(np.uint8).tobytes()
        arr = np.frombuffer(raw, st.dtype)
        arr = arr.reshape(n, ch) if layout == Layout.INTERLEAVED else arr.reshape(ch, n)
        base = ts0 + b * 1000 if ts0 else 0
        done = 0
        while done < n:
            k = min(n - done, per)
            maybe_desc()
            part = arr[done:done + k] if layout == Layout.INTERLEAVED else arr[:, done:done + k].T
            ts = base + int(done * ns_per + 0.5) if base else 0
            frames.append(build_data(desc, state["seq"], ts, state["next"], part, crc))
            state["seq"] += 1
            state["next"] += k
            done += k
        if b == 0:
            maybe_desc()
            frames.append(build_marker(7, 1, state["seq"], 777, state["next"], 0xC0DE, "işaret ü", crc))
            state["seq"] += 1
    return desc, frames, per


def _scenarios():
    out = []
    for i, code in enumerate(sorted(SAMPLE_TYPES)):
        for layout in (0, 1):
            for crc in (False, True):
                j = i + layout + 2 * crc
                ch = (1, 3, 5)[j % 3]
                maxdg = (1472, 600, 8972)[(j // 3) % 3]
                fs = (1000.0, 48000.0, 1.0e6)[i % 3]
                out.append(pytest.param(code, layout, crc, ch, maxdg, fs, 123456789 if j % 2 else 0,
                                        id=f"{SAMPLE_TYPES[code].name}-{'planar' if layout else 'interleaved'}"
                                           f"-{'crc' if crc else 'nocrc'}"))
    return out


@pytest.mark.parametrize("code,layout,crc,ch,maxdg,fs,ts0", _scenarios())
def test_c_frames_match_python_builder_and_receiver(bins, tmp_path, code, layout, crc, ch, maxdg, fs, ts0):
    st = SAMPLE_TYPES[code]
    per = (maxdg - 48 - (4 if crc else 0)) // (st.size * ch)
    n = 2 * per + 13                     # her blok 3 çerçeveye bölünür (son çerçeve kısmi)
    blocks, seed = 3, 11 + code
    out = tmp_path / "dg.bin"
    r = _run(bins["dump"], out, code, layout, int(crc), ch, n, blocks, maxdg, repr(fs), ts0, seed,
             bin_dir=bins["bin_dir"])
    assert r.returncode == 0, r.stdout + r.stderr
    got = _datagrams(out)
    desc, expected, per_py = _expected(code, layout, crc, ch, n, blocks, maxdg, fs, ts0, seed)
    assert f"ok {len(expected)} {per_py}" in r.stdout.decode()
    assert len(got) == len(expected)
    for i, (g, e) in enumerate(zip(got, expected)):
        assert g == e, f"{i}. datagram farklı (tip {g[5]}, {len(g)} / {len(e)} bayt)"
        assert len(g) <= maxdg

    events = []
    store = StreamStore(events.append, session_start_ns=0, ring_seconds=1.0, max_ring_bytes=1 << 20)
    try:
        for i, g in enumerate(got):
            store.handle_datagram(memoryview(g), len(g), "10.0.0.9", i)
        assert store.frame_errors == 0 and store.no_descriptor == 0
        (s,) = store.stats(None)
        assert (s.samples, s.lost_samples, s.duplicates, s.restarts) == (blocks * n, 0, 0, 0)
        markers = [e for e in events if e[0] == "marker"]
        assert [(m[2], m[3], m[4], m[5]) for m in markers] == [(n, 0xC0DE, "işaret ü", n)]
        info = next(e[1] for e in events if e[0] == "stream")
        assert info.descriptor == desc
    finally:
        store.close()


# ------------------------------------------------------------------------------------------ uçtan uca UDP
def _receive_from(proc: subprocess.Popen, sock: socket.socket, store: StreamStore) -> int:
    count = 0
    quiet_until = None
    buf = bytearray(65536)
    mv = memoryview(buf)
    while True:
        try:
            n, addr = sock.recvfrom_into(buf)
        except socket.timeout:
            if proc.poll() is not None:
                if quiet_until is None:
                    quiet_until = time.monotonic() + 0.5
                elif time.monotonic() > quiet_until:
                    return count
            continue
        store.handle_datagram(mv, n, addr[0], time.time_ns())
        count += 1


@pytest.fixture
def udp():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 16 << 20)
    sock.bind(("127.0.0.1", 0))
    sock.settimeout(0.05)
    yield sock
    sock.close()


def test_udp_end_to_end_ramp_is_lossless_and_exact(bins, udp):
    port = udp.getsockname()[1]
    rate, seconds = 50000.0, 1.0
    env = dict(os.environ, PATH=str(bins["bin_dir"]) + os.pathsep + os.environ.get("PATH", ""))
    proc = subprocess.Popen([str(bins["demo"]), "127.0.0.1", str(port), str(seconds), "ramp", str(rate)],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
    events = []
    store = StreamStore(events.append, session_start_ns=time.time_ns(), ring_seconds=5.0)
    try:
        datagrams = _receive_from(proc, udp, store)
        out = proc.communicate(timeout=10)[0].decode("utf-8", "replace")
        assert proc.returncode == 0, out
        assert store.frame_errors == 0 and store.no_descriptor == 0
        (s,) = store.stats(None)
        total = int(rate * seconds)
        assert (s.samples, s.lost_samples, s.duplicates) == (total, 0, 0), out
        assert s.name == "Rampa" and s.format.startswith("4 × int32")
        assert datagrams >= total // 88
        info = next(e[1] for e in events if e[0] == "stream")
        ring = SharedRing.attach(info.shm_name, np.dtype("<i4"))
        try:
            first, data = ring.read(0, ring.write_count)
            idx = np.arange(first, first + len(data), dtype=np.int64)[:, None] * 4 + np.arange(4)
            assert first == 0 and len(data) == total
            assert np.array_equal(data, idx.astype(np.int32))
        finally:
            ring.close()
        assert [(m[2], m[3], m[4], m[5]) for m in events if m[0] == "marker"] == [(1000, 7, "rampa", 1000)]
    finally:
        if proc.poll() is None:
            proc.kill()
        store.close()


def test_udp_demo_mode_streams_and_markers(bins, udp):
    port = udp.getsockname()[1]
    env = dict(os.environ, PATH=str(bins["bin_dir"]) + os.pathsep + os.environ.get("PATH", ""))
    proc = subprocess.Popen([str(bins["demo"]), "127.0.0.1", str(port), "2.2", "demo"],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
    events = []
    store = StreamStore(events.append, session_start_ns=time.time_ns(), ring_seconds=5.0)
    try:
        _receive_from(proc, udp, store)
        proc.communicate(timeout=10)
        assert proc.returncode == 0
        assert store.frame_errors == 0
        stats = {s.name: s for s in store.stats(None)}
        assert set(stats) == {"Motor akımları", "IMU"}
        assert stats["Motor akımları"].format.startswith("3 × int16")
        assert stats["IMU"].format.startswith("6 × float32")
        assert all(s.lost_samples == 0 and s.samples > 0 for s in stats.values())
        texts = [m[4] for m in events if m[0] == "marker"]
        assert texts[:2] == ["durum = RUN", "durum = IDLE"]
    finally:
        if proc.poll() is None:
            proc.kill()
        store.close()
