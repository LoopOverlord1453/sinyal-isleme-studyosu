import numpy as np
import pytest

from spt.core.ring import SharedRing
from spt.core.stream_store import StreamInfo, StreamStore
from spt.protocol.builder import build_data, build_descriptor
from spt.protocol.format import SAMPLE_TYPES_BY_NAME, ChannelInfo, Descriptor, Layout
from spt.protocol.recording import SptrReader, SptrWriter

SPF = 10


def _desc(rev=1, fs=1000.0):
    return Descriptor(7, rev, fs, SAMPLE_TYPES_BY_NAME["int16"], Layout.INTERLEAVED, "test",
                      (ChannelInfo("a", "V"),))


class Harness:
    def __init__(self):
        self.events = []
        self.store = StreamStore(self.events.append, session_start_ns=0, ring_seconds=100.0)
        self.seq = 0

    def feed(self, frame: bytes, rx_ns: int = 0):
        mv = memoryview(frame)
        self.store.handle_datagram(mv, len(frame), "10.0.0.2", rx_ns)
        self.seq += 1

    def data(self, first: int, desc=None, n=SPF):
        desc = desc or _desc()
        values = np.arange(first, first + n, dtype="<i2")[:, None]
        self.feed(build_data(desc, self.seq, 0, first, values))

    def ring(self) -> SharedRing:
        info = next(e[1] for e in reversed(self.events) if e[0] == "stream")
        return SharedRing.attach(info.shm_name, np.dtype("<i2"))

    def stats(self):
        (st,) = self.store.stats(0)
        return st


@pytest.fixture
def h():
    harness = Harness()
    yield harness
    harness.store.close()


def test_data_before_descriptor_is_dropped(h):
    h.data(0)
    assert h.store.no_descriptor == 1
    assert not [e for e in h.events if e[0] == "stream"]


def test_exact_loss_count_with_random_drops(h):
    rng = np.random.default_rng(1)
    h.feed(build_descriptor(_desc()))
    dropped = 0
    for i in range(500):
        if 0 < i < 499 and rng.random() < 0.05:
            dropped += SPF
            continue
        h.data(i * SPF)
    st = h.stats()
    assert st.lost_samples == dropped
    assert st.samples == 500 * SPF - dropped
    ring = h.ring()
    try:
        assert ring.write_count == 500 * SPF  # boşluklar dahil
        assert sum(e - s for s, e in ring.gaps_between(0, ring.write_count)) == dropped
    finally:
        ring.close()


def test_duplicate_and_restart(h):
    h.feed(build_descriptor(_desc()))
    h.data(0)
    h.data(10)
    h.data(10)       # yinelenen
    h.data(0)        # cihaz reset → yeniden başlama
    h.data(10)
    st = h.stats()
    assert (st.duplicates, st.restarts, st.lost_samples) == (1, 1, 0)


def test_descriptor_change_creates_new_stream(h):
    h.feed(build_descriptor(_desc()))
    h.data(0)
    h.feed(build_descriptor(_desc()))   # periyodik tekrar: değişiklik yok
    assert [e[0] for e in h.events].count("stream_removed") == 0
    h.feed(build_descriptor(_desc(rev=2, fs=2000.0)))
    assert [e[0] for e in h.events].count("stream_removed") == 1
    h.data(0, desc=_desc(rev=2, fs=2000.0))
    infos = [e[1] for e in h.events if e[0] == "stream"]
    assert len(infos) == 2 and isinstance(infos[-1], StreamInfo)
    assert infos[-1].descriptor.sample_rate == 2000.0


def test_recording_roundtrip(tmp_path):
    frames = [build_descriptor(_desc())] + [
        build_data(_desc(), i, 0, i * SPF, np.full((SPF, 1), i, dtype="<i2")) for i in range(20)]
    path = tmp_path / "k.sptr"
    with SptrWriter(path) as w:
        for i, f in enumerate(frames):
            w.write(f, 1_000 + i)
    with SptrReader(path) as r:
        records = [(rx, bytes(frame)) for rx, frame in r]
        assert not r.truncated
    assert records == [(1_000 + i, f) for i, f in enumerate(frames)]


def test_truncated_recording_is_detected(tmp_path):
    path = tmp_path / "k.sptr"
    with SptrWriter(path) as w:
        w.write(build_descriptor(_desc()), 1)
        w.write(build_data(_desc(), 1, 0, 0, np.zeros((SPF, 1), dtype="<i2")), 2)
    path.write_bytes(path.read_bytes()[:-5])
    with SptrReader(path) as r:
        offsets, _, _ = r.index()
        assert len(offsets) == 1 and r.truncated


# ============================================================================ protokol denetimi regresyon testleri
def test_huge_first_sample_or_jump_is_a_frame_error_not_a_crash(h):
    h.feed(build_descriptor(_desc()))
    h.data(0)
    zeros = np.zeros((SPF, 1), dtype="<i2")
    h.feed(build_data(_desc(), 1, 0, 2 ** 63 - 100, zeros))  # 64 bit sayaç taşması: önceden alıcı ölüyordu
    h.feed(build_data(_desc(), 2, 0, 10 + (1 << 41), zeros))  # absürt ileri sıçrama
    h.data(10)
    st = h.stats()
    assert h.store.frame_errors == 2 and st.samples == 20 and st.lost_samples == 0


def test_absurd_sample_rate_descriptor_is_rejected():
    for rate in (1e308, 1e-300, float("inf")):
        harness = Harness()
        harness.feed(build_descriptor(_desc(fs=rate)))  # önceden OverflowError ile alıcıyı durduruyordu
        assert harness.store.frame_errors == 1 and not harness.store.descriptor_frames()
        harness.store.close()


def test_lost_reset_frame_does_not_lock_the_stream(h):
    """Cihaz reset oldu ve first_sample=0 çerçevesi kayboldu: önceden sonraki tüm çerçeveler 'yinelenen'
    sayılıp atılıyordu (cihaz önce 1 saat çalıştıysa 1 saat veri gelmezdi)."""
    h.feed(build_descriptor(_desc()))
    for first in range(0, 1000, SPF):
        h.data(first)
    for first in range(SPF, 200, SPF):  # reset sonrası 0 kayıp, 10'dan devam
        h.data(first)
    st = h.stats()
    assert st.restarts == 1
    # ilk 3 geri çerçeve tek bir eski kopya olabilir diye atılır; 4. ardışık çerçevede reset kabul edilir
    assert st.samples == 1000 + (200 - SPF) - 3 * SPF and st.duplicates == 3
    ring = h.ring()
    assert int(ring.read(ring.write_count - 1, ring.write_count)[1][0, 0]) == 199  # yeni veri tampona yazılıyor
    ring.close()


def test_unknown_frame_type_is_skipped_and_padding_ignored(h):
    from spt.protocol.builder import build_frame
    from spt.protocol.format import FrameType
    h.feed(build_descriptor(_desc()))
    unknown = bytearray(build_frame(FrameType.MARKER, 7, 1, 0, 0, b"\0" * 16))
    unknown[5] = 9  # frame_type = 9 (gelecek sürüm)
    values = np.arange(0, SPF, dtype="<i2")[:, None]
    datagram = bytes(unknown) + build_data(_desc(), 1, 0, 0, values) + b"\0" * 40  # + sıfır dolgu
    h.feed(datagram)
    assert h.store.unknown_frames == 1 and h.store.frame_errors == 0 and h.stats().samples == SPF


def test_planar_fast_path_writes_same_samples_as_interleaved():
    events = []
    store = StreamStore(events.append, session_start_ns=0, ring_seconds=100.0)
    desc = Descriptor(3, 1, 1000.0, SAMPLE_TYPES_BY_NAME["int24"], Layout.PLANAR, "p",
                      (ChannelInfo("a"), ChannelInfo("b"), ChannelInfo("c")))
    rng = np.random.default_rng(1)
    values = rng.integers(-(1 << 23), 1 << 23, size=(500, 3))
    store.handle_datagram(memoryview(build_descriptor(desc)), len(build_descriptor(desc)), "x", 0)
    for i, first in enumerate(range(0, 500, 25)):
        frame = build_data(desc, i, 0, first, values[first:first + 25])
        store.handle_datagram(memoryview(frame), len(frame), "x", 0)
    info = next(e[1] for e in events if e[0] == "stream")
    ring = SharedRing.attach(info.shm_name, desc.sample_type.dtype)
    from spt.dsp.convert import decode_int24
    got = np.column_stack([decode_int24(ring.read(0, 500)[1][:, c], True) for c in range(3)])
    assert np.array_equal(got, values)
    ring.close()
    store.close()


def test_recording_started_mid_stream_gets_descriptor_first(h):
    h.feed(build_descriptor(_desc()))
    h.data(0)
    frames = h.store.descriptor_frames()
    assert len(frames) == 1 and frames[0] == build_descriptor(_desc())


def test_sptr_reader_resyncs_over_a_corrupt_record(tmp_path):
    path = tmp_path / "bozuk.sptr"
    desc = _desc()
    with SptrWriter(path, created_unix_ns=0) as w:
        w.write(build_descriptor(desc), 0)
        for i in range(100):
            w.write(build_data(desc, i, 0, i * SPF, np.zeros((SPF, 1), dtype="<i2")), i)
    raw = bytearray(path.read_bytes())
    with SptrReader(path) as r:
        offsets, _, _ = r.index()
    raw[int(offsets[50]):int(offsets[50]) + 4] = b"XXXX"  # 50. kaydın çerçeve başı bozuk
    path.write_bytes(bytes(raw))
    with SptrReader(path) as r:
        offsets2, _, _ = r.index()
        assert len(offsets2) == 100 and r.resyncs == 1 and r.skipped_bytes > 0 and not r.truncated


@pytest.mark.parametrize("args,expected", [
    (["--fs", "1000", "--channels", "1"], 1000),                              # 1 s < 2 çerçeve: kısa son çerçeve
    (["--fs", "10000", "--channels", "2", "--type", "float32"], 10_000),
    (["--fs", "10000", "--channels", "3", "--layout", "planar"], 10_000),
])
def test_simulator_sends_exactly_duration_samples(args, expected, capsys):
    import socket
    from spt.sources import simulator
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 << 20)
    sock.bind(("127.0.0.1", 0))
    sock.settimeout(0.5)
    events = []
    store = StreamStore(events.append, session_start_ns=0, ring_seconds=100.0)
    try:
        assert simulator.main(["--port", str(sock.getsockname()[1]), "--duration", "1", "--max-rate", *args]) == 0
        while True:
            try:
                data = sock.recv(70_000)
            except socket.timeout:
                break
            store.handle_datagram(memoryview(data), len(data), "127.0.0.1", 0)
        (st,) = store.stats(0)
        assert st.samples == expected and st.lost_samples == 0 and store.frame_errors == 0
        assert f"· {expected} örnek ·" in capsys.readouterr().out
    finally:
        store.close()
        sock.close()
