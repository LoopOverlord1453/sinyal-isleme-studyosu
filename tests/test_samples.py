"""samples_signal_data/ altındaki örnek kayıtların geçerliliği ve beklenen içerikleri."""
from pathlib import Path

import numpy as np
import pytest

from spt.core.ring import SharedRing
from spt.core.stream_store import StreamStore
from spt.dsp.convert import physical_column
from spt.protocol.recording import SptrReader

SAMPLES = Path(__file__).resolve().parents[1] / "samples_signal_data"
FILES = sorted(SAMPLES.glob("*.sptr"))


def replay(path: Path):
    """Kaydı alıcıdakiyle aynı akış katmanından geçirir; (olaylar, istatistikler, tamponlar) döndürür."""
    events = []
    with SptrReader(path) as reader:
        offsets, rx, lengths = reader.index()
        store = StreamStore(events.append, int(rx[0]), ring_seconds=120.0)
        for off, t, n in zip(offsets.tolist(), rx.tolist(), lengths.tolist()):
            store.handle_frame(reader.buffer, off, off + n, "kayit", t)
        stats = {s.name: s for s in store.stats(int(rx[-1]))}
        infos = [e[1] for e in events if e[0] == "stream"]
        rings = {i.descriptor.name: (i, SharedRing.attach(i.shm_name, i.descriptor.sample_type.dtype)) for i in infos}
        yield events, stats, rings
        for _, ring in rings.values():
            ring.close()
        store.close()


@pytest.mark.skipif(not FILES, reason="örnek kayıtlar üretilmemiş (tools/make_sample_recordings.py)")
@pytest.mark.parametrize("path", FILES, ids=lambda p: p.name[:2])
def test_sample_parses_cleanly(path):
    for events, stats, _ in replay(path):
        assert not [e for e in events if e[0] == "log" and "Hatalı" in e[2]]
        assert stats and all(s.samples > 0 for s in stats.values())
        if not path.name.startswith("08"):
            assert all(s.lost_samples == 0 and s.restarts == 0 for s in stats.values())


@pytest.mark.skipif(not FILES, reason="örnek kayıtlar üretilmemiş")
def test_lossy_sample_reports_loss_and_reset():
    (path,) = [p for p in FILES if p.name.startswith("08")]
    for _, stats, _ in replay(path):
        (st,) = stats.values()
        assert st.lost_samples > 5000      # %2 rastgele + 0.5 s toplu kayıp
        assert st.restarts == 1


@pytest.mark.skipif(not FILES, reason="örnek kayıtlar üretilmemiş")
def test_iq_sample_has_tone_at_125khz():
    (path,) = [p for p in FILES if p.name.startswith("06")]
    for _, _, rings in replay(path):
        info, ring = rings["iq-alici"]
        _, raw = ring.read(0, 65536)
        x = physical_column(raw[:, 0], info.descriptor.sample_type, info.descriptor.channels[0])
        spec = np.abs(np.fft.fft(x * np.hanning(len(x))))
        freqs = np.fft.fftfreq(len(x), 1 / info.descriptor.sample_rate)
        assert abs(freqs[np.argmax(spec)] - 125_000) < 50
