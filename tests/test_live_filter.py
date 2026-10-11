"""Canlı filtre (paylaşımlı tampon üzerinde blok blok) ve anlık görüntü (Dondur + çevrimdışı filtre)."""
from dataclasses import replace

import numpy as np
import pytest

from spt.core.live_filter import LiveFilter
from spt.core.ring import SharedRing
from spt.core.snapshot import filter_snapshot, take_snapshot
from spt.dsp import filters as F
from spt.dsp.chain import ChainStage, ProcessingChain
from spt.protocol.format import SAMPLE_TYPES_BY_NAME, ChannelInfo, Descriptor, Layout

FS = 10_000.0
SPECS = [
    F.FilterSpec(fs=FS, family="special", method="dc_block", dc_cutoff=2.0),
    F.FilterSpec(fs=FS, family="iir", method="ellip", band="bandpass", order=4, f1=300.0, f2=1500.0,
                 ripple_db=0.5, atten_db=60.0),
    F.FilterSpec(fs=FS, family="special", method="notch", notch_f0=50.0, harmonics=2),
]
CHAIN = ProcessingChain(ChainStage(s) for s in SPECS)
SCALE = 0.001  # int16 → volt


class _Handle:
    """StreamHandle'ın canlı filtrenin kullandığı kısmı (paylaşımlı tampon + tanımlayıcı)."""

    def __init__(self, ring: SharedRing, stype: str = "int16", channels: int = 2):
        self.key = "test#1"
        self.source = "test"
        self.t0 = 0.0
        self.ring = ring
        self.desc = Descriptor(1, 1, FS, SAMPLE_TYPES_BY_NAME[stype], Layout.INTERLEAVED, "test",
                               tuple(ChannelInfo(f"ch{i}", "V", SCALE, 0.0) for i in range(channels)))

    @property
    def fs(self):
        return FS


def _signal(n, rng, channels=2):
    t = np.arange(n) / FS
    clean = np.sin(2 * np.pi * 800 * t) + 0.5 * np.sin(2 * np.pi * 50 * t) + 0.3
    noise = 0.05 * rng.standard_normal((n, channels))
    return np.round((clean[:, None] + noise) / SCALE).astype(np.int16)


@pytest.fixture
def ring():
    r = SharedRing.create(200_000, 2, np.dtype("<i2"))
    yield r
    r.close()


def _reference(raw_column):
    designs, _ = CHAIN.designs(FS)
    return F.ChainRunner(designs).process(raw_column.astype(np.float64) * SCALE)


def test_live_filter_in_blocks_equals_one_shot(ring):
    rng = np.random.default_rng(5)
    data = _signal(60_000, rng)
    handle = _Handle(ring)
    lf = LiveFilter()
    lf.set_chain(CHAIN)
    lf.set_enabled(True)
    pos = 0
    while pos < len(data):  # alıcı gibi rastgele boyutlu paketler, arada arayüz güncellemesi
        n = int(rng.integers(1, 3000))
        ring.write(data[pos:pos + n])
        pos += n
        for ch in (0, 1):
            lf.update(handle, ch)
    for ch in (0, 1):
        got = lf.read_clean(handle.key, ch, 0, len(data))
        expected = _reference(data[:, ch])
        assert got is not None
        assert np.max(np.abs(got - expected)) < 1e-9


def test_gap_yields_nan_and_restarts_filter(ring):
    rng = np.random.default_rng(6)
    a, b = _signal(20_000, rng), _signal(15_000, rng)
    handle = _Handle(ring)
    lf = LiveFilter()
    lf.set_chain(CHAIN)
    lf.set_enabled(True)
    ring.write(a)
    lf.update(handle, 0)
    ring.skip(500)  # 500 örnek kayıp
    ring.write(b)
    lf.update(handle, 0)
    segs = lf.segments(handle.key, 0, 0, ring.write_count)
    y = np.concatenate([v for _, v in segs])
    assert np.all(np.isnan(y[20_000:20_500]))
    assert np.max(np.abs(y[:20_000] - _reference(a[:, 0]))) < 1e-9
    assert np.max(np.abs(y[20_500:] - _reference(b[:, 0]))) < 1e-9  # kayıptan sonra temiz yeniden başlangıç
    assert lf.read_clean(handle.key, 0, 19_000, 21_000) is None


def test_falling_behind_skips_to_newest_data():
    ring = SharedRing.create(10_000, 2, np.dtype("<i2"))
    try:
        rng = np.random.default_rng(7)
        handle = _Handle(ring)
        lf = LiveFilter()
        lf.set_chain(CHAIN)
        lf.set_enabled(True)
        ring.write(_signal(5_000, rng))
        lf.update(handle, 0)
        later = _signal(25_000, rng)  # tampon (10k) iki kez dolup taştı, arada güncelleme yok
        ring.write(later)
        lf.update(handle, 0)
        assert lf.skipped == 1
        got = lf.read_clean(handle.key, 0, ring.write_count - 10_000, ring.write_count)
        assert np.max(np.abs(got - _reference(later[-10_000:, 0]))) < 1e-9
    finally:
        ring.close()


def test_chain_change_and_disable():
    lf = LiveFilter()
    assert not lf.active
    lf.set_chain(CHAIN)
    assert not lf.active
    lf.set_enabled(True)
    assert lf.active
    lf.set_chain(CHAIN.toggled(0, False).toggled(1, False).toggled(2, False))
    assert not lf.active


def test_stage_invalid_for_stream_rate_is_reported(ring):
    handle = _Handle(ring)
    lf = LiveFilter()
    lf.set_chain(ProcessingChain([ChainStage(replace(SPECS[1], f1=3000.0, f2=6000.0))]))  # f₂ > 5 kHz Nyquist
    lf.set_enabled(True)
    ring.write(_signal(1000, np.random.default_rng(8)))
    lf.update(handle, 0)
    assert not lf.has(handle.key, 0)
    assert "Nyquist" in lf.errors[FS][0]


# ============================================================================ anlık görüntü
def test_snapshot_is_an_independent_copy_with_gaps(ring):
    rng = np.random.default_rng(9)
    a, b = _signal(30_000, rng), _signal(30_000, rng)
    ring.write(a)
    ring.skip(1000)
    ring.write(b)
    handle = _Handle(ring)
    snap = take_snapshot(handle, ring.write_count, 1, "Anlık 1")
    assert snap.samples == 61_000 and snap.ring.base == 0
    assert snap.ring.gaps == [(30_000, 31_000)]
    assert snap.ring.clean_ranges() == [(0, 30_000), (31_000, 61_000)]
    before = snap.ring.data.copy()
    ring.write(_signal(150_000, rng))  # canlı akış devam ediyor; kopya değişmemeli
    assert np.array_equal(snap.ring.data, before)
    assert snap.ring.latest_clean_range(4096) == (61_000 - 4096, 61_000)
    assert snap.ring.latest_clean_range(30_000) == (31_000, 61_000)
    assert snap.ring.latest_clean_range(30_000, end=40_000) == (0, 30_000)  # kaybın öncesine kayar
    assert snap.ring.latest_clean_range(31_000) is None


def test_offline_causal_equals_live_and_zero_phase_has_no_delay(ring):
    rng = np.random.default_rng(10)
    data = _signal(40_000, rng)
    ring.write(data)
    handle = _Handle(ring)
    snap = take_snapshot(handle, ring.write_count, 1, "Anlık 1")
    causal = filter_snapshot(snap, CHAIN, zero_phase=False, snap_id=2)
    # int16 girdi: sonuç float32'de saklanır (24 bit mantis 16 bit girdiyi korur, bellek yarı)
    assert causal.parent == 1 and causal.desc.sample_type.name == "float32" and causal.ring.data.dtype == np.float32
    assert causal.desc.channels[0].name == "ch0 · filtreli" and causal.desc.channels[0].scale == 1.0
    ref = _reference(data[:, 0])
    assert np.max(np.abs(causal.ring.data[:, 0] - ref)) <= 1e-6 * np.abs(ref).max()  # float32 yuvarlaması

    # Sıfır fazlı: 800 Hz bileşen bant geçirenden gecikmesiz ve |H|² genlikle geçer
    t = np.arange(40_000) / FS
    tone = np.round(np.sin(2 * np.pi * 800 * t) / SCALE).astype(np.int16)
    ring2 = SharedRing.create(50_000, 2, np.dtype("<i2"))
    try:
        ring2.write(np.column_stack((tone, tone)))
        snap2 = take_snapshot(_Handle(ring2), ring2.write_count, 3, "ton")
        zp = filter_snapshot(snap2, CHAIN, zero_phase=True, snap_id=4)
        designs, _ = CHAIN.designs(FS)
        gain = np.prod([abs(F.analyze(d, np.array([800.0])).h[0]) for d in designs]) ** 2
        mid = slice(10_000, 30_000)
        assert np.max(np.abs(zp.ring.data[mid, 0] - gain * tone[mid] * SCALE)) < 2e-3
    finally:
        ring2.close()


def test_offline_filter_keeps_gaps(ring):
    rng = np.random.default_rng(11)
    ring.write(_signal(20_000, rng))
    ring.skip(300)
    ring.write(_signal(20_000, rng))
    snap = take_snapshot(_Handle(ring), ring.write_count, 1, "A")
    for zero_phase in (False, True):
        out = filter_snapshot(snap, CHAIN, zero_phase, 2)
        y = out.ring.data[:, 0]
        assert np.all(np.isnan(y[20_000:20_300])) and np.all(np.isfinite(y[:20_000])) and np.all(np.isfinite(y[20_300:]))
        assert out.ring.gaps == [(20_000, 20_300)]


def test_filtering_with_empty_chain_is_an_error(ring):
    ring.write(_signal(100, np.random.default_rng(12)))
    snap = take_snapshot(_Handle(ring), ring.write_count, 1, "A")
    with pytest.raises(F.FilterError):
        filter_snapshot(snap, ProcessingChain(), False, 2)


def test_nan_samples_in_float_stream_do_not_poison_iir():
    # float akışta NaN / inf örnek: yalnız o örnekler NaN çıkar, filtre sonraki ilk sonlu örnekten temiz başlar
    ring = SharedRing.create(50_000, 1, np.dtype("<f4"))
    try:
        rng = np.random.default_rng(13)
        x = (np.sin(2 * np.pi * 800 * np.arange(30_000) / FS) + 0.1 * rng.standard_normal(30_000)).astype(np.float32)
        x[10_000:10_003] = np.nan
        x[20_000] = np.inf
        handle = _Handle(ring, "float32", channels=1)
        lf = LiveFilter()
        lf.set_chain(CHAIN)
        lf.set_enabled(True)
        for i in range(0, len(x), 2500):
            ring.write(x[i:i + 2500, None])
            lf.update(handle, 0)
        y = np.concatenate([v for _, v in lf.segments(handle.key, 0, 0, ring.write_count)])
        phys = x.astype(np.float64) * SCALE
        assert np.all(np.isnan(y[10_000:10_003])) and np.isnan(y[20_000])
        assert np.isfinite(y[:10_000]).all() and np.isfinite(y[10_003:20_000]).all() and np.isfinite(y[20_001:]).all()
        assert np.max(np.abs(y[10_003:20_000] - _reference_float(phys[10_003:20_000]))) < 1e-9
        assert np.max(np.abs(y[20_001:] - _reference_float(phys[20_001:]))) < 1e-9
    finally:
        ring.close()


def _reference_float(x):
    designs, _ = CHAIN.designs(FS)
    return F.ChainRunner(designs).process(x)


def test_snapshot_filter_restarts_after_nan_samples():
    ring = SharedRing.create(30_000, 1, np.dtype("<f4"))
    try:
        x = np.sin(2 * np.pi * 800 * np.arange(20_000) / FS).astype(np.float32)
        x[8_000:8_010] = np.nan
        ring.write(x[:, None])
        snap = take_snapshot(_Handle(ring, "float32", channels=1), ring.write_count, 1, "A")
        for zero_phase in (False, True):
            y = filter_snapshot(snap, CHAIN, zero_phase, 2).ring.data[:, 0]
            assert np.all(np.isnan(y[8_000:8_010]))
            assert np.isfinite(y[:8_000]).all() and np.isfinite(y[8_010:]).all()
    finally:
        ring.close()


def test_new_track_filters_only_requested_history_within_budget():
    # Denetim: zincir değişince (geri al dahil) 4M örneğe kadar tüm tampon tek seferde filtreleniyordu (GUI donması)
    ring = SharedRing.create(300_000, 2, np.dtype("<i2"))
    try:
        data = _signal(250_000, np.random.default_rng(21))
        ring.write(data)
        handle = _Handle(ring)
        lf = LiveFilter()
        lf.set_chain(CHAIN)
        lf.set_enabled(True)
        lf.update(handle, 0, history=20_000, budget=8_000)
        track = lf._tracks[(handle.key, 0)]
        assert track.out.write_count == 250_000 - 20_000 + 8_000  # geçmiş sınırı + bütçe
        while track.out.write_count < ring.write_count:
            lf.update(handle, 0, history=20_000, budget=8_000)
        designs, _ = CHAIN.designs(FS)
        expected = F.ChainRunner(designs).process(data[-20_000:, 0].astype(np.float64) * SCALE)
        assert np.max(np.abs(lf.read_clean(handle.key, 0, 230_000, 250_000) - expected)) < 1e-9
    finally:
        ring.close()


def test_long_median_is_not_applied_live():
    from spt.core.live_filter import LIVE_MEDIAN_MAX
    ring = SharedRing.create(20_000, 2, np.dtype("<i2"))
    try:
        ring.write(_signal(10_000, np.random.default_rng(22)))
        lf = LiveFilter()
        lf.set_chain(ProcessingChain([ChainStage(F.FilterSpec(fs=FS, family="special", method="median",
                                                              length=LIVE_MEDIAN_MAX + 2))]))
        lf.set_enabled(True)
        lf.update(_Handle(ring), 0)
        assert any("medyan" in e for e in lf.errors[FS]) and not lf.has("test#1", 0)
        lf.prune_errors([])
        assert lf.errors == {}
    finally:
        ring.close()
