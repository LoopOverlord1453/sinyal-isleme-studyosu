import numpy as np
import pytest

from spt.core.ring import SharedRing


@pytest.fixture
def ring():
    r = SharedRing.create(capacity=10, channels=2, dtype=np.dtype("<i2"))
    yield r
    r.close()


def _block(start, n):
    v = np.arange(start, start + n, dtype="<i2")
    return np.column_stack((v, -v))


def test_write_and_read_without_wrap(ring):
    ring.write(_block(0, 6))
    first, data = ring.read(0, 6)
    assert first == 0 and data[:, 0].tolist() == list(range(6))


def test_wraparound_keeps_latest(ring):
    ring.write(_block(0, 7))
    ring.write(_block(7, 8))  # toplam 15, kapasite 10
    assert ring.write_count == 15
    first, data = ring.read(0, 15)
    assert first == 5 and data[:, 0].tolist() == list(range(5, 15))
    assert len(ring.segments(5, 15)) == 2


def test_block_larger_than_capacity(ring):
    ring.write(_block(0, 25))
    first, data = ring.read(0, 25)
    assert first == 15 and data[:, 0].tolist() == list(range(15, 25))


def test_skip_records_gap_and_zero_fills(ring):
    ring.write(_block(1, 3))
    ring.skip(4)
    ring.write(_block(8, 2))
    assert ring.write_count == 9
    assert ring.gaps_between(0, 9) == [(3, 7)]
    first, data = ring.read(0, 9)
    assert data[:, 0].tolist() == [1, 2, 3, 0, 0, 0, 0, 8, 9]


def test_reader_attaches_by_name(ring):
    ring.write(_block(0, 4))
    reader = SharedRing.attach(ring.name, ring.dtype)
    try:
        assert reader.write_count == 4
        assert reader.read(0, 4)[1][:, 1].tolist() == [0, -1, -2, -3]
    finally:
        reader.close()


def test_gap_log_overflow_moves_valid_from_and_keeps_order():
    from spt.core.ring import GAP_SLOTS
    r = SharedRing.create(capacity=1 << 16, channels=1, dtype=np.dtype("<i2"))
    try:
        one = np.ones((1, 1), dtype="<i2")
        for _ in range(GAP_SLOTS):  # tam dolu kayıt: hiçbir kayıp unutulmadı
            r.write(one)
            r.skip(1)
        assert r.valid_from() == 0
        gaps = r.gaps_between(0, r.write_count)
        assert len(gaps) == GAP_SLOTS and gaps == sorted(gaps)
        for _ in range(10):  # 10 kayıp daha: en eski 10 kayıt üzerine yazıldı
            r.write(one)
            r.skip(1)
        oldest = min(gs for gs, _ in r.gaps_between(0, r.write_count))
        assert r.valid_from() == oldest > 0  # unutulan kayıpların bölgesi veri sanılmaz
        gaps = r.gaps_between(0, r.write_count)
        assert len(gaps) == GAP_SLOTS and gaps == sorted(gaps)  # halka kaydında sarılmaya rağmen sıralı
        assert r.latest_clean_range(1) == (r.write_count - 2, r.write_count - 1)
    finally:
        r.close()
