import numpy as np
from tests.perf.mem import measure, MemResult


def test_measure_records_wall_and_tracemalloc():
    with measure() as r:
        _alloc = [0] * 1_000_000  # noqa: F841
    assert isinstance(r, MemResult)
    assert r.wall_s >= 0.0
    assert r.tracemalloc_peak_mib is not None and r.tracemalloc_peak_mib > 0


def test_measure_can_disable_tracemalloc():
    with measure(track_tracemalloc=False) as r:
        _ = np.zeros(1000)
    assert r.tracemalloc_peak_mib is None


def test_measure_rss_off_by_default():
    with measure() as r:
        pass
    assert r.rss_peak_mib is None  # opt-in only
