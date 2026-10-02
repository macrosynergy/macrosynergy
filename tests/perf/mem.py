"""Memory measurement: tracemalloc peak (default, deterministic) + opt-in psutil RSS."""

from __future__ import annotations

import os
import threading
import time
import tracemalloc
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Optional

_MIB = 1024 ** 2


@dataclass
class MemResult:
    wall_s: float = 0.0
    tracemalloc_peak_mib: Optional[float] = None
    rss_peak_mib: Optional[float] = None


def _rss_now_mib() -> Optional[float]:
    try:
        import psutil  # type: ignore

        return psutil.Process(os.getpid()).memory_info().rss / _MIB
    except Exception:
        return None


@contextmanager
def measure(
    *,
    track_rss: Optional[bool] = None,
    track_tracemalloc: bool = True,
    rss_interval_s: float = 0.02,
) -> Iterator[MemResult]:
    if track_rss is None:
        track_rss = os.environ.get("MACROSYN_PERF_RSS", "0") == "1"

    result = MemResult()
    stop = threading.Event()
    peak_holder = {"rss": None}
    baseline_rss = _rss_now_mib() if track_rss else None

    def _sampler():
        while not stop.is_set():
            cur = _rss_now_mib()
            if cur is not None:
                prev = peak_holder["rss"]
                peak_holder["rss"] = cur if prev is None else max(prev, cur)
            time.sleep(rss_interval_s)

    sampler = None
    if track_rss and baseline_rss is not None:
        sampler = threading.Thread(target=_sampler, daemon=True)
        sampler.start()

    if track_tracemalloc:
        tracemalloc.start()

    t0 = time.perf_counter()
    try:
        yield result
    finally:
        result.wall_s = time.perf_counter() - t0
        if track_tracemalloc:
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            result.tracemalloc_peak_mib = peak / _MIB
        if sampler is not None:
            stop.set()
            sampler.join(timeout=1.0)
            if peak_holder["rss"] is not None and baseline_rss is not None:
                result.rss_peak_mib = max(0.0, peak_holder["rss"] - baseline_rss)
