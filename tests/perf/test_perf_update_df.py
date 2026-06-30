"""T1 benchmarks: update_df in a growing loop + single update_tickers.

Run: pytest tests/perf/test_perf_update_df.py -m perf --benchmark-only -n0 --no-cov
"""

import pytest

from macrosynergy.management.utils import update_df, update_tickers
from tests.perf.data import update_df_pieces

TIERS = ["small", "medium"]


def _growing_loop(base, pieces):
    acc = base
    for p in pieces:
        acc = update_df(acc, p)
    return acc


@pytest.mark.perf
@pytest.mark.parametrize("tier", TIERS)
@pytest.mark.parametrize("categorical", [True, False], ids=["cat", "obj"])
def test_bench_update_df_growing_loop(benchmark, tier, categorical):
    base, pieces = update_df_pieces(tier, n_pieces=5, categorical=categorical)
    benchmark(_growing_loop, base, pieces)


@pytest.mark.perf
@pytest.mark.parametrize("tier", TIERS)
def test_bench_update_tickers(benchmark, tier):
    base, pieces = update_df_pieces(tier, n_pieces=2)
    benchmark(update_tickers, base, pieces[0])
