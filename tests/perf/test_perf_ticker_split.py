"""T2 benchmarks: split_ticker / get_cid / get_xcat / ticker_df_to_qdf.

Run: pytest tests/perf/test_perf_ticker_split.py -m perf --benchmark-only -n0 --no-cov
"""

import numpy as np
import pytest

from macrosynergy.management.utils import ticker_df_to_qdf
from macrosynergy.management.utils.core import get_cid, get_xcat
from tests.perf.data import wide_ticker_frame


def _ticker_list(n_unique, repeats):
    base = [f"C{i:03d}_XCAT{i % 10}" for i in range(n_unique)]
    return list(np.repeat(base, repeats))


@pytest.mark.perf
@pytest.mark.parametrize("n_unique,repeats", [(2000, 50), (5000, 200)])
def test_bench_get_cid_large_list(benchmark, n_unique, repeats):
    tickers = _ticker_list(n_unique, repeats)
    benchmark(get_cid, tickers)


@pytest.mark.perf
@pytest.mark.parametrize("n_unique,repeats", [(2000, 50), (5000, 200)])
def test_bench_get_xcat_large_list(benchmark, n_unique, repeats):
    tickers = _ticker_list(n_unique, repeats)
    benchmark(get_xcat, tickers)


@pytest.mark.perf
@pytest.mark.parametrize("n_tickers,n_days", [(500, 1300), (2000, 2600)])
def test_bench_ticker_df_to_qdf(benchmark, n_tickers, n_days):
    wide = wide_ticker_frame(n_tickers, n_days)
    benchmark(ticker_df_to_qdf, wide)
