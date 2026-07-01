"""T2c benchmarks: _get_tickers_series / add_ticker_column / reduce_df_by_ticker.

Run: pytest tests/perf/test_perf_qdf_ticker_series.py -m perf --benchmark-only -n0 --no-cov
"""

import json

import pytest

from macrosynergy.management.types import QuantamentalDataFrame
from macrosynergy.management.types.qdf.methods import _get_tickers_series
from tests.perf.data import qdf_for_tier
from tests.perf.mem import measure

TIERS = ["small", "medium"]


@pytest.mark.perf
@pytest.mark.parametrize("tier", TIERS)
@pytest.mark.parametrize("categorical", [True, False], ids=["cat", "obj"])
def test_bench_get_tickers_series(benchmark, tier, categorical):
    qdf = qdf_for_tier(tier, categorical=categorical)
    benchmark(_get_tickers_series, qdf)


@pytest.mark.perf
@pytest.mark.parametrize("tier", TIERS)
def test_bench_add_ticker_column(benchmark, tier):
    qdf = qdf_for_tier(tier, categorical=True)
    benchmark(lambda d: QuantamentalDataFrame(d.copy(), categorical=True).add_ticker_column(), qdf)


@pytest.mark.perf
@pytest.mark.parametrize("tier", TIERS)
def test_mem_get_tickers_series(tier, perf_env, tmp_path):
    qdf = qdf_for_tier(tier, categorical=True)
    with measure() as r:
        _get_tickers_series(qdf)
    out = {"target": "get_tickers_series", "tier": tier,
           "wall_s": r.wall_s, "tracemalloc_peak_mib": r.tracemalloc_peak_mib,
           "rss_peak_mib": r.rss_peak_mib, "env": perf_env}
    (tmp_path / "mem.json").write_text(json.dumps(out))
    assert r.wall_s >= 0
