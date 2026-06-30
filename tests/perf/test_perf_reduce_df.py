"""T3 benchmarks: reduce_df on object vs categorical QDFs.

Run: pytest tests/perf/test_perf_reduce_df.py -m perf --benchmark-only -n0 --no-cov
"""

import pytest

from macrosynergy.management.utils import reduce_df
from tests.perf.data import qdf_for_tier

TIERS = ["small", "medium"]


@pytest.mark.perf
@pytest.mark.parametrize("tier", TIERS)
@pytest.mark.parametrize("categorical", [True, False], ids=["cat", "obj"])
def test_bench_reduce_df_full(benchmark, tier, categorical):
    qdf = qdf_for_tier(tier, categorical=categorical)
    benchmark(reduce_df, qdf)


@pytest.mark.perf
@pytest.mark.parametrize("tier", TIERS)
def test_bench_reduce_df_filtered(benchmark, tier):
    qdf = qdf_for_tier(tier)
    cids = sorted(qdf["cid"].unique())[:3]
    benchmark(lambda d: reduce_df(d, cids=cids), qdf)
