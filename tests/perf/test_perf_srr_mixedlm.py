"""T4 benchmarks: SignalReturnRelations MixedLM panel test (the dominant SRR cost).

Run: pytest tests/perf/test_perf_srr_mixedlm.py -m perf --benchmark-only -n0 --no-cov
"""

import pytest

from macrosynergy.signal.signal_return_relations import SignalReturnRelations
from tests.perf.data import srr_panel


def _build_srr(n_signals, n_returns):
    df = srr_panel(n_cids=6, n_dates=600, n_signals=n_signals, n_returns=n_returns)
    return SignalReturnRelations(
        df,
        rets=[f"XR{i:02d}" for i in range(n_returns)],
        sigs=[f"SIG{i:02d}" for i in range(n_signals)],
        cids=sorted(df["cid"].unique()),
        freqs=["M"],
        ms_panel_test=True,
    )


@pytest.mark.perf
@pytest.mark.parametrize("n_signals,n_returns", [(1, 1), (2, 3)])
def test_bench_srr_single_statistic_table(benchmark, n_signals, n_returns):
    srr = _build_srr(n_signals, n_returns)
    benchmark(srr.single_statistic_table, stat="accuracy")
