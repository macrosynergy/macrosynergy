"""
Benchmarks for `macrosynergy.management.utils.df_utils`.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import pytest

from macrosynergy.management.utils import (
    reduce_df,
    reduce_df_by_ticker,
    ticker_df_to_qdf,
    update_df,
)
from tests.perf.harness.measurement import measure, measure_mutating
from tests.perf.harness.panel_sizes import PANEL_SIZES

pytestmark = [pytest.mark.perf, pytest.mark.perf_group("df_utils")]

QDF_SIZES = PANEL_SIZES.qdf_sizes()
TICKER_DF_SIZES = PANEL_SIZES.ticker_df_sizes()


@pytest.mark.parametrize("panel_size", QDF_SIZES, ids=str)
def test_reduce_df(benchmark, panel_size):
    """Filter a long panel to all of its own cids and xcats, so every row is kept."""
    measure(
        benchmark,
        reduce_df,
        panel_size.as_qdf(),
        cids=panel_size.cids,
        xcats=panel_size.xcats,
    )


@pytest.mark.parametrize("panel_size", QDF_SIZES, ids=str)
def test_reduce_df_by_ticker(benchmark, panel_size):
    """Filter a long panel to the first half of its tickers."""
    tickers = [f"{cid}_{xcat}" for cid in panel_size.cids for xcat in panel_size.xcats]
    measure(
        benchmark,
        reduce_df_by_ticker,
        panel_size.as_qdf(),
        ticks=tickers[: panel_size.ticker_count // 2],
    )


@pytest.mark.parametrize("panel_size", QDF_SIZES, ids=str)
def test_update_df(benchmark, panel_size):
    """Update a panel with a quarter of its own rows, so every incoming row replaces one."""
    addition = panel_size.as_qdf().tail(panel_size.observation_count // 4)

    def make_arguments() -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        return (panel_size.as_qdf_copy(), addition), {}

    measure_mutating(benchmark, panel_size, update_df, make_arguments)


@pytest.mark.parametrize("panel_size", TICKER_DF_SIZES, ids=str)
def test_ticker_df_to_qdf(benchmark, panel_size):
    """Reshape a wide dates-by-tickers DataFrame into a long panel."""
    measure(benchmark, ticker_df_to_qdf, panel_size.as_ticker_df())
