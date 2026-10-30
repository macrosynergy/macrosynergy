"""
Benchmarks for `macrosynergy.management.types.qdf`.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import pytest

from macrosynergy.management.types import QuantamentalDataFrame
from macrosynergy.management.types.qdf.methods import _get_tickers_series
from tests.perf.harness.measurement import measure, measure_mutating
from tests.perf.harness.panel_sizes import PANEL_SIZES

pytestmark = [pytest.mark.perf, pytest.mark.perf_group("qdf")]

QDF_SIZES = PANEL_SIZES.qdf_sizes()


@pytest.mark.parametrize("panel_size", QDF_SIZES, ids=str)
@pytest.mark.parametrize("dtype", ["object", "categorical"], ids=str)
def test_get_tickers_series(benchmark, panel_size, dtype):
    """Derive each row's ticker from the cid and xcat columns of a long panel."""
    df = panel_size.as_qdf(categorical=dtype == "categorical")
    measure(benchmark, _get_tickers_series, df)


@pytest.mark.parametrize("panel_size", QDF_SIZES, ids=str)
def test_add_ticker_column(benchmark, panel_size):
    """Add a ticker column to a categorical QuantamentalDataFrame, in place."""

    def make_arguments() -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        df = panel_size.as_qdf_copy(categorical=True)
        return (QuantamentalDataFrame(df, categorical=True),), {}

    measure_mutating(
        benchmark, panel_size, QuantamentalDataFrame.add_ticker_column, make_arguments
    )
