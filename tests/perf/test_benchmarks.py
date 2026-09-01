"""
Every performance benchmark. Each function measures one target across the selected
tiers, and peak memory comes from a separate untimed call because tracemalloc cannot sit
inside the measured region.

Unlike the rest of the repository's tests, this module is pytest-native function style
rather than `unittest.TestCase`: pytest refuses to inject fixtures into `TestCase`
methods, and every benchmark here depends on pytest-benchmark's `benchmark` fixture.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable, Dict, List, Tuple

import pytest

from macrosynergy.management.types import QuantamentalDataFrame
from macrosynergy.management.types.qdf.methods import _get_tickers_series
from macrosynergy.management.utils import (
    reduce_df,
    reduce_df_by_ticker,
    ticker_df_to_qdf,
    update_df,
)
from macrosynergy.signal.signal_return_relations import SignalReturnRelations
from tests.perf.machine import PeakMemoryTracker
from tests.perf.panel_sizes import PANEL_SIZES, PanelSize

pytestmark = pytest.mark.perf

QDF_SIZES = PANEL_SIZES.qdf_sizes()
TICKER_DF_SIZES = PANEL_SIZES.ticker_df_sizes()

SIGNAL_XCAT_COUNT = 2
RETURN_XCAT_COUNT = 1


def measure(benchmark, target: Callable, *args: Any, **kwargs: Any):
    """
    Record peak memory for one untimed call, then benchmark the same call.

    Parameters
    ----------
    benchmark : Any
        pytest-benchmark's `benchmark` fixture.
    target : Callable
        The function under measurement.
    *args : Any
        Positional arguments for `target`.
    **kwargs : Any
        Keyword arguments for `target`.

    Returns
    -------
    Any
        Whatever `target` returned on the measured call.
    """
    with PeakMemoryTracker() as tracker:
        target(*args, **kwargs)
    benchmark.extra_info["peak_memory_bytes"] = tracker.peak_bytes
    return benchmark(target, *args, **kwargs)


def measure_mutating(
    benchmark, panel_size: PanelSize, target: Callable, make_arguments: Callable
):
    """
    Benchmark a target that modifies its input, rebuilding that input each round.

    `pedantic` is used only here, because it disables calibration and auto-ranging and
    pins `iterations` to 1.

    Parameters
    ----------
    benchmark : Any
        pytest-benchmark's `benchmark` fixture.
    panel_size : PanelSize
        The size being measured, read for its round budget.
    target : Callable
        The function under measurement.
    make_arguments : Callable
        Zero-argument callable returning `(args, kwargs)`, so building the input stays
        outside the measured region.

    Returns
    -------
    Any
        Whatever `target` returned on the last measured round.
    """
    args, kwargs = make_arguments()
    with PeakMemoryTracker() as tracker:
        target(*args, **kwargs)
    benchmark.extra_info.update(
        peak_memory_bytes=tracker.peak_bytes, timing_mode="pedantic"
    )
    return benchmark.pedantic(
        target, setup=make_arguments, rounds=panel_size.min_rounds, iterations=1
    )


def signal_return_sizes(*only_tiers: str) -> List[Any]:
    """
    Long-format sizes whose xcats carry signal and return roles.

    Parameters
    ----------
    *only_tiers : str
        Tiers the benchmark is limited to. Omit to allow every selected tier.

    Returns
    -------
    List[Any]
        One `pytest.param` per measured tier, each holding a size whose tickers pair
        every cid with the signal and return categories.
    """
    role_xcats = [f"SIG{i:02d}" for i in range(SIGNAL_XCAT_COUNT)]
    role_xcats += [f"XR{i:02d}" for i in range(RETURN_XCAT_COUNT)]
    parameters = []
    for parameter in PANEL_SIZES.qdf_sizes(*only_tiers):
        panel_size = parameter.values[0]
        if panel_size is None:
            parameters.append(parameter)
            continue
        tickers = tuple(f"{cid}_{xcat}" for cid in panel_size.cids for xcat in role_xcats)
        parameters.append(
            pytest.param(
                replace(panel_size, cid_count=None, xcat_count=None, tickers=tickers),
                id=panel_size.tier,
                marks=parameter.marks,
            )
        )
    return parameters


# ---------------------------------------------------------------- quantamental dataframe


@pytest.mark.perf_group("qdf")
@pytest.mark.parametrize("panel_size", QDF_SIZES, ids=str)
@pytest.mark.parametrize("dtype", ["object", "categorical"], ids=str)
def test_get_tickers_series(benchmark, panel_size, dtype):
    df = panel_size.as_qdf(categorical=dtype == "categorical")
    measure(benchmark, _get_tickers_series, df)


@pytest.mark.perf_group("qdf")
@pytest.mark.parametrize("panel_size", QDF_SIZES, ids=str)
def test_add_ticker_column(benchmark, panel_size):
    def make_arguments() -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        df = panel_size.as_qdf_copy(categorical=True)
        return (QuantamentalDataFrame(df, categorical=True),), {}

    measure_mutating(
        benchmark, panel_size, lambda df: df.add_ticker_column(), make_arguments
    )


@pytest.mark.perf_group("qdf")
@pytest.mark.parametrize("panel_size", QDF_SIZES, ids=str)
def test_reduce_df(benchmark, panel_size):
    measure(
        benchmark,
        reduce_df,
        panel_size.as_qdf(),
        cids=panel_size.cids,
        xcats=panel_size.xcats,
    )


@pytest.mark.perf_group("qdf")
@pytest.mark.parametrize("panel_size", QDF_SIZES, ids=str)
def test_reduce_df_by_ticker(benchmark, panel_size):
    tickers = [f"{cid}_{xcat}" for cid in panel_size.cids for xcat in panel_size.xcats]
    measure(
        benchmark,
        reduce_df_by_ticker,
        panel_size.as_qdf(),
        ticks=tickers[: panel_size.ticker_count // 2],
    )


@pytest.mark.perf_group("qdf")
@pytest.mark.parametrize("panel_size", QDF_SIZES, ids=str)
def test_update_df(benchmark, panel_size):
    addition = panel_size.as_qdf().tail(panel_size.observation_count // 4)

    def make_arguments() -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        return (panel_size.as_qdf_copy(), addition), {}

    measure_mutating(benchmark, panel_size, update_df, make_arguments)


# ----------------------------------------------------------------------- ticker dataframe


@pytest.mark.perf_group("qdf")
@pytest.mark.parametrize("panel_size", TICKER_DF_SIZES, ids=str)
def test_ticker_df_to_qdf(benchmark, panel_size):
    measure(benchmark, ticker_df_to_qdf, panel_size.as_ticker_df())


# ------------------------------------------------------------ signal and return relations


@pytest.mark.perf_group("signal_returns")
@pytest.mark.parametrize("panel_size", signal_return_sizes("tiny", "small"), ids=str)
def test_single_statistic_table(benchmark, panel_size):
    relations = SignalReturnRelations(
        panel_size.as_qdf(),
        rets=[f"XR{i:02d}" for i in range(RETURN_XCAT_COUNT)],
        sigs=[f"SIG{i:02d}" for i in range(SIGNAL_XCAT_COUNT)],
        cids=panel_size.cids,
        freqs=["M"],
        ms_panel_test=True,
    )
    measure(
        benchmark,
        relations.single_statistic_table,
        stat="map_pval",
        type="panel",
        show_heatmap=False,
    )
