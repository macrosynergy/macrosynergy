"""
Helpers that time a benchmark target and record its call and peak memory.
"""

from __future__ import annotations

import inspect
import tracemalloc
from typing import Any, Callable, Dict, Optional, Tuple

import pandas as pd

from tests.perf.harness.panel_sizes import PanelSize


class PeakMemoryTracker:
    """
    Context manager reporting peak allocation in bytes over the block it wraps.

    Always used outside a measured region, because tracemalloc's overhead would distort
    whatever it wrapped.
    """

    def __init__(self) -> None:
        self.peak_bytes: Optional[int] = None

    def __enter__(self) -> "PeakMemoryTracker":
        tracemalloc.start()
        return self

    def __exit__(self, *exception: Any) -> bool:
        # Returning False lets an exception raised inside the block propagate.
        self.peak_bytes = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        return False


def describe_argument(value: Any) -> str:
    """
    A short description of one argument: its shape for data, its value otherwise.

    Parameters
    ----------
    value : Any
        The argument passed to a measured call.

    Returns
    -------
    str
        For a DataFrame its class, shape and whether it holds categoricals; for a list
        or tuple its length and first three items; for anything else its repr.
    """
    if isinstance(value, pd.DataFrame):
        categorical = any(
            isinstance(dtype, pd.CategoricalDtype) for dtype in value.dtypes
        )
        return f"{type(value).__name__} {value.shape[0]:,}×{value.shape[1]}" + (
            ", categorical" if categorical else ""
        )
    if isinstance(value, (list, tuple)):
        preview = ", ".join(str(item) for item in value[:3])
        more = ", …" if len(value) > 3 else ""
        return f"{type(value).__name__}[{len(value)}]: {preview}{more}"
    return repr(value)


def describe_call(target: Callable, args: Tuple[Any, ...], kwargs: Dict[str, Any]):
    """
    Every argument of a call, keyed by the parameter it binds to.

    Parameters
    ----------
    target : Callable
        The function, method or class called.
    args : Tuple[Any, ...]
        Positional arguments of the call.
    kwargs : Dict[str, Any]
        Keyword arguments of the call.

    Returns
    -------
    Dict[str, str]
        Parameter name to `describe_argument` text, in signature order. Parameters left
        at their defaults are omitted.
    """
    bound = inspect.signature(target).bind_partial(*args, **kwargs)
    return {name: describe_argument(value) for name, value in bound.arguments.items()}


def record_call(
    benchmark, target: Callable, args: Tuple[Any, ...], kwargs: Dict[str, Any]
):
    """
    Record the measured target's qualified name and its call on `benchmark.extra_info`.

    Parameters
    ----------
    benchmark : Any
        pytest-benchmark's `benchmark` fixture.
    target : Callable
        The function under measurement.
    args : Tuple[Any, ...]
        Positional arguments of the measured call.
    kwargs : Dict[str, Any]
        Keyword arguments of the measured call.

    Returns
    -------
    None
    """
    benchmark.extra_info["target"] = f"{target.__module__}:{target.__qualname__}"
    benchmark.extra_info["call"] = describe_call(target, args, kwargs)


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
    record_call(benchmark, target, args, kwargs)
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
    record_call(benchmark, target, args, kwargs)
    with PeakMemoryTracker() as tracker:
        target(*args, **kwargs)
    benchmark.extra_info.update(
        peak_memory_bytes=tracker.peak_bytes, timing_mode="pedantic"
    )
    return benchmark.pedantic(
        target, setup=make_arguments, rounds=panel_size.min_rounds, iterations=1
    )
