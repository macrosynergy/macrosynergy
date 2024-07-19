"""
List of timeseries methods to be implemented / re-implemented / re-factored for the new architecture:

- Information state changes
- Z-score
- VolatilityEstimationMethods
- Expanding window
- Rolling window
- Moving average
- DropNA
- Fill to bday calendar
- Infer frequency
- Resample
- Shift
- Reindex
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Union, Optional, Callable, Dict, Any
from numbers import Number


def information_state_changes(ts: pd.Series, threshold: Number = 0.0) -> pd.Series:
    """
    Keep only the first observation of a series of consecutive observations that are equal.
    This is useful to identify when a time series changes state. This is effectively a
    run-length encoding of a quantamental time series (since it's daily).
    """
    mask: pd.Series = ts.diff().abs() > threshold
    mask.iloc[0] = True
    return ts[mask]


def z_score(ts: pd.Series) -> pd.Series:
    """
    Compute the z-score of a time series.
    """
    return (ts - ts.mean()) / ts.std()


def infer_frequency(ts: pd.Series) -> str:
    """
    Infer the frequency of a time series.
    """
    return pd.infer_freq(ts.index)


def forward_fill(
    ts: pd.Series,
    value: Optional[Number] = None,
    limit: Optional[int] = None,
) -> pd.Series:
    """
    Forward fill missing values in a time series.
    """
    # if there is a value to fill with, use it else ffill
    if value is not None:
        return ts.fillna(value, limit=limit)
    return ts.ffill(limit=limit)


def expanding_window_operation(
    ts: pd.Series,
    func: Callable[[pd.Series, Dict[str, Any]], pd.Series],
    func_args: Dict[str, Any] = {},
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """
    Apply a custom function over an expanding window on a time series with an optional
    limit on the size of the window.
    """
    results = []
    for i in range(1, len(ts) + 1):
        if limit is None or limit == -1:
            window = ts.iloc[:i]
        else:
            window = ts.iloc[max(0, i - limit) : i]
        result = func(window, **func_args)
        results.append(result)

    return pd.concat(results, axis=1).transpose()


def rolling_window_operation(
    ts: pd.Series,
    window_size: int,
    func: Callable[[pd.Series, Dict[str, Any]], pd.Series],
    func_args: Dict[str, Any] = {},
) -> pd.DataFrame:
    """
    Apply a custom function over a rolling window on a time series.
    """
    results = []
    for i in range(window_size, len(ts) + 1):
        window = ts.iloc[i - window_size : i]
        result = func(window, **func_args)
        results.append(result)

    return pd.concat(results, axis=1).transpose()
