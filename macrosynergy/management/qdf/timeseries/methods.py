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


def downsample(ts: pd.Series, freq: str, method: str = "mean") -> pd.Series:
    """
    Downsample a time series to a lower frequency.
    """
    agg_methods = ["mean", "median", "min", "max", "first", "last"]
    errstr = f"`method` must be one of {agg_methods}, got {method} instead."
    if method not in agg_methods:
        raise ValueError(errstr)

    return ts.resample(freq).agg(method)


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


def fill_to_business_day(ts: pd.Series, ffill: bool = True) -> pd.Series:
    """
    Forward fill a time series to a business day calendar.
    """
    date_range = pd.date_range(ts.index.min(), ts.index.max(), freq="B")
    return ts.reindex(date_range, method="ffill" if ffill else None)


def fill_to_next_period(ts: pd.Series, freq: str, ffill: bool = True) -> pd.Series:
    """
    Forward fill the last observation of a time series to the next period.
    I.E. for a monthly series, fill the last observation from this month to the start of
    the next month.
    """
    raise NotImplementedError


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


class VolatilityEstimationMethodsCore:
    """
    Class to hold methods for calculating standard deviations.
    Each method must comply to the following signature:
        `func(s: pd.Series, **kwargs) -> pd.Series`

    Currently supported methods are:

    - `std`: Standard deviation
    - `abs`: Mean absolute deviation
    - `exp`: Exponentially weighted standard deviation
    - `exp_abs`: Exponentially weighted mean absolute deviation
    """

    @staticmethod
    def std(s: pd.Series, min_periods: int, **kwargs) -> pd.Series:
        """
        Calculate the expanding standard deviation of a Series.
        """
        return s.expanding(min_periods=min_periods).std()

    @staticmethod
    def abs(s: pd.Series, min_periods: int, **kwargs) -> pd.Series:
        """
        Calculate the expanding mean absolute deviation of a Series.
        """
        mean = s.expanding(min_periods=min_periods).mean()
        return (s - mean.bfill()).abs().expanding(min_periods=min_periods).mean()

    @staticmethod
    def exp(s: pd.Series, halflife: int, min_periods: int, **kwargs) -> pd.Series:
        """
        Calculate the exponentially weighted standard deviation of a Series.

        :param <pd.Series> s: The Series to calculate the exponentially weighted standard
            deviation for.
        :param <int> halflife: The halflife of the exponential weighting.
        :param <int> min_periods: The minimum number of periods required for the
            calculation.
        :return <pd.Series>: The exponentially weighted standard deviation of the Series.
        """
        return s.ewm(halflife=halflife, min_periods=min_periods).std()

    @staticmethod
    def exp_abs(s: pd.Series, halflife: int, min_periods: int, **kwargs) -> pd.Series:
        """
        Calculate the exponentially weighted mean absolute deviation of a Series.

        :param <pd.Series> s: The Series to calculate the exponentially weighted absolute
            standard deviation for.
        :param <int> halflife: The halflife of the exponential weighting.
        :param <int> min_periods: The minimum number of periods required for the calculation.
        :return <pd.Series>: The exponentially weighted absolute standard deviation of the Series.
        """
        mean = s.ewm(halflife=halflife, min_periods=min_periods).mean()
        sd = (
            (s - mean.bfill())
            .abs()
            .ewm(halflife=halflife, min_periods=min_periods)
            .mean()
        )
        return sd
