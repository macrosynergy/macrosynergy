"""
Run Granger Causality Test on a standardized quantamental dataframe.

**NOTE: This module is under development, and is not yet ready for production use.**
"""

import warnings
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import statsmodels
from packaging import version
from statsmodels.tsa.stattools import grangercausalitytests

from macrosynergy.management.simulate import make_test_df
from macrosynergy.management.types import QuantamentalDataFrame
from macrosynergy.management.utils import (
    downsample_df_on_real_date,
    is_valid_iso_date,
    qdf_to_ticker_df,
    reduce_df_by_ticker,
)

import logging

logger = logging.getLogger(__name__)


def _statsmodels_compatibility_wrapper(
    x: Any = None, maxlag: Any = None, addconst: Any = None, verbose: Any = None
) -> Any:
    if version.parse(statsmodels.__version__) < version.parse("0.15.0"):
        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=FutureWarning)
            return grangercausalitytests(x, maxlag, addconst, False)

    else:
        return grangercausalitytests(x, maxlag, addconst)


def _granger_causality_backend(
    data: pd.DataFrame, max_lag: Union[int, List[int]], add_constant: bool = True
) -> Dict[Any, Any]:
    assert len(data.columns) == 2, "`data` must have only two columns"
    assert (
        isinstance(max_lag, int)
        or isinstance(max_lag, list)
        and all(isinstance(l, int) for l in max_lag)
        and len(max_lag) > 0
    ), "`max_lag` must be an integer or a list of integers"
    assert isinstance(add_constant, bool), "`add_constant` must be a boolean"

    arguments: Dict[str, Any] = dict(
        x=data,
        maxlag=max_lag,
        addconst=add_constant,
    )

    return _statsmodels_compatibility_wrapper(**arguments)


def _get_tickers(
    tickers: Optional[List[str]] = None,
    cids: Optional[List[str]] = None,
    xcats: Optional[List[str]] = None,
) -> List[str]:
    """
    Simply returns the tickers if they are specified. If they are not specified, then
    the function forms the list of tickers from the `cids` and `xcats` such that the
    order of the formed tickers is preserved.

    :param <List[str]> tickers: A list of tickers.
    :param <Union[str, List[str]]> cids: One or two cids.
    :param <Union[str, List[str]]> xcats: One or two xcats.
    """

    if tickers is not None:
        return tickers
    else:
        if isinstance(cids, str):
            cids: List[str] = [cids]
        if isinstance(xcats, str):
            xcats: List[str] = [xcats]
        return [f"{c}_{x}" for c in cids for x in xcats]


def _type_checks(
    df: pd.DataFrame,
    tickers: Optional[List[str]],
    cids: Optional[List[str]],
    xcats: Optional[List[str]],
    max_lag: Union[int, List[int]],
    add_constant: bool,
    start: Optional[str],
    end: Optional[str],
    freq: str,
    agg: str,
    metric: str,
) -> bool:
    """
    Does type checks on the inputs to `granger_causality_test`.
    All inputs are checked for type and value errors.

    :return <bool>: True if all type checks pass.
    :raises <TypeError>: If any of the inputs are of the wrong type.
    :raises <ValueError>: If any of the input values are invalid.
    """

    if not isinstance(df, QuantamentalDataFrame):
        raise TypeError("df must be a standardized quantamental dataframe")

    if not isinstance(metric, str):
        raise TypeError("`metric` must be a string")

    err_msg: str = f"`metric` '{metric}' not found in `df`"
    if metric not in df.columns:
        raise ValueError(err_msg)

    if not isinstance(max_lag, (int, list)):
        raise TypeError("`max_lag` must be an integer or a list of integers")
    elif isinstance(max_lag, list) and not all(isinstance(l, int) for l in max_lag):
        raise TypeError("`max_lag` must be an integer or a list of integers")

    for dt, nm in zip([start, end], ["start", "end"]):
        if dt is not None and not is_valid_iso_date(dt):
            raise ValueError(f"{nm} must be a valid ISO date")

    if isinstance(cids, str):
        cids: List[str] = [cids]
    if isinstance(xcats, str):
        xcats: List[str] = [xcats]

    if bool(cids) ^ bool(xcats):
        raise ValueError("`cids` and `xcats` must be specified together")

    bcidxcats: bool = bool(cids) and bool(xcats)

    if bool(tickers) and (bcidxcats):
        raise ValueError(
            "`tickers` cannot be specified if `cids` & `xcats` are specified"
        )

    found_tickers: List[str] = (df["cid"] + "_" + df["xcat"]).unique().tolist()

    if bool(tickers):
        # check if there are only two
        if len(set(tickers)) != 2:
            raise ValueError("Only two tickers can be specified in `tickers`")

        if not all(isinstance(t, str) for t in tickers):
            raise TypeError("`tickers` must be a list of strings")

        if not set(tickers).issubset(set(found_tickers)):
            raise ValueError(
                "All tickers specified in `tickers` must be in `df`."
                f"Missing tickers: {set(tickers) - set(found_tickers)}"
            )

    else:
        assert bcidxcats, "Failed to resolve tickers"

    if bcidxcats:
        for lx, nm in zip([cids, xcats], ["cid", "xcat"]):
            if not (isinstance(lx, list) and all(isinstance(x, str) for x in lx)):
                raise TypeError(f"`{nm}` must be a list of strings")
            if not set(lx).issubset(set(df[nm])):
                raise ValueError(
                    f"All '{nm}s' in `{nm}` specified must be in `df`. "
                    f"Missing {nm}s: {set(lx) - set(df[nm])}."
                )

        tks: List[str] = [f"{c}_{x}" for c in cids for x in xcats]
        if not len(tks) == 2:
            raise ValueError(
                "The combination of `cids` & `xcats` must yield two tickers",
                f"Found {len(tks)} tickers: {tks}, ",
                f"from `cids` {cids} and `xcats` {xcats}",
            )
        if not set(tks).issubset(set(found_tickers)):
            raise ValueError(
                "All combinations of `cids` & `xcats` (i.e. tickers) specified must "
                "be in `df`."
                f"Missing tickers: {set(tks) - set(found_tickers)}"
            )

    if not isinstance(freq, str):
        raise TypeError("`freq` must be a string")

    if not isinstance(agg, str):
        raise TypeError("`agg` must be a string")

    if add_constant not in [True, False] or not isinstance(add_constant, bool):
        raise TypeError("`add_constant` must be a boolean")

    return True


def granger_causality_test(
    df: pd.DataFrame,
    tickers: Optional[List[str]] = None,
    cids: Optional[Union[str, List[str]]] = None,
    xcats: Optional[Union[str, List[str]]] = None,
    max_lag: Union[int, List[int]] = 4,
    add_constant: bool = False,
    freq: str = "M",
    agg: str = "mean",
    start: Optional[str] = None,
    end: Optional[str] = None,
    metric: str = "value",
) -> Dict[Any, Any]:
    """
    Run Granger Causality Test on a standardized quantamental dataframe.
    Since the Graner Causality Test is a pairwise test, only two tickers can be specified.
    Specify either `tickers` or `cids` & `xcats`. When specifying `cids` & `xcats`, the
    user may input one `cid` and two `xcats`; or two `cids` and one `xcat` to yield two
    tickers. The function forms the list of tickers from the `cids` and `xcats` such that
    the order of the formed tickers is preserved. The order of the tickers is important
    as the first ticker is the one that is tested to Granger cause the second ticker.
    The function tests whether the first ticker Granger causes the second ticker.

    :param <pd.DataFrame> df: A standardized quantamental dataframe.
    :param <List[str]> tickers: A list of tickers to run the test on. A maximum of two
        tickers can be specified.
    :param <Union[str, List[str]]> cids: One or two cids to run the test on. If two cids
        are specified, then only one xcat can be specified. If one cid is specified, then
        two xcats can be specified.
    :param <Union[str, List[str]]> xcats: One or two xcats to run the test on. If two
        xcats are specified, then only one cid can be specified. If one xcat is specified,
        then two cids can be specified.
    :param <Union[int, List[int]]> max_lag: If `max_lag` is an integer, then the function
        computes the test for all lags up to `max_lag`. If `max_lag` is a list of
        integers, then the function computes the test only for lags specified in the list.
    :param <bool> add_constant: Whether to add a constant to the regression.
    :param <str> freq: The frequency to downsample the data to. Must be one of "D", "W",
        "M", "Q", "A". Default is "M".
    :param <str> agg: The aggregation method to use when downsampling the data. Must be
        one of "mean" (default), "median", "min", "max", "first" or "last".
    :param <str> start: The start date of the data. Must be a valid ISO date. If not
        specified, the earliest date in `df` is used.
    :param <str> end: The end date of the data. Must be a valid ISO date. If not
        specified, the latest date in `df` is used.
    :param <str> metric: The metric to run the test on. Must be a column in `df`.
        Default is "value".

    :return <Dict[Any, Any]>: A dictionary containing the results of the Granger Causality
        Test. The keys are the lags and the values are the results of the test.
    :raises <TypeError>: If any of the inputs are of the wrong type.
    :raises <ValueError>: If any of the input values are invalid.
    """
    ## Check inputs

    _type_checks(
        df=df,
        tickers=tickers,
        cids=cids,
        xcats=xcats,
        max_lag=max_lag,
        add_constant=add_constant,
        start=start,
        end=end,
        freq=freq,
        agg=agg,
        metric=metric,
    )
    ## value checks for `freq` and `agg` are implicitly checked in downstream functions

    ## Copy df to prevent side effects
    df: QuantamentalDataFrame = df.copy()

    ## Construct tickers from the `cids` and `xcats` if `tickers` is not specified
    tickers: List[str] = _get_tickers(tickers=tickers, cids=cids, xcats=xcats)

    ## Reduce df
    df: pd.DataFrame = reduce_df_by_ticker(df=df, ticks=tickers, start=start, end=end)

    # Downsample df
    freq = freq.upper()
    agg = agg.lower()
    df = downsample_df_on_real_date(
        df=df, groupby_columns=["cid", "xcat"], freq=freq, agg=agg
    )

    # Pivot df
    df_wide: pd.DataFrame = qdf_to_ticker_df(df=df, value_column=metric)

    # there must only be two columns in df_wide
    assert len(df_wide.columns) == 2, "df_wide must have only two columns"

    logger.info(
        "Running Granger Causality Test: Testing whether %s Granger causes %s",
        df_wide.columns[0],
        df_wide.columns[1],
    )
    # NOTE: Since no NANs are allowed in the input data, we must drop them here
    # This may yield unexpected/unreliable results for tickers with large periods of
    # missing data

    # drop any rows with NANs
    df_wide = df_wide.dropna(how="any", axis=0)
    if df_wide.empty:
        raise ValueError(
            "The input data contains only NANs. "
            "Please check the input data for missing values or "
            "consider using a different downsampling frequency/date range."
        )

    gct: Dict[Any, Any] = _granger_causality_backend(
        data=df_wide,
        max_lag=max_lag,
    )

    return gct


if __name__ == "__main__":
    cids: List[str] = ["AUD"]
    xcats: List[str] = ["FX", "EQ"]

    df: pd.DataFrame = make_test_df(
        cids=cids,
        xcats=xcats,
    )

    gct: Dict[Any, Any] = granger_causality_test(
        df=df,
        cids=cids,
        xcats=xcats,
    )

    cids: List[str] = ["AUD", "CAD"]
    xcats: str = "FX"
    # tickers =  AUD_FX, CAD_FX
    df: pd.DataFrame = make_test_df(
        cids=cids,
        xcats=xcats,
    )

    gct: Dict[Any, Any] = granger_causality_test(
        df=df,
        tickers=["AUD_FX", "CAD_FX"],
    )

    print(gct)
