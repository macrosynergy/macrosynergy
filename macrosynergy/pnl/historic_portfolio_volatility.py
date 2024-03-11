"""
Module for calculating the historic portfolio volatility for a given strategy.
"""

from typing import List, Optional, Tuple, Callable
from functools import lru_cache
import pandas as pd
import numpy as np

from macrosynergy.panel.historic_vol import expo_weights
from macrosynergy.management.types import NoneType
from macrosynergy.management.utils import (
    reduce_df,
    standardise_dataframe,
    is_valid_iso_date,
    ticker_df_to_qdf,
    get_eops,
)
from macrosynergy.management.types import QuantamentalDataFrame

RETURN_SERIES_XCAT = "_PNL_USD1S_ASD"

cache = lru_cache(maxsize=None)


@cache
def flat_weights_arr(lback_periods: int, *args, **kwargs) -> np.ndarray:
    """Flat weights for the lookback period."""
    return np.ones(lback_periods) / lback_periods


@cache
def expo_weights_arr(lback_periods: int, half_life: int, *args, **kwargs) -> np.ndarray:
    """Exponential weights for the lookback period."""
    return expo_weights(lback_periods=lback_periods, half_life=half_life)


def _weighted_covariance(x: np.ndarray, y: np.ndarray, w: np.ndarray = None):
    """
    Estimate covariance between two series after applying weights.

    """
    assert x.ndim == 1 or x.shape[1] == 1, "`x` must be a 1D array or a column vector"
    assert y.ndim == 1 or y.shape[1] == 1, "`y` must be a 1D array or a column vector"
    assert x.shape[0] == y.shape[0], "`x` and `y` must have same length"
    assert w.ndim == 1 or w.shape[1] == 1, "`w` must be a 1D array or a column vector"
    assert x.shape[0] == w.shape[0], "`x` and `w` must have same length"

    w = w / sum(w)  # weights are normalized

    rss = (x - x.mean()) * (y - y.mean())

    return w.T.dot(rss)


def _nan_ratio(x, remove_zeros: bool = True) -> float:
    return (sum(np.isnan(x)) + (sum(x == 0) if remove_zeros else 0)) / len(x)


def _mask_nan_series(
    x: np.ndarray,
    y: np.ndarray,
    lback_periods: int,
    remove_zeros: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Mask NaNs in the series."""
    # if either of the dates is NaN, mask that row
    mask = np.isnan(x) | np.isnan(y)
    if remove_zeros:
        mask = mask | (x == 0) | (y == 0)

    # drop NaNs
    x = x[~mask]
    y = y[~mask]

    # if dropping nans leaves less than lback_periods, fill with NaNs and return
    if any(len(_) < lback_periods for _ in [x, y]):
        _r = np.full(lback_periods, np.nan)
        return (_r, _r)

    return (x[-lback_periods:], y[-lback_periods:])


def _estimate_variance_covariance(
    piv_ret: pd.DataFrame,
    remove_zeros: bool,
    weights_func: Callable[[int, int], np.ndarray],
    lback_periods: int,
    half_life: int,
    *args,
    **kwargs,
) -> pd.DataFrame:
    """Estimation of the variance-covariance matrix needs to have the following configuration options

    1. Absolutely vs squared deviations,
    2. Flat weights (equal) vs. exponential weights,
    3. Frequency of estimation (daily, weekly, monthly, quarterly) and their weights.
    """
    # weights is a function at this point; either flat_weights_arr or expo_weights_arr

    cov_matr = np.zeros((len(piv_ret.columns), len(piv_ret.columns)))

    for i_b, c_b in enumerate(piv_ret.columns):
        for i_a, c_a in enumerate(piv_ret.columns[: i_b + 1]):
            x, y = _mask_nan_series(
                x=piv_ret[c_a].values,
                y=piv_ret[c_b].values,
                lback_periods=lback_periods,
                remove_zeros=remove_zeros,
            )
            # if either of the return series is not available, set the covariance to NaN
            if np.isnan(x).all() or np.isnan(y).all():
                cov_matr[i_a, i_b] = np.nan
                cov_matr[i_b, i_a] = np.nan
                continue

            w = weights_func(
                lback_periods=min(lback_periods, len(x)),
                half_life=min(half_life, len(x)),
            )  # inherently fast, and cached anyway

            est_vol = _weighted_covariance(x=x, y=y, w=w)
            cov_matr[i_a, i_b] = est_vol
            if i_a != i_b:
                cov_matr[i_b, i_a] = est_vol

    assert np.all((cov_matr.T == cov_matr) ^ np.isnan(cov_matr))

    return pd.DataFrame(cov_matr, index=piv_ret.columns, columns=piv_ret.columns)


def _calculate_portfolio_volatility(
    trigger_indices: pd.DatetimeIndex,
    pivot_returns: pd.DataFrame,
    pivot_signals: pd.DataFrame,
    lback_periods: int,
    nan_tolerance: float,
    portfolio_return_name: str,
    *args,
    **kwargs,
    # weights_func: Optional[np.ndarray],
):
    """
    Calculate the portfolio volatility for each trigger date.
    Backed function for `_hist_vol`, to increase readability.

    :param <pd.DatetimeIndex> trigger_indices: the DateTimeIndex of the trigger dates.
    :param <pd.DataFrame> pivot_returns: the pivot table of the contract returns.
    :param <pd.DataFrame> pivot_signals: the pivot table of the contract signals.
    :param <int> lback_periods: the number of periods to use for the lookback period
        of the volatility-targeting method. Default is 21.
    :param <int> half_life: Refers to the half-time for "xma" and full lookback period
        for "ma". Default is 11.
    :param <callable> weights_func: the function to use for the weights array. Default
        is None, which means that the weights are equal.
    :param <float> nan_tolerance: maximum ratio of NaNs to non-NaNs in a lookback window,
        if exceeded the resulting volatility is set to NaN. Default is 0.25.
    :param <str> portfolio_return_name: the name of the portfolio return series.
    :param <dict> kwargs: additional keyword arguments to pass to the variance-covariance
        estimation function.
    """

    return_values = []

    for trigger_date in trigger_indices:
        td = trigger_date
        lbx = -1 * int(np.ceil(lback_periods * (1 + nan_tolerance)))
        # account for possible NA drops
        piv_ret = pivot_returns.loc[pivot_returns.index <= td].iloc[lbx:]
        vcv: pd.DataFrame = _estimate_variance_covariance(
            piv_ret=piv_ret,
            lback_periods=lback_periods,
            *args,
            **kwargs,
        )

        # TODO - NaN handing for signals?

        piv_sig: pd.DataFrame = pivot_signals.loc[td, :]
        return_values += [(td, piv_sig.T.dot(vcv).dot(piv_sig))]

    return pd.DataFrame(
        return_values,
        columns=["real_date", portfolio_return_name],
    ).set_index("real_date")


def _hist_vol(
    pivot_signals: pd.DataFrame,
    pivot_returns: pd.DataFrame,
    sname: str,
    est_freq: str = "m",
    lback_periods: int = 21,
    lback_meth: str = "ma",
    half_life=11,
    nan_tolerance: float = 0.25,
    remove_zeros: bool = True,
) -> pd.DataFrame:
    """
    Calculates historic volatility for a given strategy. It assumes that the dataframe
    is composed solely of the relevant signals and returns for the strategy.

    :param <pd.DataFrame> pivot_signals: the pivot table of the contract signals.
    :param <pd.DataFrame> pivot_returns: the pivot table of the contract returns.
    :param <str> est_freq: the frequency of the volatility estimation. Default is 'm'
        for monthly. Alternatives are 'w' for business weekly, 'd' for daily, and 'q'
        for quarterly. Estimations are conducted for the end of the period.
    :param <int> lback_periods: the number of periods to use for the lookback period
        of the volatility-targeting method. Default is 21.
    :param <str> lback_meth: the method to use for the lookback period of the
        volatility-targeting method. Default is 'ma' for moving average. Alternative is
        "xma", for exponential moving average.
    :param <int> half_life: Refers to the half-time for "xma" and full lookback period
        for "ma". Default is 11.
    :param <float> nan_tolerance: maximum ratio of NaNs to non-NaNs in a lookback window,
        if exceeded the resulting volatility is set to NaN. Default is 0.25.
    :param <bool> remove_zeros: removes zeroes as invalid entries and shortens the
        effective window.
    """

    lback_meth = lback_meth.lower()
    if lback_meth not in ["ma", "xma"]:
        raise NotImplementedError(
            f"`lback_meth` must be 'ma' or 'xma'; got {lback_meth}"
        )

    # NOTE: `get_eops` helps identify the dates for which the volatility calculation
    # will be performed. This is done by identifying the last date of each cycle.
    # We use `get_eops` primarily because it has a clear and defined behaviour for all frequencies.
    # It was originally designed as part of the `historic_vol` module, but it is
    # used here as well.

    trigger_indices = get_eops(
        dates=pivot_signals.index,
        freq=est_freq,
    )

    # TODO get the correct rebalance dates

    weights_func = flat_weights_arr if lback_meth == "ma" else expo_weights_arr
    portfolio_return_name = f"{sname}{RETURN_SERIES_XCAT}"

    df_out: pd.DataFrame = _calculate_portfolio_volatility(
        trigger_indices=trigger_indices,
        pivot_returns=pivot_returns,
        pivot_signals=pivot_signals,
        lback_periods=lback_periods,
        remove_zeros=remove_zeros,
        nan_tolerance=nan_tolerance,
        weights_func=weights_func,
        half_life=half_life,
        portfolio_return_name=portfolio_return_name,
    )

    # assert portfolio_return_name the only column
    assert df_out.columns == [portfolio_return_name]

    # Annualised standard deviation (ASD)
    df_out[portfolio_return_name] = np.sqrt(df_out[portfolio_return_name] * 252)

    ffills = {"d": 1, "w": 5, "m": 24, "q": 64}
    df_out = df_out.reindex(pivot_returns.index).ffill(limit=ffills[est_freq])
    # TODO - should below not be forward filled with the previous volatility value...
    assert (
        not df_out.loc[
            df_out.first_valid_index() : df_out.last_valid_index(),
            portfolio_return_name,
        ]
        .isnull()
        .any()
    )
    # TODO or log warning if there are NaNs
    df_out.dropna(inplace=True)

    return df_out


def historic_portfolio_vol(
    df: pd.DataFrame,
    sname: str,
    fids: List[str],
    rstring: str = "XR",
    est_freq: str = "m",
    lback_periods: int = 21,
    lback_meth: str = "ma",
    half_life: int = 11,
    start: Optional[str] = None,
    end: Optional[str] = None,
    blacklist: Optional[dict] = None,
    nan_tolerance: float = 0.25,
    remove_zeros: bool = True,
) -> QuantamentalDataFrame:
    """Historical portfolio volatility.

    Estimates annualized standard deviations of a portfolio, based on historic
    variances and co-variances.

    :param <QuantamentalDataFrame> df: JPMaQS standard DataFrame containing contract-specific signals and return series.
    :param <str> sname: the name of the strategy. It must correspond to contract
        signals in the dataframe, which have the format "<cid>_<ctype>_CSIG_<sname>", and
        which are typically calculated by the function contract_signals().
    :param <List[str]> fids: list of financial contract identifiers in the format
        "<cid>_<ctype>". It must correspond to contract signals in the dataframe.

    :param <str> rstring: a general string of the return category. This identifies
        the contract returns that are required for the volatility-targeting method, based
        on the category identifier format <cid>_<ctype><rstring> in accordance with
        JPMaQS conventions. Default is 'XR'.
    :param <str> est_freq: the frequency of the volatility estimation. Default is 'm'
        for monthly. Alternatives are 'w' for business weekly, 'd' for daily, and 'q'
        for quarterly. Estimations are conducted for the end of the period.
    :param <int> lback_periods: the number of periods to use for the lookback period
        of the volatility-targeting method. Default is 21 for daily (TODO verify).
    :param <str> lback_meth: the method to use for the lookback period of the
        volatility-targeting method. Default is 'ma' for moving average. Alternative is
        "xma", for exponential moving average.
    :param <str> start: the start date of the data. Default is None, which means that
        the start date is taken from the dataframe.
    :param <str> end: the end date of the data. Default is None, which means that
        the end date is taken from the dataframe.
    :param <dict> blacklist: a dictionary of contract identifiers to exclude from
        the calculation. Default is None, which means that no contracts are excluded.
    :param <float> nan_tolerance: maximum ratio of NaNs to non-NaNs in a lookback window,
        if exceeded the resulting volatility is set to NaN. Default is 0.25.
    :param <bool> remove_zeros: if True (default) any returns that are exact zeros will
        not be included in the lookback window and prior non-zero values are added to the
        window instead.


    :return: <pd.DataFrame> JPMaQS dataframe of annualized standard deviation of
        estimated strategy PnL, with category name <sname>_PNL_USD1S_ASD.
        TODO: check if this is correct.
        The values are in % annualized. Values between estimation points are forward
        filled.

    N.B.: If returns in the lookback window are not available the function will replace
    them with the average of the available returns of the same contract type. If no
    returns are available for a contract type the function will reduce the lookback window
    up to a minimum of 11 days. If no returns are available for a contract type for
    at least 11 days the function returns an NaN for that date and sends a warning of all
    the dates for which this happened.


    """
    ## Check inputs
    for varx, namex, typex in [
        (df, "df", pd.DataFrame),
        (sname, "sname", str),
        (fids, "fids", list),
        (est_freq, "est_freq", str),
        (lback_periods, "lback_periods", int),
        (lback_meth, "lback_meth", str),
        (half_life, "half_life", int),
        (rstring, "rstring", str),
        (start, "start", (str, NoneType)),
        (end, "end", (str, NoneType)),
        (blacklist, "blacklist", (dict, NoneType)),
    ]:
        if not isinstance(varx, typex):
            raise ValueError(f"`{namex}` must be {typex}.")
        if typex in [str, list, dict] and len(varx) == 0:
            raise ValueError(f"`{namex}` must not be an empty {str(typex)}.")

    ## Standardize and copy DF
    df: pd.DataFrame = standardise_dataframe(df.copy())

    ## Check the dates
    if start is None:
        start: str = pd.Timestamp(df["real_date"].min()).strftime("%Y-%m-%d")
    if end is None:
        end: str = pd.Timestamp(df["real_date"].max()).strftime("%Y-%m-%d")

    for dx, nx in [(start, "start"), (end, "end")]:
        if not is_valid_iso_date(dx):
            raise ValueError(f"`{nx}` must be a valid ISO-8601 date string")

    ## Reduce the dataframe
    df: pd.DataFrame = reduce_df(df=df, start=start, end=end, blacklist=blacklist)

    ## Check the strategy name
    if not isinstance(sname, str):
        raise ValueError("`sname` must be a string.")

    df["ticker"] = df["cid"] + "_" + df["xcat"]

    ## Check that there is atleast one contract signal for the strategy
    if not any(df["ticker"].str.endswith(f"_CSIG_{sname}")):
        raise ValueError(f"No contract signals for strategy `{sname}`.")

    u_tickers: List[str] = list(df["ticker"].unique())
    for contx in fids:
        if not any(
            [tx.startswith(contx) and tx.endswith(f"_CSIG_{sname}") for tx in u_tickers]
        ):
            raise ValueError(f"Contract identifier `{contx}` not in dataframe.")

    if not all([f"{contx}{rstring}" in u_tickers for contx in fids]):
        missing_tickers = [
            f"{contx}_{rstring}"
            for contx in fids
            if f"{contx}_{rstring}" not in u_tickers
        ]
        raise ValueError(
            f"The dataframe is missing the following return series: {missing_tickers}"
        )

    ## Filter DF and select CSIGs and XR
    filt_csigs: List[str] = [tx for tx in u_tickers if tx.endswith(f"_CSIG_{sname}")]

    filt_xrs: List[str] = [tx for tx in u_tickers if tx.endswith(rstring)]

    df["fid"] = (
        df["cid"]
        + "_"
        + df["xcat"]
        .str.split("_")
        .map(lambda x: x[0][:-2] if x[0].endswith("XR") else x[0])
    )
    pivot_signals: pd.DataFrame = df.loc[df["ticker"].isin(filt_csigs)].pivot(
        index="real_date", columns="fid", values="value"
    )
    pivot_returns: pd.DataFrame = df.loc[df["ticker"].isin(filt_xrs)].pivot(
        index="real_date", columns="fid", values="value"
    )
    assert set(pivot_signals.columns) == set(pivot_returns.columns)

    hist_port_vol: pd.DataFrame = _hist_vol(
        pivot_returns=pivot_returns,
        pivot_signals=pivot_signals,
        sname=sname,
        est_freq=est_freq,
        lback_periods=lback_periods,
        lback_meth=lback_meth,
        half_life=half_life,
        nan_tolerance=nan_tolerance,
        remove_zeros=remove_zeros,
    )

    return ticker_df_to_qdf(df=hist_port_vol).dropna().reset_index(drop=True)


if __name__ == "__main__":
    from macrosynergy.management.simulate import simulate_returns_and_signals
    from contract_signals import contract_signals

    # np seed 42
    np.random.seed(42)

    # Signals: FXCRY_NSA, EQCRY_NSA (rename to FX_CSIG_STRAT, EQ_CSIG_STRAT)
    # Returns: FXXR_NSA, EQXR_NSA (renamed to FXXR, EQXR)

    cids: List[str] = ["EUR", "GBP", "AUD", "CAD"]
    xcats: List[str] = ["EQ"]
    ctypes = xcats.copy()
    start: str = "2000-01-01"
    xr_tickers = [f"{cid}_{xcat}XR" for cid in cids for xcat in xcats]
    cs_tickers = [f"{cid}_{xcat}_CSIG_STRAT" for cid in cids for xcat in xcats]
    fids: List[str] = [f"{cid}_{ctype}" for cid in cids for ctype in ctypes]

    df = simulate_returns_and_signals(
        cids=cids,
        xcat=xcats[0],
        return_suffix="XR",
        signal_suffix="CSIG_STRAT",
        start=start,
        years=20,
    )
    # TODO simulate_returns_and_signals are risk-signals, not contract signals. We need to adjust for volatility and common (observed) factor.
    end = df["real_date"].max().strftime("%Y-%m-%d")

    df_copy = df.copy()

    N_p_nans = 0.10
    df["value"] = df["value"].apply(
        lambda x: x if np.random.rand() > N_p_nans else np.nan
    )

    df_vol: pd.DataFrame = historic_portfolio_vol(
        df=df,
        sname="STRAT",
        fids=fids,
        est_freq="m",
        lback_periods=15,
        lback_meth="ma",
        half_life=11,
        rstring="XR",
        start=start,
        end=end,
    )

    df_copy_vol: pd.DataFrame = historic_portfolio_vol(
        df=df_copy,
        sname="STRAT",
        fids=fids,
        est_freq="m",
        lback_periods=15,
        lback_meth="ma",
        half_life=11,
        rstring="XR",
        start=start,
        end=end,
    )

    expc_idx = pd.Timestamp("2019-04-26")
    assert df_copy_vol.max()["real_date"] == expc_idx  # np.seed is 42

    # print(df_copy_vol.head(10))
    # print(df_copy_vol.tail(10))
