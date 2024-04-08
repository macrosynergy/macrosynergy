"""Estimation of Historic Portfolio Volatility.

Module for calculating the historic portfolio volatility for a given strategy.

TODO Proposed workflow:
1. Create list of trigger dates for rebalance frequency.
2. Create batches of frequencies for which to estimate volatility per trigger date (memory inefficient).
3. Estimate variance-covariance matrix for each batch (per trigger date) and per frequency (annualised).
4. Weight the variance-covariance matrices by the weights for each frequency into single matrix.
5. Aggregate the variance-covariance matrix into a single portfolio volatility estimate.

"""

import logging
import warnings

import functools
from typing import Dict, List, Optional
from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Generator,
    Any,
    Union,
)

import numpy as np
import pandas as pd
import seaborn as sns, matplotlib.pyplot as plt
from macrosynergy.panel.historic_vol import expo_std, flat_std, expo_weights
from macrosynergy.management.types import NoneType, QuantamentalDataFrame
from macrosynergy.management.constants import FFILL_LIMITS, ANNUALIZATION_FACTORS
from macrosynergy.management.utils import (
    _map_to_business_day_frequency,
    get_eops,
    is_valid_iso_date,
    reduce_df,
    standardise_dataframe,
    ticker_df_to_qdf,
    get_eops,
    get_cid,
    _map_to_business_day_frequency,
)
from macrosynergy.panel.historic_vol import expo_weights

RETURN_SERIES_XCAT = "_PNL_USD1S_ASD"


logger = logging.getLogger(__name__)

cache = functools.lru_cache(maxsize=None)


@cache
def flat_weights_arr(lback_periods: int, *args, **kwargs) -> np.ndarray:
    """Flat weights for the look-back period."""
    return np.ones(lback_periods) / lback_periods


@cache
def expo_weights_arr(lback_periods: int, half_life: int, *args, **kwargs) -> np.ndarray:
    """Exponential weights for the lookback period."""
    return expo_weights(lback_periods=lback_periods, half_life=half_life)


def _weighted_covariance(
    x: np.ndarray,
    y: np.ndarray,
    weights_func: Callable[[int, int], np.ndarray],
    lback_periods: int,
    half_life: int,
) -> float:
    """
    Estimate covariance between two series after applying weights.

    """
    assert half_life > 0, "half_life must be greater than 0"
    assert lback_periods > 0 or lback_periods == -1, "lback_periods must be >0"
    assert x.ndim == 1 or x.shape[1] == 1, "`x` must be a 1D array or a column vector"
    assert y.ndim == 1 or y.shape[1] == 1, "`y` must be a 1D array or a column vector"
    assert x.shape[0] == y.shape[0], "`x` and `y` must have same length"

    # if either of x or y is all NaNs, return NaN
    if np.isnan(x).all() or np.isnan(y).all():
        return np.nan

    wmask = np.isnan(x) | np.isnan(y)
    weightslen = min(sum(~wmask), lback_periods if lback_periods > 0 else len(x))

    # drop NaNs and only consider the most recent lback_periods
    x, y = x[~wmask][-weightslen:], y[~wmask][-weightslen:]

    if len(x) < weightslen:
        return np.nan

    assert x.shape[0] == weightslen
    w: np.ndarray = weights_func(lback_periods=weightslen, half_life=half_life)
    x_wsum, y_wsum = (w * x).sum(), (w * y).sum()
    rss = (x - x_wsum) * (y - y_wsum)

    return w.T.dot(rss)


def estimate_variance_covariance(
    piv_ret: pd.DataFrame,
    remove_zeros: bool,
    weights_func: Callable[[int, int], np.ndarray],
    lback_periods: int,
    half_life: int,
) -> pd.DataFrame:
    """Estimation of the variance-covariance matrix needs to have the following configuration options

    1. Absolutely vs squared deviations,
    2. Flat weights (equal) vs. exponential weights,
    3. Frequency of estimation (daily, weekly, monthly, quarterly) and their weights.
    """

    cov_mat = np.zeros((len(piv_ret.columns), len(piv_ret.columns)))
    logger.info(f"Estimating variance-covariance matrix for {piv_ret.columns}")

    if remove_zeros:
        piv_ret = piv_ret.replace(0, np.nan)

    for i_b, c_b in enumerate(piv_ret.columns):
        for i_a, c_a in enumerate(piv_ret.columns[: i_b + 1]):
            logger.debug(f"Estimating covariance between {c_a} and {c_b}")
            est_vol = _weighted_covariance(
                x=piv_ret[c_a].values,
                y=piv_ret[c_b].values,
                weights_func=weights_func,
                lback_periods=lback_periods,
                half_life=half_life,
            )
            cov_mat[i_a, i_b] = cov_mat[i_b, i_a] = est_vol

    assert np.all((cov_mat.T == cov_mat) ^ np.isnan(cov_mat))

    return pd.DataFrame(cov_mat, index=piv_ret.columns, columns=piv_ret.columns)


def _downsample_returns(
    piv_df: pd.DataFrame,
    freq: str = "m",
) -> pd.DataFrame:
    # TODO create as a general convert_frequency function
    # TODO current aggregator is `art` (check definition of name in R code)
    # TODO test [1] input data is daily and [2] daily gives daily output

    freq = _map_to_business_day_frequency(freq)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        piv_new_freq: pd.DataFrame = (
            (1 + piv_df / 100).resample(freq).prod() - 1
        ) * 100
        warnings.resetwarnings()
    return piv_new_freq


def _calculate_portfolio_volatility(
    pivot_returns: pd.DataFrame,
    pivot_signals: pd.DataFrame,
    rebal_freq: str,
    est_freqs: List[str],
    est_weights: List[float],
    weights_func: Callable[[int, int], np.ndarray],
    lback_periods: int,
    half_life: int,
    nan_tolerance: float,
    remove_zeros: bool,
    portfolio_return_name: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    logger.info(
        f"Calculating portfolio volatility "
        f"for FIDS={pivot_returns.columns.tolist()} "
        f"from {min(pivot_returns.index.min(), pivot_signals.index.min())} "
        f"to {max(pivot_returns.index.max(), pivot_signals.index.max())}, with "
        f"lback_periods={lback_periods}, nan_tolerance={nan_tolerance}, "
        f"remove_zeros={remove_zeros}, rebal_freq={rebal_freq}, est_freqs={est_freqs}, "
        f"est_weights={est_weights} "
    )

    rebal_dates = get_eops(dates=pivot_signals.index, freq=rebal_freq)

    signals = pivot_signals.loc[rebal_dates, :]

    # Returns batches
    logger.info(
        "Rebalance portfolio from %s to %s (%s times)",
        rebal_dates.min(),
        rebal_dates.max(),
        rebal_dates.shape[0],
    )

    td = rebal_dates.iloc[-1]

    lback_max = (
        int(np.ceil(lback_periods * (1 + nan_tolerance))) if lback_periods > 0 else 0
    )

    # TODO convert frequencies
    list_vcv: List[pd.DataFrame] = []
    list_pvol: List[Tuple[pd.Timestamp, np.float64]] = []
    for td in rebal_dates:
        window_df = pivot_returns.loc[pivot_returns.index <= td].iloc[-lback_max:]
        dict_vcv: Dict[str, pd.DataFrame] = {
            freq: estimate_variance_covariance(
                piv_ret=_downsample_returns(window_df, freq=freq),
                lback_periods=lback_periods,
                remove_zeros=remove_zeros,
                weights_func=weights_func,
                half_life=half_life,
            )
            for ix, freq in enumerate(est_freqs)
        }

        vcv_df: pd.DataFrame = sum(
            [
                est_weights[ix] * ANNUALIZATION_FACTORS[freq] * dict_vcv[freq]
                for ix, freq in enumerate(est_freqs)
            ]
        )
        # reshape and drop duplicates in vcv

        list_vcv.append(
            vcv_df.rename_axis("fid1", axis=0)
            .rename_axis("fid2", axis=1)
            .stack()
            .to_frame("value")
            .reset_index()
            .assign(real_date=td)
        )

        pvol: float = np.sqrt(signals.loc[td, :].T.dot(vcv_df).dot(signals.loc[td, :]))
        list_pvol.append((td, pvol))

    pvol = pd.DataFrame(
        list_pvol,
        columns=["real_date", portfolio_return_name],
    ).set_index("real_date")

    vcv_df = pd.concat(list_vcv, axis=0)  # add to cls.vcv

    vcv_df["helper"] = vcv_df[["fid1", "fid2", "real_date"]].apply(
        func=(lambda x: "-".join(sorted([x["fid1"], x["fid2"]])) + str(x["real_date"])),
        axis=1,
    )
    vcv_df = (
        vcv_df.drop_duplicates(subset=["helper"])
        .drop(columns=["helper"])
        .reset_index(drop=True)
    )

    return pvol, vcv_df


def _hist_vol(
    pivot_signals: pd.DataFrame,
    pivot_returns: pd.DataFrame,
    sname: str,
    rebal_freq: str = "M",
    lback_meth: str = "ma",
    lback_periods: int = 21,
    est_freqs: List[str] = ["D", "W", "M"],
    est_weights: List[float] = [1, 2, 3],
    half_life=11,
    nan_tolerance: float = 0.25,
    remove_zeros: bool = True,
    return_variance_covariance: bool = False,
) -> List[pd.DataFrame]:
    """
    Calculates historic volatility for a given strategy. It assumes that the dataframe
    is composed solely of the relevant signals and returns for the strategy.

    :param <pd.DataFrame> pivot_signals: the pivot table of the contract signals.
    :param <pd.DataFrame> pivot_returns: the pivot table of the contract returns.
    :param <str> rebal_freq: the frequency of the volatility estimation. Default is 'm'
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

    # TODO get the correct rebalance dates
    weights_func = flat_weights_arr if lback_meth == "ma" else expo_weights_arr
    logger.info(
        "Found lback_meth=%s, using weights_func=%s", lback_meth, weights_func.__name__
    )
    portfolio_return_name = f"{sname}{RETURN_SERIES_XCAT}"

    pvol_df: pd.DataFrame
    vcv_df: pd.DataFrame
    pvol_df, vcv_df = _calculate_portfolio_volatility(
        pivot_returns=pivot_returns,
        pivot_signals=pivot_signals,
        rebal_freq=rebal_freq,
        weights_func=weights_func,
        portfolio_return_name=portfolio_return_name,
        lback_periods=lback_periods,
        remove_zeros=remove_zeros,
        nan_tolerance=nan_tolerance,
        half_life=half_life,
        est_freqs=est_freqs,
        est_weights=est_weights,
    )

    # assert portfolio_return_name the only column
    assert pvol_df.columns.tolist() == [portfolio_return_name]

    # # TODO remove re-index!
    df_out = pvol_df.reindex(pivot_returns.index).ffill(limit=FFILL_LIMITS[rebal_freq])
    nanindex = df_out.index[df_out[portfolio_return_name].isnull()]
    if len(nanindex) > 0:
        df_out = df_out.dropna()
        logger.debug(
            f"Found {len(nanindex)} NaNs in {portfolio_return_name} at {nanindex}, dropping all NaNs."
        )

    # TODO check df_out is of frequency rebal_freq
    # TODO - should below not be forward filled with the previous volatility value...
    nan_df = df_out[
        df_out.loc[
            df_out.first_valid_index() : df_out.last_valid_index(),
            portfolio_return_name,
        ].isnull()
    ]

    if not nan_df.empty:
        logger.warning(
            f"Found NaNs in {portfolio_return_name} at: {nan_df.index.tolist()}, dropping all NaNs."
        )
        df_out = df_out.dropna()

    if return_variance_covariance:
        return [df_out, vcv_df]
    return [df_out]


def unstack_covariances(
    vcv_df: pd.DataFrame,
    fillna: bool = False,
) -> Dict[str, pd.DataFrame]:
    """Unstack the covariance matrix DataFrame."""
    # return vcv_df.pivot(index="real_date", columns=["fid1", "fid2"], values="value")

    vcvs: Dict[str, pd.DataFrame] = {}
    for dt, df in vcv_df.groupby("real_date"):
        vcv = df.pivot(index="fid2", columns="fid1", values="value")
        if fillna:
            vcv = vcv.fillna(vcv.T)
            assert all(vcv == vcv.T)
        vcvs[pd.Timestamp(dt).strftime("%Y-%m-%d")] = vcv

    return vcvs


def _check_input_arguments(
    arguments: List[Tuple[Any, str, Union[type, Tuple[type, type]]]]
):
    # TODO move to general utils
    for varx, namex, typex in arguments:
        if not isinstance(varx, typex):
            raise ValueError(f"`{namex}` must be {typex}.")
        if typex in [str, list, dict] and len(varx) == 0:
            raise ValueError(f"`{namex}` must not be an empty {str(typex)}.")


def _check_frequency(freq: str, freq_type: str):
    # TODO move to general utils
    try:
        _map_to_business_day_frequency(freq)
    except ValueError as e:
        raise ValueError(
            f"`{freq_type:s}` ({freq:s}) must be a valid frequency string: {e}"
        )


def _check_missing_data(
    df: pd.DataFrame, sname: str, fids: List[str], rstring: str
) -> None:
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
            f"{contx}{rstring}"
            for contx in fids
            if f"{contx}{rstring}" not in u_tickers
        ]
        raise ValueError(
            f"The dataframe is missing the following return series: {missing_tickers}"
        )


def _check_weights(weights: List[float], freqs: List[str]) -> List[float]:
    if weights is None:
        weights = [1 / len(freqs) for _ in freqs]
    else:
        if len(weights) != len(freqs):
            raise ValueError(
                f"Length of `weights` ({len(weights)}) must be equal to length of `freqs` ({len(freqs)})"
            )

        if not all([isinstance(w, (int, float)) for w in weights]):
            raise ValueError("`weights` must be a list of floats or integers.")

        if not np.isclose(sum(weights), 1):
            warnings.warn("`weights` do not sum to 1. They will be normalized.")
            weights = [w / sum(weights) for w in weights]

    return weights


def historic_portfolio_vol(
    df: pd.DataFrame,
    sname: str,
    fids: List[str],
    rstring: str = "XR",
    rebal_freq: str = "m",
    lback_meth: str = "ma",
    lback_periods: List[int] = -1,  # default all
    half_life: List[int] = 11,  # TODO what to specify here for weekly and monthly?
    est_freqs: List[str] = ["D", "W", "M"],  # "m", "w", "d", "q"
    est_weights: Optional[List[float]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    blacklist: Optional[dict] = None,
    nan_tolerance: float = 0.25,
    remove_zeros: bool = True,
    return_variance_covariance: bool = False,
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
    :param <str> rebal_freq: the frequency of rebalancing and volatility estimation.
        Default is 'M' for monthly. Alternatives are 'W' for business weekly, 'D' for
        daily, and 'Q' for quarterly. Estimations are conducted for the end of the period.
        # TODO - case insensitive!
    :param <List[str]> est_freqs: the list of frequencies for which the volatility
        is estimated. Volatility for a given period is the weighted sum of the volatilities
        estimated for each frequency. Default is ["D", "W", "M"].
    :param <List[float]> est_weights: the list of weights for each frequency in
        `est_freqs`. Weights are normalized before applying. In cases where there may be
        missing data or NaNs in the result, the remaining weights are normalized. Default
        is None, which means that the weights are equal.
    :param <str> lback_meth: the method to use for the lookback period of the
        volatility-targeting method. Default is "ma" for moving average. Alternative is
        "xma", for exponential moving average.
    :param <int> lback_periods: the number of periods to use for the lookback period
        of the volatility-targeting method. Default is 21 for daily (TODO verify).
    :param <int> half_life: the half-life of the exponential moving average for the
        volatility-targeting method. This is disregarded when using `lback_meth="ma"`.
        Default is 11.
    :param <str> start: the start date of the data. Default is None, which means that
        the start date is taken from the dataframe.
    :param <str> end: the end date of the data. Default is None, which means that
        the end date is taken from the dataframe.
    :param <dict> blacklist: a dictionary of contract identifiers to exclude from
        the calculation. Default is None, which means that no contracts are excluded.
    :param <float> nan_tolerance: maximum ratio of number of NaN values to the total
        number of values in a lookback window. If exceeded the resulting volatility is set
        to NaN, else prior non-zero values are added to the window instead. Default is 0.25.
    :param <bool> remove_zeros: if True (default) any returns that are exact zeros will
        not be included in the lookback window and prior non-zero values are added to the
        window instead.

    :return <pd.DataFrame>: JPMaQS dataframe of annualized standard deviation of
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
    # TODO create function for this? Also, do we want to create the set of failures (not just first one)?
    _check_input_arguments(
        arguments=[
            (df, "df", pd.DataFrame),
            (sname, "sname", str),
            (fids, "fids", list),
            (rebal_freq, "rebal_freq", str),
            (lback_periods, "lback_periods", int),
            (lback_meth, "lback_meth", str),
            (half_life, "half_life", int),
            (rstring, "rstring", str),
            (start, "start", (str, NoneType)),
            (end, "end", (str, NoneType)),
            (blacklist, "blacklist", (dict, NoneType)),
        ]
    )

    # Check the frequency arguments
    _check_frequency(freq=rebal_freq, freq_type="rebal_freq")

    for ix, freq in enumerate(est_freqs):
        _check_frequency(freq=freq, freq_type=f"est_freq[{ix:d}]")

    ## Check estimation frequency weights
    est_weights: List[float] = _check_weights(weights=est_weights, freqs=est_freqs)

    ## Standardize and copy DF
    df: pd.DataFrame = standardise_dataframe(df.copy())
    rebal_freq = _map_to_business_day_frequency(rebal_freq)

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
    df["ticker"] = df["cid"] + "_" + df["xcat"]
    u_tickers: List[str] = list(df["ticker"].unique())

    ## Check for missing data
    _check_missing_data(df=df, sname=sname, fids=fids, rstring=rstring)

    # Add financial identifier (fid) to DataFrame
    df["fid"] = (
        df["cid"]
        + "_"
        + df["xcat"]
        .str.split("_")
        .map(
            lambda x: (
                x[0][: -len(rstring.split("_")[0])]
                if x[0].endswith(rstring.split("_")[0])
                else x[0]
            )
        )
    )

    ## Filter out data-frame and select contract signals (CSIG) and returns (XR)
    filt_csigs: List[str] = [tx for tx in u_tickers if tx.endswith(f"_CSIG_{sname}")]
    filt_xrs: List[str] = [tx for tx in u_tickers if tx.endswith(rstring)]

    # TODO check if all exists

    pivot_signals: pd.DataFrame = df.loc[df["ticker"].isin(filt_csigs)].pivot(
        index="real_date", columns="fid", values="value"
    )

    pivot_returns: pd.DataFrame = df.loc[df["ticker"].isin(filt_xrs)].pivot(
        index="real_date", columns="fid", values="value"
    )
    assert set(pivot_signals.columns) == set(pivot_returns.columns)

    # assert

    result: List[pd.DataFrame] = _hist_vol(
        pivot_returns=pivot_returns,
        pivot_signals=pivot_signals,
        sname=sname,
        rebal_freq=rebal_freq,
        est_freqs=est_freqs,
        est_weights=est_weights,
        lback_periods=lback_periods,
        lback_meth=lback_meth,
        half_life=half_life,
        nan_tolerance=nan_tolerance,
        remove_zeros=remove_zeros,
        return_variance_covariance=return_variance_covariance,
    )

    assert len(result) == 1 + int(return_variance_covariance)

    result[0] = ticker_df_to_qdf(df=result[0])
    if return_variance_covariance:
        return result[0], result[1]
    return result[0]


if __name__ == "__main__":
    from macrosynergy.management.simulate import simulate_returns_and_signals

    np.random.seed(42)  # Fix numpy seed to 42 for reproducibility

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

    df_copy = df.copy()  # TODO why copy?

    N_p_nans = 0.01
    df["value"] = df["value"].apply(
        lambda x: x if np.random.rand() > N_p_nans else np.nan
    )

    df_vol, vcv_df = historic_portfolio_vol(
        df=df,
        sname="STRAT",
        fids=fids,
        rebal_freq="m",
        lback_periods=-1,
        lback_meth="ma",
        half_life=11,
        rstring="XR",
        start=start,
        end=end,
        return_variance_covariance=True,
    )

    vcvs_dict = unstack_covariances(vcv_df)
    dates = [
        dt.strftime("%Y-%m-%d")
        for dt in sorted(pd.to_datetime(list(vcvs_dict.keys())))[-9:]
    ]
    with sns.axes_style("whitegrid"):
        fig, ax = plt.subplots(3, 3, figsize=(15, 15))
        for ix, dt in enumerate(dates):
            sns.heatmap(vcvs_dict[dt], ax=ax[ix // 3, ix % 3])
            ax[ix // 3, ix % 3].set_title(dt)
        plt.tight_layout()
        plt.show()

    df_copy_vol: pd.DataFrame = historic_portfolio_vol(
        df=df_copy,
        sname="STRAT",
        fids=fids,
        rebal_freq="m",
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
