"""
Estimation of Historic Portfolio Volatility.
"""

import logging

import functools
from typing import Dict, List, Optional
from typing import Callable, Tuple, Any, Union
from numbers import Number

import numpy as np
import pandas as pd
from macrosynergy.panel.historic_vol import expo_weights
from macrosynergy.management.types import NoneType, QuantamentalDataFrame
from macrosynergy.management.constants import ANNUALIZATION_FACTORS
from macrosynergy.management.utils import (
    _map_to_business_day_frequency,
    get_sops,
    is_valid_iso_date,
    reduce_df,
    # standardise_dataframe,
    # ticker_df_to_qdf,
)

RETURN_SERIES_XCAT = "_PNL_USD1S_ASD"
FREQ_TO_BDAY_MAP = {"B": 1, "W-FRI": 5, "BME": 21, "BQE": 63, "BA": 252}


def _bdays_per_period(freq: str) -> int:
    return FREQ_TO_BDAY_MAP[_map_to_business_day_frequency(freq)]


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
    min_obs: int = 1,
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
    if weightslen < min_obs:
        return np.nan

    # drop NaNs and only consider the most recent lback_periods
    x, y = x[~wmask][-weightslen:], y[~wmask][-weightslen:]

    if len(x) < weightslen or weightslen == 0:
        return np.nan

    assert x.shape[0] == weightslen
    w: np.ndarray = weights_func(lback_periods=weightslen, half_life=half_life)

    err_str = f"weights produced by {weights_func.__name__} do not sum to 1"
    assert np.isclose(w.sum(), 1), err_str

    x_mean, y_mean = (w * x).sum(), (w * y).sum()
    array_of_products = (x - x_mean) * (y - y_mean)

    return w.T.dot(array_of_products)


def estimate_variance_covariance(
    piv_ret: pd.DataFrame,
    remove_zeros: bool,
    weights_func: Callable[[int, int], np.ndarray],
    lback_periods: int,
    half_life: int,
    lback_min_obs: int = 1,
) -> pd.DataFrame:
    """
    Estimation of the variance-covariance matrix needs to have the following
    configuration options

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
                min_obs=lback_min_obs,
            )
            cov_mat[i_a, i_b] = cov_mat[i_b, i_a] = est_vol

    assert np.all((cov_mat.T == cov_mat) ^ np.isnan(cov_mat))

    return pd.DataFrame(cov_mat, index=piv_ret.columns, columns=piv_ret.columns)


def _downsample_returns(piv_df: pd.DataFrame, freq: str) -> pd.DataFrame:
    n = _bdays_per_period(freq)
    n_rows = piv_df.shape[0]
    piv_df = piv_df.sort_index()

    # compute buckets counting backwards from the most recent row
    bucket = np.arange(n_rows - 1, -1, -1) // n
    keep = bucket < n_rows // n  # drop the oldest partial block

    piv_df, bucket = piv_df[keep], bucket[keep]
    bucket = bucket[::-1] # oldest bucket is 0

    # compound data within each bucket
    compounded = (1 + piv_df / 100).groupby(bucket).prod(min_count=1)
    out = 100 * (compounded - 1)

    return out


def _nan_frac_since_first_obs(window_df: pd.DataFrame) -> pd.Series:
    """
    NaN fraction of each column, measured from its first observation in the window.
    """
    n_rows = window_df.shape[0]
    if n_rows == 0:
        return pd.Series(1.0, index=window_df.columns, dtype=float)

    isna = window_df.isna().to_numpy()
    first_obs = np.where(isna.all(axis=0), n_rows, isna.argmin(axis=0))
    live_len = n_rows - first_obs
    nan_after_first = isna.sum(axis=0) - first_obs

    frac = np.ones(window_df.shape[1], dtype=float)
    np.divide(nan_after_first, live_len, out=frac, where=live_len > 0)

    return pd.Series(frac, index=window_df.columns, dtype=float)


def _blend_frequency_vcvs(
    dict_vcv: Dict[str, pd.DataFrame],
    est_freqs: List[str],
    est_weights: List[float],
    column_order: pd.Index,
) -> pd.DataFrame:
    """
    Combine the per-frequency covariance estimates into one annualized matrix.
    Each entry is the `est_weights`-weighted mean of the annualized estimates actually
    available for it, with the surviving weights renormalized so a contract dropped
    at one estimation frequency still gets an estimate from the others
    """
    estimated = set().union(*(vcv.columns for vcv in dict_vcv.values()))
    fids = [fid for fid in column_order if fid in estimated]

    stacked = np.stack(
        [
            dict_vcv[freq].reindex(index=fids, columns=fids).to_numpy()
            for freq in est_freqs
        ]
    )
    weights = np.asarray(est_weights, dtype=float)[:, None, None]
    annualization = np.asarray(
        [ANNUALIZATION_FACTORS[freq] for freq in est_freqs], dtype=float
    )[:, None, None]

    available = ~np.isnan(stacked)
    weight_sum = np.where(available, weights, 0.0).sum(axis=0)
    weighted = np.where(available, weights * annualization * stacked, 0.0).sum(axis=0)

    blended = np.full_like(weight_sum, np.nan)
    np.divide(weighted, weight_sum, out=blended, where=weight_sum > 0)

    return pd.DataFrame(blended, index=fids, columns=fids)


def _calculate_multi_frequency_vcv_for_period(
    pivot_returns: pd.DataFrame,
    rebal_date: pd.Timestamp,
    est_freqs: List[str],
    est_weights: List[float],
    weights_func: Callable[[int, int], np.ndarray],
    lback_periods: List[int],
    half_life: List[int],
    nan_tolerance: float,
    remove_zeros: bool,
    lback_min_obs: List[int],
) -> pd.DataFrame:
    dict_vcv: Dict[str, pd.DataFrame] = {}

    for freq, lb, hl, min_obs in zip(
        est_freqs, lback_periods, half_life, lback_min_obs
    ):
        lb_bdays = lb * _bdays_per_period(freq)
        first_date = None if lb == -1 else rebal_date - pd.offsets.BDay(lb_bdays)
        window_df = pivot_returns.loc[first_date : rebal_date]

        # check for nan tolerance violations
        nan_frac = _nan_frac_since_first_obs(window_df)
        window_df = window_df[window_df.columns[nan_frac <= nan_tolerance]]

        if window_df.empty:
            return pd.DataFrame()

        # down sample returns and compute covariance matrix
        piv_ret = _downsample_returns(window_df, freq=freq)

        if piv_ret.shape[0] <= piv_ret.shape[1]:
            raise ValueError(
                f"{piv_ret.shape[1] + 1} data points are required to compute a "
                f"covariance matrix for {piv_ret.shape[1]} fids, but only found "
                f"{piv_ret.shape[0]}"
            )

        dict_vcv[freq] = estimate_variance_covariance(
            piv_ret=piv_ret,
            lback_periods=lb,
            remove_zeros=remove_zeros,
            weights_func=weights_func,
            half_life=hl,
            lback_min_obs=min_obs,
        )

    return _blend_frequency_vcvs(
        dict_vcv=dict_vcv,
        est_freqs=est_freqs,
        est_weights=est_weights,
        column_order=pivot_returns.columns,
    )


def _calc_vol_tuple(
    vcv_df: pd.DataFrame,
    signals: pd.DataFrame,
    date: pd.Timestamp,
    available_cids: List[str],
) -> Tuple[pd.Timestamp, float]:
    s = signals.loc[date, :].copy()

    s = s.loc[available_cids]
    vcv_df = vcv_df.loc[available_cids, available_cids]
    if not set(s.index) == set(vcv_df.columns):
        raise ValueError(
            "Signals and variance-covariance matrix do not have the same columns."
            f"\nSignals: {s.columns.tolist()}"
            f"\nVariance-Covariance: {vcv_df.columns.tolist()}"
        )

    idx_mask = s.isna() | (s.abs() < 1e-6)
    s.loc[idx_mask] = 0
    vcv_df.loc[idx_mask, :] = 0
    vcv_df.loc[:, idx_mask] = 0

    if vcv_df.isna().any().any():
        raise ValueError("N/A values in variance-covariance matrix")

    pvol: float = np.sqrt(s.T.dot(vcv_df).dot(s))
    return date, pvol


def stack_covariances(
    vcv_df: pd.DataFrame,
    real_date: pd.Timestamp,
) -> pd.DataFrame:
    """Stack the covariance matrix DataFrame."""
    return (
        vcv_df.rename_axis("fid1", axis=0)
        .rename_axis("fid2", axis=1)
        .stack()
        .to_frame("value")
        .reset_index()
        .assign(real_date=real_date)
    )


def _get_first_usable_date(
    pivot_returns: pd.DataFrame,
    rebal_dates: pd.Series,
    est_freqs: List[str],
    lback_periods: List[int],
) -> pd.Series:
    """
    Find the first rebalance date on which each contract has enough return data to be
    included in the covariance matrix calculation. If multiple est_freqs, then this is
    the first date at which enough return data exists so that a matrix can be estimated
    for all est_freqs.

    When one request a lookback of -1 (all data), the first date when we have twice as
    many data points as fids. For example, with 5 fids we would need 10 months/weeks/
    days depending on the est_freq.
    """
    n_fids = pivot_returns.shape[1]

    max_lback_days = max(
        2 * n_fids * _bdays_per_period(est_freq)
        if lback_period == -1 else
        lback_period * _bdays_per_period(est_freq)
        for est_freq, lback_period in zip(est_freqs, lback_periods)
    )

    first_valid_dates = {}
    for fid in pivot_returns.columns.tolist():
        first_date = pivot_returns[fid].first_valid_index()
        first_date += pd.offsets.BDay(max_lback_days)

        first_rebal_date = rebal_dates[rebal_dates >= first_date].min()
        first_valid_dates[fid] = first_rebal_date

    return pd.Series(first_valid_dates, name="real_date")


def _calculate_portfolio_volatility(
    pivot_returns: pd.DataFrame,
    pivot_signals: pd.DataFrame,
    rebal_freq: str,
    est_freqs: List[str],
    est_weights: List[float],
    weights_func: Callable[[int, int], np.ndarray],
    lback_periods: List[int],
    half_life: List[int],
    nan_tolerance: float,
    remove_zeros: bool,
    lback_min_obs: List[int],
    portfolio_return_name: str,
    cov_freq: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # signals are paired with the covariance axes positionally, so they must have
    # the same order
    if set(pivot_signals.columns) != set(pivot_returns.columns):
        raise ValueError(
            "`pivot_signals` and `pivot_returns` must cover the same contracts; "
            f"signals only: {sorted(set(pivot_signals.columns) - set(pivot_returns.columns))}, "
            f"returns only: {sorted(set(pivot_returns.columns) - set(pivot_signals.columns))}"
        )
    pivot_signals = pivot_signals[pivot_returns.columns]

    logger.info(
        f"Calculating portfolio volatility "
        f"for FIDS={pivot_returns.columns.tolist()} "
        f"from {min(pivot_returns.index.min(), pivot_signals.index.min())} "
        f"to {max(pivot_returns.index.max(), pivot_signals.index.max())}, with "
        f"lback_periods={lback_periods}, nan_tolerance={nan_tolerance}, "
        f"remove_zeros={remove_zeros}, rebal_freq={rebal_freq}, est_freqs={est_freqs}, "
        f"est_weights={est_weights} "
    )

    # estimate covariance matrices according to the cov_freq schedule
    cov_estimation_dates = get_sops(pivot_returns.index, freq=cov_freq)
    cov_history: np.ndarray = _cov_matrix_history(
        pivot_returns=pivot_returns,
        estimation_dates=cov_estimation_dates,
        est_freqs=est_freqs,
        est_weights=est_weights,
        lback_periods=lback_periods,
        half_life=half_life,
        nan_tolerance=nan_tolerance,
        remove_zeros=remove_zeros,
        weights_func=weights_func,
        lback_min_obs=lback_min_obs,
    )

    # determine the signals on rebalancing dates
    rebal_dates = get_sops(dates=pivot_signals.index, freq=rebal_freq)
    rebal_sigs = pivot_signals.loc[rebal_dates]

    # find the index of the most recent cov matrix for each rebal date; rebal dates
    # that precede the first estimate have no covariance available
    cov_idx = np.searchsorted(cov_estimation_dates, rebal_dates, side="right") - 1
    estimated = cov_idx >= 0
    cov_at_rebal = np.full(
        (len(rebal_dates), *cov_history.shape[1:]), fill_value=np.nan
    )
    cov_at_rebal[estimated] = cov_history[cov_idx[estimated]]

    # replace nans with 0s in cov and signals to allow for vectorized computation of vol
    cov_zero = np.nan_to_num(cov_at_rebal, nan=0.0)
    rebal_sigs_zero = np.nan_to_num(rebal_sigs.values, nan=0.0)

    # calculate volatility in a vectorised manner
    pvar = np.einsum("ni,nij,nj->n",rebal_sigs_zero, cov_zero, rebal_sigs_zero)
    pvol = np.sqrt(pvar)

    # No estimate at all means there is no measurable portfolio on that date
    pvol[np.isnan(cov_at_rebal).all(axis=(1, 2))] = np.nan

    # create dataframes
    pvol = pd.DataFrame(
        {"real_date": rebal_sigs.index, portfolio_return_name: pvol}
    ).set_index("real_date")

    i, j = np.triu_indices(pivot_returns.shape[1])
    fids = pivot_returns.columns.to_numpy()
    vcv_df_long = pd.DataFrame({
        "fid1": np.tile(fids[i], rebal_dates.shape[0]),
        "fid2": np.tile(fids[j], rebal_dates.shape[0]),
        "real_date": np.repeat(rebal_dates, len(i)),
        "value": cov_at_rebal[:, i, j].ravel(),
    }).dropna()

    return pvol, vcv_df_long


def _hist_vol(
    pivot_signals: pd.DataFrame,
    pivot_returns: pd.DataFrame,
    sname: str,
    rebal_freq: str,
    lback_meth: str,  # TODO allow for different method at different frequencies
    lback_periods: List[int],  # default all for all
    half_life,
    lback_min_obs: List[int],
    est_freqs: List[str],
    est_weights: List[float],
    nan_tolerance: float,
    remove_zeros: bool,
    return_variance_covariance: bool,
    cov_freq: Optional[str] = None,
) -> List[pd.DataFrame]:
    """
    Calculates historic volatility for a given strategy. It assumes that the dataframe
    is composed solely of the relevant signals and returns for the strategy.

    Parameters
    ----------
    pivot_signals : pd.DataFrame
        the pivot table of the contract signals.
    pivot_returns : pd.DataFrame
        the pivot table of the contract returns.
    rebal_freq : str
        the frequency of the volatility estimation. Default is 'm' for monthly.
        Alternatives are 'w' for business weekly, 'd' for daily, and 'q' for quarterly.
        Estimations are conducted for the end of the period.
    lback_periods : int
        the number of periods to use for the lookback period of the volatility-targeting
        method. Default is 21.
    lback_meth : str
        the method to use for the lookback period of the volatility-targeting method.
        Default is 'ma' for moving average. Alternative is "xma", for exponential moving
        average.
    half_life : int
        Refers to the half-time for "xma" and full lookback period for "ma". Default is
        11.
    lback_min_obs : List[int]
        minimum required observations in each lookback window. If fewer observations
        are available the variance-covariance estimate for that period is set to NaN.
    nan_tolerance : float
        maximum ratio of NaNs to non-NaNs in a lookback window, if exceeded the
        resulting volatility is set to NaN. Default is 0.25.
    remove_zeros : bool
        removes zeroes as invalid entries and shortens the effective window.
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
        lback_min_obs=lback_min_obs,
        est_freqs=est_freqs,
        est_weights=est_weights,
        cov_freq=cov_freq or rebal_freq, # default to cov matrix re-estimated every rebal date
    )

    # assert portfolio_return_name the only column
    pvol_df = pvol_df.reset_index()
    assert set(pvol_df.columns.tolist()) == set([portfolio_return_name, "real_date"])

    nan_dates = pvol_df[pvol_df[portfolio_return_name].isna()]["real_date"].copy()
    if len(nan_dates) > 0:
        logger.warning(
            f"Found NaNs in {portfolio_return_name} at: {nan_dates.tolist()}, dropping all NaNs."
        )
        pvol_df = pvol_df[~pvol_df["real_date"].isin(nan_dates)].copy()

    pvol_df = pvol_df.set_index("real_date")

    if return_variance_covariance:
        return [pvol_df, vcv_df]
    return [pvol_df]


def unstack_covariances(
    vcv_df: pd.DataFrame,
    fillna: bool = False,
) -> Dict[str, pd.DataFrame]:
    """Unstack the covariance matrix DataFrame."""
    vcvs: Dict[str, pd.DataFrame] = {}
    for dt, df in vcv_df.groupby("real_date"):
        vcv = df.pivot(index="fid2", columns="fid1", values="value")
        if fillna:
            vcv = vcv.fillna(vcv.T)
            assert all(vcv == vcv.T)
        vcvs[pd.Timestamp(dt).strftime("%Y-%m-%d")] = vcv

    return vcvs


def _check_input_arguments(
    arguments: List[Tuple[Any, str, Union[type, Tuple[type, type]]]],
):
    # TODO move to general utils
    for varx, namex, typex in arguments:
        if not isinstance(varx, typex):
            raise TypeError(f"`{namex}` must be {typex}.")
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


def _check_est_args(
    est_freqs: List[str],
    est_weights: List[Number],
    lback_periods: List[int],
    half_life: List[int],
    lback_min_obs: List[int],
) -> Tuple[List[str], List[float], List[int], List[int], List[int]]:
    # Calculate the maximum length of the provided lists
    max_len = max(
        len(est_freqs),
        len(est_weights),
        len(lback_periods),
        len(half_life),
        len(lback_min_obs),
    )

    def expand_list(lst, name):
        if len(lst) == 1:
            return lst * max_len
        elif len(lst) != max_len:
            raise ValueError(
                "All lists must have length 1 or the same length as the longest "
                f"list ({max_len}). '{name}' has length {len(lst)}."
            )
        return lst

    # Expand lists to match the maximum length
    est_freqs = expand_list(est_freqs, "est_freqs")
    est_weights = expand_list(est_weights, "est_weights")
    lback_periods = expand_list(lback_periods, "lback_periods")
    half_life = expand_list(half_life, "half_life")
    lback_min_obs = expand_list(lback_min_obs, "lback_min_obs")

    inv_weights_msg = "Invalid weights in `est_weights` at index {ix:d}"
    inv_lback_msg = "Invalid lookback period in `lback_periods` at index {ix:d}: {lb:d}"
    inv_hl_msg = "Invalid half-life in `half_life` at index {ix:d}: {hl:d}"

    for ix, (freq, weight, lback, hl, min_obs) in enumerate(
        zip(est_freqs, est_weights, lback_periods, half_life, lback_min_obs)
    ):
        _check_frequency(freq=freq, freq_type=f"est_freq[{ix:d}]")

        if not isinstance(weight, Number) or weight < 0:
            raise ValueError(inv_weights_msg.format(ix=ix))

        # stated idiosyncratically to allow for -1
        if not isinstance(lback, int) or (lback < 0 and lback != -1):
            raise ValueError(inv_lback_msg.format(ix=ix, lb=lback))

        if not isinstance(hl, int) or hl < 0:
            raise ValueError(inv_hl_msg.format(ix=ix, hl=hl))
        if not isinstance(min_obs, int) or min_obs < 1:
            raise ValueError(
                f"Invalid minimum observations in `lback_min_obs` at index {ix:d}: {min_obs}"
            )

    # normalize est_weights
    if not np.isclose(np.sum(est_weights), 1):
        est_weights = list(np.array(est_weights) / np.sum(est_weights))

    return est_freqs, est_weights, lback_periods, half_life, lback_min_obs


def add_fid_column(df: QuantamentalDataFrame, rstring: str) -> QuantamentalDataFrame:
    """Add financial identifier (fid) to DataFrame."""
    df["fid"] = (
        df["cid"].astype(str)
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
    return df


def historic_portfolio_vol(
    df: pd.DataFrame,
    sname: str,
    fids: List[str],
    rstring: str = "XR",
    rebal_freq: str = "m",
    lback_meth: str = "ma",
    est_freqs: Union[str, List[str]] = ["D", "W", "M"],  # "m", "w", "d", "q"
    est_weights: Union[Number, List[Number]] = [1, 1, 1],  # default equal weights
    lback_periods: Union[int, List[int]] = [-1, -1, -1],  # default all for all
    half_life: Union[int, List[int]] = [11, 5, 6],
    lback_min_obs: Union[int, List[int]] = 1,
    start: Optional[str] = None,
    end: Optional[str] = None,
    blacklist: Optional[dict] = None,
    nan_tolerance: float = 0.25,
    remove_zeros: bool = True,
    return_variance_covariance: bool = True,
) -> Union[QuantamentalDataFrame, Tuple[QuantamentalDataFrame, pd.DataFrame]]:
    """
    Historical portfolio volatility.  Estimates annualized standard deviations of a
    portfolio, based on historic variances and co-variances.

    Parameters
    ----------
    df : QuantamentalDataFrame
        JPMaQS standard DataFrame containing contract-specific signals and return
        series.
    sname : str
        the name of the strategy. It must correspond to contract signals in the
        dataframe, which have the format "<cid>_<ctype>_CSIG_<sname>", and which are
        typically calculated by the function contract_signals().
    fids : List[str]
        list of financial contract identifiers in the format "<cid>_<ctype>". It must
        correspond to contract signals in the dataframe.
    rstring : str
        a general string of the return category. This identifies the contract returns
        that are required for the volatility-targeting method, based on the category
        identifier format <cid>_<ctype><rstring> in accordance with JPMaQS conventions.
        Default is 'XR'.
    rebal_freq : str
        the frequency of rebalancing and volatility estimation. Default is 'M' for
        monthly. Alternatives are 'W' for business weekly, 'D' for daily, and 'Q' for
        quarterly. Estimations are conducted for the end of the period.
    est_freqs : List[str]
        the list of frequencies for which the volatility is estimated. Volatility for a
        given period is the weighted sum of the volatilities estimated for each frequency.
        Default is ["D", "W", "M"].
    est_weights : List[float]
        the list of weights for each frequency in `est_freqs`. Weights are normalized
        before applying. In cases where there may be missing data or NaNs in the result, the
        remaining weights are normalized. Default is None, which means that the weights are
        equal.
    lback_meth : str
        the method to use for the lookback period of the volatility-targeting method.
        Default is "ma" for moving average. Alternative is "xma", for exponential moving
        average.
    lback_periods : List[int]
        the number of periods to use for the lookback period of the volatility-targeting
        method. Each element corresponds to the the same index in `est_freqs`. Passing a
        single element will apply the same value to all frequencies. Default is [-1], which
        means that the lookback period is the full available data for all specified
        frequencies.
    half_life : List[int]
        number of periods in the half-life of the exponential moving average. Each
        element corresponds to the same index in `est_freqs`.
    start : str
        the start date of the data. Default is None, which means that the start date is
        taken from the dataframe.
    end : str
        the end date of the data. Default is None, which means that the end date is
        taken from the dataframe.
    blacklist : dict
        a dictionary of contract identifiers to exclude from the calculation. Default is
        None, which means that no contracts are excluded.
    nan_tolerance : float
        maximum ratio of number of NaN values to the total number of values in a
        lookback window. If exceeded the resulting volatility is set to NaN, else prior non-
        zero values are added to the window instead. Default is 0.25.
    remove_zeros : bool
        if True (default) any returns that are exact zeros will not be included in the
        lookback window and prior non-zero values are added to the window instead.

    Returns
    -------
    pd.DataFrame
        JPMaQS dataframe of annualized standard deviation of estimated strategy PnL,
        with category name <sname>_PNL_USD1S_ASD. TODO: check if this is correct. The values
        are in % annualized. Values between estimation points are forward filled.

    Notes
    -----
    If returns in the lookback window are not available the function will replace them with
    the average of the available returns of the same contract type. If no returns are
    available for a contract type the function will reduce the lookback window up to a
    minimum of 11 days. If no returns are available for a contract type for at least 11
    days the function returns an NaN for that date and sends a warning of all the dates
    for which this happened.
    """

    if isinstance(lback_periods, Number):
        lback_periods = [lback_periods]
    if isinstance(half_life, Number):
        half_life = [half_life]
    if isinstance(est_weights, Number):
        est_weights = [est_weights]
    if isinstance(est_freqs, str):
        est_freqs = [est_freqs]
    if isinstance(lback_min_obs, Number):
        lback_min_obs = [lback_min_obs]

    ## Check inputs
    # TODO create function for this? Also, do we want to create the set of failures (not just first one)?
    _check_input_arguments(
        arguments=[
            (sname, "sname", str),
            (fids, "fids", list),
            (rstring, "rstring", str),
            (rebal_freq, "rebal_freq", str),
            (lback_meth, "lback_meth", str),
            (lback_periods, "lback_periods", list),
            (half_life, "half_life", list),
            (est_freqs, "est_freqs", list),
            (est_weights, "est_weights", list),
            (start, "start", (str, NoneType)),
            (end, "end", (str, NoneType)),
            (blacklist, "blacklist", (dict, NoneType)),
            (nan_tolerance, "nan_tolerance", float),
            (remove_zeros, "remove_zeros", bool),
            (lback_min_obs, "lback_min_obs", list),
            (return_variance_covariance, "return_variance_covariance", bool),
        ]
    )

    # Check the frequency arguments
    _check_frequency(freq=rebal_freq, freq_type="rebal_freq")

    for ix, freq in enumerate(est_freqs):
        _check_frequency(freq=freq, freq_type=f"est_freq[{ix:d}]")

    ## Check estimation frequency weights
    est_freqs, est_weights, lback_periods, half_life, lback_min_obs = _check_est_args(
        est_freqs=est_freqs,
        est_weights=est_weights,
        lback_periods=lback_periods,
        half_life=half_life,
        lback_min_obs=lback_min_obs,
    )

    ## Standardize and copy DF
    df = QuantamentalDataFrame(df)
    rebal_freq = _map_to_business_day_frequency(rebal_freq)
    est_freqs: List[str] = [_map_to_business_day_frequency(freq) for freq in est_freqs]

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
    df = QuantamentalDataFrame(df).add_ticker_column()
    u_tickers: List[str] = df.list_tickers()

    ## Check for missing data
    _check_missing_data(df=df, sname=sname, fids=fids, rstring=rstring)

    # Add financial identifier (fid) to DataFrame
    df = add_fid_column(df=df, rstring=rstring)

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
        lback_min_obs=lback_min_obs,
        nan_tolerance=nan_tolerance,
        remove_zeros=remove_zeros,
        return_variance_covariance=return_variance_covariance,
    )

    assert len(result) == 1 + int(return_variance_covariance)

    result[0] = QuantamentalDataFrame.from_wide(df=result[0])
    if return_variance_covariance:
        return result[0], result[1]
    return result[0]


def _cov_matrix_history(
    pivot_returns: pd.DataFrame,
    estimation_dates: Union[pd.Series, np.ndarray],
    est_freqs: List[str],
    est_weights: List[float],
    lback_periods: List[int],
    half_life: List[int],
    nan_tolerance: float,
    remove_zeros: bool,
    weights_func: Callable[[int, int], np.ndarray],
    lback_min_obs: List[int],
) -> np.ndarray:
    first_starts = _get_first_usable_date(
        pivot_returns=pivot_returns,
        rebal_dates=estimation_dates,
        est_freqs=est_freqs,
        lback_periods=lback_periods,
    )

    if first_starts.isna().all():
        raise ValueError("No fid has enough history to estimate a covariance matrix")

    fids = pivot_returns.columns.tolist()
    vcv_df_history: np.ndarray = np.full(
        shape=(len(estimation_dates), len(fids), len(fids)),
        fill_value=np.nan,
    )
    for i, estimation_date in enumerate(estimation_dates):
        avails = first_starts[first_starts <= estimation_date].index.tolist()
        if len(avails) == 0:
            logger.warning(
                f"Insufficient return data available on date: {estimation_date} "
                f"to compute a covariance matrix estimate using a "
                f"lookback period of {max(lback_periods)} days."  # todo fix units
            )
            continue

        vcv_df = _calculate_multi_frequency_vcv_for_period(
            pivot_returns=pivot_returns[avails],
            rebal_date=estimation_date,
            est_freqs=est_freqs,
            est_weights=est_weights,
            weights_func=weights_func,
            lback_periods=lback_periods,
            half_life=half_life,
            nan_tolerance=nan_tolerance,
            remove_zeros=remove_zeros,
            lback_min_obs=lback_min_obs,
        )

        if vcv_df.empty:
            logger.warning(f"Covariance matrix empty for date: {estimation_date}")
            continue

        vcv_df = vcv_df.reindex(pivot_returns.columns, axis="columns")
        vcv_df = vcv_df.reindex(pivot_returns.columns, axis="rows")

        vcv_df_history[i] = vcv_df.values

    return vcv_df_history



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

    # Make the returns data start 5 years before the signals data by dropping
    # the first 5 years of signal observations. The return history therefore
    # leads the signals, exercising the returns-lead branch of
    # `_get_first_usable_date`.
    is_signal = df["xcat"].str.endswith("_CSIG_STRAT")
    signals_start = pd.Timestamp(start) + pd.DateOffset(years=5)
    df = df[~(is_signal & (df["real_date"] < signals_start))].reset_index(drop=True)

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
        est_freqs=["D", "W", "M"],
        est_weights=[0.1, 0.2, 0.7],
        lback_periods=[30, 20, -1],
        half_life=[10, 5, 2],
        lback_meth="xma",
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
    # with sns.axes_style("whitegrid"):
    #     fig, ax = plt.subplots(3, 3, figsize=(15, 15))
    #     for ix, dt in enumerate(dates):
    #         sns.heatmap(vcvs_dict[dt], ax=ax[ix // 3, ix % 3])
    #         ax[ix // 3, ix % 3].set_title(dt)
    #     plt.tight_layout()
    #     plt.show()

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
        return_variance_covariance=False,
    )

    # print(df_copy_vol.head(10))
    # print(df_copy_vol.tail(10))

    ########################################

    # Another identical call, but with the signals starting earlier than the
    # returns. Here the first 5 years of return observations are dropped so the
    # signal history leads the returns, exercising the signals-lead branch of
    # `_get_first_usable_date`.
    df_sig_lead = simulate_returns_and_signals(
        cids=cids,
        xcat=xcats[0],
        return_suffix="XR",
        signal_suffix="CSIG_STRAT",
        start=start,
        years=20,
    )
    end_sig_lead = df_sig_lead["real_date"].max().strftime("%Y-%m-%d")

    is_signal = df_sig_lead["xcat"].str.endswith("_CSIG_STRAT")
    returns_start = pd.Timestamp(start) + pd.DateOffset(years=5)
    df_sig_lead = df_sig_lead[
        ~(~is_signal & (df_sig_lead["real_date"] < returns_start))
    ].reset_index(drop=True)

    df_sig_lead["value"] = df_sig_lead["value"].apply(
        lambda x: x if np.random.rand() > N_p_nans else np.nan
    )

    df_sig_lead_vol, vcv_sig_lead_df = historic_portfolio_vol(
        df=df_sig_lead,
        sname="STRAT",
        fids=fids,
        rebal_freq="m",
        est_freqs=["D", "W", "M"],
        est_weights=[0.1, 0.2, 0.7],
        lback_periods=[30, 20, -1],
        half_life=[10, 5, 2],
        lback_meth="xma",
        rstring="XR",
        start=start,
        end=end_sig_lead,
        return_variance_covariance=True,
    )
