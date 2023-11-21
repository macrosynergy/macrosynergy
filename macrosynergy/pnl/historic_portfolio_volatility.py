"""
Module for calculating the historic portfolio volatility for a given strategy.
"""

from typing import List, Optional
import pandas as pd
import numpy as np

import os, sys

sys.path.append(os.getcwd())


from macrosynergy.management.types import NoneType
from macrosynergy.management.utils import (
    reduce_df,
    standardise_dataframe,
    is_valid_iso_date,
    qdf_to_ticker_df,
    ticker_df_to_qdf,
)
from macrosynergy.panel.historic_vol import get_cycles


def expo_weights(lback_periods: int = 21, half_life: int = 11):
    """
    Calculates exponential series weights for finite horizon, normalized to 1.

    :param <int>  lback_periods: Number of lookback periods over which volatility is
        calculated. Default is 21.
    :param <int> half_life: Refers to the half-time for "xma" and full lookback period
        for "ma". Default is 11.

    :return <np.ndarray>: An Array of weights determined by the length of the lookback
        period.

    Note: 50% of the weight allocation will be applied to the number of days delimited by
          the half_life.
    """
    # Copied from macrosynergy.panel.historic_vol
    decf = 2 ** (-1 / half_life)
    weights = (1 - decf) * np.array(
        [decf ** (lback_periods - ii - 1) for ii in range(lback_periods)]
    )
    weights = weights / sum(weights)

    return weights


def expo_std(x: np.ndarray, w: np.ndarray, remove_zeros: bool = True) -> float:
    """
    Estimate standard deviation of returns based on exponentially weighted absolute
    values.

    :param <np.ndarray> x: array of returns
    :param <np.ndarray> w: array of exponential weights (same length as x); will be
        normalized to 1.
    :param <bool> remove_zeros: removes zeroes as invalid entries and shortens the
        effective window.

    :return <float>: exponentially weighted mean absolute value (as proxy of return
        standard deviation).

    """
    # Copied from macrosynergy.panel.historic_vol

    assert len(x) == len(w), "weights and window must have same length"
    if remove_zeros:
        x = x[x != 0]
        w = w[0 : len(x)] / sum(w[0 : len(x)])
    w = w / sum(w)  # weights are normalized
    mabs = np.sum(np.multiply(w, np.abs(x)))
    return mabs


def flat_std(x: np.ndarray, remove_zeros: bool = True) -> float:
    """
    Estimate standard deviation of returns based on exponentially weighted absolute
    values.

    :param <np.ndarray> x: array of returns
    :param <bool> remove_zeros: removes zeroes as invalid entries and shortens the
        effective window.

    :return <float>: flat weighted mean absolute value (as proxy of return standard
        deviation).

    """
    # Copied from macrosynergy.panel.historic_vol

    if remove_zeros:
        x = x[x != 0]
    mabs = np.mean(np.abs(x))
    return mabs


def _single_calc(
    row,
    ticker_df: pd.DataFrame,
    lback_periods: int,
    nan_tolerance: float,
    roll_func: callable,
    remove_zeros: bool,
    weights: Optional[np.ndarray] = None,
):
    # Copied from macrosynergy.panel.historic_vol

    df_wide: pd.DataFrame = ticker_df.loc[
        ticker_df.index.isin(
            pd.bdate_range(end=row["real_date"], periods=lback_periods)
        )
    ]

    if weights is None:
        out = np.sqrt(252) * df_wide.agg(roll_func, remove_zeros=remove_zeros)
    else:
        if len(weights) == len(df_wide):
            out = np.sqrt(252) * df_wide.agg(
                roll_func, w=weights, remove_zeros=remove_zeros
            )
        else:
            return pd.Series(np.nan, index=df_wide.columns)

    mask = (
        (
            df_wide.isna().sum(axis=0)
            + (df_wide == 0).sum(axis=0)
            + (lback_periods - len(df_wide))
        )
        / lback_periods
    ) <= nan_tolerance
    # NOTE: dates with NaNs, dates with missing entries, and dates with 0s
    # are all treated as missing data and trigger a NaN in the output
    out[~mask] = np.nan

    return out


def _hist_vol(
    df: pd.DataFrame,
    sname: str,
    fids: List[str],
    est_freq: str = "m",
    lback_periods: int = 21,
    lback_meth: str = "ma",
    half_life=11,
    nan_tolerance: float = 0.25,
    remove_zeros: bool = True,
):
    lback_meth = lback_meth.lower()
    if not lback_meth in ["ma", "xma"]:
        raise NotImplementedError(
            f"`lback_meth` must be 'ma' or 'xma'; got {lback_meth}"
        )

    df_wide: pd.DataFrame = qdf_to_ticker_df(df=df)
    sig_ident: str = f"_CSIG_{sname}"

    _fconts: List[str] = [f"{contx}{sig_ident}" for contx in fids]
    if not set(_fconts).issubset(set(df_wide.columns)):
        raise ValueError(
            f"Contract signals for all contracts not in dataframe. \n"
            f"Missing: {set(_fconts) - set(df_wide.columns)}"
        )

    # Create empty weights dataframe, similar to df_wide

    trigger_indices = df_wide.index[
        get_cycles(
            pd.DataFrame({"real_date": df_wide.index}),
            freq=est_freq,
        )
    ]

    dfw_calc: pd.DataFrame = pd.DataFrame(
        index=trigger_indices, columns=df_wide.columns, dtype=float
    )

    expo_weights_arr: Optional[np.ndarray] = None
    if lback_meth == "xma":
        expo_weights_arr: np.ndarray = expo_weights(
            lback_periods=lback_periods,
            half_life=half_life,
        )

    if est_freq == "d":
        _args: dict
        if lback_meth == "xma":
            _args = dict(func=expo_std, w=expo_weights_arr, remove_zeros=remove_zeros)
        elif lback_meth == "ma":
            _args = dict(func=flat_std, remove_zeros=remove_zeros)

        dfw_calc = df_wide.rolling(lback_periods).agg(**_args)

    else:
        _args = dict(
            remove_zeros=remove_zeros,
            lback_periods=lback_periods,
            nan_tolerance=nan_tolerance,
        )
        if lback_meth == "xma":
            _args["roll_func"] = expo_std
            _args["w"] = expo_weights_arr
        elif lback_meth == "ma":
            _args["roll_func"] = flat_std

        # NOTE
        # LOC Only rows for dates that are in the trigger_indices:
        # Reset Index to make real_date accessible to _single_calc
        # Apply _single_calc to each row
        # setindex back to dfw_calc's index
        # Reindex to match the original df_wide index

        dfw_calc.loc[trigger_indices, :] = (
            dfw_calc.loc[trigger_indices, :]
            .reset_index()
            .apply(
                lambda row: _single_calc(row=row, ticker_df=df_wide, **_args),
                axis=1,
            )
            .set_index(dfw_calc.index)
        )

    fills = {"d": 1, "w": 5, "m": 24, "q": 64}
    dfw_calc = dfw_calc.reindex(df_wide.index).ffill(limit=fills[est_freq])

    return ticker_df_to_qdf(df=dfw_calc)


def _var_covar_matrix(
    df: pd.DataFrame,
    fids: List[str],
    sname: str,
):
    df_wide: pd.DataFrame = qdf_to_ticker_df(df=df)
    sig_ident: str = f"_CSIG_{sname}"

    _check_conts: List[str] = [f"{contx}{sig_ident}" for contx in fids]
    if not set(_check_conts).issubset(set(df_wide.columns)):
        raise ValueError(
            f"Contract signals for all contracts not in dataframe. \n"
            f"Missing: {set(_check_conts) - set(df_wide.columns)}"
        )

    return df_wide[_check_conts].cov()


def historic_portfolio_vol(
    df: pd.DataFrame,
    sname: str,
    fids: List[str],
    est_freq: str = "m",
    lback_periods: int = 21,
    lback_meth: str = "ma",
    half_life=11,
    rstring: str = "XR",
    start: Optional[str] = None,
    end: Optional[str] = None,
    blacklist: Optional[dict] = None,
    nan_tolerance: float = 0.25,
):
    """
    Estimates the annualized standard deviations of portfolio, based on historic
    variances and co-variances.

    :param <pd.DataFrame> df:  standardized JPMaQS DataFrame with the necessary
        columns: 'cid', 'xcat', 'real_date' and 'value'.
        This dataframe must contain the contract-specific signals and return series.
    :param <str> sname: the name of the strategy. It must correspond to contract
        signals in the dataframe, which have the format "<cid>_<ctype>_CSIG_<sname>", and
        which are typically calculated by the function contract_signals().
    :param <List[str]> fids: list of financial contract identifiers in the format
        "<cid>_<ctype>". It must correspond to contract signals in the dataframe.
    :param <str> est_freq: the frequency of the volatility estimation. Default is 'm'
        for monthly. Alternatives are 'w' for business weekly, 'd' for daily, and 'q'
        for quarterly. Estimations are conducted for the end of the period.
    :param <float> dollar_per_signal: the amount of notional currency (e.g. USD) per
        contract signal value. Default is 1. The default scale has no specific meaning
        and is merely a basis for tryouts.
    :param <int> lback_periods: the number of periods to use for the lookback period
        of the volatility-targeting method. Default is 21. This passed through to
        the function `historic_portfolio_vol()`.
    :param <str> lback_meth: the method to use for the lookback period of the
        volatility-targeting method. Default is 'ma' for moving average. Alternative is
        "xma", for exponential moving average. Again this is passed through to
        the function `historic_portfolio_vol()`.
    :param <str> rstring: a general string of the return category. This identifies
        the contract returns that are required for the volatility-targeting method, based
        on the category identifier format <cid>_<ctype><rstring>_<rstring> in accordance
        with JPMaQS conventions. Default is 'XR'.
    :param <str> start: the start date of the data. Default is None, which means that
        the start date is taken from the dataframe.
    :param <str> end: the end date of the data. Default is None, which means that
        the end date is taken from the dataframe.
    :param <dict> blacklist: a dictionary of contract identifiers to exclude from
        the calculation. Default is None, which means that no contracts are excluded.
    :param <float> nan_tolerance: minimum ratio of NaNs to non-NaNs in a lookback window,
        if exceeded the resulting volatility is set to NaN. Default is 0.25.


    :return: <pd.DataFrame> with the annualized standard deviations of the portfolios.
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

    df["ticker"]: str = df["cid"] + "_" + df["xcat"]
    ## Check that there is atleast one contract signal for the strategy
    # check using endswith
    if not any(df["ticker"].str.endswith(f"_CSIG_{sname}")):
        raise ValueError(f"No contract signals for strategy `{sname}`.")

    u_tickers: List[str] = list(df["ticker"].unique())
    for contx in fids:
        if not any(
            [tx.startswith(contx) and tx.endswith(f"_CSIG_{sname}") for tx in u_tickers]
        ):
            raise ValueError(f"Contract identifier `{contx}` not in dataframe.")

    hist_vol: pd.DataFrame = _hist_vol(
        df=df,
        sname=sname,
        fids=fids,
        est_freq=est_freq,
        lback_periods=lback_periods,
        lback_meth=lback_meth,
        half_life=half_life,
        nan_tolerance=nan_tolerance,
    )

    vcv_matrix: pd.DataFrame = _var_covar_matrix(
        df=hist_vol,
        sname=sname,
        fids=fids,
    )
    
    


if __name__ == "__main__":
    from macrosynergy.management.simulate import make_test_df
    from contract_signals import contract_signals

    cids: List[str] = ["USD", "EUR", "GBP", "AUD", "CAD"]
    xcats: List[str] = ["SIG", "HR"]

    start: str = "2000-01-01"
    end: str = "2020-12-31"

    df: pd.DataFrame = make_test_df(
        cids=cids,
        xcats=xcats,
        start=start,
        end=end,
    )

    df.loc[(df["cid"] == "USD") & (df["xcat"] == "SIG"), "value"] = 1.0
    ctypes = ["FX", "IRS", "CDS"]
    cscales = [1.0, 0.5, 0.1]
    csigns = [1, -1, 1]

    hbasket = ["USD_EQ", "EUR_EQ"]
    hscales = [0.7, 0.3]

    df_cs: pd.DataFrame = contract_signals(
        df=df,
        sig="SIG",
        cids=cids,
        ctypes=ctypes,
        cscales=cscales,
        csigns=csigns,
        hbasket=hbasket,
        hscales=hscales,
        hratios="HR",
        sname="TEST",
    )
    fids: List[str] = [f"{cid}_{ctype}" for cid in cids for ctype in ctypes]
    df_vol: pd.DataFrame = historic_portfolio_vol(
        df=df_cs,
        sname="TEST",
        fids=fids,
        est_freq="m",
        lback_periods=21,
        lback_meth="ma",
        half_life=11,
        rstring="XR",
        start=start,
        end=end,
    )
