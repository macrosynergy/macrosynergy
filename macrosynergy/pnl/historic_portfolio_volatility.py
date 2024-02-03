"""
Module for calculating the historic portfolio volatility for a given strategy.
"""

from typing import List, Optional, Callable
import pandas as pd
import numpy as np

import os, sys

sys.path.append(os.getcwd())


from macrosynergy.management.types import NoneType, QuantamentalDataFrame
from macrosynergy.management.utils import (
    reduce_df,
    standardise_dataframe,
    is_valid_iso_date,
    qdf_to_ticker_df,
    ticker_df_to_qdf,
    get_eops,
    get_cid,
    get_xcat,
)

RETURN_SERIES_XCAT = "_PNL_USD1S_ASD"


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


def _rolling_window_calc(
    real_date_pdt: pd.Timestamp,
    ticker_df: pd.DataFrame,
    lback_periods: int,
    nan_tolerance: float,
    roll_func: Callable,
    remove_zeros: bool,
    weights: Optional[np.ndarray] = None,
):
    """
    Runs the volatility calculation for a given row of the dataframe.
    It needs access to the full 'ticker_df' dataframe to make the lookback possible.

    :param <pd.Series> row: a row of the dataframe, which contains the index, the
        'real_date' and the 'ticker' of the current calculation.
    :param <pd.DataFrame> ticker_df: the wide dataframe of all the tickers in the strategy.
    :param <int> lback_periods: the number of periods to use for the lookback period
        of the volatility-targeting method. Default is 21.
    :param <float> nan_tolerance: maximum ratio of NaNs to non-NaNs in a lookback window,
        if exceeded the resulting volatility is set to NaN. Default is 0.25.
    :param <callable> roll_func: the function to use for the rolling window calculation.
        The function must have a signature of (np.ndarray, *, remove_zeros: bool) -> float.
        See `expo_std` and `flat_std` for examples.
    :param <bool> remove_zeros: removes zeroes as invalid entries and shortens the
        effective window.
    :param <np.ndarray> weights: array of exponential weights for each period in the
        lookback window. Default is None, which means that the weights are equal.
    """
    # use end=row["real_date"] when using apply
    df_wide: pd.DataFrame = ticker_df.loc[
        ticker_df.index.isin(pd.bdate_range(end=real_date_pdt, periods=lback_periods))
    ]

    ## Calculate univariate volatility

    univariate_vol: pd.Series
    if weights is None:
        weights = np.ones(lback_periods) / lback_periods
        univariate_vol = np.sqrt(252) * df_wide.agg(
            roll_func, remove_zeros=remove_zeros
        )
    else:
        if len(weights) == len(df_wide):
            univariate_vol = np.sqrt(252) * df_wide.agg(
                roll_func, w=weights, remove_zeros=remove_zeros
            )
        else:
            return pd.Series(np.nan, index=df_wide.columns)

    ## Creating a mask to fill series `nan_tolerance`
    mask = (
        (
            df_wide.isna().sum(axis=0)
            + (df_wide == 0).sum(axis=0)
            + (lback_periods - len(df_wide))
        )
        / lback_periods
    ) <= nan_tolerance

    univariate_vol[~mask] = np.nan

    ## Inversed univariate volatility
    inv_univariate_vol = 1 / univariate_vol

    vcv: pd.DataFrame = df_wide.cov()
    total_variance: float = vcv.to_numpy().sum()
    period_volatility: float = np.sqrt(total_variance)
    annualized_vol: pd.Series = period_volatility * np.sqrt(252)

    return annualized_vol


def _hist_vol(
    df: pd.DataFrame,
    sname: str,
    rstring: str,
    weight_signal: str,
    est_freq: str = "m",
    lback_periods: int = 21,
    lback_meth: str = "ma",
    half_life=11,
    nan_tolerance: float = 0.25,
    remove_zeros: bool = True,
):
    """
    Calculates historic volatility for a given strategy. It assumes that the dataframe
    is composed solely of the relevant signals and returns for the strategy.

    :param <pd.DataFrame> df:  standardized JPMaQS DataFrame with the necessary
        columns: 'cid', 'xcat', 'real_date' and 'value'.
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
    if not lback_meth in ["ma", "xma"]:
        raise NotImplementedError(
            f"`lback_meth` must be 'ma' or 'xma'; got {lback_meth}"
        )

    df_wide: pd.DataFrame = qdf_to_ticker_df(df=df)

    ## Multiply the returns by weight_signals
    r_series_list = df_wide[
        df_wide.columns[
            df_wide.columns.str.endswith(f"{rstring}_{rstring}_CSIG_{sname}")
        ]
    ]
    w_series_list = df_wide[
        df_wide.columns[
            df_wide.columns.str.endswith(f"{rstring}_{weight_signal}_CSIG_{sname}")
        ]
    ]

    for r_series, w_series in zip(r_series_list, w_series_list):
        df_wide[r_series] = df_wide[r_series] * df_wide[w_series]

    df_wide = df_wide[r_series_list]

    # NOTE: `get_eops` helps identify the dates for which the volatility calculation
    # will be performed. This is done by identifying the last date of each cycle.
    # We use `get_eops` primarily because it has a clear and defined behaviour for all frequencies.
    # It was originally designed as part of the `historic_vol` module, but it is
    # used here as well.

    trigger_indices = get_eops(
        dates=df_wide.index,
        freq=est_freq,
    )
    return_series = f"{sname}{RETURN_SERIES_XCAT}"
    dfw_calc: pd.DataFrame = pd.DataFrame(
        index=trigger_indices, columns=[return_series], dtype=float
    )

    expo_weights_arr: Optional[np.ndarray] = None
    if lback_meth == "xma":
        expo_weights_arr: np.ndarray = expo_weights(
            lback_periods=lback_periods,
            half_life=half_life,
        )

    if est_freq == "d":
        _args: dict = dict(remove_zeros=remove_zeros)
        if lback_meth == "xma":
            _args.update(dict(func=expo_std, w=expo_weights_arr))
        elif lback_meth == "ma":
            _args.update(dict(func=flat_std))

    else:
        _args = dict(
            remove_zeros=remove_zeros,
            lback_periods=lback_periods,
            nan_tolerance=nan_tolerance,
        )
        if lback_meth == "xma":
            _args.update(dict(roll_func=expo_std, w=expo_weights_arr))
        elif lback_meth == "ma":
            _args.update(dict(roll_func=flat_std))

    if est_freq == "d":
        # dfw_calc = df_wide.rolling(lback_periods).agg(**_args)
        raise NotImplementedError("Daily volatility not implemented yet.")
        # TODO: Contradiction to other condition above

    else:
        for r_date in trigger_indices:
            dfw_calc.loc[r_date, :] = _rolling_window_calc(
                real_date_pdt=r_date,
                ticker_df=df_wide,
                **_args,
            )

    fills = {"d": 1, "w": 5, "m": 24, "q": 64}
    dfw_calc = dfw_calc.reindex(df_wide.index).ffill(limit=fills[est_freq])

    return ticker_df_to_qdf(df=dfw_calc)


# dfw_calc.loc[trigger_indices, :] = (
#     dfw_calc.loc[trigger_indices, :]
#     .reset_index()
#     .apply(
#         lambda row: _rolling_window_calc(row=row, ticker_df=df_wide, **_args),
#         axis=1,
#     )
#     .set_index(dfw_calc.index)
# )


def historic_portfolio_vol(
    df: pd.DataFrame,
    sname: str,
    fids: List[str],
    est_freq: str = "m",
    lback_periods: int = 21,
    lback_meth: str = "ma",
    half_life=11,
    rstring: str = "XR",
    weight_signal: str = "XRWGT",
    start: Optional[str] = None,
    end: Optional[str] = None,
    blacklist: Optional[dict] = None,
    nan_tolerance: float = 0.25,
    remove_zeros: bool = True,
):
    """
    Estimates annualized standard deviations of a portfolio, based on historic
    variances and co-variances.

    :param <pd.DataFrame> df:  standardized JPMaQS DataFrame with the necessary
        columns: 'cid', 'xcat', 'real_date' and 'value'.
        This dataframe must contain contract-specific signals and return series.
    :param <str> sname: the name of the strategy. It must correspond to contract
        signals in the dataframe, which have the format "<cid>_<ctype>_CSIG_<sname>", and
        which are typically calculated by the function contract_signals().
    :param <List[str]> fids: list of financial contract identifiers in the format
        "<cid>_<ctype>". It must correspond to contract signals in the dataframe.
    :param <str> est_freq: the frequency of the volatility estimation. Default is 'm'
        for monthly. Alternatives are 'w' for business weekly, 'd' for daily, and 'q'
        for quarterly. Estimations are conducted for the end of the period.
    :param <int> lback_periods: the number of periods to use for the lookback period
        of the volatility-targeting method. Default is 21.
    :param <str> lback_meth: the method to use for the lookback period of the
        volatility-targeting method. Default is 'ma' for moving average. Alternative is
        "xma", for exponential moving average.
    :param <str> rstring: a general string of the return category. This identifies
        the contract returns that are required for the volatility-targeting method, based
        on the category identifier format <cid>_<ctype><rstring>_<rstring>_CSIG_<sname>
        in accordance with JPMaQS conventions. Default is 'XR'.
    :param <str> weight_signal: the name of the signal (time series) to use as weights
        for the volatility calculation. Default is None, which means that the weights
        are equal. The signal must be in the format <cid>_<ctype>_<rstring>_<weight_signal>_CSIG_<sname>.
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

    df["ticker"]: str = df["cid"] + "_" + df["xcat"]

    ## Check that there is atleast one contract signal for the strategy
    if not any(df["ticker"].str.endswith(f"_CSIG_{sname}")):
        raise ValueError(f"No contract signals for strategy `{sname}`.")

    u_tickers: List[str] = list(df["ticker"].unique())
    for contx in fids:
        if not any(
            [tx.startswith(contx) and tx.endswith(f"_CSIG_{sname}") for tx in u_tickers]
        ):
            raise ValueError(f"Contract identifier `{contx}` not in dataframe.")

    ## Filter DF such that all timeseries have the correct return category, strategy name
    ## and contract identifier
    _cfids = get_cid(ticker=fids)
    filt_tickers: List[str] = [
        tx
        for tx in u_tickers
        if (
            tx.endswith(f"_CSIG_{sname}")
            and (f"{rstring}_{rstring}" in tx)
            and (get_cid(tx) in _cfids)
        )
    ]

    filt_weights: List[str] = []
    filt_weights = [
        f"{get_cid(tx)}_{rstring}_{weight_signal}_CSIG_{sname}" for tx in filt_tickers
    ]

    # missing weights - fill with 1
    for w in filt_weights:
        if w not in df["ticker"].unique():
            dtr = df.loc[
                df["ticker"].str.contains(f"{get_cid(w)}_{rstring}"), "real_date"
            ]
            ndf = pd.DataFrame(
                {
                    "cid": get_cid(w),
                    "xcat": get_xcat(w),
                    "real_date": dtr,
                    "value": 1,
                }
            )
            df = pd.concat([df, ndf])

    df: pd.DataFrame = df.loc[df["ticker"].isin(filt_tickers + filt_weights)]

    if df.empty:
        raise ValueError(
            "No data available for the given strategy and contract identifiers."
            "Please check the inputs. \n"
            f"Strategy: {sname}\nContract identifiers: {fids} \n"
            f"Return string: {rstring}\nWeight signal: {weight_signal}"
        )

    hist_port_vol: pd.DataFrame = _hist_vol(
        df=df,
        sname=sname,
        rstring=rstring,
        weight_signal=weight_signal,
        est_freq=est_freq,
        lback_periods=lback_periods,
        lback_meth=lback_meth,
        half_life=half_life,
        nan_tolerance=nan_tolerance,
        remove_zeros=remove_zeros,
    )

    assert isinstance(hist_port_vol, QuantamentalDataFrame)

    return hist_port_vol


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
    ctypes = ["FXXR_XR", "IRSXR_XR", "CDSXR_XR"]
    ctypes += [c.replace("_XR", "_WEIGHTS") for c in ctypes]
    cscales = [1.0, 0.5, 0.1]
    csigns = [1, -1, 1]

    hbasket = ["USD_EQXR_XR", "EUR_EQXR_XR"]
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
        sname="mySTRAT",
    )
    ## `df_cs` looks like:
    #        cid                   xcat  real_date         value
    # 0      AUD  CDSXR_XR_CSIG_mySTRAT 2000-01-03      0.001825
    # 1      AUD   FXXR_XR_CSIG_mySTRAT 2000-01-03      0.018252
    # 2      AUD  IRSXR_XR_CSIG_mySTRAT 2000-01-03     -0.009126
    # 3      CAD  CDSXR_XR_CSIG_mySTRAT 2000-01-03      5.005734
    # 4      CAD   FXXR_XR_CSIG_mySTRAT 2000-01-03     50.057339
    # ...    ...                    ...        ...           ...
    # 54785  USD   EQXR_XR_CSIG_mySTRAT 2020-12-31  22749.226123
    # 54786  USD   EQXR_XR_CSIG_mySTRAT 2020-12-31  22749.226123
    # 54787  USD   EQXR_XR_CSIG_mySTRAT 2020-12-31  22749.226123
    # 54788  USD   EQXR_XR_CSIG_mySTRAT 2020-12-31  22749.226123
    # 54789  USD   EQXR_XR_CSIG_mySTRAT 2020-12-31  22749.226123
    # [136975 rows x 4 columns]

    fids: List[str] = [f"{cid}_{ctype}" for cid in cids for ctype in ctypes]
    df_vol: pd.DataFrame = historic_portfolio_vol(
        df=df_cs,
        sname="mySTRAT",
        fids=fids,
        est_freq="m",
        lback_periods=15,
        lback_meth="ma",
        half_life=11,
        rstring="XR",
        start=start,
        end=end,
    )

    ## `df_vol` looks like:
    #         cid           xcat  real_date        value
    # 0     mySTRAT  PNL_USD1S_ASD 2000-01-31   331.875030
    # 1     mySTRAT  PNL_USD1S_ASD 2000-02-01   331.875030
    # 2     mySTRAT  PNL_USD1S_ASD 2000-02-02   331.875030
    # 3     mySTRAT  PNL_USD1S_ASD 2000-02-03   331.875030
    # 4     mySTRAT  PNL_USD1S_ASD 2000-02-04   331.875030
    # ...       ...            ...        ...          ...
    # 5454  mySTRAT  PNL_USD1S_ASD 2020-12-25  1016.964736
    # 5455  mySTRAT  PNL_USD1S_ASD 2020-12-28  1016.964736
    # 5456  mySTRAT  PNL_USD1S_ASD 2020-12-29  1016.964736
    # 5457  mySTRAT  PNL_USD1S_ASD 2020-12-30  1016.964736
    # 5458  mySTRAT  PNL_USD1S_ASD 2020-12-31  1000.390667
    # [5459 rows x 4 columns]
