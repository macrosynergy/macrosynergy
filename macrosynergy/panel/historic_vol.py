"""
Function for calculating historic volatility of quantamental data.
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any
import warnings
from macrosynergy.management.simulate import make_qdf
from macrosynergy.management.utils import reduce_df, standardise_dataframe, get_eops
from macrosynergy.management.types import QuantamentalDataFrame


def historic_vol(
    df: pd.DataFrame,
    xcat: str = None,
    cids: List[str] = None,
    lback_periods: int = 21,
    lback_meth: str = "ma",
    half_life=11,
    start: str = None,
    end: str = None,
    est_freq: str = "D",
    blacklist: dict = None,
    remove_zeros: bool = True,
    postfix="ASD",
    nan_tolerance: float = 0.25,
):
    """
    Estimate historic annualized standard deviations of asset returns. The function can
    calculate the volatility using either a moving average or an exponential moving
    average method.

    Parameters
    ----------
    df : ~pandas.DataFrame
        standardized DataFrame with the following necessary columns: 'cid', 'xcat',
        'real_date' and 'value'. Will contain all of the data across all macroeconomic
        fields.
    xcat : str
        extended category denoting the return series for which volatility should be
        calculated. Note: in JPMaQS returns are represented in %, i.e. 5 means 5%.
    cids : List[str]
        cross sections for which volatility is calculated; default is all available for
        the category.
    lback_periods : int
        Number of lookback periods over which volatility is calculated. Default is 21.
    lback_meth : str
        Lookback method to calculate the volatility, Default is "ma". Alternative is
        "xma", Exponential Moving Average. Expects to receive either the aforementioned
        strings.
    half_life : int
        Refers to the half-time for "xma". Default is 11.
    start : str
        earliest date in ISO format. Default is None and earliest date in df is used.
    end : str
        latest date in ISO format. Default is None and latest date in df is used.
    est_freq : str
        Frequency of (re-)estimation of volatility. Options are 'D' for end of each day
        (default), 'W' for end of each work week, 'M' for end of each month, and 'Q' for end
        of each week.
    blacklist : dict
        cross sections with date ranges that should be excluded from the data frame. If
        one cross section has several blacklist periods append numbers to the cross section
        code.
    half_life : int
        Refers to the half-time for "xma" and full lookback period for "ma".
    remove_zeros : bool
        if True (default) any returns that are exact zeros will not be included in the
        lookback window and prior non-zero values are added to the window instead.
    postfix : str
        string appended to category name for output; default is "ASD".
    nan_tolerance : float
        maximum ratio of NaNs to non-NaNs in a lookback window, if exceeded the
        resulting volatility is set to NaN. Default is 0.25.

    Returns
    -------
    ~pandas.DataFrame
        standardized DataFrame with the estimated annualized standard deviations of the
        chosen category. If the input 'value' is in % (as is the standard in 
        JPMaQS) then the output will also be in %. 'cid', 'xcat', 'real_date' and 'value'.
    """

    df: QuantamentalDataFrame = QuantamentalDataFrame(df)
    est_freq = est_freq.lower()
    assert lback_meth in ["xma", "ma"], (
        "Lookback method must be either 'xma' "
        "(exponential moving average) or 'ma' (moving average)."
    )
    if lback_meth == "xma":
        assert (
            lback_periods > half_life
        ), "Half life must be shorter than lookback period."
        assert half_life > 0, "Half life must be greater than 0."
    assert est_freq in [
        "d",
        "w",
        "m",
        "q",
    ], "Estimation frequency must be one of 'D', 'W', 'M', or 'Q'."

    # assert nan tolerance is an int or float. must be >0. if >1 must be int
    assert isinstance(
        nan_tolerance, (int, float)
    ), "nan_tolerance must be an int or float."
    assert (
        0 <= nan_tolerance <= 1
    ), "nan_tolerance must be between 0.0 and 1.0 inclusive."

    df = reduce_df(
        df, xcats=[xcat], cids=cids, start=start, end=end, blacklist=blacklist
    )

    dfw = df.pivot(index="real_date", columns="cid", values="value")

    trigger_indices = get_eops(
        dates=pd.DataFrame(dfw.index),
        freq=est_freq,
    )

    def single_calc(
        row,
        dfw: pd.DataFrame,
        lback_periods: int,
        nan_tolerance: float,
        roll_func: callable,
        remove_zeros: bool,
        weights: Optional[np.ndarray] = None,
    ):
        """
        Helper function to calculate the historic volatility for a single row in the
        DataFrame.
        """
        target_dates = pd.bdate_range(end=row["real_date"], periods=lback_periods)
        target_df: pd.DataFrame = dfw.loc[dfw.index.isin(target_dates)]

        if weights is None:
            out = np.sqrt(252) * target_df.agg(roll_func, remove_zeros=remove_zeros)
        else:
            if len(weights) == len(target_df):
                out = np.sqrt(252) * target_df.agg(
                    roll_func, w=weights, remove_zeros=remove_zeros
                )
            else:
                return pd.Series(np.nan, index=target_df.columns)

        mask = (
            (
                target_df.isna().sum(axis=0)
                + (target_df == 0).sum(axis=0)
                + (lback_periods - len(target_df))
            )
            / lback_periods
        ) <= nan_tolerance
        # NOTE: dates with NaNs, dates with missing entries, and dates with 0s
        # are all treated as missing data and trigger a NaN in the output
        out[~mask] = np.nan

        return out

    expo_weights_arr: Optional[np.ndarray] = None
    if lback_meth == "xma":
        expo_weights_arr = expo_weights(lback_periods, half_life)

    if est_freq == "d":
        _args: Dict[str, Any] = dict(remove_zeros=remove_zeros)
        if lback_meth == "xma":
            _args["w"] = expo_weights_arr
            _args["func"] = expo_std
        else:
            _args["func"] = flat_std

        dfwa = np.sqrt(252) * dfw.rolling(window=lback_periods).agg(**_args)
    else:
        dfwa = pd.DataFrame(index=dfw.index, columns=dfw.columns)
        _args: Dict[str, Any] = dict(
            lback_periods=lback_periods,
            nan_tolerance=nan_tolerance,
            remove_zeros=remove_zeros,
        )
        if lback_meth == "xma":
            _args["weights"] = expo_weights_arr
            _args["roll_func"] = expo_std

        else:
            _args["roll_func"] = flat_std

        dfwa.loc[trigger_indices, :] = (
            dfwa.loc[trigger_indices, :]
            .reset_index(False)
            .apply(
                lambda row: single_calc(
                    row=row,
                    dfw=dfw,
                    **_args,
                ),
                axis=1,
            )
            .set_index(trigger_indices)
        )

        fills = {"d": 1, "w": 5, "m": 24, "q": 64}
        dfwa = dfwa.astype(float).reindex(dfw.index).ffill(limit=fills[est_freq])

    df_out = dfwa.unstack().reset_index().rename({0: "value"}, axis=1)

    # Create an initial mask for all rows to keep
    keep_mask = pd.Series(False, index=df_out.index)

    # Iterate over each cid and mark valid rows
    for cid in cids:
        # Get the date range for the current 'cid' in the original df
        loc_bools = df["cid"] == cid
        if df[loc_bools].empty:
            warnings.warn(f"No data for {cid}_{xcat}. Skipping.")
            continue
        min_date = df.loc[loc_bools, "real_date"].min()
        max_date = df.loc[loc_bools, "real_date"].max()

        # Generate valid date range for the current 'cid'
        valid_dates = pd.bdate_range(start=min_date, end=max_date)

        # Update the keep_mask for rows corresponding to current 'cid' with valid dates
        sel_bools = df_out["cid"] == cid
        sel_dts = df_out["real_date"].isin(valid_dates)

        keep_mask |= sel_bools & sel_dts

    # Apply the mask to df_out
    df_out = df_out[keep_mask].reset_index(drop=True)

    df_out = QuantamentalDataFrame.from_long_df(
        df=df_out,
        xcat=xcat + postfix,
        categorical=df.InitializedAsCategorical,
    )
    return standardise_dataframe(df_out)


def expo_weights(lback_periods: int = 21, half_life: int = 11):
    """
    Calculates exponential series weights for finite horizon, normalized to 1.

    Parameters
    ----------
    lback_periods : int
        Number of lookback periods over which volatility is calculated. Default is 21.
    half_life : int
        Refers to the half-time for "xma" and full lookback period for "ma". Default is
        11.

    Returns
    -------
    ~numpy.ndarray
        An Array of weights determined by the length of the lookback period.

    Notes
    -----
    50% of the weight allocation will be applied to the number of days delimited by the
    half_life.
    """

    decf = 2 ** (-1 / half_life)
    weights = (1 - decf) * np.array(
        [decf ** (lback_periods - ii - 1) for ii in range(lback_periods)]
    )
    weights = weights / sum(weights)

    return weights


def expo_std(x: np.ndarray, w: np.ndarray, remove_zeros: bool = True):
    """
    Estimate standard deviation of returns based on exponentially weighted absolute
    values.

    Parameters
    ----------
    x : ~numpy.ndarray
        array of returns
    w : ~numpy.ndarray
        array of exponential weights (same length as x); will be normalized to 1.
    remove_zeros : bool
        removes zeroes as invalid entries and shortens the effective window.

    Returns
    -------
    float
        exponentially weighted mean absolute value (as proxy of return standard
        deviation).
    """

    assert len(x) == len(w), "weights and window must have same length"
    if remove_zeros:
        x = x[x != 0]
        w = w[0 : len(x)] / sum(w[0 : len(x)])
    w = w / sum(w)  # weights are normalized
    mabs = np.sum(np.multiply(w, np.abs(x)))
    return mabs


def flat_std(x: np.ndarray, remove_zeros: bool = True):
    """
    Estimate standard deviation of returns based on exponentially weighted absolute
    values.

    Parameters
    ----------
    x : ~numpy.ndarray
        array of returns
    remove_zeros : bool
        removes zeroes as invalid entries and shortens the effective window.

    Returns
    -------
    float
        flat weighted mean absolute value (as proxy of return standard deviation).
    """

    if remove_zeros:
        x = x[x != 0]
    mabs = np.mean(np.abs(x))
    return mabs


if __name__ == "__main__":
    cids = ["AUD", "CAD", "GBP", "USD"]
    xcats = ["XR", "CRY", "GROWTH", "INFL"]

    df_cids = pd.DataFrame(
        index=cids, columns=["earliest", "latest", "mean_add", "sd_mult"]
    )

    df_cids.loc["AUD"] = ["2010-01-01", "2020-12-31", 0.5, 2]
    df_cids.loc["CAD"] = ["2011-01-01", "2020-11-30", 0, 1]
    df_cids.loc["GBP"] = ["2012-01-01", "2020-10-30", -0.2, 0.5]
    df_cids.loc["USD"] = ["2013-01-01", "2020-09-30", -0.2, 0.5]

    df_xcats = pd.DataFrame(
        index=xcats,
        columns=["earliest", "latest", "mean_add", "sd_mult", "ar_coef", "back_coef"],
    )
    df_xcats.loc["XR"] = ["2010-01-01", "2020-12-31", 0, 1, 0, 0.3]
    df_xcats.loc["CRY"] = ["2011-01-01", "2020-10-30", 1, 2, 0.9, 0.5]
    df_xcats.loc["GROWTH"] = ["2012-01-01", "2020-10-30", 1, 2, 0.9, 1]
    df_xcats.loc["INFL"] = ["2013-01-01", "2020-10-30", 1, 2, 0.8, 0.5]
    dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)
    dfd["grading"] = np.ones(dfd.shape[0])

    print("Calculating historic volatility with the moving average method")
    df = historic_vol(
        dfd,
        cids=cids,
        xcat="XR",
        lback_periods=7,
        lback_meth="ma",
        est_freq="w",
        half_life=3,
        remove_zeros=True,
    )

    print(df.head(10))

    print("Calculating historic volatility with the exponential moving average method")
    df = historic_vol(
        dfd,
        cids=cids,
        xcat="XR",
        lback_periods=7,
        lback_meth="xma",
        est_freq="w",
        half_life=3,
        remove_zeros=True,
    )

    print(df.head(10))
