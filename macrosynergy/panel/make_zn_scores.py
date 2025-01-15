"""
Module for calculating z-scores for a panel around a neutral level ("zn scores").
"""

import numpy as np
import pandas as pd
from typing import List, Union
from numbers import Number
from macrosynergy.management.simulate import make_qdf
from macrosynergy.management.utils import (
    drop_nan_series,
    reduce_df,
    _map_to_business_day_frequency,
)
from macrosynergy.management.types import QuantamentalDataFrame
from numbers import Number


def make_zn_scores(
    df: pd.DataFrame,
    xcat: str,
    cids: List[str] = None,
    start: str = None,
    end: str = None,
    blacklist: dict = None,
    sequential: bool = True,
    min_obs: int = 261,
    iis: bool = True,
    neutral: Union[str, Number] = "zero",
    est_freq: str = "D",
    thresh: float = None,
    pan_weight: float = 1,
    postfix: str = "ZN",
) -> pd.DataFrame:
    """
    Computes z-scores for a panel around a neutral level ("zn scores").

    Parameters
    ----------
    df : ~pandas.Dataframe
        standardized JPMaQS DataFrame with the necessary columns: 'cid', 'xcat',
        'real_date' and 'value'.
    xcat : str
        extended category for which the zn_score is calculated.
    cids : List[str]
        cross sections for which zn_scores are calculated; default is all available for
        category.
    start : str
        earliest date in ISO format. Default is None and earliest date in df is used.
    end : str
        latest date in ISO format. Default is None and latest date in df is used.
    blacklist : dict
        cross-sections with date ranges that should be excluded from the calculation of
        zn-scores. This means that not only are there no zn-score values calculated for
        these periods, but also that they are not used for the scoring of other periods.
    sequential : bool
        if True (default) score parameters (neutral level and mean absolute deviation)
        are estimated sequentially with concurrently available information only.
    min_obs : int
        the minimum number of observations required to calculate zn_scores. Default is
        261. The parameter is only applicable if the "sequential" parameter is set to True.
        Otherwise the neutral level and the mean absolute deviation are both computed in-
        sample and will use the full sample.
    iis : bool
        if True (default) zn-scores are also calculated for the initial sample period
        defined by min-obs on an in-sample basis to avoid losing history. This is irrelevant
        if sequential is set to False.
    neutral : str, Number
        method to determine neutral level. Default is 'zero'. Alternatives are 'mean',
        'median' or a number.
    est_freq : str
        the frequency at which mean absolute deviations or means are are re-estimated.
        The options are daily, weekly, monthly & quarterly: "D", "W", "M", "Q". Default is
        daily. Re-estimation is performed at period end.
    thresh : float
        threshold value beyond which scores are winsorized, i.e. contained at that
        threshold. The threshold is the maximum absolute score value that the function is
        allowed to produce. The minimum threshold is 1 mean absolute deviation.
    pan_weight : float
        weight of panel (versus individual cross section) for calculating the z-score
        parameters, i.e. the neutral level and the mean absolute deviation. Default is 1,
        i.e. panel data are the basis for the parameters. Lowest possible value is 0, i.e.
        parameters are all specific to cross section.
    postfix : str
        string appended to category name for output; default is "ZN".

    Returns
    -------
    ~pandas.Dataframe
        standardized DataFrame with the zn-scores of the chosen category: 'cid', 'xcat',
        'real_date' and 'value'.


    .. note::
        The blacklist argument is a dictionary with cross-sections as keys and tuples of
        start and end dates of the blacklist periods in ISO formats as values. If one cross
        section has multiple blacklist periods, numbers are added to the keys (i.e. TRY_1,
        TRY_2, etc.)
    """

    expected_columns = ["cid", "xcat", "real_date", "value"]
    df = QuantamentalDataFrame(df[expected_columns])

    # --- Assertions
    err: str = (
        "The `neutral` parameter must be a number or a string with value,"
        " either 'mean', 'median' or 'zero'."
    )
    if not isinstance(neutral, Number):
        if not isinstance(neutral, str):
            raise TypeError(err)
        elif neutral not in ["mean", "median", "zero"]:
            raise ValueError(err)

    if thresh is not None:
        err: str = "The `thresh` parameter must a numerical value >= 1.0."
        if not isinstance(thresh, Number):
            raise TypeError(err)
        elif thresh < 1.0:
            raise ValueError(err)

    if not isinstance(iis, bool):
        raise TypeError("Parameter `iis` must be a boolean.")

    err = (
        "The `pan_weight` parameter must be a numerical value between 0 and 1 "
        "(inclusive)."
    )
    if not isinstance(pan_weight, Number):
        raise TypeError(err)
    elif not (0 <= pan_weight <= 1):
        raise ValueError(err)

    error_min = "Minimum observations must be a non-negative Integer value."
    if not isinstance(min_obs, int):
        raise TypeError(error_min)
    if min_obs < 0:
        raise ValueError(error_min)

    est_freq = _map_to_business_day_frequency(
        freq=est_freq, valid_freqs=["D", "W", "M", "Q"]
    )

    # --- Prepare re-estimation dates and time-series DataFrame.

    # Remove any additional metrics defined in the DataFrame.
    if cids is not None:
        missing_cids = set(cids).difference(set(df["cid"]))
        if missing_cids:
            raise ValueError(
                f"The following cids are not available in the DataFrame: "
                f"{missing_cids}."
            )
    if xcat not in df["xcat"].unique():
        raise ValueError(f"The xcat {xcat} is not available in the DataFrame.")

    df = reduce_df(
        df, xcats=[xcat], cids=cids, start=start, end=end, blacklist=blacklist
    )

    if df.isna().values.any():
        df = drop_nan_series(df=df, raise_warning=True)

    s_date = min(df["real_date"])
    e_date = max(df["real_date"])
    dates_iter = pd.date_range(start=s_date, end=e_date, freq=est_freq)

    dfw = df.pivot(index="real_date", columns="cid", values="value")
    cross_sections = dfw.columns

    # --- The actual scoring.

    dfw_zns_pan = dfw * 0
    dfw_zns_css = dfw * 0

    if dfw.shape[0] < min_obs and pan_weight < 1 and pan_weight > 0:
        raise ValueError(
            f"The DataFrame has less than {min_obs} observations. "
            "Please adjust the `min_obs` parameter."
        )

    if pan_weight > 0:
        df_neutral = expanding_stat(
            dfw,
            dates_iter,
            stat=neutral,
            sequential=sequential,
            min_obs=min_obs,
            iis=iis,
        )
        dfx = dfw.sub(df_neutral["value"], axis=0)
        df_mabs = expanding_stat(
            dfx.abs(),
            dates_iter,
            stat="mean",
            sequential=sequential,
            min_obs=min_obs,
            iis=iis,
        )
        dfw_zns_pan = dfx.div(df_mabs["value"], axis="rows")

    if pan_weight < 1:
        for cid in cross_sections:
            dfi = dfw[cid]

            df_neutral = expanding_stat(
                dfi.to_frame(name=cid),
                dates_iter,
                stat=neutral,
                sequential=sequential,
                min_obs=min_obs,
                iis=iis,
            )
            dfx = dfi - df_neutral["value"]

            df_mabs = expanding_stat(
                dfx.abs().to_frame(name=cid),
                dates_iter,
                stat="mean",
                sequential=sequential,
                min_obs=min_obs,
                iis=iis,
            )
            dfx = pd.DataFrame(data=dfx.to_numpy(), index=dfx.index, columns=["value"])
            dfx = dfx.rename_axis("cid", axis=1)

            zns_css_df = dfx / df_mabs
            dfw_zns_css.loc[:, cid] = zns_css_df["value"]

    dfw_zns = (dfw_zns_pan * pan_weight) + (dfw_zns_css * (1 - pan_weight))
    dfw_zns = dfw_zns.dropna(axis=0, how="all")

    if thresh is not None:
        dfw_zns.clip(lower=-thresh, upper=thresh, inplace=True)

    # --- Reformatting of output into standardised DataFrame.

    df_out = dfw_zns.stack().to_frame("value").reset_index()
    df_out = QuantamentalDataFrame.from_long_df(
        df=df_out,
        xcat=xcat + postfix,
        categorical=df.InitializedAsCategorical,
    )

    return df_out


def expanding_stat(
    df: pd.DataFrame,
    dates_iter: pd.DatetimeIndex,
    stat: Union[str, Number] = "mean",
    sequential: bool = True,
    min_obs: int = 261,
    iis: bool = True,
) -> pd.DataFrame:
    """
    Compute specified statistic based on an expanding sample.

    Parameters
    ----------
    df : ~pandas.Dataframe
        Daily-frequency time series DataFrame.
    dates_iter : ~pandas.DatetimeIndex
        controls the frequency of the neutral & mean absolute deviation calculations.
    stat : str, Number
        statistical method to be applied. This is typically 'mean', or 'median'.
    sequential : bool
        if True (default) the statistic is estimated sequentially. If this set to false
        a single value is calculated per time series, based on the full sample.
    min_obs : int
        minimum required observations for calculation of the statistic in days.
    iis : bool
        if set to True, the values of the initial interval determined by min_obs will be
        estimated in-sample, based on the full initial sample.

    Returns
    -------
    ~pandas.DataFrame
        Time series dataframe of the chosen statistic across all columns
    """

    df_out = pd.DataFrame(np.nan, index=df.index, columns=["value"])
    # An adjustment for individual series' first realised value is not required given the
    # returned DataFrame will be subtracted from the original DataFrame. The original
    # DataFrame will implicitly host this information through NaN values such that when
    # the arithmetic operation is made, any falsified values will be displaced by NaN
    # values.

    first_observation = df.dropna(axis=0, how="all").index[0]
    # Adjust for individual cross-sections' series commencing at different dates.
    first_estimation = df.dropna(axis=0, how="all").index[min_obs]

    obs_index = np.where(df.index == first_observation)[0][0]
    est_index = np.where(df.index == first_estimation)[0][0]

    if stat == "zero":
        df_out["value"] = 0

    elif isinstance(stat, Number):
        df_out["value"] = stat

    elif not sequential:
        # The entire series is treated as in-sample. Will automatically handle NaN
        # values.
        statval = df.stack().apply(stat)
        df_out["value"] = statval

    else:
        dates = dates_iter[dates_iter >= first_estimation]
        if stat == "mean":
            expanding_count = _get_expanding_count(
                df.loc[first_observation:], min_periods=min_obs + 1
            )
            df_mean = (
                df.loc[first_observation:]
                .sum(1)
                .expanding(min_periods=min_obs + 1)
                .sum()
                / expanding_count
            )
            df_mean = df_mean.dropna().loc[dates]
            df_mean.name = "value"
            df_out.update(df_mean)
        else:
            for date in dates:
                df_out.loc[date, "value"] = (
                    df.loc[first_observation:date].stack().apply(stat)
                )

        df_out = df_out.ffill()

        if iis and (est_index - obs_index) > 0:
            df_out = df_out.bfill(limit=int(est_index - obs_index))

    df_out.columns.name = "cid"
    return df_out


def _get_expanding_count(X: pd.DataFrame, min_periods: int = 1):
    """
    Helper method to get the number of non-NaN values in each expanding window.

    Parameters
    ----------
    X : ~pandas.DataFrame
        Pandas dataframe of input features.
    min_periods : int
        Minimum number of observations in window required to have a value (otherwise
        result is 0.).

    Returns
    -------
    ~numpy.ndarray
        Numpy array of expanding counts.
    """

    return X.expanding(min_periods).count().sum(1).to_numpy()


if __name__ == "__main__":
    np.random.seed(1)

    cids = ["AUD", "CAD", "GBP", "USD", "NZD"]
    xcats = ["XR", "CRY", "GROWTH", "INFL"]

    df_cids = pd.DataFrame(
        index=cids, columns=["earliest", "latest", "mean_add", "sd_mult"]
    )

    df_cids.loc["AUD"] = ["2010-01-01", "2020-12-31", 0.5, 2]
    df_cids.loc["CAD"] = ["2006-01-01", "2020-11-30", 0, 1]
    df_cids.loc["GBP"] = ["2008-01-01", "2020-11-30", -0.2, 0.5]
    df_cids.loc["USD"] = ["2007-01-01", "2020-09-30", -0.2, 0.5]
    df_cids.loc["NZD"] = ["2002-01-01", "2020-09-30", -0.1, 2]

    df_xcats = pd.DataFrame(
        index=xcats,
        columns=["earliest", "latest", "mean_add", "sd_mult", "ar_coef", "back_coef"],
    )
    df_xcats.loc["XR"] = ["2008-01-01", "2020-12-31", 0, 1, 0, 0.3]
    df_xcats.loc["CRY"] = ["2011-01-01", "2020-10-30", 1, 2, 0.9, 0.5]
    df_xcats.loc["GROWTH"] = ["2012-01-01", "2020-10-30", 1, 2, 0.9, 1]
    df_xcats.loc["INFL"] = ["2013-01-01", "2020-10-30", 1, 2, 0.8, 0.5]

    # Apply a blacklist period from series' start date.
    black = {"AUD": ["2010-01-01", "2013-12-31"], "GBP": ["2018-01-01", "2100-01-01"]}

    dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)
    dfd["grading"] = np.ones(dfd.shape[0])

    # Monthly: panel + cross.
    dfzm = make_zn_scores(
        dfd,
        xcat="XR",
        sequential=True,
        cids=cids,
        blacklist=black,
        iis=True,
        neutral="mean",
        pan_weight=0.75,
        min_obs=261,
        est_freq="m",
    )
    print(dfzm)

    # Weekly: panel + cross.
    dfzw = make_zn_scores(
        dfd,
        xcat="XR",
        sequential=True,
        cids=cids,
        blacklist=black,
        iis=False,
        neutral="mean",
        pan_weight=0.5,
        min_obs=261,
        est_freq="w",
    )

    # Daily: panel. Neutral and mean absolute deviation will be computed daily.
    dfzd = make_zn_scores(
        dfd,
        xcat="XR",
        sequential=True,
        cids=cids,
        blacklist=black,
        iis=True,
        neutral="mean",
        pan_weight=1.0,
        min_obs=261,
        est_freq="d",
    )

    # Daily: cross.
    dfd["ticker"] = dfd["cid"] + "_" + dfd["xcat"]
    dfzd = make_zn_scores(
        dfd,
        xcat="XR",
        sequential=True,
        cids=cids,
        blacklist=black,
        iis=True,
        neutral="mean",
        pan_weight=0.0,
        min_obs=261,
        est_freq="d",
    )

    panel_df = make_zn_scores(
        dfd,
        "CRY",
        cids,
        start="2010-01-04",
        blacklist=black,
        sequential=False,
        min_obs=0,
        neutral="mean",
        iis=True,
        thresh=None,
        pan_weight=0.75,
        postfix="ZN",
    )

    print(panel_df)

    panel_df_7 = make_zn_scores(
        dfd,
        "CRY",
        cids,
        start="2010-01-04",
        blacklist=black,
        sequential=False,
        min_obs=0,
        neutral="zero",
        iis=True,
        thresh=None,
        pan_weight=0.75,
        postfix="ZN",
    )

    print(panel_df_7)
