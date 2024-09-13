"""
Module for checking the availability of data availabity from a 
Quantamental DataFrame. Includes functions for checking start years
and end dates of a DataFrame, as well as visualizing the results.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple

from macrosynergy.management.simulate import make_qdf
from macrosynergy.management.utils import reduce_df
from macrosynergy.management.types import QuantamentalDataFrame
import macrosynergy.visuals as msv


def check_availability(
    df: pd.DataFrame,
    xcats: List[str] = None,
    cids: List[str] = None,
    start: str = None,
    start_size: Tuple[float] = None,
    end_size: Tuple[float] = None,
    start_years: bool = True,
    missing_recent: bool = True,
    use_last_businessday: bool = True,
):
    """
    Wrapper for visualizing start and end dates of a filtered DataFrame.

    :param <pd.DataFrame> df: standardized DataFrame with the following necessary
        columns: 'cid', 'xcat', 'real_date'.
    :param <List[str]> xcats: extended categories to be checked on.
        Default is all in the DataFrame.
    :param <List[str]> cids: cross sections to be checked on.
        Default is all in the DataFrame.
    :param <str> start: string representing earliest considered date. Default is None.
    :param <Tuple[float]> start_size: tuple of floats with width / length of
        the start years heatmap. Default is None (format adjusted to data).
    :param <Tuple[float]> end_size: tuple of floats with width/length of
        the end dates heatmap. Default is None (format adjusted to data).
    :param <bool> start_years: boolean indicating whether or not to display a chart
        of starting years for each cross-section and indicator.
        Default is True (display start years).
    :param <bool> missing_recent: boolean indicating whether or not to display a chart
        of missing date numbers for each cross-section and indicator.
        Default is True (display missing days).
    :param <bool> use_last_businessday: boolean indicating whether or not to use the
        last business day before today as the end date. Default is True.
    """
    if not isinstance(start_years, bool):
        raise TypeError(f"<bool> object expected and not {type(start_years)}.")
    if not isinstance(missing_recent, bool):
        raise TypeError(f"<bool> object expected and not {type(missing_recent)}.")

    dfx = reduce_df(df, xcats=xcats, cids=cids, start=start)
    if dfx.empty:
        raise ValueError(
            "No data available for the selected cross-sections and categories."
        )
    if start_years:
        dfs = check_startyears(dfx)
        visual_paneldates(
            dfs, size=start_size, use_last_businessday=use_last_businessday
        )
    if missing_recent:
        dfe = check_enddates(dfx)
        visual_paneldates(dfe, size=end_size, use_last_businessday=use_last_businessday)


def missing_in_df(
    df: QuantamentalDataFrame,
    xcats: List[str] = None,
    cids: List[str] = None,
):
    """
    Print missing cross-sections and categories

    :param <QuantamentalDataFrame> df: standardized DataFrame with the following necessary
        columns: 'cid', 'xcat', 'real_date'.
    :param <List[str]> xcats: extended categories to be checked on. Default is all
        in the DataFrame.
    :param <List[str]> cids: cross sections to be checked on. Default is all in
        the DataFrame.

    """
    if not isinstance(df, QuantamentalDataFrame):
        raise TypeError("`df` must be a QuantamentalDataFrame/pd.DataFrame")

    if df.empty:
        raise ValueError("`df` is empty.")

    for lst, name in zip([xcats, cids], ["xcats", "cids"]):
        if (lst is not None) and not (
            isinstance(lst, list) and all(isinstance(x, str) for x in lst)
        ):
            raise TypeError(f"`{name}` should be a `List[str]` and not {type(lst)}.")

    missing_across_df = list(set(xcats) - set(df["xcat"]))
    if len(missing_across_df) > 0:
        print("Missing XCATs across DataFrame: ", missing_across_df)
    else:
        print("No missing XCATs across DataFrame.")

    cids = df["cid"].unique() if cids is None else cids
    xcats_used = sorted(list(set(xcats).intersection(set(df["xcat"]))))
    if len(xcats_used) == 0:
        print("No XCATs found in the DataFrame.")
        return

    max_xcat_len = max(map(len, xcats_used))
    for xcat in xcats_used:
        cids_xcat = df.loc[df["xcat"] == xcat, "cid"].unique()
        missing_cids = sorted(set(cids) - set(cids_xcat))
        msg = f"Missing cids for {xcat}: " + " " * (max_xcat_len - len(xcat))
        print(msg, missing_cids)


def check_startyears(df: pd.DataFrame):
    """
    DataFrame with starting years across all extended categories and cross-sections

    :param <pd.DataFrame> df: standardized DataFrame with the following necessary
        columns: 'cid', 'xcat', 'real_date'.

    """
    df: pd.DataFrame = df.copy()
    df = df.dropna(how="any")
    df_starts = df[["cid", "xcat", "real_date"]].groupby(["cid", "xcat"]).min()
    df_starts["real_date"] = pd.DatetimeIndex(df_starts.loc[:, "real_date"]).year

    return df_starts.unstack().loc[:, "real_date"].astype(int, errors="ignore")


def check_enddates(df: pd.DataFrame) -> pd.DataFrame:
    """
    DataFrame with end dates across all extended categories and cross sections.

    :param <pd.DataFrame> df: standardized DataFrame with the following necessary
        columns: 'cid', 'xcat', 'real_date'.
    """
    df: pd.DataFrame = df.copy()
    df = df.dropna(how="any")
    df_ends = df[["cid", "xcat", "real_date"]].groupby(["cid", "xcat"]).max()
    df_ends["real_date"] = df_ends["real_date"].dt.strftime("%Y-%m-%d")

    return df_ends.unstack().loc[:, "real_date"]


def business_day_dif(df: pd.DataFrame, maxdate: pd.Timestamp):
    """
    Number of business days between two respective business dates.

    :param <pd.DataFrame> df: DataFrame cross-sections rows and category columns. Each
        cell in the DataFrame will correspond to the start date of the respective series.
    :param <pd.Timestamp> maxdate: maximum release date found in the received DataFrame.
        In principle, all series should have values up until the respective business
        date. The difference will represent possible missing values.

    :return <pd.DataFrame>: DataFrame consisting of business day differences for all
        series.

    """
    year_df = (maxdate.year - df.apply(lambda x: x.dt.isocalendar().year)) * 52
    week_df = maxdate.week - df.apply(lambda x: x.dt.isocalendar().week)
    # Account for difference over a year.
    week_df += year_df
    # Account for weekends.
    week_df *= 2
    df = (maxdate - df).apply(lambda x: x.dt.days)
    # set to zero if the difference is negative.
    df = df - week_df
    return df.where(df >= 0, 0)


def visual_paneldates(
    df: pd.DataFrame, size: Tuple[float] = None, use_last_businessday: bool = True
):
    """
    Visualize panel dates with color codes.

    :param <pd.DataFrame> df: DataFrame cross sections rows and category columns.
    :param <Tuple[float]> size: tuple of floats with width/length of displayed heatmap.
    :param <bool> use_last_businessday: boolean indicating whether or not to use the
        last business day before today as the end date. Default is True.

    """
    msv.view_panel_dates(df=df, size=size, use_last_businessday=use_last_businessday)


if __name__ == "__main__":
    cids = ["AUD", "CAD", "GBP"]
    xcats = ["XR", "CRY"]

    cols_1 = ["earliest", "latest", "mean_add", "sd_mult"]
    df_cids = pd.DataFrame(index=cids, columns=cols_1)
    df_cids.loc["AUD",] = ["2010-01-01", "2020-12-31", 0.5, 2]
    df_cids.loc["CAD",] = ["2010-01-01", "2020-11-30", 0, 1]
    df_cids.loc["GBP",] = ["2012-01-01", "2020-11-30", -0.2, 0.5]

    cols_2 = cols_1 + ["ar_coef", "back_coef"]
    df_xcats = pd.DataFrame(index=xcats, columns=cols_2)
    df_xcats.loc["XR",] = ["2010-01-01", "2020-12-31", 0, 1, 0, 0.3]
    df_xcats.loc["CRY",] = ["2011-01-01", "2020-10-30", 1, 2, 0.9, 0.5]

    dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)

    filt_na = (dfd["cid"] == "CAD") & (dfd["real_date"] < "2011-01-01")
    dfd.loc[filt_na, "value"] = np.nan

    xxcats = xcats + ["TREND"]
    xxcids = cids + ["USD"]

    check_availability(
        df=dfd, xcats=xcats, cids=cids, start_size=(10, 5), end_size=(10, 8)
    )
