"""
Functions for visualizing data grading and blacklisted periods from a quantamental DataFrame.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple

from macrosynergy.management.simulate import make_qdf
from macrosynergy.management.utils import reduce_df


def heatmap_grades(
    df: pd.DataFrame,
    xcats: List[str],
    cids: List[str] = None,
    start: str = "2000-01-01",
    end: str = None,
    grade: str = "grading",
    title: str = None,
    size: Tuple[float] = None,
):
    """Displays a heatmap of grading

    :param <pd.Dataframe> df: standardized DataFrame with the necessary columns:
        'cid', 'xcat', 'real_date' and 'grading'.
    :param <List[str]> xcats: extended categorys to be checked on.
    :param <List[str]> cids: cross sections to visualize. Default is all in  DataFrame.
    :param <str> start: earliest date in ISO format. Default is earliest available.
    :param <str> end: latest date in ISO format. Default is latest available.
    :param <str> grade: name of column that contains the grades. Default is 'grading'.
    :param <str> title: string of chart title; defaults depend on type of range plot.
    :param <Tuple[float]> size: Tuple of width and height of graph.
        Default is None, meaning it is set in accordance with df.

    """
    df["real_date"] = pd.to_datetime(df["real_date"], format="%Y-%m-%d")

    df_cols = list(df.columns)
    grade_error = (
        "Column that contains the grade values must be present in the "
        f"DataFrame: {df_cols}."
    )
    assert grade in df.columns, grade_error

    df, xcats, cids = reduce_df(df, xcats, cids, start, end, out_all=True)
    df = df[["xcat", "cid", "real_date", grade]]
    df[grade] = df[grade].astype(float).round(2)

    df_ags = (
        df.groupby(["xcat", "cid"])
        .mean()
        .reset_index()
        .pivot(index="xcat", columns="cid", values=grade)
    )

    if size is None:
        size = (max(df_ags.shape[0] / 2, 15), max(1, df_ags.shape[1] / 2))
    if title is None:
        sdate = df["real_date"].min().strftime("%Y-%m-%d")
        title = f"Average grade of vintages since {sdate}"
    sns.set(rc={"figure.figsize": size})
    sns.heatmap(
        df_ags,
        cmap="YlOrBr",
        vmin=1,
        vmax=3,
        annot=True,
        fmt=".1f",
        linewidth=1,
        cbar=False,
    )
    plt.xlabel("")
    plt.ylabel("")
    plt.title(title, fontsize=18)
    plt.show()


if __name__ == "__main__":
    cids = ["NZD", "AUD", "CAD", "GBP"]
    xcats = ["XR", "CRY", "GROWTH", "INFL"]
    df_cids = pd.DataFrame(
        index=cids, columns=["earliest", "latest", "mean_add", "sd_mult"]
    )
    df_cids.loc["AUD",] = ["2000-01-01", "2020-12-31", 0.1, 1]
    df_cids.loc["CAD",] = ["2001-01-01", "2020-11-30", 0, 1]
    df_cids.loc["GBP",] = ["2002-01-01", "2020-11-30", 0, 2]
    df_cids.loc["NZD",] = ["2002-01-01", "2020-09-30", -0.1, 2]

    df_xcats = pd.DataFrame(
        index=xcats,
        columns=["earliest", "latest", "mean_add", "sd_mult", "ar_coef", "back_coef"],
    )
    df_xcats.loc["XR",] = ["2000-01-01", "2020-12-31", 0.1, 1, 0, 0.3]
    df_xcats.loc["CRY",] = ["2000-01-01", "2020-10-30", 1, 2, 0.95, 1]
    df_xcats.loc["GROWTH",] = ["2001-01-01", "2020-10-30", 1, 2, 0.9, 1]
    df_xcats.loc["INFL",] = ["2001-01-01", "2020-10-30", 1, 2, 0.8, 0.5]

    dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)

    dfd["grading"] = "3"
    filter_date = dfd["real_date"] >= pd.to_datetime("2010-01-01")
    filter_cid = dfd["cid"].isin(["NZD", "AUD"])
    dfd.loc[filter_date & filter_cid, "grading"] = "1"
    filter_date = dfd["real_date"] >= pd.to_datetime("2013-01-01")
    filter_xcat = dfd["xcat"].isin(["CRY", "GROWTH"])
    dfd.loc[filter_date & filter_xcat, "grading"] = "2.1"
    filter_xcat = dfd["xcat"] == "XR"
    dfd.loc[filter_xcat, "grading"] = 1

    heatmap_grades(dfd, xcats=["CRY", "GROWTH", "INFL"], cids=cids)

    dfd.info()
