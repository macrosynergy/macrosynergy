"""
Module for plotting ranges of values across cross-sections for one or more categories.
"""

import pandas as pd
from typing import List, Tuple, Optional

from macrosynergy.management.simulate import make_qdf
import macrosynergy.visuals as msv


def view_ranges(
    df: pd.DataFrame,
    xcats: List[str],
    cids: Optional[List[str]] = None,
    start: str = "2000-01-01",
    end: Optional[str] = None,
    val: str = "value",
    kind: str = "bar",
    sort_cids_by: Optional[str] = None,
    title: Optional[str] = None,
    ylab: Optional[str] = None,
    size: Tuple[float] = (16, 8),
    xcat_labels: Optional[List[str]] = None,
    legend_loc: str = None,
    legend_bbox_to_anchor: Tuple[float] = None,
):
    """Plots averages and various ranges across sections for one or more categories.

    :param <pd.Dataframe> df: standardized DataFrame with the necessary columns:
        'cid', 'xcat', 'real_date' and at least one column with values of interest.
    :param <List[str]> xcats: extended categories to be checked on. Default is all
        in the DataFrame.
    :param <List[str]> cids: cross sections to plot. Default is all in DataFrame.
    :param <str> start: earliest date in ISO format. Default earliest date in df.
    :param <str> end: latest date in ISO format. Default is latest date in df.
    :param <str> val: name of column that contains the values. Default is 'value'.
    :param <str> kind: type of range plot. Default is 'bar'; other option is 'box'.
    :param <str> sort_cids_by: criterion for sorting cids on x-axis;
        Arguments can be 'mean' and 'std'. Default is None, i.e. original order. Ordering
        will be based on the first category if the category is defined over the complete
        panel. Otherwise, mean and standard deviation calculated, of the cross-sections,
        computed across all categories.
    :param <str> title: string of chart title; defaults depend on type of range plot.
    :param <str> ylab: y label. Default is no label.
    :param <Tuple[float]> size: Tuple of width and height of graph. Default is (16, 8).
    :param <List[str]> xcat_labels: custom labels to be used for the ranges.
    :param <str> legend_loc: location of legend; passed to matplotlib.pyplot.legend().
        Default is 'upper center'.
    :param <Tuple[float]> legend_bbox_to_anchor: passed to matplotlib.pyplot.legend().
        Default is (0.5, -0.15).

    """
    if legend_bbox_to_anchor is None and legend_loc is None:
        # -0.15 puts the legend below the x-labels if there are only 2 xcats.
        # If there is more than 2 xcats we need to move the legend further down by a
        # factor of 0.05 for each additional xcat.
        legend_bbox_to_anchor = (0.5, -0.15 - 0.05 * (len(xcats) - 2))
        legend_loc = "upper center"

    if legend_loc is None:
        legend_loc = "upper center"

    msv.view_ranges(
        df=df,
        xcats=xcats,
        cids=cids,
        start=start,
        end=end,
        val=val,
        kind=kind,
        sort_cids_by=sort_cids_by,
        title=title,
        ylab=ylab,
        size=size,
        xcat_labels=xcat_labels,
        legend_loc=legend_loc,
        legend_bbox_to_anchor=legend_bbox_to_anchor,
    )


if __name__ == "__main__":
    cids = ["AUD", "CAD", "GBP", "USD"]
    xcats = ["XR", "CRY"]
    df_cids = pd.DataFrame(
        index=cids, columns=["earliest", "latest", "mean_add", "sd_mult"]
    )
    df_cids.loc["AUD",] = ["2010-01-01", "2020-12-31", 0.5, 0.2]
    df_cids.loc["CAD",] = ["2011-01-01", "2020-11-30", 0, 1]
    df_cids.loc["GBP",] = ["2012-01-01", "2020-11-30", 0, 2]
    df_cids.loc["USD",] = ["2012-01-01", "2020-11-30", 1, 2]

    df_xcats = pd.DataFrame(
        index=xcats,
        columns=["earliest", "latest", "mean_add", "sd_mult", "ar_coef", "back_coef"],
    )
    df_xcats.loc["XR",] = ["2010-01-01", "2020-12-31", 0, 1, 0, 0.3]
    df_xcats.loc["CRY",] = ["2011-01-01", "2020-10-30", 1, 2, 0.9, 0.5]
    df_xcats.loc["INFL",] = ["2011-01-01", "2020-10-30", 1, 2, 0.9, 0.5]
    df_xcats.loc["GROWTH",] = ["2011-01-01", "2020-10-30", 1, 2, 0.9, 0.5]

    dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)

    view_ranges(
        dfd,
        xcats=["XR"],
        kind="box",
        start="2012-01-01",
        end="2018-01-01",
        sort_cids_by="std",
    )

    filter_1 = (dfd["xcat"] == "XR") & (dfd["cid"] == "AUD")
    dfd = dfd[~filter_1]

    view_ranges(
        dfd,
        xcats=["XR", "CRY", "INFL"],
        cids=cids,
        kind="box",
        start="2012-01-01",
        end="2018-01-01",
        sort_cids_by=None,
        xcat_labels=["EQXR_NSA", "CRY_NSA", "INFL_NSA"],
    )
