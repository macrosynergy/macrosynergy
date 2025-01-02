"""
Functionality to visualize time series data as line charts.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

from macrosynergy.management.simulate import make_qdf

import macrosynergy.visuals as msv


def view_timelines(
    df: pd.DataFrame,
    xcats: Optional[List[str]] = None,
    cids: Optional[List[str]] = None,
    intersect: bool = False,
    val: str = "value",
    cumsum: bool = False,
    start: str = "2000-01-01",
    end: Optional[str] = None,
    ncol: int = 3,
    legend_ncol: int = 1,
    same_y: bool = True,
    all_xticks: bool = False,
    xcat_grid: bool = False,
    xcat_labels: Optional[List[str]] = None,
    single_chart: bool = False,
    label_adj: float = 0.05,
    title: Optional[str] = None,
    title_adj: float = 1.005,
    title_xadj: float = 0.5,
    title_fontsize: int = 18,
    cs_mean: bool = False,
    size: Tuple[float, float] = (12, 7),
    aspect: float = 1.618,
    height: float = 2.85,
    legend_fontsize: int = 12,
    blacklist: Dict = None,
):
    """
    Displays a grid with subplots of time line charts of one or more categories.

    Parameters
    ----------
    df : ~pandas.Dataframe
        standardized DataFrame with the necessary columns: 'cid', 'xcat', 'real_date'
        and at least one column with values of interest.
    xcats : List[str]
        extended categories to plot. Default is all in DataFrame.
    cids : List[str]
        cross sections to plot. Default is all in DataFrame. If this contains only one
        cross section a single line chart is created.
    intersect : bool
        if True only retains cross-sections that are available for all categories.
        Default is False.
    val : str
        name of column that contains the values of interest. Default is 'value'.
    cumsum : bool
        plot cumulative sum of the values over time. Default is False.
    start : str
        earliest date in ISO format. Default is earliest date available.
    end : str
        latest date in ISO format. Default is latest date available.
    ncol : int
        number of columns in facet grid. Default is 3.
    legend_ncol : int
        number of columns in legend. Default is 1.
    same_y : bool
        if True (default) all plots in facet grid share same y axis.
    all_xticks : bool
        if True x-axis tick labels are added to all plots in grid. Default is False, i.e
        only the lowest row displays the labels.
    xcat_grid : bool
        if True, shows a facet grid of line charts for each category for a single cross
        section. Default is False, only one cross section is allowed with this option.
    xcat_labels : List[str]
        labels to be used for xcats. If not defined, the labels will be identical to
        extended categories.
    single_chart : bool
        if True, all lines are plotted in a single chart.
    title : str
        chart heading. Default is no title.
    title_adj : float
        parameter that sets top of figure to accommodate title. Default is 0.95.
    title_xadj : float
        parameter that sets x position of title. Default is 0.5.
    title_fontsize : int
        font size of title. Default is 16.
    label_adj : float
        parameter that sets bottom of figure to fit the label. Default is 0.05.
    cs_mean : bool
        if True this adds a line of cross-sectional averages to the line charts. This is
        only allowed for function calls with a single category. Default is False.
    size : Tuple[float]
        two-element tuple setting width/height of single cross section plot. Default is
        (12, 7). This is irrelevant for facet grid.
    aspect : float
        width-height ratio for plots in facet. Default is 1.7.
    height : float
        height of plots in facet. Default is 3.
    legend_fontsize : int
        font size of legend. Default is 12.
    blacklist : dict
        cross-sections with date ranges that should be excluded from the dataframe.
    """

    msv.timelines(
        df=df,
        xcats=xcats,
        cids=cids,
        intersect=intersect,
        val=val,
        cumsum=cumsum,
        start=start,
        end=end,
        ncol=ncol,
        same_y=same_y,
        all_xticks=all_xticks,
        xcat_grid=xcat_grid,
        xcat_labels=xcat_labels,
        single_chart=single_chart,
        title=title,
        legend_ncol=legend_ncol,
        cs_mean=cs_mean,
        title_fontsize=title_fontsize,
        legend_fontsize=legend_fontsize,
        # HC Params
        aspect=aspect,
        title_adj=title_adj,
        height=height,
        size=size,
        label_adj=label_adj,
        title_xadj=title_xadj,
        blacklist=blacklist,
    )


if __name__ == "__main__":
    cids = ["AUD", "CAD", "GBP", "NZD"]
    xcats = ["XR", "CRY", "INFL", "FXXR"]
    df_cids = pd.DataFrame(
        index=cids, columns=["earliest", "latest", "mean_add", "sd_mult"]
    )
    df_cids.loc["AUD"] = ["2010-01-01", "2020-12-31", 0.2, 0.2]
    df_cids.loc["CAD"] = ["2011-01-01", "2020-11-30", 0, 1]
    df_cids.loc["GBP"] = ["2012-01-01", "2020-11-30", 0, 2]
    df_cids.loc["NZD"] = ["2012-01-01", "2020-09-30", -0.1, 3]

    df_xcats = pd.DataFrame(
        index=xcats,
        columns=["earliest", "latest", "mean_add", "sd_mult", "ar_coef", "back_coef"],
    )

    df_xcats.loc["XR"] = ["2010-01-01", "2020-12-31", 0.1, 1, 0, 0.3]
    df_xcats.loc["INFL"] = ["2015-01-01", "2020-12-31", 0.1, 1, 0, 0.3]
    df_xcats.loc["CRY"] = ["2013-01-01", "2020-10-30", 1, 2, 0.95, 0.5]
    df_xcats.loc["FXXR"] = ["2013-01-01", "2020-10-30", 1, 2, 0.95, 0.5]

    dfd: pd.DataFrame = make_qdf(df_cids, df_xcats, back_ar=0.75)
    # sort by cid, xcat, and real_date
    dfd = dfd.sort_values(["cid", "xcat", "real_date"])
    ctr: int = -1
    for xcat in xcats[:2]:
        for cid in cids[:2]:
            ctr *= -1
            mask = (dfd["cid"] == cid) & (dfd["xcat"] == xcat)
            dfd.loc[mask, "value"] = (
                10
                * ctr
                * np.arange(dfd.loc[mask, "value"].shape[0])
                / (dfd.loc[mask, "value"].shape[0] - 1)
            )

    dfdx = dfd[~((dfd["cid"] == "AUD") & (dfd["xcat"] == "XR"))]

    view_timelines(
        dfd,
        xcats=["XR", "CRY"],
        cids=cids[0],
        size=(10, 5),
        title="AUD Return and Carry",
        aspect=3,
    )

    view_timelines(
        dfd,
        xcats=["XR", "CRY", "INFL"],
        cids=cids[0],
        xcat_grid=True,
        title_adj=0.8,
        xcat_labels=["Return", "Carry", "Inflation"],
        title="AUD Return, Carry & Inflation",
        aspect=3,
    )

    view_timelines(dfd, xcats=["CRY"], cids=cids, ncol=4, title="Carry", cs_mean=True)

    view_timelines(
        dfd, xcats=["XR"], cids=cids[:2], ncol=2, cumsum=True, same_y=False, aspect=2
    )

    dfd = dfd.set_index("real_date")
    view_timelines(
        dfd,
        xcats=["XR"],
        cids=cids[:2],
        ncol=2,
        cumsum=True,
        same_y=False,
        aspect=2,
        single_chart=True,
    )
