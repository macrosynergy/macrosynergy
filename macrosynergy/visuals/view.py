"""
Common functions for visualizing data.
These functions make use of the classes in the `macrosynergy.visuals` module.
"""
import os, sys

sys.path.append(os.getcwd())

from typing import Dict, List, Optional, Tuple, Union
from macrosynergy.visuals import Plotter, LinePlot, FacetPlot

import pandas as pd


def timelines(
    df: pd.DataFrame,
    xcats: Optional[List[str]] = None,
    cids: Optional[List[str]] = None,
    intersect: bool = False,
    val: str = "value",
    cumsum: bool = False,
    start: str = "2000-01-01",
    end: Optional[str] = None,
    ncol: int = 3,
    square_grid: bool = False,
    legend_ncol: int = 1,
    same_y: bool = True,
    all_xticks: bool = False,
    xcat_grid: bool = False,
    xcat_labels: Optional[List[str]] = None,
    single_chart: bool = False,
    label_adj: float = 0.05,
    title: Optional[str] = None,
    title_adj: float = 0.95,
    title_xadj: float = 0.5,
    title_fontsize: int = 16,
    cs_mean: bool = False,
    size: Tuple[float, float] = (12, 7),
    aspect: float = 1.7,
    height: float = 3.0,
    legend_fontsize: int = 12,
):
    """Displays a facet grid of time line charts of one or more categories.

    :param <pd.Dataframe> df: standardized DataFrame with the necessary columns:
        'cid', 'xcats', 'real_date' and at least one column with values of interest.
    :param <List[str]> xcats: extended categories to plot. Default is all in DataFrame.
    :param <List[str]> cids: cross sections to plot. Default is all in DataFrame.
        If this contains only one cross section a single line chart is created.
    :param <bool> intersect: if True only retains cids that are available for all xcats.
        Default is False.
    :param <str> val: name of column that contains the values of interest.
        Default is 'value'.
    :param <bool> cumsum: plot cumulative sum of the values over time. Default is False.
    :param <str> start: earliest date in ISO format. Default is earliest date available.
    :param <str> end: latest date in ISO format. Default is latest date available.
    :param <int> ncol: number of columns in facet grid. Default is 3.
    :param <int> legend_ncol: number of columns in legend. Default is 1.
    :param <bool> same_y: if True (default) all plots in facet grid share same y axis.
    :param <bool> all_xticks:  if True x-axis tick labels are added to all plots in grid.
        Default is False, i.e only the lowest row displays the labels.
    :param <bool> xcat_grid: if True, shows a facet grid of line charts for each xcat
        for a single cross section. Default is False, only one cross section is allowed
        with this option.
    :param <List[str]> xcat_labels: labels to be used for xcats. If not defined, the
        labels will be identical to extended categories.
    :param <bool> single_chart: if True, all lines are plotted in a single chart.
    :param <str> title: chart heading. Default is no title.
    :param <float> title_adj: parameter that sets top of figure to accommodate title.
        Default is 0.95.
    :param <float> title_xadj: parameter that sets x position of title. Default is 0.5.
    :param <int> title_fontsize: font size of title. Default is 16.
    :param <float> label_adj: parameter that sets bottom of figure to fit the label.
        Default is 0.05.
    :param <bool> cs_mean: if True this adds a line of cross-sectional averages to
        the line charts. This is only allowed for function calls with a single
        category. Default is False.
    :param <Tuple[float]> size: two-element tuple setting width/height of single cross
        section plot. Default is (12, 7). This is irrelevant for facet grid.
    :param <float> aspect: width-height ratio for plots in facet. Default is 1.7.
    :param <float> height: height of plots in facet. Default is 3.
    :param <int> legend_fontsize: font size of legend. Default is 12.

    """

    df: pd.DataFrame = df.copy()

    if isinstance(xcats, str):
        xcats: List[str] = [xcats]
    if isinstance(cids, str):
        cids: List[str] = [cids]

    if xcat_grid and single_chart:
        raise ValueError(
            "`xcat_grid` and `single_chart` cannot be True simultaneously."
        )
    # if not

    if cs_mean and xcat_grid:
        raise ValueError("`cs_mean` requires `xcat_grid` to be False.")

    if xcats is None:
        if xcat_labels:
            raise ValueError("`xcat_labels` requires `xcats` to be defined.")
        xcats: List[str] = df["xcats"].unique().tolist()

    if cids is None:
        cids: List[str] = df["cid"].unique().tolist()

    if cumsum:
        df[val] = (
            df.sort_values(["cid", "xcat", "real_date"])[["cid", "xcat", val]]
            .groupby(["cid", "xcat"])
            .cumsum()
        )

    cross_mean_series: Optional[str] = f"mean_{xcats[0]}" if cs_mean else None
    if cs_mean:
        if len(xcats) > 1:
            raise ValueError("`cs_mean` cannot be True for multiple categories.")

        if len(cids) == 1:
            raise ValueError("`cs_mean` cannot be True for a single cross section.")

        df_mean: pd.DataFrame = (
            df.groupby(["real_date", "xcat"])[val].mean(numeric_only=True).reset_index()
        )

        df_mean["cid"] = "mean"
        df: pd.DataFrame = pd.concat([df, df_mean], axis=0)
        # Drop to save memory
        df_mean: pd.DataFrame = pd.DataFrame()

    if xcat_labels:
        if (len(xcat_labels) != len(xcats)) or (
            cs_mean and (len(xcat_labels) != len(xcats) - 1)
        ):
            raise ValueError(
                "`xcat_labels` must have same length as `xcats` "
                "(or one extra label if `cs_mean` is True)."
            )
        df["xcat"] = df["xcat"].map(dict(zip(xcats, xcat_labels)))
        xcats: List[str] = xcat_labels.copy()

    if xcat_grid:
        with FacetPlot(
            df=df,
            xcats=xcats,
            cids=cids,
            intersect=intersect,
            metrics=[val],
            tickers=[cross_mean_series] if cs_mean else None,
            start=start,
            end=end,
        ) as fp:
            fp.lineplot(
                share_y=same_y,
                figsize=size,
                xcat_grid=True,
                # legend_labels=xcat_labels or None,
                facet_titles=xcat_labels or None,
                title=title,
                title_yadjust=title_adj,
                title_xadjust=title_xadj,
                compare_series=cross_mean_series if cs_mean else None,
                title_fontsize=title_fontsize,
                ncols=ncol,
                attempt_square=square_grid,
                legend_ncol=legend_ncol,
                legend_fontsize=legend_fontsize,
            )

    elif single_chart:
        with LinePlot(
            df=df,
            cids=cids,
            xcats=xcats,
            intersect=intersect,
            metrics=[val],
            tickers=[cross_mean_series] if cs_mean else None,
            start=start,
            end=end,
        ) as lp:
            lp.plot(
                figsize=size,
                title=title,
                title_yadjust=title_adj,
                title_xadjust=title_xadj,
                compare_series=cross_mean_series if cs_mean else None,
                title_fontsize=title_fontsize,
                legend_ncol=legend_ncol,
                legend_fontsize=legend_fontsize,
            )

    else:
        with FacetPlot(
            df=df,
            xcats=xcats,
            cids=cids,
            intersect=intersect,
            metrics=[val],
            tickers=[cross_mean_series] if cs_mean else None,
            start=start,
            end=end,
        ) as fp:
            # if no comparison series, pass legend = False
            show_legend: bool = True if cross_mean_series else False
            fp.lineplot(
                figsize=size,
                share_y=same_y,
                title=title,
                # cid_xcat_grid=True,
                cid_grid=True,
                title_yadjust=title_adj,
                title_xadjust=title_xadj,
                compare_series=cross_mean_series if cs_mean else None,
                title_fontsize=title_fontsize,
                ncols=ncol,
                attempt_square=square_grid,
                legend=show_legend,
                legend_ncol=legend_ncol,
                legend_labels=xcat_labels or None,
                legend_fontsize=legend_fontsize,
            )


if __name__ == "__main__":
    # from macrosynergy.visuals import FacetPlot
    from macrosynergy.management.simulate_quantamental_data import make_test_df
    from macrosynergy.dev.local import LocalCache

    LOCAL_CACHE = "~/Macrosynergy/Macrosynergy - Documents/SharedData/JPMaQSTickers"

    cids: List[str] = [
        "USD",
        "EUR",
        "GBP",
        "AUD",
        "CAD",
        "JPY",
        "CHF",
        "NZD",
        "SEK",
        "NOK",
        "DKK",
        "INR",
    ]
    xcats: List[str] = [
        "FXXR",
        "EQXR",
        "RIR",
        "IR",
        "REER",
        "CPI",
        "PPI",
        "M2",
        "M1",
        "M0",
        "FXVOL",
        "FX",
    ]
    sel_cids: List[str] = ["USD", "EUR", "GBP"]
    sel_xcats: List[str] = ["FXXR", "EQXR", "RIR", "IR"]
    r_styles: List[str] = [
        "linear",
        "decreasing-linear",
        "sharp-hill",
        "sine",
        "four-bit-sine",
    ]
    df: pd.DataFrame = make_test_df(
        cids=list(set(cids) - set(sel_cids)),
        xcats=xcats,
        start_date="2000-01-01",
    )

    for rstyle, xcatx in zip(r_styles, sel_xcats):
        dfB: pd.DataFrame = make_test_df(
            cids=sel_cids,
            xcats=[xcatx],
            start_date="2000-01-01",
            prefer=rstyle,
        )
        df: pd.DataFrame = pd.concat([df, dfB], axis=0)

    for ix, cidx in enumerate(sel_cids):
        df.loc[df["cid"] == cidx, "value"] = (
            ((df[df["cid"] == cidx]["value"]) * (ix + 1)).reset_index(drop=True).copy()
        )

    for ix, xcatx in enumerate(sel_xcats):
        df.loc[df["xcat"] == xcatx, "value"] = (
            ((df[df["xcat"] == xcatx]["value"]) * (ix * 10 + 1))
            .reset_index(drop=True)
            .copy()
        )

    import time

    # timer_start: float = time.time()
    timelines(
        df=df,
        xcats=sel_xcats,
        xcat_grid=True,
        xcat_labels=["ForEx", "Equity", "Real Interest Rates", "Interest Rates"],
        square_grid=True,
        cids=sel_cids,
        # single_chart=True,
    )

    # timelines(
    #     df=df,
    #     xcats=sel_xcats[0],
    #     cids=sel_cids,
    #     # cs_mean=True,
    #     # xcat_grid=False,
    #     single_chart=True,
    #     cs_mean=True,
    # )

    # timelines(
    #     df=df,
    #     same_y=False,
    #     xcats=sel_xcats[0],
    #     cids=sel_cids,
    #     title="Plotting multiple cross sections for a single category \n with different y-axis!",
    # )
