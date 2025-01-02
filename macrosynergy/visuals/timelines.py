"""
Function for visualising a facet grid of time line charts of one or more categories.

.. code-block:: python

    import macrosynergy.visuals as msv
    ...
    msv.view.timelines(df, xcats=["FXXR","EQXR", "IR"], cids=["USD", "EUR", "GBP"] )
    ...
    msv.FacetPlot(df).lineplot(cid_grid=True)

"""

from typing import Dict, List, Optional, Tuple

import pandas as pd

from macrosynergy.management.utils import reduce_df
from macrosynergy.visuals import FacetPlot, LinePlot
from numbers import Number

IDX_COLS: List[str] = ["cid", "xcat", "real_date"]


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
    all_xticks: bool = False,  # ~(same_x) basically
    xcat_grid: bool = False,
    xcat_labels: Optional[List[str]] = None,
    single_chart: bool = False,
    label_adj: float = 0.05,
    title: Optional[str] = None,
    title_adj: float = 0.95,
    title_xadj: float = 0.5,
    title_fontsize: int = 22,
    cs_mean: bool = False,
    size: Tuple[Number, Number] = (12, 7),
    aspect: Number = 1.7,
    height: Number = 3.0,
    legend_fontsize: int = 12,
    blacklist: Dict = None,
):
    """
    Displays a facet grid of time line charts of one or more categories.

    Parameters
    ----------
    df : ~pandas.DataFrame
        standardized DataFrame with the necessary columns: 'cid', 'xcat', 'real_date'
        and at least one column with values of interest.
    xcats : List[str]
        extended categories to plot. Default is all in DataFrame.
    cids : List[str]
        cross sections to plot. Default is all in DataFrame. If this contains only one
        cross section a single line chart is created.
    intersect : bool
        if True only retains cids that are available for all xcats. Default is False.
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
        if True, shows a facet grid of line charts for each xcat for given cross
        sections. Default is False.
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
    size : Tuple[Number, Number]
        two-element tuple setting width/height of single cross section plot. Default is
        (12, 7). This is irrelevant for facet grid.
    aspect : Number
        width-height ratio for plots in facet. Default is 1.7.
    height : Number
        height of plots in facet. Default is 3.
    legend_fontsize : int
        font size of legend. Default is 12.
    blacklist : dict
        cross-sections with date ranges that should be excluded from the dataframe.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("`df` must be a pandas DataFrame.")

    if len(df.columns) < 4:
        df = df.copy().reset_index()

    if val not in df.columns:
        if len(df.columns) == len(IDX_COLS) + 1:
            val: str = list(set(df.columns) - set(IDX_COLS))[0]
            if not pd.api.types.is_numeric_dtype(df[val]):
                raise ValueError(
                    f"Column '{val}' (passed as `metric`) is not numeric, and there are "
                    f"no other numeric columns in the DataFrame."
                )
        else:
            raise ValueError(
                f"Column '{val}' (passed as `metric`) does not exist, and there are "
                "none/many other numeric columns in the DataFrame."
            )

    if start is None:
        start: str = pd.Timestamp(df["real_date"].min()).strftime("%Y-%m-%d")

    if end is None:
        end: str = pd.Timestamp(df["real_date"].max()).strftime("%Y-%m-%d")

    if isinstance(xcats, str):
        xcats: List[str] = [xcats]
    if isinstance(cids, str):
        cids: List[str] = [cids]

    for varx, namex in zip([single_chart, xcat_grid], ["single_chart", "xcat_grid"]):
        if not isinstance(varx, bool):
            raise TypeError(f"`{namex}` must be a boolean.")

    if xcat_grid and single_chart:
        raise ValueError(
            "`xcat_grid` and `single_chart` cannot be True simultaneously."
        )

    if cs_mean and xcat_grid:
        raise ValueError("`cs_mean` requires `xcat_grid` to be False.")

    if blacklist:
        if not isinstance(blacklist, dict):
            raise TypeError("`blacklist` must be a dictionary.")
        for key, value in blacklist.items():
            if not isinstance(key, str):
                raise TypeError("Keys in `blacklist` must be strings.")
            if not isinstance(value, list):
                raise TypeError("Values in `blacklist` must be lists.")

    if xcats is None:
        if xcat_labels:
            raise ValueError("`xcat_labels` requires `xcats` to be defined.")
        xcats: List[str] = df["xcat"].unique().tolist()

    if cids is None:
        cids: List[str] = df["cid"].unique().tolist()

    if cumsum:
        df = reduce_df(df, xcats=xcats, cids=cids, start=start, end=end, blacklist=blacklist)
        df[val] = (
            df.sort_values(["cid", "xcat", "real_date"])[["cid", "xcat", val]]
            .groupby(["cid", "xcat"])
            .cumsum()
        )

    cross_mean_series: Optional[str] = f"mean_{xcats[0]}" if cs_mean else None
    if cs_mean:
        df = reduce_df(df, xcats=xcats, cids=cids, start=start, end=end, blacklist=blacklist)
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
        # when `cs_mean` is True, `xcat_labels` may have one extra label
        if len(xcat_labels) != len(xcats) and len(xcat_labels) != len(xcats) + int(
            cs_mean
        ):
            raise ValueError(
                "`xcat_labels` must have same length as `xcats` "
                "(or one extra label if `cs_mean` is True)."
            )

    if cs_mean:
        if xcat_labels is None:
            xcat_labels = [xcats[0]]
        if len(xcat_labels) == 1:
            xcat_labels.append("Cross-Sectional Mean")

    facet_size: Optional[Tuple[float, float]] = (
        (aspect * height, height)
        if (aspect is not None and height is not None)
        else None
    )

    if xcat_grid and (len(xcats) == 1):
        xcat_grid: bool = False
        single_chart: bool = True

    if xcat_grid:
        if ncol > len(xcats):
            ncol: int = len(xcats)

        with FacetPlot(
            df=df,
            xcats=xcats,
            cids=cids,
            intersect=intersect,
            metrics=[val],
            tickers=[cross_mean_series] if cs_mean else None,
            start=start,
            end=end,
            blacklist=blacklist,
        ) as fp:
            fp.lineplot(
                share_y=same_y,
                share_x=not all_xticks,
                figsize=size,
                xcat_grid=True,  # Not to be confused with `xcat_grid` parameter
                # legend_labels=xcat_labels or None,
                facet_titles=xcat_labels or None,
                title=title,
                title_yadjust=title_adj,
                title_xadjust=title_xadj,
                compare_series=cross_mean_series if cs_mean else None,
                title_fontsize=title_fontsize,
                # title_fontsize=24,
                ncols=ncol,
                attempt_square=square_grid,
                facet_size=facet_size,
                legend_ncol=legend_ncol,
                legend_fontsize=legend_fontsize,
                interpolate=cumsum,
            )

    elif single_chart or (len(cids) == 1):
        with LinePlot(
            df=df,
            cids=cids,
            xcats=xcats,
            intersect=intersect,
            metrics=[val],
            tickers=[cross_mean_series] if cs_mean else None,
            start=start,
            end=end,
            blacklist=blacklist,
        ) as lp:
            lp.plot(
                metric=val,
                figsize=size,
                title=title,
                title_yadjust=title_adj,
                title_xadjust=title_xadj,
                compare_series=cross_mean_series if cs_mean else None,
                title_fontsize=title_fontsize,
                # title_fontsize=18,
                legend_ncol=legend_ncol,
                legend_fontsize=legend_fontsize,
                legend_labels=xcat_labels or None,
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
            blacklist=blacklist,
        ) as fp:
            show_legend: bool = True if cross_mean_series else False
            show_legend = show_legend or (len(xcats) > 1)
            if ncol > len(cids):
                ncol: int = len(cids)

            fp.lineplot(
                figsize=size,
                share_y=same_y,
                share_x=not all_xticks,
                title=title,
                # cid_xcat_grid=True,
                cid_grid=True,
                title_yadjust=title_adj,
                title_xadjust=title_xadj,
                compare_series=cross_mean_series if cs_mean else None,
                facet_size=facet_size,
                title_fontsize=title_fontsize,
                # title_fontsize=24,
                ncols=ncol,
                attempt_square=square_grid,
                legend=show_legend,
                legend_ncol=legend_ncol,
                legend_labels=xcat_labels or None,
                legend_fontsize=legend_fontsize,
                interpolate=cumsum,
            )


if __name__ == "__main__":
    from macrosynergy.visuals import FacetPlot
    from macrosynergy.management.simulate import make_test_df
    import numpy as np

    np.random.seed(42)

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
        start="2000-01-01",
    )

    for rstyle, xcatx in zip(r_styles, sel_xcats):
        dfB: pd.DataFrame = make_test_df(
            cids=sel_cids,
            xcats=[xcatx],
            start="2000-01-01",
            style=rstyle,
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
    black = {"EUR": ["2012-01-01", "2018-01-01"], "GBP": ["2004-01-01", "2007-01-01"], 
             "USD": ["2015-01-01", "2018-01-01"]}
    
    # timer_start: float = time.time()
    timelines(
        df=df,
        xcats=sel_xcats,
        xcat_grid=True,
        xcat_labels=["ForEx", "Equity", "Real Interest Rates", "Interest Rates"],
        square_grid=True,
        cids=sel_cids[1],
        cumsum=True,
        blacklist=black,
        # single_chart=True,
    )

    timelines(
        df=df,
        xcats=sel_xcats[0],
        cids=sel_cids,
        # cs_mean=True,
        # xcat_grid=False,
        single_chart=True,
        cs_mean=True,
        blacklist=black,
    )

    timelines(
        df=df,
        same_y=False,
        xcats=sel_xcats[0],
        cids=sel_cids,
        title=(
            "Plotting multiple cross sections for a single category \n with different "
            "y-axis!"
        ),
        blacklist=black,
        cumsum=True,
    )
