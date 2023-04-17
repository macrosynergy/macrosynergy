import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.axes
import seaborn as sns
from typing import List, Union, Tuple

from macrosynergy.management.simulate_quantamental_data import make_qdf
from macrosynergy.management.check_availability import reduce_df
from macrosynergy.visuals import FacetPlot, LinePlot


def view_timelines(
    df: pd.DataFrame,
    xcats: List[str] = None,
    cids: List[str] = None,
    intersect: bool = False,
    val: str = "value",
    cumsum: bool = False,
    start: str = "2000-01-01",
    end: str = None,
    ncol: int = 3,
    same_y: bool = True,
    all_xticks: bool = False,
    xcat_grid: bool = False,
    xcat_labels: List[str] = None,
    single_chart: bool = False,
    label_adj: float = 0.05,
    title: str = None,
    title_adj: float = 0.95,
    cs_mean: bool = False,
    size: Tuple[float] = (12, 7),
    aspect: float = 1.7,
    height: float = 3,
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
    :param <float> label_adj: parameter that sets bottom of figure to fit the label.
        Default is 0.05.
    :param <bool> cs_mean: if True this adds a line of cross-sectional averages to
        the line charts. This is only allowed for function calls with a single
        category. Default is False.
    :param <Tuple[float]> size: two-element tuple setting width/height of single cross
        section plot. Default is (12, 7). This is irrelevant for facet grid.
    :param <float> aspect: width-height ratio for plots in facet. Default is 1.7.
    :param <float> height: height of plots in facet. Default is 3.

    """

    if not set(df.columns).issuperset({"cid", "xcat", "real_date", val}):
        fail_str: str = (
            f"Error : Tried to standardize DataFrame but failed."
            f"DataFrame not in the correct format. Please ensure "
            f"that the DataFrame has the following columns: "
            f"'cid', 'xcat', 'real_date' and column='{val}'."
        )
        try:
            dft = df.reset_index()
            assert set(dft.columns).issuperset(
                {"cid", "xcat", "real_date", val}
            ), fail_str

            df = dft.copy()
        except Exception as e:
            raise Exception(f"Exception message: {e}", fail_str)

    df["real_date"] = pd.to_datetime(df["real_date"], format="%Y-%m-%d")

    cs_mean_error = f"cs_mean parameter must be a Boolean Object."
    assert isinstance(cs_mean, bool), cs_mean_error
    error = (
        f"cs_mean can only be set to True if a single category is passed. The "
        f"received categories are {xcats}."
    )
    if cs_mean:
        assert len(xcats) == 1, error

    assert isinstance(xcat_grid, bool), "xcat_grid parameter must be a Boolean Object."
    assert isinstance(
        single_chart, bool
    ), "single_chart parameter must be a Boolean Object."
    assert not (
        xcat_grid and single_chart
    ), "xcat_grid and single_chart cannot both be True."

    df, xcats, cids = reduce_df(
        df, xcats, cids, start, end, out_all=True, intersect=intersect
    )

    # NOTE: casting var(cids) to list if it is a string is dependent on the reduce_df function

    assert isinstance(xcat_grid, bool), "xcat_grid parameter must be a Boolean Object."
    if xcat_grid:
        assert (
            len(cids) == 1
        ), "xcat_grid can only be set to True if a single cross-section is passed."

    if cumsum:
        df[val] = (
            df.sort_values(["cid", "xcat", "real_date"])[["cid", "xcat", val]]
            .groupby(["cid", "xcat"])
            .cumsum()
        )

    sns.set(style="darkgrid")
    face_plotter : FacetPlot = FacetPlot(df=df, cids=cids, xcats=xcats, metrics=[val], start_date=start, end_date=end)
    line_plotter : LinePlot = LinePlot(df=df, cids=cids, xcats=xcats, metrics=[val], start_date=start, end_date=end)

    if len(cids) == 1:
        if xcat_grid:
            face_plotter.plot(
                plot_by_xcat=True,
                same_y=same_y,
                ncols=ncol,
                xcat_labels=xcat_labels,
                fig_title=title,
                height=height,
                fig_title_adj=title_adj,
                figsize=size,
                aspect=aspect,
                legend=True,
                all_xticks=all_xticks,
            )
            plt.show()

        else:
            line_plotter.plot(
                    plot_by_xcat=True,
                    xcat_labels=xcat_labels,
                    fig_title=title,
                    fig_title_adj=title_adj,
                    figsize=size,
                    aspect=aspect,
                )
            plt.show()

    else:
        cross_mean : pd.Series = None
        cross_mean_label : str = None
        if cs_mean:
            cdf: pd.DataFrame = df.pivot( index="real_date", columns="cid", values="value"
            ).mean(axis=1)
            cross_mean: pd.Series = pd.Series(
                data=cdf.to_numpy(),
                index=cdf.index,
            )
            cross_mean_label: str = f"cross-sectional average of {xcats[0]}"
    
        if not single_chart:
            face_plotter.plot(
                    plot_type="line",
                    plot_by_cid=True,
                    same_y=same_y,
                    ncols=ncol,
                    # cid_labels=cid_labels,
                    add_axhline=True,
                    fig_title=title,
                    fig_title_adj=title_adj,
                    figsize=size,
                    aspect=aspect,
                    compare_series=cross_mean,
                    compare_series_label=cross_mean_label,
                )
            plt.show()

        else:
            line_plotter.plot(
                    plot_by_cid=True,
                    # cid_labels=cid_labels,
                    fig_title=title,
                    fig_title_adj=title_adj,
                    figsize=size,
                    aspect=aspect,
                    compare_series=cross_mean,
                    compare_series_label=cross_mean_label,
                )
            plt.show()



if __name__ == "__main__":

    cids = ["AUD", "CAD", "GBP", "NZD"]
    xcats = ["XR", "CRY", "INFL"]
    df_cids = pd.DataFrame(
        index=cids, columns=["earliest", "latest", "mean_add", "sd_mult"]
    )
    df_cids.loc[
        "AUD",
    ] = ["2010-01-01", "2020-12-31", 0.2, 0.2]
    df_cids.loc[
        "CAD",
    ] = ["2011-01-01", "2020-11-30", 0, 1]
    df_cids.loc[
        "GBP",
    ] = ["2012-01-01", "2020-11-30", 0, 2]
    df_cids.loc[
        "NZD",
    ] = ["2012-01-01", "2020-09-30", -0.1, 3]

    df_xcats = pd.DataFrame(
        index=xcats,
        columns=["earliest", "latest", "mean_add", "sd_mult", "ar_coef", "back_coef"],
    )

    df_xcats.loc[
        "XR",
    ] = ["2010-01-01", "2020-12-31", 0.1, 1, 0, 0.3]
    df_xcats.loc[
        "INFL",
    ] = ["2015-01-01", "2020-12-31", 0.1, 1, 0, 0.3]
    df_xcats.loc[
        "CRY",
    ] = ["2013-01-01", "2020-10-30", 1, 2, 0.95, 0.5]

    dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)
    dfdx = dfd[~((dfd["cid"] == "AUD") & (dfd["xcat"] == "XR"))]

    view_timelines(
        dfd,
        xcats=["XR", "CRY"],
        cids=cids[0],
        size=(10, 5),
        title="AUD Return and Carry",
    )

    view_timelines(
        dfd,
        xcats=["XR", "CRY", "INFL"],
        cids=cids[0],
        xcat_grid=True,
        title_adj=0.8,
        xcat_labels=["Return", "Carry", "Inflation"],
        title="AUD Return, Carry & Inflation",
    )

    view_timelines(dfd, xcats=["CRY"], cids=cids, ncol=2, title="Carry", cs_mean=True)

    view_timelines(
        dfd, xcats=["XR"], cids=cids, ncol=2, cumsum=True, same_y=False, aspect=2
    )

    dfd = dfd.set_index("real_date")
    view_timelines(
        dfd,
        xcats=["XR"],
        cids=cids,
        ncol=2,
        cumsum=True,
        same_y=False,
        aspect=2,
        single_chart=True,
    )

    dfd.info()
