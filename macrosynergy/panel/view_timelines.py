import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Union

from macrosynergy.management.simulate_quantamental_data import make_qdf
from macrosynergy.management.check_availability import reduce_df


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
    all_xticks: bool = True,
    xcat_grid: bool = False,
    xcat_labels: Optional[List[str]] = None,
    single_chart: bool = False,
    label_adj: float = 0.05,
    title: Optional[str] = None,
    title_adj: float = 1.0,
    title_xadj: float = 0.5,
    title_fontsize: int = 16,
    label_fontsize: int = 14,
    cs_mean: bool = False,
    size: Tuple[float, float] = (12, 7),
    aspect: float = 1.7,
    height: float = 3.0,
    legend_fontsize: int = 12,
    legend_loc: Union[str, Tuple[float, float]] = "lower center",
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
    :param <int> label_fontsize: font size of labels. Default is 12.
    :param <bool> cs_mean: if True this adds a line of cross-sectional averages to
        the line charts. This is only allowed for function calls with a single
        category. Default is False.
    :param <Tuple[float]> size: two-element tuple setting width/height of single cross
        section plot. Default is (12, 7). This is irrelevant for facet grid.
    :param <float> aspect: width-height ratio for plots in facet. Default is 1.7.
    :param <float> height: height of plots in facet. Default is 3.
    :param <int> legend_fontsize: font size of legend. Default is 12.
    :param <str> legend_loc: location of legend. Default is 'lower center'.
        The options are (strings) 'upper left', 'upper right', 'lower left',
        'lower right', 'upper center', 'lower center', 'center left', 'center right'.
        The string 'best' places the legend at the location, among the nine locations
        defined so far, with the minimum overlap with other drawn features.
        One can also pass a tuple (x, y) in axes coordinates to specify the
        bottom-left corner of the legend. The coordinates (0, 0) are the
        bottom-left corner of the plot's area (chart+labels).
        Please see matplotlib documentation for more details -
        https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.legend.html

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
            assert set(dft.columns).issuperset({"cid", "xcat", "real_date", val})

            df = dft.copy()
        except Exception as e:
            raise ValueError(fail_str)

    df["real_date"] = pd.to_datetime(df["real_date"], format="%Y-%m-%d")

    if not isinstance(cs_mean, bool):
        raise TypeError(f"`cs_mean` parameter must be a Boolean object.")

    if isinstance(xcats, str):
        xcats = [xcats]
    if isinstance(cids, str):
        cids = [cids]

    for varx, namex in zip([xcats, cids], ["xcats", "cids"]):
        if (
            (not isinstance(varx, list))
            or (not all(isinstance(x, str) for x in varx))
            or (not varx)
        ):
            raise TypeError(f"`{namex}` must be a list of strings.")

    if cs_mean and len(xcats) > 1:
        raise ValueError(
            f"cs_mean can only be set to True "
            "if a single category is passed. The "
            f"received categories are {xcats}."
        )

    if not isinstance(xcat_grid, bool):
        raise TypeError("`xcat_grid` parameter must be a Boolean object.")

    if not isinstance(single_chart, bool):
        raise TypeError("`single_chart` parameter must be a Boolean object.")
    if xcat_grid and single_chart:
        raise ValueError("xcat_grid and single_chart cannot both be True.")

    df, xcats, cids = reduce_df(
        df, xcats, cids, start, end, out_all=True, intersect=intersect
    )

    # NOTE: casting var(cids) to list if it is a string is dependent on the reduce_df function

    if xcat_grid:
        if not len(cids) == 1:
            raise ValueError(
                "`xcat_grid` can only be set to True if a "
                "single cross-section (`cids`) is passed."
            )

    if cumsum:
        df[val] = (
            df.sort_values(["cid", "xcat", "real_date"])[["cid", "xcat", val]]
            .groupby(["cid", "xcat"])
            .cumsum()
        )

    max_plots: int = 1
    if (len(cids) == 1) and xcat_grid:
        max_plots = len(xcats)
    elif (len(cids) > 1) and not single_chart:
        max_plots = len(cids)
    ncol: int = min(ncol, max_plots)

    cross_mean: Optional[pd.DataFrame] = None
    cs_label: str = f"cross-sectional average of {xcats[0]}."
    if cs_mean and (len(cids) > 1):
        dfw: pd.DataFrame = df.pivot(index="real_date", columns="cid", values="value")
        cm_series: pd.Series = dfw.mean(axis=1)
        cross_mean: pd.DataFrame = pd.DataFrame(
            data=cm_series.to_numpy(), index=cm_series.index, columns=["average"]
        ).reset_index(level=0)

        if xcat_labels is not None:
            if len(xcat_labels) == (len(xcats) + 1):
                cs_label = xcat_labels.pop(-1)

    # Replace xcats in the DF with their corresponding labels
    if xcat_labels is not None:
        for xc, xl in zip(xcats, xcat_labels):
            df["xcat"] = df["xcat"].replace(xc, xl)
        xcats = xcat_labels

    # use style=darkgrid
    plt.style.use("seaborn-v0_8-darkgrid")
    # plt.rcParams["figure.figsize"] = size

    fig: Optional[plt.Figure] = None
    ax: Optional[plt.Axes] = None

    if len(cids) == 1:
        if xcat_grid:
            nrows = int(np.ceil(len(xcats) / float(ncol)))

            fig, axes = plt.subplots(
                nrows,
                ncol,
                figsize=(ncol * aspect * height, nrows * height),
                sharey=same_y,
            )
            axes = np.ravel(axes)

            for i, xc in enumerate(xcats):
                ax: plt.Axes = axes[i]
                df_xc: pd.DataFrame = df[df["xcat"] == xc]
                ax.plot(df_xc["real_date"], df_xc[val], label=xc)

                ax.axhline(y=0, c=".5")
                ax.set_xlabel("")
                ax.set_ylabel("")
                ax.set_title(xc, fontsize=label_fontsize)

            for j in range(i + 1, len(axes)):
                fig.delaxes(axes[j])

        else:
            ax: plt.Axes = plt.gca()
            for xc in xcats:
                dfc: pd.DataFrame = df[df["xcat"] == xc]
                ax.plot(dfc["real_date"], dfc[val], label=xc)

            plt.axhline(y=0, c=".5")

            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.legend(
                ncol=legend_ncol,
                fontsize=legend_fontsize,
                loc=legend_loc,
            )

    else:
        if not single_chart:
            nrows = int(np.ceil(len(cids) / float(ncol)))

            fig, axes = plt.subplots(
                nrows,
                ncol,
                figsize=(ncol * aspect * height, nrows * height),
                sharey=same_y,
            )
            axes = np.ravel(axes)

            for i, cid in enumerate(cids):
                ax: plt.Axes = axes[i]
                df_cid: pd.DataFrame = df[df["cid"] == cid]

                for xc in xcats:
                    dfc: pd.DataFrame = df_cid[df_cid["xcat"] == xc]
                    ax.plot(dfc["real_date"], dfc[val], label=xc)
                    ax.axhline(y=0, c=".5")

                if cs_mean:
                    ax.plot(
                        cross_mean["real_date"],
                        cross_mean["average"],
                        color="red",
                        label=cs_label,
                    )

                ax.set_xlabel("")
                ax.set_ylabel("")
                ax.set_title(cid, fontsize=label_fontsize)

            for j in range(i + 1, len(axes)):
                fig.delaxes(axes[j])

            handles, labels = ax.get_legend_handles_labels()
            fig.legend(
                handles=handles,
                labels=labels,
                loc=legend_loc,
                ncol=legend_ncol,
                fontsize=legend_fontsize,
            )

        else:
            ax: plt.Axes = plt.gca()
            for cid in cids:
                dfc: pd.DataFrame = df[df["cid"] == cid]
                ax.plot(dfc["real_date"], dfc[val], label=cid)

            if cs_mean:
                ax.plot(
                    cross_mean["real_date"],
                    cross_mean["average"],
                    color="red",
                    label=cs_label,
                )

            plt.axhline(y=0, c=".5")
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.legend(ncol=legend_ncol, fontsize=legend_fontsize, loc=legend_loc)

    if title is not None:
        plt.suptitle(
            title,
            fontsize=title_fontsize,
            x=title_xadj,
            y=title_adj,
        )

    plt.gcf().set_size_inches(size[0], size[1])
    if fig is not None:
        for ax in fig.axes:
            plt.sca(ax)
            plt.xticks(visible=all_xticks)

        fig.set_tight_layout(True)

    else:
        plt.xticks(visible=all_xticks)
        plt.gcf().tight_layout()
    plt.subplots_adjust(bottom=label_adj)

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
        label_adj=0.1,
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
        dfd,
        xcats=["CRY"],
        cids=cids,
        ncol=2,
        title="Carry",
        cs_mean=True,
        xcat_labels=["Carry", "cs-mean-1"],
    )

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

    view_timelines(
        dfd,
        xcats=["XR"],
        single_chart=True,
        cids=cids,
        start="2010-01-01",
        title="AUD Return Comparison",
        cs_mean=True,
        aspect=1.5,
        title_adj=0.92,
        label_adj=0.08,
        xcat_labels=["AUD Return", "cs-mean"],
    )
