"""
Module for plotting ranges of values across cross-sections for one or more categories.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Callable, Optional
from packaging import version

from macrosynergy.management.simulate import make_qdf
from macrosynergy.management.utils import reduce_df
from macrosynergy.management.types import QuantamentalDataFrame


def view_ranges(
    df: pd.DataFrame,
    xcats: List[str],
    cids: Optional[List[str]] = None,
    start: str = None,
    end: Optional[str] = None,
    val: str = "value",
    kind: str = "bar",
    sort_cids_by: Optional[str] = None,
    title: Optional[str] = None,
    ylab: Optional[str] = None,
    size: Tuple[float] = (16, 8),
    xcat_labels: Optional[List[str]] = None,
    legend_loc: str = "lower center",
    legend_bbox_to_anchor: Optional[Tuple[float]] = None,
    facet: bool = False,
    ncols: int = None,
    nrows: int = None,
    drop_cid_labels: bool = False,
):
    """
    Plots averages and various ranges across sections for one or more categories.

    Parameters
    ----------
    df : ~pandas.DataFrame
        standardized DataFrame with the necessary columns: 'cid', 'xcat', 'real_date'
        and at least one column with values of interest.
    xcats : List[str]
        extended categories to be checked on. Default is all in the DataFrame.
    cids : List[str]
        cross sections to plot. Default is all in DataFrame.
    start : str
        earliest date in ISO format. Default earliest date in df.
    end : str
        latest date in ISO format. Default is latest date in df.
    val : str
        name of column that contains the values. Default is 'value'.
    kind : str
        type of range plot. Default is 'bar'; other option is 'box'.
    sort_cids_by : str
        criterion for sorting cids on x-axis; Arguments can be 'mean' and 'std'. Default
        is None, i.e. original order. Ordering will be based on the first category if the
        category is defined over the complete panel. Otherwise, mean and standard deviation
        calculated, of the cross-sections, computed across all categories.
    title : str
        string of chart title; defaults depend on type of range plot.
    ylab : str
        y label. Default is no label.
    size : Tuple[float]
        Tuple of width and height of graph. Default is (16, 8).
    xcat_labels : Union[List[str], Dict[str, str]]
        custom labels to be used for the ranges.
    legend_loc : str
        location of legend; passed to matplotlib.pyplot.legend() as `loc`. Default is
        'center right'.
    legend_bbox_to_anchor : Tuple[float]
        passed to matplotlib.pyplot.legend() as `bbox_to_anchor`. Default is None.
        Please see [Matplotlib's Legend Documentation]:
        (https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.legend.html) for more
        information on the legend parameters `loc` and `bbox_to_anchor`.
    facet : bool
        If True, facets the plot by xcat. Default is False.
    ncols : int
        Number of columns in the facet plot. Default is None, which will be set to
        `ceil(n_xcats / nrows)`.
    nrows : int
        Number of rows in the facet plot. Default is None, which will be set to
        `min(n_xcats, 2)`.
    drop_cid_labels : bool
        If True, the x-axis labels for cids will be dropped when plotting a facet. 
        Default is False. This is useful when there are many cids and the labels would 
        overlap.
    """

    df = QuantamentalDataFrame(df)

    possible_xcats = set(df["xcat"])
    missing_xcats = set(xcats).difference(possible_xcats)
    error_xcats = (
        "The categories passed in to view_ranges() must be present in the "
        f"DataFrame: missing {missing_xcats}."
    )
    if not set(xcats).issubset(possible_xcats):
        raise ValueError(error_xcats)

    sort_cids_func: Callable = None
    if sort_cids_by is not None:
        if not isinstance(sort_cids_by, str):
            raise TypeError("`sort_cids_by` must be a string.")
        sort_error = "Sorting parameter must either be 'mean' or 'std'."
        if  sort_cids_by not in ["mean", "std"]:
            raise ValueError(sort_error)
        if sort_cids_by == "mean":
            sort_cids_func = np.mean
        else:
            sort_cids_func = np.std

    error_message = (
        "The number of custom labels must match the defined number of "
        "categories in pnl_cats."
    )
    if xcat_labels is not None:
        if isinstance(xcat_labels, dict):
            xcat_labels = [xcat_labels[xcat] for xcat in xcats]
        if len(xcat_labels) != len(xcats):
            raise ValueError(error_message)

    # Unique cross-sections across the union of categories passed - not the intersection.
    # Order of categories will be preserved.
    df, xcats, cids = reduce_df(df, xcats, cids, start, end, out_all=True)

    s_date = df["real_date"].min().strftime("%Y-%m-%d")
    e_date = df["real_date"].max().strftime("%Y-%m-%d")

    sns.set(style="darkgrid")

    if title is None:
        if kind == "bar":
            title = f"Means and standard deviations from {s_date} to {e_date}."
        elif kind == "box":
            title = (
                f"Interquartile ranges, extended ranges and outliers "
                f"from {s_date} to {e_date}."
            )
    if ylab is None:
        ylab = ""

    filt_1 = df["xcat"] == xcats[0]
    first_xcat_cids = set(df[filt_1]["cid"])
    # First category is not defined over all cross-sections.
    order_condition = list(set(cids)) == list(first_xcat_cids)

    if order_condition and sort_cids_func is not None:
        # Sort exclusively on the first category.
        dfx = df[filt_1].groupby(["cid"], observed=True)[val].apply(sort_cids_func)
        order = dfx.sort_values(ascending=False).index
    elif not order_condition and sort_cids_func is not None:
        # Sort across all categories on the available cross-sections.
        dfx = df.groupby(["cid"], observed=True)[val].apply(sort_cids_func)
        order = dfx.sort_values(ascending=False).index
    else:
        order = None

    sns.set_theme(style="darkgrid")

    def _plot_bar(ax, data, order, val, xcat=None):
        if version.parse(sns.__version__) >= version.parse("0.12.0"):
            return sns.barplot(
                x="cid",
                y=val,
                hue="xcat" if xcat is None else None,
                hue_order=xcats if xcat is None else None,
                palette="Paired",
                data=data,
                errorbar="sd",
                order=order,
                ax=ax if xcat is not None else None,
            )
        else:
            return sns.barplot(
                x="cid",
                y=val,
                hue="xcat" if xcat is None else None,
                hue_order=xcats if xcat is None else None,
                palette="Paired",
                data=data,
                ci="sd",
                order=order,
                ax=ax if xcat is not None else None,
            )

    def _plot_box(ax, data, order, val, xcat=None):
        return sns.boxplot(
            x="cid",
            y=val,
            hue="xcat" if xcat is None else None,
            hue_order=xcats if xcat is None else None,
            palette="Paired",
            data=data,
            order=order,
            ax=ax if xcat is not None else None,
        )

    def _set_facet_axis(ax, title, ylab):
        ax.set_title(title, fontdict={"fontsize": 14})
        ax.set_xlabel("")
        ax.set_ylabel(ylab)
        ax.xaxis.grid(True)
        ax.axhline(0, ls="--", linewidth=1, color="black")
        if ax.get_legend() is not None:
            ax.get_legend().remove()
        if drop_cid_labels:
            ax.set_xticklabels([])
            ax.set_xlabel("")

            ax.set_xticklabels([])

    def _set_main_axis(ax, title, ylab):
        ax.set_title(title, fontdict={"fontsize": 16})
        ax.set_xlabel("")
        ax.set_ylabel(ylab)
        ax.xaxis.grid(True)
        ax.axhline(0, ls="--", linewidth=1, color="black")

    if facet:
        n_xcats = len(xcats)
        if ncols is None and nrows is None:
            max_rows = 2
            nrows = min(n_xcats, max_rows)
            ncols = int(np.ceil(n_xcats / nrows))
        figsize = (size[0] * ncols / max(1, n_xcats/2), size[1] * nrows / max(1, n_xcats/2))
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
        axes = axes.flatten()
        for i, xcat in enumerate(xcats):
            ax = axes[i]
            df_xcat = df[df["xcat"] == xcat]
            if kind == "bar":
                _plot_bar(ax, df_xcat, order, val, xcat=xcat)
            elif kind == "box":
                _plot_box(ax, df_xcat, order, val, xcat=xcat)
            facet_title = xcat_labels[i] if xcat_labels is not None else xcat
            _set_facet_axis(ax, facet_title, ylab)
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        fig.suptitle(title, fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.show()
    else:
        fig, ax = plt.subplots(figsize=size)
        if kind == "bar":
            _plot_bar(ax, df, order, val)
        elif kind == "box":
            _plot_box(ax, df, order, val)
        _set_main_axis(ax, title, ylab)
        handles, labels = ax.get_legend_handles_labels()
        if xcat_labels is not None:
            labels = xcat_labels
        if (len(xcats) == 1) and (xcat_labels is None):
            if ax.get_legend() is not None:
                ax.get_legend().remove()
        else:
            if legend_bbox_to_anchor is None:
                legend_bbox_to_anchor = (0.5, -0.15 - 0.05 * (len(xcats) - 2))
            ax.legend(
                handles=handles[0:],
                labels=labels[0:],
                loc=legend_loc,
                bbox_to_anchor=legend_bbox_to_anchor,
            )
        plt.tight_layout()
        plt.show()


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
    df_xcats.loc["XR2",] = ["2010-01-01", "2020-12-31", 0, 1, 0, 0.3]
    df_xcats.loc["XR3",] = ["2011-01-01", "2020-10-30", 1, 2, 0.9, 0.5]
    df_xcats.loc["XR4",] = ["2011-01-01", "2020-10-30", 1, 2, 0.9, 0.5]
    df_xcats.loc["XR5",] = ["2011-01-01", "2020-10-30", 1, 2, 0.9, 0.5]
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
        xcats=["XR", "CRY", "INFL", "GROWTH"],
        cids=cids,
        kind="box",
        start="2012-01-01",
        end="2018-01-01",
        sort_cids_by=None,
        # xcat_labels=["EQXR_NSA"],
    )
    view_ranges(
        dfd,
        xcats=["XR", "CRY", "INFL", "GROWTH", "XR2", "XR3"],
        cids=cids,
        kind="box",
        start="2012-01-01",
        end="2018-01-01",
        sort_cids_by=None,
        facet=True,
        ncols=2,
        nrows=3,
        drop_cid_labels=True,
    )
