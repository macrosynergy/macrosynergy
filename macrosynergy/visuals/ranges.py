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
    legend_loc: str = "lower center",
    legend_bbox_to_anchor: Optional[Tuple[float]] = None,
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
    :param <str> legend_loc: location of legend; passed to matplotlib.pyplot.legend() as
        `loc`. Default is 'center right'.
    :param <Tuple[float]> legend_bbox_to_anchor: passed to matplotlib.pyplot.legend() as
        `bbox_to_anchor`. Default is None.

    Please see [Matplotlib's Legend Documentation]:
    (https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.legend.html)
    for more information on the legend parameters `loc` and `bbox_to_anchor`.
    """

    df["real_date"] = pd.to_datetime(df["real_date"], format="%Y-%m-%d")

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
        if not sort_cids_by in ["mean", "std"]:
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
        dfx = df[filt_1].groupby(["cid"])[val].apply(sort_cids_func)
        order = dfx.sort_values(ascending=False).index
    elif not order_condition and sort_cids_func is not None:
        # Sort across all categories on the available cross-sections.
        dfx = df.groupby(["cid"])[val].apply(sort_cids_func)
        order = dfx.sort_values(ascending=False).index
    else:
        order = None

    sns.set_theme(style="darkgrid")
    fig, ax = plt.subplots(figsize=size)

    if kind == "bar":
        if version.parse(sns.__version__) >= version.parse("0.12.0"):
            ax = sns.barplot(
                x="cid",
                y=val,
                hue="xcat",
                hue_order=xcats,
                palette="Paired",
                data=df,
                errorbar="sd",
                order=order,
            )
        else:
            ax = sns.barplot(
                x="cid",
                y=val,
                hue="xcat",
                hue_order=xcats,
                palette="Paired",
                data=df,
                ci="sd",
                order=order,
            )
    elif kind == "box":
        ax = sns.boxplot(
            x="cid",
            y=val,
            hue="xcat",
            hue_order=xcats,
            palette="Paired",
            data=df,
            order=order,
        )
        ax.xaxis.grid(True)

    ax.set_title(title, fontdict={"fontsize": 16})
    ax.set_xlabel("")
    ax.set_ylabel(ylab)
    ax.xaxis.grid(True)
    ax.axhline(0, ls="--", linewidth=1, color="black")
    handles, labels = ax.get_legend_handles_labels()

    if xcat_labels is not None:
        error_message = (
            "The number of custom labels must match the defined number of "
            "categories in pnl_cats."
        )
        if len(xcat_labels) != len(xcats):
            raise ValueError(error_message)
        labels = xcat_labels

    if (len(xcats) == 1) and (xcat_labels is None):
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
    df_xcats.loc["CRY",] = ["2011-01-01", "2020-10-30", 1, 2, 0.9, 0.5]
    df_xcats.loc["INFL",] = ["2011-01-01", "2020-10-30", 1, 2, 0.9, 0.5]
    df_xcats.loc["GROWTH",] = ["2011-01-01", "2020-10-30", 1, 2, 0.9, 0.5]

    dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)

    # view_ranges(
    #     dfd,
    #     xcats=["XR"],
    #     kind="box",
    #     start="2012-01-01",
    #     end="2018-01-01",
    #     sort_cids_by="std",
    # )

    filter_1 = (dfd["xcat"] == "XR") & (dfd["cid"] == "AUD")
    dfd = dfd[~filter_1]

    view_ranges(
        dfd,
        xcats=["XR", "CRY"],
        cids=cids,
        kind="box",
        start="2012-01-01",
        end="2018-01-01",
        sort_cids_by=None,
        xcat_labels=["EQXR_NSA", "CRY_NSA"],
    )
