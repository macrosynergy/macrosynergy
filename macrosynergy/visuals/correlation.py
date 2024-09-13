"""
Functions used to visualize correlations across categories or cross-sections of
panels.
"""

import itertools
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
import seaborn as sns
from matplotlib import pyplot as plt

from macrosynergy.management.simulate import make_qdf
from macrosynergy.management.utils import _map_to_business_day_frequency, reduce_df


def view_correlation(
    df: pd.DataFrame,
    xcats: Union[str, List[str]] = None,
    cids: List[str] = None,
    xcats_secondary: Optional[Union[str, List[str]]] = None,
    cids_secondary: Optional[List[str]] = None,
    start: str = "2000-01-01",
    end: str = None,
    val: str = "value",
    freq: str = None,
    cluster: bool = False,
    lags: dict = None,
    lags_secondary: Optional[dict] = None,
    title: str = None,
    size: Tuple[float] = (14, 8),
    title_fontsize: int = 20,
    fontsize: int = 14,
    max_color: float = None,
    show: bool = True,
    xcat_labels: Optional[Union[List[str], Dict[str, str]]] = None,
    xcat_secondary_labels: Optional[Union[List[str], Dict[str, str]]] = None,
    cbar_shrink: Union[float, int] = 0.5,
    cbar_fontsize: int = 12,
    **kwargs: Any,
):
    """
    Visualize correlation across categories or cross-sections of panels.

    :param <pd.Dataframe> df: standardized JPMaQS DataFrame with the necessary columns:
        'cid', 'xcat', 'real_date' and at least one column with values of interest.
    :param <List[str]> xcats: extended categories to be correlated. Default is all in the
        DataFrame. If xcats contains only one category the correlation coefficients
        across cross sections are displayed. If xcats contains more than one category,
        the correlation coefficients across categories are displayed. Additionally, the
        order of the xcats received will be mirrored in the correlation matrix.
    :param <List[str]> cids: cross sections to be correlated. Default is all in the
        DataFrame.
    :param <List[str]> xcats_secondary: an optional second set of extended categories.
        If xcats_secondary is provided, correlations will be calculated between the
        categories in xcats and xcats_secondary.
    :param <List[str]> cids_secondary: an optional second list of cross sections. If
        cids_secondary is provided correlations will be calculated and visualized between
        these two sets.
    :param <str> start: earliest date in ISO format. Default is None and earliest date
        in df is used.
    :param <str> end: latest date in ISO format. Default is None and latest date in df
        is used.
    :param <str> val: name of column that contains the values of interest. Default is
        'value'.
    :param <str> freq: frequency option. Per default the correlations are calculated
        based on the native frequency of the datetimes in 'real_date', which is business
        daily. Down-sampling options include weekly ('W'), monthly ('M'), or quarterly
        ('Q') mean.
    :param <bool> cluster: if True the series in the correlation matrix are reordered
        by hierarchical clustering. Default is False.
    :param <dict> lags: optional dictionary of lags applied to respective categories.
        The key will be the category and the value is the lag or lags. If a
        category has multiple lags applied, pass in a list of lag values. The lag factor
        will be appended to the category name in the correlation matrix.
        If xcats_secondary is not none, this parameter will specify lags for the
        categories in xcats.
        N.B.: Lags can include a 0 if the original should also be correlated.
    :param <dict> lags_secondary: optional dictionary of lags applied to the second set of
        categories if xcats_secondary is provided.
    :param <str> title: chart heading. If none is given, a default title is used.
    :param <Tuple[float]> size: two-element tuple setting width/height of figure. Default
        is (14, 8).
    :param <int> title_fontsize: font size of the title. Default is 20.
    :param <int> fontsize: font size of the title. Default is 14.
    :param <float> max_color: maximum values of positive/negative correlation
        coefficients for color scale. Default is none. If a value is given it applies
        symmetrically to positive and negative values.
    :param <bool> show: if True the figure will be displayed, else the axis object is returned.
        Default is True.
    :param <Optional[Union[List[str], Dict[str, str]]]> xcat_labels: optional list or 
        dictionary of labels for xcats. A list should be in the same order as xcats, a 
        dictionary should map from each xcat to its label.
    :param <Optional[Union[List[str], Dict[str, str]]]> xcat_secondary_labels: optional 
        list or dictionary of labels for xcats_secondary.
    :param <Union[float, int]> cbar_shrink: shrinkage factor of the color bar. Default is 0.5.
    :param <int> cbar_fontsize: font size of the color bar. Default is 12.
    :param <Dict> **kwargs: Arbitrary keyword arguments that are passed to seaborn.heatmap.

    N.B:. The function displays the heatmap of a correlation matrix across categories or
    cross-sections (depending on which parameter has received multiple elements).
    """
    df["real_date"] = pd.to_datetime(df["real_date"], format="%Y-%m-%d")
    col_names = ["cid", "xcat", "real_date", val]
    df = df[col_names]

    if freq is not None:
        freq = _map_to_business_day_frequency(freq=freq, valid_freqs=["W", "M", "Q"])

    xcats = xcats if isinstance(xcats, list) else [xcats]

    if max_color is not None:
        assert isinstance(max_color, float), "Parameter max_color must be type <float>."

    if not isinstance(cbar_shrink, (int, float)):
        raise ValueError(
            "The parameter `cbar_shrink` must be of type <int> or <float>."
        )
    if not isinstance(cbar_fontsize, int):
        raise ValueError("The parameter `cbar_fontsize` must be of type <int>.")

    min_color = None if max_color is None else -max_color
    mask = None
    xlabel = ""
    ylabel = ""

    missing_data_msg = (
        "The provided dataframe does not contain any data for the "
        "specified categories: {xcats}. Please check the data."
    )

    xcat_labels = _parse_xcat_labels(xcats, xcat_labels)

    # If more than one set of xcats or cids have been supplied.
    if xcats_secondary or cids_secondary:
        if xcats_secondary:
            xcat_secondary_labels = _parse_xcat_labels(
                xcats_secondary, xcat_secondary_labels
            )
            xcats_secondary = (
                xcats_secondary
                if isinstance(xcats_secondary, list)
                else [xcats_secondary]
            )
        else:
            xcats_secondary = xcats
            xcat_secondary_labels = xcat_labels

        if not cids_secondary:
            cids_secondary = cids

        df1, xcats, cids = reduce_df(df.copy(), xcats, cids, start, end, out_all=True)
        df2, xcats_secondary, cids_secondary = reduce_df(
            df.copy(), xcats_secondary, cids_secondary, start, end, out_all=True
        )
        for _df, _xc in zip([df1, df2], [xcats, xcats_secondary]):
            if _df.empty:
                raise ValueError(missing_data_msg.format(xcats=_xc))

        s_date = min(df1["real_date"].min(), df2["real_date"].min()).strftime(
            "%Y-%m-%d"
        )
        e_date = max(df1["real_date"].max(), df2["real_date"].max()).strftime(
            "%Y-%m-%d"
        )

        # If only one xcat, we will compute cross sectional correlation.
        if len(xcats) == 1 and len(xcats_secondary) == 1:
            df_w1: pd.DataFrame = _transform_df_for_cross_sectional_corr(
                df=df1, val=val, freq=freq
            )
            df_w2: pd.DataFrame = _transform_df_for_cross_sectional_corr(
                df=df2, val=val, freq=freq
            )

            if title is None:
                title = (
                    f"Cross-sectional correlation of {xcats[0]} and {xcats_secondary[0]} "
                    f"from {s_date} to "
                    f"{e_date}"
                )
            xlabel = f"{xcat_labels[xcats[0]]} cross-sections"
            ylabel = f"{xcat_secondary_labels[xcats_secondary[0]]} cross-sections"

        # If more than one xcat in at least one set, we will compute cross category
        # correlation.
        else:
            df_w1: pd.DataFrame = _transform_df_for_cross_category_corr(
                df=df1, xcats=xcats, val=val, freq=freq, lags=lags
            )
            df_w2: pd.DataFrame = _transform_df_for_cross_category_corr(
                df=df2, xcats=xcats_secondary, val=val, freq=freq, lags=lags_secondary
            )
            df_w1 = df_w1.rename(columns=xcat_labels)
            df_w2 = df_w2.rename(columns=xcat_secondary_labels)
            if title is None:
                title = f"Cross-category correlation from {s_date} to " f"{e_date}"
        corr = (
            pd.concat([df_w1, df_w2], axis=1, keys=["df_w1", "df_w2"])
            .corr()
            .loc["df_w2", "df_w1"]
        )

        if cluster:
            # Since the correlation matrix is not necessarily symmetric, clustering is
            # done in two stages.
            if corr.shape[0] > 1:
                corr = _cluster_correlations(corr=corr, is_symmetric=False)

            if corr.shape[1] > 1:
                corr = _cluster_correlations(corr=corr.T, is_symmetric=False).T

    # If there is only one set of xcats and cids.
    else:
        df, xcats, cids = reduce_df(df, xcats, cids, start, end, out_all=True)
        if df.empty:
            raise ValueError(missing_data_msg.format(xcats=xcats))

        s_date: str = df["real_date"].min().strftime("%Y-%m-%d")
        e_date: str = df["real_date"].max().strftime("%Y-%m-%d")

        if len(xcats) == 1:
            df_w = _transform_df_for_cross_sectional_corr(df=df, val=val, freq=freq)

            if title is None:
                title = (
                    f"Cross-sectional correlation of {xcats[0]} from {s_date} to "
                    f"{e_date}"
                )

        else:
            df_w = _transform_df_for_cross_category_corr(
                df=df, xcats=xcats, val=val, freq=freq, lags=lags
            )

            if title is None:
                title = f"Cross-category correlation from {s_date} to {e_date}"

        df_w = df_w.rename(columns=xcat_labels)
        corr = df_w.corr(method="pearson")

        if cluster:
            corr = _cluster_correlations(corr=corr, is_symmetric=True)

        # Mask for the upper triangle.
        # Return a copy of an array with the elements below the k-th diagonal zeroed. The
        # mask is implemented because correlation coefficients are symmetric.
        mask: np.ndarray = np.triu(np.ones_like(corr, dtype=bool))

    ax: plt.Axes
    fig, ax = plt.subplots(figsize=size)
    with sns.axes_style("white"):
        with sns.axes_style("ticks"):
            sns.heatmap(
                corr,
                mask=mask,
                cmap="vlag_r",
                center=0,
                vmin=min_color,
                vmax=max_color,
                square=False,
                linewidths=0.5,
                cbar_kws={"shrink": cbar_shrink},
                xticklabels=True,
                yticklabels=True,
                **kwargs,
            )

            cbar = ax.collections[0].colorbar
            cbar.ax.tick_params(labelsize=cbar_fontsize)

    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.set_title(title, fontsize=title_fontsize)

    ax.tick_params(axis="x", labelsize=fontsize)
    ax.tick_params(axis="y", labelsize=fontsize)

    plt.tight_layout()
    if show:
        plt.show()
        return
    else:
        return ax


def _parse_xcat_labels(xcats: List[str], xcat_labels: Union[List[str], Dict[str, str]]):
    """
    Parse xcat labels for correlation plot.

    :param <List[str]> xcats: extended categories to be correlated.
    :param <Union[List[str], Dict[str, str]]> xcat_labels: optional list or dictionary of
        labels for the extended categories.
    """
    labels_dict = {}
    if xcat_labels is not None:
        assert len(xcat_labels) == len(xcats), (
            "The number of labels provided for the extended categories must match the "
            "number of extended categories."
        )
        if isinstance(xcat_labels, list):
            for xcat, xcat_label in zip(xcats, xcat_labels):
                labels_dict[xcat] = xcat_label
        elif isinstance(xcat_labels, dict):
            labels_dict = xcat_labels
        else:
            raise ValueError("The xcats parameter must be a list or a dictionary.")
    else:
        labels_dict = {xcat: xcat for xcat in xcats}
    return labels_dict


def _transform_df_for_cross_sectional_corr(
    df: pd.DataFrame, val: str = "value", freq: str = None
) -> pd.DataFrame:
    """
    Pivots dataframe and down-samples according to the specified frequency so that
    correlation can be calculated between cross-sections.

    :param <pd.Dataframe> df: standardized JPMaQS DataFrame.
    :param <str> val: name of column that contains the values of interest. Default is
        'value'.
    :param <str> freq: Down-sampling frequency option. By default values are not
        down-sampled.

    :return <pd.Dataframe>: The transformed dataframe.
    """
    df_w = df.pivot(index="real_date", columns="cid", values=val)
    if freq is not None:
        df_w = df_w.resample(freq).mean()

    return df_w


def _transform_df_for_cross_category_corr(
    df: pd.DataFrame, xcats: List[str], val: str, freq: str = None, lags: dict = None
) -> pd.DataFrame:
    """
    Pivots dataframe and down-samples according to the specified frequency so that
    correlation can be calculated between extended categories.

    :param <pd.Dataframe> df: standardized JPMaQS DataFrame.
    :param <List[str]> xcats: extended categories to be correlated.
    :param <str> val: name of column that contains the values of interest. Default is
        'value'.
    :param <str> freq: Down-sampling frequency. By default values are not down-sampled.
    :param <dict> lags: optional dictionary of lags applied to respective categories.

    :return <pd.Dataframe>: The transformed dataframe.
    """
    df_w: pd.DataFrame = df.pivot(
        index=("cid", "real_date"), columns="xcat", values=val
    )
    # Down-sample according to the passed frequency.
    if freq is not None:
        df_w = df_w.groupby(
            [pd.Grouper(level="cid"), pd.Grouper(level="real_date", freq=freq)]
        ).mean()

    # Apply the lag mechanism, to the respective categories, after the down-sampling.
    if lags is not None:
        df_w, xcat_tracker = lag_series(df_w=df_w, lags=lags, xcats=xcats)

        # Order the correlation DataFrame to reflect the order of the categories
        # parameter. Will replace the official category name with the lag appended name.
        order = [
            [x] if x not in xcat_tracker.keys() else xcat_tracker[x] for x in xcats
        ]
        order = list(itertools.chain(*order))
    else:
        order = xcats

    df_w = df_w[order]
    return df_w


def lag_series(
    df_w: pd.DataFrame, lags: dict, xcats: List[str]
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """
    Method used to lag respective categories.

    :param <pd.DataFrame> df_w: multi-index DataFrame where the columns are the
        categories, and the two indices are the cross-sections and real-dates.
    :param <dict> lags: dictionary of lags applied to respective categories.
    :param <List[str]> xcats: extended categories to be correlated.
    """

    lag_type = "The lag data structure must be of type <dict>."
    assert isinstance(lags, dict), lag_type

    lag_xcats = (
        f"The categories referenced in the lagged dictionary must be "
        f"present in the defined DataFrame, {xcats}."
    )
    assert set(lags.keys()).issubset(set(xcats)), lag_xcats

    # Modify the dictionary to adjust for single categories having multiple lags.
    # The respective lags will be held inside a list.
    lag_copy = {}
    xcat_tracker: defaultdict = defaultdict(list)
    for xcat, shift in lags.items():
        if isinstance(shift, int):
            lag_copy[xcat + f"_L{shift}"] = shift
            xcat_tracker[xcat].append(xcat + f"_L{shift}")
        else:
            xcat_temp = [xcat + f"_L{s}" for s in shift]
            # Merge the two dictionaries.
            lag_copy = {**lag_copy, **dict(zip(xcat_temp, shift))}
            xcat_tracker[xcat].extend(xcat_temp)

    df_w_copy = df_w.copy()
    # Handle for multi-index DataFrame. The interior index represents the
    # timestamps.
    for xcat, shift in lag_copy.items():
        category = xcat[:-3]
        clause = isinstance(lags[category], list)
        first_lag = category in df_w.columns

        if clause and not first_lag:
            # Duplicate the column if multiple lags on the same category and the
            # category's first lag has already been implemented. Always access
            # the series from the original DataFrame.
            df_w[xcat] = df_w_copy[category]
        else:
            # Otherwise, modify the name.
            df_w = df_w.rename(columns={category: xcat})
        # Shift the respective column (name will have been adjusted to reflect
        # lag).
        df_w[xcat] = df_w.groupby(level=0)[xcat].shift(shift)

    return df_w, xcat_tracker


def _cluster_correlations(
    corr: pd.DataFrame, is_symmetric: bool = False
) -> pd.DataFrame:
    """
    Rearrange a correlation dataframe so that more similar values are clustered.

    :param <pd.Dataframe> corr: dataframe representing a correlation matrix.
    :param <bool> is_symmetric: if True, rows and columns are rearranged identically.
        If False, only rows are reordered.

    :return <pd.Dataframe>: The sorted correlation dataframe.
    """
    # Get pairwise distances of the dataframe's rows.
    d = sch.distance.pdist(corr)

    # Perform hierarchical / agglomerative clustering. The clustering method used is
    # Farthest Point Algorithm.
    L = sch.linkage(d, method="complete")

    # The second parameter is the distance threshold, t, which will determine the
    # "number" of clusters. If the distance threshold is too small, none of the data
    # points will form a cluster, so n different clusters are returned.
    # If there are any clusters, the categories contained in the cluster will be
    # adjacent.
    ind = sch.fcluster(L, 0.5 * d.max(), "distance")

    indices = [corr.index.tolist()[i] for i in list((np.argsort(ind)))]

    if is_symmetric:
        corr = corr.loc[indices, indices]
    else:
        corr = corr.loc[indices, :]

    return corr


if __name__ == "__main__":
    np.random.seed(0)

    # Un-clustered correlation matrices.

    cids = ["AUD", "CAD", "GBP", "USD", "NZD", "EUR"]
    cids_dmsc = ["CHF", "NOK", "SEK"]
    cids_dmec = ["DEM", "ESP", "FRF", "ITL", "NLG"]
    cids += cids_dmec
    cids += cids_dmsc
    xcats = ["XR", "CRY"]

    df_cids = pd.DataFrame(
        index=cids, columns=["earliest", "latest", "mean_add", "sd_mult"]
    )

    df_cids.loc["AUD"] = ["2010-01-01", "2020-12-31", 0.5, 2]
    df_cids.loc["CAD"] = ["2011-01-01", "2020-11-30", 0, 1]
    df_cids.loc["GBP"] = ["2012-01-01", "2020-11-30", -0.2, 0.5]
    df_cids.loc["USD"] = ["2010-01-01", "2020-12-30", -0.2, 0.5]
    df_cids.loc["NZD"] = ["2002-01-01", "2020-09-30", -0.1, 2]
    df_cids.loc["EUR"] = ["2002-01-01", "2020-09-30", -0.2, 2]
    df_cids.loc["DEM"] = ["2003-01-01", "2020-09-30", -0.3, 2]
    df_cids.loc["ESP"] = ["2003-01-01", "2020-09-30", -0.1, 2]
    df_cids.loc["FRF"] = ["2003-01-01", "2020-09-30", -0.2, 2]
    df_cids.loc["ITL"] = ["2004-01-01", "2020-09-30", -0.2, 0.5]
    df_cids.loc["NLG"] = ["2003-01-01", "2020-12-30", -0.1, 0.5]
    df_cids.loc["CHF"] = ["2003-01-01", "2020-12-30", -0.3, 2.5]
    df_cids.loc["NOK"] = ["2010-01-01", "2020-12-30", -0.1, 0.5]
    df_cids.loc["SEK"] = ["2010-01-01", "2020-09-30", -0.1, 0.5]

    df_xcats = pd.DataFrame(
        index=xcats,
        columns=["earliest", "latest", "mean_add", "sd_mult", "ar_coef", "back_coef"],
    )
    df_xcats.loc["XR",] = ["2010-01-01", "2020-12-31", 0.1, 1, 0, 0.3]
    df_xcats.loc["CRY",] = ["2010-01-01", "2020-10-30", 1, 2, 0.95, 0.5]

    dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)

    start = "2012-01-01"
    end = "2020-09-30"

    lag_dict = {"XR": [0, 2, 5]}

    # Clustered correlation matrices. Test hierarchical clustering.
    view_correlation(
        df=dfd,
        xcats=["XR", "CRY"],
        # xcats_secondary=["CRY", "XR"],
        cids=cids,
        cids_secondary=None,
        start=start,
        end=end,
        val="value",
        freq=None,
        cluster=True,
        title="Correlation Matrix",
        size=(14, 8),
        max_color=None,
        lags=None,
        lags_secondary=None,
        annot=True,
        fmt=".2f",
        xcat_labels=["Returns", "Carry"],
        # xcat_secondary_labels={"XR": "Excess returns", "CRY": "Carry"},
    )
