"""
Functions used to visualize correlations across categories or cross-sections of
panels.

::docs::correl_matrix::sort_first::
"""
import itertools
import pandas as pd
import numpy as np
import seaborn as sns
import scipy.cluster.hierarchy as sch
from matplotlib import pyplot as plt
from typing import List, Union, Tuple
from collections import defaultdict

from macrosynergy.management.check_availability import reduce_df
from macrosynergy.management.simulate_quantamental_data import make_qdf


def lag_series(df_w: pd.DataFrame, lags: dict, xcats: List[str]):
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
    xcat_tracker = defaultdict(list)
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


def correl_matrix(
    df: pd.DataFrame,
    xcats: Union[str, List[str], List[List[str]]] = None,
    cids: Union[List[str], List[List[str]]] = None,
    start: str = "2000-01-01",
    end: str = None,
    val: str = "value",
    freq: str = None,
    cluster: bool = False,
    lags: dict = None,
    title: str = None,
    size: Tuple[float] = (14, 8),
    max_color: float = None,
):
    """
    Visualize correlation across categories or cross-sections of panels.

    :param <pd.Dataframe> df: standardized JPMaQS DataFrame with the necessary columns:
        'cid', 'xcats', 'real_date' and at least one column with values of interest.
    :param <List[str]> xcats: extended categories to be correlated. Default is all in the
        DataFrame. If xcats contains only one category the correlation coefficients
        across cross sections are displayed. If xcats contains more than one category,
        the correlation coefficients across categories are displayed. Additionally, the
        order of the xcats received will be mirrored in the correlation matrix. If xcats is
        a list containing two lists of xcats, correlations will be calculated between these two sets.
    :param <List[str]> cids: cross sections to be correlated. Default is all in the
        DataFrame. If cids is a list containing two lists of cids, correlations will be calculated
        and visualized between these two sets.
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
        N.B.: Lags can include a 0 if the original should also be correlated.
    :param <str> title: chart heading. If none is given, a default title is used.
    :param <Tuple[float]> size: two-element tuple setting width/height of figure. Default
        is (14, 8).
    :param <float> max_color: maximum values of positive/negative correlation
        coefficients for color scale. Default is none. If a value is given it applies
        symmetrically to positive and negative values.

    N.B:. The function displays the heatmap of a correlation matrix across categories or
    cross-sections (depending on which parameter has received multiple elements).
    """
    df["real_date"] = pd.to_datetime(df["real_date"], format="%Y-%m-%d")
    col_names = ["cid", "xcat", "real_date", val]
    df = df[col_names]

    if freq is not None:
        freq_options = ["W", "M", "Q"]
        error_message = (
            f"Frequency parameter must be one of the following options:"
            f"{freq_options}."
        )
        assert freq in freq_options, error_message

    xcats = xcats if isinstance(xcats, list) else [xcats]

    if max_color is not None:
        assert isinstance(max_color, float), "Parameter max_color must be type <float>."

    min_color = None if max_color is None else -max_color
    mask = None
    xlabel = ""
    ylabel = ""

    if (len(xcats) == 2 and _is_list_of_lists(xcats)) or (
        len(cids) == 2 and _is_list_of_lists(cids)
    ):
        if (len(xcats) == 2 and _is_list_of_lists(xcats)) and (
            len(cids) == 2 and _is_list_of_lists(cids)
        ):
            xcats1, xcats2 = xcats
            cids1, cids2 = cids
        elif len(xcats) == 2 and _is_list_of_lists(xcats):
            xcats1, xcats2 = xcats
            cids1 = cids
            cids2 = cids
        else:
            xcats1 = xcats
            xcats2 = xcats
            cids1, cids2 = cids

        df1, xcats1, cids1 = reduce_df(
            df.copy(), xcats1, cids1, start, end, out_all=True
        )
        df2, xcats2, cids2 = reduce_df(
            df.copy(), xcats2, cids2, start, end, out_all=True
        )

        s_date = min(df1["real_date"].min(), df2["real_date"].min()).strftime("%Y-%m-%d")
        e_date = max(df1["real_date"].max(), df2["real_date"].max()).strftime("%Y-%m-%d")
    
        if len(xcats1) == 1 and len(xcats2) == 1:
            df_w1 = _transform_df_for_cross_sectional_corr(df1, val, freq)
            df_w2 = _transform_df_for_cross_sectional_corr(df2, val, freq)

            if title is None:
                title = (
                    f"Cross-sectional correlation of {xcats1[0]} and {xcats2[0]} from {s_date} to "
                    f"{e_date}"
                )
            xlabel = f"{xcats1[0]} cross-sections"
            ylabel = f"{xcats2[0]} cross-sections"
        else:
            df_w1 = _transform_df_for_cross_category_corr(df=df1, xcats=xcats1, val=val, freq=freq, lags=lags)
            df_w2 = _transform_df_for_cross_category_corr(df=df2, xcats=xcats2, val=val, freq=freq, lags=lags)

            if title is None:
                title = (
                    f"Cross-category correlation of {xcats[0]} from {s_date} to "
                    f"{e_date}"
                )
        corr = pd.concat([df_w1, df_w2], axis=1, keys=['df_w1', 'df_w2']).corr().loc['df_w2', 'df_w1']

        if cluster:
            # Since the correlation matrix is not necessarily symmetric, clustering is done in two stages.
            if corr.shape[0] > 1:
                corr = _cluster_correlations(corr=corr, is_symmetric=False)

            if corr.shape[1] > 1:
                corr = _cluster_correlations(corr=corr.T, is_symmetric=False).T
    else:
        df, xcats, cids = reduce_df(df, xcats, cids, start, end, out_all=True)

        s_date = df["real_date"].min().strftime("%Y-%m-%d")
        e_date = df["real_date"].max().strftime("%Y-%m-%d")

        if len(xcats) == 1:
            df_w = _transform_df_for_cross_sectional_corr(df=df, val=val, freq=freq)

            if title is None:
                title = (
                    f"Cross-sectional correlation of {xcats[0]} from {s_date} to "
                    f"{e_date}"
                )

        else:
            df_w = _transform_df_for_cross_category_corr(df=df, xcats=xcats, val=val, freq=freq, lags=lags)

            if title is None:
                title = f"Cross-category correlation from {s_date} to {e_date}"

        corr = df_w.corr(method="pearson")

        if cluster:
            corr = _cluster_correlations(corr=corr, is_symmetric=True)

        # Mask for the upper triangle.
        # Return a copy of an array with the elements below the k-th diagonal zeroed. The
        # mask is implemented because correlation coefficients are symmetric.
        mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=size)
    sns.set(style="ticks")
    sns.heatmap(
        corr,
        mask=mask,
        cmap="vlag_r",
        center=0,
        vmin=min_color,
        vmax=max_color,
        square=False,
        linewidths=0.5,
        cbar_kws={"shrink": 0.5},
    )

    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.set_title(title, fontsize=14)

    plt.show()


def _transform_df_for_cross_sectional_corr(
    df: pd.DataFrame, val: str = "value", freq: str = None
):
    """
    Pivots dataframe and down-samples according to the specified frequency so that
    correlation can be calculated between cross-sections.

    :param <pd.Dataframe> df: standardized JPMaQS DataFrame.
    :param <str> val: name of column that contains the values of interest. Default is
        'value'.
    :param <str> freq: Down-sampling frequency option. By default values are not down-sampled.

    :return <pd.Dataframe>: The transformed dataframe.
    """
    df_w = df.pivot(index="real_date", columns="cid", values=val)
    if freq is not None:
        df_w = df_w.resample(freq).mean()

    return df_w


def _transform_df_for_cross_category_corr(
    df: pd.DataFrame, xcats: List[str], val: str, freq: str = None, lags: dict = None
):
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
    df_w = df.pivot(index=("cid", "real_date"), columns="xcat", values=val)
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
    if lags is not None:
        order = [
            [x] if x not in xcat_tracker.keys() else xcat_tracker[x] for x in xcats
        ]
        order = list(itertools.chain(*order))
    else:
        order = xcats

    df_w = df_w[order]
    return df_w


def _is_list_of_lists(ls: List):
    """
    Check if list contains only lists as elements.

    :param <List> ls: the list to be checked.

    :return <bool>: True if `ls` contains only elements of type List. False otherwise.
    """
    return all(map(lambda x: isinstance(x, list), ls))


def _cluster_correlations(corr: pd.DataFrame, is_symmetric: bool = False):
    """
    Rearrange a correlation dataframe so that more similar values are clustered.

    :param <pd.Dataframe> corr: dataframe representing a correlation matrix.
    :param <bool> is_symmetric: if True, rows and columns can be rearranged identically.
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

    df_cids = pd.DataFrame(index=cids, columns=['earliest', 'latest',
                                                'mean_add', 'sd_mult'])

    df_cids.loc['AUD'] = ['2010-01-01', '2020-12-31', 0.5, 2]
    df_cids.loc['CAD'] = ['2011-01-01', '2020-11-30', 0, 1]
    df_cids.loc['GBP'] = ['2012-01-01', '2020-11-30', -0.2, 0.5]
    df_cids.loc['USD'] = ['2010-01-01', '2020-12-30', -0.2, 0.5]
    df_cids.loc['NZD'] = ['2002-01-01', '2020-09-30', -0.1, 2]
    df_cids.loc['EUR'] = ['2002-01-01', '2020-09-30', -0.2, 2]
    df_cids.loc['DEM'] = ['2003-01-01', '2020-09-30', -0.3, 2]
    df_cids.loc['ESP'] = ['2003-01-01', '2020-09-30', -0.1, 2]
    df_cids.loc['FRF'] = ['2003-01-01', '2020-09-30', -0.2, 2]
    df_cids.loc['ITL'] = ['2004-01-01', '2020-09-30', -0.2, 0.5]
    df_cids.loc['NLG'] = ['2003-01-01', '2020-12-30', -0.1, 0.5]
    df_cids.loc['CHF'] = ['2003-01-01', '2020-12-30', -0.3, 2.5]
    df_cids.loc['NOK'] = ['2010-01-01', '2020-12-30', -0.1, 0.5]
    df_cids.loc['SEK'] = ['2010-01-01', '2020-09-30', -0.1, 0.5]

    df_xcats = pd.DataFrame(index=xcats, columns=['earliest', 'latest', 'mean_add',
                                                  'sd_mult', 'ar_coef', 'back_coef'])
    df_xcats.loc['XR', ] = ['2010-01-01', '2020-12-31', 0.1, 1, 0, 0.3]
    df_xcats.loc['CRY', ] = ['2010-01-01', '2020-10-30', 1, 2, 0.95, 0.5]

    dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)

    start = "2012-01-01"
    end = "2020-09-30"

    # Clustered correlation matrices. Test hierarchical clustering.
    correl_matrix(
        df=dfd,
        # xcats=[["XR", "CRY"], ["XR"]],
        xcats=[["XR"], ["CRY"]],
        # xcats=["XR"],
        # xcats=["CRY", "XR"],
        # cids=[["CHF", "NOK", "SEK"], ["CHF"]],
        cids=cids,
        start=start,
        end=end,
        val="value",
        freq=None,
        cluster=True,
        # title="Correlation Matrix",
        size=(14, 8),
        max_color=None,
    )
