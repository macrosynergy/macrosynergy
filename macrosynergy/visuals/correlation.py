"""
Functions used to visualize correlations across categories or cross-sections of
panels.
"""
import pandas as pd
import numpy as np
import seaborn as sns
import scipy.cluster.hierarchy as sch
from matplotlib import pyplot as plt
from typing import Tuple
from macrosynergy.management.types import Numeric
import pandas as pd
import numpy as np
from typing import List, Union, Tuple, Dict, Optional, Any

from macrosynergy.management.types import Numeric
from macrosynergy.management.simulate import make_qdf
from macrosynergy.management.utils import _map_to_business_day_frequency
from macrosynergy.panel.correlation import _corr, _cross_corr, _preprocess_for_corr, _preprocess_for_cross_corr


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
    cluster: bool = True,
    lags: dict = None,
    lags_secondary: Optional[dict] = None,
    title: str = "",
    size: Tuple[float] = (14, 8),
    max_color: Numeric = None,
    xlabel: str = "",
    ylabel: str = "",
):
    """
    Calculate and visualize correlation across categories or cross-sections of panels.

    :param <pd.Dataframe> df: standardized JPMaQS DataFrame with the necessary columns:
        'cid', 'xcats', 'real_date' and at least one column with values of interest.
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
    :param <Numeric> max_color: maximum values of positive/negative correlation
        coefficients for color scale. Default is none. If a value is given it applies
        symmetrically to positive and negative values.
    :param <bool> show: if True the figure will be displayed. Default is True.

    N.B:. The function displays the heatmap of a correlation matrix across categories or
    cross-sections (depending on which parameter has received multiple elements).
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("The df argument must be a pandas DataFrane.")
    if not isinstance(cluster, bool):
        raise TypeError("The cluster argument must be a boolean.")
    if not isinstance(title, str):
        raise TypeError("The title argument must be a string.")
    if max_color is not None and not isinstance(max_color, Numeric):
        raise TypeError("Parameter max_color must be numeric.")
    if not isinstance(size, tuple):
        raise TypeError("The size argument must be a tuple.")
    if not isinstance(xlabel, str):
        raise TypeError("The xlabel argument must be a string.")
    if not isinstance(ylabel, str):
        raise TypeError("The ylabel argument must be a string.")
    
    df["real_date"] = pd.to_datetime(df["real_date"], format="%Y-%m-%d")
    col_names = ["cid", "xcat", "real_date", val]
    df = df[col_names]

    if freq is not None:
        freq = _map_to_business_day_frequency(freq=freq, valid_freqs=["W", "M", "Q"])

    xcats = xcats if isinstance(xcats, list) else [xcats]

    if max_color is not None and not isinstance(max_color, Numeric):
        raise TypeError("Parameter max_color must be numeric.")

    mask = False
    xlabel = ""
    ylabel = ""

    # If more than one set of xcats or cids have been supplied.
    if xcats_secondary or cids_secondary:
        (
            xcats,
            cids,
            xcats_secondary,
            cids_secondary,
            df_w1,
            df_w2,
        ) = _preprocess_for_cross_corr(
            df,
            xcats,
            cids,
            xcats_secondary,
            cids_secondary,
            start,
            end,
            val,
            freq,
            lags,
            lags_secondary,
        )
        
        title = _get_cross_corr_title(xcats, xcats_secondary, title, df_w1, df_w2)

        corr = _cross_corr(df_w1, df_w2)

    # If there is only one set of xcats and cids.
    else:
        df, xcats, cids, df_w = _preprocess_for_corr(df, xcats, cids, start, end, val, freq, lags)

        title = _get_corr_title(df_w, xcats, title)

        corr = _corr(df_w)

        mask: bool = True
    
    min_color = None if max_color is None else -max_color

    if cluster:
        # Since the correlation matrix is not necessarily symmetric, clustering is
        # done in two stages.
        if corr.shape[0] > 1:
            corr = _cluster_correlations(corr=corr, is_symmetric=False)

        if corr.shape[1] > 1:
            corr = _cluster_correlations(corr=corr.T, is_symmetric=False).T

    mask_array: np.ndarray = np.triu(np.ones_like(corr, dtype=bool)) if mask else None

    ax: plt.Axes
    fig, ax = plt.subplots(figsize=size)

    with sns.axes_style("white"):
        with sns.axes_style("ticks"):
            sns.heatmap(
                corr,
                mask=mask_array,
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


def _get_corr_title(df, xcats, title):
    s_date: str = df.index.min().strftime("%Y-%m-%d")
    e_date: str = df.index.max().strftime("%Y-%m-%d")

    if len(xcats) == 1:
        if title is None:
            title = (
                    f"Cross-sectional correlation of {xcats[0]} from {s_date} to "
                    f"{e_date}"
                )
    else:
        if title is None:
            title = f"Cross-category correlation from {s_date} to {e_date}"

    return title

def _get_cross_corr_title(xcats, xcats_secondary, title, df_w1, df_w2):
    s_date = min(df_w1.index.min(), df_w2.index.min()).strftime(
            "%Y-%m-%d"
        )
    e_date = max(df_w1.index.max(), df_w2.index.max()).strftime(
            "%Y-%m-%d"
        )
    if len(xcats) == 1 and len(xcats_secondary) == 1:
        if title is None:
            title = (
                        f"Cross-sectional correlation of {xcats[0]} and {xcats_secondary[0]} "
                        f"from {s_date} to "
                        f"{e_date}"
                    )

        if title is None:
            title = f"Cross-category correlation from {s_date} to {e_date}"
    return title


if __name__ == "__main__":

    from macrosynergy.panel.correlation import correlation

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
        xcats=["XR"],
        xcats_secondary=None,
        cids=cids,
        cids_secondary=None,
        start=start,
        end=end,
        val="value",
        freq=None,
        lags=None,
        lags_secondary=None,
        cluster=True,
    )

