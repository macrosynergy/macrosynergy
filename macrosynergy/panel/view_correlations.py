"""
Functions used to visualize correlations across categories or cross-sections of
panels.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Union, Tuple, Optional, Any
from macrosynergy.management.simulate import make_qdf

import macrosynergy.visuals as msv


def correl_matrix(
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
    max_color: float = None,
    show: bool = True,
    xcat_labels: Optional[Union[List[str], Dict[str, str]]] = None,
    xcat_secondary_labels: Optional[Union[List[str], Dict[str, str]]] = None,
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
    :param <float> max_color: maximum values of positive/negative correlation
        coefficients for color scale. Default is none. If a value is given it applies
        symmetrically to positive and negative values.
    :param <bool> show: if True the figure will be displayed. Default is True.
    :param <Optional[Union[List[str], Dict[str, str]]> xcat_labels: optional list or
        dictionary of labels for xcats. A list should be in the same order as xcats, a 
        dictionary should map from each xcat to its label.
    :param <Optional[Union[List[str], Dict[str, str]]]> xcat_secondary_labels: optional 
        list or dictionary of labels for xcats_secondary.
    :param <Dict> **kwargs: Arbitrary keyword arguments that are passed to seaborn.heatmap.

    N.B:. The function displays the heatmap of a correlation matrix across categories or
    cross-sections (depending on which parameter has received multiple elements).
    """

    msv.view_correlation(
        df=df,
        xcats=xcats,
        cids=cids,
        xcats_secondary=xcats_secondary,
        cids_secondary=cids_secondary,
        start=start,
        end=end,
        val=val,
        freq=freq,
        cluster=cluster,
        lags=lags,
        lags_secondary=lags_secondary,
        title=title,
        size=size,
        max_color=max_color,
        show=show,
        xcat_labels=xcat_labels,
        xcat_secondary_labels=xcat_secondary_labels,
        **kwargs,
    )


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
    correl_matrix(
        df=dfd,
        xcats=["XR", "CRY"],
        xcats_secondary=None,
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
    )
