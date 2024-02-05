"""
Functions used to calculate correlation across categories or cross-sections of
panels.
"""
import itertools
import pandas as pd
import numpy as np
from typing import List, Union, Tuple, Dict, Optional
from collections import defaultdict

from macrosynergy.management.utils import reduce_df
from macrosynergy.management.simulate import make_qdf
from macrosynergy.management.utils import _map_to_business_day_frequency


def corr(
    df: pd.DataFrame,
    xcats: Union[str, List[str]] = None,
    cids: List[str] = None,
    xcats_secondary: Optional[Union[str, List[str]]] = None,
    cids_secondary: Optional[List[str]] = None,
    start: str = "2000-01-01",
    end: str = None,
    val: str = "value",
    freq: str = None,
    lags: dict = None,
    lags_secondary: Optional[dict] = None,
    return_dates: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, str, str]]:
    """
    Calculate correlation across categories or cross-sections of panels.

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
    :param <bool> return_dates: if True, the earliest and latest dates of the reduced
        DataFrame are returned.

    :return <Union[pd.DataFrame, Tuple[pd.DataFrame, str, str]]>: Returns either the
        correlation DataFrame or a Tuple containing the correlation DataFrame, along
        with the start and end dates of the reduced input DataFrame.
    """

    df["real_date"] = pd.to_datetime(df["real_date"], format="%Y-%m-%d")
    col_names = ["cid", "xcat", "real_date", val]
    df = df[col_names]

    if freq is not None:
        freq = _map_to_business_day_frequency(freq=freq, valid_freqs=["W", "M", "Q"])

    xcats = xcats if isinstance(xcats, list) else [xcats]

    # If more than one set of xcats or cids have been supplied.
    if xcats_secondary or cids_secondary:
        (
            xcats,
            cids,
            xcats_secondary,
            cids_secondary,
            df_w1,
            df_w2,
        ) = _preprocess_for_two_set_corr(
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

        s_date1, e_date1 = _get_dates(df_w1)
        s_date2, e_date2 = _get_dates(df_w2)
        s_date = min(s_date1, s_date2).strftime("%Y-%m-%d")
        e_date = max(e_date1, e_date2).strftime("%Y-%m-%d")

        corr_df = _two_set_corr(df_w1, df_w2)

    # If there is only one set of xcats and cids.
    else:
        df, xcats, cids, df_w = _preprocess_for_corr(
            df, xcats, cids, start, end, val, freq, lags
        )
        s_date, e_date = _get_dates(df_w)
        s_date = s_date.strftime("%Y-%m-%d")
        e_date = e_date.strftime("%Y-%m-%d")

        corr_df = _corr(df_w)

    if return_dates:
        return corr_df, s_date, e_date
    else:
        return corr_df


def _preprocess_for_corr(
    df: pd.DataFrame,
    xcats: List[str],
    cids: List[str],
    start: str,
    end: str,
    val: str,
    freq: str,
    lags: dict,
):
    """
    Method used to preprocess the DataFrame for computing correlation.

    :param <pd.Dataframe> df: standardized JPMaQS DataFrame.
    :param <List[str]> xcats: extended categories to be correlated.
    :param <List[str]> cids: cross sections to be correlated.
    :param <str> start: earliest date in ISO format.
    :param <str> end: latest date in ISO format.
    :param <str> val: name of column that contains the values of interest. Default is
        'value'.
    :param <str> freq: Down-sampling frequency option. By default values are not
        down-sampled.
    :param <dict> lags: optional dictionary of lags applied to respective categories.
    """
    df, xcats, cids = reduce_df(df, xcats, cids, start, end, out_all=True)

    if len(xcats) == 1:
        df_w = _transform_df_for_cross_sectional_corr(df=df, val=val, freq=freq)
    else:
        df_w = _transform_df_for_cross_category_corr(
            df=df, xcats=xcats, val=val, freq=freq, lags=lags
        )

    return df, xcats, cids, df_w


def _corr(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate correlation of a DataFrame.

    :param <pd.Dataframe> df: a pandas DataFrame.

    :return <pd.Dataframe>: The correlation DataFrame.
    """
    return df.corr(method="pearson")


def _two_set_corr(df: pd.DataFrame, df_secondary: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate correlation between two DataFrames.

    :param <pd.Dataframe> df: a pandas DataFrame.
    :param <pd.Dataframe> df_secondary: a pandas DataFrame.

    :return <pd.Dataframe>: The correlation DataFrame.
    """
    corr = (
        pd.concat([df, df_secondary], axis=1, keys=["df_w1", "df_w2"])
        .corr()
        .loc["df_w2", "df_w1"]
    )
    return corr


def _preprocess_for_two_set_corr(
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
):
    """
    Method used to preprocess the DataFrame for correlation between two sets of xcats or cids.

    :param <pd.Dataframe> df: standardized JPMaQS DataFrame with the necessary columns:
        'cid', 'xcat', 'real_date' and at least one column with values of interest.
    :param <List[str]> xcats: extended categories to be correlated.
    :param <List[str]> cids: cross sections to be correlated.
    :param <List[str]> xcats_secondary: an optional second set of extended categories.
    :param <List[str]> cids_secondary: an optional second list of cross sections.
    :param <str> start: earliest date in ISO format.
    :param <str> end: latest date in ISO format.
    :param <str> val: name of column that contains the values of interest.
    :param <str> freq: frequency option.
    :param <dict> lags: optional dictionary of lags applied to respective categories.
    :param <dict> lags_secondary: optional dictionary of lags applied to the second set of
        categories if xcats_secondary is provided.

    :return <Tuple[List, List, List, List, pd.DataFrame, pd.DataFrame]>:
        The updated xcats, cids, xcats_secondary, cids_secondary, df_w1, df_w2.
    """

    xcats_secondary, cids_secondary = _handle_secondary_args(
        xcats, cids, xcats_secondary, cids_secondary
    )

    # Todo: ensure secondary cids and xcats are present in the DataFrame.

    df1, xcats, cids = reduce_df(df.copy(), xcats, cids, start, end, out_all=True)
    df2, xcats_secondary, cids_secondary = reduce_df(
        df.copy(), xcats_secondary, cids_secondary, start, end, out_all=True
    )

    # If only one xcat, we will compute cross sectional correlation.
    if len(xcats) == 1 and len(xcats_secondary) == 1:
        df_w1: pd.DataFrame = _transform_df_for_cross_sectional_corr(
            df=df1, val=val, freq=freq
        )
        df_w2: pd.DataFrame = _transform_df_for_cross_sectional_corr(
            df=df2, val=val, freq=freq
        )

    # If more than one xcat in at least one set, we will compute cross category
    # correlation.
    else:
        df_w1: pd.DataFrame = _transform_df_for_cross_category_corr(
            df=df1, xcats=xcats, val=val, freq=freq, lags=lags
        )
        df_w2: pd.DataFrame = _transform_df_for_cross_category_corr(
            df=df2, xcats=xcats_secondary, val=val, freq=freq, lags=lags_secondary
        )

    return xcats, cids, xcats_secondary, cids_secondary, df_w1, df_w2


def _handle_secondary_args(
    xcats: Union[str, List[str]],
    cids: List,
    xcats_secondary: Union[str, List[str]],
    cids_secondary: List,
) -> Tuple[List, List]:
    """
    Method used to handle the optional secondary arguments passed to the correlation function.

    :param <Union[str, List[str]]> xcats: extended categories to be correlated.
    :param <List[str]> cids: cross sections to be correlated.
    :param <Union[str, List[str]]> xcats_secondary: an optional second set of extended
        categories.
    :param <List[str]> cids_secondary: an optional second list of cross sections.

    :return <Tuple[List, List]>: The updated secondary xcats and cids.
    """
    if xcats_secondary:
        xcats_secondary = (
            xcats_secondary if isinstance(xcats_secondary, list) else [xcats_secondary]
        )
    else:
        xcats_secondary = xcats

    if not cids_secondary:
        cids_secondary = cids

    return xcats_secondary, cids_secondary


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


def _get_dates(df: pd.DataFrame) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """
    Get earliest and latest date in DataFrame.

    :param <pd.DataFrame> df: pandas DataFrame either indexed by date
        or multi-indexed by cross-section and date.

    :return <Tuple[pd.Timestamp, pd.Timestamp]>: earliest and latest date in DataFrame.
    """
    if isinstance(df.index, pd.MultiIndex):
        s_date: str = df.index.get_level_values(1).min()
        e_date: str = df.index.get_level_values(1).max()
    else:
        s_date: str = df.index.min()
        e_date: str = df.index.max()
    return s_date, e_date


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
    corr_df = corr(
        df=dfd,
        xcats=["XR"],
        # xcats_secondary=["CRY"],
        cids=cids,
        # cids_secondary=cids[:2],
        start=start,
        end=end,
        val="value",
        freq=None,
        # title="Correlation Matrix",
        lags=None,
        lags_secondary=None,
    )
    print(corr_df)
