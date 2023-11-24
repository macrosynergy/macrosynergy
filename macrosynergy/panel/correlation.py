"""
Functions used to visualize correlations across categories or cross-sections of
panels.

::docs::correl_matrix::sort_first::
"""
import itertools
import pandas as pd
from typing import List, Union, Tuple, Dict, Optional, Any
from collections import defaultdict

from macrosynergy.management.utils import reduce_df
from macrosynergy.management.simulate import make_qdf
from macrosynergy.management.utils.df_utils import _map_to_business_day_frequency


def correlation(
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
):
    """
    Calculate correlation across categories or cross-sections of panels.

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
    :param <dict> lags: optional dictionary of lags applied to respective categories.
        The key will be the category and the value is the lag or lags. If a
        category has multiple lags applied, pass in a list of lag values. The lag factor
        will be appended to the category name in the correlation matrix.
        If xcats_secondary is not none, this parameter will specify lags for the
        categories in xcats.
        N.B.: Lags can include a 0 if the original should also be correlated.
    :param <dict> lags_secondary: optional dictionary of lags applied to the second set of
        categories if xcats_secondary is provided.
    """
    df["real_date"] = pd.to_datetime(df["real_date"], format="%Y-%m-%d")
    col_names = ["cid", "xcat", "real_date", val]
    df = df[col_names]

    if freq is not None:
        _map_to_business_day_frequency(freq=freq, valid_freqs=["W", "M", "Q"])

    xcats = xcats if isinstance(xcats, list) else [xcats]

    # If more than one set of xcats or cids have been supplied.
    if xcats_secondary or cids_secondary:
        if xcats_secondary:
            xcats_secondary = (
                xcats_secondary
                if isinstance(xcats_secondary, list)
                else [xcats_secondary]
            )
        else:
            xcats_secondary = xcats

        if not cids_secondary:
            cids_secondary = cids

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

        corr = (
            pd.concat([df_w1, df_w2], axis=1, keys=["df_w1", "df_w2"])
            .corr()
            .loc["df_w2", "df_w1"]
        )

    # If there is only one set of xcats and cids.
    else:
        df, xcats, cids = reduce_df(df, xcats, cids, start, end, out_all=True)

        if len(xcats) == 1:
            df_w = _transform_df_for_cross_sectional_corr(df=df, val=val, freq=freq)

        else:
            df_w = _transform_df_for_cross_category_corr(
                df=df, xcats=xcats, val=val, freq=freq, lags=lags
            )

        corr = df_w.corr(method="pearson")

    return corr


def _transform_df_for_cross_sectional_corr(
    df: pd.DataFrame, val: str = "value", freq: str = None
) -> pd.DataFrame:
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