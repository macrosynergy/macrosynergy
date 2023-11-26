"""
Utility functions for working with DataFrames.
"""

import itertools

from macrosynergy.management.types import QuantamentalDataFrame
from macrosynergy.management.constants import FREQUENCY_MAP
import warnings
from typing import Any, Dict, Iterable, List, Optional, Set, Union, overload

import numpy as np
import pandas as pd
import requests
import requests.compat
from .core import get_cid, get_xcat


def standardise_dataframe(
    df: pd.DataFrame, verbose: bool = False
) -> QuantamentalDataFrame:
    """
    Applies the standard JPMaQS Quantamental DataFrame format to a DataFrame.

    :param <pd.DataFrame> df: The DataFrame to be standardized.
    :param <bool> verbose: Whether to print warnings if the DataFrame is not in the
        correct format.
    :return <pd.DataFrame>: The standardized DataFrame.
    :raises <TypeError>: If the input is not a pandas DataFrame.
    :raises <ValueError>: If the input DataFrame is not in the correct format.
    """
    idx_cols: List[str] = QuantamentalDataFrame.IndexCols
    commonly_used_cols: List[str] = ["value", "grading", "eop_lag", "mop_lag"]
    if not set(df.columns).issuperset(set(idx_cols)):
        fail_str: str = (
            f"Error : Tried to standardize DataFrame but failed."
            f"DataFrame not in the correct format. Please ensure "
            f"that the DataFrame has the following columns: "
            f"'cid', 'xcat', 'real_date', along with any other "
            "variables you wish to include (e.g. 'value', 'mop_lag', "
            "'eop_lag', 'grading')."
        )

        try:
            dft: pd.DataFrame = df.reset_index()
            found_cols: list = dft.columns.tolist()
            fail_str += f"\nFound columns: {found_cols}"
            if not set(dft.columns).issuperset(set(idx_cols)):
                raise ValueError(fail_str)
            df = dft.copy()
        except:
            raise ValueError(fail_str)

        # check if there is atleast one more column
        if len(df.columns) < 4:
            raise ValueError(fail_str)

    df["real_date"] = pd.to_datetime(df["real_date"], format="%Y-%m-%d")
    df["cid"] = df["cid"].astype(str)
    df["xcat"] = df["xcat"].astype(str)
    df = df.sort_values(by=["real_date", "cid", "xcat"]).reset_index(drop=True)

    remaining_cols: Set[str] = set(df.columns) - set(idx_cols)

    df = df[idx_cols + sorted(list(remaining_cols))]

    # for every remaining col, try to convert to float
    for col in remaining_cols:
        try:
            df[col] = df[col].astype(float)
        except:
            pass

    non_idx_cols: list = sorted(list(set(df.columns) - set(idx_cols)))
    return_df: pd.DataFrame = df[idx_cols + non_idx_cols]
    assert isinstance(
        return_df, QuantamentalDataFrame
    ), "Failed to standardize DataFrame"
    return return_df


def drop_nan_series(
    df: pd.DataFrame, column: str = "value", raise_warning: bool = False
) -> QuantamentalDataFrame:
    """
    Drops any series that are entirely NaNs.
    Raises a user warning if any series are dropped.

    :param <pd.DataFrame> df: The dataframe to be cleaned.
    :param <str> column: The column to be used as the value column, defaults to
        "value".
    :param <bool> raise_warning: Whether to raise a warning if any series are dropped.
    :return <pd.DataFrame | QuantamentalDataFrame>: The cleaned DataFrame.
    :raises <TypeError>: If the input is not a pandas DataFrame.
    :raises <ValueError>: If the input DataFrame is not in the correct format.
    """
    if not isinstance(df, QuantamentalDataFrame):
        raise TypeError("Argument `df` must be a Quantamental DataFrame.")

    if not column in df.columns:
        raise ValueError(f"Column {column} not present in DataFrame.")

    if not df["value"].isna().any():
        return df

    if not isinstance(raise_warning, bool):
        raise TypeError("Error: The raise_warning argument must be a boolean.")

    df_orig: pd.DataFrame = df.copy()
    for cd, xc in df_orig.groupby(["cid", "xcat"]).groups:
        sel_series: pd.Series = df_orig[
            (df_orig["cid"] == cd) & (df_orig["xcat"] == xc)
        ]["value"]
        if sel_series.isna().all():
            if raise_warning:
                warnings.warn(
                    message=f"The series {cd}_{xc} is populated "
                    "with NaNs only, and will be dropped.",
                    category=UserWarning,
                )
            df = df[~((df["cid"] == cd) & (df["xcat"] == xc))]

    return df.reset_index(drop=True)


def qdf_to_ticker_df(df: pd.DataFrame, value_column: str = "value") -> pd.DataFrame:
    """
    Converts a standardized JPMaQS DataFrame to a wide format DataFrame
    with each column representing a ticker.

    :param <pd.DataFrame> df: A standardised quantamental dataframe.
    :param <str> value_column: The column to be used as the value column, defaults to
        "value". If the specified column is not present in the DataFrame, a column named
        "value" will be used. If there is no column named "value", the first
        column in the DataFrame will be used instead.
    :return <pd.DataFrame>: The converted DataFrame.
    """
    if not isinstance(df, QuantamentalDataFrame):
        raise TypeError("Argument `df` must be a QuantamentalDataFrame.")

    if not isinstance(value_column, str):
        raise TypeError("Argument `value_column` must be a string.")

    if not value_column in df.columns:
        cols: List[str] = list(set(df.columns) - set(QuantamentalDataFrame.IndexCols))
        if "value" in cols:
            value_column: str = "value"

        warnings.warn(
            f"Value column specified in `value_column` ({value_column}) "
            f"is not present in the DataFrame. Defaulting to {cols[0]}."
        )
        value_column: str = cols[0]

    df: pd.DataFrame = df.copy()

    df["ticker"] = df["cid"] + "_" + df["xcat"]
    # drop cid and xcat
    df = (
        df.drop(columns=["cid", "xcat"])
        .pivot(index="real_date", columns="ticker", values=value_column)
        .rename_axis(None, axis=1)
    )

    return df


def ticker_df_to_qdf(df: pd.DataFrame) -> QuantamentalDataFrame:
    """
    Converts a wide format DataFrame (with each column representing a ticker)
    to a standardized JPMaQS DataFrame.

    :param <pd.DataFrame> df: A wide format DataFrame.
    :return <pd.DataFrame>: The converted DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Argument `df` must be a pandas DataFrame.")

    # pivot to long format
    df = (
        df.stack(level=0)
        .reset_index()
        .rename(columns={0: "value", "level_1": "ticker"})
    )
    # split ticker using get_cid and get_xcat
    df["cid"] = get_cid(df["ticker"])
    df["xcat"] = get_xcat(df["ticker"])
    # drop ticker column

    df = df.drop(columns=["ticker"])

    # standardise and return
    return standardise_dataframe(df=df)


def apply_slip(
    df: pd.DataFrame,
    slip: int,
    cids: List[str],
    xcats: List[str],
    metrics: List[str],
    raise_error: bool = True,
) -> pd.DataFrame:
    """
    Applied a slip, i.e. a negative lag, to the target DataFrame
    for the given cross-sections and categories, on the given metrics.

    :param <pd.DataFrame> target_df: DataFrame to which the slip is applied.
    :param <int> slip: Slip to be applied.
    :param <List[str]> cids: List of cross-sections.
    :param <List[str]> xcats: List of categories.
    :param <List[str]> metrics: List of metrics to which the slip is applied.
    :return <pd.DataFrame> target_df: DataFrame with the slip applied.
    :raises <TypeError>: If the provided parameters are not of the expected type.
    :raises <ValueError>: If the provided parameters are semantically incorrect.
    """

    df = df.copy()
    if not (isinstance(slip, int) and slip >= 0):
        raise ValueError("Slip must be a non-negative integer.")

    if cids is None:
        cids = df["cid"].unique().tolist()
    if xcats is None:
        xcats = df["xcat"].unique().tolist()

    sel_tickers: List[str] = [f"{cid}_{xcat}" for cid in cids for xcat in xcats]
    df["tickers"] = df["cid"] + "_" + df["xcat"]

    if not set(sel_tickers).issubset(set(df["tickers"].unique())):
        if raise_error:
            raise ValueError(
                "Tickers targetted for applying slip are not present in the DataFrame.\n"
                f"Missing tickers: {sorted(list(set(sel_tickers) - set(df['tickers'].unique())))}"
            )
        else:
            warnings.warn(
                "Tickers targetted for applying slip are not present in the DataFrame.\n"
                f"Missing tickers: {sorted(list(set(sel_tickers) - set(df['tickers'].unique())))}"
            )

    slip: int = slip.__neg__()

    df[metrics] = df.groupby("tickers")[metrics].shift(slip)
    df = df.drop(columns=["tickers"])

    return df


def downsample_df_on_real_date(
    df: pd.DataFrame,
    groupby_columns: List[str] = [],
    freq: str = "M",
    agg: str = "mean",
):
    """
    Downsample JPMaQS DataFrame.

    :param <pd.Dataframe> df: standardized JPMaQS DataFrame with the necessary columns:
        'cid', 'xcats', 'real_date' and at least one column with values of interest.
    :param <List> groupby_columns: a list of columns used to group the DataFrame.
    :param <str> freq: frequency option. Per default the correlations are calculated
        based on the native frequency of the datetimes in 'real_date', which is business
        daily. Downsampling options include weekly ('W'), monthly ('M'), or quarterly
        ('Q') mean.
    :param <str> agg: aggregation method. Must be one of "mean" (default), "median",
        "min", "max", "first" or "last".

    :return <pd.DataFrame>: the downsampled DataFrame.
    """

    if not set(groupby_columns).issubset(df.columns):
        raise ValueError(
            "The columns specified in 'groupby_columns' were not found in the DataFrame."
        )

    if not isinstance(freq, str):
        raise TypeError("`freq` must be a string")
    else:
        freq: str = _map_to_business_day_frequency(freq)

    if not isinstance(agg, str):
        raise TypeError("`agg` must be a string")
    else:
        agg: str = agg.lower()
        if agg not in ["mean", "median", "min", "max", "first", "last"]:
            raise ValueError(
                "`agg` must be one of 'mean', 'median', 'min', 'max', 'first', 'last'"
            )

    return (
        df.set_index("real_date")
        .groupby(groupby_columns)
        .resample(freq)
        .agg(agg, numeric_only=True)
        .reset_index()
    )


def update_df(df: pd.DataFrame, df_add: pd.DataFrame, xcat_replace: bool = False):
    """
    Append a standard DataFrame to a standard base DataFrame with ticker replacement on
    the intersection.

    :param <pd.DataFrame> df: standardised base JPMaQS DataFrame with the following
        necessary columns: 'cid', 'xcat', 'real_date' and 'value'.
    :param <pd.DataFrame> df_add: another standardised JPMaQS DataFrame, with the latest
        values, to be added with the necessary columns: 'cid', 'xcat', 'real_date', and
        'value'. Columns that are present in the base DataFrame but not in the appended
        DataFrame will be populated with NaN values.
    :param <bool> xcat_replace: all series belonging to the categories in the added
        DataFrame will be replaced, rather than just the added tickers.
        N.B.: tickers are combinations of cross-sections and categories.

    :return <pd.DataFrame>: standardised DataFrame with the latest values of the modified
        or newly defined tickers added.
    """

    # index_cols = ["cid", "xcat", "real_date"]
    # Consider the other possible metrics that the DataFrame could be defined over

    df_cols = set(df.columns)
    df_add_cols = set(df_add.columns)

    error_message = f"The base DataFrame must be a Quantamental Dataframe."
    if not isinstance(df, QuantamentalDataFrame):
        raise TypeError(error_message)

    error_message = f"The added DataFrame must be a Quantamental Dataframe."
    if not isinstance(df_add, QuantamentalDataFrame):
        raise TypeError(error_message)
    
    error_message = ("The two Quantamental DataFrames must share at least "
                    "four columns including than 'real_date', 'cid', and 'xcat'.")
    
    all_cols = df_cols.union(df_add_cols)
    if all_cols != df_cols and all_cols != df_add_cols:
        raise ValueError(error_message)

    if not xcat_replace:
        df = update_tickers(df, df_add)

    else:
        df = update_categories(df, df_add)

    return df.reset_index(drop=True)


def df_tickers(df: pd.DataFrame):
    """
    Helper function used to delimit the tickers defined in a received DataFrame.

    :param <pd.DataFrame> df: standardised DataFrame.
    """
    cids_append = list(map(lambda c: c + "_", set(df["cid"])))
    tickers = list(itertools.product(cids_append, set(df["xcat"])))
    tickers = [c[0] + c[1] for c in tickers]

    return tickers


def update_tickers(df: pd.DataFrame, df_add: pd.DataFrame):
    """
    Method used to update aggregate DataFrame on a ticker level.

    :param <pd.DataFrame> df: aggregate DataFrame used to store all tickers.
    :param <pd.DataFrame> df_add: DataFrame with the latest values.

    """
    agg_df_tick = set(df_tickers(df))
    add_df_tick = set(df_tickers(df_add))

    df["ticker"] = df["cid"] + "_" + df["xcat"]

    # If the ticker is already defined in the DataFrame, replace with the new series
    # otherwise append the series to the aggregate DataFrame.
    df = df[~df["ticker"].isin(list(agg_df_tick.intersection(add_df_tick)))]

    df = pd.concat([df, df_add], axis=0, ignore_index=True)

    df = df.drop(["ticker"], axis=1)

    return df.sort_values(["xcat", "cid", "real_date"])


def update_categories(df: pd.DataFrame, df_add):
    """
    Method used to update the DataFrame on the category level.

    :param <pd.DataFrame> df: base DataFrame.
    :param <pd.DataFrame> df_add: appended DataFrame.

    """

    incumbent_categories = list(df["xcat"].unique())
    new_categories = list(df_add["xcat"].unique())

    # Union of both category columns from the two DataFrames.
    append_condition = set(incumbent_categories) | set(new_categories)
    intersect = list(set(incumbent_categories).intersection(set(new_categories)))

    if len(append_condition) == len(incumbent_categories + new_categories):
        df = pd.concat([df, df_add], axis=0, ignore_index=True)

    # Shared categories plus any additional categories previously not defined in the base
    # DataFrame.
    else:
        df = df[~df["xcat"].isin(intersect)]
        df = pd.concat([df, df_add], axis=0, ignore_index=True)

    return df


def reduce_df(
    df: pd.DataFrame,
    xcats: List[str] = None,
    cids: List[str] = None,
    start: str = None,
    end: str = None,
    blacklist: dict = None,
    out_all: bool = False,
    intersect: bool = False,
):
    """
    Filter DataFrame by xcats and cids and notify about missing xcats and cids.

    :param <pd.Dataframe> df: standardized JPMaQS DataFrame with the necessary columns:
        'cid', 'xcats', 'real_date' and 'value'.
    :param <List[str]> xcats: extended categories to be filtered on. Default is all in
        the DataFrame.
    :param <List[str]> cids: cross sections to be checked on. Default is all in the
        dataframe.
    :param <str> start: string representing the earliest date. Default is None.
    :param <str> end: string representing the latest date. Default is None.
    :param <dict> blacklist: cross-sections with date ranges that should be excluded from
        the data frame. If one cross-section has several blacklist periods append numbers
        to the cross-section code.
    :param <bool> out_all: if True the function returns reduced dataframe and selected/
        available xcats and cids.
        Default is False, i.e. only the DataFrame is returned
    :param <bool> intersect: if True only retains cids that are available for all xcats.
        Default is False.

    :return <pd.Dataframe>: reduced DataFrame that also removes duplicates or
        (for out_all True) DataFrame and available and selected xcats and cids.
    """

    dfx = df.copy()

    if start is not None:
        dfx = dfx[dfx["real_date"] >= pd.to_datetime(start)]

    if end is not None:
        dfx = dfx[dfx["real_date"] <= pd.to_datetime(end)]

    if blacklist is not None:
        masks = []
        for key, value in blacklist.items():
            filt1 = dfx["cid"] == key[:3]
            filt2 = dfx["real_date"] >= pd.to_datetime(value[0])
            filt3 = dfx["real_date"] <= pd.to_datetime(value[1])
            combined_mask = filt1 & filt2 & filt3
            masks.append(combined_mask)

        if masks:
            combined_mask = pd.concat(masks, axis=1).any(axis=1)
            dfx = dfx[~combined_mask]

    xcats_in_df = dfx["xcat"].unique()
    if xcats is None:
        xcats = sorted(xcats_in_df)
    else:
        xcats = [xcat for xcat in xcats if xcat in xcats_in_df]

    dfx = dfx[dfx["xcat"].isin(xcats)]

    if intersect:
        df_uns = dict(dfx.groupby("xcat")["cid"].unique())
        df_uns = {k: set(v) for k, v in df_uns.items()}
        cids_in_df = list(set.intersection(*list(df_uns.values())))
    else:
        cids_in_df = dfx["cid"].unique()

    if cids is None:
        cids = sorted(cids_in_df)
    else:
        cids = [cids] if isinstance(cids, str) else cids
        cids = [cid for cid in cids if cid in cids_in_df]

        cids = set(cids).intersection(cids_in_df)
        dfx = dfx[dfx["cid"].isin(cids)]

    if out_all:
        return dfx.drop_duplicates(), xcats, sorted(list(cids))
    else:
        return dfx.drop_duplicates()


def reduce_df_by_ticker(
    df: pd.DataFrame,
    ticks: List[str] = None,
    start: str = None,
    end: str = None,
    blacklist: dict = None,
):
    """
    Filter dataframe by xcats and cids and notify about missing xcats and cids

    :param <pd.Dataframe> df: standardized dataframe with the following columns:
        'cid', 'xcats', 'real_date'.
    :param <List[str]> ticks: tickers (cross sections + base categories)
    :param <str> start: string in ISO 8601 representing earliest date. Default is None.
    :param <str> end: string ISO 8601 representing the latest date. Default is None.
    :param <dict> blacklist: cross sections with date ranges that should be excluded from
        the dataframe. If one cross section has several blacklist periods append numbers
        to the cross section code.

    :return <pd.Dataframe>: reduced dataframe that also removes duplicates
    """

    dfx = df.copy()

    if start is not None:
        dfx = dfx[dfx["real_date"] >= pd.to_datetime(start)]

    if end is not None:
        dfx = dfx[dfx["real_date"] <= pd.to_datetime(end)]

    # Blacklisting by cross-section.
    if blacklist is not None:
        masks = []
        for key, value in blacklist.items():
            filt1 = dfx["cid"] == key[:3]
            filt2 = dfx["real_date"] >= pd.to_datetime(value[0])
            filt3 = dfx["real_date"] <= pd.to_datetime(value[1])
            combined_mask = filt1 & filt2 & filt3
            masks.append(combined_mask)

        if masks:
            combined_mask = pd.concat(masks, axis=1).any(axis=1)
            dfx = dfx[~combined_mask]

    dfx["ticker"] = dfx["cid"] + "_" + dfx["xcat"]
    ticks_in_df = dfx["ticker"].unique()
    if ticks is None:
        ticks = sorted(ticks_in_df)
    else:
        ticks = [tick for tick in ticks if tick in ticks_in_df]

    dfx = dfx[dfx["ticker"].isin(ticks)]

    return dfx.drop_duplicates()


def categories_df_aggregation_helper(dfx: pd.DataFrame, xcat_agg: str):
    """
    Helper method to down-sample each category in the DataFrame by aggregating over the
    intermediary dates according to a prescribed method.

    :param <List[str]> dfx: standardised DataFrame defined exclusively on a single
        category.
    :param <List[str]> xcat_agg: associated aggregation method for the respective
        category.

    """

    dfx = dfx.groupby(["xcat", "cid", "custom_date"])
    dfx = dfx.aggregate(xcat_agg, numeric_only=True).reset_index()

    if "real_date" in dfx.columns:
        dfx = dfx.drop(["real_date"], axis=1)
    dfx = dfx.rename(columns={"custom_date": "real_date"})

    return dfx


def categories_df_expln_df(
    df_w: pd.DataFrame, xpls: List[str], agg_meth: str, sum_condition: bool, lag: int
):
    """
    Produces the explanatory column(s) for the custom DataFrame.

    :param <pd.DataFrame> df_w: group-by DataFrame which has been down-sampled. The
        respective aggregation method will be applied.
    :param <List[str]> xpls: list of explanatory category(s).
    :param <str> agg_meth: aggregation method used for all explanatory variables.
    :param <dict> sum_condition: required boolean to negate erroneous zeros if the
        aggregate method used, for the explanatory variable, is sum.
    :param <int> lag: lag of explanatory category(s). Applied uniformly to each
        category.
    """

    dfw_xpls = pd.DataFrame()
    for xpl in xpls:
        if not sum_condition:
            xpl_col = df_w[xpl].agg(agg_meth).astype(dtype=np.float32)
        else:
            xpl_col = df_w[xpl].sum(min_count=1)

        if lag > 0:
            xpl_col = xpl_col.groupby(level=0).shift(lag)

        dfw_xpls[xpl] = xpl_col

    return dfw_xpls


def categories_df(
    df: pd.DataFrame,
    xcats: List[str],
    cids: List[str] = None,
    val: str = "value",
    start: str = None,
    end: str = None,
    blacklist: dict = None,
    years: int = None,
    freq: str = "M",
    lag: int = 0,
    fwin: int = 1,
    xcat_aggs: List[str] = ["mean", "mean"],
):
    """
    In principle, create custom two-categories DataFrame with appropriate frequency and,
    if applicable, lags.

    :param <pd.Dataframe> df: standardized JPMaQS DataFrame with the following necessary
        columns: 'cid', 'xcats', 'real_date' and at least one column with values of
        interest.
    :param <List[str]> xcats: extended categories involved in the custom DataFrame. The
        last category in the list represents the dependent variable, and the (n - 1)
        preceding categories will be the explanatory variables(s).
    :param <List[str]> cids: cross-sections to be included. Default is all in the
        DataFrame.
    :param <str> val: name of column that contains the values of interest. Default is
        'value'.
    :param <str> start: earliest date in ISO 8601 format. Default is None,
        i.e. earliest date in DataFrame is used.
    :param <str> end: latest date in ISO 8601 format. Default is None,
        i.e. latest date in DataFrame is used.
    :param <dict> blacklist: cross-sections with date ranges that should be excluded from
        the DataFrame. If one cross section has several blacklist periods append numbers
        to the cross section code.
    :param <int> years: number of years over which data are aggregated. Supersedes the
        "freq" parameter and does not allow lags, Default is None, i.e. no multi-year
        aggregation.
    :param <str> freq: letter denoting frequency at which the series are to be sampled.
        This must be one of 'D', 'W', 'M', 'Q', 'A'. Default is 'M'. Will always be the
        last business day of the respective frequency.
    :param <int> lag: lag (delay of arrival) of explanatory category(s) in periods
        as set by freq. Default is 0.
    :param <int> fwin: forward moving average window of first category. Default is 1,
        i.e no average.
        Note: This parameter is used mainly for target returns as dependent variable.
    :param <List[str]> xcat_aggs: exactly two aggregation methods. Default is 'mean' for
        both. The same aggregation method, the first method in the parameter, will be
        used for all explanatory variables.

    :return <pd.DataFrame>: custom DataFrame with category columns. All rows that contain
        NaNs will be excluded.

    N.B.:
    The number of explanatory categories that can be included is not restricted and will
    be appended column-wise to the returned DataFrame. The order of the DataFrame's
    columns will reflect the order of the categories list.
    """

    freq = _map_to_business_day_frequency(freq)

    assert isinstance(xcats, list), f"<list> expected and not {type(xcats)}."
    assert all([isinstance(c, str) for c in xcats]), "List of categories expected."
    xcat_error = (
        "The minimum requirement is that a single dependent and explanatory "
        "variable are included."
    )
    assert len(xcats) >= 2, xcat_error

    aggs_error = "List of strings, outlining the aggregation methods, expected."
    assert isinstance(xcat_aggs, list), aggs_error
    assert all([isinstance(a, str) for a in xcat_aggs]), aggs_error
    aggs_len = (
        "Only two aggregation methods required. The first will be used for all "
        "explanatory category(s)."
    )
    assert len(xcat_aggs) == 2, aggs_len

    assert not (years is not None) & (
        lag != 0
    ), "Lags cannot be applied to year groups."
    if years is not None:
        assert isinstance(start, str), "Year aggregation requires a start date."

        no_xcats = (
            "If the data is aggregated over a multi-year timeframe, only two "
            "categories are permitted."
        )
        assert len(xcats) == 2, no_xcats

    df, xcats, cids = reduce_df(df, xcats, cids, start, end, blacklist, out_all=True)

    metric = ["value", "grading", "mop_lag", "eop_lag"]
    val_error = (
        "The column of interest must be one of the defined JPMaQS metrics, "
        f"{metric}, but received {val}."
    )
    assert val in metric, val_error
    avbl_cols = list(df.columns)
    assert val in avbl_cols, (
        f"The passed column name, {val}, must be present in the "
        f"received DataFrame. DataFrame contains {avbl_cols}."
    )

    # Reduce the columns in the DataFrame to the necessary columns:
    # ['cid', 'xcat', 'real_date'] + [val] (name of column that contains the
    # values of interest: "value", "grading", "mop_lag", "eop_lag").
    col_names = ["cid", "xcat", "real_date", val]

    df_output = []
    if years is None:
        df_w = df.pivot(index=("cid", "real_date"), columns="xcat", values=val)

        dep = xcats[-1]
        # The possibility of multiple explanatory variables.
        xpls = xcats[:-1]

        df_w = df_w.groupby(
            [
                pd.Grouper(level="cid"),
                pd.Grouper(level="real_date", freq=freq),
            ]
        )

        dfw_xpls = categories_df_expln_df(
            df_w=df_w,
            xpls=xpls,
            agg_meth=xcat_aggs[0],
            sum_condition=(xcat_aggs[0] == "sum"),
            lag=lag,
        )

        # Handles for falsified zeros. Following the frequency conversion, if the
        # aggregation method is set to "sum", time periods that exclusively contain NaN
        # values will incorrectly be summed to the value zero which is misleading for
        # analysis.
        if not (xcat_aggs[-1] == "sum"):
            dep_col = df_w[dep].agg(xcat_aggs[1]).astype(dtype=np.float32)
        else:
            dep_col = df_w[dep].sum(min_count=1)

        if fwin > 1:
            s = 1 - fwin
            dep_col = dep_col.rolling(window=fwin).mean().shift(s)

        dfw_xpls[dep] = dep_col
        # Order such that the return category is the right-most column - will reflect the
        # order of the categories list.
        dfc = dfw_xpls[xpls + [dep]]

    else:
        s_year = pd.to_datetime(start).year
        start_year = s_year
        e_year = df["real_date"].max().year + 1

        grouping = int((e_year - s_year) / years)
        remainder = (e_year - s_year) % years

        year_groups = {}

        for group in range(grouping):
            value = [i for i in range(s_year, s_year + years)]
            key = f"{s_year} - {s_year + (years - 1)}"
            year_groups[key] = value

            s_year += years

        v = [i for i in range(s_year, s_year + (remainder + 1))]
        year_groups[f"{s_year} - now"] = v
        list_y_groups = list(year_groups.keys())

        translate_ = lambda year: list_y_groups[int((year % start_year) / years)]
        df["real_date"] = pd.to_datetime(df["real_date"], errors="coerce")
        df["custom_date"] = df["real_date"].dt.year.apply(translate_)

        dfx_list = [df[df["xcat"] == xcats[0]], df[df["xcat"] == xcats[1]]]
        df_agg = list(map(categories_df_aggregation_helper, dfx_list, xcat_aggs))
        df_output.extend([d[col_names] for d in df_agg])

        dfc = pd.concat(df_output)
        dfc = dfc.pivot(index=("cid", "real_date"), columns="xcat", values=val)

    # Adjusted to account for multiple signals requested. If the DataFrame is
    # two-dimensional, signal & a return, NaN values will be handled inside other
    # functionality, as categories_df() is simply a support function. If the parameter
    # how is set to "any", a potential unnecessary loss of data on certain categories
    # could arise.
    return dfc.dropna(axis=0, how="all")


def _map_to_business_day_frequency(freq: str, valid_freqs: List[str] = None) -> str:
    """
    Maps a frequency string to a business frequency string.

    :param <str> freq: The frequency string to be mapped.
    :param <List[str]> valid_freqs: The valid frequency strings. If None, defaults to
        ["D", "W". "M", "Q", "A"].
    """
    freq = freq.upper()

    if valid_freqs is None:
        valid_freqs = list(FREQUENCY_MAP.keys())

    if freq not in valid_freqs:
        raise ValueError(
            f"Frequency must be one of {valid_freqs}, but received {freq}."
        )
    return FREQUENCY_MAP[freq]


def years_btwn_dates(start_date: pd.Timestamp, end_date: pd.Timestamp) -> int:
    """Returns the number of years between two dates."""
    return end_date.year - start_date.year


def quarters_btwn_dates(start_date: pd.Timestamp, end_date: pd.Timestamp) -> int:
    """Returns the number of quarters between two dates."""
    return (end_date.year - start_date.year) * 4 + (
        end_date.quarter - start_date.quarter
    )


def months_btwn_dates(start_date: pd.Timestamp, end_date: pd.Timestamp) -> int:
    """Returns the number of months between two dates."""
    return (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)


def weeks_btwn_dates(start_date: pd.Timestamp, end_date: pd.Timestamp) -> int:
    """Returns the number of business weeks between two dates."""
    next_monday = start_date + pd.offsets.Week(weekday=0)
    dif = (end_date - next_monday).days // 7 + 1
    return dif


def get_eops(
    dates: Optional[Union[pd.DatetimeIndex, pd.Series, Iterable[pd.Timestamp]]] = None,
    start_date: Optional[Union[str, pd.Timestamp]] = None,
    end_date: Optional[Union[str, pd.Timestamp]] = None,
    freq: str = "M",
) -> pd.Series:
    """
    Returns a series of end-of-period dates for a given frequency.
    Dates can be passed as a series, index, a generic iterable or as a start and end date.

    :param <str> freq: The frequency string. Must be one of "D", "W", "M", "Q", "A".
    :param <pd.DatetimeIndex | pd.Series | Iterable[pd.Timestamp]> dates: The dates to
        be used to generate the end-of-period dates. Can be passed as a series, index, a
        generic iterable or as a start and end date.
    :param <str | pd.Timestamp> start_date: The start date. Must be passed if dates is
        not passed.
    """
    if (not isinstance(freq, str)) or (freq.upper() not in FREQUENCY_MAP.keys()):
        raise ValueError(
            f"Frequency must be one of {list(FREQUENCY_MAP.keys())}, but received {freq}."
        )
    freq: str = freq.upper()

    if bool(start_date) != bool(end_date):
        raise ValueError(
            "Both `start_date` and `end_date` must be passed when using "
            "dates as a start and end date."
        )

    if bool(start_date) and bool(dates):
        raise ValueError(
            "Only one of `dates` or `start_date` and `end_date` must be passed."
        )

    dts: pd.DataFrame = (
        pd.DataFrame(dates, columns=["real_date"]).apply(pd.to_datetime, axis=1)
        if dates is not None
        else pd.Series(pd.bdate_range(start_date, end_date))
    )
    if dates is not None:
        dts = dts[
            dts["real_date"].isin(
                pd.bdate_range(start=dts["real_date"].min(), end=dts["real_date"].max())
            )
        ]

    min_date: pd.Timestamp = dts["real_date"].min()

    if freq == "M":
        func = months_btwn_dates
    elif freq == "W":
        func = weeks_btwn_dates
    elif freq == "Q":
        func = quarters_btwn_dates
    elif freq == "A":
        func = years_btwn_dates
    elif freq == "D":
        func = lambda x, y: len(pd.bdate_range(x, y)) - 1
    else:
        raise ValueError("Frequency parameter must be one of D, M, W, or Q")

    dts["period"] = dts["real_date"].apply(func, args=(min_date,))

    t_indices: pd.Series = dts["period"].shift(-1) != dts["period"]

    t_dates: pd.Series = dts["real_date"].loc[t_indices].reset_index(drop=True)

    return t_dates
