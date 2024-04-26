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
import datetime
import requests
import requests.compat
from macrosynergy.management.utils.core import (
    get_cid,
    get_xcat,
    _map_to_business_day_frequency,
    is_valid_iso_date,
)


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

    return (
        df.assign(ticker=df["cid"] + "_" + df["xcat"])
        .pivot(index="real_date", columns="ticker", values=value_column)
        .rename_axis(None, axis=1)
    )


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
    Applies a slip, i.e. a negative lag, to the DataFrame
    for the given cross-sections and categories, on the given metrics.

    :param <pd.DataFrame> target_df: DataFrame to which the slip is applied.
    :param <int> slip: Slip to be applied.
    :param <List[str]> cids: List of cross-sections.
    :param <List[str]> xcats: List of target categories.
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
                f"Missing tickers: "
                f"{sorted(list(set(sel_tickers) - set(df['tickers'].unique())))}"
            )
        else:
            warnings.warn(
                "Tickers targetted for applying slip are not present in the DataFrame.\n"
                f"Missing tickers: "
                f"{sorted(list(set(sel_tickers) - set(df['tickers'].unique())))}"
            )

    slip: int = slip.__neg__()

    filtered_df = df[df["tickers"].isin(sel_tickers)]

    filtered_df.loc[:, metrics] = filtered_df.groupby("tickers")[metrics].shift(slip)

    df.loc[df["tickers"].isin(sel_tickers), metrics] = filtered_df[metrics]

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
        'cid', 'xcat', 'real_date' and at least one column with values of interest.
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

    error_message = (
        "The two Quantamental DataFrames must share at least "
        "four columns including than 'real_date', 'cid', and 'xcat'."
    )

    all_cols = df_cols.union(df_add_cols)
    if all_cols != df_cols and all_cols != df_add_cols:
        raise ValueError(error_message)

    if not xcat_replace:
        df = update_tickers(df, df_add)

    else:
        df = update_categories(df, df_add)

    # sort same as in `standardise_dataframe`
    return df.sort_values(by=["real_date", "cid", "xcat"]).reset_index(drop=True)


def update_tickers(df: pd.DataFrame, df_add: pd.DataFrame):
    """
    Method used to update aggregate DataFrame on a ticker level.

    :param <pd.DataFrame> df: aggregate DataFrame used to store all tickers.
    :param <pd.DataFrame> df_add: DataFrame with the latest values.

    """
    df = pd.concat([df, df_add], axis=0, ignore_index=True)

    df = df.drop_duplicates(
        subset=["real_date", "xcat", "cid"], keep="last"
    ).reset_index(drop=True)
    return df


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
    xcats: Union[str, List[str]] = None,
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
        'cid', 'xcat', 'real_date' and 'value'.
    :param <Union[str, List[str]]> xcats: extended categories to be filtered on. Default is
        all in the DataFrame.
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

    if xcats is not None:
        if not isinstance(xcats, list):
            xcats = [xcats]

    if start:
        df = df[df["real_date"] >= pd.to_datetime(start)]

    if end:
        df = df[df["real_date"] <= pd.to_datetime(end)]

    if blacklist is not None:
        for key, value in blacklist.items():
            df = df[
                ~(
                    (df["cid"] == key[:3])
                    & (df["real_date"] >= pd.to_datetime(value[0]))
                    & (df["real_date"] <= pd.to_datetime(value[1]))
                )
            ]

    if xcats is None:
        xcats = sorted(df["xcat"].unique())
    else:
        xcats_in_df = df["xcat"].unique()
        xcats = [xcat for xcat in xcats if xcat in xcats_in_df]

    df = df[df["xcat"].isin(xcats)]

    if intersect:
        cids_in_df = set.intersection(
            *(set(df[df["xcat"] == xcat]["cid"].unique()) for xcat in xcats)
        )
    else:
        cids_in_df = df["cid"].unique()

    if cids is None:
        cids = sorted(cids_in_df)
    else:
        cids = [cids] if isinstance(cids, str) else cids
        cids = [cid for cid in cids if cid in cids_in_df]

    df = df[df["cid"].isin(cids)]

    if out_all:
        return df.drop_duplicates(), xcats, sorted(cids)
    else:
        return df.drop_duplicates()


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
        'cid', 'xcat', 'real_date'.
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


def _categories_df_explanatory_df(
    dfw: pd.DataFrame,
    explanatory_xcats: List[str],
    agg_method: str,
    sum_condition: bool,
    lag: int,
):
    """
    Produces the explanatory column(s) for the custom DataFrame.

    :param <pd.DataFrame> dfw: group-by DataFrame which has been down-sampled. The
        respective aggregation method will be applied.
    :param <List[str]> explanatory_xcats: list of explanatory category(s).
    :param <str> agg_meth: aggregation method used for all explanatory variables.
    :param <dict> sum_condition: required boolean to negate erroneous zeros if the
        aggregate method used, for the explanatory variable, is sum.
    :param <int> lag: lag of explanatory category(s). Applied uniformly to each
        category.
    """

    dfw_explanatory = pd.DataFrame()
    for xcat in explanatory_xcats:
        if not sum_condition:
            explanatory_col = dfw[xcat].agg(agg_method).astype(dtype=np.float32)
        else:
            explanatory_col = dfw[xcat].sum(min_count=1)

        if lag > 0:
            explanatory_col = explanatory_col.groupby(level=0).shift(lag)

        dfw_explanatory[xcat] = explanatory_col

    return dfw_explanatory


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
        columns: 'cid', 'xcat', 'real_date' and at least one column with values of
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

    input_xcats = xcats
    input_cids = cids
    df, xcats, cids = reduce_df(df, xcats, cids, start, end, blacklist, out_all=True)

    if len(xcats) < 2:
        raise ValueError("The DataFrame must contain at least two categories. ")
    elif set(xcats) != set(input_xcats):
        missing_xcats = list(set(input_xcats) - set(xcats))
        warnings.warn(
            f"The following categories are missing from the DataFrame: {missing_xcats}"
        )

    if len(cids) < 1:
        raise ValueError(
            "The DataFrame must contain at least one valid cross section. "
        )
    elif input_cids and set(cids) != set(input_cids):
        missing_cids = list(set(input_cids) - set(cids))
        warnings.warn(
            f"The following cross sections are missing from the DataFrame: {missing_cids}"
        )

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
        dfw = df.pivot(index=("cid", "real_date"), columns="xcat", values=val)

        dep = xcats[-1]
        # The possibility of multiple explanatory variables.
        explanatory_xcats = xcats[:-1]

        dfw = dfw.groupby(
            [
                pd.Grouper(level="cid"),
                pd.Grouper(level="real_date", freq=freq),
            ]
        )

        dfw_explanatory = _categories_df_explanatory_df(
            dfw=dfw,
            explanatory_xcats=explanatory_xcats,
            agg_method=xcat_aggs[0],
            sum_condition=(xcat_aggs[0] == "sum"),
            lag=lag,
        )

        # Handles for falsified zeros. Following the frequency conversion, if the
        # aggregation method is set to "sum", time periods that exclusively contain NaN
        # values will incorrectly be summed to the value zero which is misleading for
        # analysis.
        if not (xcat_aggs[-1] == "sum"):
            dep_col = dfw[dep].agg(xcat_aggs[1]).astype(dtype=np.float32)
        else:
            dep_col = dfw[dep].sum(min_count=1)

        if fwin > 1:
            s = 1 - fwin
            dep_col = dep_col.rolling(window=fwin).mean().shift(s)

        dfw_explanatory[dep] = dep_col
        # Order such that the return category is the right-most column - will reflect the
        # order of the categories list.
        dfc = dfw_explanatory[explanatory_xcats + [dep]]

    else:
        start_year = pd.to_datetime(start).year
        end_year = df["real_date"].max().year + 1

        grouping = int((end_year - start_year) / years)
        remainder = (end_year - start_year) % years

        year_groups = {}

        group_start_year = start_year
        for group in range(grouping):
            value = [i for i in range(group_start_year, group_start_year + years)]
            key = f"{group_start_year} - {group_start_year + (years - 1)}"
            year_groups[key] = value

            group_start_year += years

        v = [i for i in range(group_start_year, group_start_year + (remainder + 1))]
        year_groups[f"{group_start_year} - now"] = v
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


def _get_edge_dates(
    dates: Optional[Union[pd.DatetimeIndex, pd.Series, Iterable[pd.Timestamp]]] = None,
    start_date: Optional[Union[str, pd.Timestamp]] = None,
    end_date: Optional[Union[str, pd.Timestamp]] = None,
    freq: str = "M",
    direction: str = "end",
) -> pd.Series:
    assert direction in ["start", "end"], "Direction must be either 'start' or 'end'."
    datettypes = [pd.Timestamp, str, np.datetime64, datetime.date]

    freq = _map_to_business_day_frequency(freq)

    if bool(start_date) != bool(end_date):
        raise ValueError(
            "Both `start_date` and `end_date` must be passed when using "
            "dates as a start and end date."
        )

    if dates is not None:
        if not isinstance(dates, (pd.DatetimeIndex, pd.Series, Iterable)):
            raise TypeError(
                "Dates must be a pandas DatetimeIndex, Series, or a generic iterable."
            )
        if isinstance(dates, pd.DataFrame):
            dates = dates.iloc[:, 0]
        if isinstance(dates, (pd.DatetimeIndex, pd.Series)):
            dates = dates.tolist()
        dates = list(dates)

        for ix, dt in enumerate(dates):
            try:
                dates[ix] = pd.to_datetime(dt)
            except Exception as e:
                raise TypeError(
                    f"Error converting date at index {ix} to a pandas Timestamp: {e}"
                )

    if bool(start_date) and bool(dates):
        raise ValueError(
            "Only one of `dates` or `start_date` and `end_date` must be passed."
        )

    if bool(start_date):
        assert bool(end_date)
        for date, dname in zip([start_date, end_date], ["start_date", "end_date"]):
            if not isinstance(date, (str, pd.Timestamp)):
                raise TypeError(f"{dname} must be a string or a pandas Timestamp.")
            if isinstance(date, str):
                if not is_valid_iso_date(date):
                    raise ValueError(
                        "Both `start_date` and `end_date` must be valid ISO dates when passed as "
                        "strings."
                    )

        if pd.Timestamp(start_date) > pd.Timestamp(end_date):
            start_date, end_date = end_date, start_date

    dts: pd.DataFrame = pd.DataFrame(
        (
            dates
            if (dates is not None)
            else pd.bdate_range(start=start_date, end=end_date)
        ),
        columns=["real_date"],
    ).apply(pd.to_datetime, axis=1)
    min_date: pd.Timestamp = dts["real_date"].min()

    if freq == _map_to_business_day_frequency("D"):
        max_date = dts["real_date"].max()
        dtx = pd.bdate_range(start=min_date, end=max_date)
        return dtx[dtx.isin(dts["real_date"])]

    if freq == _map_to_business_day_frequency("M"):
        func = months_btwn_dates
    elif freq == _map_to_business_day_frequency("W"):
        func = weeks_btwn_dates
    elif freq == _map_to_business_day_frequency("Q"):
        func = quarters_btwn_dates
    elif freq == _map_to_business_day_frequency("A"):
        func = years_btwn_dates
    else:
        raise ValueError("Frequency parameter must be one of D, M, W, Q, or A.")

    dts["period"] = dts["real_date"].apply(func, args=(min_date,))

    dx = -1 if direction == "end" else 1
    t_indices: pd.Series = dts["period"].shift(dx) != dts["period"]
    t_dates: pd.Series = dts["real_date"].loc[t_indices].reset_index(drop=True)

    return t_dates


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
    direction = "end"
    return _get_edge_dates(
        dates=dates,
        start_date=start_date,
        end_date=end_date,
        freq=freq,
        direction=direction,
    )


def get_sops(
    dates: Optional[Union[pd.DatetimeIndex, pd.Series, Iterable[pd.Timestamp]]] = None,
    start_date: Optional[Union[str, pd.Timestamp]] = None,
    end_date: Optional[Union[str, pd.Timestamp]] = None,
    freq: str = "M",
) -> pd.Series:
    """
    Returns a series of start-of-period dates for a given frequency.
    Dates can be passed as a series, index, a generic iterable or as a start and end date.

    :param <str> freq: The frequency string. Must be one of "D", "W", "M", "Q", "A".
    :param <pd.DatetimeIndex | pd.Series | Iterable[pd.Timestamp]> dates: The dates to
        be used to generate the start-of-period dates. Can be passed as a series, index, a
        generic iterable or as a start and end date.
    :param <str | pd.Timestamp> start_date: The start date. Must be passed if dates is
        not passed.
    """
    direction = "start"
    return _get_edge_dates(
        dates=dates,
        start_date=start_date,
        end_date=end_date,
        freq=freq,
        direction=direction,
    )


if __name__ == "__main__":
    from macrosynergy.management.simulate import make_qdf

    cids = ["AUD", "CAD", "GBP", "NZD"]
    xcats = ["XR1", "XR2", "CRY1", "CRY2"]
    df_cids = pd.DataFrame(
        index=cids, columns=["earliest", "latest", "mean_add", "sd_mult"]
    )
    df_cids.loc["AUD"] = ["2000-01-01", "2020-12-31", 0.1, 1]
    df_cids.loc["CAD"] = ["2001-01-01", "2020-11-30", 0, 1]
    df_cids.loc["GBP"] = ["2002-01-01", "2020-11-30", 0, 2]
    df_cids.loc["NZD"] = ["2002-01-01", "2020-09-30", -0.1, 2]

    df_xcats = pd.DataFrame(
        index=xcats,
        columns=["earliest", "latest", "mean_add", "sd_mult", "ar_coef", "back_coef"],
    )
    df_xcats.loc["XR1"] = ["2000-01-01", "2020-12-31", 0.1, 1, 0, 0.3]
    df_xcats.loc["XR2"] = ["2000-01-01", "2020-10-30", 1, 2, 0.95, 1]
    df_xcats.loc["CRY1"] = ["2001-01-01", "2020-10-30", 1, 2, 0.9, 1]
    df_xcats.loc["CRY2"] = ["2001-01-01", "2020-10-30", 1, 2, 0.8, 0.5]

    dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)

    dfw = categories_df(
        df=dfd,
        xcats=xcats[:2] + [ "test"],
        cids=cids,
        freq="M",
        # lag=1,
        xcat_aggs=["last", "sum"],
        # years=5,
        # start="2000-01-01",
    )
