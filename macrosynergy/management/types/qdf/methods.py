"""
Module hosting custom types and meta-classes for use with Quantamental DataFrames.
"""

from typing import List, Optional, Any, Iterable, Mapping, Union, Dict, Set, Tuple
import pandas as pd
import numpy as np
import warnings
import itertools
import functools

from macrosynergy.management.constants import JPMAQS_METRICS

from .base import QuantamentalDataFrameBase


def get_col_sort_order(df: QuantamentalDataFrameBase) -> List[str]:
    """
    Sort the columns of a QuantamentalDataFrame (in-place) in a consistent order.

    Parameters
    ----------
    df : QuantamentalDataFrame
        DataFrame to return the sorted columns of.

    Returns
    -------
    List[str]
        List of sorted column names.
    """
    if not isinstance(df, QuantamentalDataFrameBase):
        raise TypeError("`df` must be a QuantamentalDataFrame.")

    metric_cols: Set = set(df.columns) - set(QuantamentalDataFrameBase.IndexCols)
    non_jpmaqs_metrics: List[str] = sorted(metric_cols - set(JPMAQS_METRICS))
    jpmaqs_metrics: List[str] = [m for m in JPMAQS_METRICS if m in metric_cols]

    return QuantamentalDataFrameBase.IndexCols + jpmaqs_metrics + non_jpmaqs_metrics


def change_column_format(
    df: QuantamentalDataFrameBase,
    cols: List[str],
    dtype: Any,
) -> QuantamentalDataFrameBase:
    """
    Change the format of columns in a DataFrame.

    Parameters
    ----------
    df : QuantamentalDataFrame
        DataFrame to change the format of.
    cols : List[str]
        List of column names to change the format of.
    dtype : Any
        Data type to change the columns to.

    Returns
    -------
    QuantamentalDataFrame
        DataFrame with the columns changed to the specified format

    Raises
    ------
    TypeError
        If `df` is not a QuantamentalDataFrame.

    TypeError
        If `cols` is not a list of strings.

    ValueError
        If a column in `cols` is not found in the DataFrame.
    """
    if not isinstance(df, QuantamentalDataFrameBase):
        raise TypeError("`df` must be a pandas DataFrame.")

    if not isinstance(cols, list) or not all([isinstance(col, str) for col in cols]):
        raise TypeError("`cols` must be a list of strings.")

    for col in cols:
        try:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame.")
            if not df[col].dtype == dtype:
                df[col] = df[col].astype(dtype)
        except Exception as exc:
            warnings.warn(f"Could not convert column {col} to {dtype}. Error: {exc}")

    return df


def qdf_to_categorical(
    df: QuantamentalDataFrameBase,
) -> QuantamentalDataFrameBase:
    """
    Convert the index columns ("cid", "xcat") of a DataFrame to categorical format.

    Parameters
    ----------
    df : QuantamentalDataFrame
        DataFrame to convert the index columns of.

    Raises
    ------
    TypeError
        If `df` is not a QuantamentalDataFrame.

    Returns
    -------
    QuantamentalDataFrame
        DataFrame with the index columns converted to categorical format.
    """
    if not isinstance(df, QuantamentalDataFrameBase):
        raise TypeError("`df` must be a QuantamentalDataFrame.")

    df = change_column_format(df, QuantamentalDataFrameBase._StrIndexCols, "category")
    return df


def qdf_to_string_index(
    df: QuantamentalDataFrameBase,
) -> QuantamentalDataFrameBase:
    """
    Convert the index columns ("cid", "xcat") of a DataFrame to string format.

    Parameters
    ----------
    df : QuantamentalDataFrame
        DataFrame to convert the index columns of.

    Raises
    ------
    TypeError
        If `df` is not a QuantamentalDataFrame.

    Returns
    -------
    QuantamentalDataFrame
        DataFrame with the index columns converted to string format.
    """
    if not isinstance(df, QuantamentalDataFrameBase):
        raise TypeError("`df` must be a QuantamentalDataFrame.")

    df = change_column_format(df, QuantamentalDataFrameBase._StrIndexCols, "object")
    return df


def check_is_categorical(df: QuantamentalDataFrameBase) -> bool:
    """
    Check if the index columns of a DataFrame are categorical.

    Parameters
    ----------
    df : QuantamentalDataFrame
        DataFrame to check the index columns of.

    Returns
    -------
    bool
        True if the required index columns ("cid", "xcat") are categorical, False
        otherwise.
    """
    if not isinstance(df, QuantamentalDataFrameBase):
        raise TypeError("`df` must be a QuantamentalDataFrame.")

    return all(
        df[col].dtype.name == "category"
        for col in QuantamentalDataFrameBase._StrIndexCols
    )


def _get_tickers_series(
    df: QuantamentalDataFrameBase,
    cid_column: str = "cid",
    xcat_column: str = "xcat",
) -> pd.Categorical:
    """
    Get the list of tickers from the DataFrame.

    Parameters
    ----------
    df : QuantamentalDataFrame
        DataFrame to extract the tickers from.

    cid_column : str, optional
        Column name of the `cid` in the DataFrame. Default is "cid".

    xcat_column : str, optional
        Column name of the `xcat` in the DataFrame. Default is "xcat".
    """
    # check if the columns are in the dataframe and are categorical
    if cid_column not in df.columns:
        raise ValueError(f"Column '{cid_column}' not found in DataFrame.")
    if xcat_column not in df.columns:
        raise ValueError(f"Column '{xcat_column}' not found in DataFrame.")

    if not check_is_categorical(df):
        return df[cid_column] + "_" + df[xcat_column]

    cid_labels = df["cid"].cat.categories[df["cid"].cat.codes]
    xcat_labels = df["xcat"].cat.categories[df["xcat"].cat.codes]

    ticker_labels = [f"{cid}_{xcat}" for cid, xcat in zip(cid_labels, xcat_labels)]
    categories = pd.unique(pd.Series(ticker_labels))

    ticker_series = pd.Categorical(
        ticker_labels,
        categories=categories,
        ordered=True,
    )

    return ticker_series


def apply_blacklist(
    df: QuantamentalDataFrameBase,
    blacklist: Mapping[str, Iterable[Union[str, pd.Timestamp]]],
) -> QuantamentalDataFrameBase:
    """
    Apply a blacklist to a list of `cids` and `xcats`. The blacklisted data ranges are
    removed from the DataFrame. This is useful for removing data that is known to be
    incorrect or unreliable.

    Parameters
    ----------
    df : QuantamentalDataFrame
        DataFrame to apply the blacklist to.

    blacklist : dict
        Dictionary with keys as `cids` and values as a list of start and end dates
        to blacklist. Example:

        .. code-block:: python

            {"cid": ["2020-01-01", "2020-12-31"]}

        This can be extended to cover multiple periods for the same `cid` by appending
        an additional label to the end of the `cid` key. Example:

        .. code-block:: python

            {
                "usd_1": ["2020-01-01", "2020-12-31"],
                "usd_2": ["2020-01-01", "2020-12-31"],
                "eur": ["2020-01-01", "2020-12-31"],
            }

    Returns
    -------
    QuantamentalDataFrame
        DataFrame with the blacklist applied.
    """
    if not isinstance(df, QuantamentalDataFrameBase):
        raise TypeError("`df` must be a pandas DataFrame.")

    if not isinstance(blacklist, dict):
        raise TypeError("`blacklist` must be a dictionary.")

    if not all([isinstance(k, str) for k in blacklist.keys()]):
        raise TypeError("Keys of `blacklist` must be strings.")

    if not all([isinstance(v, Iterable) for v in blacklist.values()]):
        raise TypeError("Values of `blacklist` must be iterables.")

    if not all(
        [isinstance(vv, (str, pd.Timestamp)) for v in blacklist.values() for vv in v]
    ) or any([len(v) != 2 for v in blacklist.values()]):
        raise TypeError(
            "Values of `blacklist` must be lists of start & end dates (str or pd.Timestamp)."
        )

    for key, value in blacklist.items():
        df = df[
            ~(
                (df["cid"] == key[:3])
                & (df["real_date"] >= value[0])
                & (df["real_date"] <= value[1])
            )
        ]

    return df.reset_index(drop=True)


def _sync_df_categories(
    df: QuantamentalDataFrameBase,
) -> QuantamentalDataFrameBase:
    """
    Sync the categories of the DataFrame with the data.

    Parameters
    ----------
    df : QuantamentalDataFrame
        DataFrame to sync the categories of.

    Returns
    -------
    QuantamentalDataFrame
        DataFrame with the categories synced.
    """
    if not check_is_categorical(df):
        return df

    df["cid"] = df["cid"].cat.remove_unused_categories().astype("category")
    df["xcat"] = df["xcat"].cat.remove_unused_categories().astype("category")

    return df


def reduce_df(
    df: QuantamentalDataFrameBase,
    cids: Optional[List[str]] = None,
    xcats: Optional[List[str]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    blacklist: dict = None,
    out_all: bool = False,
    intersect: bool = False,
) -> Union[
    QuantamentalDataFrameBase, Tuple[QuantamentalDataFrameBase, List[str], List[str]]
]:
    """
    Filter DataFrame by `cids`, `xcats`, and `start` & `end` dates.

    Parameters
    ----------
    df : QuantamentalDataFrameBase
        The DataFrame to be filtered.
    cids : Optional[List[str]], optional
        List of `cid` values to filter by. If None, all `cid` values are included.
    xcats : Optional[List[str]], optional
        List of `xcat` values to filter by. If None, all `xcat` values are included.
    start : Optional[str], optional
        Start date for filtering. If None, no start date filtering is applied.
    end : Optional[str], optional
        End date for filtering. If None, no end date filtering is applied.
    blacklist : dict, optional
        Dictionary specifying blacklist criteria. If None, no blacklist filtering is
        applied.
    out_all : bool, optional
        If True, returns the filtered DataFrame along with the lists of `xcats` and
        `cids`; i.e. `(df, xcats, cids)`.
    intersect : bool, optional
        If True, only includes `cid` values that are present for all `xcat` values.

    Returns
    -------
    Union[QuantamentalDataFrameBase, Tuple[QuantamentalDataFrameBase, List[str], List[str]]]
        The filtered DataFrame. If `out_all` is True, also returns the lists of `xcats`
        and `cids`.
    """
    if xcats is not None:
        if not isinstance(xcats, list):
            xcats = [xcats]

    if start:
        df = df[df["real_date"] >= pd.to_datetime(start)]

    if end:
        df = df[df["real_date"] <= pd.to_datetime(end)]

    if blacklist is not None:
        df = apply_blacklist(df, blacklist)

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

    df = df[df["cid"].isin(cids)].reset_index(drop=True)

    df = _sync_df_categories(df)

    df = df.drop_duplicates().reset_index(drop=True)

    if out_all:
        return df, xcats, sorted(cids)
    else:
        return df


def reduce_df_by_ticker(
    df: QuantamentalDataFrameBase,
    tickers: List[str],
    start: Optional[str] = None,
    end: Optional[str] = None,
    blacklist: dict = None,
) -> QuantamentalDataFrameBase:
    """
    Filters the given QuantamentalDataFrameBase based on tickers, date range, and
    blacklist.

    Parameters
    ----------
    df : QuantamentalDataFrameBase
        The DataFrame to be filtered.
    tickers : List[str]
        List of tickers to filter by.
    start : Optional[str], optional
        Start date for filtering. If None, no start date filtering is applied.
    end : Optional[str], optional
        End date for filtering. If None, no end date filtering is applied.
    blacklist : dict, optional
        Dictionary specifying blacklist criteria. If None, no blacklist filtering is
        applied.

    Raises
    ------
    TypeError
        If `df` is not a QuantamentalDataFrame.

    Returns
    -------
    QuantamentalDataFrameBase
        The filtered DataFrame.
    """
    if not isinstance(df, QuantamentalDataFrameBase):
        raise TypeError("`df` must be a QuantamentalDataFrame.")

    if not isinstance(tickers, list):
        if tickers is not None:
            raise TypeError("`tickers` must be a list of strings.")

    if start is not None:
        df = df.loc[df["real_date"] >= pd.to_datetime(start)]
    if end is not None:
        df = df.loc[df["real_date"] <= pd.to_datetime(end)]

    if blacklist is not None:
        df = apply_blacklist(df, blacklist)

    ticker_series = _get_tickers_series(df)
    if tickers is None:
        tickers = sorted(ticker_series.unique())

    df = df[ticker_series.isin(tickers)].reset_index(drop=True)

    df = _sync_df_categories(df)

    return df.drop_duplicates().reset_index(drop=True)


def update_df(
    df: QuantamentalDataFrameBase,
    df_add: QuantamentalDataFrameBase,
    xcat_replace: bool = False,
) -> QuantamentalDataFrameBase:
    """
    Append a standard DataFrame to a standard base DataFrame with ticker replacement on
    the intersection.

    Parameters
    ----------
    df : QuantamentalDataFrame
        Base DataFrame to append to.
    df_add : QuantamentalDataFrame
        DataFrame to append.
    xcat_replace : bool, optional
        If True, replace the xcats in the base DataFrame with the xcats in the DataFrame
        to append. Default is False.
    """

    if not isinstance(df, QuantamentalDataFrameBase):
        raise TypeError("`df` must be a QuantamentalDataFrame.")
    if not isinstance(df_add, QuantamentalDataFrameBase):
        raise TypeError("`df_add` must be a QuantamentalDataFrame.")
    if not isinstance(xcat_replace, bool):
        raise TypeError("`xcat_replace` must be a boolean.")

    if xcat_replace:
        df = update_categories(df=df, df_add=df_add)
    else:
        df = update_tickers(df=df, df_add=df_add)
    _sortorder = QuantamentalDataFrameBase.IndexColsSortOrder
    return df.sort_values(_sortorder).reset_index(drop=True)


def update_tickers(
    df: pd.DataFrame,
    df_add: pd.DataFrame,
) -> QuantamentalDataFrameBase:
    """
    Method used to update aggregate DataFrame on the ticker level.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to update.
    df_add : pd.DataFrame
        DataFrame to add to the base DataFrame.

    Returns
    -------
    QuantamentalDataFrame
        Updated DataFrame.
    """
    if df_add.empty:
        return df
    elif df.empty:
        return df_add

    if all(
        _df[icol].dtype.name == "category"
        for _df in [df, df_add]
        for icol in ["cid", "xcat"]
    ):
        union_cids = pd.api.types.union_categoricals(
            [df["cid"].unique(), df_add["cid"].unique()]
        )
        union_xcats = pd.api.types.union_categoricals(
            [df["xcat"].unique(), df_add["xcat"].unique()]
        )
        df["cid"] = pd.Categorical(df["cid"], categories=union_cids.categories)
        df["xcat"] = pd.Categorical(df["xcat"], categories=union_xcats.categories)

        df_add["cid"] = pd.Categorical(df_add["cid"], categories=union_cids.categories)
        df_add["xcat"] = pd.Categorical(
            df_add["xcat"], categories=union_xcats.categories
        )

    df = pd.concat([df, df_add], axis=0, ignore_index=True)

    df = df.drop_duplicates(
        subset=QuantamentalDataFrameBase.IndexCols,
        keep="last",
    ).reset_index(drop=True)
    return df


def update_categories(
    df: QuantamentalDataFrameBase,
    df_add: QuantamentalDataFrameBase,
) -> QuantamentalDataFrameBase:
    """
    Method used to update the DataFrame on the category level.
    """
    if not isinstance(df, QuantamentalDataFrameBase):
        raise TypeError("`df` must be a QuantamentalDataFrame.")
    if not isinstance(df_add, QuantamentalDataFrameBase):
        raise TypeError("`df_add` must be a QuantamentalDataFrame.")

    incumbent_categories = list(df["xcat"].unique())
    new_categories = list(df_add["xcat"].unique())

    append_condition = set(incumbent_categories) | set(new_categories)
    intersect = list(set(incumbent_categories).intersection(set(new_categories)))

    if all(
        _df[icol].dtype.name == "category"
        for _df in [df, df_add]
        for icol in ["cid", "xcat"]
    ):
        union_cids = pd.api.types.union_categoricals(
            [df["cid"].unique(), df_add["cid"].unique()]
        )
        union_xcats = pd.api.types.union_categoricals(
            [df["xcat"].unique(), df_add["xcat"].unique()]
        )
        df["cid"] = pd.Categorical(df["cid"], categories=union_cids.categories)
        df["xcat"] = pd.Categorical(df["xcat"], categories=union_xcats.categories)

        df_add["cid"] = pd.Categorical(df_add["cid"], categories=union_cids.categories)
        df_add["xcat"] = pd.Categorical(
            df_add["xcat"], categories=union_xcats.categories
        )

    if len(append_condition) == len(incumbent_categories + new_categories):
        df = pd.concat([df, df_add], axis=0, ignore_index=True)

    else:
        df = df[~df["xcat"].isin(intersect)]
        df = pd.concat([df, df_add], axis=0, ignore_index=True)
    _sortorder = QuantamentalDataFrameBase.IndexColsSortOrder
    return df.sort_values(_sortorder).reset_index(drop=True)


def qdf_to_wide_df(
    df: QuantamentalDataFrameBase,
    value_column: str = "value",
) -> pd.DataFrame:
    """
    Pivot the DataFrame to a wide format with memory efficiency.
    """
    # Ensure inputs are of the correct type and exist in the DataFrame
    if not isinstance(df, QuantamentalDataFrameBase):
        raise TypeError("`df` must be a QuantamentalDataFrame.")
    if not isinstance(value_column, str):
        raise TypeError("`value_column` must be a string.")
    if value_column not in df.columns:
        raise ValueError(f"Column '{value_column}' not found in DataFrame.")

    df["ticker"] = _get_tickers_series(df)

    # Perform the pivot directly within the assignment to reduce memory footprint
    return df.pivot(
        index="real_date", columns="ticker", values=value_column
    ).rename_axis(None, axis=1)


def add_ticker_column(
    df: QuantamentalDataFrameBase,
) -> List[str]:
    """
    Get the list of tickers from the DataFrame.

    Parameters
    ----------
    df : QuantamentalDataFrame
        DataFrame to extract the tickers from.

    Raises
    ------
    TypeError
        If `df` is not a QuantamentalDataFrame.

    Returns
    -------
    List[str]
        List of tickers.
    """
    if not isinstance(df, QuantamentalDataFrameBase):
        raise TypeError("`df` must be a pandas DataFrame.")

    df["ticker"] = _get_tickers_series(df)

    return df


def _add_categorical_column(
    df: pd.DataFrame,
    column_name: str,
    fill_value: str,
) -> pd.DataFrame:
    """
    Add a categorical index column to a DataFrame. Typically `cid` or `xcat`.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to add the index column to.
    column_name : str
        Name of the index column to add.
    """
    df[column_name] = pd.Categorical.from_codes([0] * len(df), categories=[fill_value])
    return df


def rename_xcats(
    df: QuantamentalDataFrameBase,
    xcat_map: Optional[Dict[str, str]] = None,
    select_xcats: Optional[List[str]] = None,
    postfix: Optional[str] = None,
    prefix: Optional[str] = None,
    name_all: Optional[str] = None,
    fmt_string: Optional[str] = None,
) -> QuantamentalDataFrameBase:
    """
    Rename the xcats in a DataFrame based on a mapping or a format string. Only one of
    `xcat_map` or `select_xcats` must be provided. If `name_all` is provided, all xcats
    will be renamed to this value.

    NOTE: This function maintains the datatype of the xcat column as a categorical.

    Parameters
    ----------
    df : QuantamentalDataFrame
        DataFrame to rename the xcats in.
    xcat_map : dict, optional
        Dictionary mapping the old xcats to new xcats. Default is None.
    select_xcats : List[str], optional
        List of xcats to rename. Default is None.
    postfix : str, optional
        Postfix to add to the xcats. Default is None.
    prefix : str, optional
        Prefix to add to the xcats. Default is None.
    name_all : str, optional
        Name to rename all xcats to. Default is None.
    fmt_string : str, optional
        Format string to rename xcats. Default is None.

    Raises
    ------
    TypeError
        If `df` is not a QuantamentalDataFrame.
    ValueError
        If both `xcat_map` and `select_xcats` are provided.
    TypeError
        If `xcat_map` is not a dictionary with string keys and values.
    ValueError
        If `postfix`, `prefix`, `name_all`, or `fmt_string` are not provided.
    ValueError
        If `fmt_string` does not contain exactly one pair of curly braces.

    Returns
    -------
    QuantamentalDataFrame
        DataFrame with the xcats renamed.
    """
    if not isinstance(df, QuantamentalDataFrameBase):
        raise TypeError("`df` must be a QuantamentalDataFrame.")

    if bool(xcat_map) and bool(select_xcats):
        raise ValueError("Only one of `xcat_map` or `select_xcats` must be provided.")

    # Validate `xcat_map`
    if xcat_map is not None:
        if not (
            isinstance(xcat_map, dict)
            and all(
                isinstance(k, str) and isinstance(v, str) for k, v in xcat_map.items()
            )
        ):
            raise TypeError(
                "`xcat_map` must be a dictionary with string keys and values."
            )
        # Rename xcats based on `xcat_map`
        df["xcat"] = df["xcat"].cat.rename_categories(
            {old_cat: xcat_map.get(old_cat, old_cat) for old_cat in df["xcat"].unique()}
        )
        return df

    if select_xcats is None:
        select_xcats = df["xcat"].unique()

    # Ensure exactly one of postfix, prefix, name_all, or fmt_string is provided
    if not (bool(postfix) ^ bool(prefix) ^ bool(name_all) ^ bool(fmt_string)):
        raise ValueError(
            "Exactly one of `postfix`, `prefix`, `name_all`, or `fmt_string` must be provided."
        )

    funcs = {
        "postfix": lambda x: f"{x}{postfix}",
        "prefix": lambda x: f"{prefix}{x}",
        "name_all": lambda x: name_all,
        "fmt_string": lambda x: fmt_string.format(x),
    }

    curr_func = None
    for var_, name_ in zip(
        [postfix, prefix, name_all, fmt_string],
        ["postfix", "prefix", "name_all", "fmt_string"],
    ):
        if var_ is not None:
            curr_func = name_

    if fmt_string is not None:
        if fmt_string.count("{}") != 1:
            raise ValueError(
                "The `fmt_string` must contain exactly one pair of curly braces."
            )

    if name_all is not None:
        xc_col = df["xcat"].astype(str)
        xc_col = xc_col.replace({cat: funcs[curr_func](cat) for cat in select_xcats})
        df["xcat"] = pd.Categorical(xc_col, categories=list(set(xc_col)))

    else:
        df["xcat"] = df["xcat"].cat.rename_categories(
            {cat: funcs[curr_func](cat) for cat in select_xcats}
        )

    return df


def create_empty_categorical_qdf(
    cid: Optional[str] = None,
    xcat: Optional[str] = None,
    ticker: Optional[str] = None,
    metrics: List[str] = ["value"],
    date_range: Optional[pd.DatetimeIndex] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    categorical: bool = True,
) -> QuantamentalDataFrameBase:
    """
    Create an empty QuantamentalDataFrame with categorical index columns. This is useful
    for creating a DataFrame for a given ticker with the required metrics.
    The ticker can be specified using `cid` and `xcat` or directly using `ticker`. The
    data range can be specified using `date_range` or `start` and `end`.

    Parameters
    ----------
    cid : str, optional
        `cid` value to use. Must be passed with `xcat`. Default is None.
    xcat : str, optional
        `xcat` value to use. Must be passed with `cid`. Default is None.
    ticker : str, optional
        Ticker to use. Must not be passed with `cid` and `xcat`. Default is None.
    metrics : List[str], optional
        List of metrics to create columns for. Default is ["value"].
    date_range : pd.DatetimeIndex, optional
        Date range to create the DataFrame for. Must not be passed with `start` and `end`.
        Default is None.
    start : str, optional
        Start date for the DataFrame. Default is None.
    end : str, optional
        End date for the DataFrame. Default is None.

    Raises
    ------
    TypeError
        If `metrics` is not a list of strings.
    ValueError
        If `date_range` is None and `start` and `end` are not provided.
    ValueError
        If `cid` and `xcat` are not provided together.
    ValueError
        If `cid` and `xcat` are provided together.
    ValueError
        If `ticker` is provided with `cid` and `xcat`.

    Returns
    -------
    QuantamentalDataFrame
        Empty DataFrame with the required index columns and metrics.
    """
    if not all(isinstance(m, str) for m in metrics):
        raise TypeError("`metrics` must be a list of strings.")

    if (date_range is None) and (start is None or end is None):
        raise ValueError(
            "Either `date_range` or `start_date` & `end_date` must be specified."
        )

    if date_range is None:
        date_range = pd.bdate_range(start=start, end=end)

    if bool(cid) ^ bool(xcat):
        raise ValueError("`cid` and `xcat` must be specified together.")

    if not (bool(cid) ^ bool(ticker)):
        raise ValueError("Either specify `cid` & `xcat` or `ticker` but not both.")

    if ticker is not None:
        cid, xcat = ticker.split("_", 1)

    qdf = pd.DataFrame(columns=["real_date"], data=date_range)
    qdf = _add_categorical_column(qdf, "cid", cid)
    qdf = _add_categorical_column(qdf, "xcat", xcat)

    for metric in metrics:
        qdf[metric] = np.nan

    if not categorical:
        qdf = qdf_to_string_index(qdf)

    return qdf


def add_nan_series(
    df: QuantamentalDataFrameBase,
    ticker: Optional[str] = None,
    cid: Optional[str] = None,
    xcat: Optional[str] = None,
    start: Optional[Union[str, pd.Timestamp]] = None,
    end: Optional[Union[str, pd.Timestamp]] = None,
) -> QuantamentalDataFrameBase:
    """
    Add a NaN series to the DataFrame for a given ticker.

    Parameters
    ----------
    df : QuantamentalDataFrame
        DataFrame to add the NaN series to.
    ticker : str, optional
        Ticker to add the NaN series for. Must not be passed with `cid` and `xcat`.
        Default is None.
    cid : str, optional
        `cid` value to use. Must be passed with `xcat`. Default is None.
    xcat : str, optional
        `xcat` value to use. Must be passed with `cid`. Default is None.
    start : str or pd.Timestamp, optional
        Start date for the NaN series. Default is None.
    end : str or pd.Timestamp, optional
        End date for the NaN series. Default is None.

    Raises
    ------
    TypeError
        If `df` is not a QuantamentalDataFrame.
    ValueError
        If `ticker` is provided with `cid` and `xcat`.

    Returns
    -------
    QuantamentalDataFrame
        DataFrame with the NaN series added.
    """

    if not isinstance(df, QuantamentalDataFrameBase):
        raise TypeError("`df` must be a QuantamentalDataFrame.")

    metrics = df.columns.difference(QuantamentalDataFrameBase.IndexCols)

    if start is None:
        start = df["real_date"].min()
    if end is None:
        end = df["real_date"].max()

    nan_df = create_empty_categorical_qdf(
        cid=cid,
        xcat=xcat,
        ticker=ticker,
        start=start,
        end=end,
        metrics=metrics,
        categorical=check_is_categorical(df),
    )

    df = update_df(df=df, df_add=nan_df)
    return df


def drop_nan_series(
    df: QuantamentalDataFrameBase, column: str = "value", raise_warning: bool = False
) -> QuantamentalDataFrameBase:
    """
    Drops any series that are entirely NaNs. Raises a user warning if any series are
    dropped.

    Parameters
    ----------
    df : QuantamentalDataFrame
        DataFrame to drop the NaN series from.
    column : str, optional
        Column to check for NaNs. Default is "value".
    raise_warning : bool, optional
        If True, raises a warning if any series are dropped. Default is False.

    Raises
    ------
    TypeError
        If `df` is not a QuantamentalDataFrame.
    ValueError
        If `column` is not found in the DataFrame.

    Returns
    -------
    QuantamentalDataFrame
        DataFrame with the NaN series dropped.
    """

    if not isinstance(df, QuantamentalDataFrameBase):
        raise TypeError("Argument `df` must be a Quantamental DataFrame.")

    if column not in df.columns:
        raise ValueError(f"Column {column} not present in DataFrame.")

    if not df[column].isna().any():
        return df

    if not isinstance(raise_warning, bool):
        raise TypeError("Error: The raise_warning argument must be a boolean.")

    df_orig: pd.DataFrame = df.copy()
    for cd, xc in df_orig.groupby(["cid", "xcat"], observed=True).groups:
        sel_series: pd.Series = df_orig[
            (df_orig["cid"] == cd) & (df_orig["xcat"] == xc)
        ][column]
        if sel_series.isna().all():
            if raise_warning:
                warnings.warn(
                    message=f"The series {cd}_{xc} is populated "
                    "with NaNs only, and will be dropped.",
                    category=UserWarning,
                )
            df = df[~((df["cid"] == cd) & (df["xcat"] == xc))]

    return df.reset_index(drop=True)


def qdf_from_timeseries(
    timeseries: pd.Series,
    cid: Optional[str] = None,
    xcat: Optional[str] = None,
    ticker: Optional[str] = None,
    metric: str = "value",
) -> QuantamentalDataFrameBase:
    """
    Create a QuantamentalDataFrame from a time series.

    Parameters
    ----------
    timeseries : pd.Series
        Time series to create the QuantamentalDataFrame from.
    cid : str, optional
        `cid` value to use. Must be passed with `xcat`. Default is None.
    xcat : str, optional
        `xcat` value to use. Must be passed with `cid`. Default is None.
    ticker : str, optional
        Ticker to use. Must not be passed with `cid` and `xcat`. Default is None.
    metric : str, optional
        Metric name to use. Default is "value".

    Raises
    ------
    TypeError
        If `timeseries` is not a pandas Series.
    TypeError
        If `metric` is not a string.
    ValueError
        If `timeseries` does not have a datetime index.
    ValueError
        If only one of `cid` and `xcat` is provided.
    ValueError
        If `ticker` is provided with `cid` and `xcat`.

    Returns
    -------
    QuantamentalDataFrame
        DataFrame created from the time series.
    """
    if not isinstance(timeseries, pd.Series):
        raise TypeError("`timeseries` must be a pandas Series.")
    if not isinstance(metric, str):
        raise TypeError("`metric` must be a string.")

    if not isinstance(timeseries.index, pd.DatetimeIndex):
        raise ValueError("`timeseries` must have a datetime index.")

    if (cid is None) ^ (xcat is None):
        raise ValueError("Both `cid` and `xcat` must be provided.")

    if not ((cid is None) ^ (ticker is None)):
        raise ValueError("Either provide `cid` & `xcat` or `ticker`.")

    if ticker is not None:
        cid, xcat = ticker.split("_", 1)

    assert bool(cid) and bool(xcat)

    df = timeseries.reset_index().rename(columns={"index": "real_date", 0: metric})
    # assign as categorical string
    df = _add_categorical_column(df, "cid", cid)
    df = _add_categorical_column(df, "xcat", xcat)

    df = df[[*QuantamentalDataFrameBase.IndexCols, metric]]
    return QuantamentalDataFrameBase(df)


def _convert_to_single_metric_qdfs(
    qdf: QuantamentalDataFrameBase,
) -> QuantamentalDataFrameBase:
    """
    Internal function for concat_qdfs. Splits a QuantamentalDataFrame with multiple
    metrics to multiple QuantamentalDataFrames with a single metric.

    Parameters
    ----------
    qdf : QuantamentalDataFrame
        DataFrame to convert.

    Returns
    -------
    List[QuantamentalDataFrame]
        List of QuantamentalDataFrames with a single metric.
    """
    return [
        qdf[[*QuantamentalDataFrameBase.IndexCols, metric]]
        for metric in qdf.columns.difference(QuantamentalDataFrameBase.IndexCols)
    ]


def concat_qdfs(
    qdf_list: List[QuantamentalDataFrameBase],
) -> QuantamentalDataFrameBase:
    """
    Concatenate a list of QuantamentalDataFrames into a single QuantamentalDataFrame.
    Converts the index columns to categorical format, if not already categorical.

    Parameters
    ----------
    qdf_list : List[QuantamentalDataFrame]
        List of QuantamentalDataFrames to concatenate.

    Raises
    ------
    TypeError
        If `qdf_list` is not a list of QuantamentalDataFrames.

    Returns
    -------
    QuantamentalDataFrame
        DataFrame with the QuantamentalDataFrames concatenated.
    """
    if not isinstance(qdf_list, list):
        raise TypeError("`qdfs_list` must be a list of QuantamentalDataFrames.")

    if not all(isinstance(qdf, QuantamentalDataFrameBase) for qdf in qdf_list):
        raise TypeError("All elements in `qdfs_list` must be QuantamentalDataFrames.")

    if len(qdf_list) == 0:
        raise ValueError("`qdfs_list` is empty.")

    for iq, qdf in enumerate(qdf_list):
        qdf_list[iq] = qdf_to_categorical(qdf)

    comb_cids = pd.api.types.union_categoricals(
        [qdf["cid"].unique() for qdf in qdf_list]
    )
    comb_xcats = pd.api.types.union_categoricals(
        [qdf["xcat"].unique() for qdf in qdf_list]
    )

    for iq, qdf in enumerate(qdf_list):
        qdf_list[iq]["cid"] = pd.Categorical(
            qdf["cid"], categories=comb_cids.categories
        )
        qdf_list[iq]["xcat"] = pd.Categorical(
            qdf["xcat"], categories=comb_xcats.categories
        )

    qdf_list = list(itertools.chain(*map(_convert_to_single_metric_qdfs, qdf_list)))

    def _get_metric(df: QuantamentalDataFrameBase) -> str:
        return list(set(df.columns) - set(QuantamentalDataFrameBase.IndexCols))[0]

    def _group_by_metric(
        dfl: List[QuantamentalDataFrameBase], fm: List[str]
    ) -> List[List[QuantamentalDataFrameBase]]:
        r = [[] for _ in range(len(fm))]
        while dfl:
            metric = _get_metric(df=dfl[0])
            r[fm.index(metric)] += [dfl.pop(0)]
        return r

    found_metrics = list(set(map(_get_metric, qdf_list)))
    qdf_list = _group_by_metric(dfl=qdf_list, fm=found_metrics)

    df: pd.DataFrame = functools.reduce(
        lambda left, right: pd.merge(
            left,
            right,
            on=QuantamentalDataFrameBase.IndexCols,
            how="outer",
        ),
        map(
            lambda x: pd.concat(qdf_list.pop(0), axis=0, ignore_index=False),
            found_metrics,
        ),
    )

    return df.sort_values(by=QuantamentalDataFrameBase.IndexColsSortOrder).reset_index(
        drop=True
    )[get_col_sort_order(df)]
