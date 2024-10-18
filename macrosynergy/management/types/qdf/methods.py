"""
Module hosting custom types and meta-classes for use with Quantamental DataFrames.
"""

from typing import List, Optional, Any, Iterable, Mapping, Union, Dict, Set
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
    if not isinstance(df, QuantamentalDataFrameBase):
        raise TypeError("`df` must be a pandas DataFrame.")

    if not isinstance(cols, list) or not all([isinstance(col, str) for col in cols]):
        raise TypeError("`cols` must be a list of strings.")

    for col in cols:
        curr_type = df[col].dtype
        try:
            df[col] = df[col].astype(dtype)
        except:  # noqa
            warnings.warn(
                f"Could not convert column {col} to {dtype} from {curr_type}."
            )

    return df


def qdf_to_categorical(
    df: QuantamentalDataFrameBase,
) -> QuantamentalDataFrameBase:
    if not isinstance(df, QuantamentalDataFrameBase):
        raise TypeError("`df` must be a pandas DataFrame.")

    df = change_column_format(df, QuantamentalDataFrameBase._StrIndexCols, "category")
    return df


def check_is_categorical(df: QuantamentalDataFrameBase) -> bool:
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
    Apply a blacklist to a list of `cids` and `xcats`.
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
    ):
        raise TypeError(
            "Values of `blacklist` must be lists of date strings or pandas Timestamps."
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


def reduce_df(
    df: QuantamentalDataFrameBase,
    cids: Optional[List[str]] = None,
    xcats: Optional[List[str]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    blacklist: dict = None,
    out_all: bool = False,
    intersect: bool = False,
) -> QuantamentalDataFrameBase:
    # Filter DataFrame by xcats and cids and notify about missing xcats and cids.
    """
    Filter DataFrame by `cids`, `xcats`, and `start` & `end` dates.
    """
    if isinstance(cids, str):
        cids = [cids]
    if isinstance(xcats, str):
        xcats = [xcats]

    if start is not None:
        df = df.loc[df["real_date"] >= pd.to_datetime(start)]
    if end is not None:
        df = df.loc[df["real_date"] <= pd.to_datetime(end)]

    if blacklist is not None:
        df = apply_blacklist(df, blacklist)

    if cids is None:
        cids = sorted(df["cid"].unique())
    else:
        cids_in_df = df["cid"].unique()
        cids = sorted(c for c in cids if c in cids_in_df)

    if xcats is None:
        xcats = sorted(df["xcat"].unique())
    else:
        xcats_in_df = df["xcat"].unique()
        xcats = sorted(x for x in xcats if x in xcats_in_df)

    if intersect:
        cids_in_df = set.intersection(
            *(set(df[df["xcat"] == xcat]["cid"].unique()) for xcat in xcats)
        )
    else:
        cids_in_df = df["cid"].unique()
        cids = sorted(c for c in cids if c in cids_in_df)

    df = df[df["xcat"].isin(xcats)]
    df = df[df["cid"].isin(cids)]

    xcats_found = sorted(set(df["xcat"].unique()))
    cids_found = sorted(set(df["cid"].unique()))
    df = df.drop_duplicates().reset_index(drop=True)
    if out_all:
        return df, xcats_found, cids_found
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
    Filter DataFrame by `tickers` and `start` & `end` dates.
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

    df = df[ticker_series.isin(tickers)]

    return df.reset_index(drop=True)


def update_df(
    df: QuantamentalDataFrameBase,
    df_add: QuantamentalDataFrameBase,
    xcat_replace: bool = False,
) -> QuantamentalDataFrameBase:
    """
    Append a standard DataFrame to a standard base DataFrame with ticker replacement on
    the intersection.
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
):
    """
    Method used to update aggregate DataFrame on the ticker level.
    """
    if df_add.empty:
        return df
    elif df.empty:
        return df_add

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
    """
    if not isinstance(df, QuantamentalDataFrameBase):
        raise TypeError("`df` must be a pandas DataFrame.")

    df["ticker"] = _get_tickers_series(df)

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
    if not isinstance(df, QuantamentalDataFrameBase):
        raise TypeError("`df` must be a QuantamentalDataFrame.")

    if bool(xcat_map) and bool(select_xcats):
        raise ValueError("Only one of `xcat_map` or `select_xcats` must be provided.")

    # Validate `xcat_map`
    if xcat_map is not None:
        if not all(
            isinstance(k, str) and isinstance(v, str) for k, v in xcat_map.items()
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

    df["xcat"] = df["xcat"].cat.rename_categories(
        {cat: funcs[curr_func](cat) for cat in select_xcats}
    )

    return df


def _add_index_str_column(
    df: pd.DataFrame,
    column_name: str,
    fill_value: str,
) -> pd.DataFrame:
    """
    Add an index column to the DataFrame with a specified fill value.
    """
    df[column_name] = pd.Categorical.from_codes([0] * len(df), categories=[fill_value])
    return df


def add_nan_series(
    df: QuantamentalDataFrameBase,
    ticker: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> QuantamentalDataFrameBase:
    """
    Add a NaN series to the DataFrame for a given ticker.
    """

    if not isinstance(df, QuantamentalDataFrameBase):
        raise TypeError("`df` must be a QuantamentalDataFrame.")
    if not isinstance(ticker, str):
        raise TypeError("`ticker` must be a string.")
    if "_" not in ticker:
        raise ValueError("Ticker must be in the format 'cid_xcat'.")

    if start is not None:
        df = df.loc[df["real_date"] >= pd.to_datetime(start)]
    if end is not None:
        df = df.loc[df["real_date"] <= pd.to_datetime(end)]

    metrics = list(set(df.columns.tolist()) - set(QuantamentalDataFrameBase.IndexCols))

    cid, xcat = ticker.split("_", 1)

    # warn for overwriting existing entries
    if len(df[(df["cid"] == cid) & (df["xcat"] == xcat)]) > 0:
        warnings.warn(
            f"Entries for {ticker} already exist in the DataFrame in the given date range. "
            "The existing entries will be overwritten."
        )

    nan_df = pd.DataFrame(
        {
            "real_date": df["real_date"].unique(),
            **{metric: np.nan for metric in metrics},
        }
    )
    nan_df = _add_index_str_column(nan_df, "cid", cid)
    nan_df = _add_index_str_column(nan_df, "xcat", xcat)

    df = update_df(df=df, df_add=QuantamentalDataFrameBase(nan_df))
    return df


def drop_nan_series(
    df: QuantamentalDataFrameBase, column: str = "value", raise_warning: bool = False
) -> QuantamentalDataFrameBase:
    """
    Drops any series that are entirely NaNs.
    Raises a user warning if any series are dropped.
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


def qdf_from_timseries(
    timeseries: pd.Series,
    cid: Optional[str] = None,
    xcat: Optional[str] = None,
    ticker: Optional[str] = None,
    metric: str = "value",
) -> QuantamentalDataFrameBase:
    """
    Create a QuantamentalDataFrame from a time series.
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
    df = _add_index_str_column(df, "cid", cid)
    df = _add_index_str_column(df, "xcat", xcat)

    df = df[[*QuantamentalDataFrameBase.IndexCols, metric]]
    return QuantamentalDataFrameBase(df)


def create_empty_categorical_qdf(
    cid: Optional[str] = None,
    xcat: Optional[str] = None,
    ticker: Optional[str] = None,
    metrics: List[str] = ["value"],
    date_range: Optional[pd.DatetimeIndex] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    categorical: bool = True,
) -> QuantamentalDataFrameBase:

    if not all(isinstance(m, str) for m in metrics):
        raise TypeError("`metrics` must be a list of strings.")

    if (date_range is None) and (start_date is None or end_date is None):
        raise ValueError(
            "Either `date_range` or `start_date` & `end_date` must be specified."
        )

    if date_range is None:
        date_range = pd.date_range(start=start_date, end=end_date, freq="B")

    if bool(cid) ^ bool(xcat):
        raise ValueError("`cid` and `xcat` must be specified together.")

    if not (bool(cid) ^ bool(ticker)):
        raise ValueError("Either specify `cid` & `xcat` or `ticker` but not both.")

    if ticker is not None:
        cid, xcat = ticker.split("_", 1)

    qdf = pd.DataFrame(columns=["real_date"], data=date_range)
    qdf = _add_index_str_column(qdf, "cid", cid)
    qdf = _add_index_str_column(qdf, "xcat", xcat)

    for metric in metrics:
        qdf[metric] = np.nan

    return qdf


def concat_qdfs(
    qdf_list: List[QuantamentalDataFrameBase],
) -> QuantamentalDataFrameBase:

    if not isinstance(qdf_list, list):
        raise TypeError("`qdfs_list` must be a list of QuantamentalDataFrames.")

    if not all(isinstance(qdf, QuantamentalDataFrameBase) for qdf in qdf_list):
        raise TypeError("All elements in `qdfs_list` must be QuantamentalDataFrames.")

    if len(qdf_list) == 0:
        raise ValueError("`qdfs_list` is empty.")

    qdf_list = [qdf_to_categorical(qdf) for qdf in qdf_list]

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

    def _convert_to_single_metric_qdfs(
        qdf: QuantamentalDataFrameBase,
    ) -> QuantamentalDataFrameBase:
        return [
            qdf[[*QuantamentalDataFrameBase.IndexCols, metric]]
            for metric in qdf.columns.difference(QuantamentalDataFrameBase.IndexCols)
        ]

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
