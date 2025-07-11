"""
Utility functions for working with DataFrames.
"""

from macrosynergy.management.types import QuantamentalDataFrame
from macrosynergy.management.constants import FREQUENCY_MAP, FFILL_LIMITS, DAYS_PER_FREQ

import warnings
from typing import Iterable, List, Optional, Union, Dict

from numbers import Number

import numpy as np
import pandas as pd
import datetime
import macrosynergy.management.constants as ms_constants
from macrosynergy.management.utils.core import (
    get_cid,
    get_xcat,
    _map_to_business_day_frequency,
    is_valid_iso_date,
)
from macrosynergy.compat import RESAMPLE_NUMERIC_ONLY, PD_OLD_RESAMPLE
import functools

IDX_COLS_SORT_ORDER = ["cid", "xcat", "real_date"]


def is_categorical_qdf(df: pd.DataFrame) -> bool:
    """
    Check if a column in a DataFrame is categorical.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to be checked.
    column : str
        The column to be checked.

    Returns
    -------
    bool
        True if the column is categorical, False otherwise.
    """

    if not isinstance(df, QuantamentalDataFrame):
        raise TypeError("Argument `df` must be a QuantamentalDataFrame.")

    return all([df[col].dtype.name == "category" for col in ["cid", "xcat"]])


def standardise_dataframe(
    df: pd.DataFrame
) -> QuantamentalDataFrame:
    """
    Applies the standard JPMaQS Quantamental DataFrame format to a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to be standardized.

    Raises
    ------
    TypeError
        If the input is not a pandas DataFrame.
    ValueError
        If the input DataFrame is not in the correct format.

    Returns
    -------
    pd.DataFrame
        The standardized DataFrame.
    """

    idx_cols: List[str] = QuantamentalDataFrame.IndexCols
    metric_columns: List[str] = ms_constants.JPMAQS_METRICS

    # Check if the input DF contains the standard columns
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

        # check if there is at least one more column
        if len(df.columns) < 4:
            raise ValueError(fail_str)

    if isinstance(df, QuantamentalDataFrame) and type(df) is QuantamentalDataFrame:
        return QuantamentalDataFrame(df)

    # Convert date and ensure specific columns are strings in one step
    # 'datetime64[ns]' is the default dtype for datetime columns in pandas
    df["real_date"] = pd.to_datetime(
        df["real_date"],
        format="%Y-%m-%d",
    ).astype("datetime64[ns]")
    df["cid"] = df["cid"].astype("object")
    df["xcat"] = df["xcat"].astype("object")
    # sort by cid, xcat and real_date to allow viewing stacked timeseries easily
    df = (
        df.drop_duplicates(subset=idx_cols, keep="last")
        .sort_values(by=IDX_COLS_SORT_ORDER)
        .reset_index(drop=True)
    )

    # Sort the 'remaining' columns
    ## No more row-reordering or shape changes after this point

    jpmaqs_metrics = [mtr for mtr in metric_columns if mtr in df.columns]
    non_jpmaqs_metrics = (set(df.columns) - set(jpmaqs_metrics)) - set(idx_cols)
    col_order = idx_cols + jpmaqs_metrics + sorted(non_jpmaqs_metrics)
    df = df[col_order]

    # for every remaining col, try to convert to float
    for col in jpmaqs_metrics + list(non_jpmaqs_metrics):
        try:
            df[col] = df[col].astype(float)
        except:
            pass

    assert isinstance(df, QuantamentalDataFrame), "Failed to standardize DataFrame"
    return df


def drop_nan_series(
    df: pd.DataFrame, column: str = "value", raise_warning: bool = False
) -> QuantamentalDataFrame:
    """
    Drops any series that are entirely NaNs. Raises a user warning if any series are
    dropped and the raise warning flag is set to true.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to be cleaned.
    column : str
        The column to be used as the value column, defaults to "value".
    raise_warning : bool
        Whether to raise a warning if any series are dropped.

    Raises
    ------
    TypeError
        If the input is not a pandas DataFrame.
    ValueError
        If the input DataFrame is not in the correct format.

    Returns
    -------
    pd.DataFrame | QuantamentalDataFrame
        The cleaned DataFrame.
    """

    if not isinstance(df, QuantamentalDataFrame):
        raise TypeError("Argument `df` must be a Quantamental DataFrame.")

    if type(df) is QuantamentalDataFrame:
        return df.drop_nan_series(column=column, raise_warning=raise_warning)

    if not column in df.columns:
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


def qdf_to_ticker_df(df: pd.DataFrame, value_column: str = "value") -> pd.DataFrame:
    """
    Converts a standardized JPMaQS DataFrame to a wide format DataFrame with each column
    representing a ticker.

    Parameters
    ----------
    df : pd.DataFrame
        A standardised quantamental dataframe.
    value_column : str
        The column to be used as the value column, defaults to "value". If the specified
        column is not present in the DataFrame, a column named "value" will be used. If
        there is no column named "value", the first column in the DataFrame will be used
        instead.

    Returns
    -------
    pd.DataFrame
        The converted DataFrame.
    """

    if not isinstance(df, QuantamentalDataFrame):
        raise TypeError("Argument `df` must be a QuantamentalDataFrame.")

    if type(df) is QuantamentalDataFrame:
        return df.to_wide(value_column=value_column)

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
        .rename_axis(None, axis=1)  # TODO why rename axis?
    )


def ticker_df_to_qdf(df: pd.DataFrame, metric: str = "value") -> QuantamentalDataFrame:
    """
    Converts a wide format DataFrame (with each column representing a ticker) to a
    standardized JPMaQS DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        A wide format DataFrame.

    Returns
    -------
    pd.DataFrame
        The converted DataFrame.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Argument `df` must be a pandas DataFrame.")
    if not isinstance(metric, str):
        raise TypeError("Argument `metric` must be a string.")

    # pivot to long format
    df = (
        df.stack(level=0).reset_index().rename(columns={0: metric, "level_1": "ticker"})
    )

    df["cid"] = get_cid(df["ticker"])
    df["xcat"] = get_xcat(df["ticker"])
    df = df.drop(columns=["ticker"])

    return standardise_dataframe(df=df)


def concat_single_metric_qdfs(
    df_list: List[QuantamentalDataFrame],
    errors: str = "ignore",
) -> QuantamentalDataFrame:
    """
    Combines a list of Quantamental DataFrames into a single DataFrame.

    Parameters
    ----------
    df_list : List[QuantamentalDataFrame]
        A list of Quantamental DataFrames.
    errors : str
        The error handling method to use. If 'raise', then invalid items in the list
        will raise an error. If 'ignore', then invalid items will be ignored. Default is
        'ignore'.

    Returns
    -------
    QuantamentalDataFrame
        The combined DataFrame.
    """

    if not isinstance(df_list, list):
        raise TypeError("Argument `df_list` must be a list.")

    if errors not in ["raise", "ignore"]:
        raise ValueError("`errors` must be one of 'raise' or 'ignore'.")

    if errors == "raise":
        if not all([isinstance(df, QuantamentalDataFrame) for df in df_list]):
            raise TypeError(
                "All elements in `df_list` must be Quantamental DataFrames."
            )
    else:
        df_list = [df for df in df_list if isinstance(df, QuantamentalDataFrame)]
        if len(df_list) == 0:
            return None

    def _get_metric(df: QuantamentalDataFrame) -> str:
        lx = list(set(df.columns) - set(QuantamentalDataFrame.IndexCols))
        if len(lx) != 1:
            raise ValueError(
                "Each QuantamentalDataFrame must have exactly one metric column."
            )
        return lx[0]

    def _group_by_metric(
        dfl: List[QuantamentalDataFrame], fm: List[str]
    ) -> List[List[QuantamentalDataFrame]]:
        r = [[] for _ in range(len(fm))]
        while dfl:
            metric = _get_metric(df=dfl[0])
            r[fm.index(metric)] += [dfl.pop(0)]
        return r

    found_metrics = list(set(map(_get_metric, df_list)))

    df_list = _group_by_metric(dfl=df_list, fm=found_metrics)

    # use pd.merge to join on QuantamentalDataFrame.IndexCols
    df: pd.DataFrame = functools.reduce(
        lambda left, right: pd.merge(
            left, right, on=["real_date", "cid", "xcat"], how="outer"
        ),
        map(
            lambda fm: pd.concat(df_list.pop(0), axis=0, ignore_index=False),
            found_metrics,
        ),
    )

    return standardise_dataframe(df)


def apply_slip(
    df: QuantamentalDataFrame,
    slip: int,
    cids: Optional[List[str]] = None,
    xcats: Optional[List[str]] = None,
    tickers: Optional[List[str]] = None,
    metrics: List[str] = ["value"],
    extend_dates: bool = False,
    raise_error: bool = True,
) -> QuantamentalDataFrame:
    """
    Applies a "slip" to the DataFrame for the given cross-sections and categories, on the
    given metrics. A slip shifts the specified category n-days fowards in time, where n
    is the slip value. This is identical to a lag, but is measured in days, and must
    always be applied before any resampling.

    Parameters
    ----------
    target_df : QuantamentalDataFrame
        DataFrame to which the slip is applied.
    slip : int
        Slip to be applied.
    cids : List[str]
        List of cross-sections.
    xcats : List[str]
        List of target categories.
    metrics : List[str]
        List of metrics to which the slip is applied.
    extend_dates : bool
        If True, includes the dates added by the slip in the DataFrame. If False, only the
        input dates are included. Default is False.
    raise_error : bool
        If True, raises an error if the slip cannot be applied to all xcats in the target
        DataFrame. If False, raises a warning instead.

    Raises
    ------
    TypeError
        If the provided parameters are not of the expected type.
    ValueError
        If the provided parameters are semantically incorrect.

    Returns
    -------
    QuantamentalDataFrame
        DataFrame with the slip applied.
    """

    if not isinstance(df, QuantamentalDataFrame):
        raise TypeError("Argument `df` must be a QuantamentalDataFrame.")

    df = df.copy()
    if not (isinstance(slip, int) and slip >= 0):
        raise ValueError("Slip must be a non-negative integer.")

    if not any([cids, xcats, tickers]):
        raise ValueError("One of `tickers`, `cids` or `xcats` must be provided.")
    if tickers is not None:
        if cids is not None or xcats is not None:
            raise ValueError("Cannot specify both `tickers` and `cids`/`xcats`.")
    if not isinstance(extend_dates, bool):
        raise TypeError("`extend_dates` must be a boolean.")

    if not isinstance(metrics, list) or (
        isinstance(metrics, list) and not all(isinstance(m, str) for m in metrics)
    ):
        raise TypeError("`metrics` must be a list of strings.")

    missing_metrics = sorted(set(metrics) - set(df.columns))
    if missing_metrics:
        raise ValueError(f"Metrics {missing_metrics} are not present in the DataFrame.")

    if cids is None:
        cids = df["cid"].unique()
    if xcats is None:
        xcats = df["xcat"].unique()

    if tickers is not None:
        sel_tickers = tickers
    else:
        sel_tickers: List[str] = [f"{cid}_{xcat}" for cid in cids for xcat in xcats]

    if is_categorical_qdf(df):
        df = QuantamentalDataFrame(df).add_ticker_column()
    else:
        df["ticker"] = df["cid"] + "_" + df["xcat"]
    err_str = (
        "Tickers targetted for applying slip are not present in the DataFrame.\n"
        "Missing tickers: {tickers}"
    )

    if not set(sel_tickers).issubset(set(df["ticker"].unique())):
        missing_tickers = sorted(list(set(sel_tickers) - set(df["ticker"].unique())))
        _err_str = err_str.format(tickers=missing_tickers)
        if raise_error:
            raise ValueError(_err_str)
        else:
            warnings.warn(_err_str)

    if extend_dates:
        found_metrics = set(df.columns) - set(QuantamentalDataFrame.IndexCols)
        found_metrics = list(found_metrics - {"ticker"})
        new_dfs: List[QuantamentalDataFrame] = []
        for (ticker, cid, xcat), idx in df.groupby(
            ["ticker", "cid", "xcat"], observed=True
        ).groups.items():
            last_date = df.loc[idx, "real_date"].max()
            new_dts = pd.bdate_range(start=last_date, periods=slip + 1)[1:]
            assert set(new_dts).isdisjoint(set(df.loc[idx, "real_date"].unique()))
            dct = {"real_date": new_dts, "cid": cid, "xcat": xcat, "ticker": ticker}
            dct = {**dct, **{metric: np.nan for metric in found_metrics}}
            new_dfs.append(pd.DataFrame(dct))

        if is_categorical_qdf(df):
            new_df = QuantamentalDataFrame.from_qdf_list(new_dfs)
        else:
            new_df = pd.concat(new_dfs, axis=0, ignore_index=True)
        if is_categorical_qdf(df):
            df = QuantamentalDataFrame(df).update_df(df_add=new_df)
        else:
            df = pd.concat([df, new_df], axis=0, ignore_index=True)

        df = df.sort_values(by=["cid", "xcat", "real_date"])

    
    for col in metrics:
        tks_isin = df["ticker"].isin(sel_tickers)
        df.loc[tks_isin, col] = df.groupby("ticker", observed=True)[col].shift(slip)

    df = df.drop(columns=["ticker"]).reset_index(drop=True)
    assert isinstance(df, QuantamentalDataFrame), "Failed to apply slip."

    return df


def downsample_df_on_real_date(
    df: pd.DataFrame,
    groupby_columns: List[str] = [],
    freq: str = "M",
    agg: str = "mean",
):
    """
    Downsamples JPMaQS DataFrame.

    Parameters
    ----------
    df : pd.Dataframe
        standardized JPMaQS DataFrame with the necessary columns: 'cid', 'xcat',
        'real_date' and at least one column with values of interest.
    groupby_columns : List
        a list of columns used to group the DataFrame.
    freq : str
        frequency option. Per default the correlations are calculated based on the
        native frequency of the datetimes in 'real_date', which is business daily.
        Downsampling options include weekly ('W'), monthly ('M'), or quarterly ('Q') mean.
    agg : str
        aggregation method. Must be one of "mean" (default), "median", "min", "max",
        "first" or "last".

    Returns
    -------
    pd.DataFrame
        the downsampled DataFrame.
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
    non_groupby_columns = list(set(df.columns) - set(groupby_columns) - {"real_date"})
    res = (
        df.set_index("real_date")
        .groupby(groupby_columns, observed=True)[non_groupby_columns]
        .resample(freq)
    )
    if PD_OLD_RESAMPLE:  # pragma: no cover
        # resample only if the column is numeric
        res = res.agg(
            {
                col: agg
                for col in non_groupby_columns
                if pd.api.types.is_numeric_dtype(df[col])
            }
        ).reset_index()
        res.columns = res.columns.droplevel(-1)
    else:
        res = res.agg(agg, **RESAMPLE_NUMERIC_ONLY).reset_index()

    return res


def update_df(df: pd.DataFrame, df_add: pd.DataFrame, xcat_replace: bool = False):
    """
    Append a standard DataFrame to a standard base DataFrame with ticker replacement on
    the intersection.

    Parameters
    ----------
    df : pd.DataFrame
        standardised base JPMaQS DataFrame with the following necessary columns: 'cid',
        'xcat', 'real_date' and 'value'.
    df_add : pd.DataFrame
        another standardised JPMaQS DataFrame, with the latest values, to be added with
        the necessary columns: 'cid', 'xcat', 'real_date', and 'value'. Columns that are
        present in the base DataFrame but not in the appended DataFrame will be populated
        with NaN values.
    xcat_replace : bool
        all series belonging to the categories in the added DataFrame will be replaced,
        rather than just the added tickers.

    Returns
    -------
    pd.DataFrame
        standardised DataFrame with the latest values of the modified or newly defined
        tickers added.


    ..note::
        Tickers are combinations of cross-sections and categories.
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

    if type(df) is QuantamentalDataFrame:
        return df.update_df(df_add=df_add, xcat_replace=xcat_replace)

    error_message = (
        "The two Quantamental DataFrames must share at least "
        "four columns including 'real_date', 'cid', and 'xcat'."
    )

    all_cols_set = df_cols.union(df_add_cols)
    if not len(all_cols_set - set(QuantamentalDataFrame.IndexCols)):
        raise ValueError(error_message)

    if not xcat_replace:
        df = update_tickers(df, df_add)

    else:
        df = update_categories(df, df_add)

    # sort same as in `standardise_dataframe`
    return df.sort_values(by=IDX_COLS_SORT_ORDER).reset_index(drop=True)


def update_tickers(df: pd.DataFrame, df_add: pd.DataFrame):
    """
    Method used to update aggregate DataFrame on a ticker level.

    Parameters
    ----------
    df : pd.DataFrame
        aggregate DataFrame used to store all tickers.
    df_add : pd.DataFrame
        DataFrame with the latest values.
    """

    if not isinstance(df, QuantamentalDataFrame):
        raise TypeError("The base DataFrame must be a Quantamental Dataframe.")
    if not isinstance(df_add, QuantamentalDataFrame):
        raise TypeError("The added DataFrame must be a Quantamental Dataframe.")

    if df.empty:
        return df_add
    if df_add.empty:
        return df

    df = pd.concat([df, df_add], axis=0, ignore_index=True)

    df = df.drop_duplicates(
        subset=["real_date", "xcat", "cid"], keep="last"
    ).reset_index(drop=True)
    return df


def update_categories(df: pd.DataFrame, df_add):
    """
    Method used to update the DataFrame on the category level.

    Parameters
    ----------
    df : pd.DataFrame
        base DataFrame.
    df_add : pd.DataFrame
        appended DataFrame.
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

    Parameters
    ----------
    df : pd.Dataframe
        standardized JPMaQS DataFrame with the necessary columns: 'cid', 'xcat',
        'real_date' and 'value'.
    xcats : Union[str, List[str]]
        extended categories to be filtered on. Default is all in the DataFrame.
    cids : List[str]
        cross sections to be checked on. Default is all in the dataframe.
    start : str
        string representing the earliest date. Default is None.
    end : str
        string representing the latest date. Default is None.
    blacklist : dict
        cross-sections with date ranges that should be excluded from the data frame. If
        one cross-section has several blacklist periods append numbers to the cross-section
        code.
    out_all : bool
        if True the function returns reduced dataframe and selected/ available xcats and
        cids. Default is False, i.e. only the DataFrame is returned
    intersect : bool
        if True only retains cids that are available for all xcats. Default is False.

    Returns
    -------
    pd.Dataframe
        reduced DataFrame that also removes duplicates or (for out_all True) DataFrame
        and available and selected xcats and cids.
    """

    if not isinstance(df, QuantamentalDataFrame):
        raise TypeError("Argument `df` must be a standardised Quantamental DataFrame.")

    if type(df) is QuantamentalDataFrame:
        return df.reduce_df(
            cids=cids,
            xcats=xcats,
            start=start,
            end=end,
            blacklist=blacklist,
            out_all=out_all,
            intersect=intersect,
        )

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

    Parameters
    ----------
    df : pd.Dataframe
        standardized dataframe with the following columns: 'cid', 'xcat', 'real_date'.
    ticks : List[str]
        tickers (cross sections + base categories)
    start : str
        string in ISO 8601 representing earliest date. Default is None.
    end : str
        string ISO 8601 representing the latest date. Default is None.
    blacklist : dict
        cross sections with date ranges that should be excluded from the dataframe. If
        one cross section has several blacklist periods append numbers to the cross section
        code.

    Returns
    -------
    pd.Dataframe
        reduced dataframe that also removes duplicates
    """

    if type(df) is QuantamentalDataFrame:
        return df.reduce_df_by_ticker(
            tickers=ticks,
            start=start,
            end=end,
            blacklist=blacklist,
        )

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

    Parameters
    ----------
    dfx : List[str]
        standardised DataFrame defined exclusively on a single category.
    xcat_agg : List[str]
        associated aggregation method for the respective category.
    """

    dfx = dfx.groupby(["xcat", "cid", "custom_date"], observed=True)
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

    Parameters
    ----------
    dfw : pd.DataFrame
        group-by DataFrame which has been down-sampled. The respective aggregation
        method will be applied.
    explanatory_xcats : List[str]
        list of explanatory category(s).
    agg_meth : str
        aggregation method used for all explanatory variables.
    sum_condition : dict
        required boolean to negate erroneous zeros if the aggregate method used, for the
        explanatory variable, is sum.
    lag : int
        lag of explanatory category(s). Applied uniformly to each category.
    """

    dfw_explanatory = pd.DataFrame()
    for xcat in explanatory_xcats:
        if not sum_condition:
            explanatory_col = dfw[xcat].agg(agg_method).astype(dtype=np.float32)
        else:
            explanatory_col = dfw[xcat].sum(min_count=1)

        if lag > 0:
            explanatory_col: pd.Series
            explanatory_col = explanatory_col.groupby(
                level=0,
                observed=True,
            ).shift(lag)

        dfw_explanatory[xcat] = explanatory_col

    dfw_explanatory.index.names = ["cid", "real_date"]
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

    Parameters
    ----------
    df : pd.Dataframe
        standardized JPMaQS DataFrame with the following necessary columns: 'cid',
        'xcat', 'real_date' and at least one column with values of interest.
    xcats : List[str]
        extended categories involved in the custom DataFrame. The last category in the
        list represents the dependent variable, and the (n - 1) preceding categories will be
        the explanatory variables(s).
    cids : List[str]
        cross-sections to be included. Default is all in the DataFrame.
    val : str
        name of column that contains the values of interest. Default is 'value'.
    start : str
        earliest date in ISO 8601 format. Default is None, i.e. earliest date in
        DataFrame is used.
    end : str
        latest date in ISO 8601 format. Default is None, i.e. latest date in DataFrame
        is used.
    blacklist : dict
        cross-sections with date ranges that should be excluded from the DataFrame. If
        one cross section has several blacklist periods append numbers to the cross section
        code.
    years : int
        number of years over which data are aggregated. Supersedes the "freq" parameter
        and does not allow lags, Default is None, i.e. no multi-year aggregation.
    freq : str
        letter denoting frequency at which the series are to be sampled. This must be
        one of 'D', 'W', 'M', 'Q', 'A'. Default is 'M'. Will always be the last business day
        of the respective frequency.
    lag : int
        lag (delay of arrival) of explanatory category(s) in periods as set by freq.
        Default is 0.
    fwin : int
        forward moving average window of first category. Default is 1, i.e no average.
        Note: This parameter is used mainly for target returns as dependent variable.
    xcat_aggs : List[str]
        exactly two aggregation methods. Default is 'mean' for both. The same
        aggregation method, the first method in the parameter, will be used for all
        explanatory variables.

    Returns
    -------
    pd.DataFrame
        custom DataFrame with category columns. All rows that contain NaNs will be
        excluded.  N.B.: The number of explanatory categories that can be included is not
        restricted and will be appended column-wise to the returned DataFrame. The order of
        the DataFrame's columns will reflect the order of the categories list.
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
            ],
            observed=True,
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

    if dfc.index.dtypes["cid"].name == "category":
        # in case the incoming DF has a categorical index it, the index needs to be
        # converted to object type to avoid issues downstream
        new_outer_index = dfc.index.levels[0].astype("object")
        new_index = pd.MultiIndex(
            levels=[new_outer_index, dfc.index.levels[1]],
            codes=dfc.index.codes,
            names=dfc.index.names,
        )
        dfc.index = new_index

    # Adjusted to account for multiple signals requested. If the DataFrame is
    # two-dimensional, signal & a return, NaN values will be handled inside other
    # functionality, as categories_df() is simply a support function. If the parameter
    # how is set to "any", a potential unnecessary loss of data on certain categories
    # could arise.
    return dfc.dropna(axis=0, how="all")


def estimate_release_frequency(
    timeseries: Optional[pd.Series] = None,
    df_wide: Optional[pd.DataFrame] = None,
    atol: Optional[float] = None,
    rtol: Optional[float] = None,
) -> Union[Optional[str], Dict[str, Optional[str]]]:
    """
    Estimates the release frequency of a timeseries, by inferring the frequency of the
    timeseries index. Before calling `pd.infer_freq`, the function drops NaNs, and rounds
    values as specified by the tolerance parameters to allow dropping of "duplicate" values.
    
    Parameters
    ----------
    timeseries : pd.Series, optional
        The timeseries to be used to estimate the release frequency. Only one of
        `timeseries` or `df_wide` must be passed.
    df_wide : pd.DataFrame, optional
        The wide DataFrame to be used to estimate the release frequency. This mode
        processes each column of the DataFrame as a timeseries. Only one of `timeseries`
        or `df_wide` must be passed.
    atol : float, optional
        The absolute tolerance for the difference between two values. If `None`, no
        rounding is applied.
    rtol : float, optional
        The relative tolerance for the difference between two values. If `None`, no
        rounding is applied.

    Returns
    -------
    str or dict
        The estimated release frequency. If `df_wide` is passed, a dictionary with the
        column names as keys and the estimated frequencies as values is returned.
    """

    if df_wide is not None:
        if timeseries is not None:
            raise ValueError("Only one of `timeseries` or `df_wide` must be passed.")
        if not isinstance(df_wide, pd.DataFrame):
            raise TypeError("Argument `df_wide` must be a pandas DataFrame.")
        if df_wide.empty or df_wide.index.name != "real_date":
            raise ValueError(
                "Argument `df_wide` must be a non-empty pandas DataFrame with a datetime index `'real_date'`."
            )

        return {
            col: estimate_release_frequency(
                timeseries=df_wide[col], atol=atol, rtol=rtol
            )
            for col in df_wide.columns
        }

    if bool(atol) and bool(rtol):
        raise ValueError("Only one of `diff_atol` or `diff_rtol` must be passed.")

    if atol:
        if not isinstance(atol, Number) or atol <= 0:
            raise TypeError("Argument `diff_atol` must be a float.")
    elif rtol:
        if not isinstance(rtol, Number):
            raise TypeError("Argument `diff_rtol` must be a float.")
        if not (0 <= rtol <= 1):
            raise ValueError("Argument `diff_rtol` must be a float between 0 and 1.")

    for _arg, _name in zip([atol, rtol], ["atol", "rtol"]):
        if _arg is not None:
            if not isinstance(_arg, Number) or _arg < 0:
                raise TypeError(
                    f"Argument `{_name}` must be a float greater than 0 or None."
                )
    if rtol or atol:
        _scale = timeseries.abs().max() * rtol if rtol else atol
        _dp = -int(np.floor(np.log10(_scale)))
        timeseries: pd.Series = timeseries.round(_dp)
    timeseries: pd.Series = timeseries.dropna().drop_duplicates(keep="first")

    if (
        not isinstance(timeseries, pd.Series)
        or timeseries.empty
        or not isinstance(timeseries.index, pd.DatetimeIndex)
    ):
        raise TypeError(
            "Argument `timeseries` must be a non-empty pandas Series with "
            "a datetime index `'real_date'`."
        )

    return _determine_freq(timeseries.index.tolist())


def _determine_freq(dates: List[str]) -> str:
    """
    Backend function to determine the frequency of a timeseries from the dates in the
    timeseries.

    Parameters
    ----------
    dates : List[str]
        A list of dates in the timeseries.

    Returns
    -------
    str
        The estimated frequency of the timeseries. One of 'D', 'W', 'M', 'Q', 'A'.
    """
    dates: pd.DatetimeIndex = pd.to_datetime(sorted(dates))
    deltas = dates.to_series().diff().dt.days[1:]
    frequencies = {
        "D": 1,
        "W": 7,
        "M": 30,
        "Q": 91,
        "A": 365,
    }
    closest_freq = deltas.map(
        lambda x: min(frequencies, key=lambda freq: abs(x - frequencies[freq]))
    )
    return closest_freq.value_counts().idxmax()

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
    Returns a series of end-of-period dates for a given frequency. Dates can be passed
    as a series, index, a generic iterable or as a start and end date.

    Parameters
    ----------
    freq : str
        The frequency string. Must be one of "D", "W", "M", "Q", "A".
    dates : pd.DatetimeIndex | pd.Series | Iterable[pd.Timestamp]
        The dates to be used to generate the end-of-period dates. Can be passed as a
        series, index, a generic iterable or as a start and end date.
    start_date : str | pd.Timestamp
        The start date. Must be passed if dates is not passed.
    """

    direction = "end"
    return _get_edge_dates(
        dates=dates,
        start_date=start_date,
        end_date=end_date,
        freq=freq,
        direction=direction,
    )


def merge_categories(
    df: pd.DataFrame, xcats: List[str], new_xcat: str, cids: List[str] = None
):
    """
    Merges categories into a new category, given a list of categories to be merged. The
    merging is done in a preferred order, i.e. the first category in the list will be
    the preferred value for each real_date and if the first category does not have a
    value for a given real_date, the next category in the list will be used, etc...

    Parameters
    ----------
    df : pd.DataFrame
        standardized JPMaQS DataFrame with the necessary columns 'cid', 'xcat',
        'real_date' and at least one column with values of interest.
    xcats : List[str]
        extended categories to be merged.
    cids : List[str]
        cross sections to be included. Default is all in the DataFrame.
    new_xcat : str
        name of the new category to be created. Default is None.

    Returns
    -------
    pd.DataFrame
        DataFrame with the merged category.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("The DataFrame must be a pandas DataFrame.")
    if not isinstance(xcats, list):
        raise TypeError("The categories must be a list of strings.")
    if not all(isinstance(xcat, str) for xcat in xcats):
        raise TypeError("The categories must be a list of strings.")
    if not isinstance(df, QuantamentalDataFrame):
        raise TypeError("The DataFrame must be a Quantamental DataFrame.")
    if not isinstance(new_xcat, str):
        raise TypeError("The new category must be a string.")
    if not set(xcats).issubset(df["xcat"].unique()):
        raise ValueError("The categories must be present in the DataFrame.")
    if cids is None:
        cids = list(df["cid"].unique())
    if not isinstance(cids, list):
        raise TypeError("The cross sections must be a list of strings.")
    if not all(isinstance(cid, str) for cid in cids):
        raise TypeError("The cross sections must be a list of strings.")
    if not set(cids).issubset(df["cid"].unique()):
        raise ValueError("The cross sections must be present in the DataFrame.")

    unique_dates = df["real_date"].unique()
    real_dates = [pd.Timestamp(date) for date in unique_dates]

    def _get_values_for_xcat(real_dates, xcat_index, cid):
        values = df[
            (df["real_date"].isin(real_dates))
            & (df["xcat"] == xcats[xcat_index])
            & (df["cid"] == cid)
        ]
        if not real_dates == list(values["real_date"].unique()):
            if xcat_index + 1 >= len(xcats):
                return values
            values = update_df(
                values,
                _get_values_for_xcat(
                    list(set(real_dates) - set(values["real_date"].unique())),
                    xcat_index + 1,
                    cid,
                ),
            )

        values.loc[:, "xcat"] = new_xcat
        return values

    result_df = None

    for cid in cids:
        if result_df is None:
            result_df = _get_values_for_xcat(real_dates, 0, cid)
        else:
            result_df = update_df(
                result_df, _get_values_for_xcat(real_dates, xcat_index=0, cid=cid)
            )

    return result_df.sort_values(by=IDX_COLS_SORT_ORDER).reset_index(drop=True)


def get_sops(
    dates: Optional[Union[pd.DatetimeIndex, pd.Series, Iterable[pd.Timestamp]]] = None,
    start_date: Optional[Union[str, pd.Timestamp]] = None,
    end_date: Optional[Union[str, pd.Timestamp]] = None,
    freq: str = "M",
) -> pd.Series:
    """
    Returns a series of start-of-period dates for a given frequency. Dates can be passed
    as a series, index, a generic iterable or as a start and end date.

    Parameters
    ----------
    freq : str
        The frequency string. Must be one of "D", "W", "M", "Q", "A".
    dates : pd.DatetimeIndex | pd.Series | Iterable[pd.Timestamp]
        The dates to be used to generate the start-of-period dates. Can be passed as a
        series, index, a generic iterable or as a start and end date.
    start_date : str | pd.Timestamp
        The start date. Must be passed if dates is not passed.
    """

    direction = "start"
    return _get_edge_dates(
        dates=dates,
        start_date=start_date,
        end_date=end_date,
        freq=freq,
        direction=direction,
    )


def concat_categorical(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """
    Concatenate two DataFrames with categorical columns.  The dtypes of the of the
    second DataFrame will be cast to the dtypes of the first. The columns of the
    DataFrames must be identical.

    Parameters
    ----------
    df1 : pd.DataFrame
        The first DataFrame.
    df2 : pd.DataFrame
        The second DataFrame.

    Returns
    -------
    pd.DataFrame
        The concatenated DataFrame with the same columns as the input.
    """

    if not isinstance(df1, pd.DataFrame) or not isinstance(df2, pd.DataFrame):
        raise TypeError("Both DataFrames must be pandas DataFrames.")

    if not (set(df1.columns) == set(df2.columns)):
        raise ValueError("The columns of the two DataFrames must be identical.")

    # Explicitly set or create categorical columns based on the data in model_df_long
    for col in df1.select_dtypes(include="category").columns:
        df2[col] = df2[col].astype("category")

    non_categorical_cols = df1.select_dtypes(exclude="category").columns
    df2[non_categorical_cols] = df2[non_categorical_cols].astype(
        df1[non_categorical_cols].dtypes.to_dict()
    )
    # If one DataFrame is None, return the other (if both are None, return None)
    if df1.empty:
        return df2.reset_index(drop=True) if df2 is not None else None
    if df2 is None or df2.empty:
        return df1.reset_index(drop=True)
    categorical_cols = list(
        set(df1.select_dtypes(include="category").columns).union(
            df2.select_dtypes(include="category").columns
        )
    )
    for col in categorical_cols:
        # Find the combined categories from both DataFrames for the current column
        combined_categories = pd.Categorical(
            df1[col].cat.categories.union(df2[col].cat.categories)
        )

        # Re-assign the categorical column with the combined categories to both DataFrames
        df1[col] = pd.Categorical(df1[col], categories=combined_categories)
        df2[col] = pd.Categorical(df2[col], categories=combined_categories)

    # Concatenate the two DataFrames and reset the index
    concatenated_df = pd.concat([df1, df2], axis=0, ignore_index=True)

    return concatenated_df


def _insert_as_categorical(df, column_name, category_name, column_idx):
    """
    Inserts a column into a dataframe as a categorical column.
    """
    df.insert(
        column_idx,
        column_name,
        pd.Categorical(
            [category_name] * df.shape[0],
            categories=[category_name],
        ),
    )
    return df


def forward_fill_wide_df(df, blacklist=None, n=1):
    """
    Forward fills NaN values in a wide DataFrame using the last valid value in each column.
    It will not forward fill gaps in the data, only the next `n` periods after the last valid value.
    
    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to be forward filled in `wide` format, where each column represents a
        cross-section and the index are dates.
    blacklist : dict, optional
        A dictionary where keys are column names and values are lists of two elements,
        representing the start and end dates of periods to be excluded from filling.
    n : int, optional
        The number of periods to fill forward. Default is 1, meaning only the next period
    """
    if blacklist is None:
        blacklist = {}
    if not isinstance(blacklist, dict):
        raise TypeError("blacklist argument must be a dictionary.")
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")
    if not isinstance(n, int):
        raise ValueError("Parameter 'n' must be an integer.")
    
    for col in df.columns:
        series = df[col]

        last_valid_idx = series.last_valid_index()
        if last_valid_idx is None:
            continue
        last_pos = series.index.get_loc(last_valid_idx)

        fill_positions = range(last_pos + 1, min(last_pos + n + 1, len(series)))
        if not fill_positions:
            continue
        mask = pd.Series(False, index=series.index)
        mask.iloc[list(fill_positions)] = True

        blist = blacklist.get(col)
        if blist:
            start, end = pd.to_datetime(blist[0]), pd.to_datetime(blist[1])
            blacklist_mask = series.index.to_series().between(start, end)
            mask &= ~blacklist_mask
        to_fill = mask & series.isna()
        df.loc[to_fill, col] = series.iloc[last_pos]
    return df


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
        df=QuantamentalDataFrame(dfd),
        xcats=xcats,
        cids=cids,
        freq="M",
        # lag=1,
        xcat_aggs=["last", "sum"],
        # years=5,
        # start="2000-01-01",
    )
    print("HI")
