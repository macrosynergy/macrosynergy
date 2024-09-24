import warnings
import datetime
from typing import (
    Dict,
    List,
    NamedTuple,
    Optional,
    Union,
)
import pandas as pd

from macrosynergy.management.utils import reduce_df, is_valid_iso_date
from macrosynergy.management.utils.df_utils import standardise_dataframe


class QDFArgs(NamedTuple):
    """
    Contains the QuantamentalDataFrame and associated arguments.
    """

    df: pd.DataFrame
    cids: List[str]
    xcats: List[str]
    metrics: List[str]
    start: str
    end: str


def validate_and_reduce_qdf(
    df: pd.DataFrame,
    cids: Optional[List[str]] = None,
    xcats: Optional[List[str]] = None,
    metrics: Optional[List[str]] = None,
    intersect: Optional[bool] = False,
    tickers: Optional[List[str]] = None,
    blacklist: Optional[Dict[str, List[str]]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
):
    """
    Validates the inputs to a function that takes a DataFrame as its first argument.
    The DataFrame is then reduced according to the inputs.

    :param <pd.DataFrame> df: A DataFrame with the following columns:
        'cid', 'xcat', 'real_date', and at least one metric from -
        'value', 'grading', 'eop_lag', or 'mop_lag'.
    :param <List[str]> cids: A list of cids to select from the DataFrame.
        If None, all cids are selected.
    :param <List[str]> xcats: A list of xcats to select from the DataFrame.
        If None, all xcats are selected.
    :param <List[str]> metrics: A list of metrics to select from the DataFrame.
        If None, all metrics are selected.
    :param <bool> intersect: if True only retains cids that are available for
        all xcats. Default is False.
    :param <List[str]> tickers: A list of tickers that will be selected from the DataFrame
        if they exist, regardless of start, end, blacklist, and intersect arguments.
    :param <dict> blacklist: cross-sections with date ranges that should be excluded from
        the data frame. If one cross-section has several blacklist periods append numbers
        to the cross-section code.
    :param <str> start: ISO-8601 formatted date string. Select data from
        this date onwards. If None, all dates are selected.
    :param <str> end: ISO-8601 formatted date string. Select data up to
        and including this date. If None, all dates are selected.

    :return <QDFArgs>: A NamedTuple that contains the validated arguments.
    """

    df: pd.DataFrame = df.copy()
    df = standardise_dataframe(df=df)

    metrics = _validate_metrics(df=df, metrics=metrics)

    cids_provided: bool = cids is not None
    xcats_provided: bool = xcats is not None
    missing_cids: List[str]
    missing_xcats: List[str]

    cids, missing_cids = _set_or_find_missing_in_df(
        df=df, col_name="cid", values=cids, param_name="cids"
    )
    xcats, missing_xcats = _set_or_find_missing_in_df(
        df=df, col_name="xcat", values=xcats, param_name="xcats"
    )

    start, end = _validate_start_and_end_dates(df, start, end)

    ticker_df: pd.DataFrame = pd.DataFrame()
    if tickers is not None:
        ticker_df = _get_ticker_df(df=df, tickers=tickers, metrics=metrics)

    df: pd.DataFrame
    r_xcats: List[str]
    r_cids: List[str]
    df, r_xcats, r_cids = reduce_df(
        df=df,
        cids=cids if isinstance(cids, list) else [cids],
        xcats=xcats if isinstance(xcats, list) else [xcats],
        intersect=intersect,
        start=start,
        end=end,
        blacklist=blacklist,
        out_all=True,
    )

    df: pd.DataFrame = pd.concat([df, ticker_df], axis=0)
    df = df.drop_duplicates()

    if (
        ((len(r_xcats) != len(xcats) - len(missing_xcats)) and xcats_provided)
        or ((len(r_cids) != len(cids) - len(missing_cids)) and cids_provided)
    ) and not intersect:
        m_cids: List[str] = list(set(cids).difference(set(r_cids), set(missing_cids)))
        m_xcats: List[str] = list(
            set(xcats).difference(set(r_xcats), set(missing_xcats))
        )

        warnings.warn(
            "The provided arguments resulted in a DataFrame that does not "
            "contain all the requested cids and xcats. "
            + (f"Missing cids: {m_cids}. " if m_cids else "")
            + (f"Missing xcats: {m_xcats}. " if m_xcats else "")
        )
        for m_cid in m_cids:
            cids.remove(m_cid)
        for m_xcat in m_xcats:
            xcats.remove(m_xcat)

    if df.empty:
        raise ValueError(
            "The arguments provided resulted in an "
            "empty DataFrame when filtered (see `reduce_df`)."
        )

    return QDFArgs(df, cids, xcats, metrics, start, end)


def _get_ticker_df(df: pd.DataFrame, tickers: List[str], metrics: Optional[List[str]]):
    """
    Filters a QuantamentalDataFrame by tickers.

    :param <pd.DataFrame> df: a Pandas DataFrame.
    :param <List[str]> metrics: a list of metrics.
    :param <List[str]> tickers: a list of tickers.

    :return <pd.DataFrame>: the filtered DataFrame.
    """
    df_tickers: List[pd.DataFrame] = [pd.DataFrame()]
    for ticker in tickers:
        _cid, _xcat = ticker.split("_", 1)
        df_tickers.append(
            df.loc[
                (df["cid"] == _cid) & (df["xcat"] == _xcat),
                ["real_date", "cid", "xcat"] + metrics,
            ]
        )
    ticker_df: pd.DataFrame = pd.concat(df_tickers, axis=0)
    return ticker_df


def _validate_start_and_end_dates(df: pd.DataFrame, start: str, end: str):
    """
    Determines start and end dates for a DataFrame.

    :param <pd.DataFrame> df: DataFrame to be filtered.
    :param <str> start: ISO-8601 formatted date string. If None,
        the earliest date in the DataFrame is used.
    :param <str> end: ISO-8601 formatted date string. If None,
        the latest date in the DataFrame is used.

    :return <Tuple[str]>: Tuple of start and end dates.
    """
    if start is None:
        start: str = pd.Timestamp(df["real_date"].min()).strftime("%Y-%m-%d")
    if end is None:
        end: str = pd.Timestamp(df["real_date"].max()).strftime("%Y-%m-%d")
    for var, name in [(start, "start"), (end, "end")]:
        if not is_valid_iso_date(var):
            raise ValueError(f"`{name}` must be a valid ISO date string")

    return start, end


def _validate_metrics(df: pd.DataFrame, metrics: List[str]):
    """
    Validates the metrics passed to a function.

    :param <pd.DataFrame> df: a Pandas DataFrame.
    :param <List[str]> metrics: a list of metrics to be checked.

    :return <List[str]>: a list of metrics to be used.
    """
    required_columns: List[str] = ["real_date", "cid", "xcat"]
    if metrics is None:
        metrics: List[str] = list(set(df.columns) - set(required_columns))
    required_columns += metrics
    if not set(required_columns).issubset(set(df.columns)):
        raise ValueError(
            f"DataFrame must contain the following columns: {required_columns}"
        )
    return metrics


def _set_or_find_missing_in_df(
    df: pd.DataFrame, col_name: str, values: Optional[List], param_name: str
):
    """
    Returns the values passed to a function and a list of values that are not found
    in a specific column of a DataFrame. If values is None, all unique values in the
    DataFrame are returned.

    :param <pd.DataFrame> df: a Pandas DataFrame.
    :param <str> col_name: name of column in the DataFrame.
    :param <List> values: list of values to be checked.
    :param <str> param_name: name of parameter passed to function.

    :return <List>: list of values that are not in the DataFrame.
    """
    missing_values: List = []
    if values is None:
        values = df[col_name].unique().tolist()
    else:
        missing_values = _find_missing_in_df(
            df=df, col_name=col_name, values=values, param_name=param_name
        )
    return values, missing_values


def _find_missing_in_df(df: pd.DataFrame, col_name: str, values: List, param_name: str):
    """
    Finds values in a list that are not in a specific column of a DataFrame.

    :param <pd.DataFrame> df: a Pandas DataFrame.
    :param <str> col_name: name of column in the DataFrame.
    :param <List> values: list of values to be checked.
    :param <str> param_name: name of parameter passed to function.

    :return <List>: list of values that are not in the DataFrame.
    """
    missing: List = []
    if not set(values).issubset(set(df[col_name].unique())):
        # warn
        warnings.warn(
            f"The following {col_name}(s), passed in `{param_name}`,"
            " are not in the DataFrame `df`: "
            f"{list(set(values) - set(df[col_name].unique()))}."
        )
        missing = list(set(values) - set(df[col_name].unique()))

    return missing


def _validate_Xy_learning(X: pd.DataFrame, y: Union[pd.DataFrame, pd.Series]):
    """
    Validates the expected long-format inputs and targets expected for the learning
    submodule.
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError("The X argument must be a pandas DataFrame.")
    if not isinstance(y, pd.Series) and not isinstance(y, pd.DataFrame):
        raise TypeError("The y argument must be a pandas Series or DataFrame.")
    if not isinstance(X.index, pd.MultiIndex):
        raise ValueError("X must be multi-indexed.")
    if not isinstance(y.index, pd.MultiIndex):
        raise ValueError("y must be multi-indexed.")
    if not isinstance(X.index.get_level_values(1)[0], datetime.date):
        raise TypeError("The inner index of X must be datetime.date.")
    if not isinstance(y.index.get_level_values(1)[0], datetime.date):
        raise TypeError("The inner index of y must be datetime.date.")
    if not X.index.equals(y.index):
        raise ValueError(
            "The indices of the input dataframe X and the output dataframe y don't "
            "match."
        )
