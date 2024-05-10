from typing import List, Optional, Union, TypeVar, overload
from collections.abc import Iterable, Callable
import pandas as pd
import numpy as np
import datetime
import functools
import os
import glob
import fnmatch
from macrosynergy.management.types import QuantamentalDataFrame
from macrosynergy.management.utils import (
    ticker_df_to_qdf,
    qdf_to_ticker_df,
    standardise_dataframe,
    concat_qdfs,
    get_cid,
    get_xcat,
)

DateLike = TypeVar("DateLike", str, pd.Timestamp, np.datetime64, datetime.datetime)


def create_local_expressions(
    cids: Optional[Iterable[str]] = None,
    xcats: Optional[Iterable[str]] = None,
    tickers: Optional[Iterable[str]] = None,
    metrics: Optional[Union[str, Iterable[str]]] = ["all"],
) -> List[str]:
    if isinstance(metrics, str):
        metrics = [metrics]
    if "all" in metrics:
        metrics += ["value", "grading", "eop_lag", "mop_lag"]
        metrics = sorted(set(metrics))

    assert not (
        (cids is not None) ^ (xcats is not None)
    ), "Arguments `cids` and `xcats` must be both None or both not None."
    if cids is None:
        cids: List[str] = []
        xcats: List[str] = []

    if tickers is None:
        tickers: List[str] = []

    tickers += [f"{cid}_{xcat}" for cid in cids for xcat in xcats]
    tickers = sorted(set(tickers))

    return [f"{ticker},{metric}" for ticker in tickers for metric in metrics]


def check_df_type_decorator(func: Callable) -> Callable:
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if "df" in kwargs:
            if not isinstance(kwargs["df"], pd.DataFrame):
                raise TypeError("Argument `df` must be a pandas DataFrame.")
        return func(*args, **kwargs)

    return wrapper


@overload
def local_expression_to_ticker(expression: str) -> str: ...


@overload
def local_expression_to_ticker(expression: Iterable[str]) -> List[str]: ...


def check_local_expression(expression: Union[str, Iterable[str]]) -> bool:
    if isinstance(expression, str):
        return expression.count(",") == 1 and all(map(bool, expression.split(",")))

    assert isinstance(expression, Iterable)
    return all(map(check_local_expression, expression))


@overload
def local_expression_to_ticker(expression: str) -> str: ...


@overload
def local_expression_to_ticker(expression: Iterable[str]) -> List[str]: ...


def jpmaqs_expression_to_local_expression(expression: Union[str, Iterable[str]]) -> str:
    if isinstance(expression, str):
        assert expression.startswith("DB(JPMAQS,") and expression.endswith(")")
        return expression.replace("DB(JPMAQS,", "").replace(")", "")

    assert isinstance(expression, Iterable)
    return list(map(jpmaqs_expression_to_local_expression, expression))


@overload
def check_jpmaqs_expression(expression: str) -> bool: ...


@overload
def check_jpmaqs_expression(expression: Iterable[str]) -> bool: ...


def check_jpmaqs_expression(expression: Union[str, Iterable[str]]) -> bool:
    if isinstance(expression, str):
        return expression.startswith("DB(JPMAQS,") and expression.endswith(")")

    assert isinstance(expression, Iterable)
    return all(map(check_jpmaqs_expression, expression))


@check_df_type_decorator
def ticker_df_to_expression_df(df: pd.DataFrame, metric: str = "value") -> pd.DataFrame:
    if not isinstance(metric, str):
        raise TypeError("Argument `value_column` must be a string.")

    df.columns = create_local_expressions(tickers=df.columns, metrics=[metric])
    return df


def qdf_to_expression_df(df: QuantamentalDataFrame) -> pd.DataFrame:
    if not isinstance(df, QuantamentalDataFrame):
        raise TypeError("Argument `df` must be a QuantamentalDataFrame.")

    metrics = set(df.columns) - set(QuantamentalDataFrame.IndexCols)
    expr_df = pd.concat(
        [
            ticker_df_to_expression_df(qdf_to_ticker_df(df, metric), metric)
            for metric in metrics
        ],
        axis=1,
    )
    return expr_df.reindex(sorted(expr_df.columns), axis=1)


@check_df_type_decorator
def expression_df_to_ticker_df(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns=lambda x: str(x).split(",")[0])


@check_df_type_decorator
def get_valid_local_expressions(
    df: pd.DataFrame,
    cids: Optional[Iterable[str]] = None,
    xcats: Optional[Iterable[str]] = None,
    tickers: Optional[Iterable[str]] = None,
    metrics: Optional[Union[str, Iterable[str]]] = ["all"],
    cid: Optional[str] = None,
    xcat: Optional[str] = None,
    ticker: Optional[str] = None,
) -> List[str]:
    all_available_tickers = get_tickers_from_expression_df(df)

    if isinstance(cid, str):
        cids = [cid] + (cids or [])
    if isinstance(xcat, str):
        xcats = [xcat] + (xcats or [])
    if isinstance(ticker, str):
        tickers = [ticker] + (tickers or [])
    if tickers is None:
        if cids is None:
            cids = get_cids_from_expression_df(df)
        if xcats is None:
            xcats = get_xcats_from_expression_df(df)
        tickers = [f"{cid}_{xcat}" for cid in cids for xcat in xcats]

    _contains_wildcard = lambda x: "*" in x or "?" in x
    wtickers = list(filter(_contains_wildcard, tickers))
    if wtickers:
        tickers = list(filter(lambda x: not _contains_wildcard(x), tickers))
        new_tickers = []
        for wtkr in wtickers:
            new_tickers.extend(fnmatch.filter(all_available_tickers, wtkr))
        tickers = list(set(tickers + new_tickers))
    local_exprs = create_local_expressions(
        cids=cids, xcats=xcats, tickers=tickers, metrics=metrics
    )
    return sorted(set(local_exprs) & set(df.columns))


@check_df_type_decorator
def get_metrics_from_expression_df(df: pd.DataFrame) -> List[str]:
    return list(set([expr.split(",")[1] for expr in set(df.columns)]))


@check_df_type_decorator
def get_tickers_from_expression_df(df: pd.DataFrame) -> List[str]:
    return list(set([expr.split(",")[0] for expr in set(df.columns)]))


@check_df_type_decorator
def get_cids_from_expression_df(df: pd.DataFrame) -> List[str]:
    tickers: List[str] = get_tickers_from_expression_df(df)
    return list(set([ticker.split("_")[0] for ticker in tickers]))


@check_df_type_decorator
def get_xcats_from_expression_df(df: pd.DataFrame) -> List[str]:
    tickers: List[str] = get_tickers_from_expression_df(df)
    return list(set([ticker.split("_", 1)[1] for ticker in tickers]))


@check_df_type_decorator
def convert_to_local_expression_df(df: pd.DataFrame) -> pd.DataFrame:
    if check_jpmaqs_expression(df.columns):
        df.columns = jpmaqs_expression_to_local_expression(df.columns)
    else:
        assert check_local_expression(df.columns), "Invalid expression format."
    return df


@check_df_type_decorator
def expression_df_to_qdf(df: pd.DataFrame) -> QuantamentalDataFrame:
    return concat_qdfs(
        [
            ticker_df_to_qdf(
                df=expression_df_to_ticker_df(df.filter(regex=f".*{metric}$")),
                metric=metric,
            )
            for metric in get_metrics_from_expression_df(df)
        ]
    )


def load_wide_df(path: str) -> pd.DataFrame:
    df: pd.DataFrame = pd.read_pickle(path)
    if df.index.name != "real_date":
        if "real_date" not in df.columns:
            raise ValueError(
                "The DataFrame must have a column named "
                "'real_date' or have its index named 'real_date'."
            )
        df = df.set_index("real_date")

    return convert_to_local_expression_df(df)


def get_cid_from_local_expression(expression: str) -> str:
    return expression.split("_")[0]


def get_xcat_from_local_expression(expression: str) -> str:
    return expression.split("_", 1)[1]


class LargeQDF:
    def __init__(
        self,
        qdf: Optional[QuantamentalDataFrame] = None,
        expression_df: Optional[pd.DataFrame] = None,
    ):
        qdfbool, exprbool = qdf is not None, expression_df is not None
        assert (qdfbool ^ exprbool) and (qdfbool or exprbool)
        if qdfbool:
            metrics = set(qdf.columns) - set(QuantamentalDataFrame.IndexCols)
            self.metrics: List[str] = list(metrics)
            self.source_df: pd.DataFrame = qdf_to_expression_df(qdf)

        if exprbool:
            self.metrics = get_metrics_from_expression_df(expression_df)
            self.source_df = expression_df

        all_nan_rows = self.source_df.isnull().all(axis=1)
        bdate_rows = self.source_df.index.isin(
            pd.bdate_range(self.start_date, self.end_date)
        )
        self.source_df = self.source_df.loc[~all_nan_rows & bdate_rows]

    @property
    def start_date(self) -> pd.Timestamp:
        return self.source_df.index.min()

    @property
    def end_date(self) -> pd.Timestamp:
        return self.source_df.index.max()

    def _get_index_date(self, edge: str) -> pd.Series:
        assert edge in ["start", "end"]
        dictx: dict[str, dict[str, pd.Timestamp]] = {}

        def _firstindex(col: pd.Series) -> pd.Timestamp:
            return col.first_valid_index()

        def _lastindex(col: pd.Series) -> pd.Timestamp:
            return col.last_valid_index()

        getfunc = _firstindex if edge == "start" else _lastindex
        for col in self.source_df.columns:
            tkr, mtr = col.split(",")
            if tkr not in dictx:
                dictx[tkr] = {}
            dictx[tkr][mtr] = getfunc(self.source_df[col])

        for tkr in dictx:
            smtr = dictx[tkr].keys()[0]
            stk = dictx[tkr][smtr]
            assert all(stk == dictx[tkr][mtr] for mtr in dictx[tkr])
            dictx[tkr] = stk

        return pd.Series(dictx)

    @property
    def tickers(self) -> pd.Series:
        return pd.Series(self.source_df.columns)

    @property
    def start_dates(self) -> pd.Series:
        return self._get_index_date("start")

    @property
    def end_dates(self) -> pd.Series:
        return self._get_index_date("end")

    @staticmethod
    def load_from_pickle(path: str):
        if "*" in path or "?" in path:
            paths: List[str] = list(map(os.path.normpath, glob.glob(path)))
            if len(paths) > 1:
                path = max(paths, key=os.path.getctime)
            else:
                path = paths[0]

        return LargeQDF(expression_df=load_wide_df(path))

    def save_to_pickle(self, path: str):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.expanduser(path))
        self.source_df.to_pickle(path)

    def info(self) -> pd.Series:
        return pd.Series(
            {
                "start_date": self.start_date.strftime("%Y-%m-%d"),
                "end_date": self.end_date.strftime("%Y-%m-%d"),
                "ticker_count": len(self.tickers),
                "metric_count": len(self.metrics),
                "non_null_count": self.source_df.count().sum(),
                "null_count": self.source_df.isnull().sum().sum(),
                "total_count": self.source_df.size,
                "memory_usage": f"{self.source_df.memory_usage(deep=True).sum() / (1024 ** 2):.2f} MB",
            }
        )

    def __str__(self) -> str:
        return (
            f"LargeQDF(tickers={len(self.tickers)}, "
            f"metrics={len(self.metrics)}, "
            f"start_date={self.start_date}, "
            f"end_date={self.end_date})"
        )

    def __repr__(self) -> str:
        return self.__str__()

    def get(
        self,
        cid: Optional[str] = None,
        cids: Optional[Iterable[str]] = None,
        xcat: Optional[str] = None,
        xcats: Optional[Iterable[str]] = None,
        ticker: Optional[str] = None,
        metric: str = "all",
        tickers: Optional[Iterable[str]] = None,
        metrics: Optional[Union[str, Iterable[str]]] = None,
        start: Optional[DateLike] = None,
        end: Optional[DateLike] = None,
        date_range: Optional[Iterable[DateLike]] = None,
        qdf: bool = True,
    ) -> QuantamentalDataFrame:
        metrics = [metrics] if isinstance(metrics, str) else []
        metrics = metrics + (metrics or [])

        if metrics == ["all"] or not metrics:
            metrics = self.metrics
        start: DateLike = self.start_date if start is None else start
        end: DateLike = self.end_date if end is None else end
        if date_range is not None:
            date_range: pd.DatetimeIndex = pd.DatetimeIndex(
                map(pd.Timestamp, date_range)
            )
            start = date_range.min()
            end = date_range.max()

        if not isinstance(start, pd.Timestamp):
            start = pd.Timestamp(start)
        if not isinstance(end, pd.Timestamp):
            end = pd.Timestamp(end)

        if start < self.start_date:
            start = self.start_date
        if end > self.end_date:
            end = self.end_date

        if start > end:
            start, end = end, start

        valid_local_exprs = get_valid_local_expressions(
            self.source_df,
            cids=cids,
            xcats=xcats,
            tickers=tickers,
            metrics=metrics,
            cid=cid,
            xcat=xcat,
            ticker=ticker,
        )
        if not qdf:
            return self.source_df.loc[start:end, valid_local_exprs]
        else:
            return expression_df_to_qdf(
                self.source_df.loc[start:end, valid_local_exprs]
            )


if __name__ == "__main__":
    start = "2014-01-31"
    end = "2024-01-01"
    import time

    st = time.time()
    lqdf = LargeQDF.load_from_pickle("./data/jpmaqs_snap*.pkl")
    print("--" * 20)
    print(f"Time taken to load LargeQDF: {time.time() - st:.2f} seconds")

    lqdf.info()

    print("--" * 20)
    print(f"Using LargeQDF object: \n\t{lqdf}")
    print("--" * 20)
    print("lQDF.info():")
    print(lqdf.info())

    print("--" * 20)

    print(
        "Looking up CID=[USD, GBP] and XCAT=[CPI*SJA*] for"
        f" the period {start} to {end}."
    )
    print("--" * 20)
    st = time.time()
    df = lqdf.get(
        start=start,
        end=end,
        cids=["USD", "GBP"],
        xcats=["CPI*SJA*"],
        qdf=False,
    )
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    print(f"Time taken: {time.time() - st:.2f} seconds (without QDF conversion)")
    print("--" * 20)
    st = time.time()
    df = lqdf.get(
        start=start,
        end=end,
        cids=["USD", "GBP"],
        xcats=["CPI*SJA*"],
        qdf=True,
    )
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    print(f"Time taken: {time.time() - st:.2f} seconds (with QDF conversion)")
    print("--" * 20)
