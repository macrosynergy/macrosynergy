from typing import List, Optional, Union, TypeVar, overload, Dict, Any
from collections.abc import Iterable, Callable
import pandas as pd
import numpy as np
import datetime
import joblib
import functools
import itertools
import os
import glob
import fnmatch
from macrosynergy.management.types import QuantamentalDataFrame
from tqdm import tqdm
import pickle
from macrosynergy.management.utils import (
    ticker_df_to_qdf,
    qdf_to_ticker_df,
    standardise_dataframe,
    concat_qdfs,
    deconstruct_expression,
    construct_expressions,
    get_cid,
    get_xcat,
    get_ticker,
)
import macrosynergy.management.constants as msy_constants

DateLike = TypeVar("DateLike", str, pd.Timestamp, np.datetime64, datetime.datetime)


def qdf_to_df_dict(qdf: QuantamentalDataFrame) -> dict[str, pd.DataFrame]:
    """
    Convert a `QuantamentalDataFrame` to a dictionary of `pd.DataFrame`s.
    """
    metrics: List[str] = qdf.columns.difference(
        QuantamentalDataFrame.IndexCols
    ).to_list()

    df_dict: dict[str, pd.DataFrame] = {
        metric: qdf_to_ticker_df(qdf, metric) for metric in metrics
    }

    return df_dict


def df_dict_to_qdf(df_dict: dict[str, pd.DataFrame]) -> QuantamentalDataFrame:
    """
    Convert a dictionary of `pd.DataFrame`s to a `QuantamentalDataFrame`.
    """

    def _is_empty(df: pd.DataFrame) -> bool:
        return df is None or (df.empty)

    r = concat_qdfs(
        [
            ticker_df_to_qdf(df, metric)
            for metric, df in df_dict.items()
            if not _is_empty(df)
        ]
    )
    if r is None:
        return pd.DataFrame()

    return r


def expression_df_to_df_dict(
    expression_df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """
    Convert an expression dataframe to a dictionary of `pd.DataFrame`s.
    """
    d_exprs: List[List[str]] = deconstruct_expression(expression_df.columns.to_list())
    de_df = pd.DataFrame(d_exprs, columns=["cid", "xcat", "metric"])

    unique_metrics: List[str] = list(set(d_e[-1] for d_e in d_exprs))

    df_dict: dict[str, pd.DataFrame] = {
        metric: expression_df[expression_df.columns[de_df["metric"] == metric]]
        for metric in unique_metrics
    }
    for metric in unique_metrics:
        df_dict[metric].columns = [get_ticker(col) for col in df_dict[metric].columns]

    return df_dict


def get_ticker_dict_from_df_dict(
    df_dict: dict[str, pd.DataFrame]
) -> dict[str, List[str]]:
    """
    Get a dictionary of tickers from a dictionary of `pd.DataFrame`s.
    """
    return {metric: df.columns.to_list() for metric, df in df_dict.items()}


def get_tickers_from_df_dict(
    df_dict: dict[str, pd.DataFrame], common_metrics: bool = True
) -> List[str]:
    """
    Get the tickers from a dictionary of `pd.DataFrame`s.
    """
    ticker_dict: Dict[str, List[str]] = get_ticker_dict_from_df_dict(df_dict)

    if common_metrics:
        tickers: List[str] = list(set.intersection(*map(set, ticker_dict.values())))
    else:
        tickers: List[str] = list(
            set(itertools.chain.from_iterable(ticker_dict.values()))
        )

    return sorted(tickers)


def get_date_range_from_df_dict(df_dict: dict[str, pd.DataFrame]) -> pd.DatetimeIndex:
    """
    Get the date range from a dictionary of `pd.DataFrame`s.
    """
    dts = {metric: set(df.index) for metric, df in df_dict.items()}
    return pd.DatetimeIndex(sorted(set.union(*dts.values())))


class Loader:
    @staticmethod
    def load_single_csv_from_disk(csv_file: str) -> pd.DataFrame:
        """
        Load a single csv file from disk.
        """
        return pd.read_csv(csv_file, parse_dates=["real_date"]).set_index("real_date")

    @staticmethod
    def load_single_qdf_from_disk(csv_file: str) -> QuantamentalDataFrame:
        """
        Load a single qdf file from disk.
        """
        ticker = os.path.basename(csv_file).split(".")[0]
        return pd.read_csv(csv_file, parse_dates=["real_date"]).assign(
            cid=get_cid(ticker), xcat=get_xcat(ticker)
        )

    @staticmethod
    def get_csv_files(path: str) -> List[str]:
        """
        Load all the csv files in the path recursively.
        """
        listx = sorted(glob.glob(path + "/**/*.csv", recursive=True))
        if not listx:
            raise FileNotFoundError(f"No CSV files found in {path}")
        return listx

    @staticmethod
    def load_csv_batch_from_disk(
        csv_files: List[str],
    ) -> pd.DataFrame:
        """
        Load a batch of csv files from disk, only for CSV (non-qdf) files.
        """
        return pd.concat(
            joblib.Parallel()(
                joblib.delayed(Loader.load_single_csv_from_disk)(csv_file)
                for csv_file in tqdm(csv_files, desc="Loading CSVs", leave=False)
            ),
            axis=1,
        )

    @staticmethod
    def load_qdf_batch_from_disk(
        csv_files: List[str],
    ) -> Dict[str, pd.DataFrame]:
        """
        Load a batch of qdf files from disk.
        """
        return qdf_to_df_dict(
            concat_qdfs(
                joblib.Parallel()(
                    joblib.delayed(Loader.load_single_qdf_from_disk)(csv_file)
                    for csv_file in tqdm(csv_files, desc="Loading QDFs", leave=False)
                )
            )
        )

    @staticmethod
    def load_csvs_from_disk(
        path: str,
        show_progress: bool = True,
        batch_size: int = 100,
    ) -> pd.DataFrame:
        """
        All the csv files in the path recursively and try to load them as time series data.
        """
        csvs_list = Loader.get_csv_files(path)
        csv_batches = [
            csvs_list[i : i + batch_size] for i in range(0, len(csvs_list), batch_size)
        ]

        dfs = joblib.Parallel(n_jobs=-1)(
            joblib.delayed(Loader.load_csv_batch_from_disk)(csv_batch)
            for csv_batch in tqdm(
                csv_batches, disable=not show_progress, desc="Loading CSVs"
            )
        )

        return pd.concat(dfs, axis=1)

    @staticmethod
    def load_csv_from_disk_as_df_dict(
        path: str,
        show_progress: bool = True,
        batch_size: int = 100,
    ) -> Dict[str, pd.DataFrame]:
        """
        Load all the csv files in the path recursively.
        """
        return expression_df_to_df_dict(
            Loader.load_csvs_from_disk(
                path=path,
                show_progress=show_progress,
                batch_size=batch_size,
            )
        )

    @staticmethod
    def load_qdfs_from_disk_as_df_dict(
        path: str,
        show_progress: bool = True,
        batch_size: int = 100,
    ) -> Dict[str, pd.DataFrame]:
        """
        Load all the qdf files in the path recursively.
        """
        csvs_list = Loader.get_csv_files(path)
        csv_batches = [
            csvs_list[i : i + batch_size] for i in range(0, len(csvs_list), batch_size)
        ]

        df_dicts: List[Dict[str, pd.DataFrame]] = joblib.Parallel(n_jobs=-1)(
            joblib.delayed(Loader.load_qdf_batch_from_disk)(csv_batch)
            for csv_batch in tqdm(
                csv_batches, disable=not show_progress, desc="Loading QDFs"
            )
        )
        df_dict: Dict[str, pd.DataFrame] = {}
        for _ in range(len(df_dicts)):
            d = df_dicts.pop()
            for k, v in d.items():
                if k in df_dict:
                    df_dict[k] = pd.concat([df_dict[k], v], axis=1)
                else:
                    df_dict[k] = v

        for metric in df_dict.keys():
            df_dict[metric] = df_dict[metric][sorted(df_dict[metric].columns)]

        return df_dict

    @staticmethod
    def load_pkl_from_disk(path: str) -> Any:
        """
        Load a pickle file from disk.
        """
        with open(path, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def load_pkl(path: str) -> Any:
        """
        Load a pickle file from disk.
        """
        obj = Loader.load_pkl_from_disk(path)
        if isinstance(obj, pd.DataFrame):
            return expression_df_to_df_dict(obj)
        elif isinstance(obj, dict):
            return obj
        else:
            raise NotImplementedError(
                f"Invalid object type: {type(obj)}. Must be either `pd.DataFrame` or "
                "`Dict[str, pd.DataFrame]`"
            )

    @staticmethod
    def load_from_disk(
        path: str,
        format: str = "csvs",
    ) -> Dict[str, pd.DataFrame]:
        """
        Load a `QuantamentalDataFrame` from disk.

        Parameters
        :param <str> path: The path to the directory containing the data.
        :param <str> format: The format of the data. Options are "csvs", "csv",
            "pkl", or "qdfs".
        """

        def load_single_csv_from_disk_as_df_dict(
            csv_file: str,
        ) -> Dict[str, pd.DataFrame]:
            """
            Load a single csv file from disk.
            """
            return expression_df_to_df_dict(Loader.load_single_csv_from_disk(csv_file))

        if format == "pkl":
            return Loader.load_pkl(path)

        fmt_dict = {
            "csv": load_single_csv_from_disk_as_df_dict,
            "csvs": Loader.load_csv_from_disk_as_df_dict,
            "qdfs": Loader.load_qdfs_from_disk_as_df_dict,
        }
        if format not in fmt_dict:
            raise NotImplementedError(
                f"Invalid format: {format}. Options are {fmt_dict.keys()}"
            )
        return fmt_dict[format](path)


class Saver:
    @staticmethod
    def save_single_csv_to_disk(df: pd.DataFrame, path: str) -> None:
        """
        Save a single csv file to disk.
        """
        # reset index, and ensure the real_date is a column
        if not "real_date" in df.columns:
            df = df.reset_index()

        assert "real_date" in df.columns, "real_date column not found in the dataframe"
        assert len(df.columns) >= 2, "At least one ticker column is required"
        df.to_csv(path, index=False)

    @staticmethod
    def save_single_qdf_to_disk(
        df: QuantamentalDataFrame, path: str, infer_filename: bool = True
    ) -> None:
        """
        Save a single qdf file to disk.
        """
        # reset index, and ensure the real_date is a column
        if not "real_date" in df.columns:
            df = df.reset_index()

        assert "real_date" in df.columns, "real_date column not found in the dataframe"
        cidx = df["cid"].unique()
        xcatx = df["xcat"].unique()
        assert len(cidx) == 1, f"Multiple cids found: {cidx}"
        assert len(xcatx) == 1, f"Multiple xcats found: {xcatx}"
        if infer_filename:
            # check if the base path is a directory
            path = path if os.path.isdir(path) else os.path.dirname(path)
            path = os.path.join(path, f"{cidx[0]}_{xcatx[0]}.csv")

        df.to_csv(path, index=False)

    @staticmethod
    def save_pkl_to_disk(obj: Any, path: str) -> None:
        """
        Save an object to disk.
        """
        with open(path, "wb") as f:
            pickle.dump(obj, f)


def _run_ticker_query(
    qdf_manager: "QDFManager",
    cids: Optional[Iterable[str]] = None,
    xcats: Optional[Iterable[str]] = None,
    tickers: Optional[Iterable[str]] = None,
    metrics: Optional[Union[str, Iterable[str]]] = ["all"],
    cid: Optional[str] = None,
    xcat: Optional[str] = None,
    ticker: Optional[str] = None,
) -> List[str]:
    """
    Get the tickers from the query parameters.
    """
    all_available_tickers = qdf_manager.tickers
    cids = [cids] if isinstance(cids, str) else cids
    xcats = [xcats] if isinstance(xcats, str) else xcats
    if isinstance(cid, str):
        cids = [cid] + (cids or [])
    if isinstance(xcat, str):
        xcats = [xcat] + (xcats or [])
    if isinstance(ticker, str):
        tickers = [ticker] + (tickers or [])
    if tickers is None:
        if cids is None or cids == []:
            cids = qdf_manager.cids
        if xcats is None or xcats == []:
            xcats = qdf_manager.xcats
        tickers = [f"{cid}_{xcat}" for cid in cids for xcat in xcats]

    _contains_wildcard = lambda x: "*" in x or "?" in x
    wtickers = list(filter(_contains_wildcard, tickers))
    if wtickers:
        tickers = list(filter(lambda x: not _contains_wildcard(x), tickers))
        new_tickers = []
        for wtkr in wtickers:
            new_tickers.extend(fnmatch.filter(all_available_tickers, wtkr))
        tickers = list(set(tickers + new_tickers))

    metrics = list(set(metrics) & set(qdf_manager.metrics))
    if cids is None or xcats is None:
        cids: List[str] = []
        xcats: List[str] = []

    if tickers is None:
        tickers: List[str] = []

    tickers += [f"{cid}_{xcat}" for cid in cids for xcat in xcats]
    tickers = set(tickers)

    return {
        m: sorted(set(qdf_manager.df_dict[m].columns).intersection(tickers))
        for m in metrics
    }


def query_df_dict(
    qdf_manager: "QDFManager",
    cid: Optional[str] = None,
    cids: Optional[List[str]] = None,
    xcat: Optional[str] = None,
    xcats: Optional[List[str]] = None,
    ticker: Optional[str] = None,
    tickers: Optional[List[str]] = None,
    start: Optional[DateLike] = None,
    end: Optional[DateLike] = None,
    date_range: Optional[pd.DatetimeIndex] = None,
    # substring: Optional[str] = None,
    # substrings: Optional[List[str]] = None,
    metric: Optional[str] = None,
    metrics: Optional[List[str]] = None,
    cross_section_groups: Optional[List[str]] = None,
) -> dict[str, pd.DataFrame]:
    """
    Query a dictionary of `pd.DataFrame`s.
    """
    metrics = [metric] if isinstance(metric, str) else []
    metrics = [metrics] if isinstance(metrics, str) else []
    metrics = metrics + (metrics or [])

    if metrics == ["all"] or not metrics:
        metrics = qdf_manager.metrics
    start: DateLike = qdf_manager.start_date if start is None else start
    end: DateLike = qdf_manager.end_date if end is None else end
    if date_range is not None:
        date_range: pd.DatetimeIndex = pd.DatetimeIndex(map(pd.Timestamp, date_range))
        start = date_range.min()
        end = date_range.max()

    if not isinstance(start, pd.Timestamp):
        start = pd.Timestamp(start)
    if not isinstance(end, pd.Timestamp):
        end = pd.Timestamp(end)

    if start < qdf_manager.start_date:
        start = qdf_manager.start_date
    if end > qdf_manager.end_date:
        end = qdf_manager.end_date

    if start > end:
        start, end = end, start

    if isinstance(cross_section_groups, str):
        cross_section_groups = [cross_section_groups]
    cross_section_groups = cross_section_groups or []
    if cids is None:
        cids = []
    for cxg in cross_section_groups or []:
        _cx = cxg.lower()
        if _cx in msy_constants.cross_section_groups:
            cids = list(set(cids + msy_constants.cross_section_groups[_cx]))

    ticker_query: Dict[str, List[str]] = _run_ticker_query(
        qdf_manager,
        cid=cid,
        cids=cids,
        xcat=xcat,
        xcats=xcats,
        ticker=ticker,
        tickers=tickers,
        metrics=metrics,
    )

    df_dict: dict[str, pd.DataFrame] = {
        metric: qdf_manager.df_dict[metric].loc[start:end, tickers]
        for metric, tickers in ticker_query.items()
    }
    return df_dict


class QDFManager:
    """
    A class to manage a large `QuantamentalDataFrame` object.
    """

    def __init__(
        self,
        qdf: Optional[QuantamentalDataFrame] = None,
        expression_df: Optional[pd.DataFrame] = None,
        df_dict: Optional[dict[str, pd.DataFrame]] = None,
    ) -> None:
        """
        Initialise the `QDFManager` object.
        """
        if expression_df is not None:
            self.df_dict = expression_df_to_df_dict(expression_df)
        elif qdf is not None:
            self.df_dict = qdf_to_df_dict(qdf)
        elif df_dict is not None:
            self.df_dict = df_dict
        else:
            raise ValueError(
                "Either `qdf`, `expression_df`, or `df_dict` must be provided."
            )

        self._tickers = get_tickers_from_df_dict(self.df_dict, common_metrics=True)
        self.cids = sorted(set(get_cid(self._tickers)))
        self.xcats = sorted(set(get_xcat(self._tickers)))
        self.metrics = sorted(self.df_dict.keys())
        self.date_range = get_date_range_from_df_dict(self.df_dict)
        self.start_date: pd.Timestamp = self.date_range.min()
        self.end_date: pd.Timestamp = self.date_range.max()

    @property
    def tickers(self) -> List[str]:
        """
        Return the tickers in the `QuantamentalDataFrame`.
        """
        return self._tickers

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback): ...

    @staticmethod
    def load_from_disk(
        path: str,
        format: str = "csvs",
    ) -> "QDFManager":
        """
        Load a `QuantamentalDataFrame` from disk.

        Parameters
        :param <str> path: The path to the directory containing the data.
        :param <str> format: The format of the data. Options are "csvs",
        """
        return QDFManager(df_dict=Loader.load_from_disk(path, format=format))

    def save_to_disk(
        self,
        path: str,
        format: str = "pkl",
    ) -> None:
        """
        Save the `QuantamentalDataFrame` to disk.

        Parameters
        :param <str> path: The path to the directory to save the data.
        :param <str> format: The format of the data. Options are "csvs",
        """
        if format == "pkl":
            Saver.save_pkl_to_disk(self.df_dict, path)
        else:
            for metric, df in self.df_dict.items():
                Saver.save_single_csv_to_disk(df, os.path.join(path, f"{metric}.csv"))

    def _get_dict(
        self,
        *args,
        **kwargs,
        # cid: Optional[str] = None,
        # cids: Optional[List[str]] = None,
        # xcat: Optional[str] = None,
        # xcats: Optional[List[str]] = None,
        # ticker: Optional[str] = None,
        # tickers: Optional[List[str]] = None,
        # start: Optional[DateLike] = None,
        # end: Optional[DateLike] = None,
        # metric: Optional[str] = None,
        # metrics: Optional[List[str]] = None,
        # cross_section_groups: Optional[List[str]] = None,
    ) -> dict[str, pd.DataFrame]:
        """
        Query the `QuantamentalDataFrame`.
        """
        return query_df_dict(*args, **kwargs, qdf_manager=self)

    def qdf(
        self,
        cid: Optional[str] = None,
        cids: Optional[List[str]] = None,
        xcat: Optional[str] = None,
        xcats: Optional[List[str]] = None,
        ticker: Optional[str] = None,
        tickers: Optional[List[str]] = None,
        start: Optional[DateLike] = None,
        end: Optional[DateLike] = None,
        metrics: Optional[str] = "value",
        cross_section_groups: Optional[List[str]] = None,
    ) -> QuantamentalDataFrame:
        """
        Query the `QuantamentalDataFrame`.
        """
        return df_dict_to_qdf(
            self._get_dict(
                cid=cid,
                cids=cids,
                xcat=xcat,
                xcats=xcats,
                ticker=ticker,
                tickers=tickers,
                start=start,
                end=end,
                metrics=metrics,
                cross_section_groups=cross_section_groups,
            )
        )

    def ticker_df(
        self,
        cid: Optional[str] = None,
        cids: Optional[List[str]] = None,
        xcat: Optional[str] = None,
        xcats: Optional[List[str]] = None,
        ticker: Optional[str] = None,
        tickers: Optional[List[str]] = None,
        start: Optional[DateLike] = None,
        end: Optional[DateLike] = None,
        metric: str = "value",
        cross_section_groups: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Query the `QuantamentalDataFrame`.
        """
        if metric not in self.metrics:
            raise ValueError(f"Invalid metric: {metric}. Options are {self.metrics}")
        return self._get_dict(
            cid=cid,
            cids=cids,
            xcat=xcat,
            xcats=xcats,
            ticker=ticker,
            tickers=tickers,
            start=start,
            end=end,
            metric=metric,
            cross_section_groups=cross_section_groups,
        )[metric]


# qdfman: QDFManager = QDFManager.load_from_disk(
#     path=r"E:\datasets\jpmaqs\QDFs",
#     format="qdfs",
# )

# qdfman.qdf(cid="USD*")
