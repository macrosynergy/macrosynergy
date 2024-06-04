from typing import List, Optional, Union, TypeVar, overload, Dict
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
from macrosynergy.management.utils import (
    ticker_df_to_qdf,
    qdf_to_ticker_df,
    standardise_dataframe,
    concat_qdfs,
    deconstruct_expression,
    get_cid,
    get_xcat,
    get_ticker,
)

DateLike = TypeVar("DateLike", str, pd.Timestamp, np.datetime64, datetime.datetime)


def qdf_to_df_dict(qdf: QuantamentalDataFrame) -> dict[str, pd.DataFrame]:
    """
    Convert a `QuantamentalDataFrame` to a dictionary of `pd.DataFrame`s.
    """
    metrics: List[str] = qdf.columns.difference(
        QuantamentalDataFrame.IndexCols
    ).to_list()

    df_dict: dict[str, pd.DataFrame] = {
        qdf_to_ticker_df(qdf, metric) for metric in metrics
    }

    return df_dict


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
        return (
            pd.read_csv(csv_file, parse_dates=["real_date"])
            .set_index("real_date")
            .assign(cid=get_cid(ticker), xcat=get_xcat(ticker))
        )

    @staticmethod
    def get_csv_files(path: str) -> List[str]:
        """
        Load all the csv files in the path recursively.
        """
        return sorted(glob.glob(path + "/**/*.csv", recursive=True))

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
    def load_qdfs_from_disk(
        path: str,
        show_progress: bool = True,
        batch_size: int = 100,
    ) -> QuantamentalDataFrame:
        """
        Load all the qdf files in the path recursively.
        """
        csvs_list = Loader.get_csv_files(path)[:1000]
        csv_batches = [
            csvs_list[i : i + batch_size] for i in range(0, len(csvs_list), batch_size)
        ]

        # qdfs = joblib.Parallel(n_jobs=-1)(
        #     joblib.delayed(Loader.load_qdf_batch_from_disk)(csv_batch)
        #     for csv_batch in tqdm(
        #         csv_batches, disable=not show_progress, desc="Loading QDFs"
        #     )
        # )

        # return concat_qdfs(qdfs)

    def load_from_disk(
        path: str,
        format: str = "csvs",
    ) -> QuantamentalDataFrame:
        """
        Load a `QuantamentalDataFrame` from disk.

        Parameters
        :param <str> path: The path to the directory containing the data.
        :param <str> format: The format of the data. Options are "csvs",
        """
        fmt_dict = {
            "csv": Loader.load_single_csv_from_disk,
            "csvs": Loader.load_csvs_from_disk,
            "qdfs": Loader.load_qdfs_from_disk,
        }
        if format not in fmt_dict:
            raise ValueError(f"Invalid format: {format}. Options are {fmt_dict.keys()}")
        return fmt_dict[format](path)


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


d = expression_df_to_df_dict(Loader.load_from_disk(r"E:\datasets\jpmaqs\QDFs", "qdfs"))
d.keys()


class QDFManager:
    """
    A class to manage a large `QuantamentalDataFrame` object.
    """

    def __init__(
        self,
        qdf: QuantamentalDataFrame,
        expression_df: Optional[pd.DataFrame] = None,
    ) -> None:
        """
        Initialise the `QDFManager` object.
        """
        if expression_df is not None:
            self.df_dict = expression_df_to_df_dict(expression_df)
        elif qdf is not None:
            self.df_dict = qdf_to_df_dict(qdf)
        else:
            raise ValueError("Either `qdf` or `expression_df` must be provided.")

    @property
    def metrics(self) -> List[str]:
        """
        Return the metrics in the `QuantamentalDataFrame`.
        """
        return list(self.df_dict.keys())

    def load_from_disk(
        path: str,
        format: str = "csvs",
    ) -> QuantamentalDataFrame:
        """
        Load a `QuantamentalDataFrame` from disk.

        Parameters
        :param <str> path: The path to the directory containing the data.
        :param <str> format: The format of the data. Options are "csvs",
        """
        fmt_dict = {
            "csv": Loader.load_single_csv_from_disk,
            "csvs": Loader.load_csvs_from_disk,
            "qdfs": Loader.load_qdfs_from_disk,
        }
        if format not in fmt_dict:
            raise ValueError(f"Invalid format: {format}. Options are {fmt_dict.keys()}")
        return fmt_dict[format](path)
