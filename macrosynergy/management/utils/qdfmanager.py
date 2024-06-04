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
)

DateLike = TypeVar("DateLike", str, pd.Timestamp, np.datetime64, datetime.datetime)


def _load_single_csv_from_disk(csv_file: str) -> pd.DataFrame:
    """
    Load a single csv file from disk.
    """
    df = pd.read_csv(csv_file, parse_dates=["real_date"]).set_index("real_date")
    return df


# def _thread_load_csvs_from_disk(
#     csvs_list: List[str],
#     show_progress: bool = True,
# ) -> List[pd.DataFrame]:
#     """
#     Load a list of csv files from disk using a thread pool.
#     """
#     with concurrent.futures.ThreadPoolExecutor() as executor:
#         dfs = list(executor.map(_load_single_csv_from_disk))

#     return dfs


def _load_csv_files(path: str) -> List[str]:
    """
    Load all the csv files in the path recursively.
    """
    return sorted(glob.glob(path + "/**/*.csv", recursive=True))


def _load_csvs_from_disk(
    path: str,
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    All the csv files in the path recursively and try to load them as time series data.
    """
    csvs_list = _load_csv_files(path)

    dfs = joblib.Parallel(n_jobs=-1)(
        joblib.delayed(_load_single_csv_from_disk)(csv_file)
        for csv_file in tqdm(csvs_list, disable=not show_progress, desc="Loading CSVs")
    )

    return pd.concat(dfs, axis=1)


def load_from_disk(
    path: str,
    format: str = "csv",
) -> QuantamentalDataFrame:
    """
    Load a `QuantamentalDataFrame` from disk.
    """
    if format == "csv":
        df = _load_csvs_from_disk(path)
    else:
        raise ValueError(f"Unknown format: {format}")

    return df


def qdf_to_df_dict(qdf: QuantamentalDataFrame) -> dict[str, pd.DataFrame]:
    """
    Convert a `QuantamentalDataFrame` to a dictionary of `pd.DataFrame`s.
    """
    metrics: List[str] = qdf.columns.difference(["real_date", "cid", "xcat"])

    df_dict: dict[str, pd.DataFrame] = {
        qdf_to_ticker_df(qdf, metric)
        for metric in qdf.columns.difference(["real_date", "cid", "xcat"])
    }

    return df_dict


def expression_df_to_df_dict(
    expression_df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """
    Convert an expression dataframe to a dictionary of `pd.DataFrame`s.
    """
    d_exprs: List[List[str]] = deconstruct_expression(expression_df.columns)
    de_df = pd.DataFrame(d_exprs, columns=["cid", "xcat", "metric"])

    metrics: List[str] = list(set(d_e[-1] for d_e in d_exprs))


df = load_from_disk(r"C:\Users\PalashTyagi\Code\ms_copy\dataset\JPMaQSData")
df.head()
