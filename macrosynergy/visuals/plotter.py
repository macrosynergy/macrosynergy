import inspect
import logging
import warnings
from collections.abc import Callable, Iterable
from functools import wraps
from types import ModuleType
from typing import Any, Callable, Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from macrosynergy.management import reduce_df, reduce_df_by_ticker
from macrosynergy.management.utils import standardise_dataframe

logger = logging.getLogger(__name__)


def argvalidation(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args, **kwargs):
        func_sig: inspect.Signature = inspect.signature(func)
        func_params: Dict[str, inspect.Parameter] = func_sig.parameters
        func_annotations: Dict[str, Any] = func_sig.return_annotation
        func_args: Dict[str, Any] = inspect.getcallargs(func, *args, **kwargs)

        # validate the arguments
        for arg_name, arg_value in func_args.items():
            if arg_name in func_params:
                arg_type: Any = func_params[arg_name].annotation
                if arg_type is not inspect._empty:
                    if not isinstance(arg_value, arg_type.__origin__):
                        raise TypeError(
                            f"Argument `{arg_name}` must be of type `{arg_type}`."
                        )

        # validate the return value
        return_value: Any = func(*args, **kwargs)
        if func_annotations is not inspect._empty:
            if not isinstance(return_value, func_annotations):
                warnings.warn(
                    f"Return value of `{func.__name__}` is not of type "
                    f"`{func_annotations}`, but of type `{type(return_value)}`."
                )

        return return_value

    return wrapper


def argcopy(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args, **kwargs):
        copy_types = (
            list,
            dict,
            pd.DataFrame,
            np.ndarray,
            pd.Series,
            pd.Index,
            pd.MultiIndex,
            set,
            tuple,
        )
        new_args: List[Tuple[Any, ...]] = []
        for arg in args:
            if isinstance(arg, copy_types) or issubclass(type(arg), copy_types):
                new_args.append(arg.copy())
            else:
                new_args.append(arg)
        new_kwargs: Dict[str, Any] = {}
        for key, value in kwargs.items():
            if isinstance(value, copy_types) or issubclass(type(value), copy_types):
                new_kwargs[key] = value.copy()
            else:
                new_kwargs[key] = value

        return func(*new_args, **new_kwargs)

    return wrapper


class Plotter(object):
    """
    Base class for a DataFrame Plotter.
    It provides a shared interface for the plotter classes,
    and some common functionality - currently just the filtering
    of the DataFrame.

    Parameters
    :param <pd.DataFrame> df: A DataFrame with the following columns:
        'cid', 'xcat', 'real_date', and at least one metric from -
        'value', 'grading', 'eop_lag', or 'mop_lag'.
    :param <List[str]> cids: A list of cids to select from the DataFrame
        (self.df). If None, all cids are selected.
    :param <List[str]> xcats: A list of xcats to select from the DataFrame
        (self.df). If None, all xcats are selected.F
    :param <List[str]> metrics: A list of metrics to select from the DataFrame
        (self.df). If None, all metrics are selected.
    :param <str> start_date: ISO-8601 formatted date. Select data from
        this date onwards. If None, all dates are selected.
    :param <str> end_date: ISO-8601 formatted date. Select data up to
        and including this date. If None, all dates are selected.
    :param <str> backend: The plotting backend to use. Currently only
        'matplotlib' and 'seaborn' are supported, with 'matplotlib' as
        the default.
    """

    @argvalidation
    @argcopy
    def __init__(
        self,
        df: pd.DataFrame,
        cids: List[str] = None,
        xcats: List[str] = None,
        metrics: List[str] = None,
        intersect: bool = False,
        tickers: List[str] = None,
        blacklist: Dict[str, List[str]] = None,
        start: str = None,
        end: str = None,
        backend: str = "matplotlib",
    ):
        sdf: pd.DataFrame = df.copy()
        df_cols: List[str] = ["real_date", "cid", "xcat"]
        if metrics is None:
            metrics: List[str] = list(set(sdf.columns) - set(df_cols))

        df_cols += metrics
        sdf = standardise_dataframe(df=sdf)
        if not set(df_cols).issubset(set(sdf.columns)):
            raise ValueError(f"DataFrame must contain the following columns: {df_cols}")

        if cids is None:
            cids = list(sdf["cid"].unique())
        if xcats is None:
            xcats = list(sdf["xcat"].unique())

        if start is None:
            start: str = pd.Timestamp(sdf["real_date"].min()).strftime("%Y-%m-%d")
        if end is None:
            end: str = pd.Timestamp(sdf["real_date"].max()).strftime("%Y-%m-%d")

        if tickers is not None:
            sdf = reduce_df_by_ticker(
                df=sdf,
                tickers=tickers,
            )

        sdf: pd.DataFrame
        cids: List[str]
        xcats: List[str]
        sdf, xcats, cids = reduce_df(
            df=sdf,
            cids=cids if isinstance(cids, list) else [cids],
            xcats=xcats if isinstance(xcats, list) else [xcats],
            intersect=intersect,
            start=start,
            end=end,
            blacklist=blacklist,
            out_all=True,
        )

        self.df: pd.DataFrame = sdf
        self.cids: List[str] = cids
        self.xcats: List[str] = xcats
        self.metrics: List[str] = metrics
        self.intersect: bool = intersect
        self.tickers: List[str] = tickers
        self.blacklist: Dict[str, List[str]] = blacklist
        self.start: str = start
        self.end: str = end

        self.backend: ModuleType
        if backend.startswith("m"):
            self.backend = plt
            self.backend.style.use("seaborn-v0_8-darkgrid")
        elif backend.startswith("s"):
            self.backend = sns
            self.backend.set_style("darkgrid")

        else:
            raise NotImplementedError(f"Backend `{backend}` is not supported.")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass
