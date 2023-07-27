import pandas as pd
from typing import List, Dict, Union, Tuple
from types import ModuleType
from collections.abc import Callable, Iterable
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from macrosynergy.management.utils import standardise_dataframe
from macrosynergy.management import reduce_df

logger = logging.getLogger(__name__)


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

    def __init__(
        self,
        df: pd.DataFrame,
        cids: List[str] = None,
        xcats: List[str] = None,
        metrics: List[str] = None,
        tickers: List[str] = None,
        blacklist: Dict[str, List[str]] = None,
        start_date: str = None,
        end_date: str = None,
        backend: str = "matplotlib",
    ):
        sdf: pd.DataFrame = df.copy()
        df_cols: List[str] = ["real_date", "cid", "xcat"]
        df_cols += metrics
        sdf = sdf[df_cols]
        if cids:
            sdf = sdf[sdf["cid"].isin(cids)]
        if xcats:
            sdf = sdf[sdf["xcat"].isin(xcats)]
        if tickers:
            sdf = sdf[(df["cid"] + "_" + df["xcat"]).isin(tickers)]
        if start_date:
            sdf = sdf[sdf["real_date"] >= pd.to_datetime(start_date)]
        if end_date:
            sdf = sdf[sdf["real_date"] <= pd.to_datetime(end_date)]

        if blacklist:
            sdf: pd.DataFrame = reduce_df(df=sdf, blacklist=blacklist)

        self.df: pd.DataFrame = sdf

        if backend[0] == "m":
            self.backend: ModuleType = plt
            self.backend.style.use("seaborn-v0_8-darkgrid")
        elif backend[0] == "s":
            self.backend: ModuleType = sns

        else:
            raise NotImplementedError(f"Backend {backend} is not supported.")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass
