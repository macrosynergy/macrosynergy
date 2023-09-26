"""
Module containing code for the Plotter class, and all common functionality
for the plotter classes. The Plotter class is a base class for all plotter
classes, and provides a shared interface for dataframe filtering operations,
as well as `argvalidation` and `argcopy` decorators for all methods of the
plotter classes.
"""
import logging
import warnings
from types import ModuleType
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd

from macrosynergy.management import reduce_df
from macrosynergy.management.utils import standardise_dataframe
from macrosynergy.visuals.common import argcopy, argvalidation

logger = logging.getLogger(__name__)


class PlotterMetaClass(type):
    """
    Metaclass for the Plotter class. The purpose of this metaclass is to
    wrap all methods of the Plotter class with the `argvalidation` and
    `argcopy` decorators, so that all methods of the Plotter class are
    automatically validated and copied.
    Meant to be used as a metaclass, i.e. use as follows:
    ```python
    ...
    class MyCustomClass(metaclass=PlotterMetaClass):
        def __init__(self, ...):
        ...
    ```
    """

    def __init__(cls, name, bases, dct: Dict[str, Any]):
        super().__init__(name, bases, dct)
        for attr_name, attr_value in dct.items():
            if callable(attr_value):
                setattr(cls, attr_name, argcopy(argvalidation(attr_value)))


class Plotter(metaclass=PlotterMetaClass):
    """
    Base class for a DataFrame Plotter. The inherited meta class automatically wraps all
    methods of the Plotter class and any subclasses with the `argvalidation` and `argcopy`
    decorators, so that all methods of the class are automatically validated and copied.
    This class does not implement any plotting functionality, but provides a shared interface
    for the plotter classes, and some common functionality - currently just the filtering
    of the DataFrame.
    Parameters
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
    :param <List[str]> tickers: A list of tickers to select from the DataFrame.
        If None, all tickers are selected.
    :param <dict> blacklist: cross-sections with date ranges that should be excluded from
        the data frame. If one cross-section has several blacklist periods append numbers
        to the cross-section code.
    :param <str> start: ISO-8601 formatted date string. Select data from
        this date onwards. If None, all dates are selected.
    :param <str> end: ISO-8601 formatted date string. Select data up to
        and including this date. If None, all dates are selected.
    :param <str> backend: The plotting backend to use. Currently only
        'matplotlib' is supported.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        cids: Optional[List[str]] = None,
        xcats: Optional[List[str]] = None,
        metrics: Optional[List[str]] = None,
        intersect: Optional[bool] = False,
        tickers: Optional[List[str]] = None,
        blacklist: Optional[Dict[str, List[str]]] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        backend: Optional[str] = "matplotlib",
    ):
        sdf: pd.DataFrame = df.copy()
        df_cols: List[str] = ["real_date", "cid", "xcat"]
        if metrics is None:
            metrics: List[str] = list(set(sdf.columns) - set(df_cols))

        df_cols += metrics
        sdf = standardise_dataframe(df=sdf)
        if not set(df_cols).issubset(set(sdf.columns)):
            raise ValueError(f"DataFrame must contain the following columns: {df_cols}")

        cids_provided: bool = cids is not None
        xcats_provided: bool = xcats is not None
        if cids is None:
            cids = list(sdf["cid"].unique())
        if xcats is None:
            xcats = list(sdf["xcat"].unique())

        missing_cids: List[str] = []
        missing_xcats: List[str] = []
        for varx, prov_bool, namex in zip(
            [cids, xcats], [cids_provided, xcats_provided], ["cids", "xcats"]
        ):
            if prov_bool:
                df_col = namex.replace("s", "")
                if not set(varx).issubset(set(sdf[df_col].unique())):
                    # warn
                    warnings.warn(
                        f"The following {namex.upper()}, passed in `{namex}`,"
                        " are not in the DataFrame `df`: "
                        f"{list(set(varx) - set(sdf[df_col].unique()))}."
                    )
                    if namex == "cids":
                        missing_cids += list(set(varx) - set(sdf[df_col].unique()))
                    else:
                        missing_xcats += list(set(varx) - set(sdf[df_col].unique()))

        if start is None:
            start: str = pd.Timestamp(sdf["real_date"].min()).strftime("%Y-%m-%d")
        if end is None:
            end: str = pd.Timestamp(sdf["real_date"].max()).strftime("%Y-%m-%d")

        ticker_df: pd.DataFrame = pd.DataFrame()
        if tickers is not None:
            df_tickers: List[pd.DataFrame] = [pd.DataFrame()]
            for ticker in tickers:
                _cid, _xcat = ticker.split("_", 1)
                df_tickers.append(
                    sdf.loc[
                        (sdf["cid"] == _cid) & (sdf["xcat"] == _xcat),
                        ["real_date", "cid", "xcat"] + metrics,
                    ]
                )
            ticker_df: pd.DataFrame = pd.concat(df_tickers, axis=0)

        sdf: pd.DataFrame
        r_xcats: List[str]
        r_cids: List[str]
        sdf, r_xcats, r_cids = reduce_df(
            df=sdf,
            cids=cids if isinstance(cids, list) else [cids],
            xcats=xcats if isinstance(xcats, list) else [xcats],
            intersect=intersect,
            start=start,
            end=end,
            blacklist=blacklist,
            out_all=True,
        )

        sdf: pd.DataFrame = pd.concat([sdf, ticker_df], axis=0)

        if (
            ((len(r_xcats) != len(xcats) - len(missing_xcats)) and xcats_provided)
            or ((len(r_cids) != len(cids) - len(missing_cids)) and cids_provided)
        ) and not intersect:
            m_cids: List[str] = list(set(cids) - set(r_cids) - set(missing_cids))
            m_xcats: List[str] = list(set(xcats) - set(r_xcats) - set(missing_xcats))
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

        if sdf.empty:
            raise ValueError(
                "The arguments provided resulted in an "
                "empty DataFrame when filtered (see `reduce_df`)."
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
        elif ...:
            ...
        else:
            raise NotImplementedError(f"Backend `{backend}` is not supported.")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass
