"""
Module containing code for the Plotter class, and all common functionality
for the plotter classes. The Plotter class is a base class for all plotter
classes, and provides a shared interface for dataframe filtering operations,
as well as `argvalidation` and `argcopy` decorators for all methods of the
plotter classes.
"""

import logging
from types import ModuleType
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from macrosynergy.management.decorators import argcopy, argvalidation
from macrosynergy.management.validation import validate_and_reduce_qdf

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
    This class does not implement any plotting functionality, but provides a shared
    interface for the plotter classes, and some common functionality - currently just the
    filtering of the DataFrame.
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
    :param <List[str]> tickers: A list of tickers that will be selected from the DataFrame
        if they exist, regardless of start, end, blacklist, and intersect arguments.
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
        validated_args = validate_and_reduce_qdf(
            df=df,
            cids=cids,
            xcats=xcats,
            metrics=metrics,
            intersect=intersect,
            tickers=tickers,
            blacklist=blacklist,
            start=start,
            end=end,
        )

        self.df: pd.DataFrame = validated_args.df
        self.cids: List[str] = validated_args.cids
        self.xcats: List[str] = validated_args.xcats
        self.metrics: List[str] = validated_args.metrics
        self.intersect: bool = intersect
        self.tickers: List[str] = tickers
        self.blacklist: Dict[str, List[str]] = blacklist
        self.start: str = validated_args.start
        self.end: str = validated_args.end

        self.backend: ModuleType
        accepted_backends: List[str] = ["matplotlib", "plt", "mpl"]
        if not backend.strip().lower() in accepted_backends:
            raise NotImplementedError(f"Backend `{backend}` is not supported.")

        self.backend = plt
        sns.set_theme(style="darkgrid", palette="colorblind")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass
