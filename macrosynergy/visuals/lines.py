import pandas as pd
from typing import List, Dict, Union, Tuple
from types import ModuleType
from collections.abc import Callable, Iterable
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from macrosynergy.management.utils import standardise_dataframe
from macrosynergy.management import reduce_df

from .plotter import Plotter

class LinePlot(Plotter):
    """
    Class for plotting time series data on a line plot.
    Inherits from `class Plotter`.

    Parameters
    ----------
    :param <pd.DataFrame> df: A DataFrame with the following columns:
        'cid', 'xcat', 'real_date', and at least one metric from - 
        'value', 'grading', 'eop_lag', or 'mop_lag'. 
    :param <List[str]> cids: A list of cids to select from the DataFrame.
        If None, all cids are selected.
    :param <List[str]> xcats: A list of xcats to select from the DataFrame.
        If None, all xcats are selected.
    :param <List[str]> metrics: A list of metrics to select from the DataFrame.
        If None, all metrics are selected.
    :param <str> start_date: ISO-8601 formatted date. Select data from
        this date onwards. If None, all dates are selected.
    :param <str> end_date: ISO-8601 formatted date. Select data up to
        and including this date. If None, all dates are selected.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        cids: List[str] = None,
        xcats: List[str] = None,
        start_date: str = None,
        metrics: str = ["value"],
        end_date: str = None,
    ):

        super().__init__(
            df=df,
            cids=cids,
            xcats=xcats,
            start_date=start_date,
            end_date=end_date,
            metrics=metrics,
        )
        
    def plot(
        
    )
    
    
    
    
    