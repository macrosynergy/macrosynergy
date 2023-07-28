import pandas as pd
from typing import List, Dict, Union, Tuple, Optional, Union
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
        start: str = None,
        metrics: str = ["value"],
        end: str = None,
    ):
        super().__init__(
            df=df,
            cids=cids,
            xcats=xcats,
            start=start,
            end=end,
            metrics=metrics,
        )

    def plot(
        df: pd.DataFrame,
        xcats: List[str] = None,
        cids: List[str] = None,
        intersect: bool = False,
        metric: str = "value",
        compare_series: Optional[Union[str, List[str]]] = None,
        
        start: str = "2000-01-01",
        end: Optional[str] = None,
        
        figsize: Tuple[int, int] = (12, 8),
        
        title: Optional[str] = None,
        title_fontsize: int = 16,
        title_fontweight: str = "bold",
        title_fontfamily: str = "serif",
        title_xadjust: float = 0.5,
        title_yadjust: float = 1.05,
        
        
        legend: bool = True,
        labels: Optional[List[str]] = None,
        legend_loc: str = "upper left",
        
        
        ):
        pass

    """
    # TODO - implement this method
    1. Generalize the view_timelines() specific arguments to generic arguments.
    2. See if it is possible return a matplotlib figure object, that can be
        further manipulated by the user - or even used as a subplot.
    3. Add and implement:
        - legend fine-tuning
        - x-axis fine-tuning
        - y-axis fine-tuning
        - title fine-tuning
        - grid fine-tuning
    
    """
