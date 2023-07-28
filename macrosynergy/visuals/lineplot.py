import pandas as pd
from typing import List, Dict, Union, Tuple, Optional, Union
from types import ModuleType
from collections.abc import Callable, Iterable
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from macrosynergy.management.utils import standardise_dataframe
from macrosynergy.management import reduce_df


from .plotter import Plotter, argcopy, argvalidation


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
        *args,
        **kwargs,
    ):
        super().__init__(
            df=df,
            cids=cids,
            xcats=xcats,
            metrics=metrics,
            intersect=intersect,
            tickers=tickers,
            blacklist=blacklist,
            start=start,
            end=end,
            *args,
            **kwargs,
        )

        """
    # TODO - implement this method
    2. See if it is possible return a matplotlib figure object, that can be
        further manipulated by the user - or even used as a subplot.
    3. Add and implement:
        - legend fine-tuning
        - x-axis fine-tuning
        - y-axis fine-tuning
        - title fine-tuning
        - grid fine-tuning
    
    """

    @argvalidation
    @argcopy
    def plot(
        self,
        # DF specific arguments
        df: pd.DataFrame,
        xcats: List[str] = None,
        cids: List[str] = None,
        intersect: bool = False,
        start: str = "2000-01-01",
        end: Optional[str] = None,
        # df/plot args
        metric: str = "value",
        compare_series: Optional[Union[str, List[str]]] = None,
        # Plotting specific arguments
        # fig args
        figsize: Tuple[int, int] = (12, 8),
        aspect: float = 1.618,
        height: float = 0.8,
        # plot args
        grid: bool = True,
        x_axis_label: Optional[str] = None,
        y_axis_label: Optional[str] = None,
        axis_fontsize: int = 12,
        # title args
        title: Optional[str] = None,
        title_fontsize: int = 16,
        title_xadjust: float = 0.5,
        title_yadjust: float = 1.05,
        # legend args
        legend: bool = True,
        labels: Optional[List[str]] = None,
        legend_loc: str = "upper left",
        legend_fontsize: int = 12,
        legend_ncol: int = 1,
        legend_bbox_to_anchor: Optional[Tuple[float, float]] = None,
        # return args
        show: bool = True,
        save_to_file: Optional[str] = None,
        dpi: int = 300,
        return_figure: bool = False,
        # args, kwargs
        *args,
        **kwargs,
    ):
        # if any of DF specific arguments is provided, re-initialise the object with the df args
        if any([xcats, cids, intersect, start, end]):
            self.__init__(
                df=df,
                xcats=xcats,
                cids=cids,
                intersect=intersect,
                start=start,
                end=end,
            )

        #
