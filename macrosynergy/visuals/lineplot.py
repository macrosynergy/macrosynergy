"""
A subclass inheriting from `macrosynergy.visuals.plotter.Plotter`,
designed to plot time series data on a line plot.
"""

import pandas as pd
from typing import List, Dict, Union, Tuple, Optional, Union
from types import ModuleType
from collections.abc import Callable, Iterable
import matplotlib.pyplot as plt
import seaborn as sns
import logging

import sys, os

sys.path.append(os.path.abspath("."))

from macrosynergy.management.utils import standardise_dataframe
from macrosynergy.management import reduce_df
from macrosynergy.management.simulate_quantamental_data import make_test_df


from macrosynergy.visuals.plotter import Plotter, argcopy, argvalidation


class LinePlot(Plotter):
    """
    Class for plotting time series data on a line plot.
    Inherits from `macrosynergy.visuals.plotter.Plotter`.

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

    def plot(
        self,
        on_axis: Optional[plt.Axes] = None,
        # # DF specific arguments
        df: pd.DataFrame = None,
        xcats: List[str] = None,
        cids: List[str] = None,
        intersect: bool = False,
        blacklist: Dict[str, List[str]] = None,
        tickers: List[str] = None,
        start: str = "2000-01-01",
        end: Optional[str] = None,
        # df/plot args
        metric: str = "value",
        compare_series: str = None,
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
        legend_title: Optional[str] = None,
        legend_loc: str = "upper right",
        legend_fontsize: int = 12,
        legend_ncol: int = 1,
        legend_bbox_to_anchor: Tuple[float, float] = (1.2, 1.0),
        legend_frame: bool = True,
        # return args
        show: bool = True,
        save_to_file: Optional[str] = None,
        dpi: int = 300,
        return_figure: bool = False,
        # args, kwargs
        *args,
        **kwargs,
    ):
        if any([df, xcats, cids, intersect, blacklist, tickers, start, end]):
            self.__init__(
                df=df if df is not None else self.df,
                xcats=xcats,
                cids=cids,
                metrics=[metric],
                intersect=intersect,
                blacklist=blacklist,
                tickers=tickers,
                start=start,
                end=end,
            )

        if on_axis:
            fig: plt.Figure = on_axis.get_figure()
            ax: plt.Axes = on_axis
        else:
            fig: plt.Figure
            ax: plt.Axes
            fig, ax = plt.subplots(figsize=figsize)

        dfx: pd.DataFrame = self.df.copy()

        if compare_series:
            _cid, _xcat = compare_series.split("_", 1)
            if _cid not in dfx["cid"].unique() or _xcat not in dfx["xcat"].unique():
                raise ValueError(
                    f"Series `{compare_series}` not in DataFrame - used as `compare_series`."
                )

            comp_df = (
                dfx.loc[
                    (dfx["cid"] == _cid) & (dfx["xcat"] == _xcat), ["real_date", metric]
                ]
                .copy()
                .reset_index(drop=True)
            )
            # remove the compare_series from the dfx
            dfx = dfx.loc[~((dfx["cid"] == _cid) & (dfx["xcat"] == _xcat)), :]

        # use plt to create a plot, and use cid_xcat to differentiate the lines, cid_xcat are not real colors
        for cid_xcat in dfx[["cid", "xcat"]].drop_duplicates().values:
            cid, xcat = cid_xcat
            _df = dfx.loc[(dfx["cid"] == cid) & (dfx["xcat"] == xcat), :].copy()
            _df = _df.sort_values(by="real_date", ascending=True).reset_index(drop=True)
            ax.plot(_df["real_date"], _df[metric], label=f"{cid}_{xcat}")

        # if there is a compare_series, plot it on the same axis, using a red dashed line
        if compare_series:
            ax.plot(comp_df["real_date"], comp_df[metric], color="red", linestyle="--")

        if grid:
            ax.grid(axis="both", linestyle="--", alpha=0.5)

        if x_axis_label:
            ax.set_xlabel(x_axis_label, fontsize=axis_fontsize)

        if y_axis_label:
            ax.set_ylabel(y_axis_label, fontsize=axis_fontsize)

        # if there is a title, add it
        if title:
            plt.title(
                title,
                fontsize=title_fontsize,
                x=title_xadjust,
                y=title_yadjust,
            )

        # if there is a legend, add it
        if legend:
            plt.legend(
                labels=labels if labels else None,
                title=legend_title,
                loc=legend_loc,
                fontsize=legend_fontsize,
                ncol=legend_ncol,
                bbox_to_anchor=legend_bbox_to_anchor,
                frameon=legend_frame,
            )

        plt.tight_layout()
        title: str = title if title else f"LinePlot: Viewing `{metric}`"

        if save_to_file:
            plt.savefig(
                save_to_file,
                dpi=dpi,
                bbox_inches="tight",
            )

        if return_figure:
            return fig

        if show:
            plt.show()
            return


if __name__ == "__main__":
    cids: List[str] = ["USD", "EUR", "GBP", "AUD", "CAD"]
    xcats: List[str] = ["FXXR", "EQXR", "RIR"]
    df: pd.DataFrame = make_test_df(
        cids=cids,
        xcats=xcats,
        start_date="2000-01-01",
        end_date="2020-12-31",
        # prefer="sine",
    )

    LinePlot(df=df).plot(
        cids=["USD", "EUR"], xcats=["FXXR"], labels=["The US", "Europe"]
    )
