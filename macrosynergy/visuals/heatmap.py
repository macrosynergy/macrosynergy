"""
A subclass inheriting from `macrosynergy.visuals.plotter.Plotter`,
designed to plot time series data as a heatmap.
"""

import pandas as pd
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from macrosynergy.visuals.plotter import Plotter
from macrosynergy.visuals.common import Numeric, NoneType

from macrosynergy.management.simulate_quantamental_data import make_test_df
from macrosynergy.management.shape_dfs import reduce_df


class Heatmap(Plotter):
    """
    Class for plotting time series data as a heatmap.
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
        cids: Optional[List[str]] = None,
        xcats: Optional[List[str]] = None,
        metrics: Optional[List[str]] = None,
        intersect: bool = False,
        tickers: Optional[List[str]] = None,
        blacklist: Optional[Dict[str, List[str]]] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
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
        xcat: str,
        metric: str,
        # plot args
        figsize: Tuple[Numeric, Numeric] = (12, 8),
        grid: bool = False,
        x_axis_label: Optional[str] = None,
        y_axis_label: Optional[str] = None,
        axis_fontsize: int = 12,
        vmin=None,
        vmax=None,
        # title args
        title: Optional[str] = None,
        title_fontsize: int = 16,
        title_xadjust: Numeric = 0.5,
        title_yadjust: Numeric = 1.05,
        # return args
        show: bool = True,
        save_to_file: Optional[str] = None,
        dpi: int = 300,
        return_figure: bool = False,
        on_axis: Optional[plt.Axes] = None,
        *args,
        **kwargs,
    ):
        if on_axis:
            fig: plt.Figure = on_axis.get_figure()
            ax: plt.Axes = on_axis
        else:
            fig: plt.Figure
            ax: plt.Axes
            fig, ax = plt.subplots(figsize=figsize, layout="constrained")

        ax.imshow(
            self.df.to_numpy(),
            cmap="Reds",
            vmin=vmin,
            vmax=vmax,
            aspect="auto",
            **kwargs,
        )

        real_dates = self.df.columns.to_list()

        ax.set_xticks(np.arange(len(real_dates)), labels=real_dates)
        ax.set_yticks(np.arange(len([1])), labels=[xcat])

        ax.xaxis.set_major_locator(plt.MaxNLocator(min(len(real_dates), 25)))
        ax.tick_params(which="major", length=4, width=1, direction="out")
        plt.xticks(rotation=90)
        plt.grid(False)

        ax.set_title(
            title,
            fontsize=title_fontsize,
            x=title_xadjust,
            y=title_yadjust,
        )
        if grid:
            plt.grid(True)
            ax.grid(axis="both", linestyle="--", alpha=0.5)

        if x_axis_label:
            ax.set_xlabel(x_axis_label, fontsize=axis_fontsize)

        if y_axis_label:
            ax.set_ylabel(y_axis_label, fontsize=axis_fontsize)

        if save_to_file:
            plt.savefig(
                save_to_file,
                dpi=dpi,
                bbox_inches="tight",
            )

        if show:
            plt.show()
            return

        if return_figure:
            return fig


if __name__ == "__main__":
    test_cids: List[str] = [
        "USD",
    ]  # "EUR", "GBP"]
    test_xcats: List[str] = ["FX", "IR"]
    dfE: pd.DataFrame = make_test_df(
        cids=test_cids, xcats=test_xcats, style="sharp-hill"
    )

    dfM: pd.DataFrame = make_test_df(
        cids=test_cids, xcats=test_xcats, style="four-bit-sine"
    )

    dfG: pd.DataFrame = make_test_df(cids=test_cids, xcats=test_xcats, style="sine")

    dfE.rename(columns={"value": "eop_lag"}, inplace=True)
    dfM.rename(columns={"value": "mop_lag"}, inplace=True)
    dfG.rename(columns={"value": "grading"}, inplace=True)
    mergeon = ["cid", "xcat", "real_date"]
    dfx: pd.DataFrame = pd.merge(pd.merge(dfE, dfM, on=mergeon), dfG, on=mergeon)

    # view_metrics(
    #     df=dfx,
    #     xcat="FX",
    # )
    # view_metrics(
    #     df=dfx,
    #     xcat="IR",
    #     metric="mop_lag",
    # )
    # view_metrics(
    #     df=dfx,
    #     xcat="IR",
    #     metric="grading",
    # )

    heatmap = Heatmap(df=dfx, xcats=["FX"])
    heatmap.plot(xcat="FX", metric="eop_lag")
