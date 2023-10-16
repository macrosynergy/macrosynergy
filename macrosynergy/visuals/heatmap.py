"""
A subclass inheriting from `macrosynergy.visuals.plotter.Plotter`,
designed to plot time series data as a heatmap.
"""

import pandas as pd
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib as mpl

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
    :param <str> start: ISO-8601 formatted date. Select data from
        this date onwards. If None, all dates are selected.
    :param <str> end: ISO-8601 formatted date. Select data up to
        and including this date. If None, all dates are selected.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        cids: Optional[List[str]] = None,
        xcats: Optional[List[str]] = None,
        metrics: Optional[List[str]] = None,
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
            start=start,
            end=end,
            *args,
            **kwargs,
        )

    def plot(
        self,
        figsize: Tuple[Numeric, Numeric] = (12, 8),
        x_axis_label: Optional[str] = None,
        y_axis_label: Optional[str] = None,
        axis_fontsize: int = 12,
        title: Optional[str] = None,
        title_fontsize: int = 16,
        title_xadjust: Numeric = 0.5,
        title_yadjust: Numeric = 1.0,
        vmin: Optional[Numeric] = None,
        vmax: Optional[Numeric] = None,
        show: bool = True,
        save_to_file: Optional[str] = None,
        dpi: int = 300,
        return_figure: bool = False,
        on_axis: Optional[plt.Axes] = None,
        max_xticks: int = 50,
        *args,
        **kwargs,
    ) -> Optional[plt.Figure]:
        """
        Plots a DataFrame as a heatmap with the columns along the x-axis and
        rows along the y-axis.

        Parameters
        :param <Tuple> figsize: tuple specifying the size of the figure. Default is (12, 8).
        :param <str> x_axis_label: label for x-axis.
        :param <str> y_axis_label: label for y-axis.
        :param <int> axis_fontsize: the font size for the axis labels.
        :param <str> title: the figure's title.
        :param <int> title_fontsize: the font size for the title.
        :param <float> title_xadjust: sets the x position of the title text.
        :param <float> title_yadjust: sets the y position of the title text.
        :param <float> vmin: optional minimum value for heatmap scale.
        :param <float> vmax: optional maximum value for heatmap scale.
        :param <bool> show: if True, the image is displayed.
        :param <str> save_to_file: the path at which to save the heatmap as an image.
            If not specified, the plot will not be saved.
        :param <int> dpi: the resolution in dots per inch used if saving the figure.
        :param <bool> return_figure: if True, the function will return the figure.
        :param <plt.Axes> on_axis: optional `plt.Axes` object to be used instead of
            creating a new one.
        :param <int> max_xticks: the maximum number of ticks to be displayed
            along the x axis. Default is 50.
        """
        if on_axis:
            fig: plt.Figure = on_axis.get_figure()
            ax: plt.Axes = on_axis
        else:
            fig: plt.Figure
            ax: plt.Axes
            fig, ax = plt.subplots(figsize=figsize, layout="constrained")

        im = ax.imshow(
            self.df.to_numpy(),
            cmap=sns.color_palette("light:red", as_cmap=True),
            vmin=vmin,
            vmax=vmax,
            aspect="auto",
            **kwargs,
        )

        xtick_labels = self.df.columns.to_list()
        ytick_labels = self.df.index.to_list()

        ax.set_xticks(np.arange(len(xtick_labels)), labels=xtick_labels)
        ax.set_yticks(np.arange(len(ytick_labels)), labels=ytick_labels)

        ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=max_xticks-1))
        ax.tick_params(which="major", length=4, width=1, direction="out")
        plt.xticks(rotation=90)
        plt.grid(False)

        ax.set_title(
            title,
            fontsize=title_fontsize,
            x=title_xadjust,
            y=title_yadjust,
        )

        if x_axis_label:
            ax.set_xlabel(x_axis_label, fontsize=axis_fontsize)

        if y_axis_label:
            ax.set_ylabel(y_axis_label, fontsize=axis_fontsize)

        ax.figure.colorbar(im, ax=ax)

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

    heatmap = Heatmap(df=dfx, xcats=["FX"])

    heatmap.df["real_date"]: pd.Series = heatmap.df["real_date"].dt.strftime("%Y-%m-%d")
    heatmap.df = heatmap.df.pivot_table(index="cid", columns="real_date", values="grading")

    heatmap.plot(title='abc')
