"""
A subclass inheriting from `macrosynergy.visuals.plotter.Plotter`,
designed to render a facet plot containing any number of subplots.
Given that the class allows returning a `matplotlib.pyplot.Figure`,
one can easily add any number of subplots, even the FacetPlot itself:
effectively allowing for a recursive facet plot.
"""

import pandas as pd
import numpy as np
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


class FacetPlot(Plotter):
    """
    Class for rendering a facet plot containing any number of subplots.
    Inherits from `macrosynergy.visuals.plotter.Plotter`.

    Parameters

    :param <pd.DataFrame> df:

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

    @argvalidation
    @argcopy
    def from_subplots(
        # figure arguments
        plots: Union[List[plt.Figure], List[List[plt.Figure]]],
        figsize: Tuple[float, float] = (16, 9),
        ncols: int = 3,
        attempt_square: bool = False,
        # figure arguments
        grid: bool = True,
        x_axis_label: str = None,
        y_axis_label: str = None,
        axis_fontsize: int = 12,
        # title arguments
        title: str = None,
        title_fontsize: int = 16,
        title_xadjust: float = 0.5,
        title_yadjust: float = 1.05,
        # subplot arguments
        facet_size: Tuple[float, float] = (4, 4),
        facet_titles: List[str] = None,
        legend: bool = True,
        labels: List[str] = None,
        legend_loc: str = "upper right",
        legend_ncol: int = 1,
        legend_bbox_to_anchor: Tuple[float, float] = None,
        legend_frame: bool = True,
        # return args
        show: bool = True,
        save_to_file: Optional[str] = None,
        dpi: int = 300,
        return_figure: bool = False,
    ):
        """
        Render a facet plot from a list of subplots.
        """

        # check if the plots are in a list of lists
        if isinstance(plots[0], list):
            nrows: int = len(plots)
            ncols: int = len(plots[0])
        else:
            nrows: int = 1
            ncols: int = len(plots)

        # attempt to make the arrangment of facets a square
        if attempt_square:
            nrows: int = int(np.ceil(np.sqrt(nrows * ncols)))
            ncols: int = nrows

        # create the figure
        fig: plt.Figure
        axes: List[plt.Axes]
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

        # flatten the axes if necessary
        if nrows == 1:
            axes: List[plt.Axes] = [axes]

        # iterate over the axes and plots
        ax: plt.Axes
        plot: plt.Figure
        for ax, plot in zip(axes, plots):
            # set the axis title
            if facet_titles is not None:
                ax.set_title(facet_titles.pop(0))

            # set the axis labels
            if labels is not None:
                ax.set_xlabel(labels[0])
                ax.set_ylabel(labels[1])

            # add the plot to the axis
            ax.add_artist(plot)

            # add the legend
            if legend:
                ax.legend(
                    loc=legend_loc,
                    ncol=legend_ncol,
                    bbox_to_anchor=legend_bbox_to_anchor,
                    frameon=legend_frame,
                )

        # remove the remaining axes
        for ax in axes[len(plots) :]:
            ax.remove()

        # tight layout
        plt.tight_layout()
        title: str = (
            title if title is not None else f"Facet Plot of {len(plots)} Subplots"
        )

        fig: plt.Figure = plt.gcf()

        if save_to_file:
            plt.savefig(save_to_file, dpi=dpi, bbox_inches="tight")

        if return_figure:
            return fig

        if show:
            plt.show()
            return
