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
import io
import pickle
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
        df: pd.DataFrame = None,
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
        # If the plots are a list of list, plot each list as a row. max cols becomes the length of the longest row.

        if isinstance(plots[0], list):
            nrows: int = len(plots)
            ncols: int = max([len(row) for row in plots])

        # If the plots are a list of figures, use user provided ncols and infer nrows.

        else:
            if ncols > len(plots):
                ncols = len(plots)
            nrows: int = int(np.ceil(len(plots) / ncols))

            # If attempt_square is True, attempt to make the the grid of subplots square - i.e. ncols = nrows.
            if attempt_square:
                ncols: int = int(np.ceil(np.sqrt(len(plots))))
                nrows: int = int(np.ceil(len(plots) / ncols))

        # Create the figure and axes.]
        fig: plt.Figure
        axes: np.ndarray
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=figsize,
            sharex=True,
            sharey=True,
            squeeze=False,
        )

        # Flatten the axes array.
        axes: np.ndarray = axes.flatten()

        # now simply "add" the each plt.Figure to each axis on the figure

        # Loop through the plots and add them to the figure axes.
        for idx, plot in enumerate(plots):
            # Check if index is out of range of the axes.
            if idx >= len(axes):
                break

            # Serialize the plot using pickle, then deserialize it onto the axes.
            buf = io.BytesIO()
            pickle.dump(plot, buf)
            buf.seek(0)
            ax = pickle.load(buf)

            # Now add this ax to the figure.
            axes[idx].cla()  # Clear the axes first
            axes[idx].add_artist(ax)

            # Add facet titles if provided
            if facet_titles is not None:
                try:
                    axes[idx].set_title(facet_titles[idx])
                except IndexError:
                    pass  # If not enough titles are provided, ignore them.

        # Add the overall title and labels.
        if title:
            fig.suptitle(title, fontsize=title_fontsize, x=title_xadjust, y=title_yadjust)
        if x_axis_label:
            fig.set_xlabel(x_axis_label, fontsize=axis_fontsize)
        if y_axis_label:
            fig.set_ylabel(y_axis_label, fontsize=axis_fontsize)

        # Display the legend.
        if legend:
            fig.legend(labels, loc=legend_loc, ncol=legend_ncol, bbox_to_anchor=legend_bbox_to_anchor, frameon=legend_frame)

        # Save the figure if a filename is provided.
        if save_to_file:
            fig.savefig(save_to_file, dpi=dpi)

        # Show the figure.
        if show:
            plt.show()

        # Return the figure if requested.
        if return_figure:
            return fig