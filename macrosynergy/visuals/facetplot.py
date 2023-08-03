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
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
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


def _get_grid_dim(
    num_plots: int,
) -> Tuple[int, int]:
    """
    Given the number of plots, return a tuple of grid dimensions
    that is closest to a square grid.
    :param <int> num_plots: Number of plots.
    :return <Tuple[int, int]>: Tuple of grid dimensions.
    """
    # take the square root of the number of plots, and try and iteratetively find the closest set of grid dimensions that will work
    # for the number of plots.
    sqrt_num_plots: float = np.ceil(np.sqrt(num_plots))
    grid_dim: Tuple[int, int] = (int(sqrt_num_plots), int(sqrt_num_plots))
    gd_copy: Tuple[int, int] = grid_dim
    # the number of plots is less than grid_dim[0] * grid_dim[1], so iteratively try and reduce the row and column dimensions until sweet spot is found.
    while 0 != 1:
        if gd_copy[0] < gd_copy[1]:
            gd_copy: Tuple[int, int] = (gd_copy[0], gd_copy[1] - 1)
        else:
            gd_copy: Tuple[int, int] = (gd_copy[0] - 1, gd_copy[1])

        # if gd_copy[0] * gd_copy[1] is more than num_plots, copy to grid_dim. if smaller, break.
        if gd_copy[0] * gd_copy[1] >= num_plots:
            grid_dim: Tuple[int, int] = gd_copy
        else:
            break

    return grid_dim


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
        # should be forced outside the plot
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

    @staticmethod
    def _cart_plot(
        df_wide: pd.DataFrame,
        plot_func: Callable = plt.plot,
        use_x: Union[str, List[str]] = "index",
        grid_dim: Tuple[int, int] = None,
        plot_func_args: List = None,
        # figsize: Tuple[float, float] = (16, 9),
        # title arguments
        title: str = None,
        title_fontsize: int = 16,
        title_xadjust: float = 0.5,
        title_yadjust: float = 1.05,
        # subplot axis arguments
        subplot_grid: bool = True,
        ax_hline: bool = False,
        ax_hline_val: float = 0,
        ax_vline: bool = False,
        ax_vline_val: float = 0,
        x_axis_label: str = None,
        y_axis_label: str = None,
        axis_fontsize: int = 12,
        # subplot arguments
        facet_size: Tuple[float, float] = (4, 3),
        facet_titles: List[str] = None,
        facet_title_fontsize: int = 12,
        facet_title_xadjust: float = 0.5,
        facet_title_yadjust: float = 1.05,
        # legend arguments
        legend: bool = True,
        legend_labels: List[str] = None,
        legend_loc: str = "upper right",
        legend_ncol: int = 1,
        legend_bbox_to_anchor: Tuple[float, float] = None,
        legend_frame: bool = True,
        # return args
        show: bool = True,
        save_to_file: Optional[str] = None,
        dpi: int = 300,
        return_figure: bool = False,
        *args,
        **kwargs,
    ):
        """
        Render a facet plot from a wide dataframe, a grid dimension and a plotting function.
        """
        # check that the df has an index called "real_date". there should be only one index. it also must be pd.DatetimeIndex type.
        inval_df: str = (
            "`df_wide` must have a single index called "
            "`real_date` of type `pd.DatetimeIndex`."
        )
        assert df_wide.index.names == ["real_date"], inval_df
        assert isinstance(df_wide.index, pd.DatetimeIndex), inval_df

        if isinstance(use_x, str):
            if not use_x == "index":
                # make sure that the column exists in the dataframe
                inval_use_x: str = f"Column {use_x} does not exist in `df_wide`."
                assert use_x in df_wide.columns, inval_use_x
        else:
            assert isinstance(
                use_x, list
            ), "`use_x` must be a string or a list of strings."
            # make sure that all columns exist in the dataframe
            inval_use_x: str = f"Columns {use_x} do not exist in `df_wide`."
            assert all([col in df_wide.columns for col in use_x]), inval_use_x

        x_values: List[List[float]] = []
        if isinstance(use_x, str):
            if use_x == "index":
                x_values.append(df_wide.index.values.tolist())
            else:
                x_values.append(df_wide[use_x].values.tolist())
        else:
            for col in use_x:
                x_values.append(df_wide[col].values.tolist())
                df_wide: pd.DataFrame = df_wide.drop(col, axis=1)

            # ensure df_wide has the same number of columns as the number of x_values
            assert df_wide.shape[1] == len(x_values), (
                f"Number of x_axis values passed using `use_x` ({use_x}) "
                f"must be equal to the number of data columns in `df_wide` ({df_wide.shape[1]})."
            )

        if grid_dim is None:
            num_plots: int = df_wide.shape[1]
            # distribute into a square grid
            grid_dim: Tuple[int, int] = _get_grid_dim(num_plots)
        else:
            num_plots: int = grid_dim[0] * grid_dim[1]

        inval_grid_dims: str = (
            f"Grid dimensions {grid_dim} must be greater than or "
            f"equal to the number of columns in `df_wide` ({df_wide.shape[1]})."
        )

        assert num_plots >= df_wide.shape[1], inval_grid_dims

        figsize: Tuple[float, float] = (
            grid_dim[1] * facet_size[0],
            grid_dim[0] * facet_size[1],
        )

        fig = plt.figure(figsize=figsize, layout="tight")
        gs: GridSpec = GridSpec(*grid_dim, figure=fig)
        if plot_func_args is None:
            plot_func_args: List = []

        if facet_titles is None:
            facet_titles: List[str] = df_wide.columns.tolist()

        mulx: int = int(len(x_values) > 1)
        # if the index is the xval, then cast it to a pd.DatetimeIndex
        if use_x == "index":
            x_values[0] = pd.to_datetime(x_values[0])
        for i in range(num_plots):
            ax: plt.Axes = fig.add_subplot(gs[i // grid_dim[1], i % grid_dim[1]])
            plot_func(
                x_values[i * mulx],
                df_wide.iloc[:, i].values.tolist(),
                *plot_func_args,
                **kwargs,
            )
            ax.set_title(
                facet_titles[i],
                fontsize=facet_title_fontsize,
                x=facet_title_xadjust,
                y=facet_title_yadjust,
            )
            if subplot_grid:
                ax.grid()
            if ax_hline:
                ax.axhline(ax_hline_val, color="black", linestyle="--")
            if ax_vline:
                ax.axvline(ax_vline_val, color="black", linestyle="--")
            if x_axis_label is not None:
                ax.set_xlabel(x_axis_label, fontsize=axis_fontsize)
            if y_axis_label is not None:
                ax.set_ylabel(y_axis_label, fontsize=axis_fontsize)

        if title is not None:
            fig.suptitle(
                title,
                fontsize=title_fontsize,
                x=title_xadjust,
                y=title_yadjust,
            )

        if legend:
            if legend_labels is None:
                legend_labels = df_wide.columns.tolist()
                if isinstance(use_x, list) and len(use_x) >= 1:
                    legend_labels = [
                        f"{legend_labels[i]} v/s {use_x[i]}"
                        for i in range(len(legend_labels))
                    ]
                    
            fig.legend(
                labels=legend_labels,
                loc=legend_loc,
                ncol=legend_ncol,
                bbox_to_anchor=legend_bbox_to_anchor,
                frameon=legend_frame,
            )
                            
        fig.tight_layout()
        
        # plt.tight_layout()

        if save_to_file is not None:
            fig.savefig(save_to_file, dpi=dpi)

        if show:
            plt.show()

        if return_figure:
            return fig
