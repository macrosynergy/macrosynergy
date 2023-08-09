"""
A subclass inheriting from `macrosynergy.visuals.plotter.Plotter`,
designed to render a facet plot containing any number of subplots.
Given that the class allows returning a `matplotlib.pyplot.Figure`,
one can easily add any number of subplots, even the FacetPlot itself:
effectively allowing for a recursive facet plot.
"""

import io
import logging
import os
import pickle
import sys
from collections.abc import Callable, Iterable
from types import ModuleType
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

sys.path.append(os.path.abspath("."))

from macrosynergy.visuals.plotter import Plotter

def _get_square_grid(
    num_plots: int,
) -> Tuple[int, int]:
    """
    Given the number of plots, return a tuple of grid dimensions
    that is closest to a square grid.
    :param <int> num_plots: Number of plots.
    :return <Tuple[int, int]>: Tuple of grid dimensions.
    """
    sqrt_num_plots: float = np.ceil(np.sqrt(num_plots))
    grid_dim: Tuple[int, int] = (int(sqrt_num_plots), int(sqrt_num_plots))
    gd_copy: Tuple[int, int] = grid_dim
    # the number of plots is less than grid_dim[0] * grid_dim[1],
    # so iteratively try and reduce the row and column dimensions until sweet spot is found.
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
        df: Optional[pd.DataFrame] = None,
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

    def _get_grid_dim(
        self,
        tickers: List[str],
        ncols: Optional[int] = None,
        attempt_square: bool = False,
        cid_grid: bool = False,
        xcat_grid: bool = False,
        cid_xcat_grid: bool = False,
    ) -> Tuple[int, int]:
        """
        Returns a tuple of grid dimensions that matches the plot settings.
        """
        # only one of cid_grid, xcat_grid, cid_xcat_grid or attempt_square can be True
        if sum([cid_grid, xcat_grid, cid_xcat_grid]) > 1:
            raise ValueError(
                "Only one of `cid_grid`, `xcat_grid`, or "
                "`cid_xcat_grid` can be True."
            )

        if (ncols is None) and (not attempt_square):
            raise ValueError("Either `ncols` or `attempt_square` must be provided.")

        if attempt_square:
            return _get_square_grid(num_plots=len(tickers))

        if cid_grid:
            found_cids: List[str] = self.cids
            return self._get_grid_dim(
                tickers=tickers, ncols=ncols, attempt_square=attempt_square
            )

        if xcat_grid:
            found_xcats: List[str] = self.xcats
            return self._get_grid_dim(
                tickers=tickers, ncols=ncols, attempt_square=attempt_square
            )

        if ncols is not None:
            return (ncols, int(np.ceil(len(tickers) / ncols)))

        if cid_xcat_grid:
            found_cids: List[str] = self.cids
            found_xcats: List[str] = self.xcats
            return (len(found_cids), len(found_xcats))

        raise ValueError("Unable to infer grid dimensions.")

    def _cart_plot(
        self,
        grid_dim: Tuple[int, int],
        plot_dict: Dict[int, Dict[str, Any]],
        metric: Optional[str] = None,
        plot_func: Callable = plt.plot,
        plot_func_args: Optional[List[Any]] = None,
        figsize: Tuple[float, float] = (16.0, 9.0),
        # title arguments
        title: Optional[str] = None,
        title_fontsize: int = 16,
        title_xadjust: Optional[float] = None,
        title_yadjust: Optional[float] = None,
        # subplot axis arguments
        ax_grid: bool = True,
        ax_hline: bool = False,
        ax_hline_val: float = 0,
        ax_vline: bool = False,
        ax_vline_val: float = 0,
        x_axis_label: Optional[str] = None,
        y_axis_label: Optional[str] = None,
        axis_fontsize: int = 12,
        # subplot arguments
        facet_size: Optional[Tuple[float, float]] = None,
        facet_titles: Optional[List[str]] = None,
        facet_title_fontsize: int = 12,
        facet_title_xadjust: float = 0.5,
        facet_title_yadjust: float = 1.0,
        # legend arguments
        legend: bool = True,
        legend_labels: Optional[List[str]] = None,
        legend_loc: str = "lower right",
        legend_ncol: int = 1,
        legend_bbox_to_anchor: Optional[Tuple[float, float]] = None,
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

        fig = plt.figure(figsize=figsize)
        gs: GridSpec = GridSpec(*grid_dim, figure=fig)

        # if title is not None:
        fig.suptitle(
            title,
            fontsize=title_fontsize,
            x=title_xadjust,
            y=title_yadjust,
        )

        if plot_func_args is None:
            plot_func_args: List = []

        for i, (plot_id, plot_dict) in enumerate(plot_dict.items()):
            ax: plt.Axes = fig.add_subplot(gs[i // grid_dim[1], i % grid_dim[1]])
            for y in plot_dict["Y"]:
                # split on the first underscore
                cidx, xcatx = y.split("_", 1)
                plot_func(
                    self.df.loc[plot_dict["X"]],
                    self.df.loc[(df["cid"] == cidx) & (df["xcat"] == xcatx)] ,
                    *plot_func_args,
                    **kwargs,
                )

            if facet_titles is not None:
                ax.set_title(
                    facet_titles[i],
                    fontsize=facet_title_fontsize,
                    x=facet_title_xadjust,
                    y=facet_title_yadjust,
                )
            if ax_grid:
                ax.grid(axis="both", linestyle="--", alpha=0.5)
            if ax_hline:
                ax.axhline(ax_hline_val, color="black", linestyle="--")
            if ax_vline:
                ax.axvline(ax_vline_val, color="black", linestyle="--")
            if x_axis_label is not None:
                ax.set_xlabel(x_axis_label, fontsize=axis_fontsize)
            if y_axis_label is not None:
                ax.set_ylabel(y_axis_label, fontsize=axis_fontsize)

        fig.tight_layout()
        if legend:
            legend_obj = fig.legend(
                labels=legend_labels,
                loc=legend_loc,
                ncol=legend_ncol,
                bbox_to_anchor=legend_bbox_to_anchor,
                frameon=legend_frame,
            )

        if save_to_file is not None:
            fig.savefig(save_to_file, dpi=dpi)

        if show:
            plt.show()

        if return_figure:
            return fig

    def lineplot(
        self,
        # dataframe arguments
        df: Optional[pd.DataFrame] = None,
        cids: Optional[List[str]] = None,
        xcats: Optional[List[str]] = None,
        metric: Optional[str] = None,
        intersect: bool = False,
        tickers: Optional[List[str]] = None,
        blacklist: Optional[Dict[str, List[str]]] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        # plot arguments
        ncols: int = 3,
        attempt_square: bool = False,
        cid_grid: bool = False,
        xcat_grid: bool = False,
        cid_xcat_grid: bool = False,
        grid_dim: Optional[Tuple[int, int]] = None,
        cids_mean: bool = False,
        compare_series: Optional[str] = None,
        # xcats_mean: bool = False,
        # title arguments
        figsize: Tuple[float, float] = (16.0, 9.0),
        title: Optional[str] = None,
        title_fontsize: int = 20,
        title_xadjust: Optional[float] = None,
        title_yadjust: Optional[float] = None,
        # subplot axis arguments
        ax_grid: bool = True,
        ax_hline: bool = False,
        ax_hline_val: float = 0.0,
        ax_vline: bool = False,
        ax_vline_val: float = 0.0,
        x_axis_label: Optional[str] = None,
        y_axis_label: Optional[str] = None,
        axis_fontsize: int = 12,
        # subplot arguments
        facet_size: Optional[Tuple[Union[float, int], Union[float, int]]] = None,
        facet_titles: Optional[List[str]] = None,
        facet_title_fontsize: int = 12,
        facet_title_xadjust: float = 0.5,
        facet_title_yadjust: float = 1.0,
        facet_xlabel: Optional[str] = None,
        facet_ylabel: Optional[str] = None,
        facet_label_fontsize: int = 12,
        # legend arguments
        legend: bool = True,
        legend_labels: Optional[List[str]] = None,
        legend_loc: str = "center left",
        legend_ncol: int = 1,
        legend_bbox_to_anchor: Optional[Tuple[float, float]] = (1.0, 0.5),
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
        if any([arg is not None for arg in [df, cids, xcats, metric, tickers]]):
            # undesirable, as the state of the object will change kept for ease of use
            metrics: List[str] = [metric] if metric is not None else self.metrics
            metrics: Optional[List[str]] = metrics if metrics else None
            self.__init__(
                df=df if df is not None else self.df,
                cids=cids,
                xcats=xcats,
                metrics=metrics,
                intersect=intersect,
                tickers=tickers,
                blacklist=blacklist,
                start=start,
                end=end,
            )
        if metric is None:
            metric: str = self.metrics[0]

        tickers_to_plot: List[str] = (
            (self.df["cid"] + "_" + self.df["xcat"]).unique().tolist()
        )

        if grid_dim is None:
            grid_dim: Tuple[int, int] = self._get_grid_dim(
                tickers=tickers_to_plot,
                ncols=ncols,
                attempt_square=attempt_square,
                cid_grid=cid_grid,
                xcat_grid=xcat_grid,
                cid_xcat_grid=cid_xcat_grid,
            )

        # if the facet size is not none, re-calc the figsize
        if facet_size is not None:
            figsize: Tuple[float, float] = (
                grid_dim[1] * facet_size[0],
                grid_dim[0] * facet_size[1],
            )

        # form a dictionary of what goes on each plot
        # each key is an int - identifying the plot (L-R, T-B)
        # each value is a dict with keys "xs", "ys"

        if compare_series is not None:
            # check that the compare series exists in the dataframe
            if compare_series not in tickers_to_plot:
                raise ValueError(
                    f"Compare series {compare_series} not found in dataframe."
                )
            else:
                tickers_to_plot.remove(compare_series)

        if cids_mean:
            if (not cid_grid) or len(self.xcats) > 1:
                raise ValueError(
                    "`cids_mean` can only be True if `cid_grid` is True and "
                    "there is only one xcat."
                )
            else:
                # add the mean for each cid to the dataframe
                df_mean: pd.DataFrame = self.df.groupby("cid").mean(numeric_only=True).reset_index()
                df_mean["xcat"] = "mean"
                self.df: pd.DataFrame = pd.concat([self.df, df_mean], axis=0)

        plot_dict: Dict[str, List[str]] = {}

        if facet_titles is None:
            if cid_grid:
                facet_titles: List[str] = self.cids
            elif xcat_grid:
                facet_titles: List[str] = self.xcats
            elif cid_xcat_grid:
                ...
                # cid_xcat_grid facets only make sense if they have cid_xcat as the title

        if cid_grid or xcat_grid:
            # flipper handles resolution between cid_grid and xcat_grid for binary variables
            flipper: bool = 1 if cid_grid else -1
            if facet_titles is None:
                facet_titles: List[str] = [self.cids, self.xcats][::flipper][0]
            if legend_labels is None:
                legend_labels: List[str] = [self.cids, self.xcats][::flipper][0]

            for i, fvar in enumerate([self.cids, self.xcats][::flipper][0]):
                tks: List[str] = [
                    "_".join([fvar, x][::flipper])
                    for x in ([self.xcats, self.cids][::flipper][0])
                ]

                tks: List[str] = list(set(tks).intersection(tickers_to_plot))
                if cid_grid and cids_mean:
                    tks.append("_".join([fvar, "mean"]))

                plot_dict[i] = {
                    "X": "real_date",
                    "Y": tks,
                }

        if cid_xcat_grid:
            if facet_titles is None:
                facet_titles: List[str] = [
                    f"{cid}_{xcat}" for cid in self.cids for xcat in self.xcats
                ]
            # TODO fix : legend goes away in cid_xcat_grid
            legend: bool = False

            for i, cid in enumerate(self.cids):
                for j, xcat in enumerate(self.xcats):
                    tk: str = "_".join([cid, xcat])
                    if tk in tickers_to_plot:
                        plot_dict[i * len(self.xcats) + j] = {
                            "X": "real_date",
                            "Y": [tk],
                        }

        if len(plot_dict) == 0:
            raise ValueError("Unable to resolve plot settings.")

        return self._cart_plot(
            plot_dict=plot_dict,
            grid_dim=grid_dim,
            plot_func=plt.plot,
            plot_func_args=None,
            figsize=figsize,
            # title arguments
            title=title,
            title_fontsize=title_fontsize,
            title_xadjust=title_xadjust,
            title_yadjust=title_yadjust,
            # subplot axis arguments
            ax_grid=ax_grid,
            ax_hline=ax_hline,
            ax_hline_val=ax_hline_val,
            ax_vline=ax_vline,
            ax_vline_val=ax_vline_val,
            x_axis_label=x_axis_label,
            y_axis_label=y_axis_label,
            axis_fontsize=axis_fontsize,
            # subplot arguments
            facet_size=facet_size,
            facet_titles=facet_titles,
            facet_title_fontsize=facet_title_fontsize,
            facet_title_xadjust=facet_title_xadjust,
            facet_title_yadjust=facet_title_yadjust,
            facet_xlabel=facet_xlabel,
            facet_ylabel=facet_ylabel,
            facet_label_fontsize=facet_label_fontsize,
            # legend arguments
            legend=legend,
            legend_labels=legend_labels,
            legend_loc=legend_loc,
            legend_ncol=legend_ncol,
            legend_bbox_to_anchor=legend_bbox_to_anchor,
            legend_frame=legend_frame,
            # return args
            show=show,
            save_to_file=save_to_file,
            dpi=dpi,
            return_figure=return_figure,
            *args,
            **kwargs,
        )


if __name__ == "__main__":
    # from macrosynergy.visuals import FacetPlot
    from macrosynergy.management.simulate_quantamental_data import make_test_df

    cids: List[str] = [
        "USD",
        "EUR",
        "GBP",
        "AUD",
        "CAD",
        "JPY",
        "CHF",
        "NZD",
        "SEK",
        "NOK",
        "DKK",
        "INR",
    ]
    xcats: List[str] = [
        "FXXR",
        "EQXR",
        "RIR",
        "IR",
        "REER",
        "CPI",
        "PPI",
        "M2",
        "M1",
        "M0",
        "FXVOL",
        "FX",
    ]
    sel_cids: List[str] = ["USD", "EUR", "GBP"]
    sel_xcats: List[str] = ["FXXR", "EQXR", "RIR", "IR"]

    df: pd.DataFrame = make_test_df(
        cids=cids,
        xcats=xcats,
        start_date="2000-01-01",
    )

    # FacetPlot(df).lineplot()
    import time

    print("From same object:")
    sdkf: float = time.time()

    with FacetPlot(df, cids=sel_cids, xcats=[xcats[1]]) as fp:
        fp.lineplot(
            cid_grid=True,
            ncols=3,
            cids_mean=True,
            title="Test Title",
        )
