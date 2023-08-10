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

from macrosynergy.visuals.plotter import Plotter, NoneType, Numeric


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
    :param <bool> intersect: if True only retains cids that are available for
        all xcats. Default is False.
    :param <List[str]> tickers: A list of tickers to select from the DataFrame.
        If None, all tickers are selected.
    :param <dict> blacklist: cross-sections with date ranges that should be excluded from
        the data frame. If one cross-section has several blacklist periods append numbers
        to the cross-section code.
    :param <str> start: ISO-8601 formatted date string. Select data from
        this date onwards. If None, all dates are selected.
    :param <str> end: ISO-8601 formatted date string. Select data up to
        and including this date. If None, all dates are selected.

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

        if attempt_square:
            target_var: str = tickers
            if cid_grid:
                target_var: List[str] = self.cids
            elif xcat_grid:
                target_var: List[str] = self.xcats
            return _get_square_grid(num_plots=len(target_var))

        if cid_grid or xcat_grid:
            tks: List[str] = self.cids if cid_grid else self.xcats
            return self._get_grid_dim(
                tickers=tks, ncols=ncols, attempt_square=attempt_square
            )

        if cid_xcat_grid:
            return (len(self.xcats), len(self.cids))

        if ncols is not None:
            return (
                ncols,
                int(np.ceil(len(tickers) / ncols)),
            )

        raise ValueError("Unable to infer grid dimensions.")

    def _cart_plot(
        self,
        grid_dim: Tuple[int, int],
        plot_dict: Dict[int, Dict[str, Any]],
        metric: Optional[str] = None,
        plot_func: Callable = plt.plot,
        plot_func_args: Optional[List[Any]] = None,
        figsize: Tuple[Numeric, Numeric] = (16.0, 9.0),
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
        facet_size: Optional[Tuple[Numeric, Numeric]] = None,
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
        legend_loc: str = "lower right",
        legend_ncol: int = 1,
        legend_bbox_to_anchor: Optional[Tuple[Numeric, Numeric]] = None,
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
        # NB: nrows and ncols are flipped between mpl...GridSpec and mpl...figsize etc
        gs: GridSpec = GridSpec(nrows=grid_dim[1], ncols=grid_dim[0], figure=fig)

        # if title is not None:
        fig.suptitle(
            title,
            fontsize=title_fontsize,
            x=title_xadjust,
            y=title_yadjust,
        )

        if plot_func_args is None:
            plot_func_args: List = []

        for i, (plot_id, plt_dct) in enumerate(plot_dict.items()):
            # gs is a 2d grid with dims of tuple `grid_dim`
            ax: plt.Axes = fig.add_subplot(gs[i])
            if plt_dct["X"] != "real_date":
                raise NotImplementedError(
                    "Only `real_date` is supported for the X axis."
                )

            for y in plt_dct["Y"]:
                # split on the first underscore
                cidx, xcatx = str(y).split("_", 1)
                plot_func(
                    self.df[(self.df["cid"] == cidx) & (self.df["xcat"] == xcatx)][
                        str(plt_dct["X"])
                    ]
                    .reset_index(drop=True)
                    .tolist(),
                    self.df[(self.df["cid"] == cidx) & (self.df["xcat"] == xcatx)][
                        metric
                    ]
                    .reset_index(drop=True)
                    .tolist(),
                    *plot_func_args,
                    **kwargs,
                )

            if facet_titles:
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
            fig.legend(
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
        # df: Optional[pd.DataFrame] = None,
        # cids: Optional[List[str]] = None,
        # xcats: Optional[List[str]] = None,
        # intersect: bool = False,
        # tickers: Optional[List[str]] = None,
        # blacklist: Optional[Dict[str, List[str]]] = None,
        # start: Optional[str] = None,
        # end: Optional[str] = None,
        # plot arguments
        metric: Optional[str] = "value",
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
        figsize: Tuple[Numeric, Numeric] = (16.0, 9.0),
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
        facet_size: Optional[Tuple[Numeric, Numeric]] = None,
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
        Showing a FacetPlot composed of linear plots from the data available in the
        `FacetPlot` object after initialization.
        Passing any of the arguments used to initialize the `FacetPlot` object will cause
        the object to be re-initialized with the new arguments, and the plot will be rendered
        from the new object state.

        Parameters
        ----------
        :param <int> ncols: number of columns in the grid. Default is 3.
        :param <bool> attempt_square: attempt to make the facet grid square. Ignores
            `ncols` when `True`. Default is `False`.
        :param <bool> cid_grid: Create the facet grid such that each facet is a `cid`.
        :param <bool> xcat_grid: Create the facet grid such that each facet is an `xcat`.
        :param <bool> cid_xcat_grid: Create the facet grid such that each facet is an
            individual ticker. Each "row" contains plots for the same `cid`, and each
            "column" would contain plots for the same `xcat`. Therefore, this mode does
            not respect the `ncols` or `attempt_square` arguments.
            NB: `facet_titles` and `legend` are overridden in this mode.
        :param <Tuple[int, int]> grid_dim: a tuple of integers specifying the number of
            rows and columns in the facet grid. Default is `None`, meaning the grid
            dimensions will be inferred from the `ncols`/`attempt_square`/`cid_xcat_grid`
            arguments.
        :param <bool> cids_mean: Used with `cid_grid` with a single `xcat`. If `True`,
            the mean of all `cids` for that `xcat` will be plotted on all charts. If `False`,
            only the specified `cids` will be plotted. Default is `False`.
        :param <str> compare_series: Used with `cid_grid` with a single `xcat`. If
            specified, the series specified will be plotted in each facet. Ensure that
            the comparison series is in the dataframe, and not filtered out when
            initializing the `FacetPlot` object. Default is `None`.
        :param <Tuple[float, float]> figsize: a tuple of floats specifying the width and
            height of the figure. Default is `(16.0, 9.0)`.
        :param <str> title: the title of the plot. Default is `None`.
        :param <int> title_fontsize: the font size of the title. Default is `20`.
        :param <float> title_xadjust: the x-adjustment of the title. Default is `None`.
        :param <float> title_yadjust: the y-adjustment of the title. Default is `None`.
        :param <bool> ax_grid: whether to show the grid on the axes, applied to all plots.
            Default is `True`.
        :param <bool> ax_hline: whether to show a horizontal line on the axes, applied to
            all plots. Default is `False`.
        :param <float> ax_hline_val: the value of the horizontal line on the axes, applied
            to all plots. Default is `0.0`.
        :param <bool> ax_vline: whether to show a vertical line on the axes, applied to
            all plots. Default is `False`.
        :param <float> ax_vline_val: the value of the vertical line on the axes, applied
            to all plots. Default is `0.0`.
        :param <str> x_axis_label: the label for the x-axis. Default is `None`.
        :param <str> y_axis_label: the label for the y-axis. Default is `None`.
        :param <int> axis_fontsize: the font size of the axis labels. Default is `12`.
        :param <Tuple[float, float]> facet_size: a tuple of floats specifying the width
            and height of each facet. Default is `None`, meaning the facet size will be
            inferred from the `figsize` argument. If specified, the `figsize` argument
            will be ignored and the figure size will be inferred from the dimensions of
            the facet grid and the facet size.
        :param <List[str]> facet_titles: a list of strings specifying the titles of each
            facet. Default is `None`, meaning all facets will have the full
            `ticker`, `cid`, or `xcat` as the title. If no `facet_titles` are required,
            pass an empty list - `facet_titles=[]`.
        :param <int> facet_title_fontsize: the font size of the facet titles. Default is
            `12`.
        :param <float> facet_title_xadjust: the x-adjustment of the facet titles. Default
            is `None`.
        :param <float> facet_title_yadjust: the y-adjustment of the facet titles. Default
            is `None`.
        :param <bool> facet_xlabel: The label to be used as the axis-label/title for the
            x-axis of each facet. Default is `None`, meaning no label will be shown.
        :param <bool> facet_ylabel: The label to be used as the axis-label/title for the
            y-axis of each facet. Default is `None`, meaning no label will be shown.
        :param <int> facet_label_fontsize: the font size of the facet labels. Default is
            `12`.
        :param <bool> legend: whether or not to show a legend for the plot. Default is
            `True`. In `cid_xcat_grid` mode, the legend is never shown, as it would be
            redundant.
        :param <bool> legend: Show the legend. Default is `True`. When using
        `cid_xcat_grid`, the legend will not be shown as it is redundant.
        :param <list> legend_labels: Labels for the legend. Default is `None`,
            meaning a list identifying the various `cids`/`xcats` will be used.
        :param <str> legend_loc: Location of the legend. Default is `center left`.
            See https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.legend.html
            for more information.
        :param <int> legend_ncol: Number of columns in the legend. Default is `1`.
        :param <tuple> legend_bbox_to_anchor: Bounding box for the legend. Default is
            `(1.0, 0.5)`.
        :param <bool> legend_frame: Show the legend frame. Default is `True`.
        :param <bool> show: Show the plot. Default is `True`.
        :param <str> save_to_file: Save the plot to a file. Default is `None`.
        :param <int> dpi: DPI of the saved image. Default is `300`.
        :param <bool> return_figure: Return the figure object. Default is `False`.
        """
        # if any([arg is not None for arg in [df, cids, xcats, metric, tickers]]):
        #     # undesirable, as the state of the object will change kept for ease of use
        #     metrics: List[str] = [metric] if metric is not None else self.metrics
        #     metrics: Optional[List[str]] = metrics if metrics else None
        #     self.__init__(df=df if df is not None else self.df, cids=cids, xcats=xcats, metrics=metrics,
        #         intersect=intersect, tickers=tickers, blacklist=blacklist, start=start, end=end, )

        if any(
            [(argx in kwargs.keys()) for argx in ["df", "cids", "xcats", "tickers"]]
        ):
            # undesirable, as the state of the object will change kept for ease of use
            metrics: List[str] = [metric] if metric is not None else self.metrics
            metrics: Optional[List[str]] = metrics if metrics else None
            self.__init__(
                df=kwargs["df"] if "df" in kwargs.keys() else self.df,
                cids=kwargs["cids"],
                xcats=kwargs["xcats"],
                metrics=metrics,
                intersect=kwargs["intersect"],
                tickers=kwargs["tickers"],
                blacklist=kwargs["blacklist"],
                start=kwargs["start"],
                end=kwargs["end"],
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
                grid_dim[0] * facet_size[0] * 1.5,
                grid_dim[1] * facet_size[1] * 1.5,
            )
            # mul by 1.5 to account for the space taken up annotations, etc.
            # fig.tight_layout() cleans this up

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
                # create df_mean, with mean entries for each cid for every day, and xcat="mean"
                df_mean: pd.DataFrame = (
                    self.df.groupby(["real_date", "xcat"])[metric]
                    .mean(numeric_only=True)
                    .reset_index()
                )

                df_mean["cid"] = "mean"
                self.df: pd.DataFrame = pd.concat([self.df, df_mean], axis=0)

        plot_dict: Dict[str, Dict[str, Union[str, List[str]]]] = {}

        if facet_titles is None:
            if cid_grid:
                facet_titles: List[str] = self.cids
            elif xcat_grid:
                facet_titles: List[str] = self.xcats
            elif cid_xcat_grid:
                ...
                # cid_xcat_grid facets only make sense if they have cid_xcat as the title
            else:
                facet_titles: List[str] = tickers_to_plot

        if not any([cid_grid, xcat_grid, cid_xcat_grid]):
            # each ticker gets its own plot
            for i, ticker in enumerate(tickers_to_plot):
                tks: List[str] = [ticker]
                if compare_series is not None:
                    tks.append(compare_series)
                plot_dict[i]: Dict[str, Union[str, List[str]]] = {
                    "X": "real_date",
                    "Y": tks,
                }

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
                    # there is only one xcat if cid_grid is True
                    tks.append("_".join(["mean", self.xcats[0]]))

                plot_dict[i]: Dict[str, Union[str, List[str]]] = {
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

            for j, xcat in enumerate(self.xcats):
                for i, cid in enumerate(self.cids):
                    tk: str = "_".join([cid, xcat])
                    if tk in tickers_to_plot:
                        plot_dict[i * len(self.xcats) + j]: Dict[
                            str, Union[str, List[str]]
                        ] = {
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
            metric=metric,
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
        cids=list(set(cids) - set(sel_cids)),
        xcats=xcats,
        start_date="2000-01-01",
    )
    dfB: pd.DataFrame = make_test_df(
        cids=sel_cids,
        xcats=sel_xcats,
        start_date="2000-01-01",
    )
    df: pd.DataFrame = pd.concat([df, dfB], axis=0)

    import random

    random.seed(42)

    for cidx, xcatx in df[["cid", "xcat"]].drop_duplicates().values.tolist():
        # if random() > 0.5 multiply by random.random()*10
        _bools = (df["cid"] == cidx) & (df["xcat"] == xcatx)
        r = random.random()
        df.loc[_bools, "value"] = df.loc[_bools, "value"] * r

    # FacetPlot(df).lineplot()
    import time

    print("From same object:")
    timer_start: float = time.time()

    with FacetPlot(df, cids=sel_cids, xcats=sel_xcats) as fp:
        fp.lineplot(
            figsize=(5, 4),
            # cid_xcat_grid=True,
            cid_grid=True,
            title="Test Title",
            # save_to_file="test.png",
        )
    print(f"Time taken: {time.time() - timer_start}")
