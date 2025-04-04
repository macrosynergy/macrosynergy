"""
A subclass inheriting from `macrosynergy.visuals.plotter.Plotter`, designed to render a
facet plot containing any number of subplots. Given that the class allows returning a
`matplotlib.pyplot.Figure`, one can easily add any number of subplots, even the
FacetPlot itself: effectively allowing for a recursive facet plot.
"""

from numbers import Number
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from statsmodels.graphics.tsaplots import plot_acf

from macrosynergy.management.types import QuantamentalDataFrame
from macrosynergy.visuals.plotter import Plotter


def _get_square_grid(
    num_plots: int,
) -> Tuple[int, int]:
    """
    Given the number of plots, return a tuple of grid dimensions that is closest to a
    square grid.

    Parameters
    ----------
    num_plots : int
        Number of plots.

    Returns
    -------
    Tuple[int, int]
        Tuple of grid dimensions.
    """

    sqrt_num_plots: float = np.ceil(np.sqrt(num_plots))
    grid_dim: Tuple[int, int] = (int(sqrt_num_plots), int(sqrt_num_plots))
    gd_copy: Tuple[int, int] = grid_dim
    # the number of plots is less than grid_dim[0] * grid_dim[1],
    # so iteratively try and reduce the row and column dimensions until sweet spot is
    # found.
    while 0 != 1:
        if gd_copy[0] < gd_copy[1]:
            gd_copy: Tuple[int, int] = (gd_copy[0], gd_copy[1] - 1)
        else:
            gd_copy: Tuple[int, int] = (gd_copy[0] - 1, gd_copy[1])

        # if gd_copy[0] * gd_copy[1] is more than num_plots, copy to grid_dim. if smaller,
        # break.
        if gd_copy[0] * gd_copy[1] >= num_plots:
            grid_dim: Tuple[int, int] = gd_copy
        else:
            break

    return grid_dim


class FacetPlot(Plotter):
    """
    Class for rendering a facet plot containing any number of subplots. Inherits from
    `macrosynergy.visuals.plotter.Plotter`.

    Parameters
    ----------
    df : ~pandas.DataFrame
        A DataFrame with the following columns: 'cid', 'xcat', 'real_date', and at least
        one metric from - 'value', 'grading', 'eop_lag', or 'mop_lag'.
    cids : List[str]
        A list of cids to select from the DataFrame. If None, all cids are selected.
    xcats : List[str]
        A list of xcats to select from the DataFrame. If None, all xcats are selected.
    metrics : List[str]
        A list of metrics to select from the DataFrame. If None, all metrics are
        selected.
    intersect : bool
        if True only retains cids that are available for all xcats. Default is False.
    tickers : List[str]
        A list of tickers to select from the DataFrame. If None, all tickers are
        selected.
    blacklist : dict
        cross-sections with date ranges that should be excluded from the data frame. If
        one cross-section has several blacklist periods append numbers to the cross-section
        code.
    start : str
        ISO-8601 formatted date string. Select data from this date onwards. If None, all
        dates are selected.
    end : str
        ISO-8601 formatted date string. Select data up to and including this date. If
        None, all dates are selected.
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
        _cids: Optional[List[str]] = None,
        _xcats: Optional[List[str]] = None,
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

        if _cids is None:
            _cids: List[str] = self.cids
        if _xcats is None:
            _xcats: List[str] = self.xcats

        if attempt_square:
            target_var: str = tickers
            if cid_grid:
                target_var: List[str] = _cids
            elif xcat_grid:
                target_var: List[str] = _xcats
            return _get_square_grid(num_plots=len(target_var))

        if cid_grid or xcat_grid:
            tks: List[str] = _cids if cid_grid else _xcats
            return self._get_grid_dim(
                tickers=tks,
                ncols=ncols,
                attempt_square=attempt_square,
                _cids=_cids,
                _xcats=_xcats,
            )

        if cid_xcat_grid:
            return (len(_xcats), len(_cids))

        if ncols is not None:
            return (
                ncols,
                int(np.ceil(len(tickers) / ncols)),
            )

        raise ValueError("Unable to infer grid dimensions.")

    def lineplot(
        self,
        # plot arguments
        cids: Optional[List[str]] = None,
        xcats: Optional[List[str]] = None,
        metric: Optional[str] = None,
        ncols: int = 3,
        attempt_square: bool = False,
        cid_grid: bool = False,
        xcat_grid: bool = False,
        cid_xcat_grid: bool = False,
        grid_dim: Optional[Tuple[int, int]] = None,
        compare_series: Optional[str] = None,
        share_y: bool = False,
        share_x: bool = False,
        interpolate: bool = False,
        # xcats_mean: bool = False,
        # title arguments
        figsize: Tuple[Number, Number] = (16.0, 9.0),
        title: Optional[str] = None,
        title_fontsize: int = 22,
        title_xadjust: Optional[Number] = None,
        title_yadjust: Optional[Number] = None,
        # subplot axis arguments
        ax_grid: bool = False,
        ax_hline: Optional[Number] = 0.0,
        ax_vline: Optional[str] = None,
        x_axis_label: Optional[str] = None,
        y_axis_label: Optional[str] = None,
        axis_fontsize: int = 12,
        # subplot arguments
        facet_size: Optional[Tuple[Number, Number]] = None,
        facet_titles: Optional[Union[List[str], Dict[str, str]]] = None,
        facet_title_fontsize: int = 14,
        facet_title_xadjust: Number = 0.5,
        facet_title_yadjust: Number = 1.0,
        facet_xlabel: Optional[str] = None,
        facet_ylabel: Optional[str] = None,
        # legend arguments
        legend: bool = True,
        legend_labels: Optional[List[str]] = None,
        legend_loc: Optional[str] = "lower center",
        legend_fontsize: int = 12,
        legend_ncol: int = 1,
        legend_bbox_to_anchor: Optional[Tuple[Number, Number]] = None,  # (1.0, 0.5),
        legend_frame: bool = True,
        # return args
        show: bool = True,
        save_to_file: Optional[str] = None,
        dpi: int = 300,
        return_figure: bool = False,
        plot_func=None,
        plot_func_kwargs: Dict[str, Any] = {},
        *args,
        **kwargs,
    ):
        """
        Showing a FacetPlot composed of linear plots from the data available in the
        `FacetPlot` object after initialization. Passing any of the arguments used to
        initialize the `FacetPlot` object will cause the object to be re-initialized
        with the new arguments, and the plot will be rendered from the new object state.

        Parameters
        ----------
        cids : List[str]
            A list of `cids` to select from the DataFrame. If None, the `cids` are selected
            from the object initialization, or all if were specified.
        xcats : List[str]
            A list of `xcats` to select from the DataFrame. If None, all `xcats` are selected
            from the object initialization, or all if were specified.
        metric : str
            the metric to plot. Default is the first metric in the DataFrame.
        ncols : int
            number of columns in the grid. Default is 3.
        attempt_square : bool
            attempt to make the facet grid square. Ignores `ncols` when `True`. Default
            is `False`.
        cid_grid : bool
            Create the facet grid such that each facet is a `cid`.
        xcat_grid : bool
            Create the facet grid such that each facet is an `xcat`.
        cid_xcat_grid : bool
            Create the facet grid such that each facet is an individual ticker. Each
            "row" contains plots for the same `cid`, and each "column" would contain plots
            for the same `xcat`. Therefore, this mode does not respect the `ncols` or
            `attempt_square` arguments.
        grid_dim : Tuple[int, int]
            A tuple of integers specifying the number of rows and columns in the grid.
        compare_series : str
            Used with `cid_grid` with a single `xcat`. If specified, the series
            specified will be plotted in each facet, as a red dashed line. This is useful
            for comparing a single series, such as a benchmark/average. Ensure that the
            comparison series is in the dataframe, and not filtered out when initializing
            the `FacetPlot` object. Default is `None`.
        share_y : bool
            whether to share the y-axis across all plots. Default is `True`.
        share_x : bool
            whether to share the x-axis across all plots. Default is `True`.
        interpolate : bool
            if `True`, gaps in the time series will be interpolated. Default is `False`.
        figsize : Tuple[Number, Number]
            a tuple of floats specifying the width and height of the figure. Default is
            `(16.0, 9.0)`.
        title : str
            the title of the plot. Default is `None`.
        title_fontsize : int
            the font size of the title. Default is `20`.
        title_xadjust : float
            the x-adjustment of the title. Default is `None`.
        title_yadjust : float
            the y-adjustment of the title. Default is `None`.
        ax_grid : bool
            whether to show the grid on the axes, applied to all plots. Default is
            `True`.
        ax_hline : Number
            the value of the horizontal line on the axes, applied to all plots. Default
            is `None`, meaning no horizontal line will be shown.
        ax_vline : str
            the value of the vertical line on the axes, applied to all plots. The value
            must be a ISO-8601 formatted date-string. Default is `None`, meaning no vertical
            line will be shown.
        x_axis_label : str
            the label for the x-axis. Default is `None`.
        y_axis_label : str
            the label for the y-axis. Default is `None`.
        axis_fontsize : int
            the font size of the axis labels. Default is `12`.
        facet_size : Tuple[Number, Number]
            a tuple of floats specifying the width and height of each facet. Default is
            `None`, meaning the facet size will be inferred from the `figsize` argument. If
            specified, the `figsize` argument will be ignored and the figure size will be
            inferred from the dimensions of the facet grid and the facet size.
        facet_titles : List[str]
            a list of strings specifying the titles of each facet. Default is `None`,
            meaning all facets will have the full `ticker`, `cid`, or `xcat` as the title.
            If no `facet_titles` are required, pass an empty list - `facet_titles=[]`.
        facet_title_fontsize : int
            the font size of the facet titles. Default is `12`.
        facet_title_xadjust : float
            the x-adjustment of the facet titles. Default is `None`.
        facet_title_yadjust : float
            the y-adjustment of the facet titles. Default is `None`.
        facet_xlabel : bool
            The label to be used as the axis-label/title for the x-axis of each facet.
            Default is `None`, meaning no label will be shown.
        facet_ylabel : bool
            The label to be used as the axis-label/title for the y-axis of each facet.
            Default is `None`, meaning no label will be shown.
        facet_label_fontsize : int
            the font size of the facet labels. Default is `12`.
        legend : bool
            Show the legend. Default is `True`. When using `cid_xcat_grid`, the legend
            will not be shown as it is redundant.
        legend_labels : list
            Labels for the legend. Default is `None`, meaning a list identifying the
            various `cids`/`xcats` will be used.
        legend_loc : str
            Location of the legend. Default is `center left`. See
            https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.legend.html for more
            information.
        legend_fontsize : int
            Font size of the legend. Default is `12`.
        legend_ncol : int
            Number of columns in the legend. Default is `1`.
        legend_bbox_to_anchor : tuple
            Bounding box for the legend. Default is `(1.0, 0.5)`.
        legend_frame : bool
            Show the legend frame. Default is `True`.
        show : bool
            Show the plot. Default is `True`.
        save_to_file : str
            Save the plot to a file. Default is `None`.
        dpi : int
            DPI of the saved image. Default is `300`.
        return_figure : bool
            Return the figure object. Default is `False`.
        plot_func : callable
            The plotting function to use. The function must have parameters real_dates, values,
            and ax, and it will be passed these.Default is None.
        plot_func_kwargs : dict
            Additional arguments to pass to the plotting function. Default is `{}`.


        .. note::
            `compare_series` can only be used when the series is not removed by
            `reduce_df()` in the object initialization.

        .. note::
            `facet_titles` and `legend` are overridden in this mode.
        """

        comp_series_flag: bool = False

        if compare_series:
            if compare_series not in set(
                (kwargs["df"] if "df" in kwargs else self.df)[["cid", "xcat"]]
                .drop_duplicates()
                .apply(lambda x: "_".join(x), axis=1)
                .tolist()
            ):
                comp_series_flag: bool = True

        if metric is None:
            metric: str = self.metrics[0]

        _cids: List[str] = self.cids if cids is None else cids
        _xcats: List[str] = self.xcats if xcats is None else xcats

        tickers_to_plot: List[str] = QuantamentalDataFrame(self.df).list_tickers()

        _tk: List[str] = tickers_to_plot.copy()
        if compare_series:
            _tk.remove(compare_series)
        grid_dim: Tuple[int, int] = self._get_grid_dim(
            tickers=_tk,
            ncols=ncols,
            attempt_square=attempt_square,
            cid_grid=cid_grid,
            xcat_grid=xcat_grid,
            cid_xcat_grid=cid_xcat_grid,
            _cids=cids,
            _xcats=xcats,
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

        plot_dict: Dict[str, Dict[str, Union[str, List[str]]]] = {}

        colormap = plt.cm.tab10
        legend_color_map: Optional[Dict[str, str]] = None

        if facet_titles is None:
            if cid_grid:
                facet_titles: List[str] = _cids
            elif xcat_grid:
                facet_titles: List[str] = _xcats
            elif cid_xcat_grid:
                # cid_xcat_grid facets only make sense if they have cid_xcat as the title
                legend: bool = False
            else:
                facet_titles: List[str] = tickers_to_plot
        elif isinstance(facet_titles, dict):
            if cid_grid:
                if not all([x in facet_titles.keys() for x in _cids]):
                    raise ValueError(
                        "Facet titles dictionary does not contain all cids."
                    )
            elif xcat_grid:
                if not all([x in facet_titles.keys() for x in _xcats]):
                    raise ValueError(
                        "Facet titles dictionary does not contain all xcats."
                    )

        if not any([cid_grid, xcat_grid, cid_xcat_grid]):
            # each ticker gets its own plot
            for i, ticker in enumerate(tickers_to_plot):
                tks: List[str] = [ticker]
                if compare_series is not None:
                    tks.append(compare_series)
                plot_dict[i] = {
                    "X": "real_date",
                    "Y": tks,
                }
                # plot_dict[i] --> Dict[str, Union[str, List[str]]]

        if cid_grid or xcat_grid:
            # flipper handles resolution between cid_grid and xcat_grid for binary
            # variables
            flipper: bool = 1 if cid_grid else -1
            if facet_titles is None:
                # needs to be "flipped" twice, as facet_titles need to be complementary
                # to the legend labels
                facet_titles: List[str] = [_cids, _xcats][::flipper][0]
            if legend_labels is None:
                legend_labels: List[str] = [_xcats, _cids][::flipper][0]
            elif len(legend_labels) != (
                len([_xcats, _cids][::flipper][0]) + bool(compare_series)
            ):
                raise ValueError(
                    "The number of legend labels does not match the lines to plot."
                )

            if legend_color_map is None:
                legend_color_map: Dict[str, str] = {
                    x: colormap(i) for i, x in enumerate([_xcats, _cids][::flipper][0])
                }

            for i, fvar in enumerate([_cids, _xcats][::flipper][0]):
                tks: List[str] = [
                    "_".join([fvar, x][::flipper])
                    for x in ([_xcats, _cids][::flipper][0])
                ]

                if tks == [compare_series]:
                    continue

                if isinstance(facet_titles, dict):
                    facet_title = facet_titles[fvar]
                else:
                    facet_title = facet_titles[i]

                plot_dict[i] = {
                    "X": "real_date",
                    "Y": tks + ([compare_series] if compare_series else []),
                    "title": facet_title,  # facet_titles
                }
                # plot_dict[i] --> Dict[str, Union[str, List[str]]]

        if cid_xcat_grid:
            # NB : legend goes away in cid_xcat_grid
            legend: bool = False

            for i, cid in enumerate(_cids):
                for j, xcat in enumerate(_xcats):
                    tk: str = "_".join([cid, xcat])
                    plot_dict[i * len(_xcats) + j] = {
                        "X": "real_date",
                        "Y": [tk],
                        "title": tk,
                    }
                    # plot_dict[i * len(_xcats) + j] --> Dict[str, List[str]]

        if not any([cid_grid, xcat_grid, cid_xcat_grid]):
            for i, (key, plt_dct) in enumerate(plot_dict.items()):
                plt_dct["title"] = plt_dct["Y"][0] + " vs " + plt_dct["X"][0]

        if len(plot_dict) == 0:
            raise ValueError("Unable to resolve plot settings.")

        # sort by the title - only
        if all("title" in ditem.keys() for ditem in plot_dict.values()):
            _plot_dict: Dict[str, Dict[str, Union[str, List[str]]]] = {
                i: ditem for i, ditem in enumerate(plot_dict.values())
            }
            plot_dict: Dict[str, Dict[str, Union[str, List[str]]]] = _plot_dict.copy()

        ##############################
        # Plotting
        ##############################

        fig = plt.figure(figsize=figsize)
        renderer = fig.canvas.get_renderer()

        outer_gs: GridSpec = GridSpec(
            ncols=grid_dim[0],
            nrows=grid_dim[1],
            figure=fig,
        )

        # re_adj: List[float] = [Left, Bottom, Right, Top]
        re_adj: List[float] = [0, 0, 1, 1]

        suptitle = fig.suptitle(
            title,
            fontsize=title_fontsize,
            x=title_xadjust,
            y=title_yadjust,
        )

        if title is not None:
            # get the figure coordinates of the title
            fig_width, fig_height = (
                suptitle.get_window_extent(renderer=renderer).width,
                suptitle.get_window_extent(renderer=renderer).height,
            )
            title_newline_adjust: float = title_fontsize / 500
            if title_yadjust is not None and title_yadjust != 1.0:
                title_newline_adjust = abs(title_yadjust - 1.0)
            # count the number of newlines in the title

            title_height: float = (title.count("\n") + 1) * title_newline_adjust

            re_adj[3] = (
                re_adj[3]
                - title_height
                + fig_height / fig.get_window_extent(renderer=renderer).height
            )

        axs: Union[np.ndarray, plt.Axes] = outer_gs.subplots(
            sharex=share_x,
            sharey=False,
        )

        if not isinstance(axs, np.ndarray):
            axs = np.array([axs])
        ax_list = axs.flatten().tolist()

        self.df.set_index(["cid", "xcat"], inplace=True)
        self.df.sort_index(inplace=True)

        for i, (key, plt_dct) in enumerate(plot_dict.items()):
            ax_i = ax_list[i]

            if plt_dct["X"] != "real_date":
                raise NotImplementedError(
                    "Only `real_date` is supported for the X axis."
                )

            is_empty_plot = False

            if plot_func is not None:
                plot_func(df=self.df, plt_dict=plt_dct, ax=ax_i, **plot_func_kwargs)
                continue

            for iy, y in enumerate(plt_dct["Y"]):
                cidx, xcatx = str(y).split("_", 1)
                try:
                    selected_df = self.df.loc[cidx, xcatx]
                    is_valid_series = True
                except KeyError:
                    is_valid_series = False
                is_empty_plot = is_empty_plot and not is_valid_series

                if is_valid_series:
                    X = selected_df[plt_dct["X"]].values
                    Y = selected_df[metric].values
                else:
                    X, Y = [], []

                plot_func_args = {}
                if legend_color_map:
                    plot_func_args["color"] = legend_color_map.get(
                        xcatx if cid_grid else cidx
                    )

                if y == compare_series:
                    plot_func_args["color"] = "red"
                    plot_func_args["linestyle"] = "--"

                if not interpolate:
                    X, Y = self._insert_nans(X, Y)

                if plot_func is None:
                    ax_i.plot(X, Y, **plot_func_args)
                else:
                    plot_func(real_dates=X, values=Y, ax=ax_i, **plot_func_kwargs)

            if not cid_xcat_grid:
                if facet_titles:
                    ax_i.set_title(
                        plt_dct["title"],
                        fontsize=facet_title_fontsize,
                        x=facet_title_xadjust,
                        y=facet_title_yadjust,
                    )
                if x_axis_label is not None:
                    ax_i.set_xlabel(x_axis_label, fontsize=axis_fontsize)
                if y_axis_label is not None:
                    ax_i.set_ylabel(y_axis_label, fontsize=axis_fontsize)
            else:
                if i < grid_dim[0]:
                    ax_i.set_title(
                        plt_dct["Y"][0].split("_", 1)[1],
                        fontsize=axis_fontsize,
                    )
                if i % grid_dim[0] == 0:
                    ax_i.set_ylabel(
                        plt_dct["Y"][0].split("_", 1)[0],
                        fontsize=axis_fontsize,
                    )

            if is_empty_plot:
                ax_i.set_xticks([])
                ax_i.set_yticks([])
            else:
                if ax_grid:
                    ax_i.grid(axis="both", linestyle="--", alpha=0.5)
                if ax_hline is not None:
                    ax_i.axhline(ax_hline, color="black", linestyle="--")
                if ax_vline is not None:
                    raise NotImplementedError(
                        "Vertical axis lines are not supported at this time."
                    )

        self.df.reset_index(inplace=True)

        for ax in ax_list[len(plot_dict) :]:
            fig.delaxes(ax)
        ax_list = ax_list[: len(plot_dict)]

        if share_x:
            for target_ax in ax_list[(len(plot_dict) - grid_dim[0]) :]:
                target_ax.xaxis.set_tick_params(labelbottom=True)

        if share_y:
            ymins, ymaxs = zip(*(ax.get_ylim() for ax in ax_list))
            global_min, global_max = min(ymins), max(ymaxs)
            abs_max = max(abs(global_min), abs(global_max))
            global_min, global_max = -abs_max, abs_max
            for ax in ax_list:
                ax.set_ylim(global_min, global_max)

            width, height = grid_dim
            for idx, ax in enumerate(ax_list):
                col_idx = idx % width
                if col_idx != 0:
                    ax.tick_params(labelleft=False)

        if legend:
            if "lower" in legend_loc and legend_ncol < grid_dim[0]:
                legend_ncol = grid_dim[0]

            leg = fig.legend(
                labels=legend_labels,
                loc=legend_loc,
                fontsize=legend_fontsize,
                ncol=legend_ncol,
                bbox_to_anchor=legend_bbox_to_anchor,
                frameon=legend_frame,
            )

            leg_width, leg_height = (
                leg.get_window_extent(renderer=renderer).width,
                leg.get_window_extent(renderer=renderer).height,
            )
            fig_width, fig_height = (
                fig.get_window_extent(renderer=renderer).width,
                fig.get_window_extent(renderer=renderer).height,
            )

            if "lower" in legend_loc:
                re_adj[1] += leg_height / fig_height
            if "upper" in legend_loc:
                re_adj[3] -= leg_height / fig_height
            if "left" in legend_loc:
                re_adj[0] += leg_width / fig_width
            if "right" in legend_loc:
                re_adj[2] -= leg_width / fig_width

        outer_gs.tight_layout(fig, rect=re_adj)

        if save_to_file is not None:
            fig.savefig(save_to_file, dpi=dpi)

        if show:
            plt.show()

        if return_figure:
            return fig

    def _insert_nans(self, X: Union[np.ndarray, List], Y: Union[np.ndarray, List]):
        """
        Inserts a single NaN value in each inferred gap in the time series.

        Parameters
        ----------
        X : ~numpy.ndarray
            Array of dates.
        Y : ~numpy.ndarray
            Array of values.

        Returns
        -------
        Tuple[~numpy.ndarray, ~numpy.ndarray]
            Tuple of arrays with gaps filled.
        """

        if len(X) == 0 or len(Y) == 0:
            return X, Y

        df = pd.DataFrame({"X": X, "Y": Y})
        df["X"] = pd.to_datetime(df["X"])

        # Infer the most common frequency of the time series
        differences = df["X"].diff().dt.total_seconds().dropna().astype(int)
        frequency_seconds = differences.mode()[0]

        frequency = pd.to_timedelta(frequency_seconds, unit="s")

        # Ignore weekends if daily
        freq_threshold = max(frequency_seconds * 1.5, 60 * 60 * 24 * 3)
        gaps = differences > freq_threshold
        gap_positions = gaps[gaps].index.tolist()

        new_rows = []
        for pos in gap_positions:
            gap_date = df.loc[pos - 1, "X"] + frequency
            new_rows.append({"X": gap_date, "Y": np.nan})

        if new_rows:
            new_rows_df = pd.DataFrame(new_rows)
            df = (
                pd.concat([df, new_rows_df], ignore_index=True)
                .sort_values(by="X")
                .reset_index(drop=True)
            )

        return df.X, df.Y


if __name__ == "__main__":
    # from macrosynergy.visuals import FacetPlot
    import time

    from macrosynergy.management.simulate import make_test_df

    np.random.seed(0)

    cids_A: List[str] = ["AUD", "CAD", "EUR", "GBP", "USD"]
    cids_B: List[str] = ["CHF", "INR", "JPY", "NOK", "NZD", "SEK"]
    cids_C: List[str] = ["CHF", "EUR", "INR", "JPY", "NOK", "NZD", "SEK", "USD"]

    xcats_A: List[str] = [
        "CPIXFE_SA_P1M1ML12",
        "CPIXFE_SJA_P3M3ML3AR",
        "CPIXFE_SJA_P6M6ML6AR",
        "CPIXFE_SA_P1M1ML12_D1M1ML3",
        "CPIXFE_SA_P1M1ML12_D1M1ML3",
    ]
    xcats_B: List[str] = [
        "CPIC_SA_P1M1ML12",
        "CPIC_SJA_P3M3ML3AR",
        "CPIC_SJA_P6M6ML6AR",
        "CPIC_SA_P1M1ML12_D1M1ML3",
        "CPIH_SA_P1M1ML12",
        "EXALLOPENNESS_NSA_1YMA",
        "EXMOPENNESS_NSA_1YMA",
    ]
    xcats_C: List[str] = ["DU05YXR_NSA", "DU05YXR_VT10"]
    xcats_D: List[str] = [
        "FXXR_NSA",
        "EQXR_NSA",
        "FXTARGETED_NSA",
        "FXUNTRADABLE_NSA",
    ]
    all_cids: List[str] = sorted(list(set(cids_A + cids_B + cids_C)))
    all_xcats: List[str] = sorted(list(set(xcats_A + xcats_B + xcats_C + xcats_D)))

    df: pd.DataFrame = make_test_df(
        cids=all_cids,
        xcats=all_xcats,
    )

    # remove data for USD_FXXR_NSA and CHF _EQXR_NSA and _FXXR_NSA
    df: pd.DataFrame = df[
        ~((df["cid"] == "USD") & (df["xcat"] == "FXXR_NSA"))
    ].reset_index(drop=True)
    df: pd.DataFrame = df[
        ~((df["cid"] == "CHF") & (df["xcat"].isin(["EQXR_NSA", "FXXR_NSA"])))
    ].reset_index(drop=True)
    df: pd.DataFrame = df[
        ~((df["cid"] == "NOK") & (df["xcat"] == "FXUNTRADABLE_NSA"))
    ].reset_index(drop=True)

    timer_start: float = time.time()

    with FacetPlot(
        df,
    ) as fp:
        fp.lineplot(
            cids=cids_A,
            share_x=True,
            xcat_grid=True,
            ncols=2,
            title=(
                "Test Title with a very long title to see how it looks, \n and a "
                "new line - why not?"
            ),
            # save_to_file="test_0.png",
            ax_hline=75,
            show=True,
        )

        fp.lineplot(
            cids=cids_B,
            xcats=xcats_A,
            attempt_square=True,
            share_y=True,
            cid_grid=True,
            title="Another test title",
            # save_to_file="test_1.png",
            show=True,
            share_x=True,
        )
        fp.lineplot(
            cids=cids_C,
            xcats=xcats_D,
            cid_xcat_grid=True,
            title="Another test title",
            show=True,
        )

    print(f"Time taken: {time.time() - timer_start}")
