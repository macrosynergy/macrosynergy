"""
A subclass inheriting from `macrosynergy.visuals.plotter.Plotter`,
designed to render a facet plot containing any number of subplots.
Given that the class allows returning a `matplotlib.pyplot.Figure`,
one can easily add any number of subplots, even the FacetPlot itself:
effectively allowing for a recursive facet plot.

::docs::FacetPlot::sort_first::
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

from macrosynergy.visuals.plotter import Plotter
from macrosynergy.visuals.common import Numeric, NoneType


def _get_square_grid(
    num_plots: int,
) -> Tuple[int, int]:
    """
    Given the number of plots, return a tuple of grid dimensions
    that is closest to a square grid.

    Parameters
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

    # def scatterplot(
    #     self,
    #     # plot arguments
    #     cids: Optional[List[str]] = None,
    #     xcats: Optional[List[str]] = None,
    #     metric: Optional[str] = None,
    #     # fig arguments
    #     figsize: Tuple[Numeric, Numeric] = (16.0, 9.0),
    #     ncols: int = 3,
    #     attempt_square: bool = False,
    #     # xcats_mean: bool = False,
    #     # title arguments
    #     title: Optional[str] = None,
    #     title_fontsize: int = 20,
    #     title_xadjust: Optional[Numeric] = None,
    #     title_yadjust: Optional[Numeric] = None,
    #     # subplot axis arguments
    #     ax_grid: bool = True,
    #     ax_hline: bool = False,
    #     ax_hline_val: Numeric = 0.0,
    #     ax_vline: bool = False,
    #     ax_vline_val: Numeric = 0.0,
    #     x_axis_label: Optional[str] = None,
    #     y_axis_label: Optional[str] = None,
    #     axis_fontsize: int = 12,
    #     # subplot arguments
    #     facet_size: Optional[Tuple[Numeric, Numeric]] = None,
    #     facet_titles: Optional[List[str]] = None,
    #     facet_title_fontsize: int = 12,
    #     facet_title_xadjust: Numeric = 0.5,
    #     facet_title_yadjust: Numeric = 1.0,
    #     facet_xlabel: Optional[str] = None,
    #     facet_ylabel: Optional[str] = None,
    #     facet_label_fontsize: int = 12,
    #     # legend arguments
    #     legend: bool = True,
    #     legend_labels: Optional[List[str]] = None,
    #     legend_loc: str = "upper center",
    #     legend_ncol: int = 1,
    #     legend_bbox_to_anchor: Optional[Tuple[Numeric, Numeric]] = None,  # (1.0, 0.5),
    #     legend_frame: bool = True,
    #     # return args
    #     show: bool = True,
    #     save_to_file: Optional[str] = None,
    #     dpi: int = 300,
    #     return_figure: bool = False,
    #     plot_func_args: Dict[str, Any] = {},
    #     *args,
    #     **kwargs,
    # ):
    #     """
    #     **NOT IMPLEMENTED YET**

    #     Showing a FacetPlot composed of scatter plots from the data available in the
    #     `FacetPlot` object after initialization.
    #     """
    #     raise NotImplementedError("Scatterplot not implemented yet.")
    #     if metric is None:
    #         metric: str = self.metrics[0]

    #     _cids: List[str] = self.cids if cids is None else cids
    #     _xcats: List[str] = self.xcats if xcats is None else xcats

    #     ...

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
        # xcats_mean: bool = False,
        # title arguments
        figsize: Tuple[Numeric, Numeric] = (16.0, 9.0),
        title: Optional[str] = None,
        title_fontsize: int = 22,
        title_xadjust: Optional[Numeric] = None,
        title_yadjust: Optional[Numeric] = None,
        # subplot axis arguments
        ax_grid: bool = False,
        ax_hline: Optional[Numeric] = 0.0,
        ax_vline: Optional[str] = None,
        x_axis_label: Optional[str] = None,
        y_axis_label: Optional[str] = None,
        axis_fontsize: int = 12,
        # subplot arguments
        facet_size: Optional[Tuple[Numeric, Numeric]] = None,
        facet_titles: Optional[List[str]] = None,
        facet_title_fontsize: int = 14,
        facet_title_xadjust: Numeric = 0.5,
        facet_title_yadjust: Numeric = 1.0,
        facet_xlabel: Optional[str] = None,
        facet_ylabel: Optional[str] = None,
        # legend arguments
        legend: bool = True,
        legend_labels: Optional[List[str]] = None,
        legend_loc: Optional[str] = "lower center",
        legend_fontsize: int = 12,
        legend_ncol: int = 1,
        legend_bbox_to_anchor: Optional[Tuple[Numeric, Numeric]] = None,  # (1.0, 0.5),
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
        :param <bool> cids_mean: Used with `cid_grid` with a single `xcat`. If `True`,
            the mean of all `cids` for that `xcat` will be plotted on all charts. If `False`,
            only the specified `cids` will be plotted. Default is `False`.
        :param <str> compare_series: Used with `cid_grid` with a single `xcat`. If
            specified, the series specified will be plotted in each facet, as a red dashed
            line. This is useful for comparing a single series, such as a benchmark/average.
            Ensure that the comparison series is in the dataframe, and not filtered out when
            initializing the `FacetPlot` object. Default is `None`. NB: `compare_series`
            can only be used when the series is not removed by `reduce_df()` in the object
            initialization.
        :param <bool> share_y: whether to share the y-axis across all plots. Default is
            `True`.
        :param <bool> share_x: whether to share the x-axis across all plots. Default is
            `True`.
        :param <Tuple[Numeric, Numeric]> figsize: a tuple of floats specifying the width and
            height of the figure. Default is `(16.0, 9.0)`.
        :param <str> title: the title of the plot. Default is `None`.
        :param <int> title_fontsize: the font size of the title. Default is `20`.
        :param <float> title_xadjust: the x-adjustment of the title. Default is `None`.
        :param <float> title_yadjust: the y-adjustment of the title. Default is `None`.
        :param <bool> ax_grid: whether to show the grid on the axes, applied to all plots.
            Default is `True`.
        :param <Numeric> ax_hline: the value of the horizontal line on the axes, applied
            to all plots. Default is `None`, meaning no horizontal line will be shown.
        :param <str> ax_vline: the value of the vertical line on the axes, applied
            to all plots. The value must be a ISO-8601 formatted date-string.
            Default is `None`, meaning no vertical line will be shown.
        :param <str> x_axis_label: the label for the x-axis. Default is `None`.
        :param <str> y_axis_label: the label for the y-axis. Default is `None`.
        :param <int> axis_fontsize: the font size of the axis labels. Default is `12`.
        :param <Tuple[Numeric, Numeric]> facet_size: a tuple of floats specifying the width
            and height of each facet. Default is `None`, meaning the facet size will be
            inferred from the `figsize` argument. If specified, the `figsize` argument
            will be ignored and the figure size will be inferred from the dimensions of
            the facet grid and the facet size.
        :param <List[str]> facet_titles: a list of strings specifying the titles of each
            facet. Default is `None`, meaning all facets will have the full `ticker`,
            `cid`, or `xcat` as the title. If no `facet_titles` are required,
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
        :param <bool> legend: Show the legend. Default is `True`. When using `cid_xcat_grid`,
            the legend will not be shown as it is redundant.
        :param <list> legend_labels: Labels for the legend. Default is `None`,
            meaning a list identifying the various `cids`/`xcats` will be used.
        :param <str> legend_loc: Location of the legend. Default is `center left`.
            See https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.legend.html
            for more information.
        :param <int> legend_fontsize: Font size of the legend. Default is `12`.
        :param <int> legend_ncol: Number of columns in the legend. Default is `1`.
        :param <tuple> legend_bbox_to_anchor: Bounding box for the legend. Default is
            `(1.0, 0.5)`.
        :param <bool> legend_frame: Show the legend frame. Default is `True`.
        :param <bool> show: Show the plot. Default is `True`.
        :param <str> save_to_file: Save the plot to a file. Default is `None`.
        :param <int> dpi: DPI of the saved image. Default is `300`.
        :param <bool> return_figure: Return the figure object. Default is `False`.
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

        tickers_to_plot: List[str] = (
            (self.df["cid"] + "_" + self.df["xcat"]).unique().tolist()
        )

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
                # needs to be "flipped" twice, as facet_titles need to be complementary to the legend labels
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

                plot_dict[i]: Dict[str, Union[str, List[str]]] = {
                    "X": "real_date",
                    "Y": tks + ([compare_series] if compare_series else []),
                    "title": facet_titles[i],
                }

        if cid_xcat_grid:
            # NB : legend goes away in cid_xcat_grid
            legend: bool = False

            for i, cid in enumerate(_cids):
                for j, xcat in enumerate(_xcats):
                    tk: str = "_".join([cid, xcat])
                    plot_dict[i * len(_xcats) + j]: Dict[str, List[str]] = {
                        "X": "real_date",
                        "Y": [tk],
                        "title": tk,
                    }

        if len(plot_dict) == 0:
            raise ValueError("Unable to resolve plot settings.")

        # sort by the title
        _plot_dict: Dict[str, Dict[str, Union[str, List[str]]]] = dict(
            sorted(plot_dict.items(), key=lambda x: x[1]["title"])
        )
        _plot_dict: Dict[str, Dict[str, Union[str, List[str]]]] = {
            i: ditem for i, ditem in enumerate(_plot_dict.values())
        }
        plot_dict: Dict[str, Dict[str, Union[str, List[str]]]] = _plot_dict.copy()

        ##############################
        # Plotting
        ##############################

        fig = plt.figure(figsize=figsize)

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
                suptitle.get_window_extent().width,
                suptitle.get_window_extent().height,
            )
            title_newline_adjust: float = title_fontsize / 500
            if title_yadjust is not None and title_yadjust != 1.0:
                title_newline_adjust = abs(title_yadjust - 1.0)
            # count the number of newlines in the title

            title_height: float = (title.count("\n") + 1) * title_newline_adjust

            re_adj[3] = (
                re_adj[3] - title_height + fig_height / fig.get_window_extent().height
            )

        axs: Union[np.ndarray, plt.Axes] = outer_gs.subplots(
            sharex=share_x,
            sharey=share_y,
        )
        if not isinstance(axs, np.ndarray):
            axs: np.ndarray = np.array([axs])
        ax_list: List[plt.Axes] = axs.flatten().tolist()
        for i, plt_dct, ax_i in zip(plot_dict.keys(), plot_dict.values(), ax_list):
            if plt_dct["X"] != "real_date":
                raise NotImplementedError(
                    "Only `real_date` is supported for the X axis."
                )

            is_empty_plot: bool = False

            for iy, y in enumerate(plt_dct["Y"]):
                # split on the first underscore
                cidx, xcatx = str(y).split("_", 1)
                sel_bools: pd.Series = (self.df["cid"] == cidx) & (
                    self.df["xcat"] == xcatx
                )
                is_empty_plot = is_empty_plot and not sel_bools.any()
                plot_func_args: Dict = {}

                # lineplot
                if legend_color_map:
                    plot_func_args["color"] = legend_color_map[
                        xcatx if cid_grid else cidx
                    ]
                if y == compare_series:
                    plot_func_args["color"] = "red"
                    plot_func_args["linestyle"] = "--"

                ax_i.plot(
                    self.df[sel_bools][plt_dct["X"]].reset_index(drop=True).tolist(),
                    self.df[sel_bools][metric].reset_index(drop=True).tolist(),
                    **plot_func_args,
                    **kwargs,
                )

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
                    # this is the left column, and it gets a axis ylabel of the cid
                    ax_i.set_ylabel(
                        plt_dct["Y"][0].split("_", 1)[0],
                        fontsize=axis_fontsize,
                    )

            # if it's an empty plot, remove the axis labels and ticks
            if is_empty_plot:
                ax_i.set_xticklabels([])
                ax_i.set_yticklabels([])
                ax_i.set_xticks([])
                ax_i.set_yticks([])
            else:
                if ax_grid:
                    ax_i.grid(axis="both", linestyle="--", alpha=0.5)
                if ax_hline is not None:
                    ax_i.axhline(ax_hline, color="black", linestyle="--")
                if ax_vline is not None:
                    # ax_i.axvline(ax_vline, color="black", linestyle="--")
                    raise NotImplementedError(
                        "Vertical axis lines are not supported at this time."
                    )

        # if there are more axes than ax_i, remove them
        for ax in ax_list[len(plot_dict) :]:
            fig.delaxes(ax)

        # re_adj: List[float] = (0, 0, 0, 0)
        if legend:
            if "lower" in legend_loc:
                if legend_ncol < grid_dim[0]:
                    legend_ncol: int = grid_dim[0]

            leg = fig.legend(
                labels=legend_labels,
                loc=legend_loc,
                fontsize=legend_fontsize,
                ncol=legend_ncol,
                bbox_to_anchor=legend_bbox_to_anchor,
                frameon=legend_frame,
            )

            leg_width, leg_height = (
                leg.get_window_extent().width,
                leg.get_window_extent().height,
            )
            fig_width, fig_height = (
                fig.get_window_extent().width,
                fig.get_window_extent().height,
            )

            if "lower" in legend_loc:
                re_adj[1] = re_adj[1] + leg_height / fig_height
            if "upper" in legend_loc:
                re_adj[3] = re_adj[3] - leg_height / fig_height
            if "left" in legend_loc:
                re_adj[0] = re_adj[0] + leg_width / fig_width
            if "right" in legend_loc:
                re_adj[2] = re_adj[2] - leg_width / fig_width

        outer_gs.tight_layout(fig, rect=re_adj)

        if save_to_file is not None:
            fig.savefig(save_to_file, dpi=dpi)  # , bbox_inches="tight")

        if show:
            plt.show()

        if return_figure:
            return fig


if __name__ == "__main__":
    # from macrosynergy.visuals import FacetPlot
    from macrosynergy.management.simulate_quantamental_data import make_test_df

    import time

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
    all_cids: List[str] = list(set(cids_A + cids_B + cids_C))
    all_xcats: List[str] = list(set(xcats_A + xcats_B + xcats_C + xcats_D))

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
            title="Test Title with a very long title to see how it looks, \n and a new line - why not?",
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
        )
        fp.lineplot(
            cids=cids_C,
            xcats=xcats_D,
            cid_xcat_grid=True,
            title="Another test title",
            show=True,
        )

    print(f"Time taken: {time.time() - timer_start}")
