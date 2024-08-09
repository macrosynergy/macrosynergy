"""
A subclass inheriting from `macrosynergy.visuals.plotter.Plotter`,
designed to plot time series data as a heatmap.
"""

from numbers import Number
from typing import List, Optional, Tuple, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from seaborn.utils import relative_luminance

from macrosynergy.management.simulate import make_test_df
from macrosynergy.management.utils import downsample_df_on_real_date
from macrosynergy.visuals.plotter import Plotter


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

    def _plot(
        self,
        df: pd.DataFrame,
        figsize: Tuple[Number, Number] = (12, 8),
        x_axis_label: Optional[str] = None,
        y_axis_label: Optional[str] = None,
        axis_fontsize: int = 14,
        title: Optional[str] = None,
        title_fontsize: int = 22,
        title_xadjust: Number = 0.5,
        title_yadjust: Number = 1.0,
        vmin: Optional[Number] = None,
        vmax: Optional[Number] = None,
        show: bool = True,
        save_to_file: Optional[str] = None,
        dpi: int = 300,
        return_figure: bool = False,
        on_axis: Optional[plt.Axes] = None,
        max_xticks: int = 50,
        cmap: Optional[Union[str, mpl.colors.Colormap]] = None,
        rotate_xticks: Optional[Number] = 0,
        rotate_yticks: Optional[Number] = 0,
        show_tick_lines: Optional[bool] = True,
        show_colorbar: Optional[bool] = True,
        show_annotations: Optional[bool] = False,
        show_boundaries: Optional[bool] = False,
        annotation_fontsize: int = 14,
        tick_fontsize: int = 13,
        *args,
        **kwargs,
    ) -> Optional[plt.Figure]:
        """
        Plots a DataFrame as a heatmap with the columns along the x-axis and
        rows along the y-axis.

        Parameters
        :param <Tuple> figsize: tuple specifying the size of the figure. Default is
            (12, 8).
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
        :param <mpl.colors.Colormap> cmap: string or matplotlib Colormap object
            specifying the colormap of the plot.
        :param <int> rotate_xticks: number of degrees to rotate the tick labels on
            the x-axis. Default is zero.
        :param <int> rotate_yticks: number of degrees to rotate the tick labels on
            the y-axis. Default is zero.
        :param <bool> show_tick_lines: if True, lines are shown for ticks.
            Default is True.
        :param <bool> show_colorbar: if True, the colorbar is shown. Default is True.
        :param <bool> show_annotations: if True, annotations display the value of
            each cell. Default is False.
        :param <bool> show_boundaries: if True, cells are divided by a grid.
            Default is False.
        :param <int> annotation_fontsize: sets the font size of the annotations.
        :param <int> tick_fontsize: sets the font size of tick labels.
        """
        if on_axis:
            fig: plt.Figure = on_axis.get_figure()
            ax: plt.Axes = on_axis
        else:
            fig: plt.Figure
            ax: plt.Axes
            fig, ax = plt.subplots(figsize=figsize, layout="constrained")

        data = df.to_numpy()

        im = ax.imshow(
            data,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            aspect="auto",
            **kwargs,
        )

        xtick_labels = df.columns.to_list()
        ytick_labels = df.index.to_list()

        ax.set_xticks(np.arange(len(xtick_labels)), labels=xtick_labels)
        ax.set_yticks(np.arange(len(ytick_labels)), labels=ytick_labels)

        ax.set_xticklabels(
            xtick_labels,
            rotation=rotate_xticks,
            ha="center",
            minor=False,
        )
        ax.set_yticklabels(
            ytick_labels,
            rotation=rotate_yticks,
            ha="right",
            minor=False,
            rotation_mode="anchor",
        )

        ax.tick_params(axis="both", which="major", labelsize=tick_fontsize)

        if show_tick_lines:
            ax.tick_params(which="major", length=4, width=1, direction="out")

        plt.grid(False)

        if show_boundaries:
            ax.spines[:].set_visible(False)
            ax.set_xticks(np.arange(len(xtick_labels) + 1) - 0.5, minor=True)
            ax.set_yticks(np.arange(len(ytick_labels) + 1) - 0.5, minor=True)
            ax.grid(which="minor", color="w", linestyle="-", linewidth=3)
            ax.tick_params(which="minor", bottom=False, left=False)
        else:
            # Limits the number of ticks shown on the x-axis.
            ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=max_xticks - 1))

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

        if show_colorbar:
            ax.figure.colorbar(im, ax=ax)

        if show_annotations:
            data = np.around(data, decimals=1)
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    color = im.cmap(im.norm(im.get_array()))[i, j]
                    lum = relative_luminance(color)
                    text_color = ".15" if lum > 0.408 else "w"
                    text_kwargs = dict(
                        color=text_color,
                        ha="center",
                        va="center",
                        size=annotation_fontsize,
                    )
                    if not np.isnan(data[i, j]):
                        ax.text(j, i, data[i, j], **text_kwargs)

        ax.tick_params(axis="y", which="major", pad=8)

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

    def plot_metric(
        self,
        x_axis_column,
        y_axis_column,
        metric,
        xcats=None,
        cids=None,
        start=None,
        end=None,
        freq=None,
        agg="mean",
        figsize: Optional[Tuple[Number, Number]] = (12, 8),
        x_axis_label: Optional[str] = None,
        y_axis_label: Optional[str] = None,
        axis_fontsize: int = 14,
        title: Optional[str] = None,
        title_fontsize: int = 22,
        title_xadjust: Number = 0.5,
        title_yadjust: Number = 1.0,
        vmin: Optional[Number] = None,
        vmax: Optional[Number] = None,
        show: bool = True,
        save_to_file: Optional[str] = None,
        dpi: int = 300,
        return_figure: bool = False,
        on_axis: Optional[plt.Axes] = None,
        max_xticks: int = 50,
        cmap: Optional[Union[str, mpl.colors.Colormap]] = None,
        rotate_xticks: Optional[Number] = 0,
        rotate_yticks: Optional[Number] = 0,
        show_tick_lines: Optional[bool] = True,
        show_colorbar: Optional[bool] = True,
        show_annotations: Optional[bool] = False,
        show_boundaries: Optional[bool] = False,
        annotation_fontsize: int = 14,
        tick_fontsize: int = 13,
        *args,
        **kwargs,
    ):
        df = self.df.copy()
        if not xcats:
            xcats = self.xcats
        if not cids:
            cids = self.cids
        if not start:
            start = self.start
        if not end:
            end = self.end

        # Validation checks not covered by Plotter.
        if metric not in ["value", "eop_lag", "mop_lag", "grading"]:
            raise ValueError(
                "`metric` must be either 'eop_lag', 'mop_lag', 'grading', or 'value'"
            )

        if not isinstance(agg, str):
            raise TypeError("`agg` must be a string")
        else:
            agg: str = agg.lower()
            if agg not in ["mean", "median", "min", "max", "first", "last"]:
                raise ValueError(
                    "`agg` must be one of 'mean', 'median', 'min', 'max', 'first' or "
                    "'last'"
                )

        df = df[["xcat", "cid", "real_date", metric]]

        if freq:
            df: pd.DataFrame = downsample_df_on_real_date(
                df=df, groupby_columns=["cid", "xcat"], freq=freq, agg=agg
            )

        if "real_date" not in [x_axis_column, y_axis_column]:
            df = df.groupby(["xcat", "cid"]).mean().reset_index()
        else:
            df["real_date"] = df["real_date"].dt.strftime("%Y-%m-%d")

        vmax: float = max(1, df[metric].max())
        vmin: float = min(0, df[metric].min())

        df = df.pivot_table(index=y_axis_column, columns=x_axis_column, values=metric)

        if figsize is None:
            figsize = (
                max(df.shape[0] / 2, 15),
                max(1, df.shape[1] / 2),
            )
        elif isinstance(figsize, list):
            figsize = tuple(figsize)

        self._plot(
            df=df,
            figsize=figsize,
            x_axis_label=x_axis_label,
            y_axis_label=y_axis_label,
            axis_fontsize=axis_fontsize,
            title=title,
            title_fontsize=title_fontsize,
            title_xadjust=title_xadjust,
            title_yadjust=title_yadjust,
            vmin=vmin,
            vmax=vmax,
            show=show,
            save_to_file=save_to_file,
            dpi=dpi,
            return_figure=return_figure,
            on_axis=on_axis,
            max_xticks=max_xticks,
            cmap=cmap,
            rotate_xticks=rotate_xticks,
            rotate_yticks=rotate_yticks,
            show_tick_lines=show_tick_lines,
            show_colorbar=show_colorbar,
            show_annotations=show_annotations,
            show_boundaries=show_boundaries,
            annotation_fontsize=annotation_fontsize,
            tick_fontsize=tick_fontsize,
        )


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

    heatmap.df["real_date"] = heatmap.df["real_date"].dt.strftime("%Y-%m-%d")
    heatmap.df = heatmap.df.pivot_table(
        index="cid", columns="real_date", values="grading"
    )

    heatmap._plot(heatmap.df, title="abc", rotate_xticks=90)
