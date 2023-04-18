"""
Providing a high level interface to simplify visual tasks involving matplotlib and seaborn.
"""

import matplotlib
import matplotlib.figure, matplotlib.axes
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Optional, List, Tuple, Dict, Union, Callable
import logging
from macrosynergy.management.utils import standardise_dataframe
from macrosynergy.management import reduce_df
import math


class Plotter(object):
    def __init__(
        self,
        df: pd.DataFrame,
        cids: List[str] = None,
        xcats: List[str] = None,
        metrics: List[str] = None,
        start_date: str = None,
        end_date: str = None,
    ):

        sdf: pd.DataFrame = df.copy()
        sdf = sdf[
            [
                "real_date",
                "cid",
                "xcat",
            ]
            + metrics
        ]
        if cids:
            sdf = sdf[sdf["cid"].isin(cids)]
        if xcats:
            sdf = sdf[sdf["xcat"].isin(xcats)]
        if start_date:
            sdf = sdf[sdf["real_date"] >= pd.to_datetime(start_date)]
        if end_date:
            sdf = sdf[sdf["real_date"] <= pd.to_datetime(end_date)]

        self.df: pd.DataFrame = sdf

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass


logger = logging.getLogger(__name__)


class LinePlot(Plotter):
    def __init__(
        self,
        df: pd.DataFrame,
        cids: List[str] = None,
        xcats: List[str] = None,
        start_date: str = None,
        metrics: str = ["value"],
        end_date: str = None,
    ):

        super().__init__(
            df=df,
            cids=cids,
            xcats=xcats,
            start_date=start_date,
            end_date=end_date,
            metrics=metrics,
        )

    def plot(
        self,
        plot_type: str = "line",
        cids: List[str] = None,
        xcats: List[str] = None,
        metric: str = "value",
        start_date: str = None,
        end_date: str = None,
        plot_by_cid: bool = None,
        plot_by_xcat: bool = None,
        xcat_labels: List[str] = None,
        cid_labels: List[str] = None,
        font_size: int = 12,
        x_axis_label: str = None,
        y_axis_label: str = None,
        add_axhline: bool = False,
        compare_series: Optional[pd.Series] = None,
        compare_series_label: str = "ReferenceSeries",
        figsize: tuple = (8, 12),
        height: int = 3,
        plot_style: str = "darkgrid",
        aspect: float = 1.5,
        fig_title: str = None,
        fig_title_adj: float = 1.05,
        legend: bool = True,
        legend_title: str = None,
        legend_loc: str = "best",
        legend_fontsize: int = 12,
        legend_ncol: int = 1,
        legend_bbox_to_anchor: tuple = (1, 1),
    ) -> matplotlib.figure.Figure:
        """
        Plot the data in the DataFrame.
        Plot all the lines on one plot.

        Parameters
        ----------
        Parameter for filtering the DataFrame:
        :param <List[str]> cids: A list of cids to select from the DataFrame
            (self.df). If None, all cids are selected.
        :param <List[str]> xcats: A list of xcats to select from the DataFrame
            (self.df). If None, all xcats are selected.
        :param <List[str]> metrics: A list of metrics to select from the DataFrame
            (self.df). If None, all metrics are selected.
        :param <str> start_date: The start date to select from the DataFrame in
            the format 'YYYY-MM-DD'. If None, all dates are selected.
        :param <str> end_date: The end date to select from the DataFrame in
            the format 'YYYY-MM-DD'. If None, all dates are selected.

        Parameters for plotting:
        :param <bool> plot_by_cid: If True, plot the lines for each cid on a
            separate plot. If False, plot all lines on one plot. If None, plot
            all lines on one plot.
        :param <bool> plot_by_xcat: If True, plot the lines for each xcat on a
            separate plot. If False, plot all lines on one plot. If None, plot
            all lines on one plot.

        Parameters for labelling the plot:
        :param <List[str]> xcat_labels: A list of labels for the xcats. If None,
            the xcat names are used.
        :param <List[str]> cid_labels: A list of labels for the cids. If None,
            the cid names are used.
        :param <int> font_size: The font size for the labels.

        Parameters for the figure:
        :param <int> ncols: The number of columns in the figure.
        :param <bool> same_x: If True, the x-axis limits are the same
            for all plots.
        :param <bool> same_y: If True, the y-axis limits are the same
            for all plots.
        :param <tuple> figsize: The size of the figure.
        :param <float> aspect: The aspect ratio of the figure.
        :param <str> fig_title: The title of the figure.
        :param <float> fig_title_adj: The adjustment of the figure title.
        :param <bool> legend: If True, show the legend.
        :param <str> legend_title: The title of the legend.
        :param <str> legend_loc: The location of the legend.
        :param <int> legend_fontsize: The font size of the legend.
        :param <int> legend_ncol: The number of columns in the legend.
        :param <tuple> legend_bbox_to_anchor: The bounding box
            of the legend.
        """
        df: pd.DataFrame = reduce_df(
            self.df,
            cids=cids,
            xcats=xcats,
            start=start_date,
            end=end_date,
        )

        if plot_by_cid is None:
            plot_by_cid = True

        if plot_by_xcat is True:
            plot_by_cid = False

        # validate args
        # assert metric in df.columns, f"Metric '{metric}' not found in DataFrame"
        if not metric in df.columns:
            raise ValueError(
                f"Metric '{metric}' not found in DataFrame"
                f" with columns: {df.columns}"
            )

        validListOfStr: Callable[[List[str]], bool] = (
            lambda x: (isinstance(x, list)) and (len(x) > 0) and (isinstance(x[0], str))
        )

        if cids is not None:
            if not validListOfStr(cids):
                raise ValueError("`cids` must be a list of strings")
        else:
            cids: List[str] = df["cid"].unique().tolist()

        if cid_labels is not None:
            if not validListOfStr(cid_labels) or (len(cid_labels) != len(cids)):
                raise ValueError(
                    "`cid_labels` must be a list of strings"
                    " with the same length as `cids`"
                )
        else:
            cid_labels: List[str] = cids

        if xcats is not None:
            if not validListOfStr(xcats):
                raise ValueError("`xcats` must be a list of strings")
        else:
            xcats: List[str] = df["xcat"].unique().tolist()

        if xcat_labels is not None:
            if not validListOfStr(xcat_labels) or (len(xcat_labels) != len(xcats)):
                raise ValueError(
                    "`xcat_labels` must be a list of strings"
                    " with the same length as `xcats`"
                )
        else:
            xcat_labels: List[str] = xcats

        if (set(cids) != set(cid_labels)) and (len(cids) == len(cid_labels)):
            replace_dict: Dict[str, str] = dict(zip(cids, cid_labels))
            df["cid"] = df["cid"].replace(replace_dict)

        if (set(xcats) != set(xcat_labels)) and (len(xcats) == len(xcat_labels)):
            replace_dict: Dict[str, str] = dict(zip(xcats, xcat_labels))
            df["xcat"] = df["xcat"].replace(replace_dict)

        # set up plot
        plot_by_col: str = "cid" if plot_by_cid else "xcat"
        hue_col: str = "cid" if plot_by_cid else "xcat"

        # choose plot function
        plot_func: Callable = {
            "line": sns.lineplot,
            "scatter": sns.scatterplot,
        }[plot_type]

        sns.set_style(plot_style)
        ax: plt.Axes = plot_func(
            data=df,
            x="real_date",
            y=metric,
            hue=hue_col,
            estimator=None,
        )

        if compare_series is not None:
            assert isinstance(
                compare_series, pd.Series
            ), "`compare_series` must be a pandas Series"
            assert isinstance(
                compare_series_label, str
            ), "`compare_series_label` must be a string"
            sns.lineplot(
                x=compare_series.index,
                y=compare_series.values,
                ax=ax,
                label=compare_series_label,
                # color red
                color="red",
                estimator=None,
            )

        sns.set(rc={"figure.figsize": figsize})
        ax.set_title(fig_title, fontsize=font_size, pad=fig_title_adj)
        ax.set_xlabel(x_axis_label or "Date", fontsize=font_size)
        ax.set_ylabel(y_axis_label or metric, fontsize=font_size)
        ax.tick_params(axis="both", labelsize=font_size)

        if legend:
            if legend_loc == "best":
                legend_loc: str = "center left"
            ax.legend(
                title=legend_title,
                loc=legend_loc,
                fontsize=legend_fontsize,
                # ncol=legend_ncol,
                bbox_to_anchor=legend_bbox_to_anchor,
            )

        if add_axhline:
            ax.axhline(y=0, c=".5")

        # return the figure
        return ax.get_figure()


class FacetPlot(Plotter):
    def __init__(
        self,
        df: pd.DataFrame,
        cids: List[str] = None,
        xcats: List[str] = None,
        metrics: List[str] = None,
        start_date: str = None,
        end_date: str = None,
    ):

        super().__init__(
            df=df,
            cids=cids,
            xcats=xcats,
            metrics=metrics,
            start_date=start_date,
            end_date=end_date,
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def plot(
        self,
        plot_type: str = "line",
        cids: List[str] = None,
        xcats: List[str] = None,
        metric: str = "value",
        start_date: str = None,
        end_date: str = None,
        plot_by_cid: bool = None,
        plot_by_xcat: bool = None,
        xcat_labels: List[str] = None,
        cid_labels: List[str] = None,
        add_axhline: bool = True,
        compare_series: Optional[pd.Series] = None,
        compare_series_label: str = "ReferenceSeries",
        x_axis_label: str = None,
        y_axis_label: str = None,
        label_adj : float = 1.05,
        all_xticks: bool = True,
        font_size: int = 12,
        ncols: int = 4,
        same_x: bool = True,
        same_y: bool = True,
        figsize: tuple = (12, 8),
        aspect: float = 1.5,
        height: float = 3,
        fig_title: str = None,
        fig_title_adj: float = 1.05,
        fig_title_fontsize: int = 18,
        plot_style: str = "darkgrid",
        legend: bool = True,
        legend_title: str = None,
        legend_loc: str = "best",
        legend_adj: float = 1.05,
        legend_fontsize: int = 5,
        legend_ncol: int = 1,
        legend_bbox_to_anchor: tuple = (1, 1),
    ) -> matplotlib.figure.Figure:
        """
        Plot the data in the DataFrame.

        Parameters
        ----------
        Parameter for filtering the DataFrame:
        :param <List[str]> cids: A list of cids to select from the DataFrame
            (self.df). If None, all cids are selected.
        :param <List[str]> xcats: A list of xcats to select from the DataFrame
            (self.df). If None, all xcats are selected.
        :param <List[str]> metric : A metric to select from the DataFrame. Defaults to
            'value'.
        :param <str> start_date: The start date to select from the DataFrame in
            the format 'YYYY-MM-DD'. If None, all dates are selected.
        :param <str> end_date: The end date to select from the DataFrame in the
            format 'YYYY-MM-DD'. If None, all dates are selected.
        :param <bool> plot_by_cid: If True (default), each cid is plotted in a separate
            facet. If False (or None), each xcat is plotted in a separate facet.
            Must be of the same length as `xcats` or the number of unique cids in the
            DataFrame.
        :param <bool> plot_by_xcat: If True, each xcat is plotted in a separate
            facet. If False (default) (or None), each cid is plotted in a separate facet.
            Must be of the same length as `cids` or the number of unique xcats in the
            DataFrame.

        Parameters for plotting:
        :param <List[str]> xcat_labels: A list of labels to use with categories (xcat),
            in the same order as the categories. If None (default), the original
            xcat names are used.
        :param <List[str]> cid_labels: A list of labels to use with categories (cid),
            in the same order as the categories. If None (default), the original
            cid names are used.
        :param <str> x_axis_label: The label to use for the x-axis. If None (default),
            the index name (usually 'real_date') is used.
        :param <str> y_axis_label: The label to use for the y-axis. If None (default),
            the metric name is used (usually 'value').
        :param <bool> add_axhline: If True (default), an horizontal line is added at
            y=0.
        :param <int> font_size: The font size to use for the subplot titles and labels.
            Default is 12.
        :param <int> ncols: The number of columns to use in the FacetGrid. Default is 4.
        :param <bool> same_x: If True (default), the x-axis is shared across all subplots.
        :param <bool> same_y: If True (default), the y-axis is shared across all subplots.
        :param <tuple> figsize: The size of the figure. Default is (8, 12).
        :param <float> aspect: The aspect ratio to use for the subplots. Default is 1.5.
        :param <str> fig_title: The title to use for the figure. Default is None.
        :param <float> fig_title_adj: The vertical adjustment of the figure title.
            Default is 1.05.
        :param <bool> plot_style: The style to use for the plot. Default is 'darkgrid'.
        :param <bool> legend: If True (default), a legend is added to the plot.
        :param <str> legend_title: The title to use for the legend. Default is None.
        :param <str> legend_loc: The location to use for the legend. Default is 'best'.
        :param <int> legend_fontsize: The font size to use for the legend. Default is 12.
        :param <int> legend_ncol: The number of columns to use for the legend. Default is 1.
        :param <tuple> legend_bbox_to_anchor: The bounding box to use for the legend.
            Default is (1, 1).

        Returns
        -------
        :return <matplotlib.figure.Figure>: The figure object. However, to plot, one must
            call `plt.show()` or `plt.savefig()` after calling this method.

        """

        if not isinstance(plot_type, str):
            raise TypeError(f"plot_type must be a string, not {type(plot_type)}")
        else:
            plot_type: str = plot_type.lower()
            if not plot_type in ["line", "scatter"]:
                raise ValueError(f"plot_type must be 'line' or 'bar', not {plot_type}")

        if plot_by_cid is None:
            plot_by_cid = True

        if plot_by_xcat is True:
            plot_by_cid = False

        df: pd.DataFrame = reduce_df(
            self.df,
            cids=cids,
            xcats=xcats,
            start=start_date,
            end=end_date,
        )

        # validate args
        # assert metric in df.columns, f"Metric '{metric}' not found in DataFrame"
        if not metric in df.columns:
            raise ValueError(
                f"Metric '{metric}' not found in DataFrame"
                f" with columns: {df.columns}"
            )

        validListOfStr: Callable[[List[str]], bool] = (
            lambda x: (isinstance(x, list)) and (len(x) > 0) and (isinstance(x[0], str))
        )

        if cids is not None:
            if not validListOfStr(cids):
                raise ValueError("`cids` must be a list of strings")
        else:
            cids: List[str] = df["cid"].unique().tolist()

        if cid_labels is not None:
            if not validListOfStr(cid_labels) or (len(cid_labels) != len(cids)):
                raise ValueError(
                    "`cid_labels` must be a list of strings"
                    " with the same length as `cids`"
                )
        else:
            cid_labels: List[str] = cids

        if xcats is not None:
            if not validListOfStr(xcats):
                raise ValueError("`xcats` must be a list of strings")
        else:
            xcats: List[str] = df["xcat"].unique().tolist()

        if xcat_labels is not None:
            if not validListOfStr(xcat_labels) or (len(xcat_labels) != len(xcats)):
                raise ValueError(
                    "`xcat_labels` must be a list of strings"
                    " with the same length as `xcats`"
                )
        else:
            xcat_labels: List[str] = xcats

        # rename cids, xcats in df to match cid_labels, xcat_labels

        if (set(cids) != set(cid_labels)) and (len(cids) == len(cid_labels)):
            replace_dict: Dict[str, str] = dict(zip(cids, cid_labels))
            df["cid"] = df["cid"].replace(replace_dict)

        if (set(xcats) != set(xcat_labels)) and (len(xcats) == len(xcat_labels)):
            replace_dict: Dict[str, str] = dict(zip(xcats, xcat_labels))
            df["xcat"] = df["xcat"].replace(replace_dict)

        # set up plot
        plot_by_col: str = "cid" if plot_by_cid else "xcat"
        hue_col: str = "xcat" if plot_by_cid else "cid"

        if ncols > len(df[plot_by_col].unique()):
            ncols: int = len(df[plot_by_col].unique())

        # choose plot function
        plot_func: Callable = {
            "line": sns.lineplot,
            "scatter": sns.scatterplot,
        }[plot_type]

        g: sns.FacetGrid = sns.FacetGrid(
            df,
            col=plot_by_col,
            hue=hue_col,
            col_wrap=ncols,
            sharex=same_x,
            sharey=same_y,
            aspect=aspect,
            height=height,
            legend_out=True,
        )

        g.map_dataframe(
            plot_func,
            x="real_date",
            y=metric,
            hue=hue_col,
            style=hue_col,
        )

        if compare_series is not None:
            assert isinstance(
                compare_series, pd.Series
            ), f"`compare_series` must be a pandas Series, not {type(compare_series)}"
            assert isinstance(
                compare_series_label, str
            ), f"`compare_series_label` must be a string, not {type(compare_series_label)}"

            # add this series to every plot. use plot_func to determine the plot type
            ax: plt.Axes
            for ax in g.axes.flatten():
                plot_func(
                    x=compare_series.index,
                    y=compare_series.values,
                    ax=ax,
                    color="red",
                    label=compare_series_label,
                )

        # set plot titles
        g.set_titles(col_template="{col_name}", size=font_size)

        # set the plot style
        sns.set_style(plot_style)
        sns.set(rc={"figure.figsize": figsize})        
        
        g.set_axis_labels(x_axis_label or "real_date", y_axis_label or metric)

        g.figure.subplots_adjust(top=fig_title_adj, bottom=label_adj, left=label_adj, right=1 - label_adj)

        # set the title
        if fig_title is not None:
            g.fig.suptitle(fig_title, fontsize=fig_title_fontsize, y=fig_title_adj)

        # set the legend
        if legend or compare_series is not None:
            if not legend:
                legend_labels: List[str] = [compare_series_label]
            else:
                legend_labels: List[str] = xcat_labels if plot_by_cid else cid_labels
                if compare_series is not None:
                    legend_labels.append(compare_series_label)

            if legend_loc == "best":
                legend_loc: str = "center left"


            g.figure.legend(
                title=legend_title,
                loc=legend_loc,
                fontsize=legend_fontsize,
                # ncol=legend_ncol,
                bbox_to_anchor=legend_bbox_to_anchor,
            )
            # adjust the figure size to fit the legend
            g.fig.subplots_adjust(right=legend_adj)
            # move the legend to the top right of the plot

        if add_axhline:
            g.map(plt.axhline, y=0, c=".5")

        if all_xticks:
            g.tick_params(labelbottom=True, pad=0)

        return g.figure
