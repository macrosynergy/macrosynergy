"""
Providing a high level interface to simplify visual tasks involving matplotlib and seaborn.
"""

import matplotlib
import matplotlib.figure
import matplotlib.pyplot as plt
import seaborn as sns
import pandas  as pd
import numpy   as np
from typing import Optional, List, Tuple, Dict, Union, Callable
import logging
from macrosynergy.management.utils import standardise_dataframe
from macrosynergy.management import reduce_df



logger = logging.getLogger(__name__)

class PlotLines(object):
    """
    inialise with a DF.
    have optional cids, xcats, start_date, end_date, metric,
    option to filter by cid, xcat, start_date, end_date, metric
    """
    def __init__(self,
                 df : pd.DataFrame,
                 cids : List[str] = None,
                xcats : List[str] = None,
                start_date : str = None,
                end_date : str = None,
                ):
        
        self.df :pd.DataFrame = reduce_df(
            standardise_dataframe(df),
            cids=cids,
            xcats=xcats,
            start=start_date,
            end=end_date,
        )
        


    def plot(self,
                cids : List[str] = None,
                xcats : List[str] = None,
                metrics : List[str] = None,
                start_date : str = None,
                end_date : str = None,
                plot_by_cid : bool = None,
                plot_by_xcat : bool = None,
                xcat_labels : List[str] = None,
                cid_labels : List[str] = None,
                font_size : int = 12,
                metric_labels : List[str] = None,
                ncols : int = 4,
                same_x : bool = True,
                same_y : bool = True,
                figsize : tuple = (8, 12),
                aspect : float = 1.5,
                fig_title : str = None,
                fig_title_adj : float = 1.05,
                legend : bool = True,
                legend_title : str = None,
                legend_loc : str = 'best',
                legend_fontsize : int = 12,
                legend_ncol : int = 1,
                legend_bbox_to_anchor : tuple = (1, 1),
    ):
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
        :param <List[str]> metric_labels: A list of labels for the metrics. If
            None, the metric names are used.
        
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
        pass


    

class FacetPlot(object):
    """
    inialise with a DF.
    have optional cids, xcats, start_date, end_date, metric,
    option to filter by cid, xcat, start_date, end_date, metric
    """
    def __init__(self,
                 df : pd.DataFrame,
                 cids : List[str] = None,
                xcats : List[str] = None,
                metrics : List[str] = None,
                start_date : str = None,
                end_date : str = None,
                ):
        
        # sdf : pd.DataFrame = standardise_dataframe(df)
        sdf = sdf[["real_date", "cid", "xcat",] + metrics]
        if cids:
            sdf = sdf[sdf["cid"].isin(cids)]
        if xcats:
            sdf = sdf[sdf["xcat"].isin(xcats)]
        if start_date:
            sdf = sdf[sdf["real_date"] >= pd.to_datetime(start_date)]
        if end_date:
            sdf = sdf[sdf["real_date"] <= pd.to_datetime(end_date)]
        
        self.df : pd.DataFrame = sdf



    def plot(self,
                plot_type : str = 'line',
                cids : List[str] = None,
                xcats : List[str] = None,
                metric : str = "value",
                start_date : str = None,
                end_date : str = None,
                plot_by_cid : bool = None,
                plot_by_xcat : bool = None,
                xcat_labels : List[str] = None,
                cid_labels : List[str] = None,
                x_axis_label : str = None,
                y_axis_label : str = None,
                font_size : int = 12,
                ncols : int = 4,
                same_x : bool = True,
                same_y : bool = True,
                figsize : tuple = (8, 12),
                aspect : float = 1.5,
                fig_title : str = None,
                fig_title_adj : float = 1.05,
                plot_style : str = 'darkgrid',
                legend : bool = True,
                legend_title : str = None,
                legend_loc : str = 'outside',
                legend_fontsize : int = 12,
                legend_ncol : int = 1,
                legend_bbox_to_anchor : tuple = (1, 1),
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

        """

        if not isinstance(plot_type, str):
            raise TypeError(f"plot_type must be a string, not {type(plot_type)}")
        else:
            plot_type : str = plot_type.lower()
            if not plot_type in ['line', 'scatter']:
                raise ValueError(f"plot_type must be 'line' or 'bar', not {plot_type}")

        if plot_by_cid is None:
            plot_by_cid = True
        
        if plot_by_xcat is True:
            plot_by_cid = False

        df : pd.DataFrame = reduce_df(
            self.df,
            cids=cids,
            xcats=xcats,
            start=start_date,
            end=end_date,)
        
        # validate args
        # assert metric in df.columns, f"Metric '{metric}' not found in DataFrame"
        if not metric in df.columns:
            raise ValueError(f"Metric '{metric}' not found in DataFrame"
                       f" with columns: {df.columns}")
            
        validListOfStr : Callable[[List[str]], bool] = lambda x: \
            (isinstance(x, list)) and (len(x) > 0) and (isinstance(x[0], str))
        
        if plot_by_cid:
            if cids is not None:
                if not validListOfStr(cids):
                    raise ValueError("`cids` must be a list of strings")
            else:
                cids : List[str] = df['cid'].unique().tolist()

            if cid_labels is not None:
                if not validListOfStr(cid_labels) or \
                    (len(cid_labels) != len(cids)):
                    raise ValueError("`cid_labels` must be a list of strings"
                                     " with the same length as `cids`")
            else:
                cid_labels : List[str] = cids
            
        else:
            if xcats is not None:
                if not validListOfStr(xcats):
                    raise ValueError("`xcats` must be a list of strings")
            else:
                xcats : List[str] = df['xcat'].unique().tolist()

            if xcat_labels is not None:
                if not validListOfStr(xcat_labels) or \
                    (len(xcat_labels) != len(xcats)):
                    raise ValueError("`xcat_labels` must be a list of strings"
                                     " with the same length as `xcats`")
            else:
                xcat_labels : List[str] = xcats


        # rename cids, xcats in df to match cid_labels, xcat_labels

        if set(cids) != set(cid_labels):
            replace_dict : Dict[str, str] = dict(zip(cids, cid_labels))
            df['cid'] = df['cid'].replace(replace_dict)

        if set(xcats) != set(xcat_labels):
            replace_dict : Dict[str, str] = dict(zip(xcats, xcat_labels))
            df['xcat'] = df['xcat'].replace(replace_dict)

        # set up plot
        plot_by_col : str = 'cid' if plot_by_cid else 'xcat'
        hue_col : str = 'xcat' if plot_by_cid else 'cid'

        # choose plot function
        plot_func : Callable = \
        {
            'line': sns.lineplot,
            'scatter': sns.scatterplot,
        }[plot_type]

        g : sns.FacetGrid = sns.FacetGrid(
            df,
            col=plot_by_col,
            col_wrap=ncols,
            sharex=same_x,
            sharey=same_y,
            height=figsize[1],
            aspect=aspect,
            despine=True,)
        
        # plot
        g.map_dataframe(
            plot_func,
            x='real_date',
            y=metric,
            hue=hue_col,
            style=hue_col,
            markers=True,
            dashes=False,
            palette='tab10',)
            
        # set labels
        g.set_xlabels(x_axis_label or 'real_date')
        g.set_ylabels(y_axis_label or metric)
        g.set_titles(row_template='{row_name}', col_template='{col_name}')

        # add legend
        if legend:
            g.add_legend(
                title=legend_title,
                loc=legend_loc,
                fontsize=legend_fontsize,
                ncol=legend_ncol,
                bbox_to_anchor=legend_bbox_to_anchor,)
        
        # set figure title
        if fig_title is not None:
            g.figure.suptitle(fig_title, y=fig_title_adj)
        
        g.figure.set_size_inches(*figsize)
        sns.set_style(plot_style)

        return g.figure
            
