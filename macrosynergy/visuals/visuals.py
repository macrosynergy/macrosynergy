"""
Providing a high level interface to simplify visual tasks involving matplotlib and seaborn.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas  as pd
import numpy   as np
from typing import Optional, List, Tuple, Dict, Union
import logging

logger = logging.getLogger(__name__)

def facet_grid(
        df : pd.DataFrame,
        plot_by_tickers : Optional[bool] = False,
        plot_by_cid : Optional[str] = None,
        plot_xcats : Optional[List[str]] = None,
        xcat_labels : Optional[List[str]] = None,
        plot_by_xcat : Optional[str] = None,
        cid_labels : Optional[List[str]] = None,
        plot_cids : Optional[List[str]] = None,
        ncols : Optional[int] = 3,
        nrows : Optional[int] = 3,
        sharey : Optional[bool] = True,
        sharex : Optional[bool] = True,
        share_x_labels : Optional[bool] = True,
        share_y_labels : Optional[bool] = True,
        xlabel : Optional[str] = None,
        ylabel : Optional[str] = None,
        title : Optional[str] = None,
        title_adj : Optional[float] = 0.95,
        figsize : Optional[Tuple[int, int]] = None,
        aspect : Optional[float] = 1.618,
        height : Optional[float] = None,
        width : Optional[float] = None,
        facetsize : Optional[Tuple[int, int]] = None,
        show : Optional[bool] = True,
):
    """
    :param <pd.Dataframe> df: standardized DataFrame with the necessary columns:
        'cid', 'xcats', 'real_date' and at least one column with values of interest.
    :param <str> plot_by_tickers: plot all cid_xcat tickers from the DataFrame,
        with one ticker per facet.
    :param <str> plot_by_cid: plot all xcats for the specified cid from the DataFrame,
        with one xcat per facet. Paired args : `xcat_labels`, `plot_xcats`.
    :param <list> plot_xcats: plot only the specified xcats for the specified cid. 
        To be used with arg: `plot_by_cid`.
    :param <list> xcat_labels: labels for the xcats. Must be of same length as
        the number of unique xcats in the DataFrame, or the number of xcats specified
        `plot_xcats`. To be used with arg: `plot_by_cid`.
    :param <str> plot_by_xcat: plot all cids for the specified xcat from the DataFrame,
        with one cid per facet. Paired args : `cid_labels`, `plot_cids`.
    :param <list> plot_cids: plot only the specified cids for the specified xcat.
        To be used with arg: `plot_by_xcat`.
    :param <list> cid_labels: labels for the cids. Must be of same length as
        the number of unique cids in the DataFrame, or the number of cids specified
        `plot_cids`. To be used with arg: `plot_by_xcat`.
    :param <int> ncols: number of columns in the facet grid.
    :param <int> nrows: number of rows in the facet grid.
    :param <bool> sharey: share the y-axis across all facets.
    :param <bool> sharex: share the x-axis across all facets.
    :param <bool> share_x_labels: share the x-axis labels across all facets.
    :param <bool> share_y_labels: share the y-axis labels across all facets.
    :param <str> xlabel: label for the x-axis.
    :param <str> ylabel: label for the y-axis.
    :param <str> title: title for the facet grid.
    :param <float> title_adj: adjust the title position.
    :param <tuple> figsize: size of the figure.
    :param <float> aspect: aspect ratio of the figure.
    :param <float> height: height of the figure.
    :param <float> width: width of the figure.
    :param <tuple> facetsize: size of each facet.
    :param <bool> show: show the plot.

    return <matplotlib.figure.Figure>
    """

    # input validation
    assert isinstance(df, pd.DataFrame), 'The input df must be a pandas DataFrame.'
    dfcols = set(df.columns)
    mandatory_cols = set(['cid', 'xcats', 'real_date'])
    optional_cols = set(['mop_lag', 'eop_lag', 'value', 'grading'])

    # all mandatory columns must be present, atleast one optional column must be present
    if not mandatory_cols.issubset(dfcols) or not optional_cols.intersection(dfcols):
        # try some pandas magic
        dfa = df.copy().reset_index()
        dfa.columns = [c.lower() for c in dfa.columns]

        logger.warning('The DataFrame columns are not standardized. Trying to standardize them.')
        assert mandatory_cols.issubset(dfa.columns), 'The mandatory columns are not present in the DataFrame.'
        assert optional_cols.intersection(dfa.columns), 'None of the optional columns are present in the DataFrame.'

    assert (plot_by_cid is None) ^ (plot_by_xcat is None), 'Please specify either plot_by_cid or plot_by_xcat.'
    if plot_by_cid is not None:
        assert plot_by_cid in df['cid'].unique(), 'The specified cid is not present in the DataFrame.'
    if plot_by_xcat is not None:
        assert plot_by_xcat in df['xcats'].unique(), 'The specified xcat is not present in the DataFrame.'

    if figsize is None:
        assert isinstance(aspect, (int, float)), 'The aspect ratio must be a float or integer.'
        # one of height or width must be specified
        assert (height is not None) ^ (width is not None), 'Please specify either height or width.'
        if height is not None:
            assert isinstance(height, (int, float)), 'The height must be a float or integer.'
            figsize = (aspect * height, height)
        else:
            assert isinstance(width, (int, float)), 'The width must be a float or integer.'
            figsize = (width, width / aspect)
    else:
        assert isinstance(figsize, tuple), 'The figsize must be a tuple of integers.'
        assert len(figsize) == 2, 'The figsize must be a tuple of length 2.'
        assert all([isinstance(i, int) for i in figsize]), 'The figsize must be a tuple of integers.'

    
    if plot_by_tickers:
        raise NotImplementedError('plot_by_tickers is not implemented yet.')
        # form a helper column called 'tickers'
        # df['tickers'] = df['cid'] + '_' + df['xcats']
        # simply plot the tickers to a facet grid
        # g = sns.FacetGrid(df, col='tickers', col_wrap=ncols,
        #                   sharey=sharey, sharex=sharex,
        #                   height=height, aspect=aspect)
        # g.map(plt.plot, 'real_date', 'value')

    if plot_by_cid is not None:
        # filter the DataFrame
        df = df[df['cid'] == plot_by_cid]
        # plot the xcats to a facet grid
        if xcat_labels is not None:
            assert len(xcat_labels) == len(df['xcats'].unique()), 'The number of xcat labels must be equal to the number of xcats.'
            df['xcats'] = df['xcats'].replace(dict(zip(df['xcats'].unique(), xcat_labels)))

        fg = sns.FacetGrid(df, col='xcats', col_wrap=ncols,
                            sharey=sharey, sharex=sharex,
                            height=height, aspect=aspect)

        # title for each individual facet should be xcat. plot real_date on x-axis, value on y-axis
        fg.map(plt.plot, 'real_date', 'value')
        fg.set_titles('{col_name}')
        
        # set the x-axis labels
        if share_x_labels:
            fg.set_xlabels(xlabel)
        else:
            for ax in fg.axes.flat:
                ax.set_xlabel(xlabel)

        # set the y-axis labels
        if share_y_labels:
            fg.set_ylabels(ylabel)
        else:
            for ax in fg.axes.flat:
                ax.set_ylabel(ylabel)

        # set the title
        if title is not None:
            fg.fig.suptitle(title, y=title_adj)

        # set the figure size using rcParams
        plt.rcParams['figure.figsize'] = figsize

        # set the facet size using rcParams



    






    
    
