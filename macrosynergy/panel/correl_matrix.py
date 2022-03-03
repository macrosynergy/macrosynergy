
import pandas as pd
import numpy as np
import seaborn as sns
import scipy.cluster.hierarchy as sch
from matplotlib import pyplot as plt
from scipy.spatial.distance import squareform
from typing import List, Union, Tuple
from macrosynergy.management.check_availability import reduce_df


def correl_matrix_cluster(df: pd.DataFrame, xcats: List[str] = None,
                          cids: List[str] = None, start: str = '2000-01-01',
                          end: str = None, val: str = 'value',
                          title: str = None, size: Tuple[float] = (14, 8),
                          max_color: float = None):
    """
    Display correlation matrix either across xcats (if more than one xcat) or cids
    post hierarchical cluster reordering.

    :param <pd.Dataframe> df: standardized dataframe with the following necessary columns:
        'cid', 'xcats', 'real_date' and at least one column with values of interest.
    :param <List[str]> xcats: extended categories to be correlated. Default is all in the
        dataframe. If xcats contains only one category the correlation coefficients
        across cross sections are displayed. If xcats contains more than one category the
        correlation coefficients across categories are displayed.
    :param <List[str]> cids: cross sections to be correlated. Default is all in the
        dataframe.
    :param <str> start: earliest date in ISO format. Default is None and earliest date
        in df is used.
    :param <str> end: latest date in ISO format. Default is None and latest date in df
        is used.
    :param <str> val: name of column that contains the values of interest. Default is
        'value'.
    :param <str> title: chart heading. If none is given, a default title is used.
    :param <Tuple[float]> size: two-element tuple setting width/height of figure. Default
        is (14, 8).
    :param <float> max_color: maximum values of positive/negative correlation
        coefficients for color scale. Default is none. If a value is given it applies
        symmetrically to positive and negative values.

    """

    xcats = xcats if isinstance(xcats, list) else [xcats]

    min_color = None if max_color is None else -max_color  # define minimum of color scale

    df, xcats, cids = reduce_df(df, xcats, cids, start, end, out_all=True)

    s_date = df['real_date'].min().strftime('%Y-%m-%d')

    e_date = df['real_date'].max().strftime('%Y-%m-%d')

    if len(xcats) == 1:

        df_w = df.pivot(index='real_date', columns='cid', values=val)

        if title is None:
            title = f'Cross-sectional correlation of {xcats[0]} from {s_date} to ' \
                    f'{e_date}'

    else:

        df_w = df.pivot(index=('cid', 'real_date'), columns='xcat', values=val)

        if title is None:
            title = f'Cross-category correlation from {s_date} to {e_date}'

    sns.set(style="ticks")

    corr = df_w.corr()
    # cluster reorganisation

    d = sch.distance.pdist(corr)

    L = sch.linkage(d, method='complete')

    ind = sch.fcluster(L, 0.5 * d.max(), 'distance')

    columns = [corr.columns.tolist()[i] for i in list((np.argsort(ind)))]

    corr = corr.loc[columns, columns]

    mask = np.triu(np.ones_like(corr, dtype=bool))  # generate mask for upper triangle

    fig, ax = plt.subplots(figsize=size)
    sns.heatmap(corr, mask=mask, cmap='vlag_r', center=0, vmin=min_color, vmax=max_color,
                square=False, linewidths=.5, cbar_kws={"shrink": .5})
    ax.set(xlabel='', ylabel='')
    ax.set_title(title, fontsize=14)

    plt.show()