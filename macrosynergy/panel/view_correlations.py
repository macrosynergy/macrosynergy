import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Union, Tuple

from macrosynergy.management.simulate_quantamental_data import make_qdf
from macrosynergy.management.check_availability import reduce_df


def correl_matrix(df: pd.DataFrame, xcats: List[str] = None,  cids: List[str] = None,
                   start: str = '2000-01-01', end: str = None, val: str = 'value',
                   title: str = None, size: Tuple[float] = (14, 8), max_color: float=None):

    """Display correlation matrix either across xcats (if more than one xcat_) or cids

    :param <pd.Dataframe> df: standardized dataframe with the following necessary columns:
        'cid', 'xcats', 'real_date' and at least one column with values of interest.
    :param <List[str]> xcats: extended categories to be correlated. Default is all in the dataframe.
        If xcats contains only one category the correlation coefficients across sections are displayed.
        If xcats contains more than one category the correlation coefficients across categories are displayed.
    :param <List[str]> cids: cross sections to be correlated. Default is all in the dataframe.
    :param <str> start: earliest date in ISO format. Default is None and earliest date in df is used.
    :param <str> end: latest date in ISO format. Default is None and latest date in df is used.
    :param <str> val: name of column that contains the values of interest. Default is 'value'.
    :param <str> title: chart heading. If none is given, a default title is used.
    :param <Tuple[float]> size: two-element tuple setting width/height of figure. Default is (14, 8).
    :param <float> max_color: maximum values of positive/negative correlation coefficients for color scale
        Default is none. If a value is given it applies symmetrically to positive and negative values.

    """

    xcats = xcats if isinstance(xcats, list) else [xcats]
    min_color = None if max_color is None else -max_color  # define minimum of color scale

    df, xcats, cids = reduce_df(df, xcats, cids, start, end, out_all=True)

    s_date = df['real_date'].min().strftime('%Y-%m-%d')
    e_date = df['real_date'].max().strftime('%Y-%m-%d')

    if len(xcats) == 1:

        df_w = df.pivot(index='real_date', columns='cid', values=val)
        if title is None:
            title = f'Cross-sectional correlation of {xcats[0]} from {s_date} to {e_date}'

    else:

        df_w = df.pivot(index=('cid', 'real_date'), columns='xcat', values=val)
        if title is None:
            title = f'Cross-category correlation from {s_date} to {e_date}'

    sns.set(style="ticks")
    corr = df_w.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))  # generate mask for upper triangle
    fig, ax = plt.subplots(figsize=size)  # set up figure
    sns.heatmap(corr, mask=mask, cmap='vlag_r', center=0, vmin=min_color, vmax=max_color,
                square=False, linewidths=.5, cbar_kws={"shrink": .5})
    ax.set(xlabel='', ylabel='')
    ax.set_title(title, fontsize=14)

    plt.show()


if __name__ == "__main__":

    cids = ['AUD', 'CAD', 'GBP', 'NZD']
    xcats = ['XR', 'CRY', 'GROWTH', 'INFL']
    df_cids = pd.DataFrame(index=cids, columns=['earliest', 'latest', 'mean_add', 'sd_mult'])
    df_cids.loc['AUD', ] = ['2010-01-01', '2020-12-31', 0.1, 1]
    df_cids.loc['CAD', ] = ['2011-01-01', '2020-11-30', 0, 1]
    df_cids.loc['GBP', ] = ['2012-01-01', '2020-11-30', 0, 2]
    df_cids.loc['NZD', ] = ['2012-01-01', '2020-09-30', -0.1, 2]

    df_xcats = pd.DataFrame(index=xcats, columns=['earliest', 'latest', 'mean_add', 'sd_mult', 'ar_coef', 'back_coef'])
    df_xcats.loc['XR', ] = ['2010-01-01', '2020-12-31', 0.1, 1, 0, 0.3]
    df_xcats.loc['CRY', ] = ['2011-01-01', '2020-10-30', 1, 2, 0.95, 0.5]
    df_xcats.loc['GROWTH', ] = ['2011-01-01', '2020-10-30', 1, 2, 0.9, 0.5]
    df_xcats.loc['INFL', ] = ['2011-01-01', '2020-10-30', 1, 2, 0.8, 0.5]

    dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)

    correl_matrix(dfd, xcats=xcats, cids=cids, max_color=0.2)
    correl_matrix(dfd, xcats=xcats[0], cids=cids, title='Correlation')

    correl_matrix(dfd, xcats=xcats, cids=cids)

