import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Union, Tuple

from macrosynergy.management.simulate_quantamental_data import make_qdf
from macrosynergy.management.check_availability import reduce_df


def view_ranges(df: pd.DataFrame, xcats: List[str] = None,  cids: List[str] = None,
                start: str = '2000-01-01', end: str = None, val: str = 'value',
                kind: str = 'bar', sort_cids_by: str = None,
                title: str = None, ylab: str = None, size: Tuple[float] = (16, 8)):

    """Plot level bars and SD ranges across extended categories and cross sections

    :param <pd.Dataframe> df: standardized dataframe with the following necessary columns:
        'cid', 'xcats', 'real_date' and at least one column with values of interest.
    :param <List[str]> xcats: extended categories to be checked on. Default is all in the dataframe.
    :param <List[str]> cids: cross sections to be checked on. Default is all in the dataframe.
    :param <str> start: earliest date in ISO format. Default is None and earliest date in df is used.
    :param <str> end: latest date in ISO format. Default is None and latest date in df is used.
    :param <str> val: name of column that contains the values of interest. Default is 'value'.
    :param <str> kind: type of range plot to be used. Default is 'bar'; other option is 'box'.
    :param <str> sort_cids_by: criterion for sorting cids on x axis; can be 'mean' and 'std'. Default is original order.
    :param <str> title: string of chart title; defaults depend on type of range plot.
    :param <str> ylab: y label. Default is no label.
    :param <Tuple[float]> size: Tuple of width and height of graph. Default is (16, 8). 

    """

    df, xcats, cids = reduce_df(df, xcats, cids, start, end, out_all=True)

    s_date = df['real_date'].min().strftime('%Y-%m-%d')
    e_date = df['real_date'].max().strftime('%Y-%m-%d')

    sns.set(style="darkgrid")

    if title is None:
        if kind == 'bar':
            title = f'Means and standard deviations from {s_date} to {e_date}'
        elif kind == 'box':
            title = f'Interquartile ranges, extended ranges and outliers from {s_date} to {e_date}'
    if ylab is None:
        ylab = ""

    if sort_cids_by == 'mean':
        dfx = df[df['xcat']==xcats[0]].groupby(['cid'])[val].mean()
        order = dfx.sort_values(ascending=False).index
    elif sort_cids_by == 'std':
        dfx = df[df['xcat']==xcats[0]].groupby(['cid'])[val].std()
        order = dfx.sort_values(ascending=False).index
    else:
        order = None

    sns.set_theme(style="darkgrid")
    fig, ax = plt.subplots(figsize=size)

    if kind == 'bar':
        ax = sns.barplot(x='cid', y=val, hue='xcat', hue_order=xcats, palette='Paired', data=df, ci='sd', order=order)
    elif kind == 'box':
        ax = sns.boxplot(x='cid', y=val, hue='xcat', hue_order=xcats, palette='Paired', data=df,  order=order)
        ax.xaxis.grid(True)

    ax.set_title(title,  fontdict={'fontsize': 16})
    ax.set_xlabel("")
    ax.set_ylabel(ylab)
    ax.xaxis.grid(True)
    ax.axhline(0, ls='--', linewidth=1, color='black')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[0:], labels=labels[0:])
    plt.show()


if __name__ == "__main__":

    cids = ['AUD', 'CAD', 'GBP']
    xcats = ['XR', 'CRY']
    df_cids = pd.DataFrame(index=cids, columns=['earliest', 'latest', 'mean_add', 'sd_mult'])
    df_cids.loc['AUD', ] = ['2010-01-01', '2020-12-31', 0.5, 0.2]
    df_cids.loc['CAD', ] = ['2011-01-01', '2020-11-30', 0, 1]
    df_cids.loc['GBP', ] = ['2012-01-01', '2020-11-30', 0, 2]

    df_xcats = pd.DataFrame(index=xcats, columns=['earliest', 'latest', 'mean_add', 'sd_mult', 'ar_coef', 'back_coef'])
    df_xcats.loc['XR', ] = ['2010-01-01', '2020-12-31', 0, 1, 0, 0.3]
    df_xcats.loc['CRY', ] = ['2011-01-01', '2020-10-30', 1, 2, 0.9, 0.5]

    dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)

    view_ranges(dfd, xcats=['XR'], cids=cids, kind='box', sort_cids_by='std')
    view_ranges(dfd, xcats=['XR'], cids=cids, kind='box', start='2012-01-01', end='2018-01-01', sort_cids_by='std')

    view_ranges(dfd, xcats=['CRY', 'XR'], kind='bar', sort_cids_by='mean')
