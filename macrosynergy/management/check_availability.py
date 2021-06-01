import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Union, Tuple

from macrosynergy.management.simulate_quantamental_data import make_qdf


def missing_in_df(df: pd.DataFrame, xcats: List[str] = None,  cids: List[str] = None):
    """Print cross sections and extended categories that are missing or redundant in the dataframe

    :param <pd.Dataframe> df: standardized dataframe with the following necessary columns:
        'cid', 'xcats', 'real_date'.
    :param <List[str]> xcats: extended categories to be checked on. Default is all in the dataframe.
    :param <List[str]> cids: cross sections to be checked on. Default is all in the dataframe.

    """
    print('Missing xcats across df: ', set(xcats) - set(df['xcat'].unique()))  # any xcats missing
    xcats_used = sorted(list(set(xcats).intersection(set(df['xcat'].unique()))))
    for xcat in xcats_used:
        cids_xcat = df.loc[df['xcat'] == xcat, 'cid'].unique()
        print(f'Missing cids for {xcat}: ', set(cids) - set(cids_xcat))  # any cross section missing?


def reduce_df(df: pd.DataFrame, xcats: List[str] = None,  cids: List[str] = None,
              start: str = None, end: str = None, blacklist: dict = None, out_all: bool = False):
    """
    Filter dataframe by xcats and cids and notify about missing xcats and cids

    :param <pd.Dataframe> df: standardized dataframe with the following necessary columns:
        'cid', 'xcats', 'real_date'.
    :param <List[str]> xcats: extended categories to be checked on. Default is all in the dataframe.
    :param <List[str]> cids: cross sections to be checked on. Default is all in the dataframe.
    :param <str> start: string representing earliest date. Default is None.
    :param <str> end: string representing the latest date. Default is None.
    :param <dict> blacklist: cross sections with date ranges that should be excluded from the data frame.
    :param <bool> out_all: if True the function returns reduced dataframe and selected/available xcats and cids.
        Default is False, i.e. only the dataframe is returned

    :return <pd.Dataframe>: reduced dataframe
        or (for out_all True) dataframe and avalaible and selected xcats and cids
    """

    dfx = df[df['real_date'] >= pd.to_datetime(start)] if start is not None else df
    dfx = dfx[dfx['real_date'] <= pd.to_datetime(end)] if end is not None else dfx

    if blacklist is not None:
        for key, value in blacklist.items():
            filt1 = dfx['cid'] == key[:3]
            filt2 = dfx['real_date'] >= pd.to_datetime(value[0])
            filt3 = dfx['real_date'] <= pd.to_datetime(value[1])
            dfx = dfx[~(filt1 & filt2 & filt3)]

    xcats_in_df = dfx['xcat'].unique()
    if xcats is None:
        xcats = sorted(xcats_in_df)
    else:
        missing = sorted(set(xcats) - set(xcats_in_df))
        if len(missing) > 0:
            print(f'Missing cross sections: {missing}')
        xcats = sorted(list(set(xcats).intersection(set(xcats_in_df))))

    dfx = dfx[dfx['xcat'].isin(xcats)]

    cids_in_df = dfx['cid'].unique()
    if cids is None:
        cids = sorted(cids_in_df)
    else:
        if not isinstance(cids, list):
           cids = [cids]
        missing = sorted(set(cids) - set(cids_in_df))
        if len(missing) > 0:
            print(f'Missing cross sections: {missing}')
        cids = sorted(list(set(cids).intersection(set(cids_in_df))))
        dfx = dfx[dfx['cid'].isin(cids)]

    if out_all:
        return dfx, xcats, cids
    else:
        return dfx


def check_startyears(df: pd.DataFrame):
    """
    Dataframe with starting years across all extended categories and cross sections

    :param <pd.Dataframe> df: standardized dataframe with the following necessary columns:
    'cid', 'xcats', 'real_date'

    """
    df_starts = df[['cid', 'xcat', 'real_date']].groupby(['cid', 'xcat']).min()
    df_starts['real_date'] = pd.DatetimeIndex(df_starts.loc[:, 'real_date']).year

    return df_starts.unstack().loc[:, 'real_date'].astype(int, errors='ignore')


def check_enddates(df: pd.DataFrame):
    """
    Dataframe with end dates across all extended categories and cross sections

    :param <pd.Dataframe> df: standardized dataframe with the following necessary columns:
    'cid', 'xcats', 'real_date'
    """

    df_ends = df[['cid', 'xcat', 'real_date']].groupby(['cid', 'xcat']).max()
    df_ends['real_date'] = df_ends['real_date'].dt.strftime('%Y-%m-%d')

    return df_ends.unstack().loc[:, 'real_date']


def visual_paneldates(df: pd.DataFrame, size: Tuple[float] = None):
    """
    Visualize panel dates with color codes.

    :param <pd.Dataframe> df: dataframe cross sections rows and category columns
    :param <Tuple[float]> size: tuple of floats with width/length of displayed heatmap

    """

    if all(df.dtypes == object):

        df = df.apply(pd.to_datetime)
        maxdate = df.max().max()
        df = (maxdate - df).apply(lambda x: x.dt.days)
        header = f"Missing days prior to {maxdate.strftime('%Y-%m-%d')}"

    else:

        header = 'Start years of quantamental indicators'

    if size is None:
        size = (max(df.shape[0] / 2, 15), max(1, df.shape[1]/ 2))
    sns.set(rc={'figure.figsize': size})
    sns.heatmap(df.T, cmap='Reds', center=df.stack().mean(), annot=True, fmt='.0f', linewidth=1, cbar=False)
    plt.xlabel('')
    plt.ylabel('')
    plt.title(header, fontsize=18)
    plt.show()


def check_availability(df: pd.DataFrame, xcats: List[str] = None,  cids: List[str] = None, start: str = None,
                       start_size: Tuple[float] = None, end_size: Tuple[float] = None):
    """
    Wrapper for visualizing start and end dates of a filtered dataframe.

    :param <pd.Dataframe> df: standardized dataframe with the following necessary columns:
        'cid', 'xcats', 'real_date'.
    :param <List[str]> xcats: extended categories to be checked on. Default is all in the dataframe.
    :param <List[str]> cids: cross sections to be checked on. Default is all in the dataframe.
    :param <str> start: string representing earliest considered date. Default is None.
    :param <Tuple[float]> start_size: tuple of floats with width/length of start years heatmap. Default is None.
    :param <Tuple[float]> end_size: tuple of floats with width/length of end dates heatmap. Default is None.

    """
    dfx = reduce_df(df, xcats=xcats, cids=cids, start=start)
    dfs = check_startyears(dfx)
    visual_paneldates(dfs, size=start_size)
    dfe = check_enddates(dfx)
    visual_paneldates(dfe, size=end_size)


if __name__ == "__main__":

    cids = ['AUD', 'CAD', 'GBP']
    xcats = ['XR', 'CRY']
    df_cids = pd.DataFrame(index=cids, columns=['earliest', 'latest', 'mean_add', 'sd_mult'])
    df_cids.loc['AUD', ] = ['2010-01-01', '2020-12-31', 0.5, 2]
    df_cids.loc['CAD', ] = ['2011-01-01', '2020-11-30', 0, 1]
    df_cids.loc['GBP', ] = ['2012-01-01', '2020-11-30', -0.2, 0.5]

    df_xcats = pd.DataFrame(index=xcats, columns=['earliest', 'latest', 'mean_add', 'sd_mult', 'ar_coef', 'back_coef'])
    df_xcats.loc['XR', ] = ['2010-01-01', '2020-12-31', 0, 1, 0, 0.3]
    df_xcats.loc['CRY', ] = ['2011-01-01', '2020-10-30', 1, 2, 0.9, 0.5]

    dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)

    xxcats = xcats+['TREND']
    xxcids = cids+['USD']
    missing_in_df(dfd, xcats=xxcats, cids=xxcids)

    # dfd_x1 = reduce_df(dfd, xcats=xcats, cids=cids[0], start='2012-01-01', end='2018-01-31')
    # dfd_x = reduce_df(dfd, xcats=xcats, cids=cids, start='2012-01-01', end='2018-01-31')

    black = {'CAD': ['2010-01-01', '2013-12-31'], 'GBP': ['2018-01-01', '2100-01-01']}
    dfd_xb = reduce_df(dfd, xcats=xcats, cids=cids, blacklist=black)

    df_sy = check_startyears(dfd_xb)
    print(df_sy)
    visual_paneldates(df_sy)
    df_ed = check_enddates(dfd_xb)
    print(df_ed)
    visual_paneldates(df_ed)
    df_ed.info()