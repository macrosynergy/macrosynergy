import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Union, Tuple

from macrosynergy.management.simulate_quantamental_data import make_qdf
from macrosynergy.management.shape_dfs import reduce_df


def missing_in_df(df: pd.DataFrame, xcats: List[str] = None,  cids: List[str] = None):
    """
    Print cross sections and extended categories that are missing or redundant in the
    dataframe

    :param <pd.Dataframe> df: standardized dataframe with the following necessary
        columns: 'cid', 'xcats', 'real_date'.
    :param <List[str]> xcats: extended categories to be checked on. Default is all
        in the dataframe.
    :param <List[str]> cids: cross sections to be checked on. Default is all in
        the dataframe.

    """
    print('Missing xcats across df: ', set(xcats) - set(df['xcat'].unique()))
    cids = df['cid'].unique() if cids is None else cids
    xcats_used = sorted(list(set(xcats).intersection(set(df['xcat'].unique()))))
    for xcat in xcats_used:
        cids_xcat = df.loc[df['xcat'] == xcat, 'cid'].unique()
        print(f'Missing cids for {xcat}: ', set(cids) - set(cids_xcat))


def check_startyears(df: pd.DataFrame):
    """
    Dataframe with starting years across all extended categories and cross sections

    :param <pd.Dataframe> df: standardized dataframe with the following necessary
        columns: 'cid', 'xcats', 'real_date'

    """

    df = df.dropna(how='any')
    df_starts = df[['cid', 'xcat', 'real_date']].groupby(['cid', 'xcat']).min()
    df_starts['real_date'] = pd.DatetimeIndex(df_starts.loc[:, 'real_date']).year

    return df_starts.unstack().loc[:, 'real_date'].astype(int, errors='ignore')


def check_enddates(df: pd.DataFrame):
    """
    Dataframe with end dates across all extended categories and cross sections

    :param <pd.Dataframe> df: standardized dataframe with the following necessary
        columns: 'cid', 'xcats', 'real_date'
    """

    df = df.dropna(how='any')
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
    sns.heatmap(df.T, cmap='YlOrBr', center=df.stack().mean(), annot=True, fmt='.0f',
                linewidth=1, cbar=False)
    plt.xlabel('')
    plt.ylabel('')
    plt.title(header, fontsize=18)
    plt.show()


def check_availability(df: pd.DataFrame, xcats: List[str] = None,
                       cids: List[str] = None, start: str = None,
                       start_size: Tuple[float] = None, end_size: Tuple[float] = None):
    """
    Wrapper for visualizing start and end dates of a filtered dataframe.

    :param <pd.Dataframe> df: standardized dataframe with the following necessary
        columns: 'cid', 'xcats', 'real_date'.
    :param <List[str]> xcats: extended categories to be checked on.
        Default is all in the dataframe.
    :param <List[str]> cids: cross sections to be checked on.
        Default is all in the dataframe.
    :param <str> start: string representing earliest considered date. Default is None.
    :param <Tuple[float]> start_size: tuple of floats with width/length of
        the start years heatmap. Default is None (format adjusted to data).
    :param <Tuple[float]> end_size: tuple of floats with width/length of
        the end dates heatmap. Default is None (format adjusted to data).

    """
    dfx = reduce_df(df, xcats=xcats, cids=cids, start=start)
    dfs = check_startyears(dfx)
    visual_paneldates(dfs, size=start_size)
    dfe = check_enddates(dfx)
    visual_paneldates(dfe, size=end_size)


if __name__ == "__main__":

    cids = ['AUD', 'CAD', 'GBP']
    xcats = ['XR', 'CRY']

    cols_1 = ['earliest', 'latest', 'mean_add', 'sd_mult']
    df_cids = pd.DataFrame(index=cids, columns=cols_1)
    df_cids.loc['AUD', ] = ['2010-01-01', '2020-12-31', 0.5, 2]
    df_cids.loc['CAD', ] = ['2010-01-01', '2020-11-30', 0, 1]
    df_cids.loc['GBP', ] = ['2012-01-01', '2020-11-30', -0.2, 0.5]

    cols_2 = cols_1 + ['ar_coef', 'back_coef']
    df_xcats = pd.DataFrame(index=xcats, columns=cols_2)
    df_xcats.loc['XR', ] = ['2010-01-01', '2020-12-31', 0, 1, 0, 0.3]
    df_xcats.loc['CRY', ] = ['2011-01-01', '2020-10-30', 1, 2, 0.9, 0.5]

    dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)
    filt_na = (dfd['cid'] == 'CAD') & (dfd['real_date'] < '2011-01-01')
    dfd.loc[filt_na, 'value'] = np.nan   # simulate NaNs for CAD 2011

    xxcats = xcats+['TREND']
    xxcids = cids+['USD']
    missing_in_df(dfd, xcats=xxcats)
    missing_in_df(dfd, xcats=xxcats, cids=xxcids)

    df_sy = check_startyears(dfd)
    print(df_sy)
    visual_paneldates(df_sy)
    df_ed = check_enddates(dfd)
    print(df_ed)
    visual_paneldates(df_ed)
    df_ed.info()