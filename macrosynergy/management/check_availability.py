import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Union, Tuple
import time

from Simulate_Quantamental import make_qdf
from Shape_DataFrame import reduce_df
from Test_File import make_qdf_


def check_enddates(df: pd.DataFrame):
    """
    Dataframe with end dates across all extended categories and cross sections
    :param <pd.Dataframe> df: standardized dataframe with the following necessary columns:
    'cid', 'xcats', 'real_date'
    """

    df_ends = df[['cid', 'xcat', 'real_date']].groupby(['cid', 'xcat']).max()
    df_ends['real_date'] = df_ends['real_date'].dt.strftime('%Y-%m-%d')

    return df_ends.unstack().loc[:, 'real_date']



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

## The Python runtime does not enforce function and variable type annotations.
def missing_in_df(df: pd.DataFrame, fields, fields_, xcats: List[str] = None,  cids: List[str] = None):
    """Print cross sections and extended categories that are missing or redundant in the dataframe
    """
    ## print('Missing xcats across df: ', set(xcats) - set(df['xcat'].unique()))  # any xcats missing
    print(f"Missing xcats across df: {set(xcats + fields)}.")
    cids = fields_
    cids_set = set(cids)

    xcats_used = set(xcats).intersection(set(fields))
    
    for xcat in xcats_used:
        cids_xcat = cids_set
        
        print(f"Missing cids for {xcat}: {cids_set - cids_xcat}.")


def check_startyears(df: pd.DataFrame):
    """
    Dataframe with starting years across all extended categories and cross sections
    """
    df_starts = df[['cid', 'xcat', 'real_date']].groupby(['cid', 'xcat']).min()
    df_starts['real_date'] = pd.DatetimeIndex(df_starts.loc[:, 'real_date']).year
    print(df_starts.unstack(level = -1).astype(int, errors = 'ignore'))

    ## df_starts.unstack(level = -1).astype(int)
    
    return df_starts.unstack().loc[:, 'real_date'].astype(int, errors = 'ignore')
    

def visual_paneldates(df: pd.DataFrame, size: Tuple[float] = None):
    """
    Visualize panel dates with color codes.
    """

    if all(df.dtypes == object):
        
        df = df.apply(pd.to_datetime)
        maxdate = df.max().max()
        df = (maxdate - df).apply(lambda x: x.dt.days)
        header = f"Missing days prior to {maxdate.strftime('%Y-%m-%d')}"

    else:

        header = "Start years of quantamental indicators."

    if size is None:
        size = (max(df.shape[0] / 2, 15), max(1, df.shape[1]/ 2))
    sns.set(rc = {'figure.figsize': size})
    sns.heatmap(df.T, cmap='YlOrBr', center = df.stack().mean(), annot = True, fmt = '.0f', linewidth = 1, cbar = False)
    plt.xlabel('')
    plt.ylabel('')
    plt.title(header, fontsize=18)
    plt.show()



if __name__ == "__main__":

    ## Country IDs.
    cids = ['AUD', 'CAD', 'GBP']
    xcats = ['XR', 'CRY']
    ## Mean Adjusted & Standard Deviation multiple.
    df_cids = pd.DataFrame(index = cids, columns = ['earliest', 'latest', 'mean_add', 'sd_mult'])

    
    ## "IndexError: tuple index out of range."
    ## Access a group of rows and columns by label(s) or a boolean array.
    df_cids.loc['AUD'] = ['2010-01-01', '2020-12-31', 0.5, 2]
    df_cids.loc['CAD'] = ['2011-01-01', '2020-11-30', 0, 1]
    df_cids.loc['GBP'] = ['2012-01-01', '2020-11-30', -0.2, 0.5]

    df_xcats = pd.DataFrame(index = xcats, columns = ['earliest', 'latest', 'mean_add', 'sd_mult', 'ar_coef', 'back_coef'])
    df_xcats.loc['XR'] = ['2010-01-01', '2020-12-31', 0, 1, 0, 0.3]
    df_xcats.loc['CRY'] = ['2011-01-01', '2020-10-30', 1, 2, 0.9, 0.5]
    ## Establishing the dataframe.

    dfd = make_qdf(df_cids, df_xcats, back_ar = 0.75)

    fields_cats, fields_cids = make_qdf_(df_cids, df_xcats, back_ar = 0.75)

    xxcats = xcats + ['TREND']
    xxcids = cids + ['USD']

    missing_in_df(dfd, fields_cats, fields_cids, xcats = xxcats)

    
    df_sy = check_startyears(dfd)
    
    visual_paneldates(df_sy)
    ## df_ed = check_enddates(dfd)
    ## print(df_ed)
    ## visual_paneldates(df_ed)
    ## df_ed.info()
