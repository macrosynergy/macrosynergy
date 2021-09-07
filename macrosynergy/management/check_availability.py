import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Union, Tuple
import time

from simulate_quantamental_data import make_qdf_
from shape_dfs import reduce_df


def check_availability(df: pd.DataFrame, xcats: List[str] = None,  cids: List[str] = None, start: str = None,
                       start_size: Tuple[float] = None, end_size: Tuple[float] = None):
    """
    Wrapper for visualizing start and end dates of a filtered dataframe.
    """
    dfx = reduce_df(df, xcats=xcats, cids=cids, start=start)
    dfs = check_startyears(dfx)
    visual_paneldates(dfs, size=start_size)
    dfe = check_enddates(dfx)
    visual_paneldates(dfe, size=end_size)

## The Python runtime does not enforce function and variable type annotations.
def missing_in_df(fields, fields_, xcats: List[str] = None,  cids: List[str] = None):
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
    

def visual_paneldates(df: pd.DataFrame, size: Tuple[float] = None):
    """
    Visualize panel dates with color codes.
    """
    header = "Start years of Quantamental Indicators."
    
    if all(df.dtypes == object):
        
        df = df.apply(pd.to_datetime)
        maxdate = df.max().max()
        df = (maxdate - df).apply(lambda x: x.dt.days)
        header = f"Missing days prior to {maxdate.strftime('%Y-%m-%d')}."

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

    final_df, fields_cats, fields_cids, df_year, df_missing, cids_cats = make_qdf_(df_cids, df_xcats, back_ar = 0.75)

    xxcats = xcats + ['TREND']
    xxcids = cids + ['USD']

    missing_in_df(fields_cats, fields_cids, xcats = xxcats)
    
    visual_paneldates(df_year)
    
    visual_paneldates(df_missing)
    ## df_ed.info()
