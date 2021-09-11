import numpy as np
import pandas as pd
# from Test_File import make_qdf_, simulate_ar  # not available
from collections import defaultdict, deque
import time
import matplotlib.pyplot as plt
from typing import List, Union, Tuple

from macrosynergy.management.simulate_quantamental_data import make_qdf
from macrosynergy.management.shape_dfs import categories_df


def historic_vol(df: pd.DataFrame, xcat: str, cids: List[str] = None, start: str = None, end: str = None,
                 lback_meth: str = 'xma', lback_periods: int = 21, remove_zeros: bool = True,
                 cutoff: float = 0.01, postfix: str = 'ASD'):

    """
    Estimate historic annualized standard deviations of asset returns

    :param <pd.Dataframe> df: standardized data frame with the following necessary columns:
    'cid', 'xcats', 'real_date' and 'value.
    :param <str> xcat:  extended category denoting the return series for which volatility should be calculated.
    :param <List[str]> cids: cross sections for which volatility is calculated;
        default is all available for the category.
    :param <str> start: earliest date in ISO format. Default is None and earliest date in df is used.
    :param <str> end: latest date in ISO format. Default is None and latest date in df is used.
    :param <str> lback_meth: Lookback method to calculate the volatility, Default is "xma" (exponential moving average).
        Alternative is "ma", simple moving average.
    :param <int>  lback_periods: Number of lookback periods over which volatility is calculated. Default is 21.
        Refers to half-time for "xma" and full lookback period for "ma".
    :param <bool> remove_zeros: if True (default) any returns that are exact zeroes will not be included in the
        lookback window and prior non-zero values are added to the window instead.
    :param <float> cutoff: share of past observation weights in the exponential moving average that is disregarded.
        This prevents NaNs in distant history from propagating. Default is 0.01
    :param <str> postfix: string appended to category name for output; default is "ASD".

    :return <pd.Dataframe>: standardized dataframe with the estimated annualized standard deviations
    """
    pass

if __name__ == "__main__":

    cids = ['AUD', 'CAD', 'GBP', 'NZD']
    xcats = ['FXXR', 'EQXR', 'DUXR']
    df_cids = pd.DataFrame(index=cids, columns=['earliest', 'latest', 'mean_add', 'sd_mult'])
    df_cids.loc['AUD',] = ['2000-01-01', '2020-12-31', 0.1, 1]
    df_cids.loc['CAD',] = ['2001-01-01', '2020-11-30', 0, 1]
    df_cids.loc['GBP',] = ['2002-01-01', '2020-11-30', 0, 2]
    df_cids.loc['NZD',] = ['2002-01-01', '2020-09-30', -0.1, 2]

    df_xcats = pd.DataFrame(index=xcats, columns=['earliest', 'latest', 'mean_add', 'sd_mult', 'ar_coef', 'back_coef'])
    df_xcats.loc['FXXR',] = ['2000-01-01', '2020-12-31', 0.1, 1, 0, 0.3]
    df_xcats.loc['EQXR',] = ['2000-01-01', '2020-10-30', 1, 2, 0.95, 1]
    df_xcats.loc['DUXR',] = ['2001-01-01', '2020-10-30', 1, 2, 0.9, 1]


    dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)