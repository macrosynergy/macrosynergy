
import time
import numpy as np
import pandas as pd
from collections import defaultdict, deque
from typing import List, Union, Tuple
from random import choice
from macrosynergy.management.simulate_quantamental_data import make_qdf
from macrosynergy.management.shape_dfs import reduce_df


def historic_vol(df: pd.DataFrame, xcat: str = None, cids: List[str] = None,
                 lback_periods: int = 21, lback_meth: str = 'ma', half_life=11,
                 start: str = None, end: str = None, blacklist: dict = None,
                 remove_zeros: bool = True, postfix='ASD'):

    """
    Estimate historic annualized standard deviations of asset returns. User Function. Controls the functionality.

    :param <pd.Dataframe> df: standardized data frame with the following necessary columns:
    'cid', 'xcats', 'real_date' and 'value. Will contain all of the data across all macroeconomic fields.
    :param <str> xcat:  extended category denoting the return series for which volatility should be calculated.
    :param <List[str]> cids: cross sections for which volatility is calculated;
        default is all available for the category.
    :param <int>  lback_periods: Number of lookback periods over which volatility is calculated. Default is 21.
    :param <str> lback_meth: Lookback method to calculate the volatility, Default is "MA". Alternative is "EMA",
        Exponential Moving Average.
    :param <str> start: earliest date in ISO format. Default is None and earliest date in df is used.
    :param <str> end: latest date in ISO format. Default is None and latest date in df is used.
    :param <dict> blacklist: cross sections with date ranges that should be excluded from the data frame.
        If one cross section has several blacklist periods append numbers to the cross section code.
    :param <int> half_life: Refers to the half-time for "xma" and full lookback period for "ma".
    :param <bool> remove_zeros: if True (default) any returns that are exact zeros will not be included in the lookback
        window and prior non-zero values are added to the window instead.
    :param <str> postfix: string appended to category name for output; default is "ASD".

    :return <pd.Dataframe>: standardized dataframe with the estimated annualized standard deviations of the chosen xcat.
    'cid', 'xcat', 'real_date' and 'value'.
    """

    assert lback_periods > half_life, "Half life must be shorter than lookback period"
    # Todo: other key asserts

    df = reduce_df(df, xcats=[xcat], cids=cids, start=start, end=end, blacklist=blacklist)

    dfw = df.pivot(index='real_date', columns='cid', values='value')
    if lback_meth == 'xma':
        dfwa = np.sqrt(252) * dfw.rolling(window=lback_periods, win_type='exponential').std()
        # Todo: change to use solution 2 of
        #  https://stackoverflow.com/questions/57518576/how-to-use-df-rollingwindow-min-periods-win-type-exponential-sum
        #  such that the half-life information is used.
    else:
        if remove_zeros:
            dfwa = np.sqrt(252) * dfw.rolling(window=lback_periods).apply(lambda x: np.mean(np.abs(x)[x != 0]))
        else:
            dfwa = np.sqrt(252) * dfw.rolling(window=lback_periods).apply(lambda x: np.mean(np.abs(x)))

    df_out = dfwa.unstack().reset_index().rename(mapper={0: 'value'}, axis=1)
    df_out['xcat'] = xcat + postfix

    return df_out[df.columns]


if __name__ == "__main__":

    
    ## Country IDs.
    cids = ['AUD', 'CAD', 'GBP', 'USD']
    
    xcats = ['XR', 'CRY', 'GROWTH', 'INFL']

    
    df_cids = pd.DataFrame(index = cids, columns = ['earliest', 'latest', 'mean_add', 'sd_mult'])

    df_cids.loc['AUD'] = ['2010-01-01', '2020-12-31', 0.5, 2]
    df_cids.loc['CAD'] = ['2011-01-01', '2020-11-30', 0, 1]
    df_cids.loc['GBP'] = ['2012-01-01', '2020-10-30', -0.2, 0.5]
    df_cids.loc['USD'] = ['2013-01-01', '2020-09-30', -0.2, 0.5]
    df_cids.loc['NZD'] = ['2002-01-01', '2020-09-30', -0.1, 2]

    df_xcats = pd.DataFrame(index = xcats, columns = ['earliest', 'latest', 'mean_add', 'sd_mult', 'ar_coef', 'back_coef'])
    df_xcats.loc['XR'] = ['2010-01-01', '2020-12-31', 0, 1, 0, 0.3]
    df_xcats.loc['CRY'] = ['2011-01-01', '2020-10-30', 1, 2, 0.9, 0.5]
    df_xcats.loc['GROWTH'] = ['2012-01-01', '2020-10-30', 1, 2, 0.9, 1]
    df_xcats.loc['INFL'] = ['2013-01-01', '2020-10-30', 1, 2, 0.8, 0.5]

    dfd = make_qdf(df_cids, df_xcats, back_ar = 0.75)

    start = time.time()
    df = historic_vol(dfd, cids=cids, xcat='XR', lback_periods=42, lback_meth='MA', half_life=21,
                      remove_zeros=True)

    print(f"Time Elapsed: {time.time() - start}.")