
import time
import numpy as np
import pandas as pd
from collections import defaultdict, deque
from typing import List, Union, Tuple
from random import choice
from macrosynergy.management.simulate_quantamental_data import make_qdf
from macrosynergy.management.shape_dfs import reduce_df



def pandas_exponential(arr, half_life, cutoff):
    """
    Compute annualised rolling Exponential Moving Average per cross section.
    Receives a cross-section of return data for the specific xcat.
    
    :param <ndarray[timestamp, float] arr>: Two-dimensional Array consisting of timestamps and the realised return for the date.
    :param <int> half_life: Half-life - number of days 50% of the weighting is applied to.
    :param <float> cutoff: share of past observation weights in the exponential moving average that are disregarded.

    Annualised EMA for the specific Cross-Section.
    """
        
    arr = arr[:, 1]
    s = pd.Series(arr)
    df_ewm = s.ewm(halflife = half_life, min_periods = half_life * 2).std()

    df_ewm = df_ewm * np.sqrt(256)
    return df_ewm

def pandas_ma(arr, n):
    """
    Compute annualised rolling Moving Average per cross section.
    Receives a cross-section of return data for the specific xcat.
    
    :param <ndarray[timestamp, float] arr>: Two-dimensional Array consisting of timestamps and the realised return for the date.
    :param <int> n: Window for the Moving Average.

    Annualised STD for the specific Cross-Section.
    """

    arr = arr[:, 1]
    s = pd.Series(arr)
    df_std = s.rolling(n).std()
    df_std = df_std * np.sqrt(256)

    return df_std

def func(str_, int_):
    """
    Used for broadcasting. Determine the number of timestamps per cross section.
    """
    return [str_] * int_

def mult(xcats, list_):

    return list(map(func, xcats, list_))

def driver(dfd: pd.DataFrame, cids: List[str] = None, xcat: str = None, lback_meth: str = 'MA',
           lback_period: int = 21, half_life: int = 21, cutoff: int = 0.01):

    """
    Estimate historic annualized standard deviations of asset returns. Driver Function. Controls the functionality.

    :param <pd.Dataframe> df: standardized data frame with the following necessary columns: Reduced dataframe on the specific xcat.
    :param <List[str]> cids: cross sections for which volatility is calculated;
    :param <str> xcat:  extended category denoting the return series for which volatility should be calculated.
    :param <str> lback_meth: Lookback method to calculate the volatility, Default is "MA". Alternative is "EMA", Exponential Moving Average.
    :param <int>  lback_period: Number of lookback periods over which volatility is calculated. Default is 21.
    :param <int> half_life: Refers to the half-time for "xma" and full lookback period for "MA".
    :param <float> cutoff: share of past observation weights in the exponential moving average that are disregarded. This prevents NaNs in distant history from propagating. Default is 0.01
    :param <str> postfix: string appended to category name for output; default is "ASD".

    :return <pd.Dataframe>: standardized dataframe with the estimated annualized standard deviations of the chosen xcat.
    'cid', 'xcat', 'real_date' and 'RollingSTD'.
    """

    assert lback_period > 0
    assert xcat is not None

    cid_xcat = {}
    xcat = defaultdict(list)
    cid_pandas = {}

    for cid in cids:
        df_ = dfd[dfd['cid'] == cid]
        data = df_[['real_date', 'value']].to_numpy()
        xcat[cid].append(len(data[:, 0]))
        cid_xcat[cid] = data

    for cid in cids:
        if lback_meth == 'MA':
            cid_pandas[cid] = pandas_ma(cid_xcat[cid], lback_period)
        else:
            cid_pandas[cid] = pandas_exponential(cid_xcat[cid], half_life, cutoff)
            
    
    qdf_cols = ['cid', 'xcat', 'real_date', lback_meth]
    df_lists = []
    for cid in cids:
        df_out = pd.DataFrame(columns = qdf_cols)
        df_out['xcat'] = np.concatenate(mult(xcats, xcat[cid]))
        df_out['real_date'] = cid_xcat[cid][:, 0]
        df_out[lback_meth] = cid_pandas[cid]
        df_out['cid'] = cid
        df_lists.append(df_out)

    final_df = pd.concat(df_lists, ignore_index = True)
        
    return final_df


def historic_vol(dfd: pd.DataFrame, cids: List[str] = None, xcats: List[str] = None,
                 xcat: str = None, lback_period: int = 21, lback_meth: str = 'MA',
                 half_life: int = 21, remove_zeros: bool = True, cutoff: int = 0.01):

    """
    Estimate historic annualized standard deviations of asset returns. User Function. Controls the functionality.

    :param <pd.Dataframe> df: standardized data frame with the following necessary columns:
    'cid', 'xcats', 'real_date' and 'value. Will contain all of the data across all macroeconomic fields.
    :param <str> xcat:  extended category denoting the return series for which volatility should be calculated.
    :param <List[str]> cids: cross sections for which volatility is calculated;
        default is all available for the category.
    :param <List[str]> xcats: possible categories in which volatility is computed.
    :param <str> start: earliest date in ISO format. Default is None and earliest date in df is used.
    :param <str> end: latest date in ISO format. Default is None and latest date in df is used.
    :param <int>  lback_period: Number of lookback periods over which volatility is calculated. Default is 21.
    :param <str> lback_meth: Lookback method to calculate the volatility, Default is "MA". Alternative is "EMA", Exponential Moving Average.
    :param <int> half_life: Refers to the half-time for "xma" and full lookback period for "MA".
    :param <bool> remove_zeros: if True (default) any returns that are exact zeros will not be included in the lookback window and prior non-zero values are added to the window instead.
    :param <float> cutoff: share of past observation weights in the exponential moving average that are disregarded. This prevents NaNs in distant history from propagating. Default is 0.01
    :param <str> postfix: string appended to category name for output; default is "ASD".

    :return <pd.Dataframe>: standardized dataframe with the estimated annualized standard deviations of the chosen xcat.
    'cid', 'xcat', 'real_date' and 'RollingSTD'.
    """

    assert xcat in xcats

    xcat_filter = (dfd['xcat'] == xcat).to_numpy()
    df_xcat = dfd[xcat_filter]
    df_xcat.reset_index(drop = True)
    
    if remove_zeros:
        dfd = remove_zeros(df_xcat)

    rolling_df = driver(dfd = df_xcat, cids = cids, xcat = xcat, lback_meth = lback_meth,
                        lback_period = lback_period, half_life = half_life, cutoff = cutoff)


    return rolling_df.reset_index(drop = True)


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

    ## dfd, fields_cats, fields_cids, df_year, df_end, df_missing, cids_cats = make_qdf_(df_cids, df_xcats, back_ar = 0.75)
    dfd = make_qdf(df_cids, df_xcats, back_ar = 0.75)

    start = time.time()
    df = historic_vol(dfd, cids, xcats, xcat = 'INFL', lback_period = 42, lback_meth = 'MA', half_life = 21, remove_zeros = False, cutoff = 0.01)
    print(f"Time Elapsed: {time.time() - start}.")