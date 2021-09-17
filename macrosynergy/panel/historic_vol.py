
import time
import numpy as np
import pandas as pd
from collections import defaultdict, deque
from typing import List, Union, Tuple
import matplotlib.pyplot as plt
from random import choice
## from Test_File import make_qdf_, simulate_ar
from macrosynergy.management.simulate_quantamental_data import make_qdf


def remove_zeros(df: pd.DataFrame):
    values = df['value'].to_numpy()
    bool_ = values == 0.0 
    if any(bool_):
        print("Entered.")
        bool_ = ~bool_
        df = df[bool_]

    return df

def slide(arr, w, s = 1):

    return np.lib.stride_tricks.as_strided(arr,
                                           shape = ((len(arr) - w) + 1, w),
                                           strides = arr.strides * 2)[::s]

## hft -> 21. tx = days.
def expo_weights(tx: int, hft: int = 21, cutoff = 0.01):
    
    decf = 2 ** (-1 / hft)
    weights = (1 - decf) * np.array([decf ** (tx - ii - 1) for ii in range(tx)])

    return weights

def weight_series(weights, n_days):

    w_series = np.zeros(n_days)
    
    weights = weights[-n_days:]
    if weights[0] < 0.01:
        weights = weights[weights >= 0.01]

    active_w = len(weights)
    
    weights = weights / weights.sum(axis = 0)
    w_series[-active_w:] = weights
    
    return w_series


def rolling_std_stride(lists, n):

    def compute(arr):
        
        arr = arr[:, 1]
        arr = slide(arr, n)
        ## The original Array will be of type Object because it was previously hosting datetime objects.
        arr = arr.astype(dtype = float)
        std_arr = np.std(arr, axis = 1)
        std_arr = std_arr * np.sqrt(256)
        std_arr = np.append(np.zeros(n - 1) + np.nan, std_arr)
        return std_arr
        
    return list(map(compute, lists))

def rolling_ema(lists, half_life, cutoff):

    def compute(arr):
        
        arr = arr[:, 1] ## Return series.
        time = len(arr)
        rolling_std = np.zeros(time, dtype = float)

        n = half_life * 2
        ## Tracking data from the incision onwards. Instantiate the Deque.
        ## Expanding iteratively until the cutoff becomes active.
        window = list(arr[:(n - 1)])
        w_series = expo_weights(time, half_life, cutoff)
        
        for i, _ in enumerate(arr[n:]):
            window.append(_)
            ## The length of the window grows in accordance with the expansion of the return series. Will be truncated by the "cutoff" parameter.
            n_days = len(window)
            w_series = weight_series(w_series, n_days)
            win_arr = np.array(window)
            win_mean = np.mean(win_arr)
            scale = np.multiply(w_series, ((win_arr - win_mean) ** 2))
            numerator = np.sum(scale)
            std = np.sqrt(numerator)
            
            rolling_std[(n + i)] = std * np.sqrt(256) ## Annualise.
        
        return rolling_std
    
    return list(map(compute, lists))

def merge(arr):

    return arr[:, 0]

def func(str_, int_):
    return [str_] * int_

def mult(xcats, list_):

    return list(map(func, xcats, list_))


## The parameter, "window", is only applicable for the Moving Average which has a fixed window, and the weighting on the return series, inside the window, is applied uniformally.
def smoothing_func(dfd: pd.DataFrame, cids: List[str] = None, xcats: List[str] = None,
                   xcat: str = None, lback_period: int = None, half_life: int = 21,
                   cutoff: int = 0.01, lback_meth: str = 'MA'):

    assert lback_period > 0
    assert xcat is not None

    cid_xcat = {}
    cid_stride = {}
    xcat = defaultdict(list)

    for cid in cids:
        df_temp = dfd[dfd['cid'] == cid]

        def indicators(cat):
            df_ = df_temp[df_temp['xcat'] == cat]
            data = df_[['real_date', 'value']].to_numpy()
            xcat[cid].append(len(data[:, 0]))
            
            return data

        cid_xcat[cid] = list(map(indicators, xcats))

    for cid in cids:
        if lback_meth == 'MA':
            cid_stride[cid] = rolling_std_stride(cid_xcat[cid], lback_period)
        else:
            cid_stride[cid] = rolling_ema(cid_xcat[cid], half_life, cutoff)

    qdf_cols = ['cid', 'xcat', 'real_date', lback_meth]
    df_lists = []
    for cid in cids:
        df_out = pd.DataFrame(columns = qdf_cols)
        df_out['xcat'] = np.concatenate(mult(xcats, xcat[cid]))
        list_ = tuple(map(merge, cid_xcat[cid]))
        df_out['real_date'] = np.concatenate(list_)
        df_out[lback_meth] = np.concatenate(cid_stride[cid])
        df_out['cid'] = cid
        df_lists.append(df_out)

    final_df = pd.concat(df_lists, ignore_index = True)
        
    return final_df


def historic_vol(dfd: pd.DataFrame, cids: List[str] = None, xcats: List[str] = None,
                 xcat: str = None, lback_period: int = None, half_life: int = 21,
                 cutoff: int = 0.01, lback_meth: str = 'MA', zeros: bool = True):

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

    if zeros:
        dfd = remove_zeros(dfd)

    start = time.time()
    rolling_df = smoothing_func(dfd, cids, xcats, xcat, lback_period, half_life,
                                cutoff, lback_meth)
    print(f"Time Elapsed: {time.time() - start}.")

    xcat_filter = (rolling_df['xcat'] == xcat).to_numpy()
    df_xcat = rolling_df[xcat_filter]

    return df_xcat.reset_index(drop = True)
    

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
    dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)

    ## weights = expo_weights(1200, 11, cutoff = 0.01)


    df = historic_vol(dfd, cids, xcats, lback_period = 42, half_life = 21, cutoff = 0.01, lback_meth = 'MA', xcat = choice(xcats))    
    print(df)
