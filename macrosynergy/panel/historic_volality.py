import numpy as np
import pandas as pd
from collections import defaultdict, deque
import time
import matplotlib.pyplot as plt
from typing import List, Union, Tuple
from Test_File import make_qdf_, simulate_ar
from random import choice

## from macrosynergy.management.simulate_quantamental_data import make_qdf
## from macrosynergy.management.shape_dfs import categories_df

def remove_zeros(df: pd.DataFrame):
    values = df['value'].to_numpy()
    bool_ = values == 0.0 
    if any(bool_):
        print("Entered.")
        bool_ = ~bool_
        df = df[bool_]

    return df

def expo_weights(tx: int, hft: int = 11):

    decf = 2 ** (-1 / hft)
    weights = (1 - decf) * np.array([decf ** (tx - ii - 1) for ii in range(tx)])
    
    weights = weights / weights.sum(axis = 0)
    return weights

def slide(arr, w, s = 1):

    return np.lib.stride_tricks.as_strided(arr,
                                           shape = ((len(arr) - w) + 1, w),
                                           strides = arr.strides * 2)[::s]
                                                    

def ma_std(lists, n):

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


def exp_std(lists, weights, n):

    def compute_(array):
        
        nonlocal weights
        
        arr = array[:, 1]
        arr = slide(arr, n)
        arr = arr.astype(dtype = float)

        mean = np.mean(arr, axis = 1)
        mean = mean.reshape(len(mean), 1)
        ## Divide by "n" if the returns have a uniform distribution.
        ## Alternatively, multiply by the weights to scale the returns: greater emphasise on the preceding returns (t - 1, t - 2 etc).
        numerator = np.sum(np.multiply(weights, np.square((arr  - mean))), axis = 1)
        std_arr = np.sqrt(numerator)
        std_arr = std_arr * np.sqrt(256)
        std_arr = np.append(np.zeros(n - 1) + np.nan, std_arr)

        return std_arr
    
    return list(map(compute_, lists))

def stand_dev(arr):
    return (np.std(arr[:, 1]) * np.sqrt(255))

def merge(arr):

    return arr[:, 0]

def func(str_, int_):
    return [str_] * int_

def mult(xcats, list_):

    return list(map(func, xcats, list_))

def smoothing_func(dfd: pd.DataFrame, cids: List[str] = None, xcats: List[str] = None,
                   lback_meth: str = 'xma', lback_periods: int = 21):

    cid_xcat = {}
    cid_stride = {}
    xcat = defaultdict(list)

    w = expo_weights(lback_periods, 11)
    
    for cid in cids:
        df_temp = dfd[dfd['cid'] == cid]
        
        def indicators(cat):
            
            df_ = df_temp[df_temp['xcat'] == cat]
            data = df_[['real_date', 'value']].to_numpy()
            xcat[cid].append(len(data[:, 0]))
            
            return data
        
        cid_xcat[cid] = list(map(indicators, xcats))

    for cid in cids:
        if lback_meth == 'ma':
            cid_stride[cid] = ma_std(cid_xcat[cid], lback_periods)
        else:
            cid_stride[cid] = exp_std(cid_xcat[cid], w, lback_periods)

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


def historic_vol(df: pd.DataFrame, xcat: str, cids: List[str] = None, xcats: List[str] = None,
                 lback_meth: str = 'xma', lback_periods: int = 21, start: str = None, end: str = None,
                 zeros: bool = True, cutoff: float = 0.01, postfix: str = 'ASD'):

    if zeros:
        df = remove_zeros(df)
    
    rolling_df = smoothing_func(df, cids, xcats, lback_meth, lback_periods)

    xcat_filter = (rolling_df['xcat'] == xcat).to_numpy()
    df_xcat = rolling_df[xcat_filter]

    return df_xcat.reset_index(drop = True)

if __name__ == "__main__":

    cids = ['AUD', 'CAD', 'GBP', 'NZD']
    xcats = ['FXXR', 'EQXR', 'DUXR']
    df_cids = pd.DataFrame(index = cids, columns = ['earliest', 'latest', 'mean_add', 'sd_mult'])
    df_cids.loc['AUD'] = ['2000-01-01', '2020-12-31', 0.1, 1]
    df_cids.loc['CAD'] = ['2001-01-01', '2020-11-30', 0, 1]
    df_cids.loc['GBP'] = ['2002-01-01', '2020-11-30', 0, 2]
    df_cids.loc['NZD'] = ['2002-01-01', '2020-09-30', -0.1, 2]

    df_xcats = pd.DataFrame(index=xcats, columns=['earliest', 'latest', 'mean_add', 'sd_mult', 'ar_coef', 'back_coef'])
    df_xcats.loc['FXXR'] = ['2000-01-01', '2020-12-31', 0.1, 1, 0, 0.3]
    df_xcats.loc['EQXR'] = ['2000-01-01', '2020-10-30', 1, 2, 0.95, 1]
    df_xcats.loc['DUXR'] = ['2001-01-01', '2020-10-30', 1, 2, 0.9, 1]


    dfd, fields_cats, fields_cids, df_year, df_end, df_missing, cids_cats = make_qdf_(df_cids, df_xcats, back_ar = 0.75)

    start = time.time()
    df = historic_vol(dfd, choice(fields_cats), fields_cids, fields_cats, 'xma', 30)
    print(f"Time Elapsed: {time.time() - start}.")
    print(df)
