
import numpy as np
import pandas as pd
from Test_File import make_qdf_, simulate_ar
from collections import defaultdict, deque
import time
import matplotlib.pyplot as plt
    

def add_self_arg(func):
    
    def wrapped(*args, **kwargs):
        return func(wrapped, *args, **kwargs)
        
    return wrapped

def expo_weights(tx: int, hft: int = 11):

    decf = 2 ** (-1 / hft)
    weights = (1 - decf) * np.array([decf ** (tx - ii - 1) for ii in range(tx)])
    
    weights = weights / weights.sum(axis = 0)
    return weights
    

## Taking the average of the initial fixed subset of the number series and "shifting forward" until inclusive of the terminal date.
## Superfluous function. Computing Moving Average of mean returns.
def moving_average(lists, n):

    ## Exploiting scope.
    def compute(arr):
        ## Running Sum.
        ret = np.cumsum(arr[:, 1], dtype = float)
        ret[n:] = ret[n:] - ret[:-n]
        ret = ret[n - 1:] / n
        
        arr = np.append(np.zeros(n - 1) + np.nan, ret)
        return arr

    return list(map(compute, lists))


def rolling_std(lists, n):

    def compute(arr):
        
        arr = arr[:, 1]
        rolling_std = np.zeros(len(arr), dtype = float)
        window = deque(arr[:(n - 1)], n)
        for i, _ in enumerate(arr[n:]):
            window.append(_)
            std = np.std(window)
            rolling_std[(n + i)] = std * np.sqrt(256)
            
        return rolling_std
    
    return list(map(compute, lists))     

def decay(window):

    N, tau = 1, 0.4
    if window <= 30:
        tau = 2.0
    
    t = np.linspace(0, 1, window)
    weights = N * np.exp(-t / tau)
    plt.plot(t, weights) 
    weights /= weights.sum()
    plt.show()

    return weights

## Superfluous function. Computing Exponential Moving Average of mean returns.
def exp_moving(lists, weights, window):

    def compute_(array):
        
        arr = np.convolve(array[:, 1], weights)[:len(array[:, 1])]
        n = (window - 1)
        arr = np.append(np.zeros(n) + np.nan, arr[(n):])
    
        return arr
    
    return list(map(compute_, lists))


def exp_std(lists, weights, n):

    def compute_(array):
        nonlocal weights

        ## weights = weights[::-1]
        arr = array[:, 1]
        
        rolling_std = np.zeros(len(arr), dtype = float)
        window = deque(arr[:(n - 1)], n)
        
        for i, _ in enumerate(arr[n:]):
            window.append(_)
            
            numerator = np.sum(weights * (window  - np.mean(window)) ** 2 )
            std = np.sqrt(numerator)
            rolling_std[(n + i)] = std * np.sqrt(256)

        return rolling_std
    
    return list(map(compute_, lists))

def stand_dev(arr):
    return (np.std(arr[:, 1]) * np.sqrt(255))

def merge(arr):

    return arr[:, 0]

def func(str_, int_):
    return [str_] * int_

def mult(xcats, list_):

    return list(map(func, xcats, list_))
    
def rolling_df(dfd, cids, xcats, window, rolling_window):

    cid_xcat = {}
    cid_roll = {}
    cid_dates = {}
    xcat = defaultdict(list)

    w = expo_weights(window, 11)
    ## w = w + 1
    
    for cid in cids:
        df_temp = dfd[dfd['cid'] == cid]
        
        def indicators(cat):
            
            df_ = df_temp[df_temp['xcat'] == cat]
            data = df_[['real_date', 'value']].to_numpy()
            xcat[cid].append(len(data[:, 0]))
            
            return data
        
        cid_xcat[cid] = list(map(indicators, xcats))

    for cid in cids:
        if rolling_window == 'MA':
            cid_roll[cid] = rolling_std(cid_xcat[cid], window)
        else:
            cid_roll[cid] = exp_std(cid_xcat[cid], w, window)

    qdf_cols = ['cid', 'xcat', 'real_date', rolling_window]
    df_lists = []
    for cid in cids:
        
        df_out = pd.DataFrame(columns = qdf_cols)
        df_out['xcat'] = np.concatenate(mult(xcats, xcat[cid]))
        list_ = tuple(map(merge, cid_xcat[cid]))
        df_out['real_date'] = np.concatenate(list_)
        df_out[rolling_window] = np.concatenate(cid_roll[cid]) 
        df_out['cid'] = cid
        df_lists.append(df_out)

    final_df = pd.concat(df_lists, ignore_index = True)
        
    return final_df


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

    dfd, fields_cats, fields_cids, df_year, df_end, df_missing, cids_cats = make_qdf_(df_cids, df_xcats, back_ar = 0.75)

    y = dfd['real_date'].to_numpy()
    dfd['real_date'] = y.astype('datetime64', copy = False)

    start = time.time()
    final_df = rolling_df(dfd, fields_cids, fields_cats, 40, 'MA')
    print(final_df)
    print(f"Time Elapsed: {time.time() - start}.")

    start = time.time()
    final_df = rolling_df(dfd, fields_cids, fields_cats, 40, 'ExpMA')
    print(final_df)
    print(f"Time Elapsed: {time.time() - start}.")
