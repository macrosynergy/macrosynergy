
import time
import numpy as np
import pandas as pd
from collections import defaultdict, deque
from typing import List, Union, Tuple
from random import choice
from macrosynergy.management.simulate_quantamental_data import make_qdf
from macrosynergy.management.shape_dfs import reduce_df


def filter_zeros(df: pd.DataFrame):
    """
   Filters out rows of dataframe where the 'value' column contains exact zeroes

    :param <pd.DataFrame> df: Dataframe with a column called 'value' that contains numerical values.

    :return Dataframe that removes all timestand/rows for which value is zero.
    """
    
    values = df['value'].to_numpy()
    bool_ = values == 0.0 
    if any(bool_):
        print("Entered.")
        bool_ = ~bool_
        df = df[bool_]

    return df


def slide(arr, w, s=1):
    '''
    Receives one-dimensional array and returns a sliding window.

    :param <ndarray float> arr: Entire return series.
    :param <int> w: window.
    :param <int> s: controls the frequency of the returned windows.

    :return multidimensional array hosting each window
    '''

    return np.lib.stride_tricks.as_strided(arr,
                                           shape = ((len(arr) - w) + 1, w),
                                           strides = arr.strides * 2)[::s]


def expo_weights(tx: int, hft: int = 21):
    '''
    Produces numpy array of weights for exponential moving averaging

    :param <int> tx: Number of days the return series is defined over.
    :param <hft> hft: Half_Life.

    :return numpry array of weights of exponential window.
    '''
    decf = 2 ** (-1 / hft)
    weights = (1 - decf) * np.array([decf ** (tx - ii - 1) for ii in range(tx)])

    return weights


def weight_series(weights, n_days, cutoff):
    '''
    Returns truncated array of weights according to available series and cutoff.

    :param <List int> weights: Generated weights, for the entire series, from the above subroutine.
    :param <int> n_days: Number of days currently "realised".
    :param <int> cutoff: Only consider weights larger than the cutoff. Applicable days, set the weight to zero.

    Isolate the number of weights applicable to the current day count, "n_days", and verify if any are below the cutoff. If so,
    set to zero. Return the weights having adjusted for the cutoff in a Numpy Array.

    :return array of truncated weight series.
    '''
    w_series = np.zeros(n_days)
    
    weights = weights[-n_days:]
    if weights[0] < cutoff:
        weights = weights[weights >= cutoff]

    active_w = len(weights)
    
    weights = weights / weights.sum(axis = 0)
    w_series[-active_w:] = weights
    
    return w_series


def rolling_std_stride(lists, n):  # maybe rolling_std
    '''
    Returns list of arrays of simple moving averages of various xcats

    :param <Python List> lists: List consisting of four two-dimensional Arrays (dates & return series).
    :param <n> int n: The second parameter, "n" represents the window size for the Moving Average.
    
    Utilise Numpy's Slide Tricks to create the moving windows without requiring additional memory. The memory head
    will "slide" across as the window evolves through the realised series, and return each window of data in a Matrix form.
    Therefore, compute the Moving Average by using Numpy's axis feature, and subsequently return a Moving Average STD for the entire
    return series.
    The first section of dates, delimited by the size of the window, will be marked by zero values.

    :return List of array of MAs held in a Numpy Array.
    '''
    def compute(arr):
        
        arr = arr[:, 1]
        arr = slide(arr, n)
        arr = arr.astype(dtype = float)
        std_arr = np.std(arr, axis = 1)
        std_arr = std_arr * np.sqrt(256)
        std_arr = np.append(np.zeros(n - 1) + np.nan, std_arr)
        return std_arr
        
    return list(map(compute, lists))

def rolling_ema(lists, half_life, cutoff):
    '''
    Calculates exponential moving averages of return series

    :param <Python List> lists: List consisting of four two-dimensional Arrays (dates & return series).
    :param <int> half_life: Halflife - default 21 days accounts for 50% of the weighting.
    :param <float> cutoff: Cutoff - default 0.01 - weights below 0.01 will be set to zero to be discounted in the exponential weighting.

    :return a List consisting of len(xcat) number of EMAs in numpry arrays.
    '''

    def compute(arr):
        '''
        :param <Numpy ndarray> arr: Description above.
        Apply the weighted average from (half_life * 2) onwards to enforce the 50% weighting logic. In this example, the EMA will only become active
        from the 42 day onwards and represents the local variable "n".
        Iterate through the return series, from the 42 day onwards, load each return into a dynamically resizing List, and compute the weighted average standard
        deviation for that specific timestamp.
        Will return an Array containing the EMA defined over each timestamp received excluding the first (half_time * two) number of days.
        The aforementioned first set of days, without an EMA, will be defined with a zero value.
        '''

        arr = arr[:, 1]
        time = len(arr)
        rolling_std = np.zeros(time, dtype = float)

        n = half_life * 2
        window = list(arr[:(n - 1)])
        w_series = expo_weights(time, half_life)
        
        l_window = len(window)
        for i, _ in enumerate(arr[n:]):
            window.append(_)
            l_window += 1
            w_series = weight_series(w_series, l_window, cutoff)
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


def smoothing_func(dfd: pd.DataFrame, cids: List[str] = None, xcats: List[str] = None,
                   xcat: str = None, lback_period: int = None, half_life: int = 21,
                   cutoff: int = 0.01, lback_meth: str = 'MA'):
    '''
    Applies rolling standard deviations to a data frame.

    *Parameters described below. Identical Signature.
    
    The function will populate a dictionary, cid_xcat, where the keys are the countries the data is defined over, cids,
    and each key will host a List of the return series for each economic field, xcats, with the associated dates: two-dimensional Array
    where the first column is the timestamps and the second column the realised return.
    Once the dictionary has been populated, iterate through each (key / value) pair and, depending on the "lback_meth", compute the MA or EMA for each
    economic field using the above functions.
    Once each MA or EMA has been returned, held inside a List, iterate through the dictionary, cid_stride, and reconstruct the
    dataframe received with the additional column consisting of the "lback_meth" for all countries over each economic field.
    The returned dataframe will be the same dimensions as the recieved input but the Moving Average calculation will displace the realised return series.
    '''

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


def historic_vol(df: pd.DataFrame, cids: List[str] = None, xcat: str = None,
                 start: str = '2000-01-01', end: str = None, blacklist: dict = None,
                 lback_meth: str = 'xma', lback_periods: int = 21, half_life: int = 21,
                 cutoff: float = 0.01, remove_zeros: bool = True,
                 postfix: str = 'ASD'):

    """
    Estimates historic annualized standard deviations of asset returns.

    :param <pd.Dataframe> df: standardized data frame with the following necessary columns:
    'cid', 'xcats', 'real_date' and 'value.
    :param <str> xcat:  extended category denoting the return series for which volatility should be calculated.
    :param <List[str]> cids: cross sections for which volatility is calculated;
        default is all available for the category.
    :param <str> start: earliest date in ISO format. Default is None and earliest date in df is used.
    :param <str> end: latest date in ISO format. Default is None and latest date in df is used.
    :param <dict> blacklist: cross sections with date ranges that should be excluded from the data frame.
        If one cross section has several blacklist periods append numbers to the cross section code.
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

    # list the asserts

    df, xcat, cids = reduce_df(df, [xcat], cids, start, end, blacklist, out_all=True)

    # pivot the dataframe to time x countries for values
    # apply two rolling aggregations: [1] annualized standard deviation and [2] exponentinal standard deviation
    # both aggregation functions are custom and exclude the zeroes at this stage.
    # melt back to standardized dataframe and return


    cid_xcat = {}
    cid_stride = {}

    for cid in cids:
        df_temp = df[df['cid'] == cid]
        data = df_temp[['real_date', 'value']].to_numpy()
        cid_xcat[cid] = list(data)

    for cid in cids:
        if lback_meth == 'MA':
            cid_stride[cid] = rolling_std_stride(cid_xcat[cid], lback_period)
        else:
            cid_stride[cid] = rolling_ema(cid_xcat[cid], half_life, cutoff)

    qdf_cols = ['cid', 'xcat', 'real_date', lback_meth]
    df_lists = []
    for cid in cids:
        df_out = pd.DataFrame(columns=qdf_cols)
        df_out['xcat'] = np.concatenate(mult(xcats, xcat[cid]))
        list_ = tuple(map(merge, cid_xcat[cid]))
        df_out['real_date'] = np.concatenate(list_)
        df_out[lback_meth] = np.concatenate(cid_stride[cid])
        df_out['cid'] = cid
        df_lists.append(df_out)

    final_df = pd.concat(df_lists, ignore_index=True)


    rolling_df = smoothing_func(dfd, cids, xcats, xcat, lback_period, half_life,
                                cutoff, lback_meth)

    xcat_filter = (rolling_df['xcat'] == xcat).to_numpy()
    df_xcat = rolling_df[xcat_filter]

    return df_xcat.reset_index(drop=True)
    

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

    dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)

    zero_filter = (dfd['cid'] == 'AUD') & (dfd['real_date'] == '2010-01-13')
    dfd.loc[zero_filter, 'value'] = 0  # simulate exact zero

    df = historic_vol(dfd, cids, xcat='XR', lback_meth='ma', lback_periods=42, cutoff=0.01)
    print(df)
