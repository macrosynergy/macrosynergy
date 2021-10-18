import numpy as np
import pandas as pd
from typing import List, Union, Tuple
import random

from macrosynergy.management.simulate_quantamental_data import make_qdf
from macrosynergy.panel.historic_vol import expo_weights, expo_std, flat_std
from macrosynergy.management.shape_dfs import reduce_df, reduce_df_by_ticker
from macrosynergy.management.converge_row import ConvergeRow


def delete_rows(ret_arr, w_matrix, active_cross):
    
    nan_rows = np.where(active_cross == 0)[0]
    nan_size = nan_rows.size
    if not nan_size == 0:
        
        start = nan_rows[0]
        end = nan_rows[-1]
        iterator = np.array(range(start, end + 1))

        bool_size = np.all(iterator.size == nan_size)
        if bool_size and np.all(iterator == nan_rows):
            ret_arr = ret_arr[(end + 1):, :]
            w_matrix = w_matrix[(end + 1):, :]
            active_cross = active_cross[(end + 1):]
        else:
            ret_arr = np.delete(ret_arr, tuple(nan_rows), axis = 0)
            w_matrix = np.delete(w_matrix, tuple(nan_rows), axis = 0)
            active_cross = np.delete(active_cross, tuple(nan_rows))

    return ret_arr, w_matrix, active_cross


def max_weight_func(ret_arr, w_matrix, active_cross, max_weight):

    uniform = 1 / active_cross
    fixed_indices = np.where(uniform > max_weight)[0]

    rows = w_matrix.shape[0]
    bool_arr = np.zeros(rows, dtype = bool)
    bool_arr[fixed_indices] = True

    for i, row in enumerate(w_matrix):
        if bool_arr[i]:
            row = np.ceil(row)
            row = row * uniform[i]
            w_matrix[i, :] = row
        else:
            inst = ConvergeRow(row, max_weight, active_cross[i], row.size)
            inst.distribute()
            w_matrix[i, :] = inst.row

    return ret_arr, w_matrix

def boolean_array(arr):
    
    bool_arr = np.isnan(arr)
    bool_arr = ~bool_arr
    bool_arr = bool_arr.astype(dtype = np.uint8)
    return bool_arr

def normalise_w(ret_arr, w_matrix):
    bool_arr = boolean_array(ret_arr)

    w_matrix = np.multiply(bool_arr, w_matrix)

    normalise_m = np.sum(w_matrix, axis = 1)
    normalise_m = normalise_m[:, np.newaxis]
    w_matrix = np.divide(w_matrix, normalise_m)

    return w_matrix

def active_cross_sections(arr):
    nan_val = np.sum(np.isnan(arr), axis = 1)
    act_cross = arr.shape[1] - nan_val
    
    return act_cross.astype(dtype = np.float32)

def matrix_transpose(arr, transpose):

    bool_arr = boolean_array(arr)
    
    w_matrix = np.multiply(bool_arr, transpose)
    w_matrix[w_matrix == 0.0] = np.nan    

    return w_matrix

def basket_performance(df: pd.DataFrame, contracts: List[str], ret: str = 'XR_NSA',
                       cry: str = 'CRY_NSA', start: str = None, end: str = None,
                       blacklist: dict = None, weight_meth: str = 'equal', lback_meth: str = 'xma',
                       lback_periods: int = 21, weights: List[float] = None, weight_xcat: str = None,
                       max_weight: float = 1.0, basket_tik: str = 'GLB_ALL', return_weights: bool = False):

    if weights:
        assert len(set(df['cid'])) == len(weights)
    
    ticks_ret = [c + '_' + ret for c in contracts]
    if cry is not None:
       ticks_cry = [c + '_' + cry for c in contracts]
    else: ticks_cry = []
    tickers = ticks_ret + ticks_cry

    dfx = reduce_df_by_ticker(df, start=start, end=end,
                              ticks=tickers, blacklist=black)

    dfx['tick'] = dfx['cid'] + '_' + dfx['xcat']
     
    dfx_ret = dfx[dfx['tick'].isin(ticks_ret)]

    dfw_ret = dfx_ret.pivot(index='real_date', columns='cid', values='value')
    
    if cry is not None:
        dfx_cry = dfx[dfx['tick'].isin(ticks_cry)]
        dfw_cry = dfx_cry.pivot(index='real_date', columns='cid', values='value')
        cry_flag = True

    ret_arr = dfw_ret.to_numpy()
    cry_arr = dfw_cry.to_numpy()

    act_cross = active_cross_sections(ret_arr)
    
    if weight_meth == 'equal':
        act_cross[act_cross == 0.0] = np.nan
        uniform = 1 / act_cross
        uniform = uniform[:, np.newaxis]

        w_matrix = matrix_transpose(ret_arr, uniform)
        act_cross = np.nan_to_num(act_cross)
                
    elif weight_meth == 'fixed':
        normalise = np.array(weights) / sum(weights)
        w_matrix = normalise_w(ret_arr, normalise)

    elif weight_meth == 'invsd':
        if lback_meth == 'ma':
            dfwa = dfw_ret.rolling(window = lback_periods).agg(flat_std, True)
            rolling_arr = dfwa.to_numpy()
            ret_arr = matrix_transpose(rolling_arr, ret_arr)
            
            act_cross = active_cross_sections(rolling_arr)

            inv_arr = 1 / rolling_arr
            inv_arr = np.nan_to_num(inv_arr, copy = False)
            sum_arr = np.sum(inv_arr, axis = 1)
            sum_arr[sum_arr == 0.0] = np.nan
            inv_arr[inv_arr == 0.0] = np.nan

            sum_arr = sum_arr[:, np.newaxis]
            w_matrix = np.divide(inv_arr, sum_arr)
    
    elif weight_meth == 'values' or weight_meth == 'inv_values':
        normalise = np.array(weights) / sum(weights)
        w_matrix = normalise_w(ret_arr, normalise)

    ret_arr, w_matrix, act_cross = delete_rows(ret_arr, w_matrix, act_cross)
    if max_weight > 0.0:
        ret_arr, w_matrix = max_weight_func(ret_arr, w_matrix, act_cross, max_weight)

    weighted_ret = np.multiply(ret_arr, w_matrix)
    b_performance = np.nansum(weighted_ret, axis = 1)
    
    data = np.column_stack((ret_arr, b_performance))
    columns = list(dfw_ret.columns)
    columns.extend(['b_performance_' + weight_meth])
    if return_weights:

        col_w = [cross + '_weight' for cross in dfw_ret.columns]
        columns.extend(col_w)
        data = np.column_stack((data, w_matrix))

    dfw_ret = pd.DataFrame(data = data, columns = columns)

    return dfw_ret


if __name__ == "__main__":

    cids = ['AUD', 'GBP', 'NZD', 'USD', 'TRY']
    xcats = ['FX_XR', 'FX_CRY', 'EQ_XR', 'EQ_CRY']

    df_cids = pd.DataFrame(index=cids, columns=['earliest', 'latest', 'mean_add', 'sd_mult'])

    df_cids.loc['AUD'] = ['2010-12-01', '2020-12-31', 0, 1]
    df_cids.loc['GBP'] = ['2011-01-01', '2020-11-30', 0, 2]
    df_cids.loc['NZD'] = ['2012-01-01', '2020-11-30', 0, 3]
    df_cids.loc['USD'] = ['2013-01-01', '2020-09-30', 0, 4]
    df_cids.loc['TRY'] = ['2002-01-01', '2020-09-30', 0, 5]

    df_xcats = pd.DataFrame(index=xcats, columns=['earliest', 'latest', 'mean_add', 'sd_mult', 'ar_coef', 'back_coef'])
    df_xcats.loc['FX_XR'] = ['2010-01-01', '2020-12-31', 0, 1, 0, 0.2]
    df_xcats.loc['FX_CRY'] = ['2011-01-01', '2020-10-30', 1, 1, 0.9, 0.5]
    df_xcats.loc['EQ_XR'] = ['2011-01-01', '2020-10-30', 0.5, 2, 0, 0.2]
    df_xcats.loc['EQ_CRY'] = ['2013-01-01', '2020-10-30', 1, 1, 0.9, 0.5]

    random.seed(2)
    dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)
    black = {'AUD': ['2000-01-01', '2003-12-31'], 'GBP': ['2018-01-01', '2100-01-01']}

    contracts = ['AUD_FX', 'NZD_FX', 'GBP_EQ', 'USD_EQ']

    dfd_1 = basket_performance(dfd, contracts, ret='XR', cry='CRY',
                               weight_meth='invsd', lback_meth='ma', lback_periods=21,
                               weights=None, weight_xcat=None, max_weight=0.3,
                               return_weights=True)

