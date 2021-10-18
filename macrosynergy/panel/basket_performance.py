import numpy as np
import pandas as pd
from typing import List, Union, Tuple
import random

from macrosynergy.management.simulate_quantamental_data import make_qdf
from macrosynergy.panel.historic_vol import expo_weights, expo_std, flat_std
from macrosynergy.management.shape_dfs import reduce_df, reduce_df_by_ticker

## In theory, there could be a situation where there is not carry return but there is a realised return.
## Therefore, the weight matrix used for the carries is not applicable because the weighting has assumed a greater number of active cross-sections for that timestamp.
## Output would be NaN if dot product includes a NaN.
def basket_perf(df: pd.DataFrame, contracts: List[str],
                ret: str = 'XR_NSA', cry: str = 'CRY_NSA', start: str = None,
                end: str = None, blacklist: dict = None, weight_meth: str = 'equal',
                lback_meth: str = 'xma', lback_periods: int = 21, half_life: int = 11,
                weights: List[float] = None, weight_xcat: str = None,
                max_weight: float = 1.0, basket_tik: str = 'GLB_ALL',
                return_weights: bool = False):

    if weights:
        assert len(set(df['cid'])) == len(weights)
    
    ## For instance, Excess Returns on Australian Foreign Exchange. The contract: cid + xcat (asset class).
    ticks_ret = [contract + '_' + ret for contract in contracts]
    if cry is not None:
       ticks_cry = [contract + '_' + cry for contract in contracts]
    else:
        ticks_cry = []
    tickers = ticks_ret + ticks_cry

    dfx = reduce_df_by_ticker(df, start=start, end=end, ticks=tickers, blacklist=black)

    dfx['tick'] = dfx['cid'] + '_' + dfx['xcat']
    
    ## Tickers that the DataFrame is analysed over. 
    dfx_ret = dfx[dfx['tick'].isin(ticks_ret)]

    ## Pivot on the cross-sections removing the xcat and ticker field.
    ## Potentially multiple asset classes defined over a single return type. The whole notion of the basket: multiple asset classes.
    ## Will each cross-section have a single asset class ?
    dfw_ret = dfx_ret.pivot(index='real_date', columns='cid', values='value')
    
    if cry is not None:
        dfx_cry = dfx[dfx['tick'].isin(ticks_cry)]
        dfw_cry = dfx_ret.pivot(index='real_date', columns='cid', values='value')

    ret_arr = dfw_ret.to_numpy()
    cry_arr = dfw_cry.to_numpy()
    nan_val = np.sum(np.isnan(ret_arr), axis = 1)
    act_cross = ret_arr.shape[1] - nan_val
    
    if weight_meth == 'equal':
        act_cross = act_cross[:, np.newaxis]

        ## Equally weighted across the cross-sectional returns.
        rational_ret = np.divide(ret_arr, act_cross)
        rational_cry = np.divide(cry_arr, act_cross)
        weighted_ret = np.nan_to_num(rational_ret)
        weighted_cry = np.nan_to_num(rational_cry)
        
        dfw_ret[weight_meth] = np.sum(weighted_ret, axis = 1)
        dfw_cry[weight_meth] = np.sum(weighted_cry, axis = 1)
        
    elif weight_meth == 'fixed':
        normalise = np.array(weights) / sum(weights)
        bool_arr = np.isnan(ret_arr).astype(dtype = np.uint8)
        bool_arr = ~bool_arr

        bool_arr = bool_arr.astype(dtype = np.uint8)
        w_matrix = np.multiply(bool_arr, normalise)

        normalise_m = np.sum(w_matrix, axis = 1)
        normalise_m = normalise_m[:, np.newaxis]
        w_matrix = np.divide(w_matrix, normalise_m)

        w_ret_matrix = np.multiply(ret_arr, w_matrix)
        w_ret_matrix = np.nan_to_num(w_ret_matrix, copy = False)
        dfw_ret[weight_meth] = np.sum(w_ret_matrix, axis = 1)

        w_cry_matrix = np.multiply(cry_arr, w_matrix)
        w_cry_matrix = np.nan_to_num(w_cry_matrix, copy = False)
        dfw_cry[weight_meth] = np.sum(w_cry_matrix, axis = 1)

    ## Assign the weights according to the evolving volatility of each time-series: the higher the volality, the lower the weight allocation.
    ## To achieve the relationship, take the inverse of the evolving volatility and subsequently normalise the values.
    elif weight_meth == 'invsd':

        if lback_meth == 'ma':
            dfwa = np.sqrt(252) * dfw_ret.rolling(window = lback_periods).agg(flat_std, True)
        else:
            weights = expo_weights(lback_periods, half_life)
            dfwa = np.sqrt(252) * dfw_ret.rolling(window =
                                              lback_periods).agg(expo_std,
                                                                 w = weights, remove_zeros = True)
        rolling_arr = dfwa.to_numpy()
        inv_arr = 1 / rolling_arr
        inv_arr = np.nan_to_num(inv_arr, copy = False)
        sum_arr = np.sum(inv_arr, axis = 1)
        sum_arr[sum_arr == 0.0] = np.nan
        inv_arr[inv_arr == 0.0] = np.nan

        sum_arr = sum_arr[:, np.newaxis]
        ## Appropriate weights applied to each index of the original return series DataFrame.
        rational = np.divide(inv_arr, sum_arr)

        w_returns = np.multiply(ret_arr, rational) ## Weighted returns for each timestamp.
        w_returns = np.nan_to_num(w_returns, copy = False, nan = 0.0)
        b_perf = np.sum(w_returns, axis = 1) ## Basket return for each timestamp.
        
        
    elif weight_meth == 'values' or weight_meth == 'inv_values':
        pass
        # Exogenous dataframe of weights.
        # Normalise according to NaN values.

        # Todo: do not implement yet
        # Note: this requires appropriate example data

    if max_weight < 1:
        pass
        max_margin = 0.01 # 1% greater than the threshold: set endogenously.
        # Todo: do not implement yet
        # Note: This requires an algorithm that sequentially redistributes weights until convergence


    # Todo: create standard dataframe with basket returns and carry
    # Todo: If return_weights is True also add contract weights to the data frame (category is contract + "_WGT")
    # Todo: return standard dataframe


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

    dfd_1 = basket_perf(dfd, contracts, ret='XR', cry='CRY',
                        weight_meth='invsd', lback_meth='ma', lback_periods=21,
                        weights=None, weight_xcat=None, max_weight=0.0,
                        return_weights=False)
