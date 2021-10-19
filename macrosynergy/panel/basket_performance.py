import numpy as np
import pandas as pd
from typing import List
import random

from macrosynergy.management.simulate_quantamental_data import make_qdf
from macrosynergy.panel.historic_vol import expo_weights, expo_std, flat_std
from macrosynergy.management.shape_dfs import reduce_df_by_ticker
from macrosynergy.panel.converge_row import ConvergeRow


def assemble_df(dfx: pd.DataFrame, list_: List[str]):
    """
    Will receive a standardised DataFrame and filter on the list received, and subsequently
    pivot on the cross-sections.

    :param <pd.DataFrame> dfx: Standardised DataFrame.
    :param <List[str]> list_: List of the required fields the DataFrame is being filtered on.

    :return <pd.DataFrame>: Returns the pivoted DataFrame.
    """
    dfx_ret_cry = dfx[dfx['tick'].isin(list_)]

    dfw_pivot = dfx_ret_cry.pivot(index='real_date',
                                  columns='cid', values='value')

    return dfw_pivot


def active_cross_sections(arr: np.ndarray):
    """
    Function will receive an Array and determine the number of active cross-sections per
    timestamp.

    :param <np.ndarray> arr: Multidimensional Array of return series or weights.

    :return <np.ndarray>: One-dimensional Array outlining the number of cross-sections.
    """

    nan_val = np.sum(np.isnan(arr), axis=1)
    act_cross = arr.shape[1] - nan_val

    return act_cross.astype(dtype=np.float32)


def boolean_array(arr: np.ndarray):
    """
    Function will receive an Array and return its Boolean counterpart reflecting which
    cross-sections have realised returns for each timestamp.

    :param <np.ndarray> arr: Multidimensional Array of returns.

    :return <np.ndarray>: Boolean Array of the same dimensions as the received Array.
    """

    bool_arr = np.isnan(arr)
    bool_arr = ~bool_arr
    bool_arr = bool_arr.astype(dtype=np.uint8)

    return bool_arr


def matrix_transpose(arr: np.ndarray, transpose: np.ndarray):
    """
    Function will receive two Arrays. The first Array will be used to determine the
    active cross-sections per timestamp. And the second Array is multiplied by the
    Boolean Array to transpose the value to specific indices and broadcast to the right
    dimensions.
    Will return the weighted matrix.

    :param <np.ndarray> arr: Multidimensional Array of returns.
    :param <np.ndarray> transpose: Counterpart holding desired information.

    :return <np.ndarray>: Will return the weight matrix of the same dimensions as the
                          first parameter.
    """

    bool_arr = boolean_array(arr)

    w_matrix = np.multiply(bool_arr, transpose)
    w_matrix[w_matrix == 0.0] = np.nan

    return w_matrix


def normalise_w(ret_arr: np.ndarray, w_matrix: np.ndarray):
    """
    Function will receive two Arrays. The first Array will be used to determine the active
    cross-sections per timestamp using the Return Series. And the second Array will be a
    predefined, exogenously determined weight-matrix. However, the weight matrix will be
    populated without regard to which cross-sections are active per timestamp: all indices
    will have a weight. Therefore, use the return matrix to determine the valid indices,
    and subsequently normalise such that the weights sum to one.

    :param <np.ndarray> ret_arr: Multidimensional Array of returns.
    :param <np.ndarray> w_matrix: Counterpart holding desired information.

    :return <np.ndarray>: Returns the weight matrix adjusting for NaNs and normalising.
    """

    w_matrix = matrix_transpose(ret_arr, w_matrix)

    normalise_m = np.nansum(w_matrix, axis=1)
    normalise_m = normalise_m[:, np.newaxis]
    w_matrix = np.divide(w_matrix, normalise_m)

    return w_matrix


def delete_rows(ret_arr: np.ndarray, w_matrix: np.ndarray, active_cross: np.ndarray):
    """
    Function designed to remove any rows which do not host any returns. For instance, on
    a universal holiday or after the application of the inverse standard deviation weighting
    method which requires the window to be populated before computing a weight. Therefore,
    all the preceding rows will hold NaN values.

    :param <np.ndarray> ret_arr: Array of returns. Multidimensional.
    :param <np.ndarray> w_matrix: Corresponding weight matrix. Multidimensional.
    :param <np.ndarray> active_cross: Array of the number of active cross-sections for
                                      each timestamp.

    :return <np.ndarray>: Will return the three modified Arrays.
    """
    
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
            ret_arr = np.delete(ret_arr, tuple(nan_rows), axis=0)
            w_matrix = np.delete(w_matrix, tuple(nan_rows), axis=0)
            active_cross = np.delete(active_cross, tuple(nan_rows))

    return ret_arr, w_matrix, active_cross


def max_weight_func(w_matrix: np.ndarray, active_cross: np.ndarray, max_weight: np.ndarray):
    """
    Function designed to determine if all weights computed are within the maximum weight
    allowed per cross-section. If the maximum weight is less than the expected weight,
    replace the computed weight with the expected weight. For instance,
    [np.nan, 0.63, np.nan, np.nan, 0.27] becomes [np.nan, 0.5, np.nan, np.nan, 0.5].
    Otherwise, call the ConvergeRow Class to ensure all weights "converge" to a value
    within the upper-bound. Allow for a margin of error set to 0.001.

    :param <np.ndarray> w_matrix: Corresponding weight matrix. Multidimensional.
    :param <np.ndarray> active_cross: Array of the number of active cross-sections for
                                      each timestamp.
    :param <float> max_weight: Upper-bound on the weight allowed for each cross-section.

    :return <np.ndarray>: Will return the modified weight Array.
    """
    uniform = 1 / active_cross
    fixed_indices = np.where(uniform > max_weight)[0]

    rows = w_matrix.shape[0]
    bool_arr = np.zeros(rows, dtype=bool)
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

    return w_matrix


def equal_weight(ret_arr: np.ndarray, act_cross: np.ndarray):
    """
    Function will receive two Arrays: one consisting of the multi-dimensional return
    series and the other a one-dimensional Array outlining the number of active cross-
    sections per timestamp: use the second Array to delimit the weight distribution for
    each timestamp.

    :param <np.ndarray> ret_arr: Return series matrix. Multidimensional.
    :param <np.ndarray> act_cross: Array of the number of active cross-sections for
                                      each timestamp.

    :return <np.ndarray>: Will return the generated weight Array, and the active cross-
                          sections.
    """
    act_cross[act_cross == 0.0] = np.nan
    uniform = 1 / act_cross
    uniform = uniform[:, np.newaxis]

    w_matrix = matrix_transpose(ret_arr, uniform)
    act_cross = np.nan_to_num(act_cross)

    return w_matrix, act_cross


def fixed_weight(ret_arr: np.ndarray, weights: np.ndarray):
    """
    The fixed weight method will receive a matrix of values, same dimensions as the
    DataFrame, and use the values for weights. For instance, GDP figures.
    The values will be normalised and account for NaNs to obtain the formal weight
    matrix.

    :param <np.ndarray> ret_arr: Return series matrix. Multidimensional.
    :param <np.ndarray> weights: Multidimensional Array of values.

    :return <np.ndarray>: Will return the generated weight Array
    """
    normalise = np.divide(weights, np.sum(weights, axis=1))
    w_matrix = normalise_w(ret_arr, normalise)

    return w_matrix


def inverse_weight(dfw_ret: pd.DataFrame, ret_arr: np.ndarray, lback_meth: str = 'xma',
                   lback_periods: int = 21, remove_zeros: bool = True):
    """
    The weights will be computed by taking the inverse of the rolling standard deviation
    of each return series. The rolling standard deviation will be calculated either
    using the standard Moving Average or the Exponential Moving Average.
    Both Moving Average's will require a window to be populated with returns before a
    weight can be computed, and subsequently the preceding timestamps will be set to NaN
    until the window has been filled. Therefore, modify the original Return Matrix and
    the active cross-section Array to reflect the additional NaN values.

    :param <pd.DataFrame> dfw_ret: DataFrame pivot on the cross-sections.
    :param <np.ndarray> ret_arr: Return series matrix. Multidimensional.
    :param <str> lback_meth: Lookback method for "invsd" weighting method.
                             Default is "xma".
    :param <int> lback_periods: Lookback periods. Default is 21.  Refers to half-time for
                                "xma" and full lookback period for "ma".
    :param <Bool> remove_zeros: Removes the zeros. Default is set to True.


    :return <np.ndarray>: Will return the generated weight Array, the updated Return
                          Matrix,and the active cross-sections.
    """
    
    if lback_meth == 'ma':
        dfwa = dfw_ret.rolling(window=lback_periods).agg(flat_std, remove_zeros)
        dfwa *= np.sqrt(252)
        
    else:
        half_life = lback_periods
        lback_periods *= 2
        weights = expo_weights(lback_periods, half_life)
        dfwa = dfw_ret.rolling(window=lback_periods).agg(expo_std, w=weights,
                                                         remove_zeros=remove_zeros)
        dfwa *= np.sqrt(252)

    roll_arr = dfwa.to_numpy()
    ret_arr = matrix_transpose(roll_arr, ret_arr)

    act_cross = active_cross_sections(roll_arr)

    inv_arr = 1 / roll_arr
    inv_arr = np.nan_to_num(inv_arr, copy=False)
    sum_arr = np.sum(inv_arr, axis=1)
    sum_arr[sum_arr == 0.0] = np.nan
    inv_arr[inv_arr == 0.0] = np.nan

    sum_arr = sum_arr[:, np.newaxis]
    w_matrix = np.divide(inv_arr, sum_arr)

    return w_matrix, ret_arr, act_cross


def values_weight(ret_arr: np.ndarray, weights: np.ndarray):
    """
    The values weight method will receive a matrix of values produced by another category
    and the values held in the matrix will be used as the weights. The weight matrix
    should contain floating point values.

    :param <np.ndarray> ret_arr: Return series matrix. Multidimensional.
    :param <np.ndarray> weights: Multidimensional Array of exogenously computed weights.

    :return <np.ndarray>: Will return the generated weight Array
    """
    w_matrix = normalise_w(ret_arr, weights)

    return w_matrix


def basket_performance(df: pd.DataFrame, contracts: List[str], ret: str = 'XR_NSA',
                       cry: str = 'CRY_NSA', start: str = None, end: str = None,
                       blacklist: dict = None, weight_meth: str = 'equal', lback_meth:
                       str = 'xma', lback_periods: int = 21, remove_zeros: bool = True,
                       weights: List[float] = None, weight_xcat: str = None,
                       max_weight: float = 1.0, basket_tik: str = 'GLB_ALL',
                       return_weights: bool = False):

    """
    Computes approximate return and carry series for a basket of underlying contracts.

    :param <pd.Dataframe> df: Standardized DataFrame with the following columns:
                              'cid', 'xcats', 'real_date' and 'value.
    :param <List[str]> contracts: Base Tickers (combinations of cross sections and base
                                  categories) denoting contracts that go into the basket.
    :param <str> ret: Return Category postfix; default is "XR_NSA".
    :param <str> cry: Return Category postfix; default is "CRY_NSA".
                      Choose None if carry not available.
    :param <str> start: String representing earliest date. Default is None.
    :param <str> end: String representing the latest date. Default is None.
    :param <dict> blacklist: Cross-sections with date ranges that should be excluded from
                             the DataFrame. If one cross-section has several blacklist
                             periods, append numbers to the cross-section code.
    :param <str> weight_meth: Method for weighting constituent returns and carry.
                              Default is "equal": all constituents have the same weight.
                              Alternatives are:
                              "fixed": proportionate to vector values supplied separately.
                              "invsd": inverse of past return standard deviations.
                              "values": proportionate to a panel of values of another
                                        exogenous category.
                              "inv_values": inversely proportionate to a panel of
                                            exogenous values supplied separately).
    :param <str> lback_meth: Lookback method for "invsd" weighting method.
                             Default is "xma".
    :param <int> lback_periods: Lookback periods. Default is 21.  Refers to half-time for
                                "xma" and full lookback period for "ma".
    :param <Bool> remove_zeros: Removes the zeros. Default is set to True.
    :param <List[float]> weights: List of weights corresponding to the tickers.
                                  Only relevant if weight_meth = "fixed", "values" or
                                  "inv_values" is chosen.
    :param <str> weight_xcat: Extended category name of values used for "values" and
                              "inv_values" methods.
    :param <float> max_weight: Maximum weight permitted for a single cross section.
                               Default is 1: a restriction is not applied.
    :param <str> basket_tik: Name of basket base ticker for which return and (possibly)
                             carry are calculated. Default is "GLB_ALL".
    :param <bool> return_weights: Add cross-section weights to output dataframe
                                  and uses ticks with 'WGT' postfix.
                                  Default is False.


    :return <pd.Dataframe>: Standardized DataFrame with the basket performance data
                            in standard form:
                            'cid', 'xcat', 'real_date' and 'value'.
    """

    if weights:
        assert len(set(df['cid'])) == len(weights)

    ticks_ret = [c + '_' + ret for c in contracts]
    ticks_cry = []
    cry_flag = False
    if cry is not None:
        ticks_cry = [c + '_' + cry for c in contracts]
        cry_flag = True

    tickers = ticks_ret + ticks_cry

    dfx = reduce_df_by_ticker(df, start=start, end=end,
                              ticks=tickers, blacklist=black)

    dfx['tick'] = dfx['cid'] + '_' + dfx['xcat']
    dfw_ret = assemble_df(dfx, ticks_ret)

    if cry_flag:
        dfw_cry = assemble_df(dfx, ticks_cry)

    ret_arr = dfw_ret.to_numpy()
    cry_arr = dfw_cry.to_numpy()
    act_cross = active_cross_sections(ret_arr)

    if weight_meth == 'equal':
        w_matrix, act_cross = equal_weight(ret_arr, act_cross)

    elif weight_meth == 'fixed':
        w_matrix = fixed_weight(ret_arr, weights)

    elif weight_meth == 'invsd':
        w_matrix, ret_arr, act_cross = inverse_weight(dfw_ret, ret_arr, lback_meth,
                                       lback_periods, remove_zeros)

    elif weight_meth == 'values' or weight_meth == 'inv_values':
        w_matrix = values_weight(ret_arr, weights)

    else:
        raise NotImplementedError(f"Weight method unknown {weight_meth}")

    ret_arr, w_matrix, act_cross = delete_rows(ret_arr, w_matrix, act_cross)
    if max_weight > 0.0:
        w_matrix = max_weight_func(w_matrix, act_cross, max_weight)

    weighted_ret = np.multiply(ret_arr, w_matrix)
    b_performance = np.nansum(weighted_ret, axis = 1)

    data = np.column_stack((ret_arr, b_performance))
    columns = list(dfw_ret.columns)
    columns.extend(['B_Performance_' + weight_meth])
    if return_weights:

        col_w = [cross + '_weight' for cross in dfw_ret.columns]
        columns.extend(col_w)
        data = np.column_stack((data, w_matrix))

    dfw_ret = pd.DataFrame(data=data, columns=columns)

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

    dfd_2 = basket_performance(dfd, contracts, ret='XR', cry='CRY',
                               weight_meth='invsd', lback_meth='xma', lback_periods=21,
                               weights=None, weight_xcat=None, max_weight=0.3,
                               return_weights=True)
    print(dfd_2)

    dfd_3 = basket_performance(dfd, contracts, ret='XR', cry='CRY', weight_meth='equal',
                               weights=None, weight_xcat=None, max_weight=0.3,
                               return_weights=False)
    print(dfd_3)
