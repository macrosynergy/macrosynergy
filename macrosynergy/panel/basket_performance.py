import numpy as np
import pandas as pd
import random
from typing import List, Union
from macrosynergy.panel.historic_vol import expo_weights, expo_std, flat_std
from macrosynergy.management.shape_dfs import reduce_df_by_ticker
from macrosynergy.panel.converge_row import ConvergeRow
from macrosynergy.management.simulate_quantamental_data import make_qdf


def check_weights(weight: pd.DataFrame):
    check = weight.sum(axis=1)
    c = ~((abs(check - 1) < 1e-6) | (abs(check) < 1e-6))
    assert not any(c), f"weights must sum to one (or zero), not: {check[c]}"

def equal_weight(df_ret: pd.DataFrame) -> pd.DataFrame:
    """Equal weight function

    Function will receive the pivoted return DataFrame and determine the number of active
    cross-sections per timestamp, and subsequently distribute the weight evenly across
    the aforementioned cross-sections.

    :param <pd.DataFrame> df_ret: data-frame with returns.

    :return: Will return the generated weight <pd.DataFrame> of weights.
    """

    act_cross = (~df_ret.isnull())
    uniform = (1 / act_cross.sum(axis=1)).values
    uniform = uniform[:, np.newaxis]
    broadcast = np.repeat(uniform, df_ret.shape[1], axis=1)

    weight = act_cross.multiply(broadcast)
    check_weights(weight=weight)

    return weight


def fixed_weight(df_ret: pd.DataFrame, weights: List[float]):
    """
    The fixed weight method will receive a List of values, same dimensions as the
    DataFrame, and use the values for weights. For instance, GDP figures.
    The values will be normalised and account for NaNs to obtain the formal weight
    matrix.

    :param <pd.DataFrame> df_ret: Return series matrix. Multidimensional.
    :param <List[float]> weights: List of floats determining weight allocation.
                                  Example GDP.

    :return <pd.DataFrame>: Will return the generated weight Array
    """

    act_cross = (~df_ret.isnull())

    weights = np.array(weights, dtype=np.float32)
    rows = act_cross.shape[0]
    broadcast = np.tile(weights, (rows, 1))

    weight = act_cross.multiply(broadcast)
    weight_arr = weight.to_numpy()
    weight[weight.columns] = weight_arr / np.sum(weight_arr, axis=1)[:, np.newaxis]

    check_weights(weight)

    return weight


def inverse_weight(dfw_ret: pd.DataFrame, lback_meth: str = "xma",
                   lback_periods: int = 21, remove_zeros: bool = True):
    """
    The weights will be computed by taking the inverse of the rolling standard deviation
    of each return series. The rolling standard deviation will be calculated either
    using the standard Moving Average or the Exponential Moving Average.
    Both Moving Average's will require a window to be populated with returns before a
    weight can be computed, and subsequently the preceding timestamps will be set to NaN
    until the window has been filled. Therefore, modify the original Return Matrix to
    reflect the additional NaN values.

    :param <pd.DataFrame> dfw_ret: DataFrame pivot on the cross-sections.
    :param <str> lback_meth: Lookback method for "invsd" weighting method.
                             Default is "xma".
    :param <int> lback_periods: Lookback periods. Default is 21.  Refers to half-time for
                                "xma" and full lookback period for "ma".
    :param <Bool> remove_zeros: Removes the zeros. Default is set to True.

    :return <pd.DataFrame>: Will return the generated weight DataFrame.
    """

    if lback_meth == "ma":
        dfwa = (
                dfw_ret.rolling(window=lback_periods).agg(flat_std, remove_zeros)
                * np.sqrt(252)
        )
    else:
        half_life = lback_periods
        weights = expo_weights(lback_periods * 2, half_life)
        dfwa = dfw_ret.rolling(window=lback_periods * 2).agg(
               expo_std, w=weights, remove_zeros=remove_zeros) * np.sqrt(252)

    w = 1 / dfwa
    weight = w / w.sum(axis=1).values[:, np.newaxis]
    check_weights(weight)

    return weight


def values_weight(weights: pd.DataFrame):
    """
    The values weight method will receive a matrix of values produced by another category
    and the values held in the matrix will be used as the weights. The weight matrix
    should contain floating point values.

    :param <pd.DataFrame> weights: DataFrame of exogenously computed weights.

    :return <pd.DataFrame>: Will return the generated weight Array
    """

    weights = weights / weights.sum(axis=1)[:, np.newaxis]
    check_weights(weights)

    return weights

def remove_rows(w_df, dfw_ret):
    """
    Function designed to remove any rows consisting exclusively of NaN values. For
    instance, on a universal holiday or after the application of the inverse standard
    deviation weighting method which requires the window to be populated before computing
    a weight. Therefore, all the preceding rows will hold NaN values.

    :param <pd.DataFrame> w_df: Array of returns. Multidimensional.
    :param <pd.DataFrame> dfw_ret: Corresponding weight matrix. Multidimensional.

    :return <pd.DataFrame>: Will return the two modified DataFrames
    """

    df_index = np.array(w_df.index)
    weights_bool = ~w_df.isnull()
    weights_bool = weights_bool.astype(dtype=np.uint8)

    act_cross = weights_bool.sum(axis=1)

    nan_rows = np.where(act_cross == 0)[0]
    nan_size = nan_rows.size

    w_matrix = w_df.to_numpy()
    r_matrix = dfw_ret.to_numpy()
    if not nan_size == 0:
        start = nan_rows[0]
        end = nan_rows[-1]
        iterator = np.array(range(start, end + 1))

        bool_size = np.all(iterator.size == nan_size)
        if bool_size and np.all(iterator == nan_rows):
            w_matrix = w_matrix[(end + 1):, :]
            r_matrix = r_matrix[(end + 1):, :]
            df_index = df_index[(end + 1):]
        else:
            w_matrix = np.delete(w_matrix, tuple(nan_rows), axis=0)
            r_matrix = np.delete(r_matrix, tuple(nan_rows), axis=0)
            df_index = np.delete(df_index, tuple(nan_rows), axis=0)

    df_w = pd.DataFrame(data=w_matrix, columns=list(w_df.columns), index=df_index)
    dfw_ret = pd.DataFrame(data=r_matrix, columns=list(w_df.columns), index=df_index)

    return df_w, dfw_ret

def max_weight_func(weights: pd.DataFrame, max_weight: float):
    """
    Function designed to determine if all weights computed are within the maximum weight
    allowed per cross-section. If the maximum weight is less than the expected weight,
    replace the computed weight with the expected weight. For instance,
    [np.nan, 0.63, np.nan, np.nan, 0.27] becomes [np.nan, 0.5, np.nan, np.nan, 0.5].
    Otherwise, call the ConvergeRow Class to ensure all weights "converge" to a value
    within the upper-bound. Allow for a margin of error set to 0.001.

    :param <pd.DataFrame> weights: Corresponding weight matrix. Multidimensional.
    :param <float> max_weight: Upper-bound on the weight allowed for each cross-section.

    :return <pd.DataFrame>: Will return the modified weight DataFrame.
    """

    weights = weights.fillna(0.0)
    w_matrix = weights.to_numpy()

    for i, row in enumerate(w_matrix):
        inst, row = ConvergeRow.application(row, max_weight)
        weights.iloc[i, :] = row

    cols = weights.columns
    weights[cols] = weights[cols].replace({'0': np.nan, 0: np.nan})
    return weights

def basket_performance(df: pd.DataFrame, contracts: List[str], ret: str = "XR_NSA",
                       cry: str = "CRY_NSA", start: str = None, end: str = None,
                       blacklist: dict = None, weight_meth: str = "equal",
                       lback_meth: str = "xma", lback_periods: int = 21,
                       remove_zeros: bool = True, weights: List[float] = None,
                       weight_xcat: str = None, max_weight: float = 1.0,
                       basket_tik: str = "GLB_ALL", return_weights: bool = False):

    """Basket performance
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
                              "fixed": proportionate to list values supplied separately.
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

    assert isinstance(ret, str), "return category must be a <str>."
    assert isinstance(contracts, list) and all(isinstance(c, str) for c in contracts)
    assert 0.0 < max_weight <= 1.0

    ticks_ret = [c + "_" + ret for c in contracts]

    cry_flag = cry is not None
    if cry_flag:
        ticks_cry = [c + "_" + cry for c in contracts]
    else:
        ticks_cry = []
    tickers = ticks_ret + ticks_cry

    dfx = reduce_df_by_ticker(df, start=start, end=end, ticks=tickers,
                              blacklist=blacklist)

    dfx["ticker"] = dfx["cid"] + "_" + dfx["xcat"]
    dfw_ret = dfx[dfx["ticker"].isin(ticks_ret)].pivot(index="real_date", columns="cid",
                                                       values="value")
    if cry_flag:
        dfw_cry = dfx[dfx["ticker"].isin(ticks_cry)].pivot(index="real_date",
                                                           columns="cid", values="value")

    if weight_meth == "equal":
        w_matrix = equal_weight(df_ret=dfw_ret)

    elif weight_meth == "fixed":
        assert isinstance(weights, list)
        w_str = [str(elem) for elem in weights]

        for elem in w_str:
            assert not elem.isnumeric()
        assert dfw_ret.shape[1] == len(weights)

        w_matrix = fixed_weight(df_ret=dfw_ret, weights=weights)

    elif weight_meth == "invsd":

        w_matrix = inverse_weight(dfw_ret=dfw_ret, lback_meth=lback_meth, lback_periods=
                                  lback_periods, remove_zeros=remove_zeros)
    elif weight_meth == "values" or weight_meth == "inv_values":

        assert dfw_ret.shape[1] == len(weights)
        # w_matrix = values_weight()
        raise NotImplementedError(
            "Not yet implemented - need to extract the panel of weights from the input"
            " DataFrame")
    else:
        raise NotImplementedError(f"Weight method unknown {weight_meth}")

    w_matrix, dfw_ret = remove_rows(w_matrix, dfw_ret)
    if max_weight < 1.0:
        w_matrix = max_weight_func(weights=w_matrix, max_weight=max_weight)

    select = ["ticker", "real_date", "value"]
    dfxr = (dfw_ret.multiply(w_matrix)).sum(axis=1)
    dfxr.index.name = "real_date"
    dfxr = dfxr.to_frame("value").reset_index()
    dfxr = dfxr.assign(ticker=basket_tik + "_" + ret)[select]
    store = [dfxr]

    if cry_flag:
        dfcry = (dfw_cry.multiply(w_matrix)).sum(axis=1)
        dfcry.index.name = "real_date"
        store.append(dfcry.to_frame("value").reset_index().assign(ticker=basket_tik +
                                                                  "_" + cry)[select])
    if return_weights:
        w_matrix.index.name = "real_date"
        w_matrix.columns.name = "cid"
        w = w_matrix.stack().to_frame("value").reset_index()
        w["ticker"] = w["cid"] + "_WGTS"

        w = w.loc[w.value > 0, select]
        store.append(w)

    # Concatenate along the date index and subsequently drop to restore natural index.
    df = pd.concat(store, axis=0, ignore_index=True)
    return df


if __name__ == "__main__":
    cids = ['AUD', 'GBP', 'NZD', 'USD']
    xcats = ['FX_XR', 'FX_CRY', 'EQ_XR', 'EQ_CRY']

    df_cids = pd.DataFrame(index=cids, columns=['earliest', 'latest', 'mean_add',
                                                'sd_mult'])

    df_cids.loc['AUD'] = ['2010-01-01', '2020-12-31', 0, 1]
    df_cids.loc['GBP'] = ['2011-01-01', '2020-11-30', 0, 2]
    df_cids.loc['NZD'] = ['2012-01-01', '2020-11-30', 0, 3]
    df_cids.loc['USD'] = ['2013-01-01', '2020-09-30', 0, 4]

    df_xcats = pd.DataFrame(index=xcats, columns=['earliest', 'latest', 'mean_add',
                                                  'sd_mult', 'ar_coef', 'back_coef'])
    df_xcats.loc['FX_XR'] = ['2010-01-01', '2020-12-31', 0, 1, 0, 0.2]
    df_xcats.loc['FX_CRY'] = ['2011-01-01', '2020-10-30', 1, 1, 0.9, 0.5]
    df_xcats.loc['EQ_XR'] = ['2012-01-01', '2020-10-30', 0.5, 2, 0, 0.2]
    df_xcats.loc['EQ_CRY'] = ['2013-01-01', '2020-10-30', 1, 1, 0.9, 0.5]

    random.seed(2)
    dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)
    black = {'AUD': ['2000-01-01', '2003-12-31'], 'GBP': ['2018-01-01', '2100-01-01']}

    contracts = ['AUD_FX', 'NZD_FX', 'GBP_EQ', 'USD_EQ']

    gdp_figures = [17.0, 41.0, 9.0, 215.0]
    dfd_1 = basket_performance(dfd, contracts, ret='XR', cry=None,
                               weight_meth='fixed', weights=gdp_figures,
                               weight_xcat=None, max_weight=0.35, return_weights=False)

    dfd_2 = basket_performance(dfd, contracts, ret='XR', cry=None,
                               weight_meth='fixed', weights=gdp_figures,
                               weight_xcat=None, max_weight=0.35, return_weights=True)

    dfd_3 = basket_performance(dfd, contracts, ret='XR', cry=None,
                               weight_meth='invsd', weights=gdp_figures,
                               lback_meth="xma", lback_periods=21,
                               weight_xcat=None, max_weight=0.39, return_weights=True)

    print(dfd_3)
    dfd_4 = basket_performance(dfd, contracts, ret='XR', cry=None,
                               weight_meth='equal', weight_xcat=None, max_weight=0.41,
                               return_weights=False)
    print(dfd_4)