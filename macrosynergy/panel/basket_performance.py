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
    """
    Equal weight function: receives the pivoted return DataFrame and determine the number of non-NA
    cross-sections per timestamp, and subsequently distribute the weight evenly across non-NA cross-sections.

    :param <pd.DataFrame> df_ret: data-frame with returns.

    :return <pd.DataFrame> : dataframe of weights.
    """

    act_cross = (~df_ret.isnull())  # df with 1s/0s for non-NA/NA returns
    uniform = (1 / act_cross.sum(axis=1)).values  # single equal value
    uniform = uniform[:, np.newaxis]
    broadcast = np.repeat(uniform, df_ret.shape[1], axis=1)  # apply equal value to all cross sections

    weight = act_cross.multiply(broadcast)  # apply equal value only to non-NA cross section
    check_weights(weight=weight)

    return weight


def fixed_weight(df_ret: pd.DataFrame, weights: List[float]):
    """
    Calculates fixed weights based on a single list of values and a corresponding return panel dataframe

    :param <pd.DataFrame> df_ret: Return series matrix. Multidimensional.
    :param <List[float]> weights: List of floats determining weight allocation.

    :return <pd.DataFrame>: panel of weights
    """

    act_cross = (~df_ret.isnull())  # df with 1s/0s for non-NA/NA returns

    weights = np.array(weights, dtype=np.float32)  # convert list to floating point array
    rows = act_cross.shape[0]
    broadcast = np.tile(weights, (rows, 1))  # constructs array by row repetition

    weight = act_cross.multiply(broadcast)  # replaces weight factors with zeroes if concurrent return unavailable
    weight_arr = weight.to_numpy()  # convert df to np array
    weight[weight.columns] = weight_arr / np.sum(weight_arr, axis=1)[:, np.newaxis]  # fill old df with weights

    check_weights(weight)

    return weight


def inverse_weight(dfw_ret: pd.DataFrame, lback_meth: str = "xma",
                   lback_periods: int = 21, remove_zeros: bool = True):
    """
    Calculates weights inversely proportionate to recent return standard deviations

    :param <pd.DataFrame> dfw_ret: panel dataframe of returns.
    :param <str> lback_meth: Lookback method for "invsd" weighting method. Default is "xma".
    :param <int> lback_periods: Lookback periods. Default is 21.  Half-time for "xma" and full lookback period for "ma".
    :param <Bool> remove_zeros: Removes the zeros. Default is set to True.

    :return <pd.DataFrame>: Dataframe of weights.

    N.B:   The rolling standard deviation will be calculated either using the standard moving average (ma) or the
    exponential moving average (xma). Both will require returns before a first weight can be computed
    Todo: this function will need to option to shorten the window and/or backfill if history is short or precious
    """

    if lback_meth == "ma":

        dfwa = dfw_ret.rolling(window=lback_periods).agg(flat_std, remove_zeros) * np.sqrt(252)

    else:

        half_life = lback_periods
        weights = expo_weights(lback_periods * 2, half_life)
        dfwa = dfw_ret.rolling(window=lback_periods * 2).agg(expo_std, w=weights, remove_zeros=remove_zeros)

    df_isd = 1 / dfwa  # df of inverses of SD
    df_wgts = df_isd / df_isd.sum(axis=1).values[:, np.newaxis]  # divide inverse SDs by rowwise sums
    check_weights(df_wgts)

    return df_wgts


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
    Enforces maximum weight caps or - if impossible applies equal weight.
    Checks if maximum weights are below equal weights and applies latter if True.

    :param <pd.DataFrame> weights: Corresponding weight matrix. Multidimensional.
    :param <float> max_weight: Upper-bound on the weight allowed for each cross-section.

    :return <pd.DataFrame>: Will return the modified weight DataFrame.

    N.B.: If the maximum weight is less than the equal weight weight, this replaces the computed weight with the
    equal weight. For instance, [np.nan, 0.63, np.nan, np.nan, 0.27] becomes [np.nan, 0.5, np.nan, np.nan, 0.5].
    Otherwise, the function calls the ConvergeRow Class to ensure all weights "converge" to a value
    within the upper-bound. Allow for a margin of error set to 0.001.
    """

    weights = weights.fillna(0.0)
    w_matrix = weights.to_numpy()

    for i, row in enumerate(w_matrix):
        inst, row = ConvergeRow.application(row, max_weight)
        # Todo: replace by simple function made from application+distribute_simple in converge row
        weights.iloc[i, :] = row

    cols = weights.columns
    weights[cols] = weights[cols].replace({'0': np.nan, 0: np.nan})
    return weights


def basket_performance(df: pd.DataFrame, contracts: List[str], ret: str = "XR_NSA", cry: str = None,
                       start: str = None, end: str = None, blacklist: dict = None, weight_meth: str = "equal",
                       lback_meth: str = "xma", lback_periods: int = 21, remove_zeros: bool = True,
                       weights: List[float] = None, weight_xcat: str = None, max_weight: float = 1.0,
                       basket_tik: str = "GLB_ALL", return_weights: bool = False):

    """Basket performance
    Computes approximate return and carry series for a basket of underlying contracts.

    :param <pd.Dataframe> df: standardized DataFrame with following columns: 'cid', 'xcats', 'real_date' and 'value'.
    :param <List[str]> contracts: base tickers (combinations of cross sections and base  categories) denoting contracts
        that go into the basket.
    :param <str> ret: return category postfix; default is "XR_NSA".
    :param <str> cry: carry category postfix; default is None.
    :param <str> start: earliest date in ISO 8601 format. Default is None, i.e. earliest date in data frame is used.
    :param <str> end: latest date in ISO 8601 format. Default is None, i.e. latest date in data frame is used.
    :param <dict> blacklist: cross sections with date ranges that should be excluded from the data frame.
        If one cross section has several blacklist periods append numbers to the cross section code.
    :param <str> weight_meth: method used for weighting constituent returns and carry. Options are as follows:
        [1] "equal": all constituents with non-NA returns have the same weight. This is the default.
        [2] "fixed": weights are proportionate to a single list values provided separately.
        [3] "invsd": weights are inverse of past return standard deviations.
        [4] "values": weights are proportionate to a panel of values of another exogenous category.
        [5] "inv_values": weights are inversely proportionate to of values of another exogenous category.
    :param <str> lback_meth: lookback method for "invsd" weighting method. Default is "xma" (exponential MA).
        The alternative is simple moving average ("ma").
    :param <int> lback_periods: lookback periods. Default is 21.  Half-time for "xma" and full lookback period for "ma".
    :param <Bool> remove_zeros: removes the zeros. Default is set to True.  Todo: explain!
    :param <List[float]> weights: single list of weights corresponding to the base tickers in the contracts argument.
        This is only relevant for weight_meth = "fixed".
    :param <str> weight_xcat: extended category name for "values" and "inv_values" methods.
    :param <float> max_weight: maximum weight permitted for a single cross section. Default is 1, i.e no restriction.
    :param <str> basket_tik: name of basket base ticker for which return and (possibly) carry are calculated.
        Default is "GLB_ALL".
    :param <bool> return_weights: if True ddd cross-section weights to output dataframe with 'WGT' postfix.
        Default is False.


    :return <pd.Dataframe>: standardized DataFrame with the basket performance data in standard form,
                            i.e. columns are 'cid', 'xcat', 'real_date' and 'value'.
    """

    assert isinstance(ret, str), "return category must be a <str>."
    assert isinstance(contracts, list) and all(isinstance(c, str) for c in contracts)
    assert 0.0 < max_weight <= 1.0

    # A. Extract relevant data

    ticks_ret = [c + "_" + ret for c in contracts]

    cry_flag = cry is not None
    if cry_flag:
        ticks_cry = [c + "_" + cry for c in contracts]
    else:
        ticks_cry = []
    tickers = ticks_ret + ticks_cry

    # Todo: this must contain exogenous weights for the values method (to be addressed later)

    dfx = reduce_df_by_ticker(df, start=start, end=end, ticks=tickers, blacklist=blacklist)

    dfx["ticker"] = dfx["cid"] + "_" + dfx["xcat"]

    # B. Pivot relevant data

    dfw_ret = dfx[dfx["ticker"].isin(ticks_ret)].pivot(index="real_date", columns="cid", values="value")
    if cry_flag:
        dfw_cry = dfx[dfx["ticker"].isin(ticks_cry)].pivot(index="real_date", columns="cid", values="value")

    # C. Create chosen type of weights

    if weight_meth == "equal":

        w_matrix = equal_weight(df_ret=dfw_ret)

    elif weight_meth == "fixed":

        assert isinstance(weights, list), "Method fixed requires a list of weights to be assigned to 'weights'"
        assert all([str(w).isnumeric() for w in weights]), "all weight list elements must be numeric"
        assert dfw_ret.shape[1] == len(weights), "List of weights must be equal to number of available cross sections"
        w_matrix = fixed_weight(df_ret=dfw_ret, weights=weights)

    elif weight_meth == "invsd":

        w_matrix = inverse_weight(dfw_ret=dfw_ret, lback_meth=lback_meth, lback_periods= lback_periods,
                                  remove_zeros=remove_zeros)

    elif weight_meth == "values" or weight_meth == "inv_values":

        assert dfw_ret.shape[1] == len(weights)
        # w_matrix = values_weight()
        raise NotImplementedError(
            "Not yet implemented - need to extract the panel of weights from the input"
            " DataFrame")
    else:
        raise NotImplementedError(f"Weight method unknown {weight_meth}")

    # D. Remove leading NA rows

    fvi = max(w_matrix.first_valid_index(), dfw_ret.first_valid_index())  # first valid non-NA index
    w_matrix, dfw_ret = w_matrix[fvi:], dfw_ret[fvi:]  # Todo: check if this allows ditching remove_rows

    # E. Impose cap on cross-section weight

    if max_weight < 1.0:
        w_matrix = max_weight_func(weights=w_matrix, max_weight=max_weight)  # enforce max weights if convergence
        # Todo: see above, new function may simplify

    # F. Calculate and store weighted average returns

    np.sum(w_matrix, axis=1).all(), "Weights do not add up to 1 for all periods"

    select = ["ticker", "real_date", "value"]
    dfxr = (dfw_ret.multiply(w_matrix)).sum(axis=1)  # calculate weighted averages
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

    # gdp_figures = [17.0, 41.0, 9.0, 215.0]
    # dfd_1 = basket_performance(dfd, contracts, ret='XR', cry=None,
    #                            weight_meth='fixed', weights=gdp_figures,
    #                            weight_xcat=None, max_weight=0.35, return_weights=False)
    #
    # dfd_2 = basket_performance(dfd, contracts, ret='XR', cry=None,
    #                            weight_meth='fixed', weights=gdp_figures,
    #                            weight_xcat=None, max_weight=0.35, return_weights=True)

    dfd_3 = basket_performance(dfd, contracts, ret='XR', cry=None,
                               weight_meth='invsd', weights=None,
                               lback_meth="xma", lback_periods=21,
                               weight_xcat=None, max_weight=0.4, return_weights=True)

    print(dfd_3)
    dfd_4 = basket_performance(dfd, contracts, ret='XR', cry=None,
                               weight_meth='equal', weight_xcat=None, max_weight=0.41,
                               return_weights=False)
    print(dfd_4)