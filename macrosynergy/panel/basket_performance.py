import numpy as np
import pandas as pd
import random
from typing import List
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
    Equal weight function: receives the pivoted return DataFrame and determines the
    number of non-NA cross-sections per timestamp, and subsequently distribute the weight
    evenly across non-NA cross-sections.

    :param <pd.DataFrame> df_ret: data-frame with returns.

    :return <pd.DataFrame> : dataframe of weights.
    """

    act_cross = (~df_ret.isnull())  # df with 1s/0s for non-NA/NA returns
    uniform = (1 / act_cross.sum(axis=1)).values  # single equal value
    uniform = uniform[:, np.newaxis]
    # Apply equal value to all cross sections
    broadcast = np.repeat(uniform, df_ret.shape[1], axis=1)

    weight = act_cross.multiply(broadcast)
    check_weights(weight=weight)

    return weight


def fixed_weight(df_ret: pd.DataFrame, weights: List[float]):
    """
    Calculates fixed weights based on a single list of values and a corresponding return
    panel dataframe

    :param <pd.DataFrame> df_ret: Return series matrix. Multidimensional.
    :param <List[float]> weights: List of floats determining weight allocation.

    :return <pd.DataFrame>: panel of weights
    """

    act_cross = (~df_ret.isnull())  # df with 1s/0s for non-NA/NA returns

    weights = np.array(weights, dtype=np.float32)
    rows = act_cross.shape[0]
    broadcast = np.tile(weights, (rows, 1))  # Constructs array by row repetition.

    # Replaces weight factors with zeroes if concurrent return unavailable
    weight = act_cross.multiply(broadcast)
    weight_arr = weight.to_numpy()  # convert df to np array
    weight[weight.columns] = weight_arr / np.sum(weight_arr, axis=1)[:, np.newaxis]

    check_weights(weight)

    return weight


def inverse_weight(dfw_ret: pd.DataFrame, lback_meth: str = "xma",
                   lback_periods: int = 21, remove_zeros: bool = True):
    """
    Calculates weights inversely proportionate to recent return standard deviations

    :param <pd.DataFrame> dfw_ret: panel dataframe of returns.
    :param <str> lback_meth: Lookback method for "invsd" weighting method. Default is
        "xma".
    :param <int> lback_periods: Lookback periods. Default is 21.  Half-time for "xma"
        and full lookback period for "ma".
    :param <Bool> remove_zeros: Any returns that are exact zeros will not be included in
        the lookback window and prior non-zero values are added to the window instead.

    :return <pd.DataFrame>: Dataframe of weights.

    N.B.: The rolling standard deviation will be calculated either using the standard
    moving average (ma) or the exponential moving average (xma). Both will require
    returns before a first weight can be computed.
    """

    if lback_meth == "ma":
        dfwa = dfw_ret.rolling(window=lback_periods).agg(flat_std, remove_zeros)
        dfwa *= np.sqrt(252)

    else:

        half_life = lback_periods
        weights = expo_weights(lback_periods * 2, half_life)
        dfwa = dfw_ret.rolling(window=lback_periods * 2).agg(expo_std, w=weights,
                                                             remove_zeros=remove_zeros)

    df_isd = 1 / dfwa
    df_wgts = df_isd / df_isd.sum(axis=1).values[:, np.newaxis]
    check_weights(df_wgts)

    return df_wgts


def values_weight(dfw_ret: pd.DataFrame, dfw_wgt: pd.DataFrame, weight_meth: str):
    """
    Returns weights based on an external weighting category.

    :param <pd.DataFrame> dfw_ret: Standard wide dataframe of returns across time and
        contracts.
    :param <pd.DataFrame> dfw_wgt: Standard wide dataframe of weight category values
        across time and contracts.
    :param <str> weight_meth: Weighting method. must be one of "values" or "inv_values".

    :return <pd.DataFrame>: Dataframe of weights.
    """

    negative_condition = np.any((dfw_wgt < 0).to_numpy())
    if negative_condition:
        dfw_wgt[dfw_wgt < 0] = 0.0
        print("Negative values in the weight matrix set to zero.")

    exo_array = dfw_wgt.to_numpy()
    df_bool = ~dfw_ret.isnull()

    weights_df = df_bool.multiply(exo_array)
    cols = weights_df.columns

    # zeroes treated as NaNs
    weights_df[cols] = weights_df[cols].replace({'0': np.nan, 0: np.nan})

    if weight_meth != "values":
        weights_df = 1 / weights_df

    weights = weights_df.divide(weights_df.sum(axis=1), axis=0)
    check_weights(weights)

    return weights


def max_weight_func(weights: pd.DataFrame, max_weight: float):
    """
    Enforces maximum weight caps or - if impossible applies equal weight.

    :param <pd.DataFrame> weights: Corresponding weight matrix. Multidimensional.
    :param <float> max_weight: Upper-bound on the weight allowed for each cross-section.

    :return <pd.DataFrame>: Will return the modified weight DataFrame.

    N.B.: If the maximum weight is less than the equal weight weight, this replaces the
    computed weight with the equal weight. For instance,
    [np.nan, 0.63, np.nan, np.nan, 0.27] becomes [np.nan, 0.5, np.nan, np.nan, 0.5].
    Otherwise, the function calls the ConvergeRow Class to ensure all weights "converge"
    to a value within the upper-bound. Allow for a margin of error set to 0.001.
    """

    dfw_wgs = weights.to_numpy()

    for i, row in enumerate(dfw_wgs):
        row = ConvergeRow.application(row, max_weight)
        weights.iloc[i, :] = row

    return weights


def basket_performance(df: pd.DataFrame, contracts: List[str], ret: str = "XR_NSA",
                       cry: str = None, start: str = None, end: str = None,
                       blacklist: dict = None, weight_meth: str = "equal",
                       lback_meth: str = "xma", lback_periods: int = 21,
                       remove_zeros: bool = True, weights: List[float] = None,
                       wgt: str = None, max_weight: float = 1.0,
                       basket_tik: str = "GLB_ALL", return_weights: bool = False):

    """
    Basket performance returns approximate return and - optionally - carry series for a
    basket of underlying contracts.

    :param <pd.Dataframe> df: standardized DataFrame with following columns: 'cid',
        'xcats', 'real_date' and 'value'.
    :param <List[str]> contracts: base tickers (combinations of cross sections and base
        categories) denoting contracts that go into the basket.
    :param <str> ret: return category postfix; default is "XR_NSA".
    :param <str> cry: carry category postfix; default is None.
    :param <str> start: earliest date in ISO 8601 format. Default is None, i.e.
        earliest date in data frame is used.
    :param <str> end: latest date in ISO 8601 format. Default is None, i.e. latest date
        in data frame is used.
    :param <dict> blacklist: cross sections with date ranges that should be excluded from
        the data frame. If one cross section has several blacklist periods append numbers
        to the cross section code.
    :param <str> weight_meth: method used for weighting constituent returns and carry.
        Options are as follows:
    [1] "equal": all constituents with non-NA returns have the same weight.
        This is the default.
    [2] "fixed": weights are proportionate to single list of values (corresponding to
        contracts) provided passed to argument `weights`.
    [3] "invsd": weights based on inverse to standard deviations of recent returns.
    [4] "values": weights proportionate to a panel of values of exogenous weight
        category.
    [5] "inv_values": weights are inversely proportionate to of values of exogenous
        weight category.
    :param <str> lback_meth: lookback method for "invsd" weighting method. Default is
        "xma" (exponential MA). The alternative is simple moving average ("ma").
    :param <int> lback_periods: lookback periods. Default is 21. Half-time for "xma" and
        full lookback period for "ma".
    :param <bool> remove_zeros: removes the zeros. Default is set to True.
    :param <List[float]> weights: single list of weights corresponding to the base
        tickers in the contracts argument. This is only relevant for weight_meth="fixed".
    :param <str> wgt: postfix used to identify exogenous weight category. Analogously to
        carry and return postfixes this should be added to base tickers to identify the
        values that denote contract weights.
    :param <float> max_weight: maximum weight of a single contract. Default is 1, i.e no
        restriction. The purpose of the restriction is to limit concentration within the
        basket.
    :param <str> basket_tik: name of basket base ticker (analogous to contract name) to
        be used for  return and (possibly) carry are calculated. Default is "GLB_ALL".
    :param <bool> return_weights: if True ddd cross-section weights to output dataframe
        with 'WGT' postfix. Default is False.

    :return <pd.Dataframe>: standardized DataFrame with the basket return and (possibly)
        carry data in standard form, i.e. columns 'cid', 'xcats', 'real_date' and 'value'.
    """

    assert isinstance(ret, str), "return category must be a <str>."
    assert isinstance(contracts, list) and all(isinstance(c, str) for c in contracts), \
        "contracts must be string list."
    assert 0.0 < max_weight <= 1.0

    # A. Extract relevant data

    ticks_ret = [c + ret for c in contracts]
    tickers = ticks_ret.copy()  # Initiates general tickers list.

    cry_flag = cry is not None  # Boolean for carry being used.
    if cry_flag:
        # Creates a List of contract carries
        ticks_cry = [c + cry for c in contracts]
        tickers += ticks_cry  # Add to general ticker list.

    wgt_flag = (wgt is not None) and (weight_meth in ["values", "inv_values"])
    if wgt_flag:
        assert isinstance(wgt, str), f"Parameter, 'wgt', must be a string and not a " \
                                      "{type(wgt)."
        ticks_wgt = [c + wgt for c in contracts]
        tickers += ticks_wgt

    dfx = reduce_df_by_ticker(df, start=start, end=end, ticks=tickers,
                              blacklist=blacklist)  # extract relevant df

    # B. Pivot relevant data

    dfx_ticks_ret = dfx[dfx["ticker"].isin(ticks_ret)]
    dfw_ret = dfx_ticks_ret.pivot(index="real_date", columns="ticker", values="value")

    if cry_flag:
        dfw_ticks_cry = dfx[dfx["ticker"].isin(ticks_cry)]
        dfw_cry = dfw_ticks_cry.pivot(index="real_date", columns="ticker",
                                      values="value")

    # C. Apply the appropriate weighting method

    if weight_meth == "equal":
        dfw_wgs = equal_weight(df_ret=dfw_ret)

    elif weight_meth == "fixed":
        assert isinstance(weights, list), "Method 'fixed' requires a list of weights " \
                                          "to be assigned to 'weights'."
        assert dfw_ret.shape[1] == len(weights), "List of weights must be equal " \
                                                 "to number of contracts."
        for w in weights:
            try:
                int(w)
            except ValueError:
                print(f"List, {weights}, must be all numerical values.")
                raise
        dfw_wgs = fixed_weight(df_ret=dfw_ret, weights=weights)

    elif weight_meth == "invsd":
        dfw_wgs = inverse_weight(dfw_ret=dfw_ret, lback_meth=lback_meth,
                                  lback_periods=lback_periods, remove_zeros=remove_zeros)

    elif wgt_flag:
        ticks_in_df = list(set(df["ticker"].to_numpy()))
        for w_ticker in ticks_wgt:
            assert w_ticker in ticks_in_df, "Weight Ticker, {w_ticker}, absent from " \
                                            "the dataframe. Unable to be used as an " \
                                            "external weight category."
        dfw_ticks_wgt = dfx[dfx["ticker"].isin(ticks_wgt)]
        dfw_wgt = dfw_ticks_wgt.pivot(index="real_date", columns="ticker",
                                      values="value")

        dfw_wgt = dfw_wgt.shift(1)  # lag by one day to be used as weights
        dfw_ret = dfw_ret.reindex(sorted(dfw_ret.columns), axis=1)
        dfw_wgt = dfw_wgt.reindex(sorted(dfw_wgt.columns), axis=1)
        dfw_wgs = values_weight(dfw_ret, dfw_wgt, weight_meth)

    else:
        raise NotImplementedError(f"Weight method unknown {weight_meth}")

    # D. Remove leading NA rows.

    fvi = max(dfw_wgs.first_valid_index(), dfw_ret.first_valid_index())
    dfw_wgs, dfw_ret = dfw_wgs[fvi:], dfw_ret[fvi:]

    # E. Impose cap on cross-section weight

    if max_weight < 1.0:
        dfw_wgs = max_weight_func(weights=dfw_wgs, max_weight=max_weight)

    # F. Calculate and store weighted average returns

    select = ["ticker", "real_date", "value"]
    dfxr = (dfw_ret.multiply(dfw_wgs)).sum(axis=1)  # calculate weighted averages
    dfxr = dfxr.to_frame("value").reset_index()
    dfxr = dfxr.assign(ticker=basket_tik + "_" + ret)[select]
    store = [dfxr]

    if cry_flag:
        dfcry = (dfw_cry.multiply(dfw_wgs)).sum(axis=1)
        dfcry = dfcry.to_frame("value").reset_index()
        dfcry = dfcry.assign(ticker=basket_tik + "_" + cry)[select]
        store.append(dfcry)
    if return_weights:
        dfw_wgs.columns.name = "cid"
        w = dfw_wgs.stack().to_frame("value").reset_index()
        contracts_ = list(map(lambda c: c[:-len(ret)] + "WGT", w["cid"].to_numpy()))
        contracts_ = np.array(contracts_)
        w["ticker"] = contracts_
        w = w.loc[w.value > 0, select]
        store.append(w)

    # Concatenate along the date index and subsequently drop to restore natural index.
    df = pd.concat(store, axis=0, ignore_index=True)
    return df


if __name__ == "__main__":

    cids = ['AUD', 'GBP', 'NZD', 'USD']

    xcats = ['FXXR_NSA', 'FXCRY_NSA', 'EQXR_NSA', 'EQCRY_NSA',
             'FXWBASE_NSA', 'EQWBASE_NSA']

    df_cids = pd.DataFrame(index=cids, columns=['earliest', 'latest', 'mean_add',
                                                'sd_mult'])

    df_cids.loc['AUD'] = ['2010-01-01', '2020-12-31', 0, 1]
    df_cids.loc['GBP'] = ['2011-01-01', '2020-11-30', 0, 2]
    df_cids.loc['NZD'] = ['2012-01-01', '2020-12-31', 0, 3]
    df_cids.loc['USD'] = ['2013-01-01', '2020-12-31', 0, 4]

    df_xcats = pd.DataFrame(index=xcats, columns=['earliest', 'latest', 'mean_add',
                                                  'sd_mult', 'ar_coef', 'back_coef'])
    df_xcats.loc['FXXR_NSA'] = ['2010-01-01', '2020-12-31', 0, 1, 0, 0.2]
    df_xcats.loc['FXCRY_NSA'] = ['2011-01-01', '2020-10-30', 1, 1, 0.9, 0.5]
    df_xcats.loc['FXWBASE_NSA'] = ['2010-01-01', '2020-12-31', 100, 1, 0.9, 0.5]
    df_xcats.loc['EQXR_NSA'] = ['2012-01-01', '2020-10-30', 0.5, 2, 0, 0.2]
    df_xcats.loc['EQCRY_NSA'] = ['2013-01-01', '2020-10-30', 1, 1, 0.9, 0.5]
    df_xcats.loc['EQWBASE_NSA'] = ['2010-01-01', '2020-12-31', 100, 1, 0.9, 0.5]

    random.seed(2)
    dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)

    black = {'AUD': ['2000-01-01', '2003-12-31'], 'GBP': ['2018-01-01', '2100-01-01']}
    contracts = ['AUD_FX', 'AUD_EQ', 'NZD_FX', 'GBP_EQ', 'USD_EQ']

    gdp_figures = [17.0, 17.0, 41.0, 9.0, 250]
    dfd_1 = basket_performance(dfd, contracts, ret='XR_NSA', cry='CRY_NSA',
                               weight_meth='values', wgt='WBASE_NSA', max_weight=0.35,
                               return_weights=False)

    dfd_2 = basket_performance(dfd, contracts, ret='XR_NSA', cry=None,
                               weight_meth='fixed', weights=gdp_figures,
                               wgt=None, max_weight=0.35, return_weights=False)

    dfd_3 = basket_performance(dfd, contracts, ret='XR_NSA', cry=None,
                               weight_meth='invsd', weights=None, lback_meth="xma",
                               lback_periods=21)

    dfd_4 = basket_performance(dfd, contracts, ret='XR_NSA', cry=None,
                               weight_meth='equal', wgt=None, max_weight=0.41,
                               return_weights=True)
    print(dfd_4)


