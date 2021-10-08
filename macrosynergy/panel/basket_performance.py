import numpy as np
import pandas as pd
from typing import List, Union, Tuple
import random

from macrosynergy.management.simulate_quantamental_data import make_qdf
from macrosynergy.management.shape_dfs import reduce_df, reduce_df_by_ticker


def basket_performance(df: pd.DataFrame, contracts: List[str], ret: str = 'XR_NSA', cry: str = 'CRY_NSA',
                       start: str = None, end: str = None, blacklist: dict = None,
                       weight_meth: str = 'equal', lback_meth: str = 'xma', lback_periods: int = 21,
                       weights: List[float] = None, weight_xcat: str = None, max_weight: float = 1.0,
                       basket_tik: str = 'GLB_ALL', return_weights: bool = False):
    """
    Computes approximate return and carry series for a basket of underlying contracts

    :param <pd.Dataframe> df: standardized data frame with the following necessary columns:
        'cid', 'xcats', 'real_date' and 'value.
    :param <List[str]> contracts: base tickers (combinations of cross sections and base categories) denoting contracts
        that go into the basket. Base category means that the strings exclude the return/ carry postfix.
    :param <str> ret: return catgory postfix; default is "XR_NSA".
    :param <str> cry: return catgory postfix; default is "CRY_NSA". Choose None if carry not available.
    :param <str> start: string representing earliest date. Default is None.
    :param <str> end: string representing the latest date. Default is None.
    :param <dict> blacklist: cross sections with date ranges that should be excluded from the data frame.
        If one cross section has several blacklist periods append numbers to the cross section code.
    :param <str> weight_meth: method for weighting constituent returns and carry. Default is "equal", which means
        all constituents have the same weight, which weights are just the inverse of the available cross sections
        at each point in time. Alternatives are:
        "fixed" (proportionate to vector values supplied separately),
        "invsd" (inverse of past return standard deviations),
        "values" (proportionate to a panel of values of another category), and
        "inv_values" (inversely proportionate to proportionate to a panel of exogenous values supplied separately).
    :param <str> lback_meth: lookback method for "invsd" method. Default is "xma" (exponential moving average).
        Alternative is "ma", simple moving average.
    :param <int> lback_periods: lookback periods. Default is 21. Refers to half-time for "xma" and full lookback period
        for "ma".
    :param <List[float]> weights: list of weights corresponding to the tickers. Only relevant if weight_meth = 'fixed'
        is chosen. Default is None, which just means that the 'equal' method is applied.
    :param <str> weight_xcat: extended category name of values used for "values" and "inv_values" methods.
    :param <float> max_weight: maximum weight permitted for a single cross section. Default is 1 (no restriction).
        If chosen and binding, excess weights are redistributed until the condition is satisfied.
        If maximum weight is set below equal weight for available cross-sections, equal weights are chosen.
    :param <str> basket_tik: name of basket base ticker for which return and (possibly) carry are calculated.
        Default is "GLB_ALL".
    :param <bool> return_weights: add cross-section weights to output dataframe uses tiks with 'WGT' postfix.
        Default is False.

    :return <pd.Dataframe>: standardized dataframe with the basket performance data in standard form:
        'cid', 'xcat', 'real_date' and 'value'.
    """

    ticks_ret = [contract + '_' + ret for contract in contracts]
    if cry is not None:
       ticks_cry = [contract + '_' + cry for contract in contracts]
    else:
        ticks_cry = []
    tickers = ticks_ret + ticks_cry

    dfx = reduce_df_by_ticker(df, start=start, end=end, ticks=tickers, blacklist=black)

    dfx['tick'] = dfx['cid'] + '_' + dfx['xcat']
    dfx_ret = dfx[dfx['tick'].isin(ticks_ret)]
    dfw_ret = dfx_ret.pivot(index='real_date', columns='cid', values='value')

    if cry is not None:
        dfx_cry = dfx[dfx['tick'].isin(ticks_cry)]
        dfw_cry = dfx_ret.pivot(index='real_date', columns='cid', values='value')

    if weight_meth == 'equal':
        pass
        # Todo: calculate dfw_weights where cells contain 1/n, where n is the number of non-nan cells in the row
        # Todo: multiply weights with dfw_ret to get basket return
        # Todo: multiply weights with dfw_cry to get basket carry
    elif weight_meth == 'fixed':
        pass
        # Todo: calculate dfw_weights where rows correspond to weights vector elements divided by sum of non-nan weights
        # Todo: multiply weights with dfw_ret to get basket return
        # Todo: multiply weights with dfw_cry to get basket carry
    elif weight_meth == 'invsd':
        pass
        # Todo: calculate dfw_weights as inverse of standard deviations as calculated by historic_vol
        # Todo: ensure that non-nan row-weights always add up to 1.
        # Todo: multiply weights with dfw_ret to get basket return
        # Todo: multiply weights with dfw_cry to get basket carry
    elif weight_meth == 'values' or weight_meth == 'inv_values':
        pass
        # Todo: do not implement yet
        # Note: this requires appropriate example data

    if max_weight < 1:
        pass
        # Todo: do not implement yet
        # Note: This requires an algorithm that sequentially redistributes weights until convergence


    # Todo: create standard dataframe with basket returns and carry
    # Todo: If return_weights is True also add contract weights to the data frame (category is contract + "_WGT")
    # Todo: return standard dataframe


if __name__ == "__main__":

    cids = ['AUD', 'GBP', 'NZD', 'USD', 'TRY']
    xcats = ['FX_XR', 'FX_CRY', 'EQ_XR', 'EQ_CRY']

    df_cids = pd.DataFrame(index=cids, columns=['earliest', 'latest', 'mean_add', 'sd_mult'])

    df_cids.loc['AUD'] = ['2010-01-01', '2020-12-31', 0, 1]
    df_cids.loc['GBP'] = ['2011-01-01', '2020-11-30', 0, 2]
    df_cids.loc['NZD'] = ['2012-01-01', '2020-11-30', 0, 3]
    df_cids.loc['USD'] = ['2013-01-01', '2020-09-30', 0, 4]
    df_cids.loc['TRY'] = ['2002-01-01', '2020-09-30', 0, 5]

    df_xcats = pd.DataFrame(index=xcats, columns=['earliest', 'latest', 'mean_add', 'sd_mult', 'ar_coef', 'back_coef'])
    df_xcats.loc['FX_XR'] = ['2010-01-01', '2020-12-31', 0, 1, 0, 0.2]
    df_xcats.loc['FX_CRY'] = ['2011-01-01', '2020-10-30', 1, 1, 0.9, 0.5]
    df_xcats.loc['EQ_XR'] = ['2012-01-01', '2020-10-30', 0.5, 2, 0, 0.2]
    df_xcats.loc['EQ_CRY'] = ['2013-01-01', '2020-10-30', 1, 1, 0.9, 0.5]

    random.seed(2)
    dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)
    black = {'AUD': ['2000-01-01', '2003-12-31'], 'GBP': ['2018-01-01', '2100-01-01']}

    contracts = ['AUD_FX', 'NZD_FX', 'GBP_EQ', 'USD_EQ']

    dfd_1 = basket_performance(dfd, contracts, ret='XR', cry='CRY',
                               weight_meth='equal', lback_meth='xma', lback_periods=21, weights=None,
                               weight_xcat=None, max_weight=None, return_weights=False)

