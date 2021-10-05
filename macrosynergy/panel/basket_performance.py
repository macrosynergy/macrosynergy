import numpy as np
import pandas as pd
from typing import List, Union, Tuple
from macrosynergy.management.simulate_quantamental_data import make_qdf
from macrosynergy.management.shape_dfs import reduce_df


def basket_performance(df: pd.DataFrame, tiks: List[str], ret: str = 'XR_NSA', cry: str = 'CRY_NSA',
                       weight_meth: str = 'equal', lback_meth: str = 'xma', lback_periods: int = 21,
                       weights: List[float] = None, weight_xcat: str = None, max_weight: float = None,
                       basket_tik: str = 'GLB_ALL', return_weights: bool = False):
    """
    Computes approximate return and carry series for a basket of underlying contracts

    :param <pd.Dataframe> df: standardized data frame with the following necessary columns:
        'cid', 'xcats', 'real_date' and 'value.
    :param <List[str]> tiks: list of tickers (combinations of cross sections and base categories) that denote contracts
        that go into the basket. Base category means that the strings exclude the return/ carry postfix.
    :param <str> ret: return catgory postfix; default is "XR_NSA".
    :param <str> cry: return catgory postfix; default is "CRY_NSA".
    :param <str> weight_meth: method for weighting constituent returns and carry. Default is "equal", which means
        all constituents have the same weight, which weights are just the inverse of the available cross sections
        at each point in time. Alternatives are:
        "invsd" (inverse of past return standard deviations),
        "fixed" (proportionate to vector values supplied separately),
        "values" (proportionate to a panel of values of another category), and
        "inv_values" (inversely proportionate to proportionate to a panel of exogenous values supplied separately).
    :param <str> lback_meth: lookback method for "invsd" method. Default is "xma" (exponential moving average).
        Alternative is "ma", simple moving average.
    :param <int> lback_periods: lookback periods. Default is 21. Refers to half-time for "xma" and full lookback period
        for "ma".
    :param <List[float]> weights: list of weights corresponding to the tickers. Only relevant if weight_meth = 'fixed'
        is chosen. Default is None, which just means that the 'equal' method is applied.
    :param <str> weight_xcat: extended catgeory name of values used for "values" and "inv_values" methods.
    :param <float> max_weight: maximum weight permitted for a single cross section. Default is None.
        If chosen and binding, excess weights are redistributed until the condition is satified.
        If maximum weight is set below equal weight for available cross-sections, equal weights are chosen.
    :param <str> basket_tik: name of basket base ticker for which return and (possibly) carry are calculated.
        Default is "GLB_ALL".
    :param <bool> return_weights: add cross-section weights to output dataframe uses tiks with 'WGT' postfix.
        Default is False.

    :return <pd.Dataframe>: standardized dataframe with the basket performance data in standard form:
        'cid', 'xcat', 'real_date' and 'value'.
    """

    pass


if __name__ == "__main__":

    cids = ['AUD', 'CAD', 'GBP', 'USD', 'NZD']
    xcats = ['XR', 'CRY', 'GROWTH', 'INFL']

    df_cids = pd.DataFrame(index = cids, columns = ['earliest', 'latest', 'mean_add', 'sd_mult'])

    df_cids.loc['AUD'] = ['2010-01-01', '2020-12-31', 0.5, 2]
    df_cids.loc['CAD'] = ['2011-01-01', '2020-11-30', 0, 1]
    df_cids.loc['GBP'] = ['2012-01-01', '2020-11-30', -0.2, 0.5]
    df_cids.loc['USD'] = ['2013-01-01', '2020-09-30', -0.2, 0.5]
    df_cids.loc['NZD'] = ['2002-01-01', '2020-09-30', -0.1, 2]

    df_xcats = pd.DataFrame(index = xcats, columns = ['earliest', 'latest', 'mean_add', 'sd_mult', 'ar_coef', 'back_coef'])
    df_xcats.loc['XR'] = ['2010-01-01', '2020-12-31', 0, 1, 0, 0.3]
    df_xcats.loc['CRY'] = ['2011-01-01', '2020-10-30', 1, 2, 0.9, 0.5]
    df_xcats.loc['GROWTH'] = ['2012-01-01', '2020-10-30', 1, 2, 0.9, 1]
    df_xcats.loc['INFL'] = ['2013-01-01', '2020-10-30', 1, 2, 0.8, 0.5]