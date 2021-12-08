import numpy as np
import pandas as pd
from typing import List, Union, Tuple
from random import random
from macrosynergy.management.simulate_quantamental_data import make_qdf
from macrosynergy.management.shape_dfs import reduce_df_by_ticker


def naive_pnls(df: pd.DataFrame, contracts: List[str],
               psigs: List[str], ret: str = 'XR_NSA', lag: float = 1,
               start: str = None, end: str = None,
               hindsight_vol: float = None,
               contract_pnls: bool = False
               ):

    """
    Convert one or more position signals into naive PnLs

    :param <pd.Dataframe> df: standardized DataFrame with following columns: 'cid',
        'xcats', 'real_date' and 'value'.
    :param <List[str]> contracts: base tickers (combinations of cross sections and base
        categories) denoting contracts that go into the basket.
    :param <List[str]> psigs: one or more positioning signal postfixes that are appended
        to the contract base tickers to identify their positioning signal
        For example: 'POS1' is appended to 'EUR_FX' to give the ticker 'EUR_FXPOS1'.
    :param <str> ret: return category postfix; default is "XR_NSA".
    :param <str> start: earliest date in ISO format. Default is None and earliest date
        for which the signal category is available is used.
    :param <str> end: latest date in ISO format. Default is None and latest date
        for which the signal category is available is used.
    :param <float> hindsight_vol: if a number x is provided that all portfolio PnLs are
        scaled to give an annualized USD standard deviation of that value.
        This is useful for visual comparison of strategies with different scales.
    :param <bool> contract_pnls: if True Pnls in USD for all contracts are added to the
        output dataframe.

    :return <pd.Dataframe>: standardized dataframe with daily PnLs for the overall
        portfolio and (possibly) the individual contracts.
        in USD, using the columns 'cid', 'xcats', 'real_date' and 'value'.

    Note: A position signal is different from a position in two principal ways. First,
          The position signal can only be implemented with some lag. Second, the actual
          position of the strategy will be affected by other considerations, such as
          risk management and assets under management.
    """

    assert all([isinstance(psig, str) for psig in psigs])

    psig_contracts = [c + psig for c in contracts for psig in psigs]

    ticks_ret = [c + ret for c in contracts]
    tickers = ticks_ret

    dfx = reduce_df_by_ticker(df, start=start, end=end, ticks=tickers,
                              blacklist=blacklist)


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

    psigs = ['POS1', 'POS2']

    naive_pnls(dfd, contracts=contracts, psigs=psigs, ret='XR_NSA',
               lag=1, start=None, end=None, hindsight_vol=None,
               contract_pnls=False)