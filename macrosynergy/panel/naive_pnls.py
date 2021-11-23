import numpy as np
import pandas as pd
from typing import List, Union, Tuple


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

    pass