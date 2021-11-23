import numpy as np
import pandas as pd
from typing import List, Union, Tuple


def position_signals(df: pd.DataFrame, contracts: List[str],
                     sig: str, ret: str = 'XR_NSA', blacklist: dict = None,
                     start: str = None, end: str = None,
                     scale: str = 'zn', vtarg: float = 0.01,
                     lback_periods: int = 21, lback_meth: str = 'ma', half_life=11,
                     posname: str = 'POS'
                     ):

    """
    Converts contract-specific scores into signals for daily positioning

    :param <pd.Dataframe> df: standardized DataFrame with following columns: 'cid',
        'xcats', 'real_date' and 'value'.
    :param <List[str]> contracts: base tickers (combinations of cross sections and base
        categories) denoting contracts that go into the basket.
    :param <str> sig: signal category postfix; this is appended to the contracts to
        denote the tickers upon which position signals will be based.
        For example: 'CRYZN' is appended to 'EUR_FX' to give the ticker 'EUR_FXCRYZN'.
    :param <str> ret: return category postfix; default is "XR_NSA".
    :param <dict> blacklist: cross sections with date ranges that should be excluded
        from output.
    :param <str> start: earliest date in ISO format. Default is None and earliest date
        for which the signal category is available is used.
    :param <str> end: latest date in ISO format. Default is None and latest date
        for which the signal category is available is used.
    :param <str> scale: method to translate signal category into USD positions.
        [1] Default is 'zn', meaning that zn-scoring is applied based on the panel and
        that a 1 SD value translates into a USD1 position in the contract.
        [1] Method 'vt' means vol-targeting and implies that the individual position
        is set such that it meets a specific vol target based on recent historic
        volatility. The target as annualized SD in USD in the argument `vtarg`.
        This method supports a form of simple risk parity.
    :param <float> vtarg: annualized standard deviation target per position signal unit
        in USD. Default is 0.01 (1%).
    :param <int>  lback_periods: Number of lookback periods over which volatility is
        calculated. Default is 21.
    :param <str> lback_meth: Lookback method to calculate the volatility.
        Default is "ma". Alternative is "ema", exponential moving average.
    :param <int> half_life: Refers to the half-time for "xma". Default is 11.
    :param <str> posname: postfix added to contract to determine position name.

    :return <pd.Dataframe>: standardized dataframe with daily contract position signals
        in USD, using the columns 'cid', 'xcats', 'real_date' and 'value'.

    Note: A position signal is different from a position in two principal ways. First,
        The position signal can only be implemented with some lag. Second, the actual
        position of the strategy will be affected by other considerations, such as
        risk management and assets under management.
    """

    pass