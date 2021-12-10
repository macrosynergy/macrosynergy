import numpy as np
import pandas as pd
from typing import List, Union, Tuple


def target_positions(df: pd.DataFrame, contracts: List[str],
                     cat: str, ret: str = 'XR_NSA', blacklist: dict = None,
                     start: str = None, end: str = None,
                     scale: str = 'zn', vtarg: float = 0.01,
                     lback_periods: int = 21, lback_meth: str = 'ma', half_life=11,
                     signame: str = 'POS'
                     ):

    """
    Converts signals into contract-specific target positions

    :param <pd.Dataframe> df: standardized DataFrame containing the following columns:
        'cid', 'xcats', 'real_date' and 'value'.
    :param <List[str]> contracts: base tickers (combinations of cross sections and base
        categories) for which target positions are to be generated.
    :param <str> sig: category postfix that is appended to the contracts in order to
        obtain the tickers that serve as signals.
        For example: 'SIG' is appended to 'EUR_FX' to give the ticker 'EUR_FXSIG'
    :param <str> ret: return category postfix; default is "XR_NSA".
        The returns are necessary for volatility target-based signals.
    :param <dict> blacklist: cross sectional date ranges that should have zero target
        positions.
        This is a standardized dictionary with cross sections as keys and tuples of
        start and end dates of the blacklist periods in ISO formats as values.
        If one cross section has multiple blacklist periods, numbers are added to the
        keys (i.e. TRY_1, TRY_2, etc.)
    :param <str> start: earliest date in ISO format. Default is None and earliest date
        for which the signal category is available is used.
    :param <str> end: latest date in ISO format. Default is None and latest date
        for which the signal category is available is used.
    :param <str> scale: method to translate signals into target positions:
        [1] Default is 'prop', means proportionate. In this case zn-scoring is applied
            to the signal based on the panel and a 1 SD value translates into a
            USD1 position in the contract.
        [2] Method 'dig' means 'digital' and sets the individual position to either USD1
            long or short, depending on the sign of the signal.
        [2] Method 'vt' means vol-targeting and implies that the individual position
        is set such that it meets a specific vol target based on recent historic
        volatility. The target as annualized SD in USD in the argument `vtarg`.
        This method supports a form of simple risk parity.
    :param <float> vtarg: This allows volatility targeting on the contract level.
        Default is None, but if a value is chosen then for each contract the
        proportionate or digital position is translated into a position that carries
        a historic return standard deviation equal to the value given. For example, 0.1
        means that the target position carries a recent historical annualized standard
        deviation of 10% of 10 cents.
    :param <int>  lback_periods: Number of lookback periods over which volatility is
        calculated. Default is 21.
    :param <str> lback_meth: Lookback method to calculate the volatility.
        Default is "ma". Alternative is "ema", exponential moving average.
    :param <int> half_life: Refers to the half-time for "xma". Default is 11.
    :param <str> posname: postfix added to contract to denote signal name.

    :return <pd.Dataframe>: standardized dataframe with daily contract position signals
        in USD, using the columns 'cid', 'xcats', 'real_date' and 'value'.

    Note: A signal is still different from a position in two principal ways. First,
        The position signal can only be implemented with some lag. Second, the actual
        position of the strategy will be affected by other considerations, such as
        risk management and assets under management.
    """

    # Todo: this function draws heavily on make_zn_scores and historic_vol

    pass