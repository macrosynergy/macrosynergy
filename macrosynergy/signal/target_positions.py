import numpy as np
import pandas as pd
from typing import List
from macrosynergy.panel.make_zn_scores import *
from macrosynergy.management.shape_dfs import reduce_df_by_ticker


def target_positions(df: pd.DataFrame, contracts: List[str],
                     xcat: str, ret: str = 'XR_NSA', blacklist: dict = None,
                     start: str = None, end: str = None,
                     scale: str = 'prop', vtarg: float = 0.01,
                     lback_periods: int = 21, lback_meth: str = 'ma', half_life=11,
                     signame: str = 'POS'
                     ):

    """
    Converts signals into contract-specific target positions

    :param <pd.Dataframe> df: standardized DataFrame containing the following columns:
        'cid', 'xcats', 'real_date' and 'value'.
    :param <List[str]> cids: cross sections of markets or currency areas in which
        positios should be taken.
    :param <str> xcat_sig: category that serves as signal across markets.
    :param <List[str]>: ctypes: contract types that are traded across markets. They should
        correspond to return tickers. Examples are 'FX' or 'EQ'.
    :param <List[str]> baskets: cross section and contract types that denotes a basket
        that is traded in accordance with all cross section signals, for example as a
        benchmark for relative positions. A basket has the form 'cid'_'ctype', where
        cid could be 'GLB' for a global basket.
    :param <List[float]> sigrels: values that translate the single signal into contract
        type and basket signals in the order defined by baskets + ctypes.
    :param <str> xcat_ret: category denoting the cross section-specific returns,
        which may be a combination of contracts.
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
          the position signal can only be implemented with some lag. Second, the actual
          position of the strategy will be affected by other considerations, such as
          risk management and assets under management.
    """

    # Todo: this function draws heavily on make_zn_scores and historic_vol
    ticks_ret = [c + ret for c in contracts]

    # :param <str> sig: category postfix that is appended to the contracts in order to
    #  obtain the tickers that serve as signals.
    #  For example: 'SIG' is appended to 'EUR_FX' to give the ticker 'EUR_FXSIG'.
    #  Is 'SIG' a contrived postfix ? What is the DQ equivalent ?
    ticks_sig = [c + 'SIG' for c in contracts]

    # 'EUR_FXSIG'.
    tickers = ticks_ret + ticks_sig
    dfd = reduce_df_by_ticker(df, start=start, end=end, ticks=tickers,
                              blacklist=blacklist)

    # Requires understanding the neutral level to use. Building out some of the below
    # parameters into the main signature.
    if scale == 'prop':
        # Rolling signal: zn-score computed for each of the cross-sections.

        prop_sig = make_zn_scores(dfd, xcat='SIG', sequential=True, cids=cids,
                                  neutral='mean', pan_weight=0)

    # [2] Method 'dig' means 'digital' and sets the individual position to either USD1
    #  Long or short, depending on the sign of the signal.
    elif scale == 'dig':
        # One for long, 0 for short.
        value = (dfd['value'] > 0).astype(dtype=np.uint8)

    elif scale == 'vt':
        pass

