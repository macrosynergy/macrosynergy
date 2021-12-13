import numpy as np
import pandas as pd
from typing import List
from macrosynergy.panel.make_zn_scores import *
from macrosynergy.management.shape_dfs import reduce_df
from macrosynergy.panel.historic_vol import historic_vol

# The standardised dataframe only consists of a single category: the signal category.
# Depending on the values held in the category, signal generation, take proportionate
# positions.
def target_positions(df: pd.DataFrame, cids: List[str], xcat_sig: str,
                     baskets: List[str], ctypes: List[str], sigrels: List[float],
                     xcat_ret: str = 'XR_NSA', blacklist: dict = None,
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
        positions should be taken.
    :param <str> xcat_sig: category that serves as signal across markets.
    :param <List[str]> ctypes: contract types that are traded across markets. They should
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
        [3] Method 'vt' means vol-targeting and implies that the individual position
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
    :param <str> signame: postfix added to contract to denote signal name.

    :return <pd.Dataframe>: standardized dataframe with daily contract position signals
        in USD, using the columns 'cid', 'xcats', 'real_date' and 'value'.

    Note: A signal is still different from a position in two principal ways. First,
          the position signal can only be implemented with some lag. Second, the actual
          position of the strategy will be affected by other considerations, such as
          risk management and assets under management.
    """

    assert xcat_sig in set(df['xcats'].unique()), "Signal category missing from the /" \
                                                  "standardised dataframe."

    # Requires understanding the neutral level to use. Building out some of the below
    # parameters into the main signature.
    if scale == 'prop':

        # Rolling signal: zn-score computed for each of the cross-sections.
        # The dimensionality of the returned dataframe will match the dataframe
        # received.
        # Standard Deviation column of zn-scores.
        df_signal = make_zn_scores(df, xcat=xcat_sig, sequential=True, cids=cids,
                                   neutral='mean', pan_weight=0)

    # [2] Method 'dig' means 'digital' and sets the individual position to either USD1
    # Long or Short, depending on the sign of the signal.
    elif scale == 'dig':
        # One for long, -1 for short.
        # Reduce the DataFrame to the signal: singular xcat defined over the respective
        # cross-sections.

        df_signal = reduce_df(df=df, xcats=xcat_sig, cids=cids, start=start, end=end,
                              blacklist=blacklist)
        df_signal['value'] = (df_signal['value'] > 0).astype(dtype=np.uint8)

        df_signal['value'] = df_signal['value'].replace(to_replace=0, value=1)

    elif scale == 'vt':
        assert isinstance(vtarg, float), "Volatility Target is a numerical value."

        df_signal = df.copy()
        # Returns of the signal category.
        df_signal = df_signal.sort_values(by=['cid'])

        # Evolving volatility.
        df_vol = historic_vol(df_signal, xcat=xcat_sig, cids=cids,
                              lback_periods=lback_periods, lback_meth=lback_meth,
                              half_life=half_life, start=start, end=end,
                              blacklist=blacklist, remove_zeros=True, postfix="vol")

        # A 1 SD value translates into a USD1 position in the contract. The zn-score
        # equates to a one-for-one dollar position.
        # The equation means the previously computed position can be disregarded.

        # Equation: position * vol_returns = target_vol
        df_vol_order = df_vol.sort_values(by=['cid'])

        position = vtarg / df_vol_order['value']

        df_signal['value'] = position

    return df_signal

