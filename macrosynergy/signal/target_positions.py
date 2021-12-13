import numpy as np
import pandas as pd
from typing import List
from macrosynergy.panel.make_zn_scores import *
from macrosynergy.management.shape_dfs import reduce_df


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
    :param <str> signame: postfix added to contract to denote signal name.

    :return <pd.Dataframe>: standardized dataframe with daily contract position signals
        in USD, using the columns 'cid', 'xcats', 'real_date' and 'value'.

    Note: A signal is still different from a position in two principal ways. First,
          the position signal can only be implemented with some lag. Second, the actual
          position of the strategy will be affected by other considerations, such as
          risk management and assets under management.
    """

    # Todo: this function draws heavily on make_zn_scores and historic_vol
    ticks_vol_ret = [c + xcat_ret for c in cids]

    # :param <str> sig: category postfix that is appended to the contracts in order to
    #  obtain the tickers that serve as signals.
    #  For example: 'SIG' is appended to 'EUR_FX' to give the ticker 'EUR_FXSIG'.
    #  Is 'SIG' a contrived postfix ? What is the DQ equivalent ?
    ticks_sig = [c + ctype for c in cids for ctype in ctypes]

    # Requires understanding the neutral level to use. Building out some of the below
    # parameters into the main signature.
    if scale == 'prop':

        # Rolling signal: zn-score computed for each of the cross-sections.
        # The dimensionality of the returned dataframe will match the dataframe
        # received.
        prop_sig = make_zn_scores(df, xcat=xcat_sig, sequential=True, cids=cids,
                                  neutral='mean', pan_weight=0)

        # value = prop_sig.stack().to_frame("value").reset_index()
        # Isolate the value column.
        value = prop_sig['value']

    # [2] Method 'dig' means 'digital' and sets the individual position to either USD1
    # Long or Short, depending on the sign of the signal.
    elif scale == 'dig':
        # One for long, -1 for short.
        # Reduce the DataFrame to the signal: singular xcat defined over the respective
        # cross-sections.

        # Other categories have to be defined over the same respective time-period as the
        # signal category.

        xcats_unique = list(df['xcats'].unique())
        # Cross-sections must be the same for all categories: dependent on the signal
        # category.
        # Align the time-periods of the signal category and the other remaining
        # categories which represent the positions.
        # Aim to isolate the respective category. Pivot and utilise the corresponding
        # signal to determine which position to take in the remaining categories.
        df_scale = reduce_df(df=df, xcats=xcats_unique, cids=cids, start=start, end=end)

        # Pivot on the cross-sections. Each category should be defined over the same
        # set of cross-sections. Lift the signal metric and apply to that specific
        # category for every cross-section.

        df_scale_pivot = df_scale.pivot(index="real_date", columns="cids",
                                        values="value")

        long_short_df = df_scale_pivot[df_scale_pivot > 0].astype(dtype=np.uint8)

        # Pivoted dataframe with the respective cross-sections and the corresponding
        # signal per timestamp.
        long_short_df = long_short_df.replace(to_replace=0, value=1)

        dfsig = long_short_df.to_frame("value").reset_index()

    elif scale == 'vt':
        pass

