
import numpy as np
import pandas as pd
from typing import List
from macrosynergy.panel.make_zn_scores import *
from macrosynergy.management.shape_dfs import reduce_df
from macrosynergy.panel.historic_vol import historic_vol
from macrosynergy.management.simulate_quantamental_data import make_qdf
import random

def preliminary_position(dfd_reduce: pd.DataFrame, cids: List[str], xcat_sig: str,
                         blacklist: dict = None, start: str = None, end: str = None,
                         scale: str = 'prop'):
    """
    Function designed to establish the dollar position, according to the "scale"
    parameter, prior to any volatility targeting.

    :param <pd.Dataframe> dfd_reduce: standardized DataFrame containing the following
        columns: 'cid', 'xcats', 'real_date' and 'value'.
    :param <List[str]> cids: cross sections of markets or currency areas in which
        positions should be taken.
    :param <str> xcat_sig: category that serves as signal across markets.
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

    """

    if scale == 'prop':

        # Rolling signal: zn-score computed for each of the cross-sections.
        # The dimensionality of the returned dataframe will match the dataframe
        # received.
        # Zn-score acts as a one-for-one dollar conversion for position in the asset.

        # Requires understanding the neutral level to use. Building out some of the below
        # parameters into the main signature.
        df_signal = make_zn_scores(dfd_reduce, xcat=xcat_sig, sequential=True, cids=cids,
                                   neutral='mean', pan_weight=0)

    # [2] Method 'dig' means 'digital' and sets the individual position to either USD1
    # Long or Short, depending on the sign of the signal.
    else:
        # One for long, -1 for short.
        # Reduce the DataFrame to the signal: singular xcat defined over the respective
        # cross-sections.
        df_signal = reduce_df(df=dfd_reduce, xcats=[xcat_sig], cids=cids, start=start,
                              end=end, blacklist=blacklist)

        df_signal['value'] = (df_signal['value'] > 0).astype(dtype=np.uint8)

        df_signal['value'] = df_signal['value'].replace(to_replace=0, value=-1)

    return df_signal

def target_positions(df: pd.DataFrame, cids: List[str], xcats: List[str], xcat_sig: str,
                     ctypes: List[str], sigrels: List[float], baskets: List[str] = None,
                     ret: str = 'XR_NSA', blacklist: dict = None,
                     start: str = None, end: str = None,
                     scale: str = 'prop', vtarg: float = 0.01,
                     lback_periods: int = 21, lback_meth: str = 'ma', half_life=11,
                     signame: str = 'POS'):

    """
    Converts signals into contract-specific target positions

    :param <pd.Dataframe> df: standardized DataFrame containing the following columns:
        'cid', 'xcats', 'real_date' and 'value'.
    :param <List[str]> cids: cross sections of markets or currency areas in which
        positions should be taken.
    :param <List[str]> xcats: the categories the standardised dataframe is defined over.
        Will require the (ctypes + ret) for volatility targeting.
    :param <str> xcat_sig: category that serves as signal across markets.
    :param <List[str]> ctypes: contract types that are traded across markets. They should
        correspond to return tickers. Examples are 'FX' or 'EQ'.
    :param <List[str]> baskets: cross section and contract types that denotes a basket
        that is traded in accordance with all cross section signals, for example as a
        benchmark for relative positions. A basket has the form 'cid'_'ctype', where
        cid could be 'GLB' for a global basket.
    :param <List[float]> sigrels: values that translate the single signal into contract
        type and basket signals in the order defined by ctypes + baskets.
    :param <str> ret: postfix denoting the returns applied to the contract types.
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
        in USD, using the columns 'cid', 'xcat', 'real_date' and 'value'.

    Note: A signal is still different from a position in two principal ways. First,
          the position signal can only be implemented with some lag. Second, the actual
          position of the strategy will be affected by other considerations, such as
          risk management and assets under management.
    """

    assert xcat_sig in set(df['xcat'].unique()), "Signal category missing from the /" \
                                                 "standardised dataframe."
    assert len(ctypes) == len(sigrels)
    assert scale in ['prop', 'dig']

    dfd = reduce_df(df=df, xcats=xcats, cids=cids, start=start, end=end,
                    blacklist=blacklist)

    df_signal = preliminary_position(dfd_reduce=dfd, cids=cids, xcat_sig=xcat_sig,
                                     blacklist=blacklist, start=start, end=end,
                                     scale=scale)

    assert isinstance(vtarg, float), "Volatility Target is a numerical value."
    assert df_signal.shape[1] == 4, "Incorrect dataframe."

    # xcat_sig = 'FXXR_NSA'
    # Example: ctypes = ['FX', 'EQ']; sigrels = [1, -1]; ret = 'XR_NSA'
    contract_returns = [c + ret for c in ctypes]

    # Extract the returns: (FXXR_NSA * 1 + EQXR_NSA * -1)

    # The "start" & "end" parameters ensure both series are defined over the same time-
    # period.
    for i, c_ret in enumerate(contract_returns):
        dfd_c_ret = dfd[dfd['xcat'] == c_ret]
        dfd_c_ret = dfd_c_ret.pivot(index="real_date", columns="cid", values="value")

        dfd_c_ret = dfd_c_ret.sort_index(axis=1)
        dfd_c_ret *= sigrels[i]
        if i == 0:
            dfd_c_rets = dfd_c_ret.copy()
        else:
            dfd_c_rets += dfd_c_ret

    # Split the "portfolio" of returns up across the cross-sections held in the panel.
    dfd_stack = dfd_c_rets.stack().to_frame("value").reset_index()
    dfd_stack['xcat'] = ret

    # Evolving volatility. The function historic_vol() will isolate the df on the
    # respective category passed. Returns a standardised DataFrame.
    df_vol = historic_vol(dfd_stack, xcat=ret, cids=cids,
                          lback_periods=lback_periods, lback_meth=lback_meth,
                          half_life=half_life, start=start, end=end,
                          blacklist=blacklist, remove_zeros=True, postfix="")

    dfw_vol = df_vol.pivot(index="real_date", columns="cid", values="value")
    dfw_vol = dfw_vol.sort_index(axis=1)
    # Adjust the position according to the volatility target.
    vol_ratio = vtarg / dfw_vol

    data_frames = []
    # Number of contracts the signal determines the position for.
    for i, sigrel in enumerate(sigrels):
        df_signal_copy = df_signal.copy()

        df_signal_copy['value'] *= sigrel

        df_signal_pivot = df_signal_copy.pivot(index="real_date", columns="cid",
                                               values="value")
        df_signal_pivot = df_signal_pivot.sort_index(axis=1)

        # Equation: (target_vol / vol_returns) * position
        # Adjust the position according to the volatility target.
        # The volatility is calculated using the "portfolio" returns (all ctypes).
        # Is the volatility adjustment applied to all potential positions in the
        # contract ?
        df_signal_vol = df_signal_pivot.multiply(vol_ratio)

        # Pivot back.
        dfd_stack = df_signal_vol.stack().to_frame("value").reset_index()

        dfd_stack['xcat'] = contract_returns[i]
        data_frames.append(dfd_stack)

    df_agg = pd.concat(data_frames, axis=0, ignore_index=True)

    # A 1 SD value translates into a USD1 position in the contract. The zn-score
    # equates to a one-for-one dollar position.
    # The equation means the previously computed position can be disregarded.

    columns = ['cid', 'xcat', 'real_date', 'value']
    df_agg['xcat'] += '_' + signame

    df_agg = df_agg[columns]

    return df_agg


if __name__ == "__main__":

    cids = ['AUD', 'GBP', 'NZD', 'USD']

    xcats = ['FXXR_NSA', 'EQXR_NSA']

    df_cids = pd.DataFrame(index=cids, columns=['earliest', 'latest', 'mean_add',
                                                'sd_mult'])

    df_cids.loc['AUD'] = ['2010-01-01', '2020-12-31', 0, 1]
    df_cids.loc['GBP'] = ['2011-01-01', '2020-11-30', 0, 2]
    df_cids.loc['NZD'] = ['2012-01-01', '2020-12-31', 0, 3]
    df_cids.loc['USD'] = ['2013-01-01', '2020-12-31', 0, 4]

    df_xcats = pd.DataFrame(index=xcats, columns=['earliest', 'latest', 'mean_add',
                                                  'sd_mult', 'ar_coef', 'back_coef'])

    df_xcats.loc['FXXR_NSA'] = ['2010-01-01', '2020-12-31', 0, 1, 0, 0.2]
    df_xcats.loc['EQXR_NSA'] = ['2012-01-01', '2020-10-30', 0.5, 2, 0, 0.2]

    random.seed(2)
    dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)
    black = {'AUD': ['2000-01-01', '2003-12-31'], 'GBP': ['2018-01-01', '2100-01-01']}

    # xcat_sig = 'FXXR_NSA'
    # Example: ctypes = ['FX', 'EQ']; sigrels = [1, -1]; ret = 'XR_NSA'
    # A single category to determine the position on potentially multiple contracts.
    # The relevant volatility for the volatility adjustment would be the combined returns
    # of each contract. In the below instance, the combined returns of
    # (FXXR_NSA + EQXR_NSA) will be used to determine the evolving volatility.
    position_df = target_positions(df=dfd, cids=cids, xcats=xcats, xcat_sig='FXXR_NSA',
                                   ctypes=['FX', 'EQ'], sigrels=[1, -1],
                                   ret='XR_NSA', blacklist=black, start='2012-01-01',
                                   end='2020-10-30', scale='prop',
                                   vtarg=0.1, signame='POS')

    print(position_df)

    position_df = target_positions(df=dfd, cids=cids, xcats=xcats, xcat_sig='FXXR_NSA',
                                   ctypes=['FX'], sigrels=[1], ret='XR_NSA',
                                   blacklist=black, start='2012-01-01', end='2020-10-30',
                                   scale='dig', vtarg=0.1, signame='POS')

    print(position_df)