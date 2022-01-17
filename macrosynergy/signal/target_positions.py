
import numpy as np
import pandas as pd
from typing import List
from macrosynergy.panel.make_zn_scores import *
from macrosynergy.management.shape_dfs import reduce_df
from macrosynergy.panel.historic_vol import historic_vol
from macrosynergy.management.simulate_quantamental_data import make_qdf
import random


def unit_positions(df: pd.DataFrame, cids: List[str], xcat_sig: str,
                   blacklist: dict = None, start: str = None, end: str = None,
                   scale: str = 'prop', min_obs: int = 252, thresh: float = None):

    """
    Calculate unit positions from signals based on zn-scoring (proportionate method)
    or conversion to signs (digital method).

    :param <pd.Dataframe> df: standardized DataFrame containing the following columns:
        'cid', 'xcats', 'real_date' and 'value'.
    :param <List[str]> cids: cross sections of markets or currency areas in which
        positions should be taken.
    :param <str> xcat_sig: category that serves as signal across markets.
    :param <dict> blacklist: cross sectional date ranges that should have zero target
        positions.  # Todo: check if redundant in this context
    :param <str> start: earliest date in ISO format. Default is None and earliest date
        for which the signal category is available is used.
    :param <str> end: latest date in ISO format. Default is None and latest date
        for which the signal category is available is used.
    :param <str> scale: method to translate signals into target positions:
        [1] Default is 'prop', means proportionate. In this case zn-scoring is applied
            to the signal based on the panel, with the neutral level set at zero.
             A 1 SD value translates into a USD1 position in the contract.
        [2] Method 'dig' means 'digital' and sets the individual position to either USD1
            long or short, depending on the sign of the signal.
    :param <int> min_obs: the minimum number of observations required to calculate
        zn_scores. Default is 252.
    :param <float> thresh: threshold value beyond which zn-scores for propotionate
        position taking are winsorized. The threshold is the maximum absolute
        score value in standard deviations. The minimum is 1 standard deviation.

    :return <pd.Dataframe>: standardized dataframe, of the signal category, with the
        respective computed position, using the columns 'cid', 'xcat', 'real_date' and
        'value'.

    """

    options = ['prop', 'dig']
    assert scale in options, f"The scale parameter must be either {options}"

    if scale == 'prop':

        assert isinstance(min_obs, int), \
            "Minimum observation parameter must be an integer."
        df_up = make_zn_scores(df, xcat=xcat_sig, blacklist=blacklist,
                               sequential=True, cids=cids, start=start, end=end,
                               neutral='zero', pan_weight=1, min_obs=min_obs,
                               thresh=thresh)
    else:

        df_up = reduce_df(df=df, xcats=[xcat_sig], cids=cids, start=start, end=end,
                          blacklist=blacklist)
        df_up['value'] = np.sign(df_up['value'])

    return df_up


def start_end(df: pd.DataFrame, contract_returns: List[str]):
    """
    Determines the time-period over which each contract is defined.

    :param <pd.Dataframe> df: standardized DataFrame containing the following columns:
        'cid', 'xcats', 'real_date' and 'value'.
    :param <List[str]> contract_returns: list of the contract return types.

    :return <dict>: dictionary where the key is the contract and the value is a tuple
        of the start & end date.
    """

    start_end_dates = {}
    for i, c_ret in enumerate(contract_returns):

        df_c_ret = df[df['xcat'] == c_ret]
        df_c_ret = df_c_ret.pivot(index="real_date", columns="cid", values="value")
        index = df_c_ret.index
        start_end_dates[c_ret] = (index[0], index[-1])

    return start_end_dates


def composite_returns(df: pd.DataFrame, xcat_sig: str, contract_returns: List[str],
                      sigrels: List[str], time_index: pd.Series, cids: List[str],
                      ret: str = 'XR_NSA'):
    """
    Calculate returns of composite positions (that jointly depend on one signal).

    :param <pd.Dataframe> df: standardized DataFrame containing the following columns:
        'cid', 'xcats', 'real_date' and 'value'.
    :param <str> xcat_sig: category that serves as signal across markets.
    :param <List[str]> contract_returns: list of the contract return types.
    :param <List[str]> sigrels: respective signal for each contract type.
    :param <pd.Series> time_index: datetime index for which signals are available
        # Todo: remove argument
    :param <List[str]> cids: cross-sections of markets or currency areas in which
        positions should be taken.
    :param <str> ret: postfix denoting the returns in % applied to the contract types.

    :return <pd.Dataframe>: standardized dataframe with the summed portfolio returns
        which are used to calculate the evolving volatility, using the columns 'cid',
        'xcat', 'real_date' and 'value'.

    """

    assert len(contract_returns) == len(sigrels), \
        "Each individual contract requires an associated signal."

    cids = sorted(cids)
    data = np.zeros(shape=(time_index.size, len(cids)))
    # The signal will delimit the longevity of the possible position.  # Todo No!
    # Todo: composite returns should be calculate for all available dates
    framework = pd.DataFrame(data=data, columns=cids, index=time_index)
    framework.columns.name = 'cid'

    df_c_sig = df[df['xcat'] == xcat_sig]
    df_c_rets = df_c_sig.pivot(index="real_date", columns="cid", values="value")

    # Todo: pack next four lines into loop to save code
    sigrels = iter(sigrels)
    df_c_rets *= next(sigrels)

    c_returns_copy = contract_returns.copy()
    c_returns_copy.remove(xcat_sig)

    signal_start = time_index[0]  # Todo: No! This should not be used
    signal_end = time_index[-1]

    for i, c_ret in enumerate(c_returns_copy):

        df_c_ret = df[df['xcat'] == c_ret]
        df_c_ret = df_c_ret.pivot(index="real_date", columns="cid", values="value")
        cat_index = df_c_ret.index
        cat_start = cat_index[0]
        cat_end = cat_index[-1]

        # Only concerned by the return series that are aligned to the signal.
        df_c_ret = df_c_ret.truncate(before=signal_start, after=signal_end)

        if cat_start > signal_start or cat_end < signal_end:
            condition_start = next(iter(np.where(time_index == cat_start)[0]))
            date_fill = framework.iloc[:condition_start]
            df_c_ret = pd.concat([date_fill, df_c_ret])

            condition_end = next(iter(np.where(time_index == cat_end)[0]))
            date_fill = framework.iloc[(condition_end + 1):]
            df_c_ret = pd.concat([df_c_ret, date_fill])
        else:
            pass

        df_c_ret = df_c_ret.sort_index(axis=1)
        df_c_ret *= next(sigrels)

        # Add each return series of the contract.
        df_c_rets += df_c_ret

    # Believe this operation is now redundant.
    df_c_rets.dropna(how='all', inplace=True)

    df_rets = df_c_rets.stack().to_frame("value").reset_index()
    df_rets['xcat'] = ret

    return df_rets


def target_positions(df: pd.DataFrame, cids: List[str], xcats: List[str], xcat_sig: str,
                     ctypes: List[str], sigrels: List[float],
                     baskets: List[str] = None,
                     ret: str = 'XR_NSA', blacklist: dict = None, start: str = None,
                     end: str = None, scale: str = 'prop',
                     min_obs: int = 252,
                     thresh: float = None, vtarg: float = None, lback_periods: int = 21,
                     lback_meth: str = 'ma', half_life: int = 11, signame: str = 'POS'):

    """
    Converts signals into contract-specific target positions

    :param <pd.Dataframe> df: standardized DataFrame containing the following columns:
        'cid', 'xcats', 'real_date' and 'value'.
    :param <List[str]> cids: cross sections of markets or currency areas in which
        positions should be taken.
    :param <List[str]> xcats: the categories (signals amd position returns) the
        standardised dataframe is defined over.
        Must include the (ctypes + ret) for volatility targeting.
        # Todo: check if this can be made implicit
    :param <str> xcat_sig: category that serves as signal across markets.
    :param <List[str]> ctypes: contract types that are traded across markets. They should
        correspond to return tickers. Examples are 'FX' or 'EQ'.
    :param <List[str]> baskets: cross section and contract types that denotes a basket
        that is traded in accordance with all cross section signals, for example as a
        benchmark for relative positions. A basket has the form 'cid'_'ctype', where
        cid could be 'GLB' for a global basket.
    :param <List[float]> sigrels: values that translate the single signal into contract
        type and basket signals in the order defined by ctypes + baskets.
    :param <str> ret: postfix denoting the returns in % applied to the contract types.
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
            to the signal based on the panel, with the neutral level set at zero.
            A 1 SD value translates into a USD1 position in the contract.
        [2] Method 'dig' means 'digital' and sets the individual position to either USD1
            long or short, depending on the sign of the signal.
    :param <int> min_obs: the minimum number of observations required to calculate
        zn_scores. Default is 252.
    :param <float> thresh: threshold value beyond which zn-scores for propotionate
        position taking are winsorized. The threshold is the maximum absolute
        score value in standard deviations. The minimum is 1 standard deviation.
    :param <float> vtarg: This allows volatility targeting on the contract level.
        Default is None, but if a value is chosen then for each contract the
        proportionate or digital position is translated into a position that carries
        a historic return standard deviation equal to the value given. For example, 10
        means that the target position carries a recent historical annualized standard
        deviation of 10 dollars (or other currency units).
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

    # A. Initial checks

    assert xcat_sig in set(df['xcat'].unique()), \
        "Signal category missing from the standardised dataframe."
    assert isinstance(vtarg, (float, int)) or (vtarg is None) \
        and not isinstance(vtarg, bool), \
        "Volatility Target must be numeric or None."
    assert len(sigrels) == len(ctypes), \
        "The number of signals correspond to the number of contracts defined in ctypes."

    cols = ['cid', 'xcat', 'real_date', 'value']
    assert set(cols) <= set(df.columns), f"df columns must contain {cols}."

    # B. Get unit position information

    df = df.loc[:, cols]
    dfx = reduce_df(df=df, xcats=xcats, cids=cids, start=start, end=end,
                    blacklist=None)  # Todo: blacklist must be applied at end

    df_upos = unit_positions(df=dfx, cids=cids, xcat_sig=xcat_sig,
                             blacklist=blacklist, start=start, end=end,
                             scale=scale, thresh=thresh)  # zn-scores or signs

    df_upos_w = df_upos.pivot(index="real_date", columns="cid", values="value")
    # N.B.: index is determined by the longest cross-section.
    time_index = df_upos_w.index  # Todo: return lookback killer - should not be used

    contract_returns = [c + ret for c in ctypes]
    start_end_dates = start_end(dfx, contract_returns)  # get return types' starts/ends

    # C. Volatility targeting

    if isinstance(vtarg, (int, float)):

        # C.1. Composite signal-related positions as basis for volatility targeting

        df_crets = composite_returns(dfx, xcat_sig, contract_returns, sigrels,
                                     time_index, cids)
        df_crets = df_crets[cols]

        # C.2. Calculate volatility adjustment ratios

        df_vol = historic_vol(df_crets, xcat=ret, cids=cids,
                              lback_periods=lback_periods, lback_meth=lback_meth,
                              half_life=half_life, start=start, end=end,
                              blacklist=blacklist, remove_zeros=True, postfix="")
        # Todo: The dimensions may not match signals

        dfw_vol = df_vol.pivot(index="real_date", columns="cid", values="value")
        dfw_vol = dfw_vol.sort_index(axis=1)
        dfw_vtr = 100 * vtarg / dfw_vol  # vol-target ratio to be applied

        # C.3. Calculated vol-targeted positions

        data_frames = []
        for i, sigrel in enumerate(sigrels):

            df_pos = df_upos.copy()
            df_pos['value'] *= sigrel
            dfw_pos = df_pos.pivot(index="real_date", columns="cid",
                                   values="value")

            # Only able to take a position in the contract for the duration in which it
            # is defined.
            tuple_dates = start_end_dates[contract_returns[i]]
            start_date = tuple_dates[0]
            end_date = tuple_dates[1]

            # Truncate the signal dataframe to reflect the length of time of the specific
            # contract.
            dfw_pos = dfw_pos.truncate(before=start_date, after=end_date)

            dfw_pos = dfw_pos.sort_index(axis=1)
            # Applicable volatility will be applied: depending on the timeframes of each
            # contract.

            # NaNs to account for the lookback period. The position dataframe, through
            # each iteration, has been reduced to match the respective input's
            # dimensions.
            dfw_pos_vt = dfw_pos.multiply(dfw_vtr)
            dfw_pos_vt.dropna(how='all', inplace=True)

            df_crets = dfw_pos_vt.stack().to_frame("value").reset_index()
            df_crets['xcat'] = contract_returns[i]
            data_frames.append(df_crets)

        df_tpos = pd.concat(data_frames, axis=0, ignore_index=True)

    else:

        df_concat = []
        for i, elem in enumerate(contract_returns):
            # Instantiate a new copy through each iteration.
            df_upos_copy = df_upos_w.copy()

            tuple_dates = start_end_dates[elem]
            start_date = tuple_dates[0]
            end_date = tuple_dates[1]

            df_upos_copy = df_upos_copy.truncate(before=start_date,
                                                         after=end_date)

            df_upos_copy *= sigrels[i]
            # The current category, defined on the dataframe, is the signal category.
            # But the signal is being used to take a position in multiple contracts.
            # according to the long-short definition. The returned dataframe should be
            # inclusive of all the contracts.
            df_upos = df_upos_copy.stack().to_frame("value").reset_index()

            df_upos['xcat'] = elem

            df_concat.append(df_upos)

        df_tpos = pd.concat(df_concat, axis=0, ignore_index=True)

    df_tpos['xcat'] += '_' + signame
    df_tpos['xcat'] = df_tpos['cid'] + '_' + df_tpos['xcat']
    df_tpos = df_tpos[cols]

    return df_tpos


if __name__ == "__main__":

    cids = ['AUD', 'GBP', 'NZD', 'USD']
    xcats = ['FXXR_NSA', 'EQXR_NSA', 'SIG_NSA']

    ccols = ['earliest', 'latest', 'mean_add', 'sd_mult']
    df_cids = pd.DataFrame(index=cids, columns=ccols)
    df_cids.loc['AUD'] = ['2010-01-01', '2020-12-31', 0, 1]
    df_cids.loc['GBP'] = ['2011-01-01', '2020-11-30', 0, 2]
    df_cids.loc['NZD'] = ['2012-01-01', '2020-12-31', 0, 3]
    df_cids.loc['USD'] = ['2013-01-01', '2020-12-31', 0, 4]

    xcols = ccols + ['ar_coef', 'back_coef']
    df_xcats = pd.DataFrame(index=xcats, columns=xcols)
    df_xcats.loc['FXXR_NSA'] = ['2010-01-01', '2020-12-31', 0, 1, 0, 0.2]
    df_xcats.loc['EQXR_NSA'] = ['2012-01-01', '2020-10-30', 0.5, 2, 0, 0.2]
    df_xcats.loc['SIG_NSA'] = ['2010-01-01', '2020-12-3', 0, 10, 0.4, 0.2]

    random.seed(2)
    dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)
    black = {'AUD': ['2000-01-01', '2003-12-31'], 'GBP': ['2018-01-01', '2100-01-01']}

    xcat_sig = 'FXXR_NSA'

    position_df = target_positions(df=dfd, cids=cids,
                                   xcats=['FXXR_NSA', 'EQXR_NSA', 'SIG_NSA'],
                                   xcat_sig='SIG_NSA',
                                   ctypes=['FX', 'EQ'], sigrels=[1, 0.5], ret='XR_NSA',
                                   blacklist=black, start='2012-01-01', end='2020-10-30',
                                   scale='dig', vtarg=5, signame='POS')

    # print(position_df)
    #
    # position_df = target_positions(df=dfd, cids=cids, xcats=xcats, xcat_sig='FXXR_NSA',
    #                                ctypes=['FX', 'EQ'], sigrels=[1, -1], ret='XR_NSA',
    #                                blacklist=black, start='2012-01-01', end='2020-10-30',
    #                                scale='dig', vtarg=0.1, signame='POS')
    #
    # print(position_df)
    #
    # # The secondary contract, EQXR_NSA, is defined over a shorter timeframe. Therefore,
    # # on the additional dates, a valid position will be computed using the signal
    # # category but a position will not be able to be taken for EQXR_NSA.
    # position_df = target_positions(df=dfd, cids=cids, xcats=xcats, xcat_sig='FXXR_NSA',
    #                                ctypes=['FX', 'EQ'], sigrels=[1, -1], ret='XR_NSA',
    #                                blacklist=black, start='2010-01-01', end='2020-12-31',
    #                                scale='prop', vtarg=None, signame='POS')
    #
    # print(position_df)