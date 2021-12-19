
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
    Establish the unitary position depending on the scaling factor. Will not adjust for
    any volatility targets.

    :param <pd.Dataframe> df: standardized DataFrame containing the following columns:
        'cid', 'xcats', 'real_date' and 'value'.
    :param <List[str]> cids: cross sections of markets or currency areas in which
        positions should be taken.
    :param <str> xcat_sig: category that serves as signal across markets.
    :param <dict> blacklist: cross sectional date ranges that should have zero target
        positions.
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
    if scale == 'prop':
        assert isinstance(min_obs, int), "Minimum observation parameter must be an " \
                                         "Integer."

    options = ['prop', 'dig']
    assert scale in options, f"The scale parameter must be either {options}"

    if scale == 'prop':

        df_unit_pos = make_zn_scores(df, xcat=xcat_sig, blacklist=blacklist,
                                     sequential=True, cids=cids, start=start, end=end,
                                     neutral='zero', pan_weight=1, min_obs=min_obs,
                                     thresh=thresh)
    else:

        df_unit_pos = reduce_df(df=df, xcats=[xcat_sig], cids=cids, start=start, end=end,
                                blacklist=blacklist)
        df_unit_pos['value'] = np.sign(df_unit_pos['value'])

    return df_unit_pos

def time_series(dfd: pd.DataFrame, contract_returns: List[str]):
    """
    Determines the time-period over which each contract is defined.

    :param <pd.Dataframe> dfd: standardized DataFrame containing the following columns:
        'cid', 'xcats', 'real_date' and 'value'.
    :param <List[str]> contract_returns: list of the contract return types.

    :return <dict>: dictionary where the key is the contract and the value is a tuple
        of the start & end date.
    """

    durations = {}
    for i, c_ret in enumerate(contract_returns):

        dfd_c_ret = dfd[dfd['xcat'] == c_ret]
        dfd_c_ret = dfd_c_ret.pivot(index="real_date", columns="cid", values="value")
        index = dfd_c_ret.index
        durations[c_ret] = (index[0], index[-1])

    return durations

def return_series(dfd: pd.DataFrame, xcat_sig: str, contract_returns: List[str],
                  sigrels: List[str], time_index: pd.Series, cids: List[str],
                  ret: str = 'XR_NSA'):
    """
    Compute the aggregated return-series, adjusting for the respective signal, for the
    portfolio of contracts: a single signal can be used to take positions in
    multiple contract types: EQ, FX. If the signal is defined over a longer time horizon
    than the other contracts, the portfolio return should converge exclusively to the
    signal category.

    :param <pd.Dataframe> dfd: standardized DataFrame containing the following columns:
        'cid', 'xcats', 'real_date' and 'value'.
    :param <str> xcat_sig: category that serves as signal across markets.
    :param <List[str]> contract_returns: list of the contract return types.
    :param <List[str]> sigrels: respective signal for each contract type.
    :param <pd.Series> time_index: timeframe of the signal category.
    :param <List[str]> cids: cross-sections of markets or currency areas in which
        positions should be taken.
    :param <str> ret: postfix denoting the returns in % applied to the contract types.

    :return <pd.Dataframe>: standardized dataframe with the summed portfolio returns
        which are used to calculate the evolving volatility, using the columns 'cid',
        'xcat', 'real_date' and 'value'.

    """

    assert len(contract_returns) == len(sigrels), "Each individual contract requires an " \
                                                  "associated signal."

    # The signal is defined over a specific time-period. Therefore, there is little point
    # calculating the return-series, for the portfolio, beyond the signal's timeframe
    # given a position will not subsequently be computed.
    cids = sorted(cids)
    data = np.zeros(shape=(time_index.size, len(cids)))
    # The signal will delimit the longevity of the possible position.
    framework = pd.DataFrame(data=data, columns=cids, index=time_index)
    framework.columns.name = 'cid'

    dfd_c_sig = dfd[dfd['xcat'] == xcat_sig]
    dfd_c_rets = dfd_c_sig.pivot(index="real_date", columns="cid", values="value")
    sigrels = iter(sigrels)
    dfd_c_rets *= next(sigrels)

    c_returns_copy = contract_returns.copy()
    c_returns_copy.remove(xcat_sig)

    signal_start = time_index[0]
    signal_end = time_index[-1]

    for i, c_ret in enumerate(c_returns_copy):

        dfd_c_ret = dfd[dfd['xcat'] == c_ret]
        dfd_c_ret = dfd_c_ret.pivot(index="real_date", columns="cid", values="value")
        cat_index = dfd_c_ret.index
        cat_start = cat_index[0]
        cat_end = cat_index[-1]

        # Only concerned by the return series that are aligned to the signal.
        dfd_c_ret = dfd_c_ret.truncate(before=signal_start, after=signal_end)

        if cat_start > signal_start or cat_end < signal_end:
            condition_start = next(iter(np.where(time_index == cat_start)[0]))
            date_fill = framework.iloc[:condition_start]
            dfd_c_ret = pd.concat([date_fill, dfd_c_ret])

            condition_end = next(iter(np.where(time_index == cat_end)[0]))
            date_fill = framework.iloc[(condition_end + 1):]
            dfd_c_ret = pd.concat([dfd_c_ret, date_fill])
        else:
            pass

        dfd_c_ret = dfd_c_ret.sort_index(axis=1)
        dfd_c_ret *= next(sigrels)

        # Add each return series of the contract.
        dfd_c_rets += dfd_c_ret

    # The number of active timestamps, dimensions of the dataframe, will be determined by
    # the signal category. If the signal category is defined over the longer timeframe,
    # the current design allows the "portfolio" return series to still be computed but
    # exclusively using the signal's return.

    # Believe this operation is now redundant.
    dfd_c_rets.dropna(how='all', inplace=True)

    df_pos_vt = dfd_c_rets.stack().to_frame("value").reset_index()
    df_pos_vt['xcat'] = ret

    return df_pos_vt

def target_positions(df: pd.DataFrame, cids: List[str], xcats: List[str], xcat_sig: str,
                     ctypes: List[str], sigrels: List[float], baskets: List[str] = None,
                     ret: str = 'XR_NSA', blacklist: dict = None, start: str = None,
                     end: str = None, scale: str = 'prop', min_obs: int = 252,
                     thresh: float = None, vtarg: float = None, lback_periods: int = 21,
                     lback_meth: str = 'ma', half_life: int = 11, signame: str = 'POS'):

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

    assert xcat_sig in set(df['xcat'].unique()), "Signal category missing from the /" \
                                                 "standardised dataframe."
    assert isinstance(vtarg, float) or (vtarg is None), \
        "Volatility Target must be a float."

    assert len(sigrels) == len(ctypes), "The number of signals correspond to the number" \
                                        "of contracts defined in ctypes."

    cols = ['cid', 'xcat', 'real_date', 'value']
    assert set(cols) <= set(df.columns), f"df columns must contain {cols}."

    df = df[cols]

    # B. Reduce to dataframe to required slice
    df = df.loc[:, cols]
    dfd = reduce_df(df=df, xcats=xcats, cids=cids, start=start, end=end,
                    blacklist=blacklist)

    df_unit_pos = unit_positions(df=dfd, cids=cids, xcat_sig=xcat_sig,
                                 blacklist=blacklist, start=start, end=end,
                                 scale=scale, thresh=thresh)

    # Duration in which the signal is defined over: the pivot will be dictated by the
    # longest cross-section.
    df_unit_pos_pivot = df_unit_pos.pivot(index="real_date", columns="cid",
                                          values="value")

    time_index = df_unit_pos_pivot.index

    contract_returns = [c + ret for c in ctypes]
    durations = time_series(dfd, contract_returns)

    if vtarg is not None:

        df_pos_vt = return_series(dfd, xcat_sig, contract_returns, sigrels,
                                  time_index, cids)
        df_pos_vt = df_pos_vt[cols]

        # D.2. Calculate volatility adjustment ratios
        df_vol = historic_vol(df_pos_vt, xcat=ret, cids=cids,
                              lback_periods=lback_periods, lback_meth=lback_meth,
                              half_life=half_life, start=start, end=end,
                              blacklist=blacklist, remove_zeros=True, postfix="")
        # The dimensions of the portfolio's return-series will match the dimensions of
        # signal contract.

        dfw_vol = df_vol.pivot(index="real_date", columns="cid", values="value")
        dfw_vol = dfw_vol.sort_index(axis=1)
        dfw_vtr = 100 * vtarg / dfw_vol  # vol-target ratio to be applied

        # D.3. Calculated vol-targeted positions

        data_frames = []
        for i, sigrel in enumerate(sigrels):

            df_pos = df_unit_pos.copy()
            df_pos['value'] *= sigrel
            dfw_pos = df_pos.pivot(index="real_date", columns="cid",
                                   values="value")

            # Only able to take a position in the contract for the duration in which it
            # is defined.
            tuple_dates = durations[contract_returns[i]]
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

            df_pos_vt = dfw_pos_vt.stack().to_frame("value").reset_index()
            df_pos_vt['xcat'] = contract_returns[i]
            data_frames.append(df_pos_vt)

        df_tpos = pd.concat(data_frames, axis=0, ignore_index=True)

    else:

        df_concat = []
        for i, elem in enumerate(contract_returns):
            # Instantiate a new copy through each iteration.
            df_unit_pos_copy = df_unit_pos_pivot.copy()

            tuple_dates = durations[elem]
            start_date = tuple_dates[0]
            end_date = tuple_dates[1]

            df_unit_pos_copy = df_unit_pos_copy.truncate(before=start_date,
                                                         after=end_date)

            df_unit_pos_copy *= sigrels[i]
            # The current category, defined on the dataframe, is the signal category.
            # But the signal is being used to take a position in multiple contracts.
            # according to the long-short definition. The returned dataframe should be
            # inclusive of all the contracts.
            df_unit_pos = df_unit_pos_copy.stack().to_frame("value").reset_index()

            df_unit_pos['xcat'] = elem

            df_concat.append(df_unit_pos)

        df_tpos = pd.concat(df_concat, axis=0, ignore_index=True)

    df_tpos['xcat'] += '_' + signame
    df_tpos['xcat'] = df_tpos['cid'] + '_' + df_tpos['xcat']
    df_tpos = df_tpos[cols]

    return df_tpos


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
    df = make_qdf(df_cids, df_xcats, back_ar=0.75)
    black = {'AUD': ['2000-01-01', '2003-12-31'], 'GBP': ['2018-01-01', '2100-01-01']}

    xcat_sig = 'FXXR_NSA'
    # Example: ctypes = ['FX', 'EQ']; sigrels = [1, -1]; ret = 'XR_NSA'
    # A single category to determine the position on potentially multiple contracts.
    # The relevant volatility for the volatility adjustment would be the combined returns
    # of each contract. In the below instance, the combined returns of
    # (FXXR_NSA + EQXR_NSA) will be used to determine the evolving volatility.
    position_df = target_positions(df=df, cids=cids, xcats=xcats, xcat_sig='FXXR_NSA',
                                   ctypes=['FX'], sigrels=[1], ret='XR_NSA',
                                   blacklist=black, start='2012-01-01', end='2020-10-30',
                                   scale='dig', vtarg=None, signame='POS')

    print(position_df)

    position_df = target_positions(df=df, cids=cids, xcats=xcats, xcat_sig='FXXR_NSA',
                                   ctypes=['FX', 'EQ'], sigrels=[1, -1], ret='XR_NSA',
                                   blacklist=black, start='2012-01-01', end='2020-10-30',
                                   scale='dig', vtarg=0.1, signame='POS')

    print(position_df)

    # The secondary contract, EQXR_NSA, is defined over a shorter timeframe. Therefore,
    # on the additional dates, a valid position will be computed using the signal
    # category but a position will not be able to be taken for EQXR_NSA.
    position_df = target_positions(df=df, cids=cids, xcats=xcats, xcat_sig='FXXR_NSA',
                                   ctypes=['FX', 'EQ'], sigrels=[1, -1], ret='XR_NSA',
                                   blacklist=black, start='2010-01-01', end='2020-12-31',
                                   scale='prop', vtarg=None, signame='POS')

    print(position_df)