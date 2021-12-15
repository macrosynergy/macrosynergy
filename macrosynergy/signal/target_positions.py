
import numpy as np
import pandas as pd
from typing import List
from macrosynergy.panel.make_zn_scores import *
from macrosynergy.management.shape_dfs import reduce_df
from macrosynergy.panel.historic_vol import historic_vol
from macrosynergy.management.simulate_quantamental_data import make_qdf
import random

def unit_positions(df: pd.DataFrame, cids: List[str], xcat_sig: str,
                   blacklist: dict = None,
                   start: str = None, end: str = None,
                   scale: str = 'prop', thresh: float = None):
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
    :param <float> thresh: threshold value beyond which zn-scores for propotionate
        position taking are winsorized. The threshold is the maximum absolute
        score value in standard deviations. The minimum is 1 standard deviation.

    """

    if scale == 'prop':

        df_unit_pos = make_zn_scores(dfd, xcat=xcat_sig, sequential=True, cids=cids,
                                     neutral='zero', pan_weight=1, thresh=thresh)
    else:

        df_unit_pos = reduce_df(df=df, xcats=[xcat_sig], cids=cids, start=start, end=end,
                              blacklist=blacklist)
        df_unit_pos['value'] = np.sign(df_unit_pos['value']).astype(dtype=np.uint8)

    return df_unit_pos

def target_positions(df: pd.DataFrame, cids: List[str], xcats: List[str], xcat_sig: str,
                     ctypes: List[str], sigrels: List[float], baskets: List[str] = None,
                     ret: str = 'XR_NSA', blacklist: dict = None, start: str = None,
                     end: str = None, scale: str = 'prop',  thresh: float = None,
                     vtarg: float = None, lback_periods: int = 21,
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
    :param <float> thresh: threshold value beyond which zn-scores for propotionate
        posiion taking are winsorized. The threshold is the maximum absolute
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
    assert len(ctypes) == len(sigrels)
    assert scale in ['prop', 'dig']
    assert isinstance(vtarg, float) or (vtarg is None), \
        "Volatility Target must be a float."

    cols = ['cid', 'xcat', 'real_date', 'value']
    assert set(cols) <= set(df.columns), f"df columns must contain {cols}."
    df = df[cols]

    # Todo: a bit light checks for wrong input

    # B. Reduce to dataframe to required slice
    df = df.loc[:, cols]
    dfd = reduce_df(df=df, xcats=xcats, cids=cids, start=start, end=end,
                    blacklist=blacklist)

    df_unit_pos = unit_positions(df=dfd, cids=cids, xcat_sig=xcat_sig,
                                 blacklist=blacklist, start=start, end=end,
                                 scale=scale, thresh=thresh)
    contract_returns = [c + ret for c in ctypes]

    if vtarg is not None:

        for i, c_ret in enumerate(contract_returns):

            dfd_c_ret = dfd[dfd['xcat'] == c_ret]
            dfd_c_ret = dfd_c_ret.pivot(index="real_date", columns="cid", values="value")
            dfd_c_ret = dfd_c_ret.sort_index(axis=1) * sigrels[i]
            if i == 0:  # initiate return dataframe
                dfd_c_rets = dfd_c_ret.copy()
            else:  # combine returns
                dfd_c_rets += dfd_c_ret
        dfd_c_rets.dropna(how='all', inplace=True)

        df_pos_vt = dfd_c_rets.stack().to_frame("value").reset_index()
        df_pos_vt['xcat'] = ret
        df_pos_vt = df_pos_vt[cols]

        # D.2. Calculate volatility adjustment ratios

        df_vol = historic_vol(df_pos_vt, xcat=ret, cids=cids,
                              lback_periods=lback_periods, lback_meth=lback_meth,
                              half_life=half_life, start=start, end=end,
                              blacklist=blacklist, remove_zeros=True, postfix="")

        dfw_vol = df_vol.pivot(index="real_date", columns="cid", values="value")
        dfw_vol = dfw_vol.sort_index(axis=1)
        dfw_vtr = 100 * vtarg / dfw_vol  # vol-target ratio to be applied

        # D.3. Calculated vol-targeted positions

        data_frames = []  # initiate list to collect vol-targeted position dataframes
        for i, sigrel in enumerate(sigrels):
            df_pos = df_unit_pos.copy()
            df_pos['value'] *= sigrel
            dfw_pos = df_pos.pivot(index="real_date", columns="cid",
                                   values="value")
            dfw_pos = dfw_pos.sort_index(axis=1)
            dfw_pos_vt = dfw_pos.multiply(dfw_vtr)  # vol-targeted positions

            df_pos_vt = dfw_pos_vt.stack().to_frame("value").reset_index()
            df_pos_vt['xcat'] = contract_returns[i]
            data_frames.append(df_pos_vt)

        df_tpos = pd.concat(data_frames, axis=0, ignore_index=True)

    else:

        df_concat = []
        for i, elem in enumerate(contract_returns):

            df_unit_pos *= sigrels[i]
            # The current category, defined on the dataframe, is the signal category.
            # But the signal is being used to take a position in multiple contracts.
            # according to the long-short definition. The returned dataframe should be
            # inclusive of all the contracts.
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
    dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)
    black = {'AUD': ['2000-01-01', '2003-12-31'], 'GBP': ['2018-01-01', '2100-01-01']}

    # xcat_sig = 'FXXR_NSA'
    # Example: ctypes = ['FX', 'EQ']; sigrels = [1, -1]; ret = 'XR_NSA'
    # A single category to determine the position on potentially multiple contracts.
    # The relevant volatility for the volatility adjustment would be the combined returns
    # of each contract. In the below instance, the combined returns of
    # (FXXR_NSA + EQXR_NSA) will be used to determine the evolving volatility.
    # position_df = target_positions(df=dfd, cids=cids, xcats=xcats, xcat_sig='FXXR_NSA',
    #                                ctypes=['FX', 'EQ'], sigrels=[1, -1],
    #                                ret='XR_NSA', blacklist=black, start='2012-01-01',
    #                                end='2020-10-30', scale='prop',
    #                                vtarg=0.1, signame='POS')
    #
    # print(position_df)

    position_df = target_positions(df=dfd, cids=cids, xcats=xcats, xcat_sig='FXXR_NSA',
                                   ctypes=['FX', 'EQ'], sigrels=[1, -1], ret='XR_NSA',
                                   blacklist=black, start='2012-01-01', end='2020-10-30',
                                   scale='dig', vtarg=0.1, signame='POS')

    print(position_df)