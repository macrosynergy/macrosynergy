
import numpy as np
import pandas as pd
from typing import List
from macrosynergy.panel.make_zn_scores import *
from macrosynergy.management.shape_dfs import reduce_df
from macrosynergy.panel.historic_vol import historic_vol
from macrosynergy.management.simulate_quantamental_data import make_qdf
import random


def modify_signals(df: pd.DataFrame, cids: List[str], xcat_sig: str, start: str = None,
                   end: str = None, scale: str = 'prop',  min_obs: int = 252,
                   thresh: float = None):

    """
    Calculate modified cross-section signals based on zn-scoring (proportionate method)
    or conversion to signs (digital method).

    :param <pd.Dataframe> df: standardized DataFrame containing the following columns:
        'cid', 'xcats', 'real_date' and 'value'.
    :param <List[str]> cids: cross sections of markets or currency areas in which
        positions should be taken.
    :param <str> xcat_sig: category that serves as signal across markets.
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
        Note: For the initial period of the signal time series in-sample
        zn-scoring is used.
    :param <float> thresh: threshold value beyond which zn-scores for propotionate
        position taking are winsorized. The threshold is the maximum absolute
        score value in standard deviations. The minimum is 1 standard deviation.

    :return <pd.Dataframe>: standardized dataframe, of modified signaks, using the 
        columns 'cid', 'xcat', 'real_date' and 'value'.

    """

    options = ['prop', 'dig']
    assert scale in options, f"The scale parameter must be either {options}"

    if scale == 'prop':

        df_ms = make_zn_scores(df, xcat=xcat_sig, sequential=True, cids=cids,
                               start=start, end=end, neutral='zero', pan_weight=1,
                               min_obs=min_obs, thresh=thresh)
    else:

        df_ms = reduce_df(df=df, xcats=[xcat_sig], cids=cids, start=start, end=end,
                          blacklist=None)
        df_ms['value'] = np.sign(df_ms['value'])

    return df_ms


def cs_unit_returns(df: pd.DataFrame, contract_returns: List[str],
                    sigrels: List[str], ret: str = 'XR_NSA'):

    """
    Calculate returns of composite unit positions (that jointly depend on one signal).

    :param <pd.Dataframe> df: standardized DataFrame containing the following columns:
        'cid', 'xcats', 'real_date' and 'value'.
    :param <List[str]> contract_returns: list of the contract return types.
    :param <List[float]> sigrels: respective signal for each contract type.
    :param <str> ret: postfix denoting the returns in % applied to the contract types.

    :return <pd.Dataframe>: standardized dataframe with the summed portfolio returns
        which are used to calculate the evolving volatility, using the columns 'cid',
        'xcat', 'real_date' and 'value'.

    """

    assert len(contract_returns) == len(sigrels), \
        "Each individual contract requires an associated signal."

    for i, c_ret in enumerate(contract_returns):

        df_c_ret = df[df['xcat'] == c_ret]
        df_c_ret = df_c_ret.pivot(index="real_date", columns="cid", values="value")

        df_c_ret = df_c_ret.sort_index(axis=1)
        df_c_ret *= sigrels[i]

        if i == 0:  # Add each return series of the contract.
            df_c_rets = df_c_ret
        else:
            df_c_rets += df_c_ret

    # Any dates not shared by all categories will be removed.
    df_c_rets.dropna(how='all', inplace=True)

    df_rets = df_c_rets.stack().to_frame("value").reset_index()
    df_rets['xcat'] = ret

    return df_rets


def target_positions(df: pd.DataFrame, cids: List[str], xcat_sig: str, ctypes: List[str],
                     sigrels: List[float], baskets: List[str] = None, ret: str = 'XR_NSA',
                     start: str = None, end: str = None,
                     scale: str = 'prop', min_obs: int = 252, thresh: float = None,
                     cs_vtarg: float = None, lback_periods: int = 21,
                     lback_meth: str = 'ma', half_life: int = 11, posname: str = 'POS'):

    """
    Converts signals into contract-specific target positions

    :param <pd.Dataframe> df: standardized DataFrame containing at least the following
        columns: 'cid', 'xcats', 'real_date' and 'value'.
    :param <List[str]> cids: cross-sections of markets or currency areas in which
        positions should be taken.
    :param <str> xcat_sig: category that serves as signal across markets.
    :param <List[str]> ctypes: contract types that are traded across markets. They should
        correspond to return categories in the dataframe if the `ret` argument is
        appended. Examples are 'FX' or 'EQ'.
    :param <Dict[Any, Dict]> baskets: dictionary of basket dictionaries. The inner
        dictionary takes a string of form <cross_section>_<contract_type> as key and a
        list of string of the same form as value. The key labels the basket. The value
        defines the contracts that are used for forming the basket. Pe default the
        contract have equal weights. An example would be:
        {{'APC_FX' : ['AUD_FX', ''NZD_FX', 'JPY_FX']},
         {'APC_EQ' : ['AUD_EQ', ''CNY_EQ', 'INR_EQ', 'JPY_EQ']}}
        # Todo: has yet to be implemented
    :param <List[float]> sigrels: values that translate the single signal into contract
        type and basket signals in the order defined by keys.
    :param <str> ret: postfix denoting the returns in % associated with contract types.
        For JPMaQS derivatives return data this is typically "XR_NSA".
        The returns are necessary for volatility target-based signals.
    :param <str> start: earliest date in ISO format. Default is None and earliest date
        for which the signal category is available is used.
    :param <str> end: latest date in ISO format. Default is None and latest date
        for which the signal category is available is used.
    :param <str> scale: method that translates signals into unit target positions:
        [1] Default is 'prop' for proportionate. In this case zn-scoring is applied
            to the signal based on the panel, with the neutral level set at zero.
            A 1 SD value translates into a USD1 position in the contract.
            This translation may apply winsorization through the `thresh` argument
        [2] Method 'dig' means 'digital' and sets the individual position to either USD1
            long or short, depending on the sign of the signal.
        Note that unit target positions may subsequently be calibrated to meet cross-
        section volatility targets.
    :param <int> min_obs: the minimum number of observations required to calculate
        zn_scores. Default is 252.
        Note: For the initial minimum period of the signal time series in-sample
        zn-scoring is used.
    :param <float> thresh: threshold value beyond which zn-scores for proportionate
        position taking are winsorized. The threshold is the maximum absolute
        score value in standard deviations. The minimum is 1 standard deviation.
    :param <float> cs_vtarg: This allows volatility targeting at the cross-section level.
        Default is None, but if a value is chosen then for each cross-section a unit
        position is defined as a position for which the annual return standard deviation
        is equal to that value.
        For example, a target of 10 and a cross-section signal of 0.5 standard deviations
        would translate into a target position that carries a recent historical
        annualized standard deviation of 5 dollars (or other currency units).
    :param <int>  lback_periods: Number of lookback periods over which volatility is
        calculated. Default is 21.
    :param <str> lback_meth: Lookback method to calculate the volatility.
        Default is "ma". Alternative is "ema", exponential moving average.
    :param <int> half_life: Refers to the half-time for "xma". Default is 11.
    :param <str> posname: postfix added to contract to denote position name.

    :return <pd.Dataframe>: standardized dataframe with daily target positions
        in USD, using the columns 'cid', 'xcat', 'real_date' and 'value'.

    Note: A target position differs from a signal insofar as it is a dollar amount and
          determines to what extent size of signal (as opposed to direction) matters.
          A target position also differs from an actual position in two ways. First,
          the actual position can only be aligned with the target with some lag. Second,
          the actual position will be affected by other considerations, such as
          risk management and assets under management.
    """

    # A. Initial checks

    assert xcat_sig in set(df['xcat'].unique()), \
        "Signal category missing from the standardised dataframe."
    assert isinstance(cs_vtarg, (float, int)) or (cs_vtarg is None) \
        and not isinstance(cs_vtarg, bool), \
        "Volatility Target must be numeric or None."
    assert len(sigrels) == len(ctypes), \
        "The number of signal relations must be equal to the number of contracts " \
        "defined in ctypes."
    assert isinstance(min_obs, int), \
        "Minimum observation parameter must be an integer."

    # Todo: asserts for other argument types and permissible values

    cols = ['cid', 'xcat', 'real_date', 'value']
    assert set(cols) <= set(df.columns), f"df columns must contain {cols}."

    # B. Reduce frame to necessary data

    df = df.loc[:, cols]
    contract_returns = [c + ret for c in ctypes]
    xcats = contract_returns + [xcat_sig]

    dfx = reduce_df(df=df, xcats=xcats, cids=cids, start=start, end=end, blacklist=None)

    # C. Calculate and reformat modified cross-sectional signals.

    df_mods = modify_signals(df=dfx, cids=cids, xcat_sig=xcat_sig,
                             start=start, end=end, scale=scale, min_obs=min_obs,
                             thresh=thresh)  # (USD 1 per SD or sign)

    df_mods_w = df_mods.pivot(index="real_date", columns="cid", values="value")

    # D. Volatility target ratios (if required).

    use_vtr = False
    if isinstance(cs_vtarg, (int, float)):

        # D.1. Composite signal-related positions as basis for volatility targeting.

        df_csurs = cs_unit_returns(dfx, contract_returns=contract_returns,
                                     sigrels=sigrels)  # gives cross-section returns
        df_csurs = df_csurs[cols]

        # D.2. Calculate volatility adjustment ratios.

        df_vol = historic_vol(df_csurs, xcat=ret, cids=cids,
                              lback_periods=lback_periods, lback_meth=lback_meth,
                              half_life=half_life, start=start, end=end,
                              remove_zeros=True,
                              postfix="")  # gives unit position vols.

        dfw_vol = df_vol.pivot(index="real_date", columns="cid", values="value")
        dfw_vol = dfw_vol.sort_index(axis=1)
        dfw_vtr = 100 * cs_vtarg / dfw_vol  # vol-target ratio to be applied.
        use_vtr = True

    # E. Actual position calculation

    data_frames = []
    for i, sigrel in enumerate(sigrels):  # loop through legs of cross-section positions

        df_mods_copy = df_mods_w.copy()  # copy of all modified signals
        df_mods_copy *= sigrel  # modified signal x sigrel = pre-VT position of leg

        if use_vtr:
            dfw_pos_vt = df_mods_copy.multiply(dfw_vtr)  # apply vtr
            dfw_pos_vt.dropna(how='all', inplace=True)
            df_mods_copy = dfw_pos_vt  # Todo: why not modify directly?

        # Todo: if basket this translates into n basket contract positions
        # Todo: contract position = basket_position / n

        df_posi = df_mods_copy.stack().to_frame("value").reset_index()
        df_posi['xcat'] = ctypes[i]
        data_frames.append(df_posi)

    df_tpos = pd.concat(data_frames, axis=0, ignore_index=True)

    df_tpos['xcat'] += '_' + posname
    df_tpos['xcat'] = df_tpos['cid'] + '_' + df_tpos['xcat']
    df_tpos = df_tpos[cols]

    df_tpos = reduce_df(df=df_tpos, xcats=None, cids=None, start=start, end=end)

    df_tpos = df_tpos.sort_values(['cid', 'xcat', 'real_date'])[cols]
    # Todo: if baskets are used position have to be consolidated
    # Todo: this means positions with same ['cid', 'xcat', 'real_date'] must be added
    return df_tpos.reset_index(drop=True)


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
                                   xcat_sig='SIG_NSA',
                                   ctypes=['FX', 'EQ'], sigrels=[1, 0.5], ret='XR_NSA',
                                   start='2012-01-01', end='2020-10-30',
                                   scale='prop', min_obs=252, cs_vtarg=5, posname='POS')
    print(position_df)

    position_df = target_positions(df=dfd, cids=cids, xcat_sig='FXXR_NSA',
                                   ctypes=['FX', 'EQ'], sigrels=[1, -1], ret='XR_NSA',
                                   start='2012-01-01', end='2020-10-30',
                                   scale='dig', cs_vtarg=0.1, posname='POS')

    print(position_df)

    # The secondary contract, EQXR_NSA, is defined over a shorter timeframe. Therefore,
    # on the additional dates, a valid position will be computed using the signal
    # category but a position will not be able to be taken for EQXR_NSA.
    position_df = target_positions(df=dfd, cids=cids, xcat_sig='FXXR_NSA',
                                   ctypes=['FX', 'EQ'], sigrels=[1, -1], ret='XR_NSA',
                                   start='2010-01-01', end='2020-12-31',
                                   scale='prop', cs_vtarg=None, posname='POS')

    print(position_df)