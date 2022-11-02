
import numpy as np
import pandas as pd
from typing import List, Union, Tuple
from macrosynergy.management.simulate_quantamental_data import make_qdf
from macrosynergy.management.shape_dfs import reduce_df


def expo_weights(lback_periods: int = 21, half_life: int = 11):
    """
    Calculates exponential series weights for finite horizon, normalized to 1.
    
    :param <int>  lback_periods: Number of lookback periods over which volatility is
        calculated. Default is 21.
    :param <int> half_life: Refers to the half-time for "xma" and full lookback period
        for "ma". Default is 11.

    :return <np.ndarray>: An Array of weights determined by the length of the lookback
        period.

    Note: 50% of the weight allocation will be applied to the number of days delimited by
          the half_life.
    """
    decf = 2 ** (-1 / half_life)
    weights = (1 - decf) * np.array([decf ** (lback_periods - ii - 1)
                                     for ii in range(lback_periods)])
    weights = weights/sum(weights)
    
    return weights


def expo_std(x: np.ndarray, w: np.ndarray, remove_zeros: bool = True):
    """
    Estimate standard deviation of returns based on exponentially weighted absolute
    values.

    :param <np.ndarray> x: array of returns
    :param <np.ndarray> w: array of exponential weights (same length as x); will be
        normalized to 1.
    :param <bool> remove_zeros: removes zeroes as invalid entries and shortens the
        effective window.

    :return <float>: exponentially weighted mean absolute value (as proxy of return
        standard deviation).

    """
    assert len(x) == len(w), "weights and window must have same length"
    if remove_zeros:
        x = x[x != 0]
        w = w[0:len(x)] / sum(w[0:len(x)])
    w = w / sum(w)  # weights are normalized
    mabs = np.sum(np.multiply(w, np.abs(x)))
    return mabs


def flat_std(x: np.ndarray, remove_zeros: bool = True):
    """
    Estimate standard deviation of returns based on exponentially weighted absolute
    values.

    :param <np.ndarray> x: array of returns
    :param <bool> remove_zeros: removes zeroes as invalid entries and shortens the
        effective window.

    :return <float>: flat weighted mean absolute value (as proxy of return standard
        deviation).

    """
    if remove_zeros:
        x = x[x != 0]
    mabs = np.mean(np.abs(x))
    return mabs


def historic_vol(df: pd.DataFrame, xcat: str = None, cids: List[str] = None,
                 lback_periods: int = 21, lback_meth: str = 'ma', half_life=11,
                 start: str = None, end: str = None, blacklist: dict = None,
                 remove_zeros: bool = True, postfix='ASD'):

    """
    Estimate historic annualized standard deviations of asset returns. User Function.
    Controls the functionality.

    :param <pd.DataFrame> df: standardized DataFrame with the following necessary columns:
        'cid', 'xcats', 'real_date' and 'value'. Will contain all of the data across all
        macroeconomic fields.
    :param <str> xcat:  extended category denoting the return series for which volatility
        should be calculated.
        Note: in JPMaQS returns are represented in %, i.e. 5 means 5%.
    :param <List[str]> cids: cross sections for which volatility is calculated;
        default is all available for the category.
    :param <int>  lback_periods: Number of lookback periods over which volatility is
        calculated. Default is 21.
    :param <str> lback_meth: Lookback method to calculate the volatility, Default is
        "ma". Alternative is "ema", Exponential Moving Average. Expects to receive either
        the aforementioned strings.
    :param <int> half_life: Refers to the half-time for "xma". Default is 11.
    :param <str> start: earliest date in ISO format. Default is None and earliest date in
        df is used.
    :param <str> end: latest date in ISO format. Default is None and latest date in df is
        used.
    :param <dict> blacklist: cross sections with date ranges that should be excluded from
        the data frame. If one cross section has several blacklist periods append numbers
        to the cross section code.
    :param <int> half_life: Refers to the half-time for "xma" and full lookback period
        for "ma".
    :param <bool> remove_zeros: if True (default) any returns that are exact zeros will
        not be included in the lookback window and prior non-zero values are added to the
        window instead.
    :param <str> postfix: string appended to category name for output; default is "ASD".

    :return <pd.DataFrame>: standardized DataFrame with the estimated annualized standard
        deviations of the chosen xcat.
        If the input 'value' is in % (as is the standard in JPMaQS) then the output
        will also be in %.
        'cid', 'xcat', 'real_date' and 'value'.
    """

    df["real_date"] = pd.to_datetime(df["real_date"], format="%Y-%m-%d")
    df = df[["cid", "xcat", "real_date", "value"]]

    assert lback_periods > half_life, "Half life must be shorter than lookback period."
    assert lback_meth in ['xma', 'ma'], "Incorrect request."

    df = reduce_df(
        df, xcats=[xcat], cids=cids, start=start, end=end, blacklist=blacklist
    )
    dfw = df.pivot(index='real_date', columns='cid', values='value')

    # The pandas in-built method df.rolling() will account for NaNs and start from the
    # "first valid index".
    if lback_meth == 'xma':
        weights = expo_weights(lback_periods, half_life)
        dfwa = np.sqrt(252) * dfw.rolling(window=lback_periods).agg(
            expo_std, w=weights, remove_zeros=remove_zeros
        )
    else:
        dfwa = np.sqrt(252) * dfw.rolling(window=lback_periods).agg(
            flat_std, remove_zeros=remove_zeros
        )

    df_out = dfwa.stack().to_frame("value").reset_index()

    df_out['xcat'] = xcat + postfix

    return df_out[df.columns].sort_values(['cid', 'xcat', 'real_date'])


if __name__ == "__main__":

    cids = ['AUD', 'CAD', 'GBP', 'USD']
    xcats = ['XR', 'CRY', 'GROWTH', 'INFL']

    df_cids = pd.DataFrame(index=cids, columns=['earliest', 'latest',
                                                'mean_add', 'sd_mult'])

    df_cids.loc['AUD'] = ['2010-01-01', '2020-12-31', 0.5, 2]
    df_cids.loc['CAD'] = ['2011-01-01', '2020-11-30', 0, 1]
    df_cids.loc['GBP'] = ['2012-01-01', '2020-10-30', -0.2, 0.5]
    df_cids.loc['USD'] = ['2013-01-01', '2020-09-30', -0.2, 0.5]

    df_xcats = pd.DataFrame(index=xcats, columns=['earliest', 'latest', 'mean_add',
                                                  'sd_mult', 'ar_coef', 'back_coef'])
    df_xcats.loc['XR'] = ['2010-01-01', '2020-12-31', 0, 1, 0, 0.3]
    df_xcats.loc['CRY'] = ['2011-01-01', '2020-10-30', 1, 2, 0.9, 0.5]
    df_xcats.loc['GROWTH'] = ['2012-01-01', '2020-10-30', 1, 2, 0.9, 1]
    df_xcats.loc['INFL'] = ['2013-01-01', '2020-10-30', 1, 2, 0.8, 0.5]
    dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)
    dfd["grading"] = np.ones(dfd.shape[0])

    weights = expo_weights(lback_periods=21, half_life=11)

    df = historic_vol(
        dfd, cids=cids, xcat='XR', lback_periods=21, lback_meth='ma', half_life=11,
        remove_zeros=True
    )
