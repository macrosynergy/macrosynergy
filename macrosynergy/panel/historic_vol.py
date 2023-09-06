"""
Function for calculating historic volatility of quantamental data.

::docs::historic_vol::sort_first::
"""
import numpy as np
import pandas as pd
from typing import List, Union, Tuple, Optional
from macrosynergy.management.simulate_quantamental_data import make_qdf
from macrosynergy.management.shape_dfs import reduce_df
from datetime import timedelta


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

def get_cycles(dates_df: pd.DataFrame, freq: str = "m",) -> pd.Series:
    """Returns a DataFrame with values aggregated by the frequency specified.

    :param <pd.DataFrame>  dates_df: standardized DataFrame with the following necessary columns:
        'cid', 'xcats', 'real_date' and 'value'. Will contain all of the data across all
        macroeconomic fields.
    :param <str> freq: Frequency of the data. Options are 'w' (weekly), 'm' (monthly),
        'q' (quarterly) and 'd' (daily). Default is 'm'.
        
    :return <pd.Series>: A boolean mask which is True where the calculation is triggered.
    """

    def quarters_btwn_dates(start_date : pd.Timestamp, end_date : pd.Timestamp):
        """Returns the number of quarters between two dates."""
        return (end_date.year - start_date.year) * 4 \
            + (end_date.quarter - start_date.quarter)

    def months_btwn_dates(start_date : pd.Timestamp, end_date : pd.Timestamp):
        """Returns the number of months between two dates."""
        return (end_date.year - start_date.year) * 12 + \
                    (end_date.month - start_date.month)
    
    def weeks_btwn_dates(start_date : pd.Timestamp, end_date : pd.Timestamp):
        """Returns the number of business weeks between two dates."""
        next_monday = start_date + pd.offsets.Week(weekday=0)
        dif = (end_date - next_monday).days // 7 + 1
        return dif
    
    freq = freq.lower()
    dfc = dates_df.copy()
    start_date = dfc['real_date'].min()
    funcs = {   'q': quarters_btwn_dates,
                'm': months_btwn_dates,
                'w': weeks_btwn_dates,
                'd': lambda x, y: len(pd.bdate_range(x, y)) - 1}
    
    group_func = funcs[freq]
    dfc['cycleCount'] = dfc['real_date'].apply(
                            lambda x: group_func(start_date, x))

    triggers = dfc['cycleCount'].shift(-1) != dfc['cycleCount']
    # triggers is now a boolean mask which is True where the calculation is triggered
    # ____-____-____-____-_... <-- triggers (_ = False, - = True)
    
    return triggers





def historic_vol(df: pd.DataFrame, xcat: str = None, cids: List[str] = None,
                 lback_periods: int = 21, lback_meth: str = 'ma', half_life=11,
                 start: str = None, end: str = None, est_freq: str = 'd', 
                 blacklist: dict = None, remove_zeros: bool = True, postfix='ASD',
                 nan_tolerance: float = 0.25,):

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
        "ma". Alternative is "xma", Exponential Moving Average. Expects to receive either
        the aforementioned strings.
    :param <int> half_life: Refers to the half-time for "xma". Default is 11.
    :param <str> start: earliest date in ISO format. Default is None and earliest date in
        df is used.
    :param <str> end: latest date in ISO format. Default is None and latest date in df is
        used.
    :param <str> est_freq: Frequency of (re-)estimation of volatility. Options are 'd'
        for end of each day (default), 'w' for end of each work week, 'm' for end of each month,
         and 'q' for end of each week.
    :param <dict> blacklist: cross sections with date ranges that should be excluded from
        the data frame. If one cross section has several blacklist periods append numbers
        to the cross section code.
    :param <int> half_life: Refers to the half-time for "xma" and full lookback period
        for "ma".
    :param <bool> remove_zeros: if True (default) any returns that are exact zeros will
        not be included in the lookback window and prior non-zero values are added to the
        window instead.
    :param <str> postfix: string appended to category name for output; default is "ASD".
    :param <float> nan_tolerance: minimum ratio of NaNs to non-NaNs in a lookback window,
        if exceeded the resulting volatility is set to NaN. Default is 0.25.

    :return <pd.DataFrame>: standardized DataFrame with the estimated annualized standard
        deviations of the chosen xcat.
        If the input 'value' is in % (as is the standard in JPMaQS) then the output
        will also be in %.
        'cid', 'xcat', 'real_date' and 'value'.
    """

    df["real_date"] = pd.to_datetime(df["real_date"], format="%Y-%m-%d")
    df = df[["cid", "xcat", "real_date", "value"]]
    in_df = df.copy()
    est_freq = est_freq.lower()
    assert lback_meth in ['xma', 'ma'], ("Lookback method must be either 'xma' "
                                         "(exponential moving average) or 'ma' (moving average).")
    if lback_meth == 'xma':
        assert lback_periods > half_life, "Half life must be shorter than lookback period."
        assert half_life > 0, "Half life must be greater than 0."
    assert est_freq in ['d', 'w', 'm', 'q'], "Estimation frequency must be one of 'd', 'w', 'm', 'q'."
    
    # assert nan tolerance is an int or float. must be >0. if >1 must be int
    assert isinstance(nan_tolerance, (int, float)), "nan_tolerance must be an int or float."
    assert 0 <= nan_tolerance <= 1, "nan_tolerance must be between 0.0 and 1.0 inclusive."
    
    df = reduce_df(
        df, xcats=[xcat], cids=cids, start=start, end=end, blacklist=blacklist)
    
    dfw = df.pivot(index='real_date', columns='cid', values='value')

    trigger_indices = dfw.index[get_cycles(pd.DataFrame({'real_date': dfw.index}), 
                                           freq=est_freq,)]
    
    def single_calc(row, dfw : pd.DataFrame, lback_periods : int, 
                    nan_tolerance : float, roll_func : callable,
                    remove_zeros : bool, weights : Optional[np.ndarray] = None):

        target_dates = pd.bdate_range(end=row['real_date'], periods=lback_periods)
        target_df : pd.DataFrame = dfw.loc[dfw.index.isin(target_dates)]

        if weights is None:
            out = np.sqrt(252) * \
                target_df.agg(roll_func, remove_zeros=remove_zeros)
        else:
            if len(weights) == len(target_df):
                out = np.sqrt(252) * \
                    target_df.agg(roll_func, w=weights, remove_zeros=remove_zeros)
            else:
                return pd.Series(np.nan, index=target_df.columns)

        mask = ((target_df.isna().sum(axis=0) + 
                    (target_df == 0).sum(axis=0) + 
                    (lback_periods - len(target_df))) 
                / lback_periods) <= nan_tolerance
        # NOTE: dates with NaNs, dates with missing entries, and dates with 0s
        # are all treated as missing data and trigger a NaN in the output        
        out[~mask] = np.nan

        return out
    
    
    if est_freq == 'd':
        if lback_meth == 'xma':
            weights = expo_weights(lback_periods, half_life)
            dfwa = np.sqrt(252) * dfw.rolling(window=lback_periods).agg(
                expo_std, w=weights, remove_zeros=remove_zeros
            )
        else:
            dfwa = np.sqrt(252) * dfw.rolling(window=lback_periods).agg(
                flat_std, remove_zeros=remove_zeros
            )
    else:
        dfwa = pd.DataFrame(index=dfw.index, columns=dfw.columns)
        if lback_meth == 'xma':
            weights = expo_weights(lback_periods, half_life)
            dfwa.loc[trigger_indices, :] =  dfwa.loc[trigger_indices, :].reset_index(False) \
                                            .apply(lambda row: single_calc(
                                                row=row, dfw=dfw, lback_periods=lback_periods, 
                                                nan_tolerance=nan_tolerance, roll_func=expo_std,
                                                remove_zeros=remove_zeros, weights=weights), axis=1) \
                                            .set_index(trigger_indices)

        else:
            dfwa.loc[trigger_indices, :] =  dfwa.loc[trigger_indices, :].reset_index(False) \
                                            .apply(lambda row: single_calc( 
                                                row=row, dfw=dfw, lback_periods=lback_periods,
                                                nan_tolerance=nan_tolerance, roll_func=flat_std,
                                                remove_zeros=remove_zeros), axis=1) \
                                            .set_index(trigger_indices)


        fills = {'d': 1, 'w': 5, 'm': 24, 'q': 64}
        dfwa = dfwa.reindex(dfw.index).fillna(method='ffill', limit=fills[est_freq])

    df_out = dfwa.unstack().reset_index().rename({0: 'value'}, axis=1)
    df_out['xcat'] = xcat + postfix

    # iteratively ensure that each cid has the same date entries as the input df
    df_out_copy = df_out.copy()
    df_out = pd.DataFrame(columns=df_out.columns)

    for cid in cids:
        date_list = in_df[in_df['cid'] == cid]['real_date'].unique()
        date_range = pd.date_range(date_list.min(), date_list.max())
        # only copy over values that are in the date range
        df_out = pd.concat([df_out, 
                            df_out_copy[df_out_copy['real_date'].isin(date_range) 
                                        & (df_out_copy['cid'] == cid)]
                            ])
        
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


    print("Calculating historic volatility with the moving average method")
    df = historic_vol(
        dfd, cids=cids, xcat='XR', lback_periods=7, lback_meth='ma', est_freq="w",
        half_life=3, remove_zeros=True)

    print(df.head(10))

    print("Calculating historic volatility with the exponential moving average method")
    df = historic_vol(dfd, cids=cids, xcat='XR', lback_periods=7, lback_meth='xma',
                        est_freq="w", half_life=3, remove_zeros=True,)

    print(df.head(10))