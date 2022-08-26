
import numpy as np
import pandas as pd
from typing import List
from collections import deque
from macrosynergy.management.shape_dfs import reduce_df
from macrosynergy.management.simulate_quantamental_data import make_qdf

def rolling_date(df_out_index: pd.DatetimeIndex, deck: deque, date: pd.Timestamp):
    """Adjusts the start date of the time-period the neutral level is computed over.

    :param <pd.DatetimeIndex> df_out_index: original daily frequency. Will iterate
        through according to the estimation frequency.
    :param <deque> deck: data structure to hold the rolling period according to the
        maximum number of observations.
    :param <pd.Timestamp> date: the next re-estimation date according to the specified
        frequency.

    :return: rolling timestamp.
    """

    last_date_index = np.where(df_out_index == deck[-1])[0][0]
    next_date_index = last_date_index + 1

    final_date_index = np.where(df_out_index == date)[0][0]

    # Not inclusive of the final date.
    new_dates = list(df_out_index)[next_date_index:final_date_index]
    deck.extend(new_dates)

    first_observation = deck[0]
    return first_observation

def expanding_stat(df: pd.DataFrame, dates_iter: pd.DatetimeIndex,
                   stat: str = 'mean', sequential: bool = True,
                   min_obs: int = 261, max_wind: int = None, iis: bool = True):

    """
    Compute statistic based on an expanding sample.

    :param <pd.DataFrame> df: daily-frequency time series DataFrame.
    :param <pd.DatetimeIndex> dates_iter: controls the frequency of the neutral &
        standard deviation calculations.
    :param <str> stat: statistical method to be applied. This is typically 'mean',
        or 'median'.
    :param <bool> sequential: if True (default) the statistic is estimated sequentially.
        If this set to false a single value is calculated per time series, based on
        the full sample.
    :param <int> max_wind: the maximum number of observations, business days, allowed in
        the expanding window to calculate zn-scores.
    :param <int> min_obs: minimum required observations for calculation of the
        statistic in days.

    :param <bool> iis: if set to True, the values of the initial interval determined
        by min_obs will be estimated in-sample, based on the full initial sample.

    :return: Time series DataFrame of the chosen statistic across all columns.
    """
    max_wind_bool = True if max_wind is not None else False

    index_df = df.index
    df_out = pd.DataFrame(np.nan, index=index_df, columns=['value'])
    # An adjustment for individual series' first realised value is not required given the
    # returned DataFrame will be subtracted from the original DataFrame. The original
    # DataFrame will implicitly host this information through NaN values such that when
    # the arithmetic operation is made, any falsified values will be displaced by NaN
    # values.

    first_observation = df.dropna(axis=0, how='all').index[0]
    first_index = np.where(df_out.index == first_observation)[0][0]

    if max_wind_bool:
        # Instantiate the double-ended Queue with the first "max_wind" number of
        # elements.
        deck = deque(
            df.index[first_index:(first_index + max_wind)], maxlen=max_wind
        )

    # Adjust for individual cross-sections' series commencing at different dates.
    first_estimation = df.dropna(axis=0, how='all').index[min_obs]

    obs_index = next(iter(np.where(df.index == first_observation)))[0]
    est_index = next(iter(np.where(df.index == first_estimation)))[0]

    if stat == "zero":

        df_out["value"] = 0

    elif not sequential:
        # The entire series is treated as in-sample. Will automatically handle NaN
        # values.
        statval = df.stack().apply(stat)
        df_out["value"] = statval

    else:

        dates = dates_iter[dates_iter >= first_estimation]
        for date in dates:

            if max_wind_bool and date > deck[-1]:

                first_observation = rolling_date(
                    df_out_index=index_df, deck=deck, date=date
                )

            # The stack operation will return an empty list instead of a floating point
            # value which precludes it being inserted in the output DataFrame.
            df_out.loc[date, "value"] = df.loc[first_observation:date].stack().apply(stat)

        df_out = df_out.fillna(method='ffill')
        if iis:
            df_out = df_out.fillna(method="bfill", limit=(est_index - obs_index))

    df_out.columns.name = 'cid'
    return df_out

def make_zn_scores(df: pd.DataFrame, xcat: str, cids: List[str] = None,
                   start: str = None, end: str = None, blacklist: dict = None,
                   sequential: bool = True, min_obs: int = 261, max_wind: int = None,
                   iis: bool = True, neutral: str = 'zero', est_freq: str = 'd',
                   thresh: float = None, pan_weight: float = 1, postfix: str = 'ZN'):

    """
    Computes z-scores for a panel around a neutral level ("zn scores").
    
    :param <pd.Dataframe> df: standardized JPMaQS DataFrame with the necessary columns:
        'cid', 'xcat', 'real_date' and 'value'.
    :param <str> xcat:  extended category for which the zn_score is calculated.
    :param <List[str]> cids: cross sections for which zn_scores are calculated; default
        is all available for category.
    :param <str> start: earliest date in ISO format. Default is None and earliest date in
        df is used.
    :param <str> end: latest date in ISO format. Default is None and latest date in df is
        used.
    :param <dict> blacklist: cross-sections with date ranges that should be excluded from
        the calculation of zn-scores.
        This means that not only are there no zn-score values calculated for these
        periods, but also that they are not used for the scoring of other periods.
        N.B.: The argument is a dictionary with cross-sections as keys and tuples of
        start and end dates of the blacklist periods in ISO formats as values.
        If one cross section has multiple blacklist periods, numbers are added to the
        keys (i.e. TRY_1, TRY_2, etc.)
    :param <bool> sequential: if True (default) score parameters (neutral level and
        standard deviations) are estimated sequentially with concurrently available
        information only.
    :param <int> min_obs: the minimum number of observations required to calculate
        zn_scores. Default is 261. The parameter is only applicable if the "sequential"
        parameter is set to True. Otherwise the neutral level and the standard deviation
        are both computed in-sample and will use the full sample.
    :param <int> max_wind: the maximum number of observations, business days, allowed in
        the expanding window to calculate zn-scores. If defined, the expanding
        window will transition to a rolling window of size 'max_wind'. Its purpose is to
        exclude realised values that are considered too "old" (different data generating
        process). Default is None and the entire expanding series will be used.
    :param <bool> iis: if True (default) zn-scores are also calculated for the initial
        sample period defined by min-obs on an in-sample basis to avoid losing history.
        This is irrelevant if sequential is set to False.
    :param <str> neutral: method to determine neutral level. Default is 'zero'.
        Alternatives are 'mean' and "median".
    :param <str> est_freq: the frequency at which standard deviations or means are
        are re-estimated. The options are weekly, monthly & quarterly "w", "m", "q".
        Default is daily, "d". Re-estimation is performed at period end.
    :param <float> thresh: threshold value beyond which scores are winsorized,
        i.e. contained at that threshold. The threshold is the maximum absolute
        score value that the function is allowed to produce. The minimum threshold is 1
        standard deviation.
    :param <float> pan_weight: weight of panel (versus individual cross section) for
        calculating the z-score parameters, i.e. the neutral level and the standard
        deviation. Default is 1, i.e. panel data are the basis for the parameters.
        Lowest possible value is 0, i.e. parameters are all specific to cross section.
    :param <str> postfix: string appended to category name for output; default is "ZN".

    :return <pd.Dataframe>: standardized dataframe with the zn-scores of the chosen xcat:
        'cid', 'xcat', 'real_date' and 'value'.
    """

    # --- Assertions

    assert neutral in ["mean", "median", "zero"]

    if thresh is not None:
        assert thresh > 1, "The 'thresh' parameter must be larger than 1."

    assert 0 <= pan_weight <= 1, "The 'pan_weight' parameter must be between 0 and 1."
    assert isinstance(iis, bool), "Boolean Object required."

    error_min = "Minimum observations must be a non-negative Integer value."
    assert isinstance(min_obs, int) and min_obs >= 0, error_min

    frequencies = ["d", "w", "m", "q"]
    error_freq = f"String Object required and must be one of the available frequencies: " \
                 f"{frequencies}."
    assert isinstance(est_freq, str) and est_freq in frequencies, error_freq
    pd_freq = dict(zip(frequencies, ['B', 'W-Fri', 'BM', 'BQ']))

    if max_wind is not None:
        assert isinstance(max_wind, int) and max_wind >= 0, error_min
        pd_freq_dur = dict(zip(pd_freq.keys(), [1, 5, 21, 63]))

        max_wind_error = "The size of the rolling window must be greater than the chosen" \
                         " estimation frequency."
        assert max_wind >= pd_freq_dur[est_freq], max_wind_error

    # --- Prepare re-estimation dates and time-series DataFrame.

    df = df.loc[:, ['cid', 'xcat', 'real_date', 'value']]
    df = reduce_df(df, xcats=[xcat], cids=cids,
                   start=start, end=end, blacklist=blacklist)

    s_date = min(df['real_date'])
    e_date = max(df['real_date'])
    dates_iter = pd.date_range(start=s_date, end=e_date, freq=pd_freq[est_freq])

    dfw = df.pivot(index='real_date', columns='cid', values='value')
    cross_sections = dfw.columns

    # --- The actual scoring.

    dfw_zns_pan = dfw * 0
    dfw_zns_css = dfw * 0

    if pan_weight > 0:

        df_neutral = expanding_stat(
            dfw, dates_iter, stat=neutral, sequential=sequential, min_obs=min_obs,
            max_wind=max_wind, iis=iis
        )
        dfx = dfw.sub(df_neutral['value'], axis=0)
        df_mabs = expanding_stat(
            dfx.abs(), dates_iter, stat="mean", sequential=sequential, min_obs=min_obs,
            iis=iis
        )
        dfw_zns_pan = dfx.div(df_mabs['value'], axis='rows')

    if pan_weight < 1:

        for cid in cross_sections:
            dfi = dfw[cid]

            df_neutral = expanding_stat(dfi.to_frame(name=cid), dates_iter, stat=neutral,
                                        sequential=sequential,
                                        min_obs=min_obs, iis=iis)
            dfx = dfi - df_neutral['value']

            df_mabs = expanding_stat(dfx.abs().to_frame(name=cid), dates_iter,
                                     stat="mean", sequential=sequential,
                                     min_obs=min_obs, iis=iis)
            dfx = pd.DataFrame(data=dfx.to_numpy(),
                               index=dfx.index, columns=['value'])
            dfx = dfx.rename_axis('cid', axis=1)

            zns_css_df = dfx / df_mabs
            dfw_zns_css.loc[:, cid] = zns_css_df.to_numpy()

    dfw_zns = (dfw_zns_pan * pan_weight) + (dfw_zns_css * (1 - pan_weight))
    dfw_zns = dfw_zns.dropna(axis=0, how='all')
    
    if thresh is not None:
        dfw_zns.clip(lower=-thresh, upper=thresh, inplace=True)

    # --- Reformatting of output into standardised DataFrame.

    df_out = dfw_zns.stack().to_frame("value").reset_index()
    df_out['xcat'] = xcat + postfix

    col_names = ['cid', 'xcat', 'real_date', 'value']
    df_out = df_out.sort_values(['cid', 'real_date'])[col_names]

    return df_out[df.columns].reset_index(drop=True)


if __name__ == "__main__":

    cids = ['AUD', 'CAD', 'GBP', 'USD', 'NZD']
    xcats = ['XR', 'CRY', 'GROWTH', 'INFL']

    df_cids = pd.DataFrame(index = cids, columns = ['earliest', 'latest', 'mean_add',
                                                    'sd_mult'])

    df_cids.loc['AUD'] = ['2010-01-01', '2020-12-31', 0.5, 2]
    df_cids.loc['CAD'] = ['2006-01-01', '2020-11-30', 0, 1]
    df_cids.loc['GBP'] = ['2008-01-01', '2020-11-30', -0.2, 0.5]
    df_cids.loc['USD'] = ['2007-01-01', '2020-09-30', -0.2, 0.5]
    df_cids.loc['NZD'] = ['2002-01-01', '2020-09-30', -0.1, 2]

    df_xcats = pd.DataFrame(index = xcats, columns = ['earliest', 'latest', 'mean_add',
                                                      'sd_mult', 'ar_coef', 'back_coef'])
    df_xcats.loc['XR'] = ['2008-01-01', '2020-12-31', 0, 1, 0, 0.3]
    df_xcats.loc['CRY'] = ['2011-01-01', '2020-10-30', 1, 2, 0.9, 0.5]
    df_xcats.loc['GROWTH'] = ['2012-01-01', '2020-10-30', 1, 2, 0.9, 1]
    df_xcats.loc['INFL'] = ['2013-01-01', '2020-10-30', 1, 2, 0.8, 0.5]

    # Apply a blacklist period from series' start date.
    black = {'AUD': ['2010-01-01', '2013-12-31'],
             'GBP': ['2018-01-01', '2100-01-01']}

    dfd = make_qdf(df_cids, df_xcats, back_ar = 0.75)

    # Monthly: panel + cross.
    # Apply the maximum window.
    dfzm = make_zn_scores(
        dfd, xcat='XR', sequential=True, cids=cids, blacklist=black, iis=True,
        neutral='mean', pan_weight=0.75, min_obs=261, max_wind=500, est_freq="q"
    )
