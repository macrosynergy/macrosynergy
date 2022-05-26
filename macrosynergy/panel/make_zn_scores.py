
import numpy as np
import pandas as pd
from typing import List
from itertools import filterfalse
from macrosynergy.management.shape_dfs import reduce_df
from macrosynergy.panel.expanding_statistics import expanding_mean_with_nan
from macrosynergy.management.simulate_quantamental_data import make_qdf


def func_executor(df: pd.DataFrame, neutral: str, n: int,
                  dates_iter: List[pd.Timestamp], min_obs: int = 261):
    """
    Used to calculate the expanding neutral level or standard deviation across a panel.

    :param <pd.DataFrame> df:
    :param <str> neutral:
    :param <int> n: number of dates the neutral level is computed over.
    :param <List[pd.Timestamps]> dates_iter:
    :param <int> min_obs:

    return <pd.DataFrame>: DataFrame containing the neutral level populated daily.
    """

    daily = n == len(dates_iter)
    # Inclusive of the first date of the DataFrame and respective intervals. The
    # "dates_iter" data structure, depending on the frequency, will start on the first
    # re-estimation date as opposed to the first realised date of the return series.
    f_date = df.index[0]
    # Daily dates DataFrame. Used if down-sampling has been applied.
    dates_df = pd.DataFrame(index=df.index)

    if neutral == "mean" and not daily:
        # If down-sampling, the neutral level will still be computed using daily data but
        # the calculation will occur at the stated estimation frequency using all of the
        # preceding business dates. It is imperative to include the daily data to capture
        # the asset's variance.
        ar_neutral = np.array([df.loc[f_date:d, :].stack().mean()
                               for d in dates_iter])
    elif neutral == "mean":
        # If daily frequency, utilise the computationally faster algorithm.
        ar_neutral = expanding_mean_with_nan(dfw=df)
        # In-sampling period.
        ar_neutral[:min_obs] = np.nan
    else:
        ar_neutral = np.array([df.loc[f_date:d, :].stack().median()
                               for d in dates_iter])

    neutral_df = pd.DataFrame(data=ar_neutral, index=dates_iter)
    neutral_df.index.name = 'real_date'
    neutral_df.columns = ['value']
    if not daily:
        neutral_df = dates_df.merge(neutral_df, how='left', on='real_date')
        neutral_df = neutral_df.fillna(method='ffill')

    return neutral_df


def pan_neutral(df: pd.DataFrame, dates_iter: List[pd.Timestamp], neutral: str = 'zero',
                sequential: bool = False, min_obs: int = 261, iis: bool = False):

    """
    Compute neutral values of return series based on a panel.

    :param <pd.Dataframe> df: "wide" DataFrame.
    :param <List[pd.Timestamp]> dates_iter: controls the frequency of the neutral &
        standard deviation calculations.
    :param <str> neutral: method to determine neutral level. Default is 'zero'.
        Alternatives are 'mean' and "median".
    :param <bool> sequential: if True (default) score parameters (neutral level and
        standard deviations) are estimated sequentially with cumulative concurrently
        available information only. If False one neutral value will be calculated for
        the whole panel.
    :param <int> min_obs: minimum required observations in days.
    :param <bool> iis: if set to True, the number of days outlined by "min_obs" will be
        calculated in sample (single neutral value for the time period) whilst the
        remaining days are calculated on a rolling basis. However, if False,
        and "sequential" equals True, the rolling neutral value will be calculated from
        the start date.

    :return <pd.DataFrame> neutral_df: row-wise neutral statistic. A single value
        produced per timestamp.

    NB: It is worth noting that the evolving neutral level, if the "sequential" parameter
        is set to True, will be computed using the available cross-sections on the
        respective date. The code will not adjust for an incomplete set of cross-sections
        on each date. For instance, if the first 100 days only have 3 cross-sections with
        realised values out of the 4 defined, the rolling mean will be calculated using
        the available subset.
    """
    no_rows = df.shape[0]
    func_dict = {'mean': np.mean, 'median': np.median}

    # The median neutral level is primarily used if the sample set of data is exposed
    # heavily to outliers. In such instances, the mean statistic will misrepresent the
    # sample of data.
    if neutral in func_dict.keys():
        if sequential and not iis:

            neutral_df = func_executor(df=df, neutral=neutral, n=no_rows,
                                       dates_iter=dates_iter)
            neutral_df.iloc[0:min_obs] = np.nan

        elif sequential and iis:
            neutral_df = func_executor(df=df, neutral=neutral, n=no_rows,
                                       dates_iter=dates_iter)
            # The back-fill mechanism will use the next valid observation to fill the
            # gap. The aforementioned observation date's calculation will be inclusive of
            # the entire in-sampling period. Therefore, use the calculation to populate
            # the in-sampling period.
            neutral_df = neutral_df.fillna(method='backfill')
        else:
            iis_period = pd.DataFrame(df.stack().to_numpy())
            neutral_val = iis_period.apply(func_dict[neutral])
            neutral_arr = np.repeat(float(neutral_val), no_rows)
            neutral_df = pd.DataFrame(data=neutral_arr, index=df.index)

    else:
        neutral_df = pd.DataFrame(data=np.zeros(no_rows), index=df.index)

    neutral_df.columns = ['value']
    return neutral_df


def index_info(df_row_no: int, column: pd.Series, min_obs: int):
    """
    Method used to determine the first date where the cross-section has a realised value.
    Will vary across the panel for each cross-section.

    :param: <pd.Series> column: individual cross-section's data-series.
    :param: <int> min_obs:
    :param: <int> df_row_no: the number of rows defined in the original pivoted dataframe.
        The number of rows the dataframe is defined over corresponds to the first and
        last date across the panel. Therefore, certain cross-sections will have NaN
        values if their series do not align.

    :return <int>:
    """

    index = column.index
    date = column.first_valid_index()
    # Integer index at which the series' first realised value occurs.
    date_index = next(iter(np.where(index == date)[0]))

    # Number of "active" dates for the cross-section's series.
    df_row_no -= date_index
    first_date = date_index + min_obs

    return df_row_no, first_date, date_index


def in_sample_series(column: pd.Series, neutral: str, no_timestamps: int,
                     date_index: int, cid: str):
    """
    The return series' neutral level is calculated exclusively in-sample.

    :param <pd.Series> column: individual cross-section's time-series data including any
        preceding unrealised timestamps.
    :param <str> neutral:
    :param <int> no_timestamps: number of realised dates the cross-section is defined
        over.
    :param <int> date_index: index of the first realised return.
    :param <str> cid: associated cross-section.

    """

    func_dict = {'mean': np.mean, 'median': np.median}
    column_r = column[date_index:]

    # Isolate the realised returns for each cross-section and host the return series in
    # a pd.DataFrame to utilise the pd.apply() method.
    df_realised = pd.DataFrame(data=column_r.to_numpy(), index=column_r.index)
    n_val = df_realised.apply(func_dict[neutral])
    ar_neutral = np.repeat(float(n_val), no_timestamps)

    undefined = np.empty(date_index)
    undefined[:] = np.nan

    ar_neutral = np.concatenate([undefined, ar_neutral], axis=0)
    neutral_df = pd.DataFrame(data=ar_neutral, index=column.index)
    neutral_df.columns = [cid]

    return neutral_df

def neutral_calc(column: pd.Series, dates_iter: List[pd.Timestamp], iis: bool,
                 neutral: str, date_index: int, min_obs: int, cid: str):
    """
    Helper function to compute the cross-sectional expanding neutral values. Will adjust
    for down-sampling.

    :param <pd.Series> column: individual cross-section's time-series data.
    :param <List[pd.Timestamp]> dates_iter: controls the frequency of the neutral &
        standard deviation calculations.
    :param <bool> iis:
    :param <str> neutral:
    :param <int> date_index: index of the first active trading day.
    :param <int> min_obs:
    :param <str> cid: associated cross-section

    :return <pd.DataFrame> computed neutral levels.
    """

    func_dict = {'mean': np.mean, 'median': np.median}
    column_r = column[date_index:]
    df_realised = pd.DataFrame(data=column_r.to_numpy(), index=column_r.index)
    first_date = column_r.index[0]

    r_dates = list(filterfalse(lambda d: d < first_date, dates_iter))
    ur_dates = [d for d in dates_iter if d not in r_dates]

    iis_period = column_r.iloc[:min_obs]
    os_neutral = np.array([float(df_realised.loc[first_date:d].apply(func_dict[neutral]))
                           for d in r_dates])

    undefined = np.empty(len(ur_dates))
    undefined[:] = np.nan
    ar_neutral = np.concatenate([undefined, os_neutral])
    neutral_df = pd.DataFrame(data=ar_neutral, index=(ur_dates + r_dates))
    neutral_df.index.name = 'real_date'
    dates_df = pd.DataFrame(index=column.index)

    neutral_df = dates_df.merge(neutral_df, how='left', on='real_date')
    neutral_df = neutral_df.fillna(method='ffill')

    iis_end = (date_index + min_obs)
    # To prevent loss of information, calculate the first minimum number of observation
    # days using an in-sampling technique.
    if iis:
        iis_df = pd.DataFrame(data=iis_period)
        neutral_iis = iis_df.apply(func_dict[neutral])
        neutral_df.iloc[date_index: iis_end] = float(neutral_iis)
    else:
        neutral_df.iloc[date_index: iis_end] = np.nan

    neutral_df.columns = [cid]

    return neutral_df


def cross_neutral(df: pd.DataFrame, neutral: str = 'zero', est_freq: str = 'd',
                  sequential: bool = False, min_obs: int = 261, iis: bool = False):
    """
    Compute neutral values of return series individually for all cross-sections.

    :param <pd.Dataframe> df: pivoted DataFrame. The DataFrame's columns will naturally
        consist of each cross-section's return series.
    :param <str> neutral: method to determine neutral level. Default is 'zero'.
        Alternatives are 'mean' and "median".
    :param <str> est_freq: the frequency at which standard deviations or means are
        are re-estimated.
    :param <bool> sequential: if True (default) score parameters (neutral level and
        standard deviations) are estimated sequentially using the preceding dates'
        realised returns. If False one neutral value will be calculated for the
        whole cross-section.
    :param <int> min_obs:
    :param <bool> iis:

    :return <pd.DataFrame> pd_neutral: column-wise neutral statistic. Same dimensions as
        the received DataFrame.
    """

    dates_iter = pd.date_range(start=df.index[0], end=df.index[-1], freq=est_freq)

    if neutral != 'zero':
        pd_neutral = pd.DataFrame(index=df.index)
        for i, cross in enumerate(df.columns):

            column = df.iloc[:, i]
            original_index_no = df.shape[0]
            df_row_no, first_date, date_index = index_info(original_index_no, column,
                                                           min_obs=min_obs)
            if sequential:
                neutral_df = neutral_calc(column=column, dates_iter=dates_iter,
                                          iis=iis, neutral=neutral,
                                          date_index=date_index, min_obs=min_obs,
                                          cid=cross)
            else:
                neutral_df = in_sample_series(column, neutral, df_row_no, date_index,
                                              cross)
            pd_neutral = pd_neutral.join(neutral_df, on='real_date', how='left')
    else:
        pd_neutral = pd.DataFrame(data=np.zeros(df.shape), index=df.index,
                                  columns=df.columns)

    return pd_neutral


def iis_std_panel(dfx: pd.DataFrame, dates_iter: List[pd.Timestamp], min_obs: int,
                  sequential: bool = True, iis: bool = True):
    """
    Function designed to compute the standard deviations but accounts for in-sampling
    period. The in-sampling standard deviation will be a fixed value.

    :param <pd.DataFrame> dfx: DataFrame recording the differences from the neutral
        level.
    :param <List[pd.Timestamp]> dates_iter: controls the frequency of the neutral &
        standard deviation calculations.
    :param <int> min_obs:
    :param <bool> sequential:
    :param <bool> iis:

    :return <pd.DataFrame> df_sds: a DataFrame of daily standard deviations.
    """

    no_dates = dfx.shape[0]
    if sequential:

        # Each data point in the DataFrame is measuring the distance from neutral level
        # where the neutral level is computed on a rolling basis (computed across the
        # panel for each preceding date).
        # Therefore, take the absolute values and subsequently calculate the average
        # across the panel (inclusive of all previous dates).
        df_sds = func_executor(df=dfx.abs(), neutral='mean', n=dfx.shape[0],
                               dates_iter=dates_iter)
        if iis:
            iis_dfx = dfx.iloc[0:min_obs, :]
            iis_sds = np.array(iis_dfx.stack().abs().mean())
            df_sds.iloc[0:min_obs] = iis_sds
    else:
        df_sds = pd.DataFrame(np.repeat(dfx.stack().abs().mean(), no_dates),
                              index=dfx.index)

    df_sds.columns = ['value']
    return df_sds


def iis_std_cross(column: pd.Series, dates_iter: List[pd.Timestamp], cid: str,
                  min_obs: int, sequential: bool = True, iis: bool = True):
    """
    Standard deviation for cross-sectional zn_scores. Will account for the in-sampling
    period.

    :param <pd.Series> column: individual cross-section's deviation from the respective
        cross-section's neutral level.
    :param <List[pd.Timestamp]> dates_iter: controls the frequency of the neutral &
        standard deviation calculations.
    :param <str> cid: associated cross-section.
    :param <int> min_obs:
    :param <bool> sequential:
    :param <bool> iis:

    :return <pd.DataFrame> sds_df:
    """

    index = column.index
    date = column.first_valid_index()
    # Integer index at which the series' first realised value occurs.
    date_index = next(iter(np.where(index == date)[0]))

    # Inclusive of undefined dates.
    daily = column.size == len(dates_iter)
    no_dates = column.size

    if sequential:
        # The absolute difference from the neutral level, and subsequently compute the
        # rolling standard deviation.
        # The in-built pandas method, applied to a pd.Series, will only compute the
        # standard deviation from the active dates onwards. The preceding dates will
        # remain NaN values allowing the array to match the dimensions of the original
        # pivoted DataFrame.
        ar_sds = np.array([column.loc[column.index[0]:d].abs().mean()
                           for d in dates_iter])
        sds_df = pd.DataFrame(data=ar_sds, index=dates_iter)
        if not daily:
            sds_df = sds_df.fillna(method='ffill')

        if iis:
            iis_end_date = date_index + min_obs
            iis_column = column[date_index:iis_end_date]
            sds_df.iloc[date_index: iis_end_date] = iis_column.abs().mean()

    else:
        sds_value = np.array(column.abs().mean())
        sds_df = pd.DataFrame(data=np.repeat(sds_value, no_dates), index=column.index)

    sds_df.columns = [cid]
    return sds_df


def make_zn_scores(df: pd.DataFrame, xcat: str, cids: List[str] = None,
                   start: str = None, end: str = None, blacklist: dict = None,
                   sequential: bool = True, min_obs: int = 261,  iis: bool = True,
                   neutral: str = 'zero', est_freq: str = 'd', thresh: float = None,
                   pan_weight: float = 1, postfix: str = 'ZN'):

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

    df = df.loc[:, ['cid', 'xcat', 'real_date', 'value']]
    df = reduce_df(df, xcats=[xcat], cids=cids, start=start, end=end,
                   blacklist=blacklist)
    s_date = min(df['real_date'])
    e_date = max(df['real_date'])
    dates_iter = pd.date_range(start=s_date, end=e_date, freq=pd_freq[est_freq])

    dfw = df.pivot(index='real_date', columns='cid', values='value')
    cross_sections = dfw.columns

    if pan_weight > 0:

        df_neutral = pan_neutral(dfw, dates_iter, neutral, sequential, min_obs, iis)
        dfx = dfw.sub(df_neutral['value'], axis='rows')
        ar_sds = iis_std_panel(dfx, min_obs=min_obs, sequential=sequential, iis=iis,
                               dates_iter=dates_iter)
        dfw_zns_pan = dfx.div(ar_sds['value'], axis='rows')
    else:
        dfw_zns_pan = dfw * 0

    if pan_weight < 1:

        arr_neutral = cross_neutral(dfw, neutral, est_freq, sequential, min_obs, iis)
        dfx = dfw.sub(arr_neutral, axis='rows')
        pd_sds = pd.DataFrame(index=dfx.index)

        for i, c in enumerate(cross_sections):
            column = dfx.iloc[:, i]
            std_cross = iis_std_cross(column=column, dates_iter=dates_iter, cid=c,
                                      min_obs=min_obs, sequential=sequential,
                                      iis=iis)
            pd_sds = pd_sds.join(std_cross, on='real_date', how='left')

        dfw_zns_css = dfx.div(pd_sds, axis='rows')

    else:
        dfw_zns_css = dfw * 0

    dfw_zns = (dfw_zns_pan * pan_weight) + (dfw_zns_css * (1 - pan_weight))
    dfw_zns = dfw_zns.dropna(axis=0, how='all')
    
    if thresh is not None:
        dfw_zns.clip(lower=-thresh, upper=thresh, inplace=True)

    df_out = dfw_zns.stack().to_frame("value").reset_index()
    df_out['xcat'] = xcat + postfix

    col_names = ['cid', 'xcat', 'real_date', 'value']
    df_out = df_out.sort_values(['cid', 'real_date'])[col_names]

    return df_out[df.columns]


if __name__ == "__main__":

    cids = ['AUD', 'CAD', 'GBP', 'USD', 'NZD']
    xcats = ['XR', 'CRY', 'GROWTH', 'INFL']

    df_cids = pd.DataFrame(index = cids, columns = ['earliest', 'latest', 'mean_add',
                                                    'sd_mult'])

    df_cids.loc['AUD'] = ['2010-01-01', '2020-12-31', 0.5, 2]
    df_cids.loc['CAD'] = ['2011-01-01', '2020-11-30', 0, 1]
    df_cids.loc['GBP'] = ['2012-01-01', '2020-11-30', -0.2, 0.5]
    df_cids.loc['USD'] = ['2013-01-01', '2020-09-30', -0.2, 0.5]
    df_cids.loc['NZD'] = ['2002-01-01', '2020-09-30', -0.1, 2]

    df_xcats = pd.DataFrame(index = xcats, columns = ['earliest', 'latest', 'mean_add',
                                                      'sd_mult', 'ar_coef', 'back_coef'])
    df_xcats.loc['XR'] = ['2010-01-01', '2020-12-31', 0, 1, 0, 0.3]
    df_xcats.loc['CRY'] = ['2011-01-01', '2020-10-30', 1, 2, 0.9, 0.5]
    df_xcats.loc['GROWTH'] = ['2012-01-01', '2020-10-30', 1, 2, 0.9, 1]
    df_xcats.loc['INFL'] = ['2013-01-01', '2020-10-30', 1, 2, 0.8, 0.5]

    dfd = make_qdf(df_cids, df_xcats, back_ar = 0.75)

    df_output = make_zn_scores(dfd, xcat='XR', sequential=True, cids=cids, iis=True,
                               neutral='mean', pan_weight=1.0, min_obs=261,
                               est_freq="m")

    filt1 = dfd['xcat'] == 'XR'
    dfd = dfd[filt1]
    dfw = dfd.pivot(index='real_date', columns='cid', values='value')
    min_obs = 251
    df_mean = cross_neutral(dfw, neutral='mean', est_freq='m', sequential=True,
                            min_obs=min_obs, iis=False)

    daily_dates = pd.date_range(start='2010-01-01', end='2020-10-30', freq='m')

    df_mean = pan_neutral(df=dfw, dates_iter=daily_dates, neutral='mean',
                          sequential=True, min_obs=261, iis=True)

    df_output = make_zn_scores(dfd, xcat='XR', sequential=True, cids=cids, iis=True,
                               neutral='mean', pan_weight=1.0, min_obs=261,
                               est_freq="d")