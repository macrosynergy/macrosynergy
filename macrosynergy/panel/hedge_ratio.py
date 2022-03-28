import warnings

import numpy as np
import pandas as pd
from typing import List, Union
import statsmodels.api as sm
from macrosynergy.management.simulate_quantamental_data import make_qdf
from macrosynergy.management.shape_dfs import reduce_df
import matplotlib.pyplot as plt

def date_alignment(unhedged_return: pd.Series, benchmark_return: pd.Series):
    """
    Method used to align the two Series over the same timestamps: the sample data for the
    endogenous & exogenous variables must match throughout the re-estimation calculation.

    :param <pd.DataFrame> unhedged_return: the return series of the asset that is being
        hedged.
    :param <pd.Series> benchmark_return: the return series of the asset being used to hedge
        against the main asset.

    :return <pd.Timestamp, pd.Timestamp>: the shared start and end date across the two
        series.
    """
    ma_dates = unhedged_return.index
    ha_dates = benchmark_return.index

    if ma_dates[0] > ha_dates[0]:
        start_date = ma_dates[0]
    else:
        start_date = ha_dates[0]

    if ma_dates[-1] > ha_dates[-1]:
        end_date = ha_dates[-1]
    else:
        end_date = ma_dates[-1]

    return start_date, end_date


def hedge_calculator(unhedged_return: pd.DataFrame, benchmark_return: pd.Series,
                     rdates: List[pd.Timestamp], cross_section: str, min_obs: int = 24):
    """
    Calculate the hedge ratios for each cross-section in the panel being hedged. It is
    worth noting that the sample of data used for calculating the hedge ratio will
    increase according to the dates parameter: each date represents an additional number
    of timestamps where the numeracy of dates added to the sample is instructed by the
    "refreq" parameter.

    :param <pd.DataFrame> unhedged_return: the return series of the asset that is being
        hedged.
    :param <pd.Series> benchmark_return: the return series of the asset being used to hedge
        against the main asset.
    :param <List[pd.Timestamp]> rdates: the dates controlling the frequency of
        re-estimation.
    :param <str> cross_section: cross-section responsible for the "benchmark_return" series.
    :param <int> min_obs: a hedge ratio will only be computed if the number of days has
        surpassed the integer held by the parameter.

    :return <pd.DataFrame>: returns a dataframe of the hedge ratios for the respective
        cross-section.
    """

    hedging_ratio = []

    benchmark_return = benchmark_return[benchmark_return.first_valid_index():]
    s_date, e_date = date_alignment(unhedged_return=unhedged_return,
                                    benchmark_return=benchmark_return)

    unhedged_return = unhedged_return.truncate(before=s_date, after=e_date)
    benchmark_return = benchmark_return.truncate(before=s_date, after=e_date)
    date_adjustment = lambda computed_date: computed_date + pd.DateOffset(1)

    date_series = unhedged_return.index
    min_obs_date = date_series[min_obs]

    rdates_copy = list(rdates.copy())
    for d in rdates:
        if d > min_obs_date:
            Y = unhedged_return.loc[:d]
            X = benchmark_return.loc[:d]
            X = sm.add_constant(X)

            mod = sm.OLS(Y, X)
            results = mod.fit()
            # Isolate the Beta coefficient to use as the hedging component.
            hedging_ratio.append(results.params[1])
        else:
            rdates_copy.remove(d)


    no_dates = len(rdates_copy)
    cid = np.repeat(cross_section, no_dates)
    dates_hedge = list(map(date_adjustment, rdates_copy))

    dates_hedge = np.array(dates_hedge)
    data = np.column_stack((cid, dates_hedge, np.array(hedging_ratio)))

    return pd.DataFrame(data=data, columns=['cid', 'real_date', 'value'])

def dates_groups(dates_refreq: List[pd.Timestamp], benchmark_return: pd.Series):
    """
    Method used to break up the hedging asset's return series into the re-estimation
    periods. The method will return a dictionary where the key will be the re-estimation
    timestamp and the corresponding value will be the following timestamps until the
    next re-estimation date. It is the following returns that the hedge ratio is applied
    to.

    :param <List[pd.Timestamp]> dates_refreq:
    :param <pd.Series> benchmark_return:

    :return <dict>:
    """
    refreq_buckets = {}

    no_reest_dates = len(dates_refreq)
    for i, d in enumerate(dates_refreq):
        if i < (no_reest_dates - 1):
            intermediary_series = benchmark_return.truncate(before=d,
                                                      after=dates_refreq[(i + 1)])
            refreq_buckets[d + pd.DateOffset(1)] = intermediary_series

    return refreq_buckets

def adjusted_returns(dates_refreq: List[pd.Timestamp], benchmark_return: pd.Series,
                     hedge_df: pd.DataFrame, dfw: pd.DataFrame):
    """
    Method used to compute the hedge ratio returns on the hedging asset which will
    subsequently be subtracted from the returns of the position contracts to calculate
    the adjusted returns (adjusted for the hedged position). For instance, if using US
    Equity to hedge Australia FX: AUD_FXXR_NSA_H = AUD_FXXR_NSA - HR_AUD * USD_EQXR_NSA.

    :param <List[pd.Timestamps]> dates_refreq: list of dates the hedge ratio is
        recomputed for each contract being hedged.
    :param <pd.Series> benchmark_return: return series of the hedging asset.
    :param <pd.DataFrame> hedge_df: standardised dataframe with the hedge ratios.
    :param <pd.DataFrame> dfw: pivoted dataframe of the relevant returns.

    :return <pd.DataFrame> standardised dataframe of adjusted returns.
    """

    refreq_buckets = dates_groups(dates_refreq=dates_refreq,
                                  benchmark_return=benchmark_return)
    hedge_pivot = hedge_df.pivot(index='real_date', columns='cid', values='value')

    storage_dict = {}
    for c in hedge_pivot:
        series_hedge = hedge_pivot[c]
        storage = []
        for k, v in refreq_buckets.items():
            try:
                hedge_value = series_hedge.loc[k]
            except KeyError:
                pass
            else:
                hedged_position = v * hedge_value
                storage.append(hedged_position)
        storage_dict[c] = pd.concat(storage)

    hedged_returns_df = pd.DataFrame.from_dict(storage_dict)
    hedged_returns_df.index.name = "real_date"

    output = dfw - hedged_returns_df
    df_stack = output.stack().to_frame("value").reset_index()
    df_stack.columns = ['real_date', 'cid', 'value']

    return df_stack

def hedge_ratio(df: pd.DataFrame, xcat: str = None, cids: List[str] = None,
                benchmark_return: str = None, start: str = None, end: str = None,
                blacklist: dict = None, meth: str = 'ols', oos: bool = True,
                # Todo: meth and oos not used in function
                refreq: str = 'm', min_obs: int = 24, hedged_returns: bool = False):

    """
    Return dataframe of hedge ratios for one or more return categories.
    
    :param <pd.Dataframe> df: standardized data frame with the necessary columns:
        'cid', 'xcats', 'real_date' and 'value.
    :param <str> xcat:  extended category denoting the return series for which the
        hedge ratios are calculated. Thus, the positions of a panel of returns
        are hedged against a single hedge position.
    :param <List[str]> cids: cross sections of the returns for which hedge ratios are
        to be calculated. Default is all available for the category.
    :param <str> benchmark_return: ticker of return of the hedge asset or basket. This is
        a single series, such as "USD_EQXR_NSA".
    :param <str> start: earliest date in ISO format. Default is None and earliest date in
        df is used.
    :param <str> end: latest date in ISO format. Default is None and latest date in df is
        used.
    :param <dict> blacklist: cross-sections with date ranges that should be excluded from
        the sample of data used for estimating hedge ratios.  The estimated ratios
        during blacklist periods will be set equal to the last valid estimate.
    :param <bool> oos: if True (default) hedge ratios are calculated out-of-sample,
        i.e. for the period subsequent to the estimation period at the given
        re-estimation frequency.
    :param <str> refreq: re-estimation frequency at which hedge ratios are periodically
        re-estimated. The re-estimation is conducted at the end of the period and
        used as hedge ratio  all days of the following period. Re-estimation can have
        weekly, monthly, and quarterly frequency with the notation 'w', 'm', and 'q'
        respectively. The default frequency is monthly.
    :param <int> min_obs: the minimum number of observations required in order to
        estimate a hedge ratio. The default value is 24 days.
    :param <str> meth: method to estimate hedge ratio. At present the only method is
        OLS regression ('ols').
    :param <bool> hedged_returns: append the hedged returns to the dataframe.

    :return <pd.Dataframe> hedge_df: dataframe with hedge ratios which are based on an
        estimation using prior data at the set re-estimation frequency.
        Additionally,  the dataframe can include the hedged returns if the parameter
        "benchmark_return" has been set to True.

    N.B.: A hedge ratio is the estimated sensitivity of the main return with respect to
    the asset used for hedging. The ratio is recorded for the period after the estimation
    sample up the next update.
    
    """

    cols = ['cid', 'xcat', 'real_date', 'value']
    assert list(df.columns) == cols, f"Expects a standardised dataframe with columns: " \
                                     f"{cols}."

    post_fix = benchmark_return.split('_')
    xcat_hedge = '_'.join(post_fix[1:])
    cid_hedge = post_fix[0]
    available_categories = df['xcat'].unique()
    xcat_error = f"Category, {xcat_hedge}, not defined in the dataframe. Available " \
                 f"categories are: {available_categories}."
    assert xcat_hedge in available_categories, xcat_error
    # Todo: check if ticker is available, not category

    error_xcat = f"The field, xcat, must be a string but received <{type(xcat)}>. Only" \
                 f" a single category is used to hedge against the main asset."
    assert isinstance(xcat, str), error_xcat

    error_hedging = f"The return category used to be hedged, {xcat}, is " \
                    f"not defined in the dataframe."
    assert xcat in list(available_categories), error_hedging

    refreq_options = ['w', 'm', 'q']
    error_refreq = f"Re-estimation frequency parameter must be one of the following:" \
                   f"{refreq_options}."
    assert refreq in refreq_options, error_refreq

    if xcat_hedge == xcat:
        cids.remove(cid_hedge)  # if hedged is main asset type its cid cannot be hedged
        warnings.warn("One of the returns to be hedged is the hedge return"
                      "and has been removed from the panel")

    dfp = reduce_df(df, xcats=[xcat], cids=cids, start=start, end=end,
                    blacklist=blacklist)
    dfw = dfp.pivot(index='real_date', columns='cid', values='value')
    dfw = dfw.dropna(axis=0, how="all")

    dfh = reduce_df(df, xcats=[xcat_hedge], cids=cid_hedge,
                    start=dfw.index[0], end=dfw.index[-1], blacklist=blacklist)
    hedge = dfh.pivot(index='real_date', columns='cid', values='value')
    hedge.columns = ['hedge']
    dfw = pd.merge(dfw, hedge, how='inner', on='real_date')
    br = dfw['hedge']

    rf = {'w': 'W', 'm': 'BM', 'q': 'BQ'}[refreq]
    dates_re = dfw.asfreq(rf).index

    aggregate = []
    for c in cids:
        xr = dfw[c]
        df_hr = hedge_calculator(unhedged_return=xr, benchmark_return=br,
                                 rdates=dates_re, cross_section=c, min_obs=min_obs)
        aggregate.append(df_hr)

    hedge_df = pd.concat(aggregate).reset_index(drop=True)
    hedge_df['xcat'] = xcat
    if hedged_returns:
        hedged_return_df = adjusted_returns(dates_refreq=dates_re, benchmark_return=br,
                                            hedge_df=hedge_df, dfw=dfw)
        hedged_return_df = hedged_return_df.sort_values(['cid', 'real_date'])
        hedged_return_df['xcat'] = xcat + "_" + "H"
        hedge_df = hedge_df.append(hedged_return_df)
        hedge_df = hedge_df.reset_index(drop=True)

    return hedge_df[cols]

def hedge_ratio_display(df_hedge: pd.DataFrame, subplots: bool = False):
    """
    Method used to visualise the hedging ratios across the panel: assumes a single
    category is used to hedge the primary asset.

    :param <pd.DataFrame> df_hedge: dataframe with hedge ratios.
    :param <bool> subplots: matplotlib parameter to determine if each hedging series is
        displayed on separate subplots.

    """

    condition = lambda c: c.split('_')[-1] != 'H'
    # Isolate the hedge ratios. The adjusted returns will have the postfix "H" attached
    # to the category name.
    apply = list(map(condition, df_hedge['xcat']))
    df_hedge = df_hedge[apply]

    dfw_ratios = df_hedge.pivot(index='real_date', columns='cid', values='value')

    dfw_ratios.plot(subplots=subplots, title="Hedging Ratios.",
                    legend=True)
    plt.xlabel('real_date, years')
    plt.show()


if __name__ == "__main__":
    # Emerging Market Asian countries.
    cids = ['IDR', 'INR', 'KRW', 'MYR', 'PHP']
    # Add the US - used as the hedging asset.
    cids += ['USD']
    xcats = ['FXXR_NSA', 'GROWTHXR_NSA', 'INFLXR_NSA', 'EQXR_NSA']

    df_cids = pd.DataFrame(index=cids, columns=['earliest', 'latest', 'mean_add',
                                                'sd_mult'])

    df_cids.loc['IDR'] = ['2010-01-01', '2020-12-31', 0.5, 2]
    df_cids.loc['INR'] = ['2011-01-01', '2020-11-30', 0, 1]
    df_cids.loc['KRW'] = ['2012-01-01', '2020-11-30', -0.2, 0.5]
    df_cids.loc['MYR'] = ['2013-01-01', '2020-09-30', -0.2, 0.5]
    df_cids.loc['PHP'] = ['2002-01-01', '2020-09-30', -0.1, 2]
    df_cids.loc['USD'] = ['2000-01-01', '2022-03-14', 0, 1.25]

    df_xcats = pd.DataFrame(index=xcats, columns=['earliest', 'latest', 'mean_add',
                                                  'sd_mult', 'ar_coef', 'back_coef'])

    df_xcats.loc['FXXR_NSA'] = ['2010-01-01', '2020-10-30', 1, 2, 0.9, 1]
    df_xcats.loc['GROWTHXR_NSA'] = ['2012-01-01', '2020-10-30', 1, 2, 0.9, 1]
    df_xcats.loc['INFLXR_NSA'] = ['2013-01-01', '2020-10-30', 1, 2, 0.8, 0.5]
    df_xcats.loc['EQXR_NSA'] = ['2010-01-01', '2022-03-14', 0.5, 2, 0, 0.2]

    dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)
    black = {'IDR': ['2010-01-01', '2014-01-04'], 'INR': ['2010-01-01', '2013-12-31']}

    xcat_hedge = "EQXR_NSA"
    # S&P500.
    benchmark_return = "USD_EQXR_NSA"
    df_hedge = hedge_ratio(df=dfd, xcat=xcat_hedge, cids=cids,
                           benchmark_return=benchmark_return, start='2010-01-01',
                           end='2020-10-30',
                           blacklist=black, meth='ols', oos=True,
                           refreq='m', min_obs=24, hedged_returns=True)
    print(df_hedge)
    hedge_ratio_display(df_hedge=df_hedge, subplots=False)

    # Long position in S&P500 or the Nasdeq, and subsequently using US FX to hedge the
    # long position.
    xcats = 'FXXR_NSA'
    cids = ['USD']
    benchmark_return = "USD_EQXR_NSA"
    xcat_hedge_two = hedge_ratio(df=dfd, xcat=xcats, cids=cids,
                                 benchmark_return=benchmark_return, start='2010-01-01',
                                 end='2020-10-30',
                                 blacklist=black, meth='ols', oos=True,
                                 refreq='m', min_obs=24)
    print(xcat_hedge_two)