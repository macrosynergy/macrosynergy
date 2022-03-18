
import numpy as np
import pandas as pd
from typing import List, Union
import statsmodels.api as sm
from macrosynergy.management.simulate_quantamental_data import make_qdf
from macrosynergy.management.shape_dfs import reduce_df


def refreq_groupby(start_date: pd.Timestamp, end_date: pd.Timestamp, refreq: str = 'm'):
    """
    The hedging ratio is re-estimated according to the frequency parameter. Therefore,
    break up the respective return series, which are defined daily, into the re-estimated
    frequency paradigm. To achieve this ensure the dates produced fall on business days,
    and will subsequently be present in the return-series dataframes.

    :param <pd.Timestamp> start_date:
    :param <pd.Timestamp> end_date:
    :param <str> refreq:

    return <List[pd.Timestamp]>: List of timestamps where each date is a valid business
        day, and the gap between each date is delimited by the frequency parameter.
    """

    dates = pd.date_range(start_date, end_date, freq=refreq)
    d_copy = list(dates)
    condition = lambda date: date.dayofweek > 4

    for i, d in enumerate(dates):
        if condition(d):
            new_date = d + pd.DateOffset(1)
            while condition(new_date):
                new_date += pd.DateOffset(1)

            d_copy.remove(d)
            d_copy.insert(i, new_date)
        else:
            continue

    return d_copy

def hedge_calculator(main_asset: pd.DataFrame, hedging_asset: pd.Series,
                     groups: List[pd.Timestamp], cross_section: str):
    """
    The hedging of a contract can be achieved by taking positions across an entire panel.
    Therefore, compute the hedge ratio for each cross-section across the defined panel.
    The sample of data used for hedging will increase according to the dates parameter:
    each date represents an additional number of timestamps where the numeracy is
    instructed by the "refreq" parameter.

    :param <pd.DataFrame> main_asset: the return series of the asset that is being
        hedged.
    :param <pd.Series> hedging_asset: the return series of the asset being used to hedge
        against the main asset.
    :param <List[pd.Timestamp]> groups: the dates controlling the frequency of
        re-estimation.
    :param <str> cross_section: cross-section responsible for the "hedging_asset" series.

    :return <pd.DataFrame>: returns a dataframe of the hedge ratios for the respective
        cross-section.
    """

    hedging_ratio = []
    main_asset = pd.Series(data=main_asset['value'], index=main_asset['real_date'])

    for d in groups[1:]:
        evolving_independent = main_asset.loc[:d]
        hedging_sample = hedging_asset.loc[:d]

        mod = sm.OLS(evolving_independent, hedging_sample)
        results = mod.fit()
        hedging_ratio.append(results.rsquared)

    no_dates = len(groups)
    cid = np.repeat(no_dates, cross_section)
    dates = np.array(groups)
    data = np.column_stack((cid, dates, np.array(hedging_ratio)))

    return pd.DataFrame(data=data, columns=['cids', 'real_date', 'value'])

def hedge_ratio(df: pd.DataFrame, xcat: str = None, cids: List[str] = None,
                hedge_return: str = None, start: str = None, end: str = None,
                blacklist: dict = None, meth: str = 'ols', oos: bool = True,
                refreq: str = 'm', min_obs: int = 24):

    """
    Return dataframe of hedge ratios for one or more return categories.
    
    :param <pd.Dataframe> df: standardized data frame with the following necessary columns:
        'cid', 'xcats', 'real_date' and 'value. Will contain all of the data across all
        macroeconomic fields.
    :param <str> xcat:  extended category denoting the return series for which the
        hedge ratios are calculated. In order to hedge against the main asset, compute
        hedging ratios across the panel. For instance, a possible strategy would be to
        hedge a range of local equity index positions against the S&P 500. The main asset
        is defined by the parameter "hedge_return": in the above example it would be
        represented by the postfix USD_EQ.
    :param <List[str]> cids: cross sections for which hedge ratios are calculated;
        default is all available for the category.
    :param <str> hedge_return: ticker of return of the hedge asset or basket. The
        parameter represents a single series. For instance, "USD_EQ".
    :param <str> start: earliest date in ISO format. Default is None and earliest date in
        df is used.
    :param <str> end: latest date in ISO format. Default is None and latest date in df is
        used.
    :param <dict> blacklist: cross-sections with date ranges that should be excluded from
        the sample of data used for the rolling regression. If an entire period of data
        is excluded by the blacklisting, where a period is defined by the frequency
        parameter "refreq", the regression coefficient will remain constant for that
        period, or all periods coinciding with the blacklist.
    :param <bool> oos: if True (default) hedge ratio are calculated out-of-sample,
        i.e. for the period subsequent to the estimation period at the given
        re-estimation frequency.
    :param <str> refreq: re-estimation frequency. Frequency at which hedge ratios are
        re-estimated. The re-estimation is conducted at the end of the period and
        fills all days of the following period. The re-estimation can be computed on a
        weekly, monthly, and quarterly frequency with the notation 'w', 'm', and 'q'
        respectively. The default frequency is monthly.
    :param <int> min_obs: the minimum number of observations required in order to start
        classifying a hedge ratio between the assets. The default value is 24 timestamps
        (business days).
    :param <str> meth: method to estimate hedge ratio. At present the only method is
        OLS regression.
    
    N.B.: A hedge ratio is defined as the sensitivity of the main return with respect
    to the asset used for hedging.
    
    """

    cols = ['cid', 'xcat', 'real_date', 'value']
    assert list(df.columns) == cols, f"Expects a standardised dataframe with columns: " \
                                     f"{cols}."

    post_fix = hedge_return.split('_')
    error_hedge = "The expected form of the 'hedge_return' parameter is <cid_xcat>. The" \
                  "parameter expects to define a single series."
    assert len(post_fix) == 2, error_hedge

    xcat_hedge = post_fix[1]
    cid_hedge = post_fix[0]
    available_categories = df['xcat'].unique()
    xcat_error = f"Category, {xcat_hedge}, not defined in the dataframe. Available " \
                 f"categories are: {available_categories}."
    assert xcat_hedge in available_categories, xcat_error

    error_hedging = f"The category used to hedge against the primary asset, {xcat}, is " \
                    f"not defined in the dataframe."
    assert set(xcats).issubset(set(available_categories)), error_hedging

    refreq_options = ['w', 'm', 'q']
    error_refreq = f"The re-estimation frequency parameter must be one of the following:" \
                   f"{refreq_options}."
    assert refreq in refreq_options, error_refreq

    df_copy = df.copy()
    hedge_series = reduce_df(df_copy, xcats=[xcat_hedge], cids=cid_hedge, start=start,
                             end=end, blacklist=blacklist)

    if xcat_hedge == xcat:
        cids.remove(cid_hedge)

    dfd = reduce_df(df, xcats=[xcat], cids=cids, start=start, end=end,
                    blacklist=blacklist)

    dfw = dfd.pivot(index='real_date', columns='cid', values='value')
    dates = dfw.index

    dates = refreq_groupby(start_date=dates[0], end_date=dates[-1],
                           refreq=refreq)

    # A "rolling" hedge ratio is computed for each cross-section across the defined
    # panel.
    aggregate = []
    for c in cids:
        series = dfw[c]
        hedge_data = hedge_calculator(main_asset=hedge_series, hedging_asset=series,
                                      groups=dates, cross_section=c)
        aggregate.append(hedge_data)

    return dfd


if __name__ == "__main__":
    cids = ['AUD', 'CAD', 'GBP', 'USD', 'NZD']
    xcats = ['FXXR', 'GROWTHXR', 'INFLXR', 'EQXR']

    df_cids = pd.DataFrame(index=cids, columns=['earliest', 'latest', 'mean_add',
                                                'sd_mult'])

    df_cids.loc['AUD'] = ['2010-01-01', '2020-12-31', 0.5, 2]
    df_cids.loc['CAD'] = ['2011-01-01', '2020-11-30', 0, 1]
    df_cids.loc['GBP'] = ['2012-01-01', '2020-11-30', -0.2, 0.5]
    df_cids.loc['USD'] = ['2013-01-01', '2020-09-30', -0.2, 0.5]
    df_cids.loc['NZD'] = ['2002-01-01', '2020-09-30', -0.1, 2]

    df_xcats = pd.DataFrame(index=xcats, columns=['earliest', 'latest', 'mean_add',
                                                  'sd_mult', 'ar_coef', 'back_coef'])

    df_xcats.loc['FXXR'] = ['2010-01-01', '2020-10-30', 1, 2, 0.9, 1]
    df_xcats.loc['GROWTHXR'] = ['2012-01-01', '2020-10-30', 1, 2, 0.9, 1]
    df_xcats.loc['INFLXR'] = ['2013-01-01', '2020-10-30', 1, 2, 0.8, 0.5]
    df_xcats.loc['EQXR'] = ['2010-01-01', '2022-03-14', 0.5, 2, 0, 0.2]

    dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)
    black = {'AUD': ['2010-01-01', '2013-12-31'], 'GBP': ['2010-01-01', '2013-12-31']}

    xcat_hedge = "EQXR"
    # S&P500.
    hedge_return = "USD_EQXR"
    df_hedge = hedge_ratio(df=dfd, xcat=xcat_hedge, cids=cids,
                           hedge_return=hedge_return, start='2010-01-01',
                           end='2020-10-30',
                           blacklist=black, meth='ols', oos=True,
                           refreq='m', min_obs=24)