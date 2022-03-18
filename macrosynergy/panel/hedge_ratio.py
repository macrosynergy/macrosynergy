
import numpy as np
import pandas as pd
from typing import List, Union
from macrosynergy.management.simulate_quantamental_data import make_qdf
from macrosynergy.management.shape_dfs import reduce_df

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

    xcat_hedge = post_fix[0]
    cid_hedge = post_fix[1]
    available_categories = df['xcat'].unique()
    xcat_error = f"Category not defined in the dataframe. Available categories are: " \
                 f"{available_categories}."
    assert xcat_hedge in available_categories, xcat_error

    error_hedging = f"The category used to hedge against the primary asset, {xcat}, is " \
                    f"not defined in the dataframe."
    assert set(xcats).issubt(available_categories), error_hedging

    refreq_options = ['w', 'm', 'q']
    error_refreq = f"The re-estimation frequency parameter must be one of the following:" \
                   f"{refreq_options}."
    assert refreq in refreq_options, error_refreq

    df_copy = df.copy()
    hedge_series = reduce_df(df_copy, xcats=xcat_hedge, cids=cid_hedge, start=start,
                             end=end, blacklist=blacklist)

    if xcat_hedge == xcat:
        cids.remove(cid_hedge)

    dfd = reduce_df(df, xcats=[xcat], cids=cids, start=start, end=end,
                    blacklist=blacklist)

    dfw = dfd.pivot(index='real_date', columns='cid', values='value')

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
                hedge_return=hedge_return, start='2010-01-01', end='2020-10-30',
                blacklist=black, meth='ols', oos=True,
                refreq='m', min_obs=24)