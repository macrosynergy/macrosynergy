
"""Module with functions to simulate data for testing."""
import numpy as np
import pandas as pd
from typing import List, Union, Tuple
from statsmodels.tsa.arima_process import ArmaProcess
import random
from macrosynergy.management.shape_dfs import reduce_df_by_ticker
from itertools import chain

def simulate_ar(nobs: int, mean: float = 0, sd_mult: float = 1, ar_coef: float = 0.75):
    """
    Create an auto-correlated data-series as Numpy Array.

    :param <int> nobs: number of observations.
    :param <float> mean: mean of values, default is zero.
    :param <float> sd_mult: standard deviation multipliers of values, default is 1.
        This affects non-zero means.
    :param <float> ar_coef: auto-regression coefficient (between 0 and 1): default is
        0.75.

    return <np.array>: auto-correlated data series.
    """

    ar_params = np.r_[
        1, -ar_coef]
    ar_proc = ArmaProcess(ar_params)
    ser = ar_proc.generate_sample(nobs)
    ser = ser + mean - np.mean(ser)
    return sd_mult * ser / np.std(ser)


def make_qdf(df_cids: pd.DataFrame, df_xcats: pd.DataFrame, back_ar: float = 0):
    """
    Make quantamental DataFrame with basic columns: 'cid', 'xcat', 'real_date', 'value'

    :param <pd.DataFrame> df_cids: dataframe with parameters by cid. Row indices are
        cross-sections. Columns are:
    'earliest': string of earliest date (ISO) for which country values are available;
    'latest': string of latest date (ISO) for which country values are available;
    'mean_add': float of country-specific addition to any category's mean;
    'sd_mult': float of country-specific multiplier of an category's standard deviation.
    :param <pd.DataFrame> df_xcats: dataframe with parameters by xcat. Row indices are
        cross-sections. Columns are:
    'earliest': string of earliest date (ISO) for which category values are available;
    'latest': string of latest date (ISO) for which category values are available;
    'mean_add': float of category-specific addition;
    'sd_mult': float of country-specific multiplier of an category's standard deviation;
    'ar_coef': float between 0 and 1 denoting set autocorrelation of the category;
    'back_coef': float, coefficient with which communal (mean 0, SD 1) background factor
                 is added to category values.
    :param <float> back_ar: float between 0 and 1 denoting set autocorrelation of the
        background factor. Default is zero.

    :return <pd.DataFrame>: basic quantamental dataframe according to specifications.

    """
    qdf_cols = ['cid', 'xcat', 'real_date', 'value']
    df_out = []

    if any(df_xcats['back_coef'] != 0):
        sdate = min(min(df_cids.loc[:, 'earliest']), min(df_xcats.loc[:, 'earliest']))
        edate = max(max(df_cids.loc[:, 'latest']), max(df_xcats.loc[:, 'latest']))
        all_days = pd.date_range(sdate, edate)
        work_days = all_days[all_days.weekday < 5]
        ser = simulate_ar(len(work_days), mean=0, sd_mult=1, ar_coef=back_ar)
        df_back = pd.DataFrame(index=work_days, columns=['value'])
        df_back['value'] = ser

    for cid in df_cids.index:
        for xcat in df_xcats.index:

            df_add = pd.DataFrame(columns=qdf_cols)

            sdate = pd.to_datetime(max(df_cids.loc[cid, 'earliest'],
                                       df_xcats.loc[xcat, 'earliest']))
            edate = pd.to_datetime(min(df_cids.loc[cid, 'latest'],
                                       df_xcats.loc[xcat, 'latest']))
            all_days = pd.date_range(sdate, edate)
            work_days = all_days[all_days.weekday < 5]

            df_add['real_date'] = work_days
            df_add['cid'] = cid
            df_add['xcat'] = xcat

            ser_mean = df_cids.loc[cid, 'mean_add'] + df_xcats.loc[xcat, 'mean_add']
            ser_sd = df_cids.loc[cid, 'sd_mult'] * df_xcats.loc[xcat, 'sd_mult']
            ser_arc = df_xcats.loc[xcat, 'ar_coef']
            df_add['value'] = simulate_ar(len(work_days), mean=ser_mean, sd_mult=ser_sd,
                                          ar_coef=ser_arc)

            back_coef = df_xcats.loc[xcat, 'back_coef']
            if back_coef != 0:
                df_add['value'] = df_add['value'] + \
                                  back_coef * df_back.loc[df_add['real_date'],
                                                          'value'].reset_index(drop=True)

            df_out.append(df_add)

    return pd.concat(df_out).reset_index(drop=True)


def dataframe_basket(ret = 'XR_NSA', cry = ['CRY_NSA'], start = None, end = None,
                     black = None):

    cids = ['AUD', 'GBP', 'NZD', 'USD']
    xcats = ['FXXR_NSA', 'FXCRY_NSA', 'FXCRR_NSA', 'EQXR_NSA', 'EQCRY_NSA', 'EQCRR_NSA',
             'FXWBASE_NSA', 'EQWBASE_NSA']

    df_cids = pd.DataFrame(index=cids, columns=['earliest', 'latest', 'mean_add',
                                                'sd_mult'])

    df_cids.loc['AUD'] = ['2010-12-01', '2020-12-31', 0, 1]
    df_cids.loc['GBP'] = ['2011-01-01', '2020-11-30', 0, 2]
    df_cids.loc['NZD'] = ['2012-01-01', '2020-11-30', 0, 3]
    df_cids.loc['USD'] = ['2013-01-01', '2020-09-30', 0, 4]

    df_xcats = pd.DataFrame(index=xcats, columns=['earliest', 'latest', 'mean_add',
                                                  'sd_mult', 'ar_coef', 'back_coef'])

    df_xcats.loc['FXXR_NSA'] = ['2010-01-01', '2020-12-31', 0, 1, 0, 0.2]
    df_xcats.loc['FXCRY_NSA'] = ['2010-01-01', '2020-12-31', 1, 1, 0.9, 0.2]
    df_xcats.loc['FXCRR_NSA'] = ['2010-01-01', '2020-12-31', 0.5, 0.8, 0.9, 0.2]
    df_xcats.loc['EQXR_NSA'] = ['2012-01-01', '2020-12-31', 0.5, 2, 0, 0.2]
    df_xcats.loc['EQCRY_NSA'] = ['2010-01-01', '2020-12-31', 2, 1.5, 0.9, 0.5]
    df_xcats.loc['EQCRR_NSA'] = ['2010-01-01', '2020-12-31', 1.5, 1.5, 0.9, 0.5]
    df_xcats.loc['FXWBASE_NSA'] = ['2010-01-01', '2020-12-31', 1, 1.5, 0.8, 0.5]
    df_xcats.loc['EQWBASE_NSA'] = ['2010-01-01', '2020-12-31', 1, 1.5, 0.9, 0.5]

    random.seed(2)
    dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)

    contracts = ['AUD_FX', 'AUD_EQ', 'NZD_FX', 'GBP_EQ', 'USD_EQ']
    ticks_cry = []
    for c_ in cry:

        ticks_cry.append([c + c_ for c in contracts])

    ticks_ret = [c + ret for c in contracts]
    tickers = ticks_ret + list(chain.from_iterable(ticks_cry))
    dfx = reduce_df_by_ticker(dfd, start=start, end=end,
                              blacklist=black, ticks=tickers)

    dfw_ret = dfx[dfx["ticker"].isin(ticks_ret)].pivot(index="real_date",
                                                       columns="ticker", values="value")
    dfw_cry_list = []
    for t in ticks_cry:
        dfw_cry = dfx[dfx["ticker"].isin(t)].pivot(index="real_date",
                                                   columns="ticker", values="value")
        dfw_cry_list.append(dfw_cry)
    if len(dfw_cry_list) == 1:
        dfw_cry_list = next(iter(dfw_cry_list))

    return dfw_ret, dfw_cry_list, dfd


# DataFrame used for more scrupulous, thorough testing.
def construct_df():
    weights = [random.random() for i in range(65)]
    weights = np.array(weights)
    weights = weights.reshape((13, 5))

    weights[0:4, 0] = np.nan
    weights[-3:, 1] = np.nan
    weights[-6:, 2] = np.nan
    weights[-2:, -1] = np.nan
    weights[:3, -1] = np.nan

    sum_ = np.nansum(weights, axis=1)
    sum_ = sum_[:, np.newaxis]

    weights = np.divide(weights, sum_)
    cols = ['col_' + str(i + 1) for i in range(weights.shape[1])]
    pseudo_df = pd.DataFrame(data=weights, columns=cols)

    return pseudo_df


if __name__ == "__main__":
    ser_ar = simulate_ar(100, mean=0, sd_mult=1, ar_coef=0.75)

    cids = ['AUD', 'CAD', 'GBP']
    xcats = ['XR', 'CRY']
    df_cids = pd.DataFrame(index=cids,
                           columns=['earliest', 'latest', 'mean_add', 'sd_mult'])
    df_cids.loc['AUD'] = ['2010-01-01', '2020-12-31', 0.5, 2]
    df_cids.loc['CAD'] = ['2011-01-01', '2020-11-30', 0, 1]
    df_cids.loc['GBP'] = ['2011-01-01', '2020-11-30', -0.2, 0.5]

    df_xcats = pd.DataFrame(index=xcats,
                            columns=['earliest', 'latest', 'mean_add', 'sd_mult',
                                     'ar_coef', 'back_coef'])
    df_xcats.loc['XR'] = ['2010-01-01', '2020-12-31', 0, 1, 0, 0.3]
    df_xcats.loc['CRY'] = ['2011-01-01', '2020-10-30', 1, 2, 0.9, 0.5]

    dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)

    dfw_ret, dfw_cry, dfd = dataframe_basket()
