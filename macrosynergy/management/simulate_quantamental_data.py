import numpy as np
import pandas as pd
from typing import List, Union, Tuple
from statsmodels.tsa.arima_process import ArmaProcess
import matplotlib.pyplot as plt
import datetime


def simulate_ar(nobs, mean=0, sd=1, ar_coef=0.75):
    """Create an autocorrelated time series"""

    ar_params = np.r_[1, -ar_coef]  # define relative parameters for creating an AR process
    ar_proc = ArmaProcess(ar_params)  # define ARMA process
    ser = ar_proc.generate_sample(nobs)
    ser = ser + mean - np.mean(ser)
    return sd * ser / np.std(ser)


def simulate_cors(nobs, means=[0, 0], sds=[1, 1], cor_coef=0.2):
    """Create two correlated standard normal series"""

    cor_mat = np.array([[1, cor_coef], [cor_coef, 1]])
    cov_mat = cor_mat * np.outer(sds, sds)
    sers = np.random.multivariate_normal(means, cov_mat, size=nobs)
    return sers


def contaminate(sers, back_ser, snb_coefs=0):
    """Add background series to create correlations"""

    for i in len(sers):
        ser = back_ser


def make_qdf(df_cids, df_xcats, back_ar=0):
    """
    Make quantamental dataframe

    """
    qdf_cols = ['cid', 'xcat', 'real_date', 'value']
    df_out = pd.DataFrame(columns=qdf_cols)

    if any(df_xcats['back_coef'] != 0):

        sdate = min(min(df_cids.loc[:, 'earliest']), min(df_xcats.loc[:, 'earliest']))
        edate = max(max(df_cids.loc[:, 'latest']), max(df_xcats.loc[:, 'latest']))
        all_days = pd.date_range(sdate, edate)
        work_days = all_days[all_days.weekday < 5]
        ser = simulate_ar(len(work_days), mean=0, sd=1, ar_coef=back_ar)
        df_back = pd.DataFrame(index=work_days, columns=['value'])
        df_back['value'] = ser

    for cid in df_cids.index:
        for xcat in df_xcats.index:

            df_add = pd.DataFrame(columns=qdf_cols)

            sdate = pd.to_datetime(max(df_cids.loc[cid, 'earliest'], df_xcats.loc[xcat, 'earliest']))
            edate = pd.to_datetime(min(df_cids.loc[cid, 'latest'], df_xcats.loc[xcat, 'latest']))
            all_days = pd.date_range(sdate, edate)
            work_days = all_days[all_days.weekday < 5]

            df_add['real_date'] = work_days
            df_add.loc['cid', 'xcat'] = cid, xcat

            ser_mean = df_cids.loc[cid, 'mean_add'] + df_xcats.loc[xcat, 'mean_add']
            ser_sd = df_cids.loc[cid, 'sd_mult'] * df_xcats.loc[xcat, 'sd_mult']
            ser_arc = df_xcats.loc[xcat, 'ar_coef']
            df_add['value'] = simulate_ar(len(work_days), mean=ser_mean, sd=ser_sd, ar_coef=ser_arc)

    return df_out


if __name__ == "__main__":

    cids = ['AUD', 'CAD', 'GBP']
    xcats = ['XR', 'CRY']

    df_cids = pd.DataFrame(index=cids, columns=['earliest', 'latest', 'mean_add', 'sd_mult'])
    df_cids.loc['AUD', ] = ['2010-01-01', '2020-12-31', 0.5, 2]
    df_cids.loc['CAD', ] = ['2011-01-01', '2020-11-30', 0, 1]
    df_cids.loc['GBP', ] = ['2011-01-01', '2020-11-30', -0.2, 0.5]

    df_xcats = pd.DataFrame(index=xcats,  columns=['earliest', 'latest', 'mean_add', 'sd_mult', 'ar_coef', 'back_coef'])
    df_xcats.loc['XR', ] = ['2010-01-01', '2020-12-31', 0, 1, 0, 0.05]
    df_xcats.loc['CRY', ] = ['2011-01-01', '2020-10-30', 1, 2, 0.9, 0.1]


    dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)

    # sers_cor = simulate_cors(100, )
    # ser_ar = simulate_ar(100, mean=0, sd=1, ar_coef=0.75)



