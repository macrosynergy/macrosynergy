import numpy as np
import pandas as pd
from typing import List, Union, Tuple
import matplotlib.pyplot as plt
import math
import datetime as dt


class VintageData:
    """Creates standardized dataframe of single-ticker vintages

    This class creates standardized grade 1 and grade 2 vintage data

    :param <str> ticker: ticker name
    :param <str> cutoff: last possible release date. The format must be '%Y-%m-%d'.
            All other dates are calculated from this one. Default is end 2020.
    :param <list> release_lags: list of integers in ascending order denoting lags of the first, second etc. release
            in (calendar) days. Default is first release after 15 days and revision after 30 days.
            If days fall on weekend they will be delayed to  Monday.
    :param <int> number_firsts: number of first-release vintages in the simulated data set. Default is 24.
    :param <int> shortest: number of observations in the first (shortest) vintage. Default is 36.
    :param <str> freq: letter denoting the frequency of the vintage data. Must be one of 'M' (monthly, default),
            'Q' (quarterly)  or 'W' (weekly).
    :param <float> start_value: expected first value of the random series. Default is 100.
    :param <float> trend_ar: annualized trend. Default is 5% linear drift per year.
            This is applied to the start value. If the start value is not positive the linear trend is added as number.
    :param <float> sd_ar: annualized standard deviation. Default is sqrt(12).
    :param <float> seasonal: adds seasonal pattern (applying linear factor from low to high through the year)
            with value denoting the average % seasonal factor through the year. Default is None.
            The seasonal pattern makes only sense for values that are strictly positive and are interpreted as indices.
    :param <int> added_dates: number of added first release dates, used for grade 2 dataframe generation.
            Default is 12.

    """

    def __init__(self, ticker, cutoff='2020-12-31', release_lags=[15, 30], number_firsts=24, shortest=36, freq='M',
                 start_value=100, trend_ar=5, sd_ar=math.sqrt(12), seasonal=None, added_dates=12):
        self.ticker = ticker
        self.cutoff = dt.datetime.strptime(cutoff, '%Y-%m-%d').date()
        self.release_lags = release_lags
        self.number_firsts = number_firsts
        self.shortest = shortest

        self.freq = freq
        self.af = int(pd.DataFrame([4, 12, 52], index=['Q', 'M', 'W']).loc[freq])

        self.start_value = start_value
        self.trend_ar = trend_ar
        self.sd_ar = sd_ar
        self.seasonal = seasonal

        self.number_firsts_gr2 = self.number_firsts + added_dates

    def make_grade1(self):
        ref_date = (self.cutoff - dt.timedelta(self.release_lags[0])).replace(day=1)
        eop_dates = pd.date_range(end=ref_date, periods=self.number_firsts, freq=self.freq).date
        vin_lengths = [vl for vl in range(self.shortest, self.shortest + self.number_firsts)]
        df_gr1 = pd.DataFrame(columns=['release_date', 'observation_date', 'value'])
        for eop_date in eop_dates:
            for rl in self.release_lags:
                rel_date = eop_date + dt.timedelta(days=rl)
                if rel_date.weekday() >= 5:  # shift release dates out of weekends
                    rel_date = rel_date - dt.timedelta(days=1) if (rel_date.weekday() == 5) \
                        else rel_date + dt.timedelta(days=1)
                vin_length = vin_lengths[int(np.where(eop_dates == eop_date)[0])]  # pick vintage length
                obs_dates = pd.date_range(end=eop_date, periods=vin_length, freq=self.freq)
                if self.start_value > 0:
                    trend = self.start_value * (1+np.array([i * self.trend_ar/self.af for i in range(vin_length)])/100)
                else:  # if trend cannot be proportionate just add as number
                    trend = self.start_value + np.array([i * self.trend_ar/self.af for i in range(vin_length)])
                values = trend + np.random.normal(0, self.sd_ar / math.sqrt(self.af), vin_length)
                if self.seasonal is not None:
                    linear_scale = list(range(self.af)) - np.mean(list(range(self.af)))
                    seas_factors = (self.seasonal / np.std(linear_scale)) * linear_scale
                    if self.freq == 'M':
                        values = values * (1 + seas_factors[obs_dates.month - 1]/100)
                    if self.freq == 'Q':
                        values = values * (1 + seas_factors[(obs_dates.month/3 - 1).astype(np.int64)]/100)
                    if self.freq == 'W':
                        values = values * (1 + seas_factors[np.clip(pd.Int64Index(obs_dates.isocalendar().week) - 1,
                                                                    a_min=0, a_max=51)] / 100)
                df_rel = pd.DataFrame({'release_date': rel_date, 'observation_date': obs_dates.date, 'value': values})
                df_gr1 = df_gr1.append(df_rel)

        df_gr1["grading"] = "1"

        return self.add_ticker_parts(df_gr1)

    def make_graded(self, grading, upgrades=[]):

        """
        Simulates an explicitly graded dataframe with a column 'grading'
        :param <list> grading: optional addition of grading column. List of grades used from lowest to highest.
        Default is None. Must be a subset of [3, 2.2, 2.1, 1].
        :param <list> upgrades: indices of release dates at which the series upgrade.
        Must have length of grading minus one. Default is None.
        """
        assert len(upgrades) == (len(grading) - 1)
        df_grd = self.make_grade1()
        df_grd['grading'] = str(grading[0])
        for up in range(len(upgrades)):
            filt = df_grd['release_date'] >= df_grd['release_date'].unique()[upgrades[up]]
            df_grd.loc[filt, 'grading'] = str(grading[up + 1])
        return df_grd

    def make_grade2(self):
        ref_date = (self.cutoff - dt.timedelta(self.release_lags[0])).replace(day=1)
        eop_dates = pd.date_range(end=ref_date, periods=self.number_firsts_gr2, freq=self.freq).date
        df_gr2 = pd.DataFrame(columns=['release_date', 'observation_date'])
        for eop_date in eop_dates:
            for rl in self.release_lags:
                rel_date = eop_date + dt.timedelta(days=rl)
                if rel_date.weekday() >= 5:  # shift release dates out of weekends
                    rel_date = rel_date - dt.timedelta(days=1) if (rel_date.weekday() == 5) \
                        else rel_date + dt.timedelta(days=1)
                df_gr2.loc[len(df_gr2)] = [rel_date, eop_date]
        df_gr2['ticker'] = self.ticker
        return self.add_ticker_parts(df_gr2)

    def add_ticker_parts(self, df):
        old_cols = list(df.columns)
        add_cols = ['cross_section', 'category_code', 'adjustment', 'transformation']
        ticker_parts = self.ticker.split('_', 3)
        for i in range(len(ticker_parts)):
            df[add_cols[i]] = ticker_parts[i]
        if 'transformation' not in df.columns:
            df['transformation'] = None
        return df.loc[:, add_cols + old_cols]


if __name__ == "__main__":

    vins_m = VintageData('USD_INDX_SA', cutoff="2019-06-30", release_lags=[3, 20, 25],  number_firsts=12,
                         shortest=12, sd_ar=5, trend_ar=20, seasonal=10, added_dates=6)
    dfm1 = vins_m.make_grade1()
    dfm1.groupby('release_date').agg(['mean', 'count'])
    dfm2 = vins_m.make_grade2()
    dfmg = vins_m.make_graded(grading=[3, 2.1, 1], upgrades=[12, 24])

    vins_q = VintageData('USD_INDX_SA', release_lags=[3, 20, 25], number_firsts=2, shortest=8, freq='Q',
                         seasonal = 10, added_dates = 4)
    dfq1 = vins_q.make_grade1()
    dfq1.groupby('release_date').agg(['mean', 'count'])
    dfq2 = vins_q.make_grade2()

    vins_w = VintageData('USD_INDX_SA', cutoff="2019-06-30", release_lags=[3, 20, 25],
                         number_firsts=3 * 52, shortest=26, freq='W', seasonal = 10, added_dates = 52)
    dfw1 = vins_w.make_grade1()
    dfw1.groupby('release_date').agg(['mean', 'count'])
    dfw2 = vins_w.make_grade2()

    dfm1.info()