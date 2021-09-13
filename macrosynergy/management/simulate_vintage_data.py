import numpy as np
import pandas as pd
from typing import List, Union, Tuple
import matplotlib.pyplot as plt
import math
import datetime as dt
import time


##vins_m = VintageData('USD_INDX_SA', cutoff = "2019-06-30", release_lags = [3, 20, 25],  n_firsts = 12, shortest = 12, sd_ar = 5, trend_ar = 20, seasonal = 10, added_dates = 6)
class VintageData:
    """Creates standardized dataframe of single-ticker vintages
    """

    def __init__(self, ticker, cutoff = '2020-12-31', release_lags = [15, 30], n_firsts = 24,
                 shortest = 36, freq = 'M', start_value = 0, trend_ar = 5, sd_ar = math.sqrt(12),
                 seasonal = None, added_dates = 12):
        
        self.ticker = ticker
        ## Last possible release date.
        self.cutoff = dt.datetime.strptime(cutoff, '%Y-%m-%d').date()
        ## List of Integers in ascending order denoting lags of the first, second etc. release in (calendar) days.
        self.release_lags = release_lags
        ## Number of first-release vintages in the simulated data set.
        self.n_firsts = n_firsts
        ## Number of observations, days in the sequence, in the first, shortest, vintage.
        self.shortest = shortest

        self.freq = freq
        self.freq_int = dict(zip(['Q', 'M', 'W'], [4, 12, 52]))
        self.af = self.freq_int[freq]

        self.start_value = start_value
        ## Annualised trend. Default is 5% linear drift per year: deterministic trend.
        self.trend_ar = trend_ar
        ## Annualised standard deviation.
        self.sd_ar = sd_ar
        self.seasonal = seasonal
        ## Number of added first release dates used for grade 2 dataframe generation.
        self.n_firsts_gr2 = self.n_firsts + added_dates

        self.dates = set()


    ## Static Method can neither modify the object state nor class state: it will not update the instance's dictionary.
    ## Python will simply enforce the access restrictions by not passing in the "self" or the "cls" argument when a static method is called: the methods will act like regular functions.
    @staticmethod
    def week_day(rel_date, day):
        if day == 0:
            return rel_date - dt.timedelta(days = 1)
        return rel_date + dt.timedelta(days = 1)

    def make_grade1(self):

        ref_date = self.cutoff - dt.timedelta(self.release_lags[0])
        ref_date = ref_date.replace(day = 1)

        ## If exactly one of the parameters "start", "end", or "freq" is not specified, the missing parameter can be computed given the periods stated which equates to the number of timesteps in the range.
        ## For instance, the twelve months prior to the end date, and will return a List of datetime objects.
        eop_dates = pd.date_range(end = ref_date, periods = self.n_firsts, freq = self.freq)
        eop_list = eop_dates.date
        
        vin_lengths = list(range(self.shortest, self.shortest + self.n_firsts))
        v_first = vin_lengths[0]
        
        df_gr1 = pd.DataFrame(columns = ['release_date', 'observation_date', 'value'])

        obs_dates = pd.date_range(end = eop_dates[0], periods = v_first, freq = self.freq)

        list_ = np.linspace(0, (self.af - 1), self.af)
        linear_scale = list_ - np.mean(list_)
                    
        seas_factors = (self.seasonal / np.std(linear_scale))
        seas_factors *= linear_scale

        for i, eop_date in enumerate(eop_list):

            v = vin_lengths[i]
            if i > 0:
                ## Vintage Length.
                length = vin_lengths[i - 1]
                date = pd.Timestamp(eop_date)
                obs_dates = obs_dates.insert(loc = length, item = date)
                
            for rl in self.release_lags:
                
                rel_date = eop_date + dt.timedelta(days = rl)
                day = rel_date.weekday() - 5
                if day >= 0:
                    rel_date = self.week_day(rel_date, day)

                self.dates.add(rel_date)
                data = np.linspace(0, (v - 1), v)
                if self.start_value > 0:
                    
                    data = ((data * (self.trend_ar / self.af)) / 100)
                    data = (1 + data)
                    trend = self.start_value * data 
                else:
                    data = (data * (self.trend_ar / self.af))
                    trend = self.start_value + data

                
                values = trend + np.random.normal(0, self.sd_ar / math.sqrt(self.af), v)
                if self.seasonal is not None:
                    
                    if self.freq == 'M':
                        values = values * (1 + seas_factors[obs_dates.month - 1] / 100)
                    if self.freq == 'Q':
                        values = values * (1 + seas_factors[((obs_dates.month / 3) - 1).astype(np.int64)] / 100)
                    if self.freq == 'W':
                        values = values * (1 + seas_factors[np.clip(pd.Int64Index(obs_dates.isocalendar().week) - 1,
                                                                    a_min = 0, a_max = 51)] / 100)

                df_rel = pd.DataFrame({'release_date': rel_date, 'observation_date': obs_dates.date,
                                       'value': values})
                df_gr1 = df_gr1.append(df_rel)

        df_gr1["grading"] = "1"

        return self.add_ticker_parts(df_gr1)
    
    @staticmethod
    def map_weekday(date):
        
        day = date.weekday() - 5
        if day >= 0:
            date = VintageData.week_day(date, day)
        
        return date

    ## 20x improvement in performance speed against previous version.
    def make_grade2(self):
        ref_date = self.cutoff - dt.timedelta(self.release_lags[0])
        ref_date = ref_date.replace(day = 1)
        
        eop_dates = pd.date_range(end = ref_date, periods = self.n_firsts_gr2,
                                  freq = self.freq)
        eop_list = eop_dates.date
        eop_arr = np.array(eop_list)
        
        df_gr2 = pd.DataFrame(columns = ['release_date', 'observation_date'])

        n_ends = len(eop_list)
        n_release = len(self.release_lags)
        
        data = np.zeros((n_ends, n_release), dtype = object)
        
        for i in range(n_release):
            dates = eop_arr + dt.timedelta(days = self.release_lags[i])
            dates = list(map(self.map_weekday, dates))
            data[:, i] = np.array(dates)
                                                
        shape = data.shape
        data = data.reshape((shape[0] * shape[1]))
        eop_arr = np.sort(np.tile(eop_arr, n_release))

        df_gr2['release_date'] = data
        df_gr2['observation_date'] = eop_arr
        df_gr2['ticker'] = self.ticker

        return self.add_ticker_parts(df_gr2)
    

    def make_graded(self, grading, upgrades = []):

        assert len(upgrades) == (len(grading) - 1)
        df_grd = self.make_grade1()
        df_grd['grading'] = str(grading[0])
        dates = sorted(self.dates)
        
        for i, up in enumerate(upgrades):
            
            filt = df_grd['release_date'] >= dates[up]
            df_grd.loc[filt, 'grading'] = str(grading[i + 1])
            
        return df_grd
    

    def add_ticker_parts(self, df):
        old_cols = list(df.columns)
        add_cols = ['cross_section', 'category_code', 'adjustment', 'transformation']
        ticker_parts = self.ticker.split('_', 3)
        
        for i, tick in enumerate(ticker_parts):
            df[add_cols[i]] = tick
            
        if 'transformation' not in df.columns:
            df['transformation'] = None

        return df.loc[:, add_cols + old_cols]


if __name__ == "__main__":

    vins_m = VintageData('USD_INDX_SA', cutoff = "2019-06-30", release_lags = [3, 20, 25],  n_firsts = 12,
                         shortest = 12, sd_ar = 5, trend_ar = 20, seasonal = 10, added_dates = 6)
    
    ## dfm1 = vins_m.make_grade1()
    start = time.time()
    dfmg = vins_m.make_graded(grading = [3, 2.1, 1], upgrades = [12, 24])
    print(f"Time Elapsed, test_file: {time.time() - start}.")
    ## dfm1.groupby('release_date').agg(['mean', 'count'])
    start = time.time()
    dfm2 = vins_m.make_grade2()
    print(f"Time Elapsed, test_file: {time.time() - start}.")

    ## vins_q = VintageData('USD_INDX_SA', release_lags=[3, 20, 25], number_firsts=2, shortest=8, freq='Q',
                         ## seasonal = 10, added_dates = 4)
    ## dfq1 = vins_q.make_grade1()
    ## dfq1.groupby('release_date').agg(['mean', 'count'])
    ## dfq2 = vins_q.make_grade2()

    ## vins_w = VintageData('USD_INDX_SA', cutoff="2019-06-30", release_lags=[3, 20, 25],
                         ## number_firsts=3 * 52, shortest=26, freq='W', seasonal = 10, added_dates = 52)
    ## dfw1 = vins_w.make_grade1()
    ## dfw1.groupby('release_date').agg(['mean', 'count'])
    ## dfw2 = vins_w.make_grade2()

    ## dfm1.info()
