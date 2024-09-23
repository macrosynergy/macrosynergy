"""
Module with functionality for generating mock 
quantamental data vintages for testing purposes.
"""

import numpy as np
import pandas as pd
import math
import datetime as dt
from typing import List
from macrosynergy.management.utils import _map_to_business_day_frequency
from macrosynergy.management.constants import ANNUALIZATION_FACTORS


class VintageData:
    """
    Creates standardized dataframe of single-ticker vintages. This class creates
    standardized grade 1 and grade 2 vintage data.

    :param <str> ticker: ticker name
    :param <str> cutoff: last possible release date. The format must be '%Y-%m-%d'.
            All other dates are calculated from this one. Default is end 2020.
    :param <list> release_lags: list of integers in ascending order denoting lags of the
        first, second etc. release in (calendar) days. Default is first release after 15
        days and revision after 30 days. If days fall on weekend they will be delayed to
        Monday.
    :param <int> number_firsts: number of first-release vintages in the simulated data
        set. Default is 24.
    :param <int> shortest: number of observations in the first (shortest) vintage.
        Default is 36.
    :param <str> freq: letter denoting the frequency of the vintage data. Must be one of
        'M' (monthly, default), 'Q' (quarterly)  or 'W' (weekly).
    :param <float> start_value: expected first value of the random series. Default is 100.
    :param <float> trend_ar: annualized trend. Default is 5% linear drift per year.
        This is applied to the start value. If the start value is not positive the
        linear trend is added as number.
    :param <float> sd_ar: annualized standard deviation. Default is sqrt(12).
    :param <float> seasonal: adds seasonal pattern (applying linear factor from low to
        high through the year) with value denoting the average % seasonal factor through
        the year. Default is None. The seasonal pattern makes only sense for values that
        are strictly positive and are interpreted as indices.
    :param <int> added_dates: number of added first release dates, used for grade 2
        dataframe generation. Default is 12.

    """

    def __init__(
        self,
        ticker,
        cutoff="2020-12-31",
        release_lags=[15, 30],
        number_firsts=24,
        shortest=36,
        freq="M",
        start_value=100,
        trend_ar=5,
        sd_ar=math.sqrt(12),
        seasonal=None,
        added_dates=12,
    ):
        self.ticker = ticker

        self.cutoff = self.date_check(cutoff)
        self.release_lags = release_lags
        self.number_firsts = number_firsts
        self.shortest = shortest

        self.freq = _map_to_business_day_frequency(freq)
        self.freq_int = ANNUALIZATION_FACTORS
        self.af = self.freq_int[freq]

        self.start_value = start_value
        self.trend_ar = trend_ar
        self.sd_ar = sd_ar
        self.seasonal = seasonal

        self.number_firsts_gr2 = self.number_firsts + added_dates
        self.dates = set()

    @staticmethod
    def date_check(date_string):
        """
        Validates that the dates passed are valid timestamp expressions and will convert
        to the required form '%Y-%m-%d'.

        :param <str> date_string: valid date expression. For instance, "1st January,
            2000."
        :raises <TypeError>: if the date_string is not a string.
        :raises <ValueError>: if the date_string is not in the correct format.
        """
        date_error = "Expected form of string: '%Y-%m-%d'."
        if date_string is not None:
            if not isinstance(date_string, str):
                raise TypeError("`date_string` must be a string.")
            try:
                pd.Timestamp(date_string).strftime("%Y-%m-%d")
            except ValueError:
                raise ValueError(date_error)
            else:
                date = pd.Timestamp(date_string).strftime("%Y-%m-%d")
                return pd.Timestamp(date)

    @staticmethod
    def week_day(rel_date, day):
        if day == 0:
            return rel_date - dt.timedelta(days=1)

        return rel_date + dt.timedelta(days=1)

    def seasonal_adj(self, obs_dates, seas_factors, values):
        """
        Method used to seasonally adjust the series. Economic data can vary according to
        the season.

        :param <List[pd.Timestamps]> obs_dates: observation dates for the series.
        :param <List[float]> seas_factors: seasonal factors.
        :param <List[float]> values: existing values that have not been seasonally
            adjusted.

        :return <List[float]>: returns a list of values which have been adjusted
            seasonally
        """
        if self.freq == "W":
            week_dates = obs_dates.isocalendar().week
            condition = np.where(week_dates > 52)[0]
            if condition.size > 0:
                index = next(iter(condition))
                week_dates[index] = 52

            week = (week_dates - 1).to_numpy().astype(dtype=np.uint8)
            seasonal = seas_factors[week]
            return values * (1 + (seasonal / 100))

        month = 12 / self.af
        month_freq = obs_dates.month // month
        month_freq = month_freq.astype(dtype=np.uint8)

        return values * (1 + seas_factors[month_freq - 1] / 100)

    def make_grade1(self):
        ref_date = self.cutoff - dt.timedelta(self.release_lags[0])
        ref_date = ref_date.replace(day=1)

        eop_dates = pd.date_range(
            end=ref_date, periods=self.number_firsts, freq=self.freq
        )
        eop_dates = eop_dates.date
        vin_lengths = list(range(self.shortest, self.shortest + self.number_firsts))
        v_first = vin_lengths[0]

        df_gr1 = pd.DataFrame(columns=["release_date", "observation_date", "value"])
        obs_dates = pd.date_range(end=eop_dates[0], periods=v_first, freq=self.freq)

        list_ = np.linspace(0, (self.af - 1), self.af)
        linear_scale = list_ - np.mean(list_)

        seas_factors = self.seasonal / np.std(linear_scale)
        seas_factors *= linear_scale
        df_rels: List[pd.DataFrame] = []
        for i, eop_date in enumerate(eop_dates):
            v = vin_lengths[i]
            if i > 0:
                length = vin_lengths[i - 1]
                date = pd.Timestamp(eop_date)
                obs_dates = obs_dates.insert(loc=length, item=date)

            data = np.linspace(0, (v - 1), v)
            if self.start_value > 0:
                data = (data * (self.trend_ar / self.af)) / 100
                data = 1 + data
                trend = self.start_value * data
            else:
                data = data * (self.trend_ar / self.af)
                trend = self.start_value + data

            for rl in self.release_lags:
                rel_date = eop_date + dt.timedelta(days=rl)
                day = rel_date.weekday() + 5
                if day >= 0:
                    rel_date = self.week_day(rel_date, day)

                self.dates.add(rel_date)
                values = trend + np.random.normal(0, self.sd_ar / math.sqrt(self.af), v)
                if self.seasonal is not None:
                    values = self.seasonal_adj(obs_dates, seas_factors, values)

                df_rel = pd.DataFrame(
                    {
                        "release_date": rel_date,
                        "observation_date": obs_dates.date,
                        "value": values,
                    }
                )
                df_rels.append(df_rel)

        df_gr1 = pd.concat(df_rels, ignore_index=True)

        df_gr1["grading"] = 1
        return self.add_ticker_parts(df_gr1)

    def make_graded(self, grading, upgrades=[]):
        """
        Simulates an explicitly graded dataframe with a column 'grading'.

        :param <list> grading: optional addition of grading column. List of grades used
            from lowest to highest.
            Default is None. Must be a subset of [3, 2.2, 2.1, 1].
        :param <list> upgrades: indices of release dates at which the series upgrade.
            Must have length of grading minus one. Default is None.
        """

        assert len(upgrades) == (len(grading) - 1)
        df_grd = self.make_grade1()
        df_grd["grading"] = str(grading[0])
        dates = sorted(self.dates)
        for i, up in enumerate(upgrades):
            filt = df_grd["release_date"] >= dates[up]
            df_grd.loc[filt, "grading"] = str(grading[(i + 1)])

        return df_grd

    @staticmethod
    def map_weekday(date):
        day = date.weekday() - 5
        if day >= 0:
            date = VintageData.week_day(date, day)

        return date

    def make_grade2(self):
        """
        Method used to construct a dataframe that consists of each respective observation
        date and the corresponding release date(s) (the release dates are computed using
        the observation date and the time-period(s) specified in the field
        "release_lags").

        :return <pd.DataFrame>: Will return the DataFrame with the additional columns.
        """
        ref_date = self.cutoff - dt.timedelta(self.release_lags[0])
        ref_date = ref_date.replace(day=1)

        eop_dates = pd.date_range(
            end=ref_date, periods=self.number_firsts_gr2, freq=self.freq
        )
        eop_list = eop_dates.date
        eop_arr = np.array(eop_list)

        df_gr2 = pd.DataFrame(columns=["release_date", "observation_date"])

        n_ends = len(eop_list)
        n_release = len(self.release_lags)

        data = np.zeros((n_ends, n_release), dtype=object)

        for i in range(n_release):
            dates = eop_arr + dt.timedelta(days=self.release_lags[i])
            dates = list(map(self.map_weekday, dates))
            data[:, i] = np.array(dates)

        shape = data.shape
        data = data.reshape((shape[0] * shape[1]))
        eop_arr = np.sort(np.tile(eop_arr, n_release))

        df_gr2["release_date"] = data
        df_gr2["observation_date"] = eop_arr
        df_gr2["ticker"] = self.ticker

        return self.add_ticker_parts(df_gr2)

    def add_ticker_parts(self, df):
        """
        Method used to add the associated tickers.

        :param <pd.DataFrame> df: standardised dataframe.

        :return <pd.DataFrame>: Will return the DataFrame with the additional columns.
        """
        old_cols = list(df.columns)
        add_cols = ["cross_section", "category_code", "adjustment", "transformation"]
        ticker_parts = self.ticker.split("_", 3)

        for i, tick in enumerate(ticker_parts):
            df[add_cols[i]] = tick

        if "transformation" not in df.columns:
            df["transformation"] = None

        return df.loc[:, add_cols + old_cols]


if __name__ == "__main__":
    vins_m = VintageData(
        "USD_INDX_SA",
        cutoff="2019-06-30",
        release_lags=[3, 20, 25],
        number_firsts=12,
        shortest=12,
        sd_ar=5,
        trend_ar=20,
        seasonal=10,
        added_dates=6,
    )

    dfm1 = vins_m.make_grade1()

    dfm1.groupby("release_date").agg(["mean", "count"])
    dfm2 = vins_m.make_grade2()
    dfmg = vins_m.make_graded(grading=[3, 2.1, 1], upgrades=[12, 24])

    vins_q = VintageData(
        "USD_INDX_SA",
        release_lags=[3, 20, 25],
        number_firsts=2,
        shortest=8,
        freq="Q",
        seasonal=10,
        added_dates=4,
    )
    dfq1 = vins_q.make_grade1()
    dfq1.groupby("release_date").agg(["mean", "count"])
    dfq2 = vins_q.make_grade2()

    vins_w = VintageData(
        "USD_INDX_SA",
        cutoff="2019-06-30",
        release_lags=[3, 20, 25],
        number_firsts=3 * 52,
        shortest=26,
        freq="W",
        seasonal=10,
        added_dates=52,
    )
    dfw1 = vins_w.make_grade1()
    dfw1.groupby("release_date").agg(["mean", "count"])
    dfw2 = vins_w.make_grade2()

    dfm1.info()
