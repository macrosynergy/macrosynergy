
import unittest
import numpy as np
import pandas as pd
import warnings
from tests.simulate import make_qdf
from macrosynergy.panel.hedge_ratio import *
from macrosynergy.management.shape_dfs import reduce_df
import math

class TestAll(unittest.TestCase):

    def dataframe_construction(self):

        # Emerging Market Asian countries.
        cids = ['IDR', 'INR', 'KRW', 'MYR', 'PHP']
        # Add the US - used as the hedging asset.
        cids += ['USD']

        self.__dict__['cids'] = cids
        xcats = ['FXXR_NSA', 'GROWTHXR_NSA', 'INFLXR_NSA', 'EQXR_NSA']
        self.__dict__['xcats'] = xcats

        df_cids = pd.DataFrame(index=self.cids, columns=['earliest', 'latest', 'mean_add',
                                                         'sd_mult'])

        df_cids.loc['IDR'] = ['2010-01-01', '2020-12-31', 0.5, 2]
        df_cids.loc['INR'] = ['2011-01-01', '2020-11-30', 0, 1]
        df_cids.loc['KRW'] = ['2012-01-01', '2020-11-30', -0.2, 0.5]
        df_cids.loc['MYR'] = ['2013-01-01', '2020-09-30', -0.2, 0.5]
        df_cids.loc['PHP'] = ['2002-01-01', '2020-09-30', -0.1, 2]
        df_cids.loc['USD'] = ['2000-01-01', '2020-03-20', 0, 1.25]

        df_xcats = pd.DataFrame(index=xcats, columns=['earliest', 'latest', 'mean_add',
                                                      'sd_mult', 'ar_coef', 'back_coef'])

        df_xcats.loc['FXXR_NSA'] = ['2012-01-01', '2020-10-30', 1, 2, 0.9, 1]
        df_xcats.loc['GROWTHXR_NSA'] = ['2012-01-01', '2020-10-30', 1, 2, 0.9, 1]
        df_xcats.loc['INFLXR_NSA'] = ['2013-01-01', '2020-10-30', 1, 2, 0.8, 0.5]
        df_xcats.loc['EQXR_NSA'] = ['2000-01-01', '2022-03-14', 0.5, 2, 0, 0.2]

        # If the asset being used as the hedge experiences a blackout period, then it is
        # probably not an appropriate asset to use in the hedging strategy.
        black = {'IDR': ['2010-01-01', '2012-01-04'],
                 'INR': ['2010-01-01', '2013-12-31'],
                 }
        self.__dict__['blacklist'] = black

        # Standard df for tests.
        dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)
        self.__dict__['dfd'] = dfd

        # The Unit Test will be based on the hedging strategy: hedge FX returns
        # (FXXR_NSA) against US Equity, S&P 500, (USD_EQXR_NSA).
        cid_hedge = 'USD'
        xcat_hedge = 'EQXR_NSA'
        self.__dict__['benchmark_df'] = reduce_df(dfd, xcats=[xcat_hedge],
                                                  cids=cid_hedge)

        self.__dict__["unhedged_df"] = reduce_df(dfd, xcats=['FXXR_NSA'],
                                                 cids=cids)
        self.__dict__["dfp_w"] = self.unhedged_df.pivot(index='real_date', columns='cid',
                                                        values='value')

    def test_date_alignment(self):
        """
        Firstly, hedge_ratio.py will potentially use a single asset to hedge a panel
        which can consist of multiple cross-sections, and each cross-section could be
        defined over differing time-series. Therefore, the .date_alignment() method is
        used to ensure the asset being used as the hedge and the asset being hedged are
        defined over the same timestamps. The method will return the proposed start &
        end date.
        """

        self.dataframe_construction()

        # Verify that two series passed will be aligned after applying the respective
        # method.
        # Test on MYR_FXXR_NSA against the hedging asset, USD_EQXR_NSA (both are defined
        # over different time horizons).
        c = 'MYR'
        xr = self.dfp_w[c]
        # Adjusts for the effect of pivoting.
        xr = xr.dropna(axis=0, how="all")

        br = pd.Series(data=self.benchmark_df['value'],
                       index=self.benchmark_df['real_date'])

        start_date, end_date = date_alignment(unhedged_return=xr,
                                              benchmark_return=br)
        # The latest start date of the two pd.Series.
        target_start = '2013-01-01'
        start_date = pd.Timestamp(start_date).strftime("%Y-%m-%d")
        self.assertTrue(start_date == target_start)

        end_date = pd.Timestamp(end_date).strftime("%Y-%m-%d")
        target_end = '2020-03-20'
        self.assertTrue(end_date == target_end)

    def test_hedge_calculator(self):
        """
        Method designed to calculate the hedge ratios used across the panel: each cross-
        section in the panel will have a different sensitivity parameter relative to the
        benchmark.
        Further, the frequency in which the hedge ratios are calculated is delimited by
        the 'refreq' parameter. The sample size of data, number of dates used in the
        re-estimation, will increase at a rate controlled by the parameter: the hedge
        ratio will be continuously re-estimated as days pass but will always include all
        realised timestamps (inclusive of the start_date).
        """

        self.dataframe_construction()

        # The method returns a standardised DataFrame. Confirm the first date in the
        # DataFrame is after the minimum observation date. The parameter 'min_obs'
        # ensures a certain number of days have passed until a hedge ratio is calculated.

        # Analysis completed using a single cross-section from the panel.
        c = 'KRW'
        xr = self.dfp_w[c]
        xr = xr.astype(dtype=np.float16)
        # Adjusts for the effect of pivoting.
        xr = xr.dropna(axis=0, how="all")

        br = pd.Series(data=self.benchmark_df['value'].to_numpy(),
                       index=self.benchmark_df['real_date'])
        br = br.astype(dtype=np.float16)

        # Apply the .date_alignment() method to establish the start & end date of the
        # re-estimation date series. Confirms the re-estimation frequency has been
        # correctly applied.
        # The frequency tested on will be monthly: business month end frequency.
        start_date, end_date = date_alignment(unhedged_return=xr, benchmark_return=br)
        dates_re = pd.date_range(start=start_date, end=end_date, freq='BM')
        str_time = lambda date: pd.Timestamp(str(date))
        dates_re = list(map(str_time, dates_re))

        min_observation = 50
        # Produce daily business day date series to determine the date that corresponds
        # to the specified minimum observation.
        daily_dates = pd.date_range(start=start_date, end=end_date, freq='B')
        test_min_obs = daily_dates.to_numpy()[60]

        df_hr = hedge_calculator(unhedged_return=xr, benchmark_return=br,
                                 rdates=dates_re, cross_section=c, meth='ols',
                                 min_obs=min_observation)
        # Confirm the first computed hedge ratio value falls after the minimum
        # observation date.
        test_date = df_hr['real_date'].iloc[0]
        self.assertTrue(test_date > test_min_obs)

        # In the example, the hedge ratio is computed monthly - last business day of the
        # respective month. Therefore, assuming the minimum observation date does not
        # fall on the final day of the month, the first date recorded in the returned
        # DataFrame should be the final business date of the same month as the minimum
        # observation date. For instance, 23/03/2010 -> 30/03/2010.
        # Test both dates are defined during the same month.
        test_min_obs_month = pd.Timestamp(test_min_obs).month
        self.assertTrue(test_min_obs_month == test_date.month)

        # The re-estimated hedge ratio will be computed using realised data up until the
        # respective date (in the above example, the final business day of the month).
        # However, the hedge ratio is applied from the following date (the date the
        # position will change) and to all the intermediary dates up until the next
        # re-estimation date. Therefore, confirm that the first date in the DataFrame is
        # a np.nan value representing the shift mechanism.
        first_value = df_hr['value'].iloc[0]
        self.assertTrue(math.isnan(first_value))

    def test_date_index(self, start_date: pd.Timestamp = None,
                        end_date: pd.Timestamp = None, refreq: str = 'm'):
        """
        The hedging ratio is re-estimated according to the frequency parameter.
        Therefore, break up the respective return series, which are defined daily, into
        the re-estimated frequency paradigm. To achieve this ensure the dates produced
        fall on business days, and will subsequently be present in the return-series
        dataframes (the daily series). The method used, in the source code, to delimit
        the resampling frequency is pd.resample(), and subsequently use this method to
        confirm the operation has been applied correctly.

        :param <pd.Timestamp> start_date:
        :param <pd.Timestamp> end_date:
        :param <str> refreq:

        return <List[pd.Timestamp]>: List of timestamps where each date is a valid
            business day, and the gap between each date is delimited by the frequency
            parameter.
        """

        start_date = "2000-01-01"
        end_date = "2020-01-01"

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

    def test_adjusted_returns(self, dates_refreq: List[pd.Timestamp] = [],
                              hedge_df: pd.DataFrame = pd.DataFrame(),
                              dfw: pd.DataFrame = pd.DataFrame(),
                              benchmark_return: pd.Series = None):

        refreq_buckets = self.dates_groups(dates_refreq=dates_refreq,
                                           benchmark_return=benchmark_return)
        # Hedge ratios across the respective panel: cross-sections included on the
        # category.
        # hedge_pivot = hedge_df.pivot(index='real_date', columns='cid',
        #                              values='value')
        hedge_pivot = pd.DataFrame()

        storage_dict = {}
        for c in hedge_pivot:
            series_hedge = hedge_pivot[c]
            storage = []
            for k, v in refreq_buckets.items():
                try:
                    hedge_value = series_hedge.loc[k]
                # Asset being hedged might not be available for that timestamp.
                except KeyError:
                    pass
                else:
                    hedged_position = v * hedge_value
                    storage.append(hedged_position)
            storage_dict[c] = pd.concat(storage)

        hedged_returns_df = pd.DataFrame.from_dict(storage_dict)

        output = dfw - hedged_returns_df

    def dates_groups(self, dates_refreq: List[pd.Timestamp],
                     benchmark_return: pd.Series):
        """
        Method used to break up the hedging asset's return series into the re-estimation
        periods. The method will return a dictionary where the key will be the
        re-estimation timestamp and the corresponding value will be the following
        timestamps until the next re-estimation date. It is the following returns that
        the hedge ratio is applied to: the hedge ratio is calculated using the preceding
        dates but is applied to the following dates until the next re-estimation period.

        :param <List[pd.Timestamp]> dates_refreq:
        :param <pd.Series> benchmark_return: the return series of the asset being used to
            hedge against the main asset. Used to compute the hedge ratio multiplied by
            the respective returns.

        :return <dict>: the dictionary's keys will be pd.Timestamps and the value will be
            a truncated pd.Series.
        """
        refreq_buckets = {}

        no_reest_dates = len(dates_refreq)
        for i, d in enumerate(dates_refreq):
            if i < (no_reest_dates - 1):
                intermediary_series = benchmark_return.truncate(before=d,
                                                                after=dates_refreq[
                                                                    (i + 1)])
                refreq_buckets[d + pd.DateOffset(1)] = intermediary_series

        return refreq_buckets

    def test_date_weekend(self, rdates: List[pd.Timestamp] = []):
        """
        Adjusts for weekends following the shift by a single day to adjust for when the
        hedge ratio becomes active.

        :param <List[pd.Timestamp]> rdates: the dates controlling the frequency of
            re-estimation.

        :return <List[pd.Timestamp]>: date-adjusted list of dates.
        """

        rdates_copy = []
        for d in rdates:
            if d.weekday() == 5:
                rdates_copy.append(d + pd.DateOffset(2))
            elif d.weekday() == 6:
                rdates_copy.append(d + pd.DateOffset(1))
            else:
                rdates_copy.append(d)

        return rdates_copy

    # Todo: add test of correct hedge ratio by comparing one or more ratio with a
    #  regression result generated outside the functions.


if __name__ == '__main__':

    unittest.main()