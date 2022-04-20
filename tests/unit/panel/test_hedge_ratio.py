
import unittest
import numpy as np
import pandas as pd
import warnings
from tests.simulate import make_qdf
from macrosynergy.panel.hedge_ratio import *
from random import randint

class TestAll(unittest.TestCase):

    def dataframe_construction(self):

        self.__dict__['cids'] = ['AUD', 'CAD', 'GBP']
        self.__dict__['xcats'] = ['CRY', 'XR']

        df_cids = pd.DataFrame(index=self.cids, columns=['earliest', 'latest', 'mean_add',
                                                         'sd_mult'])
        df_cids.loc['AUD', :] = ['2010-01-01', '2020-12-31', 0.5, 2]
        df_cids.loc['CAD', :] = ['2010-01-01', '2020-11-30', 0, 1]
        df_cids.loc['GBP', :] = ['2012-01-01', '2020-11-30', -0.2, 0.5]

        df_xcats = pd.DataFrame(index=self.xcats,
                                columns=['earliest', 'latest', 'mean_add',
                                         'sd_mult', 'ar_coef', 'back_coef'])
        df_xcats.loc['CRY', :] = ['2010-01-01', '2020-10-30', 1, 2, 0.9, 0.5]
        df_xcats.loc['XR', :] = ['2011-01-01', '2020-12-31', 0, 1, 0, 0.3]

        # Standard df for tests.
        dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)
        self.__dict__['dfd'] = dfd[dfd['xcat'] == 'CRY']
        self.__dict__['dfw'] = self.dfd.pivot(index='real_date', columns='cid',
                                              values='value')

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

    # Todo: add test of correct hedge ratio by comparing one or more ratio with a
    #  regression result generated outside the functions.

if __name__ == '__main__':

    unittest.main()