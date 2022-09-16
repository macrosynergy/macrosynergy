
import unittest
import random
import numpy as np
import pandas as pd
from tests.simulate import make_qdf
from macrosynergy.management.shape_dfs import reduce_df, categories_df
from math import ceil, floor
from datetime import timedelta
from pandas.tseries.offsets import BMonthEnd

class TestAll(unittest.TestCase):

    def dataframe_constructor(self):

        self.__dict__['cids'] = ['AUD', 'CAD', 'GBP']
        self.__dict__['xcats'] = ['CRY', 'XR', 'GROWTH', 'INFL', 'GDP']

        df_cids = pd.DataFrame(index=self.cids,
                               columns=['earliest', 'latest', 'mean_add', 'sd_mult'])
        df_cids.loc['AUD', :] = ['2011-01-01', '2020-12-31', 0.5, 2]
        df_cids.loc['CAD', :] = ['2011-01-01', '2020-12-31', 0, 1]
        df_cids.loc['GBP', :] = ['2011-01-01', '2020-12-31', -0.2, 0.5]

        df_xcats = pd.DataFrame(index=self.xcats,
                                columns=['earliest', 'latest', 'mean_add', 'sd_mult',
                                         'ar_coef', 'back_coef'])
        df_xcats.loc['CRY', :] = ['2011-01-01', '2020-12-31', 1, 2, 0.9, 0.5]
        df_xcats.loc['XR', :] = ['2011-01-01', '2020-12-31', 0, 1, 0, 0.3]
        df_xcats.loc['GROWTH', :] = ['2011-01-01', '2020-12-31', 0, 2, 0, 0.4]
        df_xcats.loc['INFL', :] = ['2011-01-01', '2020-12-31', 0, 3, 0, 0.6]
        df_xcats.loc['GDP', :] = ['2011-01-01', '2020-12-31', 0, 1, 0, 0.7]

        random.seed(1)
        np.random.seed(0)
        self.__dict__['dfd'] = make_qdf(df_cids, df_xcats, back_ar=0.75)

    def test_reduce_df_general(self):
        self.dataframe_constructor()

        dfd_x = reduce_df(self.dfd, xcats=['CRY'], cids=self.cids[0:2],
                          start='2013-01-01', end='2019-01-01')

        self.assertTrue(all(dfd_x['cid'].unique() == np.array(['AUD', 'CAD'])))
        self.assertTrue(all(dfd_x['xcat'].unique() == np.array(['CRY'])))
        # Test the dimensions through the date keys.
        self.assertTrue(dfd_x['real_date'].min() == pd.to_datetime('2013-01-01'))
        self.assertTrue(dfd_x['real_date'].max() == pd.to_datetime('2019-01-01'))

    def test_reduce_df_intersect(self):
        self.dataframe_constructor()

        filt1 = ~((self.dfd['cid'] == 'AUD') & (self.dfd['xcat'] == 'XR'))
        dfdx = self.dfd[filt1]  # Simulate missing cross sections.
        dfd_x1, xctx, cidx = reduce_df(dfdx, xcats=['XR', 'CRY'], cids=self.cids,
                                       intersect=True, out_all=True)
        self.assertTrue(cidx == ['CAD', 'GBP'])

    def test_reduce_df_black(self):
        self.dataframe_constructor()

        black = {'AUD_1': ['2011-01-01', '2012-12-31'],
                 'AUD_2': ['2018-01-01', '2100-01-01']}

        dfd_x = reduce_df(self.dfd, xcats=['XR'], cids=self.cids, blacklist=black)
        dfd_aud = dfd_x[dfd_x['cid'] == 'AUD']
        # Adjustment for the blacklist period applied.
        self.assertTrue(dfd_aud['real_date'].min() == pd.to_datetime('2013-01-01'))

        black_range_1 = pd.date_range(start='2011-01-01', end='2012-12-31')
        self.assertTrue(not any(item in black_range_1 for item in dfd_aud['real_date']))

        black = {'CAD': ['2014-01-01', '2014-12-31']}
        dfd_x = reduce_df(self.dfd, xcats=['XR'], cids=self.cids, blacklist=black)
        dfd_cad = dfd_x[dfd_x['cid'] == 'CAD']

        black_range_2 = pd.date_range(start='2014-01-01', end='2014-12-31')
        self.assertTrue(not any(item in black_range_2 for item in dfd_cad['real_date']))

    def test_categories_df_conversion(self):
        self.dataframe_constructor()

        dfc = categories_df(self.dfd, xcats=['XR', 'CRY'], cids=self.cids,
                            freq='M', lag=0, xcat_aggs=['mean', 'last'])

        self.dfd['real_date'] = pd.to_datetime(self.dfd['real_date'], errors='coerce')
        column = self.dfd['real_date']

        filt1 = (self.dfd['real_date'].dt.year == 2011) & \
                (self.dfd['real_date'].dt.month == 10)
        filt2 = (self.dfd['cid'] == 'AUD') & (self.dfd['xcat'] == 'XR')

        # Check correct application of monthly mean: the procedure used to test the
        # value for each defined frequency is to isolate the last value of each
        # respective period, and compare again the "manual" calculation, x1.
        x1 = self.dfd[filt1 & filt2]['value'].mean()
        x2 = float(dfc.loc[('AUD', '2011-10-31'), 'XR'])
        self.assertAlmostEqual(x1, x2)

        # Check correct application of end-of-month aggregation. Naturally, if the 'freq'
        # parameter is set equal to monthly, each monthly time-period is reduced to a
        # single value.
        filt2 = (self.dfd['cid'] == 'AUD') & (self.dfd['xcat'] == 'CRY')
        x1 = self.dfd[filt1 & filt2]['value'].iloc[-1]
        x2 = float(dfc.loc[('AUD', '2011-10-31'), 'CRY'])
        self.assertAlmostEqual(x1, x2)

    def test_categories_df_year(self):

        self.dataframe_constructor()

        # The year aggregation of the data is computed from the specified start date (the
        # parameter received), as opposed to the earliest date in the dataframe.
        # Therefore, if the "years" parameter is not equal to None, the "start" parameter
        # must be defined.
        with self.assertRaises(AssertionError):
            dfc = categories_df(self.dfd, xcats=['XR', 'CRY'], cids=['CAD'],
                                freq='M', lag=0, xcat_aggs=['mean', 'mean'],
                                start=None, years=7)

        # Lag must be zero if "years" is not equal to None.
        with self.assertRaises(AssertionError):
            dfc = categories_df(self.dfd, xcats=['XR', 'CRY'], cids=['CAD'],
                                freq='M', lag=1, xcat_aggs=['mean', 'mean'],
                                start=None, years=7)

        # Test a specific cross-section. Check the year conversion. Applied from the
        # defined start date. The yearly interval will be computed from the "start"
        # parameter.
        start = pd.Timestamp('2000-01-01').year
        years = 10

        filt1 = (self.dfd['cid'] == 'CAD')
        dfd = self.dfd[filt1]

        s_year = dfd['real_date'].min().year
        e_year = dfd['real_date'].max().year + 1
        # The number of possible buckets the cross-section could be defined over:
        # dependent on the date of the first realised value.
        no_buckets = ceil((e_year - start) / years)

        # Adjust for the formal start date.
        start_bucket = int(floor(s_year - start) / years)

        self.dfd['real_date'] = pd.to_datetime(self.dfd['real_date'], errors='coerce')
        dfc = categories_df(dfd, xcats=['XR', 'CRY'], cids=['CAD'],
                            freq='M', lag=0, xcat_aggs=['mean', 'mean'],
                            start='2000-01-01', years=years)

        realised_buckets = no_buckets - start_bucket
        filter_df = dfc.loc['CAD', :]
        self.assertTrue(realised_buckets == filter_df.shape[0])

        # Apply the same logic but to a different testcase.
        years = 4
        no_buckets = ceil((e_year - start) / years)

        # Adjust for the formal start date.
        start_bucket = int(floor(s_year - start) / years)

        dfc = categories_df(dfd, xcats=['XR', 'CRY'], cids=['CAD'],
                            freq='M', lag=0, xcat_aggs=['mean', 'mean'],
                            start='2000-01-01', years=years)

        realised_buckets = no_buckets - start_bucket
        filter_df = dfc.loc['CAD', :]
        self.assertTrue(realised_buckets == filter_df.shape[0])

        # Test the aggregator parameter 'last': as the name suggests, 'last' will isolate
        # the terminal value of each time-period. Therefore, check the returned value, in
        # the DataFrame, confirms the above logic.
        dfc = categories_df(self.dfd, xcats=['XR', 'CRY'],
                            cids=['AUD', 'CAD'], xcat_aggs=['last', 'mean'],
                            start='2005-01-01', years=6)

        # Manual check.
        reduced_df = self.dfd[self.dfd['xcat'] == 'XR']
        reduced_df = reduced_df[reduced_df['real_date'] == '2016-12-30']
        aud = reduced_df[reduced_df['cid'] == 'AUD']
        cad = reduced_df[reduced_df['cid'] == 'CAD']
        aud = aud['value'].to_numpy()[0]
        cad = cad['value'].to_numpy()[0]

        # Isolate the first value for both cross-sectional series: '2011 - 2016'.
        aud_xr = dfc['XR'].loc['AUD'][0]
        cad_xr = dfc['XR'].loc['CAD'][0]
        self.assertTrue(aud == aud_xr)
        self.assertTrue(cad == cad_xr)

    def test_categories_df_lags(self):
        self.dataframe_constructor()

        dfc = categories_df(self.dfd, xcats=['CRY', 'XR'], cids=self.cids, freq='M',
                            lag=1, fwin=3, xcat_aggs=['last', 'mean'])

        # Check the correct application of 1st series forward (average) window.
        self.dfd['real_date'] = pd.to_datetime(self.dfd['real_date'], errors='coerce')

        filt1 = (self.dfd['real_date'].dt.year == 2013) & \
                (self.dfd['real_date'].dt.month.isin([10, 11, 12]))
        filt2 = (self.dfd['cid'] == 'AUD') & (self.dfd['xcat'] == 'XR')
        x1 = round(float(np.mean(self.dfd[filt1 & filt2].set_index('real_date').
                                 resample('M').mean())), 10)

        x2 = round(float(dfc.loc[('AUD', '2013-10-31'), 'XR']), 10)
        self.assertAlmostEqual(x1, x2)

        # Check correct application of 2nd series lag.
        filt1 = (self.dfd['real_date'].dt.year == 2011) & \
                (self.dfd['real_date'].dt.month == 10)
        filt2 = (self.dfd['cid'] == 'AUD') & (self.dfd['xcat'] == 'CRY')
        x1 = self.dfd[filt1 & filt2]['value'].iloc[-1]
        x2 = float(dfc.loc[('AUD', '2011-11-30'), 'CRY'])
        self.assertAlmostEqual(x1, x2)

        # The relationship between a signal and the return can materialise over different
        # time intervals which is reflected in the use of lags. Additionally, the
        # time sensitivity can vary between different pairs of signals & returns: weekly,
        # monthly, quarterly etc.

        df = self.dfd.copy()
        dfc = categories_df(df, xcats=['CRY', 'XR'], cids=self.cids, freq='W',
                            lag=3, fwin=0, xcat_aggs=['mean', 'mean'])

        # Isolate a fixed week to test on and a single cross-section. It is the signal
        # that is being lagged, 'CRY'.
        dfc_aud = dfc.loc['AUD']

        fixed_date = '2011-02-25'
        test_value = dfc_aud.loc[fixed_date]['CRY']

        # Lagged the arbitrarily chosen date by 3 weeks. The frequency has been reduced
        # to weekly and the applied lag is three.
        lagged_date = '2011-02-04'
        date_lag = pd.Timestamp(lagged_date)
        # Weekly aggregation will return a value at the end of the business-week (each
        # date will be a Friday). Therefore, confirm the date is Friday and use the
        # preceding days to confirm the calculation manually.
        self.assertEqual(date_lag.dayofweek, 4)

        filter_1 = (df['cid'] == 'AUD') & (df['xcat'] == 'CRY')
        df_cr_aud = df[filter_1]
        df_cr_aud = df_cr_aud.pivot(index='real_date', columns='xcat', values='value')

        # Subtract to the previous Sunday to include the entire week's individual series.
        test_week = df_cr_aud.loc[date_lag - timedelta(6):lagged_date]
        # Compute the average manually and confirm the lag has been applied correctly.
        manual_calc = test_week.mean()
        condition = abs(float(test_value) - float(manual_calc))
        self.assertTrue(condition < 0.00001)

    def test_categories_df_multiple_xcats(self):
        # The method categories_df allows for multiple signals to be passed but the same
        # aggregation method will be used for all signals received. Therefore, confirm
        # the logic holds with the addition of further signals: the functionality is
        # preserved as n becomes large.

        self.dataframe_constructor()

        extra_signals = ['CRY', 'GROWTH', 'INFL', 'XR']
        ret = extra_signals[-1]
        dfc = categories_df(self.dfd, xcats=extra_signals,
                            cids=self.cids, freq='M', lag=1, fwin=0,
                            xcat_aggs=['last', 'mean'])
        # The first aspect to test is that all categories are present in the returned
        # DataFrame and that the order of the columns matches the order passed to the
        # category parameter. The dependent variable will invariably be in the right-most
        # most column with the preceding columns occupied by the signals.
        dfc_columns = list(dfc.columns)
        self.assertTrue(dfc_columns[-1] == 'XR')
        self.assertTrue(dfc_columns == extra_signals)

        # Confirm the index is monthly.
        # All categories and cross-sections start and end on the same day. This is
        # significant given the application of df.dropna(how='any'): any row with a NaN
        # value will be removed.
        earliest_date = min(self.dfd['real_date'])
        last_date = max(self.dfd['real_date'])

        dates = pd.date_range(start=earliest_date, end=last_date, freq='M')
        # Reduce to a single cross-section.
        index = dfc.loc['AUD', :].index

        self.assertTrue(len(index) == len(dates))

        # Subtract one from the manually assembled dates to adjust for the application of
        # a lag. Implicitly tests that the lag has been applied correctly.
        # Confirm that a lag, of a single day, has been applied correctly by applying
        # dropna() to the returned DataFrame. The method, categories_df(), will only
        # drop rows where none of the cross-sections have a realised value.
        dfc = dfc.dropna(how='any')
        index = dfc.loc['AUD', :].index
        self.assertTrue(len(index) == len(dates) - 1)

        # Finally, test the aggregation method. There will always be two aggregation
        # methods passed into argument 'xcat_aggs'. The second element will be applied
        # exclusively to the dependent category, and the first element will be used for
        # all the signals received. Test the above logic.

        # To test the signal categories, confirm each timestamp, in the returned
        # DataFrame, corresponds to the last value of each time-period according to the
        # down-sampling frequency.

        fixed_date = index[len(index) // 2]
        # Adjust for the lag applied.
        adj_lag = fixed_date - timedelta(days=30)
        offset = BMonthEnd()
        adj_lag = offset.rollforward(adj_lag)

        df = self.dfd
        # Test on a single cross-section.
        filt_1 = (df['real_date'] == adj_lag) & (df['cid'] == 'AUD')
        dfd_values = df[filt_1][['xcat', 'value']]
        dfd_values = dict(zip(dfd_values['xcat'], dfd_values['value']))

        # Isolate the signals.
        test_values = dfc[dfc.index.get_level_values('real_date') == fixed_date]
        test_values_sigs = test_values.loc['AUD'][extra_signals[:-1]]

        for xcat in test_values_sigs.columns:
            t_value = float(test_values_sigs[xcat])
            condition = abs(t_value - dfd_values[xcat])
            self.assertTrue(condition < 0.00001)

        # Test the return category whose summation method is mean.
        df_copy = df.copy()
        df_copy['month'] = df_copy['real_date'].dt.month
        df_copy['year'] = df_copy['real_date'].dt.year

        filt_3 = (df_copy['month'] == fixed_date.month) & \
                 (df_copy['year'] == fixed_date.year) & (df_copy['cid'] == 'AUD') & \
                 (df_copy['xcat'] == ret)
        df_copy = df_copy[filt_3]['value']

        test_value_ret = test_values.loc['AUD'][ret]
        condition = abs(float(test_value_ret) - df_copy.mean())

        self.assertTrue(condition < 0.00001)

        # Test the sum aggregation method to confirm NaN values are not falsely being
        # converted to zero which misleads analysis.

    def test_categories_df_black(self):
        self.dataframe_constructor()

        black = {'CAD': ['2014-01-01', '2014-12-31']}

        dfc = categories_df(self.dfd, xcats=['CRY', 'XR'], cids=self.cids,
                            freq='M', xcat_aggs=['mean', 'last'], blacklist=black)

        dfc_cad = dfc[np.array(dfc.reset_index(level=0)['cid']) == 'CAD']
        black_range_1 = pd.date_range(start='2014-01-01', end='2014-12-31')
        self.assertTrue(len([item for item in dfc_cad.reset_index()['real_date']
                             if item in black_range_1]) == 0)


if __name__ == '__main__':

    unittest.main()