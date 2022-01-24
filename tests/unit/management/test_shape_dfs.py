
import unittest
import random
import numpy as np
import pandas as pd
from tests.simulate import make_qdf
from macrosynergy.management.shape_dfs import reduce_df, categories_df
from math import ceil, floor


class TestAll(unittest.TestCase):

    def dataframe_constructor(self):

        self.__dict__['cids'] = ['AUD', 'CAD', 'GBP']
        self.__dict__['xcats'] = ['CRY', 'XR']

        df_cids = pd.DataFrame(index=self.cids,
                               columns=['earliest', 'latest', 'mean_add', 'sd_mult'])
        df_cids.loc['AUD', :] = ['2010-01-01', '2020-12-31', 0.5, 2]
        df_cids.loc['CAD', :] = ['2011-01-01', '2020-11-30', 0, 1]
        df_cids.loc['GBP', :] = ['2012-01-01', '2020-11-30', -0.2, 0.5]

        df_xcats = pd.DataFrame(index=self.xcats,
                                columns=['earliest', 'latest', 'mean_add', 'sd_mult',
                                         'ar_coef', 'back_coef'])
        df_xcats.loc['CRY', :] = ['2011-01-01', '2020-10-30', 1, 2, 0.9, 0.5]
        df_xcats.loc['XR', :] = ['2010-01-01', '2020-12-31', 0, 1, 0, 0.3]

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
        self.assertTrue(dfd_aud['real_date'].min() == pd.to_datetime('2010-01-01'))

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

        # Check correct application of monthly mean: the procedure used to compute the
        # value for each defined frequency is to isolate the last value of each
        # respective period.
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
        # dfc = categories_df(dfd, xcats=['XR', 'CRY'], cids=['CAD'],
                            # freq='M', lag=0, xcat_aggs=['mean', 'mean'],
                            # start='2000-01-01', years=years)

        realised_buckets = no_buckets - start_bucket
        # filter_df = dfc.loc['CAD', :]

        # self.assertTrue(realised_buckets == filter_df.shape[0])

        # Apply the same logic but to a different testcase.
        years = 4
        no_buckets = ceil((e_year - start) / years)

        # Adjust for the formal start date.
        start_bucket = int(floor(s_year - start) / years)

        # dfc = categories_df(dfd, xcats=['XR', 'CRY'], cids=['CAD'],
                            # freq='M', lag=0, xcat_aggs=['mean', 'mean'],
                            # start='2000-01-01', years=years)

        realised_buckets = no_buckets - start_bucket
        # filter_df = dfc.loc['CAD', :]

        # self.assertTrue(realised_buckets == filter_df.shape[0])

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

    def test_categories_df_black(self):
        self.dataframe_constructor()

        black = {'CAD': ['2014-01-01', '2014-12-31']}
        dfc = categories_df(self.dfd, xcats=['XR', 'CRY'], cids=self.cids,
                            freq='M', xcat_aggs=['mean', 'last'], blacklist=black)

        dfc_cad = dfc[np.array(dfc.reset_index(level=0)['cid']) == 'CAD']
        black_range_1 = pd.date_range(start='2014-01-01', end='2014-12-31')
        self.assertTrue(len([item for item in dfc_cad.reset_index()['real_date']
                             if item in black_range_1]) == 0)


if __name__ == '__main__':

    unittest.main()