import unittest
import random
import numpy as np
import pandas as pd

from tests.simulate import make_qdf
from macrosynergy.management.shape_dfs import reduce_df, categories_df


cids = ['AUD', 'CAD', 'GBP']
xcats = ['CRY', 'XR']
df_cids = pd.DataFrame(index=cids, columns=['earliest', 'latest', 'mean_add', 'sd_mult'])
df_cids.loc['AUD', :] = ['2010-01-01', '2020-12-31', 0.5, 2]
df_cids.loc['CAD', :] = ['2011-01-01', '2020-11-30', 0, 1]
df_cids.loc['GBP', :] = ['2012-01-01', '2020-11-30', -0.2, 0.5]

df_xcats = pd.DataFrame(index=xcats, columns=['earliest', 'latest', 'mean_add', 'sd_mult', 'ar_coef', 'back_coef'])
df_xcats.loc['CRY', :] = ['2011-01-01', '2020-10-30', 1, 2, 0.9, 0.5]
df_xcats.loc['XR', :] = ['2010-01-01', '2020-12-31', 0, 1, 0, 0.3]

random.seed(1)
np.random.seed(0)
dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)


class TestAll(unittest.TestCase):
    def test_reduce_df_general(self):
        dfd_x = reduce_df(dfd, xcats=['CRY'], cids=cids[0:2], start='2013-01-01', end='2019-01-01')
        self.assertTrue(all(dfd_x['cid'].unique() == np.array(['AUD', 'CAD'])))
        self.assertTrue(all(dfd_x['xcat'].unique() == np.array(['CRY'])))
        self.assertTrue(dfd_x['real_date'].min() == pd.to_datetime('2013-01-01'))
        self.assertTrue(dfd_x['real_date'].max() == pd.to_datetime('2019-01-01'))

    def test_reduce_df_intersect(self):
        filt1 = ~((dfd['cid'] == 'AUD') & (dfd['xcat'] == 'XR'))
        dfdx = dfd[filt1]  # simulate missing cross sections
        dfd_x1, xctx, cidx = reduce_df(dfdx, xcats=['XR', 'CRY'], cids=cids, intersect=True, out_all=True)
        self.assertTrue(cidx == ['CAD', 'GBP'])

    def test_reduce_df_black(self):
        black = {'AUD_1': ['2011-01-01', '2012-12-31'], 'AUD_2': ['2018-01-01', '2100-01-01']}
        dfd_x = reduce_df(dfd, xcats=['XR'], cids=cids, blacklist=black)
        dfd_aud = dfd_x[dfd_x['cid'] == 'AUD']
        self.assertTrue(dfd_aud['real_date'].min() == pd.to_datetime('2010-01-01'))
        black_range_1 = pd.date_range(start='2011-01-01', end='2012-12-31')
        self.assertTrue(not any(item in black_range_1 for item in dfd_aud['real_date']))

        black = {'CAD': ['2014-01-01', '2014-12-31']}
        dfd_x = reduce_df(dfd, xcats=['XR'], cids=cids, blacklist=black)
        dfd_cad = dfd_x[dfd_x['cid'] == 'CAD']
        black_range_2 = pd.date_range(start='2014-01-01', end='2014-12-31')
        self.assertTrue(not any(item in black_range_2 for item in dfd_cad['real_date']))

    def test_categories_df_conversion(self):
        dfc = categories_df(dfd, xcats=['XR', 'CRY'], cids=cids, freq='M', lag=0, xcat_aggs=['mean', 'last'])

        filt1 = (dfd['real_date'].dt.year == 2011) & (dfd['real_date'].dt.month == 10)
        filt2 = (dfd['cid'] == 'AUD') & (dfd['xcat'] == 'XR')
        x1 = dfd[filt1 & filt2]['value'].mean()
        x2 = float(dfc.loc[('AUD', '2011-10-31'), 'XR'])
        self.assertAlmostEqual(x1, x2)  # check correct application of monthly mean

        filt2 = (dfd['cid'] == 'AUD') & (dfd['xcat'] == 'CRY')
        x1 = dfd[filt1 & filt2]['value'].iloc[-1]
        x2 = float(dfc.loc[('AUD', '2011-10-31'), 'CRY'])
        self.assertAlmostEqual(x1, x2)  # check correct application of end-of-month aggregation

    def test_categories_df_lags(self):
        dfc = categories_df(dfd, xcats=['XR', 'CRY'], cids=cids, freq='M', lag=1, fwin=3, xcat_aggs=['mean', 'last'])

        filt1 = (dfd['real_date'].dt.year == 2013) & (dfd['real_date'].dt.month.isin([10, 11, 12]))
        filt2 = (dfd['cid'] == 'AUD') & (dfd['xcat'] == 'XR')
        x1 = round(float(np.mean(dfd[filt1 & filt2].set_index('real_date').resample('M').mean())), 10)
        x2 = round(float(dfc.loc[('AUD', '2013-10-31'), 'XR']), 10)
        self.assertAlmostEqual(x1, x2)  # check correct application of 1st series forward (average) window

        filt1 = (dfd['real_date'].dt.year == 2011) & (dfd['real_date'].dt.month == 10)
        filt2 = (dfd['cid'] == 'AUD') & (dfd['xcat'] == 'CRY')
        x1 = dfd[filt1 & filt2]['value'].iloc[-1]
        x2 = float(dfc.loc[('AUD', '2011-11-30'), 'CRY'])
        self.assertAlmostEqual(x1, x2)  # check correct application of 2nd series lag

    def test_categories_df_black(self):
        black = {'CAD': ['2014-01-01', '2014-12-31']}
        dfc = categories_df(dfd, xcats=['XR', 'CRY'], cids=cids, freq='M', xcat_aggs=['mean', 'last'], blacklist=black)
        dfc_cad = dfc[np.array(dfc.reset_index(level=0)['cid']) == 'CAD']
        black_range_1 = pd.date_range(start='2014-01-01', end='2014-12-31')
        self.assertTrue(len([item for item in dfc_cad.reset_index()['real_date'] if item in black_range_1]) == 0)


if __name__ == '__main__':

    unittest.main()