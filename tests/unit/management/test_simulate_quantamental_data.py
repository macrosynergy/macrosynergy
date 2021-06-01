import unittest
import random
import numpy as np
import pandas as pd
from macrosynergy.management.simulate_quantamental_data import make_qdf, simulate_ar

cids = ['AUD', 'CAD', 'GBP']
xcats = ['XR', 'CRY']
df_cids = pd.DataFrame(index=cids, columns=['earliest', 'latest', 'mean_add', 'sd_mult'])
df_cids.loc['AUD', ] = ['2010-01-01', '2020-12-31', 0.5, 2]
df_cids.loc['CAD', ] = ['2011-01-01', '2020-11-30', 0, 1]
df_cids.loc['GBP', ] = ['2011-01-01', '2020-11-30', -0.2, 0.5]

df_xcats = pd.DataFrame(index=xcats, columns=['earliest', 'latest', 'mean_add', 'sd_mult', 'ar_coef', 'back_coef'])
df_xcats.loc['XR', ] = ['2010-01-01', '2020-12-31', 0, 1, 0, 0.3]
df_xcats.loc['CRY', ] = ['2011-01-01', '2020-10-30', 1, 2, 0.9, 0.5]

random.seed(1)
dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)


def ar1_coef(x):
    return np.corrcoef(np.array([x[:-1], x[1:]]))[0, 1]


def cor_coef(df, ticker_x, ticker_y):
    x = ticker_x.split('_', 1)
    y = ticker_y.split('_', 1)
    filt_x = (df['cid'] == x[0]) & (dfd['xcat'] == x[1])
    filt_y = (df['cid'] == y[0]) & (dfd['xcat'] == y[1])
    dfd_x = dfd.loc[filt_x,].set_index('real_date')['value']
    dfd_y = dfd.loc[filt_y,].set_index('real_date')['value']
    dfd_xy = pd.merge(dfd_x, dfd_y, how='inner', left_index=True, right_index=True)
    return dfd_xy.corr().iloc[0, 1]


class Test_All(unittest.TestCase):

    def test_simulate_ar(self):

        random.seed(1)
        ser_ar = simulate_ar(100, mean=2, sd_mult=3, ar_coef=0.75)
        self.assertGreater(ar1_coef(ser_ar), 0.25)
        self.assertEqual(np.round(np.std(ser_ar), 2), 3)
        self.assertGreater(np.mean(ser_ar), 2)

    def test_qdf_starts(self):

        filt1 = (dfd['cid'] == 'AUD') & (dfd['xcat'] == 'CRY')
        self.assertEqual(np.min(dfd.loc[filt1, 'real_date']), pd.to_datetime('2011-01-03'))
        filt1 = (dfd['cid'] == 'GBP') & (dfd['xcat'] == 'XR')
        self.assertEqual(np.min(dfd.loc[filt1, 'real_date']), pd.to_datetime('2011-01-03'))

    def test_qdf_ends(self):

        filt1 = (dfd['cid'] == 'AUD') & (dfd['xcat'] == 'CRY')
        self.assertEqual(np.max(dfd.loc[filt1, 'real_date']), pd.to_datetime('2020-10-30'))
        filt1 = (dfd['cid'] == 'GBP') & (dfd['xcat'] == 'XR')
        self.assertEqual(np.max(dfd.loc[filt1, 'real_date']), pd.to_datetime('2020-11-30'))

    def test_qdf_correl(self):

        self.assertGreater(cor_coef(dfd, 'AUD_XR', 'CAD_XR'), 0)
        self.assertGreater(cor_coef(dfd, 'AUD_XR', 'GBP_XR'), 0)

    def test_qdf_ar(self):

        filt1 = (dfd['cid'] == 'AUD') & (dfd['xcat'] == 'CRY')
        self.assertGreater(ar1_coef(dfd.loc[filt1, 'value']), 0.25)
        filt1 = (dfd['cid'] == 'CAD') & (dfd['xcat'] == 'CRY')
        self.assertGreater(ar1_coef(dfd.loc[filt1, 'value']), 0.25)
        filt1 = (dfd['cid'] == 'GBP') & (dfd['xcat'] == 'CRY')
        self.assertGreater(ar1_coef(dfd.loc[filt1, 'value']), 0.25)


if __name__ == '__main__':

    unittest.main()
