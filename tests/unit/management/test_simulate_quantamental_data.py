import unittest
import random
import numpy as np
import pandas as pd
from macrosynergy.management.simulate_quantamental_data import *


class Test_All(unittest.TestCase):

    def df_construction(self):
        cids = ['AUD', 'CAD', 'GBP']
        xcats = ['XR', 'CRY']
        df_cids = pd.DataFrame(index=cids, columns=['earliest', 'latest', 'mean_add',
                                                    'sd_mult'])
        df_cids.loc['AUD', :] = ['2010-01-01', '2020-12-31', 0.5, 2]
        df_cids.loc['CAD', :] = ['2011-01-01', '2020-11-30', 0, 1]
        df_cids.loc['GBP', :] = ['2011-01-01', '2020-11-30', -0.2, 0.5]

        df_xcats = pd.DataFrame(index=xcats, columns=['earliest', 'latest', 'mean_add',
                                                      'sd_mult', 'ar_coef', 'back_coef'])

        df_xcats.loc['XR', :] = ['2010-01-01', '2020-12-31', 0, 1, 0, 0.3]
        df_xcats.loc['CRY', :] = ['2011-01-01', '2020-10-30', 1, 2, 0.9, 0.5]

        random.seed(1)
        self.dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)

    @staticmethod
    def ar1_coef(x):
        return np.corrcoef(np.array([x[:-1], x[1:]]))[0, 1]

    def cor_coef(self, df, ticker_x, ticker_y):
        x = ticker_x.split('_', 1)
        y = ticker_y.split('_', 1)
        filt_x = (df['cid'] == x[0]) & (self.dfd['xcat'] == x[1])
        filt_y = (df['cid'] == y[0]) & (self.dfd['xcat'] == y[1])
        dfd_x = self.dfd.loc[filt_x, ].set_index('real_date')['value']
        dfd_y = self.dfd.loc[filt_y, ].set_index('real_date')['value']

        dfd_xy = pd.merge(dfd_x, dfd_y, how='inner', left_index=True, right_index=True)
        return dfd_xy.corr().iloc[0, 1]

    def test_simulate_ar(self):

        random.seed(1)
        ser_ar = simulate_ar(100, mean=2, sd_mult=3, ar_coef=0.75)
        self.assertGreater(self.ar1_coef(ser_ar), 0.25)
        self.assertEqual(np.round(np.std(ser_ar), 2), 3)
        self.assertGreater(np.mean(ser_ar), 2)

    def test_qdf_starts(self):

        self.df_construction()
        # Utilise ampersand for element - wise logical - "and". Will return a Boolean
        # Pandas Series or Numpy Array.
        filt1 = (self.dfd['cid'] == 'AUD') & (self.dfd['xcat'] == 'CRY')
        self.assertEqual(np.min(self.dfd.loc[filt1, 'real_date']),
                         pd.to_datetime('2011-01-03'))

        filt1 = (self.dfd['cid'] == 'GBP') & (self.dfd['xcat'] == 'XR')
        self.assertEqual(np.min(self.dfd.loc[filt1, 'real_date']),
                         pd.to_datetime('2011-01-03'))

    def test_qdf_ends(self):

        self.df_construction()
        filt1 = (self.dfd['cid'] == 'AUD') & (self.dfd['xcat'] == 'CRY')
        self.assertEqual(np.max(self.dfd.loc[filt1, 'real_date']),
                         pd.to_datetime('2020-10-30'))

        filt1 = (self.dfd['cid'] == 'GBP') & (self.dfd['xcat'] == 'XR')
        self.assertEqual(np.max(self.dfd.loc[filt1, 'real_date']),
                         pd.to_datetime('2020-11-30'))

    def test_qdf_correl(self):

        self.df_construction()
        self.assertGreater(self.cor_coef(self.dfd, 'AUD_XR', 'CAD_XR'), 0)
        self.assertGreater(self.cor_coef(self.dfd, 'AUD_XR', 'GBP_XR'), 0)

    def test_qdf_ar(self):

        self.df_construction()
        filt1 = (self.dfd['cid'] == 'AUD') & (self.dfd['xcat'] == 'CRY')
        self.assertGreater(self.ar1_coef(self.dfd.loc[filt1, 'value']), 0.25)

        filt1 = (self.dfd['cid'] == 'CAD') & (self.dfd['xcat'] == 'CRY')
        self.assertGreater(self.ar1_coef(self.dfd.loc[filt1, 'value']), 0.25)

        filt1 = (self.dfd['cid'] == 'GBP') & (self.dfd['xcat'] == 'CRY')
        self.assertGreater(self.ar1_coef(self.dfd.loc[filt1, 'value']), 0.25)


if __name__ == '__main__':

    unittest.main()
