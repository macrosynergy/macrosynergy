

import unittest
import random
import numpy as np
import pandas as pd
from collections import deque

from macrosynergy.management.simulate_quantamental_data import make_qdf
from macrosynergy.panel.historic_vol import expo_weights, expo_std, flat_std, historic_vol

cids = ['AUD', 'CAD', 'GBP']
xcats = ['CRY', 'XR']
df_cids = pd.DataFrame(index=cids, columns=['earliest', 'latest', 'mean_add', 'sd_mult'])
df_cids.loc['AUD', :] = ['2010-01-01', '2020-12-31', 0.5, 2]
df_cids.loc['CAD', :] = ['2011-01-01', '2020-11-30', 0, 1]
df_cids.loc['GBP', :] = ['2012-01-01', '2020-11-30', -0.2, 0.5]

df_xcats = pd.DataFrame(index=xcats, columns=['earliest', 'latest', 'mean_add', 'sd_mult', 'ar_coef', 'back_coef'])
df_xcats.loc['CRY', :] = ['2011-01-01', '2020-10-30', 1, 2, 0.9, 0.5]
df_xcats.loc['XR', :] = ['2010-01-01', '2020-12-31', 0, 1, 0, 0.3]

dfd = make_qdf(df_cids, df_xcats, back_ar = 0.75)

class TestAll(unittest.TestCase):

    def test_expo_weights(self):
        lback_periods = 21
        half_life = 11
        w_series = expo_weights(lback_periods, half_life)

        self.assertIsInstance(w_series, np.ndarray)
        self.assertTrue(len(w_series) == lback_periods)
        self.assertTrue(sum(w_series) - 1.0 < 0.00000001)
        self.assertTrue(all(w_series == sorted(w_series)))

    def test_expo_std(self):
        lback_periods = 21
        half_life = 11
        w_series = expo_weights(lback_periods, half_life)

        with self.assertRaises(AssertionError):
            data = np.random.randint(0, 25, size = lback_periods + 1)
            expo_std(data, w_series, False)

        data = np.random.randint(0, 25, size = lback_periods)
        output = expo_std(data, w_series, False)
        self.assertIsInstance(output, float)

        ## Contrived data set to test the zero functionality where the entire weighting is applied to the singular non-zero value.
        arr = [0, 0, -7, 0, 0, 0, 0, 0, 0]
        arr = np.array(arr)
        s_weights = expo_weights(len(arr), 5)
        output_expo = expo_std(arr, s_weights, True)
        self.assertEqual(output_expo, 7)
        

    def test_flat_std(self):
        data = [2, -11, 9, -3, 1, 17, 19]
        output_flat = float(flat_std(data, remove_zeros = False))
        output_flat = round(output_flat, ndigits = 6)
        data = [abs(elem) for elem in data]
        output_test = round(sum(data) / len(data), 6)
        self.assertEqual(output_flat, output_test)

        lback_periods = 21
        data = np.random.randint(0, 25, size = lback_periods)

        output = flat_std(data, True)
        self.assertIsInstance(output, float)

    def test_historic_vol(self):
        xcat = 'XR'
        
        df_output = historic_vol(dfd, xcat, cids, lback_periods=21, lback_meth='ma', half_life=11, start=None,
                    end=None, blacklist=None, remove_zeros=True, postfix='ASD')
        self.assertTrue(all(df_output.columns == dfd.columns))
        cross_sections = sorted(list(set(df_output['cid'].values)))
        self.assertTrue(cross_sections == cids)
        self.assertTrue(all(df_output['xcat'] == xcat + 'ASD'))

        with self.assertRaises(AssertionError):
            historic_vol(dfd, 'XR', cids, lback_periods=7, lback_meth='ma', half_life=11, start=None,
                         end = None, blacklist = None, remove_zeros=True, postfix='ASD')

        with self.assertRaises(AssertionError):
            historic_vol(dfd, 'CRY', cids, lback_periods=7, lback_meth='ema', half_life=11, start=None,
                         end=None, blacklist = None, remove_zeros=True, postfix='ASD')

if __name__ == '__main__':
    unittest.main()