import unittest
import random
import numpy as np
import pandas as pd
from collections import deque

from tests.simulate import make_qdf
from macrosynergy.panel.historic_vol import *
from macrosynergy.management.shape_dfs import reduce_df


class TestAll(unittest.TestCase):

    def dataframe_generator(self):

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

        dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)
        self.__dict__['dfd'] = dfd

    def test_expo_weights(self):
        lback_periods = 21
        half_life = 11
        w_series = expo_weights(lback_periods, half_life)

        self.assertIsInstance(w_series, np.ndarray)
        self.assertTrue(len(w_series) == lback_periods)  # Check correct length.
        # Check that weights add up to zero.
        self.assertTrue(sum(w_series) - 1.0 < 0.00000001)
        # Check that weights array is monotonic.
        self.assertTrue(all(w_series == sorted(w_series)))

    def test_expo_std(self):
        lback_periods = 21
        half_life = 11
        w_series = expo_weights(lback_periods, half_life)

        with self.assertRaises(AssertionError):
            data = np.random.randint(0, 25, size=lback_periods + 1)
            expo_std(data, w_series, False)

        data = np.random.randint(0, 25, size=lback_periods)
        output = expo_std(data, w_series, False)
        self.assertIsInstance(output, float)  # check type

        arr = np.array([i for i in range(1, 11)])
        pd_ewm = pd.Series(arr).ewm(halflife=5, min_periods=10).mean()[9]
        s_weights = expo_weights(len(arr), 5)
        output_expo = expo_std(arr, s_weights, True)
        self.assertAlmostEqual(output_expo, pd_ewm)  # Check value consistent with pandas calculation.

        arr = np.array([0, 0, -7, 0, 0, 0, 0, 0, 0])
        s_weights = expo_weights(len(arr), 5)
        output_expo = expo_std(arr, s_weights, True)
        # Check if single non-zero value becomes average.
        self.assertEqual(output_expo, 7)

    def test_flat_std(self):
        data = [2, -11, 9, -3, 1, 17, 19]
        output_flat = float(flat_std(data, remove_zeros=False))
        output_flat = round(output_flat, ndigits=6)
        data = [abs(elem) for elem in data]
        output_test = round(sum(data) / len(data), 6)
        self.assertEqual(output_flat, output_test)  # test correct average

        lback_periods = 21
        data = np.random.randint(0, 25, size=lback_periods)

        output = flat_std(data, True)
        self.assertIsInstance(output, float)  # test type

    def test_historic_vol(self):

        self.dataframe_generator()
        xcat = 'XR'

        lback_periods = 21
        df_output = historic_vol(self.dfd, xcat, self.cids, lback_periods=lback_periods,
                                 lback_meth='ma', half_life=11, start=None,
                                 end=None, blacklist=None, remove_zeros=True,
                                 postfix='ASD')

        # Test correct column names.
        self.assertTrue(all(df_output.columns == self.dfd.columns))
        cross_sections = sorted(list(set(df_output['cid'].values)))
        self.assertTrue(cross_sections == self.cids)
        self.assertTrue(all(df_output['xcat'] == xcat + 'ASD'))

        # Test the stacking procedure to reconstruct the standardised dataframe from the
        # pivoted counterpart.
        # The in-built pandas method, df.stack(), used will, by default, drop
        # all NaN values, as the preceding pivoting operation requires populating each
        # column field such that each field is defined over the same index (time-period).
        # Therefore, the stack() method treats NaN values as contrived inputs generated
        # from the pivot mechanism, and subsequently the respective dates of the lookback
        # period will also be dropped.
        # The overall outcome is that the returned standardised dataframe should be
        # reduced by the number cross-sections multiplied by the length of the lookback
        # period minus one.
        # Test the above logic.

        # Reduce the dataframe to the singular xcat to test the dimensionality reduction.
        df_reduce = reduce_df(self.dfd, xcats=[xcat], cids=self.cids, start=None,
                              end=None, blacklist=None)
        no_rows_input = df_reduce.shape[0]
        no_rows_output = df_output.shape[0]
        difference = len(self.cids) * (lback_periods - 1)

        self.assertTrue((no_rows_input - no_rows_output) == difference)

        with self.assertRaises(AssertionError):
            historic_vol(self.dfd, 'XR', self.cids, lback_periods=7, lback_meth='ma',
                         half_life=11, start=None, end=None, blacklist=None,
                         remove_zeros=True, postfix='ASD')

        with self.assertRaises(AssertionError):
            historic_vol(self.dfd, 'CRY', self.cids, lback_periods=7, lback_meth='ema',
                         half_life=11, start=None, end=None, blacklist=None,
                         remove_zeros=True, postfix='ASD')

        # Todo: check correct exponential averages for a whole series on toy data set using (.rolling) and .ewm


if __name__ == '__main__':

    unittest.main()