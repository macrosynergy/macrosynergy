
import unittest
import numpy as np
import pandas as pd
import warnings
from tests.simulate import make_qdf
from macrosynergy.management.shape_dfs import reduce_df
from macrosynergy.panel.make_zn_scores import *
from random import randint, choice, shuffle, seed
from collections import defaultdict

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)

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

    def test_pan_neutral(self):

        self.dataframe_construction()

        ar_neutral = pan_neutral(self.dfw, neutral='mean', sequential=True)
        self.assertIsInstance(ar_neutral, np.ndarray)  # Check type of output.
        # Test length of neutral array.
        self.assertTrue(self.dfw.shape[0] == len(ar_neutral))

        ar_neutral = pan_neutral(self.dfw, neutral='mean', sequential=False)
        # Check first value equal to panel mean.
        self.assertEqual(ar_neutral[0], self.dfw.stack().mean())
        # Check also last value equal to panel mean.
        self.assertEqual(ar_neutral[self.dfw.shape[0]-1], self.dfw.stack().mean())

        ar_neutral = pan_neutral(self.dfw, neutral='mean', sequential=True)
        self.assertEqual(ar_neutral[999], self.dfw.iloc[0:1000, :].stack().mean())

        ar_neutral = pan_neutral(self.dfw, neutral='median', sequential=False)
        # Check first value equal to panel median.
        self.assertEqual(ar_neutral[0], self.dfw.stack().median())
        # Check last value equal to panel median
        self.assertEqual(ar_neutral[self.dfw.shape[0]-1], self.dfw.stack().median())

        ar_neutral = pan_neutral(self.dfw, neutral='median', sequential=True)
        self.assertEqual(ar_neutral[999], self.dfw.iloc[0:1000, :].stack().median())

    @staticmethod
    def handle_nan(arr):
        arr = np.nan_to_num(arr)
        arr = arr[arr != 0.0]
        return arr

    def test_cross_neutral(self):

        self.dataframe_construction()

        arr_neutral = cross_neutral(self.dfw, 'mean', True)
        self.assertIsInstance(arr_neutral, np.ndarray)  # check correct type

        df_shape = self.dfw.shape
        self.assertEqual(df_shape, arr_neutral.shape)  # check correct dimensions

        epsilon = 0.0000001

        ar_mean = cross_neutral(self.dfw, neutral='mean', sequential=False)
        ar_median = cross_neutral(self.dfw, neutral='median', sequential=False)
        for i, cross in enumerate(self.cids):
            column = self.dfw[[cross]].to_numpy()
            column = np.squeeze(column, axis=1)
            column = self.handle_nan(column)
            
            mean = np.sum(column) / len(column)
            dif = np.abs(ar_mean[:, i] - mean)
            self.assertTrue(np.all(dif < epsilon))  # Test if function mean is correct.

            median = np.median(column)
            dif = np.abs(ar_median[:, i] - median)
            self.assertTrue(np.all(dif < epsilon))  # Test if function median is correct.

        ar_mean = cross_neutral(self.dfw, neutral='mean', sequential=True)
        ar_median = cross_neutral(self.dfw, neutral='median', sequential=True)
        for i, cross in enumerate(self.cids):
            
            column = self.dfw[[cross]]

            cum_mean = column.expanding(min_periods=1).mean()
            cum_mean = self.handle_nan(cum_mean[cross].to_numpy())
            dif = self.handle_nan(ar_mean[:, i]) - cum_mean
            self.assertTrue(np.all(dif < epsilon))  # Check correct cumulative means.

            cum_median = column.expanding(min_periods=1).median()
            cum_median = self.handle_nan(cum_median[cross].to_numpy())
            # Check correct cumulative median.
            self.assertTrue(np.all(self.handle_nan(ar_median[:, i]) == cum_median))

    def test_nan_insert(self):
        self.dataframe_construction()

        min_obs = 3
        dfw_zns = nan_insert(self.dfw, min_obs, iis=False)  # Test DataFrame.

        # Determine where the indices of the first active value.
        data = self.dfw.to_numpy()
        nan_arr = np.isnan(data)
        indices = np.where(nan_arr == False)
        indices_d = tuple(zip(indices[1], indices[0]))
        indices_dict = defaultdict(list)
        for tup in indices_d:
            indices_dict[tup[0]].append(tup[1])

        active_indices = {}
        for k, v in indices_dict.items():
            active_indices[k] = v[0] + min_obs

        for k, v in active_indices.items():
            col = dfw_zns.iloc[:, k]
            first_val = col.first_valid_index()
            self.assertTrue(v, first_val)

        # If the "iis" is set to True, the dimensions of the dataframe should remain the
        # same. The make_zn_scores function will compute a zn_score for all dates:
        # starting from the first available date. Therefore, leave the code unmodified.
        self.assertTrue(self.dfw.shape == dfw_zns.shape)

    def test_zn_scores(self):

        self.dataframe_construction()

        with self.assertRaises(AssertionError):
            # Test catching neutral value error.
            df = make_zn_scores(self.dfd, 'XR', self.cids, sequential=False, neutral='std',
                                thresh=1.5, postfix='ZN')
        with self.assertRaises(AssertionError):
            # Test catching non-valid thresh value.
            df = make_zn_scores(self.dfd, 'XR', self.cids, sequential=False,
                                neutral='std', thresh=0.5, pan_weight=1.0, postfix='ZN')

        with self.assertRaises(AssertionError):
            # Test catching panel weight.
            df = make_zn_scores(self.dfd, 'XR', self.cids, sequential=False,
                                pan_weight=1.2)

        with self.assertRaises(AssertionError):
            # Test the iis parameter being a boolean value.
            df = make_zn_scores(self.dfd, 'XR', self.cids, sequential=False,
                                pan_weight=1.2, iis=0)

        # Testing on Panel = 1.0 (default value)
        df_panel = make_zn_scores(self.dfd, 'CRY', self.cids, sequential=True, min_obs=0,
                                  neutral='mean', thresh=None, postfix='ZN')

        df_panel = df_panel.pivot(index='real_date', columns='cid', values='value')
        
        ar_neutral = pan_neutral(self.dfw, 'mean', True)
        dfx = self.dfw.sub(ar_neutral, axis='rows')
        ar_sds = np.array([dfx.iloc[0:(i + 1), :].stack().abs().mean()
                           for i in range(dfx.shape[0])])
        dfw_zns_pan = dfx.div(ar_sds, axis='rows')
        dfw_zns_pan = dfw_zns_pan.dropna(axis = 0, how='all')

        zn_scores = df_panel.to_numpy()
        arr_zns_pan = dfw_zns_pan.to_numpy()
        dif = zn_scores - arr_zns_pan
        dif = np.nan_to_num(dif, nan = 0.0)

        epsilon = 0.000001
        self.assertTrue(np.all(dif < epsilon))

        # Test weighting function.
        panel_df = make_zn_scores(self.dfd, 'CRY', self.cids, start="2010-01-04",
                                  sequential=True, min_obs=252, neutral='mean',
                                  thresh=None, pan_weight=1.0, postfix='ZN')
        df_cross = make_zn_scores(self.dfd, 'CRY', self.cids, start="2010-01-04",
                                  sequential=True, min_obs=252, neutral='mean',
                                  thresh=None, pan_weight=0.0, postfix='ZN')

        df_average = make_zn_scores(self.dfd, 'CRY', self.cids, start="2010-01-04",
                                    sequential=True, min_obs=252,
                                    neutral='mean', thresh=None, pan_weight=0.5,
                                    postfix='ZN')

        panel_df = panel_df.pivot(index='real_date', columns='cid', values='value')
        df_cross = df_cross.pivot(index='real_date', columns='cid', values='value')
        df_average = df_average.pivot(index='real_date', columns='cid', values='value')

        # Drop the first row in the panel data.
        panel_df = panel_df.drop(panel_df.index[[0]])
        df_check = (panel_df + df_cross) / 2
        check_arr = df_check.to_numpy()
        average_arr = df_average.to_numpy()

        dif = check_arr - average_arr
        dif = np.nan_to_num(dif, nan = 0.0)
        self.assertTrue(np.all(dif < epsilon))

        threshold = 2.35
        df_thresh = make_zn_scores(self.dfd, 'CRY', self.cids, start="2010-01-01",
                                   sequential=True, min_obs=252, neutral='mean',
                                   thresh=threshold, pan_weight=0.65, postfix='ZN')

        df_thresh = df_thresh.pivot(index='real_date', columns='cid', values='value')
        thresh_arr = df_thresh.to_numpy()
        # Compress multidimensional array into a one-dimensional array.
        values = thresh_arr.ravel()

        check = sum(values[~np.isnan(values)] > threshold)

        self.assertTrue(check == 0)

        
if __name__ == '__main__':

    unittest.main()