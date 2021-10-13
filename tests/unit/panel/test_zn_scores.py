
import unittest
import numpy as np
import pandas as pd
from itertools import groupby
from random import randint, choice, shuffle, seed
from collections import defaultdict
import warnings

from macrosynergy.management.simulate_quantamental_data import make_qdf
from macrosynergy.management.shape_dfs import reduce_df
from make_zn_scores import pan_neutral, cross_neutral, make_zn_scores, nan_insert


cids = ['AUD', 'CAD', 'GBP']
xcats = ['CRY', 'XR']
df_cids = pd.DataFrame(index=cids, columns=['earliest', 'latest', 'mean_add', 'sd_mult'])
df_cids.loc['AUD', :] = ['2010-01-01', '2020-12-31', 0.5, 2]
df_cids.loc['CAD', :] = ['2010-01-01', '2020-11-30', 0, 1]
df_cids.loc['GBP', :] = ['2012-01-01', '2020-11-30', -0.2, 0.5]

df_xcats = pd.DataFrame(index=xcats, columns=['earliest', 'latest', 'mean_add', 'sd_mult', 'ar_coef', 'back_coef'])
df_xcats.loc['CRY', :] = ['2010-01-01', '2020-10-30', 1, 2, 0.9, 0.5]
df_xcats.loc['XR', :] = ['2011-01-01', '2020-12-31', 0, 1, 0, 0.3]

dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)  # standard df for tests
dfd = dfd[dfd['xcat']=='CRY']
dfw = dfd.pivot(index='real_date', columns='cid', values='value')

warnings.filterwarnings("ignore")
class TestAll(unittest.TestCase):

    def test_pan_neutral(self):

        ar_neutral = pan_neutral(dfw, neutral='mean', sequential=True)
        self.assertIsInstance(ar_neutral, np.ndarray)  # check type of output
        self.assertTrue(dfw.shape[0] == len(ar_neutral))  # test length of neutral array

        ar_neutral = pan_neutral(dfw, neutral='mean', sequential=False)
        self.assertEqual(ar_neutral[0], dfw.stack().mean())  # check first value equal to panel mean
        self.assertEqual(ar_neutral[dfw.shape[0]-1], dfw.stack().mean())  # check also last value equal to panel mean

        ar_neutral = pan_neutral(dfw, neutral='mean', sequential=True)
        self.assertEqual(ar_neutral[999], dfw.iloc[0:1000, :].stack().mean())

        # Todo: same for median
        ar_neutral = pan_neutral(dfw, neutral='median', sequential=False)
        self.assertEqual(ar_neutral[0], dfw.stack().median())  # check first value equal to panel median
        self.assertEqual(ar_neutral[dfw.shape[0]-1], dfw.stack().median())  # check also last value equal to panel median

        ar_neutral = pan_neutral(dfw, neutral='median', sequential=True)
        self.assertEqual(ar_neutral[999], dfw.iloc[0:1000, :].stack().median())

    @staticmethod
    def handle_nan(arr):
        arr = np.nan_to_num(arr)
        arr = arr[arr != 0.0]
        return arr

    def test_cross_neutral(self):

        arr_neutral = cross_neutral(dfw, 'mean', True)
        self.assertIsInstance(arr_neutral, np.ndarray)  # check correct type

        df_shape = dfw.shape
        self.assertEqual(df_shape, arr_neutral.shape)  # check correct dimensions

        epsilon = 0.0000001
        # Check the cross sectional feature: computation occurs over individual columns.
        ar_neutral = cross_neutral(dfw, neutral='mean', sequential=False)
        for i, cross in enumerate(cids):
            column = dfw[[cross]]
            column = column.to_numpy()
            column = np.squeeze(column, axis = 1)
            column = self.handle_nan(column)
            
            mean = np.sum(column) / len(column)

            dif = np.abs(ar_neutral[:, i] - mean)
            self.assertTrue(np.all(dif < epsilon))

        # Check the rolling feature on cross-sectional computation.
        ar_neutral = cross_neutral(dfw, neutral='mean', sequential=True)
        for i, cross in enumerate(cids):
            
            column = dfw[[cross]]
            rol_mean = column.expanding(min_periods = 1).mean()
            rol_mean = self.handle_nan(rol_mean[cross].to_numpy())

            dif = self.handle_nan(ar_neutral[:, i]) - rol_mean
            self.assertTrue(np.all(dif < epsilon))


        ar_neutral = cross_neutral(dfw, neutral='median', sequential=True)
        for i, cross in enumerate(cids):
            column = dfw[[cross]]
            rol_median = column.expanding(min_periods = 1).median()
            rol_median = self.handle_nan(rol_median[cross].to_numpy())
            
            self.assertTrue(np.all(self.handle_nan(ar_neutral[:, i]) == rol_median))


    def test_nan_insert(self):

        # Todo: short code testing first non-NA with or without min_obs across columns: Series.first_valid_index().
        min_obs = 3
        dfw_zns = nan_insert(dfw, min_obs)  # Test DataFrame.

        # Determine where the indices of the first active value.
        data = dfw.to_numpy()
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
            

    def test_zn_scores(self):

        epsilon = 0.0000001
        dfd = make_qdf(df_cids, df_xcats, back_ar=0.75) 
        dfd = dfd[dfd['xcat']=='CRY']
        dfw = dfd.pivot(index='real_date', columns='cid', values='value')

        ## Using the globally defined DataFrame.
        with self.assertRaises(AssertionError):
            df = make_zn_scores(dfd, 'XR', cids, sequential=False, neutral='std',
                                thresh=1.5, postfix='ZN')  # test catching neutral value error
        with self.assertRaises(AssertionError):
            df = make_zn_scores(dfd, 'XR', cids, sequential=False, neutral='std', thresh=0.5,
                                pan_weight=1.0, postfix='ZN')  # test catching non-valid thresh value

        with self.assertRaises(AssertionError):
            df = make_zn_scores(dfd, 'XR', cids, sequential=False, pan_weight=1.2)  # test catching panel weight

        # Testing on Panel = 1.0 (default value)
        df_panel = make_zn_scores(dfd, 'CRY', cids, sequential=True, min_obs=0, neutral='mean',
                                   thresh=None, postfix='ZN')
        df_panel = df_panel.pivot(index='real_date', columns='cid', values='value')
        
        ar_neutral = pan_neutral(dfw, 'mean', True)
        dfx = dfw.sub(ar_neutral, axis='rows')
        ar_sds = np.array([dfx.iloc[0:(i + 1), :].stack().abs().mean() for i in range(dfx.shape[0])])
        dfw_zns_pan = dfx.div(ar_sds, axis='rows')
        dfw_zns_pan = dfw_zns_pan.dropna(axis = 0, how='all')

        zn_scores = df_panel.to_numpy()
        arr_zns_pan = dfw_zns_pan.to_numpy()
        dif = zn_scores - arr_zns_pan
        dif = np.nan_to_num(dif, nan = 0.0)
        
        self.assertTrue(np.all(dif < epsilon))

        # Test weighting function.
        panel_df = make_zn_scores(dfd, 'CRY', cids, start="2010-01-04", sequential=True, min_obs=252,
                                  neutral='mean', thresh=None, pan_weight=1.0, postfix='ZN')
        df_cross = make_zn_scores(dfd, 'CRY', cids, start="2010-01-04", sequential=True, min_obs=252,
                                  neutral='mean', thresh=None, pan_weight=0.0, postfix='ZN')

        df_average = make_zn_scores(dfd, 'CRY', cids, start="2010-01-04", sequential=True, min_obs=252,
                                    neutral='mean', thresh=None, pan_weight=0.5, postfix='ZN')

        panel_df = panel_df.pivot(index='real_date', columns='cid', values='value')
        df_cross = df_cross.pivot(index='real_date', columns='cid', values='value')
        df_average = df_average.pivot(index='real_date', columns='cid', values='value')
        
        panel_df = panel_df.drop(panel_df.index[[0]]) # Drop the first row in the panel data to adjust for the first row in the cross-sectional dataframe being removed.       
        df_check = (panel_df + df_cross) / 2
        check_arr = df_check.to_numpy()
        average_arr = df_average.to_numpy()

        dif = check_arr - average_arr
        dif = np.nan_to_num(dif, nan = 0.0)
        self.assertTrue(np.all(dif < epsilon))

        threshold = 2.35
        df_thresh = make_zn_scores(dfd, 'CRY', cids, start="2010-01-01", sequential=True, min_obs=252,
                                    neutral='mean', thresh=threshold, pan_weight=0.65, postfix='ZN')

        df_thresh = df_thresh.pivot(index='real_date', columns='cid', values='value')
        thresh_arr = df_thresh.to_numpy()
        values = thresh_arr.ravel() # Compress multidimensional array into a one-dimensional array.
        
        check = np.where(values > threshold)[0] # Unpack the Array from the tuple.

        self.assertTrue(check.size == 0)
        
if __name__ == '__main__':

    unittest.main()
