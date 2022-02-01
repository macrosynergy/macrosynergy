
import unittest
import numpy as np
import pandas as pd
import warnings
from tests.simulate import make_qdf
from macrosynergy.panel.make_zn_scores import *
from random import randint


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
        # Check last value equal to panel median.
        self.assertEqual(ar_neutral[self.dfw.shape[0]-1], self.dfw.stack().median())

        ar_neutral = pan_neutral(self.dfw, neutral='median', sequential=True)
        self.assertEqual(ar_neutral[999], self.dfw.iloc[0:1000, :].stack().median())

        # Check the application of the in-sampling procedure.
        # The first testcase set the in-sample to False and the expected values for the
        # first minimum number of observation days should be equal to np.nan.
        ar_neutral = pan_neutral(self.dfw, neutral='mean', sequential=True,
                                 min_obs=261, iis=False)
        self.assertTrue(all(np.nan_to_num(ar_neutral[:261]) == 0.0))

        # Check the inclusion of the in-sampling data being included in the returned
        # Array. The first minimum number observations, for the neutral level, will all
        # be the same value.
        ar_neutral = pan_neutral(self.dfw, neutral='mean', sequential=True,
                                 min_obs=261, iis=True)
        self.assertTrue(all(ar_neutral[:261] == ar_neutral[0]))

        # Check the above for the application of 'median' as the neutral level.
        ar_neutral = pan_neutral(self.dfw, neutral='median', sequential=True,
                                 min_obs=261, iis=True)
        self.assertTrue(all(ar_neutral[:261] == ar_neutral[0]))

    @staticmethod
    def valid_index(column):

        index = column.index
        date = column.first_valid_index()
        date_index = next(iter(np.where(index == date)[0]))

        return date_index

    @staticmethod
    def handle_nan(arr):
        arr = np.nan_to_num(arr)
        arr = arr[arr != 0.0]

        return arr

    def test_cross_neutral(self):

        self.dataframe_construction()

        arr_neutral = cross_neutral(self.dfw, 'mean', False)
        self.assertIsInstance(arr_neutral, np.ndarray)

        df_shape = self.dfw.shape
        self.assertEqual(df_shape, arr_neutral.shape)

        epsilon = 0.0001

        ar_mean = cross_neutral(self.dfw, neutral='mean', sequential=False)
        ar_median = cross_neutral(self.dfw, neutral='median', sequential=False)

        # Arbitrarily chosen index to test the logic.
        index = 411
        for i, cross in enumerate(self.cids):

            column = self.dfw[[cross]].to_numpy()
            column = np.squeeze(column, axis=1)
            column = self.handle_nan(column)

            mean_col = self.handle_nan(ar_mean[:, i])
            
            mean = np.sum(column) / len(column)
            dif = np.abs(mean_col - mean)
            # Test if function mean is correct.
            self.assertTrue(np.nan_to_num(dif[index]) < epsilon)

            median = np.median(column)
            median_col = self.handle_nan(ar_median[:, i])
            median_col = list(set(median_col))
            self.assertTrue(len(median_col) == 1)

            value = median_col[0]
            dif = np.abs(median - value)
            # Test if function median is correct.
            # self.assertTrue(dif < epsilon)

        min_obs = 261
        ar_mean = cross_neutral(self.dfw, neutral='mean', sequential=True,
                                min_obs=min_obs, iis=False)
        ar_median = cross_neutral(self.dfw, neutral='median', sequential=True,
                                  min_obs=min_obs, iis=True)
        cross_sections = list(self.dfw.columns)

        for i, cid in enumerate(cross_sections):
            column_mean = ar_mean[:, i]

            series_mean = pd.Series(data=column_mean)
            mean_index = self.valid_index(series_mean)
            column = self.dfw.iloc[:, i]

            date_index = self.valid_index(column)
            self.assertTrue(mean_index == (date_index + min_obs))

            column = self.dfw[[cid]]
            cum_mean = column.expanding(min_periods=(min_obs + 1)).mean()
            cum_mean = self.handle_nan(cum_mean[cid].to_numpy())

            dif = self.handle_nan(ar_mean[:, i]) - cum_mean
            # Check correct cumulative means.
            self.assertTrue(np.nan_to_num(dif[index]) < epsilon)

            iis_period = ar_median[date_index:(date_index + min_obs), i]
            first_val_iis = iis_period[0]
            self.assertTrue(np.all(iis_period == first_val_iis))

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
                                pan_weight=0.2, iis=0)

        with self.assertRaises(AssertionError):
            # Test the minimum observation parameter (non-negative Integer value).
            df = make_zn_scores(self.dfd, 'XR', self.cids, sequential=True,
                                pan_weight=0.3, min_obs=-1, iis=0)

        # Testing on Panel = 1.0 (default value)
        df_panel = make_zn_scores(self.dfd, 'CRY', self.cids, sequential=True,
                                  min_obs=0, iis=False, neutral='mean',
                                  thresh=None, postfix='ZN')

        df_panel = df_panel.pivot(index='real_date', columns='cid', values='value')
        ar_neutral = pan_neutral(self.dfw, neutral='mean', sequential=True,
                                 min_obs=0, iis=False)
        dfx = self.dfw.sub(ar_neutral, axis='rows')
        
        ar_sds = np.array([dfx.iloc[0:(i + 1), :].stack().abs().mean()
                           for i in range(dfx.shape[0])])
        dfw_zns_pan = dfx.div(ar_sds, axis='rows')
        dfw_zns_pan = dfw_zns_pan.dropna(axis = 0, how='all')

        # Check the zn_scores, across a panel, on a specific date. Discount the
        # internal randomness.
        no_rows = dfw_zns_pan.shape[0]
        index = randint(0, no_rows)

        zn_scores = df_panel.to_numpy()
        arr_zns_pan = dfw_zns_pan.to_numpy()
        dif = zn_scores[index] - arr_zns_pan[index]

        epsilon = 0.000001
        # self.assertTrue(np.all(np.nan_to_num(dif) < epsilon))

        # Test weighting function.
        min_obs = 252
        panel_df = make_zn_scores(self.dfd, 'CRY', self.cids, start="2010-01-04",
                                  sequential=False, min_obs=0, neutral='mean',
                                  iis=False, thresh=None, pan_weight=0.75, postfix='ZN')
        df_cross = make_zn_scores(self.dfd, 'CRY', self.cids, start="2010-01-04",
                                  sequential=False, min_obs=0, neutral='mean',
                                  iis=False, thresh=None, pan_weight=0.25, postfix='ZN')

        df_average = make_zn_scores(self.dfd, 'CRY', self.cids, start="2010-01-04",
                                    sequential=False, min_obs=0, iis=False,
                                    neutral='mean', thresh=None, pan_weight=0.5,
                                    postfix='ZN')

        panel_df = panel_df.pivot(index='real_date', columns='cid', values='value')
        df_cross = df_cross.pivot(index='real_date', columns='cid', values='value')
        df_average = df_average.pivot(index='real_date', columns='cid', values='value')
        index = randint(0, df_average.shape[0])

        # Drop the first row in the panel data.
        panel_df = panel_df.drop(panel_df.index[[0]])
        df_check = (panel_df + df_cross) / 2
        check_arr = df_check.to_numpy()
        average_arr = df_average.to_numpy()

        # Again, validate on a randomly chosen index.
        index = 121
        dif = check_arr[index] - average_arr[index]
        dif = np.nan_to_num(dif)

        # self.assertTrue(np.all(dif < epsilon))

        # Test the usage of the threshold parameter.
        threshold = 2.35
        df_thresh = make_zn_scores(self.dfd, 'CRY', self.cids, start="2010-01-01",
                                   sequential=True, min_obs=252, neutral='mean',
                                   thresh=threshold, pan_weight=0.65, postfix='ZN')

        df_thresh = df_thresh.pivot(index='real_date', columns='cid', values='value')
        thresh_arr = df_thresh.to_numpy()
        # Compress multidimensional array into a one-dimensional array.
        values = thresh_arr.ravel()
        values = values.astype(dtype=np.float64)

        check = sum(values[~np.isnan(values)] > threshold)

        self.assertTrue(check == 0)

        
if __name__ == '__main__':

    unittest.main()