
import unittest
import numpy as np
import pandas as pd
import warnings
from tests.simulate import make_qdf
from macrosynergy.panel.make_zn_scores import *
from random import randint
from itertools import groupby


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

        daily_dates = pd.date_range(start='2010-01-01', end='2020-10-30', freq='B')
        self.__dict__['dates_iter'] = daily_dates
        self.__dict__['func_dict'] = {'mean': np.mean, 'median': np.median}

    def in_sampling(self, dfw, neutral, min_obs):
        """
        Used to test the application of pandas in-built back-fill mechanism.
        """
        self.dataframe_construction()

        # Convert to a one-dimensional DataFrame to facilitate pd.apply() method
        # to calculate in-sampling period. The pd.stack() feature removes the
        # unrealised cross-sections.
        iis_period = pd.DataFrame(dfw.iloc[0:min_obs].stack().to_numpy())
        iis_val = iis_period.apply(self.func_dict[neutral])

        return round(float(iis_val), 4)

    def test_pan_neutral(self):

        self.dataframe_construction()

        df_neutral = pan_neutral(df=self.dfw, dates_iter=self.dates_iter, neutral='mean',
                                 sequential=True)
        self.assertIsInstance(df_neutral, pd.DataFrame)
        # Test length of neutral array.
        self.assertTrue(self.dfw.shape[0] == df_neutral.shape[0])

        df_neutral = pan_neutral(self.dfw, dates_iter=self.dates_iter, neutral='mean',
                                 sequential=False)
        # Check first value equal to panel mean.
        self.assertEqual(float(df_neutral.iloc[0]), self.dfw.stack().mean())
        # Check also last value equal to panel mean.
        last_val = float(df_neutral.iloc[self.dfw.shape[0] - 1])
        self.assertEqual(last_val, self.dfw.stack().mean())

        df_neutral = pan_neutral(self.dfw, dates_iter=self.dates_iter, neutral='mean',
                                 sequential=True)
        val = round(float(df_neutral.iloc[999]), 4)
        benchmark = self.dfw.iloc[0:1000, :].stack().mean()
        self.assertEqual(val, round(benchmark, 4))

        df_neutral = pan_neutral(self.dfw, dates_iter=self.dates_iter, neutral='median',
                                 sequential=False)
        # Check first value equal to panel median.
        self.assertEqual(float(df_neutral.iloc[0]), self.dfw.stack().median())
        # Check last value equal to panel median.
        last_val = float(df_neutral.iloc[self.dfw.shape[0] - 1])
        self.assertEqual(last_val,
                         self.dfw.stack().median())

        df_neutral = pan_neutral(self.dfw, dates_iter=self.dates_iter, neutral='median',
                                 sequential=True, min_obs=261, iis=False)
        val = float(df_neutral.iloc[999])
        median_benchmark = self.dfw.iloc[0:1000, :].stack().median()
        self.assertEqual(val, median_benchmark)

        # Check the application of the in-sampling procedure.
        # The first testcase set the in-sample to False and the expected values for the
        # first minimum number of observation days should be equal to np.nan.
        df_neutral = pan_neutral(self.dfw, dates_iter=self.dates_iter, neutral='mean',
                                 sequential=True, min_obs=261, iis=False)
        self.assertTrue(all(np.nan_to_num(df_neutral.iloc[:261]) == 0.0))

        # Check the inclusion of the in-sampling data being included in the returned
        # Array. The first minimum number observations, for the neutral level, will all
        # be the same value.
        df_neutral = pan_neutral(self.dfw, dates_iter=self.dates_iter, neutral='mean',
                                 sequential=True, min_obs=261, iis=True)
        self.assertTrue(all(df_neutral.iloc[:261] == df_neutral.iloc[0]))

        test_val = self.in_sampling(dfw=self.dfw, neutral='mean', min_obs=261)
        test_data = df_neutral.iloc[:261].to_numpy().reshape(261)

        bm_vals = [round(v, 4) for v in test_data]
        for v in bm_vals:
            self.assertTrue(abs(v - test_val) < 0.1)

        # Check the above for the application of 'median' as the neutral level.
        # Unable to check for equality on np.nan values.
        df_neutral = pan_neutral(self.dfw, dates_iter=self.dates_iter, neutral='median',
                                 sequential=True, min_obs=261, iis=True)
        self.assertTrue(all(df_neutral.iloc[:261] == df_neutral.iloc[0]))

    @staticmethod
    def dates_iterator(df, est_freq):
        """
        Method used to produce the dates data structure.
        """

        s_date = min(df['real_date'])
        e_date = max(df['real_date'])
        dates_iter = pd.date_range(start=s_date, end=e_date, freq=est_freq)
        return dates_iter

    def test_downsampling(self):
        """
        Often there is little value added from computing the neutral level and standard
        deviation on a daily basis. The calculations are computationally intensive and
        the change over a daily period will, in most instances, be inconsequential. For
        the neutral level or standard deviation to have a significant change, especially
        as the days pass, the return series must have experienced a sustained period of
        inverted returns. A reversal in the underlying trend will normally manifest
        after a number of days have elapsed.
        """
        self.dataframe_construction()

        df = self.dfd
        df_copy = df.copy()

        df['real_date'] = pd.to_datetime(df['real_date'], errors='coerce')
        df['year'] = df['real_date'].dt.year
        # Test on monthly down-sampling to ensure the expanding window is still being
        # applied correctly but on a monthly basis. Each statistic, computed on the lower
        # frequency, will use all of the preceding days data to capture the underlying
        # trend.

        df['month'] = df['real_date'].dt.month
        dfw_multidx = df.pivot(index=['year', 'month', 'real_date'], columns='cid',
                               values='value')

        # Test on the 'mean' neutral level.
        test = []
        aggregate = np.empty(0)
        for date, new_df in dfw_multidx.groupby(level=[0, 1]):
            new_arr = new_df.stack().to_numpy()
            aggregate = np.concatenate([aggregate, new_arr])
            test.append(np.mean(aggregate))

        dfw = df_copy.pivot(index='real_date', columns='cid', values='value')
        dates_iter = self.dates_iterator(df_copy, est_freq='BM')
        # Test against the existing solution.
        # The below method will return a one-dimensional DataFrame hosting the neutral
        # values produced from the expanding window. The DataFrame will be daily values
        # and, if down-sampling has been applied, the intermediary dates between
        # re-estimation dates will be populated by forward filling technique.
        # Therefore, the number of unique neutral values will correspond to the number of
        # re-estimation dates.
        df_mean = pan_neutral(df=dfw, dates_iter=dates_iter, neutral='mean',
                              sequential=True, min_obs=261, iis=True)
        bm_values = df_mean['value'].to_numpy()
        # Avoid using a set which orders the data.
        bm_values = [k for k, g in groupby(bm_values)]

        # Assert there are the same number of re-estimation dates.
        self.assertTrue(len(bm_values) == len(test))

        # Confirm the expanding neutral calculations are correct.
        for i, bm_val in enumerate(bm_values):
            condition = abs(bm_val - test[i]) < 0.001
            self.assertTrue(condition)

        # The above logic also concurrently tests computing the standard deviation on a
        # down-sampled series. Uses the same method.

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

        df_neutral = cross_neutral(df=self.dfw, neutral='mean', sequential=False)
        self.assertIsInstance(df_neutral, pd.DataFrame)

        df_shape = self.dfw.shape
        self.assertEqual(df_shape, df_neutral.shape)

        epsilon = 0.0001

        df_mean = cross_neutral(self.dfw, neutral='mean', sequential=False)
        df_median = cross_neutral(self.dfw, neutral='median', sequential=False)

        # Arbitrarily chosen index to test the logic.
        index = 411
        for i, cross in enumerate(self.cids):

            column = self.dfw[[cross]].to_numpy()
            column = np.squeeze(column, axis=1)
            column = self.handle_nan(column)

            mean_col = self.handle_nan(df_mean.iloc[:, i])
            
            mean = np.sum(column) / len(column)
            dif = np.abs(mean_col - mean)
            # Test if function mean is correct.
            self.assertTrue(np.nan_to_num(dif[index]) < epsilon)

            median = np.median(column)
            median_col = self.handle_nan(df_median.iloc[:, i])
            median_col = list(set(median_col))
            self.assertTrue(len(median_col) == 1)

            value = median_col[0]
            dif = np.abs(median - value)
            # Test if function median is correct.
            # self.assertTrue(dif < epsilon)

        min_obs = 261
        df_mean = cross_neutral(self.dfw, neutral='mean', sequential=True,
                                min_obs=min_obs, iis=False)
        df_median = cross_neutral(self.dfw, neutral='median', sequential=True,
                                  min_obs=min_obs, iis=True)
        cross_sections = list(self.dfw.columns)

        for i, cid in enumerate(cross_sections):
            column_mean = df_mean.iloc[:, i]

            series_mean = pd.Series(data=column_mean)
            mean_index = self.valid_index(series_mean)
            column = self.dfw.iloc[:, i]

            date_index = self.valid_index(column)
            self.assertTrue(mean_index == (date_index + min_obs))

            column = self.dfw[[cid]]
            cum_mean = column.expanding(min_periods=(min_obs + 1)).mean()
            cum_mean = self.handle_nan(cum_mean[cid].to_numpy())

            dif = self.handle_nan(df_mean.iloc[:, i]) - cum_mean
            # Check correct cumulative means.
            self.assertTrue(np.nan_to_num(dif[index]) < epsilon)

            iis_period = df_median.iloc[date_index:(date_index + min_obs), i]
            first_val_iis = iis_period[0]
            self.assertTrue(np.all(iis_period == first_val_iis))

    def test_cross_down_sampling(self):
        """
        Neutral level is computed on a cross-sectional basis.
        """
        self.dataframe_construction()

        # Test the method neutral_calc() which is the associated helper function used for
        # individual cross-sections.

        df = self.dfd
        dfw = self.dfw
        # Isolate an individual cross-section's return series.
        cross_series = dfw['AUD']
        date_index = self.valid_index(column=cross_series)

        # Test on quarterly data.
        dates_iter = self.dates_iterator(df, est_freq='BQ')
        neutral_df = neutral_calc(column=cross_series, dates_iter=dates_iter, iis=True,
                                  neutral='mean', date_index=date_index, min_obs=261,
                                  cid='AUD')
        # Choose a random re-estimation date and confirm the corresponding re-estimated
        # value is equivalent to in-sampling.
        random_index = len(dates_iter) // 2
        random_date = dates_iter[random_index]
        test_series = cross_series.loc[:random_date]
        test_value = np.mean(test_series.to_numpy())

        benchmark_value = float(neutral_df.loc[random_date])
        self.assertTrue(np.abs(test_value - benchmark_value) < 0.001)

        # Confirm the dates, over the next quarter, are the same as the value referenced
        # above.
        next_index = random_index + 1
        next_date_quarter = dates_iter[next_index]
        benchmark_quarter = neutral_df.loc[random_date:next_date_quarter].to_numpy()
        benchmark_quarter = benchmark_quarter.reshape(len(benchmark_quarter))

        # Exclude the next re-estimation date where the neutral level changes.
        for bm_elem in benchmark_quarter[:-1]:
            self.assertTrue(np.abs(test_value - bm_elem) < 0.001)

    def test_zn_scores(self):

        self.dataframe_construction()

        with self.assertRaises(AssertionError):
            # Test catching neutral value error.
            df = make_zn_scores(self.dfd, 'XR', self.cids, sequential=False,
                                neutral='std', thresh=1.5, postfix='ZN')
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
        df_neutral = pan_neutral(self.dfw, dates_iter=self.dates_iter, neutral='mean',
                                 sequential=True, min_obs=0, iis=False)
        dfx = self.dfw.sub(df_neutral['value'], axis='rows')
        
        ar_sds = np.array([dfx.iloc[0:(i + 1), :].stack().abs().mean()
                           for i in range(dfx.shape[0])])
        dfw_zns_pan = dfx.div(ar_sds, axis='rows')
        dfw_zns_pan = dfw_zns_pan.dropna(axis = 0, how='all')

        # Check the zn_scores, across a panel, on a specific date. Discount the
        # internal randomness.
        no_rows = dfw_zns_pan.shape[0]
        index = randint(0, no_rows)
        index = int(no_rows / 2)

        zn_scores = df_panel.to_numpy()
        arr_zns_pan = dfw_zns_pan.to_numpy()
        dif = zn_scores[index] - arr_zns_pan[index]

        epsilon = 0.000001
        self.assertTrue(np.all(np.nan_to_num(dif) < epsilon))

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
        index = int(df_average.shape[0] / 2)

        # Drop the first row in the panel data.
        panel_df = panel_df.drop(panel_df.index[[0]])
        df_check = (panel_df + df_cross) / 2
        check_arr = df_check.to_numpy()
        average_arr = df_average.to_numpy()

        # Again, validate on a randomly chosen index.
        index = 121
        dif = check_arr[index] - average_arr[index]
        dif = np.nan_to_num(dif)

        # self.assertTrue(np.sum(dif) < epsilon)

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