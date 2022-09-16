
import unittest
import pandas as pd
import numpy as np
from tests.simulate import make_qdf
from macrosynergy.panel.view_correlations import correl_matrix, lag_series


class TestAll(unittest.TestCase):

    def dataframe_construction(self):

        self.__dict__['cids'] = ['AUD', 'CAD', 'GBP']
        self.__dict__['xcats'] = ['XR', 'CRY', 'GROWTH', 'INFL']

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
        df_xcats.loc['GROWTH', :] = ['2010-01-01', '2020-10-30', 1, 2, 0.9, 0.5]
        df_xcats.loc['INFL', :] = ['2010-01-01', '2020-10-30', 0, 2, 0.9, 0.5]

        # Standard df for tests.
        dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)
        self.__dict__['dfd'] = dfd

    def test_lag_series(self):
        """
        Test the method used to lag the categories included
        """
        self.dataframe_construction()

        df_w = self.dfd.pivot(index=("cid", "real_date"), columns="xcat",
                              values="value")

        # Confirm the application of the lag to the respective category is correct.
        # Compare against the original DataFrame.

        # Lag inflation by a range of possible options.
        lag_dict = {"INFL": [0, 2, 5]}
        # Returns a multi-index DataFrame. Therefore, ensure the lag has been applied
        # correctly on each individual cross-section.
        df_w, xcat_tracker = lag_series(df_w, lag_dict, xcats=self.xcats)

        # Firstly, confirm the DataFrame includes the expected columns: incumbent
        # categories & additional categories that have had the lag postfix appended.
        test_columns = list(df_w.columns)
        test_columns_set = set(test_columns)
        # The inflation category will be removed from the wide DataFrame as it is being
        # replaced by a lagged version.
        xcats_copy = self.xcats.copy()
        xcats_copy.remove("INFL")
        self.assertTrue(set(xcats_copy).issubset(test_columns_set))

        # Remove the incumbent categories and confirm that the residual categories in
        # the DataFrame are the respective lagged inflation series: ['INFL_L0',
        # 'INFL_L2', 'INFL_L5'].

        update_test_columns = [xcat for xcat in test_columns
                               if xcat not in xcats_copy]

        update_test_columns = sorted(update_test_columns)
        self.assertTrue(update_test_columns == ["INFL_L0", "INFL_L2", "INFL_L5"])

        # Confirm the lag mechanism works correctly. The DataFrame will be a multi-index
        # DataFrame, and subsequently it is important to confirm that the lag works
        # correctly on an individual ticker level.
        df_w_lag = df_w[["INFL_L0", "INFL_L2", "INFL_L5"]]

        # The benchmark value will be "INFL_L0" where a lag has not been applied. Test on
        # two cross-sections: ['AUD', 'GBP'].
        df_w_lag = df_w_lag.loc[['AUD', 'GBP'], :]

        # Arbitrary date to test the logic.
        fixed_date = '2013-03-11'
        condition = df_w_lag.index.get_level_values('real_date') == fixed_date
        period_t = df_w_lag[condition]["INFL_L0"]
        period_t_aud = float(period_t.loc["AUD"])

        lagged_date_2 = pd.Timestamp(fixed_date) + pd.DateOffset(2)
        condition_2 = df_w_lag.index.get_level_values('real_date') == lagged_date_2
        period_t_2 = df_w_lag[condition_2]["INFL_L2"]
        period_t_2_aud = float(period_t_2.loc["AUD"])

        # Confirm that the value at '2013-03-11' has been lagged correctly in INFL_L2 on
        # AUD: the same value should be held at '2013-03-13'.
        self.assertEqual(period_t_aud, period_t_2_aud)

        # Apply the same logic above but for INFL_L2.
        dates_index = list(df_w_lag.loc['AUD', :].index)
        dates_index = list(map(lambda d: str(d).split(' ')[0], dates_index))

        fixed_date_index = dates_index.index(fixed_date)
        lagged_date_5_index = fixed_date_index + 5

        lagged_date_5 = dates_index[lagged_date_5_index]
        condition_5 = df_w_lag.index.get_level_values('real_date') == lagged_date_5

        period_t_5 = df_w_lag[condition_5]["INFL_L5"]
        period_t_5_aud = float(period_t_5.loc["AUD"])
        self.assertEqual(period_t_aud, period_t_5_aud)

        period_t_gbp = float(period_t.loc["GBP"])
        period_t_2_gbp = float(period_t_2.loc["GBP"])

        # Confirm that the value at '2013-03-11' has been lagged correctly in INFL_L2 on
        # GBP: the same value should be held at '2013-03-13'.
        self.assertEqual(period_t_gbp, period_t_2_gbp)

        # INFL_L5 on GBP.
        period_t_5_gbp = float(period_t_5.loc["GBP"])
        self.assertEqual(period_t_gbp, period_t_5_gbp)

        # Lag inflation & Growth by a range of possible options.
        lag_dict = {"INFL": [0, 5], "GROWTH": [3]}
        df_w = self.dfd.pivot(index=("cid", "real_date"), columns="xcat",
                              values="value")

        df_w, xcat_tracker = lag_series(df_w, lag_dict, xcats=self.xcats)
        test_columns = list(df_w.columns)

        lagged_columns = [xcat for xcat in test_columns if xcat not in self.xcats]
        self.assertTrue(sorted(lagged_columns) ==
                        ["GROWTH_L3", "INFL_L0", "INFL_L5"])

    def test_correl_matrix(self):

        self.dataframe_construction()

        # Mainly test assertions given the function is used for visualisation. The
        # function can easily be tested through the graph returned.

        lag_dict = {'INFL': [0, 1, 2, 5]}
        with self.assertRaises(AssertionError):
            # Test the frequency options: either ['W', 'M', 'Q'].
            correl_matrix(self.dfd, xcats=['XR', 'CRY'], cids=self.cids,
                          freq='BW', lags=lag_dict, max_color=0.1)

        with self.assertRaises(AssertionError):
            # Test the max_color value. Expects a floating point value.
            correl_matrix(self.dfd, xcats=['XR', 'CRY'], cids=self.cids,
                          lags=lag_dict, max_color=1)

        with self.assertRaises(AssertionError):
            # Test the received lag data structure. Dictionary expected.
            lag_list = [0, 60]
            correl_matrix(self.dfd, xcats=['XR', 'CRY'], cids=self.cids,
                          lags=lag_list, max_color=1)

        with self.assertRaises(AssertionError):
            # Test that the lagged categories are present in the received DataFrame.
            # The category, GROWTH, is not in the received categories.
            lag_dict = {'GROWTH': [0, 1, 2, 5]}
            correl_matrix(self.dfd, xcats=['XR', 'CRY'], cids=self.cids,
                          lags=lag_dict, max_color=1)


if __name__ == '__main__':
    unittest.main()