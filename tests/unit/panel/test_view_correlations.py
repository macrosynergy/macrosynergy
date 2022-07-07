
import unittest
import pandas as pd
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

        df_w = self.dfd.pivot(index=('cid', 'real_date'), columns='xcat',
                              values='value')

        # Confirm the application of the lag to the respective category is correct.
        # Compare against the original DataFrame.

        # Lag inflation by a range of possible options.
        lag_dict = {'INFL': [0, 2, 5]}
        # Returns a multi-index DataFrame. Therefore, ensure the lag has been applied
        # correctly on each individual cross-section.
        df_w, xcat_tracker = lag_series(df_w, lag_dict, xcats=self.xcats)

        # Firstly, confirm the DataFrame includes the expected columns: incumbent
        # categories & additional categories that have had the lag postfix appended.
        test_columns = df_w.columns
        test_columns = set(test_columns)
        # The inflation category will be removed from the wide DataFrame as it is being
        # replaced by a lagged version.
        xcats_copy = self.xcats
        xcats_copy.remove('INFL')
        self.assertTrue(set(xcats_copy).issubset(test_columns))

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