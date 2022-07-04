import unittest
import random
import numpy as np
import pandas as pd
from tests.simulate import make_qdf
from macrosynergy.management.update_df import update_df, update_tickers, \
    update_categories
from macrosynergy.panel.make_relative_value import make_relative_value


class TestAll(unittest.TestCase):

    def dataframe_constructor(self):

        self.__dict__['cids'] = ['AUD', 'CAD', 'GBP', 'NZD']
        self.__dict__['xcats'] = ['GROWTH', 'INFL', 'XR']

        df_cids = pd.DataFrame(index=self.cids,
                               columns=['earliest', 'latest', 'mean_add', 'sd_mult'])
        df_cids.loc['AUD', :] = ['2010-01-01', '2020-12-31', 0.5, 2]
        df_cids.loc['CAD', :] = ['2011-01-01', '2020-11-30', 0, 1]
        df_cids.loc['GBP', :] = ['2012-01-01', '2020-11-30', -0.2, 0.5]
        df_cids.loc['NZD'] = ['2002-01-01', '2020-09-30', -0.1, 2]

        df_xcats = pd.DataFrame(index=self.xcats,
                                columns=['earliest', 'latest', 'mean_add', 'sd_mult',
                                         'ar_coef', 'back_coef'])
        df_xcats.loc['GROWTH', :] = ['2011-01-01', '2020-10-30', 1, 2, 0.9, 0.5]
        df_xcats.loc['INFL', :] = ['2010-01-01', '2020-12-31', 0, 1, 0, 0.3]
        df_xcats.loc['XR', :] = ['2000-01-01', '2020-12-31', 0.1, 1, 0, 0.3]

        random.seed(1)
        np.random.seed(0)
        self.__dict__['dfd'] = make_qdf(df_cids, df_xcats, back_ar=0.75)

        black = {'AUD': ['2000-01-01', '2003-12-31'],
                 'GBP': ['2018-01-01', '2100-01-01']}
        self.__dict__['blacklist'] = black

    def test_update_tickers(self):
        """
        Method used to test updating the DataFrame across on the ticker level.
        """

        self.dataframe_constructor()

        # Method used to update the original DataFrame on a ticker level.
        # Original DataFrame will be defined over the categories ['GROWTH', 'INFL',
        # 'XR'].
        dfd = self.dfd
        # Design a DataFrame such that two of "new" tickers are already held in the
        # aggregate DataFrame. Aim to confirm the existing tickers are replaced.
        dfd_1_rv = make_relative_value(self.dfd, xcats=['GROWTH'], cids=['AUD', 'CAD'],
                                       rel_meth='subtract', postfix='RV')
        dfd = pd.concat([dfd, dfd_1_rv])

        dfd_2_rv = make_relative_value(self.dfd, xcats=['GROWTH', 'INFL'],
                                       cids=['AUD', 'CAD'], rel_meth='divide',
                                       postfix='RV')
        dfd = update_tickers(df=dfd, df_add=dfd_2_rv)

        # Firstly, confirm that the returned DataFrame contains the new tickers:
        # AUD_INFLRV, CAD_INFLRV.
        dfd_copy = dfd.copy()
        dfd_copy['tickers'] = dfd_copy['cid'] + "_" + dfd_copy['xcat']
        agg_tickers = dfd_copy['tickers'].unique()

        new_tickers = ['AUD_INFLRV', 'CAD_INFLRV']
        for t in new_tickers:
            self.assertTrue(t in agg_tickers)

        # After confirming expected presence, test the values.
        filter_1 = ((dfd_copy['cid'] == 'AUD') & (dfd_copy['xcat'] == 'INFLRV'))
        filt_df = dfd_copy[filter_1]

        filter_2 = ((dfd_2_rv['cid'] == 'AUD') & (dfd_2_rv['xcat'] == 'INFLRV'))
        compare_df = dfd_2_rv[filter_2]

        # Series has been appended.
        self.assertTrue(np.all(filt_df['value'].to_numpy() ==
                               compare_df['value'].to_numpy()))

        # Test the values of the two series that are being replaced. Ensure the series
        # held in the aggregate DataFrame are from the secondary DataFrame. The two
        # series are ['AUD_GROWTHRV', 'CAD_GROWTHRV'].
        replace_tickers = ['AUD_GROWTHRV', 'CAD_GROWTHRV']
        for t in replace_tickers:
            self.assertTrue(t in agg_tickers)

        filter_3 = ((dfd_copy['cid'] == 'AUD') & (dfd_copy['xcat'] == 'GROWTHRV'))
        filt_df = dfd_copy[filter_3]

        filter_4 = ((dfd_2_rv['cid'] == 'AUD') & (dfd_2_rv['xcat'] == 'GROWTHRV'))
        compare_df = dfd_2_rv[filter_4]

        # Series has been replaced correctly.
        self.assertTrue(np.all(filt_df['value'].to_numpy() ==
                               compare_df['value'].to_numpy()))

    def test_update_categories(self):
        """
        Method used to test updating the DataFrame across a whole panel.
        """

        self.dataframe_constructor()
        dfd = self.dfd
        dfd_1_rv = make_relative_value(self.dfd, xcats=['GROWTH', 'INFL'], cids=None,
                                       blacklist=None, rel_meth='subtract',
                                       rel_xcats=None,
                                       postfix='RV')
        dfd_1_rv_growth = ((dfd_1_rv['xcat'] == 'GROWTHRV') & (dfd_1_rv['cid'] == 'AUD'))
        dfd_1_rv_growth = dfd_1_rv[dfd_1_rv_growth]

        # An arbitrarily chosen date used to ensure the logic is correct and the
        # aggregate DataFrame hosts the correct values.
        fixed_date = dfd_1_rv_growth['real_date'][dfd_1_rv_growth.shape[0] // 2]

        # Both categories generated from the make_relative_value() function will not be
        # present in the aggregated dataframe. Therefore, confirm both categories are
        # appended to the existing dataframe.
        dfd_add = update_categories(df=dfd, df_add=dfd_1_rv)
        # First, confirm the expected dimensions.
        self.assertTrue(dfd_add.shape[0] == (dfd.shape[0] + dfd_1_rv.shape[0]))

        # Confirm the appended dataframe's categories are a subset of the combined
        # dataframe.
        new_categories = set(dfd_1_rv['xcat'])
        self.assertTrue(new_categories.issubset(set(dfd_1_rv['xcat'])))

        dfd_1_rv_growth = dfd_1_rv[dfd_1_rv['xcat'] == 'GROWTHRV']
        dfd_1_rv_growth_aud = dfd_1_rv_growth[dfd_1_rv_growth['cid'] == 'AUD']

        value = dfd_1_rv_growth_aud[dfd_1_rv_growth_aud['real_date'] ==
                                    fixed_date]['value']

        # Confirm the values do not change during the aggregation mechanism.
        dfd_add_growth = dfd_add[dfd_add['xcat'] == 'GROWTHRV']
        dfd_add_aud = dfd_add_growth[dfd_add_growth['cid'] == 'AUD']

        test = dfd_add_aud[dfd_add_aud['real_date'] == fixed_date]['value']
        self.assertTrue(float(test) == float(value))

        # Test the replacement mechanism: categories are already in the dataframe but are
        # to be replaced by new values.
        dfd_2_rv_divide = make_relative_value(dfd, xcats=['GROWTH', 'INFL'],
                                              cids=None,
                                              blacklist=None, rel_meth='divide',
                                              rel_xcats=None,
                                              postfix='RV')

        dfd_add_2 = update_categories(df=dfd_add, df_add=dfd_2_rv_divide)
        self.assertTrue(list(set(dfd_add_2['xcat'])) == list(set(dfd_add['xcat'])))

        # Confirm the value does not equal the previous value which will be held in the
        # original dataframe, "dfd_add".
        dfd_add_growth_2 = dfd_add_2[dfd_add_2['xcat'] == 'GROWTHRV']
        dfd_add_aud_2 = dfd_add_growth_2[dfd_add_growth_2['cid'] == 'AUD']

        test_2 = dfd_add_aud_2[dfd_add_aud_2['real_date'] == fixed_date]['value']
        self.assertTrue(float(test_2) != float(value))

        # Hence, confirm the new value stored in "dfd_add_2" is sourced from the latest
        # dataframe "dfd_1_rv_divide".
        dfd_2_rv_growth = dfd_2_rv_divide[dfd_2_rv_divide['xcat'] == 'GROWTHRV']
        dfd_2_rv_growth_aud = dfd_2_rv_growth[dfd_2_rv_growth['cid'] == 'AUD']

        value_2 = dfd_2_rv_growth_aud[dfd_2_rv_growth_aud['real_date'] ==
                                      fixed_date]['value']
        self.assertTrue(float(test_2) == float(value_2))

        # The final test is to confirm that the method is able to replace categories
        # that already exist in the dataframe, with the latest computed values, whilst
        # adding a new category if it is not already present in the aggregated dataframe.
        dfd_copy = self.dfd.copy()
        dfd_3 = make_relative_value(dfd_copy, xcats=['XR', 'GROWTH', 'INFL'], cids=None,
                                    blacklist=None, basket=None,
                                    rel_meth='subtract', rel_xcats=None, postfix="RV")

        # DataFrame used for the initial aggregation. The added categories will become
        # legacy data and removed by the second call of update_df().
        dfd_copy_2 = self.dfd.copy()
        dfd_4 = make_relative_value(dfd_copy_2, xcats=['GROWTH', 'INFL'], cids=None,
                                    blacklist=None, rel_meth='divide', rel_xcats=None,
                                    postfix='RV')
        # Aggregate the two DataFrames.
        dfd_add_1 = update_categories(df=dfd_copy, df_add=dfd_4)

        dfd_add_2 = update_categories(df=dfd_add_1, df_add=dfd_3)
        categories = set(dfd_add_2['xcat'])
        # Preliminary confirmation that the method will replace the incumbent categories
        # whilst adding the new category, "XRRV".
        self.assertTrue(len(categories) == 6)

        # Confirm the old values are not present in the latest dataframe, "dfd_add_2".
        dfd_1_rv_growth = dfd_4[dfd_4['xcat'] == 'GROWTHRV']
        dfd_1_rv_growth_aud = dfd_1_rv_growth[dfd_1_rv_growth['cid'] == 'AUD']

        incorrect_value = dfd_1_rv_growth_aud[dfd_1_rv_growth_aud['real_date'] ==
                                              fixed_date]
        incorrect_value = incorrect_value['value']
        # Isolate the respective date on the output dataframe, "dfd_add_2", and confirm
        # the value is not equal to the legacy value.

        dfd_add_growth_2 = dfd_add_2[dfd_add_2['xcat'] == 'GROWTHRV']
        dfd_add_aud_2 = dfd_add_growth_2[dfd_add_growth_2['cid'] == 'AUD']

        test_1 = dfd_add_aud_2[dfd_add_aud_2['real_date'] == fixed_date]['value']
        self.assertTrue(float(test_1) != float(incorrect_value))

        # The latest value will be held in the dataframe, "dfd_3".
        dfd_1_rv_growth = dfd_3[dfd_3['xcat'] == 'GROWTHRV']
        dfd_1_rv_growth_aud = dfd_1_rv_growth[dfd_1_rv_growth['cid'] == 'AUD']

        value = dfd_1_rv_growth_aud[dfd_1_rv_growth_aud['real_date'] == fixed_date]
        value = value['value']
        self.assertTrue(float(test_1) == float(value))

        # Confirm the new category's values are correct, "XRRV".
        dfd_1_rv_xrrv = dfd_3[dfd_3['xcat'] == 'XRRV']
        dfd_1_rv_xrrv_aud = dfd_1_rv_xrrv[dfd_1_rv_xrrv['cid'] == 'AUD']

        value = dfd_1_rv_xrrv_aud[dfd_1_rv_xrrv_aud['real_date'] == fixed_date]
        value = value['value']

        dfd_add_xrrv_2 = dfd_add_2[dfd_add_2['xcat'] == 'XRRV']
        dfd_add_aud_2 = dfd_add_xrrv_2[dfd_add_xrrv_2['cid'] == 'AUD']
        test_2 = dfd_add_aud_2[dfd_add_aud_2['real_date'] == fixed_date]['value']

        self.assertTrue(float(test_2) == float(value))

    def test_update_df(self):

        self.dataframe_constructor()
        dfd = self.dfd
        dfd_1_rv = make_relative_value(self.dfd, xcats=['GROWTH', 'INFL'], cids=None,
                                       blacklist=None, rel_meth='subtract',
                                       rel_xcats=None,
                                       postfix='RV')

        # Test the assertion that both dataframes must be in the standardised form.
        with self.assertRaises(AssertionError):
            dfd_1_rv_growth = dfd_1_rv[dfd_1_rv['xcat'] == 'GROWTHRV']
            dfd_pivot = dfd_1_rv_growth.pivot(index="real_date", columns="cid",
                                              values="value")
            dfd_add = update_df(df=dfd, df_add=dfd_pivot)

        # Test the assertion that both dataframes must be defined over the same subset
        # of standardised columns plus possible metrics.
        with self.assertRaises(AssertionError):
            dfd_1_rv_growth = dfd_1_rv[dfd_1_rv['xcat'] == 'GROWTHRV']
            # Contrived DataFrame that includes the 'grading' metric.
            dfd_1_rv_grading = dfd_1_rv_growth.copy()
            dfd_1_rv_grading['grading'] = np.ones(dfd_1_rv_grading.shape[0])
            # Should through an error as both DataFrames do not have the same columns.
            dfd_add = update_df(df=dfd, df_add=dfd_1_rv_grading)

        # Confirm the update_df function operates on DataFrames defined over
        # multiple metrics. The sample space of options are ['value', 'grading',
        # 'mop_lag', 'eop_lag']
        # The tests are completed on a panel-level changes.

        dfd = self.dfd
        dfd_growth = dfd[dfd['xcat'] == 'GROWTH']
        dfd_growth['grading'] = np.ones(dfd_growth.shape[0])

        dfd_1_rv = make_relative_value(self.dfd, xcats=['GROWTH'],
                                       cids=['AUD', 'CAD'],
                                       rel_meth='subtract', postfix='RV')
        dfd_1_rv['grading'] = np.ones(dfd_1_rv.shape[0])

        dfd_test = pd.concat([dfd_growth, dfd_1_rv])
        dfd_test = dfd_test.reset_index(drop=True)

        dfd_update = update_df(df=dfd_growth, df_add=dfd_1_rv,
                               xcat_replace=True)
        self.assertTrue(dfd_test.shape == dfd_update.shape)

        # Order the two DataFrames and confirm the values match.
        dfd_test = dfd_test.sort_values(['xcat', 'cid', 'real_date'])
        dfd_update = dfd_update.sort_values(['xcat', 'cid', 'real_date'])
        self.assertTrue(np.all(dfd_test['value'] == dfd_update['value']))

        dfd_2_rv = make_relative_value(self.dfd, xcats=['GROWTH'],
                                       cids=['AUD', 'CAD'],
                                       rel_meth='divide', postfix='RV')
        dfd_2_rv['grading'] = np.ones(dfd_2_rv.shape[0])
        dfd_update = update_df(df=dfd_growth, df_add=dfd_2_rv,
                               xcat_replace=True)

        dfd_update_rv = dfd_update[((dfd_update['xcat'] == 'GROWTHRV') &
                                    (dfd_update['cid'] == 'AUD'))]

        fixed_date = '2011-01-07'
        test_row = dfd_update_rv[dfd_update_rv['real_date'] ==
                                 pd.Timestamp(fixed_date)]['value']
        original_data = dfd_2_rv[dfd_2_rv['real_date'] == pd.Timestamp(fixed_date)]
        original_data_aud = original_data[original_data['cid'] == 'AUD']['value']

        self.assertTrue(float(test_row) == float(original_data_aud))

        # Lastly, confirm that if the aggregate DataFrame has been defined on additional
        # metrics than the secondary DataFrame, update_df() function will instate the
        # missing columns in the secondary DataFrame with NaN values.
        # Confirm the logic works.
        dfd = self.dfd
        # Contrived columns for the purpose of testing.
        dfd['grading'] = np.ones(dfd.shape[0])
        dfd['mop_lag'] = list(range(dfd.shape[0]))

        dfd_1_rv = make_relative_value(self.dfd, xcats=['INFL'], cids=['AUD', 'CAD'],
                                       blacklist=None, rel_meth='subtract',
                                       rel_xcats=None, postfix='RV')
        dfd_update = update_df(dfd, dfd_1_rv)

        # Confirm that the aggregate DataFrame contains the two additional tickers:
        # ['AUD_INFLRV', 'CAD_INFLRV'].
        self.assertTrue(dfd.shape[0] + dfd_1_rv.shape[0] == dfd_update.shape[0])

        dfd_update_rv = (dfd_update['xcat'] == 'INFLRV') & (dfd_update['cid'] == 'AUD')
        dfd_update_df = dfd_update[dfd_update_rv]

        # First, confirm the data in the value column is correct.
        test_value = dfd_update_df[dfd_update_df['real_date'] ==
                                   pd.Timestamp(fixed_date)]['value']

        dfd_1_rv_aud = dfd_1_rv[(dfd_1_rv['cid'] == 'AUD') &
                                (dfd_1_rv['real_date'] == pd.Timestamp(fixed_date))]

        self.assertTrue(float(test_value) == float(dfd_1_rv_aud['value']))

        # Secondly, confirm that the two metrics that are not present in dfd_1_rv,
        # ['grading', 'mop_lag'], are present and exclusively contain NaN values.
        # Confirm all timestamps have NaN values.
        self.assertTrue(dfd_update_df['grading'].isnull().all())
        self.assertTrue(dfd_update_df['mop_lag'].isnull().all())


if __name__ == '__main__':
    unittest.main()