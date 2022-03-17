
import unittest
import random
import numpy as np
import pandas as pd
from tests.simulate import make_qdf
from macrosynergy.management.update_df import category_add
from macrosynergy.panel.make_relative_value import make_relative_value

class TestAll(unittest.TestCase):

    def dataframe_constructor(self):

        self.__dict__['cids'] = ['AUD', 'CAD', 'GBP', 'NZD']
        self.__dict__['xcats'] = ['GROWTH', 'INFL']

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

        random.seed(1)
        np.random.seed(0)
        self.__dict__['dfd'] = make_qdf(df_cids, df_xcats, back_ar=0.75)

        black = {'AUD': ['2000-01-01', '2003-12-31'],
                 'GBP': ['2018-01-01', '2100-01-01']}
        self.__dict__['blacklist'] = black

    def test_category_add(self):

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
            dfd_add = category_add(dfd=dfd, dfd_add=dfd_pivot)

        # Test the above method by using the in-built make_relative_value() method.
        dfd_1_rv = make_relative_value(self.dfd, xcats=['GROWTH', 'INFL'], cids=None,
                                       blacklist=None, rel_meth='subtract',
                                       rel_xcats=None,
                                       postfix='RV')

        # Both categories generated from the make_relative_value() function will not be
        # present in the aggregated dataframe. Therefore, confirm both categories are
        # appended to the existing dataframe.
        dfd_add = category_add(dfd=dfd, dfd_add=dfd_1_rv)
        # First, confirm the expected dimensions.
        self.assertTrue(dfd_add.shape[0] == (dfd.shape[0] + dfd_1_rv.shape[0]))

        # Confirm the appended dataframe's categories are a subset of the combined
        # dataframe.
        new_categories = set(dfd_1_rv['xcat'])
        self.assertTrue(new_categories.issubset(set(dfd_1_rv['xcat'])))

        dfd_1_rv_growth = dfd_1_rv[dfd_1_rv['xcat'] == 'GROWTHRV']
        dfd_1_rv_growth_aud = dfd_1_rv_growth[dfd_1_rv_growth['cid'] == 'AUD']
        # Choose a random date.
        date = list(dfd_pivot.index)[1000]

        value = dfd_1_rv_growth_aud[dfd_1_rv_growth_aud['real_date'] == date]['value']

        # Confirm the values do not change during the aggregation mechanism.
        dfd_add_growth = dfd_add[dfd_add['xcat'] == 'GROWTHRV']
        dfd_add_aud = dfd_add_growth[dfd_add_growth['cid'] == 'AUD']

        test = dfd_add_aud[dfd_add_aud['real_date'] == date]['value']
        self.assertTrue(float(test) == float(value))

        # Test the replacement mechanism: categories are already in the dataframe but are
        # to be replaced by new values.
        dfd_2_rv_divide = make_relative_value(dfd, xcats=['GROWTH', 'INFL'],
                                                 cids=None,
                                                 blacklist=None, rel_meth='divide',
                                                 rel_xcats=None,
                                                 postfix='RV')

        dfd_add_2 = category_add(dfd=dfd_add, dfd_add=dfd_2_rv_divide)
        self.assertTrue(list(set(dfd_add_2['xcat'])) == list(set(dfd_add['xcat'])))

        # Confirm the value does not equal the previous value which will be held in the
        # original dataframe, "dfd_add".
        dfd_add_growth_2 = dfd_add_2[dfd_add_2['xcat'] == 'GROWTHRV']
        dfd_add_aud_2 = dfd_add_growth_2[dfd_add_growth_2['cid'] == 'AUD']

        test_2 = dfd_add_aud_2[dfd_add_aud_2['real_date'] == date]['value']
        self.assertTrue(float(test_2) != float(value))

        # Hence, confirm the new value stored in "dfd_add_2" is sourced from the latest
        # dataframe "dfd_1_rv_divide".
        dfd_2_rv_growth = dfd_2_rv_divide[dfd_2_rv_divide['xcat'] == 'GROWTHRV']
        dfd_2_rv_growth_aud = dfd_2_rv_growth[dfd_2_rv_growth['cid'] == 'AUD']

        value_2 = dfd_2_rv_growth_aud[dfd_2_rv_growth_aud['real_date'] == date]['value']
        self.assertTrue(float(test_2) == float(value_2))


if __name__ == '__main__':
    unittest.main()