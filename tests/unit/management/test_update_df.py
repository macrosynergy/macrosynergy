
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
            dfd_pivot = dfd_1_rv_growth.pivot(index="real_date", columns="ticker",
                                              values="value")
            dfd_add = category_add(dfd=dfd, dfd_add=dfd_pivot)

        # Test the above method by using the in-built make_relative_value() method.
        dfd_1_rv = make_relative_value(self.dfd, xcats=['GROWTH', 'INFL'], cids=None,
                                       blacklist=None, rel_meth='subtract',
                                       rel_xcats=None,
                                       postfix='RV')

        dfd_add = category_add(dfd=self.dfd, dfd_add=dfd_1_rv)


if __name__ == '__main__':
    unittest.main()