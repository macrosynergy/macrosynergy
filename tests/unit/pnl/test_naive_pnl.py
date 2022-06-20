
from tests.simulate import make_qdf
from macrosynergy.pnl.naive_pnl import NaivePnL

import unittest
import numpy as np
import pandas as pd

class TestAll(unittest.TestCase):

    def dataframe_construction(self):

        self.__dict__['cids'] = ['AUD', 'CAD', 'GBP', 'NZD', 'USD', 'EUR']
        self.__dict__['xcats'] = ['EQXR', 'CRY', 'GROWTH', 'INFL', 'DUXR']

        df_cids = pd.DataFrame(index=self.cids, columns=['earliest', 'latest', 'mean_add',
                                                    'sd_mult'])
        df_cids.loc['AUD', :] = ['2010-01-01', '2020-12-31', 0.5, 2]
        df_cids.loc['CAD', :] = ['2010-01-01', '2020-11-30', 0, 1]
        df_cids.loc['GBP', :] = ['2012-01-01', '2020-11-30', -0.2, 0.5]
        df_cids.loc['NZD'] = ['2002-01-01', '2020-09-30', -0.1, 2]
        df_cids.loc['USD'] = ['2001-01-01', '2020-12-31', 0.2, 2]
        df_cids.loc['EUR'] = ['2001-01-01', '2020-12-31', 0.1, 2]

        df_xcats = pd.DataFrame(index=self.xcats,
                                columns=['earliest', 'latest', 'mean_add',
                                         'sd_mult', 'ar_coef', 'back_coef'])

        df_xcats.loc['EQXR'] = ['2005-01-01', '2020-12-31', 0.1, 1, 0, 0.3]
        df_xcats.loc['CRY'] = ['2000-01-01', '2020-10-30', 1, 2, 0.95, 1]
        df_xcats.loc['GROWTH'] = ['2005-01-01', '2020-10-30', 1, 2, 0.9, 1]
        df_xcats.loc['INFL'] = ['2001-01-01', '2020-10-30', 1, 2, 0.8, 0.5]
        df_xcats.loc['DUXR'] = ['2000-01-01', '2020-12-31', 0.1, 0.5, 0, 0.1]

        black = {'AUD': ['2000-01-01', '2003-12-31'],
                 'GBP': ['2018-01-01', '2100-01-01']}
        self.__dict__['blacklist'] = black

        # Standard df for tests.
        dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)
        self.__dict__['dfd'] = dfd

    def test_constructor(self):
        # Test NaivePnL's constructor and the instantiation of the respective fields.

        self.dataframe_construction()

        ret = ['EQXR']
        sigs = ['CRY', 'GROWTH', 'INFL']
        pnl = NaivePnL(self.dfd, ret=ret[0], sigs=sigs, cids=self.cids,
                       start='2000-01-01', blacklist=self.blacklist,
                       )
        # Confirm the categories held in the reduced DataFrame, on the instance's field,
        # are exclusively the return and signal category. This will occur if benchmarks
        # have not been defined.
        test_categories = list(pnl.df['xcat'].unique())
        self.assertTrue(sorted(test_categories) == sorted(ret + sigs))

        # Add "external" benchmarks to the instance: a category that is neither the
        # return field or one of the categories. The benchmarks will be added to the
        # instance's DataFrame.
        pnl = NaivePnL(self.dfd, ret=ret[0], sigs=sigs, cids=self.cids,
                       start='2000-01-01', blacklist=self.blacklist,
                       bms=["EUR_DUXR", "USD_DUXR"]
                       )
        test_categories = list(pnl.df['xcat'].unique())
        self.assertTrue(sorted(test_categories) == sorted(ret + sigs + ['DUXR']))

        # Test that both the benchmarks are held in the DataFrame. Implicitly validating
        # that add_bm() method works correctly.
        first_bm = pnl.df[(pnl.df['cid'] == "EUR") & (pnl.df['xcat'] == "DUXR")]
        self.assertTrue(not first_bm.empty)
        second_bm = pnl.df[(pnl.df['cid'] == "USD") & (pnl.df['xcat'] == "DUXR")]
        self.assertTrue(not second_bm.empty)

        # Confirm the values are correct.
        eur_duxr = self.dfd[(self.dfd['cid'] == "EUR") & (self.dfd['xcat'] == "DUXR")]
        self.assertTrue(np.all(first_bm['value'] == eur_duxr['value']))

    def test_make_pnl(self):

        self.dataframe_construction()

        pnl = NaivePnL(self.dfd, ret=ret[0], sigs=sigs, cids=self.cids,
                       start='2000-01-01', blacklist=self.blacklist,
                       bms=["EUR_DUXR", "USD_DUXR"]
                       )


if __name__ == '__main__':

    unittest.main()


