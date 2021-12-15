

import unittest
from macrosynergy.signal.target_positions import target_positions
from tests.simulate import make_qdf
import random
import pandas as pd

class TestAll(unittest.TestCase):

    def __init__(self):
        self.cids = ['AUD', 'GBP', 'NZD', 'USD']
        self.xcats = ['FXXR_NSA', 'EQXR_NSA']

    def dataframe_generator(self):

        df_cids = pd.DataFrame(index=self.cids, columns=['earliest', 'latest',
                                                         'mean_add', 'sd_mult'])

        df_cids.loc['AUD'] = ['2010-01-01', '2020-12-31', 0, 1]
        df_cids.loc['GBP'] = ['2011-01-01', '2020-11-30', 0, 2]
        df_cids.loc['NZD'] = ['2012-01-01', '2020-12-31', 0, 3]
        df_cids.loc['USD'] = ['2013-01-01', '2020-12-31', 0, 4]

        df_xcats = pd.DataFrame(index=self.xcats, columns=['earliest', 'latest',
                                                           'mean_add', 'sd_mult',
                                                           'ar_coef', 'back_coef'])

        df_xcats.loc['FXXR_NSA'] = ['2010-01-01', '2020-12-31', 0, 1, 0, 0.2]
        df_xcats.loc['EQXR_NSA'] = ['2012-01-01', '2020-10-30', 0.5, 2, 0, 0.2]

        random.seed(2)
        dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)

        self.__dict__['dfd'] = dfd

    def test_target_positions(self):

        black = {'AUD': ['2000-01-01', '2003-12-31'],
                 'GBP': ['2018-01-01', '2100-01-01']}

        with self.assertRaises(AssertionError):
            # Test the assertion that the signal field must be present in the defined
            # dataframe. Will through an assertion.
            xcat_sig = 'INTGRWTH_NSA'
            position_df = target_positions(df=self.dfd, cids=self.cids, xcats=self.xcats,
                                           xcat_sig=xcat_sig,
                                           ctypes=['FX', 'EQ'], sigrels=[1, -1],
                                           ret='XR_NSA', blacklist=black,
                                           start='2012-01-01',
                                           end='2020-10-30', scale='prop',
                                           vtarg=0.1, signame='POS')


if __name__ == "__main__":

    unittest.main()