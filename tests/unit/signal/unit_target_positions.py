

import unittest
from macrosynergy.signal.target_positions import *
from macrosynergy.management.shape_dfs import reduce_df
from tests.simulate import make_qdf
import random
import pandas as pd
import numpy as np

class TestAll(unittest.TestCase):

    def dataframe_generator(self):

        self.__dict__['cids'] = ['AUD', 'GBP', 'NZD', 'USD']
        self.__dict__['xcats'] = ['FXXR_NSA', 'EQXR_NSA']
        self.__dict__['ctypes'] = ['FX', 'EQ']

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

        black = {'AUD': ['2000-01-01', '2003-12-31'],
                 'GBP': ['2018-01-01', '2100-01-01']}

        self.__dict__['blacklist'] = black

        assert 'dfd' in vars(self).keys(), "Instantiation of DataFrame missing from " \
                                           "field dictionary."

    def test_unitary_positions(self):
        self.dataframe_generator()

        with self.assertRaises(AssertionError):
            # Testing the assertion on the scale parameter: required ['prop', 'dig'].
            # Pass in noise.
            scale = 'vtarg'
            position_df = target_positions(df=self.dfd, cids=self.cids, xcats=self.xcats,
                                           xcat_sig='FXXR_NSA',
                                           ctypes=['FX', 'EQ'], sigrels=[1, -1],
                                           ret='XR_NSA', blacklist=self.blacklist,
                                           start='2012-01-01',
                                           end='2020-10-30', scale=scale,
                                           vtarg=0.1, signame='POS')

        # Unitary Position function should return a dataframe consisting of a single
        # category, the signal, and the respective dollar position.
        xcat_sig = 'FXXR_NSA'
        df_unit_pos = unit_positions(df=self.dfd, cids=self.cids, xcat_sig=xcat_sig,
                                     blacklist=self.blacklist, start='2012-01-01',
                                     end='2020-10-30', scale='dig', thresh=2.5)

        self.assertTrue(df_unit_pos['xcat'].unique().size == 1)
        self.assertTrue(np.all(df_unit_pos['xcat'] == xcat_sig))

        # Digital unitary position can be validated by summing the absolute values of the
        # 'value' column and validating that the summed value equates to the number of
        # rows.
        summation = np.sum(np.abs(df_unit_pos['value']))
        self.assertTrue(summation == df_unit_pos.shape[0])

        # Reduce the dataframe & check the logic is correct.
        dfd_reduced = reduce_df(df=self.dfd, xcats=[xcat_sig], cids=self.cids,
                                start='2012-01-01', end='2020-10-30',
                                blacklist=self.blacklist)

        condition = np.where(dfd_reduced['value'].to_numpy() < 0)[0]
        first_negative_index = next(iter(condition))

        val_column = df_unit_pos['value'].to_numpy()
        self.assertTrue(val_column[first_negative_index] == -1)

    def test_return_series(self):

        self.dataframe_generator()

        sigrels = [1, -1]
        ret = 'XR_NSA'
        contract_returns = [c + 'XR_NSA' for c in self.ctypes]

        df_pos_vt = return_series(dfd=self.dfd, contract_returns=contract_returns,
                                  sigrels=sigrels, ret=ret)

    def test_target_positions(self):

        self.dataframe_generator()

        with self.assertRaises(AssertionError):
            # Test the assertion that the signal field must be present in the defined
            # dataframe. Will through an assertion.
            xcat_sig = 'INTGRWTH_NSA'
            position_df = target_positions(df=self.dfd, cids=self.cids, xcats=self.xcats,
                                           xcat_sig=xcat_sig,
                                           ctypes=['FX', 'EQ'], sigrels=[1, -1],
                                           ret='XR_NSA', blacklist=self.blacklist,
                                           start='2012-01-01',
                                           end='2020-10-30', scale='prop',
                                           vtarg=0.1, signame='POS')

        with self.assertRaises(AssertionError):
            # Testing the assertion on the scale parameter: required ['prop', 'dig'].
            # Pass in noise.
            scale = 'vtarg'
            position_df = target_positions(df=self.dfd, cids=self.cids, xcats=self.xcats,
                                           xcat_sig=xcat_sig,
                                           ctypes=['FX', 'EQ'], sigrels=[1, -1],
                                           ret='XR_NSA', blacklist=self.blacklist,
                                           start='2012-01-01',
                                           end='2020-10-30', scale=scale,
                                           vtarg=0.1, signame='POS')

        with self.assertRaises(AssertionError):
            # Length of "sigrels" and "ctypes" must be the same. The number of "sigrels"
            # corresponds to the number of "ctypes" defined in the data structure passed
            # into target_positions().

            sigrels = [1, -1, 0.5, 0.25]
            ctypes = ['FX', 'EQ']
            position_df = target_positions(df=self.dfd, cids=self.cids, xcats=self.xcats,
                                           xcat_sig=xcat_sig,
                                           ctypes=ctypes, sigrels=sigrels,
                                           ret='XR_NSA', blacklist=self.blacklist,
                                           start='2012-01-01',
                                           end='2020-10-30', scale=scale,
                                           vtarg=0.1, signame='POS')


if __name__ == "__main__":

    unittest.main()