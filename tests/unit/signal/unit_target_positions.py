

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
        self.__dict__['xcat_sig'] = 'FXXR_NSA'

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

        dfd_reduced = reduce_df(df=self.dfd, xcats=[self.xcat_sig], cids=self.cids,
                                start='2012-01-01', end='2020-10-30',
                                blacklist=self.blacklist)
        self.__dict__['dfd_reduced'] = dfd_reduced
        self.__dict__['df_pivot'] = dfd_reduced.pivot(index="real_date", columns="cid",
                                                      values="value")

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
        condition = np.where(self.dfd_reduced['value'].to_numpy() < 0)[0]
        first_negative_index = next(iter(condition))

        val_column = df_unit_pos['value'].to_numpy()
        self.assertTrue(val_column[first_negative_index] == -1)

        # Little need to test the application of make_zn_scores(), as the function has
        # its own respective Unit Test but test on the dimensions. If the minimum number
        # of observations parameter is set to zero, the dimensions of the reduced
        # dataframe should match the output dataframe.
        df_unit_pos = unit_positions(df=self.dfd, cids=self.cids, xcat_sig=xcat_sig,
                                     blacklist=self.blacklist, start='2012-01-01',
                                     end='2020-10-30', scale='prop', min_obs=0,
                                     thresh=2.5)

        self.assertTrue(df_unit_pos.shape == self.dfd_reduced.shape)

    def unit_time_series(self):

        self.dataframe_generator()
        ret = 'XR_NSA'
        contract_returns = [c + ret for c in self.ctypes]

        # Tractable function - requires few tests. Validate each contract type is held in
        # the dictionary, and its associated value is a tuple.
        durations = time_series(self.dfd_reduced, contract_returns)

        self.assertTrue(list(durations.keys()) == contract_returns)
        self.assertTrue(all([isinstance(v, tuple) for v in durations.values()]))

    def test_return_series(self):

        self.dataframe_generator()

        sigrels = [1, -1]
        ret = 'XR_NSA'
        contract_returns = [c + ret for c in self.ctypes]
        xcat_sig = 'FXXR_NSA'

        dates = self.df_pivot.index
        dfd = reduce_df(df=self.dfd, xcats=self.xcats, cids=self.cids,
                        start='2012-01-01', end='2020-10-30',
                        blacklist=self.blacklist)

        df_pos_vt = return_series(dfd=dfd, xcat_sig=xcat_sig,
                                  contract_returns=contract_returns, sigrels=sigrels,
                                  time_index=dates, cids=self.cids, ret=ret)

        # First aspect to validate is that the return series, used for volatility
        # adjusting, is defined over the same time-period as the signal.
        dfd_sig = dfd[dfd['xcat'] == xcat_sig]
        self.assertTrue(df_pos_vt.shape == dfd_sig.shape)

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