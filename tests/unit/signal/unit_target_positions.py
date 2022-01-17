

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
        # its own respective Unit Test but test on the dimensions.
        # If the minimum number of observations parameter is set to zero,
        # the dimensions of the reduced dataframe should match the output dataframe.
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

        # The main aspect to validate is that the return series, used for volatility
        # adjusting, is defined over the same time-period as the signal. In the below
        # test framework the signal is the longest contract, so any of the truncating
        # functionality will not be required. Further, both return series are reduced
        # to the length of the shorter contract (start = '2012-01-01',
        # end = '2020-10-30'), and subsequently there will not be any alignment issue.
        # Each return, of the portfolio, will be a consequence of summing the individual
        # returns from both contracts.
        dfd_sig = dfd[dfd['xcat'] == xcat_sig]
        self.assertTrue(df_pos_vt.shape == dfd_sig.shape)

        # In the below test framework, the signal will be defined over a longer time-
        # period than the other contract, EQXR_NSA. Therefore, the design of the
        # algorithm ensures the return-series dataframe, used for volatility adjustments,
        # will consist of exclusively the signal contract for the aforementioned dates
        # (timestamps up until EQXR_NSA is realised).
        # Test the above logic.
        dfd = reduce_df(df=self.dfd, xcats=self.xcats, cids=self.cids,
                        start='2010-01-01', end='2020-12-31',
                        blacklist=self.blacklist)

        df_signal = dfd[dfd['xcat'] == xcat_sig]
        df_signal_piv = df_signal.pivot(index="real_date", columns="cid",
                                        values="value").sort_index(axis=1)
        # The timestamps the signal is defined over are used for constructing the
        # portfolio dataframe.
        dates = df_signal_piv.index
        df_signal_trunc = df_signal_piv.truncate(after='2012-01-01')

        # Until '2012-01-01', the returned value should match the signal's return given
        # the sigrel is equal to one for the signal.
        df_pos_vt = return_series(dfd=dfd, xcat_sig=xcat_sig,
                                  contract_returns=contract_returns, sigrels=sigrels,
                                  time_index=dates, cids=self.cids, ret=ret)

        df_pos_vt_piv = df_pos_vt.pivot(index="real_date", columns="cid",
                                        values="value").sort_index(axis=1)
        df_pos_vt_trunc = df_pos_vt_piv.truncate(after='2012-01-01')

        assert df_signal_trunc.shape == df_pos_vt_trunc.shape
        difference = np.nan_to_num(df_signal_trunc.to_numpy() - df_pos_vt_trunc.to_numpy())

        # Accounts for floating point precision.
        self.assertTrue(np.all(difference < 0.000001))

        # The first joint return, portfolio return consisting of the two categories,
        # should be, if the logic is correct, on '2012-01-01' (if a valid business day).
        # Coincidentally 2012-01-01 is a weekend, so test on 2012-01-02 which would still
        # validate if the logic on the function is correct.
        test = df_pos_vt_piv.loc['2012-01-02'].to_numpy()

        # The signal will invariably populate the returned dataframe first: it is the
        # basis of the portfolio's return.
        signal = df_signal_piv.sort_index(axis=1).loc['2012-01-02'] * sigrels[0]

        secondary_cat = dfd[dfd['xcat'] == 'EQXR_NSA'].pivot(index="real_date",
                                                             columns="cid",
                                                             values="value")
        secondary_cat = secondary_cat.loc['2012-01-02'] * sigrels[-1]
        logic = np.nan_to_num(signal.to_numpy() + secondary_cat.to_numpy())

        self.assertTrue(np.all(np.nan_to_num(test) == logic))

        # The other testcase is if the other contract is defined over a longer
        # time-period but the returned dataframe should match the dimensions of the
        # signal contract (time-period of interest).
        xcat_sig = 'EQXR_NSA'
        sigrels = [0.5, -0.5]
        # Required to obtain the time-index and use as the benchmark.
        df_signal = reduce_df(df=dfd, xcats=[xcat_sig], cids=self.cids,
                              start='2010-01-01', end='2020-12-31',
                              blacklist=self.blacklist)
        df_signal_piv = df_signal.pivot(index="real_date", columns="cid", values="value")
        dates = df_signal_piv.index

        # Reduce the dataframe to the length of the longer category, FXXR_NSA, such that
        # the dataframe is applicable to testcase.
        df_pos_vt = return_series(dfd=dfd, xcat_sig=xcat_sig,
                                  contract_returns=contract_returns, sigrels=sigrels,
                                  time_index=dates, cids=self.cids, ret=ret)

        self.assertEqual(df_signal.shape, df_pos_vt.shape)
        # Test the values to confirm the logic using the first index.
        test = df_pos_vt.pivot(index="real_date", columns="cid",
                               values="value").sort_index(axis=1).loc['2012-01-02']

        signal = df_signal_piv.loc['2012-01-02'] * sigrels[0]
        secondary_cat = dfd[dfd['xcat'] == 'FXXR_NSA'].pivot(index="real_date",
                                                             columns="cid",
                                                             values="value")
        secondary_cat = secondary_cat.sort_index(axis=1).loc['2012-01-02'] * sigrels[-1]

        logic = np.nan_to_num((signal + secondary_cat).to_numpy())
        self.assertTrue(np.all(np.nan_to_num(test.to_numpy()) == logic))

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
                                           ctypes=self.ctypes, sigrels=sigrels,
                                           ret='XR_NSA', blacklist=self.blacklist,
                                           start='2012-01-01',
                                           end='2020-10-30', scale=scale,
                                           vtarg=0.1, signame='POS')

        # In the current test framework there are two contract types, FX & EQ, and the
        # signal used to subsequently construct a position is FXXR_NSA. If EQXR_NSA is
        # defined over a shorter timeframe, the position signal must be aligned to its
        # respective timeframe.
        # Check the logic matches the above description.

        # The returned standardised dataframe will consist of the signal and the
        # secondary category where the secondary category will be defined over a shorter
        # time-period.
        xcat_sig = 'FXXR_NSA'
        sigrels = [1, -1]
        output_df = target_positions(df=self.dfd, cids=self.cids, xcats=self.xcats,
                                     xcat_sig=xcat_sig, ctypes=self.ctypes,
                                     sigrels=sigrels, ret='XR_NSA',
                                     blacklist=self.blacklist, start='2010-01-01',
                                     end='2020-12-31', scale='dig',
                                     vtarg=None, signame='POS')
        # Testing the truncation feature. Concept might appear simple but implementation
        # is subtle. Isolate the secondary category in the output dataframe. The
        # dimensions should match.
        output_df['xcat'] = list(map(lambda str_: str_[4:-4], output_df['xcat']))

        df_eqxr = output_df[output_df['xcat'] == 'EQXR_NSA'].pivot(index="real_date",
                                                                   columns="cid",
                                                                   values="value")
        df_eqxr_input = self.dfd[self.dfd['xcat'] == 'EQXR_NSA'].pivot(index="real_date",
                                                                       columns="cid",
                                                                       values="value")
        self.assertTrue(df_eqxr.shape == df_eqxr_input.shape)

        # Lastly, check the stacking procedure. In this instance, the output dateframe
        # should match the input dataframe.
        dfd = reduce_df(df=self.dfd, xcats=self.xcats, cids=self.cids,
                        start='2010-01-01', end='2020-12-31', blacklist=self.blacklist)
        self.assertTrue(output_df.shape == dfd.shape)

        # Test the dimensions of the signal to confirm the differing size of the two
        # individual dataframes that are concatenated into a single structure. The output
        # dataframe should match the dimensions of the two respective inputs.
        df_signal = output_df[output_df['xcat'] == xcat_sig].pivot(index="real_date",
                                                                   columns="cid",
                                                                   values="value")
        df_signal_input = self.dfd[self.dfd['xcat'] == xcat_sig].pivot(index="real_date",
                                                                       columns="cid",
                                                                       values="value")
        self.assertTrue(df_signal.shape == df_signal_input.shape)

        # A limited but valid test to determine the logic is correct for the volatility
        # target is to set the value equal to zero, and consequently all the
        # corresponding positions should also be zero. If zero volatility is required,
        # unable to take a position in an asset that has non-zero standard deviation.
        output_df = target_positions(df=self.dfd, cids=self.cids, xcats=self.xcats,
                                     xcat_sig=xcat_sig, ctypes=self.ctypes,
                                     sigrels=sigrels, ret='XR_NSA',
                                     blacklist=self.blacklist, start='2010-01-01',
                                     end='2020-12-31', scale='dig',
                                     vtarg=0.0, signame='POS')
        self.assertTrue(np.all(output_df['value'] == 0.0))

        # Final check is if the signal category is defined over the shorter timeframe
        # both contracts individual position dataframes should match the signal.
        xcat_sig = 'EQXR_NSA'
        output_df = target_positions(df=self.dfd, cids=self.cids, xcats=self.xcats,
                                     xcat_sig=xcat_sig, ctypes=self.ctypes,
                                     sigrels=sigrels, ret='XR_NSA',
                                     blacklist=self.blacklist, start='2010-01-01',
                                     end='2020-12-31', scale='dig',
                                     vtarg=0.85, signame='POS')

        # Will not be able to compare against the signal's input dimensions because of
        # the application of volatility targeting and the associated lookback period.
        output_df['xcat'] = list(map(lambda str_: str_[4:-4], output_df['xcat']))
        df_signal = output_df[output_df['xcat'] == xcat_sig]
        df_fxxr = output_df[output_df['xcat'] == 'FXXR_NSA']

        self.assertTrue(df_signal.shape == df_fxxr.shape)


if __name__ == "__main__":

    pass
    # unittest.main()