import unittest
import random
import numpy as np
import pandas as pd
from collections import deque

from tests.simulate import make_qdf
from macrosynergy.panel.historic_vol import *
from macrosynergy.management.shape_dfs import reduce_df


class TestAll(unittest.TestCase):

    def dataframe_generator(self):

        self.cids : List[str] = ['AUD', 'CAD', 'GBP']
        self.xcats : List[str] = ['CRY', 'XR']

        df_cids = pd.DataFrame(index=self.cids,
                               columns=['earliest', 'latest', 'mean_add', 'sd_mult'])
        df_cids.loc['AUD', :] = ['2010-01-01', '2020-12-31', 0.5, 2]
        df_cids.loc['CAD', :] = ['2011-01-01', '2020-11-30', 0, 1]
        df_cids.loc['GBP', :] = ['2012-01-01', '2020-11-30', -0.2, 0.5]

        df_xcats = pd.DataFrame(index=self.xcats,
                                columns=['earliest', 'latest', 'mean_add', 'sd_mult',
                                         'ar_coef', 'back_coef'])

        df_xcats.loc['CRY', :] = ['2011-01-01', '2020-10-30', 1, 2, 0.9, 0.5]
        df_xcats.loc['XR', :] = ['2010-01-01', '2020-12-31', 0, 1, 0, 0.3]

        dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)
        self.dfd : pd.DataFrame = dfd

    def test_expo_weights(self):
        lback_periods = 21
        half_life = 11
        w_series = expo_weights(lback_periods, half_life)

        self.assertIsInstance(w_series, np.ndarray)
        self.assertTrue(len(w_series) == lback_periods)  # Check correct length.
        # Check that weights add up to zero.
        self.assertTrue(sum(w_series) - 1.0 < 0.00000001)
        # Check that weights array is monotonic.
        self.assertTrue(all(w_series == sorted(w_series)))

    def test_expo_std(self):
        lback_periods = 21
        half_life = 11
        w_series = expo_weights(lback_periods, half_life)

        with self.assertRaises(AssertionError):
            data = np.random.randint(0, 25, size=lback_periods + 1)
            expo_std(data, w_series, False)

        data = np.random.randint(0, 25, size=lback_periods)
        output = expo_std(data, w_series, False)
        self.assertIsInstance(output, float)  # check type

        arr = np.array([i for i in range(1, 11)])
        pd_ewm = pd.Series(arr).ewm(halflife=5, min_periods=10).mean()[9]
        s_weights = expo_weights(len(arr), 5)
        output_expo = expo_std(arr, s_weights, True)
        self.assertAlmostEqual(output_expo, pd_ewm)  # Check value consistent with pandas calculation.

        arr = np.array([0, 0, -7, 0, 0, 0, 0, 0, 0])
        s_weights = expo_weights(len(arr), 5)
        output_expo = expo_std(arr, s_weights, True)
        # Check if single non-zero value becomes average.
        self.assertEqual(output_expo, 7)

    def test_flat_std(self):
        data = [2, -11, 9, -3, 1, 17, 19]
        output_flat = float(flat_std(data, remove_zeros=False))
        output_flat = round(output_flat, ndigits=6)
        data = [abs(elem) for elem in data]
        output_test = round(sum(data) / len(data), 6)
        self.assertEqual(output_flat, output_test)  # test correct average

        lback_periods = 21
        data = np.random.randint(0, 25, size=lback_periods)

        output = flat_std(data, True)
        self.assertIsInstance(output, float)  # test type
        
    def test_get_cycles(self):
        daterange1 = pd.bdate_range(start="2023-01-28", end="2023-02-02")
        test_case_1 = pd.DataFrame({"real_date": pd.Series(daterange1)})
        # NOTE: get_cycles(freq=...) is case insensitive
        test_result_1 = get_cycles(test_case_1, freq="M")
        # expected results : 2023-01-31 (last of cycle), last index

        expc_vals = set(
            np.array(
                [pd.Timestamp("2023-01-31"), daterange1[-1]], dtype="datetime64[ns]"
            ).tolist()
        )
        test_vals = set(test_case_1[test_result_1]["real_date"].values.tolist())
        self.assertEqual(expc_vals, test_vals)

        daterange2 = pd.bdate_range(start="2023-03-20", end="2023-07-10")
        test_case_2 = pd.DataFrame({"real_date": pd.Series(daterange2)})
        test_result_2 = get_cycles(test_case_2, freq="q")
        # expected results : 2023-03-31 (last of cycle), 2023-06-30 (last of cycle), last index

        expc_vals = set(
            np.array(
                [pd.Timestamp("2023-03-31"), pd.Timestamp("2023-06-30"), daterange2[-1]],
                dtype="datetime64[ns]",
            ).tolist()
        )
        test_vals = set(test_case_2[test_result_2]["real_date"].values.tolist())
        self.assertEqual(expc_vals, test_vals)

        daterange3 = pd.bdate_range(start="2000-01-01", end="2023-07-10")
        test_case_3 = pd.DataFrame({"real_date": pd.Series(daterange3)})
        test_result_3 = get_cycles(test_case_3, freq="m")

        # expc_vals = set(
        # print(test_case_3[test_result_3]["real_date"].values.tolist())
        expc_vals = []
        r_start_date = pd.Timestamp("2000-01-01")
        r_end_date = pd.Timestamp("2023-07-10")

        expc_no_cycles = (r_end_date.year - r_start_date.year) * 12 + (
            r_end_date.month - r_start_date.month
        )
        self.assertEqual(
            int(expc_no_cycles), int(sum(test_result_3) - 1)
        )  # -1 as last index is always True

        test_vals = [
            [pd.Timestamp("2023-01-31"), True],
            [pd.Timestamp("2016-02-29"), True],
            # [pd.Timestamp("2023-02-29"), False],
            # this timestamp doesn't exist and will raise an Exception
            [pd.Timestamp("2023-02-28"), True],
            [pd.Timestamp("2023-03-20"), False],
            [pd.Timestamp("2005-12-30"), True],
            [pd.Timestamp("2000-12-29"), True],
        ]
        for tval in test_vals:
            self.assertEqual(
                test_result_3[test_case_3["real_date"] == tval[0]].values[0], tval[1]
            )



    def test_historic_vol(self):

        self.dataframe_generator()
        xcat = 'XR'

        lback_periods = 21
        half_life = 14        
        df_output = historic_vol(self.dfd, xcat, self.cids, lback_periods=lback_periods,
                                 lback_meth='xma', half_life=3, start=None,
                                 end=None, blacklist=None, remove_zeros=True,
                                 postfix='ASD', est_freq='w', nan_tolerance=0)

        # Test correct column names.
        self.assertTrue(all(df_output.columns == self.dfd.columns))
        cross_sections = sorted(list(set(df_output['cid'].values)))
        self.assertTrue(cross_sections == self.cids)
        self.assertTrue(all(df_output['xcat'] == xcat + 'ASD'))

        # assert that the first (lback_periods - 1) rows of df_output are NaN.
        # TODO: implement tests

        # Test the stacking procedure to reconstruct the standardised dataframe from the
        # pivoted counterpart.
        # The in-built pandas method, df.stack(), used will, by default, drop
        # all NaN values, as the preceding pivoting operation requires populating each
        # column field such that each field is defined over the same index (time-period).
        # Therefore, the stack() method treats NaN values as contrived inputs generated
        # from the pivot mechanism, and subsequently the respective dates of the lookback
        # period will also be dropped.
        # The overall outcome is that the returned standardised dataframe should be
        # reduced by the number cross-sections multiplied by the length of the lookback
        # period minus one.
        # Test the above logic.

        # select 1 cross-section and 1 xcat to test the dimensionality reduction.
        df_reduce = reduce_df(df=self.dfd, xcats=[xcat], cids=self.cids, start=None,
                              end=None, blacklist=None)

        # the shape of the df_output should be the same as the shape of reduce_df.
        self.assertTrue(df_output[['cid', 'xcat', 'real_date']].shape \
                        == df_reduce[['cid', 'xcat', 'real_date']].shape)

        lback_periods = 20
        half_life = 8
        for freqst in ["m", "q"]:
            for xcatt in ['XR', 'CRY']:
                df_test_res = historic_vol(self.dfd, xcatt, self.cids, lback_periods=lback_periods,
                                             lback_meth='ma', half_life=half_life, start=None,
                                             end=None, blacklist=None, remove_zeros=True,
                                                postfix='ASD', est_freq=freqst)
                df_reduce = reduce_df(df=self.dfd, xcats=[xcatt], cids=self.cids, start=None,
                                      end=None, blacklist=None)
                self.assertTrue(df_test_res[['cid', 'xcat', 'real_date']].shape ==\
                                df_reduce[['cid', 'xcat', 'real_date']].shape)
                
                
        # Test the number of NaN values in the long format dataframe.
        
        df_nas_test = historic_vol(self.dfd, xcat, self.cids, 
                                        lback_periods=lback_periods,
                                        lback_meth='ma', half_life=half_life, start=None,
                                        end=None, blacklist=None, remove_zeros=True,
                                        postfix='ASD', est_freq='w', nan_tolerance=0)

        
        # NOTE: ideally, one would use the get_cycles() function in conjunction with the
        # est_freq behaviour to determine the number of NaN values in the long format.
        # The below approach also works;
        
        # for GBP, from 2012-01-01 to 2012-02-01, there should only be 4 non-NaN values.
        nas_test_res = df_nas_test[(df_nas_test['cid'] == 'GBP') & 
                                   (df_nas_test['real_date'].isin(
                                       pd.bdate_range('2012-01-01','2012-02-01')))
                                   ]['value']
        
        self.assertTrue(nas_test_res.notna().sum() == 4)
        self.assertTrue(nas_test_res.isna().sum() == 19)
        
        # since the last 4 are non-NaNs, the 5th to last is a NaN value.
        self.assertTrue(nas_test_res.isna().tolist()[-5] == True)
        self.assertFalse(any(nas_test_res.isna().tolist()[-4:]))
        
        # test again, but for CAD from 2011-01-01 to 2011-02-01.
        # this case should have 3 non-NaN values (last 3) and 19 NaN values.
        nas_test_res = df_nas_test[(df_nas_test['cid'] == 'CAD') & 
                                   (df_nas_test['real_date'].isin(
                                       pd.bdate_range('2011-01-01', '2011-02-01')))
                                   ]['value']

        self.assertTrue(nas_test_res.notna().sum() == 3)
        self.assertTrue(nas_test_res.isna().sum() == 19)
        
        # since the last 3 are non-NaNs, the 4th to last is a NaN value.
        self.assertTrue(nas_test_res.isna().tolist()[-4] == True)
        self.assertFalse(any(nas_test_res.isna().tolist()[-3:]))
        
        # repeat the test for the 'xma' method but use monthly estimation frequency.
        df_nas_test = historic_vol(self.dfd, xcat, self.cids, lback_periods=25,
                                        lback_meth='xma', half_life=10, start=None,
                                        end=None, blacklist=None, remove_zeros=True,
                                        postfix='ASD', est_freq='m', nan_tolerance=0)

        nas_test_res = df_nas_test[(df_nas_test['cid'] == 'CAD') & 
                                   (df_nas_test['real_date'].isin(
                                       pd.bdate_range('2011-01-01', '2011-03-01')))]

        self.assertTrue(nas_test_res.set_index('real_date')['value']
                            .first_valid_index() == pd.Timestamp('2011-02-28'))
        
        df_nas_test = historic_vol(self.dfd, xcat, self.cids, lback_periods=50,
                                lback_meth='xma', half_life=10, start=None,
                                end=None, blacklist=None, remove_zeros=True,
                                postfix='ASD', est_freq='m', nan_tolerance=0)
        
        nas_test_res = df_nas_test[(df_nas_test['cid'] == 'CAD') & 
                            (df_nas_test['real_date'].isin(
                                pd.bdate_range('2011-01-01', '2011-05-01')))]
        
        # the first valid index should be 2011-03-31.
        self.assertTrue(nas_test_res.set_index('real_date')['value']
                            .first_valid_index() == pd.Timestamp('2011-03-31'))
        
        
        # run with the same args, but est_freq='q'. should have the same first valid index.
        df_nas_test = historic_vol(self.dfd, xcat, self.cids, lback_periods=50,
                                lback_meth='xma', half_life=10, start=None,
                                end=None, blacklist=None, remove_zeros=True,
                                postfix='ASD', est_freq='q', nan_tolerance=0)

        nas_test_res = df_nas_test[(df_nas_test['cid'] == 'CAD') &
                            (df_nas_test['real_date'].isin(
                                pd.bdate_range('2011-01-01', '2011-05-01')))]

        self.assertTrue(nas_test_res.set_index('real_date')['value']
                            .first_valid_index() == pd.Timestamp('2011-03-31'))
              
        
        # pass a half-life value that is greater than the lookback period. to see errors
        # this case should not raise an error ('ma' unaffected by half-life).
        self.assertTrue(isinstance(historic_vol(self.dfd, 'XR', self.cids, 
                        lback_periods=7, lback_meth='ma',half_life=11, 
                        start=None, end=None, blacklist=None, est_freq='m',
                        remove_zeros=True, postfix='ASD'), pd.DataFrame))
        
        # same args but with 'xma' method should raise an error.
        with self.assertRaises(AssertionError):
            historic_vol(self.dfd, 'XR', self.cids, lback_periods=7, lback_meth='xma',
                         half_life=11, start=None, end=None, blacklist=None,
                         remove_zeros=True, postfix='ASD')

        # this should raise an error as the lbakc_meth is invalid.
        with self.assertRaises(AssertionError):
            historic_vol(self.dfd, 'CRY', self.cids, lback_periods=7, lback_meth='ema',
                         half_life=11, start=None, end=None, blacklist=None,
                         remove_zeros=True, postfix='ASD')

        # Todo: check correct exponential averages for a whole series on toy data set using (.rolling) and .ewm


if __name__ == '__main__':

    unittest.main()