import unittest
import numpy as np
import pandas as pd

from typing import List

from macrosynergy.management.simulate import make_qdf
from macrosynergy.panel.historic_vol import (
    historic_vol,
    expo_weights,
    expo_std,
    flat_std,
    sq_std,
)
from macrosynergy.management.utils import reduce_df


class TestEstimationMethods(unittest.TestCase):
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
        self.assertAlmostEqual(
            output_expo, pd_ewm
        )  # Check value consistent with pandas calculation.

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

    def test_sq_std(self):
        lback_periods = 10
        half_life = 5
        w_series = expo_weights(lback_periods, half_life)

        # Mismatched lengths should raise AssertionError
        with self.assertRaises(AssertionError):
            data = np.random.randn(lback_periods + 1)
            sq_std(data, w_series)

        # Output should be float for valid inputs
        data = np.random.randn(lback_periods)
        output = sq_std(data, w_series, remove_zeros=False)
        self.assertIsInstance(output, float)

        # Compare with numpy weighted std for consistency
        arr = np.array([1, 2, 3, 4, 5])
        w = expo_weights(len(arr), 2)
        manual = np.sqrt(np.sum(w * (arr - np.sum(w * arr)) ** 2))
        self.assertAlmostEqual(sq_std(arr, w, False), manual)

        # Test with zeros removed
        arr = np.array([0, 0, 7, 0, 0])
        w = expo_weights(len(arr), 3)
        result = sq_std(arr, w, True)
        self.assertEqual(result, 0.0)  # only one non-zero → std = 0


class TestAll(unittest.TestCase):
    def setUp(self) -> None:
        return super().setUp()

    def tearDown(self) -> None:
        return super().tearDown()

    def dataframe_generator(self):
        self.cids: List[str] = ["AUD", "CAD", "GBP"]
        self.xcats: List[str] = ["CRY", "XR"]

        df_cids = pd.DataFrame(
            index=self.cids, columns=["earliest", "latest", "mean_add", "sd_mult"]
        )
        df_cids.loc["AUD", :] = ["2010-01-01", "2020-12-31", 0.5, 2]
        df_cids.loc["CAD", :] = ["2011-01-01", "2020-11-30", 0, 1]
        df_cids.loc["GBP", :] = ["2012-01-01", "2020-11-30", -0.2, 0.5]

        df_xcats = pd.DataFrame(
            index=self.xcats,
            columns=[
                "earliest",
                "latest",
                "mean_add",
                "sd_mult",
                "ar_coef",
                "back_coef",
            ],
        )

        df_xcats.loc["CRY", :] = ["2011-01-01", "2020-10-30", 1, 2, 0.9, 0.5]
        df_xcats.loc["XR", :] = ["2010-01-01", "2020-12-31", 0, 1, 0, 0.3]

        dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)
        self.dfd: pd.DataFrame = dfd

    def test_historic_vol(self):
        self.dataframe_generator()
        xcat = "XR"

        lback_periods = 21
        half_life = 14
        df_output = historic_vol(
            self.dfd,
            xcat,
            self.cids,
            lback_periods=lback_periods,
            lback_meth="xma",
            half_life=3,
            start=None,
            end=None,
            blacklist=None,
            remove_zeros=True,
            postfix="ASD",
            est_freq="w",
            nan_tolerance=0,
        )

        # Test correct column names.
        self.assertEqual(set(df_output.columns), set(self.dfd.columns))
        cross_sections = sorted(list(set(df_output["cid"].values)))
        self.assertTrue(cross_sections == self.cids)
        self.assertTrue(all(df_output["xcat"] == xcat + "ASD"))

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
        df_reduce = reduce_df(
            df=self.dfd,
            xcats=[xcat],
            cids=self.cids,
            start=None,
            end=None,
            blacklist=None,
        )

        # the shape of the df_output should be the same as the shape of reduce_df.
        self.assertTrue(
            df_output[["cid", "xcat", "real_date"]].shape
            == df_reduce[["cid", "xcat", "real_date"]].shape
        )

        lback_periods = 20
        half_life = 8
        for freqst in ["m", "q"]:
            for xcatt in ["XR", "CRY"]:
                df_test_res = historic_vol(
                    self.dfd,
                    xcatt,
                    self.cids,
                    lback_periods=lback_periods,
                    lback_meth="ma",
                    half_life=half_life,
                    start=None,
                    end=None,
                    blacklist=None,
                    remove_zeros=True,
                    postfix="ASD",
                    est_freq=freqst,
                )
                df_reduce = reduce_df(
                    df=self.dfd,
                    xcats=[xcatt],
                    cids=self.cids,
                    start=None,
                    end=None,
                    blacklist=None,
                )
                self.assertTrue(
                    df_test_res[["cid", "xcat", "real_date"]].shape
                    == df_reduce[["cid", "xcat", "real_date"]].shape
                )

        # Test the number of NaN values in the long format dataframe.

        df_nas_test = historic_vol(
            self.dfd,
            xcat,
            self.cids,
            lback_periods=lback_periods,
            lback_meth="ma",
            half_life=half_life,
            start=None,
            end=None,
            blacklist=None,
            remove_zeros=True,
            postfix="ASD",
            est_freq="w",
            nan_tolerance=0,
        )

        # NOTE: ideally, one would use the get_eops() function in conjunction with the
        # est_freq behaviour to determine the number of NaN values in the long format.
        # The below approach also works;

        # for GBP, from 2012-01-01 to 2012-02-01, there should only be 4 non-NaN values.
        nas_test_res = df_nas_test[
            (df_nas_test["cid"] == "GBP")
            & (
                df_nas_test["real_date"].isin(
                    pd.bdate_range("2012-01-01", "2012-02-01")
                )
            )
        ]["value"]

        self.assertTrue(nas_test_res.notna().sum() == 4)
        self.assertTrue(nas_test_res.isna().sum() == 19)

        # since the last 4 are non-NaNs, the 5th to last is a NaN value.
        self.assertTrue(nas_test_res.isna().tolist()[-5])
        self.assertFalse(any(nas_test_res.isna().tolist()[-4:]))

        # test again, but for CAD from 2011-01-01 to 2011-02-01.
        # this case should have 3 non-NaN values (last 3) and 19 NaN values.
        nas_test_res = df_nas_test[
            (df_nas_test["cid"] == "CAD")
            & (
                df_nas_test["real_date"].isin(
                    pd.bdate_range("2011-01-01", "2011-02-01")
                )
            )
        ]["value"]

        self.assertTrue(nas_test_res.notna().sum() == 3)
        self.assertTrue(nas_test_res.isna().sum() == 19)

        # since the last 3 are non-NaNs, the 4th to last is a NaN value.
        self.assertTrue(nas_test_res.isna().tolist()[-4])
        self.assertFalse(any(nas_test_res.isna().tolist()[-3:]))

        # repeat the test for the 'xma' method but use monthly estimation frequency.
        df_nas_test = historic_vol(
            self.dfd,
            xcat,
            self.cids,
            lback_periods=25,
            lback_meth="xma",
            half_life=10,
            start=None,
            end=None,
            blacklist=None,
            remove_zeros=True,
            postfix="ASD",
            est_freq="m",
            nan_tolerance=0,
        )

        nas_test_res = df_nas_test[
            (df_nas_test["cid"] == "CAD")
            & (
                df_nas_test["real_date"].isin(
                    pd.bdate_range("2011-01-01", "2011-03-01")
                )
            )
        ]

        self.assertTrue(
            nas_test_res.set_index("real_date")["value"].first_valid_index()
            == pd.Timestamp("2011-02-28")
        )

        df_nas_test = historic_vol(
            self.dfd,
            xcat,
            self.cids,
            lback_periods=50,
            lback_meth="xma",
            half_life=10,
            start=None,
            end=None,
            blacklist=None,
            remove_zeros=True,
            postfix="ASD",
            est_freq="m",
            nan_tolerance=0,
        )

        nas_test_res = df_nas_test[
            (df_nas_test["cid"] == "CAD")
            & (
                df_nas_test["real_date"].isin(
                    pd.bdate_range("2011-01-01", "2011-05-01")
                )
            )
        ]

        # the first valid index should be 2011-03-31.
        self.assertTrue(
            nas_test_res.set_index("real_date")["value"].first_valid_index()
            == pd.Timestamp("2011-03-31")
        )

        # run with the same args, but est_freq='q'. should have the same first valid index.
        df_nas_test = historic_vol(
            self.dfd,
            xcat,
            self.cids,
            lback_periods=50,
            lback_meth="xma",
            half_life=10,
            start=None,
            end=None,
            blacklist=None,
            remove_zeros=True,
            postfix="ASD",
            est_freq="q",
            nan_tolerance=0,
        )

        nas_test_res = df_nas_test[
            (df_nas_test["cid"] == "CAD")
            & (
                df_nas_test["real_date"].isin(
                    pd.bdate_range("2011-01-01", "2011-05-01")
                )
            )
        ]

        self.assertTrue(
            nas_test_res.set_index("real_date")["value"].first_valid_index()
            == pd.Timestamp("2011-03-31")
        )

        # pass a half-life value that is greater than the lookback period. to see errors
        # this case should not raise an error ('ma' unaffected by half-life).
        self.assertTrue(
            isinstance(
                historic_vol(
                    self.dfd,
                    "XR",
                    self.cids,
                    lback_periods=7,
                    lback_meth="ma",
                    half_life=11,
                    start=None,
                    end=None,
                    blacklist=None,
                    est_freq="m",
                    remove_zeros=True,
                    postfix="ASD",
                ),
                pd.DataFrame,
            )
        )

        # same args but with 'xma' method should raise an error.
        with self.assertRaises(AssertionError):
            historic_vol(
                self.dfd,
                "XR",
                self.cids,
                lback_periods=7,
                lback_meth="xma",
                half_life=11,
                start=None,
                end=None,
                blacklist=None,
                remove_zeros=True,
                postfix="ASD",
            )

        # this should raise an error as the lbakc_meth is invalid.
        with self.assertRaises(AssertionError):
            historic_vol(
                self.dfd,
                "CRY",
                self.cids,
                lback_periods=7,
                lback_meth="ema",
                half_life=11,
                start=None,
                end=None,
                blacklist=None,
                remove_zeros=True,
                postfix="ASD",
            )

        # Todo: check correct exponential averages for a whole series on toy data set using (.rolling) and .ewm


if __name__ == "__main__":
    unittest.main()
