import unittest
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from unittest.mock import patch
from tests.simulate import make_qdf
from macrosynergy.panel.correlation import (
    _get_dates,
    _handle_secondary_args,
    corr,
    lag_series,
    _transform_df_for_cross_sectional_corr,
    _transform_df_for_cross_category_corr,
)

from macrosynergy.management.utils import reduce_df


class TestAll(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        # Prevents plots from being displayed during tests.
        plt.close("all")
        self.mpl_backend: str = matplotlib.get_backend()
        self.mock_show = patch("matplotlib.pyplot.show").start()

    @classmethod
    def tearDownClass(self) -> None:
        patch.stopall()
        plt.close("all")
        patch.stopall()
        matplotlib.use(self.mpl_backend)

    def setUp(self) -> None:
        self.cids = ["AUD", "CAD", "GBP"]
        self.xcats = ["XR", "CRY", "GROWTH", "INFL"]
        self.val = "value"
        self.freq = "D"

        df_cids = pd.DataFrame(
            index=self.cids, columns=["earliest", "latest", "mean_add", "sd_mult"]
        )
        df_cids.loc["AUD", :] = ["2010-01-01", "2020-12-31", 0.5, 2]
        df_cids.loc["CAD", :] = ["2010-01-01", "2020-11-30", 0, 1]
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
        df_xcats.loc["CRY", :] = ["2010-01-01", "2020-10-30", 1, 2, 0.9, 0.5]
        df_xcats.loc["XR", :] = ["2011-01-01", "2020-12-31", 0, 1, 0, 0.3]
        df_xcats.loc["GROWTH", :] = ["2010-01-01", "2020-10-30", 1, 2, 0.9, 0.5]
        df_xcats.loc["INFL", :] = ["2010-01-01", "2020-10-30", 0, 2, 0.9, 0.5]

        # Standard df for tests.
        dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)
        self.dfd = dfd

    def tearDown(self) -> None:
        return super().tearDown()
    
    def test_cross_category_corr(self):
        corr_df = corr(
            self.dfd,
            xcats=self.xcats,
        )
        self.assertEqual(corr_df.shape, (len(self.xcats), len(self.xcats)))
        self.assertEqual(corr_df.columns.to_list(), self.xcats)
        self.assertEqual(corr_df.index.to_list(), self.xcats)

    def test_cross_sectional_corr(self):
        corr_df = corr(
            self.dfd,
            xcats=["XR"],
            cids=self.cids,
        )
        self.assertEqual(corr_df.shape, (len(self.cids), len(self.cids)))
        self.assertEqual(corr_df.columns.to_list(), self.cids)
        self.assertEqual(corr_df.index.to_list(), self.cids)

    def test_cross_category_corr_with_xcats_secondary(self):
        xcats_secondary = ["XR"]
        corr_df = corr(
            self.dfd,
            xcats=self.xcats,
            xcats_secondary=xcats_secondary,
        )
        self.assertEqual(corr_df.shape, (len(xcats_secondary), len(self.xcats)))
        self.assertEqual(corr_df.columns.to_list(), self.xcats)
        self.assertEqual(corr_df.index.to_list(), xcats_secondary)

    def test_cross_sectional_corr_with_cids_secondary(self):
        cids_secondary = ["AUD"]
        corr_df = corr(
            self.dfd,
            xcats=["XR"],
            cids=self.cids,
            cids_secondary=cids_secondary,
        )
        self.assertEqual(corr_df.shape, (len(cids_secondary), len(self.cids)))
        self.assertEqual(corr_df.columns.to_list(), self.cids)
        self.assertEqual(corr_df.index.to_list(), cids_secondary)

    def test_cross_category_corr_with_both_secondary(self):
        xcats_secondary = ["XR"]
        corr_df = corr(
            self.dfd,
            xcats=self.xcats,
            xcats_secondary=xcats_secondary,
            cids=self.cids,
            cids_secondary=self.cids[:2],
        )
        self.assertEqual(corr_df.shape, (len(xcats_secondary), len(self.xcats)))
        self.assertEqual(corr_df.columns.to_list(), self.xcats)
        self.assertEqual(corr_df.index.to_list(), xcats_secondary)

    def test_cross_sectional_corr_with_both_secondary(self):
        cids_secondary = ["AUD"]
        corr_df = corr(
            self.dfd,
            xcats=["XR"],
            xcats_secondary=["CRY"],
            cids=self.cids,
            cids_secondary=cids_secondary,
        )
        self.assertEqual(corr_df.shape, (len(cids_secondary), len(self.cids)))
        self.assertEqual(corr_df.columns.to_list(), self.cids)
        self.assertEqual(corr_df.index.to_list(), cids_secondary)

    def test_lag_series(self):
        """
        Test the method used to lag the categories included
        """

        df_w = self.dfd.pivot(
            index=("cid", "real_date"), columns="xcat", values="value"
        )

        # Confirm the application of the lag to the respective category is correct.
        # Compare against the original DataFrame.

        # Lag inflation by a range of possible options.
        lag_dict = {"INFL": [0, 2, 5]}
        # Returns a multi-index DataFrame. Therefore, ensure the lag has been applied
        # correctly on each individual cross-section.
        df_w, xcat_tracker = lag_series(df_w, lag_dict, xcats=self.xcats)

        # Firstly, confirm the DataFrame includes the expected columns: incumbent
        # categories & additional categories that have had the lag postfix appended.
        test_columns = list(df_w.columns)
        test_columns_set = set(test_columns)
        # The inflation category will be removed from the wide DataFrame as it is being
        # replaced by a lagged version.
        xcats_copy = self.xcats.copy()
        xcats_copy.remove("INFL")
        self.assertTrue(set(xcats_copy).issubset(test_columns_set))

        # Remove the incumbent categories and confirm that the residual categories in
        # the DataFrame are the respective lagged inflation series: ['INFL_L0',
        # 'INFL_L2', 'INFL_L5'].

        update_test_columns = [xcat for xcat in test_columns if xcat not in xcats_copy]

        update_test_columns = sorted(update_test_columns)
        self.assertTrue(update_test_columns == ["INFL_L0", "INFL_L2", "INFL_L5"])

        # Confirm the lag mechanism works correctly. The DataFrame will be a multi-index
        # DataFrame, and subsequently it is important to confirm that the lag works
        # correctly on an individual ticker level.
        df_w_lag = df_w[["INFL_L0", "INFL_L2", "INFL_L5"]]

        # The benchmark value will be "INFL_L0" where a lag has not been applied. Test on
        # two cross-sections: ['AUD', 'GBP'].
        df_w_lag = df_w_lag.loc[["AUD", "GBP"], :]

        # Arbitrary date to test the logic.
        fixed_date = "2013-03-11"
        condition = df_w_lag.index.get_level_values("real_date") == fixed_date
        period_t = df_w_lag[condition]["INFL_L0"]
        period_t_aud = float(period_t.loc["AUD"].iloc[0])

        lagged_date_2 = pd.Timestamp(fixed_date) + pd.DateOffset(2)
        condition_2 = df_w_lag.index.get_level_values("real_date") == lagged_date_2
        period_t_2 = df_w_lag[condition_2]["INFL_L2"]
        period_t_2_aud = float(period_t_2.loc["AUD"].iloc[0])

        # Confirm that the value at '2013-03-11' has been lagged correctly in INFL_L2 on
        # AUD: the same value should be held at '2013-03-13'.
        self.assertEqual(period_t_aud, period_t_2_aud)

        # Apply the same logic above but for INFL_L2.
        dates_index = list(df_w_lag.loc["AUD", :].index)
        dates_index = list(map(lambda d: str(d).split(" ")[0], dates_index))

        fixed_date_index = dates_index.index(fixed_date)
        lagged_date_5_index = fixed_date_index + 5

        lagged_date_5 = dates_index[lagged_date_5_index]
        condition_5 = df_w_lag.index.get_level_values("real_date") == lagged_date_5

        period_t_5 = df_w_lag[condition_5]["INFL_L5"]
        period_t_5_aud = float(period_t_5.loc["AUD"].iloc[0])
        self.assertEqual(period_t_aud, period_t_5_aud)

        period_t_gbp = float(period_t.loc["GBP"].iloc[0])
        period_t_2_gbp = float(period_t_2.loc["GBP"].iloc[0])

        # Confirm that the value at '2013-03-11' has been lagged correctly in INFL_L2 on
        # GBP: the same value should be held at '2013-03-13'.
        self.assertEqual(period_t_gbp, period_t_2_gbp)

        # INFL_L5 on GBP.
        period_t_5_gbp = float(period_t_5.loc["GBP"].values[0])
        self.assertEqual(period_t_gbp, period_t_5_gbp)

        # Lag inflation & Growth by a range of possible options.
        lag_dict = {"INFL": [0, 5], "GROWTH": [3]}
        df_w = self.dfd.pivot(
            index=("cid", "real_date"), columns="xcat", values="value"
        )

        df_w, xcat_tracker = lag_series(df_w, lag_dict, xcats=self.xcats)
        test_columns = list(df_w.columns)

        lagged_columns = [xcat for xcat in test_columns if xcat not in self.xcats]
        self.assertTrue(sorted(lagged_columns) == ["GROWTH_L3", "INFL_L0", "INFL_L5"])

    def test_corr(self):
        try:
            corr(self.dfd, xcats=["XR"], cids=self.cids)
        except Exception as e:
            self.fail(f"corr raised {e} unexpectedly")

        try:
            corr(
                self.dfd,
                xcats=["XR"],
                xcats_secondary=["CRY"],
                cids=self.cids,
            )
        except Exception as e:
            self.fail(f"corr raised {e} unexpectedly")

        try:
            corr(
                self.dfd,
                xcats=["XR"],
                xcats_secondary=["CRY"],
                cids=["AUD"],
                cids_secondary=["GBP"],
            )
        except Exception as e:
            self.fail(f"corr raised {e} unexpectedly")

        try:
            corr(
                self.dfd,
                xcats=["XR"],
                cids=["AUD"],
                cids_secondary=["GBP"],
            )
        except Exception as e:
            self.fail(f"corr raised {e} unexpectedly")

        lag_dict = {"INFL": [0, 1, 2, 5]}
        with self.assertRaises(ValueError):
            # Test the frequency options: either ['W', 'M', 'Q'].
            corr(
                self.dfd,
                xcats=["XR", "CRY"],
                cids=self.cids,
                freq="BW",
                lags=lag_dict,
            )

        with self.assertRaises(AssertionError):
            # Test the max_color value. Expects a floating point value.
            corr(
                self.dfd,
                xcats=["XR", "CRY"],
                cids=self.cids,
                lags=lag_dict,
            )

        with self.assertRaises(AssertionError):
            # Test the received lag data structure. Dictionary expected.
            lag_list = [0, 60]
            corr(
                self.dfd,
                xcats=["XR", "CRY"],
                cids=self.cids,
                lags=lag_list,
            )

        with self.assertRaises(AssertionError):
            # Test that the lagged categories are present in the received DataFrame.
            # The category, GROWTH, is not in the received categories.
            lag_dict = {"GROWTH": [0, 1, 2, 5]}
            corr(
                self.dfd,
                xcats=["XR", "CRY"],
                cids=self.cids,
                lags=lag_dict,
            )

    def test_transform_df_for_cross_sectional_corr(self):
        df = reduce_df(self.dfd, xcats=["XR"], cids=self.cids)
        df_w = _transform_df_for_cross_sectional_corr(df, val="value")

        # Test that columns are now the cids.
        self.assertEqual(df_w.columns.to_list(), self.cids)

    def test_transform_df_for_cross_category_corr(self):
        xcats = ["XR", "CRY"]
        df = reduce_df(self.dfd, xcats=xcats, cids=self.cids)
        df_w = _transform_df_for_cross_category_corr(df, xcats=xcats, val="value")

        # Test that columns are now the xcats.
        self.assertEqual(df_w.columns.to_list(), xcats)

    def test_get_dates(self):
        df = reduce_df(self.dfd, xcats=["XR"])

        min_date = df["real_date"].min()
        max_date = df["real_date"].max()

        df_w = _transform_df_for_cross_sectional_corr(
            df=df, val=self.val, freq=self.freq
        )
        start_date, end_date = _get_dates(df_w)

        self.assertEqual(start_date, min_date)
        self.assertEqual(end_date, max_date)

    def test_get_dates_multiindexed(self):
        xcats = ["XR", "CRY"]
        df = reduce_df(self.dfd, xcats=xcats)

        min_date = df["real_date"].min()
        max_date = df["real_date"].max()

        df_w = _transform_df_for_cross_category_corr(
            df=df, xcats=xcats, val=self.val, freq=self.freq
        )
        start_date, end_date = _get_dates(df_w)

        self.assertEqual(start_date, min_date)
        self.assertEqual(end_date, max_date)

    def test_with_secondary_args(self):
        # Test with both secondary arguments provided
        xcats = ["xcat1", "xcat2"]
        cids = ["cid1", "cid2"]
        xcats_secondary = ["xcat3", "xcat4"]
        cids_secondary = ["cid3", "cid4"]
        result = _handle_secondary_args(xcats, cids, xcats_secondary, cids_secondary)
        self.assertEqual(result, (xcats_secondary, cids_secondary))

    def test_with_secondary_arg_as_string(self):
        # Test with secondary xcat argument provided as single strings
        xcats = ["xcat1", "xcat2"]
        cids = ["cid1", "cid2"]
        xcats_secondary = "xcat3"
        cids_secondary = ["cid3"]
        result = _handle_secondary_args(xcats, cids, xcats_secondary, cids_secondary)
        self.assertEqual(result, (["xcat3"], ["cid3"]))

    def test_without_secondary_xcats(self):
        # Test without secondary xcats provided
        xcats = ["xcat1", "xcat2"]
        cids = ["cid1", "cid2"]
        cids_secondary = ["cid3", "cid4"]
        result = _handle_secondary_args(xcats, cids, None, cids_secondary)
        self.assertEqual(result, (xcats, cids_secondary))

    def test_without_secondary_cids(self):
        # Test without secondary cids provided
        xcats = ["xcat1", "xcat2"]
        cids = ["cid1", "cid2"]
        xcats_secondary = ["xcat3", "xcat4"]
        result = _handle_secondary_args(xcats, cids, xcats_secondary, None)
        self.assertEqual(result, (xcats_secondary, cids))

    def test_without_any_secondary_args(self):
        # Test without any secondary arguments provided
        xcats = ["xcat1", "xcat2"]
        cids = ["cid1", "cid2"]
        result = _handle_secondary_args(xcats, cids, None, None)
        self.assertEqual(result, (xcats, cids))


if __name__ == "__main__":
    unittest.main()
