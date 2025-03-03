import unittest
import warnings
from tests.simulate import make_qdf
from macrosynergy.panel.make_zn_scores import (
    _get_expanding_count,
    make_zn_scores,
    expanding_stat,
)
from macrosynergy.management.utils import _map_to_business_day_frequency
import pandas as pd
import numpy as np
from itertools import groupby
from typing import List, Dict, Callable
from parameterized import parameterized


class TestAll(unittest.TestCase):
    def setUp(self) -> None:
        self.cids: List[str] = ["AUD", "CAD", "GBP"]
        self.xcats: List[str] = ["CRY", "XR"]

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

        # Standard df for tests.
        dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)
        self.dfd: pd.DataFrame = dfd[dfd["xcat"] == "CRY"]
        self.dfw: pd.DataFrame = self.dfd.pivot(
            index="real_date", columns="cid", values="value"
        )

        daily_dates = pd.date_range(start="2010-01-01", end="2020-10-30", freq="B")
        self.dates_iter: pd.DatetimeIndex = daily_dates
        self.func_dict: Dict[str, Callable] = {"mean": np.mean, "median": np.median}

    def tearDown(self) -> None:
        return super().tearDown()

    def in_sampling(self, dfw: pd.DataFrame, neutral: str, min_obs: int) -> float:
        """
        Used to test the application of pandas in-built back-fill mechanism.
        """

        # Convert to a one-dimensional DataFrame to facilitate pd.apply() method
        # to calculate in-sampling period. The pd.stack() feature removes the
        # unrealised cross-sections.
        iis_period = pd.DataFrame(dfw.iloc[0 : (min_obs + 1)].stack().to_numpy())
        iis_val = iis_period.apply(self.func_dict[neutral])

        return round(float(iis_val.iloc[0]), 4)

    def test_pan_neutral(self):
        # Panel weight is equated to one, so pass in the entire DataFrame to
        # expanding_stat() method.
        # The default frequency for the iterative dates data structure is daily.

        df_neutral = expanding_stat(
            df=self.dfw, dates_iter=self.dates_iter, stat="mean", sequential=True
        )
        self.assertIsInstance(df_neutral, pd.DataFrame)
        # Test length of neutral DataFrame.
        self.assertTrue(self.dfw.shape[0] == df_neutral.shape[0])

        # --- Sequential equal False, Mean & Median: all in-sample.

        df_neutral = expanding_stat(
            self.dfw, dates_iter=self.dates_iter, stat="mean", sequential=False
        )
        # Check first value equal to panel mean.
        self.assertEqual(df_neutral.iloc[0].values[0], self.dfw.stack().mean())

        # Check also last value equal to panel mean.
        last_val: np.float64 = df_neutral.iloc[self.dfw.shape[0] - 1].values[0]
        self.assertEqual(last_val, self.dfw.stack().mean())

        df_neutral = expanding_stat(
            self.dfw, dates_iter=self.dates_iter, stat="median", sequential=False
        )
        self.assertEqual(df_neutral.iloc[0].values[0], self.dfw.stack().median())

        last_val = float(df_neutral.iloc[self.dfw.shape[0] - 1].values[0])
        self.assertEqual(last_val, self.dfw.stack().median())

        # --- Sequential equal True, iis = False.

        # Choose a random index, 999, and confirm the computed value.
        df_neutral = expanding_stat(
            self.dfw,
            dates_iter=self.dates_iter,
            stat="mean",
            sequential=True,
            iis=False,
        )
        val = round(float(df_neutral.iloc[999].values[0]), 4)
        benchmark = self.dfw.iloc[0:1000, :].stack().mean()
        self.assertEqual(val, round(benchmark, 4))

        # Again, test on a random index, 999.
        df_neutral = expanding_stat(
            self.dfw,
            dates_iter=self.dates_iter,
            stat="median",
            sequential=True,
            min_obs=261,
            iis=False,
        )
        val = float(df_neutral.iloc[999].values[0])
        median_benchmark = self.dfw.iloc[0:1000, :].stack().median()
        self.assertEqual(val, median_benchmark)

        # --- iis = True. Confirm in-sampling method.

        # Check the inclusion of the in-sampling data being included in the returned
        # DataFrame. The first minimum number observations, for the neutral level, will
        # all be the same value.
        df_neutral = expanding_stat(
            self.dfw,
            dates_iter=self.dates_iter,
            stat="mean",
            sequential=True,
            min_obs=261,
            iis=True,
        )
        self.assertTrue(all(df_neutral.iloc[:261] == df_neutral.iloc[0]))

        test_val = self.in_sampling(dfw=self.dfw, neutral="mean", min_obs=261)
        test_data = df_neutral.iloc[:261].to_numpy().reshape(261)

        bm_vals = [round(v, 4) for v in test_data]
        for v in bm_vals:
            self.assertTrue(abs(v - test_val) < 0.0001)

        # Check the above for the application of 'median' as the neutral level.
        # Unable to check for equality on np.nan values.
        df_neutral = expanding_stat(
            self.dfw,
            dates_iter=self.dates_iter,
            stat="median",
            sequential=True,
            min_obs=261,
            iis=True,
        )
        self.assertTrue(all(df_neutral.iloc[:261] == df_neutral.iloc[0]))

    @staticmethod
    def dates_iterator(df, est_freq):
        """
        Method used to produce the dates data structure.
        """

        s_date = min(df["real_date"])
        e_date = max(df["real_date"])
        dates_iter = pd.date_range(start=s_date, end=e_date, freq=est_freq)
        return dates_iter

    def test_downsampling(self):
        """
        Often there is little value added from computing the neutral level and standard
        deviation on a daily basis. The calculations are computationally intensive and
        the change over a daily period will, in most instances, be inconsequential. For
        the neutral level or standard deviation to have a significant change, especially
        as the days pass, the return series must have experienced a sustained period of
        inverted returns. A reversal in the underlying trend will normally manifest
        after a number of days have elapsed. Therefore, compute less frequently to
        reflect the above paradigm.
        """

        # --- Down-sampling pan-neutral, monthly.

        df = self.dfd
        df_copy = df.copy()

        df["real_date"] = pd.to_datetime(df["real_date"], errors="coerce")
        df["year"] = df["real_date"].dt.year
        # Test on monthly down-sampling to ensure the expanding window is still being
        # applied correctly but on a monthly basis. Each statistic, computed on the lower
        # frequency, will use all of the preceding days data to capture the underlying
        # trend.

        df["month"] = df["real_date"].dt.month
        dfw_multidx = df.pivot(
            index=["year", "month", "real_date"], columns="cid", values="value"
        )

        # Test on the 'mean' neutral level.
        test = []
        aggregate = np.empty(0)
        # Group the DataFrames monthly.
        for date, new_df in dfw_multidx.groupby(level=[0, 1]):
            new_arr = new_df.stack().to_numpy()
            aggregate = np.concatenate([aggregate, new_arr])
            test.append(np.mean(aggregate))

        dfw = df_copy.pivot(index="real_date", columns="cid", values="value")

        bfreq = _map_to_business_day_frequency("M")
        dates_iter = self.dates_iterator(df_copy, est_freq=bfreq)
        # Test against the existing solution.
        # The below method will return a one-dimensional DataFrame hosting the neutral
        # values produced from the expanding window. The DataFrame will be daily values
        # and, if down-sampling has been applied, the intermediary dates between
        # re-estimation dates will be populated by forward filling technique.
        # Therefore, the number of unique neutral values will correspond to the number of
        # re-estimation dates.
        df_mean = expanding_stat(
            df=dfw,
            dates_iter=dates_iter,
            stat="mean",
            sequential=True,
            min_obs=0,
            iis=False,
        )
        # Remove any NaN values when validating the logic. Exclusively check the values
        # computed on each re-estimation date.
        df_mean.dropna(axis=0, how="all", inplace=True)

        bm_values = df_mean["value"].to_numpy()
        # Avoid using a Set which orders the data. The itertools.groupby() method makes
        # an iterator that returns consecutive keys and groups from the iterable.
        bm_values = [k for k, g in groupby(bm_values)]

        # Assert there are the same number of re-estimation dates.
        self.assertTrue(len(bm_values) == len(test))

        # Confirm the expanding neutral calculations are correct.
        for i, bm_val in enumerate(bm_values):
            condition = abs(bm_val - test[i]) < 0.001
            self.assertTrue(condition)

        # The above logic also concurrently tests computing the standard deviation on a
        # down-sampled series. Uses the same methodology.

    @staticmethod
    def valid_index(column):
        """
        Returns the index of the first realised value in the series.
        """

        index = column.index
        date = column.first_valid_index()
        date_index = next(iter(np.where(index == date)[0]))

        return date_index

    @staticmethod
    def handle_nan(arr):
        arr = np.nan_to_num(arr)
        arr = arr[arr != 0.0]

        return arr

    def cross_neutral(self, stat, sequential, iis):
        """
        Auxiliary method used to create the cross-neutral DataFrame using
        make_zn_scores.py functionality. Recreates the workflow from the aforementioned
        file which will be examined below.

        :param <str> stat: statistical method to be applied. This is typically 'mean' or
            'median'.
        :param <bool> sequential: neutral level and standard deviation are estimated
            sequentially.
        :param <bool> iis: in-sampling.
        """

        dfw_zns_css = self.dfw * 0
        for i, cid in enumerate(self.cids):
            # pd.core.DataFrame.
            dfi = self.dfw.loc[:, cid]
            dfi = pd.DataFrame(data=dfi.to_numpy(), index=dfi.index, columns=[cid])
            df_neutral = expanding_stat(
                dfi,
                dates_iter=self.dates_iter,
                stat=stat,
                sequential=sequential,
                iis=iis,
            )
            dfw_zns_css.loc[df_neutral.index, cid] = df_neutral["value"]

        return dfw_zns_css

    def test_cross_neutral(self):
        df_neutral = self.cross_neutral("mean", False, iis=False)
        self.assertIsInstance(df_neutral, pd.DataFrame)

        df_shape = self.dfw.shape
        self.assertEqual(df_shape, df_neutral.shape)

        epsilon = 0.0001
        # --- Cross-neutral mean, sequential equal False.

        df_mean = self.cross_neutral(stat="mean", sequential=False, iis=False)
        # Arbitrarily chosen index to test the logic.
        index = 411
        # Testing the cross-neutral level if sequential equals False. The entire series
        # will have a single neutral level (all handled as in-sample). Therefore, isolate
        # the column and calculate the mean adjusting for NaN values.
        for i, cross in enumerate(self.cids):
            mean = np.nanmean(self.dfw[cross].to_numpy())
            mean_col = df_mean.loc[:, cross]

            # Will exclude NaN values from the calculation.
            dif = np.abs(mean_col - mean)
            # Test if function mean is correct.
            self.assertTrue(np.nan_to_num(dif.iloc[index]) < epsilon)

        # --- Cross-neutral median, sequential equal False.

        df_median = self.cross_neutral(stat="median", sequential=False, iis=False)
        # Again, same logic as above. Test the cross-sectional median value when the
        # sequential parameter is set to False.
        for i, cross in enumerate(self.cids):
            median = np.nanmedian(self.dfw[cross].to_numpy())

            median_cross = df_median.loc[:, cross]
            median_cross.dropna(axis=0, how="any", inplace=True)
            median_value = median_cross.unique()
            self.assertTrue(len(median_value) == 1)

            # Choose a random index to confirm the value.
            value = float(median_cross.iloc[1000])
            dif = np.abs(median - value)
            # Test if function median is correct.
            self.assertTrue(dif < epsilon)

        # --- Cross-neutral mean, sequential equal True. Use pandas in-built
        # pd.expanding() method to validate the expanding window.

        min_obs = 261
        df_mean = self.cross_neutral(stat="mean", sequential=True, iis=False)
        cross_sections = list(self.dfw.columns)
        # Test the cross-sectional expanding mean calculated on a daily basis. To
        # validate the logic used in make_zn_scores() use pd.expanding().mean() on each
        # individual series. The rolling neutral level in make_zn_scores() is calculated
        # through an iterative loop.
        for i, cid in enumerate(cross_sections):
            column_mean = df_mean.loc[:, cid].to_numpy()
            column_mean = self.handle_nan(column_mean)

            column = self.dfw.loc[:, cid]
            cum_mean = column.expanding(min_periods=(min_obs + 1)).mean()
            cum_mean = cum_mean.to_numpy()
            test_arr = self.handle_nan(cum_mean)

            dif = column_mean - test_arr
            # Check correct cumulative means.
            self.assertTrue(np.nan_to_num(dif[index]) < epsilon)

    def test_cross_down_sampling(self):
        """
        Neutral level is computed on a cross-sectional basis.
        """

        df = self.dfd
        dfw = self.dfw
        # Isolate an individual cross-section's return series.
        cross_series = dfw["AUD"]
        cross_series = pd.DataFrame(
            data=cross_series.to_numpy(), index=dfw.index, columns=["AUD"]
        )
        date_index = self.valid_index(column=cross_series)

        # Test on quarterly data.
        bfreq = _map_to_business_day_frequency("Q")
        dates_iter = self.dates_iterator(df, est_freq=bfreq)
        neutral_df = expanding_stat(
            df=cross_series,
            dates_iter=dates_iter,
            stat="mean",
            sequential=True,
            min_obs=261,
        )

        # Choose a random re-estimation date and confirm the corresponding re-estimated
        # value is equivalent to in-sampling up to the respective date.
        random_index = len(dates_iter) // 2
        random_date = dates_iter[random_index]
        test_series = cross_series.loc[:random_date]
        test_value = np.mean(test_series.to_numpy())

        benchmark_value = float(neutral_df.loc[random_date].iloc[0])
        self.assertTrue(np.isclose(test_value, benchmark_value, atol=0.001))

        # Confirm the dates, over the next quarter, are the same as the value referenced
        # above.
        next_index = random_index + 1
        next_date_quarter = dates_iter[next_index]
        benchmark_quarter = neutral_df.loc[random_date:next_date_quarter].to_numpy()
        benchmark_quarter = benchmark_quarter.reshape(len(benchmark_quarter))

        # Exclude the next re-estimation date where the neutral level changes.
        for bm_elem in benchmark_quarter[:-1]:
            self.assertTrue(np.abs(test_value - bm_elem) < 0.001)

    def test_zn_scores(self):
        with self.assertRaises(ValueError):
            # Test catching neutral value error.
            df = make_zn_scores(
                self.dfd,
                "XR",
                self.cids,
                sequential=False,
                neutral="std",
                thresh=1.5,
                postfix="ZN",
            )
        with self.assertRaises(ValueError):
            # Test catching non-valid thresh value.
            df = make_zn_scores(
                self.dfd,
                "XR",
                self.cids,
                sequential=False,
                neutral="mean",
                thresh=0.5,
                pan_weight=1.0,
                postfix="ZN",
            )

        with self.assertRaises(ValueError):
            # Test catching panel weight.
            df = make_zn_scores(
                self.dfd, "XR", self.cids, sequential=False, pan_weight=1.2
            )

        with self.assertRaises(TypeError):
            # Test the iis parameter being a boolean value.
            df = make_zn_scores(
                self.dfd, "XR", self.cids, sequential=False, pan_weight=0.2, iis=0
            )

        with self.assertRaises(TypeError):
            # Test the minimum observation parameter (non-negative Integer value).
            df = make_zn_scores(
                self.dfd,
                "XR",
                self.cids,
                sequential=True,
                pan_weight=0.3,
                min_obs=-1,
                iis=0,
            )

        # Testing on Panel = 1.0 (default value)
        df_panel = make_zn_scores(
            self.dfd,
            "CRY",
            self.cids,
            sequential=True,
            min_obs=0,
            iis=False,
            neutral="mean",
            thresh=None,
            postfix="ZN",
        )

        df_panel = df_panel.pivot(index="real_date", columns="cid", values="value")
        df_neutral = expanding_stat(
            self.dfw,
            dates_iter=self.dates_iter,
            stat="mean",
            sequential=True,
            min_obs=0,
            iis=False,
        )

        dfx = self.dfw.sub(df_neutral["value"], axis="rows")

        ar_sds = np.array(
            [dfx.iloc[0 : (i + 1), :].stack().abs().mean() for i in range(dfx.shape[0])]
        )
        dfw_zns_pan = dfx.div(ar_sds, axis="rows")
        dfw_zns_pan = dfw_zns_pan.dropna(axis=0, how="all")

        # Check the zn_scores, across a panel, on a specific date. Discount the
        # internal randomness.
        no_rows = dfw_zns_pan.shape[0]
        index = int(no_rows / 2)

        zn_scores = df_panel.to_numpy()
        arr_zns_pan = dfw_zns_pan.to_numpy()
        # Confirm the values on a random index.
        dif = zn_scores[index] - arr_zns_pan[index]

        epsilon = 0.000001
        self.assertTrue(np.all(np.nan_to_num(dif) < epsilon))

        dfd = self.dfd
        # Test weighting function.
        panel_df = make_zn_scores(
            dfd,
            "CRY",
            self.cids,
            start="2010-01-04",
            sequential=False,
            min_obs=0,
            neutral="mean",
            iis=True,
            thresh=None,
            pan_weight=0.75,
            postfix="ZN",
        )
        df_cross = make_zn_scores(
            dfd,
            "CRY",
            self.cids,
            start="2010-01-04",
            sequential=False,
            min_obs=0,
            neutral="mean",
            iis=True,
            thresh=None,
            pan_weight=0.25,
            postfix="ZN",
        )

        df_average = make_zn_scores(
            dfd,
            "CRY",
            self.cids,
            start="2010-01-04",
            sequential=False,
            min_obs=0,
            iis=True,
            neutral="mean",
            thresh=None,
            pan_weight=0.5,
            postfix="ZN",
        )

        panel_df = panel_df.pivot(index="real_date", columns="cid", values="value")
        df_cross = df_cross.pivot(index="real_date", columns="cid", values="value")
        df_average = df_average.pivot(index="real_date", columns="cid", values="value")

        # Drop the first row in the panel data.
        panel_df = panel_df.drop(panel_df.index[[0]])
        df_check = (panel_df + df_cross) / 2
        check_arr = df_check.to_numpy()
        average_arr = df_average.to_numpy()

        # Again, validate on a randomly chosen index.
        index = 1212
        dif = check_arr[index] - average_arr[index]
        self.assertTrue(np.sum(dif) < epsilon)

        # Test the usage of the threshold parameter.
        threshold = 2.35
        df_thresh = make_zn_scores(
            self.dfd,
            "CRY",
            self.cids,
            start="2010-01-01",
            sequential=True,
            min_obs=252,
            neutral="mean",
            thresh=threshold,
            pan_weight=0.65,
            postfix="ZN",
        )

        df_thresh = df_thresh.pivot(index="real_date", columns="cid", values="value")
        thresh_arr = df_thresh.to_numpy()
        # Compress multidimensional array into a one-dimensional array.
        values = thresh_arr.ravel()
        values = values.astype(dtype=np.float64)

        check = sum(values[~np.isnan(values)] > threshold)

        self.assertTrue(check == 0)

    def test_zn_scores_warning(self):
        with self.assertWarns(UserWarning):
            warnings.simplefilter("always")
            r_cid: str = self.cids[0]
            r_xcat: str = "CRY"
            self.dfd.loc[
                (self.dfd["cid"] == r_cid) & (self.dfd["xcat"] == r_xcat), "value"
            ] = pd.NA
            dfr: pd.DataFrame = make_zn_scores(
                df=self.dfd,
                xcat=r_xcat,
                cids=self.cids,
                start="2010-01-01",
            )
            dfr["xcat"] = dfr["xcat"].str.replace("ZN", "")
            self.assertEqual(dfr["xcat"].unique()[0], r_xcat)
            self.assertFalse(r_cid in dfr["cid"].unique())
            warnings.resetwarnings()

        with self.assertRaises(ValueError):
            try:
                test_cids: List[str] = self.cids.copy()
                test_cids.append("unknown_cid")
                r_xcat: str = "CRY"
                dfr: pd.DataFrame = make_zn_scores(
                    df=self.dfd,
                    xcat=r_xcat,
                    cids=test_cids,
                    start="2010-01-01",
                )
            except ValueError as e:
                self.assertTrue("unknown_cid" in str(e))
                raise ValueError(e)

        with self.assertRaises(ValueError):
            try:
                r_xcat: str = "unknown_xcat"
                dfr: pd.DataFrame = make_zn_scores(
                    df=self.dfd,
                    xcat=r_xcat,
                    cids=self.cids,
                    start="2010-01-01",
                )
            except ValueError as e:
                self.assertTrue("unknown_xcat" in str(e))
                raise ValueError(e)

    def test_get_expanding_count(self):
        data = {
            "AUD": [float("nan"), float("nan"), float("nan"), float("nan")],
            "CAD": [0.350685, 0.983600, -0.636405, 0.526873],
            "GBP": [0.099416, 0.329336, -0.491647, -0.494627],
            "NZD": [-0.253184, 0.201844, 3.495799, 2.372900],
            "USD": [-0.179115, -0.381266, -0.038212, -1.034298],
        }
        index = ["2008-01-01", "2008-01-02", "2008-01-03", "2008-01-04"]

        df = pd.DataFrame(data, index=index)
        df.index.name = "real_date"

        expanding_count = _get_expanding_count(df, 1)
        self.assertTrue(np.allclose(expanding_count, np.array([4, 8, 12, 16])))

        expanding_count = _get_expanding_count(df, 2)
        self.assertTrue(np.allclose(expanding_count, np.array([0, 8, 12, 16])))

        expanding_count = _get_expanding_count(df, 3)
        self.assertTrue(np.allclose(expanding_count, np.array([0, 0, 12, 16])))

        expanding_count = _get_expanding_count(df, 4)
        self.assertTrue(np.allclose(expanding_count, np.array([0, 0, 0, 16])))

    def test_get_expanding_count_nans(self):
        n_rows = np.random.randint(10, 100)
        n_cols = np.random.randint(10, 100)
        data = np.random.randn(n_rows, n_cols)

        df = pd.DataFrame(data)

        expanding_count = _get_expanding_count(df, 1)
        self.assertTrue(np.isclose(expanding_count[-1], n_rows * n_cols))

    @parameterized.expand([0.0, 0.5])
    def test_unscore(self, pan_weight):

        df_panel = make_zn_scores(
            self.dfd,
            "CRY",
            self.cids,
            sequential=True,
            min_obs=0,
            iis=False,
            neutral="mean",
            thresh=None,
            postfix="ZN",
            pan_weight=pan_weight,
            unscore=True,
        )
        df_panel = df_panel.sort_values(by=["cid", "xcat", "real_date"])

        expected_values = (
            self.dfd.sort_values(by=["cid", "xcat", "real_date"])
            .groupby(["cid", "xcat"], group_keys=False)
            .apply(lambda g: g.iloc[1:])
        )

        np.testing.assert_array_almost_equal(
            df_panel["value"].values,
            expected_values.sort_values(by=["cid", "xcat", "real_date"])[
                "value"
            ].values,
            decimal=4,
        )

    def test_unscore_panel(self):
        pan_weight = 1
        df_panel = make_zn_scores(
            self.dfd,
            "CRY",
            self.cids,
            sequential=True,
            min_obs=0,
            iis=False,
            neutral="mean",
            postfix="ZN",
            pan_weight=pan_weight,
            unscore=True,
        )
        df_panel = df_panel.sort_values(by=["cid", "xcat", "real_date"])
        np.testing.assert_array_almost_equal(
            df_panel["value"].values,
            self.dfd.sort_values(by=["cid", "xcat", "real_date"])["value"].values,
            decimal=4,
        )

    @parameterized.expand([1, 2, 3])
    def test_winsorization_unscored(self, thresh):
        dates_iter = pd.date_range("2020-01-01", periods=5, freq="B")

        # Example data
        data = {
            "cid": ["A", "A", "A", "A", "A", "B", "B", "B", "B", "B"],
            "xcat": ["CRY"] * 10,
            "real_date": list(dates_iter) * 2,
            "value": [10, 12, 30, 14, 15, 12, 10, 30, 80, 10],
        }
        df = pd.DataFrame(data)

        df_panel = make_zn_scores(
            df,
            "CRY",
            ["A", "B"],
            sequential=True,
            min_obs=0,
            iis=False,
            neutral="mean",
            thresh=thresh,
            postfix="ZN",
            pan_weight=1,
            unscore=True,
        )

        self.assertEqual(df_panel.value.median(), df.value.median())


if __name__ == "__main__":
    unittest.main()
