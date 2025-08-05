import unittest
import io
import sys
import matplotlib.pyplot
import numpy as np
import pandas as pd
import matplotlib
from random import randint
from tests.simulate import make_qdf
from macrosynergy.management.simulate import make_test_df
from macrosynergy.panel.category_relations import CategoryRelations
from macrosynergy.management.utils import categories_df
from typing import List, Tuple, Dict, Union, Optional
import warnings


class TestAll(unittest.TestCase):
    # Method used to construct the respective DataFrame.

    def setUp(self) -> None:
        self.cids: List[str] = ["AUD", "CAD", "GBP", "NZD", "JPY", "CHF"]
        self.xcats: List[str] = ["XR", "CRY", "GROWTH", "INFL"]

        df_cids = pd.DataFrame(
            index=self.cids, columns=["earliest", "latest", "mean_add", "sd_mult"]
        )
        df_cids.loc["AUD"] = ["2000-01-01", "2020-12-31", 0.1, 1]
        df_cids.loc["CAD"] = ["2001-01-01", "2020-11-30", 0, 1]
        df_cids.loc["GBP"] = ["2002-01-01", "2020-11-30", 0, 2]
        df_cids.loc["NZD"] = ["2002-01-01", "2020-09-30", -0.1, 2]
        df_cids.loc["JPY"] = ["2002-01-01", "2020-09-30", -0.3, 3]
        df_cids.loc["CHF"] = ["2002-01-01", "2020-09-30", 0.3, 1]

        cols = ["earliest", "latest", "mean_add", "sd_mult", "ar_coef", "back_coef"]
        df_xcats = pd.DataFrame(index=self.xcats, columns=cols)
        df_xcats.loc["XR"] = ["2000-01-01", "2020-12-31", 0.1, 1, 0, 0.3]
        df_xcats.loc["CRY"] = ["2000-01-01", "2020-10-30", 1, 2, 0.95, 1]
        df_xcats.loc["GROWTH"] = ["2001-01-01", "2020-10-30", 1, 2, 0.9, 1]
        df_xcats.loc["INFL"] = ["2001-01-01", "2020-10-30", 1, 2, 0.8, 0.5]

        dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)

        self.dfd: pd.DataFrame = dfd
        self.black: Dict[str, List[str]] = {
            "AUD": ["2000-01-01", "2003-12-31"],
            "GBP": ["2018-01-01", "2100-01-01"],
        }

        # Filter the DataFrame to test the Set Theory logic.
        # The category, Growth, will not be defined for the cross-section 'AUD'.
        # The purpose is to return the shared cross-sections across specific categories.
        # Therefore, if the category Growth is passed, 'AUD' will be excluded from the
        # returned list of cross-sections.
        # Same logic applies to 'INFL' and 'NZD'.
        filt1 = (dfd["xcat"] == "GROWTH") & (dfd["cid"] == "AUD")
        filt2 = (dfd["xcat"] == "INFL") & (dfd["cid"] == "NZD")
        filt3 = (dfd["xcat"] == "INFL") & (dfd["cid"] == "JPY")
        # Reduced dataframe.

        self.filter_tuple: Tuple[pd.Series, pd.Series, pd.Series] = (
            filt1 | filt2 | filt3
        )
        dfdx = dfd[~self.filter_tuple]

        self.dfdx: pd.DataFrame = dfdx

        # Define a List of the cross-sections that one is interested in modelling. The
        # dataframe might potentially be defined on a greater number of cross-sections.
        # The List, 'cidx', should be a subset or incorporate all cross-sections defined
        # in the dataframe.
        # The above stipulation on 'cidx' will require being validated in the defined
        # functionality.
        self.cidx: List[str] = ["AUD", "CAD", "GBP"]

        matplotlib.pyplot.close("all")
        self.curr_backend = matplotlib.get_backend()
        matplotlib.use("Agg")

    def tearDown(self) -> None:
        matplotlib.pyplot.close("all")
        matplotlib.use(self.curr_backend)
        return super().tearDown()

    def test_constructor(self):
        # Testing the various assert statements built into the Class's Constructor.

        with self.assertRaises(ValueError):
            # Test the notion that the metric of interest is present in the DataFrame. If
            # not, an assertion will be thrown.
            cr = CategoryRelations(
                self.dfdx,
                xcats=["GROWTH", "INFL"],
                cids=self.cidx,
                freq="d",
                xcat_aggs=["mean", "mean"],
                lag=1,
                start="2000-01-01",
                years=None,
                blacklist=self.black,
                val="grading",
            )

        with self.assertRaises(ValueError):
            # Test the restrictions placed on the frequency parameter.
            cr = CategoryRelations(
                self.dfdx,
                xcats=["GROWTH", "INFL"],
                cids=self.cidx,
                freq="r",
                xcat_aggs=["mean", "mean"],
                lag=1,
                start="2000-01-01",
                years=None,
                blacklist=self.black,
            )

        with self.assertRaises(ValueError):
            # Test the notion that the category List can only receive two categories.
            cr = CategoryRelations(
                self.dfdx,
                xcats=["GROWTH", "INFL", "XR"],
                cids=self.cidx,
                freq="M",
                xcat_aggs=["mean", "mean"],
                lag=1,
                start="2000-01-01",
                years=None,
                blacklist=self.black,
            )

        with self.assertRaises(AssertionError):
            # Test the parameter 'changes' - only able to receive two string values:
            # i) 'diff'; ii) 'pch'.
            cr = CategoryRelations(
                self.dfdx,
                xcats=["GROWTH", "INFL"],
                cids=self.cidx,
                freq="M",
                xcat_aggs=["mean", "mean"],
                lag=1,
                start="2000-01-01",
                years=None,
                blacklist=self.black,
                xcat1_chg="pchg",
                n_periods=1,
            )

        with self.assertRaises(AssertionError):
            # If the 'changes' parameter is not set to None, the number of periods,
            # n_periods, in which the differencing or percentage change is computed must
            # be specified and be an Integer value.
            cr = CategoryRelations(
                self.dfdx,
                xcats=["GROWTH", "INFL"],
                cids=self.cidx,
                freq="M",
                xcat_aggs=["mean", "mean"],
                lag=1,
                start="2000-01-01",
                years=None,
                blacklist=self.black,
                xcat1_chg="pch",
                n_periods=None,
            )

        with self.assertRaises(AssertionError):
            # Trivial check to confirm the length of "xcat_trims" parameter.
            cr = CategoryRelations(
                self.dfdx,
                xcats=["GROWTH", "INFL"],
                cids=self.cidx,
                freq="M",
                xcat_aggs=["mean", "mean"],
                lag=1,
                start="2000-01-01",
                years=None,
                blacklist=self.black,
                xcat1_chg=None,
                xcat_trims=[3.25, 3.0, 2.0],
            )

        with self.assertRaises(TypeError):
            # Test freq is of type string.
            cr = CategoryRelations(
                self.dfdx,
                xcats=["GROWTH", "INFL"],
                cids=self.cidx,
                freq=1,
                xcat_aggs=["mean", "mean"],
                lag=1,
                start="2000-01-01",
                years=None,
                blacklist=self.black,
            )

        with self.assertRaises(TypeError):
            # Test val is of type string.
            cr = CategoryRelations(
                self.dfdx,
                xcats=["GROWTH", "INFL"],
                cids=self.cidx,
                freq="D",
                xcat_aggs=["mean", "mean"],
                lag=1,
                start="2000-01-01",
                years=None,
                blacklist=self.black,
                val=1,
            )

        with self.assertRaises(TypeError):
            # Test xcat is of type either List or Tuple.
            cr = CategoryRelations(
                self.dfdx,
                xcats="GROWTH, INFL",
                cids=self.cidx,
                freq="D",
                xcat_aggs=["mean", "mean"],
                lag=1,
                start="2000-01-01",
                years=None,
                blacklist=self.black,
            )

        with self.assertRaises(TypeError):
            # Test slip is of type int.
            cr = CategoryRelations(
                self.dfdx,
                xcats=["GROWTH", "INFL"],
                cids=self.cidx,
                freq="D",
                xcat_aggs=["mean", "mean"],
                lag=1,
                start="2000-01-01",
                years=None,
                blacklist=self.black,
                slip="1",
            )

        with self.assertRaises(ValueError):
            # Test slip is non-negative.
            cr = CategoryRelations(
                self.dfdx,
                xcats=["GROWTH", "INFL"],
                cids=self.cidx,
                freq="D",
                xcat_aggs=["mean", "mean"],
                lag=1,
                start="2000-01-01",
                years=None,
                blacklist=self.black,
                slip=-2,
            )

        with self.assertRaises(TypeError):
            # Test xcat_aggs is of type List or Tuple.
            cr = CategoryRelations(
                self.dfdx,
                xcats=["GROWTH", "INFL"],
                cids=self.cidx,
                freq="D",
                xcat_aggs="mean, mean",
                lag=1,
                start="2000-01-01",
                years=None,
                blacklist=self.black,
            )

    def test_intersection_cids(self):
        self.cidx = ["AUD", "CAD", "GBP", "USD", "CHF"]

        with warnings.catch_warnings(record=True) as w:
            # Isolate the cross-sections available for both the corresponding categories.
            shared_cids = CategoryRelations.intersection_cids(
                self.dfdx, ["GROWTH", "INFL"], self.cidx
            )
            # assert that 2 userwarnings are raised. store the warning messages as warn_statement.
            self.assertEqual(len(w), 2)
            warn_statement = [str(warn.message) for warn in w]

        # Split on the full stops, so subsequently removed from the Strings displayed in
        # the console if the conditions are satisfied.
        self.assertTrue(warn_statement[0] == "GROWTH misses: ['AUD', 'USD'].")
        # Account for the space in the console separating both print statements.
        self.assertTrue(warn_statement[1] == "INFL misses: ['USD'].")

        # Aim to test the returned list of cross-sections.
        # Broaden the testcase to further test the accuracy.
        self.cidx = ["AUD", "CAD", "GBP", "USD", "EUR", "JPY", "NZD", "CHF"]
        # Print statements will be returned to the console.
        # need to catch warnings
        with warnings.catch_warnings(record=True) as w:
            shared_cids = CategoryRelations.intersection_cids(
                self.dfdx, ["GROWTH", "INFL"], self.cidx
            )

        self.assertTrue(sorted(shared_cids) == ["CAD", "CHF", "GBP"])

    @staticmethod
    def first_difference(temp_df: pd.DataFrame, expln_var: str, n_periods):
        """
        First-differencing of the explanatory variable. Used for a single cross-section.
        """

        explan_col = temp_df[expln_var].to_numpy()
        shift = np.empty(explan_col.size)
        shift[:] = np.nan
        shift[n_periods:] = explan_col[:-n_periods]

        temp_df[expln_var] -= shift
        temp_df = temp_df.astype(dtype=np.float32)

        return temp_df

    # Test the conversion method from raw value to either n-period differencing or
    # percentage change.
    def test_time_series(self):
        # Generate the DataFrame passed into the time_series() method: the procedure
        # occurs inside the Class's constructor.
        with warnings.catch_warnings(record=True) as w:
            shared_cids = CategoryRelations.intersection_cids(
                self.dfdx, ["GROWTH", "INFL"], self.cidx
            )

        no_cross_sections = len(shared_cids)
        # DataFrame passed into time_series() method.
        original_df = categories_df(
            self.dfdx,
            ["GROWTH", "INFL"],
            shared_cids,
            val="value",
            freq="W",
            blacklist=self.black,
            start="2000-01-01",
            years=None,
            lag=1,
            xcat_aggs=["mean", "mean"],
        )
        original_df_copy = original_df.copy()
        no_rows_original = original_df.shape[0]

        # The first aspect of the method that can be tested is the dimensionality of the
        # returned DataFrame. By computing the difference or percentage change over a
        # fixed number of time-periods, the first "n-periods" dates will not have the
        # preceding dates required to obtain a differenced value for that respective
        # date. Therefore, the index will be filled with a NaN which will subsequently be
        # dropped.
        # Test the above occurs for each cross-section. The number of rows in the
        # returned DataFrame will drop by (n_periods * cross_sections).

        n_periods = 3
        df_time_series = CategoryRelations.time_series(
            original_df,
            change="diff",
            n_periods=n_periods,
            shared_cids=shared_cids,
            expln_var="INFL",
        )

        row_formula = lambda no_rows, n: no_rows - (no_cross_sections * n)
        self.assertTrue(
            df_time_series.shape[0] == row_formula(no_rows_original, n_periods)
        )

        # Again, test the row logic but on a different testcase.
        n_periods = 6
        df_time_series = CategoryRelations.time_series(
            original_df_copy,
            change="pch",
            n_periods=n_periods,
            shared_cids=shared_cids,
            expln_var="INFL",
        )

        # The number of cross-sections remains unchanged from the above example.
        self.assertTrue(
            df_time_series.shape[0] == row_formula(no_rows_original, n_periods)
        )

        # Test the logic of the differencing method and percentage change. Take a random
        # index and manually check the logic is correct.
        # To test the fundamental logic of the time_series() method construct a separate
        # DataFrame.
        # Test on a single cross-section.
        cidx = ["GBP"]
        shared_cids = CategoryRelations.intersection_cids(
            self.dfdx, ["GROWTH", "INFL"], cidx
        )

        test_df = categories_df(
            self.dfdx,
            ["GROWTH", "INFL"],
            shared_cids,
            val="value",
            freq="W",
            blacklist=self.black,
            start="2000-01-01",
            years=None,
            lag=1,
            xcat_aggs=["mean", "mean"],
        )

        test_df_copy = test_df.copy().droplevel(level="cid")
        no_rows = test_df_copy.shape[0]
        # Isolate a randomly chosen row, index, to test the differencing logic.
        row_test = randint(n_periods, no_rows)

        n_periods = 4
        df_time_series = CategoryRelations.time_series(
            test_df,
            change="diff",
            n_periods=n_periods,
            shared_cids=shared_cids,
            expln_var="INFL",
        )

        df_time_series = df_time_series.droplevel(level="cid")
        # Accounts for the removal of NaN values.
        nan_adjustment = row_test - n_periods
        test_value = df_time_series.iloc[nan_adjustment]["INFL"]

        # Logic: manual computation.
        difference = (
            test_df_copy.iloc[row_test]["INFL"]
            - test_df_copy.iloc[nan_adjustment]["INFL"]
        )

        condition = abs(test_value - difference)
        self.assertTrue(condition < 0.0001)

        # Test the first-differencing in the context of a multi-index DataFrame.
        shared_cids = CategoryRelations.intersection_cids(
            self.dfd, ["GROWTH", "INFL"], self.cids
        )
        original_df = categories_df(
            self.dfd,
            ["GROWTH", "INFL"],
            shared_cids,
            val="value",
            freq="W",
            blacklist=self.black,
            start="2000-01-01",
            years=None,
            lag=1,
            xcat_aggs=["mean", "mean"],
        )

        n_periods = 4
        df_time_series = CategoryRelations.time_series(
            original_df,
            change="diff",
            n_periods=n_periods,
            shared_cids=shared_cids,
            expln_var="INFL",
        )

        test_df_aud = original_df.loc["AUD"]
        test_df_aud_dif = self.first_difference(
            test_df_aud, expln_var="INFL", n_periods=4
        )
        test_df_aud_dif = test_df_aud_dif.dropna(axis=0, how="any")

        df_time_series_aud = df_time_series[
            df_time_series.index.get_level_values("cid") == "AUD"
        ]
        df_time_series_aud_infl = df_time_series_aud["INFL"]

        # Handling of Python 3.7 GitHub.
        if test_df_aud_dif.shape[0] == df_time_series_aud_infl.shape[0]:
            condition = np.abs(
                test_df_aud_dif["INFL"].to_numpy() - df_time_series_aud_infl.to_numpy()
            )
            self.assertTrue(np.all(condition < 0.0001))

        # The logic and assembly of a the new DataFrame have both been tested. The other
        # methods in the Class are for visualisation and heavily dependent on external
        # packages.

    def test_outlier_trim(self):
        # Generate the dataframe passed into the outlier_trim() method: the procedure
        # occurs inside the Class's constructor.
        with warnings.catch_warnings(record=True) as w:
            shared_cids = CategoryRelations.intersection_cids(
                self.dfdx, ["GROWTH", "INFL"], self.cidx
            )

        no_cross_sections = len(shared_cids)
        # DataFrame passed into time_series() method or outlier_trim() depending on
        # parameter.
        original_df = categories_df(
            self.dfdx,
            ["GROWTH", "INFL"],
            shared_cids,
            val="value",
            freq="W",
            blacklist=self.black,
            start="2000-01-01",
            years=None,
            lag=1,
            xcat_aggs=["mean", "mean"],
        )

        xcat_trims = [2.5, 2.75]
        df = CategoryRelations.outlier_trim(
            df=original_df, xcats=["GROWTH", "INFL"], xcat_trims=xcat_trims
        )
        tuple_unpack = lambda tup: tup[0]
        c_sections = set(map(tuple_unpack, df.index))
        self.assertTrue(sorted(c_sections) == sorted(shared_cids))

        explanatory = df["GROWTH"]
        condition = np.where(explanatory > 2.5)
        list_ = next(iter(condition))
        self.assertTrue(len(list_) == 0)

        dependent = df["INFL"]
        condition = np.where(dependent > 2.75)
        list_ = next(iter(condition))
        self.assertTrue(len(list_) == 0)

        # Trivial test. Validate that if the floating point values in "xcat_trims" are
        # both set to their maximum value, the output dataframe's dimensions should match
        # the input dataframe.

        # Further, also test the method works changed dataframe (time-series modified).
        n_periods = 3
        df_time_series = CategoryRelations.time_series(
            original_df,
            change="diff",
            n_periods=n_periods,
            shared_cids=shared_cids,
            expln_var="GROWTH",
        )
        val_1 = np.max(np.abs(df_time_series["GROWTH"]))
        val_2 = np.max(np.abs(df_time_series["INFL"]))
        epsilon = 0.00001
        xcat_trims = [(val_1 + epsilon), (val_2 + epsilon)]

        df_outlier = CategoryRelations.outlier_trim(
            df=df_time_series, xcats=["GROWTH", "INFL"], xcat_trims=xcat_trims
        )

        # Adjust for the application of the lag.
        self.assertTrue(df_outlier.shape == df_time_series.shape)

    def test_apply_slip(self):
        warnings.simplefilter("always")
        # pick 3 random cids
        sel_xcats: List[str] = ["XR", "CRY"]
        sel_cids: List[str] = ["AUD", "CAD", "GBP"]
        sel_dates: pd.DatetimeIndex = pd.bdate_range(
            start="2020-01-01", end="2020-02-01"
        )

        # reduce the dataframe to the selected cids and xcats
        test_df: pd.DataFrame = self.dfd.copy()
        test_df = test_df[
            test_df["cid"].isin(sel_cids)
            & test_df["xcat"].isin(sel_xcats)
            & test_df["real_date"].isin(sel_dates)
        ].reset_index(drop=True)

        df: pd.DataFrame = test_df.copy()

        # Test Case 1

        # for every unique cid, xcat pair add a column "vx" which is just an integer 0→n ,
        # where n is the number of unique dates for that cid, xcat pair
        df["vx"] = (
            df.groupby(["cid", "xcat"])["real_date"].rank(method="dense").astype(int)
        )
        test_slip: int = 5
        # apply the slip method
        out_df = CategoryRelations.apply_slip(
            df=df,
            slip=test_slip,
            xcats=sel_xcats,
            cids=sel_cids,
            metrics=["value", "vx"],
        )

        # NOTE: casting df.vx to int as pandas casts it to float64
        self.assertEqual(int(df["vx"].max()) - test_slip, int(out_df["vx"].max()))

        for cid in sel_cids:
            for xcat in sel_xcats:
                inan_count = (
                    df[(df["cid"] == cid) & (df["xcat"] == xcat)]["vx"].isna().sum()
                )
                onan_count = (
                    out_df[(out_df["cid"] == cid) & (out_df["xcat"] == xcat)]["vx"]
                    .isna()
                    .sum()
                )
                self.assertEqual(inan_count, onan_count - test_slip)

        # Test Case 2 - slip is greater than the number of unique dates for a cid, xcat pair

        df: pd.DataFrame = test_df.copy()
        df["vx"] = (
            df.groupby(["cid", "xcat"])["real_date"].rank(method="dense").astype(int)
        )

        test_slip = int(max(df["vx"])) + 1

        out_df = CategoryRelations.apply_slip(
            df=df,
            slip=test_slip,
            xcats=sel_xcats,
            cids=sel_cids,
            metrics=["value", "vx"],
        )

        self.assertTrue(out_df["vx"].isna().all())
        self.assertTrue(out_df["value"].isna().all())

        out_df = CategoryRelations.apply_slip(
            df=df,
            slip=test_slip,
            xcats=sel_xcats,
            cids=sel_cids,
            metrics=["value"],
        )

        self.assertTrue((df["vx"] == out_df["vx"]).all())
        self.assertTrue(out_df["value"].isna().all())

        # case 3 - slip is negative
        df: pd.DataFrame = test_df.copy()

        with self.assertRaises(ValueError):
            CategoryRelations.apply_slip(
                df=df,
                slip=-1,
                xcats=sel_xcats,
                cids=sel_cids,
                metrics=["value"],
            )

        # check that a value error is raised when cids and xcats are not in the dataframe
        with warnings.catch_warnings(record=True) as w:
            CategoryRelations.apply_slip(
                df=df,
                slip=2,
                xcats=["metallica"],
                cids=["ac_dc"],
                metrics=["value"],
            )
            CategoryRelations.apply_slip(
                df=df,
                slip=2,
                xcats=["metallica"],
                cids=sel_cids,
                metrics=["value"],
            )
            CategoryRelations.apply_slip(
                df=df,
                slip=2,
                xcats=["metallica"],
                cids=sel_cids,
                metrics=["value"],
            )
            efilter = (
                "Tickers targetted for applying slip are not present in the DataFrame."
            )
            wlist = [_w for _w in w if efilter in str(_w.message)]

            self.assertEqual(len(wlist), 3)
        try:
            cat_rel: CategoryRelations = CategoryRelations(
                df=df, xcats=sel_xcats, cids=sel_cids, slip=100
            )
        except:
            self.fail("CategoryRelations init failed")

        warnings.resetwarnings()

    def test_reg_scatter_one_cid(self):
        sel_xcats: List[str] = ["XR", "CRY"]
        sel_cids: List[str] = ["AUD"]
        cr = CategoryRelations(
            self.dfd,
            xcats=sel_xcats,
            cids=sel_cids,
            freq="M",
            xcat_aggs=["mean", "mean"],
            lag=1,
            start="2000-01-01",
            years=None,
            blacklist=self.black,
        )
        # test warning raised when map is used with one cid
        with warnings.catch_warnings(record=True) as w:
            cr.reg_scatter(
                title="Carry and Return",
                xlab="Carry",
                ylab="Return",
                coef_box="lower right",
                prob_est="map",
            )
            self.assertEqual(len(w), 2) # Account for FigureCanvas due to mocking
            self.assertTrue(
                "The 'map' estimator is not applicable to a single cross-section. "
                "Using 'pool' instead." in str(w[0].message)
            )

    def test_reg_scatter(self):
        sel_xcats: List[str] = ["XR", "CRY"]
        sel_cids: List[str] = ["AUD", "CAD", "GBP"]
        cr = CategoryRelations(
            self.dfd,
            xcats=sel_xcats,
            cids=sel_cids,
            freq="M",
            xcat_aggs=["mean", "mean"],
            lag=1,
            start="2000-01-01",
            years=None,
            blacklist=self.black,
        )

        with self.assertRaises(AssertionError):
            cr.reg_scatter(
                labels=False,
                separator=2010,
                title="Carry and Return",
                xlab="Carry",
                ylab="Return",
                coef_box=2,
                ncol=5,
            )

        with self.assertRaises(AssertionError):
            cr.reg_scatter(
                labels=True,
                separator=2010,
                title="Carry and Return",
                xlab="Carry",
                ylab="Return",
                coef_box=2,
                ncol=5,
                prob_est="WRONG",
            )

        with self.assertRaises(TypeError):
            cr.reg_scatter(
                labels=True,
                separator=2010,
                title="Carry and Return",
                xlab="Carry",
                ylab="Return",
                ncol=5,
                ax=1,
            )

        with self.assertRaises(ValueError):
            cr.reg_scatter(
                labels=True,
                separator=2010,
                title="Carry and Return",
                xlab="Carry",
                ylab="Return",
                ncol=5,
                coef_box_font_size="WRONG",
            )

        with self.assertRaises(ValueError):
            cr.reg_scatter(
                labels=True,
                separator=2010,
                title="Carry and Return",
                xlab="Carry",
                ylab="Return",
                ncol=5,
                coef_box_font_size=-1,
            )

        with self.assertRaises(TypeError):
            cr.reg_scatter(
                labels=True,
                separator=2010,
                title="Carry and Return",
                xlab="Carry",
                ylab="Return",
                ncol=5,
                size=("1", 7),
            )

        with self.assertRaises(TypeError):
            cr.reg_scatter(
                labels=True,
                separator=2010,
                title="Carry and Return",
                xlab="Carry",
                ylab="Return",
                ncol=5,
                size=8,
            )

        with self.assertRaises(TypeError):
            cr.reg_scatter(
                labels=True,
                separator=2010,
                title="Carry and Return",
                xlab="Carry",
                ylab="Return",
                ncol=5,
                size=(1, 7, 33),
            )

    def test_xcat_chg1(self):
        # Test the first-differencing and percentage change methods
        try:
            cids_test = ["USD", "GBP", "JPY", "EUR"]
            xcats_test = ["XCAT1", "XCAT2"]

            dfx = make_test_df(cids=cids_test, xcats=xcats_test, start="2010-01-01")

            cr = CategoryRelations(
                df=dfx,
                xcats=xcats_test,
                cids=cids_test,
                freq="M",
                lag=1,
                xcat_aggs=["last", "sum"],
                slip=1,
                xcat1_chg="diff",
            )

            cr.reg_scatter(coef_box="lower right", prob_est="kendall", separator=2012)
        except:
            self.fail(
                "CategoryRelations failed when using seperator=2012, xcat1_chg=diff"
            )

    def test_reg_scatter_no_label(self):
        try:
            cids_test = ["USD", "GBP", "JPY", "EUR"]
            xcats_test = ["XCAT1", "XCAT2"]

            dfx = make_test_df(cids=cids_test, xcats=xcats_test, start="2010-01-01")

            cr = CategoryRelations(
                df=dfx,
                xcats=xcats_test,
                cids=cids_test,
                freq="M",  # TODO in principle select monthly to get more accurate picture...
                lag=0,
                xcat_aggs=["last", "last"],
                start="2000-01-01",
                xcat_trims=[None, None],
            )

            cr.reg_scatter(
                labels=False,
                coef_box="upper left",
                title="FX consistent core CPI excess inflation",
                xlab="ZN-scored signal",
                ylab="signal",
                separator="cids",
            )
        except:
            self.fail(
                "CategoryRelations failed when using seperator=cids, labels=False"
            )

    def test_int_seperator(self):
        try:
            cids_test = ["USD", "GBP", "JPY", "EUR"]
            xcats_test = ["XCAT1", "XCAT2"]

            dfx = make_test_df(cids=cids_test, xcats=xcats_test, start="2010-01-01")

            cr = CategoryRelations(
                df=dfx,
                xcats=xcats_test,
                cids=cids_test,
                freq="M",
                lag=1,
                xcat_aggs=["last", "sum"],
                slip=1,
            )

            cr.reg_scatter(coef_box="lower right", prob_est="kendall", separator=2012)
        except:
            self.fail("CategoryRelations failed when using seperator=2012")


if __name__ == "__main__":
    unittest.main()
