import unittest
from macrosynergy.signal.signal_return_relations import SignalReturnRelations

from tests.simulate import make_qdf
from sklearn.metrics import accuracy_score, precision_score
from scipy import stats
import random
import pandas as pd
import numpy as np
from typing import List, Dict
import matplotlib
from matplotlib import pyplot as plt
from unittest.mock import patch


class TestAll(unittest.TestCase):
    def setUp(self) -> None:
        """
        Create a standardised DataFrame defined over the three categories.
        """

        self.cids: List[str] = ["AUD", "CAD", "GBP", "NZD", "USD"]
        self.xcats: List[str] = ["XR", "CRY", "GROWTH", "INFL"]

        df_cids = pd.DataFrame(
            index=self.cids, columns=["earliest", "latest", "mean_add", "sd_mult"]
        )

        # Purposefully choose a different start date for all cross-sections. Used to test
        # communal sampling.
        df_cids.loc["AUD"] = ["2011-01-01", "2020-12-31", 0, 1]
        df_cids.loc["CAD"] = ["2009-01-01", "2020-10-30", 0, 2]
        df_cids.loc["GBP"] = ["2010-01-01", "2020-08-30", 0, 5]
        df_cids.loc["NZD"] = ["2008-01-01", "2020-06-30", 0, 3]
        df_cids.loc["USD"] = ["2012-01-01", "2020-12-31", 0, 4]

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

        df_xcats.loc["XR"] = ["2000-01-01", "2020-12-31", 0.1, 1, 0, 0.3]
        df_xcats.loc["CRY"] = ["2000-01-01", "2020-10-30", 0, 2, 0.95, 1]
        df_xcats.loc["GROWTH"] = ["2001-01-01", "2020-10-30", 0, 2, 0.9, 1]
        df_xcats.loc["INFL"] = ["2001-01-01", "2020-10-30", 0, 2, 0.8, 0.5]

        random.seed(2)
        dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)

        self.dfd: pd.DataFrame = dfd

        black = {
            "AUD": ["2000-01-01", "2003-12-31"],
            "GBP": ["2018-01-01", "2100-01-01"],
        }

        self.blacklist: Dict[str, List[str]] = black

        assert "dfd" in vars(self).keys(), (
            "Instantiation of DataFrame missing from " "field dictionary."
        )

    def tearDown(self) -> None:
        return super().tearDown()

    def test_constructor(self):
        # Test the Class's constructor.

        # First, test the assertions.
        # Trivial test to confirm the primary signal must be present in the passed
        # DataFrame.
        with self.assertRaises(AssertionError):
            srr = SignalReturnRelations(
                self.dfd, ret="XR", sig="Missing", freq="D", blacklist=self.blacklist
            )

        # Test that frequency must be one of the following: 'D', 'W', 'M', 'Q', 'Y'.
        with self.assertRaises(ValueError):
            srr = SignalReturnRelations(
                self.dfd,
                ret="XR",
                sig="CRY",
                freq="S",
                blacklist=self.blacklist,
                sig_neg=True,
            )

        signal = "CRY"
        srr: SignalReturnRelations = SignalReturnRelations(
            self.dfd, ret="XR", sig=signal, freq="D", blacklist=self.blacklist
        )

        # The signal will invariably be used as the explanatory variable and the return
        # as the dependent variable.
        # Confirm that the signal is lagged after applying categories_df().

        # Choose an arbitrary date and confirm that the signal in the original DataFrame
        # has been lagged by a day. Confirm on multiple cross-sections: AUD & USD.
        df_signal = self.dfd[self.dfd["xcat"] == signal]
        arbitrary_date_one = "2011-01-10"
        arbitrary_date_two = "2020-10-27"

        test_aud: pd.Series = df_signal[df_signal["real_date"] == arbitrary_date_one]
        test_aud = test_aud[test_aud["cid"] == "AUD"]["value"]

        test_usd: pd.Series = df_signal[df_signal["real_date"] == arbitrary_date_two]
        test_usd = test_usd[test_usd["cid"] == "USD"]["value"]

        lagged_df: pd.DataFrame = srr.df
        aud_lagged: float = lagged_df.loc["AUD", signal]["2011-01-11"]
        condition: float = round(test_aud.values[0], 5) - round(aud_lagged, 5)
        self.assertTrue(abs(condition) < 0.0001)

        usd_lagged: float = lagged_df.loc["USD", signal]["2020-10-28"]
        condition: float = round(test_usd.values[0], 5) - round(usd_lagged, 5)
        self.assertTrue(condition < 0.0001)

        # In addition to the DataFrame returned by categories_df(), an instance of the
        # Class will hold two "tables" for each segmentation type.
        # Confirm the indices are the expected: cross-sections or years.
        test_index: List[str] = list(srr.df_cs.index)[3:]
        self.assertTrue(sorted(self.cids) == sorted(test_index))

    def test_constructor_multiple_sigs(self):
        # The signal return Class allows for additional signals to be passed, upon
        # instantiation, to understand the primary signal's performance relative to other
        # possible signals. The analysis will be completed on the panel level.

        # First, test the assertions.
        # Any additional signals passed must either be a string or list.
        with self.assertRaises(AssertionError):
            srr = SignalReturnRelations(
                self.dfd,
                ret="XR",
                sig="CRY",
                rival_sigs=set(["GROWTH", "INFL"]),
                freq="D",
                blacklist=self.blacklist,
            )
        # Return Signal must be specified
        with self.assertRaises(ValueError):
            srr = SignalReturnRelations(
                self.dfd,
            )
        with self.assertRaises(ValueError):
            srr = SignalReturnRelations(
                self.dfd,
                ret="XR",
            )
        # Signals passed must be a subset of the categories defined in the DataFrame. If
        # not, will raise an assertion.
        with self.assertRaises(AssertionError):
            # GDP is not a defined category.
            srr = SignalReturnRelations(
                self.dfd,
                ret="XR",
                sig="CRY",
                rival_sigs=["GROWTH", "INFL", "GDP"],
                freq="D",
                blacklist=self.blacklist,
            )

        primary_signal = "CRY"
        rival_signals = ["GROWTH", "INFL"]
        srr = SignalReturnRelations(
            self.dfd,
            ret="XR",
            sig=primary_signal,
            rival_sigs=rival_signals,
            sig_neg=False,
            freq="D",
            blacklist=self.blacklist,
        )
        # First, confirm the signal list stored on the instance comprises both the
        # primary signal and the rival signals.
        self.assertTrue([primary_signal] + rival_signals == srr.signals)

        # Secondly, confirm the DataFrame is defined over the expected columns.
        self.assertTrue(
            list(srr.df.columns) == [primary_signal] + rival_signals + ["XR"]
        )

        # Test the negative conversion if the parameter 'sig_neg' is set to True. If set
        # to True, the signal fields will have a postfix, '_NEG', appended.
        srr_neg = SignalReturnRelations(
            self.dfd,
            ret="XR",
            sig=primary_signal,
            rival_sigs=rival_signals,
            sig_neg=True,
            blacklist=self.blacklist,
            freq="D",
        )

        # Firstly, confirm the signal names have been updated correctly.
        srr_signals = list(map(lambda s: s.split("_"), srr_neg.signals))
        signals = [primary_signal] + rival_signals
        for i, s in enumerate(srr_signals):
            self.assertTrue(s[1] == "NEG")
            self.assertTrue(s[0] == signals[i])

        self.assertEqual(srr_neg.sigs[0][-4:], "_NEG")
        self.assertEqual(srr_neg.sigs[0][:-4], primary_signal)

        # Secondly, confirm the actual DataFrame's columns have been updated.
        test_columns = list(srr_neg.df.columns)
        # Confirms the update has been made on the DataFrame level.
        self.assertTrue(test_columns == srr_neg.signals + ["XR"])

        # Lastly, check the original values have been multiplied by minus one. Therefore,
        # add the two DataFrames which should equate to zero. The multiplication by minus
        # one only occurs on the signals.

        srr_neg.df.rename(columns=dict(zip(srr_neg.signals, srr.signals)), inplace=True)
        zero_df = srr.df + srr_neg.df
        zero_df_sigs = zero_df.loc[:, srr.signals]
        sum_columns = zero_df_sigs.sum(axis=0)
        self.assertTrue(np.all(sum_columns.to_numpy() == 0.0))

    def test_constructor_communal(self):
        # Used to test the communal sample period by setting the parameter equal to True.
        # The DataFrame instantiated on the instance is a multi-index DataFrame where the
        # outer index will be qualified by the available cross-sections and the interior
        # index will be timestamps. The columns will be the respective signals plus
        # return.
        # If the parameter is set to True, the individual cross-section's start dates
        # will be aligned across the panels: the proposed start date will have a
        # realised value for all categories for that specific cross-section. The logic is
        # applied to each cross-section.

        primary_signal = "CRY"
        rival_signals = ["GROWTH", "INFL"]
        # Set "cosp" equal to True.
        srr_cosp = SignalReturnRelations(
            self.dfd,
            ret="XR",
            sig=primary_signal,
            rival_sigs=rival_signals,
            sig_neg=False,
            cosp=True,
            freq="D",
            blacklist=None,
        )

        # The start date for the communal series should be:
        # start_dates = {'AUD': '2011-01-01', 'CAD': '2009-01-01', 'GBP': '2010-01-01',
        #                'NZD': '2008-01-01', 'USD': '2012-01-01'}

        df = srr_cosp.df
        # Test across all cross-sections - aligned on the cross-section's intersection.
        # Account for weekends.
        expected_date = {
            "AUD": "2011-01-03",
            "CAD": "2009-01-01",
            "GBP": "2010-01-01",
            "NZD": "2008-01-01",
            "USD": "2012-01-02",
        }
        for c, cid_df in df.groupby(level=0):
            # Isolate the interior index.
            series_s_date = str(cid_df.iloc[0, :].name[1]).split(" ")[0]
            self.assertEqual(expected_date[c], series_s_date)

        # Test the values.
        dfd = srr_cosp.dfd
        filt_1 = (dfd["real_date"] == "2011-01-04") & (dfd["xcat"] == "XR")
        dfd_filt = dfd[filt_1]
        benchmark_value = float(dfd_filt[dfd_filt["cid"] == "AUD"]["value"].iloc[0])
        benchmark_value = round(benchmark_value, 5)

        test_row = srr_cosp.df.loc["AUD"].loc["2011-01-04"]
        condition = abs(benchmark_value - round(test_row["XR"], 5)) < 0.00001
        self.assertTrue(condition)

        # Account for lagging the signals. Therefore, the signal values will reference
        # the previous day.
        filt_2 = (dfd["real_date"] == "2011-01-03") & (dfd["cid"] == "AUD")
        dfd_filt = dfd[filt_2]
        signals = [primary_signal] + rival_signals

        for s in signals:
            test_value = float(dfd_filt[dfd_filt["xcat"] == s]["value"].iloc[0])
            self.assertTrue(np.isclose(test_value, test_row[s]))

        # Confirm the dimensions of the return column remains unchanged - alignment
        # occurs exclusively on the signals. To test, confirm the columns are the same
        # regardless of whether communal sampling is applied.
        srr = SignalReturnRelations(
            self.dfd,
            ret="XR",
            sig=primary_signal,
            rival_sigs=rival_signals,
            sig_neg=False,
            cosp=False,
            freq="D",
            blacklist=None,
        )
        self.assertTrue(
            srr.df.loc[:, srr.rets].shape == srr_cosp.df.loc[:, srr.rets].shape
        )

    def test__slice_df__(self):
        # Method used to confirm that the segmentation of the original DataFrame is
        # being applied correctly: either cross-sectional or yearly basis. Therefore, if
        # a specific "cs" is passed, will the DataFrame be reduced correctly ?

        signal = "CRY"
        srr = SignalReturnRelations(
            self.dfd, sig=signal, ret="XR", freq="D", blacklist=self.blacklist
        )
        df = srr.df.dropna(how="any").copy()

        # First, test cross-sectional basis.
        # Choose a "random" cross-section.

        df_cs = srr.__slice_df__(df=df, cs="GBP", cs_type="cids")

        # Test the values on a fixed date.
        fixed_date = "2010-01-04"
        test_values = dict(df.loc["GBP", fixed_date])
        segment_values = dict(df_cs.loc[fixed_date, :])

        for c, v in test_values.items():
            self.assertTrue(v - segment_values[c] < 0.0001)

        # Test the yearly segmentation.
        df.loc[:, "year"] = np.array(
            df.reset_index(level=1).loc[:, "real_date"].dt.year
        )

        df_cs = srr.__slice_df__(df=df, cs="2013", cs_type="years")

        # Confirm that the year column contains exclusively '2013'. If so, able to deduce
        # that the segmentation works correctly for yearly type.
        df_cs_year = df_cs["year"].to_numpy()
        df_cs_year = np.array(list(map(lambda y: str(y), df_cs_year)))
        self.assertTrue(np.all(df_cs_year == "2013"))

    def test__output_table__(self):
        # Test the method responsible for producing the table of metrics assessing the
        # signal-return relationship.

        # Firstly, for the six "Positive Ratio" statistics, confirm the computed value
        # for the accuracy score is correct. If so, able to conclude the other scores
        # are being assembled in the returned table correctly.
        signal = "CRY"
        return_ = "XR"
        srr = SignalReturnRelations(
            self.dfd, sig=signal, ret=return_, freq="D", blacklist=self.blacklist
        )
        df_cs = srr.__output_table__(cs_type="cids")

        # The lagged signal & returns have been reduced to[-1, 1] which are interpreted
        # as indicator random variables.
        # Test value.
        df_cs_aud_acc = df_cs.loc["AUD", "accuracy"]

        # Accounts for removal of dropna() from categories_df() function.
        srr_df = srr.df.dropna(axis=0, how="any")

        aud_df = srr_df.loc["AUD", :]
        # Remove zero values.
        aud_df = aud_df[~((aud_df.iloc[:, 0] == 0) | (aud_df.iloc[:, 1] == 0))]
        # In the context of the accuracy score, reducing the DataFrame to boolean values
        # will work equivalently to [1, -1].
        aud_df = aud_df > 0
        y_pred = aud_df[signal]
        y_true = aud_df[return_]
        accuracy = accuracy_score(y_true, y_pred)
        self.assertTrue(abs(df_cs_aud_acc - accuracy) < 0.00001)

        # Aim to test Kendall Tau correlation statistic via stats.rankdata() - Kendall
        # Tau is implicitly ranking the data but using the original values.
        # Kendall Tau is non-parametric, and both the return & signal series will be
        # used as quasi-ranking data.
        df_cs_usd_ken = df_cs.loc["USD", "kendall"]

        usd_df = srr_df.loc["USD", :]
        usd_df = usd_df[~((usd_df.iloc[:, 0] == 0) | (usd_df.iloc[:, 1] == 0))]
        x = stats.rankdata(usd_df[signal]).astype(int)
        y = stats.rankdata(usd_df[return_]).astype(int)

        kendall_tau, p_value = stats.kendalltau(x, y)
        # Kendall Tau offers value when used in conjunction with Pearson's
        # correlation coefficient which is a linear measure.
        # For instance, if the Pearson correlation coefficient is close to zero but
        # the Kendall Tau is close to one, it can be deduced that there is a
        # relationship between the two variables but a non-linear relationship.
        # Alternatively, if the Pearson coefficient is close to one but the Kendall
        # Tau is closer to zero, it suggests that the sample is exposed to a small
        # number of sharp outliers.
        self.assertTrue(abs(df_cs_usd_ken - kendall_tau) < 0.00001)

        # Test the linear correlation measure, Pearson correlation coefficient.
        df_cs_usd_pearson = df_cs.loc["USD", "pearson"]
        sig_ret_cov_matrix = np.cov(usd_df[signal], usd_df[return_])
        sig_ret_cov = sig_ret_cov_matrix[0, 1]

        # Covariance divided by the product of the variance.
        manual_calc = sig_ret_cov / (np.std(usd_df[signal]) * np.std(usd_df[return_]))

        self.assertTrue(abs(df_cs_usd_pearson - manual_calc) < 0.0001)

        # Test the precision score which will record the signal's positive bias. This is
        # important because the greater the number of false positives, the more exposed a
        # strategy is and subsequently any gains, through true positives, will be
        # reversed.
        positive_signals_index = usd_df[signal] > 0
        positive_signals = usd_df[signal][positive_signals_index]
        positive_signals = np.sign(positive_signals)

        negative_signals = usd_df[signal][~positive_signals_index]
        negative_signals = np.sign(negative_signals)

        return_series_pos = np.sign(usd_df[return_][positive_signals_index])
        return_series_neg = np.sign(usd_df[return_][~positive_signals_index])

        positive_accuracy = accuracy_score(positive_signals, return_series_pos)
        negative_accuracy = accuracy_score(negative_signals, return_series_neg)

        manual_precision = (positive_accuracy + (1 - negative_accuracy)) / 2
        df_cs_usd_posprec = df_cs.loc["USD", "pos_prec"]

        self.assertTrue(abs(manual_precision - df_cs_usd_posprec) < 0.1)

        # Lastly, confirm that 'Mean' row is computed using exclusively the respective
        # segmentation types. Test on yearly data and balanced accuracy.
        df_ys = srr.__output_table__(cs_type="years")
        df_ys_mean = df_ys.loc["Mean", "bal_accuracy"]

        dfx = df_ys[~df_ys.index.isin(["Panel", "Mean", "PosRatio"])]
        dfx_balance = dfx["bal_accuracy"]
        condition = np.abs(np.mean(dfx_balance) - df_ys_mean)
        self.assertTrue(condition < 0.00001)

    def test__rival_sigs__(self):
        # Method is used to produce the metric table for the secondary signals. The
        # analysis will be completed on the panel level.

        # Test the construction of the table is correct and the values include all
        # cross-sections.
        primary_signal = "CRY"
        rival_signals = ["GROWTH", "INFL"]
        srr = SignalReturnRelations(
            self.dfd,
            ret="XR",
            sig=primary_signal,
            rival_sigs=rival_signals,
            sig_neg=False,
            freq="D",
            blacklist=self.blacklist,
        )

        df_sigs = srr.__rival_sigs__(ret="XR")

        # Firstly, confirm that the index consists of only the primary and rival signals.
        self.assertEqual(list(df_sigs.index), [primary_signal] + rival_signals)

        # Secondly, test the actual calculation on a single signal. Test the accuracy
        # score. If correct, all metrics should be correct.
        growth_accuracy = df_sigs.loc["GROWTH", "accuracy"]

        test_df = srr.df.loc[:, ["GROWTH", "XR"]]
        test_df = test_df.dropna(axis=0, how="any")
        df_sgs = np.sign(test_df)
        manual_value = accuracy_score(df_sgs["GROWTH"], df_sgs["XR"])
        self.assertEqual(growth_accuracy, manual_value)

    def test_signals_table(self):
        # If defined, will return the panel-level table for the rival signals. The method
        # receives a single parameter, "sigs", whose default is set to None (all
        # available signals are returned).

        primary_signal = "CRY"
        rival_signals = ["GROWTH", "INFL"]
        srr = SignalReturnRelations(
            self.dfd,
            ret="XR",
            sig=primary_signal,
            rival_sigs=rival_signals,
            sig_neg=False,
            freq="M",
            blacklist=self.blacklist,
        )
        # Firstly, confirm that if the parameter 'sigs' is left undefined, all signals
        # will be displayed in the table (default setting).
        df_sigs = srr.signals_table()
        self.assertEqual(list(df_sigs.index), [primary_signal] + rival_signals)

        # Secondly, confirm that a specific subset of the signals can be displayed.
        df_sigs = srr.signals_table(sigs=["CRY", "INFL"])
        self.assertEqual(list(df_sigs.index), [primary_signal] + ["INFL"])

        # Confirm that a list of signals must be passed. The method is for comparison
        # purposes. Therefore, at a minimum, it expects to receive the primary signal
        # plus an additional rival signal.
        with self.assertRaises(AssertionError):
            # Pass in a single signal as a string.
            df_sigs = srr.signals_table(sigs="INFL")

        # Lastly, confirm that an AttributeError will be thrown if the method is called
        # but additional signals have not been passed to the instance.
        srr = SignalReturnRelations(
            self.dfd,
            ret="XR",
            sig=primary_signal,
            rival_sigs=None,
            sig_neg=False,
            freq="M",
            blacklist=self.blacklist,
        )
        with self.assertRaises(AttributeError):
            df_sigs = srr.signals_table()

    def test__yaxis_lim__(self):
        signal = "CRY"
        return_ = "XR"
        srr = SignalReturnRelations(
            self.dfd, sig=signal, ret=return_, freq="D", blacklist=self.blacklist
        )
        df_cs = srr.__output_table__(cs_type="cids")
        dfx = df_cs[~df_cs.index.isin(["PosRatio"])]
        dfx_acc = dfx.loc[:, ["accuracy", "bal_accuracy"]]
        arr_acc = dfx_acc.to_numpy()
        arr_acc = arr_acc.flatten()

        # Flatten the array - only concerned with the minimum across both dimensions. If
        # the minimum value is less than 0.45, use the minimum value to initiate the
        # range. Test the above logic.
        ylim = srr.__yaxis_lim__(accuracy_df=dfx_acc)

        min_value = min(arr_acc)
        if min_value < 0.45:
            self.assertTrue(ylim == min_value)
        else:
            self.assertTrue(ylim == 0.45)

    def test_apply_slip(self):
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

        out_df = SignalReturnRelations.apply_slip(
            df=df,
            slip=test_slip,
            xcats=sel_xcats,
            cids=sel_cids,
            metrics=["value", "vx"],
        )

        # NOTE: casting df.vx to int as pandas casts it to float64
        self.assertEqual(int(min(df["vx"])) + test_slip, int(min(out_df["vx"])))

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
                assert inan_count == onan_count - test_slip

        # Test Case 2 - slip is greater than the number of unique dates for a cid, xcat pair

        df: pd.DataFrame = test_df.copy()
        df["vx"] = (
            df.groupby(["cid", "xcat"])["real_date"].rank(method="dense").astype(int)
        )

        test_slip = int(max(df["vx"])) + 1

        out_df = SignalReturnRelations.apply_slip(
            df=df,
            slip=test_slip,
            xcats=sel_xcats,
            cids=sel_cids,
            metrics=["value", "vx"],
        )

        self.assertTrue(out_df["vx"].isna().all())
        self.assertTrue(out_df["value"].isna().all())

        out_df = SignalReturnRelations.apply_slip(
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
            SignalReturnRelations.apply_slip(
                df=df,
                slip=-1,
                xcats=sel_xcats,
                cids=sel_cids,
                metrics=["value"],
            )

        with self.assertRaises(ValueError):
            SignalReturnRelations.apply_slip(
                df=df,
                slip=-1,
                xcats=sel_xcats,
                cids=["ac_dc"],
                metrics=["value"],
            )

        # check that a value error is raised when cids and xcats are not in the dataframe
        with self.assertWarns(UserWarning):
            SignalReturnRelations.apply_slip(
                df=df,
                slip=2,
                xcats=["metallica"],
                cids=["ac_dc"],
                metrics=["value"],
            )

        with self.assertWarns(UserWarning):
            SignalReturnRelations.apply_slip(
                df=df,
                slip=2,
                xcats=["metallica"],
                cids=sel_cids,
                metrics=["value"],
            )

        try:
            rival_signals: List[str] = ["GROWTH", "INFL"]
            primary_signal: str = "CRY"
            srr = SignalReturnRelations(
                self.dfd,
                ret="XR",
                sig=primary_signal,
                rival_sigs=rival_signals,
                sig_neg=False,
                freq="M",
                blacklist=self.blacklist,
                slip=100,
            )
        except:
            self.fail("SignalReturnRelations init failed")

    def test_cross_section_and_yearly_table(self):
        srr = SignalReturnRelations(
            self.dfd,
            ret="XR",
            sig="CRY",
            rival_sigs=["GROWTH", "INFL"],
            sig_neg=False,
            freq="M",
            blacklist=self.blacklist,
        )
        self.assertTrue(srr.cross_section_table().shape == (8, 10))
        self.assertTrue(srr.yearly_table().shape == (16, 10))

    def test_accuracy_and_correlation_bars(self):
        plt.close("all")
        mock_plt = patch("matplotlib.pyplot.show").start()
        mpl_backend = matplotlib.get_backend()
        matplotlib.use("Agg")

        srr = SignalReturnRelations(
            self.dfd,
            ret="XR",
            sig="CRY",
            rival_sigs=["GROWTH", "INFL"],
            sig_neg=False,
            freq="M",
            blacklist=self.blacklist,
        )

        # Check that accuracy bars actually outputs an image
        try:
            srr.accuracy_bars()
        except Exception as e:
            self.fail(f"accuracy_bars raised {e} unexpectedly")

        try:
            srr.correlation_bars()
        except Exception as e:
            self.fail(f"correlation_bars raised {e} unexpectedly")

        try:
            srr.accuracy_bars(sig="XR")
        except Exception as e:
            self.fail(f"accuracy_bars raised {e} unexpectedly")

        try:
            srr.correlation_bars(sig="CRY")
        except Exception as e:
            self.fail(f"correlation_bars raised {e} unexpectedly")

        try:
            srr.accuracy_bars(ret="XR")
        except Exception as e:
            self.fail(f"accuracy_bars raised {e} unexpectedly")

        try:
            srr.correlation_bars(ret="XR")
        except Exception as e:
            self.fail(f"correlation_bars raised {e} unexpectedly")

        try:
            srr.accuracy_bars(ret="XR", sig="CRY")
        except Exception as e:
            self.fail(f"accuracy_bars raised {e} unexpectedly")

        try:
            srr.correlation_bars(ret="XR", sig="CRY")
        except Exception as e:
            self.fail(f"correlation_bars raised {e} unexpectedly")

        srr = SignalReturnRelations(
            self.dfd,
            ret=["XR", "GROWTH"],
            sig=["CRY", "INFL"],
            sig_neg=False,
            freq="M",
            blacklist=self.blacklist,
        )

        try:
            srr.accuracy_bars(ret="GROWTH", sig="INFL")
        except Exception as e:
            self.fail(f"accuracy_bars raised {e} unexpectedly")

        try:
            srr.correlation_bars(ret="GROWTH", sig="INFL")
        except Exception as e:
            self.fail(f"correlation_bars raised {e} unexpectedly")

        plt.close("all")
        matplotlib.use(mpl_backend)
        patch.stopall()

    def test_summary_table(self):
        srr = SignalReturnRelations(
            self.dfd,
            ret="XR",
            sig="CRY",
            rival_sigs=["GROWTH", "INFL"],
            sig_neg=False,
            freq="M",
            blacklist=self.blacklist,
        )

        self.assertTrue(srr.summary_table().shape == (5, 10))

    def test_single_relation_table(self):
        sr = SignalReturnRelations(
            df=self.dfd,
            ret="XR",
            sig="CRY",
            freq="Q",
            blacklist=self.blacklist,
            slip=1,
        )

        sr_no_slip = SignalReturnRelations(
            df=self.dfd,
            ret="XR",
            sig="CRY",
            freq="Q",
            blacklist=self.blacklist,
            slip=0,
        )

        # Test that each argument must be of the correct type
        with self.assertRaises(TypeError):
            sr.single_relation_table(ret=2)

        with self.assertRaises(TypeError):
            sr.single_relation_table(xcat=2)

        with self.assertRaises(TypeError):
            sr.single_relation_table(freq=2)

        with self.assertRaises(TypeError):
            sr.single_relation_table(agg_sigs=2)

        sr.single_relation_table()
        sr_no_slip.single_relation_table()

        # Test that dataframe has been reduced to just the relevant columns and has
        # applied slippage

        self.assertTrue(set(sr.dfd["xcat"]) == set(["XR", "CRY"]))

        self.assertTrue(sr.dfd["value"][0] != sr.df["value"][0])

        self.assertTrue(sr_no_slip.dfd["value"][0] == sr_no_slip.df["value"][0])

        sr.single_relation_table(ret="XR", xcat="CRY", freq="Q", agg_sigs="last")

        self.assertTrue(set(sr.dfd["xcat"]) == set(["XR", "CRY"]))

        self.assertTrue(sr.dfd["value"][0] != sr.df["value"][0])

        # Test Negative signs are correctly handled

        with self.assertRaises(TypeError):
            sr_sign_fail = SignalReturnRelations(
                df=self.dfd,
                ret="XR",
                sig="CRY",
                freq="Q",
                sig_neg=["FAIL"],
                blacklist=self.blacklist,
                slip=1,
            )

        # Ensure that the signs doesn't have a longer length than the number of signals
        with self.assertRaises(ValueError):
            sr_long_signs = SignalReturnRelations(
                df=self.dfd,
                ret="XR",
                sig="CRY",
                freq="Q",
                sig_neg=[True, True],
                blacklist=self.blacklist,
                slip=1,
            )

        # Test table outputted is correct
        data = {
            "cid": [
                "AUD",
                "AUD",
                "AUD",
                "AUD",
                "AUD",
                "AUD",
                "AUD",
                "AUD",
                "AUD",
                "AUD",
            ],
            "xcat": ["XR", "XR", "XR", "XR", "XR", "CRY", "CRY", "CRY", "CRY", "CRY"],
            "real_date": [
                "1990-01-01",
                "1990-01-02",
                "1990-01-03",
                "1990-01-04",
                "1990-01-05",
                "1990-01-01",
                "1990-01-02",
                "1990-01-03",
                "1990-01-04",
                "1990-01-05",
            ],
            "value": [1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0],
        }

        test_df = pd.DataFrame(data)

        sr_correct = SignalReturnRelations(
            df=test_df,
            ret="XR",
            sig="CRY",
            freq="D",
            blacklist=None,
            slip=0,
        )

        srt = sr_correct.single_relation_table()

        correct_stats = [
            0.25,
            0.25,
            0.5,
            0.75,
            0.5,
            0.0,
            -0.57735,
            0.42265,
            -0.57735,
            0.31731,
        ]

        for val1, val2 in zip(srt.iloc[0].values.tolist(), correct_stats):
            self.assertTrue(np.isclose(val1, val2))

        # Check when signs are negative

        sr_correct_neg = SignalReturnRelations(
            df=test_df,
            ret="XR",
            sig="CRY",
            freq="D",
            sig_neg=True,
            blacklist=None,
            slip=0,
        )

        srt = sr_correct_neg.single_relation_table()

        correct_stats = [
            0.75,
            0.75,
            0.5,
            0.75,
            1.0,
            0.5,
            0.57735,
            0.42265,
            0.57735,
            0.31731,
        ]

        for val1, val2 in zip(srt.iloc[0].values.tolist(), correct_stats):
            self.assertTrue(np.isclose(val1, val2))

    def test_multiple_relation_table(self):
        num_of_acc_cols = 10

        sr_unsigned = SignalReturnRelations(
            df=self.dfd,
            ret="XR",
            sig="CRY",
            freq="Q",
            agg_sig="last",
            blacklist=self.blacklist,
            slip=1,
        )

        self.assertTrue(
            sr_unsigned.multiple_relations_table(rets="XR", xcats="CRY").shape
            == (1, num_of_acc_cols)
        )

        sr_mrt = SignalReturnRelations(
            df=self.dfd,
            ret=["XR", "GROWTH"],
            sig=["CRY", "INFL"],
            freq=["Q", "M"],
            agg_sig=["last", "mean"],
            blacklist=self.blacklist,
        )

        self.assertTrue(
            sr_mrt.multiple_relations_table().shape == (16, num_of_acc_cols)
        )

        with self.assertRaises(ValueError):
            sr_mrt.multiple_relations_table(rets="TEST")

        with self.assertRaises(ValueError):
            sr_mrt.multiple_relations_table(xcats="TEST")

        with self.assertRaises(ValueError):
            sr_mrt.multiple_relations_table(freqs="TEST")

        with self.assertRaises(ValueError):
            sr_mrt.multiple_relations_table(agg_sigs="TEST")

        # Test that the table is inputs can take in both a list of strings and a string
        # self.assertTrue(sr_mrt.multiple_relations_table(rets="XR", freqs='Q'))

        rets = ["XR", "GROWTH"]
        xcats = ["INFL"]
        freqs = ["Q", "M"]
        agg_sigs = ["mean"]
        mrt = sr_mrt.multiple_relations_table(
            rets=rets, xcats=xcats, freqs=freqs, agg_sigs=agg_sigs
        )
        self.assertTrue(mrt.shape == (4, num_of_acc_cols))

    def test_single_statistic_table(self):
        sr = SignalReturnRelations(
            df=self.dfd,
            ret="XR",
            sig="CRY",
            freq="Q",
            blacklist=self.blacklist,
            slip=1,
        )

        with self.assertRaises(ValueError):
            sr.single_statistic_table(stat="FAIL")

        with self.assertRaises(ValueError):
            sr.single_statistic_table(stat="accuracy", type="FAIL")

        with self.assertRaises(TypeError):
            sr.single_statistic_table(stat="accuracy", rows="FAIL")

        with self.assertRaises(TypeError):
            sr.single_statistic_table(stat="accuracy", columns="FAIL")

        with self.assertRaises(ValueError):
            sr.single_statistic_table(stat="accuracy", rows=["FAIL"])

        with self.assertRaises(ValueError):
            sr.single_statistic_table(stat="accuracy", columns=["FAIL"])

        # Test that table is correctly shaped

        self.assertTrue(sr.single_statistic_table(stat="accuracy").shape == (1, 1))

        sr = SignalReturnRelations(
            df=self.dfd,
            ret=["XR", "GROWTH"],
            sig=["CRY", "INFL"],
            freq=["Q", "M"],
            agg_sig=["last", "mean"],
            blacklist=self.blacklist,
            slip=1,
        )

        self.assertTrue(sr.single_statistic_table(stat="accuracy").shape == (4, 4))

        # Test that the table is correctly shaped when rows and columns are specified

        self.assertTrue(
            sr.single_statistic_table(
                stat="accuracy", rows=["freq", "xcat", "ret"], columns=["agg_sigs"]
            ).shape
            == (8, 2)
        )

        sr = SignalReturnRelations(
            df=self.dfd,
            ret=["XR", "GROWTH"],
            sig=["CRY", "INFL"],
            freq=["Q", "M"],
            agg_sig=["last", "mean"],
            blacklist=self.blacklist,
            sig_neg=[False, True],
        )

        self.assertTrue(
            sr.single_statistic_table(
                stat="accuracy", rows=["xcat"], columns=["freq", "agg_sigs", "ret"]
            ).index[1]
            == "INFL_NEG"
        )

    def test_set_df_labels(self):
        rets = ["XR", "GROWTH"]
        freqs = ["Q", "M"]
        sigs = ["CRY", "INFL"]
        agg_sigs = ["mean", "last"]

        rows_dict = {"xcat": sigs, "ret": rets, "freq": freqs, "agg_sigs": agg_sigs}
        rows = ["xcat", "ret", "freq"]
        columns = ["agg_sigs"]

        sr = SignalReturnRelations(
            df=self.dfd,
            ret=rets,
            sig=sigs,
            freq=freqs,
            agg_sig=agg_sigs,
            blacklist=self.blacklist,
        )

        rows_names, columns_names = sr.set_df_labels(
            rows_dict=rows_dict, rows=rows, columns=columns
        )
        expected_col_names = ["mean", "last"]
        expected_row_names = [
            "CRY/XR/Q",
            "CRY/XR/M",
            "CRY/GROWTH/Q",
            "CRY/GROWTH/M",
            "INFL/XR/Q",
            "INFL/XR/M",
            "INFL/GROWTH/Q",
            "INFL/GROWTH/M",
        ]

        self.assertTrue(rows_names == expected_row_names)
        self.assertTrue(columns_names == expected_col_names)

        rows = ["xcat", "ret"]
        columns = ["agg_sigs", "freq"]

        rows_names, columns_names = sr.set_df_labels(
            rows_dict=rows_dict, rows=rows, columns=columns
        )

        expected_col_names = ["mean/Q", "mean/M", "last/Q", "last/M"]
        expected_row_names = ["CRY/XR", "CRY/GROWTH", "INFL/XR", "INFL/GROWTH"]

        self.assertTrue(rows_names == expected_row_names)
        self.assertTrue(columns_names == expected_col_names)

        rows = ["xcat"]
        columns = columns = ["agg_sigs", "ret", "freq"]

        rows_names, columns_names = sr.set_df_labels(
            rows_dict=rows_dict, rows=rows, columns=columns
        )

        expected_col_names = [
            "mean/XR/Q",
            "mean/XR/M",
            "mean/GROWTH/Q",
            "mean/GROWTH/M",
            "last/XR/Q",
            "last/XR/M",
            "last/GROWTH/Q",
            "last/GROWTH/M",
        ]
        expected_row_names = ["CRY", "INFL"]

        self.assertTrue(rows_names == expected_row_names)
        self.assertTrue(columns_names == expected_col_names)

        return 0

    def test_get_rowcol(self):
        rets = ["XR", "GROWTH"]
        freqs = ["Q", "M"]
        sigs = ["CRY", "INFL"]
        agg_sigs = ["mean", "last"]

        sr = SignalReturnRelations(
            df=self.dfd,
            ret=rets,
            sig=sigs,
            freq=freqs,
            agg_sig=agg_sigs,
            blacklist=self.blacklist,
        )

        hash = "XR/CRY/Q/mean"
        rows = ["xcat", "ret", "freq"]
        columns = ["agg_sigs"]

        self.assertTrue(sr.get_rowcol(hash, rows) == "CRY/XR/Q")
        self.assertTrue(sr.get_rowcol(hash, columns) == "mean")

    def test_single_statistic_table_show_heatmap(self):
        self.mpl_backend: str = matplotlib.get_backend()
        matplotlib.use("Agg")

        sr = SignalReturnRelations(
            df=self.dfd,
            ret="XR",
            sig="CRY",
            freq="Q",
            blacklist=self.blacklist,
            slip=1,
        )

        try:
            sr.single_statistic_table(stat="accuracy", show_heatmap=True)
        except Exception as e:
            self.fail(f"single_statistic_table raised {e} unexpectedly")

        try:
            sr.single_statistic_table(
                stat="accuracy", show_heatmap=True, row_names=["X"], column_names=["Y"]
            )
        except Exception as e:
            self.fail(f"single_statistic_table raised {e} unexpectedly")

        try:
            sr.single_statistic_table(
                stat="accuracy",
                show_heatmap=True,
                title="Test",
                min_color=0,
                max_color=1,
                annotate=False,
                figsize=(10, 10),
            )
        except Exception as e:
            self.fail(f"single_statistic_table raised {e} unexpectedly")


if __name__ == "__main__":
    unittest.main()
