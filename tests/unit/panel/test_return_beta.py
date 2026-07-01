import unittest
import math
from typing import List, Dict

import numpy as np
import pandas as pd
import pytest
from parameterized import parameterized
from sklearn.linear_model import LinearRegression

from tests.simulate import make_qdf

from macrosynergy.panel.return_beta import (
    date_alignment,
    hedge_calculator,
    adjusted_returns,
    return_beta,
    weighted_least_squares,
)
from macrosynergy.management.utils import reduce_df, _map_to_business_day_frequency
from macrosynergy.learning import TimeWeightedLinearRegression


class TestAll(unittest.TestCase):
    def setUp(self) -> None:
        # Emerging Market Asian countries.
        cids: List[str] = ["IDR", "INR", "KRW", "MYR", "PHP"]
        # Add the US - used as the hedging asset.
        cids += ["USD"]

        self.cids: List[str] = cids
        xcats: List[str] = ["FXXR_NSA", "GROWTHXR_NSA", "INFLXR_NSA", "EQXR_NSA"]
        self.xcats: List[str] = xcats

        df_cids = pd.DataFrame(
            index=self.cids, columns=["earliest", "latest", "mean_add", "sd_mult"]
        )

        df_cids.loc["IDR"] = ["2010-01-01", "2020-12-31", 0.5, 2]
        df_cids.loc["INR"] = ["2011-01-01", "2020-11-30", 0, 1]
        df_cids.loc["KRW"] = ["2012-01-01", "2020-11-30", -0.2, 0.5]
        df_cids.loc["MYR"] = ["2013-01-01", "2020-09-30", -0.2, 0.5]
        df_cids.loc["PHP"] = ["2002-01-01", "2020-09-30", -0.1, 2]
        df_cids.loc["USD"] = ["2000-01-01", "2020-03-20", 0, 1.25]

        df_xcats = pd.DataFrame(
            index=xcats,
            columns=[
                "earliest",
                "latest",
                "mean_add",
                "sd_mult",
                "ar_coef",
                "back_coef",
            ],
        )

        df_xcats.loc["FXXR_NSA"] = ["2012-01-01", "2020-10-30", 1, 2, 0.9, 1]
        df_xcats.loc["GROWTHXR_NSA"] = ["2012-01-01", "2020-10-30", 1, 2, 0.9, 1]
        df_xcats.loc["INFLXR_NSA"] = ["2013-01-01", "2020-10-30", 1, 2, 0.8, 0.5]
        df_xcats.loc["EQXR_NSA"] = ["2000-01-01", "2022-03-14", 0.5, 2, 0, 0.2]

        # If the asset being used as the hedge experiences a blackout period, then it is
        # probably not an appropriate asset to use in the hedging strategy.
        blacklist = {
            "IDR": ["2010-01-01", "2012-01-04"],
            "INR": ["2010-01-01", "2013-12-31"],
        }
        self.blacklist: Dict[str, List[str]] = blacklist

        # Standard df for tests.
        self.dfd: pd.DataFrame = make_qdf(df_cids, df_xcats, back_ar=0.75)

        # The Unit Test will be based on the hedging strategy: hedge FX returns
        # (FXXR_NSA) against US Equity, S&P 500, (USD_EQXR_NSA).
        cid_hedge = "USD"
        xcat_hedge = "EQXR_NSA"
        self.benchmark_df: pd.DataFrame = reduce_df(
            self.dfd, xcats=[xcat_hedge], cids=cid_hedge
        )

        self.unhedged_df: pd.DataFrame = reduce_df(
            self.dfd, xcats=["FXXR_NSA"], cids=cids
        )
        self.dfp_w = self.unhedged_df.pivot(
            index="real_date", columns="cid", values="value"
        )

    def tearDown(self) -> None:
        return super().tearDown()

    def test_df_cols(self):
        """
        The dataframe passed to the return_beta() method needs to have the following
        columns: 'cid', 'xcid', 'real_date', 'value'. Any extra columns will be dropped.
        This test checks if the function successfully raises a ValueError if the
        dataframe does not have the required columns.
        """

        df_test: pd.DataFrame = self.dfd.copy()
        # DO NOT CHANGE THE ORDER OF THE FOLLOWING LIST `expc_cols`
        expc_cols: List[str] = ["cid", "xcat", "real_date", "value"]

        for col_name in expc_cols:
            df_test.rename(columns={col_name: col_name + "_"}, inplace=True)
            with self.assertRaises(ValueError):
                return_beta(
                    df=df_test,
                    cids=self.cids,
                    xcat=self.xcats[0],
                    benchmark_return=f"{self.cids[0]}_{self.xcats[0]}",
                    start="2010-01-01",
                )

    def test_date_alignment(self):
        """
        Firstly, return_beta.py will potentially use a single asset to hedge a panel
        which can consist of multiple cross-sections, and each cross-section could be
        defined over differing time-series. Therefore, the .date_alignment() method is
        used to ensure the asset being used as the hedge and the asset being hedged are
        defined over the same timestamps. The method will return the proposed start &
        end date.
        """

        # Verify that two series passed will be aligned after applying the respective
        # method.
        # Test on MYR_FXXR_NSA against the hedging asset, USD_EQXR_NSA (both are defined
        # over different time horizons).
        c = "MYR"
        xr = self.dfp_w[c]
        # Adjusts for the effect of pivoting.
        xr = xr.dropna(axis=0, how="all")

        br = pd.Series(
            data=self.benchmark_df["value"], index=self.benchmark_df["real_date"]
        )

        start_date, end_date = date_alignment(unhedged_return=xr, benchmark_return=br)
        # The latest start date of the two pd.Series.
        target_start = "2013-01-01"
        start_date = pd.Timestamp(start_date).strftime("%Y-%m-%d")
        self.assertTrue(start_date == target_start)

        end_date = pd.Timestamp(end_date).strftime("%Y-%m-%d")
        target_end = "2020-03-20"
        self.assertTrue(end_date == target_end)

    @parameterized.expand(["ols", "twls"])
    def test_hedge_calculator(self, method):
        """
        Method designed to calculate the hedge ratios used across the panel: each cross-
        section in the panel will have a different sensitivity parameter relative to the
        benchmark.
        Further, the frequency in which the hedge ratios are calculated is delimited by
        the 'refreq' parameter. The sample size of data, number of dates used in the
        re-estimation, will increase at a rate controlled by the parameter: the hedge
        ratio will be continuously re-estimated as days pass but will always include all
        realised timestamps (inclusive of the start_date).
        """

        # The method returns a standardised DataFrame. Confirm the first date in the
        # DataFrame is after the minimum observation date. The parameter 'min_obs'
        # ensures a certain number of days have passed until a hedge ratio is calculated.

        # Analysis completed using a single cross-section from the panel.
        cross_section: str = "KRW"
        xr: pd.Series = (self.dfp_w[cross_section]).astype(dtype=np.float32)
        # Adjusts for the effect of pivoting.
        xr = xr.dropna(axis=0, how="all")

        br: pd.Series = pd.Series(
            data=self.benchmark_df["value"].to_numpy(),
            index=self.benchmark_df["real_date"],
            name="BR",
        ).astype(dtype=np.float16)

        # Apply the .date_alignment() method to establish the start & end date of the
        # re-estimation date series. Confirms the re-estimation frequency has been
        # correctly applied.
        # The frequency tested on will be monthly: business month end frequency.
        min_observation: int = 50
        MAX_OBS: int = 100

        start_date: pd.Timestamp
        end_date: pd.Timestamp
        start_date, end_date = date_alignment(unhedged_return=xr, benchmark_return=br)
        freq = _map_to_business_day_frequency("m")
        dates_re: List[pd.Timestamp] = pd.date_range(
            start=start_date + pd.offsets.BDay(min_observation),
            end=end_date,
            freq=freq,
        ).tolist()

        # Produce daily business day date series to determine the date that corresponds
        # to the specified minimum observation.
        test_min_obs: np.datetime64 = pd.date_range(
            start=start_date, end=end_date, freq="B"
        ).to_numpy()[60]

        df_hr: pd.DataFrame = hedge_calculator(
            unhedged_return=xr,
            benchmark_return=br,
            rdates=dates_re,
            meth=method,
            min_obs=min_observation,
            max_obs=MAX_OBS,
        )
        # Confirm the first computed hedge ratio value falls after the minimum
        # observation date.
        test_date: pd.Timestamp = df_hr["real_date"].iloc[0]
        self.assertTrue(test_date > test_min_obs)

        # In the example, the hedge ratio is computed monthly - last business day of the
        # respective month. Therefore, assuming the minimum observation date does not
        # fall on the final day of the month, the first date recorded in the returned
        # DataFrame should be the final business date of the same month as the minimum
        # observation date. For instance, 23/03/2010 -> 30/03/2010.
        # Test both dates are defined during the same month.
        test_min_obs_month = pd.Timestamp(test_min_obs).month
        self.assertTrue(test_min_obs_month == test_date.month)

        # The re-estimated hedge ratio will be computed using realised data up until the
        # respective date (in the above example, the final business day of the month).
        # However, the hedge ratio is applied from the following date (the date the
        # position will change) and to all the intermediary dates up until the next
        # re-estimation date. Therefore, confirm that the first date in the DataFrame is
        # a np.nan value representing the shift mechanism.
        first_value = df_hr["value"].iloc[0]
        self.assertTrue(math.isnan(first_value))

        # The examination of the hedging mechanism will come through graphical
        # interpretation.
        # However, test the computed hedge ratio on a "random" re-estimation date to
        # confirm both return series up until the respective date have been used.
        # Test date is '2013-03-29'.

        s_date, e_date = date_alignment(unhedged_return=xr, benchmark_return=br)
        xr = xr.truncate(before=s_date, after=e_date)
        br = br.truncate(before=s_date, after=e_date)

        data_column = np.empty(len(dates_re))
        data_column[:] = np.nan
        df_hrat = pd.DataFrame(data=data_column, index=dates_re, columns=["value"])

        for d in dates_re:
            # Inclusive of the re-estimation date.
            yvar = xr.loc[:d].values[-MAX_OBS:]
            xvar = br.loc[:d].values[-MAX_OBS:].reshape(-1, 1)

            if method == "ols":
                weights = np.ones_like(yvar)
            elif method == "twls":
                weights = np.power(2, -np.arange(yvar.shape[0]) / 11)[::-1]

            betas = weighted_least_squares(
                X=np.column_stack((np.ones(xvar.shape[0]), xvar)),
                y=yvar,
                weights=weights,
            )

            df_hrat.loc[d] = betas[1]

        df_hrat = df_hrat.dropna(axis=0, how="all")
        df_hrat.index.name = "real_date"
        df_hrat = df_hrat.reset_index(level=0)

        # Test on the next business day given the shift. The hedge ratio computed on the
        # re-estimation date is applied to the return series on the next business day
        # after re-estimation. NOTE : 30,31-Mar-2013 are weekend dates.

        # therefore, `last_test_date` is set hard-coded to '2013-03-29'.
        last_test_date: str = "2013-03-29"
        # check_date <- one business day after the re-estimation date - 2013-04-01.
        check_date: str = "2013-04-01"
        test_value = float(df_hr[df_hr["real_date"] == check_date]["value"].iloc[0])
        result = float(df_hrat[df_hrat["real_date"] == last_test_date]["value"].iloc[0])
        self.assertTrue(abs(result - test_value) < 1e-9)

    def test_adjusted_returns(self):
        """
        Method used to compute the hedge ratio returns. The hedge ratio will determine
        the position taken in the benchmark asset. Therefore, adjust the returns across
        the panel to account for the short position taken the hedging asset (proportional
        to the computed sensitivity parameter between the cross-section and the
        benchmark). A simple example of the formula is:
        IDR_FXXR_NSA_H = IDR_FXXR_NSA - HR_IDR * USD_EQXR_NSA.
        """

        br = pd.Series(
            data=self.benchmark_df["value"].to_numpy(),
            index=self.benchmark_df["real_date"],
        )
        br = br.astype(dtype=np.float64)

        # The method, adjusted_returns(), will compute the hedged return across the
        # entire panel. Call hedge_ratio method and pass the returned DataFrame
        # separately into adjusted_returns() method.
        # The adjusted_returns() method will be called inside the main hedge_ratio()
        # subroutine if a parameter is set to set to True.

        br_cat = "USD_EQXR_NSA"
        # Standardised dataframe consisting of exclusively the hedge-ratios.
        df_hedge = return_beta(
            df=self.dfd,
            xcat="FXXR_NSA",
            cids=self.cids,
            benchmark_return=br_cat,
            start="2010-01-01",
            end="2020-10-30",
            blacklist=self.blacklist,
            meth="ols",
            oos=True,
            refreq="m",
            min_obs=60,
            hedged_returns=False,
        )

        dfw = self.unhedged_df.pivot(index="real_date", columns="cid", values="value")

        # Standardised dataframe of the adjusted returns.
        df_stack = adjusted_returns(benchmark_return=br, df_hedge=df_hedge, dfw=dfw)

        # Choose a "random" date and confirm the values of two cross-sections through
        # manuel calculation.
        # "Random" date is "2016-06-01"
        dates = list(dfw.index)
        date = dfw.index[len(dates) // 2]

        test_date = df_stack[df_stack["real_date"] == date]
        # Test on the two cross-sections: 'IDR' & 'INR'.
        # Hedge Return.
        INR_HR = float(test_date[test_date["cid"] == "INR"]["value"].iloc[0])
        IDR_HR = float(test_date[test_date["cid"] == "IDR"]["value"].iloc[0])

        hedge_row = df_hedge[df_hedge["real_date"] == date]
        INR_H = float(hedge_row[hedge_row["cid"] == "INR"]["value"].iloc[0])
        IDR_H = float(hedge_row[hedge_row["cid"] == "IDR"]["value"].iloc[0])

        return_row = dfw.loc[date]
        INR_R = return_row["INR"]
        IDR_R = return_row["IDR"]

        br_date = br.loc[date]

        # Manual calculation.
        INR_return = INR_R - (INR_H * br_date)
        IDR_return = IDR_R - (IDR_H * br_date)

        self.assertTrue(INR_return == INR_HR)
        self.assertTrue(IDR_return == IDR_HR)

    @parameterized.expand(["ols", "twls"])
    def test_hedge_ratio(self, meth):
        """
        Estimates hedge ratios with respect to a hedge benchmark. The subroutine also
        allows for returning hedged returns if the respective parameter is set to True.
        The method will primarily test the workflow of the function: the logic & source
        code have been covered by previous Unit Tests.
        As stated, the main function of the method is to test the efficacy of the
        assert statements included and the workflow of the main driver function.
        """

        br_cat = "USD_EQXR_NSA"

        with self.assertRaises(ValueError):
            # The categories the respective DataFrame is defined over are
            # ['FXXR_NSA', 'GROWTHXR_NSA', 'INFLXR_NSA', 'EQXR_NSA']. Therefore, choosing
            # a benchmark of USD intuitive GDP growth will throw an error given the
            # ticker is not in the database.
            test_br = "USD_INTRGDP_NSA"
            df_hedge = return_beta(
                df=self.dfd,
                xcat="FXXR_NSA",
                cids=self.cids,
                benchmark_return=test_br,
                start="2010-01-01",
                end="2020-10-30",
                blacklist=self.blacklist,
                meth="ols",
                oos=True,
                refreq="m",
                min_obs=60,
                hedged_returns=False,
            )

        # Test the re-estimation frequency parameter.
        with self.assertRaises(ValueError):
            # The re-estimation frequency can either be weekly, monthly or quarterly:
            # ['w', 'm', 'q']. Set the 'refreq' parameter to an incorrect value.
            df_hedge = return_beta(
                df=self.dfd,
                xcat="FXXR_NSA",
                cids=self.cids,
                benchmark_return=br_cat,
                start="2010-01-01",
                end="2020-10-30",
                blacklist=self.blacklist,
                meth="ols",
                oos=True,
                refreq="b",
                min_obs=60,
                hedged_returns=False,
            )

        # The default number of minimum observations required to compute a hedge ratio is
        # 24. However, if the parameter is defined, the specified number must be greater
        # than 10 business days, two weeks.
        with self.assertRaises(ValueError):
            df_hedge = return_beta(
                df=self.dfd,
                xcat="FXXR_NSA",
                cids=self.cids,
                benchmark_return=br_cat,
                start="2010-01-01",
                end="2020-10-30",
                blacklist=self.blacklist,
                meth="ols",
                oos=True,
                refreq="w",
                min_obs=8,
                hedged_returns=False,
            )

        # Confirm the re-estimation frequency parameter is working correctly. Test on
        # weekly data where the final day of the week will invariably be the Friday. The
        # new re-estimation value should be applied from the next business day, the
        # following Monday.
        df_hedge = return_beta(
            df=self.dfd,
            xcat="FXXR_NSA",
            cids=self.cids,
            benchmark_return=br_cat,
            start="2010-01-01",
            end="2020-10-30",
            blacklist=self.blacklist,
            meth=meth,
            oos=True,
            refreq="w",
            min_obs=24,
            hedged_returns=False,
        )
        # Confirm on a single cross-section.
        df_hedge_INR = df_hedge[df_hedge["cid"] == "INR"]

        # Test on a random date, 2014-02-14. The date should be a Friday.
        date = pd.Timestamp("2014-02-14")
        self.assertTrue(pd.Timestamp(date).dayofweek == 4)
        df_hedge_INR_val = (df_hedge_INR[df_hedge_INR["real_date"] == date])[
            "value"
        ].iloc[0]
        df_hedge_INR_val = float(df_hedge_INR_val)

        # Confirm the date in the DataFrame is a Monday and the hedge ratio is
        # re-estimated on the respective date.

        index = np.where(df_hedge_INR["real_date"] == date)
        index = next(iter(index))[0]
        next_index = index + 1
        test_row = df_hedge_INR.iloc[next_index]
        test_date = test_row["real_date"]

        self.assertTrue(pd.Timestamp(test_date).dayofweek == 0)
        test_value = test_row["value"]
        self.assertTrue(test_value != df_hedge_INR_val)


class TestHedgeCalculator:
    """Tests for the hege_calculator function"""
    @staticmethod
    def make_series():
        index = pd.date_range("2020-01-01",  periods=500, freq="D", name="real_date")
        data = np.random.randn(500)
        return pd.Series(data, index=index)

    def test_invalid_inputs(self):
        # Method must be ols or twls
        ur = self.make_series()
        br = self.make_series()
        rdates = [pd.Timestamp("2020-06-01"), pd.Timestamp("2021-01-01")]
        with pytest.raises(ValueError, match="meth must be"):
            hedge_calculator(ur, br, rdates, meth="ridge")

        # min_obs must be positive
        ur = self.make_series()
        br = self.make_series()
        rdates = [pd.Timestamp("2020-06-01"), pd.Timestamp("2021-01-01")]
        with pytest.raises(ValueError, match="min_obs must be"):
            hedge_calculator(ur, br, rdates, min_obs=0)

        # max_obs cannot be less than min_obs
        ur = self.make_series()
        br = self.make_series()
        with pytest.raises(ValueError, match="max_obs"):
            hedge_calculator(ur, br, rdates, min_obs=20, max_obs=10)

        # rdate passed that doesn't satisfy min_obs
        ur = self.make_series()
        br = self.make_series()
        rdates = [pd.Timestamp("2020-02-01"), pd.Timestamp("2021-01-01")]
        with pytest.raises(ValueError, match="Re-estimation dates"):
            hedge_calculator(ur, br, rdates, min_obs=40)

    def test_insufficient_overlap_raises(self):
        """Fewer overlapping observations than min_obs raises a ValueError."""
        idx = pd.bdate_range("2020-01-01", periods=5, name="real_date")
        ur = pd.Series(np.random.randn(5), index=idx)
        br = pd.Series(np.random.randn(5), index=idx)
        with pytest.raises(ValueError, match="overlapping observations"):
            hedge_calculator(ur, br, [pd.Timestamp("2020-01-07")], min_obs=24)

    @pytest.mark.parametrize("method", ["ols", "twls"])
    def test_valid_runs(self, method):
        ur = self.make_series()
        br = self.make_series()
        rdates = [pd.Timestamp("2020-06-01"), pd.Timestamp("2021-01-01"), pd.Timestamp("2021-06-01")]
        result = hedge_calculator(ur, br, rdates, meth=method)
        assert not result.empty

    @pytest.mark.parametrize("method", ["ols", "twls"])
    def test_perfect_correlation_ratio_near_one(self, method):
        ur = self.make_series()
        br = ur.copy()
        rdates = [pd.Timestamp("2020-06-01"), pd.Timestamp("2021-01-01"), pd.Timestamp("2021-06-01")]
        result = hedge_calculator(ur, br, rdates, meth=method)

        assert np.allclose(result["value"].values[1:], 1.0, atol=1e-6)

    @pytest.mark.parametrize("method", ["ols", "twls"])
    def test_scaled_perfect_correlation(self, method):
        br = self.make_series()
        ur = 2 * br
        rdates = [pd.Timestamp("2020-06-01"), pd.Timestamp("2021-01-01"), pd.Timestamp("2021-06-01")]
        result = hedge_calculator(ur, br, rdates, meth=method)

        assert np.allclose(result["value"].values[1:], 2, atol=1e-4)

    def test_hedge_ratio_is_shifted_by_one(self):
        """
        The hedge ratio calculated at rdate T should appear on the row T+1
        """
        ur = self.make_series()
        br = self.make_series()
        # Single re-estimation date
        rdate = ur.index[30]
        result = hedge_calculator(ur, br, [rdate], meth="ols", min_obs=24)

        assert np.isnan(result["value"][0].item())

    def test_misaligned_series(self):
        """Series with different lengths should be aligned on intersection."""
        dates_long = pd.bdate_range("2020-01-01", periods=120, name="real_date")
        dates_short = pd.bdate_range("2020-03-01", periods=80, name="real_date")
        ur = pd.Series(np.random.randn(120), index=dates_long)
        br = pd.Series(np.random.randn(80), index=dates_short)

        common_dates = dates_long.intersection(dates_short)
        br[common_dates] = ur[common_dates]

        rdates = [pd.Timestamp("2020-04-01"), pd.Timestamp("2020-05-01")]

        result = hedge_calculator(ur, br, rdates, min_obs=15)

        assert not result.empty
        assert result["real_date"].min() == pd.Timestamp("2020-04-01")
        assert result["real_date"].max() == dates_long.max()
        assert np.allclose(result["value"].values[1:], 1.0, atol=1e-6)


class TestReturnBeta:
    """Tests for the return_beta function."""

    @staticmethod
    def make_qdf_df(
        cids=("AUD", "GBP", "USD"),
        xcats=("FXXR_NSA", "EQXR_NSA"),
        earliest="2010-01-01",
        latest="2020-12-31",
    ):
        df_cids = pd.DataFrame(
            index=list(cids), columns=["earliest", "latest", "mean_add", "sd_mult"]
        )
        for cid in cids:
            df_cids.loc[cid] = [earliest, latest, 0, 1]

        df_xcats = pd.DataFrame(
            index=list(xcats),
            columns=["earliest", "latest", "mean_add", "sd_mult", "ar_coef", "back_coef"],
        )
        for xcat in xcats:
            df_xcats.loc[xcat] = [earliest, latest, 0, 1, 0, 0.2]

        return make_qdf(df_cids, df_xcats)

    def test_returns_standardised_quantamental_dataframe(self):
        """Output is a standardised QDF whose only category is ``xcat + ratio_name``."""
        cids = ["AUD", "GBP", "USD"]
        dfd = self.make_qdf_df(cids)

        result = return_beta(
            df=dfd,
            xcat="FXXR_NSA",
            cids=cids,
            benchmark_return="USD_EQXR_NSA",
            refreq="m",
            min_obs=24,
        )

        assert list(result.columns) == ["real_date", "cid", "xcat", "value"]
        assert list(result["xcat"].unique()) == ["FXXR_NSA_HR"]
        assert sorted(result["cid"].unique()) == cids

    def test_unknown_benchmark_raises(self):
        """A benchmark ticker absent from the DataFrame raises a ValueError."""
        dfd = self.make_qdf_df()
        with pytest.raises(ValueError, match="Benchmark return ticker"):
            return_beta(
                df=dfd,
                xcat="FXXR_NSA",
                cids=["AUD", "GBP", "USD"],
                benchmark_return="USD_INTRGDP_NSA",
                refreq="m",
            )

    def test_xcat_not_in_dataframe_raises(self):
        """A category that is not present in the DataFrame raises a ValueError."""
        dfd = self.make_qdf_df()
        with pytest.raises(ValueError, match="not defined in the dataframe"):
            return_beta(
                df=dfd,
                xcat="GROWTHXR_NSA",
                cids=["AUD", "GBP", "USD"],
                benchmark_return="USD_EQXR_NSA",
                refreq="m",
            )

    def test_min_obs_below_floor_raises(self):
        dfd = self.make_qdf_df()
        with pytest.raises(ValueError, match="minimum observations"):
            return_beta(
                df=dfd,
                xcat="FXXR_NSA",
                cids=["AUD", "GBP", "USD"],
                benchmark_return="USD_EQXR_NSA",
                refreq="m",
                min_obs=9,
            )

    def test_max_obs_below_min_obs_raises(self):
        dfd = self.make_qdf_df()
        with pytest.raises(ValueError, match="max_obs"):
            return_beta(
                df=dfd,
                xcat="FXXR_NSA",
                cids=["AUD", "GBP", "USD"],
                benchmark_return="USD_EQXR_NSA",
                refreq="m",
                min_obs=24,
                max_obs=10,
            )

    def test_invalid_refreq_raises(self):
        dfd = self.make_qdf_df()
        with pytest.raises(ValueError):
            return_beta(
                df=dfd,
                xcat="FXXR_NSA",
                cids=["AUD", "GBP", "USD"],
                benchmark_return="USD_EQXR_NSA",
                refreq="b",
                min_obs=24,
            )

    def test_hedged_returns_appended(self):
        dfd = self.make_qdf_df()
        result = return_beta(
            df=dfd,
            xcat="FXXR_NSA",
            cids=["AUD", "GBP", "USD"],
            benchmark_return="USD_EQXR_NSA",
            refreq="m",
            min_obs=24,
            hedged_returns=True,
        )

        assert sorted(result["xcat"].unique()) == ["FXXR_NSA_H", "FXXR_NSA_HR"]

    def test_custom_ratio_and_hedge_labels(self):
        dfd = self.make_qdf_df()
        result = return_beta(
            df=dfd,
            xcat="FXXR_NSA",
            cids=["AUD", "GBP", "USD"],
            benchmark_return="USD_EQXR_NSA",
            refreq="m",
            min_obs=24,
            hedged_returns=True,
            ratio_name="_BETA",
            hr_name="HEDGED",
        )

        assert sorted(result["xcat"].unique()) == ["FXXR_NSA_BETA", "FXXR_NSA_HEDGED"]

    def test_benchmark_cross_section_removed_from_panel(self):
        """
        When the hedged category is the benchmark's category, the benchmark's own
        cross-section is dropped from the panel and a warning is issued.
        """
        dfd = self.make_qdf_df()
        with pytest.warns(UserWarning, match="has been removed from the panel"):
            result = return_beta(
                df=dfd,
                xcat="EQXR_NSA",
                cids=["AUD", "GBP", "USD"],
                benchmark_return="USD_EQXR_NSA",
                refreq="m",
                min_obs=24,
            )

        assert sorted(result["cid"].unique()) == ["AUD", "GBP"]

    def test_cids_subset_respected(self):
        """Only the requested cross-sections appear in the output."""
        dfd = self.make_qdf_df()
        result = return_beta(
            df=dfd,
            xcat="FXXR_NSA",
            cids=["AUD"],
            benchmark_return="USD_EQXR_NSA",
            refreq="m",
            min_obs=24,
        )

        assert sorted(result["cid"].unique()) == ["AUD"]

    @parameterized.expand(["ols", "twls"])
    def test_estimation_methods_produce_ratios(self, meth):
        dfd = self.make_qdf_df()
        result = return_beta(
            df=dfd,
            xcat="FXXR_NSA",
            cids=["AUD", "GBP", "USD"],
            benchmark_return="USD_EQXR_NSA",
            refreq="m",
            min_obs=24,
            meth=meth,
        )

        assert not result.empty
        assert result["value"].notna().any()

    def test_some_cids_dont_satisfy_min_obs(self):
        dfd = self.make_qdf_df()
        dfd.loc[dfd["cid"].eq("AUD"), "value"] = np.nan

        with pytest.warns(UserWarning, match="Cannot calculate beta for the"):
            result = return_beta(
                df=dfd,
                xcat="FXXR_NSA",
                cids=["AUD", "GBP"],
                benchmark_return="USD_EQXR_NSA",
                refreq="m",
                min_obs=24,
                meth="twls",
            )

        assert sorted(result["cid"].unique()) == ["GBP"]

    def test_all_cids_dont_satisfy_min_obs(self):
        dfd = self.make_qdf_df()
        dfd = dfd.groupby(["cid", "xcat"], as_index=False).head(23)

        with pytest.raises(RuntimeError, match="None of"):
            return_beta(
                df=dfd,
                xcat="FXXR_NSA",
                cids=["AUD", "GBP"],
                benchmark_return="USD_EQXR_NSA",
                refreq="m",
                min_obs=24,
                meth="twls",
            )

    def test_cid_satisfied_only_after_last_rdate(self):
        """
        A cross-section that reaches min_obs only after the final re-estimation
        date (panel ends mid-period) is warned-and-skipped rather than crashing the
        run for the satisfied cross-sections.
        """
        dfd = self.make_qdf_df(latest="2021-01-15")
        late_start = (
            dfd["cid"].eq("AUD")
            & dfd["xcat"].eq("FXXR_NSA")
            & dfd["real_date"].lt(pd.Timestamp("2021-01-04"))
        )
        dfd.loc[late_start, "value"] = np.nan

        with pytest.warns(UserWarning, match="Cannot calculate beta"):
            result = return_beta(
                df=dfd,
                xcat="FXXR_NSA",
                cids=["AUD", "GBP", "USD"],
                benchmark_return="USD_EQXR_NSA",
                refreq="m",
                min_obs=10,
            )

        assert sorted(result["cid"].unique()) == ["GBP", "USD"]


class TestWeightedLeastSquares:
    """Tests for the weighted_least_squares function."""

    def test_uniform_weights_match_ols(self):
        """With equal weights, WLS should match ordinary least squares."""
        beta_true = np.array([3.0, -2.0])

        X = np.column_stack([np.ones(50), np.random.randn(50)])
        y = X @ beta_true + 0.01 * np.random.randn(50)

        beta_wls = weighted_least_squares(X, y, np.ones(50))
        beta_ols, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

        np.testing.assert_allclose(beta_wls, beta_ols, atol=1e-10)

    def test_recovers_exact_coefficients_no_noise(self):
        """With no noise the solver should recover the true coefficients exactly."""
        beta_true = np.array([5.0, -3.0])

        X = np.column_stack([np.ones(20), np.linspace(0, 1, 20)])
        y = X @ beta_true
        weights = np.random.uniform(0.5, 2.0, size=20)

        beta_wls = weighted_least_squares(X, y, weights)
        np.testing.assert_allclose(beta_wls, beta_true, atol=1e-10)

    def test_scaling_weights_does_not_change_result(self):
        """Multiplying all weights by a constant shouldn't change beta."""

        X = np.column_stack([np.ones(30), np.random.randn(30)])
        y = np.random.randn(30)
        weights = np.random.uniform(0.1, 5.0, size=30)

        beta1 = weighted_least_squares(X, y, weights)
        beta2 = weighted_least_squares(X, y, weights * 42.0)

        np.testing.assert_allclose(beta1, beta2, atol=1e-10)

    def test_known_solution(self):
        """
        Hand-verifiable 3-observation, 2-predictor (intercept + slope) case.
        """
        X = np.array([[1.0, 0.0],
                      [1.0, 1.0],
                      [1.0, 2.0]])
        y = np.array([1.0, 3.0, 2.0])
        weights = np.array([1.0, 2.0, 1.0])

        beta = weighted_least_squares(X, y, weights)
        beta_expected = [1.75, 0.5]
        np.testing.assert_allclose(beta, beta_expected, atol=1e-10)

    def test_ols_against_sklearn(self):
        beta_true = np.array([1.0, 2.0, 3.0, -2.0, 4.0])
        n_samples = 100
        X = np.random.randn(n_samples, 4)
        X = np.column_stack((np.ones(n_samples), X))
        y = X @ beta_true + 0.01 * np.random.randn(n_samples)


        result = weighted_least_squares(X, y, np.ones(100))
        expected = LinearRegression(fit_intercept=False).fit(X, y).coef_

        assert np.allclose(result, expected, atol=1e-5)

    def test_wls_against_sklearn(self):
        beta_true = np.array([1.0, 2.0, 3.0, -2.0, 4.0])
        n_samples = 100
        X = np.random.randn(n_samples, 4)
        X = np.column_stack((np.ones(n_samples), X))
        y = X @ beta_true + 0.01 * np.random.randn(n_samples)

        weights = np.random.uniform(0.5, 2.0, size=n_samples)

        result = weighted_least_squares(X, y, weights)
        expected = LinearRegression(fit_intercept=False).fit(X, y, weights).coef_

        assert np.allclose(result, expected, atol=1e-5)

    def test_against_msl_twls(self):
        # create data
        index = pd.MultiIndex.from_product([["CAD"], pd.date_range("2020-01-01", periods=100)])
        x_data = np.column_stack((np.ones(100), np.random.randn(100, 4)))
        X = pd.DataFrame(index=index, data=x_data)
        y = pd.Series(np.random.randn(100), index=index)

        # fit twls model (fit_intercept = False because we have a 1s col in X)
        twls_model = TimeWeightedLinearRegression(fit_intercept=False).fit(X, y)
        twls_coefs = twls_model.coef_

        # fit pure numpy implementation using same weights
        twls_model_weights = twls_model._calculate_time_weights(y)
        result = weighted_least_squares(X.values, y.values, twls_model_weights)

        assert np.allclose(result, twls_coefs, atol=1e-5)



if __name__ == "__main__":
    unittest.main()
