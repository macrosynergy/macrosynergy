from tests.simulate import make_qdf
from macrosynergy.pnl.naive_pnl import NaivePnL, create_results_dataframe
from macrosynergy.management.utils import reduce_df
import re
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from typing import List, Dict
import matplotlib
from matplotlib import pyplot as plt
from macrosynergy.management.types import QuantamentalDataFrame


class TestAll(unittest.TestCase):
    def setUp(self) -> None:
        self.cids: List[str] = ["AUD", "CAD", "GBP", "NZD", "USD", "EUR"]
        self.xcats: List[str] = ["EQXR", "CRY", "GROWTH", "INFL", "DUXR"]

        df_cids = pd.DataFrame(
            index=self.cids, columns=["earliest", "latest", "mean_add", "sd_mult"]
        )
        df_cids.loc["AUD", :] = ["2008-01-03", "2020-12-31", 0.5, 2]
        df_cids.loc["CAD", :] = ["2010-01-03", "2020-11-30", 0, 1]
        df_cids.loc["GBP", :] = ["2012-01-03", "2020-11-30", -0.2, 0.5]
        df_cids.loc["NZD"] = ["2002-01-03", "2020-09-30", -0.1, 2]
        df_cids.loc["USD"] = ["2015-01-03", "2020-12-31", 0.2, 2]
        df_cids.loc["EUR"] = ["2008-01-03", "2020-12-31", 0.1, 2]

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

        df_xcats.loc["EQXR"] = ["2005-01-03", "2020-12-31", 0.1, 1, 0, 0.3]
        df_xcats.loc["CRY"] = ["2000-01-01", "2020-10-30", 1, 2, 0.95, 1]
        df_xcats.loc["GROWTH"] = ["2010-01-03", "2020-10-30", 1, 2, 0.9, 1]
        df_xcats.loc["INFL"] = ["2001-01-01", "2020-10-30", 1, 2, 0.8, 0.5]
        df_xcats.loc["DUXR"] = ["2000-01-01", "2020-12-31", 0.1, 0.5, 0, 0.1]

        black = {
            "AUD": ["2000-01-01", "2003-12-31"],
            "GBP": ["2018-01-01", "2100-01-01"],
        }
        self.blacklist: Dict[str, List[str]] = black

        # Standard df for tests.
        dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)
        self.dfd: pd.DataFrame = reduce_df(dfd, blacklist=self.blacklist)

    def tearDown(self) -> None:
        return super().tearDown()

    def test_constructor(self):
        # Test NaivePnL's constructor and the instantiation of the respective fields.

        ret = ["EQXR"]
        sigs = ["CRY", "GROWTH", "INFL"]
        pnl = NaivePnL(
            self.dfd,
            ret=ret[0],
            sigs=sigs,
            cids=self.cids,
            start="2000-01-01",
            blacklist=self.blacklist,
        )
        # Confirm the categories held in the reduced DataFrame, on the instance's field,
        # are exclusively the return and signal category. This will occur if benchmarks
        # have not been defined.
        test_categories = list(pnl.df["xcat"].unique())
        self.assertTrue(sorted(test_categories) == sorted(ret + sigs))

        # Add "external" benchmarks to the instance: a category that is neither the
        # return field or one of the categories. The benchmarks will be added to the
        # instance's DataFrame.
        pnl = NaivePnL(
            self.dfd,
            ret=ret[0],
            sigs=sigs,
            cids=self.cids,
            start="2000-01-01",
            blacklist=self.blacklist,
            bms=["EUR_DUXR", "USD_DUXR"],
        )
        test_categories = list(pnl.df["xcat"].unique())
        self.assertTrue(sorted(test_categories) == sorted(ret + sigs + ["DUXR"]))

        # Test that both the benchmarks are held in the DataFrame. Implicitly validating
        # that add_bm() method works correctly. The benchmark series will be appended to
        # the DataFrame held on the instance: confirm their presence.
        first_bm = pnl.df[(pnl.df["cid"] == "EUR") & (pnl.df["xcat"] == "DUXR")]
        self.assertTrue(not first_bm.empty)
        second_bm = pnl.df[(pnl.df["cid"] == "USD") & (pnl.df["xcat"] == "DUXR")]
        self.assertTrue(not second_bm.empty)

        # Additionally, confirm that the benchmark dictionary has been populated
        # correctly as both benchmarks are present in the passed DataFrame.
        bm_tickers = list(pnl._bm_dict.keys())
        self.assertTrue(sorted(bm_tickers) == ["EUR_DUXR", "USD_DUXR"])

        # Confirm the values are correct. Confirm the values in each benchmark series
        # have been correctly lifted from the original, standardised DataFrame.
        eur_duxr = self.dfd[(self.dfd["cid"] == "EUR") & (self.dfd["xcat"] == "DUXR")]

        self.assertTrue(
            np.all(first_bm["value"].to_numpy() == eur_duxr["value"].to_numpy())
        )

        self.assertTrue(
            np.all(
                np.squeeze(pnl._bm_dict["EUR_DUXR"].to_numpy())
                == eur_duxr["value"].to_numpy()
            )
        )

        # Confirm the benchmark functionality works when passing in a single ticker.
        # Also, the benchmark will already be present on the instance's DataFrame.
        pnl = NaivePnL(
            self.dfd,
            ret=ret[0],
            sigs=sigs,
            cids=self.cids,
            start="2000-01-01",
            blacklist=self.blacklist,
            bms="EUR_EQXR",
        )
        bm_tickers = list(pnl._bm_dict.keys())
        self.assertTrue(sorted(bm_tickers) == ["EUR_EQXR"])

    def test_make_signal(self):
        df = self.dfd

        ret = "EQXR"
        sigs = ["CRY", "GROWTH", "INFL"]
        pnl = NaivePnL(
            self.dfd,
            ret=ret,
            sigs=sigs,
            cids=self.cids,
            start="2000-01-01",
            blacklist=self.blacklist,
            bms=["EUR_DUXR", "USD_DUXR"],
        )

        # Test the method used for producing the signals. The signal is based on a single
        # category and the function allows for applying transformations to the signal to
        # determine the extent of the position.
        # For instance, distance from the neutral level measured in standard deviations.
        # Or a digital transformation: if the signal category is positive, take a unitary
        # long position.

        # Specifically chosen signal that will have leading NaN values. To test if
        # functionality incorrectly populates unrealised dates.
        sig = "GROWTH"
        dfx = df[df["xcat"].isin([ret, sig])]

        # Adjust for any blacklist periods.
        dfx = reduce_df(
            df=dfx,
            xcats=[ret, sig],
            cids=self.cids,
            blacklist=self.blacklist,
            out_all=False,
        )

        # Will return a DataFrame with the transformed signal.
        dfw = pnl._make_signal(
            dfx=dfx,
            sig=sig,
            sig_op="zn_score_pan",
            min_obs=252,
            iis=True,
            sequential=True,
            neutral="zero",
            thresh=None,
        )
        self.__dict__["signal_dfw"] = dfw

        # Confirm the first dates for each cross-section's signal are the expected start
        # dates. There are not any falsified signals being created. The signal is
        # 'GROWTH'.
        # Dates have been adjusted for the first business day.
        expected_start = {
            "AUD": "2010-01-04",
            "CAD": "2010-01-04",
            "GBP": "2012-01-03",
            "NZD": "2010-01-04",
            "USD": "2015-01-05",
            "EUR": "2010-01-04",
        }
        signal_column = dfw["psig"]
        signal_column = signal_column.reset_index()
        signal_column = signal_column.rename(columns={"psig": "value"})
        signal_column["xcat"] = "psig"

        dfw_signal = signal_column.pivot(
            index="real_date", columns="cid", values="value"
        )
        cross_sections = dfw_signal.columns
        # Confirms make_zn_scores does not produce any signals for non-realised dates.
        for c in cross_sections:
            column = dfw_signal.loc[:, c]
            self.assertTrue(
                column.first_valid_index() == pd.Timestamp(expected_start[c])
            )

    @staticmethod
    def diff_month(d1, d2):
        return (d1.year - d2.year) * 12 + d1.month - d2.month

    def test_rebalancing_dates(self):
        self.test_make_signal()

        dfw = self.signal_dfw
        dfw.reset_index(inplace=True)
        dfw = dfw.rename_axis(None, axis=1)

        dfw = dfw.sort_values(["cid", "real_date"])

        sig_series = NaivePnL.rebalancing(dfw, rebal_freq="monthly")
        dfw["sig"] = np.squeeze(sig_series.to_numpy())

        dfw_signal_rebal = dfw.pivot(index="real_date", columns="cid", values="sig")

        # Confirm, on a single cross-section that re-balancing occurs on a monthly basis.
        # The number of unique values will equate to the number of months in the
        # time-series.
        dfw_signal_rebal_aud = dfw_signal_rebal.loc[:, "AUD"]
        aud_array = np.squeeze(dfw_signal_rebal_aud.to_numpy())
        unique_values_aud = set(aud_array)

        start_date = dfw_signal_rebal.index[0]
        end_date = dfw_signal_rebal.index[-1]

        no_months = self.diff_month(end_date, start_date)

        self.assertTrue(no_months - 1 == len(unique_values_aud))

    def test_make_pnl(self):
        self.test_make_signal()

        # Signal is produced daily. The calculation of the neutral level and standard
        # deviation are also calculated daily. Only for a highly volatile asset would
        # there be any value in calculating at a daily frequency. In most instances, the
        # neutral level would remain fairly constant over the duration of a week.
        dfw = self.signal_dfw

        # The PnL DataFrame is appended to the instance DataFrame.
        ret = "EQXR"
        sigs = ["CRY", "GROWTH", "INFL"]
        pnl = NaivePnL(
            self.dfd,
            ret=ret,
            sigs=sigs,
            cids=self.cids,
            start="2000-01-01",
            blacklist=self.blacklist,
            bms=["EUR_DUXR", "USD_DUXR"],
        )

        pnl.make_pnl(
            sig="GROWTH",
            sig_op="zn_score_pan",
            rebal_freq="daily",
            vol_scale=None,
            rebal_slip=0,
            pnl_name="PNL_GROWTH",
            min_obs=252,
            iis=True,
            sequential=True,
            neutral="zero",
            thresh=None,
        )

        # Test the PnL value produced across the panel. Implicitly tests the PnL values
        # for each cross-section.
        # Confirms the category for the PnL is being named correctly.
        pnl_df = pnl.df[pnl.df["xcat"] == "PNL_GROWTH"]

        # Confirm the first PnL value for each cross-section is aligned to the first date
        # of the respective signal, GROWTH. A PnL value should only be produced if the
        # signal is available for the respective date.
        expected_start = {
            "AUD": "2010-01-04",
            "CAD": "2010-01-04",
            "GBP": "2012-01-03",
            "NZD": "2010-01-04",
            "USD": "2015-01-05",
            "EUR": "2010-01-04",
        }

        pnl_dfw = pnl_df.pivot(index="real_date", columns="cid", values="value")
        cross_sections = self.cids
        for c in cross_sections:
            column = pnl_dfw.loc[:, c]
            # Adjust the expected start dates by one day to account for the shift
            # mechanism. The computed signal is used for the following day's position.
            self.assertTrue(
                column.first_valid_index()
                == pd.Timestamp(expected_start[c]) + pd.DateOffset(1)
            )

        # Choose a quasi-random sample of dates to confirm the logic of computing the
        # PnL. Multiply each cross-section's signal by their respective return.
        # A "random" sample of dates (will be inclusive of dates where some
        # cross-sections have NaN values and blacklists have been applied.).
        fixed_dates = ["2010-01-13", "2012-01-26", "2015-01-20", "2019-01-08"]

        # Shift the signal by a single date. Replicating the logic in make_pnl().
        dfw["psig"] = dfw["psig"].groupby(level=0).shift(1)
        dfw.reset_index(inplace=True)
        dfw = dfw.rename_axis(None, axis=1)
        dfw = dfw.sort_values(["cid", "real_date"])
        dfw = dfw.rename({"psig": "sig"}, axis=1)

        dfw_sig = dfw.pivot(index="real_date", columns="cid", values="sig")

        # Confirm the logic on a small but representative sample of dates.
        for date in fixed_dates:
            signals = dfw_sig.loc[date, :]
            signal_dict = dict(signals)
            df = self.dfd

            returns = df[(df["xcat"] == ret)]
            returns_dfw = returns.pivot(
                index="real_date", columns="cid", values="value"
            )

            return_dict = dict(returns_dfw.loc[date, :])

            # Aggregate the individual cross-section's PnL to calculate the PnL across
            # the panel (weighted according to the signal).
            pnl_return_date = 0
            condition = lambda a, b: str(a) == "nan" or str(b) == "nan"
            for cid, value in signal_dict.items():
                # Mitigates for NaN values. Exclude from calculation - only sum on
                # realised dates.
                if condition(return_dict[cid], value):
                    pass
                else:
                    pnl_return_date += return_dict[cid] * value

            test_data = pnl_dfw["ALL"].loc[date]
            self.assertTrue(round(float(test_data), 4) == round(pnl_return_date, 4))

    def test_make_pnl_args(self):
        def _random_func(): ...

        ret = "EQXR"
        sigs = ["CRY", "GROWTH", "INFL"]
        pnl = NaivePnL(
            self.dfd,
            ret=ret,
            sigs=sigs,
            cids=self.cids,
            start="2000-01-01",
            blacklist=self.blacklist,
            bms=["EUR_DUXR", "USD_DUXR"],
        )

        argdict = dict(
            sig="GROWTH",
            sig_op="zn_score_pan",
            rebal_freq="daily",
            vol_scale=None,
            leverage=2,
            rebal_slip=0,
            pnl_name="PNL_GROWTH",
            min_obs=252,
            iis=True,
            sequential=True,
            neutral="zero",
            thresh=None,
        )
        for key in argdict:
            argdict_copy = argdict.copy()
            argdict_copy[key] = _random_func
            with self.assertRaises(TypeError):
                pnl.make_pnl(**argdict_copy)

        for argx in ["sig", "sig_op", "rebal_freq", "neutral"]:
            argdict_copy = argdict.copy()  # replace with random string
            argdict_copy[argx] = "random_string"
            with self.assertRaises(ValueError):
                pnl.make_pnl(**argdict_copy)

        for argx in ["thresh", "leverage", "vol_scale"]:
            argdict_copy = argdict.copy()
            argdict_copy[argx] = -1
            with self.assertRaises(ValueError):
                pnl.make_pnl(**argdict_copy)

        argdict = {k: v for k, v in argdict.items() if k in ["vol_scale", "leverage"]}
        argdict["label"] = None
        for argx in ["vol_scale", "leverage"]:
            argdict_copy = argdict.copy()
            argdict_copy[argx] = -1
            with self.assertRaises(ValueError):
                pnl.make_long_pnl(**argdict_copy)
            argdict_copy[argx] = "random_string"
            with self.assertRaises(TypeError):
                pnl.make_long_pnl(**argdict_copy)

    def test_make_pnl_neg(self):
        # The majority of the logic for make_pnl is tested through the method
        # test_make_pnl(). Therefore, aim to isolate the application of the negative
        # signal through evaluate_pnl() method.
        # For make_pnl(), the sig_neg parameter will be set to True and the associated
        # transformed signal will be multiplied by minus one.
        # To test the negative signal, call make_pnl() on the same raw signal but set the
        # sig_neg parameter to True and False. The two produced PnL series should have an
        # inverse relationship with any benchmark.

        ret = "EQXR"
        sigs = ["CRY", "GROWTH", "INFL"]
        bms = ["EUR_DUXR", "USD_DUXR"]

        pnl = NaivePnL(
            self.dfd,
            ret=ret,
            sigs=sigs,
            cids=self.cids,
            start="2000-01-01",
            blacklist=self.blacklist,
            bms=bms,
        )

        # Set the signal to True.
        # Will implicitly test if the PnL name, using the default mechanism, will have
        # the postfix "_NEG" appended given sig_neg is set to True.
        pnl.make_pnl(
            sig="INFL",
            sig_op="zn_score_pan",
            sig_neg=True,
            rebal_freq="monthly",
            vol_scale=5,
            rebal_slip=1,
            min_obs=250,
            thresh=2,
        )

        # Same parameter but sig_neg is set to False.
        pnl.make_pnl(
            sig="INFL",
            sig_op="zn_score_pan",
            sig_neg=False,
            rebal_freq="monthly",
            vol_scale=5,
            rebal_slip=1,
            min_obs=250,
            thresh=2,
        )

        # Confirm the direct negative correlation across the two PnLs. By adding the
        # correlation coefficients with the benchmarks, the value should equate to
        # zero.
        df_eval = pnl.evaluate_pnls(pnl_cats=["PNL_INFL", "PNL_INFL_NEG"])

        bm_correl = df_eval.loc[[b + " correl" for b in bms], :]
        self.assertTrue(np.all(bm_correl.sum(axis=1).to_numpy()) == 0)
        # Default sr_thresholds are [0.25, 0.5, 0.75].
        self.assertIn("Prob. Sharpe Ratio > 0.25", df_eval.index)
        self.assertIn("Prob. Sharpe Ratio > 0.5", df_eval.index)
        self.assertIn("Prob. Sharpe Ratio > 0.75", df_eval.index)

        self.assertIn("Max Draw Recovery (months)", df_eval.index)
        recovery = df_eval.loc["Max Draw Recovery (months)"]
        # Never NaN: 0 if there was no drawdown, Traded Months if the worst
        # drawdown never recovered by the end of the sample.
        self.assertFalse(recovery.isna().any())
        self.assertTrue((recovery >= 0).all())

        df_none = pnl.evaluate_pnls(pnl_cats=["PNL_INFL"], sr_thresholds=[])
        self.assertFalse(
            any(idx.startswith("Prob. Sharpe Ratio >") for idx in df_none.index)
        )

        df_custom = pnl.evaluate_pnls(
            pnl_cats=["PNL_INFL"], sr_thresholds=[0.1, 1.0]
        )
        self.assertIn("Prob. Sharpe Ratio > 0.1", df_custom.index)
        self.assertIn("Prob. Sharpe Ratio > 1", df_custom.index)
        self.assertNotIn("Prob. Sharpe Ratio > 0.25", df_custom.index)
        self.assertTrue(
            df_custom.loc["Prob. Sharpe Ratio > 0.1"].dropna().between(0.0, 1.0).all()
        )

        # test it works with no pnl_cats input
        try:
            pnl.evaluate_pnls()
        except Exception as e:
            self.fail(f"evaluate_pnls raised {e} unexpectedly")

    def test_evaluate_pnls_pretty(self):
        from IPython.display import HTML

        ret = "EQXR"
        sigs = ["CRY", "GROWTH", "INFL"]

        # A single benchmark, so the "Benchmark correlation" placeholder in
        # the default groups resolves to the one real "USD_DUXR correl" row.
        pnl_single_bm = NaivePnL(
            self.dfd,
            ret=ret,
            sigs=sigs,
            cids=self.cids,
            start="2000-01-01",
            blacklist=self.blacklist,
            bms=["USD_DUXR"],
        )
        pnl_single_bm.make_pnl(
            sig="INFL", sig_op="zn_score_pan", rebal_freq="monthly",
            vol_scale=5, min_obs=250, pnl_name="PNL_HEAD",
        )
        pnl_single_bm.make_pnl(
            sig="INFL", sig_op="zn_score_pan", sig_neg=True, rebal_freq="monthly",
            vol_scale=5, min_obs=250, pnl_name="PNL_BENCH",
        )

        result = pnl_single_bm.evaluate_pnls_html(
            headline="PNL_HEAD", bench="PNL_BENCH"
        )
        self.assertIsInstance(result, HTML)
        html = result.data
        # Default groups: 3 performance + 5 risk & drawdown + 5 robustness rows.
        for metric in (
            "Return %",
            "Sharpe Ratio",
            "Sortino Ratio",
            "St. Dev. %",
            "Peak to Trough Draw %",
            "Top 5% Monthly PnL Share",
            "USD_DUXR correl",
            "Max Draw Recovery (months)",
            "Sharpe Stability Ratio",
            "Prob. Sharpe Ratio > 0.25",
            "Prob. Sharpe Ratio > 0.5",
            "Prob. Sharpe Ratio > 0.75",
            "Traded Months",
        ):
            self.assertIn(metric, html)

        # row_labels: "Benchmark correlation" as a key is shorthand for every
        # resolved correlation row, so this anonymizes the ticker without the
        # caller ever naming it.
        anon = pnl_single_bm.evaluate_pnls_html(
            headline="PNL_HEAD",
            bench="PNL_BENCH",
            row_labels={"Benchmark correlation": "Benchmark correlation"},
        )
        self.assertIn("Benchmark correlation", anon.data)
        self.assertNotIn("USD_DUXR correl", anon.data)

        # An arbitrary row can also be relabeled by its real name.
        relabeled = pnl_single_bm.evaluate_pnls_html(
            headline="PNL_HEAD",
            bench="PNL_BENCH",
            row_labels={"St. Dev. %": "Vol %"},
        )
        self.assertIn("Vol %", relabeled.data)
        self.assertNotIn("St. Dev. %", relabeled.data)

        # Multiple benchmarks: the "Benchmark correlation" shorthand can't
        # give two rows the same label (duplicate index -> ambiguous lookup
        # in pnl_table_html), so it numbers them instead of colliding.
        pnl_multi_bm = NaivePnL(
            self.dfd,
            ret=ret,
            sigs=sigs,
            cids=self.cids,
            start="2000-01-01",
            blacklist=self.blacklist,
            bms=["EUR_DUXR", "USD_DUXR"],
        )
        pnl_multi_bm.make_pnl(
            sig="INFL", sig_op="zn_score_pan", rebal_freq="monthly",
            vol_scale=5, min_obs=250, pnl_name="PNL_HEAD",
        )
        pnl_multi_bm.make_pnl(
            sig="INFL", sig_op="zn_score_pan", sig_neg=True, rebal_freq="monthly",
            vol_scale=5, min_obs=250, pnl_name="PNL_BENCH",
        )
        anon_multi = pnl_multi_bm.evaluate_pnls_html(
            headline="PNL_HEAD",
            bench="PNL_BENCH",
            row_labels={"Benchmark correlation": "Benchmark correlation"},
        )
        self.assertIn("Benchmark correlation 1", anon_multi.data)
        self.assertIn("Benchmark correlation 2", anon_multi.data)
        self.assertNotIn("EUR_DUXR correl", anon_multi.data)
        self.assertNotIn("USD_DUXR correl", anon_multi.data)

        # A list lets each benchmark get its own explicit label, in the
        # order `bms` was passed to the constructor (EUR_DUXR, USD_DUXR).
        explicit_multi = pnl_multi_bm.evaluate_pnls_html(
            headline="PNL_HEAD",
            bench="PNL_BENCH",
            row_labels={"Benchmark correlation": ["EUR leg", "USD leg"]},
        )
        self.assertIn("EUR leg", explicit_multi.data)
        self.assertIn("USD leg", explicit_multi.data)
        self.assertNotIn("EUR_DUXR correl", explicit_multi.data)
        self.assertNotIn("USD_DUXR correl", explicit_multi.data)

        # Wrong-length list raises rather than silently mismatching/dropping.
        with self.assertRaises(ValueError):
            pnl_multi_bm.evaluate_pnls_html(
                headline="PNL_HEAD",
                bench="PNL_BENCH",
                row_labels={"Benchmark correlation": ["Only one label"]},
            )

        # Multiple benchmark *portfolios*: bench/bench_label accept lists,
        # each rendered as its own muted column.
        pnl_multi_bm.make_pnl(
            sig="INFL", sig_op="zn_score_pan", rebal_freq="monthly",
            vol_scale=5, min_obs=250, pnl_name="PNL_BENCH2",
        )
        multi_bench = pnl_multi_bm.evaluate_pnls_html(
            headline="PNL_HEAD",
            bench=["PNL_BENCH", "PNL_BENCH2"],
            bench_label=["Passive A", "Passive B"],
        )
        self.assertIn("Passive A", multi_bench.data)
        self.assertIn("Passive B", multi_bench.data)

        # bench_label length must match bench.
        with self.assertRaises(ValueError):
            pnl_multi_bm.evaluate_pnls_html(
                headline="PNL_HEAD",
                bench=["PNL_BENCH", "PNL_BENCH2"],
                bench_label=["Only one label"],
            )

        # Custom groups restrict rows to exactly what was asked for, in the
        # order given, and don't require an `order` override to do so.
        custom = pnl_single_bm.evaluate_pnls_html(
            headline="PNL_HEAD",
            bench="PNL_BENCH",
            groups={"Performance": ["Sharpe Ratio", "Return %"]},
        )
        self.assertIn("Sharpe Ratio", custom.data)
        self.assertIn("Return %", custom.data)
        self.assertNotIn("Sortino Ratio", custom.data)

        # A `groups`/`order` request for a metric that truly doesn't exist
        # still raises, even though the default-groups path silently drops
        # a missing benchmark correlation row (tested below via a
        # no-benchmark instance).
        with self.assertRaises(KeyError):
            pnl_single_bm.evaluate_pnls_html(
                headline="PNL_HEAD",
                bench="PNL_BENCH",
                groups={"X": ["Not A Real Metric"]},
            )

        # No benchmark configured: default groups silently drop the
        # correlation row rather than raising.
        pnl_no_bm = NaivePnL(
            self.dfd,
            ret=ret,
            sigs=sigs,
            cids=self.cids,
            start="2000-01-01",
            blacklist=self.blacklist,
        )
        pnl_no_bm.make_pnl(
            sig="INFL", sig_op="zn_score_pan", rebal_freq="monthly",
            vol_scale=5, min_obs=250, pnl_name="PNL_HEAD",
        )
        pnl_no_bm.make_pnl(
            sig="INFL", sig_op="zn_score_pan", sig_neg=True, rebal_freq="monthly",
            vol_scale=5, min_obs=250, pnl_name="PNL_BENCH",
        )
        no_bm_result = pnl_no_bm.evaluate_pnls_html(
            headline="PNL_HEAD", bench="PNL_BENCH"
        )
        self.assertNotIn("USD_DUXR correl", no_bm_result.data)
        self.assertIsNone(re.search(r" correl</td>", no_bm_result.data))

    def test_evaluate_pnls_pretty_ongoing_drawdown(self):
        # Up for half the sample, then straight down with no recovery: the
        # "Max Draw Recovery (months)" cell should read "+2" (an open
        # drawdown, not a completed one) and the table should carry a
        # footnote explaining the "+". A PnL that fully recovers should
        # show neither.
        dates = pd.bdate_range("2010-01-01", periods=100)
        values = np.concatenate([np.ones(50), -np.ones(50)])
        df = pd.concat([
            pd.DataFrame({"cid": "USD", "xcat": "XR", "real_date": dates, "value": values}),
            pd.DataFrame({"cid": "USD", "xcat": "SIG", "real_date": dates, "value": np.ones(100)}),
        ])
        pnl = NaivePnL(df, ret="XR", sigs=["SIG"], cids=["USD"], start="2010-01-01")
        pnl.make_pnl(
            sig="SIG", sig_op="raw", rebal_freq="daily", vol_scale=None,
            min_obs=1, pnl_name="PNL",
        )
        ongoing_result = pnl.evaluate_pnls_html(
            headline="PNL",
            bench="PNL",
            groups={"Risk & drawdown": ["Max Draw Recovery (months)"]},
        )
        self.assertIn("+2", ongoing_result.data)
        self.assertIn(
            "drawdown had not yet recovered as of the last observation",
            ongoing_result.data,
        )

        # footnotes=False suppresses the explanatory text but keeps the "+"
        # prefix on the cell itself.
        no_footnote_result = pnl.evaluate_pnls_html(
            headline="PNL",
            bench="PNL",
            groups={"Risk & drawdown": ["Max Draw Recovery (months)"]},
            footnotes=False,
        )
        self.assertIn("+2", no_footnote_result.data)
        self.assertNotIn(
            "drawdown had not yet recovered as of the last observation",
            no_footnote_result.data,
        )

        recovered_values = np.concatenate(
            [np.ones(20), -np.ones(20), np.ones(20)]
        )
        df_recovered = pd.concat([
            pd.DataFrame({
                "cid": "USD", "xcat": "XR", "real_date": pd.bdate_range("2010-01-01", periods=60),
                "value": recovered_values,
            }),
            pd.DataFrame({
                "cid": "USD", "xcat": "SIG", "real_date": pd.bdate_range("2010-01-01", periods=60),
                "value": np.ones(60),
            }),
        ])
        pnl_recovered = NaivePnL(
            df_recovered, ret="XR", sigs=["SIG"], cids=["USD"], start="2010-01-01"
        )
        pnl_recovered.make_pnl(
            sig="SIG", sig_op="raw", rebal_freq="daily", vol_scale=None,
            min_obs=1, pnl_name="PNL",
        )
        recovered_result = pnl_recovered.evaluate_pnls_html(
            headline="PNL",
            bench="PNL",
            groups={"Risk & drawdown": ["Max Draw Recovery (months)"]},
        )
        self.assertNotIn("+", recovered_result.data.split("<tbody>")[1].split("</tbody>")[0])
        self.assertNotIn(
            "drawdown had not yet recovered as of the last observation",
            recovered_result.data,
        )

        # Renaming "Max Draw Recovery (months)"/"Traded Months" via
        # row_labels must not lose their whole-number rounding -- the cells
        # should still read as clean integers, not 2-decimal values.
        renamed_result = pnl.evaluate_pnls_html(
            headline="PNL",
            bench="PNL",
            groups={
                "Risk & drawdown": ["Max Draw Recovery (months)"],
                "Robustness": ["Traded Months"],
            },
            row_labels={
                "Max Draw Recovery (months)": "Recovery (months)",
                "Traded Months": "Months Traded",
            },
        )
        self.assertIn("Recovery (months)", renamed_result.data)
        self.assertIn("Months Traded", renamed_result.data)
        # pnl-num-split is only applied to decimal-formatted values; its
        # absence from the table body confirms both renamed rows kept
        # their whole-number rounding instead of falling back to 2
        # decimals (the class is still *defined* in the default
        # stylesheet regardless, so check usage, not definition).
        tbody = renamed_result.data.split("<tbody>")[1].split("</tbody>")[0]
        self.assertNotIn("pnl-num-split", tbody)

    def test_evaluate_pnls_pretty_custom_css(self):
        from IPython.display import HTML

        ret = "EQXR"
        sigs = ["CRY", "GROWTH", "INFL"]
        pnl = NaivePnL(
            self.dfd,
            ret=ret,
            sigs=sigs,
            cids=self.cids,
            start="2000-01-01",
            blacklist=self.blacklist,
        )
        pnl.make_pnl(
            sig="INFL", sig_op="zn_score_pan", rebal_freq="monthly",
            vol_scale=5, min_obs=250, pnl_name="PNL_HEAD",
        )
        pnl.make_pnl(
            sig="INFL", sig_op="zn_score_pan", sig_neg=True, rebal_freq="monthly",
            vol_scale=5, min_obs=250, pnl_name="PNL_BENCH",
        )

        default_html = pnl.evaluate_pnls_html(headline="PNL_HEAD", bench="PNL_BENCH")
        custom = ".pnl-title { color: #7a1f1f; } .pnl-value-bench { color: #ff00ff; }"
        custom_html = pnl.evaluate_pnls_html(
            headline="PNL_HEAD", bench="PNL_BENCH", custom_css=custom,
        )
        self.assertIsInstance(custom_html, HTML)

        # The custom block is injected verbatim, after the default
        # stylesheet, so equal-specificity selectors are overridden by
        # source order without needing !important.
        self.assertIn(custom, custom_html.data)
        self.assertLess(
            custom_html.data.index(".pnl-title {"),
            custom_html.data.rindex(".pnl-title {"),
        )

        # Passing no custom_css must render identically to before (no
        # regression in the default look from the class-based refactor).
        without_custom = pnl.evaluate_pnls_html(headline="PNL_HEAD", bench="PNL_BENCH")
        self.assertEqual(default_html.data, without_custom.data)

    def test_evaluate_pnls_pretty_bench_defaults_to_bms(self):
        ret = "EQXR"
        sigs = ["CRY", "GROWTH", "INFL"]

        pnl_multi_bm = NaivePnL(
            self.dfd,
            ret=ret,
            sigs=sigs,
            cids=self.cids,
            start="2000-01-01",
            blacklist=self.blacklist,
            bms=["EUR_DUXR", "USD_DUXR"],
        )
        pnl_multi_bm.make_pnl(
            sig="INFL", sig_op="zn_score_pan", rebal_freq="monthly",
            vol_scale=5, min_obs=250, pnl_name="PNL_HEAD",
        )

        # Omitting bench= entirely defaults to the raw bms tickers as their
        # own columns -- same benchmarks used for the correlation rows,
        # without needing a synthetic "long only" PnL.
        result = pnl_multi_bm.evaluate_pnls_html(headline="PNL_HEAD")
        self.assertIn("EUR_DUXR", result.data)
        self.assertIn("USD_DUXR", result.data)
        self.assertIn("EUR_DUXR correl", result.data)
        self.assertIn("USD_DUXR correl", result.data)

        # No bench and no bms configured at all: clear error, not a
        # confusing failure deep in evaluate_pnls().
        pnl_no_bm = NaivePnL(
            self.dfd, ret=ret, sigs=sigs, cids=self.cids,
            start="2000-01-01", blacklist=self.blacklist,
        )
        pnl_no_bm.make_pnl(
            sig="INFL", sig_op="zn_score_pan", rebal_freq="monthly",
            vol_scale=5, min_obs=250, pnl_name="PNL_HEAD",
        )
        with self.assertRaises(ValueError):
            pnl_no_bm.evaluate_pnls_html(headline="PNL_HEAD")

        # Explicit bench= naming an actual bms ticker (not a PnL name) also
        # works, going through the same raw-ticker stats path. USD_DUXR
        # still gets a correlation row (evaluate_pnls() always reports
        # correlation against every configured bms ticker, independent of
        # which one is shown as the bench column) but isn't a column.
        explicit_ticker = pnl_multi_bm.evaluate_pnls_html(
            headline="PNL_HEAD", bench="EUR_DUXR"
        )
        headers = re.findall(r'<th class="pnl-th">([^<]+)</th>', explicit_ticker.data)
        self.assertEqual(headers, ["PNL_HEAD", "EUR_DUXR"])

    def test_evaluate_pnls_pretty_negative_top5_share(self):
        # Strong negative drift overall, with one very strong positive
        # month: total PnL ends up negative while the best (top-5%) month
        # is positive, so "Top 5% Monthly PnL Share" (top-5%-months PnL /
        # total PnL) comes out negative -- the case the footnote explains.
        dates = pd.bdate_range("2010-01-01", periods=300)
        rng = np.random.default_rng(7)
        values = rng.normal(-0.01, 0.003, len(dates))
        values[10:31] += 0.03
        df = pd.concat([
            pd.DataFrame({"cid": "USD", "xcat": "XR", "real_date": dates, "value": values}),
            pd.DataFrame({"cid": "USD", "xcat": "SIG", "real_date": dates, "value": np.ones(len(dates))}),
        ])
        pnl = NaivePnL(df, ret="XR", sigs=["SIG"], cids=["USD"], start="2010-01-01")
        pnl.make_pnl(
            sig="SIG", sig_op="raw", rebal_freq="daily", vol_scale=None,
            min_obs=1, pnl_name="PNL",
        )
        raw = pnl.evaluate_pnls(pnl_cats=["PNL"])
        self.assertLess(raw.loc["Top 5% Monthly PnL Share", "PNL"], 0)

        result = pnl.evaluate_pnls_html(
            headline="PNL", bench="PNL",
            groups={"Robustness": ["Top 5% Monthly PnL Share"]},
        )
        self.assertIn(
            "Top 5% Monthly PnL Share is top-5%-months PnL", result.data
        )

        no_footnote_result = pnl.evaluate_pnls_html(
            headline="PNL", bench="PNL",
            groups={"Robustness": ["Top 5% Monthly PnL Share"]},
            footnotes=False,
        )
        self.assertNotIn(
            "Top 5% Monthly PnL Share is top-5%-months PnL", no_footnote_result.data
        )

    def test_make_long_pnl(self):
        ret = "EQXR"
        sigs = ["CRY", "GROWTH", "INFL"]
        pnl = NaivePnL(
            self.dfd,
            ret=ret,
            sigs=sigs,
            cids=self.cids,
            start="2000-01-01",
            blacklist=self.blacklist,
            bms=["EUR_DUXR", "USD_DUXR"],
        )

        pnl.make_pnl(
            sig="GROWTH",
            sig_op="zn_score_pan",
            rebal_freq="daily",
            vol_scale=None,
            rebal_slip=0,
            pnl_name="PNL_GROWTH",
            min_obs=252,
            iis=True,
            sequential=True,
            neutral="zero",
            thresh=None,
        )

        pnl.make_long_pnl(vol_scale=None, label="Unit_Long_EQXR")

        long_equity = pnl.df[pnl.df["xcat"] == "Unit_Long_EQXR"]
        # Long-only is naturally computed across the panel (individual cross-section's
        # returns are already present in the DataFrame). Therefore, confirm that the
        # only cross-section in 'cid' column is "ALL".
        self.assertTrue(list(long_equity["cid"].unique()) == ["ALL"])

        df = self.dfd
        return_df = df[df["xcat"] == "EQXR"]

        # Test on a random date.
        random_date = "2016-01-19"
        return_dfw = return_df.pivot(index="real_date", columns="cid", values="value")
        # Sum across the row: unitary position.
        return_calc = sum(return_dfw.loc[random_date, :])
        # Convert to a pd.Series.
        long_equity_series = long_equity.pivot(
            index="real_date", columns="cid", values="value"
        )

        self.assertTrue(
            np.isclose(
                return_calc,
                float(long_equity_series.loc[random_date].iloc[0]),
                atol=0.0001,
            )
        )

        # The remaining methods in NaivePnL are graphical plots which display the values
        # computed using the functions above. Therefore, if the functionality is correct
        # above, the plotting methods do not explicitly need to be tested in the Unit
        # Test as a visual assessment will be sufficient.

        # Another test run with vol_scale=None

        ret = "EQXR"
        sigs = ["CRY", "GROWTH", "INFL"]
        pnl = NaivePnL(
            self.dfd,
            ret=ret,
            sigs=sigs,
            cids=self.cids,
            start="2000-01-01",
            blacklist=self.blacklist,
            bms=["EUR_DUXR", "USD_DUXR"],
        )

        pnl.make_pnl(
            sig="GROWTH",
            sig_op="zn_score_pan",
            rebal_freq="daily",
            vol_scale=None,
            rebal_slip=0,
            pnl_name="PNL_GROWTH",
            min_obs=252,
            iis=True,
            sequential=True,
            neutral="zero",
            thresh=None,
        )

        pnl.make_long_pnl(vol_scale=None, label="Unit_Long_EQXR")

        # same conditions as vol_scale=0 should apply.
        long_equity = pnl.df[pnl.df["xcat"] == "Unit_Long_EQXR"]
        self.assertTrue(list(long_equity["cid"].unique()) == ["ALL"])

        df = self.dfd
        return_df = df[df["xcat"] == "EQXR"]
        random_date = "2016-01-19"
        return_dfw = return_df.pivot(index="real_date", columns="cid", values="value")
        return_calc = sum(return_dfw.loc[random_date, :])
        long_equity_series = long_equity.pivot(
            index="real_date", columns="cid", values="value"
        )

        self.assertTrue(
            np.isclose(
                return_calc,
                float(long_equity_series.loc[random_date].iloc[0]),
                atol=0.0001,
            )
        )
        pnl.make_long_pnl(vol_scale=None, label=None)

    def test_evaluate_pnls_type_checks(self):
        ret = "EQXR"
        sigs = ["CRY", "GROWTH", "INFL"]

        pnl = NaivePnL(self.dfd, ret=ret, sigs=sigs)

        for arg in ["pnl_cids", "pnl_cats"]:
            for argval in [1, "A", [1]]:
                with self.assertRaises(TypeError):
                    pnl.evaluate_pnls(**{arg: argval})

        for argval in [1, "A", ["A"]]:
            with self.assertRaises(TypeError):
                pnl.evaluate_pnls(sr_thresholds=argval)

        # pass a random pnl_cat
        with self.assertRaises(ValueError):
            pnl.evaluate_pnls(pnl_cats=["banana"])

    def test_plotting_methods(self):
        plt.close("all")
        mock_plt = patch("matplotlib.pyplot.show").start()
        mpl_backend = matplotlib.get_backend()
        matplotlib.use("Agg")

        ret = "EQXR"
        sigs = ["CRY", "GROWTH", "INFL"]

        pnl = NaivePnL(
            self.dfd,
            ret=ret,
            sigs=sigs,
            cids=self.cids,
            start="2000-01-01",
            blacklist=self.blacklist,
            bms=["EUR_DUXR", "USD_DUXR"],
        )

        pnl.make_pnl(
            sig="GROWTH",
            sig_op="zn_score_pan",
            rebal_freq="daily",
            vol_scale=None,
            rebal_slip=0,
            pnl_name="PNL_GROWTH",
            min_obs=252,
            iis=True,
            sequential=True,
            neutral="zero",
            thresh=None,
        )

        pnl.make_long_pnl(vol_scale=1, label="Unit_Long_EQXR")

        # Confirm the plotting methods do not raise any errors.

        try:
            pnl.plot_pnls(pnl_cats=["PNL_GROWTH", "Unit_Long_EQXR"])
        except Exception as e:
            self.fail(f"plot_pnl raised {e} unexpectedly")

        try:
            pnl.plot_pnls(pnl_cats=["PNL_GROWTH", "Unit_Long_EQXR"], compounding=True)
        except Exception as e:
            self.fail(f"plot_pnl raised {e} unexpectedly")

        with self.assertRaises(TypeError):
            pnl.plot_pnls(pnl_cats=["PNL_GROWTH", "Unit_Long_EQXR"], xcat_labels=1)

        with self.assertWarns(Warning):
            pnl.plot_pnls(
                pnl_cats=["PNL_GROWTH", "Unit_Long_EQXR"], xcat_labels=["A", "B"]
            )

        with self.assertRaises(ValueError):
            pnl.plot_pnls(
                pnl_cats=["PNL_GROWTH", "Unit_Long_EQXR"],
                xcat_labels={"PNL_GROWTH": "A"},
            )

        try:
            pnl.signal_heatmap(pnl_name="PNL_GROWTH")
        except Exception as e:
            self.fail(f"signal_heatmap raised {e} unexpectedly")

        try:
            pnl.agg_signal_bars(pnl_name="PNL_GROWTH")
        except Exception as e:
            self.fail(f"agg_signal_bars raised {e} unexpectedly")

        patch.stopall()
        plt.close("all")
        matplotlib.use(mpl_backend)

    def test_validation_of_create_results_dataframe(self):
        ret = 1
        sigs = ["CRY", "GROWTH", "INFL"]

        with self.assertRaises(TypeError):
            results_df = create_results_dataframe(
                title="Performance metrics, PARITY vs OLS, equity",
                df=self.dfd,
                ret=ret,
                sigs=sigs,
                cids=self.cids,
                sig_ops="zn_score_pan",
                sig_adds=0,
                neutrals="zero",
                threshs=2,
                sig_negs=[True, False, False],
                bm="USD_EQXR",
                cosp=True,
                start="2004-01-01",
                freqs="M",
                agg_sigs="last",
                slip=1,
            )

        ret = "EQXR"
        sigs = ["CRY", "GROWTH", 1]

        with self.assertRaises(TypeError):
            results_df = create_results_dataframe(
                title="Performance metrics, PARITY vs OLS, equity",
                df=self.dfd,
                ret=ret,
                sigs=sigs,
                cids=self.cids,
                sig_ops="zn_score_pan",
                sig_adds=0,
                neutrals="zero",
                threshs=2,
                sig_negs=[True, False, False],
                bm="USD_EQXR",
                cosp=True,
                start="2004-01-01",
                freqs="M",
                agg_sigs="last",
                slip=1,
            )

        ret = "EQXR"
        sigs = ["CRY", "GROWTH", "INFL"]

        with self.assertRaises(TypeError):
            results_df = create_results_dataframe(
                title="Performance metrics, PARITY vs OLS, equity",
                df=self.dfd,
                ret=ret,
                sigs=sigs,
                cids=1,
                sig_ops="zn_score_pan",
                sig_adds=0,
                neutrals="zero",
                threshs=2,
                sig_negs=[True, False, False],
                bm="USD_EQXR",
                cosp=True,
                start="2004-01-01",
                freqs="M",
                agg_sigs="last",
                slip=1,
            )

        with self.assertRaises(TypeError):
            results_df = create_results_dataframe(
                title="Performance metrics, PARITY vs OLS, equity",
                df=self.dfd,
                ret=ret,
                sigs=sigs,
                cids=self.cids,
                sig_ops=["zn_score_pan", 4, 4],
                sig_adds=0,
                neutrals="zero",
                threshs=2,
                sig_negs=[True, False, False],
                bm="USD_EQXR",
                cosp=True,
                start="2004-01-01",
                freqs="M",
                agg_sigs="last",
                slip=1,
            )

        with self.assertRaises(TypeError):
            results_df = create_results_dataframe(
                title="Performance metrics, PARITY vs OLS, equity",
                df=self.dfd,
                ret=ret,
                sigs=sigs,
                cids=self.cids,
                sig_ops="zn_score_pan",
                sig_adds=[0, "jsajf"],
                neutrals="zero",
                threshs=2,
                sig_negs=[True, False, False],
                bm="USD_EQXR",
                cosp=True,
                start="2004-01-01",
                freqs="M",
                agg_sigs="last",
                slip=1,
            )

        with self.assertRaises(TypeError):
            results_df = create_results_dataframe(
                title="Performance metrics, PARITY vs OLS, equity",
                df=self.dfd,
                ret=ret,
                sigs=sigs,
                cids=self.cids,
                sig_ops="zn_score_pan",
                sig_adds=0,
                neutrals=["zero", 132213],
                threshs=2,
                sig_negs=[True, False, False],
                bm="USD_EQXR",
                cosp=True,
                start="2004-01-01",
                freqs="M",
                agg_sigs="last",
                slip=1,
            )

        with self.assertRaises(TypeError):
            results_df = create_results_dataframe(
                title="Performance metrics, PARITY vs OLS, equity",
                df=self.dfd,
                ret=ret,
                sigs=sigs,
                cids=self.cids,
                sig_ops="zn_score_pan",
                sig_adds=0,
                neutrals="zero",
                threshs="2",
                sig_negs=[True, False, False],
                bm="USD_EQXR",
                cosp=True,
                start="2004-01-01",
                freqs="M",
                agg_sigs="last",
                slip=1,
            )

    def test_result_of_create_results_dataframe(self):
        ret = "EQXR"
        sigs = ["CRY", "GROWTH", "INFL"]
        sig_negs = [True, False, False]

        pnl = NaivePnL(
            self.dfd,
            ret=ret,
            sigs=sigs,
            cids=self.cids,
            start="2000-01-01",
            blacklist=self.blacklist,
            bms=["USD_DUXR", "EUR_DUXR"],
        )
        for i, sig in enumerate(sigs):
            pnl.make_pnl(
                sig=sig,
                sig_op="zn_score_pan",
                sig_neg=sig_negs[i],
                rebal_freq="monthly",
                thresh=2,
                neutral="zero",
                sig_add=0,
            )

        results_df = create_results_dataframe(
            title="Performance metrics, PARITY vs OLS, equity",
            pnl=pnl,
            cosp=True,
            agg_sigs="last",
            slip=1,
        )

        if isinstance(results_df, pd.DataFrame):
            results = results_df
        elif isinstance(results_df, pd.io.formats.style.Styler):
            results = results_df.data
        else:
            raise ValueError("results_df is not a DataFrame or Styler object.")

        negative_sigs = [
            sig + "_NEG" if sig_negs[sigs.index(sig)] else sig for sig in sigs
        ]

        self.assertEqual(set(results.index), set(negative_sigs))

        self.assertEqual(len(results.columns), 8)

    def test_data_accessors(self):
        ret = "EQXR"
        sigs = ["CRY", "GROWTH"]
        pnl = NaivePnL(
            self.dfd,
            ret=ret,
            sigs=sigs,
            cids=self.cids,
            start="2000-01-01",
            blacklist=self.blacklist,
        )

        pnl_name = "PNL_CRY"
        pnl.make_pnl(
            sig="CRY",
            sig_op="zn_score_pan",
            rebal_freq="daily",
            vol_scale=None,
            rebal_slip=0,
            pnl_name=pnl_name,
            min_obs=252,
            iis=True,
            sequential=True,
            neutral="zero",
            thresh=None,
        )

        input_df = pnl.get_input_signals()
        self.assertSetEqual(set(input_df["xcat"].unique()), set(sigs))
        self.assertSetEqual(set(input_df["cid"].unique()), set(self.cids))
        expected_input_df = reduce_df(pnl.df, xcats=sigs, cids=self.cids)
        pd.testing.assert_frame_equal(
            QuantamentalDataFrame(input_df),
            QuantamentalDataFrame(expected_input_df),
        )

        returns_df = pnl.get_asset_returns_data()
        self.assertSetEqual(set(returns_df["xcat"].unique()), {ret})
        self.assertSetEqual(set(returns_df["cid"].unique()), set(self.cids))
        expected_returns_df = reduce_df(pnl.df, xcats=[ret], cids=self.cids)
        pd.testing.assert_frame_equal(
            QuantamentalDataFrame(returns_df),
            QuantamentalDataFrame(expected_returns_df),
        )

        signals_df = pnl.get_generated_signals()
        self.assertFalse(signals_df.empty)
        self.assertSetEqual(set(signals_df["xcat"].unique()), {pnl_name})
        self.assertTrue(set(signals_df["cid"].unique()).issubset(set(self.cids)))
        self.assertIn("value", signals_df.columns)
        expected_signals_df = (
            pd.concat([pnl.signal_df[pnl_name].assign(xcat=pnl_name)])
            .dropna()
            .reset_index(drop=True)
            .rename(columns={"sig": "value"})
        )
        pd.testing.assert_frame_equal(
            QuantamentalDataFrame(signals_df),
            QuantamentalDataFrame(expected_signals_df),
        )

        pnls_df = pnl.get_pnls_returns_data()
        self.assertFalse(pnls_df.empty)
        self.assertSetEqual(set(pnls_df["xcat"].unique()), {pnl_name})
        self.assertTrue(set(pnls_df["cid"].unique()).issubset(set(self.cids)))
        expected_pnls_df = reduce_df(pnl.df, xcats=[pnl_name], cids=self.cids)
        pd.testing.assert_frame_equal(
            QuantamentalDataFrame(pnls_df),
            QuantamentalDataFrame(expected_pnls_df),
        )


if __name__ == "__main__":
    unittest.main()
