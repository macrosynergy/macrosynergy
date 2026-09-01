from tests.simulate import make_qdf
from macrosynergy.pnl.naive_pnl import NaivePnL, create_results_dataframe
from macrosynergy.management.utils import reduce_df, update_df, get_sops
import re
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from typing import List, Dict
import matplotlib
from matplotlib import pyplot as plt
from macrosynergy.management.types import QuantamentalDataFrame
from macrosynergy.pnl.pnl_table import HTMLTable


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
        # A signal is only carried within its own re-balancing period, so the months
        # after the signal category ends hold no position: count the live signals.
        unique_values_aud = set(aud_array[~np.isnan(aud_array)])

        start_date = dfw_signal_rebal.index[0]
        end_date = dfw_signal_rebal.index[-1]

        no_months = self.diff_month(end_date, start_date)

        self.assertTrue(no_months - 1 == len(unique_values_aud))

        # 'GROWTH' ends 2020-10-30 while the returns run to year end. Without a signal
        # on their re-balancing date, the remaining months close the position instead
        # of holding the last live signal to the end of the sample.
        self.assertTrue(dfw_signal_rebal_aud.loc["2020-11-01":].isna().all())

    def test_rebalancing_period_calendar_and_carry(self):
        # (a) A week belongs to exactly one re-balancing period. The Gregorian year
        # combined with the ISO week number used to split the turn of the year, dropping
        # the last week of December and re-balancing on January 1st instead. The dates
        # must match the package's own period edges.
        dates = pd.bdate_range("2018-01-01", "2021-12-31")
        dfw = pd.DataFrame(
            {
                "cid": "AUD",
                "real_date": dates,
                # psig is the row ordinal, so each output value names its rebal date.
                "psig": np.arange(len(dates), dtype=float),
            }
        )
        sig = np.squeeze(NaivePnL.rebalancing(dfw, rebal_freq="weekly").to_numpy())
        self.assertEqual(
            set(dates[np.unique(sig).astype(int)]),
            set(pd.to_datetime(get_sops(dates=dates, freq="W"))),
        )

        # (b) A signal is only carried within its own re-balancing period: once the
        # signal has ended the position closes at the end of that period.
        dates = pd.bdate_range("2020-01-01", "2020-12-31")
        dfw = pd.DataFrame({"cid": "AUD", "real_date": dates, "psig": 1.0})
        dfw.loc[dfw["real_date"] > "2020-06-15", "psig"] = np.nan
        dfw["sig"] = np.squeeze(
            NaivePnL.rebalancing(dfw, rebal_freq="monthly").to_numpy()
        )
        self.assertEqual(
            dfw.loc[dfw["sig"].notna(), "real_date"].max(), pd.Timestamp("2020-06-30")
        )

    def test_rebalancing_weekly_year_boundary(self):
        # A week must belong to exactly one re-balancing period. Keying on the Gregorian
        # year together with the ISO week number collided at every turn of the year: the
        # last week of December inherited the previous week's signal and the re-balance
        # was displaced onto January 1st.
        cids = ["AUD", "CAD"]
        dates = pd.bdate_range("2018-01-01", "2021-12-31")  # spans three year-ends.
        dfw = pd.DataFrame(
            {
                "real_date": np.tile(dates, len(cids)),
                "cid": np.repeat(cids, len(dates)),
                # psig is the row's date ordinal, so each output value names the date
                # its signal was taken from, i.e. the re-balancing date.
                "psig": np.tile(np.arange(len(dates), dtype=float), len(cids)),
            }
        ).sort_values(["cid", "real_date"])

        dfw["sig"] = np.squeeze(
            NaivePnL.rebalancing(dfw.copy(), rebal_freq="weekly").to_numpy()
        )

        # The single-calendar grouping: one period per Monday-to-Sunday week.
        expected = set(dates.to_series().groupby(dates.to_period("W")).min())
        for cid in cids:
            observed = set(
                dates[dfw.loc[dfw["cid"] == cid, "sig"].to_numpy().astype(int)]
            )
            self.assertEqual(observed, expected)

            # The two calendars disagree only around the year-end, so pin those dates
            # explicitly. The final week of December owns its own re-balancing date.
            for boundary in ["2018-12-31", "2019-12-30", "2020-12-28"]:
                self.assertIn(pd.Timestamp(boundary), observed)
            # And no re-balance is displaced onto January 1st, which always falls
            # mid-week within the period that started in December.
            for spurious in ["2019-01-01", "2020-01-01", "2021-01-01"]:
                self.assertNotIn(pd.Timestamp(spurious), observed)

    def test_rebalancing_ffill_one_period_only(self):
        # A signal is carried within its own re-balancing period and no further, so a
        # signal category that ends while the returns continue closes the position at
        # the end of the period it died in, rather than holding it to the end of the
        # sample.
        dates = pd.bdate_range("2020-01-01", "2020-12-31")
        # 2020-06-15 is a Monday, hence both the first day of its week and mid-month:
        # the weekly carry expires on the Friday, the monthly carry at month end.
        for rebal_freq, last_live in [
            ("monthly", "2020-06-30"),
            ("weekly", "2020-06-19"),
        ]:
            dfw = pd.DataFrame({"cid": "AUD", "real_date": dates, "psig": 1.0})
            dfw.loc[dfw["real_date"] > "2020-06-15", "psig"] = np.nan
            dfw["sig"] = np.squeeze(
                NaivePnL.rebalancing(dfw.copy(), rebal_freq=rebal_freq).to_numpy()
            )

            live = dfw.loc[dfw["real_date"] <= last_live, "sig"]
            dead = dfw.loc[dfw["real_date"] > last_live, "sig"]
            # The period the signal died in keeps the value taken on its re-balancing
            # date; every subsequent period re-balances onto a missing signal.
            self.assertEqual(set(live.unique()), {1.0})
            self.assertTrue(dead.isna().all())
            self.assertTrue(len(dead) > 0)

        # Where the signal is fully populated the period bound never bites: the output
        # is identical to an unbounded forward fill of the re-balancing signals, at
        # every frequency and on a ragged panel.
        period_alias = {
            "daily": "D",
            "weekly": "W",
            "monthly": "M",
            "quarterly": "Q",
            "annual": "Y",
        }
        dates = pd.bdate_range("2015-01-01", "2020-12-31")
        rng = np.random.default_rng(1)
        dfw = (
            pd.concat(
                [
                    pd.DataFrame(
                        {
                            "cid": cid,
                            "real_date": dates[i * 53 :],
                            "psig": rng.standard_normal(len(dates) - i * 53),
                        }
                    )
                    for i, cid in enumerate(["AUD", "GBP", "USD"])
                ]
            )
            .sort_values(["cid", "real_date"])
            .reset_index(drop=True)
        )
        for rebal_freq, alias in period_alias.items():
            first_of_period = ~pd.MultiIndex.from_arrays(
                [dfw["cid"], dfw["real_date"].dt.to_period(alias)]
            ).duplicated()
            unbounded = (
                dfw["psig"]
                .where(first_of_period)
                .groupby(dfw["cid"], observed=True)
                .ffill()
            )
            sig = np.squeeze(
                NaivePnL.rebalancing(dfw.copy(), rebal_freq=rebal_freq).to_numpy()
            )
            self.assertTrue(np.array_equal(unbounded.to_numpy(), sig))
            self.assertFalse(np.isnan(sig).any())

    def test_rebalancing_input_flexibility(self):
        # The signal is returned labelled by (cid, real_date) in canonical sort order,
        # so neither the row order nor the dtype of 'cid' can attach a signal to the
        # wrong cross-section.
        cids = ["AUD", "CAD", "GBP"]
        dates = pd.bdate_range("2020-01-01", "2020-06-30")
        # psig is unique per row, so a misaligned signal cannot coincide with the
        # correct one: every output value names the row it was taken from.
        sorted_dfw = pd.DataFrame(
            {
                "cid": np.repeat(cids, len(dates)),
                "real_date": np.tile(dates, len(cids)),
                "psig": np.arange(len(cids) * len(dates), dtype=float),
            }
        )

        def str_keyed(sig):
            # 'cid' may come back as a Categorical; compare on the labels themselves.
            return sig.set_axis(
                pd.MultiIndex.from_arrays(
                    [
                        sig.index.get_level_values("cid").astype(str),
                        sig.index.get_level_values("real_date"),
                    ],
                    names=["cid", "real_date"],
                )
            )

        expected = str_keyed(
            NaivePnL.rebalancing(sorted_dfw.copy(), rebal_freq="monthly")["psig"]
        )
        # The reference actually re-balances: one signal per cid per month, not the
        # daily signal handed in.
        self.assertEqual(expected.nunique(), len(cids) * 6)

        def check(dfw):
            out = NaivePnL.rebalancing(dfw, rebal_freq="monthly")
            # Canonical order: the panel sorted by cross-section then date, whatever
            # order and date dtype it was handed in.
            canonical = dfw.assign(
                real_date=pd.to_datetime(dfw["real_date"])
            ).sort_values(["cid", "real_date"])
            pd.testing.assert_index_equal(
                out.index,
                pd.MultiIndex.from_arrays([canonical["cid"], canonical["real_date"]]),
            )
            # Every row carries the signal belonging to its own label.
            labelled = str_keyed(out["psig"])
            pd.testing.assert_series_equal(labelled.sort_index(), expected.sort_index())
            # Independent of the reference above: psig is the row ordinal of the
            # cid-major frame, so a value's block of ordinals names its cross-section.
            np.testing.assert_array_equal(
                (labelled.to_numpy() // len(dates)).astype(int),
                [cids.index(c) for c in labelled.index.get_level_values("cid")],
            )

        # Cross-section-major order, as make_pnl hands it over.
        check(sorted_dfw.copy())
        # Date-major order: cross-sections interleaved on every row.
        check(sorted_dfw.sort_values(["real_date", "cid"]))
        # Fully shuffled order.
        check(sorted_dfw.sample(frac=1, random_state=0))
        # 'cid' as a Categorical whose category order is the reverse of the lexical
        # one, both cid-major and date-major, and as plain strings (above).
        reversed_cat = sorted_dfw.assign(
            cid=pd.Categorical(sorted_dfw["cid"], categories=cids[::-1])
        )
        check(reversed_cat.sort_values(["cid", "real_date"]))
        check(reversed_cat.sort_values(["real_date", "cid"]))
        # 'real_date' handed over as ISO strings is coerced in flow.
        check(
            sorted_dfw.assign(
                real_date=sorted_dfw["real_date"].dt.strftime("%Y-%m-%d")
            ).sort_values(["real_date", "cid"])
        )

        # A genuinely unparseable date is left to raise pandas' own error.
        with self.assertRaises(ValueError):
            NaivePnL.rebalancing(
                sorted_dfw.assign(real_date="not-a-date"), rebal_freq="monthly"
            )

    def test_rebalancing_slip_no_cross_cid_leak(self):
        # With rebal_slip >= 1 the slip must be applied within each cross-section: the
        # first observation(s) of a cid must never inherit the preceding cid's signal.
        cids = ["AUD", "CAD", "GBP"]
        signals = {"AUD": 1.0, "CAD": 2.0, "GBP": 3.0}
        dates = pd.bdate_range("2020-01-01", "2020-06-30")

        dfw = pd.DataFrame(
            {
                "real_date": np.tile(dates, len(cids)),
                "cid": np.repeat(cids, len(dates)),
                "psig": np.repeat([signals[cid] for cid in cids], len(dates)),
            }
        ).sort_values(["cid", "real_date"])

        sig_series = NaivePnL.rebalancing(dfw, rebal_freq="monthly", rebal_slip=1)
        dfw["sig"] = np.squeeze(sig_series.to_numpy())

        # Each cid's signal is constant, so the only non-NA value it can legitimately
        # hold is its own. Anything else has leaked across the cid boundary.
        for cid in cids:
            observed = set(dfw.loc[dfw["cid"] == cid, "sig"].dropna().unique())
            self.assertEqual(observed, {signals[cid]})

    def test_rebalancing_slip_type_value_checks(self):
        # rebal_slip must be a non-negative integer: bad types raise TypeError, a
        # negative integer raises ValueError, and a valid value must go through.
        cids = ["AUD", "CAD"]
        dates = pd.bdate_range("2020-01-01", "2020-03-31")
        dfw = pd.DataFrame(
            {
                "real_date": np.tile(dates, len(cids)),
                "cid": np.repeat(cids, len(dates)),
                "psig": np.repeat([1.0, 2.0], len(dates)),
            }
        ).sort_values(["cid", "real_date"])

        for bad_slip in ["1", 1.5, None]:
            with self.assertRaises(TypeError):
                NaivePnL.rebalancing(
                    dfw.copy(), rebal_freq="monthly", rebal_slip=bad_slip
                )

        with self.assertRaises(ValueError):
            NaivePnL.rebalancing(dfw.copy(), rebal_freq="monthly", rebal_slip=-1)

        # A valid slippage must not raise. Note bools are ints in Python, so they pass.
        NaivePnL.rebalancing(dfw.copy(), rebal_freq="monthly", rebal_slip=0)

    def test_rebalancing_slip_no_cross_cid_leak_edge_cases(self):
        cids = ["AUD", "CAD", "GBP"]
        signals = {"AUD": 1.0, "CAD": 2.0, "GBP": 3.0}
        dates = pd.bdate_range("2020-01-01", "2020-03-31")

        # (a) A categorical cid column - including one carrying an unused category -
        # must give the same result as the object-dtype column and must not leak.
        dfw = pd.DataFrame(
            {
                "real_date": np.tile(dates, len(cids)),
                "cid": np.repeat(cids, len(dates)),
                "psig": np.repeat([signals[cid] for cid in cids], len(dates)),
            }
        ).sort_values(["cid", "real_date"])
        dfw_cat = dfw.assign(
            cid=pd.Categorical(dfw["cid"], categories=["AUD", "CAD", "CHF", "GBP"])
        )
        obj_sig = np.squeeze(
            NaivePnL.rebalancing(
                dfw.copy(), rebal_freq="monthly", rebal_slip=2
            ).to_numpy()
        )
        cat_sig = np.squeeze(
            NaivePnL.rebalancing(
                dfw_cat.copy(), rebal_freq="monthly", rebal_slip=2
            ).to_numpy()
        )
        self.assertTrue(np.array_equal(obj_sig, cat_sig, equal_nan=True))
        dfw_cat["sig"] = cat_sig
        for cid in cids:
            cid_sig = dfw_cat.loc[dfw_cat["cid"] == cid, "sig"]
            # A slip of two blanks exactly the first two dates of each cross-section.
            self.assertEqual(
                list(cid_sig.isna()), [True, True] + [False] * (len(dates) - 2)
            )
            self.assertEqual(set(cid_sig.dropna().unique()), {signals[cid]})

        # (b) A cid whose signal is entirely NaN, sat between two populated cids, must
        # stay NaN and must not absorb either neighbour's signal.
        dfw_nan = pd.DataFrame(
            {
                "real_date": np.tile(dates, len(cids)),
                "cid": np.repeat(cids, len(dates)),
                "psig": np.concatenate(
                    [
                        np.full(len(dates), signals["AUD"]),
                        np.full(len(dates), np.nan),
                        np.full(len(dates), signals["GBP"]),
                    ]
                ),
            }
        ).sort_values(["cid", "real_date"])
        dfw_nan["sig"] = np.squeeze(
            NaivePnL.rebalancing(
                dfw_nan.copy(), rebal_freq="monthly", rebal_slip=1
            ).to_numpy()
        )
        self.assertTrue(dfw_nan.loc[dfw_nan["cid"] == "CAD", "sig"].isna().all())
        for cid in ["AUD", "GBP"]:
            cid_sig = dfw_nan.loc[dfw_nan["cid"] == cid, "sig"]
            self.assertEqual(list(cid_sig.isna()), [True] + [False] * (len(dates) - 1))
            self.assertEqual(set(cid_sig.dropna().unique()), {signals[cid]})

        # (c) A slip longer than one cid's own history must empty that cid alone; the
        # longer neighbours lose exactly rebal_slip observations each.
        dfw_short = pd.concat(
            [
                pd.DataFrame(
                    {"real_date": dates, "cid": "AUD", "psig": signals["AUD"]}
                ),
                pd.DataFrame(
                    {"real_date": dates[:3], "cid": "CAD", "psig": signals["CAD"]}
                ),
                pd.DataFrame(
                    {"real_date": dates, "cid": "GBP", "psig": signals["GBP"]}
                ),
            ],
            ignore_index=True,
        ).sort_values(["cid", "real_date"])
        dfw_short["sig"] = np.squeeze(
            NaivePnL.rebalancing(
                dfw_short.copy(), rebal_freq="daily", rebal_slip=5
            ).to_numpy()
        )
        self.assertTrue(dfw_short.loc[dfw_short["cid"] == "CAD", "sig"].isna().all())
        for cid in ["AUD", "GBP"]:
            cid_sig = dfw_short.loc[dfw_short["cid"] == cid, "sig"]
            self.assertEqual(
                list(cid_sig.isna()), [True] * 5 + [False] * (len(dates) - 5)
            )
            self.assertEqual(set(cid_sig.dropna().unique()), {signals[cid]})

        # (d) The leak was once per cid boundary, so it scales with the panel width:
        # with many cross-sections every non-first cid must still hold only its own
        # signal, blanked for exactly the first observation.
        for n_cids in (5, 10, 50, 100):
            many_cids = [f"CID{i:03d}" for i in range(n_cids)]
            many_signals = {cid: float(i + 1) for i, cid in enumerate(many_cids)}
            dfw_many = pd.DataFrame(
                {
                    "real_date": np.tile(dates, n_cids),
                    "cid": np.repeat(many_cids, len(dates)),
                    "psig": np.repeat(
                        [many_signals[cid] for cid in many_cids], len(dates)
                    ),
                }
            ).sort_values(["cid", "real_date"])
            dfw_many["sig"] = np.squeeze(
                NaivePnL.rebalancing(
                    dfw_many.copy(), rebal_freq="monthly", rebal_slip=1
                ).to_numpy()
            )
            for cid in many_cids:
                cid_sig = dfw_many.loc[dfw_many["cid"] == cid, "sig"]
                self.assertEqual(
                    list(cid_sig.isna()), [True] + [False] * (len(dates) - 1)
                )
                self.assertEqual(set(cid_sig.dropna().unique()), {many_signals[cid]})

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

        df_custom = pnl.evaluate_pnls(pnl_cats=["PNL_INFL"], sr_thresholds=[0.1, 1.0])
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

    def test_max_drawdown_recovery_months(self):
        recovery_months = NaivePnL._max_drawdown_recovery_months

        # No drawdown at all.
        cum_pnl = pd.Series(np.arange(1, 101, dtype=float))
        self.assertEqual(recovery_months(cum_pnl), 0.0)

        # Rises 21 days to a peak, falls for 21 days to a trough, then
        # climbs back past the prior peak over another 21 days. Recovery is
        # measured from the *peak*, not the trough, so this is 42 trading
        # days - 2 months - end to end.
        up = np.arange(1, 22, dtype=float)
        down = up[-1] - np.arange(1, 22, dtype=float)
        recover = down[-1] + np.arange(1, 22, dtype=float)
        cum_pnl = pd.Series(np.concatenate([up, down, recover]))
        self.assertEqual(recovery_months(cum_pnl), 2.0)

        # 21 days up to a peak, then 21 days down, ending underwater with no
        # recovery. The drawdown has only been running since the peak (21
        # trading days = 1 month), regardless of how much history preceded
        # the peak - it must NOT fall back to the whole traded history.
        down_no_recovery = up[-1] - np.arange(1, 22, dtype=float)
        cum_pnl = pd.Series(np.concatenate([up, down_no_recovery]))
        self.assertEqual(recovery_months(cum_pnl), 1.0)
        self.assertEqual(recovery_months(cum_pnl, return_ongoing=True), (1.0, True))

        # Peak reached a quarter of the way through a 4-month series, then
        # underwater for the remaining three quarters with no recovery -
        # the ongoing drawdown should read as ~3 months, not the full 4.
        long_down_no_recovery = up[-1] - np.arange(1, 64, dtype=float)
        cum_pnl = pd.Series(np.concatenate([up, long_down_no_recovery]))
        self.assertEqual(recovery_months(cum_pnl), 3.0)

        # No non-NaN observations.
        cum_pnl = pd.Series([], dtype=float)
        self.assertTrue(np.isnan(recovery_months(cum_pnl)))

    def test_evaluate_pnls_pretty(self):
        from macrosynergy.pnl.pnl_table import HTMLTable

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
            sig="INFL",
            sig_op="zn_score_pan",
            rebal_freq="monthly",
            vol_scale=5,
            min_obs=250,
            pnl_name="PNL_HEAD",
        )
        pnl_single_bm.make_pnl(
            sig="INFL",
            sig_op="zn_score_pan",
            sig_neg=True,
            rebal_freq="monthly",
            vol_scale=5,
            min_obs=250,
            pnl_name="PNL_BENCH",
        )

        result = pnl_single_bm.evaluate_pnls_html(
            headline="PNL_HEAD", bench="PNL_BENCH"
        )
        self.assertIsInstance(result, HTMLTable)
        # Renders inline in a notebook via the standard rich-display hook.
        self.assertEqual(result._repr_html_(), result.data)
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
            sig="INFL",
            sig_op="zn_score_pan",
            rebal_freq="monthly",
            vol_scale=5,
            min_obs=250,
            pnl_name="PNL_HEAD",
        )
        pnl_multi_bm.make_pnl(
            sig="INFL",
            sig_op="zn_score_pan",
            sig_neg=True,
            rebal_freq="monthly",
            vol_scale=5,
            min_obs=250,
            pnl_name="PNL_BENCH",
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
            sig="INFL",
            sig_op="zn_score_pan",
            rebal_freq="monthly",
            vol_scale=5,
            min_obs=250,
            pnl_name="PNL_BENCH2",
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
            sig="INFL",
            sig_op="zn_score_pan",
            rebal_freq="monthly",
            vol_scale=5,
            min_obs=250,
            pnl_name="PNL_HEAD",
        )
        pnl_no_bm.make_pnl(
            sig="INFL",
            sig_op="zn_score_pan",
            sig_neg=True,
            rebal_freq="monthly",
            vol_scale=5,
            min_obs=250,
            pnl_name="PNL_BENCH",
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
        df = pd.concat(
            [
                pd.DataFrame(
                    {"cid": "USD", "xcat": "XR", "real_date": dates, "value": values}
                ),
                pd.DataFrame(
                    {
                        "cid": "USD",
                        "xcat": "SIG",
                        "real_date": dates,
                        "value": np.ones(100),
                    }
                ),
            ]
        )
        pnl = NaivePnL(df, ret="XR", sigs=["SIG"], cids=["USD"], start="2010-01-01")
        pnl.make_pnl(
            sig="SIG",
            sig_op="raw",
            rebal_freq="daily",
            vol_scale=None,
            min_obs=1,
            pnl_name="PNL",
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

        recovered_values = np.concatenate([np.ones(20), -np.ones(20), np.ones(20)])
        df_recovered = pd.concat(
            [
                pd.DataFrame(
                    {
                        "cid": "USD",
                        "xcat": "XR",
                        "real_date": pd.bdate_range("2010-01-01", periods=60),
                        "value": recovered_values,
                    }
                ),
                pd.DataFrame(
                    {
                        "cid": "USD",
                        "xcat": "SIG",
                        "real_date": pd.bdate_range("2010-01-01", periods=60),
                        "value": np.ones(60),
                    }
                ),
            ]
        )
        pnl_recovered = NaivePnL(
            df_recovered, ret="XR", sigs=["SIG"], cids=["USD"], start="2010-01-01"
        )
        pnl_recovered.make_pnl(
            sig="SIG",
            sig_op="raw",
            rebal_freq="daily",
            vol_scale=None,
            min_obs=1,
            pnl_name="PNL",
        )
        recovered_result = pnl_recovered.evaluate_pnls_html(
            headline="PNL",
            bench="PNL",
            groups={"Risk & drawdown": ["Max Draw Recovery (months)"]},
        )
        self.assertNotIn(
            "+", recovered_result.data.split("<tbody>")[1].split("</tbody>")[0]
        )
        self.assertNotIn(
            "drawdown had not yet recovered as of the last observation",
            recovered_result.data,
        )

        # Renaming "Max Draw Recovery (months)"/"Traded Months" via
        # row_labels must not lose their whole-number rounding - the cells
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
            sig="INFL",
            sig_op="zn_score_pan",
            rebal_freq="monthly",
            vol_scale=5,
            min_obs=250,
            pnl_name="PNL_HEAD",
        )
        pnl.make_pnl(
            sig="INFL",
            sig_op="zn_score_pan",
            sig_neg=True,
            rebal_freq="monthly",
            vol_scale=5,
            min_obs=250,
            pnl_name="PNL_BENCH",
        )

        default_html = pnl.evaluate_pnls_html(headline="PNL_HEAD", bench="PNL_BENCH")
        custom = ".pnl-title { color: #7a1f1f; } .pnl-value-bench { color: #ff00ff; }"
        custom_html = pnl.evaluate_pnls_html(
            headline="PNL_HEAD",
            bench="PNL_BENCH",
            custom_css=custom,
        )
        self.assertIsInstance(custom_html, HTMLTable)

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
            sig="INFL",
            sig_op="zn_score_pan",
            rebal_freq="monthly",
            vol_scale=5,
            min_obs=250,
            pnl_name="PNL_HEAD",
        )

        # Omitting bench= entirely defaults to the raw bms tickers as their
        # own columns - same benchmarks used for the correlation rows,
        # without needing a synthetic "long only" PnL.
        result = pnl_multi_bm.evaluate_pnls_html(headline="PNL_HEAD")
        self.assertIn("EUR_DUXR", result.data)
        self.assertIn("USD_DUXR", result.data)
        self.assertIn("EUR_DUXR correl", result.data)
        self.assertIn("USD_DUXR correl", result.data)

        # No bench and no bms configured at all: clear error, not a
        # confusing failure deep in evaluate_pnls().
        pnl_no_bm = NaivePnL(
            self.dfd,
            ret=ret,
            sigs=sigs,
            cids=self.cids,
            start="2000-01-01",
            blacklist=self.blacklist,
        )
        pnl_no_bm.make_pnl(
            sig="INFL",
            sig_op="zn_score_pan",
            rebal_freq="monthly",
            vol_scale=5,
            min_obs=250,
            pnl_name="PNL_HEAD",
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
        # total PnL) comes out negative - the case the footnote explains.
        dates = pd.bdate_range("2010-01-01", periods=300)
        rng = np.random.default_rng(7)
        values = rng.normal(-0.01, 0.003, len(dates))
        values[10:31] += 0.03
        df = pd.concat(
            [
                pd.DataFrame(
                    {"cid": "USD", "xcat": "XR", "real_date": dates, "value": values}
                ),
                pd.DataFrame(
                    {
                        "cid": "USD",
                        "xcat": "SIG",
                        "real_date": dates,
                        "value": np.ones(len(dates)),
                    }
                ),
            ]
        )
        pnl = NaivePnL(df, ret="XR", sigs=["SIG"], cids=["USD"], start="2010-01-01")
        pnl.make_pnl(
            sig="SIG",
            sig_op="raw",
            rebal_freq="daily",
            vol_scale=None,
            min_obs=1,
            pnl_name="PNL",
        )
        raw = pnl.evaluate_pnls(pnl_cats=["PNL"])
        self.assertLess(raw.loc["Top 5% Monthly PnL Share", "PNL"], 0)

        result = pnl.evaluate_pnls_html(
            headline="PNL",
            bench="PNL",
            groups={"Robustness": ["Top 5% Monthly PnL Share"]},
        )
        self.assertIn("Top 5% Monthly PnL Share is top-5%-months PnL", result.data)

        no_footnote_result = pnl.evaluate_pnls_html(
            headline="PNL",
            bench="PNL",
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

    def test_plot_pnl_consistency(self):
        plt.close("all")
        patch("matplotlib.pyplot.show").start()
        mpl_backend = matplotlib.get_backend()
        matplotlib.use("Agg")

        pnl = NaivePnL(
            self.dfd,
            ret="EQXR",
            sigs=["CRY", "GROWTH", "INFL"],
            cids=self.cids,
            start="2000-01-01",
            blacklist=self.blacklist,
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

        window = 252

        try:
            pnl.plot_pnl_consistency(
                pnl_cat="PNL_GROWTH",
                benchmark_pnl_cat="Unit_Long_EQXR",
                window=window,
            )
        except Exception as e:
            self.fail(f"plot_pnl_consistency raised {e} unexpectedly")

        # A figure with the two expected panels is returned.
        fig = pnl.plot_pnl_consistency(
            pnl_cat="PNL_GROWTH",
            benchmark_pnl_cat="Unit_Long_EQXR",
            window=window,
            title="Consistency of the macro value-add",
            xcat_labels={"PNL_GROWTH": "Macro", "Unit_Long_EQXR": "Long only"},
            return_fig=True,
        )
        self.assertIsInstance(fig, plt.Figure)
        self.assertEqual(len(fig.axes), 2)

        # The active return is the first category minus the second: confirm the
        # full-sample information ratio annotated on the figure.
        dfp = pnl.pnl_df(["PNL_GROWTH", "Unit_Long_EQXR"]).pivot(
            index="real_date", columns="xcat", values="value"
        )
        act = (dfp["PNL_GROWTH"] - dfp["Unit_Long_EQXR"]).dropna()
        expected_ir = act.mean() / act.std() * 252**0.5

        annotations = [
            child.get_text()
            for child in fig.axes[1].texts
            if child.get_text().startswith("full sample:")
        ]
        self.assertEqual(len(annotations), 1)
        self.assertEqual(annotations[0], f"full sample: {expected_ir:.2f}")

        # Default panel titles are generated from the labels and the window length.
        self.assertEqual(
            fig.axes[0].get_title(), "Cumulative active PnL: Macro minus Long only"
        )
        self.assertEqual(
            fig.axes[1].get_title(),
            "Rolling 1-year information ratio of the active return",
        )

        # Both panel titles can be overridden.
        fig = pnl.plot_pnl_consistency(
            pnl_cat="PNL_GROWTH",
            benchmark_pnl_cat="Unit_Long_EQXR",
            window=window,
            pnl_title="Value-add of the growth strategy",
            ir_title="Consistency of the value-add",
            return_fig=True,
        )
        self.assertEqual(fig.axes[0].get_title(), "Value-add of the growth strategy")
        self.assertEqual(fig.axes[1].get_title(), "Consistency of the value-add")

        with self.assertRaises(TypeError):
            pnl.plot_pnl_consistency(
                pnl_cat=1, benchmark_pnl_cat="Unit_Long_EQXR", window=window
            )

        with self.assertRaises(TypeError):
            pnl.plot_pnl_consistency(
                pnl_cat="PNL_GROWTH",
                benchmark_pnl_cat="Unit_Long_EQXR",
                window=window,
                xcat_labels=["Macro", "Long only"],
            )

        # Label missing for one of the two categories.
        with self.assertRaises(ValueError):
            pnl.plot_pnl_consistency(
                pnl_cat="PNL_GROWTH",
                benchmark_pnl_cat="Unit_Long_EQXR",
                window=window,
                xcat_labels={"PNL_GROWTH": "Macro"},
            )

        # Category not defined on the class.
        with self.assertRaises(ValueError):
            pnl.plot_pnl_consistency(
                pnl_cat="PNL_GROWTH",
                benchmark_pnl_cat="PNL_UNDEFINED",
                window=window,
            )

        # Identical categories would give a zero active return.
        with self.assertRaises(ValueError):
            pnl.plot_pnl_consistency(
                pnl_cat="PNL_GROWTH",
                benchmark_pnl_cat="PNL_GROWTH",
                window=window,
            )

        # Window longer than the available sample.
        with self.assertRaises(ValueError):
            pnl.plot_pnl_consistency(
                pnl_cat="PNL_GROWTH",
                benchmark_pnl_cat="Unit_Long_EQXR",
                window=252 * 100,
            )

        # kind="bars": the active return summed by calendar period.
        fig = pnl.plot_pnl_consistency(
            pnl_cat="PNL_GROWTH",
            benchmark_pnl_cat="Unit_Long_EQXR",
            kind="bars",
            return_fig=True,
        )
        self.assertIsInstance(fig, plt.Figure)
        ax = fig.axes[0]

        dfp = pnl.pnl_df(["PNL_GROWTH", "Unit_Long_EQXR"]).pivot(
            index="real_date", columns="xcat", values="value"
        )
        act = (dfp["PNL_GROWTH"] - dfp["Unit_Long_EQXR"]).dropna()
        act_year = act.groupby(act.index.year).sum()

        heights = [p.get_height() for p in ax.patches]
        np.testing.assert_allclose(heights, act_year.to_numpy(), rtol=1e-9)
        self.assertEqual(
            [t.get_text() for t in ax.get_xticklabels()],
            [str(y) for y in act_year.index],
        )

        # Bars are coloured by sign, and the count of positive periods is annotated.
        pos_colors = {
            tuple(p.get_facecolor()) for p, v in zip(ax.patches, act_year) if v > 0
        }
        neg_colors = {
            tuple(p.get_facecolor()) for p, v in zip(ax.patches, act_year) if v <= 0
        }
        self.assertFalse(pos_colors & neg_colors)
        self.assertIn(
            f"positive periods: {int((act_year > 0).sum())} of {len(act_year)}",
            [t.get_text() for t in ax.texts],
        )

        # The count annotation can be suppressed, and the bars re-binned by period.
        fig = pnl.plot_pnl_consistency(
            pnl_cat="PNL_GROWTH",
            benchmark_pnl_cat="Unit_Long_EQXR",
            kind="bars",
            annotate_count=False,
            return_fig=True,
        )
        self.assertEqual([t.get_text() for t in fig.axes[0].texts], [])

        fig = pnl.plot_pnl_consistency(
            pnl_cat="PNL_GROWTH",
            benchmark_pnl_cat="Unit_Long_EQXR",
            kind="bars",
            freq="Q",
            bar_title="Quarterly active return",
            return_fig=True,
        )
        self.assertEqual(fig.axes[0].get_title(), "Quarterly active return")
        self.assertTrue(
            all(
                re.fullmatch(r"\d{4}Q[1-4]", t.get_text())
                for t in fig.axes[0].get_xticklabels()
            )
        )

        # 'bars' does not need a rolling window, so a long window must not block it.
        try:
            pnl.plot_pnl_consistency(
                pnl_cat="PNL_GROWTH",
                benchmark_pnl_cat="Unit_Long_EQXR",
                kind="bars",
                window=252 * 100,
            )
        except Exception as e:
            self.fail(f"plot_pnl_consistency(kind='bars') raised {e} unexpectedly")

        with self.assertRaises(ValueError):
            pnl.plot_pnl_consistency(
                pnl_cat="PNL_GROWTH",
                benchmark_pnl_cat="Unit_Long_EQXR",
                kind="lines",
            )

        patch.stopall()
        plt.close("all")
        matplotlib.use(mpl_backend)

    def test_plot_pnl_attribution(self):
        plt.close("all")
        patch("matplotlib.pyplot.show").start()
        mpl_backend = matplotlib.get_backend()
        matplotlib.use("Agg")

        pnl = NaivePnL(
            self.dfd,
            ret="EQXR",
            sigs=["CRY", "GROWTH", "INFL"],
            cids=self.cids,
            start="2000-01-01",
            blacklist=self.blacklist,
        )
        pnl.make_pnl(
            sig="GROWTH",
            sig_op="zn_score_pan",
            rebal_freq="monthly",
            rebal_slip=1,
            vol_scale=None,
            pnl_name="PNL_GROWTH",
        )

        try:
            pnl.plot_pnl_attribution(pnl_name="PNL_GROWTH")
        except Exception as e:
            self.fail(f"signal_table raised {e} unexpectedly")

        fig = pnl.plot_pnl_attribution(pnl_name="PNL_GROWTH", return_fig=True)
        self.assertIsInstance(fig, plt.Figure)

        # The rendered table is the signal averaged by calendar year, scaled, with
        # cross-sections on the vertical axis and years on the horizontal.
        sig = pnl.signal_df["PNL_GROWTH"].pivot(
            index="real_date", columns="cid", values="sig"
        )
        sig.columns = pd.Index(sig.columns.astype(str))
        expected = (sig.groupby(sig.index.year).mean()[pnl.cids] * 100).T

        fig = pnl.plot_pnl_attribution(
            pnl_name="PNL_GROWTH", scale=100, fmt=".2f", return_fig=True
        )
        ax = fig.axes[0]
        self.assertEqual(
            [t.get_text() for t in ax.get_yticklabels()], list(expected.index)
        )
        self.assertEqual(
            [t.get_text() for t in ax.get_xticklabels()],
            [str(y) for y in expected.columns],
        )
        np.testing.assert_allclose(
            ax.collections[0].get_array().data.reshape(expected.shape),
            expected.to_numpy(),
            rtol=1e-9,
        )

        # Custom labels are applied to the cross-sections, and the requested order is
        # honoured even when it is not alphabetical.
        fig = pnl.plot_pnl_attribution(
            pnl_name="PNL_GROWTH",
            pnl_cids=["USD", "AUD", "CAD"],
            cid_labels={"AUD": "Australia"},
            return_fig=True,
        )
        self.assertEqual(
            [t.get_text() for t in fig.axes[0].get_yticklabels()],
            ["USD", "Australia", "CAD"],
        )
        # ... and the rows carry the right cross-section's data, not just the label.
        usd_row = ax_arr = fig.axes[0].collections[0].get_array().data.reshape(3, -1)[0]
        np.testing.assert_allclose(
            usd_row,
            sig.groupby(sig.index.year).mean()["USD"].to_numpy(),
            rtol=1e-9,
        )

        # Quarterly and monthly periods are labelled within the year.
        fig = pnl.plot_pnl_attribution(pnl_name="PNL_GROWTH", freq="Q", return_fig=True)
        xticks = [t.get_text() for t in fig.axes[0].get_xticklabels()]
        self.assertTrue(all(re.fullmatch(r"\d{4}Q[1-4]", x) for x in xticks))

        fig = pnl.plot_pnl_attribution(pnl_name="PNL_GROWTH", freq="M", return_fig=True)
        xticks = [t.get_text() for t in fig.axes[0].get_xticklabels()]
        self.assertTrue(all(re.fullmatch(r"\d{4}-\d{2}", x) for x in xticks))

        # A signal that takes both signs gets a symmetric colour scale; a non-negative
        # one gets a scale anchored at zero.
        fig = pnl.plot_pnl_attribution(pnl_name="PNL_GROWTH", return_fig=True)
        vmin, vmax = fig.axes[0].collections[0].get_clim()
        self.assertLess(vmin, 0)
        self.assertAlmostEqual(vmin, -vmax)

        # A portfolio-weight signal: non-negative raw weights summing to one across the
        # panel, as a wealth-allocation strategy would supply them.
        wdf = self.dfd[self.dfd["xcat"] == "EQXR"].pivot(
            index="real_date", columns="cid", values="value"
        )
        wdf.columns = pd.Index(wdf.columns.astype(str))
        wgt = pd.DataFrame(1.0 / len(wdf.columns), index=wdf.index, columns=wdf.columns)
        dfa = wgt.stack().rename("value").reset_index()
        dfa.columns = ["real_date", "cid", "value"]
        dfa["xcat"] = "WEIGHT"
        dfw_in = update_df(self.dfd, dfa)

        wpnl = NaivePnL(
            dfw_in,
            ret="EQXR",
            sigs=["WEIGHT"],
            cids=self.cids,
            start="2000-01-01",
        )
        wpnl.make_pnl(
            sig="WEIGHT",
            sig_op="raw",
            rebal_freq="monthly",
            rebal_slip=1,
            vol_scale=None,
            pnl_name="PNL_WEIGHT",
        )
        fig = wpnl.plot_pnl_attribution(
            pnl_name="PNL_WEIGHT", scale=100, return_fig=True
        )
        self.assertEqual(fig.axes[0].collections[0].get_clim()[0], 0.0)

        # kind="returns": the return category the PnL was built on, summed per period.
        fig = wpnl.plot_pnl_attribution(
            pnl_name="PNL_WEIGHT", kind="returns", return_fig=True
        )
        self.assertIsInstance(fig, plt.Figure)
        pos = wpnl._wide_signal("PNL_WEIGHT", wpnl.cids)
        rets = wpnl._wide_returns(wpnl.cids, pos.index)
        expected_r = rets.groupby(rets.index.year).sum().T
        np.testing.assert_allclose(
            fig.axes[0].collections[0].get_array().data.reshape(expected_r.shape),
            expected_r.to_numpy(),
            rtol=1e-9,
        )

        # kind="contribution": position times return, and the stacked bars must sum to
        # the PnL's own period totals.
        fig = wpnl.plot_pnl_attribution(
            pnl_name="PNL_WEIGHT", kind="contribution", return_fig=True
        )
        self.assertIsInstance(fig, plt.Figure)
        contrib = (pos * rets).groupby(pos.index.year).sum(min_count=1)
        dfn = wpnl.pnl_df(["PNL_WEIGHT"])
        totals = dfn.groupby(dfn["real_date"].dt.year)["value"].sum()
        common = contrib.index.intersection(totals.index)
        np.testing.assert_allclose(
            contrib.loc[common].sum(axis=1).to_numpy(),
            totals.loc[common].to_numpy(),
            atol=1e-9,
        )
        # The markers plot those totals, so the decomposition is visibly reconciled.
        offsets = fig.axes[0].collections[-1].get_offsets()
        np.testing.assert_allclose(
            np.asarray(offsets)[:, 1],
            totals.reindex(contrib.index).to_numpy(),
            atol=1e-9,
        )

        # kind="area": stacked weights over time, optionally the pre-rebalance series.
        fig = wpnl.plot_pnl_attribution(
            pnl_name="PNL_WEIGHT",
            kind="area",
            cid_labels={"AUD": "Australia"},
            return_fig=True,
        )
        self.assertIsInstance(fig, plt.Figure)
        legend = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
        self.assertIn("Australia", legend)
        self.assertEqual(len(legend), len(wpnl.cids))

        # A supplied weights frame is used in place of the applied signal.
        fig = wpnl.plot_pnl_attribution(
            pnl_name="PNL_WEIGHT", kind="area", weights=wgt, return_fig=True
        )
        stacked_top = fig.axes[0].collections[-1].get_paths()[0].vertices[:, 1].max()
        self.assertAlmostEqual(stacked_top, 1.0, places=6)

        with self.assertRaises(TypeError):
            wpnl.plot_pnl_attribution(
                pnl_name="PNL_WEIGHT", kind="area", weights="not a frame"
            )

        # A weights frame missing a cross-section is rejected rather than silently
        # dropping a sleeve from the allocation.
        with self.assertRaises(ValueError):
            wpnl.plot_pnl_attribution(
                pnl_name="PNL_WEIGHT",
                kind="area",
                weights=wgt.drop(columns=[wgt.columns[0]]),
            )

        with self.assertRaises(ValueError):
            pnl.plot_pnl_attribution(pnl_name="PNL_GROWTH", kind="stacked")

        with self.assertRaises(TypeError):
            pnl.plot_pnl_attribution(pnl_name=1)

        with self.assertRaises(ValueError):
            pnl.plot_pnl_attribution(pnl_name="PNL_UNDEFINED")

        with self.assertRaises(ValueError):
            pnl.plot_pnl_attribution(pnl_name="PNL_GROWTH", freq="D")

        with self.assertRaises(ValueError):
            pnl.plot_pnl_attribution(pnl_name="PNL_GROWTH", agg="average")

        with self.assertRaises(ValueError):
            pnl.plot_pnl_attribution(pnl_name="PNL_GROWTH", pnl_cids=["XXX"])

        with self.assertRaises(TypeError):
            pnl.plot_pnl_attribution(pnl_name="PNL_GROWTH", scale="100")

        with self.assertRaises(TypeError):
            pnl.plot_pnl_attribution(pnl_name="PNL_GROWTH", cid_labels=["AUD"])

        with self.assertRaises(TypeError):
            pnl.plot_pnl_attribution(pnl_name="PNL_GROWTH", xlabel=1)

        # No data left after truncation.
        with self.assertRaises(ValueError):
            pnl.plot_pnl_attribution(pnl_name="PNL_GROWTH", start="2100-01-01")

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
