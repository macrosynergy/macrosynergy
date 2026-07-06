import unittest
import numpy as np
import pandas as pd
from typing import List, Tuple

from macrosynergy.securities.index import (
    _resolve_reconstitution_freq,
    _assign_period_labels,
    _build_reconstitution_membership,
    _apply_er_formula,
    compute_daily_weights,
    compute_index_returns,
    compute_excess_returns,
)


def _make_constituents(
    cids: List[str],
    start: str = "2020-01-01",
    end: str = "2020-03-31",
    all_member: bool = True,
) -> pd.DataFrame:
    bdays = pd.bdate_range(start, end)
    membership = 1 if all_member else 0
    rows = [
        {"cid": cid, "real_date": dt, "membership": membership}
        for cid in cids
        for dt in bdays
    ]
    return pd.DataFrame(rows)


def _make_returns(
    cids: List[str],
    start: str = "2020-01-01",
    end: str = "2020-03-31",
    xcat: str = "EQXR",
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    bdays = pd.bdate_range(start, end)
    values = rng.normal(0, 1, (len(cids), len(bdays)))
    rows = []
    for i, cid in enumerate(cids):
        for j, dt in enumerate(bdays):
            rows.append(
                {"cid": cid, "real_date": dt, "xcat": xcat, "value": float(values[i, j])}
            )
    return pd.DataFrame(rows)


class TestResolveReconstitutionFreq(unittest.TestCase):
    def test_returns_reconstitution_freq_when_given(self):
        self.assertEqual(_resolve_reconstitution_freq("M", "Q"), "Q")
        self.assertEqual(_resolve_reconstitution_freq("B", "Y"), "Y")
        self.assertEqual(_resolve_reconstitution_freq("M", "M"), "M")

    def test_returns_rebalance_freq_when_none(self):
        self.assertEqual(_resolve_reconstitution_freq("M", None), "M")
        self.assertEqual(_resolve_reconstitution_freq("Q", None), "Q")
        self.assertEqual(_resolve_reconstitution_freq("B", None), "B")


class TestAssignPeriodLabels(unittest.TestCase):
    def setUp(self):
        self.bdays = pd.bdate_range("2020-01-01", "2020-06-30")

    def test_returns_period_index(self):
        result = _assign_period_labels(self.bdays, "M")
        self.assertIsInstance(result, pd.PeriodIndex)

    def test_length_matches_input(self):
        for freq in ["M", "Q", "Y"]:
            result = _assign_period_labels(self.bdays, freq)
            self.assertEqual(len(result), len(self.bdays))

    def test_monthly_labels_correct(self):
        result = _assign_period_labels(self.bdays, "M")
        jan_mask = self.bdays.month == 1
        self.assertTrue(
            all(p == pd.Period("2020-01", "M") for p in result[jan_mask])
        )
        feb_mask = self.bdays.month == 2
        self.assertTrue(
            all(p == pd.Period("2020-02", "M") for p in result[feb_mask])
        )

    def test_quarterly_labels_correct(self):
        result = _assign_period_labels(self.bdays, "Q")
        q1_mask = self.bdays.month <= 3
        q2_mask = (self.bdays.month >= 4) & (self.bdays.month <= 6)
        self.assertTrue(all(p.quarter == 1 for p in result[q1_mask]))
        self.assertTrue(all(p.quarter == 2 for p in result[q2_mask]))


class TestBuildReconstitutionMembership(unittest.TestCase):
    def _make_membership_wide(self) -> pd.DataFrame:
        bdays = pd.bdate_range("2020-01-01", "2020-02-28")
        mem_wide = pd.DataFrame({"AAA": 1, "BBB": 0}, index=bdays, dtype=int)
        # BBB joins on Jan 10; still absent on first bday of Jan (Jan 2)
        mem_wide.loc[bdays >= pd.Timestamp("2020-01-10"), "BBB"] = 1
        return mem_wide

    def test_snaps_to_first_business_day_of_period(self):
        mem_wide = self._make_membership_wide()
        result = _build_reconstitution_membership(mem_wide, "M")
        bdays = mem_wide.index
        jan_bdays = bdays[bdays.month == 1]
        feb_bdays = bdays[bdays.month == 2]
        # BBB was 0 on Jan 2 (first bday of Jan) → 0 for all January
        self.assertTrue((result.loc[jan_bdays, "BBB"] == 0).all())
        # BBB was 1 on Feb 3 (first bday of Feb) → 1 for all February
        self.assertTrue((result.loc[feb_bdays, "BBB"] == 1).all())

    def test_result_has_same_shape_and_index(self):
        mem_wide = self._make_membership_wide()
        result = _build_reconstitution_membership(mem_wide, "M")
        self.assertEqual(result.shape, mem_wide.shape)
        self.assertTrue(result.index.equals(mem_wide.index))
        self.assertTrue(result.columns.equals(mem_wide.columns))

    def test_always_member_unchanged(self):
        mem_wide = self._make_membership_wide()
        result = _build_reconstitution_membership(mem_wide, "M")
        # AAA is 1 on every first bday → stays 1 for every day
        self.assertTrue((result["AAA"] == 1).all())

    def test_all_days_in_period_get_same_value(self):
        mem_wide = self._make_membership_wide()
        result = _build_reconstitution_membership(mem_wide, "M")
        # All days within each month must share the same membership value
        for col in result.columns:
            for month_val in [1, 2]:
                mask = result.index.month == month_val
                monthly_vals = result.loc[mask, col]
                self.assertEqual(monthly_vals.nunique(), 1)


class TestApplyErFormula(unittest.TestCase):
    def setUp(self):
        self.stock = pd.DataFrame({"AAA": [0.02], "BBB": [0.05]})
        self.bench = pd.Series([0.01])

    def test_ratio_formula(self):
        result = _apply_er_formula(self.stock, self.bench, "ratio")
        self.assertAlmostEqual(result["AAA"].iloc[0], (1.02 / 1.01) - 1, places=12)
        self.assertAlmostEqual(result["BBB"].iloc[0], (1.05 / 1.01) - 1, places=12)

    def test_log_formula(self):
        result = _apply_er_formula(self.stock, self.bench, "log")
        self.assertAlmostEqual(
            result["AAA"].iloc[0], np.log1p(0.02) - np.log1p(0.01), places=12
        )
        self.assertAlmostEqual(
            result["BBB"].iloc[0], np.log1p(0.05) - np.log1p(0.01), places=12
        )

    def test_diff_formula(self):
        result = _apply_er_formula(self.stock, self.bench, "diff")
        self.assertAlmostEqual(result["AAA"].iloc[0], 0.01, places=12)
        self.assertAlmostEqual(result["BBB"].iloc[0], 0.04, places=12)

    def test_zero_excess_when_equal(self):
        bench = pd.Series([0.02])
        result_ratio = _apply_er_formula(self.stock, bench, "ratio")
        self.assertAlmostEqual(result_ratio["AAA"].iloc[0], 0.0, places=12)

        result_log = _apply_er_formula(self.stock, bench, "log")
        self.assertAlmostEqual(result_log["AAA"].iloc[0], 0.0, places=12)

        result_diff = _apply_er_formula(self.stock, bench, "diff")
        self.assertAlmostEqual(result_diff["AAA"].iloc[0], 0.0, places=12)


class TestComputeDailyWeights(unittest.TestCase):
    def setUp(self):
        self.cids = ["AAA", "BBB", "CCC"]
        self.start = "2020-01-01"
        self.end = "2020-06-30"
        self.constituents = _make_constituents(self.cids, self.start, self.end)
        self.returns = _make_returns(self.cids, self.start, self.end)

    def test_output_columns(self):
        result = compute_daily_weights(
            self.constituents.copy(), self.returns.copy(), "M"
        )
        for col in ["real_date", "cid", "value"]:
            self.assertIn(col, result.columns)

    def test_weights_sum_to_one(self):
        result = compute_daily_weights(
            self.constituents.copy(), self.returns.copy(), "M"
        )
        by_date = result.groupby("real_date")["value"].sum()
        self.assertTrue((by_date - 1.0).abs().max() < 1e-10)

    def test_weights_are_positive(self):
        result = compute_daily_weights(
            self.constituents.copy(), self.returns.copy(), "M"
        )
        self.assertTrue((result["value"] > 0).all())

    def test_equal_weights_on_first_date(self):
        result = compute_daily_weights(
            self.constituents.copy(), self.returns.copy(), "M"
        )
        first_date = result["real_date"].min()
        first_day = result[result["real_date"] == first_date]
        expected = 1.0 / len(self.cids)
        self.assertTrue((first_day["value"] - expected).abs().max() < 1e-10)

    def test_non_member_excluded(self):
        constituents = self.constituents.copy()
        constituents.loc[constituents["cid"] == "CCC", "membership"] = 0
        result = compute_daily_weights(constituents, self.returns.copy(), "M")
        self.assertNotIn("CCC", result["cid"].unique())
        self.assertIn("AAA", result["cid"].unique())

    def test_blacklist_excludes_in_period_and_restores_after(self):
        blacklist = {
            "AAA": (pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-31"))
        }
        result = compute_daily_weights(
            self.constituents.copy(), self.returns.copy(), "M", blacklist=blacklist
        )
        jan = result[pd.to_datetime(result["real_date"]).dt.month == 1]
        self.assertNotIn("AAA", jan["cid"].unique())
        self.assertIn("BBB", jan["cid"].unique())
        self.assertIn("CCC", jan["cid"].unique())
        # Weights still sum to 1 in January with AAA absent
        jan_by_date = jan.groupby("real_date")["value"].sum()
        self.assertTrue((jan_by_date - 1.0).abs().max() < 1e-10)
        # AAA returns after blacklist period ends
        feb_plus = result[pd.to_datetime(result["real_date"]).dt.month > 1]
        self.assertIn("AAA", feb_plus["cid"].unique())

    def test_invalid_rebalance_freq_raises(self):
        with self.assertRaises(ValueError):
            compute_daily_weights(
                self.constituents.copy(), self.returns.copy(), "X"
            )

    def test_invalid_reconstitution_freq_raises(self):
        with self.assertRaises(ValueError):
            compute_daily_weights(
                self.constituents.copy(), self.returns.copy(), "M", "Z"
            )

    def test_reconstitution_freq_delays_membership_change(self):
        # Monthly rebalancing, quarterly reconstitution.
        # BBB is absent in January (first bday of Q1) but present from February.
        # With quarterly reconstitution BBB stays excluded for all of Q1 (Jan-Mar).
        cids = ["AAA", "BBB"]
        bdays = pd.bdate_range("2020-01-01", "2020-06-30")
        rows = []
        for dt in bdays:
            rows.append({"cid": "AAA", "real_date": dt, "membership": 1})
            rows.append(
                {"cid": "BBB", "real_date": dt, "membership": 0 if dt.month == 1 else 1}
            )
        constituents = pd.DataFrame(rows)
        returns = _make_returns(cids, "2020-01-01", "2020-06-30")

        result = compute_daily_weights(
            constituents, returns, rebalance_freq="M", reconstitution_freq="Q"
        )
        q1 = result[pd.to_datetime(result["real_date"]).dt.month.isin([1, 2, 3])]
        self.assertNotIn("BBB", q1["cid"].unique())
        q2 = result[pd.to_datetime(result["real_date"]).dt.month.isin([4, 5, 6])]
        self.assertIn("BBB", q2["cid"].unique())


class TestComputeIndexReturns(unittest.TestCase):
    def setUp(self):
        self.cids = ["AAA", "BBB"]
        self.start = "2020-01-01"
        self.end = "2020-03-31"
        self.constituents = _make_constituents(self.cids, self.start, self.end)
        self.returns = _make_returns(self.cids, self.start, self.end)

    def test_returns_two_element_tuple(self):
        result = compute_index_returns(
            self.constituents.copy(), self.returns.copy()
        )
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_daily_index_has_required_columns(self):
        daily_index, _ = compute_index_returns(
            self.constituents.copy(), self.returns.copy()
        )
        for col in ["real_date", "value"]:
            self.assertIn(col, daily_index.columns)

    def test_weights_long_has_required_columns(self):
        _, weights_long = compute_index_returns(
            self.constituents.copy(), self.returns.copy()
        )
        for col in ["real_date", "cid", "value"]:
            self.assertIn(col, weights_long.columns)

    def test_zero_returns_give_zero_index(self):
        bdays = pd.bdate_range(self.start, self.end)
        returns_zero = pd.DataFrame([
            {"cid": cid, "real_date": dt, "xcat": "EQXR", "value": 0.0}
            for cid in self.cids
            for dt in bdays
        ])
        daily_index, _ = compute_index_returns(
            self.constituents.copy(), returns_zero
        )
        self.assertTrue((daily_index["value"].abs() < 1e-10).all())

    def test_single_stock_index_matches_stock_return(self):
        # With one stock and constant returns, the index return equals the stock return.
        bdays = pd.bdate_range(self.start, self.end)
        const_return = 2.0
        constituents = pd.DataFrame([
            {"cid": "AAA", "real_date": dt, "membership": 1} for dt in bdays
        ])
        returns = pd.DataFrame([
            {"cid": "AAA", "real_date": dt, "xcat": "EQXR", "value": const_return}
            for dt in bdays
        ])
        daily_index, _ = compute_index_returns(constituents, returns, "M")
        self.assertTrue((daily_index["value"] - const_return).abs().max() < 1e-10)


class TestComputeExcessReturns(unittest.TestCase):
    def _single_stock_data(
        self, stock_ret: float = 2.0, bench_ret: float = 1.0, start: str = "2020-01-01",
        end: str = "2020-03-31"
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        bdays = pd.bdate_range(start, end)
        returns = pd.DataFrame([
            {"cid": "AAA", "real_date": dt, "xcat": "EQXR", "value": stock_ret}
            for dt in bdays
        ])
        index_returns = pd.DataFrame({"real_date": bdays, "value": bench_ret})
        return returns, index_returns

    def test_diff_method(self):
        stock_ret, bench_ret = 2.0, 1.0
        returns, index_returns = self._single_stock_data(stock_ret, bench_ret)
        result = compute_excess_returns(returns, index_returns, method="diff")
        self.assertTrue(
            (result["value"] - (stock_ret - bench_ret)).abs().max() < 1e-10
        )

    def test_ratio_method(self):
        stock_ret, bench_ret = 2.0, 1.0
        returns, index_returns = self._single_stock_data(stock_ret, bench_ret)
        result = compute_excess_returns(returns, index_returns, method="ratio")
        # 2pp = 0.02 decimal, 1pp = 0.01 decimal
        expected_pct = 100.0 * ((1.02 / 1.01) - 1)
        self.assertTrue((result["value"] - expected_pct).abs().max() < 1e-8)

    def test_log_method(self):
        stock_ret, bench_ret = 2.0, 1.0
        returns, index_returns = self._single_stock_data(stock_ret, bench_ret)
        result = compute_excess_returns(returns, index_returns, method="log")
        expected_pct = 100.0 * (np.log1p(0.02) - np.log1p(0.01))
        self.assertTrue((result["value"] - expected_pct).abs().max() < 1e-8)

    def test_zero_excess_when_stock_equals_bench(self):
        returns, index_returns = self._single_stock_data(stock_ret=1.5, bench_ret=1.5)
        for method in ["ratio", "log", "diff"]:
            result = compute_excess_returns(returns, index_returns, method=method)
            self.assertTrue(result["value"].abs().max() < 1e-10, msg=f"method={method}")

    def test_output_freq_reduces_row_count(self):
        returns, index_returns = self._single_stock_data()
        daily = compute_excess_returns(returns, index_returns)
        monthly = compute_excess_returns(returns, index_returns, output_freq="M")
        # Jan–Mar 2020 → 3 monthly rows vs ~65 daily rows
        self.assertEqual(len(monthly), 3)
        self.assertLess(len(monthly), len(daily))

    def test_output_freq_compounds_correctly(self):
        # One month, constant returns → verify ratio compounding over N business days.
        bdays = pd.bdate_range("2020-01-01", "2020-01-31")
        n = len(bdays)
        returns, index_returns = self._single_stock_data(
            stock_ret=2.0, bench_ret=1.0, start="2020-01-01", end="2020-01-31"
        )
        result = compute_excess_returns(
            returns, index_returns, method="ratio", output_freq="M"
        )
        self.assertEqual(len(result), 1)
        expected = 100.0 * ((1.02 / 1.01) ** n - 1)
        self.assertAlmostEqual(result["value"].iloc[0], expected, places=8)

    def test_invalid_method_raises(self):
        returns, index_returns = self._single_stock_data()
        with self.assertRaises(AssertionError):
            compute_excess_returns(returns, index_returns, method="invalid")

    def test_invalid_output_freq_raises(self):
        returns, index_returns = self._single_stock_data()
        with self.assertRaises(ValueError):
            compute_excess_returns(returns, index_returns, output_freq="X")


if __name__ == "__main__":
    unittest.main()
