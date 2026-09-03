import unittest

import numpy as np
import pandas as pd

from macrosynergy.management.types import QuantamentalDataFrame
from macrosynergy.pnl.pnl_evaluation import evaluate_pnl


class TestEvaluatePnL(unittest.TestCase):
    portfolio_name = "GLB"

    def _make_pnl_qdf(
        self,
        xcat: str,
        values: np.ndarray,
        cid: str = None,
        start: str = "2020-01-01",
    ) -> QuantamentalDataFrame:
        cid = cid or self.portfolio_name
        dates = pd.bdate_range(start=start, periods=len(values))
        return QuantamentalDataFrame(
            pd.DataFrame(
                {
                    "real_date": dates,
                    "cid": cid,
                    "xcat": xcat,
                    "value": np.asarray(values, dtype=float),
                }
            )
        )

    def test_invalid_argument_types(self):
        df_pnl = self._make_pnl_qdf("PNL", np.ones(252))

        with self.assertRaisesRegex(TypeError, "Argument aum"):
            evaluate_pnl(df_pnl=df_pnl, aum="100")

        with self.assertRaisesRegex(TypeError, "Argument df_pnl"):
            evaluate_pnl(df_pnl="not-a-frame", aum=100)

        with self.assertRaisesRegex(TypeError, "Argument df_pnle"):
            evaluate_pnl(df_pnl=df_pnl, aum=100, df_pnle="not-a-frame")

        with self.assertRaisesRegex(TypeError, "Argument df_tcosts"):
            evaluate_pnl(df_pnl=df_pnl, aum=100, df_tcosts="not-a-frame")

        with self.assertRaisesRegex(TypeError, "Argument label_dict"):
            evaluate_pnl(df_pnl=df_pnl, aum=100, label_dict=[])

        with self.assertRaisesRegex(TypeError, "Argument start"):
            evaluate_pnl(df_pnl=df_pnl, aum=100, start=123)

        with self.assertRaisesRegex(TypeError, "Argument end"):
            evaluate_pnl(df_pnl=df_pnl, aum=100, end=567)

        with self.assertRaisesRegex(TypeError, "Argument benchmark_data"):
            evaluate_pnl(df_pnl=df_pnl, aum=100, benchmark_data=np.ndarray([]))

    def test_output_columns_match_input_xcat(self):
        out = evaluate_pnl(
            df_pnl=self._make_pnl_qdf("PNL", np.arange(252, dtype=float)), aum=100
        )
        self.assertEqual(list(out.columns), ["PNL"])

    def test_aum_scales_return_inversely(self):
        df_pnl = self._make_pnl_qdf("PNL", 0.01 * np.ones(252))

        out_100 = evaluate_pnl(df_pnl=df_pnl, aum=100)
        out_200 = evaluate_pnl(df_pnl=df_pnl, aum=200)

        self.assertAlmostEqual(out_100.loc["Return %", "PNL"], 2.52)
        self.assertAlmostEqual(out_200.loc["Return %", "PNL"], 2.52 / 2)
        self.assertAlmostEqual(out_100.loc["St. Dev. %", "PNL"], 0.0)
        self.assertAlmostEqual(out_200.loc["St. Dev. %", "PNL"], 0.0)

    def test_mean_zero_alternating_series(self):
        # Alternating +/-1 PnL with aum=100 => daily returns +/-1%.
        # daily mean = 0, sample std = 1 * sqrt(n / (n - 1)).
        n_days = 600
        values = np.tile([1.0, -1.0], n_days // 2)

        out = evaluate_pnl(df_pnl=self._make_pnl_qdf("PNL", values), aum=100)

        self.assertAlmostEqual(out.loc["Return %", "PNL"], 0.0)
        expected_std = 1 * np.sqrt(n_days / (n_days - 1)) * np.sqrt(252)
        self.assertAlmostEqual(out.loc["St. Dev. %", "PNL"], expected_std)
        self.assertAlmostEqual(out.loc["Sharpe Ratio", "PNL"], 0.0)

    def test_df_pnle_adds_pnle_column(self):
        out = evaluate_pnl(
            df_pnl=self._make_pnl_qdf("PNL", 0.01 * np.ones(252)),
            aum=100,
            df_pnle=self._make_pnl_qdf("PNLe", np.full(252, 0.02)),
        )

        self.assertEqual(sorted(out.columns.tolist()), ["PNL", "PNLe"])
        self.assertAlmostEqual(out.loc["Return %", "PNL"], 2.52)
        self.assertAlmostEqual(out.loc["Return %", "PNLe"], 2.52 * 2)

    def test_df_tcosts_adds_transaction_cost_row(self):
        out = evaluate_pnl(
            df_pnl=self._make_pnl_qdf("PNL", np.ones(252)),
            aum=100,
            df_tcosts=self._make_pnl_qdf("TCOST", np.full(252, 0.5)),
        )

        self.assertIn("Transaction Cost", out.index)
        self.assertAlmostEqual(out.loc["Transaction Cost", "PNL"], 252 * 0.5)

    def test_df_tcosts_zero_for_pnle_column(self):
        # Costs are attributed only to the net-of-costs series; the
        # excluding-costs column carries a zero.
        out = evaluate_pnl(
            df_pnl=self._make_pnl_qdf("PNL", np.ones(252)),
            aum=100,
            df_pnle=self._make_pnl_qdf("PNLe", np.full(252, 2.0)),
            df_tcosts=self._make_pnl_qdf("TCOST", np.full(252, 0.5)),
        )

        self.assertAlmostEqual(out.loc["Transaction Cost", "PNL"], 252 * 0.5)
        self.assertAlmostEqual(out.loc["Transaction Cost", "PNLe"], 0.0)

    def test_transaction_costs_filtered_by_portfolio_name(self):
        # Cost rows for other cids must not inflate the total.
        tcosts = pd.concat(
            [
                self._make_pnl_qdf("TCOST", np.full(252, 0.5)),
                self._make_pnl_qdf("TCOST", np.full(252, 9.0), cid="USD"),
            ],
            ignore_index=True,
        )

        out = evaluate_pnl(
            df_pnl=self._make_pnl_qdf("PNL", np.ones(252)),
            aum=100,
            df_tcosts=tcosts,
            portfolio_name=self.portfolio_name,
        )

        self.assertAlmostEqual(out.loc["Transaction Cost", "PNL"], 252 * 0.5)

    def test_portfolio_name_selects_matching_cid(self):
        df_pnl = pd.concat(
            [
                self._make_pnl_qdf("PNL", 0.01 * np.ones(252)),
                self._make_pnl_qdf("PNL", 0.99 * np.ones(252), cid="USD"),
            ],
            ignore_index=True,
        )

        out = evaluate_pnl(df_pnl=df_pnl, aum=100, portfolio_name=self.portfolio_name)

        self.assertEqual(list(out.columns), ["PNL"])
        self.assertAlmostEqual(out.loc["Return %", "PNL"], 2.52)

    def test_label_dict_renames_columns(self):
        out = evaluate_pnl(
            df_pnl=self._make_pnl_qdf("PNL", np.ones(252)),
            aum=100,
            label_dict={"PNL": "Strategy A"},
        )
        self.assertEqual(list(out.columns), ["Strategy A"])

    def test_end_filters_dates(self):
        n_days = 252
        df_pnl = self._make_pnl_qdf(
            "PNL", np.arange(n_days, dtype=float), start="2020-01-01"
        )
        end_short = pd.bdate_range("2020-01-01", periods=21)[-1].strftime("%Y-%m-%d")

        full = evaluate_pnl(df_pnl=df_pnl, aum=100)
        clipped = evaluate_pnl(df_pnl=df_pnl, aum=100, end=end_short)

        self.assertGreater(
            full.loc["Traded Months", "PNL"],
            clipped.loc["Traded Months", "PNL"],
        )

    def test_start_filters_dates(self):
        n_days = 252
        df_pnl = self._make_pnl_qdf(
            "PNL", np.arange(n_days, dtype=float), start="2020-01-01"
        )
        start_late = pd.bdate_range("2020-01-01", periods=200)[-1].strftime("%Y-%m-%d")

        full = evaluate_pnl(df_pnl=df_pnl, aum=100)
        clipped = evaluate_pnl(df_pnl=df_pnl, aum=100, start=start_late)

        self.assertGreater(
            full.loc["Traded Months", "PNL"],
            clipped.loc["Traded Months", "PNL"],
        )

    def test_benchmark_data_adds_correlation_row(self):
        n_days = 252
        values = np.arange(n_days, dtype=float)

        bm = pd.DataFrame(
            {
                "real_date": pd.bdate_range("2020-01-01", periods=n_days),
                "cid": "USD",
                "xcat": "EQ",
                "value": values,
            }
        )

        out = evaluate_pnl(
            df_pnl=self._make_pnl_qdf("PNL", values), aum=100, benchmark_data=bm
        )

        self.assertIn("USD_EQ correl", out.index)
        self.assertAlmostEqual(out.loc["USD_EQ correl", "PNL"], 1.0)

    def test_empty_benchmark_data_adds_no_correlation_row(self):
        out = evaluate_pnl(
            df_pnl=self._make_pnl_qdf("PNL", np.arange(252, dtype=float)),
            aum=100,
            benchmark_data=pd.DataFrame(),
        )
        self.assertFalse(any("correl" in str(row) for row in out.index))

    def test_benchmark_data_not_mutated(self):
        n_days = 100
        bm = pd.DataFrame(
            {
                "real_date": pd.bdate_range("2020-01-01", periods=n_days),
                "cid": "USD",
                "xcat": "EQ",
                "value": np.arange(n_days, dtype=float),
            }
        )
        original_cols = list(bm.columns)

        evaluate_pnl(
            df_pnl=self._make_pnl_qdf("PNL", np.arange(n_days, dtype=float)),
            aum=100,
            benchmark_data=bm,
        )

        self.assertEqual(list(bm.columns), original_cols)


if __name__ == "__main__":
    unittest.main()
