import unittest
from typing import Sequence
from unittest.mock import patch

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from macrosynergy.visuals.proxy_pnl_visualisers import (
    _cost_columns,
    _ordered_levels,
    _resolve_cost_columns,
    _sensitivity_frame,
    compare_proxy_pnls,
    implied_leverage_plot,
    plot_metric_sensitivity,
    plot_metrics_before_and_after_costs,
    transaction_cost_heatmap,
)


def make_qdf(
    cid: str,
    xcat: str,
    values: Sequence[float],
    start: str = "2020-01-01",
) -> pd.DataFrame:
    """One quantamental series, in the long format the visualisers expect."""
    return pd.DataFrame(
        {
            "real_date": pd.bdate_range(start=start, periods=len(values)),
            "cid": cid,
            "xcat": xcat,
            "value": np.asarray(values, dtype=float),
        }
    )


def make_pnl_frames(
    portfolio: str,
    net: Sequence[float],
    gross: Sequence[float],
) -> "tuple[pd.DataFrame, pd.DataFrame]":
    """The PnL including and excluding costs of a single portfolio."""
    return (
        make_qdf(portfolio, f"{portfolio}_PNL", net),
        make_qdf(portfolio, f"{portfolio}_PNLe", gross),
    )


def make_eval(
    net: Sequence[float],
    gross: Sequence[float],
    metrics: Sequence[str] = ("Sharpe Ratio", "Return %"),
    pnl_name: str = "GLB_STRAT_PNL",
) -> pd.DataFrame:
    """
    An `evaluate_pnl` style output: one row per metric, one column for the PnL
    net of costs and one for the PnL gross of costs.
    """
    return pd.DataFrame(
        {pnl_name: list(net), f"{pnl_name}e": list(gross)},
        index=list(metrics),
    )


class PlotTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mpl_backend = matplotlib.get_backend()
        matplotlib.use("Agg")
        cls.mock_show = patch("matplotlib.pyplot.show").start()

    @classmethod
    def tearDownClass(cls) -> None:
        patch.stopall()
        plt.close("all")
        matplotlib.use(cls.mpl_backend)

    def tearDown(self) -> None:
        plt.close("all")


class TestOrderedLevels(unittest.TestCase):
    def test_deduplicates_in_order_of_first_appearance(self):
        levels = _ordered_levels(["b", "a", "b", "c"], None, "group_order")
        self.assertEqual(levels, ["b", "a", "c"])

    def test_requested_order_is_honoured(self):
        levels = _ordered_levels(["b", "a", "c"], ["a", "b", "c"], "group_order")
        self.assertEqual(levels, ["a", "b", "c"])

    def test_requested_order_may_drop_levels(self):
        levels = _ordered_levels(["a", "b", "c"], ["c", "a"], "group_order")
        self.assertEqual(levels, ["c", "a"])

    def test_unknown_level_raises(self):
        with self.assertRaisesRegex(ValueError, "group_order"):
            _ordered_levels(["a", "b"], ["a", "z"], "group_order")


class TestResolveCostColumns(unittest.TestCase):
    def test_resolved_from_the_pnl_name_suffix(self):
        df = make_eval(net=[1.0], gross=[2.0], metrics=["Sharpe Ratio"])

        net, gross = _resolve_cost_columns(df=df, key="A")

        self.assertEqual((net, gross), ("GLB_STRAT_PNL", "GLB_STRAT_PNLe"))

    def test_resolved_from_display_labels(self):
        df = pd.DataFrame(
            {"Portfolio incl. costs": [1.0], "Portfolio excl. costs": [2.0]},
            index=["Sharpe Ratio"],
        )

        net, gross = _resolve_cost_columns(df=df, key="A")

        self.assertEqual(net, "Portfolio incl. costs")
        self.assertEqual(gross, "Portfolio excl. costs")

    def test_explicit_columns_take_precedence(self):
        df = make_eval(net=[1.0], gross=[2.0], metrics=["Sharpe Ratio"])

        net, gross = _resolve_cost_columns(
            df=df, key="A", net_col="GLB_STRAT_PNLe", gross_col="GLB_STRAT_PNL"
        )

        self.assertEqual((net, gross), ("GLB_STRAT_PNLe", "GLB_STRAT_PNL"))

    def test_unidentifiable_columns_raise(self):
        df = pd.DataFrame({"one": [1.0], "two": [2.0]}, index=["Sharpe Ratio"])

        with self.assertRaisesRegex(ValueError, "Could not identify the net column"):
            _resolve_cost_columns(df=df, key="A")

    def test_net_and_gross_resolving_to_one_column_raises(self):
        df = make_eval(net=[1.0], gross=[2.0], metrics=["Sharpe Ratio"])

        with self.assertRaisesRegex(ValueError, "both"):
            _resolve_cost_columns(
                df=df, key="A", net_col="GLB_STRAT_PNL", gross_col="GLB_STRAT_PNL"
            )

    def test_cost_columns_resolves_every_output(self):
        evals = {
            "A": make_eval(net=[1.0], gross=[2.0], metrics=["Sharpe Ratio"]),
            "B": pd.DataFrame(
                {"Net": [1.0], "Gross": [2.0]},
                index=["Sharpe Ratio"],
            ),
        }

        self.assertEqual(
            _cost_columns(evals),
            {"A": ("GLB_STRAT_PNL", "GLB_STRAT_PNLe"), "B": ("Net", "Gross")},
        )


class TestSensitivityFrame(unittest.TestCase):
    def test_pair_keys_give_a_single_unnamed_group(self):
        evals = {("base", 5): pd.DataFrame(), ("alt", 10): pd.DataFrame()}

        lines, groups, series, grouped = _sensitivity_frame(evals, None, None)

        self.assertFalse(grouped)
        self.assertEqual(groups, [None])
        self.assertEqual(series, ["base", "alt"])
        self.assertEqual(lines[(None, "base")], [(5.0, ("base", 5))])

    def test_triple_keys_add_a_group_level(self):
        evals = {
            ("G1", "base", 5): pd.DataFrame(),
            ("G2", "base", 5): pd.DataFrame(),
        }

        lines, groups, series, grouped = _sensitivity_frame(evals, None, None)

        self.assertTrue(grouped)
        self.assertEqual(groups, ["G1", "G2"])
        self.assertEqual(series, ["base"])
        self.assertEqual(lines[("G2", "base")], [(5.0, ("G2", "base", 5))])

    def test_points_are_sorted_by_x(self):
        evals = {
            ("base", 10): pd.DataFrame(),
            ("base", 2): pd.DataFrame(),
            ("base", 5): pd.DataFrame(),
        }

        lines, _, _, _ = _sensitivity_frame(evals, None, None)

        self.assertEqual([x for x, _ in lines[(None, "base")]], [2.0, 5.0, 10.0])

    def test_unsupported_key_length_raises(self):
        evals = {("base",): pd.DataFrame()}

        with self.assertRaisesRegex(ValueError, "triples or"):
            _sensitivity_frame(evals, None, None)

    def test_mixed_key_lengths_raise(self):
        evals = {("base", 5): pd.DataFrame(), ("G1", "base", 5): pd.DataFrame()}

        with self.assertRaisesRegex(ValueError, "same length"):
            _sensitivity_frame(evals, None, None)


class TestTransactionCostHeatmap(PlotTestCase):
    def setUp(self) -> None:
        # Per (cid, xcat) the values below sum to 3, 4, 5 and 6 respectively.
        self.df = pd.concat(
            [
                make_qdf("AUD", "FX_TCOST", [1.0, 2.0]),
                make_qdf("AUD", "IRS_TCOST", [4.0]),
                make_qdf("CAD", "FX_TCOST", [5.0]),
                make_qdf("CAD", "IRS_TCOST", [6.0]),
                make_qdf("GLB", "FX_TCOST", [100.0]),
                make_qdf("AUD", "XR", [999.0]),
            ],
            ignore_index=True,
        )

    def cell_values(self, ax: plt.Axes) -> np.ndarray:
        values = np.asarray(ax.collections[0].get_array()).ravel()
        return np.sort(values[~np.isnan(values)])

    def test_costs_are_summed_and_non_cost_categories_dropped(self):
        ax = transaction_cost_heatmap(self.df)

        np.testing.assert_allclose(self.cell_values(ax), [3.0, 4.0, 5.0, 6.0])

    def test_excluded_cross_sections_are_dropped(self):
        ax = transaction_cost_heatmap(self.df, exclude_cids=("GLB", "CAD"))

        self.assertEqual({t.get_text() for t in ax.get_xticklabels()}, {"AUD"})


class TestCompareProxyPnls(PlotTestCase):
    def setUp(self) -> None:
        self.pnl, self.pnle = make_pnl_frames("GLB", net=[1.0] * 5, gross=[2.0] * 5)

    def call(self, **kwargs):
        kwargs.setdefault("pnl_dfs", [self.pnl])
        kwargs.setdefault("pnle_dfs", [self.pnle])
        kwargs.setdefault("portfolio_names", ["GLB"])
        return compare_proxy_pnls(**kwargs)

    def test_one_subplot_per_portfolio(self):
        pnls, pnles, names = [], [], []
        for i in range(4):
            name = f"P{i}"
            net, gross = make_pnl_frames(name, net=[1.0] * 5, gross=[2.0] * 5)
            pnls.append(net)
            pnles.append(gross)
            names.append(name)

        fig, axes = compare_proxy_pnls(
            pnl_dfs=pnls, pnle_dfs=pnles, portfolio_names=names
        )

        # four portfolios over at most three columns, with the spare panel removed
        self.assertEqual(axes.shape, (2, 3))
        self.assertEqual(len(fig.axes), 4)

    def test_cumsum_false_plots_the_raw_values(self):
        _, axes = self.call(cumsum=False)

        for line in axes[0, 0].lines[:2]:
            self.assertEqual(len(set(line.get_ydata())), 1)

    def test_aum_rescales_to_percent_of_risk_capital(self):
        _, axes = self.call(aum=50)

        # 100 * 1 / 50 = 2 per day, cumulated over five days
        finals = {line.get_ydata()[-1] for line in axes[0, 0].lines[:2]}
        self.assertEqual(finals, {10.0, 20.0})
        self.assertEqual(axes[0, 0].get_ylabel(), "% of risk capital")

    def test_ylabel_defaults_to_currency_without_aum(self):
        _, axes = self.call()

        self.assertEqual(axes[0, 0].get_ylabel(), "USD mn")

    def test_invalid_aum_raises(self):
        with self.assertRaisesRegex(TypeError, "must hold numbers"):
            self.call(aum="100")

        with self.assertRaisesRegex(ValueError, "must be positive"):
            self.call(aum=0)

        with self.assertRaisesRegex(ValueError, "one number per portfolio"):
            self.call(aum=[100, 200])


class TestImpliedLeveragePlot(PlotTestCase):
    def setUp(self) -> None:
        self.npos = pd.concat(
            [
                make_qdf("GLB", "A_NPOS", [0.0, 1.0, -2.0]),
                make_qdf("GLB", "B_NPOS", [0.0, 3.0, 4.0]),
            ],
            ignore_index=True,
        )

    def test_leverage_is_gross_exposure_over_aum(self):
        _, ax = implied_leverage_plot(
            self.npos, labels="Strategy", aum=2, drop_leading_zeros=False
        )

        self.assertEqual(len(ax.lines), 1)
        np.testing.assert_allclose(ax.lines[0].get_ydata(), [0.0, 2.0, 3.0])

    def test_leading_zeros_are_dropped(self):
        _, ax = implied_leverage_plot(self.npos, labels="Strategy", aum=2)

        np.testing.assert_allclose(ax.lines[0].get_ydata(), [2.0, 3.0])

    def test_one_line_per_dataframe(self):
        _, ax = implied_leverage_plot(
            [self.npos, self.npos], labels=["One", "Two"], aum=2
        )

        _, labels = ax.get_legend_handles_labels()
        self.assertEqual(labels, ["One", "Two"])

    def test_baseline_adds_a_reference_line(self):
        _, ax = implied_leverage_plot(
            self.npos, labels="Strategy", aum=2, baseline=True
        )

        self.assertEqual(len(ax.lines), 2)
        _, labels = ax.get_legend_handles_labels()
        self.assertIn("1x leverage", labels)


class TestPlotMetricsBeforeAndAfterCosts(PlotTestCase):
    def setUp(self) -> None:
        self.evals = {
            "A": make_eval(net=[1.0, 10.0], gross=[1.5, 11.0]),
            "B": make_eval(net=[2.0, 20.0], gross=[2.5, 21.0]),
        }

    def test_one_panel_per_metric_and_one_bar_group_per_key(self):
        fig, axes = plot_metrics_before_and_after_costs(self.evals)

        self.assertEqual(axes.shape, (1, 2))
        self.assertEqual(len(fig.axes), 2)
        self.assertEqual(
            [t.get_text() for t in axes[0, 0].get_xticklabels()], ["A", "B"]
        )

    def test_bars_hold_the_gross_and_net_values(self):
        _, axes = plot_metrics_before_and_after_costs(
            self.evals, metrics="Sharpe Ratio"
        )

        # the dashed gross outlines are drawn first, then the filled net bars
        heights = [patch.get_height() for patch in axes[0, 0].patches]
        self.assertEqual(heights, [1.5, 2.5, 1.0, 2.0])

    def test_show_gross_false_draws_the_net_bars_alone(self):
        _, axes = plot_metrics_before_and_after_costs(
            self.evals, metrics="Sharpe Ratio", show_gross=False
        )

        heights = [patch.get_height() for patch in axes[0, 0].patches]
        self.assertEqual(heights, [1.0, 2.0])

    def test_metric_labels_title_the_panels(self):
        _, axes = plot_metrics_before_and_after_costs(
            self.evals,
            metrics="Sharpe Ratio",
            metric_labels={"Sharpe Ratio": "SR"},
        )

        self.assertEqual(axes[0, 0].get_title(), "SR")

    def test_invalid_evals_raise(self):
        with self.assertRaisesRegex(ValueError, "must be a dict"):
            plot_metrics_before_and_after_costs({})

        with self.assertRaisesRegex(TypeError, "must be a pd.DataFrame"):
            plot_metrics_before_and_after_costs({"A": "not a frame"})

        with self.assertRaisesRegex(ValueError, "all be single labels"):
            plot_metrics_before_and_after_costs(
                {"A": self.evals["A"], ("B", "x"): self.evals["B"]}
            )

        with self.assertRaisesRegex(ValueError, "Metrics not found"):
            plot_metrics_before_and_after_costs(self.evals, metrics="Nonexistent")


class TestPlotMetricSensitivity(PlotTestCase):
    def setUp(self) -> None:
        # a sweep over x in (5, 10) for two series
        self.evals = {
            ("base", 5): make_eval(net=[1.0, 10.0], gross=[1.5, 11.0]),
            ("base", 10): make_eval(net=[2.0, 20.0], gross=[2.5, 21.0]),
            ("alt", 5): make_eval(net=[3.0, 30.0], gross=[3.5, 31.0]),
            ("alt", 10): make_eval(net=[4.0, 40.0], gross=[4.5, 41.0]),
        }

    def test_pair_keys_give_one_panel_per_metric(self):
        fig, axes = plot_metric_sensitivity(self.evals)

        self.assertEqual(axes.shape, (1, 2))
        self.assertEqual(len(fig.axes), 2)
        self.assertEqual(axes[0, 0].get_title(), "Sharpe Ratio")

    def test_triple_keys_give_a_panel_per_group(self):
        evals = {
            ("G1", "base", 5): self.evals[("base", 5)],
            ("G1", "base", 10): self.evals[("base", 10)],
            ("G2", "base", 5): self.evals[("alt", 5)],
            ("G2", "base", 10): self.evals[("alt", 10)],
        }

        _, axes = plot_metric_sensitivity(evals, metrics="Sharpe Ratio")

        self.assertEqual(axes.shape, (1, 2))
        self.assertEqual(axes[0, 0].get_title(), "G1")
        self.assertEqual(axes[0, 1].get_title(), "G2")

    def test_lines_follow_the_sweep(self):
        _, axes = plot_metric_sensitivity(self.evals, metrics="Sharpe Ratio")

        # per series the dashed gross line is drawn before the solid net line
        gross_base, net_base, gross_alt, net_alt = axes[0, 0].lines

        np.testing.assert_allclose(net_base.get_xdata(), [5.0, 10.0])
        np.testing.assert_allclose(net_base.get_ydata(), [1.0, 2.0])
        np.testing.assert_allclose(gross_base.get_ydata(), [1.5, 2.5])
        np.testing.assert_allclose(net_alt.get_ydata(), [3.0, 4.0])
        np.testing.assert_allclose(gross_alt.get_ydata(), [3.5, 4.5])

    def test_show_gross_false_draws_the_net_lines_alone(self):
        _, axes = plot_metric_sensitivity(
            self.evals, metrics="Sharpe Ratio", show_gross=False
        )

        self.assertEqual(len(axes[0, 0].lines), 2)

    def test_show_gross_restricted_to_unplotted_metric_raises(self):
        with self.assertRaisesRegex(ValueError, "not plotted"):
            plot_metric_sensitivity(
                self.evals, metrics="Sharpe Ratio", show_gross=["Return %"]
            )

    def test_invalid_evals_raise(self):
        with self.assertRaisesRegex(ValueError, "non-empty dict"):
            plot_metric_sensitivity({})

        with self.assertRaisesRegex(TypeError, "must be a pd.DataFrame"):
            plot_metric_sensitivity({("base", 5): "not a frame"})

        with self.assertRaisesRegex(ValueError, "Metrics not found"):
            plot_metric_sensitivity(self.evals, metrics="Nonexistent")


if __name__ == "__main__":
    unittest.main()
