import unittest
import warnings
from typing import Dict, List

import numpy as np
import pandas as pd

from macrosynergy.management.types import QuantamentalDataFrame
from macrosynergy.securities.analyse import (
    ACTIVE_WEIGHT_STATS,
    STANDALONE_WEIGHT_STATS,
    WEIGHT_STAT_LABELS,
    PortfolioAnalyser,
    _align_active,
    _as_wide,
    _cid_label_map,
    _concentration_stats,
    _group_labels,
    _no_trade_weights,
    _stat_xcat,
    _trade_dates,
    _turnover_against,
    _weight_autocorr,
    weight_stat_labels,
)

# The shared fixture spans 60 business days; weekly rebalancing leaves ~13 trade dates,
# enough for the turnover and autocorrelation columns to carry real readings.
FREQ = "W"


def _wide(values: List[List[float]], cids: List[str], start: str = "2020-01-01"):
    idx = pd.bdate_range(start, periods=len(values))
    frame = pd.DataFrame(values, index=idx, columns=cids, dtype=float)
    frame.index.name = "real_date"
    frame.columns.name = "cid"
    return frame


def _to_long(wide: pd.DataFrame, xcat: str = None) -> pd.DataFrame:
    long = (
        wide.rename_axis("real_date")
        .reset_index()
        .melt(id_vars="real_date", var_name="cid", value_name="value")
    )
    if xcat is not None:
        long["xcat"] = xcat
    return long


def _random_portfolio(seed: int = 0, n_cids: int = 8, n_dates: int = 60):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_dates)
    cids = [f"S{i:02d}" for i in range(n_cids)]
    raw = pd.DataFrame(rng.random((n_dates, n_cids)), index=dates, columns=cids)
    weights = raw.div(raw.sum(axis=1), axis=0)
    bm_raw = pd.DataFrame(rng.random((n_dates, n_cids)), index=dates, columns=cids)
    benchmark = bm_raw.div(bm_raw.sum(axis=1), axis=0)
    returns = pd.DataFrame(
        rng.normal(0, 1, (n_dates, n_cids)), index=dates, columns=cids
    )
    groups: Dict[str, str] = {
        cid: ("TECH" if i % 2 else "FINS") for i, cid in enumerate(cids)
    }
    return weights, benchmark, returns, groups


class TestAsWide(unittest.TestCase):
    def setUp(self):
        self.wide = _wide([[0.6, 0.4], [0.5, 0.5]], ["AAA", "BBB"])

    def test_wide_passthrough(self):
        result = _as_wide(self.wide, "weights")
        pd.testing.assert_frame_equal(result, self.wide, check_freq=False)

    def test_long_roundtrip(self):
        result = _as_wide(_to_long(self.wide), "weights")
        pd.testing.assert_frame_equal(result, self.wide, check_freq=False)

    def test_long_with_single_xcat(self):
        result = _as_wide(_to_long(self.wide, xcat="EQXR"), "weights")
        pd.testing.assert_frame_equal(result, self.wide, check_freq=False)

    def test_real_date_column_is_used_as_index(self):
        frame = self.wide.reset_index()
        pd.testing.assert_frame_equal(
            _as_wide(frame, "weights"), self.wide, check_freq=False
        )

    def test_categorical_cids_give_plain_string_columns(self):
        # A QuantamentalDataFrame holds `cid` as categorical, which would otherwise
        # pivot into a CategoricalIndex and make the same portfolio compare unequal
        # depending on whether it arrived long or wide.
        long = _to_long(self.wide)
        long["cid"] = pd.Categorical(long["cid"], categories=["AAA", "BBB", "CCC"])
        result = _as_wide(long, "weights")

        self.assertEqual(list(result.columns), ["AAA", "BBB"])
        self.assertNotIsInstance(result.columns, pd.CategoricalIndex)
        pd.testing.assert_index_equal(
            result.columns, _as_wide(self.wide, "weights").columns
        )

    def test_caller_frame_is_not_mutated(self):
        long = _to_long(self.wide)
        long["real_date"] = long["real_date"].astype(str)
        before = long.copy()
        _as_wide(long, "weights")
        pd.testing.assert_frame_equal(long, before)

    def test_multiple_xcats_raise(self):
        long = _to_long(self.wide, xcat="EQXR")
        long.loc[0, "xcat"] = "OTHER"
        with self.assertRaisesRegex(ValueError, "more than one xcat"):
            _as_wide(long, "returns")

    def test_duplicate_observations_raise(self):
        long = pd.concat([_to_long(self.wide)] * 2, ignore_index=True)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            _as_wide(long, "weights")

    def test_missing_columns_raise(self):
        long = _to_long(self.wide).drop(columns=["value"])
        with self.assertRaisesRegex(ValueError, "missing columns"):
            _as_wide(long, "weights")

    def test_non_dataframe_raises(self):
        with self.assertRaises(TypeError):
            _as_wide([1, 2, 3], "weights")

    def test_empty_raises(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            _as_wide(pd.DataFrame(), "weights")

    def test_numeric_index_raises(self):
        # A RangeIndex would otherwise be read as epoch nanoseconds.
        frame = self.wide.reset_index(drop=True)
        with self.assertRaisesRegex(ValueError, "index is numeric"):
            _as_wide(frame, "weights")

    def test_unparseable_index_raises(self):
        frame = self.wide.copy()
        frame.index = ["not-a-date", "also-not"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # pandas' dateutil fallback
            with self.assertRaisesRegex(
                ValueError, "could not be interpreted as dates"
            ):
                _as_wide(frame, "weights")

    def test_non_numeric_values_raise(self):
        frame = self.wide.astype(object)
        frame.iloc[0, 0] = "x"
        with self.assertRaisesRegex(ValueError, "non-numeric"):
            _as_wide(frame, "weights")


class TestAlignActive(unittest.TestCase):
    def test_union_of_cids_and_intersection_of_dates(self):
        weights = _wide([[1.0], [1.0], [1.0]], ["AAA"])
        benchmark = _wide([[0.5, 0.5], [0.4, 0.6]], ["AAA", "BBB"])
        w, b, active = _align_active(weights, benchmark)

        self.assertEqual(list(w.columns), ["AAA", "BBB"])
        self.assertEqual(len(w.index), 2)
        # A name held on only one side is a full position against a zero weight.
        self.assertTrue((w["BBB"] == 0.0).all())
        pd.testing.assert_frame_equal(active, w - b)

    def test_disjoint_dates_raise(self):
        weights = _wide([[1.0]], ["AAA"], start="2020-01-01")
        benchmark = _wide([[1.0]], ["AAA"], start="2021-01-01")
        with self.assertRaisesRegex(ValueError, "share no dates"):
            _align_active(weights, benchmark)


class TestConcentrationStats(unittest.TestCase):
    def test_equal_weights_give_effective_n_equal_to_count(self):
        stats = _concentration_stats(_wide([[0.25] * 4], [f"S{i}" for i in range(4)]))
        self.assertEqual(stats["n_holdings"].iloc[0], 4)
        self.assertAlmostEqual(stats["effective_n"].iloc[0], 4.0)
        self.assertAlmostEqual(stats["weight"].iloc[0], 100.0)
        self.assertAlmostEqual(stats["gross_weight"].iloc[0], 100.0)

    def test_zero_and_nan_weights_are_not_holdings(self):
        stats = _concentration_stats(
            _wide([[0.5, 0.0, np.nan, 0.5]], ["A", "B", "C", "D"])
        )
        self.assertEqual(stats["n_holdings"].iloc[0], 2)
        self.assertAlmostEqual(stats["effective_n"].iloc[0], 2.0)

    def test_short_positions_count_gross_not_net(self):
        stats = _concentration_stats(_wide([[0.75, -0.25]], ["A", "B"]))
        self.assertAlmostEqual(stats["weight"].iloc[0], 50.0)
        self.assertAlmostEqual(stats["gross_weight"].iloc[0], 100.0)

    def test_empty_date_has_no_reading(self):
        stats = _concentration_stats(_wide([[np.nan, 0.0]], ["A", "B"]))
        self.assertEqual(stats["n_holdings"].iloc[0], 0)
        for col in ["effective_n", "weight", "gross_weight"]:
            self.assertTrue(np.isnan(stats[col].iloc[0]))


class TestTradeDates(unittest.TestCase):
    def test_first_business_day_of_each_period(self):
        idx = pd.bdate_range("2020-01-01", "2020-03-31")
        result = _trade_dates(idx, "M")
        self.assertEqual(
            [str(d.date()) for d in result],
            ["2020-01-01", "2020-02-03", "2020-03-02"],
        )

    def test_coarser_frequency_gives_fewer_dates(self):
        idx = pd.bdate_range("2020-01-01", "2020-12-31")
        self.assertEqual(len(_trade_dates(idx, "Q")), 4)
        self.assertEqual(len(_trade_dates(idx, "M")), 12)
        self.assertEqual(len(_trade_dates(idx, "B")), len(idx))


class TestNoTradeWeights(unittest.TestCase):
    def setUp(self):
        # Three dates; a rebalancing on the first and the last.
        self.weights = _wide([[0.5, 0.5], [0.5, 0.5], [0.2, 0.8]], ["A", "B"])
        self.trade_dates = self.weights.index[[0, 2]]

    def test_without_returns_the_anchor_is_carried_flat(self):
        carried = _no_trade_weights(self.weights, self.trade_dates)
        self.assertEqual(list(carried.index), [self.weights.index[2]])
        np.testing.assert_allclose(carried.iloc[0], [0.5, 0.5])

    def test_returns_grow_the_anchor_over_the_whole_period(self):
        # A doubles over the two days, B is flat: 0.5/0.5 becomes 1.0/0.5, i.e. 2:1.
        returns = _wide([[100.0, 0.0], [0.0, 0.0], [0.0, 0.0]], ["A", "B"])
        carried = _no_trade_weights(self.weights, self.trade_dates, returns)
        np.testing.assert_allclose(carried.iloc[0], [2 / 3, 1 / 3])

    def test_drifting_weights_are_reproduced_exactly(self):
        # Weights that already drift must come back unchanged, so that the turnover
        # measured against them on a non-trading rebalancing is zero.
        returns = _wide([[10.0, -5.0], [3.0, 2.0], [0.0, 0.0]], ["A", "B"])
        grown = np.array([0.5 * 1.10 * 1.03, 0.5 * 0.95 * 1.02])
        drifting = self.weights.copy()
        drifting.iloc[2] = grown / grown.sum()

        carried = _no_trade_weights(drifting, self.trade_dates, returns)
        np.testing.assert_allclose(carried.iloc[0], drifting.iloc[2], atol=1e-12)

    def test_market_neutral_book_does_not_blow_up(self):
        # A zero-sum book has no row sum to normalise by; capital must grow with the
        # exposure-weighted return instead.
        weights = _wide([[0.5, -0.5], [0.5, -0.5], [0.3, -0.3]], ["A", "B"])
        returns = _wide([[10.0, 0.0], [0.0, 0.0], [0.0, 0.0]], ["A", "B"])
        carried = _no_trade_weights(weights, weights.index[[0, 2]], returns)
        self.assertTrue(np.isfinite(carried.to_numpy()).all())


class TestTurnoverAgainst(unittest.TestCase):
    def test_known_value(self):
        weights = _wide([[0.5, 0.5], [0.25, 0.75]], ["A", "B"])
        trade_dates = weights.index
        carried = _no_trade_weights(weights, trade_dates)
        turnover = _turnover_against(weights, carried)

        self.assertTrue(np.isnan(turnover.iloc[0]))  # no prior rebalancing
        self.assertAlmostEqual(turnover.iloc[1], 25.0)

    def test_entry_registers_as_full_move_from_zero(self):
        weights = _wide([[1.0, np.nan], [0.5, 0.5]], ["A", "B"])
        carried = _no_trade_weights(weights, weights.index)
        self.assertAlmostEqual(_turnover_against(weights, carried).iloc[1], 50.0)

    def test_non_trading_dates_carry_no_reading(self):
        weights = _wide([[0.5, 0.5]] * 6, ["A", "B"])
        trade_dates = weights.index[[0, 3]]
        turnover = _turnover_against(
            weights, _no_trade_weights(weights, trade_dates)
        )
        self.assertEqual(int(turnover.notna().sum()), 1)
        self.assertFalse(np.isnan(turnover.iloc[3]))

    def test_unchanged_book_reports_zero_rather_than_nan(self):
        # Rebalancing back to the same weights is real information, not missing data.
        weights = _wide([[0.5, 0.5], [0.5, 0.5]], ["A", "B"])
        carried = _no_trade_weights(weights, weights.index)
        self.assertAlmostEqual(_turnover_against(weights, carried).iloc[1], 0.0)

    def test_columns_restrict_to_a_subgroup(self):
        weights = _wide([[0.5, 0.5], [0.25, 0.75]], ["A", "B"])
        carried = _no_trade_weights(weights, weights.index)
        part = _turnover_against(weights, carried, pd.Index(["A"]))
        self.assertAlmostEqual(part.iloc[1], 12.5)


class TestWeightAutocorr(unittest.TestCase):
    @staticmethod
    def _reference(wide: pd.DataFrame, trade_dates: pd.DatetimeIndex) -> np.ndarray:
        vals = wide.fillna(0.0)
        out = np.full(len(wide), np.nan)
        positions = {d: i for i, d in enumerate(wide.index)}
        for prev_d, cur_d in zip(trade_dates[:-1], trade_dates[1:]):
            cur, prev = vals.loc[cur_d].to_numpy(), vals.loc[prev_d].to_numpy()
            mask = (cur != 0) | (prev != 0)
            if mask.sum() > 1 and np.std(cur[mask]) > 0 and np.std(prev[mask]) > 0:
                out[positions[cur_d]] = np.corrcoef(cur[mask], prev[mask])[0, 1]
        return out

    def test_matches_pairwise_reference(self):
        weights, benchmark, _, _ = _random_portfolio(seed=7)
        trade_dates = _trade_dates(weights.index, FREQ)
        for frame in (weights, weights - benchmark):
            np.testing.assert_allclose(
                _weight_autocorr(frame, trade_dates).to_numpy(),
                self._reference(frame, trade_dates),
                equal_nan=True,
            )

    def test_measured_across_rebalancings_not_days(self):
        weights, _, _, _ = _random_portfolio(seed=7)
        trade_dates = _trade_dates(weights.index, FREQ)
        result = _weight_autocorr(weights, trade_dates)
        self.assertEqual(int(result.notna().sum()), len(trade_dates) - 1)
        self.assertTrue(result.drop(index=trade_dates[1:]).isna().all())

    def test_identical_vectors_correlate_perfectly(self):
        weights = _wide([[0.2, 0.3, 0.5], [0.2, 0.3, 0.5]], ["A", "B", "C"])
        self.assertAlmostEqual(
            _weight_autocorr(weights, weights.index).iloc[1], 1.0
        )

    def test_flat_vectors_have_no_reading(self):
        # An equal-weighted book has no cross-sectional dispersion to correlate.
        weights = _wide([[0.5, 0.5], [0.5, 0.5]], ["A", "B"])
        self.assertTrue(_weight_autocorr(weights, weights.index).isna().all())

    def test_rounding_cannot_make_a_flat_vector_look_dispersed(self):
        flat = 1.0 / 3.0
        rows = [[flat, flat, 1.0 - 2 * flat], [flat, 1.0 - 2 * flat, flat]]
        weights = _wide(rows, ["A", "B", "C"])
        self.assertTrue(_weight_autocorr(weights, weights.index).isna().all())

    def test_single_security_is_all_nan(self):
        weights = _wide([[1.0], [1.0]], ["A"])
        self.assertTrue(_weight_autocorr(weights, weights.index).isna().all())

    def test_ambient_zero_universe_is_excluded(self):
        # Two names trade and fully reverse; eight sit at zero throughout. Including
        # the idle eight would drag the correlation towards one instead of -1.
        cids = [f"S{i}" for i in range(10)]
        rows = [[0.6, 0.4] + [0.0] * 8, [0.4, 0.6] + [0.0] * 8]
        weights = _wide(rows, cids)
        self.assertAlmostEqual(_weight_autocorr(weights, weights.index).iloc[1], -1.0)


class TestGroupLabels(unittest.TestCase):
    def test_unmapped_securities_fall_back(self):
        labels = _group_labels(pd.Index(["A", "B"]), {"A": "TECH"}, "OTHER")
        self.assertEqual(labels["A"], "TECH")
        self.assertEqual(labels["B"], "OTHER")

    def test_missing_label_falls_back(self):
        labels = _group_labels(pd.Index(["A"]), {"A": np.nan}, "OTHER")
        self.assertEqual(labels["A"], "OTHER")

    def test_no_mapping_labels_everything_other(self):
        labels = _group_labels(pd.Index(["A", "B"]), None, "OTHER")
        self.assertTrue((labels == "OTHER").all())


class TestWeightStatLabels(unittest.TestCase):
    def test_every_statistic_has_a_label(self):
        for stat in STANDALONE_WEIGHT_STATS + ACTIVE_WEIGHT_STATS:
            self.assertIn(stat, WEIGHT_STAT_LABELS)

    def test_keys_match_the_categories_weight_stats_writes(self):
        weights, benchmark, _, _ = _random_portfolio()
        analyser = PortfolioAnalyser(weights, FREQ, benchmark=benchmark)
        labels = weight_stat_labels("PORT")
        for active, stats in ((False, STANDALONE_WEIGHT_STATS), (True, ACTIVE_WEIGHT_STATS)):
            qdf = analyser.weight_stats(active=active, as_qdf=True, xcat_prefix="PORT")
            self.assertTrue(set(map(str, qdf["xcat"].unique())).issubset(labels))

    def test_prefix_is_applied(self):
        labels = weight_stat_labels("PORTEW")
        self.assertIn("PORTEW_ACTIVE_SHARE", labels)
        self.assertNotIn("PORT_ACTIVE_SHARE", labels)

    def test_benchmark_qualifies_only_the_active_labels(self):
        labels = weight_stat_labels("PORT", benchmark="SP500")
        self.assertEqual(labels["PORT_ACTIVE_SHARE"], "Active share, % (vs SP500)")
        self.assertEqual(labels["PORT_N_HOLDINGS"], "Non-zero holdings")

    def test_off_benchmark_labels_read_with_the_benchmark_suffix(self):
        labels = weight_stat_labels("PORT", benchmark="SP500")
        self.assertEqual(
            labels["PORT_OFF_BENCHMARK_N"], "Off-benchmark holdings (vs SP500)"
        )
        self.assertEqual(
            labels["PORT_OFF_BENCHMARK_WEIGHT"], "Off-benchmark weight, % (vs SP500)"
        )
        self.assertEqual(
            labels["PORT_BENCHMARK_ONLY_N"], "Benchmark-only holdings (vs SP500)"
        )

    def test_stats_restricts_the_output(self):
        labels = weight_stat_labels("PORTEW", stats=ACTIVE_WEIGHT_STATS)
        self.assertEqual(len(labels), len(ACTIVE_WEIGHT_STATS))
        self.assertNotIn("PORTEW_N_HOLDINGS", labels)

    def test_two_benchmarks_merge_without_clashing(self):
        labels = {
            **weight_stat_labels("PORT", benchmark="SP500"),
            **weight_stat_labels(
                "PORTEW", benchmark="equal wgt", stats=ACTIVE_WEIGHT_STATS
            ),
        }
        self.assertEqual(labels["PORT_ACTIVE_SHARE"], "Active share, % (vs SP500)")
        self.assertEqual(
            labels["PORTEW_ACTIVE_SHARE"], "Active share, % (vs equal wgt)"
        )

    def test_unknown_statistic_raises(self):
        with self.assertRaisesRegex(KeyError, "No label defined"):
            weight_stat_labels("PORT", stats=["not_a_stat"])

    def test_empty_prefix_raises(self):
        with self.assertRaises(TypeError):
            _stat_xcat("", "n_holdings")


class TestCidLabelMap(unittest.TestCase):
    def test_underscores_become_hyphens(self):
        self.assertEqual(_cid_label_map(["INFO_TECH"]), {"INFO_TECH": "INFO-TECH"})

    def test_collision_raises(self):
        with self.assertRaisesRegex(ValueError, "collide"):
            _cid_label_map(["INFO_TECH", "INFO-TECH"])


class TestPortfolioAnalyserConstruction(unittest.TestCase):
    def setUp(self):
        self.weights, self.benchmark, self.returns, self.groups = _random_portfolio()

    def test_rebalance_freq_is_required_and_validated(self):
        with self.assertRaises(TypeError):
            PortfolioAnalyser(self.weights)
        with self.assertRaisesRegex(ValueError, "rebalance_freq"):
            PortfolioAnalyser(self.weights, "daily")

    def test_trade_dates_follow_the_rebalance_frequency(self):
        analyser = PortfolioAnalyser(self.weights, "M")
        pd.testing.assert_index_equal(
            analyser.trade_dates, _trade_dates(self.weights.index, "M")
        )

    def test_trade_dates_come_from_the_portfolio_not_the_benchmark(self):
        # The benchmark drifts and reconstitutes on its own schedule; only the
        # portfolio's rebalancings say when the book was traded.
        benchmark = self.benchmark.iloc[::2]
        analyser = PortfolioAnalyser(self.weights, "M", benchmark=benchmark)
        pd.testing.assert_index_equal(
            analyser.trade_dates, _trade_dates(self.weights.index, "M")
        )

    def test_too_few_periods_is_flagged(self):
        with self.assertLogs("macrosynergy.securities.analyse", level="WARNING") as log:
            PortfolioAnalyser(self.weights, "Y")
        self.assertIn("fewer than two", "".join(log.output))

    def test_groups_must_be_mapping(self):
        with self.assertRaises(TypeError):
            PortfolioAnalyser(self.weights, FREQ, groups=["TECH"])

    def test_groups_accepts_series(self):
        analyser = PortfolioAnalyser(self.weights, FREQ, groups=pd.Series(self.groups))
        self.assertEqual(set(analyser.groups.unique()), {"TECH", "FINS"})

    def test_universe_spans_portfolio_and_benchmark(self):
        benchmark = self.benchmark.copy()
        benchmark["EXTRA"] = 0.0
        analyser = PortfolioAnalyser(self.weights, FREQ, benchmark=benchmark)
        self.assertIn("EXTRA", analyser.cids)
        # The raw weights are untouched, so standalone stats stay on their own universe.
        self.assertNotIn("EXTRA", analyser.weights.columns)

    def test_start_and_end_trim_every_frame(self):
        analyser = PortfolioAnalyser(
            self.weights,
            FREQ,
            benchmark=self.benchmark,
            returns=self.returns,
            start="2020-02-01",
            end="2020-02-28",
        )
        for frame in (analyser.weights, analyser.benchmark, analyser.returns):
            self.assertTrue((frame.index >= pd.Timestamp("2020-02-01")).all())
            self.assertTrue((frame.index <= pd.Timestamp("2020-02-28")).all())

    def test_empty_window_raises(self):
        with self.assertRaisesRegex(ValueError, "No dates remain"):
            PortfolioAnalyser(self.weights, FREQ, start="2030-01-01")

    def test_percentage_weights_are_flagged(self):
        with self.assertLogs("macrosynergy.securities.analyse", level="WARNING") as log:
            PortfolioAnalyser(self.weights * 100.0, FREQ)
        self.assertIn("percentage points", "".join(log.output))


class TestAdjustWeightsWithDrift(unittest.TestCase):
    def test_static_targets_drift_between_rebalancings(self):
        targets = _wide([[0.5, 0.5]] * 4, ["A", "B"], start="2020-01-01")
        returns = _wide([[0.0, 0.0], [100.0, 0.0], [0.0, 0.0], [0.0, 0.0]], ["A", "B"])
        drifted = PortfolioAnalyser.adjust_weights_with_drift(targets, returns, "Y")

        # A doubles on day two, so from day three the book is 2:1 rather than 1:1.
        np.testing.assert_allclose(drifted.iloc[0], [0.5, 0.5])
        np.testing.assert_allclose(drifted.iloc[2], [2 / 3, 1 / 3])

    def test_rebalancing_resets_to_target(self):
        targets = _wide([[0.5, 0.5]] * 4, ["A", "B"])
        returns = _wide([[100.0, 0.0]] * 4, ["A", "B"])
        drifted = PortfolioAnalyser.adjust_weights_with_drift(targets, returns, "B")
        # Reset every day, so the book never leaves its target.
        np.testing.assert_allclose(drifted.to_numpy(), 0.5)

    def test_rows_sum_to_one(self):
        weights, _, returns, _ = _random_portfolio()
        drifted = PortfolioAnalyser.adjust_weights_with_drift(weights, returns, FREQ)
        np.testing.assert_allclose(drifted.sum(axis=1), 1.0)

    def test_frequency_is_validated(self):
        weights, _, returns, _ = _random_portfolio()
        with self.assertRaisesRegex(ValueError, "rebalance_freq"):
            PortfolioAnalyser.adjust_weights_with_drift(weights, returns, "daily")

    def test_accepts_long_format(self):
        weights, _, returns, _ = _random_portfolio()
        pd.testing.assert_frame_equal(
            PortfolioAnalyser.adjust_weights_with_drift(weights, returns, FREQ),
            PortfolioAnalyser.adjust_weights_with_drift(
                _to_long(weights), _to_long(returns, xcat="EQXR"), FREQ
            ),
        )


class TestWeightStats(unittest.TestCase):
    def setUp(self):
        self.weights, self.benchmark, self.returns, self.groups = _random_portfolio()
        self.analyser = PortfolioAnalyser(
            self.weights,
            FREQ,
            benchmark=self.benchmark,
            returns=self.returns,
            groups=self.groups,
        )

    def test_standalone_columns(self):
        stats = self.analyser.weight_stats()
        self.assertEqual(list(stats.columns), ["real_date"] + STANDALONE_WEIGHT_STATS)
        self.assertEqual(len(stats), len(self.weights))

    def test_active_columns(self):
        stats = self.analyser.weight_stats(active=True)
        self.assertEqual(list(stats.columns), ["real_date"] + ACTIVE_WEIGHT_STATS)

    def test_group_columns(self):
        stats = self.analyser.weight_stats(by_group=True)
        self.assertEqual(
            list(stats.columns), ["real_date", "group"] + STANDALONE_WEIGHT_STATS
        )
        self.assertEqual(set(stats["group"].unique()), {"TECH", "FINS"})

    def test_concentration_is_daily_but_turnover_is_not(self):
        stats = self.analyser.weight_stats().set_index("real_date")
        n_trades = len(self.analyser.trade_dates) - 1

        for col in ["n_holdings", "effective_n", "weight", "gross_weight"]:
            self.assertEqual(int(stats[col].notna().sum()), len(stats), col)
        for col in ["turnover", "weight_autocorr"]:
            self.assertLessEqual(int(stats[col].notna().sum()), n_trades, col)
        self.assertEqual(int(stats["turnover"].notna().sum()), n_trades)
        # Readings land only on rebalancing dates.
        self.assertTrue(
            stats["turnover"].dropna().index.isin(self.analyser.trade_dates).all()
        )

    def test_active_turnover_columns_are_also_trade_date_only(self):
        stats = self.analyser.weight_stats(active=True).set_index("real_date")
        n_trades = len(self.analyser.trade_dates) - 1
        for col in ["n_active_holdings", "active_weight", "active_share"]:
            self.assertEqual(int(stats[col].notna().sum()), len(stats), col)
        for col in ["active_turnover", "active_weight_turnover"]:
            self.assertEqual(int(stats[col].notna().sum()), n_trades, col)

    def test_drift_is_removed_from_turnover(self):
        # Weights that only drift have traded nothing, whatever the rebalance dates.
        base = self.weights.iloc[[0]].to_numpy()
        growth = (1.0 + self.returns / 100.0).cumprod().shift(1)
        growth.iloc[0] = 1.0
        grown = base * growth.to_numpy()
        drifting = pd.DataFrame(
            grown / grown.sum(axis=1, keepdims=True),
            index=self.weights.index,
            columns=self.weights.columns,
        )
        analyser = PortfolioAnalyser(drifting, FREQ, returns=self.returns)
        turnover = analyser.weight_stats()["turnover"]
        np.testing.assert_allclose(turnover.dropna(), 0.0, atol=1e-10)

    def test_turnover_without_returns_leaves_drift_in(self):
        with_returns = PortfolioAnalyser(
            self.weights, FREQ, returns=self.returns
        ).weight_stats()["turnover"]
        without = PortfolioAnalyser(self.weights, FREQ).weight_stats()["turnover"]
        self.assertEqual(with_returns.notna().sum(), without.notna().sum())
        self.assertFalse(np.allclose(with_returns.dropna(), without.dropna()))

    def test_fully_invested_portfolio_weighs_one_hundred(self):
        stats = self.analyser.weight_stats()
        np.testing.assert_allclose(stats["weight"], 100.0)

    def test_active_share_of_known_weights(self):
        weights = _wide([[0.6, 0.4], [0.6, 0.4]], ["A", "B"])
        benchmark = _wide([[0.5, 0.5], [0.5, 0.5]], ["A", "B"])
        stats = PortfolioAnalyser(weights, "B", benchmark=benchmark).weight_stats(
            active=True
        )
        np.testing.assert_allclose(stats["active_share"], 10.0)
        np.testing.assert_allclose(stats["active_weight"], 0.0, atol=1e-12)

    def test_group_statistics_sum_to_the_whole_portfolio(self):
        total = self.analyser.weight_stats().set_index("real_date")
        grouped = self.analyser.weight_stats(by_group=True).pivot(
            index="real_date", columns="group"
        )
        for col in ["n_holdings", "weight", "gross_weight", "turnover"]:
            np.testing.assert_allclose(
                grouped[col].sum(axis=1, min_count=1),
                total[col],
                rtol=1e-10,
                atol=1e-9,
            )

    def test_active_group_statistics_sum_to_the_whole_portfolio(self):
        total = self.analyser.weight_stats(active=True).set_index("real_date")
        grouped = self.analyser.weight_stats(by_group=True, active=True).pivot(
            index="real_date", columns="group"
        )
        for col in [
            "n_active_holdings",
            "active_weight",
            "active_share",
            "active_turnover",
            "active_weight_turnover",
        ]:
            np.testing.assert_allclose(
                grouped[col].sum(axis=1, min_count=1),
                total[col],
                rtol=1e-10,
                atol=1e-9,
            )

    def test_effective_n_never_exceeds_holdings(self):
        stats = self.analyser.weight_stats(by_group=True)
        self.assertTrue((stats["effective_n"] <= stats["n_holdings"] + 1e-9).all())

    def test_active_requires_a_benchmark(self):
        analyser = PortfolioAnalyser(self.weights, FREQ, groups=self.groups)
        with self.assertRaisesRegex(ValueError, "`benchmark` must be supplied"):
            analyser.weight_stats(active=True)

    def test_by_group_requires_a_mapping(self):
        analyser = PortfolioAnalyser(self.weights, FREQ, benchmark=self.benchmark)
        with self.assertRaisesRegex(ValueError, "`groups` must be supplied"):
            analyser.weight_stats(by_group=True)

    def test_as_qdf_uses_group_labels_as_cross_sections(self):
        qdf = self.analyser.weight_stats(by_group=True, as_qdf=True)
        self.assertIsInstance(qdf, QuantamentalDataFrame)
        self.assertEqual(set(map(str, qdf["cid"].unique())), {"TECH", "FINS"})
        self.assertEqual(
            set(map(str, qdf["xcat"].unique())),
            {f"PORT_{stat.upper()}" for stat in STANDALONE_WEIGHT_STATS},
        )
        self.assertFalse(qdf["value"].isna().any())

    def test_as_qdf_uses_portfolio_name_when_ungrouped(self):
        qdf = self.analyser.weight_stats(as_qdf=True, xcat_prefix="PW")
        self.assertEqual(list(map(str, qdf["cid"].unique())), ["PORTFOLIO"])
        self.assertTrue(all(str(x).startswith("PW_") for x in qdf["xcat"].unique()))

    def test_as_qdf_sanitises_underscored_labels(self):
        analyser = PortfolioAnalyser(
            self.weights,
            FREQ,
            groups={cid: "INFO_TECH" for cid in self.weights.columns},
        )
        qdf = analyser.weight_stats(by_group=True, as_qdf=True)
        self.assertEqual(list(map(str, qdf["cid"].unique())), ["INFO-TECH"])


class TestOffBenchmarkStats(unittest.TestCase):
    """
    The portfolio and the benchmark hold different universes on both sides: A and E
    are held but never in the benchmark, D is in the benchmark but never held, and C
    sits in both universes at a zero portfolio weight on most dates. The third date
    holds nothing at all and the fourth has no benchmark.
    """

    OFF_BENCHMARK_STATS = [
        "off_benchmark_n",
        "off_benchmark_weight",
        "benchmark_only_n",
    ]

    def setUp(self):
        self.weights = _wide(
            [
                [0.5, 0.3, 0.0, 0.2],
                [0.5, 0.5, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.2, 0.3, 0.5, 0.0],
            ],
            ["A", "B", "C", "E"],
        )
        self.benchmark = _wide(
            [
                [0.4, 0.3, 0.3],
                [0.4, 0.3, 0.3],
                [0.4, 0.3, 0.3],
                [0.0, 0.0, 0.0],
            ],
            ["B", "C", "D"],
        )
        self.groups = {"A": "X", "B": "X", "C": "Y", "D": "Y", "E": "Y"}
        self.analyser = PortfolioAnalyser(
            self.weights, "B", benchmark=self.benchmark, groups=self.groups
        )

    def _active(self):
        return self.analyser.weight_stats(active=True).set_index("real_date")

    def test_columns_are_reported_on_the_active_side_only(self):
        for stat in self.OFF_BENCHMARK_STATS:
            self.assertIn(stat, ACTIVE_WEIGHT_STATS)
            self.assertNotIn(stat, STANDALONE_WEIGHT_STATS)
        self.assertEqual(
            list(self.analyser.weight_stats().columns),
            ["real_date"] + STANDALONE_WEIGHT_STATS,
        )
        self.assertEqual(
            list(self.analyser.weight_stats(active=True).columns),
            ["real_date"] + ACTIVE_WEIGHT_STATS,
        )

    def test_universes_differing_on_both_sides(self):
        stats = self._active()
        # A and E are held off-benchmark on the first date, A alone on the second.
        np.testing.assert_allclose(stats["off_benchmark_n"].iloc[:2], [2.0, 1.0])
        np.testing.assert_allclose(stats["off_benchmark_weight"].iloc[:2], [70.0, 50.0])
        # C and D sit in the benchmark unheld on both.
        np.testing.assert_allclose(stats["benchmark_only_n"].iloc[:2], [2.0, 2.0])

    def test_zero_weight_is_held_by_neither_side(self):
        # C carries a zero portfolio weight on the first two dates: it is neither an
        # off-benchmark holding nor absent from the count of benchmark-only names.
        stats = self._active()
        self.assertEqual(stats["benchmark_only_n"].iloc[0], 2.0)
        # On the last date C is genuinely held, but the benchmark is empty there, so
        # the split is not reported at all.
        self.assertTrue(np.isnan(stats["off_benchmark_n"].iloc[3]))

    def test_off_benchmark_and_shared_holdings_rebuild_n_holdings(self):
        standalone = self.analyser.weight_stats().set_index("real_date")
        active = self._active()
        held = self.analyser.weights.ne(0.0) & self.analyser.weights.notna()
        aligned_bm = self.analyser.benchmark.reindex(
            index=held.index, columns=held.columns
        ).fillna(0.0)
        shared = (held & aligned_bm.ne(0.0)).sum(axis=1)
        # The last date has no benchmark, so the split carries no reading there.
        measured = active["off_benchmark_n"].notna()
        np.testing.assert_allclose(
            active["off_benchmark_n"][measured] + shared[measured],
            standalone["n_holdings"][measured],
        )

    def test_group_statistics_sum_to_the_whole_portfolio(self):
        total = self._active()
        grouped = self.analyser.weight_stats(by_group=True, active=True).pivot(
            index="real_date", columns="group"
        )
        self.assertEqual(set(grouped["off_benchmark_n"].columns), {"X", "Y"})
        for col in self.OFF_BENCHMARK_STATS:
            np.testing.assert_allclose(
                grouped[col].sum(axis=1, min_count=1),
                total[col],
                rtol=1e-10,
                atol=1e-9,
            )

    def test_group_split_is_measured_within_the_subgroup(self):
        grouped = self.analyser.weight_stats(by_group=True, active=True).pivot(
            index="real_date", columns="group"
        )
        # X holds A off-benchmark and B in it; Y holds E off-benchmark against C and D.
        np.testing.assert_allclose(grouped["off_benchmark_n"]["X"].iloc[0], 1.0)
        np.testing.assert_allclose(grouped["off_benchmark_weight"]["X"].iloc[0], 50.0)
        np.testing.assert_allclose(grouped["benchmark_only_n"]["X"].iloc[0], 0.0)
        np.testing.assert_allclose(grouped["off_benchmark_n"]["Y"].iloc[0], 1.0)
        np.testing.assert_allclose(grouped["off_benchmark_weight"]["Y"].iloc[0], 20.0)
        np.testing.assert_allclose(grouped["benchmark_only_n"]["Y"].iloc[0], 2.0)

    def test_empty_date_keeps_the_counts_but_drops_the_weight(self):
        stats = self._active()
        self.assertEqual(stats["off_benchmark_n"].iloc[2], 0.0)
        self.assertTrue(np.isnan(stats["off_benchmark_weight"].iloc[2]))
        # The benchmark is still there on that date, so its unheld members do count.
        self.assertEqual(stats["benchmark_only_n"].iloc[2], 3.0)

    def test_absent_benchmark_leaves_no_reading(self):
        stats = self._active()
        self.assertTrue(stats[self.OFF_BENCHMARK_STATS].iloc[3].isna().all())

    def test_zero_weight_is_distinct_from_no_reading(self):
        # Every holding sits inside the benchmark: a real zero, not a missing value.
        weights = _wide([[0.6, 0.4], [0.6, 0.4]], ["A", "B"])
        benchmark = _wide([[0.5, 0.5], [0.5, 0.5]], ["A", "B"])
        stats = PortfolioAnalyser(weights, "B", benchmark=benchmark).weight_stats(
            active=True
        )
        np.testing.assert_allclose(stats["off_benchmark_n"], 0.0)
        np.testing.assert_allclose(stats["off_benchmark_weight"], 0.0)
        np.testing.assert_allclose(stats["benchmark_only_n"], 0.0)

    def test_split_is_a_daily_reading(self):
        weights, benchmark, _, _ = _random_portfolio()
        analyser = PortfolioAnalyser(weights, FREQ, benchmark=benchmark)
        stats = analyser.weight_stats(active=True)
        n_trades = len(analyser.trade_dates) - 1
        for col in self.OFF_BENCHMARK_STATS:
            self.assertEqual(int(stats[col].notna().sum()), len(stats), col)
        self.assertGreater(len(stats), n_trades)

    def test_as_qdf_carries_the_split_categories(self):
        qdf = self.analyser.weight_stats(active=True, as_qdf=True)
        xcats = set(map(str, qdf["xcat"].unique()))
        for stat in self.OFF_BENCHMARK_STATS:
            self.assertIn(f"PORT_{stat.upper()}", xcats)


class TestAttribution(unittest.TestCase):
    def setUp(self):
        self.weights, self.benchmark, self.returns, self.groups = _random_portfolio()
        self.analyser = PortfolioAnalyser(
            self.weights,
            FREQ,
            benchmark=self.benchmark,
            returns=self.returns,
            groups=self.groups,
        )

    def test_weights_are_lagged_against_returns(self):
        weights = _wide([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]], ["A", "B"])
        returns = _wide([[10.0, -10.0], [10.0, -10.0], [10.0, -10.0]], ["A", "B"])
        contrib = PortfolioAnalyser(weights, "B", returns=returns).attribution()

        # Day 0 has no prior weight; day 1 earns on the day-0 weight (all in A).
        self.assertAlmostEqual(contrib["PORTFOLIO"].iloc[0], 0.0)
        self.assertAlmostEqual(contrib["PORTFOLIO"].iloc[1], 10.0)
        self.assertAlmostEqual(contrib["PORTFOLIO"].iloc[2], -10.0)

    def test_zero_lag_applies_same_day_weights(self):
        weights = _wide([[1.0, 0.0], [0.0, 1.0]], ["A", "B"])
        returns = _wide([[10.0, -10.0], [10.0, -10.0]], ["A", "B"])
        contrib = PortfolioAnalyser(weights, "B", returns=returns).attribution(lag=0)
        self.assertAlmostEqual(contrib["PORTFOLIO"].iloc[0], 10.0)
        self.assertAlmostEqual(contrib["PORTFOLIO"].iloc[1], -10.0)

    def test_attribution_stays_daily(self):
        contrib = self.analyser.attribution()
        self.assertEqual(len(contrib), len(self.weights))

    def test_total_is_the_sum_of_contributions(self):
        contrib = self.analyser.attribution()
        securities = [c for c in contrib.columns if c != "PORTFOLIO"]
        np.testing.assert_allclose(
            contrib[securities].sum(axis=1), contrib["PORTFOLIO"], rtol=1e-10
        )

    def test_group_attribution_reconciles_to_the_portfolio(self):
        total = self.analyser.attribution()["PORTFOLIO"]
        grouped = self.analyser.attribution(by_group=True)
        self.assertEqual(
            [c for c in grouped.columns if c != "PORTFOLIO"], ["FINS", "TECH"]
        )
        np.testing.assert_allclose(grouped["PORTFOLIO"], total, rtol=1e-10)

    def test_active_attribution_is_portfolio_minus_benchmark(self):
        active = self.analyser.attribution(active=True)["PORTFOLIO"]
        portfolio = self.analyser.attribution()["PORTFOLIO"]
        benchmark = PortfolioAnalyser(
            self.benchmark, FREQ, returns=self.returns
        ).attribution()["PORTFOLIO"]
        np.testing.assert_allclose(active, portfolio - benchmark, rtol=1e-10)

    def test_include_total_can_be_switched_off(self):
        contrib = self.analyser.attribution(by_group=True, include_total=False)
        self.assertEqual(list(contrib.columns), ["FINS", "TECH"])

    def test_total_label_collision_raises(self):
        analyser = PortfolioAnalyser(
            self.weights,
            FREQ,
            returns=self.returns,
            groups=self.groups,
            portfolio_name="TECH",
        )
        with self.assertRaisesRegex(ValueError, "collides"):
            analyser.attribution(by_group=True)

    def test_returns_are_required(self):
        analyser = PortfolioAnalyser(self.weights, FREQ, groups=self.groups)
        with self.assertRaisesRegex(ValueError, "`returns` must be supplied"):
            analyser.attribution()

    def test_lag_must_be_a_non_negative_integer(self):
        for bad in (-1, 1.5, "1", True):
            with self.assertRaises(TypeError):
                self.analyser.attribution(lag=bad)

    def test_securities_without_returns_contribute_zero(self):
        weights = _wide([[0.5, 0.5], [0.5, 0.5]], ["A", "B"])
        returns = _wide([[10.0], [10.0]], ["A"])
        contrib = PortfolioAnalyser(weights, "B", returns=returns).attribution()
        self.assertAlmostEqual(contrib["B"].iloc[1], 0.0)
        self.assertAlmostEqual(contrib["PORTFOLIO"].iloc[1], 5.0)

    def test_as_qdf_carries_one_category(self):
        qdf = self.analyser.attribution(as_qdf=True)
        self.assertIsInstance(qdf, QuantamentalDataFrame)
        self.assertEqual(list(map(str, qdf["xcat"].unique())), ["CONTRIB"])

    def test_as_qdf_defaults_to_a_distinct_active_category(self):
        qdf = self.analyser.attribution(active=True, as_qdf=True)
        self.assertEqual(list(map(str, qdf["xcat"].unique())), ["ACTIVE_CONTRIB"])


class TestLongFormatParity(unittest.TestCase):
    def test_long_and_wide_inputs_agree(self):
        weights, benchmark, returns, groups = _random_portfolio(seed=3)
        wide = PortfolioAnalyser(
            weights, FREQ, benchmark=benchmark, returns=returns, groups=groups
        )
        long = PortfolioAnalyser(
            _to_long(weights),
            FREQ,
            benchmark=_to_long(benchmark, xcat="BMW"),
            returns=_to_long(returns, xcat="EQXR"),
            groups=groups,
        )
        pd.testing.assert_frame_equal(
            wide.weight_stats(by_group=True, active=True),
            long.weight_stats(by_group=True, active=True),
        )
        pd.testing.assert_frame_equal(
            wide.attribution(), long.attribution(), check_freq=False
        )


if __name__ == "__main__":
    unittest.main()
