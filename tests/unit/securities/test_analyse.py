import unittest
import warnings
from typing import Dict, List

import numpy as np
import pandas as pd

from macrosynergy.management.types import QuantamentalDataFrame
from macrosynergy.securities.analyse import (
    ACTIVE_WEIGHT_STATS,
    BRINSON_STATS,
    STANDALONE_WEIGHT_STATS,
    WEIGHT_STAT_LABELS,
    PortfolioAnalyser,
    _align_active,
    _as_wide,
    _assert_unbroken_schedule,
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


def _all_investable(frame: pd.DataFrame) -> pd.DataFrame:
    """An unchanging universe, i.e. the identity argument to ``universe``."""
    return pd.DataFrame(True, index=frame.index, columns=frame.columns)


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
        drifted = PortfolioAnalyser.adjust_weights_with_drift(
            targets, returns, _all_investable(targets), "Y"
        )

        # A doubles on day two, so from day three the book is 2:1 rather than 1:1.
        np.testing.assert_allclose(drifted.iloc[0], [0.5, 0.5])
        np.testing.assert_allclose(drifted.iloc[2], [2 / 3, 1 / 3])

    def test_rebalancing_resets_to_target(self):
        targets = _wide([[0.5, 0.5]] * 4, ["A", "B"])
        returns = _wide([[100.0, 0.0]] * 4, ["A", "B"])
        drifted = PortfolioAnalyser.adjust_weights_with_drift(
            targets, returns, _all_investable(targets), "B"
        )
        # Reset every day, so the book never leaves its target.
        np.testing.assert_allclose(drifted.to_numpy(), 0.5)

    def test_rows_sum_to_one(self):
        weights, _, returns, _ = _random_portfolio()
        drifted = PortfolioAnalyser.adjust_weights_with_drift(
            weights, returns, _all_investable(weights), FREQ
        )
        np.testing.assert_allclose(drifted.sum(axis=1), 1.0)

    def test_frequency_is_validated(self):
        weights, _, returns, _ = _random_portfolio()
        with self.assertRaisesRegex(ValueError, "rebalance_freq"):
            PortfolioAnalyser.adjust_weights_with_drift(
                weights, returns, _all_investable(weights), "daily"
            )

    def test_universe_is_required(self):
        weights, _, returns, _ = _random_portfolio()
        with self.assertRaises(TypeError):
            PortfolioAnalyser.adjust_weights_with_drift(weights, returns, FREQ)

    def test_accepts_long_format(self):
        weights, _, returns, _ = _random_portfolio()
        universe = _all_investable(weights)
        pd.testing.assert_frame_equal(
            PortfolioAnalyser.adjust_weights_with_drift(
                weights, returns, universe, FREQ
            ),
            PortfolioAnalyser.adjust_weights_with_drift(
                _to_long(weights),
                _to_long(returns, xcat="EQXR"),
                _to_long(universe.astype(float)),
                FREQ,
            ),
        )

    def test_returns_before_the_first_target_are_ignored(self):
        # The first period would otherwise open before any target exists, be normalised
        # by a zero target sum, and be deleted. The workaround this replaces was to
        # slice the returns to the weights' own span by hand.
        # The returns reach back to January; the targets only start in February, and
        # expire at the end of March, so every period on the calendar carries one.
        returns = _wide([[1.0, -0.5]] * 60, ["A", "B"], start="2020-01-01")
        targets = _wide([[0.6, 0.4], [0.3, 0.7]], ["A", "B"], start="2020-02-12")

        long_history = PortfolioAnalyser.adjust_weights_with_drift(
            targets, returns, _all_investable(returns), "M"
        )
        sliced_returns = returns.loc[targets.index.min() :]
        sliced = PortfolioAnalyser.adjust_weights_with_drift(
            targets, sliced_returns, _all_investable(sliced_returns), "M"
        )

        self.assertEqual(long_history.index.min(), targets.index.min())
        np.testing.assert_allclose(long_history.sum(axis=1), 1.0)
        pd.testing.assert_frame_equal(long_history, sliced, check_freq=False)


class TestAdjustWeightsWithDriftCarry(unittest.TestCase):
    """
    A target is in force for the rest of its own rebalancing period and for the one
    that follows, then expires - so a stalled signal feed cannot read as a live book.
    """

    def setUp(self):
        self.dates = pd.bdate_range("2020-01-01", periods=108)  # January to May
        self.cids = ["A", "B"]
        self.returns = pd.DataFrame(
            0.0, index=self.dates, columns=self.cids
        )  # zero returns, so the book is the carried target itself

    def _drift(self, targets, freq="M"):
        return PortfolioAnalyser.adjust_weights_with_drift(
            targets, self.returns, _all_investable(self.returns), freq
        )

    def test_month_end_targets_cover_the_following_month(self):
        # The canonical sparse panel: one row per rebalancing, recorded at month end
        # and traded through the month that follows.
        targets = pd.DataFrame(
            [[0.5, 0.5], [0.2, 0.8], [0.7, 0.3], [0.4, 0.6], [0.1, 0.9]],
            index=pd.to_datetime(
                ["2020-01-31", "2020-02-28", "2020-03-31", "2020-04-30", "2020-05-29"]
            ),
            columns=self.cids,
        )
        drifted = self._drift(targets)

        np.testing.assert_allclose(drifted.sum(axis=1), 1.0)
        np.testing.assert_allclose(drifted.loc["2020-02-14"], [0.5, 0.5])
        np.testing.assert_allclose(drifted.loc["2020-03-16"], [0.2, 0.8])

    def test_period_start_targets_cover_their_own_period(self):
        targets = pd.DataFrame(
            [[0.5, 0.5], [0.2, 0.8], [0.7, 0.3], [0.4, 0.6], [0.1, 0.9]],
            index=pd.to_datetime(
                ["2020-01-01", "2020-02-03", "2020-03-02", "2020-04-01", "2020-05-01"]
            ),
            columns=self.cids,
        )
        drifted = self._drift(targets)

        np.testing.assert_allclose(drifted.sum(axis=1), 1.0)
        np.testing.assert_allclose(drifted.loc["2020-01-20"], [0.5, 0.5])
        np.testing.assert_allclose(drifted.loc["2020-02-20"], [0.2, 0.8])

    def test_targets_dated_off_the_business_calendar_still_land(self):
        # `resample("ME")` states a target on a calendar month end, which falls on a
        # weekend roughly a third of the time. Dropping those rows rather than reading
        # them would expire the book a period early.
        dates = pd.bdate_range("2020-02-01", "2020-04-30")
        returns = pd.DataFrame(0.0, index=dates, columns=self.cids)
        targets = pd.DataFrame(
            [[0.5, 0.5], [0.2, 0.8]],
            index=pd.to_datetime(["2020-02-29", "2020-03-31"]),  # Saturday, Tuesday
            columns=self.cids,
        )
        drifted = PortfolioAnalyser.adjust_weights_with_drift(
            targets, returns, _all_investable(returns), "M"
        )

        self.assertEqual(drifted.index.min(), pd.Timestamp("2020-03-02"))
        np.testing.assert_allclose(drifted.loc["2020-03-16"], [0.5, 0.5])
        np.testing.assert_allclose(drifted.loc["2020-04-15"], [0.2, 0.8])

    def test_targets_do_not_outlive_the_returns(self):
        # A single target against five months of returns: it expires at the end of
        # February rather than running to the end of the sample.
        targets = _wide([[0.5, 0.5]], self.cids, start="2020-01-31")
        with self.assertRaisesRegex(ValueError, "no target in force"):
            self._drift(targets)

    def test_a_skipped_period_raises(self):
        targets = pd.DataFrame(
            [[0.5, 0.5], [0.2, 0.8]],
            index=pd.to_datetime(["2020-01-31", "2020-04-30"]),  # February, March gone
            columns=self.cids,
        )
        with self.assertRaisesRegex(ValueError, r"2020-03") as ctx:
            self._drift(targets)
        # February is covered by January's carry; only March is left with nothing.
        self.assertNotIn("2020-02", str(ctx.exception))

    def test_a_coarser_cadence_absorbs_the_gap(self):
        targets = pd.DataFrame(
            [[0.5, 0.5], [0.2, 0.8]],
            index=pd.to_datetime(["2020-01-31", "2020-04-30"]),
            columns=self.cids,
        )
        drifted = self._drift(targets, freq="Q")
        np.testing.assert_allclose(drifted.sum(axis=1), 1.0)

    def test_expiry_is_per_security(self):
        # B is re-stated throughout, A only in January, so A expires at the end of
        # February and B carries the book alone from March.
        targets = pd.DataFrame(
            [[0.5, 0.5], [np.nan, 0.8], [np.nan, 0.3], [np.nan, 0.6]],
            index=pd.to_datetime(
                ["2020-01-31", "2020-02-28", "2020-03-31", "2020-04-30"]
            ),
            columns=self.cids,
        )
        drifted = self._drift(targets)

        np.testing.assert_allclose(drifted.loc["2020-02-14"], [0.5, 0.5])
        np.testing.assert_allclose(drifted.loc["2020-03-16"], [0.0, 1.0])


class TestAdjustWeightsWithDriftUniverse(unittest.TestCase):
    """
    ``universe`` separates "not re-recorded today" from "no longer investable", the two
    readings of a missing target that the forward-fill cannot tell apart on its own.
    """

    def setUp(self):
        self.cids = ["A", "B", "C"]
        self.dates = pd.bdate_range("2020-01-01", periods=44)  # January and February
        self.returns = pd.DataFrame(
            [[1.0, -0.5, 2.0]] * len(self.dates), index=self.dates, columns=self.cids
        )
        # One target row per monthly rebalancing, i.e. the sparse panel the
        # forward-fill exists for.
        self.targets = pd.DataFrame(
            [[0.5, 0.3, 0.2], [0.2, 0.3, 0.5]],
            index=[pd.Timestamp("2020-01-01"), pd.Timestamp("2020-02-03")],
            columns=self.cids,
        )

    def _drift(self, universe):
        return PortfolioAnalyser.adjust_weights_with_drift(
            self.targets, self.returns, universe, "M"
        )

    def _exits(self, cid: str, date: str) -> pd.DataFrame:
        frame = _all_investable(self.returns)
        frame.loc[pd.Timestamp(date) :, cid] = False
        return frame

    def test_sparse_targets_survive_an_unchanging_universe(self):
        # The regression guard: an all-true universe leaves the sparse panel exactly as
        # the caller's own forward-fill would, and masks nothing.
        drifted = self._drift(_all_investable(self.returns))
        dense = self.targets.reindex(self.dates).ffill()

        pd.testing.assert_frame_equal(
            drifted,
            PortfolioAnalyser.adjust_weights_with_drift(
                dense, self.returns, _all_investable(dense), "M"
            ),
        )
        self.assertFalse(drifted.isna().any().any())
        np.testing.assert_allclose(drifted.sum(axis=1), 1.0)
        np.testing.assert_allclose(drifted.iloc[0], [0.5, 0.3, 0.2])

    def test_exit_on_a_rebalancing_boundary(self):
        exit_date = pd.Timestamp("2020-02-03")
        drifted = self._drift(self._exits("C", exit_date))

        self.assertTrue(drifted.loc[exit_date:, "C"].isna().all())
        self.assertFalse(drifted.loc[:exit_date, "C"].iloc[:-1].isna().any())
        np.testing.assert_allclose(drifted.sum(axis=1), 1.0)

        # From the rebalancing the exit falls on, the book is the one the surviving
        # universe would have produced on its own.
        survivors = PortfolioAnalyser.adjust_weights_with_drift(
            self.targets[["A", "B"]],
            self.returns[["A", "B"]],
            _all_investable(self.returns[["A", "B"]]),
            "M",
        )
        pd.testing.assert_frame_equal(
            drifted.loc[exit_date:, ["A", "B"]],
            survivors.loc[exit_date:],
            check_freq=False,
        )

    def test_exit_mid_period_rescales_the_survivors(self):
        exit_date = pd.Timestamp("2020-02-12")  # not a rebalancing date
        drifted = self._drift(self._exits("C", exit_date))
        unmasked = self._drift(_all_investable(self.returns))

        self.assertTrue(drifted.loc[exit_date:, "C"].isna().all())
        self.assertFalse(drifted.loc[:exit_date, "C"].iloc[:-1].isna().any())
        np.testing.assert_allclose(drifted.sum(axis=1), 1.0)
        # The survivors are rescaled, not re-set, on a date that is no rebalancing:
        # their relative sizes are the drifting book's, and only the weight freed by C
        # is redistributed between them.
        self.assertAlmostEqual(
            drifted.loc[exit_date, "A"] / drifted.loc[exit_date, "B"],
            unmasked.loc[exit_date, "A"] / unmasked.loc[exit_date, "B"],
        )
        self.assertGreater(drifted.loc[exit_date, "A"], unmasked.loc[exit_date, "A"])

    def test_matches_caller_side_mask_and_renormalise(self):
        universe = self._exits("C", "2020-02-12")
        drifted = self._drift(universe)

        # The workaround being replaced: drift the whole book, mask it, renormalise.
        unmasked = self._drift(_all_investable(self.returns))
        masked = unmasked.where(universe.reindex(unmasked.index).fillna(True))
        masked = masked.div(masked.sum(axis=1).replace(0.0, np.nan), axis=0)

        pd.testing.assert_frame_equal(masked, drifted, check_freq=False)

    def test_membership_is_carried_forward(self):
        # Membership is a state, so recording only the change is enough; an exit is
        # stated as False rather than as an absence, so an unlimited carry is safe here
        # in a way it is not for a target.
        sparse = pd.DataFrame(
            [[True, True, False]],
            index=[pd.Timestamp("2020-02-12")],
            columns=self.cids,
        )
        pd.testing.assert_frame_equal(
            self._drift(sparse), self._drift(self._exits("C", "2020-02-12"))
        )

    def test_dates_before_the_first_record_are_investable(self):
        # A partial frame narrows the book; it must not empty the dates it omits.
        sparse = pd.DataFrame(
            [[True, True, False]],
            index=[pd.Timestamp("2020-02-12")],
            columns=self.cids,
        )
        drifted = self._drift(sparse)
        self.assertFalse(drifted.loc[: pd.Timestamp("2020-02-11")].isna().any().any())

    def test_accepts_long_format_universe(self):
        universe = self._exits("C", "2020-02-12")
        pd.testing.assert_frame_equal(
            self._drift(_to_long(universe.astype(float))), self._drift(universe)
        )

    def test_unknown_security_raises(self):
        universe = self._exits("C", "2020-02-12")
        universe["D"] = True
        with self.assertRaisesRegex(ValueError, "unknown to `weights` and `returns`"):
            self._drift(universe)

    def test_missing_security_raises(self):
        # The failure this parameter exists to prevent: a stale membership frame that
        # would silently drop every security it has fallen behind on.
        with self.assertRaisesRegex(ValueError, "Missing from `universe`"):
            self._drift(self._exits("C", "2020-02-12").drop(columns=["C"]))

    def test_masked_weights_never_register_as_holdings(self):
        # NaN and 0.0 read the same downstream; only the panel is more informative.
        stats = _concentration_stats(self._drift(self._exits("C", "2020-02-12")))
        self.assertTrue(stats.loc[:"2020-02-11", "n_holdings"].eq(3).all())
        self.assertTrue(stats.loc["2020-02-12":, "n_holdings"].eq(2).all())


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


class TestUnbrokenSchedule(unittest.TestCase):
    @staticmethod
    def _weights(dates):
        return pd.DataFrame(
            0.5, index=dates, columns=["A", "B"], dtype=float
        ).rename_axis("real_date")

    def test_contiguous_sample_is_accepted(self):
        dates = pd.bdate_range("2020-01-01", "2020-06-30")
        for freq in ["B", "W", "M", "Q"]:
            PortfolioAnalyser(self._weights(dates), freq)

    def test_a_skipped_period_raises(self):
        dates = pd.bdate_range("2020-01-01", "2020-06-30")
        gapped = dates[dates.month != 3]
        with self.assertRaisesRegex(ValueError, r"no observation in 1 'M'"):
            PortfolioAnalyser(self._weights(gapped), "M")

    def test_the_error_names_the_missing_periods(self):
        dates = pd.bdate_range("2020-01-01", "2020-12-31")
        gapped = dates[~dates.month.isin([3, 7])]
        with self.assertRaises(ValueError) as caught:
            PortfolioAnalyser(self._weights(gapped), "M")
        message = str(caught.exception)
        self.assertIn("2020-03", message)
        self.assertIn("2020-07", message)
        self.assertIn("no observation in 2 'M'", message)

    def test_business_day_holidays_are_not_a_gap(self):
        # Each "B" period is a single observation, so an absent one is a market
        # holiday, not missing data - and 252 is already net of holidays.
        dates = pd.bdate_range("2020-01-01", "2020-06-30")
        holidays = dates[~dates.isin(dates[[10, 11, 40]])]
        analyser = PortfolioAnalyser(self._weights(holidays), "B")
        self.assertEqual(len(analyser.trade_dates), len(holidays))

    def test_partial_periods_at_the_edges_are_not_a_gap(self):
        # Starting and ending mid-month leaves both periods short, not absent.
        dates = pd.bdate_range("2020-01-17", "2020-05-08")
        analyser = PortfolioAnalyser(self._weights(dates), "M")
        self.assertEqual(len(analyser.trade_dates), 5)

    def test_helper_raises_on_a_broken_period_index(self):
        _assert_unbroken_schedule(pd.PeriodIndex(["2020-01", "2020-02"], freq="M"), "M")
        with self.assertRaises(ValueError):
            _assert_unbroken_schedule(
                pd.PeriodIndex(["2020-01", "2020-03"], freq="M"), "M"
            )


class TestAnnualisedTurnover(unittest.TestCase):
    ANNUALISED = [
        "turnover_annualised",
        "active_turnover_annualised",
        "active_weight_turnover_annualised",
    ]
    FACTORS = {"B": 252.0, "W": 52.0, "M": 12.0, "Q": 4.0, "Y": 1.0}

    def setUp(self):
        self.weights, self.benchmark, self.returns, self.groups = _random_portfolio()
        self.analyser = PortfolioAnalyser(
            self.weights,
            FREQ,
            benchmark=self.benchmark,
            returns=self.returns,
            groups=self.groups,
        )

    def test_factor_follows_the_rebalance_frequency(self):
        for freq, factor in self.FACTORS.items():
            analyser = PortfolioAnalyser(self.weights, freq)
            self.assertEqual(analyser.rebalancings_per_year, factor)

    def test_annualised_is_the_raw_reading_times_the_factor(self):
        stats = self.analyser.weight_stats(active=True)
        standalone = self.analyser.weight_stats()
        factor = self.analyser.rebalancings_per_year
        np.testing.assert_allclose(
            standalone["turnover_annualised"].dropna(),
            standalone["turnover"].dropna() * factor,
        )
        for stat in ["active_turnover", "active_weight_turnover"]:
            np.testing.assert_allclose(
                stats[f"{stat}_annualised"].dropna(), stats[stat].dropna() * factor
            )

    def test_annualised_columns_are_trade_date_only(self):
        stats = self.analyser.weight_stats(active=True).set_index("real_date")
        standalone = self.analyser.weight_stats().set_index("real_date")
        n_trades = len(self.analyser.trade_dates) - 1
        self.assertEqual(int(standalone["turnover_annualised"].notna().sum()), n_trades)
        pd.testing.assert_series_equal(
            standalone["turnover"].isna(),
            standalone["turnover_annualised"].isna(),
            check_names=False,
        )
        for stat in self.ANNUALISED[1:]:
            self.assertEqual(int(stats[stat].notna().sum()), n_trades, stat)

    def test_annual_rebalancing_is_the_identity(self):
        weights, _, returns, _ = _random_portfolio(n_dates=800)
        analyser = PortfolioAnalyser(weights, "Y", returns=returns)
        stats = analyser.weight_stats()
        np.testing.assert_allclose(
            stats["turnover_annualised"].dropna(), stats["turnover"].dropna()
        )

    def test_group_statistics_sum_to_the_whole_portfolio(self):
        total = self.analyser.weight_stats(active=True).set_index("real_date")
        grouped = self.analyser.weight_stats(by_group=True, active=True).pivot(
            index="real_date", columns="group"
        )
        for col in self.ANNUALISED[1:]:
            np.testing.assert_allclose(
                grouped[col].sum(axis=1, min_count=1), total[col], rtol=1e-10, atol=1e-9
            )

    def test_monotone_path_annualises_alike_across_cadences(self):
        # The point of the feature: a book walking steadily from one allocation to
        # another trades the same amount a year however finely the journey is cut,
        # because turnover telescopes along a path that never doubles back.
        dates = pd.bdate_range("2020-01-01", periods=252 * 3)
        ramp = np.linspace(0.5, 0.9, len(dates))
        weights = pd.DataFrame(
            {"A": ramp, "B": 1.0 - ramp}, index=dates
        ).rename_axis("real_date")

        annualised = {}
        for freq in self.FACTORS:
            stats = PortfolioAnalyser(weights, freq).weight_stats()
            annualised[freq] = stats["turnover_annualised"].dropna().mean()

        # The spread is the calendar convention alone: 252 trading days a year is
        # stated net of holidays, while the fixture runs on ~261 business days.
        readings = np.array(list(annualised.values()))
        self.assertLess(np.ptp(readings) / readings.mean(), 0.05)
        np.testing.assert_allclose(annualised["M"], annualised["Y"], rtol=0.01)

    def test_reversing_path_does_not_annualise_alike(self):
        # The documented caveat: turnover is the length of the path the weights
        # travel, not the distance between its endpoints, so a book that doubles back
        # trades far more a year when it is rebalanced more often. Same endpoints.
        dates = pd.bdate_range("2020-01-01", periods=252 * 3)
        ramp = np.linspace(0.5, 0.9, len(dates))
        wiggle = ramp + 0.05 * np.sin(np.linspace(0, 40 * np.pi, len(dates)))
        wiggle[0], wiggle[-1] = ramp[0], ramp[-1]
        weights = pd.DataFrame(
            {"A": wiggle, "B": 1.0 - wiggle}, index=dates
        ).rename_axis("real_date")

        daily, yearly = (
            PortfolioAnalyser(weights, freq)
            .weight_stats()["turnover_annualised"]
            .dropna()
            .mean()
            for freq in ("B", "Y")
        )
        self.assertGreater(daily / yearly, 5.0)

    def test_drift_correction_scales_with_the_root_of_the_frequency(self):
        # A constant target pulled back to itself each period trades only to undo
        # drift. That is a random walk in weight space, whose path length grows with
        # the square root of the number of steps, so the annualised figure grows with
        # the square root of the cadence rather than staying put.
        rng = np.random.default_rng(0)
        dates = pd.bdate_range("2020-01-01", periods=252 * 3)
        cids = [f"S{i}" for i in range(20)]
        returns = pd.DataFrame(
            rng.normal(0, 1.0, (len(dates), len(cids))), index=dates, columns=cids
        )
        target = pd.DataFrame(1.0 / len(cids), index=dates, columns=cids)

        daily, yearly = (
            PortfolioAnalyser(target, freq, returns=returns)
            .weight_stats()["turnover_annualised"]
            .dropna()
            .mean()
            for freq in ("B", "Y")
        )
        self.assertGreater(daily / yearly, 0.4 * np.sqrt(252))
        self.assertLess(daily / yearly, 2.0 * np.sqrt(252))

    def test_labels_and_categories(self):
        labels = weight_stat_labels("PORT", benchmark="SP500")
        self.assertEqual(labels["PORT_TURNOVER_ANNUALISED"], "Portfolio turnover, % p.a.")
        self.assertEqual(
            labels["PORT_ACTIVE_TURNOVER_ANNUALISED"],
            "Signal-driven turnover, % p.a. (vs SP500)",
        )
        qdf = self.analyser.weight_stats(active=True, as_qdf=True)
        self.assertIn(
            "PORT_ACTIVE_WEIGHT_TURNOVER_ANNUALISED",
            set(map(str, qdf["xcat"].unique())),
        )


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


class TestBrinsonAttribution(unittest.TestCase):
    def setUp(self):
        self.weights, self.benchmark, self.returns, self.groups = _random_portfolio()
        self.analyser = PortfolioAnalyser(
            self.weights,
            FREQ,
            benchmark=self.benchmark,
            returns=self.returns,
            groups=self.groups,
        )

    def test_pure_selection_case(self):
        # Both sides fully invested in the same single group, holding opposite
        # securities: the group weights match exactly (no allocation bet), so the
        # whole active return must show up as selection.
        weights = _wide([[1.0, 0.0], [1.0, 0.0]], ["A", "B"])
        benchmark = _wide([[0.0, 1.0], [0.0, 1.0]], ["A", "B"])
        returns = _wide([[0.0, 0.0], [10.0, -10.0]], ["A", "B"])
        analyser = PortfolioAnalyser(
            weights,
            "B",
            benchmark=benchmark,
            returns=returns,
            groups={"A": "G", "B": "G"},
        )
        contrib = analyser.brinson_attribution(include_total=False)
        row = contrib[contrib["real_date"] == returns.index[1]].iloc[0]
        self.assertAlmostEqual(row["allocation"], 0.0)
        self.assertAlmostEqual(row["selection"], 20.0)

    def test_pure_allocation_case(self):
        # Single-security groups: portfolio and benchmark hold the same security in
        # each group, just at different weights, so there is no selection to be made
        # and the whole active return must show up as allocation.
        weights = _wide([[0.7, 0.3], [0.7, 0.3]], ["A", "B"])
        benchmark = _wide([[0.5, 0.5], [0.5, 0.5]], ["A", "B"])
        returns = _wide([[0.0, 0.0], [10.0, -10.0]], ["A", "B"])
        analyser = PortfolioAnalyser(
            weights,
            "B",
            benchmark=benchmark,
            returns=returns,
            groups={"A": "G1", "B": "G2"},
        )
        day1 = analyser.brinson_attribution(include_total=False)
        day1 = day1[day1["real_date"] == returns.index[1]]
        self.assertAlmostEqual(day1["allocation"].sum(), 4.0)
        self.assertAlmostEqual(day1["selection"].sum(), 0.0)

    def test_total_reconciles_to_active_return(self):
        active_return = self.analyser.attribution(active=True)["PORTFOLIO"]
        total = self.analyser.brinson_attribution()
        total = total[total["group"] == self.analyser.portfolio_name].set_index(
            "real_date"
        )
        reconciled = total["allocation"] + total["selection"]
        np.testing.assert_allclose(
            reconciled.reindex(active_return.index), active_return, rtol=1e-8, atol=1e-8
        )

    def test_group_effects_sum_to_total(self):
        brinson = self.analyser.brinson_attribution()
        by_group = brinson[brinson["group"] != self.analyser.portfolio_name]
        summed = by_group.groupby("real_date")[BRINSON_STATS].sum()
        total = brinson[
            brinson["group"] == self.analyser.portfolio_name
        ].set_index("real_date")[BRINSON_STATS]
        pd.testing.assert_frame_equal(summed, total, check_like=True)

    def test_include_total_can_be_switched_off(self):
        brinson = self.analyser.brinson_attribution(include_total=False)
        self.assertNotIn(self.analyser.portfolio_name, brinson["group"].unique())

    def test_returns_are_required(self):
        analyser = PortfolioAnalyser(
            self.weights, FREQ, benchmark=self.benchmark, groups=self.groups
        )
        with self.assertRaisesRegex(ValueError, "`returns` must be supplied"):
            analyser.brinson_attribution()

    def test_groups_are_required(self):
        analyser = PortfolioAnalyser(
            self.weights, FREQ, benchmark=self.benchmark, returns=self.returns
        )
        with self.assertRaisesRegex(ValueError, "`groups` must be supplied"):
            analyser.brinson_attribution()

    def test_benchmark_is_required(self):
        analyser = PortfolioAnalyser(
            self.weights, FREQ, returns=self.returns, groups=self.groups
        )
        with self.assertRaisesRegex(ValueError, "`benchmark` must be supplied"):
            analyser.brinson_attribution()

    def test_lag_must_be_a_non_negative_integer(self):
        for bad in (-1, 1.5, "1", True):
            with self.assertRaises(TypeError):
                self.analyser.brinson_attribution(lag=bad)

    def test_total_label_collision_raises(self):
        analyser = PortfolioAnalyser(
            self.weights,
            FREQ,
            benchmark=self.benchmark,
            returns=self.returns,
            groups=self.groups,
            portfolio_name="TECH",
        )
        with self.assertRaisesRegex(ValueError, "collides"):
            analyser.brinson_attribution()

    def test_as_qdf_carries_two_categories(self):
        qdf = self.analyser.brinson_attribution(as_qdf=True)
        self.assertIsInstance(qdf, QuantamentalDataFrame)
        self.assertEqual(
            sorted(map(str, qdf["xcat"].unique())),
            ["PORT_ALLOCATION", "PORT_SELECTION"],
        )


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
