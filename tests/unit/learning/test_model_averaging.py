import unittest
import warnings

import numpy as np

from macrosynergy.learning.model_selection.model_averaging import (
    ModelAveragingRegressor,
)


def _softmax(scores, spread):
    scaled = np.asarray(scores, dtype=float) / spread
    scaled = scaled - np.max(scaled)
    weights = np.exp(scaled)
    return weights / weights.sum()


class TestModelAveragingWeights(unittest.TestCase):
    def _reg(self, names):
        reg = ModelAveragingRegressor.__new__(ModelAveragingRegressor)
        reg.model_names_ = list(names)
        return reg

    def _weights(self, scores, temperature="max-min", min_weight=0.0):
        names = list(scores)
        reg = self._reg(names)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", RuntimeWarning)
            weights = reg._compute_weights(scores, temperature, min_weight)
        return weights, caught

    def test_unequal_finite_scores_match_range_softmax(self):
        scores = {"ridge": 0.2, "lasso": 0.8}
        weights, caught = self._weights(scores)
        expected = _softmax([0.2, 0.8], 0.6)
        self.assertEqual(caught, [])
        self.assertAlmostEqual(weights["ridge"], expected[0])
        self.assertAlmostEqual(weights["lasso"], expected[1])
        self.assertAlmostEqual(sum(weights.values()), 1.0)

    def test_scale_methods_match_softmax_of_finite_scores(self):
        scores = {"a": 1.0, "b": 2.0, "c": 5.0}
        values = np.array([1.0, 2.0, 5.0])
        spreads = {
            "max-min": np.max(values) - np.min(values),
            "std": np.std(values),
            "mad": np.median(np.abs(values - np.median(values))),
            "iqr": np.percentile(values, 75) - np.percentile(values, 25),
        }
        for temperature, spread in spreads.items():
            weights, caught = self._weights(scores, temperature=temperature)
            expected = _softmax(values, spread)
            self.assertEqual(caught, [], temperature)
            for name, exp in zip(scores, expected):
                self.assertAlmostEqual(weights[name], exp, msg=temperature)

    def test_nonfinite_member_is_zero_and_others_keep_softmax(self):
        scores = {"ridge": 0.2, "lasso": 0.8, "constant": np.nan}
        weights, caught = self._weights(scores)
        expected = _softmax([0.2, 0.8], 0.6)
        self.assertEqual(len(caught), 1)
        self.assertIn("constant", str(caught[0].message))
        self.assertAlmostEqual(weights["ridge"], expected[0])
        self.assertAlmostEqual(weights["lasso"], expected[1])
        self.assertEqual(weights["constant"], 0.0)
        self.assertTrue(np.isfinite(list(weights.values())).all())

    def test_infinite_score_is_excluded(self):
        scores = {"ridge": 0.4, "broken": np.inf}
        weights, caught = self._weights(scores)
        self.assertEqual(len(caught), 1)
        self.assertAlmostEqual(weights["ridge"], 1.0)
        self.assertEqual(weights["broken"], 0.0)

    def test_single_member_has_weight_one(self):
        weights, caught = self._weights({"ridge": 0.3})
        self.assertEqual(caught, [])
        self.assertAlmostEqual(weights["ridge"], 1.0)

    def test_tied_scores_share_weight(self):
        weights, caught = self._weights({"a": 0.5, "b": 0.5, "c": 0.5})
        self.assertEqual(caught, [])
        for name in ("a", "b", "c"):
            self.assertAlmostEqual(weights[name], 1.0 / 3.0)

    def test_all_nonfinite_scores_fall_back_to_equal_weights(self):
        weights, caught = self._weights({"a": np.nan, "b": np.inf})
        self.assertEqual(len(caught), 1)
        self.assertIn("equal weights", str(caught[0].message))
        self.assertAlmostEqual(weights["a"], 0.5)
        self.assertAlmostEqual(weights["b"], 0.5)

    def test_zero_temperature_uses_equal_weights(self):
        weights, caught = self._weights(
            {"a": 1.0, "b": 4.0}, temperature=0.0
        )
        self.assertEqual(caught, [])
        self.assertAlmostEqual(weights["a"], 0.5)
        self.assertAlmostEqual(weights["b"], 0.5)

    def test_negative_temperature_still_inverts_ranking(self):
        weights, caught = self._weights(
            {"low": 1.0, "high": 3.0}, temperature=-1.0
        )
        self.assertEqual(caught, [])
        self.assertGreater(weights["low"], weights["high"])
        self.assertAlmostEqual(sum(weights.values()), 1.0)

    def test_numeric_floor_still_drops_a_small_finite_weight(self):
        weights, _ = self._weights({"small": 0.0, "large": 10.0}, min_weight=0.4)
        self.assertEqual(weights["small"], 0.0)
        self.assertAlmostEqual(weights["large"], 1.0)

    def test_median_floor_ignores_excluded_member(self):
        # Finite softmax is about (0.731, 0.269). The median of those two is
        # 0.5, so the smaller model is removed. Including the excluded zero
        # would make the median 0.269 and keep that smaller model.
        scores = {"strong": 10.0, "weak": 0.0, "constant": np.nan}
        weights, caught = self._weights(scores, min_weight="median")
        self.assertEqual(len(caught), 1)
        self.assertAlmostEqual(weights["strong"], 1.0)
        self.assertEqual(weights["weak"], 0.0)
        self.assertEqual(weights["constant"], 0.0)


if __name__ == "__main__":
    unittest.main()
