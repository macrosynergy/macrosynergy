import unittest
import warnings

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import make_scorer

from macrosynergy.learning import (
    ExpandingKFoldPanelSplit,
    ModelAveragingRegressor,
    correlation_coefficient,
)


def _panel(seed=0, n_cids=3, n_months=120, n_feats=4):
    """A small panel with a linear signal, indexed by (cid, real_date)."""
    rng = np.random.default_rng(seed)
    cids = [f"C{i}" for i in range(n_cids)]
    dates = pd.bdate_range("2010-01-29", periods=n_months, freq="BME")
    index = pd.MultiIndex.from_product([cids, dates], names=["cid", "real_date"])
    X = pd.DataFrame(
        rng.normal(size=(len(index), n_feats)),
        index=index,
        columns=[f"F{i}" for i in range(n_feats)],
    )
    beta = np.linspace(0.5, -0.5, n_feats)
    y = pd.Series(X.values @ beta + rng.normal(scale=0.5, size=len(index)), index=index, name="XR")
    return X, y


class TestModelAveragingWeights(unittest.TestCase):
    def setUp(self):
        self.X, self.y = _panel()
        self.scoring = make_scorer(correlation_coefficient)
        self.cv = ExpandingKFoldPanelSplit(n_splits=3)

    def _fit(self, estimators, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return ModelAveragingRegressor(
                estimators=estimators, scoring=self.scoring, cv=self.cv, **kwargs
            ).fit(self.X, self.y)

    def test_regular_ensemble_weights_are_finite_and_sum_to_one(self):
        model = self._fit(
            [
                ("ridge_1", Ridge(alpha=1.0), {}),
                ("ridge_100", Ridge(alpha=100.0), {}),
                ("ridge_1e6", Ridge(alpha=1e6), {}),
            ]
        )
        w = np.array(list(model.weights_.values()))
        self.assertTrue(np.all(np.isfinite(w)))
        self.assertAlmostEqual(w.sum(), 1.0, places=10)
        self.assertTrue(np.all(w > 0.0))
        preds = model.predict(self.X)
        self.assertTrue(np.all(np.isfinite(preds)))

    def test_constant_prediction_member_gets_zero_weight_not_nan(self):
        # A constant forecast has an undefined correlation with the target, so its
        # cross-validation score is NaN. Before the fix this made every weight NaN
        # and the ensemble predicted NaN.
        model = self._fit(
            [
                ("ridge_1", Ridge(alpha=1.0), {}),
                ("ridge_100", Ridge(alpha=100.0), {}),
                ("constant", DummyRegressor(strategy="constant", constant=0.0), {}),
            ]
        )
        self.assertTrue(np.isnan(model.cv_scores_["constant"]))
        w = model.weights_
        self.assertTrue(all(np.isfinite(v) for v in w.values()))
        self.assertEqual(w["constant"], 0.0)
        self.assertAlmostEqual(w["ridge_1"] + w["ridge_100"], 1.0, places=10)
        preds = model.predict(self.X)
        self.assertTrue(np.all(np.isfinite(preds)))

    def test_constant_prediction_member_warns(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ModelAveragingRegressor(
                estimators=[
                    ("ridge_1", Ridge(alpha=1.0), {}),
                    ("constant", DummyRegressor(strategy="constant", constant=0.0), {}),
                ],
                scoring=self.scoring,
                cv=self.cv,
            ).fit(self.X, self.y)
        messages = [str(c.message) for c in caught if issubclass(c.category, RuntimeWarning)]
        self.assertTrue(any("non-finite cross-validation score" in m for m in messages))

    def test_single_member_gets_weight_one(self):
        # With one estimator the spread of scores is zero; the softmax limit is
        # weight one rather than a division by zero
        model = self._fit([("ridge_1", Ridge(alpha=1.0), {})])
        self.assertEqual(model.weights_["ridge_1"], 1.0)
        preds = model.predict(self.X)
        self.assertTrue(np.all(np.isfinite(preds)))
        np.testing.assert_allclose(preds, Ridge(alpha=1.0).fit(self.X, self.y).predict(self.X))

    def test_tied_scores_give_equal_weights(self):
        model = self._fit(
            [
                ("ridge_a", Ridge(alpha=10.0), {}),
                ("ridge_b", Ridge(alpha=10.0), {}),
            ]
        )
        self.assertAlmostEqual(model.weights_["ridge_a"], 0.5, places=10)
        self.assertAlmostEqual(model.weights_["ridge_b"], 0.5, places=10)

    def test_all_members_non_finite_falls_back_to_equal_weights(self):
        model = self._fit(
            [
                ("constant_a", DummyRegressor(strategy="constant", constant=0.0), {}),
                ("constant_b", DummyRegressor(strategy="constant", constant=1.0), {}),
            ]
        )
        self.assertAlmostEqual(model.weights_["constant_a"], 0.5, places=10)
        self.assertAlmostEqual(model.weights_["constant_b"], 0.5, places=10)
        self.assertTrue(np.all(np.isfinite(model.predict(self.X))))

    def test_min_weight_quantile_ignores_excluded_members(self):
        # the "median" floor is read off the finite members only, so the excluded
        # member's zero does not drag the floor to zero
        model = self._fit(
            [
                ("ridge_1", Ridge(alpha=1.0), {}),
                ("ridge_100", Ridge(alpha=100.0), {}),
                ("ridge_1e6", Ridge(alpha=1e6), {}),
                ("constant", DummyRegressor(strategy="constant", constant=0.0), {}),
            ],
            min_weight="median",
        )
        w = model.weights_
        self.assertEqual(w["constant"], 0.0)
        self.assertAlmostEqual(sum(w.values()), 1.0, places=10)
        self.assertGreaterEqual(sum(1 for v in w.values() if v == 0.0), 2)


if __name__ == "__main__":
    unittest.main()
