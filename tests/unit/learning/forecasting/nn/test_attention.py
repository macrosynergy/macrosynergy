import unittest

import numpy as np
import pandas as pd

from macrosynergy.learning.forecasting.nn import AttentionRegressor


class TestAttentionRegressor(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        np.random.seed(0)

        cids = ["AUD", "CAD", "GBP"]
        dates = pd.date_range("2020-01-31", periods=30, freq="M")
        index = pd.MultiIndex.from_product([cids, dates], names=["cid", "real_date"])

        f1 = np.random.normal(size=len(index))
        f2 = np.random.normal(size=len(index))
        y = 0.7 * f1 - 0.3 * f2 + np.random.normal(scale=0.1, size=len(index))

        self.X = pd.DataFrame(
            {"f1": f1.astype(np.float32), "f2": f2.astype(np.float32)},
            index=index,
        )
        self.y = pd.Series(y.astype(np.float32), index=index, name="target")

    def _make_reg(self, **overrides):
        defaults = dict(
            lookback=6,
            d_model=8,
            n_heads=2,
            n_layers=1,
            batch_size=8,
            learning_rate=1e-3,
            epochs=20,
            patience=5,
            train_pct=0.7,
            random_state=123,
            verbose=False,
        )
        defaults.update(overrides)
        return AttentionRegressor(**defaults)

    def test_init_params(self):
        reg = self._make_reg()
        params = reg.get_params(deep=True)
        self.assertIn("lookback", params)
        self.assertIn("d_model", params)
        self.assertIn("n_heads", params)
        self.assertIn("n_layers", params)

    def test_fit_returns_self(self):
        reg = self._make_reg()
        out = reg.fit(self.X, self.y)
        self.assertIs(out, reg)
        self.assertTrue(reg.is_fitted_)
        self.assertTrue(hasattr(reg, "model_"))
        self.assertTrue(hasattr(reg, "x_scaler_"))
        self.assertTrue(hasattr(reg, "y_scaler_"))

    def test_predict_shape_and_finite(self):
        reg = self._make_reg()
        reg.fit(self.X, self.y)

        preds = reg.predict(self.X)
        self.assertIsInstance(preds, pd.Series)
        self.assertTrue(np.isfinite(preds.to_numpy()).all())
        self.assertEqual(preds.index.nlevels, 2)

    def test_predict_with_attention(self):
        reg = self._make_reg()
        reg.fit(self.X, self.y)

        preds, attn = reg.predict_with_attention(self.X)
        self.assertIsInstance(preds, pd.Series)
        self.assertIsInstance(attn, pd.DataFrame)
        self.assertEqual(len(preds), len(attn))
        self.assertEqual(attn.shape[1], reg.lookback)
        np.testing.assert_allclose(attn.sum(axis=1).to_numpy(), 1.0, atol=1e-5)

    def test_predict_before_fit_raises(self):
        reg = self._make_reg()
        with self.assertRaises(RuntimeError):
            reg.predict(self.X)

    def test_invalid_fit_params(self):
        with self.assertRaises(ValueError):
            self._make_reg(lookback=1).fit(self.X, self.y)
        with self.assertRaises(ValueError):
            self._make_reg(batch_size=0).fit(self.X, self.y)
        with self.assertRaises(ValueError):
            self._make_reg(train_pct=1.0).fit(self.X, self.y)


if __name__ == "__main__":
    unittest.main()
