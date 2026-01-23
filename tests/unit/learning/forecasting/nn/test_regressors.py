import unittest
import numpy as np
import pandas as pd

from macrosynergy.learning.forecasting.nn import MLPRegressor


class TestMLPRegressor(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        # Generate data with true linear relationship
        cids = ["AUD", "CAD", "GBP", "USD"]
        xcats = ["XR", "XR2", "GROWTH", "RIR"]

        df_cids = pd.DataFrame(index=cids, columns=["earliest", "latest"])
        df_cids.loc["AUD"] = ["2019-01-01", "2020-12-31"]
        df_cids.loc["CAD"] = ["2020-01-01", "2020-12-31"]
        df_cids.loc["GBP"] = ["2020-01-01", "2020-12-31"]
        df_cids.loc["USD"] = ["2019-06-01", "2020-12-31"]

        tuples = []

        for cid in cids:
            # get list of all eligible dates
            sdate = df_cids.loc[cid]["earliest"]
            edate = df_cids.loc[cid]["latest"]
            all_days = pd.date_range(sdate, edate)
            work_days = all_days[all_days.weekday < 5]
            for work_day in work_days:
                tuples.append((cid, work_day))

        n_samples = len(tuples)
        ftrs = np.random.normal(loc=0, scale=1, size=(n_samples, 2))
        labels1 = np.matmul(ftrs, [1, 2]) + np.random.normal(0, 0.5, len(ftrs))
        labels2 = np.matmul(ftrs, [-1, 0.5]) + np.random.normal(0, 0.5, len(ftrs))
        labels2 += 0.1 - 0.75 * labels1 + np.random.normal(0, 0.5, len(ftrs))
        df = pd.DataFrame(
            data=np.concatenate((np.reshape(labels1, (-1, 1)), np.reshape(labels2, (-1, 1)), ftrs), axis=1),
            index=pd.MultiIndex.from_tuples(tuples, names=["cid", "real_date"]),
            columns=xcats,
            dtype=np.float32,
        )

        self.X = df.drop(columns=["XR", "XR2"])
        self.X_numpy = self.X.values
        self.X_nan = self.X.copy()
        self.X_nan["nan_col"] = np.nan
        self.y = df[["XR", "XR2"]]
        self.y_numpy = self.y.values
        self.y_nan = self.y.copy()
        self.y_nan.iloc[0] = np.nan

    def _make_reg(self, **overrides):
        # Keep training short for unit tests
        defaults = dict(
            n_latent=8,
            epochs=60,
            patience=10,
            batch_size=32,
            learning_rate=1e-3,
            train_pct=0.7,
            verbose=False,
            random_state=123,
            encoder_activation="tanh",
            head_activation="identity",
            fit_encoder_intercept=False,
            fit_head_intercept=True,
        )
        defaults.update(overrides)
        return MLPRegressor(**defaults)

    def test_init_params(self):
        reg = self._make_reg()
        p = reg.get_params(deep=True)

        for k in [
            "n_latent",
            "weight_decay",
            "reg_turnover",
            "batch_size",
            "learning_rate",
            "use_ts_sampler",
            "encoder_activation",
            "head_activation",
            "fit_encoder_intercept",
            "fit_head_intercept",
            "epochs",
            "patience",
            "train_pct",
            "verbose",
            "random_state",
        ]:
            self.assertIn(k, p)

        reg.set_params(learning_rate=5e-4, fit_head_intercept=False, head_activation="tanh")
        self.assertEqual(reg.learning_rate, 5e-4)
        self.assertFalse(reg.fit_head_intercept)
        self.assertEqual(reg.head_activation, "tanh")

    def test_fit_returns_self_and_scalers(self):
        reg = self._make_reg()
        out = reg.fit(self.X, self.y)
        self.assertIs(out, reg)

        self.assertIsNotNone(reg.model)
        self.assertTrue(hasattr(reg, "x_scaler"))
        self.assertTrue(hasattr(reg, "y_scaler"))

    def test_predict_shape_and_finite(self):
        reg = self._make_reg()
        reg.fit(self.X, self.y)

        preds = reg.predict(self.X[:100])
        self.assertEqual(preds.shape, self.y[:100].shape)
        self.assertTrue(np.isfinite(preds).all())

    def test_predict_before_fit_raises(self):
        reg = self._make_reg()
        with self.assertRaises(Exception):
            _ = reg.predict(self.X[:10])

    def test_reproducibility_same_seed(self):
        reg1 = self._make_reg(random_state=7)
        reg2 = self._make_reg(random_state=7)

        reg1.fit(self.X, self.y)
        reg2.fit(self.X, self.y)

        p1 = reg1.predict(self.X[:200])
        p2 = reg2.predict(self.X[:200])
        np.testing.assert_allclose(p1, p2, rtol=1e-5, atol=1e-5)
