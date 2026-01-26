# test_mlp_trainer_unittest.py
import unittest

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from macrosynergy.learning.forecasting.torch import MultiLayerPerceptron, MLPTrainer


class TestMLPTrainer(unittest.TestCase):

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


    def test_time_split_no_leakage(self):
        tr = MLPTrainer(train_pct=0.6, x_scaler=None, y_scaler=None)
        X_tr, y_tr, X_va, y_va = tr._time_split(self.X, self.y)

        dates = sorted(self.X.index.get_level_values(1).unique())
        cut = int(0.6 * len(dates))
        train_dates = set(dates[:cut])
        val_dates = set(dates[cut:])

        self.assertEqual(set(X_tr.index.get_level_values("real_date").unique()), train_dates)
        self.assertEqual(set(X_va.index.get_level_values("real_date").unique()), val_dates)
        self.assertEqual(set(y_tr.index.get_level_values("real_date").unique()), train_dates)
        self.assertEqual(set(y_va.index.get_level_values("real_date").unique()), val_dates)


    def test_fit_fits_scalers_and_updates_weights(self):
        tr = MLPTrainer(
            train_pct=0.75,
            batch_size=128,
            epochs=5,
            patience=2,
            learning_rate=5e-3,
            x_scaler=StandardScaler(with_mean=False),
            y_scaler=StandardScaler(with_mean=False),
            random_state=0,
        )

        model = MultiLayerPerceptron(
            n_inputs=self.X.shape[1],
            n_latent=8,
            n_outputs=self.y.shape[1],
        )

        before = {k: v.detach().clone() for k, v in model.state_dict().items()}

        fitted_model, x_scaler, y_scaler = tr.fit(model, self.X, self.y)

        self.assertIs(fitted_model, model)
        self.assertIs(x_scaler, tr.x_scaler)
        self.assertIs(y_scaler, tr.y_scaler)

        self.assertTrue(hasattr(x_scaler, "scale_"))
        self.assertTrue(hasattr(y_scaler, "scale_"))

        after = model.state_dict()
        changed = any(not torch.equal(before[k], after[k]) for k in before.keys())
        self.assertTrue(changed)

    def test_fit_with_time_series_sampler_path(self):
        tr = MLPTrainer(
            train_pct=0.8,
            batch_size=128,
            epochs=2,
            patience=1,
            use_ts_sampler=True,
            x_scaler=StandardScaler(with_mean=False),
            y_scaler=StandardScaler(with_mean=False),
            random_state=0,
        )

        model = MultiLayerPerceptron(
            n_inputs=self.X.shape[1],
            n_latent=8,
            n_outputs=self.y.shape[1],
        )

        fitted_model, _, _ = tr.fit(model, self.X, self.y)
        self.assertIs(fitted_model, model)


if __name__ == "__main__":
    unittest.main()
