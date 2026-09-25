import unittest

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from macrosynergy.learning.forecasting.nn import PanelMLPRegressor
from macrosynergy.learning.forecasting.torch.losses import (
    NegCrossSectionalIC,
    NegSharpeRatio,
    RankingRiskLoss,
)
from macrosynergy.learning.forecasting.torch.modules import LongShortModule


def make_panel(n_assets=20, n_periods=90, seed=0, interaction=True, noise=1.0):
    """
    A synthetic panel whose signal is an interaction between a macro state and a
    security-specific loading.

    Each security carries a fixed loading `b_i`. The macro state `x_t` is identical
    across securities at each date, so the only way to rank securities within a date is
    to use the interaction `b_i * x_t`. That is precisely the "different securities build
    on different macro sensitivities" structure a shared head has to represent through
    characteristics rather than through free per-security coefficients.
    """
    rng = np.random.default_rng(seed)

    macro = np.zeros((n_periods, 2))
    for t in range(1, n_periods):
        macro[t] = 0.7 * macro[t - 1] + rng.normal(size=2)
    macro = (macro - macro.mean(0)) / macro.std(0)

    loadings = rng.normal(size=n_assets)

    rows, targets = [], []
    index = []
    dates = pd.bdate_range("2012-01-31", periods=n_periods, freq="BME")
    for i in range(n_assets):
        for t in range(n_periods):
            features = [macro[t, 0], macro[t, 1]]
            if interaction:
                features.append(loadings[i])
            rows.append(features)
            signal = loadings[i] * macro[t, 0] if interaction else macro[t, 0]
            targets.append(signal + noise * rng.normal())
            index.append((f"S{i:03d}", dates[t]))

    idx = pd.MultiIndex.from_tuples(index, names=["cid", "real_date"])
    columns = ["MACRO1", "MACRO2"] + (["LOADING"] if interaction else [])
    X = pd.DataFrame(np.array(rows), index=idx, columns=columns).sort_index()
    y = pd.Series(np.array(targets), index=idx, name="XR").sort_index()
    return X, y


def mean_cross_sectional_ic(preds, y):
    """Mean over dates of the within-date correlation between forecasts and outcomes."""
    frame = pd.DataFrame({"pred": preds, "true": y})
    per_date = frame.groupby(level=1).apply(
        lambda g: g["pred"].corr(g["true"]) if g["pred"].std() > 0 else np.nan
    )
    return float(per_date.mean())


class TestPanelMLPRegressor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X, cls.y = make_panel()
        cls.X_macro, cls.y_macro = make_panel(interaction=False)

    def test_types_fit(self):
        model = PanelMLPRegressor(epochs=2, patience=1)
        self.assertRaises(TypeError, model.fit, 1, self.y)
        self.assertRaises(TypeError, model.fit, self.X.values, self.y)
        self.assertRaises(ValueError, model.fit, self.X.reset_index(), self.y)
        self.assertRaises(TypeError, model.fit, self.X, 1)
        self.assertRaises(ValueError, model.fit, self.X, self.y.iloc[:-1])

        X_nan = self.X.copy()
        X_nan.iloc[0, 0] = np.nan
        self.assertRaises(ValueError, model.fit, X_nan, self.y)

        # A single forward return per observation
        with self.assertRaises(ValueError):
            PanelMLPRegressor(epochs=2, patience=1).fit(
                self.X, pd.DataFrame({"a": self.y, "b": self.y})
            )

        # Batching by security needs to know how many securities per batch
        with self.assertRaises(ValueError):
            PanelMLPRegressor(epochs=2, patience=1, batch_mode="asset").fit(self.X, self.y)
        with self.assertRaises(ValueError):
            PanelMLPRegressor(epochs=2, patience=1, batch_mode="grid").fit(self.X, self.y)

        # Shrinking an embedding that does not exist
        with self.assertRaises(ValueError):
            PanelMLPRegressor(epochs=2, patience=1, reg_embedding=0.1).fit(self.X, self.y)

    def test_valid_fit_and_predict(self):
        model = PanelMLPRegressor(
            n_latent=[16, 8], epochs=5, patience=3, batch_periods=16
        ).fit(self.X, self.y)

        self.assertEqual(len(model.models_), 1)
        self.assertEqual(model.n_assets_, 20)
        self.assertEqual(model.n_features_, 3)

        preds = model.predict(self.X)
        self.assertIsInstance(preds, pd.Series)
        pd.testing.assert_index_equal(preds.index, self.X.index)
        self.assertFalse(preds.isna().any())

    def test_valid_determinism(self):
        kwargs = dict(n_latent=8, epochs=4, patience=2, batch_periods=16)
        first = PanelMLPRegressor(random_state=1, **kwargs).fit(self.X, self.y).predict(self.X)
        again = PanelMLPRegressor(random_state=1, **kwargs).fit(self.X, self.y).predict(self.X)
        other = PanelMLPRegressor(random_state=2, **kwargs).fit(self.X, self.y).predict(self.X)

        pd.testing.assert_series_equal(first, again)
        with self.assertRaises(AssertionError):
            pd.testing.assert_series_equal(first, other)

    def test_valid_recovers_interaction_signal(self):
        """
        Known answer: on a panel whose only rankable structure is the interaction between
        the macro state and a security's loading, the shared head must find it.
        """
        model = PanelMLPRegressor(
            n_latent=[32, 16],
            encoder_activation="tanh",
            epochs=150,
            patience=40,
            batch_periods=32,
            learning_rate=1e-2,
            loss_func=NegCrossSectionalIC(min_names=10),
            random_state=0,
        ).fit(self.X, self.y)

        ic = mean_cross_sectional_ic(model.predict(self.X), self.y)
        self.assertGreater(ic, 0.15, msg=f"failed to recover the interaction, IC={ic:.4f}")

    def test_valid_macro_only_features_are_degenerate(self):
        """
        The precondition the class warns about: with features that do not vary across
        securities, every security receives the same forecast and no ranking exists.
        """
        model = PanelMLPRegressor(
            n_latent=8, epochs=5, patience=3, batch_periods=16
        ).fit(self.X_macro, self.y_macro)

        self.assertAlmostEqual(model.prediction_dispersion_, 0.0, places=10)

        preds = model.predict(self.X_macro)
        spread = preds.groupby(level=1).std()
        np.testing.assert_allclose(spread.to_numpy(), 0.0, atol=1e-6)

        # And with security-varying features it is not degenerate
        informative = PanelMLPRegressor(
            n_latent=8, epochs=5, patience=3, batch_periods=16
        ).fit(self.X, self.y)
        self.assertGreater(informative.prediction_dispersion_, 0.0)

    def test_valid_embedding_adds_per_security_freedom(self):
        model = PanelMLPRegressor(
            n_latent=8,
            embedding_dim=3,
            reg_embedding=1e-3,
            epochs=5,
            patience=3,
            batch_periods=16,
        ).fit(self.X, self.y)

        network = model.models_[0]
        self.assertIsNotNone(network.embedding)
        self.assertEqual(tuple(network.embedding.weight.shape), (20, 3))
        self.assertGreater(float(network.embedding_penalty()), 0.0)

        preds = model.predict(self.X)
        self.assertFalse(preds.isna().any())

    def test_valid_embedding_breaks_macro_only_degeneracy(self):
        """
        An embedding is the one way a shared head can separate securities without
        security-varying features -- which is exactly why it needs shrinking.
        """
        model = PanelMLPRegressor(
            n_latent=8,
            embedding_dim=2,
            epochs=10,
            patience=5,
            batch_periods=16,
            learning_rate=1e-2,
        ).fit(self.X_macro, self.y_macro)
        self.assertGreater(model.prediction_dispersion_, 0.0)

    def test_valid_batch_modes(self):
        """Every batching mode must train and predict."""
        for mode, kwargs in [
            ("period", dict(batch_periods=16)),
            ("asset", dict(batch_assets=8)),
            ("block", dict(batch_periods=16, batch_assets=8)),
            ("mc", dict(batch_periods=16, asset_fraction=0.75)),
        ]:
            model = PanelMLPRegressor(
                n_latent=8,
                epochs=4,
                patience=2,
                batch_mode=mode,
                loss_func=NegCrossSectionalIC(min_names=5),
                **kwargs,
            ).fit(self.X, self.y)
            preds = model.predict(self.X)
            self.assertFalse(preds.isna().any(), msg=mode)
            self.assertEqual(len(preds), len(self.X), msg=mode)

    def test_valid_mc_subsampling(self):
        """
        Monte Carlo cross-section resampling: each batch holds a block of periods and an
        independent draw of securities, so the fit cannot lean on any one security's
        realisation.
        """
        model = PanelMLPRegressor(
            n_latent=8,
            epochs=6,
            patience=4,
            batch_mode="mc",
            batch_periods=16,
            asset_fraction=0.75,
            draws_per_block=3,
            loss_func=NegCrossSectionalIC(min_names=5),
        ).fit(self.X, self.y)

        preds = model.predict(self.X)
        self.assertFalse(preds.isna().any())
        self.assertGreater(model.prediction_dispersion_, 0.0)

        # It needs exactly one of batch_assets or asset_fraction
        with self.assertRaises(ValueError):
            PanelMLPRegressor(epochs=2, patience=1, batch_mode="mc").fit(self.X, self.y)
        with self.assertRaises(ValueError):
            PanelMLPRegressor(
                epochs=2, patience=1, batch_mode="mc",
                batch_assets=10, asset_fraction=0.75,
            ).fit(self.X, self.y)

    def test_valid_portfolio_loss_on_the_grid(self):
        """
        The existing portfolio objectives work unchanged, because a batch is reshaped
        into the (period, security) matrix they expect.
        """
        model = PanelMLPRegressor(
            n_latent=8,
            epochs=5,
            patience=3,
            batch_periods=16,
            signal_modifier=LongShortModule(dollar_neutral=True),
            loss_func=NegSharpeRatio(),
        ).fit(self.X, self.y)
        self.assertFalse(model.predict(self.X).isna().any())

    def test_valid_ranking_risk_loss(self):
        model = PanelMLPRegressor(
            n_latent=8,
            epochs=5,
            patience=3,
            batch_mode="block",
            batch_periods=16,
            batch_assets=10,
            loss_func=RankingRiskLoss(reg_risk=0.1, risk="period_var", min_names=5),
        ).fit(self.X, self.y)
        self.assertFalse(model.predict(self.X).isna().any())

    def test_valid_unbalanced_panel(self):
        """Securities with partial history must not break the grid reshaping."""
        X = self.X.drop(self.X.loc["S000"].index[:40], level=1, errors="ignore")
        keep = ~(
            (X.index.get_level_values(0) == "S001")
            & (X.index.get_level_values(1) < X.index.get_level_values(1)[30])
        )
        X = X[keep]
        y = self.y.loc[X.index]

        model = PanelMLPRegressor(
            n_latent=8, epochs=4, patience=2, batch_periods=16,
            loss_func=NegCrossSectionalIC(min_names=5),
        ).fit(X, y)
        preds = model.predict(X)
        self.assertEqual(len(preds), len(X))
        self.assertFalse(preds.isna().any())

    def test_valid_ensemble_over_seeds(self):
        model = PanelMLPRegressor(
            n_latent=8, epochs=4, patience=2, batch_periods=16, random_state=[0, 1, 2]
        ).fit(self.X, self.y)
        self.assertEqual(len(model.models_), 3)
        self.assertEqual(len(model.training_history_), 3)
        self.assertFalse(model.predict(self.X).isna().any())

    def test_valid_training_history(self):
        model = PanelMLPRegressor(
            n_latent=8, epochs=12, patience=12, batch_periods=16
        ).fit(self.X, self.y)

        history = model.training_history_[0]
        self.assertEqual(len(history), 12)
        for record in history:
            self.assertIn("epoch", record)
            self.assertIn("train_loss", record)
            self.assertIn("valid_loss", record)
            self.assertIn("seed", record)
            self.assertIn("fold", record)
        # The reporting epochs carry the cross-sectional IC
        reporting = [r for r in history if not np.isnan(r["train_ic"])]
        self.assertGreater(len(reporting), 0)

    def test_valid_split_is_chronological(self):
        """No date may appear on both sides of the training cut."""
        model = PanelMLPRegressor(epochs=1, patience=1, batch_periods=8)
        train_rows, valid_rows = model._create_splits(self.X, self.y, 0.7)[0]
        train_dates = set(self.X.index.get_level_values(1)[train_rows])
        valid_dates = set(self.X.index.get_level_values(1)[valid_rows])
        self.assertEqual(train_dates & valid_dates, set())
        self.assertLess(max(train_dates), min(valid_dates))

    def test_valid_scaler_fitted_on_training_split_only(self):
        model = PanelMLPRegressor(
            n_latent=8, epochs=2, patience=1, batch_periods=16
        ).fit(self.X, self.y)

        from sklearn.preprocessing import StandardScaler

        train_rows, _ = model._create_splits(self.X, self.y, 0.7)[0]
        reference = StandardScaler().fit(self.X.iloc[train_rows])
        np.testing.assert_allclose(model.scalers_[0].mean_, reference.mean_)
        np.testing.assert_allclose(model.scalers_[0].scale_, reference.scale_)

    def test_valid_forward_grid(self):
        """A batch is scattered into a (period, security) matrix with holes as NaN."""
        model = PanelMLPRegressor(n_latent=4, epochs=1, patience=1, batch_periods=4)
        model.signal_modifier = None

        network = torch.nn.Module()
        network.embedding = None
        network.forward = lambda x, codes=None: x[:, :1] * 0 + torch.arange(
            x.shape[0], dtype=torch.float32
        ).reshape(-1, 1)

        # Three periods x two securities, with one cell absent
        X_i = torch.zeros(5, 2)
        y_i = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        period_i = torch.tensor([0, 0, 1, 1, 2])
        asset_i = torch.tensor([0, 1, 0, 1, 0])

        grid_pred, grid_true = model._forward_grid(network, (X_i, y_i, period_i, asset_i))
        self.assertEqual(tuple(grid_pred.shape), (3, 2))
        self.assertEqual(tuple(grid_true.shape), (3, 2))

        # The absent cell carries a NaN target and a zero prediction
        self.assertTrue(bool(torch.isnan(grid_true[2, 1])))
        self.assertEqual(float(grid_pred[2, 1]), 0.0)
        np.testing.assert_allclose(
            grid_true[:2].numpy(), np.array([[1.0, 2.0], [3.0, 4.0]])
        )


if __name__ == "__main__":
    unittest.main()
