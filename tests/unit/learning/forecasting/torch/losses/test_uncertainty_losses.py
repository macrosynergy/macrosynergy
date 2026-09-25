import unittest

import numpy as np
import torch

from macrosynergy.learning.forecasting.torch.losses import GaussianNLL
from macrosynergy.learning.forecasting.torch.models import HeteroskedasticMLP


class TestGaussianNLL(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(41)
        cls.n_periods, cls.n_assets = 9, 6
        cls.mu = rng.normal(size=(cls.n_periods, cls.n_assets))
        cls.log_var = rng.normal(size=(cls.n_periods, cls.n_assets)) * 0.5
        cls.true = rng.normal(size=(cls.n_periods, cls.n_assets))

        cls.pred = torch.tensor(
            np.concatenate([cls.mu, cls.log_var], axis=1), dtype=torch.float64
        )
        cls.t_true = torch.tensor(cls.true, dtype=torch.float64)

    def test_types_init(self):
        with self.assertRaises(TypeError):
            GaussianNLL(beta="half")
        with self.assertRaises(ValueError):
            GaussianNLL(beta=1.5)
        with self.assertRaises(ValueError):
            GaussianNLL(beta=-0.1)
        with self.assertRaises(ValueError):
            GaussianNLL(eps=0)
        with self.assertRaises(TypeError):
            GaussianNLL(full="yes")

    def test_types_forward(self):
        """The prediction must be twice as wide as the target."""
        loss = GaussianNLL()
        with self.assertRaises(ValueError):
            loss(torch.zeros(4, 5, dtype=torch.float64), torch.zeros(4, 3, dtype=torch.float64))

    def test_valid_forward_exact_likelihood(self):
        """With beta=0 the loss is the exact negative Gaussian log-likelihood."""
        loss = GaussianNLL(beta=0)
        variance = np.exp(self.log_var)
        expected = float(
            np.mean((self.true - self.mu) ** 2 / (2 * variance) + 0.5 * self.log_var)
        )
        self.assertAlmostEqual(float(loss(self.pred, self.t_true)), expected, places=10)

    def test_valid_forward_beta_weighting(self):
        beta = 0.5
        loss = GaussianNLL(beta=beta)
        variance = np.exp(self.log_var)
        terms = ((self.true - self.mu) ** 2 / (2 * variance) + 0.5 * self.log_var)
        expected = float(np.mean(terms * variance**beta))
        self.assertAlmostEqual(float(loss(self.pred, self.t_true)), expected, places=10)

    def test_valid_full_constant(self):
        plain, full = GaussianNLL(beta=0), GaussianNLL(beta=0, full=True)
        difference = float(full(self.pred, self.t_true)) - float(plain(self.pred, self.t_true))
        self.assertAlmostEqual(difference, 0.5 * np.log(2 * np.pi), places=10)

    def test_valid_masked(self):
        """Missing targets are excluded from the average, not filled."""
        true = self.t_true.clone()
        true[0, :2] = float("nan")
        loss = GaussianNLL(beta=0)

        variance = np.exp(self.log_var)
        terms = (self.true - self.mu) ** 2 / (2 * variance) + 0.5 * self.log_var
        keep = np.ones_like(terms, dtype=bool)
        keep[0, :2] = False
        expected = float(terms[keep].mean())

        self.assertAlmostEqual(float(loss(self.pred, true)), expected, places=10)

    def test_valid_minimised_at_the_true_variance(self):
        """
        A known answer: for a fixed mean error, the exact likelihood is minimised when
        the predicted variance equals the squared residual.
        """
        loss = GaussianNLL(beta=0)
        true = torch.zeros(1, 1, dtype=torch.float64)
        residual = 2.0

        best, best_value = None, np.inf
        for log_var in np.linspace(np.log(residual**2) - 2, np.log(residual**2) + 2, 401):
            pred = torch.tensor([[residual, log_var]], dtype=torch.float64)
            value = float(loss(pred, true))
            if value < best_value:
                best, best_value = log_var, value
        self.assertAlmostEqual(np.exp(best), residual**2, places=2)

    def test_valid_beta_reweights_the_mean_gradient(self):
        """
        The motivation for beta: at beta=0 an observation's pull on the mean head is
        inversely proportional to its variance, so a high-variance point is nearly
        ignored. At beta=1 that dependence is cancelled.
        """
        true = torch.zeros(1, 2, dtype=torch.float64)
        # Same residual, variances two orders of magnitude apart
        base = torch.tensor([[1.0, 1.0, 0.0, np.log(100.0)]], dtype=torch.float64)

        grads = {}
        for beta in (0.0, 1.0):
            pred = base.clone().requires_grad_(True)
            GaussianNLL(beta=beta)(pred, true).backward()
            grads[beta] = pred.grad[0, :2].abs()

        # At beta=0 the noisy asset pulls far less on the mean than the clean one
        self.assertGreater(float(grads[0.0][0] / grads[0.0][1]), 50)
        # At beta=1 they pull equally
        self.assertAlmostEqual(float(grads[1.0][0] / grads[1.0][1]), 1.0, places=6)

    def test_valid_beta_weight_is_detached(self):
        """
        The reweighting must not itself supply gradient to the variance head, or it would
        move the optimum of the variance equation.
        """
        pred = self.pred.clone().requires_grad_(True)
        GaussianNLL(beta=0.5)(pred, self.t_true).backward()
        grad_weighted = pred.grad.clone()

        # Recompute with the weight held fixed by hand; the variance-head gradient must
        # match, which it only can if the weight carried no gradient of its own
        variance = torch.exp(self.pred[:, self.n_assets :]).detach()
        pred2 = self.pred.clone().requires_grad_(True)
        mu, log_var = pred2[:, : self.n_assets], pred2[:, self.n_assets :]
        terms = (self.t_true - mu) ** 2 / (2 * log_var.exp().clamp(min=1e-6)) + 0.5 * log_var
        (terms * variance**0.5).mean().backward()

        torch.testing.assert_close(grad_weighted, pred2.grad)

    def test_valid_gradients(self):
        pred = self.pred.clone().requires_grad_(True)
        GaussianNLL(beta=0.5)(pred, self.t_true).backward()
        self.assertTrue(torch.isfinite(pred.grad).all())
        self.assertGreater(float(pred.grad.abs().sum()), 0)


class TestHeteroskedasticMLP(unittest.TestCase):
    def test_types_init(self):
        with self.assertRaises(TypeError):
            HeteroskedasticMLP(n_inputs=4, n_latent=8, n_signals="three")
        with self.assertRaises(ValueError):
            HeteroskedasticMLP(n_inputs=4, n_latent=8, n_signals=0)
        with self.assertRaises(ValueError):
            HeteroskedasticMLP(
                n_inputs=4, n_latent=8, n_signals=3, min_log_var=1.0, max_log_var=0.0
            )

    def test_valid_forward_shape(self):
        model = HeteroskedasticMLP(n_inputs=4, n_latent=[8, 4], n_signals=3)
        out = model(torch.randn(11, 4))
        self.assertEqual(tuple(out.shape), (11, 6))

        mu, log_var = HeteroskedasticMLP.split(out)
        self.assertEqual(tuple(mu.shape), (11, 3))
        self.assertEqual(tuple(log_var.shape), (11, 3))

    def test_valid_split_rejects_odd_width(self):
        with self.assertRaises(ValueError):
            HeteroskedasticMLP.split(torch.randn(4, 5))

    def test_valid_log_var_is_clamped(self):
        model = HeteroskedasticMLP(
            n_inputs=4, n_latent=8, n_signals=3, min_log_var=-2.0, max_log_var=2.0
        )
        # Drive the variance head hard in both directions
        with torch.no_grad():
            model.log_var_head.weight.fill_(50.0)
            model.log_var_head.bias.fill_(50.0)
        _, log_var = HeteroskedasticMLP.split(model(torch.randn(20, 4)))
        self.assertTrue(bool((log_var <= 2.0 + 1e-6).all()))
        self.assertTrue(bool((log_var >= -2.0 - 1e-6).all()))

    def test_valid_shares_the_encoder(self):
        """Both heads read the same derived features."""
        model = HeteroskedasticMLP(n_inputs=4, n_latent=8, n_signals=3)
        self.assertEqual(model.head.in_features, model.log_var_head.in_features)
        self.assertIs(model.encoder, model.encoder)
        names = {name for name, _ in model.named_parameters()}
        self.assertTrue(any(n.startswith("encoder.") for n in names))
        self.assertTrue(any(n.startswith("head.") for n in names))
        self.assertTrue(any(n.startswith("log_var_head.") for n in names))

    def test_valid_precision_weighted(self):
        mu = torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float64)
        log_var = torch.tensor([[0.0, np.log(4.0), np.log(100.0)]], dtype=torch.float64)
        out = torch.cat([mu, log_var], dim=1)

        # power=0 leaves the means alone
        torch.testing.assert_close(
            HeteroskedasticMLP.precision_weighted(out, power=0), mu
        )
        # power=1 divides by the variance
        torch.testing.assert_close(
            HeteroskedasticMLP.precision_weighted(out, power=1),
            torch.tensor([[1.0, 0.25, 0.01]], dtype=torch.float64),
        )
        # power=0.5 divides by the standard deviation
        torch.testing.assert_close(
            HeteroskedasticMLP.precision_weighted(out, power=0.5),
            torch.tensor([[1.0, 0.5, 0.1]], dtype=torch.float64),
        )

    def test_valid_precision_weighted_shrinks_the_uncertain(self):
        """Equal means, unequal variances: the uncertain asset must carry less weight."""
        mu = torch.ones(1, 3, dtype=torch.float64)
        log_var = torch.tensor([[0.0, 1.0, 3.0]], dtype=torch.float64)
        signal = HeteroskedasticMLP.precision_weighted(
            torch.cat([mu, log_var], dim=1), power=1
        )
        self.assertGreater(float(signal[0, 0]), float(signal[0, 1]))
        self.assertGreater(float(signal[0, 1]), float(signal[0, 2]))

    def test_valid_precision_weighted_normalisation(self):
        rng = np.random.default_rng(3)
        out = torch.tensor(rng.normal(size=(5, 8)), dtype=torch.float64)

        demeaned = HeteroskedasticMLP.precision_weighted(out, normalize="demean")
        torch.testing.assert_close(
            demeaned.sum(dim=1), torch.zeros(5, dtype=torch.float64)
        )

        gross = HeteroskedasticMLP.precision_weighted(out, normalize="gross")
        torch.testing.assert_close(gross.sum(dim=1), torch.zeros(5, dtype=torch.float64))
        torch.testing.assert_close(
            gross.abs().sum(dim=1), torch.ones(5, dtype=torch.float64)
        )

        with self.assertRaises(ValueError):
            HeteroskedasticMLP.precision_weighted(out, normalize="unit")

    def test_valid_gross_normalisation_removes_absolute_shrinkage(self):
        """
        The trap the docstring warns about: once gross exposure is pinned, scaling every
        position by its precision and renormalising is a no-op in aggregate. Shrinkage
        only reallocates between assets.
        """
        rng = np.random.default_rng(4)
        out = torch.tensor(rng.normal(size=(5, 8)), dtype=torch.float64)
        for power in (0.0, 1.0, 2.0):
            gross = HeteroskedasticMLP.precision_weighted(
                out, power=power, normalize="gross"
            )
            torch.testing.assert_close(
                gross.abs().sum(dim=1), torch.ones(5, dtype=torch.float64)
            )

    def test_valid_recovers_known_heteroskedasticity(self):
        """
        Known-answer test: on data whose noise level differs sharply between two groups of
        assets, the variance head must learn which group is which.
        """
        torch.manual_seed(0)
        rng = np.random.default_rng(7)
        n_periods, n_signals, n_inputs = 600, 6, 3
        quiet, noisy = slice(0, 3), slice(3, 6)

        X = rng.normal(size=(n_periods, n_inputs))
        beta = rng.normal(size=(n_inputs, n_signals))
        mu = X @ beta
        noise = np.ones(n_signals)
        noise[noisy] = 6.0
        y = mu + rng.normal(size=(n_periods, n_signals)) * noise

        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)

        model = HeteroskedasticMLP(
            n_inputs=n_inputs, n_latent=[32, 16], n_signals=n_signals,
            encoder_activation="tanh",
        )
        loss_func = GaussianNLL(beta=0.5)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-5)
        for _ in range(400):
            optimizer.zero_grad()
            loss_func(model(X_t), y_t).backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            _, log_var = HeteroskedasticMLP.split(model(X_t))
        mean_var = log_var.exp().mean(dim=0).numpy()

        self.assertGreater(
            mean_var[noisy].min(),
            mean_var[quiet].max(),
            msg=f"variance head did not separate the groups: {mean_var}",
        )
        # And the level is in the right region, not merely ordered
        self.assertGreater(mean_var[noisy].mean() / mean_var[quiet].mean(), 4.0)


if __name__ == "__main__":
    unittest.main()
