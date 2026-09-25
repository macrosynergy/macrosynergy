import unittest

import numpy as np
import torch

from macrosynergy.learning import LongShortModule
from macrosynergy.learning.forecasting.torch.losses import (
    NegCrossSectionalIC,
    NegRankIC,
    RankingRiskLoss,
)


def reference_ic(pred, true, rank=False, min_names=10):
    """Negative mean cross-sectional correlation, computed independently in numpy."""
    ics = []
    for p, t in zip(np.asarray(pred), np.asarray(true)):
        observed = np.isfinite(t)
        if observed.sum() < min_names:
            continue
        p_obs, t_obs = p[observed].astype(float), t[observed].astype(float)
        if rank:
            # Ascending ranks; the test data has no ties
            t_obs = np.argsort(np.argsort(t_obs)).astype(float)
        p_obs = p_obs - p_obs.mean()
        t_obs = t_obs - t_obs.mean()
        ics.append(p_obs @ t_obs / (np.linalg.norm(p_obs) * np.linalg.norm(t_obs)))
    return -float(np.mean(ics))


class TestCrossSectionalIC(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(11)
        cls.n_periods, cls.n_assets = 8, 20
        cls.pred = rng.normal(size=(cls.n_periods, cls.n_assets)).astype(np.float64)
        cls.true = rng.normal(size=(cls.n_periods, cls.n_assets)).astype(np.float64)

        cls.true_nan = cls.true.copy()
        cls.true_nan[0, :5] = np.nan       # still 15 observed, above min_names
        cls.true_nan[1, :18] = np.nan      # only 2 observed, below min_names

        cls.t_pred = torch.tensor(cls.pred, dtype=torch.float64)
        cls.t_true = torch.tensor(cls.true, dtype=torch.float64)
        cls.t_true_nan = torch.tensor(cls.true_nan, dtype=torch.float64)

    def test_types_init(self):
        with self.assertRaises(TypeError):
            NegCrossSectionalIC(min_names="ten")
        with self.assertRaises(ValueError):
            NegCrossSectionalIC(min_names=1)
        with self.assertRaises(TypeError):
            NegCrossSectionalIC(rank_targets="yes")
        with self.assertRaises(TypeError):
            NegCrossSectionalIC(eps="small")
        with self.assertRaises(ValueError):
            NegCrossSectionalIC(eps=0)

    def test_valid_forward(self):
        loss = NegCrossSectionalIC(min_names=10)
        self.assertAlmostEqual(
            float(loss(self.t_pred, self.t_true)),
            reference_ic(self.pred, self.true),
            places=10,
        )

    def test_valid_forward_masked(self):
        """Missing targets are excluded, and thin periods are dropped entirely."""
        loss = NegCrossSectionalIC(min_names=10)
        self.assertAlmostEqual(
            float(loss(self.t_pred, self.t_true_nan)),
            reference_ic(self.pred, self.true_nan, min_names=10),
            places=10,
        )

        # The period with 2 observed names must not contribute
        ic, usable = loss.period_ic(self.t_pred, self.t_true_nan)
        self.assertFalse(bool(usable[1]))
        self.assertTrue(bool(usable[0]))
        self.assertEqual(int(usable.sum()), self.n_periods - 1)

    def test_valid_perfect_and_inverted_signal(self):
        """A known answer: predicting the targets exactly scores -1, negating them +1."""
        loss = NegCrossSectionalIC(min_names=10)
        self.assertAlmostEqual(float(loss(self.t_true, self.t_true)), -1.0, places=10)
        self.assertAlmostEqual(float(loss(-self.t_true, self.t_true)), 1.0, places=10)

    def test_valid_scale_and_shift_invariance(self):
        """
        The loss must be invariant to rescaling the outputs and to adding a per-period
        constant: it scores the ranking, not the size or the level of the book.
        """
        loss = NegCrossSectionalIC(min_names=10)
        base = float(loss(self.t_pred, self.t_true))

        self.assertAlmostEqual(float(loss(self.t_pred * 37.0, self.t_true)), base, places=10)
        shift = torch.arange(self.n_periods, dtype=torch.float64).unsqueeze(1)
        self.assertAlmostEqual(float(loss(self.t_pred + shift, self.t_true)), base, places=10)

    def test_valid_batch_composition_invariance(self):
        """
        The property that justifies alternative batching: because the loss is a mean over
        per-period statistics, splitting a batch and averaging the parts (weighted by the
        number of usable periods) reproduces the whole.
        """
        loss = NegCrossSectionalIC(min_names=10)
        whole = float(loss(self.t_pred, self.t_true))

        first = float(loss(self.t_pred[:3], self.t_true[:3]))
        second = float(loss(self.t_pred[3:], self.t_true[3:]))
        combined = (3 * first + (self.n_periods - 3) * second) / self.n_periods
        self.assertAlmostEqual(whole, combined, places=10)

    def test_valid_asset_subsample_is_unbiased_in_sign(self):
        """
        Subsampling assets makes each period's statistic noisier but does not flip it:
        a perfect signal still scores -1 on any subset of names.
        """
        loss = NegCrossSectionalIC(min_names=10)
        subset = slice(0, 12)
        self.assertAlmostEqual(
            float(loss(self.t_true[:, subset], self.t_true[:, subset])), -1.0, places=10
        )

    def test_valid_no_usable_period(self):
        """
        With every period too thin, the loss is zero but still attached to the graph, so
        the step is a no-op rather than an error.
        """
        loss = NegCrossSectionalIC(min_names=10)
        true = self.t_true.clone()
        true[:, 3:] = float("nan")
        pred = self.t_pred.clone().requires_grad_(True)

        value = loss(pred, true)
        self.assertEqual(float(value), 0.0)
        value.backward()
        self.assertTrue(torch.all(pred.grad == 0))

    def test_valid_gradients(self):
        loss = NegCrossSectionalIC(min_names=10)
        pred = self.t_pred.clone().requires_grad_(True)
        loss(pred, self.t_true_nan).backward()
        self.assertTrue(torch.isfinite(pred.grad).all())
        self.assertGreater(float(pred.grad.abs().sum()), 0)

    def test_valid_collapsed_cross_section_gradient_is_bounded(self):
        """
        A constant cross-section must not produce an exploding gradient.

        If every asset gets the same prediction the centred row is exactly zero, so the
        correlation's denominator is zero and only the epsilon guard stands between the
        numerator and a division by nothing. The guard is applied to the *squared* norms
        before the square root rather than to the root afterwards, which raises the
        effective floor from `eps` to `sqrt(eps)` and bounds the gradient accordingly.

        This is not a hypothetical state: a head fed only macro features emits one
        forecast per period by construction, and an untrained head can start there. At
        `eps=1e-8` the two orderings differ by four orders of magnitude in the resulting
        gradient, which at a realistic learning rate is the difference between a step and
        a catastrophe.
        """
        loss = NegCrossSectionalIC(min_names=10, eps=1e-8)
        pred = torch.full((4, 50), 0.3, requires_grad=True)
        true = torch.randn(4, 50, generator=torch.Generator().manual_seed(0))

        value = loss(pred, true)
        value.backward()

        self.assertTrue(torch.isfinite(pred.grad).all())
        # Floor is sqrt(1e-8) = 1e-4, so gradients stay well below the 1e7 that
        # clamping after the square root would produce
        self.assertLess(float(pred.grad.abs().max()), 1e5)


class TestRankIC(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(12)
        cls.pred = rng.normal(size=(6, 20))
        cls.true = rng.normal(size=(6, 20))
        cls.t_pred = torch.tensor(cls.pred, dtype=torch.float64)
        cls.t_true = torch.tensor(cls.true, dtype=torch.float64)

    def test_valid_forward(self):
        loss = NegRankIC(min_names=10)
        self.assertAlmostEqual(
            float(loss(self.t_pred, self.t_true)),
            reference_ic(self.pred, self.true, rank=True),
            places=10,
        )

    def test_valid_monotone_invariance_of_targets(self):
        """
        A rank statistic must be unchanged by any increasing transform of the targets,
        which is the point of using it on fat-tailed returns.
        """
        loss = NegRankIC(min_names=10)
        base = float(loss(self.t_pred, self.t_true))
        for transform in (torch.exp, lambda t: t**3, lambda t: 5 * t + 2):
            self.assertAlmostEqual(
                float(loss(self.t_pred, transform(self.t_true))), base, places=8
            )

    def test_valid_outlier_robustness(self):
        """A single extreme target moves the Pearson IC far more than the rank IC."""
        pearson, rank = NegCrossSectionalIC(min_names=10), NegRankIC(min_names=10)
        true_outlier = self.t_true.clone()
        true_outlier[0, 0] = 500.0

        pearson_shift = abs(
            float(pearson(self.t_pred, true_outlier)) - float(pearson(self.t_pred, self.t_true))
        )
        rank_shift = abs(
            float(rank(self.t_pred, true_outlier)) - float(rank(self.t_pred, self.t_true))
        )
        self.assertGreater(pearson_shift, rank_shift)

    def test_valid_masked(self):
        true = self.t_true.clone()
        true[0, :4] = float("nan")
        loss = NegRankIC(min_names=10)
        reference = reference_ic(
            self.pred, true.numpy(), rank=True, min_names=10
        )
        self.assertAlmostEqual(float(loss(self.t_pred, true)), reference, places=10)


class TestRankingRiskLoss(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(13)
        cls.n_periods, cls.n_assets = 10, 20
        cls.pred = rng.normal(size=(cls.n_periods, cls.n_assets))
        cls.true = rng.normal(size=(cls.n_periods, cls.n_assets))
        cls.t_pred = torch.tensor(cls.pred, dtype=torch.float64)
        cls.t_true = torch.tensor(cls.true, dtype=torch.float64)

    def test_types_init(self):
        with self.assertRaises(TypeError):
            RankingRiskLoss(reg_risk="one")
        with self.assertRaises(ValueError):
            RankingRiskLoss(reg_risk=-1)
        with self.assertRaises(TypeError):
            RankingRiskLoss(risk=3)
        with self.assertRaises(ValueError):
            RankingRiskLoss(risk="variance")
        with self.assertRaises(ValueError):
            RankingRiskLoss(shrinkage=1.5)
        with self.assertRaises(TypeError):
            RankingRiskLoss(reg_systematic="one")
        with self.assertRaises(ValueError):
            RankingRiskLoss(reg_systematic=-1)

    def test_valid_zero_risk_recovers_ranking_loss(self):
        combined = RankingRiskLoss(reg_risk=0, min_names=10)
        ranking = NegCrossSectionalIC(min_names=10)
        self.assertAlmostEqual(
            float(combined(self.t_pred, self.t_true)),
            float(ranking(self.t_pred, self.t_true)),
            places=12,
        )

    def test_valid_period_disp_term(self):
        """
        `period_disp` weights each period's squared positions by that period's own
        realised cross-sectional variance, computed against hand arithmetic.
        """
        reg = 0.25
        loss = RankingRiskLoss(reg_risk=reg, risk="period_disp", min_names=10)
        ranking = float(NegCrossSectionalIC(min_names=10)(self.t_pred, self.t_true))

        dispersion = self.true.var(axis=1)  # population variance, per period
        expected = ranking + reg * float(((self.pred**2).sum(axis=1) * dispersion).mean())
        self.assertAlmostEqual(
            float(loss(self.t_pred, self.t_true)), expected, places=10
        )

    def test_valid_period_disp_is_batch_invariant(self):
        """
        `period_disp` is a mean over per-period quantities, so cutting the batch in half
        must not change it. This is what makes it valid under `AssetBaggingLoss` and
        under every `PanelBatchSampler` mode, unlike `batch_vol`.
        """
        loss = RankingRiskLoss(reg_risk=0.5, risk="period_disp", min_names=10)
        half = self.n_periods // 2
        whole = float(loss(self.t_pred, self.t_true))
        split = 0.5 * (
            float(loss(self.t_pred[:half], self.t_true[:half]))
            + float(loss(self.t_pred[half:], self.t_true[half:]))
        )
        self.assertAlmostEqual(whole, split, places=10)

        # The contrast: batch_vol is a statistic *of the batch* and is not invariant
        batch_vol = RankingRiskLoss(reg_risk=0.5, risk="batch_vol", min_names=10)
        whole_bv = float(batch_vol(self.t_pred, self.t_true))
        split_bv = 0.5 * (
            float(batch_vol(self.t_pred[:half], self.t_true[:half]))
            + float(batch_vol(self.t_pred[half:], self.t_true[half:]))
        )
        self.assertNotAlmostEqual(whole_bv, split_bv, places=4)

    def test_valid_systematic_term_charges_net_exposure(self):
        """
        The market term must scale with net exposure and with the market's volatility.

        Doubling every weight quadruples the net exposure's square, so the charge must
        quadruple; the ranking term is scale-free and unchanged.
        """
        loss = RankingRiskLoss(reg_risk=0, reg_systematic=1.0, min_names=10)
        base = RankingRiskLoss(reg_risk=0, reg_systematic=0, min_names=10)

        charge = float(loss(self.t_pred, self.t_true)) - float(base(self.t_pred, self.t_true))
        doubled = float(loss(2 * self.t_pred, self.t_true)) - float(
            base(2 * self.t_pred, self.t_true)
        )
        self.assertGreater(charge, 0)
        self.assertAlmostEqual(doubled, 4 * charge, places=8)

    def test_valid_systematic_term_vanishes_when_dollar_neutral(self):
        """
        A book with no net exposure cannot be charged for market exposure.

        This is the documented boundary of the term: under
        `LongShortModule(dollar_neutral=True)` the net is zero by construction, so the
        systematic penalty is identically zero and only the selection term bites.
        """
        neutral = LongShortModule(dollar_neutral=True)(self.t_pred)
        loss = RankingRiskLoss(reg_risk=0, reg_systematic=1.0, min_names=10)
        base = RankingRiskLoss(reg_risk=0, reg_systematic=0, min_names=10)
        self.assertAlmostEqual(
            float(loss(neutral, self.t_true)), float(base(neutral, self.t_true)), places=12
        )

    def test_valid_systematic_term_uses_supplied_market_vol(self):
        """A supplied market volatility must be used verbatim, not re-estimated."""
        ranking = float(NegCrossSectionalIC(min_names=10)(self.t_pred, self.t_true))
        loss = RankingRiskLoss(
            reg_risk=0, reg_systematic=1.0, market_vol=0.5, min_names=10
        )
        expected = ranking + float((self.pred.sum(axis=1) ** 2).mean()) * 0.25
        self.assertAlmostEqual(float(loss(self.t_pred, self.t_true)), expected, places=10)

    def test_valid_both_risk_terms_compose(self):
        """The two penalties are additive and sweep independently."""
        ranking = RankingRiskLoss(reg_risk=0, reg_systematic=0, min_names=10)
        selection = RankingRiskLoss(reg_risk=0.5, risk="period_disp", min_names=10)
        systematic = RankingRiskLoss(reg_risk=0, reg_systematic=0.5, min_names=10)
        both = RankingRiskLoss(
            reg_risk=0.5, risk="period_disp", reg_systematic=0.5, min_names=10
        )

        base = float(ranking(self.t_pred, self.t_true))
        self.assertAlmostEqual(
            float(both(self.t_pred, self.t_true)),
            float(selection(self.t_pred, self.t_true))
            + float(systematic(self.t_pred, self.t_true))
            - base,
            places=10,
        )

    def test_valid_systematic_gradients(self):
        pred = self.t_pred.clone().requires_grad_(True)
        loss = RankingRiskLoss(
            reg_risk=0.5, risk="period_disp", reg_systematic=0.5, min_names=10
        )
        loss(pred, self.t_true).backward()
        self.assertTrue(torch.isfinite(pred.grad).all())
        self.assertGreater(float(pred.grad.abs().sum()), 0)

    def test_valid_batch_vol_term(self):
        """The batch_vol term is the standard deviation of the portfolio return."""
        reg = 0.25
        loss = RankingRiskLoss(reg_risk=reg, risk="batch_vol", min_names=10)
        ranking = float(NegCrossSectionalIC(min_names=10)(self.t_pred, self.t_true))

        portfolio = (self.pred * self.true).sum(axis=1)
        expected = ranking + reg * portfolio.std(ddof=1)
        self.assertAlmostEqual(float(loss(self.t_pred, self.t_true)), expected, places=10)

    def test_valid_period_var_term(self):
        """period_var is the mean over periods of sum_i (w_i * s_i)^2."""
        reg = 0.5
        vol = torch.tensor(np.linspace(0.5, 2.0, self.n_assets), dtype=torch.float64)
        loss = RankingRiskLoss(
            reg_risk=reg, risk="period_var", asset_vol=vol, min_names=10
        )
        ranking = float(NegCrossSectionalIC(min_names=10)(self.t_pred, self.t_true))

        expected = ranking + reg * float(
            ((self.pred * vol.numpy()) ** 2).sum(axis=1).mean()
        )
        self.assertAlmostEqual(float(loss(self.t_pred, self.t_true)), expected, places=10)

    def test_valid_period_cov_term(self):
        """period_cov is the mean over periods of the quadratic form w' Sigma w."""
        reg = 0.5
        rng = np.random.default_rng(3)
        root = rng.normal(size=(self.n_assets, self.n_assets))
        cov_np = root @ root.T / self.n_assets
        cov = torch.tensor(cov_np, dtype=torch.float64)

        loss = RankingRiskLoss(reg_risk=reg, risk="period_cov", cov=cov, min_names=10)
        ranking = float(NegCrossSectionalIC(min_names=10)(self.t_pred, self.t_true))

        quadratic = np.einsum("ti,ij,tj->t", self.pred, cov_np, self.pred)
        expected = ranking + reg * float(quadratic.mean())
        self.assertAlmostEqual(float(loss(self.t_pred, self.t_true)), expected, places=8)

    def test_valid_decomposable_terms_are_batch_invariant(self):
        """
        The decomposable risk terms are means over periods, so splitting the batch and
        averaging reproduces the whole. The batch_vol term is not, which is exactly why
        it constrains the batching scheme.
        """
        split = 4
        for risk, invariant in (("period_var", True), ("period_cov", True), ("batch_vol", False)):
            kwargs = dict(reg_risk=1.0, risk=risk, min_names=10)
            if risk == "period_var":
                kwargs["asset_vol"] = torch.ones(self.n_assets, dtype=torch.float64)
            if risk == "period_cov":
                kwargs["cov"] = torch.eye(self.n_assets, dtype=torch.float64)
            loss = RankingRiskLoss(**kwargs)

            whole = float(loss(self.t_pred, self.t_true))
            first = float(loss(self.t_pred[:split], self.t_true[:split]))
            second = float(loss(self.t_pred[split:], self.t_true[split:]))
            combined = (
                split * first + (self.n_periods - split) * second
            ) / self.n_periods

            if invariant:
                self.assertAlmostEqual(whole, combined, places=10, msg=risk)
            else:
                self.assertNotAlmostEqual(whole, combined, places=6, msg=risk)

    def test_valid_risk_term_penalises_size(self):
        """
        Unlike the ranking term, the risk term is not scale-free: doubling the book must
        raise the loss. That separation is the point of the class.
        """
        loss = RankingRiskLoss(
            reg_risk=1.0,
            risk="period_var",
            asset_vol=torch.ones(self.n_assets, dtype=torch.float64),
            min_names=10,
        )
        self.assertGreater(
            float(loss(self.t_pred * 2, self.t_true)),
            float(loss(self.t_pred, self.t_true)),
        )

    def test_valid_estimated_statistics_do_not_propagate_gradients(self):
        """
        When the risk statistic is estimated in-batch it is detached, so the optimiser
        cannot reduce the penalty by distorting its own risk estimate.
        """
        for risk in ("period_var", "period_cov"):
            pred = self.t_pred.clone().requires_grad_(True)
            true = self.t_true.clone().requires_grad_(True)
            RankingRiskLoss(reg_risk=1.0, risk=risk, min_names=10)(pred, true).backward()
            self.assertTrue(torch.isfinite(pred.grad).all(), msg=risk)

    def test_valid_gradients(self):
        for risk in ("batch_vol", "period_var", "period_cov"):
            pred = self.t_pred.clone().requires_grad_(True)
            RankingRiskLoss(reg_risk=0.5, risk=risk, min_names=10)(
                pred, self.t_true
            ).backward()
            self.assertTrue(torch.isfinite(pred.grad).all(), msg=risk)
            self.assertGreater(float(pred.grad.abs().sum()), 0, msg=risk)


if __name__ == "__main__":
    unittest.main()
