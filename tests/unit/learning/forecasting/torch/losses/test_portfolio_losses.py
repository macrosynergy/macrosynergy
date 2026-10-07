import torch 
import torch.nn as nn 

from sklearn.base import BaseEstimator

from macrosynergy.learning.forecasting.torch.losses import (
    NegSharpeRatio,
    NegSharpeRatioExAnteVol,
    PortfolioVariance,
    NegMeanPortfolioReturn,
    NegMeanVarianceUtility,
    NegMeanVarianceExAnteVol,
)

import unittest 

from parameterized import parameterized

import itertools

portfolio_losses = [
    NegSharpeRatio,
    PortfolioVariance,
    NegMeanPortfolioReturn,
    NegMeanVarianceUtility,
]
loss_names = [loss.__name__ for loss in portfolio_losses]

class TestPortfolioLosses(unittest.TestCase):
    @classmethod 
    def setUpClass(cls):
        cls.basic_losses = {
            loss_names[0]: portfolio_losses[0],
            loss_names[1]: portfolio_losses[1],
            loss_names[2]: portfolio_losses[2],
        }

    def test_types_init(self):
        """ Test types of constructor parameters for each loss function """
        for loss_name, loss in self.basic_losses.items():
            # Test that reg_concentration must be a positive number
            self.assertRaises(TypeError, loss, reg_concentration = "invalid_string")
            self.assertRaises(ValueError, loss, reg_concentration = -1)
            # Test that skip_validation must be a boolean
            self.assertRaises(TypeError, loss, skip_validation = "invalid_string")

            if loss_name == "NegMeanVarianceUtility":
                # Test that alpha must be a positive number
                self.assertRaises(TypeError, loss, reg_concentration = 0, alpha = "invalid_string")
                self.assertRaises(ValueError, loss, reg_concentration = 0, alpha = -1)
                self.assertRaises(TypeError, loss, reg_concentration = 1, alpha = "invalid_string")
                self.assertRaises(ValueError, loss, reg_concentration = 1, alpha = -1)

    def test_valid_init(self):
        """ Test valid initialization for each loss function """
        for loss_name, loss in self.basic_losses.items():
            # Each should be a subclass of nn.Module and BaseEstimator
            default_loss = loss()
            self.assertIsInstance(default_loss, nn.Module)
            self.assertIsInstance(default_loss, BaseEstimator)

            # Test defaults are set correctly
            self.assertEqual(default_loss.reg_concentration, 0)
            self.assertEqual(default_loss.skip_validation, True)
            if loss_name == "NegMeanVarianceUtility":
                self.assertEqual(default_loss.alpha, 1.0)

            # Test that reg_concentration is set correctly
            try: 
                instance = loss(reg_concentration = 0.1)
            except Exception as e:
                self.fail(f"{loss_name} raised {type(e)} unexpectedly!")
            self.assertEqual(instance.reg_concentration, 0.1)

            # Test that skip_validation is set correctly
            try: 
                instance = loss(reg_concentration = 0.1, skip_validation = False)
            except Exception as e:
                self.fail(f"{loss_name} raised {type(e)} unexpectedly!")
            self.assertEqual(instance.skip_validation, False)

            # Test that alpha is set correctly for NegMeanVarianceUtility
            if loss_name == "NegMeanVarianceUtility":
                try: 
                    instance = loss(reg_concentration = 0.7, alpha = 0.5)
                except Exception as e:
                    self.fail(f"{loss_name} raised {type(e)} unexpectedly!")
                self.assertEqual(instance.alpha, 0.5)
                self.assertEqual(instance.reg_concentration, 0.7)

    def test_types_forward(self):
        """ Test types of forward parameters for each loss function """
        for loss_name, loss in self.basic_losses.items():
            if loss_name == "NegMeanVarianceUtility":
                instance = loss(reg_concentration = 0.1, alpha = 0.5, skip_validation = False)
            else:
                instance = loss(reg_concentration = 0.1, skip_validation = False)

            # y_true should be a torch.Tensor with shape (batch_size, n_assets)
            self.assertRaises(
                TypeError,
                instance.forward,
                y_true = "invalid_string",
                y_pred = torch.randn(10, 5),
            )
            self.assertRaises(
                ValueError,
                instance.forward,
                y_true = torch.randn(10, 1),
                y_pred = torch.randn(10, 3),
            )

            # y_pred should be a torch.Tensor with shape (batch_size, n_assets)
            self.assertRaises(
                TypeError,
                instance.forward,
                y_true = torch.randn(10, 5),
                y_pred = "invalid_string",
            )
            self.assertRaises(
                ValueError,
                instance.forward,
                y_true = torch.randn(10, 3),
                y_pred = torch.randn(10, 1),
            )

    def test_valid_forward(self):
        """ Test valid forward pass for each loss function """
        for loss_name, loss in self.basic_losses.items():
            if loss_name == "NegMeanVarianceUtility":
                instance = loss(reg_concentration = 0.1, alpha = 0.5, skip_validation = True)
            else:
                instance = loss(reg_concentration = 0.1, skip_validation = True)

            try:
                y_true_sample = torch.randn(20, 5)
                y_pred_sample = torch.randn(20, 5)
                loss_value = instance(y_true = y_true_sample, y_pred = y_pred_sample)
            except Exception as e:
                self.fail(f"{loss_name} raised {type(e)} unexpectedly!")
            # The loss value should be a scalar tensor
            self.assertIsInstance(loss_value, torch.Tensor)
            self.assertEqual(loss_value.dim(), 0)

            # Check correctness of each loss 
            signal_returns = y_true_sample * y_pred_sample
            portfolio_returns = torch.sum(signal_returns, dim=1)
            if loss_name == "NegSharpeRatio":
                self.assertEqual(
                    loss_value,
                    -torch.mean(portfolio_returns) / (torch.std(portfolio_returns, unbiased = True) + 1e-8) + 0.1 * torch.mean(torch.sum(y_pred_sample ** 2, dim=1))
                )
            elif loss_name == "PortfolioVariance":
                self.assertEqual(
                    loss_value,
                    torch.var(portfolio_returns, unbiased = True) + 0.1 * torch.mean(torch.sum(y_pred_sample ** 2, dim=1))
                )
            elif loss_name == "NegMeanPortfolioReturn":
                self.assertEqual(
                    loss_value,
                    -torch.mean(portfolio_returns) + 0.1 * torch.mean(torch.sum(y_pred_sample ** 2, dim=1))
                )
            elif loss_name == "NegMeanVarianceUtility":
                alpha = 0.5
                self.assertEqual(
                    loss_value,
                    -torch.mean(portfolio_returns) + alpha * torch.var(portfolio_returns, unbiased = True) + 0.1 * torch.mean(torch.sum(y_pred_sample ** 2, dim=1))
                )

        # Check that meanportfolio returns, mean variance align 
        mean_portfolio_loss = NegMeanPortfolioReturn(reg_concentration = 0.1, skip_validation = True)
        mean_variance_loss = NegMeanVarianceUtility(reg_concentration = 0.1, alpha = 0, skip_validation = True)

        y_true_sample = torch.randn(20, 5) 
        y_pred_sample = torch.randn(20, 5)

        self.assertEqual(
            mean_portfolio_loss(y_true = y_true_sample, y_pred = y_pred_sample),
            mean_variance_loss(y_true = y_true_sample, y_pred = y_pred_sample),
        )


class TestNegSharpeRatioExAnteVol(unittest.TestCase):
    def test_requires_vol_flag(self):
        self.assertTrue(NegSharpeRatioExAnteVol.requires_vol)
        for loss in [NegSharpeRatio, PortfolioVariance, NegMeanPortfolioReturn, NegMeanVarianceUtility]:
            self.assertFalse(loss.requires_vol)

    def test_forward_requires_vol(self):
        instance = NegSharpeRatioExAnteVol()
        self.assertRaises(
            ValueError, instance.forward, y_pred = torch.randn(10, 5), y_true = torch.randn(10, 5),
        )

    def test_types_forward(self):
        instance = NegSharpeRatioExAnteVol(skip_validation = False)
        y_pred, y_true, vol = torch.randn(10, 5), torch.randn(10, 5), torch.rand(10, 5).abs() + 0.1
        # vol must be a tensor, same shape as y_true
        self.assertRaises(TypeError, instance.forward, y_pred = y_pred, y_true = y_true, vol = "invalid")
        self.assertRaises(
            ValueError, instance.forward, y_pred = y_pred, y_true = y_true, vol = torch.rand(10, 3),
        )

    def test_valid_forward_matches_manual_formula(self):
        instance = NegSharpeRatioExAnteVol(reg_concentration = 0.1, skip_validation = True)
        y_pred = torch.randn(20, 5)
        y_true = torch.randn(20, 5)
        vol = torch.rand(20, 5).abs() + 0.1

        loss_value = instance(y_pred = y_pred, y_true = y_true, vol = vol)

        portfolio_returns = torch.sum(y_pred * y_true, dim=1)
        mean_return = torch.mean(portfolio_returns)
        risk = torch.sqrt(torch.mean(torch.sum((y_pred ** 2) * (vol ** 2), dim=1)))
        expected = -mean_return / (risk + instance.eps) + 0.1 * torch.mean(torch.sum(y_pred ** 2, dim=1))

        self.assertTrue(torch.allclose(loss_value, expected))

    def test_masks_missing_return_and_missing_vol_alike(self):
        """A cell missing either y_true or vol contributes zero return and zero risk --
        the same way the base class's masking already treats a missing y_true alone."""
        instance = NegSharpeRatioExAnteVol(skip_validation = True)
        y_pred = torch.tensor([[1.0, 2.0, 3.0]])
        y_true = torch.tensor([[0.1, float("nan"), 0.3]])
        vol = torch.tensor([[0.2, 0.2, float("nan")]])

        loss_value = instance(y_pred = y_pred, y_true = y_true, vol = vol)

        # Only the first name (index 0) has both a return and a vol; the other two are
        # masked out entirely, so this should match a single-name manual calculation
        expected_return = 1.0 * 0.1
        expected_risk = (1.0 ** 2 * 0.2 ** 2) ** 0.5
        expected = -expected_return / (expected_risk + instance.eps)
        self.assertAlmostEqual(loss_value.item(), expected, places=5)

    def test_scale_invariance(self):
        """Uniformly rescaling every weight leaves the loss unchanged (both numerator and
        the risk term are homogeneous degree 1 in y_pred)."""
        instance = NegSharpeRatioExAnteVol(skip_validation = True)
        y_pred = torch.randn(20, 5)
        y_true = torch.randn(20, 5)
        vol = torch.rand(20, 5).abs() + 0.1

        base = instance(y_pred = y_pred, y_true = y_true, vol = vol)
        scaled = instance(y_pred = 3.0 * y_pred, y_true = y_true, vol = vol)
        self.assertAlmostEqual(base.item(), scaled.item(), places=5)


class TestNegMeanVarianceExAnteVol(unittest.TestCase):
    def test_requires_vol_flag(self):
        self.assertTrue(NegMeanVarianceExAnteVol.requires_vol)

    def test_forward_requires_vol(self):
        instance = NegMeanVarianceExAnteVol()
        self.assertRaises(
            ValueError, instance.forward, y_pred = torch.randn(10, 5), y_true = torch.randn(10, 5),
        )

    def test_types_init(self):
        for name in ("alpha", "beta", "gamma"):
            self.assertRaises(TypeError, NegMeanVarianceExAnteVol, **{name: "invalid_string"})
            self.assertRaises(ValueError, NegMeanVarianceExAnteVol, **{name: -1})

    def test_valid_init(self):
        instance = NegMeanVarianceExAnteVol()
        self.assertEqual(instance.alpha, 1.0)
        self.assertEqual(instance.beta, 0.0)
        self.assertEqual(instance.gamma, 0.0)
        self.assertEqual(instance.reg_concentration, 0)

        instance = NegMeanVarianceExAnteVol(alpha=0.5, beta=0.3, gamma=0.2, reg_concentration=0.1)
        self.assertEqual(instance.alpha, 0.5)
        self.assertEqual(instance.beta, 0.3)
        self.assertEqual(instance.gamma, 0.2)
        self.assertEqual(instance.reg_concentration, 0.1)

    def test_types_forward(self):
        instance = NegMeanVarianceExAnteVol(skip_validation = False)
        y_pred, y_true, vol = torch.randn(10, 5), torch.randn(10, 5), torch.rand(10, 5).abs() + 0.1
        self.assertRaises(TypeError, instance.forward, y_pred = y_pred, y_true = y_true, vol = "invalid")
        self.assertRaises(
            ValueError, instance.forward, y_pred = y_pred, y_true = y_true, vol = torch.rand(10, 3),
        )

    def test_valid_forward_matches_manual_formula(self):
        """alpha and reg_concentration only (beta=gamma=0): a single-period mean-variance
        loss plus the base class's own concentration penalty."""
        instance = NegMeanVarianceExAnteVol(alpha = 0.5, reg_concentration = 0.1, skip_validation = True)
        y_pred = torch.randn(20, 5)
        y_true = torch.randn(20, 5)
        vol = torch.rand(20, 5).abs() + 0.1

        loss_value = instance(y_pred = y_pred, y_true = y_true, vol = vol)

        portfolio_returns = torch.sum(y_pred * y_true, dim=1)
        risk_per_period = torch.sum((y_pred ** 2) * (vol ** 2), dim=1)
        single_step = -portfolio_returns + 0.5 * 0.5 * risk_per_period
        expected = torch.mean(single_step) + 0.1 * torch.mean(torch.sum(y_pred ** 2, dim=1))

        self.assertTrue(torch.allclose(loss_value, expected))

    def test_total_variance_term(self):
        """beta>0 adds the realised time-series variance of the portfolio return path."""
        instance = NegMeanVarianceExAnteVol(alpha = 0.5, beta = 0.3, skip_validation = True)
        y_pred = torch.randn(20, 5)
        y_true = torch.randn(20, 5)
        vol = torch.rand(20, 5).abs() + 0.1

        loss_value = instance(y_pred = y_pred, y_true = y_true, vol = vol)

        portfolio_returns = torch.sum(y_pred * y_true, dim=1)
        risk_per_period = torch.sum((y_pred ** 2) * (vol ** 2), dim=1)
        single_step = -portfolio_returns + 0.5 * 0.5 * risk_per_period
        expected = (torch.mean(single_step)
                   + 0.3 * torch.var(portfolio_returns, unbiased=True))

        self.assertTrue(torch.allclose(loss_value, expected))

    def test_turnover_term(self):
        """gamma>0 adds the squared period-to-period change in each name's weight."""
        instance = NegMeanVarianceExAnteVol(alpha = 0.5, gamma = 0.2, skip_validation = True)
        y_pred = torch.randn(20, 5)
        y_true = torch.randn(20, 5)
        vol = torch.rand(20, 5).abs() + 0.1

        loss_value = instance(y_pred = y_pred, y_true = y_true, vol = vol)

        portfolio_returns = torch.sum(y_pred * y_true, dim=1)
        risk_per_period = torch.sum((y_pred ** 2) * (vol ** 2), dim=1)
        single_step = -portfolio_returns + 0.5 * 0.5 * risk_per_period
        weight_changes = y_pred[1:] - y_pred[:-1]
        turnover = torch.mean(torch.sum(weight_changes ** 2, dim=1))
        expected = torch.mean(single_step) + 0.2 * turnover

        self.assertTrue(torch.allclose(loss_value, expected))

    def test_masks_missing_return_and_missing_vol_alike(self):
        instance = NegMeanVarianceExAnteVol(alpha = 1.0, skip_validation = True)
        y_pred = torch.tensor([[1.0, 2.0, 3.0]])
        y_true = torch.tensor([[0.1, float("nan"), 0.3]])
        vol = torch.tensor([[0.2, 0.2, float("nan")]])

        loss_value = instance(y_pred = y_pred, y_true = y_true, vol = vol)

        # Only the first name (index 0) has both a return and a vol
        expected_return = 1.0 * 0.1
        expected_risk = 1.0 ** 2 * 0.2 ** 2
        expected = -expected_return + 0.5 * 1.0 * expected_risk
        self.assertAlmostEqual(loss_value.item(), expected, places=5)

    def test_turnover_masks_entering_names(self):
        """A name absent in the first of two consecutive periods (entering the universe)
        contributes no turnover for that transition."""
        instance = NegMeanVarianceExAnteVol(alpha = 0.0, gamma = 1.0, skip_validation = True)
        y_pred = torch.tensor([[1.0, 5.0], [2.0, 7.0]])
        y_true = torch.tensor([[0.1, float("nan")], [0.2, 0.3]])
        vol = torch.tensor([[0.2, 0.2], [0.2, 0.2]])

        loss_value = instance(y_pred = y_pred, y_true = y_true, vol = vol)

        # name 1 is absent from period 0, so only name 0 (change 2 - 1) enters the turnover
        portfolio_returns = torch.tensor([1.0 * 0.1, 2.0 * 0.2 + 7.0 * 0.3])
        expected = torch.mean(-portfolio_returns) + 1.0 * (2.0 - 1.0) ** 2
        self.assertAlmostEqual(loss_value.item(), expected.item(), places=5)

    def test_turnover_masks_leaving_names(self):
        """A name absent in the second of two consecutive periods (leaving the universe)
        contributes no turnover for that transition."""
        instance = NegMeanVarianceExAnteVol(alpha = 0.0, gamma = 1.0, skip_validation = True)
        y_pred = torch.tensor([[1.0, 5.0], [2.0, 7.0]])
        y_true = torch.tensor([[0.1, 0.3], [0.2, float("nan")]])
        vol = torch.tensor([[0.2, 0.2], [0.2, 0.2]])

        loss_value = instance(y_pred = y_pred, y_true = y_true, vol = vol)

        portfolio_returns = torch.tensor([1.0 * 0.1 + 5.0 * 0.3, 2.0 * 0.2])
        expected = torch.mean(-portfolio_returns) + 1.0 * (2.0 - 1.0) ** 2
        self.assertAlmostEqual(loss_value.item(), expected.item(), places=5)

    def test_turnover_gap_in_the_middle_removes_both_transitions(self):
        """A name present at periods 0 and 2 but absent at 1 has neither the 0->1 nor the
        1->2 transition counted; it is not bridged across the gap."""
        instance = NegMeanVarianceExAnteVol(alpha = 0.0, gamma = 1.0, skip_validation = True)
        y_pred = torch.tensor([[1.0, 5.0], [2.0, 7.0], [4.0, 9.0]])
        y_true = torch.tensor([[0.1, 0.1], [0.1, float("nan")], [0.1, 0.1]])
        vol = torch.full((3, 2), 0.2)

        loss_value = instance(y_pred = y_pred, y_true = y_true, vol = vol)

        portfolio_returns = torch.tensor([1.0 * 0.1 + 5.0 * 0.1, 2.0 * 0.1, 4.0 * 0.1 + 9.0 * 0.1])
        # name 0 changes 1 -> 2 -> 4 (squares 1 and 4); name 1 contributes nothing
        turnover = torch.mean(torch.tensor([(2.0 - 1.0) ** 2, (4.0 - 2.0) ** 2]))
        expected = torch.mean(-portfolio_returns) + 1.0 * turnover
        self.assertAlmostEqual(loss_value.item(), expected.item(), places=5)

    def test_turnover_masks_missing_vol_like_missing_return(self):
        instance = NegMeanVarianceExAnteVol(alpha = 0.0, gamma = 1.0, skip_validation = True)
        y_pred = torch.tensor([[1.0, 5.0], [2.0, 7.0]])
        y_true = torch.tensor([[0.1, 0.3], [0.2, 0.3]])
        vol = torch.tensor([[0.2, float("nan")], [0.2, 0.2]])

        loss_value = instance(y_pred = y_pred, y_true = y_true, vol = vol)

        portfolio_returns = torch.tensor([1.0 * 0.1, 2.0 * 0.2 + 7.0 * 0.3])
        expected = torch.mean(-portfolio_returns) + 1.0 * (2.0 - 1.0) ** 2
        self.assertAlmostEqual(loss_value.item(), expected.item(), places=5)

    def test_turnover_gives_no_gradient_to_masked_transitions(self):
        """The weight of a name that is absent in either period of a transition receives no
        gradient from the turnover term."""
        instance = NegMeanVarianceExAnteVol(alpha = 0.0, gamma = 1.0, skip_validation = True)
        y_pred = torch.tensor([[1.0, 5.0], [2.0, 7.0]], requires_grad = True)
        # zero returns, so the only gradient is the turnover term's
        y_true = torch.tensor([[0.0, float("nan")], [0.0, 0.0]])
        vol = torch.full((2, 2), 0.2)

        instance(y_pred = y_pred, y_true = y_true, vol = vol).backward()

        self.assertTrue(torch.isfinite(y_pred.grad).all())
        self.assertEqual(y_pred.grad[0, 1].item(), 0.0)
        self.assertEqual(y_pred.grad[1, 1].item(), 0.0)
        self.assertNotEqual(y_pred.grad[0, 0].item(), 0.0)

    def test_single_period_batch_sets_beta_and_gamma_terms_to_zero(self):
        """A one-period batch has no variance across periods and no transition: the loss
        stays finite and equals the single-period loss, with finite gradients."""
        y_pred = torch.randn(1, 5, requires_grad = True)
        y_true = torch.randn(1, 5)
        vol = torch.rand(1, 5).abs() + 0.1

        full = NegMeanVarianceExAnteVol(alpha = 0.5, beta = 0.7, gamma = 0.3, skip_validation = True)
        plain = NegMeanVarianceExAnteVol(alpha = 0.5, skip_validation = True)

        loss_full = full(y_pred = y_pred, y_true = y_true, vol = vol)
        loss_plain = plain(y_pred = y_pred, y_true = y_true, vol = vol)

        self.assertTrue(torch.isfinite(loss_full))
        self.assertTrue(torch.allclose(loss_full, loss_plain))
        loss_full.backward()
        self.assertTrue(torch.isfinite(y_pred.grad).all())

    def test_two_period_batch_still_uses_beta(self):
        """The one-period guard does not switch the beta term off for longer batches."""
        y_pred = torch.randn(2, 5)
        y_true = torch.randn(2, 5)
        vol = torch.rand(2, 5).abs() + 0.1

        with_beta = NegMeanVarianceExAnteVol(alpha = 0.5, beta = 0.7, skip_validation = True)
        without_beta = NegMeanVarianceExAnteVol(alpha = 0.5, skip_validation = True)

        difference = (with_beta(y_pred = y_pred, y_true = y_true, vol = vol)
                      - without_beta(y_pred = y_pred, y_true = y_true, vol = vol))
        portfolio_returns = torch.sum(y_pred * y_true, dim=1)
        self.assertAlmostEqual(
            difference.item(), 0.7 * torch.var(portfolio_returns, unbiased=True).item(), places=5
        )

    def test_not_scale_invariant(self):
        """Unlike NegSharpeRatioExAnteVol, uniformly rescaling every weight changes the
        loss: the risk term is genuinely homogeneous degree 2 (no square root), not degree 1,
        so alpha pins down an optimal portfolio size -- the defining mean-variance property."""
        instance = NegMeanVarianceExAnteVol(alpha = 1.0, skip_validation = True)
        y_pred = torch.randn(20, 5)
        y_true = torch.randn(20, 5)
        vol = torch.rand(20, 5).abs() + 0.1

        base = instance(y_pred = y_pred, y_true = y_true, vol = vol)
        scaled = instance(y_pred = 3.0 * y_pred, y_true = y_true, vol = vol)
        self.assertNotAlmostEqual(base.item(), scaled.item(), places=2)