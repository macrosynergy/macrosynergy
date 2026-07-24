import torch 
import torch.nn as nn 

from sklearn.base import BaseEstimator

from macrosynergy.learning.forecasting.torch.losses import (
    NegSharpeRatio,
    PortfolioVariance,
    NegMeanPortfolioReturn,
    NegMeanVarianceUtility,
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