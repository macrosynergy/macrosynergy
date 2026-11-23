import numbers

import torch
import torch.nn as nn

from sklearn.base import BaseEstimator

class PortfolioLoss(nn.Module, BaseEstimator):
    """
    Base class for portfolio loss functions.

    Parameters
    ----------
    reg_concentration : float, optional
        Regularization parameter for concentration penalty. Default is 0 (no penalty).
    skip_validation : bool, optional
        Whether to skip input validation checks for the `forward` method. Default is True.

    Notes
    -----
    This is a base class for loss functions based on portfolio optimization. It
    expects the model to output quantities interpretable as portfolio weights or signals. 
    """
    def __init__(self, reg_concentration = 0, skip_validation = True):
        super().__init__()

        # Checks
        if not isinstance(reg_concentration, numbers.Number):
            raise TypeError("reg_concentration must be a number.")
        if reg_concentration < 0:
            raise ValueError("reg_concentration must be non-negative.")
        if not isinstance(skip_validation, bool):
            raise TypeError("skip_validation must be a boolean.")
        
        self.reg_concentration = reg_concentration
        self.skip_validation = skip_validation

    def forward(self, y_pred, y_true, sample_weight = None):
        """
        Calculate loss.

        Parameters
        ----------
        y_pred : torch.Tensor
            Predicted portfolio weights. Dimension: (batch_size, n_assets)
        y_true : torch.Tensor
            True asset returns. Dimension: (batch_size, n_assets)
        sample_weight : torch.Tensor, optional
            Weight contributions of each sample in the batch to the loss over the batch. 
            Dimension: (batch_size,).
        """
        if not self.skip_validation:
            self._forward_checks(y_pred, y_true, sample_weight)

        mask = torch.isfinite(y_true)
        y_true_masked = torch.where(mask, y_true, torch.zeros_like(y_true))
        
        returns = y_pred * y_true_masked
        portfolio_returns = torch.sum(returns, dim=1)

        portfolio_loss = self._portfolio_loss(portfolio_returns, sample_weight)
        portfolio_loss = self._apply_reg_concentration(portfolio_loss, y_pred) # Should sample weights be applied here?

        return portfolio_loss

    def _apply_reg_concentration(self, loss, y_pred):
        """
        Apply concentration regularization to the loss.

        Parameters
        ----------
        loss : torch.Tensor
            The original loss value.
        y_pred : torch.Tensor
            Predicted portfolio weights. Dimension: (batch_size, n_assets)
        """
        if self.reg_concentration > 0:
            concentration = torch.mean(torch.sum(y_pred ** 2, dim=1))
            loss += self.reg_concentration * concentration

        return loss

    def _weighted_mean(self, x, normalized_weight):
        return torch.sum(normalized_weight * x)

    def _weighted_variance(self, x, normalized_weight, unbiased = True):
        mean = self._weighted_mean(x, normalized_weight)

        variance = torch.sum(normalized_weight * (x - mean)**2)

        if unbiased:
            correction = 1 - torch.sum(normalized_weight ** 2)
            variance = variance / correction

        return variance

    def _weighted_moment3(self, x, normalized_weight):
        mean = self._weighted_mean(x, normalized_weight)

        moment3 = torch.sum(normalized_weight * (x - mean)**3)

        return moment3

    def _portfolio_loss(self, portfolio_returns, sample_weight):
        """
        Calculate the portfolio loss based on the portfolio returns.

        Parameters
        ----------
        portfolio_returns : torch.Tensor
            Portfolio returns. Dimension: (batch_size,)
        sample_weight : torch.Tensor, optional
            Weight contributions of each sample in the batch to the loss over the batch. 
            Dimension: (batch_size,).
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def _forward_checks(self, y_pred, y_true, sample_weight):
        """
        Perform input validation checks for the forward method.

        Parameters
        ----------
        y_pred : torch.Tensor
            Predicted portfolio weights. Dimension: (batch_size, n_assets)
        y_true : torch.Tensor
            True asset returns. Dimension: (batch_size, n_assets)
        sample_weight : torch.Tensor, optional
            Weight contributions of each sample in the batch to the loss over the batch.
            Dimension: (batch_size,)
        """
        if not isinstance(y_pred, torch.Tensor):
            raise TypeError("y_pred must be a torch.Tensor.")
        if not isinstance(y_true, torch.Tensor):
            raise TypeError("y_true must be a torch.Tensor.")
        if y_pred.shape != y_true.shape:
            raise ValueError("y_pred and y_true must have the same shape.")
        if sample_weight is not None:
            if not isinstance(sample_weight, torch.Tensor):
                raise TypeError("sample_weight must be a torch.Tensor.")
            if not all(sample_weight > 0):
                raise ValueError("sample_weights must be positive.")

class NegMeanPortfolioReturn(PortfolioLoss):
    """
    PyTorch loss function to maximise the mean return of a portfolio, or equivalently
    minimise the negative mean return.

    Parameters
    ----------
    reg_concentration : float, optional
        Regularization parameter for concentration penalty. Default is 0 (no penalty).

    Notes
    -----
    This loss function is designed for portfolio optimization tasks, meaning that it 
    expects the model to output quantities interpretable as portfolio weights or signals. 
    """
    def _portfolio_loss(self, portfolio_returns, sample_weight):
        """
        Calculate the negative mean return of the portfolio.

        Parameters
        ----------
        portfolio_returns : torch.Tensor
            Portfolio returns. Dimension: (batch_size,)
        sample_weight : torch.Tensor, optional
            Weight contributions of each sample in the batch to the loss over the batch. 
            Dimension: (batch_size,).
        """
        if sample_weight is None:
            return - torch.mean(portfolio_returns)

        weights = sample_weight / sample_weight.sum()
        mean = self._weighted_mean(
            x = portfolio_returns,
            normalized_weight = weights
        )
        batch_importance = sample_weight.mean()

        return - mean * batch_importance
    
class PortfolioVariance(PortfolioLoss):
    """
    PyTorch loss function to minimise the variance of a portfolio.

    Parameters
    ----------
    reg_concentration : float, optional
        Regularization parameter for concentration penalty. Default is 0 (no penalty).

    Notes
    -----
    This loss function is designed for portfolio optimization tasks, meaning that it 
    expects the model to output quantities interpretable as portfolio weights or signals. 
    """
    def _portfolio_loss(self, portfolio_returns, sample_weight):
        """
        Calculate the variance of the portfolio.

        Parameters
        ----------
        portfolio_returns : torch.Tensor
            Portfolio returns. Dimension: (batch_size,)
        sample_weight : torch.Tensor, optional
            Weight contributions of each sample in the batch to the loss over the batch. 
            Dimension: (batch_size,).
        """
        if sample_weight is None:
            return torch.var(portfolio_returns)

        normalized_weights = sample_weight / sample_weight.sum()
        variance = self._weighted_variance(
            x = portfolio_returns,
            normalized_weight=normalized_weights
        )

        batch_importance = sample_weight.mean()

        return batch_importance * variance
    
class NegMeanVarianceUtility(PortfolioLoss):
    """
    Pytorch loss function to maximise the mean-variance utility of a portfolio, or
    equivalently minimise the negative mean-variance utility.

    Parameters
    ----------
    alpha : float, optional
        Risk aversion parameter. Default is 1.
    reg_concentration : float, optional
        Regularization parameter for concentration penalty. Default is 0 (no penalty).
    skip_validation : bool, optional
        Whether to skip input validation checks for the `forward` method. Default is True.

    Notes
    -----
    This loss function is designed for portfolio optimization tasks, meaning that it 
    expects the model to output quantities interpretable as portfolio weights or signals. 
    """
    def __init__(self, alpha = 1, reg_concentration = 0, skip_validation = True):
        super().__init__(reg_concentration = reg_concentration, skip_validation = skip_validation)
        self.alpha = alpha

    def _portfolio_loss(self, portfolio_returns, sample_weight):
        """
        Calculate the negative mean-variance utility of the portfolio.

        Parameters
        ----------
        portfolio_returns : torch.Tensor
            Portfolio returns. Dimension: (batch_size,)
        sample_weight : torch.Tensor, optional
            Weight contributions of each sample in the batch to the loss over the batch. 
            Dimension: (batch_size,).

        """
        if sample_weight is None:
            mean_return = torch.mean(portfolio_returns)
            variance = torch.var(portfolio_returns)
            utility = mean_return - 0.5 * self.alpha * variance

            return -utility

        normalized_weight = sample_weight / sample_weight.sum()

        mean = self._weighted_mean(
            x = portfolio_returns,
            normalized_weight=normalized_weight
        )
        variance = self._weighted_variance(
            x = portfolio_returns,
            normalized_weight = normalized_weight
        )
        utility = mean - 0.5 * self.alpha * variance

        batch_importance = sample_weight.mean()

        return - utility * batch_importance


class NegMeanVarianceSkewnessUtility(PortfolioLoss):
    """
    Pytorch loss function to maximise the mean-variance-skewness utility of a portfolio, or
    equivalently minimise the negative mean-variance-skewness utility.

    Parameters
    ----------
    alpha : float, optional
        Risk aversion parameter for variance and skewness. Default is 1.
    reg_concentration : float, optional
        Regularization parameter for concentration penalty. Default is 0 (no penalty).
    skip_validation : bool, optional
        Whether to skip input validation checks for the `forward` method. Default is True.

    Notes
    -----
    This loss function is designed for portfolio optimization tasks, meaning that it 
    expects the model to output quantities interpretable as portfolio weights or signals. 
    """
    def __init__(self, alpha = 1, reg_concentration = 0, skip_validation = True):
        super().__init__(reg_concentration = reg_concentration, skip_validation = skip_validation)
        self.alpha = alpha

    def _portfolio_loss(self, portfolio_returns, sample_weight):
        """
        Calculate the negative mean-variance-skewness utility of the portfolio.

        Parameters
        ----------
        portfolio_returns : torch.Tensor
            Portfolio returns. Dimension: (batch_size,)
        sample_weight : torch.Tensor, optional
            Weight contributions of each sample in the batch to the loss over the batch. 
            Dimension: (batch_size,).
        """
        if sample_weight is None:
            mean_return = torch.mean(portfolio_returns)
            variance = torch.var(portfolio_returns)
            skewness = torch.mean((portfolio_returns - mean_return) ** 3)

            utility = mean_return - 0.5 * self.alpha * variance + (1/6) * self.alpha * skewness

            return -utility
        else:
            normalized_weight = sample_weight / sample_weight.sum()

            mean = self._weighted_mean(
                x = portfolio_returns,
                normalized_weight=normalized_weight
            )
            variance = self._weighted_variance(
                x = portfolio_returns,
                normalized_weight=normalized_weight,
            )
            skew = self._weighted_moment3(
                x = portfolio_returns,
                normalized_weight=normalized_weight,
            )
            utility = mean - 0.5 * self.alpha * variance + (1/6) * self.alpha * skew
            batch_importance = sample_weight.mean()

            return - utility * batch_importance
    
class NegSharpeRatio(PortfolioLoss):
    """
    PyTorch loss function to maximise the Sharpe ratio of a portfolio, or equivalently
    minimise the negative Sharpe ratio.

    Parameters
    ----------
    unbiased : bool, optional
        Whether to use the unbiased estimator for variance. Default is True.
    reg_concentration : float, optional
        Regularization parameter for concentration penalty. Default is 0 (no penalty).
    eps : float, optional
        Small value to avoid division by zero. Default is 1e-8.
    skip_validation : bool, optional
        Whether to skip input validation checks for the `forward` method. Default is True.

    Notes
    -----
    This loss function is designed for portfolio optimization tasks, meaning that it
    expects the model to output quantities interpretable as portfolio weights or signals.

    For simplicity, we leave out the risk free rate in the Sharpe ratio calculation.
    """
    def __init__(self, unbiased=True, reg_concentration=0, eps=1e-8, skip_validation=True):
        super().__init__(reg_concentration = reg_concentration, skip_validation = skip_validation)
        
        self.unbiased = unbiased
        self.eps = eps

    def _portfolio_loss(self, portfolio_returns, sample_weight):
        """
        Calculate loss.

        Parameters
        ----------
        portfolio_returns : torch.Tensor
            Portfolio returns. Dimension: (batch_size,)
        sample_weight : torch.Tensor
            Sample weights for each portfolio return. Dimension: (batch_size,)
        """
        if sample_weight is None:
            mean_return = torch.mean(portfolio_returns)
            std_return = torch.std(portfolio_returns, unbiased=self.unbiased)

            sharpe_ratio = mean_return / (std_return + self.eps)

            loss = -sharpe_ratio

            return loss
        else:
            normalized_weight = sample_weight / sample_weight.sum()
            mean = self._weighted_mean(
                x = portfolio_returns,
                normalized_weight = normalized_weight
            )
            variance = self._weighted_variance(
                x = portfolio_returns,
                normalized_weight = normalized_weight,
                unbiased = self.unbiased
            )
            std = torch.sqrt(variance)

            sharpe_ratio = mean / (std + self.eps)
            loss = -sharpe_ratio

            batch_importance = sample_weight.mean()

            return loss * batch_importance
