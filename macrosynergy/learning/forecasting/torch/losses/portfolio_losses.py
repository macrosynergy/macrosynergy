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

    def forward(self, y_pred, y_true):
        """
        Calculate loss.

        Parameters
        ----------
        y_pred : torch.Tensor
            Predicted portfolio weights. Dimension: (batch_size, n_assets)
        y_true : torch.Tensor
            True asset returns. Dimension: (batch_size, n_assets)
        """
        if not self.skip_validation:
            self._forward_checks(y_pred, y_true)

        mask = torch.isfinite(y_true)
        y_true_masked = torch.where(mask, y_true, torch.zeros_like(y_true))
        
        returns = y_pred * y_true_masked
        portfolio_returns = torch.sum(returns, dim=1)

        portfolio_loss = self._portfolio_loss(portfolio_returns)
        portfolio_loss = self._apply_reg_concentration(portfolio_loss, y_pred)

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

    def _portfolio_loss(self, portfolio_returns):
        """
        Calculate the portfolio loss based on the portfolio returns.

        Parameters
        ----------
        portfolio_returns : torch.Tensor
            Portfolio returns. Dimension: (batch_size,)
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def _forward_checks(self, y_pred, y_true):
        """
        Perform input validation checks for the forward method.

        Parameters
        ----------
        y_pred : torch.Tensor
            Predicted portfolio weights. Dimension: (batch_size, n_assets)
        y_true : torch.Tensor
            True asset returns. Dimension: (batch_size, n_assets)
        """
        if not isinstance(y_pred, torch.Tensor):
            raise TypeError("y_pred must be a torch.Tensor.")
        if not isinstance(y_true, torch.Tensor):
            raise TypeError("y_true must be a torch.Tensor.")
        if y_pred.shape != y_true.shape:
            raise ValueError("y_pred and y_true must have the same shape.")

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
    def _portfolio_loss(self, portfolio_returns):
        """
        Calculate the negative mean return of the portfolio.

        Parameters
        ----------
        portfolio_returns : torch.Tensor
            Portfolio returns. Dimension: (batch_size,)
        """
        return - torch.mean(portfolio_returns)
    
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
    def _portfolio_loss(self, portfolio_returns):
        """
        Calculate the variance of the portfolio.

        Parameters
        ----------
        portfolio_returns : torch.Tensor
            Portfolio returns. Dimension: (batch_size,)
        """
        return torch.var(portfolio_returns)
    
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

    def _portfolio_loss(self, portfolio_returns):
        """
        Calculate the negative mean-variance utility of the portfolio.

        Parameters
        ----------
        portfolio_returns : torch.Tensor
            Portfolio returns. Dimension: (batch_size,)
        """
        mean_return = torch.mean(portfolio_returns)
        variance = torch.var(portfolio_returns)
        utility = mean_return - 0.5 * self.alpha * variance

        return -utility

class NegMeanVarianceSkewnessUtility(PortfolioLoss):
    """
    Pytorch loss function to maximise the mean-variance-skewness utility of a portfolio, or
    equivalently minimise the negative mean-variance-skewness utility.

    Parameters
    ----------
    alpha : float, optional
        Risk aversion parameter for variance. Default is 1.
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

    def _portfolio_loss(self, portfolio_returns):
        """
        Calculate the negative mean-variance-skewness utility of the portfolio.

        Parameters
        ----------
        portfolio_returns : torch.Tensor
            Portfolio returns. Dimension: (batch_size,)
        """
        mean_return = torch.mean(portfolio_returns)
        variance = torch.var(portfolio_returns)
        skewness = torch.mean((portfolio_returns - mean_return) ** 3) / (torch.std(portfolio_returns) ** 3 + 1e-8)

        utility = mean_return - 0.5 * self.alpha * variance + (1/6) * self.alpha * skewness

        return -utility
    
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

    def _portfolio_loss(self, portfolio_returns):
        """
        Calculate loss.

        Parameters
        ----------
        portfolio_returns : torch.Tensor
            Portfolio returns. Dimension: (batch_size,)
        """
        mean_return = torch.mean(portfolio_returns)
        std_return = torch.std(portfolio_returns, unbiased=self.unbiased)

        sharpe_ratio = mean_return / (std_return + self.eps)

        loss = -sharpe_ratio

        return loss
