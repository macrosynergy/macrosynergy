import torch
import torch.nn as nn

from sklearn.base import BaseEstimator

# TODO: write base class 

class NegMeanPortfolioReturn(nn.Module, BaseEstimator):
    """
    PyTorch loss function to maximise the mean return of a portfolio, or equivalently
    minimise the negative mean return.

    Parameters
    ----------
    alpha : float, optional
        Regularization parameter for concentration penalty. Default is 0 (no penalty).

    Notes
    -----
    This loss function is designed for portfolio optimization tasks, meaning that it 
    expects the model to output quantities interpretable as portfolio weights or signals. 
    """
    def __init__(self, alpha = 0):
        super().__init__()
        self.alpha = alpha

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
        returns = y_pred * y_true
        portfolio_returns = torch.sum(returns, dim=1)

        loss = - torch.mean(portfolio_returns)
        
        if self.alpha > 0:
            concentration = torch.mean(torch.sum(y_pred ** 2, dim=1))
            loss += self.alpha * concentration

        return loss
    
class PortfolioVariance(nn.Module):
    """
    PyTorch loss function to minimise the variance of a portfolio.

    Notes
    -----
    This loss function is designed for portfolio optimization tasks, meaning that it 
    expects the model to output quantities interpretable as portfolio weights or signals. 
    """
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
        returns = y_pred * y_true
        portfolio_returns = torch.sum(returns, dim=1)

        loss = torch.var(portfolio_returns)

        return loss
    
class NegMeanVarianceUtility(nn.Module, BaseEstimator):
    """
    Pytorch loss function to maximise the mean-variance utility of a portfolio, or
    equivalently minimise the negative mean-variance utility.

    Parameters
    ----------
    alpha : float, optional
        Risk aversion parameter. Default is 1.

    Notes
    -----
    This loss function is designed for portfolio optimization tasks, meaning that it 
    expects the model to output quantities interpretable as portfolio weights or signals. 
    """
    def __init__(self, alpha = 1):
        super().__init__()
        self.alpha = alpha

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
        returns = y_pred * y_true
        portfolio_returns = torch.sum(returns, dim=1)

        mean_return = torch.mean(portfolio_returns)
        var_return = torch.var(portfolio_returns)

        loss = - (mean_return - 0.5 * self.alpha * var_return)

        return loss
    
class NegSharpeRatio(nn.Module):
    """
    PyTorch loss function to maximise the Sharpe ratio of a portfolio, or equivalently
    minimise the negative Sharpe ratio.

    Parameters
    ----------
    unbiased : bool, optional
        Whether to use the unbiased estimator for variance. Default is True.
    eps : float, optional
        Small value to avoid division by zero. Default is 1e-8.

    Notes
    -----
    This loss function is designed for portfolio optimization tasks, meaning that it
    expects the model to output quantities interpretable as portfolio weights or signals.

    For simplicity, we leave out the risk free rate in the Sharpe ratio calculation.
    """
    def __init__(self, unbiased=True, eps=1e-8):
        super().__init__()
        self.unbiased = unbiased
        self.eps = eps

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
        returns = y_pred * y_true
        portfolio_returns = torch.sum(returns, dim=1)

        mean_return = torch.mean(portfolio_returns)
        std_return = torch.std(portfolio_returns, unbiased=self.unbiased)

        sharpe_ratio = mean_return / (std_return + self.eps)

        loss = -sharpe_ratio

        return loss
