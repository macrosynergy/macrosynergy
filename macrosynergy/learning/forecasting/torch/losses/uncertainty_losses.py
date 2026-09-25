import numbers

import torch
import torch.nn as nn

from sklearn.base import BaseEstimator


class GaussianNLL(nn.Module, BaseEstimator):
    r"""
    Negative Gaussian log-likelihood for a network predicting a mean and a variance.

    Parameters
    ----------
    beta : float, optional
        Exponent of the variance-dependent weight applied to each observation's
        contribution, between 0 and 1. 0 gives the exact negative log-likelihood; 1
        recovers the gradient structure of mean squared error. Default is 0.5.
    eps : float, optional
        Small value guarding the variance. Default is 1e-6.
    full : bool, optional
        Whether to include the constant `0.5 * log(2 * pi)`. It does not affect the
        gradients and is off by default, so the reported loss is comparable with the
        other objectives rather than with a likelihood. Default is False.

    Notes
    -----
    Expects a stacked prediction of width `2 * n_assets`, as produced by
    `HeteroskedasticMLP`: the conditional means followed by the conditional log
    variances. The loss is the negative log-likelihood of a Gaussian observation model
    with an observation-specific variance,

    .. code-block:: none

        L = mean over observed cells of
            w * [ (y - mu)^2 / (2 * s2)  +  0.5 * log(s2) ]

    which for `beta = 0` (so `w = 1`) is exactly the negative log-likelihood of a
    heteroskedastic Gaussian regression. Maximum likelihood then estimates the mean and
    variance equations jointly, and the first term is a precision-weighted sum of squares
    — generalised least squares — while the second is the penalty that stops the model
    from declaring everything uncertain to escape the first.

    **Why `beta` defaults away from the exact likelihood.** The exact NLL weights each
    observation by `1 / s2`, so early in training, when the mean head is still poor, the
    cheapest way to reduce the loss is to inflate the variance on the points fitted worst.
    Those points then contribute almost no gradient to the mean head, and the model
    settles into explaining the signal away as noise. Weighting each term by
    `stop_grad(s2) ^ beta` cancels that feedback: at `beta = 1` the effective weight on
    the squared-error term is constant, so the mean head trains as it would under mean
    squared error, while the variance head still learns. `beta = 0.5` is the usual
    compromise and is the default here. The weight is detached, so it changes the
    gradient without changing the stationary point of the variance equation.

    Missing targets are excluded from the average rather than filled, so an unbalanced
    panel contributes only its observed cells.
    """

    def __init__(self, beta=0.5, eps=1e-6, full=False):
        super().__init__()

        if not isinstance(beta, numbers.Real):
            raise TypeError("beta must be a real number.")
        if not (0 <= beta <= 1):
            raise ValueError("beta must be between 0 and 1.")
        if not isinstance(eps, numbers.Real):
            raise TypeError("eps must be a real number.")
        if eps <= 0:
            raise ValueError("eps must be positive.")
        if not isinstance(full, bool):
            raise TypeError("full must be a boolean.")

        self.beta = beta
        self.eps = eps
        self.full = full

    def forward(self, y_pred, y_true):
        """
        Evaluate the loss.

        Parameters
        ----------
        y_pred : torch.Tensor
            Stacked means and log variances. Dimension: (batch_size, 2 * n_assets).
        y_true : torch.Tensor
            Realised targets. Dimension: (batch_size, n_assets).
        """
        if y_pred.shape[1] != 2 * y_true.shape[1]:
            raise ValueError(
                "y_pred must be twice as wide as y_true, holding a conditional mean and "
                "a conditional log variance per asset. Got %d and %d."
                % (y_pred.shape[1], y_true.shape[1])
            )

        n_assets = y_true.shape[1]
        mu = y_pred[:, :n_assets]
        log_var = y_pred[:, n_assets:]
        variance = log_var.exp().clamp(min=self.eps)

        mask = torch.isfinite(y_true)
        residual = torch.where(mask, y_true - mu, torch.zeros_like(mu))

        terms = residual**2 / (2 * variance) + 0.5 * log_var
        if self.full:
            terms = terms + 0.5 * torch.log(torch.tensor(2 * torch.pi, dtype=terms.dtype))

        if self.beta > 0:
            # Detached, so this reweights the gradient without moving the optimum of the
            # variance equation
            terms = terms * variance.detach() ** self.beta

        terms = torch.where(mask, terms, torch.zeros_like(terms))
        count = mask.sum().clamp(min=1)
        return terms.sum() / count
