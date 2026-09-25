"""
Objectives and constraints for an active book defined relative to a benchmark.

A note on what benchmark weights can and cannot do, because it determines which of the
pieces below actually change the allocation.

Write the total book as `w = b + a`, benchmark plus active weights, with the active
weights summing to zero. The active return is then

.. code-block:: none

    w'r - b'r = a'r

so the quantity being maximised depends on `a` alone. And because `a'1 = 0`, adding any
constant to every asset's return leaves `a'r` unchanged: the active return is already
invariant to the common component, whether `r` is a total or a relative return. The
active risk `a' Sigma a` is likewise a function of `a` only.

**So for a dollar-neutral active book, the benchmark weights drop out of both the active
return and the active risk.** Simply reparameterising to active weights does not, on its
own, let the benchmark guide anything. That is worth knowing before spending a run on it.

Where the benchmark genuinely enters is narrower and more specific:

1. **Feasibility.** A long-only mandate requires `b_i + a_i >= 0`, so the room to
   underweight is exactly `b_i`. A name at 0.05% of the index cannot be shorted by 1%;
   one at 5% can. This binds differently for every asset and is pure benchmark
   information — see `BenchmarkFeasibilityPenalty`.
2. **What an error costs.** A 1% active position in a 5% index name is a materially
   larger real bet than the same 1% in a 0.05% name. Weighting each asset's contribution
   to the objective by its benchmark weight aligns the loss with the book that will
   actually be run — see `BenchmarkWeightedIC`.
3. **Turnover.** Costs accrue on the total book `b + a`, and the benchmark itself turns
   over at reconstitution.
4. **The risk budget.** Active share and tracking error are defined against `b`, and
   fixing them is what makes the model's risk-taking explicit — see
   `ActiveWeightModule`.

The classes here implement (1), (2) and (4). Index weights for this panel can be built
with `macrosynergy.securities.compute_daily_weights`; where they are unavailable, every
class below falls back to an equal-weight benchmark over the observed assets, which is
also the 1/N control that the estimation-error literature recommends carrying
permanently.
"""

import numbers

import torch
import torch.nn as nn

from sklearn.base import BaseEstimator

from macrosynergy.learning.forecasting.torch.losses.ranking_losses import (
    _masked_cross_sectional_moments,
)


def _resolve_benchmark(benchmark, y_true, mask, counts):
    """
    Broadcast a benchmark specification to per-period weights over the observed assets.

    Parameters
    ----------
    benchmark : None, str or torch.Tensor
        None or "equal" for an equal-weight benchmark over the assets observed in each
        period; a tensor of shape (n_assets,) for a fixed benchmark; or a tensor of
        shape (batch_size, n_assets) for period-varying weights.
    y_true : torch.Tensor
        Realised targets, used only for shape, device and dtype.
    mask : torch.Tensor
        Boolean tensor marking the observed entries of `y_true`.
    counts : torch.Tensor
        Number of observed assets per period.

    Returns
    -------
    torch.Tensor
        Benchmark weights, zero on unobserved assets and summing to one per period.
    """
    if benchmark is None or (isinstance(benchmark, str) and benchmark == "equal"):
        weights = mask.to(y_true.dtype)
    else:
        weights = benchmark.to(y_true.device, y_true.dtype)
        if weights.dim() == 1:
            weights = weights.unsqueeze(0).expand_as(y_true)
        weights = torch.where(mask, weights, torch.zeros_like(weights))

    # Renormalise over the assets actually observed, so that a name with no return this
    # period is dropped from the benchmark rather than held at a stale weight
    totals = weights.sum(dim=1, keepdim=True)
    return weights / totals.clamp(min=torch.finfo(y_true.dtype).eps)


class ActiveWeightModule(nn.Module):
    """
    Normalises network outputs into active weights with a fixed active share.

    Parameters
    ----------
    active_share : float, optional
        Target active share, i.e. half the sum of absolute active weights. Default is
        0.5, meaning half the book is positioned away from the benchmark.
    eps : float, optional
        Small value guarding the denominator. Default is 1e-8.

    Notes
    -----
    Designed as the final layer of a network whose outputs are to be read as positions
    away from a benchmark. The outputs are demeaned across assets, so that they sum to
    zero and the total book `b + a` remains fully invested, and then scaled so that

    .. code-block:: none

        0.5 * sum_i |a_i| = active_share

    Fixing the active share rather than letting the optimiser choose it separates the two
    decisions the model would otherwise make at once: *where* to deviate from the
    benchmark, and *how much* to deviate in total. The first is the forecasting problem;
    the second is a risk-budget decision that belongs to the mandate, not to the fit.
    Holding it fixed also makes runs comparable, since two models that differ only in
    gross exposure are not differing in skill.

    Unlike `LongShortModule`, this guards the denominator, because an active book can
    legitimately approach zero when the model has no view.
    """

    def __init__(self, active_share=0.5, eps=1e-8):
        super().__init__()

        if not isinstance(active_share, numbers.Real):
            raise TypeError("active_share must be a real number.")
        if active_share <= 0:
            raise ValueError("active_share must be positive.")
        if not isinstance(eps, numbers.Real):
            raise TypeError("eps must be a real number.")
        if eps <= 0:
            raise ValueError("eps must be positive.")

        self.active_share = active_share
        self.eps = eps

    def forward(self, x):
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Raw head outputs. Dimension: (batch_size, n_assets).
        """
        x = x - x.mean(dim=-1, keepdim=True)
        gross = x.abs().sum(dim=-1, keepdim=True)
        return 2.0 * self.active_share * x / (gross + self.eps)


class ActiveReturnLoss(nn.Module, BaseEstimator):
    """
    Negative mean active return, or negative information ratio, of an active book.

    Parameters
    ----------
    objective : str, optional
        One of "mean" for the negative mean active return, or "ir" for the negative
        information ratio. Default is "ir".
    unbiased : bool, optional
        Whether to use the unbiased estimator of the active-return standard deviation.
        Default is True.
    reg_tracking_error : float, optional
        Weight on a per-period tracking-error penalty, `mean_t sum_i (a_{t,i} * s_i)^2`,
        with `s` taken from `asset_vol`. Default is 0.
    asset_vol : torch.Tensor, optional
        Per-asset volatility for the tracking-error penalty. Dimension: (n_assets,).
        Default is None, in which case it is estimated from the batch's own targets and
        is therefore not causal.
    eps : float, optional
        Small value guarding the denominator. Default is 1e-8.

    Notes
    -----
    Expects the network outputs to be *active* weights summing to zero, as produced by
    `ActiveWeightModule`. The active return of period t is then `sum_i a_{t,i} r_{t,i}`,
    which as explained in this module's header is the book's return in excess of the
    benchmark's regardless of whether `r` holds total or relative returns.

    With `objective="ir"` this is the information ratio, and it inherits every property
    of a Sharpe-style objective: the mean and standard deviation are taken over the
    batch's periods, so the statistic is estimated from `batch_size` observations, it is
    non-causal within the batch, and its value depends on how the batch was composed.
    `objective="mean"` with `reg_tracking_error` set is the separated alternative, and it
    is decomposable over periods.
    """

    def __init__(
        self,
        objective="ir",
        unbiased=True,
        reg_tracking_error=0,
        asset_vol=None,
        eps=1e-8,
    ):
        super().__init__()

        if not isinstance(objective, str):
            raise TypeError("objective must be a string.")
        if objective not in {"mean", "ir"}:
            raise ValueError("objective must be one of 'mean' or 'ir'.")
        if not isinstance(unbiased, bool):
            raise TypeError("unbiased must be a boolean.")
        if not isinstance(reg_tracking_error, numbers.Real):
            raise TypeError("reg_tracking_error must be a real number.")
        if reg_tracking_error < 0:
            raise ValueError("reg_tracking_error must be non-negative.")

        self.objective = objective
        self.unbiased = unbiased
        self.reg_tracking_error = reg_tracking_error
        self.asset_vol = asset_vol
        self.eps = eps

    def forward(self, y_pred, y_true):
        """
        Evaluate the loss.

        Parameters
        ----------
        y_pred : torch.Tensor
            Active weights. Dimension: (batch_size, n_assets).
        y_true : torch.Tensor
            Realised returns. Dimension: (batch_size, n_assets).
        """
        mask = torch.isfinite(y_true)
        y_true_masked = torch.where(mask, y_true, torch.zeros_like(y_true))

        active_returns = (y_pred * y_true_masked).sum(dim=1)

        if self.objective == "mean":
            loss = -active_returns.mean()
        else:
            if active_returns.numel() < 2:
                loss = -active_returns.mean()
            else:
                std = active_returns.std(unbiased=self.unbiased)
                loss = -active_returns.mean() / (std + self.eps)

        if self.reg_tracking_error > 0:
            if self.asset_vol is not None:
                vol = self.asset_vol.to(y_pred.device, y_pred.dtype)
            else:
                counts = mask.sum(dim=0).clamp(min=1)
                means = y_true_masked.sum(dim=0) / counts
                centred = torch.where(
                    mask, y_true_masked - means, torch.zeros_like(y_true_masked)
                )
                vol = torch.sqrt((centred**2).sum(dim=0) / counts + self.eps).detach()
            tracking = ((y_pred * vol.unsqueeze(0)) ** 2).sum(dim=1).mean()
            loss = loss + self.reg_tracking_error * tracking

        return loss


class BenchmarkWeightedIC(nn.Module, BaseEstimator):
    """
    Negative benchmark-weighted cross-sectional correlation between outputs and targets.

    Parameters
    ----------
    benchmark : str or torch.Tensor, optional
        "equal" for an equal-weight benchmark, a tensor of shape (n_assets,) for fixed
        weights, or a tensor of shape (batch_size, n_assets) for period-varying weights.
        Default is "equal".
    power : float, optional
        Exponent applied to the benchmark weights before use, between 0 and 1. At 1 each
        asset's contribution is proportional to its index weight; at 0.5 to its square
        root; at 0 the loss reduces to an unweighted cross-sectional correlation.
        Default is 1.
    min_names : int, optional
        Minimum number of observed assets for a period to contribute. Default is 10.
    eps : float, optional
        Small value guarding the denominator. Default is 1e-8.

    Notes
    -----
    The weighted counterpart of `NegCrossSectionalIC`. Every mean, inner product and norm is
    taken under the benchmark measure rather than the counting measure:

    .. code-block:: none

        mean_q(x) = sum_i q_i x_i                with q_i the normalised benchmark weight
        rho_t     = sum_i q_i wc_i rc_i
                    / sqrt(sum_i q_i wc_i^2  *  sum_i q_i rc_i^2)
        L         = - mean_t rho_t

    with `wc` and `rc` centred under the same measure.

    This is the one ranking objective in which benchmark weights change what is learned.
    An unweighted correlation treats a misranking among the smallest names in the index
    as costing exactly as much as one among the largest, which is not how the resulting
    book behaves: the room to express a view is proportional to the index weight, so
    errors there are more expensive and are harder to offset elsewhere.

    `power` exists because full proportional weighting concentrates the objective on a
    handful of mega-caps and can reduce the effective breadth of the fit sharply. The
    square root is the usual compromise, and `power=0` recovers `NegCrossSectionalIC`
    exactly, which makes it the control for measuring whether the weighting helped.
    """

    def __init__(self, benchmark="equal", power=1, min_names=10, eps=1e-8):
        super().__init__()

        if not isinstance(power, numbers.Real):
            raise TypeError("power must be a real number.")
        if not (0 <= power <= 1):
            raise ValueError("power must be between 0 and 1.")
        if not isinstance(min_names, numbers.Integral):
            raise TypeError("min_names must be an integer.")
        if min_names < 2:
            raise ValueError("min_names must be at least 2.")

        self.benchmark = benchmark
        self.power = power
        self.min_names = min_names
        self.eps = eps

    def forward(self, y_pred, y_true):
        """
        Evaluate the loss.

        Parameters
        ----------
        y_pred : torch.Tensor
            Network outputs. Dimension: (batch_size, n_assets).
        y_true : torch.Tensor
            Realised targets. Dimension: (batch_size, n_assets).
        """
        mask = torch.isfinite(y_true)
        counts = mask.sum(dim=1)

        q = _resolve_benchmark(self.benchmark, y_true, mask, counts)
        if self.power != 1:
            # Re-mask after exponentiating: a zero weight raised to the power zero is
            # one, which would silently readmit the unobserved assets
            q = torch.where(mask, q**self.power, torch.zeros_like(q))
            q = q / q.sum(dim=1, keepdim=True).clamp(min=self.eps)

        y_true_filled = torch.where(mask, y_true, torch.zeros_like(y_true))

        pred_mean = (q * y_pred).sum(dim=1, keepdim=True)
        true_mean = (q * y_true_filled).sum(dim=1, keepdim=True)
        wc = torch.where(mask, y_pred - pred_mean, torch.zeros_like(y_pred))
        rc = torch.where(mask, y_true_filled - true_mean, torch.zeros_like(y_true_filled))

        numerator = (q * wc * rc).sum(dim=1)
        denominator = torch.sqrt((q * wc**2).sum(dim=1) * (q * rc**2).sum(dim=1))
        ic = numerator / denominator.clamp(min=self.eps)

        usable = counts >= self.min_names
        if not bool(usable.any()):
            return y_pred.sum() * 0.0
        return -ic[usable].mean()


class BenchmarkFeasibilityPenalty(nn.Module, BaseEstimator):
    """
    Penalises active weights that a long-only mandate could not implement.

    Parameters
    ----------
    benchmark : str or torch.Tensor, optional
        "equal", a tensor of shape (n_assets,), or a tensor of shape
        (batch_size, n_assets). Default is "equal".
    reg_feasibility : float, optional
        Weight on the penalty. Default is 1.

    Notes
    -----
    A long-only book can underweight asset `i` by at most its benchmark weight, since the
    total position `b_i + a_i` cannot go below zero. The shortfall

    .. code-block:: none

        L = reg_feasibility * mean_t sum_i max(0, -(b_{t,i} + a_{t,i}))^2

    is zero for any implementable book and grows quadratically in the violation.

    This is the term through which benchmark weights most directly guide the allocation.
    The active return and the active risk of a dollar-neutral book are both functions of
    the active weights alone, so the benchmark cancels out of them; it is the constraint
    set that it shapes. A name at five per cent of the index can absorb a large
    underweight, one at five basis points effectively cannot, and a model trained without
    this term will happily place underweights it could never implement — which shows up
    as a gap between backtested and implementable performance rather than as an error.

    Used as a penalty rather than a hard projection so that it stays differentiable and
    so that the strength of the constraint can be swept. A hard constraint belongs in the
    signal modifier, not the loss.
    """

    def __init__(self, benchmark="equal", reg_feasibility=1):
        super().__init__()

        if not isinstance(reg_feasibility, numbers.Real):
            raise TypeError("reg_feasibility must be a real number.")
        if reg_feasibility < 0:
            raise ValueError("reg_feasibility must be non-negative.")

        self.benchmark = benchmark
        self.reg_feasibility = reg_feasibility

    def forward(self, y_pred, y_true):
        """
        Evaluate the penalty.

        Parameters
        ----------
        y_pred : torch.Tensor
            Active weights. Dimension: (batch_size, n_assets).
        y_true : torch.Tensor
            Realised targets, used to identify the observed assets.
        """
        mask = torch.isfinite(y_true)
        counts = mask.sum(dim=1)
        b = _resolve_benchmark(self.benchmark, y_true, mask, counts)

        shortfall = torch.clamp(-(b + y_pred), min=0.0)
        shortfall = torch.where(mask, shortfall, torch.zeros_like(shortfall))

        return self.reg_feasibility * (shortfall**2).sum(dim=1).mean()
