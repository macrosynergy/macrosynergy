import numbers

import torch
import torch.nn as nn

from sklearn.base import BaseEstimator


def _masked_cross_sectional_moments(values, mask, counts):
    """
    Centre a tensor across the observed assets of each period.

    Parameters
    ----------
    values : torch.Tensor
        Dimension: (batch_size, n_assets).
    mask : torch.Tensor
        Boolean tensor marking the observed entries of `values`.
    counts : torch.Tensor
        Number of observed assets per period. Dimension: (batch_size,).

    Returns
    -------
    torch.Tensor
        `values` less its per-period mean over the observed assets, with unobserved
        entries set to zero so that they contribute nothing to any subsequent sum.
    """
    filled = torch.where(mask, values, torch.zeros_like(values))
    means = filled.sum(dim=1) / counts.clamp(min=1)
    centred = values - means.unsqueeze(1)
    return torch.where(mask, centred, torch.zeros_like(centred))


@torch.no_grad()
def _cross_sectional_ranks(y_true, mask, counts):
    """
    Rank the observed targets of each period, ascending, as floats centred on zero.

    Parameters
    ----------
    y_true : torch.Tensor
        Realised targets. Dimension: (batch_size, n_assets).
    mask : torch.Tensor
        Boolean tensor marking the observed entries of `y_true`.
    counts : torch.Tensor
        Number of observed assets per period. Dimension: (batch_size,).

    Notes
    -----
    The targets are data, not a function of the network's parameters, so ranking them
    needs no differentiable surrogate: an exact sort is used. Correlating the predictions
    against these ranks therefore yields a Spearman-style statistic that remains
    differentiable in the predictions, without any of the relaxations a differentiable
    sort would require. `torch.no_grad` states that intent rather than relying on it:
    `argsort` and `scatter_` produce integer tensors that carry no gradient anyway, so the
    decorator changes nothing at runtime and everything for a reader.

    Unobserved entries are sent to the end of the ordering and then zeroed, so they
    contribute nothing to the correlation.

    Ties take arbitrary distinct ranks rather than their average, because `argsort` does
    not resolve them. For continuous returns ties have measure zero; for rounded or
    truncated data they do not, and the affected names are separated arbitrarily.
    """
    sentinel = torch.finfo(y_true.dtype).max
    filled = torch.where(mask, y_true, torch.full_like(y_true, sentinel))

    order = filled.argsort(dim=1)
    positions = torch.arange(y_true.shape[1], device=y_true.device).expand_as(order)
    ranks = torch.empty_like(order)
    ranks.scatter_(1, order, positions)

    ranks = ranks.to(y_true.dtype)
    # Centre on the midpoint of each period's own ranks, so that periods with different
    # numbers of observed assets are on a common scale
    midpoints = (counts.to(y_true.dtype) - 1) / 2
    centred = ranks - midpoints.unsqueeze(1)
    return torch.where(mask, centred, torch.zeros_like(centred))


class NegCrossSectionalIC(nn.Module, BaseEstimator):
    """
    Negative mean cross-sectional correlation between network outputs and realised
    targets.

    The `Neg` prefix follows `NegSharpeRatio` and the rest of the `portfolio_losses`
    family: the class returns the *negative* of the quantity named, because an optimiser
    minimises. A value of −0.03 is an information coefficient of +0.03.

    Parameters
    ----------
    min_names : int, optional
        Minimum number of observed assets for a period to contribute. Default is 10.
    rank_targets : bool, optional
        Whether to replace the targets by their within-period ranks, giving a
        Spearman-style statistic rather than a Pearson one. Default is False.
    eps : float, optional
        Small value guarding the denominator. Default is 1e-8.

    Notes
    -----
    This is the differentiable counterpart of the diagnostic that `MLPRegressor` already
    reports each epoch, so that the quantity being measured and the quantity being
    optimised are the same one:

    .. code-block:: none

        rho_t = <w_t - mean(w_t), r_t - mean(r_t)>
                / (||w_t - mean(w_t)|| * ||r_t - mean(r_t)||)
        L     = - mean_t rho_t

    with both series centred across the assets observed in period t.

    Three properties distinguish it from the portfolio objectives in
    `portfolio_losses.py`, and they are the reason it exists:

    1. **It is purely cross-sectional.** Centring removes the period's common component,
       so the loss cannot be improved by timing the market — only by ranking assets
       within a period. Where time-series variation dominates the panel, this is what
       stops that variation from swamping the signal being learned.
    2. **It is scale-free.** `rho_t` is invariant to rescaling the outputs, so the
       objective says nothing about position size. That is a feature here: sizing is the
       job of the separate risk term in `RankingRiskLoss`, not of the ranking term.
    3. **It is a mean over periods.** Unlike a Sharpe ratio, which is a ratio over the
       batch, the value of this loss does not depend on how rows were grouped into
       batches. Any `PanelBatchSampler` mode may be used with it.

    Periods with fewer than `min_names` observed assets are dropped rather than
    down-weighted: a correlation over two assets is exactly +/-1 whatever the data, and
    over a handful it is dominated by sampling noise.
    """

    def __init__(self, min_names=10, rank_targets=False, eps=1e-8):
        super().__init__()

        if not isinstance(min_names, numbers.Integral):
            raise TypeError("min_names must be an integer.")
        if min_names < 2:
            raise ValueError("min_names must be at least 2.")
        if not isinstance(rank_targets, bool):
            raise TypeError("rank_targets must be a boolean.")
        if not isinstance(eps, numbers.Real):
            raise TypeError("eps must be a real number.")
        if eps <= 0:
            raise ValueError("eps must be positive.")

        self.min_names = min_names
        self.rank_targets = rank_targets
        self.eps = eps

    def period_ic(self, y_pred, y_true):
        """
        Cross-sectional correlation of each period, and which periods are usable.

        Parameters
        ----------
        y_pred : torch.Tensor
            Network outputs. Dimension: (batch_size, n_assets).
        y_true : torch.Tensor
            Realised targets. Dimension: (batch_size, n_assets).

        Returns
        -------
        tuple of torch.Tensor
            The per-period correlation, and a boolean mask of the periods carrying at
            least `min_names` observed assets.
        """
        mask = torch.isfinite(y_true)
        counts = mask.sum(dim=1)

        targets = (
            _cross_sectional_ranks(y_true, mask, counts)
            if self.rank_targets
            else _masked_cross_sectional_moments(y_true, mask, counts)
        )
        if self.rank_targets:
            # Ranks are centred on each period's midpoint, which is the mean only when
            # every asset is observed; centre again over the observed entries
            targets = _masked_cross_sectional_moments(targets, mask, counts)

        weights = _masked_cross_sectional_moments(y_pred, mask, counts)

        numerator = (weights * targets).sum(dim=1)
        # Clamp the *squared* norms before taking the root, not the root afterwards.
        # Both orderings avoid dividing by zero, but the floor they impose differs by a
        # square root: clamping after leaves a denominator as small as `eps`, clamping
        # before leaves one no smaller than `sqrt(eps)`. On a collapsed cross-section --
        # every prediction identical, so the centred row is exactly zero -- the first
        # ordering emits gradients of order 1e7 where the second emits 1e3. That case is
        # not hypothetical: a macro-only head produces one forecast per period by
        # construction, and an untrained head can start there.
        denominator = torch.sqrt(
            ((weights**2).sum(dim=1) * (targets**2).sum(dim=1)).clamp(min=self.eps)
        )
        return numerator / denominator, counts >= self.min_names

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
        ic, usable = self.period_ic(y_pred, y_true)
        if not bool(usable.any()):
            # No period in this batch carries enough names to be informative. Returning a
            # zero that still depends on the parameters keeps the graph intact, so the
            # step is a no-op rather than an error
            return y_pred.sum() * 0.0
        return -ic[usable].mean()


class NegRankIC(NegCrossSectionalIC):
    """
    Negative mean cross-sectional rank correlation between outputs and realised targets.

    Parameters
    ----------
    min_names : int, optional
        Minimum number of observed assets for a period to contribute. Default is 10.
    eps : float, optional
        Small value guarding the denominator. Default is 1e-8.

    Notes
    -----
    `NegCrossSectionalIC` with the targets replaced by their within-period ranks. Ranking
    the targets bounds the influence of any single asset's return, which matters at
    monthly and shorter horizons where the cross-sectional return distribution is
    materially fat-tailed and a Pearson correlation can be dominated by one name.
    """

    def __init__(self, min_names=10, eps=1e-8):
        super().__init__(min_names=min_names, rank_targets=True, eps=eps)


class RankingRiskLoss(nn.Module, BaseEstimator):
    """
    Cross-sectional ranking objective with a separate, explicitly weighted risk penalty.

    Parameters
    ----------
    reg_risk : float, optional
        Weight on the selection risk term. Zero disables it. Default is 1.
    risk : str, optional
        Which selection risk term to penalise. One of "batch_vol", "period_var",
        "period_cov" or "period_disp". Default is "period_var".
    reg_systematic : float, optional
        Weight on the market-exposure term, which charges the book for its net exposure
        scaled by the market's volatility. Zero disables it, which is the default and
        recovers the previous behaviour. Requires total-return targets to be meaningful.
    market_vol : float or torch.Tensor, optional
        Volatility of the market return used by the systematic term. Default is None, in
        which case it is estimated from the batch's own cross-sectional means.
    asset_vol : torch.Tensor, optional
        Per-asset volatility used by "period_var". Dimension: (n_assets,). Default is
        None, in which case it is estimated from the batch's own targets.
    cov : torch.Tensor, optional
        Asset covariance matrix used by "period_cov". Dimension: (n_assets, n_assets).
        Default is None, in which case it is estimated from the batch's own targets and
        shrunk towards its diagonal.
    shrinkage : float, optional
        Weight on the diagonal target when a covariance is estimated in-batch, between 0
        and 1. Default is 0.5.
    min_names : int, optional
        Minimum number of observed assets for a period to contribute to the ranking term.
        Default is 10.
    rank_targets : bool, optional
        Whether the ranking term uses rank correlation. Default is False.
    eps : float, optional
        Small value guarding denominators. Default is 1e-8.

    Notes
    -----
    This separates the two things a Sharpe-ratio objective conflates:

    .. code-block:: none

        L = - mean_t rho_t  +  reg_risk * risk

    where `rho_t` is the per-period cross-sectional correlation of `NegCrossSectionalIC`,
    and `risk` is one of the three terms below.

    In `- mean(r) / std(r)`, ranking skill and risk control trade off against one another
    at a rate set by the current estimate of that ratio. Estimated from `batch_size`
    periods — sixteen or thirty-two of them — that rate is mostly noise, and it changes
    every step. Separating the terms replaces it with a fixed, interpretable `reg_risk`
    that can be swept, and lets the two be diagnosed independently: a run can be seen to
    rank well and size badly, which a single ratio cannot show.

    **The three risk terms differ in whether they are decomposable over periods**, which
    determines which batching schemes remain valid:

    - **"batch_vol"** is `std_t(r_t)`, the standard deviation of the portfolio return
      across the batch's periods. This is the term implicit in `NegSharpeRatio`. It is a
      statistic *of the batch*, so its value depends on batch composition and it requires
      `PanelBatchSampler` in "period" mode. It is also non-causal within the batch: the
      weight at period t is penalised against dispersion that includes later periods.
    - **"period_var"** is `mean_t sum_i (w_{t,i} * s_i)^2`, a diagonal risk model. It is a
      mean over per-period quantities, so it is decomposable and any batching mode is
      valid. It ignores cross-asset correlation.
    - **"period_cov"** is `mean_t w_t' Sigma w_t` for a fixed `Sigma`. Also decomposable.
      This is the term to use when `Sigma` is estimated causally, outside the batch, and
      passed in.
    - **"period_disp"** is `mean_t [ sum_i w_{t,i}^2 * s_t^2 ]`, where `s_t^2` is the
      *realised cross-sectional variance* of period t's own observed returns. Decomposable.
      Unlike "period_var" it needs no volatility estimate at all: dispersion is measured
      within the period from every name present, so at five hundred names it is precise,
      and it varies with the period rather than being a per-asset constant. This is the
      term to prefer when the targets are total returns.

    **Two risk sources, not one, and total returns are what separate them.** With total
    returns the cross-sectional mean of a period is the equal-weighted market return, so

    .. code-block:: none

        r_i,t = m_t + e_i,t
        r_p,t = (sum_i w_{t,i}) * m_t   +   sum_i w_{t,i} * e_{t,i}
                `-- net exposure --'        `----- selection -----'

    `reg_systematic` charges the first, `reg_risk` and `risk` the second, and the two
    sweep independently. The decomposition is better conditioned than a full covariance in
    both halves: the market's variance is a single number, and the cross-sectional
    variance of "period_disp" is measured across the whole cross-section *within* a
    period rather than estimated over time, where an in-batch `Sigma` has at most
    `n_periods - 1` non-zero eigenvalues.

    On **index-relative** targets the split collapses: the cross-section is already
    centred, so `m_t` is approximately zero and `reg_systematic` has nothing to charge.
    It is the move to total returns that makes the term meaningful.

    **The systematic term requires the outputs to be weights.** `NegCrossSectionalIC`
    centres both series, so the ranking objective is invariant to adding a constant to a
    period's outputs, and with no `signal_modifier` the level of `y_pred` is unidentified.
    Penalising `sum_i w_i` then merely picks a gauge -- it pins the free level at zero,
    which is a harmless regulariser but is not risk control. For the term to charge real
    market exposure the model must emit weights, i.e. a `signal_modifier` must be applied,
    and it must be one that leaves net exposure free: under
    `LongShortModule(dollar_neutral=True)` the net is zero by construction and the term is
    identically zero.

    When `asset_vol` or `cov` is not supplied it is estimated from the batch's own
    targets. That is convenient for exploration but **not causal** — the estimate uses
    periods the weights are being chosen for. Supply the statistic, estimated on data
    strictly prior to the batch, for anything whose result is to be believed. An in-batch
    covariance is additionally rank-deficient whenever there are more assets than periods
    in the batch, which is why it is shrunk towards its own diagonal by `shrinkage`.
    """

    def __init__(
        self,
        reg_risk=1,
        risk="period_var",
        reg_systematic=0.0,
        market_vol=None,
        asset_vol=None,
        cov=None,
        shrinkage=0.5,
        min_names=10,
        rank_targets=False,
        eps=1e-8,
    ):
        super().__init__()

        if not isinstance(reg_risk, numbers.Real):
            raise TypeError("reg_risk must be a real number.")
        if reg_risk < 0:
            raise ValueError("reg_risk must be non-negative.")
        if not isinstance(reg_systematic, numbers.Real):
            raise TypeError("reg_systematic must be a real number.")
        if reg_systematic < 0:
            raise ValueError("reg_systematic must be non-negative.")
        if not isinstance(risk, str):
            raise TypeError("risk must be a string.")
        if risk not in {"batch_vol", "period_var", "period_cov", "period_disp"}:
            raise ValueError(
                "risk must be one of 'batch_vol', 'period_var', 'period_cov' or "
                "'period_disp'."
            )
        if not isinstance(shrinkage, numbers.Real):
            raise TypeError("shrinkage must be a real number.")
        if not (0 <= shrinkage <= 1):
            raise ValueError("shrinkage must be between 0 and 1.")

        self.reg_risk = reg_risk
        self.risk = risk
        self.reg_systematic = reg_systematic
        self.market_vol = market_vol
        self.asset_vol = asset_vol
        self.cov = cov
        self.shrinkage = shrinkage
        self.min_names = min_names
        self.rank_targets = rank_targets
        self.eps = eps

        self.ranking_loss = NegCrossSectionalIC(
            min_names=min_names, rank_targets=rank_targets, eps=eps
        )

    def _risk_term(self, y_pred, y_true_masked, mask):
        """
        Evaluate the risk penalty.

        Parameters
        ----------
        y_pred : torch.Tensor
            Network outputs. Dimension: (batch_size, n_assets).
        y_true_masked : torch.Tensor
            Realised targets with unobserved entries set to zero.
        mask : torch.Tensor
            Boolean tensor marking the observed entries of the targets.
        """
        if self.risk == "batch_vol":
            portfolio_returns = (y_pred * y_true_masked).sum(dim=1)
            if portfolio_returns.numel() < 2:
                return portfolio_returns.sum() * 0.0
            return portfolio_returns.std(unbiased=True)

        if self.risk == "period_disp":
            # Realised cross-sectional variance of each period, from that period's own
            # observed names. Available only because the targets are total returns: on
            # index-relative returns the cross-section is already centred and this
            # measures the same thing the selection term does, twice.
            counts = mask.sum(dim=1).clamp(min=1)
            means = y_true_masked.sum(dim=1) / counts
            centred = torch.where(
                mask, y_true_masked - means.unsqueeze(1), torch.zeros_like(y_true_masked)
            )
            dispersion = ((centred**2).sum(dim=1) / counts).detach()
            return ((y_pred**2).sum(dim=1) * dispersion).mean()

        if self.risk == "period_var":
            if self.asset_vol is not None:
                vol = self.asset_vol.to(y_pred.device, y_pred.dtype)
            else:
                counts = mask.sum(dim=0).clamp(min=1)
                means = y_true_masked.sum(dim=0) / counts
                centred = torch.where(mask, y_true_masked - means, torch.zeros_like(y_true_masked))
                vol = torch.sqrt((centred**2).sum(dim=0) / counts + self.eps)
                vol = vol.detach()
            return ((y_pred * vol.unsqueeze(0)) ** 2).sum(dim=1).mean()

        # "period_cov"
        if self.cov is not None:
            cov = self.cov.to(y_pred.device, y_pred.dtype)
        else:
            centred = y_true_masked - y_true_masked.mean(dim=0, keepdim=True)
            n_periods = max(y_true_masked.shape[0] - 1, 1)
            cov = (centred.T @ centred) / n_periods
            # An in-batch covariance over more assets than periods has at most
            # n_periods - 1 non-zero eigenvalues, so it is shrunk towards its diagonal
            cov = (1 - self.shrinkage) * cov + self.shrinkage * torch.diag(torch.diagonal(cov))
            cov = cov.detach()
        quadratic = (y_pred @ cov * y_pred).sum(dim=1)
        return quadratic.mean()

    def _systematic_term(self, y_pred, y_true_masked, mask):
        """
        Penalty on the book's exposure to the period's common return.

        Parameters
        ----------
        y_pred : torch.Tensor
            Network outputs. Dimension: (batch_size, n_assets).
        y_true_masked : torch.Tensor
            Realised targets with unobserved entries set to zero.
        mask : torch.Tensor
            Boolean tensor marking the observed entries of the targets.

        Notes
        -----
        With total returns the cross-sectional mean of a period *is* the equal-weighted
        market return, so the portfolio return splits exactly into

        .. code-block:: none

            r_p,t = (sum_i w_{t,i}) * m_t  +  sum_i w_{t,i} * e_{t,i}

        and the first term's variance is `net_t^2 * var(m)`. Penalising it is a direct
        charge on net exposure, scaled by how volatile the market actually was.

        `var(m)` is a scalar, so the term is `var(m) * mean_t(net_t^2)` and remains a mean
        over per-period quantities: **decomposable, hence valid under any batching mode
        and under `AssetBaggingLoss`**. Only the *estimate* of `var(m)` is batch-dependent
        when it is taken from the batch; supply `market_vol` for a causal one.
        """
        if self.market_vol is not None:
            market_var = torch.as_tensor(
                self.market_vol, device=y_pred.device, dtype=y_pred.dtype
            ) ** 2
        else:
            counts = mask.sum(dim=1).clamp(min=1)
            market = y_true_masked.sum(dim=1) / counts
            market_var = (
                market.var(unbiased=True).detach()
                if market.numel() > 1
                else torch.zeros((), device=y_pred.device, dtype=y_pred.dtype)
            )
        return (y_pred.sum(dim=1) ** 2).mean() * market_var

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
        loss = self.ranking_loss(y_pred, y_true)

        if self.reg_risk > 0 or self.reg_systematic > 0:
            mask = torch.isfinite(y_true)
            y_true_masked = torch.where(mask, y_true, torch.zeros_like(y_true))
            if self.reg_risk > 0:
                loss = loss + self.reg_risk * self._risk_term(y_pred, y_true_masked, mask)
            if self.reg_systematic > 0:
                loss = loss + self.reg_systematic * self._systematic_term(
                    y_pred, y_true_masked, mask
                )

        return loss
