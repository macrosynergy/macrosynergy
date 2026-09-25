import numbers

import torch
import torch.nn as nn

from sklearn.base import BaseEstimator


class AssetBaggingLoss(nn.Module, BaseEstimator):
    """
    Wrap a portfolio loss so that each evaluation scores a random subset of the assets.

    Parameters
    ----------
    loss_func : torch.nn.Module
        The loss to wrap. Called once per draw on the subsetted weights and returns.
    asset_fraction : float, optional
        Fraction of assets retained in each draw, in (0, 1]. Default is 0.75.
    draws : int, optional
        Independent asset subsets averaged over per call. Default is 1.
    renormalize : str, optional
        How the weights of a subset are rescaled: "auto", "l1", "sum" or "none". Default
        is "auto", which infers the constraint from the full weight row.
    min_assets : int, optional
        Floor on the number of assets retained, so a small cross-section is not reduced
        to a degenerate one. Default is 2.
    generator : torch.Generator, optional
        Source of randomness for the draws. Default is None, which uses the ambient torch
        random state and so is controlled by the estimator's `random_state`.

    Notes
    -----
    **What this is.** The multi-head panel model presents one row per period with every
    asset as a column, so the loss sees the complete cross-section at every step. This
    wrapper instead forms the portfolio from a random fraction of the names, redrawn each
    step. Averaged over training that is bagging over the cross-section: the gradient
    becomes an average over subsets rather than a single draw from the full panel, which
    is the Michaud resampling argument applied to the objective rather than to the
    optimiser's inputs. It is the loss-side counterpart of `PanelBatchSampler(mode="mc")`,
    which does the same thing by rows for the shared-head layout.

    **Why the subset is shared across the batch's periods.** One draw is taken per call
    and applied to every row, rather than redrawn per row. `NegSharpeRatio` and its
    relatives collapse assets into one portfolio return per period and then take a mean
    and a standard deviation *over the batch's periods*. If the asset subset changed from
    period to period, that standard deviation would be taken over a sequence of different
    portfolios and would measure the resampling noise rather than the portfolio's
    volatility. Holding the subset fixed across the block keeps the portfolio's identity
    constant through time, which is the quantity the ratio is about.

    **Why the weights are renormalized.** The signal modifier imposes its constraint over
    the full cross-section: `LongShortModule` gives `sum(|w|) = 1` and optionally
    `sum(w) = 0`, and a softmax head gives `sum(w) = 1` with `w >= 0`. Dropping columns
    breaks whichever constraint was imposed, leaving a book that is smaller than unit
    gross and, for a dollar-neutral one, no longer neutral -- so the subsetted portfolio
    would carry a net exposure that the full one did not, and the loss would score market
    drift rather than the signal. "auto" detects the constraint from the full row and
    restores it on the subset, re-centering first where the full book was dollar-neutral.

    **What this does not fix.** `PortfolioLoss.forward` zero-fills non-finite returns
    without renormalizing the weights, so a position in a name with no observed return is
    scored as returning exactly zero. That bias is inherited here, and is a Phase B
    prerequisite rather than something this wrapper addresses.
    """

    def __init__(
        self,
        loss_func,
        asset_fraction=0.75,
        draws=1,
        renormalize="auto",
        min_assets=2,
        generator=None,
    ):
        super().__init__()

        # Checks
        if not isinstance(loss_func, nn.Module):
            raise TypeError("loss_func must be a torch.nn.Module.")
        if not isinstance(asset_fraction, numbers.Number) or isinstance(asset_fraction, bool):
            raise TypeError("asset_fraction must be a number.")
        if not 0 < asset_fraction <= 1:
            raise ValueError("asset_fraction must lie in (0, 1].")
        if not isinstance(draws, numbers.Integral) or isinstance(draws, bool):
            raise TypeError("draws must be an integer.")
        if draws < 1:
            raise ValueError("draws must be positive.")
        if renormalize not in ("auto", "l1", "sum", "none"):
            raise ValueError("renormalize must be one of 'auto', 'l1', 'sum' or 'none'.")
        if not isinstance(min_assets, numbers.Integral) or isinstance(min_assets, bool):
            raise TypeError("min_assets must be an integer.")
        if min_assets < 1:
            raise ValueError("min_assets must be positive.")

        self.loss_func = loss_func
        self.asset_fraction = asset_fraction
        self.draws = draws
        self.renormalize = renormalize
        self.min_assets = min_assets
        self.generator = generator

    def forward(self, y_pred, y_true):
        """
        Average the wrapped loss over independent asset subsets.

        Parameters
        ----------
        y_pred : torch.Tensor
            Predicted portfolio weights. Dimension: (batch_size, n_assets)
        y_true : torch.Tensor
            True asset returns. Dimension: (batch_size, n_assets)
        """
        n_assets = y_pred.shape[1]
        n_draw = max(self.min_assets, int(round(self.asset_fraction * n_assets)))
        if n_draw >= n_assets:
            return self.loss_func(y_pred, y_true)

        scheme = self._detect_scheme(y_pred) if self.renormalize == "auto" else self.renormalize

        total = None
        for _ in range(self.draws):
            keep = torch.randperm(n_assets, generator=self.generator, device=y_pred.device)
            keep = keep[:n_draw]
            weights = self._rescale(y_pred[:, keep], scheme)
            loss = self.loss_func(weights, y_true[:, keep])
            total = loss if total is None else total + loss

        return total / self.draws

    def _detect_scheme(self, y_pred):
        """
        Infer the constraint the signal modifier imposed, from the full weight row.

        Detection is on the full cross-section and before any subsetting, so it reads the
        constraint as the modifier left it. A row that satisfies neither convention --
        an unconstrained head -- is left alone.
        """
        with torch.no_grad():
            row_sum = y_pred.sum(dim=1)
            gross = y_pred.abs().sum(dim=1)
            nonnegative = bool((y_pred >= 0).all())
            unit_sum = bool(torch.allclose(row_sum, torch.ones_like(row_sum), atol=1e-4))
            unit_gross = bool(torch.allclose(gross, torch.ones_like(gross), atol=1e-4))
            neutral = bool(torch.allclose(row_sum, torch.zeros_like(row_sum), atol=1e-4))

        if nonnegative and unit_sum:
            return "sum"
        if unit_gross:
            return "l1_neutral" if neutral else "l1"
        return "none"

    def _rescale(self, weights, scheme):
        """Restore the detected constraint on a subset of the columns."""
        if scheme == "none":
            return weights
        if scheme == "l1_neutral":
            # Re-center before rescaling: dropping columns leaves a net exposure that the
            # full book did not have, and rescaling alone would preserve it.
            weights = weights - weights.mean(dim=1, keepdim=True)
            denominator = weights.abs().sum(dim=1, keepdim=True)
        elif scheme == "l1":
            denominator = weights.abs().sum(dim=1, keepdim=True)
        else:
            denominator = weights.sum(dim=1, keepdim=True)

        return weights / denominator.clamp(min=1e-12)
