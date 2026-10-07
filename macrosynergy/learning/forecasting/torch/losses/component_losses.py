import numbers

import torch
import torch.nn as nn

from sklearn.base import BaseEstimator

from .ranking_losses import NegCrossSectionalIC, _masked_cross_sectional_moments


class NegThreeComponentLoss(nn.Module, BaseEstimator):
    """
    Negative sized cross-sectional objective: a ranking loss whose months are sized by two
    causal estimates, one per remaining concern of a book.

    The book a network produces each month is split into three concerns, each with its own
    estimate and its own lever, so that each can be switched on, off or reversed without
    touching the others:

    .. code-block:: none

        weights   w_t  =  level_t  +  g_t * s_t        s_t = (y_pred_t - mean) / std over the
                                                       observed names: unit-score, level-free

        ranking                     the unit score s_t, the ONLY thing the network is asked to
                                    produce. Scored by its cross-sectional correlation with the
                                    realised returns (`form="ic"`) or by the return of the
                                    unit-score book (`form="sharpe"`)
        aggregate volatility        a causal estimate A_t of the month's aggregate risk: the
                                    geometric mean of the names' ex-ante vols, computed here from
                                    `vol`. Lever: `vol_exponent`; the month's size is A_t^(-a)
        cross-sectional dispersion  a causal forecast D_t of the month's realised cross-sectional
                                    dispersion of returns, passed in as the side input
                                    `disp_forecast`. Lever: `dispersion_exponent`; the month's
                                    size is D_t^(d): d > 0 sizes WITH the opportunity, d < 0
                                    against it

    The month's size is `g_t = A_t^(-a) * D_t^(d)`, normalised to mean 1 over the batch's usable
    months and clipped to `[1/size_clip, size_clip]` so that no month's estimate can dominate.
    Both estimates are data: they carry no gradient, and every parameter of the network enters
    through the score alone. (The net level, the market return times the book's net exposure,
    is deliberately not a component: the loss is level-free.)

    Parameters
    ----------
    form : {"ic", "sharpe"}, optional
        `"ic"` (default): `L = - sum_t g_t * IC_t / sum_t g_t`, the size-weighted mean of the
        monthly cross-sectional correlations (`NegCrossSectionalIC`'s own statistic). With
        `a = d = 0` it IS `NegCrossSectionalIC`. Needs `vol` only if `vol_exponent != 0`.

        `"sharpe"`: the ratio of the sized unit-score book, with the ex-ante risk of
        `NegSharpeRatioExAnteVol`:

        .. code-block:: none

            L = - mean_t(g_t * R_t) / sqrt(mean_t(g_t^2 * sum_i s_ti^2 * vol_ti^2) + eps),
            R_t = sum_i s_ti * r_ti

        Always needs `vol`.
    vol_exponent : float, optional
        The exponent `a` of the aggregate-vol lever. 0 (default) switches it off; 1 sizes
        inversely with aggregate vol (a vol-managed gross exposure).
    dispersion_exponent : float, optional
        The exponent `d` of the dispersion lever. 0 (default) switches it off. Non-zero needs
        the side input `disp_forecast`, a positive, causal forecast of the month's realised
        cross-sectional dispersion of returns; a month where it is missing or not positive is
        dropped from the batch.
    min_names : int, optional
        Minimum number of observed assets for a month to contribute. Default is 10.
    rank_targets : bool, optional
        Replace the targets of `form="ic"` by their within-month ranks (`NegRankIC`'s
        statistic). Default is False. Ignored by `form="sharpe"`.
    size_clip : float or None, optional
        The largest ratio, above or below the batch mean, a month's size can take. Default is
        5. `None` switches clipping off.
    eps : float, optional
        Small value guarding denominators. Default is 1e-8.

    Notes
    -----
    `requires_vol` and `side_inputs` are properties of the configuration, set at construction:
    `MLPRegressor.fit(X, y, vol=..., side=...)` reads them to know what to require. `side` is a
    frame with one column per name in `side_inputs`, indexed like `X`: per-month series, not
    per-asset ones, passed to `forward` as a tensor of shape (batch, len(side_inputs)) in that
    order.

    Why the score is standardised inside the loss: every lever is then a statement about the
    SIZE of a month, in the same units for every month, and the network cannot satisfy or
    evade a lever by rescaling its outputs. The loss is invariant to adding a constant to, or
    scaling, the outputs, month by month.

    The sizes are estimates a book could have used at the time. `vol` is the ex-ante panel the
    other volatility losses use, and `disp_forecast` must be built from information through the
    previous month; nothing here checks that, so the builder must (`mlpd.side_info` in the
    research line does, and tests it).
    """

    SIDE_DISPERSION = "disp_forecast"

    def __init__(self, form="ic", vol_exponent=0.0, dispersion_exponent=0.0, min_names=10,
                 rank_targets=False, size_clip=5.0, eps=1e-8):
        super().__init__()

        if form not in ("ic", "sharpe"):
            raise ValueError("form must be 'ic' or 'sharpe'.")
        for name, value in (("vol_exponent", vol_exponent), ("dispersion_exponent", dispersion_exponent)):
            if not isinstance(value, numbers.Real) or isinstance(value, bool):
                raise TypeError(f"{name} must be a real number.")
        if not isinstance(min_names, numbers.Integral):
            raise TypeError("min_names must be an integer.")
        if min_names < 2:
            raise ValueError("min_names must be at least 2.")
        if not isinstance(rank_targets, bool):
            raise TypeError("rank_targets must be a boolean.")
        if size_clip is not None:
            if not isinstance(size_clip, numbers.Real) or isinstance(size_clip, bool):
                raise TypeError("size_clip must be a real number or None.")
            if size_clip < 1:
                raise ValueError("size_clip must be at least 1.")
        if not isinstance(eps, numbers.Real):
            raise TypeError("eps must be a real number.")
        if eps <= 0:
            raise ValueError("eps must be positive.")

        self.form = form
        self.vol_exponent = vol_exponent
        self.dispersion_exponent = dispersion_exponent
        self.min_names = min_names
        self.rank_targets = rank_targets
        self.size_clip = size_clip
        self.eps = eps

        # What fit() must supply, a property of this configuration
        self.requires_vol = bool(form == "sharpe" or vol_exponent != 0)
        self.side_inputs = (self.SIDE_DISPERSION,) if dispersion_exponent != 0 else ()
        self._ic = NegCrossSectionalIC(min_names=min_names, rank_targets=rank_targets, eps=eps)

    # ------------------------------------------------------------------ the estimates

    def aggregate_vol(self, vol, mask):
        """The geometric mean of the observed names' ex-ante vols, per month: `exp(mean log vol)`.
        Dimension: (batch_size,). A name with a non-positive vol is not observed."""
        positive = mask & (vol > 0)
        counts = positive.sum(dim=1).clamp(min=1)
        logs = torch.where(positive, torch.log(torch.where(positive, vol, torch.ones_like(vol))),
                           torch.zeros_like(vol))
        return torch.exp(logs.sum(dim=1) / counts)

    def sizes(self, vol, side, mask, usable, dtype):
        """Each month's size `g_t`, normalised to mean 1 over the usable months and clipped;
        unusable months get 0. Carries no gradient."""
        with torch.no_grad():
            g = torch.ones(mask.shape[0], dtype=dtype, device=mask.device)
            if vol is not None and self.vol_exponent != 0:
                g = g * self.aggregate_vol(vol, mask).clamp(min=self.eps) ** (-self.vol_exponent)
            if self.dispersion_exponent != 0:
                g = g * side[:, 0].clamp(min=self.eps) ** self.dispersion_exponent
            g = torch.where(usable, g, torch.zeros_like(g))
            mean = g.sum() / usable.sum().clamp(min=1)
            g = g / mean.clamp(min=self.eps)
            if self.size_clip is not None:
                g = torch.clamp(g, 1.0 / self.size_clip, self.size_clip)
            return torch.where(usable, g, torch.zeros_like(g))

    # ------------------------------------------------------------------------- the loss

    def forward(self, y_pred, y_true, vol=None, side=None):
        """
        Evaluate the loss.

        Parameters
        ----------
        y_pred : torch.Tensor
            Network outputs. Dimension: (batch_size, n_assets).
        y_true : torch.Tensor
            Realised targets. Dimension: (batch_size, n_assets).
        vol : torch.Tensor, optional
            Ex-ante per-asset volatility, same shape as `y_true`. Required if `requires_vol`.
        side : torch.Tensor, optional
            Per-month side inputs, in the order of `side_inputs`. Dimension: (batch_size,
            len(side_inputs)). Required if `side_inputs` is not empty.
        """
        if self.requires_vol and vol is None:
            raise ValueError(
                f"{type(self).__name__}(form={self.form!r}, vol_exponent={self.vol_exponent}) requires `vol` "
                "(ex-ante per-asset volatility, same shape as y_true), passed as a keyword argument to "
                "forward(); MLPRegressor.fit(X, y, vol=...) supplies it during training."
            )
        if self.side_inputs:
            if side is None:
                raise ValueError(
                    f"{type(self).__name__} requires the side input {self.side_inputs} "
                    "(`side`, shape (batch, 1)); MLPRegressor.fit(X, y, side=...) supplies it during training."
                )
            if side.shape[0] != y_pred.shape[0] or side.shape[1] != len(self.side_inputs):
                raise ValueError(f"side must have shape ({y_pred.shape[0]}, {len(self.side_inputs)}), "
                                 f"not {tuple(side.shape)}.")

        mask = torch.isfinite(y_true)
        if vol is not None and (self.form == "sharpe" or self.vol_exponent != 0):
            mask = mask & torch.isfinite(vol)
        counts = mask.sum(dim=1)
        usable = counts >= self.min_names
        if self.side_inputs:
            usable = usable & torch.isfinite(side[:, 0]) & (side[:, 0] > 0)
        if not bool(usable.any()):
            # No month in this batch can be scored: a zero that still depends on the parameters
            # keeps the graph intact, so the step is a no-op rather than an error
            return y_pred.sum() * 0.0

        g = self.sizes(vol, side, mask, usable, y_pred.dtype)

        if self.form == "ic":
            ic, ic_usable = self._ic.period_ic(y_pred, y_true if vol is None else torch.where(
                mask, y_true, torch.full_like(y_true, float("nan"))))
            weights = torch.where(usable & ic_usable, g, torch.zeros_like(g))
            return -(weights * ic).sum() / weights.sum().clamp(min=self.eps)

        # form == "sharpe": the ratio of the sized unit-score book
        r = torch.where(mask, y_true, torch.zeros_like(y_true))
        sigma = torch.where(mask, vol, torch.zeros_like(vol))
        centred = _masked_cross_sectional_moments(y_pred, mask, counts)
        spread = torch.sqrt(((centred ** 2).sum(dim=1) / counts.clamp(min=1)).clamp(min=self.eps))
        s = centred / spread.unsqueeze(1)
        reward = (s * r).sum(dim=1)
        variance = ((s ** 2) * (sigma ** 2)).sum(dim=1)
        n = usable.sum().clamp(min=1)
        numerator = (g * reward).sum() / n
        risk = torch.sqrt(((g ** 2) * variance).sum() / n + self.eps)
        return -numerator / risk
