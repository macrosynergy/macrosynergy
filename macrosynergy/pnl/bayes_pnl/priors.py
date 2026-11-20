"""
Priors for the (mean, sd) of a period performance metric.

All priors expose ``logpdf`` evaluated on the *sd* (not variance/precision)
scale, so a grid over (mean, sd) can be used for every prior. Densities are
only required up to a constant for the grid posterior; ``normalised`` says
whether the constant is included (needed for marginal likelihoods / Bayes
factors in later extensions).

Default priors per metric
-------------------------
``default_prior(metric, vol_target)`` gives weakly informative priors for the
three period metrics, scaled consistently: a long-term Sharpe prior N(0, 1) with
sd ~ (1.0, 1.5) becomes, for annualised returns in % at ``vol_target``% vol,
N(0, V) with sd ~ (V, 1.5 V); realised annual volatility gets N(V, 0.5 V)
with sd ~ (0.2 V, 0.2 V).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np
from scipy import optimize, special, stats

METRICS = ("sharpe", "ann_ret", "ann_vol")

SDKind = str


def _invgamma_alpha_for_sd_moments(E: float, s: float) -> float:
    """
    Shape alpha of an inverse-gamma prior on the variance such that the implied
    prior on sd = sqrt(variance) has mean E and standard deviation s.

    For V ~ IG(alpha, beta): E[sqrt V] = sqrt(beta) G(alpha - 1/2) / G(alpha) and
    E[V] = beta / (alpha - 1), so E[sd]^2 / E[sd^2] depends on alpha only.
    """
    target = E**2 / (E**2 + s**2)

    def f(a):
        return (a - 1.0) * np.exp(
            2 * (special.gammaln(a - 0.5) - special.gammaln(a))
        ) - target

    return float(optimize.brentq(f, 1.0 + 1e-12, 1e9))


@dataclass(frozen=True)
class NormalPrior:
    """Normal prior on the mean of the period metric."""

    mean: float
    sd: float

    def __post_init__(self):
        if not np.isfinite(self.mean) or not np.isfinite(self.sd) or not self.sd > 0:
            raise ValueError(
                "NormalPrior mean must be finite and sd positive and finite."
            )

    def logpdf(self, mu):
        return stats.norm.logpdf(mu, self.mean, self.sd)

    def ppf(self, q):
        return stats.norm.ppf(q, self.mean, self.sd)

    def cdf(self, mu):
        return stats.norm.cdf(mu, self.mean, self.sd)


@dataclass(frozen=True)
class SDPrior:
    """
    Prior on the standard deviation (dispersion) of the period metric,
    parameterised by the prior mean ``mean`` and prior sd ``sd`` *of the SD*.

    kind
        ``"invgamma_var"`` inverse-gamma on the variance (= gamma on precision with a
                           *rate*), moment-matched so that the SD has the requested
                           mean and sd; Jacobian applied. Default.
        ``"gamma_sd"``     gamma directly on the SD (shape E^2/s^2, rate E/s^2).
        ``"lognormal_sd"`` log-normal on the SD with matching mean and sd.
    """

    mean: float
    sd: float
    kind: SDKind = "invgamma_var"
    _params: dict = field(default_factory=dict, init=False, repr=False, compare=False)

    def __post_init__(self):
        if not np.isfinite([self.mean, self.sd]).all() or not (
            self.mean > 0 and self.sd > 0
        ):
            raise ValueError("SDPrior mean and sd must be positive and finite.")
        E, s = self.mean, self.sd
        p = self._params
        if self.kind == "invgamma_var":
            a = _invgamma_alpha_for_sd_moments(E, s)
            p["alpha"], p["beta"] = a, (E**2 + s**2) * (a - 1.0)
        elif self.kind == "gamma_sd":
            p["shape"], p["rate"] = E**2 / s**2, E / s**2
        elif self.kind == "lognormal_sd":
            v = np.log1p(s**2 / E**2)
            p["mu"], p["sigma"] = np.log(E) - v / 2, np.sqrt(v)
        else:
            raise ValueError(f"Unknown SDPrior kind {self.kind!r}")

    @property
    def params(self) -> Dict[str, float]:
        return dict(self._params)

    normalised = True

    def logpdf(self, sd):
        values = np.asarray(sd, dtype=float)
        sd = np.where(values > 0, values, 1.0)
        p = self._params
        if self.kind == "invgamma_var":
            density = stats.invgamma.logpdf(
                sd**2, a=p["alpha"], scale=p["beta"]
            ) + np.log(2.0 * sd)
        elif self.kind == "gamma_sd":
            density = stats.gamma.logpdf(sd, a=p["shape"], scale=1.0 / p["rate"])
        else:
            density = stats.lognorm.logpdf(sd, s=p["sigma"], scale=np.exp(p["mu"]))
        return np.where(values > 0, density, -np.inf)

    def ppf(self, q):
        """Prior quantiles of the SD."""
        p, q = self._params, np.asarray(q, dtype=float)
        if self.kind == "invgamma_var":
            return np.sqrt(stats.invgamma.ppf(q, a=p["alpha"], scale=p["beta"]))
        if self.kind == "gamma_sd":
            return stats.gamma.ppf(q, a=p["shape"], scale=1.0 / p["rate"])
        return stats.lognorm.ppf(q, s=p["sigma"], scale=np.exp(p["mu"]))


@dataclass(frozen=True)
class IndependentPrior:
    """p(mean, sd) = p(mean) p(sd)."""

    mean: NormalPrior
    sd: SDPrior

    def logpdf(self, mu, sd):
        return self.mean.logpdf(mu) + self.sd.logpdf(sd)

    @property
    def normalised(self) -> bool:
        return self.sd.normalised


def default_prior(metric: str = "sharpe", vol_target: float = 10.0) -> IndependentPrior:
    """
    Weakly informative prior for the mean and sd of a period metric.

    Sharpe and return expectation priors are centred at zero (no expected
    value). This is a continuous prior, not a point mass at zero.

    metric : "sharpe" (annualised Sharpe ratio), "ann_ret" (annualised return,
        % of AUM) or "ann_vol" (annualised volatility, %).
    vol_target : the strategy's annual volatility in % (scales "ann_ret" and "ann_vol").
    """
    V = float(vol_target)
    if not np.isfinite(V) or V <= 0:
        raise ValueError(
            "vol_target must be positive and finite, in the same units as the PnL."
        )
    if metric == "sharpe":
        return IndependentPrior(NormalPrior(0.0, 1.0), SDPrior(1.0, 1.5))
    if metric == "ann_ret":
        return IndependentPrior(NormalPrior(0.0, V), SDPrior(V, 1.5 * V))
    if metric == "ann_vol":
        return IndependentPrior(NormalPrior(V, 0.5 * V), SDPrior(0.2 * V, 0.2 * V))
    raise ValueError(f"Unknown metric {metric!r}; choose from {METRICS}.")
