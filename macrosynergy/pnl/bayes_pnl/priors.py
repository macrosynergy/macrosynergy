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
N(0, V) with sd ~ (V, 1.5 V). Annual volatility has a positive log-normal
mean prior with arithmetic mean V and sd 0.5 V, and a positive dispersion
prior with arithmetic mean 0.2 V and sd 0.2 V.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Union

import numpy as np
from scipy import optimize, special, stats

METRICS = ("sharpe", "ann_ret", "ann_vol")

SDKind = str


def _log_sd_moment_ratio(shape_minus_one: float) -> float:
    """Log of E[SD]**2 / E[SD**2] for inverse-gamma variance.

    Expressing the shape as ``b = alpha - 1`` avoids cancellation near one.
    For large b, the log-gamma ratio has a Stirling expansion whose leading
    term is -1/(4*b); subtracting two gammaln values loses that small term.
    """
    b = shape_minus_one
    if b >= 32:
        inverse = 1.0 / b
        return inverse * (
            -0.25
            + inverse**2
            * (1.0 / 96 + inverse**2 * (-1.0 / 320 + inverse**2 * 17 / 7168))
        )
    return float(2 * (special.gammaln(b + 0.5) - special.gammaln(b)) - np.log(b))


def _invgamma_alpha_for_sd_moments(E: float, s: float) -> float:
    """
    Shape alpha of an inverse-gamma prior on the variance such that the implied
    prior on sd = sqrt(variance) has mean E and standard deviation s.

    For V ~ IG(alpha, beta): E[sqrt V] = sqrt(beta) G(alpha - 1/2) / G(alpha) and
    E[V] = beta / (alpha - 1), so E[sd]^2 / E[sd^2] depends on alpha only.
    """
    log_cv = np.log(s) - np.log(E)
    target = float(np.logaddexp(0.0, 2 * log_cv))
    if target == 0:
        raise ValueError("SDPrior moments exceed finite numerical precision.")

    def f(log_b):
        return _log_sd_moment_ratio(float(np.exp(log_b))) + target

    # Solve on a logarithmic shape axis, starting from the concentrated-prior
    # approximation and widening the bracket only as needed.
    minimum = float(np.log(np.nextafter(1.0, np.inf) - 1.0))
    maximum = float(np.log(np.finfo(float).max))
    center = float(np.clip(-2 * log_cv - np.log(4.0), minimum, maximum))
    radius = 1.0
    while True:
        lower, upper = max(minimum, center - radius), min(maximum, center + radius)
        if f(lower) <= 0 <= f(upper):
            break
        if lower == minimum and upper == maximum:
            raise ValueError("SDPrior moments exceed finite numerical precision.")
        radius *= 2
    log_b = optimize.brentq(f, lower, upper, xtol=1e-13, rtol=1e-14)
    alpha = float(1.0 + np.exp(log_b))
    if abs(_log_sd_moment_ratio(alpha - 1.0) + target) > 1e-6:
        raise ValueError("SDPrior moments exceed finite numerical precision.")
    return alpha


def _lognormal_parameters(mean: float, sd: float):
    """Convert arithmetic moments to the location and scale of log values."""
    variance = float(np.logaddexp(0.0, 2 * (np.log(sd) - np.log(mean))))
    if variance == 0:
        raise ValueError("Log-normal moments exceed finite numerical precision.")
    return float(np.log(mean) - variance / 2), float(np.sqrt(variance))


@dataclass(frozen=True)
class NormalPrior:
    """Normal prior on the mean of the period metric."""

    mean: float
    sd: float

    normalised = True

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
class LogNormalPrior:
    """Positive mean prior, specified by its arithmetic mean and sd.

    ``mean`` and ``sd`` use the metric's original units, rather than the
    parameters of the logarithm. This keeps prior diagnostics and changes of
    PnL units consistent with NormalPrior.
    """

    mean: float
    sd: float
    _log_location: float = field(init=False, repr=False)
    _log_scale: float = field(init=False, repr=False)

    normalised = True

    def __post_init__(self):
        if not np.isfinite([self.mean, self.sd]).all() or not (
            self.mean > 0 and self.sd > 0
        ):
            raise ValueError("LogNormalPrior mean and sd must be positive and finite.")
        location, scale = _lognormal_parameters(self.mean, self.sd)
        object.__setattr__(self, "_log_location", location)
        object.__setattr__(self, "_log_scale", scale)

    def logpdf(self, mu):
        return stats.lognorm.logpdf(
            mu, s=self._log_scale, scale=np.exp(self._log_location)
        )

    def ppf(self, q):
        return stats.lognorm.ppf(q, s=self._log_scale, scale=np.exp(self._log_location))

    def cdf(self, mu):
        return stats.lognorm.cdf(
            mu, s=self._log_scale, scale=np.exp(self._log_location)
        )


@dataclass(frozen=True)
class SDPrior:
    """
    Prior on the standard deviation (dispersion) of the period metric,
    parameterised by the prior mean ``mean`` and prior sd ``sd`` *of the SD*.

    Parameters
    ----------
    mean : float
        Positive arithmetic mean of the dispersion, in metric units.
    sd : float
        Positive standard deviation of the dispersion, in metric units.
    kind : {"invgamma_var", "gamma_sd", "lognormal_sd"}
        The default inverse-gamma variance prior is moment-matched to the
        requested SD moments and includes the variance-to-SD Jacobian.
        Alternatives put a gamma or log-normal prior directly on SD.
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
        log_mean, log_sd = np.log(E), np.log(s)
        p = self._params
        with np.errstate(over="ignore", under="ignore"):
            if self.kind == "invgamma_var":
                a = _invgamma_alpha_for_sd_moments(E, s)
                p["alpha"] = a
                p["beta"] = np.exp(
                    np.logaddexp(2 * log_mean, 2 * log_sd) + np.log(a - 1.0)
                )
            elif self.kind == "gamma_sd":
                p["shape"] = np.exp(2 * (log_mean - log_sd))
                p["rate"] = np.exp(log_mean - 2 * log_sd)
            elif self.kind == "lognormal_sd":
                p["mu"], p["sigma"] = _lognormal_parameters(E, s)
            else:
                raise ValueError(f"Unknown SDPrior kind {self.kind!r}")
        if not np.isfinite(list(p.values())).all() or any(
            value <= 0 for key, value in p.items() if key != "mu"
        ):
            raise ValueError("SDPrior moments exceed finite numerical precision.")

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

    mean: Union[NormalPrior, LogNormalPrior]
    sd: SDPrior

    def logpdf(self, mu, sd):
        return self.mean.logpdf(mu) + self.sd.logpdf(sd)

    @property
    def normalised(self) -> bool:
        return bool(
            getattr(self.mean, "normalised", False)
            and getattr(self.sd, "normalised", False)
        )


def default_prior(metric: str = "sharpe", vol_target: float = 10.0) -> IndependentPrior:
    """
    Weakly informative prior for the mean and sd of a period metric.

    Sharpe and return expectation priors are centred at zero (no expected
    value). This is a continuous prior, not a point mass at zero.

    Parameters
    ----------
    metric : {"sharpe", "ann_ret", "ann_vol"}
        Annualised Sharpe ratio, return or volatility. Return and volatility
        retain the PnL units.
    vol_target : float
        Annual volatility reference in PnL units (10 for percent, 0.10 for
        decimal returns). Scales return and volatility priors.
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
        return IndependentPrior(LogNormalPrior(V, 0.5 * V), SDPrior(0.2 * V, 0.2 * V))
    raise ValueError(f"Unknown metric {metric!r}; choose from {METRICS}.")
