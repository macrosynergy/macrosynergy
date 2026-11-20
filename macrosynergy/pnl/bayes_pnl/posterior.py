"""Numerical-backend contract and shared posterior reports."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd
from scipy import optimize

from .inputs import probabilities, sample_size
from .likelihoods import get_likelihood

if TYPE_CHECKING:
    from .grid import GridPosterior


def bracketed_quantile(function, center, width):
    """Solve a monotone CDF residual in coordinates scaled to its spread."""
    width = max(float(width), np.finfo(float).tiny ** 0.25)
    for _ in range(50):
        lo, hi = center - 8 * width, center + 8 * width
        if function(lo) <= 0 and function(hi) >= 0:
            return optimize.brentq(function, lo, hi, xtol=1e-12 * width, rtol=1e-12)
        width *= 2
    raise ValueError("Could not bracket quantile within finite precision.")


class Posterior(ABC):
    """Inference for one observed period metric.

    Parameter names are ``mean`` and ``sd``. Predictive evaluations and quantiles
    return one-dimensional NumPy arrays; parameter quantiles/CDFs preserve scalar
    input. ``sample`` returns mean/sd columns. ``to_grid`` explicitly requests a
    finite approximation for visualization, never needed by shared inference.
    """

    data: np.ndarray
    prior: object
    metric: str
    period_years: float
    likelihood_weight: float
    log_evidence: Optional[float]

    @abstractmethod
    def expectation(self, param):
        """Posterior expectation of a parameter, in metric units."""

    @abstractmethod
    def std(self, param):
        """Posterior standard deviation of a parameter."""

    @abstractmethod
    def cdf(self, param, value):
        """Parameter marginal CDF."""

    @abstractmethod
    def quantile(self, param, q):
        """Parameter marginal quantiles."""

    @abstractmethod
    def interval(self, param, level=0.9, kind="eti"):
        """Parameter credible interval as a pair of floats."""

    @abstractmethod
    def mode(self):
        """Joint mean/sd mode, or a zero-sd boundary limit."""

    @abstractmethod
    def density(self, param, values):
        """Parameter density at explicit coordinates, as a 1D array."""

    @abstractmethod
    def predictive_cdf(self, x):
        """CDF of the next same-duration observed metric."""

    @abstractmethod
    def predictive_pdf(self, x):
        """Density of the next same-duration observed metric."""

    @abstractmethod
    def sample(self, n, rng=None):
        """Parameter draws as a DataFrame with mean and sd columns."""

    @abstractmethod
    def to_grid(self) -> "GridPosterior":
        """Return a finite GridPosterior for visualization."""

    def predictive_quantile(self, q):
        center = self.expectation("mean")
        width = max(self.expectation("sd"), self.std("mean"))
        result = []
        for p in probabilities(q):
            if p in (0, 1):
                result.append(get_likelihood(self.metric).support[int(p)])
            else:
                # Solve near zero even when the metric uses large location units.
                root = bracketed_quantile(
                    lambda z: self.predictive_cdf(center + width * z)[0] - p,
                    0.0,
                    1.0,
                )
                result.append(center + width * root)
        return np.asarray(result)

    def predictive_sample(self, n, rng=None):
        sample_size(n)
        rng = np.random.default_rng(rng)
        draws = self.sample(n, rng=rng)
        return get_likelihood(self.metric).sample(
            draws["mean"].to_numpy(), draws["sd"].to_numpy(), rng
        )

    def prob_mean_below(self, threshold=0.0):
        return float(self.cdf("mean", threshold))

    def _summary_diagnostics(self):
        return {}

    def summary(
        self,
        level=0.9,
        thresholds=(0.0,),
        predictive_q=(0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95),
    ):
        """Human-readable parameter summaries, predictions and backend diagnostics."""
        rows = {}
        for param in ("mean", "sd"):
            rows[param + ": posterior mean"] = self.expectation(param)
            rows[param + ": posterior median"] = float(self.quantile(param, 0.5))
            rows[param + ": posterior sd"] = self.std(param)
            for kind in ("eti", "hdi"):
                lo, hi = self.interval(param, level, kind)
                rows["{}: {:.0%} {}".format(param, level, kind.upper())] = (
                    "[{:.3f}, {:.3f}]".format(lo, hi)
                )
        m, s = self.mode()
        rows["joint mode (mean, sd)"] = (
            "({:.3f}, {:.3f})".format(m, s)
            if s > 0
            else "boundary limit ({:.3f}, 0)".format(m)
        )
        for t in thresholds:
            rows["Pr(mean < {:g})".format(t)] = self.prob_mean_below(t)
            rows["Pr(next period < {:g})".format(t)] = float(self.predictive_cdf(t)[0])
        for q, value in zip(predictive_q, self.predictive_quantile(predictive_q)):
            rows["predictive q{:02d}".format(int(round(q * 100)))] = float(value)
        rows.update(self._summary_diagnostics())
        return pd.Series(rows, name="value").to_frame()
