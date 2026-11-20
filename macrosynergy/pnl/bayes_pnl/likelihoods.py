"""Observation models parameterized by a metric's arithmetic mean and sd."""

import numpy as np
from scipy import special, stats


class NormalMetric:
    """Real-valued observed Sharpe ratios or annual returns."""

    support = (-np.inf, np.inf)

    @staticmethod
    def log_likelihood(inputs, mean, sd):
        n = inputs.weighted_n
        return (
            -n * np.log(sd)
            - 0.5 * n * np.log(2 * np.pi)
            - (
                inputs.likelihood_weight * inputs.scatter
                + n * (inputs.average - mean) ** 2
            )
            / (2 * sd**2)
        )

    @staticmethod
    def cdf(x, mean, sd):
        return special.ndtr((x - mean) / sd)

    @staticmethod
    def pdf(x, mean, sd):
        return stats.norm.pdf(x, mean, sd)

    @staticmethod
    def sample(mean, sd, rng):
        return rng.normal(mean, sd)


class GammaMetric:
    """Positive observed annual volatility, with positive arithmetic mean/sd."""

    support = (0.0, np.inf)

    @staticmethod
    def log_likelihood(inputs, mean, sd):
        valid = mean > 0
        safe_mean = np.where(valid, mean, 1.0)
        shape = (safe_mean / sd) ** 2
        rate = safe_mean / sd**2
        # Gamma sufficient statistics include sum(log x), not squared residuals.
        n = inputs.weighted_n
        result = (
            n * (shape * np.log(rate) - special.gammaln(shape))
            + (shape - 1) * inputs.likelihood_weight * np.log(inputs.data).sum()
            - rate * n * inputs.average
        )
        return np.where(valid, result, -np.inf)

    @staticmethod
    def cdf(x, mean, sd):
        return stats.gamma.cdf(x, a=(mean / sd) ** 2, scale=sd**2 / mean)

    @staticmethod
    def pdf(x, mean, sd):
        return stats.gamma.pdf(x, a=(mean / sd) ** 2, scale=sd**2 / mean)

    @staticmethod
    def sample(mean, sd, rng):
        return rng.gamma((mean / sd) ** 2, sd**2 / mean)


def get_likelihood(metric):
    """Return the observation law in the same units as the supplied metrics."""
    if metric == "ann_vol":
        return GammaMetric
    if metric in ("sharpe", "ann_ret"):
        return NormalMetric
    raise ValueError("Unknown metric {!r}.".format(metric))
