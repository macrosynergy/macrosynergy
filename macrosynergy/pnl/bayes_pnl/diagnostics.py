"""Simulation diagnostics for assumptions about observed period metrics.

These functions return data, independently of plotting and posterior reporting.
"""

import numpy as np
import pandas as pd

from .inputs import sample_size
from .likelihoods import get_likelihood
from .posterior import Posterior
from .priors import IndependentPrior, LogNormalPrior, NormalPrior, SDPrior


def prior_predictive_sample(prior, n, rng=None, metric="sharpe"):
    """Draw possible period metrics before observing performance data.

    Parameters
    ----------
    prior : IndependentPrior
        Exact built-in independent prior with SDPrior dispersion. Sharpe and
        return metrics require NormalPrior for the mean; volatility requires
        LogNormalPrior for its positive mean. Arbitrary joint priors require
        their own sampling implementation.
    n : int
        Nonnegative number of independently drawn prior-predictive outcomes.
    rng : int or numpy.random.Generator, optional
        Seed or random generator.
    metric : {"sharpe", "ann_ret", "ann_vol"}
        Selects the observation distribution, in natural metric units.

    Returns
    -------
    numpy.ndarray
        One-dimensional array of length n. Each outcome uses a fresh pair of
        prior parameter draws, rather than sharing one pair across the sample.
    """
    sample_size(n)
    likelihood = get_likelihood(metric)
    mean_type = LogNormalPrior if metric == "ann_vol" else NormalPrior
    if (
        type(prior) is not IndependentPrior
        or type(prior.mean) is not mean_type
        or type(prior.sd) is not SDPrior
    ):
        raise TypeError(
            "prior_predictive_sample requires built-in independent priors "
            "with a mean prior appropriate for the metric."
        )
    rng = np.random.default_rng(rng)
    # Generator.random excludes one; guard its possible zero before ppf.
    uniforms = np.maximum(rng.random((2, n)), np.nextafter(0.0, 1.0))
    means, deviations = prior.mean.ppf(uniforms[0]), prior.sd.ppf(uniforms[1])
    return np.asarray(likelihood.sample(means, deviations, rng))


def posterior_predictive_check(fit, repetitions=1000, rng=None):
    """Compare observed data summaries with replicated period histories.

    Each replicated history has the same observation count as the original and
    shares one posterior parameter draw, matching the conditional-iid model.
    Replicates retain parameter uncertainty and observed performance variation.

    Parameters
    ----------
    fit : BayesianPnL or Posterior
        Fitted model or numerical posterior. No visualization grid is requested.
    repetitions : int
        Positive number of replicated histories, default 1000.
    rng : int or numpy.random.Generator, optional
        Seed or random generator.

    Returns
    -------
    pandas.DataFrame
        Rows are mean, min, max, and sample sd (the latter for n > 1).
        Columns are observed, predictive_lo, predictive_median, predictive_hi,
        and tail_probability. Bounds are the 5th and 95th percentiles of
        replicated summaries. Tail probability is twice the smaller empirical
        fraction at or beyond the observed summary, capped at one. It is a
        descriptive posterior-predictive discrepancy, not a hypothesis-test
        p-value or proof of calibration. The fitted data are reused in fitting
        and checking; use held-out periods for predictive validation.
    """
    from .model import BayesianPnL

    if isinstance(fit, BayesianPnL):
        post = fit.posterior
    elif isinstance(fit, Posterior):
        post = fit
    else:
        raise TypeError("fit must be a BayesianPnL or Posterior.")
    sample_size(repetitions)
    if repetitions == 0:
        raise ValueError("repetitions must be a positive integer.")
    rng = np.random.default_rng(rng)
    draws = post.sample(repetitions, rng=rng)
    data = np.asarray(post.data)
    shape = (repetitions, len(data))
    means = np.broadcast_to(draws["mean"].to_numpy()[:, None], shape)
    deviations = np.broadcast_to(draws["sd"].to_numpy()[:, None], shape)
    replicated = get_likelihood(post.metric).sample(means, deviations, rng)
    summaries = {
        "mean": (float(data.mean()), replicated.mean(axis=1)),
        "min": (float(data.min()), replicated.min(axis=1)),
        "max": (float(data.max()), replicated.max(axis=1)),
    }
    if len(data) > 1:
        summaries["sd"] = (float(data.std(ddof=1)), replicated.std(axis=1, ddof=1))
    rows = {}
    for name, (observed, predicted) in summaries.items():
        lo, median, hi = np.quantile(predicted, [0.05, 0.5, 0.95])
        tail = min(np.mean(predicted <= observed), np.mean(predicted >= observed))
        rows[name] = {
            "observed": observed,
            "predictive_lo": lo,
            "predictive_median": median,
            "predictive_hi": hi,
            "tail_probability": min(1.0, 2 * tail),
        }
    result = pd.DataFrame.from_dict(rows, orient="index")
    result.index.name = "statistic"
    result.attrs.update(
        repetitions=repetitions,
        n_observations=len(data),
        metric=post.metric,
        period_years=post.period_years,
    )
    return result
