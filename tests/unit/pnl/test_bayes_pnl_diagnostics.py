"""Simulation checks for period-metric model assumptions."""

import numpy as np
import pandas as pd
import pytest

from macrosynergy.pnl import BayesianPnL
from macrosynergy.pnl.bayes_pnl.diagnostics import (
    posterior_predictive_check,
    prior_predictive_sample,
)
from macrosynergy.pnl.bayes_pnl import IndependentPrior, NormalPrior, SDPrior


def test_prior_predictive_samples_have_the_total_mean_and_variance():
    prior = IndependentPrior(NormalPrior(0.3, 0.2), SDPrior(0.7, 0.1, "gamma_sd"))
    draws = prior_predictive_sample(prior, 40000, rng=17)
    assert draws.shape == (40000,)
    assert draws.mean() == pytest.approx(0.3, abs=0.015)
    # Var(X) = Var(mu) + E(sd**2).
    assert draws.var() == pytest.approx(0.2**2 + 0.7**2 + 0.1**2, abs=0.02)
    np.testing.assert_array_equal(draws, prior_predictive_sample(prior, 40000, rng=17))


def test_prior_predictive_zero_draws_are_empty():
    prior = IndependentPrior(NormalPrior(0, 1), SDPrior(1, 0.5))
    assert prior_predictive_sample(prior, 0, rng=3).shape == (0,)


@pytest.mark.parametrize("count", [-1, True, 1.5])
def test_prior_predictive_rejects_invalid_sample_counts(count):
    prior = IndependentPrior(NormalPrior(0, 1), SDPrior(1, 0.5))
    with pytest.raises(ValueError):
        prior_predictive_sample(prior, count)


def test_prior_predictive_requires_a_supported_sampling_prior():
    with pytest.raises(TypeError, match="built-in independent"):
        prior_predictive_sample(object(), 10)
    prior = IndependentPrior(NormalPrior(0, 1), SDPrior(1, 0.5))
    with pytest.raises(TypeError, match="appropriate for the metric"):
        prior_predictive_sample(prior, 10, metric="ann_vol")
    with pytest.raises(ValueError):
        prior_predictive_sample(prior, 10, metric="unknown")


def test_positive_prior_predictive_draws_preserve_volatility_support():
    from macrosynergy.pnl.bayes_pnl.priors import LogNormalPrior

    prior = IndependentPrior(LogNormalPrior(10, 2), SDPrior(2, 0.5))
    draws = prior_predictive_sample(prior, 20000, rng=17, metric="ann_vol")
    assert np.all(draws > 0)
    assert draws.mean() == pytest.approx(10, abs=0.1)
    assert draws.var() == pytest.approx(2**2 + 2**2 + 0.5**2, abs=0.3)


def test_predictive_check_detects_extreme_observed_shape_without_plotting(monkeypatch):
    fit = BayesianPnL(np.r_[np.zeros(99), 10.0])
    monkeypatch.setattr(
        fit.posterior, "to_grid", lambda: pytest.fail("plot grid requested")
    )
    check = posterior_predictive_check(fit, repetitions=2000, rng=71)
    assert list(check.index) == ["mean", "min", "max", "sd"]
    assert check.loc["max", "observed"] == 10
    assert check.loc["max", "predictive_hi"] < 10
    assert check.loc["max", "tail_probability"] < 0.01
    assert check.loc["mean", "predictive_lo"] < 0.1 < check.loc["mean", "predictive_hi"]
    assert check.tail_probability.between(0, 1).all()
    pd.testing.assert_frame_equal(
        check, posterior_predictive_check(fit.posterior, repetitions=2000, rng=71)
    )
    assert check.attrs["n_observations"] == 100


def test_predictive_check_omits_undefined_single_observation_sd():
    fit = BayesianPnL([0.4])
    check = posterior_predictive_check(fit, repetitions=100, rng=1)
    assert list(check.index) == ["mean", "min", "max"]
    assert check.loc["mean", "observed"] == 0.4
    assert (check.loc["mean"] == check.loc["min"]).all()
    assert (check.loc["mean"] == check.loc["max"]).all()


@pytest.mark.parametrize("repetitions", [0, -1, True, 1.5])
def test_predictive_check_requires_positive_integer_repetitions(repetitions):
    with pytest.raises(ValueError):
        posterior_predictive_check(BayesianPnL([0.2, 0.5, -0.1]), repetitions)


def test_predictive_check_rejects_unrelated_objects():
    with pytest.raises(TypeError, match="BayesianPnL or Posterior"):
        posterior_predictive_check(object())
