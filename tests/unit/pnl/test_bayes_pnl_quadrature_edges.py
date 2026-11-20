"""Endpoint propriety and sampler contracts independent of visualization grids."""

import numpy as np
import pytest

from macrosynergy.pnl import BayesianPnL
from macrosynergy.pnl.bayes_pnl import IndependentPrior, NormalPrior, SDPrior


@pytest.mark.parametrize("shape", [0.5, 1.0])
def test_identical_data_reject_nonintegrable_gamma_sd(shape):
    prior = IndependentPrior(
        NormalPrior(0, 1), SDPrior(1, 1 / np.sqrt(shape), "gamma_sd")
    )
    with pytest.raises(ValueError, match="Improper posterior"):
        BayesianPnL([0.4, 0.4], prior=prior, method="quadrature")


def test_proper_gamma_posterior_can_have_singular_predictive_density():
    prior = IndependentPrior(NormalPrior(0, 1), SDPrior(1, np.sqrt(2), "gamma_sd"))
    fit = BayesianPnL([0.4], prior=prior, method="quadrature")
    assert np.isinf(fit.predictive_pdf(0.4)[0])
    assert np.isinf(fit.posterior.density("mean", [0.4])[0])
    assert 0 < fit.predictive_cdf(0.4)[0] < 1
    assert np.isfinite(fit.expectation("mean"))
    assert fit.posterior.std("mean") > 0
    assert np.isfinite(fit.predictive_pdf(0.0)[0])


def test_posterior_representation_does_not_construct_plot_grid():
    fit = BayesianPnL([0.1, -0.2, 0.6], method="quadrature")
    assert "method='quadrature'" in repr(fit.posterior)
    assert fit.posterior._plot_grid is None


def test_adaptive_sampler_matches_predictive_cdf_and_is_reproducible():
    fit = BayesianPnL([0.9, -0.3, 0.2, 0.7], method="quadrature")
    first = fit.predictive_sample(20000, rng=17)
    np.testing.assert_array_equal(first, fit.predictive_sample(20000, rng=17))
    for point in (-0.5, 0.0, 0.5, 1.0):
        assert np.mean(first <= point) == pytest.approx(
            fit.predictive_cdf(point)[0], abs=0.012
        )
    assert fit.posterior.diagnostics["sampling_checked_cdf_error"] < 1e-7
    assert fit.posterior._plot_grid is None


def test_overriding_prior_subclass_is_not_silently_analytically_eliminated():
    class DifferentMean(NormalPrior):
        def logpdf(self, mean):
            return super().logpdf(mean) - np.abs(mean)

    prior = IndependentPrior(DifferentMean(0, 1), SDPrior(1, 1))
    with pytest.raises(TypeError, match="Quadrature requires"):
        BayesianPnL([0.2, 0.4], prior=prior, method="quadrature")
