"""Regression checks for inverse-gamma SD moment matching."""

import numpy as np
import pytest
from scipy import integrate

from macrosynergy.pnl.bayes_pnl import SDPrior
from macrosynergy.pnl.bayes_pnl.priors import (
    IndependentPrior,
    LogNormalPrior,
    NormalPrior,
    default_prior,
)


@pytest.mark.parametrize("deviation", [1e-4, 1e-3, 0.01, 0.1])
def test_concentrated_inverse_gamma_has_requested_moments(deviation):
    prior = SDPrior(1.0, deviation)

    # Integrate in standardized SD coordinates to resolve the narrow density.
    # This checks the resulting distribution, independently of the root solver.
    def density(z):
        return np.exp(prior.logpdf(1 + deviation * z)) * deviation

    lower = max(-12.0, -1.0 / deviation)
    mass, _ = integrate.quad(density, lower, 20, epsabs=1e-8)
    mean, _ = integrate.quad(lambda z: z * density(z), lower, 20, epsabs=1e-8)
    second, _ = integrate.quad(lambda z: z**2 * density(z), lower, 20, epsabs=1e-8)
    assert mass == pytest.approx(1.0, abs=2e-6)
    assert mean == pytest.approx(0.0, abs=2e-6)
    assert np.sqrt(second - mean**2) == pytest.approx(1.0, rel=2e-6)


@pytest.mark.parametrize("scale", [1e-6, 0.01, 100.0])
def test_inverse_gamma_shape_is_invariant_to_units(scale):
    original = SDPrior(1.0, 1e-4)
    scaled = SDPrior(scale, scale * 1e-4)
    assert scaled.params["alpha"] == pytest.approx(original.params["alpha"], rel=1e-12)
    assert scaled.params["beta"] / scale**2 == pytest.approx(
        original.params["beta"], rel=1e-12
    )
    np.testing.assert_allclose(
        scaled.ppf([0.05, 0.5, 0.95]) / scale,
        original.ppf([0.05, 0.5, 0.95]),
        rtol=1e-12,
    )


def test_unrepresentable_inverse_gamma_moments_raise_clear_error():
    with pytest.raises(ValueError, match="finite numerical precision"):
        SDPrior(1.0, 1e10)


def test_positive_mean_prior_has_requested_arithmetic_moments():
    prior = LogNormalPrior(10, 5)
    moments = [
        integrate.quad(
            lambda value: value**order * np.exp(prior.logpdf(value)),
            0,
            np.inf,
            epsabs=1e-8,
        )[0]
        for order in range(3)
    ]
    assert prior.normalised
    assert moments[0] == pytest.approx(1, abs=1e-8)
    assert moments[1] == pytest.approx(10, abs=1e-8)
    assert np.sqrt(moments[2] - moments[1] ** 2) == pytest.approx(5, abs=1e-8)
    np.testing.assert_array_equal(prior.cdf([-np.inf, -1, 0]), [0, 0, 0])
    np.testing.assert_array_equal(prior.logpdf([-1, 0]), [-np.inf, -np.inf])
    q = np.array([0.001, 0.1, 0.5, 0.9, 0.999])
    assert (prior.ppf(q) > 0).all()
    np.testing.assert_allclose(prior.cdf(prior.ppf(q)), q, atol=1e-14)


@pytest.mark.parametrize("scale", [1e-6, 0.01, 100.0])
def test_positive_mean_prior_scales_with_pnl_units(scale):
    prior, scaled = LogNormalPrior(10, 5), LogNormalPrior(10 * scale, 5 * scale)
    values = np.array([0.1, 1, 10, 100])
    np.testing.assert_allclose(scaled.cdf(values * scale), prior.cdf(values))
    np.testing.assert_allclose(
        scaled.logpdf(values * scale) + np.log(scale), prior.logpdf(values)
    )
    np.testing.assert_allclose(
        scaled.ppf([0.05, 0.5, 0.95]) / scale, prior.ppf([0.05, 0.5, 0.95])
    )


@pytest.mark.parametrize("mean,sd", [(0, 1), (-1, 1), (1, 0), (1, -1), (np.inf, 1)])
def test_positive_mean_prior_rejects_invalid_moments(mean, sd):
    with pytest.raises(ValueError, match="positive and finite"):
        LogNormalPrior(mean, sd)


def test_default_volatility_prior_is_positive_on_both_axes():
    prior = default_prior("ann_vol", vol_target=10)
    assert isinstance(prior.mean, LogNormalPrior)
    assert prior.mean.mean == 10
    assert prior.mean.sd == 5
    assert prior.normalised
    assert prior.logpdf(-1, 2) == -np.inf
    assert prior.logpdf(10, -1) == -np.inf


def test_joint_prior_normalization_requires_both_components():
    class UnnormalizedMean:
        def logpdf(self, mean):
            return np.zeros_like(mean)

    assert IndependentPrior(NormalPrior(0, 1), SDPrior(1, 1)).normalised
    assert not IndependentPrior(UnnormalizedMean(), SDPrior(1, 1)).normalised
