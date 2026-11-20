"""Independent numerical references for adaptive Bayesian PnL integration."""

import math
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import integrate, special, stats

from macrosynergy.pnl import BayesianPnL
from macrosynergy.pnl.bayes_pnl import (
    Posterior,
    GridSpec,
    GridTruncationWarning,
    IndependentPrior,
    NormalPrior,
    SDPrior,
    posterior_grid,
)

DATA = np.array([0.9, -0.3, 0.2, 0.7])


@pytest.fixture(scope="module")
def independent_prior():
    return IndependentPrior(NormalPrior(0.1, 0.7), SDPrior(0.8, 0.35, "gamma_sd"))


@pytest.fixture(scope="module")
def quadrature_fit(independent_prior):
    return BayesianPnL(DATA, prior=independent_prior, method="quadrature")


@pytest.fixture(scope="module")
def direct_two_dimensional_reference():
    """Integrate the original joint density, without conditional-normal formulas.

    A normal mean prior and gamma sd prior keep mass outside these deliberately
    wide bounds negligible. Computing both halves separately also gives the
    posterior probability of a negative mean without a discontinuous integrand.
    """
    prior_mean, prior_sd = 0.1, 0.7
    sd_mean, sd_sd = 0.8, 0.35
    shape, rate = sd_mean**2 / sd_sd**2, sd_mean / sd_sd**2
    prior_constant = shape * math.log(rate) - special.gammaln(shape)
    normal_constant = -math.log(prior_sd * math.sqrt(2 * math.pi))
    n, total, squares = len(DATA), float(DATA.sum()), float(np.sum(DATA**2))

    def integrate_half(lower_mean, upper_mean):
        def outer(log_sd):
            sd = math.exp(log_sd)
            # Includes the sd -> log(sd) Jacobian, independently of production code.
            outer_log_density = prior_constant + shape * log_sd - rate * sd

            def joint_moments(mean):
                log_likelihood = -n * (log_sd + math.log(2 * math.pi) / 2) - (
                    squares - 2 * mean * total + n * mean**2
                ) / (2 * sd**2)
                log_prior_mean = (
                    normal_constant - ((mean - prior_mean) / prior_sd) ** 2 / 2
                )
                mass = math.exp(outer_log_density + log_prior_mean + log_likelihood)
                return mass * np.array(
                    [
                        1.0,
                        mean,
                        mean**2,
                        sd,
                        sd**2,
                        special.ndtr(-mean / sd),
                    ]
                )

            return integrate.quad_vec(
                joint_moments, lower_mean, upper_mean, epsabs=1e-11, epsrel=1e-9
            )[0]

        return integrate.quad_vec(outer, -5, 4, epsabs=1e-10, epsrel=1e-8)[0]

    negative = integrate_half(-6, 0)
    positive = integrate_half(0, 6)
    combined = negative + positive
    normalized = combined / combined[0]
    return {
        "mean": normalized[1],
        "mean_sd": math.sqrt(normalized[2] - normalized[1] ** 2),
        "sd": normalized[3],
        "sd_sd": math.sqrt(normalized[4] - normalized[3] ** 2),
        "prob_mean_negative": negative[0] / combined[0],
        "predictive_cdf_zero": normalized[5],
        "log_evidence": math.log(combined[0]),
    }


def test_quadrature_matches_direct_joint_integration(
    quadrature_fit,
    direct_two_dimensional_reference,
):
    fit, expected = quadrature_fit, direct_two_dimensional_reference
    assert fit.method == "quadrature"
    assert isinstance(fit.posterior, Posterior)
    assert fit.expectation("mean") == pytest.approx(expected["mean"], abs=2e-7)
    assert fit.expectation("sd") == pytest.approx(expected["sd"], abs=2e-7)
    assert fit.posterior.std("mean") == pytest.approx(expected["mean_sd"], abs=2e-7)
    assert fit.posterior.std("sd") == pytest.approx(expected["sd_sd"], abs=2e-7)
    assert fit.prob_mean_below(0) == pytest.approx(
        expected["prob_mean_negative"], abs=2e-7
    )
    assert fit.predictive_cdf(0)[0] == pytest.approx(
        expected["predictive_cdf_zero"], abs=2e-7
    )
    assert fit.posterior.log_evidence == pytest.approx(
        expected["log_evidence"], abs=2e-7
    )


def test_inference_does_not_evaluate_or_depend_on_visual_grid(independent_prior):
    # This grid excludes the posterior entirely. Only explicit visualization
    # should evaluate it or issue truncation warnings.
    narrow = GridSpec((9, 10), (5, 6), 3, 3)
    broad = GridSpec((-4, 4), (0.02, 8), 70, 90, "log")
    with warnings.catch_warnings():
        warnings.simplefilter("error", GridTruncationWarning)
        fits = [
            BayesianPnL(DATA, prior=independent_prior, grid=grid, method="quadrature")
            for grid in (narrow, broad)
        ]
        for parameter in ("mean", "sd"):
            assert fits[0].expectation(parameter) == pytest.approx(
                fits[1].expectation(parameter),
                rel=1e-9,
            )
            np.testing.assert_allclose(
                fits[0].interval(parameter), fits[1].interval(parameter), atol=1e-8
            )
        np.testing.assert_allclose(
            fits[0].predictive_quantile([0.05, 0.5, 0.95]),
            fits[1].predictive_quantile([0.05, 0.5, 0.95]),
            atol=1e-8,
        )
        assert fits[0].posterior.log_evidence == pytest.approx(
            fits[1].posterior.log_evidence,
            abs=1e-9,
        )


@pytest.mark.parametrize("kind", ["invgamma_var", "gamma_sd", "lognormal_sd"])
def test_each_sd_prior_matches_a_wide_grid(kind):
    prior = IndependentPrior(NormalPrior(0.1, 0.7), SDPrior(0.8, 0.35, kind))
    fit = BayesianPnL(DATA, prior=prior, method="quadrature")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", GridTruncationWarning)
        reference = posterior_grid(
            DATA,
            prior,
            GridSpec((-4, 4), (0.015, 8), 600, 700, "log"),
        )
    for parameter in ("mean", "sd"):
        assert fit.expectation(parameter) == pytest.approx(
            reference.expectation(parameter), abs=0.003
        )
        np.testing.assert_allclose(
            fit.interval(parameter), reference.interval(parameter), atol=0.004
        )
    np.testing.assert_allclose(
        fit.predictive_cdf([-1, 0, 1]), reference.predictive_cdf([-1, 0, 1]), atol=0.002
    )
    assert fit.posterior.log_evidence == pytest.approx(
        reference.log_evidence, abs=0.002
    )


def test_nearly_known_sd_recovers_normal_conjugate_update():
    fixed_sd, prior_mean, prior_sd = 0.75, -0.1, 0.6
    prior = IndependentPrior(
        NormalPrior(prior_mean, prior_sd), SDPrior(fixed_sd, 0.0001, "lognormal_sd")
    )
    fit = BayesianPnL(DATA, prior=prior, method="quadrature")
    variance = 1 / (1 / prior_sd**2 + len(DATA) / fixed_sd**2)
    mean = variance * (prior_mean / prior_sd**2 + DATA.sum() / fixed_sd**2)
    assert fit.expectation("mean") == pytest.approx(mean, abs=1e-6)
    assert fit.posterior.std("mean") == pytest.approx(np.sqrt(variance), abs=1e-6)
    np.testing.assert_allclose(
        fit.predictive_quantile([0.05, 0.5, 0.95]),
        stats.norm.ppf(
            [0.05, 0.5, 0.95], loc=mean, scale=np.sqrt(fixed_sd**2 + variance)
        ),
        atol=2e-6,
    )


@pytest.mark.parametrize("weight", [0.2, 0.5])
def test_power_likelihood_matches_independent_grid(weight, independent_prior):
    fit = BayesianPnL(
        DATA, prior=independent_prior, method="quadrature", likelihood_weight=weight
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", GridTruncationWarning)
        reference = posterior_grid(
            DATA,
            independent_prior,
            GridSpec((-4, 4), (0.015, 8), 550, 650, "log"),
            likelihood_weight=weight,
        )
    for parameter in ("mean", "sd"):
        np.testing.assert_allclose(
            fit.interval(parameter), reference.interval(parameter), atol=0.004
        )
    np.testing.assert_allclose(
        fit.predictive_cdf([-1, 0, 1]), reference.predictive_cdf([-1, 0, 1]), atol=0.002
    )
    assert fit.posterior.log_evidence is None


@pytest.mark.parametrize("data", [[0.4], [0.4, 0.4, 0.4]])
def test_small_or_identical_sample_keeps_proper_posterior(data, independent_prior):
    fit = BayesianPnL(data, prior=independent_prior, method="quadrature")
    assert np.isfinite([fit.expectation("mean"), fit.expectation("sd")]).all()
    assert fit.posterior.std("mean") > 0
    assert fit.posterior.std("sd") > 0
    quantiles = fit.predictive_quantile([0.1, 0.5, 0.9])
    assert np.isfinite(quantiles).all()
    assert (np.diff(quantiles) > 0).all()
    np.testing.assert_allclose(
        fit.predictive_cdf(quantiles), [0.1, 0.5, 0.9], atol=1e-7
    )


@pytest.mark.parametrize("scale", [1e-6, 100.0])
def test_change_of_units_preserves_inference(scale, quadrature_fit):
    prior = IndependentPrior(
        NormalPrior(0.1 * scale, 0.7 * scale),
        SDPrior(0.8 * scale, 0.35 * scale, "gamma_sd"),
    )
    scaled = BayesianPnL(DATA * scale, prior=prior, method="quadrature")
    for parameter in ("mean", "sd"):
        assert scaled.expectation(parameter) / scale == pytest.approx(
            quadrature_fit.expectation(parameter),
            abs=1e-7,
        )
        np.testing.assert_allclose(
            np.asarray(scaled.interval(parameter)) / scale,
            quadrature_fit.interval(parameter),
            atol=1e-7,
        )
    assert scaled.prob_mean_below(0) == pytest.approx(
        quadrature_fit.prob_mean_below(0), abs=1e-8
    )
    np.testing.assert_allclose(
        scaled.predictive_pdf(DATA * scale) * scale,
        quadrature_fit.predictive_pdf(DATA),
        rtol=2e-7,
        atol=1e-8,
    )
    expected_evidence = quadrature_fit.posterior.log_evidence - len(DATA) * np.log(
        scale
    )
    assert scaled.posterior.log_evidence == pytest.approx(expected_evidence, abs=1e-7)


def test_large_location_shift_is_not_lost_to_initial_integration_bounds(quadrature_fit):
    shift = 20.0
    prior = IndependentPrior(
        NormalPrior(0.1 + shift, 0.7), SDPrior(0.8, 0.35, "gamma_sd")
    )
    shifted = BayesianPnL(DATA + shift, prior=prior, method="quadrature")
    assert shifted.expectation("mean") - shift == pytest.approx(
        quadrature_fit.expectation("mean"),
        abs=1e-7,
    )
    assert shifted.expectation("sd") == pytest.approx(
        quadrature_fit.expectation("sd"), abs=1e-7
    )
    np.testing.assert_allclose(
        shifted.predictive_quantile([0.05, 0.5, 0.95]) - shift,
        quadrature_fit.predictive_quantile([0.05, 0.5, 0.95]),
        atol=1e-7,
    )
    assert shifted.posterior.log_evidence == pytest.approx(
        quadrature_fit.posterior.log_evidence,
        abs=1e-7,
    )


def test_tight_mean_prior_remains_resolved():
    prior = IndependentPrior(NormalPrior(-0.2, 1e-5), SDPrior(0.8, 0.35, "gamma_sd"))
    fit = BayesianPnL(DATA, prior=prior, method="quadrature")
    assert fit.expectation("mean") == pytest.approx(-0.2, abs=1e-8)
    assert fit.posterior.std("mean") == pytest.approx(1e-5, rel=1e-4)
    assert fit.posterior.cdf("mean", -0.2) == pytest.approx(0.5, abs=1e-4)


def test_predictive_and_parameter_quantiles_resolve_small_tail_probabilities(
    quadrature_fit,
):
    probabilities = np.array([1e-7, 0.1, 0.5, 0.9, 1 - 1e-7])
    predictive = quadrature_fit.predictive_quantile(probabilities)
    assert np.isfinite(predictive).all() and (np.diff(predictive) > 0).all()
    np.testing.assert_allclose(
        quadrature_fit.predictive_cdf(predictive), probabilities, atol=2e-9, rtol=0
    )
    for parameter in ("mean", "sd"):
        quantiles = quadrature_fit.posterior.quantile(parameter, probabilities)
        assert np.isfinite(quantiles).all() and (np.diff(quantiles) > 0).all()
        np.testing.assert_allclose(
            quadrature_fit.posterior.cdf(parameter, quantiles),
            probabilities,
            atol=2e-9,
            rtol=0,
        )
    np.testing.assert_array_equal(
        quadrature_fit.predictive_cdf([-np.inf, np.inf]), [0, 1]
    )
    np.testing.assert_array_equal(
        quadrature_fit.predictive_quantile([0, 1]), [-np.inf, np.inf]
    )
    assert quadrature_fit.posterior.cdf("sd", 0) == 0


class JointPrior:
    """Proper scale-dependent mean prior, unsupported by analytic elimination."""

    normalised = True

    def logpdf(self, mean, sd):
        return stats.norm.logpdf(mean, 0, sd / 2) + stats.lognorm.logpdf(sd, 0.3)


def test_auto_selects_supported_backend_and_explicit_grid_remains_available(
    independent_prior,
):
    automatic = BayesianPnL(DATA, prior=independent_prior, method="auto")
    assert automatic.method == "quadrature"
    grid = GridSpec((-4, 4), (0.02, 8), 160, 180, "log")
    explicit = BayesianPnL(DATA, prior=independent_prior, grid=grid, method="grid")
    assert explicit.method == "grid"
    custom = BayesianPnL(DATA, prior=JointPrior(), grid=grid, method="auto")
    assert custom.method == "grid"
    with pytest.raises((TypeError, ValueError)):
        BayesianPnL(DATA, prior=JointPrior(), grid=grid, method="quadrature")
    with pytest.raises(ValueError):
        BayesianPnL(DATA, method="unknown")


def test_from_pnl_passes_through_backend_choice():
    dates = pd.bdate_range("2001-01-01", "2006-12-31")
    pnl = pd.Series(np.random.RandomState(7).normal(0.02, 0.6, len(dates)), index=dates)
    fit = BayesianPnL.from_pnl(pnl, method="quadrature")
    assert fit.method == "quadrature"
    assert len(fit.periods) == 2
    assert fit.posterior.period_years == 3
