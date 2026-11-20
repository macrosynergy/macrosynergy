"""Backend contracts and numerical regressions independent of plotting."""

import math

import numpy as np
import pytest
from scipy import integrate, stats, special

from macrosynergy.pnl import BayesianPnL
from macrosynergy.pnl.bayes_pnl import (
    GridSpec,
    IndependentPrior,
    NormalPrior,
    SDPrior,
    auto_grid,
    default_prior,
    posterior_grid,
)


@pytest.mark.parametrize("method", ["auto", "grid", "quadrature"])
def test_improper_normal_posterior_rejected_by_every_backend(method):
    prior = IndependentPrior(NormalPrior(0, 1), SDPrior(1, 1, "gamma_sd"))
    with pytest.raises(ValueError, match="Improper posterior"):
        BayesianPnL([0.4, 0.4], prior=prior, method=method)


def test_finite_heavy_tail_sd_moment_matches_independent_integral():
    prior = IndependentPrior(NormalPrior(0, 1), SDPrior(1, 10))
    weight = 0.01
    fit = BayesianPnL([0], prior=prior, likelihood_weight=weight)
    a, b = prior.sd.params["alpha"], prior.sd.params["beta"]

    def moment(order):
        # Independently integrate p(sd) * sd**(-weight) /
        # sqrt(1 + weight / sd**2), after substituting t=log(sd).
        def integrand(t):
            penalty = math.exp(-2 * t) if t > -350 else np.inf
            log_mass = (
                (order - 2 * a - weight) * t
                - b * penalty
                - 0.5 * np.logaddexp(0, math.log(weight) - 2 * t)
            )
            return math.exp(log_mass) if log_mass > -745 else 0.0

        return integrate.quad(integrand, -np.inf, np.inf, epsabs=1e-9)[0]

    z = moment(0)
    expected = math.sqrt(moment(2) / z - (moment(1) / z) ** 2)
    assert fit.posterior.std("sd") == pytest.approx(expected, rel=1e-7)


def test_auto_grid_diagnostics_match_returned_bounds():
    prior = default_prior()
    data = [1.54, 0.33, 0.58]
    spec = auto_grid(data, prior, n=80, coarse_n=60, max_iter=1, tol=1e-8)
    post = posterior_grid(data, prior, spec, warn_tol=1)
    assert spec.diagnostics["edge_mass"] == post.edge_mass()


def test_quadrature_inference_has_no_implicit_grid_dependency(monkeypatch):
    post = BayesianPnL([0.2, -0.1, 0.7]).posterior
    monkeypatch.setattr(post, "to_grid", lambda: pytest.fail("Unexpected grid request"))
    assert not hasattr(post, "mass")
    assert not hasattr(post, "mu")
    assert "integration relative error" in post.summary().index


@pytest.fixture(scope="module")
def volatility_fit():
    return BayesianPnL([8, 10, 12], metric="ann_vol")


def test_volatility_inference_matches_independent_prior_importance_reference(
    volatility_fit,
):
    # Prior draws weighted by SciPy's original Gamma densities, independently
    # of production sufficient statistics and grid construction.
    rng = np.random.default_rng(41)
    count = 400000
    prior = default_prior("ann_vol")
    means = prior.mean.ppf(rng.random(count))
    deviations = prior.sd.ppf(rng.random(count))
    shapes, scales = (means / deviations) ** 2, deviations**2 / means
    logs = stats.gamma.logpdf(
        np.array([8, 10, 12])[:, None], a=shapes, scale=scales
    ).sum(axis=0)
    log_z = special.logsumexp(logs) - math.log(count)
    weights = np.exp(logs - special.logsumexp(logs))
    assert volatility_fit.method == "grid"
    assert volatility_fit.expectation("mean") == pytest.approx(
        weights @ means, abs=0.035
    )
    assert volatility_fit.expectation("sd") == pytest.approx(
        weights @ deviations, abs=0.035
    )
    assert volatility_fit.posterior.log_evidence == pytest.approx(log_z, abs=0.025)
    expected_cdf = weights @ stats.gamma.cdf(9, a=shapes, scale=scales)
    assert volatility_fit.predictive_cdf(9)[0] == pytest.approx(expected_cdf, abs=0.006)


def test_volatility_predictions_and_prior_have_positive_support(volatility_fit):
    np.testing.assert_array_equal(
        volatility_fit.predictive_cdf([-np.inf, -1, 0]), [0, 0, 0]
    )
    np.testing.assert_array_equal(volatility_fit.predictive_pdf([-1]), [0])
    # A Gamma component with shape < 1 has infinite boundary density at zero,
    # while its probability of any nonpositive outcome is still exactly zero.
    assert volatility_fit.predictive_quantile(0)[0] == 0
    quantiles = volatility_fit.predictive_quantile([0.01, 0.5, 0.99])
    assert (quantiles > 0).all()
    np.testing.assert_allclose(
        volatility_fit.predictive_cdf(quantiles), [0.01, 0.5, 0.99], atol=1e-8
    )
    draws = volatility_fit.predictive_sample(30000, rng=5)
    assert (draws > 0).all()
    assert np.mean(draws <= 9) == pytest.approx(
        volatility_fit.predictive_cdf(9)[0], abs=0.01
    )
    assert volatility_fit.posterior.prior.mean.cdf(0) == 0


def test_volatility_rejects_incompatible_model_inputs():
    with pytest.raises(ValueError, match="Gamma likelihood"):
        BayesianPnL([9, 10], metric="ann_vol", method="quadrature")
    with pytest.raises(ValueError, match="strictly positive"):
        BayesianPnL([0, 10], metric="ann_vol")
    with pytest.raises(ValueError, match="positive mean prior"):
        BayesianPnL([9, 10], metric="ann_vol", prior=default_prior("sharpe"))


def test_volatility_change_of_units_preserves_inference(volatility_fit):
    scaled = BayesianPnL([0.08, 0.10, 0.12], metric="ann_vol", vol_target=0.10)
    assert scaled.expectation("mean") * 100 == pytest.approx(
        volatility_fit.expectation("mean"), rel=1e-5
    )
    np.testing.assert_allclose(
        scaled.predictive_cdf([0.09, 0.11]),
        volatility_fit.predictive_cdf([9, 11]),
        atol=1e-5,
    )


def test_positive_grid_parameter_samples_are_discrete_and_positive():
    fit = BayesianPnL(
        [0.001, 0.002, 0.0015],
        metric="ann_vol",
        vol_target=0.001,
        grid=GridSpec((0.001, 0.1), (0.0001, 0.01), 2, 2),
    )
    draws = fit.posterior.sample(10000, rng=4)
    assert (draws.to_numpy() > 0).all()
    assert set(draws["mean"]) <= set(fit.posterior.mu)


def test_positive_mean_prior_does_not_trigger_false_boundary_impropriety():
    from macrosynergy.pnl.bayes_pnl import LogNormalPrior

    prior = IndependentPrior(LogNormalPrior(1, 0.5), SDPrior(1, 1, "gamma_sd"))
    fit = BayesianPnL([0, 0], prior=prior, method="grid")
    assert fit.expectation("sd") > 0


@pytest.mark.parametrize("method", ["grid", "quadrature"])
def test_parameter_cdf_rejects_nan_consistently(method):
    fit = BayesianPnL([0.1, 0.5, -0.2], method=method)
    with pytest.raises(ValueError, match="NaNs"):
        fit.posterior.cdf("mean", np.nan)


def test_volatility_grid_must_preserve_positive_mean_support(volatility_fit):
    assert volatility_fit.posterior.mu.min() > 0
    assert volatility_fit.prob_mean_below(0) == 0
    with pytest.raises(ValueError, match="positive mean bounds"):
        BayesianPnL(
            [0.01, 0.02],
            metric="ann_vol",
            vol_target=0.01,
            grid=GridSpec((-0.01, 0.02), (0.001, 0.02), 5, 20),
        )
