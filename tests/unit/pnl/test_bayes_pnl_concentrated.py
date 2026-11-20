"""End-to-end inference with nearly known inverse-gamma dispersion."""

import numpy as np
import pytest
from scipy import stats

from macrosynergy.pnl import BayesianPnL
from macrosynergy.pnl.bayes_pnl import IndependentPrior, NormalPrior, SDPrior


@pytest.mark.parametrize("deviation", [1e-4, 1e-5, 1e-6])
def test_concentrated_inverse_gamma_recovers_known_sd_inference(deviation):
    data = np.array([0.4, 0.2, -0.1])
    prior = IndependentPrior(NormalPrior(0, 1), SDPrior(1, deviation))
    fit = BayesianPnL(data, prior=prior)
    # With sd=1 known, the normal mean update has an analytic solution.
    variance = 1 / (1 + len(data))
    mean = data.sum() * variance
    assert fit.expectation("mean") == pytest.approx(mean, abs=1e-7)
    assert fit.posterior.std("mean") == pytest.approx(np.sqrt(variance), abs=1e-7)
    assert fit.expectation("sd") == pytest.approx(1, abs=3e-8)
    assert fit.posterior.std("sd") == pytest.approx(deviation, rel=1e-5)
    expected_predictive = stats.norm.ppf(
        [0.05, 0.5, 0.95], loc=mean, scale=np.sqrt(1 + variance)
    )
    np.testing.assert_allclose(
        fit.predictive_quantile([0.05, 0.5, 0.95]), expected_predictive, atol=1e-7
    )
    # Integrating a unit-variance normal likelihood against N(0,1) gives an
    # n-dimensional Gaussian with covariance I + 11'; it checks constants too.
    expected_evidence = stats.multivariate_normal.logpdf(
        data,
        mean=np.zeros(len(data)),
        cov=np.eye(len(data)) + np.ones((len(data), len(data))),
    )
    assert fit.posterior.log_evidence == pytest.approx(expected_evidence, abs=1e-7)


def test_concentrated_inverse_gamma_inference_preserves_units():
    scale = 1e-4
    original = BayesianPnL(
        [0.4, 0.2, -0.1], prior=IndependentPrior(NormalPrior(0, 1), SDPrior(1, 1e-4))
    )
    scaled = BayesianPnL(
        np.array([0.4, 0.2, -0.1]) * scale,
        prior=IndependentPrior(NormalPrior(0, scale), SDPrior(scale, scale * 1e-4)),
    )
    assert scaled.expectation("sd") / scale == pytest.approx(
        original.expectation("sd"), rel=1e-10
    )
    assert scaled.posterior.log_evidence == pytest.approx(
        original.posterior.log_evidence - 3 * np.log(scale), abs=1e-8
    )
