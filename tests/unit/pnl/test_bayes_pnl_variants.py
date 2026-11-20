"""Independent Gaussian and mixture references for correlated variant inference."""

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from macrosynergy.pnl import BayesianPnL
from macrosynergy.pnl.bayes_pnl import IndependentPrior, NormalPrior, SDPrior
from macrosynergy.pnl.bayes_pnl.variants import BayesianPnLVariants

DATA = pd.DataFrame(
    [[0.2, 0.4, -0.1], [-0.3, -0.1, -0.5], [0.8, 0.7, 0.5], [0.4, 0.6, 0.1]],
    columns=["a", "b", "c"],
)
R = np.array([[1.0, 0.7, 0.5], [0.7, 1.0, 0.6], [0.5, 0.6, 1.0]])


def fixed_fit(data=DATA, correlation=R, **kwargs):
    return BayesianPnLVariants(
        data, correlation=correlation, sigma_grid=[0.7], tau_grid=[0.3], **kwargs
    )


def test_fixed_scales_match_dense_joint_gaussian_likelihood_and_conditioning():
    prior = IndependentPrior(NormalPrior(0.1, 0.8), SDPrior(1.0, 1.0))
    fit = fixed_fit(prior=prior)
    x, T, K = DATA.to_numpy(), len(DATA), DATA.shape[1]
    G, V = 0.8**2 * np.ones((K, K)) + 0.3**2 * np.eye(K), 0.7**2 * R
    covariance = np.kron(np.eye(T), V) + np.kron(np.ones((T, T)), G)
    expected_evidence = stats.multivariate_normal.logpdf(
        x.ravel(), mean=np.full(T * K, 0.1), cov=covariance
    )
    assert fit.log_evidence == pytest.approx(expected_evidence, abs=1e-11)
    precision = np.linalg.inv(G) + T * np.linalg.inv(V)
    conditional_cov = np.linalg.inv(precision)
    conditional_mean = conditional_cov @ (
        np.linalg.solve(G, np.full(K, 0.1)) + T * np.linalg.solve(V, x.mean(axis=0))
    )
    np.testing.assert_allclose(fit.conditional_means[0], conditional_mean, atol=1e-12)
    np.testing.assert_allclose(
        fit.conditional_covariances[0], conditional_cov, atol=1e-12
    )


def test_random_implementation_is_a_distribution_mixture_not_weighted_mean():
    q = np.array([0.1, 0.2, 0.7])
    fit = fixed_fit(weights=q)
    means = fit.conditional_means[0]
    covariance = fit.conditional_covariances[0]
    expected_probability = q @ stats.norm.cdf(0, means, np.sqrt(np.diag(covariance)))
    assert fit.prob_mean_below() == pytest.approx(expected_probability)
    assert fit.expectation(target="draw") == pytest.approx(
        fit.expectation(target="family")
    )
    assert fit.std(target="draw") > fit.std(target="family")
    assert fit.std(target="family") == pytest.approx(np.sqrt(q @ covariance @ q))
    other_weights = fixed_fit(weights=[0.7, 0.2, 0.1])
    np.testing.assert_allclose(fit.conditional_means, other_weights.conditional_means)
    np.testing.assert_allclose(
        fit.conditional_covariances, other_weights.conditional_covariances
    )


def test_declared_aliases_preserve_weights_and_inference_without_automatic_merging():
    data, correlation = DATA[["a", "b"]], R[:2, :2]
    baseline = fixed_fit(data, correlation, weights=[0.6, 0.4])
    duplicate = data.assign(copy=data.a)
    duplicate_R = correlation[np.ix_([0, 1, 0], [0, 1, 0])]
    copied = fixed_fit(
        duplicate, duplicate_R, weights=[0.3, 0.4, 0.3], aliases={"copy": "a"}
    )
    assert copied.names == ("a", "b")
    np.testing.assert_allclose(copied.weights, baseline.weights)
    np.testing.assert_allclose(
        copied.predictive_cdf([-1, 0, 1]), baseline.predictive_cdf([-1, 0, 1])
    )
    assert copied.prob_mean_below(0, "copy") == pytest.approx(
        copied.prob_mean_below(0, "a")
    )
    equal_by_chance = fixed_fit(duplicate, np.eye(3))
    assert equal_by_chance.n_variants == 3


def test_single_variant_without_effects_recovers_scalar_adaptive_posterior():
    prior = IndependentPrior(NormalPrior(0, 1), SDPrior(1, 1.5))
    reference = BayesianPnL(DATA.a.to_numpy(), prior=prior, method="quadrature")
    fit = BayesianPnLVariants(
        DATA[["a"]],
        prior=prior,
        tau_grid=[0.0],
        sigma_grid=np.geomspace(0.025, 40, 900),
    )
    assert fit.expectation() == pytest.approx(reference.expectation("mean"), abs=2e-5)
    assert fit.std() == pytest.approx(reference.posterior.std("mean"), abs=2e-5)
    np.testing.assert_allclose(
        fit.predictive_cdf([-1, 0, 1]), reference.predictive_cdf([-1, 0, 1]), atol=2e-5
    )
    assert fit.log_evidence == pytest.approx(reference.posterior.log_evidence, abs=2e-5)


def test_path_samples_keep_one_implementation_and_parameters_per_world():
    fit = fixed_fit(weights=[0.2, 0.3, 0.5])
    paths = fit.sample_paths(30000, 3, rng=42)
    assert paths["values"].shape == (30000, 3)
    np.testing.assert_array_equal(paths["years"], [3, 6, 9])
    assert set(paths["variant"]) == set(DATA.columns)
    assert np.mean(paths["variant"] == "c") == pytest.approx(0.5, abs=0.012)
    residuals = paths["values"] - paths["means"][:, None]
    assert np.var(residuals) == pytest.approx(0.7**2, rel=0.025)
    assert abs(np.corrcoef(residuals.T)[0, 1]) < 0.02
    assert np.cov(paths["values"].T)[0, 1] == pytest.approx(fit.std() ** 2, abs=0.015)
    np.testing.assert_array_equal(
        paths["values"], fit.sample_paths(30000, 3, rng=42)["values"]
    )


def test_daily_input_uses_common_complete_blocks_and_return_units():
    dates = pd.bdate_range("2001-01-01", "2012-12-31")
    rng = np.random.default_rng(7)
    pnl = pd.DataFrame(
        rng.normal(0.02, 0.6, (len(dates), 2)), index=dates, columns=["a", "b"]
    )
    pnl.loc["2005-04-01", "b"] = np.nan
    with pytest.warns(UserWarning):
        fit = BayesianPnLVariants.from_pnl(
            pnl,
            correlation=[[1, 0.8], [0.8, 1]],
            metric="ann_ret",
            sigma_grid=[10.0],
            tau_grid=[2.0],
        )
    assert fit.n_periods == 3
    assert len(fit.periods) == 6
    assert fit.metric == "ann_ret"
    draws = fit.sample(10, rng=1)
    assert set(["mean", "sd", "variant", "sigma", "tau"]) <= set(draws.columns)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"correlation": None},
        {"correlation": [[1, 2, 0], [2, 1, 0], [0, 0, 1]]},
        {"weights": [-1, 1, 1]},
        {"weights": [0, 0, 0]},
        {"tau_grid": [-1.0]},
        {"sigma_grid": [0.0]},
        {"metric": "ann_vol"},
        {"aliases": {"a": "b", "b": "a"}},
    ],
)
def test_invalid_hierarchical_inputs_are_rejected(kwargs):
    arguments = dict(correlation=R, sigma_grid=[0.7], tau_grid=[0.3])
    arguments.update(kwargs)
    with pytest.raises((TypeError, ValueError)):
        BayesianPnLVariants(DATA, **arguments)


def test_integrated_grid_is_normalized_and_research_weights_do_not_change_likelihood():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        a = BayesianPnLVariants(
            DATA, correlation=R, n_sigma=60, n_tau=51, weights=[0.2, 0.3, 0.5]
        )
        b = BayesianPnLVariants(
            DATA, correlation=R, n_sigma=60, n_tau=51, weights=[0.8, 0.1, 0.1]
        )
    assert a.mass.sum() == pytest.approx(1.0)
    np.testing.assert_array_equal(a.mass, b.mass)
    np.testing.assert_allclose(
        a.predictive_cdf(a.predictive_quantile([0.1, 0.5, 0.9])),
        [0.1, 0.5, 0.9],
        atol=1e-8,
    )
    assert a.diagnostics["covariance_uncertainty"].startswith("R and C are fixed")


def test_positive_semidefinite_construction_kernel_is_allowed():
    fit = fixed_fit(construction_correlation=np.ones((3, 3)))
    np.testing.assert_allclose(
        fit.conditional_means[0], np.repeat(fit.conditional_means[0, 0], 3)
    )
    assert np.linalg.eigvalsh(fit.conditional_covariances[0]).min() >= -1e-12


@pytest.mark.parametrize(
    "kwargs",
    [
        {"correlation": R.astype(complex) + np.eye(3) * 1j},
        {"weights": [1e308, 1e308, 1e308]},
        {"construction_correlation": [[1, 2, 0], [2, 1, 0], [0, 0, 1]]},
    ],
)
def test_complex_or_invalid_covariance_and_overflowing_weights_are_rejected(kwargs):
    arguments = dict(correlation=R, sigma_grid=[0.7], tau_grid=[0.3])
    arguments.update(kwargs)
    with pytest.raises(ValueError):
        BayesianPnLVariants(DATA, **arguments)


def test_permuting_names_correlations_and_weights_preserves_inference():
    order = [2, 0, 1]
    C = 0.3 * np.ones((3, 3)) + 0.7 * np.eye(3)
    q = np.array([0.2, 0.3, 0.5])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        a = BayesianPnLVariants(
            DATA,
            correlation=R,
            construction_correlation=C,
            weights=q,
            n_sigma=55,
            n_tau=51,
        )
        b = BayesianPnLVariants(
            DATA.iloc[:, order],
            correlation=R[np.ix_(order, order)],
            construction_correlation=C[np.ix_(order, order)],
            weights=q[order],
            n_sigma=55,
            n_tau=51,
        )
    np.testing.assert_allclose(a.mass, b.mass, atol=2e-14)
    np.testing.assert_allclose(
        a.predictive_cdf([-1, 0, 1]), b.predictive_cdf([-1, 0, 1]), atol=1e-12
    )
    for name in DATA.columns:
        assert a.expectation(target=name) == pytest.approx(
            b.expectation(target=name), abs=1e-12
        )


def test_default_grid_is_stable_under_doubling_for_near_copy_constructions():
    common = np.array([-0.4, 0.1, 0.9, -0.2, 0.4, 0.2, 0.5])
    data = pd.DataFrame(
        {
            "a": common,
            "b": common + 0.001,
            "c": common + np.array([0.002, -0.001, 0, 0.001, -0.002, 0, 0.001]),
        }
    )
    R_close = 0.995 * np.ones((3, 3)) + 0.005 * np.eye(3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        a = BayesianPnLVariants(data, correlation=R_close)
        b = BayesianPnLVariants(data, correlation=R_close, n_sigma=200, n_tau=161)
    assert a.expectation() == pytest.approx(b.expectation(), abs=0.002)
    assert a.prob_mean_below() == pytest.approx(b.prob_mean_below(), abs=0.003)
    np.testing.assert_allclose(a.interval(), b.interval(), atol=0.008)


def test_identical_period_gamma_zero_scale_case_is_explicitly_unsupported():
    prior = IndependentPrior(NormalPrior(0, 1), SDPrior(1, 0.2, "gamma_sd"))
    data = pd.DataFrame([[0.2, 0.3], [0.2, 0.3]], columns=["a", "b"])
    with pytest.raises(ValueError, match="unsupported"):
        BayesianPnLVariants(data, correlation=np.eye(2), prior=prior)
    fixed = BayesianPnLVariants(
        data, correlation=np.eye(2), prior=prior, sigma_grid=[0.5], tau_grid=[0.3]
    )
    assert np.isfinite(fixed.expectation())
