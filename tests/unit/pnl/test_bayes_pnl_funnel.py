"""The block-metric funnel retains parameter uncertainty along each full path."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from macrosynergy.pnl import BayesianPnL
from macrosynergy.pnl.bayes_pnl import GridSpec
from macrosynergy.pnl.bayes_pnl.funnel import (
    MetricFunnel,
    bridge_metric_funnel,
    simulate_metric_funnel,
)


class FixedPosterior:
    """Known parameters provide an analytic reference for the forward model."""

    metric = "sharpe"
    period_years = 3.0

    def __init__(self, mean=0.4, sd=0.8):
        self.mean = mean
        self.sd = sd
        self.calls = []

    def sample(self, n, rng=None):
        self.calls.append(n)
        return pd.DataFrame(
            {"mean": np.repeat(self.mean, n), "sd": np.repeat(self.sd, n)}
        )


class UncertainMeanPosterior(FixedPosterior):
    """Long-term mean uncertainty persists after averaging many observations."""

    def sample(self, n, rng=None):
        self.calls.append(n)
        rng = np.random.default_rng(rng)
        return pd.DataFrame({"mean": rng.normal(0.2, 0.4, n), "sd": np.repeat(0.5, n)})


def test_known_parameter_running_average_matches_normal_reference():
    posterior = FixedPosterior()
    result = simulate_metric_funnel(posterior, n_paths=60000, n_periods=5, rng=71)
    assert isinstance(result, MetricFunnel)
    assert posterior.calls == [60000]
    assert result.period_metrics.shape == result.running_mean.shape == (60000, 5)
    np.testing.assert_array_equal(result.years, [3, 6, 9, 12, 15])
    np.testing.assert_allclose(result.running_mean.mean(axis=0), 0.4, atol=0.008)
    np.testing.assert_allclose(
        result.running_mean.var(axis=0), 0.8**2 / np.arange(1, 6), rtol=0.025
    )
    expected = stats.norm.ppf(0.05, loc=0.4, scale=0.8 / np.sqrt(np.arange(1, 6)))
    np.testing.assert_allclose(result.pointwise_interval(0.9)[0], expected, atol=0.015)
    assert abs(np.corrcoef(result.period_metrics[:, :2].T)[0, 1]) < 0.02
    assert result.metadata["interval_coverage"] == "pointwise"
    assert result.metadata["target"] == "running_mean_period_metric"


def test_parameter_uncertainty_is_retained_instead_of_redrawn_each_period():
    posterior = UncertainMeanPosterior()
    result = simulate_metric_funnel(posterior, n_paths=60000, n_periods=8, rng=11)
    assert posterior.calls == [60000]
    expected_variance = 0.4**2 + 0.5**2 / np.arange(1, 9)
    np.testing.assert_allclose(
        result.running_mean.var(axis=0), expected_variance, rtol=0.025
    )
    covariance = np.cov(result.period_metrics[:, :2].T)[0, 1]
    assert covariance == pytest.approx(0.4**2, abs=0.006)
    # Conditional residuals have independent shocks, while unconditional blocks
    # share the same uncertain mean. Averaging cannot remove that uncertainty.
    residuals = result.period_metrics - result.parameter_means[:, None]
    assert abs(np.corrcoef(residuals[:, :2].T)[0, 1]) < 0.02
    np.testing.assert_allclose(
        result.mean_interval(), stats.norm.ppf([0.05, 0.95], 0.2, 0.4), atol=0.012
    )


def test_first_period_matches_fitted_next_period_predictive():
    fit = BayesianPnL(
        [0.9, 0.2, 1.1, 0.4, -0.3, 0.7],
        method="grid",
        grid=GridSpec((-3, 3), (0.03, 5), 150, 180, "log"),
    )
    result = simulate_metric_funnel(fit, n_paths=60000, n_periods=3, rng=31)
    np.testing.assert_array_equal(
        result.running_mean[:, 0], result.period_metrics[:, 0]
    )
    for threshold in (-0.5, 0.0, 0.5, 1.5):
        probability = np.mean(result.running_mean[:, 0] <= threshold)
        assert probability == pytest.approx(fit.predictive_cdf(threshold)[0], abs=0.008)
    assert result.mean_interval()[0] < result.mean_interval()[1]


def test_seed_reproducibility_and_running_average_arithmetic():
    first = simulate_metric_funnel(FixedPosterior(), n_paths=12, n_periods=3, rng=12)
    second = simulate_metric_funnel(FixedPosterior(), n_paths=12, n_periods=3, rng=12)
    np.testing.assert_array_equal(first.period_metrics, second.period_metrics)
    np.testing.assert_allclose(
        first.running_mean[:, 1], first.period_metrics[:, :2].mean(axis=1)
    )
    np.testing.assert_allclose(
        first.running_mean[:, 2], first.period_metrics.mean(axis=1)
    )
    assert first.n_paths == 12 and first.n_periods == 3
    assert first.metric == "sharpe" and first.period_years == 3
    assert first.construction_labels is None


def test_cumulative_pnl_uses_annual_returns_and_nominal_block_duration():
    posterior = FixedPosterior(mean=4.0, sd=6.0)
    posterior.metric = "ann_ret"
    result = simulate_metric_funnel(posterior, n_paths=60000, n_periods=4, rng=22)
    np.testing.assert_allclose(
        result.cumulative_return, 3 * np.cumsum(result.period_metrics, axis=1)
    )
    np.testing.assert_array_equal(result.expected_cumulative_return, [12, 24, 36, 48])
    np.testing.assert_allclose(
        result.cumulative_return.mean(axis=0),
        result.expected_cumulative_return,
        atol=0.3,
    )
    np.testing.assert_allclose(
        result.cumulative_return.var(axis=0), 3**2 * 6**2 * np.arange(1, 5), rtol=0.025
    )
    # No compounding, conversion from Sharpe, or assumed daily path is involved.
    assert result.metadata["cumulative_return"].startswith("annual-return draws")


@pytest.mark.parametrize("metric", ["sharpe", "ann_vol"])
def test_cumulative_pnl_cannot_be_inferred_from_other_metrics(metric):
    posterior = FixedPosterior(mean=10.0, sd=2.0)
    posterior.metric = metric
    result = simulate_metric_funnel(posterior, n_paths=10, rng=1)
    with pytest.raises(ValueError, match="annual-return"):
        result.cumulative_return
    with pytest.raises(ValueError, match="ann_ret"):
        result.expected_cumulative_return


def test_construction_is_selected_once_and_retained_for_the_whole_path():
    class ConstructionMixture(FixedPosterior):
        def sample(self, n, rng=None):
            self.calls.append(n)
            rng = np.random.default_rng(rng)
            choice = rng.choice(["a", "b"], size=n, p=[0.25, 0.75])
            means = np.where(choice == "a", -2.0, 2.0)
            return pd.DataFrame(
                {"mean": means, "sd": np.repeat(0.05, n), "variant": choice}
            )

    mixture = ConstructionMixture()
    result = simulate_metric_funnel(mixture, n_paths=10000, n_periods=5, rng=1)
    assert mixture.calls == [10000]
    assert np.mean(result.construction_labels == "a") == pytest.approx(0.25, abs=0.02)
    np.testing.assert_array_equal(
        result.parameter_means < 0, result.construction_labels == "a"
    )
    np.testing.assert_array_equal(
        (result.period_metrics < 0).all(axis=1), result.construction_labels == "a"
    )
    assert np.all(result.running_mean[result.construction_labels == "b"] > 0)


def test_volatility_uses_positive_gamma_observation_law():
    posterior = FixedPosterior(mean=10, sd=2)
    posterior.metric = "ann_vol"
    result = simulate_metric_funnel(posterior, n_paths=40000, n_periods=4, rng=17)
    assert np.all(result.period_metrics > 0)
    np.testing.assert_allclose(result.running_mean.mean(axis=0), 10, atol=0.04)
    np.testing.assert_allclose(
        result.running_mean.var(axis=0), 4 / np.arange(1, 5), rtol=0.03
    )
    np.testing.assert_array_equal(result.prob_nonpositive(), [0, 0, 0, 0])


def test_reported_probabilities_and_quantiles_use_each_running_average():
    result = simulate_metric_funnel(FixedPosterior(), n_paths=1000, n_periods=3, rng=21)
    np.testing.assert_array_equal(
        result.prob_nonpositive(), np.mean(result.running_mean <= 0, axis=0)
    )
    np.testing.assert_array_equal(
        result.probability_below(0.3), np.mean(result.running_mean < 0.3, axis=0)
    )
    np.testing.assert_allclose(
        result.pointwise_quantile([0.1, 0.5, 0.9]),
        np.quantile(result.running_mean, [0.1, 0.5, 0.9], axis=0),
    )
    assert result.pointwise_quantile(0.5).shape == (1, 3)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_paths": 0},
        {"n_periods": 0},
        {"n_paths": -1},
        {"n_periods": 1.5},
        {"n_paths": True},
    ],
)
def test_funnel_requires_positive_integer_simulation_dimensions(kwargs):
    with pytest.raises((TypeError, ValueError)):
        simulate_metric_funnel(FixedPosterior(), **kwargs)


@pytest.mark.parametrize("level", [0, 1, -0.1, np.nan])
def test_funnel_intervals_reject_invalid_levels(level):
    result = simulate_metric_funnel(FixedPosterior(), n_paths=10, rng=1)
    with pytest.raises(ValueError):
        result.pointwise_interval(level)
    with pytest.raises(ValueError):
        result.mean_interval(level)


@pytest.mark.parametrize("duration", [0, -1, np.nan, np.inf])
def test_funnel_rejects_undefined_period_duration(duration):
    posterior = FixedPosterior()
    posterior.period_years = duration
    with pytest.raises(ValueError):
        simulate_metric_funnel(posterior, n_paths=10, rng=1)


@pytest.mark.parametrize("mean,sd", [(np.nan, 1), (0.1, 0), (0.1, -1), (0.1, np.inf)])
def test_funnel_rejects_invalid_parameter_draws(mean, sd):
    with pytest.raises(ValueError):
        simulate_metric_funnel(FixedPosterior(mean, sd), n_paths=10, rng=1)


def test_funnel_rejects_nonposterior_objects():
    with pytest.raises(TypeError):
        simulate_metric_funnel(object())


def test_explicit_path_sampler_preserves_time_dependence_without_resimulation():
    class PathPosterior(FixedPosterior):
        def sample(self, n, rng=None):
            raise AssertionError(
                "Do not replace complete paths with iid parameter-pair draws."
            )

        def sample_paths(self, n_paths, n_periods, rng=None):
            rng = np.random.default_rng(rng)
            shock = rng.normal(0.4, 0.8, n_paths)
            return {
                "values": np.repeat(shock[:, None], n_periods, axis=1),
                "means": np.repeat(0.4, n_paths),
                "sds": np.repeat(0.8, n_paths),
                "years": 3 * np.arange(1, n_periods + 1),
                "variant": np.repeat("construction", n_paths),
                "component": np.repeat("persistent-model", n_paths),
            }

    result = simulate_metric_funnel(PathPosterior(), n_paths=100, n_periods=3, rng=1)
    np.testing.assert_array_equal(
        result.period_metrics[:, 0], result.period_metrics[:, 2]
    )
    assert (result.component_labels == "persistent-model").all()
    assert (result.construction_labels == "construction").all()
    assert (
        result.metadata["conditional_observations"] == "provided by fitted path sampler"
    )


@pytest.mark.parametrize(
    "broken_field,value",
    [
        ("values", np.zeros((5, 2))),
        ("values", np.full((5, 3), np.nan)),
        ("means", np.zeros((5, 1))),
        ("sds", np.zeros(5)),
        ("years", [1, 2, 3]),
        ("variant", ["only-one-label"]),
        ("expected_values", np.ones((5, 2))),
        ("expected_values", np.full((5, 3), np.nan)),
    ],
)
def test_path_protocol_rejects_incompatible_shapes_and_boundaries(broken_field, value):
    class BrokenPaths(FixedPosterior):
        def sample_paths(self, n_paths, n_periods, rng=None):
            result = {
                "values": np.ones((5, 3)),
                "means": np.ones(5),
                "sds": np.ones(5),
                "years": [3, 6, 9],
            }
            result[broken_field] = value
            return result

    with pytest.raises(ValueError):
        simulate_metric_funnel(BrokenPaths(), n_paths=5, n_periods=3, rng=1)


def test_dynamic_conditional_expectations_are_not_replaced_by_long_term_mean():
    class DynamicReturns(FixedPosterior):
        metric = "ann_ret"

        def sample_paths(self, n_paths, n_periods, rng=None):
            expected = np.broadcast_to([1.0, 2.0, 4.0], (n_paths, n_periods))
            rng = np.random.default_rng(rng)
            return {
                "values": expected + rng.normal(0, 0.1, (n_paths, n_periods)),
                "expected_values": expected,
                "means": np.repeat(100.0, n_paths),
                "sds": np.ones(n_paths),
                "years": [3, 6, 9],
            }

    result = simulate_metric_funnel(DynamicReturns(), n_paths=1000, n_periods=3, rng=1)
    np.testing.assert_allclose(result.expected_cumulative_return, [3, 9, 21])
    np.testing.assert_allclose(result.expected_running_mean, [1, 1.5, 7 / 3])
    np.testing.assert_allclose(
        result.expected_path_draws("cumulative_return"),
        np.broadcast_to([3, 9, 21], (1000, 3)),
    )
    assert result.parameter_means.mean() == 100


def test_unknown_conditional_trajectory_uses_simulations_without_inventing_a_band():
    class UnknownTrajectory(FixedPosterior):
        metric = "ann_ret"

        def sample_paths(self, n_paths, n_periods, rng=None):
            return {
                "values": np.broadcast_to([1.0, 2.0, 4.0], (n_paths, n_periods)),
                "means": np.repeat(100.0, n_paths),
                "sds": np.ones(n_paths),
                "years": [3, 6, 9],
            }

    result = simulate_metric_funnel(UnknownTrajectory(), n_paths=10, n_periods=3, rng=1)
    assert result.expected_period_metrics is None
    assert result.expected_path_draws("cumulative_return") is None
    assert result.expected_path_draws("running_mean") is None
    np.testing.assert_allclose(result.expected_cumulative_return, [3, 9, 21])
    np.testing.assert_allclose(result.expected_running_mean, [1, 1.5, 7 / 3])


@pytest.fixture(scope="module")
def brownian_bridge_reference():
    # With block annual-return sd V/sqrt(T), independent Gaussian endpoints
    # plus bridges recover Brownian cumulative variance V**2 * elapsed years.
    posterior = FixedPosterior(mean=0.4, sd=10 / np.sqrt(3))
    posterior.metric = "ann_ret"
    source = simulate_metric_funnel(posterior, n_paths=24000, n_periods=2, rng=91)
    fine = bridge_metric_funnel(
        source, annual_volatility=10, frequency="monthly", rng=92
    )
    return source, fine


def test_bridge_preserves_original_endpoints_and_parameter_identity(
    brownian_bridge_reference,
):
    source, fine = brownian_bridge_reference
    np.testing.assert_array_equal(
        fine.cumulative_return[:, 35::36], source.cumulative_return
    )
    np.testing.assert_array_equal(fine.years[35::36], source.years)
    np.testing.assert_array_equal(fine.parameter_means, source.parameter_means)
    np.testing.assert_array_equal(
        fine.simulation_info["source_parameter_sds"], source.parameter_sds
    )
    np.testing.assert_allclose(fine.parameter_sds, 10 * np.sqrt(12))
    np.testing.assert_allclose(
        fine.period_metrics.reshape(24000, 2, 36).mean(axis=2),
        source.period_metrics,
        atol=1e-11,
    )
    assert fine.period_years == pytest.approx(1 / 12)
    assert fine.metadata["source_period_years"] == 3
    assert fine.metadata["approximation"] == "Gaussian Brownian bridge"
    assert "source_parameter_sds" not in fine.metadata


def test_bridge_has_sqrt_time_brownian_limit(brownian_bridge_reference):
    _, fine = brownian_bridge_reference
    indices = np.array([2, 5, 11, 23, 35, 47, 71])
    years = fine.years[indices]
    cumulative = fine.cumulative_return[:, indices]
    np.testing.assert_allclose(cumulative.std(axis=0), 10 * np.sqrt(years), rtol=0.025)
    np.testing.assert_allclose(cumulative.mean(axis=0), 0.4 * years, atol=0.35)


def test_within_block_bridge_has_conditional_gaussian_covariance(
    brownian_bridge_reference,
):
    source, fine = brownian_bridge_reference
    times = np.array([0.5, 1.5, 2.5])
    indices = (times * 12).astype(int) - 1
    residual = fine.cumulative_return[:, indices] - source.cumulative_return[:, :1] * (
        times / 3
    )
    expected_covariance = 10**2 * (
        np.minimum.outer(times, times) - np.outer(times, times) / 3
    )
    np.testing.assert_allclose(
        np.cov(residual.T), expected_covariance, atol=1.5, rtol=0.04
    )
    assert not np.array_equal(
        fine.cumulative_return[:, indices],
        source.cumulative_return[:, :1] * (times / 3),
    )


def test_daily_bridge_keeps_labels_and_does_not_resample_worlds():
    class LabelledPosterior(FixedPosterior):
        metric = "ann_ret"

        def sample(self, n, rng=None):
            result = super().sample(n, rng)
            result["variant"] = ["a" if i % 2 else "b" for i in range(n)]
            result["component"] = "fixed-model"
            return result

    posterior = LabelledPosterior()
    source = simulate_metric_funnel(posterior, n_paths=20, n_periods=2, rng=3)
    fine = bridge_metric_funnel(source, 10, frequency="daily", rng=4)
    assert posterior.calls == [20]
    assert fine.n_periods == 2 * 3 * 252
    np.testing.assert_array_equal(
        fine.cumulative_return[:, 755::756], source.cumulative_return
    )
    np.testing.assert_array_equal(fine.construction_labels, source.construction_labels)
    np.testing.assert_array_equal(fine.component_labels, source.component_labels)
    np.testing.assert_array_equal(fine.parameter_means, source.parameter_means)
    assert fine.metadata["nominal_steps_per_year"] == 252


def test_bridge_preserves_nonconstant_expected_trajectory_instead_of_sampled_endpoints():
    class DynamicReturns(FixedPosterior):
        metric = "ann_ret"

        def sample_paths(self, n_paths, n_periods, rng=None):
            expected = np.broadcast_to([1.0, 2.0], (n_paths, n_periods))
            rng = np.random.default_rng(rng)
            return {
                "values": expected + rng.normal(size=(n_paths, n_periods)),
                "means": np.repeat(100.0, n_paths),
                "sds": np.ones(n_paths),
                "expected_values": expected,
                "years": [3, 6],
            }

    source = simulate_metric_funnel(DynamicReturns(), n_paths=100, n_periods=2, rng=1)
    fine = bridge_metric_funnel(source, 10, rng=2)
    np.testing.assert_allclose(fine.expected_cumulative_return[:36], fine.years[:36])
    np.testing.assert_allclose(
        fine.expected_cumulative_return[36:], 3 + 2 * (fine.years[36:] - 3)
    )
    np.testing.assert_allclose(
        fine.expected_path_draws("cumulative_return")[:, 35::36],
        source.expected_path_draws("cumulative_return"),
    )
    assert not np.allclose(fine.expected_cumulative_return, fine.years * 100)
    assert not np.allclose(
        fine.expected_cumulative_return[35::36], source.cumulative_return.mean(axis=0)
    )


def test_bridge_leaves_unknown_conditional_expectations_unspecified():
    class UnknownTrajectory(FixedPosterior):
        metric = "ann_ret"

        def sample_paths(self, n_paths, n_periods, rng=None):
            return {
                "values": np.ones((n_paths, n_periods)),
                "means": np.zeros(n_paths),
                "sds": np.ones(n_paths),
                "years": [3, 6],
            }

    source = simulate_metric_funnel(UnknownTrajectory(), n_paths=20, n_periods=2, rng=1)
    fine = bridge_metric_funnel(source, 10, rng=2)
    assert fine.expected_period_metrics is None
    assert fine.expected_path_draws("cumulative_return") is None


def test_sharpe_bridge_conversion_is_explicit_and_matches_return_units():
    sharpe_source = simulate_metric_funnel(
        FixedPosterior(mean=0.4, sd=0.8), n_paths=50, n_periods=2, rng=1
    )
    returns_posterior = FixedPosterior(mean=4, sd=8)
    returns_posterior.metric = "ann_ret"
    returns_source = simulate_metric_funnel(
        returns_posterior, n_paths=50, n_periods=2, rng=1
    )
    converted = bridge_metric_funnel(sharpe_source, 10, rng=3)
    returns = bridge_metric_funnel(returns_source, 10, rng=3)
    assert converted.metric == "ann_ret"
    assert converted.metadata["source_metric"] == "sharpe"
    np.testing.assert_allclose(
        converted.cumulative_return, returns.cumulative_return, atol=1e-12
    )
    np.testing.assert_allclose(
        converted.parameter_means, sharpe_source.parameter_means * 10
    )


def test_bridge_is_reproducible_and_follows_return_units():
    posterior = FixedPosterior(mean=4, sd=8)
    posterior.metric = "ann_ret"
    source = simulate_metric_funnel(posterior, n_paths=40, n_periods=2, rng=1)
    original = bridge_metric_funnel(source, 10, rng=2)
    repeated = bridge_metric_funnel(source, 10, rng=2)
    np.testing.assert_array_equal(
        original.cumulative_return, repeated.cumulative_return
    )
    scaled_posterior = FixedPosterior(mean=4e-6, sd=8e-6)
    scaled_posterior.metric = "ann_ret"
    scaled_source = simulate_metric_funnel(
        scaled_posterior, n_paths=40, n_periods=2, rng=1
    )
    scaled = bridge_metric_funnel(scaled_source, 10e-6, rng=2)
    np.testing.assert_allclose(
        scaled.cumulative_return / 1e-6,
        original.cumulative_return,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(scaled.parameter_sds / 1e-6, original.parameter_sds)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"annual_volatility": 0},
        {"annual_volatility": -1},
        {"annual_volatility": np.nan},
        {"annual_volatility": np.inf},
        {"frequency": "weekly"},
        {"max_points": 0},
        {"max_points": 1.5},
    ],
)
def test_bridge_rejects_invalid_approximation_configuration(kwargs):
    source = simulate_metric_funnel(FixedPosterior(), n_paths=10, n_periods=1, rng=1)
    arguments = {"annual_volatility": 10}
    arguments.update(kwargs)
    with pytest.raises((TypeError, ValueError)):
        bridge_metric_funnel(source, **arguments)


def test_bridge_resource_limit_rejects_large_expansion_without_resampling():
    posterior = FixedPosterior()
    source = simulate_metric_funnel(posterior, n_paths=100, n_periods=2, rng=1)
    with pytest.raises(ValueError, match="reduce n_paths"):
        bridge_metric_funnel(source, 10, frequency="daily", max_points=1000)
    assert posterior.calls == [100]


def test_bridge_rejects_volatility_scales_that_overflow_fine_interval_dispersion():
    posterior = FixedPosterior(mean=0, sd=1)
    posterior.metric = "ann_ret"
    source = simulate_metric_funnel(posterior, n_paths=2, n_periods=1, rng=1)
    with pytest.raises(ValueError, match="numerical precision"):
        bridge_metric_funnel(source, 1e308)


def test_bridge_rejects_nonaligned_duration_and_volatility_metric():
    posterior = FixedPosterior()
    posterior.period_years = 0.1
    source = simulate_metric_funnel(posterior, n_paths=10, n_periods=1, rng=1)
    with pytest.raises(ValueError, match="integer number"):
        bridge_metric_funnel(source, 10)
    posterior.metric = "ann_vol"
    posterior.mean = 10
    posterior.period_years = 3
    source = simulate_metric_funnel(posterior, n_paths=10, n_periods=1, rng=1)
    with pytest.raises(ValueError, match="annual returns"):
        bridge_metric_funnel(source, 10)
