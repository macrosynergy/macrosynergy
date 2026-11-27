"""Fixed mixture probabilities and whole-path component selection invariants."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from macrosynergy.pnl.bayes_pnl.funnel import simulate_metric_funnel
from macrosynergy.pnl.bayes_pnl.mixture import BayesianPnLMixture


class KnownNormal:
    metric = "sharpe"
    period_years = 3.0

    def __init__(self, mean, sd=0.4, variant=None):
        self.mean, self.sd, self.variant = mean, sd, variant
        self.sample_calls = []

    def expectation(self, param):
        return self.mean if param == "mean" else self.sd

    def prob_mean_below(self, threshold=0.0):
        return float(self.mean < threshold)

    def predictive_pdf(self, x):
        return stats.norm.pdf(np.atleast_1d(x), self.mean, self.sd)

    def predictive_cdf(self, x):
        return stats.norm.cdf(np.atleast_1d(x), self.mean, self.sd)

    def sample(self, n, rng=None):
        self.sample_calls.append(n)
        result = pd.DataFrame(
            {"mean": np.repeat(self.mean, n), "sd": np.repeat(self.sd, n)}
        )
        if self.variant is not None:
            result["variant"] = self.variant
        return result


def make_mixture():
    return BayesianPnLMixture(
        {"low": KnownNormal(-0.3, 0.4), "high": KnownNormal(0.7, 0.6)},
        weights={"low": 1, "high": 3},
    )


def test_weighted_densities_and_parameter_expectations_match_analytic_mixture():
    mixture = make_mixture()
    values = np.array([-2, -0.1, 0.4, 1.2, 3])
    expected_pdf = 0.25 * stats.norm.pdf(values, -0.3, 0.4) + 0.75 * stats.norm.pdf(
        values, 0.7, 0.6
    )
    expected_cdf = 0.25 * stats.norm.cdf(values, -0.3, 0.4) + 0.75 * stats.norm.cdf(
        values, 0.7, 0.6
    )
    np.testing.assert_allclose(mixture.predictive_pdf(values), expected_pdf)
    np.testing.assert_allclose(mixture.predictive_cdf(values), expected_cdf)
    assert mixture.expectation("mean") == pytest.approx(0.45)
    assert mixture.expectation("sd") == pytest.approx(0.55)
    assert mixture.prob_mean_below(0) == pytest.approx(0.25)
    assert mixture.metric == "sharpe" and mixture.period_years == 3
    assert mixture.weight_role == "research"
    np.testing.assert_allclose(mixture.predictive_cdf([-np.inf, np.inf]), [0, 1])


def test_exact_duplicate_with_split_weight_leaves_the_distribution_unchanged():
    original = make_mixture()
    duplicate = BayesianPnLMixture(
        {
            "low1": KnownNormal(-0.3, 0.4),
            "low2": KnownNormal(-0.3, 0.4),
            "high": KnownNormal(0.7, 0.6),
        },
        weights=[0.1, 0.15, 0.75],
    )
    values = np.linspace(-3, 3, 41)
    np.testing.assert_allclose(
        duplicate.predictive_pdf(values), original.predictive_pdf(values), atol=1e-15
    )
    np.testing.assert_allclose(
        duplicate.predictive_cdf(values), original.predictive_cdf(values), atol=1e-15
    )
    assert duplicate.expectation() == pytest.approx(original.expectation(), abs=1e-15)
    assert duplicate.prob_mean_below(0) == original.prob_mean_below(0)


def test_named_weights_are_order_independent_and_returned_as_a_copy():
    original = make_mixture()
    reversed_mixture = BayesianPnLMixture(
        {"high": KnownNormal(0.7, 0.6), "low": KnownNormal(-0.3, 0.4)},
        weights={"low": 1, "high": 3},
    )
    np.testing.assert_allclose(
        reversed_mixture.predictive_cdf([-1, 0, 1]), original.predictive_cdf([-1, 0, 1])
    )
    snapshot = original.weights
    snapshot.iloc[:] = 0
    np.testing.assert_allclose(original.weights, [0.25, 0.75])


def test_zero_weight_component_is_never_evaluated_or_sampled():
    class UnusableMember(KnownNormal):
        def expectation(self, param):
            raise AssertionError("Zero-weight member was evaluated.")

        def prob_mean_below(self, threshold=0.0):
            raise AssertionError("Zero-weight member was evaluated.")

        def predictive_pdf(self, x):
            raise AssertionError("Zero-weight member was evaluated.")

        def predictive_cdf(self, x):
            raise AssertionError("Zero-weight member was evaluated.")

        def sample(self, n, rng=None):
            raise AssertionError("Zero-weight member was sampled.")

    mixture = BayesianPnLMixture(
        {"only": KnownNormal(0.2), "ignored": UnusableMember(100)}, weights=[1, 0]
    )
    assert mixture.expectation() == 0.2
    assert mixture.prob_mean_below(0) == 0
    np.testing.assert_allclose(
        mixture.predictive_pdf([0]), stats.norm.pdf([0], 0.2, 0.4)
    )
    np.testing.assert_allclose(
        mixture.predictive_cdf([0]), stats.norm.cdf([0], 0.2, 0.4)
    )
    assert (mixture.sample(100, rng=3).component == "only").all()
    paths = mixture.sample_paths(100, 3, rng=3)
    assert (paths["component"] == "only").all()


def test_parameter_sampling_preserves_component_and_nested_variant_identity():
    low, high = KnownNormal(-2, variant="construction_a"), KnownNormal(2)
    mixture = BayesianPnLMixture(
        {"model_a": low, "model_b": high}, [0.3, 0.7], weight_role="predictive_pool"
    )
    draws = mixture.sample(10000, rng=7)
    assert len(low.sample_calls) == len(high.sample_calls) == 1
    assert len(draws) == 10000
    assert np.mean(draws.component == "model_a") == pytest.approx(0.3, abs=0.02)
    assert (
        draws.loc[draws.component == "model_a", "variant"] == "construction_a"
    ).all()
    assert (draws.loc[draws.component == "model_b", "variant"] == "model_b").all()
    np.testing.assert_array_equal(draws["mean"] < 0, draws.component == "model_a")
    pd.testing.assert_frame_equal(draws, mixture.sample(10000, rng=7))


def test_future_paths_retain_the_selected_components_temporal_dependence():
    class PersistentPaths(KnownNormal):
        def sample(self, n, rng=None):
            raise AssertionError("An explicit complete-path law must take precedence.")

        def sample_paths(self, n_paths, n_periods, rng=None):
            rng = np.random.default_rng(rng)
            outcome = rng.normal(self.mean, self.sd, n_paths)
            return {
                "values": np.repeat(outcome[:, None], n_periods, axis=1),
                "means": np.repeat(self.mean, n_paths),
                "sds": np.repeat(self.sd, n_paths),
                "years": 3 * np.arange(1, n_periods + 1),
                "variant": np.repeat("persistent", n_paths),
            }

    mixture = BayesianPnLMixture(
        {"low": PersistentPaths(-2), "high": PersistentPaths(2)},
        weights=[0.4, 0.6],
        weight_role="predictive_pool",
    )
    result = simulate_metric_funnel(mixture, n_paths=10000, n_periods=4, rng=23)
    np.testing.assert_array_equal(
        result.period_metrics, np.repeat(result.period_metrics[:, :1], 4, axis=1)
    )
    np.testing.assert_array_equal(
        result.parameter_means < 0, result.component_labels == "low"
    )
    assert (result.construction_labels == "persistent").all()
    assert np.mean(result.component_labels == "low") == pytest.approx(0.4, abs=0.02)
    assert "selected component" in result.metadata["conditional_observations"]


def test_mixture_sampler_matches_analytic_next_period_probabilities():
    mixture = make_mixture()
    draws = mixture.predictive_sample(40000, rng=51)
    for value in (-0.5, 0, 0.5, 1.5):
        assert np.mean(draws <= value) == pytest.approx(
            mixture.predictive_cdf(value)[0], abs=0.008
        )
    np.testing.assert_array_equal(draws, mixture.predictive_sample(40000, rng=51))


def test_path_mixture_preserves_nonconstant_conditional_mean_trajectories():
    class DynamicReturns(KnownNormal):
        metric = "ann_ret"

        def sample_paths(self, n_paths, n_periods, rng=None):
            expected = np.broadcast_to(
                self.mean + np.arange(n_periods), (n_paths, n_periods)
            )
            return {
                "values": expected.copy(),
                "expected_values": expected,
                "means": np.repeat(100.0, n_paths),
                "sds": np.ones(n_paths),
                "years": 3 * np.arange(1, n_periods + 1),
            }

    mixture = BayesianPnLMixture(
        {"early": DynamicReturns(1), "late": DynamicReturns(5)},
        weights=[0.4, 0.6],
        weight_role="predictive_pool",
    )
    result = simulate_metric_funnel(mixture, n_paths=1000, n_periods=3, rng=29)
    np.testing.assert_array_equal(result.expected_period_metrics, result.period_metrics)
    np.testing.assert_allclose(
        result.expected_cumulative_return, result.cumulative_return.mean(axis=0)
    )
    np.testing.assert_allclose(
        result.expected_running_mean, result.running_mean.mean(axis=0)
    )
    np.testing.assert_allclose(
        result.expected_path_draws("cumulative_return"), result.cumulative_return
    )
    assert result.expected_cumulative_return[0] < 15.1
    assert result.parameter_means.mean() == 100


def test_mixture_of_real_variant_fits_preserves_targets_labels_and_expected_paths():
    from macrosynergy.pnl.bayes_pnl.variants import BayesianPnLVariants

    data = pd.DataFrame([[0.2, 0.4], [-0.3, 0.1], [0.8, 0.6]], columns=["a", "b"])
    first = BayesianPnLVariants(
        data,
        correlation=[[1, 0.7], [0.7, 1]],
        sigma_grid=[0.7],
        tau_grid=[0.3],
        weights=[0.2, 0.8],
    )
    second = BayesianPnLVariants(
        data + 0.5,
        correlation=[[1, 0.4], [0.4, 1]],
        sigma_grid=[1.1],
        tau_grid=[0.2],
        weights=[0.7, 0.3],
    )
    mixture = BayesianPnLMixture(
        {"first": first, "second": second}, [0.25, 0.75], weight_role="predictive_pool"
    )
    for parameter in ("mean", "sd"):
        expected = 0.25 * first.expectation(parameter) + 0.75 * second.expectation(
            parameter
        )
        assert mixture.expectation(parameter) == pytest.approx(expected)
    assert mixture.expectation("sd") == pytest.approx(1.0)
    assert mixture.prob_mean_below(0) == pytest.approx(
        0.25 * first.prob_mean_below(0) + 0.75 * second.prob_mean_below(0),
    )
    draws = mixture.sample(400, rng=10)
    assert set(draws.component) == {"first", "second"}
    assert set(draws.variant) == {"a", "b"}
    np.testing.assert_array_equal(
        draws.sd, np.where(draws.component == "first", 0.7, 1.1)
    )
    result = simulate_metric_funnel(mixture, n_paths=400, n_periods=3, rng=11)
    assert set(result.component_labels) == {"first", "second"}
    assert set(result.construction_labels) == {"a", "b"}
    np.testing.assert_array_equal(
        result.expected_period_metrics,
        np.repeat(result.parameter_means[:, None], 3, axis=1),
    )
    np.testing.assert_array_equal(
        result.parameter_sds, np.where(result.component_labels == "first", 0.7, 1.1)
    )


def test_missing_conditional_expectations_in_one_member_do_not_create_a_flat_band():
    class UnknownTrajectory(KnownNormal):
        def sample_paths(self, n_paths, n_periods, rng=None):
            return {
                "values": np.ones((n_paths, n_periods)),
                "means": np.zeros(n_paths),
                "sds": np.ones(n_paths),
                "years": 3 * np.arange(1, n_periods + 1),
            }

    mixture = BayesianPnLMixture(
        {"known": KnownNormal(0.3), "unknown": UnknownTrajectory(0)}, [0.5, 0.5]
    )
    result = simulate_metric_funnel(mixture, n_paths=100, n_periods=3, rng=5)
    assert set(result.component_labels) == {"known", "unknown"}
    assert result.expected_period_metrics is None
    assert result.expected_path_draws() is None


@pytest.mark.parametrize(
    "weights",
    [
        [0, 0],
        [-1, 2],
        [np.inf, 1],
        [np.nan, 1],
        [1],
        [[1, 2]],
        [1 + 1j, 2],
        {"low": 1},
        {"low": 1, "wrong": 2},
    ],
)
def test_invalid_weights_are_rejected(weights):
    with pytest.raises((TypeError, ValueError)):
        BayesianPnLMixture({"low": KnownNormal(-1), "high": KnownNormal(1)}, weights)


def test_large_finite_weights_normalize_without_overflow():
    mixture = BayesianPnLMixture(
        {"low": KnownNormal(-1), "high": KnownNormal(1)}, [1e308, 1e308]
    )
    np.testing.assert_array_equal(mixture.weights, [0.5, 0.5])


@pytest.mark.parametrize("mismatch", ["metric", "duration"])
def test_incompatible_metric_targets_cannot_be_pooled(mismatch):
    incompatible = KnownNormal(1)
    if mismatch == "metric":
        incompatible.metric = "ann_ret"
    else:
        incompatible.period_years = 1
    with pytest.raises(ValueError, match="same metric"):
        BayesianPnLMixture({"first": KnownNormal(0), "second": incompatible})


def test_empty_mixture_invalid_role_and_invalid_parameter_are_rejected():
    with pytest.raises(ValueError):
        BayesianPnLMixture({})
    with pytest.raises(ValueError):
        BayesianPnLMixture(
            {"one": KnownNormal(0)}, weight_role="posterior_model_probability"
        )
    with pytest.raises(TypeError):
        BayesianPnLMixture({"one": object()})
    with pytest.raises(ValueError):
        make_mixture().expectation("variance")


def test_empty_samples_have_stable_shapes_and_columns():
    mixture = make_mixture()
    draws = mixture.sample(0, rng=1)
    assert list(draws.columns) == ["mean", "sd", "component", "variant"]
    assert len(draws) == 0
    assert mixture.predictive_sample(0, rng=1).shape == (0,)
    assert mixture.sample_paths(0, 3, rng=1)["values"].shape == (0, 3)
