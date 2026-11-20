"""Figure contracts for the fixed-period Bayesian PnL benchmark."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from macrosynergy.pnl import BayesianPnL
from macrosynergy.visuals import (
    plot_bayesian_posterior,
    plot_bayesian_predictive,
    plot_bayesian_prior_posterior,
    plot_bayesian_funnel,
)


def test_predictive_accepts_numpy_thresholds(fit):
    ax = plot_bayesian_predictive(fit, thresholds=np.array([0.0, 1.0]))
    ax.figure.canvas.draw()
    plt.close(ax.figure)


@pytest.mark.parametrize("thresholds", [[np.nan], [[0]], [np.inf]])
def test_predictive_validates_thresholds(fit, thresholds):
    with pytest.raises(ValueError, match="thresholds"):
        plot_bayesian_predictive(fit, thresholds=thresholds)


@pytest.fixture(scope="module")
def fit():
    return BayesianPnL([0.3, 0.9, -0.2, 0.5, 0.7])


def test_joint_parameter_plot_places_means_on_marginals(fit):
    fig = plot_bayesian_posterior(fit)
    main, top, right = fig.axes
    assert "expected" in main.get_ylabel()
    assert "across periods" in main.get_xlabel()
    assert not any(line.get_marker() == "o" for line in main.lines)
    assert any(
        np.allclose(line.get_xdata(), fit.expectation("sd"))
        for line in top.lines
        if len(line.get_xdata()) == 2
    )
    assert any(
        np.allclose(line.get_ydata(), fit.expectation("mean"))
        for line in right.lines
        if len(line.get_ydata()) == 2
    )
    fig.canvas.draw()
    plt.close(fig)


def test_joint_no_value_region_preserves_density_and_labels_parameter_probability(fit):
    fig = plot_bayesian_posterior(fit)
    main, _, marginal = fig.axes
    region = next(
        patch
        for patch in main.patches
        if "No positive long-term value" in patch.get_label()
    )
    assert region.get_y() + region.get_height() == pytest.approx(0)
    assert region.get_facecolor()[-1] == 0  # Hatching has no opaque background.
    assert region.get_hatch()
    assert f"{fit.prob_mean_below(0):.1%}" in region.get_label()
    assert main.collections  # Joint density and contours remain present.
    assert marginal.collections  # The mean marginal retains its density layers.
    fig.canvas.draw()
    plt.close(fig)


def test_predictive_accepts_existing_axes_and_does_not_show(fit, monkeypatch):
    monkeypatch.setattr(
        plt, "show", lambda: pytest.fail("Plotting should not call show().")
    )
    fig, ax = plt.subplots()
    actual = plot_bayesian_predictive(fit, ax=ax)
    assert actual is ax
    assert "Realised" in ax.get_xlabel()
    assert "predictive" in ax.get_title(loc="left")
    fig.canvas.draw()
    plt.close(fig)


def test_prior_diagnostic_contains_both_parameters_and_diagnostics(fit):
    fig = plot_bayesian_prior_posterior(fit.posterior)
    assert len(fig.axes) == 2
    assert fig.diagnostics.index.tolist() == ["mean", "sd"]
    assert (fig.diagnostics["contraction"] < 1).all()
    assert "Long-term expected" in fig.axes[0].get_xlabel()
    assert "Long-term expected" in fig.axes[0].get_title(loc="left")
    fig.canvas.draw()
    plt.close(fig)


def test_prior_diagnostic_labels_parameter_loss_probability_not_predictive_loss(fit):
    fig = plot_bayesian_prior_posterior(fit)
    mean_axis = fig.axes[0]
    labels = mean_axis.get_legend_handles_labels()[1]
    expected = fit.prob_mean_below(0)
    assert any(
        "No positive long-term value" in label and f"{expected:.1%}" in label
        for label in labels
    )
    assert fig.diagnostics.loc["mean", "prob_mean_nonpositive"] == pytest.approx(
        expected
    )
    assert expected != pytest.approx(fit.predictive_cdf(0)[0])
    tail = next(
        collection
        for collection in mean_axis.collections
        if "No positive long-term value" in collection.get_label()
    )
    assert all(path.vertices[:, 0].max() <= 0 for path in tail.get_paths())
    plt.close(fig)


def test_prior_diagnostic_does_not_label_low_volatility_as_no_value():
    model = BayesianPnL([9, 11, 10, 8], metric="ann_vol")
    fig = plot_bayesian_prior_posterior(model)
    assert not any(
        "No positive long-term value" in label
        for label in fig.axes[0].get_legend_handles_labels()[1]
    )
    plt.close(fig)


@pytest.mark.parametrize("metric", ["ann_ret", "ann_vol"])
def test_plots_label_return_and_volatility_units(metric):
    values = [3, 9, -2, 5] if metric == "ann_ret" else [9, 11, 10, 8]
    model = BayesianPnL(values, metric=metric)
    fig = plot_bayesian_posterior(model)
    assert "PnL units" in fig.axes[0].get_ylabel()
    plt.close(fig)


def test_decimal_volatility_is_not_mislabeled_as_percent():
    model = BayesianPnL([0.09, 0.11, 0.1, 0.08], metric="ann_vol", vol_target=0.1)
    fig = plot_bayesian_posterior(model)
    assert "PnL units" in fig.axes[0].get_ylabel()
    assert "(%)" not in fig.axes[0].get_ylabel()
    # A decimal expectation is displayed in its original units, not as 0.1%.
    assert not any("0.1%" in text.get_text() for ax in fig.axes for text in ax.texts)
    plt.close(fig)


@pytest.mark.parametrize(
    "plot",
    [plot_bayesian_posterior, plot_bayesian_predictive, plot_bayesian_prior_posterior],
)
def test_plots_reject_unrelated_inputs(plot):
    with pytest.raises(TypeError):
        plot([0.3, 0.4])


@pytest.mark.parametrize("flat", [False, True])
def test_coarse_joint_grid_handles_tied_or_constant_density_levels(flat):
    from macrosynergy.pnl.bayes_pnl import (
        GridPosterior,
        GridSpec,
        GridTruncationWarning,
    )

    if flat:
        posterior = GridPosterior(
            mu=np.array([-1.0, 1.0]),
            sd=np.array([0.1, 2.0]),
            log_post=np.zeros((2, 2)),
            mass=np.full((2, 2), 0.25),
            data=np.array([0.3]),
            prior=None,
        )
    else:
        with pytest.warns(GridTruncationWarning):
            posterior = BayesianPnL(
                [0.2, 0.4, 0.5],
                grid=GridSpec((-1, 1), (0.1, 2), 2, 2),
                method="grid",
            ).posterior
    figure = plot_bayesian_posterior(posterior)
    figure.canvas.draw()
    assert figure.axes[0].collections
    assert all(np.diff(axis.get_xlim())[0] > 0 for axis in figure.axes)
    plt.close(figure)


@pytest.mark.parametrize("levels", [[0], [1], [np.nan], [[0.5]]])
def test_joint_plot_rejects_invalid_hdr_levels(fit, levels):
    with pytest.raises(ValueError, match="hdr_levels"):
        plot_bayesian_posterior(fit, hdr_levels=levels)
    plt.close("all")


def test_predictive_plot_does_not_construct_visualization_grid(monkeypatch):
    posterior = BayesianPnL([0.3, 0.9, -0.2, 0.5, 0.7]).posterior
    monkeypatch.setattr(
        posterior, "to_grid", lambda: pytest.fail("Predictive plots need no grid.")
    )
    axis = plot_bayesian_predictive(posterior)
    axis.figure.canvas.draw()
    plt.close(axis.figure)


def test_funnel_marks_only_block_endpoints_and_separates_mean_uncertainty(
    fit, monkeypatch
):
    from macrosynergy.pnl.bayes_pnl.funnel import simulate_metric_funnel

    funnel = simulate_metric_funnel(fit, n_paths=3000, n_periods=3, rng=17)
    monkeypatch.setattr(
        plt, "show", lambda: pytest.fail("Funnel should not call show().")
    )
    figure = plot_bayesian_funnel(funnel)
    axis, risk_axis = figure.axes
    np.testing.assert_array_equal(risk_axis.get_xticks(), [3, 6, 9])
    first_interval = next(
        patch
        for patch in axis.patches
        if patch.get_label() == "95% predictive interval"
    )
    assert first_interval.get_x() + first_interval.get_width() / 2 == 3
    lower, upper = np.quantile(funnel.running_mean[:, 0], [0.025, 0.975])
    assert first_interval.get_y() == pytest.approx(lower)
    assert first_interval.get_y() + first_interval.get_height() == pytest.approx(upper)
    assert "Running average" in axis.get_ylabel()
    assert "Sharpe ratios" in axis.get_ylabel()
    assert "cumulative PnL" in figure.texts[-1].get_text()
    assert "pointwise" in figure.texts[-1].get_text()
    bars = risk_axis.patches
    np.testing.assert_allclose(
        [bar.get_height() for bar in bars],
        np.mean(funnel.running_mean <= 0, axis=0),
    )
    assert any(
        "long-term expectation" in label
        for label in axis.get_legend_handles_labels()[1]
    )
    figure.canvas.draw()
    plt.close(figure)


def test_return_funnel_uses_cumulative_arithmetic_pnl_and_expected_path():
    from macrosynergy.pnl.bayes_pnl.funnel import simulate_metric_funnel

    model = BayesianPnL([3.0, -2.0, 8.0, 1.0, 5.0], metric="ann_ret")
    funnel = simulate_metric_funnel(model, n_paths=3000, n_periods=3, rng=18)
    figure = plot_bayesian_funnel(funnel)
    axis, risk_axis = figure.axes
    assert "Cumulative arithmetic PnL" in axis.get_ylabel()
    expected_line = next(
        line for line in axis.lines if line.get_label() == "Expected cumulative PnL"
    )
    np.testing.assert_allclose(
        expected_line.get_ydata(),
        funnel.years * np.mean(funnel.parameter_means),
    )
    assert expected_line.get_linestyle() == "None"
    np.testing.assert_allclose(
        [bar.get_height() for bar in risk_axis.patches],
        np.mean(funnel.cumulative_return <= 0, axis=0),
    )
    assert "no daily-path coverage" in figure.texts[-1].get_text()
    figure.canvas.draw()
    plt.close(figure)


def test_funnel_accepts_fit_without_constructing_grid(monkeypatch):
    model = BayesianPnL([0.3, 0.9, -0.2, 0.5, 0.7])
    monkeypatch.setattr(
        model.posterior, "to_grid", lambda: pytest.fail("Funnels need no grid.")
    )
    figure = plot_bayesian_funnel(model, n_paths=2000, n_periods=2, rng=19)
    expected_line = next(
        line
        for line in figure.axes[0].lines
        if line.get_label().startswith("Posterior expected value")
    )
    np.testing.assert_allclose(expected_line.get_ydata(), model.expectation("mean"))
    plt.close(figure)


@pytest.mark.parametrize("levels", [[], [0], [1], [np.nan], [[0.5]]])
def test_funnel_rejects_invalid_probability_levels(fit, levels):
    with pytest.raises(ValueError, match="levels"):
        plot_bayesian_funnel(fit, levels=levels)


def test_funnel_rejects_cumulative_sharpe_and_invalid_quantity(fit):
    with pytest.raises(ValueError, match="quantity"):
        plot_bayesian_funnel(fit, quantity="daily_pnl")
    with pytest.raises(ValueError, match="annual-return|ann_ret"):
        plot_bayesian_funnel(
            fit, quantity="cumulative_return", n_paths=500, n_periods=2, rng=20
        )


def test_funnel_expected_path_preserves_nonconstant_model_dynamics():
    class ChangingExpectation:
        metric, period_years = "ann_ret", 3.0

        def expectation(self):
            return 0.0  # Long-term value differs from the current expected path.

        def sample_paths(self, n_paths, n_periods, rng=None):
            expected = np.broadcast_to([1.0, 2.0, 4.0], (n_paths, n_periods))
            return dict(
                values=expected
                + np.random.default_rng(rng).normal(size=expected.shape),
                expected_values=expected,
                means=np.zeros(n_paths),
                sds=np.ones(n_paths),
                years=np.array([3.0, 6.0, 9.0]),
            )

    figure = plot_bayesian_funnel(
        ChangingExpectation(), n_paths=1000, n_periods=3, rng=21
    )
    expected = next(
        line
        for line in figure.axes[0].lines
        if line.get_label() == "Expected cumulative PnL"
    )
    np.testing.assert_allclose(expected.get_ydata(), [3.0, 9.0, 21.0])
    plt.close(figure)


def test_funnel_does_not_invent_expected_path_uncertainty_from_long_term_mean():
    from macrosynergy.pnl.bayes_pnl.funnel import MetricFunnel

    observations = np.tile([1.0, 3.0], (100, 1))
    funnel = MetricFunnel(
        years=np.array([3.0, 6.0]),
        period_metrics=observations,
        running_mean=np.tile([1.0, 2.0], (100, 1)),
        parameter_means=np.zeros(100),
        parameter_sds=np.ones(100),
        metric="ann_ret",
        period_years=3.0,
    )
    figure = plot_bayesian_funnel(funnel)
    labels = figure.axes[0].get_legend_handles_labels()[1]
    assert not any("uncertainty in expected path" in label for label in labels)
    expected = next(
        line
        for line in figure.axes[0].lines
        if line.get_label() == "Expected cumulative PnL"
    )
    np.testing.assert_allclose(expected.get_ydata(), [3.0, 12.0])
    plt.close(figure)


def test_monthly_funnel_uses_bridge_and_preserves_historical_sample_counts():
    from macrosynergy.pnl.bayes_pnl import BayesianPnLVariants

    data = pd.DataFrame(
        [[1.0, 1.2], [-0.3, -0.1], [0.7, 0.4], [1.1, 1.0]],
        columns=["base", "alternative"],
    )
    model = BayesianPnLVariants(
        data,
        correlation=[[1, 0.8], [0.8, 1]],
        metric="ann_ret",
        sigma_grid=[2.0],
        tau_grid=[0.5],
    )
    figure = plot_bayesian_funnel(
        model,
        n_paths=1000,
        n_periods=2,
        frequency="monthly",
        annual_volatility=3.0,
        baseline_name="base",
        rng=31,
    )
    meta = figure.funnel_metadata
    assert meta["frequency"] == "monthly"
    assert meta["source_period_years"] == 3
    assert meta["construction"]["n_shared_periods"] == 4
    assert meta["construction"]["n_distinct_implementations"] == 2
    assert any("baseline + 1 alternatives" in text.get_text() for text in figure.texts)
    assert any(
        "4 shared 3-year observations" in text.get_text() for text in figure.texts
    )
    mean_line = next(
        line
        for line in figure.axes[0].lines
        if line.get_label() == "Expected cumulative PnL"
    )
    assert (
        len(mean_line.get_xdata()) == 73
    )  # Origin plus 72 months, not new historical observations.
    np.testing.assert_allclose(
        mean_line.get_ydata(), mean_line.get_xdata() * model.expectation()
    )
    assert figure.axes[0].collections
    assert not figure.axes[1].patches  # Smooth probabilities, not 72 overlapping bars.
    figure.canvas.draw()
    plt.close(figure)


def test_fine_funnel_requires_an_explicit_volatility_assumption(fit):
    with pytest.raises(ValueError, match="annual_volatility"):
        plot_bayesian_funnel(fit, frequency="monthly")
    with pytest.raises(ValueError, match="frequency"):
        plot_bayesian_funnel(fit, frequency="weekly")


def test_monthly_sharpe_conversion_scales_the_expected_pnl_as_well_as_bands(fit):
    figure = plot_bayesian_funnel(
        fit,
        n_paths=500,
        n_periods=1,
        frequency="monthly",
        annual_volatility=10.0,
        rng=32,
    )
    line = next(
        line
        for line in figure.axes[0].lines
        if line.get_label() == "Expected cumulative PnL"
    )
    np.testing.assert_allclose(
        line.get_ydata(), np.asarray(line.get_xdata()) * fit.expectation() * 10
    )
    assert figure.funnel_metadata["source_metric"] == "sharpe"
    assert figure.funnel_metadata["metric"] == "ann_ret"
    plt.close(figure)
