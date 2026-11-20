"""Figure contracts for the fixed-period Bayesian PnL benchmark."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from macrosynergy.pnl import BayesianPnL
from macrosynergy.visuals import (
    plot_bayesian_posterior,
    plot_bayesian_predictive,
    plot_bayesian_prior_posterior,
)


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
