"""Construction-importance figures distinguish controlled and baseline comparisons."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from macrosynergy.pnl import BayesianPnL
from macrosynergy.pnl.bayes_pnl import BayesianPnLVariants
from macrosynergy.visuals import plot_bayesian_variants


@pytest.fixture(scope="module")
def construction_fit():
    values = np.array([-0.3, 0.2, 0.1, -0.1])
    matrix = pd.DataFrame(
        {"baseline": values, "growth": values + 0.7, "inflation": values - 0.4}
    )
    return BayesianPnLVariants(
        matrix,
        correlation=0.7 * np.ones((3, 3)) + 0.3 * np.eye(3),
        weights=[0.5, 0.25, 0.25],
        sigma_grid=[0.6],
        tau_grid=[1.0],
    )


def test_variants_plot_uses_true_mixture_and_controlled_recentred_densities(
    construction_fit, monkeypatch
):
    monkeypatch.setattr(plt, "show", lambda: pytest.fail("Plot must not call show()."))
    fit = construction_fit
    figure = plot_bayesian_variants(fit, baseline_name="baseline")
    axis, variance_axis = figure.axes
    full = next(
        line
        for line in axis.lines
        if line.get_label() == "Choose one construction: full distribution"
    )
    within = next(
        line
        for line in axis.lines
        if line.get_label() == "Within only: component means aligned"
    )
    coordinates = full.get_xdata()
    np.testing.assert_allclose(full.get_ydata(), fit.density(coordinates))
    center = fit.expectation()
    expected_within = sum(
        fit.weights[name]
        * fit.density(coordinates + fit.expectation(target=name) - center, target=name)
        for name in fit.names
    )
    np.testing.assert_allclose(within.get_ydata(), expected_within)
    decomposition = figure.variance_decomposition
    assert decomposition["between_variance"] > 0
    assert decomposition["total_variance"] == pytest.approx(fit.std() ** 2)
    assert decomposition["total_variance"] == pytest.approx(
        decomposition["within_variance"] + decomposition["between_variance"]
    )
    np.testing.assert_allclose(
        [rectangle.get_height() for rectangle in variance_axis.patches],
        [decomposition["within_variance"], decomposition["between_variance"]],
    )
    figure.canvas.draw()
    plt.close(figure)


def test_equal_component_means_have_no_between_center_variance():
    values = np.array([-0.2, 0.1, 0.4, 0.2])
    data = pd.DataFrame({"baseline": values, "alternative": values})
    fit = BayesianPnLVariants(
        data, correlation=np.eye(2), sigma_grid=[0.6], tau_grid=[0.0]
    )
    figure = plot_bayesian_variants(fit)
    assert figure.variance_decomposition["between_variance"] == pytest.approx(
        0, abs=1e-15
    )
    axis = figure.axes[0]
    full = next(
        line
        for line in axis.lines
        if line.get_label() == "Choose one construction: full distribution"
    )
    within = next(
        line
        for line in axis.lines
        if line.get_label() == "Within only: component means aligned"
    )
    np.testing.assert_allclose(full.get_ydata(), within.get_ydata())
    assert any("0%" in text.get_text() for text in figure.axes[1].texts)
    plt.close(figure)


def test_baseline_is_optional_separate_reference_not_within_only_comparison(
    construction_fit,
):
    baseline = BayesianPnL(construction_fit.data[:, 0])
    figure = plot_bayesian_variants(
        construction_fit, baseline=baseline, baseline_name="baseline"
    )
    reference = next(
        line
        for line in figure.axes[0].lines
        if line.get_label() == "Separate baseline fit (reference)"
    )
    np.testing.assert_allclose(
        reference.get_ydata(), baseline.posterior.density("mean", reference.get_xdata())
    )
    text = "\n".join(label.get_text() for label in figure.texts)
    assert "includes baseline" in text
    assert "credible intervals need not widen" in text
    assert "4 shared 3-year periods" in text
    assert "future-period noise" in text
    plt.close(figure)


def test_counts_distinguish_declared_positive_weight_and_independent_periods(
    construction_fit,
):
    fit = BayesianPnLVariants(
        pd.DataFrame(construction_fit.data, columns=construction_fit.names),
        correlation=construction_fit.correlation,
        weights=[0.7, 0.3, 0.0],
        sigma_grid=[0.6],
        tau_grid=[1.0],
    )
    figure = plot_bayesian_variants(fit, baseline_name="baseline")
    text = "\n".join(label.get_text() for label in figure.texts)
    assert "3 distinct constructions" in text
    assert "2 with positive research weight" in text
    assert "4 shared 3-year periods" in text
    assert "inflation 0%" in text
    assert "effective" not in text
    plt.close(figure)


def test_counts_separate_declared_aliases_from_distinct_constructions(construction_fit):
    values = pd.DataFrame(construction_fit.data[:, :2], columns=["baseline", "growth"])
    values["copy"] = values.baseline
    correlation = construction_fit.correlation[np.ix_([0, 1, 0], [0, 1, 0])]
    fit = BayesianPnLVariants(
        values,
        correlation=correlation,
        aliases={"copy": "baseline"},
        weights=[0.25, 0.5, 0.25],
        sigma_grid=[0.6],
        tau_grid=[1.0],
    )
    figure = plot_bayesian_variants(fit, baseline_name="baseline")
    text = "\n".join(label.get_text() for label in figure.texts)
    assert "2 distinct constructions" in text
    assert "3 declared entries before aliases" in text
    assert "4 shared 3-year periods" in text
    assert "baseline 50%" in text
    plt.close(figure)


def test_variants_plot_rejects_incompatible_models_and_baseline(construction_fit):
    with pytest.raises(TypeError, match="joint fit"):
        plot_bayesian_variants(BayesianPnL([0.2, 0.3]))
    with pytest.raises(ValueError, match="declared construction"):
        plot_bayesian_variants(construction_fit, baseline_name="missing")
    with pytest.raises(ValueError, match="metric and period"):
        plot_bayesian_variants(
            construction_fit, baseline=BayesianPnL([1.0, 2.0], metric="ann_ret")
        )
