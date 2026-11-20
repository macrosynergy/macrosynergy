"""Construction sensitivity of long-term expected performance."""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from .bayes_pnl import (
    ACCENT,
    BLUE,
    GRID,
    INK,
    INK_2,
    INK_3,
    _RC,
    _get_posterior,
    _style_ax,
    _title,
)


def plot_bayesian_variants(
    fit,
    baseline=None,
    baseline_name: Optional[str] = None,
    title: Optional[str] = None,
    figsize=(11.2, 7.4),
):
    """Show how construction differences affect expected-performance uncertainty.

    Accepts a BayesianPnLVariants joint fit. The blue density describes the
    long-term expectation of one implementation selected with the declared
    research weights. The grey comparison recentres every component at the
    same overall mean, preserving each component's posterior uncertainty while
    removing differences between their posterior means. It is a controlled
    comparison, not another fitted posterior or a portfolio distribution.

    For posterior component means m_v, variances s_v**2 and research weights q,
    within variance W=sum(q*s_v**2), between-centre variance
    B=sum(q*(m_v-mbar)**2), and total variance is W+B. This guarantees a variance
    difference relative to the recentred comparison, not wider credible
    intervals or widening relative to an original construction fitted alone.
    Uncertain construction effects within a component remain part of W.

    baseline optionally supplies a separate single-construction BayesianPnL or
    Posterior as a labelled reference. The caller must align its data, units
    and return/cost basis with the joint fit. baseline_name labels a declared
    construction and confirms that it is included in the reported counts.
    Counts describe declared alternatives and shared periods, never effective
    independent time observations. The figure does not include next-period
    performance noise, and R/C are fixed assumptions of the supplied fit.

    Returns a Matplotlib Figure with variance_decomposition metadata, without
    showing it or changing research weights.
    """
    from macrosynergy.pnl.bayes_pnl.variants import BayesianPnLVariants

    if not isinstance(fit, BayesianPnLVariants):
        raise TypeError("fit must be a BayesianPnLVariants joint fit.")
    if baseline_name is not None and baseline_name not in fit.input_names:
        raise ValueError("baseline_name must name a declared construction.")
    reference = None if baseline is None else _get_posterior(baseline)
    if reference is not None and (
        reference.metric != fit.metric or reference.period_years != fit.period_years
    ):
        raise ValueError(
            "The baseline reference must share the metric and period duration."
        )

    names = fit.names
    weights = fit.weights.reindex(names).to_numpy()
    means = np.array([fit.expectation(target=name) for name in names])
    deviations = np.array([fit.std(target=name) for name in names])
    decomposition = fit.uncertainty_decomposition()
    within = float(decomposition["within_variance"])
    between = float(decomposition["between_variance"])
    total = float(decomposition["total_variance"])
    center = float(weights @ means)
    share = between / total if total > 0 else 0.0
    bounds = list(fit.quantile([0.002, 0.998]))
    positive = weights > 0
    bounds.extend((center - 4 * deviations[positive]).tolist())
    bounds.extend((center + 4 * deviations[positive]).tolist())
    if reference is not None:
        bounds.extend(reference.quantile("mean", [0.002, 0.998]))
    lo, hi = min(bounds), max(bounds)
    padding = max(0.05 * (hi - lo), np.finfo(float).eps)
    coordinates = np.linspace(lo - padding, hi + padding, 400)
    complete_density = fit.density(coordinates)
    within_density = np.zeros_like(coordinates)
    weighted_densities = []
    for name, weight, mean in zip(names, weights, means):
        if weight > 0:
            weighted_densities.append(weight * fit.density(coordinates, target=name))
            within_density += weight * fit.density(
                coordinates + mean - center, target=name
            )

    metric_label = (
        "Expected long-term {:g}-year Sharpe ratio".format(fit.period_years)
        if fit.metric == "sharpe"
        else "Expected annual return (input PnL units)"
    )
    declared = len(fit.input_names)
    count_label = "{} distinct constructions | {} with positive research weight | {} shared {:g}-year periods".format(
        len(names), int(positive.sum()), fit.n_periods, fit.period_years
    )
    if declared != len(names):
        count_label += " | {} declared entries before aliases".format(declared)
    if baseline_name is not None:
        count_label += " | includes {}".format(baseline_name)

    with plt.rc_context(_RC):
        figure = plt.figure(figsize=figsize)
        grid = figure.add_gridspec(
            1,
            2,
            width_ratios=(3.5, 1.1),
            wspace=0.28,
            left=0.09,
            right=0.97,
            bottom=0.245,
            top=0.765,
        )
        axis = figure.add_subplot(grid[0, 0])
        variance_axis = figure.add_subplot(grid[0, 1])
        _style_ax(axis)
        _style_ax(variance_axis)
        axis.fill_between(coordinates, within_density, color=GRID, alpha=0.7)
        axis.plot(
            coordinates,
            within_density,
            color=INK_3,
            lw=1.5,
            ls="--",
            label="Within only: component means aligned",
        )
        for index, density in enumerate(weighted_densities):
            axis.plot(
                coordinates,
                density,
                color=BLUE[400],
                alpha=0.36,
                lw=1.0,
                label="Research-weighted component densities" if index == 0 else None,
            )
        axis.plot(
            coordinates,
            complete_density,
            color=BLUE[600],
            lw=2.5,
            label="Choose one construction: full distribution",
        )
        if reference is not None:
            axis.plot(
                coordinates,
                reference.density("mean", coordinates),
                color=INK,
                lw=1.2,
                ls="-.",
                label="Separate baseline fit (reference)",
            )
        top = max(float(complete_density.max()), float(within_density.max()))
        for index, (mean, weight) in enumerate(zip(means, weights)):
            if weight > 0:
                axis.plot(
                    [mean, mean],
                    [-0.075 * top, -0.02 * top],
                    color=BLUE[600],
                    lw=max(1.0, 5 * weight),
                    label=(
                        "Component posterior means"
                        if index == np.flatnonzero(positive)[0]
                        else None
                    ),
                )
        axis.axvline(center, color=ACCENT, lw=1.2, ls=":", label="Weighted expectation")
        axis.set_xlim(coordinates[0], coordinates[-1])
        axis.set_ylim(bottom=-0.10 * top)
        axis.set_xlabel(metric_label)
        axis.set_ylabel("Posterior density of expectation")
        axis.set_title("One plausible implementation", loc="left", fontsize=11, pad=12)
        axis.grid(axis="y", visible=False)

        variance_axis.bar(
            0,
            within,
            width=0.65,
            color=GRID,
            edgecolor=INK_3,
            lw=0.7,
            label="Within implementation (W)",
        )
        variance_axis.bar(
            0,
            between,
            bottom=within,
            width=0.65,
            color=ACCENT,
            label="Between posterior means (B)",
        )
        variance_axis.set_xlim(-0.8, 0.8)
        variance_axis.set_ylim(0, max(total * 1.42, np.finfo(float).eps))
        variance_axis.set_xticks([])
        variance_axis.set_ylabel("Posterior variance")
        variance_axis.set_title("Variance = W + B", loc="left", fontsize=11, pad=12)
        variance_axis.text(
            0.5,
            0.97,
            "{}\nfrom different\nconstruction means".format(
                "<0.1%" if 0 < share < 0.001 else "{:.1%}".format(share)
            ),
            transform=variance_axis.transAxes,
            ha="center",
            va="top",
            fontsize=10,
            color=INK_2,
        )
        variance_axis.legend(
            loc="upper center", bbox_to_anchor=(0.5, -0.04), fontsize=7.5
        )
        handles, labels = axis.get_legend_handles_labels()
        figure.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.51, 0.91),
            ncol=2,
            fontsize=8,
        )
        _title(
            figure,
            title or "How much do construction choices matter?",
            count_label,
            x=0.09,
            y=0.985,
        )
        weight_label = "; ".join(
            "{} {:.0%}".format(name, weight) for name, weight in zip(names, weights)
        )
        figure.text(
            0.09,
            0.155,
            "Declared research weights: " + weight_label,
            ha="left",
            va="top",
            fontsize=8,
            color=INK_2,
        )
        figure.text(
            0.09,
            0.11,
            "W preserves within-implementation uncertainty; B adds differences between posterior means.\n"
            "Neither curve includes future-period noise. Fixed correlation assumptions R/C; no selection correction.\n"
            "The centred comparison is controlled; credible intervals need not widen relative to a separate baseline fit.",
            ha="left",
            va="top",
            fontsize=8,
            color=INK_3,
        )
    figure.variance_decomposition = dict(decomposition)
    return figure
