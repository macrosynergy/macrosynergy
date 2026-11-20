"""Posterior, predictive and prior-diagnostic figures for BayesianPnL fits."""

from __future__ import annotations

from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

from macrosynergy.pnl.bayes_pnl.grid import _cell_widths


def _get_posterior(fit):
    from macrosynergy.pnl.bayes_pnl import BayesianPnL
    from macrosynergy.pnl.bayes_pnl.posterior import Posterior

    if isinstance(fit, BayesianPnL):
        return fit.posterior
    if isinstance(fit, Posterior):
        return fit
    raise TypeError("Expected a BayesianPnL fit or Posterior.")


# ------------------------------------------------------------------- style tokens
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_3 = "#8a8984"
GRID = "#e4e3df"
BLUE = {
    100: "#cde2fb",
    150: "#b7d3f6",
    200: "#9ec5f4",
    250: "#86b6ef",
    300: "#6da7ec",
    400: "#3987e5",
    500: "#256abf",
    600: "#184f95",
    700: "#0d366b",
}
ACCENT = "#eb6834"  # posterior mean lines
LOSS = "#d03b3b"  # status critical: probability-of-loss area, always labelled
DENSITY_CMAP = LinearSegmentedColormap.from_list(
    "density_blue",
    [
        SURFACE,
        BLUE[100],
        BLUE[200],
        BLUE[300],
        BLUE[400],
        BLUE[500],
        BLUE[600],
        BLUE[700],
    ],
)

_FONT = "DejaVu Sans"
_RC = {
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": _FONT,
    "font.size": 9.5,
    "axes.edgecolor": GRID,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelcolor": INK_2,
    "axes.labelsize": 9.5,
    "axes.titlecolor": INK,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "grid.linestyle": "-",
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "xtick.major.size": 0,
    "ytick.major.size": 0,
    "legend.frameon": False,
    "legend.fontsize": 8.5,
    "legend.labelcolor": INK_2,
}


def style():
    """Context manager with the house style, for building figures: ``with style(): fig, ax = plt.subplots()``."""
    return plt.rc_context(_RC)


def _style_ax(ax):
    """Apply the style to axes created outside ``style()`` (e.g. passed in by the caller)."""
    ax.figure.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(0.8)
    ax.grid(True, color=GRID, lw=0.8, ls="-")
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_2, labelsize=8.5, length=0)
    for t in (ax.title, ax.xaxis.label, ax.yaxis.label):
        t.set_fontfamily(_FONT)
    ax.xaxis.label.set_color(INK_2)
    ax.yaxis.label.set_color(INK_2)
    return ax


_NAMES = {
    "sharpe": ("Sharpe ratio", ""),
    "ann_ret": ("annual return", "PnL units"),
    "ann_vol": ("annual volatility", "PnL units"),
}
_PLURAL = {
    "Sharpe ratio": "Sharpe ratios",
    "annual return": "annual returns",
    "annual volatility": "annual volatilities",
}


def _metric_label(
    post, years: Optional[float] = None, plural: bool = False, unit: bool = True
) -> str:
    """Label the metric without assuming percent or decimal PnL units."""
    name, u = _NAMES.get(post.metric, (post.metric, ""))
    y = post.period_years if years is None else years
    return (
        f"{y:g}-year "
        + (_PLURAL.get(name, name + "s") if plural else name)
        + (f" ({u})" if unit and u else "")
    )


def _fmt(v: float, metric: str) -> str:
    return f"{v:.3g}" if metric in ("ann_ret", "ann_vol") else f"{v:.2f}"


def _title(
    fig_or_ax, title: str, subtitle: Optional[str], x: float = 0.0, y: float = 1.0
):
    """Left-aligned title with a smaller subtitle under it (figure coordinates for a figure)."""
    if hasattr(fig_or_ax, "suptitle"):
        fig_or_ax.text(
            x,
            y,
            title,
            ha="left",
            va="top",
            fontsize=13,
            fontweight="bold",
            color=INK,
            transform=fig_or_ax.transFigure,
        )
        if subtitle:
            fig_or_ax.text(
                x,
                y - 0.035,
                subtitle,
                ha="left",
                va="top",
                fontsize=9.5,
                color=INK_2,
                transform=fig_or_ax.transFigure,
            )
    else:
        fig_or_ax.set_title(
            title + ("\n" if subtitle else ""), loc="left", pad=14 if subtitle else 6
        )
        if subtitle:
            fig_or_ax.text(
                0,
                1.02,
                subtitle,
                ha="left",
                va="bottom",
                fontsize=9,
                color=INK_2,
                transform=fig_or_ax.transAxes,
            )


def _zoom(grid, dens: np.ndarray, mass_level: float = 0.995, pad: float = 0.12):
    """Bounding box (mean, sd) of the highest-density region holding ``mass_level`` of the mass."""
    thr = grid.hdr_thresholds((mass_level,))[mass_level]
    i, j = np.nonzero(dens >= thr)
    mlo, mhi = grid.mu[i.min()], grid.mu[i.max()]
    slo, shi = grid.sd[j.min()], grid.sd[j.max()]
    pm = max(pad * (mhi - mlo), 0.5 * _cell_widths(grid.mu)[i].max())
    ps = max(pad * (shi - slo), 0.5 * _cell_widths(grid.sd)[j].max())
    return (mlo - pm, mhi + pm), (max(slo - ps, 0.0), shi + ps)


def _marginal_density(post, grid, param):
    """Evaluate inference densities on explicitly selected plotting coordinates."""
    coordinates = grid.mu if param == "mean" else grid.sd
    return pd.Series(
        post.density(param, coordinates),
        index=pd.Index(coordinates, name=param),
        name="density",
    )


def _no_value_label(probability):
    return "No positive long-term value\nP(expectation <= 0 | data) = {:.1%}".format(
        probability
    )


def _shade_nonpositive(ax, coordinates, density, *, horizontal=False, label=None):
    """Shade a parameter-density tail, interpolating its boundary at zero."""
    tail = coordinates[coordinates <= 0]
    if coordinates[0] < 0 < coordinates[-1]:
        tail = np.append(tail, 0.0)
    fill = ax.fill_betweenx if horizontal else ax.fill_between
    return fill(
        tail,
        np.interp(tail, coordinates, density),
        color=LOSS,
        alpha=0.45,
        lw=0,
        label=label,
    )


def _plot_joint_density(ax, grid, density, hdr_levels):
    """Draw finite-grid regions, combining labels when regions share a threshold."""
    levels = np.asarray(hdr_levels, dtype=float)
    if (
        levels.ndim != 1
        or not np.isfinite(levels).all()
        or ((levels <= 0) | (levels >= 1)).any()
    ):
        raise ValueError(
            "hdr_levels must be a one-dimensional sequence between 0 and 1."
        )
    thresholds = grid.hdr_thresholds((0.995,) + tuple(levels))
    lower, upper = thresholds[0.995], float(density.max())
    if lower < upper:
        ax.contourf(
            grid.sd,
            grid.mu,
            density,
            levels=np.linspace(lower, upper, 14),
            cmap=DENSITY_CMAP,
        )
    else:
        # A valid coarse or flat grid may contain only one distinct density.
        ax.pcolormesh(grid.sd, grid.mu, density, shading="nearest", cmap=DENSITY_CMAP)
    labels = {}
    for level in levels:
        threshold = thresholds[level]
        if density.min() < threshold < upper:
            labels.setdefault(threshold, []).append("{:.0%}".format(level))
    if labels:
        contours = ax.contour(
            grid.sd,
            grid.mu,
            density,
            levels=sorted(labels),
            colors=INK_2,
            linewidths=0.8,
        )
        ax.clabel(
            contours,
            fmt={key: " / ".join(value) for key, value in labels.items()},
            fontsize=7.5,
            colors=INK_2,
            inline_spacing=2,
        )


# ------------------------------------------------------------------ joint posterior


def plot_bayesian_posterior(
    post,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    hdr_levels=(0.5, 0.8, 0.95),
    truth: Optional[Tuple[float, float]] = None,
    show_data: bool = True,
    figsize=(8.4, 7.4),
):
    """
    Joint posterior of the mean (vertical) and sd (horizontal) of the period metric:
    density shading, highest-density contours, marginal densities with 90%
    highest-density intervals, posterior means marked as accent lines on the
    marginals, and the observed period values
    (ticks along the mean margin). Axes are zoomed to the 99.5% highest-density region.
    ``truth=(mean, sd)`` marks known values (simulations).

    Accepts BayesianPnL or Posterior and returns a Matplotlib Figure.
    Return/volatility values retain the input PnL units; no percent conversion
    is applied. The plot does not call show(). Joint contours use the explicit
    finite-grid adapter; marginal densities and annotations use the fitted backend.
    For Sharpe/return metrics, transparent hatching marks non-positive long-term
    expectation, with matching posterior-tail shading on the mean marginal.
    """
    post = _get_posterior(post)
    grid = post.to_grid()
    dens = grid.mass / np.outer(_cell_widths(grid.mu), _cell_widths(grid.sd))
    (ylo, yhi), (xlo, xhi) = _zoom(grid, dens)
    label = _metric_label(post)
    em, es = post.expectation("mean"), post.expectation("sd")
    m_lo, m_hi = post.interval("mean", 0.9, "hdi")
    s_lo, s_hi = post.interval("sd", 0.9, "hdi")
    with plt.rc_context(_RC):
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(
            2,
            2,
            width_ratios=(4, 1),
            height_ratios=(1, 4),
            wspace=0.04,
            hspace=0.04,
            left=0.1,
            right=0.97,
            bottom=0.08,
            top=0.88,
        )
        ax = fig.add_subplot(gs[1, 0])
        ax_top = fig.add_subplot(gs[0, 0], sharex=ax)
        ax_right = fig.add_subplot(gs[1, 1], sharey=ax)

        _plot_joint_density(ax, grid, dens, hdr_levels)
        no_value_region = post.metric != "ann_vol" and ylo < 0
        if no_value_region:
            ax.add_patch(
                Rectangle(
                    (xlo, ylo),
                    xhi - xlo,
                    min(0, yhi) - ylo,
                    facecolor="none",
                    edgecolor=LOSS,
                    alpha=0.35,
                    hatch="///",
                    linewidth=0,
                    zorder=1.5,
                    label=_no_value_label(post.prob_mean_below(0)),
                )
            )
            ax.axhline(0, color=LOSS, lw=1, ls="--", zorder=3)
        if truth is not None:
            ax.plot(
                truth[1],
                truth[0],
                "X",
                ms=9,
                mfc=INK,
                mec=SURFACE,
                mew=1.5,
                zorder=6,
                label="true value",
            )
        if no_value_region or truth is not None:
            ax.legend(loc="lower right" if no_value_region else "upper right")
        ax.set_xlim(xlo, xhi)
        ax.set_ylim(ylo, yhi)
        ax.set_xlabel(
            f"Standard deviation of {_metric_label(post, plural=True, unit=False)} across periods"
            + (" (PnL units)" if post.metric != "sharpe" else "")
        )
        ax.set_ylabel(f"Long-term expected {label}")

        mx = _marginal_density(post, grid, "sd")
        my = _marginal_density(post, grid, "mean")
        for a, m, lo, hi, vertical in (
            (ax_top, mx, s_lo, s_hi, True),
            (ax_right, my, m_lo, m_hi, False),
        ):
            x, d = m.index.to_numpy(), m.to_numpy()
            inside = (x >= lo) & (x <= hi)
            if vertical:
                a.fill_between(x, d, color=BLUE[100], lw=0)
                a.fill_between(x, d, where=inside, color=BLUE[250], lw=0)
                a.plot(x, d, color=BLUE[500], lw=1.6)
                a.axvline(es, color=ACCENT, lw=1.6, label="posterior mean", zorder=5)
                a.set_ylim(0, d.max() * 1.25)
            else:
                a.fill_betweenx(x, d, color=BLUE[100], lw=0)
                a.fill_betweenx(x, d, where=inside, color=BLUE[250], lw=0)
                if post.metric != "ann_vol":
                    _shade_nonpositive(a, x, d, horizontal=True)
                    a.axhline(0, color=LOSS, lw=1, ls="--")
                a.plot(d, x, color=BLUE[500], lw=1.6)
                a.axhline(em, color=ACCENT, lw=1.6, label="posterior mean", zorder=5)
                a.set_xlim(0, d.max() * 1.25)
            a.grid(False)
            for sp in a.spines.values():
                sp.set_visible(False)
            a.tick_params(labelbottom=False, labelleft=False)
        ax_top.text(
            1.0,
            0.95,
            f"sd, 90% interval: {_fmt(s_lo, post.metric)} to {_fmt(s_hi, post.metric)}\n"
            f"Red line: posterior mean {_fmt(es, post.metric)}",
            transform=ax_top.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            color=INK_2,
        )
        ax_right.text(
            0.95,
            1.0,
            f"expectation, 90% interval:\n{_fmt(m_lo, post.metric)} to {_fmt(m_hi, post.metric)}\n"
            f"Red line: posterior\nmean {_fmt(em, post.metric)}",
            transform=ax_right.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            color=INK_2,
        )
        if show_data:
            span = ax_right.get_xlim()[1]
            for v in post.data:
                ax_right.plot(
                    [0, 0.12 * span], [v, v], color=INK, lw=1.2, solid_capstyle="butt"
                )
            ax_right.text(
                0.13 * span,
                ylo + 0.02 * (yhi - ylo),
                f"ticks: the {post.data.size}\nobserved periods",
                fontsize=7.5,
                color=INK_3,
                va="bottom",
            )

        _title(
            fig,
            title or "Long-term parameters: joint posterior",
            subtitle
            or (
                f"{post.data.size} periods · expected value {_fmt(em, post.metric)} (90%: {_fmt(m_lo, post.metric)} to {_fmt(m_hi, post.metric)})"
                f" · sd across periods {_fmt(es, post.metric)}"
                + (
                    f" · Pr(mean < 0) = {post.prob_mean_below(0):.0%}"
                    if post.metric != "ann_vol"
                    else ""
                )
            ),
            x=0.03,
            y=0.975,
        )
        fig.text(
            0.03,
            0.905,
            r"$p(\mu,\sigma\mid\mathrm{data})$: uncertainty about long-term expectation and dispersion across periods",
            ha="left",
            va="top",
            fontsize=8.5,
            color=INK_2,
        )
    return fig


def plot_bayesian_predictive(
    post, thresholds=(0.0,), ax=None, title: Optional[str] = None
):
    """Plot next-period realised performance, integrating parameter uncertainty.

    Accepts a BayesianPnL or Posterior and returns a Matplotlib Axes.
    Predictions refer to the same duration as the fitted observations.
    Return/volatility values retain the input PnL units.
    """
    post = _get_posterior(post)
    thresholds = np.asarray(thresholds, dtype=float)
    if thresholds.ndim != 1 or not np.isfinite(thresholds).all():
        raise ValueError("thresholds must be a one-dimensional finite sequence.")
    with plt.rc_context(_RC):
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 4.2))
        _style_ax(ax)
        lo, hi = post.predictive_quantile([0.002, 0.998])
        xs = np.linspace(lo, hi, 500)
        pdf = post.predictive_pdf(xs)
        q05, q50, q95 = post.predictive_quantile([0.05, 0.5, 0.95])
        t0 = min(thresholds) if len(thresholds) else -np.inf
        ax.fill_between(
            xs, pdf, where=xs >= t0, color=BLUE[150], lw=0, interpolate=True
        )
        for t in thresholds:
            p = post.predictive_cdf(t)[0]
            ax.fill_between(
                xs,
                pdf,
                where=xs <= t,
                color="#f6c9c9",
                lw=0,
                interpolate=True,
                label=f"Pr(< {_fmt(t, post.metric)}) = {p:.0%}",
            )
            ax.axvline(t, color=INK_2, lw=0.8)
        ax.plot(xs, pdf, color=BLUE[500], lw=2)
        base = -0.045 * pdf.max()
        ax.plot(
            [q05, q95],
            [base, base],
            color=BLUE[600],
            lw=3,
            solid_capstyle="round",
            label=f"90% range {_fmt(q05, post.metric)} to {_fmt(q95, post.metric)}",
        )
        ax.axvline(q50, color=BLUE[700], lw=1, ls=(0, (1, 2)))
        ax.annotate(
            f"median {_fmt(q50, post.metric)}",
            (q50, pdf.max() * 1.03),
            xytext=(5, 0),
            textcoords="offset points",
            ha="left",
            va="bottom",
            fontsize=8.5,
            color=INK_2,
        )
        ax.set_ylim(2.2 * base, pdf.max() * 1.15)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.grid(axis="y", visible=False)
        years = post.period_years
        ax.set_xlabel(f"Realised {_metric_label(post, years)}")
        ax.legend(loc="upper left")
        _title(
            ax,
            title
            or f"Next {_metric_label(post, years, unit=False)}: predictive distribution",
            "Period variation + parameter uncertainty",
        )
    return ax


def plot_bayesian_prior_posterior(post, title: Optional[str] = None, figsize=(12, 4.2)):
    """
    Diagnostic: prior and posterior marginal densities of the mean and of the sd
    across periods, with the observed period values (mean panel).

    Read it as: how much did the data teach us? Contraction = 1 - posterior sd /
    prior sd (near 0: the posterior is the prior; near 1: the data dominate).
    Shift = (posterior mean - prior mean) / prior sd (large: data and prior
    disagree; check the prior). Needs an ``IndependentPrior``.

    Accepts BayesianPnL or Posterior and returns a Matplotlib Figure with
    a diagnostics DataFrame attribute. Values retain the input PnL units.
    For Sharpe and return metrics, red shading marks posterior probability of
    a non-positive long-term expectation, not a negative next-period outcome
    or an exact point mass at zero. Volatility is not interpreted as value.
    """
    post = _get_posterior(post)
    prior = post.prior
    if not (
        hasattr(prior, "mean") and hasattr(prior, "sd") and hasattr(prior.sd, "ppf")
    ):
        raise TypeError(
            "plot_bayesian_prior_posterior needs an IndependentPrior (mean prior and sd prior)."
        )
    grid = post.to_grid()
    rows = {}
    with plt.rc_context(_RC):
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        for ax, param, pr in ((axes[0], "mean", prior.mean), (axes[1], "sd", prior.sd)):
            m = _marginal_density(post, grid, param)
            x_post, d_post = m.index.to_numpy(), m.to_numpy()
            # x range: the prior's central 90% and the posterior's 99.8% (prior tails would dominate otherwise)
            q = np.atleast_1d(pr.ppf([0.05, 0.95]))
            plo, phi = post.quantile(param, [0.001, 0.999])
            lo, hi = min(q[0], plo), max(q[-1], phi)
            pad = 0.05 * (hi - lo)
            lo, hi = lo - pad, hi + pad
            if param == "sd":
                lo = max(lo, 1e-9)
            xs = np.linspace(lo, hi, 600)
            d_prior = np.exp(pr.logpdf(xs))
            prior_mean = float(pr.mean)
            prior_sd = float(pr.sd)
            post_mean, post_sd = post.expectation(param), post.std(param)
            contraction = 1 - post_sd / prior_sd
            shift = (post_mean - prior_mean) / prior_sd
            rows[param] = dict(
                prior_mean=prior_mean,
                prior_sd=prior_sd,
                post_mean=post_mean,
                post_sd=post_sd,
                contraction=contraction,
                shift=shift,
            )
            ax.fill_between(xs, d_prior, color=GRID, lw=0)
            ax.plot(xs, d_prior, color=INK_3, lw=1.6, label="prior")
            ax.fill_between(x_post, d_post, color=BLUE[150], lw=0, alpha=0.9)
            ax.plot(x_post, d_post, color=BLUE[500], lw=2, label="posterior")
            top = max(d_prior.max(), d_post.max())
            if param == "mean":
                if post.metric != "ann_vol":
                    probability = post.prob_mean_below(0)
                    rows[param]["prob_mean_nonpositive"] = probability
                    _shade_nonpositive(
                        ax, x_post, d_post, label=_no_value_label(probability)
                    )
                    ax.axvline(0, color=INK_2, lw=0.9, ls="--")
                for v in post.data:
                    ax.plot(
                        [v, v],
                        [-0.06 * top, -0.01 * top],
                        color=INK,
                        lw=1.2,
                        solid_capstyle="butt",
                    )
                ax.text(
                    0.0,
                    -0.075 * top,
                    f"ticks: {post.data.size} observed periods",
                    fontsize=7.5,
                    color=INK_3,
                    va="top",
                    transform=ax.get_yaxis_transform(),
                )
            ax.set_ylim(-0.13 * top if param == "mean" else 0, top * 1.12)
            ax.set_xlim(lo, hi)
            ax.set_yticks([])
            ax.spines["left"].set_visible(False)
            ax.grid(axis="y", visible=False)
            what = (
                "Long-term expected"
                if param == "mean"
                else "Standard deviation across periods of the"
            )
            ax.set_xlabel(f"{what} {_metric_label(post, plural=param == 'sd')}")
            ax.legend(loc="best")
            _title(
                ax,
                (
                    "Long-term expected performance"
                    if param == "mean"
                    else "Dispersion across periods"
                ),
                f"Contraction {contraction:.0%}, shift {shift:+.1f} prior sd",
            )
        if title:
            fig.suptitle(
                title, x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK
            )
        fig.tight_layout()
    fig.diagnostics = pd.DataFrame(rows).T
    return fig
