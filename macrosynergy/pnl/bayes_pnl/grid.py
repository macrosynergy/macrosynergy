"""
Bayesian grid approximation for the mean and standard deviation of a short
vector of period performance metrics (e.g. Sharpe ratios, returns or
volatilities of 3-year periods).

Model
-----
    x_i ~ Normal(mean, sd)             (Sharpe/return likelihood)
    x_i ~ Gamma(mean, sd)              (positive annual-volatility likelihood)
    (mean, sd) ~ prior                 (e.g. ``IndependentPrior``; ``default_prior(metric)``)

The posterior is approximated on a rectangular grid of (mean, sd) and
normalised over the grid cells. Everything downstream (marginals, intervals,
the predictive distribution of the next period's metric) is
computed deterministically from the grid: the predictive mixes the chosen
observation law over cells, weighted by their posterior mass.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import special, stats

from .priors import IndependentPrior
from .inputs import (
    ModelInputs,
    probabilities as _probabilities,
    sample_size as _sample_size,
    evaluation_points,
    parameter_values,
)
from .posterior import Posterior
from .likelihoods import get_likelihood

Param = str


class GridTruncationWarning(UserWarning):
    """Material posterior mass sits at the edge of the grid (grid too narrow)."""


# --------------------------------------------------------------------------- grid


@dataclass(frozen=True)
class GridSpec:
    """Rectangular mean/sd grid, with bounds in the fitted metric's units.

    The sd bounds must be positive. A logarithmic sd axis is supported;
    posterior cell masses include the unequal cell widths.
    """

    mean_range: Tuple[float, float]
    sd_range: Tuple[float, float]
    n_mean: int = 400
    n_sd: int = 400
    sd_spacing: str = "linear"
    diagnostics: dict = field(default_factory=dict, compare=False, repr=False)

    def __post_init__(self):
        lo, hi = self.mean_range
        slo, shi = self.sd_range
        if not np.isfinite([lo, hi, slo, shi]).all() or not (hi > lo and shi > slo > 0):
            raise ValueError(
                "Need mean_range[1] > mean_range[0] and sd_range[1] > sd_range[0] > 0."
            )
        for count in (self.n_mean, self.n_sd):
            if (
                isinstance(count, bool)
                or not isinstance(count, (int, np.integer))
                or count < 2
            ):
                raise ValueError("Grid dimensions must be integers of at least two.")
        if self.sd_spacing not in ("linear", "log"):
            raise ValueError("sd_spacing must be 'linear' or 'log'.")

    def axes(self) -> Tuple[np.ndarray, np.ndarray]:
        mu = np.linspace(*self.mean_range, self.n_mean)
        if self.sd_spacing == "log":
            sd = np.geomspace(*self.sd_range, self.n_sd)
        else:
            sd = np.linspace(*self.sd_range, self.n_sd)
        return mu, sd


def _cell_widths(x: np.ndarray) -> np.ndarray:
    if x.size == 1:
        return np.ones(1)
    return np.gradient(x)


# ------------------------------------------------------------------ posterior obj


@dataclass
class GridPosterior(Posterior):
    """Finite-grid posterior and same-duration predictive distribution.

    Construct through posterior_grid or BayesianPnL. Parameter summaries are
    distinct from next-period realised-performance predictions.
    """

    mu: np.ndarray
    sd: np.ndarray
    log_post: np.ndarray  # unnormalised log density on the grid
    mass: np.ndarray  # posterior mass per cell, sums to 1
    data: np.ndarray
    prior: object
    likelihood_weight: float = 1.0
    log_evidence: Optional[float] = (
        None  # normalised prior and untempered likelihood only
    )
    period_years: float = 3.0
    metric: str = "sharpe"  # "sharpe", "ann_ret" (% of AUM) or "ann_vol" (%)

    method = "grid"

    def to_grid(self):
        """This backend already is a finite grid."""
        return self

    def density(self, param, values):
        """Piecewise-linear marginal density at explicit coordinates."""
        values = evaluation_points(values)
        marginal = self.marginal_density(param)
        return np.interp(
            values, marginal.index.to_numpy(), marginal.to_numpy(), left=0.0, right=0.0
        )

    # ---------------------------------------------------------------- basics
    def _axis(self, param: Param) -> np.ndarray:
        if param not in ("mean", "sd"):
            raise ValueError("param must be 'mean' or 'sd'.")
        return {"mean": self.mu, "sd": self.sd}[param]

    def marginal(self, param: Param) -> pd.Series:
        m = self.mass.sum(axis=1) if param == "mean" else self.mass.sum(axis=0)
        return pd.Series(m, index=pd.Index(self._axis(param), name=param), name="mass")

    def marginal_density(self, param: Param) -> pd.Series:
        m = self.marginal(param)
        return m / _cell_widths(self._axis(param))

    def expectation(self, param: Param) -> float:
        m = self.marginal(param)
        return float((m.index.to_numpy() * m.to_numpy()).sum())

    def std(self, param: Param) -> float:
        m = self.marginal(param)
        x = m.index.to_numpy()
        e = (x * m.to_numpy()).sum()
        return float(np.sqrt(((x - e) ** 2 * m.to_numpy()).sum()))

    def cdf(self, param: Param, value) -> np.ndarray:
        """Marginal posterior CDF (piecewise linear between cell centres)."""
        value = parameter_values(value)
        m = self.marginal(param)
        x = m.index.to_numpy()
        c = np.cumsum(m.to_numpy()) - 0.5 * m.to_numpy()
        return np.interp(value, x, c, left=0.0, right=1.0)

    def quantile(self, param: Param, q) -> np.ndarray:
        _probabilities(q)
        m = self.marginal(param)
        x = m.index.to_numpy()
        c = np.cumsum(m.to_numpy()) - 0.5 * m.to_numpy()
        return np.interp(q, c, x)

    def mode(self) -> Tuple[float, float]:
        dens = self.mass / np.outer(_cell_widths(self.mu), _cell_widths(self.sd))
        i, j = np.unravel_index(np.argmax(dens), dens.shape)
        return float(self.mu[i]), float(self.sd[j])

    def interval(
        self, param: Param, level: float = 0.9, kind: str = "eti"
    ) -> Tuple[float, float]:
        if not np.isfinite(level) or not 0 < level < 1:
            raise ValueError("level must lie strictly between zero and one.")
        if kind not in ("eti", "hdi"):
            raise ValueError("kind must be 'eti' or 'hdi'.")
        if kind == "eti":
            lo, hi = self.quantile(param, [(1 - level) / 2, (1 + level) / 2])
            return float(lo), float(hi)
        dens = self.marginal_density(param)
        mass = self.marginal(param).to_numpy()
        order = np.argsort(dens.to_numpy())[::-1]
        k = np.searchsorted(np.cumsum(mass[order]), level) + 1
        sel = np.sort(order[:k])
        if np.any(np.diff(sel) > 1):
            warnings.warn(f"HDI for {param} is not contiguous; reporting its hull.")
        x = self._axis(param)
        return float(x[sel[0]]), float(x[sel[-1]])

    def hdr_thresholds(self, levels=(0.5, 0.8, 0.95)) -> Dict[float, float]:
        """Joint density thresholds enclosing the given posterior mass (for contours)."""
        _probabilities(levels)
        dens = (
            self.mass / np.outer(_cell_widths(self.mu), _cell_widths(self.sd))
        ).ravel()
        mass = self.mass.ravel()
        order = np.argsort(dens)[::-1]
        cm = np.cumsum(mass[order])
        return {
            lv: float(dens[order][min(np.searchsorted(cm, lv), dens.size - 1)])
            for lv in levels
        }

    def edge_mass(self, frac: float = 0.02) -> Dict[str, float]:
        """Posterior mass in the outermost ``frac`` of each grid axis."""
        if not np.isfinite(frac) or not 0 < frac <= 0.5:
            raise ValueError("frac must be greater than zero and at most one half.")
        out = {}
        for param in ("mean", "sd"):
            m = self.marginal(param).to_numpy()
            k = max(1, int(np.ceil(frac * m.size)))
            out[f"{param}_lo"] = float(m[:k].sum())
            out[f"{param}_hi"] = float(m[-k:].sum())
        return out

    # ------------------------------------------------------------- sampling
    def sample(self, n: int, rng=None) -> pd.DataFrame:
        """Draw discrete parameter coordinates from the finite-grid posterior."""
        _sample_size(n)
        rng = np.random.default_rng(rng)
        flat = self.mass.ravel()
        idx = rng.choice(flat.size, size=n, p=flat / flat.sum())
        i, j = np.unravel_index(idx, self.mass.shape)
        mu, sd = self.mu[i].astype(float), self.sd[j].astype(float)
        return pd.DataFrame({"mean": mu, "sd": sd})

    # ----------------------------------------------------------- predictive
    def _cells(self, tail: float = 1e-10):
        flat = self.mass.ravel()
        order = np.argsort(flat)[::-1]
        k = np.searchsorted(np.cumsum(flat[order]), 1 - tail) + 1
        keep = order[:k]
        i, j = np.unravel_index(keep, self.mass.shape)
        w = flat[keep] / flat[keep].sum()
        return self.mu[i], self.sd[j], w

    def predictive_cdf(self, x) -> np.ndarray:
        """Finite-grid predictive CDF of the next period's metric."""
        mu, sd, w = self._cells()
        s = sd
        x = evaluation_points(x)
        out = np.empty_like(x)
        for a in range(0, x.size, 64):
            xs = x[a : a + 64, None]
            out[a : a + 64] = (get_likelihood(self.metric).cdf(xs, mu, s) * w).sum(
                axis=1
            )
        out[np.isneginf(x)] = 0.0
        out[np.isposinf(x)] = 1.0
        return np.clip(out, 0.0, 1.0)

    def predictive_pdf(self, x) -> np.ndarray:
        mu, sd, w = self._cells()
        s = sd
        x = evaluation_points(x)
        out = np.empty_like(x)
        for a in range(0, x.size, 64):
            xs = x[a : a + 64, None]
            out[a : a + 64] = (get_likelihood(self.metric).pdf(xs, mu, s) * w).sum(
                axis=1
            )
        return out

    def _summary_diagnostics(self):
        return {"max edge mass": max(self.edge_mass().values())}


# ------------------------------------------------------------------- estimation


def posterior_grid(
    data,
    prior,
    grid: Union[GridSpec, str] = "auto",
    likelihood_weight: float = 1.0,
    period_years: float = 3.0,
    metric: str = "sharpe",
    warn_tol: float = 1e-3,
) -> GridPosterior:
    """
    Grid posterior of (mean, sd) for a short vector of period metrics.

    Parameters
    ----------
    data : array-like
        Period metrics (e.g. 3-year Sharpe ratios).
    prior : object with ``logpdf(mu, sd)``
        E.g. ``IndependentPrior(NormalPrior(...), SDPrior(...))`` or another vectorised joint prior with logpdf(mean, sd).
    grid : GridSpec or "auto"
        Grid; "auto" calls ``auto_grid``.
    likelihood_weight : float
        Positive power-likelihood weight. Values below one provide a sensitivity
        approximation, not an inferred correction for dependence or selection.
    """
    inputs = ModelInputs.validate(data, prior, metric, period_years, likelihood_weight)
    x = inputs.data
    if not np.isfinite(warn_tol) or not 0 <= warn_tol <= 1:
        raise ValueError("warn_tol must be between zero and one.")
    if isinstance(grid, str) and grid == "auto":
        grid = auto_grid(x, prior, likelihood_weight=likelihood_weight, metric=metric)
    if not isinstance(grid, GridSpec):
        raise TypeError("grid must be a GridSpec or 'auto'.")
    if metric == "ann_vol" and grid.mean_range[0] <= 0:
        raise ValueError("Volatility grids require strictly positive mean bounds.")
    mu, sd = grid.axes()
    log_area = np.log(np.outer(_cell_widths(mu), _cell_widths(sd)))
    ll = get_likelihood(metric).log_likelihood(inputs, mu[:, None], sd[None, :])
    M, SD = np.meshgrid(mu, sd, indexing="ij")
    log_post = prior.logpdf(M, SD) + ll
    lw = log_post + log_area
    log_z = special.logsumexp(lw)
    if not np.isfinite(log_z):
        raise ValueError(
            "The grid has no finite posterior mass; check the prior and grid bounds."
        )
    mass = np.exp(lw - log_z)
    post = GridPosterior(
        mu=mu,
        sd=sd,
        log_post=log_post,
        mass=mass,
        data=x,
        prior=prior,
        likelihood_weight=likelihood_weight,
        log_evidence=(
            float(log_z)
            if getattr(prior, "normalised", False) and likelihood_weight == 1
            else None
        ),
        period_years=period_years,
        metric=metric,
    )
    em = post.edge_mass()
    if max(em.values()) > warn_tol:
        warnings.warn(
            "Posterior mass at grid edges "
            + ", ".join(f"{k}={v:.2%}" for k, v in em.items() if v > warn_tol)
            + "; the grid truncates the posterior.",
            GridTruncationWarning,
        )
    return post


# --------------------------------------------------------------------- auto grid


def _prior_ranges(prior, q: float):
    mean_r = sd_r = None
    if isinstance(prior, IndependentPrior):
        mean_r = tuple(prior.mean.ppf([q, 1 - q]))
        sd_r = tuple(prior.sd.ppf([q, 1 - q]))
    return mean_r, sd_r


def auto_grid(
    data,
    prior,
    n: int = 400,
    k: float = 6.0,
    tol: float = 1e-4,
    max_iter: int = 8,
    likelihood_weight: float = 1.0,
    coarse_n: int = 200,
    metric: str = "sharpe",
) -> GridSpec:
    """
    Data- and prior-driven grid that contains essentially all posterior mass.

    1. Initial box: union of likelihood support (mean +- k se; chi-square bounds for
       the sd) and central prior quantiles.
    2. Coarse pass: trim to the [tol/2, 1 - tol/2] marginal quantiles plus padding.
    3. Fine pass: expand any side whose edge mass exceeds ``tol`` until it does not.
    """
    inputs = ModelInputs.validate(
        data, prior, metric, likelihood_weight=likelihood_weight
    )
    x = inputs.data
    for count in (n, coarse_n):
        if (
            isinstance(count, bool)
            or not isinstance(count, (int, np.integer))
            or count < 2
        ):
            raise ValueError("n and coarse_n must be integers of at least two.")
    if (
        isinstance(max_iter, bool)
        or not isinstance(max_iter, (int, np.integer))
        or max_iter < 1
    ):
        raise ValueError("max_iter must be a positive integer.")
    if not np.isfinite(likelihood_weight) or likelihood_weight <= 0:
        raise ValueError("likelihood_weight must be positive and finite.")
    if not np.isfinite(k) or k <= 0 or not np.isfinite(tol) or not 0 < tol < 0.5:
        raise ValueError(
            "k must be positive and tol must lie between zero and one half."
        )
    nw = max(likelihood_weight * x.size, 1.0)
    xbar = float(x.mean())
    mean_r, sd_r = _prior_ranges(prior, 1e-4)
    s = float(x.std(ddof=1)) if x.size > 1 else (np.mean(sd_r) if sd_r else 1.0)
    # Numerical floors must follow the metric's units, including decimal PnLs.
    reference_scale = (
        prior.sd.mean if isinstance(prior, IndependentPrior) else max(s, abs(xbar))
    )
    reference_scale = reference_scale if reference_scale > 0 else 1.0
    sd_floor = max(reference_scale * 1e-8, np.finfo(float).tiny ** 0.25)
    s = max(s, reference_scale * 1e-6, sd_floor)
    dof = max(nw - 1, 1.0)
    sd_lo = s * np.sqrt(dof / stats.chi2.ppf(1 - 1e-6, dof))
    sd_hi = s * np.sqrt(dof / stats.chi2.ppf(1e-6, dof))
    if sd_r:
        # A proper sd prior bounds the posterior's upper tail; the chi-square bound
        # alone explodes for tiny samples (n = 3: ~1000 x s) and makes the start grid
        # too coarse. The edge-mass loop below still widens the grid if needed.
        sd_lo, sd_hi = min(sd_lo, sd_r[0]), max(sd_r[1], min(sd_hi, 3 * s))
    se_hi = sd_hi / np.sqrt(nw)
    lo, hi = xbar - k * se_hi, xbar + k * se_hi
    if mean_r:
        lo, hi = min(lo, mean_r[0]), max(hi, mean_r[1])
    if metric == "ann_vol":
        lo = max(lo, (mean_r[0] if mean_r else x.min()) * 1e-4)
    sd_lo = max(sd_lo, sd_floor)

    def spacing(a, b):
        return "log" if b / a > 20 else "linear"

    def fit(spec):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", GridTruncationWarning)
            return posterior_grid(
                x, prior, spec, likelihood_weight=likelihood_weight, metric=metric
            )

    coarse = fit(
        GridSpec((lo, hi), (sd_lo, sd_hi), coarse_n, coarse_n, spacing(sd_lo, sd_hi))
    )
    mlo, mhi = coarse.quantile("mean", [tol / 2, 1 - tol / 2])
    slo, shi = coarse.quantile("sd", [tol / 2, 1 - tol / 2])
    pad_m = 0.15 * (mhi - mlo)
    mean_range = [mlo - pad_m, mhi + pad_m]
    if metric == "ann_vol":
        mean_range[0] = max(mean_range[0], lo)
    # sd is positive and right-skewed: pad multiplicatively
    pad_s = (shi / max(slo, sd_floor)) ** 0.15
    sd_range = [max(slo / pad_s, sd_floor), shi * pad_s]

    it, post = 0, None
    for it in range(1, max_iter + 1):
        spec = GridSpec(tuple(mean_range), tuple(sd_range), n, n, spacing(*sd_range))
        post = fit(spec)
        em = post.edge_mass()
        grown = False
        w_m, w_s = mean_range[1] - mean_range[0], sd_range[1] - sd_range[0]
        if em["mean_lo"] > tol:
            mean_range[0] = (
                mean_range[0] / 2 if metric == "ann_vol" else mean_range[0] - 0.5 * w_m
            )
            grown = True
        if em["mean_hi"] > tol:
            mean_range[1] += 0.5 * w_m
            grown = True
        if em["sd_lo"] > tol and sd_range[0] > sd_floor:
            sd_range[0] = max(sd_range[0] / 2, sd_floor)
            grown = True
        if em["sd_hi"] > tol:
            sd_range[1] += 0.5 * w_s
            grown = True
        if not grown:
            break
    spec = GridSpec(tuple(mean_range), tuple(sd_range), n, n, spacing(*sd_range))
    post = fit(spec)
    cells_per_sd = {}
    for p in ("mean", "sd"):
        ax = post._axis(p)
        i_med = int(np.clip(np.searchsorted(ax, post.quantile(p, 0.5)), 0, ax.size - 1))
        cells_per_sd[p] = float(post.std(p) / _cell_widths(ax)[i_med])
    if min(cells_per_sd.values()) < 5:
        warnings.warn(f"Grid is coarse relative to posterior spread: {cells_per_sd}")
    return GridSpec(
        tuple(mean_range),
        tuple(sd_range),
        n,
        n,
        spacing(*sd_range),
        diagnostics=dict(
            iterations=it,
            edge_mass=post.edge_mass(),
            cells_per_posterior_sd=cells_per_sd,
        ),
    )
