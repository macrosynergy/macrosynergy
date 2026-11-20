"""Forecast running averages of successive observed period-performance metrics.

The period model identifies block outcomes, not daily paths. Baseline simulations
use complete fitted-period boundaries; an optional Brownian bridge adds an
explicit within-block approximation. A running average of block Sharpes is not
a concatenated-history Sharpe. Annual-return observations support nominal-period
cumulative arithmetic PnL, without identifying daily dynamics or compounding.
"""

from dataclasses import dataclass
from collections.abc import Mapping
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from .inputs import probabilities, sample_size
from .likelihoods import get_likelihood


def _credible_level(level):
    if not np.isscalar(level) or not np.isfinite(level) or not 0 < level < 1:
        raise ValueError("level must be finite and strictly between zero and one.")
    return float(level)


@dataclass(frozen=True)
class MetricFunnel:
    """Simulated metrics and their running averages at whole-period boundaries.

    Rows of period_metrics and running_mean are complete simulated paths;
    columns correspond to years. Each path retains one parameter pair and,
    when supplied by a construction-mixture sampler, one construction label.
    Parameter draws express uncertainty about the long-term expected metric.
    The default block sampler adds conditionally independent performance noise;
    an explicit path sampler retains its own temporal dependence.

    Quantiles are Monte Carlo estimates. Intervals are pointwise across paths,
    not simultaneous bands for an entire path. No between-boundary evolution,
    strategy-switching policy, or daily cumulative-return process is specified.
    """

    years: np.ndarray
    period_metrics: np.ndarray
    running_mean: np.ndarray
    parameter_means: np.ndarray
    parameter_sds: np.ndarray
    metric: str
    period_years: float
    construction_labels: Optional[np.ndarray] = None
    component_labels: Optional[np.ndarray] = None
    conditional_observations: str = "independent successive periods"
    expected_period_metrics: Optional[np.ndarray] = None
    simulation_info: Optional[dict] = None
    cumulative_values: Optional[np.ndarray] = None

    @property
    def n_paths(self):
        """Number of independently simulated posterior-predictive paths."""
        return self.running_mean.shape[0]

    @property
    def n_periods(self):
        """Number of complete future periods in each path."""
        return self.running_mean.shape[1]

    @property
    def cumulative_return(self):
        """Cumulative arithmetic PnL at whole-period endpoints, for ann_ret only.

        Each sampled annualised return is multiplied by nominal period_years,
        then added without compounding. This block-duration approximation uses
        the fitted metric's units (percent of capital or decimal returns); it
        does not reconstruct sums over an actual daily trading calendar.
        Sharpe and volatility fits cannot identify cumulative-return outcomes.
        """
        if self.metric != "ann_ret":
            raise ValueError(
                "Cumulative return requires an annual-return fit (metric='ann_ret')."
            )
        if self.cumulative_values is not None:
            return self.cumulative_values
        return self.period_years * np.cumsum(self.period_metrics, axis=1)

    @property
    def expected_cumulative_return(self):
        """Posterior expected cumulative PnL at nominal endpoints.

        Average supplied conditional expectations when available. Otherwise
        estimate the expectation from actual simulated paths, never from a
        potentially unrelated stationary or long-term mean parameter.
        """
        if self.metric != "ann_ret":
            raise ValueError("Expected cumulative return requires metric='ann_ret'.")
        conditional = self.expected_path_draws("cumulative_return")
        return np.mean(
            self.cumulative_return if conditional is None else conditional, axis=0
        )

    @property
    def expected_running_mean(self):
        """Expected running average, estimated from conditional or actual paths."""
        conditional = self.expected_path_draws("running_mean")
        return np.mean(
            self.running_mean if conditional is None else conditional, axis=0
        )

    def expected_path_draws(self, quantity="running_mean"):
        """Conditional expected paths, or None when the sampler did not supply them.

        Their cross-path variation describes uncertainty about that trajectory.
        Long-term parameter means alone do not identify these expectations for
        a dynamic path model. They must not substitute for missing trajectories.
        """
        if quantity not in ("running_mean", "cumulative_return"):
            raise ValueError("quantity must be 'running_mean' or 'cumulative_return'.")
        if quantity == "cumulative_return" and self.metric != "ann_ret":
            raise ValueError("Expected cumulative return requires metric='ann_ret'.")
        if self.expected_period_metrics is None:
            return None
        total = np.cumsum(self.expected_period_metrics, axis=1)
        return (
            self.period_years * total
            if quantity == "cumulative_return"
            else total / np.arange(1, self.n_periods + 1)
        )

    @property
    def metadata(self):
        """Interpretation of this result, suitable for figure annotations."""
        result = {
            "target": "running_mean_period_metric",
            "interval_coverage": "pointwise",
            "metric": self.metric,
            "period_years": self.period_years,
            "n_paths": self.n_paths,
            "n_periods": self.n_periods,
            "conditional_observations": self.conditional_observations,
            "parameters": "retained once per path",
            "conditional_expectations": (
                "supplied per period"
                if self.expected_period_metrics is not None
                else "unspecified; expected trajectory estimated from simulated paths"
            ),
            "cumulative_return": (
                "annual-return draws times nominal block duration; arithmetic, not compounded"
                if self.metric == "ann_ret"
                else "not identified by this fitted metric"
            ),
        }
        if self.simulation_info:
            # Retain provenance arrays on simulation_info without dumping them
            # into figure annotations or a human-readable metadata summary.
            result.update(
                {
                    key: value
                    for key, value in self.simulation_info.items()
                    if np.isscalar(value) or value is None or key == "construction"
                }
            )
        return result

    def pointwise_quantile(self, q):
        """Quantiles of the running average; shape (len(q), n_periods)."""
        return np.quantile(self.running_mean, probabilities(q), axis=0)

    def pointwise_interval(self, level=0.9) -> Tuple[np.ndarray, np.ndarray]:
        """Equal-tailed predictive intervals at each complete-period boundary."""
        level = _credible_level(level)
        lower, upper = self.pointwise_quantile([(1 - level) / 2, (1 + level) / 2])
        return lower, upper

    def mean_interval(self, level=0.9) -> Tuple[float, float]:
        """Posterior interval for the retained long-term expectation itself."""
        level = _credible_level(level)
        bounds = np.quantile(self.parameter_means, [(1 - level) / 2, (1 + level) / 2])
        return float(bounds[0]), float(bounds[1])

    def probability_below(self, threshold=0.0):
        """Empirical probability that each running average is below threshold."""
        if not np.isscalar(threshold) or not np.isfinite(threshold):
            raise ValueError("threshold must be a finite scalar.")
        return np.mean(self.running_mean < threshold, axis=0)

    def prob_nonpositive(self):
        """Empirical probability of a nonpositive running average at each time."""
        return np.mean(self.running_mean <= 0, axis=0)


def simulate_metric_funnel(fit, n_paths=10000, n_periods=5, rng=None):
    """Simulate successive block metrics while retaining uncertainty per path.

    Parameters
    ----------
    fit : BayesianPnL, Posterior or compatible construction-mixture fit
        The numerical posterior must expose sample(n, rng), metric and
        period_years. sample returns a DataFrame containing mean and sd columns;
        an optional variant column records the construction chosen for a path.
        A BayesianPnL wrapper is unwrapped to its posterior. The observation
        law is the package's fitted normal Sharpe/return or Gamma volatility law.
        A supplied sample_paths(n_paths, n_periods, rng) takes precedence: its
        complete paths are preserved instead of drawing iid block observations.
        This protocol returns a mapping with values (paths by periods), means
        and sds (one retained parameter pair per path), years (period boundaries),
        optional expected_values (conditional expected metric per path/period),
        and optional variant/component labels (one per path). Future daily
        engines need an adapter that aggregates their paths to these targets.
    n_paths : int
        Positive number of simulated paths, default 10000.
    n_periods : int
        Positive number of complete fitted periods, default five. With a
        three-year fit, boundaries are 3, 6, 9, ... years. Within-period paths
        and shorter-horizon forecasts are not identified by this model.
    rng : int or numpy.random.Generator, optional
        Seed or random generator.

    Returns
    -------
    MetricFunnel
        Block metrics, their arithmetic running averages, retained parameter
        draws and optional construction labels. Pointwise simulation intervals
        include parameter uncertainty and future performance variation. They
        do not acquire simultaneous-path coverage from their nominal level.
    """
    for name, count in (("n_paths", n_paths), ("n_periods", n_periods)):
        sample_size(count)
        if count == 0:
            raise ValueError("{} must be a positive integer.".format(name))

    def has_sampler(item):
        return any(
            callable(getattr(item, name, None)) for name in ("sample", "sample_paths")
        )

    post = fit if has_sampler(fit) else getattr(fit, "posterior", None)
    if post is None or not has_sampler(post):
        raise TypeError("fit must expose a posterior parameter or path sampler.")
    metric = getattr(post, "metric", None)
    likelihood = get_likelihood(metric)
    duration = getattr(post, "period_years", None)
    if (
        duration is None
        or not np.isscalar(duration)
        or not np.isfinite(duration)
        or duration <= 0
    ):
        raise ValueError("The fitted period duration must be positive and finite.")
    rng = np.random.default_rng(rng)
    years = float(duration) * np.arange(1, n_periods + 1)
    labels, components = None, None
    expected_values = None
    observation_assumption = "independent successive periods"
    if callable(getattr(post, "sample_paths", None)):
        paths = post.sample_paths(n_paths, n_periods, rng=rng)
        if not isinstance(paths, Mapping) or not {
            "values",
            "means",
            "sds",
            "years",
        }.issubset(paths):
            raise TypeError(
                "sample_paths must return values, means, sds and years in a mapping."
            )
        observations = _real_array(paths["values"], (n_paths, n_periods), "path values")
        means = _real_array(paths["means"], (n_paths,), "path means")
        deviations = _real_array(paths["sds"], (n_paths,), "path standard deviations")
        path_years = _real_array(paths["years"], (n_periods,), "path years")
        if not np.allclose(path_years, years, rtol=1e-12, atol=0):
            raise ValueError("Path years must match complete fitted-period boundaries.")
        if "expected_values" in paths and paths["expected_values"] is not None:
            expected_values = _real_array(
                paths["expected_values"],
                (n_paths, n_periods),
                "conditional expected path values",
            )
        for name in ("variant", "component"):
            if name in paths:
                values = np.asarray(paths[name])
                if values.shape != (n_paths,):
                    raise ValueError(
                        "Path {} labels must have one value per path.".format(name)
                    )
                if name == "variant":
                    labels = values.copy()
                else:
                    components = values.copy()
        observation_assumption = str(
            paths.get("conditional_observations", "provided by fitted path sampler")
        )
    else:
        draws = post.sample(n_paths, rng=rng)
        if not isinstance(draws, pd.DataFrame) or not {"mean", "sd"}.issubset(
            draws.columns
        ):
            raise TypeError(
                "Posterior samples must be a DataFrame with mean and sd columns."
            )
        if len(draws) != n_paths:
            raise ValueError(
                "The posterior sampler returned an incorrect number of draws."
            )
        means = _real_array(draws["mean"], (n_paths,), "posterior means")
        deviations = _real_array(
            draws["sd"], (n_paths,), "posterior standard deviations"
        )
        labels = draws["variant"].to_numpy(copy=True) if "variant" in draws else None
        components = (
            draws["component"].to_numpy(copy=True) if "component" in draws else None
        )
        observations = None
    if np.any(deviations <= 0):
        raise ValueError(
            "Posterior means must be finite and standard deviations positive and finite."
        )
    if metric == "ann_vol" and np.any(means <= 0):
        raise ValueError("Volatility posterior means must be positive.")
    if observations is None:
        shape = (n_paths, n_periods)
        expected_values = np.broadcast_to(means[:, None], shape).copy()
        observations = likelihood.sample(
            np.broadcast_to(means[:, None], shape),
            np.broadcast_to(deviations[:, None], shape),
            rng,
        )
        observations = _real_array(observations, shape, "simulated path values")
    if metric == "ann_vol" and np.any(observations <= 0):
        raise ValueError("Volatility path observations must be positive.")
    if (
        metric == "ann_vol"
        and expected_values is not None
        and np.any(expected_values <= 0)
    ):
        raise ValueError(
            "Conditional expected volatility observations must be positive."
        )
    construction = None
    for source in (fit, post):
        describe = getattr(source, "construction_metadata", None)
        if callable(describe):
            construction = describe()
            break
    if construction is None:
        data = getattr(post, "data", None)
        if data is not None and np.ndim(data) == 1:
            construction = dict(
                n_declared_implementations=1,
                n_distinct_implementations=1,
                n_positive_weight_implementations=1,
                n_shared_periods=len(data),
            )
    return MetricFunnel(
        years=years,
        period_metrics=observations,
        running_mean=np.cumsum(observations, axis=1) / np.arange(1, n_periods + 1),
        parameter_means=means.copy(),
        parameter_sds=deviations.copy(),
        metric=metric,
        period_years=float(duration),
        construction_labels=labels,
        component_labels=components,
        conditional_observations=observation_assumption,
        expected_period_metrics=expected_values,
        simulation_info={"construction": construction} if construction else None,
    )


def _real_array(values, shape, name):
    """Validate path protocol arrays without silently casting complex values."""
    values = np.asarray(values)
    if np.iscomplexobj(values):
        raise ValueError("{} must contain finite real values.".format(name))
    values = np.asarray(values, dtype=float)
    if values.shape != shape or not np.isfinite(values).all():
        raise ValueError("{} must be finite with shape {}.".format(name, shape))
    return values


def bridge_metric_funnel(
    funnel,
    annual_volatility,
    frequency="monthly",
    rng=None,
    max_points=12000000,
):
    """Approximate within-block arithmetic PnL with Gaussian Brownian bridges.

    This adds assumptions; the fitted period model does not identify daily or
    monthly dynamics. Given a sampled block PnL Y and duration T, the within-
    block cumulative increment at time t is (t/T)*Y + V*(W(t)-(t/T)*W(T)),
    where V is the supplied annual volatility. Bridge innovations in different
    blocks are independent. Original block endpoints and component/construction
    choices are retained, rather than resampled.

    Parameters
    ----------
    funnel : MetricFunnel
        Complete-block simulations for annual returns or Sharpe ratios.
        Sharpe observations are converted to annual returns by the explicitly
        supplied fixed annual volatility. This conversion assumes that same
        volatility reference across all constructions, blocks and paths.
    annual_volatility : float
        Positive annual PnL volatility V in the desired return units. For an
        annual-return funnel it must match the return units already supplied.
        No volatility or within-period dynamics are inferred by this function.
    frequency : {"monthly", "daily"}
        Nominal equally spaced steps: 12 months or 252 trading days per year.
        These are time fractions, not a reconstructed exchange calendar.
    rng : int or numpy.random.Generator, optional
        Seed or generator for the added bridge innovations.
    max_points : int
        Maximum n_paths * n_fine_periods, default 12000000. Several arrays of
        this size are needed. Oversized requests fail before allocation: reduce
        the path count or explicitly choose a different resource limit.

    Returns
    -------
    MetricFunnel
        Annual-return increments on the finer time grid, with exact stored
        cumulative block endpoints. Conditional expected values interpolate
        the source's conditional expectation, not its random sampled endpoint.
        Unknown expected trajectories remain unknown. simulation_info records
        the approximation, source duration and original dispersion draws.

    Notes
    -----
    Bands have the Brownian sqrt(time) limit when the source endpoint variance
    is V**2*T. Other fitted endpoint distributions are preserved and need not
    produce monotonically widening bands. Increment dependence is retained;
    these fine increments must not be refitted as independent historical data.
    """
    if not isinstance(funnel, MetricFunnel):
        raise TypeError("funnel must be a simulated MetricFunnel.")
    if funnel.metric not in ("ann_ret", "sharpe"):
        raise ValueError(
            "A bridge requires annual returns or an explicit fixed-volatility Sharpe conversion."
        )
    if (
        not np.isscalar(annual_volatility)
        or np.iscomplexobj(annual_volatility)
        or not np.isfinite(annual_volatility)
        or annual_volatility <= 0
    ):
        raise ValueError("annual_volatility must be positive and finite.")
    if frequency not in ("monthly", "daily"):
        raise ValueError("frequency must be 'monthly' or 'daily'.")
    sample_size(max_points)
    if max_points == 0:
        raise ValueError("max_points must be positive.")
    steps_per_year = 12 if frequency == "monthly" else 252
    duration = float(funnel.period_years)
    steps_float = duration * steps_per_year
    steps = int(round(steps_float)) if np.isfinite(steps_float) else 0
    if steps < 1 or not np.isclose(steps_float, steps, rtol=0, atol=1e-9):
        raise ValueError(
            "The fitted period must contain an integer number of nominal time steps."
        )
    n_paths, blocks = funnel.n_paths, funnel.n_periods
    n_fine = blocks * steps
    if n_paths * n_fine > max_points:
        raise ValueError(
            "Bridge needs {} path-time points, exceeding max_points={}; reduce n_paths or choose monthly frequency.".format(
                n_paths * n_fine, max_points
            )
        )
    if n_paths == 0 or blocks == 0:
        raise ValueError("The source funnel must contain at least one path and block.")
    _real_array(funnel.years, (blocks,), "source years")
    if not np.allclose(
        funnel.years, duration * np.arange(1, blocks + 1), rtol=1e-12, atol=0
    ):
        raise ValueError("Source years must be consecutive complete-period boundaries.")
    source = _real_array(
        funnel.period_metrics, (n_paths, blocks), "source block metrics"
    )
    conversion = float(annual_volatility) if funnel.metric == "sharpe" else 1.0
    with np.errstate(over="ignore", invalid="ignore"):
        annual_returns = source * conversion
        endpoints = (
            funnel.cumulative_return
            if funnel.metric == "ann_ret"
            else duration * np.cumsum(annual_returns, axis=1)
        )
    endpoints = _real_array(endpoints, (n_paths, blocks), "source cumulative PnL")
    with np.errstate(over="ignore", invalid="ignore"):
        parameter_means = (
            _real_array(funnel.parameter_means, (n_paths,), "source parameter means")
            * conversion
        )
        source_sds = (
            _real_array(
                funnel.parameter_sds, (n_paths,), "source parameter standard deviations"
            )
            * conversion
        )
    if np.any(source_sds <= 0):
        raise ValueError("Source parameter standard deviations must be positive.")
    delta = 1.0 / steps_per_year
    vol = float(annual_volatility)
    with np.errstate(over="ignore", invalid="ignore"):
        fine_sds = np.hypot(
            source_sds, vol * np.sqrt(max(0.0, 1.0 / delta - 1.0 / duration))
        )
    if not np.isfinite(parameter_means).all() or not np.isfinite(fine_sds).all():
        raise ValueError(
            "Bridge parameter scales exceed finite numerical precision; use smaller return units."
        )
    rng = np.random.default_rng(rng)
    cumulative = np.empty((n_paths, n_fine))
    fractions = np.arange(1, steps + 1) / float(steps)
    previous = np.zeros(n_paths)
    for block in range(blocks):
        # Working storage is one block, rather than another full-horizon array.
        bridge = rng.normal(0, np.sqrt(delta), (n_paths, steps))
        np.cumsum(bridge, axis=1, out=bridge)
        bridge -= bridge[:, -1:].copy() * fractions[None, :]
        bridge *= vol
        bridge += (endpoints[:, block] - previous)[:, None] * fractions[None, :]
        bridge += previous[:, None]
        bridge[:, -1] = endpoints[:, block]  # preserve original endpoints bitwise
        cumulative[:, block * steps : (block + 1) * steps] = bridge
        previous = endpoints[:, block]
    increments = np.empty_like(cumulative)
    increments[:, 0] = cumulative[:, 0] / delta
    np.subtract(cumulative[:, 1:], cumulative[:, :-1], out=increments[:, 1:])
    increments[:, 1:] /= delta
    years = np.arange(1, n_fine + 1) / float(steps_per_year)
    # Preserve labels from the original time grid exactly at block boundaries.
    years[steps - 1 :: steps] = funnel.years
    expected = None
    if funnel.expected_period_metrics is not None:
        source_expected = _real_array(
            funnel.expected_period_metrics,
            (n_paths, blocks),
            "source conditional expected values",
        )
        expected = np.repeat(source_expected * conversion, steps, axis=1)
    return MetricFunnel(
        years=years,
        period_metrics=increments,
        running_mean=cumulative / years[None, :],
        parameter_means=parameter_means,
        parameter_sds=fine_sds,
        metric="ann_ret",
        period_years=delta,
        construction_labels=(
            None
            if funnel.construction_labels is None
            else funnel.construction_labels.copy()
        ),
        component_labels=(
            None if funnel.component_labels is None else funnel.component_labels.copy()
        ),
        conditional_observations="endpoint-conditioned Gaussian Brownian bridges; increments are dependent",
        expected_period_metrics=expected,
        simulation_info={
            **(funnel.simulation_info or {}),
            "approximation": "Gaussian Brownian bridge",
            "frequency": frequency,
            "nominal_steps_per_year": steps_per_year,
            "annual_volatility": vol,
            "source_period_years": duration,
            "source_metric": funnel.metric,
            "source_parameter_sds": source_sds.copy(),
            "source_metric_conversion": conversion,
        },
        cumulative_values=cumulative,
    )
