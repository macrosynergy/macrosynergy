"""Adaptive one-dimensional integration after eliminating the normal mean.

The rectangular grid is constructed lazily for figures only. Parameter and
predictive probabilities integrate over the complete positive sd support.
"""

import math
import warnings

import numpy as np
import pandas as pd
from scipy import integrate, interpolate, optimize, special

from .grid import (
    GridPosterior,
    GridSpec,
    _observations,
    _probabilities,
    _sample_size,
    posterior_grid,
)
from .priors import IndependentPrior, METRICS, NormalPrior, SDPrior


def supports_quadrature(prior):
    """Whether the prior permits the implemented analytic mean elimination."""
    # An overriding subclass can define a different density even when it keeps
    # the same attributes. Never silently eliminate that user's mean prior.
    return (
        type(prior) is IndependentPrior
        and type(prior.mean) is NormalPrior
        and type(prior.sd) is SDPrior
    )


class QuadraturePosterior(GridPosterior):
    """Adaptive posterior with a lazy, explicitly separate visualization grid.

    Ordinary inference uses adaptive QUADPACK integration, with normal mean
    integrated analytically. Large samples use a separately refined numerical
    inverse CDF; its checked error is recorded in ``diagnostics``.
    """

    method = "quadrature"

    def __repr__(self):
        return "QuadraturePosterior(n_periods={}, metric={!r}, period_years={!r}, method='quadrature')".format(
            len(self.data), self.metric, self.period_years
        )

    def __init__(
        self,
        data,
        prior,
        grid="auto",
        likelihood_weight=1.0,
        period_years=3.0,
        metric="sharpe",
    ):
        if not supports_quadrature(prior):
            raise TypeError(
                "Quadrature requires IndependentPrior(NormalPrior, SDPrior)."
            )
        if metric not in METRICS:
            raise ValueError("Unknown metric {!r}.".format(metric))
        if not np.isfinite(likelihood_weight) or likelihood_weight <= 0:
            raise ValueError("likelihood_weight must be positive and finite.")
        if not np.isfinite(period_years) or period_years <= 0:
            raise ValueError("period_years must be positive and finite.")
        if not isinstance(grid, GridSpec) and not (
            isinstance(grid, str) and grid == "auto"
        ):
            raise TypeError("grid must be a GridSpec or 'auto'.")
        self.data = _observations(data)
        self.prior, self.metric = prior, metric
        self.period_years, self.likelihood_weight = period_years, likelihood_weight
        self._grid_spec, self._plot_grid = grid, None
        self._location, self._scale = float(prior.mean.mean), float(prior.sd.mean)
        x = (self.data - self._location) / self._scale
        identical = bool(np.all(self.data == self.data[0]))
        self._average = float(x[0] if identical else x[0] + (x - x[0]).mean())
        self._n = float(likelihood_weight * len(x))
        self._scatter = (
            0.0
            if identical
            else float(likelihood_weight * np.square(x - self._average).sum())
        )
        self._s0 = float(prior.mean.sd / self._scale)
        if (
            not np.isfinite([self._average, self._scatter, self._s0]).all()
            or self._s0 <= 0
        ):
            raise ValueError(
                "Data/prior scale ratios exceed finite numerical precision."
            )
        self._log_s0 = math.log(self._s0)
        self._log_kappa = math.log(self._n) + 2 * self._log_s0
        self._prior_params = prior.sd.params
        if prior.sd.kind == "gamma_sd" and self._scatter == 0:
            if self._prior_params["shape"] <= self._n - 1:
                raise ValueError(
                    "Improper posterior: identical data require gamma sd shape > weighted n - 1."
                )
        self._cache, self._inverse = {}, None
        self._mode_t, self._width, self._height, self._peaks = (
            self._integration_geometry()
        )
        self._normalizer, error = self._integral(lambda t: 1.0, normalized=False)
        if not np.isfinite(self._normalizer) or self._normalizer <= 0:
            raise ValueError(
                "Adaptive integration found no finite posterior normalizer."
            )
        self.log_evidence = (
            (
                self._height
                + math.log(self._width)
                + math.log(self._normalizer)
                - self._n * math.log(self._scale)
            )
            if likelihood_weight == 1
            else None
        )
        self.diagnostics = {
            "normalization_relative_error": error / self._normalizer,
            "integration": "adaptive log-sd; analytic conditional mean",
        }

    @staticmethod
    def _exp(value):
        return math.exp(value) if value < 709 else np.inf

    def _log_prior(self, t):
        p, kind = self._prior_params, self.prior.sd.kind
        if kind == "invgamma_var":
            log_beta = math.log(p["beta"]) - 2 * math.log(self._scale)
            penalty = self._exp(log_beta - 2 * t)
            return (
                math.log(2.0)
                + p["alpha"] * log_beta
                - special.gammaln(p["alpha"])
                - (2 * p["alpha"] + 1) * t
                - penalty
            )
        if kind == "gamma_sd":
            log_rate = math.log(p["rate"]) + math.log(self._scale)
            return (
                p["shape"] * log_rate
                - special.gammaln(p["shape"])
                + (p["shape"] - 1) * t
                - self._exp(log_rate + t)
            )
        center = p["mu"] - math.log(self._scale)
        return (
            -t
            - math.log(p["sigma"] * math.sqrt(2 * math.pi))
            - 0.5 * ((t - center) / p["sigma"]) ** 2
        )

    def _log_kernel(self, t):
        if not np.isfinite(t):
            return -np.inf
        log_denom = float(np.logaddexp(2 * t, self._log_kappa))
        scatter = (
            0.0 if self._scatter == 0 else self._exp(math.log(self._scatter) - 2 * t)
        )
        distance = (
            0.0
            if self._average == 0
            else self._exp(
                math.log(self._n) + 2 * math.log(abs(self._average)) - log_denom
            )
        )
        return (
            self._log_prior(t)
            + (1 - self._n) * t
            - 0.5 * self._n * math.log(2 * math.pi)
            - 0.5 * scatter
            + t
            - 0.5 * log_denom
            - 0.5 * distance
        )

    def _conditional(self, t, predictive=False):
        ratio = float(special.expit(self._log_kappa - 2 * t))
        mean = self._average * ratio
        log_var = 2 * self._log_s0 + 2 * t - float(np.logaddexp(2 * t, self._log_kappa))
        if predictive:
            log_var = float(np.logaddexp(log_var, 2 * t))
        return mean, 0.5 * log_var

    def _integration_geometry(self):
        # Prior quantiles and data scales locate candidate peaks without using a
        # plotting grid. Include a local bracket for nearly degenerate priors.
        quantiles = self.prior.sd.ppf([1e-7, 0.01, 0.5, 0.99, 1 - 1e-7]) / self._scale
        points = [math.log(v) for v in quantiles if np.isfinite(v) and v > 0]
        points.extend([0.0, self._log_s0, math.log(max(abs(self._average), self._s0))])
        if self._scatter > 0:
            points.append(0.5 * math.log(self._scatter / self._n))
        elif self.prior.sd.kind == "gamma_sd":
            rate = self._prior_params["rate"] * self._scale
            points.append(math.log((self._prior_params["shape"] - self._n + 1) / rate))
        elif self.prior.sd.kind == "lognormal_sd":
            p = self._prior_params
            points.append(
                p["mu"] - math.log(self._scale) + (1 - self._n) * p["sigma"] ** 2
            )
        elif self.prior.sd.kind == "invgamma_var":
            p = self._prior_params
            points.append(
                0.5
                * (
                    math.log(2 * p["beta"])
                    - 2 * math.log(self._scale)
                    - math.log(2 * p["alpha"] + self._n - 1)
                )
            )
        lower, upper = min(points) - 4.0, max(points) + 4.0
        scan = sorted(set(np.linspace(lower, upper, 81).tolist() + points))
        self._scan = scan
        candidates = []
        values = [self._log_kernel(t) for t in scan]
        for i in range(1, len(scan) - 1):
            if values[i] >= values[i - 1] and values[i] >= values[i + 1]:
                result = optimize.minimize_scalar(
                    lambda t: -self._log_kernel(t),
                    bounds=(scan[i - 1], scan[i + 1]),
                    method="bounded",
                    options={"xatol": 1e-12},
                )
                candidates.append(float(result.x))
        if not candidates:
            raise ValueError("Could not locate a finite log-sd posterior peak.")
        mode = max(candidates, key=self._log_kernel)
        height = self._log_kernel(mode)
        step = 1e-5
        if self.prior.sd.kind == "lognormal_sd":
            step = min(step, self._prior_params["sigma"] / 20)
        curvature = (
            -(
                self._log_kernel(mode + step)
                - 2 * height
                + self._log_kernel(mode - step)
            )
            / step**2
        )
        width = (
            1 / math.sqrt(curvature)
            if np.isfinite(curvature) and curvature > 0
            else 1.0
        )
        return mode, width, height, candidates

    def _integral(self, function, lower=-np.inf, upper=np.inf, normalized=True):
        lo, hi = (lower - self._mode_t) / self._width, (
            upper - self._mode_t
        ) / self._width
        knots = sorted(
            set(
                [lo, hi]
                + [
                    (t - self._mode_t) / self._width
                    for t in self._peaks
                    if lower < t < upper
                ]
            )
        )
        total, error = 0.0, 0.0

        def integrand(u):
            t = self._mode_t + self._width * u
            log_weight = self._log_kernel(t) - self._height
            if log_weight < -745 or not np.isfinite(log_weight):
                return 0.0
            return math.exp(log_weight) * function(t)

        with warnings.catch_warnings():
            warnings.simplefilter("error", integrate.IntegrationWarning)
            try:
                for a, b in zip(knots[:-1], knots[1:]):
                    value, err = integrate.quad(
                        integrand, a, b, epsabs=2e-11, epsrel=2e-10, limit=250
                    )
                    total += value
                    error += err
            except integrate.IntegrationWarning as exc:
                raise ValueError(
                    "Adaptive integration did not meet its error tolerance."
                ) from exc
        return (
            (total / self._normalizer, error / self._normalizer)
            if normalized
            else (total, error)
        )

    def _expect(self, function):
        return self._integral(function)[0]

    def expectation(self, param):
        self._axis_name(param)
        if ("expectation", param) not in self._cache:
            if param == "mean":
                result = self._location + self._scale * self._expect(
                    lambda t: self._conditional(t)[0]
                )
            else:
                result = self._scale * self._expect(lambda t: self._exp(t))
            self._cache["expectation", param] = result
        return self._cache["expectation", param]

    @staticmethod
    def _axis_name(param):
        if param not in ("mean", "sd"):
            raise ValueError("param must be 'mean' or 'sd'.")

    def std(self, param):
        self._axis_name(param)
        if ("std", param) not in self._cache:
            center = (
                (self.expectation(param) - self._location) / self._scale
                if param == "mean"
                else self.expectation(param) / self._scale
            )
            if param == "mean":

                def variance(t):
                    m, log_sd = self._conditional(t)
                    return self._exp(2 * log_sd) + (m - center) ** 2

            else:

                def variance(t):
                    return (self._exp(t) - center) ** 2

            self._cache["std", param] = self._scale * math.sqrt(
                max(0.0, self._expect(variance))
            )
        return self._cache["std", param]

    def _normal_value(self, value, t, predictive=False, density=False):
        m, log_sd = self._conditional(t, predictive)
        x = (value - self._location) / self._scale
        # This form retains the small difference when x equals the sample mean.
        difference = (
            (x - self._average) + self._average * special.expit(2 * t - self._log_kappa)
            if 2 * t < self._log_kappa
            else x - m
        )
        if difference == 0:
            z = 0.0
        else:
            log_z = math.log(abs(difference)) - log_sd
            if density and log_z > 350:
                return 0.0
            z = math.copysign(self._exp(log_z), difference)
        if density:
            if not np.isfinite(z):
                return 0.0
            return self._exp(
                -0.5 * z**2
                - 0.5 * math.log(2 * math.pi)
                - log_sd
                - math.log(self._scale)
            )
        return float(special.ndtr(z))

    def _sd_cdf_log(self, t):
        if t <= self._mode_t:
            return self._integral(lambda _: 1.0, upper=t)[0]
        return 1 - self._integral(lambda _: 1.0, lower=t)[0]

    def cdf(self, param, value):
        self._axis_name(param)
        values = np.asarray(value, dtype=float)
        if np.isnan(values).any():
            raise ValueError("CDF values must not contain NaNs.")
        results = []
        for v in values.ravel():
            if np.isinf(v):
                result = 0.0 if v < 0 else 1.0
            elif param == "sd":
                result = 0.0 if v <= 0 else self._sd_cdf_log(math.log(v / self._scale))
            else:
                result = self._expect(lambda t: self._normal_value(float(v), t))
            results.append(np.clip(result, 0.0, 1.0))
        output = np.asarray(results).reshape(values.shape)
        return float(output) if output.ndim == 0 else output

    def _quantile_one(self, param, q):
        if q in (0, 1):
            return (0.0 if param == "sd" else -np.inf) if q == 0 else np.inf
        if param == "sd":
            center, width = self._mode_t, self._width
            function = lambda t: self._sd_cdf_log(t) - q
        else:
            center = (self.expectation("mean") - self._location) / self._scale
            width = max(self.std("mean") / self._scale, np.finfo(float).eps)
            function = lambda x: self.cdf("mean", self._location + self._scale * x) - q
        lo, hi = center - 8 * width, center + 8 * width
        for _ in range(50):
            if function(lo) <= 0 and function(hi) >= 0:
                break
            width *= 2
            lo, hi = center - 8 * width, center + 8 * width
        else:
            raise ValueError(
                "Could not bracket posterior quantile within finite precision."
            )
        root = optimize.brentq(
            function, lo, hi, xtol=max(1e-12 * width, np.finfo(float).tiny), rtol=1e-12
        )
        return (
            self._scale * self._exp(root)
            if param == "sd"
            else self._location + self._scale * root
        )

    def quantile(self, param, q):
        self._axis_name(param)
        probabilities = _probabilities(q)
        values = np.array([self._quantile_one(param, p) for p in probabilities])
        return values[0] if np.ndim(q) == 0 else values

    def predictive_cdf(self, x):
        return self._predictive(x, False)

    def predictive_pdf(self, x):
        return self._predictive(x, True)

    def _predictive(self, x, density):
        values = np.atleast_1d(np.asarray(x, dtype=float))
        if values.ndim != 1 or np.isnan(values).any():
            raise ValueError(
                "x must be a scalar or one-dimensional array without NaNs."
            )
        result = []
        for value in values:
            if np.isinf(value):
                result.append(0.0 if density or value < 0 else 1.0)
            elif density and self._singular_density(value):
                result.append(np.inf)
            else:
                result.append(
                    self._expect(lambda t: self._normal_value(value, t, True, density))
                )
        return np.asarray(result)

    def _singular_density(self, value):
        return (
            self._scatter == 0
            and self.prior.sd.kind == "gamma_sd"
            and self._prior_params["shape"] <= self._n
            and value == self.data[0]
        )

    def predictive_quantile(self, q):
        result = []
        for p in _probabilities(q):
            if p in (0, 1):
                result.append(-np.inf if p == 0 else np.inf)
                continue
            center = (self.expectation("mean") - self._location) / self._scale
            width = max(
                self.expectation("sd") / self._scale, self.std("mean") / self._scale
            )
            function = (
                lambda z: self.predictive_cdf(self._location + self._scale * z)[0] - p
            )
            lo, hi = center - 8 * width, center + 8 * width
            for _ in range(50):
                if function(lo) <= 0 and function(hi) >= 0:
                    break
                width *= 2
                lo, hi = center - 8 * width, center + 8 * width
            else:
                raise ValueError("Could not bracket predictive quantile.")
            root = optimize.brentq(function, lo, hi, xtol=1e-12 * width, rtol=1e-12)
            result.append(self._location + self._scale * root)
        return np.asarray(result)

    def interval(self, param, level=0.9, kind="eti"):
        """Equal-tailed or shortest contiguous credible interval.

        The latter is a highest-density interval for a unimodal marginal, but
        does not represent a disjoint highest-density set for a multimodal law.
        """
        self._axis_name(param)
        if not np.isfinite(level) or not 0 < level < 1 or kind not in ("eti", "hdi"):
            raise ValueError("Require 0 < level < 1 and kind 'eti' or 'hdi'.")
        key = ("interval", param, float(level), kind)
        if key not in self._cache:
            if kind == "eti":
                bounds = self.quantile(param, [(1 - level) / 2, (1 + level) / 2])
            else:
                # Shortest contiguous credible interval; no plotting-grid mass.
                def width(p):
                    return self._quantile_one(param, p + level) - self._quantile_one(
                        param, p
                    )

                minimum = optimize.minimize_scalar(
                    width,
                    bounds=(1e-10, 1 - level - 1e-10),
                    method="bounded",
                    options={"xatol": 1e-5},
                )
                p = float(minimum.x)
                bounds = self.quantile(param, [p, p + level])
            self._cache[key] = (float(bounds[0]), float(bounds[1]))
        return self._cache[key]

    def _ensure_grid(self):
        if self._plot_grid is None:
            spec = self._grid_spec
            if isinstance(spec, str):
                mean_bounds = self.quantile("mean", [1e-5, 1 - 1e-5])
                sd_bounds = self.quantile("sd", [1e-5, 1 - 1e-5])
                mean_pad = 0.05 * np.diff(mean_bounds)[0]
                spec = GridSpec(
                    (mean_bounds[0] - mean_pad, mean_bounds[1] + mean_pad),
                    (sd_bounds[0] * 0.8, sd_bounds[1] * 1.2),
                    220,
                    240,
                    "log",
                )
            self._plot_grid = posterior_grid(
                self.data,
                self.prior,
                spec,
                likelihood_weight=self.likelihood_weight,
                period_years=self.period_years,
                metric=self.metric,
            )
        return self._plot_grid

    @property
    def mu(self):
        return self._ensure_grid().mu

    @property
    def sd(self):
        return self._ensure_grid().sd

    @property
    def mass(self):
        return self._ensure_grid().mass

    @property
    def log_post(self):
        return self._ensure_grid().log_post

    def marginal(self, param):
        """Visualization-only marginal masses on the lazy plotting grid."""
        return self._ensure_grid().marginal(param)

    def marginal_density(self, param):
        """Continuous marginal density evaluated on visualization coordinates."""
        coordinates = self.mu if param == "mean" else self.sd
        return pd.Series(
            self.density(param, coordinates),
            index=pd.Index(coordinates, name=param),
            name="density",
        )

    def density(self, param, values):
        self._axis_name(param)
        values = np.atleast_1d(np.asarray(values, dtype=float))
        result = []
        for value in values:
            if param == "sd":
                if value <= 0 or not np.isfinite(value):
                    result.append(0.0)
                else:
                    t = math.log(value / self._scale)
                    result.append(
                        self._exp(
                            self._log_kernel(t)
                            - self._height
                            - math.log(self._width)
                            - math.log(self._normalizer)
                            - math.log(value)
                        )
                    )
            elif self._singular_density(value):
                result.append(np.inf)
            else:
                result.append(
                    self._expect(lambda t: self._normal_value(value, t, density=True))
                )
        return np.asarray(result)

    def hdr_thresholds(self, levels=(0.5, 0.8, 0.95)):
        """Approximate visualization contours, separate from adaptive inference."""
        return self._ensure_grid().hdr_thresholds(levels)

    def edge_mass(self, frac=0.02):
        """Plot-grid diagnostic only; not a quadrature truncation measure."""
        return self._ensure_grid().edge_mass(frac)

    def mode(self):
        if (
            self._scatter == 0
            and self.prior.sd.kind == "gamma_sd"
            and self._prior_params["shape"] < self._n + 1
        ):
            return (
                float(self.data[0]),
                0.0,
            )  # unbounded boundary density, not an interior mode

        def log_joint_peak(t):
            return self._log_kernel(t) - t - self._conditional(t)[1]

        scan = list(self._scan) + list(self._peaks)
        if self._scatter == 0 and self.prior.sd.kind == "gamma_sd":
            exponent = self._prior_params["shape"] - self._n - 1
            if exponent > 0:
                scan.append(
                    math.log(exponent / (self._prior_params["rate"] * self._scale))
                )
        scan = sorted(set(scan))
        values = [log_joint_peak(t) for t in scan]
        candidates = []
        for i in range(1, len(scan) - 1):
            if values[i] >= values[i - 1] and values[i] >= values[i + 1]:
                fit = optimize.minimize_scalar(
                    lambda t: -log_joint_peak(t),
                    bounds=(scan[i - 1], scan[i + 1]),
                    method="bounded",
                    options={"xatol": 1e-11},
                )
                candidates.append(float(fit.x))
        boundary = -np.inf
        if (
            self._scatter == 0
            and self.prior.sd.kind == "gamma_sd"
            and self._prior_params["shape"] == self._n + 1
        ):
            p = self._prior_params
            boundary = (
                p["shape"] * math.log(p["rate"] * self._scale)
                - special.gammaln(p["shape"])
                - 0.5 * self._n * math.log(2 * math.pi)
                - self._log_s0
                - 0.5 * (self._average / self._s0) ** 2
            )
        if np.isfinite(boundary) and (
            not candidates or boundary >= max(map(log_joint_peak, candidates))
        ):
            return float(self.data[0]), 0.0  # finite boundary supremum
        if not candidates:
            raise ValueError("No finite interior joint posterior mode was resolved.")
        t = max(candidates, key=log_joint_peak)
        m = self._conditional(t)[0]
        return self._location + self._scale * m, self._scale * math.exp(t)

    def summary(
        self,
        level=0.9,
        thresholds=(0.0,),
        predictive_q=(0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95),
    ):
        # Do not call GridPosterior.summary, whose final edge diagnostic would
        # construct a plotting grid and make inference depend on its bounds.
        rows = {}
        for p in ("mean", "sd"):
            rows[p + ": posterior mean"] = self.expectation(p)
            rows[p + ": posterior median"] = float(self.quantile(p, 0.5))
            rows[p + ": posterior sd"] = self.std(p)
            for kind in ("eti", "hdi"):
                lo, hi = self.interval(p, level, kind)
                rows["{}: {:.0%} {}".format(p, level, kind.upper())] = (
                    "[{:.3f}, {:.3f}]".format(lo, hi)
                )
        m, s = self.mode()
        rows["joint mode (mean, sd)"] = (
            "({:.3f}, {:.3f})".format(m, s)
            if s > 0
            else "boundary limit ({:.3f}, 0)".format(m)
        )
        for t in thresholds:
            rows["Pr(mean < {:g})".format(t)] = self.prob_mean_below(t)
            rows["Pr(next period < {:g})".format(t)] = float(self.predictive_cdf(t)[0])
        for q, value in zip(predictive_q, self.predictive_quantile(predictive_q)):
            rows["predictive q{:02d}".format(int(round(q * 100)))] = float(value)
        rows["integration relative error"] = self.diagnostics[
            "normalization_relative_error"
        ]
        return pd.Series(rows, name="value").to_frame()

    def sample(self, n, rng=None, jitter=True):
        """Conditional-normal draws using a refined numerical inverse sd CDF."""
        _sample_size(n)
        rng = np.random.default_rng(rng)
        if n == 0:
            return pd.DataFrame({"mean": [], "sd": []})
        if self._inverse is None:
            self._build_inverse()
        uniforms = rng.random(n)
        t = self._inverse(uniforms)
        outside = ~np.isfinite(t)
        for i in np.flatnonzero(outside):
            t[i] = math.log(self._quantile_one("sd", uniforms[i]) / self._scale)
        means, deviations = np.array([self._conditional(v) for v in t]).T
        return pd.DataFrame(
            {
                "mean": self._location
                + self._scale * rng.normal(means, np.exp(deviations)),
                "sd": self._scale * np.exp(t),
            }
        )

    def _build_inverse(self):
        lower = math.log(self._quantile_one("sd", 1e-12) / self._scale)
        upper = math.log(self._quantile_one("sd", 1 - 1e-12) / self._scale)
        checks = np.array([0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999])
        # This numerical inverse is independent of the visualization grid.
        for count in (1025, 2049, 4097, 8193, 16385):
            ts = np.linspace(lower, upper, count)
            ordinates = np.array(
                [self._exp(self._log_kernel(t) - self._height) for t in ts]
            )
            cdf = integrate.cumulative_trapezoid(ordinates, ts, initial=0)
            cdf = 1e-12 + (1 - 2e-12) * cdf / cdf[-1]
            unique = np.r_[True, np.diff(cdf) > 0]
            inverse = interpolate.PchipInterpolator(
                cdf[unique], ts[unique], extrapolate=False
            )
            error = max(abs(self._sd_cdf_log(float(inverse(q))) - q) for q in checks)
            if error < 1e-7:
                self._inverse = inverse
                self.diagnostics["sampling_checked_cdf_error"] = error
                return
        raise ValueError("Numerical inverse sd CDF did not reach sampling tolerance.")

    def predictive_sample(self, n, rng=None):
        _sample_size(n)
        rng = np.random.default_rng(rng)
        draws = self.sample(n, rng, jitter=False)
        return rng.normal(draws["mean"].to_numpy(), draws["sd"].to_numpy())
