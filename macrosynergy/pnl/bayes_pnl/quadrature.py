"""Adaptive one-dimensional integration after eliminating the normal mean.

The rectangular grid is constructed lazily for figures only. Parameter and
predictive probabilities integrate over the complete positive sd support.
"""

import math
import warnings

import numpy as np
import pandas as pd
from scipy import integrate, interpolate, optimize, special

from .grid import GridSpec, posterior_grid
from .inputs import (
    ModelInputs,
    probabilities as _probabilities,
    sample_size as _sample_size,
    evaluation_points,
    parameter_values,
    supports_quadrature,
)
from .posterior import Posterior, bracketed_quantile


class QuadraturePosterior(Posterior):
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
        if metric == "ann_vol":
            raise ValueError(
                "Volatility uses a Gamma likelihood; select method='grid' or 'auto'."
            )
        if not supports_quadrature(prior):
            raise TypeError(
                "Quadrature requires IndependentPrior(NormalPrior, SDPrior)."
            )
        inputs = ModelInputs.validate(
            data, prior, metric, period_years, likelihood_weight
        )
        if not isinstance(grid, GridSpec) and not (
            isinstance(grid, str) and grid == "auto"
        ):
            raise TypeError("grid must be a GridSpec or 'auto'.")
        self.data = inputs.data
        self.prior, self.metric = prior, metric
        self.period_years, self.likelihood_weight = period_years, likelihood_weight
        self._grid_spec, self._plot_grid = grid, None
        self._location, self._scale = float(prior.mean.mean), float(prior.sd.mean)
        self._average = (inputs.average - self._location) / self._scale
        self._n = inputs.weighted_n
        self._scatter = inputs.likelihood_weight * inputs.scatter / self._scale**2
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
            alpha = p["alpha"]
            log_alpha = math.log(alpha)
            # Center at the log-SD prior mode. The usual inverse-gamma formula
            # subtracts terms of order alpha*log(alpha), losing the shape of a
            # concentrated density. This equivalent form keeps its deviation
            # from the peak separate from the normalized peak height.
            z = -2 * (t - 0.5 * (log_beta - log_alpha))
            if z > 709:
                return -np.inf
            if abs(z) < 1e-3:
                remainder = (
                    -z
                    * z
                    * (0.5 + z * (1 / 6 + z * (1 / 24 + z * (1 / 120 + z / 720))))
                )
            else:
                remainder = z - math.expm1(z)
            if alpha >= 32:
                inverse = 1 / alpha
                correction = inverse * (
                    1 / 12
                    + inverse**2
                    * (-1 / 360 + inverse**2 * (1 / 1260 - inverse**2 / 1680))
                )
                peak = (
                    math.log(2.0)
                    + 0.5 * (log_alpha - math.log(2 * math.pi))
                    - correction
                )
            else:
                peak = (
                    math.log(2.0) + alpha * log_alpha - special.gammaln(alpha) - alpha
                )
            return peak - t + alpha * remainder
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

    def _integral(
        self, function, lower=-np.inf, upper=np.inf, normalized=True, log_function=False
    ):
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
            if not np.isfinite(log_weight):
                return 0.0
            if log_function:
                log_weight += function(t)
                return self._exp(log_weight) if log_weight > -745 else 0.0
            if log_weight < -745:
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
                result = self._scale * self._integral(lambda t: t, log_function=True)[0]
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

                moment = self._expect(variance)
            else:
                log_center = math.log(center)

                def log_variance(t):
                    distance = abs(t - log_center)
                    if distance == 0:
                        return -np.inf
                    return 2 * (max(t, log_center) + math.log(-math.expm1(-distance)))

                moment = self._integral(log_variance, log_function=True)[0]
            self._cache["std", param] = self._scale * math.sqrt(max(0.0, moment))
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
        values = parameter_values(value)
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
        root = bracketed_quantile(function, center, width)
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
        values = evaluation_points(x)
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

    def to_grid(self):
        """Construct/cache a finite plotting approximation, explicitly on request."""
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

    def density(self, param, values):
        self._axis_name(param)
        values = evaluation_points(values)
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

    def _summary_diagnostics(self):
        return {
            "integration relative error": self.diagnostics[
                "normalization_relative_error"
            ]
        }

    def sample(self, n, rng=None):
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
