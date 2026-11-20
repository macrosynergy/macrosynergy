"""Shared validation and sufficient statistics for period-metric inference."""

from dataclasses import dataclass

import numpy as np

from .priors import IndependentPrior, METRICS, NormalPrior, LogNormalPrior, SDPrior


def observations(data):
    """Copy one finite metric vector; never pool columns or drop observations."""
    if np.iscomplexobj(data):
        raise ValueError("data must contain real-valued period metrics.")
    try:
        x = np.asarray(data, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("data must be a numeric one-dimensional vector.") from exc
    if x.ndim != 1 or x.size == 0 or not np.isfinite(x).all():
        raise ValueError(
            "data must be a nonempty, finite one-dimensional metric vector."
        )
    return x.copy()


def probabilities(q):
    values = np.atleast_1d(np.asarray(q, dtype=float))
    if (
        values.ndim != 1
        or not np.isfinite(values).all()
        or ((values < 0) | (values > 1)).any()
    ):
        raise ValueError("Probabilities must be finite and between zero and one.")
    return values


def evaluation_points(x):
    values = np.atleast_1d(np.asarray(x, dtype=float))
    if values.ndim != 1 or np.isnan(values).any():
        raise ValueError("x must be a scalar or one-dimensional array without NaNs.")
    return values


def parameter_values(value):
    """CDF coordinates preserve input shape and may include infinite endpoints."""
    values = np.asarray(value, dtype=float)
    if np.isnan(values).any():
        raise ValueError("CDF values must not contain NaNs.")
    return values


def sample_size(n):
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 0:
        raise ValueError("n must be a nonnegative integer.")


def supports_quadrature(prior):
    """Exact built-in types only: subclasses may override the assumed density."""
    return (
        type(prior) is IndependentPrior
        and type(prior.mean) is NormalPrior
        and type(prior.sd) is SDPrior
    )


@dataclass(frozen=True)
class ModelInputs:
    """Validated observations and normal-likelihood sufficient statistics."""

    data: np.ndarray
    prior: object
    metric: str
    period_years: float
    likelihood_weight: float
    average: float
    scatter: float

    @property
    def weighted_n(self):
        return self.likelihood_weight * self.data.size

    @classmethod
    def validate(
        cls, data, prior, metric="sharpe", period_years=3.0, likelihood_weight=1.0
    ):
        if metric not in METRICS:
            raise ValueError(
                "Unknown metric {!r}; choose from {}.".format(metric, METRICS)
            )
        if not np.isfinite(likelihood_weight) or likelihood_weight <= 0:
            raise ValueError("likelihood_weight must be positive and finite.")
        if not np.isfinite(period_years) or period_years <= 0:
            raise ValueError("period_years must be positive and finite.")
        if not callable(getattr(prior, "logpdf", None)):
            raise TypeError("prior must expose a vectorised logpdf(mean, sd).")
        x = observations(data)
        if metric == "ann_vol" and (x <= 0).any():
            raise ValueError(
                "ann_vol observations must be strictly positive for the Gamma likelihood."
            )
        if (
            metric == "ann_vol"
            and isinstance(prior, IndependentPrior)
            and isinstance(prior.mean, NormalPrior)
        ):
            raise ValueError(
                "ann_vol requires a positive mean prior, such as LogNormalPrior."
            )
        average = float(x[0] + np.mean(x - x[0]))
        scatter = float(np.square(x - average).sum())
        if not np.isfinite([average, scatter]).all():
            raise ValueError("Data exceed finite numerical precision.")
        built_in = (
            type(prior) is IndependentPrior
            and type(prior.mean) in (NormalPrior, LogNormalPrior)
            and type(prior.sd) is SDPrior
        )
        interior_mean = type(getattr(prior, "mean", None)) is NormalPrior or x[0] > 0
        if built_in and interior_mean and prior.sd.kind == "gamma_sd":
            if (
                np.all(x == x[0])
                and prior.sd.params["shape"] <= likelihood_weight * len(x) - 1
            ):
                raise ValueError(
                    "Improper posterior: identical data require gamma sd shape > weighted n - 1."
                )
        x.setflags(write=False)
        return cls(
            x,
            prior,
            metric,
            float(period_years),
            float(likelihood_weight),
            average,
            scatter,
        )
