"""Fixed mixtures of compatible, separately fitted performance distributions."""

from collections.abc import Mapping

import numpy as np
import pandas as pd

from .funnel import simulate_metric_funnel
from .inputs import sample_size
from .likelihoods import get_likelihood


class BayesianPnLMixture:
    """Combine separate fits using fixed, caller-declared component weights.

    Parameters
    ----------
    components : mapping
        Nonempty mapping from unique string names to BayesianPnL fits,
        posteriors, or compatible evaluators. All members must predict the same
        metric and period duration. The caller must also align units, forecast
        origin, return/cost basis and target interpretation; matching labels
        alone cannot establish economic compatibility.
    weights : mapping or one-dimensional array-like, optional
        Finite nonnegative weights in component order, normalized internally.
        A mapping must contain exactly the component names. Defaults to equal
        weights. Zero-weight members never contribute evaluations or draws.
    weight_role : {"research", "predictive_pool"}
        Meaning of the fixed weights. Research weights describe a hypothetical
        construction choice; predictive-pool weights describe a declared
        combination of forecasting models. Neither is automatically a Bayesian
        posterior model probability or a correction for strategy selection.

    Notes
    -----
    This mixture does not refit members or learn from their combined histories.
    It represents a randomly chosen member, not the uncertainty of a weighted
    family mean and not a portfolio formed by averaging PnLs. Components need
    not be independent: their probabilities are summed, not multiplied.

    Parameter sampling selects one member per draw. Path sampling selects one
    member per whole future path and delegates to its path sampler when one is
    supplied, retaining that member's temporal dependence. Current period fits
    otherwise use their declared conditional-iid block observation model. This
    wrapper does not create daily dynamics from a period-metric posterior.
    """

    def __init__(self, components, weights=None, *, weight_role="research"):
        if not isinstance(components, Mapping) or not components:
            raise ValueError("components must be a nonempty mapping of named fits.")
        if any(not isinstance(name, str) or not name for name in components):
            raise ValueError("Component names must be nonempty strings.")
        if weight_role not in ("research", "predictive_pool"):
            raise ValueError("weight_role must be 'research' or 'predictive_pool'.")
        self._names = tuple(components)
        self._members = tuple(self._resolve(fit) for fit in components.values())
        self._weights = self._validate_weights(weights)
        self.weight_role = weight_role
        self.metric = self._members[0].metric
        self.period_years = float(self._members[0].period_years)
        get_likelihood(self.metric)
        if not np.isfinite(self.period_years) or self.period_years <= 0:
            raise ValueError("Component period duration must be positive and finite.")
        for member in self._members:
            if member.metric != self.metric or member.period_years != self.period_years:
                raise ValueError(
                    "All components must share the same metric and period duration."
                )

    @staticmethod
    def _resolve(fit):
        has_sampler = any(
            callable(getattr(fit, name, None)) for name in ("sample", "sample_paths")
        )
        member = fit if has_sampler else getattr(fit, "posterior", None)
        required = (
            "expectation",
            "prob_mean_below",
            "predictive_pdf",
            "predictive_cdf",
        )
        if member is None or any(
            not callable(getattr(member, name, None)) for name in required
        ):
            raise TypeError(
                "Each component must expose posterior expectations and predictive probabilities."
            )
        if not any(
            callable(getattr(member, name, None)) for name in ("sample", "sample_paths")
        ):
            raise TypeError(
                "Each component must provide parameter or complete-path sampling."
            )
        if not hasattr(member, "metric") or not hasattr(member, "period_years"):
            raise TypeError(
                "Each component must declare its metric and period duration."
            )
        return member

    def _validate_weights(self, weights):
        if weights is None:
            values = np.ones(len(self._names))
        elif isinstance(weights, Mapping):
            if set(weights) != set(self._names):
                raise ValueError("Weight names must exactly match component names.")
            values = np.asarray([weights[name] for name in self._names])
        else:
            values = np.asarray(weights)
        if np.iscomplexobj(values):
            raise ValueError("Weights must be finite, real and nonnegative.")
        values = np.asarray(values, dtype=float)
        if (
            values.shape != (len(self._names),)
            or not np.isfinite(values).all()
            or np.any(values < 0)
        ):
            raise ValueError(
                "Weights must be a finite nonnegative vector with one entry per component."
            )
        maximum = values.max()
        if maximum <= 0:
            raise ValueError("At least one component must have positive weight.")
        # Scaling first prevents overflow when finite weights have a huge sum.
        values = values / maximum
        values = values / values.sum()
        values.setflags(write=False)
        return values

    @property
    def weights(self):
        """Normalized fixed weights, returned as a copy indexed by component."""
        return pd.Series(self._weights.copy(), index=self._names, name="weight")

    def construction_metadata(self):
        """Declared mixture components; this is not an independent sample count."""
        return dict(
            n_components=len(self._names),
            n_positive_weight_components=int((self._weights > 0).sum()),
            component_role=self.weight_role,
            component_weights=self.weights.to_dict(),
        )

    def expectation(self, param="mean"):
        """Mixture expectation of the selected member's mean or dispersion.

        expectation('sd') is E[sigma_J], not the standard deviation of either
        the pooled predictive distribution or the family-mean expectation.
        """
        if param not in ("mean", "sd"):
            raise ValueError("param must be 'mean' or 'sd'.")
        return float(
            sum(
                weight * member.expectation(param)
                for weight, member in zip(self._weights, self._members)
                if weight > 0
            )
        )

    def prob_mean_below(self, threshold=0.0):
        """Probability of a selected member's long-term expectation below threshold."""
        return float(
            sum(
                weight * member.prob_mean_below(threshold)
                for weight, member in zip(self._weights, self._members)
                if weight > 0
            )
        )

    def _predictive(self, values, method):
        result = None
        for weight, member in zip(self._weights, self._members):
            if weight > 0:
                contribution = weight * np.asarray(
                    getattr(member, method)(values), dtype=float
                )
                result = contribution if result is None else result + contribution
        return result

    def predictive_pdf(self, x):
        """Weighted density of the same-duration metric for one selected member."""
        return self._predictive(x, "predictive_pdf")

    def predictive_cdf(self, x):
        """Weighted CDF of the same-duration metric for one selected member."""
        return self._predictive(x, "predictive_cdf")

    def _choices(self, n, rng):
        sample_size(n)
        return rng.choice(len(self._names), size=n, p=self._weights)

    def sample(self, n, rng=None):
        """Choose a component once, then sample that member's parameter pair.

        Returns mean/sd plus component identity. A member's variant label is
        preserved; absent variant labels default to the component name.
        """
        rng = np.random.default_rng(rng)
        choices = self._choices(n, rng)
        pieces = []
        for index, member in enumerate(self._members):
            positions = np.flatnonzero(choices == index)
            if not len(positions):
                continue
            if not callable(getattr(member, "sample", None)):
                raise TypeError(
                    "Selected member provides complete paths only; use sample_paths."
                )
            draws = member.sample(len(positions), rng=rng)
            if (
                not isinstance(draws, pd.DataFrame)
                or draws.columns.has_duplicates
                or not {"mean", "sd"}.issubset(draws.columns)
            ):
                raise TypeError(
                    "Member samples must contain mean and sd DataFrame columns."
                )
            if len(draws) != len(positions):
                raise ValueError(
                    "Member sampler returned an incorrect number of draws."
                )
            draws = draws.copy()
            parameters = draws[["mean", "sd"]].to_numpy()
            if np.iscomplexobj(parameters):
                raise ValueError("Sampled parameters must be finite real values.")
            parameters = np.asarray(parameters, dtype=float)
            if not np.isfinite(parameters).all():
                raise ValueError("Sampled parameters must be finite real values.")
            if np.any(parameters[:, 1] <= 0) or (
                self.metric == "ann_vol" and np.any(parameters[:, 0] <= 0)
            ):
                raise ValueError(
                    "Sampled dispersion and volatility means must be positive."
                )
            draws["mean"], draws["sd"] = parameters[:, 0], parameters[:, 1]
            draws["component"] = self._names[index]
            if "variant" not in draws:
                draws["variant"] = self._names[index]
            draws.index = positions
            pieces.append(draws)
        if not pieces:
            return pd.DataFrame(
                {
                    "mean": np.array([], dtype=float),
                    "sd": np.array([], dtype=float),
                    "component": np.array([], dtype=object),
                    "variant": np.array([], dtype=object),
                }
            )
        return pd.concat(pieces).sort_index()

    def sample_paths(self, n_paths, n_periods, rng=None):
        """Select one member per whole path and preserve its forecasting law.

        Each member's complete path sampler takes precedence over conditional
        iid sampling. Daily engines require their own adapter producing the
        same block metrics and endpoint years; they are never replaced by iid
        normals based solely on a mean/sd pair.
        """
        sample_size(n_periods)
        if n_periods == 0:
            raise ValueError("n_periods must be positive.")
        rng = np.random.default_rng(rng)
        choices = self._choices(n_paths, rng)
        values = np.empty((n_paths, n_periods))
        expected_values = np.empty_like(values)
        has_expectations = True
        means, deviations = np.empty(n_paths), np.empty(n_paths)
        variants, components = np.empty(n_paths, dtype=object), np.empty(
            n_paths, dtype=object
        )
        for index, member in enumerate(self._members):
            positions = np.flatnonzero(choices == index)
            if not len(positions):
                continue
            path = simulate_metric_funnel(member, len(positions), n_periods, rng=rng)
            values[positions] = path.period_metrics
            if path.expected_period_metrics is None:
                has_expectations = False
            else:
                expected_values[positions] = path.expected_period_metrics
            means[positions], deviations[positions] = (
                path.parameter_means,
                path.parameter_sds,
            )
            variants[positions] = (
                path.construction_labels
                if path.construction_labels is not None
                else self._names[index]
            )
            components[positions] = self._names[index]
        result = {
            "values": values,
            "means": means,
            "sds": deviations,
            "variant": variants,
            "component": components,
            "years": self.period_years * np.arange(1, n_periods + 1),
            "conditional_observations": "complete paths supplied by the selected component",
        }
        if has_expectations:
            result["expected_values"] = expected_values
        return result

    def predictive_sample(self, n, rng=None):
        """Simulate the next block from a component selected once per outcome."""
        return self.sample_paths(n, 1, rng=rng)["values"][:, 0]
