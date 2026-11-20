"""Correlated period metrics with persistent, partially pooled construction effects.

This model uses one multivariate observation per market period. Research-choice
weights select a plausible implementation; they never multiply the likelihood
or represent portfolio positions. Correlation matrices are explicit assumptions,
not estimates known without uncertainty.
"""

import warnings

import numpy as np
import pandas as pd
from scipy import linalg, optimize, special, stats

from .inputs import evaluation_points, probabilities, sample_size
from .periods import period_metrics
from .priors import IndependentPrior, NormalPrior, SDPrior, default_prior


def _matrix(value, size, name):
    if np.iscomplexobj(value):
        raise ValueError("{} must be real-valued.".format(name))
    result = np.asarray(value, dtype=float)
    if (
        result.shape != (size, size)
        or not np.isfinite(result).all()
        or not np.allclose(result, result.T, rtol=1e-10, atol=1e-12)
        or not np.allclose(np.diag(result), 1.0, rtol=1e-10, atol=1e-12)
    ):
        raise ValueError(
            "{} must be a finite symmetric correlation matrix with diagonal one.".format(
                name
            )
        )
    return result.copy()


def _widths(axis):
    if len(axis) == 1:
        return np.ones(1)
    spacing = np.diff(axis)
    return np.r_[spacing[0] / 2, (spacing[:-1] + spacing[1:]) / 2, spacing[-1] / 2]


def _axis(values, positive, name):
    result = np.asarray(values, dtype=float)
    if result.ndim != 1 or not result.size or not np.isfinite(result).all():
        raise ValueError(
            "{} must be a nonempty finite one-dimensional axis.".format(name)
        )
    if (result <= 0).any() if positive else (result < 0).any():
        raise ValueError(
            "{} has values outside its positive/nonnegative support.".format(name)
        )
    if (np.diff(result) <= 0).any():
        raise ValueError("{} must be strictly increasing.".format(name))
    return result.copy()


class BayesianPnLVariants:
    """Joint normal hierarchy for aligned Sharpe or annual-return observations.

    Each row of data is a shared three-year period and each column a strategy.
    Conditional on persistent expected values beta, rows are independent
    N(beta, sigma**2 R). The prior is beta = mu*1 + a, mu ~ N(m0,s0**2),
    a ~ N(0,tau**2 C). Mu and a are integrated analytically at each sigma/tau
    grid cell. A half-normal(tau_scale) prior regularises construction spread.

    Parameters
    ----------
    data : pandas.DataFrame or two-dimensional array
        Aligned finite period metrics. No rows/columns are silently dropped.
    correlation : array-like
        Explicit period-noise correlation R. Required for multiple variants.
        It must be positive definite after explicitly declared aliases collapse.
        Daily PnL correlation is not automatically Sharpe-error correlation.
    weights : dict or one-dimensional array, optional
        Nonnegative research-choice probabilities, normalised to sum one.
        Defaults to equal weight across distinct declared implementations.
    prior : IndependentPrior, optional
        Built-in normal prior on shared expectation and SDPrior on period sigma.
        Defaults follow default_prior(metric, vol_target).
    tau_scale : float, optional
        Half-normal scale for persistent construction differences, in metric units.
        Default 0.5 for Sharpe and 0.5*vol_target for annual return.
    construction_correlation : array-like, optional
        Prior correlation C of remaining construction effects; default identity
        is an exchangeability assumption, not a claim of residual independence.
    aliases : dict, optional
        Explicit alias-name -> canonical-name declarations for the *same* rule.
        Their data and supplied correlation rows must agree. Weights aggregate.
        Equal observed columns alone never trigger merging.
    sigma_grid, tau_grid : one-dimensional arrays, optional
        Numerical hyperparameter axes. A singleton axis fixes that parameter;
        tau_grid=[0] disables construction effects for a reference calculation.
    n_sigma, n_tau : int
        Default axis resolutions. Grid convergence must be checked for decisions.

    Notes
    -----
    This is a conditional, iid-period model, not a selection correction or a
    dynamic regime model. Fixed R and C uncertainty is not integrated. The period
    dispersion combines metric sampling noise and genuine performance changes.
    "family" means the weighted mean of strategy expectations, not portfolio
    Sharpe. "draw" selects one implementation and retains it for a whole path.
    """

    period_years = 3.0
    method = "hierarchical_grid"

    def __init__(
        self,
        data,
        correlation=None,
        weights=None,
        prior=None,
        tau_scale=None,
        metric="sharpe",
        vol_target=10.0,
        sigma_grid=None,
        tau_grid=None,
        n_sigma=100,
        n_tau=81,
        aliases=None,
        variant_names=None,
        construction_correlation=None,
    ):
        if metric not in ("sharpe", "ann_ret"):
            raise ValueError("The variants hierarchy supports 'sharpe' and 'ann_ret'.")
        if np.iscomplexobj(data):
            raise ValueError("Variant observations must be real.")
        x = np.asarray(data, dtype=float)
        if x.ndim != 2 or not x.size or not np.isfinite(x).all():
            raise ValueError("Supply a nonempty finite period-by-variant matrix.")
        names = (
            list(data.columns)
            if isinstance(data, pd.DataFrame)
            else (
                list(variant_names)
                if variant_names is not None
                else ["V{}".format(i) for i in range(x.shape[1])]
            )
        )
        if (
            len(names) != x.shape[1]
            or len(set(names)) != len(names)
            or any(not isinstance(v, str) for v in names)
        ):
            raise ValueError("Variant names must be unique strings, one per column.")
        if any(v in ("draw", "family", "idea") for v in names):
            raise ValueError(
                "Variant names cannot be reserved targets: draw, family, idea."
            )
        k = len(names)
        if correlation is None and k > 1:
            raise ValueError(
                "Supply an explicit period correlation matrix for multiple variants."
            )
        R = _matrix(np.eye(k) if correlation is None else correlation, k, "correlation")
        C = (
            None
            if construction_correlation is None
            else _matrix(construction_correlation, k, "construction_correlation")
        )
        declarations = {} if aliases is None else dict(aliases)
        resolved = {}
        for name in names:
            target, visited = name, set()
            while target in declarations:
                if target in visited:
                    raise ValueError("Alias declarations contain a cycle.")
                visited.add(target)
                target = declarations[target]
            if target not in names:
                raise ValueError("Alias targets must be existing variant names.")
            resolved[name] = target
        if set(declarations) - set(names):
            raise ValueError("Alias names must exist in the observations.")
        canonical = [name for name in names if resolved[name] == name]
        keep = [names.index(name) for name in canonical]
        for name, target in resolved.items():
            if name != target:
                a, b = names.index(name), names.index(target)
                if not np.array_equal(x[:, a], x[:, b]) or not np.allclose(R[a], R[b]):
                    raise ValueError(
                        "Declared aliases require equal observations and period-correlation rows."
                    )
                if C is not None and not np.allclose(C[a], C[b]):
                    raise ValueError(
                        "Declared aliases require equal construction-correlation rows."
                    )
        if weights is None:
            q = np.full(len(canonical), 1.0 / len(canonical))
        else:
            if isinstance(weights, dict):
                if set(weights) != set(names):
                    raise ValueError(
                        "Weights must name every supplied variant exactly once."
                    )
                w = np.array([weights[name] for name in names], dtype=float)
            else:
                if np.iscomplexobj(weights):
                    raise ValueError("Weights must be real-valued.")
                w = np.asarray(weights, dtype=float)
            with np.errstate(over="ignore"):
                total_weight = float(w.sum())
            if (
                w.shape != (k,)
                or not np.isfinite(w).all()
                or (w < 0).any()
                or not np.isfinite(total_weight)
                or total_weight <= 0
            ):
                raise ValueError(
                    "Weights must be finite, nonnegative and have positive total mass."
                )
            q = np.array(
                [
                    sum(
                        w[j] for j, name in enumerate(names) if resolved[name] == target
                    )
                    for target in canonical
                ]
            )
            q /= q.sum()
        R = R[np.ix_(keep, keep)]
        C = np.eye(len(keep)) if C is None else C[np.ix_(keep, keep)]
        try:
            np.linalg.cholesky(R)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                "R must be positive definite after explicit aliases collapse."
            ) from exc
        eigenvalues, eigenvectors = np.linalg.eigh(C)
        if eigenvalues.min() < -1e-10:
            raise ValueError("C must be positive semidefinite.")
        if eigenvalues.min() < 0:
            C = (eigenvectors * np.maximum(eigenvalues, 0.0)) @ eigenvectors.T
        selected = default_prior(metric, vol_target) if prior is None else prior
        if (
            type(selected) is not IndependentPrior
            or type(selected.mean) is not NormalPrior
            or type(selected.sd) is not SDPrior
        ):
            raise TypeError(
                "The hierarchy requires IndependentPrior(NormalPrior, SDPrior)."
            )
        scale = (
            (0.5 if metric == "sharpe" else 0.5 * float(vol_target))
            if tau_scale is None
            else float(tau_scale)
        )
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("tau_scale must be positive and finite.")
        for count in (n_sigma, n_tau):
            if (
                isinstance(count, bool)
                or not isinstance(count, (int, np.integer))
                or count < 2
            ):
                raise ValueError("Grid sizes must be integers of at least two.")
        self.metric, self.prior, self.tau_scale = metric, selected, scale
        self.names, self.aliases = tuple(canonical), resolved
        self.input_names = tuple(names)
        self._weights = pd.Series(q, index=self.names, name="research_weight")
        self.correlation, self.construction_correlation = R, C
        self.data = x[:, keep].copy()
        self.data.setflags(write=False)
        self.n_periods, self.n_variants = self.data.shape
        self.periods = None
        self._cache = {}
        identical_periods = bool(
            np.array_equal(self.data, np.broadcast_to(self.data[0], self.data.shape))
        )
        average = (
            self.data[0].copy()
            if identical_periods
            else self.data[0] + (self.data - self.data[0]).mean(axis=0)
        )
        residuals = self.data - average
        self._scatter = residuals.T @ residuals
        self._average = average
        if (
            selected.sd.kind == "gamma_sd"
            and (sigma_grid is None or np.size(sigma_grid) > 1)
            and identical_periods
        ):
            raise ValueError(
                "The identical-period gamma-sd zero-scale propriety case is unsupported; use an inverse-gamma variance prior or fix sigma."
            )
        typical = float(
            np.sqrt(
                np.trace(linalg.solve(R, self._scatter, assume_a="pos"))
                / max(1, (self.n_periods - 1) * self.n_variants)
            )
        )
        prior_limits = selected.sd.ppf([1e-5, 1 - 1e-5])
        low = min(prior_limits[0], typical / 8) if typical > 0 else prior_limits[0]
        high = max(prior_limits[1], 8 * typical)
        upper_tau = max(
            float(stats.halfnorm.ppf(1 - 1e-5, scale=scale)), float(np.std(average)) * 4
        )
        sigma_axis = (
            np.geomspace(low, high, n_sigma)
            if sigma_grid is None
            else _axis(sigma_grid, True, "sigma_grid")
        )
        tau_axis = (
            np.linspace(0, upper_tau, n_tau)
            if tau_grid is None
            else _axis(tau_grid, False, "tau_grid")
        )
        self._fit(sigma_axis, tau_axis)
        edges = self.diagnostics["edge_mass"]
        if max(edges.values()) > 0.01:
            warnings.warn(
                "Hierarchical grid has material edge mass {}; widen or refine its axes.".format(
                    edges
                ),
                UserWarning,
                stacklevel=2,
            )

    @property
    def weights(self):
        """Normalised research-choice weights, returned as an independent copy."""
        return self._weights.copy()

    def _fit(self, sigma, tau):
        self.sigma, self.tau = sigma, tau
        sig, effect = np.meshgrid(sigma, tau, indexing="ij")
        sig, effect = sig.ravel(), effect.ravel()
        T, K = self.n_periods, self.n_variants
        m0, s0 = self.prior.mean.mean, self.prior.mean.sd
        G = (
            s0**2 * np.ones((len(sig), K, K))
            + effect[:, None, None] ** 2 * self.construction_correlation
        )
        H = G + sig[:, None, None] ** 2 * self.correlation / T
        chol = np.linalg.cholesky(H)
        identity = np.broadcast_to(np.eye(K), H.shape)
        inverse = np.linalg.solve(
            np.swapaxes(chol, 1, 2), np.linalg.solve(chol, identity)
        )
        displacement = self._average - m0
        solved = np.einsum("nij,j->ni", inverse, displacement)
        conditional_mean = m0 + np.einsum("nij,nj->ni", G, solved)
        gain = G @ inverse
        remaining = identity - gain
        observation_cov = sig[:, None, None] ** 2 * self.correlation / T
        # Joseph form avoids cancellation between large, almost equal matrices.
        conditional_cov = remaining @ G @ np.swapaxes(
            remaining, 1, 2
        ) + gain @ observation_cov @ np.swapaxes(gain, 1, 2)
        conditional_cov = (conditional_cov + np.swapaxes(conditional_cov, 1, 2)) / 2
        eigenvalues, eigenvectors = np.linalg.eigh(conditional_cov)
        tolerance = 1e-10 * np.maximum(
            np.max(np.abs(eigenvalues), axis=1), np.finfo(float).tiny
        )
        if (eigenvalues[:, 0] < -tolerance).any():
            raise ValueError(
                "Conditional covariance lost numerical positive semidefiniteness."
            )
        if (eigenvalues < 0).any():
            conditional_cov = (
                eigenvectors * np.maximum(eigenvalues, 0.0)[:, None, :]
            ) @ np.swapaxes(eigenvectors, 1, 2)
        logdet_R = np.linalg.slogdet(self.correlation)[1]
        scatter = np.trace(
            linalg.solve(self.correlation, self._scatter, assume_a="pos")
        )
        log_likelihood = (
            -0.5 * T * K * np.log(2 * np.pi)
            - 0.5 * (T - 1) * (2 * K * np.log(sig) + logdet_R)
            - 0.5 * K * np.log(T)
            - 0.5 * scatter / sig**2
            - np.log(np.diagonal(chol, axis1=1, axis2=2)).sum(axis=1)
            - 0.5 * np.einsum("i,ni->n", displacement, solved)
        )
        log_mass = log_likelihood.copy()
        if len(sigma) > 1:
            log_mass += self.prior.sd.logpdf(sig)
        if len(tau) > 1:
            log_mass += stats.halfnorm.logpdf(effect, scale=self.tau_scale)
        areas = np.outer(_widths(sigma), _widths(tau)).ravel()
        log_mass += np.log(areas)
        normalizer = special.logsumexp(log_mass)
        if not np.isfinite(normalizer):
            raise ValueError("No finite mass on the hierarchical scale grid.")
        self.mass = np.exp(log_mass - normalizer).reshape(len(sigma), len(tau))
        self.log_evidence = float(normalizer)
        self.conditional_means = conditional_mean
        self.conditional_covariances = conditional_cov
        self._sigmas, self._taus = sig, effect
        shared_noise = (
            effect[:, None, None] ** 2 * self.construction_correlation + observation_cov
        )
        noise_inverse = np.linalg.inv(shared_noise)
        self._shared_variances = 1 / (1 / s0**2 + noise_inverse.sum(axis=(1, 2)))
        self._shared_means = m0 + self._shared_variances * np.einsum(
            "nij,j->n", noise_inverse, displacement
        )
        marginal_sigma, marginal_tau = self.mass.sum(axis=1), self.mass.sum(axis=0)
        self.diagnostics = {
            "edge_mass": {
                "sigma_low": float(marginal_sigma[0]) if len(sigma) > 1 else 0.0,
                "sigma_high": float(marginal_sigma[-1]) if len(sigma) > 1 else 0.0,
                "tau_high": float(marginal_tau[-1]) if len(tau) > 1 else 0.0,
            },
            "n_periods": T,
            "n_distinct_variants": K,
            "correlation_condition_number": float(np.linalg.cond(self.correlation)),
            "covariance_uncertainty": "R and C are fixed assumptions; not integrated",
            "selection_adjustment": False,
        }

    @classmethod
    def from_pnl(
        cls,
        pnl,
        correlation=None,
        weights=None,
        prior=None,
        tau_scale=None,
        metric="sharpe",
        annualization_factor=252,
        end_year=None,
        **kwargs,
    ):
        """Fit daily PnLs using only complete three-year blocks shared by all columns."""
        if isinstance(pnl, pd.Series):
            pnl = pnl.to_frame(name=pnl.name if pnl.name is not None else "V0")
        if not isinstance(pnl, pd.DataFrame):
            raise TypeError("pnl must be a dated daily Series or DataFrame.")
        if metric not in ("sharpe", "ann_ret"):
            raise ValueError("The variants hierarchy supports 'sharpe' and 'ann_ret'.")
        periods = period_metrics(
            pnl, annualization_factor=annualization_factor, end_year=end_year
        )
        if periods.empty:
            raise ValueError("No complete shared three-year periods are available.")
        matrix = periods.pivot(
            index="period_end", columns="variant", values=metric
        ).reindex(columns=pnl.columns)
        complete = matrix.notna().all(axis=1) & np.isfinite(matrix.to_numpy()).all(
            axis=1
        )
        if not complete.all():
            warnings.warn(
                "Periods without finite metrics for every variant are excluded jointly.",
                UserWarning,
                stacklevel=2,
            )
        matrix = matrix.loc[complete]
        if matrix.empty:
            raise ValueError("No complete shared three-year periods are available.")
        result = cls(
            matrix,
            correlation=correlation,
            weights=weights,
            prior=prior,
            tau_scale=tau_scale,
            metric=metric,
            **kwargs,
        )
        result.periods = periods.loc[periods.period_end.isin(matrix.index)].copy()
        return result

    def _components(self, target="draw", predictive=False):
        masses = self.mass.ravel()
        if target == "draw":
            means = self.conditional_means.ravel()
            variances = np.diagonal(
                self.conditional_covariances, axis1=1, axis2=2
            ).ravel()
            weights = (masses[:, None] * self.weights.to_numpy()).ravel()
            noise = np.repeat(self._sigmas**2, self.n_variants)
        elif target == "family":
            q = self.weights.to_numpy()
            means = self.conditional_means @ q
            variances = np.einsum("i,nij,j->n", q, self.conditional_covariances, q)
            weights = masses
            noise = self._sigmas**2 * (q @ self.correlation @ q)
        elif target == "idea":
            if predictive:
                raise ValueError(
                    "The shared hypermean has no uniquely defined realised-period outcome; use draw or a named variant."
                )
            means, variances, weights = (
                self._shared_means,
                self._shared_variances,
                masses,
            )
            noise = np.zeros_like(means)
        else:
            if target not in self.aliases:
                raise ValueError(
                    "Unknown target {!r}; use draw, family, idea or a variant name.".format(
                        target
                    )
                )
            index = self.names.index(self.aliases[target])
            means = self.conditional_means[:, index]
            variances = self.conditional_covariances[:, index, index]
            weights, noise = masses, self._sigmas**2
        return (
            means,
            np.maximum(0.0, variances) + (noise if predictive else 0.0),
            weights,
        )

    def _period_dispersion(self, target):
        if target == "idea":
            raise ValueError(
                "The shared hypermean has no unique realised-period dispersion."
            )
        if target == "family":
            q = self.weights.to_numpy()
            return self._sigmas * np.sqrt(q @ self.correlation @ q)
        if target != "draw" and target not in self.aliases:
            raise ValueError("Unknown construction target {!r}.".format(target))
        return self._sigmas

    def expectation(self, param="mean", *, target="draw"):
        """Posterior expectation of long-term metric or conditional period sd."""
        if param == "sd":
            return float(self.mass.ravel() @ self._period_dispersion(target))
        if param != "mean":
            raise ValueError(
                "param must be 'mean' or 'sd'; select constructions with target=."
            )
        means, _, weights = self._components(target)
        return float(weights @ means)

    def std(self, param="mean", *, target="draw"):
        """Parameter uncertainty; mean includes random construction differences."""
        if param == "sd":
            deviations = self._period_dispersion(target)
            weights = self.mass.ravel()
            return float(np.sqrt(weights @ (deviations - weights @ deviations) ** 2))
        if param != "mean":
            raise ValueError(
                "param must be 'mean' or 'sd'; select constructions with target=."
            )
        means, variances, weights = self._components(target)
        center = weights @ means
        return float(np.sqrt(weights @ (variances + (means - center) ** 2)))

    def construction_metadata(self):
        """Declared research universe, rather than counts seen in random draws."""
        return dict(
            n_declared_implementations=len(self.input_names),
            n_distinct_implementations=self.n_variants,
            n_positive_weight_implementations=int((self.weights > 0).sum()),
            n_shared_periods=self.n_periods,
            research_weights=self.weights.to_dict(),
            aliases={
                name: target for name, target in self.aliases.items() if name != target
            },
        )

    def uncertainty_decomposition(self):
        """Variance of a randomly selected implementation's expected performance.

        Total = weighted within-implementation posterior variance + variance
        between the implementations' posterior means. Uncertain construction
        effects also contribute to the first term; the second term alone is not
        all construction-related uncertainty. This does not imply a wider
        distribution than any particular single-strategy fit.
        """
        means = np.array([self.expectation(target=name) for name in self.names])
        variances = np.array([self.std(target=name) ** 2 for name in self.names])
        weights = self.weights.to_numpy()
        center = float(weights @ means)
        within = float(weights @ variances)
        between = float(weights @ (means - center) ** 2)
        total = within + between
        return dict(
            expected_mean=center,
            within_variance=within,
            between_variance=between,
            total_variance=total,
            within_sd=float(np.sqrt(within)),
            total_sd=float(np.sqrt(total)),
            construction_share=between / total if total > 0 else 0.0,
            n_implementations=self.n_variants,
            n_weighted=int((weights > 0).sum()),
            n_periods=self.n_periods,
        )

    def _distribution(self, x, target, predictive, density):
        values = evaluation_points(x)
        means, variances, weights = self._components(target, predictive)
        deviations = np.sqrt(variances)
        positive = deviations > 0
        result = []
        for value in values:
            if density:
                components = np.zeros_like(means)
                components[positive] = stats.norm.pdf(
                    value, means[positive], deviations[positive]
                )
                if np.any((~positive) & (means == value) & (weights > 0)):
                    result.append(np.inf)
                    continue
            else:
                components = np.asarray(value >= means, dtype=float)
                components[positive] = special.ndtr(
                    (value - means[positive]) / deviations[positive]
                )
            result.append(weights @ components)
        return np.asarray(result)

    def cdf(self, x, target="draw"):
        return self._distribution(x, target, False, False)

    def density(self, x, target="draw"):
        return self._distribution(x, target, False, True)

    def prob_mean_below(self, threshold=0.0, target="draw"):
        return float(self.cdf(threshold, target)[0])

    def predictive_cdf(self, x, target="draw"):
        return self._distribution(x, target, True, False)

    def predictive_pdf(self, x, target="draw"):
        return self._distribution(x, target, True, True)

    def _quantiles(self, q, target, predictive):
        means, variances, _ = self._components(target, predictive)
        center, width = float(np.mean(means)), max(
            float(np.sqrt(variances.max())), float(np.ptp(means)), 1e-12
        )
        function = self.predictive_cdf if predictive else self.cdf
        result = []
        for probability in probabilities(q):
            if probability in (0, 1):
                result.append(-np.inf if probability == 0 else np.inf)
                continue
            radius = width * 8
            for _ in range(30):
                low, high = center - radius, center + radius
                if function(low, target)[0] <= probability <= function(high, target)[0]:
                    break
                radius *= 2
            else:
                raise ValueError("Unable to bracket mixture quantile.")
            result.append(
                optimize.brentq(
                    lambda z: function(z, target)[0] - probability,
                    low,
                    high,
                    xtol=max(width * 1e-11, np.finfo(float).tiny),
                    rtol=1e-12,
                )
            )
        return np.asarray(result)

    def quantile(self, q, target="draw"):
        return self._quantiles(q, target, False)

    def interval(self, level=0.9, target="draw"):
        if not np.isfinite(level) or not 0 < level < 1:
            raise ValueError("level must lie strictly between zero and one.")
        return tuple(self.quantile([(1 - level) / 2, (1 + level) / 2], target))

    def predictive_quantile(self, q, target="draw"):
        return self._quantiles(q, target, True)

    def summary(self, level=0.9):
        """Long-term and next-period results; family is not a portfolio Sharpe."""
        rows = []
        for target in ("draw", "family") + self.names:
            low, high = self.interval(level, target)
            p_low, p_high = self.predictive_quantile(
                [(1 - level) / 2, (1 + level) / 2], target
            )
            rows.append(
                dict(
                    target=target,
                    expected_metric=self.expectation(target=target),
                    posterior_sd=self.std(target=target),
                    mean_lo=low,
                    mean_hi=high,
                    prob_mean_positive=1 - self.prob_mean_below(0, target),
                    predictive_lo=p_low,
                    predictive_hi=p_high,
                    prob_next_period_negative=float(self.predictive_cdf(0, target)[0]),
                )
            )
        return pd.DataFrame(rows).set_index("target")

    def sample(self, n, rng=None, target="draw"):
        """Select one hypercell and persistent construction/mean per sampled world."""
        sample_size(n)
        generator = np.random.default_rng(rng)
        cells = generator.choice(self.mass.size, size=n, p=self.mass.ravel())
        if target == "draw":
            choices = generator.choice(
                self.n_variants, size=n, p=self.weights.to_numpy()
            )
            means = self.conditional_means[cells, choices]
            variances = self.conditional_covariances[cells, choices, choices]
            labels = np.asarray(self.names, dtype=object)[choices]
            noise = self._sigmas[cells]
        else:
            means_all, vars_all, _ = self._components(target)
            means, variances = means_all[cells], vars_all[cells]
            labels = np.full(n, target, dtype=object)
            if target == "idea":
                noise = np.full(n, np.nan)
            else:
                q_noise = (
                    (
                        self.weights.to_numpy()
                        @ self.correlation
                        @ self.weights.to_numpy()
                    )
                    if target == "family"
                    else 1.0
                )
                noise = self._sigmas[cells] * np.sqrt(q_noise)
        return pd.DataFrame(
            dict(
                mean=generator.normal(means, np.sqrt(np.maximum(0.0, variances))),
                sd=noise,
                variant=labels,
                sigma=self._sigmas[cells],
                tau=self._taus[cells],
            )
        )

    sample_parameters = sample

    def predictive_sample(self, n, rng=None, target="draw"):
        generator = np.random.default_rng(rng)
        draws = self.sample(n, generator, target)
        if target == "idea":
            raise ValueError(
                "Select an implementation target for realised predictions."
            )
        return generator.normal(draws["mean"].to_numpy(), draws["sd"].to_numpy())

    def sample_paths(self, n_paths, n_periods, rng=None, target="draw"):
        """Simulate repeated whole three-year periods, not within-period PnL paths.

        The chosen implementation, parameters and hypercell remain fixed per
        path. Fresh independent period shocks are drawn conditional on them.
        Annual-return paths remain annualised block returns, not cumulative PnL.
        """
        sample_size(n_periods)
        if n_periods == 0 or target == "idea":
            raise ValueError(
                "Require positive n_periods and a realised implementation target."
            )
        generator = np.random.default_rng(rng)
        draws = self.sample(n_paths, generator, target)
        means, deviations = draws["mean"].to_numpy(), draws["sd"].to_numpy()
        return dict(
            values=generator.normal(
                means[:, None], deviations[:, None], (n_paths, n_periods)
            ),
            means=means,
            sds=deviations,
            variant=draws.variant.to_numpy(),
            expected_values=np.broadcast_to(
                means[:, None], (n_paths, n_periods)
            ).copy(),
            conditional_observations="independent successive periods",
            years=self.period_years * np.arange(1, n_periods + 1),
        )

    predictive_paths = sample_paths

    def correlation_sensitivity(self, scenarios, threshold=0.0):
        """Refit declared fixed-R scenarios; this does not integrate covariance uncertainty."""
        rows = []
        for name, correlation in scenarios.items():
            fit = type(self)(
                pd.DataFrame(self.data, columns=self.names),
                correlation=correlation,
                construction_correlation=self.construction_correlation,
                weights=self.weights.to_dict(),
                prior=self.prior,
                tau_scale=self.tau_scale,
                metric=self.metric,
                sigma_grid=self.sigma,
                tau_grid=self.tau,
            )
            rows.append(
                dict(
                    scenario=name,
                    expected_metric=fit.expectation(),
                    posterior_sd=fit.std(),
                    prob_mean_positive=1 - fit.prob_mean_below(threshold),
                )
            )
        return pd.DataFrame(rows).set_index("scenario")
