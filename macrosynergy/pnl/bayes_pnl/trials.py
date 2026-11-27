"""Evaluate a selected strategy version conditional on every version explored.

Reporting the best of many explored versions inflates its apparent value when
the version is assessed on its own. ``BayesianPnLTrials`` models all explored
versions jointly, with known sampling noise and correlated sampling errors, so
the posterior of the reported version is conditioned on the whole search. As
selection is a function of the data, a posterior that conditions on all the data
needs no further adjustment for the selection itself.
"""

import warnings

import numpy as np
import pandas as pd
from scipy import linalg, optimize, special, stats

from .grid import GridTruncationWarning
from .inputs import evaluation_points, probabilities
from .periods import period_metrics
from .priors import NormalPrior
from .variants import _axis, _matrix, _widths

RESERVED_TARGETS = ("selected", "idea", "draw")
_EULER_GAMMA = float(np.euler_gamma)
_REFINE_TAIL = 1e-9


def effective_number_of_trials(correlation):
    """Effective number of independent versions implied by a correlation matrix.

    Computed as ``K**2 / ||R||_F**2``, where ``||R||_F**2`` is the sum of
    squared entries of R. It equals K for uncorrelated versions and one for
    identical versions.

    Parameters
    ----------
    correlation : array-like
        K x K symmetric correlation matrix with unit diagonal.

    Returns
    -------
    float
    """
    R = np.asarray(correlation, dtype=float)
    if R.ndim != 2 or R.shape[0] != R.shape[1] or not R.size:
        raise ValueError("correlation must be a nonempty square matrix.")
    R = _matrix(R, R.shape[0], "correlation")
    return float(R.shape[0] ** 2 / np.sum(R**2))


def expected_maximum_sharpe(sharpes, n_trials=None):
    """Expected maximum Sharpe ratio of N unskilled trials (deflated-Sharpe benchmark).

    ``sd(sharpes, ddof=1) * ((1 - g) * Phi^-1(1 - 1/N) + g * Phi^-1(1 - 1/(N e)))``
    with g the Euler-Mascheroni constant (Bailey and Lopez de Prado, 2014).

    Parameters
    ----------
    sharpes : array-like
        Sharpe ratios of the explored trials; their sample sd sets the scale.
    n_trials : int, optional
        Number of (effective) independent trials N. Defaults to ``len(sharpes)``.

    Returns
    -------
    float
        Benchmark in the units of ``sharpes``; 0.0 when N <= 1.
    """
    values = np.asarray(sharpes, dtype=float).ravel()
    if not np.isfinite(values).all():
        raise ValueError("sharpes must be finite.")
    n = float(values.size if n_trials is None else n_trials)
    if not np.isfinite(n) or n <= 0:
        raise ValueError("n_trials must be positive and finite.")
    if n <= 1:
        return 0.0
    if values.size < 2:
        raise ValueError("At least two Sharpe ratios are needed for their dispersion.")
    spread = float(np.std(values, ddof=1))
    return spread * (
        (1 - _EULER_GAMMA) * stats.norm.ppf(1 - 1 / n)
        + _EULER_GAMMA * stats.norm.ppf(1 - 1 / (n * np.e))
    )


def _positive(value, name):
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("{} must be a positive finite number.".format(name)) from exc
    if not np.isfinite(result) or result <= 0:
        raise ValueError("{} must be a positive finite number.".format(name))
    return result


def _count(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 2:
        raise ValueError("{} must be an integer of at least two.".format(name))
    return int(value)


def _mixture_cdf(values, means, sds, weights):
    positive = sds > 0
    result = []
    for value in values:
        components = np.asarray(value >= means, dtype=float)
        components[positive] = special.ndtr(
            (value - means[positive]) / sds[positive]
        )
        result.append(float(weights @ components))
    return np.asarray(result)


class BayesianPnLTrials:
    """Selection-aware hierarchy for the Sharpe ratios of all explored versions.

    Data are a T x K matrix of annualised Sharpe ratios s_tv over T
    non-overlapping periods of ``period_years`` years for K explored versions::

        s_t | theta, eta_t ~ N_K(theta + eta_t * 1, V),  V = noise_sd**2 * R
        eta_t ~ N(0, omega**2)          (period effect common to all versions)
        theta = m * 1 + a,  a ~ N_K(0, tau**2 C),  m ~ N(m0, s0**2)
        omega ~ half-normal(omega_scale),  tau ~ half-normal(tau_scale)

    V is *known* sampling noise of a period Sharpe ratio: there is no free
    common dispersion, so dispersion cannot be learned from differences between
    near-identical versions, and R is the correlation of the versions' sampling
    errors (in practice their daily PnL correlation), not an assumed correlation
    of performance. m, a and eta are integrated analytically; (omega, tau) on a
    two-dimensional grid with trapezoid cell widths.

    Because the posterior conditions on every explored version, selecting the
    best in-sample version does not inflate its posterior: the selection rule
    is a function of the data the posterior already conditions on. Near-duplicate
    versions add almost no information because their sampling errors are
    almost perfectly correlated.

    Parameters
    ----------
    data : pandas.DataFrame or two-dimensional array
        Finite period Sharpe ratios, one column per explored version (T >= 1).
    noise_correlation : array-like or pandas.DataFrame, optional
        K x K positive-definite correlation R of the sampling errors. Required
        for K > 1. A DataFrame is aligned to the version names.
    period_years : float
        Length of each period in years.
    selected : str, optional
        Name of the reported version; the default target when given.
    noise_sd : float, optional
        Common sd of a period Sharpe's sampling error. Defaults to
        ``sqrt((1 + g**2 / (2 * observations_per_year)) / period_years)`` with
        g the grand mean of data: the large-sample variance of an annualised
        Sharpe ratio estimated from independent returns (Lo, 2002).
        It is deliberately one number for all versions and periods: with
        correlated versions, version-specific standard errors that rise with
        each version's own Sharpe estimate bias the pooled level downward.
    observations_per_year : float
        Return observations per year behind each period Sharpe, default 252
        (daily). Only used for the default ``noise_sd``.
    mean_prior : NormalPrior, optional
        Prior of the common level m, default N(0, 1).
    construction_correlation : None, "noise" or array-like, optional
        Prior correlation C of the construction effects a. Default None is the
        identity (exchangeable, independent effects). "noise" sets C = R, so
        versions whose PnLs are highly correlated also have similar true
        Sharpes. With the identity, a near-duplicate of an explored version
        counts as a second independent draw of a construction effect that
        happens to coincide with the first: it shrinks tau and moves the
        other versions' posteriors. With C = R a near-duplicate adds almost
        nothing. An explicit matrix must be a positive-semidefinite
        correlation matrix (a DataFrame is aligned to the version names).
    tau_scale, omega_scale : float
        Half-normal scales of the version spread tau and the period effect omega.
    weights : dict or one-dimensional array, optional
        Research-choice probabilities of the versions for target "draw";
        default equal. They never multiply the likelihood.
    variant_names : sequence of str, optional
        Names for array data.
    n_tau, n_omega : int
        Default grid resolutions.
    tau_grid, omega_grid : one-dimensional arrays, optional
        Explicit nonnegative, increasing axes. A singleton axis fixes the
        parameter (its prior is then not applied).

    Notes
    -----
    Default axes run linearly from zero to the larger of the half-normal
    1 - 1e-5 quantile and a data-based bound. When the posterior occupies only
    a small part of a default axis, that axis is refitted once on a linear grid
    spanning the cells with marginal mass above 1e-9 (the zero end is kept
    unless its mass is negligible). A ``GridTruncationWarning`` flags more than
    1% of posterior mass on an upper edge.

    Predictive distributions for a future period of ``horizon_years`` assume
    period effects and sampling errors independent across periods, so both
    variances scale with ``period_years / horizon_years``. The model assumes
    the true values theta are constant over time and does not integrate
    uncertainty about R or ``noise_sd``.
    """

    method = "trials_grid"

    def __init__(
        self,
        data,
        noise_correlation=None,
        *,
        period_years,
        selected=None,
        noise_sd=None,
        mean_prior=None,
        tau_scale=0.5,
        omega_scale=0.5,
        weights=None,
        variant_names=None,
        n_tau=61,
        n_omega=61,
        tau_grid=None,
        omega_grid=None,
        construction_correlation=None,
        observations_per_year=252,
    ):
        if np.iscomplexobj(data):
            raise ValueError("Sharpe observations must be real.")
        try:
            x = np.asarray(data, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("data must be a numeric period-by-version matrix.") from exc
        if x.ndim != 2 or not x.size or not np.isfinite(x).all():
            raise ValueError("Supply a nonempty finite period-by-version matrix.")
        if isinstance(data, pd.DataFrame):
            names = list(data.columns)
        elif variant_names is not None:
            names = list(variant_names)
        else:
            names = ["V{}".format(i) for i in range(x.shape[1])]
        if (
            len(names) != x.shape[1]
            or len(set(names)) != len(names)
            or any(not isinstance(v, str) for v in names)
        ):
            raise ValueError("Version names must be unique strings, one per column.")
        if any(v in RESERVED_TARGETS for v in names):
            raise ValueError(
                "Version names cannot be reserved targets: selected, idea, draw."
            )
        k = len(names)
        if selected is not None and selected not in names:
            raise ValueError("selected must name one of the versions.")
        self.period_years = _positive(period_years, "period_years")
        if noise_correlation is None:
            if k > 1:
                raise ValueError(
                    "Supply noise_correlation (e.g. daily PnL correlation) for multiple versions."
                )
            R = np.eye(1)
        else:
            R = _matrix(self._aligned(noise_correlation, names), k, "noise_correlation")
        try:
            np.linalg.cholesky(R)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                "noise_correlation must be positive definite; drop duplicate versions."
            ) from exc
        if np.linalg.eigvalsh(R)[0] < 1e-10:
            raise ValueError(
                "noise_correlation is numerically singular; drop duplicate versions."
            )
        if construction_correlation is None:
            C = np.eye(k)
        elif isinstance(construction_correlation, str):
            if construction_correlation != "noise":
                raise ValueError(
                    "construction_correlation must be None, 'noise' or a matrix."
                )
            C = R.copy()
        else:
            C = _matrix(
                self._aligned(construction_correlation, names),
                k,
                "construction_correlation",
            )
            if np.linalg.eigvalsh(C)[0] < -1e-10:
                raise ValueError("construction_correlation must be positive semidefinite.")
        self.construction_correlation = C
        prior = NormalPrior(0.0, 1.0) if mean_prior is None else mean_prior
        if type(prior) is not NormalPrior:
            raise TypeError("mean_prior must be a NormalPrior.")
        self.tau_scale = _positive(tau_scale, "tau_scale")
        self.omega_scale = _positive(omega_scale, "omega_scale")
        n_tau, n_omega = _count(n_tau, "n_tau"), _count(n_omega, "n_omega")
        grand = float(x.mean())
        per_year = _positive(observations_per_year, "observations_per_year")
        self.noise_sd = (
            float(np.sqrt((1 + grand**2 / (2 * per_year)) / self.period_years))
            if noise_sd is None
            else _positive(noise_sd, "noise_sd")
        )
        self._weights = pd.Series(
            self._validate_weights(weights, names), index=names, name="research_weight"
        )
        self.names, self.selected, self.mean_prior = tuple(names), selected, prior
        self.data = x.copy()
        self.data.setflags(write=False)
        self.n_periods, self.n_variants = x.shape
        self.noise_correlation = R
        self.effective_trials = effective_number_of_trials(R)
        self.periods = None
        self._prepare()
        upper_omega = max(
            float(stats.halfnorm.ppf(1 - 1e-5, scale=self.omega_scale)),
            4 * float(np.std(x.mean(axis=1), ddof=1)) if self.n_periods > 1 else 0.0,
        )
        upper_tau = max(
            float(stats.halfnorm.ppf(1 - 1e-5, scale=self.tau_scale)),
            4 * float(np.std(x.mean(axis=0))) + abs(float(x.mean()) - prior.mean),
        )
        omega_axis = (
            np.linspace(0.0, upper_omega, n_omega)
            if omega_grid is None
            else _axis(omega_grid, False, "omega_grid")
        )
        tau_axis = (
            np.linspace(0.0, upper_tau, n_tau)
            if tau_grid is None
            else _axis(tau_grid, False, "tau_grid")
        )
        self._fit(omega_axis, tau_axis)
        refine_omega, refine_tau = omega_grid is None, tau_grid is None
        if refine_omega or refine_tau:
            new_omega = (
                self._refined(omega_axis, self.mass.sum(axis=1))
                if refine_omega
                else omega_axis
            )
            new_tau = (
                self._refined(tau_axis, self.mass.sum(axis=0)) if refine_tau else tau_axis
            )
            if new_omega is not omega_axis or new_tau is not tau_axis:
                self._fit(new_omega, new_tau)
        edges = self.diagnostics["edge_mass"]
        if max(edges.values()) > 0.01:
            warnings.warn(
                "Trials grid has material edge mass {}; widen the omega/tau axes.".format(
                    edges
                ),
                GridTruncationWarning,
                stacklevel=2,
            )

    # ------------------------------------------------------------------ setup

    @staticmethod
    def _validate_weights(weights, names):
        k = len(names)
        if weights is None:
            return np.full(k, 1.0 / k)
        if isinstance(weights, dict):
            if set(weights) != set(names):
                raise ValueError("Weights must name every version exactly once.")
            w = np.array([weights[name] for name in names], dtype=float)
        else:
            if np.iscomplexobj(weights):
                raise ValueError("Weights must be real-valued.")
            w = np.asarray(weights, dtype=float)
        with np.errstate(over="ignore"):
            total = float(w.sum()) if w.size else 0.0
        if (
            w.shape != (k,)
            or not np.isfinite(w).all()
            or (w < 0).any()
            or not np.isfinite(total)
            or total <= 0
        ):
            raise ValueError(
                "Weights must be finite, nonnegative and have positive total mass."
            )
        return w / total

    @staticmethod
    def _aligned(matrix, names):
        if not isinstance(matrix, pd.DataFrame):
            return matrix
        if set(matrix.index) != set(names) or set(matrix.columns) != set(names):
            raise ValueError("Correlation labels must match the version names.")
        return matrix.loc[names, names].to_numpy()

    @property
    def weights(self):
        """Normalised research-choice weights, returned as an independent copy."""
        return self._weights.copy()

    def _prepare(self):
        """Data summaries in the basis that diagonalises V and C simultaneously.

        The generalised eigenproblem C w = mu V w gives W with W'VW = I and
        W'CW = diag(mu), so A = tau**2 C + V/T is diagonal in that basis for
        every tau: W'AW = diag(tau**2 mu + 1/T). M = W^-1 maps back.
        """
        T, K = self.data.shape
        mean = self.data.mean(axis=0)
        residuals = self.data - mean
        scatter = residuals.T @ residuals
        V = self.noise_sd**2 * self.noise_correlation
        mu, W = linalg.eigh(self.construction_correlation, V)
        mu = np.maximum(mu, 0.0)
        u1, us = W.sum(axis=0), W.T @ mean
        rotated_scatter = W.T @ scatter @ W
        self._W, self._M, self._mu, self._u1, self._us = W, linalg.inv(W), mu, u1, us
        self._q11 = float(u1 @ u1)
        self._trace_VS = float(np.trace(rotated_scatter))
        self._wSw = float(u1 @ rotated_scatter @ u1)
        self._logdet_V = 2 * float(np.log(np.diag(np.linalg.cholesky(V))).sum())

    @staticmethod
    def _refined(axis, marginal):
        if len(axis) < 3:
            return axis
        support = np.flatnonzero(marginal > _REFINE_TAIL)
        if not support.size:
            return axis
        low, high = max(support[0] - 1, 0), min(support[-1] + 1, len(axis) - 1)
        if high - low + 1 >= len(axis) / 3:
            return axis
        return np.linspace(axis[low], axis[high], len(axis))

    def _fit(self, omega, tau):
        """Exact Gaussian integration over m, a and eta at each (omega, tau) cell.

        With A = tau**2 C + V/T (diagonal in the basis of ``_prepare``) and
        c = s0**2 + omega**2/T, the period mean is s_bar = b 1 + z with
        b = m + mean(eta) ~ N(m0, c) independent of z = a + mean(e) ~ N(0, A).
        Lambda = A + c 11' and every quantity follows from Sherman-Morrison.
        Given (b, z), m depends on b only and a on z only, which yields the
        posterior moments of theta and m as sums of nonnegative terms.
        """
        self.omega, self.tau = omega, tau
        T, K = self.data.shape
        m0, s0 = self.mean_prior.mean, self.mean_prior.sd
        mu, u1, us, M = self._mu, self._u1, self._us, self._M
        ud = us - m0 * u1
        om2, ta2 = omega**2, tau**2
        # Residual scatter term (omega only): Sigma_w = V + omega^2 11'.
        shrink = 1 + om2 * self._q11
        scatter_term = -0.5 * (T - 1) * (self._logdet_V + np.log(shrink)) - 0.5 * (
            self._trace_VS - om2 / shrink * self._wSw
        )
        # Per-tau summaries of A = tau^2 C + V/T in the joint eigenbasis.
        lam = ta2[:, None] * mu[None, :] + 1.0 / T
        a11 = (u1**2 / lam).sum(axis=1)
        a1d = (u1 * ud / lam).sum(axis=1)
        add = (ud**2 / lam).sum(axis=1)
        a1s = (u1 * us / lam).sum(axis=1)
        logdet_A = self._logdet_V + np.log(lam).sum(axis=1)
        # tau^2 C A^-1 x = tau^2 M' diag(mu / lam) W'x; Var(a | z) diagonal.
        effect_1 = ta2[:, None] * ((mu * u1 / lam) @ M)
        effect_s = ta2[:, None] * ((mu * us / lam) @ M)
        a_post_var = (ta2[:, None] * mu / (T * lam)) @ (M**2)
        # Cells: rows omega, columns tau.
        c = (s0**2 + om2 / T)[:, None]
        den = 1 + c * a11[None, :]
        quad = add[None, :] - c * a1d[None, :] ** 2 / den
        mean_term = -0.5 * (K * np.log(2 * np.pi) + logdet_A[None, :] + np.log(den) + quad)
        constant = -0.5 * (T - 1) * K * np.log(2 * np.pi) - 0.5 * K * np.log(T)
        log_likelihood = constant + scatter_term[:, None] + mean_term
        vb = c / den
        b_hat = vb * (m0 / c + a1s[None, :])
        ratio = s0**2 / c
        within_m = s0**2 * (om2[:, None] / T) / c
        idea_mean = m0 + ratio * (b_hat - m0)
        idea_var = within_m + ratio**2 * vb
        beta = ratio[:, :, None] - effect_1[None, :, :]
        alpha = m0 * (1 - ratio)[:, :, None] + effect_s[None]
        theta_mean = alpha + beta * b_hat[:, :, None]
        theta_var = within_m[:, :, None] + a_post_var[None] + beta**2 * vb[:, :, None]
        omega_cells = np.broadcast_to(omega[:, None], log_likelihood.shape)
        tau_cells = np.broadcast_to(tau[None, :], log_likelihood.shape)
        log_mass = log_likelihood.copy()
        if len(omega) > 1:
            log_mass += stats.halfnorm.logpdf(omega_cells, scale=self.omega_scale)
        if len(tau) > 1:
            log_mass += stats.halfnorm.logpdf(tau_cells, scale=self.tau_scale)
        log_mass += np.log(np.outer(_widths(omega), _widths(tau)))
        normalizer = special.logsumexp(log_mass)
        if not np.isfinite(normalizer):
            raise ValueError("No finite mass on the omega/tau grid.")
        self.mass = np.exp(log_mass - normalizer)
        self.log_evidence = float(normalizer)
        self._omega_cells, self._tau_cells = omega_cells.ravel(), tau_cells.ravel()
        self._theta_mean = theta_mean.reshape(-1, K)
        self._theta_var = np.maximum(theta_var.reshape(-1, K), 0.0)
        self._idea_mean, self._idea_var = idea_mean.ravel(), idea_var.ravel()
        marginal_omega, marginal_tau = self.mass.sum(axis=1), self.mass.sum(axis=0)
        self.diagnostics = {
            "edge_mass": {
                "omega_high": float(marginal_omega[-1]) if len(omega) > 1 else 0.0,
                "tau_high": float(marginal_tau[-1]) if len(tau) > 1 else 0.0,
            },
            "n_periods": T,
            "n_versions": K,
            "effective_trials": self.effective_trials,
            "noise_sd": self.noise_sd,
        }

    # ------------------------------------------------------------- from_pnl

    @classmethod
    def from_pnl(
        cls,
        pnl,
        *,
        period_years=1,
        annualization_factor=252,
        end_year=None,
        selected=None,
        **kwargs,
    ):
        """Fit from daily PnL, one column per explored version.

        Period Sharpe ratios come from ``period_metrics`` in calendar blocks of
        ``period_years`` whole years; periods without a finite Sharpe for every
        version are dropped jointly with a warning. The noise correlation is the
        correlation of daily PnL on days inside the retained periods where all
        versions have data.
        """
        if isinstance(pnl, pd.Series):
            pnl = pnl.to_frame(name=pnl.name if pnl.name is not None else "V0")
        if not isinstance(pnl, pd.DataFrame):
            raise TypeError("pnl must be a dated daily Series or DataFrame.")
        if "noise_correlation" in kwargs:
            raise TypeError("from_pnl computes noise_correlation from the daily PnL.")
        periods = period_metrics(
            pnl,
            annualization_factor=annualization_factor,
            end_year=end_year,
            period_years=period_years,
        )
        if periods.empty:
            raise ValueError("No complete shared periods are available.")
        matrix = periods.pivot(
            index="period_end", columns="variant", values="sharpe"
        ).reindex(columns=pnl.columns)
        complete = matrix.notna().all(axis=1) & np.isfinite(matrix.to_numpy()).all(
            axis=1
        )
        if not complete.all():
            warnings.warn(
                "Periods without finite Sharpe ratios for every version are excluded jointly.",
                UserWarning,
                stacklevel=2,
            )
        matrix = matrix.loc[complete]
        if matrix.empty:
            raise ValueError("No complete shared periods are available.")
        retained = periods.loc[periods.period_end.isin(matrix.index)]
        inside = np.zeros(len(pnl), dtype=bool)
        for start, end in retained[["period_start", "period_end"]].drop_duplicates().itertuples(
            index=False
        ):
            inside |= (pnl.index >= start) & (pnl.index < end + pd.Timedelta(1, unit="D"))
        daily = pnl.loc[inside].dropna(how="any")
        if pnl.shape[1] > 1:
            correlation = np.corrcoef(daily.to_numpy(dtype=float), rowvar=False)
            if not np.isfinite(correlation).all():
                raise ValueError("Daily PnL correlation is undefined (constant series).")
            correlation = (correlation + correlation.T) / 2
            np.fill_diagonal(correlation, 1.0)
            names = list(pnl.columns)
            for i in range(len(names)):
                for j in range(i):
                    if abs(correlation[i, j]) > 1 - 1e-10:
                        raise ValueError(
                            "Versions {!r} and {!r} have perfectly collinear daily PnL; "
                            "drop the duplicate.".format(names[j], names[i])
                        )
            if np.linalg.eigvalsh(correlation)[0] < 1e-10:
                raise ValueError(
                    "Daily PnL of some versions is perfectly collinear; drop duplicate "
                    "or linearly dependent versions."
                )
            correlation = pd.DataFrame(correlation, index=names, columns=names)
        else:
            correlation = None
        kwargs.setdefault("observations_per_year", annualization_factor)
        result = cls(
            matrix,
            correlation,
            period_years=period_years,
            selected=selected,
            **kwargs,
        )
        result.periods = retained.copy()
        return result

    # ------------------------------------------------------------- targets

    def _resolve(self, target):
        if target is None:
            return "selected" if self.selected is not None else "draw"
        if target == "selected":
            if self.selected is None:
                raise ValueError("No selected version was declared.")
            return target
        if target in ("idea", "draw") or target in self.names:
            return target
        raise ValueError(
            "Unknown target {!r}; use selected, idea, draw or a version name.".format(
                target
            )
        )

    def _horizon(self, horizon_years):
        return (
            self.period_years
            if horizon_years is None
            else _positive(horizon_years, "horizon_years")
        )

    def _components(self, target, predictive=False, horizon_years=None):
        target = self._resolve(target)
        masses = self.mass.ravel()
        noise = (
            (self._omega_cells**2 + self.noise_sd**2)
            * self.period_years
            / self._horizon(horizon_years)
            if predictive
            else np.zeros_like(masses)
        )
        if target == "idea":
            if predictive:
                raise ValueError(
                    "The common level has no realised-period outcome; use a version target."
                )
            return self._idea_mean, self._idea_var, masses
        if target == "draw":
            q = self.weights.to_numpy()
            return (
                self._theta_mean.ravel(),
                (self._theta_var + noise[:, None]).ravel(),
                (masses[:, None] * q).ravel(),
            )
        index = self.names.index(self.selected if target == "selected" else target)
        return self._theta_mean[:, index], self._theta_var[:, index] + noise, masses

    # ------------------------------------------------------------- moments

    def expectation(self, target=None):
        """Posterior expectation of the long-term Sharpe of the target."""
        means, _, weights = self._components(target)
        return float(weights @ means)

    def std(self, target=None):
        """Posterior sd of the long-term Sharpe of the target."""
        means, variances, weights = self._components(target)
        center = weights @ means
        return float(np.sqrt(weights @ (variances + (means - center) ** 2)))

    def cdf(self, x, target=None):
        means, variances, weights = self._components(target)
        return _mixture_cdf(evaluation_points(x), means, np.sqrt(variances), weights)

    def prob_mean_below(self, threshold=0.0, target=None):
        return float(self.cdf(threshold, target)[0])

    def prob_mean_above(self, threshold=0.0, target=None):
        return 1.0 - self.prob_mean_below(threshold, target)

    def predictive_cdf(self, x, target=None, horizon_years=None):
        """CDF of the target's Sharpe over a future period of ``horizon_years``."""
        means, variances, weights = self._components(target, True, horizon_years)
        return _mixture_cdf(evaluation_points(x), means, np.sqrt(variances), weights)

    def predictive_prob_above(self, threshold=0.0, target=None, horizon_years=None):
        return 1.0 - float(self.predictive_cdf(threshold, target, horizon_years)[0])

    def _quantiles(self, q, target, predictive, horizon_years=None):
        means, variances, weights = self._components(target, predictive, horizon_years)
        keep = weights > 0
        means, sds, weights = means[keep], np.sqrt(variances[keep]), weights[keep]
        center = float(weights @ means)
        width = max(float(sds.max()), float(np.ptp(means)), 1e-12)

        def function(z):
            return _mixture_cdf([z], means, sds, weights)[0]

        result = []
        for probability in probabilities(q):
            if probability in (0, 1):
                result.append(-np.inf if probability == 0 else np.inf)
                continue
            radius = width * 8
            for _ in range(30):
                low, high = center - radius, center + radius
                if function(low) <= probability <= function(high):
                    break
                radius *= 2
            else:
                raise ValueError("Unable to bracket mixture quantile.")
            result.append(
                optimize.brentq(
                    lambda z: function(z) - probability,
                    low,
                    high,
                    xtol=max(width * 1e-11, np.finfo(float).tiny),
                    rtol=1e-12,
                )
            )
        return np.asarray(result)

    def quantile(self, q, target=None):
        return self._quantiles(q, target, False)

    def interval(self, level=0.9, target=None):
        if not np.isfinite(level) or not 0 < level < 1:
            raise ValueError("level must lie strictly between zero and one.")
        return tuple(self.quantile([(1 - level) / 2, (1 + level) / 2], target))

    def predictive_quantile(self, q, target=None, horizon_years=None):
        return self._quantiles(q, target, True, horizon_years)

    # ------------------------------------------------------------- summaries

    def scale_summary(self, level=0.9):
        """Posterior mean and equal-tailed interval of omega and tau."""
        if not np.isfinite(level) or not 0 < level < 1:
            raise ValueError("level must lie strictly between zero and one.")
        rows = []
        for name, axis, marginal in (
            ("omega", self.omega, self.mass.sum(axis=1)),
            ("tau", self.tau, self.mass.sum(axis=0)),
        ):
            cumulative = np.cumsum(marginal)
            bounds = [
                float(axis[min(np.searchsorted(cumulative, p), len(axis) - 1)])
                for p in ((1 - level) / 2, (1 + level) / 2)
            ]
            rows.append(
                dict(
                    parameter=name,
                    mean=float(marginal @ axis),
                    lo=bounds[0],
                    hi=bounds[1],
                )
            )
        return pd.DataFrame(rows).set_index("parameter")

    def summary(self, level=0.9, thresholds=(0.0, 0.5), horizon_years=None):
        """Posterior and next-period results per target.

        Rows: selected (when declared), idea, draw and every version. Predictive
        columns are NaN for idea, which has no realised-period outcome.
        """
        targets = (("selected",) if self.selected is not None else ()) + (
            "idea",
            "draw",
        ) + self.names
        rows = []
        for target in targets:
            low, high = self.interval(level, target)
            row = dict(
                target=target,
                expectation=self.expectation(target),
                sd=self.std(target),
                lo=low,
                hi=high,
            )
            for threshold in thresholds:
                label = "{:g}".format(threshold)
                row["prob_mean_above_" + label] = self.prob_mean_above(threshold, target)
                row["prob_next_above_" + label] = (
                    np.nan
                    if target == "idea"
                    else self.predictive_prob_above(threshold, target, horizon_years)
                )
            rows.append(row)
        return pd.DataFrame(rows).set_index("target")
