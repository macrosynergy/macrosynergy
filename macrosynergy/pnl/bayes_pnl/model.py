"""Public single-strategy interface to the period-metric grid benchmark."""

from typing import Optional, Union

import numpy as np
import pandas as pd

from .grid import GridSpec, posterior_grid
from .periods import period_metrics
from .priors import default_prior
from .quadrature import QuadraturePosterior
from .inputs import supports_quadrature


class BayesianPnL:
    """Estimate a period metric's expectation, dispersion and next-period risk.

    Parameters
    ----------
    data : one-dimensional array-like
        Finite precomputed period metrics for one strategy, such as seven
        annualised Sharpe ratios from non-overlapping three-year periods.
        At least one observation is required. Sharpe/return metrics have a normal
        likelihood; strictly positive volatility metrics have a Gamma likelihood.
        Period metrics are conditionally independent. Neither observation model
        estimates regimes or a research-selection correction.
    metric : {"sharpe", "ann_ret", "ann_vol"}
        Metric label and default-prior choice. Return and volatility use the
        same units as the PnL supplied to :meth:`from_pnl`.
    prior : object, optional
        Vectorised logpdf(mean, sd). Defaults to an IndependentPrior with a
        normal mean for Sharpe/return or positive log-normal mean for volatility,
        and inverse-gamma variance prior transformed to the sd axis.
        A normalised=True attribute permits ordinary evidence for
        an untempered fit. Custom priors without quantiles may require an
        explicit GridSpec and careful boundary checks.
    grid : GridSpec or "auto"
        Finite grid for explicit grid inference or lazy quadrature figures.
    method : {"auto", "quadrature", "grid"}
        Auto uses adaptive integration for normal likelihoods with an independent
        normal-mean prior; volatility and custom joint priors use grid inference.
        Quadrature eliminates the
        mean analytically and its summaries do not depend on the plotting grid.
    period_years : float
        Duration represented by each observation; default three years.
        Predictions concern another period of that same duration.
    likelihood_weight : float
        Positive power-likelihood weight, default 1. Down-weighting is a
        sensitivity approximation, not an estimated dependence correction.
    vol_target : float
        Annual volatility reference for default return/volatility priors.
        Default 10 assumes PnL in percent; use 0.10 for decimal returns.

    Notes
    -----
    The joint posterior describes the underlying expectation and dispersion.
    The predictive distribution integrates both parameters and includes future
    performance variation. Neither treats additional strategy variants as
    independent historical periods. Expected period Sharpe is not necessarily
    the Sharpe of the concatenated return history. For volatility, both the
    default mean prior and the Gamma observation model have positive support.
    The dispersion prior describes variation between measured period metrics,
    combining sampling noise and genuine variation; it is not a regime model.

    Examples
    --------
    >>> fit = BayesianPnL([0.4, -0.2, 0.8, 0.1, 0.5])
    >>> interval = fit.predictive_quantile([0.05, 0.95])
    """

    def __init__(
        self,
        data,
        metric: str = "sharpe",
        prior=None,
        grid: Union[GridSpec, str] = "auto",
        period_years: float = 3.0,
        likelihood_weight: float = 1.0,
        vol_target: float = 10.0,
        method: str = "auto",
    ):
        selected_prior = default_prior(metric, vol_target) if prior is None else prior
        if method not in ("auto", "quadrature", "grid"):
            raise ValueError("method must be 'auto', 'quadrature' or 'grid'.")
        selected_method = (
            (
                "quadrature"
                if metric != "ann_vol" and supports_quadrature(selected_prior)
                else "grid"
            )
            if method == "auto"
            else method
        )
        backend = (
            QuadraturePosterior if selected_method == "quadrature" else posterior_grid
        )
        self.posterior = backend(
            data,
            selected_prior,
            grid=grid,
            metric=metric,
            period_years=period_years,
            likelihood_weight=likelihood_weight,
        )
        self.method = selected_method
        self.periods = None

    @classmethod
    def from_pnl(
        cls,
        pnl: Union[pd.Series, pd.DataFrame],
        metric: str = "sharpe",
        annualization_factor: float = 252,
        prior=None,
        grid: Union[GridSpec, str] = "auto",
        likelihood_weight: float = 1.0,
        vol_target: float = 10.0,
        method: str = "auto",
        *,
        end_year: Optional[int] = None,
    ):
        """Fit daily PnL in complete, non-overlapping three-calendar-year blocks.

        Daily observations are annualised with annualization_factor, without
        monthly aggregation. Metrics use arithmetic mean and sample sd.
        end_year optionally selects an earlier calendar-year partition anchor.
        See :func:`period_metrics` for input units and missing-data rules.
        Supply separate fits for strategy variants rather than pooling columns.
        """
        if isinstance(pnl, pd.DataFrame) and pnl.shape[1] != 1:
            raise ValueError("Fit one strategy at a time; select a single PnL column.")
        if metric not in ("sharpe", "ann_ret", "ann_vol"):
            raise ValueError("metric must be 'sharpe', 'ann_ret' or 'ann_vol'.")
        periods = period_metrics(
            pnl,
            end_year=end_year,
            annualization_factor=annualization_factor,
        )
        if periods.empty:
            raise ValueError(
                "No complete periods are available for the requested block length."
            )
        if not np.isfinite(periods[metric].to_numpy()).all():
            raise ValueError(
                "The period metric is undefined; check for zero-volatility blocks."
            )
        result = cls(
            periods[metric].to_numpy(),
            metric=metric,
            prior=prior,
            grid=grid,
            period_years=3.0,
            likelihood_weight=likelihood_weight,
            vol_target=vol_target,
            method=method,
        )
        result.periods = periods
        return result

    @property
    def metric(self):
        """Name of the fitted period metric."""
        return self.posterior.metric

    @property
    def period_years(self):
        """Duration represented by one metric observation."""
        return self.posterior.period_years

    def summary(self, level=0.9):
        """Return parameter summaries and next-period predictive probabilities."""
        return self.posterior.summary(level=level)

    def expectation(self, param="mean"):
        """Posterior mean of the expected metric or its across-period sd."""
        return self.posterior.expectation(param)

    def interval(self, param="mean", level=0.9, kind="eti"):
        """Parameter credible interval, distinct from a predictive interval."""
        return self.posterior.interval(param, level=level, kind=kind)

    def prob_mean_below(self, threshold=0.0):
        """Posterior probability that expected performance is below threshold."""
        return self.posterior.prob_mean_below(threshold)

    def predictive_pdf(self, x):
        """Density of the next same-duration period's realised metric."""
        return self.posterior.predictive_pdf(x)

    def predictive_cdf(self, x):
        """Probability of a next-period metric at or below each supplied value."""
        return self.posterior.predictive_cdf(x)

    def predictive_quantile(self, q):
        """Quantiles of next-period realised performance, including uncertainty."""
        return self.posterior.predictive_quantile(q)

    def predictive_sample(self, n, rng=None):
        """Draw next-period outcomes; rng accepts a seed or NumPy Generator."""
        return self.posterior.predictive_sample(n, rng=rng)
