"""Bayesian inference for daily PnL and observed three-year performance metrics."""

from .model import BayesianPnL
from .grid import (
    GridPosterior,
    GridSpec,
    GridTruncationWarning,
    auto_grid,
    posterior_grid,
)
from .periods import period_metrics
from .priors import (
    IndependentPrior,
    NormalPrior,
    LogNormalPrior,
    SDPrior,
    default_prior,
)
from .quadrature import QuadraturePosterior
from .posterior import Posterior
from .diagnostics import prior_predictive_sample, posterior_predictive_check
from .variants import BayesianPnLVariants
from .funnel import MetricFunnel, simulate_metric_funnel, bridge_metric_funnel
from .mixture import BayesianPnLMixture
from .trials import (
    BayesianPnLTrials,
    effective_number_of_trials,
    expected_maximum_sharpe,
)

__all__ = [
    "Posterior",
    "prior_predictive_sample",
    "posterior_predictive_check",
    "BayesianPnL",
    "BayesianPnLVariants",
    "BayesianPnLMixture",
    "BayesianPnLTrials",
    "effective_number_of_trials",
    "expected_maximum_sharpe",
    "MetricFunnel",
    "simulate_metric_funnel",
    "bridge_metric_funnel",
    "QuadraturePosterior",
    "GridPosterior",
    "GridSpec",
    "GridTruncationWarning",
    "auto_grid",
    "posterior_grid",
    "period_metrics",
    "IndependentPrior",
    "NormalPrior",
    "LogNormalPrior",
    "SDPrior",
    "default_prior",
]
