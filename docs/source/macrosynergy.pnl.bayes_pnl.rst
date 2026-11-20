Bayesian period-performance benchmark
=====================================

``BayesianPnL`` estimates the distribution of observed period metrics for one
strategy. Supply daily arithmetic excess PnL to ``from_pnl``, or supply a finite
one-dimensional vector of precomputed, equal-duration metrics directly.

Daily history is partitioned into non-overlapping three-calendar-year blocks.
There is no monthly aggregation or overlapping-window option. The optional
``end_year`` selects the final year of the latest block, allowing sensitivity
checks against alternative partitions. Partial initial/final years are omitted;
explicit missing values invalidate their block. Coverage uses a documented
seven-day boundary/gap heuristic, rather than an exchange holiday calendar.

.. code-block:: python

   from macrosynergy.pnl import BayesianPnL
   from macrosynergy.pnl.bayes_pnl import (
       IndependentPrior, NormalPrior, SDPrior,
       prior_predictive_sample, posterior_predictive_check,
   )

   fit = BayesianPnL.from_pnl(daily_pnl)
   expected_sharpe = fit.expectation("mean")
   parameter_interval = fit.interval("mean")
   next_period_interval = fit.predictive_quantile([0.05, 0.95])
   checks = posterior_predictive_check(fit, rng=7)
   possible_before_data = prior_predictive_sample(fit.posterior.prior, 10000, rng=7)

Model and interpretation
------------------------

Sharpe and annual-return observations use a normal likelihood with an independent
normal prior on their expectation and an inverse-gamma prior on their variance
(transformed to the dispersion axis). The built-in prior expectation is zero.
Normal mean integration is analytic; adaptive one-dimensional quadrature handles
dispersion. An explicit grid remains available for arbitrary joint priors.

Annual-volatility observations use a **Gamma likelihood**, parameterized by its
arithmetic mean and standard deviation. The default expectation prior is
``LogNormalPrior(vol_target, 0.5 * vol_target)``; the dispersion prior is
``SDPrior(0.2 * vol_target, 0.2 * vol_target)``. Both parameters and future
volatilities have positive support. ``LogNormalPrior`` accepts arithmetic moments
in input units, not log-scale parameters. Volatility uses grid inference; explicit
normal-model quadrature is rejected. Its Gamma density may diverge at zero for
some parameter values, but it assigns zero probability to nonpositive outcomes.

The inverse-gamma prior on dispersion alone would not constrain the sign of a
normal observation model. Positivity here comes from the combination of a
positive expectation prior and a positive observation likelihood.

The mean parameter describes expected **observed period performance**. Dispersion
combines sampling noise in each period estimate with genuine performance
variation. Expected period Sharpe need not equal the Sharpe of concatenated
returns. Parameter intervals and next-period predictive intervals answer
different questions.

Non-overlapping periods do not establish independence or stationarity. Daily
annualisation uses ``252 * mean`` and ``sqrt(252) * sample_sd`` by default; it
does not estimate a serial-correlation correction. Likelihood tempering is a
sensitivity assumption, not an estimated correction for dependence. The model
also does not adjust for research selection or estimate regimes. See
`Lo, The Statistics of Sharpe Ratios <https://doi.org/10.2469/faj.v58.n4.2453>`_.

Assessing assumptions
---------------------

With a small number of three-year blocks, report prior and partition sensitivity
alongside posterior probabilities. Change prior widths explicitly and refit the
same observations. For daily data, refit using alternative ``end_year`` anchors;
compare both results and the periods retained, since anchors can change sample
size. Return and volatility prior units must match the PnL (``vol_target=0.10``
for decimal returns targeting ten-percent annual volatility).

``prior_predictive_sample`` shows which outcomes a built-in prior permits before
observing data. ``posterior_predictive_check`` replicates histories of the same
length, comparing their mean, dispersion and extremes with the observed history.
Its tail probabilities are descriptive discrepancies, not calibrated hypothesis
tests. Check predictive performance on held-out periods where possible. See the
`Stan guide to predictive checks <https://mc-stan.org/docs/stan-users-guide/posterior-predictive-checks.html>`_.

Inference, diagnostics and figures have separate interfaces. ``Posterior`` is
the common inference contract; ``GridPosterior`` and ``QuadraturePosterior`` are
separate implementations. ``posterior.to_grid()`` explicitly requests a finite
visualization approximation. Quadrature reports, samples and predictive checks
do not request this grid. Plot functions return Matplotlib objects without
calling ``show``.

API
---

.. automodule:: macrosynergy.pnl.bayes_pnl
   :members:
   :imported-members:
   :undoc-members:
   :show-inheritance:
