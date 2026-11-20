"""Bayesian period-performance benchmark, using synthetic daily PnL.

Install the checkout with ``pip install -e .``, then run from the repository:
    python examples/bayes_pnl.py
For a headless run that saves figures:
    python examples/bayes_pnl.py --no-show --output-dir dev/bayes-pnl-example
For the explicit grid approximation instead of adaptive integration:
    python examples/bayes_pnl.py --method grid --no-show

Daily PnL is cut into non-overlapping three-year blocks. A normal likelihood
over their Sharpe ratios updates a configurable prior on expected performance
and its dispersion. The parameter posterior differs from the next-period
predictive distribution: the latter also includes future performance variation.
Related constructions are fitted separately, not pooled as independent history.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def synthetic_pnls():
    """Three fixed allocation choices with shared shocks and known risk scaling."""
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2004-01-01", "2024-12-31")
    covariance = np.array([[1.0, 0.3], [0.3, 1.0]])
    shocks = rng.multivariate_normal([0, 0], covariance, size=len(dates))
    # Changing expected performance is part of this toy data generator. The grid
    # summarises period variation; it does not estimate the hidden state process.
    states = rng.normal(0.35, 0.4, 7)
    annual_mean = states[(dates.year - 2004) // 3]
    assets = (10 / np.sqrt(252)) * shocks + (10 / 252) * annual_mean[:, None]
    constructions = {
        "baseline": [0.75, 0.25],
        "equal_weight": [0.5, 0.5],
        "concentrated": [1.0, 0.0],
    }
    result = {}
    for name, values in constructions.items():
        weights = np.asarray(values)
        # Uses the generator's declared covariance, not estimated future data.
        result[name] = assets @ weights / np.sqrt(weights @ covariance @ weights)
    return pd.DataFrame(result, index=dates)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, help="Optional directory for PNG figures."
    )
    parser.add_argument(
        "--no-show", action="store_true", help="Use a headless plotting backend."
    )
    parser.add_argument("--prior-mean", type=float, default=0.0)
    parser.add_argument("--prior-sd", type=float, default=1.0)
    parser.add_argument(
        "--checks",
        action="store_true",
        help="Print predictive checks and sensitivity to priors/calendar partitions.",
    )
    parser.add_argument(
        "--method", choices=("auto", "quadrature", "grid"), default="auto"
    )
    args = parser.parse_args()

    import matplotlib

    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from macrosynergy.pnl import BayesianPnL
    from macrosynergy.pnl.bayes_pnl import (
        IndependentPrior,
        NormalPrior,
        SDPrior,
        prior_predictive_sample,
        posterior_predictive_check,
    )
    from macrosynergy.visuals import (
        plot_bayesian_posterior,
        plot_bayesian_predictive,
        plot_bayesian_prior_posterior,
    )

    pnl = synthetic_pnls()
    prior = IndependentPrior(
        NormalPrior(args.prior_mean, args.prior_sd), SDPrior(1.0, 1.5)
    )
    fits = {
        name: BayesianPnL.from_pnl(pnl[name], prior=prior, method=args.method)
        for name in pnl
    }
    baseline = fits["baseline"]
    print(
        "Daily metrics: annual return = 252 * mean; volatility = sqrt(252) * sample sd."
    )
    print("Three-year blocks supply one annualised Sharpe observation each.")
    print("Numerical integration: {}.".format(baseline.method))
    print(
        "Explicit prior on expected Sharpe: N({}, {} squared).".format(
            args.prior_mean, args.prior_sd
        )
    )
    print(baseline.periods.round(3).to_string(index=False))
    print("\nParameter uncertainty and next-period outcomes:")
    print(baseline.summary().to_string())

    rows = []
    for name, fit in fits.items():
        lo, hi = fit.interval("mean")
        rows.append(
            dict(
                construction=name,
                n_periods=len(fit.periods),
                expected_sharpe=fit.expectation(),
                mean_lo90=lo,
                mean_hi90=hi,
                prob_mean_negative=fit.prob_mean_below(),
                prob_next_period_negative=fit.predictive_cdf(0)[0],
            )
        )
    print("\nSeparate construction fits, sharing the same market history:")
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print(
        "These comparisons do not estimate selection-adjusted or independent family evidence."
    )

    if args.checks:
        print("\nPrior-predictive 5%, 50%, 95% period Sharpe:")
        print(
            np.quantile(prior_predictive_sample(prior, 50000, rng=7), [0.05, 0.5, 0.95])
        )
        print(
            "\nPosterior predictive discrepancies (descriptive, not calibrated p-values):"
        )
        print(posterior_predictive_check(baseline, rng=7).round(3).to_string())
        sensitivity = []
        for width in (0.5, 1.0, 2.0):
            for end_year in (2022, 2023, 2024):
                changed_prior = IndependentPrior(
                    NormalPrior(args.prior_mean, args.prior_sd * width),
                    SDPrior(1.0, 1.5),
                )
                changed = BayesianPnL.from_pnl(
                    pnl["baseline"],
                    prior=changed_prior,
                    method=args.method,
                    end_year=end_year,
                )
                sensitivity.append(
                    dict(
                        prior_mean_sd=args.prior_sd * width,
                        end_year=end_year,
                        n_periods=len(changed.periods),
                        expected_sharpe=changed.expectation(),
                        prob_mean_negative=changed.prob_mean_below(),
                    )
                )
        print(
            "\nPrior/partition sensitivity (anchors can change retained sample size):"
        )
        print(pd.DataFrame(sensitivity).round(3).to_string(index=False))

    figures = {
        "posterior": plot_bayesian_posterior(baseline),
        "predictive": plot_bayesian_predictive(baseline).figure,
        "prior_posterior": plot_bayesian_prior_posterior(baseline),
    }
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for name, figure in figures.items():
            figure.savefig(
                args.output_dir / (name + ".png"), dpi=120, bbox_inches="tight"
            )
        print("\nSaved figures to {}".format(args.output_dir.resolve()))
    if not args.no_show:
        plt.show()
    for figure in figures.values():
        plt.close(figure)


if __name__ == "__main__":
    main()
