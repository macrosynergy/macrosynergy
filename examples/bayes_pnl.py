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
Related constructions are also fitted jointly, retaining one vector per period.
Their correlation is an explicit sensitivity assumption, not inferred precisely
from seven observations. Research-choice weights select one implementation;
they are not portfolio allocations. A fourth chart projects arithmetic PnL at
three-year endpoints using a separate joint annual-return fit.
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
    parser.add_argument(
        "--variant-rho",
        type=float,
        default=0.8,
        help="Assumed common correlation of period outcomes; not an estimate.",
    )
    parser.add_argument(
        "--paths", type=int, help="Paths; default 20000 (2000 for daily resolution)."
    )
    parser.add_argument("--future-periods", type=int, default=5)
    parser.add_argument(
        "--funnel-frequency", choices=("period", "monthly", "daily"), default="monthly"
    )
    parser.add_argument(
        "--funnel-vol",
        type=float,
        default=10.0,
        help="Explicit annual PnL volatility for the bridge approximation, in input units.",
    )
    args = parser.parse_args()
    if args.paths is None:
        args.paths = 2000 if args.funnel_frequency == "daily" else 20000
    if not 0 <= args.variant_rho < 1:
        parser.error("--variant-rho must be at least zero and below one.")
    if args.paths < 1 or args.future_periods < 1:
        parser.error("--paths and --future-periods must be positive.")

    import matplotlib

    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from macrosynergy.pnl import BayesianPnL, BayesianPnLVariants, BayesianPnLMixture
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
        plot_bayesian_funnel,
        plot_bayesian_variants,
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

    # Research choices fixed before looking at performance. The likelihood does
    # not use these weights: it uses each aligned period vector once.
    research_weights = {"baseline": 0.5, "equal_weight": 0.25, "concentrated": 0.25}
    separate_mixture = BayesianPnLMixture(fits, weights=research_weights)
    print("\nDeclared mixture of separate fits (no cross-strategy learning):")
    print(
        "P(no long-term value) = {:.1%}; P(negative next period) = {:.1%}.".format(
            separate_mixture.prob_mean_below(), separate_mixture.predictive_cdf(0)[0]
        )
    )

    def period_correlation(rho):
        return (1 - rho) * np.eye(len(pnl.columns)) + rho * np.ones(
            (len(pnl.columns), len(pnl.columns))
        )

    joint = BayesianPnLVariants.from_pnl(
        pnl,
        correlation=period_correlation(args.variant_rho),
        weights=research_weights,
        prior=prior,
        tau_scale=0.5,
    )
    print("\nJoint three-year Sharpe model: one market-period vector per row.")
    print(
        "{} shared periods, {} distinct implementations.".format(
            joint.n_periods, joint.n_variants
        )
    )
    print("Declared research-choice weights: {}".format(research_weights))
    print("Assumed shared period correlation: {:.2f}.".format(args.variant_rho))
    print(
        "'draw' selects one implementation; 'family' averages metrics, not invested PnLs."
    )
    print(joint.summary().to_string())
    correlation_sensitivity = []
    for rho in sorted(set([0.5, args.variant_rho, 0.95])):
        changed = (
            joint
            if rho == args.variant_rho
            else BayesianPnLVariants.from_pnl(
                pnl,
                correlation=period_correlation(rho),
                weights=research_weights,
                prior=prior,
                tau_scale=0.5,
            )
        )
        correlation_sensitivity.append(
            dict(
                assumed_correlation=rho,
                prob_no_long_term_value=changed.prob_mean_below(),
                prob_negative_next_period=changed.predictive_cdf(0)[0],
            )
        )
    print("\nSensitivity to assumed period correlation (covariance is not known):")
    print(pd.DataFrame(correlation_sensitivity).round(3).to_string(index=False))

    # Returns have their own fitted likelihood. No conversion from a sampled
    # Sharpe into PnL, or daily interpolation of a three-year observation.
    joint_returns = BayesianPnLVariants.from_pnl(
        pnl,
        metric="ann_ret",
        correlation=period_correlation(args.variant_rho),
        weights=research_weights,
        prior=IndependentPrior(
            NormalPrior(10 * args.prior_mean, 10 * args.prior_sd), SDPrior(10, 15)
        ),
        tau_scale=5.0,
    )
    print(
        "\nThe funnel selects one implementation per full simulated path. "
        "This benchmark assumes conditionally independent future three-year "
        "blocks. Monthly/daily curves are an explicitly assumed Brownian bridge, "
        "preserving those forecasts, not estimated daily dynamics."
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
        "funnel": plot_bayesian_funnel(
            joint_returns,
            n_periods=args.future_periods,
            n_paths=args.paths,
            rng=7,
            title="A plausible implementation: cumulative PnL funnel",
            frequency=args.funnel_frequency,
            annual_volatility=args.funnel_vol,
            baseline_name="baseline",
        ),
        "variants": plot_bayesian_variants(
            joint, baseline=baseline, baseline_name="baseline"
        ),
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
