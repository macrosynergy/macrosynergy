"""Numerical references and public input contracts for Bayesian PnL estimates."""

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import integrate, special, stats

from macrosynergy.pnl import BayesianPnL
from macrosynergy.pnl.bayes_pnl import (
    GridSpec,
    GridTruncationWarning,
    IndependentPrior,
    NormalPrior,
    SDPrior,
    default_prior,
    period_metrics,
    posterior_grid,
)

OBSERVATIONS = np.array([0.9, 0.2, 1.1, 0.4, -0.3, 0.7])


class ConjugateReferencePrior:
    """Test-local normal/inverse-gamma joint prior, expressed on the sd axis."""

    normalised = True
    mean = 0.3
    strength = 1.0
    shape = 3.0
    scale = 1.0

    def logpdf(self, mean, sd):
        return (
            stats.norm.logpdf(mean, self.mean, sd / np.sqrt(self.strength))
            + stats.invgamma.logpdf(sd**2, a=self.shape, scale=self.scale)
            + np.log(2.0 * sd)
        )


def conjugate_reference(data, weight=1.0):
    """Independent analytic posterior and predictive, including power likelihoods."""
    prior = ConjugateReferencePrior()
    n = weight * len(data)
    average = np.mean(data)
    strength = prior.strength + n
    mean = (prior.strength * prior.mean + n * average) / strength
    shape = prior.shape + n / 2.0
    scale = (
        prior.scale
        + weight * np.sum((data - average) ** 2) / 2.0
        + prior.strength * n * (average - prior.mean) ** 2 / (2.0 * strength)
    )
    mean_dist = stats.t(2 * shape, loc=mean, scale=np.sqrt(scale / (shape * strength)))
    predictive = stats.t(
        2 * shape, loc=mean, scale=np.sqrt(scale * (strength + 1) / (shape * strength))
    )
    log_evidence = (
        special.gammaln(shape)
        - special.gammaln(prior.shape)
        + prior.shape * np.log(prior.scale)
        - shape * np.log(scale)
        + (np.log(prior.strength) - np.log(strength)) / 2.0
        - n * np.log(2.0 * np.pi) / 2.0
    )
    return mean_dist, predictive, shape, scale, log_evidence


@pytest.fixture(scope="module")
def conjugate_fit():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", GridTruncationWarning)
        return BayesianPnL(
            OBSERVATIONS,
            prior=ConjugateReferencePrior(),
            grid=GridSpec((-3.0, 3.5), (0.04, 5.0), 440, 480, "log"),
            method="grid",
        )


@pytest.fixture(scope="module")
def default_fit():
    return BayesianPnL(OBSERVATIONS)


@pytest.fixture(scope="module")
def daily_pnl():
    dates = pd.bdate_range("2001-01-01", "2010-12-31")
    values = np.random.RandomState(23).normal(0.02, 0.6, len(dates))
    return pd.Series(values, index=dates, name="strategy")


def test_grid_posterior_matches_analytic_parameter_marginals(conjugate_fit):
    mean_dist, _, shape, scale, _ = conjugate_reference(OBSERVATIONS)
    post = conjugate_fit.posterior
    assert np.sum(post.mass) == pytest.approx(1.0)
    locations = np.linspace(-0.5, 1.5, 31)
    np.testing.assert_allclose(
        post.cdf("mean", locations), mean_dist.cdf(locations), atol=0.002
    )
    spreads = np.linspace(0.3, 1.5, 25)
    np.testing.assert_allclose(
        post.cdf("sd", spreads),
        stats.invgamma.cdf(spreads**2, a=shape, scale=scale),
        atol=0.003,
    )
    assert conjugate_fit.expectation("mean") == pytest.approx(
        mean_dist.mean(), abs=0.002
    )
    np.testing.assert_allclose(
        conjugate_fit.interval("mean"), mean_dist.ppf([0.05, 0.95]), atol=0.003
    )


def test_predictive_matches_analytic_student_t(conjugate_fit):
    _, predictive, _, _, _ = conjugate_reference(OBSERVATIONS)
    locations = np.linspace(-1.5, 2.5, 21)
    np.testing.assert_allclose(
        conjugate_fit.predictive_cdf(locations), predictive.cdf(locations), atol=0.002
    )
    np.testing.assert_allclose(
        conjugate_fit.predictive_pdf(locations), predictive.pdf(locations), atol=0.002
    )
    np.testing.assert_allclose(
        conjugate_fit.predictive_quantile([0.05, 0.5, 0.95]),
        predictive.ppf([0.05, 0.5, 0.95]),
        atol=0.004,
    )


def test_grid_evidence_includes_sd_jacobian_and_cell_area(conjugate_fit):
    expected = conjugate_reference(OBSERVATIONS)[-1]
    assert conjugate_fit.posterior.log_evidence == pytest.approx(expected, abs=0.002)


def test_weighted_likelihood_matches_analytic_reference():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", GridTruncationWarning)
        fit = BayesianPnL(
            OBSERVATIONS,
            prior=ConjugateReferencePrior(),
            likelihood_weight=0.5,
            grid=GridSpec((-4.0, 4.0), (0.04, 7.0), 440, 480, "log"),
            method="grid",
        )
    mean_dist, predictive, _, _, _ = conjugate_reference(OBSERVATIONS, weight=0.5)
    np.testing.assert_allclose(
        fit.interval("mean"), mean_dist.ppf([0.05, 0.95]), atol=0.004
    )
    np.testing.assert_allclose(
        fit.predictive_cdf([-0.5, 0.5, 1.5]),
        predictive.cdf([-0.5, 0.5, 1.5]),
        atol=0.003,
    )
    # A power-likelihood normalizer is not ordinary marginal model evidence.
    assert fit.posterior.log_evidence is None


@pytest.mark.parametrize("kind", ["invgamma_var", "gamma_sd", "lognormal_sd"])
def test_sd_prior_normalization_and_requested_moments(kind):
    prior = SDPrior(0.8, 0.4, kind=kind)
    moments = [
        integrate.quad(
            lambda x: x**order * np.exp(prior.logpdf(x)), 0, np.inf, epsabs=1e-8
        )[0]
        for order in range(3)
    ]
    assert moments[0] == pytest.approx(1.0, abs=1e-6)
    assert moments[1] == pytest.approx(0.8, abs=1e-6)
    assert np.sqrt(moments[2] - moments[1] ** 2) == pytest.approx(0.4, abs=1e-6)


def test_adaptive_grid_captures_small_sample_mass():
    fit = BayesianPnL([1.54, 0.33, 0.58], method="grid")
    assert max(fit.posterior.edge_mass().values()) < 0.001
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", GridTruncationWarning)
        reference = posterior_grid(
            [1.54, 0.33, 0.58],
            default_prior("sharpe"),
            GridSpec((-8, 8), (0.01, 20), 700, 850, "log"),
        )
    for parameter in ("mean", "sd"):
        assert fit.expectation(parameter) == pytest.approx(
            reference.expectation(parameter),
            abs=0.004,
        )


def test_too_narrow_grid_warns():
    with pytest.warns(GridTruncationWarning):
        BayesianPnL(
            OBSERVATIONS,
            grid=GridSpec((0.44, 0.46), (0.48, 0.52), 40, 40),
            method="grid",
        )


def test_predictive_samples_and_quantiles_agree(default_fit):
    draws = default_fit.predictive_sample(50000, rng=41)
    assert draws.shape == (50000,)
    np.testing.assert_array_equal(
        draws[:20], default_fit.predictive_sample(50000, rng=41)[:20]
    )
    for threshold in (-0.5, 0.0, 0.5, 1.5):
        assert np.mean(draws < threshold) == pytest.approx(
            default_fit.predictive_cdf(threshold)[0],
            abs=0.01,
        )
    probabilities = [0.1, 0.5, 0.9]
    quantiles = default_fit.predictive_quantile(probabilities)
    np.testing.assert_allclose(
        default_fit.predictive_cdf(quantiles), probabilities, atol=1e-7
    )


def test_wrapper_summary_and_parameter_uncertainty(default_fit):
    summary = default_fit.summary()
    assert isinstance(summary, pd.DataFrame)
    assert default_fit.periods is None
    assert default_fit.prob_mean_below(0) == pytest.approx(
        default_fit.posterior.cdf("mean", 0),
    )
    # The posterior mean's sign and the next realised outcome are distinct targets.
    assert default_fit.predictive_cdf(0)[0] > default_fit.prob_mean_below(0)
    eti = default_fit.interval("sd", level=0.9, kind="eti")
    hdi = default_fit.interval("sd", level=0.9, kind="hdi")
    assert hdi[1] - hdi[0] <= eti[1] - eti[0]


@pytest.mark.parametrize("metric", ["sharpe", "ann_ret", "ann_vol"])
def test_public_constructor_supports_each_metric(metric):
    values = {
        "sharpe": [0.3, 0.9, -0.2, 0.5],
        "ann_ret": [3.0, 9.0, -2.0, 5.0],
        "ann_vol": [9.5, 11.0, 10.2, 8.7],
    }[metric]
    fit = BayesianPnL(values, metric=metric, vol_target=10.0)
    assert fit.posterior.metric == metric
    assert np.isfinite(fit.expectation("mean"))
    assert np.isfinite(fit.predictive_quantile([0.05, 0.95])).all()


@pytest.mark.parametrize("metric", ["sharpe", "ann_ret"])
def test_expected_performance_prior_defaults_are_centered_at_zero(metric):
    prior = default_prior(metric, vol_target=10)
    assert prior.mean.mean == 0.0
    assert prior.mean.cdf(0) == pytest.approx(0.5)


@pytest.mark.parametrize("factor", [252, 260])
def test_daily_periods_match_direct_calculation_and_preserve_scale(daily_pnl, factor):
    periods = period_metrics(daily_pnl, 36, annualization_factor=factor)
    assert len(periods) == 3
    assert periods.period_start.min() == pd.Timestamp("2002-01-31")
    assert periods.period_end.max() == pd.Timestamp("2010-12-31")
    block = daily_pnl.loc["2008-01-01":]
    final = periods.iloc[-1]
    assert final.ann_ret == pytest.approx(factor * block.mean())
    assert final.ann_vol == pytest.approx(np.sqrt(factor) * block.std(ddof=1))
    assert final.sharpe == pytest.approx(
        np.sqrt(factor) * block.mean() / block.std(ddof=1)
    )
    scaled = period_metrics(10 * daily_pnl, 36, annualization_factor=factor)
    np.testing.assert_allclose(scaled.sharpe, periods.sharpe)
    np.testing.assert_allclose(scaled.ann_vol, 10 * periods.ann_vol)


def test_from_pnl_retains_periods_and_matches_vector_fit(daily_pnl):
    prior = IndependentPrior(NormalPrior(0.0, 0.5), SDPrior(1.0, 1.5))
    fit = BayesianPnL.from_pnl(daily_pnl, prior=prior)
    direct = BayesianPnL(fit.periods.sharpe.to_numpy(), prior=prior)
    assert len(fit.periods) == 3
    assert fit.posterior.period_years == 3.0
    np.testing.assert_allclose(
        fit.predictive_quantile([0.1, 0.9]), direct.predictive_quantile([0.1, 0.9])
    )
    frame_fit = BayesianPnL.from_pnl(daily_pnl.to_frame(), prior=prior)
    assert frame_fit.expectation("mean") == pytest.approx(fit.expectation("mean"))


def test_from_pnl_custom_annualization_and_period_length(daily_pnl):
    fit = BayesianPnL.from_pnl(
        daily_pnl, period_months=24, metric="ann_ret", annualization_factor=260
    )
    assert fit.posterior.period_years == 2.0
    last = daily_pnl.loc["2009-01-01":]
    assert fit.periods.iloc[-1].ann_ret == pytest.approx(260 * last.mean())


def test_explicit_nan_invalidates_whole_daily_block(daily_pnl):
    broken = daily_pnl.copy()
    broken.loc["2009-04-01"] = np.nan
    with pytest.warns(UserWarning):
        periods = period_metrics(broken)
    assert len(periods) == 2
    assert periods.period_end.max() == pd.Timestamp("2007-12-31")


def test_wholly_absent_daily_month_invalidates_calendar_block(daily_pnl):
    missing = daily_pnl.loc[daily_pnl.index.to_period("M") != pd.Period("2009-04", "M")]
    with pytest.warns(UserWarning):
        periods = period_metrics(missing)
    assert len(periods) == 2
    assert periods.period_start.min() == pd.Timestamp("2002-01-31")
    assert periods.period_end.max() == pd.Timestamp("2007-12-31")


@pytest.mark.parametrize(
    "bad",
    [
        [],
        0.5,
        [[0.2, 0.3]],
        [[0.2], [0.3]],
        [0.2, np.nan],
        [0.2, np.inf],
        [0.2, -np.inf],
        ["good", "bad"],
    ],
)
def test_numeric_vector_rejects_invalid_data(bad):
    with pytest.raises((TypeError, ValueError)):
        BayesianPnL(bad)


@pytest.mark.parametrize("bad", [[0.2, np.nan], [0.2, np.inf], [[0.2, 0.3]]])
def test_numeric_helper_does_not_silently_drop_invalid_values(bad):
    with pytest.raises((TypeError, ValueError)):
        posterior_grid(bad, default_prior("sharpe"))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"metric": "unknown"},
        {"period_years": 0},
        {"period_years": np.inf},
        {"likelihood_weight": 0},
        {"likelihood_weight": -1},
        {"likelihood_weight": np.nan},
        {"vol_target": -10},
    ],
)
def test_constructor_rejects_invalid_configuration(kwargs):
    with pytest.raises((TypeError, ValueError)):
        BayesianPnL(OBSERVATIONS, **kwargs)


@pytest.mark.parametrize("factor", [0, -1, np.nan, np.inf])
def test_from_pnl_rejects_invalid_daily_annualization(daily_pnl, factor):
    with pytest.raises((TypeError, ValueError)):
        BayesianPnL.from_pnl(daily_pnl, annualization_factor=factor)


@pytest.mark.parametrize("legacy_option", [{"freq_in": "M"}, {"metric_freq": "M"}])
def test_daily_only_api_rejects_removed_frequency_options(daily_pnl, legacy_option):
    with pytest.raises(TypeError):
        BayesianPnL.from_pnl(daily_pnl, **legacy_option)
    with pytest.raises(TypeError):
        period_metrics(daily_pnl, **legacy_option)


@pytest.mark.parametrize("frequency", ["MS", "W"])
def test_daily_contract_rejects_coarser_observation_streams(frequency):
    dates = pd.date_range("2001-01-01", "2006-12-31", freq=frequency)
    values = np.random.RandomState(17).normal(0.1, 0.5, len(dates))
    coarse = pd.Series(values, index=dates, name="strategy")
    # Dated values alone do not establish daily frequency. Reject before
    # multiplying monthly or weekly observations by a daily annualisation factor.
    with pytest.raises(ValueError):
        period_metrics(coarse)
    with pytest.raises(ValueError):
        BayesianPnL.from_pnl(coarse)


def test_from_pnl_rejects_multiple_strategies(daily_pnl):
    with pytest.raises((TypeError, ValueError)):
        BayesianPnL.from_pnl(
            pd.concat([daily_pnl.rename("one"), daily_pnl.rename("two")], axis=1)
        )


def test_pnl_rejects_duplicate_daily_timestamps(daily_pnl):
    duplicate = pd.concat([daily_pnl, daily_pnl.iloc[[-1]]])
    with pytest.raises((TypeError, ValueError)):
        BayesianPnL.from_pnl(duplicate)


def test_pnl_requires_dates_and_enough_history():
    with pytest.raises((TypeError, ValueError)):
        BayesianPnL.from_pnl(pd.Series([0.1, 0.2]))
    short = pd.Series([0.1, 0.2], index=pd.bdate_range("2001-01-01", periods=2))
    with pytest.raises((TypeError, ValueError)):
        BayesianPnL.from_pnl(short)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mean_range": (1, -1), "sd_range": (0.1, 2)},
        {"mean_range": (-1, 1), "sd_range": (0, 2)},
        {"mean_range": (-1, 1), "sd_range": (0.1, np.inf)},
    ],
)
def test_grid_rejects_invalid_bounds(kwargs):
    with pytest.raises((TypeError, ValueError)):
        GridSpec(**kwargs)


@pytest.mark.parametrize("bad", [0, -0.1, np.nan, np.inf])
def test_priors_require_finite_positive_sd(bad):
    with pytest.raises((TypeError, ValueError)):
        NormalPrior(0, bad)
    with pytest.raises((TypeError, ValueError)):
        SDPrior(1, bad)


def test_first_release_predictions_do_not_accept_horizon_scaling(default_fit):
    with pytest.raises(TypeError):
        default_fit.predictive_cdf(0, horizon_years=1)


@pytest.mark.parametrize("probabilities", [-0.01, 1.01, np.nan, [0.1, np.inf]])
def test_quantiles_reject_invalid_probabilities(default_fit, probabilities):
    with pytest.raises(ValueError):
        default_fit.predictive_quantile(probabilities)
    with pytest.raises(ValueError):
        default_fit.posterior.quantile("mean", probabilities)


def test_predictive_quantile_endpoints_include_normal_tails(default_fit):
    quantiles = default_fit.predictive_quantile([0, 1])
    np.testing.assert_array_equal(quantiles, [-np.inf, np.inf])
    np.testing.assert_array_equal(default_fit.predictive_cdf(quantiles), [0, 1])


@pytest.mark.parametrize("level", [0, 1, -0.1, np.nan])
def test_intervals_reject_invalid_credible_mass(default_fit, level):
    with pytest.raises(ValueError):
        default_fit.interval("mean", level=level)


def test_interval_rejects_unknown_definition(default_fit):
    with pytest.raises(ValueError):
        default_fit.interval("mean", kind="unknown")


@pytest.mark.parametrize(
    "kwargs", [{"n_sd": 1}, {"n_sd": 2.5}, {"n_mean": 0}, {"sd_spacing": "unknown"}]
)
def test_grid_requires_resolvable_axes(kwargs):
    with pytest.raises((TypeError, ValueError)):
        GridSpec((-1, 1), (0.1, 2), **kwargs)


def test_proper_prior_outside_grid_support_has_no_posterior():
    class CompactPrior:
        normalised = True

        def logpdf(self, mean, sd):
            return stats.uniform.logpdf(mean, loc=-1, scale=2) + stats.uniform.logpdf(
                sd,
                loc=5,
                scale=1,
            )

    with pytest.raises(ValueError, match="finite posterior mass"):
        BayesianPnL(
            OBSERVATIONS,
            prior=CompactPrior(),
            grid=GridSpec((-1, 1), (0.1, 2), 32, 32),
            method="grid",
        )


def test_single_observation_is_allowed_with_proper_regularizing_prior():
    prior = IndependentPrior(NormalPrior(0, 0.5), SDPrior(0.6, 0.2))
    fit = BayesianPnL([0.4], prior=prior)
    assert np.isfinite(fit.expectation("mean"))
    assert fit.expectation("sd") > 0
    lo, hi = fit.predictive_quantile([0.05, 0.95])
    assert np.isfinite([lo, hi]).all() and lo < hi


def test_complex_metrics_and_pnl_are_rejected(daily_pnl):
    with pytest.raises((TypeError, ValueError)):
        BayesianPnL([0.1, 0.2 + 0.1j])
    with pytest.raises((TypeError, ValueError)):
        BayesianPnL.from_pnl(daily_pnl.astype(complex) + 0.1j)


def test_all_missing_history_cannot_be_fitted(daily_pnl):
    with pytest.raises(ValueError):
        BayesianPnL.from_pnl(daily_pnl * np.nan)


def test_zero_volatility_blocks_have_no_defined_sharpe(daily_pnl):
    constant = daily_pnl * 0
    periods = period_metrics(constant)
    assert (periods.ann_vol == 0).all()
    assert periods.sharpe.isna().all()
    with pytest.raises(ValueError, match="undefined"):
        BayesianPnL.from_pnl(constant)


@pytest.mark.parametrize("scale", [0.01, 1e-6, 1000.0])
def test_return_inference_and_prediction_follow_input_units(scale):
    values = np.array([3.0, 9.0, -2.0, 5.0])
    reference = BayesianPnL(values, metric="ann_ret", vol_target=10)
    scaled = BayesianPnL(values * scale, metric="ann_ret", vol_target=10 * scale)
    assert scaled.expectation("mean") / scale == pytest.approx(
        reference.expectation("mean"), rel=1e-5
    )
    np.testing.assert_allclose(
        scaled.interval("mean") / np.array([scale, scale]),
        reference.interval("mean"),
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        scaled.predictive_quantile([0.05, 0.95]) / scale,
        reference.predictive_quantile([0.05, 0.95]),
        rtol=1e-5,
    )


def test_fit_keeps_snapshot_of_supplied_metrics():
    data = OBSERVATIONS.copy()
    fit = BayesianPnL(data)
    data[:] = 100.0
    np.testing.assert_array_equal(fit.posterior.data, OBSERVATIONS)


def test_intraday_observations_are_not_annualized_as_daily(daily_pnl):
    extra = pd.Series([0.1], index=[daily_pnl.index[-1] + pd.Timedelta(hours=1)])
    with pytest.raises(ValueError, match="one observation per calendar day"):
        BayesianPnL.from_pnl(pd.concat([daily_pnl, extra]))
