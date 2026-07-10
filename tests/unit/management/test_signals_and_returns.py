from typing import Tuple

import numpy as np
import pandas as pd
import pytest

from macrosynergy.management.simulate import SignalsAndReturnsGenerator

SEED = 123
END_DATE = "2026-01-30"


def simulate(
    signal_model: str = "expected_return",
    n_fids: int = 4,
    n_periods: int = 50,
    seed: int = SEED,
    **params: float,
) -> Tuple[SignalsAndReturnsGenerator, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    generator = SignalsAndReturnsGenerator(
        n_fids=n_fids, signal_model=signal_model, **params
    )
    signals, returns, realized_vol = generator.simulate_signals_and_returns(
        n_periods=n_periods, end_date=END_DATE, seed=seed
    )
    return generator, signals, returns, realized_vol


def standardized_innovations(
    returns: pd.DataFrame, realized_vol: pd.DataFrame
) -> np.ndarray:
    # mean_return defaults to zero, so z = returns / realized_vol.
    return returns.to_numpy() / realized_vol.to_numpy()


def max_drawdown(pnl: np.ndarray) -> float:
    cumulative = np.cumsum(pnl)
    peak = np.maximum.accumulate(cumulative)
    return float(np.max(peak - cumulative))


def max_drawdown_window(pnl: np.ndarray) -> Tuple[int, int]:
    cumulative = np.cumsum(pnl)
    drawdown = np.maximum.accumulate(cumulative) - cumulative
    trough = int(np.argmax(drawdown))
    peak = int(np.argmax(cumulative[: trough + 1]))
    return peak, trough


class TestExpectedReturnModelRegression:
    def test_default_model_reproduces_pinned_output(self):
        generator = SignalsAndReturnsGenerator(n_fids=4)
        signals, returns, realized_vol = generator.simulate_signals_and_returns(
            n_periods=50, end_date=END_DATE, seed=SEED
        )

        approx = lambda x: pytest.approx(x, rel=1e-12)
        assert signals.iloc[0, 0] == approx(0.351430081279883)
        assert signals.iloc[10, 2] == approx(-1.221408204358569)
        assert signals.to_numpy().sum() == approx(-52.75421500494619)
        assert returns.iloc[0, 0] == approx(-0.008700671848389518)
        assert returns.to_numpy().sum() == approx(0.025687155420991162)
        assert realized_vol.to_numpy().sum() == approx(2.2195642755254523)


class TestReturnsInvariance:
    def test_returns_identical_across_signal_models(self):
        _, signals_er, returns_er, vol_er = simulate("expected_return", n_periods=200)
        _, signals_pm, returns_pm, vol_pm = simulate("posterior_mean", n_periods=200)

        pd.testing.assert_frame_equal(returns_er, returns_pm)
        pd.testing.assert_frame_equal(vol_er, vol_pm)
        assert not np.allclose(signals_er.to_numpy(), signals_pm.to_numpy())


class TestPosteriorMeanConventions:
    def test_shapes_index_and_last_row(self):
        _, signals, returns, _ = simulate("posterior_mean")

        assert signals.shape == returns.shape
        assert (signals.index == returns.index).all()
        assert (signals.iloc[-1] == 0.0).all()

    def test_seed_reproducibility(self):
        _, first, _, _ = simulate("posterior_mean")
        _, second, _, _ = simulate("posterior_mean")

        pd.testing.assert_frame_equal(first, second)

    def test_signal_quality_exposed_on_generator(self):
        generator, signals, _, _ = simulate("posterior_mean")

        quality = generator.signal_quality
        assert isinstance(quality, pd.DataFrame)
        assert (quality.index == signals.index).all()
        assert quality.shape == (len(signals), 1)
        values = quality.to_numpy()
        assert np.all((values > 0.0) & (values < 1.0))

    def test_signal_quality_is_none_for_expected_return_model(self):
        generator, _, _, _ = simulate("expected_return")
        assert generator.signal_quality is None

    def test_unknown_signal_model_raises(self):
        with pytest.raises(ValueError, match="signal_model"):
            simulate("not_a_model")

    @pytest.mark.parametrize("signal_ic", [0.0, -0.05, 1.0])
    def test_posterior_mean_requires_ic_in_unit_interval(self, signal_ic):
        with pytest.raises(ValueError, match="signal_ic"):
            simulate("posterior_mean", signal_ic=signal_ic)


class TestPosteriorMeanUncertaintyMechanism:
    def test_conditional_accuracy_tracks_signal_quality(self):
        generator, signals, returns, realized_vol = simulate(
            "posterior_mean", n_fids=10, n_periods=20_000
        )
        z = standardized_innovations(returns, realized_vol)

        predictors = signals.to_numpy()[:-1]
        outcomes = z[1:]
        quality = generator.signal_quality.to_numpy()[:-1, 0]

        edges = np.quantile(quality, np.linspace(0.0, 1.0, 6))
        bucket_corrs, bucket_qualities = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            mask = (quality >= lo) & (quality <= hi)
            realized = np.corrcoef(
                predictors[mask].ravel(), outcomes[mask].ravel()
            )[0, 1]
            bucket_corrs.append(realized)
            bucket_qualities.append(quality[mask].mean())

        for realized, expected in zip(bucket_corrs, bucket_qualities):
            assert realized == pytest.approx(expected, abs=0.02)
        assert bucket_corrs[-1] > bucket_corrs[0] + 0.03

    def test_signal_scale_tracks_quality(self):
        generator, signals, _, _ = simulate(
            "posterior_mean", n_fids=20, n_periods=20_000
        )

        cross_sectional_std = signals.to_numpy()[:-1].std(axis=1)
        quality = generator.signal_quality.to_numpy()[:-1, 0]

        assert np.corrcoef(cross_sectional_std, quality)[0, 1] > 0.85

    def test_signals_collapse_jointly_in_high_uncertainty_regimes(self):
        generator, signals, _, _ = simulate(
            "posterior_mean", n_fids=20, n_periods=20_000
        )
        aggregate = np.linalg.norm(signals.to_numpy()[:-1], axis=1)
        quality = generator.signal_quality.to_numpy()[:-1, 0]

        low, high = np.quantile(quality, [0.1, 0.9])
        collapse_ratio = aggregate[quality >= high].mean() / aggregate[
            quality <= low
        ].mean()

        assert aggregate.std() / aggregate.mean() > 0.4
        assert collapse_ratio > 3.0

    def test_expected_return_model_has_no_joint_collapse(self):
        _, signals, _, _ = simulate("expected_return", n_fids=20, n_periods=20_000)
        aggregate = np.linalg.norm(signals.to_numpy()[:-1], axis=1)

        assert aggregate.std() / aggregate.mean() < 0.3

    def test_median_conditional_ic_matches_target(self):
        target_ic = 0.05
        generator, _, _, _ = simulate(
            "posterior_mean", n_fids=2, n_periods=50_000, signal_ic=target_ic
        )
        median_quality = float(np.median(generator.signal_quality.to_numpy()))

        assert median_quality == pytest.approx(target_ic, rel=0.2)


class TestStandardize:
    @pytest.mark.parametrize("signal_model", ["expected_return", "posterior_mean"])
    def test_standardized_signals_have_zero_mean_unit_std(self, signal_model):
        _, signals, _, _ = simulate(signal_model, n_periods=500, standardize=True)

        informative = signals.to_numpy()[:-1]
        assert informative.mean(axis=0) == pytest.approx(np.zeros(4), abs=1e-12)
        assert informative.std(axis=0) == pytest.approx(np.ones(4), rel=1e-12)

    def test_standardize_preserves_last_row_zero(self):
        _, signals, _, _ = simulate("posterior_mean", standardize=True)
        assert (signals.iloc[-1] == 0.0).all()

    def test_standardize_is_full_sample_affine_rescale(self):
        _, raw, _, _ = simulate("posterior_mean", n_periods=500)
        _, standardized, _, _ = simulate(
            "posterior_mean", n_periods=500, standardize=True
        )

        informative = raw.to_numpy()[:-1]
        expected = (informative - informative.mean(axis=0)) / informative.std(axis=0)
        np.testing.assert_allclose(standardized.to_numpy()[:-1], expected)

    def test_standardize_leaves_returns_unchanged(self):
        _, _, returns_raw, vol_raw = simulate("posterior_mean", n_periods=200)
        _, _, returns_std, vol_std = simulate(
            "posterior_mean", n_periods=200, standardize=True
        )

        pd.testing.assert_frame_equal(returns_raw, returns_std)
        pd.testing.assert_frame_equal(vol_raw, vol_std)

    def test_standardize_preserves_joint_collapse(self):
        generator, signals, _, _ = simulate(
            "posterior_mean", n_fids=20, n_periods=20_000, standardize=True
        )

        cross_sectional_std = signals.to_numpy()[:-1].std(axis=1)
        quality = generator.signal_quality.to_numpy()[:-1, 0]

        assert np.corrcoef(cross_sectional_std, quality)[0, 1] > 0.85

    def test_standardize_requires_enough_periods(self):
        with pytest.raises(ValueError, match="standardize"):
            simulate("posterior_mean", n_periods=2, standardize=True)


class TestPosteriorMeanProperties:
    @pytest.mark.parametrize("signal_ic", [0.02, 0.1, 0.3])
    @pytest.mark.parametrize("uncertainty_persistence", [0.9, 0.99, 0.999])
    @pytest.mark.parametrize("uncertainty_vol", [0.01, 0.1, 0.3])
    def test_output_is_finite_and_well_shaped(
        self, signal_ic, uncertainty_persistence, uncertainty_vol
    ):
        generator, signals, returns, _ = simulate(
            "posterior_mean",
            n_fids=3,
            n_periods=50,
            signal_ic=signal_ic,
            uncertainty_persistence=uncertainty_persistence,
            uncertainty_vol=uncertainty_vol,
        )

        assert signals.shape == returns.shape
        assert np.isfinite(signals.to_numpy()).all()
        quality = generator.signal_quality.to_numpy()
        assert np.all((quality > 0.0) & (quality < 1.0))


DEEP_BREAKDOWN = dict(
    signal_ic=0.1,
    uncertainty_vol=0.141,
    breakdown_threshold=0.5,
    breakdown_sharpness=4.0,
    breakdown_floor=-0.8,
)


class TestBreakdownModel:
    def test_returns_identical_to_other_models(self):
        _, _, returns_pm, vol_pm = simulate("posterior_mean", n_periods=200)
        _, _, returns_bd, vol_bd = simulate("posterior_mean_breakdown", n_periods=200)

        pd.testing.assert_frame_equal(returns_pm, returns_bd)
        pd.testing.assert_frame_equal(vol_pm, vol_bd)

    def test_floor_one_reduces_to_posterior_mean(self):
        _, signals_pm, _, _ = simulate("posterior_mean", n_periods=500)
        _, signals_bd, _, _ = simulate(
            "posterior_mean_breakdown", n_periods=500, breakdown_floor=1.0
        )

        np.testing.assert_allclose(
            signals_bd.to_numpy(), signals_pm.to_numpy(), rtol=1e-9, atol=1e-12
        )

    def test_shapes_last_row_and_reproducibility(self):
        _, signals, returns, _ = simulate("posterior_mean_breakdown")
        assert signals.shape == returns.shape
        assert (signals.iloc[-1] == 0.0).all()

        _, again, _, _ = simulate("posterior_mean_breakdown")
        pd.testing.assert_frame_equal(signals, again)

    @pytest.mark.parametrize(
        "params, match",
        [
            (dict(breakdown_floor=1.5), "breakdown_floor"),
            (dict(breakdown_floor=-1.5), "breakdown_floor"),
            (dict(breakdown_sharpness=0.0), "breakdown_sharpness"),
            (dict(breakdown_sharpness=-1.0), "breakdown_sharpness"),
            (dict(signal_ic=0.0), "signal_ic"),
            (dict(uncertainty_vol=0.0), "uncertainty_vol"),
        ],
    )
    def test_validation(self, params, match):
        with pytest.raises(ValueError, match=match):
            simulate("posterior_mean_breakdown", **params)

    def test_ground_truth_exposes_perceived_and_true_quality(self):
        generator, signals, _, _ = simulate(
            "posterior_mean_breakdown", n_periods=2_000, **DEEP_BREAKDOWN
        )

        quality = generator.signal_quality
        assert list(quality.columns) == ["SIGNAL_QUALITY", "TRUE_SIGNAL_QUALITY"]
        assert (quality.index == signals.index).all()

        perceived = quality["SIGNAL_QUALITY"].to_numpy()
        true = quality["TRUE_SIGNAL_QUALITY"].to_numpy()
        assert np.all(true <= perceived + 1e-12)
        assert true.min() < 0.0  # inversions occur under a deep-breakdown config

        calm = perceived >= np.quantile(perceived, 0.9)
        assert (true[calm] / perceived[calm]).min() > 0.9

    def test_realized_conditional_ic_matches_true_quality(self):
        generator, signals, returns, realized_vol = simulate(
            "posterior_mean_breakdown", n_fids=10, n_periods=20_000, **DEEP_BREAKDOWN
        )
        z = standardized_innovations(returns, realized_vol)

        predictors = signals.to_numpy()[:-1]
        outcomes = z[1:]
        perceived = generator.signal_quality["SIGNAL_QUALITY"].to_numpy()[:-1]
        true = generator.signal_quality["TRUE_SIGNAL_QUALITY"].to_numpy()[:-1]

        edges = np.quantile(true, np.linspace(0.0, 1.0, 6))
        for lo, hi in zip(edges[:-1], edges[1:]):
            mask = (true >= lo) & (true <= hi)
            realized = np.corrcoef(
                predictors[mask].ravel(), outcomes[mask].ravel()
            )[0, 1]
            # pooled correlation over a bucket: E[q*w] / sqrt(E[q^2]),
            # since Cov(s, z | regime) = q*w and Var(s | regime) = q^2
            expected = (perceived[mask] * true[mask]).mean() / np.sqrt(
                (perceived[mask] ** 2).mean()
            )
            assert realized == pytest.approx(expected, abs=0.02)

        worst = true <= edges[1]
        realized_worst = np.corrcoef(
            predictors[worst].ravel(), outcomes[worst].ravel()
        )[0, 1]
        assert realized_worst < 0.0

    def test_signals_stay_small_while_wrong(self):
        generator, signals, _, _ = simulate(
            "posterior_mean_breakdown", n_fids=20, n_periods=20_000, **DEEP_BREAKDOWN
        )

        cross_sectional_std = signals.to_numpy()[:-1].std(axis=1)
        perceived = generator.signal_quality["SIGNAL_QUALITY"].to_numpy()[:-1]

        assert np.corrcoef(cross_sectional_std, perceived)[0, 1] > 0.85

    def test_breakdowns_cause_systematic_vol_target_drawdowns(self):
        for seed in range(5):
            generator, signals, returns, realized_vol = simulate(
                "posterior_mean_breakdown",
                n_fids=15,
                n_periods=15_000,
                seed=seed,
                signal_ic=0.1,
                uncertainty_vol=0.141,
                breakdown_threshold=0.75,
                breakdown_sharpness=6.0,
                breakdown_floor=-1.0,
            )

            positions = signals.to_numpy()[:-1]
            next_returns = returns.to_numpy()[1:]
            vol = realized_vol.to_numpy()[:-1]

            scaled = positions * vol
            portfolio_vol = np.sqrt(
                np.einsum("ti,ij,tj->t", scaled, generator.corr, scaled)
            )
            pnl_dollar = (positions * next_returns).sum(axis=1)
            pnl_vol_target = pnl_dollar / portfolio_vol

            breakdown = (
                generator.signal_quality["TRUE_SIGNAL_QUALITY"].to_numpy()[:-1] < 0.0
            )
            assert breakdown.any()

            # (a) systematic loss at full risk while the model is inverted
            assert pnl_vol_target[breakdown].mean() < 0.0

            # (b) the max-drawdown window overlaps breakdowns disproportionately
            peak, trough = max_drawdown_window(pnl_vol_target)
            assert breakdown[peak : trough + 1].mean() > breakdown.mean()

            # (c) vol-normalized drawdown exceeds dollar sizing in every seed
            dd_vol_target = max_drawdown(pnl_vol_target / pnl_vol_target.std())
            dd_dollar = max_drawdown(pnl_dollar / pnl_dollar.std())
            assert dd_vol_target > dd_dollar


class TestBreakdownModelProperties:
    @pytest.mark.parametrize("signal_ic", [0.02, 0.1, 0.3])
    @pytest.mark.parametrize("breakdown_threshold", [0.5, 1.0, 2.0])
    @pytest.mark.parametrize("breakdown_sharpness", [0.5, 2.0, 8.0])
    @pytest.mark.parametrize("breakdown_floor", [-1.0, -0.5, 0.0, 1.0])
    def test_output_is_finite_and_validity_bounded(
        self, signal_ic, breakdown_threshold, breakdown_sharpness, breakdown_floor
    ):
        generator, signals, returns, _ = simulate(
            "posterior_mean_breakdown",
            n_fids=3,
            n_periods=50,
            signal_ic=signal_ic,
            breakdown_threshold=breakdown_threshold,
            breakdown_sharpness=breakdown_sharpness,
            breakdown_floor=breakdown_floor,
        )

        assert signals.shape == returns.shape
        assert np.isfinite(signals.to_numpy()).all()

        quality = generator.signal_quality
        validity = (
            quality["TRUE_SIGNAL_QUALITY"] / quality["SIGNAL_QUALITY"]
        ).to_numpy()
        assert np.all(validity >= breakdown_floor - 1e-12)
        assert np.all(validity <= 1.0 + 1e-12)


class TestVolTargetingAcceptance:
    def test_vol_targeted_drawdowns_exceed_dollar_per_signal(self):
        drawdowns_vol_target, drawdowns_dollar = [], []
        for seed in range(5):
            generator, signals, returns, realized_vol = simulate(
                "posterior_mean", n_fids=10, n_periods=3_000, seed=seed
            )

            positions = signals.to_numpy()[:-1]
            next_returns = returns.to_numpy()[1:]
            vol = realized_vol.to_numpy()[:-1]

            scaled = positions * vol
            portfolio_vol = np.sqrt(
                np.einsum("ti,ij,tj->t", scaled, generator.corr, scaled)
            )

            pnl_dollar = (positions * next_returns).sum(axis=1)
            pnl_vol_target = pnl_dollar / portfolio_vol

            drawdowns_dollar.append(max_drawdown(pnl_dollar / pnl_dollar.std()))
            drawdowns_vol_target.append(
                max_drawdown(pnl_vol_target / pnl_vol_target.std())
            )

        assert np.mean(drawdowns_vol_target) > 1.2 * np.mean(drawdowns_dollar)
