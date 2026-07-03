import numpy as np
import pandas as pd
import pytest

from macrosynergy.management.simulate import SignalsAndReturnsGenerator

CORR_2X2 = np.array([[1.0, 0.5], [0.5, 1.0]])
SIGNAL_NAMES = ["USD_SIG", "EUR_SIG"]
RETURN_NAMES = ["USD_XR", "EUR_XR"]
END_DATE = "2025-12-31"


def make_generator(**overrides) -> SignalsAndReturnsGenerator:
    params = dict(n_fids=2, corr=CORR_2X2, base_vol=np.array([0.01, 0.02]))
    params.update(overrides)
    return SignalsAndReturnsGenerator(**params)


def simulate(generator: SignalsAndReturnsGenerator, n_periods: int = 300, **overrides):
    params = dict(
        n_periods=n_periods,
        end_date=END_DATE,
        signal_names=SIGNAL_NAMES,
        return_names=RETURN_NAMES,
        seed=7,
    )
    params.update(overrides)
    return generator.simulate_signals_and_returns(**params)


class TestSimulateSignalsAndReturns:
    def test_returns_three_frames_with_expected_shape(self) -> None:
        signals, returns, realized_vol = simulate(make_generator(), n_periods=250)

        assert signals.shape == (250, 2)
        assert returns.shape == (250, 2)
        assert realized_vol.shape == (250, 2)

    def test_frames_share_business_day_index_ending_at_end_date(self) -> None:
        signals, returns, realized_vol = simulate(make_generator())

        assert isinstance(signals.index, pd.DatetimeIndex)
        assert signals.index[-1] == pd.Timestamp(END_DATE)
        assert signals.index.is_monotonic_increasing
        assert (signals.index.dayofweek < 5).all()
        assert signals.index.equals(returns.index)
        assert signals.index.equals(realized_vol.index)

    def test_custom_column_names_are_used(self) -> None:
        signals, returns, realized_vol = simulate(make_generator())

        assert list(signals.columns) == SIGNAL_NAMES
        assert list(returns.columns) == RETURN_NAMES
        assert list(realized_vol.columns) == RETURN_NAMES

    def test_same_seed_reproduces_identical_data(self) -> None:
        signals_a, returns_a, _ = simulate(make_generator(), seed=11)
        signals_b, returns_b, _ = simulate(make_generator(), seed=11)

        pd.testing.assert_frame_equal(signals_a, signals_b)
        pd.testing.assert_frame_equal(returns_a, returns_b)

    def test_different_seeds_produce_different_data(self) -> None:
        _, returns_a, _ = simulate(make_generator(), seed=11)
        _, returns_b, _ = simulate(make_generator(), seed=12)

        assert not returns_a.equals(returns_b)

    def test_results_are_stored_on_the_generator(self) -> None:
        generator = make_generator()
        signals, returns, realized_vol = simulate(generator)

        pd.testing.assert_frame_equal(generator.signals, signals)
        pd.testing.assert_frame_equal(generator.returns, returns)
        pd.testing.assert_frame_equal(generator.realized_vol, realized_vol)

    def test_wrong_shape_corr_raises(self) -> None:
        generator = make_generator(n_fids=3)  # 3 fids but 2x2 corr

        with pytest.raises(ValueError, match="corr"):
            simulate(generator, signal_names=None, return_names=None)

    def test_corr_accepts_nested_lists(self) -> None:
        generator = make_generator(corr=[[1.0, 0.5], [0.5, 1.0]])

        signals, _, _ = simulate(generator)

        assert signals.shape == (300, 2)

    def test_last_signal_row_is_zero(self) -> None:
        signals, _, _ = simulate(make_generator())

        assert (signals.iloc[-1] == 0.0).all()

    def test_signals_predict_next_period_returns_at_target_ic(self) -> None:
        target_ic = 0.2
        generator = make_generator(signal_ic=target_ic, vol_of_vol=0.0)
        signals, returns, _ = simulate(generator, n_periods=20_000)

        for signal_name, return_name in zip(SIGNAL_NAMES, RETURN_NAMES):
            realized_ic = np.corrcoef(
                signals[signal_name].to_numpy()[:-1],
                returns[return_name].to_numpy()[1:],
            )[0, 1]
            assert realized_ic == pytest.approx(target_ic, abs=0.03)

    def test_signals_have_target_autocorrelation(self) -> None:
        target_autocorr = 0.9
        generator = make_generator(signal_ic=0.0, signal_autocorr=target_autocorr)
        signals, _, _ = simulate(generator, n_periods=20_000)

        values = signals[SIGNAL_NAMES[0]].to_numpy()[:-1]  # last row is padding
        realized = np.corrcoef(values[:-1], values[1:])[0, 1]
        assert realized == pytest.approx(target_autocorr, abs=0.03)

    def test_returns_volatility_matches_base_vol_when_vol_is_constant(self) -> None:
        base_vol = np.array([0.01, 0.02])
        generator = make_generator(base_vol=base_vol, vol_of_vol=0.0)
        _, returns, realized_vol = simulate(generator, n_periods=20_000)

        np.testing.assert_allclose(
            realized_vol.to_numpy(), np.broadcast_to(base_vol, realized_vol.shape)
        )
        np.testing.assert_allclose(returns.std().to_numpy(), base_vol, rtol=0.05)

    def test_returns_correlation_matches_target_corr(self) -> None:
        generator = make_generator(vol_of_vol=0.0)
        _, returns, _ = simulate(generator, n_periods=20_000)

        realized_corr = returns.corr().to_numpy()
        np.testing.assert_allclose(realized_corr, CORR_2X2, atol=0.03)

    def test_mean_return_shifts_the_drift(self) -> None:
        drift = 0.001
        generator = make_generator(mean_return=drift, vol_of_vol=0.0)
        _, returns, _ = simulate(generator, n_periods=20_000)

        np.testing.assert_allclose(returns.mean().to_numpy(), drift, atol=5e-4)


class TestDefaultCorrelationMatrix:
    def test_default_corr_is_a_valid_correlation_matrix(self) -> None:
        generator = SignalsAndReturnsGenerator(n_fids=4)
        corr = generator.corr

        np.testing.assert_allclose(np.diag(corr), 1.0)
        np.testing.assert_allclose(corr, corr.T)
        assert (np.linalg.eigvalsh(corr) > 0).all()

    def test_default_base_vol_is_one_percent(self) -> None:
        generator = SignalsAndReturnsGenerator(n_fids=3)

        np.testing.assert_allclose(generator.base_vol, 0.01)


class TestQuantamentalConversion:
    def test_signals_convert_to_long_format(self) -> None:
        generator = make_generator()
        simulate(generator, n_periods=100)

        qdf = generator.quantamental_signals()

        assert set(qdf.columns) == {"real_date", "cid", "xcat", "value"}
        assert len(qdf) == 100 * 2
        assert set(qdf["cid"]) == {"USD", "EUR"}
        assert set(qdf["xcat"]) == {"SIG"}

    def test_returns_convert_to_long_format(self) -> None:
        generator = make_generator()
        simulate(generator, n_periods=100)

        qdf = generator.quantamental_returns()

        assert len(qdf) == 100 * 2
        assert set(qdf["xcat"]) == {"XR"}

    def test_conversion_works_with_default_names(self) -> None:
        generator = make_generator()
        simulate(generator, n_periods=100, signal_names=None, return_names=None)

        signals_qdf = generator.quantamental_signals()
        returns_qdf = generator.quantamental_returns()

        assert len(signals_qdf) == 100 * 2
        assert len(returns_qdf) == 100 * 2

    def test_combined_frame_holds_signals_and_returns(self) -> None:
        generator = make_generator()
        simulate(generator, n_periods=100)

        combined = generator.quantamental_returns_and_signals()

        assert len(combined) == 2 * 100 * 2
        assert set(combined["xcat"]) == {"SIG", "XR"}

    def test_conversion_before_simulation_raises(self) -> None:
        generator = make_generator()

        with pytest.raises(ValueError, match="simulate"):
            generator.quantamental_signals()


class TestRealizedCov:
    def test_long_format_with_upper_triangle_pairs(self) -> None:
        generator = make_generator()
        simulate(generator)

        cov = generator.realized_cov(freq="BMS")

        assert list(cov.columns) == ["fid1", "fid2", "value", "real_date"]
        pairs_per_date = cov.groupby("real_date").size()
        assert (pairs_per_date == 3).all()  # 2 fids -> 3 upper-triangle pairs

    def test_dates_are_interval_starts_within_sample(self) -> None:
        generator = make_generator()
        signals, _, _ = simulate(generator)

        cov = generator.realized_cov(freq="BMS")

        rebalance_dates = pd.DatetimeIndex(cov["real_date"].unique())
        assert rebalance_dates.min() >= signals.index[0]
        assert rebalance_dates.max() < signals.index[-1]
        assert (rebalance_dates.day == 1).all() or (rebalance_dates.dayofweek < 5).all()

    def test_matches_annualized_ground_truth_when_vol_is_constant(self) -> None:
        # annualized (x252 for business-daily simulation) so the values are in
        # the same units as the estimator VCV from `notional_positions`
        base_vol = np.array([0.01, 0.02])
        generator = make_generator(base_vol=base_vol, vol_of_vol=0.0)
        simulate(generator)

        cov = generator.realized_cov(freq="BMS")

        expected = 252 * np.outer(base_vol, base_vol) * CORR_2X2
        variance_usd = cov.loc[(cov["fid1"] == "USD_XR") & (cov["fid2"] == "USD_XR")]
        variance_eur = cov.loc[(cov["fid1"] == "EUR_XR") & (cov["fid2"] == "EUR_XR")]
        covariance = cov.loc[(cov["fid1"] != cov["fid2"])]

        np.testing.assert_allclose(variance_usd["value"], expected[0, 0])
        np.testing.assert_allclose(variance_eur["value"], expected[1, 1])
        np.testing.assert_allclose(covariance["value"], expected[0, 1])

    def test_before_simulation_raises(self) -> None:
        generator = make_generator()

        with pytest.raises(ValueError, match="simulate"):
            generator.realized_cov()
