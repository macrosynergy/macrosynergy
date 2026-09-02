import unittest
from typing import List

import numpy as np
import pandas as pd

from macrosynergy.management.constants import ANNUALIZATION_FACTORS
from macrosynergy.management.simulate.signals_and_returns import (
    SignalsAndReturnsGenerator,
    _simulate_returns,
    _simulate_signals,
    _simulate_signals_and_returns,
    _simulate_volatility,
)
from macrosynergy.management.types import QuantamentalDataFrame

# A known, well conditioned correlation matrix used throughout.
CORR_3: np.ndarray = np.array(
    [
        [1.0, 0.3, 0.0],
        [0.3, 1.0, -0.2],
        [0.0, -0.2, 1.0],
    ]
)
BASE_VOL_3: np.ndarray = np.array([0.01, 0.02, 0.05])

# Sample size for the statistical tests
N_LONG: int = 60_000


def _innovations(
    n_periods: int,
    corr: np.ndarray,
    seed: int = 7,
    base_vol: np.ndarray = None,
    vol_of_vol: float = 0.0,
) -> np.ndarray:
    """
    Correlated unit-variance innovations `z`, as consumed by `_simulate_signals`.
    """

    n_fids = corr.shape[0]
    base_vol = np.full(n_fids, 0.01) if base_vol is None else base_vol
    rng = np.random.default_rng(seed)
    realized_vol = _simulate_volatility(
        n_periods=n_periods,
        n_fids=n_fids,
        base_vol=base_vol,
        vol_persistence=0.94,
        vol_of_vol=vol_of_vol,
        rng=rng,
    )
    _, z = _simulate_returns(
        n_periods=n_periods,
        n_fids=n_fids,
        corr=corr,
        mean_return=np.zeros(n_fids),
        realized_vol=realized_vol,
        rng=rng,
        return_z=True,
    )
    return z


def _lag1_autocorr(arr: np.ndarray) -> np.ndarray:
    """
    Lag-1 autocorrelation of each column of a 2D array.
    """
    return np.array(
        [np.corrcoef(arr[:-1, k], arr[1:, k])[0, 1] for k in range(arr.shape[1])]
    )


def _lead_correlation(signals: np.ndarray, z: np.ndarray) -> np.ndarray:
    """
    Realized IC: correlation of each signal with the *next* period's innovation.
    """
    return np.array(
        [np.corrcoef(signals[:-1, k], z[1:, k])[0, 1] for k in range(signals.shape[1])]
    )


class TestSimulateVolatility(unittest.TestCase):
    def test_shape_and_positivity(self):
        vol = _simulate_volatility(
            n_periods=100,
            n_fids=3,
            base_vol=BASE_VOL_3,
            vol_persistence=0.94,
            vol_of_vol=0.15,
            rng=np.random.default_rng(1),
        )

        self.assertIsInstance(vol, np.ndarray)
        self.assertEqual(vol.shape, (100, 3))
        self.assertTrue(np.all(vol > 0.0))
        self.assertTrue(np.all(np.isfinite(vol)))

    def test_zero_vol_of_vol_is_constant_base_vol(self):
        """
        Test that without shocks the log-variance stays pinned at its long-run level,
        so every period must return exactly `base_vol`.
        """
        vol = _simulate_volatility(
            n_periods=50,
            n_fids=3,
            base_vol=BASE_VOL_3,
            vol_persistence=0.94,
            vol_of_vol=0.0,
            rng=np.random.default_rng(1),
        )

        np.testing.assert_allclose(vol, np.tile(BASE_VOL_3, (50, 1)))

    def test_long_run_level(self):
        # The log-variance is a stationary AR(1) centred on log(base_vol ** 2),
        # so its sample mean recovers the requested base volatility.
        vol = _simulate_volatility(
            n_periods=20_000,
            n_fids=3,
            base_vol=BASE_VOL_3,
            vol_persistence=0.94,
            vol_of_vol=0.15,
            rng=np.random.default_rng(3),
        )

        log_var = np.log(vol**2)
        np.testing.assert_allclose(
            log_var.mean(axis=0), np.log(BASE_VOL_3**2), atol=0.1
        )

    def test_persistence_sets_autocorrelation(self):
        # Lag-1 autocorrelation of an AR(1) equals its coefficient.
        for persistence in (0.5, 0.94):
            with self.subTest(vol_persistence=persistence):
                vol = _simulate_volatility(
                    n_periods=20_000,
                    n_fids=3,
                    base_vol=BASE_VOL_3,
                    vol_persistence=persistence,
                    vol_of_vol=0.15,
                    rng=np.random.default_rng(3),
                )

                autocorr = _lag1_autocorr(np.log(vol**2))
                np.testing.assert_allclose(autocorr, persistence, atol=0.03)

    def test_zero_persistence_is_iid(self):
        """
        Test that with no persistence the log-variance is
        iid around its long-run level, with standard deviation `vol_of_vol`.
        """
        vol_of_vol = 0.2
        vol = _simulate_volatility(
            n_periods=20_000,
            n_fids=3,
            base_vol=BASE_VOL_3,
            vol_persistence=0.0,
            vol_of_vol=vol_of_vol,
            rng=np.random.default_rng(3),
        )

        log_var = np.log(vol**2)
        np.testing.assert_allclose(_lag1_autocorr(log_var), 0.0, atol=0.03)
        np.testing.assert_allclose(log_var.std(axis=0), vol_of_vol, rtol=0.05)

    def test_contracts_are_independent(self):
        """
        Test contracts are independent of each other. Each contract gets its
        own shock, so the volatility paths should not be cross-sectionally correlated.
        """
        vol = _simulate_volatility(
            n_periods=20_000,
            n_fids=3,
            base_vol=BASE_VOL_3,
            vol_persistence=0.94,
            vol_of_vol=0.15,
            rng=np.random.default_rng(3),
        )

        corr = np.corrcoef(np.log(vol**2).T)
        off_diagonal = corr[~np.eye(3, dtype=bool)]
        np.testing.assert_allclose(off_diagonal, 0.0, atol=0.15)


class TestSimulateReturns(unittest.TestCase):
    def setUp(self):
        self.realized_vol = np.tile(BASE_VOL_3, (500, 1))

    def test_decomposition_is_exact(self):
        mean_return = np.array([0.001, -0.002, 0.0])
        realized_vol = _simulate_volatility(
            n_periods=500,
            n_fids=3,
            base_vol=BASE_VOL_3,
            vol_persistence=0.94,
            vol_of_vol=0.15,
            rng=np.random.default_rng(2),
        )
        returns, z = _simulate_returns(
            n_periods=500,
            n_fids=3,
            corr=CORR_3,
            mean_return=mean_return,
            realized_vol=realized_vol,
            rng=np.random.default_rng(4),
            return_z=True,
        )

        np.testing.assert_allclose(returns, mean_return + realized_vol * z)

    def test_zero_volatility_gives_mean_return(self):
        mean_return = np.array([0.001, -0.002, 0.0])
        returns = _simulate_returns(
            n_periods=25,
            n_fids=3,
            corr=CORR_3,
            mean_return=mean_return,
            realized_vol=np.zeros((25, 3)),
            rng=np.random.default_rng(1),
        )

        np.testing.assert_allclose(returns, np.tile(mean_return, (25, 1)))

    def test_innovations_match_corr_and_unit_variance(self):
        _, z = _simulate_returns(
            n_periods=N_LONG,
            n_fids=3,
            corr=CORR_3,
            mean_return=np.zeros(3),
            realized_vol=np.tile(BASE_VOL_3, (N_LONG, 1)),
            rng=np.random.default_rng(9),
            return_z=True,
        )

        np.testing.assert_allclose(np.corrcoef(z.T), CORR_3, atol=0.02)
        np.testing.assert_allclose(z.std(axis=0), 1.0, rtol=0.02)
        np.testing.assert_allclose(z.mean(axis=0), 0.0, atol=0.02)

    def test_return_moments_match_vol_and_corr(self):
        mean_return = np.array([0.001, -0.002, 0.0])
        returns = _simulate_returns(
            n_periods=N_LONG,
            n_fids=3,
            corr=CORR_3,
            mean_return=mean_return,
            realized_vol=np.tile(BASE_VOL_3, (N_LONG, 1)),
            rng=np.random.default_rng(9),
        )

        np.testing.assert_allclose(returns.std(axis=0), BASE_VOL_3, rtol=0.02)
        np.testing.assert_allclose(np.corrcoef(returns.T), CORR_3, atol=0.02)
        np.testing.assert_allclose(
            returns.mean(axis=0), mean_return, atol=BASE_VOL_3.max() * 0.05
        )

    def test_non_positive_definite_corr_raises(self):
        with self.assertRaises(np.linalg.LinAlgError):
            _simulate_returns(
                n_periods=10,
                n_fids=2,
                corr=np.array([[1.0, 1.5], [1.5, 1.0]]),
                mean_return=np.zeros(2),
                realized_vol=np.full((10, 2), 0.01),
                rng=np.random.default_rng(1),
            )


class TestSimulateSignals(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.z = _innovations(N_LONG, CORR_3, seed=7)

    def _signals(self, signal_ic: float, signal_autocorr: float = 0.9, seed: int = 3):
        return _simulate_signals(
            n_periods=self.z.shape[0],
            n_fids=self.z.shape[1],
            signal_ic=signal_ic,
            signal_autocorr=signal_autocorr,
            z=self.z,
            rng=np.random.default_rng(seed),
        )

    def test_shape_and_last_row_is_zero(self):
        signals = _simulate_signals(
            n_periods=100,
            n_fids=3,
            signal_ic=0.05,
            signal_autocorr=0.9,
            z=_innovations(100, CORR_3),
            rng=np.random.default_rng(3),
        )

        self.assertEqual(signals.shape, (100, 3))
        self.assertTrue(np.all(np.isfinite(signals)))
        np.testing.assert_array_equal(signals[-1], np.zeros(3))

    def test_realized_ic_matches_target(self):
        for signal_ic in (0.05, 0.3, -0.3):
            with self.subTest(signal_ic=signal_ic):
                signals = self._signals(signal_ic)
                np.testing.assert_allclose(
                    _lead_correlation(signals, self.z), signal_ic, atol=0.02
                )

    def test_zero_ic_has_no_predictive_power(self):
        signals = self._signals(0.0)
        np.testing.assert_allclose(_lead_correlation(signals, self.z), 0.0, atol=0.02)

    def test_ic_is_clipped_to_unit_interval(self):
        # An out-of-range IC must not produce NaNs.
        for signal_ic, expected in ((2.0, 0.999), (-2.0, -0.999)):
            with self.subTest(signal_ic=signal_ic):
                signals = self._signals(signal_ic)

                self.assertTrue(np.all(np.isfinite(signals)))
                np.testing.assert_allclose(
                    _lead_correlation(signals, self.z), expected, atol=0.02
                )

    def test_unit_variance(self):
        signals = self._signals(0.05)

        np.testing.assert_allclose(signals.std(axis=0), 1.0, rtol=0.05)
        np.testing.assert_allclose(signals.mean(axis=0), 0.0, atol=0.05)

    def test_autocorrelation_matches_persistence(self):
        # The IC mixture dilutes the AR(1) persistence by (1 - ic ** 2).
        for signal_autocorr in (0.5, 0.9):
            with self.subTest(signal_autocorr=signal_autocorr):
                signal_ic = 0.3
                signals = self._signals(signal_ic, signal_autocorr=signal_autocorr)

                expected = (1.0 - signal_ic**2) * signal_autocorr
                np.testing.assert_allclose(
                    _lag1_autocorr(signals[:-1]), expected, atol=0.03
                )

    def test_zero_autocorr_has_no_persistence(self):
        signals = self._signals(0.05, signal_autocorr=0.0)
        np.testing.assert_allclose(_lag1_autocorr(signals[:-1]), 0.0, atol=0.03)


class TestSimulateSignalsAndReturns(unittest.TestCase):
    def setUp(self):
        self.kwargs = dict(
            n_fids=3,
            n_periods=250,
            base_vol=BASE_VOL_3,
            vol_persistence=0.94,
            vol_of_vol=0.15,
            corr=CORR_3,
            mean_return=0.0,
            signal_autocorr=0.9,
            signal_ic=0.05,
            end_date="2020-12-31",
            seed=42,
        )

    def test_returns_three_aligned_frames(self):
        signals, returns, realized_vol = _simulate_signals_and_returns(**self.kwargs)

        for frame in (signals, returns, realized_vol):
            self.assertIsInstance(frame, pd.DataFrame)
            self.assertEqual(frame.shape, (250, 3))
            self.assertIsInstance(frame.index, pd.DatetimeIndex)
            self.assertTrue(np.all(np.isfinite(frame.to_numpy())))

        pd.testing.assert_index_equal(signals.index, returns.index)
        pd.testing.assert_index_equal(signals.index, realized_vol.index)

    def test_index_respects_end_date_and_frequency(self):
        signals, _, _ = _simulate_signals_and_returns(**self.kwargs)

        self.assertEqual(len(signals.index), 250)
        self.assertEqual(signals.index[-1], pd.Timestamp("2020-12-31"))
        self.assertEqual(signals.index.freqstr, "B")

    def test_default_end_date_is_today(self):
        signals, _, _ = _simulate_signals_and_returns(
            **{**self.kwargs, "end_date": None}
        )

        self.assertLessEqual(signals.index[-1], pd.Timestamp.now().normalize())

    def test_custom_frequency(self):
        signals, _, realized_vol = _simulate_signals_and_returns(
            **{**self.kwargs, "freq": "W-FRI"}
        )

        self.assertEqual(signals.index.freqstr, "W-FRI")
        self.assertEqual(realized_vol.index.freqstr, "W-FRI")
        self.assertTrue(np.all(signals.index.dayofweek == 4))

    def test_scalar_base_vol_is_broadcast(self):
        _, _, realized_vol = _simulate_signals_and_returns(
            **{**self.kwargs, "base_vol": 0.02, "vol_of_vol": 0.0}
        )

        np.testing.assert_allclose(realized_vol.to_numpy(), 0.02)

    def test_scalar_mean_return_is_broadcast(self):
        mean_return = 0.01
        _, returns, _ = _simulate_signals_and_returns(
            **{
                **self.kwargs,
                "n_periods": 20_000,
                "mean_return": mean_return,
                "vol_of_vol": 0.0,
            }
        )

        np.testing.assert_allclose(
            returns.mean(axis=0).to_numpy(), mean_return, atol=0.001
        )

    def test_array_mean_return_per_contract(self):
        mean_return = np.array([0.01, -0.01, 0.0])
        _, returns, _ = _simulate_signals_and_returns(
            **{
                **self.kwargs,
                "n_periods": 20_000,
                "mean_return": mean_return,
                "vol_of_vol": 0.0,
            }
        )

        np.testing.assert_allclose(
            returns.mean(axis=0).to_numpy(), mean_return, atol=0.001
        )

    def test_matches_individual_component_functions(self):
        signals, returns, realized_vol = _simulate_signals_and_returns(**self.kwargs)

        rng = np.random.default_rng(self.kwargs["seed"])
        expected_vol = _simulate_volatility(
            n_periods=250,
            n_fids=3,
            base_vol=BASE_VOL_3,
            vol_persistence=self.kwargs["vol_persistence"],
            vol_of_vol=self.kwargs["vol_of_vol"],
            rng=rng,
        )
        expected_returns, z = _simulate_returns(
            n_periods=250,
            n_fids=3,
            corr=CORR_3,
            mean_return=np.zeros(3),
            realized_vol=expected_vol,
            rng=rng,
            return_z=True,
        )
        expected_signals = _simulate_signals(
            n_periods=250,
            n_fids=3,
            signal_ic=self.kwargs["signal_ic"],
            signal_autocorr=self.kwargs["signal_autocorr"],
            z=z,
            rng=rng,
        )

        np.testing.assert_allclose(realized_vol.to_numpy(), expected_vol)
        np.testing.assert_allclose(returns.to_numpy(), expected_returns)
        np.testing.assert_allclose(signals.to_numpy(), expected_signals)

    def test_signals_predict_next_period_returns(self):
        # End-to-end IC check. With constant volatility the correlation against
        # the returns themselves is exactly the target IC.
        signal_ic = 0.1
        signals, returns, _ = _simulate_signals_and_returns(
            **{
                **self.kwargs,
                "n_periods": N_LONG,
                "signal_ic": signal_ic,
                "vol_of_vol": 0.0,
            }
        )

        realized = _lead_correlation(signals.to_numpy(), returns.to_numpy())
        np.testing.assert_allclose(realized, signal_ic, atol=0.02)

    def test_returns_reproduce_requested_risk(self):
        _, returns, _ = _simulate_signals_and_returns(
            **{**self.kwargs, "n_periods": N_LONG, "vol_of_vol": 0.0}
        )

        np.testing.assert_allclose(
            returns.std(axis=0).to_numpy(), BASE_VOL_3, rtol=0.02
        )
        np.testing.assert_allclose(np.corrcoef(returns.to_numpy().T), CORR_3, atol=0.02)



class TestGeneratorInit(unittest.TestCase):
    def test_default_corr_is_a_valid_correlation_matrix(self):
        generator = SignalsAndReturnsGenerator(n_fids=4)

        self.assertEqual(generator.corr.shape, (4, 4))
        np.testing.assert_allclose(generator.corr, generator.corr.T)
        np.testing.assert_allclose(np.diag(generator.corr), 1.0)
        self.assertTrue(np.all(np.abs(generator.corr) <= 1.0 + 1e-12))
        # Positive definite, so `_simulate_returns` can factorise it.
        self.assertGreater(np.linalg.eigvalsh(generator.corr).min(), 0.0)

    def test_default_base_vol(self):
        generator = SignalsAndReturnsGenerator(n_fids=3)
        np.testing.assert_allclose(generator.base_vol, np.full(3, 0.01))

    def test_explicit_parameters_are_stored(self):
        generator = SignalsAndReturnsGenerator(
            n_fids=3,
            corr=CORR_3,
            base_vol=BASE_VOL_3,
            signal_ic=0.2,
            signal_autocorr=0.5,
            vol_persistence=0.8,
            vol_of_vol=0.25,
            mean_return=0.001,
        )

        self.assertEqual(generator.n_fids, 3)
        np.testing.assert_allclose(generator.corr, CORR_3)
        np.testing.assert_allclose(generator.base_vol, BASE_VOL_3)
        self.assertEqual(generator.signal_ic, 0.2)
        self.assertEqual(generator.signal_autocorr, 0.5)
        self.assertEqual(generator.vol_persistence, 0.8)
        self.assertEqual(generator.vol_of_vol, 0.25)
        self.assertEqual(generator.mean_return, 0.001)

    def test_no_simulation_state_before_simulating(self):
        generator = SignalsAndReturnsGenerator(n_fids=3)

        self.assertIsNone(generator.signals)
        self.assertIsNone(generator.returns)
        self.assertIsNone(generator.realized_vol)
        self.assertIsNone(generator.freq)


class TestGeneratorSimulate(unittest.TestCase):
    def setUp(self):
        self.generator = SignalsAndReturnsGenerator(
            n_fids=3, corr=CORR_3, base_vol=BASE_VOL_3
        )

    def test_stores_what_it_returns(self):
        signals, returns, realized_vol = self.generator.simulate_signals_and_returns(
            n_periods=100, end_date="2020-12-31"
        )

        self.assertIs(signals, self.generator.signals)
        self.assertIs(returns, self.generator.returns)
        self.assertIs(realized_vol, self.generator.realized_vol)
        self.assertEqual(self.generator.freq, "B")

    def test_resimulation_overwrites_state(self):
        self.generator.simulate_signals_and_returns(
            n_periods=100, end_date="2020-12-31"
        )
        first = self.generator.signals
        self.generator.simulate_signals_and_returns(
            n_periods=50, end_date="2019-12-31", freq="W-FRI"
        )

        self.assertEqual(self.generator.signals.shape, (50, 3))
        self.assertEqual(self.generator.returns.shape, (50, 3))
        self.assertEqual(self.generator.realized_vol.shape, (50, 3))
        self.assertEqual(self.generator.freq, "W-FRI")
        self.assertIsNot(self.generator.signals, first)

    def test_matches_module_level_function(self):
        generator = SignalsAndReturnsGenerator(
            n_fids=3,
            corr=CORR_3,
            base_vol=BASE_VOL_3,
            signal_ic=0.2,
            signal_autocorr=0.5,
            vol_persistence=0.8,
            vol_of_vol=0.25,
            mean_return=0.001,
        )
        from_method = generator.simulate_signals_and_returns(
            n_periods=120, end_date="2020-12-31", seed=17
        )
        from_function = _simulate_signals_and_returns(
            n_fids=3,
            n_periods=120,
            corr=CORR_3,
            base_vol=BASE_VOL_3,
            signal_ic=0.2,
            signal_autocorr=0.5,
            vol_persistence=0.8,
            vol_of_vol=0.25,
            mean_return=0.001,
            end_date="2020-12-31",
            seed=17,
        )

        for lhs, rhs in zip(from_method, from_function):
            pd.testing.assert_frame_equal(lhs, rhs)


class TestGeneratorQuantamentalConversions(unittest.TestCase):
    def setUp(self):
        self.signal_names = ["CAD_SIG", "GBP_SIG", "JPY_SIG"]
        self.return_names = ["CAD_XR", "GBP_XR", "JPY_XR"]
        self.n_periods = 100
        self.generator = SignalsAndReturnsGenerator(
            n_fids=3, corr=CORR_3, base_vol=BASE_VOL_3
        )
        self.generator.simulate_signals_and_returns(
            n_periods=self.n_periods,
            end_date="2020-12-31",
            signal_names=self.signal_names,
            return_names=self.return_names,
        )

    def test_quantamental_signals(self):
        qdf = self.generator.quantamental_signals()

        self.assertIsInstance(qdf, QuantamentalDataFrame)
        self.assertEqual(set(qdf.columns), {"real_date", "cid", "xcat", "value"})
        self.assertEqual(len(qdf), self.n_periods * 3)
        self.assertEqual(sorted(qdf["cid"].unique()), ["CAD", "GBP", "JPY"])
        self.assertEqual(sorted(qdf["xcat"].unique()), ["SIG"])

        wide = QuantamentalDataFrame(qdf).to_wide()
        np.testing.assert_allclose(
            wide[self.signal_names].to_numpy(), self.generator.signals.to_numpy()
        )

    def test_quantamental_returns_are_in_percent(self):
        qdf = self.generator.quantamental_returns()

        self.assertIsInstance(qdf, QuantamentalDataFrame)
        self.assertEqual(len(qdf), self.n_periods * 3)
        self.assertEqual(sorted(qdf["xcat"].unique()), ["XR"])

        wide = QuantamentalDataFrame(qdf).to_wide()
        np.testing.assert_allclose(
            wide[self.return_names].to_numpy(),
            100.0 * self.generator.returns.to_numpy(),
        )

    def test_quantamental_returns_and_signals(self):
        qdf = self.generator.quantamental_returns_and_signals()
        signals = self.generator.quantamental_signals()
        returns = self.generator.quantamental_returns()

        self.assertEqual(len(qdf), len(signals) + len(returns))
        self.assertEqual(sorted(qdf["xcat"].unique()), ["SIG", "XR"])
        self.assertEqual(sorted(qdf["cid"].unique()), ["CAD", "GBP", "JPY"])
        self.assertTrue(qdf["value"].notna().all())

        wide = QuantamentalDataFrame(qdf).to_wide()
        np.testing.assert_allclose(
            wide[self.signal_names].to_numpy(), self.generator.signals.to_numpy()
        )
        np.testing.assert_allclose(
            wide[self.return_names].to_numpy(),
            100.0 * self.generator.returns.to_numpy(),
        )


class TestGeneratorRealizedCov(unittest.TestCase):
    def setUp(self):
        self.return_names = ["CAD_XR", "GBP_XR", "JPY_XR"]
        self.generator = SignalsAndReturnsGenerator(
            n_fids=3, corr=CORR_3, base_vol=BASE_VOL_3
        )
        self.generator.simulate_signals_and_returns(
            n_periods=400, end_date="2020-12-31", return_names=self.return_names
        )
        self.rebalance_dates = pd.date_range(
            start=self.generator.realized_vol.index[0],
            end=self.generator.realized_vol.index[-1],
            freq="BMS",
        )

    def _constant_vol_generator(self, freq: str = "B", n_periods: int = 400):
        generator = SignalsAndReturnsGenerator(
            n_fids=3, corr=CORR_3, base_vol=BASE_VOL_3, vol_of_vol=0.0
        )
        generator.simulate_signals_and_returns(
            n_periods=n_periods,
            end_date="2020-12-31",
            freq=freq,
            return_names=self.return_names,
        )
        return generator

    def test_long_format_holds_upper_triangle(self):
        cov = self.generator.realized_cov()
        order = {name: i for i, name in enumerate(self.return_names)}

        self.assertEqual(set(cov["fid1"]) | set(cov["fid2"]), set(self.return_names))
        self.assertTrue(
            all(order[a] <= order[b] for a, b in zip(cov["fid1"], cov["fid2"]))
        )

    def test_wide_format_matrices(self):
        cov = self.generator.realized_cov(long=False)

        self.assertIsInstance(cov, dict)
        self.assertEqual(list(cov), list(self.rebalance_dates[:-1]))
        for date, matrix in cov.items():
            self.assertIsInstance(date, pd.Timestamp)
            self.assertEqual(matrix.shape, (3, 3))
            np.testing.assert_allclose(matrix, matrix.T)
            self.assertGreater(np.linalg.eigvalsh(matrix).min(), 0.0)

    def test_long_and_wide_formats_agree(self):
        long = self.generator.realized_cov(long=True)
        wide = self.generator.realized_cov(long=False)
        order = {name: i for i, name in enumerate(self.return_names)}

        for row in long.itertuples(index=False):
            expected = wide[row.real_date][order[row.fid1], order[row.fid2]]
            self.assertAlmostEqual(row.value, expected, places=10)

    def test_constant_vol_and_known_corr_are_exact(self):
        generator = self._constant_vol_generator()
        cov = generator.realized_cov(long=False)
        expected = (
            10_000
            * ANNUALIZATION_FACTORS["B"]
            * (CORR_3 * np.outer(BASE_VOL_3, BASE_VOL_3))
        )

        self.assertTrue(len(cov) > 0)
        for matrix in cov.values():
            np.testing.assert_allclose(matrix, expected)
            # Variances in percent^2: 0.01 per day -> 252 annualized.
            np.testing.assert_allclose(np.diag(matrix), 10_000 * 252 * BASE_VOL_3**2)

    def test_diagonal_corr_has_no_covariance(self):
        generator = SignalsAndReturnsGenerator(
            n_fids=3, corr=np.eye(3), base_vol=BASE_VOL_3, vol_of_vol=0.0
        )
        generator.simulate_signals_and_returns(n_periods=400, end_date="2020-12-31")
        cov = generator.realized_cov(long=False)

        for matrix in cov.values():
            np.testing.assert_allclose(matrix[~np.eye(3, dtype=bool)], 0.0)

    def test_matches_manual_recomputation(self):
        # Ground truth: the average of D_t C D_t over the interval, annualized
        # and rescaled to percent^2.
        cov = self.generator.realized_cov(long=False)
        realized_vol = self.generator.realized_vol
        scale = 10_000 * ANNUALIZATION_FACTORS["B"]

        for start, end in zip(self.rebalance_dates, self.rebalance_dates[1:]):
            vol = realized_vol.loc[start:end].to_numpy()
            expected = scale * np.einsum("ti,ij,tj->tij", vol, CORR_3, vol).mean(axis=0)
            np.testing.assert_allclose(cov[start], expected)

    def test_annualization_follows_simulation_frequency(self):
        generator = self._constant_vol_generator(freq="W-FRI", n_periods=200)
        cov = generator.realized_cov(long=False)
        expected = (
            10_000
            * ANNUALIZATION_FACTORS["W-FRI"]
            * (CORR_3 * np.outer(BASE_VOL_3, BASE_VOL_3))
        )

        self.assertTrue(len(cov) > 0)
        for matrix in cov.values():
            np.testing.assert_allclose(matrix, expected)

    def test_rebalance_frequency_sets_the_number_of_intervals(self):
        monthly = self.generator.realized_cov(freq="BMS")
        weekly = self.generator.realized_cov(freq="W-FRI")

        self.assertGreater(
            weekly["real_date"].nunique(), monthly["real_date"].nunique()
        )
        for cov, freq in ((monthly, "BMS"), (weekly, "W-FRI")):
            dates = pd.date_range(
                start=self.generator.realized_vol.index[0],
                end=self.generator.realized_vol.index[-1],
                freq=freq,
            )
            self.assertEqual(cov["real_date"].nunique(), len(dates) - 1)

    def test_no_intervals_when_sample_is_too_short(self):
        generator = SignalsAndReturnsGenerator(n_fids=3, corr=CORR_3)
        generator.simulate_signals_and_returns(n_periods=5, end_date="2020-01-10")

        self.assertEqual(generator.realized_cov(long=False), {})


if __name__ == "__main__":
    unittest.main()
