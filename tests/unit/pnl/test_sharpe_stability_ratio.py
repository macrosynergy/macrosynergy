import unittest
import numpy as np
import pandas as pd

from macrosynergy.pnl.sharpe_stability_ratio import (
    sharpe_stability_ratio,
    _andrews_ar1_bandwidth,
    _newey_west_lrv,
)


class TestSharpeStabilityRatio(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        N = 1500
        # Stable: consistent positive daily returns
        self.stable = pd.Series(rng.normal(0.001, 0.01, N))
        # Episodic: same total mean, but concentrated in first 20%
        episodic = np.zeros(N)
        episodic[:300] = rng.normal(0.005, 0.01, 300)
        episodic[300:] = rng.normal(0.0, 0.01, 1200)
        self.episodic = pd.Series(episodic)

    # --- Return type and basic validity ---

    def test_return_type_is_float(self):
        result = sharpe_stability_ratio(self.stable)
        self.assertIsInstance(result, float)

    def test_result_is_not_nan_for_valid_data(self):
        result = sharpe_stability_ratio(self.stable)
        self.assertFalse(np.isnan(result))

    def test_numpy_array_input(self):
        result = sharpe_stability_ratio(self.stable.values)
        self.assertIsInstance(result, float)
        self.assertFalse(np.isnan(result))

    # --- Core discriminating property ---

    def test_stable_higher_ssr_than_episodic(self):
        ssr_stable = sharpe_stability_ratio(self.stable)
        ssr_episodic = sharpe_stability_ratio(self.episodic)
        self.assertGreater(ssr_stable, ssr_episodic)

    # --- Benchmark SR ---

    def test_higher_benchmark_lowers_ssr(self):
        ssr_0 = sharpe_stability_ratio(self.stable, benchmark_sr=0.0)
        ssr_05 = sharpe_stability_ratio(self.stable, benchmark_sr=0.5)
        self.assertGreater(ssr_0, ssr_05)

    # --- Edge cases ---

    def test_short_series_returns_nan(self):
        short = pd.Series(np.random.normal(0.001, 0.01, 50))
        result = sharpe_stability_ratio(short, window=252)
        self.assertTrue(np.isnan(result))

    def test_constant_returns_returns_nan(self):
        # Constant returns → rolling std = 0 → rolling SR = NaN/inf
        constant = pd.Series(np.full(600, 0.001))
        result = sharpe_stability_ratio(constant)
        self.assertTrue(np.isnan(result))

    def test_zero_returns_returns_nan(self):
        zeros = pd.Series(np.zeros(600))
        result = sharpe_stability_ratio(zeros)
        self.assertTrue(np.isnan(result))

    def test_nan_values_dropped(self):
        rng = np.random.default_rng(10)
        clean = pd.Series(rng.normal(0.001, 0.01, 1500))
        with_nans = clean.copy()
        with_nans.iloc[::20] = np.nan  # inject NaNs every 20 obs
        ssr_clean = sharpe_stability_ratio(clean)
        ssr_nans = sharpe_stability_ratio(with_nans)
        # Both should be finite
        self.assertFalse(np.isnan(ssr_clean))
        self.assertFalse(np.isnan(ssr_nans))

    def test_negative_mean_returns_negative_ssr(self):
        rng = np.random.default_rng(99)
        neg = pd.Series(rng.normal(-0.002, 0.01, 1500))
        result = sharpe_stability_ratio(neg, benchmark_sr=0.0)
        self.assertLess(result, 0.0)

    # --- Window parameter ---

    def test_shorter_window_also_finite(self):
        ssr_126 = sharpe_stability_ratio(self.stable, window=126)
        ssr_252 = sharpe_stability_ratio(self.stable, window=252)
        self.assertFalse(np.isnan(ssr_126))
        self.assertFalse(np.isnan(ssr_252))

    # --- Input validation ---

    def test_invalid_window_raises(self):
        with self.assertRaises(ValueError):
            sharpe_stability_ratio(self.stable, window=1)

    def test_invalid_annualization_raises(self):
        with self.assertRaises(ValueError):
            sharpe_stability_ratio(self.stable, annualization_factor=0)

    # --- Sanity check on magnitude ---

    def test_iid_positive_returns_reasonable_ssr(self):
        rng = np.random.default_rng(55)
        returns = pd.Series(rng.normal(0.001, 0.01, 2000))
        result = sharpe_stability_ratio(returns)
        self.assertGreater(result, 0.0)
        self.assertLess(result, 50.0)


if __name__ == "__main__":
    unittest.main()
