import unittest
import numpy as np
import pandas as pd

from macrosynergy.pnl.max_drawdown_recovery_months import max_drawdown_recovery_months


class TestMaxDrawdownRecoveryMonths(unittest.TestCase):
    def test_no_drawdown_returns_zero(self):
        cum_pnl = pd.Series(np.arange(1, 101, dtype=float))
        self.assertEqual(max_drawdown_recovery_months(cum_pnl, traded_months=12), 0.0)

    def test_recovers_within_sample(self):
        # Rises 21 days to a peak, falls for 21 days to a trough, then
        # climbs back past the prior peak over another 21 days. Recovery is
        # measured from the *peak*, not the trough, so this is 42 trading
        # days -- 2 months -- end to end.
        up = np.arange(1, 22, dtype=float)
        down = up[-1] - np.arange(1, 22, dtype=float)
        recover = down[-1] + np.arange(1, 22, dtype=float)
        cum_pnl = pd.Series(np.concatenate([up, down, recover]))
        self.assertEqual(
            max_drawdown_recovery_months(cum_pnl, traded_months=12), 2.0
        )

    def test_never_recovers_defaults_to_traded_months(self):
        up = np.arange(1, 22, dtype=float)
        down = up[-1] - np.arange(1, 22, dtype=float)  # ends underwater, no recovery
        cum_pnl = pd.Series(np.concatenate([up, down]))
        self.assertEqual(
            max_drawdown_recovery_months(cum_pnl, traded_months=7), 7
        )

    def test_empty_series_returns_nan(self):
        cum_pnl = pd.Series([], dtype=float)
        result = max_drawdown_recovery_months(cum_pnl, traded_months=12)
        self.assertTrue(np.isnan(result))


if __name__ == "__main__":
    unittest.main()
