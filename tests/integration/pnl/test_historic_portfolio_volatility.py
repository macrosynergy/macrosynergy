"""Test historical volatility estimates with simulate returns from random normal distribution"""

import unittest
import pandas as pd
import numpy as np

from macrosynergy.pnl.historic_portfolio_volatility import historic_portfolio_vol
from macrosynergy.panel.historic_vol import historic_vol
from macrosynergy.management.types import QuantamentalDataFrame


class TestHistoricPortfolioVol(unittest.TestCase):
    def setUp(self):
        # TODO write into a class and use for monte-carlo simulation (checks on logic)
        self.periods = 252 * 20
        self.n_cids = 3  # keep number of cross section manageable for now
        annual_vol = np.array([10, 20, 30]) ** 2
        print(
            "Annual volatility:",
            annual_vol,
            " - annual standard deviation:",
            np.sqrt(annual_vol),
        )

        self.mean = np.sqrt(annual_vol) / 252
        print("mean (daily):", self.mean)

        sigma = np.sqrt(annual_vol) / 252  # daily volatility
        correlations_map = {
            (0, 1): 0.5,  # 1 to 2 and 2 to 1
            (0, 2): 0.4,  # 1 to 3 and 3 to 1
            (1, 2): 0.3,  # 2 to 3 and 3 to 2
        }

        print("sigma (daily volatility):", sigma)
        cov = np.diag(sigma)
        for i in range(self.n_cids):
            for j in range(self.n_cids):
                if i != j:
                    cov[i, j] = (
                        sigma[i]
                        * sigma[j]
                        * correlations_map[tuple(sorted((i, j)))]
                    )

        self.cov = cov
        self.sigma = sigma
        # TODO build variance-covariance matrix (simulate...)
        # TODO build sharpe ratios...

    def test_monte_carlo_checks_on_estimates(self):
        # Monte Carlo checks on estimates (validations)
        n_sim: int = 100
        mean_est = np.empty(shape=(n_sim, self.n_cids))
        std_est = np.empty(shape=(n_sim, self.n_cids))

        for i in range(n_sim):
            rtn = np.random.multivariate_normal(
                mean=self.mean, cov=self.cov, size=self.periods
            )
            mean_est[i, :] = rtn.mean(axis=0)
            std_est[i, :] = rtn.std(axis=0)

        # TODO convert in to actual check...
        print("estimate of mean:", (self.mean - mean_est.mean(axis=0)).round(6))
        # TODO describe distribution...
        print("Estimate of std:", (np.sqrt(self.sigma) - std_est.mean(axis=0)).round(6))

    def _shape_df(self, dfx: pd.DataFrame) -> pd.DataFrame:
        dfx.index.name = "real_date"
        dfx.columns.name = "ticker"
        dfx = dfx.stack().to_frame("value").reset_index()
        dfx[["cid", "xcat"]] = dfx.ticker.str.split("_", expand=True, n=1)
        return dfx

    def _gen_sig_df(self, dates, sig_type: tuple = (1, 1, 1)) -> pd.DataFrame:
        df_signals = pd.DataFrame(
            [sig_type],
            columns=[f"XX{i}_AA_CSIG_STRAT" for i in range(self.n_cids)],
            index=dates,
        )

        return self._shape_df(df_signals)

    def _gen_rtn_df(self, dates):
        rtn = np.random.multivariate_normal(
            mean=self.mean, cov=self.cov, size=self.periods
        )
        df_returns = pd.DataFrame(
            rtn, columns=[f"XX{i}_AAXR" for i in range(self.n_cids)], index=dates
        )

        return self._shape_df(df_returns)

    def test_unit_weights(self):
        # TODO signal types: [1] (1, 1, 1), [2] (0, 0, 0), [3] (1, 0, 0), [4] (0, 1, 0), [5] (0, 0, 1)
        # TODO check logic of historic_portfolio_volatility follows above "contract signals"
        # TODO monte-carlo logic checks on historic_portfolio_volatility
        signal_types = [
            (1, 1, 1),
            (0, 0, 0),
            (1, 0, 0),
            (0, 1, 0),
            (0, 0, 1),
        ]

        # Check portfolio volatility
        dates = pd.bdate_range(
            end=pd.Timestamp.today() + pd.offsets.BDay(n=0), periods=self.periods
        )
        start = dates.min()
        end = dates.max()

        def _get_test_df(dates, sig_type):
            _dfx = pd.concat(
                (
                    self._gen_sig_df(dates=dates, sig_type=sig_type),
                    self._gen_rtn_df(dates=dates),
                ),
                axis=0,
            )
            _dfx[["cid", "xcat"]] = _dfx.ticker.str.split("_", expand=True, n=1)
            return _dfx

        good_args = dict(
            sname="STRAT",
            rstring="XR",
            est_freqs=["m"],
            lback_periods=21,
            lback_meth="ma",
            half_life=11,
            start=None,
            end=None,
            blacklist=None,
            nan_tolerance=0.25,
            remove_zeros=True,
            fids=[f"XX{i}_AA" for i in range(self.n_cids)],
        )

        rtn_df = self._gen_rtn_df(dates=dates)
        sig_df = self._gen_sig_df(dates=dates, sig_type=signal_types[1])
        df = pd.concat((rtn_df, sig_df), axis=0)

        pvol_vcv = historic_portfolio_vol(df=df, **good_args)
        df_pvol, df_vcv = pvol_vcv
        self.assertIsInstance(df_pvol, QuantamentalDataFrame)
        self.assertTrue(all(df_pvol["value"] == 0))

        results = {}
        for i in range(3):
            rtn_df = self._gen_rtn_df(dates=dates)
            sig_df = self._gen_sig_df(dates=dates, sig_type=signal_types[2 + i])
            df_pvol = historic_portfolio_vol(
                df=pd.concat((rtn_df, sig_df), axis=0), **good_args
            )
            results[i] = dict(df_pvol=df_pvol, rtn_df=rtn_df, sig_df=sig_df)
            hvol = (
                historic_vol(
                    df=rtn_df,
                    xcat="AAXR",
                    est_freq="m",
                    lback_periods=21,
                    lback_meth="ma",
                    cids=["XX" + str(list(signal_types[2 + i]).index(1))],
                    start=start.strftime("%Y-%m-%d"),
                    end=end.strftime("%Y-%m-%d"),
                )
                .dropna()
                .reset_index(drop=True)
            )
            hvol, df_pvol

        results


if __name__ == "__main__":
    import macrosynergy.visuals as msv

    unittest.main()
