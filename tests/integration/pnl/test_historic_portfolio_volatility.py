"""Test historical volatility estimates with simulate returns from random normal distribution"""
import unittest
import pandas as pd
import numpy as np

from macrosynergy.pnl.historic_portfolio_volatility import historic_portfolio_vol


class TestHistoricPortfolioVol(unittest.TestCase):
    def setUp(self):        
        # TODO write into a class and use for monte-carlo simulation (checks on logic)
        self.periods = 252*20
        self.n_cids = 3  # keep number of cross section manageable for now
        annual_vol = np.array([10, 20, 30]) ** 2
        print("Annual volatility:", annual_vol, " - annual standard deviation:", np.sqrt(annual_vol))

        self.mean = np.sqrt(annual_vol) / 252
        print("mean (daily):", self.mean)

        sigma = np.sqrt(annual_vol) / 252 # daily volatility
        correlations_map = {
            (0, 1): 0.5, # 1 to 2 and 2 to 1
            (0, 2): 0.4, # 1 to 3 and 3 to 1
            (1, 2): 0.3, # 2 to 3 and 3 to 2
        }

        print("sigma (daily volatility):", sigma)
        cov = np.diag(sigma)
        for ii in range(self.n_cids):
            for jj in range(self.n_cids):
                if ii != jj:
                    cov[ii, jj] = sigma[ii] * sigma[jj] * correlations_map[tuple(sorted((ii, jj)))]
        self.cov = cov
        # TODO build variance-covariance matrix (simulate...)
        # TODO build sharpe ratios...

    def test_monte_carlo_checks_on_estimates(self):
        # Monte Carlo checks on estimates (validations)
        n_sim: int = 100
        mean_est = np.empty(shape=(n_sim, self.n_cids))
        std_est = np.empty(shape=(n_sim, self.n_cids))

        for ii in range(n_sim):
            rtn = np.random.multivariate_normal(mean=self.mean, cov=self.cov, size=self.periods)
            mean_est[ii, :] = rtn.mean(axis=0)
            std_est[ii, :] = rtn.std(axis=0)

        # TODO convert in to actual check...
        print("estimate of mean:", (self.mean - mean_est.mean(axis=0)).round(6))  # TODO describe distribution...
        print("Estimate of std:", (np.sqrt(self.sigma) - std_est.mean(axis=0)).round(6))

    def test_clean_run(self):
        # Check portfolio volatility
        rtn = np.random.multivariate_normal(mean=self.mean, cov=self.cov, size=self.periods)
        dates = pd.bdate_range(end=pd.Timestamp.today() + pd.offsets.BDay(n=0), periods=self.periods)

        df_returns = pd.DataFrame(rtn, columns=[f"XX{ii}_AAXR" for ii in range(self.n_cids)], index=dates)
        df_returns.index.name = "real_date"
        df_returns.columns.name = "ticker"

        df_signals = pd.DataFrame([(1, 1, 1)], columns=[f"XX{ii}_AA_CSIG_STRAT" for ii in range(self.n_cids)], index=dates)
        df_signals.index.name = "real_date"
        df_signals.columns.name = "ticker"

        df = pd.concat((df_returns.stack().to_frame("value").reset_index(), df_signals.stack().to_frame("value").reset_index()), axis=0)
        df[["cid", "xcat"]] = df.ticker.str.split("_", expand=True, n=1)
        fids = [f"XX{ii}_AA" for ii in range(self.n_cids)]
        # TODO signal types: [1] (1, 1, 1), [2] (0, 0, 0), [3] (1, 0, 0), [4] (0, 1, 0), [5] (0, 0, 1)
        # TODO check logic of historic_portfolio_volatility follows above "contract signals"
        # TODO monte-carlo logic checks on historic_portfolio_volatility
        df_pvol = historic_portfolio_vol(
            df=df,
            sname="STRAT",
            fids=fids,
            rstring="XR",
            est_freq="m",
            lback_periods=21,
            lback_meth="ma",
            half_life=11,
            start=None,
            end=None,
            blacklist=None,
            nan_tolerance=0.25,
            remove_zeros=True
        )
        # TODO check values...
        self.assertIsInstance(df_pvol, pd.DataFrame)

    def test_start_date(self):
        # Check portfolio volatility
        rtn = np.random.multivariate_normal(mean=self.mean, cov=self.cov, size=self.periods)
        dates = pd.bdate_range(end=pd.Timestamp.today() + pd.offsets.BDay(n=0), periods=self.periods)

        df_returns = pd.DataFrame(rtn, columns=[f"XX{ii}_AAXR" for ii in range(self.n_cids)], index=dates)
        df_returns.index.name = "real_date"
        df_returns.columns.name = "ticker"

        df_signals = pd.DataFrame([(1, 1, 1)], columns=[f"XX{ii}_AA_CSIG_STRAT" for ii in range(self.n_cids)], index=dates)
        df_signals.index.name = "real_date"
        df_signals.columns.name = "ticker"

        df = pd.concat((df_returns.stack().to_frame("value").reset_index(), df_signals.stack().to_frame("value").reset_index()), axis=0)
        df[["cid", "xcat"]] = df.ticker.str.split("_", expand=True, n=1)
        fids = [f"XX{ii}_AA" for ii in range(self.n_cids)]
        # TODO signal types: [1] (1, 1, 1), [2] (0, 0, 0), [3] (1, 0, 0), [4] (0, 1, 0), [5] (0, 0, 1)
        # TODO check logic of historic_portfolio_volatility follows above "contract signals"
        # TODO monte-carlo logic checks on historic_portfolio_volatility
        df_pvol = historic_portfolio_vol(
            df=df,
            sname="STRAT",
            fids=fids,
            rstring="XR",
            est_freq="m",
            lback_periods=21,
            lback_meth="ma",
            half_life=11,
            start=f"{df_returns.index[30]:%Y-%m-%d}",
            end=None,
            blacklist=None,
            nan_tolerance=0.25,
            remove_zeros=True
        )
        # TODO check values...
        self.assertIsInstance(df_pvol, pd.DataFrame)



if __name__ == "__main__":
    unittest.main()




