from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from macrosynergy.management.constants import ANNUALIZATION_FACTORS
from macrosynergy.management.types import QuantamentalDataFrame


def _simulate_volatility(
    n_periods: int,
    n_fids: int,
    base_vol: np.ndarray,
    vol_persistence: float,
    vol_of_vol: float,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    # Time-varying volatility via a log-variance EWMA / GARCH-like process.
    # log-variance mean-reverts to log(base_vol^2) with persistent shocks,
    # producing volatility clustering rather than constant vol.
    rng = rng or np.random.default_rng()

    log_var = np.empty((n_periods, n_fids))
    long_run_log_var = np.log(base_vol ** 2)
    state = long_run_log_var.copy()
    for t in range(n_periods):
        shock = rng.standard_normal(n_fids) * vol_of_vol
        state = vol_persistence * state + (1.0 - vol_persistence) * long_run_log_var + shock
        log_var[t] = state

    realized_vol = np.sqrt(np.exp(log_var))

    return realized_vol

def _simulate_returns(
    n_periods: int,
    n_fids: int,
    corr: np.ndarray,
    mean_return: np.ndarray,
    realized_vol: np.ndarray,
    rng: Optional[np.random.Generator] = None,
    return_z: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    rng = rng or np.random.default_rng()

    L = np.linalg.cholesky(corr)
    z = rng.standard_normal((n_periods, n_fids)) @ L.T

    returns = mean_return + realized_vol * z

    return (returns, z) if return_z else returns


def _simulate_signals(
    n_periods: int,
    n_fids: int,
    signal_ic: float,
    signal_autocorr: float,
    z: np.ndarray,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    # Persistent signals correlated with the *next* period's innovation.
    # First build a persistent AR(1) signal, then mix it with the innovation z
    # it is meant to predict so the realized IC = corr(signal_t, z_t) hits the
    # target regardless of the persistence level.
    ic = float(np.clip(signal_ic, -0.999, 0.999))
    a = signal_autocorr

    persistent = np.empty((n_periods, n_fids))
    s = np.zeros(n_fids)
    eps = rng.standard_normal((n_periods, n_fids))
    for t in range(n_periods):
        s = a * s + np.sqrt(1.0 - a ** 2) * eps[t]
        persistent[t] = s  # unit-variance AR(1), independent of z

    b = ic
    aligned = b * z + np.sqrt(1.0 - b ** 2) * persistent

    # aligned[t] is correlated with z[t]. We want signals[t] to predict
    # returns[t+1], i.e. signals[t] must carry information about z[t+1].
    # Shift backward by one so signals[t] = aligned[t+1].
    signals = np.roll(aligned, -1, axis=0)
    signals[-1] = 0.0  # last signal predicts an unobserved future return

    return signals


def _simulate_signals_and_returns(
    n_fids: int,
    n_periods: int,
    # volatility modelling params
    base_vol: np.ndarray,
    vol_persistence: float,
    vol_of_vol: float,
    # return modelling params
    corr: np.ndarray,
    mean_return: float,
    # signal modelling params
    signal_autocorr: float,
    signal_ic: float,
    # other params
    end_date: Optional[str] = None,
    signal_names: Optional[List[str]] = None,
    return_names: Optional[List[str]] = None,
    freq: str = "B",
    seed: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # checks
    if n_periods < 1:
        raise ValueError("n_periods must be at least 1")
    if n_fids < 1:
        raise ValueError("n_fids must be at least 1")

    corr = np.asarray(corr, dtype=float)
    if corr.shape != (n_fids, n_fids):
        raise ValueError("corr must be (n_fids, n_fids)")
    if not np.allclose(corr, corr.T, atol=1e-10):
        raise ValueError("corr must be symmetric")
    if not np.allclose(np.diag(corr), 1.0, atol=1e-10):
        raise ValueError("corr must have unit diagonal")

    if np.any(base_vol <= 0.0):
        raise ValueError("base_vol must be strictly positive")
    if not (-1.0 < signal_autocorr < 1.0):
        raise ValueError("signal_autocorr must be strictly between -1 and 1")

    if end_date is None:
        end_date = pd.Timestamp.now().strftime("%Y-%m-%d")

    # set seed
    rng = np.random.default_rng(seed)

    # ensure correct dimensions
    base_vol = np.broadcast_to(np.asarray(base_vol, dtype=float), (n_fids,)).copy()
    mean_return = np.broadcast_to(np.asarray(mean_return, dtype=float), (n_fids,)).copy()

    realized_vol = _simulate_volatility(
        n_periods=n_periods,
        n_fids=n_fids,
        base_vol=base_vol,
        vol_persistence=vol_persistence,
        vol_of_vol=vol_of_vol,
        rng=rng,
    )

    returns, z = _simulate_returns(
        n_periods=n_periods,
        n_fids=n_fids,
        corr=corr,
        mean_return=mean_return,
        realized_vol=realized_vol,
        return_z=True,
        rng=rng,
    )

    signals = _simulate_signals(
        n_periods=n_periods,
        n_fids=n_fids,
        z=z,
        signal_ic=signal_ic,
        signal_autocorr=signal_autocorr,
        rng=rng,
    )

    # convert to dataframes
    index = pd.date_range(end=end_date, periods=n_periods, freq=freq)
    if signal_names is None:
        signal_names = [f"CID{i}_SIG" for i in range(n_fids)]
    if return_names is None:
        return_names = [f"CID{i}_XR" for i in range(n_fids)]

    signals = pd.DataFrame(signals, index=index, columns=signal_names)
    returns = pd.DataFrame(returns, index=index, columns=return_names)
    realized_vol = pd.DataFrame(realized_vol, index=index, columns=return_names)

    return signals, returns, realized_vol


class SignalsAndReturnsGenerator:
    def __init__(
        self,
        n_fids: int,
        corr: np.ndarray = None,
        base_vol: np.ndarray = None,
        signal_ic: float = 0.05,
        signal_autocorr: float = 0.9,
        vol_persistence: float = 0.94,
        vol_of_vol: float = 0.15,
        mean_return: float = 0.0,
    ) -> None:
        self.n_fids = n_fids
        self.corr = None if corr is None else np.asarray(corr, dtype=float)
        self.base_vol = base_vol
        self.signal_ic = signal_ic
        self.signal_autocorr = signal_autocorr
        self.vol_persistence = vol_persistence
        self.vol_of_vol = vol_of_vol
        self.mean_return = mean_return

        if corr is None:
            rng = np.random.default_rng(0)
            A = rng.standard_normal((n_fids, n_fids))
            C = A @ A.T
            d = np.sqrt(np.diag(C))
            self.corr = C / np.outer(d, d)

        if base_vol is None:
            self.base_vol = np.full(n_fids, 0.01)

        self.signals = None
        self.returns = None
        self.realized_vol = None
        self.freq = None

    def simulate_signals_and_returns(
        self,
        n_periods: int,
        end_date: Optional[str] = None,
        signal_names: Optional[List[str]] = None,
        return_names: Optional[List[str]] = None,
        freq: str = "B",
        seed: int = 29,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        signals, returns, realized_vol = _simulate_signals_and_returns(
            n_fids=self.n_fids,
            n_periods=n_periods,
            corr=self.corr,
            base_vol=self.base_vol,
            signal_ic=self.signal_ic,
            signal_autocorr=self.signal_autocorr,
            vol_persistence=self.vol_persistence,
            vol_of_vol=self.vol_of_vol,
            mean_return=self.mean_return,
            freq=freq,
            end_date=end_date,
            signal_names=signal_names,
            return_names=return_names,
            seed=seed,
        )

        self.signals = signals
        self.returns = returns
        self.realized_vol = realized_vol
        self.freq = freq

        return signals, returns, realized_vol

    def _require_simulated(self) -> None:
        if self.signals is None:
            raise ValueError(
                "No simulated data available. Call simulate_signals_and_returns first."
            )

    def quantamental_signals(self) -> pd.DataFrame:
        """convert signals to a quantemental dataframe"""
        self._require_simulated()
        return QuantamentalDataFrame.from_wide(self.signals)

    def quantamental_returns(self) -> pd.DataFrame:
        """convert returns to a quantemental dataframe"""
        self._require_simulated()
        return QuantamentalDataFrame.from_wide(100 * self.returns)

    def quantamental_returns_and_signals(self):
        signals = self.quantamental_signals()
        returns = self.quantamental_returns()
        return pd.concat((signals, returns), ignore_index=True)

    def realized_cov(self, freq: str = "BMS") -> pd.DataFrame:
        """
        Ground-truth realized covariance for each interval [dates[t], dates[t+1]].

            Sigma_m = (1/|m|) * sum_{t in m} D_t C D_t,   D_t = diag(realized_vol_t)

        annualized with the factor for the simulation frequency (252 for business
        days) and expressed in percent^2 (returns x100, variance x10,000), matching
        the units of the estimator VCV from `notional_positions`, which consumes
        the percent returns of `quantamental_returns`.
        Returned in long format (fid1, fid2, value, real_date) matching
        `stack_covariances`, with real_date set to the interval start so each block
        lines up with the estimate made on that rebalance date.
        """
        self._require_simulated()
        annualization = ANNUALIZATION_FACTORS[self.freq]
        names = np.asarray(self.realized_vol.columns)
        n = len(names)
        i, j = (
            grid.ravel()
            for grid in np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        )
        tri = i <= j
        i, j = i[tri], j[tri]

        frames = []
        dates = pd.date_range(
            start=self.realized_vol.index[0],
            end=self.realized_vol.index[-1],
            freq=freq,
        )
        for start_date, end_date in zip(dates, dates[1:]):
            vol = self.realized_vol.loc[start_date:end_date].to_numpy()  # (days, n)
            cov = (vol[:, :, None] * self.corr[None, :, :] * vol[:, None, :]).mean(
                axis=0
            )

            frames.append(
                pd.DataFrame(
                    {
                        "fid1": names[i],
                        "fid2": names[j],
                        "value": 10_000 * annualization * cov[i, j],
                        "real_date": start_date,
                    }
                )
            )

        return pd.concat(frames, ignore_index=True)



