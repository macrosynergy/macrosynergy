from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from macrosynergy.management.constants import ANNUALIZATION_FACTORS
from macrosynergy.management.types import QuantamentalDataFrame
from scipy.special import gamma as gamma_fn


def _simulate_signals_and_returns(
    n_fids: int,
    n_periods: int,
    corr: np.ndarray,
    base_vol: np.ndarray,
    signal_ic: float,
    signal_autocorr: float,
    vol_persistence: float,
    vol_of_vol: float,
    mean_return: float,
    end_date: Optional[str] = None,
    signal_names: Optional[List[str]] = None,
    return_names: Optional[List[str]] = None,
    freq: str = "B",
    ic_magnitude_gamma: float = 0.0,
    seed: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Simulate trading signals and asset returns with controllable covariance
    and realistic (time-varying, persistent) volatility.

    Returns are drawn from a multivariate distribution whose *correlation*
    structure is fixed by `corr`, while each asset's instantaneous volatility
    follows a GARCH-like (EWMA) process so it clusters rather than being
    constant. This is well suited to testing vol-targeting code.

    Parameters
    ----------
    n_fids : int
        Number of fids to generate signals and returns for.
    n_periods : int
        Number of time steps to simulate.
    corr : (n_assets, n_assets) array-like, optional
        Target correlation matrix of returns. Must be symmetric PSD with unit
        diagonal. If None, a fixed random correlation matrix is generated.
    base_vol : (n_assets,) array-like, optional
        Long-run per-period volatility for each asset (the level vols revert
        to). If None, defaults to 0.01 (i.e. 1% per period) for every asset.
    signal_ic : float
        Target information coefficient: correlation between the signal at t and
        the return at t+1. Controls how predictive the signals are.
    signal_autocorr : float
        AR(1) coefficient for signal persistence (0 = white noise, ->1 = slow).
    vol_persistence : float
        EWMA decay on the variance process (closer to 1 = stickier vol regimes).
    vol_of_vol : float
        Magnitude of shocks to the variance process (higher = more vol
        clustering / spikiness).
    mean_return : float or (n_assets,) array-like
        Per-period drift added to returns.
    end_date : str or datetime
        Last date of the index.
    signal_names : list of str, optional
        Column names for the signals. Defaults to ['sig0', 'sig1', ...].
    return_names : list of str, optional
        Column names for the returns. Defaults to ['ret0', 'ret1', ...].
    freq : str
        Pandas frequency for the date index. Defaults to 'B' (business days).
        Use 'D' for calendar days.
    ic_magnitude_gamma : float
        Controls how much predictive strength scales with signal magnitude.
        0.0 (default) reproduces the original constant-IC behaviour exactly:
        every signal value is equally reliable. With gamma > 0 the local IC
        grows like |signal|^gamma, so larger-magnitude signals are more
        reliable predictors, while the *overall* corr(signal_t, return_t+1)
        still targets `signal_ic`. The scaling constant is seeded analytically
        from the Gaussian absolute moment E[|s|^gamma] and refined by a
        slope estimate measured on the simulated sample, so the realized IC
        matches the target up to sampling noise (as in the gamma == 0 case).
    seed : int, optional
        RNG seed for reproducibility.

    Returns
    -------
    signals : (n_periods, n_assets) DataFrame
        Signal for each asset/period, indexed by date. signals.iloc[t] is
        predictive of returns.iloc[t+1].
    returns : (n_periods, n_assets) DataFrame
        Simulated returns, indexed by date.
    realized_vol : (n_periods, n_assets) DataFrame
        The instantaneous (true) volatility used at each step, indexed by date.
        Useful as a ground-truth benchmark for vol-targeting tests.
    """
    rng = np.random.default_rng(seed)

    if end_date is None:
        end_date = pd.Timestamp.now().strftime("%Y-%m-%d")

    if corr.shape != (n_fids, n_fids):
        raise ValueError("corr must be (n_assets, n_assets)")

    base_vol = np.broadcast_to(np.asarray(base_vol, dtype=float), (n_fids,)).copy()
    mean_return = np.broadcast_to(
        np.asarray(mean_return, dtype=float), (n_fids,)
    ).copy()

    # Cholesky of the correlation matrix to inject cross-sectional dependence.
    # Correlation (not covariance) is held fixed; vols vary over time, so the
    # covariance at each step is diag(vol_t) @ corr @ diag(vol_t).
    L = np.linalg.cholesky(corr)

    # Time-varying volatility via a log-variance EWMA / GARCH-like process.
    # log-variance mean-reverts to log(base_vol^2) with persistent shocks,
    # producing volatility clustering rather than constant vol.
    log_var = np.empty((n_periods, n_fids))
    long_run_log_var = np.log(base_vol**2)
    state = long_run_log_var.copy()
    for t in range(n_periods):
        shock = rng.standard_normal(n_fids) * vol_of_vol
        state = (
            vol_persistence * state + (1.0 - vol_persistence) * long_run_log_var + shock
        )
        log_var[t] = state
    realized_vol = np.sqrt(np.exp(log_var))

    # Correlated standardized innovations.
    z = rng.standard_normal((n_periods, n_fids)) @ L.T  # unit-vol, corr = corr

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

    # Combine so that corr(signals[t], z[t+1]) == ic, keeping the mix stable.
    # If ic_magnitude_gamma == 0 this is the original constant-IC mix. Otherwise
    # the local IC b(s) = k * |s|^gamma scales predictiveness with magnitude:
    # larger-|signal| observations become more reliable predictors.
    #
    # Because b depends only on s and s is independent of z,
    #   corr(aligned, z) ~= E[b(s)] = k * E[|s|^gamma],
    # so k is seeded from the Gaussian absolute moment
    #   E[|s|^p] = 2^(p/2) * Gamma((p+1)/2) / sqrt(pi),  p = gamma.
    #
    # The seed is then refined against the sample. The measured IC of the mix
    # is slope * k plus a noise intercept: the spurious sample correlation
    # between `persistent` and z, present even at k = 0. Measuring that
    # intercept separately and differencing isolates the slope, and
    # k = ic / slope targets the population IC. A plain ratio rescale
    # (k * ic / measured) misattributes the intercept to the slope and flips
    # the sign of k whenever small-sample noise dominates the seed pass.
    if ic_magnitude_gamma == 0.0:
        b = ic
        aligned = b * z + np.sqrt(1.0 - b ** 2) * persistent
    elif ic == 0.0:
        aligned = persistent
    else:
        p = ic_magnitude_gamma
        gaussian_abs_moment = (
            2.0 ** (p / 2.0) * gamma_fn((p + 1.0) / 2.0) / np.sqrt(np.pi)
        )
        k = ic / gaussian_abs_moment
        shape = np.abs(persistent) ** ic_magnitude_gamma

        def measured_ic(mix: np.ndarray) -> np.ndarray:
            # Same-time corr with z over the periods the emitted signals
            # cover (aligned[0] is dropped by the roll, z[0] predicts nothing).
            return np.array(
                [np.corrcoef(mix[1:, j], z[1:, j])[0, 1] for j in range(n_fids)]
            )

        b = np.clip(k * shape, -0.999, 0.999)
        seed_mix = b * z + np.sqrt(1.0 - b ** 2) * persistent
        noise_intercept = measured_ic(persistent)  # what k = 0 would measure
        slope = (measured_ic(seed_mix) - noise_intercept) / k

        # slope ~= E[|s|^gamma] > 0; fall back to the analytic seed in the
        # pathological case where the sample estimate is unusable.
        slope_is_usable = np.isfinite(slope) & (slope > 0.0)
        k_adj = np.where(
            slope_is_usable, ic / np.where(slope_is_usable, slope, 1.0), k
        )
        b = np.clip(k_adj * shape, -0.999, 0.999)
        aligned = b * z + np.sqrt(1.0 - b ** 2) * persistent

    # aligned[t] is correlated with z[t]. We want signals[t] to predict
    # returns[t+1], i.e. signals[t] must carry information about z[t+1].
    # Shift backward by one so signals[t] = aligned[t+1].
    signals = np.roll(aligned, -1, axis=0)
    signals[-1] = 0.0  # last signal predicts an unobserved future return

    # Returns: drift + time-varying vol * correlated innovation.
    returns = mean_return + realized_vol * z

    # Wrap in DataFrames with a daily (business-day) index.
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
        ic_magnitude_gamma: float = 0.0,
    ) -> None:
        self.n_fids = n_fids
        self.corr = None if corr is None else np.asarray(corr, dtype=float)
        self.base_vol = base_vol
        self.signal_ic = signal_ic
        self.signal_autocorr = signal_autocorr
        self.vol_persistence = vol_persistence
        self.vol_of_vol = vol_of_vol
        self.mean_return = mean_return
        self.ic_magnitude_gamma = ic_magnitude_gamma

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
            ic_magnitude_gamma=self.ic_magnitude_gamma,
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
        days), matching the units of the estimator VCV from `notional_positions`.
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
                        "value": annualization * cov[i, j],
                        "real_date": start_date,
                    }
                )
            )

        return pd.concat(frames, ignore_index=True)



