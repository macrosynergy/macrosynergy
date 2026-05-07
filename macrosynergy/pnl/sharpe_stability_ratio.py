"""Sharpe Stability Ratio: HAC-robust t-stat for the mean rolling Sharpe, accounting for sample size and serial dependence."""

import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acovf
from typing import Optional, Union


def sharpe_stability_ratio(
    returns: Union[pd.Series, np.ndarray],
    window: int = 252,
    benchmark_sr: float = 0.0,
    annualization_factor: int = 252,
    min_periods: Optional[int] = None,
) -> float:
    """
    The SSR is the ratio between an average (rolling) Sharpe ratio and its
    estimated deviation from a true mean ("estimated parameter error"). Thus,
    it accounts not only for the risk-adjusted return of a strategy but also
    for its uncertainty given the sample size, seasonality, and autocorrelation.

    By default, Sharpe ratios and their estimated errors are calculated using
    rolling 252-trading-day windows on a daily return series. The error is
    estimated using the HAC (Heteroskedasticity and Autocorrelation Consistent)
    Newey-West (1987) variance estimator, which adjusts the variance of the
    sample mean for serial dependence in the rolling Sharpe series.

    The SSR can be interpreted as a signal-to-noise measure, indicating how
    strongly the Sharpe ratios deviate from zero after accounting for their
    serial dependence. If the SSR is above 1, its value exceeds its estimated
    noise, providing evidence of a non-zero mean. Under standard regularity
    conditions, the SSR approximates a t-statistic for the mean of the Sharpe
    ratio: a value of 1 corresponds to ~68% confidence in a non-zero mean,
    1.64 to ~90%, 1.96 to ~95%, and 2.58 to ~99%.

    For non-daily inputs, ``window`` and ``annualization_factor`` must be set
    together to match the input frequency (monthly: 12/12, weekly: 52/52).

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Return series at the frequency implied by ``window`` /
        ``annualization_factor``. NaNs are dropped.
    window : int, default 252
        Rolling window length (number of observations) for computing the
        Sharpe ratio series. Must be >= 2.
    benchmark_sr : float, default 0.0
        Benchmark Sharpe ratio (SR*) the mean rolling Sharpe is tested against.
    annualization_factor : int, default 252
        Periods per year. Must match input frequency.
    min_periods : int or None, default None
        Minimum non-NaN observations per window. Defaults to ``window``.

    Returns
    -------
    float
        SSR, or NaN if data is insufficient or the rolling Sharpe series has
        zero variance.
    """
    if not isinstance(window, int) or window < 2:
        raise ValueError("window must be an integer >= 2")
    if not isinstance(annualization_factor, int) or annualization_factor <= 0:
        raise ValueError("annualization_factor must be a positive integer")

    # Coerce to numpy, drop NaN
    if isinstance(returns, pd.Series):
        ret = returns.dropna().values.astype(float)
    else:
        ret = np.asarray(returns, dtype=float)
        ret = ret[~np.isnan(ret)]

    if len(ret) < window + 2:
        return float("nan")

    # Rolling Sharpe series (annualized)
    ret_s = pd.Series(ret)
    _min_p = min_periods if min_periods is not None else window
    roll = ret_s.rolling(window=window, min_periods=_min_p)
    z_series = (roll.mean() / roll.std()) * np.sqrt(annualization_factor)
    z = z_series.values.astype(float)
    z = z[np.isfinite(z)]  # drop NaN and ±inf (e.g. from zero-variance windows)

    N = len(z)
    if N < 3:
        return float("nan")

    # Set bandwidth to window; acovf requires nlag < nobs - 1
    L = window
    if N <= L + 1:
        min_obs = 2 * window + 2
        warnings.warn(
            f"Insufficient data: need at least {min_obs} observations "
            f"(~{min_obs / annualization_factor:.1f} years); returning NaN.",
            UserWarning,
            stacklevel=2,
        )
        return float("nan")

    z_bar = float(np.mean(z))

    # Newey-West long-run variance
    lrv = _newey_west_lrv(z, L)

    # HAC standard error of the mean
    hac_se = np.sqrt(lrv / N)
    if hac_se == 0.0:
        return float("nan")

    return float((z_bar - benchmark_sr) / hac_se)


def _newey_west_lrv(z: np.ndarray, L: int) -> float:
    """Newey-West (1987) Bartlett-kernel long-run variance with bandwidth ``L``."""
    gammas = acovf(z, nlag=L, fft=True, demean=True)
    lrv = gammas[0]
    for k in range(1, L + 1):
        weight = 1.0 - k / (L + 1.0)
        lrv += 2.0 * weight * gammas[k]
    return max(float(lrv), 0.0)


def _newey_west_heuristic_bandwidth(N: int) -> int:
    """Newey-West heuristic bandwidth: ``L = floor(4*(N/100)**(2/9))``, fixed in N."""
    if N < 3:
        return 1
    L = int(np.floor(4.0 * (N / 100.0) ** (2.0 / 9.0)))
    return max(1, min(L, N - 1))


def _andrews_ar1_bandwidth(z: np.ndarray) -> int:
    """Andrews (1991) AR(1) plug-in bandwidth — adapts to series persistence."""
    N = len(z)
    if N < 3:
        return 1

    z_dm = z - z.mean()

    # OLS AR(1): regress z_dm[1:] on z_dm[:-1]
    denom = float(z_dm[:-1] @ z_dm[:-1])
    if denom == 0.0:
        return 1
    rho = float(z_dm[:-1] @ z_dm[1:]) / denom
    rho = float(np.clip(rho, -0.99, 0.99))

    # Andrews (1991) Table 1, Bartlett kernel
    a1 = 4.0 * rho**2 / (1.0 - rho**2) ** 2
    L = int(np.floor(1.1447 * (a1 * N) ** (1.0 / 3.0)))

    return max(1, min(L, N - 1))


if __name__ == "__main__":


    def _sharpe(r: pd.Series, ann: int = 252) -> float:
        return float(r.mean() / r.std() * np.sqrt(ann))

    N = 3000  # ~12y of daily obs

    # ---- Pair A: smooth drift vs multi-year regime switching ----
    rng = np.random.default_rng(0)

    a_consistent = pd.Series(rng.normal(0.0008, 0.01, N))

    rng = np.random.default_rng(1)
    regime = np.zeros(N)
    regime_len = 504  # ~2y on / ~2y off
    for i, start in enumerate(range(0, N, regime_len)):
        if i % 2 == 0:
            regime[start:start + regime_len] = 0.0016  # 2x drift when "on"
    a_episodic = pd.Series(regime + rng.normal(0.0, 0.01, N))

    # ---- Pair B: lower-Sharpe pair, same idea ----
    rng = np.random.default_rng(2)
    b_consistent = pd.Series(rng.normal(0.0004, 0.01, N))

    rng = np.random.default_rng(3)
    regime = np.zeros(N)
    for i, start in enumerate(range(0, N, regime_len)):
        if i % 2 == 0:
            regime[start:start + regime_len] = 0.0008
    b_episodic = pd.Series(regime + rng.normal(0.0, 0.01, N))

    cases = [
        ("Pair A consistent (smooth drift)", a_consistent),
        ("Pair A episodic   (2y on/off)   ", a_episodic),
        ("Pair B consistent (smooth drift)", b_consistent),
        ("Pair B episodic   (2y on/off)   ", b_episodic),
    ]

    print(f"{'case':38s}   {'SR':>6s}   {'SSR':>7s}")
    print("-" * 60)
    for label, series in cases:
        sr = _sharpe(series)
        ssr = sharpe_stability_ratio(series)
        print(f"{label:38s}   {sr:6.3f}   {ssr:7.3f}")
