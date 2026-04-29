"""
Sharpe Stability Ratio (SSR)

Implements the Sharpe Stability Ratio as described in:
    Bajo Traver & Rodríguez Domínguez (2026), "The Sharpe Stability Ratio:
    Temporal Consistency of Risk-Adjusted Performance".

SSR measures the temporal consistency of risk-adjusted returns. It is defined
as the HAC-robust t-statistic for the hypothesis that the mean of the rolling
Sharpe ratio series equals a benchmark value:

    SSR = (Z_bar - SR*) / HAC_SE(Z)

where Z is the rolling Sharpe ratio series, Z_bar its sample mean, SR* a
benchmark Sharpe ratio (default 0), and HAC_SE is the Newey-West HAC standard
error of the mean, i.e. sqrt(LRV / N).
"""

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
    Compute the Sharpe Stability Ratio (SSR) for a return series.

    SSR measures the temporal consistency of risk-adjusted returns. It is the
    HAC-robust t-statistic for the hypothesis that the mean of the rolling
    Sharpe ratio series equals a benchmark Sharpe ratio:

        SSR = (Z_bar - SR*) / HAC_SE(Z)

    where Z is the rolling Sharpe ratio series with window length ``window``,
    Z_bar its sample mean, SR* is ``benchmark_sr``, and HAC_SE is the
    Newey-West HAC standard error of the mean (sqrt(LRV / N)).

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Daily return series. NaN values are dropped before computation.
    window : int, default 252
        Rolling window length (number of observations) for computing the
        Sharpe ratio series. Must be >= 2.
    benchmark_sr : float, default 0.0
        Benchmark Sharpe ratio (SR*) against which the mean rolling SR is
        tested. The default of 0.0 tests for positive temporal consistency.
    annualization_factor : int, default 252
        Number of business days per year for annualizing the rolling Sharpe
        ratios.
    min_periods : int or None, default None
        Minimum non-NaN observations required in each rolling window. If None,
        defaults to ``window`` (requires a fully-filled window).

    Returns
    -------
    float
        The SSR scalar. Returns np.nan if there is insufficient data or if
        the rolling SR series has zero variance.
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

    z_bar = float(np.mean(z))

    # Newey-West rule-of-thumb bandwidth
    # L = _newey_west_rule_of_thumb_bandwidth(N)
    L = window

    # Newey-West long-run variance
    lrv = _newey_west_lrv(z, L)

    # HAC standard error of the mean
    hac_se = np.sqrt(lrv / N)
    if hac_se == 0.0:
        return float("nan")

    return float((z_bar - benchmark_sr) / hac_se)


def _newey_west_lrv(z: np.ndarray, L: int) -> float:
    """
    Newey-West (1987) long-run variance with Bartlett kernel.

    LRV = gamma_0 + 2 * sum_{k=1}^{L} (1 - k/(L+1)) * gamma_k

    Parameters
    ----------
    z : np.ndarray
        1-D array (the rolling Sharpe ratio series).
    L : int
        Bandwidth (number of lags to include).

    Returns
    -------
    float
        Long-run variance estimate, floored at 0.0.
    """
    gammas = acovf(z, nlag=L, fft=True, demean=True)
    lrv = gammas[0]
    for k in range(1, L + 1):
        weight = 1.0 - k / (L + 1.0)
        lrv += 2.0 * weight * gammas[k]
    return max(float(lrv), 0.0)


def _newey_west_rule_of_thumb_bandwidth(N: int) -> int:
    """
    Newey-West rule-of-thumb bandwidth for the Bartlett kernel:

        L = floor(4 * (N / 100) ** (2 / 9))

    clamped to [1, N-1].

    Parameters
    ----------
    N : int
        Length of the series.

    Returns
    -------
    int
        Bandwidth parameter L >= 1.
    """
    if N < 3:
        return 1
    L = int(np.floor(4.0 * (N / 100.0) ** (2.0 / 9.0)))
    return max(1, min(L, N - 1))


def _andrews_ar1_bandwidth(z: np.ndarray) -> int:
    """
    Andrews (1991) AR(1) plug-in bandwidth for the Bartlett kernel.

    Fits an AR(1) model on the demeaned series and returns the bandwidth L
    based on the formula in Andrews (1991), Table 1:

        alpha_1 = 4 * rho^2 / (1 - rho^2)^2
        L = floor(1.1447 * (alpha_1 * N)^(1/3))

    clamped to [1, N-1].

    Parameters
    ----------
    z : np.ndarray
        1-D array of the rolling Sharpe ratio series.

    Returns
    -------
    int
        Bandwidth parameter L >= 1.
    """
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
    # The point of SSR: two strategies with similar realized Sharpes can have
    # very different temporal consistency. SSR separates them; Sharpe alone
    # cannot. Two paired demos below — each pair has matched Sharpe, contrasting
    # SSR.

    def _sharpe(r: pd.Series, ann: int = 252) -> float:
        return float(r.mean() / r.std() * np.sqrt(ann))

    N = 3000  # ~12y of daily obs

    # ---- Pair A: smooth drift vs multi-year regime switching ----
    # Both target Sharpe ~ 1.25 over the full sample.
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
