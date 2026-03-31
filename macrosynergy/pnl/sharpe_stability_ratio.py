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

    Notes
    -----
    The Andrews (1991) AR(1) plug-in bandwidth formula for the Bartlett kernel
    is used for automatic lag selection:

        L = floor(1.1447 * (4*rho^2 / (1-rho^2)^2 * N)^(1/3))
        clamped to [1, N-1]

    where rho is the first-order autocorrelation of the rolling SR series and
    N is the number of rolling SR observations.

    High SSR values indicate consistent risk-adjusted returns across subperiods
    (persistent skill), while low values suggest episodic outperformance
    concentrated in a few favorable windows.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from macrosynergy.pnl import sharpe_stability_ratio
    >>> np.random.seed(0)
    >>> returns = pd.Series(np.random.normal(0.001, 0.01, 1500))
    >>> sharpe_stability_ratio(returns)
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

    # Andrews automatic bandwidth
    L = _andrews_ar1_bandwidth(z)

    # Newey-West long-run variance
    lrv = _newey_west_lrv(z, L)

    # HAC standard error of the mean
    hac_se = np.sqrt(lrv / N)
    if hac_se == 0.0:
        return float("nan")

    return float((z_bar - benchmark_sr) / hac_se)


if __name__ == "__main__":
    np.random.seed(42)

    # 1. Consistent positive returns — should yield high SSR
    consistent = pd.Series(np.random.normal(0.001, 0.01, 2000))
    ssr_consistent = sharpe_stability_ratio(consistent)
    print(f"Consistent positive returns  SSR: {ssr_consistent:.3f}")

    # 2. Zero-mean noise — should yield SSR near 0
    noise = pd.Series(np.random.normal(0.0, 0.01, 2000))
    ssr_noise = sharpe_stability_ratio(noise)
    print(f"Zero-mean noise              SSR: {ssr_noise:.3f}")

    # 3. Episodic: good first half, bad second half — lower SSR than consistent
    episodic = pd.Series(
        np.concatenate([
            np.random.normal(0.002, 0.01, 1000),
            np.random.normal(-0.001, 0.01, 1000),
        ])
    )
    ssr_episodic = sharpe_stability_ratio(episodic)
    print(f"Episodic (good/bad halves)   SSR: {ssr_episodic:.3f}")

    # 4. Test benchmark_sr — same consistent series but tested against SR*=1.0
    ssr_vs_hurdle = sharpe_stability_ratio(consistent, benchmark_sr=1.0)
    print(f"Consistent vs hurdle SR=1.0  SSR: {ssr_vs_hurdle:.3f}")
