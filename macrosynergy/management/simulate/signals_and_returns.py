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
    r"""
    Simulate time-varying volatility via a log-variance EWMA / GARCH-like process.

    Parameters
    ----------
    n_periods : int
        number of periods (rows) to simulate.
    n_fids : int
        number of financial contract identifiers (columns) to simulate.
    base_vol : np.ndarray
        (n_fids,) array of long-run per-period volatilities, to which the
        log-variance reverts. Must be strictly positive.
    vol_persistence : float
        AR(1) coefficient of the log-variance process, between 0 and 1. Values close
        to 1 give long-lived volatility regimes.
    vol_of_vol : float
        standard deviation of the shocks to the log-variance.
    rng : Optional[np.random.Generator]
        random number generator. Default is None, in which case a fresh unseeded
        generator is used.

    Returns
    -------
    np.ndarray
        (n_periods, n_fids) array of per-period realized volatilities.

    Notes
    -----
    For each contract, the log-variance mean-reverts to
    :math:`\log \bar{\sigma}^{2}` with persistent shocks, producing volatility
    clustering rather than constant volatility:

    .. math::

        \log \sigma_{t}^{2} = \phi \log \sigma_{t-1}^{2}
        + (1 - \phi) \log \bar{\sigma}^{2} + \eta_{t},
        \qquad \eta_{t} \sim N(0, \omega^{2})

    where :math:`\bar{\sigma}` is `base_vol`, :math:`\phi` is `vol_persistence` and
    :math:`\omega` is `vol_of_vol`. The returned volatilities are
    :math:`\sigma_{t} = \sqrt{\exp(\log \sigma_{t}^{2})}`. The process is initialised
    at its long-run level, :math:`\log \sigma_{0}^{2} = \log \bar{\sigma}^{2}`.
    """

    rng = rng or np.random.default_rng()

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
    r"""
    Simulate cross-sectionally correlated returns along a given volatility path.

    Parameters
    ----------
    n_periods : int
        number of periods (rows) to simulate.
    n_fids : int
        number of financial contract identifiers (columns) to simulate.
    corr : np.ndarray
        (n_fids, n_fids) correlation matrix of the return innovations. Must be
        symmetric positive definite with unit diagonal.
    mean_return : np.ndarray
        (n_fids,) array of per-period mean returns.
    realized_vol : np.ndarray
        (n_periods, n_fids) array of per-period volatilities, as returned by
        `_simulate_volatility`.
    rng : Optional[np.random.Generator]
        random number generator. Default is None, in which case a fresh unseeded
        generator is used.
    return_z : bool
        if True, also return the underlying unit-variance correlated innovations,
        which are needed to construct signals with a target information coefficient.
        Default is False.

    Returns
    -------
    np.ndarray or Tuple[np.ndarray, np.ndarray]
        (n_periods, n_fids) array of returns; if `return_z` is True, a tuple of the
        returns and the (n_periods, n_fids) array of innovations `z`.

    Notes
    -----
    Rerurns are computed according to the following formula:

    .. math::

        r_{t} = \mu + D_{t} z_{t},
        \qquad z_{t} = L\xi_{t},
        \qquad \xi_{t} \sim N(0, I),
        \qquad C = LL^\top,
        \qquad D_{t} = \mathrm{diag}(\sigma_{1,t}, \dots, \sigma_{N,t})

    where :math:`\mu` is `mean_return`, :math:`C` is `corr` and :math:`\sigma_{i,t}` is
    `realized_vol`
    """

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
    r"""
    Simulate persistent signals correlated with the next period's return.

    Parameters
    ----------
    n_periods : int
        number of periods (rows) to simulate.
    n_fids : int
        number of financial contract identifiers (columns) to simulate.
    signal_ic : float
        target information coefficient, i.e. `corr(signals[t], z[t + 1])`. Clipped to
        the interval (-0.999, 0.999).
    signal_autocorr : float
        AR(1) coefficient of the persistent component, strictly between -1 and 1.
    z : np.ndarray
        (n_periods, n_fids) array of unit-variance return innovations, as returned by
        `_simulate_returns` with `return_z=True`.
    rng : Optional[np.random.Generator]
        random number generator. Unlike the other simulation helpers this argument has
        no fallback, so a generator must be passed.

    Returns
    -------
    np.ndarray
        (n_periods, n_fids) array of approximately unit-variance signals.

    Notes
    -----
    A unit-variance AR(1) component is built first,

    .. math::

        p_{t} = a p_{t-1} + \sqrt{1 - a^{2}} \, \epsilon_{t},
        \qquad \epsilon__{t} \sim N(0, I)

    and then mixed with the innovation it is meant to predict, so that the realized
    information coefficient hits the target regardless of the persistence level:

    .. math::

        \tilde{s}_{t} = \rho z_{t} + \sqrt{1 - \rho^{2}} \, p_{t},
        \qquad \mathrm{corr}(\tilde{s}_{t}, z_{t}) = \rho

    where :math:`a` is `signal_autocorr` and :math:`\rho` is `signal_ic`. The mixture
    is then shifted back by one period, so that a signal observed at :math:`t` predicts
    the return of :math:`t + 1`:

    .. math::

        s_{t} = \tilde{s}_{t+1},
        \qquad \mathrm{corr}(s_{t}, z_{t+1}) = \rho,
        \qquad s_{T} = 0

    The last row is set to zero, as it would otherwise predict an unobserved future
    return.
    """

    ic = float(np.clip(signal_ic, -0.999, 0.999))
    a = signal_autocorr

    persistent = np.empty((n_periods, n_fids))
    s = np.zeros(n_fids)
    eps = rng.standard_normal((n_periods, n_fids))
    for t in range(n_periods):
        s = a * s + np.sqrt(1.0 - a**2) * eps[t]
        persistent[t] = s  # unit-variance AR(1), independent of z

    b = ic
    aligned = b * z + np.sqrt(1.0 - b**2) * persistent

    # aligned[t] is correlated with z[t]. We want signals[t] to predict
    # returns[t+1], i.e. signals[t] must carry information about z[t+1].
    # Shift backward by one so signals[t] = aligned[t+1].
    signals = np.roll(aligned, -1, axis=0)
    signals[-1] = 0.0  # last signal predicts an unobserved future return

    return signals


def _simulate_signals_and_returns(
    n_fids: int,
    n_periods: int,
    base_vol: np.ndarray,
    vol_persistence: float,
    vol_of_vol: float,
    corr: np.ndarray,
    mean_return: float,
    signal_autocorr: float,
    signal_ic: float,
    end_date: Optional[str] = None,
    signal_names: Optional[List[str]] = None,
    return_names: Optional[List[str]] = None,
    freq: str = "B",
    seed: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Simulate signals, returns and their volatility path as wide DataFrames.

    Chains `_simulate_volatility`, `_simulate_returns` and `_simulate_signals` off a
    single seeded generator, so that the returns of each period have covariance
    `D_t C D_t` (with `D_t` the diagonal matrix of that period's volatilities) and each
    signal predicts the *next* period's return innovation with correlation
    `signal_ic`.

    Parameters
    ----------
    n_fids : int
        number of financial contract identifiers (columns) to simulate.
    n_periods : int
        number of periods (rows) to simulate.
    base_vol : np.ndarray
        long-run per-period volatilities, broadcast to (n_fids,). Must be strictly
        positive.
    vol_persistence : float
        AR(1) coefficient of the log-variance process, between 0 and 1.
    vol_of_vol : float
        standard deviation of the shocks to the log-variance.
    corr : np.ndarray
        (n_fids, n_fids) correlation matrix of the return innovations. Must be
        symmetric positive definite with unit diagonal.
    mean_return : float
        per-period mean return, broadcast to (n_fids,).
    signal_autocorr : float
        AR(1) coefficient of the persistent signal component, strictly between -1 and
        1.
    signal_ic : float
        target information coefficient of the signals against the next period's return
        innovations.
    end_date : Optional[str]
        last date of the simulated index, in ISO format. Default is None, in which case
        today's date is used.
    signal_names : Optional[List[str]]
        column names of the signals DataFrame. Default is None, in which case
        `CID{i}_SIG` is used.
    return_names : Optional[List[str]]
        column names of the returns and volatility DataFrames. Default is None, in
        which case `CID{i}_XR` is used.
    freq : str
        pandas frequency of the simulated index. Default is "B" (business days).
    seed : Optional[int]
        seed of the random number generator. Default is None, i.e. unseeded.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        wide DataFrames of signals, returns and realized volatilities, each indexed by
        the simulated dates. Returns and volatilities are expressed as fractions, not
        percentages.
    """

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
    mean_return = np.broadcast_to(
        np.asarray(mean_return, dtype=float), (n_fids,)
    ).copy()

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
    """
    Generates mock signals and returns with known ground-truth risk characteristics.

    Holds the data generating process on the instance, so that a simulation can be
    requested once and then served in the formats consumed by the package - wide
    DataFrames, quantamental DataFrames, or the realized covariance matrices the
    portfolio construction code estimates.

    Parameters
    ----------
    n_fids : int
        number of financial contract identifiers to simulate.
    corr : np.ndarray
        (n_fids, n_fids) correlation matrix of the return innovations. Default is None,
        in which case a random (but deterministic) correlation matrix is drawn.
    base_vol : np.ndarray
        long-run per-period volatilities, broadcast to (n_fids,). Default is None,
        in which case 0.01 per period is used for every contract, i.e. roughly 16%
        annualized at business-day frequency.
    signal_ic : float
        target information coefficient of the signals against the next period's return
        innovations. Default is 0.05.
    signal_autocorr : float
        AR(1) coefficient of the persistent signal component, strictly between -1 and
        1. Default is 0.9.
    vol_persistence : float
        AR(1) coefficient of the log-variance process, between 0 and 1. Default is
        0.94.
    vol_of_vol : float
        standard deviation of the shocks to the log-variance. Default is 0.15.
    mean_return : float
        per-period mean return, broadcast to (n_fids,). Default is 0.0.
    """

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
        """
        Simulate signals and returns using the parameters held on the instance.

        The results are stored on the instance, so that the conversion methods can be
        called afterwards. Calling this method again overwrites the previous
        simulation.

        Parameters
        ----------
        n_periods : int
            number of periods (rows) to simulate.
        end_date : Optional[str]
            last date of the simulated index, in ISO format. Default is None, in which
            case today's date is used.
        signal_names : Optional[List[str]]
            column names of the signals DataFrame. Default is None, in which case
            `CID{i}_SIG` is used.
        return_names : Optional[List[str]]
            column names of the returns and volatility DataFrames. Default is None, in
            which case `CID{i}_XR` is used.
        freq : str
            pandas frequency of the simulated index. Default is "B" (business days).
        seed : int
            seed of the random number generator. Default is 29.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
            wide DataFrames of signals, returns and realized volatilities, each indexed
            by the simulated dates. Returns and volatilities are expressed as
            fractions, not percentages.
        """

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
        """
        Convert the simulated signals to a quantamental DataFrame.
        """
        self._require_simulated()
        return QuantamentalDataFrame.from_wide(self.signals)

    def quantamental_returns(self) -> pd.DataFrame:
        """
        Convert the simulated returns to a quantamental DataFrame.

        The returns are scaled by 100, i.e. expressed in percent, matching the units of
        JPMaQS return categories.
        """
        self._require_simulated()
        return QuantamentalDataFrame.from_wide(100 * self.returns)

    def quantamental_returns_and_signals(self):
        """
        Concatenate the quantamental signals and returns into a single DataFrame.
        """
        signals = self.quantamental_signals()
        returns = self.quantamental_returns()
        return pd.concat((signals, returns), ignore_index=True)

    def realized_cov(self, freq: str = "BMS", long: bool = True) -> pd.DataFrame:
        r"""
        Ground-truth realized covariance for each rebalance interval.

        Parameters
        ----------
        freq : str
            pandas frequency of the rebalance dates delimiting the intervals. Default
            is "BMS" (business month starts). The trailing partial interval is
            dropped, as it has no closing rebalance date.
        long : bool
            if True, return a single long DataFrame holding the upper triangle of each
            covariance matrix. If False, return a dictionary of the full matrices keyed
            by interval start. Default is True.

        Returns
        -------
        pd.DataFrame or Dict[pd.Timestamp, np.ndarray]
            long DataFrame with columns 'fid1', 'fid2', 'value' and 'real_date'; or, if
            `long` is False, a dictionary mapping each interval start to the
            (n_fids, n_fids) covariance matrix of that interval.

        Notes
        -----
        For each interval :math:`m` spanning the dates :math:`[d_{k}, d_{k+1}]`,
        the realized covariance is the average of the per-period covariances
        of `_simulate_returns` over that interval:

        .. math::

            \Sigma_{m} = \frac{1}{|m|} \sum_{t \in m} D_{t} C D_{t},
            \qquad D_{t} = \mathrm{diag}(\sigma_{1,t}, \dots, \sigma_{N,t})

        The result is annualized with the factor for the simulation frequency
        (252 for business days) and expressed in percent squared - returns are scaled
        by 100, hence variances by 10,000:

        .. math::

            \Sigma_{m}^{\text{ann}} = 10^{4} \times A \times \Sigma_{m}

        This matches the units of the estimator VCV from `notional_positions`, which
        consumes the percent returns of `quantamental_returns`. The long format
        (fid1, fid2, value, real_date) matches `stack_covariances`, with `real_date`
        set to the interval start :math:`d_{k}`, so each block lines up with the
        estimate made on that rebalance date.
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

        out = [] if long else {}
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

            if long:
                out.append(
                    pd.DataFrame(
                        {
                            "fid1": names[i],
                            "fid2": names[j],
                            "value": 10_000 * annualization * cov[i, j],
                            "real_date": start_date,
                        }
                    )
                )
            else:
                out[start_date] = 10_000 * annualization * cov

        return pd.concat(out, ignore_index=True) if long else out
