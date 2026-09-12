import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from macrosynergy.management.simulate import SignalsAndReturnsGenerator
from macrosynergy.pnl.historic_portfolio_volatility import (
    _check_est_args,
    _cov_matrix_history,
    expo_weights_arr,
    flat_weights_arr,
)

logger = logging.getLogger(__name__)

CONFIG_KEYS = {"est_freqs", "est_weights", "lback_meth", "lback_periods", "half_life"}


def _long_cov_to_dict(
    cov_long: pd.DataFrame,
    fids: List[str] = None,
    check_psd: bool = True,
    psd_tol: float = 1e-8,
) -> Dict[pd.Timestamp, np.ndarray]:
    """
    Convert a long covariance frame into the date-keyed matrices this module consumes.
    """
    required = {"fid1", "fid2", "real_date", "value"}
    missing = required - set(cov_long.columns)
    if missing:
        raise ValueError(f"long cov df missing columns: {sorted(missing)}")

    if fids is None:
        fid1s = cov_long["fid1"].unique()
        fid2s = cov_long["fid2"].unique()
        fids = sorted(set(fid1s).union(fid2s))

    n_fids = len(fids)
    fid_pos = {fid: i for i, fid in enumerate(fids)}

    out = {}
    for date, grp in cov_long.groupby("real_date", sort=True):
        cov = np.full(shape=(n_fids, n_fids), fill_value=np.nan)

        # vectorised fill of both triangles from the upper-triangle rows
        i = grp["fid1"].map(fid_pos).to_numpy()
        j = grp["fid2"].map(fid_pos).to_numpy()
        v = grp["value"].to_numpy(dtype=float)
        cov[i, j] = v
        cov[j, i] = v  # mirror; diagonal rows (i == j) simply overwrite themselves

        if np.isnan(cov).any():
            n_missing = int(np.isnan(cov).sum())
            raise ValueError(f"{date}: matrix has {n_missing} unfilled entries")

        if check_psd:
            # symmetry is exact by construction; check PSD
            min_eig = np.linalg.eigvalsh(cov).min()
            if min_eig < -psd_tol:
                raise ValueError(
                    f"{date}: covariance not PSD (min eigenvalue {min_eig:.3e})"
                )

        out[date] = cov

    return out


def _min_var_weights(cov: np.ndarray) -> np.ndarray:
    r"""
    Minimum-variance weights, :math:`w \propto \Sigma^{-1} \mathbf{1}`, summing to 1.

    Parameters
    ----------
    cov : np.ndarray
        (n_fids, n_fids) covariance matrix. Normally positive definite; a rank-deficient
        or near-singular one is handled by the fallback described below.
    """
    ones = np.ones(cov.shape[0])
    try:
        w = np.linalg.solve(cov, ones)
        if not np.all(np.isfinite(w)):
            raise np.linalg.LinAlgError
    except np.linalg.LinAlgError:
        ridge = 1e-6 * np.trace(cov) / cov.shape[0]
        w = np.linalg.solve(cov + ridge * np.eye(cov.shape[0]), ones)
    return w / w.sum()


def realized_to_forecast_vol_ratios(
    cov_true: Dict[pd.Timestamp, np.ndarray],
    cov_ests: List[Dict[pd.Timestamp, np.ndarray]],
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    sqrt(w' cov_true w / w' cov_est w) per true date (rows) and estimator (columns).

    Parameters
    ----------
    cov_true : Dict[pd.Timestamp, np.ndarray]
        the covariance being forecast, as one (n_fids, n_fids) matrix per date. Its
        dates, sorted, are the rows of the result.
    cov_ests : List[Dict[pd.Timestamp, np.ndarray]]
        one such mapping per estimator under comparison, on the same fid axis order as
        `cov_true`. An estimator need not cover every true date; a date it is missing,
        or one whose matrix holds NaN, is left NaN in that estimator's column.
    weights : Optional[np.ndarray]
        portfolio weights, held fixed across dates and estimators. Default is None, in
        which case minimum-variance weights are derived from the *true* covariance of
        each date.

    Returns
    -------
    np.ndarray
        (n_true_dates, n_estimators) array of ratios, rows in sorted date order.
    """
    # a long frame has `.keys()` too - it would run on the column names and fail much
    # later, inside a quadratic form, with a shape mismatch that names nothing
    if isinstance(cov_true, pd.DataFrame) or any(
        isinstance(cov_est, pd.DataFrame) for cov_est in cov_ests
    ):
        raise TypeError(
            "`cov_true` and `cov_ests` take date-keyed covariance matrices, not long "
            "frames. Convert a long frame with `_long_cov_to_dict` first."
        )

    true_dates = sorted(cov_true.keys())
    weights_by_date = {
        date: weights if weights is not None else _min_var_weights(cov_true[date])
        for date in true_dates
    }

    # row i is true_dates[i] in every column, so entries stay comparable
    # across estimators with different date coverage
    ratios = np.full(shape=(len(true_dates), len(cov_ests)), fill_value=np.nan)
    for j, cov_est in enumerate(cov_ests):
        for i, date in enumerate(true_dates):
            est: Optional[np.ndarray] = cov_est.get(date)
            if est is None:
                continue
            truth: np.ndarray = cov_true[date]

            w = weights_by_date[date]

            # A degenerate estimate (e.g. monthly sampling, many assets) can give
            # a non-positive quadratic form; guard it so the entry stays NaN
            forecast_var = w @ est @ w
            if not np.isfinite(forecast_var) or forecast_var <= 0:
                continue

            ratios[i, j] = np.sqrt(w @ truth @ w / forecast_var)

    return ratios


def _resolve_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate one estimator config and expand it into `_cov_matrix_history` arguments.

    Parameters
    ----------
    config : Dict[str, Any]
        `est_freqs` and `lback_periods` are required, as is `lback_meth` ("ma" for flat
        weights, "xma" for exponential; case-insensitive). `half_life` is required under
        "xma" and ignored under "ma". `est_weights` defaults to equal weights.

    Returns
    -------
    Dict[str, Any]
        keyword arguments for `_cov_matrix_history`: the five checked lists plus
        `weights_func`.
    """
    unknown = set(config) - CONFIG_KEYS
    if unknown:
        raise ValueError(
            f"unknown key(s) in estimator config: {sorted(unknown)}. "
            f"Expected some of: {sorted(CONFIG_KEYS)}."
        )

    missing = {"est_freqs", "lback_periods", "lback_meth"} - set(config)
    if missing:
        raise ValueError(f"estimator config is missing {sorted(missing)}: {config}")

    lback_meth = str(config["lback_meth"]).lower()
    if lback_meth not in ("ma", "xma"):
        raise ValueError(
            f"`lback_meth` must be 'ma' or 'xma'; got {config['lback_meth']!r}"
        )

    if lback_meth == "xma" and "half_life" not in config:
        raise ValueError(
            "`lback_meth='xma'` needs an explicit `half_life`. Defaulting it to "
            "`lback_periods` would truncate the window at the point where half the "
            "weight is still outstanding, which is not an estimator anyone asked for."
        )

    # `flat_weights_arr` ignores `half_life`, but `_weighted_covariance` asserts it is
    # positive, so "ma" needs a placeholder. It cannot be `lback_periods`, which is -1
    # when all available history is wanted.
    half_life = config.get("half_life", [1])

    est_freqs, est_weights, lback_periods, half_life, lback_min_obs = _check_est_args(
        est_freqs=config["est_freqs"],
        est_weights=config.get("est_weights", [1]),
        lback_periods=config["lback_periods"],
        half_life=half_life,
        lback_min_obs=[1],
    )

    return dict(
        est_freqs=est_freqs,
        est_weights=est_weights,
        lback_periods=lback_periods,
        half_life=half_life,
        lback_min_obs=lback_min_obs,
        weights_func=flat_weights_arr if lback_meth == "ma" else expo_weights_arr,
    )


def cov_estimators_bias_variance(
    configs: List[Dict[str, Any]],
    corr: np.ndarray,
    base_vol: np.ndarray,
    vol_persistence: float,
    vol_of_vol: float,
    fid_names: List[str],
    n_periods: int,
    n_iter: int = 20,
    seed: int = 42,
    common_sample: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Bias and dispersion of each covariance estimator's portfolio volatility forecast.

    Simulates `n_iter` panels from a known data generating process, runs every config in
    `configs` over each, and scores its forecast against the DGP's own covariance for the
    interval that follows. The score is `sqrt(w' cov_true w / w' cov_est w)`, realized vol
    over the target a portfolio scaled under that forecast would have been aiming at.

    Parameters
    ----------
    configs : List[Dict[str, Any]]
        one estimator per entry. `est_freqs`, `lback_periods` and `lback_meth` are
        required, `half_life` under `lback_meth="xma"`, and `est_weights` defaults to
        equal weights.
    corr : np.ndarray
        (n_fids, n_fids) correlation matrix of the simulated return innovations.
    base_vol : np.ndarray
        long-run daily volatilities of the simulated contracts, as fractions.
    vol_persistence : float
        AR(1) coefficient of the simulated log-variance process.
    vol_of_vol : float
        standard deviation of the shocks to the simulated log-variance. Zero gives
        constant volatility, under which a trailing sample covariance is a consistent
        forecast and any residual bias is sampling noise rather than model error.
    fid_names : List[str]
        contract identifiers to simulate; their count sets the size of the matrices.
    n_periods : int
        business days to simulate per iteration. It must comfortably exceed the longest
        lookback in `configs`, which is consumed as warm-up before the first estimate.
    n_iter : int
        independent panels to simulate. Default is 20.
    seed : int
        seed of the generator that draws each iteration's seed. Default is 42.
    common_sample : bool
        whether to score every config on the dates where all of them have an estimate.
        Default is True. Configs warm up at different rates, so otherwise each is
        averaged over its own set of dates and the comparison between them is made on
        unequal samples. The cost is that the slowest-warming config sets the start
        date for all of them.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        `bias` and `std`, one entry per config in the order given. `bias` is signed:
        negative means the estimator under-forecast risk, so realized volatility
        overshot the target and the positions it sized were too large. `std` is the
        dispersion of the ratio across dates and iterations pooled.
    """
    resolved_configs = [_resolve_config(config) for config in configs]

    rng = np.random.default_rng(seed=seed)
    data_generator = SignalsAndReturnsGenerator(
        n_fids=len(fid_names),
        corr=corr,
        base_vol=base_vol,
        vol_persistence=vol_persistence,
        vol_of_vol=vol_of_vol,
    )

    results = []
    for iter_seed in rng.integers(low=0, high=10000, size=n_iter):
        data_generator.simulate_signals_and_returns(
            n_periods=n_periods,
            signal_names=[f"{fid}SIG" for fid in fid_names],
            return_names=[f"{fid}XR" for fid in fid_names],
            seed=iter_seed,
        )

        cov_true = data_generator.realized_cov(long=False)

        estimation_dates = pd.Series(list(cov_true.keys()))
        cov_ests = []
        for index, resolved in enumerate(resolved_configs):
            try:
                cov_est = _cov_matrix_history(
                    pivot_returns=100 * data_generator.returns,
                    estimation_dates=estimation_dates,
                    nan_tolerance=0,
                    remove_zeros=False,
                    **resolved,
                )
            except ValueError as err:
                raise ValueError(
                    f"config at index {index} ({configs[index]}) could not be estimated "
                    f"on a {n_periods}-period panel: {err}"
                ) from err

            cov_est = {date: cov for date, cov in zip(estimation_dates, cov_est)}

            cov_ests.append(cov_est)

        ratios = realized_to_forecast_vol_ratios(cov_true=cov_true, cov_ests=cov_ests)

        results.append(ratios)

    results = np.vstack(results)

    if common_sample:
        common = np.isfinite(results).all(axis=1)
        if not common.any():
            raise ValueError(
                "no date carries an estimate from every config, so they cannot be "
                "scored on a common sample."
            )

        results = results[common]

    bias = 1 - np.nanmean(results, axis=0)
    std = np.nanstd(results, axis=0)

    return bias, std


if __name__ == "__main__":
    cov_est_configs = [
        {"est_freqs": ["D"], "lback_meth": "ma", "lback_periods": [10]},
        {"est_freqs": ["M"], "lback_meth": "ma", "lback_periods": [10]},
        {"est_freqs": ["D"], "lback_meth": "ma", "lback_periods": [20]},
        {"est_freqs": ["M"], "lback_meth": "ma", "lback_periods": [20]},
        {"est_freqs": ["D"], "lback_meth": "ma", "lback_periods": [30]},
        {"est_freqs": ["M"], "lback_meth": "ma", "lback_periods": [30]},
        {"est_freqs": ["D"], "lback_meth": "ma", "lback_periods": [60]},
        {"est_freqs": ["M"], "lback_meth": "ma", "lback_periods": [60]},
    ]

    corr = np.array(
        [
            [1.0, 0.5, 0.3, 0.4, 0.2],
            [0.5, 1.0, 0.4, 0.3, 0.3],
            [0.3, 0.4, 1.0, 0.5, 0.2],
            [0.4, 0.3, 0.5, 1.0, 0.4],
            [0.2, 0.3, 0.2, 0.4, 1.0],
        ]
    )

    bias, std = cov_estimators_bias_variance(
        corr=corr,
        base_vol=np.array([0.010, 0.015, 0.008, 0.012, 0.009]),
        vol_persistence=0.94,
        vol_of_vol=0.15,
        n_periods=2520,
        fid_names=[f"CID{i}_FX" for i in range(5)],
        configs=cov_est_configs,
        n_iter=5,
        seed=40,
    )

    print(f"Bias: {bias}")
    print(f"Std: {std}")
