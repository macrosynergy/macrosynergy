from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from macrosynergy.management.simulate import SignalsAndReturnsGenerator
from macrosynergy.pnl import notional_positions
from macrosynergy.pnl.historic_portfolio_volatility import (
    flat_weights_arr,
    expo_weights_arr,
    _cov_matrix_history,
)


def _long_cov_to_dict(
    cov_long: pd.DataFrame,
    fids: List[str] = None,
    check_psd: bool = True,
    psd_tol: float = 1e-8,
) -> Dict[str, np.ndarray]:
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
    """
    Minimum-variance weights, w prop to Sigma^{-1} 1, normalised to sum 1.

    A rank-deficient or near-singular estimate can't be inverted directly.
    Fall back to a tiny ridge on the diagonal so the pipeline still produces weights.
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
    cov_true: pd.DataFrame,
    cov_ests: List[pd.DataFrame],
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    sqrt(w' cov_true w / w' cov_est w) per true date (rows) and estimator (columns).

    Equals realized vol over targeted vol for a portfolio scaled to hit any vol
    target under the forecast — the target cancels, as does the scaling of w.

    When `weights` is None, minimum-variance weights are derived from the *true*
    covariance of each date. Deriving them from the estimate would bias the
    forecast variance low (the optimizer picks what looks cheapest under its own
    estimation noise), inflating the ratio for noisy estimators.
    """
    # cov_true_dict = _long_cov_to_dict(cov_true)
    cov_true_dict = cov_true
    true_dates = sorted(cov_true_dict.keys())
    weights_by_date = {
        date: weights if weights is not None else _min_var_weights(cov_true_dict[date])
        for date in true_dates
    }

    # row i is true_dates[i] in every column, so entries stay comparable
    # across estimators with different date coverage
    ratios = np.full(shape=(len(true_dates), len(cov_ests)), fill_value=np.nan)
    for j, cov_est in enumerate(cov_ests):
        # cov_est_dict = _long_cov_to_dict(cov_est)
        cov_est_dict = cov_est

        for i, date in enumerate(true_dates):
            est: Optional[np.ndarray] = cov_est_dict.get(date)
            if est is None:
                continue
            truth: np.ndarray = cov_true_dict[date]

            w = weights_by_date[date]

            # A degenerate estimate (e.g. monthly sampling, many assets) can give
            # a non-positive quadratic form; guard it so the entry stays NaN
            forecast_var = w @ est @ w
            if not np.isfinite(forecast_var) or forecast_var <= 0:
                continue

            ratios[i, j] = np.sqrt(w @ truth @ w / forecast_var)

    return ratios


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
):
    rng = np.random.default_rng(seed=seed)
    data_generator = SignalsAndReturnsGenerator(
        n_fids=len(fid_names),
        corr=corr,
        base_vol=base_vol,
        vol_persistence=vol_persistence,
        vol_of_vol=vol_of_vol,
    )

    results = []
    for seed in rng.integers(low=0, high=10000, size=n_iter):
        data_generator.simulate_signals_and_returns(
            n_periods=n_periods,
            signal_names=[f"{fid}SIG" for fid in fid_names],
            return_names=[f"{fid}XR" for fid in fid_names],
            seed=seed,
        )

        cov_true = data_generator.realized_cov(long=False)
        
        estimation_dates = np.array(list(cov_true.keys()))
        cov_ests = []
        for config in configs:
            est_freqs = config["est_freqs"]
            est_weights = config.get("est_weights", [1])
            lback_meth = config["lback_meth"]
            lback_periods = config.get("lback_periods", config.get("half_life"))
            half_life = config.get("half_life", config.get("lback_periods"))

            weights_func = flat_weights_arr if lback_meth == "ma" else expo_weights_arr

            cov_est = _cov_matrix_history(
                pivot_returns=100 * data_generator.returns,
                estimation_dates=estimation_dates,
                est_freqs=est_freqs,
                est_weights=est_weights,
                lback_periods=lback_periods,
                half_life=half_life,
                nan_tolerance=0,
                remove_zeros=False,
                weights_func=weights_func,
                lback_min_obs=[1 for _ in est_freqs],
            )

            cov_est = {date: cov for date, cov in zip(estimation_dates, cov_est)}

            cov_ests.append(cov_est)

        ratios = realized_to_forecast_vol_ratios(cov_true=cov_true, cov_ests=cov_ests)

        results.append(ratios)

    results = np.vstack(results)

    bias = np.abs(1 - np.nanmean(results, axis=0))
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
