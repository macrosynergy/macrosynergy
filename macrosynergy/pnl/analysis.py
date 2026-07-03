from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from macrosynergy.management.simulate import SignalsAndReturnsGenerator
from macrosynergy.pnl import notional_positions


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
    cov_true_dict = _long_cov_to_dict(cov_true)
    true_dates = sorted(cov_true_dict.keys())
    weights_by_date = {
        date: weights if weights is not None else _min_var_weights(cov_true_dict[date])
        for date in true_dates
    }

    # row i is true_dates[i] in every column, so entries stay comparable
    # across estimators with different date coverage
    ratios = np.full(shape=(len(true_dates), len(cov_ests)), fill_value=np.nan)
    for j, cov_est in enumerate(cov_ests):
        cov_est_dict = _long_cov_to_dict(cov_est)

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
    n_fids: int,
    corr: np.ndarray,
    base_vol: np.ndarray,
    signal_ic: float,
    signal_autocorr: float,
    vol_persistence: float,
    vol_of_vol: float,
    mean_return: float,
    freq: str,
    signal_names: List[str],
    return_names: List[str],
    fid_names: List[str],
    configs: List[Dict[str, Any]],
    n_periods: int,
    end_date: str,
    n_iter: int = 20,
    seed: int = 42,
):
    rng = np.random.default_rng(seed=seed)
    data_generator = SignalsAndReturnsGenerator(
        n_fids=n_fids,
        corr=corr,
        base_vol=base_vol,
        signal_ic=signal_ic,
        signal_autocorr=signal_autocorr,
        vol_persistence=vol_persistence,
        vol_of_vol=vol_of_vol,
        mean_return=mean_return,
    )

    results = []
    for seed in rng.integers(low=0, high=1000, size=n_iter):
        data_generator.simulate_signals_and_returns(
            n_periods=n_periods,
            end_date=end_date,
            seed=seed,
            signal_names=signal_names,
            return_names=return_names,
            freq=freq,
        )
        signals_and_returns = data_generator.quantamental_returns_and_signals()

        cov_true = data_generator.realized_cov()
        cov_ests = [
            # todo have a special function for getting the covariance matrix
            notional_positions(
                df=signals_and_returns,
                sname="STRAT",
                fids=fid_names,
                vol_target=10,
                return_vcv=True,
                **config,
            )[1]
            for config in configs
        ]

        ratios = realized_to_forecast_vol_ratios(cov_true=cov_true, cov_ests=cov_ests)

        results.append(ratios)

    results = np.vstack(results)

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
        n_fids=5,
        corr=corr,
        base_vol=np.array([0.010, 0.015, 0.008, 0.012, 0.009]),
        signal_ic=0.05,
        signal_autocorr=0.9,
        vol_persistence=0.94,
        vol_of_vol=0.15,
        mean_return=0.0003,
        n_periods=2520,
        freq="B",
        end_date="2025-12-31",
        signal_names=[f"CID{i}_FX_CSIG_STRAT" for i in range(5)],
        return_names=[f"CID{i}_FXXR" for i in range(5)],
        fid_names=[f"CID{i}_FX" for i in range(5)],
        configs=cov_est_configs,
        n_iter=25,
        seed=40,
    )

    print(f"Bias: {bias}")
    print(f"Std: {std}")
