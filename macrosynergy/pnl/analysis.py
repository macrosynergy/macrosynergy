import logging
from collections.abc import Mapping
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from macrosynergy.management.simulate import SignalsAndReturnsGenerator
from macrosynergy.pnl import (
    notional_positions,
    proxy_pnl_calc,
    evaluate_pnl,
)
from macrosynergy.pnl.transaction_costs import TransactionCostsDictAdapter
from macrosynergy.pnl.historic_portfolio_volatility import (
    _check_est_args,
    _cov_matrix_history,
    expo_weights_arr,
    flat_weights_arr,
)

logger = logging.getLogger(__name__)

CONFIG_KEYS = {
    "est_freqs",
    "est_weights",
    "lback_meth",
    "lback_periods",
    "half_life",
    "dof_correct",
}

# the cid `proxy_pnl_calc` aggregates the per-contract PnL and costs under
PORTFOLIO_NAME = "GLB"
DEFAULT_END_DATE = "2025-01-15"


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


def _weights_by_date(
    weights: Union[np.ndarray, Dict[pd.Timestamp, np.ndarray]],
    true_dates: List[pd.Timestamp],
    n_fids: int,
) -> Dict[pd.Timestamp, np.ndarray]:
    """
    Resolve `weights` into one vector per date, checked against the fid axis.

    Parameters
    ----------
    weights : Union[np.ndarray, Dict[pd.Timestamp, np.ndarray]]
        one (n_fids,) vector held fixed across dates, or one per date. A mapping must
        cover every date in `true_dates`; a date it is missing is an error rather than a
        silently skipped row, since a weight is not something this module can invent.
    true_dates : List[pd.Timestamp]
        the dates to resolve weights for, i.e. those of the covariance being forecast.
    n_fids : int
        length every weight vector must have, taken from the covariance matrices.
    """
    if isinstance(weights, Mapping):
        missing = [date for date in true_dates if date not in weights]
        if missing:
            raise ValueError(
                f"`weights` is missing {len(missing)} of the {len(true_dates)} "
                f"dates in `cov_true`, the first being {missing[0].date()}."
            )

        resolved = {}
        for date in true_dates:
            w = np.asarray(weights[date], dtype=float)
            if w.shape != (n_fids,):
                raise ValueError(
                    f"weights for {date.date()} have shape {tuple(w.shape)}, but "
                    f"`cov_true` is over {n_fids} fids."
                )
            resolved[date] = w
        return resolved

    w = np.asarray(weights, dtype=float)
    if w.shape != (n_fids,):
        raise ValueError(
            f"`weights` has shape {tuple(w.shape)}, but `cov_true` is over "
            f"{n_fids} fids."
        )
    return {date: w for date in true_dates}


def realized_to_forecast_vol_ratios(
    cov_true: Dict[pd.Timestamp, np.ndarray],
    cov_ests: List[Dict[pd.Timestamp, np.ndarray]],
    weights: Union[np.ndarray, Dict[pd.Timestamp, np.ndarray]],
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
    weights : Union[np.ndarray, Dict[pd.Timestamp, np.ndarray]]
        portfolio weights the ratio is evaluated at: one (n_fids,) vector held fixed
        across dates, or one per date, on the same fid axis order as `cov_true`.

    Returns
    -------
    np.ndarray
        (n_true_dates, n_estimators) array of ratios, rows in sorted date order.
    """
    if isinstance(cov_true, pd.DataFrame) or any(
        isinstance(cov_est, pd.DataFrame) for cov_est in cov_ests
    ):
        raise TypeError(
            "`cov_true` and `cov_ests` take date-keyed covariance matrices, not long "
            "frames. Convert a long frame with `_long_cov_to_dict` first."
        )

    true_dates = sorted(cov_true.keys())
    n_fids = next(iter(cov_true.values())).shape[0]
    weights_by_date = _weights_by_date(
        weights=weights, true_dates=true_dates, n_fids=n_fids
    )

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

            forecast_var = w @ est @ w
            if not np.isfinite(forecast_var) or forecast_var <= 0:
                continue

            ratios[i, j] = np.sqrt(w @ truth @ w / forecast_var)

    return ratios


def _iteration_seeds(seed: int, n_iter: int) -> List[int]:
    """
    One seed per iteration, distinct and independent, deterministic in `seed`.
    """
    return [
        int(child.generate_state(1, dtype=np.uint64)[0])
        for child in np.random.SeedSequence(seed).spawn(n_iter)
    ]


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
        dof_correct=bool(config.get("dof_correct", False)),
    )


def _bias_and_dispersion(
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
    signal_half_life: Optional[float] = 21,
    signal_ic: float = 0.05,
    signal_autocorr: float = 0.9,
    weights: Optional[Union[np.ndarray, Dict[pd.Timestamp, np.ndarray]]] = None,
    end_date: str = DEFAULT_END_DATE,
) -> Tuple[np.ndarray, np.ndarray]:
    resolved_configs = [_resolve_config(config) for config in configs]

    data_generator = SignalsAndReturnsGenerator(
        n_fids=len(fid_names),
        corr=corr,
        base_vol=base_vol,
        vol_persistence=vol_persistence,
        vol_of_vol=vol_of_vol,
        signal_ic=signal_ic,
        signal_autocorr=signal_autocorr,
        half_life=signal_half_life,
    )

    results = []
    for iter_seed in _iteration_seeds(seed=seed, n_iter=n_iter):
        data_generator.simulate_signals_and_returns(
            n_periods=n_periods,
            signal_names=[f"{fid}SIG" for fid in fid_names],
            return_names=[f"{fid}XR" for fid in fid_names],
            seed=iter_seed,
            end_date=end_date,
        )

        cov_true = data_generator.realized_cov(long=False)

        estimation_dates = pd.Series(list(cov_true.keys()))
        cov_ests = []
        for index, resolved in enumerate(resolved_configs):
            cov_est = _cov_matrix_history(
                pivot_returns=100 * data_generator.returns,
                estimation_dates=estimation_dates,
                nan_tolerance=0.0,
                remove_zeros=False,
                **resolved,
            )

            cov_est = {date: cov for date, cov in zip(estimation_dates, cov_est)}

            cov_ests.append(cov_est)

        scoring_weights = (
            {
                date: data_generator.signals.loc[date].to_numpy(dtype=float)
                for date in cov_true
            }
            if weights is None
            else weights
        )

        ratios = realized_to_forecast_vol_ratios(
            cov_true=cov_true,
            cov_ests=cov_ests,
            weights=scoring_weights,
        )

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
    std = np.nanstd(results, axis=0, ddof=1)

    return bias, std


def scaling_factor_bias_variance(
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
    signal_half_life: float = 21,
    signal_ic: float = 0.05,
    signal_autocorr: float = 0.9,
    weights: Optional[Union[np.ndarray, Dict[pd.Timestamp, np.ndarray]]] = None,
    report_floor: bool = True,
    end_date: str = DEFAULT_END_DATE,
) -> Tuple[np.ndarray, ...]:
    """
    Volatility-target miss of each covariance estimator, against its own noise floor.

    Simulates `n_iter` panels from a known data generating process, runs every config in
    `configs` over each, and scores its forecast against the DGP's own covariance for the
    interval that follows. The score is `sqrt(w' cov_true w / w' cov_est w)`, the realized
    volatility over the target a portfolio scaled under that forecast would have been
    aiming at.

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
    signal_half_life : Optional[float]
        half life, in business days, of the simulated signal's forecasting powerl
    signal_ic : float
        information coefficient of the simulated signals. Default is 0.05. Bounded above
        by `sqrt(1 - decay ** 2)` for the decay implied by `signal_half_life`.
    signal_autocorr : float
        AR(1) coefficient of the persistent signal component. Default is 0.9. It sets how
        fast the weights move, and so the turnover a cost study sees.
    common_sample : bool
        whether to score every config on the dates where all of them have an estimate.
        Default is True. Configs warm up at different rates, so otherwise each is
        averaged over its own set of dates and the comparison between them is made on
        unequal samples. The cost is that the slowest-warming config sets the start
        date for all of them.
    weights : Optional[Union[np.ndarray, Dict[pd.Timestamp, np.ndarray]]]
        portfolio weights to score the estimators at, as
        `realized_to_forecast_vol_ratios` takes them.
    end_date : str
        last date of the simulated panel
    report_floor : bool
        whether to measure each config's noise floor by rerunning the DGP with
        `vol_of_vol=0`. Default is True.

    Notes
    -----
    Positions scale with `1 / sqrt(variance)`, which is convex, so a noisy
    but perfectly centred estimate still oversizes on average - averaging a convex
    function of a noisy input exceeds the function of the average (Jensen's inequality).
    The size of that effect depends only on how noisy the estimator is, so it falls
    monotonically with the lookback. A short lookback therefore scores worse than
    a long one even on a DGP where volatility is constant and there is nothing to forecast.

    `report_floor` measures that floor rather than assuming it away: the same configs are
    run again over the same DGP with `vol_of_vol=0`, where every estimator is correctly
    specified, so whatever they score is noise. `excess_bias` is the reading net of it,
    and is the column to rank on.
    """
    shared = dict(
        configs=configs,
        corr=corr,
        base_vol=base_vol,
        vol_persistence=vol_persistence,
        fid_names=fid_names,
        n_periods=n_periods,
        n_iter=n_iter,
        seed=seed,
        common_sample=common_sample,
        signal_half_life=signal_half_life,
        signal_ic=signal_ic,
        signal_autocorr=signal_autocorr,
        weights=weights,
        end_date=end_date,
    )

    bias, std = _bias_and_dispersion(vol_of_vol=vol_of_vol, **shared)

    if report_floor:
        noise_floor, _ = _bias_and_dispersion(vol_of_vol=0.0, **shared)
    else:
        noise_floor = np.full(len(configs), np.nan)

    if report_floor:
        return bias, std, noise_floor, bias - noise_floor

    return bias, std


def _config_to_positions_kwargs(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate one estimator config and expand it into `notional_positions` arguments.
    """
    resolved = _resolve_config(config)
    return dict(
        lback_meth=str(config["lback_meth"]).lower(),
        est_freqs=resolved["est_freqs"],
        est_weights=resolved["est_weights"],
        lback_periods=resolved["lback_periods"],
        half_life=resolved["half_life"],
        dof_correct=resolved["dof_correct"],
    )



def cov_estimators_cost_accuracy(
    configs: List[Dict[str, Any]],
    corr: np.ndarray,
    base_vol: np.ndarray,
    vol_persistence: float,
    vol_of_vol: float,
    fid_names: List[str],
    n_periods: int,
    tcost_obj: TransactionCostsDictAdapter,
    aum: float = 100,
    vol_target: float = 10,
    n_iter: int = 5,
    seed: int = 42,
    signal_half_life: float = 21,
    signal_ic: float = 0.05,
    signal_autocorr: float = 0.9,
    rebal_freq: str = "M",
    rstring: str = "XR",
    slip: int = 0,
    sname: str = "STRAT",
    end_date: str = DEFAULT_END_DATE,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transaction cost of running each covariance estimator, measured through a real PnL.

    Each config is put through `notional_positions` at the shared vol target and then
    `proxy_pnl_calc`, so the cost reflects the turnover the estimator's scaling actually
    generates.

    Parameters
    ----------
    configs : List[Dict[str, Any]]
        one estimator per entry, in the form `scaling_factor_bias_variance` takes.
    corr : np.ndarray
        (n_fids, n_fids) correlation matrix of the simulated return innovations.
    base_vol : np.ndarray
        long-run daily volatilities of the simulated contracts, as fractions.
    vol_persistence : float
        AR(1) coefficient of the simulated log-variance process.
    vol_of_vol : float
        standard deviation of the shocks to the simulated log-variance.
    fid_names : List[str]
        contract identifiers to simulate, as "<cid>_<ctype>".
    n_periods : int
        business days to simulate per iteration
    tcost_obj : TransactionCostsDictAdapter
        Transaction cost object
    aum : float
        assets under management, in USD millions. Default is 100.
    vol_target : float
        target volatility.
    n_iter : int
        independent panels to simulate. Default is 5, lower than the analytical harness's
        because each iteration runs a full PnL per config.
    seed : int
        seed of the generator that draws each iteration's seed. Default is 42.
    signal_half_life : float
        half life, in business days, of the simulated signal's forecasting power
    signal_ic : float
        information coefficient of the simulated signals. Default is 0.05.
    signal_autocorr : float
        AR(1) coefficient of the persistent signal component. Default is 0.9
    rebal_freq : str
        rebalancing frequency. Default is "M", which puts the rebalance dates on the
        business month starts the ground-truth covariance is keyed to.
    rstring : str
        string identifying the return series. Default is "XR".
    slip : int
        days to wait before applying a signal. Default is 0
    sname : str
        strategy name. Default is "STRAT".
    end_date : str
        last date of the simulated panel, in ISO format.
    """
    positions_kwargs = [_config_to_positions_kwargs(config) for config in configs]

    data_generator = SignalsAndReturnsGenerator(
        n_fids=len(fid_names),
        corr=corr,
        base_vol=base_vol,
        vol_persistence=vol_persistence,
        vol_of_vol=vol_of_vol,
        signal_ic=signal_ic,
        signal_autocorr=signal_autocorr,
        half_life=signal_half_life,
    )

    costs = np.full(shape=(n_iter, len(configs)), fill_value=np.nan)
    realized_vol = np.full(shape=(n_iter, len(configs)), fill_value=np.nan)

    for i, iter_seed in enumerate(_iteration_seeds(seed=seed, n_iter=n_iter)):
        data_generator.simulate_signals_and_returns(
            n_periods=n_periods,
            signal_names=[f"{fid}_CSIG_{sname}" for fid in fid_names],
            return_names=[f"{fid}{rstring}" for fid in fid_names],
            seed=iter_seed,
            end_date=end_date,
        )
        df_rets = data_generator.quantamental_returns()
        df = data_generator.quantamental_returns_and_signals()

        for j, kwargs in enumerate(positions_kwargs):
            npos = notional_positions(
                df=df,
                sname=sname,
                fids=list(fid_names),
                aum=aum,
                slip=slip,
                vol_target=vol_target,
                rebal_freq=rebal_freq,
                rstring=rstring,
                nan_tolerance=0.0,
                remove_zeros=False,
                **kwargs,
            )

            pnl_df, costs_df = proxy_pnl_calc(
                df=pd.concat((df_rets, npos), ignore_index=True),
                spos=f"{sname}_POS",
                rstring=rstring,
                roll_freq=rebal_freq,
                portfolio_name=PORTFOLIO_NAME,
                transaction_costs_object=tcost_obj,
                return_costs=True,
            )

            metrics = evaluate_pnl(df_pnl=pnl_df, df_tcosts=costs_df, aum=aum)

            costs[i, j] =  metrics.loc["Transaction Cost"].iat[0]
            realized_vol[i, j] = metrics.loc["St. Dev. %"].iat[0]

    cost_pct = costs.mean(axis=0)
    vol_pct = realized_vol.mean(axis=0)

    return cost_pct, vol_pct


if __name__ == "__main__":
    cov_est_configs = [
        {"est_freqs": ["D"], "lback_meth": "ma", "lback_periods": [15]},
        {"est_freqs": ["D"], "lback_meth": "ma", "lback_periods": [30]},
        {"est_freqs": ["D"], "lback_meth": "ma", "lback_periods": [60]},
        {"est_freqs": ["M"], "lback_meth": "ma", "lback_periods": [12]},
        {"est_freqs": ["M"], "lback_meth": "ma", "lback_periods": [24]},
        {"est_freqs": ["M"], "lback_meth": "ma", "lback_periods": [36]},
        {"est_freqs": ["W"], "lback_meth": "ma", "lback_periods": [15]},
        {"est_freqs": ["W"], "lback_meth": "ma", "lback_periods": [30]},
        {"est_freqs": ["W"], "lback_meth": "ma", "lback_periods": [60]},
        {
            "est_freqs": ["D"],
            "lback_meth": "xma",
            "half_life": [12],
            "lback_periods": [-1],
        },
        {
            "est_freqs": ["D"],
            "lback_meth": "xma",
            "half_life": [15],
            "lback_periods": [-1],
        },
        {
            "est_freqs": ["D"],
            "lback_meth": "xma",
            "half_life": [25],
            "lback_periods": [-1],
        },
        {
            "est_freqs": ["D", "M", "W"],
            "lback_meth": "ma",
            "lback_periods": [15, 12, 15],
            "est_weights": [1, 1, 1],
        },
        {
            "est_freqs": ["D", "M", "W"],
            "lback_meth": "ma",
            "lback_periods": [30, 24, 30],
            "est_weights": [1, 1, 1],
        },
        {
            "est_freqs": ["D", "M", "W"],
            "lback_meth": "ma",
            "lback_periods": [60, 24, 24],
            "est_weights": [1, 1, 1],
        },
        {
            "est_freqs": ["D", "M", "W"],
            "lback_meth": "xma",
            "half_life": [15, 12, 15],
            "est_weights": [1, 1, 1],
            "lback_periods": [-1, -1, -1],
        },
        {
            "est_freqs": ["D", "M", "W"],
            "lback_meth": "xma",
            "half_life": [30, 24, 30],
            "est_weights": [1, 1, 1],
            "lback_periods": [-1, -1, -1],
        },
        {
            "est_freqs": ["D", "M", "W"],
            "lback_meth": "xma",
            "half_life": [60, 24, 24],
            "est_weights": [1, 1, 1],
            "lback_periods": [-1, -1, -1],
        },
    ]

    tcost_dict = {}

    corr_mat = [
        [1.00, 0.57, 0.58, 0.50, 0.42, 0.43, 0.50, 0.42, 0.43],
        [0.57, 1.00, 0.61, 0.42, 0.54, 0.46, 0.42, 0.54, 0.46],
        [0.58, 0.61, 1.00, 0.43, 0.46, 0.58, 0.43, 0.46, 0.58],
        [0.50, 0.42, 0.43, 1.00, 0.57, 0.58, 0.50, 0.42, 0.43],
        [0.42, 0.54, 0.46, 0.57, 1.00, 0.61, 0.42, 0.54, 0.46],
        [0.43, 0.46, 0.58, 0.58, 0.61, 1.00, 0.43, 0.46, 0.58],
        [0.50, 0.42, 0.43, 0.50, 0.42, 0.43, 1.00, 0.57, 0.58],
        [0.42, 0.54, 0.46, 0.42, 0.54, 0.46, 0.57, 1.00, 0.61],
        [0.43, 0.46, 0.58, 0.43, 0.46, 0.58, 0.58, 0.61, 1.00],
    ]
    base_vol = [0.01, 0.015, 0.008, 0.012, 0.009, 0.01, 0.01, 0.01, 0.01]

    fid_names = [f"CID{i}_IRS" for i in range(9)]
    dgp = dict(
        corr=np.array(corr_mat),
        base_vol=np.array(base_vol),
        vol_persistence=0.94,
        vol_of_vol=0.15,
        n_periods=252 * 10,
        fid_names=fid_names,
        configs=cov_est_configs,
        seed=40,
        signal_half_life=21,
    )

    # the same count in both, or the columns below are averages over different panels
    # and the gap between them is sampling noise rather than anything about the method
    n_iter = 3

    # forecast error comes from the analytical harness: exact, and cheap
    acc_df = scaling_factor_bias_variance(n_iter=n_iter, **dgp)

    # cost needs a traded path - one full proxy PnL per config per iteration
    cost_df = cov_estimators_cost_accuracy(
        n_iter=n_iter,
        aum=100,
        vol_target=10,
        tcost_obj=TransactionCostsDictAdapter(tcost_dict),
        **dgp,
    )
