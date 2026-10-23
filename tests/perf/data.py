"""Deterministic synthetic QuantamentalDataFrame builders at controlled scale.

Built on macrosynergy.management.simulate.make_qdf (seeded, object-dtype) so benchmarks
have a clear, reproducible target. Row count is approximate per tier; the seed pins values.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

from macrosynergy.management.simulate import make_qdf
from macrosynergy.management.types import QuantamentalDataFrame

# Approximate total rows = n_cids * n_xcats * n_days (business days).
SCALE_TIERS = {
    "tiny": {"n_cids": 4, "n_xcats": 3, "n_days": 250},        # ~3k rows
    "small": {"n_cids": 10, "n_xcats": 8, "n_days": 1300},     # ~104k rows
    "medium": {"n_cids": 20, "n_xcats": 15, "n_days": 3500},   # ~1.05M rows
    "large": {"n_cids": 40, "n_xcats": 30, "n_days": 5200},    # ~6.2M rows
}


def _cid_codes(n: int) -> List[str]:
    # 3-char uppercase codes with no underscore (valid cid). AAA, AAB, ...
    codes = []
    i = 0
    while len(codes) < n:
        a, b, c = i // 676, (i // 26) % 26, i % 26
        codes.append(chr(65 + a % 26) + chr(65 + b) + chr(65 + c))
        i += 1
    return codes


def _xcat_codes(n: int) -> List[str]:
    return [f"XCAT{j:03d}" for j in range(n)]


def _days_to_latest(n_days: int, earliest: str = "2000-01-01") -> str:
    # n_days business days from `earliest`; pad calendar days by ~7/5.
    cal = int(n_days * 7 / 5) + 10
    return (pd.Timestamp(earliest) + pd.Timedelta(days=cal)).strftime("%Y-%m-%d")


def make_perf_qdf(
    n_cids: int, n_xcats: int, n_days: int, *, categorical: bool = False, seed: int = 42
) -> pd.DataFrame:
    """Object-dtype QDF (or categorical) with `cid, xcat, real_date, value` columns."""
    cids = _cid_codes(n_cids)
    xcats = _xcat_codes(n_xcats)
    latest = _days_to_latest(n_days)

    df_cids = pd.DataFrame(index=cids, columns=["earliest", "latest", "mean_add", "sd_mult"])
    for k, cid in enumerate(cids):
        df_cids.loc[cid] = ["2000-01-01", latest, 0.0, 1.0 + (k % 3) * 0.5]

    df_xcats = pd.DataFrame(
        index=xcats,
        columns=["earliest", "latest", "mean_add", "sd_mult", "ar_coef", "back_coef"],
    )
    for j, xc in enumerate(xcats):
        df_xcats.loc[xc] = ["2000-01-01", latest, 0.0, 1.0, 0.5, 0.0]

    df = make_qdf(df_cids, df_xcats, back_ar=0.0, seed=seed)
    df = df[["cid", "xcat", "real_date", "value"]].reset_index(drop=True)
    if categorical:
        return QuantamentalDataFrame(df, categorical=True)
    return df


def qdf_for_tier(tier: str, *, categorical: bool = False, seed: int = 42) -> pd.DataFrame:
    cfg = SCALE_TIERS[tier]
    return make_perf_qdf(
        cfg["n_cids"], cfg["n_xcats"], cfg["n_days"], categorical=categorical, seed=seed
    )


def wide_ticker_frame(n_tickers: int, n_days: int, *, seed: int = 42) -> pd.DataFrame:
    """Wide frame: DatetimeIndex rows, one `cid_xcat` column per ticker (for ticker_df_to_qdf)."""
    rng = np.random.default_rng(seed)
    n_cids = max(1, int(np.ceil(np.sqrt(n_tickers))))
    cids = _cid_codes(n_cids)
    cols = []
    for cid in cids:
        for j in range(n_cids):
            if len(cols) >= n_tickers:
                break
            cols.append(f"{cid}_XCAT{j:03d}")
    cols = cols[:n_tickers]
    idx = pd.bdate_range("2000-01-01", periods=n_days, name="real_date")
    data = rng.standard_normal((n_days, len(cols)))
    return pd.DataFrame(data, index=idx, columns=cols)


def update_df_pieces(
    tier: str, n_pieces: int, *, categorical: bool = False, seed: int = 42
) -> Tuple[pd.DataFrame, List[pd.DataFrame]]:
    """A base QDF plus `n_pieces` non-empty row-slices to feed update_df in a growing loop.

    Splitting by rows (not by xcat) guarantees every piece is non-empty for any
    ``1 <= n_pieces <= len(full)``, so callers can request more pieces than there are
    categories without silently producing empty slices.
    """
    full = qdf_for_tier(tier, categorical=categorical, seed=seed)
    xcats = list(pd.unique(full["xcat"]))
    base = full[full["xcat"].isin(xcats[: max(1, len(xcats) // 2)])].reset_index(drop=True)
    pieces = [
        full.iloc[idx].reset_index(drop=True)
        for idx in np.array_split(np.arange(len(full)), n_pieces)
        if len(idx) > 0
    ]
    return base, pieces


def srr_panel(
    n_cids: int, n_dates: int, n_signals: int, n_returns: int, *, seed: int = 42
) -> pd.DataFrame:
    """QDF with `n_signals` signal xcats (SIGn) and `n_returns` return xcats (XRn)."""
    cids = _cid_codes(n_cids)
    latest = _days_to_latest(n_dates)
    sig_xcats = [f"SIG{i:02d}" for i in range(n_signals)]
    ret_xcats = [f"XR{i:02d}" for i in range(n_returns)]
    xcats = sig_xcats + ret_xcats

    df_cids = pd.DataFrame(index=cids, columns=["earliest", "latest", "mean_add", "sd_mult"])
    for cid in cids:
        df_cids.loc[cid] = ["2000-01-01", latest, 0.0, 1.0]
    df_xcats = pd.DataFrame(
        index=xcats,
        columns=["earliest", "latest", "mean_add", "sd_mult", "ar_coef", "back_coef"],
    )
    for xc in xcats:
        df_xcats.loc[xc] = ["2000-01-01", latest, 0.0, 1.0, 0.3, 0.4]
    df = make_qdf(df_cids, df_xcats, back_ar=0.5, seed=seed)
    return df[["cid", "xcat", "real_date", "value"]].reset_index(drop=True)
