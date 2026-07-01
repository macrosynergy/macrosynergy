"""
Validation harness: custom closed-form ML estimator vs statsmodels MixedLM.

Run standalone (no pytest needed):
    python tests/perf/validate_mixedlm_agreement.py

Computes, for each test panel:
  - p_statsmodels: full-precision re.pvalues[1] from mlm.fit(reml=False)
  - p_custom:      result of map_pval (the closed-form estimator)
  Detects statsmodels non-convergence (ConvergenceWarning or
  re.converged=False), excludes those segments from the tolerance gate,
  and reports them separately.

Tolerance gate (ALL three must hold on converged segments):
  1. abs(p_custom - p_statsmodels) <= 1e-3  for every non-nan segment
  2. zero significance-decision flips at the 0.9 threshold (raw p < 0.1)
  3. identical nan set

3-dp boundary flips are reported but do NOT fail the gate.

BLOCKER 2 fix: the harness now runs a seed sweep (seeds 0-99) for [1-1]
panels AND explicitly injects a near-zero-tau^2 panel, so the tau^2->0
boundary is genuinely exercised and any regression will be caught.
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from macrosynergy.management.utils import categories_df
from macrosynergy.signal.signal_return_relations import SignalReturnRelations
from tests.perf.data import srr_panel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _statsmodels_pval_full_precision(ret_vals, sig_vals):
    """
    Run statsmodels MixedLM (reml=False) on the same inputs map_pval would see.
    Returns (p_sm, converged, nan_reason) where:
      p_sm      = np.asarray(re.pvalues)[1]  (full precision), or np.nan
      converged = True iff no ConvergenceWarning and re.converged is True
      nan_reason = None | 'linalg' | 'empty_se' | 'le1_cid'
    """
    if (
        "cid" not in ret_vals.index.names
        or ret_vals.index.get_level_values("cid").nunique() <= 1
    ):
        return np.nan, True, "le1_cid"

    X = sm.add_constant(ret_vals)
    y = sig_vals.copy()
    groups = ret_vals.index.get_level_values("real_date")
    mlm = sm.MixedLM(y, X, groups=groups)

    cw_captured = []
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            re = mlm.fit(reml=False)
            for warning in w:
                if issubclass(warning.category, ConvergenceWarning):
                    cw_captured.append(warning)
    except np.linalg.LinAlgError:
        return np.nan, True, "linalg"

    # Check convergence
    converged = (len(cw_captured) == 0)
    # Also check re.converged attribute if available
    if hasattr(re, "converged") and re.converged is not None:
        converged = converged and bool(re.converged)

    # Check for undefined SE (empty string in summary)
    # Use re.bse directly for speed
    bse = np.asarray(re.bse)
    if len(bse) < 2 or np.isnan(bse[1]) or bse[1] == 0.0:
        return np.nan, converged, "empty_se"

    p_sm = float(np.asarray(re.pvalues)[1])
    return p_sm, converged, None


def _custom_pval_full_precision(ret_vals, sig_vals):
    """
    Run the custom estimator at FULL PRECISION (before 3-dp rounding).

    The brief specifies that the validation comparison is done at full
    precision (p_custom_fullprec vs p_sm_fullprec), not at the 3-dp rounded
    value returned by map_pval. We call _mixedlm_slope_pval directly.

    Returns the full-precision float p-value, or np.nan on degenerate inputs.
    """
    from scipy import stats as scipy_stats_local
    if (
        "cid" not in ret_vals.index.names
        or ret_vals.index.get_level_values("cid").nunique() <= 1
    ):
        return np.nan

    X = sm.add_constant(ret_vals)
    y = sig_vals.copy()
    groups = ret_vals.index.get_level_values("real_date")
    _, group_ids = np.unique(groups, return_inverse=True)
    X_arr = np.asarray(X, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        beta1, se_beta1 = SignalReturnRelations._mixedlm_slope_pval(
            y_arr, X_arr, group_ids
        )

    if np.isnan(beta1) or np.isnan(se_beta1) or se_beta1 <= 0.0:
        return np.nan
    z = beta1 / se_beta1
    return float(2.0 * (1.0 - scipy_stats_local.norm.cdf(abs(z))))


def _build_wide(df, sig, ret, cids, freq="M"):
    """Build the wide panel (categories_df) as map_pval sees it."""
    wide = categories_df(
        df,
        xcats=[sig, ret],
        cids=cids,
        val="value",
        freq=freq,
        lag=1,
        fwin=1,
        xcat_aggs=["last", "sum"],
    ).dropna()
    return wide[ret], wide[sig]


# ---------------------------------------------------------------------------
# Panel configurations
# ---------------------------------------------------------------------------

PANEL_CONFIGS = [
    # (label, n_cids, n_dates, n_signals, n_returns, seed)
    ("synthetic_1s1r_small",  6, 400, 1, 1, 42),
    ("synthetic_1s3r_medium", 6, 600, 1, 3, 42),
    ("synthetic_2s3r_medium", 6, 600, 2, 3, 42),
    ("synthetic_1s1r_large",  12, 800, 1, 1, 99),
    ("synthetic_2s3r_large",  12, 800, 2, 3, 99),
]

# Well-conditioned panel: inject real per-date random effects so tau^2 > 0.
# This is the meaningful agreement testbed (statsmodels converges cleanly).
WELL_CONDITIONED_CONFIGS = [
    # (label, n_cids, n_dates, n_signals, n_returns, intercept_sd)
    ("wellcond_moderate_re", 8, 400, 1, 1, 0.5),
    ("wellcond_large_re",    8, 400, 1, 1, 2.0),
    ("wellcond_10cids",      10, 500, 2, 2, 1.0),
    ("wellcond_4cids",       4, 300, 1, 1, 1.0),   # small N
]


def _build_well_conditioned_panel(
    n_cids, n_dates, n_signals, n_returns, intercept_sd, seed=42
):
    """
    Build a synthetic panel with genuine per-date random intercepts.
    y = beta0 + beta1 * x + u_date + eps
    u_date ~ N(0, intercept_sd^2)
    """
    rng = np.random.default_rng(seed)
    base_panel = srr_panel(n_cids, n_dates, n_signals, n_returns, seed=seed)

    # Inject random per-date intercepts into signal xcats
    dates = sorted(base_panel["real_date"].unique())
    date_intercepts = {d: rng.normal(0, intercept_sd) for d in dates}

    sig_xcats = [f"SIG{i:02d}" for i in range(n_signals)]
    mask = base_panel["xcat"].isin(sig_xcats)
    base_panel = base_panel.copy()
    base_panel.loc[mask, "value"] = (
        base_panel.loc[mask, "value"]
        + base_panel.loc[mask, "real_date"].map(date_intercepts)
    )
    return base_panel


def _build_near_zero_tau2_panel(n_cids=8, n_dates=400, seed=42):
    """
    Build a panel designed to produce tau^2 -> 0 at the MLE.

    No per-date random effects are injected: the date-level variance is
    indistinguishable from zero, so the profile optimiser lands at theta=0
    (tau2_hat~0). This exercises the tau^2->0 boundary (Settled decision 5:
    must return finite OLS-limit p-value, NOT nan).
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2000-01-01", periods=n_dates, freq="MS")
    cids = [f"C{i:02d}" for i in range(n_cids)]

    rows = []
    for d in dates:
        for c in cids:
            rows.append({"cid": c, "real_date": d, "xcat": "XR00",
                         "value": rng.normal(0, 1)})
            rows.append({"cid": c, "real_date": d, "xcat": "SIG00",
                         "value": rng.normal(0, 1)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Agreement check for one (ret_vals, sig_vals) pair
# ---------------------------------------------------------------------------

def check_one_segment(ret_vals, sig_vals, label):
    """Returns a dict with agreement metrics for one segment."""
    p_sm, converged, sm_nan_reason = _statsmodels_pval_full_precision(
        ret_vals, sig_vals
    )
    p_cust = _custom_pval_full_precision(ret_vals, sig_vals)

    sm_is_nan = np.isnan(p_sm)
    cu_is_nan = np.isnan(p_cust)
    nan_match = (sm_is_nan == cu_is_nan)

    abs_diff = (
        abs(p_cust - p_sm) if not sm_is_nan and not cu_is_nan else np.nan
    )
    tol_pass = (abs_diff <= 1e-3) if not np.isnan(abs_diff) else None

    # Significance decisions at p < 0.1  (i.e. 1-p > 0.9 threshold)
    sig_sm = (p_sm < 0.1) if not sm_is_nan else None
    sig_cu = (p_cust < 0.1) if not cu_is_nan else None
    decision_flip = (
        (sig_sm != sig_cu)
        if (sig_sm is not None and sig_cu is not None)
        else False
    )

    # 3-dp rounding check: a true rounding-boundary flip is when the rounded
    # values differ. Compare round(p_sm, 3) vs round(p_cust, 3) so that
    # sub-rounding full-precision differences don't spuriously flag every
    # segment.
    sm_3dp = round(p_sm, 3) if not sm_is_nan else np.nan
    cu_3dp = round(p_cust, 3) if not cu_is_nan else np.nan
    dp3_flip = (sm_3dp != cu_3dp) if not sm_is_nan and not cu_is_nan else False

    return {
        "label": label,
        "p_sm": p_sm,
        "p_cust": p_cust,
        "abs_diff": abs_diff,
        "converged": converged,
        "sm_nan_reason": sm_nan_reason,
        "nan_match": nan_match,
        "tol_pass": tol_pass,
        "decision_flip": decision_flip,
        "dp3_flip": dp3_flip,
    }


# ---------------------------------------------------------------------------
# Main harness
# ---------------------------------------------------------------------------

def run_validation():
    print("=" * 70)
    print("VALIDATION HARNESS: custom ML estimator vs statsmodels MixedLM")
    print("=" * 70)

    all_results = []

    # ---- Standard synthetic panels (srr_panel, raw) ----
    print("\n--- Standard synthetic panels (srr_panel) ---")
    for (label, n_cids, n_dates, n_signals, n_returns, seed) in PANEL_CONFIGS:
        df = srr_panel(n_cids, n_dates, n_signals, n_returns, seed=seed)
        cids = sorted(df["cid"].unique())
        for i in range(n_signals):
            for j in range(n_returns):
                seg_label = f"{label}/SIG{i:02d}-XR{j:02d}"
                try:
                    ret_vals, sig_vals = _build_wide(
                        df, f"SIG{i:02d}", f"XR{j:02d}", cids, "M"
                    )
                    result = check_one_segment(ret_vals, sig_vals, seg_label)
                    all_results.append(result)
                except Exception as e:
                    print(f"  ERROR in {seg_label}: {e}")

    # ---- Well-conditioned synthetic panels ----
    print(
        "\n--- Well-conditioned synthetic panels "
        "(injected per-date random effects) ---"
    )
    for (label, n_cids, n_dates, n_signals, n_returns, intercept_sd) in (
        WELL_CONDITIONED_CONFIGS
    ):
        df = _build_well_conditioned_panel(
            n_cids, n_dates, n_signals, n_returns, intercept_sd
        )
        cids = sorted(df["cid"].unique())
        for i in range(n_signals):
            for j in range(n_returns):
                seg_label = f"{label}/SIG{i:02d}-XR{j:02d}"
                try:
                    ret_vals, sig_vals = _build_wide(
                        df, f"SIG{i:02d}", f"XR{j:02d}", cids, "M"
                    )
                    result = check_one_segment(ret_vals, sig_vals, seg_label)
                    all_results.append(result)
                except Exception as e:
                    print(f"  ERROR in {seg_label}: {e}")

    # ---- Seed sweep over [1-1] panels (seeds 0-99) ----
    # This exercises the tau^2->0 boundary across many random seeds and would
    # expose NaN SE or large abs_diff failures on [1-1] (BLOCKER 1 regression
    # test). Any seed producing tau2_hat~0 verifies the forward-difference fix.
    print("\n--- Seed sweep [1-1] small panel (seeds 0-99) ---")
    sweep_results = []
    for seed in range(100):
        df = srr_panel(6, 400, 1, 1, seed=seed)
        cids = sorted(df["cid"].unique())
        seg_label = f"sweep_1s1r_small/seed{seed:03d}"
        try:
            ret_vals, sig_vals = _build_wide(df, "SIG00", "XR00", cids, "M")
            result = check_one_segment(ret_vals, sig_vals, seg_label)
            sweep_results.append(result)
            all_results.append(result)
        except Exception as e:
            print(f"  ERROR in {seg_label}: {e}")

    # Report sweep summary inline
    sweep_converged = [r for r in sweep_results if r["converged"]]
    sweep_max_diff = max(
        (r["abs_diff"] for r in sweep_converged
         if not np.isnan(r["abs_diff"])),
        default=0.0,
    )
    sweep_tol_failures = [r for r in sweep_converged if r["tol_pass"] is False]
    sweep_decision_flips = [r for r in sweep_converged if r["decision_flip"]]
    sweep_nan_failures = [r for r in sweep_converged if not r["nan_match"]]
    print(
        f"  Sweep seeds 0-99: {len(sweep_converged)}/100 converged, "
        f"max_abs_diff={sweep_max_diff:.2e}, "
        f"tol_failures={len(sweep_tol_failures)}, "
        f"decision_flips={len(sweep_decision_flips)}, "
        f"nan_mismatches={len(sweep_nan_failures)}"
    )

    # ---- Near-zero tau^2 boundary injection ----
    # Exercises Settled decision 5: tau^2->0 must give finite OLS-limit
    # p-value, NOT nan.
    print("\n--- Near-zero tau^2 boundary panel (no per-date random effects) ---")
    for seed in [0, 7, 13, 42, 99]:
        df_zt = _build_near_zero_tau2_panel(n_cids=8, n_dates=400, seed=seed)
        seg_label = f"near_zero_tau2/seed{seed:03d}"
        try:
            ret_vals_zt = df_zt[df_zt["xcat"] == "XR00"].set_index(
                ["cid", "real_date"]
            )["value"]
            sig_vals_zt = df_zt[df_zt["xcat"] == "SIG00"].set_index(
                ["cid", "real_date"]
            )["value"]
            # Align indices
            common_idx = ret_vals_zt.index.intersection(sig_vals_zt.index)
            ret_vals_zt = ret_vals_zt.loc[common_idx]
            sig_vals_zt = sig_vals_zt.loc[common_idx]
            result = check_one_segment(ret_vals_zt, sig_vals_zt, seg_label)
            all_results.append(result)
            p_sm_s = (
                f"{result['p_sm']:.6f}"
                if not np.isnan(result["p_sm"]) else "nan"
            )
            p_cu_s = (
                f"{result['p_cust']:.6f}"
                if not np.isnan(result["p_cust"]) else "nan"
            )
            diff_s = (
                f"{result['abs_diff']:.2e}"
                if not np.isnan(result["abs_diff"]) else "nan"
            )
            tol_s = (
                "PASS" if result["tol_pass"]
                else ("FAIL" if result["tol_pass"] is False else "-")
            )
            print(
                f"  {seg_label}: p_sm={p_sm_s}, p_cust={p_cu_s}, "
                f"abs_diff={diff_s}, tol={tol_s}"
            )
        except Exception as e:
            print(f"  ERROR in {seg_label}: {e}")

    # ---- Summarise ----
    print("\n" + "=" * 70)
    print("DETAILED RESULTS (standard + well-conditioned + near-zero-tau2)")
    print("=" * 70)
    hdr = (
        f"{'label':<52} {'p_sm':>8} {'p_cust':>8} "
        f"{'abs_diff':>10} {'conv':>5} {'nan_ok':>6} "
        f"{'tol':>5} {'flip':>5} {'3dp':>5}"
    )
    print(hdr)
    print("-" * 115)

    # Print only non-sweep results in detail (sweep is summarised separately)
    non_sweep = [r for r in all_results if not r["label"].startswith("sweep_")]
    for r in non_sweep:
        conv_str = "Y" if r["converged"] else "N"
        nan_ok_str = "Y" if r["nan_match"] else "N"
        tol_str = (
            "Y" if r["tol_pass"] is True
            else ("N" if r["tol_pass"] is False else "-")
        )
        flip_str = "Y" if r["decision_flip"] else "N"
        dp3_str = "Y" if r["dp3_flip"] else "N"
        p_sm_str = f"{r['p_sm']:.6f}" if not np.isnan(r["p_sm"]) else "nan"
        p_cu_str = (
            f"{r['p_cust']:.6f}" if not np.isnan(r["p_cust"]) else "nan"
        )
        diff_str = (
            f"{r['abs_diff']:.2e}" if not np.isnan(r["abs_diff"]) else "nan"
        )
        print(
            f"{r['label']:<52} {p_sm_str:>8} {p_cu_str:>8} "
            f"{diff_str:>10} {conv_str:>5} {nan_ok_str:>6} "
            f"{tol_str:>5} {flip_str:>5} {dp3_str:>5}"
        )

    n_converged = 0
    n_non_converged = 0
    n_nan_mismatch = 0
    n_tol_fail_converged = 0
    n_decision_flips = 0
    n_dp3_flips = 0
    max_abs_diff_converged = 0.0
    max_abs_diff_label = None

    for r in all_results:
        if not r["nan_match"]:
            n_nan_mismatch += 1

        if not r["converged"]:
            n_non_converged += 1
        else:
            n_converged += 1
            if r["tol_pass"] is False:
                n_tol_fail_converged += 1

        if r["decision_flip"]:
            n_decision_flips += 1

        if r["dp3_flip"]:
            n_dp3_flips += 1

        if not np.isnan(r["abs_diff"]) and r["converged"]:
            if r["abs_diff"] > max_abs_diff_converged:
                max_abs_diff_converged = r["abs_diff"]
                max_abs_diff_label = r["label"]

    n_total = len(all_results)
    print("=" * 70)
    print("\nSUMMARY")
    print(f"  Total segments:          {n_total}")
    print(f"  Converged (statsmodels): {n_converged}")
    print(
        f"  Non-converged (excluded from tol gate): {n_non_converged}"
    )
    print(f"  Nan-set mismatches:      {n_nan_mismatch}")
    print(
        f"  Tol failures (converged, abs_diff > 1e-3): {n_tol_fail_converged}"
    )
    print(
        f"  Decision flips (all, p<0.1 boundary): {n_decision_flips}"
    )
    print(
        f"  3-dp boundary flips (reported, not fatal): {n_dp3_flips}"
    )
    print(
        f"  Max abs_diff (converged): {max_abs_diff_converged:.2e}"
        f"  [{max_abs_diff_label}]"
    )
    print(
        f"  Seed sweep max_abs_diff (converged): {sweep_max_diff:.2e}"
    )
    print(
        f"  Seed sweep tol failures: {len(sweep_tol_failures)}"
    )
    print(
        f"  Seed sweep decision flips: {len(sweep_decision_flips)}"
    )

    print("\nGATE CRITERIA:")
    gate1 = (n_tol_fail_converged == 0)
    gate2 = (n_decision_flips == 0)
    gate3 = (n_nan_mismatch == 0)
    print(
        f"  [{'PASS' if gate1 else 'FAIL'}] "
        f"abs(p_custom - p_sm) <= 1e-3 on ALL converged segments"
    )
    print(
        f"  [{'PASS' if gate2 else 'FAIL'}] "
        f"Zero significance-decision flips at 0.9 threshold"
    )
    print(
        f"  [{'PASS' if gate3 else 'FAIL'}] "
        f"Identical nan set"
    )
    all_pass = gate1 and gate2 and gate3
    print(f"\n  => OVERALL: {'PASS' if all_pass else 'FAIL'}")

    return all_pass


if __name__ == "__main__":
    import sys
    ok = run_validation()
    sys.exit(0 if ok else 1)
