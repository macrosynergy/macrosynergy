# Sharpe ratio & Sharpe Stability Ratio for `single_statistic_table` — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. **Work on branch `feature/srr-sharpe-ssr`. Do NOT push to `origin` (github.com/macrosynergy/macrosynergy) until a human has reviewed.**

**Goal:** Add two new statistics to `SignalReturnRelations.single_statistic_table` — an annualized **Sharpe ratio** (`"sharpe"`) of the signal-conditioned strategy, and a **Sharpe Stability Ratio** (`"ssr"`) that can occupy the secondary/bracketed slot currently reserved for p-values — and generalize the heatmap annotation/highlight path so the secondary slot can render a raw t-stat (the SSR) instead of only a `1 - pval` probability.

**Architecture:** Both new stats are computed from a **daily** signal-conditioned strategy return series, NOT from the frequency-resampled `self.df` that the existing correlation/accuracy stats use. The signal is rebalanced at the analysis `freq` (e.g. weekly) but the PnL accrues **daily**, so the Sharpe/SSR annualization is always daily (252). We reuse two existing, well-tested pieces verbatim: `NaivePnL.rebalancing()` (static, instance-free) for "hold the low-frequency signal across daily dates", and `macrosynergy.pnl.sharpe_stability_ratio()` for the SSR math. The only genuinely new code is (a) a private helper that assembles the daily strategy-return series from `self.original_df`, (b) two `elif` branches in `calculate_single_stat`, and (c) a small generalization of the secondary-stat display path so a "higher-is-more-significant" metric (the SSR t-stat) is shown raw and thresholded directly instead of being transformed by `1 - x`.

**Tech Stack:** Python, pandas, numpy, scipy, statsmodels (already a dependency of `sharpe_stability_ratio`), unittest, the repo's `make_qdf` simulator.

---

## Context the implementer must read first

Read these before writing any code. The plan quotes them, but you must see them in context:

- `macrosynergy/signal/signal_return_relations.py`
  - `__init__` metrics list — **lines 179–205** (where stat names are registered).
  - `manipulate_df` — **lines 725–786** (how `self.df` is built: `reduce_df` → `apply_slip` → `categories_df(..., freq=freq, lag=1, ...)` pivot to MultiIndex `(cid, real_date)` × `[ret, sig]`). `self.original_df` (set at **line 289**) is the **long** QuantamentalDataFrame `[cid, xcat, real_date, value]` at the **native (daily)** input frequency.
  - `calculate_single_stat` — **lines 1067–1186** (the `if/elif` dispatch you will extend; note it currently reads the *resampled* `self.df`).
  - `single_statistic_table` — **lines ~1495–1952**; specifically the secondary-stat plumbing: `pval_stat` validation **1684–1691**, the `df_pval` table build **1747–1752 / 1775–1778**, and the display block **1863–1950** (`df_psig = 1.0 - df_pval` at **1875**, `_format_dual_annot` call at **1878**, `highlight_mask = df_psig > threshold` at **1886–1888**).
  - `_format_dual_annot` — **lines 2173–2201** (builds the `"<stat>\n(<secondary>)"` cell strings).
  - `__slice_df__` — **line 650** (segment slicing by cid/year used inside `calculate_single_stat`).
- `macrosynergy/pnl/naive_pnl.py`
  - `rebalancing(dfw, rebal_freq="daily", rebal_slip=0)` — **lines 653–740**, a `@staticmethod`. Input: a frame with columns `["real_date", "psig", "cid"]` (the daily, already-1-day-lagged signal in `psig`). It resamples to the first date of each rebal period, forward-fills across daily dates, applies `rebal_slip`. Returns a `pd.Series` indexed by `real_date` (sorted by `cid` then `real_date`). **Note the rebal-freq vocabulary: `"daily" | "weekly" | "monthly" | "quarterly" | "annual"`** — NOT the `"D"/"W"/"M"/"Q"` codes used by `SignalReturnRelations.freqs`. You must map between them (Task 1).
  - `make_pnl` lag convention — the signal gets a `.shift(1)` (one-day lag, **line 349**) *before* rebalancing; the strategy return is `position * daily_return` (**line 393**).
- `macrosynergy/pnl/sharpe_stability_ratio.py`
  - `sharpe_stability_ratio(returns, window=252, benchmark_sr=0.0, annualization_factor=252, min_periods=None)` — entire file (200 lines). Takes a daily return series, returns the HAC-robust t-stat (or `NaN` if `len(ret) < window + 2`, or the rolling-Sharpe series is degenerate). For daily input keep `window=252, annualization_factor=252`.
- `tests/unit/signal/test_signal_return_relations.py`
  - Fixture `setUp` — **lines 17–72** (`make_qdf` with `random.seed(2)`, cids `["AUD","CAD","GBP","NZD","USD"]`, xcats `["XR","CRY","GROWTH","INFL"]`, a blacklist).
  - `test__output_table__` — **lines 236–332** (the canonical pattern: recompute a stat independently with scipy/pandas and assert `abs(impl - manual) < tol`). Mirror this for the new stats.
  - `test_single_statistic_table` — **lines 861–929** (shape & validation tests; `assertRaises(ValueError)` for bad stat names).
  - Heatmap smoke-test pattern — **lines ~1040–1085** (`matplotlib.use("Agg")`, call with `show_heatmap=True`, `self.fail` on exception).
  - `tests/unit/pnl/` — find and read the existing `sharpe_stability_ratio` tests to match tolerance/style conventions.

---

## Design decisions (locked, but FLAG for human review)

These were decided with the requester; call them out explicitly in the PR description so the reviewer can confirm.

1. **Position convention = `np.sign(signal)` (long/short ±1), default.** This mirrors the entire existing `SignalReturnRelations` philosophy (accuracy, AUC, precision all use `np.sign`) and needs no z-scoring/winsorization machinery (YAGNI). Expose it as a constructor/​method-level knob `sharpe_position: str = "sign"` with the only other accepted value `"raw"` (use the signal value itself as the position weight) so a reviewer can switch to magnitude-aware positions without a rewrite. **Recommendation: ship `"sign"` as default; document the trade-off (sign discards the cyclical-strength magnitude) in the docstring.**
2. **PnL is daily; rebalancing is at `freq`.** The Sharpe/SSR annualization factor is always **252** regardless of the rebalance frequency. `freq` only controls how often the position updates.
3. **SSR is kept as a raw t-stat** in the bracketed slot — NOT converted to a probability. The display path must therefore skip the `1 - x` transform for "score-style" secondary stats and threshold the raw value (default highlight at **`1.96`** ≈ 95% confidence). A future probability transform (normal CDF) is explicitly out of scope.
4. **Segment support:** `"sharpe"` and `"ssr"` support `type="panel"` (aggregate the daily strategy return across cids into one series) and `type="mean_cids"` (per-cid stat, then mean). For `type in {"mean_years","pr_years","pr_cids"}` they return `NaN` with a `warnings.warn` (a single year is too short for a 252-day rolling SSR, and a positive-ratio of a Sharpe is not meaningful). The headline use case (`cids=["GLB"]`, `type="panel"`) is fully covered.

---

## File structure

| File | Change |
| --- | --- |
| `macrosynergy/signal/signal_return_relations.py` | Add `"sharpe"`, `"ssr"` to `self.metrics`; add `sharpe_position` knob; add private helpers `_freq_to_rebal_freq`, `_daily_strategy_returns`, `_score_style_secondary` set; add two `elif` branches in `calculate_single_stat`; generalize the secondary-stat display block + `_format_dual_annot` call. |
| `tests/unit/signal/test_signal_return_relations.py` | New tests: value-correctness for `sharpe` & `ssr` (independent recompute), validation, `type` guards, secondary-slot display (raw t-stat, no `1-x`), heatmap smoke test. |

No new modules. Everything reuses `naive_pnl.rebalancing` and `pnl.sharpe_stability_ratio` by import.

---

## Task 1: Frequency-code → rebal-freq mapping helper

**Files:**
- Modify: `macrosynergy/signal/signal_return_relations.py` (add a module-level constant + staticmethod near the other helpers, e.g. just above `calculate_single_stat` ~line 1066)
- Test: `tests/unit/signal/test_signal_return_relations.py`

`SignalReturnRelations.freqs` uses codes `"D","W","M","Q","A"`; `NaivePnL.rebalancing` expects words `"daily","weekly","monthly","quarterly","annual"`. Bridge them.

- [ ] **Step 1: Write the failing test**

Add to the test class:

```python
def test_freq_to_rebal_freq(self):
    from macrosynergy.signal.signal_return_relations import SignalReturnRelations as S
    self.assertEqual(S._freq_to_rebal_freq("D"), "daily")
    self.assertEqual(S._freq_to_rebal_freq("W"), "weekly")
    self.assertEqual(S._freq_to_rebal_freq("M"), "monthly")
    self.assertEqual(S._freq_to_rebal_freq("Q"), "quarterly")
    self.assertEqual(S._freq_to_rebal_freq("A"), "annual")
    with self.assertRaises(ValueError):
        S._freq_to_rebal_freq("BOGUS")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/signal/test_signal_return_relations.py::TestAll::test_freq_to_rebal_freq -v`
Expected: FAIL — `AttributeError: ... has no attribute '_freq_to_rebal_freq'`.

- [ ] **Step 3: Write minimal implementation**

Add near the top of the class body (module-level constant above the class, mapping inside a staticmethod):

```python
_FREQ_TO_REBAL_FREQ = {
    "D": "daily",
    "W": "weekly",
    "M": "monthly",
    "Q": "quarterly",
    "A": "annual",
}

@staticmethod
def _freq_to_rebal_freq(freq: str) -> str:
    """Translate a SignalReturnRelations frequency code to the word form
    expected by ``NaivePnL.rebalancing``."""
    try:
        return SignalReturnRelations._FREQ_TO_REBAL_FREQ[freq]
    except KeyError:
        raise ValueError(
            f"Unsupported frequency {freq!r}; expected one of "
            f"{list(SignalReturnRelations._FREQ_TO_REBAL_FREQ)}."
        )
```

(Place `_FREQ_TO_REBAL_FREQ` as a class attribute so it is reachable as `SignalReturnRelations._FREQ_TO_REBAL_FREQ`.)

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/signal/test_signal_return_relations.py::TestAll::test_freq_to_rebal_freq -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add macrosynergy/signal/signal_return_relations.py tests/unit/signal/test_signal_return_relations.py
git commit -m "feat(signal): add freq-code to rebal-freq mapping helper"
```

---

## Task 2: Register the new stats and the `sharpe_position` knob

**Files:**
- Modify: `macrosynergy/signal/signal_return_relations.py:179-205` (metrics list) and the `__init__` signature/body.
- Test: `tests/unit/signal/test_signal_return_relations.py`

- [ ] **Step 1: Write the failing test**

```python
def test_sharpe_ssr_registered(self):
    sr = SignalReturnRelations(
        df=self.dfd, rets="XR", sigs="CRY", freqs="W",
        blacklist=self.blacklist, slip=1,
    )
    self.assertIn("sharpe", sr.metrics)
    self.assertIn("ssr", sr.metrics)
    self.assertEqual(sr.sharpe_position, "sign")
    with self.assertRaises(ValueError):
        SignalReturnRelations(
            df=self.dfd, rets="XR", sigs="CRY", freqs="W",
            blacklist=self.blacklist, slip=1, sharpe_position="BOGUS",
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/signal/test_signal_return_relations.py::TestAll::test_sharpe_ssr_registered -v`
Expected: FAIL — `"sharpe" not in sr.metrics` / unexpected `sharpe_position` kwarg.

- [ ] **Step 3: Write minimal implementation**

Add `sharpe_position: str = "sign"` to the `__init__` signature. Validate and store it near the other `__init__` validations:

```python
if sharpe_position not in ("sign", "raw"):
    raise ValueError("sharpe_position must be one of {'sign', 'raw'}.")
self.sharpe_position = sharpe_position
```

Extend the metrics list (after `"auc"`, before the `ms_panel_test`/`additional_metrics` blocks at lines 191–205):

```python
self.metrics = [
    "accuracy",
    "bal_accuracy",
    "pos_sigr",
    "pos_retr",
    "pos_prec",
    "neg_prec",
    "pearson",
    "pearson_pval",
    "kendall",
    "kendall_pval",
    "auc",
    "sharpe",
    "ssr",
]
```

Also define the class-level set used later by the display path:

```python
# Secondary stats whose magnitude grows with significance (t-stat style):
# shown raw in the bracketed slot, thresholded directly (no 1 - x transform).
_SCORE_STYLE_SECONDARY = {"ssr"}
```

> **Caution — the `pr_*` index slicing at lines 1181–1186 is positional.** It does `stat in self.metrics[0:6] + ["auc"]`, `self.metrics[6:9:2]`, `self.metrics[7:10:2]`. Appending `"sharpe"`/`"ssr"` at the **end** keeps indices 0–10 intact, so those slices are unaffected. **Do not insert the new names earlier in the list.** Add an explicit comment at the list noting this positional dependency.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/signal/test_signal_return_relations.py::TestAll::test_sharpe_ssr_registered -v`
Expected: PASS.

Also run the existing positional-slice-sensitive test to confirm no regression:
Run: `pytest tests/unit/signal/test_signal_return_relations.py::TestAll::test_single_statistic_table -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add macrosynergy/signal/signal_return_relations.py tests/unit/signal/test_signal_return_relations.py
git commit -m "feat(signal): register sharpe/ssr metrics and sharpe_position knob"
```

---

## Task 3: Daily strategy-return helper `_daily_strategy_returns`

**Files:**
- Modify: `macrosynergy/signal/signal_return_relations.py` (new private method, place just above `calculate_single_stat`).
- Test: `tests/unit/signal/test_signal_return_relations.py`

This is the core new logic. It builds a **daily** strategy-return series for one `(ret, sig)` pair at rebalance frequency `freq`, from `self.original_df` (the long, daily QDF) — bypassing the resampled `self.df`. It reuses `NaivePnL.rebalancing` for the hold-across-days step.

**Algorithm (mirrors `make_pnl` minimal core):**
1. `reduce_df(self.original_df, xcats=[sig, ret], cids=css_or_self.cids, start, end, blacklist)` then `apply_slip` on the **signal** xcat only (same as `manipulate_df` lines 759–765). Keep it **daily** (no `categories_df` freq resampling).
2. Pivot long→wide per cid: a daily frame with columns `[sig, ret]` indexed by `(cid, real_date)`. Use the same `categories_df(...)` call as `manipulate_df` **but with `freq="D"` and `lag=0`** (we apply the 1-day signal lag ourselves to match `make_pnl`), or pivot directly with `pandas`. **Verify which is correct against `categories_df`'s lag semantics during implementation** — the invariant the test in Step 1 pins down is: position on day *t* uses signal observed on day *t−1*, held constant until the next rebalance date.
3. Build the rebalancing input frame with columns `["real_date", "psig", "cid"]` where `psig` = the **1-day-lagged** signal position. Apply the position convention: `psig = np.sign(signal_lagged)` if `self.sharpe_position == "sign"` else `signal_lagged`.
4. `sig_series = NaivePnL.rebalancing(dfw, rebal_freq=self._freq_to_rebal_freq(freq), rebal_slip=0)` → daily held position.
5. `daily_pnl_per_cid = held_position * daily_return` aligned on `(cid, real_date)`.
6. Return a **dict `{cid: daily_return_series}`** (each a `pd.Series` indexed by `real_date`, NaNs dropped). The caller aggregates: `panel` → align and **sum across cids** into one series; `mean_cids` → keep per-cid.

> Returning per-cid series (not a pre-aggregated panel) keeps the helper reusable for both `panel` and `mean_cids` and makes the aggregation explicit at the call site.

- [ ] **Step 1: Write the failing test (pins the lag + rebalance invariant)**

Construct a tiny deterministic daily panel so the expected daily PnL can be computed by hand, independent of the rebalancing internals:

```python
def test_daily_strategy_returns_lag_and_hold(self):
    import pandas as pd, numpy as np
    # One cid, ~3 weeks of business days, signal flips sign once.
    dates = pd.bdate_range("2020-01-06", periods=15)  # Mondays start weeks
    sig_vals = np.array([1.0]*7 + [-1.0]*8)           # flips mid-series
    ret_vals = np.arange(1, 16, dtype=float) / 1000.0 # 0.001 .. 0.015
    rows = []
    for d, s, r in zip(dates, sig_vals, ret_vals):
        rows.append({"cid": "USD", "xcat": "SIG", "real_date": d, "value": s})
        rows.append({"cid": "USD", "xcat": "RET", "real_date": d, "value": r})
    df = pd.DataFrame(rows)

    sr = SignalReturnRelations(df=df, rets="RET", sigs="SIG", freqs="W", slip=0)
    out = sr._daily_strategy_returns(ret="RET", sig="SIG", freq="W")
    self.assertIn("USD", out)
    series = out["USD"]
    # Position on day t = sign(signal_{t-1}), held to the weekly rebalance date.
    # Assert: first day has no position (NaN/0 -> dropped); a known mid-series
    # day where signal was +1 the prior week yields pnl = +1 * ret_that_day.
    # (Fill in the exact expected index/value after reading categories_df lag
    #  semantics; the invariant is position_t = sign(sig_{prior rebalance}).)
    self.assertTrue((series.dropna().abs() <= ret_vals.max() + 1e-12).all())
    # No look-ahead: strategy return on the very first available date must not
    # use that same date's signal.
    self.assertGreaterEqual(len(series.dropna()), 1)
```

> The implementer must tighten the two commented assertions into exact numeric checks once the pivot/lag mechanics are confirmed (read `categories_df`). The non-negotiable invariant: **no same-day signal in the position** (no look-ahead), and **position held constant between weekly rebalance dates**.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/signal/test_signal_return_relations.py::TestAll::test_daily_strategy_returns_lag_and_hold -v`
Expected: FAIL — `AttributeError: ... '_daily_strategy_returns'`.

- [ ] **Step 3: Write minimal implementation**

```python
def _daily_strategy_returns(self, ret: str, sig: str, freq: str,
                            css: Optional[List[str]] = None) -> Dict[str, pd.Series]:
    """Daily signal-conditioned strategy returns per cross-section.

    The signal position is rebalanced at ``freq`` (held constant between
    rebalance dates) but the PnL accrues daily, so downstream Sharpe/SSR
    annualization is daily (252). Reuses ``NaivePnL.rebalancing`` for the
    hold-across-days step. Returns ``{cid: daily_return_series}``.
    """
    from macrosynergy.pnl.naive_pnl import NaivePnL

    cids = css if css is not None else self.cids
    dfd = reduce_df(
        self.original_df, xcats=[sig, ret], cids=cids,
        start=self.start, end=self.end, blacklist=self.blacklist,
    )
    metric_cols = list(
        set(dfd.columns) - {"real_date", "xcat", "cid", "ticker", "last_updated"}
    )
    dfd = self.apply_slip(df=dfd, slip=self.slip, cids=cids,
                          xcats=[sig], metrics=metric_cols)

    out: Dict[str, pd.Series] = {}
    rebal_freq = self._freq_to_rebal_freq(freq)
    for cid in sorted(dfd["cid"].unique()):
        wide = (
            dfd[dfd["cid"] == cid]
            .pivot_table(index="real_date", columns="xcat", values="value")
            .sort_index()
        )
        if sig not in wide or ret not in wide:
            continue
        # 1-day lag on the signal (no look-ahead), then position convention.
        lagged = wide[sig].shift(1)
        position = np.sign(lagged) if self.sharpe_position == "sign" else lagged
        dfw = pd.DataFrame({
            "real_date": wide.index,
            "psig": position.values,
            "cid": cid,
        })
        held = NaivePnL.rebalancing(dfw, rebal_freq=rebal_freq, rebal_slip=0)
        held = held["psig"].reindex(wide.index)  # daily held position
        pnl = (held.values * wide[ret].values)
        out[cid] = pd.Series(pnl, index=wide.index).dropna()
    return out
```

> **Implementation notes for the engineer:** (1) Confirm `reduce_df`, `apply_slip` are already imported in this module (they are — `manipulate_df` uses them). (2) `Dict` must be imported from `typing` (check the existing imports; add if missing). (3) `NaivePnL.rebalancing` mutates/returns a frame indexed by `real_date`; the `.reindex(wide.index)` realigns it to this cid's daily grid. Verify the returned Series column name is `"psig"` (it drops `"cid"`; if it returns a single-column frame, take that column). (4) Do **not** import `NaivePnL` at module top if it introduces a circular import — the local import inside the method avoids that risk; verify.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/signal/test_signal_return_relations.py::TestAll::test_daily_strategy_returns_lag_and_hold -v`
Expected: PASS (after tightening the numeric assertions).

- [ ] **Step 5: Commit**

```bash
git add macrosynergy/signal/signal_return_relations.py tests/unit/signal/test_signal_return_relations.py
git commit -m "feat(signal): daily signal-conditioned strategy-return helper"
```

---

## Task 4: `sharpe` and `ssr` branches in `calculate_single_stat`

**Files:**
- Modify: `macrosynergy/signal/signal_return_relations.py:1119-1186` (inside `calculate_single_stat`).
- Test: `tests/unit/signal/test_signal_return_relations.py`

The existing loop in `calculate_single_stat` iterates segments of the **resampled** `self.df`. The new stats ignore that per-segment resampled data and instead call `_daily_strategy_returns`. Implement them **before** the per-segment `for cs in css:` loop (early-return), because their aggregation model differs from the sign/correlation stats.

- [ ] **Step 1: Write the failing value-correctness test**

Mirror `test__output_table__`'s recompute-and-compare style. Use daily freq and a single signal/return so the expected Sharpe can be recomputed independently:

```python
def test_sharpe_value(self):
    import numpy as np
    signal, return_ = "CRY", "XR"
    sr = SignalReturnRelations(
        df=self.dfd, rets=return_, sigs=signal, freqs="D",
        blacklist=self.blacklist, slip=1,
    )
    sr.df = sr.original_df.copy()
    sr.manipulate_df(xcats=[signal, return_], freq="D", agg_sig="last")
    impl = sr.calculate_single_stat("sharpe", ret=return_, sig=signal, type="panel")

    # Independent recompute: daily PnL = sign(sig_{t-1}) * ret_t, summed across
    # cids, Sharpe = mean/std*sqrt(252).
    series_by_cid = sr._daily_strategy_returns(return_, signal, "D")
    panel = None
    for s in series_by_cid.values():
        panel = s if panel is None else panel.add(s, fill_value=0.0)
    panel = panel.dropna()
    expected = panel.mean() / panel.std() * np.sqrt(252)
    self.assertTrue(abs(impl - expected) < 1e-9)

def test_ssr_value(self):
    from macrosynergy.pnl import sharpe_stability_ratio
    signal, return_ = "CRY", "XR"
    sr = SignalReturnRelations(
        df=self.dfd, rets=return_, sigs=signal, freqs="W",
        blacklist=self.blacklist, slip=1,
    )
    impl = sr.calculate_single_stat("ssr", ret=return_, sig=signal, type="panel")
    series_by_cid = sr._daily_strategy_returns(return_, signal, "W")
    panel = None
    for s in series_by_cid.values():
        panel = s if panel is None else panel.add(s, fill_value=0.0)
    expected = sharpe_stability_ratio(panel.dropna(), window=252,
                                      annualization_factor=252)
    # Both NaN, or both finite & equal.
    import numpy as np
    if np.isnan(expected):
        self.assertTrue(np.isnan(impl))
    else:
        self.assertTrue(abs(impl - expected) < 1e-9)

def test_sharpe_ssr_type_guard(self):
    import numpy as np, warnings
    sr = SignalReturnRelations(
        df=self.dfd, rets="XR", sigs="CRY", freqs="W",
        blacklist=self.blacklist, slip=1,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for t in ("mean_years", "pr_years", "pr_cids"):
            self.assertTrue(np.isnan(
                sr.calculate_single_stat("sharpe", ret="XR", sig="CRY", type=t)))
            self.assertTrue(np.isnan(
                sr.calculate_single_stat("ssr", ret="XR", sig="CRY", type=t)))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/signal/test_signal_return_relations.py::TestAll -k "sharpe_value or ssr_value or sharpe_ssr_type_guard" -v`
Expected: FAIL — `ValueError("Invalid statistic.")` from `calculate_single_stat`.

- [ ] **Step 3: Write minimal implementation**

Insert an early branch at the **top** of `calculate_single_stat`, immediately after the `type` → `css/cs_type` resolution block (after line 1117, before `list_of_results = []` at 1119):

```python
        if stat in ("sharpe", "ssr"):
            if type in ("mean_years", "pr_years", "pr_cids"):
                warnings.warn(
                    f"Statistic {stat!r} is not defined for type={type!r}; "
                    "returning NaN. Use type='panel' or 'mean_cids'."
                )
                return float("nan")
            # freq is encoded in self.df's construction; recover it from the
            # rebalance context via the single freq active in this call.
            freq = self._active_freq  # set by single_statistic_table; see Task 5
            css_arg = None if type == "panel" else sorted(set(self.cids))
            series_by_cid = self._daily_strategy_returns(ret, sig, freq, css=css_arg)
            if not series_by_cid:
                return float("nan")

            def _sharpe(s):
                s = s.dropna()
                if len(s) < 2 or s.std() == 0:
                    return float("nan")
                return float(s.mean() / s.std() * np.sqrt(252))

            def _ssr(s):
                return sharpe_stability_ratio(
                    s.dropna(), window=252, annualization_factor=252
                )

            metric_fn = _sharpe if stat == "sharpe" else _ssr
            if type == "panel":
                panel = None
                for s in series_by_cid.values():
                    panel = s if panel is None else panel.add(s, fill_value=0.0)
                return metric_fn(panel)
            else:  # mean_cids
                vals = [metric_fn(s) for s in series_by_cid.values()]
                vals = [v for v in vals if not np.isnan(v)]
                return float(np.mean(vals)) if vals else float("nan")
```

Add the import at the top of the module (near the other `macrosynergy` imports):

```python
from macrosynergy.pnl import sharpe_stability_ratio
```

> **Note on `freq`:** `calculate_single_stat`'s signature does not currently receive `freq`. Task 5 threads it via a transient `self._active_freq` set inside the `single_statistic_table` loop. Alternatively (cleaner, preferred if low-risk) **add a `freq: str = None` parameter to `calculate_single_stat`** and pass it from the loop at line 1772. Choose the parameter approach if it does not break other callers of `calculate_single_stat` (grep for call sites first — `single_relation_table`, bar charts, etc. — and default `freq=None`, falling back to `self.freqs[0]`). Document whichever you pick.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/signal/test_signal_return_relations.py::TestAll -k "sharpe_value or ssr_value or sharpe_ssr_type_guard" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add macrosynergy/signal/signal_return_relations.py tests/unit/signal/test_signal_return_relations.py
git commit -m "feat(signal): compute sharpe and ssr in calculate_single_stat"
```

---

## Task 5: Thread `freq` into the stat call & make `single_statistic_table` accept the new stats

**Files:**
- Modify: `macrosynergy/signal/signal_return_relations.py` loop **1764–1778** (and `calculate_single_stat` signature if you took the parameter route in Task 4).
- Test: `tests/unit/signal/test_signal_return_relations.py`

- [ ] **Step 1: Write the failing test**

```python
def test_single_statistic_table_sharpe_ssr(self):
    import pandas as pd
    sr = SignalReturnRelations(
        df=self.dfd, rets=["XR", "GROWTH"], sigs=["CRY", "INFL"],
        freqs="W", blacklist=self.blacklist, slip=1,
    )
    tbl = sr.single_statistic_table(stat="sharpe")
    self.assertIsInstance(tbl, pd.DataFrame)
    self.assertEqual(tbl.shape, (2, 2))   # 2 sigs x 2 rets, single freq/agg
    tbl_ssr = sr.single_statistic_table(stat="ssr")
    self.assertIsInstance(tbl_ssr, pd.DataFrame)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/signal/test_signal_return_relations.py::TestAll::test_single_statistic_table_sharpe_ssr -v`
Expected: FAIL — either an `AttributeError` on `self._active_freq` or a wrong/NaN result if `freq` is not threaded.

- [ ] **Step 3: Write minimal implementation**

In the loop body (line ~1767), set the active freq before computing the stat:

```python
        for ret, sig, freq, agg_sig in loop_tuples:
            xcat = [sig, ret]
            self.manipulate_df(xcats=xcat, freq=freq, agg_sig=agg_sig)
            self._active_freq = freq  # consumed by calculate_single_stat for sharpe/ssr
            hash = f"{ret}/{sig}/{freq}/{agg_sig}"
            ...
```

(Or, if you took the parameter route: `self.calculate_single_stat(stat, ret, sig, type, freq=freq)`.)

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/signal/test_signal_return_relations.py::TestAll::test_single_statistic_table_sharpe_ssr -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add macrosynergy/signal/signal_return_relations.py tests/unit/signal/test_signal_return_relations.py
git commit -m "feat(signal): thread freq into single_statistic_table for sharpe/ssr"
```

---

## Task 6: Generalize the secondary-stat display for the SSR t-stat

**Files:**
- Modify: `macrosynergy/signal/signal_return_relations.py:1863-1888` (display block) and `_format_dual_annot` call.
- Test: `tests/unit/signal/test_signal_return_relations.py`

Currently the secondary slot is hard-wired to p-values: `df_psig = 1.0 - df_pval`, bracket shows `df_psig`, highlight where `df_psig > significance_threshold` (default 0.9). For `pval_stat="ssr"` (a score-style metric) we must **show the raw SSR** and threshold it directly with a t-stat default (1.96).

- [ ] **Step 1: Write the failing test**

```python
def test_single_statistic_table_ssr_secondary_raw(self):
    import numpy as np, matplotlib
    backend = matplotlib.get_backend(); matplotlib.use("Agg")
    try:
        sr = SignalReturnRelations(
            df=self.dfd, rets="XR", sigs="CRY", freqs="W",
            blacklist=self.blacklist, slip=1,
        )
        # The bracketed annotation must equal the raw SSR (NOT 1 - ssr).
        annot = sr._format_dual_annot  # exists already
        df_stat = sr.single_statistic_table(stat="sharpe")
        df_sec = sr.single_statistic_table(stat="ssr")
        out = annot(df_stat, df_sec, 3, 3)
        # find a finite SSR cell and assert it appears verbatim (rounded) in annot
        for r in df_sec.index:
            for c in df_sec.columns:
                v = df_sec.loc[r, c]
                if isinstance(v, float) and not np.isnan(v):
                    self.assertIn(f"{v:.3f}", str(out.loc[r, c]))
                    self.assertNotIn(f"{1.0 - v:.3f}", str(out.loc[r, c]))
                    break
        # smoke: heatmap path with ssr as the secondary stat must not raise
        sr.single_statistic_table(
            stat="sharpe", pval_stat="ssr", show_heatmap=True,
            significance_threshold=1.96,
        )
    except Exception as e:
        self.fail(f"ssr secondary display raised {e}")
    finally:
        import matplotlib.pyplot as plt; plt.close("all"); matplotlib.use(backend)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/signal/test_signal_return_relations.py::TestAll::test_single_statistic_table_ssr_secondary_raw -v`
Expected: FAIL — the heatmap path computes `1.0 - df_pval`, so the bracket shows `1 - ssr` and the assertion `assertNotIn(1 - v)` fails (and the highlight threshold is wrong-scaled).

- [ ] **Step 3: Write minimal implementation**

Replace the display block at lines 1872–1888 with a branch on score-style vs pval-style:

```python
            # Secondary stat: p-value style (lower = more significant) is
            # converted to probability of significance (1 - pval); score
            # style (e.g. the SSR t-stat, higher = more significant) is shown
            # raw and thresholded directly.
            if df_pval is not None:
                score_style = pval_stat in self._SCORE_STYLE_SECONDARY
                df_psig = df_pval.copy() if score_style else (1.0 - df_pval)
                default_thr = 1.96 if score_style else 0.9
            else:
                df_psig = None
                default_thr = None

            if annotate and df_psig is not None:
                heatmap_annot = self._format_dual_annot(
                    df_result, df_psig, round, round_pval
                )
                heatmap_fmt = ""
            else:
                heatmap_annot = annotate
                heatmap_fmt = f".{round}f"

            highlight_mask = None
            if df_psig is not None:
                thr = significance_threshold
                # Apply a score-style default only when the caller left the
                # p-value default (0.9) untouched.
                if score_style and significance_threshold == 0.9:
                    thr = default_thr
                if thr is not None:
                    highlight_mask = df_psig > float(thr)
```

> Keep the `significance_threshold` parameter default at `0.9` (do not break existing p-value callers). The "caller left default" heuristic above swaps in `1.96` only for score-style secondaries. If the reviewer prefers an explicit `None` sentinel default instead of the `== 0.9` heuristic, that is a clean follow-up — note it in the PR.

Also relax the `pval_stat` validation at lines 1684–1691 so `"ssr"` is accepted (it already is, since `"ssr"` is in `self.metrics`; just confirm no `map_pval`-style special-case blocks it).

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/signal/test_signal_return_relations.py::TestAll::test_single_statistic_table_ssr_secondary_raw -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add macrosynergy/signal/signal_return_relations.py tests/unit/signal/test_signal_return_relations.py
git commit -m "feat(signal): render raw SSR t-stat in secondary heatmap slot"
```

---

## Task 7: Full regression + docstring + changelog

**Files:**
- Modify: `single_statistic_table` docstring (document `stat="sharpe"`, `stat="ssr"`, `pval_stat="ssr"`, `sharpe_position`, the daily-PnL/252 convention, and the `type` restriction).
- Modify: repo changelog / release notes if the repo keeps one (check `docs/` or `CHANGELOG`).

- [ ] **Step 1: Update the docstrings**

In `single_statistic_table` and `show_single_statistic_table`, add to the `stat` description that `"sharpe"` is the annualized (252) Sharpe of the daily signal-conditioned strategy (position = `sharpe_position`), and `"ssr"` is the Sharpe Stability Ratio (HAC t-stat). Document that `pval_stat="ssr"` renders the raw t-stat in brackets (default highlight 1.96), and that both stats support only `type in {"panel","mean_cids"}`. Document the new `__init__` `sharpe_position` knob.

- [ ] **Step 2: Run the full signal test module**

Run: `pytest tests/unit/signal/test_signal_return_relations.py -rEf --verbose`
Expected: PASS (all pre-existing tests + the new ones).

- [ ] **Step 3: Run the PnL tests (we import `rebalancing` and `sharpe_stability_ratio`)**

Run: `pytest tests/unit/pnl/ -rEf --verbose`
Expected: PASS (no regression — we only import, never modify those modules).

- [ ] **Step 4: Run the broader unit suite touched by the module**

Run: `pytest tests/unit/signal/ -rEf --verbose`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "docs(signal): document sharpe/ssr stats and sharpe_position"
```

---

## Acceptance criteria (the requester's complemented use case)

After Task 6, this must run and render a heatmap whose cells show the Sharpe with the raw SSR t-stat in brackets, highlighting cells where SSR > 1.96:

```python
srr = mss.SignalReturnRelations(
    dfx, sigs=sigs, rets=rets_panel, cids=["GLB"],
    ms_panel_test=False, **srr_kwargs,
)
_ = srr.single_statistic_table(
    stat="sharpe",
    pval_stat="ssr",
    significance_threshold=1.96,
    title="Global cyclical-strength surprises and subsequent global returns",
    min_color=-1.0, max_color=1.0,
    xlabel="Next-week global returns",
    footnote="Global GDP-weighted signal on cid GLB; sign positions rebalanced "
             "weekly, one-day slip; daily PnL; annualized Sharpe with SSR t-stat.",
    return_name_dict={r: ret_labels[r] for r in rets_panel},
    figsize=(16, fig_h),
    **heatmap_kwargs,
)
```

(Note for the academy notebook: `min_color/max_color` were tuned for correlations [−0.1, 0.1]; widen for Sharpe.)

---

## Self-review checklist (run before opening the PR)

1. **Spec coverage:** Sharpe stat ✓ (Task 4), SSR stat ✓ (Task 4), SSR-in-secondary-slot as raw t-stat ✓ (Task 6), daily-PnL-from-weekly-signal reusing `rebalancing` ✓ (Task 3), reuse of existing `sharpe_stability_ratio` ✓ (Task 4), solid tests with independent recompute ✓ (Tasks 4 & 6).
2. **No look-ahead:** Task 3 test pins the 1-day lag.
3. **Positional-slice safety:** Task 2 appends new metrics at the end; `pr_*` slices (lines 1181–1186) untouched.
4. **No push:** confirm `git log origin/feature/srr-sharpe-ssr` does not exist; branch stays local pending review.
5. **Open items flagged for reviewer:** position convention (`sign` vs `raw`); the `significance_threshold == 0.9` default-swap heuristic; whether to thread `freq` as a parameter vs `self._active_freq`. List these in the PR description.
