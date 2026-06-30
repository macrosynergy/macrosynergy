# macrosynergy performance — book of work (`feature/performance`)

> **Status: baseline complete (all 24 cells profiled), targets ranked and designed.**
> **No package code has been changed.** This is the FIRST-TASK deliverable — diagnose + rank +
> design — stopping here for review before implementing any `perf/<slug>` sub-branch.
>
> **TL;DR ranking** (full rationale in §3): **T2c** `_get_tickers_series` vectorize (lead cost
> of the two slowest cells, 18 & 27) → **T1** `update_df` factorized sort (breadth + the 419s
> cell 17) → **T2** `split_ticker`/`ticker_df_to_qdf` vectorize (cell 13, 244s/12 GiB) → **T3**
> `reduce_df` fast dedup (broad) → **T4** parallelize the SRR MixedLM panel test (25% of total,
> but higher risk). T2c, T1, T2, T3 are all object↔categorical string-handling fixes with
> byte-identical output and together address the ~74% of runtime in the building chain.

## 1. What this measures

Driver notebook: `academy/drafts/surprises/Cyclical strength composite.ipynb` (the
cyclical-strength economic-surprise composite — the slow notebook). It is exercised by an
offline per-cell benchmark/profile harness in `academy/drafts/surprises/performance/`
(Scope-1). The harness reconstructs each of 24 benchmarked cells' exact on-entry working
dataframe `dfx` from frozen parquet fixtures and runs that cell's macrosynergy calls in
isolation, measuring wall time, peak RSS, and per-function cost (cProfile).

- macrosynergy is an **editable install** of this checkout (`pip show macrosynergy` → Location
  `~/repos/macrosynergy`; version `1.7.1dev0+cbd1503`), so all measurements reflect
  `feature/performance` (branched from `upstream/develop` @ `cbd15037`).

### Method notes / caveats (so later runs are comparable)

- **Fixtures are frozen.** They were generated against the previous install (v1.6.1) but are
  pure data snapshots (the on-entry `dfx` per cell). The GATE compares before/after on
  *identical* inputs and the output-parity golden is captured by running develop's code on
  these same fixtures, so provenance does not affect validity — only that inputs stay constant
  across all before/after runs. Do **not** regenerate fixtures mid-campaign.
- **Baseline profiler = lightweight (no tracemalloc).** The harness's `profile_cells.py` runs
  `tracemalloc`, which is pathologically slow on allocation-heavy code (e.g.
  `InformationStateChanges` building per-ticker dicts → cell 13 took 7+ min and did not
  finish under tracemalloc). Per the harness README, **RSS (psutil) is the truer memory
  signal; tracemalloc Python-heap numbers undercount numpy anyway.** So the baseline here uses
  a lightweight driver (wall + peak RSS via psutil + cProfile, sort by self-time) that drops
  only the weaker memory metric and makes the sweep tractable. The committed `profile_cells.py`
  is unchanged; the lightweight driver lives outside the repo.
- "peak RSS" is the **delta** over the cell (process RSS rise during the call), sampled at
  20 ms. Numbers are single-run; treat as order-of-magnitude, not ±1%. A `pytest-benchmark`
  sweep (`results/bench.json`) will add clean multi-round wall times as confirmation.

## 2. Baseline (per cell)

`wall_s` = wall time of the cell's macrosynergy work (cProfile-instrumented, so modestly
inflated but consistent across cells); `peak_rss` = peak RSS delta. "macrosynergy attribution"
= the dominant macrosynergy function(s) by cumulative time, from the cProfile dumps in
`performance/profiles/cell_NN.txt`.

| Cell | What it does | wall_s | peak_rss MiB | Dominant macrosynergy cost |
|---|---|--:|--:|---|
| 12 | freq detection | 5.6 | 1928 | `reduce_df` 5.0s (→ `drop_duplicates` 3.3s, `factorize` 1.5s) on object dtype |
| 13 | ARMA normalize+annualize | **243.6** | **12079** | `InformationStateChanges.to_qdf` → `ticker_df_to_qdf` 133s (`split_ticker` 61.5M calls ≈ 100s); `update_df` 13s |
| 14 | Q→M relabel | 64.9 | 6182 | *(none — pure pandas `.replace`+`sort_values`+`drop_duplicates` on object dtype; notebook-side)* |
| 15 | 5-day EWM sum | 37.6 | 9240 | `update_df` 15.9s (`update_tickers` 9.7s); rest pandas pivot/ewm |
| 17 | aggregation ladder | **418.6** | 9007 | **`update_df` 330s** (20 calls; `drop_duplicates` 182s + `factorize` 132s on object dtype) |
| 18 | GLB / vGLB / vBM variants | **751**¹ | 8477 | **`QuantamentalDataFrame.add_ticker_column` 432s** (`_get_tickers_series` per-row f-string + `Categorical` rebuild, via `panel_calculator`/`make_relative_value`); `update_df` 175s |
| 19 | terminal `_ZN` standardize | 139.3¹ | 8217 | `make_zn_scores` loop → `factorize` 58s + categorical `_from_sequence` 50s; `update_df` 14s |
| 21 | global timelines (fig) | 0.9 | 84 | `view_timelines` (negligible) |
| 22 | per-economy timelines (fig) | 3.4 | 57 | `view_timelines` (negligible) |
| 24 | concept correl matrix | 6.1 | 3726 | `correl_matrix` (minor) |
| 25 | economy correl matrix | 5.6 | 3825 | `correl_matrix` (minor) |
| 27 | derived targets / baskets | **589.7**¹ | 10336 | `Basket`→`reduce_df_by_ticker` ~252s (`Categorical.__init__` 166s); `add_ticker_column` 119s; `update_df` 144s |
| 31 | rates SRR heatmap | **187.6** | 7369 | **`MixedLM.fit`** (statsmodels `solver`/`solve`/`_smw_solver`/`loglike`) — ~35 panel-test fits |
| 33 | equity SRR heatmap | **179.4** | 7377 | `MixedLM.fit` (panel test) |
| 35 | FX SRR heatmap | **114.0** | 7375 | `MixedLM.fit` (panel test) |
| 37 | global SRR heatmap | 7.3 | 2719 | `SignalReturnRelations` **no panel test** → cheap (control: confirms MixedLM is the SRR cost) |
| 39 | relative SRR heatmap | 77.7 | 7370 | `MixedLM.fit` (panel test) |
| 41 | naive PnL vs IRS | 1.8 | 92 | `NaivePnL` (cheap) |
| 42 | naive PnL vs metals | 0.7 | 47 | `NaivePnL` (cheap) |
| 43 | cross-asset perf table | 9.3 | 95 | `NaivePnL` ×6 (modest) |
| 45 | significance triad | **101.1** | 7408 | `MixedLM.fit` (panel test, ×6 specs) |
| 47 | stability diagnostics | 8.0 | 94 | `NaivePnL` ×6 (modest) |
| 49 | signal concentration | 6.0 | 379 | `NaivePnL` cs=True (modest) |
| 50 | timing robustness | **89.0** | **11279** | `MixedLM.fit` (panel test ×3) + `apply_slip`; highest peak RSS |

**Totals:** ~3048s wall across the 24 cells. **Building chain (12–19, 27) ≈ 2252s (74%)** —
all object-dtype string work and categorical reconstruction. **SRR MixedLM panel test
(31/33/35/39/45/50) ≈ 749s (25%)**. Viz/PnL/correl ≈ 50s (negligible). Cell 14 (65s) is
notebook-side pure pandas (not a macrosynergy target). Peak RSS routinely 7–12 GiB — memory is
co-equal with time, and on this machine the highest-RSS cells page, inflating their wall.

¹ Cell 18 wall (751s) far exceeds its cProfile main-thread time (~107s): the +8.5 GiB peak drives
memory paging, which inflates wall beyond CPU time. This *reinforces* the memory-reduction angle —
cutting peak RSS should also cut wall on the heavy cells. The pytest-benchmark sweep will give a
cleaner wall figure.

**Headline:** the building chain (cells 12–19, 27) is dominated by **object-dtype string
work** — `update_df` / `reduce_df` / `ticker_df_to_qdf` repeatedly `factorize` / `drop_duplicates`
/ `sort_values` / `split_ticker` over multi-million-row frames whose `cid`/`xcat` columns are
plain `object` strings. The notebook's `dfx` is a plain `pd.DataFrame` (parquet/DataQuery
origin), so it satisfies the **structural** `isinstance(df, QuantamentalDataFrame)` check but
**not** `type(df) is QuantamentalDataFrame` — therefore every `reduce_df` / `update_df` call
skips its categorical fast path and runs the slow object-dtype fallback.

## 3. Ranked targets

Rank = (impact × confidence) / risk. Impact = total wall + peak-RSS reduction across the cells
that exercise it. All targets must pass the GATE (§5). The IDs (T1, T2, T2c…) are stable labels
from drafting, **not** the ranking — the ranking is the order in this table:

| # | Target | Function | Lead cells | ~Impact | Confidence | Risk |
|--:|---|---|---|---|---|---|
| 1 | **T2c** | `_get_tickers_series` vectorize | 18, 27 | ~800s+ (the 2 slowest cells' lead cost) + big RSS | high (clear per-row-loop bug) | low-med |
| 2 | **T1** | `update_df` factorized sort/dedup | 17, 18, 27, 15, 19, 13 | ~330s in 17 + broad secondary | high | med (very broad use) |
| 3 | **T2** | `split_ticker`/`ticker_df_to_qdf` vectorize | 13 | ~100s + the 12 GiB peak | high | low (Level A) |
| 4 | **T3** | `reduce_df` fast dedup | 12, 19, 43, 47, + inside others | moderate, broad | med | med (very broad use) |
| 5 | **T4** | parallelize SRR MixedLM panel test | 31,33,35,39,45,50 | ~749s (25% of total) | med | **high** (concurrency in core class) |
| 6 | **T5** | `make_zn_scores` repeated `reduce_df` | 19 | subsumed by T3 | med | low (depends on T3) |

T2c, T1, T2, T3 are one coherent theme — **the notebook's `dfx` is object-dtype, so the data
oscillates between object and categorical representations and pays string-handling / conversion
costs in both directions.** They are independent sub-branches but compose; together they target
the ~74% of runtime in the building chain. T4 is a separate, higher-risk lever for the SRR 25%.

---

### T1 — `update_df` object-dtype fallback: factorized sort + dedup
*(perf/update-df-categorical-sort)*

- **File/function:** `macrosynergy/management/utils/df_utils.py::update_df` (line 561) →
  `update_tickers` (627); the final `df.sort_values(IDX_COLS_SORT_ORDER)` (624) and the
  `drop_duplicates(subset=[real_date,xcat,cid])` (651).
- **Current cost:** cell 17 **≈330s / 80% of the cell** (20 calls; `drop_duplicates` 182s +
  `factorize` 132s); cell 15 ≈16s; cell 13 ≈13s; plus cells 18/19/27 (`update_df` loops,
  numbers pending). The single largest time sink in the notebook.
- **Why it's slow:** `dfx` is object-dtype, so `type(df) is QuantamentalDataFrame` is `False`
  → the fast path (`df.update_df(...)`, which uses `union_categoricals`) is skipped. The
  fallback runs `concat → drop_duplicates → sort_values` on three object string columns; both
  `drop_duplicates` and `sort_values` internally `factorize`/hash the object strings. The
  notebook compounds this: `dfx = update_df(dfx, piece)` in a loop re-sorts + re-dedups the
  **entire growing frame** on every call (O(k·n log n)).
- **Proposed change (API + output identical):** in the object-dtype branch, do the dedup and
  sort on **integer factor codes** of `cid`/`xcat` rather than on the object strings:
  factorize each of `cid`,`xcat` once, build an integer key, `np.lexsort`/`argsort` on
  `(cid_code, xcat_code, real_date)`, and dedup on the same codes. Restore the original object
  dtype and return the rows in the identical canonical order. No signature, return-type, or
  dtype change; byte-identical output. Constant-factor win on every call (object string
  sort/hash → int sort/hash).
- **Expected gain:** large. cell 17 ~330s → target ≪100s; cells 13/15/18/19/27 proportional;
  meaningful peak-RSS reduction (fewer intermediate object copies).
- **Risk/blast radius:** **medium-broad** — `update_df` is used pervasively across macrosynergy.
  Mitigated because the change is internal to the fallback and output is provably identical
  (same rows, same order, same dtype). Must run the full macrosynergy test suite.
- **Verified by:** cells 13, 15, 17, 18, 19, 27 (harness wall + peak RSS); golden parity on
  each of those cells' output `dfx`; `pytest` in `~/repos/macrosynergy`.
- **Stretch (separate, deeper):** route the structural-QDF object path through the existing
  categorical `union_categoricals` fast path so `cid`/`xcat` come back as `category`. Faster
  *and* speeds every downstream cell (they'd then hit fast paths too) — but it changes returned
  dtype, so it needs explicit parity treatment and is **T1b**, not part of T1.

---

### T2 — `split_ticker` / `get_cid` / `get_xcat`: vectorize the iterable branch (+ `ticker_df_to_qdf`)
*(perf/ticker-split-vectorize)*

- **File/function:** `macrosynergy/management/utils/core.py::split_ticker` (line 44; iterable
  branch line 74) via `get_cid`/`get_xcat`; consumed by
  `macrosynergy/management/utils/df_utils.py::ticker_df_to_qdf` (251), itself called from
  `InformationStateChanges.to_qdf` (`management/utils/sparse.py`).
- **Current cost:** cell 13 — `ticker_df_to_qdf` **133s cumulative (>½ the cell)**, of which
  `split_ticker` is called **61.5M times** (≈62s self) plus its string ops
  (`str.split` 12s, `str.strip` 11s, `str.lower` 6.5s, `isinstance` 9s). Also a major
  contributor to the cell's +12 GiB peak. Any other `to_qdf`-heavy path benefits too.
- **Why it's slow:** `split_ticker`'s iterable branch is a Python list comprehension that calls
  itself **once per element** (re-running `mode.lower().strip()`, isinstance checks, and a
  string split each time). `ticker_df_to_qdf` `stack()`s a wide frame, so the `"ticker"` column
  repeats each unique ticker once per date — ~thousands of distinct tickers but ~30M rows, so
  the per-row loop does ~10⁴× redundant work.
- **Proposed change (API + output identical), two composable levels:**
  - **Level A — factorize-on-uniques in `split_ticker`'s iterable branch.** Semantics: `cid` =
    text before first `_`, `xcat` = remainder (maxsplit=1). Replace the per-element recursion:
    ```python
    arr = np.asarray(ticker, dtype=object)
    if arr.size == 0:
        raise ValueError("Argument `ticker` must not be empty.")
    codes, uniq = pd.factorize(arr, sort=False)        # hash pass; ~thousands of uniques
    split = np.array([split_ticker(t, mode) for t in uniq], dtype=object)  # scalar path = validation
    return split[codes].tolist()                       # identical Python list, full length
    ```
    Returns the identical `list`; validation/error semantics preserved via the unchanged
    scalar path (a malformed ticker among the uniques still raises `ValueError`).
  - **Level B — split the column labels in `ticker_df_to_qdf` instead of the stacked column.**
    The wide frame's columns are already the unique tickers; split them once (cheap via Level A)
    and carry `cid`/`xcat` through the stack rather than building a 30M-row `"ticker"` string
    column and two derived object columns:
    ```python
    cids, xcats = get_cid(df.columns), get_xcat(df.columns)
    df.columns = pd.MultiIndex.from_arrays([cids, xcats], names=["cid", "xcat"])
    out = df.stack(["cid", "xcat"], future_stack=True).reset_index().rename(columns={0: metric})
    return standardise_dataframe(out)
    ```
    This removes the row-level split entirely and cuts the peak-RSS spike (no giant duplicated
    string column). A depends-on relationship: A makes B's label split cheap; ship them together.
- **Expected gain:** cell 13 wall 243s → target ≪120s (removes ~100s of split + reduces the
  `to_qdf` reshape), and a large cut to the +12 GiB peak. Broad secondary benefit to every
  `to_qdf` / `get_cid`/`get_xcat` caller.
- **Risk/blast radius:** **low for Level A** (pure speed, identical list, validation intact),
  **low-medium for Level B** (reshape rewrite — must match `standardise_dataframe` column
  order/dtype exactly). `get_cid`/`get_xcat`/`split_ticker` are widely used → run full suite.
- **Verified by:** cell 13 (wall + peak RSS); golden parity on cell 13 output `dfx`;
  `pytest` in `~/repos/macrosynergy`; a focused unit test on `split_ticker` scalar/iterable/
  malformed inputs.

---

### T2c — `_get_tickers_series`: vectorize categorical ticker build  ★ TOP TARGET
*(perf/qdf-ticker-series-vectorize)*

- **File/function:** `macrosynergy/management/types/qdf/methods.py::_get_tickers_series` (172).
  **This one function is the common root** of two call paths that dominate the two largest
  cells: `QuantamentalDataFrame.add_ticker_column` (`classes.py:141`, used by `panel_calculator`
  / `make_relative_value`) **and** `reduce_df_by_ticker` (`methods.py:397`, line 447 — used by
  `msp.Basket`). Fix `_get_tickers_series` once → both get fast.
- **Current cost:** cell 18 — `add_ticker_column` **432s** (`pd.Categorical.__init__` 251s
  inside); cell 27 — `reduce_df_by_ticker` **252s** (`Categorical.__init__` 166s) + another
  `add_ticker_column` 119s. These are the lead costs of the two slowest cells (751s, 590s).
- **Why it's slow:** for **categorical** `cid`/`xcat` (lines 200–210) it materializes
  full-length label arrays then builds the ticker with a **per-row Python f-string
  comprehension** `[f"{cid}_{xcat}" for cid, xcat in zip(cid_labels, xcat_labels)]` over
  millions of rows, then constructs a `pd.Categorical` from those millions of strings. The
  non-categorical branch (line 198) is the *vectorized* `df["cid"] + "_" + df["xcat"]` — so the
  categorical "fast" path is actually the slow one.
- **Proposed change (API + output identical):** build the ticker on the **observed unique
  (cid_code, xcat_code) pairs** (a few thousand), not per row: combine the two integer code
  arrays into a single key, take uniques, format `f"{cid}_{xcat}"` only for those uniques to get
  the category labels, and assemble the result `Categorical` directly from codes — or simply use
  the vectorized `cid.astype(str) + "_" + xcat.astype(str)` and re-categorize. Either yields the
  identical ordered `Categorical`; the per-row Python loop is eliminated.
- **Expected gain:** large — most of cell 18's 432s and a share of cell 27; broad benefit to
  every `panel_calculator` / `make_relative_value` / `add_ticker_column` caller.
- **Risk/blast radius:** low-medium — must reproduce the exact category set + order (the current
  code sets `ordered=True` with categories in first-appearance order via `pd.unique`). Output
  `Categorical` must compare equal. Full suite.
- **Verified by:** cells 18, 27 (harness wall + peak RSS); golden parity on those `dfx`;
  `pytest` in `~/repos/macrosynergy`.

---

### T3 — `reduce_df` object-dtype fallback
*(perf/reduce-df-fast-dedup)*

- **File/function:** `macrosynergy/management/utils/df_utils.py::reduce_df` (688); fallback
  ends with `df.drop_duplicates()` over **all** columns (793).
- **Current cost:** cell 12 ≈5.0s (`drop_duplicates` 3.3s + `factorize` 1.5s). Recurs inside
  `make_zn_scores` (cell 19), `linear_composite` (cells 17/18), `NaivePnL.__init__` (cells
  41/42/43/47/49 — 6× each in 43/47), `make_relative_value`, and directly in cells 13/15/27.
  Aggregate is large via call count (numbers being collected).
- **Why it's slow:** plain-DataFrame path skips the `type(df) is QuantamentalDataFrame` fast
  path (734) and finishes with an all-column `drop_duplicates()` on object dtype. On a clean
  panel there are no full-row duplicates, so it is pure overhead, paid on every call.
- **Proposed change (API + output identical):** dedup on the index-col subset using factor
  codes (mirrors T1), or skip the terminal `drop_duplicates()` when the index is already unique
  (cheap `duplicated(subset=IndexCols).any()` guard) — returning the identical frame faster.
  Preserve exact returned rows/order/dtype.
- **Expected gain:** medium per call, large in aggregate via the many call sites.
- **Risk/blast radius:** medium-broad (very widely used). Output-identical; full suite.
- **Verified by:** cells 12, 19, 43, 47 (harness); golden parity; `pytest`.

---

### T4 — `SignalReturnRelations` Macrosynergy Panel Test (MixedLM): parallelize fits (+ drop double `summary()`)
*(perf/srr-parallel-mixedlm)*

- **File/function:** `macrosynergy/signal/signal_return_relations.py` — the `ms_panel_test`
  path: `map_pval` (931) wraps `statsmodels.MixedLM(...).fit()`, called once per
  (signal × return) segment from `single_statistic_table`.
- **Current cost:** cell 31 ≈**187s / +7.4 GiB**; the cell's time is almost entirely inside
  `MixedLM.fit` internals (`solver` 17.8s, numpy `linalg.solve` 16.4s, `_smw_solver` 15s,
  `loglike`/`score_full`/`get_fe_params` ≈ tens of s each — pure statsmodels/numpy linear
  algebra). Same pattern in cells 33 (179s) / 35 (114s) / 39 (78s) / 45 (101s, ×6) / 50 (89s,
  ×3) — ≈749s total. **Control:** cell 37 uses `ms_panel_test=False` and costs only 7.3s,
  confirming the MixedLM panel test is essentially the entire SRR cost.
- **Why it's slow:** the panel test fits a full mixed-effects model per (sig, ret) segment —
  cell 31 alone is up to 5 signals × 7 returns = **35 independent MixedLM fits**, each an
  iterative optimizer, several re-running with fallback optimizers (lbfgs/cg) on the
  `ConvergenceWarning`s we observe. This is the irreducible numeric core; it cannot be made
  per-fit cheaper without changing results.
- **Proposed change (output identical):** the fits are **embarrassingly parallel** and
  independent. Dispatch the per-segment `map_pval` / per-(sig,ret) statistic computation across
  a worker pool (`concurrent.futures`/joblib, n_jobs param defaulting to 1 to preserve current
  behavior unless opted in — *or* internal thread/process pool guarded so the public API and
  returned table are unchanged). Same fits, same numbers, concurrent → wall ÷ ~n_workers on the
  heavy heatmap cells. **Secondary (cheap, fold in):** `map_pval` builds the statsmodels
  `summary()` **twice** (lines 967, 972) to parse one p-value — call it once / read `re.pvalues`
  reproducing the 3-dp rounding; small but free.
- **Expected gain:** large wall reduction on 31/33/35/39/45/50 (parallel); peak RSS roughly
  unchanged (or modestly up with processes — measure).
- **Risk/blast radius:** **medium-high** — concurrency in a core analysis class; must guarantee
  identical table output and deterministic ordering, handle the convergence-warning paths, and
  keep a serial default. The `summary()` dedup sub-change is low-risk on its own.
- **Verified by:** cells 31/33/35/39/45/50 (harness wall + peak RSS); golden parity on each
  table (numeric tolerance); `pytest` in `~/repos/macrosynergy`.
- **Note:** if concurrency is deemed too invasive, a contained fallback is to cap/skip the
  multi-optimizer retry cascade — but that risks changing p-values, so it is **not** preferred.

---

### T5 — `make_zn_scores` repeated `reduce_df` in the cell-19 loop
*(perf/zn-scores-reduce)*

- **File/function:** `macrosynergy/panel/make_zn_scores.py::make_zn_scores` (calls `reduce_df`
  internally).
- **Current cost:** cell 19 (139s) calls `make_zn_scores` ~20+ times (per xcat × scope), each
  `reduce_df`-ing the full object-dtype frame; profile shows `factorize` 58s + categorical
  `_from_sequence` 50s. Largely subsumed by T3 (faster `reduce_df`); kept as a distinct item in
  case a per-call caching/scoping win exists beyond T3.
- **Proposed change:** primarily benefits from T3; investigate whether the repeated full-frame
  `reduce_df` per category can be hoisted/scoped without changing output.
- **Risk:** low (depends on T3).
- **Verified by:** cell 19 (harness); golden parity; `pytest`.

---

## 4. Output-parity baseline capture (GATE criterion 2)

A capture script is ready at `academy/drafts/surprises/performance/capture_golden.py`. It runs
each benchmarked cell once on its frozen fixture (identical invocation to `profile_cells.py`)
and serializes the output to `performance/golden/`, plus `golden/index.json` (output kind +
content hash):

- **building cells** (12–19, 27) → returned `dfx` to parquet (canonical row order) + `extras`
  JSON. Parity = QuantamentalDataFrame value-equality + extras deep-equality.
- **DataFrame** outputs (31/33/35/37/39/41/42/43/45/47/49) → parquet. Parity = same
  shape/index/columns + `np.allclose` on numerics (NaN-equal), exact on object cells.
- **Series** (50) → parquet (one-col). Same rule.
- **None** (21/22/24/25 — `view_timelines` and `correl_matrix` plot in place and return `None`,
  confirmed by the capture smoke-test) → recorded `kind="none"`; no return-value golden. **Their
  visual output is transitively guaranteed by building-cell `dfx` parity** — these are
  deterministic renders of `dfx`, which *is* captured — so no perf target can change them without
  first breaking a building-cell golden. (The capture script retains a Figure handler — Axes line
  `(x,y)`/image arrays to `.npz` — for any cell that does return a `Figure`.)

**Procedure:** on `feature/performance` (clean, before any perf change) run
`python capture_golden.py` once → commit `golden/index.json` (hashes; the parquet/npz blobs are
gitignored like the other fixtures). Each `perf/<slug>` sub-branch re-captures and asserts
parity against this baseline (a small `assert_golden.py` diff using the rules above) as part of
its GATE. Capture runs **after** the profiling sweep frees memory (not concurrently).

## 5. The GATE (every `perf/<slug>` sub-branch must pass before merging to `feature/performance`)

1. **Public API unchanged** — same names, signatures, parameter semantics, return types.
2. **Output parity** — each affected cell's output matches the §4 golden (DataFrame.equals /
   numeric tolerance; data-level equality for figures).
3. **Measurable win** — wall and/or peak RSS reduced on the affected cells per the harness,
   with no regression on others. Record before/after in `QUEUE.md`.
4. **macrosynergy suite passes** — `pytest` in `~/repos/macrosynergy`.

## 6. Sub-branch plan

| Rank | Target | Sub-branch | Depends on |
|--:|---|---|---|
| 1 | T2c | `perf/qdf-ticker-series-vectorize` | — |
| 2 | T1 | `perf/update-df-categorical-sort` | — |
| 3 | T2 (A+B) | `perf/ticker-split-vectorize` | — |
| 4 | T3 | `perf/reduce-df-fast-dedup` | — |
| 5 | T4 | `perf/srr-parallel-mixedlm` (incl. `summary()` dedup) | — |
| 6 | T5 | `perf/zn-scores-reduce` | T3 |
| 7 | T6 | `perf/basket-categorical-loc` (Basket `.loc[fvi:]`) | — (enables QDF-native notebook; see §7.2) |
| — | T1b (stretch) | `perf/qdf-categorical-propagation` | **demoted — see §7.1** |

## 7. Notebook-side (academy) complementary optimization — flagged, NOT part of Scope-2

Scope-2 is **macrosynergy-side only**: T1–T5 keep the public API and outputs identical and speed
the package up *regardless of the caller's dtype*, so every macrosynergy user benefits and the
notebook needs no edits. That is the right primary lever and where the queue is focused.

But the diagnosis points at one **notebook-side** change worth a controlled experiment:

- **Root fact:** the notebook calls `JPMaQSDownload.download(...)` **without**
  `categorical_dataframe`, which defaults to `False` → `dfx` is **object-dtype**. Confirmed
  against the live notebook; the object-dtype baseline above is therefore faithful (not a fixture
  artifact).
- **The lever:** pass `categorical_dataframe=True` to `download(...)` (or convert once with
  `dfx = QuantamentalDataFrame(dfx, categorical=True)` right after download). Then `cid`/`xcat`
  are `category`, and `update_df`/`reduce_df`/`update_tickers` hit their existing
  `union_categoricals` fast paths (the `type(df) is QuantamentalDataFrame` branch) instead of the
  slow object fallback. It would also speed cell 14 (the pure-pandas `.replace`/`sort_values`,
  65s) — a cell no macrosynergy change can touch.
- **Why it is NOT a substitute for T1–T5, and is coupled to them:** the categorical path is not
  uniformly faster today. T2c's `_get_tickers_series` is *slower* on categorical input — the
  cell-18 (432s) and cell-27 (252s) costs were measured **on the categorical path** inside
  `panel_calculator`/`make_relative_value`/`Basket`. So flipping `dfx` to categorical **before
  T2c lands could regress cells 18/27** even as it speeds 13/15/17/19. After **T2c + T1**, a
  categorical `dfx` becomes a clean, large win across the whole building chain.
- **`add_ticker_column` etc. are internal helpers** — the notebook should *not* call them
  directly; they are invoked under the hood by the panel functions. The only notebook-facing knob
  is the download dtype / a single `QuantamentalDataFrame(..., categorical=True)` conversion.
- **Caveats for the experiment:** changing `dfx` dtype can alter returned-table dtypes and
  `groupby(observed=...)` semantics downstream → must pass the same golden parity (§4) and the
  notebook's own run. It is an **academy** change on its own review track, sequenced **after**
  T2c/T1, and measured by this same harness.

### 7.1 Measured result — categorical `dfx` experiment

Ran the full sweep a second time with each cell's input wrapped as
`QuantamentalDataFrame(df, categorical=True)` (simulating `download(categorical_dataframe=True)`;
script `cat_profile.py`, same wall+RSS+cProfile method, per-cell errors caught). Object vs
categorical:

| Cell | Object wall / RSS | Categorical wall / RSS | Verdict |
|---|---|---|---|
| 12 | 5.6s / 1928 | **ERROR** `Categorical + str` | notebook `cid + "_"` |
| 13 | 244s / 12079 | **ERROR** `Categorical + str` | notebook `cid + "_"` / `xcat += "A"` |
| 14 | 64.9s / 6182 | **6.5s / 4234** | **10× faster** (pure-pandas `.replace`/sort/dedup) |
| 15 | 37.6s / 9240 | **ERROR** `Categorical + str` | notebook ticker `cid + "_"` |
| 17 | 418.6s / 9007 | 423.2s / 8984 | **no change** (the 419s cell — the big one) |
| 18 | 751s / 8477 | 484.6s / 9226 | −35% wall (paging-noisy), RSS flat |
| 19 | 139.3s / 8217 | 112.6s / 8052 | −19% |
| 27 | 589.7s / 10336 | **ERROR** `InvalidIndexError` | categorical indexing in basket path |
| 31/33/35/39/45/50 | 749s total / ~7.4 GiB | 766s total / ~7.6 GiB | **~same / marginally slower** |
| viz/PnL (21,22,24,25,37,41–49) | ~50s | ~40s | trivial either way |

**What this proves about prioritization:**

1. **Categorical is not a drop-in: it breaks 4 of 8 building cells** (12, 13, 15, 27 — half the
   pipeline, including the 244s and 590s cells). The notebook builds tickers/xcats with string
   ops (`cid + "_" + xcat`, `xcat += "A"`) that raise on categorical dtype. So
   `categorical_dataframe=True` would require **rewriting the notebook's core construction code**,
   not a one-line flag.
2. **It does not help the dominant cost.** Cell 17 (419s, the largest building cell) is
   **unchanged** — because `update_df`'s `df_add` (the `linear_composite` output) is object, so
   `concat(categorical, object)` collapses back to object and the bottleneck `sort_values`/
   `drop_duplicates` is paid anyway. The categorical input "washes out" after the first
   `update_df`. **The SRR cells (25% of runtime) are unaffected** (MixedLM is dtype-insensitive;
   categorical is marginally *slower*).
3. **The only large clean win is cell 14 — which is pure pandas, not a macrosynergy target.**

**Conclusion (this is the better data you asked for):**

- **The macrosynergy-side targets (T2c → T1 → T2 → T3) are confirmed as the priority.** They
  speed the package up regardless of caller dtype, need **zero notebook changes**, and break
  nothing. The categorical route helps neither the biggest building cell (17) nor the SRR 25%.
- **`categorical_dataframe=True` is demoted from "follow-up experiment" to "blocked / not
  worthwhile as-is"** — it requires notebook rewrites just to *run*, and even then doesn't touch
  the dominant costs. Revisit only if the notebook is rewritten to be categorical-native AND the
  panel functions (`linear_composite`/`panel_calculator`/`make_zn_scores`) are made to *return*
  categorical (so the dtype doesn't wash out) — a far larger effort than T1–T3.
- **The stretch target T1b (categorical propagation) is also demoted** by the same cell-17
  evidence: propagating categorical only pays off if every panel function preserves it end to
  end; today they don't, so T1 (fast object-dtype path) is strictly the better investment.
- **Bonus insight:** the breakages are exactly the manual `cid + "_" + xcat` constructions that
  `_get_tickers_series`/`add_ticker_column` exist to do — so once **T2c** makes those fast, the
  notebook could *both* drop its hand-rolled concatenations *and* be categorical-safe. A neat
  post-T2c notebook cleanup, but not on the macrosynergy critical path.

(The categorical harness mode lives in the scratch `cat_profile.py`.)

### 7.2 Measured result — notebook fixed to be QDF-native, end-to-end re-run

Followed the flip: instead of changing macrosynergy, **fixed the benchmark cells** so the pipeline
runs on a categorical `QuantamentalDataFrame` (`download(categorical_dataframe=True)`), then re-ran
the full sweep. **Scope note:** these fixes are in our benchmark-harness cell versions
(`academy/drafts/surprises/performance/cells.py`) only — the source `.ipynb` is being revised
independently by a colleague (separate version, arriving next day) and the two will be reconciled
then; the categorical edits below + T6 should carry into the combined notebook. Fixes:

- **Cells 12, 13, 15** — the `cid + "_" + xcat` / `xcat += "A"` string ops were rewritten
  `.astype(str)`-first, and the cell-12 `groupby` got `observed=True` (so categorical grouping
  keeps only observed combinations = the object result). Dtype-agnostic, parity-preserving.
- **Cell 27** — *could not* be fixed notebook-side: `msp.Basket.make_weights` (`basket.py:502`)
  slices its wide frame with `dfw_wgs[fvi:]` (a Timestamp slice), which raises `InvalidIndexError`
  on a `CategoricalIndex`. Worked around by feeding `Basket` a string-typed view
  (`QuantamentalDataFrame(dfx, categorical=False)`) — which **copies the frame** and is the cause
  of cell 27's higher peak RSS below. The real fix is macrosynergy-side → **new target T6**.

**Value parity:** object vs categorical outputs are **identical** for cells 12/13/15/27 (dfx
values + extras), verified on a representative cid subset (`parity_check.py`).

| Cell | Object wall / RSS(MiB) | QDF-native wall / RSS(MiB) | Δ |
|---|---|---|---|
| 12 | 5.6 / 1928 | 4.3 / 1699 | faster, lighter |
| 13 | 243.6 / 12079 | 246.6 / **12730** | unchanged, +RSS |
| 14 | 64.9 / 6182 | **6.3** / 4221 | **−90% wall** |
| 15 | 37.6 / 9240 | 39.6 / 9106 | unchanged |
| 17 | 418.6 / 9007 | 429.3 / 9003 | unchanged |
| 18 | 751.2 / 8477 | 452.4 / **9190** | −40% wall (paging-noisy), +RSS |
| 19 | 139.3 / 8217 | 63.1 / 8146 | **−55% wall** |
| 27 | 589.7 / 10336 | 575.0 / **12915** | ~same wall, **+25% RSS** |
| 31/33/35/39/45/50 (SRR) | 749 total | 723 total | ~same |
| **building chain (12–19,27)** | **2250s** | **1817s** | **−19% wall** |
| **whole sweep** | **~3048s** | **~2565s** | **−16% wall** |
| **peak RSS (max cell)** | **12079 (c13)** | **12915 (c27)** | **+7% (worse)** |

**What this proves (the data you asked for):**

1. **A QDF-native notebook is feasible and parity-safe** — three cells need trivial
   `.astype(str)` / `observed=True` edits. **But cell 27 can't be done notebook-side** — `Basket`
   rejects categorical input (a real macrosynergy bug, T6).
2. **The wall win is real but narrow (~16%)** and comes almost entirely from **cell 14**
   (pure-pandas relabel, 65s→6.3s) and **cell 19** (139s→63s). The three dominant cells —
   **13 (`split_ticker`/T2), 17 (`update_df`/T1), 27 (`Basket`/`reduce_df_by_ticker`/T2c)** — are
   **unchanged**. So categorical does *not* substitute for T1/T2/T2c.
3. **Memory got worse, not better.** Peak RSS rose 12.1→12.9 GiB. The "QDF is a smaller object"
   intuition holds **at rest** (the stored `dfx`'s `cid`/`xcat` become int codes), but **peak
   working set** — the metric that drives OOM and paging — is set by the transient **object-dtype
   intermediates** the panel functions still materialize, plus the categorical metadata overhead
   and (cell 27) the Basket object-copy. Notebook-side categorical alone cannot lower the peak;
   that needs the macrosynergy-side fixes (T1/T2c/T2/T3 reduce the intermediates themselves).

**Revised recommendation:**

- **Adopt the QDF-native notebook edits** (cells 12/13/15 + the cell-27 workaround, and
  `download(categorical_dataframe=True)`): they're cheap, parity-safe, and buy ~16% wall — *and*
  a categorical `dfx` is the better default representation. Worth doing **for the wall win on
  14/19 and as the canonical dtype**, not for memory.
- **It does not change the macrosynergy priority.** T2c → T1 → T2 → T3 still own the dominant
  costs (cells 13/17/27) and are the only way to actually cut peak RSS. Ship them regardless.
- **The two compose best together:** with a categorical `dfx` *and* T2c (which makes the
  categorical `_get_tickers_series` fast) *and* T1 (fast `update_df` so categorical doesn't wash
  out), the building chain would improve far more than either alone.

### T6 — `Basket` categorical-input bug (enables the QDF-native notebook)
*(perf/basket-categorical-loc)*

- **File/function:** `macrosynergy/panel/basket.py::make_weights` (line 502) — `dfw_wgs[fvi:]`
  where `fvi` is a `Timestamp` and `dfw_wgs` has a `CategoricalIndex` of tickers/cids → pandas
  evaluates `fvi in columns` → `CategoricalIndex.__contains__` → **`InvalidIndexError`**.
- **Repro:** any `msp.Basket(df, …).make_basket(...)` where `df` is a categorical
  `QuantamentalDataFrame` (cell 27). Object-dtype input works, so it's latent.
- **Fix (output-identical):** use label-based slicing — `dfw_wgs = dfw_wgs.loc[fvi:]` — which does
  not probe the columns index. Audit `Basket` for other `df[ts:]`/`in columns` patterns.
- **Risk:** low; `.loc[fvi:]` is the correct row-label slice. Run the macrosynergy suite.
- **Why it matters:** without it the notebook can't pass a categorical frame to `Basket` and must
  copy back to object (the +25% RSS on cell 27). Prerequisite for a clean categorical-native
  notebook; small and self-contained.
