# macrosynergy performance — work queue (`feature/performance`)

Queue-driven book of work. Each item is a `perf/<slug>` sub-branch off `feature/performance`.
An item may only merge back after passing the **GATE** (see `TARGETS.md` §5). Full diagnosis,
designs, and the baseline that seeds this queue are in `TARGETS.md`.

**Workflow per item:** branch `perf/<slug>` → implement the design → capture golden parity
(`academy/drafts/surprises/performance/capture_golden.py` baseline vs re-capture) → run harness
on the item's cells (record before/after) → `pytest` in `~/repos/macrosynergy` → fill the
before/after table below → merge to `feature/performance`. Do **not** merge to develop/main.

Status legend: `TODO` · `IN PROGRESS` · `IN REVIEW` · `DONE` · `BLOCKED`

---

## Q1 — T2c · `perf/qdf-ticker-series-vectorize`  — **TODO**
Vectorize `_get_tickers_series` (qdf/methods.py:172): replace the per-row f-string comprehension
+ `pd.Categorical` rebuild with a uniques-based / vectorized build. Fixes both
`add_ticker_column` and `reduce_df_by_ticker` (→ `panel_calculator`, `make_relative_value`,
`Basket`).
- **Cells:** 18, 27 (also any `add_ticker_column`/`Basket` path).
- **Baseline:** cell 18 = 751s / 8477 MiB; cell 27 = 590s / 10336 MiB.
- **After:** _tbd_ · **Parity:** _tbd_ · **macrosynergy pytest:** _tbd_

## Q2 — T1 · `perf/update-df-categorical-sort`  — **TODO**
`update_df`/`update_tickers` (df_utils.py:561/627) object-dtype branch: dedup + sort on
factorized integer codes instead of object strings; restore dtype + order (byte-identical).
**Fix BOTH implementations (TARGETS §7.3):** the object path `df_utils.py::update_df/update_tickers`
AND the categorical/QDF-native twins `qdf/methods.py::update_df`(458)/`update_tickers`(493) +
`qdf/classes.py::update_df`(291) — the categorical path is still ~330s in cell 17 (re-sorts the
growing frame; `union_categoricals` skipped when `df_add` is object → washes to object).
- **Cells:** 17, 18, 27, 15, 19, 13.
- **Baseline:** cell 17 = 419s / 9007 MiB (update_df ≈330s); cell 15 = 38s; cell 13 = 244s.
- **After:** _tbd_ · **Parity:** _tbd_ · **macrosynergy pytest:** _tbd_

## Q3 — T2 · `perf/ticker-split-vectorize`  — **TODO**
`split_ticker` iterable branch (core.py:44) factorize-on-uniques (Level A) +
`ticker_df_to_qdf` (df_utils.py:251) split column labels pre-stack (Level B). Output-identical.
- **Cells:** 13 (`InformationStateChanges.to_qdf`).
- **Baseline:** cell 13 = 244s / 12079 MiB (`ticker_df_to_qdf` ~133s; `split_ticker` 61.5M calls).
- **After:** _tbd_ · **Parity:** _tbd_ · **macrosynergy pytest:** _tbd_

## Q4 — T3 · `perf/reduce-df-fast-dedup`  — **TODO**
`reduce_df` (df_utils.py:688) object-dtype fallback: skip/fast the terminal all-column
`drop_duplicates()` (factor-code dedup, or unique-index guard). Output-identical.
**Fix BOTH implementations (TARGETS §7.3):** object path `df_utils.py::reduce_df`(688) AND the
categorical/QDF-native twin `qdf/methods.py::reduce_df`(309).
- **Cells:** 12, 19, 43, 47 (+ inside `linear_composite`/`make_zn_scores`/`NaivePnL`).
- **Baseline:** cell 12 = 5.6s / 1928 MiB (`drop_duplicates` 3.3s).
- **After:** _tbd_ · **Parity:** _tbd_ · **macrosynergy pytest:** _tbd_

## Q5 — T4 · `perf/srr-parallel-mixedlm`  — **TODO** (higher risk)
Parallelize the independent MixedLM panel-test fits in `SignalReturnRelations`
(signal_return_relations.py); fold in the `map_pval` double-`summary()` dedup (lines 967/972).
Serial default; identical table output. **Slice first, fan out the minimum** — parallelize at the
per-`(sig,ret)` boundary (`map_pval` already takes pre-extracted `ret_vals`/`sig_vals`); dispatch
the small `(y,X,groups)` arrays, not the shared `dfx`. Then **either** threads (`MixedLM.fit` is
BLAS, releases the GIL → no extra memory) **or** processes (tiny payloads) work. See TARGETS §3.1.
- **Cells:** 31, 33, 35, 39, 45, 50.
- **Baseline:** 31=188s, 33=179s, 35=114s, 39=78s, 45=101s, 50=89s (≈749s, all ~7.4 GiB; 50=11.3 GiB).
- **After:** _tbd_ · **Parity:** _tbd_ · **macrosynergy pytest:** _tbd_

## Q6 — T5 · `perf/zn-scores-reduce`  — **TODO** (depends on Q4)
Investigate hoisting/scoping the repeated full-frame `reduce_df` inside the cell-19
`make_zn_scores` loop beyond what Q4 already buys.
- **Cells:** 19. **Baseline:** 139s / 8217 MiB. **After:** _tbd_

## Q7 — T6 · `perf/basket-categorical-loc`  — **TODO** (small, enables QDF-native notebook)
`Basket.make_weights` (basket.py:502) `dfw_wgs[fvi:]` raises `InvalidIndexError` on categorical
input (CategoricalIndex columns). Fix to label slice `dfw_wgs.loc[fvi:]`; audit Basket for similar
`df[ts:]` patterns. Output-identical. Prerequisite for passing a categorical `dfx` to `Basket`
(cell 27) without an object-copy (which today costs +25% peak RSS on that cell).
- **Cells:** 27. **Baseline:** object 590s / 10336 MiB; categorical+object-copy 575s / 12915 MiB.
- **Test gap (must fix as part of T6):** `Basket`'s unit tests only cover object-dtype input, so
  this `InvalidIndexError` on categorical was never caught — it should have been flagged at PR
  time. Add a regression test running `Basket` on a categorical `QuantamentalDataFrame`, and add a
  categorical-input case to the shared panel-function test matrix (audit `linear_composite` /
  `make_relative_value` / `panel_calculator` / `make_zn_scores` / `SignalReturnRelations` too).
- **After:** _tbd_ · **Parity:** _tbd_ · **macrosynergy pytest:** _tbd_

## Q8 — T7 · `perf/zn-scores-parallel`  — **OPTIONAL / low priority** (do after Q4; gated)
Add opt-in `n_jobs` to fan the independent per-xcat scorings in `make_zn_scores` (serial default).
**Likely deferred:** cell-19 cost is `reduce_df` (Q4), not the z-score math (expanding estimation
already `O(n)`); inner per-series estimation is sequential. With **per-xcat slicing** (fan out one
xcat's panel per worker, via an iterator producer) the process-pool memory objection goes away, but
the residual upside is small. Reassess only if meaningful cost remains after Q4. See TARGETS §3.1.
- **Cells:** 19. **Baseline:** 139s / 8217 MiB. **After:** _tbd_

> **Notebook-side companion (academy, separate repo/branch):** the cyclical-strength notebook can
> be made QDF-native (`download(categorical_dataframe=True)` + `.astype(str)`/`observed=True` edits
> to cells 12/13/15 + the cell-27 Basket workaround until T6 lands). Parity-verified, ~16% wall
> (cells 14 & 19); does **not** reduce peak RSS. Adopt as canonical dtype, but **not** a substitute
> for Q1–Q5. See TARGETS.md §7.2.

---

## Stretch (not scheduled)
- **T1b** `perf/qdf-categorical-propagation` (depends on Q2): route the structural-QDF object
  path through the categorical fast path so `cid`/`xcat` return as `category` and speed all
  downstream cells. Changes return dtype → explicit parity treatment required.
