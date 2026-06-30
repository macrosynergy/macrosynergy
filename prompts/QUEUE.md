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
- **Cells:** 12, 19, 43, 47 (+ inside `linear_composite`/`make_zn_scores`/`NaivePnL`).
- **Baseline:** cell 12 = 5.6s / 1928 MiB (`drop_duplicates` 3.3s).
- **After:** _tbd_ · **Parity:** _tbd_ · **macrosynergy pytest:** _tbd_

## Q5 — T4 · `perf/srr-parallel-mixedlm`  — **TODO** (higher risk)
Parallelize the independent MixedLM panel-test fits in `SignalReturnRelations`
(signal_return_relations.py); fold in the `map_pval` double-`summary()` dedup (lines 967/972).
Serial default; identical table output.
- **Cells:** 31, 33, 35, 39, 45, 50.
- **Baseline:** 31=188s, 33=179s, 35=114s, 39=78s, 45=101s, 50=89s (≈749s, all ~7.4 GiB; 50=11.3 GiB).
- **After:** _tbd_ · **Parity:** _tbd_ · **macrosynergy pytest:** _tbd_

## Q6 — T5 · `perf/zn-scores-reduce`  — **TODO** (depends on Q4)
Investigate hoisting/scoping the repeated full-frame `reduce_df` inside the cell-19
`make_zn_scores` loop beyond what Q4 already buys.
- **Cells:** 19. **Baseline:** 139s / 8217 MiB. **After:** _tbd_

---

## Stretch (not scheduled)
- **T1b** `perf/qdf-categorical-propagation` (depends on Q2): route the structural-QDF object
  path through the categorical fast path so `cid`/`xcat` return as `category` and speed all
  downstream cells. Changes return dtype → explicit parity treatment required.
