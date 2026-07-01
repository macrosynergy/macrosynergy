# macrosynergy performance — work queue (`feature/performance`)

Queue-driven book of work. Each item is a `perf/<slug>` sub-branch off `feature/performance`.
An item may only merge back after passing the **GATE** (see `TARGETS.md` §5). Full diagnosis,
designs, and the baseline that seeds this queue are in `TARGETS.md`.

**Workflow per item:** branch `perf/<slug>` → implement the design (fix BOTH object + categorical
implementations where §7.3 applies) → **in-repo GATE** (`tests/perf/`, see TARGETS §5.1): the
default `pytest` gate incl. `tests/perf/test_parity_*` + the appended API/edge tests stays green
(parity preserved), and the `-m perf` benchmark before/after (recorded via `tests/perf/record.py`)
shows a win → `pytest` in `~/repos/macrosynergy` → **macro confirmation** (optional/when available):
the external 24-cell harness (`academy/drafts/surprises/performance/capture_golden.py` baseline vs
re-capture + the cell wall/RSS sweep) → fill the before/after table below → merge to
`feature/performance`. Do **not** merge to develop/main.

> **Driving the queue (manager/creator/reviewer team — mirrors academy `run-research-pipeline` and
> knowledge `run-parse-pipeline`):** items are driven one at a time (Q1 first). A manager owns the
> branch/worktree/queue bookkeeping; a builder agent implements one item end-to-end in an isolated
> worktree against its brief + the in-repo GATE; a read-only adversarial reviewer returns
> APPROVE/CHANGES before merge. The `tests/perf/` framework is the builder's self-verification and
> the reviewer's re-run target.

Status legend: `TODO` · `IN PROGRESS` · `IN REVIEW` · `DONE` · `BLOCKED`

---

## Q1 — T2c · `perf/qdf-ticker-series-vectorize`  — **DONE** (PR #4, squash `d8e376f4`)
Vectorize `_get_tickers_series` (qdf/methods.py:172): replace the per-row f-string comprehension
+ `pd.Categorical` rebuild with a uniques-based / vectorized build. Fixes both
`add_ticker_column` and `reduce_df_by_ticker` (→ `panel_calculator`, `make_relative_value`,
`Basket`).
- **Cells:** 18, 27 (also any `add_ticker_column`/`Basket` path).
- **Baseline:** cell 18 = 751s / 8477 MiB; cell 27 = 590s / 10336 MiB.
- **After:** micro-benchmark (record.py, `-m perf -k small`): `get_tickers_series[cat-small]`
  31.9 ms → 1.3 ms (~24×, −96%); `add_ticker_column[small]` 33.8 ms → 3.4 ms (~10×, −90%); obj
  branch unchanged (within machine noise). Macro cell re-capture pending external harness.
  · **Parity:** byte-identical — `test_parity_qdf_ticker_series` 3 passed, `tests/perf/golden/`
  unchanged; output `Categorical` equal (categories, first-appearance order, `ordered`, codes).
  · **macrosynergy pytest:** `tests/unit/management` + `tests/unit/panel` 479 passed. Reviewer:
  APPROVE, 0 blockers, 1 round.

## Q2 — T1 · `perf/update-df-categorical-sort`  — **DONE** (PR #5, squash `7d27a24c`)
`update_df`/`update_tickers` (df_utils.py:561/627) object-dtype branch: dedup + sort on
factorized integer codes instead of object strings; restore dtype + order (byte-identical).
**Fix BOTH implementations (TARGETS §7.3):** the object path `df_utils.py::update_df/update_tickers`
AND the categorical/QDF-native twins `qdf/methods.py::update_df`(458)/`update_tickers`(493) +
`qdf/classes.py::update_df`(291) — the categorical path is still ~330s in cell 17 (re-sorts the
growing frame; `union_categoricals` skipped when `df_add` is object → washes to object).
- **Cells:** 17, 18, 27, 15, 19, 13.
- **Baseline:** cell 17 = 419s / 9007 MiB (update_df ≈330s); cell 15 = 38s; cell 13 = 244s.
- **After:** micro-benchmark (record.py, `-m perf -k small`): `update_df_growing_loop[obj-small]`
  106.7 ms → 69 ms (~−35%); `[cat-small]` 53.1 ms → ~44–48 ms (~−10–18%, noisy at small tier);
  `update_tickers[small]` neutral (within noise — now routes through the shared factorize/lexsort
  helper and additionally sorts). Single `factorize(sort=True)+np.lexsort`+vectorized adjacent-dup
  pass replaces object-string `drop_duplicates(keep=last)+sort_values`; categorical twin
  re-categorizes object `df_add` to keep the frame categorical instead of upcasting. NaT `real_date`
  sorts last (na_position='last' parity). `classes.py::update_df` delegates to `methods.py` (no
  change needed). Macro cell re-capture pending external harness.
  · **Parity:** byte-identical — `test_parity_update_df` 3 passed (last-wins + IDX_COLS_SORT_ORDER +
  no-mutate), `UpdateDfEdge` 7 passed, `tests/perf/golden/` unchanged.
  · **macrosynergy pytest:** `tests/unit/management` 269 passed. Reviewer: APPROVE, 0 blockers,
  2 rounds (round 1: empty-base sort regression + unvectorized public `update_tickers` → fixed).

## Q3 — T2 · `perf/ticker-split-vectorize`  — **DONE** (PR #6, squash `cfa0991c`)
`split_ticker` iterable branch (core.py:44) factorize-on-uniques (Level A) +
`ticker_df_to_qdf` (df_utils.py:251) split column labels pre-stack (Level B). Output-identical.
- **Cells:** 13 (`InformationStateChanges.to_qdf`).
- **Baseline:** cell 13 = 244s / 12079 MiB (`ticker_df_to_qdf` ~133s; `split_ticker` 61.5M calls).
- **After:** micro-benchmark (record.py, `-m perf`): `get_cid_large_list[2000-50]` **−77%**,
  `get_xcat_large_list[2000-50]` **−74%**, `ticker_df_to_qdf[500-1300]` **−64%** (Level A
  factorize-on-uniques + Level B label-split, which eliminates the 30M-row `"ticker"` column).
  Cell-13 wall/peak-RSS macro delta pending external harness. · **Parity:** byte-identical —
  `test_parity_ticker_split` 2 passed, `SplitTickerDirect` 9 passed, `tests/perf/golden/` unchanged.
  · **macrosynergy pytest:** integrated tree 480 passed (management+panel). Reviewer: APPROVE,
  0 blockers, 1 round.

## Q4 — T3 · `perf/reduce-df-fast-dedup`  — **DONE** (PR #7, squash `dd7ff3c1`)
`reduce_df` (df_utils.py:688) object-dtype fallback: skip/fast the terminal all-column
`drop_duplicates()` (factor-code dedup, or unique-index guard). Output-identical.
**Fix BOTH implementations (TARGETS §7.3):** object path `df_utils.py::reduce_df`(688) AND the
categorical/QDF-native twin `qdf/methods.py::reduce_df`(309).
- **Cells:** 12, 19, 43, 47 (+ inside `linear_composite`/`make_zn_scores`/`NaivePnL`).
- **Baseline:** cell 12 = 5.6s / 1928 MiB (`drop_duplicates` 3.3s).
- **After:** micro-benchmark (record.py, `-m perf -k small`): `reduce_df_full[obj-small]` **−39%**,
  `[cat-small]` **−42%**, `reduce_df_filtered[small]` **−23%** (both twins: unique-index guard on
  `(cid,xcat,real_date)` skips the terminal all-column `drop_duplicates()` on clean panels; runs it
  unchanged when the key is non-unique). Cell-12 macro delta pending external harness. · **Parity:**
  byte-identical — `test_parity_reduce_df` 2 passed, `ReduceDfEdgeAPI` 5 passed, `tests/perf/golden/`
  unchanged. · **macrosynergy pytest:** integrated tree 480 passed. Reviewer: APPROVE, 0 blockers,
  1 round (nit: no test for key-same/value-differs, but True-branch calls all-column dedup — safe).

## Q5 — T4 · `perf/srr-parallel-mixedlm`  — **IN REVIEW** (higher risk; APPROVED parity, NO win on this machine — PR #8 HELD OPEN for human decision per instruction; see PR body for 4 findings)
Parallelize the independent MixedLM panel-test fits in `SignalReturnRelations`
(signal_return_relations.py); fold in the `map_pval` double-`summary()` dedup (lines 967/972).
Serial default; identical table output. **Slice first, fan out the minimum** — parallelize at the
per-`(sig,ret)` boundary (`map_pval` already takes pre-extracted `ret_vals`/`sig_vals`); dispatch
the small `(y,X,groups)` arrays, not the shared `dfx`. Then **either** threads (`MixedLM.fit` is
BLAS, releases the GIL → no extra memory) **or** processes (tiny payloads) work. See TARGETS §3.1.
- **Cells:** 31, 33, 35, 39, 45, 50.
- **Baseline:** 31=188s, 33=179s, 35=114s, 39=78s, 45=101s, 50=89s (≈749s, all ~7.4 GiB; 50=11.3 GiB).
- **After:** _tbd_ · **Parity:** _tbd_ · **macrosynergy pytest:** _tbd_

## Q6 — T5 · `perf/zn-scores-reduce`  — **DONE (investigation: no change — subsumed by Q4)**
Investigate hoisting/scoping the repeated full-frame `reduce_df` inside the cell-19
`make_zn_scores` loop beyond what Q4 already buys.
- **Cells:** 19. **Baseline:** 139s / 8217 MiB.
- **After:** DEFERRED, no code change. Measured post-Q4 (20-cid × 6-xcat panel, 24 single-xcat
  calls): `reduce_df` is only **7.4%** of `make_zn_scores` (10.0 ms/call); the call is inherently
  per-xcat (each selects a different xcat) so there is no hoist/scope win — confirmed subsumed by
  Q4 (T3). No PR. `test_zn_scores.py` 17 passed, `tests/unit/panel` 210 passed, `reduce_df` parity
  2 passed; `tests/perf/golden/` unchanged. Measured on the Q4 worktree (Q4 not yet merged).
  · **Findings spun out → see Stretch (T5a/T5b).**

## Q7 — T6 · `perf/basket-categorical-loc`  — **DONE** (PR #9, squash `0778ea88`)
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
- **After:** correctness enable (no `-m perf` benchmark for `Basket`). Fix = `dfw_wgs[fvi:]` →
  `dfw_wgs.loc[fvi:]` (label row slice); categorical `QuantamentalDataFrame` input no longer raises
  `InvalidIndexError`. Removes cell-27's **+25% peak-RSS object-copy** (macro-confirmed externally).
  · **Parity:** object-dtype output byte-identical; new `test_categorical_qdf_parity` fails-before /
  passes-after (reviewer reproduced directly); sibling patterns audited safe; `tests/perf/golden/`
  unchanged. · **macrosynergy pytest:** integrated tree 480 passed. Reviewer: APPROVE, 0 blockers,
  1 round.

## Q8 — T7 · `perf/zn-scores-parallel`  — **DONE (investigation: no change — deferred, not worthwhile after Q4)**
Add opt-in `n_jobs` to fan the independent per-xcat scorings in `make_zn_scores` (serial default).
**Likely deferred:** cell-19 cost is `reduce_df` (Q4), not the z-score math (expanding estimation
already `O(n)`); inner per-series estimation is sequential. With **per-xcat slicing** (fan out one
xcat's panel per worker, via an iterator producer) the process-pool memory objection goes away, but
the residual upside is small. Reassess only if meaningful cost remains after Q4. See TARGETS §3.1.
- **Cells:** 19. **Baseline:** 139s / 8217 MiB.
- **After:** DEFERRED, no code change. Measured post-Q4 (cProfile, neutral=zero / cell-19 case):
  z-score expanding math is only **~3%** of `make_zn_scores`; the dominant cost is `min()`/`max()`
  datetime-builtin iteration (~47%) + `date_range` (~13%), which are sequential per-call overhead
  with no parallelism opportunity, and the small per-xcat granule (~134 ms) is below worker-spawn
  overhead. Parallelism not worthwhile after Q4 (T7 confirmed deferred per TARGETS §T7/§3.1).
  `test_zn_scores.py` 17 passed, `tests/unit/panel` 210 passed. No PR. Corroborates the T5a finding
  (the real `make_zn_scores` lever is the `min`/`max` builtins, not parallelism). Measured on the
  Q4 worktree (Q4 not yet merged).

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
- **T5a** `perf/zn-scores-minmax` (spun out of Q6, trivial, high-value): in
  `make_zn_scores.py::_make_zn_scores_for_xcat` (≈257-258) the Python builtins
  `min(df["real_date"])`/`max(df["real_date"])` iterate the datetime Series element-by-element
  (~37% of `make_zn_scores` runtime). Replace with `df["real_date"].min()`/`.max()` —
  byte-identical output, ~37% win. Strong candidate to schedule.
- **T5b** QDF-wrapping overhead (spun out of Q6): `QuantamentalDataFrame(...)` wrapping at
  `make_zn_scores.py` start costs ~22.6% per call (categorical conversion) — points back at
  T1/T2c QDF-construction overhead, not a standalone fix.
