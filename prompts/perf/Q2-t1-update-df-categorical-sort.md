# Q2 · T1 · `perf/update-df-categorical-sort` — builder brief

> Item: **Q2** (QUEUE.md) · Target: **T1** (TARGETS.md §T1, rank 2) · depends-on: **none**
> Branch/worktree: `perf/update-df-categorical-sort` · Base: `feature/performance`

## Goal

Make `update_df`/`update_tickers` stop sorting and deduplicating on **object strings**. In the
object-dtype fallback, do the dedup + the final `IDX_COLS_SORT_ORDER` sort on **factorized integer
codes** of `cid`/`xcat`, then restore the original dtype and the identical canonical row order. This
is the single largest time sink in the notebook — cell 17 spends **≈330s / 80% of the cell** here
(`drop_duplicates` 182s + `factorize` 132s), because `dfx = update_df(dfx, piece)` in a loop
re-sorts + re-dedups the whole growing frame on every call.

Read the full design in **`prompts/TARGETS.md` §T1** (and §3, §7.3, §5/§5.1).

## Dual implementation — fix BOTH (TARGETS §7.3)

`update_df` has an object path and a categorical/QDF-native twin, and **both are slow**:

- **Object path** (the notebook's path today): `macrosynergy/management/utils/df_utils.py`
  — `update_df` (561) → `update_tickers` (627); the final `df.sort_values(IDX_COLS_SORT_ORDER)`
  (≈624) and `drop_duplicates(subset=[real_date, xcat, cid])` (≈651).
- **Categorical/QDF-native twin** (TARGETS §7.3): `macrosynergy/management/types/qdf/methods.py`
  — `update_df` (458) / `update_tickers` (493) — **and** `macrosynergy/management/types/qdf/classes.py`
  — `update_df` (291). Still ~330s in cell 17 under categorical input: it re-sorts the growing frame
  every call, and `union_categoricals` is skipped when `df_add` is object (the
  `linear_composite`/`panel_calculator` output) so the concat upcasts back to object and the
  categorical "washes out". The categorical path should keep the categorical dtype by
  re-categorizing the small `df_add` rather than upcasting the whole frame, and sort/dedup on codes.

Fixing only one implementation is an incomplete item — the reviewer checks both.

## Files

- **Modify:** `macrosynergy/management/utils/df_utils.py` — object-path `update_df`/`update_tickers`.
- **Modify:** `macrosynergy/management/types/qdf/methods.py` — `update_df`/`update_tickers`.
- **Modify:** `macrosynergy/management/types/qdf/classes.py` — `update_df`.
- Do **not** change any signature, return type, returned dtype, or row order. Do **not** touch
  `tests/perf/golden/` (the parity contract).

## Design (API + output identical)

- **Last-write-wins + canonical order are sacred.** The dedup must keep the same row that the
  current code keeps (last occurrence per `(real_date, xcat, cid)`), and the output must be sorted
  ascending by `IDX_COLS_SORT_ORDER` exactly as today. `test_update_df_invariants_last_wins_and_sorted`
  pins both — keep it green.
- **Object path:** factorize `cid` and `xcat` once, build an integer key, `np.lexsort`/`argsort` on
  `(cid_code, xcat_code, real_date)`, dedup on the same codes, then reindex back to object dtype.
  Same rows, same order, same dtype — a constant-factor win (object string sort/hash → int sort/hash).
- **Categorical twin:** keep `cid`/`xcat` categorical across the update — re-categorize the (small)
  `df_add` to the running frame's categories (or `union_categoricals` the add in) instead of
  upcasting the whole frame to object; sort/dedup on the category codes.
- Must **not mutate the input** frame — `test_update_df_does_not_mutate_input` pins this.

## GATE (verify ALL before hand-back; `--no-cov -n0`, never `-p no:cov`)

1. **Parity + behaviour preserved (GATE-1/2) — must stay GREEN:**
   ```bash
   pytest tests/perf/test_parity_update_df.py -v --no-cov -n0                                  # 3 passed
   pytest tests/unit/management/test_update_df.py -k "UpdateDfEdge" -v --no-cov -n0             # 7 passed
   ```
2. **Measurable win (GATE-3):**
   ```bash
   pytest tests/perf/test_perf_update_df.py -m perf -k small --benchmark-only -n0 --no-cov \
     --benchmark-json=<scratch>/after.json
   python tests/perf/record.py <baseline-json> <scratch>/after.json
   ```
   `test_bench_update_df_growing_loop[obj-small]` and `[cat-small]` plus `test_bench_update_tickers[small]`
   should drop substantially (the growing-loop case is where the re-sort/re-dedup compounds). **Neither
   the `obj` nor the `cat` branch may regress** — this is a both-paths item.
3. **macrosynergy suite (GATE-4):** `update_df` is used pervasively, so run broadly —
   ```bash
   pytest tests/unit/management --no-cov -n0
   ```
   (The manager runs the full suite as the merge gate.)
4. **Hygiene:** `git status` shows only the three target files changed; no `tests/perf/golden/*`
   modified; no scratch files.

## Acceptance criteria

- [ ] **Both** implementations fixed — `df_utils.py` object path **and** the
  `qdf/methods.py`+`qdf/classes.py` categorical twins (TARGETS §7.3).
- [ ] Object path dedups/sorts on integer factor codes, not object strings; output rows, order, and
  dtype are byte-identical to pre-change.
- [ ] Categorical path keeps `cid`/`xcat` categorical (re-categorizes the add; no whole-frame
  upcast) and sorts/dedups on codes.
- [ ] Last-write-wins + `IDX_COLS_SORT_ORDER` ascending order preserved; input not mutated
  (`test_parity_update_df.py` 3 passed).
- [ ] Public signatures unchanged (`test_update_df_signature_unchanged`,
  `test_update_tickers_signature_unchanged` in `TestUpdateDfEdge`).
- [ ] Benchmark shows a material win (record.py before/after) on both `obj` and `cat`, no regression.
- [ ] `tests/unit/management` passes; `tests/perf/golden/` unchanged.

## Notes

- The categorical-twin parity is benchmarked (object vs categorical params) but not yet goldened
  (TARGETS §5.1) — if you can cheaply add a categorical golden via
  `python tests/perf/capture_parity.py --update`, **report it as a finding for the manager** rather
  than committing a new golden yourself (the manager owns goldens; items never regenerate them).
- If you find the real cell-17 win needs *batching* the composite loop (one `update_df` instead of
  N), that is a **notebook-side / separate** change (TARGETS §3.1) — report it, don't widen scope.
- T1b (categorical propagation / changing returned dtype) is **out of scope** and demoted (§7.1).
