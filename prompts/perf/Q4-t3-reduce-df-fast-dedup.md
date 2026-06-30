# Q4 · T3 · `perf/reduce-df-fast-dedup` — builder brief

> Item: **Q4** (QUEUE.md) · Target: **T3** (TARGETS.md §T3, rank 4) · depends-on: **none**
> Branch/worktree: `perf/reduce-df-fast-dedup` · Base: `feature/performance`

## Goal

Make `reduce_df`'s object-dtype fallback stop paying for a terminal all-column `drop_duplicates()`
on every call. On a clean panel there are no full-row duplicates, so that dedup is pure overhead —
yet it costs cell 12 ≈5.0s (`drop_duplicates` 3.3s + `factorize` 1.5s) and recurs inside
`make_zn_scores` (19), `linear_composite` (17/18), `NaivePnL.__init__` (41/42/43/47/49, 6× each in
43/47), `make_relative_value`, and directly in cells 13/15/27 — large in aggregate via call count.

Read the full design in **`prompts/TARGETS.md` §T3** (and §3, §7.3, §5/§5.1).

## Dual implementation — fix BOTH (TARGETS §7.3)

- **Object path** (notebook's path today): `macrosynergy/management/utils/df_utils.py::reduce_df`
  (688) — fallback ends with `df.drop_duplicates()` over **all** columns (≈793).
- **Categorical/QDF-native twin:** `macrosynergy/management/types/qdf/methods.py::reduce_df` (309).

Fixing only one implementation is an incomplete item — the reviewer checks both.

## Files

- **Modify:** `macrosynergy/management/utils/df_utils.py` — `reduce_df` object fallback.
- **Modify:** `macrosynergy/management/types/qdf/methods.py` — `reduce_df`.
- Do **not** change any signature (incl. the `out_all` tuple-vs-frame return contract), return type,
  dtype, or row order. Do **not** touch `tests/perf/golden/` (the parity contract).

## Design (API + output identical)

The terminal `drop_duplicates()` must keep producing the identical frame, faster. Two acceptable
approaches (pick the one that's provably identical):

- **Fast dedup on codes:** dedup on the index-col subset using factor codes (mirrors T1) rather than
  hashing all object columns.
- **Unique-index guard:** if the index columns are already unique
  (`df.duplicated(subset=IndexCols).any()` is `False`), skip the all-column `drop_duplicates()`
  entirely and return the frame unchanged.

Either way: **exact same rows, same order, same dtype.** The two parity tests pin the behaviour that
matters in both directions:
- `test_reduce_df_dedup_matches_golden` — when there *are* exact-duplicate rows, they must still be
  removed (don't skip dedup unconditionally).
- `test_reduce_df_no_spurious_row_drop_on_clean_panel` — on a clean panel, no rows may be dropped.

## GATE (verify ALL before hand-back; `--no-cov -n0`, never `-p no:cov`)

1. **Parity + behaviour preserved (GATE-1/2) — must stay GREEN:**
   ```bash
   pytest tests/perf/test_parity_reduce_df.py -v --no-cov -n0                                 # 2 passed
   pytest tests/unit/management/test_qdf.py -k "ReduceDfEdgeAPI" -v --no-cov -n0               # 5 passed
   ```
2. **Measurable win (GATE-3):**
   ```bash
   pytest tests/perf/test_perf_reduce_df.py -m perf -k small --benchmark-only -n0 --no-cov \
     --benchmark-json=<scratch>/after.json
   python tests/perf/record.py <baseline-json> <scratch>/after.json
   ```
   `test_bench_reduce_df_full[obj-small]`, `[cat-small]`, and `test_bench_reduce_df_filtered[small]`
   should drop (removing the redundant all-column dedup). Neither `obj` nor `cat` may regress.
3. **macrosynergy suite (GATE-4):** `reduce_df` is very widely used —
   ```bash
   pytest tests/unit/management tests/unit/panel --no-cov -n0
   ```
   (The manager runs the full suite as the merge gate.)
4. **Hygiene:** `git status` shows only the two target files changed; no `tests/perf/golden/*`
   modified; no scratch files.

## Acceptance criteria

- [ ] **Both** implementations fixed — `df_utils.py` object fallback **and** `qdf/methods.py` twin.
- [ ] Terminal all-column `drop_duplicates()` is replaced by a code-based dedup or guarded so it's
  skipped only when provably a no-op.
- [ ] Genuine exact-duplicate rows are still removed (`test_reduce_df_dedup_matches_golden`); clean
  panels lose no rows (`test_reduce_df_no_spurious_row_drop_on_clean_panel`).
- [ ] `out_all` tuple-return contract and signature unchanged (`test_out_all_returns_tuple`,
  `test_signature_unchanged` in `ReduceDfEdgeAPI`).
- [ ] Benchmark shows a win on `obj` and `cat` (record.py before/after), no regression.
- [ ] `tests/unit/management` + `tests/unit/panel` pass; `tests/perf/golden/` unchanged.

## Notes

- **Q6 (T5) and Q8 (T7) depend on this item** — they reuse the reduce_df modules and reassess the
  cell-19 residual after T3. Land a clean, output-identical fix.
- If you find `reduce_df` callers that re-reduce the same frame redundantly (e.g. the `make_zn_scores`
  loop), **report it as a finding** for Q6 — do not widen scope here.
