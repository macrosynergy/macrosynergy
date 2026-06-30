# Q3 · T2 · `perf/ticker-split-vectorize` — builder brief

> Item: **Q3** (QUEUE.md) · Target: **T2** (TARGETS.md §T2, rank 3) · depends-on: **none**
> Branch/worktree: `perf/ticker-split-vectorize` · Base: `feature/performance`

## Goal

Stop `split_ticker` (and `get_cid`/`get_xcat`) from re-splitting the same handful of unique tickers
millions of times, and stop `ticker_df_to_qdf` from building a 30M-row `"ticker"` string column it
immediately splits. In cell 13, `ticker_df_to_qdf` is **133s (>½ the cell)**, of which `split_ticker`
is called **61.5M times** (~62s self) over only ~thousands of distinct tickers — ~10⁴× redundant work
— and is a major contributor to the cell's +12 GiB peak.

Read the full design in **`prompts/TARGETS.md` §T2** (and §3, §5/§5.1). This target is
**dtype-independent** — TARGETS §7.3 dual-implementation does **not** apply (mark that review
dimension N/A).

## Files

- **Modify:** `macrosynergy/management/utils/core.py` — `split_ticker` iterable branch (≈ line 74;
  function at 44), consumed by `get_cid` (98) / `get_xcat` (116).
- **Modify:** `macrosynergy/management/utils/df_utils.py` — `ticker_df_to_qdf` (251).
- Do **not** change the scalar branch's semantics, any signature, or the returned list/QDF type/order.
  Do **not** touch `tests/perf/golden/` (the parity contract).

## Design (API + output identical) — two composable levels, ship together

- **Level A — factorize-on-uniques in `split_ticker`'s iterable branch.** Semantics unchanged:
  `cid` = text before first `_`, `xcat` = remainder (`maxsplit=1`). Replace the per-element recursion
  with a hash-once-on-uniques pattern:
  ```python
  arr = np.asarray(ticker, dtype=object)
  if arr.size == 0:
      raise ValueError("Argument `ticker` must not be empty.")
  codes, uniq = pd.factorize(arr, sort=False)        # ~thousands of uniques
  split = np.array([split_ticker(t, mode) for t in uniq], dtype=object)  # scalar path = validation
  return split[codes].tolist()                        # identical full-length Python list
  ```
  Validation/error semantics are preserved because the unchanged **scalar** path still runs on each
  unique (a malformed ticker among the uniques still raises `ValueError`).
- **Level B — split the column labels in `ticker_df_to_qdf`, not the stacked column.** The wide
  frame's columns are already the unique tickers; split them once (cheap via Level A) and carry
  `cid`/`xcat` through the stack instead of materializing a 30M-row `"ticker"` column + two derived
  object columns:
  ```python
  cids, xcats = get_cid(df.columns), get_xcat(df.columns)
  df.columns = pd.MultiIndex.from_arrays([cids, xcats], names=["cid", "xcat"])
  out = df.stack(["cid", "xcat"], future_stack=True).reset_index().rename(columns={0: metric})
  return standardise_dataframe(out)
  ```
  This removes the row-level split entirely and cuts the peak-RSS spike. B's label split is made
  cheap by A — ship both. The output must match `standardise_dataframe`'s exact column order/dtype.

## GATE (verify ALL before hand-back; `--no-cov -n0`, never `-p no:cov`)

1. **Parity + behaviour preserved (GATE-1/2) — must stay GREEN:**
   ```bash
   pytest tests/perf/test_parity_ticker_split.py -v --no-cov -n0                              # 2 passed
   pytest tests/unit/management/test_utils.py -k "SplitTickerDirect" -v --no-cov -n0           # 9 passed
   ```
   `test_ticker_df_to_qdf_columns` pins the exact `["real_date", "cid", "xcat", "value"]` order;
   `test_ticker_df_to_qdf_matches_golden` pins value parity.
2. **Measurable win (GATE-3):**
   ```bash
   pytest tests/perf/test_perf_ticker_split.py -m perf -k "2000-50 or 500-1300" --benchmark-only -n0 \
     --no-cov --benchmark-json=<scratch>/after.json
   python tests/perf/record.py <baseline-json> <scratch>/after.json
   ```
   `test_bench_get_cid_large_list[2000-50]`, `test_bench_get_xcat_large_list[2000-50]`, and
   `test_bench_ticker_df_to_qdf[500-1300]` should drop substantially (the get_cid/get_xcat cases scale
   with redundant repeats; `ticker_df_to_qdf` benefits from Level B). No regression on the scalar path.
3. **macrosynergy suite (GATE-4):** `split_ticker`/`get_cid`/`get_xcat`/`ticker_df_to_qdf` are widely
   used —
   ```bash
   pytest tests/unit/management --no-cov -n0
   ```
   (The manager runs the full suite as the merge gate.)
4. **Hygiene:** `git status` shows only `core.py` and `df_utils.py` changed; no `tests/perf/golden/*`
   modified; no scratch files.

## Acceptance criteria

- [ ] `split_ticker`'s iterable branch no longer recurses per element — it factorizes to uniques,
  runs the scalar path only on uniques, and reindexes to a full-length identical list.
- [ ] Malformed-ticker / empty-input / non-string error semantics unchanged (the `SplitTickerDirect`
  edge tests, incl. `test_non_string_ticker_raises_typeerror`, pass — 9 passed).
- [ ] `ticker_df_to_qdf` builds `cid`/`xcat` from the column labels (Level B); output columns,
  order, dtype, and values are byte-identical (`test_parity_ticker_split.py` 2 passed).
- [ ] Public signatures unchanged (`test_signature_unchanged` in `SplitTickerDirect`).
- [ ] Benchmark shows a material win (record.py before/after); scalar path not regressed.
- [ ] `tests/unit/management` passes; `tests/perf/golden/` unchanged.

## Notes

- Level A and Level B are a depends-on pair (A makes B cheap) but are independently parity-safe —
  if B's reshape is risky, land A first and verify, then B; both must be in the final diff.
- `get_cid`/`get_xcat` are thin wrappers over `split_ticker` and get fast for free — you should not
  need to edit them beyond what Level A provides. If you spot another `to_qdf`-heavy path,
  **report it as a finding**, don't widen scope.
