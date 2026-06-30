# Q1 · T2c · `perf/qdf-ticker-series-vectorize` — builder brief

> Item: **Q1** (QUEUE.md) · Target: **T2c** (TARGETS.md §T2c, the ★ top target) · depends-on: **none**
> Branch/worktree: `perf/qdf-ticker-series-vectorize` · Base: `feature/performance`

## Goal

Vectorize `_get_tickers_series` so the **categorical** branch stops doing a per-row Python f-string
comprehension + full-length `pd.Categorical` rebuild over millions of rows. Build the ticker on the
**observed unique `(cid_code, xcat_code)` pairs** (a few thousand) instead of per row. This one
function is the common root of the two slowest cells (18, 27) via `add_ticker_column`
(`panel_calculator`/`make_relative_value`) and `reduce_df_by_ticker` (`msp.Basket`).

Read the full design in **`prompts/TARGETS.md` §T2c** (and §3, §5/§5.1). This is **dtype-independent
/ single function** — TARGETS §7.3 dual-implementation does **not** apply (mark that review
dimension N/A).

## Files

- **Modify (only this):** `macrosynergy/management/types/qdf/methods.py` — the function
  `_get_tickers_series` (≈ lines 172–212), specifically the categorical branch (≈ 200–210) that
  currently does:
  ```python
  ticker_labels = [f"{cid}_{xcat}" for cid, xcat in zip(cid_labels, xcat_labels)]
  categories = pd.unique(pd.Series(ticker_labels))
  ticker_series = pd.Categorical(ticker_labels, categories=categories, ordered=True)
  ```
- Do **not** change the non-categorical branch's observable result, the function signature, or any
  other file. Do **not** touch `tests/perf/golden/` (the parity contract).

## Design (output-identical)

Replace the per-row loop with a uniques-based build that yields the **identical ordered
`Categorical`** (same categories, same order, same codes):

- Combine the two integer code arrays (`df["cid"].cat.codes`, `df["xcat"].cat.codes`) into a single
  per-row key; take the **observed unique** pairs; format `f"{cid}_{xcat}"` only for those uniques to
  get the labels; assemble the result `Categorical` from codes. **OR** the simpler vectorized form
  `cid.astype(str) + "_" + xcat.astype(str)` then re-categorize.
- Either way the result must reproduce the **first-appearance category order** with `ordered=True`
  exactly as the current code does (categories via `pd.unique` over the row labels in row order).
  The parity test `test_get_tickers_series_categorical_contract` pins this — keep it green.

## GATE (verify ALL before hand-back; `--no-cov -n0`, never `-p no:cov`)

1. **Parity + behaviour preserved (GATE-1/2) — must stay GREEN:**
   ```bash
   pytest tests/perf/test_parity_qdf_ticker_series.py -v --no-cov -n0          # 3 passed
   pytest tests/unit/management/test_qdf.py -k "GetTickersSeriesEdge or AddTickerColumnAPI" -v --no-cov -n0   # 7 passed
   ```
2. **Measurable win (GATE-3):**
   ```bash
   pytest tests/perf/test_perf_qdf_ticker_series.py -m perf -k small --benchmark-only -n0 --no-cov \
     --benchmark-json=<scratch>/after.json
   python tests/perf/record.py <baseline-json> <scratch>/after.json
   ```
   Expect `test_bench_get_tickers_series[cat-small]` and `test_bench_add_ticker_column[small]` to
   drop **substantially** (baseline ≈ 32 ms each; the categorical path should no longer be ~6× the
   object branch — target roughly the object branch's ~5 ms, or at least a large reduction). The
   `obj` branch must **not** regress.
3. **macrosynergy suite (GATE-4):** the affected callers live under panel + management —
   ```bash
   pytest tests/unit/management tests/unit/panel --no-cov -n0
   ```
   (The manager runs the broader suite as the merge gate.)
4. **Hygiene:** `git status` shows only `macrosynergy/management/types/qdf/methods.py` changed; no
   `tests/perf/golden/*` modified; no scratch files.

## Acceptance criteria

- [ ] `_get_tickers_series`'s categorical branch no longer contains a per-row Python f-string
  comprehension over all rows (it builds on observed unique code-pairs or a vectorized
  astype+recategorize).
- [ ] Output `Categorical` compares **equal** to pre-change (categories set AND order, `ordered`
  flag, codes) — `test_get_tickers_series_categorical_contract` and the object/categorical parity
  tests pass.
- [ ] Public signature unchanged — `test_signature_unchanged` (params `["df","cid_column","xcat_column"]`,
  defaults `"cid"`/`"xcat"`) passes.
- [ ] Benchmark shows a material categorical-branch speed-up (record.py before/after) with no object-branch regression.
- [ ] `tests/unit/management` + `tests/unit/panel` pass; `tests/perf/golden/` unchanged.

## Notes

- The non-categorical branch (`return df[cid_column] + "_" + df[xcat_column]`) is already vectorized
  — leave its result identical; you may keep it as-is.
- If you find a second hot path or a related bug outside this function, **report it as a finding** —
  do not widen scope (e.g. `reduce_df_by_ticker`/`add_ticker_column` call this function and get fast
  for free; you should not need to edit them).
