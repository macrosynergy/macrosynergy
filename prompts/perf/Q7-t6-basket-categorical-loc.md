# Q7 · T6 · `perf/basket-categorical-loc` — builder brief

> Item: **Q7** (QUEUE.md) · Target: **T6** (TARGETS.md §T6 / §7.2) · depends-on: **none**
> Branch/worktree: `perf/basket-categorical-loc` · Base: `feature/performance`

## Goal

Fix the latent `InvalidIndexError` that makes `Basket` reject a **categorical**
`QuantamentalDataFrame`, and close the test-coverage gap that let it ship. `Basket.make_weights`
slices its wide frame with `dfw_wgs[fvi:]` (a `Timestamp` slice); when the columns are a
`CategoricalIndex`, pandas evaluates `fvi in columns` → `CategoricalIndex.__contains__` →
**`InvalidIndexError`**. Object-dtype input works, so the bug is latent. This blocks passing a
categorical `dfx` to `Basket` (cell 27) without an object-copy that costs **+25% peak RSS**.

Read **`prompts/TARGETS.md` §T6** and §7.2. This is a **correctness + test-coverage** item (small,
self-contained), not a speed change to the object path — see the GATE framing below.

## Files

- **Modify:** `macrosynergy/panel/basket.py` — `make_weights` (411); the bug is the slice at
  **line 502** `dfw_wgs = dfw_wgs[fvi:]`. **Audit** `Basket` for sibling patterns (e.g. the column
  subselect at ≈760 `dfw_wgs[[single_ticker]]`, and any other `df[ts:]` / `ts in columns` usage).
- **Add (new test):** `tests/unit/panel/test_basket_performance.py` (class `TestAll`) — a regression
  test running `Basket` (`make_basket`/`make_weights`/`return_basket`) on a **categorical**
  `QuantamentalDataFrame`, asserting output equals the object-dtype run. (This is the brief's only
  test target; the builder writes test code, never goldens or git.)
- Do **not** change `Basket`'s public API or output. Do **not** touch `tests/perf/golden/`.

## Design (output identical)

- **The fix:** use label-based row slicing — `dfw_wgs = dfw_wgs.loc[fvi:]` — which slices by the row
  label and does **not** probe the columns index, so a `CategoricalIndex` on the columns no longer
  raises. Confirm the row index is the date axis at that point (so `.loc[fvi:]` is the correct
  start-from-`fvi` row slice) and that the result equals today's object-path output exactly.
- **Audit, don't over-fix:** only convert the patterns that actually break on categorical columns;
  a `df[[col]]` list-column subselect is label-based already and likely fine — verify rather than
  blindly rewrite. Each change must be output-identical on object input.
- **Regression test (root cause = object-only coverage of a dtype-polymorphic API):** add a
  categorical-`QuantamentalDataFrame` case that exercises the basket path end-to-end and asserts
  value parity with the object-dtype result.

## GATE (verify ALL before hand-back; `--no-cov -n0`, never `-p no:cov`)

> **Coverage note:** T6 has **no pre-existing `tests/perf` benchmark/parity module** — the new
> regression test you add *is* the GATE-2 artifact, alongside the existing basket suite. The "win" is
> a correctness enable (categorical input works) + removal of cell-27's +25% object-copy RSS, which
> the manager confirms on the macro harness. There is no in-repo object-path speed benchmark to beat.

1. **Bug fixed + parity preserved (GATE-1/2) — must be GREEN:**
   ```bash
   pytest tests/unit/panel/test_basket_performance.py --no-cov -n0     # incl. your new categorical case
   ```
   The new test must **fail before** your `basket.py` fix (it reproduces `InvalidIndexError`) and
   **pass after** — verify both directions and note it in your report. Object-input cases stay green.
2. **No object-path regression (GATE-3 proxy):** the existing object-dtype basket tests run with no
   slowdown or behaviour change. (No `-m perf` benchmark exists for `Basket`; do not invent a golden.)
3. **macrosynergy suite (GATE-4):**
   ```bash
   pytest tests/unit/panel --no-cov -n0
   ```
   (The manager runs the full suite + the macro cell-27 RSS confirmation as the merge gate.)
4. **Hygiene:** `git status` shows only `basket.py` and `test_basket_performance.py` changed; no
   `tests/perf/golden/*` modified; no scratch files.

## Acceptance criteria

- [ ] `make_weights` slices rows with `.loc[fvi:]` (or equivalent label-safe slice); categorical
  `QuantamentalDataFrame` input no longer raises `InvalidIndexError`.
- [ ] `Basket` audited for sibling `df[ts:]` / `ts in columns` patterns; any that break on
  categorical columns are fixed output-identically, others verified safe.
- [ ] New regression test added that runs `Basket` on a categorical `QuantamentalDataFrame` and
  asserts parity with the object-dtype run; it fails pre-fix and passes post-fix.
- [ ] Object-dtype basket output unchanged; `Basket` public API unchanged.
- [ ] `tests/unit/panel` passes; `tests/perf/golden/` unchanged.

## Notes

- TARGETS §T6 asks to treat "object-only test coverage of a dtype-polymorphic API" as the root
  cause and to add a categorical-input case to the **shared panel-function test matrix** (audit
  `linear_composite` / `make_relative_value` / `panel_calculator` / `make_zn_scores` /
  `SignalReturnRelations`). If extending the matrix beyond `Basket` is more than a small addition,
  **report the gap as a finding** so the manager can scope it — keep this item focused on `Basket`.
- This item unblocks the QDF-native notebook (removes the cell-27 object-copy) but does not by itself
  change the dominant cell costs — that's T2c/T1/T2/T3.
