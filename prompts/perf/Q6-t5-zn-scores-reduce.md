# Q6 · T5 · `perf/zn-scores-reduce` — builder brief

> Item: **Q6** (QUEUE.md) · Target: **T5** (TARGETS.md §T5, rank 6) · **depends-on: Q4 (T3) DONE**
> Branch/worktree: `perf/zn-scores-reduce` · Base: `feature/performance` (with Q4 already merged)

## Goal

Investigate whether the repeated full-frame `reduce_df` inside the cell-19 `make_zn_scores` loop can
be **hoisted/scoped** to buy more than Q4 (T3) already does. Cell 19 calls `make_zn_scores` ~20+ times
(per xcat × scope), each `reduce_df`-ing the full object-dtype frame (profile: `factorize` 58s +
categorical `_from_sequence` 50s). T3 already makes each `reduce_df` faster; T5 asks whether the
**call pattern** itself is wasteful (e.g. the same frame reduced redundantly across iterations).

Read **`prompts/TARGETS.md` §T5** and §3.1 (note: this is *not* a parallelism target — that's the
optional Q8). This item is **gated on residual cost after Q4** — see "Possible outcomes" below.

## Files

- **Investigate / modify:** `macrosynergy/panel/make_zn_scores.py::make_zn_scores` (19) — the
  internal `reduce_df` call (≈ line 250) inside the per-call flow.
- Do **not** change `make_zn_scores`'s signature, return type, or output. Do **not** re-implement
  `reduce_df` (that was Q4). Do **not** touch `tests/perf/golden/`.

## Design (output identical)

- First, **measure the residual** after Q4: profile cell 19 (macro harness) / the reduce_df benchmark
  to see how much of the loop cost is still `reduce_df` call-pattern overhead vs the (already-`O(n)`)
  z-score estimation.
- If there is a genuine win, hoist/scope the `reduce_df` so the same frame isn't reduced redundantly
  across iterations (e.g. reduce once to the needed cids/xcats/date-window before the loop, or cache
  the reduced slice when the reduction args are invariant across iterations). The output of
  `make_zn_scores` must be **byte-identical** — the reduction must select exactly the same rows it
  does today for each call.

## Possible outcomes (both are acceptable hand-backs)

1. **A real, output-identical win** exists → implement it, show the benchmark/macro delta.
2. **T5 is subsumed by T3** (the loop cost is the z-score math / per-series estimation, which is
   already `O(n)` and sequential, not redundant `reduce_df`) → **hand back a "no change, subsumed by
   Q4" finding with the supporting measurement.** Do **not** force a speculative refactor that risks
   parity for no measured gain. This is the expected outcome per TARGETS §T5/§3.1 if Q4 already
   captured the cost.

## GATE (verify ALL before hand-back; `--no-cov -n0`, never `-p no:cov`)

> **Coverage note:** there is no dedicated `tests/perf` zn-scores module (TARGETS §5.1) — T5 reuses
> the T3 `reduce_df` benchmark/parity and is confirmed on the macro cell-19 harness. If you implement
> a change, the parity guard is the `make_zn_scores` unit suite (output must be unchanged).

1. **Output parity (GATE-1/2) — must stay GREEN:**
   ```bash
   pytest tests/unit/panel/test_zn_scores.py --no-cov -n0           # make_zn_scores output unchanged
   pytest tests/perf/test_parity_reduce_df.py -v --no-cov -n0       # 2 passed (reduce_df contract intact)
   ```
2. **Measurable win (GATE-3)** — *only if you implemented a change*:
   ```bash
   pytest tests/perf/test_perf_reduce_df.py -m perf -k small --benchmark-only -n0 --no-cov \
     --benchmark-json=<scratch>/after.json
   python tests/perf/record.py <baseline-json> <scratch>/after.json
   ```
   The decisive evidence for T5 is the **macro cell-19** wall/RSS delta (manager runs it). If the
   finding is "subsumed by Q4", state that with the measurement instead of a benchmark win.
3. **macrosynergy suite (GATE-4):**
   ```bash
   pytest tests/unit/panel --no-cov -n0
   ```
4. **Hygiene:** `git status` shows only `make_zn_scores.py` changed (or nothing, if the outcome is a
   no-change finding); no `tests/perf/golden/*` modified; no scratch files.

## Acceptance criteria

- [ ] Residual cell-19 reduce_df cost (post-Q4) is **measured** and reported.
- [ ] Either an output-identical hoist/scope win is implemented (with delta), **or** a clear
  "subsumed by Q4 — no change" finding is handed back with the measurement.
- [ ] `make_zn_scores` signature, return type, and output unchanged (`test_zn_scores.py` green).
- [ ] No speculative refactor that risks parity for no measured gain.
- [ ] `tests/unit/panel` passes; `tests/perf/golden/` unchanged.

## Notes

- Q8 (T7, optional) is the *parallel* zn-scores lever and is separate — do not parallelize here.
- If you find the cost is the categorical `_from_sequence` conversion rather than `reduce_df`,
  **report it as a finding** (it may point back at T1/T2c), don't widen scope.
