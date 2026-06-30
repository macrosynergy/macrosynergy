# Q8 · T7 · `perf/zn-scores-parallel` — builder brief  (OPTIONAL / low priority)

> Item: **Q8** (QUEUE.md) · Target: **T7** (TARGETS.md §T7 / §3.1) · **depends-on: Q4 (T3) DONE;
> gated on residual cost after Q4 (and Q6)**
> Branch/worktree: `perf/zn-scores-parallel` · Base: `feature/performance` (Q4 already merged)

## Goal

Add an **opt-in** `n_jobs` to fan the **independent per-xcat** scorings in `make_zn_scores` across
workers (serial default, identical output). This is **likely deferred**: cell-19 cost is dominated
by `reduce_df` (Q4) and categorical conversion, not the z-score math (expanding estimation is already
`O(n)`), and the inner per-series estimation is sequential and GIL-bound. With **per-xcat slicing**
the process-pool memory objection goes away, but the residual upside is expected to be small.

Read **`prompts/TARGETS.md` §T7** and §3.1. **Do this only if a residual, parallelizable cost
remains after Q4 (and Q6's investigation).**

## Gate-before-you-build (decision step — do this FIRST)

Measure the post-Q4 cell-19 / `make_zn_scores` cost and decide:

- **If the residual outer-loop (per-xcat) cost is meaningful** and the inner estimation is the
  bottleneck only per-series (not parallelizable) → proceed with the opt-in `n_jobs` design below.
- **If the residual is small** (cost was `reduce_df`/conversion, already addressed by Q4) → **hand
  back a "deferred — not worthwhile after Q4" finding with the measurement, and make NO code change.**
  This is the expected outcome per TARGETS §T7/§3.1.

State which branch you took and the supporting measurement in your report.

## Files

- **Modify (only if proceeding):** `macrosynergy/panel/make_zn_scores.py::make_zn_scores` (19) — add
  an opt-in worker param defaulting to serial.
- Do **not** change existing default behaviour, signature semantics for existing callers, return
  type, or output. Do **not** touch `tests/perf/golden/`.

## Design (output identical; serial default) — only if proceeding

- **Per-xcat slicing (§3.1):** fan out **one xcat's panel per worker** via an iterator/producer that
  yields per-xcat slices lazily — each worker holds ~`1/n_xcats` of the frame, so a process pool no
  longer multiplies the cell's ~8 GiB peak. Do **not** hand each worker the whole frame.
- **Serial default.** New `n_jobs`-style param defaults to 1 / current behaviour. The inner
  per-series expanding estimation stays sequential (it is not parallelizable without changing
  results).
- **Determinism.** Output must be identical regardless of worker count — reassemble per-xcat results
  in the original xcat order, byte-identical to the serial result.

## GATE (verify ALL before hand-back; `--no-cov -n0`, never `-p no:cov`)

> **Coverage note:** no dedicated `tests/perf` zn-scores module (TARGETS §5.1) — T7 reuses the T3
> `reduce_df` modules and the macro cell-19 harness; the parity guard for any change is the
> `make_zn_scores` unit suite (output must be unchanged at default and with `n_jobs>1`).

1. **Output parity (GATE-1/2) — must stay GREEN at default AND with workers:**
   ```bash
   pytest tests/unit/panel/test_zn_scores.py --no-cov -n0     # output identical, serial default unchanged
   ```
   Add/extend a test asserting `n_jobs>1` produces output identical to the serial run.
2. **Measurable win (GATE-3)** — *only if you implemented the parallel path*: demonstrate the
   per-xcat parallel speedup on the macro cell-19 harness (manager runs it) with peak RSS **not**
   inflated (the per-xcat slicing is what keeps memory bounded — show it). If deferred, report the
   measurement that justified deferral instead.
3. **macrosynergy suite (GATE-4):**
   ```bash
   pytest tests/unit/panel --no-cov -n0
   ```
4. **Hygiene:** `git status` shows only `make_zn_scores.py` changed (or nothing, if deferred); no
   `tests/perf/golden/*` modified; no scratch files.

## Acceptance criteria

- [ ] Post-Q4 residual cell-19 cost **measured**; the proceed-vs-defer decision is stated with evidence.
- [ ] If proceeding: opt-in `n_jobs` with **serial default**; per-xcat slicing via an iterator (no
  whole-frame fan-out); output byte-identical at default and with `n_jobs>1`; peak RSS not inflated.
- [ ] If deferring: clear "not worthwhile after Q4" finding with the measurement; no code change.
- [ ] `make_zn_scores` default behaviour, signature, and output unchanged for existing callers
  (`test_zn_scores.py` green).
- [ ] `tests/unit/panel` passes; `tests/perf/golden/` unchanged.

## Notes

- This is the lowest-priority queue item and is explicitly **gated**. Prefer the honest "deferred"
  finding over a speculative parallel refactor that adds concurrency risk for little measured gain.
- The inner per-series estimation is **not** a parallelism target (sequential, already `O(n)`); a
  walk-forward iterator is only a memory-bounded *producer* here, not a speed lever (§3.1).
