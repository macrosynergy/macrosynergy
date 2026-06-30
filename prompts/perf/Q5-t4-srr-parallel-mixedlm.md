# Q5 · T4 · `perf/srr-parallel-mixedlm` — builder brief  (HIGHER RISK)

> Item: **Q5** (QUEUE.md) · Target: **T4** (TARGETS.md §T4, rank 5) · depends-on: **none**
> Branch/worktree: `perf/srr-parallel-mixedlm` · Base: `feature/performance`

## Goal

Parallelize the **independent** MixedLM panel-test fits in `SignalReturnRelations` (serial default,
identical table output), and fold in the `map_pval` double-`summary()` dedup. The SRR MixedLM panel
test is ~**749s / 25%** of total runtime across cells 31/33/35/39/45/50, each cell up to 35 fully
independent `(signal × return)` MixedLM fits. Control cell 37 (`ms_panel_test=False`) costs only 7.3s,
confirming the panel test is essentially the entire SRR cost.

Read the full design in **`prompts/TARGETS.md` §T4** and the parallelization principles in **§3.1**
(slice first, fan out the minimum). This is the **highest-risk** item — concurrency in a core
analysis class. **A serial default and byte-/tolerance-identical table output are non-negotiable.**

## Files

- **Modify:** `macrosynergy/signal/signal_return_relations.py` — the `ms_panel_test` path:
  `map_pval` (931) wrapping `statsmodels.MixedLM(...).fit()`, called per `(sig, ret)` segment from
  `single_statistic_table` (1495). The double `summary()` is at ≈ lines 967 and 972.
- Do **not** change the public API of `SignalReturnRelations`, `single_statistic_table`, or
  `map_pval` in a way that alters existing behaviour. Do **not** touch `tests/perf/golden/`.

## Design (output identical; serial default)

- **Slice first, fan out the minimum (§3.1).** Parallelize at the per-`(sig, ret)` boundary.
  `map_pval` already receives pre-extracted `ret_vals`/`sig_vals`, so dispatch the small
  `(y, X, groups)` arrays to workers — **never** the shared `dfx`. Tiny payloads mean **either** a
  thread pool (`MixedLM.fit` is BLAS/`linalg.solve` and releases the GIL → no extra memory, preferred
  given these cells already peak ~7.4 GiB) **or** a process pool works.
- **Serial by default.** Expose the worker count via an **opt-in** param (default 1 / current
  behaviour) — do not change results for existing callers. Threads vs processes: prefer threads.
- **Determinism.** The assembled table rows/columns and every numeric cell must be identical
  regardless of worker count or completion order — collect results back into the original
  `(sig, ret)` order, not arrival order. Handle the convergence-warning / fallback-optimizer
  (lbfgs/cg) paths exactly as today.
- **`summary()` dedup (cheap, fold in).** `map_pval` builds the statsmodels `summary()` **twice**
  (≈967, 972) to parse one p-value — call it once / read `re.pvalues` reproducing the existing
  3-dp rounding. Small, free, low-risk; keep the parsed value bit-identical.

## GATE (verify ALL before hand-back; `--no-cov -n0`, never `-p no:cov`)

> **Parity note:** there is no `tests/perf/test_parity_srr_*` golden — SRR parity is guarded by the
> macro 24-cell harness (numeric-tolerance table golden) **plus** the SRR unit suite and the
> `MapPvalDirect` direct tests. Run all of them.

1. **Parity + behaviour preserved (GATE-1/2) — must stay GREEN:**
   ```bash
   pytest tests/unit/signal/test_signal_return_relations.py -k "MapPvalDirect" -v --no-cov -n0   # 2 passed
   pytest tests/unit/signal/test_signal_return_relations.py --no-cov -n0                          # full SRR suite green
   ```
   Confirm the worker-count param defaults to serial: running with default settings produces the
   identical table as before (the suite encodes this).
2. **Measurable win (GATE-3):**
   ```bash
   pytest tests/perf/test_perf_srr_mixedlm.py -m perf -k "2-3" --benchmark-only -n0 --no-cov \
     --benchmark-json=<scratch>/after.json
   python tests/perf/record.py <baseline-json> <scratch>/after.json
   ```
   The win shows when parallelism is enabled — `test_bench_srr_single_statistic_table[2-3]`
   (multiple independent fits) is the meaningful case. If the benchmark runs the serial default,
   add a parallel-enabled variant **only** if it doesn't change the public benchmark contract;
   otherwise demonstrate the speedup via the `[2-3]` case with the param set and **report the
   measurement** to the manager. The `summary()` dedup also gives a small serial win.
3. **macrosynergy suite (GATE-4):**
   ```bash
   pytest tests/unit/signal --no-cov -n0
   ```
   (The manager runs the full suite as the merge gate; the macro harness re-capture is the parity
   confirmation for the SRR table.)
4. **Hygiene:** `git status` shows only `signal_return_relations.py` changed; no `tests/perf/golden/*`
   modified; no scratch files.

## Acceptance criteria

- [ ] Per-`(sig, ret)` fits are parallelizable via an **opt-in** worker param; **default is serial**
  and bit/tolerance-identical to current output.
- [ ] Workers receive minimal `(y, X, groups)` arrays, not the shared `dfx` (§3.1).
- [ ] Result assembly is order-deterministic (original `(sig, ret)` order), independent of worker
  count or completion order; convergence-warning/fallback paths preserved.
- [ ] `map_pval` builds `summary()` once, not twice; parsed p-value bit-identical (3-dp rounding).
- [ ] Public API of `SignalReturnRelations`/`single_statistic_table`/`map_pval` unchanged
  (`MapPvalDirect` incl. `test_map_pval_signature_unchanged` passes — 2 passed).
- [ ] Full `tests/unit/signal` suite green; benchmark shows a parallel win; no regression serial.

## Notes

- If concurrency proves too invasive to keep output identical, the contained fallback (cap/skip the
  multi-optimizer retry cascade) **risks changing p-values and is NOT preferred** — report back to
  the manager rather than shipping a behaviour change.
- Do not parallelize anything outside the per-`(sig, ret)` fit boundary (no whole-`dfx` fan-out).
