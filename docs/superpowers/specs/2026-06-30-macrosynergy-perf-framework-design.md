# macrosynergy performance-testing framework — design spec

> **Status:** design approved (brainstorming), pending written-spec review → implementation plan.
> **Branch:** `feature/performance`. This framework is a *prerequisite* deliverable, written
> **before** any `perf/<slug>` optimization sub-branch (T1–T5 in `prompts/TARGETS.md`).
> **No package code is optimized here** — this scopes the test/benchmark scaffolding that the
> optimizations will be measured and guarded against.

## 1. Purpose

Provide an in-repo framework that lets every optimization in `prompts/QUEUE.md` (T1–T5) be:

1. **Benchmarked for speed** (CPU/wall time) against a clear, reproducible target.
2. **Benchmarked for memory** (peak allocation / RSS) the same way.
3. **Guarded for correctness** — comprehensive unit tests so optimization cannot break the
   public API, signatures, or output of any target function.

It **complements** the existing `tests/` suite and the external academy harness; it does not
replace either. It directly feeds the **GATE** (`TARGETS.md` §5): API unchanged · output parity ·
measurable win · suite passes.

### Relationship to the external harness

The academy notebook (`Cyclical strength composite.ipynb`), its frozen parquet fixtures, and the
external `profile_cells.py` / `capture_golden.py` drivers live **outside this repo**. They own the
*macro* (whole-notebook, real-data, 24-cell) view. This in-repo framework owns the *micro*
(per-function, synthetic-data, reproducible) view. The two are deliberately decoupled: the in-repo
suite has **no dependency** on the external fixtures.

## 2. Approved decisions (brainstorming)

| Decision | Choice |
|---|---|
| Granularity | **Micro per-function** — benchmark the 6 target functions in isolation on controlled synthetic QDFs. |
| Memory metric | **tracemalloc peak (default, deterministic) + opt-in psutil RSS** (truer, noisy, local). |
| CI relationship | **Opt-in, record-only** — `@pytest.mark.perf`, deselected from the default gate; produces JSON artifacts recorded in `QUEUE.md`. |
| Parity data | **Synthetic, in-repo** — generated from `make_qdf` with fixed seeds; no external fixture dependency. |
| Environment | **Fingerprint every result** (CPU/chip/RAM/OS/lib versions/CI label) so cross-machine numbers are never silently compared. |

### Explicitly out of scope (YAGNI)

- No in-repo replay of the 24 notebook cells (external harness owns macro).
- No dependency on the external academy parquet fixtures.
- No CI perf-threshold gating (opt-in record-only chosen).
- No optimization of package code (that is the `perf/<slug>` branches).

## 3. Target functions and their QUEUE mapping

| QUEUE | Target | Function(s) | Perf module | Lead cells |
|---|---|---|---|---|
| Q1 | T2c | `_get_tickers_series`, `add_ticker_column`, `reduce_df_by_ticker` | `test_perf_qdf_ticker_series.py` | 18, 27 |
| Q2 | T1 | `update_df`, `update_tickers` | `test_perf_update_df.py` | 17, 18, 27, 15, 19, 13 |
| Q3 | T2 | `split_ticker`, `get_cid`, `get_xcat`, `ticker_df_to_qdf` | `test_perf_ticker_split.py` | 13 |
| Q4 | T3 | `reduce_df` | `test_perf_reduce_df.py` | 12, 19, 43, 47 |
| Q5 | T4 | `SignalReturnRelations` panel test / `map_pval` | `test_perf_srr_mixedlm.py` | 31, 33, 35, 39, 45, 50 |
| Q6 | T5 | `make_zn_scores` repeated `reduce_df` | (covered by Q4 module + a scenario test) | 19 |

## 4. Architecture — two halves on one data foundation

```
tests/
  perf/
    __init__.py
    conftest.py                     # perf marker; scale-tier selection; mem fixture;
                                    #   pytest_benchmark_update_machine_info hook (env stamp)
    data.py                         # scaled synthetic QDF builders (wrap make_qdf); object + categorical
    mem.py                          # measure(): tracemalloc peak (default) + opt-in psutil RSS
    env.py                          # environment_fingerprint(): CPU/chip/RAM/OS/libs/git/CI
    record.py                       # diff baseline<->branch result JSON -> QUEUE.md markdown; fingerprint guard
    capture_parity.py               # snapshot current-code output on seeded synthetic inputs -> golden parquet + hashes
    results/                        # bench/mem JSON; gitignored except committed per-machine baselines
    golden/                         # tiny/small parity snapshots (parquet) + index.json (hashes)
    README.md                       # how to run, record, and read results
    test_perf_update_df.py
    test_perf_reduce_df.py
    test_perf_ticker_split.py
    test_perf_qdf_ticker_series.py
    test_perf_srr_mixedlm.py
  unit/management/                  # EXTENDED with parity + gap-filling edge tests (default gate)
  unit/signal/                      # EXTENDED with map_pval + SRR parity tests
```

**Half A — perf suite** (`tests/perf/test_perf_*.py`): opt-in, record-only, measures speed + memory.
**Half B — parity + API-guard tests**: live in the existing `tests/unit/...` files, run in the
**default** gate, and must pass on every `perf/<slug>` branch.

### 4.1 Component — Synthetic data (`tests/perf/data.py`)

Built on `tests/simulate.py::make_qdf` (seeded; produces **object-dtype** QDFs — the notebook's
exact slow case). Parametrized builders produce QDFs at controlled scale via `(n_cids, n_xcats,
n_days)`, exposed as named **scale tiers**:

| Tier | ~rows | Use |
|---|--:|---|
| `tiny` | ~10 k | parity/edge unit tests (default gate, fast) |
| `small` | ~100 k | quick benchmark |
| `medium` | ~1 M | representative benchmark (CI-feasible) |
| `large` | ~5–10 M | local deep-dive only, opt-in (mirrors notebook pathology) |

Each builder yields **object** and **categorical** variants (the optimization boundary). Three
**pathological-pattern** helpers reproduce the notebook's hot loops at reduced scale:

- **wide ticker frame** for `ticker_df_to_qdf` (`stack()` of a wide unique-ticker frame) — T2/Q3.
- **growing-loop `df_add` sequence** for `update_df` called repeatedly on a growing frame
  (the O(k·n·log n) re-sort/re-dedup pattern) — T1/Q2.
- **signal×return panel** for the MixedLM panel test (e.g. up to 5 signals × 7 returns) — T4/Q5.

All builders are fixed-seed and deterministic — this is the "clear and reproducible target."

### 4.2 Component — Memory measurement (`tests/perf/mem.py`)

`measure(fn, *args, **kwargs)` / a context manager returning
`(wall_s, tracemalloc_peak_mib, rss_peak_mib | None)`:

- **tracemalloc peak** — deterministic, default-recorded metric (Python-heap; undercounts numpy,
  but stable and CI-comparable).
- **psutil RSS delta** — opt-in via `MACROSYN_PERF_RSS=1`, sampled (~20 ms), guarded import
  (psutil is optional). Truer to real memory, but noisy/machine-dependent.
- The `large` tier **skips tracemalloc** (pathologically slow on alloc-heavy paths, per TARGETS
  cell-13 note) and records RSS only.

Memory results are written as plain JSON (not pytest-benchmark format) stamped with the
environment fingerprint (§4.5).

### 4.3 Component — Speed benchmarks (`pytest-benchmark`)

Each target has benchmark tests parametrized over **(scale tier × dtype)**. `pytest-benchmark`
gives multi-round stable wall times and supports `--benchmark-compare`/`--benchmark-histogram`.

- Marked `@pytest.mark.perf`; **deselected from the default run** via
  `addopts = "... -m 'not perf'"`, so the standard `pytest` gate never runs them.
- Run explicitly: `pytest tests/perf -m perf --benchmark-only --benchmark-json=<results-file>`.
- `record.py` turns the JSON into the `QUEUE.md` before/after table rows.

### 4.4 Component — Parity + API guard (default gate; the "comprehensive coverage" half)

For **each** target, three kinds of test live in the existing `tests/unit/...` files:

**(a) Output-parity goldens.** `capture_parity.py` snapshots current-code output on seeded
`tiny`/`small` inputs to `golden/*.parquet` + `golden/index.json` (content hashes). Tests assert
**byte-identical** output:
- QDF/DataFrame → `QuantamentalDataFrame`/value equality + `np.allclose` (NaN-equal) on numerics,
  exact on object cells, identical column order/dtype.
- `Categorical` outputs (`_get_tickers_series`) → identical **category set AND order** plus codes.
- Lists (`split_ticker` iterable) → identical list, identical length, identical validation errors.

Plus **invariant assertions** that pin the contract independent of the snapshot, e.g.:
- `update_df`: result = last-wins dedup on `[real_date, xcat, cid]`, sorted by
  `IDX_COLS_SORT_ORDER`, original dtype preserved, input not mutated.
- `reduce_df`: returned rows/order/dtype identical to pre-change; no spurious row drops.

**(b) Edge/contract tests** filling the gaps found in the coverage audit:
- **`split_ticker` direct tests** (currently untested) — scalar, iterable, malformed, empty,
  whitespace, multi-underscore, `mode` validation.
- **object vs categorical dtype** for *every* target (categorical paths are under-tested and are
  exactly what T2c/T1 touch).
- empty df, single row, NaN in `cid`/`xcat`/`value`, duplicate index rows, out-of-order input,
  sort stability, return-dtype preservation.
- **`map_pval` direct test** (currently only constructor-level SRR coverage) — p-value value +
  3-dp rounding, convergence-warning path.

**(c) API-signature tripwire.** One `inspect.signature` assertion per target (param names,
defaults, return annotation). Cheap, direct guard for GATE criterion 1 (API unchanged).

These run in the default gate and are fast (`tiny` scale) — they gate every `perf/<slug>` branch.

### 4.5 Component — Environment fingerprint (`tests/perf/env.py`)

`environment_fingerprint()` returns a dict stamped onto **every** result file so cross-machine
numbers are never silently compared:

- **CPU/chip:** brand string (e.g. "Apple M2", "AMD EPYC 7763"), arch (`platform.machine()`),
  physical + logical core count, base frequency.
- **Memory:** total RAM.
- **Platform:** OS, release, Python version/impl.
- **Library versions:** numpy, pandas, statsmodels, scipy, pyarrow, + `macrosynergy` version &
  git SHA.
- **Context:** hostname, CI detection (`GITHUB_ACTIONS` / `RUNNER_NAME`) so CI runs are labelled.
- **Timestamp.**

Mechanics:
- `pytest-benchmark` already captures `machine_info` (CPU via its `py-cpuinfo` dep — no new
  explicit dep for CPU info). Extended through the official
  `pytest_benchmark_update_machine_info(config, machine_info)` hook in `conftest.py` to add RAM,
  git SHA, library versions, and CI label.
- The same fingerprint is stamped into `mem.py` JSON results.
- **Result filenames key on a short fingerprint hash** (e.g. `bench_<host>_<cpuhash>.json`) so one
  machine's baseline never overwrites another's.
- **`record.py` compares fingerprints before diffing:** if CPU brand / core count / OS differ, it
  prints a prominent `⚠ cross-machine — advisory only` banner and refuses a pass/fail verdict
  (shows both numbers side-by-side). Same-machine diffs get the real before/after verdict for the
  QUEUE table.
- RAM + physical-core count use psutil when present, falling back to `os`/`platform` otherwise.

### 4.6 Component — Recording (`tests/perf/record.py`)

Convenience to convert result JSON into the `QUEUE.md` before/after markdown rows, applying the
§4.5 fingerprint guard. Workflow per `perf/<slug>` branch:

1. (once, on clean `feature/performance`) capture per-machine baseline + parity golden.
2. branch `perf/<slug>` → implement → `python tests/perf/capture_parity.py` re-capture + assert.
3. `pytest tests/perf -m perf --benchmark-only --benchmark-json=results/<branch>.json`.
4. `python tests/perf/record.py results/<baseline>.json results/<branch>.json` → paste into QUEUE.
5. `pytest` (default gate, incl. parity + API-guard) in `~/repos/macrosynergy`.

## 5. Dependencies & policy

- **Add `pytest-benchmark`** to `[project.optional-dependencies].test` in `pyproject.toml`
  (pulls `py-cpuinfo` transitively → CPU fingerprint).
- **psutil** — optional, guarded import (RAM + physical cores + RSS); not a hard dependency.

> **Sonatype vetting — waived for this change (user decision, 2026-06-30).** These are dev/test-only
> dependencies; the user opted to skip Sonatype MCP vetting for them and pin to latest stable
> (e.g. `pytest-benchmark>=4.0`). The global Sonatype-first policy still applies to any *runtime*
> dependency added to the package.

## 6. Success criteria for this framework (not the optimizations)

1. Every T1–T5 target has: a speed benchmark, a memory measurement, parity goldens + invariants,
   edge/contract tests, and an API-signature tripwire.
2. The coverage gaps from the audit are closed — notably **`split_ticker` direct tests**,
   **categorical-dtype paths** on all targets, and a **direct `map_pval` test**.
3. Benchmarks are reproducible: same seed + same tier + same machine → stable numbers; results
   carry an environment fingerprint and are not silently compared across machines.
4. The default `pytest` gate is unchanged in speed (perf suite deselected by marker) and gains the
   new parity/edge/API tests.
5. `record.py` produces QUEUE-ready before/after rows.

## 7. Open items / risks

- **Scale-tier row counts** (`tiny`/`small`/`medium`/`large`) need calibration to wall-time
  budgets on a reference machine during implementation — the ~row figures above are starting
  targets, not final.
- **`large` tier** may be too slow/heavy for some machines → gated behind opt-in env flag, RSS-only.
- **Golden snapshot churn:** parity goldens are committed; regenerating them must be a deliberate,
  reviewed action (a `--update` flag on `capture_parity.py`), never automatic.
