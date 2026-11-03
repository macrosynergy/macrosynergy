# Performance + parity testing framework

Complements the main `tests/` suite for the T1–T5 optimization targets
(see `prompts/TARGETS.md`, `prompts/QUEUE.md`, and the design spec in
`docs/superpowers/specs/2026-06-30-macrosynergy-perf-framework-design.md`).

## Two halves

- **Parity / edge / API guards** — `tests/perf/test_parity_*.py` and additions in
  `tests/unit/...`. Run in the **default** `pytest` gate. They must pass on every
  `perf/<slug>` branch (encode current behaviour as the contract).
- **Benchmarks** — `tests/perf/test_perf_*.py`, marked `@pytest.mark.perf`,
  **deselected by default**. Measure speed (pytest-benchmark) and memory (`mem.py`).

## Run the default gate (includes parity, excludes benchmarks)

    pytest            # addopts already applies -m 'not perf'

## Run the benchmarks

Disable xdist and coverage (they skew/slow benchmarks):

    pytest tests/perf -m perf --benchmark-only -n0 --no-cov \
        --benchmark-json=tests/perf/results/bench_$(hostname).json

Enable opt-in RSS memory sampling:

    MACROSYN_PERF_RSS=1 pytest tests/perf -m perf -n0 --no-cov

## Scale tiers

`tests/perf/data.py::SCALE_TIERS` — `tiny` (~3k rows, parity/CI), `small` (~100k),
`medium` (~1M), `large` (~6M, local deep-dive). Benchmarks default to `small`+`medium`.

## Record before/after for QUEUE.md

1. On clean `feature/performance`, capture a per-machine baseline:
   `pytest tests/perf -m perf --benchmark-only -n0 --no-cov --benchmark-json=tests/perf/results/baseline_<host>.json`
2. On the `perf/<slug>` branch, repeat into `results/<slug>_<host>.json`.
3. `python tests/perf/record.py results/baseline_<host>.json results/<slug>_<host>.json`
   → paste the markdown into the QUEUE item's before/after. The recorder prints a
   `⚠ cross-machine — advisory only` banner if the two runs are from different hardware/OS.

## Regenerate parity goldens (deliberate only)

    python tests/perf/capture_parity.py --update   # then commit golden/index.json

## Environment fingerprint

Every benchmark JSON carries `machine_info.macrosynergy_env` (CPU/chip/RAM/OS/lib
versions/git SHA/CI label) so results are never silently compared across machines.
