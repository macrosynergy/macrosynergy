# Performance suite

Benchmarks package hot paths at four data sizes and writes one JSON file per run. It
measures only: correctness belongs to the package's unit tests under `tests/unit/`, which
a change made for speed must pass. Before optimising a function, check its unit tests
cover the behaviour you are about to touch, and add any missing ones there.

## Run

```bash
python -m pytest tests/perf -m perf -n0 --no-cov --benchmark-only \
    --perf-tiers=small,medium --benchmark-json=perf-results.json

python -m tests.perf.report.page perf-results.json
python -m tests.perf.report.page perf-results.json --baseline old.json  # compare
```

`-n0 --no-cov` is required: the `benchmark` fixture is disabled under xdist, and coverage
instrumentation distorts timings. `--benchmark-only` skips the framework's own
subprocess tests, which share the `perf` marker. Benchmarks are deselected by default, so
plain `pytest` does not run them.

The report adds derived statistics to the JSON file and writes a self-contained `.html`
page beside it. With `--baseline`, it also lists the cases that regressed or improved.

The harness and report tests run in the default gate:

```bash
python -m pytest tests/perf/selftests -m "not perf" -n0 --no-cov
```

## CI

The `Performance benchmarks` workflow is run manually, with `tiers` (all four by default)
and an optional `baseline_ref` (a tag, commit or branch). It runs the benchmarks on a
dedicated runner, so runs are comparable, and:

1. stores the results;
2. compares them with the newest earlier run on the same hardware, for `baseline_ref`'s
   commit or, without it, for the newest benchmarked ancestor of the measured commit;
3. uploads the JSON and the page as the `perf-benchmark-report` artifact;
4. fails if `baseline_ref` was given and no run was found for it.

## Tiers

Tiers come from `--perf-tiers`, then `MACROSYN_PERF_TIERS`, then `small,medium`.

| tier | observations | cids x xcats | dates | rounds | budget |
| --- | ---: | --- | ---: | ---: | ---: |
| tiny | 3,000 | 4 x 3 | 250 | 25 | 1.0 s |
| small | 104,000 | 10 x 8 | 1,300 | 15 | 3.0 s |
| medium | 1,050,000 | 20 x 15 | 3,500 | 8 | 10.0 s |
| large | 6,240,000 | 40 x 30 | 5,200 | 5 | 30.0 s |

Observations are `ticker_count * date_count`, the same for the long and wide formats, so
every benchmark is comparable on them.

## Add a benchmark

One function in the `benchmarks/` module named after the package module it measures,
`test_df_utils.py` for `macrosynergy.management.utils.df_utils`. Benchmarks are plain
pytest functions, not `unittest.TestCase` classes like the rest of the tests, because they
need the `benchmark` fixture. A new module starts with a `pytestmark` naming its group
after itself, and the sizes it measures:

```python
import pytest

from tests.perf.harness.measurement import measure, measure_mutating
from tests.perf.harness.panel_sizes import PANEL_SIZES

pytestmark = [pytest.mark.perf, pytest.mark.perf_group("df_utils")]

QDF_SIZES = PANEL_SIZES.qdf_sizes()
TICKER_DF_SIZES = PANEL_SIZES.ticker_df_sizes()
```

```python
@pytest.mark.parametrize("panel_size", QDF_SIZES, ids=str)
def test_my_target(benchmark, panel_size):
    """Say in one line what this measures, in which mode."""
    measure(benchmark, my_target, panel_size.as_qdf())
```

The docstring's first line is the case's label in the report; a `perf`-marked benchmark
without one, or without `perf_group`, fails. When the target needs preparing, such as a
class to construct, record that call too:

```python
benchmark.extra_info["setup"] = {"MyClass": describe_call(MyClass, (df,), options)}
```

`as_qdf()` and `as_ticker_df()` return one cached frame shared by every benchmark, and it
is built outside the measured region. Never modify it. A function with several modes is
benchmarked once per mode by parametrizing the mode. For a target that modifies its input
use `measure_mutating`, which takes a callable returning `(args, kwargs)` so rebuilding the
input each round stays outside the measured region:

```python
@pytest.mark.parametrize("panel_size", QDF_SIZES, ids=str)
def test_my_mutating_target(benchmark, panel_size):
    def make_arguments():
        return (panel_size.as_qdf_copy(),), {}

    measure_mutating(benchmark, panel_size, my_target, make_arguments)
```

Use `TICKER_DF_SIZES` and `panel_size.as_ticker_df()` for a target that consumes the wide
format. Cap a slow benchmark to the tiers it can finish by naming them:

```python
@pytest.mark.parametrize("panel_size", PANEL_SIZES.qdf_sizes("tiny", "small"), ids=str)
```

To try a new benchmark on its own:

```bash
python -m pytest tests/perf -m perf -n0 --no-cov --benchmark-only --perf-tiers=tiny -k my_target
```

`--perf-tiers` is read only when `tests/perf` is in the command; otherwise set
`MACROSYN_PERF_TIERS`. CI picks up new benchmarks by itself; a new case has no baseline
until it has run once.

## Comparing runs

Runs compare only on the same `hardware_id` (CPU model, usable CPUs, RAM). A case is
`regressed` or `improved` only when p < 0.05 and the whole 95% interval of its change lies
beyond ±5%; a case whose workload changed is reported as such rather than compared.

## Output

The file is a `pytest-benchmark` payload with additions under one `macrosynergy` key, so
`pytest-benchmark compare` still reads it. The run leaves out the plugin's host name and
kernel release; after that, `report` only adds keys and never renames, reshapes or drops
anything the plugin wrote.

```json
{
  "machine_info": {"macrosynergy": {
      "cpu_model": "Intel(R) Core(TM) i7-1185G7 @ 3.00GHz", "usable_cpu_count": 8,
      "total_ram_gb": 16, "operating_system": "Linux",
      "python_version": "3.11.9",
      "library_versions": {"numpy": "2.5.1", "pandas": "2.3.3", "scipy": "1.18.0"},
      "blas_name": "scipy-openblas", "blas_thread_count": 8,
      "package_version": "1.8.1dev0+d21d6ab",
      "commit_sha": "f0ca69701ab081848e7c171b478cfa2432738b3f",
      "commit_time": "2026-10-22T17:15:06+01:00", "package_diff_hash": null,
      "hardware_id": "c09382f2", "software_id": "7b8302c6"}},

  "benchmarks": [{
    "stats": {"...plugin, including data[]..."},
    "extra_info": {
      "group": "qdf", "benchmark_name": "reduce_df",
      "label": "Filter a long panel to all of its own cids and xcats, so every row is kept.",
      "target": "macrosynergy.management.utils.df_utils:reduce_df",
      "call": {"df": "DataFrame 104,000×4", "xcats": "list[8]: XCAT000, XCAT001, XCAT002, …",
               "cids": "list[10]: AAA, AAB, AAC, …"},
      "variant": {"dtype": "categorical"}, "timing_mode": "calibrated",
      "peak_memory_bytes": 11831186,
      "panel_size": {"tier": "small", "df_format": "long", "ticker_count": 80,
                     "cid_count": 10, "xcat_count": 8, "date_count": 1300,
                     "metrics": ["value"], "observation_count": 104000,
                     "row_count": 104000, "column_count": 4}},
    "macrosynergy": {
      "measurement_id": "qdf/reduce_df/small/dtype=categorical",
      "group": "qdf", "benchmark_name": "reduce_df",
      "target": "macrosynergy.management.utils.df_utils:reduce_df",
      "label": "Filter a long panel to all of its own cids and xcats, so every row is kept.",
      "arguments": "df=DataFrame 104,000×4; xcats=list[8]: …; cids=list[10]: …", "tier": "small",
      "variant": "dtype=categorical", "timing_mode": "calibrated",
      "observation_count": 104000, "workload_id": "a3d4a53e",
      "mean_seconds": 0.0655, "mean_seconds_low": 0.0648, "mean_seconds_high": 0.0663,
      "median_seconds": 0.0654, "stddev_seconds": 0.0021,
      "median_absolute_deviation_seconds": 0.0009,
      "observations_per_second": 1585952.0, "peak_memory_bytes": 11831186}}],

  "macrosynergy": {"schema_version": 2, "run_id": "2026-07-31T...", "tiers_measured": ["small", "medium"],
                   "bootstrap": {"resamples": 10000, "seed": 12345, "confidence": 0.95}}
}
```

`extra_info` is written while the benchmark runs; the `macrosynergy` blocks are added by
`report`. Times are seconds, memory is bytes, and the unit is in every name. Each timing
statistic has `_low` and `_high` bootstrap bounds, reproducible from the raw samples in
`stats.data` and the recorded seed.

| Field | Question | Built from | Comparison |
| --- | --- | --- | --- |
| `hardware_id` | same machine? | CPU model, usable CPUs, RAM | must match |
| `software_id` | same runtime? | OS, Python, BLAS and its threads, numpy, pandas, scipy | may differ; say which moved |
| `package_version` | which release? | `macrosynergy.__version__` | expected to differ |
| `commit_sha`, `commit_time` | which code? | the checked-out commit | expected to differ |
| `package_diff_hash` | uncommitted edits? | `git diff HEAD` plus untracked files, under `macrosynergy/` only | `null` when the package matches the commit |
| `measurement_id` (per case) | same case? | group, benchmark name, tier, variant | joins history |
| `workload_id` (per case) | same work? | `panel_size` block, variant, target, call, setup | must match |
| `run_id` | which run? | the payload's timestamp | always unique |

`measurement_id` never comes from a pytest node id, so renaming a file or reordering
decorators keeps a benchmark's history. Changing a tier or an argument keeps
`measurement_id` but moves `workload_id`, so two different workloads are never compared.

## Files

| File | Contents |
| --- | --- |
| `benchmarks/` | the benchmarks, one module per package module they measure |
| `harness/panel_sizes.py` | the tiers and the DataFrame builders |
| `harness/measurement.py` | `measure`, `measure_mutating`, `describe_call` |
| `harness/machine.py` | the machine profile recorded with each run |
| `report/` | statistics, comparison and the HTML page |
| `conftest.py` | tier selection and per-case metadata |
| `selftests/` | tests for the harness and the report, in the default gate |
