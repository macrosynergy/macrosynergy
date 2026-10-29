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
subprocess tests, which share the `perf` marker and cost about twelve seconds each in
interpreter startup. Benchmarks are deselected by default, so plain `pytest` does not run
them.

The harness and report tests in `selftests/` run in the default gate:

```bash
python -m pytest tests/perf/selftests -m "not perf" -n0 --no-cov
```

The report adds the derived statistics to the JSON file in place and writes an `.html`
file beside it. The page is one report whether or not `--baseline` is given; a baseline
adds sections and columns rather than a second page. It is self-contained (inline SVG, a data-URI CSV,
a few lines of script to open dialogs) and titled with the measured code as
`v1.8.1.dev+f0ca6970` (release, `.dev` for a development build, the first eight digits of
the commit, `.dirty` when the package had uncommitted changes). In order:

- a header with the version, the baseline's version, when the run was recorded, and a
  "Download results as CSV" link;
- **Regressions** and **Improvements** (with a baseline): each changed case with its
  baseline and current mean, change, 95% interval and p-value, worst first;
- **Methods**: one tile per benchmarked method, naming the callable and what it measures,
  with a chart switched page-wide between time, throughput, peak memory and (with a
  baseline) change by tier, and a table of its cases by variant and tier;
- **Fastest** and **Slowest**: cases ranked by observations per second at their largest
  tier;
- **Environment**: both runs' provenance, differing values in bold.

Any case in any table opens a dialog with its timing histogram, round-order plot and,
with a baseline, the bootstrap distribution of its change, then its statistics beside the
baseline's and every argument it was called with.

## CI and storage

The `Performance benchmarks` workflow is run manually, with `tiers` and an optional
`baseline_ref` (a tag, commit or branch). It:

1. runs the benchmarks and adds a `ci` block (run URL, ref, SHA, tiers, baseline ref) to
   the JSON;
2. stores the JSON in the R2 bucket as `{hardware_id}-{commit_sha}-{run_id}.json`, where
   `run_id` is the run's UTC start time (`20261029T145212Z`);
3. fetches a baseline from the bucket: the newest other run on the same `hardware_id`
   for `baseline_ref`'s commit or, without it, for the newest benchmarked ancestor of the
   measured commit;
4. renders the page, with the baseline when one was found, and uploads the JSON and the
   page as the `perf-benchmark-report` artifact;
5. fails if `baseline_ref` was given and no run was found for it. The results are stored
   and uploaded either way.

The bucket holds JSON only; a page is rebuilt from it with `report.page`. The lookup and
upload are in `.github/scripts/perf_bucket.sh`, which needs `PERF_BENCH_BUCKET` and the
`AWS_*` variables. The workflow takes them from the `PERF_BENCH_R2_ACCESS_KEY_ID`,
`PERF_BENCH_R2_SECRET_ACCESS_KEY`, `PERF_BENCH_R2_ENDPOINT` and `PERF_BENCH_R2_BUCKET`
repository secrets.

## Tiers

Tiers come from `--perf-tiers`, then `MACROSYN_PERF_TIERS`, then `small,medium`.

| tier | observations | cids x xcats | dates | rounds | budget |
| --- | ---: | --- | ---: | ---: | ---: |
| tiny | 3,000 | 4 x 3 | 250 | 25 | 1.0 s |
| small | 104,000 | 10 x 8 | 1,300 | 15 | 3.0 s |
| medium | 1,050,000 | 20 x 15 | 3,500 | 8 | 10.0 s |
| large | 6,240,000 | 40 x 30 | 5,200 | 5 | 30.0 s |

Observations are `ticker_count * date_count`, which equals row count for the long format.
Both formats use the same ticker and date counts at a tier, so a wide DataFrame holds the
same observations as the long one, transposed. Observations are therefore the one number
every benchmark is comparable on, whichever format it consumes.

`rounds` and `budget` ride on each `pytest.param` as
`pytest.mark.benchmark(min_rounds, max_time)`, so a benchmark author writes an ordinary
`@parametrize` and gets the right budget for every tier. The full suite at `tiny` takes
about 18 s.

Dates come from `pd.bdate_range(start, periods=date_count)` after checking the package
still maps the daily frequency to `"B"`, because a tier label has to be exact. Both
builders assert the DataFrame they produced against the size that asked for it, so a
recorded size can never disagree with the data behind it.

## Add a benchmark

One function in the `benchmarks/` module named after the package module it measures,
`test_df_utils.py` for `macrosynergy.management.utils.df_utils`. A new module starts with
a `pytestmark` naming its group after itself:

```python
pytestmark = [pytest.mark.perf, pytest.mark.perf_group("df_utils")]
```

```python
@pytest.mark.parametrize("panel_size", QDF_SIZES, ids=str)
def test_my_target(benchmark, panel_size):
    """Say in one line what this measures, in which mode."""
    measure(benchmark, my_target, panel_size.as_qdf())
```

The docstring's first line is the case's label in the report, and a `perf`-marked
benchmark without one fails. `measure` records the target as `module:qualname` and every
argument bound to its parameter name, described by shape for data (`DataFrame
104,000×4, categorical`) and by value otherwise, so the report shows the call that was
timed. It then records peak memory from one untimed call and benchmarks the same call.
When the target needs preparing, such as a class to construct, record that call too:

```python
benchmark.extra_info["setup"] = {"MyClass": describe_call(MyClass, (df,), options)}
```

A function with several modes is benchmarked once per mode by parametrizing the mode,
which then shows as the case's variant (`dtype=categorical`). For a
target that modifies its input use `measure_mutating`, which takes a callable returning
`(args, kwargs)` so rebuilding the input each round stays outside the measured region:

```python
@pytest.mark.parametrize("panel_size", QDF_SIZES, ids=str)
def test_my_mutating_target(benchmark, panel_size):
    def make_arguments():
        return (panel_size.as_qdf_copy(),), {}

    measure_mutating(benchmark, panel_size, my_target, make_arguments)
```

`measure_mutating` uses `benchmark.pedantic`, which disables calibration and auto-ranging
and pins `iterations` to 1, so it is reserved for the mutating case.

Use `TICKER_DF_SIZES` and `panel_size.as_ticker_df()` for a target that consumes the wide
format. Cap a slow benchmark to the tiers it can finish by naming them:

```python
@pytest.mark.parametrize("panel_size", PANEL_SIZES.qdf_sizes("tiny", "small"), ids=str)
```

A cap sharing no tier with the active selection produces one skip naming both sides, not
a silent gap. `perf_group` is required; a `perf`-marked benchmark without it fails.

## Output

The file is a `pytest-benchmark` payload with additions under one `macrosynergy` key, so
`pytest-benchmark compare` still reads it. `report` only adds keys; it never renames,
reshapes or drops anything the plugin wrote.

```json
{
  "machine_info": {"macrosynergy": {
      "cpu_model": "AMD EPYC Processor (with IBPB)", "usable_cpu_count": 12,
      "total_ram_gb": 47, "operating_system": "Linux 6.8.0-136-generic",
      "python_version": "3.12.3",
      "library_versions": {"numpy": "2.5.1", "pandas": "2.3.3", "scipy": "1.18.0"},
      "blas_name": "scipy-openblas", "blas_thread_count": 12,
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

Times are seconds, memory is bytes, and every unit is in the name because these numbers
outlive the function that computed them. Each timing statistic carries `_low` and `_high`
companions holding percentile-bootstrap bounds, reproducible from the raw samples and the
recorded seed. The plugin supplies every other statistic, including outlier
classification, and those are read rather than recomputed.

`measurement_id` is built from `extra_info`, never from a pytest node id, so renaming a
file or reordering decorators does not orphan a benchmark's history. Raw samples stay in
`stats.data`, so no chart needs a rerun. `workload_id` hashes the `panel_size` block,
variant, target, call and setup: retuning a tier or changing an argument keeps
`measurement_id` but moves `workload_id`, so history does not silently join two different
workloads. Runs recorded before targets and calls were (schema version 1) still load and
render, without labels.

`machine_info.macrosynergy` records the fields in which a difference can change a timing
by a multiple, and identifies a run along separate axes because each answers a different
question:

| Field | Question | Built from | Comparison |
| --- | --- | --- | --- |
| `hardware_id` | same machine? | CPU model, usable CPUs, RAM | must match |
| `software_id` | same runtime? | OS and kernel, Python, BLAS and its threads, numpy, pandas, scipy | may differ; say which moved |
| `package_version` | which release? | `macrosynergy.__version__` | expected to differ |
| `commit_sha`, `commit_time` | which code? | the checked-out commit | expected to differ |
| `package_diff_hash` | uncommitted edits? | `git diff HEAD` plus untracked files, under `macrosynergy/` only | `null` when the package matches the commit |
| `workload_id` (per case) | same work? | `panel_size` block, variant, target, call, setup | must match |
| `run_id` | which run? | the payload's timestamp | always unique |

The package version is kept out of `library_versions` and so out of `software_id`,
because it is the code being measured. The hash inside `package_version` is fixed at
install time, so the page labels a run with `commit_sha` instead.
`usable_cpu_count` is the count available to the process (`len(os.sched_getaffinity(0))`),
not the count the host has, because BLAS sizes its thread pool from availability and that
is what enters a timing. Every collector returns `None` rather than raising, so a missing
`/proc` or git costs one field instead of the run. `total_ram_gb` is rounded to whole
gibibytes so that machines of the same size, such as two hosted runners, share a
`hardware_id`.

## Files

| File | Contents |
| --- | --- |
| `benchmarks/` | the benchmarks, one module per package module they measure |
| `harness/panel_sizes.py` | `PanelSize`, `PanelSizeCatalog`, `PANEL_SIZES`, `TARGET_OBSERVATION_COUNTS`, the cached builders and `clear_df_cache` |
| `harness/measurement.py` | `measure`, `measure_mutating`, `describe_call`, `PeakMemoryTracker` |
| `harness/machine.py` | `MachineProfile` |
| `report/results.py` | `ConfidenceInterval`, `BenchmarkMeasurement`, `BenchmarkRunResults` |
| `report/page.py` | enriches a results file and renders its HTML page |
| `conftest.py` | tier selection, host stamping, `extra_info` stamping |
| `selftests/test_harness.py` | tests for `harness/` and `conftest.py`, in the default gate |
| `selftests/test_report.py` | tests for `report/`, in the default gate |

## Data flow

```text
  PANEL_SIZES["small"] -> PanelSize
                            |
                            +-- as_qdf()        long DataFrame
                            +-- as_ticker_df()  wide DataFrame
                                          |
                                          v
                              pytest-benchmark measures the target
                                          |
  MachineProfile.capture() --+            |
  conftest records extra_info +-----------+
                                          v
                                  pytest-benchmark JSON
                                          |
                            BenchmarkRunResults.load(path)
                                          |
                    BenchmarkMeasurement + ConfidenceInterval per case
                                          v
                    BenchmarkRunResults.with_derived_statistics()
                                          |
                                    enriched JSON
                                          |
                                 report.render_page()
                                          |
                                     HTML page
```

`BenchmarkRunResults.compare_against(baseline)` decides each shared case as Criterion.rs
does. `mean_change` gives the relative change in mean timing with bootstrap bounds, and a
bootstrap t-test p-value (both runs resampled from their pooled timings). The verdict is
`regressed` or `improved` only when p < `SIGNIFICANCE_LEVEL` (0.05) and the whole
interval lies beyond ±`NOISE_THRESHOLD` (5%); a significant change reaching into that
band is `within noise`, anything else `no change`, and a case whose `workload_id` moved
is `workload changed`. Runs whose `hardware_id`s differ raise `ValueError`.

The noise threshold is per runner and should be calibrated by comparing repeated runs of
identical code until they read as no change. On a shared host that does not work: two
runs of the same commit here differed by 15-70% on six cases, with p near zero, because
the host's load moved between runs. Within-run statistics cannot see that, so trustworthy
verdicts need a dedicated runner.

## Components

### `PanelSize`

Defines one data size and builds the DataFrame for it. Frozen and hashable, which is what
lets it serve as the `lru_cache` key and the target of `dataclasses.replace`.

Tickers are given **either** as an explicit tuple **or** as `cid_count` and `xcat_count`,
never both, matching the contract of `make_test_df`. `__post_init__` rejects
both-or-neither, a lone half, duplicate tickers and a non-positive `date_count`.

Counts are derived, never stored, so a size cannot contradict itself: `ticker_count`,
`observation_count`, `cids`, `xcats`, `qdf_shape`, `ticker_df_shape`, and `shape` and
`df_format` for whichever of the two formats the size was selected for.

`as_qdf(categorical=False)` returns the shared long DataFrame, `as_ticker_df()` the
shared wide one, and `as_qdf_copy(categorical=False)` an independent frame for targets
that modify what they are given. `describe()` returns the `panel_size` block recorded
with a measurement.

### `PanelSizeCatalog` / `PANEL_SIZES`

Holds the named sizes and decides which of them a run measures. `PANEL_SIZES[tier]`
returns one size; `tier_names` and `selected_tiers` are plain attributes.

`select_tiers("tiny,small")` sets the measured tiers and raises naming any unknown one.
It is called once, from `pytest_configure`. `qdf_sizes(*only_tiers)` and
`ticker_df_sizes(*only_tiers)` return one `pytest.param` per measured tier, or one
explained skip when `only_tiers` and the selection share nothing.

### `MachineProfile`

`MachineProfile.capture()` reads the fields described under **Output**; `describe()`
serialises them with `hardware_id` and `software_id`, which are also properties.

### `measure` / `measure_mutating` / `PeakMemoryTracker`

`measure` and `measure_mutating` record the target and its call with `record_call`, take
peak memory from one untimed call, then hand the target to pytest-benchmark.
`describe_call` binds each argument to its parameter name and `describe_argument`
describes it by shape for data and by value otherwise. `PeakMemoryTracker` is a context manager over `tracemalloc` exposing `peak_bytes`. It is
always used outside the measured region, because its overhead would distort whatever it
wrapped.

### `ConfidenceInterval` / `BenchmarkMeasurement` / `BenchmarkRunResults`

`ConfidenceInterval` carries a `value` with `low` and `high` bounds.
`from_samples(samples, statistic, seed)` builds one by percentile-bootstrap resampling; a
single sample gives a value with no bounds.

`BenchmarkMeasurement.from_payload_entry(entry)` reads one `benchmarks[]` entry and
raises naming the case when `extra_info` lacks `group`, `benchmark_name` or `panel_size`.
It exposes `identifier` plus `mean_seconds`, `median_seconds`, `stddev_seconds`,
`median_absolute_deviation_seconds` and `observations_per_second`. Flat throughput across
tiers means the target scales linearly.

`BenchmarkRunResults.load(path)` reads a run and rejects a repeated identifier.
`measurement_table()` is the single definition of the per-case statistics, one row per
measurement; `with_derived_statistics()` builds the payload's derived blocks from those
rows, so there is one place where a column is defined.

## Wiring

`tests/perf/conftest.py` declares `--perf-tiers`. Pytest reads options only from
conftests on the path named on the command line, so the flag needs `tests/perf` (or a
directory under it) in the command; `MACROSYN_PERF_TIERS` works with any path. It
resolves the tiers in `pytest_configure`, which runs before
collection and therefore before the catalogue's size methods are evaluated inside
`@parametrize`. `_MachineProfileStamper` writes the profile into `machine_info` and is
registered as a plugin object, only when the benchmark plugin is active: a module-level
`pytest_benchmark_*` function in a conftest makes pluggy abort collection of the whole
repository whenever that plugin is absent. `machine_profile()` captures on first use
rather than at startup, because capture costs a couple of seconds and `addopts` sets
`-n auto`. The autouse `_record_benchmark_metadata` fixture writes `group`,
`benchmark_name`, `label`, `variant`, `timing_mode` and the `panel_size` block onto
`extra_info` (the benchmark itself records `target`, `call` and `setup`, because only it
holds the call), the size block **after** the test body so it describes a DataFrame that
was actually built. `pytest_sessionfinish` releases every built frame.

`extra_info` is the only channel from a test to the results file. Nothing downstream
recovers identity by parsing a test name.

## Invariants

- Counts are derived from a `PanelSize`, and both builders assert the frame matches them.
- DataFrame construction never happens inside a measured region.
- `extra_info` is the only test-to-file channel.
- `with_derived_statistics` adds keys and never removes or rewrites one.
- A measurement identifier comes from metadata, not from a pytest node id.
- Every benchmark has a docstring label and records the call it times.
- Timings are seconds, memory is bytes, and the unit is in the name.
- Benchmarks, expensive frame builds and subprocess tests are all `perf`-marked, so the
  default gate stays around a second.

## Conventions

Tests here follow the repository's `unittest.TestCase` plus `parameterized.expand` style,
with two documented exceptions. The `benchmarks/` modules are pytest-native function
style because pytest refuses to inject fixtures into `TestCase` methods and every
benchmark depends on the `benchmark` fixture. For the same reason, the two conftest
tests in `test_harness.py` that need fixtures are bare pytest functions beside its
`TestCase` classes. Note
also that a pytest mark does not survive `parameterized.expand`, so a mark that must
apply to generated cases goes on the class.

## Known failures

Two repo-wide collection errors come from `linearmodels` being declared as a test
dependency but not installed. They are unrelated to this suite.
