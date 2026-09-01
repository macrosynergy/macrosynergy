# Performance suite

Benchmarks package hot paths at four data sizes and writes one JSON file per run. A
second, smaller group of tests checks that the same package functions stay
self-consistent, so a change made for speed cannot quietly change an answer.

## Run

```bash
python -m pytest tests/perf -m perf -n0 --no-cov --benchmark-only \
    --perf-tiers=small,medium --benchmark-json=tests/perf/results/run.json

python -m tests.perf.analyse tests/perf/results/run.json
```

`-n0 --no-cov` is required: the `benchmark` fixture is disabled under xdist, and coverage
instrumentation distorts timings. `--benchmark-only` skips the framework's own
subprocess tests, which share the `perf` marker and cost about twelve seconds each in
interpreter startup. Benchmarks are deselected by default, so plain `pytest` does not run
them.

The framework and parity tests run in the default gate:

```bash
python -m pytest tests/perf -m "not perf" -n0 --no-cov
```

`analyse` rewrites the file in place; `-o PATH` writes elsewhere and
`--drop-round-timings` omits the per-round samples.

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

One function in `test_benchmarks.py`:

```python
@pytest.mark.perf_group("qdf")
@pytest.mark.parametrize("panel_size", QDF_SIZES, ids=str)
def test_my_target(benchmark, panel_size):
    measure(benchmark, my_target, panel_size.as_qdf())
```

`measure` records peak memory from one untimed call, then benchmarks the same call. For a
target that modifies its input use `measure_mutating`, which takes a callable returning
`(args, kwargs)` so rebuilding the input each round stays outside the measured region:

```python
@pytest.mark.perf_group("qdf")
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
`pytest-benchmark compare` still reads it. `analyse` only adds keys; it never renames,
reshapes or drops anything the plugin wrote.

```json
{
  "machine_info": {"macrosynergy": {
      "cpu_model": "AMD EPYC Processor (with IBPB)", "usable_cpu_count": 12,
      "total_ram_gb": 47.0, "operating_system": "Linux 6.8.0-136-generic",
      "python_version": "3.12.3",
      "library_versions": {"numpy": "2.5.1", "pandas": "2.3.3", "macrosynergy": "1.8.1"},
      "blas_name": "scipy-openblas", "blas_thread_count": 12,
      "git_commit": "0762f008", "fingerprint": "4a20fffd"}},

  "benchmarks": [{
    "stats": {"...plugin, including data[]..."},
    "extra_info": {
      "group": "qdf", "benchmark_name": "reduce_df",
      "variant": {"dtype": "categorical"}, "timing_mode": "calibrated",
      "peak_memory_bytes": 11831186,
      "panel_size": {"tier": "small", "df_format": "long", "ticker_count": 80,
                     "cid_count": 10, "xcat_count": 8, "date_count": 1300,
                     "metrics": ["value"], "observation_count": 104000,
                     "row_count": 104000, "column_count": 4}},
    "macrosynergy": {
      "measurement_id": "qdf/reduce_df/small/dtype=categorical",
      "group": "qdf", "benchmark_name": "reduce_df", "tier": "small",
      "variant": "dtype=categorical", "timing_mode": "calibrated",
      "observation_count": 104000,
      "mean_seconds": 0.0655, "mean_seconds_low": 0.0648, "mean_seconds_high": 0.0663,
      "median_seconds": 0.0654, "stddev_seconds": 0.0021,
      "median_absolute_deviation_seconds": 0.0009,
      "observations_per_second": 1585952.0, "peak_memory_bytes": 11831186}}],

  "macrosynergy": {"run_id": "2026-07-31T...", "tiers_measured": ["small", "medium"],
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
`stats.data` unless `--drop-round-timings` is passed, so no chart needs a rerun.

`machine_info.macrosynergy` records the fields in which a difference can change a timing
by a multiple. Comparability is one `fingerprint` comparison rather than per-field
negotiation: two runs either agree on all nine fields or they do not. `usable_cpu_count`
is the count available to the process (`len(os.sched_getaffinity(0))`), not the count the
host has, because BLAS sizes its thread pool from availability and that is what enters a
timing. Every collector returns `None` rather than raising, so a missing `/proc` costs one
field instead of the run.

## Files

| File | Contents |
| --- | --- |
| `panel_sizes.py` | `PanelSize`, `PanelSizeCatalog`, `PANEL_SIZES`, `TARGET_OBSERVATION_COUNTS`, the cached builders and `clear_df_cache` |
| `machine.py` | `MachineProfile`, `PeakMemoryTracker` |
| `results.py` | `ConfidenceInterval`, `BenchmarkMeasurement`, `BenchmarkRunResults` |
| `conftest.py` | tier selection, host stamping, `extra_info` stamping |
| `test_benchmarks.py` | the benchmarks, plus `measure` and `measure_mutating` |
| `analyse.py` | the enrichment CLI |
| `parity.py` | comparison helpers for the parity tests |
| `test_panel_sizes.py`, `test_machine.py`, `test_results.py`, `test_conftest.py` | framework tests, in the default gate |
| `test_parity_*.py` | self-consistency guards, in the default gate |
| `results/` | run output, gitignored |

A test module mirrors the source module it covers, as elsewhere in `tests/`.

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
```

One direction only. `BenchmarkRunResults.compare_against(baseline)` reads two runs and
returns a DataFrame of `measurement_id`, `baseline_seconds`, `current_seconds`,
`percent_change` and a `verdict` of `improved`, `regressed` or `no change`. The verdict
comes from whether the two mean intervals overlap rather than from a percentage
threshold, so run-to-run noise reads as no change. It is built but nothing calls it yet.

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

### `MachineProfile` / `PeakMemoryTracker`

`MachineProfile.capture()` reads the nine fields described under **Output**;
`describe()` serialises them and `fingerprint` hashes that serialisation.

`PeakMemoryTracker` is a context manager over `tracemalloc` exposing `peak_bytes`. It is
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

The repo-root `conftest.py` declares `--perf-tiers` and nothing else: no imports, no
fixtures, no hooks. It exists because pytest honours `pytest_addoption` only in initial
conftest files, so an option declared under `tests/perf/` would fail for invocations
without a path argument.

`tests/perf/conftest.py` resolves the tiers in `pytest_configure`, which runs before
collection and therefore before the catalogue's size methods are evaluated inside
`@parametrize`. `_MachineProfileStamper` writes the profile into `machine_info` and is
registered as a plugin object, only when the benchmark plugin is active: a module-level
`pytest_benchmark_*` function in a conftest makes pluggy abort collection of the whole
repository whenever that plugin is absent. `machine_profile()` captures on first use
rather than at startup, because capture costs a couple of seconds and `addopts` sets
`-n auto`. The autouse `_record_benchmark_metadata` fixture writes `group`,
`benchmark_name`, `variant`, `timing_mode` and the `panel_size` block onto `extra_info`,
the size block **after** the test body so it describes a DataFrame that was actually
built. `pytest_sessionfinish` releases every built frame.

`extra_info` is the only channel from a test to the results file. Nothing downstream
recovers identity by parsing a test name.

## Invariants

- Counts are derived from a `PanelSize`, and both builders assert the frame matches them.
- DataFrame construction never happens inside a measured region.
- `extra_info` is the only test-to-file channel.
- `with_derived_statistics` adds keys and never removes or rewrites one.
- A measurement identifier comes from metadata, not from a pytest node id.
- Timings are seconds, memory is bytes, and the unit is in the name.
- Benchmarks, expensive frame builds and subprocess tests are all `perf`-marked, so the
  default gate stays around a second.

## Conventions

Tests here follow the repository's `unittest.TestCase` plus `parameterized.expand` style,
with two documented exceptions. `test_benchmarks.py` is pytest-native function style
because pytest refuses to inject fixtures into `TestCase` methods and every benchmark
depends on the `benchmark` fixture. `test_conftest.py` reaches the two fixtures it needs
through one autouse fixture that stashes them on the instance, for the same reason. Note
also that a pytest mark does not survive `parameterized.expand`, so a mark that must
apply to generated cases goes on the class.

## Known failures

Two repo-wide collection errors come from `linearmodels` being declared as a test
dependency but not installed. They are unrelated to this suite.
