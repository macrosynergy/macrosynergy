# `macrosynergy.securities`

Tools for constituent-level index calculations, portfolio analysis, and signal
optimization across changing constituent regimes.

## Changing portfolio targets

`DynamicSignalOptimizer` wraps one `SignalOptimizer` per explicit regime. Supply a
single JPMaQS dataframe containing the fixed features and all available target
histories, a list of feature categories, and a mapping of regime IDs to dates and
target names. The regime mapping is required and must be nonempty. The wrapper
derives the candidate targets from that mapping and
handles per-regime input preparation, history checks, prediction starts and output
slicing. Pydantic validates the inputs; a dataclass carries each run's outcome.

For the Macro-aware S&P500 notebook, rotate stock returns to target categories once
outside the class, then combine them with the macro features. Keep all history from
`_ANALYSIS_START`; do not trim each target's history to its index admission date.
The notebook's `compressed_index_summary` can be adapted directly:

```python
from macrosynergy.securities import DynamicSignalOptimizer

regimes = {
    regime_id: {
        "start": details["start"],
        "end": details["end"],
        "end_return": details["end_return"],
        "targets": [
            blueprint_stock_relative_return_category.replace("{cid}", stock)
            for stock in sorted(details["constituents"])
        ],
    }
    for regime_id, details in compressed_index_summary.items()
}

# df_inputs contains macro features and the rotated stock return categories
# under the same cross-section. Stock rotation and output renaming stay outside
# the wrapper, using the notebook's existing conventions.
optimizer = DynamicSignalOptimizer(
    df=df_inputs,
    features=features_xcats,
    cids=cidx,
    start=_ANALYSIS_START,
    freq=forecast_frequency,
    lag=1,
    xcat_aggs=["last", "sum"],
    drop_nas="X",
    regimes=regimes,
)
print(optimizer.get_regimes())  # Inspect the plan without fitting.

optimizer.calculate_predictions(
    name=strat_name,
    models={strat_name: make_ensemble()},
    multi_target_fill="zero",
    store_additional_data=["weights_"],
    min_cids=1,
    min_periods=MIN_TRAIN_PERIODS,
    test_size=REFIT_PERIODS,
    skip_validation=True,
    n_jobs_vintages=2,
)
weights = optimizer.get_optimized_signals(strat_name)
report = optimizer.get_regimes(strat_name)
print(report[["status", "missing_targets", "skip_reason"]])
```

Each regime preserves its ID and its supplied target order. Regimes are sorted by
start date, must not overlap, and remain separate even when their target lists are
identical. Gaps between regimes produce no signals. Each child receives the fixed
features followed by that regime's available targets, with `n_targets` set to the
number of available targets. No separate candidate list or target count is needed.
This `regimes` mapping replaces the earlier date-to-list `target_schedule` interface.

The execution sequence follows the notebook:

1. Select the regime's target records through `end`, within any global input date
   bounds and cross-section restrictions. Record targets with no rows as missing.
2. Count distinct calendar months across those returns, including rows with missing
   values. Skip the regime when fewer than `min_periods` months are represented.
   This check happens **before** `end_return` and does not require that many months
   for each target separately. It counts months even if `freq` is daily or weekly.
3. Cap both features and returns at the earlier of `end` and `end_return`, inclusively.
4. Construct a native `SignalOptimizer`. Find its latest processed learning date on
   or before the regime's `start`, and pass that date as `start_date`. Skip if no
   such date or no subsequent learning date exists. Other child errors propagate.
5. Calculate predictions and retain records from the business-day rollback of
   `start`, inclusive, to the business-day rollback of `end`, exclusive.

The regime's `start` controls its prediction clock, not the beginning of training
history. Global constructor `start` and `end` filter history for every regime.
`freq` controls analysis frequency; `test_size` controls retraining intervals in
those periods. Do not pass `start_date`: the wrapper derives
it separately for each child.

Use `n_jobs_vintages` for parallel regimes: a positive integer up to
`joblib.cpu_count()`, or `-1` for all available cores. The default is 1. Child
`n_jobs_outer` and `n_jobs_inner` are fixed at 1; process workers limit native thread
pools to one thread. One progress bar covers every supplied regime, including skips,
and advances as results are collected in schedule order. Use `show_tqdm=False` to
hide it.

`get_regimes()` returns the original plan. `get_regimes(process_name)` adds available
and missing targets, history-month counts, first prediction boundaries, status and
skip reason. Empty target lists and entirely unavailable target sets are skipped.
If all regimes are skipped, the process stores empty outputs and a complete report.
A successful rerun replaces that process's outputs, children and report; a failed
run preserves previously published results.

Combined signals retain `<target>_<process>` names when there are multiple candidate
targets, including regimes with only one available target. Diagnostic getters return
records dated within each retained output interval. Child optimizers are available
by ID for full diagnostics, input data and plots:

```python
fitted_regime_id = next(iter(optimizer.optimizers[strat_name]))
child = optimizer.optimizers[strat_name][fitted_regime_id]
child.models_heatmap(strat_name)
```

Skipped regimes have no child entry. Retaining fitted children keeps their training
data and full outputs in memory. The wrapper uses composition and exposes no
inherited `run`, `X`, `y` or plotting methods.

For ordinary fixed-target fits, use `SignalOptimizer` directly. This wrapper
always derives targets from `regimes`; omitting it, passing None or supplying an
empty mapping raises an error. Dynamic features and cross-sections remain outside
this version.

A self-contained example with seeded synthetic data is available at the bottom of
`dynamic_signal_optimizer.py` under `if __name__ == "__main__":`.
