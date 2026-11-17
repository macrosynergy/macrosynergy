# `macrosynergy.learning`

## Description

The `macrosynergy.learning` subpackage contains functions and classes to assist the 
creation of machine learning solutions with macro quantamental data. 

Currently, the functionality is built around integrating the `macrosynergy` package and 
associated JPMaQS data with the popular `scikit-learn` library, which provides a simple
interface for fitting common statistical machine learning models, as well as feature
selection methods, cross-validation classes and performance metrics.

The panel format of our quantamental dataframes renders many of the common classes and 
functions in scikit-learn either impossible or difficult to use, either conceptually or
practically. We provide custom scikit-learn estimators, transformers, metrics, wrappers
and classes to build a bridge between tradable economics and machine learning.

## Changing portfolio targets

`DynamicSignalOptimizer` is a small wrapper around `SignalOptimizer` for a portfolio
whose target assets change over time. Features and cross-sections stay fixed.
It owns the individual optimizers through composition; it does not inherit from
`SignalOptimizer` or construct a combined training panel during initialization.
Pydantic validates the target schedule; a dataclass represents each target period.
There is no custom splitter or automatic feature selection.

In `Wealth management with macro factors.ipynb`, `facts_zn` contains the fixed macro
features, `rets_nn` contains the candidate sleeve returns, and `dfnn` stores both
under `cid="GLB"`. The same inputs can be used here:

```python
from macrosynergy.learning import DynamicSignalOptimizer

# Illustrative membership dates and subsets; replace with your actual schedule.
target_schedule = {
    "2010-01-29": rets_nn[:6],
    "2015-01-30": rets_nn,
}

so_nn = DynamicSignalOptimizer(
    df=dfnn,
    xcats=facts_zn + rets_nn,
    n_targets=len(rets_nn),  # candidate targets, not the active count
    cids=["GLB"],
    freq="M",
    lag=1,
    xcat_aggs=["last", "sum"],
    drop_nas=True,
    target_schedule=target_schedule,
)
print(so_nn.get_regimes())

# Use the notebook's existing model dictionary and calculation options.
so_nn.calculate_predictions(
    name="MLP_LO",
    models=models,
    min_cids=1,
    min_periods=36,
    test_size=3,
    n_jobs_vintages=2,
    show_tqdm=True,
)
weights = so_nn.get_optimized_signals("MLP_LO")
```

Move the notebook's existing `models={...}` dictionary into a `models` variable to
use this example. Its model averaging, surrogate model, scoring and stored attributes
can be passed through unchanged; this wrapper adds no model defaults.

Use `n_jobs_vintages` to train target periods in parallel: a positive integer,
or `-1` for all available logical cores. The default is 1. Pydantic checks that
the count does not exceed `joblib.cpu_count()` (which respects CPU affinity and
container limits). Each child SignalOptimizer is called with `n_jobs_outer=1`
and `n_jobs_inner=1`, so time-period fitting and cross-validation stay serial.
These two arguments are not exposed by the wrapper. Process workers also limit
native BLAS/OpenMP thread pools to one thread.

There is one progress bar for the active vintages, updated as their results are
collected in schedule order. Child optimizer bars are disabled. Set
`show_tqdm=False` to hide the overall bar. Results and the `optimizers` list retain
schedule order regardless of worker completion order.

A schedule accepts either a date-to-list mapping or a boolean DataFrame with dates
as its index and candidate targets as columns. Dates are inclusive boundaries on
**output signal dates**, with no automatic rounding. A selection remains effective
until the next entry. There is no output before the first entry, and an empty list
suspends output. Omitting the schedule uses every target throughout.
`get_regimes()` describes this schedule within the selected raw input date window.
It does not apply frequency aggregation, feature lagging or training-history checks;
those belong to each child optimizer.

For each period, the wrapper constructs a normal `SignalOptimizer` with the fixed
features and selected targets, calls `calculate_predictions`, and retains results
within that period. The underlying optimizer controls missing data, training history,
cross-validation, rolling windows and retraining. Available history before an asset's
admission can be used for training. A membership change does not force an extra fit;
`test_size`, `start_date` and `split_functions` retain their ordinary per-run meaning.

Signals are combined without filling across target-period boundaries. With multiple
candidate targets, names remain `<target>_<process>` even when only one target is
active. Inactive targets have no rows. The wrapper provides combined diagnostic
tables through `get_optimal_models`, `get_selected_features`,
`get_feature_importances`, `get_intercepts` and `get_feature_correlations`.
Successful reruns replace that process's results.

Plotting and training-data inspection use an actual child optimizer, for example:

```python
vintage = so_nn.optimizers["MLP_LO"][0]
vintage.models_heatmap("MLP_LO")
vintage.feature_selection_heatmap("MLP_LO")
```

These plots show that child's full history, including fits outside its retained
output interval. For plots across retained intervals, use the wrapper's combined
diagnostic dataframes. The wrapper no longer exposes inherited plotting methods,
`run`, `X` or `y`.

This MVP runs each target selection over the supplied history before slicing its
outputs, so it may repeat costly fits. Avoiding those repeated fits is a possible
later improvement. Dynamic features, dynamic cross-sections, custom calendars and
availability rules are deliberately outside this version.
