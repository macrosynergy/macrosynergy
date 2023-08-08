# macrosynergy.visuals

## Description

The `macrosynergy.visuals` subpackage contains functions for visualizing quantamental data, and to provide `view.*` functions for the `macrosynergy` package.

The functionality is built around the ability to create quick generic plots of data, and to provide a framework for creating custom plots.
Simplicity, speed, consistency and flexibility are the main goals.
The core visualization functions are built around `matplotlib`, with goals of mimicking the rich and dataframe-friendly nature of `seaborn`.

## Functionality

The `macrosynergy.visuals` subpackage features the following functionality:

### User-facing functions:

- `view.*` functions: A set of preset methods to view data in commonly used contexts.

  - `timelines()`: A function for viewing time-series data. Uses `FacetPlot.lineplot()` and `LinePlot` internally.
  - `availability()`: A function for viewing data availability. Uses `Heatmap` internally.
  - `correlation()`: A function for viewing correlation matrices. Uses `Heatmap` internally.
  - `metrics()`: A function for viewing metrics. Uses `Heatmap` internally.
  - `ranges()`: A function for viewing ranges. Uses `BoxPlot` and `BarPlot` internally.
  - `distribution()`: A function for viewing distributions. Uses `FacetPlot.boxplot()` and `BoxPlot` internally.
  - `reg_scatter()`: A function for viewing scatter plots. Uses `FacetPlot.scatterplot()` and `ScatterPlot` internally.

### Backend functions:

- `FacetPlot`: A class for creating facet plots of data.

  - `.lineplot()`: A method for creating line plots.
  - `.scatterplot()`: A method for creating scatter plots.
  - `.from_subplots()`: A method for copying and arranging a list of individual subplots (of any type) into a facet plot.

- `LinePlot`: A class for a single line chart.
- `ScatterPlot`: A class for a single scatter chart.
- `BoxPlot`: A class for a single box plot.
- `BarPlot`: A class for a single bar chart.
- `Heatmap`: A class for a single heatmap.

## Design principles

The interfaces follow a object-oriented approach, as opposed to a functional approach.

The base class for all of the "plotter" classes is `Plotter` - `macrosynergy.visuals.Plotter`.
This class is not intended to be used directly, but rather to be subclassed by other classes to inherit common functionality.
The base class implements dataframe reduction and filtering, functions. In the future, this can be extended to allow handling of larger-than-memory dataframes.
Another important feature is the metaclass the `Plotter` is based on - `macrosynergy.visuals.PlotterMetaClass`.
The metaclass implements two important features:

- `argvalidation`: A decorator for validating arguments passed to methods.
- `argcopy`: A decorator for copying arguments passed to methods, preventing the original arguments from being modified.

Since `PlotterMetaClass` is the metaclass to `Plotter` class, this behaviour is inherited by all subclasses. Therefore, all arguments are automatically syntactically validated against their type hints, reducing 30-50% LOC for most "large" functions. The only caveat this introduces is that all arguments must be type-hinted - extensively.

The following parameters, are essentially all that is required to for the dataframe to be reduced and filtered to the desired subset:

```
:param <pd.DataFrame> df: A DataFrame with the following columns:
    'cid', 'xcat', 'real_date', and at least one metric from -
    'value', 'grading', 'eop_lag', or 'mop_lag'.
:param <List[str]> cids: A list of cids to select from the DataFrame
    (self.df). If None, all cids are selected.
:param <List[str]> xcats: A list of xcats to select from the DataFrame
    (self.df). If None, all xcats are selected.
:param <List[str]> metrics: A list of metrics to select from the DataFrame
    (self.df). If None, all metrics are selected.
:param <bool> intersect: if True only retains cids that are available for
    all xcats. Default is False.
:param <List[str]> tickers: A list of tickers to select from the DataFrame
    (self.df). If None, all tickers are selected.
:param <str> start: ISO-8601 formatted date string. Select data from
    this date onwards. If None, all dates are selected.
:param <str> end: ISO-8601 formatted date string. Select data up to
    and including this date. If None, all dates are selected.
:param <str> backend: The plotting backend to use. Currently only
    'matplotlib' is supported.

```

These are effectively the arguments to `macrosynergy.management.shape_dfs.reduce_df()`, and `.reduce_df_by_ticker()`.

This means, once a plotter class is instantiated, it can be re-used (typically with a context manager) to create multiple plots with different arguments, without having to re-filter/re-load the dataframe.

Example:

```python
with FacetPlot(df=data_qdf, cids=sel_cids, xcats=sel_xcats, start='2010-01-01', end='2019-12-31') as fp:
    fp.lineplot(figsize=(12, 8), title='Quantamental data - lineplot')
    fp.lineplot(metric='grading', figsize=(12, 8), title='Grading for time-series')
    fp.lineplot(cid_xcat_grid=True)
```

With this approach, it is possible to integrate the utilities defined in this subpackage in existing code, while still allowing for quick and intuitive plotting of data in notebooks/research mode.
