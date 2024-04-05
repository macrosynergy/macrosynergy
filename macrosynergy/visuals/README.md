# `macrosynergy.visuals`

## Description

The `macrosynergy.visuals` subpackage contains functions for visualizing quantamental data, and to provide `view.*` functions for the `macrosynergy` package.

The functionality is built around the ability to create quick generic plots of data, and to provide a framework for creating custom plots.
Simplicity, speed, consistency and flexibility are the main goals.
The core visualization functions are built around matplotlib, with goals of mimicking the rich and dataframe-friendly nature of seaborn without the overheads it comes with.

## Functionality

The `macrosynergy.visuals` subpackage features the following functionality:

### User-facing functions:

- `timelines()`: A function for viewing time-series data as line plots.
- `metrics()`: A function for viewing metrics as heatmaps.
- `ranges()`: A function for viewing ranges with box plots and bar charts.
- `multiple_reg_scatter()`: A function for viewing multiple scatter plots with regression lines.
- `metrics()`: A function for viewing metrics. Designed to view any on a heatmap; not ideal for viewing `value`.
- `grading()`: A function for viewing grading data.
- `correlation()`: A function for viewing correlation matrices.

### Backend functions:

- `FacetPlot`: A class for creating facet plots of data.

  - `.lineplot()`: A method for creating line plots.

  To be implemented:

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
To maintain readability, two generic type-aliases are defined in `macrosynergy.visuals.plotter`:

The following parameters, are essentially all that is required to for the dataframe to be reduced and filtered to the desired subset:

```text
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
with FacetPlot( df=data_qdf,
                cids=sel_cids,
                xcats=sel_xcats,
                start='2010-01-01',
                end='2019-12-31') as fp:

    fp.lineplot(figsize=(12, 8), title='Quantamental data - lineplot')

    fp.lineplot(metric='grading', figsize=(12, 8),
                title='Grading for time-series')

    fp.lineplot(cid_xcat_grid=True)

# produces 3 plots, with the same data, but different arguments.

```

With this approach, it is possible to integrate the utilities defined in this subpackage in existing code, while still allowing for quick and intuitive plotting of data in notebooks/research mode.

We use a direct subset of `matplotlib`'s functions, with some renaming where convenient. We also introduce some method specific parameters, such as `cid_xcat_grid` for `FacetPlot.lineplot()`, which allows for more intuitive plotting with context to the data.

Here are the generic arguments that behave the same across all plotting methods:

```text
:param <Tuple[Number, Number]> figsize: a tuple of floats specifying the width and height of the figure.

:param <str> title: the title of the plot.

:param <int> title_fontsize: the font size of the title.

:param <float> title_xadjust: the x-adjustment of the title.

:param <float> title_yadjust: the y-adjustment of the title.

:param <bool> legend: whether or not to show a legend for the plot.

:param <str> legend_loc: Location of the legend.

:param <int> legend_ncol: Number of columns in the legend.

:param <tuple> legend_bbox_to_anchor: Bounding box for the legend.

:param <bool> legend_frame: Show the legend frame.

:param <bool> show: Show the plot.

:param <str> save_to_file: Save the plot to a file.

:param <int> dpi: DPI of the saved image.

:param <bool> return_figure: Return the figure object.

:param <Dict[str, Any]> plot_func_args: A dictionary of arguments to pass to the plotting function.
```

## Implementation specifics

### Type hinting and `argvalidation`

Type hinting is used extensively, and allows for syntax validation across almost all utilities. The validation is done by `argvalidation`. It uses a series of "tricks" to allow for type hinting of arguments, and to validate them against their type hints. These are better explained by reading the code itself. (See `macrosynergy.visuals.plotter`)

One problem with type hinting is that it can quickly become too verbose, and therefore unreadable.
To mitigate situations it is recommended we use more generic type hints where possible (such as Iterable (over list), and Number (over Union[int, float]).

### Arg Copying and `argcopy`

The `argcopy` decorator is used to copy arguments passed to methods, preventing the original arguments from being modified. This is useful for methods that modify arguments in-place, such as `reduce_df()`, or even preventing input lists from being modified.
However, we only copy specific mutable types, such as `list`, `dict`, `np.ndarray`, and `pd.DataFrame`. This way, immutable types and other "unnecessary" types are not copied, and memory usage is kept to a minimum.
