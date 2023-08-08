# macrosynergy.visuals

## Description

The `macrosynergy.visuals` subpackage contains functions for visualizing quantamental data, and to provide `view.*` functions for the `macrosynergy` package.

The functionality is built around the ability to create quick generic plots of data, and to provide a framework for creating custom plots.
Simplicity, speed, consistency and flexibility are the main goals.
The core visualization functions are built around `matplotlib`, with goals of mimicking the rich and dataframe-friendly nature of `seaborn`.

## Functionality and usage

The `macrosynergy.visuals` subpackage features the following functionality:

### Backend functions:

- `FacetPlot`: A class for creating facet plots of data.

  - `.lineplot()`: A method for creating line plots.
  - `.scatterplot()`: A method for creating scatter plots.
  - `.from_subplots()`: A method for copying and arranging a list of individual subplots (of any type) into a facet plot.

- `LinePlot`: A class for a single line chart.

- `ScatterPlot`: A class for a single scatter chart.

- `BoxPlot`: A class for a single box plot.

- `Heatmap`: A class for a single heatmap.

- `BarPlot`: A class for a single bar chart.

### User-facing functions:

- `view.*` functions: A set of functions for quick preset methods to view data with commonly used, intuitive functions.

  - `timelines()`: A function for viewing time-series data. Uses `FacetPlot.lineplot()` and `LinePlot` internally.
  - `availability()`: A function for viewing data availability. Uses `Heatmap` internally.
  - `correlation()`: A function for viewing correlation matrices. Uses `Heatmap` internally.
  - `distribution()`: A function for viewing distributions. Uses `FacetPlot.boxplot()` and `BoxPlot` internally.
  - `reg_scatter()`: A function for viewing scatter plots. Uses `FacetPlot.scatterplot()` and `ScatterPlot` internally.