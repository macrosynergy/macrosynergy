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
- `BarPlot`: A class for a single bar chart.
- `Heatmap`: A class for a single heatmap.

### User-facing functions:

- `view.*` functions: A set of preset methods to view data in commonly used contexts.

  - `timelines()`: A function for viewing time-series data. Uses `FacetPlot.lineplot()` and `LinePlot` internally.
  - `availability()`: A function for viewing data availability. Uses `Heatmap` internally.
  - `correlation()`: A function for viewing correlation matrices. Uses `Heatmap` internally.
  - `metrics()`: A function for viewing metrics. Uses `Heatmap` internally.
  - `ranges()`: A function for viewing ranges. Uses `BoxPlot` and `BarPlot` internally.
  - `distribution()`: A function for viewing distributions. Uses `FacetPlot.boxplot()` and `BoxPlot` internally.
  - `reg_scatter()`: A function for viewing scatter plots. Uses `FacetPlot.scatterplot()` and `ScatterPlot` internally.
