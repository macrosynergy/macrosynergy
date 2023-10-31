# from .plots import FacetPlot, LinePlot, Plotter, Heatmap
from .facetplot import FacetPlot
from .lineplot import LinePlot
from .plotter import Plotter
from .heatmap import Heatmap
from .grades import view_grades
from .metrics import view_metrics
from .timelines import timelines
from .view_panel_dates import view_panel_dates
from .correlation import view_correlation
from .ranges import view_ranges
from .table import view_table

TYPES = ["NoneType", "Numeric"]
CLASSES = ["FacetPlot", "LinePlot", "Plotter", "Heatmap"]
MODULES = []
FUNCTIONS = [
    "timelines",
    "view_metrics",
    "view_grades",
    "view_panel_dates",
    "view_correlation",
    "view_ranges",
    "view_table"
]

__all__ = CLASSES + MODULES
