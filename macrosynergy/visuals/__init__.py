# from .plots import FacetPlot, LinePlot, Plotter, Heatmap
from .facetplot import FacetPlot
from .lineplot import LinePlot
from .plotter import Plotter
from .heatmap import Heatmap
from .grades import view_grades
from .metrics import view_metrics
from .timelines import timelines
from .visual_paneldates import visual_paneldates
from .correlation import view_correlation
from .ranges import view_ranges

TYPES = ["NoneType", "Numeric"]
CLASSES = ["FacetPlot", "LinePlot", "Plotter", "Heatmap"]
MODULES = []
FUNCTIONS = [
    "timelines",
    "view_metrics",
    "view_grades",
    "visual_paneldates",
    "view_correlation",
    "view_ranges"
]

__all__ = CLASSES + MODULES
