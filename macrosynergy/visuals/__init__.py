from .facetplot import FacetPlot
from .lineplot import LinePlot
from .plotter import Plotter
from .heatmap import Heatmap
from .common import NoneType, Numeric
from . import view


TYPES = ["NoneType", "Numeric"]
CLASSES = ["FacetPlot", "LinePlot", "Plotter", "Heatmap"]
MODULES = []

__all__ = TYPES + CLASSES + MODULES
