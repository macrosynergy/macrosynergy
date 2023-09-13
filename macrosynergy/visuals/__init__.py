from .facetplot import FacetPlot
from .lineplot import LinePlot
from .plotter import Plotter
from . import view

import numpy as np
from typing import Union, SupportsInt, SupportsFloat

NoneType = type(None)
Numeric = Union[int, float, np.int64, np.float64, SupportsInt, SupportsFloat]


__all__ = ["Plotter", "LinePlot", "FacetPlot", "view"]
