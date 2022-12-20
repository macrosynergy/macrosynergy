"""Old DataQuery Interface path"""
import warnings
from macrosynergy.dataquery.api import Interface as DataQueryInterface

warnings.warn(
    "DataQueryInterface has been moved to macrosynergy.dataquery.api - will be removed in v0.1.0",
    DeprecationWarning,
    stacklevel=2
)

__all__ = ["DataQueryInterface"]
