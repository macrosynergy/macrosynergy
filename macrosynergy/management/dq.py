"""Old DataQuery Interface path"""
import warnings
from macrosynergy.dataquery.api import Interface as DataQueryInterface

warnings.warn(
    "DataQueryInterface has been moved to macrosynergy.dataquery.api",
    DeprecationWarning,
    stacklevel=2
)

__all__ = ["DataQueryInterface"]
