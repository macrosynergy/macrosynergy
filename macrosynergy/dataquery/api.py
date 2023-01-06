"""Old DataQuery Interface path"""
import warnings
from ..download.jpmaqs import JPMaQSDownload as Interface

warnings.warn(
    "api.Interface has been moved to macrosynergy.download.jpmaqs - will be removed in v0.1.0",
    DeprecationWarning,
    stacklevel=2
)

__all__ = ["Interface"]
