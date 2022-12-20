from .api import Interface
import warnings

warnings.warn(
    "Functionality from macrosynergy.dataquery has been moved to macrosynergy.download.jpmaqs - will be removed in v0.1.0",
    DeprecationWarning,
    stacklevel=2
)

__all__ = ["Interface"]
