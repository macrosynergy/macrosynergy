"""Old DataQuery Interface path"""
import warnings
from ..download.jpmaqs import JPMaQSDownload

class Interface(JPMaQSDownload):
    """
    Extends `macrosynergy.download.jpmaqs.JPMaQSDownload` to provide backwards 
    compatibility with the deprecated `macrosynergy.dataquery.api.Interface` class.
    """

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "api.Interface has been moved to macrosynergy.download.jpmaqs - will be removed in v0.1.0",
            DeprecationWarning,
            stacklevel=2
        )

        super().__init__(*args, **kwargs)
        
warnings.warn(
    "api.Interface has been moved to macrosynergy.download.jpmaqs - will be removed in v0.1.0",
    DeprecationWarning,
    stacklevel=2
)

__all__ = ["Interface"]
