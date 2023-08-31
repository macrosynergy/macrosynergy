"""Deprecated `macrosynergy.management.dq.DataQueryInterface` class"""
import warnings
from ..download.jpmaqs import JPMaQSDownload


class DataQueryInterface(JPMaQSDownload):
    """
    Extends `macrosynergy.download.jpmaqs.JPMaQSDownload` to provide backwards
    compatibility with the deprecated `macrosynergy.management.dq.DataQueryInterface` class.
    """

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "Functionality for DataQueryInterface has been moved to `macrosynergy.download.jpmaqs.JPMaQSDownload` - will be removed in v0.1.0",
            DeprecationWarning,
            stacklevel=2,
        )

        super().__init__(*args, **kwargs)


__all__ = ["DataQueryInterface"]
