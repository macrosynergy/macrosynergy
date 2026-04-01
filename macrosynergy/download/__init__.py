"""
Re-export layer: all download functionality is provided by the
``macrosynergy-download`` package (``macrosynergy_download``).

Existing imports such as ``from macrosynergy.download import JPMaQSDownload``
continue to work unchanged.
"""

from macrosynergy_download import (
    JPMaQSDownload,
    DataQueryInterface,
    custom_download,
    transform_to_qdf,
    JPMaQSFusionClient,
    DataQueryFileAPIClient,
)

__all__ = [
    "JPMaQSDownload",
    "DataQueryInterface",
    "custom_download",
    "transform_to_qdf",
    "JPMaQSFusionClient",
    "DataQueryFileAPIClient",
]
