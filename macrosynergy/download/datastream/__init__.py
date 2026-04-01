"""
Re-export from macrosynergy_download.datastream.

Provides :class:`DatastreamConnection` for authentication / lifecycle management
and :class:`DatastreamDataManager` for data retrieval and post-processing.
"""

__all__ = [
    "DatastreamConnection",
    "DatastreamDataManager",
]


def __getattr__(name):
    _datastream_names = {
        "DatastreamConnection",
        "DatastreamDataManager",
    }
    if name in _datastream_names:
        from macrosynergy_download.datastream.connection import DatastreamConnection
        from macrosynergy_download.datastream.data_manager import DatastreamDataManager

        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
