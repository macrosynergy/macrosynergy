"""
Datastream Web Service (DSWS) integration package.

Provides :class:`DatastreamConnection` for authentication / lifecycle management
and :class:`DatastreamDataManager` for data retrieval and post-processing.
"""

from .connection import DatastreamConnection
from .data_manager import DatastreamDataManager

__all__ = [
    "DatastreamConnection",
    "DatastreamDataManager",
]
