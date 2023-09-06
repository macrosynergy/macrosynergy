"""
Custom exceptions for the `macrosynergy.download` subpackage.
"""

import requests


class ExceptionAdapter(Exception):
    """Base class for all exceptions raised by the macrosynergy package."""

    def __init__(self, message: str = ""):
        self.message = message

    def __str__(self):
        # print type and message
        return f"{self.__class__.__name__}: {self.message} \n{self.__class__.__name__}"


class AuthenticationError(ExceptionAdapter):
    """Raised when authentication fails."""


class DownloadError(ExceptionAdapter):
    """Raised when a download fails or is incomplete."""


class InvalidResponseError(ExceptionAdapter):
    """Raised when a response is not valid."""


class HeartbeatError(ExceptionAdapter):
    """Raised when a heartbeat fails."""


class InvalidDataframeError(ExceptionAdapter):
    """Raised when a dataframe is not valid."""


class MissingDataError(ExceptionAdapter):
    """Raised when data is missing from a requested dataframe."""


class NoContentError(ExceptionAdapter):
    """Raised when no data is returned from a request."""


KNOWN_EXCEPTIONS = [
    requests.exceptions.ConnectionError,
    requests.exceptions.ConnectTimeout,
    requests.exceptions.ReadTimeout,
    ConnectionResetError,
    requests.exceptions.Timeout,
    requests.exceptions.TooManyRedirects,
    requests.exceptions.RequestException,
    requests.exceptions.HTTPError,
    requests.exceptions.InvalidURL,
    requests.exceptions.InvalidSchema,
    requests.exceptions.ChunkedEncodingError,
]
