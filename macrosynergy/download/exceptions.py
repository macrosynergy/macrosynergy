""" Exception Classes for the download module. """


class AuthenticationError(Exception):
    """Raised when authentication fails."""


class InvalidDataframeError(Exception):
    """Raised when a dataframe is not valid."""


class MissingDataError(Exception):
    """Raised when data is missing from a requested dataframe."""


class DownloadError(Exception):
    """Raised when a download fails or is incomplete."""


class InvalidResponseError(Exception):
    """Raised when a response is not valid."""
