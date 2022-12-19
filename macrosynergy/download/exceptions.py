""" Exception Classes for the download module. """


class AuthenticationError(Exception):
    """Raised when authentication fails."""

    pass


class InvalidDataframeError(Exception):
    """Raised when a dataframe is not valid."""

    pass

class MissingDataError(Exception):
    """Raised when data is missing from a requested dataframe."""

    pass

class DownloadError(Exception):
    """Raised when a download fails or is incomplete."""
    
    pass

class InvalidResponseError(Exception):
    """Raised when a response is not valid."""
    
    pass