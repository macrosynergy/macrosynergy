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