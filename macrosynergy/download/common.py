"""
Custom exceptions for the `macrosynergy.download` subpackage.
"""

import requests

OAUTH_TOKEN_URL: str = "https://authe.jpmchase.com/as/token.oauth2"
OAUTH_DQ_RESOURCE_ID: str = "JPMC:URI:RS-06785-DataQueryExternalApi-PROD"
JPMAQS_GROUP_ID: str = "JPMAQS"
API_DELAY_PARAM: float = 0.3  # 300ms delay between requests
API_RETRY_COUNT: int = 5  # retry count for transient errors
HL_RETRY_COUNT: int = 5  # retry count for "high-level" requests
MAX_CONTINUOUS_FAILURES: int = 5  # max number of continuous errors before stopping
HEARTBEAT_ENDPOINT: str = "/services/heartbeat"
TIMESERIES_ENDPOINT: str = "/expressions/time-series"
CATALOGUE_ENDPOINT: str = "/group/instruments"
HEARTBEAT_TRACKING_ID: str = "heartbeat"
OAUTH_TRACKING_ID: str = "oauth"
TIMESERIES_TRACKING_ID: str = "timeseries"
CATALOGUE_TRACKING_ID: str = "catalogue"

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
CERT_BASE_URL: str = "https://platform.jpmorgan.com/research/dataquery/api/v2"
OAUTH_BASE_URL: str = (
    "https://api-developer.jpmorgan.com/research/dataquery-authe/api/v2"
)

