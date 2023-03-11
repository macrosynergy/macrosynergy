"""
Interface for downloading data from the JPMorgan DataQuery API.
This module is not intended to be used directly, but rather through
macrosynergy.download.jpmaqs.py. However, for a use cases independent
of JPMaQS, this module can be used directly to download data from the
JPMorgan DataQuery API.
"""
import concurrent.futures
import time
import logging
import itertools
import base64
import os
import io
import requests
from typing import List, Optional, Dict
from datetime import datetime
from tqdm import tqdm

CERT_BASE_URL: str = "https://platform.jpmorgan.com/research/dataquery/api/v2"
OAUTH_BASE_URL: str = (
    "https://api-developer.jpmorgan.com/research/dataquery-authe/api/v2"
)
OAUTH_TOKEN_URL: str = "https://authe.jpmchase.com/as/token.oauth2"
OAUTH_DQ_RESOURCE_ID: str = "JPMC:URI:RS-06785-DataQueryExternalApi-PROD"
API_DELAY_PARAM: float = 0.3  # 300ms delay between requests
API_RETRY_COUNT: int = 5  # retry count for transient errors
HL_RETRY_COUNT: int = 5  # retry count for "high-level" requests
API_EXPR_LIMIT: int = 20  # 20 is the max number of expressions per API call
HEARTBEAT_ENDPOINT: str = "/services/heartbeat"
TIMESERIES_ENDPOINT: str = "/expressions/time-series"

logger = logging.getLogger(__name__)
debug_stream_handler = logging.StreamHandler(io.StringIO())
debug_stream_handler.setLevel(logging.NOTSET)
debug_stream_handler.setFormatter(
    logging.Formatter(
        "%(asctime)s - %(levelname)s - %(module)s - %(funcName)s :: %(message)s"
    )
)
logger.addHandler(debug_stream_handler)


class AuthenticationError(Exception):
    """Raised when authentication fails."""


class DownloadError(Exception):
    """Raised when a download fails or is incomplete."""


class InvalidResponseError(Exception):
    """Raised when a response is not valid."""


class HeartbeatError(Exception):
    """Raised when a heartbeat fails."""


def validate_response(response: requests.Response) -> dict:
    """
    Validates a response from the API. Raises an exception if the response
    is invalid (e.g. if the response is not a 200 status code).

    :param <requests.Response> response: response object from requests.request().

    :return <dict>: response as a dictionary. If the response is not valid,
        this function will raise an exception.

    :raises <InvalidResponseError>: if the response is not valid.
    :raises <AuthenticationError>: if the response is a 401 status code.
    :raises <KeyboardInterrupt>: if the user interrupts the download.
    """

    error_str = (
        f"Response : {response}\n"
        f"Requested URL: {response.request.url}\n"
        f"Response status code: {response.status_code}\n"
        f"Response headers: {response.headers}\n"
        f"Response text: {response.text}\n"
        f"Timestamp (UTC) : {datetime.utcnow().isoformat()}; \n"
    )
    # TODO : Use response.raise_for_status() as a better way to check for errors
    if response.status_code == 200:
        try:
            response_dict = response.json()
            if response_dict is None:
                raise InvalidResponseError(f"Response is empty.\n{error_str}")
            return response_dict
        except Exception as exc:
            if isinstance(exc, KeyboardInterrupt):
                raise exc

            raise InvalidResponseError(
                error_str + f"Error parsing response as JSON: {exc}"
            )

    else:
        if response.status_code == 401:
            raise AuthenticationError(error_str)

        raise InvalidResponseError(
            f"Request did not return a 200 status code.\n{error_str}"
        )


def form_full_url(url: str, params: Dict = {}) -> str:
    """
    Forms a full URL from a base URL and a dictionary of parameters.
    Useful for logging and debugging.

    :param <str> url: base URL.
    :param <dict> params: dictionary of parameters.

    :return <str>: full URL
    """
    return requests.compat.quote(
        (f"{url}?{requests.compat.urlencode(params)}" if params else url),
        safe="%/:=&?~#+!$,;'@()*[]",
    )


def request_wrapper(
    url: str,
    headers: Optional[Dict] = None,
    params: Optional[Dict] = None,
    method: str = "get",
    **kwargs,
) -> dict:
    """
    Wrapper for requests.request() that handles retries and logging.
    All paramaters and kwargs are passed to requests.request() (except for
    "tracking_id", which is used for logging purposes only).

    :param <str> url: URL to request.
    :param <dict> headers: headers to pass to requests.request().
    :param <dict> params: params to pass to requests.request().
    :param <str> method: HTTP method to use. Must be one of "get"
        or "post". Defaults to "get".
    :param <dict> kwargs: kwargs to pass to requests.request().

    :return <dict>: response as a dictionary.

    :raises <InvalidResponseError>: if the response is not valid.
    :raises <AuthenticationError>: if the response is a 401 status code.
    :raises <DownloadError>: if the request fails after retrying.
    :raises <KeyboardInterrupt>: if the user interrupts the download.
    :raises <ValueError>: if the method is not one of "get" or "post".
    :raises <Exception>: other exceptions may be raised by requests.request().
    """

    tracking_id = kwargs.pop("tracking_id", "")
    if not method in ["get", "post"]:
        raise ValueError(f"Invalid method: {method}")

    log_url = form_full_url(url, params)
    logger.info(f"Requesting URL: {log_url} , tracking_id: {tracking_id}")

    retry_count = 0
    # if kwards contains "tracking_id", use that, otherwise generate a new one
    while retry_count < API_RETRY_COUNT:
        try:
            response = requests.request(
                method, url, headers=headers, params=params, **kwargs
            )
            return validate_response(response)

        except Exception as exc:
            # if keyboard interrupt, raise as usual
            if isinstance(exc, KeyboardInterrupt):
                print("KeyboardInterrupt -- halting download")
                raise exc

            # authentication error, clearly not a transient error
            if isinstance(exc, AuthenticationError):
                raise exc

            # NOTE: exceptions that need the code to break should be caught before this
            # all other exceptions are caught here and retried after a delay

            error_statement = (
                f"Request to {log_url} failed with error {exc}. "
                f"Retry count: {retry_count}. "
                f"Tracking ID: {tracking_id}"
            )

            known_exceptions = [
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

            if any([isinstance(exc, e) for e in known_exceptions]):
                logger.warning(error_statement)
                retry_count += 1
                time.sleep(API_DELAY_PARAM)
            else:
                raise exc

    raise DownloadError(
        f"Request to {log_url} failed with status code {response.status_code}. "
        "No longer retrying."
    )


class OAuth(object):
    """
    Class for handling OAuth authentication for the DataQuery API.

    :param <str> client_id: client ID for the OAuth application.
    :param <str> client_secret: client secret for the OAuth application.
    :param <dict> proxy: proxy to use for requests. Defaults to None.
    :param <str> base_url: base URL for OAuth access.
    :param <str> token_url: URL for getting OAuth tokens.
    :param <str> dq_resource_id: resource ID for the JPMaQS Application.

    :return <OAuth>: OAuth object.

    :raises <ValueError>: if client_id or client_secret are not strings.
    :raises <Exception>: other exceptions may be raised by underlying functions.
    """

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        proxy: Optional[dict] = None,
        base_url: str = OAUTH_BASE_URL,
        token_url: str = OAUTH_TOKEN_URL,
        dq_resource_id: str = OAUTH_DQ_RESOURCE_ID,
    ):
        try:
            id_error = f"client_id argument must be a <str> and not {type(client_id)}."
            assert isinstance(client_id, str), id_error
            secret_error = (
                f"client_secret must be a <str> and not {type(client_secret)}."
            )
            assert isinstance(client_secret, str), secret_error
            proxy_error = f"proxy must be a <dict> and not {type(proxy)}."
            assert isinstance(proxy, dict) or proxy is None, proxy_error
            url_error = f"base_url must be a <str> and not {type(base_url)}."
            assert isinstance(base_url, str), url_error
            token_url_error = f"token_url must be a <str> and not {type(token_url)}."
            assert isinstance(token_url, str), token_url_error
            dq_resource_id_error = (
                f"dq_resource_id must be a <str> and not {type(dq_resource_id)}."
            )
            assert isinstance(dq_resource_id, str), dq_resource_id_error

        except AssertionError as exc:
            raise ValueError(exc)
        except Exception as exc:
            raise exc
        self.base_url: str = base_url
        self.__token_url: str = token_url
        self.__dq_api_resource_id: str = dq_resource_id
        self.proxy: Optional[dict] = proxy

        self.client_id: str = client_id

        self.client_secret: str = client_secret

        self._stored_token: Optional[dict] = None
        self.token_data = {
            "grant_type": "client_credentials",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "aud": self.__dq_api_resource_id,
        }

    def _valid_token(self) -> bool:
        """
        Method to check if the stored token is valid.

        :return <bool>: True if the token is valid, False otherwise.
        """
        if self._stored_token is None:
            return False

        created: datetime.datetime = self._stored_token["created_at"]
        expires: int = self._stored_token["expires_in"]
        is_active = (datetime.now() - created).total_seconds() / 60 >= (expires - 1)
        return is_active

    def _get_token(self) -> str:
        """
        Method to get a new OAuth token.

        :return <str>: OAuth token.
        """
        if not self._valid_token():
            js = request_wrapper(
                url=self.__token_url,
                data=self.token_data,
                method="post",
                proxies=self.proxy,
                tracking_id="get_oauth_token",
            )
            time.sleep(API_DELAY_PARAM)
            # TODO : Is sleep needed here?
            # on failure, exception will be raised by request_wrapper

            self._stored_token: dict = {
                "created_at": datetime.now(),
                "access_token": js["access_token"],
                "expires_in": js["expires_in"],
            }

        return self._stored_token["access_token"]

    def _request(
        self,
        url: str,
        params: dict = None,
        proxy: Optional[dict] = None,
        tracking_id: Optional[str] = None,
    ) -> dict:
        """
        Wrapper for request_wrapper to add the correct authorization and headers.

        :param <str> url: URL to request.
        :param <dict> params: parameters to pass to the request.
        :param <dict> proxy: proxy to use for the request.
        :param <str> tracking_id: tracking ID to use for the request (for logging).

        :return <dict>: JSON response from the request.

        :raises <Exception>: other exceptions may be raised by underlying functions.
        """

        # this method is only needed to insert the relavant authorization header
        return request_wrapper(
            url=url,
            params=params,
            headers={"Authorization": "Bearer " + self._get_token()},
            proxies=proxy,
            tracking_id=tracking_id,
        )


class CertAuth(object):
    """
    Class for handling certificate based authentication for the DataQuery API.

    :param <str> username: username for the DataQuery API.
    :param <str> password: password for the DataQuery API.
    :param <str> crt: path to the certificate file.
    :param <str> key: path to the key file.
    :param <str> base_url: base URL for the DataQuery API.
    :param <dict> proxy: proxy to use for requests. Defaults to None.

    :return <CertAuth>: CertAuth object.

    :raises <ValueError>: if parameters are not of the correct type.
    :raises <Exception>: other exceptions may be raised by underlying functions.
    """

    def __init__(
        self,
        username: str,
        password: str,
        crt: str = "api_macrosynergy_com.crt",
        key: str = "api_macrosynergy_com.key",
        base_url: str = CERT_BASE_URL,
        proxy: Optional[dict] = None,
    ):
        try:
            error_user = f"username must be a <str> and not {type(username)}."
            assert isinstance(username, str), error_user
            error_password = f"password must be a <str> and not {type(password)}."
            assert isinstance(password, str), error_password
            error_crt = f"crt must be a <str> and not {type(crt)}."
            assert isinstance(crt, str), error_crt
            error_key = f"key must be a <str> and not {type(key)}."
            assert isinstance(key, str), error_key
            error_base_url = f"base_url must be a <str> and not {type(base_url)}."
            assert isinstance(base_url, str), error_base_url
            error_proxy = f"proxy must be a <dict> and not {type(proxy)}."
            assert isinstance(proxy, dict) or proxy is None, error_proxy
        except AssertionError as e:
            raise ValueError(e)
        except Exception as e:
            raise e

        self.auth: str = base64.b64encode(
            bytes(f"{username:s}:{password:s}", "utf-8")
        ).decode("ascii")

        self.headers: Dict[str, str] = {"Authorization": f"Basic {self.auth:s}"}
        self.base_url: str = base_url
        self.proxy: Optional[dict] = proxy

        # Key and Certificate check
        for f in [key, crt]:
            if not os.path.isfile(f):
                raise FileNotFoundError(f"The file '{f}' does not exist.")
        self.key: str = key
        self.crt: str = crt

    def _request(
        self,
        url: str,
        params: dict = None,
        proxy: Optional[dict] = None,
        tracking_id: Optional[str] = None,
    ) -> dict:
        """
        Wrapper for request_wrapper to use the relevant certificate and headers.

        :param <str> url: URL to request.
        :param <dict> params: parameters to pass to the request.
        :param <dict> proxy: proxy to use for the request.
        :param <str> tracking_id: tracking ID to use for the request (for logging).

        :return <dict>: JSON response from the request.
        """

        js = request_wrapper(
            url=url,
            cert=(self.crt, self.key),
            headers=self.headers,
            params=params,
            proxies=proxy,
            tracking_id=tracking_id,
        )
        return js


class DataQueryInterface(object):
    """
    High level interface for the DataQuery API.
    Uses one of the CertAuth or the OAuth object to allow for authentication.

    :param <bool> oauth: whether to use OAuth authentication. Defaults to True.
    :param <bool> debug: whether to print debug messages. Defaults to False.
    :param <bool> concurrent: whether to use concurrent requests. Defaults to True.
    :param <int> batch_size: number of expressions to send in a single request. Defaults to API_EXPR_LIMIT.
    :param <bool> heartbeat: whether to send a heartbeat request. Defaults to True.
    :param <bool> suppress_warnings: whether to suppress warnings. Defaults to True.

    When using OAuth authentication, the following parameters are used:
    :param <str> client_id: client ID for the DataQuery API.
    :param <str> client_secret: client secret for the DataQuery API.
    :param <str> base_url: base URL for the DataQuery API. Defaults to OAUTH_BASE_URL.
    :param <dict> proxy: proxy to use for requests. Defaults to None.

    When using certificate based authentication, the following parameters are used:
    :param <str> username: username for the DataQuery API.
    :param <str> password: password for the DataQuery API.
    :param <str> crt: path to the certificate file. Defaults to "api_macrosynergy_com.crt".
    :param <str> key: path to the key file. Defaults to "api_macrosynergy_com.key".
    :param <str> base_url: base URL for the DataQuery API. Defaults to CERT_BASE_URL.
    :param <dict> proxy: proxy to use for requests. Defaults to None.

    :return <DataQueryInterface>: DataQueryInterface object.

    :raises <AssertionError>: if the parameters are not valid for the chosen
        authentication method.
    :raises <InvalidResponseError>: if the response from the server is not valid.
    :raises <DownloadError>: if the download fails to complete after a number of retries.
    :raises <HeartbeatError>: if the heartbeat (check connection) fails.
    :raises <Exception>: other exceptions may be raised by underlying functions.
    """

    def __init__(
        self,
        oauth: bool = True,
        debug: bool = False,
        concurrent: bool = True,
        batch_size: int = API_EXPR_LIMIT,
        heartbeat: bool = True,
        suppress_warnings: bool = True,
        **kwargs,
    ):
        self.proxy = kwargs.pop("proxy", kwargs.pop("proxies", None))
        self.heartbeat: bool = heartbeat
        self.msg_errors: List[str] = []
        self.msg_warnings: List[str] = []
        self.debug: bool = debug
        self.concurrent: bool = concurrent
        self.suppress_warnings: bool = suppress_warnings
        self.batch_size: int = batch_size

        if oauth:
            # ensure that we have a client_id and client_secret
            try:
                for k in ["client_id", "client_secret"]:
                    assert k in kwargs, f"{k} must be provided."
            except AssertionError as e:
                raise ValueError(e)
            except Exception as e:
                raise e

            self.access_method: OAuth = OAuth(
                client_id=kwargs["client_id"],
                client_secret=kwargs["client_secret"],
                base_url=OAUTH_BASE_URL,
                token_url=OAUTH_TOKEN_URL,
                dq_resource_id=OAUTH_DQ_RESOURCE_ID,
                proxy=self.proxy,
            )

        else:
            # ensure that we have a username and password, crt and key
            try:
                for k in ["username", "password", "crt", "key"]:
                    assert k in kwargs, f"{k} must be provided."
            except AssertionError as e:
                raise ValueError(e)
            except Exception as e:
                raise e

            self.access_method: CertAuth = CertAuth(
                username=kwargs["username"],
                password=kwargs["password"],
                crt=kwargs["crt"],
                key=kwargs["key"],
                base_url=CERT_BASE_URL,
                proxy=self.proxy,
            )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type:
            print(f"Exception: {exc_type} {exc_value}")
        return True

    def check_connection(self, verbose=False) -> bool:
        """
        Check the connection to the DataQuery API using the Heartbeat endpoint.

        :param <bool> verbose: whether to print a message if the heartbeat
            is successful. Useful for debugging. Defaults to False.

        :return <bool>: True if the connection is successful, False otherwise.
        """

        js = self.access_method._request(
            url=self.access_method.base_url + HEARTBEAT_ENDPOINT,
            params={"data": "NO_REFERENCE_DATA"},
            proxy=self.proxy,
            tracking_id="heartbeat",
        )
        # if "info" not in js:
        #   raise ConnectionError(HeartbeatError("Heartbeat failed."))
        result = "info" in js
        if verbose:
            print("Heartbeat successful!" if result else "Heartbeat failed.")
        return result

    def _request_thread(
        self,
        url: str,
        params: dict = None,
        proxy: Optional[dict] = None,
        tracking_id: Optional[str] = None,
    ) -> List[Dict]:
        """
        Make a request to the DataQuery API using the specified parameters.
        Used to wrap a request in a thread for concurrent requests, or to
        simplify the code for single requests.

        :param <str> url: URL to request.
        :param <dict> params: parameters to send with the request.
        :param <dict> proxy: proxy to use for the request.
        :param <str> tracking_id: tracking ID to use for the request.

        :return <List[Dict]>: list of dictionaries containing the response data.

        :raises <InvalidResponseError>: if the response from the server is not valid.
        :raises <Exception>: other exceptions may be raised by underlying functions.
        """

        downloaded_data: List[Dict] = []
        curr_response: Dict = {}
        curr_url: str = url
        current_params: Dict = params.copy()
        get_pagination: bool = True
        log_url: str = form_full_url(curr_url, current_params)

        while get_pagination:
            curr_response: Dict = self.access_method._request(
                url=url,
                params=params,
                proxy=proxy,
                tracking_id=tracking_id,
            )
            if (curr_response is None) or ("instruments" not in curr_response.keys()):
                raise InvalidResponseError(
                    f"Invalid response from DataQuery: {curr_response}\n"
                    f"URL: {log_url}"
                    f"Timestamp (UTC): {datetime.utcnow().isoformat()}"
                )
            else:
                downloaded_data.extend(curr_response["instruments"])
                if "links" in curr_response.keys():
                    if curr_response["links"][1]["next"] is None:
                        get_pagination = False
                        break
                    else:
                        curr_url = OAUTH_BASE_URL + curr_response["links"][1]["next"]
                        current_params = {}
                        log_url = form_full_url(curr_url, current_params)

        return downloaded_data

    def get_catalogue(self):
        """
        Method to get the JPMaQS catalogue.
        Not yet implemented.
        """
        raise NotImplementedError("This method has not been implemented yet.")

    def filter_exprs_from_catalogue(self, expressions: List[str]) -> List[str]:
        """
        Method to filter a list of expressions against the JPMaQS catalogue.
        Would avoid unnecessary calls or passing invalid expressions to the API.

        Not yet implemented.
        """
        raise NotImplementedError("This method has not been implemented yet.")

    def validate_download_args(
        self,
        expressions: List[str],
        start_date: str = "2000-01-01",
        end_date: str = None,
        show_progress: bool = False,
        endpoint: str = TIMESERIES_ENDPOINT,
        calender: str = "CAL_ALLDAYS",
        frequency: str = "FREQ_DAY",
        conversion: str = "CONV_LASTBUS_ABS",
        nan_treatment: str = "NA_NOTHING",
        reference_data: str = "NO_REFERENCE_DATA",
        retry_counter: int = 0,
        delay_param: int = API_DELAY_PARAM,
        tracking_id: str = None,
    ):
        """
        Validate the arguments passed to the download_data method.

        :params -- see download_data method.

        :returns True if all arguments are valid.

        :raises <ValueError>: if any arguments are of invalid type or value.
        :raises <Exception>: any other exceptions may be raised.
        """

        def is_valid_date(date: str) -> bool:
            """
            Check if a date is in the ISO-8601 format (YYYY-MM-DD).

            :param <str> date: date to check.

            :return <bool>: True if the date is valid, False otherwise.
            """
            try:
                datetime.strptime(date, "%Y-%m-%d")
                return True
            except ValueError:
                return False

        try:
            assert isinstance(
                expressions, list
            ), "`expressions` must be a list of strings."
            assert all(
                isinstance(expr, str) for expr in expressions
            ), "`expressions` must be a list of strings."
            assert isinstance(start_date, str), "`start_date` must be a string."
            assert isinstance(end_date, str), "`end_date` must be a string."
            assert is_valid_date(start_date), (
                "`start_date` must be a string in " "the ISO-8601 format (YYYY-MM-DD)."
            )
            assert is_valid_date(end_date), (
                "`end_date` must be a string in " "the ISO-8601 format (YYYY-MM-DD)."
            )
            assert isinstance(show_progress, bool), "`show_progress` must be a boolean."
            assert isinstance(endpoint, str), "`endpoint` must be a string."
            assert isinstance(calender, str), "`calender` must be a string."
            assert isinstance(frequency, str), "`frequency` must be a string."
            assert isinstance(conversion, str), "`conversion` must be a string."
            assert isinstance(nan_treatment, str), "`nan_treatment` must be a string."
            assert isinstance(reference_data, str), "`reference_data` must be a string."
            assert isinstance(retry_counter, int), "`retry_counter` must be an integer."
            assert isinstance(delay_param, float), "`delay_param` must be an integer."
            assert isinstance(tracking_id, str) or (
                tracking_id is None
            ), "`tracking_id` must be a string or None."
        except AssertionError as e:
            raise ValueError(e)
        except Exception as e:
            raise e

        return True

    def download_data(
        self,
        expressions: List[str],
        start_date: str = "2000-01-01",
        end_date: str = None,
        show_progress: bool = False,
        endpoint: str = TIMESERIES_ENDPOINT,
        calender: str = "CAL_ALLDAYS",
        frequency: str = "FREQ_DAY",
        conversion: str = "CONV_LASTBUS_ABS",
        nan_treatment: str = "NA_NOTHING",
        reference_data: str = "NO_REFERENCE_DATA",
        retry_counter: int = 0,
        delay_param: int = API_DELAY_PARAM,
        tracking_id: str = None,
        # filter_from_catalogue: bool = True,
    ):
        """
        Download data from the DataQuery API.

        :param <List[str]> expressions: list of expressions to download.
        :param <str> start_date: start date for the data in the ISO-8601 format
            (YYYY-MM-DD).
        :param <str> end_date: end date for the data in the ISO-8601 format
            (YYYY-MM-DD).
        :param <bool> show_progress: whether to show a progress bar for the download.
        :param <str> endpoint: endpoint to use for the download.
        :param <str> calender: calendar setting to use for the download.
        :param <str> frequency: frequency of data points to use for the download.
        :param <str> conversion: conversion setting to use for the download.
        :param <str> nan_treatment: NaN treatment setting to use for the download.
        :param <str> reference_data: reference data to pass to the API kwargs.
        :param <int> retry_counter: number of times the download has been retried.
        :param <int> delay_param: delay between requests to the API.
        :param <str> tracking_id: Optional tracking ID to use for the download
            (used for debugging purposes)(default : YYYYMMDD_HHMMSS-OS.PID).

        :return <List[Dict]>: list of dictionaries containing the response data.

        :raises <ValueError>: if any arguments are invalid or semantically incorrect
            (see validate_download_args()).
        :raises <DownloadError>: if the download fails.
        :raises <ConnectionError(HeartbeatError)>: if the heartbeat fails.
        :raises <Exception>: other exceptions may be raised by underlying functions.
        """

        if not self.validate_download_args(
            expressions=expressions,
            start_date=start_date,
            end_date=end_date,
            show_progress=show_progress,
            endpoint=endpoint,
            calender=calender,
            frequency=frequency,
            conversion=conversion,
            nan_treatment=nan_treatment,
            reference_data=reference_data,
            retry_counter=retry_counter,
            delay_param=delay_param,
            tracking_id=tracking_id,
        ):
            raise ValueError("Invalid arguments passed to download_data method.")

        if end_date is None:
            end_date = datetime.today().strftime("%Y-%m-%d")
            # if "future dates" are passed, they must be passed by parent functions
            # see jpmaqs.py

        # remove dashes from dates to match DQ format
        start_date = start_date.replace("-", "")
        end_date = end_date.replace("-", "")

        if tracking_id is None:
            tracking_id = f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}-{os.getpid()}"

        if retry_counter > HL_RETRY_COUNT:
            raise DownloadError(
                f"Failed {retry_counter} times to download data all requested data.\n"
                f"No longer retrying."
            )

        if self.heartbeat:
            if not self.check_connection():
                raise ConnectionError(
                    HeartbeatError(
                        f"Heartbeat failed. Timestamp (UTC):"
                        f" {datetime.utcnow().isoformat()}"
                    )
                )
            time.sleep(API_DELAY_PARAM)

        params_dict: Dict = {
            "format": "JSON",
            "start-date": start_date,
            "end-date": end_date,
            "calendar": calender,
            "frequency": frequency,
            "conversion": conversion,
            "nan_treatment": nan_treatment,
            "data": reference_data,
        }

        # if filter_from_catalogue:
        #     expressions = self.filter_exprs_from_catalogue(expressions)

        expr_batches: List[List[str]] = [
            expressions[i : min(i + self.batch_size, len(expressions))]
            for i in range(0, len(expressions), self.batch_size)
        ]

        download_outputs: List[List[Dict]] = []
        failed_batches: List[List[str]] = []

        if self.concurrent:
            future_objects: List[concurrent.futures.Future] = []

            with concurrent.futures.ThreadPoolExecutor() as executor:
                for ib, expr_batch in tqdm(
                    enumerate(expr_batches),
                    desc="Requesting data",
                    disable=not show_progress,
                    total=len(expr_batches),
                ):
                    curr_params: Dict = params_dict.copy()
                    curr_params["expressions"] = expr_batch
                    future_objects.append(
                        executor.submit(
                            self._request_thread,
                            url=self.access_method.base_url + endpoint,
                            params=curr_params,
                            proxy=self.proxy,
                            tracking_id=f"request_{ib}",
                        )
                    )
                for ib, future in tqdm(
                    enumerate(future_objects),
                    desc="Downloading data",
                    disable=not show_progress,
                    total=len(future_objects),
                ):
                    try:
                        result = future.result()
                        download_outputs.append(result)
                    except Exception as exc:
                        if isinstance(exc, (KeyboardInterrupt, AuthenticationError)):
                            raise exc
                        else:
                            failed_batches.append(expr_batches[ib])
                            self.msg_errors.append(
                                f"Batch {ib} failed with exception: {exc}"
                            )
                            if self.debug:
                                raise exc
                            else:
                                continue

        else:
            for ib, expr_batch in tqdm(
                enumerate(expr_batches),
                desc="Requesting data",
                disable=not show_progress,
            ):
                curr_params: Dict = params_dict.copy()
                curr_params["expressions"] = expr_batch
                try:
                    result = self._request_thread(
                        url=self.access_method.base_url + endpoint,
                        params=curr_params,
                        proxy=self.proxy,
                        tracking_id=f"request_{ib}",
                    )
                    download_outputs.append(result)
                except Exception as exc:
                    if isinstance(exc, (KeyboardInterrupt, AuthenticationError)):
                        raise exc
                    else:
                        failed_batches.append(expr_batch)
                        self.msg_errors.append(
                            f"Batch {ib} failed with exception: {exc}"
                        )
                        if self.debug:
                            raise exc
                        else:
                            continue

        if len(failed_batches) > 0:
            flat_failed_batches: List[str] = list(
                itertools.chain.from_iterable(failed_batches)
            )
            self.download_data(
                expressions=flat_failed_batches,
                start_date=start_date,
                end_date=end_date,
                show_progress=show_progress,
                endpoint=endpoint,
                calender=calender,
                frequency=frequency,
                conversion=conversion,
                nan_treatment=nan_treatment,
                reference_data=reference_data,
                retry_counter=retry_counter + 1,
                delay_param=delay_param + 0.1,
            )

        final_output: List[Dict] = list(itertools.chain.from_iterable(download_outputs))
        return final_output


if __name__ == "__main__":
    import os

    client_id = os.environ["JPMAQS_API_CLIENT_ID"]
    client_secret = os.environ["JPMAQS_API_CLIENT_SECRET"]

    expressions = [
        "DB(JPMAQS,USD_EQXR_VT10,value)",
        "DB(JPMAQS,AUD_EXALLOPENNESS_NSA_1YMA,value)",
    ]
    start_date: str = "2020-01-25"
    end_date: str = "2023-02-05"

    with DataQueryInterface(
        client_id=client_id,
        client_secret=client_secret,
    ) as dq:
        assert dq.check_connection(verbose=True)

        data = dq.download_data(
            expressions=expressions,
            start_date=start_date,
            end_date=end_date,
            show_progress=True,
        )

    print(f"Succesfully downloaded data for {len(data)} expressions.")
