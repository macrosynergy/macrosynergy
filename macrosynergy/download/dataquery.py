"""
Interface for downloading data from the JPMorgan DataQuery API.
This module is not intended to be used directly, but rather through
macrosynergy.download.jpmaqs.py. However, for a use cases independent
of JPMaQS, this module can be used directly to download data from the
JPMorgan DataQuery API.

::docs::DataQueryInterface::sort_first::
"""
import concurrent.futures
import time
import os
import logging
import itertools
import base64
import uuid
import io
import warnings
import requests
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Union, Tuple
from timeit import default_timer as timer
from tqdm import tqdm

from macrosynergy import __version__ as ms_version_info
from macrosynergy.download.exceptions import (
    AuthenticationError,
    DownloadError,
    InvalidResponseError,
    HeartbeatError,
    NoContentError,
    KNOWN_EXCEPTIONS,
)
from macrosynergy.management.utils import (
    is_valid_iso_date,
    form_full_url,
)

CERT_BASE_URL: str = "https://platform.jpmorgan.com/research/dataquery/api/v2"
OAUTH_BASE_URL: str = (
    "https://api-developer.jpmorgan.com/research/dataquery-authe/api/v2"
)
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

logger: logging.Logger = logging.getLogger(__name__)
debug_stream_handler = logging.StreamHandler(io.StringIO())
debug_stream_handler.setLevel(logging.NOTSET)
debug_stream_handler.setFormatter(
    logging.Formatter(
        "%(asctime)s - %(levelname)s - %(module)s - %(funcName)s :: %(message)s"
    )
)
logger.addHandler(debug_stream_handler)

def validate_response(
    response: requests.Response,
    user_id: str,
) -> dict:
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

    error_str: str = (
        f"Response: {response}\n"
        f"User ID: {user_id}\n"
        f"Requested URL: {response.request.url}\n"
        f"Response status code: {response.status_code}\n"
        f"Response headers: {response.headers}\n"
        f"Response text: {response.text}\n"
        f"Timestamp (UTC): {datetime.utcnow().isoformat()}; \n"
    )
    # TODO : Use response.raise_for_status() as a better way to check for errors
    if not response.ok:
        logger.info("Response status is NOT OK : %s", response.status_code)
        if response.status_code == 401:
            raise AuthenticationError(error_str)

        if HEARTBEAT_ENDPOINT in response.request.url:
            raise HeartbeatError(error_str)

        raise InvalidResponseError(
            f"Request did not return a 200 status code.\n{error_str}"
        )

    try:
        response_dict = response.json()
        if response_dict is None:
            raise InvalidResponseError(f"Response is empty.\n{error_str}")
        return response_dict
    except Exception as exc:
        if isinstance(exc, KeyboardInterrupt):
            raise exc

        raise InvalidResponseError(error_str + f"Error parsing response as JSON: {exc}")


def request_wrapper(
    url: str,
    headers: Optional[Dict] = None,
    params: Optional[Dict] = None,
    method: str = "get",
    tracking_id: Optional[str] = None,
    proxy: Optional[Dict] = None,
    cert: Optional[Tuple[str, str]] = None,
    **kwargs,
) -> dict:
    """
    Wrapper for requests.request() that handles retries and logging.
    All parameters and kwargs are passed to requests.request().

    :param <str> url: URL to request.
    :param <dict> headers: headers to pass to requests.request().
    :param <dict> params: params to pass to requests.request().
    :param <str> method: HTTP method to use. Must be one of "get"
        or "post". Defaults to "get".
    :param <dict> kwargs: kwargs to pass to requests.request().
    :param <str> tracking_id: default None, unique tracking ID of request.
    :param <dict> proxy: default None, dictionary of proxy settings for request.
    :param <Tuple[str, str]> cert: default None, tuple of string for filename
        of certificate and key.

    :return <dict>: response as a dictionary.

    :raises <InvalidResponseError>: if the response is not valid.
    :raises <AuthenticationError>: if the response is a 401 status code.
    :raises <DownloadError>: if the request fails after retrying.
    :raises <KeyboardInterrupt>: if the user interrupts the download.
    :raises <ValueError>: if the method is not one of "get" or "post".
    :raises <Exception>: other exceptions may be raised by requests.request().
    """

    if method not in ["get", "post"]:
        raise ValueError(f"Invalid method: {method}")

    user_id: str = kwargs.pop("user_id", "unknown")

    # insert tracking info in headers
    if headers is None:
        headers: Dict = {}
    headers["User-Agent"]: str = f"MacrosynergyPackage/{ms_version_info}"

    uuid_str: str = str(uuid.uuid4())
    if (tracking_id is None) or (tracking_id == ""):
        tracking_id: str = uuid_str
    else:
        tracking_id: str = f"uuid::{uuid_str}::{tracking_id}"

    headers["X-Tracking-Id"]: str = tracking_id

    log_url: str = form_full_url(url, params)
    logger.debug(f"Requesting URL: {log_url} with tracking_id: {tracking_id}")
    raised_exceptions: List[Exception] = []
    error_statements: List[str] = []
    error_statement: str = ""
    retry_count: int = 0
    response: Optional[requests.Response] = None
    while retry_count < API_RETRY_COUNT:
        try:
            prepared_request: requests.PreparedRequest = requests.Request(
                method, url, headers=headers, params=params, **kwargs
            ).prepare()

            response: requests.Response = requests.Session().send(
                prepared_request,
                proxies=proxy,
                cert=cert,
            )

            if isinstance(response, requests.Response):
                return validate_response(response=response, user_id=user_id)

        except Exception as exc:
            # if keyboard interrupt, raise as usual
            if isinstance(exc, KeyboardInterrupt):
                print("KeyboardInterrupt -- halting download")
                raise exc

            # authentication error, clearly not a transient error
            if isinstance(exc, AuthenticationError):
                raise exc

            error_statement = (
                f"Request to {log_url} failed with error {exc}. "
                f"User ID: {user_id}. "
                f"Retry count: {retry_count}. "
                f"Tracking ID: {tracking_id}"
            )
            raised_exceptions.append(exc)
            error_statements.append(error_statement)

            known_exceptions = KNOWN_EXCEPTIONS + [HeartbeatError]
            # NOTE : HeartBeat is a special case

            # NOTE: exceptions that need the code to break should be caught before this
            # all other exceptions are caught here and retried after a delay

            if any([isinstance(exc, e) for e in known_exceptions]):
                logger.warning(error_statement)
                retry_count += 1
                time.sleep(API_DELAY_PARAM)
            else:
                raise exc

    if isinstance(raised_exceptions[-1], HeartbeatError):
        raise HeartbeatError(error_statement)

    errs_str = "\n\n".join(
        ("\t" + str(e) + " - \n\t\t" + est)
        for e, est in zip(raised_exceptions, error_statements)
    )

    e_str = f"Request to {log_url} failed with error {raised_exceptions[-1]}. \n"
    e_str += "-" * 20 + "\n"
    if isinstance(response, requests.Response):
        e_str += f" Status code: {response.status_code}."
    e_str += (
        f" No longer retrying. Tracking ID: {tracking_id}"
        f"Exceptions raised:\n{errs_str}"
    )

    raise DownloadError(e_str)


class OAuth(object):
    """
    Class for handling OAuth authentication for the DataQuery API.

    :param <str> client_id: client ID for the OAuth application.
    :param <str> client_secret: client secret for the OAuth application.
    :param <dict> proxy: proxy to use for requests. Defaults to None.
    :param <str> token_url: URL for getting OAuth tokens.
    :param <str> dq_resource_id: resource ID for the JPMaQS Application.

    :return <OAuth>: OAuth object.

    :raises <ValueError>: if any of the parameters are semantically incorrect.
    :raises <TypeError>: if any of the parameters are of the wrong type.
    :raises <Exception>: other exceptions may be raised by underlying functions.
    """

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        proxy: Optional[dict] = None,
        token_url: str = OAUTH_TOKEN_URL,
        dq_resource_id: str = OAUTH_DQ_RESOURCE_ID,
    ):
        logger.debug("Instantiate OAuth pathway to DataQuery")
        vars_types_zip: zip = zip(
            [client_id, client_secret, token_url, dq_resource_id],
            [
                "client_id",
                "client_secret",
                "token_url",
                "dq_resource_id",
            ],
        )

        for varx, namex in vars_types_zip:
            if not isinstance(varx, str):
                raise TypeError(f"{namex} must be a <str> and not {type(varx)}.")

        if not isinstance(proxy, dict) and proxy is not None:
            raise TypeError(f"proxy must be a <dict> and not {type(proxy)}.")

        self.token_url: str = token_url
        self.proxy: Optional[dict] = proxy

        self._stored_token: Optional[dict] = None
        self.token_data = {
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
            "aud": dq_resource_id,
        }

    def _valid_token(self) -> bool:
        """
        Method to check if the stored token is valid.

        :return <bool>: True if the token is valid, False otherwise.
        """
        if self._stored_token is None:
            logger.debug("No token stored")
            return False

        created: datetime = self._stored_token["created_at"]  # utc time of creation
        expires: datetime = created + timedelta(
            seconds=self._stored_token["expires_in"]
        )

        utcnow = datetime.utcnow()
        is_active: bool = expires > utcnow

        logger.debug(
            "Active token: %s, created: %s, expires: %s, now: %s",
            is_active,
            created,
            expires,
            utcnow,
        )

        return is_active

    def _get_token(self) -> str:
        """Method to get a new OAuth token.

        :return <str>: OAuth token.
        """
        if not self._valid_token():
            logger.debug("Request new OAuth token")
            js = request_wrapper(
                url=self.token_url,
                data=self.token_data,
                method="post",
                proxy=self.proxy,
                tracking_id=OAUTH_TRACKING_ID,
                user_id=self._get_user_id(),
            )
            # on failure, exception will be raised by request_wrapper

            # NOTE : use UTC time for token expiry
            self._stored_token: dict = {
                "created_at": datetime.utcnow(),
                "access_token": js["access_token"],
                "expires_in": js["expires_in"],
            }

        return self._stored_token["access_token"]

    def _get_user_id(self) -> str:
        return "OAuth_ClientID - " + self.token_data["client_id"]

    def get_auth(self) -> Dict[str, Union[str, Optional[Tuple[str, str]]]]:
        """
        Returns a dictionary with the authentication information, in the same
        format as the `macrosynergy.download.dataquery.CertAuth.get_auth()` method.
        """
        headers: Dict = {"Authorization": "Bearer " + self._get_token()}
        return {
            "headers": headers,
            "cert": None,
            "user_id": self._get_user_id(),
        }


class CertAuth(object):
    """
    Class for handling certificate based authentication for the DataQuery API.

    :param <str> username: username for the DataQuery API.
    :param <str> password: password for the DataQuery API.
    :param <str> crt: path to the certificate file.
    :param <str> key: path to the key file.

    :return <CertAuth>: CertAuth object.

    :raises <AssertionError>: if any of the parameters are of the wrong type.
    :raises <FileNotFoundError>: if certificate or key file is missing from filesystem.
    :raises <Exception>: other exceptions may be raised by underlying functions.
    """

    def __init__(
        self,
        username: str,
        password: str,
        crt: str,
        key: str,
        proxy: Optional[dict] = None,
    ):
        for varx, namex in zip([username, password], ["username", "password"]):
            if not isinstance(varx, str):
                raise TypeError(f"{namex} must be a <str> and not {type(varx)}.")

        self.auth: str = base64.b64encode(
            bytes(f"{username:s}:{password:s}", "utf-8")
        ).decode("ascii")

        # Key and Certificate check
        for varx, namex in zip([crt, key], ["crt", "key"]):
            if not isinstance(varx, str):
                raise TypeError(f"{namex} must be a <str> and not {type(varx)}.")
            if not os.path.isfile(varx):
                raise FileNotFoundError(f"The file '{varx}' does not exist.")
        self.key: str = key
        self.crt: str = crt
        self.username: str = username
        self.password: str = password
        self.proxy: Optional[dict] = proxy

    def get_auth(self) -> Dict[str, Union[str, Optional[Tuple[str, str]]]]:
        """
        Returns a dictionary with the authentication information, in the same
        format as the `macrosynergy.download.dataquery.OAuth.get_auth()` method.
        """
        headers = {"Authorization": f"Basic {self.auth:s}"}
        user_id = "CertAuth_Username - " + self.username
        return {
            "headers": headers,
            "cert": (self.crt, self.key),
            "user_id": user_id,
        }


def validate_download_args(
    expressions: List[str],
    start_date: str,
    end_date: str,
    show_progress: bool,
    endpoint: str,
    calender: str,
    frequency: str,
    conversion: str,
    nan_treatment: str,
    reference_data: str,
    retry_counter: int,
    delay_param: float,
):
    """
    Validate the arguments passed to the `download_data()` method.

    :params : -- see `download_data()` method.

    :return <bool>: True if all arguments are valid.

    :raises <TypeError>: if any of the arguments are of the wrong type.
    :raises <ValueError>: if any of the arguments are semantically incorrect.
    """

    if expressions is None:
        raise ValueError("`expressions` must be a list of strings.")

    if not isinstance(expressions, list):
        raise TypeError("`expressions` must be a list of strings.")

    if not all(isinstance(expr, str) for expr in expressions):
        raise TypeError("`expressions` must be a list of strings.")

    for varx, namex in zip([start_date, end_date], ["start_date", "end_date"]):
        if (varx is None) or not isinstance(varx, str):
            raise TypeError(f"`{namex}` must be a string.")
        if not is_valid_iso_date(varx):
            raise ValueError(
                f"`{namex}` must be a string in the ISO-8601 format (YYYY-MM-DD)."
            )

    if not isinstance(show_progress, bool):
        raise TypeError("`show_progress` must be a boolean.")

    if not isinstance(retry_counter, int):
        raise TypeError("`retry_counter` must be an integer.")

    if not isinstance(delay_param, float):
        raise TypeError("`delay_param` must be a float >=0.2 (seconds).")
    elif delay_param < 0.2:
        raise ValueError("`delay_param` must be a float >=0.2 (seconds).")

    vars_types_zip: zip = zip(
        [
            endpoint,
            calender,
            frequency,
            conversion,
            nan_treatment,
            reference_data,
        ],
        [
            "endpoint",
            "calender",
            "frequency",
            "conversion",
            "nan_treatment",
            "reference_data",
        ],
    )
    for varx, namex in vars_types_zip:
        if not isinstance(varx, str):
            raise TypeError(f"`{namex}` must be a string.")

    return True


def _get_unavailable_expressions(
    expected_exprs: List[str],
    dicts_list: List[Dict],
) -> List[str]:
    """
    Method to get the expressions that are not available in the response.
    Looks at the dict["attributes"][0]["expression"] field of each dict
    in the list.

    :param <List[str]> expected_exprs: list of expressions that were requested.
    :param <List[Dict]> dicts_list: list of dicts to search for the expressions.

    :return <List[str]>: list of expressions that were not found in the dicts.
    """
    found_exprs: List[str] = [
        curr_dict["attributes"][0]["expression"]
        for curr_dict in dicts_list
        if curr_dict["attributes"][0]["time-series"] is not None
    ]
    return list(set(expected_exprs) - set(found_exprs))


class DataQueryInterface(object):
    """
    High level interface for the DataQuery API.

    When using OAuth authentication:

    :param <str> client_id: client ID for the OAuth application.
    :param <str> client_secret: client secret for the OAuth application.

    When using certificate authentication:

    :param <str> crt: path to the certificate file.
    :param <str> key: path to the key file.
    :param <str> username: username for the DataQuery API.
    :param <str> password: password for the DataQuery API.

    :param <bool> oauth: whether to use OAuth authentication. Defaults to True.
    :param <bool> debug: whether to print debug messages. Defaults to False.
    :param <bool> concurrent: whether to use concurrent requests. Defaults to True.
    :param <int> batch_size: default 20, number of expressions to send in a single
        request. Must be a number between 1 and 20 (both included).
    :param <bool> check_connection: whether to send a check_connection request.
        Defaults to True.
    :param <str> base_url: base URL for the DataQuery API. Defaults to OAUTH_BASE_URL
        if `oauth` is True, CERT_BASE_URL otherwise.
    :param <str> token_url: token URL for the DataQuery API. Defaults to OAUTH_TOKEN_URL.
    :param <bool> suppress_warnings: whether to suppress warnings. Defaults to True.

    :return <DataQueryInterface>: DataQueryInterface object.

    :raises <TypeError>: if any of the parameters are of the wrong type.
    :raises <ValueError>: if any of the parameters are semantically incorrect.
    :raises <InvalidResponseError>: if the response from the server is not valid.
    :raises <DownloadError>: if the download fails to complete after a number of retries.
    :raises <HeartbeatError>: if the heartbeat (check connection) fails.
    :raises <Exception>: other exceptions may be raised by underlying functions.
    """

    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        crt: Optional[str] = None,
        key: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        proxy: Optional[dict] = None,
        oauth: bool = True,
        debug: bool = False,
        batch_size: int = 20,
        check_connection: bool = True,
        base_url: str = OAUTH_BASE_URL,
        token_url: str = OAUTH_TOKEN_URL,
        suppress_warnings: bool = True,
    ):
        self._check_connection: bool = check_connection
        self.msg_errors: List[str] = []
        self.msg_warnings: List[str] = []
        self.unavailable_expressions: List[str] = []
        self.debug: bool = debug
        self.suppress_warnings: bool = suppress_warnings
        self.batch_size: int = batch_size

        for varx, namex, typex in [
            (client_id, "client_id", str),
            (client_secret, "client_secret", str),
            (crt, "crt", str),
            (key, "key", str),
            (username, "username", str),
            (password, "password", str),
            (proxy, "proxy", dict),
        ]:
            if not isinstance(varx, typex) and varx is not None:
                raise TypeError(f"{namex} must be a {typex} and not {type(varx)}.")

        self.auth: Optional[Union[CertAuth, OAuth]] = None
        if oauth and not all([client_id, client_secret]):
            warnings.warn(
                "OAuth authentication requested but client ID and/or client secret "
                "not found. Falling back to certificate authentication.",
                UserWarning,
            )
            if not all([username, password, crt, key]):
                raise ValueError(
                    "Certificate credentials not found. "
                    "Check the parameters passed to the DataQueryInterface class."
                )
            else:
                oauth: bool = False

        if oauth:
            self.auth: OAuth = OAuth(
                client_id=client_id,
                client_secret=client_secret,
                token_url=token_url,
                proxy=proxy,
            )
        else:
            if base_url == OAUTH_BASE_URL:
                base_url: str = CERT_BASE_URL

            self.auth: CertAuth = CertAuth(
                username=username,
                password=password,
                crt=crt,
                key=key,
                proxy=proxy,
            )

        assert self.auth is not None, (
            "Unable to instantiate authentication object. "
            "Check the parameters passed to the DataQueryInterface class."
        )

        self.proxy: Optional[dict] = proxy
        self.base_url: str = base_url

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type:
            logger.error("Exception %s - %s", exc_type, exc_value)
            print(f"Exception: {exc_type} {exc_value}")

    def check_connection(self, verbose=False) -> bool:
        """
        Check the connection to the DataQuery API using the Heartbeat endpoint.

        :param <bool> verbose: whether to print a message if the heartbeat
            is successful. Useful for debugging. Defaults to False.

        :return <bool>: True if the connection is successful, False otherwise.

        :raises <HeartbeatError>: if the heartbeat fails.
        """
        logger.debug("Check if connection can be established to JPMorgan DataQuery")
        js: dict = request_wrapper(
            url=self.base_url + HEARTBEAT_ENDPOINT,
            params={"data": "NO_REFERENCE_DATA"},
            proxy=self.proxy,
            tracking_id=HEARTBEAT_TRACKING_ID,
            **self.auth.get_auth(),
        )

        result: bool = True
        if (js is None) or (not isinstance(js, dict)) or ("info" not in js):
            logger.warning("Connection to JPMorgan DataQuery heartbeat failed")
            result: bool = False

        if result:
            result = (int(js["info"]["code"]) == 200) and (
                js["info"]["message"] == "Service Available."
            )

        if verbose:
            print("Connection successful!" if result else "Connection failed.")
        return result

    def _fetch(
        self,
        url: str,
        params: dict = None,
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
        response: Dict = request_wrapper(
            url=url,
            params=params,
            proxy=self.proxy,
            tracking_id=tracking_id,
            **self.auth.get_auth(),
        )

        if (response is None) or ("instruments" not in response.keys()):
            if response is not None:
                if (
                    ("info" in response)
                    and ("code" in response["info"])
                    and (int(response["info"]["code"]) == 204)
                ):
                    raise NoContentError(
                        f"Content was not found for the request: {response}\n"
                        f"User ID: {self.auth.get_auth()['user_id']}\n"
                        f"URL: {form_full_url(url, params)}\n"
                        f"Timestamp (UTC): {datetime.utcnow().isoformat()}"
                    )

            raise InvalidResponseError(
                f"Invalid response from DataQuery: {response}\n"
                f"User ID: {self.auth.get_auth()['user_id']}\n"
                f"URL: {form_full_url(url, params)}"
                f"Timestamp (UTC): {datetime.utcnow().isoformat()}"
            )

        downloaded_data.extend(response["instruments"])

        if "links" in response.keys() and response["links"][1]["next"] is not None:
            logger.info("DQ response paginated - get next response page")
            downloaded_data.extend(
                self._fetch(
                    url=self.base_url + response["links"][1]["next"],
                    params={},
                    tracking_id=tracking_id,
                )
            )

        return downloaded_data

    def get_catalogue(
        self,
        group_id: str = JPMAQS_GROUP_ID,
    ) -> List[str]:
        """
        Method to get the JPMaQS catalogue.
        Queries the DataQuery API's Groups/Search endpoint to get the list of
        tickers in the JPMaQS group. The group ID can be changed to fetch a
        different group's catalogue.

        :param <str> group_id: the group ID to fetch the catalogue for.

        :return <List[str]>: list of tickers in the JPMaQS group.

        :raises <ValueError>: if the response from the server is not valid.
        """
        old_group_id: str = "CA_QI_MACRO_SYNERGY"
        try:
            response_list: Dict = self._fetch(
                url=self.base_url + CATALOGUE_ENDPOINT,
                params={"group-id": group_id},
                tracking_id=CATALOGUE_TRACKING_ID,
            )
        except NoContentError as e:
            response_list: Dict = self._fetch(
                url=self.base_url + CATALOGUE_ENDPOINT,
                params={"group-id": old_group_id},
                tracking_id=CATALOGUE_TRACKING_ID,
            )
        except Exception as e:
            raise e

        tickers: List[str] = [d["instrument-name"] for d in response_list]
        utkr_count: int = len(tickers)
        tkr_idx: List[int] = sorted([d["item"] for d in response_list])

        if not (
            (min(tkr_idx) == 1)
            and (max(tkr_idx) == utkr_count)
            and (len(set(tkr_idx)) == utkr_count)
        ):
            raise ValueError("The downloaded catalogue is corrupt.")

        return tickers

    def _download(
        self,
        expressions: List[str],
        params: dict,
        url: str,
        tracking_id: str,
        delay_param: float,
        show_progress: bool = False,
        retry_counter: int = 0,
    ) -> List[dict]:
        """
        Backend method to download data from the DataQuery API.
        Used by the `download_data()` method.
        """

        if retry_counter > 0:
            print("Retrying failed downloads. Retry count:", retry_counter)

        if retry_counter > HL_RETRY_COUNT:
            raise DownloadError(
                f"Failed {retry_counter} times to download data all requested data.\n"
                f"No longer retrying."
            )

        expr_batches: List[List[str]] = [
            expressions[i : i + self.batch_size]
            for i in range(0, len(expressions), self.batch_size)
        ]

        download_outputs: List[List[Dict]] = []
        failed_batches: List[List[str]] = []
        continuous_failures: int = 0
        last_five_exc: List[Exception] = []

        future_objects: List[concurrent.futures.Future] = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for ib, expr_batch in tqdm(
                enumerate(expr_batches),
                desc="Requesting data",
                disable=not show_progress,
                total=len(expr_batches),
            ):
                curr_params: Dict = params.copy()
                curr_params["expressions"] = expr_batch
                future_objects.append(
                    executor.submit(
                        self._fetch,
                        url=url,
                        params=curr_params,
                        tracking_id=tracking_id,
                    )
                )
                time.sleep(delay_param)

            for ib, future in tqdm(
                enumerate(future_objects),
                desc="Downloading data",
                disable=not show_progress,
                total=len(future_objects),
            ):
                try:
                    download_outputs.append(future.result())
                    continuous_failures = 0
                except Exception as exc:
                    if isinstance(exc, (KeyboardInterrupt, AuthenticationError)):
                        raise exc

                    failed_batches.append(expr_batches[ib])
                    self.msg_errors.append(f"Batch {ib} failed with exception: {exc}")
                    continuous_failures += 1
                    last_five_exc.append(exc)
                    if continuous_failures > MAX_CONTINUOUS_FAILURES:
                        exc_str: str = "\n".join([str(e) for e in last_five_exc])
                        raise DownloadError(
                            f"Failed {continuous_failures} times to download data."
                            f" Last five exceptions: \n{exc_str}"
                        )

                    if self.debug:
                        raise exc

        final_output: List[Dict] = list(itertools.chain.from_iterable(download_outputs))

        if len(failed_batches) > 0:
            flat_failed_batches: List[str] = list(
                itertools.chain.from_iterable(failed_batches)
            )
            logger.warning(
                "Failed batches %d - retry download for %d expressions",
                len(failed_batches),
                len(flat_failed_batches),
            )
            retried_output: List[dict] = self._download(
                expressions=flat_failed_batches,
                params=params,
                url=url,
                tracking_id=tracking_id,
                delay_param=delay_param + 0.1,
                show_progress=show_progress,
                retry_counter=retry_counter + 1,
            )

            final_output += retried_output  # extend retried output

        return final_output

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
        delay_param: float = API_DELAY_PARAM,  # TODO do we want the user to have access to this?
    ) -> List[Dict]:
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
        :param <float> delay_param: delay between requests to the API.

        :return <List[Dict]>: list of dictionaries containing the response data.

        :raises <ValueError>: if any arguments are invalid or semantically incorrect
            (see validate_download_args()).
        :raises <DownloadError>: if the download fails.
        :raises <ConnectionError(HeartbeatError)>: if the heartbeat fails.
        :raises <Exception>: other exceptions may be raised by underlying functions.
        """
        tracking_id: str = TIMESERIES_TRACKING_ID
        if end_date is None:
            end_date = datetime.today().strftime("%Y-%m-%d")
            # NOTE : if "future dates" are passed, they must be passed by parent functions
            # see jpmaqs.py

        # NOTE : args validated only on first call, not on retries
        # this is because the args can be modified by the retry mechanism
        # (eg. date format)

        validate_download_args(
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
        )

        if datetime.strptime(end_date, "%Y-%m-%d") < datetime.strptime(
            start_date, "%Y-%m-%d"
        ):
            logger.warning(
                "Start date (%s) is after end-date (%s): swap them!",
                start_date,
                end_date,
            )
            start_date, end_date = end_date, start_date

        # remove dashes from dates to match DQ format
        start_date: str = start_date.replace("-", "")
        end_date: str = end_date.replace("-", "")

        # check heartbeat before each "batch" of requests
        if self._check_connection:
            if not self.check_connection():
                raise ConnectionError(
                    HeartbeatError(
                        f"Heartbeat failed. Timestamp (UTC):"
                        f" {datetime.utcnow().isoformat()}\n"
                        f"User ID: {self.auth.get_auth()['user_id']}\n"
                    )
                )
            time.sleep(delay_param)

        logger.info(
            "Download %d expressions from DataQuery from %s to %s",
            len(expressions),
            datetime.strptime(start_date, "%Y%m%d").date(),
            datetime.strptime(end_date, "%Y%m%d").date(),
        )
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

        final_output: List[dict] = self._download(
            expressions=expressions,
            params=params_dict,
            url=self.base_url + endpoint,
            tracking_id=tracking_id,
            delay_param=delay_param,
            show_progress=show_progress,
        )

        self.unavailable_expressions = _get_unavailable_expressions(
            expected_exprs=expressions, dicts_list=final_output
        )
        logger.info(
            "Downloaded expressions: %d, unavailable: %d",
            len(final_output),
            len(self.unavailable_expressions),
        )

        return final_output


if __name__ == "__main__":
    import os

    client_id = os.getenv("DQ_CLIENT_ID")
    client_secret = os.getenv("DQ_CLIENT_SECRET")

    expressions = [
        "DB(JPMAQS,USD_EQXR_VT10,value)",
        "DB(JPMAQS,AUD_EXALLOPENNESS_NSA_1YMA,value)",
    ]

    with DataQueryInterface(
        client_id=client_id,
        client_secret=client_secret,
    ) as dq:
        assert dq.check_connection(verbose=True)

        data = dq.download_data(
            expressions=expressions,
            start_date="2020-01-25",
            end_date="2023-02-05",
            show_progress=True,
        )

    print(f"Succesfully downloaded data for {len(data)} expressions.")
