import time
import uuid
import requests, requests.compat
from datetime import datetime
import logging


from typing import List, Optional, Dict, Tuple

from macrosynergy import __version__ as ms_version_info
from macrosynergy.download.exceptions import (
    AuthenticationError,
    DownloadError,
    InvalidResponseError,
    HeartbeatError,
    KNOWN_EXCEPTIONS,
)
from .constants import (
    API_DELAY_PARAM,
    API_RETRY_COUNT,
    HEARTBEAT_ENDPOINT,
)

logger: logging.Logger = logging.getLogger(__name__)


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
