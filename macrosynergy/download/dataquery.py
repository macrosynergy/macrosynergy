"""DataQuery Interface."""
import warnings
import concurrent.futures
import time
import datetime
import logging
from math import ceil, floor
from itertools import chain

import base64
import os
import requests
from typing import List, Optional, Dict, Tuple
from datetime import datetime
from macrosynergy.download.exceptions import *

CERT_BASE_URL: str = "https://platform.jpmorgan.com/research/dataquery/api/v2"
OAUTH_BASE_URL: str = (
    "https://api-developer.jpmorgan.com/research/dataquery-authe/api/v2"
)
OAUTH_TOKEN_URL: str = "https://authe.jpmchase.com/as/token.oauth2"
OAUTH_DQ_RESOURCE_ID: str = "JPMC:URI:RS-06785-DataQueryExternalApi-PROD"

logger = logging.getLogger(__name__)


def valid_response(r: requests.Response) -> Tuple[Optional[dict], bool, Optional[dict]]:
    """
    Prior to requesting any data, the function will confirm if a connection to the
    DataQuery API is able to be established given the credentials passed. If the status
    code is 200, able to access DataQuery's API.
    """
    msg: Optional[dict] = None
    if not r.ok:
        msg: Dict[str, str] = {
            "headers": r.headers,
            "status_code": r.status_code,
            "text": r.text,
            "url": r.url,
        }
        js: Optional[dict] = None

        logger.error(f"Request failed. msg : {msg}")

    else:
        js = r.json()

    return js, r.ok, msg


def dq_request(
    url: str,
    headers: dict = None,
    params: dict = None,
    method: str = "get",
    cert: Optional[Tuple[str, str]] = None,
    **kwargs,
) -> Tuple[Optional[dict], bool, str, Optional[dict]]:
    """Will return the request from DataQuery."""
    request_error = f"Unknown request method {method} not in ('get', 'post')."
    assert method in ("get", "post"), request_error

    log_url = f"{url}?{requests.compat.urlencode(params)}" if params else url
    log_url = requests.compat.quote(log_url, safe="%/:=&?~#+!$,;'@()*[]")
    logger.info(f"Requesting URL: {log_url}")

    with requests.request(
        method=method,
        url=url,
        cert=cert,
        headers=headers,
        params=params,
        **kwargs,
    ) as r:
        last_url: str = r.url
        js, success, msg = valid_response(r=r)

    if not success:
        logger.error(
            f"Request failed for URL: {last_url}"
            f" with message: {msg}"
            f" and response: {js}"
        )
        if msg["status_code"] == 401:
            logger.error(
                f"Invalid credentials."
                f"Request failed for URL: {last_url}"
                f" with message: {msg}"
                f" and response: {js}"
            )
            raise AuthenticationError(
                Exception("Invalid credentials for DataQuery API.")
            )

    return js, success, last_url, msg


class CertAuth(object):
    """Certificate Authentication.

    Class used to access DataQuery via certificate and private key. To access the API
    login both username & password are required as well as a certified certificate and
    private key to verify the request.

    :param <str> username: username for login to REST API for JP Morgan DataQuery.
    :param <str> password: password.
    :param <str> crt: string with location of public certificate.
    :param <str> key: string with private key location.

    """

    def __init__(
        self,
        username: str,
        password: str,
        crt: str = "api_macrosynergy_com.crt",
        key: str = "api_macrosynergy_com.key",
        base_url: str = CERT_BASE_URL,
    ):

        error_user = f"username must be a <str> and not {type(username)}."
        assert isinstance(username, str), error_user

        error_password = f"password must be a <str> and not {type(password)}."
        assert isinstance(password, str), error_password

        self.auth: str = base64.b64encode(
            bytes(f"{username:s}:{password:s}", "utf-8")
        ).decode("ascii")

        self.headers: Dict[str, str] = {"Authorization": f"Basic {self.auth:s}"}
        self.base_url: str = base_url

        # Key and Certificate.
        self.key: str = self.valid_path(key, "key")
        self.crt: str = self.valid_path(crt, "crt")

        # For debugging purposes save last request response.
        self.status_code: Optional[int] = None
        self.last_response: Optional[str] = None
        self.last_url: Optional[str] = None

    @staticmethod
    def valid_path(file_path: str, file_type: str) -> Optional[str]:
        """Validates the key & certificate exist in the referenced directory.

        :param <str> directory: directory hosting the respective files.
        :param <str> file_type: parameter used to distinguish between the certificate or
            key being received.

        """
        assert isinstance(file_path, str), "file_path must be a <str>."
        assert isinstance(file_type, str), "file_type must be a <str>."
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The path '{file_path}' does not exist.")

        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"The path '{file_path}' is not a file.")

        return file_path

    def get_dq_api_result(
        self, url: str, params: dict = None, proxy: Optional[dict] = None
    ) -> dict:
        """Method used exclusively to request data from the API.

        :param <str> url: url to access DQ API.
        :param <dict> params: dictionary containing the required parameters for the
            ticker series.
        :param <dict> proxy: proxy settings for request.
        """
        js, success, self.last_url, msg = dq_request(
            url=url,
            cert=(self.crt, self.key),
            headers=self.headers,
            params=params,
            proxies=proxy,
        )
        self.last_response = {"json": js, "success": success, "msg": msg}
        return js, success, msg


class OAuth(object):
    """Accessing DataQuery via OAuth.

    :param <str> client_id: string with client id, username.
    :param <str> client_secret: string with client secret, password.
    :param <str> url:
    :param <str> token_url:
    :param <str> dq_resource_id:
    """

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        url: str = OAUTH_BASE_URL,
        token_url: str = OAUTH_TOKEN_URL,
        dq_resource_id: str = OAUTH_DQ_RESOURCE_ID,
        token_proxy: Optional[dict] = None,
    ):

        self.base_url: str = url
        self.__token_url: str = token_url
        self.__dq_api_resource_id: str = dq_resource_id

        id_error = f"client_id argument must be a <str> and not {type(client_id)}."
        assert isinstance(client_id, str), id_error
        self.client_id: str = client_id

        secret_error = f"client_secret must be a str and not {type(client_secret)}."
        assert isinstance(client_secret, str), secret_error

        self.client_secret: str = client_secret
        self._stored_token: Optional[dict] = None
        self.token_data = {
            "grant_type": "client_credentials",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "aud": self.__dq_api_resource_id,
        }

        # For debugging purposes save last request response.
        self.status_code: Optional[int] = None
        self.last_response: Optional[str] = None
        self.last_url: Optional[str] = None
        self.token_proxy: Optional[dict] = token_proxy

    def _active_token(self) -> bool:
        """Confirms if the token being used has not expired."""
        created: datetime = self._stored_token["created_at"]
        expires: int = self._stored_token["expires_in"]

        return (datetime.now() - created).total_seconds() / 60 >= (expires - 1)

    def _valid_token(self) -> bool:
        """Confirms if the credentials passed correspond to a valid token."""
        return not (self._stored_token is None or self._active_token())

    def _get_token(self) -> str:
        """Retrieves the token which is used to access DataQuery via OAuth method."""

        if not self._valid_token():
            js, success, self.last_url, msg = dq_request(
                url=self.__token_url,
                data=self.token_data,
                method="post",
                proxies=self.token_proxy,
            )
            if not success:
                raise AuthenticationError(
                    RuntimeError(
                        f"Unable to retrieve authenticationn token. Error details: {msg}"
                    )
                )
            self._stored_token: dict = {
                "created_at": datetime.now(),
                "access_token": js["access_token"],
                "expires_in": js["expires_in"],
            }

        return self._stored_token["access_token"]

    def get_dq_api_result(
        self, url: str, params: dict = None, proxy: Optional[dict] = None
    ) -> dict:
        """Method used exclusively to request data from the API.

        :param <str> url: url to access DQ API.
        :param <dict> params: dictionary containing the required parameters for the
            ticker series.
        :param <Optional[dict]> proxy: dictionary of proxy server.
        """

        js, success, self.last_url, msg = dq_request(
            url=url,
            params=params,
            headers={"Authorization": "Bearer " + self._get_token()},
            proxies=proxy,
        )
        self.last_response = {"json": js, "success": success, "msg": msg}
        return js, success, msg


class Interface(object):
    """API Interface to ©JP Morgan DataQuery.

    :param <bool> debug: boolean,
        if True run the interface in debugging mode.
    :param <bool> concurrent: run the requests concurrently.
    :param <int> batch_size: number of JPMaQS expressions handled in a single request
        sent to DQ API. Each request will be handled concurrently by DataQuery.
    :param <str> client_id: optional argument required for OAuth authentication
    :param <str> client_secret: optional argument required for OAuth authentication
    :param <dict> kwargs: dictionary of optional arguments such as OAuth client_id <str>, client_secret <str>,
        base_url <str>, token_url <str> (OAuth), resource_id <str> (OAuth), and username, password, crt, and key
        (SSL certificate authentication).

    """

    def __init__(
        self,
        oauth: bool = False,
        debug: bool = False,
        concurrent: bool = True,
        batch_size: int = 20,
        heartbeat: bool = False,
        **kwargs,
    ):

        self.proxy = kwargs.pop("proxy", kwargs.pop("proxies", None))
        self.heartbeat = heartbeat

        if oauth:
            self.access: OAuth = OAuth(
                client_id=kwargs.pop("client_id"),
                client_secret=kwargs.pop("client_secret"),
                url=kwargs.pop("base_url", OAUTH_BASE_URL),
                token_url=kwargs.pop("token_url", OAUTH_TOKEN_URL),
                dq_resource_id=kwargs.pop("resource_id", OAUTH_DQ_RESOURCE_ID),
                token_proxy=kwargs.pop("token_proxy", self.proxy),
            )
        else:
            self.access: CertAuth = CertAuth(
                username=kwargs.pop("username"),
                password=kwargs.pop("password"),
                crt=kwargs.pop("crt"),
                key=kwargs.pop("key"),
                base_url=kwargs.pop("base_url", CERT_BASE_URL),
            )

        self.debug: bool = debug
        self.last_url: Optional[str] = None
        self.status_code: Optional[int] = None
        self.last_response: Optional[str] = None
        self.concurrent: bool = concurrent
        self.batch_size: int = batch_size

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type:
            print(f"Execution {exc_type} with value (exc_value):\n{exc_value}")
            logger.error(f"Execution {exc_type} with value (exc_value):\n{exc_value}")

    def check_connection(self) -> Tuple[bool, dict]:
        """Check connection (heartbeat) to DataQuery."""
        endpoint = "/services/heartbeat"
        js, success, msg = self.access.get_dq_api_result(
            url=self.access.base_url + endpoint,
            params={"data": "NO_REFERENCE_DATA"},
            proxy=self.proxy,
        )

        if not success:
            return False, {}

        if "info" not in js:
            logger.error(
                f"Invalid response from DataQuery. {js}"
                f"request {self.last_url:s} error response at {datetime.datetime.utcnow().isoformat()}: {js}"
            )
            raise ValueError(
                f"Invalid response from DataQuery."
                "'info' missing from response.keys():"
                f"{js.keys()}, request {self.last_url:s} error response at {datetime.datetime.utcnow().isoformat()}: {js}"
            )

        results: dict = js["info"]
        assert int(results["code"]) == 200, f"Error message from DataQuery: {results}"
        return int(results["code"]) == 200, results

    def _fetch_threading(self, endpoint, params: dict, server_count: int = 5):
        """
        Method responsible for requesting Tickers from the API. Able to pass in 20
        Tickers in a single request. If there is a request failure, the function will
        return a None-type Object and the request will be made again but with a slower
        delay.

        :param <str> endpoint:
        :param <dict> params: dictionary containing the required parameters.
        :param <int> server_count: count of servers to be retried.

        return <dict>: singular dictionary obtaining maximum 20 elements.
        """

        # The url is instantiated on the ancillary Classes as it depends on the DQ access
        # method chosen.
        url = self.access.base_url + endpoint
        select = "instruments"
        response = {}
        results = []
        counter = 0
        invalid_responses = 0
        while (
            (not (select in response.keys()))
            and (counter <= server_count)
            and (invalid_responses <= server_count)
        ):
            try:
                # The required fields will already be instantiated on the instance of the
                # Class.
                logger.info(
                    f"Requesting {url} with params {params}"
                    + (f"with proxy {self.proxy}" if self.proxy else "")
                )
                logger.info(
                    f"Failed requests counter: {counter}, invalid_responses: {invalid_responses}"
                )
                response, status, msg = self.access.get_dq_api_result(
                    url=url, params=params, proxy=self.proxy
                )

                if status:
                    if response is None:
                        # When these conditions are true, the endpoint is actively returning None.
                        # This is an indication that the delay is too short.
                        # triggers a retry with a longer delay
                        return None
                else:
                    logger.warning(
                        f"respone returned with HTTP Status Code {int(msg['status_code'])}."
                        f"response : {response},"
                        f"status_code : {int(msg['status_code'])},"
                        f"msg : {msg},"
                        f"url : {url},"
                        f"params : {params},"
                        f"dq_api.Interface.last_url : {self.last_url},"
                        f"status_code : {int(msg['status_code'])}"
                    )
                    raise ValueError(
                        f"Invalid response from DataQuery. response : {response}"
                    )

            except ConnectionResetError:
                counter += 1
                time.sleep(0.05)
                logger.warning(
                    f"Server error: will retry. Retry number: {counter+invalid_responses}."
                    f"ConnectionResetError count: {counter},"
                    f"invalid_responses count: {invalid_responses},"
                    f"dq_api.Interface.last_url : {self.last_url},"
                    f"dq_api.Interface.last_response : {self.last_response},"
                )
                continue
            except ValueError:
                invalid_responses += 1
                time.sleep(0.05)
                logger.warning(
                    f"Server error: Invalid response received. Retry number: {counter+invalid_responses}."
                    f"ConnectionResetError count: {counter}."
                    f"invalid_responses count: {invalid_responses}."
                    f"response : {response},"
                    f"status : {status},"
                    f"msg : {msg},"
                    f"url : {url},"
                    f"params : {params},"
                    f"dq_api.Interface.last_url : {self.last_url}"
                )
            else:
                logger.info(f"Request successful.")
                if select in response.keys():
                    results.extend(response[select])

                if response["links"][1]["next"] is None:
                    break

                url = f"{self.access.base_url:s}{response['links'][1]['next']:s}"
                params = {}

        if (counter > server_count) or (invalid_responses > server_count):
            raise ConnectionError(
                f"Connection to DataQuery failed. counter: {counter}, invalid_responses: {invalid_responses}"
                f"dq_api.Interface.last_url : {self.last_url},"
                f"dq_api.Interface.last_response : {self.last_response},"
            )

        if (len(results) == 0) or (None in results):
            return None
        else:
            return results, status, msg

    def _request(
        self,
        endpoint: str,
        tickers: List[str],
        params: dict,
        delay: int = 0,
        count: int = 0,
        start_date: str = None,
        end_date: str = None,
        calendar: str = "CAL_ALLDAYS",
        frequency: str = "FREQ_DAY",
        conversion: str = "CONV_LASTBUS_ABS",
        nan_treatment: str = "NA_NOTHING",
        debug: bool = False,
    ):
        """
        Method designed to concurrently request tickers from the API. Each initiated
        thread will handle batches of 20 tickers, and 10 threads will be active
        concurrently. Able to request data sequentially if required or server overload.

        :param <str> endpoint: url.
        :param <List[str]> tickers: List of Tickers.
        :param <dict> params: dictionary of required parameters for request.
        :param <Integer> delay: each release of a thread requires a delay (roughly 200
            milliseconds) to prevent overwhelming DataQuery. Computed dynamically if DQ
            is being hit too hard. Naturally, if the code is run sequentially, the delay
            parameter is not applicable. Thus, default value is zero.
        :param <Integer> count: tracks the number of recursive calls of the method. The
            first call requires defining the parameter dictionary used for the request
            API.
        :param <str> start_date:
        :param <str> end_date:
        :param <str> calendar:
        :param <str> frequency: frequency metric - default is daily.
        :param <str> conversion:
        :param <str> nan_treatment:

        return <dict>: single dictionary containing all the requested Tickers and their
            respective time-series over the defined dates.
        """

        if delay > 0.9999:
            error_delay = "Issue with DataQuery - requests should not be throttled."
            raise RuntimeError(error_delay)

        no_tickers = len(tickers)
        print(f"Number of expressions requested :\t {no_tickers}")
        logger.info(f"Number of expressions requested : {no_tickers}")

        if not count:
            params_ = {
                "format": "JSON",
                "start-date": start_date,
                "end-date": end_date,
                "calendar": calendar,
                "frequency": frequency,
                "conversion": conversion,
                "nan_treatment": nan_treatment,
                "data": "NO_REFERENCE_DATA",
            }
            params.update(params_)

        b = self.batch_size
        iterations = ceil(no_tickers / b)
        tick_list_compr = [tickers[(i * b) : (i * b) + b] for i in range(iterations)]

        unpack = list(chain(*tick_list_compr))
        assert len(unpack) == len(set(unpack)), "List comprehension incorrect."

        thread_output = []
        final_output = []
        error_tickers = []
        error_messages = []
        if self.concurrent:

            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = []

                for r_list in tick_list_compr:

                    params_copy = params.copy()
                    params_copy["expressions"] = r_list
                    futures.append(
                        [
                            executor.submit(
                                self._fetch_threading, endpoint, params_copy
                            ),
                            r_list,
                        ]
                    )

                    time.sleep(delay)
                    thread_output.append(futures[-1][0])

                for i, fto in enumerate(concurrent.futures.as_completed(thread_output)):
                    try:
                        response, status, msg = fto.result()
                        if not status:
                            error_tickers.extend(tick_list_compr[i])
                            error_messages.append(msg)
                            logger.warning(
                                f"Error in requestion tickers: {', '.join(futures[i][1])}."
                                f"Error details: {msg}"
                            )

                        if fto.__dict__["_result"][0] is None:
                            return None
                    except ValueError:
                        delay += 0.05
                        error_tickers.extend(futures[i][1])
                        logger.warning(
                            f"Error requesting tickers: {', '.join(futures[i][1])}."
                        )
                    else:
                        if isinstance(response, list):
                            final_output.extend(response)
                        else:
                            continue
                            # error_tickers.extend(futures[i][1])
                            # error_messages.append(msg)

        else:
            # Runs through the Tickers sequentially. Thus, breaking the requests into
            # subsets is not required.
            for elem in tick_list_compr:
                params["expressions"] = elem
                results = self._fetch_threading(endpoint=endpoint, params=params)
                final_output.extend(results)

        error_tickers = list(chain(*error_tickers))

        if error_tickers:
            count += 1
            recursive_call = True
            while recursive_call:
                delay += 0.1
                try:
                    (
                        rec_final_output,
                        rec_error_tickers,
                        rec_error_messages,
                    ) = self._request(
                        endpoint=endpoint,
                        tickers=list(set(error_tickers)),
                        params=params,
                        delay=delay,
                        count=count,
                    )
                    # NOTE: now the new error tickers are the only error tickers,
                    # but error messages and final_output are appended
                    error_tickers = rec_error_tickers
                    error_messages.extend(rec_error_messages)
                    final_output.extend(rec_final_output)
                    if not error_tickers:
                        recursive_call = False
                    elif count > 5:
                        recursive_call = False
                        logger.warning(
                            f"Error requesting tickers: {', '.join(error_tickers)}. No longer retrying."
                        )

                except TypeError:
                    continue

        return final_output, error_tickers, error_messages

    @staticmethod
    def delay_compute(no_tickers):
        """
        DataQuery is only able to handle a request every 200 milliseconds. However, given
        the erratic behaviour of threads, the time delay between the release of each
        request must be a function of the size of the request: the smaller the request,
        the closer the delay parameter can be to the limit of 200 milliseconds.
        Therefore, the function adjusts the delay parameter to the number of tickers
        requested.

        :param <int> no_tickers: number of tickers requested.

        :return <float> delay: internally computed value.
        """

        if not floor(no_tickers / 100):
            delay = 0.05
        elif not floor(no_tickers / 1000):
            delay = 0.2
        else:
            delay = 0.3

        return delay

    def get_ts_expression(
        self, expression, original_metrics, suppress_warning, **kwargs
    ):
        """
        Main driver function. Receives the Tickers and returns the respective dataframe.

        :param <List[str]> expression: categories & respective cross-sections requested.
        :param <List[str]> original_metrics: List of required metrics: the returned
            DataFrame will reflect the order of the received List.
        :param <bool> suppress_warning: required for debugging.
        :param <dict> kwargs: dictionary of additional arguments.

        :return: <pd.DataFrame> df: ['cid', 'xcat', 'real_date'] + [original_metrics].
        """
        if self.heartbeat:
            logger.info("Checking connection using heartbeat")
            clause, results = self.check_connection()
        else:
            clause, results = True, None

        if not clause:
            logger.error(f"Connection failed. Error message: {results}.")
            return None

        c_delay = self.delay_compute(len(expression))
        results = None

        while results is None:
            results = self._request(
                endpoint="/expressions/time-series",
                tickers=expression,
                params={},
                delay=c_delay,
                **kwargs,
            )
            c_delay += 0.1

        results, error_tickers, error_messages = results

        unavailable_expressions, invalid_expressions = [], []
        for i, res in enumerate(results):
            if res["attributes"][0]["time-series"] is None:
                if (
                    res["attributes"][0]["message"].strip()
                    == "FAILED - Error in parsing JSON data"
                ):
                    invalid_expressions.append(res["attributes"][0]["expression"])
                elif (
                    res["attributes"][0]["message"]
                    .strip()
                    .startswith("No data available for")
                ):
                    unavailable_expressions.append(res["attributes"][0]["expression"])

        valid_results_count = (
            len(results) - len(invalid_expressions) - len(unavailable_expressions)
        )
        logger.warning(f"Invalid expressions: {', '.join(invalid_expressions)}")
        logger.warning(f"Number of invalid expressions: {len(invalid_expressions)}")
        logger.warning(f"Unavailable expressions: {', '.join(unavailable_expressions)}")
        logger.warning(
            f"Number of unavailable expressions: {len(unavailable_expressions)}"
        )
        logger.warning(f"Number of expressions returned : {valid_results_count}")
        print(f"(Number of unavailable expressions: {len(unavailable_expressions)})")
        print(f"(Number of invalid expressions: {len(invalid_expressions)})")
        print(f"Number of expressions returned :\t {valid_results_count}")
        if valid_results_count < len(expression):
            print(
                "Some expressions were not returned. Checker logger output for more details."
            )

        if error_tickers:
            logger.warning(f"Request failed for tickers: {', '.join(error_tickers)}.")
            logger.warning(f"Error messages: [{', '.join(error_messages)}].")

        r = {
            "results": results,
            "error_tickers": error_tickers,
            "error_messages": error_messages,
        }
        return r
