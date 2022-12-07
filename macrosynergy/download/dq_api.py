"""DataQuery Interface."""
import pandas as pd
import numpy as np
import warnings
import concurrent.futures
import time
import socket
import datetime
import logging

from typing import List, Tuple
from math import ceil, floor
from collections import defaultdict
from itertools import chain
from typing import Optional
from macrosynergy.dataquery.auth import (
    CertAuth, OAuth, CERT_BASE_URL, OAUTH_BASE_URL, OAUTH_TOKEN_URL, OAUTH_DQ_RESOURCE_ID
)

logger = logging.getLogger(__name__)


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
        **kwargs
    ):

        self.proxy = kwargs.pop("proxy", None)

        if oauth:
            self.access: OAuth = OAuth(
                client_id=kwargs.pop("client_id"),
                client_secret=kwargs.pop("client_secret"),
                url=kwargs.pop("base_url", OAUTH_BASE_URL),
                token_url=kwargs.pop("token_url", OAUTH_TOKEN_URL),
                dq_resource_id=kwargs.pop("resource_id", OAUTH_DQ_RESOURCE_ID),
                token_proxy=kwargs.pop("token_proxy", self.proxy)
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

    def check_connection(self) -> Tuple[bool, dict]:
        """Check connection (heartbeat) to DataQuery.
        """
        endpoint = "/services/heartbeat"
        js, success, msg = self.access.get_dq_api_result(
            url=self.access.base_url + endpoint,
            params={"data": "NO_REFERENCE_DATA"},
            proxy=self.proxy
        )

        if success:
            try:
                results: dict = js["info"]
            except KeyError:
                url = self.access.last_url
                now = datetime.datetime.utcnow()
                raise ConnectionError(
                    f"DataQuery request {url:s} error response at {now.isoformat()}: {js}"
                )
            else:
                return int(results["code"]) == 200, results
        else:
            # TODO msg?
            return False, {}

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
        while (not (select in response.keys())) and (counter <= server_count):
            try:
                # The required fields will already be instantiated on the instance of the
                # Class.
                response, status, msg  = self.access.get_dq_api_result(url=url, params=params, 
                                                                    proxy=self.proxy)
            except ConnectionResetError:
                counter += 1
                time.sleep(0.05)
                print(f"Server error: will retry. Attempt number: {counter}.")
                continue


            if select in response.keys():
                results.extend(response[select])

            if response["links"][1]["next"] is None:
                break

            url = f"{self.access.base_url:s}{response['links'][1]['next']:s}"
            params = {}

        if isinstance(results, list):
            return results, status, msg
        else:
            return [], status, msg

    def _request(self, endpoint: str, tickers: List[str], params: dict,
                 delay: int = 0, count: int = 0, start_date: str = None,
                 end_date: str = None, calendar: str = "CAL_ALLDAYS",
                 frequency: str = "FREQ_DAY", conversion: str = "CONV_LASTBUS_ABS",
                 nan_treatment: str = "NA_NOTHING", debug : bool = False):
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
        print(f"Number of expressions requested {no_tickers}.")

        if not count:
            params_ = {
                "format": "JSON",
                "start-date": start_date,
                "end-date": end_date,
                "calendar": calendar,
                "frequency": frequency,
                "conversion": conversion,
                "nan_treatment": nan_treatment,
                "data": "NO_REFERENCE_DATA"
            }
            params.update(params_)

        b = self.batch_size
        iterations = ceil(no_tickers / b)
        tick_list_compr = [tickers[(i * b): (i * b) + b] for i in range(iterations)]

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
                    futures.append([executor.submit(
                        self._fetch_threading, endpoint, params_copy), r_list])

                    time.sleep(delay)
                    thread_output.append(futures[-1][0])

                for i, fto in enumerate(concurrent.futures.as_completed(thread_output)):
                    try:
                        response, status, msg = fto.result()
                        if not status:
                            error_tickers.extend(tick_list_compr[i])
                            error_messages.append(msg)
                            logger.warning(f"Error in requestion tickers: {', '.join(futures[i][1])}.")
                        
                        if fto.__dict__["_result"][0] is None:
                            return None
                    except ValueError:
                        delay += 0.05
                        error_tickers.extend(futures[i][1])
                        logger.warning(f"Error requesting tickers: {', '.join(futures[i][1])}.")
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
                    rec_final_output, rec_error_tickers, rec_error_messages = self._request(
                        endpoint=endpoint,
                        tickers=list(set(error_tickers)),
                        params=params,
                        delay=delay, count=count
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
                        logger.warning(f"Error requesting tickers: {', '.join(error_tickers)}. No longer retrying.")

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
        clause, results = self.check_connection()
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
                delay = c_delay,
                **kwargs
            )
            c_delay += 0.1
        
        results, error_tickers, error_messages = results
        if error_tickers:
            logger.warning(f"Request failed for tickers: {', '.join(error_tickers)}.")
            logger.warning(f"Error messages: [{', '.join(error_messages)}].")

        r = {'results' : results, 'error_tickers' : error_tickers, 'error_messages' : error_messages}
        return r

