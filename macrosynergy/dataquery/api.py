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
from macrosynergy.dataquery.auth import CertAuth, OAuth
from macrosynergy.dataquery.exceptions import DQException

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

    """

    def __init__(
        self,
        oauth: bool = False,
        debug: bool = False,
        concurrent: bool = True,
        batch_size: int = 20,
        **kwargs,
    ):

        if oauth:
            self.access: OAuth = OAuth(
                client_id=kwargs.pop("client_id"),
                client_secret=kwargs.pop("client_secret"),
            )
        else:
            self.access: CertAuth = CertAuth(
                username=kwargs.pop("username"),
                password=kwargs.pop("password"),
                crt=kwargs.pop("crt"),
                key=kwargs.pop("key"),
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
        """Check connection (heartbeat) to DataQuery."""
        endpoint = "/services/heartbeat"
        dq_api_result = self.access.get_dq_api_result(
            url=self.access.base_url + endpoint, params={"data": "NO_REFERENCE_DATA"}
        )
        js, success, msg = dq_api_result
        if success:
            try:
                results: dict = js["info"]
            except KeyError:
                url = self.access.last_url
                raise DQException(
                    message="DataQuery request error response",
                    url=url,
                    response=self.access.headers,
                    base_exception=ConnectionError,
                )
            else:

                return int(results["code"]) == 200, results

    @staticmethod
    def server_retry(response: dict, select: str):
        """Server retry.

        DQ requests are powered by four servers. Therefore, if a single server is failing
        try the remaining three servers for a request. In theory, trying the sample space
        of servers should invariably result in a successful request: assuming all four
        servers are not concurrently down. The number of trials is five: the sample space
        of servers should be exhausted.

        :param <dict> response: server response.
        :param <str> select: key hosting the server's response in the dictionary.

        :return <bool> server_response:
        """
        # TODO: reimplement
        # NOTE: Thihs function is only useful for unit testing, debugging and
        #       logging.
        # Else it can just be mimicked by :
        # assert select in response.keys(), raise DQException?
        try:
            response[select]
        except KeyError:
            if response:
                logger.warning(
                    f"Key {select} not found in response: {response} --- will "
                    f"retry download."
                )
            return False
        else:
            return True

    def _fetch_threading(
        self, endpoint, params: dict, server_count: int = 5
    ) -> List[dict]:
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
        # use server_retry if makes sense
        while (not (select in response.keys())) and (counter <= server_count):
            try:
                # The required fields will already be instantiated on the instance of the
                # Class.
                response, msg, status = self.access.get_dq_api_result(
                    url=url, params=params
                )
            except ConnectionResetError:
                counter += 1
                time.sleep(0.05)
                print(f"Server error: will retry. Attempt number: {counter}.")
                continue

            # TODO use status bool to determine if the request was successful.
            # NOTE or would using the status bool be problematic?
            #       as the status bool is only a representative of the actual response

            if select in response.keys():
                results.extend(response[select])

            if response["links"][1]["next"] is None:
                break

            url = f"{self.access.base_url:s}{response['links'][1]['next']:s}"
            params = {}

        if isinstance(results, list):
            return [results, msg, status]
        else:
            return [[], msg, status]

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
            # raise RuntimeError(error_delay)
            raise DQException(
                message=error_delay,
                url=endpoint,
                # response=None,
                base_exception=RuntimeError(error_delay),
            )
            # only the endpoint is available in this scope, so it is used as the url.
            # response is not available in this scope.

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
                "data": "NO_REFERENCE_DATA",
            }
            params.update(params_)

        b = self.batch_size
        iterations = ceil(no_tickers / b)
        tick_list_compr = [tickers[(i * b) : (i * b) + b] for i in range(iterations)]

        unpack = list(chain(*tick_list_compr))
        # assert len(unpack) == len(set(unpack)), "List comprehension incorrect."
        if len(unpack) != len(set(unpack)):
            error = "List comprehension incorrect."
            raise DQException(message=error, base_exception=RuntimeError(error))
            # not relevant to add url and response here.
            # is this a ValueError?

        thread_output = []
        final_output = []
        tickers_server = []  # problem tickers; tickers to retry
        if self.concurrent:

            with concurrent.futures.ThreadPoolExecutor() as executor:
                for r_list in tick_list_compr:

                    params_copy = params.copy()
                    params_copy["expressions"] = r_list
                    dq_api_results = executor.submit(
                        self._fetch_threading, endpoint, params_copy
                    )

                    time.sleep(delay)
                    dq_api_results.__dict__[str(id(dq_api_results))] = r_list
                    thread_output.append(dq_api_results)

                for f in concurrent.futures.as_completed(thread_output):
                    try:
                        dq_api_result = f.result()

                        # NOTE: if the _result is None, the request has failed.
                        # this is because they were being sent too frequently.
                        # the return None terminates the execution of _request func,
                        # passing it back to get_ts_expression
                        # This "None" is thus a valid result returned by the API.

                        if f.__dict__["_result"] is None:
                            return None
                        if dq_api_result[-1] and isinstance(dq_api_result[0], list):
                            final_output.extend(dq_api_result[0])
                        else:
                            tickers_server.extend(f.__dict__[str(id(f))])

                    # NOTE: ValueError occurs when the thread hangs/dies/corrupts.
                    # This is a rare occurrence, but it is important to handle it.
                    # Any other exception is still raised.
                    except ValueError:
                        delay += 0.05
                        tickers_server.extend(f.__dict__[str(id(f))])



        else:
            # Runs through the Tickers sequentially. Thus, breaking the requests into
            # subsets is not required.
            for elem in tick_list_compr:
                params["expressions"] = elem
                results, msg, status = self._fetch_threading(
                    endpoint=endpoint, params=params
                )
                final_output.extend(results)

        tickers_server = list(chain(*tickers_server))

        if tickers_server:
            count += 1
            recursive_call = True
            while recursive_call:

                delay += 0.1
                try:
                    recursive_output = final_output + self._request(
                        endpoint=endpoint,
                        tickers=list(set(tickers_server)),
                        params=params,
                        delay=delay,
                        count=count,
                    )
                except TypeError:
                    continue
                else:
                    recursive_call = False

            return recursive_output

        return final_output

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

    @staticmethod
    def jpmaqs_indicators(metrics, tickers):
        """
        Functionality used to convert tickers into formal JPMaQS expressions.
        """
        # NOTE: Why not?
        # return [f"DB(JPMAQS,{ticker},{metric})" for ticker in tickers for metric in metrics]
        dq_tix = []
        for metric in metrics:
            dq_tix += ["DB(JPMAQS," + tick + f",{metric})" for tick in tickers]

        return dq_tix

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

        for metric in original_metrics:
            if metric not in ["value", "eop_lag", "mop_lag", "grading"]:
                error = f"Incorrect metric passed: {metric}."
                raise DQException(
                    message=error,
                )

        unique_tix = list(set(expression))

        dq_tix = self.jpmaqs_indicators(metrics=original_metrics, tickers=unique_tix)
        expression = dq_tix

        c_delay = self.delay_compute(len(dq_tix))
        results = None

        while results is None:
            results, msg, status = self._request(
                endpoint="/expressions/time-series",
                tickers=expression,
                params={},
                delay=c_delay,
                **kwargs,
            )
            c_delay += 0.1

        # print(results)
        return results
        # here starts the parsing of the results.

    def tickers(
        self,
        tickers: list,
        metrics: list = ["value"],
        start_date: str = "2000-01-01",
        suppress_warning=False,
    ):
        """
        Returns standardized dataframe of specified base tickers and metric. Will also
        validate the connection to DataQuery through the api using the method
        .check_connection().

        :param <List[str]> tickers: JPMaQS ticker of form <cid>_<xcat>.
        :param <List[str]> metrics: must choose one or more from 'value', 'eop_lag',
            'mop_lag', or 'grading'. Default is ['value'].
        :param <str> start_date: first date in ISO 8601 string format.
        :param <bool> suppress_warning: used to suppress warning of any invalid
            ticker received by DataQuery.

        :return <pd.Dataframe> standardized dataframe with columns 'cid', 'xcats',
            'real_date' and chosen metrics.
        """

        clause, results = self.check_connection()
        if clause:
            print(results["description"])

            df = self.get_ts_expression(
                expression=tickers,
                original_metrics=metrics,
                start_date=start_date,
                suppress_warning=suppress_warning,
            )

            if isinstance(df, pd.DataFrame):
                df = df.sort_values(["cid", "xcat", "real_date"]).reset_index(drop=True)

            return df
        else:
            logger.error(
                "DataQuery response %s with description: %s",
                results["message"],
                results["description"],
            )
            error = "Unable to connect to DataQuery. Reach out to DQ Support."
            # raise ConnectionError(error)
            endpoint = "/services/heartbeat"  # to be removed once  check_connection() is changed.
            raise DQException(
                message=error,
                url=self.access.base_url + endpoint,
                header=self.access.headers,
                base_exception=ConnectionError,
            )

    def download(
        self,
        tickers=None,
        xcats=None,
        cids=None,
        metrics=["value"],
        start_date="2000-01-01",
        suppress_warning=False,
    ):
        """
        Returns standardized dataframe of specified base tickers and metrics.

        :param <List[str]> tickers: JPMaQS ticker of form <cid>_<xcat>. Can be combined
            with selection of categories.
        :param <List[str]> xcats: JPMaQS category codes. Downloaded for all standard
            cross sections identifiers available (if cids are not specified) or those
            selected (if cids are specified). Standard cross sections here include major
            developed and emerging currency markets. See JPMaQS documentation.
        :param <List[str]> cids: JPMaQS cross-section identifiers, typically based  on
            currency code. See JPMaQS documentation.
        :param <str> metrics: must choose one or more from 'value', 'eop_lag', 'mop_lag',
            or 'grading'. Default is ['value'].
        :param <str> start_date: first date in ISO 8601 string format.
        :param <bool> suppress_warning: used to suppress warning of any invalid
            ticker received by DataQuery.

        :return <pd.Dataframe> df: standardized dataframe with columns 'cid', 'xcats',
            'real_date' and chosen metrics.
        """

        if (cids is None) & (xcats is not None):
            cids_dmca = [
                "AUD",
                "CAD",
                "CHF",
                "EUR",
                "GBP",
                "JPY",
                "NOK",
                "NZD",
                "SEK",
                "USD",
            ]  # DM currency areas
            cids_dmec = ["DEM", "ESP", "FRF", "ITL", "NLG"]  # DM euro area countries
            cids_latm = ["BRL", "COP", "CLP", "MXN", "PEN"]  # Latam countries
            cids_emea = [
                "HUF",
                "ILS",
                "PLN",
                "RON",
                "RUB",
                "TRY",
                "ZAR",
            ]  # EMEA countries
            cids_emas = [
                "CZK",
                "CNY",
                "IDR",
                "INR",
                "KRW",
                "MYR",
                "PHP",
                "SGD",
                "THB",
                "TWD",
            ]  # EM Asia countries
            cids_dm = cids_dmca + cids_dmec
            cids_em = cids_latm + cids_emea + cids_emas
            cids = sorted(cids_dm + cids_em)  # Standard default.

        if isinstance(tickers, str):
            tickers = [tickers]
        elif tickers is None:
            tickers = []

        # assert isinstance(tickers, list)
        if not isinstance(tickers, list):
            raise DQException(
                message="'tickers' must be a list of strings", base_exception=TypeError
            )

        if isinstance(xcats, str):
            xcats = [xcats]

        if isinstance(cids, str):
            cids = [cids]

        if isinstance(metrics, str):
            metrics = [metrics]

        if xcats is not None:
            # assert isinstance(xcats, (list, tuple))
            if not isinstance(xcats, (list, tuple)):
                raise DQException(
                    message="'xcats' must be a list of strings",
                    base_exception=TypeError,
                )

            add_tix = [cid + "_" + xcat for xcat in xcats for cid in cids]
            tickers = tickers + add_tix

        df = self.tickers(
            tickers,
            metrics=metrics,
            suppress_warning=suppress_warning,
            start_date=start_date,
        )

        # if not isinstance(df, pd.DataFrame) or df.empty:
        #     raise DQException(
        #         message="No/corrupt data returned from DataQuery",
        #     )

        return df
