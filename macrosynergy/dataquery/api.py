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
        **kwargs
    ):

        if oauth:
            self.access: OAuth = OAuth(
                client_id=kwargs.pop("client_id"),
                client_secret=kwargs.pop("client_secret")
            )
        else:
            self.access: CertAuth = CertAuth(
                username=kwargs.pop("username"),
                password=kwargs.pop("password"),
                crt=kwargs.pop("crt"),
                key=kwargs.pop("key")
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
        js: dict = self.access.get_dq_api_result(
            url=self.access.base_url + endpoint,
            params={"data": "NO_REFERENCE_DATA"}
        )

        try:
            results: dict = js["info"]
        except KeyError:
            # dq_url = self.access.base_url
            # print("base url:", dq_url)
            # ip_addr = socket.gethostbyname(dq_url)
            url = self.access.last_url
            now = datetime.datetime.utcnow()
            raise ConnectionError(
                f"DataQuery request {url:s} error response at {now.isoformat()}: {js}"
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

        try:
            response[select]
        except KeyError:
            print(f"Key {select} not found in response: {response} --- will retry "
                  f"download.")
            return False
        else:
            return True

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

        results = []
        counter = 0
        while counter <= server_count:
            try:
                # The required fields will already be instantiated on the instance of the
                # Class.
                response: dict = self.access.get_dq_api_result(url=url, params=params)
            except ConnectionResetError:
                counter += 1
                time.sleep(0.05)
                print(f"Server error: will retry. Attempt number: {counter}.")
                continue

            count = 0
            while not self.server_retry(response, select):
                count += 1
                if count > 5:
                    raise RuntimeError("All servers are down.")

            if select in response.keys():
                results.extend(response[select])

            if response["links"][1]["next"] is None:
                break

            url = f"{self.access.base_url:s}{response['links'][1]['next']:s}"
            params = {}

        if isinstance(results, list):
            return results
        else:
            return None

    def _request(self, endpoint: str, tickers: List[str], params: dict,
                 delay: int = 0, count: int = 0, start_date: str = None,
                 end_date: str = None, calendar: str = "CAL_ALLDAYS",
                 frequency: str = "FREQ_DAY", conversion: str = "CONV_LASTBUS_ABS",
                 nan_treatment: str = "NA_NOTHING"):
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
        tickers_server = []
        if self.concurrent:

            with concurrent.futures.ThreadPoolExecutor() as executor:
                for r_list in tick_list_compr:

                    params_copy = params.copy()
                    params_copy["expressions"] = r_list
                    results = executor.submit(
                        self._fetch_threading, endpoint, params_copy
                    )

                    time.sleep(delay)
                    results.__dict__[str(id(results))] = r_list
                    thread_output.append(results)

                for f in concurrent.futures.as_completed(thread_output):
                    try:
                        response = f.result()
                        if f.__dict__["_result"] is None:
                            return None

                    except ValueError:
                        delay += 0.05
                        tickers_server.append(f.__dict__[str(id(f))])
                    else:
                        if isinstance(response, list):
                            final_output.extend(response)
                        else:
                            continue

        else:
            # Runs through the Tickers sequentially. Thus, breaking the requests into
            # subsets is not required.
            for elem in tick_list_compr:
                params["expressions"] = elem
                results = self._fetch_threading(endpoint=endpoint, params=params)
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
                        delay=delay, count=count
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
            assert metric in [
                "value",
                "eop_lag",
                "mop_lag",
                "grading"], f"Incorrect metric passed: {metric}."

        unique_tix = list(set(expression))

        dq_tix = self.jpmaqs_indicators(
            metrics=original_metrics, tickers=unique_tix
        )
        expression = dq_tix

        c_delay = self.delay_compute(len(dq_tix))
        results = self._request(
            endpoint="/expressions/time-series",
            tickers=expression,
            params={},
            delay=c_delay,
            **kwargs
        )

        while results is None:
            c_delay += 0.1
            results = self._request(
                endpoint="/expressions/time-series",
                tickers=expression,
                params={},
                delay = c_delay,
                **kwargs
            )

        no_metrics = len(set([tick.split(",")[-1][:-1] for tick in expression]))

        results_dict, output_dict, s_list = self.isolate_timeseries(
            results, original_metrics, self.debug, False
        )

        # Conditional statement which is only applicable if multiple metrics have been
        # requested. If any Ticker is not defined over all requested metrics, run
        # sequentially to confirm it is missing from the database. If all metrics are not
        # available, the Ticker will not be included in the output DataFrame.
        if s_list:
            sequential = True
            self.__dict__["concurrent"] = False
            results_seq = self._request(
                endpoint="/expressions/time-series", tickers=s_list, params={}, **kwargs
            )
            r_dict, o_dict, s_list = self.isolate_timeseries(
                results_seq, original_metrics, debug=False, sequential=sequential
            )
            results_dict = {**results_dict, **r_dict}

        results_dict = self.valid_ticker(results_dict, suppress_warning, self.debug)

        results_copy = results_dict.copy()
        try:
            results_copy.popitem()
        except Exception as err:
            print(err)
            print("None of the tickers are available in the Database.")
            return
        else:
            return self.dataframe_wrapper(results_dict, no_metrics, original_metrics)

    @staticmethod
    def array_construction(metrics: List[str], output_dict: dict,
                           debug: bool, sequential: bool):
        """
        Helper function that will pass through the dictionary and aggregate each
        time-series stored in the interior dictionary. Will return a single dictionary
        where the keys are the tickers and the values are the aggregated time-series.

        :param <List[str]> metrics: metrics requested from the API.
        :param <dict> output_dict: nested dictionary where the keys are the tickers and
            the value is itself a dictionary. The interior dictionary's keys will be the
            associated metrics and the values will be their respective time-series.
        :param <bool> debug: used to understand any underlying issue.
        :param <bool> sequential: if series are not returned, potentially the fault of
            the threading mechanism, isolate each Ticker and run sequentially.

        """

        modified_dict = {}
        d_frame_order = ['real_date'] + metrics

        ticker_list = []
        for k, v in output_dict.items():

            available_metrics = set(v.keys())
            expected_metrics = set(d_frame_order)

            missing_metrics = list(expected_metrics.difference(available_metrics))
            if not missing_metrics:
                # Aggregates across all metrics requested and order according to the
                # prescribed list.
                ticker_df = pd.DataFrame.from_dict(v)[d_frame_order]

                modified_dict[k] = ticker_df.to_numpy()

            elif missing_metrics and not sequential:
                # If a requested metric has not been returned, its absence could be
                # ascribed to a potential data leak from multithreading. Therefore,
                # collect the respective tickers, defined over each metric, and run
                # the requests sequentially to avoid any scope for data leakage.

                temp_list = ['DB(JPMAQS,' + k + ',' + m + ')' for m in metrics]
                ticker_list += temp_list

                if debug:
                    print(
                        f"The ticker, {k}, is missing the metric(s) "
                        f"'{missing_metrics}' whilst the requests are running "
                        f"concurrently - will check the API sequentially."
                    )
            elif sequential and debug:
                print(
                    f"The ticker, {k}, is missing from the API after "
                    f"running sequentially - will not be in the returned "
                    f"DataFrame."
                )
            else:
                continue

        return modified_dict, ticker_list

    def isolate_timeseries(
        self,
        list_,
        metrics: List[str],
        debug: bool,
        sequential: bool
    ):
        """
        Isolates the metrics, across all categories & cross-sections, held in the List,
        and concatenates the time-series, column-wise, into a single structure, and
        subsequently stores that structure in a dictionary where the dictionary's
        keys will be each Ticker.
        Will validate that each requested metric is available, in the data dictionary,
        for each Ticker. If not, will run the Tickers sequentially to confirm the issue
        is not ascribed to multithreading overloading the load balancer.

        :param <List[dict]> list_: returned from DataQuery.
        :param <List[str]> metrics: metrics requested from the API.
        :param <bool> debug: used to understand any underlying issue.
        :param <bool> sequential: if series are not returned, potentially the fault of
            the threading mechanism, isolate each Ticker and run sequentially.

        :return: <dict> modified_dict.
        """
        output_dict = defaultdict(dict)
        size = len(list_)
        if debug:
            print(f"Number of returned expressions from JPMaQS: {size}.")

        unavailable_series = []
        # Each element inside the List will be a dictionary for an individual Ticker
        # returned by DataQuery.
        for r in list_:

            dictionary = r["attributes"][0]
            ticker = dictionary["expression"].split(",")
            metric = ticker[-1][:-1]

            ticker_split = ",".join(ticker[1:-1])
            ts_arr = np.array(dictionary["time-series"])

            # Catches tickers that are defined correctly but will not have a valid
            # associated series. For example, "USD_FXXR_NSA" or "NLG_FXCRR_VT10". The
            # request to the API will return the expression but the "time-series" value
            # will be a None Object.
            # Occasionally, on large requests, DataQuery will incorrectly return a None
            # Object for a series that is available in the database.
            if ts_arr.size == 1:
                unavailable_series.append(ticker_split)

            else:
                if ticker_split not in output_dict.keys():
                    output_dict[ticker_split]["real_date"] = ts_arr[:, 0]
                    output_dict[ticker_split][metric] = ts_arr[:, 1]
                # Each encountered metric should be unique and one of "value", "grading",
                # "eop_lag" or "mop_lag".
                else:
                    output_dict[ticker_split][metric] = ts_arr[:, 1]

        output_dict_c = output_dict.copy()

        modified_dict, ticker_list = self.array_construction(
            metrics=metrics, output_dict=output_dict_c, debug=debug,
            sequential=sequential
        )
        if debug:
            print(f"The number of tickers requested that are unavailable is: "
                  f"{len(unavailable_series)}.")
            self.__dict__["unavailable_series"] = unavailable_series

        return modified_dict, output_dict, ticker_list

    @staticmethod
    def column_check(v, col, no_cols, debug):
        """
        Checking the values of the returned TimeSeries.

        :param <np.array> v:
        :param <integer> col: used to isolate the column being checked.
        :param <integer> no_cols: number of metrics requested.
        :param <bool> debug:

        :return <bool> condition.
        """
        returns = list(v[:, col])
        condition = all([isinstance(elem, type(None)) for elem in returns])

        if condition:
            other_metrics = list(v[:, 2:no_cols].flatten())

            if debug and all([isinstance(e, type(None)) for e in other_metrics]):
                warnings.warn("Error has occurred in the Database.")

        return condition

    def valid_ticker(self, _dict, suppress_warning, debug):
        """
        Iterates through each Ticker and determines whether the Ticker is held in the
        Database or not. The validation mechanism will isolate each column, in all the
        Tickers held in the dictionary, where the columns reflect the metrics passed,
        and validates that each value is not a NoneType Object. If all values are
        NoneType Objects, the Ticker is not valid, and it will be popped from the
        dictionary.

        :param <dict> _dict:
        :param <bool> suppress_warning:
        :param <bool> debug:

        :return: <dict> dict_copy.
        """

        ticker_missing = 0
        dict_copy = _dict.copy()

        for k, v in _dict.items():
            no_cols = v.shape[1]
            condition = self.column_check(v, col=1, no_cols=no_cols, debug=debug)

            if condition:
                ticker_missing += 1
                dict_copy.pop(k)

                if not suppress_warning:
                    print(f"The ticker, {k}), does not exist in the Database.")

        print(f"Number of missing time-series from the Database: {ticker_missing}.")
        return dict_copy

    @staticmethod
    def dataframe_wrapper(_dict, no_metrics, original_metrics):
        """
        Receives a Dictionary containing every Ticker and the respective time-series data
        held inside an Array. Will iterate through the dictionary and stack each Array
        into a single DataFrame retaining the order both row-wise, in terms of cross-
        sections, and column-wise, in terms of the metrics.

        :param <dict> _dict:
        :param <Integer> no_metrics: Number of metrics requested.
        :param <List[str]> original_metrics: Order of the metrics passed.

        :return: pd.DataFrame: ['cid', 'xcat', 'real_date'] + [original_metrics]
        """

        tickers_no = len(_dict.keys())
        length = list(_dict.values())[0].shape[0]

        arr = np.empty(shape=(length * tickers_no, 3 + no_metrics), dtype=object)

        i = 0
        for k, v in _dict.items():

            ticker = k.split("_")

            cid = ticker[0]
            xcat = "_".join(ticker[1:])

            cid_broad = np.repeat(cid, repeats=v.shape[0])
            xcat_broad = np.repeat(xcat, repeats=v.shape[0])
            data = np.column_stack((cid_broad, xcat_broad, v))

            row = i * v.shape[0]
            arr[row: row + v.shape[0], :] = data
            i += 1

        columns = ["cid", "xcat", "real_date"]
        cols_output = columns + original_metrics

        df = pd.DataFrame(data=arr, columns=cols_output)

        df["real_date"] = pd.to_datetime(df["real_date"], yearfirst=True)
        df = df[df["real_date"].dt.dayofweek < 5]
        df = df.fillna(value=np.nan)
        df = df.reset_index(drop=True)

        for m in original_metrics:
            df[m] = df[m].astype(dtype=np.float32)

        df.real_date = pd.to_datetime(df.real_date)
        return df

    def tickers(
        self,
        tickers: list,
        metrics: list = ['value'],
        start_date: str = '2000-01-01',
        suppress_warning=False
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
                suppress_warning=suppress_warning
            )

            if isinstance(df, pd.DataFrame):
                df = df.sort_values(["cid", "xcat", "real_date"]).reset_index(drop=True)

            return df
        else:
            logger.error(
                "DataQuery response %s with description: %s", results["message"],
                results["description"]
            )
            error = "Unable to connect to DataQuery. Reach out to DQ Support."
            raise ConnectionError(error)

    def download(
        self,
        tickers=None,
        xcats=None,
        cids=None,
        metrics=['value'],
        start_date='2000-01-01',
        suppress_warning=False
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
            cids_dmca = ["AUD", "CAD", "CHF", "EUR", "GBP", "JPY", "NOK", "NZD", "SEK",
                         "USD"]  # DM currency areas
            cids_dmec = ["DEM", "ESP", "FRF", "ITL", "NLG"]  # DM euro area countries
            cids_latm = ["BRL", "COP", "CLP", "MXN", "PEN"]  # Latam countries
            cids_emea = ["HUF", "ILS", "PLN", "RON", "RUB", "TRY", "ZAR"]  # EMEA countries
            cids_emas = ["CZK", "CNY", "IDR", "INR", "KRW", "MYR", "PHP", "SGD", "THB",
                         "TWD"]  # EM Asia countries
            cids_dm = cids_dmca + cids_dmec
            cids_em = cids_latm + cids_emea + cids_emas
            cids = sorted(cids_dm + cids_em)  # Standard default.

        if isinstance(tickers, str):
            tickers = [tickers]
        elif tickers is None:
            tickers = []

        assert isinstance(tickers, list)

        if isinstance(xcats, str):
            xcats = [xcats]

        if isinstance(cids, str):
            cids = [cids]

        if isinstance(metrics, str):
            metrics = [metrics]

        if xcats is not None:
            assert isinstance(xcats, (list, tuple))
            add_tix = [cid + "_" + xcat for xcat in xcats for cid in cids]
            tickers = tickers + add_tix

        df = self.tickers(
            tickers,
            metrics=metrics,
            suppress_warning=suppress_warning,
            start_date=start_date
        )

        return df
