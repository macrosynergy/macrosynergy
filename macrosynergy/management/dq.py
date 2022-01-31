"""DataQuery Interface

Interface to the JP Morgan DataQuery and how to interact with the API.

Requires a API login (username + password) as well as a certified certificate
and private key to verify the request.
"""
import requests
import base64
from typing import List
import json
import pandas as pd
import numpy as np
import os
from math import ceil, floor
from collections import defaultdict
import warnings
import concurrent.futures
import time
from itertools import chain

BASE_URL = "https://platform.jpmorgan.com/research/dataquery/api/v2"

DQ_ERROR_MSG = {
    204: "Content requested unavailable.",
    400: "Request but it was malformed or invalid.",
    401: "The user has not successfully authenticated with the service.",
    500: "There was an internal server error."
}


class DataQueryInterface(object):
    """
    Initiate the object DataQueryInterface.

    ©JP Morgan

    Authentication:
      1. Client authentication through 2-way SSL.
      2. User authentication (HTTP basic).

    The URL http://www.jpmm.com points to https://markets.jpmorgan.com

    JP Morgan DataQuery API is based on the OpenAPI standard
    (https://en.wikipedia.org/wiki/OpenAPI_Specification)
    which is build on Swagger (https://swagger.io/docs/).

    Data models for returns:
      - TimeSeriesResponse: "instruments"
      - FiltersResponse: "filters"
      - AttributesResponse: "instruments"
      - InstrumentsResponse: "instruments"
      - GroupsResponse: "groups"

    :param <str> username: username for login to REST API for
        JP Morgan DataQuery.
    :param <str> password: password
    :param <str> crt: string with location of public certificate
    :param <str> key: string with private key location
    :param <str> base_url: string with base URL for DataQuery (entry point)
        usually platform.jpmorgan.com
    :param <bool> date_all: default False,
        if True download all history of data.
    :param <bool> debug: boolean,
        if True run the interface in debugging mode.
    :param <bool> concurrent: run the requests concurrently.

    :return: None
    """
    source_code = "JPMDQ"
    source_name = "DataQuery"
    __name__ = f"{source_name:s}Interface"

    def __init__(self, username: str,
                 password: str,
                 crt: str = "api_macrosynergy_com.crt",
                 key: str = "api_macrosynergy_com.key",
                 base_url: str = BASE_URL,
                 date_all: bool = False,
                 debug: bool = False,
                 concurrent: bool = True,
                 thread_handler: int = 20):

        assert isinstance(username, str),\
            f"username must be a <str> and not {type(username)}: {username}"

        assert isinstance(password, str), \
            f"password must be a <str> and not {type(password)}: {password}"
        self.auth = base64.b64encode(bytes(f'{username:s}:{password:s}',
                                           "utf-8")).decode('ascii')
        self.headers = {"Authorization": f"Basic {self.auth:s}"}

        if base_url is None:
            base_url = BASE_URL

        self.base_url = base_url

        # Key and certificate
        if not (isinstance(key, str) and os.path.exists(key)):
            msg = f"key file, {key}, must be a <str> and exists as a file"
            raise ValueError(msg)
        self.key = key

        if not (isinstance(crt, str) and os.path.exists(crt)):
            msg = f"crt file, {crt}, must be a <str> and exists as a file"
            raise ValueError(msg)
        self.crt = crt

        self.debug = debug
        self.last_url = None
        self.status_code = None
        self.last_response = None
        self.date_all = date_all
        self.concurrent = concurrent
        self.thread_handler = thread_handler

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type:
            print(f'exc_type: {exc_type}')
            print(f'exc_value: {exc_value}')
            print(f'exc_traceback: {exc_traceback}')

    @staticmethod
    def server_retry(response: dict, select: str):
        """
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
            print(f"{response['errors'][0]['message']} - will try a different server.")
            return False
        else:
            return True

    def _fetch_threading(self, endpoint, params: dict):
        """
        Method responsible for requesting Tickers from the API. Able to pass in 20
        Tickers in a single request. If there is a request failure, the function will
        return a None-type Object and the request will be made again but with a slower
        delay.

        :param <str> endpoint:
        :param <dict> params: dictionary containing the required parameters.

        return <dict>: singular dictionary obtaining maximum 20 elements.
        """

        url = self.base_url + endpoint
        select = "instruments"

        results = []
        counter = 0
        clause = lambda counter: (counter <= 5)

        while clause(counter):
            try:
                r = requests.get(url=url, cert=(self.crt, self.key),
                                headers=self.headers, params=params)
            except ConnectionResetError:
                counter += 1
                time.sleep(0.05)
                print(f"Server error: will retry. Attempt number: {counter}.")
                continue
            else:
                last_response = r.text
                response = json.loads(last_response)

                count = 0
                while not self.server_retry(response, select):
                    count += 1
                    if count > 5:
                        raise RuntimeError("All servers are down.")

                dictionary = response[select][0]['attributes'][0]
                if not isinstance(dictionary['time-series'], list):
                    return None

                if not select in response.keys():
                    break
                else:
                    results.extend(response[select])

                if response["links"][1]["next"] is None:
                    break

                url = f"{self.base_url:s}{response['links'][1]['next']:s}"
                params = {}

        if isinstance(results, list):
            return results
        else:
            return None

    def _request(self, endpoint: str, tickers: List[str], params: dict,
                 delay: int = None, count: int = 0, start_date: str = None,
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
            is being hit too hard.
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

        no_tickers = len(tickers)
        if not count:
            params_ = {"format": "JSON", "start-date": start_date, "end-date": end_date,
                       "calendar": calendar, "frequency": frequency, "conversion":
                       conversion, "nan_treatment": nan_treatment}
            params.update(params_)

        t = self.thread_handler
        iterations = ceil(no_tickers / t)
        tick_list_compr = [tickers[(i * t): (i * t) + t] for i in range(iterations)]

        unpack = list(chain(*tick_list_compr))
        assert len(unpack) == len(set(unpack)), "List comprehension incorrect."

        exterior_iterations = ceil(len(tick_list_compr) / 10)

        final_output = []
        tickers_server = []
        if self.concurrent:
            for i in range(exterior_iterations):

                output = []
                if i > 0:
                    time.sleep(delay)
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    for elem in tick_list_compr[(i * 10): (i + 1) * 10]:

                        params_copy = params.copy()
                        params_copy["expressions"] = elem
                        results = executor.submit(self._fetch_threading, endpoint,
                                                  params_copy)
                        time.sleep(delay)
                        results.__dict__[str(id(results))] = elem
                        output.append(results)

                    for f in concurrent.futures.as_completed(output):
                        try:
                            response = f.result()
                            if f.__dict__['_result'] == None:
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
                    recursive_output = final_output + self._request(endpoint=endpoint,
                                                                    tickers=list(set(tickers_server)),
                                                                    params=params,
                                                                    delay=delay, count=count)
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

    def get_ts_expression(self, expression, original_metrics, suppress_warning,
                          **kwargs):
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
            assert metric in ['value', 'eop_lag', 'mop_lag', 'grading'], \
                f"Incorrect metric passed: {metric}."

        unique_tix = list(set(expression))
        dq_tix = []
        for metric in original_metrics:
            dq_tix += ["DB(JPMAQS," + tick + f",{metric})" for tick in unique_tix]

        expression = dq_tix

        c_delay = self.delay_compute(len(dq_tix))
        results = self._request(endpoint="/expressions/time-series",
                                tickers=expression, params={},
                                delay=c_delay, **kwargs)

        while results is None:
            c_delay += 0.1
            results = self._request(endpoint="/expressions/time-series",
                                    tickers=expression, params={},
                                    delay = c_delay, **kwargs)

        no_metrics = len(set([tick.split(',')[-1][:-1] for tick in expression]))

        results_dict, output_dict, s_list = self.isolate_timeseries(results,
                                                                    original_metrics,
                                                                    self.debug,
                                                                    False)
        if s_list:
            sequential = True
            self.__dict__['concurrent'] = False
            results_seq = self._request(endpoint="/expressions/time-series",
                                        tickers=s_list, params={}, **kwargs)
            r_dict, o_dict, s_list = self.isolate_timeseries(results_seq,
                                                             original_metrics,
                                                             self.debug,
                                                             sequential=sequential)
            results_dict = {**results_dict, **r_dict}

        results_dict = self.valid_ticker(results_dict, suppress_warning)

        results_copy = results_dict.copy()
        try:
            results_copy.popitem()
        except Exception as err:
            print(err)
            print("None of the tickers are available in the Database.")
            return
        else:
            return self.dataframe_wrapper(results_dict, no_metrics,
                                          original_metrics)

    @staticmethod
    def isolate_timeseries(list_, metrics, debug, sequential):
        """
        Isolates the metrics, across all categories & cross-sections, held in the List,
        and concatenates the time-series, column-wise, into a single structure, and
        subsequently stores that structure in a dictionary where the dictionary's
        keys will be each Ticker.
        Will validate that each requested metric is available, in the data dictionary,
        for each Ticker. If not, will run the Tickers sequentially to confirm the issue
        is not ascribed to multithreading overloading the load balancer.

        :param list_: returned from DataQuery.
        :param metrics: metrics requested from the API.
        :param debug: used to understand any underlying issue.
        :param sequential: if series are not returned, potentially the fault of the
            threading mechanism, isolate each Ticker and run sequentially.

        :return: <dict> modified_dict.
        """
        output_dict = defaultdict(dict)
        size = len(list_)

        for i in range(size):
            flag = False
            try:
                r = list_.pop()
            except IndexError:
                break
            else:
                dictionary = r['attributes'][0]
                ticker = dictionary['expression'].split(',')
                metric = ticker[-1][:-1]

                ticker_split = ','.join(ticker[:-1])
                ts_arr = np.array(dictionary['time-series'])
                if ts_arr.size == 1:
                    flag = True

                if not flag:
                    if ticker_split not in output_dict:
                        output_dict[ticker_split]['real_date'] = ts_arr[:, 0]
                        output_dict[ticker_split][metric] = ts_arr[:, 1]
                    elif metric not in output_dict[ticker_split]:
                        output_dict[ticker_split][metric] = ts_arr[:, 1]
                    else:
                        continue

        output_dict_c = output_dict.copy()
        t_dict = next(iter(output_dict_c.values()))
        no_rows = next(iter(t_dict.values())).size

        modified_dict = {}
        d_frame_order = ['real_date'] + metrics

        ticker_list = []
        for k, v in output_dict.items():

            arr = np.empty(shape=(no_rows, len(d_frame_order)), dtype=object)
            clause = True
            for i, metric in enumerate(d_frame_order):
                try:
                    arr[:, i] = v[metric]
                except KeyError:
                    if debug:
                        print(f"The ticker, {k[3:]}, is missing the metric '{metric}' "
                              f"whilst the requests are running concurrently - will "
                              f"check the API sequentially.")

                    temp_list = [k + ',' + m + ')' for m in metrics]
                    ticker_list += temp_list
                    if sequential:
                        if 'value' in v.keys():
                            arr[:, i] = np.nan
                        else:
                            print(f"The ticker, {k[3:]}, is missing from the API after "
                                  f"running sequentially - will not be in the returned "
                                  f"dataframe.")
                            clause = False
                            break
                    else:
                        break

            if clause:
                modified_dict[k] = arr

        return modified_dict, output_dict, ticker_list

    @staticmethod
    def column_check(v, col):
        """
        Checking the values of the returned TimeSeries.

        :param <np.array> v:
        :param <Integer> col: used to isolate the column being checked.

        :return <bool> condition.
        """
        returns = list(v[:, col])
        condition = all([isinstance(elem, type(None)) for elem in returns])

        return condition

    def valid_ticker(self, _dict, suppress_warning):
        """
        Iterates through each Ticker and determines whether the Ticker is held in the
        DataBase or not. The validation mechanism will isolate each column, in all the
        Tickers held in the dictionary, where the columns reflect the metrics passed,
        and validates that each value is not a NoneType Object. If all values are
        NoneType Objects, the Ticker is not valid, and it will be popped from the
        dictionary.

        :param <dict> _dict:
        :param <bool> suppress_warning:

        :return: <dict> dict_copy.
        """

        ticker_missing = 0
        dict_copy = _dict.copy()
        for k, v in _dict.items():
            no_cols = v.shape[1]

            condition = self.column_check(v, 1)
            if condition:
                ticker_missing += 1
                for i in range(2, no_cols):
                    condition = self.column_check(v, i)
                    if not condition:
                        if self.debug:
                            warnings.warn("Error has occurred in the DataBase.")

                if not suppress_warning:
                    print(f"The ticker, {k}), does not exist in the Database.")
                dict_copy.pop(k)
            else:
                continue

        print(f"Number of missing time-series from the DataBase: {ticker_missing}.")
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

            ticker = k.split(',')
            ticker = ticker[1].split('_')

            cid = ticker[0]
            xcat = '_'.join(ticker[1:])

            cid_broad = np.repeat(cid, repeats=v.shape[0])
            xcat_broad = np.repeat(xcat, repeats=v.shape[0])
            data = np.column_stack((cid_broad, xcat_broad, v))

            row = i * v.shape[0]
            arr[row:row + v.shape[0], :] = data
            i += 1

        columns = ['cid', 'xcat', 'real_date']
        cols_output = columns + original_metrics

        df = pd.DataFrame(data=arr, columns=cols_output)

        df['real_date'] = pd.to_datetime(df['real_date'], yearfirst=True)
        df = df[df['real_date'].dt.dayofweek < 5]
        df = df.fillna(value=np.nan)
        df = df.reset_index(drop=True)

        for m in original_metrics:
            df[m] = df[m].astype(dtype=np.float32)

        df.real_date = pd.to_datetime(df.real_date)
        return df

    def tickers(self, tickers: list, metrics: list = ['value'],
                start_date: str='2000-01-01', suppress_warning=False):
        """
        Returns standardized dataframe of specified base tickers and metric/

        :param <List[str]> tickers: JPMaQS ticker of form <cid>_<xcat>.
        :param <List[str]> metrics: must choose one or more from 'value', 'eop_lag',
            'mop_lag', or 'grading'. Default is ['value'].
        :param <str> start_date: first date in ISO 8601 string format.
        :param <bool> suppress_warning: used to suppress warning of any invalid
            ticker received by DataQuery.

        :return <pd.Dataframe> standardized dataframe with columns 'cid', 'xcats',
            'real_date' and chosen metrics.
        """

        df = self.get_ts_expression(expression=tickers, original_metrics=metrics,
                                    start_date=start_date,
                                    suppress_warning=suppress_warning)

        if isinstance(df, pd.DataFrame):
            df = df.sort_values(['cid', 'xcat', 'real_date']).reset_index(drop=True)

        return df

    def download(self, tickers=None, xcats=None, cids=None, metrics=['value'],
                 start_date='2000-01-01', suppress_warning=False):
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
            or 'grade'. Default is ['value'].
        :param <str> start_date: first date in ISO 8601 string format.
        :param <str> path: relative path from notebook to credential files.
        :param <bool> suppress_warning: used to suppress warning of any invalid
            ticker received by DataQuery.

        :return <pd.Dataframe> df: standardized dataframe with columns 'cid', 'xcats',
            'real_date' and chosen metrics.
        """

        if (cids is None) & (xcats is not None):
            cids_dmca = ['AUD', 'CAD', 'CHF', 'EUR', 'GBP', 'JPY', 'NOK', 'NZD', 'SEK',
                         'USD']  # DM currency areas
            cids_dmec = ['DEM', 'ESP', 'FRF', 'ITL', 'NLG']  # DM euro area countries
            cids_latm = ['BRL', 'COP', 'CLP', 'MXN', 'PEN']  # Latam countries
            cids_emea = ['HUF', 'ILS', 'PLN', 'RON', 'RUB', 'TRY', 'ZAR']  # EMEA countries
            cids_emas = ['CZK', 'CNY', 'IDR', 'INR', 'KRW', 'MYR', 'PHP', 'SGD', 'THB',
                         'TWD']  # EM Asia countries
            cids_dm = cids_dmca + cids_dmec
            cids_em = cids_latm + cids_emea + cids_emas
            cids = sorted(cids_dm + cids_em)  # standard default

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
            add_tix = [cid + '_' + xcat for xcat in xcats for cid in cids]
            tickers = tickers + add_tix

        df = self.tickers(tickers, metrics=metrics,
                          suppress_warning=suppress_warning,
                          start_date=start_date)

        return df
