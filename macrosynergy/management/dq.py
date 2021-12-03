"""DataQuery Interface

Interface to the JP Morgan DataQuery and how to interact with the API.

Requires a API login (username + password) as well as a certified certificate
and private key to verify the request.
"""
import requests
import base64
from typing import Optional, Union, List
import json
import pandas as pd
import numpy as np
import os
import logging
from math import ceil
from collections import defaultdict
import warnings
import threading
import concurrent.futures
import time

BASE_URL = "https://platform.jpmorgan.com/research/dataquery/api/v2"

DQ_ERROR_MSG = {
    204: "Content requested unavailable.",
    400: "Request but it was malformed or invalid.",
    401: "The user has not successfully authenticated with the service.",
    500: "There was an internal server error."
}


class DataQueryInterface(object):
    """
    Initiate the object DataQueryInterface

    ©JP Morgan

    DataQuery Data Universet
    * Fixed Income
    * Securitzed Products
    * Credit Products
    * Emerging Markets: Cash bonds, swaps, ...
    * Foreign Exchange rates

    Authentication:
      1. Client authentication through 2-way SSL
      2. User authentication (HTTP basic)

    Restriction of 5 concurrent requests per second per certificate.

    There is a difference between "Basic" and "Premium" user access
    in API v2.0

    Functional overview
      1. "/groups"
      2. "/groups/search"
      ...

    API end points? End point categories:
      1. Catalog discovery: "/group", "/group/search", ...
      2. Reference data extraction
      3. Market data extraction

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
                 concurrent: bool = True):

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

        # For debugging
        self.debug = debug
        self.last_url = None
        self.status_code = None
        self.last_response = None
        self.date_all = date_all
        self.concurrent = concurrent

        # assert self.check_connection()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type:
            print(f'exc_type: {exc_type}')
            print(f'exc_value: {exc_value}')
            print(f'exc_traceback: {exc_traceback}')

    @staticmethod
    def _debug_response(response: requests.Response) -> None:
        """Debug error response from DataQuery request

        :param <requests.Response> response:
            response from REST API request to DataQuery.
        :return: None

        """
        print("[", response.status_code, "] -", response.text)
        print("apparent encoding:", response.apparent_encoding)
        print("cookies:", response.cookies)
        print("elapsed:", response.elapsed)
        print("encoding:", response.encoding)
        print("history:", response.history, "permanent redirect:",
              response.is_permanent_redirect,
              "redirect:", response.is_redirect)
        # print("JSON:", response.json())
        print("links:", response.links)
        print("OK:", response.ok)
        print("reason:", response.reason)
        print("request:", response.request)
        print("url:", response.url)
        print("text:", response.text)
        print("raw:", response.raw)
        print("response headers:", response.headers)
        print("Send headers:", response.request.headers)

    def _fetch(self, endpoint: str = "/groups", select: str = "groups",
               params: dict = None) -> Optional[Union[list, dict]]:
        """Fetch the response from DataQuery

        :param <str> endpoint: default '/groups',
            end-point of DataQuery to be explored.
        :param <str> select: default 'groups',
            string with select for within the endpoint.
        :param <str> params: dictionary of parameters to be passed to request

        :return: list of response from DataQuery
        :rtype: <list>

        """

        # TODO map select to endpoint
        assert isinstance(select, str), \
            f"select must be a string and not {type(select)}: {select}"

        check = ["instruments", "groups", "filters", "info"]
        assert select in check, \
            f"select statement, {select:s} not found in list {check}"

        assert isinstance(endpoint, str), \
            f"endpoint must be a <str> and not {type(endpoint)}: {endpoint}"

        url = self.base_url + endpoint
        self.last_url = url

        logging.debug(f"request from endpoint: {url:s}")

        # TODO count/checksum of items...
        results = []
        count = 0

        while True:
            count += 1
            # TODO move to separate function...
            with requests.get(url=url, cert=(self.crt, self.key),
                              headers=self.headers, params=params) as r:
                self.status_code = r.status_code = r.status_code
                self.last_response = r.text

                self.last_url = r.url

                if self.debug:
                    self._debug_response(response=r)

                if r.status_code != requests.codes.ok:
                    code = r.status_code
                    msg = f"response status code {r.status_code:d}"
                    if code in DQ_ERROR_MSG.keys():
                        msg += f": {DQ_ERROR_MSG[code]:s}"

                    logging.error(msg)
                    return None

                if r.status_code != 200:
                    logging.warning(f"response code {r.status_code:d}")

            response = json.loads(self.last_response)

            if "info" in response.keys():
                # TODO check select == 'info'?
                return response["info"]

            if "error" in response.keys() or "errors" in response.keys():
                logging.error(f"Error in response %s for url %s", response, self.last_url)
                raise ValueError(f"Error in response from DQ: {response}")

            logging.debug(f"count: %d, items: %d, page-size: %d", count, response["items"], response["page-size"])

            # TODO parse response...
            assert select in response.keys()
            results.extend(response[select])

            assert "links" in response.keys(), \
                f"'links' not found in keys {response.keys()}"

            assert "next" in response['links'][1].keys(), \
                f"'next' missing from links keys:" \
                f" {response['links'][1].keys()}"

            if response["links"][1]["next"] is None:
                break

            url = f"{self.base_url:s}{response['links'][1]['next']:s}"
            params = {}

        return results

    def _fetch_threading(self, endpoint, params: dict):

        url = self.base_url + endpoint
        select = "instruments"

        results = []

        while True:
            with requests.get(url=url, cert=(self.crt, self.key), headers=self.headers,
                              params=params) as r:
                last_response = r.text

            response = json.loads(last_response)
            results.extend(response[select])

            assert "next" in response['links'][1].keys(), \
                f"'next' missing from links keys: " \
                f" {response['links'][1].keys()}"

            if response["links"][1]["next"] is None:
                break

            url = f"{self.base_url:s}{response['links'][1]['next']:s}"
            params = {}

        return results

    def check_connection(self) -> bool:
        """Check connect (heartbeat) to DataQuery

        :return: success of connection check if True (return code 200),
            and False otherwise.
        :rtype: <bool>

        """

        results = self._fetch(endpoint="/services/heartbeat", select='info')

        assert isinstance(results, dict), f"Response from DQ: {results}"

        if int(results["code"]) != 200:
            msg = f"Message: {results['message']:s}," \
                  f" Description: {results['description']:s}"
            logging.error(msg)

        return int(results["code"]) == 200

    def _request(self, endpoint: str, tickers: List[str], params: dict,
                 start_date: str = None, end_date: str = None,
                 calendar: str = "CAL_ALLDAYS", frequency: str = "FREQ_DAY",
                 conversion: str = "CONV_LASTBUS_ABS",
                 nan_treatment: str = "NA_NOTHING"):

        params_ = {"format": "JSON", "start-date": start_date, "end-date": end_date,
                   "calendar": calendar, "frequency": frequency, "conversion":
                   conversion, "nan_treatment": nan_treatment}
        params.update(params_)

        no_tickers = len(tickers)
        iterations = ceil(no_tickers / 20)

        tick_list_compr = [tickers[(i * 20): (i * 20) + 20] for i in range(iterations)]

        no_batches = len(tick_list_compr)
        exterior_iterations = ceil(no_batches / 10)

        final_output = []
        output = []
        if self.concurrent:
            for i in range(exterior_iterations):
                if i > 0: time.sleep(0.3)
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    for elem in tick_list_compr[(i * 10):(i + 1) * 10]:
                        params["expressions"] = elem
                        results = executor.submit(self._fetch_threading, endpoint, params)
                        time.sleep(0.75)
                        output.append(results)

                    for f in concurrent.futures.as_completed(output):
                        try:
                            response = f.result()
                        except ValueError:
                            raise ValueError("Server being hit too quickly with requests.")
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

        return final_output

    def get_ts_expression(self, expression, original_metrics, suppress_warning,
                          bool_df, **kwargs):
        """

        start_date: str = None, end_date: str = None,
                          calendar: str = "CAL_ALLDAYS",
                          frequency: str = "FREQ_DAY",
                          conversion: str = "CONV_LASTBUS_ABS",
                          nan_treatment: str = "NA_NOTHING"):

        Get timeseries (ts) using expression from old DataQuery notation.
        Will manipulate all the cross-sections

        :param expression: Categories & respective cross-sections requested.
        :param original_metrics: List of required metrics:
                                 the returned DataFrame will reflect the received List.
        :param **kwargs: dictionary of additional arguments
        :param bool_df: temporary parameter for reconciliation with Athena.

        :return: pd.DataFrame: ['cid', 'xcat', 'real_date'] + [original_metrics]

        """
        for metric in original_metrics:
            assert metric in ['value', 'eop_lag', 'mop_lag', 'grading'], \
                f"Incorrect metric passed: {metric}."

        unique_tix = list(set(expression))
        dq_tix = []
        for metric in original_metrics:
            dq_tix += ["DB(JPMAQS," + tick + f",{metric})" for tick in unique_tix]

        expression = dq_tix

        params = {}
        if "data" in kwargs.keys():
            assert kwargs["data"] == "ALL"
            params.update({"data":  kwargs.pop("data")})

        results = self._request(endpoint="/expressions/time-series",
                                tickers=expression, params=params, **kwargs)

        # (O(n) + O(nlog(n)) operation.
        no_metrics = len(set([tick.split(',')[-1][:-1] for tick in expression]))

        results_dict, output_dict = self.isolate_timeseries(results, original_metrics)
        if bool_df:
            df_column_wise = self.df_column(output_dict, original_metrics)
            return df_column_wise

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
    def isolate_timeseries(list_, metrics):
        """
        Isolates the metrics, across all categories & cross-sections, held in the List,
        and concatenates the time-series, column-wise, into a single structure, and
        subsequently stores that structure in a dictionary where the dictionary's
        keys will be each Ticker.

        :param: List returned from DataQuery.

        :return: dictionary.
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
                    print(f"Invalid expression, {ticker_split + ','+ metric + ')'}, "
                          f"passed into DataQuery.")
                    flag = True

                if not flag:
                    if ticker_split not in output_dict:
                        output_dict[ticker_split]['real_date'] = ts_arr[:, 0]
                        output_dict[ticker_split][metric] = ts_arr[:, 1]
                    else:
                        output_dict[ticker_split][metric] = ts_arr[:, 1]

        output_dict_c = output_dict.copy()
        t_dict = next(iter(output_dict_c.values()))
        no_rows = next(iter(t_dict.values())).size

        modified_dict = {}
        d_frame_order = ['real_date'] + metrics

        for k, v in output_dict.items():
            arr = np.empty(shape=(no_rows, len(d_frame_order)), dtype=object)
            for i, metric in enumerate(d_frame_order):
                arr[:, i] = v[metric]

            modified_dict[k] = arr

        return modified_dict, output_dict

    @staticmethod
    def column_check(v, col):
        """
        Checking the values of the returned TimeSeries.

        :param <np.array> v:
        :param <Integer> col: used to isolate the column being checked.

        :return Boolean.
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

        :param: Dictionary.

        :return: Dictionary.
        """

        dict_copy = _dict.copy()
        for k, v in _dict.items():
            no_cols = v.shape[1]

            condition = self.column_check(v, 1)
            if condition:
                for i in range(2, no_cols):
                    condition = self.column_check(v, i)
                    if not condition:
                        warnings.warn("Error has occurred in the DataBase.")

                if not suppress_warning:
                    print(f"The ticker, {k}, does not exist in the Database.")
                dict_copy.pop(k)
            else:
                continue

        return dict_copy

    @staticmethod
    def dataframe_wrapper(_dict, no_metrics, original_metrics):
        """
        Receives a Dictionary containing every Ticker and the respective time-series data
        held inside an Array. Will iterate through the dictionary and stack each Array
        into a single DataFrame retaining the order both row-wise, in terms of cross-
        sections, and column-wise, in terms of the metrics.

        :param <Dictionary> _dict:
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

        return df

    def tickers(self, tickers: list, metrics: list = ['value'],
                start_date: str='2000-01-01', suppress_warning=False,
                bool_df=False):
        """
        Returns standardized dataframe of specified base tickers and metric

        :param <List[str]> tickers: JPMaQS ticker of form <cid>_<xcat>.
        :param <List[str]> metrics: must choose one or more from 'value', 'eop_lag',
                                    'mop_lag', or 'grading'. Default is ['value'].
        :param <str> start_date: first date in ISO 8601 string format.
        :param <boolean> bool_df: temporary parameter (alignment with Athena).
        :param <boolean> suppress_warning: used to suppress warning of any invalid
                                           ticker received by DataQuery.

        :return <pd.Dataframe> standardized dataframe with columns 'cid', 'xcats',
                               'real_date' and chosen metrics.
        """

        df = self.get_ts_expression(expression=tickers, original_metrics=metrics,
                                    start_date=start_date,
                                    suppress_warning=suppress_warning,
                                    bool_df=bool_df)

        if isinstance(df, pd.DataFrame):
            df = df.sort_values(['cid', 'xcat', 'real_date']).reset_index(drop=True)

        return df

    def download(self, tickers=None, xcats=None, cids=None, metrics=['value'],
                 start_date='2000-01-01', suppress_warning=False):
        """
        Returns standardized dataframe of specified base tickers and metrics

        :param <List[str]> tickers: JPMaQS ticker of form <cid>_<xcat>. Can be combined
                                    with selection of categories.

        :param <List[str]> xcats: JPMaQS category codes. Downloaded for all standard
                                  cross sections identifiers available
        (if cids are not specified) or those selected (if cids are specified).
        Standard cross sections here include major developed and emerging currency
        markets. See JPMaQS documentation.
        :param <List[str]> cids: JPMaQS cross-section identifiers, typically based  on
                                 currency code. See JPMaQS documentation.
        :param <str> metrics: must choose one or more from 'value', 'eop_lag', 'mop_lag',
                              or 'grade'. Default is ['value'].

        :param <str> start_date: first date in ISO 8601 string format.
        :param <str> path: relative path from notebook to credential files.
        :param <boolean> suppress_warning: used to suppress warning of any invalid
                                           ticker received by DataQuery.

        :return <pd.Dataframe> standardized dataframe with columns 'cid', 'xcats',
                               'real_date' and chosen metrics.
        """

        # A. Collect all standard cross sections.

        if (cids is None) & (xcats is not None):
            cids_dmca = ['AUD', 'CAD', 'CHF', 'EUR', 'GBP', 'JPY', 'NOK', 'NZD', 'SEK',
                         'USD']  # DM currency areas
            cids_dmec = ['DEM', 'ESP', 'FRF', 'ITL', 'NLG']  # DM euro area countries
            cids_latm = ['BRL', 'COP', 'CLP', 'MXN', 'PEN']  # Latam countries
            cids_emea = ['HUF', 'ILS', 'PLN', 'RON', 'RUB', 'TRY', 'ZAR']  # EMEA countries
            cids_emas = ['CNY', 'IDR', 'INR', 'KRW', 'MYR', 'PHP', 'SGD', 'THB',
                         'TWD']  # EM Asia countries
            cids_dm = cids_dmca + cids_dmec
            cids_em = cids_latm + cids_emea + cids_emas
            cids = sorted(cids_dm + cids_em)  # standard default

        # B. Collect all tickers and metrics to be downloaded
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

        df = self.tickers(tickers, metrics=metrics, suppress_warning=suppress_warning,
                          start_date=start_date)

        return df
