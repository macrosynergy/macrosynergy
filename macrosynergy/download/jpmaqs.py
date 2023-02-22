""" JPMaQS Download Interface """

from typing import List, Optional, Tuple
import pandas as pd
import numpy as np
from collections import defaultdict
import warnings
from macrosynergy.download import dataquery
from macrosynergy.download.exceptions import *
import logging
import io

logger = logging.getLogger(__name__)
debug_stream_handler = logging.StreamHandler(io.StringIO())
debug_stream_handler.setLevel(logging.NOTSET)
debug_stream_handler.setFormatter(
    logging.Formatter(
        "%(asctime)s - %(levelname)s - %(module)s - %(funcName)s :: %(message)s"
    )
)
logger.addHandler(debug_stream_handler)


class JPMaQSDownload(object):
    """JPMaQS Download Interface Object
    :param <bool> oauth: True if using oauth, False if using username/password with crt/key
    :param <str> client_id: oauth client_id, required if oauth=True
    :param <str> client_secret: oauth client_secret, required if oauth=True
    :param <bool> debug: True if debug mode, False if not
    :param <bool> suppress_warning: True if suppress warning, False if not
    :param <bool> check_connection: True if the interface should check the connection to
        the server before sending requests, False if not. False by default.
    :param <dict> kwargs: additional arguments to pass to the DataQuery API object such as
        <str> crt: path to crt file, <str> key: path to key file, <str> username: username
        for certificate based authentication, <str> password : paired with username for
        certificatem, <dict> proxy: proxy server(s) to be used for requests,
        <str> base_url, etc.
    """

    def __init__(
        self,
        oauth: bool = True,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        suppress_warning: bool = True,
        debug: bool = False,
        print_debug_data: bool = False,
        check_connection: bool = False,
        **kwargs,
    ):
        self.debug = debug
        self.print_debug_data = print_debug_data
        self.msg_errors: List[str] = []
        self.suppress_warning = suppress_warning
        self.proxy = kwargs.pop("proxy", kwargs.pop("proxies", None))
        self.unavailable_tickers = []
        if client_id is None and client_secret is None:
            oauth = False

        if oauth:
            if not (isinstance(client_id, str) and isinstance(client_secret, str)):
                raise ValueError("client_id and client_secret must be strings.")

            dq_args = {
                "client_id": client_id,
                "client_secret": client_secret,
                "proxy": self.proxy,
            }
        else:
            username = kwargs.get("username", None)
            password = kwargs.get("password", None)
            crt = kwargs.get("crt", None)
            key = kwargs.get("key", None)
            dq_args = {
                "username": username,
                "password": password,
                "crt": crt,
                "key": key,
                "proxy": self.proxy,
            }
        if "heartbeat" in kwargs:
            check_connection = kwargs.pop("heartbeat")
        dq_args["heartbeat"] = check_connection
        dq_args["debug"] = debug
        dq_args["suppress_warning"] = suppress_warning
        dq_args["oauth"] = oauth

        self.dq_args = dq_args.copy()
        logger.info("JPMaQSDownload object created.")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        e_str = f"{exc_type} {exc_value} {traceback}"
        if exc_type:
            logger.error(e_str)
        if self.print_debug_data or exc_type:
            debug_stream_handler.stream.flush()
            debug_stream_handler.stream.seek(0)
            self.msg_errors += debug_stream_handler.stream.readlines()
            print("-" * 23, " DEBUG DATA ", "-" * 23)
            print("-" * 60)
            for m in self.msg_errors:
                print(m.strip())
            print(("-" * 60 + "\n") * 2)

        if exc_type:
            raise exc_type(exc_value)
        else:
            return True

    def array_construction(
        self, metrics: List[str], output_dict: dict, debug: bool, sequential: bool
    ):
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
        d_frame_order = ["real_date"] + metrics

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

                temp_list = ["DB(JPMAQS," + k + "," + m + ")" for m in metrics]
                ticker_list += temp_list

                if debug:
                    logger.warning(
                        f"The ticker, {k}, is missing the metric(s) "
                        f"'{missing_metrics}'. These will not be returned."
                        f"Check JPMaQSDownload."
                    )

            elif sequential and debug:
                logger.warning(
                    f"The ticker, {k}, is missing from the API after "
                    f"running sequentially - will not be in the returned "
                    f"DataFrame."
                )
            else:
                continue

        return modified_dict, ticker_list

    def isolate_timeseries(
        self, list_, metrics: List[str], debug: bool, sequential: bool
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
        # if debug:
        #     print(f"Number of returned expressions from JPMaQS: {size}.")

        unavailable_tickers = []
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
                unavailable_tickers.append(ticker_split)

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
            metrics=metrics,
            output_dict=output_dict_c,
            debug=debug,
            sequential=sequential,
        )
        if debug and not (self.suppress_warning) and len(unavailable_tickers) > 0:
            logger.warning(
                f"The following tickers were not returned from the API; as they are either invalid or unavailable: "
                f"{unavailable_tickers}. "
                "Appending list to JPMaQSDownload.unavailable_tickers."
            )

        self.unavailable_tickers += unavailable_tickers

        return modified_dict, output_dict, ticker_list

    def column_check(self, v, col, no_cols, debug):
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

    def dataframe_wrapper(self, _dict, no_metrics, original_metrics):
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
            arr[row : row + v.shape[0], :] = data
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

    def check_connection(self) -> Tuple[bool, dict]:
        """
        Interface to the DQ API's check_connection method,
        which in turn checks the connection (heartbeat) to the DQ API.
        """

        with dataquery.Interface(**self.dq_args) as dq:
            return dq.check_connection()

    @staticmethod
    def remove_jpmaqs_expr_formatting(expressions: List[str]) -> List[Tuple[str, str]]:

        """
        Removes the DB(JPMAQS, <ticker>, <metric>) formatting from a list of JPMaQS expressions.

        :param <List[str]> expressions: List of JPMaQS expressions.
        :return <List[Tuple[str, str]]>: List of tuples containing the ticker and metric.
        """

        return [
            e.replace("DB(JPMAQS,", "").replace(")", "").split(",") for e in expressions
        ]

    @staticmethod
    def jpmaqs_indicators(metrics, tickers):
        """
        Functionality used to convert tickers into formal JPMaQS expressions.
        """
        return [f"DB(JPMAQS,{tick},{metric})" for tick in tickers for metric in metrics]

    def download(
        self,
        tickers=None,
        xcats=None,
        cids=None,
        metrics=["value"],
        start_date="2000-01-01",
        end_date=None,
        suppress_warning=True,
        debug=False,
        print_debug_data=False,
        show_progress=False,
    ):
        """
        Downloads and returns a standardised DataFrame of the specified base tickers and metrics.

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
        :param <bool> debug: used to print out the tickers that are being downloaded.
        :param <bool> print_debug_data: used to print out debug information.
        :param <bool> show_progress: used to show progress bar. Default is False.

        :return <pd.Dataframe> df: standardized dataframe with columns 'cid', 'xcats',
            'real_date' and chosen metrics.
        """
        self.debug = self.debug or debug
        self.print_debug_data = self.print_debug_data or print_debug_data

        if self.suppress_warning != suppress_warning:
            self.suppress_warning = suppress_warning

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

        if isinstance(metrics, str):
            metrics = [metrics]
        if isinstance(xcats, str):
            xcats = [xcats]
        if isinstance(cids, str):
            cids = [cids]

        if isinstance(tickers, str):
            tickers = [tickers]
        elif tickers is None:
            tickers = []

        assert isinstance(metrics, list), "Metrics must be a list of strings"
        assert isinstance(tickers, list), "Tickers must be a list of strings"

        for metric in metrics:
            assert metric in [
                "value",
                "eop_lag",
                "mop_lag",
                "grading",
            ], f"Incorrect metric passed: {metric}."

        if xcats is not None:
            assert isinstance(xcats, list), "Xcats must be a list of strings"
            add_tix = [cid + "_" + xcat for cid in cids for xcat in xcats]
            tickers = tickers + add_tix
        self.dq_args["suppress_warning"] = suppress_warning
        self.dq_args["debug"] = debug

        tickers = list(set(tickers))  # Should this be stored in a copy?
        expressions = self.jpmaqs_indicators(metrics=metrics, tickers=tickers)

        logger.info(
            f"Downloading {len(expressions)} expressions from JPMaQS "
            f"for {len(tickers)} tickers & {len(metrics)} metrics. "
            f"Start date: {start_date}. End date: {end_date}."
        )

        try:
            with dataquery.Interface(**self.dq_args) as dq:
                dq_result_dict = dq.get_ts_expression(
                    expressions=expressions,
                    original_metrics=metrics,
                    start_date=start_date,
                    end_date=end_date,
                    suppress_warning=suppress_warning,
                    debug=debug,
                    show_progress=show_progress,
                )
            dq_msg_errors = dq.msg_errors
            debug_stream_handler.stream.write("\n".join(dq_msg_errors) + "\n")
            logger.info("Download complete. DataQuery interface closed.")

        except ConnectionError:
            logger.error("Failed to download data. ConnectionError.")
            logger.error("Appending error messages to JPMaQSDownload.download_output")
            self.download_output = dq_result_dict.copy()
            raise DownloadError(ConnectionError, "Failed to download data.")

        if dq_result_dict is None:
            logger.error("Failed to download data.")
            logger.error("Appending error messages to JPMaQSDownload.download_output")
            self.download_output = dq_result_dict.copy()
            raise DownloadError(
                "Failed to download data for some tickers. See log for details."
            )

        results = dq_result_dict["results"]
        error_tickers = dq_result_dict["error_tickers"]
        error_messages = dq_result_dict["error_messages"]
        unavailable_expressions = dq_result_dict["unavailable_expressions"]
        unavailable_expressions_list = list(u[0] for u in unavailable_expressions)

        self.unavailable_expressions = unavailable_expressions.copy()
        if error_tickers:
            logger.error(f"Error tickers: {error_tickers}")
            logger.error(f"Failed to download above tickers.")
            logger.error(f"Error messages: {error_messages}")
            logger.error("Appending error messages to JPMaQSDownload.download_output")
            self.download_output = dq_result_dict.copy()
            raise DownloadError(
                "Failed to download data for some tickers. See log for details."
            )
        else:
            logger.info("No error tickers; Starting to data parse.")
            results_dict, output_dict, s_list = self.isolate_timeseries(
                list_=results, metrics=metrics, debug=debug, sequential=True
            )  #

            if s_list:
                logger.warning(f"Warning tickers: {s_list}")
                logger.warning(
                    "Warning messages: Some of the tickers are not available in the Database."
                )
                if suppress_warning:
                    logger.warning("Warning suppressed.")
                if debug:
                    raise InvalidDataframeError(
                        "The database has missing entries for some expressions. See log for details."
                    )
                else:
                    logger.warning(
                        f"Debug mode is off; adding download ouput to JPMaQSDownload.download_output"
                        f"Debug mode is off; adding parsed output to JPMaQSDownload.parsed_output"
                    )
                self.download_output = dq_result_dict
                self.parsed_output = {
                    "results": results_dict,
                    "results_nested_dictionary": output_dict,
                    "missing_tickers": s_list,
                }

            logger.info("Data parse complete. Starting data validation.")

            results_dict = self.valid_ticker(results_dict, suppress_warning, self.debug)

            results_copy = results_dict.copy()
            try:
                results_copy.popitem()
            except Exception as err:
                logger.error(f"Error: {err}")
                logger.error("None of the tickers are available in the Database.")
                df = None
            else:
                no_metrics = len(
                    set([tick.split(",")[-1][:-1] for tick in expressions])
                )
                df = self.dataframe_wrapper(
                    _dict=results_dict, no_metrics=no_metrics, original_metrics=metrics
                )
            logger.info("Data validation complete. Creating and validating dataframe.")

            if (not isinstance(df, pd.DataFrame)) or (df.empty):
                logger.error("No data returned from DataQuery")
                if debug:
                    logger.error(
                        f"Debug mode is on; adding download ouput to JPMaQSDownload.download_output"
                        f"Debug mode is on; adding parsed output to JPMaQSDownload.parsed_output"
                    )
                    self.download_output = dq_result_dict
                    self.parsed_output = {
                        "results": results_dict,
                        "results_nested_dictionary": output_dict,
                        "missing_tickers": s_list,
                    }
                print("No data returned from DataQuery")
                raise InvalidDataframeError(
                    "No data returned from DataQuery. See log for details."
                )
            else:
                df = df.sort_values(["cid", "xcat", "real_date"]).reset_index(drop=True)
                logger.info("Dataframe created and validated.")
                logger.info("Returning dataframe and exiting JPMaQSDownload.download.")
                return df
