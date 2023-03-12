""" JPMaQS Download Interface """

from typing import List, Optional, Tuple, Dict
import pandas as pd
import numpy as np
from collections import defaultdict
import warnings
from macrosynergy.download.dataquery import DataQueryInterface
from macrosynergy.download.exceptions import *
import datetime
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
    :param <bool> oauth: True if using oauth, False if using username/password with crt/key.
    :param <str> client_id: oauth client_id, required if oauth=True.
    :param <str> client_secret: oauth client_secret, required if oauth=True.
    :param <bool> debug: True if debug mode, False if not.
    :param <bool> suppress_warning: True if suppressing warnings, False if not.
    :param <bool> check_connection: True if the interface should check the connection to
        the server before sending requests, False if not. False by default.
    :param <dict> proxy: proxy to use for requests, None if not using proxy (default).
    :param <bool> print_debug_data: True if debug data should be printed, False if not
        (default).
    :param <dict> dq_kwargs: additional arguments to pass to the DataQuery API object such
        `calender` and `frequency` for the DataQuery API. For more fine-grained usage,
        initialize the DataQueryInterface object explicitly.
    :param <dict> kwargs: additional arguments to pass to the DataQuery API object such as
        <str> crt: path to crt file, <str> key: path to key file, <str> username: username
        for certificate based authentication, <str> password : paired with username for
        certificate.
        See macrosynergy.download.dataquery.DataQueryInterface for more.

    :return <JPMaQSDownload>: JPMaQSDownload object

    :raises <ValueError>: if provided arguments are invalid or semantically incorrect.

    """

    def __init__(
        self,
        oauth: bool = True,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        check_connection: bool = True,
        proxy: Optional[Dict] = None,
        suppress_warning: bool = True,
        debug: bool = False,
        print_debug_data: bool = False,
        dq_download_kwargs: dict = {},
        **kwargs,
    ):
        try:
            assert isinstance(oauth, bool), "`oauth` must be a boolean."
            assert isinstance(
                check_connection, bool
            ), "`check_connection` must be a boolean."
            assert isinstance(
                suppress_warning, bool
            ), "`suppress_warning` must be a boolean."
            assert isinstance(debug, bool), "`debug` must be a boolean."
            assert isinstance(
                print_debug_data, bool
            ), "`print_debug_data` must be a boolean."
            assert (
                isinstance(proxy, dict) or proxy is None
            ), "`proxy` must be a dictionary or None."
            assert isinstance(
                dq_download_kwargs, dict
            ), "`dq_download_kwargs` must be a dictionary."
        except AssertionError as e:
            raise ValueError(e)
        except Exception as e:
            raise e

        self.suppress_warning = suppress_warning
        self.debug = debug
        self.print_debug_data = print_debug_data
        self._check_connection = check_connection
        self.dq_download_kwargs = dq_download_kwargs

        if oauth:
            self.dq_interface: DataQueryInterface = DataQueryInterface(
                oauth=oauth,
                client_id=client_id,
                client_secret=client_secret,
                check_connection=check_connection,
                proxy=proxy,
                **kwargs,
            )
        else:
            # ensure "crt", "key", "username", and "password" are in kwargs
            for key in ["crt", "key", "username", "password"]:
                if key not in kwargs:
                    raise ValueError(f"Missing required argument {key}")

            crt = kwargs.pop("crt")
            key = kwargs.pop("key")
            username = kwargs.pop("username")
            password = kwargs.pop("password")

            self.dq_interface: DataQueryInterface = DataQueryInterface(
                oauth=oauth,
                check_connection=check_connection,
                crt=crt,
                key=key,
                username=username,
                password=password,
                proxy=proxy,
                **kwargs,
            )

        if self._check_connection:
            self.check_connection()

    @staticmethod
    def construct_expressions(
        tickers: Optional[List[str]] = None,
        cids: Optional[List[str]] = None,
        xcats: Optional[List[str]] = None,
        metrics: Optional[List[str]] = None,
    ) -> List[str]:
        """Construct expressions from the provided arguments.

        :param <list[str]> tickers: list of tickers.
        :param <list[str]> cids: list of cids.
        :param <list[str]> xcats: list of xcats.
        :param <list[str]> metrics: list of metrics.

        :return <list[str]>: list of expressions.
        """

        if tickers is None:
            tickers = []
        if cids is not None and xcats is not None:
            tickers += [f"{cid}_{xcat}" for cid in cids for xcat in xcats]

        if expressions is None:
            expressions = []
        expressions += [
            f"DB(JPMAQS,{tick},{metric})" for tick in tickers for metric in metrics
        ]

        return expressions

    def time_series_to_df(self, dicts_list: List[Dict]) -> pd.DataFrame:
        """
        Convert the downloaded data to a pandas DataFrame.
        Parameters
        :param dicts_list <list>: List of dictionaries containing time series
            data from the DataQuery API
        Returns
        :return <pd.DataFrame>: DataFrame containing the data
        """
        # TODO : make sure all metrics are supported
        dfs: List = []
        for d in dicts_list:
            df = pd.DataFrame(
                d["attributes"][0]["time-series"], columns=["real_date", "value"]
            )
            df["expression"] = d["attributes"][0]["expression"]
            dfs += [df]

        return_df = pd.concat(dfs, axis=0).reset_index(drop=True)[
            ["real_date", "expression", "value"]
        ]
        return_df["real_date"] = pd.to_datetime(return_df["real_date"])
        return return_df

    def validate_downloaded_df(
        self, data_df: pd.DataFrame, expressions: List[str], verbose: bool = True
    ) -> bool:
        """Validate the downloaded data.
        :param data_df <pd.DataFrame>: DataFrame containing the downloaded data.
        :param expressions <list>: List of expressions used to download the data.
        :param verbose <bool>: Whether to print the validation results.

        :return <bool>: True if valid, False if not.
        """
        # TODO : Complete this function to report number of missing values
        # if verbose print number of reqd. expressions and num of actual expressions in df
        if data_df.empty:
            return False
        return True

    def check_connection(self, verbose: bool = False) -> bool:
        """Check if the interface is connected to the server.
        :return <bool>: True if connected, False if not.
        """
        return self.dq_interface.check_connection(verbose=verbose)

    def validate_download_args(
        self,
        tickers: List[str],
        cids: List[str],
        xcats: List[str],
        metrics: List[str],
        start_date: str,
        end_date: str,
        expressions: List[str],
        show_progress: bool,
        as_dataframe: bool,
    ) -> bool:
        """Validate the arguments passed to the download function.

        :params -- see macrosynergy.download.jpmaqs.JPMaQSDownload.download()

        :return <bool>: True if valid.

        :raises <ValueError>: if provided arguments are invalid or
            semantically incorrect.
        """

        def is_valid_date(date: str) -> bool:
            try:
                datetime.datetime.strptime(date, "%Y-%m-%d")
                return True
            except ValueError:
                return False

        try:
            assert (
                isinstance(tickers, list) or tickers is None
            ), "`tickers` must be a list of strings."
            if tickers is not None:
                assert all(
                    [isinstance(ticker, str) for ticker in tickers]
                ), "`tickers` must be a list of strings."
            assert (
                isinstance(cids, list) or cids is None
            ), "`cids` must be a list of strings."
            if cids is not None:
                assert all(
                    [isinstance(cid, str) for cid in cids]
                ), "`cids` must be a list of strings."
            assert (
                isinstance(xcats, list) or xcats is None
            ), "`xcats` must be a list of strings."
            if xcats is not None:
                assert all(
                    [isinstance(xcat, str) for xcat in xcats]
                ), "`xcats` must be a list of strings."

            # if specifying cids then xcats must be not be None and vice versa
            estr = (
                "If specifying `cids`, `xcats` must also be specified and "
                "vice versa. Both can also be None."
            )
            if cids is not None:
                assert xcats is not None, estr
            else:
                assert xcats is None, estr

            assert isinstance(metrics, list), "`metrics` must be a list of strings."
            assert (
                all([isinstance(metric, str) for metric in metrics])
                and len(metrics) > 0
            ), "`metrics` must be a list of strings."
            assert (
                isinstance(expressions, list) or expressions is None
            ), "`expressions` must be a list of strings."
            if expressions is not None:
                assert all(
                    [isinstance(expression, str) for expression in expressions]
                ), "`expressions` must be a list of strings."
            assert isinstance(show_progress, bool), "`show_progress` must be a boolean."
            assert isinstance(as_dataframe, bool), "`as_dataframe` must be a boolean."
            assert is_valid_date(
                start_date
            ), "`start_date` must be a valid date in the format YYYY-MM-DD."
            assert (
                is_valid_date(end_date) or end_date is None
            ), "`end_date` must be a valid date in the format YYYY-MM-DD."

            if (
                tickers is None
                and cids is None
                and xcats is None
                and expressions is None
            ):
                raise ValueError(
                    "Must provide at least one of `tickers`, "
                    "`expressions`, or `cids` and `xcats` together."
                )

        except AssertionError as e:
            raise ValueError(e)

        except Exception as e:
            raise e

        return True

    def download(
        self,
        tickers=None,
        cids=None,
        xcats=None,
        metrics=["value"],
        start_date="2000-01-01",
        end_date=None,
        expressions=None,
        show_progress=False,
        as_dataframe=True,
    ) -> Optional[pd.DataFrame | List[Dict]]:
        """Driver function to download data from JPMaQS via the DataQuery API.
        Timeseries data can be requested using `tickers` with `metrics`, or
        passing formed DataQuery expressions.
        `cids` and `xcats` (along with `metrics`) are used to construct
        expressions, which are ultimately passed to the DataQuery Interface.

        :param <list[str]> tickers: list of tickers.
        :param <list[str]> cids: list of cids.
        :param <list[str]> xcats: list of xcats.
        :param <list[str]> metrics: list of metrics, one of "value", "grading",
            "eop_lag", "mop_lag".
        :param <str> start_date: start date of the data to download, in the
            ISO format - YYYY-MM-DD.
        :param <str> end_date: end date of the data to download in the ISO
            format - YYYY-MM-DD.
        :param <list[str]> expressions: list of DataQuery expressions.
        :param <bool> show_progress: True if progress bar should be shown,
            False if not (default).
        :param <bool> suppress_warning: True if suppressing warnings. Default
            is True.
        :param <bool> debug: True if debug mode, False if not (default).
        :param <bool> print_debug_data: True if debug data should be printed,
            False if not (default). If debug=True, this is set to True.
        :param <bool> as_dataframe: Return a dataframe if True (default),
            a list of dictionaries if False.

        :return <pd.DataFrame|list[Dict]>: dataframe of data if
            `as_dataframe` is True, list of dictionaries if False.

        :raises <ValueError>: if provided arguments are invalid or
            semantically incorrect (see
            macrosynergy.download.jpmaqs.JPMaQSDownload.validate_download_args()).

        """

        if all([_arg is None for _arg in [tickers, cids, xcats, expressions]]):
            expressions: List[str] = [
                "DB(JPMAQS,USD_EQXR_VT10,value)",
                "DB(JPMAQS,AUD_EXALLOPENNESS_NSA_1YMA,value)",
            ]
        # NOTE : This is simply so that we can test the download() function
        #   without having to pass in a bunch of arguments.

        # Validate arguments.
        if not self.validate_download_args(
            tickers=tickers,
            cids=cids,
            xcats=xcats,
            metrics=metrics,
            start_date=start_date,
            end_date=end_date,
            expressions=expressions,
            show_progress=show_progress,
            as_dataframe=as_dataframe,
        ):
            raise ValueError("Invalid arguments passed to download().")

        # Construct expressions.
        expressions = self.construct_expressions(
            tickers=tickers,
            cids=cids,
            xcats=xcats,
            metrics=metrics,
        )

        if end_date is None:
            end_date = (datetime.datetime.today() + pd.offsets.BusinessDay(2)).strftime(
                "%Y-%m-%d"
            )
            # NOTE : due to timezone conflicts, we choose to request data for 2 days in the future.
            # NOTE : DataQuery specifies YYYYMMDD as the date format, but we use YYYY-MM-DD for consistency.
            #   This is date is cast to YYYYMMDD in macrosynergy.download.dataquery.py.

        # Download data.
        with self.dq_interface as dq:
            print(
                "Downloading data from JPMaQS. Timestamp UTC: ",
                datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            )
            data: List[Dict] = dq.download_data(
                expressions=expressions,
                start_date=start_date,
                end_date=end_date,
                show_progress=show_progress,
                **self.dq_download_kwargs,
            )

        if as_dataframe:
            data = self.time_series_to_df(dicts_list=data)
            if not self.validate_downloaded_df(data=data, expressions=expressions):
                raise ValueError("Invalid dataframe returned by download().")

        return data
