""" JPMaQS Download Interface 

::docs::JPMaQSDownload::sort_first::
"""

from typing import List, Optional, Dict, Union, Tuple
import pandas as pd
import traceback as tb
import datetime
import logging
import warnings
import io
import os
from timeit import default_timer as timer

import sys, os

sys.path.append(os.getcwd())

from macrosynergy.download.dataquery import DataQueryInterface
from macrosynergy.download.exceptions import *
from macrosynergy.management.utils import is_valid_iso_date

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
    """
    JPMaQSDownload Object. This object is used to download JPMaQS data via the DataQuery API.
    It can be extended to include the use of proxies, and even request generic DataQuery expressions.

    :param <bool> oauth: True if using oauth, False if using username/password with crt/key.

    When using oauth:
    :param <str> client_id: oauth client_id, required if oauth=True.
    :param <str> client_secret: oauth client_secret, required if oauth=True.

    When using username/password with crt/key:
    :param <str> crt: path to crt file.
    :param <str> key: path to key file.
    :param <str> username: username for certificate based authentication.
    :param <str> password : paired with username for certificate.

    When using a config file:
    :param <str> credentials_config: path to config file.

    The config file should contain the client_id and client_secret for oauth, or the
    crt, key, username, and password for certificate based authentication.
    (see macrosynergy.management.utils.JPMaQSAPIConfigObject)

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
            :param <str> base_url: base url for the DataQuery API.
            :param <str> calendar: calendar setting to use with the DataQuery API.
            :param <str> frequency: frequency setting to use with the DataQuery API.
            ...
        See macrosynergy.download.dataquery.DataQueryInterface for more.

    :return <JPMaQSDownload>: JPMaQSDownload object

    :raises <TypeError>: if provided arguments are not of the correct type.
    :raises <ValueError>: if provided arguments are invalid or semantically incorrect.

    """

    def __init__(
        self,
        oauth: bool = True,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        crt: Optional[str] = None,
        key: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        check_connection: bool = True,
        proxy: Optional[Dict] = None,
        suppress_warning: bool = True,
        debug: bool = False,
        print_debug_data: bool = False,
        dq_download_kwargs: dict = {},
        **kwargs,
    ):
        vars_types_zip: List[Tuple[str, str]] = [
            (oauth, "oauth", bool),
            (check_connection, "check_connection", bool),
            (suppress_warning, "suppress_warning", bool),
            (debug, "debug", bool),
            (print_debug_data, "print_debug_data", bool),
        ]

        for varx, namex, typex in vars_types_zip:
            if not isinstance(varx, typex):
                raise TypeError(f"`{namex}` must be of type {typex}.")

        if not isinstance(proxy, dict) and proxy is not None:
            raise TypeError("`proxy` must be a dictionary or None.")

        if not isinstance(dq_download_kwargs, dict):
            raise TypeError("`dq_download_kwargs` must be a dictionary.")

        self.suppress_warning = suppress_warning
        self.debug = debug
        self.print_debug_data = print_debug_data
        self._check_connection = check_connection
        self.dq_download_kwargs = dq_download_kwargs

        for varx, namex in [
            (client_id, "client_id"),
            (client_secret, "client_secret"),
            (crt, "crt"),
            (key, "key"),
            (username, "username"),
            (password, "password"),
        ]:
            if varx is not None:
                if not isinstance(varx, str):
                    raise TypeError(f"`{namex}` must be a string.")

        if not (all([client_id, client_secret]) or all([crt, key, username, password])):
            raise ValueError(
                "Must provide either `client_id` and `client_secret` for oauth, or "
                "`crt`, `key`, `username`, and `password` for certificate based authentication."
            )

        self.dq_interface: DataQueryInterface = DataQueryInterface(
            oauth=oauth,
            check_connection=check_connection,
            client_id=client_id,
            client_secret=client_secret,
            crt=crt,
            key=key,
            username=username,
            password=password,
            proxy=proxy,
            debug=debug,
            **kwargs,
        )

        self.valid_metrics: List[str] = ["value", "grading", "eop_lag", "mop_lag"]
        self.msg_errors: List[str] = []
        self.msg_warnings: List[str] = []
        self.unavailable_expressions: List[str] = []
        self.downloaded_data: Dict = {}

        if self._check_connection:
            self.check_connection()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type:
            print(f"Exception: {exc_type} {exc_value}")
            print(tb.print_exc())
            raise exc_type(exc_value)

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

        return [f"DB(JPMAQS,{tick},{metric})" for tick in tickers for metric in metrics]

    @staticmethod
    def deconstruct_expression(
        expression: Union[str, List[str]]
    ) -> Union[List[str], List[List[str]]]:
        """
        Deconstruct an expression into a list of cid, xcat, and metric.
        Coupled with to JPMaQSDownload.time_series_to_df(), achieves the inverse of
        JPMaQSDownload.construct_expressions().

        :param <str> expression: expression to deconstruct. If a list is provided,
            each element will be deconstructed and returned as a list of lists.

        :return <list[str]>: list of cid, xcat, and metric.

        :raises TypeError: if `expression` is not a string or a list of strings.
        :raises ValueError: if `expression` is an empty list.
        """
        if not isinstance(expression, (str, list)):
            raise TypeError("`expression` must be a string or a list of strings.")

        if isinstance(expression, list):
            if not all(isinstance(exprx, str) for exprx in expression):
                raise TypeError("All elements of `expression` must be strings.")
            elif len(expression) == 0:
                raise ValueError("`expression` must be a non-empty list.")
            return [
                JPMaQSDownload.deconstruct_expression(exprx) for exprx in expression
            ]
        else:
            try:
                exprx: str = expression.replace("DB(JPMAQS,", "").replace(")", "")
                ticker, metric = exprx.split(",")
                result: List[str] = ticker.split("_", 1) + [metric]
                if len(result) != 3:
                    raise ValueError(f"{exprx} is not a valid JPMaQS expression.")
                return ticker.split("_", 1) + [metric]
            except Exception as e:
                warnings.warn(
                    f"Failed to deconstruct expression `{expression}`: {e}",
                    UserWarning,
                )
                # fail safely, return list where all entries are =expression
                return [expression, expression, expression]

    def validate_downloaded_df(
        self,
        data_df: pd.DataFrame,
        expected_expressions: List[str],
        found_expressions: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        verbose: bool = True,
    ) -> bool:
        """
        Validate the downloaded data in the provided dataframe.

        :param <pd.DataFrame> data_df: dataframe containing the downloaded data.
        :param <list[str]> expected_expressions: list of expressions that were expected to be
            downloaded.
        :param <list[str]> found_expressions: list of expressions that were actually downloaded.
        :param <str> start_date: start date of the downloaded data.
        :param <str> end_date: end date of the downloaded data.
        :param <bool> verbose: whether to print the validation results.

        :return <bool>: True if the downloaded data is valid, False otherwise.

        :raises <TypeError>: if `data_df` is not a dataframe.
        """

        if not isinstance(data_df, pd.DataFrame):
            raise TypeError("`data_df` must be a dataframe.")
        if data_df.empty:
            return False

        # check if all expressions are present in the df
        exprs_f = set(found_expressions)
        expr_expected = set(expected_expressions)
        expr_missing = expr_expected - exprs_f
        self.unavailable_expressions += list(expr_missing)
        unexpected_exprs = exprs_f - expr_expected
        if unexpected_exprs:
            raise InvalidDataframeError(
                f"Unexpected expressions were found in the downloaded data: {unexpected_exprs}"
            )

        if expr_missing:
            log_str = (
                f"Some expressions are missing from the downloaded data."
                " Check logger output for complete list. \n"
                f"{len(expr_missing)} out of {len(expr_expected)} expressions are missing."
                f"To download the catalogue of all available expressions and filter the"
                " unavailable expressions, set `get_catalogue=True` in the "
                " call to `JPMaQSDownload.download()`."
            )

            logger.info(log_str)
            logger.info(f"Missing expressions: {expr_missing}")
            if verbose:
                print(log_str)

        # check if all dates are present in the df
        # NOTE : Hardcoded max start date to 1990-01-01. This is because the JPMAQS database
        #        does not have data before this date.
        if datetime.datetime.strptime(
            start_date, "%Y-%m-%d"
        ) < datetime.datetime.strptime("1990-01-01", "%Y-%m-%d"):
            start_date = "1950-01-01"

        dates_expected = pd.bdate_range(start=start_date, end=end_date)
        dates_missing = len(data_df) - len(
            data_df[data_df["real_date"].isin(dates_expected)]
        )

        if dates_missing > 0:
            log_str = (
                f"Some dates are missing from the downloaded data. \n"
                f"{len(dates_missing)} out of {len(dates_expected)} dates are missing."
            )
            logger.warning(log_str)
            if verbose:
                print(log_str)

        # check number of nas in each column
        nas = data_df.isna().sum(axis=0)
        nas = nas[nas > 0]
        if len(nas) > 0:
            log_str = (
                "Some columns have missing values.\n"
                f"Total rows : {len(data_df)} \n"
                "Missing values by column:\n"
                f"{nas.head(10)}"
            )
            logger.warning(log_str)
            if verbose:
                print(log_str)

        return True

    def time_series_to_df(
        self,
        dicts_list: List[Dict],
        validate_df: bool = True,
        expected_expressions: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Convert the downloaded data to a pandas DataFrame.

        :param dicts_list <list>: List of dictionaries containing time series
            data from the DataQuery API

        :return <pd.DataFrame>: JPMaQS standard dataframe with columns:
            real_date, cid, xcat, <metric>. The <metric> column contains the
            observed data for the given cid and xcat on the given real_date.

        :raises <InvalidDataError>: if the downloaded dataframe is invalid.
        """
        dfs: List[pd.DataFrame] = []
        cid: str
        xcat: str
        found_expressions: List[str] = []
        _missing_exprs: List[str] = []
        self.unavailable_expr_messages = []
        # _missing_exprs is just a sanity check to verify self.unavailable_expressions
        for d in dicts_list:
            cid, xcat, metricx = JPMaQSDownload.deconstruct_expression(
                d["attributes"][0]["expression"]
            )
            if d["attributes"][0]["time-series"] is not None:
                found_expressions.append(d["attributes"][0]["expression"])
                df: pd.DataFrame = (
                    pd.DataFrame(
                        d["attributes"][0]["time-series"],
                        columns=["real_date", metricx],
                    )
                    .assign(cid=cid, xcat=xcat, metric=metricx)
                    .rename(columns={metricx: "obs"})
                )
                df = df[["real_date", "cid", "xcat", "obs", "metric"]]
                dfs.append(df)
            else:
                _missing_exprs.append(d["attributes"][0]["expression"])
                if "message" in d["attributes"][0]:
                    self.unavailable_expr_messages.append(d["attributes"][0]["message"])
                else:
                    self.unavailable_expr_messages.append(
                        f"DataQuery did not return data or error message for expression {d['attributes'][0]['expression']}"
                    )

        assert set(_missing_exprs) == set(
            self.unavailable_expressions
        ), "Downloaded `dicts_list` has been modified before calling `time_series_to_df`"

        if len(dfs) == 0:
            raise InvalidDataframeError(
                "No data was downloaded. Check logger output for complete list of missing expressions."
            )

        final_df: pd.DataFrame = pd.concat(dfs, ignore_index=True)
        dups_df: pd.DataFrame = final_df.groupby(
            ["real_date", "cid", "xcat", "metric"]
        )["obs"].count()
        if sum(dups_df > 1) > 0:
            dups_df = pd.DataFrame(dups_df[dups_df > 1].index.tolist())
            err_str: str = "Duplicate data found for the following expressions:\n"
            for i in dups_df.groupby([1, 2, 3]).groups:
                dts_series: pd.Series = dups_df.iloc[
                    dups_df.groupby([1, 2, 3]).groups[i]
                ][0]
                dts: List[str] = dts_series.tolist()
                max_date: str = pd.to_datetime(max(dts)).strftime("%Y-%m-%d")
                min_date: str = pd.to_datetime(min(dts)).strftime("%Y-%m-%d")
                expression: str = self.construct_expressions(
                    cids=[i[0]], xcats=[i[1]], metrics=[i[2]]
                )[0]
                err_str += (
                    f"Expression: {expression}, Dates: {min_date} to {max_date}\n"
                )

            raise InvalidDataframeError(err_str)

        final_df = (
            final_df.set_index(["real_date", "cid", "xcat", "metric"])["obs"]
            .unstack(3)
            .rename_axis(None, axis=1)
            .reset_index()
        )  # thank you @mikiinterfiore

        final_df["real_date"] = pd.to_datetime(final_df["real_date"])

        # list expected business dates, and non-business dates
        expc_bdates: pd.DatetimeIndex = pd.bdate_range(start=start_date, end=end_date)
        expc_nbdates: pd.DatetimeIndex = pd.DatetimeIndex(
            list(set(pd.date_range(start=start_date, end=end_date)) - set(expc_bdates))
        )

        # check if any dates in the downloaded data are not in the expected dates (business + non-business)
        dates_bools: pd.Series = final_df["real_date"].isin(expc_bdates) | final_df[
            "real_date"
        ].isin(expc_nbdates)

        unexpected_dates: pd.DatetimeIndex = final_df[~dates_bools][
            "real_date"
        ].unique()

        if any(~dates_bools):
            raise InvalidDataframeError(
                f"Unexpected dates were found in the downloaded data: {unexpected_dates}"
            )

        # finally, filter out the non-business dates
        final_df = final_df[final_df["real_date"].isin(expc_bdates)]

        final_df = final_df.sort_values(["real_date", "cid", "xcat"])

        found_metrics = sorted(
            list(set(final_df.columns) - {"real_date", "cid", "xcat"}),
            key=lambda x: self.valid_metrics.index(x),
        )
        # sort found_metrics in the order of self.valid_metrics, then re-order the columns
        final_df = final_df[["real_date", "cid", "xcat"] + found_metrics]

        # IMPORTANT NOTE:
        # Drop NA containing rows to be revisited when blacklisting is implemented

        final_df = final_df.dropna(axis=0, how="any").reset_index(drop=True)

        if validate_df:
            vdf = self.validate_downloaded_df(
                data_df=final_df,
                expected_expressions=expected_expressions,
                found_expressions=found_expressions,
                start_date=start_date,
                end_date=end_date,
                verbose=verbose,
            )
            if not vdf:
                self.downloaded_data: Dict = {
                    "data_df": final_df,
                    "expected_expressions": expected_expressions,
                    "found_expressions": found_expressions,
                    "start_date": start_date,
                    "end_date": end_date,
                    "downloaded_dicts": dicts_list,
                }

                raise InvalidDataframeError(
                    "The downloaded data is not valid. "
                    "Please check JPMaQSDownload.downloaded_data "
                    "(dict) for more information."
                )

        return final_df

    def check_connection(
        self, verbose: bool = False, raise_error: bool = False
    ) -> bool:
        """Check if the interface is connected to the server.
        :return <bool>: True if connected, False if not.
        """

        res = self.dq_interface.check_connection(verbose=verbose)
        if raise_error and not res:
            raise ConnectionError(HeartbeatError("Heartbeat failed."))
        return res

    def validate_download_args(
        self,
        tickers: List[str],
        cids: List[str],
        xcats: List[str],
        metrics: List[str],
        start_date: str,
        end_date: str,
        get_catalogue: bool,
        expressions: List[str],
        show_progress: bool,
        as_dataframe: bool,
        report_time_taken: bool,
        report_egress: bool,
    ) -> bool:
        """Validate the arguments passed to the download function.

        :params:  -- see `macrosynergy.download.jpmaqs.JPMaQSDownload.download()`.

        :return <bool>: True if valid.

        :raises <TypeError>: If any of the arguments are not of the correct type.
        :raises <ValueError>: If any of the arguments are semantically incorrect.

        """

        if not isinstance(show_progress, bool):
            raise TypeError("`show_progress` must be a boolean.")

        if not isinstance(as_dataframe, bool):
            raise TypeError("`as_dataframe` must be a boolean.")

        if not isinstance(report_time_taken, bool):
            raise TypeError("`report_time_taken` must be a boolean.")

        if not isinstance(report_egress, bool):
            raise TypeError("`report_egress` must be a boolean.")

        if not isinstance(get_catalogue, bool):
            raise TypeError("`get_catalogue` must be a boolean.")

        if all([tickers is None, cids is None, xcats is None, expressions is None]):
            raise ValueError(
                "Must provide at least one of `tickers`, "
                "`expressions`, or `cids` & `xcats` together."
            )
        vars_types_zip: zip = zip(
            [tickers, cids, xcats, expressions],
            ["tickers", "cids", "xcats", "expressions"],
        )
        for varx, namex in vars_types_zip:
            if not isinstance(varx, list) and varx is not None:
                raise TypeError(f"`{namex}` must be a list of strings.")
            if varx is not None:
                if len(varx) > 0:
                    if not all([isinstance(ticker, str) for ticker in varx]):
                        raise TypeError(f"`{namex}` must be a list of strings.")
                else:
                    raise ValueError(f"`{namex}` must be a non-empty list of strings.")

        if metrics is None:
            raise ValueError("`metrics` must be a non-empty list of strings.")
        else:
            if all([metric not in self.valid_metrics for metric in metrics]):
                raise ValueError(f"`metrics` must be a subset of {self.valid_metrics}.")

        if cids is not None:
            if xcats is None:
                raise ValueError(
                    "If specifying `cids`, `xcats` must also be specified."
                )
        else:
            if xcats is not None:
                raise ValueError(
                    "If specifying `xcats`, `cids` must also be specified."
                )

        for varx, namex in zip([start_date, end_date], ["start_date", "end_date"]):
            if not isinstance(varx, str):
                raise TypeError(f"`{namex}` must be a string.")
            if not is_valid_iso_date(varx):
                raise ValueError(
                    f"`{namex}` must be a valid date in the format YYYY-MM-DD."
                )
            if pd.to_datetime(varx, errors="coerce") is pd.NaT:
                raise ValueError(
                    f"`{namex}` must be a valid date > "
                    f"{pd.Timestamp.min.strftime('%Y-%m-%d')} "
                )
            if pd.to_datetime(varx) < pd.to_datetime("1950-01-01"):
                warnings.warn(
                    message=(
                        f"`{namex}` is set before 1950-01-01."
                        "Data before 1950-01-01 may not be available,"
                        " and will cause errors/missing data."
                    ),
                    category=UserWarning,
                )

        return True

    def get_catalogue(self):
        self.catalogue: List[str] = self.dq_interface.get_catalogue()
        return self.catalogue

    def filter_expressions_from_catalogue(
        self, expressions: List[str], verbose: bool = True
    ) -> List[str]:
        """
        Method to filter a list of expressions against the JPMaQS catalogue.
        This avoids requesting data for expressions that are not in the catalogue,
        and provides the user wuth the complete list of expressions that are in the
        catalogue.

        :param <List[str]> tickers: list of tickers to filter.

        :return <List[str]>: list of tickers that are in the JPMaQS catalogue.
        """
        print("Downloading the JPMaQS catalogue from DataQuery...")
        catalogue_tickers: List[str] = self.get_catalogue()
        catalogue_expressions: List[str] = self.construct_expressions(
            tickers=catalogue_tickers, metrics=self.valid_metrics
        )
        r: List[str] = sorted(
            list(set(expressions).intersection(catalogue_expressions))
        )
        if verbose:
            filtered: int = len(expressions) - len(r)
            if filtered > 0:
                print(
                    f"Removed {filtered}/{len(expressions)} expressions that are not in the JPMaQS catalogue."
                )

        return r

    def download(
        self,
        tickers: Optional[List[str]] = None,
        cids: Optional[List[str]] = None,
        xcats: Optional[List[str]] = None,
        metrics: List[str] = ["value"],
        start_date: str = "2000-01-01",
        end_date: Optional[str] = None,
        expressions: Optional[List[str]] = None,
        get_catalogue: bool = False,
        show_progress: bool = False,
        debug: bool = False,
        suppress_warning: bool = False,
        as_dataframe: bool = True,
        report_time_taken: bool = False,
        report_egress: bool = False,
    ) -> Union[pd.DataFrame, List[Dict]]:
        """Driver function to download data from JPMaQS via the DataQuery API.
        Timeseries data can be requested using `tickers` with `metrics`, or
        passing formed DataQuery expressions.
        `cids` and `xcats` (along with `metrics`) are used to construct
        expressions, which are ultimately passed to the DataQuery Interface.

        :param <list[str]> tickers: list of tickers.
        :param <list[str]> cids: list of cids.
        :param <list[str]> xcats: list of xcats.
        :param <list[str]> metrics: list of metrics, one of "value" (default),
            "grading", "eop_lag", "mop_lag". "all" is also accepted.
        :param <str> start_date: start date of the data to download, in the
            ISO format - YYYY-MM-DD.
        :param <str> end_date: end date of the data to download in the ISO
            format - YYYY-MM-DD.
        :param <list[str]> expressions: list of DataQuery expressions.
        :param <bool> get_catalogue: If True, the JPMaQS catalogue is
            downloaded and used to filter the list of tickers. Default is
            False.
        :param <bool> show_progress: True if progress bar should be shown,
            False if not (default).
        :param <bool> suppress_warning: True if suppressing warnings. Default
            is True.
        :param <bool> debug: Override the debug behaviour of the JPMaQSDownload
            class. If True, debug mode is enabled.
        :param <bool> print_debug_data: True if debug data should be printed,
            False if not (default). If debug=True, this is set to True.
        :param <bool> as_dataframe: Return a dataframe if True (default),
            a list of dictionaries if False.
        :param <bool> report_time_taken: If True, the time taken to download
            and apply data transformations is reported.
        :param <bool> report_egress: If True, the number of bytes downloaded
            is reported along with the transmission speed in kilobits/second.

        :return <pd.DataFrame|list[Dict]>: dataframe of data if
            `as_dataframe` is True, list of dictionaries if False.

        :raises <ValueError>: if provided arguments are invalid or
            semantically incorrect (see
            macrosynergy.download.jpmaqs.JPMaQSDownload.validate_download_args()).

        """

        # override the default warning behaviour and debug behaviour
        self.suppress_warning = suppress_warning
        self.debug = debug

        vartolist = lambda x: [x] if isinstance(x, str) else x
        tickers = vartolist(tickers)
        cids = vartolist(cids)
        xcats = vartolist(xcats)
        expressions = vartolist(expressions)
        metrics = vartolist(metrics)

        if len(metrics) == 1:
            if metrics[0] == "all":
                metrics = self.valid_metrics

        if end_date is None:
            end_date = (datetime.datetime.today() + pd.offsets.BusinessDay(2)).strftime(
                "%Y-%m-%d"
            )
            # NOTE : due to timezone conflicts, we choose to request data for 2 days in the future.
            # NOTE : DataQuery specifies YYYYMMDD as the date format, but we use YYYY-MM-DD for consistency.
            #   This is date is cast to YYYYMMDD in macrosynergy.download.dataquery.py.

        # Validate arguments.
        if not self.validate_download_args(
            tickers=tickers,
            cids=cids,
            xcats=xcats,
            metrics=metrics,
            start_date=start_date,
            end_date=end_date,
            expressions=expressions,
            get_catalogue=get_catalogue,
            show_progress=show_progress,
            as_dataframe=as_dataframe,
            report_time_taken=report_time_taken,
            report_egress=report_egress,
        ):
            raise ValueError("Invalid arguments passed to download().")
        if pd.to_datetime(start_date) > pd.to_datetime(end_date):
            warnings.warn(
                message=(
                    f"`start_date` ({start_date}) is after `end_date` ({end_date}). "
                    "These dates will be swapped."
                ),
                category=UserWarning,
            )
            start_date, end_date = end_date, start_date

        # Construct expressions.
        if expressions is None:
            expressions: List[str] = []

        expressions += self.construct_expressions(
            tickers=tickers,
            cids=cids,
            xcats=xcats,
            metrics=metrics,
        )
        expressions = list(set(expressions))

        if get_catalogue:
            expressions = self.filter_expressions_from_catalogue(expressions)

        # Download data.
        data: List[Dict] = []
        egress_data: Dict = {}
        download_time_taken: float = timer()
        with self.dq_interface as dq:
            print(
                "Downloading data from JPMaQS.\nTimestamp UTC: ",
                datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            )
            self.dq_interface.check_connection(verbose=True)
            print(f"Number of expressions requested: {len(expressions)}")
            data = dq.download_data(
                expressions=expressions,
                start_date=start_date,
                end_date=end_date,
                show_progress=show_progress,
                **self.dq_download_kwargs,
            )

            if len(self.dq_interface.msg_errors) > 0:
                self.msg_errors += self.dq_interface.msg_errors

            if len(self.dq_interface.msg_warnings) > 0:
                self.msg_warnings += self.dq_interface.msg_warnings

            if len(self.dq_interface.unavailable_expressions) > 0:
                self.unavailable_expressions += (
                    self.dq_interface.unavailable_expressions
                )

            if report_egress:
                egress_data = self.dq_interface.egress_data

        download_time_taken: float = timer() - download_time_taken
        dfs_time_taken: float = timer()
        if as_dataframe:
            data_df: pd.DataFrame = self.time_series_to_df(
                dicts_list=data,
                validate_df=True,
                expected_expressions=expressions,
                start_date=start_date,
                end_date=end_date,
                verbose=not self.suppress_warning,
            )
            data = data_df

        dfs_time_taken: float = timer() - dfs_time_taken

        if report_time_taken:
            print(f"Time taken to download data: \t{download_time_taken:.2f} seconds.")
            if as_dataframe:
                print(
                    f"Time taken to convert to dataframe: \t{dfs_time_taken:.2f} seconds."
                )

        if report_egress:
            # create averages for egress_data like
            #  egress_data[tracking_id] = {
            #     "url": log_url,
            #     "upload_size": upload_size,
            #     "download_size": download_size,
            #     "time_taken": time_taken,
            #   }

            total_upload: int = 0
            total_download: int = 0
            total_time_taken: float = 0
            longest_time_taken: float = 0
            longest_time_taken_url: str = ""
            for tracking_id in egress_data:
                total_upload += egress_data[tracking_id]["upload_size"]
                total_download += egress_data[tracking_id]["download_size"]
                total_time_taken += egress_data[tracking_id]["time_taken"]
                if egress_data[tracking_id]["time_taken"] > longest_time_taken:
                    longest_time_taken = egress_data[tracking_id]["time_taken"]
                    longest_time_taken_url = egress_data[tracking_id]["url"]

            avg_upload_size_kb: float = total_upload / (1024)
            avg_download_size_kb: float = total_download / (1024)
            avg_time_taken: float = total_time_taken / len(egress_data)
            avg_transfer_rate_kbit: float = (
                (avg_download_size_kb + avg_upload_size_kb) * 8 / avg_time_taken
            )
            print(f"Average upload size: \t{avg_upload_size_kb:.2f} KB")
            print(f"Average download size: \t{avg_download_size_kb:.2f} KB")
            print(f"Average time taken: \t{avg_time_taken:.2f} seconds")
            print(f"Longest time taken: \t{longest_time_taken:.2f} seconds")
            print(f"Average transfer rate : \t{avg_transfer_rate_kbit:.2f} Kbps")

        if len(self.msg_errors) > 0:
            if not self.suppress_warning:
                print(
                    f"{len(self.msg_errors)} errors encountered during the download. \n"
                    f"The errors did not compromise the download. \n"
                    f"Please check `JPMaQSDownload.msg_errors` for more information."
                )

        return data


if __name__ == "__main__":
    cids = [
        "AUD",
        "BRL",
        "CAD",
        "CHF",
        "CLP",
        "CNY",
        "COP",
        "CZK",
        "DEM",
        "ESP",
        "EUR",
        "FRF",
        "GBP",
        "USD",
    ]
    xcats = [
        "RIR_NSA",
        "FXXR_NSA",
        "FXXR_VT10",
        "DU05YXR_NSA",
        "DU05YXR_VT10",
    ]
    metrics = "all"
    start_date: str = "2023-01-01"
    end_date: str = "2023-03-20"

    client_id = os.getenv("DQ_CLIENT_ID")
    client_secret = os.getenv("DQ_CLIENT_SECRET")

    with JPMaQSDownload(
        client_id=client_id,
        client_secret=client_secret,
        debug=True,
    ) as jpmaqs:
        data = jpmaqs.download(
            cids=cids,
            xcats=xcats,
            metrics=metrics,
            start_date=start_date,
            get_catalogue=True,
            end_date=end_date,
            show_progress=True,
            suppress_warning=False,
            report_time_taken=True,
            report_egress=True,
        )

        print(data.head())
