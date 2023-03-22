""" JPMaQS Download Interface """

from typing import List, Optional, Dict, Union
import pandas as pd
import warnings
import yaml
import json
import traceback as tb
from macrosynergy.download.dataquery import DataQueryInterface, HeartbeatError
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

def oauth_credential_loader(path_to_credentials : str) -> dict:
    """Load oauth credentials from a yaml file.
    :param <str> path_to_credentials: path to yaml file containing credentials.
    :return <dict>: dictionary containing credentials.
    """
    # if ends with yml, yaml try:
    if path_to_credentials.endswith("yml") or path_to_credentials.endswith("yaml"):
        with open(path_to_credentials, "r") as f:
            credentials = yaml.safe_load(f)

    if path_to_credentials.endswith("json"):
        with open(path_to_credentials, "r") as f:
            credentials = json.load(f)

    # look for client_id and client_secret substrings
    for key in credentials.keys():
        if "client_id" in key:
            client_id = credentials[key]
        if "client_secret" in key:
            client_secret = credentials[key]
        
    credentials = {
        "client_id": client_id,
        "client_secret": client_secret
    }

    return credentials


class InvalidDataframeError(Exception):
    """Raised when a dataframe is not valid."""


class MissingDataError(Exception):
    """Raised when data is missing from a requested dataframe."""


class JPMaQSDownload(object):
    """JPMaQS Download Interface Object
    :param <bool> oauth: True if using oauth, False if using username/password with crt/key.

    When using oauth:
    :param <str> client_id: oauth client_id, required if oauth=True.
    :param <str> client_secret: oauth client_secret, required if oauth=True.

    When using username/password with crt/key:
    :param <str> crt: path to crt file.
    :param <str> key: path to key file.
    :param <str> username: username for certificate based authentication.
    :param <str> password : paired with username for certificate.

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
        base_url, timeout, etc. For more fine-grained usage, initialize the 
        DataQueryInterface object explicitly.
        See macrosynergy.download.dataquery.DataQueryInterface for more.

    :return <JPMaQSDownload>: JPMaQSDownload object

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
        oauth_config: Optional[str] = None,
        proxy: Optional[Dict] = None,
        suppress_warning: bool = True,
        debug: bool = False,
        print_debug_data: bool = False,
        dq_download_kwargs: dict = {},
        **kwargs,
    ):
        for varx, namex in zip(
            [oauth, check_connection, suppress_warning, debug, print_debug_data],
            [
                "oauth",
                "check_connection",
                "suppress_warning",
                "debug",
                "print_debug_data",
            ],
        ):
            if not isinstance(varx, bool):
                raise ValueError(f"`{namex}` must be a boolean.")

        if not isinstance(proxy, dict) and proxy is not None:
            raise ValueError("`proxy` must be a dictionary or None.")

        if not isinstance(dq_download_kwargs, dict):
            raise ValueError("`dq_download_kwargs` must be a dictionary.")
        


        self.suppress_warning = suppress_warning
        self.debug = debug
        self.print_debug_data = print_debug_data
        self._check_connection = check_connection
        self.dq_download_kwargs = dq_download_kwargs
        if oauth and (
            (not isinstance(client_id, str)) or (not isinstance(client_secret, str))
        ):
            if oauth_config is None:
                raise ValueError("If using oauth, `client_id` and `client_secret` must be provided."
                                 " Alternatively, provide a path to a yaml file containing the credentials "
                                 "using the `oauth_config` argument.")
            else:
                credentials = oauth_credential_loader(oauth_config)
                client_id = credentials["client_id"]
                client_secret = credentials["client_secret"]

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
            for varx, namex in zip(
                [crt, key, username, password],
                ["crt", "key", "username", "password"],
            ):
                if not isinstance(varx, str):
                    raise ValueError(f"`{namex}` must be a string.")

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
        self.valid_metrics = ["value", "grading", "eop_lag", "mop_lag"]

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
    def deconstruct_expressions(
        expressions: List[str], as_tickers=False
    ) -> List[List[str]]:
        """Deconstruct expressions into tickers and metrics, or cids, xcats, and metrics."""

        expressions = [
            exp.replace("DB(JPMAQS,", "").replace(")", "") for exp in expressions
        ]

        if as_tickers:
            return [exp.split(",") for exp in expressions]
        else:
            r: List[List[str]] = []
            for exp in expressions:
                tick, metric = exp.split(",")
                cid, xcat = tick.split("_", 1)
                r += [[cid, xcat, metric]]
            return r

    def validate_downloaded_df(
        self,
        data_df: pd.DataFrame,
        expected_expressions: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        verbose: bool = True,
    ) -> bool:
        """Validate the downloaded data.
        :param data_df <pd.DataFrame>: DataFrame containing the downloaded data.
        :param expressions <list>: List of expressions used to download the data.
        :param start_date <str>: Start date of the data.
        :param end_date <str>: End date of the data.
        :param verbose <bool>: Whether to print the validation results.

        :return <bool>: True if valid, False if not.
        """
        # TODO : Complete this function to report number of missing values
        # if verbose print number of reqd. expressions and num of actual expressions in df
        if not isinstance(data_df, pd.DataFrame):
            return False
        if data_df.empty:
            return False

        # check if all expressions are present in the df
        exprs_f = set(data_df["expression"])
        expr_expected = set(expected_expressions)
        expr_missing = expr_expected - exprs_f

        if expr_missing:
            log_str = (
                f"Some expressions are missing from the downloaded data. \n"
                f"{len(expr_missing)} out of {len(expr_expected)} expressions are missing."
            )
            logger.warning(log_str)
            if verbose:
                print(log_str)

        # check if all dates are present in the df
        dates_f = data_df["real_date"]
        dates_expected = pd.bdate_range(start=start_date, end=end_date)

        # check that all expected dates are in the df
        dates_missing = set(dates_expected) - set(dates_f)

        if dates_missing:
            log_str = (
                f"Some dates are missing from the downloaded data. \n"
                f"{len(dates_missing)} out of {len(dates_expected)} dates are missing."
            )
            logger.warning(log_str)
            if verbose:
                print(log_str)

        bday_data_df = data_df[data_df["real_date"].isin(dates_expected)]

        # find any "NA" values in the df
        na_values = bday_data_df.isna().sum().sum()
        if na_values > 0:
            log_str = (
                "Missing values found in the downloaded data. \n"
                f"{na_values} missing values (NAs). Check output dataframe for details."
            )
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
        Parameters
        :param dicts_list <list>: List of dictionaries containing time series
            data from the DataQuery API
        Returns
        :return <pd.DataFrame>: DataFrame containing the data
        """
        dfs: List = []
        for d in dicts_list:
            df = pd.DataFrame(
                d["attributes"][0]["time-series"], columns=["real_date", "observation"]
            )
            df["expression"] = d["attributes"][0]["expression"]
            dfs += [df]

        temp_df = pd.concat(dfs, axis=0).reset_index(drop=True)[
            ["real_date", "expression", "observation"]
        ]
        temp_df["real_date"] = pd.to_datetime(temp_df["real_date"])

        # split expression into cid, xcat, and metric using deconstruct_expressions()
        temp_df[["cid", "xcat", "metric"]] = pd.DataFrame(
            self.deconstruct_expressions(temp_df["expression"].tolist())
        )
        # drop expression column
        if not self.validate_downloaded_df(
            expected_expressions=expected_expressions,
            data_df=temp_df,
            start_date=start_date,
            end_date=end_date,
            verbose=verbose,
        ):
            if validate_df:
                raise InvalidDataframeError(
                    "The downloaded data is not valid. Please check the logs for details."
                )

        # form df into long format, with cid, xcat, and <metric> as columns
        temp_df.drop(columns=["expression"], inplace=True)
        # use metric as column name, and observation as value
        return_df = (
            temp_df.pivot_table(
                index=["real_date", "cid", "xcat"],
                columns="metric",
                values="observation",
            )
            .reset_index()
            .rename_axis(None, axis=1)
        )

        # only business days should be in the df
        bdates = pd.bdate_range(
            start=return_df["real_date"].min(), end=return_df["real_date"].max()
        )
        return_df = return_df[return_df["real_date"].isin(bdates)]

        # sort
        return_df = return_df.sort_values(by=["cid", "xcat", "real_date"]).reset_index(
            drop=True
        )

        # format as cid, xcat, real_date, <metric>
        _metrics = list(
            set(return_df.columns.tolist()) - set(["cid", "xcat", "real_date"])
        )
        return_df = return_df[["cid", "xcat", "real_date"] + _metrics]

        return return_df

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
        expressions: List[str],
        show_progress: bool,
        as_dataframe: bool,
    ) -> bool:
        """Validate the arguments passed to the download function.

        :params -- see macrosynergy.download.jpmaqs.JPMaQSDownload.download()

        :return <bool>: True if valid.

        :raises <TypeError>: If any of the arguments are not of the correct type.
        :raises <ValueError>: If any of the arguments are semantically incorrect.

        """

        def is_valid_date(date: str) -> bool:
            try:
                datetime.datetime.strptime(date, "%Y-%m-%d")
                return True
            except ValueError:
                return False

        if not isinstance(show_progress, bool):
            raise TypeError("`show_progress` must be a boolean.")

        if not isinstance(as_dataframe, bool):
            raise TypeError("`as_dataframe` must be a boolean.")

        if all([tickers is None, cids is None, xcats is None, expressions is None]):
            raise ValueError(
                "Must provide at least one of `tickers`, "
                "`expressions`, or `cids` & `xcats` together."
            )

        for varx, namex in zip(
            [tickers, cids, xcats, expressions, metrics],
            ["tickers", "cids", "xcats", "expressions", "metrics"],
        ):
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
            if not is_valid_date(varx):
                raise ValueError(
                    f"`{namex}` must be a valid date in the format YYYY-MM-DD."
                )

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
        suppress_warning=True,
        as_dataframe=True,
    ) -> Union[pd.DataFrame , List[Dict]]:
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

        if suppress_warning != self.suppress_warning:
            self.suppress_warning = suppress_warning
            # self.set_logging_level()

        if all([_arg is None for _arg in [tickers, cids, xcats, expressions]]):
            cids = ["USD", "AUD"]
            xcats = ["EQXR_VT10", "EXALLOPENNESS_NSA_1YMA"]
            metrics = ["value", "grading"]
        # NOTE : This is simply so that we can test the download() function
        #   without having to pass in a bunch of arguments.

        for varx in [tickers, cids, xcats, expressions]:
            if isinstance(varx, str):
                varx = [varx]

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
            show_progress=show_progress,
            as_dataframe=as_dataframe,
        ):
            raise ValueError("Invalid arguments passed to download().")

        # Construct expressions.
        if expressions is None:
            expressions: List[str] = []

        expressions += self.construct_expressions(
            tickers=tickers,
            cids=cids,
            xcats=xcats,
            metrics=metrics,
        )

        # Download data.
        data: List[Dict] = []
        with self.dq_interface as dq:
            print(
                "Downloading data from JPMaQS.\nTimestamp UTC: ",
                datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            )
            data = dq.download_data(
                expressions=expressions,
                start_date=start_date,
                end_date=end_date,
                show_progress=show_progress,
                **self.dq_download_kwargs,
            )

        if as_dataframe:
            data_df: pd.DataFrame = self.time_series_to_df(
                dicts_list=data,
                validate_df=True,
                expected_expressions=expressions,
                start_date=start_date,
                end_date=end_date,
                verbose=True,
            )
            return data_df
        else:
            return data


if __name__ == "__main__":
    import os

    client_id = os.environ["JPMAQS_API_CLIENT_ID"]
    client_secret = os.environ["JPMAQS_API_CLIENT_SECRET"]

    cids = ["USD", "AUD"]
    xcats = ["EQXR_VT10", "EXALLOPENNESS_NSA_1YMA"]
    metrics = ["value", "grading"]
    start_date: str = "2020-01-25"
    end_date: str = "2023-02-05"

    with JPMaQSDownload(
        client_id=client_id,
        client_secret=client_secret,
        debug=True,
    ) as jpmaqs:
        jpmaqs.check_connection(verbose=True)

        data = jpmaqs.download(
            cids=cids,
            xcats=xcats,
            metrics=metrics,
            start_date=start_date,
            end_date=end_date,
            show_progress=True,
        )

        print(data.head())
