"""
JPMaQS Download Interface 
"""

import datetime
import io
import logging
import os
import traceback as tb
import warnings
from timeit import default_timer as timer
from typing import Dict, List, Optional, Tuple, Union, Any, Generator
import functools
import itertools

import pandas as pd

from macrosynergy.download.dataquery import DataQueryInterface
from macrosynergy.download.exceptions import HeartbeatError, InvalidDataframeError
from macrosynergy.management.utils import is_valid_iso_date, standardise_dataframe
from macrosynergy.management.types import QuantamentalDataFrame

logger = logging.getLogger(__name__)
debug_stream_handler = logging.StreamHandler(io.StringIO())
debug_stream_handler.setLevel(logging.NOTSET)
debug_stream_handler.setFormatter(
    logging.Formatter(
        "%(asctime)s - %(levelname)s - %(module)s - %(funcName)s :: %(message)s"
    )
)
logger.addHandler(debug_stream_handler)


def deconstruct_expression(
    expression: Union[str, List[str]]
) -> Union[List[str], List[List[str]]]:
    """
    Deconstruct an expression into a list of cid, xcat, and metric.
    Achieves the inverse of construct_expressions(). For non-JPMaQS expressions,
    the returned list will be [expression, expression, 'value']. The metric is set to
    'value' to ensure the reported metric is consistent with the standard JPMaQS metrics
    (JPMaQSDownload.valid_metrics).

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
        return list(map(deconstruct_expression, expression))
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
            # fail safely, return list where cid = xcat = expression,
            #  and metric = 'value'
            return [expression, expression, "value"]


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


def get_expression_from_qdf(df: Union[pd.DataFrame, List[pd.DataFrame]]) -> List[str]:

    if isinstance(df, list):
        return list(itertools.chain.from_iterable(map(get_expression_from_qdf, df)))

    metrics = list(set(df.columns) - set(QuantamentalDataFrame.IndexCols))
    exprs = []
    for metric in metrics:
        for cid, xcat in df[["cid", "xcat"]].drop_duplicates().values:
            if any(~df.loc[(df["cid"] == cid) & (df["xcat"] == xcat), metric].isna()):
                exprs.append(f"DB(JPMAQS,{cid}_{xcat},{metric})")
    return exprs


def get_expression_from_wide_df(
    df: Union[pd.DataFrame, List[pd.DataFrame]]
) -> List[str]:
    if isinstance(df, list):
        return list(itertools.chain.from_iterable(map(get_expression_from_wide_df, df)))
    return list(set(df.columns))


def timeseries_to_qdf(timeseries: Dict[str, Any]) -> QuantamentalDataFrame:
    """
    Converts a dictionary of time series to a QuantamentalDataFrame.

    :param <Dict[str, Any]> timeseries: A dictionary of time series.
    :return <QuantamentalDataFrame>: The converted DataFrame.
    """
    if not isinstance(timeseries, dict):
        raise TypeError("Argument `timeseries` must be a dictionary.")

    if timeseries["attributes"][0]["time-series"] is None:
        return None

    cid, xcat, metric = deconstruct_expression(
        timeseries["attributes"][0]["expression"]
    )

    df: pd.DataFrame = (
        pd.DataFrame(
            timeseries["attributes"][0]["time-series"],
            columns=["real_date", metric],
        )
        .assign(cid=cid, xcat=xcat)
        .dropna()
    )

    df["real_date"] = pd.to_datetime(df["real_date"], format="%Y%m%d")
    if df.empty:
        return None
    return df


def concat_single_metric_qdfs(
    df_list: List[QuantamentalDataFrame],
    errors: str = "ignore",
) -> QuantamentalDataFrame:
    """
    Combines a list of Quantamental DataFrames into a single DataFrame.

    :param <List[QuantamentalDataFrame]> df_list: A list of Quantamental DataFrames.
    :param <str> errors: The error handling method to use. If 'raise', then invalid
        items in the list will raise an error. If 'ignore', then invalid items will be
        ignored. Default is 'ignore'.
    :return <QuantamentalDataFrame>: The combined DataFrame.
    """
    if not isinstance(df_list, list):
        raise TypeError("Argument `df_list` must be a list.")

    if errors not in ["raise", "ignore"]:
        raise ValueError("`errors` must be one of 'raise' or 'ignore'.")

    if errors == "raise":
        if not all([isinstance(df, QuantamentalDataFrame) for df in df_list]):
            raise TypeError(
                "All elements in `df_list` must be Quantamental DataFrames."
            )
    else:
        df_list = [df for df in df_list if isinstance(df, QuantamentalDataFrame)]
        if len(df_list) == 0:
            return None

    def _get_metric(df: QuantamentalDataFrame) -> str:
        lx = list(set(df.columns) - set(QuantamentalDataFrame.IndexCols))
        if len(lx) != 1:
            raise ValueError(
                "Each QuantamentalDataFrame must have exactly one metric column."
            )
        return lx[0]

    def _group_by_metric(
        dfl: List[QuantamentalDataFrame], fm: List[str]
    ) -> List[List[QuantamentalDataFrame]]:
        r = [[] for _ in range(len(fm))]
        while dfl:
            metric = _get_metric(df=dfl[0])
            r[fm.index(metric)] += [dfl.pop(0)]
        return r

    found_metrics = list(set(map(_get_metric, df_list)))

    df_list = _group_by_metric(dfl=df_list, fm=found_metrics)

    # use pd.merge to join on QuantamentalDataFrame.IndexCols
    df: pd.DataFrame = functools.reduce(
        lambda left, right: pd.merge(
            left, right, on=["real_date", "cid", "xcat"], how="outer"
        ),
        map(
            lambda fm: pd.concat(df_list.pop(0), axis=0, ignore_index=False),
            found_metrics,
        ),
    )

    return standardise_dataframe(df)


def timeseries_to_column(
    timeseries: Dict[str, Any], errors: str = "ignore"
) -> pd.DataFrame:
    """
    Converts a dictionary of time series to a DataFrame with a single column.

    :param <Dict[str, Any]> timeseries: A dictionary of time series.
    :param <str> errors: The error handling method to use. If 'raise', then invalid
        items in the list will raise an error. If 'ignore', then invalid items will be
        ignored. Default is 'ignore'.
    :return <pd.DataFrame>: The converted DataFrame.
    """
    if not isinstance(timeseries, dict):
        raise TypeError("Argument `timeseries` must be a dictionary.")

    if errors not in ["raise", "ignore"]:
        raise ValueError("`errors` must be one of 'raise' or 'ignore'.")

    expression = timeseries["attributes"][0]["expression"]

    if timeseries["attributes"][0]["time-series"] is None:
        if errors == "raise":
            raise ValueError("Time series is empty.")
        return None

    df: pd.DataFrame = pd.DataFrame(
        timeseries["attributes"][0]["time-series"], columns=["real_date", expression]
    )
    df["real_date"] = pd.to_datetime(df["real_date"], format="%Y%m%d")
    df = df.dropna()
    if df.empty:
        if errors == "raise":
            raise ValueError("Time series is empty.")
        return None
    return df.set_index("real_date")


def concat_column_dfs(
    df_list: List[pd.DataFrame], errors: str = "ignore"
) -> pd.DataFrame:
    """
    Concatenates a list of DataFrames into a single DataFrame.

    :param <List[pd.DataFrame]> df_list: A list of DataFrames.
    :param <str> errors: The error handling method to use. If 'raise', then invalid
        items in the list will raise an error. If 'ignore', then invalid items will be
        ignored. Default is 'ignore'.
    :return <pd.DataFrame>: The concatenated DataFrame.
    """
    if not isinstance(df_list, list):
        raise TypeError("Argument `df_list` must be a list.")

    if errors not in ["raise", "ignore"]:
        raise ValueError("`errors` must be one of 'raise' or 'ignore'.")

    if not all([isinstance(df, pd.DataFrame) for df in df_list]):
        if errors == "raise":
            raise TypeError("All elements in `df_list` must be DataFrames.")
        df_list = [df for df in df_list if isinstance(df, pd.DataFrame)]

    def _pop_df_list() -> Generator[pd.DataFrame, None, None]:
        while df_list:
            yield df_list.pop(0)

    # all the dfs are indexed by real_date, so we can just concat them adn drop dates when all values are NaN
    # df: pd.DataFrame = pd.concat(df_list, axis=1).dropna(how="all", axis=0)
    return pd.concat(_pop_df_list(), axis=1).dropna(how="all", axis=0)


def validate_downloaded_df(
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
    :param <list[str]> expected_expressions: list of expressions that were expected
        to be downloaded.
    :param <list[str]> found_expressions: list of expressions that were actually
        downloaded.
    :param <str> start_date: start date of the downloaded data.
    :param <str> end_date: end date of the downloaded data.
    :param <bool> verbose: whether to print the validation results.

    :return <bool>: True if the downloaded data is valid, False otherwise.

    :raises <TypeError>: if `data_df` is not a dataframe.
    """

    if not isinstance(data_df, pd.DataFrame):
        raise InvalidDataframeError(
            "Empty or invalid dataframe, please check download parameters."
        )
    if data_df.empty:
        return False

    # check if all expressions are present in the df
    exprs_f, expr_expected = set(found_expressions), set(expected_expressions)
    expr_missing = expr_expected - exprs_f
    unexpected_exprs = exprs_f - expr_expected
    if unexpected_exprs:
        raise InvalidDataframeError(
            f"Unexpected expressions were found in the downloaded data: "
            f"{unexpected_exprs}"
        )

    if expr_missing:
        log_str = (
            f"Some expressions are missing from the downloaded data. "
            "Check logger output for complete list.\n"
            f"{len(expr_missing)} out of {len(expr_expected)} expressions are missing. "
            f"To download the catalogue of all available expressions and filter the "
            "unavailable expressions, set `get_catalogue=True` in the "
            "call to `JPMaQSDownload.download()`."
        )

        logger.info(log_str)
        logger.info(f"Missing expressions: {expr_missing}")
        if verbose:
            print(log_str)

    # check if all dates are present in the df
    # NOTE : Hardcoded max start date to 1990-01-01. This is because the JPMAQS
    # database does not have data before this date.
    if datetime.datetime.strptime(start_date, "%Y-%m-%d") < datetime.datetime.strptime(
        "1990-01-01", "%Y-%m-%d"
    ):
        start_date = "1950-01-01"

    dates_expected = pd.bdate_range(start=start_date, end=end_date)
    found_dates = (
        data_df["real_date"].unique()
        if isinstance(data_df, QuantamentalDataFrame)
        else data_df.index.unique()
    )
    dates_missing = list(set(dates_expected) - set(found_dates))
    log_str = (
        "The expressions in the downloaded data are not a subset of the expected expressions."
        " Missing expressions: {missing_exprs}"
    )
    err_statement = (
        "The expressions in the downloaded data are not a subset of the "
        "expected expressions."
    )
    check_exprs = set()
    if isinstance(data_df, QuantamentalDataFrame):
        found_metrics = list(
            set(data_df.columns) - set(QuantamentalDataFrame.IndexCols)
        )
        for col in QuantamentalDataFrame.IndexCols:
            if not len(data_df[col].unique()) > 0:
                raise InvalidDataframeError(f"Column {col} is empty.")

        check_exprs = construct_expressions(
            tickers=(data_df["cid"] + "_" + data_df["xcat"]).unique(),
            metrics=found_metrics,
        )

    else:
        check_exprs = data_df.columns.tolist()

    missing_exprs = set(check_exprs) - set(found_expressions)
    if len(missing_exprs) > 0:
        logger.critical(log_str.format(missing_exprs=missing_exprs))

    if len(dates_missing) > 0:
        log_str = (
            f"Some dates are missing from the downloaded data. \n"
            f"{len(dates_missing)} out of {len(dates_expected)} dates are missing."
        )
        logger.warning(log_str)
        if verbose:
            print(log_str)

    return True


class JPMaQSDownload(DataQueryInterface):
    """
    JPMaQSDownload Object. This object is used to download JPMaQS data via the DataQuery API.
    It can be extended to include the use of proxies, and even request generic DataQuery expressions.

    :param <bool> oauth: True if using oauth, False if using username/password with crt/key.
    :param <Optional[str]> client_id: oauth client_id, required if oauth=True.
    :param <Optional[str]> client_secret: oauth client_secret, required if oauth=True.
    :param <Optional[str]> crt: path to crt file.
    :param <Optional[str]> key: path to key file.
    :param <Optional[str]> username: username for certificate based authentication.
    :param <Optional[str]> password: paired with username for certificate
    :param <bool> debug: True if debug mode, False if not.
    :param <bool> suppress_warning: True if suppressing warnings, False if not.
    :param <bool> check_connection: True if the interface should check the connection to
        the server before sending requests, False if not. False by default.
    :param <Optional[dict]> proxy: proxy to use for requests, None if not using proxy (default).
    :param <bool> print_debug_data: True if debug data should be printed, False if not
        (default).
    :param <dict> dq_kwargs: additional arguments to pass to the DataQuery API object such
        `calender` and `frequency` for the DataQuery API. For more fine-grained usage,
        initialize the DataQueryInterface object explicitly.
    :param <dict> kwargs: any other keyword arguments.

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
        *args,
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

        self.valid_metrics: List[str] = ["value", "grading", "eop_lag", "mop_lag"]
        self.msg_errors: List[str] = []
        self.msg_warnings: List[str] = []
        self.unavailable_expressions: List[str] = []
        self.downloaded_data: Dict = {}

        super().__init__(
            oauth=oauth,
            client_id=client_id,
            client_secret=client_secret,
            crt=crt,
            key=key,
            username=username,
            password=password,
            proxy=proxy,
            check_connection=check_connection,
            suppress_warning=suppress_warning,
            debug=debug,
            *args,
            **kwargs,
        )

        if self._check_connection:
            self.check_connection()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback): ...

    def _get_unavailable_expressions(
        self,
        expected_exprs: List[str],
        dicts_list: Optional[List[dict]] = None,
        downloaded_df: Optional[Union[pd.DataFrame, QuantamentalDataFrame]] = None,
    ) -> List[str]:
        assert (dicts_list is not None) ^ (downloaded_df is not None)
        if dicts_list is not None:
            return super()._get_unavailable_expressions(
                expected_exprs=expected_exprs, dicts_list=dicts_list
            )

        if downloaded_df is not None:
            if len(downloaded_df) == 0:
                return expected_exprs
            if isinstance(downloaded_df, QuantamentalDataFrame):
                found_expressions = get_expression_from_qdf(downloaded_df)
            else:
                found_expressions = get_expression_from_wide_df(downloaded_df)
            return list(set(expected_exprs) - set(found_expressions))

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
        dataframe_format: str,
        report_time_taken: bool,
    ) -> bool:
        """Validate the arguments passed to the download function.

        :params:  -- see `macrosynergy.download.jpmaqs.JPMaQSDownload.download()`.

        :return <bool>: True if valid.

        :raises <TypeError>: If any of the arguments are not of the correct type.
        :raises <ValueError>: If any of the arguments are semantically incorrect.

        """

        for var, name in [
            (get_catalogue, "get_catalogue"),
            (show_progress, "show_progress"),
            (as_dataframe, "as_dataframe"),
            (report_time_taken, "report_time_taken"),
        ]:
            if not isinstance(var, bool):
                raise TypeError(f"`{name}` must be a boolean.")

        if not isinstance(dataframe_format, str):
            raise TypeError("`dataframe_format` must be a string.")
        elif dataframe_format.lower() not in ["qdf", "wide"]:
            raise ValueError("`dataframe_format` must be one of 'qdf' or 'wide'.")

        if all([tickers is None, cids is None, xcats is None, expressions is None]):
            raise ValueError(
                "Must provide at least one of `tickers`, "
                "`expressions`, or `cids` & `xcats` together."
            )

        for var, name in [
            (tickers, "tickers"),
            (cids, "cids"),
            (xcats, "xcats"),
            (expressions, "expressions"),
        ]:
            if not isinstance(var, list) and var is not None:
                raise TypeError(f"`{name}` must be a list of strings.")
            if var is not None:
                if len(var) > 0:
                    if not all([isinstance(ticker, str) for ticker in var]):
                        raise TypeError(f"`{name}` must be a list of strings.")
                else:
                    raise ValueError(f"`{name}` must be a non-empty list of strings.")

        if metrics is None:
            raise ValueError("`metrics` must be a non-empty list of strings.")
        else:
            if all([metric not in self.valid_metrics for metric in metrics]):
                raise ValueError(f"`metrics` must be a subset of {self.valid_metrics}.")

        if bool(cids) ^ bool(xcats):
            raise ValueError(
                "If specifying `xcats`, `cids` must also be specified (and vice versa)."
            )

        for var, name in [
            (start_date, "start_date"),
            (end_date, "end_date"),
        ]:
            if not is_valid_iso_date(var):  # type check covered by `is_valid_iso_date`
                raise ValueError(
                    f"`{name}` must be a valid date in the format YYYY-MM-DD."
                )
            if pd.to_datetime(var, errors="coerce") is pd.NaT:
                raise ValueError(
                    f"`{name}` must be a valid date > "
                    f"{pd.Timestamp.min.strftime('%Y-%m-%d')}.\n"
                    "Check pandas documentation:"
                    " https://pandas.pydata.org/docs/user_guide/timeseries.html#timestamp-limitations`"
                )
            if pd.to_datetime(var) < pd.to_datetime("1950-01-01"):
                warnings.warn(
                    message=(
                        f"`{name}` is set before 1950-01-01."
                        "Data before 1950-01-01 may not be available,"
                        " and will cause errors/missing data."
                    ),
                    category=UserWarning,
                )

        return True

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

        catalogue_tickers: List[str] = self.get_catalogue(verbose=verbose)
        catalogue_expressions: List[str] = construct_expressions(
            tickers=catalogue_tickers, metrics=self.valid_metrics
        )
        r: List[str] = sorted(
            list(set(expressions).intersection(set(catalogue_expressions)))
        )
        if verbose:
            filtered: int = len(expressions) - len(r)
            if filtered > 0:
                print(
                    f"Removed {filtered}/{len(expressions)} expressions "
                    "that are not in the JPMaQS catalogue."
                )

        return r

    def _chain_download_outputs(
        self, download_outputs: Union[List[Dict], List[pd.DataFrame]]
    ) -> Union[List[Dict], List[pd.DataFrame]]:
        if not isinstance(download_outputs, list):
            raise TypeError("`download_outputs` must be a list.")

        download_outputs = [x for x in download_outputs if len(x) > 0]
        if len(download_outputs) == 0:
            return []

        if isinstance(download_outputs[0], pd.DataFrame):
            return concat_column_dfs(df_list=download_outputs)

        if isinstance(download_outputs[0][0], (dict, QuantamentalDataFrame)):
            logger.debug(f"Chaining {len(download_outputs)} outputs.")
            _ch_types = list(
                itertools.chain.from_iterable(
                    [list(map(type, x)) for x in download_outputs]
                )
            )
            logger.debug(f"Object types in the downloaded data: {_ch_types}")

            download_outputs = list(itertools.chain.from_iterable(download_outputs))

        if isinstance(download_outputs[0], dict):
            return download_outputs
        if isinstance(download_outputs[0], QuantamentalDataFrame):
            return concat_single_metric_qdfs(download_outputs)
            # cannot chain QDFs with different metrics

        raise NotImplementedError(
            f"Cannot chain download outputs that are List of : {list(set(map(type, download_outputs)))}."
        )

    def _fetch_timeseries(
        self,
        url: str,
        params: dict,
        tracking_id: str,
        as_dataframe: bool = True,
        dataframe_format: str = "qdf",
        *args,
        **kwargs,
    ) -> Union[pd.DataFrame, List[Dict]]:
        ts_list = self._fetch(url=url, params=params, tracking_id=tracking_id)
        for its, ts in enumerate(ts_list):
            if ts["attributes"][0]["time-series"] is None:
                self.unavailable_expressions.append(ts["attributes"][0]["expression"])
                if "message" in ts["attributes"][0]:
                    self.msg_warnings.append(ts["attributes"][0]["message"])
                else:
                    self.msg_warnings.append(
                        f"Time series for expression {ts['attributes'][0]['expression']} is empty. "
                        " No explanation was provided."
                    )
                ts_list[its] = None

        ts_list = list(filter(None, ts_list))

        if as_dataframe:
            if dataframe_format == "qdf":
                ts_list = [timeseries_to_qdf(ts) for ts in ts_list if ts is not None]
            elif dataframe_format == "wide":
                ts_list = concat_column_dfs(
                    df_list=[timeseries_to_column(ts) for ts in ts_list]
                )
        logger.debug(f"Downloaded data for {len(ts_list)} expressions.")
        logger.debug(f"Unavailble expressions: {self.unavailable_expressions}")

        downloaded_types = list(set(map(type, ts_list)))
        logger.debug(f"Object types in the downloaded data: {downloaded_types}")

        return ts_list

    def download_data(self, *args, **kwargs):
        return super().download_data(*args, **kwargs)

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
        dataframe_format: str = "qdf",
        report_time_taken: bool = False,
        *args,
        **kwargs,
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
        :param <str> dataframe_format: Format of the dataframe to return, one of "qdf"
            or "wide". QDF is the Quantamental Dataframe format, and wide is the wide
            format with each expression as a column, and a single date column.
        :param <bool> report_time_taken: If True, the time taken to download
            and apply data transformations is reported.

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
            # NOTE : due to timezone conflicts, we choose to request data for 2 days in
            # the future.
            # NOTE : DataQuery specifies YYYYMMDD as the date format, but we use
            # YYYY-MM-DD for consistency.
            # This is date is cast to YYYYMMDD in macrosynergy.download.dataquery.py.

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
            dataframe_format=dataframe_format,
            report_time_taken=report_time_taken,
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

        dataframe_format = dataframe_format.lower()
        # Construct expressions.
        if expressions is None:
            expressions: List[str] = []

        expressions += construct_expressions(
            tickers=tickers,
            cids=cids,
            xcats=xcats,
            metrics=metrics,
        )
        expressions = list(set(expressions))

        if get_catalogue:
            expressions = self.filter_expressions_from_catalogue(expressions)

        # Download data.
        print(
            "Downloading data from JPMaQS.\nTimestamp UTC: ",
            datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        )
        data: List[Dict] = []
        download_time_taken: float = timer()
        data: List[Union[dict, QuantamentalDataFrame]] = self.download_data(
            expressions=expressions,
            start_date=start_date,
            end_date=end_date,
            show_progress=show_progress,
            as_dataframe=as_dataframe,
            dataframe_format=dataframe_format,
            *args,
            **kwargs,
        )

        download_time_taken: float = timer() - download_time_taken

        if report_time_taken:
            print(f"Time taken to download data: \t{download_time_taken:.2f} seconds.")

        if len(self.msg_errors) > 0:
            if not self.suppress_warning:
                warnings.warn(
                    f"{len(self.msg_errors)} errors encountered during the download. \n"
                    f"The errors did not compromise the download. \n"
                    f"Please check `JPMaQSDownload.msg_errors` for more information."
                )

        if as_dataframe:
            found_expressions: List[str] = list(
                set(expressions) - set(self.unavailable_expressions)
            )

            if not validate_downloaded_df(
                data_df=data,
                expected_expressions=expressions,
                found_expressions=found_expressions,
                start_date=start_date,
                end_date=end_date,
                verbose=True,
            ):
                raise InvalidDataframeError("Downloaded data is invalid.")

            if dataframe_format == "qdf":
                assert isinstance(data, QuantamentalDataFrame)

        return data


if __name__ == "__main__":
    cids = ["AUD", "BRL", "CAD", "CHF", "CNY", "CZK", "EUR", "GBP", "USD"]
    xcats = ["RIR_NSA", "FXXR_NSA", "FXXR_VT10", "DU05YXR_NSA", "DU05YXR_VT10"]
    start_date: str = "2024-02-07"
    end_date: str = "2024-02-09"

    with JPMaQSDownload(
        client_id=os.getenv("DQ_CLIENT_ID"),
        client_secret=os.getenv("DQ_CLIENT_SECRET"),
    ) as jpmaqs:
        data = jpmaqs.download(
            cids=cids,
            xcats=xcats,
            start_date=start_date,
            end_date=end_date,
            show_progress=True,
            report_time_taken=True,
        )
        print(data.info())
        print(data)
