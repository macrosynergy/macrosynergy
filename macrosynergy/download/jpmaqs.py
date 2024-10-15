"""
JPMaQS Download Interface 
"""

import datetime
from dateutil.relativedelta import relativedelta
import io
import logging
import os
import glob
import shutil
import json
import warnings
from timeit import default_timer as timer
from typing import Dict, List, Optional, Tuple, Union, Any, Generator
import itertools
import joblib
from tqdm import tqdm

import pandas as pd

from macrosynergy.download.dataquery import (
    JPMAQS_GROUP_ID,
    DataQueryInterface,
    API_DELAY_PARAM,
)
from macrosynergy.download.exceptions import InvalidDataframeError
from macrosynergy.management.utils import (
    is_valid_iso_date,
    concat_single_metric_qdfs,
    ticker_df_to_qdf,
)
from macrosynergy.management.constants import JPMAQS_METRICS
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

DEFAULT_CLIENT_ID_ENV_VAR: str = "DQ_CLIENT_ID"
DEFAULT_CLIENT_SECRET_ENV_VAR: str = "DQ_CLIENT_SECRET"


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

    :raises <TypeError>: if `expression` is not a string or a list of strings.
    :raises <ValueError>: if `expression` is an empty list.
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

    if _get_ts(timeseries) is None:
        return None

    cid, xcat, metric = deconstruct_expression(_get_expr(timeseries))

    df: pd.DataFrame = (
        pd.DataFrame(
            _get_ts(timeseries),
            columns=["real_date", metric],
        )
        .assign(cid=cid, xcat=xcat)
        .dropna()
    )

    df["real_date"] = pd.to_datetime(df["real_date"], format="%Y%m%d")
    if df.empty:
        return None
    return df


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

    expression = _get_expr(timeseries)

    if _get_ts(timeseries) is None:
        if errors == "raise":
            raise ValueError("Time series is empty.")
        return None

    df: pd.DataFrame = pd.DataFrame(
        _get_ts(timeseries), columns=["real_date", expression]
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


def _get_expr(ts: dict) -> str:
    return ts["attributes"][0]["expression"]


def _get_ts(ts: dict) -> dict:
    return ts["attributes"][0]["time-series"]


def _get_ticker(ts: dict) -> str:
    return _get_expr(ts).split(",")[1]


def _get_xcat(ticker: str) -> str:
    return ticker.split("_", 1)[1]


def _ticker_filename(ticker: str, save_path: str) -> str:
    return os.path.join(save_path, _get_xcat(ticker), f"{ticker}.csv")


def _save_qdf(data: List[dict], save_path: str) -> None:

    for ticker in sorted(set(map(_get_ticker, data))):
        ticker_filename = _ticker_filename(ticker, save_path)
        os.makedirs(os.path.dirname(ticker_filename), exist_ok=True)
        ts = [_ts for _ts in data if _get_ticker(_ts) == ticker]
        df: QuantamentalDataFrame = concat_single_metric_qdfs(
            [timeseries_to_qdf(_ts) for _ts in ts]
        ).drop(columns=["cid", "xcat"])
        if os.path.exists(_ticker_filename(ticker, save_path)):
            edf = pd.read_csv(
                ticker_filename, parse_dates=["real_date"], index_col="real_date"
            )
            edf = edf.drop(columns=[col for col in edf.columns if col in df.columns])
            df = pd.concat([edf, df.set_index("real_date")], axis=1).reset_index()
            os.remove(ticker_filename)
        df.to_csv(ticker_filename, index=False)

    return


def _save_timeseries_as_column(data: List[dict], save_path: str) -> None:
    for ts in data:
        if _get_ts(ts) is None:
            continue
        expr = _get_expr(ts)
        df = timeseries_to_column(ts)
        if df.empty:
            continue
        df.reset_index().to_csv(os.path.join(save_path, f"{expr}.csv"), index=False)
    return


def _save_timeseries(data: List[dict], save_path: str):
    data = [d for d in data if d is not None]
    if len(data) == 0:
        return

    for ts in data:
        with open(os.path.join(save_path, f"{_get_expr(ts)}.json"), "w") as f:
            json.dump(ts, f)

    return


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

    dates_missing = list(set(dates_expected) - set(pd.to_datetime(found_dates)))
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


def _get_expressions_from_qdf_csv(file_path: str) -> List[str]:
    ticker = os.path.basename(file_path).split(".")[0]
    with open(file_path, "r", encoding="utf-8") as f:
        headers = f.readline().strip().split(",")

        assert len(set(headers)) == len(headers), f"Duplicate headers in {file_path}"
        metrics = set(headers) - set(["real_date"])
        return [f"DB(JPMAQS,{ticker},{metric})" for metric in metrics]


def _get_expressions_from_wide_csv(file_path: str) -> List[str]:
    with open(file_path, "r", encoding="utf-8") as f:
        headers = f.readline().strip().split(",")
        assert len(set(headers)) == len(headers), f"Duplicate headers in {file_path}"
        expression = list(set(headers) - set(["real_date"]))
        return expression


def _get_expressions_from_json(file_path: str) -> List[str]:
    with open(file_path, "r", encoding="utf-8") as f:
        return [_get_expr(json.load(f))]


def get_expressions_from_file(
    file_path: str, as_dataframe: bool = True, dataframe_format: str = "qdf"
) -> List[str]:
    """
    Loads the expressions found in a downloaded timeseries file (either JSON or CSV).

    :param <str> file_path: path to the file.
    :param <bool> as_dataframe: whether to load the file as a dataframe.
    :param <str> dataframe_format: the format of the dataframe. Must be one of 'qdf' or 'wide'.
    :return <List[str]>: list of expressions found in the file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist.")

    if not as_dataframe:
        return _get_expressions_from_json(file_path)

    if not dataframe_format in ["qdf", "wide"]:
        raise ValueError("`dataframe_format` must be one of 'qdf' or 'wide'.")

    if dataframe_format == "qdf":
        return _get_expressions_from_qdf_csv(file_path)
    elif dataframe_format == "wide":
        return _get_expressions_from_wide_csv(file_path)


def validate_downloaded_data(
    path: str,
    expected_expressions: List[str],
    as_dataframe: bool = True,
    dataframe_format: str = "qdf",
    show_progress: bool = True,
) -> List[str]:
    """
    Validate the downloaded data in the provided path.

    :param <str> path: path to the downloaded data.
    :param <list[str]> expected_expressions: list of expressions that were expected
        to be downloaded.
    :param <bool> as_dataframe: whether to load the files as dataframes.
    :param <str> dataframe_format: the format of the dataframe. Must be one of 'qdf' or 'wide'.
    :param <bool> show_progress: whether to show a progress bar.
    :return <list[str]>: list of expressions that are missing from the downloaded data.
    """
    if not os.path.isdir(path):
        raise ValueError(f"Path {path} does not exist.")

    ext = "csv" if as_dataframe else "json"
    files = glob.glob(f"{path}/**/*.{ext}", recursive=True)

    def get_expression_func(
        file_path: str, as_dataframe=as_dataframe, dataframe_format=dataframe_format
    ) -> List[str]:
        return get_expressions_from_file(
            file_path, as_dataframe=as_dataframe, dataframe_format=dataframe_format
        )

    all_exprs = []
    all_exprs = joblib.Parallel(n_jobs=-1)(
        joblib.delayed(get_expression_func)(file_path)
        for file_path in tqdm(
            files,
            desc="Validating downloaded data",
            disable=not show_progress,
        )
    )
    # join all the lists of expressions
    all_exprs = list(itertools.chain.from_iterable(all_exprs))

    missing_exprs = sorted(set(expected_expressions) - set(all_exprs))

    if len(missing_exprs) > 0:
        logger.critical(
            f"Some expressions are missing from the downloaded data. "
            f"Missing expressions: {missing_exprs}"
        )

    return missing_exprs


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

        if not all([crt, key, username, password]):
            if not all([client_id, client_secret]):
                # check the environment variables
                _clid = os.getenv(DEFAULT_CLIENT_ID_ENV_VAR)
                _clsc = os.getenv(DEFAULT_CLIENT_SECRET_ENV_VAR)
                if all([_clid, _clsc]):
                    client_id = _clid
                    client_secret = _clsc

        if not (all([client_id, client_secret]) or all([crt, key, username, password])):
            raise ValueError(
                "Must provide either `client_id` and `client_secret` for oauth, or "
                "`crt`, `key`, `username`, and `password` for certificate based authentication."
            )

        self.valid_metrics: List[str] = JPMAQS_METRICS
        self.msg_errors: List[str] = []
        self.msg_warnings: List[str] = []
        self.unavailable_expressions: List[str] = []
        self.downloaded_data: Dict = {}
        self.jpmaqs_access: bool = True

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

        if isinstance(download_outputs[0][0], (dict, bool, QuantamentalDataFrame)):
            logger.debug(f"Chaining {len(download_outputs)} outputs.")
            _ch_types = list(
                itertools.chain.from_iterable(
                    [list(map(type, x)) for x in download_outputs]
                )
            )
            logger.debug(f"Object types in the downloaded data: {_ch_types}")

            download_outputs = list(itertools.chain.from_iterable(download_outputs))

        if isinstance(download_outputs[0], (dict, bool)):
            return download_outputs
        if isinstance(download_outputs[0], QuantamentalDataFrame):
            return concat_single_metric_qdfs(download_outputs)
            # cannot chain QDFs with different metrics
        if not self.jpmaqs_access:
            raise ValueError(
                f"The credentials you have provided are for Dataquery access only and "
                f"hence have no JPMaQS entitlements. Because of this you are only able "
                f"to download data from 2000-01-01 to "
                f"{(datetime.datetime.now() - relativedelta(months=6)).strftime('%Y-%m-%d')}."
            )

        raise NotImplementedError(
            f"Cannot chain download outputs that are List of : {list(set(map(type, download_outputs)))}."
        )

    def _save_data(
        self,
        data: List[dict],
        as_dataframe: bool,
        dataframe_format: str,
        save_path: str,
    ) -> bool:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if as_dataframe:
            if dataframe_format == "qdf":
                _save_qdf(data, save_path)
            elif dataframe_format == "wide":
                _save_timeseries_as_column(data, save_path)
        else:
            _save_timeseries(data, save_path)
        return True

    def _fetch_timeseries(
        self,
        url: str,
        params: dict,
        tracking_id: str,
        as_dataframe: bool = True,
        dataframe_format: str = "qdf",
        save_path: Optional[str] = None,
        *args,
        **kwargs,
    ) -> Union[pd.DataFrame, List[Dict]]:
        ts_list: List[dict] = self._fetch(
            url=url, params=params, tracking_id=tracking_id
        )
        for its, ts in enumerate(ts_list):
            if _get_ts(ts) is None:
                self.unavailable_expressions.append(_get_expr(ts))
                if "message" in ts["attributes"][0]:
                    self.msg_warnings.append(ts["attributes"][0]["message"])
                else:
                    self.msg_warnings.append(
                        f"Time series for expression {ts['attributes'][0]['expression']} is empty. "
                        " No explanation was provided."
                    )
                ts_list[its] = None

            if "message" in ts["attributes"][0]:
                if "[FT] Limited Dataset Access Only" in ts["attributes"][0]["message"]:
                    self.jpmaqs_access = False

        ts_list: List[dict] = list(filter(None, ts_list))
        logger.debug(f"Downloaded data for {len(ts_list)} expressions.")
        logger.debug(f"Unavailble expressions: {self.unavailable_expressions}")
        if save_path is not None:
            try:
                ts_list = [
                    self._save_data(
                        data=ts_list,
                        as_dataframe=as_dataframe,
                        dataframe_format=dataframe_format,
                        save_path=save_path,
                        *args,
                        **kwargs,
                    )
                ]
            except Exception as exc:
                logger.error(f"Failed to save data to disk: {exc}")
                self.msg_errors.append(f"Failed to save data to disk: {exc}")
                raise exc
        elif as_dataframe:
            if dataframe_format == "qdf":
                ts_list = [timeseries_to_qdf(ts) for ts in ts_list if ts is not None]
            elif dataframe_format == "wide":
                ts_list = concat_column_dfs(
                    df_list=[timeseries_to_column(ts) for ts in ts_list]
                )

        downloaded_types = list(set(map(type, ts_list)))
        logger.debug(
            f"Object types in the downloaded data: {downloaded_types}"
            + ("(saving to disk)" if save_path is not None else "")
        )

        return ts_list

    def get_catalogue(
        self,
        group_id: str = JPMAQS_GROUP_ID,
        page_size: int = 1000,
        verbose: bool = True,
    ) -> List[str]:
        return super().get_catalogue(group_id, page_size, verbose)

    def download_all_to_disk(
        self,
        path: str,
        expressions: Optional[List[str]] = None,
        as_dataframe: bool = True,
        dataframe_format: str = "qdf",
        show_progress: bool = True,
        delay_param: float = API_DELAY_PARAM,
        batch_size: Optional[int] = None,
        retry: int = 3,
        overwrite: bool = True,
        *args,
        **kwargs,
    ) -> None:
        """
        Downloads all JPMaQS data to disk.

        :param <str> path: path to the directory where the data will be saved.
        :param <Optional[List[str]> expressions: Default is None, meaning all expressions
            in the JPMaQS catalogue will be downloaded. If provided, only the expressions
            in the list will be downloaded.
        :param <bool> as_dataframe: Default is True, meaning the data will be saved as a
            DataFrame (either in the Quantamental Data Format ('qdf') or wide format ('wide')).
            If False, the data will be saved as JSON files, with one expression per file.
        :param <str> dataframe_format: Default is 'qdf'. If `as_dataframe` is True, this
            parameter specifies the format of the DataFrame. Must be one of 'qdf' or 'wide'.
        :param <bool> show_progress: Default is True, meaning the progress of the download
            will be displayed. If False, the progress will not be displayed.
        :param <float> delay_param: Default is 0.2 seconds (fastest allowed by DataQuery API).
            The delay parameter to use when making requests to the DataQuery API. Ideally, this
            should not be changed.
        :param <int> batch_size: Default is None, meaning the batch size will be set to the
            default size (20). If provided, this parameter specifies the number of expressions
            to download in each batch.
        :param <int> retry: Default is 3, meaning the download will be retried 3 times for
            any expressions that fail to download. If set to 0, no retries will be attempted.
        :param <bool> overwrite: Default is True, meaning the data will be overwritten if it
            already exists. If False, the data will not be overwritten.
        :param <dict> kwargs: any other keyword arguments.
        """
        save_path: Optional[str] = None
        if path == "":
            print(
                "Path explicitly provided as an empty string. "
                "Assuming alternate saving method implemented."
            )
            save_path = ""
        else:
            path = os.path.expandvars(os.path.expanduser(path))
            save_path = os.path.join(path, "JPMaQSData")
            os.makedirs(save_path, exist_ok=True)
            if overwrite:
                msg = f"Overwriting data in {save_path}."
                warnings.warn(msg)  # the user should be warned
                logger.info(msg)  # but log doesn't need to be warning, info is fine
                shutil.rmtree(save_path)
                os.makedirs(save_path, exist_ok=True)

            print(f"Downloading all JPMaQS data to disk. Saving to: `{save_path}`.")

        start_date = "1990-01-01"
        end_date = (datetime.datetime.today() + pd.offsets.BusinessDay(2)).strftime(
            "%Y-%m-%d"
        )
        self.batch_size = batch_size or self.batch_size

        self.check_connection(verbose=True)

        if not expressions:
            catalogue: List[str] = self.get_catalogue()
            expressions = sorted(
                construct_expressions(tickers=catalogue, metrics=self.valid_metrics)
            )

        # if all the expressions do not contain DB(JPMAQS, then we need set dataframe format to wide
        if (
            as_dataframe
            and dataframe_format == "qdf"
            and not all([expr.startswith("DB(JPMAQS") for expr in expressions])
        ):
            dataframe_format = "wide"
            warnings.warn(
                "The list of expressions contains non-JPMAQS expressions. "
                "Setting dataframe format to 'wide'."
            )
        print(
            "Downloading data from JPMaQS.\nTimestamp UTC: ",
            datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        )
        download_time_taken: float = timer()
        data: List[Union[dict, QuantamentalDataFrame]] = self.download_data(
            save_path=save_path,
            expressions=expressions,
            start_date=start_date,
            end_date=end_date,
            show_progress=show_progress,
            as_dataframe=as_dataframe,
            dataframe_format=dataframe_format,
            delay_param=delay_param,
            *args,
            **kwargs,
        )
        download_time_taken: float = timer() - download_time_taken
        assert all([d is True for d in data])
        if path == "":
            return
        d_exprs = [
            os.path.basename(csv).split(".")[0]
            for csv in glob.glob(f"{save_path}/**/*.csv", recursive=True)
        ]
        if len(d_exprs) == 0:
            raise ValueError("No data was downloaded.")

        if as_dataframe and dataframe_format == "qdf":
            d_exprs = construct_expressions(tickers=d_exprs, metrics=self.valid_metrics)

        logger.info(f"Downloaded {len(d_exprs)} expressions.")

        unavailable_expressions = validate_downloaded_data(
            path=save_path,
            expected_expressions=expressions,
            as_dataframe=as_dataframe,
            dataframe_format=dataframe_format,
            show_progress=show_progress,
        )

        if len(unavailable_expressions) > 0:
            if retry > 0:
                logger.info(
                    f"Retrying {len(unavailable_expressions)} unavailable expressions."
                )
                self.download_all_to_disk(
                    path=path,
                    expressions=unavailable_expressions,
                    as_dataframe=as_dataframe,
                    dataframe_format=dataframe_format,
                    show_progress=show_progress,
                    delay_param=delay_param,
                    batch_size=batch_size,
                    retry=retry - 1,
                    overwrite=False,
                    *args,
                    **kwargs,
                )
            else:
                print(f"Failed to download {len(unavailable_expressions)} expressions.")
                for expr in unavailable_expressions:
                    print(f"\t{expr}")

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

        if not self.jpmaqs_access:
            print(
                "Credentials provided only have access to Dataquery and have not been "
                "granted access to JPMaQS. You can only access data after 2000-01-01 and "
                "before 6 months from the current date."
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


def custom_download(
    tickers, download_func, metrics=["value"], start_date=None, end_date=None
):
    """
    Custom download function to download data for a list of tickers using a custom download function.

    :param <list[str]> tickers: list of tickers to download data for.
    :param <callable> download_func: custom download function.
    :param <list[str]> metrics: list of metrics to download.
    :param <str> start_date: start date of the data to download.
    :param <str> end_date: end date of the data to download.

    :return <pd.DataFrame>: dataframe of downloaded data.
    """
    dfs = []
    for metric in metrics:
        expressions = []
        for ticker in list(set(tickers)):
            dq_expr = f"DB(JPMAQS,{ticker},{metric})"
            expressions.append(dq_expr)
        df = pd.DataFrame()

        step_size = 100
        df_store = []
        for idx in range(0, len(expressions) + 1, step_size):
            df_chunk = download_func(
                expressions[idx : idx + step_size],
                startDate=start_date,
                endDate=end_date,
            )
            df_chunk = df_chunk.dropna(axis=1, how="all")
            df_store.append(df_chunk)
        df = pd.concat(df_store, axis=1)

        df.columns = df.columns.str.split(",").str[1]
        df.index.name = "real_date"

        df = ticker_df_to_qdf(df, metric=metric)

        dfs.append(df)

    df = concat_single_metric_qdfs(dfs)
    return df


if __name__ == "__main__":
    cids = ["AUD", "BRL", "CAD", "CHF", "CNY", "CZK", "EUR", "GBP", "USD"]
    xcats = ["RIR_NSA", "FXXR_NSA", "FXXR_VT10", "DU05YXR_NSA", "DU05YXR_VT10"]
    start_date: str = "2024-01-07"
    end_date: str = "2024-02-09"

    with JPMaQSDownload(
        client_id=os.getenv("DQ_CLIENT_ID"),
        client_secret=os.getenv("DQ_CLIENT_SECRET"),
    ) as jpmaqs:
        data = jpmaqs.download(
            cids=cids,
            xcats=xcats,
            metrics="all",
            start_date=start_date,
            end_date=end_date,
            show_progress=True,
            report_time_taken=True,
        )
        print(data.info())
        print(data)
