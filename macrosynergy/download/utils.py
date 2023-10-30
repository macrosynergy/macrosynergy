import time
import uuid
import requests, requests.compat
from datetime import datetime
import logging
import warnings
import pandas as pd

import os, sys

sys.path.append(os.getcwd())

from typing import List, Optional, Dict, Tuple, Union, Any, overload, Iterable

from macrosynergy import __version__ as ms_version_info
from macrosynergy.download.exceptions import (
    AuthenticationError,
    DownloadError,
    InvalidResponseError,
    HeartbeatError,
    KNOWN_EXCEPTIONS,
)
from macrosynergy.download.constants import (
    API_DELAY_PARAM,
    API_RETRY_COUNT,
    HEARTBEAT_ENDPOINT,
)

logger: logging.Logger = logging.getLogger(__name__)


def form_full_url(url: str, params: Dict = {}) -> str:
    """
    Forms a full URL from a base URL and a dictionary of parameters.
    Useful for logging and debugging.

    :param <str> url: base URL.
    :param <dict> params: dictionary of parameters.

    :return <str>: full URL
    """
    return requests.compat.quote(
        (f"{url}?{requests.compat.urlencode(params)}" if params else url),
        safe="%/:=&?~#+!$,;'@()*[]",
    )


def validate_response(
    response: requests.Response,
    user_id: str,
) -> dict:
    """
    Validates a response from the API. Raises an exception if the response
    is invalid (e.g. if the response is not a 200 status code).

    :param <requests.Response> response: response object from requests.request().

    :return <dict>: response as a dictionary. If the response is not valid,
        this function will raise an exception.

    :raises <InvalidResponseError>: if the response is not valid.
    :raises <AuthenticationError>: if the response is a 401 status code.
    :raises <KeyboardInterrupt>: if the user interrupts the download.
    """

    error_str: str = (
        f"Response: {response}\n"
        f"User ID: {user_id}\n"
        f"Requested URL: {response.request.url}\n"
        f"Response status code: {response.status_code}\n"
        f"Response headers: {response.headers}\n"
        f"Response text: {response.text}\n"
        f"Timestamp (UTC): {datetime.utcnow().isoformat()}; \n"
    )
    # TODO : Use response.raise_for_status() as a better way to check for errors
    if not response.ok:
        logger.info("Response status is NOT OK : %s", response.status_code)
        if response.status_code == 401:
            raise AuthenticationError(error_str)

        if HEARTBEAT_ENDPOINT in response.request.url:
            raise HeartbeatError(error_str)

        raise InvalidResponseError(
            f"Request did not return a 200 status code.\n{error_str}"
        )

    try:
        response_dict = response.json()
        if response_dict is None:
            raise InvalidResponseError(f"Response is empty.\n{error_str}")
        return response_dict
    except Exception as exc:
        if isinstance(exc, KeyboardInterrupt):
            raise exc

        raise InvalidResponseError(error_str + f"Error parsing response as JSON: {exc}")


def request_wrapper(
    url: str,
    headers: Optional[Dict] = None,
    params: Optional[Dict] = None,
    method: str = "get",
    tracking_id: Optional[str] = None,
    proxy: Optional[Dict] = None,
    cert: Optional[Tuple[str, str]] = None,
    **kwargs,
) -> dict:
    """
    Wrapper for requests.request() that handles retries and logging.
    All parameters and kwargs are passed to requests.request().

    :param <str> url: URL to request.
    :param <dict> headers: headers to pass to requests.request().
    :param <dict> params: params to pass to requests.request().
    :param <str> method: HTTP method to use. Must be one of "get"
        or "post". Defaults to "get".
    :param <dict> kwargs: kwargs to pass to requests.request().
    :param <str> tracking_id: default None, unique tracking ID of request.
    :param <dict> proxy: default None, dictionary of proxy settings for request.
    :param <Tuple[str, str]> cert: default None, tuple of string for filename
        of certificate and key.

    :return <dict>: response as a dictionary.

    :raises <InvalidResponseError>: if the response is not valid.
    :raises <AuthenticationError>: if the response is a 401 status code.
    :raises <DownloadError>: if the request fails after retrying.
    :raises <KeyboardInterrupt>: if the user interrupts the download.
    :raises <ValueError>: if the method is not one of "get" or "post".
    :raises <Exception>: other exceptions may be raised by requests.request().
    """

    if method not in ["get", "post"]:
        raise ValueError(f"Invalid method: {method}")

    user_id: str = kwargs.pop("user_id", "unknown")

    # insert tracking info in headers
    if headers is None:
        headers: Dict = {}
    headers["User-Agent"]: str = f"MacrosynergyPackage/{ms_version_info}"

    uuid_str: str = str(uuid.uuid4())
    if (tracking_id is None) or (tracking_id == ""):
        tracking_id: str = uuid_str
    else:
        tracking_id: str = f"uuid::{uuid_str}::{tracking_id}"

    headers["X-Tracking-Id"]: str = tracking_id

    log_url: str = form_full_url(url, params)
    logger.debug(f"Requesting URL: {log_url} with tracking_id: {tracking_id}")
    raised_exceptions: List[Exception] = []
    error_statements: List[str] = []
    error_statement: str = ""
    retry_count: int = 0
    response: Optional[requests.Response] = None
    while retry_count < API_RETRY_COUNT:
        try:
            prepared_request: requests.PreparedRequest = requests.Request(
                method, url, headers=headers, params=params, **kwargs
            ).prepare()

            response: requests.Response = requests.Session().send(
                prepared_request,
                proxies=proxy,
                cert=cert,
            )

            if isinstance(response, requests.Response):
                return validate_response(response=response, user_id=user_id)

        except Exception as exc:
            # if keyboard interrupt, raise as usual
            if isinstance(exc, KeyboardInterrupt):
                print("KeyboardInterrupt -- halting download")
                raise exc

            # authentication error, clearly not a transient error
            if isinstance(exc, AuthenticationError):
                raise exc

            error_statement = (
                f"Request to {log_url} failed with error {exc}. "
                f"User ID: {user_id}. "
                f"Retry count: {retry_count}. "
                f"Tracking ID: {tracking_id}"
            )
            raised_exceptions.append(exc)
            error_statements.append(error_statement)

            known_exceptions = KNOWN_EXCEPTIONS + [HeartbeatError]
            # NOTE : HeartBeat is a special case

            # NOTE: exceptions that need the code to break should be caught before this
            # all other exceptions are caught here and retried after a delay

            if any([isinstance(exc, e) for e in known_exceptions]):
                logger.warning(error_statement)
                retry_count += 1
                time.sleep(API_DELAY_PARAM)
            else:
                raise exc

    if isinstance(raised_exceptions[-1], HeartbeatError):
        raise HeartbeatError(error_statement)

    errs_str = "\n\n".join(
        ("\t" + str(e) + " - \n\t\t" + est)
        for e, est in zip(raised_exceptions, error_statements)
    )

    e_str = f"Request to {log_url} failed with error {raised_exceptions[-1]}. \n"
    e_str += "-" * 20 + "\n"
    if isinstance(response, requests.Response):
        e_str += f" Status code: {response.status_code}."
    e_str += (
        f" No longer retrying. Tracking ID: {tracking_id}"
        f"Exceptions raised:\n{errs_str}"
    )

    raise DownloadError(e_str)


@overload
def deconstruct_expression(expression: str) -> Tuple[str]:
    ...


@overload
def deconstruct_expression(expression: Iterable[str]) -> List[Tuple[str]]:
    ...


def deconstruct_expression(
    expression: Union[str, List[str]]
) -> Union[Tuple[str], List[Tuple[str]]]:
    """
    Deconstruct an expression into a list of cid, xcat, and metric.
    Coupled with JPMaQSDownload.time_series_to_df(), achieves the inverse of
    JPMaQSDownload.construct_expressions(). For non-JPMaQS expressions, the returned
    list will be [expression, expression, 'value']. The metric is set to 'value' to
    ensure the reported metric is consistent with the standard JPMaQS metrics
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
        return [deconstruct_expression(exprx) for exprx in expression]
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
    tickers: Optional[Iterable[str]] = None,
    cids: Optional[Iterable[str]] = None,
    xcats: Optional[Iterable[str]] = None,
    metrics: Optional[Iterable[str]] = None,
) -> Iterable[str]:
    """Construct expressions from the provided arguments.

    :param <Iterable[str]> tickers: list of tickers.
    :param <Iterable[str]> cids: list of cids.
    :param <Iterable[str]> xcats: list of xcats.
    :param <Iterable[str]> metrics: list of metrics.

    :return <Iterable[str]>: list of expressions.
    """
    if isinstance(tickers, str):
        tickers = [tickers]
    if isinstance(cids, str):
        cids = [cids]
    if isinstance(xcats, str):
        xcats = [xcats]
    if isinstance(metrics, str):
        metrics = [metrics]

    for argx, arg_name in zip(
        [tickers, cids, xcats, metrics],
        ["tickers", "cids", "xcats", "metrics"],
    ):
        if not (isinstance(argx, (list, tuple, set, pd.Series)) or argx is None):
            raise TypeError(f"`{arg_name}` must be a list, tuple, set, or pd.Series.")

    try:
        if tickers is None:
            tickers = []
        if bool(cids) and bool(xcats):
            tickers += [f"{cid}_{xcat}" for cid in cids for xcat in xcats]

        return [f"DB(JPMAQS,{tick},{metric})" for tick in tickers for metric in metrics]

    except Exception as exc:
        if isinstance(exc, TypeError):
            raise TypeError(
                "All elements of `tickers`, `cids`, `xcats`, & `metrics` "
                "must be strings when provided."
            )
        raise exc


@overload
def timeseries_to_df(timeseries_dict: dict) -> pd.DataFrame:
    ...


@overload
def timeseries_to_df(
    timeseries_dict: Iterable[dict],
) -> Union[List[pd.DataFrame], pd.DataFrame]:
    ...


def _timeseries_to_df_helper(
    tsdict: Dict[Any, Any],
    indexed: bool = False,
) -> pd.DataFrame:
    assert isinstance(tsdict, dict), "`tsdict` must be a timeseries dictionary."

    cid, xcat, metricx = deconstruct_expression(tsdict["attributes"][0]["expression"])
    df = (
        pd.DataFrame(
            tsdict["attributes"][0]["time-series"],
            columns=["real_date", metricx],
        )
        .assign(cid=cid, xcat=xcat, metric=metricx)
        .rename(columns={metricx: "obs"})
    )
    if indexed:
        df = df.set_index(["cid", "xcat", "real_date"])
    return df


def timeseries_to_df(
    timeseries_dict: Union[Dict[Any, Any], Iterable[Dict[Any, Any]]],
    combine_dfs: bool = False,
) -> Union[pd.DataFrame, Union[List[pd.DataFrame], pd.DataFrame]]:
    """Convert a timeseries dictionary to a pandas DataFrame.
    When a list of timeseries dictionaries is provided, a list of DataFrames is returned.
    If `combine_dfs` is True, a single Quantamenal DataFrame is returned.

    :param <dict> timeseries_dict: timeseries dictionary.
    :param <bool> combine_dfs: default True, whether to combine the returned
        DataFrames into a single DataFrame.

    :return <pd.DataFrame>: DataFrame of timeseries data.
    """
    if not isinstance(timeseries_dict, (dict, list)):
        raise TypeError(
            "`timeseries_dict` must be a dictionary or a list of dictionaries."
        )

    if isinstance(timeseries_dict, dict):
        return _timeseries_to_df_helper(tsdict=timeseries_dict)

    if not all(isinstance(tsdict, dict) for tsdict in timeseries_dict):
        raise TypeError("All elements of `timeseries_dict` must be dictionaries.")

    dfs_list: List[pd.DataFrame] = [
        _timeseries_to_df_helper(tsdict=tsdict) for tsdict in timeseries_dict
    ]

    if not combine_dfs:
        return dfs_list
    # now set the index to cid, xcat, real date for all dfs_list and concat them

    # join all on the index of cid, xcat, real_date
    df: pd.DataFrame = pd.concat(dfs_list, axis=0, ignore_index=True)
    df["real_date"] = pd.to_datetime(df["real_date"])
    # now make all the metrics columns with obs as the value
    df = df.pivot_table(index=["cid", "xcat", "real_date"], columns="metric")
    df.columns = df.columns.droplevel(0)
    return df.reset_index()


if __name__ == "__main__":

    def mock_request_wrapper(
        dq_expressions: List[str], start_date: str, end_date: str
    ) -> List[Dict[str, Any]]:
        """
        Contrived request method to replicate output from DataQuery. Will replicate the
        form of a JPMaQS expression from DataQuery which will subsequently be used to
        test methods held in the api.Interface() Class.
        """
        aggregator: List[dict] = []
        dates: pd.DatetimeIndex = pd.bdate_range(start_date, end_date)
        for i, elem in enumerate(dq_expressions):
            elem_dict = {
                "item": (i + 1),
                "group": None,
                "attributes": [
                    {
                        "expression": elem,
                        "label": None,
                        "attribute-id": None,
                        "attribute-name": None,
                        "time-series": [[d.strftime("%Y%m%d"), 1] for d in dates],
                    },
                ],
                "instrument-id": None,
                "instrument-name": None,
            }
            aggregator.append(elem_dict)

        return aggregator

    cids: List[str] = ["USD", "CAD", "GBP", "EUR" "JPY"]
    xcats: List[str] = ["FXXR", "EQXR", "RIR_NSA", "RIR_SA", "RIR_LAG"]
    metrics: List[str] = ["value", "grading", "eop_lag", "mop_lag"]

    expressions: List[str] = construct_expressions(
        cids=cids, xcats=xcats, metrics=metrics
    )

    tss: List[dict] = mock_request_wrapper(
        dq_expressions=expressions, start_date="19900101", end_date="20200101"
    )

    df: pd.DataFrame = timeseries_to_df(timeseries_dict=tss, combine_dfs=True)
    print(df)

    import time, random, json

    timings: List[Dict] = []

    stdate: str = "20100101"
    endate: str = "20200101"
    for i in range(10):
        # shuffle the expressions
        random.shuffle(expressions)
        mock_tss: List[dict] = mock_request_wrapper(
            dq_expressions=expressions, start_date=stdate, end_date=endate
        )
        start_time: float = time.time()
        df: pd.DataFrame = timeseries_to_df(timeseries_dict=mock_tss, combine_dfs=True)
        end_time: float = time.time()
        timings.append({"time": end_time - start_time, "n": len(expressions)})
        print(f"Completed iteration {i} in {end_time - start_time} seconds.")
        print(json.dumps(timings[-1], indent=4))

    # do it without combining the dfs
    for i in range(10):
        # shuffle the expressions
        random.shuffle(expressions)
        mock_tss: List[dict] = mock_request_wrapper(
            dq_expressions=expressions, start_date=stdate, end_date=endate
        )
        start_time: float = time.time()
        df: List[pd.DataFrame] = timeseries_to_df(
            timeseries_dict=mock_tss, combine_dfs=False
        )
        end_time: float = time.time()
        timings.append({"time": end_time - start_time, "n": len(expressions)})
        print(f"Completed iteration {i} in {end_time - start_time} seconds.")
        print(json.dumps(timings[-1], indent=4))

    timings_df: pd.DataFrame = pd.DataFrame(timings)
    print(timings_df)
