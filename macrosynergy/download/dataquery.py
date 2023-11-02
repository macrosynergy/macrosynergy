"""
Interface for downloading data from the JPMorgan DataQuery API.
This module is not intended to be used directly, but rather through
macrosynergy.download.jpmaqs.py. However, for a use cases independent
of JPMaQS, this module can be used directly to download data from the
JPMorgan DataQuery API.

::docs::DataQueryInterface::sort_first::
"""
import concurrent.futures
import time
import os
import logging
import itertools
import os
import json
import warnings
from datetime import datetime
from typing import List, Optional, Dict, Union
from tqdm import tqdm
from .dq_auth import OAuth, CertAuth
import pandas as pd

from .common import (
    # exceptions
    AuthenticationError,
    DownloadError,
    InvalidResponseError,
    HeartbeatError,
    NoContentError,
    # constants
    CERT_BASE_URL,
    OAUTH_BASE_URL,
    OAUTH_TOKEN_URL,
    JPMAQS_GROUP_ID,
    API_DELAY_PARAM,
    HL_RETRY_COUNT,
    MAX_CONTINUOUS_FAILURES,
    HEARTBEAT_ENDPOINT,
    TIMESERIES_ENDPOINT,
    CATALOGUE_ENDPOINT,
    HEARTBEAT_TRACKING_ID,
    TIMESERIES_TRACKING_ID,
    CATALOGUE_TRACKING_ID,
)
from macrosynergy.management.utils import is_valid_iso_date

from .utils import request_wrapper, form_full_url, timeseries_to_df

logger: logging.Logger = logging.getLogger(__name__)


def validate_download_args(
    expressions: List[str],
    start_date: str,
    end_date: str,
    show_progress: bool,
    endpoint: str,
    calender: str,
    frequency: str,
    conversion: str,
    nan_treatment: str,
    reference_data: str,
    retry_counter: int,
    delay_param: float,
):
    """
    Validate the arguments passed to the `download_data()` method.

    :params : -- see `download_data()` method.

    :return <bool>: True if all arguments are valid.

    :raises <TypeError>: if any of the arguments are of the wrong type.
    :raises <ValueError>: if any of the arguments are semantically incorrect.
    """

    if expressions is None:
        raise ValueError("`expressions` must be a list of strings.")

    if not isinstance(expressions, list):
        raise TypeError("`expressions` must be a list of strings.")

    if not all(isinstance(expr, str) for expr in expressions):
        raise TypeError("`expressions` must be a list of strings.")

    for varx, namex in zip([start_date, end_date], ["start_date", "end_date"]):
        if (varx is None) or not isinstance(varx, str):
            raise TypeError(f"`{namex}` must be a string.")
        if not is_valid_iso_date(varx):
            raise ValueError(
                f"`{namex}` must be a string in the ISO-8601 format (YYYY-MM-DD)."
            )

    if not isinstance(show_progress, bool):
        raise TypeError("`show_progress` must be a boolean.")

    if not isinstance(retry_counter, int):
        raise TypeError("`retry_counter` must be an integer.")

    if not isinstance(delay_param, float):
        raise TypeError("`delay_param` must be a float >=0.2 (seconds).")
    elif delay_param < 0.2:
        raise ValueError("`delay_param` must be a float >=0.2 (seconds).")

    vars_types_zip: zip = zip(
        [
            endpoint,
            calender,
            frequency,
            conversion,
            nan_treatment,
            reference_data,
        ],
        [
            "endpoint",
            "calender",
            "frequency",
            "conversion",
            "nan_treatment",
            "reference_data",
        ],
    )
    for varx, namex in vars_types_zip:
        if not isinstance(varx, str):
            raise TypeError(f"`{namex}` must be a string.")

    return True


def _get_unavailable_expressions(
    expected_exprs: List[str],
    dicts_list: Optional[List[Dict]] = None,
    dfs_list: Optional[List[pd.DataFrame]] = None,
) -> List[str]:
    """
    Method to get the expressions that are not available in the response.
    Looks at the dict["attributes"][0]["expression"] field of each dict
    in the list.

    :param <List[str]> expected_exprs: list of expressions that were requested.
    :param <List[Dict]> dicts_list: list of dicts to search for the expressions.
    :param <List[pd.DataFrame]> dfs_list: list of dataframes to search for the expressions.

    :return <List[str]>: list of expressions that were not found in the dicts.
    """
    found_exprs: List[str] = []

    if dicts_list is not None:
        found_exprs: List[str] = [
            curr_dict["attributes"][0]["expression"]
            for curr_dict in dicts_list
            if curr_dict["attributes"][0]["time-series"] is not None
        ]

    return list(set(expected_exprs) - set(found_exprs))


class DataQueryInterface(object):
    """
    High level interface for the DataQuery API.

    When using OAuth authentication:

    :param <str> client_id: client ID for the OAuth application.
    :param <str> client_secret: client secret for the OAuth application.

    When using certificate authentication:

    :param <str> crt: path to the certificate file.
    :param <str> key: path to the key file.
    :param <str> username: username for the DataQuery API.
    :param <str> password: password for the DataQuery API.

    :param <bool> oauth: whether to use OAuth authentication. Defaults to True.
    :param <bool> debug: whether to print debug messages. Defaults to False.
    :param <bool> concurrent: whether to use concurrent requests. Defaults to True.
    :param <int> batch_size: default 20, number of expressions to send in a single
        request. Must be a number between 1 and 20 (both included).
    :param <bool> check_connection: whether to send a check_connection request.
        Defaults to True.
    :param <str> base_url: base URL for the DataQuery API. Defaults to OAUTH_BASE_URL
        if `oauth` is True, CERT_BASE_URL otherwise.
    :param <str> token_url: token URL for the DataQuery API. Defaults to OAUTH_TOKEN_URL.
    :param <bool> suppress_warnings: whether to suppress warnings. Defaults to True.

    :return <DataQueryInterface>: DataQueryInterface object.

    :raises <TypeError>: if any of the parameters are of the wrong type.
    :raises <ValueError>: if any of the parameters are semantically incorrect.
    :raises <InvalidResponseError>: if the response from the server is not valid.
    :raises <DownloadError>: if the download fails to complete after a number of retries.
    :raises <HeartbeatError>: if the heartbeat (check connection) fails.
    :raises <Exception>: other exceptions may be raised by underlying functions.
    """

    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        crt: Optional[str] = None,
        key: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        proxy: Optional[dict] = None,
        oauth: bool = True,
        debug: bool = False,
        batch_size: int = 20,
        check_connection: bool = True,
        base_url: str = OAUTH_BASE_URL,
        token_url: str = OAUTH_TOKEN_URL,
        suppress_warnings: bool = True,
    ):
        self._check_connection: bool = check_connection
        self.msg_errors: List[str] = []
        self.msg_warnings: List[str] = []
        self.unavailable_expressions: List[str] = []
        self.debug: bool = debug
        self.suppress_warnings: bool = suppress_warnings
        self.batch_size: int = batch_size

        for varx, namex, typex in [
            (client_id, "client_id", str),
            (client_secret, "client_secret", str),
            (crt, "crt", str),
            (key, "key", str),
            (username, "username", str),
            (password, "password", str),
            (proxy, "proxy", dict),
        ]:
            if not isinstance(varx, typex) and varx is not None:
                raise TypeError(f"{namex} must be a {typex} and not {type(varx)}.")

        self.auth: Optional[Union[CertAuth, OAuth]] = None
        if oauth and not all([client_id, client_secret]):
            warnings.warn(
                "OAuth authentication requested but client ID and/or client secret "
                "not found. Falling back to certificate authentication.",
                UserWarning,
            )
            if not all([username, password, crt, key]):
                raise ValueError(
                    "Certificate credentials not found. "
                    "Check the parameters passed to the DataQueryInterface class."
                )
            else:
                oauth: bool = False

        if oauth:
            self.auth: OAuth = OAuth(
                client_id=client_id,
                client_secret=client_secret,
                token_url=token_url,
                proxy=proxy,
            )
        else:
            if base_url == OAUTH_BASE_URL:
                base_url: str = CERT_BASE_URL

            self.auth: CertAuth = CertAuth(
                username=username,
                password=password,
                crt=crt,
                key=key,
                proxy=proxy,
            )

        assert self.auth is not None, (
            "Unable to instantiate authentication object. "
            "Check the parameters passed to the DataQueryInterface class."
        )

        self.proxy: Optional[dict] = proxy
        self.base_url: str = base_url

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type:
            logger.error("Exception %s - %s", exc_type, exc_value)
            print(f"Exception: {exc_type} {exc_value}")

    def check_connection(self, verbose=False) -> bool:
        """
        Check the connection to the DataQuery API using the Heartbeat endpoint.

        :param <bool> verbose: whether to print a message if the heartbeat
            is successful. Useful for debugging. Defaults to False.

        :return <bool>: True if the connection is successful, False otherwise.

        :raises <HeartbeatError>: if the heartbeat fails.
        """
        logger.debug("Check if connection can be established to JPMorgan DataQuery")
        js: dict = request_wrapper(
            url=self.base_url + HEARTBEAT_ENDPOINT,
            params={"data": "NO_REFERENCE_DATA"},
            proxy=self.proxy,
            tracking_id=HEARTBEAT_TRACKING_ID,
            **self.auth.get_auth(),
        )

        result: bool = True
        if (js is None) or (not isinstance(js, dict)) or ("info" not in js):
            logger.warning("Connection to JPMorgan DataQuery heartbeat failed")
            result: bool = False

        if result:
            result = (int(js["info"]["code"]) == 200) and (
                js["info"]["message"] == "Service Available."
            )

        if verbose:
            print("Connection successful!" if result else "Connection failed.")
        return result

    def _fetch(
        self,
        url: str,
        params: dict = None,
        tracking_id: Optional[str] = None,
    ) -> List[Dict]:
        """
        Make a request to the DataQuery API using the specified parameters.
        Used to wrap a request in a thread for concurrent requests, or to
        simplify the code for single requests.

        :param <str> url: URL to request.
        :param <dict> params: parameters to send with the request.
        :param <dict> proxy: proxy to use for the request.
        :param <str> tracking_id: tracking ID to use for the request.

        :return <List[Dict]>: list of dictionaries containing the response data.

        :raises <InvalidResponseError>: if the response from the server is not valid.
        :raises <Exception>: other exceptions may be raised by underlying functions.
        """

        downloaded_data: List[Dict] = []
        response: Dict = request_wrapper(
            url=url,
            params=params,
            proxy=self.proxy,
            tracking_id=tracking_id,
            **self.auth.get_auth(),
        )

        if (response is None) or ("instruments" not in response.keys()):
            if response is not None:
                if (
                    ("info" in response)
                    and ("code" in response["info"])
                    and (int(response["info"]["code"]) == 204)
                ):
                    raise NoContentError(
                        f"Content was not found for the request: {response}\n"
                        f"User ID: {self.auth.get_auth()['user_id']}\n"
                        f"URL: {form_full_url(url, params)}\n"
                        f"Timestamp (UTC): {datetime.utcnow().isoformat()}"
                    )

            raise InvalidResponseError(
                f"Invalid response from DataQuery: {response}\n"
                f"User ID: {self.auth.get_auth()['user_id']}\n"
                f"URL: {form_full_url(url, params)}"
                f"Timestamp (UTC): {datetime.utcnow().isoformat()}"
            )

        downloaded_data.extend(response["instruments"])

        if "links" in response.keys() and response["links"][1]["next"] is not None:
            logger.info("DQ response paginated - get next response page")
            downloaded_data.extend(
                self._fetch(
                    url=self.base_url + response["links"][1]["next"],
                    params={},
                    tracking_id=tracking_id,
                )
            )

        return downloaded_data

    def get_catalogue(
        self,
        group_id: str = JPMAQS_GROUP_ID,
    ) -> List[str]:
        """
        Method to get the JPMaQS catalogue.
        Queries the DataQuery API's Groups/Search endpoint to get the list of
        tickers in the JPMaQS group. The group ID can be changed to fetch a
        different group's catalogue.

        :param <str> group_id: the group ID to fetch the catalogue for.

        :return <List[str]>: list of tickers in the JPMaQS group.

        :raises <ValueError>: if the response from the server is not valid.
        """
        old_group_id: str = "CA_QI_MACRO_SYNERGY"
        try:
            response_list: Dict = self._fetch(
                url=self.base_url + CATALOGUE_ENDPOINT,
                params={"group-id": group_id},
                tracking_id=CATALOGUE_TRACKING_ID,
            )
        except NoContentError as e:
            response_list: Dict = self._fetch(
                url=self.base_url + CATALOGUE_ENDPOINT,
                params={"group-id": old_group_id},
                tracking_id=CATALOGUE_TRACKING_ID,
            )
        except Exception as e:
            raise e

        tickers: List[str] = [d["instrument-name"] for d in response_list]
        utkr_count: int = len(tickers)
        tkr_idx: List[int] = sorted([d["item"] for d in response_list])

        if not (
            (min(tkr_idx) == 1)
            and (max(tkr_idx) == utkr_count)
            and (len(set(tkr_idx)) == utkr_count)
        ):
            raise ValueError("The downloaded catalogue is corrupt.")

        return tickers

    def get_timeseries(
        self,
        url: str,
        params: dict,
        tracking_id: str,
        to_path: Optional[str] = None,
        as_dataframe: bool = True,
    ) -> Union[List[Dict], List[bool]]:
        """
        Method to get timeseries data. Allows saving the data to a file. When saving to a
        file, each expression is saved to a separate JSON file. If the file already exists,
        it will be overwritten.
        """
        fetch_result: List[Dict] = self._fetch(
            url=url,
            params=params,
            tracking_id=tracking_id,
        )

        if (not as_dataframe) and (to_path is None):
            return fetch_result

        if to_path is not None:
            result_bools: List[bool] = []
            for d in fetch_result:
                try:
                    expr: str = d["attributes"][0]["expression"]
                    with open(os.path.join(to_path, f"{expr}.json"), "w") as f:
                        f.write(json.dumps(d))

                    logger.info(f"Saved expression {expr} to file.")
                    result_bools.append(True)
                except Exception as exc:
                    if isinstance(exc, KeyboardInterrupt):
                        raise exc
                    logger.error(
                        f"Failed to save expression {expr} to file. Exception: {exc}"
                    )
                    result_bools.append(False)

            return result_bools

        assert as_dataframe
        result_ts: pd.DataFrame = timeseries_to_df(timeseries_dict=fetch_result)

        return result_ts

    def _download(
        self,
        expressions: List[str],
        params: dict,
        url: str,
        tracking_id: str,
        delay_param: float,
        show_progress: bool = False,
        retry_counter: int = 0,
        to_path: Optional[str] = None,
        as_dataframe: bool = True,
    ) -> Union[List[Dict], List[bool], List[pd.DataFrame]]:
        """
        Backend method to download data from the DataQuery API.
        Used by the `download_data()` method.
        """

        if retry_counter > 0:
            print("Retrying failed downloads. Retry count:", retry_counter)

        if retry_counter > HL_RETRY_COUNT:
            raise DownloadError(
                f"Failed {retry_counter} times to download data all requested data.\n"
                f"No longer retrying."
            )

        expr_batches: List[List[str]] = [
            expressions[i : i + self.batch_size]
            for i in range(0, len(expressions), self.batch_size)
        ]

        download_outputs: List[List[pd.DataFrame]] = []
        failed_batches: List[List[str]] = []
        continuous_failures: int = 0
        last_five_exc: List[Exception] = []

        future_objects: List[concurrent.futures.Future] = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for ib, expr_batch in tqdm(
                enumerate(expr_batches),
                desc="Requesting data",
                disable=not show_progress,
                total=len(expr_batches),
            ):
                curr_params: Dict = params.copy()
                curr_params["expressions"] = expr_batch
                future_objects.append(
                    executor.submit(
                        self.get_timeseries,
                        url=url,
                        params=curr_params,
                        tracking_id=tracking_id,
                        to_path=to_path,
                        as_dataframe=as_dataframe,
                    )
                )
                time.sleep(delay_param)

            for ib, future in tqdm(
                enumerate(future_objects),
                desc="Downloading data",
                disable=not show_progress,
                total=len(future_objects),
            ):
                try:
                    download_outputs.append(future.result())
                    continuous_failures = 0
                    if to_path is not None:
                        if download_outputs[-1] == False:
                            raise DownloadError(
                                f"Failed to download expression {expr_batches[ib]}."
                            )
                except Exception as exc:
                    if isinstance(exc, (KeyboardInterrupt, AuthenticationError)):
                        raise exc

                    failed_batches.append(expr_batches[ib])
                    self.msg_errors.append(f"Batch {ib} failed with exception: {exc}")
                    continuous_failures += 1
                    last_five_exc.append(exc)
                    if continuous_failures > MAX_CONTINUOUS_FAILURES:
                        exc_str: str = "\n".join([str(e) for e in last_five_exc])
                        raise DownloadError(
                            f"Failed {continuous_failures} times to download data."
                            f" Last five exceptions: \n{exc_str}"
                        )

                    if self.debug:
                        raise exc

        final_output: List[pd.DataFrame] = list(
            itertools.chain.from_iterable(download_outputs)
        )

        if len(failed_batches) > 0:
            flat_failed_batches: List[str] = list(
                itertools.chain.from_iterable(failed_batches)
            )
            logger.warning(
                "Failed batches %d - retry download for %d expressions",
                len(failed_batches),
                len(flat_failed_batches),
            )
            # if the to_path is specified, the failed expressions list is a list of bools
            if to_path is not None:
                flat_failed_batches: List[str] = [
                    expressions[i]
                    for i, expr in enumerate(expressions)
                    if not download_outputs[i]
                ]

            retried_output: List = self._download(
                expressions=flat_failed_batches,
                params=params,
                url=url,
                as_dataframe=as_dataframe,
                tracking_id=tracking_id,
                delay_param=delay_param + 0.1,
                show_progress=show_progress,
                retry_counter=retry_counter + 1,
            )

            final_output += retried_output  # extend retried output

        return [f for f in final_output if not f.empty]

    def download_data(
        self,
        expressions: List[str],
        start_date: str = "2000-01-01",
        end_date: Optional[str] = None,
        show_progress: bool = False,
        to_path: Optional[str] = None,
        as_dataframe: bool = True,
        endpoint: str = TIMESERIES_ENDPOINT,
        calender: str = "CAL_ALLDAYS",
        frequency: str = "FREQ_DAY",
        conversion: str = "CONV_LASTBUS_ABS",
        nan_treatment: str = "NA_NOTHING",
        reference_data: str = "NO_REFERENCE_DATA",
        retry_counter: int = 0,
        delay_param: float = API_DELAY_PARAM,
    ) -> Union[List[Dict], List[bool], List[pd.DataFrame]]:
        """
        Download data from the DataQuery API.

        :param <List[str]> expressions: list of expressions to download.
        :param <str> start_date: start date for the data in the ISO-8601 format
            (YYYY-MM-DD).
        :param <str> end_date: end date for the data in the ISO-8601 format
            (YYYY-MM-DD).
        :param <bool> show_progress: whether to show a progress bar for the download.
        :param <str> endpoint: endpoint to use for the download.
        :param <str> calender: calendar setting to use for the download.
        :param <str> frequency: frequency of data points to use for the download.
        :param <str> conversion: conversion setting to use for the download.
        :param <str> nan_treatment: NaN treatment setting to use for the download.
        :param <str> reference_data: reference data to pass to the API kwargs.
        :param <int> retry_counter: number of times the download has been retried.
        :param <float> delay_param: delay between requests to the API.

        :return <List[Dict]>: list of dictionaries containing the response data.

        :raises <ValueError>: if any arguments are invalid or semantically incorrect
            (see validate_download_args()).
        :raises <DownloadError>: if the download fails.
        :raises <ConnectionError(HeartbeatError)>: if the heartbeat fails.
        :raises <Exception>: other exceptions may be raised by underlying functions.
        """
        tracking_id: str = TIMESERIES_TRACKING_ID
        if end_date is None:
            end_date = datetime.today().strftime("%Y-%m-%d")
            # NOTE : if "future dates" are passed, they must be passed by parent functions
            # see jpmaqs.py

        # NOTE : args validated only on first call, not on retries
        # this is because the args can be modified by the retry mechanism
        # (eg. date format)

        validate_download_args(
            expressions=expressions,
            start_date=start_date,
            end_date=end_date,
            show_progress=show_progress,
            endpoint=endpoint,
            calender=calender,
            frequency=frequency,
            conversion=conversion,
            nan_treatment=nan_treatment,
            reference_data=reference_data,
            retry_counter=retry_counter,
            delay_param=delay_param,
        )

        if datetime.strptime(end_date, "%Y-%m-%d") < datetime.strptime(
            start_date, "%Y-%m-%d"
        ):
            logger.warning(
                "Start date (%s) is after end-date (%s): swap them!",
                start_date,
                end_date,
            )
            start_date, end_date = end_date, start_date

        # remove dashes from dates to match DQ format
        start_date: str = start_date.replace("-", "")
        end_date: str = end_date.replace("-", "")

        # check heartbeat before each "batch" of requests
        if self._check_connection:
            if not self.check_connection():
                raise ConnectionError(
                    HeartbeatError(
                        f"Heartbeat failed. Timestamp (UTC):"
                        f" {datetime.utcnow().isoformat()}\n"
                        f"User ID: {self.auth.get_auth()['user_id']}\n"
                    )
                )
            time.sleep(delay_param)

        if to_path is not None:
            os.makedirs(to_path, exist_ok=True)

        logger.info(
            "Download %d expressions from DataQuery from %s to %s",
            len(expressions),
            datetime.strptime(start_date, "%Y%m%d").date(),
            datetime.strptime(end_date, "%Y%m%d").date(),
        )
        params_dict: Dict = {
            "format": "JSON",
            "start-date": start_date,
            "end-date": end_date,
            "calendar": calender,
            "frequency": frequency,
            "conversion": conversion,
            "nan_treatment": nan_treatment,
            "data": reference_data,
        }

        final_output: Union[List[bool], List[pd.DataFrame]] = self._download(
            expressions=expressions,
            params=params_dict,
            url=self.base_url + endpoint,
            tracking_id=tracking_id,
            delay_param=delay_param,
            show_progress=show_progress,
            to_path=to_path,
            as_dataframe=as_dataframe,
        )

        if to_path is None and not as_dataframe:
            self.unavailable_expressions = _get_unavailable_expressions(
                expected_exprs=expressions, dicts_list=final_output
            )
        logger.info(
            "Downloaded expressions: %d, unavailable: %d",
            len(final_output),
            len(self.unavailable_expressions) if to_path is None else sum(final_output),
        )

        return final_output


if __name__ == "__main__":
    import os

    client_id = os.getenv("DQ_CLIENT_ID")
    client_secret = os.getenv("DQ_CLIENT_SECRET")

    expressions = [
        "DB(JPMAQS,USD_EQXR_VT10,value)",
        "DB(JPMAQS,AUD_EXALLOPENNESS_NSA_1YMA,value)",
    ]

    with DataQueryInterface(
        client_id=client_id,
        client_secret=client_secret,
    ) as dq:
        assert dq.check_connection(verbose=True)

        data = dq.download_data(
            expressions=expressions,
            start_date="2020-01-25",
            end_date="2023-02-05",
            show_progress=True,
            to_path="data",
        )

    print(f"Succesfully downloaded data for {len(data)} expressions.")
