from typing import List, Union, Optional, Tuple, Dict, Any
import pandas as pd
import os
import glob
from functools import lru_cache
import logging
import pickle
import json
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import concurrent.futures
import time
import itertools
import datetime
from macrosynergy.download.jpmaqs import JPMaQSDownload
from macrosynergy.download.dataquery import (
    DataQueryInterface,
    API_DELAY_PARAM,
    HL_RETRY_COUNT,
    MAX_CONTINUOUS_FAILURES,
    TIMESERIES_ENDPOINT,
    TIMESERIES_TRACKING_ID,
)
from macrosynergy.download.exceptions import (
    AuthenticationError,
    HeartbeatError,
    DownloadError,
)
from macrosynergy.management.utils import Config, form_full_url

logger = logging.getLogger(__name__)
cache = lru_cache(maxsize=None)


class LocalDataQueryInterface(DataQueryInterface):
    def __init__(self, local_path: str, fmt="pkl"):
        self.local_path = os.path.abspath(local_path)
        # check if the local path exists
        if not os.path.exists(self.local_path):
            raise FileNotFoundError(
                f"The local path provided : {self.local_path}, does not exist."
            )
        self.store_format = fmt
        logger.info(f"LocalDataQuery initialized with local_path: {self.local_path}")

    @cache
    def _find_ticker_files(
        self,
    ) -> List[str]:
        """
        Returns a list of files in the local path
        """
        # get all files in the local path with the correct extension, at any depth

        files: List[str] = glob.glob(
            os.path.join(self.local_path, f"*.{self.store_format}"), recursive=True
        )
        return files

    @cache
    def _get_ticker_path(self, ticker: str) -> str:
        """
        Returns the absolute path to the ticker file.

        :param ticker: The ticker to find the path for.
        :return: The absolute path to the ticker file.
        :raises FileNotFoundError: If the ticker is not found in the local path.
        """
        files: List[str] = self._find_ticker_files()
        for f in files:
            if ticker == f.split(os.sep)[-1].split(".")[0]:
                return f
        raise FileNotFoundError(f"Ticker {ticker} not found in {self.local_path}")

    def get_catalogue(self, *args, **kwargs) -> List[str]:
        """
        Returns a list of tickers available in the local
        tickerstore.
        """
        tickers: List[str] = [
            os.path.basename(f).split(".")[0] for f in self._find_ticker_files()
        ]
        return tickers

    def check_connection(self, verbose=False) -> bool:
        # check if _find_ticker_files returns anything
        if len(self._find_ticker_files()) > 0:
            return True
        else:
            fmt_long: str = (
                "pickle (*.pkl)" if self.store_format == "pkl" else "csv (*.csv)"
            )
            raise FileNotFoundError(
                f"The local path provided : {self.local_path}, "
                f"does not contain {fmt_long} files."
            )

    def load_data(
        self,
        expressions: List[str],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        def _load_df(ticker: str) -> pd.DataFrame:
            if self.store_format == "pkl":
                return pd.read_pickle(os.path.join(self.local_path, ticker + ".pkl"))
            elif self.store_format == "csv":
                return pd.read_csv(os.path.join(self.local_path, ticker + ".csv"))

        def _get_df(
            cid: str, xcat: str, metrics: List[str], start_date: str, end_date: str
        ) -> pd.DataFrame:
            df: pd.DataFrame = _load_df(ticker=f"{cid}_{xcat}")
            df = df[["real_date"] + metrics]
            df["real_date"] = pd.to_datetime(df["real_date"])
            df = df.loc[(df["real_date"] >= start_date) & (df["real_date"] <= end_date)]
            df["cid"] = cid
            df["xcat"] = xcat
            return df[["real_date", "cid", "xcat"] + metrics]

        # is overloaded to accept a list of expressions
        deconstr_expressions: List[List[str]] = JPMaQSDownload.deconstruct_expression(
            expression=expressions
        )

        pd.DataFrame = pd.concat(
            [
                _get_df(
                    cid=cidx,
                    xcat=xcatx,
                    metrics=metricsx,
                    start_date=start_date,
                    end_date=end_date,
                )
                for cidx, xcatx, metricsx in deconstr_expressions
            ],
            ignore_index=True,
            axis=0,
        )
        return pd.DataFrame

    def download_data(
        self,
        expressions: List[str],
        start_date: str = "2000-01-01",
        end_date: str = None,
        show_progress: bool = False,
        endpoint: str = ...,
        calender: str = "CAL_ALLDAYS",
        frequency: str = "FREQ_DAY",
        conversion: str = "CONV_LASTBUS_ABS",
        nan_treatment: str = "NA_NOTHING",
        reference_data: str = "NO_REFERENCE_DATA",
        retry_counter: int = 0,
        delay_param: float = ...,
    ) -> pd.DataFrame:
        # divide expressions into batches of 10
        batched_expressions: List[List[str]] = [
            expressions[i : i + 10] for i in range(0, len(expressions), 10)
        ]

        with Pool(cpu_count() - 1) as p:
            df: pd.DataFrame = pd.concat(
                list(
                    tqdm(
                        p.imap(
                            self.load_data,
                            batched_expressions,
                        ),
                        total=len(batched_expressions),
                        disable=not show_progress,
                        desc="Downloading data",
                    )
                ),
                ignore_index=True,
                axis=0,
            )

        return df


class LocalCache(JPMaQSDownload):
    def __init__(self, local_path: str, fmt="pkl"):
        self.local_path = os.path.abspath(local_path)
        self.store_format = fmt
        super().__init__(
            client_id="<local>",
            client_secret=f"<{self.local_path}>",
            check_connection=False,
        )
        self.dq_interface = LocalDataQueryInterface(
            local_path=self.local_path, fmt=self.store_format
        )

    def download(
        self,
        jpmaqs_df=True,
        *args,
        **kwargs,
    ) -> pd.DataFrame:
        kwargs.update({"as_dataframe": True, "get_catalogue": True})
        df: pd.DataFrame = super().download(*args, **kwargs)
        if jpmaqs_df:
            return df
        else:
            df["ticker"]: str = df["cid"] + "_" + df["xcat"]
            df = df.drop(["cid", "xcat"], axis=1)
            return df


class DownloadTimeseries(DataQueryInterface):
    def __init__(self, store_path: str, store_format: str = "pkl", *args, **kwargs):

        if store_format not in ["pkl", "json"]:
            raise ValueError(f"Store format {store_format} not supported.")

        self.store_path = os.path.abspath(store_path)
        self.store_format = store_format

        os.makedirs(self.store_path, exist_ok=True)
        os.makedirs(os.path.join(self.store_path, self.store_format), exist_ok=True)
        # get client id and secret from kwargs. remove them from kwargs
        client_id: str = kwargs.pop("client_id", None)
        client_secret: str = kwargs.pop("client_secret", None)
        cfg: Config = Config(client_id=client_id, client_secret=client_secret)
        
        super().__init__(config=cfg, *args, **kwargs)

    def _extract_timeseries(
        self,
        timeseries: Union[Dict[str, Any], List[Dict[str, Any]]],
    ):
        if isinstance(timeseries, list):
            if len(timeseries) == 0:
                return [{}]

            return [self._extract_timeseries(timeseries=ts) for ts in timeseries]

        logger.debug(
            f"Extracting timeseries for {timeseries['attributes'][0]['expression']}"
        )
        if timeseries["attributes"][0]["time-series"] is None:
            return {}
        else:
            return {
                "attributes": [
                    {
                        "expression": timeseries["attributes"][0]["expression"],
                        "time-series": timeseries["attributes"][0]["time-series"],
                    }
                ]
            }

    def _save_timeseries(
        self,
        timeseries: Union[Dict[str, Any], List[Dict[str, Any]]],
    ):

        if isinstance(timeseries, list):
            for ts in timeseries:
                if ts:
                    self._save_timeseries(timeseries=ts)
            return

        expr = timeseries["attributes"][0]["expression"]
        pathx = os.path.join(self.store_path, self.store_format)
        logger.debug(
            f"Saving timeseries for {expr}, format: {self.store_format}, path: {pathx}"
        )
        if self.store_format == "pkl":
            with open(os.path.join(pathx, expr + ".pkl"), "wb") as f:
                pickle.dump(timeseries, f)
        elif self.store_format == "json":
            with open(os.path.join(pathx, expr + ".json"), "w") as f:
                json.dump(timeseries, f)

        return True

    def _get_data(
        self,
        url: str,
        params: dict,
        tracking_id: str,
    ):
        try:
            data = self._fetch(url=url, params=params, tracking_id=tracking_id)
            data = self._extract_timeseries(timeseries=data)
            self._save_timeseries(timeseries=data)
            return True
        except Exception as e:
            logger.error(f"Failed to download data for {tracking_id} due to {e}")
            return False

    def _download(
        self,
        expressions: List[str],
        params: dict,
        url: str,
        tracking_id: str,
        delay_param: float,
        show_progress: bool = False,
        retry_counter: int = 0,
    ) -> List[bool]:
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

        failed_batches: List[List[str]] = []
        continuous_failures: int = 0
        last_five_exc: List[Exception] = []

        future_objects: List[concurrent.futures.Future] = []
        results: List[bool] = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for ib, expr_batch in tqdm(
                enumerate(expr_batches),
                desc="Requesting data",
                disable=not show_progress,
                total=len(expr_batches),
            ):
                curr_params: Dict = params.copy()
                curr_params["expressions"] = expr_batch
                logger.debug(f"Sending request with params: {curr_params}")

                future_objects.append(
                    executor.submit(
                        self._get_data,
                        url=url,
                        params=curr_params,
                        tracking_id=tracking_id,
                    )
                )
                time.sleep(delay_param)

            for ib, future_object in tqdm(
                enumerate(concurrent.futures.as_completed(future_objects)),
                desc="Downloading data",
                disable=not show_progress,
                total=len(future_objects),
            ):
                try:

                    res: Dict[str, Any] = future_object.result()
                    results.append(res)
                    if not res:
                        raise DownloadError(
                            "Could not download data, will retry.\n"
                            f"User ID: {self.auth.get_auth()['user_id']}\n"
                            f"URL: {form_full_url(url, params)}"
                            f"Timestamp (UTC): {datetime.datetime.utcnow().isoformat()}"
                        )

                    continuous_failures = 0

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

        if len(failed_batches) > 0:
            print(f"Retrying {len(failed_batches)} failed batches. ")
            self.msg_errors.append(
                f"Failed to download {len(failed_batches)} batches. Retrying..."
            )
            results += self._download(
                expressions=itertools.chain(*failed_batches),
                params=params,
                url=url,
                tracking_id=tracking_id,
                delay_param=delay_param,
                show_progress=show_progress,
                retry_counter=retry_counter + 1,
            )

        return results

    def download_data(
        self,
        expressions: List[str] = None,
        start_date: str = "1990-01-01",
        end_date: str = None,
        show_progress: bool = False,
        endpoint: str = TIMESERIES_ENDPOINT,
        calender: str = "CAL_ALLDAYS",
        frequency: str = "FREQ_DAY",
        conversion: str = "CONV_LASTBUS_ABS",
        nan_treatment: str = "NA_NOTHING",
        reference_data: str = "NO_REFERENCE_DATA",
        retry_counter: int = 0,
        delay_param: float = API_DELAY_PARAM,  # TODO do we want the user to have access to this?
    ):
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
        download_start_time: float = time.time()

        tracking_id: str = TIMESERIES_TRACKING_ID
        if end_date is None:
            end_date = datetime.datetime.today().strftime("%Y-%m-%d")

        print("Downloading tickers catalogue...")
        tickers: List[str] = self.get_catalogue()

        metrics: List[str] = ["value", "grading", "eop_lag", "mop_lag"]
        expressions: List[str] = JPMaQSDownload.construct_expressions(
            tickers=tickers, metrics=metrics
        )

        if datetime.datetime.strptime(
            end_date, "%Y-%m-%d"
        ) < datetime.datetime.strptime(start_date, "%Y-%m-%d"):
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
                        f" {datetime.datetime.utcnow().isoformat()}\n"
                        f"User ID: {self.auth.get_auth()['user_id']}\n"
                    )
                )
            time.sleep(delay_param)

        logger.info(
            "Download %d expressions from DataQuery from %s to %s",
            len(expressions),
            datetime.datetime.strptime(start_date, "%Y%m%d").date(),
            datetime.datetime.strptime(end_date, "%Y%m%d").date(),
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

        final_output: List[dict] = self._download(
            expressions=expressions,
            params=params_dict,
            url=self.base_url + endpoint,
            tracking_id=tracking_id,
            delay_param=delay_param,
            show_progress=show_progress,
        )

        download_time_taken: float = time.time() - download_start_time

        expressions_saved_files: List[str] = [
            os.path.abspath(fx)
            for fx in glob.glob(
                os.path.join(self.store_path, self.store_format, "*"), recursive=True
            )
        ]
        expressions_saved: List[str] = [
            os.path.basename(fx).split(".")[0] for fx in expressions_saved_files
        ]
        expressions_missing: List[str] = list(set(expressions) - set(expressions_saved))

        print(f"Number of expressions requested: {len(expressions)}")
        print(f"Number of expressions downloaded: {len(expressions_saved)}")

        print(f"Number of expressions missing: {len(expressions_missing)}")
        if expressions_missing:
            for expression in sorted(expressions_missing):
                print(f"Esxpression missing: {expression}")

        size_downloaded: float = sum(
            [os.path.getsize(fx) for fx in expressions_saved_files]
        )
        print(
            f"Total size of files downloaded: {size_downloaded / (1024 ** 2) :.2f} MB | {size_downloaded / (1024 ** 3) :.2f} GB"
        )

        print(
            f"Time taken to download files: {download_time_taken / 60 :.2f} minutes | {download_time_taken / 3600 :.2f} hours"
        )
