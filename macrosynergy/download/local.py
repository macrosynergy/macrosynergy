from typing import List, Union, Optional, Dict, Any, Callable, Tuple

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
import random
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
    InvalidDataframeError,
)
from macrosynergy.management.utils import form_full_url

logger = logging.getLogger(__name__)
cache = lru_cache(maxsize=None)


class LocalDataQueryInterface(DataQueryInterface):
    def __init__(self, local_path: str, fmt="pkl", *args, **kwargs):
        if local_path.startswith("~"):
            local_path = os.path.expanduser(local_path)
        self.local_path = os.path.abspath(local_path)
        # check if the local path exists
        if not os.path.exists(self.local_path):
            raise FileNotFoundError(
                f"The local path provided : {self.local_path}, does not exist."
            )
        self.store_format = fmt
        logger.info(f"LocalDataQuery initialized with local_path: {self.local_path}")
        self._find_expression_files()
        super().__init__(*args, **kwargs)

    @cache
    def _find_expression_files(
        self,
    ) -> List[str]:
        """
        Returns a list of files in the local path
        """
        # get all files in the local path with the correct extension, at any depth
        files: List[str] = glob.glob(
            os.path.join(self.local_path, self.store_format, f"*.{self.store_format}"),
            recursive=True,
        )
        self.expression_paths: Dict[str, str] = {
            os.path.basename(f).split(".")[0]: os.path.normpath(os.path.abspath(f))
            for f in files
        }

        return files

    @cache
    def _get_expression_path(self, expression: str) -> str:
        """
        Returns the absolute path to the ticker file.

        :param ticker: The ticker to find the path for.
        :return: The absolute path to the ticker file.
        :raises FileNotFoundError: If the ticker is not found in the local path.
        """
        files: List[str] = self._find_expression_files()
        r = self.expression_paths.get(expression, None)
        if r is None:
            raise FileNotFoundError(
                f"Could not find expression {expression} in the local path."
            )
        return r

    def get_catalogue(self, *args, **kwargs) -> List[str]:
        """
        Returns a list of tickers available in the local
        tickerstore.
        """
        exprs: List[List[str]] = JPMaQSDownload.deconstruct_expression(
            expression=[
                os.path.basename(f).split(".")[0] for f in self._find_expression_files()
            ]
        )
        tickers: List[str] = sorted(
            list(set([f"{expr[0]}_{expr[1]}" for expr in exprs]))
        )
        return tickers

    def get_metrics(self, *args, **kwargs) -> List[str]:
        """
        Returns a list of metrics available in the local
        tickerstore.
        """
        exprs: List[List[str]] = [
            JPMaQSDownload.deconstruct_expression(
                expression=os.path.basename(f).split(".")[0]
            )
            for f in self._find_expression_files()
        ]
        metrics: List[str] = sorted(list(set([expr[2] for expr in exprs])))
        return metrics

    def check_connection(self, verbose=False) -> bool:
        # check if _find_ticker_files returns anything
        if len(self._find_expression_files()) > 0:
            ctl: List[str] = self.get_catalogue()
            metrics: List[str] = self.get_metrics()
            for ticker in ctl:
                self._get_expression_path(
                    JPMaQSDownload.construct_expressions(
                        tickers=[ticker], metrics=metrics
                    )[0]
                )
                # verifies local paths, and builds cache
            if verbose:
                print("Connection to local tickerstore successful.")

            return True
        else:
            fmt_long: str = (
                "pickle (*.pkl)" if self.store_format == "pkl" else "csv (*.csv)"
            )
            raise FileNotFoundError(
                f"The local path provided : {self.local_path}, "
                f"does not contain {fmt_long} files."
            )

    def _load_timeseries(self, expression: str) -> Dict[str, Any]:
        loader: Callable = pickle.load if self.store_format == "pkl" else json.load
        with open(self._get_expression_path(expression=expression), "rb") as f:
            return loader(f)

    def binary_search_dates(
        self,
        date_values: List[Tuple[str, float]],
        start_date: datetime.date,
        end_date: datetime.date,
    ) -> int:
        low = 0
        high = len(date_values) - 1
        while low <= high:
            curr = (low + high) // 2
            curr_date = datetime.datetime.strptime(
                date_values[curr][0], "%Y%m%d"
            ).date()

            if start_date <= curr_date <= end_date:
                return curr
            elif curr_date < start_date:
                low = curr + 1
            else:
                high = curr - 1

        return -1

    def _filter_timeseries(
        self,
        timeseries: Dict[str, Any],
        start_date: str,
        end_date: str,
    ) -> Dict[str, Any]:
        sorted_date_values = sorted(
            timeseries["attributes"][0]["time-series"], key=lambda x: x[0]
        )
        sdt: datetime.date = datetime.datetime.strptime(start_date, "%Y-%m-%d").date()
        edt: datetime.date = datetime.datetime.strptime(end_date, "%Y-%m-%d").date()

        start_index = self.binary_search_dates(
            date_values=sorted_date_values, start_date=sdt, end_date=sdt
        )
        if start_index == -1:
            start_index = len(sorted_date_values) - 1

        end_index = self.binary_search_dates(
            date_values=sorted_date_values, start_date=edt, end_date=edt
        )
        if end_index == -1:
            end_index = len(sorted_date_values) - 1

        # filtered_values = sorted_date_values[start_index : end_index + 1]
        timeseries["attributes"][0]["time-series"] = sorted_date_values[
            start_index : end_index + 1
        ]
        return timeseries

    def _load_expressions(
        self,
        expressions: List[str],
        start_date: str = "2000-01-01",
        end_date: str = datetime.datetime.today().strftime("%Y-%m-%d"),
    ) -> List[Dict[str, Any]]:
        """
        Loads a list of expressions from the local path.

        :param expressions: list of expressions to load.
        """

        return [
            self._filter_timeseries(
                timeseries=self._load_timeseries(expression=expr),
                start_date=start_date,
                end_date=end_date,
            )
            for expr in expressions
        ]

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
    ) -> List[Dict[str, Any]]:
        batched_expressions: List[List[str]] = [
            expressions[i : i + 50] for i in range(0, len(expressions), 50)
        ]

        return list(
            itertools.chain.from_iterable(
                [
                    self._load_expressions(expressions=expr_batch)
                    for expr_batch in tqdm(
                        batched_expressions,
                        desc="Loading data",
                        disable=not show_progress,
                        total=len(batched_expressions),
                    )
                ]
            )
        )


class LocalCache(JPMaQSDownload):
    def __init__(self, local_path: str, fmt="pkl", *args, **kwargs):
        if local_path.startswith("~"):
            local_path = os.path.expanduser(local_path)
        self.local_path = os.path.abspath(local_path)
        self.store_format = fmt
        config: Dict[str, str] = dict(
            client_id="<local>", client_secret=f"<{self.local_path}>"
        )
        super().__init__(
            check_connection=False,
            **config,
        )
        self.dq_interface = LocalDataQueryInterface(
            local_path=self.local_path,
            fmt=self.store_format,
            **config,
        )

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
        :return <pd.DataFrame>: JPMaQS standard dataframe with columns:
            real_date, cid, xcat, <metric>. The <metric> column contains the
            observed data for the given cid and xcat on the given real_date.

        :raises <InvalidDataError>: if the downloaded dataframe is invalid.
        """
        dfs: List[pd.DataFrame] = []
        cid: str
        xcat: str
        found_expressions: List[str] = []
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
                pass

        final_df: pd.DataFrame = pd.concat(dfs, ignore_index=True)

        final_df = (
            final_df.set_index(["real_date", "cid", "xcat", "metric"])["obs"]
            .unstack(3)
            .rename_axis(None, axis=1)
            .reset_index()
        )

        final_df["real_date"] = pd.to_datetime(final_df["real_date"])

        expc_bdates: pd.DatetimeIndex = pd.bdate_range(start=start_date, end=end_date)
        final_df = final_df[final_df["real_date"].isin(expc_bdates)]
        final_df = final_df.sort_values(["real_date", "cid", "xcat"])
        found_metrics = sorted(
            list(set(final_df.columns) - {"real_date", "cid", "xcat"}),
            key=lambda x: self.valid_metrics.index(x),
        )
        final_df = final_df[["real_date", "cid", "xcat"] + found_metrics]

        final_df = final_df.dropna(axis=0, how="any").reset_index(drop=True)

        vdf = self.validate_downloaded_df(
            data_df=final_df,
            expected_expressions=expected_expressions,
            found_expressions=found_expressions,
            start_date=start_date,
            end_date=end_date,
            verbose=verbose,
        )
        if not vdf:
            raise InvalidDataframeError(f"Downloaded dataframe is invalid.")

        return final_df

    def download(
        self,
        jpmaqs_df=True,
        *args,
        **kwargs,
    ) -> pd.DataFrame:
        kwargs.update(
            {"as_dataframe": True, "get_catalogue": True, "report_egress": False}
        )
        self.check_connection()
        df: pd.DataFrame = super().download(*args, **kwargs)
        if jpmaqs_df:
            return df
        else:
            df["ticker"]: str = df["cid"] + "_" + df["xcat"]
            df = df.drop(["cid", "xcat"], axis=1)
            return df


class DownloadTimeseries(DataQueryInterface):
    def __init__(
        self,
        store_path: str,
        store_format: str = "pkl",
        test_mode: bool = False,
        *args,
        **kwargs,
    ):
        if store_format not in ["pkl", "json"]:
            raise ValueError(f"Store format {store_format} not supported.")

        self.store_path = os.path.abspath(store_path)
        self.store_format = store_format
        self.test_mode: bool = test_mode

        os.makedirs(self.store_path, exist_ok=True)
        os.makedirs(os.path.join(self.store_path, self.store_format), exist_ok=True)
        # get client id and secret from kwargs. remove them from kwargs
        client_id: str = kwargs.pop("client_id", None)
        client_secret: str = kwargs.pop("client_secret", None)
        cfg: Dict[str, str] = dict(client_id=client_id, client_secret=client_secret)

        super().__init__(*args, **kwargs, **cfg)

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

        if self.test_mode:
            if isinstance(self.test_mode, int):            
                tickers = tickers[: self.test_mode]
            else:
                tickers = tickers[:100]

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


def create_store(
    store_path: str,
    client_id: str,
    client_secret: str,
    fmt: str = "pkl",
    test_mode: bool = False,
) -> None:
    DownloadTimeseries(
        store_path=store_path,
        client_id=client_id,
        client_secret=client_secret,
        store_format=fmt,
        test_mode=test_mode,
    ).download_data(
        show_progress=True,
    )
    total_start_time: float = time.time()

    lc: LocalCache = LocalCache(local_path=store_path, fmt=fmt)
    # get 100 random tickers
    start_time: float = time.time()
    catalogue: List[str] = lc.get_catalogue()

    print(f"Time taken: {(time.time() - start_time) * 1000 :.2f} milliseconds")

    tickers: List[str] = random.sample(catalogue, 100)

    start_time: float = time.time()
    df: pd.DataFrame = lc.download(tickers=tickers, start_date="1990-01-01")

    print(f"Time taken: {(time.time() - start_time) * 1000 :.2f} milliseconds")

    print(df.head())
    print(df.info())

    print(f"Total time taken: {(time.time() - total_start_time) / 60 :.2f} minutes")


if __name__ == "__main__":
    import argparse

    client_id: str = os.getenv("DQ_CLIENT_ID")
    client_secret: str = os.getenv("DQ_CLIENT_SECRET")

    store_path: str = "./JPMaQSTickers"

    parser = argparse.ArgumentParser(
        description="Download JPMaQS data from DataQuery and store it locally."
    )

    parser.add_argument(
        "--client_id",
        type=str,
        default=client_id,
        help="Client ID for DataQuery API.",
    )
    parser.add_argument(
        "--client_secret",
        type=str,
        default=client_secret,
        help="Client Secret for DataQuery API.",
    )
    parser.add_argument(
        "--store_path",
        type=str,
        default=store_path,
        help="Path to store the downloaded data.",
    )

    args = parser.parse_args()

    create_store(
        store_path=args.store_path,
        client_id=args.client_id,
        client_secret=args.client_secret,
    )
