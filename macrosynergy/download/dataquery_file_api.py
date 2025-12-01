"""
Client for downloading JPMaQS data files from the JPMorgan DataQuery File API.

This module provides the `DataQueryFileAPIClient`, a high-level wrapper for the
JPMorgan DataQuery File API.


.. note::
    This functionality is currently in BETA and is subject to significant changes
    without deprecation cycles.

Consumption & Examples
----------------------

Before using the client, ensure your API credentials are set as environment variables:

.. code-block:: bash

    export DQ_CLIENT_ID="your_client_id"
    export DQ_CLIENT_SECRET="your_client_secret"

**Example 1: Initialize the client and list all available JPMaQS files.**

.. code-block:: python

    from macrosynergy.download import DataQueryFileAPIClient
    import pandas as pd

    client = DataQueryFileAPIClient()

    # Fetch a DataFrame of all available files for the JPMaQS group
    available_files_df = client.list_available_files_for_all_file_groups()
    print("Available JPMaQS files:")
    print(available_files_df.head())

**Example 2: Download all new or updated files for the current day.**

This is the recommended way to get a daily snapshot of all JPMaQS data,
including full datasets, deltas, and metadata.

.. code-block:: python

    from macrosynergy.download import DataQueryFileAPIClient
    client = DataQueryFileAPIClient(out_dir="./jpmaqs_data")

    print(f"Downloading today's files to {client.out_dir}...")
    client.download_full_snapshot()
    print("Download complete.")

**Example 3: Download all new or updated files for the day, and load data from them
as a dataframe.**

Here, the client checks locally available files, compares them to the latest files.
It automatically downloads new or updated files, and loads data for the specified `cids`, `xcats`,
`tickers`, and `start_date`/`end_date` as appropriate.
The resulting dataframe is returned to the user in the chosen dataframe format
(quantamental format/tickers format) and dataframe type (`pandas`/`polars`).


.. code-block:: python

    from macrosynergy.download import DataQueryFileAPIClient

    cids = ['AUD', 'CAD', 'USD', 'JPY']
    xcats = ['EQXR_NSA', 'RIR_NSA']
    start_date = '2000-01-01'

    with DataQueryFileAPIClient(out_dir="./jpmaqs_data") as client:
        df = client.download(cids=cids, xcats=xcats, start_date=start_date)
        print(df.head())


.. code-block:: python

       real_date  cid     xcat  value  eop_lag  mop_lag  grading        last_updated
    0 2000-01-03  AUD  RIR_NSA  4.078      0.0     55.0     1.25 2024-07-25 07:27:22
    1 2000-01-04  AUD  RIR_NSA  3.778      0.0     56.0     1.25 2024-07-25 07:27:22
    2 2000-01-05  AUD  RIR_NSA  3.747      0.0     56.0     1.25 2024-07-25 07:27:22
    3 2000-01-06  AUD  RIR_NSA  3.710      0.0     56.0     1.25 2024-07-25 07:27:22
    4 2000-01-07  AUD  RIR_NSA  3.697      0.0     57.0     1.25 2024-07-25 07:27:22


**Example 4: Download all new or updated delta-files since a specific date/time.**

.. code-block:: python

    from macrosynergy.download import DataQueryFileAPIClient
    import pandas as pd

    client = DataQueryFileAPIClient("./jpmaqs_data")
    since_datetime = pd.Timestamp.today() - pd.DateOffset(days=10)

    client.download_full_snapshot(
        since_datetime=since_datetime,
        include_full_snapshots=False,
        include_metadata=True,
        include_delta=True,
    )
    print("Download complete.")


**Example 5: Download a single, specific historical file.**

.. code-block:: python

    from macrosynergy.download import DataQueryFileAPIClient
    client = DataQueryFileAPIClient("./jpmaqs_data")
    # This specific filename can be found using the list_available_files... methods
    target_filename = "JPMAQS_MACROECONOMIC_BALANCE_SHEETS_20250414.parquet"

    print(f"Downloading {target_filename}...")
    file_path = client.download_file(filename=target_filename)
    print(f"File downloaded to: {file_path}")

**Example 6: Check availability for a specific file-group.**

.. code-block:: python

    from macrosynergy.download import DataQueryFileAPIClient
    client = DataQueryFileAPIClient()
    file_group_id = "JPMAQS_MACROECONOMIC_BALANCE_SHEETS"

    available_files = client.list_available_files(file_group_id=file_group_id)

    # print the earliest file's details
    print(available_files.iloc[-1])

**Example 7: Download all historical full snapshot files (vintages) for JPMaQS.**

Please note:
    - This is a **VERY LARGE** download, taking 1hr+ and around 1GB/snapshot.
    - This method is **NOT** recommended for regular use.
    - This method should **ONLY** be used for audit and archival purposes.

.. code-block:: python

    from macrosynergy.download import DataQueryFileAPIClient
    client = DataQueryFileAPIClient(out_dir="./jpmaqs_full_snapshots")
    earliest_date = "20220101" # a date before the earliest available file

    client.download_full_snapshot(
        since_datetime=earliest_date,
        include_delta=False,
        include_metadata=False,
    )

"""

import os
import pandas as pd
import polars as pl

import functools
import time
from pathlib import Path
from enum import Enum

import concurrent.futures as cf
import logging
import shutil
import traceback as tb
import uuid
from typing import Dict, Any, Optional, List, Tuple, Union, Sequence
from tqdm import tqdm
import json

import requests
from macrosynergy.compat import PD_2_0_OR_LATER, PYTHON_3_8_OR_LATER
from macrosynergy.management.constants import JPMAQS_METRICS
from macrosynergy.download.dataquery import JPMAQS_GROUP_ID
from macrosynergy.download.fusion_interface import (
    request_wrapper,
    request_wrapper_stream_bytes_to_disk,
    _wait_for_api_call,
    convert_ticker_based_parquet_file_to_qdf,
)
from macrosynergy.download.dataquery import OAUTH_TOKEN_URL
from macrosynergy.download.exceptions import DownloadError, InvalidResponseError
from macrosynergy.download.jpm_oauth import JPMorganOAuth

DQ_FILE_API_BASE_URL: str = (
    "https://api-strm-gw01.jpmchase.com/research/dataquery-authe/api/v2"
)
DQ_FILE_API_SCOPE: str = "JPMC:URI:RS-06785-DataQueryExternalApi-PROD"
DQ_FILE_API_TIMEOUT: float = 300.0
DQ_FILE_API_HEADERS_TIMEOUT: float = DQ_FILE_API_TIMEOUT / 10.0
DQ_FILE_API_DELAY_PARAM: float = 0.04  # =1/25 ; 25 transactions per second
DQ_FILE_API_DELAY_MARGIN: float = 1.1  # 10% safety margin
DQ_FILE_API_SEGMENT_SIZE_MB: float = 8.0  # 8 MB
DQ_FILE_API_STREAM_CHUNK_SIZE: int = 8192  # 8 KB


JPMAQS_EARLIEST_FILE_DATE = "20220101"

logger = logging.getLogger(__name__)


class DataQueryFileAPIOauth(JPMorganOAuth):
    """
    A class to handle OAuth authentication for the JPMorgan DataQuery File API.
    """

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        resource: str = DQ_FILE_API_SCOPE,
        auth_url: str = OAUTH_TOKEN_URL,
        root_url: str = DQ_FILE_API_BASE_URL,
        application_name: str = "DataQueryFileAPI",
        proxies: Optional[Dict[str, str]] = None,
        verify: bool = True,
        **kwargs,
    ):
        super().__init__(
            client_id=client_id,
            client_secret=client_secret,
            resource=resource,
            application_name=application_name,
            auth_url=auth_url,
            root_url=root_url,
            proxies=proxies,
            verify=verify,
            **kwargs,
        )


class DataQueryFileAPIClient:
    """
    A client for accessing JPMaQS product files via the JPMorgan DataQuery File API.

    This client provides an alternative distribution channel to the Fusion API for JPMaQS
    data. It is designed to list and download JPMaQS data files, which are
    available as full snapshots, daily deltas, and metadata files. The client handles
    authentication, API requests, and file downloads, including large file downloads
    using a segmented, concurrent approach.

    Parameters
    ----------
    client_id : Optional[str]
        Client ID for authentication. If not provided, it will be sourced from
        environment variables (`DQ_CLIENT_ID` or `DATAQUERY_CLIENT_ID`).
    client_secret : Optional[str]
        Client Secret for authentication. If not provided, it will be sourced from
        environment variables (`DQ_CLIENT_SECRET` or `DATAQUERY_CLIENT_SECRET`).
    out_dir : Optional[str]
        Default output directory for downloads. Can be overridden in download methods.
    base_url : str
        The base URL for the DataQuery File API. Defaults to `DQ_FILE_API_BASE_URL`.
    scope : str
        The API scope for authentication. Defaults to `DQ_FILE_API_SCOPE`.
    proxies : Optional[Dict[str, str]]
        Optional proxies to use for HTTP requests. Defaults to None.
    verify_ssl : bool
        If True, verifies SSL certificates for all requests. Defaults to True.
    """

    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        out_dir: Optional[str] = None,
        base_url: str = DQ_FILE_API_BASE_URL,
        scope: str = DQ_FILE_API_SCOPE,
        proxies: Optional[Dict[str, str]] = None,
        verify_ssl: bool = True,
    ):
        if not (bool(client_id) and bool(client_secret)):
            client_id, client_secret = get_client_id_secret()

        if not (bool(client_id) and bool(client_secret)):
            raise ValueError(
                "Client ID and Client Secret must be provided either as arguments or "
                "via environment variables DQ_CLIENT_ID & DQ_CLIENT_SECRET or "
                "DATAQUERY_CLIENT_ID & DATAQUERY_CLIENT_SECRET"
            )

        self.client_id = client_id
        self.client_secret = client_secret
        self.out_dir = out_dir or "./jpmaqs-download"

        self.base_url = base_url.rstrip("/")
        self.scope = scope
        self.proxies = proxies
        self.verify_ssl = verify_ssl
        self.catalog_file_group_id = "JPMAQS_METADATA_CATALOG"

        self.oauth = DataQueryFileAPIOauth(
            client_id=self.client_id,
            client_secret=self.client_secret,
            resource=self.scope,
            verify=self.verify_ssl,
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            logger.error(tb.format_exc())
        return False

    def _get_save_dir(self, out_dir: Optional[str] = None) -> str:
        base_dir = Path(out_dir or self.out_dir)
        if base_dir.name != "jpmaqs-download":
            return str(base_dir / "jpmaqs-download")
        return str(base_dir)

    def _get(
        self, endpoint: str, params: Optional[Dict[str, Any]] = None, retries: int = 3
    ) -> Dict[str, Any]:
        """
        Executes a GET request to a specified endpoint with retry logic.

        Parameters
        ----------
        endpoint : str
            The API endpoint to call.
        params : Optional[Dict[str, Any]]
            A dictionary of query parameters for the request.
        retries : int
            The number of times to retry the request in case of failure.

        Returns
        -------
        Dict[str, Any]
            The JSON response from the API as a dictionary.
        """
        url = f"{self.base_url}{endpoint}"
        headers = self.oauth.get_headers()
        for _ in range(retries):
            try:
                return request_wrapper(
                    method="GET",
                    url=url,
                    headers=headers,
                    params=params or {},
                    proxies=self.proxies,
                    as_json=True,
                    api_delay=DQ_FILE_API_DELAY_PARAM,
                    verify_ssl=self.verify_ssl,
                )
            except Exception as e:
                logger.error(f"Error occurred during GET request: {e}")
                if _ == retries - 1:
                    raise
                logger.info(f"Retrying... ({_ + 1}/{retries})")
                time.sleep(2**_)

    def list_groups(self) -> pd.DataFrame:
        """
        Lists all available data provider groups.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing details of available groups.
        """
        endpoint = "/groups"
        payload = self._get(endpoint, {})
        return pd.json_normalize(payload, record_path=["groups"])

    def search_groups(self, keywords: str) -> pd.DataFrame:
        """
        Searches for data provider groups that match the given keywords.

        Parameters
        ----------
        keywords : str
            Keywords to search for in group names and descriptions.

        Returns
        -------
        pd.DataFrame
            A DataFrame of groups matching the search criteria.
        """
        endpoint = "/groups/search"
        payload = self._get(endpoint, {"keywords": keywords})
        return pd.json_normalize(payload, record_path=["groups"])

    @functools.lru_cache(maxsize=None)
    def list_group_files(
        self,
        group_id: str = JPMAQS_GROUP_ID,
        include_full_snapshots: bool = True,
        include_delta: bool = True,
        include_metadata: bool = True,
    ) -> pd.DataFrame:
        """
        Lists all file groups (datasets) for a specific data provider.

        Parameters
        ----------
        group_id : str
            The identifier for the data provider group, defaults to the JPMaQS group.
        include_full_snapshots : bool
            If True, include full snapshot file groups in the result.
        include_delta : bool
            If True, include delta file groups in the result.
        include_metadata : bool
            If True, include metadata file groups in the result.

        Returns
        -------
        pd.DataFrame
            A DataFrame listing the available file groups.
        """
        if not any([include_full_snapshots, include_delta, include_metadata]):
            raise ValueError(
                "At least one of `include_full_snapshots`, `include_delta`, or "
                "`include_metadata` must be True"
            )

        endpoint = "/group/files"
        payload = self._get(endpoint, {"group-id": group_id})
        df = pd.json_normalize(payload, record_path=["file-group-ids"])

        isdeltafile = df["file-group-id"].str.endswith("_DELTA")
        ismetadata = df["file-group-id"].str.contains("_METADATA")
        isfullsnapshot = ~(isdeltafile | ismetadata)

        mask = pd.Series(False, index=df.index)
        if include_full_snapshots:
            mask |= isfullsnapshot
        if include_delta:
            mask |= isdeltafile
        if include_metadata:
            mask |= ismetadata
        df = df[mask]

        df = df.sort_values(by=["item"]).reset_index(drop=True)

        return df

    def list_available_files(
        self,
        file_group_id: str,
        group_id: str = JPMAQS_GROUP_ID,
        start_date: str = JPMAQS_EARLIEST_FILE_DATE,
        end_date: str = None,
        convert_metadata_timestamps: bool = True,
        include_unavailable: bool = False,
    ) -> pd.DataFrame:
        """
        Lists all available files for a specific file group within a date range.

        Parameters
        ----------
        file_group_id : str
            The identifier for the file group (e.g., "JPMAQS_MACROECONOMIC_BALANCE_SHEETS").
        group_id : str
            The identifier for the data provider group.
        start_date : str
            The start date for the search in "YYYYMMDD" format.
        end_date : str
            The end date for the search in "YYYYMMDD" format. Defaults to today.
        convert_metadata_timestamps : bool
            If True, convert timestamp columns to datetime objects.
        include_unavailable : bool
            If True, includes files that are listed but not currently available.

        Returns
        -------
        pd.DataFrame
            A DataFrame of available files with their details.
        """
        if end_date is None:
            end_date = pd.Timestamp.utcnow().strftime("%Y%m%d")
        endpoint = "/group/files/available-files"
        params = {
            "group-id": group_id,
            "file-group-id": file_group_id,
            "start-date": start_date,
            "end-date": end_date,
        }
        payload = self._get(endpoint, params)
        df = pd.json_normalize(payload, record_path=["available-files"])

        if "file-datetime" not in df.columns:
            raise InvalidResponseError(
                f'Missing "file-datetime" in response from {endpoint} with params {params}'
            )
        if not include_unavailable:
            df = df[df["is-available"]].copy()
        df.loc[:, "file-datetime"] = df["file-datetime"].astype(str)

        # Sort by real timestamp while leaving the column as string
        df["_ts"] = pd_to_datetime_compat(df["file-datetime"], utc=True)
        df = (
            df.sort_values("_ts", ascending=False)
            .drop(columns="_ts")
            .reset_index(drop=True)
        )

        if convert_metadata_timestamps:
            for col in ["file-datetime", "last-modified"]:
                if col not in df.columns:
                    raise InvalidResponseError(f'Missing "{col}" in response')
                df[col] = pd_to_datetime_compat(df[col], utc=True)
        return df

    def list_available_files_for_all_file_groups(
        self,
        group_id: str = JPMAQS_GROUP_ID,
        start_date: str = JPMAQS_EARLIEST_FILE_DATE,
        end_date: str = None,
        include_full_snapshots: bool = True,
        include_delta: bool = True,
        include_metadata: bool = True,
        convert_metadata_timestamps: bool = True,
        include_unavailable: bool = False,
    ) -> pd.DataFrame:
        """
        Fetches and consolidates available files for all relevant file groups.

        This method concurrently queries for available files across all specified
        file group types (full snapshots, deltas, metadata) for a given provider.

        Parameters
        ----------
        group_id : str
            The identifier for the data provider group.
        start_date : str
            The start date for the search in "YYYYMMDD" format.
        end_date : str
            The end date for the search in "YYYYMMDD" format. Defaults to today.
        include_full_snapshots : bool
            If True, query for full snapshot file groups.
        include_delta : bool
            If True, query for delta file groups.
        include_metadata : bool
            If True, query for metadata file groups.
        convert_metadata_timestamps : bool
            If True, convert timestamp columns to datetime objects.
        include_unavailable : bool
            If True, include files that are listed but not currently available.

        Returns
        -------
        pd.DataFrame
            A consolidated DataFrame of all available files.
        """
        files_groups = self.list_group_files(
            include_full_snapshots=include_full_snapshots,
            include_delta=include_delta,
            include_metadata=include_metadata,
        )["file-group-id"].tolist()
        results = []
        with cf.ThreadPoolExecutor() as executor:
            futures = {}
            for file_group_id in files_groups:
                futures[
                    executor.submit(
                        self.list_available_files,
                        group_id=group_id,
                        file_group_id=file_group_id,
                        start_date=start_date,
                        end_date=end_date,
                        convert_metadata_timestamps=convert_metadata_timestamps,
                        include_unavailable=include_unavailable,
                    )
                ] = file_group_id
                time.sleep(DQ_FILE_API_DELAY_PARAM)

            for future in cf.as_completed(futures):
                available_files = future.result()
                results.append(available_files)

        files_df = pd.concat(results).reset_index(drop=True)

        return files_df

    def filter_available_files_by_datetime(
        self,
        since_datetime: Optional[str] = None,
        to_datetime: Optional[str] = None,
        include_full_snapshots: bool = True,
        include_delta: bool = True,
        include_metadata: bool = True,
        include_unavailable: bool = False,
    ) -> pd.DataFrame:
        """
        Retrieves files whose 'last-modified' timestamp falls within a datetime window.

        Parameters
        ----------
        since_datetime : Optional[str]
            The start of the time window (inclusive). Format "YYYYMMDD" or "YYYYMMDDTHHMMSS".
            Defaults to the start of the current day (UTC).
        to_datetime : Optional[str]
            The end of the time window (inclusive). Format "YYYYMMDD" or "YYYYMMDDTHHMMSS".
            Defaults to the current timestamp (UTC).
        include_full_snapshots : bool
            If True, include full snapshot files in the search.
        include_delta : bool
            If True, include delta files in the search.
        include_metadata : bool
            If True, include metadata files in the search.
        include_unavailable : bool
            If True, include files that are not currently available for download.

        Returns
        -------
        pd.DataFrame
            A DataFrame of files modified within the specified time window.
        """
        if since_datetime is None:
            since_datetime = pd.Timestamp.utcnow().strftime("%Y%m%d")
        if to_datetime is None:
            to_datetime = pd.Timestamp.utcnow().strftime("%Y%m%dT%H%M%S")
        validate_dq_timestamp(since_datetime, var_name="since_datetime")
        validate_dq_timestamp(to_datetime, var_name="to_datetime")

        since_ts = pd_to_datetime_compat(since_datetime, utc=True)
        to_ts = pd_to_datetime_compat(to_datetime, utc=True)

        if "T" not in str(since_datetime):
            since_ts = since_ts.normalize()

        if "T" not in str(to_datetime):
            to_ts = (
                to_ts.normalize() + pd.DateOffset(days=1) - pd.Timedelta(nanoseconds=1)
            )

        filter_date = since_ts.normalize()

        if since_ts > to_ts:
            logger.warning(
                f"`since_datetime` ({since_ts}) is after `to_datetime` ({to_ts}). Swapping values."
            )
            since_ts, to_ts = to_ts, since_ts

        # Using DQ's internal filtering does not work as expected for JPMaQS end users,
        # hence filtering is done locally instead of passing API parameters.
        files_df = self.list_available_files_for_all_file_groups(
            include_full_snapshots=include_full_snapshots,
            include_delta=include_delta,
            include_metadata=include_metadata,
            include_unavailable=include_unavailable,
        )
        files_df = files_df[files_df["file-datetime"] >= filter_date]
        files_df = files_df[files_df["last-modified"].between(since_ts, to_ts)]
        files_df = files_df.sort_values(
            by=["file-datetime", "last-modified"],
            ascending=[False, False],
        ).reset_index(drop=True)
        return files_df

    def check_file_availability(
        self,
        file_group_id: str = None,
        file_datetime: str = None,
        filename: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Checks if a specific file is available for download.

        Provide either (`file_group_id` and `file_datetime`) or `filename`.

        Parameters
        ----------
        file_group_id : str
            The identifier for the file group.
        file_datetime : str
            The file's timestamp identifier.
        filename : Optional[str]
            The full name of the file (e.g., "JPMAQS_GENERIC_RETURNS_20250501.parquet").

        Returns
        -------
        pd.DataFrame
            A DataFrame with the file's availability status.
        """
        if not ((bool(file_group_id) and bool(file_datetime)) ^ bool(filename)):
            raise ValueError(
                "One of `file_group_id` & `file_datetime`, or `filename` must be provided."
            )
        endpoint = "/group/file/availability"
        params = {"file-group-id": file_group_id, "file-datetime": file_datetime}
        payload = self._get(endpoint, params)
        return pd.json_normalize(payload)

    def download_file(
        self,
        file_group_id: str = None,
        file_datetime: str = None,
        filename: Optional[str] = None,
        out_dir: Optional[str] = None,
        overwrite: bool = False,
        qdf: bool = False,
        as_csv: bool = False,
        keep_raw_data: bool = False,
        chunk_size: Optional[int] = None,
        timeout: Optional[float] = DQ_FILE_API_TIMEOUT,
        max_retries: int = 3,
    ) -> str:
        """
        Downloads a single Parquet file to a specified directory.

        This method can be called with either (`file_group_id` and `file_datetime`)
        or a `filename`. For large files, it automatically uses the
        `SegmentedFileDownloader` for a robust, multi-part download.

        Parameters
        ----------
        file_group_id : str
            The identifier of the file group to download from.
        file_datetime : str
            The timestamp of the file to download.
        filename : Optional[str]
            The full filename to download. Overrides `file_group_id` and `file_datetime`.
        out_dir : str
            The directory where the file will be saved.
        overwrite : bool
            If True, overwrites the file if it already exists. Default is False.
        qdf : bool
            If True, converts the DataFrame to a QuantamentalDataFrame. If False, files
            are saved as-is in the ticker-based Parquet format. Default is False.
        as_csv : bool
            If True, saves the downloaded datasets as CSV files. Default is False, with
            Parquet as the default format.
        keep_raw_data : bool
            If True, keeps the raw data files after conversion. Default is False.
        chunk_size : Optional[int]
            The chunk size for streaming downloads (in bytes).
        timeout : Optional[float]
            The timeout for the download request in seconds.
        max_retries : int
            The number of retries for the entire file download.

        Returns
        -------
        str
            The full path to the downloaded file.
        """
        out_dir = self._get_save_dir(out_dir)
        if not ((bool(file_group_id) and bool(file_datetime)) ^ bool(filename)):
            raise ValueError(
                "One of `file_group_id` & `file_datetime`, or `filename` must be provided."
            )
        if not file_group_id:
            try:
                file_group_id, file_datetime_with_ext = filename.rsplit("_", 1)
                file_datetime = file_datetime_with_ext.split(".")[0]
            except ValueError:
                raise ValueError(f"Invalid filename format: {filename}")
        endpoint = "/group/file/download"
        url = f"{self.base_url}{endpoint}"
        headers = self.oauth.get_headers()
        params = {"file-group-id": file_group_id, "file-datetime": file_datetime}

        file_name = filename or f"{file_group_id}_{file_datetime}.parquet"
        file_date = pd_to_datetime_compat(file_datetime).strftime("%Y-%m-%d")
        file_path = Path(out_dir) / Path(file_date) / Path(file_name)

        file_path.parent.mkdir(parents=True, exist_ok=True)
        if file_path.exists():
            if not overwrite:
                logger.warning(f"File {file_path} already exists. Skipping download.")
                return str(file_path)
            logger.warning(f"File {file_path} already exists. It will be overwritten.")
            file_path.unlink()

        logger.info(f"Starting download of {file_name}...")
        start = time.time()

        download_args = dict(
            filename=str(file_path),
            url=url,
            headers=headers,
            params=params,
            proxies=self.proxies,
            chunk_size=chunk_size,
            timeout=timeout,
            api_delay=DQ_FILE_API_DELAY_PARAM,
            verify_ssl=self.verify_ssl,
        )

        is_small_file = any(x in file_group_id.lower() for x in ["delta", "metadata"])
        if "_DELTA" in file_group_id:
            is_small_file = file_datetime not in large_delta_file_datetimes()

        is_catalog_file = file_group_id == self.catalog_file_group_id
        if is_small_file:
            request_wrapper_stream_bytes_to_disk(**download_args)
        else:
            SegmentedFileDownloader(
                **download_args,
                max_file_retries=max_retries,
                start_download=True,
            )

        time_taken = time.time() - start
        logger.info(
            f"Downloaded {file_name} in {time_taken:.2f} seconds to {file_path}"
        )
        if not (qdf or as_csv) or is_catalog_file or not file_path.suffix == ".parquet":
            return str(file_path)

        convert_args = dict(
            filename=str(file_path),
            as_csv=as_csv,
            qdf=qdf,
            keep_raw_data=keep_raw_data,
        )

        if PYTHON_3_8_OR_LATER:
            convert_ticker_based_parquet_file_to_qdf_pl(**convert_args)
        else:
            convert_ticker_based_parquet_file_to_qdf(**convert_args)
        if qdf:
            msg_str = (
                f"Successfully converted {filename} to Quantamental Data Format (QDF)"
            )
            if as_csv:
                msg_str += " and saved as CSV"
            logger.info(msg_str)
        return str(file_path)

    def delete_corrupt_files(
        self,
        out_dir: Optional[str] = None,
        files: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Deletes corrupt files from the provided list based on file integrity checks.

        Parameters
        ----------
        out_dir : Optional[str]
            The directory to scan for corrupt files. If None, uses the client's default
            output directory.
        files : Optional[List[str]]
            A list of file paths to check for corruption. If None, scans all downloaded
            files in the specified output directory.

        Returns
        -------
        List[str]
            A list of file paths that were identified as corrupt and deleted.
        """
        out_dir = self._get_save_dir(out_dir)
        avail_files = self.list_downloaded_files(out_dir=out_dir)
        if avail_files.empty:
            return []
        if files is not None:
            if not all(isinstance(f, str) for f in files):
                raise ValueError(
                    "All items in `files` must be strings representing file paths."
                )
            avail_files = avail_files[avail_files["file-name"].isin(files)]
        files = sorted(set(map(str, avail_files["path"])))
        extensions = sorted(set(Path(f).suffix.rsplit(".", 1)[-1] for f in files))
        return _delete_corrupt_files(files=files, extensions=extensions)

    def download_multiple_files(
        self,
        filenames: List[str],
        out_dir: Optional[str] = None,
        overwrite: bool = False,
        qdf: bool = False,
        as_csv: bool = False,
        keep_raw_data: bool = False,
        max_retries: int = 3,
        n_jobs: int = None,
        chunk_size: Optional[int] = None,
        timeout: Optional[float] = DQ_FILE_API_TIMEOUT,
        show_progress: bool = True,
    ) -> None:
        """
        Downloads a list of files concurrently with progress indication.

        Parameters
        ----------
        filenames : List[str]
            A list of full filenames to be downloaded.
        out_dir : str
            The directory to save the downloaded files.
        overwrite : bool
            If True, overwrites files if they already exist. Default is False.
        qdf : bool
            If True, converts the DataFrame to a QuantamentalDataFrame. If False, files
            are saved as-is in the ticker-based Parquet format. Default is False.
        as_csv : bool
            If True, saves the DataFrame as a CSV file. Default is False.
        keep_raw_data : bool
            If True, keeps the raw data files after conversion. Default is False.
        max_retries : int
            The number of times to retry downloading the entire list of failed files.
        n_jobs : int
            The number of concurrent download jobs. If -1, it uses all available cores.
        chunk_size : Optional[int]
            The chunk size for streaming downloads (in bytes).
        timeout : Optional[float]
            The timeout for each download request in seconds.
        show_progress : bool
            If True, displays a progress bar for the downloads.
        """
        out_dir = self._get_save_dir(out_dir)
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        start_time = time.time()
        logger.info(f"Starting download of {len(filenames)} files.")
        failed_files = []
        if n_jobs == -1:
            n_jobs = None
        with cf.ThreadPoolExecutor(max_workers=n_jobs) as executor:
            futures = {}
            for filename in tqdm(
                filenames,
                desc="Requesting files",
                disable=not show_progress,
            ):
                futures[
                    executor.submit(
                        self.download_file,
                        filename=filename,
                        out_dir=out_dir,
                        overwrite=overwrite,
                        qdf=qdf,
                        as_csv=as_csv,
                        keep_raw_data=keep_raw_data,
                        chunk_size=chunk_size,
                        timeout=timeout,
                    )
                ] = filename
                time.sleep(DQ_FILE_API_DELAY_PARAM)

            for future in tqdm(
                cf.as_completed(futures),
                total=len(futures),
                desc="Downloading files",
                disable=not show_progress,
            ):
                fname = futures[future]
                try:
                    future.result()
                except KeyboardInterrupt:
                    executor.shutdown(wait=False, cancel_futures=True)
                    raise
                except Exception as e:
                    logger.error(f"Failed to download {fname}: {e}")
                    failed_files.append(fname)
        found_corrupt_files = self.delete_corrupt_files(
            out_dir=out_dir, files=filenames
        )
        failed_files = sorted(set(failed_files + found_corrupt_files))
        if not failed_files:
            total_time = time.time() - start_time
            logger.info(
                f"Successfully downloaded {len(filenames)} files in {total_time:.2f} seconds."
            )
            return  # All downloads scuccessful

        log_msg = f"Failed to download {len(failed_files)} files"
        if max_retries > 0:
            log_msg += f"; retrying {max_retries} more times"
        else:
            log_msg += "; no retries left"
        logger.warning(log_msg)
        if max_retries == 0:
            logger.error(f"Files failed after retries: {failed_files}")
            raise DownloadError(f"Files failed after retries: {failed_files}")

        return self.download_multiple_files(
            filenames=failed_files,
            out_dir=out_dir,
            max_retries=max_retries - 1,
            n_jobs=n_jobs,
            chunk_size=chunk_size,
            timeout=timeout,
            show_progress=show_progress,
        )

    def download_catalog_file(
        self,
        out_dir: Optional[str] = None,
        add_dataset_column: bool = False,
        as_csv: bool = False,
        overwrite: bool = False,
        keep_raw_data: bool = False,
        timeout: Optional[float] = DQ_FILE_API_TIMEOUT,
    ) -> str:
        out_dir = self._get_save_dir(out_dir)
        available_catalogs = self.list_available_files(self.catalog_file_group_id)
        if available_catalogs.empty:
            raise DownloadError("No catalog files available for download.")
        latest_catalog = available_catalogs.sort_values(
            by=["file-datetime", "last-modified"], ascending=False
        ).iloc[0]
        latest_filename = latest_catalog["file-name"]
        logger.info(f"Latest catalog file identified: {latest_filename}")

        # check if file already exists
        file_path = None
        existing_files = self.list_downloaded_files(out_dir=out_dir)
        if not overwrite and not existing_files.empty:
            if latest_filename in sorted(existing_files["file-name"]):
                file_path = existing_files[
                    existing_files["file-name"] == latest_filename
                ]["path"].values[0]

        if file_path is None:
            file_path = self.download_file(
                filename=latest_filename,
                out_dir=out_dir,
                overwrite=overwrite,
                timeout=timeout,
            )

        if not (add_dataset_column or as_csv):
            return file_path

        df = pd.read_parquet(file_path)

        if add_dataset_column:
            df.loc[:, "Dataset"] = df["Theme"].apply(
                lambda x: "JPMAQS_" + str(x).upper().replace(" ", "_")
            )

        if as_csv:
            csv_file_path = Path(file_path).with_suffix(".csv")
            df.to_csv(csv_file_path, index=False)
            if not keep_raw_data:
                Path(file_path).unlink()
            file_path = str(csv_file_path)
        else:
            df.to_parquet(file_path, index=False)

        return file_path

    def get_datasets_for_indicators(
        self,
        tickers: Optional[List[str]] = None,
        cids: Optional[List[str]] = None,
        xcats: Optional[List[str]] = None,
        case_sensitive: bool = False,
        out_dir: Optional[str] = None,
    ) -> List[str]:
        for param, name in zip(
            [tickers, cids, xcats],
            ["tickers", "cids", "xcats"],
        ):
            if param is not None:
                if not isinstance(param, list) or not all(
                    isinstance(x, str) for x in param
                ):
                    raise ValueError(f"`{name}` must be a list of strings.")

        if not any(bool(x) for x in [tickers, cids, xcats]):
            raise ValueError(
                "At least one of `tickers`, `cids`, or `xcats` must be set."
            )

        if tickers is None:
            tickers = []

        if bool(cids) ^ bool(xcats):
            raise ValueError("Either both `cids` and `xcats` must be set, or neither.")

        if cids is None:
            cids, xcats = [], []

        tickers = sorted(set(tickers + [f"{c}_{x}" for c in cids for x in xcats]))
        if not tickers or not any(t.strip() for t in tickers):
            raise ValueError("No valid tickers to search for.")

        catalog_file = self.download_catalog_file(
            out_dir=out_dir,
            add_dataset_column=True,
            as_csv=False,
        )

        catalog_df = pd.read_parquet(catalog_file)

        if case_sensitive:
            catalog_df = catalog_df[catalog_df["Ticker"].isin(tickers)]
        else:
            catalog_df = catalog_df[
                catalog_df["Ticker"].str.lower().isin(t.lower() for t in tickers)
            ]

        datasets_to_keep = sorted(set(catalog_df["Dataset"]))
        return datasets_to_keep

    def list_downloaded_files(
        self,
        out_dir: Optional[str] = None,
    ) -> pd.DataFrame:
        out_dir = self._get_save_dir()
        col_order = [
            "filename",
            "file-datetime",
            "dataset",
            "filetype",
            "file-timestamp",
            "path",
        ]
        dfs = [
            _downloaded_files_df(out_dir, file_format=fmt, include_metadata_files=True)
            for fmt in ["parquet", "csv", "json"]
        ]
        dfs = [_ for _ in dfs if _ is not _.empty]
        if not dfs:
            return pd.DataFrame(columns=col_order)
        files_df = pd.concat(dfs).reset_index(drop=True)
        if files_df.empty:
            return files_df

        files_df = files_df[col_order].rename(columns={"filename": "file-name"})
        return files_df

    def download_full_snapshot(
        self,
        out_dir: Optional[str] = None,
        since_datetime: Optional[str] = None,
        to_datetime: Optional[str] = None,
        file_datetime: Optional[str] = None,
        overwrite: bool = False,
        qdf: bool = False,
        as_csv: bool = False,
        keep_raw_data: bool = False,
        chunk_size: Optional[int] = None,
        timeout: Optional[float] = DQ_FILE_API_TIMEOUT,
        include_full_snapshots: bool = True,
        include_delta: bool = True,
        include_metadata: bool = True,
        file_group_ids: Optional[List[str]] = None,
        show_progress: bool = True,
    ) -> None:
        """
        Downloads a complete snapshot of files based on specified criteria.

        This method fetches a list of files modified within a given time window and
        then downloads them. It can be customized to download only specific file types
        or from a specific list of file groups.

        Parameters
        ----------
        out_dir : str
            The directory where files will be saved.
        since_datetime : Optional[str]
            Download files modified since this timestamp (inclusive).
            Defaults to the start of the current day (UTC) if `file_datetime` is not set.
        to_datetime : Optional[str]
            Download files modified up to this timestamp (inclusive).
        file_datetime : Optional[str]
            A specific file date to check for. Overrides `since_datetime`.
        overwrite : bool
            If True, overwrites files if they already exist. Default is False.
        qdf : bool
            If True, converts the DataFrame to a QuantamentalDataFrame. If False, files
            are saved as-is in the ticker-based Parquet format. Default is False.
        as_csv : bool
            If True, saves the downloaded datasets as CSV files. Default is False, with
            Parquet as the default format.
        keep_raw_data : bool
            If True, keeps the raw data files after conversion. Default is False.
        chunk_size : Optional[int]
            The chunk size for streaming downloads (in bytes).
        timeout : Optional[float]
            The timeout for each download request in seconds.
        include_full_snapshots : bool
            If True, download full snapshot files.
        include_delta : bool
            If True, download delta files.
        include_metadata : bool
            If True, download metadata files.
        file_group_ids : Optional[List[str]]
            A specific list of file groups to download from. If provided, only files
            from these groups will be downloaded.
        show_progress : bool
            If True, displays a progress bar for downloads.
        """
        out_dir = self._get_save_dir(out_dir)
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        start_time = time.time()

        if file_datetime is None and since_datetime is None:
            since_datetime = pd.Timestamp.utcnow().strftime("%Y%m%d")

        effective_ts = file_datetime or since_datetime
        logger.info(
            f"Starting snapshot download to '{out_dir}' for files since {effective_ts}."
        )

        validate_dq_timestamp(
            effective_ts,
            var_name="file_datetime" if file_datetime else "since_datetime",
        )

        files_df = self.filter_available_files_by_datetime(
            since_datetime=since_datetime,
            to_datetime=to_datetime,
            include_full_snapshots=include_full_snapshots,
            include_delta=include_delta,
            include_metadata=include_metadata,
        )

        if file_group_ids is not None:
            if not isinstance(file_group_ids, list) or not all(
                isinstance(x, str) for x in file_group_ids
            ):
                raise ValueError("`file_group_ids` must be a list of strings.")
            files_df = files_df[files_df["file-group-id"].isin(file_group_ids)].copy()

        downloaded_files_df = self.list_downloaded_files(out_dir=out_dir)
        if not overwrite and not downloaded_files_df.empty:
            files_df = files_df[
                ~(files_df["file-name"].isin(downloaded_files_df["file-name"]))
            ].copy()
            num_files_to_download = len(files_df["file-name"])

        num_files_to_download = len(files_df["file-name"])
        logger.info(f"Found {num_files_to_download} new files to download.")
        if not num_files_to_download:
            logger.info("No new files to download.")
            return

        files_df["download-priority"] = (
            files_df["file-name"]
            .str.lower()
            .apply(lambda x: (3 if "_metadata" in x else (2 if "_delta" in x else 1)))
        )
        download_order = files_df.sort_values(
            by=["download-priority", "file-datetime", "file-name"],
        )["file-name"].tolist()

        self.download_multiple_files(
            filenames=download_order,
            out_dir=out_dir,
            overwrite=overwrite,
            qdf=qdf,
            as_csv=as_csv,
            keep_raw_data=keep_raw_data,
            chunk_size=chunk_size,
            timeout=timeout,
            show_progress=show_progress,
        )

        total_time = time.time() - start_time
        logger.info(f"Snapshot download completed in {total_time:.2f} seconds.")

    def download(
        self,
        tickers: Optional[List[str]] = None,
        cids: Optional[List[str]] = None,
        xcats: Optional[List[str]] = None,
        metrics: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        dataframe_format: str = "qdf",
        dataframe_type: str = "pandas",
        categorical_dataframe: bool = True,
        include_delta_files: bool = False,
        show_progress: bool = True,
        out_dir: Optional[str] = None,
        overwrite: bool = False,
        qdf: bool = False,
        keep_raw_data: bool = False,
        as_csv: bool = False,
    ) -> Union[pd.DataFrame, pl.DataFrame, pl.LazyFrame]:
        """
        A method to download data and load it as a DataFrame based on specified
        indicators, and specified date range.

        Parameters
        ----------
        tickers : Optional[List[str]]
            A list of tickers to filter datasets. Each ticker must be in the standard
            format "CID_XCAT" used in JPMaQS.
        cids : Optional[List[str]]
            A list of cross-sectional identifiers (CIDs) to filter datasets.
        xcats : Optional[List[str]]
            A list of extended categories (XCATS) to filter datasets.
        metrics : Optional[List[str]]
            A list of JPMaQS metrics to filter the data. Available metrics are "value",
            "grading", "eop_lag", "mop_lag", and "last_updated". The available metrics
            are also defined in `macrosynergy.constants.JPMAQS_METRICS`. The default
            is None, in which case all metrics are returned.
        start_date : Optional[str]
            The start date for the returned data in the ISO format "YYYY-MM-DD".
            If None, data is returned from the earliest available date.
        end_date : Optional[str]
            The end date for the returned data in the ISO format "YYYY-MM-DD".
            If None, data is returned up to the latest available date.
        dataframe_format : str
            The format of the returned DataFrame. Options are "qdf" for QuantamentalDataFrame
            or "tickers" for a standard DataFrame with tickers as columns. Default is "qdf".
        dataframe_type : str
            The type of DataFrame to return. Options are "pandas" for a pandas DataFrame,
            "polars" for a polars DataFrame, or "polars-lazy" for a polars LazyFrame.
            Default is "pandas".
        categorical_dataframe : bool
            If True and `dataframe_type` is "pandas", the returned DataFrame will use
            categorical dtypes for object columns. Default is True.
        include_delta_files : bool
            If True, delta files will be included in the download. Default is False.
        show_progress : bool
            If True, displays a progress bar during downloads. Default is True.
        out_dir : Optional[str]
            The output directory for downloaded files. The default directory being used
            by the DataQueryFileAPI instance is used if None.
        overwrite : bool
            If True, overwrites files if they already exist. Default is False.
        qdf : bool
            If True, each downloaded dataframe will be saved as a QuantamentalDataFrame,
            otherwise files are saved as-is in the ticker-based Parquet format.
            Default is False.
        keep_raw_data : bool
            If True, keeps the raw data files after conversion. Default is False.
        as_csv : bool
            If True, saves the downloaded datasets as CSV files. Default is False, with
            Parquet as the default format.

        Returns
        -------
        Union[pd.DataFrame, pl.DataFrame, pl.LazyFrame]
            A DataFrame containing the requested data.
        """
        if include_delta_files:
            raise NotImplementedError(
                "Downloading delta files is not implemented in this method."
            )

        out_dir = self._get_save_dir(out_dir)
        datasets_to_download = self.get_datasets_for_indicators(
            tickers=tickers, cids=cids, xcats=xcats
        )
        self.download_full_snapshot(
            out_dir=out_dir,
            since_datetime=pd.Timestamp.utcnow().strftime("%Y%m%d"),
            file_group_ids=datasets_to_download,
            overwrite=overwrite,
            qdf=qdf,
            as_csv=as_csv,
            keep_raw_data=keep_raw_data,
            show_progress=show_progress,
            include_full_snapshots=True,
            include_delta=include_delta_files,
            include_metadata=False,
        )
        return lazy_load_from_parquets(
            files_dir=out_dir,
            tickers=tickers,
            cids=cids,
            xcats=xcats,
            metrics=metrics,
            start_date=start_date,
            end_date=end_date,
            dataframe_format=dataframe_format,
            dataframe_type=dataframe_type,
            categorical_dataframe=categorical_dataframe,
            datasets=datasets_to_download,
        )


def _pd_to_datetime_compat(ts: str, utc: bool):
    formats = [
        "%Y%m%d",
        "%Y%m%dT%H%M%S",
        "%Y-%m-%d",
        "%Y-%m-%dT%H:%M:%S",
        # ISO with timezone information
        "%Y-%m-%dT%H:%M:%SZ",  # UTC with Z (e.g. 2025-09-16T12:34:56Z)
        "%Y-%m-%dT%H:%M:%S%z",  # With numeric offset (e.g. 2025-09-16T12:34:56+02:00 or +0200)
    ]
    formats_str = f"[{', '.join(formats).replace('%', '').upper()}]"
    for fmt in formats:
        try:
            return pd.to_datetime(ts, format=fmt, utc=utc)
        except (ValueError, TypeError):
            continue
    raise ValueError(
        f"Timestamp '{ts}' does not match expected formats. Use one of {formats_str}."
    )


def pd_to_datetime_compat(
    ts: Union[str, pd.Series],
    format: str = "mixed",
    utc: bool = True,
):
    if PD_2_0_OR_LATER:
        return pd.to_datetime(ts, format=format, utc=utc)
    if isinstance(ts, pd.Series):
        return ts.apply(lambda x: _pd_to_datetime_compat(x, utc=utc))
    return _pd_to_datetime_compat(ts, utc=utc)


def validate_dq_timestamp(
    ts: str, var_name: str = None, raise_error: bool = True
) -> bool:
    """Validate a timestamp string for DataQuery API."""
    try:
        if PD_2_0_OR_LATER:
            pd.to_datetime(ts, format="mixed", utc=True)
        else:
            pd_to_datetime_compat(ts, utc=True)
        return True
    except (ValueError, TypeError):
        if raise_error:
            vn = f"`{var_name}`" if var_name else "Timestamp"
            raise ValueError(
                f"Invalid {vn} format. Use YYYYMMDD, YYYYMMDDTHHMMSS, or a "
                "recognized timestamp format with timezone."
            )
        else:
            return False


def get_client_id_secret() -> Optional[Tuple[str, str]]:
    """Retrieve client ID and secret from environment variables."""
    pairs = [
        ("DQ_CLIENT_ID", "DQ_CLIENT_SECRET"),
        ("DATAQUERY_CLIENT_ID", "DATAQUERY_CLIENT_SECRET"),
    ]
    for client_id_env, client_secret_env in pairs:
        client_id = os.getenv(client_id_env)
        client_secret = os.getenv(client_secret_env)
        if client_id and client_secret:
            logger.info(
                f"Using {client_id_env} and {client_secret_env} from environment"
            )
            return client_id, client_secret

    return None, None


@functools.lru_cache(maxsize=1)
def large_delta_file_datetimes(as_str: bool = True) -> List[str]:
    """
    Plausible file datetimes for large delta files, which are typically
    generated at the end of each month and on business month ends, with timestamps of
    end-of-day (23:59:59).
    """
    sd, ed = JPMAQS_EARLIEST_FILE_DATE, pd.Timestamp.today()
    dt1 = list(pd.date_range(start=sd, end=ed, freq="M"))
    dt2 = list(pd.date_range(start=sd, end=ed, freq="BM"))
    all_dates = sorted(set(dt1 + dt2))
    all_dates = [
        d.normalize() + pd.Timedelta(hours=23, minutes=59, seconds=59)
        for d in all_dates
    ]
    if not as_str:
        return all_dates
    return [d.strftime("%Y%m%dT%H%M%S") for d in all_dates]


def _delete_corrupt_files(
    files: List[Path],
    extensions: List[str] = ["parquet", "json"],
    allow_empty: bool = False,
) -> List[Path]:
    """Deletes corrupt files based on their extensions."""
    removed_files = []
    for file_path in map(Path, files):
        if not file_path.exists():
            continue
        if file_path.suffix.lower() not in [
            f".{ext.strip('.').lower()}" for ext in extensions
        ]:
            continue
        try:
            if file_path.suffix.lower() == ".parquet":
                head = pl.scan_parquet(file_path).head().collect()
                if not allow_empty and head.is_empty():
                    raise ValueError("File is empty")
            elif file_path.suffix.lower() == ".json":
                with open(file_path, "r", encoding="utf-8") as f:
                    js = json.load(f)
                    if not allow_empty and not js:
                        raise ValueError("File is empty")
            else:
                continue
        except KeyboardInterrupt:
            raise
        except Exception:
            logger.warning(f"Deleting corrupt file: {file_path}")
            file_path.unlink()
            removed_files.append(file_path)

    return sorted(map(str, removed_files))


class SegmentedFileDownloader:
    """
    A utility class to manage the multi-part, concurrent download of a single large file.
    """

    def __init__(
        self,
        filename: str,
        url: str,
        headers: Dict[str, str],
        params: Dict[str, str],
        proxies: Optional[Dict[str, str]] = None,
        chunk_size: int = DQ_FILE_API_STREAM_CHUNK_SIZE,
        segment_size_mb: int = DQ_FILE_API_SEGMENT_SIZE_MB,
        timeout: int = DQ_FILE_API_TIMEOUT,
        api_delay: float = DQ_FILE_API_DELAY_PARAM,
        api_delay_margin: float = DQ_FILE_API_DELAY_MARGIN,
        headers_timeout: int = DQ_FILE_API_HEADERS_TIMEOUT,
        max_concurrent_downloads: int = None,
        max_file_retries: int = 3,
        verify_ssl: bool = True,
        start_download: bool = False,
        debug: bool = False,
    ):
        """Initializes the downloader with URL, headers, and download parameters."""
        self.filename = Path(filename)
        self.url = url
        self.headers = headers
        self.params = params
        if not set(["file-group-id", "file-datetime"]).issubset(params):
            raise ValueError(
                "Missing required parameters: 'file-group-id' and 'file-datetime'"
            )

        self.file_id = params["file-group-id"] + "_" + params["file-datetime"]
        self.proxies = proxies
        self.out_dir = Path(self.filename.parent)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_size = chunk_size
        self.segment_size_mb = segment_size_mb
        self.timeout = timeout
        self.api_delay = api_delay * api_delay_margin
        self.headers_timeout = headers_timeout
        self.max_concurrent_downloads = max_concurrent_downloads
        self.max_file_retries = max_file_retries
        self.verify_ssl = verify_ssl
        self.debug = debug
        self.temp_dir = self.out_dir / f"_tmp_{self.filename.name}_{uuid.uuid4().hex}"

        if start_download:
            try:
                self.download()
            except Exception:
                self.cleanup()
                raise

    def __enter__(self):
        """Allows the downloader to be used as a context manager."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Ensures cleanup of temporary files upon exiting the context."""
        if exc_type is not None:
            logger.error(tb.format_exc())
        self.cleanup()
        return False

    def log(self, msg: str, part_num: int = None, level: int = logging.INFO):
        """Logs a message with downloader-specific context."""
        part_info = f"[part={part_num}]" if part_num is not None else ""
        logger.log(
            level, f"[SegmentedFileDownloader][file={self.file_id}]{part_info} {msg}"
        )

    def download(self, retries: int = None) -> Path:
        """Orchestrates the entire file download process, including retries."""
        last_exception = None
        if retries is None:
            retries = self.max_file_retries

        try:
            self.log("Starting segmented file download")
            start_time = time.time()
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
            self.temp_dir.mkdir(exist_ok=True, parents=True)

            total_size = self._get_file_size()
            self.log(f"File size: {total_size / (1024*1024):.2f} MB")

            chunk_size = int(self.segment_size_mb * 1024 * 1024)
            chunks = range(0, total_size, chunk_size)
            self.log(f"Creating {len(chunks)} download tasks")

            self._download_chunks_concurrently(chunks, total_size)

            final_path = Path(self.filename).resolve()
            self._assemble_parts(final_path, len(chunks))

            duration = time.time() - start_time
            self.log(f"Download complete in {duration:.2f} seconds.")
            self.log(f"Saved to: {final_path}")
            return final_path
        except KeyboardInterrupt:
            raise
        except Exception as e:
            last_exception = e
            self.log(f"Download failed. Error: {e}", level=logging.ERROR)
            if self.debug:
                raise e
            if retries > 0:
                self.log(
                    f"Retrying download (attempt {self.max_file_retries - retries + 1}/{self.max_file_retries})..."
                )
                time.sleep(self.api_delay)
                self.cleanup()
                return self.download(retries=retries - 1)

            self.cleanup()

        raise last_exception

    def _get_file_size(self) -> int:
        """Fetches the total size of the file using a HEAD request."""
        self.log("Fetching file size...")
        _wait_for_api_call(self.api_delay)
        start_time = time.time()
        response = requests.head(
            self.url,
            params=self.params,
            headers=self.headers,
            proxies=self.proxies,
            verify=self.verify_ssl,
        )
        response.raise_for_status()
        duration = time.time() - start_time
        self.log(f"Received headers in {duration:.2f} seconds.")
        cl_header = response.headers.get("Content-Length")
        try:
            content_length = int(cl_header)
        except (ValueError, TypeError):
            raise ValueError(
                f"[SegmentedFileDownloader][file={self.file_id}] Invalid or missing Content-Length header: {cl_header}."
            )
        self.log(f"Content-Length: {content_length}")
        return content_length

    def _download_chunks_concurrently(self, chunks: range, total_size: int):
        """Manages the parallel download of all file chunks."""
        with cf.ThreadPoolExecutor(
            max_workers=self.max_concurrent_downloads
        ) as executor:
            futures = []
            for i, start in enumerate(chunks):
                # wait before next API call
                future = executor.submit(
                    self._download_chunk,
                    i,
                    start,
                    min(start + chunks.step - 1, total_size - 1),
                )
                futures.append(future)
            try:
                for future in cf.as_completed(futures):
                    if future.exception():
                        executor.shutdown(wait=False, cancel_futures=True)
                        raise future.exception()
            except KeyboardInterrupt:
                executor.shutdown(wait=False, cancel_futures=True)
                raise

    def _download_chunk(self, part_num: int, start_byte: int, end_byte: int) -> None:
        """Starts the download process for a single file chunk."""
        self._download_chunk_retry(part_num, start_byte, end_byte, retries=1)

    def _download_chunk_retry(
        self, part_num: int, start_byte: int, end_byte: int, retries: int
    ) -> None:
        """Downloads a specific byte range of the file with a retry mechanism."""
        self.log(f"Downloading bytes [{start_byte}-{end_byte}]", part_num=part_num)
        segment_headers = self.headers.copy()
        segment_headers["Range"] = f"bytes={start_byte}-{end_byte}"
        part_path = self.temp_dir / f"part_{part_num}"

        try:
            _wait_for_api_call(self.api_delay)
            with requests.get(
                headers=segment_headers,
                url=self.url,
                params=self.params,
                proxies=self.proxies,
                stream=True,
                timeout=self.timeout,
                verify=self.verify_ssl,
            ) as response:
                response.raise_for_status()
                with open(part_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=self.chunk_size):
                        f.write(chunk)
            self.log("Finished download.", part_num=part_num)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            if isinstance(e, requests.exceptions.HTTPError):
                if hasattr(e, "response") and hasattr(e.response, "status_code"):
                    if 400 <= e.response.status_code < 500:
                        retries = 0
                        raise e
            self.log(
                f"FAILED download. Error: {e}", part_num=part_num, level=logging.ERROR
            )
            if retries > 0:
                self.log("Retrying download...", part_num=part_num)
                self._download_chunk_retry(part_num, start_byte, end_byte, retries - 1)
            else:
                raise

    def _assemble_parts(self, final_path: Path, num_parts: int):
        """Combines the downloaded chunks into a single final file."""
        self.log(f"Assembling {num_parts} parts")
        with open(final_path, "wb") as final_file:
            for i in range(num_parts):
                part_path = self.temp_dir / f"part_{i}"
                with open(part_path, "rb") as part_file:
                    shutil.copyfileobj(part_file, final_file)
        final_size = final_path.stat().st_size
        self.log(f"Assembled file size: {final_size / (1024*1024):.2f} MB")
        self.cleanup()

    def cleanup(self):
        """Removes the temporary directory and all downloaded parts."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            self.log("Cleaned up temporary files.")


def _atomic_sink_csv(lf: pl.LazyFrame, final_out: Path, sidecar: Path) -> None:
    """Atomic sink for CSV files - ensures complete writes/cleans up on failure."""
    try:
        sidecar.unlink()
    except FileNotFoundError:
        pass

    try:
        lf.sink_csv(str(sidecar))
        os.replace(sidecar, final_out)
    except BaseException:
        try:
            sidecar.unlink()
        except FileNotFoundError:
            pass
        raise


def _atomic_sink_parquet(
    lf: pl.LazyFrame, final_out: Path, sidecar: Path, *, compression: str
) -> None:
    """Atomic sink for Parquet files - ensures complete writes/cleans up on failure."""
    try:
        sidecar.unlink()
    except FileNotFoundError:
        pass

    try:
        lf.sink_parquet(str(sidecar), compression=compression)
        os.replace(sidecar, final_out)
    except BaseException:
        try:
            sidecar.unlink()
        except FileNotFoundError:
            pass
        raise


def _convert_ticker_based_parquet_file_to_qdf_pl(
    filename: str,
    compression: str = "zstd",
    as_csv: bool = False,
    qdf: bool = False,
    keep_raw_data: bool = False,
) -> None:
    src = Path(filename)
    if not src.is_file():
        raise FileNotFoundError(f"No such file: {filename}")

    base = src.with_suffix("")
    dirpath = src.parent

    # passthrough to CSV from sink_csv
    if as_csv and not qdf:
        final_out = base.with_suffix(".csv")
        sidecar = dirpath / f".{final_out.name}.inprogress"
        _atomic_sink_csv(pl.scan_parquet(str(src)), final_out, sidecar)
        if not keep_raw_data:
            src.unlink(missing_ok=True)
        return

    if not qdf:
        return

    lf = pl.scan_parquet(str(src))
    parts = pl.col("ticker").str.splitn("_", 2)
    lf = lf.with_columns(
        cid=parts.struct.field("field_0"),
        xcat=parts.struct.field("field_1"),
    )

    wanted = ["real_date", "value", "grading", "eop_lag", "mop_lag", "last_updated"]
    present = [c for c in wanted if c in lf.collect_schema().names()]
    lf = lf.select(present + ["cid", "xcat"])

    if as_csv:
        final_out = (
            base.with_suffix(".csv")
            if not keep_raw_data
            else dirpath / f"{base.name}_qdf.csv"
        )
        sidecar = dirpath / f".{final_out.name}.inprogress"
        _atomic_sink_csv(lf, final_out, sidecar)
        if not keep_raw_data:
            src.unlink(missing_ok=True)
    else:
        if keep_raw_data:
            final_out = dirpath / f"{base.name}_qdf.parquet"
        else:
            final_out = src
        sidecar = dirpath / f".{final_out.name}.inprogress"
        _atomic_sink_parquet(lf, final_out, sidecar, compression=compression)
        if not keep_raw_data and final_out is not src:
            src.unlink(missing_ok=True)


def convert_ticker_based_parquet_file_to_qdf_pl(
    filename: str,
    compression: str = "zstd",
    as_csv: bool = False,
    qdf: bool = True,
    keep_raw_data: bool = False,
) -> None:
    try:
        _convert_ticker_based_parquet_file_to_qdf_pl(
            filename=filename,
            compression=compression,
            as_csv=as_csv,
            qdf=qdf,
            keep_raw_data=keep_raw_data,
        )
    except Exception as e:
        logger.error(f"Error converting file {filename}: {e}")
        try:
            p = Path(filename)
            for cand in [
                f".{p.with_suffix('.csv').name}.inprogress",
                f".{p.name}.inprogress",
                f".{p.with_suffix('').name}_qdf.csv.inprogress",
                f".{p.with_suffix('').name}_qdf.parquet.inprogress",
            ]:
                try:
                    (p.parent / cand).unlink()
                except FileNotFoundError:
                    pass
        except Exception:
            pass

        if not Path(filename).is_file():
            raise FileNotFoundError(
                f"Conversion failed and file not found: {filename}"
            ) from e
        raise


def _check_lazy_load_inputs(
    files_dir: Union[str, Path],
    file_format: str,
    tickers: Optional[List[str]],
    cids: Optional[List[str]],
    xcats: Optional[List[str]],
    metrics: Optional[List[str]],
    start_date: Optional[Union[str, pd.Timestamp]],
    end_date: Optional[Union[str, pd.Timestamp]],
    dataframe_format: str,
    dataframe_type: str,
    categorical_dataframe: bool,
    datasets: Optional[List[str]] = None,
):
    files_dir = Path(files_dir)
    if not files_dir.is_dir():
        raise FileNotFoundError(f"No such directory: {files_dir}")

    if file_format not in ["parquet", "csv"]:
        raise ValueError("`file_format` must be one of 'parquet' or 'csv'.")
    if file_format == "csv":
        raise NotImplementedError("CSV file format is not yet supported.")
    # check whether or not there are any parquet files in the glob directory -recursive
    if not _list_downloaded_files(files_dir, file_format):
        raise FileNotFoundError(
            f"No {file_format} files found in directory: {files_dir}"
        )

    for param, name in [
        (tickers, "tickers"),
        (cids, "cids"),
        (xcats, "xcats"),
        (metrics, "metrics"),
        (datasets, "datasets"),
    ]:
        if param is not None and (
            not isinstance(param, list) or not all(isinstance(x, str) for x in param)
        ):
            raise ValueError(f"If provided, `{name}` must be a list of strings.")

    if bool(cids) ^ bool(xcats):
        raise ValueError(
            "Both `cids` and `xcats` must be provided together, or neither."
        )

    for param, name in [
        (start_date, "start_date"),
        (end_date, "end_date"),
    ]:
        if param is not None and not isinstance(param, (str, pd.Timestamp)):
            raise ValueError(f"`{name}` must be a string or pandas Timestamp.")

    if dataframe_format not in ["qdf", "wide", "tickers"]:
        raise ValueError("`dataframe_format` must be one of 'qdf', 'wide', 'tickers'.")

    if dataframe_type not in ["pandas", "polars", "polars-lazy"]:
        raise ValueError(
            "`dataframe_type` must be one of 'pandas', 'polars', 'polars-lazy'."
        )
    if not isinstance(categorical_dataframe, bool):
        raise ValueError("`categorical_dataframe` must be a boolean.")


def _list_downloaded_files(files_dir: Path, file_format: str = "parquet") -> List[Path]:
    files_dir = Path(files_dir)
    assert files_dir.is_dir(), f"No such directory: {files_dir}"
    if file_format not in ["parquet", "csv", "json"]:
        raise ValueError("`file_format` must be one of 'parquet', 'csv', or 'json'.")
    files = sorted(files_dir.glob(f"**/*.{file_format}"))
    return files


def _downloaded_files_df(
    files_dir: Path,
    file_format: str = "parquet",
    include_metadata_files: bool = False,
) -> pd.DataFrame:
    if not Path(files_dir).is_dir():
        return pd.DataFrame(columns=["path", "filename", "filetype", "dataset"])
    files_list = _list_downloaded_files(files_dir, file_format)
    df = pd.DataFrame({"path": files_list})
    if df.empty:
        return df
    df["path"] = df["path"].apply(lambda x: Path(x).resolve())
    df["filename"] = df["path"].apply(lambda x: Path(x).name)
    if not include_metadata_files:
        df = df[~df["filename"].str.contains("_METADATA")].copy()
    df["filetype"] = df["path"].apply(lambda x: Path(x).suffix.split(".")[-1])

    df["dataset"] = df["filename"].apply(
        lambda x: str(x).split(".")[0].rsplit("_", 1)[0]
    )
    df["file-datetime"] = df["filename"].apply(
        lambda x: str(x).split(".")[0].rsplit("_", 1)[-1]
    )
    df["file-timestamp"] = df["file-datetime"].apply(lambda x: pd_to_datetime_compat(x))
    df = df.reset_index(drop=True)
    return df


def _filter_to_latest_files(
    files_df: pd.DataFrame,
    include_delta_files: bool = False,
) -> pd.DataFrame:
    if include_delta_files:
        raise NotImplementedError(
            "Filtering to latest files including delta files is not implemented."
        )

    if files_df.empty:
        return files_df

    if not include_delta_files:
        files_df = files_df[~files_df["filename"].str.contains("_DELTA")].copy()

    # Filter to rows where file-timestamp == per-dataset max
    latest_mask = files_df["file-timestamp"].eq(
        files_df.groupby("dataset")["file-timestamp"].transform("max")
    )

    latest_files = (
        files_df.loc[latest_mask]
        .sort_values(["dataset", "file-timestamp", "filename"])
        .reset_index(drop=True)
    )

    return latest_files


def lazy_load_from_parquets(
    files_dir: Union[str, Path],
    file_format: str = "parquet",
    tickers: Optional[List[str]] = None,
    cids: Optional[List[str]] = None,
    xcats: Optional[List[str]] = None,
    metrics: Optional[List[str]] = None,
    start_date: Optional[Union[str, pd.Timestamp]] = None,
    end_date: Optional[Union[str, pd.Timestamp]] = None,
    dataframe_format: str = "qdf",
    dataframe_type: str = "pandas",
    categorical_dataframe: bool = True,
    datasets: Optional[List[str]] = None,
    include_delta_files: bool = False,
    include_metadata_files: bool = False,
) -> pd.DataFrame:
    files_dir = Path(files_dir)
    if (not metrics) or (metrics == "all") or ("all" in metrics):
        metrics = JPMAQS_METRICS

    _check_lazy_load_inputs(
        files_dir,
        file_format,
        tickers,
        cids,
        xcats,
        metrics,
        start_date,
        end_date,
        dataframe_format,
        dataframe_type,
        categorical_dataframe,
    )

    available_files_df: pd.DataFrame = _downloaded_files_df(
        files_dir=files_dir,
        file_format=file_format,
        include_metadata_files=include_metadata_files,
    )
    available_files_df: pd.DataFrame = _filter_to_latest_files(
        files_df=available_files_df,
        include_delta_files=include_delta_files,
    )
    if datasets:
        available_files_df = available_files_df.loc[
            available_files_df["dataset"].isin(datasets)
        ]

    tickers = tickers or []
    if cids:
        tickers += [f"{c}_{x}" for c in cids for x in xcats]

    lf: pl.LazyFrame = _lazy_load_filtered_parquets(
        paths=sorted(available_files_df["path"]),
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        return_qdf=(dataframe_format == "qdf"),
    )
    if metrics and set(metrics) != set(JPMAQS_METRICS):
        cols_to_keep = ["real_date", "cid", "xcat", "ticker"] + metrics
        if PYTHON_3_8_OR_LATER:
            lf = lf.select(
                [pl.col(c) for c in cols_to_keep if c in lf.collect_schema().names()]
            )
        else:
            lf = lf.select([pl.col(c) for c in cols_to_keep if c in lf.schema.keys()])
    if dataframe_type == "polars-lazy":
        return lf

    cat_cols = ["cid", "xcat", "ticker"]
    if dataframe_type == "polars":
        if categorical_dataframe:
            cols = None
            if PYTHON_3_8_OR_LATER:
                cols = [c for c in cat_cols if c in lf.collect_schema().names()]
            else:
                cols = [c for c in cat_cols if c in lf.schema.keys()]
            if cols:
                lf = lf.with_columns([pl.col(c).cast(pl.Categorical) for c in cols])
        return lf.collect()
    if dataframe_type == "pandas":
        df = lf.collect().to_pandas()
        if categorical_dataframe:
            cols = [c for c in cat_cols if c in df.columns]
            if cols:
                df[cols] = df[cols].astype("category")
        return df

    raise ValueError("Unknown dataframe type")


class JPMaQSParquetSchemaKind(Enum):
    TICKER = "ticker"
    QDF = "qdf"


def _identify_schema_type(lf: pl.LazyFrame) -> JPMaQSParquetSchemaKind:
    if PYTHON_3_8_OR_LATER:
        cols = set(lf.collect_schema().keys())
    else:
        cols = set(lf.schema.keys())
    if "ticker" in cols:
        return JPMaQSParquetSchemaKind.TICKER
    if {"cid", "xcat"}.issubset(cols):
        return JPMaQSParquetSchemaKind.QDF
    raise ValueError(
        "Unknown schema: need either 'ticker' or both 'cid' and 'xcat'. "
        f"Found columns: {sorted(cols)}"
    )


def _expr_split_ticker(ticker_expr: pl.Expr) -> Tuple[pl.Expr, pl.Expr]:
    """
    Robust split of 'CID_XCAT...' into (cid, xcat) WITHOUT using splitn().
    Works across Polars versions (avoids struct vs list return type issues).
    """
    splitx = ticker_expr.str.splitn("_", 2)
    cid = splitx.struct.field("field_0")
    xcat = splitx.struct.field("field_1")
    return cid, xcat


def _ensure_columns(lf: pl.LazyFrame, cols: Sequence[str]) -> pl.LazyFrame:
    """
    Ensure all `cols` exist before .select(...).
    This runs schema-only (lf.collect_schema()), not a materialization.
    """
    if PYTHON_3_8_OR_LATER:
        have = set(lf.collect_schema().keys())
    else:
        have = set(lf.schema.keys())
    missing = [c for c in cols if c not in have]
    return lf.with_columns(**{c: pl.lit(None) for c in missing}) if missing else lf


def _filter_lazy_frame_by_tickers(
    lf: pl.LazyFrame,
    kind: JPMaQSParquetSchemaKind,
    tickers: Sequence[str],
    start_date: Optional[Union[str, pd.Timestamp]],
    end_date: Optional[Union[str, pd.Timestamp]],
) -> pl.LazyFrame:
    tickers_list = [t for t in tickers if t]
    if kind is JPMaQSParquetSchemaKind.TICKER:
        return lf.filter(pl.col("ticker").is_in(tickers_list))
    lf = (
        lf.with_columns(
            _ticker=pl.concat_str([pl.col("cid"), pl.lit("_"), pl.col("xcat")])
        )
        .filter(pl.col("_ticker").is_in(tickers_list))
        .drop("_ticker")
    )
    if start_date:
        start_date = pd_to_datetime_compat(start_date).strftime("%Y-%m-%d")
        lf = lf.filter(pl.col("real_date") >= pl.lit(start_date).str.to_date())
    if end_date:
        end_date = pd_to_datetime_compat(end_date).strftime("%Y-%m-%d")
        lf = lf.filter(pl.col("real_date") <= pl.lit(end_date).str.to_date())
    return lf


def _to_output_schema(
    lf: pl.LazyFrame, src_kind: JPMaQSParquetSchemaKind, want_qdf: bool
) -> pl.LazyFrame:
    """Normalize columns to qdf or ticker-based shape."""
    cols = "real_date.ticker.value.eop_lag.mop_lag.grading.last_updated"
    ticker_cols = cols.split(".")
    qdf_cols = cols.replace("ticker", "cid.xcat").split(".")

    if want_qdf:
        if src_kind is JPMaQSParquetSchemaKind.TICKER:
            cid_expr, xcat_expr = _expr_split_ticker(pl.col("ticker"))
            lf = lf.with_columns(cid=cid_expr, xcat=xcat_expr)
        lf = _ensure_columns(lf, qdf_cols)
        return lf.select(qdf_cols)

    if src_kind is JPMaQSParquetSchemaKind.QDF:
        lf = lf.with_columns(
            ticker=pl.concat_str([pl.col("cid"), pl.lit("_"), pl.col("xcat")])
        )
    lf = _ensure_columns(lf, ticker_cols)
    return lf.select(ticker_cols)


def _scan_and_prepare_single_parquet(
    path: str,
    tickers: Sequence[str],
    start_date: Optional[Union[str, pd.Timestamp]],
    end_date: Optional[Union[str, pd.Timestamp]],
    return_qdf: bool,
) -> pl.LazyFrame:
    lf = pl.scan_parquet(path)
    kind = _identify_schema_type(lf)
    lf = _filter_lazy_frame_by_tickers(
        lf=lf,
        kind=kind,
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
    )
    lf = _to_output_schema(lf, kind, return_qdf)
    return lf


def _lazy_load_filtered_parquets(
    paths: List[str],
    tickers: List[str],
    start_date: Optional[Union[str, pd.Timestamp]],
    end_date: Optional[Union[str, pd.Timestamp]],
    return_qdf: bool = True,
) -> pl.LazyFrame:
    if not paths:
        raise ValueError("No paths provided")

    tickers_list: List[str] = list(dict.fromkeys(tickers))

    lazy_parts: List[pl.LazyFrame] = [
        _scan_and_prepare_single_parquet(
            path=p,
            tickers=tickers_list,
            start_date=start_date,
            end_date=end_date,
            return_qdf=return_qdf,
        )
        for p in paths
    ]

    out = pl.concat(lazy_parts, how="vertical")
    return out


if __name__ == "__main__":
    print("Current time UTC:", pd.Timestamp.utcnow().isoformat())

    start = time.time()
    since_datetime = pd.Timestamp.now() - pd.offsets.BDay(5)
    print(
        f"Downloading full-snapshots, delta-files, and metadata files published since {since_datetime}"
    )
    since_datetime = since_datetime.strftime("%Y%m%d")
    with DataQueryFileAPIClient(out_dir="./data/jpmaqs-data/") as dq:
        dq.download_catalog_file()
        dq.download_full_snapshot(since_datetime=since_datetime)
    end = time.time()

    print(f"Download completed in {end - start:.2f} seconds")

    cids = ["AUD", "BRL", "CAD", "CHF", "CNY", "CZK", "EUR", "GBP", "USD"]
    xcats = ["RIR_NSA", "FXXR_NSA", "FXXR_VT10", "DU05YXR_NSA", "DU05YXR_VT10"]
    tickers = [f"{c}_{x}" for c in cids for x in xcats]

    with DataQueryFileAPIClient(out_dir="./data/jpmaqs-data/") as dq:
        df = dq.download(tickers=tickers)
        print(df.head())

    with DataQueryFileAPIClient(out_dir="./data/jpmaqs-data/") as dq:
        pl_df: pl.DataFrame = dq.download(
            cids=cids,
            xcats=xcats,
            dataframe_format="tickers",
            dataframe_type="polars",
        )
        print(pl_df.head())
