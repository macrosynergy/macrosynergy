"""

.. note::
    This functionality is currently in BETA and is subject to significant changes
    without deprecation cycles.

Client for downloading JPMaQS data files from the JPMorgan DataQuery File API.

This module provides the `DataQueryFileAPIClient`, a high-level wrapper for the
JPMorgan DataQuery File API.

The client maintains a local on-disk cache of downloaded files. By default, downloads
are written into a folder called `jpmaqs-download`. If `out_dir` is not already named
`jpmaqs-download`, the client will create/use `<out_dir>/jpmaqs-download`.

Setting up API Credentials
--------------------------

Please obtain your DataQuery API credentials (Client ID and Client Secret) from the
JPMorgan Developer Portal, as directed by DataQuery/JPMaQS support channels.
Before using the client, ensure your API credentials are set as environment variables:

.. code-block:: bash

    export DQ_CLIENT_ID="your_client_id"
    export DQ_CLIENT_SECRET="your_client_secret"

In line with the DataQuery SDK, the client will also check for the environment
variables `DATAQUERY_CLIENT_ID` and `DATAQUERY_CLIENT_SECRET`.
Please refer to the official documentation and IT/Tech support channels for
guidance on setting environment variables securely.

Common usage examples
---------------------

**Example 1: Initialize the client and list all available JPMaQS files.**

.. code-block:: python

    from macrosynergy.download import DataQueryFileAPIClient
    import pandas as pd

    client = DataQueryFileAPIClient()

    # Fetch a DataFrame of all available files for the JPMaQS group
    available_files_df = client.list_available_files_for_all_file_groups()
    print("Available JPMaQS files:")
    print(available_files_df.head())

.. code-block:: python

    Available JPMaQS files:
    group-id                    file-group-id             file-datetime  is-available                                         file-name             last-modified
    0   JPMAQS  JPMAQS_STYLIZED_TRADING_FACTORS 2026-01-23 00:00:00+00:00          True  JPMAQS_STYLIZED_TRADING_FACTORS_20260123.parquet 2026-01-23 06:13:48+00:00
    1   JPMAQS  JPMAQS_STYLIZED_TRADING_FACTORS 2026-01-22 00:00:00+00:00          True  JPMAQS_STYLIZED_TRADING_FACTORS_20260122.parquet 2026-01-22 06:14:18+00:00
    2   JPMAQS  JPMAQS_STYLIZED_TRADING_FACTORS 2026-01-21 00:00:00+00:00          True  JPMAQS_STYLIZED_TRADING_FACTORS_20260121.parquet 2026-01-21 06:13:11+00:00
    3   JPMAQS  JPMAQS_STYLIZED_TRADING_FACTORS 2026-01-20 00:00:00+00:00          True  JPMAQS_STYLIZED_TRADING_FACTORS_20260120.parquet 2026-01-20 06:08:55+00:00
    4   JPMAQS  JPMAQS_STYLIZED_TRADING_FACTORS 2026-01-19 00:00:00+00:00          True  JPMAQS_STYLIZED_TRADING_FACTORS_20260119.parquet 2026-01-19 06:11:40+00:00


**Example 2: Download all new or updated files for the current day.**

This is the recommended way to get a daily snapshot of all JPMaQS data,
including full datasets, deltas, and metadata.

.. code-block:: python

    from macrosynergy.download import DataQueryFileAPIClient

    # Use a stable local cache directory.
    client = DataQueryFileAPIClient(out_dir="./jpmaqs_data")

    print(f"Downloading today's files to {client.out_dir} ...")
    client.download_full_snapshot()
    print("Download complete.")

**Example 3: Download and load a filtered dataset**

`download()` is the main "one-stop" method: it downloads the necessary snapshot/delta
files into the local cache (unless `skip_download=True`), then loads the requested
timeseries as a DataFrame.


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

**Example 3b: `download()` - ticker schema.**

Use `dataframe_format="tickers"` to keep a `ticker` column (instead of `cid`/`xcat`).
This is useful if you want to pivot to a matrix for modeling.

.. code-block:: python

    from macrosynergy.download import DataQueryFileAPIClient

    tickers = ["USD_RIR_NSA", "EUR_RIR_NSA", "JPY_RIR_NSA"]

    with DataQueryFileAPIClient(out_dir="./jpmaqs_data") as client:
        df = client.download(
            tickers=tickers,
            metrics=["value"],
            start_date="2015-01-01",
            dataframe_format="tickers",
        )
        df_pivot = df.pivot(index="real_date", columns="ticker", values="value")

**Example 3c: `download()` - large pulls with Polars (lazy)**

For large requests, `dataframe_type="polars-lazy"` keeps the result lazy so you can
filter/transform before collecting.

.. code-block:: python

    import pandas as pd
    import polars as pl
    from macrosynergy.download import DataQueryFileAPIClient


    with DataQueryFileAPIClient(out_dir="./jpmaqs_data") as client:
        cat_df = pd.read_parquet(client.download_catalog_file())
        cat_df = cat_df[cat_df["Ticker"].str.startswith(("USD_", "EUR_"))]
        usd_eur_tickers = cat_df["Ticker"].tolist()

        lf = client.download(
            tickers=usd_eur_tickers,
            start_date="2010-01-01",
            metrics=["value", "last_updated"],
            include_file_column=True,
            dataframe_type="polars-lazy",
        )

        # Example: filter further before materializing
        df = lf.filter(pl.col("real_date") >= pl.date(2020, 1, 1)).collect()

**Example 4: Download all new or updated delta-files since a specific date/time.**

.. code-block:: python

    from macrosynergy.download import DataQueryFileAPIClient
    import pandas as pd

    client = DataQueryFileAPIClient(out_dir="./jpmaqs_data")
    since_datetime = (pd.Timestamp.utcnow() - pd.DateOffset(days=10)).strftime("%Y%m%d")

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
    client = DataQueryFileAPIClient(out_dir="./jpmaqs_data")
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

**Example 8: Load "notification" metadata (missing updates & revisions).**

JPMaQS publishes daily metadata notification JSON files that summarize:

- Missing updates ("Missing Updates")
- Additional info about missing updates ("Additional information on missing updates")
- Changed historical values ("Changed historical values")

The helpers below download the relevant metadata for the requested date (UTC, business-day
window) if needed, and return the notifications as pandas DataFrames.

.. code-block:: python

    from macrosynergy.download import DataQueryFileAPIClient

    with DataQueryFileAPIClient(out_dir="./jpmaqs_data") as client:
        missing_df = client.get_missing_data_notifications(date="2026-01-19")
        revisions_df = client.get_revisions_notifications(date="2026-01-19")

        print(missing_df.head())
        print(revisions_df.head())


---

Please find below the documentation for the `DataQueryFileAPIClient` and related
classes/methods.
"""

import os
import threading
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
import calendar
import datetime
import requests
from macrosynergy.compat import (
    PD_2_0_OR_LATER,
    PYTHON_3_8_OR_LATER,
    POLARS_0_17_13_OR_EARLIER,
)
from macrosynergy.management.constants import JPMAQS_METRICS
from macrosynergy.download.dataquery import JPMAQS_GROUP_ID
from macrosynergy.download.fusion_interface import (
    request_wrapper,
    request_wrapper_stream_bytes_to_disk,
    cache_decorator,
)
from macrosynergy.download.dataquery import OAUTH_TOKEN_URL
from macrosynergy.download.exceptions import DownloadError, InvalidResponseError
from macrosynergy.download.jpm_oauth import JPMorganOAuth


DQ_FILE_API_BASE_URL: str = (
    "https://api-strm-gw01.jpmchase.com/research/dataquery-authe/api/v2"
)
DQ_FILE_API_SCOPE: str = "JPMC:URI:RS-06785-DataQueryExternalApi-PROD"
DQ_FILE_API_TIMEOUT: float = 300.0
DQ_FILE_API_HEADERS_TIMEOUT: float = 60.0
DQ_FILE_API_DELAY_PARAM: float = 0.04  # =1/25 ; 25 transactions per second
DQ_FILE_API_DELAY_MARGIN: float = 1.05  # 5% safety margin
DQ_FILE_API_SEGMENT_SIZE_MB: float = 8.0  # 8 MB
DQ_FILE_API_STREAM_CHUNK_SIZE: int = 8192  # 8 KB


JPMAQS_EARLIEST_FILE_DATE = "20220101"

JPMAQS_DATASET_THEME_MAPPING = {
    "Economic surprises": "JPMAQS_ECONOMIC_SURPRISES",
    "Financial conditions": "JPMAQS_FINANCIAL_CONDITIONS",
    "Generic returns": "JPMAQS_GENERIC_RETURNS",
    "Macroeconomic balance sheets": "JPMAQS_MACROECONOMIC_BALANCE_SHEETS",
    "Macroeconomic trends": "JPMAQS_MACROECONOMIC_TRENDS",
    "Shocks and risk measures": "JPMAQS_SHOCKS_RISK_MEASURES",
    "Stylized trading factors": "JPMAQS_STYLIZED_TRADING_FACTORS",
}

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


class RateLimitedRequester:
    """
    Provides a thread-safe rate-limiting mechanism for API requests.
    """

    def __init__(self, api_delay: float):
        if api_delay < 0:
            raise ValueError("`api_delay` must be non-negative.")
        self._api_delay = api_delay
        self._rate_limit_lock = threading.Lock()
        self._last_api_call: Optional[datetime.datetime] = None

    def _wait_for_api_call(self, api_delay: Optional[float] = None) -> bool:
        """
        Blocks until the required delay since the last API call has passed.
        """
        delay = self._api_delay if api_delay is None else api_delay
        if delay <= 0:
            return True
        with self._rate_limit_lock:
            now = datetime.datetime.now()
            if self._last_api_call is None:
                self._last_api_call = now
                return True

            diff = (now - self._last_api_call).total_seconds()
            sleep_for = delay - diff
            if sleep_for > 0:
                if sleep_for > 1:
                    logger.info(
                        f"Sleeping for {sleep_for:.2f} seconds for API rate limit."
                    )
                time.sleep(sleep_for)

            self._last_api_call = datetime.datetime.now()
        return True


class DataQueryFileAPIClient(RateLimitedRequester):
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
        Base output directory for downloads. The effective cache directory is always a
        folder named `jpmaqs-download` (either `out_dir` itself, or
        `<out_dir>/jpmaqs-download`). A client instance is bound to this directory.
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
        api_delay: float = DQ_FILE_API_DELAY_PARAM,
        api_delay_margin: float = DQ_FILE_API_DELAY_MARGIN,
    ):
        super().__init__(api_delay=api_delay * api_delay_margin)
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
        self.out_dir = self._normalize_out_dir(out_dir or "./jpmaqs-download")

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

    @staticmethod
    def _normalize_out_dir(out_dir: Union[str, Path]) -> str:
        """
        Normalize an output directory to the effective JPMaQS cache directory.

        The DataQuery File API client stores all downloads under a folder called
        `jpmaqs-download`. If `out_dir` is not already named `jpmaqs-download`, this
        method appends a `jpmaqs-download` subdirectory.
        """
        out_dir_str = os.fspath(out_dir)
        stripped = out_dir_str.rstrip("/\\")
        if os.path.basename(stripped) == "jpmaqs-download":
            return stripped
        return str(Path(stripped) / "jpmaqs-download")

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
                self._wait_for_api_call()
                return request_wrapper(
                    method="GET",
                    url=url,
                    headers=headers,
                    params=params or {},
                    proxies=self.proxies,
                    as_json=True,
                    api_delay=0,  # handled by self._wait_for_api_call
                    skip_wait=True,
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

    @cache_decorator(ttl=60)
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

    @cache_decorator(ttl=60)
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
            The start of the time window (inclusive). Format "YYYYMMDD", "YYYYMMDDTHHMMSS",
            or an ISO 8601 datetime string (for example "YYYY-MM-DDTHH:MM:SSZ").
            Defaults to the start of the current day (UTC).
        to_datetime : Optional[str]
            The end of the time window (inclusive). Uses the same formats as `since_datetime`.
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
        files_df = files_df[files_df["file-datetime"].between(since_ts, to_ts)]
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
        if filename:
            try:
                file_group_id, _fdt_with_ext = Path(filename).name.rsplit("_", 1)
                file_datetime = _fdt_with_ext.split(".")[0]
            except ValueError as e:
                raise ValueError(f"Invalid filename format: {filename}") from e
        endpoint = "/group/file/availability"
        params = {"file-group-id": file_group_id, "file-datetime": file_datetime}
        payload = self._get(endpoint, params)
        return pd.json_normalize(payload)

    def download_file(
        self,
        file_group_id: str = None,
        file_datetime: str = None,
        filename: Optional[str] = None,
        overwrite: bool = False,
        chunk_size: Optional[int] = None,
        timeout: Optional[float] = DQ_FILE_API_TIMEOUT,
        max_retries: int = 3,
    ) -> str:
        """
        Downloads a single Parquet file to the client's output directory.

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
        overwrite : bool
            If True, overwrites the file if it already exists. Default is False.
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
        file_path = Path(self.out_dir) / Path(file_date) / Path(file_name)

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
            verify_ssl=self.verify_ssl,
        )

        is_small_file = any(x in file_group_id.lower() for x in ["delta", "metadata"])
        if "_DELTA" in file_group_id:
            is_small_file = file_datetime not in _large_delta_file_datetimes()

        if is_small_file:
            self._wait_for_api_call()
            request_wrapper_stream_bytes_to_disk(
                **download_args,
                api_delay=0,
                skip_wait=True,
            )
        else:
            SegmentedFileDownloader(
                **download_args,
                api_delay=DQ_FILE_API_DELAY_PARAM,
                api_delay_margin=DQ_FILE_API_DELAY_MARGIN,
                parent_requester=self,
                max_file_retries=max_retries,
                start_download=True,
            )

        time_taken = time.time() - start
        logger.info(
            f"Downloaded {file_name} in {time_taken:.2f} seconds to {file_path}"
        )
        return str(file_path)

    def delete_corrupt_files(
        self,
        files: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Deletes corrupt files from the provided list based on file integrity checks.

        Parameters
        ----------
        files : Optional[List[str]]
            A list of file names (as in `list_downloaded_files()["file-name"]`) to check
            for corruption. If None, scans all downloaded files in the client's output
            directory.

        Returns
        -------
        List[str]
            A list of file paths that were identified as corrupt and deleted.
        """
        avail_files = self.list_downloaded_files()
        if avail_files.empty:
            return []
        if files is not None:
            if not all(isinstance(f, str) for f in files):
                raise ValueError(
                    "All items in `files` must be strings representing file names."
                )
            avail_files = avail_files[avail_files["file-name"].isin(files)]
        files = sorted(set(map(str, avail_files["path"])))
        extensions = sorted(set(Path(f).suffix.rsplit(".", 1)[-1] for f in files))

        return _delete_corrupt_files(
            files=files, extensions=extensions, root_dir=Path(self.out_dir)
        )

    def cleanup_old_files(
        self,
        days_to_keep: int = 5,
    ) -> List[str]:
        """
        Deletes files older than the specified number of days from the output directory.

        Parameters
        ----------
        days_to_keep : int
            The number of days to retain files. This is measured from the latest file date
            within each file group. Files older than this threshold will be deleted.

        Returns
        -------
        List[str]
            A list of file paths that were deleted.
        """
        if not isinstance(days_to_keep, int):
            raise ValueError("`days_to_keep` must be a non-negative integer.")
        if days_to_keep < 0:
            logger.warning(
                "`days_to_keep` is negative; it will be treated as the absolute value."
                f" ({days_to_keep} -> {abs(days_to_keep)})"
            )
            days_to_keep = abs(days_to_keep)
        if days_to_keep == 0:
            return []
        found_files = self.list_downloaded_files()
        if found_files.empty:
            return []
        dataset_col = "file-group-id"
        if dataset_col not in found_files.columns:
            dataset_col = "dataset"
        if dataset_col not in found_files.columns:
            raise InvalidResponseError(
                'Missing expected column "dataset" or "file-group-id" in downloaded files listing.'
            )
        fg_dt_mapping: Dict[str, pd.Timestamp] = (
            found_files.groupby(dataset_col)["file-timestamp"].max().to_dict()
        )
        cutoff_dates = {
            fg: (dt - pd.Timedelta(days=days_to_keep)).normalize()
            for fg, dt in fg_dt_mapping.items()
        }
        files_to_delete = []
        deleted_files: List[str] = []
        for _, row in found_files.iterrows():
            fg = row[dataset_col]
            fdt = row["file-timestamp"]
            if fdt < cutoff_dates[fg]:
                files_to_delete.append(str(row["path"]))

        for file_path in files_to_delete:
            try:
                Path(file_path).unlink()
                deleted_files.append(file_path)
                logger.info(f"Deleted old file: {file_path}")
            except Exception as e:
                logger.error(f"Failed to delete file {file_path}: {e}")
        return sorted(deleted_files)

    def download_multiple_files(
        self,
        filenames: List[str],
        overwrite: bool = False,
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
        overwrite : bool
            If True, overwrites files if they already exist. Default is False.
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
        Path(self.out_dir).mkdir(parents=True, exist_ok=True)
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
                        overwrite=overwrite,
                        chunk_size=chunk_size,
                        timeout=timeout,
                    )
                ] = filename

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
        found_corrupt_files = self.delete_corrupt_files(files=filenames)
        corrupt_filenames = [Path(p).name for p in found_corrupt_files]
        failed_files = sorted(set(failed_files + corrupt_filenames))
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
            max_retries=max_retries - 1,
            n_jobs=n_jobs,
            chunk_size=chunk_size,
            timeout=timeout,
            show_progress=show_progress,
        )

    def download_catalog_file(
        self,
        overwrite: bool = False,
        timeout: Optional[float] = DQ_FILE_API_TIMEOUT,
    ) -> str:
        # check if file already exists
        file_path = None
        existing_files = self.list_downloaded_files()
        if not overwrite and not existing_files.empty:
            todayts = pd.Timestamp.utcnow().strftime("%Y%m%d")
            today_file = f"JPMAQS_METADATA_CATALOG_{todayts}.parquet"
            if today_file in sorted(existing_files["file-name"]):
                file_path = existing_files[existing_files["file-name"] == today_file][
                    "path"
                ].values[0]
                logger.info(f"Catalog file already downloaded: {file_path}")
                return str(file_path)

        available_catalogs = self.list_available_files(self.catalog_file_group_id)
        if available_catalogs.empty:
            raise DownloadError("No catalog files available for download.")
        latest_catalog = available_catalogs.sort_values(
            by=["file-datetime", "last-modified"], ascending=False
        ).iloc[0]
        latest_filename = latest_catalog["file-name"]
        logger.info(f"Latest catalog file identified: {latest_filename}")

        if file_path is None:
            file_path = self.download_file(
                filename=latest_filename,
                overwrite=overwrite,
                timeout=timeout,
            )

        return str(file_path)

    def get_datasets_for_indicators(
        self,
        tickers: Optional[List[str]] = None,
        cids: Optional[List[str]] = None,
        xcats: Optional[List[str]] = None,
        case_sensitive: bool = False,
        catalog_file: Optional[str] = None,
    ) -> List[str]:
        tickers = _construct_all_tickers_list(tickers=tickers, cids=cids, xcats=xcats)
        if not tickers or not any(t.strip() for t in tickers):
            raise ValueError("No valid tickers to search for.")

        catalog_file = catalog_file or self.download_catalog_file()
        catalog_df = pd.read_parquet(catalog_file)
        catalog_df.loc[:, "Dataset"] = catalog_df["Theme"].map(
            JPMAQS_DATASET_THEME_MAPPING
        )
        if not set(catalog_df["Theme"]) == set(JPMAQS_DATASET_THEME_MAPPING.keys()):
            missing_themes = set(catalog_df["Theme"]) - set(
                JPMAQS_DATASET_THEME_MAPPING.keys()
            )
            logger.warning(
                f"Catalog file contains unknown themes: {missing_themes}. "
                "Please check for newer versions of the `macrosynergy` Python package."
            )
        if case_sensitive:
            catalog_df = catalog_df[catalog_df["Ticker"].isin(tickers)]
        else:
            catalog_df = catalog_df[
                catalog_df["Ticker"].str.lower().isin(t.lower() for t in tickers)
            ]

        datasets_to_keep = sorted(set(catalog_df["Dataset"]))
        return datasets_to_keep

    def filter_to_valid_tickers(
        self,
        tickers: List[str],
        case_sensitive: bool = False,
        catalog_file: Optional[str] = None,
    ) -> List[str]:
        """
        Filters a list of tickers to only those that are valid according to the catalog.

        Parameters
        ----------
        tickers : List[str]
            A list of tickers to validate.
        case_sensitive : bool
            If True, performs case-sensitive matching. Default is False.
        """
        if not isinstance(tickers, list) or not all(
            isinstance(x, str) for x in tickers
        ):
            raise ValueError("`tickers` must be a list of strings.")

        catalog_file = catalog_file or self.download_catalog_file()

        catalog_tickers: List[str] = list(
            set(pd.read_parquet(catalog_file)["Ticker"].dropna().astype(str))
        )

        if case_sensitive:
            catalog_set = set(catalog_tickers)
            valid = {t for t in catalog_tickers if t in catalog_set and t in tickers}
            return sorted(valid)

        # case-insensitive - build canonical ticker name lookup
        canonical_map = {t.lower(): t for t in catalog_tickers}

        valid = {
            canonical_map[t.strip().lower()]
            for t in tickers
            if t.strip() and t.strip().lower() in canonical_map
        }

        return sorted(valid)

    def list_downloaded_files(
        self,
        include_last_modified_columns: bool = True,
    ) -> pd.DataFrame:
        col_order = [
            "filename",
            "file-datetime",
            "dataset",
            "filetype",
            "file-timestamp",
            "path",
        ]
        dfs = [
            _downloaded_files_df(
                self.out_dir, file_format=fmt, include_metadata_files=True
            )
            for fmt in ["parquet", "json"]
        ]
        dfs = [df for df in dfs if not df.empty]
        if not dfs:
            return pd.DataFrame(columns=col_order)
        files_df = pd.concat(dfs).reset_index(drop=True)
        if files_df.empty:
            return files_df

        files_df = files_df[col_order].rename(columns={"filename": "file-name"})

        if include_last_modified_columns:
            dq_files_df = self.list_available_files_for_all_file_groups()
            dq_files_df = dq_files_df[
                dq_files_df["file-name"].isin(files_df["file-name"])
            ]
            files_df = files_df.merge(
                dq_files_df[["file-name", "last-modified"]],
                on="file-name",
            )
        return files_df

    def _load_metadata_jsons(
        self,
        date: Optional[Union[pd.Timestamp, str]] = None,
        normalize_headers: bool = True,
        skip_download: bool = False,
    ) -> Dict[str, pd.DataFrame]:
        """Load JPMaQS metadata notification JSONs for a date."""
        date: pd.Timestamp = (
            pd_to_datetime_compat(date) if date is not None else pd.Timestamp.utcnow()
        ).normalize()
        if date > pd.Timestamp.utcnow().normalize():
            new_dt = pd.Timestamp.utcnow().normalize()
            logger.warning(
                f"Provided date {date.date()} is in the future."
                f" Setting date to today: {new_dt.date()}."
            )
            date = new_dt
        if not skip_download:
            to_dt = date + pd.offsets.BDay(1) - pd.Timedelta(seconds=1)
            self.download_full_snapshot(
                since_datetime=date,
                to_datetime=to_dt,
                include_full_snapshots=False,
                include_delta=False,
                include_metadata=True,
            )
        df = self.list_downloaded_files()
        df: pd.DataFrame = df[
            (df["dataset"] == "JPMAQS_METADATA_NOTIFICATIONS")
            & df["file-name"].str.lower().str.endswith(".json")
        ]
        date = date.normalize()
        df = df[df["file-timestamp"].dt.normalize() == date]
        if df.empty:
            logger.warning(f"No notification files found for date: {date.date()}")
            return {}
        json_contentts: Dict[str, pd.DataFrame] = {}
        err_str = 'Invalid notification file (missing "sub_title"): '
        title_err_str = "Unexpected notification title in file: "
        expected_titles = [
            "Missing Updates",
            "Changed historical values",
            "Additional information on missing updates",
        ]
        canonical_title_map = {t.upper(): t for t in expected_titles}
        for jp in df["path"].apply(str).tolist():
            _json = {}
            with open(jp, "r", encoding="utf-8") as f:
                _json: Dict[str, dict] = json.load(f)
            if _json.get("metadata", {}).get("sub_title", None) is None:
                logger.warning(err_str + jp)
                continue
            j_title: str = _json["metadata"]["sub_title"]
            if j_title.upper() not in map(str.upper, expected_titles):
                logger.warning(title_err_str + jp)
                continue
            canonical_title = canonical_title_map[j_title.upper()]
            json_contentts[canonical_title] = pd.json_normalize(
                _json, record_path=["data"]
            )

        if normalize_headers:
            for key in json_contentts:
                new_cols = [
                    _col.replace(" ", "_")
                    .replace("-", "_")
                    .replace("(%)", "pct")
                    .lower()
                    for _col in json_contentts[key].columns
                ]
                json_contentts[key].columns = new_cols

        return json_contentts

    def get_revisions_notifications(
        self,
        date: Optional[Union[pd.Timestamp, str]] = None,
        normalize_headers: bool = True,
    ) -> pd.DataFrame:
        """
        Return "Changed historical values" notifications for a given date.

        This loads daily JPMaQS metadata notification JSON(s) for the requested date
        and returns the table describing historical revisions. If no matching
        notification file(s) are found, an empty DataFrame is returned.

        Parameters
        ----------
        date : Optional[Union[pd.Timestamp, str]]
            Target date (UTC). Strings can be "YYYY-MM-DD", "YYYYMMDD", or ISO 8601.
            Defaults to today (UTC).
        normalize_headers : bool
            If True, normalizes column names to lowercase snake_case and converts
            "(%)" to "pct". Defaults to True.

        Returns
        -------
        pd.DataFrame
            A DataFrame of revision notifications. Empty if none are found.
        """
        jsons = self._load_metadata_jsons(
            date=date, normalize_headers=normalize_headers
        )
        if "Changed historical values" not in jsons:
            logger.warning("No `Changed historical values` notifications found.")
            return pd.DataFrame()
        return jsons["Changed historical values"]

    def get_missing_data_notifications(
        self,
        date: Optional[Union[pd.Timestamp, str]] = None,
        normalize_headers: bool = True,
    ) -> pd.DataFrame:
        """
        Return missing-update notifications (with optional additional information).

        This loads daily JPMaQS metadata notification JSON(s) for the requested date.
        It returns:

        - "Missing Updates" rows
        - left-joined with "Additional information on missing updates" when available

        If only one of the two tables is available, that table is returned. If
        neither is available, an empty DataFrame is returned.

        Parameters
        ----------
        date : Optional[Union[pd.Timestamp, str]]
            Target date (UTC). Strings can be "YYYY-MM-DD", "YYYYMMDD", or ISO 8601.
            Defaults to today (UTC).
        normalize_headers : bool
            If True, normalizes column names to lowercase snake_case and converts
            "(%)" to "pct". Defaults to True.

        Returns
        -------
        pd.DataFrame
            A DataFrame of missing-update notifications (optionally enriched).
        """
        jsons = self._load_metadata_jsons(
            date=date, normalize_headers=normalize_headers
        )
        df1 = jsons.get("Missing Updates", pd.DataFrame())
        df2 = jsons.get("Additional information on missing updates", pd.DataFrame())

        if df1.empty and df2.empty:
            logger.warning("No `Missing Updates` or related notifications found.")
            return pd.DataFrame()
        if df2.empty:
            logger.warning(
                "No `Additional information on missing updates` notifications found."
            )
            return df1
        if df1.empty:
            logger.warning("No `Missing Updates` notifications found.")
            return df2

        left_join_key = None
        if "Ticker" in df1.columns and "ticker" in df2.columns:
            df1 = df1.rename(columns={"Ticker": "ticker"})
        elif "ticker" in df1.columns and "Ticker" in df2.columns:
            df2 = df2.rename(columns={"Ticker": "ticker"})

        for candidate in ("Ticker", "ticker"):
            if candidate in df1.columns and candidate in df2.columns:
                left_join_key = candidate
                break
        if left_join_key is None:
            raise KeyError(
                'Expected a common join key ("Ticker" or "ticker") in notification data.'
            )

        df1 = (
            df1.merge(df2, how="left", on=left_join_key)
            .sort_values(by=left_join_key, ascending=True)
            .reset_index(drop=True)
        )
        return df1

    def download_full_snapshot(
        self,
        since_datetime: Optional[str] = None,
        to_datetime: Optional[str] = None,
        overwrite: bool = False,
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
        since_datetime : Optional[str]
            Download files modified since this timestamp (inclusive).
            Defaults to the start of the current day (UTC).
        to_datetime : Optional[str]
            Download files modified up to this timestamp (inclusive).
            Note: `since_datetime` and `to_datetime` only affect which files are downloaded.
            Loading uses all locally available cached snapshot/delta files.
        overwrite : bool
            If True, overwrites files if they already exist. Default is False.
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
        Path(self.out_dir).mkdir(parents=True, exist_ok=True)
        start_time = time.time()

        if since_datetime is None:
            # JPMaQS data files are not published on weekends, so "today" can yield
            # no snapshot/delta files even though the catalog is daily.
            if include_full_snapshots or include_delta:
                since_datetime = get_current_or_last_business_day().strftime("%Y%m%d")
            else:
                since_datetime = pd.Timestamp.utcnow().strftime("%Y%m%d")

        logger.info(
            f"Starting snapshot download to '{self.out_dir}' for files since {since_datetime}."
        )

        validate_dq_timestamp(since_datetime, var_name="since_datetime")

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

        downloaded_files_df = self.list_downloaded_files()
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
            overwrite=overwrite,
            chunk_size=chunk_size,
            timeout=timeout,
            show_progress=show_progress,
        )

        total_time = time.time() - start_time
        logger.info(f"Snapshot download completed in {total_time:.2f} seconds.")

    def load_data(
        self,
        tickers: Optional[List[str]] = None,
        cids: Optional[List[str]] = None,
        xcats: Optional[List[str]] = None,
        metrics: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        min_last_updated: Optional[Union[str, pd.Timestamp]] = None,
        max_last_updated: Optional[Union[str, pd.Timestamp]] = None,
        include_file_column: bool = False,
        dataframe_format: str = "qdf",
        dataframe_type: str = "pandas",
        categorical_dataframe: bool = True,
        include_delta_files: bool = True,
        delta_treatment: str = "latest",
        since_datetime: Optional[str] = None,
        to_datetime: Optional[str] = None,
        catalog_file: Optional[str] = None,
        datasets: Optional[List[str]] = None,
    ) -> Union[pd.DataFrame, pl.DataFrame, pl.LazyFrame]:
        """
        Load JPMaQS timeseries from the local cache for the requested selection.

        This method performs the "load" part of `download()`: it resolves tickers to the
        underlying JPMaQS datasets (using the catalog file) and returns the filtered data
        from locally cached snapshot/delta parquet files.

        Unlike `download()`, this method does **not** download snapshot/delta/metadata
        files. It assumes the relevant files are already present in `out_dir`.
        The catalog file is still downloaded/validated unless `catalog_file` is provided.

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
            are also defined in `macrosynergy.management.constants.JPMAQS_METRICS`. The default
            is None, in which case all metrics are returned.
        start_date : Optional[str]
            The start date for the returned data in "YYYY-MM-DD" (or "YYYYMMDD") format.
            If None, data is returned from the earliest available date.
        end_date : Optional[str]
            The end date for the returned data in "YYYY-MM-DD" (or "YYYYMMDD") format.
            If None, data is returned up to the latest available date.
        min_last_updated : Optional[Union[str, pd.Timestamp]]
            If provided, only data points with `last_updated` on or after this timestamp
            are returned. Strings can be "YYYY-MM-DDThh:mm:ss", "YYYYMMDDhhmmss", or
            ISO 8601 format.
        max_last_updated : Optional[Union[str, pd.Timestamp]]
            If provided, only data points with `last_updated` on or before this timestamp
            are returned. Strings can be "YYYY-MM-DDThh:mm:ss", "YYYYMMDDhhmmss", or
            ISO 8601 format.
        include_file_column : bool
            If True, includes a column indicating the source file for each data point.
            Default is False.
        dataframe_format : str
            The output schema. Options are:

            - "qdf": quantamental schema with `cid` and `xcat` columns.
            - "tickers": ticker schema with a single `ticker` column (instead of `cid`/`xcat`).

            Note: if you want a wide matrix (date x ticker), pivot the returned data
            using pandas/Polars. Default is "qdf".
        dataframe_type : str
            The type of DataFrame to return. Options are "pandas" for a pandas DataFrame,
            "polars" for a polars DataFrame, or "polars-lazy" for a polars LazyFrame.
            Default is "pandas".
        categorical_dataframe : bool
            If True and `dataframe_type` is "pandas" (or "polars"/"polars-lazy" with
            compatible Polars versions), converts selected string columns to categorical
            dtype. Default is True.
        include_delta_files : bool
            If True, includes delta files in the load process (recommended).
            Default is True.
        delta_treatment : str
            Determines how to treat duplicate values between snapshots and deltas. Options are:

            - "latest": keep the latest value per series/date.
            - "earliest": keep the earliest value per series/date.
            - "all": keep all entries.

            Default is "latest".
        since_datetime : Optional[str]
            Restrict which locally available snapshot/delta files are considered to those
            modified since this timestamp (inclusive). If None, all locally available files
            are considered.
        to_datetime : Optional[str]
            Restrict which locally available snapshot/delta files are considered to those
            modified up to this timestamp (inclusive). If None, all locally available files
            are considered.

            Notes:
            - If `to_datetime` is provided and `max_last_updated` is not, the loader
              defaults `max_last_updated` to `to_datetime` (interpreting date-only strings
              as end-of-day). This is important for monthly delta regimes where the
              covering delta file can be timestamped at month-end (after `to_datetime`),
              and row-level filtering by `last_updated` is needed to honor the requested
              data.
        catalog_file : Optional[str]
            Optional path to a local JPMaQS catalog parquet file. If not provided, the
            client will download/validate the latest catalog file for ticker resolution.
        datasets : Optional[List[str]]
            Optional list of JPMaQS datasets (file-group IDs) to restrict which locally cached
            snapshot/delta parquet files are scanned/loaded. If not provided, datasets are
            inferred from the requested tickers using the catalog.

        Returns
        -------
        Union[pd.DataFrame, pl.DataFrame, pl.LazyFrame]
            A DataFrame containing the requested data.
        """
        catalog_file = catalog_file or self.download_catalog_file()

        rqstd_tickers = _construct_all_tickers_list(
            tickers=tickers, cids=cids, xcats=xcats
        )
        if not bool(rqstd_tickers):
            raise ValueError(
                "At least one ticker must be specified via `tickers`, or `cids` & `xcats`."
            )

        valid_tickers = self.filter_to_valid_tickers(
            tickers=rqstd_tickers, catalog_file=catalog_file
        )
        valid_norm = {t.lower() for t in valid_tickers}
        missing = sorted({t for t in rqstd_tickers if t.lower() not in valid_norm})
        if not valid_tickers:
            raise ValueError(
                "No valid tickers found with the provided `tickers`, `cids`, and `xcats`."
            )
        if missing:
            lmiss = min(5, len(missing))
            nmore = f"{len(missing) - lmiss} more" if len(missing) > lmiss else ""
            miss_str = "[" + ", ".join(missing[:lmiss]) + "..." + nmore + "]"
            miss_str = f"{len(missing)} tickers requested do not exist in the catalog, these are: {miss_str}"
            logger.warning(miss_str)

        rqstd_tickers = valid_tickers

        datasets_to_download = datasets
        if datasets_to_download is None:
            datasets_to_download = self.get_datasets_for_indicators(
                tickers=rqstd_tickers, catalog_file=catalog_file
            )
            if datasets_to_download and include_delta_files:
                datasets_to_download += [f"{ds}_DELTA" for ds in datasets_to_download]

        effective_max_last_updated = max_last_updated
        if to_datetime is not None and max_last_updated is None:
            effective_max_last_updated = _normalize_last_updated_cutoff(to_datetime)

        warn_if_no_full_snapshots = since_datetime is not None
        return lazy_load_from_parquets(
            files_dir=self.out_dir,
            tickers=rqstd_tickers,
            metrics=metrics,
            start_date=start_date,
            end_date=end_date,
            min_last_updated=min_last_updated,
            max_last_updated=effective_max_last_updated,
            include_delta_files=include_delta_files,
            delta_treatment=delta_treatment,
            dataframe_format=dataframe_format,
            dataframe_type=dataframe_type,
            categorical_dataframe=categorical_dataframe,
            datasets=datasets_to_download,
            include_file_column=include_file_column,
            catalog_file=catalog_file,
            warn_if_no_full_snapshots=warn_if_no_full_snapshots,
            since_datetime=since_datetime,
            to_datetime=to_datetime,
        )

    def _get_effective_snapshot_switchover_ts(
        self, datasets: List[str]
    ) -> Optional[pd.Timestamp]:
        """
        Return the effective (per-request) earliest full-snapshot timestamp.

        Notes
        -----
        JPMaQS can remove older full snapshots over time. For a given set of datasets
        we define the "switchover" as the *latest* of the datasets' earliest currently
        available full snapshots. If any dataset has no full snapshots at all, returns
        None.
        """
        if not datasets:
            return None

        # Datasets are passed around as file-group-ids; treat *_DELTA as updates to the
        # base dataset. Exclude metadata-only datasets.
        base_datasets = sorted(
            {
                str(d).replace("_DELTA", "")
                for d in datasets
                if isinstance(d, str) and d and ("_METADATA" not in d.upper())
            }
        )
        base_datasets = [d for d in base_datasets if d != self.catalog_file_group_id]
        if not base_datasets:
            return None

        earliest_by_dataset: List[pd.Timestamp] = []
        for ds in base_datasets:
            df = self.list_available_files(file_group_id=ds)
            if df.empty:
                return None
            if "file-datetime" not in df.columns:
                raise InvalidResponseError(
                    f'Missing "file-datetime" in available-files response for {ds}'
                )
            earliest_by_dataset.append(df["file-datetime"].min())

        return max(earliest_by_dataset) if earliest_by_dataset else None

    def download(
        self,
        tickers: Optional[List[str]] = None,
        cids: Optional[List[str]] = None,
        xcats: Optional[List[str]] = None,
        metrics: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        min_last_updated: Optional[Union[str, pd.Timestamp]] = None,
        max_last_updated: Optional[Union[str, pd.Timestamp]] = None,
        include_file_column: bool = False,
        dataframe_format: str = "qdf",
        dataframe_type: str = "pandas",
        categorical_dataframe: bool = True,
        include_delta_files: bool = True,
        include_metadata_files: bool = True,
        delta_treatment: str = "latest",
        show_progress: bool = True,
        overwrite: bool = False,
        since_datetime: Optional[str] = None,
        to_datetime: Optional[str] = None,
        skip_download: bool = False,
        cleanup_old_files_n_days: Optional[int] = 5,
    ) -> Union[pd.DataFrame, pl.DataFrame, pl.LazyFrame]:
        """
        Download JPMaQS files into the local cache and load the requested timeseries.

        This is the main "one-stop" method: it resolves tickers to the underlying JPMaQS
        datasets, downloads the necessary snapshot/delta/metadata files into the local
        cache (unless `skip_download=True`), and returns the filtered data.

        For a "load-only" workflow (no snapshot/delta downloads), call `load_data()`
        directly (or call this method with `skip_download=True`).

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
            are also defined in `macrosynergy.management.constants.JPMAQS_METRICS`. The default
            is None, in which case all metrics are returned.
        start_date : Optional[str]
            The start date for the returned data in "YYYY-MM-DD" (or "YYYYMMDD") format.
            If None, data is returned from the earliest available date.
        end_date : Optional[str]
            The end date for the returned data in "YYYY-MM-DD" (or "YYYYMMDD") format.
            If None, data is returned up to the latest available date.
        min_last_updated : Optional[Union[str, pd.Timestamp]]
            If provided, only data points with `last_updated` on or after this timestamp
            are returned. Strings can be "YYYY-MM-DDThh:mm:ss", "YYYYMMDDhhmmss", or
            ISO 8601 format.
        max_last_updated : Optional[Union[str, pd.Timestamp]]
            If provided, only data points with `last_updated` on or before this timestamp
            are returned. Strings can be "YYYY-MM-DDThh:mm:ss", "YYYYMMDDhhmmss", or
            ISO 8601 format.
        include_file_column : bool
            If True, includes a column indicating the source file for each data point.
            Default is False.
        dataframe_format : str
            The output schema. Options are:

            - "qdf": quantamental schema with `cid` and `xcat` columns.
            - "tickers": ticker schema with a single `ticker` column (instead of `cid`/`xcat`).

            Note: if you want a wide matrix (date x ticker), pivot the returned data
            using pandas/Polars. Default is "qdf".
        dataframe_type : str
            The type of DataFrame to return. Options are "pandas" for a pandas DataFrame,
            "polars" for a polars DataFrame, or "polars-lazy" for a polars LazyFrame.
            Default is "pandas".
        categorical_dataframe : bool
            If True and `dataframe_type` is "pandas", the returned DataFrame will use
            categorical dtypes for object columns. Default is True.
        include_delta_files : bool
            If True, delta files will be downloaded (and applied when loading, via
            `delta_treatment`). Default is True.
        include_metadata_files : bool
            If True, metadata files will be included in the download. Default is True.
        delta_treatment : str
            Specifies how to treat new or updated entries across files from different
            dates (based on `last_updated`). Options are:

            - "latest": keep the latest value per series/date (default).
            - "earliest": keep the earliest value per series/date.
            - "all": keep all entries.
        show_progress : bool
            If True, displays a progress bar during downloads. Default is True.
        overwrite : bool
            If True, overwrites files if they already exist. Default is False.
        since_datetime : Optional[str]
            Download files modified since this timestamp (inclusive).
            Defaults to the start of the current day (UTC).
        to_datetime : Optional[str]
            Download files modified up to this timestamp (inclusive).
            Note: `since_datetime` and `to_datetime` only affect which files are downloaded.
            The returned timeseries is controlled by `start_date`/`end_date`.

            Note for historical ("delta-only") vintages:
            - If `to_datetime` falls within a month where only monthly delta files exist,
              the client may expand the download window to the month-end timestamp so the
              covering delta file is available locally. Row-level filtering is then
              enforced via `max_last_updated` during the load step.
        skip_download : bool
            If True, do not download snapshot/delta/metadata files and only load from the
            local cache. The catalog is still downloaded/validated. Default is False.
        cleanup_old_files_n_days : Optional[int]
            If set to an integer value, deletes files older than this number of days
            from the local cache after the download is complete. This integer value is
            passed to `cleanup_old_files()`. If None, no cleanup is performed.
            Default is 5 (days).

        Returns
        -------
        Union[pd.DataFrame, pl.DataFrame, pl.LazyFrame]
            A DataFrame containing the requested data.
        """
        catalog_file = self.download_catalog_file()

        rqstd_tickers = _construct_all_tickers_list(
            tickers=tickers, cids=cids, xcats=xcats
        )
        if not bool(rqstd_tickers):
            raise ValueError(
                "At least one ticker must be specified via `tickers`, or `cids` & `xcats`."
            )
        else:
            valid_tickers = self.filter_to_valid_tickers(
                tickers=rqstd_tickers, catalog_file=catalog_file
            )
            valid_norm = {t.lower() for t in valid_tickers}
            missing = sorted({t for t in rqstd_tickers if t.lower() not in valid_norm})
            if not valid_tickers:
                raise ValueError(
                    "No valid tickers found with the provided `tickers`, `cids`, and `xcats`."
                )
            if missing:
                lmiss = min(5, len(missing))
                nmore = f"{len(missing) - lmiss} more" if len(missing) > lmiss else ""
                miss_str = "[" + ", ".join(missing[:lmiss]) + "..." + nmore + "]"
                miss_str = f"{len(missing)} tickers requested do not exist in the catalog, these are: {miss_str}"
                logger.warning(miss_str)

            rqstd_tickers = valid_tickers

        datasets_to_download = self.get_datasets_for_indicators(
            tickers=rqstd_tickers, catalog_file=catalog_file
        )
        if datasets_to_download and include_delta_files:
            datasets_to_download += [f"{ds}_DELTA" for ds in datasets_to_download]
        if not skip_download:
            # If the user requests a historical vintage (`to_datetime`) that predates the
            # earliest available full snapshots, JPMaQS data can only be reconstructed by
            # applying the entire delta history since `JPMAQS_EARLIEST_FILE_DATE`.
            requires_full_delta_history = False
            switchover_ts: Optional[pd.Timestamp] = None
            if to_datetime is not None:
                validate_dq_timestamp(to_datetime, var_name="to_datetime")
                to_ts = pd_to_datetime_compat(to_datetime)

                if (
                    isinstance(to_datetime, str)
                    and ("T" not in to_datetime)
                    and (":" not in to_datetime)
                ):
                    to_ts = (
                        to_ts.normalize()
                        + pd.DateOffset(days=1)
                        - pd.Timedelta(nanoseconds=1)
                    )

                earliest_file_ts = pd_to_datetime_compat(JPMAQS_EARLIEST_FILE_DATE)
                if to_ts < earliest_file_ts:
                    raise ValueError(
                        "`to_datetime` is earlier than the earliest supported JPMaQS "
                        f"file date ({JPMAQS_EARLIEST_FILE_DATE})."
                    )

                switchover_ts = self._get_effective_snapshot_switchover_ts(
                    datasets=datasets_to_download
                )
                if switchover_ts is None or to_ts < switchover_ts:
                    requires_full_delta_history = True
                    if not include_delta_files:
                        raise ValueError(
                            "The requested vintage predates the earliest available full "
                            "snapshots, so `include_delta_files` must be True."
                        )
                    logger.info(
                        "No full snapshots are available for the requested vintage "
                        f"(to_datetime={to_datetime}, switchover={switchover_ts}); "
                        f"downloading all delta files since {JPMAQS_EARLIEST_FILE_DATE}."
                    )

            download_since_datetime = (
                since_datetime or get_current_or_last_business_day().strftime("%Y%m%d")
            )
            download_to_datetime = to_datetime
            if requires_full_delta_history and to_datetime is not None:
                # Monthly delta files can be timestamped at month-end, which can be after
                # an in-month `to_datetime`. Expand the download window so the covering
                # month-end delta file is available locally (row-level filtering is done
                # via `max_last_updated` in the load step).
                download_to_datetime = _month_end_dq_timestamp(to_datetime)
                validate_dq_timestamp(download_to_datetime, var_name="to_datetime")
            if to_datetime is not None:
                since_dt = pd_to_datetime_compat(download_since_datetime)
                to_dt = pd_to_datetime_compat(to_datetime)
                if to_dt < since_dt:
                    new_since = (to_dt - pd.offsets.BDay(1)).strftime("%Y%m%d")
                    logger.warning(
                        "`to_datetime` is before `since_datetime`; adjusting "
                        "`since_datetime` to be one business day before `to_datetime`. "
                        f"New `since_datetime`: {new_since}"
                    )
                    download_since_datetime = new_since

            include_full_snapshots = True
            effective_since_datetime_for_load = since_datetime
            if requires_full_delta_history:
                download_since_datetime = JPMAQS_EARLIEST_FILE_DATE
                include_full_snapshots = False
                effective_since_datetime_for_load = None

            self.download_full_snapshot(
                since_datetime=download_since_datetime,
                to_datetime=download_to_datetime,
                file_group_ids=datasets_to_download,
                overwrite=overwrite,
                show_progress=show_progress,
                include_full_snapshots=include_full_snapshots,
                include_delta=include_delta_files,
                include_metadata=include_metadata_files,
            )
            if not isinstance(cleanup_old_files_n_days, (type(None), int)):
                raise ValueError(
                    "`cleanup_old_files_n_days` must be an integer or None."
                )
            if isinstance(cleanup_old_files_n_days, int):
                # `cleanup_old_files()` uses calendar days (Timedelta(days=...)),
                # so `cleanup_old_files_n_days` must be interpreted as calendar days too.
                # Using business-day counts here can lead to under-cleaning and, worse,
                # deleting files that were just downloaded for historical/vintage pulls.
                since_day = pd_to_datetime_compat(download_since_datetime).normalize()
                today_day = pd.Timestamp.utcnow().normalize()
                n_days_implied = max(0, int((today_day - since_day).days))
                cleanup_old_files_n_days = abs(cleanup_old_files_n_days)
                if cleanup_old_files_n_days < n_days_implied:
                    old_value = cleanup_old_files_n_days
                    cleanup_old_files_n_days = n_days_implied
                    logger.warning(
                        "`cleanup_old_files_n_days` is less than the number of calendar "
                        "days since `since_datetime`, and is being adjusted "
                        f"from {old_value} to {cleanup_old_files_n_days}."
                    )
                self.cleanup_old_files(days_to_keep=cleanup_old_files_n_days)

        if skip_download and isinstance(cleanup_old_files_n_days, int):
            if cleanup_old_files_n_days > 0:
                logger.warning(
                    "`cleanup_old_files_n_days` is ignored when `skip_download=True`."
                )

        return self.load_data(
            tickers=rqstd_tickers,
            metrics=metrics,
            start_date=start_date,
            end_date=end_date,
            min_last_updated=min_last_updated,
            max_last_updated=max_last_updated,
            include_file_column=include_file_column,
            dataframe_format=dataframe_format,
            dataframe_type=dataframe_type,
            categorical_dataframe=categorical_dataframe,
            include_delta_files=include_delta_files,
            delta_treatment=delta_treatment,
            since_datetime=effective_since_datetime_for_load
            if (not skip_download)
            else since_datetime,
            to_datetime=to_datetime,
            catalog_file=catalog_file,
            datasets=datasets_to_download,
        )

    def download_as_of(
        self,
        tickers: Optional[List[str]] = None,
        cids: Optional[List[str]] = None,
        xcats: Optional[List[str]] = None,
        metrics: Optional[List[str]] = None,
        as_of_datetime: str = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        include_file_column: bool = False,
        dataframe_format: str = "qdf",
        dataframe_type: str = "pandas",
        categorical_dataframe: bool = True,
        include_delta_files: bool = True,
        delta_treatment: str = "latest",
        show_progress: bool = True,
        overwrite: bool = False,
        skip_download: bool = False,
        cleanup_old_files_n_days: Optional[int] = 5,
        *args,
        **kwargs,
    ) -> Union[pd.DataFrame, pl.DataFrame, pl.LazyFrame]:
        """
        Return data "as of" a specific point in time.

        This is a very lightweight wrapper around `download()`: it only translates the
        intent ("as of") into the correct `download()` arguments:
          - `to_datetime`: the file-vintage cutoff (which files can be used)
          - `max_last_updated`: the row-level vintage cutoff (which updates are allowed)

        Notes
        -----
        - If `as_of_datetime` is date-only (e.g., "2025-11-12"), it is interpreted as
          end-of-day UTC (23:59:59Z) for the *row-level* cutoff (`max_last_updated`), and
          uses the date itself as the *file-vintage* cutoff (`to_datetime="YYYYMMDD"`).
          If you want an intraday vintage, pass an explicit datetime string with timezone
          (e.g., "2025-11-12T06:30:00Z").
        - This wrapper sets `since_datetime` to the as-of date (UTC), to ensure the relevant
          snapshot/delta files for that day are downloaded into an empty cache without relying
          on the "today" default.
        """
        if args:
            raise TypeError(
                "download_as_of() only accepts keyword arguments (unexpected positional arguments)."
            )
        if as_of_datetime is None:
            raise ValueError("`as_of_datetime` must be provided.")

        validate_dq_timestamp(as_of_datetime, var_name="as_of_datetime")

        as_of_ts = pd_to_datetime_compat(as_of_datetime)
        as_of_day = as_of_ts.normalize()

        if _is_date_only_string(as_of_datetime):
            # Date-only: interpret as the full day (end-of-day) for the row-level cutoff,
            # but use the date itself as the file-vintage cutoff.
            to_datetime_str = as_of_day.strftime("%Y%m%d")
            max_last_updated_str = (
                as_of_day + pd.DateOffset(days=1) - pd.Timedelta(seconds=1)
            ).strftime("%Y-%m-%dT%H:%M:%SZ")
        else:
            # Datetime: treat as a precise vintage.
            to_datetime_str = as_of_ts.strftime("%Y-%m-%dT%H:%M:%SZ")
            max_last_updated_str = to_datetime_str

        # Download window start: use the as-of date (not "today") so an empty cache can
        # still download the relevant file-vintage for that day.
        since_str = as_of_day.strftime("%Y%m%d")

        # Main driver remains `self.download()`; keep logic out of this wrapper.
        return self.download(
            tickers=tickers,
            cids=cids,
            xcats=xcats,
            metrics=metrics,
            start_date=start_date,
            end_date=end_date,
            max_last_updated=max_last_updated_str,
            include_file_column=include_file_column,
            dataframe_format=dataframe_format,
            dataframe_type=dataframe_type,
            categorical_dataframe=categorical_dataframe,
            include_delta_files=include_delta_files,
            delta_treatment=delta_treatment,
            show_progress=show_progress,
            overwrite=overwrite,
            since_datetime=since_str,
            to_datetime=to_datetime_str,
            skip_download=skip_download,
            cleanup_old_files_n_days=cleanup_old_files_n_days,
            **kwargs,
        )


def _construct_all_tickers_list(
    tickers: Optional[List[str]] = None,
    cids: Optional[List[str]] = None,
    xcats: Optional[List[str]] = None,
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
        raise ValueError("At least one of `tickers`, `cids`, or `xcats` must be set.")

    if tickers is None:
        tickers = []

    if bool(cids) ^ bool(xcats):
        raise ValueError("Either both `cids` and `xcats` must be set, or neither.")

    if cids is None:
        cids, xcats = [], []

    tickers = sorted(set(tickers + [f"{c}_{x}" for c in cids for x in xcats]))
    return tickers


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


def _pd_to_utc_timestamp(ts: pd.Timestamp, utc: bool) -> pd.Timestamp:
    if not utc:
        return ts
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _pd_scalar_to_datetime_compat(
    ts: Union[str, datetime.date, datetime.datetime, pd.Timestamp],
    *,
    format: str,
    utc: bool,
) -> pd.Timestamp:
    if isinstance(ts, pd.Timestamp):
        return _pd_to_utc_timestamp(ts, utc=utc)
    if isinstance(ts, datetime.datetime):
        return _pd_to_utc_timestamp(pd.Timestamp(ts), utc=utc)
    if isinstance(ts, datetime.date):
        return _pd_to_utc_timestamp(pd.Timestamp(ts), utc=utc)
    if isinstance(ts, str):
        if PD_2_0_OR_LATER:
            return pd.to_datetime(ts, format=format, utc=utc)
        return _pd_to_datetime_compat(ts, utc=utc)
    raise TypeError(
        "`ts` must be a string, date, datetime, pandas Timestamp, or a Series."
    )


def pd_to_datetime_compat(
    ts: Union[str, datetime.date, datetime.datetime, pd.Timestamp, pd.Series],
    format: str = "mixed",
    utc: bool = True,
):
    """
    Parse common timestamp-like inputs into pandas datetime objects.

    - Scalars return a `pd.Timestamp`
    - `pd.Series` returns a Series of Timestamps

    Notes
    -----
    - Strings accept the same formats as `_pd_to_datetime_compat` (and in pandas>=2.0
      also support `format="mixed"`).
    - Non-string scalars (date/datetime/Timestamp) are converted via `pd.Timestamp`
      and optionally localized/converted to UTC.
    """

    if isinstance(ts, pd.Series):
        if PD_2_0_OR_LATER:
            return pd.to_datetime(ts, format=format, utc=utc)
        return ts.apply(
            lambda x: _pd_scalar_to_datetime_compat(x, format=format, utc=utc)
        )

    return _pd_scalar_to_datetime_compat(ts, format=format, utc=utc)


def pd_timestamp_compat(
    ts: Optional[Union[str, datetime.date, datetime.datetime, pd.Timestamp]] = None,
    *,
    utc: bool = True,
) -> pd.Timestamp:
    """
    Convert common timestamp-like inputs into a pandas Timestamp.

    This is a thin wrapper around `pd_to_datetime_compat` for scalar inputs. It keeps
    the legacy convenience of `ts=None` defaulting to "now".
    """
    if ts is None:
        return pd.Timestamp.utcnow() if utc else pd.Timestamp.now()

    out = pd_to_datetime_compat(ts, utc=utc)
    if isinstance(out, pd.Series):
        raise TypeError("`pd_timestamp_compat` expects a scalar input, not a Series.")
    return out


def get_current_or_last_business_day(
    now_utc: Optional[
        Union[str, datetime.date, datetime.datetime, pd.Timestamp]
    ] = None,
) -> pd.Timestamp:
    """
    Return "today" (UTC) if it's a weekday, otherwise the previous business day.

    Notes
    -----
    - "Business day" is Monday-Friday (no holiday calendar).
    - Returned Timestamp is UTC and normalized to midnight.
    """
    now_ts = pd_to_datetime_compat(
        now_utc if now_utc is not None else pd.Timestamp.utcnow(), utc=True
    ).normalize()
    # Monday=0 ... Sunday=6; weekend => >=5
    if now_ts.weekday() >= 5:
        return (now_ts - pd.offsets.BDay(1)).normalize()
    return now_ts


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


def _month_ends_between(
    start: datetime.date,
    end: datetime.date,
) -> List[datetime.date]:
    year, month = start.year, start.month
    out = []
    while (year, month) <= (end.year, end.month):
        dtx = datetime.date(year, month, calendar.monthrange(year, month)[1])
        if start <= dtx <= end:
            out.append(dtx)
        if month == 12:
            year, month = year + 1, 1
        else:
            month += 1
    return out


def _previous_business_day(d: datetime.date) -> datetime.date:
    if isinstance(d, datetime.datetime):
        d = d.date()
    while d.weekday() >= 5:  # 5=Sat, 6=Sun
        d -= datetime.timedelta(days=1)
    return d


@functools.lru_cache(maxsize=1)
def _large_delta_file_datetimes(as_str: bool = True) -> List[str]:
    sd = pd_to_datetime_compat(JPMAQS_EARLIEST_FILE_DATE).date()
    if isinstance(sd, datetime.datetime):
        sd = sd.date()
    ed = datetime.date.today()

    listA = _month_ends_between(sd, ed)
    listB = [_previous_business_day(d) for d in listA]

    all_dates = sorted(set(listA + listB))
    dt_list = [
        datetime.datetime.combine(d, datetime.time(23, 59, 59)) for d in all_dates
    ]
    dt_list = sorted(map(pd_to_datetime_compat, dt_list))
    if not as_str:
        return dt_list

    return [d.strftime("%Y%m%dT%H%M%S") for d in dt_list]


def pl_string_type():
    if POLARS_0_17_13_OR_EARLIER:
        return pl.Utf8
    return pl.String


class JPMaQSParquetExpectedColumns(Enum):
    TICKER = {
        "ticker": pl_string_type(),
        "real_date": pl.Date,
        "value": pl.Float64,
        "grading": pl.Float64,
        "eop_lag": pl.Float64,
        "mop_lag": pl.Float64,
        "last_updated": pl.Datetime(time_unit="us", time_zone=None),
    }
    METADATA = {
        "Theme": pl_string_type(),
        "Group": pl_string_type(),
        "Category": pl_string_type(),
        "Market Group": pl_string_type(),
        "Market": pl_string_type(),
        "Ticker": pl_string_type(),
        "Definition": pl_string_type(),
        "Last Updated": pl.Datetime(time_unit="ns", time_zone=None),
    }


def _check_individual_file_parquet_columns(
    file_path: Path,
) -> bool:
    assert isinstance(file_path, Path)
    base_name = file_path.name.upper()
    if not base_name.startswith("JPMAQS_") or not base_name.endswith(".PARQUET"):
        logger.warning(f"File {file_path} is not a recognized JPMAQS parquet file.")
        return False
    expected_cols = {}
    if "_METADATA" in base_name:
        expected_cols = JPMaQSParquetExpectedColumns.METADATA.value
    else:
        expected_cols = JPMaQSParquetExpectedColumns.TICKER.value

    schema = {}
    try:
        lf = pl.scan_parquet(file_path)
        if PYTHON_3_8_OR_LATER:
            schema = lf.collect_schema()
        else:
            schema = lf.schema
        schema = dict(schema)
        if lf.head(1).collect().is_empty():
            return False
    except Exception as e:
        logger.warning(f"Failed to read parquet file {file_path}: {e}")
        return False

    return schema == expected_cols


def _delete_corrupt_files(
    files: List[Path],
    extensions: List[str] = ["parquet", "json"],
    root_dir: Path = None,
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
                if not _check_individual_file_parquet_columns(
                    file_path=file_path,
                ):
                    raise ValueError("File is corrupt or has invalid schema")
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

    if root_dir is not None and root_dir.exists() and root_dir.is_dir():
        for dirpath, _, _ in os.walk(root_dir, topdown=False):
            dir_path = Path(dirpath)
            if not any(dir_path.iterdir()):
                try:
                    dir_path.rmdir()
                    logger.info(f"Removed empty directory: {dir_path}")
                except Exception:
                    logger.warning(f"Failed to remove directory: {dir_path}")

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
        *,
        parent_requester: RateLimitedRequester,
        debug: bool = False,
    ):
        """Initializes the downloader with URL, headers, and download parameters."""
        if parent_requester is None:
            raise ValueError("`parent_requester` must be provided for rate limiting.")
        self.parent_requester: RateLimitedRequester = parent_requester
        self.api_delay_seconds = api_delay * api_delay_margin
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

    def _wait_for_api_call(self) -> bool:
        return self.parent_requester._wait_for_api_call()

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
            self.log(f"File size: {total_size / (1024 * 1024):.2f} MB")

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
                time.sleep(self.api_delay_seconds)
                self.cleanup()
                return self.download(retries=retries - 1)

            self.cleanup()

        raise last_exception

    def _get_file_size(self) -> int:
        """Fetches the total size of the file using a HEAD request."""
        self.log("Fetching file size...")
        self._wait_for_api_call()
        start_time = time.time()
        response = requests.head(
            self.url,
            params=self.params,
            headers=self.headers,
            proxies=self.proxies,
            timeout=self.headers_timeout,
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
            self._wait_for_api_call()
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
        self.log(f"Assembled file size: {final_size / (1024 * 1024):.2f} MB")
        self.cleanup()

    def cleanup(self):
        """Removes the temporary directory and all downloaded parts."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            self.log("Cleaned up temporary files.")


def _check_lazy_load_inputs(
    files_dir: Union[str, Path],
    file_format: str,
    tickers: Optional[List[str]],
    cids: Optional[List[str]],
    xcats: Optional[List[str]],
    metrics: Optional[List[str]],
    start_date: Optional[Union[str, pd.Timestamp]],
    end_date: Optional[Union[str, pd.Timestamp]],
    min_last_updated: Optional[Union[str, pd.Timestamp]],
    max_last_updated: Optional[Union[str, pd.Timestamp]],
    delta_treatment: str,
    dataframe_format: str,
    dataframe_type: str,
    categorical_dataframe: bool,
    datasets: Optional[List[str]] = None,
):
    files_dir = Path(files_dir)
    if not files_dir.is_dir():
        raise FileNotFoundError(f"No such directory: {files_dir}")

    if file_format != "parquet":
        raise ValueError("`file_format` must be 'parquet'.")
    # check whether or not there are any parquet files in the glob directory -recursive
    if not _list_downloaded_files(files_dir, file_format):
        raise FileNotFoundError(
            f"No {file_format} files found in directory: {files_dir}"
        )
    if delta_treatment not in ["latest", "earliest", "all"]:
        raise ValueError(
            "`delta_treatment` must be one of 'latest', 'earliest', or 'all'."
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

    tickers_list = [
        t.strip() for t in (tickers or []) if isinstance(t, str) and t.strip()
    ]
    if not (bool(tickers_list) or (bool(cids) and bool(xcats))):
        raise ValueError(
            "No tickers specified. Provide `tickers=[...]` (or `cids` and `xcats`). "
            "If you want to load all tickers, pass `tickers` as a list of all tickers "
            "you want to load."
        )

    for param, name in [
        (start_date, "start_date"),
        (end_date, "end_date"),
        (min_last_updated, "min_last_updated"),
        (max_last_updated, "max_last_updated"),
    ]:
        if param is not None and not isinstance(param, (str, pd.Timestamp)):
            raise ValueError(f"`{name}` must be a string or pandas Timestamp.")
        if isinstance(param, str):
            try:
                pd_to_datetime_compat(param, utc=True)
            except ValueError:
                raise ValueError(
                    f"`{name}` has invalid timestamp format. Use YYYY-MM-DD or a "
                    "recognized timestamp format with timezone."
                )

    if dataframe_format not in ["qdf", "tickers"]:
        raise ValueError("`dataframe_format` must be one of 'qdf' or 'tickers'.")

    if dataframe_type not in ["pandas", "polars", "polars-lazy"]:
        raise ValueError(
            "`dataframe_type` must be one of 'pandas', 'polars', 'polars-lazy'."
        )
    if not isinstance(categorical_dataframe, bool):
        raise ValueError("`categorical_dataframe` must be a boolean.")


def _list_downloaded_files(files_dir: Path, file_format: str = "parquet") -> List[Path]:
    files_dir = Path(files_dir)
    assert files_dir.is_dir(), f"No such directory: {files_dir}"
    if file_format not in ["parquet", "json"]:
        raise ValueError("`file_format` must be one of 'parquet' or 'json'.")
    files = sorted(files_dir.glob(f"**/*.{file_format}"))
    return files


def _downloaded_files_df(
    files_dir: Path,
    file_format: str = "parquet",
    include_effective_dataset_column: bool = True,
    include_metadata_files: bool = False,
) -> pd.DataFrame:
    """
    Build a DataFrame of locally downloaded DataQuery files.

    Notes
    -----
    - `dataset` is the DataQuery "file-group-id" part of the filename (everything before
      the trailing timestamp segment).
    - `e-dataset` ("effective dataset") maps delta datasets back to their base dataset:
      delta file-groups (those ending with `_DELTA`) are treated as updates to the base
      dataset, not a separate dataset in their own right.
    """
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
    if include_effective_dataset_column:
        df["e-dataset"] = df["dataset"].str.replace(r"_DELTA$", "", regex=True)
    df = df.reset_index(drop=True)
    return df


def _is_date_only_string(x: Any) -> bool:
    """
    True for date-only strings like "YYYYMMDD" or "YYYY-MM-DD".

    This is used to interpret "date-only" `since_datetime`/`to_datetime` inputs as
    whole-day windows when selecting files or filtering by `last_updated`.
    """
    if not isinstance(x, str):
        return False
    return ("T" not in x) and (":" not in x)


def _normalize_file_timestamp_cutoff(
    x: Union[str, pd.Timestamp, datetime.date, datetime.datetime],
) -> pd.Timestamp:
    """
    Normalize a file-vintage cutoff.

    - For date-only strings, interpret as *end of that day* (inclusive).
    - For datetime-like inputs, keep the exact timestamp.
    """
    ts = pd_to_datetime_compat(x)
    if _is_date_only_string(x):
        ts = ts.normalize() + pd.DateOffset(days=1) - pd.Timedelta(nanoseconds=1)
    return ts


def _normalize_last_updated_cutoff(
    x: Union[str, pd.Timestamp, datetime.date, datetime.datetime],
) -> pd.Timestamp:
    """
    Normalize a `last_updated` cutoff.

    - For date-only strings, interpret as *end of that day* (inclusive) so that
      updates on that date are retained.
    - For datetime-like inputs, keep the exact timestamp.
    """
    return _normalize_file_timestamp_cutoff(x)


def _covering_large_delta_timestamp(
    to_ts: pd.Timestamp, delta_file_timestamps: Sequence[pd.Timestamp]
) -> Optional[pd.Timestamp]:
    """
    For monthly "large delta" regimes, return the first month-end-ish delta timestamp
    that covers `to_ts`, even if it is after `to_ts`.

    This is needed because the monthly delta file for a given month can be timestamped
    at month-end (or the previous business day), which may fall *after* a user's
    requested `to_datetime` within the month. In that case, we still need to load the
    covering delta file and then filter rows using `max_last_updated <= to_datetime`.
    """
    if not delta_file_timestamps:
        return None

    delta_ts_set = set(delta_file_timestamps)
    large_delta_ts_all = _large_delta_file_datetimes(as_str=False)
    if not large_delta_ts_all:
        return None

    # Candidates are "large delta" timestamps in the same (year, month) at or after to_ts.
    candidates = [
        ts
        for ts in large_delta_ts_all
        if (ts.year == to_ts.year and ts.month == to_ts.month and ts >= to_ts)
    ]
    for ts in sorted(candidates):
        if ts in delta_ts_set:
            return ts
    return None


def _month_end_dq_timestamp(
    x: Union[str, pd.Timestamp, datetime.date, datetime.datetime],
) -> str:
    """
    Return a DataQuery-style timestamp string for month-end 23:59:59 of `x`'s month.

    Used to expand a `to_datetime` download window when monthly delta files are used,
    so that the covering month-end delta file is included in the download.
    """
    ts = pd_to_datetime_compat(x)
    y, m = int(ts.year), int(ts.month)
    last_day = calendar.monthrange(y, m)[1]
    dt = datetime.datetime(y, m, last_day, 23, 59, 59)
    return dt.strftime("%Y%m%dT%H%M%S")


def _select_local_files_for_load(
    files_df: pd.DataFrame,
    *,
    since_datetime: Optional[Union[str, pd.Timestamp]] = None,
    to_datetime: Optional[Union[str, pd.Timestamp]] = None,
    include_delta_files: bool = True,
    warn_if_no_full_snapshots: bool = False,
) -> pd.DataFrame:
    """
    Single-responsibility helper: choose which local snapshot/delta files to load.

    The selection is per effective dataset ("e-dataset"):
    - If full snapshots exist for the dataset and at least one snapshot is present in the
      requested file-vintage window, load the latest snapshot in the window and any delta
      files newer than that snapshot (also within the window).
    - If no full snapshots exist at all for the dataset (delta-only history), load *all*
      available deltas up to the requested vintage. For monthly "large delta" regimes,
      also include the covering month-end delta file even if it timestamps after
      `to_datetime` (row-level filtering is handled via `max_last_updated`).
    """
    if files_df.empty:
        return files_df

    df = files_df.copy()

    if "e-dataset" in df.columns:
        group_col = "e-dataset"
    else:
        group_col = "dataset"
        if group_col not in df.columns:
            raise ValueError("Expected column 'dataset' in files_df")
        df[group_col] = (
            df[group_col].astype(str).str.replace(r"_DELTA$", "", regex=True)
        )

    if "file-timestamp" not in df.columns:
        raise ValueError("Expected column 'file-timestamp' in files_df")

    since_ts = (
        pd_to_datetime_compat(since_datetime) if since_datetime is not None else None
    )
    if since_ts is not None and _is_date_only_string(since_datetime):
        since_ts = since_ts.normalize()

    vintage_to_ts = (
        _normalize_file_timestamp_cutoff(to_datetime)
        if to_datetime is not None
        else df["file-timestamp"].max()
    )

    # For snapshot-led selection we keep the historical behaviour of treating
    # (`since_datetime`, `to_datetime`) as an unordered window and swapping if needed.
    window_since_ts = since_ts
    window_to_ts = vintage_to_ts
    if window_since_ts is not None and window_since_ts > window_to_ts:
        window_since_ts, window_to_ts = window_to_ts, window_since_ts

    earliest_snapshot_ts: Optional[pd.Timestamp] = None
    if warn_if_no_full_snapshots and window_since_ts is not None:
        is_delta_all = df["filename"].astype(str).str.contains("_DELTA")
        is_metadata_all = df["filename"].astype(str).str.contains("_METADATA")
        snapshots_all = df.loc[~is_delta_all & ~is_metadata_all].copy()
        earliest_snapshot_ts = snapshots_all["file-timestamp"].min()
        if pd.isna(earliest_snapshot_ts):
            earliest_snapshot_ts = None

    selected = []
    for _, g in df.groupby(group_col):
        if g.empty:
            continue

        is_delta_g = g["filename"].astype(str).str.contains("_DELTA")
        is_metadata_g = g["filename"].astype(str).str.contains("_METADATA")
        snapshots_all_g = g.loc[~is_delta_g & ~is_metadata_g].copy()

        # Delta-only history: no snapshots exist at all for this dataset.
        if snapshots_all_g.empty:
            if not include_delta_files:
                continue

            deltas_all = g.loc[is_delta_g].copy()
            if deltas_all.empty:
                continue

            effective_to_ts = vintage_to_ts
            if to_datetime is not None:
                cover_ts = _covering_large_delta_timestamp(
                    to_ts=vintage_to_ts,
                    delta_file_timestamps=deltas_all["file-timestamp"].tolist(),
                )
                if cover_ts is not None:
                    effective_to_ts = cover_ts

            deltas_sel = deltas_all[
                deltas_all["file-timestamp"].le(effective_to_ts)
            ].copy()
            selected.append(deltas_sel)
            continue

        # Windowed candidate set (matches historical behaviour for snapshot-led selection).
        if window_since_ts is not None:
            g_window = g[
                g["file-timestamp"].between(window_since_ts, window_to_ts)
            ].copy()
        else:
            g_window = g[g["file-timestamp"].le(window_to_ts)].copy()
        if g_window.empty:
            continue

        is_delta_w = g_window["filename"].astype(str).str.contains("_DELTA")
        is_metadata_w = g_window["filename"].astype(str).str.contains("_METADATA")
        snapshots_w = g_window.loc[~is_delta_w & ~is_metadata_w].copy()

        # Snapshot-led selection (preserves the window semantics):
        if snapshots_w.empty:
            # No snapshots within the requested window: fall back to deltas in-window (if any).
            if include_delta_files:
                selected.append(g_window.loc[is_delta_w].copy())
            continue

        latest_snapshot_ts = snapshots_w["file-timestamp"].max()
        snapshots_sel = snapshots_w[
            snapshots_w["file-timestamp"] == latest_snapshot_ts
        ].copy()
        if not include_delta_files:
            selected.append(snapshots_sel)
            continue

        deltas_w = g_window.loc[is_delta_w].copy()
        deltas_sel = deltas_w[deltas_w["file-timestamp"] >= latest_snapshot_ts].copy()
        selected.append(pd.concat([snapshots_sel, deltas_sel], ignore_index=True))

    out = (
        pd.concat(selected, ignore_index=True).reset_index(drop=True)
        if selected
        else df.iloc[0:0].copy()
    )

    if out.empty:
        return out

    if warn_if_no_full_snapshots and window_since_ts is not None:
        is_delta_out = out["filename"].astype(str).str.contains("_DELTA")
        is_metadata_out = out["filename"].astype(str).str.contains("_METADATA")
        snapshots_out = out.loc[~is_delta_out & ~is_metadata_out].copy()
        if snapshots_out.empty and bool(is_delta_out.any()):
            earliest_snapshot_str = None
            if earliest_snapshot_ts is not None:
                earliest_snapshot_str = earliest_snapshot_ts.strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                )
            else:
                earliest_snapshot_str = "N/A"
            logger.warning(
                "No full snapshots available in the requested window "
                f"since={window_since_ts.strftime('%Y-%m-%dT%H:%M:%SZ')} "
                f"to={window_to_ts.strftime('%Y-%m-%dT%H:%M:%SZ')} "
                f"earliest_snapshot={earliest_snapshot_str}"
            )

    if not include_delta_files:
        # keep only snapshots
        is_delta = out["filename"].astype(str).str.contains("_DELTA")
        is_metadata = out["filename"].astype(str).str.contains("_METADATA")
        out = out.loc[~is_delta & ~is_metadata].copy()

    out = out.sort_values([group_col, "file-timestamp", "filename"]).reset_index(
        drop=True
    )
    return out


def _filter_to_latest_files(
    files_df: pd.DataFrame,
    since_datetime: Optional[Union[str, pd.Timestamp]] = None,
    to_datetime: Optional[Union[str, pd.Timestamp]] = None,
    include_delta_files: bool = True,
    delta_treatment: str = "all",
    warn_if_no_full_snapshots: bool = False,
) -> pd.DataFrame:
    """
    Backwards-compatible wrapper around `_select_local_files_for_load()`.

    Historically this function selected the latest full snapshot per dataset (within an
    optional window) plus any newer deltas. It now also supports "delta-only history"
    regimes (no full snapshots available), including monthly "large delta" coverage.
    """
    return _select_local_files_for_load(
        files_df,
        since_datetime=since_datetime,
        to_datetime=to_datetime,
        include_delta_files=include_delta_files,
        warn_if_no_full_snapshots=warn_if_no_full_snapshots,
    )


def lazy_load_from_parquets(
    files_dir: Union[str, Path],
    file_format: str = "parquet",
    tickers: Optional[List[str]] = None,
    metrics: Optional[List[str]] = None,
    start_date: Optional[Union[str, pd.Timestamp]] = None,
    end_date: Optional[Union[str, pd.Timestamp]] = None,
    min_last_updated: Optional[Union[str, pd.Timestamp]] = None,
    max_last_updated: Optional[Union[str, pd.Timestamp]] = None,
    dataframe_format: str = "qdf",
    dataframe_type: str = "pandas",
    categorical_dataframe: bool = True,
    datasets: Optional[List[str]] = None,
    include_delta_files: bool = True,
    delta_treatment: str = "latest",
    since_datetime: Optional[Union[str, pd.Timestamp]] = None,
    to_datetime: Optional[Union[str, pd.Timestamp]] = None,
    include_file_column: bool = True,
    catalog_file: Optional[str] = None,
    warn_if_no_full_snapshots: bool = False,
) -> Union[pd.DataFrame, pl.DataFrame, pl.LazyFrame]:
    """
    This function helps to lazily load JPMaQS parquet files from a specified directory.
    It operates using the exact ticker names provided.

    Notes
    -----
    The `datasets` argument applies to
    the "effective dataset" (e-dataset), meaning that delta datasets (those ending with
    `_DELTA`) are treated as updates to their base dataset, not as separate datasets.

    Vintage selection (`to_datetime`)
    -------------------------------
    When JPMaQS removes older full snapshots, historical data may only be reconstructible
    from delta files. In monthly "large delta" regimes, the delta file for a given month
    is timestamped at month-end (or the previous business day), which can fall *after*
    an in-month `to_datetime` (e.g., `to_datetime="2025-03-15"`).

    In that case the loader will still select the covering month-end delta file and you
    should use `max_last_updated <= to_datetime` to exclude updates beyond the requested
    vintage. `DataQueryFileAPIClient.load_data()` applies this default automatically
    when `to_datetime` is provided without `max_last_updated`.
    """
    files_dir = Path(files_dir)
    if (not metrics) or (metrics == "all") or ("all" in metrics):
        metrics = JPMAQS_METRICS

    delta_treatment = delta_treatment.lower()

    _check_lazy_load_inputs(
        files_dir=files_dir,
        file_format=file_format,
        tickers=tickers,
        cids=None,  # intentionally set to None
        xcats=None,  # intentionally set to None
        metrics=metrics,
        start_date=start_date,
        end_date=end_date,
        min_last_updated=min_last_updated,
        max_last_updated=max_last_updated,
        delta_treatment=delta_treatment,
        dataframe_format=dataframe_format,
        dataframe_type=dataframe_type,
        categorical_dataframe=categorical_dataframe,
        datasets=datasets,
    )

    all_data_files_df: pd.DataFrame = _downloaded_files_df(
        files_dir=files_dir,
        file_format=file_format,
        include_metadata_files=False,  # no metadata files - cannot scan with QDF like schema
    )
    effective_to_datetime = to_datetime
    if (
        warn_if_no_full_snapshots
        and (since_datetime is not None)
        and (to_datetime is None)
    ):
        effective_to_datetime = pd.Timestamp.utcnow().strftime("%Y%m%dT%H%M%S")
    available_files_df: pd.DataFrame = _filter_to_latest_files(
        files_df=all_data_files_df,
        since_datetime=since_datetime,
        to_datetime=effective_to_datetime,
        include_delta_files=include_delta_files,
        delta_treatment=delta_treatment,
        warn_if_no_full_snapshots=warn_if_no_full_snapshots,
    )
    if datasets:
        datasets = sorted(set([d.replace("_DELTA", "") for d in datasets]))
        if "e-dataset" in available_files_df.columns:
            available_files_df = available_files_df.loc[
                available_files_df["e-dataset"].isin(datasets)
            ]
        else:
            available_files_df = available_files_df.iloc[0:0].copy()

    include_file_column = "source_file" if include_file_column else None

    tickers = [t.strip() for t in tickers if isinstance(t, str) and t.strip()]

    if not tickers:
        raise ValueError(
            "No tickers specified. Provide `tickers=[...]` (or `cids` and `xcats`). "
            "If you want to load all tickers, pass `tickers` as a list of all tickers "
            "you want to load."
        )

    paths = (
        sorted(available_files_df["path"])
        if (not available_files_df.empty and ("path" in available_files_df.columns))
        else []
    )
    if not paths:
        total_parquets = len(_list_downloaded_files(files_dir, file_format="parquet"))
        total_data_parquets = len(all_data_files_df)
        today_utc = pd.Timestamp.utcnow().normalize()
        last_bd = get_current_or_last_business_day(today_utc)
        datasets_str = (
            ", ".join(list(datasets)[:5]) + ("..." if len(datasets) > 5 else "")
            if datasets
            else "N/A"
        )
        extra_hint = ""
        cache_hint = (
            f"Local cache scanned: '{files_dir.resolve()}'. "
            f"Found {total_parquets} parquet file(s) total."
        )
        if total_parquets > 0 and total_data_parquets == 0:
            extra_hint = (
                " Found parquet file(s), but they appear to be metadata/catalog only "
                "(no data snapshot/delta files)."
            )
        raise FileNotFoundError(
            "No JPMaQS data snapshot/delta parquet files were found in the local cache "
            f"to load for the requested selection (datasets={datasets_str}).{extra_hint} "
            f"{cache_hint} "
            "This is a local-cache issue (not necessarily a DataQuery availability issue). "
            "Common causes: only metadata files were downloaded for the selected vintage, "
            "or cache cleanup removed older files. "
            "If you are making a historical/vintage request, try setting "
            "`cleanup_old_files_n_days=None`. "
            "JPMaQS data files are published on business days only (the catalog is daily). "
            f"Today (UTC) is {today_utc.date()}; latest business day is {last_bd.date()}. "
            "Try running `download()` again (with `skip_download=False`) or set "
            f"`since_datetime='{last_bd.strftime('%Y%m%d')}'` (or earlier)."
        )

    if catalog_file:
        catalog_path = Path(catalog_file)
        if not catalog_path.is_file():
            raise FileNotFoundError(f"No such file: {catalog_path}")

        catalog_lf = pl.scan_parquet(str(catalog_path))
        if PYTHON_3_8_OR_LATER:
            schema_cols = catalog_lf.collect_schema().names()
        else:
            schema_cols = catalog_lf.schema.keys()
        ticker_col = "Ticker" if "Ticker" in schema_cols else "ticker"
        tickers_lf_col = catalog_lf.select(pl.col(ticker_col)).drop_nulls().unique()
        if ticker_col in schema_cols:
            catalog_tickers = tickers_lf_col.collect()[ticker_col].to_list()
            if catalog_tickers:
                catalog_set = {str(t).lower() for t in catalog_tickers}
                missing = sorted({t for t in tickers if t.lower() not in catalog_set})
                if missing:
                    raise ValueError(
                        f"Ticker(s) not present in JPMaQS catalog: {', '.join(missing)}."
                    )

    lf: pl.LazyFrame = _lazy_load_filtered_parquets(
        paths=paths,
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        delta_treatment=delta_treatment,
        min_last_updated=min_last_updated,
        max_last_updated=max_last_updated,
        return_qdf=(dataframe_format == "qdf"),
        include_file_column=include_file_column,
    )
    if (metrics and set(metrics) != set(JPMAQS_METRICS)) or include_file_column:
        cols_to_keep = ["real_date", "cid", "xcat", "ticker"] + metrics
        if include_file_column:
            cols_to_keep.append(include_file_column)
        if PYTHON_3_8_OR_LATER:
            lf = lf.select(
                [pl.col(c) for c in cols_to_keep if c in lf.collect_schema().names()]
            )
        else:
            lf = lf.select([pl.col(c) for c in cols_to_keep if c in lf.schema.keys()])
    cat_cols = ["cid", "xcat", "ticker"]
    if include_file_column:
        cat_cols.append(include_file_column)
    if dataframe_type in {"polars", "polars-lazy"}:
        if categorical_dataframe:
            categorical_dtype = getattr(pl, "Categorical", None)
            if categorical_dtype is not None:
                _names = (
                    lf.collect_schema().names()
                    if PYTHON_3_8_OR_LATER
                    else lf.schema.keys()
                )
                cols = [c for c in cat_cols if c in _names]
                for c in cols:
                    try:
                        lf = lf.with_columns(pl.col(c).cast(categorical_dtype))
                    except Exception:
                        logger.warning(
                            f"Failed to cast '{c}' to Categorical; keeping as string."
                        )
        return lf if dataframe_type == "polars-lazy" else lf.collect()
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


def _expr_split_ticker(ticker_expr: pl.Expr) -> Tuple[pl.Expr, pl.Expr]:
    """
    Robust split of 'CID_XCAT...' into (cid, xcat) WITHOUT using splitn().
    Works across Polars versions (avoids struct vs list return type issues).
    """
    splitx = ticker_expr.str.splitn("_", 2)
    cid = splitx.struct.field("field_0")
    xcat = splitx.struct.field("field_1")
    return cid, xcat


def _ensure_columns(
    lf: pl.LazyFrame,
    cols: Sequence[str],
    dtypes: Optional[Dict[str, "pl.DataType"]] = None,
) -> pl.LazyFrame:
    """
    Ensure all `cols` exist before .select(...).
    This runs schema-only (lf.collect_schema()), not a materialization.
    """
    if PYTHON_3_8_OR_LATER:
        have = set(lf.collect_schema().keys())
    else:
        have = set(lf.schema.keys())
    missing = [c for c in cols if c not in have]
    if not missing:
        return lf

    add_exprs = {}
    for c in missing:
        expr = pl.lit(None)
        if dtypes and c in dtypes:
            expr = expr.cast(dtypes[c])
        add_exprs[c] = expr

    return lf.with_columns(**add_exprs)


def _filter_lazy_frame_by_tickers(
    lf: pl.LazyFrame,
    tickers: Sequence[str],
    start_date: Optional[Union[str, pd.Timestamp]],
    end_date: Optional[Union[str, pd.Timestamp]],
    min_last_updated: Optional[Union[str, pd.Timestamp]],
    max_last_updated: Optional[Union[str, pd.Timestamp]],
) -> pl.LazyFrame:
    tickers_list = [t for t in tickers if t]
    lf = lf.filter(pl.col("ticker").is_in(tickers_list))
    if start_date:
        start_date = pd_to_datetime_compat(start_date).strftime("%Y-%m-%d")
        lf = lf.filter(pl.col("real_date") >= pl.lit(start_date).str.to_date())
    if end_date:
        end_date = pd_to_datetime_compat(end_date).strftime("%Y-%m-%d")
        lf = lf.filter(pl.col("real_date") <= pl.lit(end_date).str.to_date())
    if min_last_updated:
        min_last_updated = pd_to_datetime_compat(min_last_updated).to_datetime64()
        lf = lf.filter(pl.col("last_updated") >= pl.lit(min_last_updated))
    if max_last_updated:
        max_last_updated = pd_to_datetime_compat(max_last_updated).to_datetime64()
        lf = lf.filter(pl.col("last_updated") <= pl.lit(max_last_updated))

    return lf


def _to_output_schema(
    lf: pl.LazyFrame,
    include_file_column: Optional[str],
    want_qdf: bool,
) -> pl.LazyFrame:
    """Normalize columns to qdf or ticker-based shape."""
    cols = "real_date.ticker.value.eop_lag.mop_lag.grading.last_updated"
    if include_file_column:
        cols += "." + include_file_column
    ticker_cols = cols.split(".")
    qdf_cols = cols.replace("ticker", "cid.xcat").split(".")

    dtype_map = dict(JPMaQSParquetExpectedColumns.TICKER.value)
    dtype_map.update({"cid": pl_string_type(), "xcat": pl_string_type()})

    if want_qdf:
        cid_expr, xcat_expr = _expr_split_ticker(pl.col("ticker"))
        lf = lf.with_columns(cid=cid_expr, xcat=xcat_expr)
        lf = _ensure_columns(lf, qdf_cols, dtypes=dtype_map)
        return lf.select(qdf_cols)

    lf = _ensure_columns(lf, ticker_cols, dtypes=dtype_map)
    return lf.select(ticker_cols)


def _build_filtered_parquet_lazyframe(
    paths: Sequence[Union[str, os.PathLike]],
    tickers_list: Sequence[str],
    *,
    start_date: Optional[Union[pd.Timestamp, str]] = None,
    end_date: Optional[Union[pd.Timestamp, str]] = None,
    min_last_updated: Optional[Union[pd.Timestamp, str]] = None,
    max_last_updated: Optional[Union[pd.Timestamp, str]] = None,
    include_file_column: Optional[str] = None,
    return_qdf: bool = False,
) -> pl.LazyFrame:
    """
    Scan multiple parquet paths into a single LazyFrame, optionally adding a file-path
    column in a way compatible with Polars 0.17.13 (Python 3.7).

    NOTE: categorical casting is intentionally not done here. Casting per-file columns
    to `pl.Categorical` before concatenation can break on older Polars/Python (e.g.
    Polars 0.17.x on Python 3.7). Callers should cast categoricals after concat.
    """
    lazy_parts: List[pl.LazyFrame] = []

    for pth in paths:
        file_base_name = Path(pth).name
        pth_str = os.fspath(pth)
        lf = pl.scan_parquet(pth_str)

        lf = _filter_lazy_frame_by_tickers(
            lf=lf,
            tickers=tickers_list,
            start_date=start_date,
            end_date=end_date,
            min_last_updated=min_last_updated,
            max_last_updated=max_last_updated,
        )

        if include_file_column:
            lf = lf.with_columns(pl.lit(file_base_name).alias(include_file_column))
        lf = _to_output_schema(
            lf=lf,
            include_file_column=include_file_column,
            want_qdf=return_qdf,
        )

        lazy_parts.append(lf)

    if not lazy_parts:
        return pl.DataFrame().lazy()

    return pl.concat(lazy_parts, how="vertical")


def _lazy_load_filtered_parquets(
    paths: List[str],
    tickers: List[str],
    start_date: Optional[Union[str, pd.Timestamp]],
    end_date: Optional[Union[str, pd.Timestamp]],
    min_last_updated: Optional[Union[str, pd.Timestamp]],
    max_last_updated: Optional[Union[str, pd.Timestamp]],
    delta_treatment: str,
    include_file_column: Optional[str],
    return_qdf: bool = True,
) -> pl.LazyFrame:
    if not paths:
        raise ValueError("No paths provided")

    tickers_list: List[str] = list(dict.fromkeys(tickers))

    out: pl.LazyFrame = _build_filtered_parquet_lazyframe(
        paths=paths,
        tickers_list=tickers_list,
        start_date=start_date,
        end_date=end_date,
        min_last_updated=min_last_updated,
        max_last_updated=max_last_updated,
        include_file_column=include_file_column,
        return_qdf=return_qdf,
    )

    key_cols = ["cid", "xcat"] if return_qdf else ["ticker"]
    full_key = key_cols + ["real_date"]

    if delta_treatment != "all":
        if delta_treatment == "latest":
            out = out.sort(
                full_key + ["last_updated"], descending=[False] * len(full_key) + [True]
            ).unique(subset=full_key, keep="first")
        elif delta_treatment == "earliest":
            out = out.sort(
                full_key + ["last_updated"],
                descending=[False] * len(full_key) + [False],
            ).unique(subset=full_key, keep="first")
        else:
            raise ValueError(f"Unknown delta_treatment: {delta_treatment}")
    sort_cols = ["cid", "xcat"] if return_qdf else ["ticker"]
    out = out.sort(sort_cols + ["real_date"])

    return out


if __name__ == "__main__":
    print("Current time UTC:", pd.Timestamp.utcnow().isoformat())

    # start = time.time()
    # since_datetime = pd.Timestamp.now() - pd.offsets.BDay(7)
    # print(
    #     f"Downloading full-snapshots, delta-files, and metadata files published since {since_datetime}"
    # )
    # since_datetime = since_datetime.strftime("%Y%m%d")
    # with DataQueryFileAPIClient(out_dir="./data/jpmaqs-data/") as dq:
    #     dq.download_catalog_file()
    #     dq.download_full_snapshot(since_datetime=since_datetime)
    #     print(dq.get_revisions_notifications().head())
    #     print(dq.get_missing_data_notifications().head())
    # end = time.time()
    # print(f"Download completed in {end - start:.2f} seconds")

    test_cids = ["AUD", "BRL", "CAD", "CHF", "CNY", "CZK", "EUR", "GBP", "USD"]
    test_xcats = ["RIR_NSA", "FXXR_NSA", "FXXR_VT10", "DU05YXR_NSA", "DU05YXR_VT10"]
    tickers = [f"{c}_{x}" for c in test_cids for x in test_xcats]

    with DataQueryFileAPIClient(out_dir="./data/jpmaqs-data/") as dq:
        df = dq.download(
            tickers=tickers,
            include_file_column=True,
        )
        print(df.head())

    # with DataQueryFileAPIClient(out_dir="./data/jpmaqs-data/") as dq:
    #     pl_df: pl.DataFrame = dq.download(
    #         cids=test_cids,
    #         xcats=test_xcats,
    #         dataframe_format="tickers",
    #         dataframe_type="polars",
    #     )
    #     print(pl_df.head())
