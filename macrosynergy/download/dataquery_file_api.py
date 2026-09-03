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
    available_files_df = client.list_available_files()
    print("Available JPMaQS files:")
    print(available_files_df.head())

**Example 2: Download all new or updated files for the current day.**

This is the recommended way to get a daily snapshot of all JPMaQS data,
including full datasets, deltas, and metadata.

.. code-block:: python

    from macrosynergy.download import DataQueryFileAPIClient
    client = DataQueryFileAPIClient(out_dir="./jpmaqs_data")

    print(f"Downloading today's files to {client.out_dir}...")
    client.download_latest_files()
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


**Example 3a: Every version of a row, with `delta_treatment="all"`.**

By default `download` returns one row per (cid, xcat, real_date): the most recently
published one. `delta_treatment="all"` keeps every version instead, so the same
(cid, xcat, real_date) can appear more than once, told apart by `last_updated`.

.. code-block:: python

    from macrosynergy.download import DataQueryFileAPIClient

    with DataQueryFileAPIClient(out_dir="./jpmaqs_data") as client:
        df = client.download(
            cids=["AUD", "CAD"],
            xcats=["EQXR_NSA"],
            delta_treatment="all",
            dropna=False,
        )


.. code-block:: text

       real_date  cid      xcat  value  eop_lag  mop_lag  grading        last_updated
    0 2024-01-02  AUD  EQXR_NSA  0.400      0.0      1.0      1.0 2024-01-02 06:00:00
    1 2024-01-02  AUD  EQXR_NSA  0.412      0.0      1.0      1.0 2024-01-04 06:00:00
    2 2024-01-03  AUD  EQXR_NSA -0.118      0.0      2.0      1.0 2024-01-03 06:00:00
    3 2024-01-03  AUD  EQXR_NSA    NaN      NaN      NaN      NaN 2024-01-05 06:00:00
    4 2024-01-02  CAD  EQXR_NSA  0.221      0.0      1.0      1.0 2024-01-02 06:00:00
    5 2024-01-03  CAD  EQXR_NSA  0.305      0.0      2.0      1.0 2024-01-03 06:00:00

Reading that output:

- Rows 0 and 1 are the same (cid, xcat, real_date) at two `last_updated` values: row 1 is
  an updated value for that observation.
- Row 3 has all four metrics NaN. That is a removal of data for (AUD, EQXR_NSA,
  2024-01-03), recorded at its own `last_updated`.
- Rows 4 and 5 have one version each, and look exactly as they do by default.

Rows are ordered by `cid`, `xcat` (or `ticker`), `real_date`, then `last_updated`, so each
observation reads oldest first. `dropna=False` is required here, as the all-NaN metrics
rows are the record of removal.


**Example 4: Download all new or updated delta-files since a specific date/time.**

.. code-block:: python

    from macrosynergy.download import DataQueryFileAPIClient
    import pandas as pd

    client = DataQueryFileAPIClient("./jpmaqs_data")
    since_datetime = pd.Timestamp.today() - pd.DateOffset(days=10)

    client.download_files(
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

**Example 7: Load "notification" metadata (missing updates & revisions).**

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

"""

import calendar
import concurrent.futures as cf
import contextlib
import datetime
import json
import logging
import os
import shutil
import time
import traceback as tb
import uuid
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, overload

import pandas as pd
import polars as pl
import requests
from tqdm import tqdm

from macrosynergy.compat import PD_2_0_OR_LATER, PYTHON_3_8_OR_LATER
from macrosynergy.download.dataquery import JPMAQS_GROUP_ID, OAUTH_TOKEN_URL
from macrosynergy.download.exceptions import DownloadError, InvalidResponseError
from macrosynergy.download.fusion_interface import (
    _wait_for_api_call,
    cache_decorator,
    request_wrapper,
    request_wrapper_stream_bytes_to_disk,
)
from macrosynergy.download.jpm_oauth import JPMorganOAuth
from macrosynergy.management.constants import JPMAQS_METRICS

DQ_FILE_API_BASE_URL: str = (
    "https://api-dataquery.jpmchase.com/research/dataquery-authe/api/v2"
)
DQ_FILE_API_FALLBACK_BASE_URL: str = (
    "https://api-strm-gw01.jpmchase.com/research/dataquery-authe/api/v2"
)
DQ_FILE_API_SCOPE: str = "JPMC:URI:RS-06785-DataQueryExternalApi-PROD"
DQ_FILE_API_TIMEOUT: float = 300.0
DQ_FILE_API_HEADERS_TIMEOUT: float = DQ_FILE_API_TIMEOUT / 10.0
DQ_FILE_API_DELAY_PARAM: float = 0.04  # =1/25 ; 25 transactions per second
DQ_FILE_API_DELAY_MARGIN: float = 1.1  # 10% safety margin
DQ_FILE_API_SEGMENT_SIZE_MB: float = 8.0  # 8 MB
DQ_FILE_API_STREAM_CHUNK_SIZE: int = 8192  # 8 KB

JPMAQS_DATASET_THEME_MAPPING = {
    "Economic surprises": "JPMAQS_ECONOMIC_SURPRISES",
    "Financial conditions": "JPMAQS_FINANCIAL_CONDITIONS",
    "Generic returns": "JPMAQS_GENERIC_RETURNS",
    "Macroeconomic balance sheets": "JPMAQS_MACROECONOMIC_BALANCE_SHEETS",
    "Macroeconomic trends": "JPMAQS_MACROECONOMIC_TRENDS",
    "Shocks and risk measures": "JPMAQS_SHOCKS_RISK_MEASURES",
    "Stylized trading factors": "JPMAQS_STYLIZED_TRADING_FACTORS",
}


JPMAQS_EARLIEST_FILE_DATE = "20220101"

logger = logging.getLogger(__name__)


def _abbreviate_tickers_list(items: List[str], limit: int = 10) -> str:
    """Render a list for a message, naming how many were left out rather than eliding."""
    if len(items) <= limit:
        return str(items)
    return f"{items[:limit]} (+{len(items) - limit} more)"


@contextlib.contextmanager
def _suppressed_warnings(active: bool):
    """
    Silence both warning channels for the duration, restoring them even on an exception.
    """
    if not active:
        yield
        return
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        previous_level = logger.level
        logger.setLevel(logging.ERROR)
        try:
            yield
        finally:
            logger.setLevel(previous_level)


def utc_now() -> pd.Timestamp:
    """
    Returns the current UTC timestamp as a pandas Timestamp object.
    """
    return pd.Timestamp(datetime.datetime.now(datetime.timezone.utc))


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


def _resolve_base_url(
    primary: str,
    fallback: str,
    timeout: float = 10.0,
    verify: bool = True,
    proxies: Optional[Dict[str, str]] = None,
) -> str:
    """
    Probe which DataQuery File API base URL is reachable.

    Tries *primary* first; on connection failure, falls back to *fallback*.
    Each ``DataQueryFileAPIClient`` instance calls this during construction,
    so the probe runs once per instance.
    """
    for url, is_fallback in [(primary, False), (fallback, True)]:
        try:
            requests.head(url, timeout=timeout, verify=verify, proxies=proxies)
        except requests.exceptions.RequestException:
            if not is_fallback:
                logger.debug(
                    "Primary DataQuery File API URL not reachable (%s), "
                    "trying fallback...",
                    primary,
                )
            continue

        if is_fallback:
            warnings.warn(
                f"The primary DataQuery File API URL is not reachable: "
                f"{primary}\n"
                f"Falling back to: {url}\n"
                f"Please whitelist/allow the primary URL in your "
                f"network/firewall configuration.",
                UserWarning,
                stacklevel=2,
            )
            logger.warning(
                "DataQuery File API URL fallback active: using %s instead of %s.",
                url,
                primary,
            )

        return url

    # Both unreachable - return primary and let normal error handling surface it
    return primary


class DataQueryFileAPIClient:
    """
    A client for accessing JPMaQS product files via the JPMorgan DataQuery File API.

    This client provides an alternative distribution channel to the Fusion API for JPMaQS
    data. It is designed to list and download JPMaQS data files, which are
    available as full snapshots, daily deltas, and metadata files. The client handles
    authentication, API requests, and file downloads, including large file downloads
    using a segmented, concurrent approach.

    Files are saved to disk unmodified, in the ticker-based Parquet format in which they
    are delivered. Conversion to a QuantamentalDataFrame (or other format) is performed
    on read, when the data is loaded via `download`.

    Parameters
    ----------
    client_id : Optional[str]
        Client ID for authentication. If not provided, it will be sourced from
        environment variables (`DQ_CLIENT_ID` or `DATAQUERY_CLIENT_ID`).
    client_secret : Optional[str]
        Client Secret for authentication. If not provided, it will be sourced from
        environment variables (`DQ_CLIENT_SECRET` or `DATAQUERY_CLIENT_SECRET`).
    out_dir : Optional[str]
        Output directory for all downloads. Files are saved under a "jpmaqs-download"
        subdirectory of this path (e.g. "~/jpmaqs-data/jpmaqs-download"). Defaults to
        "~/jpmaqs-data".
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
        base_url: str = DQ_FILE_API_BASE_URL,
        scope: str = DQ_FILE_API_SCOPE,
        proxies: Optional[Dict[str, str]] = None,
        verify_ssl: bool = True,
        out_dir: Optional[str] = None,
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
        self.out_dir = out_dir or "~/jpmaqs-data"
        self.out_dir = Path(self.out_dir).expanduser().resolve()
        self.base_url = _resolve_base_url(
            primary=base_url,
            fallback=DQ_FILE_API_FALLBACK_BASE_URL,
            verify=verify_ssl,
            proxies=proxies,
        ).rstrip("/")
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

    def _get_save_dir(self) -> str:
        """
        Return the directory files are saved to: a "jpmaqs-download" subdirectory of
        `self.out_dir`. If `self.out_dir` is already named "jpmaqs-download", it is used
        as-is, so the suffix is never doubled.
        """
        base_dir = Path(self.out_dir)
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
        file_group_id: Optional[str] = None,
        group_id: str = JPMAQS_GROUP_ID,
        start_date: str = JPMAQS_EARLIEST_FILE_DATE,
        end_date: Optional[str] = None,
        convert_metadata_timestamps: bool = True,
        include_unavailable: bool = False,
    ) -> pd.DataFrame:
        """
        Lists all available files for a specific file group within a date range.

        Parameters
        ----------
        file_group_id : Optional[str]
            The identifier for the file group (e.g. "JPMAQS_MACROECONOMIC_BALANCE_SHEETS").
            If None, returns all files for the group_id. Defaults to None.
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
            end_date = utc_now().strftime("%Y%m%d")
        endpoint = "/group/files/available-files"
        params = {
            "group-id": group_id,
            "start-date": start_date,
            "end-date": end_date,
        }
        if file_group_id is not None:
            params["file-group-id"] = file_group_id
        _wait_for_api_call(1)
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

    @cache_decorator(ttl=60)
    def list_available_files_for_all_file_groups(
        self,
        group_id: str = JPMAQS_GROUP_ID,
        start_date: str = JPMAQS_EARLIEST_FILE_DATE,
        end_date: Optional[str] = None,
        include_full_snapshots: bool = True,
        include_delta: bool = True,
        include_metadata: bool = True,
        convert_metadata_timestamps: bool = True,
        include_unavailable: bool = False,
    ) -> pd.DataFrame:
        """
        Fetches and consolidates available files for all relevant file groups.

        Makes a single query for the provider's group, then filters the result locally
        to the requested file types (full snapshots, deltas, metadata).

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
        files_df = self.list_available_files(
            file_group_id=None,
            group_id=group_id,
            start_date=start_date,
            end_date=end_date,
            convert_metadata_timestamps=convert_metadata_timestamps,
            include_unavailable=include_unavailable,
        )

        if files_df.empty:
            return files_df

        if not any([include_full_snapshots, include_delta, include_metadata]):
            raise ValueError(
                "At least one of `include_full_snapshots`, `include_delta`, or "
                "`include_metadata` must be True"
            )

        if "file-name" not in files_df.columns:
            raise InvalidResponseError('Missing "file-name" in response')

        delta_mask = (
            files_df["file-name"]
            .astype(str)
            .str.contains("_DELTA_", case=False, na=False)
        )
        metadata_mask = (
            files_df["file-name"]
            .astype(str)
            .str.contains("_METADATA_", case=False, na=False)
        )
        full_snapshot_mask = ~(delta_mask | metadata_mask)

        mask = pd.Series(False, index=files_df.index)
        if include_full_snapshots:
            mask |= full_snapshot_mask
        if include_delta:
            mask |= delta_mask
        if include_metadata:
            mask |= metadata_mask

        return files_df.loc[mask].copy()

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
            since_datetime = utc_now().strftime("%Y%m%d")
        if to_datetime is None:
            to_datetime = utc_now().strftime("%Y%m%dT%H%M%S")
        validate_dq_timestamp(since_datetime, var_name="since_datetime")
        validate_dq_timestamp(to_datetime, var_name="to_datetime")

        since_ts = pd_to_datetime_compat(since_datetime, utc=True)
        to_ts = pd_to_datetime_compat(to_datetime, utc=True)

        if "T" not in str(since_datetime):
            since_ts = since_ts.normalize()

        if "T" not in str(to_datetime):
            to_ts = (
                to_ts.normalize() + pd.DateOffset(days=1) - pd.Timedelta(1, unit="ns")
            )

        if since_ts > to_ts:
            logger.warning(
                f"`since_datetime` ({since_ts}) is after `to_datetime` ({to_ts}). Swapping values."
            )
            since_ts, to_ts = to_ts, since_ts

        filter_date = since_ts.normalize()

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
        file_group_id: Optional[str] = None,
        file_datetime: Optional[str] = None,
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
            file_group_id, file_datetime = _split_jpmaqs_filename(filename)
        endpoint = "/group/file/availability"
        params = {"file-group-id": file_group_id, "file-datetime": file_datetime}
        payload = self._get(endpoint, params)
        return pd.json_normalize(payload)

    def download_file(
        self,
        file_group_id: Optional[str] = None,
        file_datetime: Optional[str] = None,
        filename: Optional[str] = None,
        overwrite: bool = False,
        chunk_size: Optional[int] = None,
        timeout: Optional[float] = DQ_FILE_API_TIMEOUT,
        max_retries: int = 3,
    ) -> str:
        """
        Downloads a single Parquet file to the client's output directory.

        Call with either (`file_group_id` and `file_datetime`) or `filename`, not both.
        For large files, it automatically uses the `SegmentedFileDownloader` for a
        robust, multi-part download.

        Parameters
        ----------
        file_group_id : str
            The identifier of the file group to download from.
        file_datetime : str
            The timestamp of the file to download.
        filename : Optional[str]
            The full filename, in place of `file_group_id` and `file_datetime`.
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
        out_dir = self._get_save_dir()
        if not ((bool(file_group_id) and bool(file_datetime)) ^ bool(filename)):
            raise ValueError(
                "One of `file_group_id` & `file_datetime`, or `filename` must be provided."
            )
        if not file_group_id:
            file_group_id, file_datetime = _split_jpmaqs_filename(filename)
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
            _delete_jpmaqs_file(file_path)

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

        if is_small_file:
            request_wrapper_stream_bytes_to_disk(**download_args)
        else:
            try:
                request_wrapper_stream_bytes_to_disk(**download_args)
            except Exception as e:
                logger.warning(
                    f"Initial download attempt failed for {file_name}: {e}. "
                    f"Retrying with segmented download..."
                )
                try:
                    SegmentedFileDownloader(
                        **download_args,
                        max_file_retries=max_retries,
                        start_download=True,
                    )
                except Exception as seg_e:
                    logger.error(
                        f"Segmented download also failed for {file_name}: {seg_e}. "
                        f"Cleaning up partial file and raising exception."
                    )
                    _delete_jpmaqs_file(file_path)
                    raise DownloadError(
                        f"Failed to download {file_name} after {max_retries} retries."
                    ) from seg_e

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
            A list of file names or file paths to check for corruption. A file name
            checks every copy of that name in the client's output directory; a file path
            checks only that exact file. If None, scans all downloaded files.

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
                    "All items in `files` must be strings representing file names or "
                    "file paths."
                )
            # the "path" column is resolved, so resolve the given paths to match
            wanted = {str(Path(f).resolve()) for f in files}
            avail_files = avail_files[
                avail_files["file-name"].isin(files)
                | avail_files["path"].astype(str).isin(wanted)
            ]
        files = sorted(set(map(str, avail_files["path"])))
        extensions = sorted(set(Path(f).suffix.rsplit(".", 1)[-1] for f in files))
        return _delete_corrupt_files(files=files, extensions=extensions)

    def download_multiple_files(
        self,
        filenames: List[str],
        overwrite: bool = False,
        max_retries: int = 3,
        n_jobs: Optional[int] = None,
        chunk_size: Optional[int] = None,
        timeout: Optional[float] = DQ_FILE_API_TIMEOUT,
        show_progress: bool = True,
    ) -> List[str]:
        """
        Downloads a list of files concurrently, with a progress bar.

        Parameters
        ----------
        filenames : List[str]
            A list of full filenames to be downloaded.
        overwrite : bool
            If True, overwrites files if they already exist. Default is False.
        max_retries : int
            The number of times to retry downloading the entire list of failed files.
        n_jobs : Optional[int]
            The number of concurrent download jobs, passed to `ThreadPoolExecutor` as
            `max_workers`. If None (default), it picks the worker count itself.
        chunk_size : Optional[int]
            The chunk size for streaming downloads (in bytes).
        timeout : Optional[float]
            The timeout for each download request in seconds.
        show_progress : bool
            If True, displays a progress bar for the downloads.

        Returns
        -------
        List[str]
            The filenames that were downloaded successfully.
        """
        out_dir = self._get_save_dir()
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        start_time = time.time()
        if len(filenames) == 0:
            logger.info("No files to download.")
            return []
        logger.info(f"Starting download of {len(filenames)} files.")
        failed_files = []
        completed_file_names = set()
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
                    completed_file_names.add(fname)
                except KeyboardInterrupt:
                    executor.shutdown(wait=False, cancel_futures=True)
                    raise
                except Exception as e:
                    logger.error(f"Failed to download {fname}: {e}")
                    failed_files.append(fname)
        found_corrupt_files = self.delete_corrupt_files(files=filenames)
        completed_file_names -= set(found_corrupt_files)
        failed_files = sorted(set(failed_files + found_corrupt_files))
        if not failed_files:
            total_time = time.time() - start_time
            logger.info(
                f"Successfully downloaded {len(filenames)} files in {total_time:.2f} seconds."
            )
            return sorted(completed_file_names)  # All downloads successful

        log_msg = f"Failed to download {len(failed_files)} files"
        if max_retries > 0:
            log_msg += f"; retrying {max_retries} more times"
        else:
            log_msg += "; no retries left"
        logger.warning(log_msg)
        if max_retries <= 0:
            logger.error(f"Files failed after retries: {failed_files}")
            raise DownloadError(f"Files failed after retries: {failed_files}")

        retried = self.download_multiple_files(
            filenames=failed_files,
            overwrite=overwrite,
            max_retries=max_retries - 1,
            n_jobs=n_jobs,
            chunk_size=chunk_size,
            timeout=timeout,
            show_progress=show_progress,
        )
        return sorted(completed_file_names | set(retried))

    def download_catalog_file(
        self,
        overwrite: bool = False,
        timeout: Optional[float] = DQ_FILE_API_TIMEOUT,
    ) -> str:
        """Downloads the latest JPMaQS metadata catalog file and returns its path."""
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
        existing_files = self.list_downloaded_files()
        if not overwrite and not existing_files.empty:
            if latest_filename in sorted(existing_files["file-name"]):
                file_path = existing_files[
                    existing_files["file-name"] == latest_filename
                ]["path"].values[0]

        if file_path is None:
            file_path = self.download_file(
                filename=latest_filename,
                overwrite=overwrite,
                timeout=timeout,
            )

        return file_path

    def load_catalog(self) -> pd.DataFrame:
        """Loads the latest catalog file into a DataFrame."""
        return pd.read_parquet(self.download_catalog_file())

    def list_all_tickers(self) -> List[str]:
        """Returns a list of all available tickers in the catalog."""
        return sorted(self.load_catalog()["Ticker"].unique())

    def get_datasets_for_indicators(
        self,
        tickers: Optional[List[str]] = None,
        cids: Optional[List[str]] = None,
        xcats: Optional[List[str]] = None,
        case_sensitive: bool = False,
        as_dict: bool = False,
    ) -> Union[List[str], Dict[str, List[str]]]:
        """
        Returns a list of datasets (or a dictionary mapping datasets to tickers) that
        contain the specified tickers, CIDs, or XCATs.

        Parameters
        ----------
        tickers : Optional[List[str]]
            A list of tickers to search for.
        cids : Optional[List[str]]
            A list of CIDs to search for (must be used in conjunction with `xcats`).
        xcats : Optional[List[str]]
            A list of XCATs to search for (must be used in conjunction with `cids`).
        case_sensitive : bool
            If True, the search will be case-sensitive. Defaults to False.
        as_dict : bool
            If True, returns a dictionary mapping datasets to lists of tickers. Defaults
            to False, which returns a list of datasets.

        Returns
        -------
        Union[List[str], Dict[str, List[str]]]
            A list of datasets or a dictionary mapping datasets to tickers, depending on
            the `as_dict` parameter. Tickers absent from the catalog have no dataset: they
            are warned about, or raise a ValueError if none of them is in the catalog.
        """

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

        catalog_file = self.download_catalog_file()

        catalog_df = pd.read_parquet(catalog_file)
        catalog_df["Dataset"] = (
            catalog_df["Theme"].map(JPMAQS_DATASET_THEME_MAPPING).fillna("Unknown")
        )

        if case_sensitive:
            catalog_df = catalog_df[catalog_df["Ticker"].isin(tickers)]
        else:
            catalog_df = catalog_df[
                catalog_df["Ticker"].str.lower().isin(t.lower() for t in tickers)
            ]

        normalise = str if case_sensitive else str.lower
        found_tickers = {normalise(t) for t in catalog_df["Ticker"]}
        missing_tickers = [t for t in tickers if normalise(t) not in found_tickers]
        if missing_tickers:
            if len(missing_tickers) == len(tickers):
                raise ValueError(
                    "None of the requested tickers are available in the JPMaQS "
                    f"catalog: {_abbreviate_tickers_list(missing_tickers)}."
                )
            logger.warning(
                "Tickers not available in the JPMaQS catalog: %s",
                _abbreviate_tickers_list(missing_tickers),
            )

        unknown_theme_tickers = catalog_df["Dataset"] == "Unknown"
        if unknown_theme_tickers.any():
            logger.warning(
                "Catalog themes missing from `JPMAQS_DATASET_THEME_MAPPING`: %s. "
                "The requested tickers under them belong to no dataset and are skipped.",
                sorted(set(catalog_df.loc[unknown_theme_tickers, "Theme"])),
            )
        datasets_to_keep = sorted(set(catalog_df["Dataset"]) - {"Unknown"})
        if as_dict:
            rdict = {
                dataset: sorted(
                    set(catalog_df[catalog_df["Dataset"] == dataset]["Ticker"])
                )
                for dataset in datasets_to_keep
            }
            return rdict
        return datasets_to_keep

    def list_downloaded_files(self) -> pd.DataFrame:
        """
        Lists the files already downloaded to the output directory.

        Returns
        -------
        pd.DataFrame
            One row per file, with columns "file-name", "file-datetime", "dataset",
            "file-type", "file-timestamp" and "path".
        """
        out_dir = self._get_save_dir()
        col_order = [
            "file-name",
            "file-datetime",
            "dataset",
            "file-type",
            "file-timestamp",
            "path",
        ]
        dfs = [
            _downloaded_files_df(out_dir, file_format=fmt, include_metadata_files=True)
            for fmt in ["parquet", "csv", "json"]
        ]
        # `_downloaded_files_df` returns a narrower frame when it finds nothing, so drop
        # the empties before concatenating: the result then always has `col_order`.
        dfs = [_ for _ in dfs if not _.empty]
        if not dfs:
            return pd.DataFrame(columns=col_order)
        files_df = pd.concat(dfs).reset_index(drop=True)
        return files_df[col_order]

    def _load_metadata_jsons(
        self,
        date: Optional[Union[pd.Timestamp, str]] = None,
        normalize_headers: bool = True,
        skip_download: bool = False,
    ) -> Dict[str, pd.DataFrame]:
        """Load JPMaQS metadata notification JSONs for a date."""
        date: pd.Timestamp = (
            pd_to_datetime_compat(date) if date is not None else utc_now()
        ).normalize()
        if date > utc_now().normalize():
            today_utc = utc_now().normalize()
            raise ValueError(
                "Provided `date` is in the future (UTC). "
                f"Requested: {date.date()}, today (UTC): {today_utc.date()}."
            )
        if not skip_download:
            to_dt = date + pd.offsets.BDay(1) - pd.Timedelta(1, unit="s")
            self.download_files(
                since_datetime=date,
                to_datetime=to_dt,
                include_full_snapshots=False,
                include_delta=False,
                include_metadata=True,
            )
        df = self.list_downloaded_files()
        if df.empty:
            logger.warning(f"No notification files found for date: {date.date()}")
            return {}
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

    def cleanup_old_files(
        self,
        keep_n_days_old_files: int = 0,
        to_datetime: Optional[Union[str, pd.Timestamp]] = None,
        dry_run: bool = False,
        show_progress: bool = True,
        protect_files: Optional[Sequence[str]] = None,
        retain_snap_dates: Optional[Sequence[str]] = None,
    ) -> Optional[pd.DataFrame]:
        """
        Deletes old files from the output directory, judged on the snapshot date in each
        file name rather than when it was downloaded.

        Pass `retain_snap_dates` to keep a number of published snapshots, or `to_datetime`
        to count back in calendar days. With neither, nothing is deleted: the method warns
        and returns what today's date would have removed.

        Parameters
        ----------
        keep_n_days_old_files : Optional[int]
            How much history to keep beyond the latest, in snapshot dates when
            `retain_snap_dates` is given and in calendar days otherwise. At 0 (default),
            only the latest is kept. If None, or -1, no files are deleted.
        to_datetime : Optional[Union[str, pd.Timestamp]]
            The date to count back from, used only when `retain_snap_dates` is not given.
            Counts calendar days, so a weekend consumes the allowance. Time is ignored.
        dry_run : bool
            If True, logs and returns the files that would be deleted without deleting
            them. Default is False.
        show_progress : bool
            If True, displays a progress bar for the deletion process.
        protect_files : Optional[Sequence[str]]
            File names never to delete, whatever their snapshot date.
        retain_snap_dates : Optional[Sequence[str]]
            Snapshot dates ("YYYYMMDD") to keep; files dated before the oldest of them are
            deleted. Counted in published snapshots, so weekends do not consume the
            allowance. Takes precedence over `to_datetime`. An empty sequence deletes
            nothing. A date that is not on disk is warned about.

        Returns
        -------
        Optional[pd.DataFrame]
            The files deleted, or the files that would be deleted in dry-run mode and when
            no retention is given; otherwise None.
        """
        MAX_FILES_LISTED_IN_WARNING: int = 50
        if keep_n_days_old_files is None or keep_n_days_old_files < 0:
            return

        downloaded_df = self.list_downloaded_files()
        if downloaded_df.empty:
            return

        files_df = self._add_snap_date_column(downloaded_df)
        snap_dates = pd.to_datetime(files_df["snap-date"], format="%Y%m%d", utc=True)
        on_disk_dates = set(files_df["snap-date"])
        no_anchor = retain_snap_dates is None and to_datetime is None

        if retain_snap_dates is not None:
            if not retain_snap_dates:
                logger.info("No snapshot dates to retain were given. No files deleted.")
                return
            oldest_kept = min(retain_snap_dates)
            since_dt = pd.Timestamp(oldest_kept, tz="UTC").normalize()
            missing = sorted(set(retain_snap_dates) - on_disk_dates)
            if missing:
                held = sorted(set(retain_snap_dates) & on_disk_dates)
                warnings.warn(
                    f"Asked to keep {keep_n_days_old_files} snapshot date(s) beyond the "
                    f"latest, but {missing} is not in the output directory, so that "
                    f"history is not available locally. Held: {held}. Files dated before "
                    f"{oldest_kept} are still deleted.",
                    stacklevel=2,
                )
        else:
            anchor = utc_now() if to_datetime is None else to_datetime
            since_dt = pd_to_datetime_compat(
                anchor, utc=True
            ).normalize() - pd.Timedelta(keep_n_days_old_files, unit="D")

        files_to_delete = files_df[snap_dates < since_dt]
        if protect_files:
            files_to_delete = files_to_delete[
                ~files_to_delete["file-name"].isin(set(protect_files))
            ]

        if no_anchor:
            # deleting against the wall clock destroys a snapshot that has simply not been
            # replaced yet: at weekends, on holidays, and before the daily publication.
            # Report what it would take out instead of doing it.

            listed = sorted(files_to_delete["file-name"])[:MAX_FILES_LISTED_IN_WARNING]
            unlisted = len(files_to_delete) - len(listed)
            warnings.warn(
                f"`cleanup_old_files` needs `retain_snap_dates` or `to_datetime` to know "
                f"what to keep, so nothing was deleted. Anchored on today "
                f"({since_dt.date()}) it would have deleted {len(files_to_delete)} "
                f"file(s): {listed}"
                + (f" ... and {unlisted} more" if unlisted > 0 else ""),
                stacklevel=2,
            )
            return files_to_delete

        if files_to_delete.empty:
            logger.info(
                "No files found with snap-date before %s. No files deleted.",
                since_dt.date(),
            )
            return

        if dry_run:
            logger.info(
                "Dry run: %d files with snap-date before %s would be deleted.",
                len(files_to_delete),
                since_dt.date(),
            )
            return files_to_delete

        for file_path in tqdm(
            sorted(files_to_delete["path"]),
            desc=f"Cleaning up files older than {since_dt.date()}",
            disable=not show_progress,
        ):
            _delete_jpmaqs_file(file_path)

        # remove any date folders left empty by the deletion
        for parent_dir in {Path(p).parent for p in files_to_delete["path"]}:
            try:
                parent_dir.rmdir()
            except OSError:
                pass

        logger.info(
            "Deleted %d files with snap-date before %s.",
            len(files_to_delete),
            since_dt.date(),
        )

    def _sort_file_for_download_order(self, files_df: pd.DataFrame) -> pd.DataFrame:
        """
        Sorts files for download order based on priority:
        1. Full snapshots
        2. Delta files
        3. Metadata files

        Within each category, files are sorted by 'file-datetime' and then by 'file-name'.

        Parameters
        ----------
        files_df : pd.DataFrame
            DataFrame containing file details.

        Returns
        -------
        pd.DataFrame
            Sorted DataFrame ready for download.
        """
        sorted_df = files_df.copy()
        sorted_df["download-priority"] = sorted_df["file-name"].apply(
            lambda x: 3 if "_METADATA" in x else (2 if "_DELTA" in x else 1)
        )
        sorted_df = sorted_df.sort_values(
            by=["download-priority", "file-datetime", "file-name"],
            ascending=[True, True, True],
        ).reset_index(drop=True)
        return sorted_df

    def _add_snap_date_column(self, files_df: pd.DataFrame) -> pd.DataFrame:
        SNAP_ASSUMED_TIME = "T060000"
        # files published from 06:00 UTC belong to that day's snapshot, earlier to the previous
        SNAP_WINDOW_START = pd.Timedelta(6, unit="h")
        files_df = files_df.copy()
        # tail of the filename: "20260728" or "20260728T080000"
        timestamps = (
            files_df["file-name"]
            .astype(str)
            .str.rsplit("_", n=1)
            .str[-1]
            .str.split(".")
            .str[0]
        )
        snap_dt_col = pd.to_datetime(
            timestamps.where(
                timestamps.str.contains("T"), timestamps + SNAP_ASSUMED_TIME
            ),
            format="%Y%m%dT%H%M%S",
            errors="coerce",
        )
        if snap_dt_col.isna().any():
            bad = files_df.loc[snap_dt_col.isna(), "file-name"].tolist()
            raise ValueError(
                f"Incorrectly named file(s), cannot parse timestamp: {bad}"
            )
        files_df["snap-date"] = (snap_dt_col - SNAP_WINDOW_START).dt.strftime("%Y%m%d")
        return files_df

    def _latest_complete_snapshot_date(self, files_df: pd.DataFrame) -> Optional[str]:
        """
        Returns the latest snap-date whose full-snapshot set covers every JPMaQS theme,
        or None if no snap-date has a complete set.
        """
        if files_df.empty:
            return None
        expected = set(JPMAQS_DATASET_THEME_MAPPING.values())
        checkfile = lambda x: any(s in x for s in ["_DELTA", "_METADATA"])  # noqa: E731
        snap_df = files_df[~files_df["file-name"].apply(checkfile)]
        if "snap-date" not in snap_df.columns:
            snap_df = self._add_snap_date_column(snap_df)
        complete = snap_df.groupby("snap-date")["file-group-id"].agg(expected.issubset)
        complete_dates = complete.index[complete]
        return max(complete_dates) if len(complete_dates) else None

    def download_files(
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
    ) -> List[str]:
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

        Returns
        -------
        List[str]
            The names of the files downloaded, empty if none of the matching files were
            missing from the output directory.
        """
        out_dir = self._get_save_dir()
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        start_time = time.time()

        if since_datetime is None:
            since_datetime = utc_now().strftime("%Y%m%d")

        logger.info(
            f"Starting snapshot download to '{out_dir}' for files since {since_datetime}."
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
            return []

        download_order = self._sort_file_for_download_order(files_df)[
            "file-name"
        ].tolist()
        downloaded_files = self.download_multiple_files(
            filenames=download_order,
            overwrite=overwrite,
            chunk_size=chunk_size,
            timeout=timeout,
            show_progress=show_progress,
        )

        total_time = time.time() - start_time
        logger.info(f"Snapshot download completed in {total_time:.2f} seconds.")
        logger.info(f"Downloaded {len(downloaded_files)} files to '{out_dir}'.")
        return downloaded_files

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
    ) -> List[str]:
        """
        Deprecated method to download files, now superseded by
        :func:`DataQueryFileAPIClient.download_files`. This method will be removed in a
        future release.
        """
        warnings.warn(
            "The `DataQueryFileAPIClient.download_full_snapshot` method is deprecated "
            "and will be removed in a future release. "
            "Please use `DataQueryFileAPIClient.download_files` instead.",
            FutureWarning,
            stacklevel=2,
        )
        return self.download_files(
            since_datetime=since_datetime,
            to_datetime=to_datetime,
            overwrite=overwrite,
            chunk_size=chunk_size,
            timeout=timeout,
            include_full_snapshots=include_full_snapshots,
            include_delta=include_delta,
            include_metadata=include_metadata,
            file_group_ids=file_group_ids,
            show_progress=show_progress,
        )

    def download_latest_files(
        self,
        file_group_ids: Optional[List[str]] = None,
        keep_n_days_old_files: Optional[int] = 0,
        overwrite: bool = False,
        chunk_size: Optional[int] = None,
        timeout: Optional[float] = DQ_FILE_API_TIMEOUT,
        show_progress: bool = True,
    ) -> List[str]:
        """
        Downloads the files for the latest snapshot date (full snapshots, deltas and
        metadata), optionally restricted to `file_group_ids`. Files already on disk are
        skipped unless `overwrite` is set, and older files are removed according to
        `keep_n_days_old_files`.

        Parameters
        ----------
        file_group_ids : Optional[List[str]]
            Restrict the download to these file groups, e.g. "JPMAQS_GENERIC_RETURNS".
            Matching is by prefix, so a group's delta files are downloaded alongside its
            full snapshot. Only the groups listed are downloaded, so list the metadata
            catalog group if the files are to be loaded. If None (default), all available
            file groups are downloaded, the catalog among them.
        keep_n_days_old_files : int
            How many published snapshots to keep besides the latest, after the download.
            At 0 (the default), only the latest day's files are kept: its full snapshots,
            deltas and metadata. Counted in published snapshots, not calendar days, so
            weekends do not consume the allowance. If None or negative, no files are
            deleted.
            See :func:`DataQueryFileAPIClient.cleanup_old_files` for more details.
        overwrite : bool
            If True, overwrites files if they already exist. Default is False.
        chunk_size : Optional[int]
            The chunk size for streaming downloads (in bytes).
        timeout : Optional[float]
            The timeout for each download request in seconds.
        show_progress : bool
            If True, displays a progress bar for downloads.

        Returns
        -------
        List[str]
            The list of files downloaded.
        """
        files_df = self._add_snap_date_column(
            self.filter_available_files_by_datetime(
                since_datetime=JPMAQS_EARLIEST_FILE_DATE,
                include_delta=True,
                include_full_snapshots=True,
                include_metadata=True,
            )
        )
        # resolve and filter by the latest snapshot date across all groups
        latest_snapshot_date = self._latest_complete_snapshot_date(files_df)
        if latest_snapshot_date is None:
            logger.warning("No snapshot available to download.")
            return []

        files_for_snapshot = files_df[files_df["snap-date"] == latest_snapshot_date]
        # file_group_ids=None means "all groups", which includes the metadata catalog
        if file_group_ids is not None:
            # an empty or unmatched selection stays empty: nothing was asked for
            files_for_snapshot = files_for_snapshot[
                files_for_snapshot["file-group-id"].str.startswith(
                    tuple(file_group_ids)
                )
            ]
        # every file this snapshot consists of, whether or not this call has to fetch it.
        # cleanup must never prune these, so capture them before the dedup below narrows
        # the frame to what is missing.
        required_files = files_for_snapshot["file-name"].tolist()
        if not required_files:
            # the request selected no files at all, e.g. file_group_ids matched nothing.
            # Nothing is needed, so nothing may be pruned on the strength of it either.
            logger.info(
                f"No files match the requested file groups for the latest snapshot "
                f"dated {latest_snapshot_date}."
            )
            return []

        # Retention is counted in snapshot dates that exist upstream, not calendar days,
        # so it steps over weekends and holidays. Any date carrying files counts, whether
        # it is a business day or not.
        retain_snap_dates = None
        if keep_n_days_old_files is not None and keep_n_days_old_files >= 0:
            published = sorted(
                {d for d in files_df["snap-date"] if d <= latest_snapshot_date},
                reverse=True,
            )
            retain_snap_dates = published[: keep_n_days_old_files + 1]

        files_to_download = files_for_snapshot
        avail_files_df = self.list_downloaded_files()
        if not avail_files_df.empty and not overwrite:
            files_to_download = files_to_download[
                ~files_to_download["file-name"].isin(avail_files_df["file-name"])
            ]
        download_order = self._sort_file_for_download_order(files_to_download)[
            "file-name"
        ].tolist()
        if not download_order:
            logger.info(
                f"No new files to download for the latest snapshot dated "
                f"{latest_snapshot_date}."
            )
            # every required file is already on disk, so the snapshot is complete and
            # older files can go. `required_files` is non-empty here, checked above.
            self.cleanup_old_files(
                keep_n_days_old_files=keep_n_days_old_files,
                to_datetime=latest_snapshot_date,
                protect_files=required_files,
                retain_snap_dates=retain_snap_dates,
                show_progress=show_progress,
            )
            return []

        downloaded_files = self.download_multiple_files(
            filenames=download_order,
            overwrite=overwrite,
            chunk_size=chunk_size,
            timeout=timeout,
            show_progress=show_progress,
        )
        logger.info(
            f"Downloaded {len(downloaded_files)} files for the latest snapshot "
            f"dated {latest_snapshot_date}."
        )

        self.cleanup_old_files(
            keep_n_days_old_files=keep_n_days_old_files,
            to_datetime=latest_snapshot_date,
            protect_files=required_files,
            retain_snap_dates=retain_snap_dates,
            show_progress=show_progress,
        )
        return downloaded_files

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
        include_delta_files: bool = True,
        delta_treatment: str = "latest",
        show_progress: bool = True,
        overwrite: bool = False,
        keep_n_days_old_files: Optional[int] = 0,
        include_source_file: bool = False,
        dropna: bool = True,
        datasets: Optional[List[str]] = None,
        categorical_source_file_column: bool = True,
        suppress_warnings: bool = False,
    ) -> Union[pd.DataFrame, pl.DataFrame, pl.LazyFrame]:
        """
        Downloads data for the specified `tickers`, `cids`, or `xcats` and returns it as
        a DataFrame for the given date range.

        Every file of the latest snapshot for the relevant datasets is downloaded to disk
        (full snapshots, delta files, metadata files and the metadata catalog), then the
        requested data is loaded and filtered from them. The files on disk are unmodified;
        the conversion to `dataframe_format` (e.g. "qdf") is applied lazily on read.

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
            If None, data is returned from the earliest available date. A range given
            the wrong way round is swapped, with a warning, rather than returning
            nothing.
        end_date : Optional[str]
            The end date for the returned data in the ISO format "YYYY-MM-DD".
            If None, data is returned up to the latest available date. Equal bounds
            return that single date.
        dataframe_format : str
            The format of the returned DataFrame, default is "qdf". Options are:
                - "qdf": QuantamentalDataFrame with columns (real_date, cid, xcat, metric1, ..., metricN)
                - "tickers": QuantamentalDataFrame with columns (real_date, ticker, metric1, ..., metricN)
                - "wide": QuantamentalDataFrame with columns (real_date, ticker1, ticker2, ..., tickerN)
                    for a single metric. Cannot be combined with `include_source_file`.
        dataframe_type : str
            The type of DataFrame to return. Options are "pandas" for a pandas DataFrame,
            "polars" for a polars DataFrame, or "polars-lazy" for a polars LazyFrame.
            Default is "pandas".
        categorical_dataframe : bool
            If True and `dataframe_type` is "pandas", the returned DataFrame will use
            categorical dtypes for object columns. Ignored for `dataframe_format="wide"`.
            Default is True.
        include_delta_files : bool
            Whether delta files are folded in when the data is read. Default is True.
            This controls reading only - delta files are always downloaded, as the
            local snapshot would otherwise be incomplete for any later load.
        delta_treatment : str
            How rows restated by a delta file are resolved, per (ticker, real_date):
            "latest" (the default) keeps the row with the newest `last_updated`,
            "earliest" the oldest, and "all" keeps every version. "all" requires
            `dropna=False`, and cannot be used with `dataframe_format="wide"`.
        show_progress : bool
            If True, displays a progress bar during downloads. Default is True.
        overwrite : bool
            If True, overwrites files if they already exist. Default is False.
        keep_n_days_old_files : int
            How many published snapshots to keep besides the latest, after the download.
            At 0 (the default), only the latest day's files are kept: its full snapshots,
            deltas and metadata. Counted in published snapshots, not calendar days, so
            weekends do not consume the allowance. If None or negative, no files are
            deleted.
            See :func:`DataQueryFileAPIClient.cleanup_old_files` for more details.
        include_source_file : bool
            If True, the returned DataFrame will include a column indicating the source
            file from which each row was loaded. As this loads a large amount of string
            data, the `"source_file"` column is categorical by default. If False
            (default), no source file information is included.

            Default is False.
        dropna : bool
            If True (the default), no null reaches the output: a row is kept only where
            every requested metric is populated. That drops expiry rows - JPMaQS withdraws
            a (ticker, real_date) observation by publishing one whose metrics are all null,
            which under `delta_treatment="latest"` is the row that survives - and equally
            any other row null in a requested metric. Scoped to `metrics`, so a row null
            only in a metric you did not ask for is kept. Applied after delta resolution,
            so an expiry still supersedes the row it withdraws before being dropped itself.
            Pass False to keep every null, and see the `last_updated` each expiry landed at.
        datasets : Optional[List[str]]
            Restrict the download and the load to these JPMaQS datasets, e.g.
            `["JPMAQS_MACROECONOMIC_TRENDS"]`. The datasets holding the requested
            indicators are derived from the catalog, and this narrows that set rather than
            replacing it; naming only datasets that hold none of them raises. If None
            (default), every dataset holding a requested indicator is used.
        categorical_source_file_column : bool
            If True (default), the `"source_file"` column added by `include_source_file`
            uses a categorical dtype, which is much cheaper than storing the file name as a
            string on every row. Ignored unless `include_source_file=True`.
        suppress_warnings : bool
            If True, silences warnings from this function. Default is False.

        Returns
        -------
        Union[pd.DataFrame, pl.DataFrame, pl.LazyFrame]
            A DataFrame containing the requested data.
        """
        out_dir = self._get_save_dir()
        with _suppressed_warnings(suppress_warnings):
            datasets_to_download = self.get_datasets_for_indicators(
                tickers=tickers, cids=cids, xcats=xcats
            )
            if datasets is not None:
                # narrow rather than replace: downloading a dataset that holds none of
                # the requested indicators would only fetch files the load then ignores
                narrowed = [d for d in datasets_to_download if d in set(datasets)]
                if not narrowed:
                    raise ValueError(
                        f"`datasets={sorted(set(datasets))}` holds none of the requested "
                        f"indicators, which live in {sorted(datasets_to_download)}."
                    )
                datasets_to_download = narrowed
            self.download_latest_files(
                overwrite=overwrite,
                show_progress=show_progress,
                keep_n_days_old_files=keep_n_days_old_files,
                file_group_ids=datasets_to_download,
            )
            catalog_path = Path(self.download_catalog_file())
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
                include_delta_files=include_delta_files,
                catalog_path=catalog_path,
                include_source_file=include_source_file,
                delta_treatment=delta_treatment,
                dropna=dropna,
                categorical_source_file_column=categorical_source_file_column,
            )


def _pd_to_datetime_compat(ts: str, utc: bool) -> pd.Timestamp:
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


@overload
def pd_to_datetime_compat(
    ts: Union[str, pd.Timestamp],
    format: str = "mixed",
    utc: bool = True,
) -> pd.Timestamp: ...


@overload
def pd_to_datetime_compat(
    ts: pd.Series,
    format: str = "mixed",
    utc: bool = True,
) -> pd.Series: ...


def pd_to_datetime_compat(
    ts: Union[str, pd.Timestamp, datetime.date, pd.Series],
    format: str = "mixed",
    utc: bool = True,
) -> Union[pd.Timestamp, pd.Series]:
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


@cache_decorator(ttl=60)
def large_delta_file_datetimes(as_str: bool = True) -> List[str]:
    """
    Plausible file datetimes for large delta files, which are typically
    generated at the end of each month and on business month ends, with timestamps of
    end-of-day (23:59:59).
    """
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
    dt_list = sorted(map(pd.Timestamp, dt_list))
    if not as_str:
        return dt_list

    return [d.strftime("%Y%m%dT%H%M%S") for d in dt_list]


def _ordered_date_range(
    start_date: Optional[Union[str, pd.Timestamp]],
    end_date: Optional[Union[str, pd.Timestamp]],
) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    """
    Warn and swap a reversed range, so `start_date` is always the earlier bound.

    Matches `_filter_available_files_by_datetime`, which does the same for file
    datetimes; without it a reversed range quietly returned no rows at all.
    """
    if start_date is None or end_date is None:
        return start_date, end_date
    start_ts = pd_to_datetime_compat(start_date)
    end_ts = pd_to_datetime_compat(end_date)
    if start_ts > end_ts:
        logger.warning(
            f"`start_date` ({start_ts.date()}) is after `end_date` "
            f"({end_ts.date()}). Swapping values."
        )
        return end_ts, start_ts
    return start_date, end_date


@cache_decorator(ttl=60)
def _read_catalog(catalog_path: Union[str, Path]) -> pd.DataFrame:
    """Read the JPMaQS metadata catalog. Cached: callers must not mutate the result."""
    return pd.read_parquet(Path(catalog_path).resolve())


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
            if _delete_jpmaqs_file(file_path):
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
        max_concurrent_downloads: Optional[int] = None,
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
    delta_treatment: str = "latest",
    dropna: bool = True,
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

    if not tickers and not cids and not xcats:
        # an empty selection is a request for every ticker on disk, which is far too
        # much to load by accident. Loading everything has to be asked for explicitly
        raise ValueError(
            "At least one of `tickers`, `cids`, or `xcats` must be set. To load "
            "every ticker, pass them explicitly: "
            "`tickers=client.list_all_tickers()`."
        )

    for param, name in [
        (start_date, "start_date"),
        (end_date, "end_date"),
    ]:
        if param is not None and not isinstance(param, (str, pd.Timestamp)):
            raise ValueError(f"`{name}` must be a string or pandas Timestamp.")

    if dataframe_format not in ["qdf", "wide", "tickers"]:
        raise ValueError("`dataframe_format` must be one of 'qdf', 'wide', 'tickers'.")

    if dataframe_format == "wide" and metrics is not None and len(metrics) != 1:
        raise ValueError(
            "`dataframe_format='wide'` only supports a single metric (typically 'value')."
            f" Please provide a single metric in the `metrics` parameter; got {metrics}."
        )

    if dataframe_type not in ["pandas", "polars", "polars-lazy"]:
        raise ValueError(
            "`dataframe_type` must be one of 'pandas', 'polars', 'polars-lazy'."
        )
    if not isinstance(categorical_dataframe, bool):
        raise ValueError("`categorical_dataframe` must be a boolean.")

    if delta_treatment not in DELTA_TREATMENTS:
        raise ValueError(_delta_treatment_error())
    if not isinstance(dropna, bool):
        raise ValueError("`dropna` must be a boolean.")
    # checked before the `dropna` pairing below, so that `wide` + "all" reports the
    # incompatibility that cannot be worked around first
    if dataframe_format == "wide" and delta_treatment == "all":
        raise ValueError(
            "`dataframe_format='wide'` cannot be combined with "
            "`delta_treatment='all'`: a wide frame holds one cell per "
            "(ticker, real_date), which cannot hold several restatements of it. Use "
            "'latest' or 'earliest', or a long format ('qdf' or 'tickers')."
        )
    if delta_treatment == "all" and dropna:
        raise ValueError(
            "`dropna=True` cannot be combined with `delta_treatment='all'`: dropping "
            "the expiry rows would leave the very rows they expire, so an expired "
            "observation would reappear with its stale value. Pass `dropna=False` to "
            "keep the full restatement history."
        )


def _split_jpmaqs_filename(filename: str) -> Tuple[str, str]:
    """Split "JPMAQS_X_20250501.parquet" into ("JPMAQS_X", "20250501")."""
    try:
        file_group_id, datetime_with_ext = filename.rsplit("_", 1)
    except ValueError:
        raise ValueError(f"Invalid filename format: {filename}")
    return file_group_id, datetime_with_ext.split(".")[0]


def _is_jpmaqs_file(path: Union[str, Path]) -> bool:
    """A downloaded file is a JPMaQS file iff it is named JPMAQS_....{parquet,csv,json}."""
    p = Path(path)
    return p.name.startswith("JPMAQS_") and p.suffix.lower() in (
        ".parquet",
        ".csv",
        ".json",
    )


def _delete_jpmaqs_file(path: Union[str, Path]) -> bool:
    """
    Central deletion for downloaded files. Deletes `path` only if it is a JPMaQS file
    (see `_is_jpmaqs_file`); otherwise it warns and leaves the file untouched. Returns
    True if the file was deleted.
    """
    path = Path(path)
    if not _is_jpmaqs_file(path):
        logger.warning(f"Refusing to delete non-JPMaQS file: {path}")
        return False
    path.unlink(missing_ok=True)
    return True


def _list_downloaded_files(files_dir: Path, file_format: str = "parquet") -> List[Path]:
    files_dir = Path(files_dir)
    assert files_dir.is_dir(), f"No such directory: {files_dir}"
    if file_format not in ["parquet", "csv", "json"]:
        raise ValueError("`file_format` must be one of 'parquet', 'csv', or 'json'.")
    files = sorted(files_dir.glob(f"**/*.{file_format}"))
    for f in files:
        if not _is_jpmaqs_file(f):
            logger.warning(
                f"File is not a JPMaQS file and is not meant to be here: {f}"
            )
    return [f for f in files if _is_jpmaqs_file(f)]


def _downloaded_files_df(
    files_dir: Path,
    file_format: str = "parquet",
    include_metadata_files: bool = False,
) -> pd.DataFrame:
    if not Path(files_dir).is_dir():
        return pd.DataFrame(columns=["path", "file-name", "file-type", "dataset"])
    files_list = _list_downloaded_files(files_dir, file_format)
    df = pd.DataFrame({"path": files_list})
    if df.empty:
        return df
    df["path"] = df["path"].apply(lambda x: Path(x).resolve())
    df["file-name"] = df["path"].apply(lambda x: Path(x).name)
    if not include_metadata_files:
        df = df[~df["file-name"].str.contains("_METADATA")].copy()
    df["file-type"] = df["path"].apply(lambda x: Path(x).suffix.split(".")[-1])

    df["dataset"] = df["file-name"].apply(
        lambda x: str(x).split(".")[0].rsplit("_", 1)[0]
    )

    df["file-datetime"] = df["file-name"].apply(
        lambda x: str(x).split(".")[0].rsplit("_", 1)[-1]
    )

    def _dt_or_none(x):
        try:
            return pd_to_datetime_compat(x)
        except Exception:
            return None

    df["file-timestamp"] = df["file-datetime"].apply(lambda x: _dt_or_none(x))
    df = df[~df["file-timestamp"].isnull()]
    df = df.reset_index(drop=True)
    return df


def _filter_to_latest_files(
    files_df: pd.DataFrame,
    include_delta_files: bool = True,
) -> pd.DataFrame:
    if files_df.empty:
        return files_df

    if not include_delta_files:
        files_df = files_df[~files_df["file-name"].str.contains("_DELTA")].copy()

    # the snapshot date comes from the date-only filenames; delta/metadata files carry a
    # "T" timestamp and cannot define it
    snapshot_datetimes = files_df[~files_df["file-datetime"].str.contains("T")][
        "file-datetime"
    ]
    if snapshot_datetimes.empty:
        raise ValueError(
            "No full-snapshot files found in the output directory (only timestamped "
            "delta/metadata files), so the latest snapshot date cannot be determined. "
            "Download a full snapshot first."
        )
    latest_timestamp = pd_to_datetime_compat(snapshot_datetimes.max())
    latest_files: pd.DataFrame = files_df[
        files_df["file-timestamp"] >= latest_timestamp
    ].copy()

    latest_files["e-dataset"] = latest_files["dataset"].apply(
        lambda x: str(x).replace("_DELTA", "")
    )

    return latest_files.sort_values(
        ["dataset", "file-timestamp", "file-name"]
    ).reset_index(drop=True)


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
    include_delta_files: bool = True,
    delta_treatment: str = "latest",
    catalog_path: Optional[Union[str, Path]] = None,
    include_source_file: bool = False,
    categorical_source_file_column: bool = True,
    dropna: bool = True,
) -> Union[pd.DataFrame, pl.DataFrame, pl.LazyFrame]:
    """
    Loads previously downloaded JPMaQS files into a single DataFrame.

    Scans the files of the latest snapshot found under `files_dir`, applies the ticker,
    date and metric filters while scanning, and only then materialises the result. The
    files on disk are never modified. Delta files are folded into the snapshot unless
    `include_delta_files=False`. Metadata files are never read as data: the catalog is
    passed separately as `catalog_path`, and the notification JSONs are served by
    :func:`DataQueryFileAPIClient.get_missing_data_notifications` and
    :func:`DataQueryFileAPIClient.get_revisions_notifications`.

    At least one of `tickers`, `cids` or `xcats` is required. To load every ticker,
    ask for them explicitly with
    :func:`DataQueryFileAPIClient.list_all_tickers`, rather than leaving the
    selection empty.

    `catalog_path` is required, as the catalog resolves which dataset holds which ticker.
    Obtain one from :func:`DataQueryFileAPIClient.download_catalog_file`.

    `delta_treatment` resolves rows a delta file restated, per (ticker, real_date):
    "latest" (default) keeps the newest `last_updated`; "earliest" the first print, which
    skips expiry rows, a withdrawal not being a print; "all" keeps every version.

    `dropna=True` (default) means no null reaches the output: a row survives only where
    every requested metric is populated. That covers expiry rows - an observation is
    withdrawn by publishing one whose metrics are all null - and any other row null in a
    requested metric. It is applied after delta resolution, so an expiry still supersedes
    the row it withdraws. "all" requires `dropna=False`, as dropping the nulls while keeping
    the rows they expire would resurrect stale values, and cannot be combined with
    `dataframe_format="wide"`, which has only one cell per (ticker, real_date).

    `dataframe_format` chooses the shape:

    - "qdf" (default): columns (real_date, cid, xcat, <metrics>)
    - "tickers": columns (real_date, ticker, <metrics>)
    - "wide": one column per ticker, for a single metric, so `metrics` must name exactly
      one. `real_date` is the index for "pandas" but a column for the polars types, and
      the format cannot be combined with `include_source_file`.

    `dataframe_type` chooses the library: "pandas" (default), "polars", or "polars-lazy",
    which returns the LazyFrame uncollected. `file_format="csv"` raises
    `NotImplementedError`.

    Returns
    -------
    Union[pd.DataFrame, pl.DataFrame, pl.LazyFrame]
        The filtered data, in the requested format and type.
    """
    files_dir = Path(files_dir)
    if (not metrics) or (metrics == "all") or ("all" in metrics):
        metrics = list(JPMAQS_METRICS)
    if catalog_path is None:
        raise ValueError(
            "`catalog_path` is required. Use "
            "`DataQueryFileAPIClient.download_catalog_file()` to obtain one."
        )
    for flag, flag_name in [
        (include_source_file, "include_source_file"),
        (categorical_source_file_column, "categorical_source_file_column"),
    ]:
        if not isinstance(flag, bool):
            raise ValueError(f"`{flag_name}` must be a boolean.")

    _check_lazy_load_inputs(
        files_dir=files_dir,
        file_format=file_format,
        tickers=tickers,
        cids=cids,
        xcats=xcats,
        metrics=metrics,
        start_date=start_date,
        end_date=end_date,
        dataframe_format=dataframe_format,
        dataframe_type=dataframe_type,
        categorical_dataframe=categorical_dataframe,
        delta_treatment=delta_treatment,
        dropna=dropna,
    )
    start_date, end_date = _ordered_date_range(start_date, end_date)

    available_files_df: pd.DataFrame = _downloaded_files_df(
        files_dir=files_dir,
        file_format=file_format,
        # metadata files are not quantamental data; the notification JSONs are served
        # by `get_missing_data_notifications` and `get_revisions_notifications`
        include_metadata_files=False,
    )
    available_files_df: pd.DataFrame = _filter_to_latest_files(
        files_df=available_files_df,
        include_delta_files=include_delta_files,
    )
    if datasets:
        available_files_df = available_files_df.loc[
            available_files_df["e-dataset"].isin(datasets)
        ]

    # copy: `+=` and `.remove()` below must not mutate the caller's lists
    tickers = list(tickers or [])
    if cids:
        tickers += [f"{c}_{x}" for c in cids for x in xcats]
    if "source_file" in metrics:
        warnings.warn(
            'Please use `include_source_file=True` instead of including "source_file" in metrics.'
        )
        include_source_file = True
        # fall back to the default metrics if "source_file" was the only one requested
        metrics = [m for m in metrics if m != "source_file"] or list(JPMAQS_METRICS)

    if dataframe_format == "wide" and include_source_file:
        raise ValueError(
            "`dataframe_format='wide'` cannot be combined with `include_source_file`, "
            "as a wide frame holds a single metric per column."
        )

    catalog_df = _read_catalog(catalog_path)
    valid_tickers = sorted(
        catalog_df[catalog_df["Ticker"].str.lower().isin([t.lower() for t in tickers])][
            "Ticker"
        ].tolist()
    )
    matched = {t.lower() for t in valid_tickers}
    unknown_tickers = sorted({t for t in tickers if t.lower() not in matched})
    if unknown_tickers:
        warnings.warn(
            f"The following tickers are not in the JPMaQS catalog and will be "
            f"ignored: {unknown_tickers}",
            stacklevel=2,
        )
    if not valid_tickers:
        raise ValueError(
            f"None of the requested tickers are in the JPMaQS catalog: "
            f"{sorted(tickers)}"
        )

    paths = sorted(available_files_df["path"])
    lf: pl.LazyFrame = _lazy_load_filtered_parquets(
        paths=paths,
        tickers=valid_tickers,
        start_date=start_date,
        end_date=end_date,
        return_qdf=(dataframe_format == "qdf"),
        catalog_path=catalog_path,
        include_source_file=include_source_file,
        categorical_source_file_column=categorical_source_file_column,
        delta_treatment=delta_treatment,
        dropna=dropna,
        metrics=metrics,
    )
    if include_source_file:
        metrics = metrics + ["source_file"]
    if set(metrics) != set(JPMAQS_METRICS):
        cols_to_keep = {"real_date", "cid", "xcat", "ticker", *metrics}
        # select in the frame's own column order, so the output order does not
        # depend on whether this projection runs
        names = (
            lf.collect_schema().names()
            if PYTHON_3_8_OR_LATER
            else list(lf.schema.keys())
        )
        lf = lf.select([pl.col(c) for c in names if c in cols_to_keep])

    cat_cols = ["cid", "xcat", "ticker"]
    if dataframe_type in ["polars", "polars-lazy"]:
        if dataframe_format == "wide":
            lf = lf.pivot(
                "ticker",
                on_columns=valid_tickers,
                index="real_date",
                values=metrics[0],
                aggregate_function=None,
                separator=";",
            ).sort("real_date")
        if categorical_dataframe and dataframe_format != "wide":
            cols = None
            if PYTHON_3_8_OR_LATER:
                cols = [c for c in cat_cols if c in lf.collect_schema().names()]
            else:
                cols = [c for c in cat_cols if c in lf.schema.keys()]
            if cols:
                lf = lf.with_columns([pl.col(c).cast(pl.Categorical) for c in cols])
        if dataframe_type == "polars-lazy":
            return lf
        return _collect_naming_paths(lf, paths)
    if dataframe_type == "pandas":
        df = _collect_naming_paths(lf, paths).to_pandas()
        if dataframe_format == "wide":
            # reindex so the column set matches the polars path: one column per
            # requested ticker, even where the data holds no rows for it
            wide = df.pivot(
                index="real_date", columns="ticker", values=metrics[0]
            ).reindex(columns=valid_tickers)
            # gated, so `dropna=False` keeps the all-null dates the polars path keeps
            return wide.dropna(how="all") if dropna else wide
        if categorical_dataframe and dataframe_format != "wide":
            cols = [c for c in cat_cols if c in df.columns]
            if cols:
                df[cols] = df[cols].astype("category")
        return df

    raise ValueError("Unknown dataframe type")


def _filter_lazy_frame_by_tickers(
    lf: pl.LazyFrame,
    tickers: Sequence[str],
    start_date: Optional[Union[str, pd.Timestamp]],
    end_date: Optional[Union[str, pd.Timestamp]],
) -> pl.LazyFrame:
    tickers_list = [t for t in tickers if t]
    if tickers_list:
        lf = lf.filter(pl.col("ticker").is_in(tickers_list))
    if start_date:
        start_date = pd_to_datetime_compat(start_date).strftime("%Y-%m-%d")
        lf = lf.filter(pl.col("real_date") >= pl.lit(start_date).str.to_date())
    if end_date:
        end_date = pd_to_datetime_compat(end_date).strftime("%Y-%m-%d")
        lf = lf.filter(pl.col("real_date") <= pl.lit(end_date).str.to_date())
    return lf


EXPECTED_JPMAQS_PARQUET_SCHEMA = {
    "real_date": pl.Date,
    "ticker": pl.String,
    "value": pl.Float64,
    "eop_lag": pl.Float64,
    "mop_lag": pl.Float64,
    "grading": pl.Float64,
    "last_updated": pl.Datetime,
}

DELTA_TREATMENTS = ("latest", "earliest", "all")

# an expiry row nulls every one of these; `last_updated` records when it was withdrawn
JPMAQS_VALUE_METRICS = [c for c in JPMAQS_METRICS if c != "last_updated"]


def _to_output_schema(
    lf: pl.LazyFrame,
    want_qdf: bool,
    include_source_file: bool = False,
) -> pl.LazyFrame:
    expc_schema = EXPECTED_JPMAQS_PARQUET_SCHEMA.copy()
    if include_source_file:
        expc_schema["source_file"] = pl.Categorical

    if want_qdf:
        # substitute cid/xcat *at* ticker's position, so qdf output keeps the
        # conventional real_date, cid, xcat, <metrics> ordering
        items = list(expc_schema.items())
        idx = [k for k, _ in items].index("ticker")
        items[idx : idx + 1] = [("cid", pl.String), ("xcat", pl.String)]
        expc_schema = dict(items)
        ticker_col_split = pl.col("ticker").str.splitn("_", 2)
        lf = lf.with_columns(
            cid=ticker_col_split.struct.field("field_0"),
            xcat=ticker_col_split.struct.field("field_1"),
        )

    keep_cols = list(expc_schema.keys())
    curr_cols = lf.collect_schema().keys() if PYTHON_3_8_OR_LATER else lf.schema.keys()
    missing_cols = [c for c in keep_cols if c not in curr_cols]
    if missing_cols:
        raise ValueError(
            f"Missing expected columns in LazyFrame: {missing_cols}. "
            f"Current columns: {sorted(curr_cols)}"
        )
    lf = lf.select([pl.col(c) for c in keep_cols])
    return lf


def _scan_check_and_cast_single_parquet(
    path: str,
    include_source_file: bool = False,
    categorical_source_file_column: bool = True,
) -> pl.LazyFrame:
    """Scan one parquet and normalise it to `EXPECTED_JPMAQS_PARQUET_SCHEMA`."""
    lf = pl.scan_parquet(path)
    schema = dict(lf.collect_schema()) if PYTHON_3_8_OR_LATER else dict(lf.schema)
    if schema.get("grading", None) == pl.String:
        lf = lf.with_columns(pl.col("grading").cast(pl.Float64))
    if include_source_file:
        if "source_file" in schema:
            raise ValueError(
                f"The 'source_file' column already exists in `{path}`. JPMaQS files "
                "never carry this column, so it was added locally; re-download the file "
                "or load it with `include_source_file=False`."
            )
        pth_str = Path(path).name.rsplit(".", 1)[0]
        assert pth_str, f"Invalid path: {path}"
        lf = lf.with_columns(
            pl.lit(pth_str)
            .alias("source_file")
            .cast(pl.Categorical if categorical_source_file_column else pl.String)
        )

    if ("cid" in schema) != ("xcat" in schema):
        raise ValueError(
            "Parquet file must have both 'cid' and 'xcat' columns or neither."
        )

    # this conversion is later undone in _to_output_schema() if want_qdf is True
    # however, the cost of this conversion is small compared to the cost of maintaing
    # a dual read schema and offers fewer code paths and less complexity.
    # this is also why reading QDF saved by older version of the package is supported
    # for now, but the QDF write path has been removed.
    if "cid" in schema:
        err_str = (
            f"A modified schema was detected for file `{path}`. "
            "Please update the version of the Macrosynergy Package used. "
            "Modifying the schema of downloaded files will not be supported in future versions of the Macrosynergy Package."
        )
        warnings.warn(err_str)
        if "ticker" not in schema:
            lf = lf.with_columns(
                ticker=pl.concat_str([pl.col("cid"), pl.lit("_"), pl.col("xcat")])
            )
        lf = lf.drop(["cid", "xcat"])

    # if now missing the ticker or real_date columns, raise an error
    schema = dict(lf.collect_schema()) if PYTHON_3_8_OR_LATER else dict(lf.schema)
    must_have_cols = ["real_date", "ticker"]
    for col in must_have_cols:
        if col not in schema:
            raise ValueError(
                f"Parquet file {path} is missing required column: '{col}'."
            )

    for col, expected_type in EXPECTED_JPMAQS_PARQUET_SCHEMA.items():
        if col in schema:
            if schema[col] == expected_type:
                continue
            if schema[col] == pl.String and expected_type == pl.Datetime:
                # cast() from String is deprecated and removed in polars 2.0
                lf = lf.with_columns(pl.col(col).str.to_datetime())
            else:
                lf = lf.with_columns(pl.col(col).cast(expected_type))
        else:
            lf = lf.with_columns(pl.lit(None).cast(expected_type).alias(col))
    return lf


def _scan_and_prepare_single_parquet(
    path: str,
    tickers: Sequence[str],
    start_date: Optional[Union[str, pd.Timestamp]],
    end_date: Optional[Union[str, pd.Timestamp]],
    return_qdf: bool,
    include_source_file: bool = False,
    categorical_source_file_column: bool = True,
) -> pl.LazyFrame:
    lf = _scan_check_and_cast_single_parquet(
        path=path,
        include_source_file=include_source_file,
        categorical_source_file_column=categorical_source_file_column,
    )

    lf = _filter_lazy_frame_by_tickers(
        lf=lf,
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
    )
    lf = _to_output_schema(
        lf=lf, want_qdf=return_qdf, include_source_file=include_source_file
    )
    return lf


def _lazy_load_filtered_parquets(
    paths: List[str],
    tickers: List[str],
    start_date: Optional[Union[str, pd.Timestamp]],
    end_date: Optional[Union[str, pd.Timestamp]],
    catalog_path: Union[str, Path],
    return_qdf: bool = True,
    delta_treatment: str = "latest",
    include_source_file: bool = False,
    categorical_source_file_column: bool = True,
    dropna: bool = True,
    metrics: Optional[List[str]] = None,
) -> pl.LazyFrame:
    if not paths:
        raise ValueError("No paths provided")

    ticker_lazyframes_df = build_filtered_lazy_frames_df(
        paths=paths,
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        catalog_path=catalog_path,
        return_qdf=return_qdf,
        include_source_file=include_source_file,
        categorical_source_file_column=categorical_source_file_column,
    )
    if ticker_lazyframes_df.empty or ticker_lazyframes_df["lazyframe"].isna().any():
        raise ValueError(
            "No data could be loaded for the requested tickers from the files on disk."
        )

    for _, row in ticker_lazyframes_df.iterrows():
        lf = row["lazyframe"]
        # dedup restated rows for both output shapes; `return_qdf` selects the key columns
        lf = _apply_delta_treatment(
            lf=lf,
            delta_treatment=delta_treatment,
            return_qdf=return_qdf,
        )
        ticker_lazyframes_df.loc[_, "lazyframe"] = lf
    out_lf: pl.LazyFrame = pl.concat(
        [
            row["lazyframe"]
            for _, row in ticker_lazyframes_df.iterrows()
            if row["lazyframe"] is not None
        ],
        how="vertical",
    )
    sort_cols = (["cid", "xcat"] if return_qdf else ["ticker"]) + ["real_date"]
    # `last_updated` breaks ties, so `delta_treatment="all"` returns each row's versions
    # oldest first instead of in whatever order the files happened to concatenate. The
    # other treatments leave one row per key, where it never gets consulted
    out_lf = out_lf.sort(sort_cols + ["last_updated"])
    if dropna:
        # `dropna` governs only what reaches the output: keep the rows whose requested
        # metrics are all populated. Expiring a row is `delta_treatment`'s job and has
        # already happened above, so an expiry has superseded the row it withdraws
        # before being dropped here for being null. Named from the metrics rather than
        # by subtracting the key columns: in qdf shape "ticker" has given way to
        # cid/xcat, so subtraction would leave it behind
        cols = [
            c for c in (metrics or JPMAQS_VALUE_METRICS) if c in JPMAQS_VALUE_METRICS
        ]
        if cols:
            out_lf = out_lf.filter(pl.all_horizontal(pl.col(cols).is_not_null()))

    return out_lf


def _collect_naming_paths(
    lf: pl.LazyFrame, paths: Sequence[Union[str, Path]]
) -> pl.DataFrame:
    """Collect `lf`, and on failure name the files it was reading (at most 20)."""
    MAX_PATHS_IN_ERROR: int = 20
    try:
        return lf.collect()
    except Exception as e:
        shown = [str(p) for p in list(paths)[:MAX_PATHS_IN_ERROR]]
        if len(paths) > MAX_PATHS_IN_ERROR:
            shown.append(f"... and {len(paths) - MAX_PATHS_IN_ERROR} more")
        raise ValueError(f"Failed to load data from {shown}: {e}") from e


def _delta_treatment_error() -> str:
    """One message for both entry points, so the same input reads the same way."""
    return (
        "`delta_treatment` must be one of "
        f"{', '.join(repr(t) for t in DELTA_TREATMENTS)}."
    )


def _apply_delta_treatment(
    lf: pl.LazyFrame,
    delta_treatment: str = "latest",
    return_qdf: bool = True,
) -> pl.LazyFrame:
    """
    Resolve rows restated by delta files: keep the "latest" or "earliest" row per
    (key, real_date) by `last_updated`, or "all" of them.
    """
    if delta_treatment not in DELTA_TREATMENTS:
        raise ValueError(_delta_treatment_error())
    key_cols = ["cid", "xcat"] if return_qdf else ["ticker"]
    full_key = key_cols + ["real_date"]

    if delta_treatment != "all":
        if delta_treatment == "latest":
            lf = lf.sort(
                full_key + ["last_updated"], descending=[False] * len(full_key) + [True]
            ).unique(subset=full_key, keep="first")
        elif delta_treatment == "earliest":
            # "earliest" answers what the first print said, and an expiry is never that:
            # it records a withdrawal, not a print. Dropping expiries before the dedup
            # keeps the first real print however the expiry happens to sort. Only the
            # metrics present are tested, as callers may pass a narrower frame
            names = (
                lf.collect_schema().names()
                if PYTHON_3_8_OR_LATER
                else list(lf.schema.keys())
            )
            metric_cols = [c for c in JPMAQS_VALUE_METRICS if c in names]
            if metric_cols:
                lf = lf.filter(pl.any_horizontal(pl.col(metric_cols).is_not_null()))
            lf = lf.sort(
                full_key + ["last_updated"],
                descending=[False] * len(full_key) + [False],
            ).unique(subset=full_key, keep="first")
    return lf


def build_filtered_lazy_frames_df(
    paths: List[str],
    tickers: List[str],
    start_date: Optional[Union[str, pd.Timestamp]],
    end_date: Optional[Union[str, pd.Timestamp]],
    catalog_path: Union[str, Path],
    return_qdf: bool,
    include_source_file: bool = False,
    categorical_source_file_column: bool = True,
) -> pd.DataFrame:
    """
    Map each dataset to its parquet files and its requested tickers.

    Returns
    -------
    pd.DataFrame
        One row per dataset, with columns "e-dataset", "ticker", "path" and a
        "lazyframe" holding that dataset's files scanned and filtered.
    """
    tickers_list: List[str] = list(dict.fromkeys(tickers))
    catalog_df = _read_catalog(catalog_path).copy()  # copy: "Dataset" is added below
    catalog_df["Dataset"] = (
        catalog_df["Theme"].map(JPMAQS_DATASET_THEME_MAPPING).fillna("Unknown")
    )

    dataset_tickers_dict = catalog_df[
        catalog_df["Ticker"].str.lower().isin(map(str.lower, tickers_list))
    ][["Ticker", "Dataset"]]

    tickers_in_ds = (
        dataset_tickers_dict.rename(
            columns={"Ticker": "ticker", "Dataset": "e-dataset"}
        )
        .groupby("e-dataset")["ticker"]
        .agg(sorted)
        .reset_index()
    )
    files_for_ds = (
        pd.DataFrame(
            [
                [
                    Path(p),
                    Path(p).name.split(".")[0].rsplit("_", 1)[0].replace("_DELTA", ""),
                ]
                for p in paths
            ],
            columns=["path", "e-dataset"],
        )
        .groupby("e-dataset")["path"]
        .agg(sorted)
        .reset_index()
    )
    ticker_ds_file_mapping = tickers_in_ds.merge(
        files_for_ds, on="e-dataset", how="outer", indicator=True
    )

    # A row with no `path` means requested tickers whose dataset was never downloaded;
    # a row with no `ticker` means files on disk for a dataset nobody asked for. Neither
    # is loadable, so name the datasets instead of iterating a NaN. "Unknown" is the
    # legitimate no-dataset bucket and is skipped below.
    no_files = sorted(
        set(
            ticker_ds_file_mapping.loc[
                ticker_ds_file_mapping["path"].isna(), "e-dataset"
            ]
        )
        - {"Unknown"}
    )
    not_requested = sorted(
        set(
            ticker_ds_file_mapping.loc[
                ticker_ds_file_mapping["ticker"].isna(), "e-dataset"
            ]
        )
    )
    if no_files or not_requested:
        raise ValueError(
            "Could not match the requested tickers to the files on disk. "
            f"Datasets with requested tickers but no downloaded file: {no_files}. "
            f"Datasets with downloaded files but no requested tickers: {not_requested}. "
            "Download the missing datasets, or pass `datasets=` to restrict the load."
        )
    ticker_ds_file_mapping = ticker_ds_file_mapping.drop(columns="_merge")

    # this df is just a store for the lazyframes which maps each dataset to
    # the associated parquet files, and ticker to be loaded from it
    # these are NOT concatted here - which avoids delta-dedup/filtering on the full lazyframe
    ticker_ds_file_mapping["lazyframe"] = None
    for _, row in ticker_ds_file_mapping.iterrows():
        curr_dataset = row["e-dataset"]
        curr_tickers = row["ticker"]
        curr_paths = row["path"]
        if curr_dataset == "Unknown":
            logger.warning(
                f"Dataset for tickers {curr_tickers} is unknown. Skipping these tickers."
            )
            continue
        lazy_frame: pl.LazyFrame = pl.concat(
            [
                _scan_and_prepare_single_parquet(
                    path=p,
                    tickers=curr_tickers,
                    start_date=start_date,
                    end_date=end_date,
                    return_qdf=return_qdf,
                    include_source_file=include_source_file,
                    categorical_source_file_column=categorical_source_file_column,
                )
                for p in curr_paths
            ],
            how="vertical",
        )
        ticker_ds_file_mapping.loc[_, "lazyframe"] = lazy_frame

    # drop where lazyframe is None (i.e., unknown datasets)
    ticker_ds_file_mapping = ticker_ds_file_mapping.dropna(subset=["lazyframe"])
    return ticker_ds_file_mapping


if __name__ == "__main__":
    now_datetime = datetime.datetime.now(datetime.timezone.utc)
    print("Current time UTC:", now_datetime.isoformat())
    path = Path("~/jpmaqs-data").expanduser()
    start = time.time()
    print(
        "Downloading full-snapshots, delta-files, "
        f"and metadata files published as of {now_datetime.isoformat()}"
    )

    with DataQueryFileAPIClient() as dq:
        dq.download_files(since_datetime=now_datetime - datetime.timedelta(days=3))
        catalog_df = dq.load_catalog()
        random_tickers = catalog_df["Ticker"].sample(n=20, random_state=42).tolist()

        df = dq.download(tickers=random_tickers, keep_n_days_old_files=3)
        # print(df.head())
    end = time.time()
    print(f"Download completed in {end - start:.2f} seconds.")
