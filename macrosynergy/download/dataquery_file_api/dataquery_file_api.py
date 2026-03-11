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

**Example 4: Download a slice of the dataset "as of" a specific date-time.**

This method downloads the necessary snapshot and delta files to reconstruct the
dataset as of the specified date-time.

.. code-block:: python

    from macrosynergy.download import DataQueryFileAPIClient
    import pandas as pd

    tickers = ['AUD_EQXR_NSA', 'CAD_EQXR_NSA', 'USD_EQXR_NSA', 'JPY_EQXR_NSA']

    with DataQueryFileAPIClient(out_dir="./data/jpmaqs-data/") as dq:
        # Cut off at noon on 2025-11-12 (UTC)
        df = dq.download_as_of(tickers=tickers, as_of_datetime="2025-11-12T12:00:00")

        # inline with T-1 release schedule
        assert df['real_date'].max() <= pd.Timestamp("2025-11-12")
        assert df['last_updated'].max() <= pd.Timestamp("2025-11-12T12:00:00")
        print(df.head())


**Example 5: Download all new or updated delta-files since a specific date/time.**

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


**Example 6: Download a single, specific historical file.**

.. code-block:: python

    from macrosynergy.download import DataQueryFileAPIClient
    client = DataQueryFileAPIClient(out_dir="./jpmaqs_data")
    # This specific filename can be found using the list_available_files... methods
    target_filename = "JPMAQS_MACROECONOMIC_BALANCE_SHEETS_20250414.parquet"

    print(f"Downloading {target_filename}...")
    file_path = client.download_file(filename=target_filename)
    print(f"File downloaded to: {file_path}")

**Example 7: Check availability for a specific file-group.**

.. code-block:: python

    from macrosynergy.download import DataQueryFileAPIClient
    client = DataQueryFileAPIClient()
    file_group_id = "JPMAQS_MACROECONOMIC_BALANCE_SHEETS"

    available_files = client.list_available_files(file_group_id=file_group_id)

    # print the earliest file's details
    print(available_files.iloc[-1])

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


**Example 9: Download all historical full snapshot files (vintages) for JPMaQS.**

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


---

Please find below the documentation for the `DataQueryFileAPIClient` and related
classes/methods.
"""

import os
import pandas as pd
import polars as pl

import time
from pathlib import Path

import concurrent.futures as cf
import logging
import traceback as tb
from typing import Dict, Any, Optional, List, Tuple, Union
from tqdm import tqdm
import json
from macrosynergy.compat import PYTHON_3_8_OR_LATER
from macrosynergy.download.dataquery import JPMAQS_GROUP_ID
from macrosynergy.download.fusion_interface import (
    request_wrapper,
    request_wrapper_stream_bytes_to_disk,
    cache_decorator,
)
from macrosynergy.download.dataquery import OAUTH_TOKEN_URL
from macrosynergy.download.exceptions import DownloadError, InvalidResponseError
from macrosynergy.download.jpm_oauth import JPMorganOAuth

from macrosynergy.download.dataquery_file_api.constants import (  # noqa: F401
    JPMAQS_DATASET_THEME_MAPPING,
    JPMAQS_EARLIEST_FILE_DATE,
    DQ_FILE_API_BASE_URL,
    DQ_FILE_API_SCOPE,
    DQ_FILE_API_TIMEOUT,
    DQ_FILE_API_HEADERS_TIMEOUT,
    DQ_FILE_API_DELAY_PARAM,
    DQ_FILE_API_DELAY_MARGIN,
    DQ_FILE_API_SEGMENT_SIZE_MB,
    DQ_FILE_API_STREAM_CHUNK_SIZE,
)

from macrosynergy.download.dataquery_file_api.common import (
    RateLimitedRequester,
    JPMaQSParquetExpectedColumns,
    pd_to_datetime_compat,
    validate_dq_timestamp,
    _large_delta_file_datetimes,
    get_current_or_last_business_day,
    _is_date_only_string,
    _normalize_file_timestamp_cutoff,
    _normalize_last_updated_cutoff,
    _downloaded_files_df,
)

from macrosynergy.download.dataquery_file_api.file_selector import FileSelector
from macrosynergy.download.dataquery_file_api.segmented_file_downloader import (
    SegmentedFileDownloader,
)

from macrosynergy.download.dataquery_file_api.file_loader import lazy_load_from_parquets

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
        self._file_selector = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            logger.error(tb.format_exc())
        return False

    @property
    def file_selector(self) -> FileSelector:
        """
        Cached `FileSelector` instance for this client.

        Notes
        -----
        This property is intentionally designed so it does not fetch the API file
        inventory when first accessed. Download operations (e.g. `download_full_snapshot`)
        refresh the selector with the latest unfiltered API inventory as needed.
        """
        if self._file_selector is None:
            local_files_df = self.list_downloaded_files(
                include_last_modified_columns=False
            )
            self._file_selector = FileSelector(
                api_files_df=None,
                local_files_df=local_files_df,
                file_name_col="file-name",
            )
        return self._file_selector

    def _refresh_file_selector(self) -> None:
        """
        Refresh the cached `FileSelector` local inventory from disk.

        This is a lightweight, network-free refresh used after downloads so subsequent
        selection/load operations see the updated local cache.
        """
        if self._file_selector is None:
            return
        try:
            local_files_df = self.list_downloaded_files(
                include_last_modified_columns=False
            )
            self._file_selector.refresh(local_files_df=local_files_df)
        except Exception:
            logger.warning("Failed to refresh file selector inventory after download.")
            pass

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
        file_group_id: Optional[str] = None,
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
            end_date = pd.Timestamp.utcnow().strftime("%Y%m%d")
        endpoint = "/group/files/available-files"
        params = {
            "group-id": group_id,
            "file-group-id": file_group_id,
            "start-date": start_date,
            "end-date": end_date,
        }
        self._wait_for_api_call(1)
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
        end_date: str = None,
        include_full_snapshots: bool = True,
        include_delta: bool = True,
        include_metadata: bool = True,
        convert_metadata_timestamps: bool = True,
        include_unavailable: bool = False,
    ) -> pd.DataFrame:
        """
        Fetches and consolidates available files for all relevant file groups.
        This method is simply a convenience wrapper for `list_available_files`.

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
        Retrieve files whose *file timestamp* (`file-datetime`) falls within a datetime window.

        Notes
        -----
        Despite the wording in older docs, this method filters on `file-datetime`
        (file-vintage timestamp), not `last-modified`.

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
            A DataFrame of files whose `file-datetime` falls in the specified window.
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
        Download a single DataQuery file to the client's output directory.

        This method can be called with either (`file_group_id` and `file_datetime`) or
        a `filename`.

        - Snapshot/delta datasets are typically `.parquet`.
        - Some metadata file groups publish `.json` files (pass `filename=...`).

        For large snapshot files, it automatically uses the `SegmentedFileDownloader`
        for a robust, multi-part download.

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
        """
        Download (or resolve) the most recent JPMaQS catalog parquet file.

        The catalog is used for ticker validation and mapping tickers to underlying
        JPMaQS datasets.

        Notes
        -----
        - The "latest" catalog is determined by an API call to
          `list_available_files(self.catalog_file_group_id)`.
        - If the latest catalog already exists locally and `overwrite=False`, this
          method returns the local path (no download required).
        - If the latest catalog cannot be downloaded, this method raises an error (no
          fallback to older local catalogs).
        """

        available_catalogs = self.list_available_files(self.catalog_file_group_id)
        if available_catalogs.empty:
            raise DownloadError("No catalog files available for download.")

        latest_catalog = available_catalogs.sort_values(
            by=["file-datetime", "last-modified", "file-name"], ascending=False
        ).iloc[0]
        latest_filename = str(latest_catalog["file-name"])
        logger.info(f"Latest catalog file identified: {latest_filename}")

        existing_files = self.list_downloaded_files(include_last_modified_columns=False)
        if (not overwrite) and (not existing_files.empty):
            local_match = existing_files[
                existing_files["file-name"].astype(str).eq(latest_filename)
            ]
            if not local_match.empty:
                file_path = str(local_match.iloc[0]["path"])
                logger.info(f"Catalog file already downloaded (latest): {file_path}")
                self._refresh_file_selector()
                return file_path

        try:
            file_path = self.download_file(
                filename=latest_filename,
                overwrite=overwrite,
                timeout=timeout,
            )
        except KeyboardInterrupt:
            raise
        except Exception as e:
            raise DownloadError(
                f"Failed to download latest catalog file '{latest_filename}': {e}"
            ) from e

        self._refresh_file_selector()
        return str(file_path)

    def get_datasets_for_indicators(
        self,
        tickers: Optional[List[str]] = None,
        cids: Optional[List[str]] = None,
        xcats: Optional[List[str]] = None,
        case_sensitive: bool = False,
        catalog_file: Optional[str] = None,
    ) -> List[str]:
        """
        Return the list of JPMaQS datasets that contain the requested tickers.

        This loads the JPMaQS catalog parquet and maps the catalog `Theme` column to
        DataQuery file-group-ids via `JPMAQS_DATASET_THEME_MAPPING`.

        Notes
        -----
        - Unknown themes are mapped to `"UnknownTheme"` (to avoid `NaN` propagation and
          sorting issues) and logged as a warning.
        """
        tickers = _construct_all_tickers_list(tickers=tickers, cids=cids, xcats=xcats)
        if not tickers or not any(t.strip() for t in tickers):
            raise ValueError("No valid tickers to search for.")

        catalog_file = catalog_file or self.download_catalog_file()
        catalog_df = pd.read_parquet(catalog_file)
        catalog_df.loc[:, "Dataset"] = catalog_df["Theme"].map(
            JPMAQS_DATASET_THEME_MAPPING
        )
        catalog_df.loc[:, "Dataset"] = catalog_df["Dataset"].fillna("UnknownTheme")
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
        include_last_modified_columns: bool = False,
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
            today_utc = pd.Timestamp.utcnow().normalize()
            raise ValueError(
                "Provided `date` is in the future (UTC). "
                f"Requested: {date.date()}, today (UTC): {today_utc.date()}."
            )
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
        _selection_since_datetime: Optional[str] = None,
        _selection_to_datetime: Optional[str] = None,
        _selection_min_last_updated: Optional[Union[str, pd.Timestamp]] = None,
        _selection_max_last_updated: Optional[Union[str, pd.Timestamp]] = None,
    ) -> None:
        """
        Downloads a complete snapshot of files based on specified criteria.

        This method fetches the full available-file inventory from the API (unbounded
        by `since_datetime` / `to_datetime`) and delegates vintage-aware selection to
        `FileSelector`. The vintage window (`since_datetime`, `to_datetime`) controls
        which files are selected for download, not which files are listed from the API.

        Parameters
        ----------
        since_datetime : Optional[str]
            Vintage window start (inclusive) used for file selection.
            Defaults to the start of the current day (UTC).
        to_datetime : Optional[str]
            Vintage window end (inclusive) used for file selection.
            Note: loading uses all locally available cached snapshot/delta files.
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

        Notes
        -----
        Internal parameters `_selection_since_datetime`, `_selection_to_datetime`,
        `_selection_min_last_updated`, and `_selection_max_last_updated` are
        used by higher-level helpers (such as `download()`) to pass selection intent that
        differs from the raw `since_datetime`/`to_datetime` window (for example, when row-
        vintage cutoffs via `max_last_updated` affect delta coverage decisions).
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
        if to_datetime is not None:
            validate_dq_timestamp(to_datetime, var_name="to_datetime")
            since_dt = pd_to_datetime_compat(since_datetime)
            to_dt = pd_to_datetime_compat(to_datetime)
            if to_dt < since_dt:
                new_since = (to_dt - pd.offsets.BDay(1)).strftime("%Y%m%d")
                logger.warning(
                    "`to_datetime` is before `since_datetime`; adjusting "
                    "`since_datetime` to be one business day before `to_datetime`. "
                    f"New `since_datetime`: {new_since}"
                )
                since_datetime = new_since
                validate_dq_timestamp(since_datetime, var_name="since_datetime")

        if file_group_ids is not None:
            if not isinstance(file_group_ids, list) or not all(
                isinstance(x, str) for x in file_group_ids
            ):
                raise ValueError("`file_group_ids` must be a list of strings.")

        # Always refresh the selector with an unfiltered API inventory so it can make
        # consistent vintage decisions from the full history.
        selector = self.file_selector
        api_files_df = self.list_available_files_for_all_file_groups()
        selector.refresh(api_files_df=api_files_df)

        downloaded_files_df = self.list_downloaded_files(
            include_last_modified_columns=False
        )
        selection_since = (
            _selection_since_datetime
            if _selection_since_datetime is not None
            else since_datetime
        )
        selection_to = (
            _selection_to_datetime
            if _selection_to_datetime is not None
            else to_datetime
        )

        selector.refresh(api_files_df=api_files_df, local_files_df=downloaded_files_df)
        oldest_ts = selector.oldest_api_file_timestamp()
        if (selection_to is not None) and (oldest_ts is not None):
            to_cutoff = _normalize_file_timestamp_cutoff(selection_to)
            if to_cutoff < oldest_ts:
                raise ValueError(
                    "`to_datetime` predates the oldest available JPMaQS file "
                    "timestamp reported by the API "
                    f"({oldest_ts.strftime('%Y-%m-%dT%H:%M:%SZ')})."
                )
        to_download = selector.select_files_for_download(
            overwrite=overwrite,
            since_datetime=selection_since,
            to_datetime=selection_to,
            file_group_ids=file_group_ids,
            include_full_snapshots=include_full_snapshots,
            include_delta_files=include_delta,
            include_metadata_files=include_metadata,
            warn_if_no_full_snapshots=bool(selection_since),
            min_last_updated=_selection_min_last_updated,
            max_last_updated=_selection_max_last_updated,
        )
        to_download = list(set(to_download))

        num_files_to_download = len(to_download)
        logger.info(f"Found {num_files_to_download} new files to download.")
        if not num_files_to_download:
            logger.info("No new files to download.")
            return

        selected_df = api_files_df[api_files_df["file-name"].isin(to_download)].copy()
        selected_df["download-priority"] = (
            selected_df["file-name"]
            .astype(str)
            .str.lower()
            .apply(lambda x: (3 if "_metadata" in x else (2 if "_delta" in x else 1)))
        )
        download_order = selected_df.sort_values(
            by=["download-priority", "file-datetime", "file-name"],
        )["file-name"].tolist()

        self.download_multiple_files(
            filenames=download_order,
            overwrite=overwrite,
            chunk_size=chunk_size,
            timeout=timeout,
            show_progress=show_progress,
        )

        self._refresh_file_selector()

        total_time = time.time() - start_time
        logger.info(f"Snapshot download completed in {total_time:.2f} seconds.")

    def download_delta_files(
        self,
        since_datetime: Optional[str] = None,
        to_datetime: Optional[str] = None,
        overwrite: bool = False,
        chunk_size: Optional[int] = None,
        timeout: Optional[float] = DQ_FILE_API_TIMEOUT,
        include_delta: bool = True,
        include_metadata: bool = True,
        file_group_ids: Optional[List[str]] = None,
        show_progress: bool = True,
        *args,
        **kwargs,
    ):
        """
        A convenience function to allow downloading only delta files within a given window.
        This is a wrapper around `download_full_snapshot()` with `include_full_snapshots=False`.
        """

        return self.download_full_snapshot(
            since_datetime=since_datetime,
            to_datetime=to_datetime,
            overwrite=overwrite,
            chunk_size=chunk_size,
            timeout=timeout,
            include_full_snapshots=False,
            include_delta=include_delta,
            include_metadata=include_metadata,
            file_group_ids=file_group_ids,
            show_progress=show_progress,
            *args,
            **kwargs,
        )

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
            with file timestamps on/after this cutoff (inclusive). This uses the file
            timestamp embedded in the filename (`file-datetime`), not the HTTP metadata
            field `last-modified`. If None, all locally available files are considered.
        to_datetime : Optional[str]
            Restrict which locally available snapshot/delta files are considered to those
            with file timestamps on/before this cutoff (inclusive). This uses the file
            timestamp embedded in the filename (`file-datetime`). If None, all locally
            available files are considered.

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

        if to_datetime is not None:
            if isinstance(to_datetime, str):
                validate_dq_timestamp(to_datetime, var_name="to_datetime")
            local_files_df = self.list_downloaded_files(
                include_last_modified_columns=False
            )
            if (not local_files_df.empty) and (
                "file-timestamp" in local_files_df.columns
            ):
                is_parquet = (
                    local_files_df["file-name"]
                    .astype(str)
                    .str.lower()
                    .str.endswith(".parquet")
                )
                is_metadata = (
                    local_files_df["file-name"]
                    .astype(str)
                    .str.contains("_METADATA", case=False, na=False)
                )
                is_catalog = (
                    local_files_df["dataset"].astype(str).eq(self.catalog_file_group_id)
                )
                data_df = local_files_df.loc[
                    is_parquet & (~is_metadata) & (~is_catalog)
                ]
                oldest_local_ts = (
                    data_df["file-timestamp"].min()
                    if (not data_df.empty) and ("file-timestamp" in data_df.columns)
                    else None
                )
                if pd.isna(oldest_local_ts):
                    oldest_local_ts = None
                if oldest_local_ts is not None:
                    to_cutoff = _normalize_file_timestamp_cutoff(to_datetime)
                    if to_cutoff < oldest_local_ts:
                        raise ValueError(
                            "`to_datetime` predates the oldest JPMaQS data file timestamp "
                            "found in the local cache "
                            f"({oldest_local_ts.strftime('%Y-%m-%dT%H:%M:%SZ')})."
                        )

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
        selector = self.file_selector
        selector.refresh(api_files_df=self.list_available_files_for_all_file_groups())
        return selector.effective_snapshot_switchover_ts(
            file_group_ids=base_datasets,
            catalog_file_group_id=self.catalog_file_group_id,
        )

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
        cleanup_old_files_n_days: Optional[int] = None,
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
            File-vintage window start (inclusive) for selecting which snapshot/delta/metadata
            files to download. This is based on the file timestamp (`file-datetime`) rather
            than the HTTP metadata field `last-modified`. Defaults to the start of the
            current day (UTC).
        to_datetime : Optional[str]
            File-vintage window end (inclusive) for selecting which snapshot/delta/metadata
            files to download. Note: `since_datetime` and `to_datetime` only affect which
            files are downloaded; the returned timeseries is controlled by
            `start_date`/`end_date`.

            Note for historical ("delta-only") vintages:
            - If `to_datetime` falls within a month where only monthly delta files exist,
              the selector may still include the covering month-end delta file even if its
              file timestamp is after `to_datetime`. Row-level filtering is then enforced
              via `max_last_updated` during the load step.
        skip_download : bool
            If True, do not download snapshot/delta/metadata files and only load from the
            local cache. In this mode, the client will use the most recent *local* catalog
            parquet file (at or before `to_datetime` if provided) and will not make any
            network requests. If no local catalog is available, this method raises a
            ValueError. Default is False.
        cleanup_old_files_n_days : Optional[int]
            If set to an integer value, deletes files older than this number of days
            from the local cache after the download is complete. This integer value is
            passed to `cleanup_old_files()`. If None, no cleanup is performed.
            Default is None.

        Returns
        -------
        Union[pd.DataFrame, pl.DataFrame, pl.LazyFrame]
            A DataFrame containing the requested data.
        """

        def _empty_for_type(df_type: str):
            if df_type == "pandas":
                return pd.DataFrame()
            if df_type == "polars":
                return pl.DataFrame()
            if df_type == "polars-lazy":
                return pl.DataFrame().lazy()
            return pd.DataFrame()

        fs = None
        if skip_download:
            existing = self.list_downloaded_files(include_last_modified_columns=False)
            fs = FileSelector(
                api_files_df=None,
                local_files_df=existing,
                file_name_col="file-name",
            )
        else:
            fs = self.file_selector
        most_recent_local_catalog_path = fs._most_recent_local_catalog(
            to_datetime=to_datetime
        )
        if skip_download and most_recent_local_catalog_path is None:
            raise ValueError(
                "Cannot skip download when no local catalog file is available. "
                "Please run this method once with `skip_download=False` to populate the cache."
            )

        catalog_file = None
        if skip_download:
            catalog_file = str(most_recent_local_catalog_path)
        else:
            try:
                catalog_file = self.download_catalog_file()
            except DownloadError as e:
                # If the API has no catalog (or catalog download fails), fall back to a
                # local cache if possible.
                catalog_file = most_recent_local_catalog_path
                if catalog_file:
                    logger.warning(
                        "Failed to download the JPMaQS catalog file; falling back to the "
                        f"most recent local catalog at '{catalog_file}'. Error: {e}"
                    )
                else:
                    logger.warning(
                        f"Failed to download the JPMaQS catalog file and no local catalog exists. "
                        f"Returning an empty DataFrame. Error: {e}"
                    )
                    return _empty_for_type(dataframe_type)

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
        if not valid_tickers:
            raise ValueError(
                "No valid tickers found with the provided `tickers`, `cids`, and `xcats`."
            )

        valid_norm = {t.lower() for t in valid_tickers}
        missing = sorted({t for t in rqstd_tickers if t.lower() not in valid_norm})
        if missing:
            lmiss = min(5, len(missing))
            nmore = f"{len(missing) - lmiss} more" if len(missing) > lmiss else ""
            miss_str = "[" + ", ".join(missing[:lmiss]) + "..." + nmore + "]"
            miss_str = (
                f"{len(missing)} tickers requested do not exist in the catalog, "
                f"these are: {miss_str}"
            )
            logger.warning(miss_str)

        rqstd_tickers = valid_tickers

        datasets_to_download = self.get_datasets_for_indicators(
            tickers=rqstd_tickers, catalog_file=catalog_file
        )
        if "UnknownTheme" in datasets_to_download:
            logger.warning(
                "Some tickers map to unknown catalog themes. "
                "These will be ignored for download selection until "
                "`JPMAQS_DATASET_THEME_MAPPING` is updated."
            )
            datasets_to_download = [
                d for d in datasets_to_download if d != "UnknownTheme"
            ]
        if datasets_to_download and include_delta_files:
            datasets_to_download += [f"{ds}_DELTA" for ds in datasets_to_download]
        if not skip_download:
            download_since_datetime = since_datetime
            if download_since_datetime is None:
                if to_datetime is not None:
                    to_dt = pd_to_datetime_compat(to_datetime)
                    download_since_datetime = (to_dt - pd.offsets.BDay(1)).strftime(
                        "%Y%m%d"
                    )
                else:
                    download_since_datetime = (
                        get_current_or_last_business_day().strftime("%Y%m%d")
                    )

            if (to_datetime is not None) and (not include_delta_files):
                validate_dq_timestamp(to_datetime, var_name="to_datetime")
                to_ts = _normalize_file_timestamp_cutoff(to_datetime)
                switchover_ts = self._get_effective_snapshot_switchover_ts(
                    datasets=datasets_to_download
                )
                if switchover_ts is None or to_ts < switchover_ts:
                    raise ValueError(
                        "The requested vintage predates the earliest available full "
                        "snapshots, so `include_delta_files` must be True."
                    )

            self.download_full_snapshot(
                since_datetime=download_since_datetime,
                to_datetime=to_datetime,
                file_group_ids=datasets_to_download,
                overwrite=overwrite,
                show_progress=show_progress,
                include_full_snapshots=True,
                include_delta=include_delta_files,
                include_metadata=include_metadata_files,
                _selection_min_last_updated=min_last_updated,
                _selection_max_last_updated=max_last_updated,
            )
            if not isinstance(cleanup_old_files_n_days, (type(None), int)):
                raise ValueError(
                    "`cleanup_old_files_n_days` must be an integer or None."
                )
            if isinstance(cleanup_old_files_n_days, int):
                cleanup_old_files_n_days = abs(cleanup_old_files_n_days)
                self.cleanup_old_files(days_to_keep=cleanup_old_files_n_days)

        if skip_download and isinstance(cleanup_old_files_n_days, int):
            if cleanup_old_files_n_days > 0:
                logger.warning(
                    "`cleanup_old_files_n_days` is ignored when `skip_download=True`."
                )

        load_since_datetime = since_datetime
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
            since_datetime=load_since_datetime,
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
        cleanup_old_files_n_days: Optional[int] = None,
        *args,
        **kwargs,
    ) -> Union[pd.DataFrame, pl.DataFrame, pl.LazyFrame]:
        """
        Return data "as of" a specific point in time.

        This is a very lightweight wrapper around `download()`: it only translates the
        intent ("as of") into the correct `download()` arguments:
          - `to_datetime`: the file-vintage cutoff (which files can be used)
          - `max_last_updated`: the row-level vintage cutoff (which updates are allowed)

        See func:`macrosynergy.download.dataquery_file_api.dataquery_file_api.DataQueryFileAPIClient.download()`
        for details on the other parameters and return value.

        Parameters
        ----------
        as_of_datetime : str
            The point in time to view the data as of. This can be a date-only string
            (e.g. "2023-12-31" or "20231231") or a datetime string
            (e.g. "2023-12-31T15:30:00Z" or "20231231153000"). Date-only strings are
            interpreted as end-of-day.
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
        now_ts = pd_to_datetime_compat(pd.Timestamp.utcnow()).tz_convert("UTC")
        today_utc = now_ts.normalize()

        if _is_date_only_string(as_of_datetime):
            if as_of_day > today_utc:
                raise ValueError(
                    "`as_of_datetime` is in the future (UTC). "
                    f"Requested: {as_of_day.date()}, today (UTC): {today_utc.date()}."
                )
            to_datetime_str = as_of_day.strftime("%Y%m%d")
            max_last_updated_str = (
                as_of_day + pd.DateOffset(days=1) - pd.Timedelta(seconds=1)
            ).strftime("%Y-%m-%dT%H:%M:%SZ")
        else:
            if as_of_ts > now_ts:
                raise ValueError(
                    "`as_of_datetime` is in the future (UTC). "
                    f"Requested: {as_of_ts.strftime('%Y-%m-%dT%H:%M:%SZ')}, "
                    f"now (UTC): {now_ts.strftime('%Y-%m-%dT%H:%M:%SZ')}."
                )
            to_datetime_str = as_of_ts.strftime("%Y-%m-%dT%H:%M:%SZ")
            max_last_updated_str = to_datetime_str

        selector = self.file_selector
        if selector.api_files_df.empty:
            selector.refresh(
                api_files_df=self.list_available_files_for_all_file_groups()
            )
        oldest_ts = selector.oldest_api_file_timestamp()
        if oldest_ts is not None:
            to_cutoff = _normalize_file_timestamp_cutoff(to_datetime_str)
            if to_cutoff < oldest_ts:
                raise ValueError(
                    "`as_of_datetime` predates the oldest available JPMaQS file timestamp "
                    "reported by the API "
                    f"({oldest_ts.strftime('%Y-%m-%dT%H:%M:%SZ')})."
                )

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
            since_datetime=None,
            to_datetime=to_datetime_str,
            skip_download=skip_download,
            cleanup_old_files_n_days=cleanup_old_files_n_days,
            include_metadata_files=False,
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

    # with DataQueryFileAPIClient(out_dir="./data/jpmaqs-data/") as dq:
    #     df = dq.download(
    #         tickers=tickers,
    #         include_file_column=True,
    #     )
    #     print(df.head())

    with DataQueryFileAPIClient(out_dir="./data/jpmaqs-data/") as dq:
        dq.download_delta_files(since_datetime="20220101")

        # with DataQueryFileAPIClient(out_dir="./data/jpmaqs-data/") as dq:
        df = dq.download_as_of(
            tickers=tickers,
            as_of_datetime="2026-01-13",
            include_file_column=True,
        )
        df
    # with DataQueryFileAPIClient(out_dir="./data/jpmaqs-data/") as dq:
    #     df = dq.download_as_of(tickers=tickers, as_of_datetime="2025-10-08T12:16:14Z")
    #     assert df["real_date"].max() <= pd.Timestamp("2025-11-12")
    #     assert df["last_updated"].max() <= pd.Timestamp("2025-10-08T12:16:14")
    #     print(df.head())

    # with DataQueryFileAPIClient(out_dir="./data/jpmaqs-data/") as dq:
    #     pl_df: pl.DataFrame = dq.download(
    #         cids=test_cids,
    #         xcats=test_xcats,
    #         dataframe_format="tickers",
    #         dataframe_type="polars",
    #     )
    #     print(pl_df.head())
