import os
import pandas as pd

import functools
import time
from pathlib import Path

import concurrent.futures as cf
import logging
from typing import Dict, Any, Optional, List

from tqdm import tqdm

from macrosynergy.download.dataquery import JPMAQS_GROUP_ID
from macrosynergy.download.fusion_interface import (
    request_wrapper,
    request_wrapper_stream_bytes_to_disk,
    FusionOAuth,
)

from macrosynergy.download.exceptions import DownloadError, InvalidResponseError

DQ_FILE_API_BASE_URL: str = (
    "https://api-strm-gw01.jpmchase.com/research/dataquery-authe/api/v2"
)
DQ_FILE_API_SCOPE: str = "JPMC:URI:RS-06785-DataQueryExternalApi-PROD"
DQ_FILE_API_TIMEOUT: float = 600.0
DQ_FILE_API_DELAY_PARAM: float = 0.04  # =1/25 ; 25 transactions per second
JPMAQS_START_DATE = "20200101"

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def validate_dq_timestamp(
    ts: str, var_name: str = None, raise_error: bool = True
) -> bool:
    try:
        pd.Timestamp(ts)
        return True
    except ValueError:
        if raise_error:
            vn = f"`{var_name}`" if var_name else "Timestamp"
            raise ValueError(f"Invalid {vn} format. Use YYYYMMDD or YYYYMMDDTHHMMSS")
        else:
            return False


class DataQueryFileAPIClient:
    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        base_url: str = DQ_FILE_API_BASE_URL,
        scope: str = DQ_FILE_API_SCOPE,
        proxies: Optional[Dict[str, str]] = None,
    ):
        """
        Client for JPM DataQuery File APIs using request_wrapper utilities.
        """
        self.client_id = client_id or os.getenv("DQ_CLIENT_ID")
        self.client_secret = client_secret or os.getenv("DQ_CLIENT_SECRET")
        if not self.client_id or not self.client_secret:
            raise ValueError("Missing DQ_CLIENT_ID or DQ_CLIENT_SECRET")

        self.base_url = base_url.rstrip("/")
        self.scope = scope
        self.proxies = proxies

        self.oauth = FusionOAuth(
            client_id=self.client_id,
            client_secret=self.client_secret,
            resource=self.scope,
        )

    def _get(
        self, endpoint: str, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generic GET with request_wrapper."""
        url = f"{self.base_url}{endpoint}"
        headers = self.oauth.get_auth()
        return request_wrapper(
            method="GET",
            url=url,
            headers=headers,
            params=params or {},
            proxies=self.proxies,
            as_json=True,
            api_delay=DQ_FILE_API_DELAY_PARAM,
        )

    def list_groups(self) -> pd.DataFrame:
        """List all groups (data providers)."""
        endpoint = "/groups"
        payload = self._get(endpoint, {})
        return pd.json_normalize(payload, record_path=["groups"])

    def search_groups(self, keywords: str) -> pd.DataFrame:
        """Search for groups (data providers) by keywords."""
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
        List all files for a specific group.

        Parameters
        ----------
        full_snapshot_only: bool
            If True, only full snapshot files are returned.
        delta_only: bool
            If True, only delta files are returned.
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
        ismetadata = df["file-group-id"].str.contains("_METADATA_")
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
        start_date: str = JPMAQS_START_DATE,
        end_date: str = None,
    ) -> pd.DataFrame:
        """
        List all available files for a specific file in a group within a date range.
        """
        if end_date is None:
            end_date = pd.Timestamp.now().strftime("%Y%m%d")
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

        df["file-datetime"] = df["file-datetime"].astype(str)

        # Sort by real timestamp while leaving the column as string
        df["_ts"] = pd.to_datetime(df["file-datetime"], format="mixed", errors="coerce")
        # mergesort keeps order stable for equal timestamps
        df = (
            df.sort_values("_ts", ascending=False, kind="mergesort")
            .drop(columns="_ts")
            .reset_index(drop=True)
        )

        return df

    def list_available_files_for_file_groups(
        self,
        group_id: str = JPMAQS_GROUP_ID,
        start_date: str = JPMAQS_START_DATE,
        end_date: str = None,
        include_full_snapshots: bool = True,
        include_delta: bool = True,
        include_metadata: bool = True,
    ) -> pd.DataFrame:
        files_groups = self.list_group_files(
            include_full_snapshots=include_full_snapshots,
            include_delta=include_delta,
            include_metadata=include_metadata,
        )["file-group-id"].tolist()
        results = []
        with cf.ThreadPoolExecutor() as executor:
            futures = {}
            for file_group_id in tqdm(files_groups):
                futures[
                    executor.submit(
                        self.list_available_files,
                        group_id=group_id,
                        file_group_id=file_group_id,
                        start_date=start_date,
                        end_date=end_date,
                    )
                ] = file_group_id
                time.sleep(DQ_FILE_API_DELAY_PARAM)

            for future in tqdm(cf.as_completed(futures), total=len(files_groups)):
                available_files = future.result()
                results.append(available_files)

        files_df = pd.concat(results).reset_index(drop=True)
        files_df["file-datetime"] = pd.to_datetime(
            files_df["file-datetime"], format="mixed"
        )
        return files_df

    def check_file_availability(
        self,
        file_group_id: str = None,
        file_datetime: str = None,
        filename: Optional[str] = None,
    ) -> pd.DataFrame:
        """Check the availability of a specific file in a group."""
        if not ((bool(file_group_id) and bool(file_datetime)) ^ bool(filename)):
            raise ValueError(
                "One of `file_group_id` & `file_datetime`, or `filename` must be provided."
            )
        endpoint = "/group/file/availability"
        params = {"file-group-id": file_group_id, "file-datetime": file_datetime}
        payload = self._get(endpoint, params)
        return pd.json_normalize(payload)

    def download_parquet_file(
        self,
        file_group_id: str = None,
        file_datetime: str = None,
        filename: Optional[str] = None,
        out_dir: str = "./download",
        chunk_size: Optional[int] = None,
        timeout: Optional[float] = DQ_FILE_API_TIMEOUT,
    ) -> str:
        """
        Stream a Parquet file directly to disk using request_wrapper_stream_bytes_to_disk.
        Returns the full file path on success.
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
        headers = self.oauth.get_auth()
        params = {"file-group-id": file_group_id, "file-datetime": file_datetime}

        Path(out_dir).mkdir(parents=True, exist_ok=True)
        file_name = filename or f"{file_group_id}_{file_datetime}.parquet"
        file_path = Path(out_dir) / Path(file_name)

        start = time.time()
        request_wrapper_stream_bytes_to_disk(
            filename=file_path,
            url=url,
            method="GET",
            headers=headers,
            params=params,
            proxies=self.proxies,
            chunk_size=chunk_size,
            timeout=timeout,
            api_delay=DQ_FILE_API_DELAY_PARAM,
        )
        time_taken = time.time() - start
        logger.info(
            f"Downloaded {file_name} in {time_taken:.2f} seconds to {file_path}"
        )
        return file_path

    def download_multiple_parquet_files(
        self,
        filenames: List[str],
        out_dir: str = "./download",
        max_retries: int = 3,
        n_jobs: int = None,
        chunk_size: Optional[int] = None,
        timeout: Optional[float] = DQ_FILE_API_TIMEOUT,
    ) -> None:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        failed_files = []
        if n_jobs == -1:
            n_jobs = None
        with cf.ThreadPoolExecutor(max_workers=n_jobs) as executor:
            futures = {}
            for filename in tqdm(filenames, desc="Requesting Parquet files"):
                futures[
                    executor.submit(
                        self.download_parquet_file,
                        filename=filename,
                        out_dir=out_dir,
                        chunk_size=chunk_size,
                        timeout=timeout,
                    )
                ] = filename
                time.sleep(DQ_FILE_API_DELAY_PARAM)

            for future in tqdm(
                cf.as_completed(futures),
                total=len(futures),
                desc="Downloading Parquet files",
            ):
                fname = futures[future]
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Failed to download {fname}: {e}")
                    failed_files.append(fname)

        if not failed_files:
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

        return self.download_multiple_parquet_files(
            filenames=failed_files,
            max_retries=max_retries - 1,
            n_jobs=n_jobs,
            chunk_size=chunk_size,
            timeout=timeout,
        )

    def download_full_snapshot(
        self,
        out_dir: str = "./download",
        since_datetime: Optional[str] = None,
        file_datetime: Optional[str] = None,
        chunk_size: Optional[int] = None,
        timeout: Optional[float] = DQ_FILE_API_TIMEOUT,
        include_full_snapshots: bool = True,
        include_delta: bool = True,
        include_metadata: bool = True,
    ) -> None:
        Path(out_dir).mkdir(parents=True, exist_ok=True)

        if file_datetime is None and since_datetime is None:
            since_datetime = pd.Timestamp.now().strftime("%Y%m%d")

        effective_ts = file_datetime or since_datetime

        validate_dq_timestamp(
            effective_ts,
            var_name="file_datetime" if file_datetime else "since_datetime",
        )

        filter_ts = pd.Timestamp(effective_ts)
        if "T" not in effective_ts:
            filter_ts = filter_ts.normalize()

        files_df = self.list_available_files_for_file_groups(
            include_full_snapshots=include_full_snapshots,
            include_delta=include_delta,
            include_metadata=include_metadata,
        )

        files_df = files_df[files_df["file-datetime"] >= filter_ts]
        files_df = files_df[files_df["is-available"]]

        return self.download_multiple_parquet_files(
            filenames=sorted(files_df["file-name"].tolist()),
            out_dir=out_dir,
            chunk_size=chunk_size,
            timeout=timeout,
        )


if __name__ == "__main__":
    dq = DataQueryFileAPIClient()

    print("Current time:", pd.Timestamp.now().isoformat())

    print("Calling `/group/files`")
    start = time.time()
    # print(dq.list_group_files())
    end = time.time()
    print(f"Call completed in {end - start:.2f} seconds")

    available_files = dq.list_available_files(
        file_group_id="JPMAQS_MACROECONOMIC_TRENDS_DELTA"
    )
    pd.to_datetime(
        available_files[available_files["is-available"]]["file-datetime"],
        format="mixed",
    ).max()
    c = dq.check_file_availability(
        file_group_id="JPMAQS_MACROECONOMIC_TRENDS_DELTA",
        file_datetime="20250828T040348",
    )

    print("Starting download")
    dq.download_full_snapshot(
        out_dir="./data/dqfiles/test/",
        since_datetime=pd.Timestamp.now().strftime("%Y%m%d"),
    )

    # print("Starting download")
    # print("Current time:", pd.Timestamp.now().isoformat())
    # start = time.time()
    # dq.download_parquet_file(
    #     file_group_id="JPMAQS_GENERIC_RETURNS",
    #     # file_datetime="20250819",
    #     file_datetime=pd.Timestamp.now().strftime("%Y%m%d"),
    #     out_dir="./data/dqfiles/",
    # )
    # end = time.time()
    # print(f"Download completed in {end - start:.2f} seconds")
