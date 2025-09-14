import os
import pandas as pd
import functools
import time
from pathlib import Path
import concurrent.futures as cf
import logging
import shutil
import traceback as tb
import uuid
from typing import Dict, Any, Optional, List, Tuple
from tqdm import tqdm
import requests
import threading
import datetime

from macrosynergy.download.dataquery import JPMAQS_GROUP_ID
from macrosynergy.download.fusion_interface import (
    request_wrapper,
    request_wrapper_stream_bytes_to_disk,
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


JPMAQS_START_DATE = "20200101"

logger = logging.getLogger(__name__)


class RateLimitedRequester:
    """
    Provides a rate-limiting mechanism for API requests.
    """

    def __init__(self, api_delay: float):
        self._api_delay = api_delay
        self._rate_limit_lock = threading.Lock()
        self._last_api_call = None

    def _wait_for_api_call(self) -> bool:
        """
        Blocks until the required delay since the last API call has passed.
        """
        with self._rate_limit_lock:
            now = datetime.datetime.now()
            if self._last_api_call is None:
                self._last_api_call = now
                return True

            diff = (now - self._last_api_call).total_seconds()
            sleep_for = self._api_delay - diff
            if sleep_for > 0:
                if sleep_for > 1:
                    logger.info(
                        f"Sleeping for {sleep_for:.2f} seconds for API rate limit."
                    )
                time.sleep(sleep_for)

            self._last_api_call = datetime.datetime.now()
        return True


def validate_dq_timestamp(
    ts: str, var_name: str = None, raise_error: bool = True
) -> bool:
    try:
        pd.to_datetime(ts, format="mixed", utc=True)
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


class DataQueryFileAPIOauth(JPMorganOAuth):
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
    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        base_url: str = DQ_FILE_API_BASE_URL,
        scope: str = DQ_FILE_API_SCOPE,
        proxies: Optional[Dict[str, str]] = None,
        verify_ssl: bool = True,
    ):
        """
        Client for JPM DataQuery File APIs using request_wrapper utilities.
        """
        super().__init__(api_delay=DQ_FILE_API_DELAY_PARAM)
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

        self.base_url = base_url.rstrip("/")
        self.scope = scope
        self.proxies = proxies
        self.verify_ssl = verify_ssl

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

    def _get(
        self, endpoint: str, params: Optional[Dict[str, Any]] = None, retries: int = 3
    ) -> Dict[str, Any]:
        """Generic GET with request_wrapper."""
        url = f"{self.base_url}{endpoint}"
        headers = self.oauth.get_auth()
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
                    api_delay=0,
                    verify_ssl=self.verify_ssl,
                )
            except Exception as e:
                logger.error(f"Error occurred during GET request: {e}")
                if _ == retries - 1:
                    raise
                logger.info(f"Retrying... ({_ + 1}/{retries})")
                time.sleep(2**_)

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
        start_date: str = JPMAQS_START_DATE,
        end_date: str = None,
        include_unavailable: bool = False,
    ) -> pd.DataFrame:
        """
        List all available files for a specific file in a group within a date range.
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
        df["_ts"] = pd.to_datetime(
            df["file-datetime"], format="mixed", errors="coerce", utc=True
        )
        df = (
            df.sort_values("_ts", ascending=False)
            .drop(columns="_ts")
            .reset_index(drop=True)
        )

        return df

    def list_available_files_for_all_file_groups(
        self,
        group_id: str = JPMAQS_GROUP_ID,
        start_date: str = JPMAQS_START_DATE,
        end_date: str = None,
        include_full_snapshots: bool = True,
        include_delta: bool = True,
        include_metadata: bool = True,
        convert_metadata_timestamps: bool = True,
        include_unavailable: bool = False,
    ) -> pd.DataFrame:
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
                        include_unavailable=include_unavailable,
                    )
                ] = file_group_id
                time.sleep(DQ_FILE_API_DELAY_PARAM)

            for future in cf.as_completed(futures):
                available_files = future.result()
                results.append(available_files)

        files_df = pd.concat(results).reset_index(drop=True)

        if convert_metadata_timestamps:
            for col in ["file-datetime", "last-modified"]:
                if col not in files_df.columns:
                    raise InvalidResponseError(f'Missing "{col}" in response')
                files_df[col] = pd.to_datetime(files_df[col], format="mixed", utc=True)
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
        if since_datetime is None:
            since_datetime = pd.Timestamp.utcnow().strftime("%Y%m%d")
        if to_datetime is None:
            to_datetime = pd.Timestamp.utcnow().strftime("%Y%m%dT%H%M%S")
        validate_dq_timestamp(since_datetime, var_name="since_datetime")
        validate_dq_timestamp(to_datetime, var_name="to_datetime")

        since_ts = pd.to_datetime(since_datetime, utc=True)
        to_ts = pd.to_datetime(to_datetime, utc=True)

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

        # DQ's internal date filtering is not as expected by end users,
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
        max_retries: int = 3,
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

        if file_path.exists():
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
            api_delay=0,
            verify_ssl=self.verify_ssl,
        )

        is_small_file = any(x in file_group_id.lower() for x in ["delta", "metadata"])

        if is_small_file:
            self._wait_for_api_call()
            request_wrapper_stream_bytes_to_disk(**download_args)
        else:
            download_args["api_delay"] = DQ_FILE_API_DELAY_PARAM
            SegmentedFileDownloader(
                **download_args,
                max_file_retries=max_retries,
                start_download=True,
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
        show_progress: bool = True,
    ) -> None:
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
                desc="Requesting Parquet files",
                disable=not show_progress,
            ):
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

        return self.download_multiple_parquet_files(
            filenames=failed_files,
            out_dir=out_dir,
            max_retries=max_retries - 1,
            n_jobs=n_jobs,
            chunk_size=chunk_size,
            timeout=timeout,
            show_progress=show_progress,
        )

    def download_full_snapshot(
        self,
        out_dir: str = "./download",
        since_datetime: Optional[str] = None,
        to_datetime: Optional[str] = None,
        file_datetime: Optional[str] = None,
        chunk_size: Optional[int] = None,
        timeout: Optional[float] = DQ_FILE_API_TIMEOUT,
        include_full_snapshots: bool = True,
        include_delta: bool = True,
        include_metadata: bool = True,
        file_group_ids: Optional[List[str]] = None,
        show_progress: bool = True,
    ) -> None:
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
            if not isinstance(file_group_ids, list) and not all(
                isinstance(x, str) for x in file_group_ids
            ):
                raise ValueError("`file_group_ids` must be a list of strings.")
            files_df = files_df[files_df["file-group-id"].isin(file_group_ids)]

        num_files_to_download = len(files_df["file-name"])
        if not num_files_to_download:
            logger.info("No new files to download.")
            return

        logger.info(f"Found {num_files_to_download} new files to download.")

        files_df["download-priority"] = (
            files_df["file-name"]
            .str.lower()
            .apply(lambda x: (3 if "_metadata" in x else (2 if "_delta" in x else 1)))
        )
        download_order = files_df.sort_values(
            by=["download-priority", "file-datetime", "file-name"],
        )["file-name"].tolist()

        self.download_multiple_parquet_files(
            filenames=download_order,
            out_dir=out_dir,
            chunk_size=chunk_size,
            timeout=timeout,
            show_progress=show_progress,
        )

        total_time = time.time() - start_time
        logger.info(f"Snapshot download completed in {total_time:.2f} seconds.")


class SegmentedFileDownloader(RateLimitedRequester):
    """Manages the concurrent download of a single file."""

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
        super().__init__(api_delay=api_delay * api_delay_margin)
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
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            logger.error(tb.format_exc())
        self.cleanup()
        return False

    def log(self, msg: str, part_num: int = None, level: int = logging.INFO):
        part_info = f"[part={part_num}]" if part_num is not None else ""
        logger.log(
            level, f"[SegmentedFileDownloader][file={self.file_id}]{part_info} {msg}"
        )

    def download(self, retries: int = None) -> Path:
        """Orchestrates the file download process."""
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
                time.sleep(self._api_delay)
                self.cleanup()
                return self.download(retries=retries - 1)

            self.cleanup()

        raise last_exception

    def _get_file_size(self) -> int:
        """Fetches the total size of the file."""
        self.log("Fetching file size...")
        self._wait_for_api_call()
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
        """Downloads all file chunks in parallel."""
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
        """Wrapper to start the recursive download of a single file chunk."""
        self._download_chunk_retry(part_num, start_byte, end_byte, retries=1)

    def _download_chunk_retry(
        self, part_num: int, start_byte: int, end_byte: int, retries: int
    ) -> None:
        """Downloads a single file chunk with a recursive retry mechanism."""
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
        """Assembles the downloaded chunks into a single file."""
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
        """Cleans up temporary files."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            self.log("Cleaned up temporary files.")


if __name__ == "__main__":
    dq = DataQueryFileAPIClient()

    print("Current time UTC:", pd.Timestamp.utcnow().isoformat())

    print("Calling `/group/files`")
    start = time.time()
    print(dq.list_group_files())
    end = time.time()
    print(f"Call completed in {end - start:.2f} seconds")

    available_files = dq.list_available_files(
        file_group_id="JPMAQS_MACROECONOMIC_TRENDS_DELTA"
    )
    latest_file_timestamp = available_files["file-datetime"].iloc[0]
    print(
        dq.check_file_availability(
            file_group_id="JPMAQS_MACROECONOMIC_TRENDS_DELTA",
            file_datetime=latest_file_timestamp,
        )
    )

    print("Starting download")
    start = time.time()
    dq = DataQueryFileAPIClient()

    today_str = '20250910'
    dq.download_full_snapshot(
        out_dir="./data/dqfiles/test/",
        since_datetime=today_str,
    )
    end = time.time()
    print(f"Download completed in {end - start:.2f} seconds")

    def check_and_download(interval_minutes: int = 5):
        last_scan = pd.Timestamp.utcnow().strftime("%Y%m%d")
        while True:
            print("Checking for new delta files...")
            curr_time = pd.Timestamp.utcnow().strftime("%Y%m%dT%H%M%S")
            start = time.time()
            dq.download_full_snapshot(
                out_dir="./data/dqfiles/continuous/",
                since_datetime=last_scan,
                include_full_snapshots=False,
                include_metadata=True,
                include_delta=True,
                show_progress=False,
            )
            end = time.time()
            last_scan = curr_time
            print(f"Scan completed in {end - start:.2f} seconds")
            time.sleep(interval_minutes * 60)

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
