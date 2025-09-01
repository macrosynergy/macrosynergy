import os
import pandas as pd
from typing import Dict, Any, Optional

from macrosynergy.download.dataquery import JPMAQS_GROUP_ID
from macrosynergy.download.fusion_interface import (
    request_wrapper,
    request_wrapper_stream_bytes_to_disk,
    FusionOAuth,
)

import time
from pathlib import Path

DQ_FILE_API_BASE_URL: str = (
    "https://api-strm-gw01.jpmchase.com/research/dataquery-authe/api/v2"
)
DQ_FILE_API_SCOPE: str = "JPMC:URI:RS-06785-DataQueryExternalApi-PROD"


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

    def list_group_files(self, group_id: str) -> pd.DataFrame:
        """List all files for a specific group."""
        endpoint = "/group/files"
        payload = self._get(endpoint, {"group-id": group_id})
        return pd.json_normalize(payload, record_path=["file-group-ids"])

    def list_available_files(
        self,
        group_id: str,
        file_group_id: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """
        List all available files for a specific file in a group within a date range.
        """
        endpoint = "/group/files/available-files"
        params = {
            "group-id": group_id,
            "file-group-id": file_group_id,
            "start-date": start_date,
            "end-date": end_date,
        }
        payload = self._get(endpoint, params)
        return pd.json_normalize(payload, record_path=["available-files"])

    def check_file_availability(
        self, file_group_id: str, file_datetime: str
    ) -> Dict[str, Any]:
        """Check the availability of a specific file in a group."""
        endpoint = "/group/file/availability"
        params = {"file-group-id": file_group_id, "file-datetime": file_datetime}
        return self._get(endpoint, params)

    def download_parquet_file(
        self,
        file_group_id: str,
        file_datetime: str,
        out_dir: str = "./download",
        filename: Optional[str] = None,
        chunk_size: Optional[int] = None,
        timeout: Optional[float] = 500.0,
    ) -> str:
        """
        Stream a Parquet file directly to disk using request_wrapper_stream_bytes_to_disk.
        Returns the full file path on success.
        """

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
        )
        print(f"Data complete: {time.time() - start:.2f} seconds")
        print(f"File downloaded successfully and saved as {file_path}")
        return file_path


if __name__ == "__main__":
    dq = DataQueryFileAPIClient()

    print("Current time:", pd.Timestamp.now().isoformat())
    print("Token acquired")

    print("Calling `/group/files`")
    start = time.time()
    print(dq.list_group_files(JPMAQS_GROUP_ID))
    end = time.time()
    print(f"Call completed in {end - start:.2f} seconds")

    print("Starting download")
    print("Current time:", pd.Timestamp.now().isoformat())
    start = time.time()
    dq.download_parquet_file(
        file_group_id="JPMAQS_GENERIC_RETURNS",
        file_datetime="20250819",
        out_dir="./data/dqfiles/",
    )
    end = time.time()
    print(f"Download completed in {end - start:.2f} seconds")
