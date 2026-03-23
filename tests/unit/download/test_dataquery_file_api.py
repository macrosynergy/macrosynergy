import os
import json
import datetime
import unittest
import warnings
import pandas as pd
import requests
from unittest.mock import patch, MagicMock
import functools
import logging
import tempfile
from pathlib import Path
from macrosynergy.compat import PD_2_0_OR_LATER, PYTHON_3_8_OR_LATER
from macrosynergy.download.dataquery_file_api.common import (
    pd_timestamp_compat,
)
from macrosynergy.download.dataquery_file_api.dataquery_file_api import (
    validate_dq_timestamp,
    get_client_id_secret,
    get_current_or_last_business_day,
    pd_to_datetime_compat,
    DataQueryFileAPIClient,
    SegmentedFileDownloader,
    RateLimitedRequester,
    DownloadError,
    InvalidResponseError,
    DQ_FILE_API_SCOPE,
    JPMAQS_DATASET_THEME_MAPPING,
    _resolve_base_url,
    _base_url_cache,
    DQ_FILE_API_BASE_URL,
    DQ_FILE_API_FALLBACK_BASE_URL,
)


def suppress_logging(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logging.disable(logging.CRITICAL)
        try:
            return func(*args, **kwargs)
        finally:
            logging.disable(logging.NOTSET)

    return wrapper


class TestStandaloneFunctions(unittest.TestCase):
    def test_validate_dq_timestamp(self):
        self.assertTrue(validate_dq_timestamp("20230101"))
        self.assertTrue(validate_dq_timestamp("20230101T123000"))
        with self.assertRaises(ValueError):
            validate_dq_timestamp("invalid-date")
        self.assertFalse(validate_dq_timestamp("invalid-date", raise_error=False))
        with self.assertRaisesRegex(ValueError, "Invalid `my_ts` format"):
            validate_dq_timestamp("invalid-date", var_name="my_ts")

    def test_pd_timestamp_compat(self):
        self.assertEqual(
            pd_timestamp_compat("2023-01-05T01:02:03Z"),
            pd.Timestamp("2023-01-05T01:02:03Z"),
        )
        self.assertEqual(
            pd_timestamp_compat(pd.Timestamp("2023-01-05T01:02:03Z")),
            pd.Timestamp("2023-01-05T01:02:03Z"),
        )
        with self.assertRaises(TypeError):
            pd_timestamp_compat(pd.Series(["2023-01-01"]))

    def test_pd_to_datetime_compat_accepts_non_string_scalars(self):
        self.assertEqual(
            pd_to_datetime_compat(datetime.date(2023, 1, 5)),
            pd.Timestamp("2023-01-05T00:00:00Z"),
        )

        # naive datetime -> localized to UTC when utc=True
        self.assertEqual(
            pd_to_datetime_compat(datetime.datetime(2023, 1, 5, 1, 2, 3)),
            pd.Timestamp("2023-01-05T01:02:03Z"),
        )

        # tz-aware datetime -> converted to UTC
        self.assertEqual(
            pd_to_datetime_compat(
                datetime.datetime(
                    2023,
                    1,
                    5,
                    1,
                    2,
                    3,
                    tzinfo=datetime.timezone(datetime.timedelta(hours=2)),
                )
            ),
            pd.Timestamp("2023-01-04T23:02:03Z"),
        )

    def test_get_current_or_last_business_day(self):
        self.assertEqual(
            get_current_or_last_business_day("2026-01-28T12:00:00Z"),
            pd.Timestamp("2026-01-28T00:00:00Z"),
        )
        # Sunday -> Friday
        self.assertEqual(
            get_current_or_last_business_day("2026-02-01T17:20:22Z"),
            pd.Timestamp("2026-01-30T00:00:00Z"),
        )

    @patch("macrosynergy.download.dataquery_file_api.dataquery_file_api.os.getenv")
    def test_get_client_id_secret(self, mock_getenv):
        mock_getenv.side_effect = lambda key: {
            "DQ_CLIENT_ID": "id1",
            "DQ_CLIENT_SECRET": "secret1",
        }.get(key)
        self.assertEqual(get_client_id_secret(), ("id1", "secret1"))

        mock_getenv.side_effect = lambda key: {
            "DATAQUERY_CLIENT_ID": "id2",
            "DATAQUERY_CLIENT_SECRET": "secret2",
        }.get(key)
        self.assertEqual(get_client_id_secret(), ("id2", "secret2"))

        mock_getenv.side_effect = lambda key: None
        self.assertEqual(get_client_id_secret(), (None, None))


class TestRateLimiting(unittest.TestCase):
    @patch("macrosynergy.download.dataquery_file_api.dataquery_file_api.time.sleep")
    def test_rate_limiter_sleeps_on_quick_successive_calls(self, mock_sleep):
        requester = RateLimitedRequester(api_delay=1.0)

        t0 = datetime.datetime(2024, 1, 1, 0, 0, 0)
        now_times = iter(
            [
                t0,
                t0 + datetime.timedelta(seconds=0.25),
                t0 + datetime.timedelta(seconds=1.0),
            ]
        )

        class FakeDateTime(datetime.datetime):
            @classmethod
            def now(cls):
                return next(now_times)

        with patch(
            "macrosynergy.download.dataquery_file_api.common.datetime.datetime",
            FakeDateTime,
        ):
            requester._wait_for_api_call()
            requester._wait_for_api_call()

        mock_sleep.assert_called_once()
        self.assertAlmostEqual(mock_sleep.call_args[0][0], 0.75, places=2)

    def test_rate_limiter_concurrent_threads_respect_spacing_without_serialization(
        self,
    ):
        """Concurrent threads should fire in strict succession of the delay
        without holding the lock during sleep."""
        import threading
        import time as _time

        delay = 0.10  # 100ms - short enough for a fast test
        n_threads = 4
        requester = RateLimitedRequester(api_delay=delay)

        completion_times = []
        results_lock = threading.Lock()

        def worker():
            requester._wait_for_api_call()
            with results_lock:
                completion_times.append(_time.monotonic())

        # First call to initialise _last_api_call
        requester._wait_for_api_call()

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        start = _time.monotonic()
        for t in threads:
            t.start()

        # The lock should NOT be held during sleep - verify we can acquire it
        # quickly while the worker threads are sleeping.
        _time.sleep(0.02)  # small delay for threads to enter _wait_for_api_call
        lock_acquired = requester._rate_limit_lock.acquire(timeout=0.05)
        if lock_acquired:
            requester._rate_limit_lock.release()

        for t in threads:
            t.join(timeout=5)
        elapsed = _time.monotonic() - start

        completion_times.sort()

        # Rate limit respected: consecutive completions spaced by >= delay
        for i in range(1, len(completion_times)):
            gap = completion_times[i] - completion_times[i - 1]
            self.assertGreaterEqual(
                gap,
                delay * 0.7,
                f"Gap {i}: {gap:.4f}s < expected ~{delay}s - rate limit violated",
            )

        # No excessive serialisation: total time should be bounded
        expected_total = (n_threads - 1) * delay
        self.assertLessEqual(
            elapsed,
            expected_total * 4.0,
            f"Total {elapsed:.3f}s >> expected ~{expected_total:.3f}s",
        )

        # Lock was not held during sleep
        self.assertTrue(
            lock_acquired,
            "Could not acquire the rate-limiter lock while threads were sleeping "
            "- lock is being held during sleep (unnecessary serialisation)",
        )


class TestDataQueryFileAPIClient(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.test_dir = self.temp_dir.name

    def _write_local_catalog_file(
        self,
        client: DataQueryFileAPIClient,
        *,
        file_datetime: str = "20230101",
    ) -> str:
        """
        Create a dummy local catalog parquet file inside the client's cache directory.

        Some `download(..., skip_download=True)` unit tests require a local catalog to
        exist because `skip_download=True` now implies *no network calls*, including
        avoiding `download_catalog_file()`.
        """
        file_date_dir = pd.to_datetime(file_datetime, utc=True).strftime("%Y-%m-%d")
        pth = (
            Path(client.out_dir)
            / file_date_dir
            / f"JPMAQS_METADATA_CATALOG_{file_datetime}.parquet"
        )
        pth.parent.mkdir(parents=True, exist_ok=True)
        pth.touch()
        return str(pth)

    @patch(
        "macrosynergy.download.dataquery_file_api.dataquery_file_api.DataQueryFileAPIOauth"
    )
    @patch(
        "macrosynergy.download.dataquery_file_api.dataquery_file_api.get_client_id_secret",
        return_value=(None, None),
    )
    def test_init_no_credentials_raises_error(self, mock_get_client, mock_oauth):
        with self.assertRaisesRegex(
            ValueError, "Client ID and Client Secret must be provided"
        ):
            DataQueryFileAPIClient(out_dir=self.test_dir)

    @patch(
        "macrosynergy.download.dataquery_file_api.dataquery_file_api.DataQueryFileAPIOauth"
    )
    def test_init_with_args(self, mock_oauth_constructor):
        test_dir = os.path.join(self.test_dir, "test", "dir")
        client = DataQueryFileAPIClient(
            client_id="arg_id", client_secret="arg_secret", out_dir=test_dir
        )
        self.assertEqual(client.client_id, "arg_id")
        self.assertEqual(Path(client.out_dir), Path(test_dir) / "jpmaqs-download")
        mock_oauth_constructor.assert_called_once_with(
            client_id="arg_id",
            client_secret="arg_secret",
            resource=DQ_FILE_API_SCOPE,
            verify=True,
        )

    @suppress_logging
    @patch(
        "macrosynergy.download.dataquery_file_api.dataquery_file_api._delete_corrupt_files"
    )
    @patch.object(DataQueryFileAPIClient, "list_downloaded_files")
    @patch(
        "macrosynergy.download.dataquery_file_api.dataquery_file_api.DataQueryFileAPIOauth"
    )
    def test_delete_corrupt_files_passes_root_dir(
        self, mock_oauth, mock_list_downloaded_files, mock_delete_corrupt_files
    ):
        client = DataQueryFileAPIClient(
            client_id="id", client_secret="secret", out_dir=self.test_dir
        )
        fake_file_path = Path(client.out_dir) / "2023-01-01" / "f1.json"
        mock_list_downloaded_files.return_value = pd.DataFrame(
            {"file-name": ["f1.json"], "path": [str(fake_file_path)]}
        )
        mock_delete_corrupt_files.return_value = []

        client.delete_corrupt_files()

        self.assertEqual(
            mock_delete_corrupt_files.call_args[1]["root_dir"], Path(client.out_dir)
        )

    @patch(
        "macrosynergy.download.dataquery_file_api.dataquery_file_api.DataQueryFileAPIOauth"
    )
    @patch(
        "macrosynergy.download.dataquery_file_api.dataquery_file_api.get_client_id_secret",
        return_value=("env_id", "env_secret"),
    )
    def test_init_with_env_vars(self, mock_get_client, mock_oauth_constructor):
        client = DataQueryFileAPIClient()
        self.assertEqual(client.client_id, "env_id")
        self.assertEqual(client.client_secret, "env_secret")
        self.assertEqual(Path(client.out_dir), Path("jpmaqs-download"))
        mock_get_client.assert_called_once()
        mock_oauth_constructor.assert_called_once_with(
            client_id="env_id",
            client_secret="env_secret",
            resource=DQ_FILE_API_SCOPE,
            verify=True,
        )

    @patch("macrosynergy.download.dataquery_file_api.dataquery_file_api.logger")
    @patch(
        "macrosynergy.download.dataquery_file_api.dataquery_file_api.DataQueryFileAPIOauth"
    )
    def test_context_manager_exception_logging(self, mock_oauth, mock_logger):
        with self.assertRaises(ValueError):
            with DataQueryFileAPIClient(
                client_id="id", client_secret="secret", out_dir=self.test_dir
            ):
                raise ValueError("Test Exception")
        mock_logger.error.assert_called_once()

    @suppress_logging
    @patch(
        "macrosynergy.download.dataquery_file_api.dataquery_file_api.DataQueryFileAPIOauth"
    )
    @patch(
        "macrosynergy.download.dataquery_file_api.dataquery_file_api.request_wrapper"
    )
    @patch(
        "macrosynergy.download.dataquery_file_api.dataquery_file_api.time.sleep",
        MagicMock(),
    )
    def test_get_retry(self, mock_request_wrapper, mock_oauth_class):
        mock_oauth_instance = MagicMock()
        mock_oauth_instance.get_headers.return_value = {
            "Authorization": "Bearer fake_token"
        }
        mock_oauth_class.return_value = mock_oauth_instance

        client = DataQueryFileAPIClient(
            client_id="id", client_secret="secret", out_dir=self.test_dir
        )
        mock_request_wrapper.side_effect = [Exception("API error"), {"key": "value"}]
        response = client._get("/test", retries=2)
        self.assertEqual(response, {"key": "value"})
        self.assertEqual(mock_request_wrapper.call_count, 2)

        mock_request_wrapper.side_effect = [Exception("API error")] * 2
        with self.assertRaises(Exception):
            client._get("/test", retries=2)
        self.assertEqual(mock_request_wrapper.call_count, 4)

    @patch.object(DataQueryFileAPIClient, "_get")
    def test_list_and_search_groups(self, mock_get):
        client = DataQueryFileAPIClient(
            client_id="id", client_secret="secret", out_dir=self.test_dir
        )
        mock_get.side_effect = [
            {"groups": [{"id": "g1", "name": "Group 1"}]},
            {"groups": [{"id": "g_search", "name": "Group Search"}]},
        ]
        df_list = client.list_groups()
        pd.testing.assert_frame_equal(
            df_list, pd.DataFrame([{"id": "g1", "name": "Group 1"}])
        )
        df_search = client.search_groups(keywords="search_term")
        pd.testing.assert_frame_equal(
            df_search, pd.DataFrame([{"id": "g_search", "name": "Group Search"}])
        )
        self.assertEqual(mock_get.call_count, 2)

    @patch.object(DataQueryFileAPIClient, "_get")
    def test_list_group_files_filtering(self, mock_get):
        client = DataQueryFileAPIClient(
            client_id="id", client_secret="secret", out_dir=self.test_dir
        )
        mock_get.return_value = {
            "file-group-ids": [
                {"item": 1, "file-group-id": "FULL_SNAPSHOT"},
                {"item": 2, "file-group-id": "FILE_DELTA"},
                {"item": 3, "file-group-id": "FILE_METADATA"},
            ]
        }

        df_delta = client.list_group_files(
            include_full_snapshots=False, include_metadata=False
        )
        self.assertEqual(df_delta["file-group-id"].tolist(), ["FILE_DELTA"])

        df_full_only = client.list_group_files(
            include_delta=False, include_metadata=False
        )
        self.assertEqual(df_full_only["file-group-id"].tolist(), ["FULL_SNAPSHOT"])

    @patch.object(DataQueryFileAPIClient, "_get")
    def test_list_group_files_value_error(self, mock_get):
        client = DataQueryFileAPIClient(
            client_id="id", client_secret="secret", out_dir=self.test_dir
        )
        with self.assertRaises(ValueError):
            client.list_group_files(
                include_full_snapshots=False,
                include_delta=False,
                include_metadata=False,
            )

    @patch.object(DataQueryFileAPIClient, "_get")
    def test_list_group_files_cache(self, mock_get):
        client = DataQueryFileAPIClient(
            client_id="id", client_secret="secret", out_dir=self.test_dir
        )
        mock_get.return_value = {
            "file-group-ids": [
                {"item": 1, "file-group-id": "FULL_SNAPSHOT"},
                {"item": 2, "file-group-id": "FILE_DELTA"},
                {"item": 3, "file-group-id": "FILE_METADATA"},
            ]
        }
        df_all = client.list_group_files()
        self.assertEqual(len(df_all), 3)
        # now it should hit cache
        client.list_group_files()
        client.list_group_files()
        mock_get.assert_called_once()

    @patch("pandas.Timestamp.utcnow")
    @patch.object(DataQueryFileAPIClient, "_get")
    def test_list_available_files(self, mock_get, mock_now):
        client = DataQueryFileAPIClient(
            client_id="id", client_secret="secret", out_dir=self.test_dir
        )
        mock_now.return_value = pd.Timestamp("2023-01-02")
        mock_get.return_value = {
            "available-files": [
                {
                    "file-datetime": "20230101T100000",
                    "last-modified": "20230101T100000",
                    "is-available": True,
                }
            ]
        }
        client.list_available_files(file_group_id="test_id")
        self.assertEqual(mock_get.call_args[0][1]["end-date"], "20230102")

    @patch.object(DataQueryFileAPIClient, "_get")
    def test_list_available_files_include_unavailable(self, mock_get):
        client = DataQueryFileAPIClient(
            client_id="id", client_secret="secret", out_dir=self.test_dir
        )
        mock_get.return_value = {
            "available-files": [
                {
                    "file-datetime": "20230101T100000",
                    "last-modified": "20230101T100000",
                    "is-available": False,
                },
                {
                    "file-datetime": "20230102T100000",
                    "last-modified": "20230102T100000",
                    "is-available": True,
                },
            ]
        }
        df = client.list_available_files(
            file_group_id="test_id", include_unavailable=True
        )
        self.assertEqual(len(df), 2)
        df_available = client.list_available_files(
            file_group_id="test_id", include_unavailable=False
        )
        self.assertEqual(len(df_available), 1)

    @patch.object(DataQueryFileAPIClient, "_get")
    def test_list_available_files_invalid_response(self, mock_get):
        client = DataQueryFileAPIClient(
            client_id="id", client_secret="secret", out_dir=self.test_dir
        )
        mock_get.return_value = {"available-files": [{"is-available": True}]}
        with self.assertRaises(InvalidResponseError):
            # missing "file-datetime"
            client.list_available_files(file_group_id="test_id")

        mock_get.return_value = {
            "available-files": [
                {"file-datetime": "20230101T100000", "is-available": True}
            ]
        }
        with self.assertRaises(InvalidResponseError):
            # missing "last-modified"
            client.list_available_files(file_group_id="test_id")

    @patch.object(DataQueryFileAPIClient, "list_available_files")
    def test_list_available_files_for_all_with_conversion(self, mock_list_available):
        client = DataQueryFileAPIClient(
            client_id="id", client_secret="secret", out_dir=self.test_dir
        )
        mock_list_available.return_value = pd.DataFrame(
            {
                "file-datetime": pd.to_datetime(["20230101T120000"], utc=True),
                "last-modified": pd.to_datetime(["20230101T120000"], utc=True),
                "file-name": ["FG1_20230101.parquet"],
            }
        )
        df = client.list_available_files_for_all_file_groups()
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(df["file-datetime"]))

    @patch.object(DataQueryFileAPIClient, "list_available_files")
    def test_list_available_files_for_all_missing_column_error(
        self, mock_list_available
    ):
        client = DataQueryFileAPIClient(
            client_id="id", client_secret="secret", out_dir=self.test_dir
        )
        mock_list_available.side_effect = InvalidResponseError(
            'Missing "last-modified" in response'
        )
        with self.assertRaisesRegex(InvalidResponseError, 'Missing "last-modified"'):
            # mssing 'last-modified'
            client.list_available_files_for_all_file_groups()

    @patch.object(DataQueryFileAPIClient, "list_available_files_for_all_file_groups")
    def test_filter_available_files_by_datetime(self, mock_list_all_files):
        client = DataQueryFileAPIClient(
            client_id="id", client_secret="secret", out_dir=self.test_dir
        )
        mock_list_all_files.return_value = pd.DataFrame(
            {
                "file-datetime": pd.to_datetime(
                    ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04"], utc=True
                ),
                "last-modified": pd.to_datetime(
                    [
                        "2023-01-01T12:00:00Z",
                        "2023-01-02T12:00:00Z",
                        "2023-01-03T12:00:00Z",
                        "2023-01-04T12:00:00Z",
                    ]
                ),
                "file-name": ["f1", "f2", "f3", "f4"],
            }
        )

        filtered_df = client.filter_available_files_by_datetime(
            since_datetime="20230102", to_datetime="20230103"
        )
        self.assertEqual(filtered_df["file-name"].tolist(), ["f3", "f2"])
        mock_list_all_files.assert_called_once()

    @patch("macrosynergy.download.dataquery_file_api.dataquery_file_api.logger")
    @patch("pandas.Timestamp.utcnow")
    @patch.object(DataQueryFileAPIClient, "list_available_files_for_all_file_groups")
    def test_filter_available_files_by_datetime_defaults_and_swap(
        self, mock_list_all_files, mock_now, mock_logger
    ):
        client = DataQueryFileAPIClient(
            client_id="id", client_secret="secret", out_dir=self.test_dir
        )
        mock_now.return_value = pd.Timestamp("2023-01-05T12:00:00Z")
        mock_list_all_files.return_value = pd.DataFrame(
            columns=["file-datetime", "last-modified", "file-name"]
        )
        client.filter_available_files_by_datetime()
        mock_list_all_files.assert_called_once()
        client.filter_available_files_by_datetime(
            since_datetime="20230104", to_datetime="20230102"
        )
        mock_logger.warning.assert_called_once()
        self.assertIn("Swapping values", mock_logger.warning.call_args[0][0])

    @patch(
        "macrosynergy.download.dataquery_file_api.segmented_file_downloader.SegmentedFileDownloader._wait_for_api_call",
        MagicMock(),
    )
    @patch(
        "macrosynergy.download.dataquery_file_api.segmented_file_downloader.requests.head"
    )
    def test_segmented_downloader_head_uses_headers_timeout(self, mock_head):
        response = MagicMock()
        response.headers = {"Content-Length": "123"}
        mock_head.return_value = response

        parent = MagicMock()
        parent._wait_for_api_call = MagicMock(return_value=True)
        downloader = SegmentedFileDownloader(
            filename=os.path.join(self.test_dir, "out.parquet"),
            url="https://example.invalid/file",
            headers={"Authorization": "Bearer t"},
            params={"file-group-id": "FG", "file-datetime": "20240101"},
            headers_timeout=7,
            parent_requester=parent,
        )

        size = downloader._get_file_size()
        self.assertEqual(size, 123)
        mock_head.assert_called_once()
        # self.assertEqual(mock_head.call_args.kwargs.get("timeout"), 7)
        if PYTHON_3_8_OR_LATER:
            self.assertEqual(mock_head.call_args.kwargs.get("timeout"), 7)
        else:
            self.assertEqual(mock_head.call_args[1].get("timeout"), 7)

    @patch(
        "macrosynergy.download.dataquery_file_api.segmented_file_downloader.requests.head"
    )
    def test_segmented_downloader_uses_parent_rate_limiter(self, mock_head):
        parent = MagicMock()
        parent._wait_for_api_call = MagicMock(return_value=True)

        response = MagicMock()
        response.headers = {"Content-Length": "123"}
        mock_head.return_value = response

        downloader = SegmentedFileDownloader(
            filename=os.path.join(self.test_dir, "out.parquet"),
            url="https://example.invalid/file",
            headers={"Authorization": "Bearer t"},
            params={"file-group-id": "FG", "file-datetime": "20240101"},
            headers_timeout=7,
            parent_requester=parent,
        )

        downloader._get_file_size()
        parent._wait_for_api_call.assert_called()

    @patch(
        "macrosynergy.download.dataquery_file_api.segmented_file_downloader.requests.get"
    )
    def test_segmented_downloader_chunk_calls_wait_for_api_call(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.__enter__.return_value = mock_resp
        mock_resp.__exit__.return_value = False
        mock_resp.raise_for_status.return_value = None
        mock_resp.iter_content.return_value = [b"abc"]
        mock_get.return_value = mock_resp

        parent = MagicMock()
        parent._wait_for_api_call = MagicMock(return_value=True)
        downloader = SegmentedFileDownloader(
            filename=os.path.join(self.test_dir, "out.parquet"),
            url="https://example.invalid/file",
            headers={"Authorization": "Bearer t"},
            params={"file-group-id": "FG", "file-datetime": "20240101"},
            max_concurrent_downloads=1,
            parent_requester=parent,
        )
        downloader.temp_dir.mkdir(parents=True, exist_ok=True)
        with patch.object(
            downloader, "_wait_for_api_call", return_value=True
        ) as mock_wait:
            downloader._download_chunk_retry(
                part_num=0, start_byte=0, end_byte=2, retries=1
            )
        mock_wait.assert_called_once()
        downloader.cleanup()

    @patch.object(DataQueryFileAPIClient, "_get")
    def test_check_file_availability(self, mock_get):
        client = DataQueryFileAPIClient(
            client_id="id", client_secret="secret", out_dir=self.test_dir
        )

        # Valid cases
        client.check_file_availability(file_group_id="FG", file_datetime="20230101")
        mock_get.assert_called_with(
            "/group/file/availability",
            {"file-group-id": "FG", "file-datetime": "20230101"},
        )

        client.check_file_availability(filename="FG_20230101.parquet")
        mock_get.assert_called_with(
            "/group/file/availability",
            {"file-group-id": "FG", "file-datetime": "20230101"},
        )

        client.check_file_availability(filename="FG_20230101")
        mock_get.assert_called_with(
            "/group/file/availability",
            {"file-group-id": "FG", "file-datetime": "20230101"},
        )

        client.check_file_availability(
            filename=os.path.join("some", "nested", "dir", "FG_20230101.parquet")
        )
        mock_get.assert_called_with(
            "/group/file/availability",
            {"file-group-id": "FG", "file-datetime": "20230101"},
        )

        # Matches real naming: <dataset>_DELTA_<YYYYMMDDTHHMMSS>.parquet
        client.check_file_availability(
            filename="JPMAQS_GENERIC_RETURNS_DELTA_20251223T080402.parquet"
        )
        mock_get.assert_called_with(
            "/group/file/availability",
            {
                "file-group-id": "JPMAQS_GENERIC_RETURNS_DELTA",
                "file-datetime": "20251223T080402",
            },
        )

        # Invalid cases
        with self.assertRaises(ValueError):
            client.check_file_availability()

        with self.assertRaises(ValueError):
            client.check_file_availability(
                filename="f.pq", file_group_id="FG", file_datetime="20230101"
            )

        with self.assertRaisesRegex(ValueError, "Invalid filename format"):
            client.check_file_availability(filename="badformat.parquet")

    @suppress_logging
    @patch("macrosynergy.download.dataquery_file_api.dataquery_file_api.Path")
    @patch(
        "macrosynergy.download.dataquery_file_api.dataquery_file_api.DataQueryFileAPIOauth"
    )
    def test_download_file_no_overwrite(self, mock_oauth, mock_path):
        client = DataQueryFileAPIClient(
            client_id="id", client_secret="secret", out_dir=self.test_dir
        )
        mock_final_path = MagicMock()
        (mock_path.return_value.__truediv__.return_value.__truediv__.return_value) = (
            mock_final_path
        )
        mock_final_path.exists.return_value = True
        result = client.download_file(
            filename="TEST_FULL_20230101.parquet", overwrite=False
        )
        mock_final_path.unlink.assert_not_called()
        self.assertEqual(result, str(mock_final_path))

    @suppress_logging
    @patch(
        "macrosynergy.download.dataquery_file_api.dataquery_file_api.SegmentedFileDownloader"
    )
    @patch("macrosynergy.download.dataquery_file_api.dataquery_file_api.Path")
    @patch(
        "macrosynergy.download.dataquery_file_api.dataquery_file_api.DataQueryFileAPIOauth"
    )
    def test_download_file_overwrite(
        self, mock_oauth, mock_path, mock_segmented_downloader
    ):
        client = DataQueryFileAPIClient(
            client_id="id", client_secret="secret", out_dir=self.test_dir
        )
        mock_final_path = MagicMock()
        (mock_path.return_value.__truediv__.return_value.__truediv__.return_value) = (
            mock_final_path
        )
        mock_final_path.exists.return_value = True

        client.download_file(filename="TEST_FULL_20230101.parquet", overwrite=True)
        mock_final_path.unlink.assert_called_once()

    @suppress_logging
    @patch(
        "macrosynergy.download.dataquery_file_api.dataquery_file_api.request_wrapper_stream_bytes_to_disk"
    )
    @patch(
        "macrosynergy.download.dataquery_file_api.dataquery_file_api.SegmentedFileDownloader"
    )
    @patch("macrosynergy.download.dataquery_file_api.dataquery_file_api.Path")
    @patch(
        "macrosynergy.download.dataquery_file_api.dataquery_file_api.DataQueryFileAPIOauth"
    )
    def test_download_file_small_file_logic(
        self, mock_oauth, mock_path, mock_segmented_downloader, mock_request_wrapper
    ):
        client = DataQueryFileAPIClient(
            client_id="id", client_secret="secret", out_dir=self.test_dir
        )
        mock_file_path = MagicMock()
        mock_file_path.exists.return_value = False
        (mock_path.return_value.__truediv__.return_value.__truediv__.return_value) = (
            mock_file_path
        )

        client.download_file(filename="TEST_DELTA_20230101T010101.parquet")
        mock_request_wrapper.assert_called_once()
        mock_segmented_downloader.assert_not_called()

        mock_request_wrapper.reset_mock()
        mock_segmented_downloader.reset_mock()

        client.download_file(filename="TEST_METADATA_20230101.parquet")
        mock_request_wrapper.assert_called_once()
        mock_segmented_downloader.assert_not_called()

    @suppress_logging
    @patch(
        "macrosynergy.download.dataquery_file_api.dataquery_file_api.SegmentedFileDownloader"
    )
    @patch("macrosynergy.download.dataquery_file_api.dataquery_file_api.Path")
    @patch(
        "macrosynergy.download.dataquery_file_api.dataquery_file_api.DataQueryFileAPIOauth"
    )
    def test_download_file_large_file_passes_parent_requester(
        self, mock_oauth, mock_path, mock_segmented_downloader
    ):
        client = DataQueryFileAPIClient(
            client_id="id", client_secret="secret", out_dir=self.test_dir
        )
        mock_file_path = MagicMock()
        mock_file_path.exists.return_value = False
        (mock_path.return_value.__truediv__.return_value.__truediv__.return_value) = (
            mock_file_path
        )

        client.download_file(filename="TEST_FULL_20230101.parquet")
        self.assertTrue(mock_segmented_downloader.called)
        if PYTHON_3_8_OR_LATER:
            kwargs = mock_segmented_downloader.call_args.kwargs
        else:
            kwargs = mock_segmented_downloader.call_args[1]
        self.assertIs(kwargs.get("parent_requester"), client)

    @patch(
        "macrosynergy.download.dataquery_file_api.dataquery_file_api.DataQueryFileAPIOauth"
    )
    def test_download_file_invalid_filename_format(self, mock_oauth):
        client = DataQueryFileAPIClient(
            client_id="id", client_secret="secret", out_dir=self.test_dir
        )
        with self.assertRaisesRegex(ValueError, "Invalid filename format"):
            client.download_file(filename="invalidformat.parquet")

    @suppress_logging
    @patch(
        "macrosynergy.download.dataquery_file_api.dataquery_file_api.DataQueryFileAPIOauth"
    )
    def test_cleanup_old_files_deletes_expected_files_and_returns_list(
        self, mock_oauth
    ):
        client = DataQueryFileAPIClient(
            client_id="id", client_secret="secret", out_dir=self.test_dir
        )
        base = Path(self.test_dir)
        f_old_a = base / "old_a.parquet"
        f_new_a = base / "new_a.parquet"
        f_old_b = base / "old_b.parquet"
        f_edge_b = base / "edge_b.parquet"
        for p in [f_old_a, f_new_a, f_old_b, f_edge_b]:
            p.write_bytes(b"x")

        found_files = pd.DataFrame(
            [
                {
                    "dataset": "FG_A",
                    "file-timestamp": pd.Timestamp("2024-01-01"),
                    "path": str(f_old_a),
                },
                {
                    "dataset": "FG_A",
                    "file-timestamp": pd.Timestamp("2024-01-03"),
                    "path": str(f_new_a),
                },
                {
                    "dataset": "FG_B",
                    "file-timestamp": pd.Timestamp("2024-01-29"),
                    "path": str(f_old_b),
                },
                {
                    "dataset": "FG_B",
                    "file-timestamp": pd.Timestamp("2024-01-31"),
                    "path": str(f_edge_b),
                },
            ]
        )

        with patch.object(client, "list_downloaded_files", return_value=found_files):
            deleted = client.cleanup_old_files(days_to_keep=1)

        self.assertEqual(set(deleted), {str(f_old_a), str(f_old_b)})
        self.assertFalse(f_old_a.exists())
        self.assertFalse(f_old_b.exists())
        self.assertTrue(f_new_a.exists())
        self.assertTrue(f_edge_b.exists())

    @patch(
        "macrosynergy.download.dataquery_file_api.dataquery_file_api.DataQueryFileAPIOauth"
    )
    def test_cleanup_old_files_days_to_keep_zero_is_noop(self, mock_oauth):
        client = DataQueryFileAPIClient(
            client_id="id", client_secret="secret", out_dir=self.test_dir
        )
        with patch.object(client, "list_downloaded_files") as mock_list:
            deleted = client.cleanup_old_files(days_to_keep=0)
        self.assertEqual(deleted, [])
        mock_list.assert_not_called()

    @patch(
        "macrosynergy.download.dataquery_file_api.dataquery_file_api.cf.as_completed"
    )
    @patch(
        "macrosynergy.download.dataquery_file_api.dataquery_file_api.cf.ThreadPoolExecutor"
    )
    @patch(
        "macrosynergy.download.dataquery_file_api.dataquery_file_api.DataQueryFileAPIOauth"
    )
    def test_download_multiple_files_success(
        self, mock_oauth, mock_executor_cls, mock_as_completed
    ):
        client = DataQueryFileAPIClient(
            client_id="id", client_secret="secret", out_dir=self.test_dir
        )
        future1, future2 = MagicMock(), MagicMock()
        mock_executor = mock_executor_cls.return_value.__enter__.return_value
        mock_executor.submit.side_effect = [future1, future2]
        mock_as_completed.return_value = [future1, future2]

        with patch.object(
            client,
            "download_multiple_files",
            wraps=client.download_multiple_files,
        ) as spy:
            client.download_multiple_files(
                filenames=["f1.parquet", "f2.parquet"], show_progress=False
            )
            spy.assert_called_once()

        future1.result.assert_called_once()
        future2.result.assert_called_once()

    @suppress_logging
    @patch(
        "macrosynergy.download.dataquery_file_api.dataquery_file_api.cf.as_completed"
    )
    @patch(
        "macrosynergy.download.dataquery_file_api.dataquery_file_api.cf.ThreadPoolExecutor"
    )
    @patch(
        "macrosynergy.download.dataquery_file_api.dataquery_file_api.DataQueryFileAPIOauth"
    )
    def test_download_multiple_files_retry(
        self, mock_oauth, mock_executor_cls, mock_as_completed
    ):
        client = DataQueryFileAPIClient(
            client_id="id", client_secret="secret", out_dir=self.test_dir
        )
        future_success, future_fail = MagicMock(), MagicMock()
        future_fail.result.side_effect = Exception("Download failed!")
        mock_executor = mock_executor_cls.return_value.__enter__.return_value
        mock_executor.submit.side_effect = lambda fn, *args, **kwargs: (
            future_success if kwargs.get("filename") == "f1.parquet" else future_fail
        )
        mock_as_completed.side_effect = lambda futures_dict: list(futures_dict.keys())

        with patch.object(
            client,
            "download_multiple_files",
            wraps=client.download_multiple_files,
        ) as spy:
            with self.assertRaises(DownloadError):
                client.download_multiple_files(
                    filenames=["f1.parquet", "f2.parquet"],
                    max_retries=1,
                    show_progress=False,
                )
            self.assertEqual(spy.call_count, 2)
            res = None
            expected = ["f2.parquet"]
            if PD_2_0_OR_LATER:
                res = spy.call_args_list[1].kwargs["filenames"]
            else:
                res = spy.call_args_list[1][1]["filenames"]
            self.assertEqual(res, expected)

    @suppress_logging
    @patch.object(DataQueryFileAPIClient, "delete_corrupt_files")
    @patch(
        "macrosynergy.download.dataquery_file_api.dataquery_file_api.cf.as_completed"
    )
    @patch(
        "macrosynergy.download.dataquery_file_api.dataquery_file_api.cf.ThreadPoolExecutor"
    )
    @patch(
        "macrosynergy.download.dataquery_file_api.dataquery_file_api.DataQueryFileAPIOauth"
    )
    def test_download_multiple_files_retries_corrupt_files_as_filenames(
        self,
        mock_oauth,
        mock_executor_cls,
        mock_as_completed,
        mock_delete_corrupt_files,
    ):
        client = DataQueryFileAPIClient(
            client_id="id", client_secret="secret", out_dir=self.test_dir
        )
        mock_executor = mock_executor_cls.return_value.__enter__.return_value
        # Simulate "successful" futures, but corruption detected after download.
        # Use a factory so recursive retries can submit again.
        mock_executor.submit.side_effect = lambda *args, **kwargs: MagicMock()
        mock_as_completed.side_effect = lambda futures_dict: list(futures_dict.keys())
        # delete_corrupt_files returns filesystem paths, not filenames
        mock_delete_corrupt_files.return_value = [
            os.path.join(self.test_dir, "jpmaqs-download", "2023-01-01", "f2.parquet")
        ]

        with patch.object(
            client,
            "download_multiple_files",
            wraps=client.download_multiple_files,
        ) as spy:
            with self.assertRaises(DownloadError):
                client.download_multiple_files(
                    filenames=["f1.parquet", "f2.parquet"],
                    max_retries=1,
                    show_progress=False,
                )
            # Second call should retry just the basename, not the full path.
            if PD_2_0_OR_LATER:
                res = spy.call_args_list[1].kwargs["filenames"]
            else:
                res = spy.call_args_list[1][1]["filenames"]
            self.assertEqual(res, ["f2.parquet"])

    @patch(
        "macrosynergy.download.dataquery_file_api.dataquery_file_api.cf.as_completed"
    )
    @patch(
        "macrosynergy.download.dataquery_file_api.dataquery_file_api.cf.ThreadPoolExecutor"
    )
    @suppress_logging
    def test_download_multiple_files_keyboard_interrupt(
        self, mock_executor_cls, mock_as_completed
    ):
        client = DataQueryFileAPIClient(
            client_id="id", client_secret="secret", out_dir=self.test_dir
        )
        future1 = MagicMock()
        future1.result.side_effect = KeyboardInterrupt
        mock_executor = mock_executor_cls.return_value.__enter__.return_value
        mock_executor.submit.return_value = future1
        mock_as_completed.return_value = [future1]
        with self.assertRaises(KeyboardInterrupt):
            client.download_multiple_files(
                filenames=["f1.parquet"], show_progress=False
            )
        mock_executor.shutdown.assert_called_once_with(wait=False, cancel_futures=True)

    @patch(
        "macrosynergy.download.dataquery_file_api.dataquery_file_api.pd.read_parquet"
    )
    @patch.object(DataQueryFileAPIClient, "download_file")
    @patch.object(DataQueryFileAPIClient, "list_available_files")
    def test_download_catalog_file(
        self, mock_list_files, mock_download, mock_read_parquet
    ):
        mock_list_files.return_value = pd.DataFrame(
            {
                "file-name": ["CATALOG_20230102.parquet"],
                "file-datetime": pd.to_datetime(["2023-01-02"]),
                "last-modified": pd.to_datetime(["2023-01-02"]),
            }
        )
        cat_dir = os.path.join(self.test_dir, "cat")
        client = DataQueryFileAPIClient(
            client_id="id", client_secret="secret", out_dir=cat_dir
        )
        fake_path_str = os.path.join(
            cat_dir, "jpmaqs-download", "CATALOG_20230102.parquet"
        )
        mock_download.return_value = fake_path_str

        # simple base case
        client.download_catalog_file(overwrite=True)
        mock_download.assert_called_once_with(
            filename="CATALOG_20230102.parquet",
            overwrite=True,
            timeout=300.0,
        )
        mock_read_parquet.assert_not_called()

        # error case
        mock_list_files.return_value = pd.DataFrame()
        with self.assertRaises(DownloadError):
            client.download_catalog_file()

    @patch("macrosynergy.download.dataquery_file_api.dataquery_file_api.logger")
    @patch(
        "macrosynergy.download.dataquery_file_api.dataquery_file_api.pd.read_parquet"
    )
    @patch(
        "macrosynergy.download.dataquery_file_api.dataquery_file_api.DataQueryFileAPIOauth"
    )
    def test_get_datasets_for_indicators_maps_themes_to_datasets(
        self, mock_oauth, mock_read_parquet, mock_logger
    ):
        client = DataQueryFileAPIClient(
            client_id="id", client_secret="secret", out_dir=self.test_dir
        )

        themes = list(JPMAQS_DATASET_THEME_MAPPING.keys())
        tickers = [f"TICKER_{i}" for i in range(len(themes))]
        mock_read_parquet.return_value = pd.DataFrame(
            {"Theme": themes, "Ticker": tickers}
        )

        result = client.get_datasets_for_indicators(
            tickers=[tickers[0].lower()],
            catalog_file="JPMAQS_METADATA_CATALOG_20230101.parquet",
        )

        self.assertEqual(result, [JPMAQS_DATASET_THEME_MAPPING[themes[0]]])
        mock_logger.warning.assert_not_called()

    @patch("macrosynergy.download.dataquery_file_api.dataquery_file_api.logger")
    @patch(
        "macrosynergy.download.dataquery_file_api.dataquery_file_api.pd.read_parquet"
    )
    @patch(
        "macrosynergy.download.dataquery_file_api.dataquery_file_api.DataQueryFileAPIOauth"
    )
    def test_get_datasets_for_indicators_warns_on_unknown_themes(
        self, mock_oauth, mock_read_parquet, mock_logger
    ):
        client = DataQueryFileAPIClient(
            client_id="id", client_secret="secret", out_dir=self.test_dir
        )

        themes = list(JPMAQS_DATASET_THEME_MAPPING.keys())
        tickers = [f"TICKER_{i}" for i in range(len(themes))]
        unknown_theme = "New unseen theme"
        mock_read_parquet.return_value = pd.DataFrame(
            {"Theme": themes + [unknown_theme], "Ticker": tickers + ["BAD_TICKER"]}
        )

        result = client.get_datasets_for_indicators(
            tickers=[tickers[0]],
            catalog_file="JPMAQS_METADATA_CATALOG_20230101.parquet",
            case_sensitive=True,
        )

        self.assertEqual(result, [JPMAQS_DATASET_THEME_MAPPING[themes[0]]])
        mock_logger.warning.assert_called_once()
        warning_msg = mock_logger.warning.call_args[0][0]
        self.assertIn("unknown themes", warning_msg.lower())
        self.assertIn(unknown_theme, warning_msg)

    @patch("macrosynergy.download.dataquery_file_api.dataquery_file_api.logger")
    @patch.object(DataQueryFileAPIClient, "download_multiple_files")
    @patch.object(DataQueryFileAPIClient, "list_available_files_for_all_file_groups")
    def test_download_full_snapshot(
        self, mock_list_all_files, mock_download_multi, mock_logger
    ):
        class_dir = os.path.join(self.test_dir, "class", "dir")
        client = DataQueryFileAPIClient(
            client_id="id", client_secret="secret", out_dir=class_dir
        )
        self.assertEqual(Path(client.out_dir), Path(class_dir) / "jpmaqs-download")
        mock_list_all_files.return_value = pd.DataFrame(
            {
                "file-name": [
                    "C_delta_20250201T110456.parquet",
                    "A_metadata_20250201T110000.parquet",
                    "B_full_20250201.parquet",
                    "A_full_20250101.parquet",
                    "A_full_20250201.parquet",
                ],
                "file-datetime": [
                    "20250201T110456",
                    "20250201T110000",
                    "20250201T000000",
                    "20250101T000000",
                    "20250201T000000",
                ],
            }
        )
        client.download_full_snapshot(since_datetime="20250201", show_progress=False)

        expected_order = [
            "A_full_20250201.parquet",
            "B_full_20250201.parquet",
            "C_delta_20250201T110456.parquet",
            "A_metadata_20250201T110000.parquet",
        ]

        mock_download_multi.assert_called_once_with(
            filenames=expected_order,
            overwrite=False,
            chunk_size=None,
            timeout=300.0,
            show_progress=False,
        )

        mock_download_multi.reset_mock()
        method_dir = os.path.join(self.test_dir, "method", "dir")
        client2 = DataQueryFileAPIClient(
            client_id="id", client_secret="secret", out_dir=method_dir
        )
        self.assertEqual(Path(client2.out_dir), Path(method_dir) / "jpmaqs-download")
        client2.download_full_snapshot(since_datetime="20250201", show_progress=False)
        mock_download_multi.assert_called_once_with(
            filenames=expected_order,
            overwrite=False,
            chunk_size=None,
            timeout=300.0,
            show_progress=False,
        )

    @patch.object(DataQueryFileAPIClient, "download_multiple_files")
    @patch.object(DataQueryFileAPIClient, "list_available_files_for_all_file_groups")
    def test_download_full_snapshot_with_since_datetime(
        self, mock_list_all_files, mock_download_multi
    ):
        client = DataQueryFileAPIClient(
            client_id="id", client_secret="secret", out_dir=self.test_dir
        )
        mock_list_all_files.return_value = pd.DataFrame(
            {
                "file-name": ["f1.parquet"],
                "file-datetime": ["20230101T000000"],
            }
        )
        client.download_full_snapshot(since_datetime="20230101", show_progress=False)

        self.assertEqual(mock_download_multi.call_args[1]["filenames"], ["f1.parquet"])

    @patch(
        "macrosynergy.download.dataquery_file_api.dataquery_file_api.DataQueryFileAPIOauth"
    )
    @patch.object(DataQueryFileAPIClient, "download_multiple_files")
    @patch.object(DataQueryFileAPIClient, "list_downloaded_files")
    @patch.object(DataQueryFileAPIClient, "list_available_files_for_all_file_groups")
    def test_download_full_snapshot_metadata_skipped_when_local_present_and_up_to_date(
        self,
        mock_list_all_files,
        mock_list_downloaded_files,
        mock_download_multi,
        _mock_oauth,
    ):
        with tempfile.TemporaryDirectory() as td:
            meta_path = Path(td) / "A_metadata_20250201T110000.parquet"
            meta_path.write_bytes(b"x")

            api_df = pd.DataFrame(
                {
                    "file-name": [
                        "B_full_20250201.parquet",
                        "A_metadata_20250201T110000.parquet",
                    ],
                    "file-datetime": [
                        "20250201T000000",
                        "20250201T110000",
                    ],
                    "last-modified": pd.to_datetime(
                        ["2025-02-01T00:00:00Z", "2025-02-01T12:00:00Z"], utc=True
                    ),
                }
            )
            mock_list_all_files.return_value = api_df
            mock_list_downloaded_files.return_value = pd.DataFrame(
                {
                    "file-name": ["A_metadata_20250201T110000.parquet"],
                    "path": [str(meta_path)],
                    "last-modified": pd.to_datetime(["2025-02-01T12:00:00Z"], utc=True),
                }
            )

            client = DataQueryFileAPIClient(
                client_id="id", client_secret="secret", out_dir=self.test_dir
            )
            client.download_full_snapshot(
                since_datetime="20250201", show_progress=False
            )

            filenames = mock_download_multi.call_args[1]["filenames"]
            self.assertEqual(filenames, ["B_full_20250201.parquet"])

    @patch(
        "macrosynergy.download.dataquery_file_api.dataquery_file_api.DataQueryFileAPIOauth"
    )
    @patch.object(DataQueryFileAPIClient, "download_multiple_files")
    @patch.object(DataQueryFileAPIClient, "list_downloaded_files")
    @patch.object(DataQueryFileAPIClient, "list_available_files_for_all_file_groups")
    def test_download_full_snapshot_metadata_redownloaded_when_api_last_modified_newer(
        self,
        mock_list_all_files,
        mock_list_downloaded_files,
        mock_download_multi,
        _mock_oauth,
    ):
        with tempfile.TemporaryDirectory() as td:
            meta_path = Path(td) / "A_metadata_20250201T110000.parquet"
            meta_path.write_bytes(b"x")

            api_df = pd.DataFrame(
                {
                    "file-name": [
                        "B_full_20250201.parquet",
                        "A_metadata_20250201T110000.parquet",
                    ],
                    "file-datetime": [
                        "20250201T000000",
                        "20250201T110000",
                    ],
                    "last-modified": pd.to_datetime(
                        ["2025-02-01T00:00:00Z", "2025-02-01T13:00:00Z"], utc=True
                    ),
                }
            )
            mock_list_all_files.return_value = api_df
            mock_list_downloaded_files.return_value = pd.DataFrame(
                {
                    "file-name": ["A_metadata_20250201T110000.parquet"],
                    "path": [str(meta_path)],
                    "last-modified": pd.to_datetime(["2025-02-01T12:00:00Z"], utc=True),
                }
            )

            client = DataQueryFileAPIClient(
                client_id="id", client_secret="secret", out_dir=self.test_dir
            )
            client.download_full_snapshot(
                since_datetime="20250201", show_progress=False
            )

            filenames = mock_download_multi.call_args[1]["filenames"]
            self.assertEqual(
                filenames,
                [
                    "B_full_20250201.parquet",
                    "A_metadata_20250201T110000.parquet",
                ],
            )

    @patch(
        "macrosynergy.download.dataquery_file_api.dataquery_file_api.DataQueryFileAPIOauth"
    )
    @patch.object(DataQueryFileAPIClient, "download_multiple_files")
    @patch.object(DataQueryFileAPIClient, "list_downloaded_files")
    @patch.object(DataQueryFileAPIClient, "list_available_files_for_all_file_groups")
    def test_download_full_snapshot_include_metadata_false_excludes_metadata_files(
        self,
        mock_list_all_files,
        mock_list_downloaded_files,
        mock_download_multi,
        _mock_oauth,
    ):
        # Even if metadata rows are present in the API listing, `include_metadata=False`
        # should prevent them from being downloaded.
        api_df = pd.DataFrame(
            {
                "file-name": [
                    "B_full_20250201.parquet",
                    "A_metadata_20250201T110000.parquet",
                ],
                "file-datetime": [
                    "20250201T000000",
                    "20250201T110000",
                ],
            }
        )
        mock_list_all_files.return_value = api_df
        mock_list_downloaded_files.return_value = pd.DataFrame(
            columns=["file-name", "path", "last-modified"]
        )

        client = DataQueryFileAPIClient(
            client_id="id", client_secret="secret", out_dir=self.test_dir
        )
        client.download_full_snapshot(
            since_datetime="20250201",
            include_metadata=False,
            show_progress=False,
        )

        filenames = mock_download_multi.call_args[1]["filenames"]
        self.assertEqual(filenames, ["B_full_20250201.parquet"])

    @patch("pandas.Timestamp.utcnow")
    @patch(
        "macrosynergy.download.dataquery_file_api.dataquery_file_api.FileSelector.select_files_for_download"
    )
    @patch.object(DataQueryFileAPIClient, "list_available_files_for_all_file_groups")
    def test_download_full_snapshot_defaults_to_last_business_day_on_weekend(
        self, mock_list_all_files, mock_select, mock_utcnow
    ):
        client = DataQueryFileAPIClient(
            client_id="id", client_secret="secret", out_dir=self.test_dir
        )
        # Sunday (UTC)
        mock_utcnow.return_value = pd.Timestamp("2023-01-08T01:02:03Z")
        mock_list_all_files.return_value = pd.DataFrame(columns=["file-name"])
        mock_select.return_value = []

        client.download_full_snapshot(since_datetime=None, show_progress=False)

        if PYTHON_3_8_OR_LATER:
            called_since = mock_select.call_args.kwargs["since_datetime"]
        else:
            called_since = mock_select.call_args[1]["since_datetime"]
        self.assertEqual(called_since, "20230106")

    @patch("macrosynergy.download.dataquery_file_api.dataquery_file_api.logger")
    @patch.object(DataQueryFileAPIClient, "download_multiple_files")
    @patch.object(DataQueryFileAPIClient, "list_available_files_for_all_file_groups")
    def test_download_full_snapshot_no_new_files(
        self, mock_list_all_files, mock_download_multi, mock_logger
    ):
        client = DataQueryFileAPIClient(
            client_id="id", client_secret="secret", out_dir=self.test_dir
        )
        mock_list_all_files.return_value = pd.DataFrame(columns=["file-name"])
        client.download_full_snapshot(
            since_datetime="20230102T000000", show_progress=False
        )
        mock_download_multi.assert_not_called()
        mock_logger.info.assert_any_call("No new files to download.")

    @patch.object(DataQueryFileAPIClient, "download_multiple_files")
    @patch.object(DataQueryFileAPIClient, "list_available_files_for_all_file_groups")
    def test_download_full_snapshot_file_group_ids(
        self, mock_list_all_files, mock_download_multi
    ):
        client = DataQueryFileAPIClient(
            client_id="id", client_secret="secret", out_dir=self.test_dir
        )
        mock_list_all_files.return_value = pd.DataFrame(
            {
                "file-group-id": ["FG1", "FG2", "FG1"],
                "file-name": ["f1", "f2", "f3"],
                "file-datetime": ["20230101", "20230101", "20230101"],
            }
        )
        client.download_full_snapshot(
            since_datetime="20230101",
            file_group_ids=["FG1"],
            show_progress=False,
        )
        called_args = mock_download_multi.call_args[1]
        self.assertCountEqual(called_args["filenames"], ["f1", "f3"])

        with self.assertRaises(ValueError):
            client.download_full_snapshot(
                since_datetime="20230101", file_group_ids="not-a-list"
            )

    @patch("pandas.Timestamp.utcnow")
    @patch.object(DataQueryFileAPIClient, "download_full_snapshot")
    @patch.object(DataQueryFileAPIClient, "filter_to_valid_tickers")
    @patch.object(DataQueryFileAPIClient, "download_catalog_file")
    @patch(
        "macrosynergy.download.dataquery_file_api.dataquery_file_api.lazy_load_from_parquets"
    )
    @patch.object(DataQueryFileAPIClient, "get_datasets_for_indicators")
    def test_download_defaults_since_datetime_uses_utc(
        self,
        mock_get_datasets_for_indicators,
        mock_lazy_load,
        mock_download_catalog_file,
        mock_filter_to_valid_tickers,
        mock_download_full_snapshot,
        mock_utcnow,
    ):
        client = DataQueryFileAPIClient(
            client_id="id", client_secret="secret", out_dir=self.test_dir
        )
        mock_get_datasets_for_indicators.return_value = []
        mock_lazy_load.return_value = pd.DataFrame()
        mock_download_catalog_file.return_value = (
            "JPMAQS_METADATA_CATALOG_20230101.parquet"
        )
        mock_filter_to_valid_tickers.return_value = ["USD_GROWTH"]
        mock_utcnow.return_value = pd.Timestamp("2023-01-05T01:02:03Z")

        client.download(
            tickers=["USD_GROWTH"], since_datetime=None, show_progress=False
        )

        mock_download_full_snapshot.assert_called_once()
        # called_since = mock_download_full_snapshot.call_args.kwargs["since_datetime"]
        if PYTHON_3_8_OR_LATER:
            called_since = mock_download_full_snapshot.call_args.kwargs[
                "since_datetime"
            ]
            self.assertIsNone(mock_lazy_load.call_args.kwargs["since_datetime"])
            self.assertIsNone(mock_lazy_load.call_args.kwargs["to_datetime"])
        else:
            called_since = mock_download_full_snapshot.call_args[1]["since_datetime"]
            self.assertIsNone(mock_lazy_load.call_args[1]["since_datetime"])
            self.assertIsNone(mock_lazy_load.call_args[1]["to_datetime"])

        self.assertEqual(called_since, "20230105")

    @patch("pandas.Timestamp.utcnow")
    @patch.object(DataQueryFileAPIClient, "download_full_snapshot")
    @patch.object(DataQueryFileAPIClient, "filter_to_valid_tickers")
    @patch.object(DataQueryFileAPIClient, "download_catalog_file")
    @patch(
        "macrosynergy.download.dataquery_file_api.dataquery_file_api.lazy_load_from_parquets"
    )
    @patch.object(DataQueryFileAPIClient, "get_datasets_for_indicators")
    def test_download_defaults_since_datetime_uses_last_business_day_on_weekend(
        self,
        mock_get_datasets_for_indicators,
        mock_lazy_load,
        mock_download_catalog_file,
        mock_filter_to_valid_tickers,
        mock_download_full_snapshot,
        mock_utcnow,
    ):
        client = DataQueryFileAPIClient(
            client_id="id", client_secret="secret", out_dir=self.test_dir
        )
        mock_get_datasets_for_indicators.return_value = []
        mock_lazy_load.return_value = pd.DataFrame()
        mock_download_catalog_file.return_value = (
            "JPMAQS_METADATA_CATALOG_20230101.parquet"
        )
        mock_filter_to_valid_tickers.return_value = ["USD_GROWTH"]
        # Sunday (UTC)
        mock_utcnow.return_value = pd.Timestamp("2023-01-08T01:02:03Z")

        client.download(
            tickers=["USD_GROWTH"], since_datetime=None, show_progress=False
        )

        if PYTHON_3_8_OR_LATER:
            called_since = mock_download_full_snapshot.call_args.kwargs[
                "since_datetime"
            ]
        else:
            called_since = mock_download_full_snapshot.call_args[1]["since_datetime"]
        self.assertEqual(called_since, "20230106")

    @patch.object(DataQueryFileAPIClient, "load_data")
    @patch.object(DataQueryFileAPIClient, "download_full_snapshot")
    @patch.object(DataQueryFileAPIClient, "filter_to_valid_tickers")
    @patch.object(DataQueryFileAPIClient, "download_catalog_file")
    @patch.object(DataQueryFileAPIClient, "get_datasets_for_indicators")
    @patch.object(DataQueryFileAPIClient, "list_available_files_for_all_file_groups")
    def test_download_historical_to_datetime_bootstraps_full_delta_history(
        self,
        mock_list_all_files,
        mock_get_datasets_for_indicators,
        mock_download_catalog_file,
        mock_filter_to_valid_tickers,
        mock_download_full_snapshot,
        mock_load_data,
    ):
        client = DataQueryFileAPIClient(
            client_id="id", client_secret="secret", out_dir=self.test_dir
        )
        mock_download_catalog_file.return_value = (
            "JPMAQS_METADATA_CATALOG_20230101.parquet"
        )
        mock_filter_to_valid_tickers.return_value = ["USD_GROWTH"]
        mock_get_datasets_for_indicators.return_value = ["JPMAQS_GENERIC_RETURNS"]

        # Full snapshots exist, but only from a later date onward.
        mock_list_all_files.return_value = pd.DataFrame(
            {
                "file-name": [
                    "JPMAQS_GENERIC_RETURNS_20250602.parquet",
                    "JPMAQS_GENERIC_RETURNS_20250603.parquet",
                ],
                "file-datetime": [
                    pd.Timestamp("2025-06-02T00:00:00Z"),
                    pd.Timestamp("2025-06-03T00:00:00Z"),
                ],
            }
        )

        sentinel = pd.DataFrame({"a": [1]})
        mock_load_data.return_value = sentinel

        out = client.download(
            tickers=["USD_GROWTH"],
            to_datetime="20250328",
            show_progress=False,
            cleanup_old_files_n_days=None,
        )

        self.assertIs(out, sentinel)
        if PYTHON_3_8_OR_LATER:
            kwargs = mock_download_full_snapshot.call_args.kwargs
            self.assertEqual(kwargs["since_datetime"], "20250327")
            self.assertEqual(kwargs["to_datetime"], "20250328")
            self.assertTrue(kwargs["include_full_snapshots"])

            load_kwargs = mock_load_data.call_args.kwargs
            self.assertIsNone(load_kwargs["since_datetime"])
        else:
            kwargs = mock_download_full_snapshot.call_args[1]
            self.assertEqual(kwargs["since_datetime"], "20250327")
            self.assertEqual(kwargs["to_datetime"], "20250328")
            self.assertTrue(kwargs["include_full_snapshots"])

            load_kwargs = mock_load_data.call_args[1]
            self.assertIsNone(load_kwargs["since_datetime"])

    @patch(
        "macrosynergy.download.dataquery_file_api.dataquery_file_api.lazy_load_from_parquets"
    )
    @patch.object(DataQueryFileAPIClient, "download_catalog_file")
    @patch.object(DataQueryFileAPIClient, "filter_to_valid_tickers")
    @patch.object(DataQueryFileAPIClient, "get_datasets_for_indicators")
    def test_load_data_defaults_max_last_updated_from_to_datetime_date_only(
        self,
        mock_get_datasets_for_indicators,
        mock_filter_to_valid_tickers,
        mock_download_catalog_file,
        mock_lazy_load,
    ):
        client = DataQueryFileAPIClient(
            client_id="id", client_secret="secret", out_dir=self.test_dir
        )
        mock_download_catalog_file.return_value = (
            "JPMAQS_METADATA_CATALOG_20230101.parquet"
        )
        mock_filter_to_valid_tickers.return_value = ["USD_GROWTH"]
        mock_get_datasets_for_indicators.return_value = ["JPMAQS_GENERIC_RETURNS"]
        mock_lazy_load.return_value = pd.DataFrame()

        client.load_data(tickers=["USD_GROWTH"], to_datetime="20250328")

        if PYTHON_3_8_OR_LATER:
            kwargs = mock_lazy_load.call_args.kwargs
            passed_max = kwargs["max_last_updated"]
        else:
            kwargs = mock_lazy_load.call_args[1]
            passed_max = kwargs["max_last_updated"]

        expected = pd.Timestamp("2025-03-28T23:59:59.999999999Z")
        self.assertEqual(passed_max, expected)

    @patch(
        "macrosynergy.download.dataquery_file_api.dataquery_file_api.lazy_load_from_parquets"
    )
    @patch.object(DataQueryFileAPIClient, "list_downloaded_files")
    @patch.object(DataQueryFileAPIClient, "download_catalog_file")
    @patch.object(DataQueryFileAPIClient, "filter_to_valid_tickers")
    @patch.object(DataQueryFileAPIClient, "get_datasets_for_indicators")
    def test_load_data_raises_when_to_datetime_before_earliest_file_date(
        self,
        mock_get_datasets_for_indicators,
        mock_filter_to_valid_tickers,
        mock_download_catalog_file,
        mock_list_downloaded_files,
        mock_lazy_load,
    ):
        client = DataQueryFileAPIClient(
            client_id="id", client_secret="secret", out_dir=self.test_dir
        )
        mock_download_catalog_file.return_value = (
            "JPMAQS_METADATA_CATALOG_20230101.parquet"
        )
        mock_filter_to_valid_tickers.return_value = ["USD_GROWTH"]
        mock_get_datasets_for_indicators.return_value = ["JPMAQS_GENERIC_RETURNS"]
        mock_list_downloaded_files.return_value = pd.DataFrame(
            [
                {
                    "file-name": "JPMAQS_GENERIC_RETURNS_20220103.parquet",
                    "dataset": "JPMAQS_GENERIC_RETURNS",
                    "file-timestamp": pd.Timestamp("2022-01-03T00:00:00Z"),
                    "path": "x",
                }
            ]
        )

        with self.assertRaises(ValueError) as cm:
            client.load_data(tickers=["USD_GROWTH"], to_datetime="20210101")
        self.assertIn("2022-01-03T00:00:00Z", str(cm.exception))
        mock_lazy_load.assert_not_called()

    @patch.object(DataQueryFileAPIClient, "download")
    @patch.object(DataQueryFileAPIClient, "list_available_files_for_all_file_groups")
    @patch(
        "macrosynergy.download.dataquery_file_api.dataquery_file_api.DataQueryFileAPIOauth"
    )
    def test_download_as_of_date_only_defaults_to_eod_utc(
        self, _mock_oauth, mock_list_all_files, mock_download
    ):
        client = DataQueryFileAPIClient(
            client_id="id", client_secret="secret", out_dir=self.test_dir
        )
        mock_list_all_files.return_value = pd.DataFrame(
            [
                {
                    "file-name": "JPMAQS_METADATA_CATALOG_20220101.parquet",
                    "file-datetime": pd.Timestamp("2022-01-01T00:00:00Z"),
                    "is-available": True,
                }
            ]
        )
        sentinel = pd.DataFrame({"a": [1]})
        mock_download.return_value = sentinel

        out = client.download_as_of(
            tickers=["USD_GROWTH"],
            as_of_datetime="20250328",
            show_progress=False,
            cleanup_old_files_n_days=None,
        )

        self.assertIs(out, sentinel)
        if PYTHON_3_8_OR_LATER:
            kwargs = mock_download.call_args.kwargs
        else:
            kwargs = mock_download.call_args[1]

        self.assertEqual(kwargs["to_datetime"], "20250328")
        self.assertEqual(kwargs["max_last_updated"], "2025-03-28T23:59:59Z")
        self.assertIsNone(kwargs["since_datetime"])

    @patch.object(DataQueryFileAPIClient, "download")
    @patch.object(DataQueryFileAPIClient, "list_available_files_for_all_file_groups")
    @patch(
        "macrosynergy.download.dataquery_file_api.dataquery_file_api.DataQueryFileAPIOauth"
    )
    def test_download_as_of_datetime_preserves_time_component(
        self, _mock_oauth, mock_list_all_files, mock_download
    ):
        client = DataQueryFileAPIClient(
            client_id="id", client_secret="secret", out_dir=self.test_dir
        )
        mock_list_all_files.return_value = pd.DataFrame(
            [
                {
                    "file-name": "JPMAQS_METADATA_CATALOG_20220101.parquet",
                    "file-datetime": pd.Timestamp("2022-01-01T00:00:00Z"),
                    "is-available": True,
                }
            ]
        )
        mock_download.return_value = pd.DataFrame()

        client.download_as_of(
            tickers=["USD_GROWTH"],
            as_of_datetime="2025-03-28T12:34:56Z",
            show_progress=False,
            cleanup_old_files_n_days=None,
        )

        if PYTHON_3_8_OR_LATER:
            kwargs = mock_download.call_args.kwargs
        else:
            kwargs = mock_download.call_args[1]
        self.assertEqual(kwargs["to_datetime"], "2025-03-28T12:34:56Z")
        self.assertEqual(kwargs["max_last_updated"], "2025-03-28T12:34:56Z")
        self.assertIsNone(kwargs["since_datetime"])

    @patch.object(DataQueryFileAPIClient, "download")
    @patch.object(DataQueryFileAPIClient, "list_available_files_for_all_file_groups")
    @patch(
        "macrosynergy.download.dataquery_file_api.dataquery_file_api.DataQueryFileAPIOauth"
    )
    def test_download_as_of_raises_when_before_earliest_file_date(
        self, _mock_oauth, mock_list_all_files, mock_download
    ):
        client = DataQueryFileAPIClient(
            client_id="id", client_secret="secret", out_dir=self.test_dir
        )
        mock_list_all_files.return_value = pd.DataFrame(
            [
                {
                    "file-name": "JPMAQS_METADATA_CATALOG_20220101.parquet",
                    "file-datetime": pd.Timestamp("2022-01-01T00:00:00Z"),
                    "is-available": True,
                }
            ]
        )
        with self.assertRaises(ValueError) as cm:
            client.download_as_of(
                tickers=["USD_GROWTH"],
                as_of_datetime="20210101",
                show_progress=False,
                cleanup_old_files_n_days=None,
            )
        self.assertIn("2022-01-01T00:00:00Z", str(cm.exception))
        mock_download.assert_not_called()

    @patch.object(DataQueryFileAPIClient, "download")
    @patch("pandas.Timestamp.utcnow")
    @patch(
        "macrosynergy.download.dataquery_file_api.dataquery_file_api.DataQueryFileAPIOauth"
    )
    def test_download_as_of_future_datetime_raises(
        self, _mock_oauth, mock_now, mock_download
    ):
        client = DataQueryFileAPIClient(
            client_id="id", client_secret="secret", out_dir=self.test_dir
        )
        mock_now.return_value = pd.Timestamp("2026-02-11T12:34:56Z")
        mock_download.return_value = pd.DataFrame()

        with self.assertRaises(ValueError) as cm:
            client.download_as_of(
                tickers=["USD_GROWTH"],
                as_of_datetime="2029-10-08T06:00:00Z",
                show_progress=False,
                cleanup_old_files_n_days=None,
            )
        self.assertIn("now (UTC)", str(cm.exception))
        mock_download.assert_not_called()

    @patch.object(DataQueryFileAPIClient, "download")
    @patch("pandas.Timestamp.utcnow")
    @patch(
        "macrosynergy.download.dataquery_file_api.dataquery_file_api.DataQueryFileAPIOauth"
    )
    def test_download_as_of_future_date_only_raises(
        self, _mock_oauth, mock_now, mock_download
    ):
        client = DataQueryFileAPIClient(
            client_id="id", client_secret="secret", out_dir=self.test_dir
        )
        mock_now.return_value = pd.Timestamp("2026-02-11T12:34:56Z")
        mock_download.return_value = pd.DataFrame()

        with self.assertRaises(ValueError) as cm:
            client.download_as_of(
                tickers=["USD_GROWTH"],
                as_of_datetime="2029-10-08",
                show_progress=False,
                cleanup_old_files_n_days=None,
            )
        self.assertIn("today (UTC)", str(cm.exception))
        mock_download.assert_not_called()

    @patch("macrosynergy.download.dataquery_file_api.dataquery_file_api.logger")
    @patch.object(DataQueryFileAPIClient, "download_catalog_file")
    @patch.object(DataQueryFileAPIClient, "filter_to_valid_tickers")
    @patch.object(DataQueryFileAPIClient, "get_datasets_for_indicators")
    @patch(
        "macrosynergy.download.dataquery_file_api.dataquery_file_api.lazy_load_from_parquets"
    )
    def test_download_warns_for_missing_tickers_but_loads_valid(
        self,
        mock_lazy_load,
        mock_get_datasets_for_indicators,
        mock_filter_to_valid_tickers,
        mock_download_catalog_file,
        mock_logger,
    ):
        client = DataQueryFileAPIClient(
            client_id="id", client_secret="secret", out_dir=self.test_dir
        )
        self._write_local_catalog_file(client, file_datetime="20230101")
        mock_download_catalog_file.return_value = (
            "JPMAQS_METADATA_CATALOG_20230101.parquet"
        )
        mock_filter_to_valid_tickers.return_value = ["USD_GROWTH"]
        mock_get_datasets_for_indicators.return_value = []
        mock_lazy_load.return_value = pd.DataFrame()

        client.download(
            tickers=["USD_GROWTH", "BAD1", "BAD2", "BAD3", "BAD4", "BAD5", "BAD6"],
            skip_download=True,
            show_progress=False,
        )

        mock_logger.warning.assert_any_call(
            "6 tickers requested do not exist in the catalog, these are: "
            "[BAD1, BAD2, BAD3, BAD4, BAD5...1 more]"
        )
        mock_get_datasets_for_indicators.assert_called_once()

        if PYTHON_3_8_OR_LATER:
            result = mock_get_datasets_for_indicators.call_args.kwargs["tickers"]
        else:
            result = mock_get_datasets_for_indicators.call_args[1]["tickers"]

        self.assertEqual(result, ["USD_GROWTH"])

    @patch.object(DataQueryFileAPIClient, "download_catalog_file")
    @patch.object(DataQueryFileAPIClient, "filter_to_valid_tickers")
    def test_download_raises_when_no_valid_tickers(self, mock_filter, mock_catalog):
        client = DataQueryFileAPIClient(
            client_id="id", client_secret="secret", out_dir=self.test_dir
        )
        self._write_local_catalog_file(client, file_datetime="20230101")
        mock_catalog.return_value = "JPMAQS_METADATA_CATALOG_20230101.parquet"
        mock_filter.return_value = []

        with self.assertRaisesRegex(ValueError, "No valid tickers found"):
            client.download(
                tickers=["BAD1", "BAD2"],
                skip_download=True,
                show_progress=False,
            )

    @patch.object(DataQueryFileAPIClient, "download_catalog_file")
    @patch.object(DataQueryFileAPIClient, "filter_to_valid_tickers")
    @patch.object(DataQueryFileAPIClient, "get_datasets_for_indicators")
    @patch(
        "macrosynergy.download.dataquery_file_api.dataquery_file_api.lazy_load_from_parquets"
    )
    @patch.object(DataQueryFileAPIClient, "download_full_snapshot")
    @patch("pandas.Timestamp.utcnow", return_value=pd.Timestamp("2024-01-02T00:00:00Z"))
    def test_download_calls_cleanup_old_files_when_configured(
        self,
        _mock_utcnow,
        _mock_download_full_snapshot,
        mock_lazy_load,
        mock_get_datasets_for_indicators,
        mock_filter_to_valid_tickers,
        mock_download_catalog_file,
    ):
        client = DataQueryFileAPIClient(
            client_id="id", client_secret="secret", out_dir=self.test_dir
        )
        mock_download_catalog_file.return_value = (
            "JPMAQS_METADATA_CATALOG_20230101.parquet"
        )
        mock_filter_to_valid_tickers.return_value = ["USD_GROWTH"]
        mock_get_datasets_for_indicators.return_value = []
        mock_lazy_load.return_value = pd.DataFrame()

        with patch.object(client, "cleanup_old_files", return_value=[]) as mock_cleanup:
            client.download(
                tickers=["USD_GROWTH"],
                skip_download=False,
                cleanup_old_files_n_days=7,
                since_datetime="20240101",
                show_progress=False,
            )

        mock_cleanup.assert_called_once_with(days_to_keep=7)

    @patch("macrosynergy.download.dataquery_file_api.dataquery_file_api.logger")
    @patch.object(DataQueryFileAPIClient, "download_catalog_file")
    @patch.object(DataQueryFileAPIClient, "filter_to_valid_tickers")
    @patch.object(DataQueryFileAPIClient, "get_datasets_for_indicators")
    @patch(
        "macrosynergy.download.dataquery_file_api.dataquery_file_api.lazy_load_from_parquets"
    )
    @patch.object(DataQueryFileAPIClient, "download_full_snapshot")
    @patch("pandas.Timestamp.utcnow", return_value=pd.Timestamp("2024-01-11T00:00:00Z"))
    def test_download_cleanup_old_files_n_days_adjusts_to_since_datetime_calendar_days(
        self,
        _mock_utcnow,
        _mock_download_full_snapshot,
        mock_lazy_load,
        mock_get_datasets_for_indicators,
        mock_filter_to_valid_tickers,
        mock_download_catalog_file,
        mock_logger,
    ):
        client = DataQueryFileAPIClient(
            client_id="id", client_secret="secret", out_dir=self.test_dir
        )
        mock_download_catalog_file.return_value = (
            "JPMAQS_METADATA_CATALOG_20230101.parquet"
        )
        mock_filter_to_valid_tickers.return_value = ["USD_GROWTH"]
        mock_get_datasets_for_indicators.return_value = []
        mock_lazy_load.return_value = pd.DataFrame()

        with patch.object(client, "cleanup_old_files", return_value=[]) as mock_cleanup:
            client.download(
                tickers=["USD_GROWTH"],
                skip_download=False,
                cleanup_old_files_n_days=-2,
                since_datetime="20240101",
                show_progress=False,
            )

        mock_logger.warning.assert_not_called()
        mock_cleanup.assert_called_once_with(days_to_keep=2)

    @patch("macrosynergy.download.dataquery_file_api.dataquery_file_api.logger")
    @patch.object(DataQueryFileAPIClient, "download_catalog_file")
    @patch.object(DataQueryFileAPIClient, "filter_to_valid_tickers")
    @patch.object(DataQueryFileAPIClient, "get_datasets_for_indicators")
    @patch(
        "macrosynergy.download.dataquery_file_api.dataquery_file_api.lazy_load_from_parquets"
    )
    def test_download_does_not_call_cleanup_old_files_when_none(
        self,
        mock_lazy_load,
        mock_get_datasets_for_indicators,
        mock_filter_to_valid_tickers,
        mock_download_catalog_file,
        mock_logger,
    ):
        client = DataQueryFileAPIClient(
            client_id="id", client_secret="secret", out_dir=self.test_dir
        )
        self._write_local_catalog_file(client, file_datetime="20230101")
        mock_download_catalog_file.return_value = (
            "JPMAQS_METADATA_CATALOG_20230101.parquet"
        )
        mock_filter_to_valid_tickers.return_value = ["USD_GROWTH"]
        mock_get_datasets_for_indicators.return_value = []
        mock_lazy_load.return_value = pd.DataFrame()

        with patch.object(client, "cleanup_old_files", return_value=[]) as mock_cleanup:
            client.download(
                tickers=["USD_GROWTH"],
                skip_download=True,
                cleanup_old_files_n_days=None,
                show_progress=False,
            )

        mock_cleanup.assert_not_called()
        mock_logger.warning.assert_not_called()

    @patch("macrosynergy.download.dataquery_file_api.dataquery_file_api.logger")
    @patch.object(DataQueryFileAPIClient, "download_catalog_file")
    @patch.object(DataQueryFileAPIClient, "filter_to_valid_tickers")
    @patch.object(DataQueryFileAPIClient, "get_datasets_for_indicators")
    @patch(
        "macrosynergy.download.dataquery_file_api.dataquery_file_api.lazy_load_from_parquets"
    )
    def test_download_skip_download_ignores_cleanup_old_files_n_days(
        self,
        mock_lazy_load,
        mock_get_datasets_for_indicators,
        mock_filter_to_valid_tickers,
        mock_download_catalog_file,
        mock_logger,
    ):
        client = DataQueryFileAPIClient(
            client_id="id", client_secret="secret", out_dir=self.test_dir
        )
        self._write_local_catalog_file(client, file_datetime="20230101")
        mock_download_catalog_file.return_value = (
            "JPMAQS_METADATA_CATALOG_20230101.parquet"
        )
        mock_filter_to_valid_tickers.return_value = ["USD_GROWTH"]
        mock_get_datasets_for_indicators.return_value = []
        mock_lazy_load.return_value = pd.DataFrame()

        with patch.object(client, "cleanup_old_files", return_value=[]) as mock_cleanup:
            client.download(
                tickers=["USD_GROWTH"],
                skip_download=True,
                cleanup_old_files_n_days=7,
                show_progress=False,
            )

        mock_cleanup.assert_not_called()
        mock_logger.warning.assert_any_call(
            "`cleanup_old_files_n_days` is ignored when `skip_download=True`."
        )

    @patch.object(DataQueryFileAPIClient, "download_catalog_file")
    @patch.object(DataQueryFileAPIClient, "filter_to_valid_tickers")
    @patch.object(DataQueryFileAPIClient, "get_datasets_for_indicators")
    @patch(
        "macrosynergy.download.dataquery_file_api.dataquery_file_api.lazy_load_from_parquets"
    )
    @patch.object(DataQueryFileAPIClient, "download_full_snapshot")
    def test_download_cleanup_old_files_n_days_invalid_type_raises_when_downloading(
        self,
        _mock_download_full_snapshot,
        mock_lazy_load,
        mock_get_datasets_for_indicators,
        mock_filter_to_valid_tickers,
        mock_download_catalog_file,
    ):
        client = DataQueryFileAPIClient(
            client_id="id", client_secret="secret", out_dir=self.test_dir
        )
        mock_download_catalog_file.return_value = (
            "JPMAQS_METADATA_CATALOG_20230101.parquet"
        )
        mock_filter_to_valid_tickers.return_value = ["USD_GROWTH"]
        mock_get_datasets_for_indicators.return_value = []
        mock_lazy_load.return_value = pd.DataFrame()

        with self.assertRaisesRegex(ValueError, "`cleanup_old_files_n_days` must be"):
            client.download(
                tickers=["USD_GROWTH"],
                skip_download=False,
                cleanup_old_files_n_days="bad",  # type: ignore[arg-type]
                since_datetime="20240101",
                show_progress=False,
            )

    @patch.object(DataQueryFileAPIClient, "download_full_snapshot")
    @patch.object(DataQueryFileAPIClient, "download_catalog_file")
    @patch.object(DataQueryFileAPIClient, "filter_to_valid_tickers")
    @patch.object(DataQueryFileAPIClient, "get_datasets_for_indicators")
    @patch(
        "macrosynergy.download.dataquery_file_api.dataquery_file_api.lazy_load_from_parquets"
    )
    def test_load_data_skips_snapshot_download(
        self,
        mock_lazy_load,
        mock_get_datasets_for_indicators,
        mock_filter_to_valid_tickers,
        mock_download_catalog_file,
        mock_download_full_snapshot,
    ):
        client = DataQueryFileAPIClient(
            client_id="id", client_secret="secret", out_dir=self.test_dir
        )
        mock_download_catalog_file.return_value = (
            "JPMAQS_METADATA_CATALOG_20230101.parquet"
        )
        mock_filter_to_valid_tickers.return_value = ["USD_GROWTH"]
        mock_get_datasets_for_indicators.return_value = []
        mock_lazy_load.return_value = pd.DataFrame()

        client.load_data(tickers=["USD_GROWTH"])

        mock_download_catalog_file.assert_called_once()
        mock_filter_to_valid_tickers.assert_called_once()
        mock_get_datasets_for_indicators.assert_called_once()
        mock_download_full_snapshot.assert_not_called()
        mock_lazy_load.assert_called_once()

    @patch.object(DataQueryFileAPIClient, "download_catalog_file")
    @patch.object(DataQueryFileAPIClient, "filter_to_valid_tickers")
    @patch.object(DataQueryFileAPIClient, "get_datasets_for_indicators")
    @patch(
        "macrosynergy.download.dataquery_file_api.dataquery_file_api.lazy_load_from_parquets"
    )
    def test_load_data_uses_catalog_file_argument_without_downloading_catalog(
        self,
        mock_lazy_load,
        mock_get_datasets_for_indicators,
        mock_filter_to_valid_tickers,
        mock_download_catalog_file,
    ):
        client = DataQueryFileAPIClient(
            client_id="id", client_secret="secret", out_dir=self.test_dir
        )
        mock_filter_to_valid_tickers.return_value = ["USD_GROWTH"]
        mock_get_datasets_for_indicators.return_value = []
        mock_lazy_load.return_value = pd.DataFrame()

        client.load_data(
            tickers=["USD_GROWTH"],
            catalog_file="JPMAQS_METADATA_CATALOG_20230101.parquet",
        )

        mock_download_catalog_file.assert_not_called()
        mock_filter_to_valid_tickers.assert_called_once()
        mock_get_datasets_for_indicators.assert_called_once()
        mock_lazy_load.assert_called_once()

    @patch.object(DataQueryFileAPIClient, "load_data")
    @patch.object(DataQueryFileAPIClient, "download_full_snapshot")
    @patch.object(DataQueryFileAPIClient, "download_catalog_file")
    @patch.object(DataQueryFileAPIClient, "filter_to_valid_tickers")
    @patch.object(DataQueryFileAPIClient, "get_datasets_for_indicators")
    def test_download_delegates_to_load_data(
        self,
        mock_get_datasets_for_indicators,
        mock_filter_to_valid_tickers,
        mock_download_catalog_file,
        mock_download_full_snapshot,
        mock_load_data,
    ):
        client = DataQueryFileAPIClient(
            client_id="id", client_secret="secret", out_dir=self.test_dir
        )
        self._write_local_catalog_file(client, file_datetime="20230101")
        mock_download_catalog_file.return_value = (
            "JPMAQS_METADATA_CATALOG_20230101.parquet"
        )
        mock_filter_to_valid_tickers.return_value = ["USD_GROWTH"]
        mock_get_datasets_for_indicators.return_value = ["JPMAQS_GENERIC_RETURNS"]
        sentinel = pd.DataFrame({"a": [1]})
        mock_load_data.return_value = sentinel

        out = client.download(
            tickers=["USD_GROWTH"], skip_download=True, show_progress=False
        )

        self.assertIs(out, sentinel)
        mock_download_full_snapshot.assert_not_called()
        mock_load_data.assert_called_once()
        if PYTHON_3_8_OR_LATER:
            kwargs = mock_load_data.call_args.kwargs
            self.assertIn("datasets", kwargs)
            self.assertIn("catalog_file", kwargs)
            self.assertEqual(kwargs["tickers"], ["USD_GROWTH"])
            self.assertEqual(
                kwargs["datasets"],
                ["JPMAQS_GENERIC_RETURNS", "JPMAQS_GENERIC_RETURNS_DELTA"],
            )
        else:
            kwargs = mock_load_data.call_args[1]
            self.assertIn("datasets", kwargs)
            self.assertIn("catalog_file", kwargs)
            self.assertEqual(kwargs["tickers"], ["USD_GROWTH"])
            self.assertEqual(
                kwargs["datasets"],
                ["JPMAQS_GENERIC_RETURNS", "JPMAQS_GENERIC_RETURNS_DELTA"],
            )


class TestDataQueryFileAPIClientNotificationLoading(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.test_dir = self.temp_dir.name

    def _write_notification_json(self, path: Path, sub_title: str, data: list):
        payload = {
            "metadata": {
                "title": "JPMaQS Notifications",
                "sub_title": sub_title,
                "schema": "mock",
                "datetime": "2026-01-19T06:05:01Z",
                "notification_type": "mock",
                "message_type": "mock",
            },
            "data": data,
            "disclaimer": "mock",
            "tags": ["mock"],
        }
        path.write_text(json.dumps(payload), encoding="utf-8")

    @patch.object(DataQueryFileAPIClient, "list_downloaded_files")
    @patch.object(DataQueryFileAPIClient, "download_full_snapshot")
    def test_load_metadata_jsons_filters_normalizes_and_canonicalizes_titles(
        self, mock_download_full_snapshot, mock_list_downloaded_files
    ):
        client = DataQueryFileAPIClient(
            client_id="id", client_secret="secret", out_dir=self.test_dir
        )
        out_dir = Path(client.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        p_missing = out_dir / "JPMAQS_METADATA_NOTIFICATIONS_20260119T060501.json"
        p_changed = out_dir / "JPMAQS_METADATA_NOTIFICATIONS_20260119T060502.json"
        p_addl = out_dir / "JPMAQS_METADATA_NOTIFICATIONS_20260119T072451.json"

        self._write_notification_json(
            p_missing,
            sub_title="missing updates",
            data=[{"Ticker": "IEP_HPI_SA_P1M1ML12", "Last update": "2020-07-30"}],
        )
        self._write_notification_json(
            p_changed,
            sub_title="Changed historical values",
            data=[
                {
                    "Ticker": "AED_INFLRISK_NSA",
                    "Observations affected": 3576,
                    "Observations affected (%)": "92.38",
                    "Mean absolute value change": "0.01",
                    "Dates of changes": "2012-01-31/2026-01-15",
                }
            ],
        )
        self._write_notification_json(
            p_addl,
            sub_title="Additional information on missing updates",
            data=[
                {
                    "Ticker": "IDR_DU10YXR_NSA",
                    "Additional Information": "Some info",
                }
            ],
        )

        date = pd.Timestamp("2026-01-19T00:00:00Z")
        mock_list_downloaded_files.return_value = pd.DataFrame(
            [
                {
                    "file-name": p_missing.name,
                    "dataset": "JPMAQS_METADATA_NOTIFICATIONS",
                    "file-timestamp": date + pd.Timedelta(hours=6),
                    "path": str(p_missing),
                },
                {
                    "file-name": p_changed.name,
                    "dataset": "JPMAQS_METADATA_NOTIFICATIONS",
                    "file-timestamp": date + pd.Timedelta(hours=6, minutes=1),
                    "path": str(p_changed),
                },
                {
                    "file-name": p_addl.name,
                    "dataset": "JPMAQS_METADATA_NOTIFICATIONS",
                    "file-timestamp": date + pd.Timedelta(hours=7, minutes=24),
                    "path": str(p_addl),
                },
                {
                    "file-name": "OTHER_20260119T000000.json",
                    "dataset": "OTHER",
                    "file-timestamp": date,
                    "path": str(out_dir / "OTHER_20260119T000000.json"),
                },
            ]
        )

        result = client._load_metadata_jsons(
            date="2026-01-19",
            normalize_headers=True,
            skip_download=True,
        )

        self.assertIn("Missing Updates", result)
        self.assertIn("Changed historical values", result)
        self.assertIn("Additional information on missing updates", result)

        changed = result["Changed historical values"]
        self.assertEqual(
            sorted(changed.columns.tolist()),
            sorted(
                [
                    "ticker",
                    "observations_affected",
                    "observations_affected_pct",
                    "mean_absolute_value_change",
                    "dates_of_changes",
                ]
            ),
        )

        missing = result["Missing Updates"]
        self.assertEqual(missing.columns.tolist(), ["ticker", "last_update"])

        addl = result["Additional information on missing updates"]
        self.assertEqual(addl.columns.tolist(), ["ticker", "additional_information"])

        mock_download_full_snapshot.assert_not_called()

    @patch("macrosynergy.download.dataquery_file_api.dataquery_file_api.logger")
    @patch("pandas.Timestamp.utcnow")
    @patch.object(DataQueryFileAPIClient, "list_downloaded_files")
    @patch.object(DataQueryFileAPIClient, "download_full_snapshot")
    def test_load_metadata_jsons_future_date_warns_and_downloads(
        self,
        mock_download_full_snapshot,
        mock_list_downloaded_files,
        mock_now,
        mock_logger,
    ):
        client = DataQueryFileAPIClient(
            client_id="id", client_secret="secret", out_dir=self.test_dir
        )
        mock_now.return_value = pd.Timestamp("2026-01-19T12:00:00Z")
        mock_list_downloaded_files.return_value = pd.DataFrame(
            {
                "file-name": pd.Series([], dtype="object"),
                "dataset": pd.Series([], dtype="object"),
                "file-timestamp": pd.Series([], dtype="datetime64[ns, UTC]"),
                "path": pd.Series([], dtype="object"),
            }
        )

        with self.assertRaisesRegex(ValueError, "future"):
            client._load_metadata_jsons(date="2026-01-25")
        mock_logger.warning.assert_not_called()
        mock_download_full_snapshot.assert_not_called()
        mock_list_downloaded_files.assert_not_called()

    @patch("macrosynergy.download.dataquery_file_api.dataquery_file_api.logger")
    def test_get_revisions_notifications(self, mock_logger):
        client = DataQueryFileAPIClient(
            client_id="id", client_secret="secret", out_dir=self.test_dir
        )
        df = pd.DataFrame({"ticker": ["X"], "observations_affected": [1]})
        with patch.object(
            DataQueryFileAPIClient, "_load_metadata_jsons", return_value={}
        ):
            out = client.get_revisions_notifications()
            self.assertTrue(out.empty)
            mock_logger.warning.assert_called_with(
                "No `Changed historical values` notifications found."
            )

        with patch.object(
            DataQueryFileAPIClient,
            "_load_metadata_jsons",
            return_value={"Changed historical values": df},
        ):
            out = client.get_revisions_notifications()
            pd.testing.assert_frame_equal(out, df)

    @patch("macrosynergy.download.dataquery_file_api.dataquery_file_api.logger")
    def test_get_missing_data_notifications_merge_and_fallbacks(self, mock_logger):
        client = DataQueryFileAPIClient(
            client_id="id", client_secret="secret", out_dir=self.test_dir
        )
        df1 = pd.DataFrame({"ticker": ["A", "B"], "last_update": ["2020-01-01", None]})
        df2 = pd.DataFrame({"ticker": ["A"], "additional_information": ["info"]})

        with patch.object(
            DataQueryFileAPIClient,
            "_load_metadata_jsons",
            return_value={
                "Missing Updates": df1,
                "Additional information on missing updates": df2,
            },
        ):
            out = client.get_missing_data_notifications()
            self.assertEqual(out["ticker"].tolist(), ["A", "B"])
            self.assertIn("additional_information", out.columns)

        mock_logger.reset_mock()
        with patch.object(
            DataQueryFileAPIClient,
            "_load_metadata_jsons",
            return_value={"Missing Updates": df1},
        ):
            out = client.get_missing_data_notifications()
            pd.testing.assert_frame_equal(out, df1)
            mock_logger.warning.assert_called_with(
                "No `Additional information on missing updates` notifications found."
            )

        mock_logger.reset_mock()
        with patch.object(
            DataQueryFileAPIClient,
            "_load_metadata_jsons",
            return_value={"Additional information on missing updates": df2},
        ):
            out = client.get_missing_data_notifications()
            pd.testing.assert_frame_equal(out, df2)
            mock_logger.warning.assert_called_with(
                "No `Missing Updates` notifications found."
            )

        mock_logger.reset_mock()
        with patch.object(
            DataQueryFileAPIClient, "_load_metadata_jsons", return_value={}
        ):
            out = client.get_missing_data_notifications()
            self.assertTrue(out.empty)
            mock_logger.warning.assert_called_with(
                "No `Missing Updates` or related notifications found."
            )


class TestResolveBaseUrl(unittest.TestCase):
    """Tests for the module-level _resolve_base_url URL-fallback logic."""

    PRIMARY = "https://primary.example.com/api/v2"
    FALLBACK = "https://fallback.example.com/api/v2"

    def setUp(self):
        _base_url_cache.clear()

    def tearDown(self):
        _base_url_cache.clear()

    def _assert_no_user_warnings(self, callable_fn):
        """Call *callable_fn* and assert it emits no UserWarning."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = callable_fn()
        user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
        self.assertEqual(user_warnings, [], f"Unexpected UserWarnings: {user_warnings}")
        return result

    @patch("requests.head")
    def test_primary_reachable_returns_primary(self, mock_head):
        """When the primary URL is reachable, return it with no warning."""
        mock_head.return_value = MagicMock(status_code=200)

        result = self._assert_no_user_warnings(
            lambda: _resolve_base_url(self.PRIMARY, self.FALLBACK)
        )

        self.assertEqual(result, self.PRIMARY)
        mock_head.assert_called_once()
        self.assertEqual(_base_url_cache[self.PRIMARY], self.PRIMARY)

    @patch("requests.head")
    def test_primary_unreachable_fallback_works(self, mock_head):
        """When primary is unreachable but fallback responds, return fallback + warn."""

        def _side_effect(url, **kwargs):
            if url == self.PRIMARY:
                raise requests.exceptions.ConnectionError("unreachable")
            return MagicMock(status_code=200)

        mock_head.side_effect = _side_effect

        with self.assertWarns(UserWarning) as cm:
            result = _resolve_base_url(self.PRIMARY, self.FALLBACK)

        self.assertEqual(result, self.FALLBACK)
        self.assertIn("not reachable", str(cm.warning))
        self.assertIn(self.PRIMARY, str(cm.warning))
        self.assertIn(self.FALLBACK, str(cm.warning))
        self.assertEqual(_base_url_cache[self.PRIMARY], self.FALLBACK)

    @patch("requests.head")
    def test_primary_timeout_falls_back(self, mock_head):
        """A timeout on the primary URL should trigger the fallback path."""

        def _side_effect(url, **kwargs):
            if url == self.PRIMARY:
                raise requests.exceptions.Timeout("timed out")
            return MagicMock(status_code=200)

        mock_head.side_effect = _side_effect

        with self.assertWarns(UserWarning):
            result = _resolve_base_url(self.PRIMARY, self.FALLBACK)

        self.assertEqual(result, self.FALLBACK)

    @patch("requests.head")
    def test_both_unreachable_returns_primary(self, mock_head):
        """When both URLs fail, return primary (let normal error handling surface it)."""
        mock_head.side_effect = requests.exceptions.ConnectionError("unreachable")

        result = self._assert_no_user_warnings(
            lambda: _resolve_base_url(self.PRIMARY, self.FALLBACK)
        )

        self.assertEqual(result, self.PRIMARY)
        self.assertEqual(mock_head.call_count, 2)
        self.assertEqual(_base_url_cache[self.PRIMARY], self.PRIMARY)

    @patch("requests.head")
    def test_cache_prevents_second_probe(self, mock_head):
        """After the first probe, subsequent calls use the cache (no network)."""
        mock_head.return_value = MagicMock(status_code=200)

        result1 = self._assert_no_user_warnings(
            lambda: _resolve_base_url(self.PRIMARY, self.FALLBACK)
        )
        self.assertEqual(mock_head.call_count, 1)

        result2 = _resolve_base_url(self.PRIMARY, self.FALLBACK)
        self.assertEqual(result1, result2)
        self.assertEqual(mock_head.call_count, 1)  # no additional probe

    @patch("requests.head")
    def test_cache_persists_across_client_instances(self, mock_head):
        """The cache is module-level: creating a new client re-uses the cached URL."""

        def _side_effect(url, **kwargs):
            if url == DQ_FILE_API_BASE_URL:
                raise requests.exceptions.ConnectionError("unreachable")
            return MagicMock(status_code=200)

        mock_head.side_effect = _side_effect

        with patch.dict(os.environ, {"DQ_CLIENT_ID": "x", "DQ_CLIENT_SECRET": "y"}):
            with self.assertWarns(UserWarning):
                client1 = DataQueryFileAPIClient()

            probe_count = mock_head.call_count  # 2 calls (primary fail + fallback ok)

            client2 = DataQueryFileAPIClient()
            self.assertEqual(mock_head.call_count, probe_count)  # no new probes

            self.assertEqual(client1.base_url, DQ_FILE_API_FALLBACK_BASE_URL)
            self.assertEqual(client2.base_url, DQ_FILE_API_FALLBACK_BASE_URL)

    @patch("requests.head")
    def test_http_error_counts_as_reachable(self, mock_head):
        """Any HTTP response (even 401/403) means the server is reachable."""
        mock_head.return_value = MagicMock(status_code=401)

        result = self._assert_no_user_warnings(
            lambda: _resolve_base_url(self.PRIMARY, self.FALLBACK)
        )

        self.assertEqual(result, self.PRIMARY)
        mock_head.assert_called_once()

    @patch("requests.head")
    def test_passes_verify_and_proxies(self, mock_head):
        """Ensure verify/proxies kwargs are forwarded to requests.head."""
        mock_head.return_value = MagicMock(status_code=200)
        proxies = {"https": "http://proxy:8080"}

        result = _resolve_base_url(
            self.PRIMARY, self.FALLBACK, verify=False, proxies=proxies
        )

        self.assertEqual(result, self.PRIMARY)
        mock_head.assert_called_once_with(
            self.PRIMARY, timeout=10.0, verify=False, proxies=proxies
        )

    @patch("requests.head")
    def test_cached_fallback_returns_silently(self, mock_head):
        """After fallback is cached, subsequent calls return it without re-warning."""

        def _side_effect(url, **kwargs):
            if url == self.PRIMARY:
                raise requests.exceptions.ConnectionError("unreachable")
            return MagicMock(status_code=200)

        mock_head.side_effect = _side_effect

        # First call: warns and caches fallback
        with self.assertWarns(UserWarning):
            _resolve_base_url(self.PRIMARY, self.FALLBACK)

        # Second call: must return fallback silently, no new network call
        result = self._assert_no_user_warnings(
            lambda: _resolve_base_url(self.PRIMARY, self.FALLBACK)
        )
        self.assertEqual(result, self.FALLBACK)
        self.assertEqual(mock_head.call_count, 2)  # no additional probes

    @patch("requests.head")
    def test_different_primaries_are_independent(self, mock_head):
        """Two different primary URLs each probe independently."""
        other_primary = "https://other.example.com/api/v2"
        mock_head.return_value = MagicMock(status_code=200)

        r1 = _resolve_base_url(self.PRIMARY, self.FALLBACK)
        r2 = _resolve_base_url(other_primary, self.FALLBACK)

        self.assertEqual(r1, self.PRIMARY)
        self.assertEqual(r2, other_primary)
        self.assertEqual(mock_head.call_count, 2)
        self.assertIn(self.PRIMARY, _base_url_cache)
        self.assertIn(other_primary, _base_url_cache)

    @patch("requests.head")
    def test_ssl_error_triggers_fallback(self, mock_head):
        """SSLError (a RequestException subclass) on primary triggers fallback."""

        def _side_effect(url, **kwargs):
            if url == self.PRIMARY:
                raise requests.exceptions.SSLError("cert verify failed")
            return MagicMock(status_code=200)

        mock_head.side_effect = _side_effect

        with self.assertWarns(UserWarning):
            result = _resolve_base_url(self.PRIMARY, self.FALLBACK)

        self.assertEqual(result, self.FALLBACK)

    @patch("requests.head")
    def test_both_fail_with_different_exceptions(self, mock_head):
        """Primary=ConnectionError, Fallback=Timeout - both fail, returns primary."""
        call_count = 0

        def _side_effect(url, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise requests.exceptions.ConnectionError("unreachable")
            raise requests.exceptions.Timeout("timed out")

        mock_head.side_effect = _side_effect

        result = self._assert_no_user_warnings(
            lambda: _resolve_base_url(self.PRIMARY, self.FALLBACK)
        )

        self.assertEqual(result, self.PRIMARY)
        self.assertEqual(mock_head.call_count, 2)

    @patch("requests.head")
    def test_custom_timeout_is_forwarded(self, mock_head):
        """Custom timeout value is passed through to requests.head."""
        mock_head.return_value = MagicMock(status_code=200)

        _resolve_base_url(self.PRIMARY, self.FALLBACK, timeout=3.0)

        mock_head.assert_called_once_with(
            self.PRIMARY, timeout=3.0, verify=True, proxies=None
        )

    @patch("requests.head")
    def test_warning_includes_whitelist_guidance(self, mock_head):
        """The fallback warning must include whitelisting guidance."""

        def _side_effect(url, **kwargs):
            if url == self.PRIMARY:
                raise requests.exceptions.ConnectionError("unreachable")
            return MagicMock(status_code=200)

        mock_head.side_effect = _side_effect

        with self.assertWarns(UserWarning) as cm:
            _resolve_base_url(self.PRIMARY, self.FALLBACK)

        self.assertIn("whitelist", str(cm.warning).lower())

    @patch("macrosynergy.download.dataquery_file_api.dataquery_file_api.logger")
    @patch("requests.head")
    def test_logs_debug_on_primary_failure(self, mock_head, mock_logger):
        """A debug message is logged when the primary URL fails."""

        def _side_effect(url, **kwargs):
            if url == self.PRIMARY:
                raise requests.exceptions.ConnectionError("unreachable")
            return MagicMock(status_code=200)

        mock_head.side_effect = _side_effect

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _resolve_base_url(self.PRIMARY, self.FALLBACK)

        mock_logger.debug.assert_called_once()
        debug_msg = mock_logger.debug.call_args[0][0]
        self.assertIn("not reachable", debug_msg.lower())

    @patch("macrosynergy.download.dataquery_file_api.dataquery_file_api.logger")
    @patch("requests.head")
    def test_logs_warning_on_fallback_activation(self, mock_head, mock_logger):
        """A warning-level log is emitted when fallback is activated."""

        def _side_effect(url, **kwargs):
            if url == self.PRIMARY:
                raise requests.exceptions.ConnectionError("unreachable")
            return MagicMock(status_code=200)

        mock_head.side_effect = _side_effect

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _resolve_base_url(self.PRIMARY, self.FALLBACK)

        mock_logger.warning.assert_called_once()
        log_msg = mock_logger.warning.call_args[0][0]
        self.assertIn("fallback", log_msg.lower())

    @patch("requests.head")
    def test_http_500_counts_as_reachable(self, mock_head):
        """HTTP 500 is a server error but means the host is reachable."""
        mock_head.return_value = MagicMock(status_code=500)

        result = self._assert_no_user_warnings(
            lambda: _resolve_base_url(self.PRIMARY, self.FALLBACK)
        )

        self.assertEqual(result, self.PRIMARY)
        mock_head.assert_called_once()

    @patch("requests.head")
    def test_prepopulated_cache_is_respected(self, mock_head):
        """If the cache is pre-populated, no network call is made."""
        _base_url_cache[self.PRIMARY] = self.FALLBACK

        result = _resolve_base_url(self.PRIMARY, self.FALLBACK)

        self.assertEqual(result, self.FALLBACK)
        mock_head.assert_not_called()


if __name__ == "__main__":
    unittest.main()
