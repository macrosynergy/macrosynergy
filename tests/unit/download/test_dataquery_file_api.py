import os
import unittest
import warnings
from pathlib import Path
import json
import pandas as pd
import requests
from unittest.mock import patch, MagicMock
import functools
import logging
import tempfile
from macrosynergy.compat import PD_2_0_OR_LATER
from macrosynergy.download import dataquery_file_api as dq_file_api
from macrosynergy.download.dataquery_file_api import (
    validate_dq_timestamp,
    get_client_id_secret,
    DataQueryFileAPIClient,
    JPMAQS_DATASET_THEME_MAPPING,
    DownloadError,
    InvalidResponseError,
    DQ_FILE_API_SCOPE,
    _resolve_base_url,
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


def _make_client(out_dir="."):
    """Build a client without touching the network (oauth + URL probe stubbed)."""
    with patch("macrosynergy.download.dataquery_file_api.DataQueryFileAPIOauth"):
        with patch(
            "macrosynergy.download.dataquery_file_api._resolve_base_url",
            return_value="http://x",
        ):
            return DataQueryFileAPIClient(
                client_id="id", client_secret="secret", out_dir=out_dir
            )


class TestStandaloneFunctions(unittest.TestCase):
    def test_validate_dq_timestamp(self):
        self.assertTrue(validate_dq_timestamp("20230101"))
        self.assertTrue(validate_dq_timestamp("20230101T123000"))
        with self.assertRaises(ValueError):
            validate_dq_timestamp("invalid-date")
        self.assertFalse(validate_dq_timestamp("invalid-date", raise_error=False))
        with self.assertRaisesRegex(ValueError, "Invalid `my_ts` format"):
            validate_dq_timestamp("invalid-date", var_name="my_ts")

    @patch("macrosynergy.download.dataquery_file_api.os.getenv")
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


class TestDataQueryFileAPIClient(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.test_dir = self.temp_dir.name

    @patch("macrosynergy.download.dataquery_file_api.DataQueryFileAPIOauth")
    @patch(
        "macrosynergy.download.dataquery_file_api.get_client_id_secret",
        return_value=(None, None),
    )
    def test_init_no_credentials_raises_error(self, mock_get_client, mock_oauth):
        with self.assertRaisesRegex(
            ValueError, "Client ID and Client Secret must be provided"
        ):
            DataQueryFileAPIClient(out_dir=self.test_dir)

    @patch("macrosynergy.download.dataquery_file_api.DataQueryFileAPIOauth")
    def test_init_with_args(self, mock_oauth_constructor):
        test_dir = os.path.join(self.test_dir, "test", "dir")
        client = DataQueryFileAPIClient(
            client_id="arg_id", client_secret="arg_secret", out_dir=test_dir
        )
        self.assertEqual(client.client_id, "arg_id")
        self.assertEqual(client.out_dir, Path(test_dir).expanduser().resolve())
        mock_oauth_constructor.assert_called_once_with(
            client_id="arg_id",
            client_secret="arg_secret",
            resource=DQ_FILE_API_SCOPE,
            verify=True,
        )

    @patch("macrosynergy.download.dataquery_file_api.DataQueryFileAPIOauth")
    @patch(
        "macrosynergy.download.dataquery_file_api.get_client_id_secret",
        return_value=("env_id", "env_secret"),
    )
    def test_init_with_env_vars(self, mock_get_client, mock_oauth_constructor):
        client = DataQueryFileAPIClient()
        self.assertEqual(client.client_id, "env_id")
        self.assertEqual(client.client_secret, "env_secret")
        self.assertEqual(client.out_dir, Path("~/jpmaqs-data").expanduser().resolve())
        mock_get_client.assert_called_once()
        mock_oauth_constructor.assert_called_once_with(
            client_id="env_id",
            client_secret="env_secret",
            resource=DQ_FILE_API_SCOPE,
            verify=True,
        )

    @patch("macrosynergy.download.dataquery_file_api.logger")
    @patch("macrosynergy.download.dataquery_file_api.DataQueryFileAPIOauth")
    def test_context_manager_exception_logging(self, mock_oauth, mock_logger):
        with self.assertRaises(ValueError):
            with DataQueryFileAPIClient(
                client_id="id", client_secret="secret", out_dir=self.test_dir
            ):
                raise ValueError("Test Exception")
        mock_logger.error.assert_called_once()

    @suppress_logging
    @patch("macrosynergy.download.dataquery_file_api.DataQueryFileAPIOauth")
    @patch("macrosynergy.download.dataquery_file_api.request_wrapper")
    @patch("macrosynergy.download.dataquery_file_api.time.sleep", MagicMock())
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

    @patch("macrosynergy.download.dataquery_file_api.utc_now")
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
    @patch.object(DataQueryFileAPIClient, "list_group_files")
    def test_list_available_files_for_all_with_conversion(
        self, mock_list_groups, mock_list_available
    ):
        client = DataQueryFileAPIClient(
            client_id="id", client_secret="secret", out_dir=self.test_dir
        )
        mock_list_groups.return_value = pd.DataFrame({"file-group-id": ["FG1"]})
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
    @patch.object(DataQueryFileAPIClient, "list_group_files")
    def test_list_available_files_for_all_missing_column_error(
        self, mock_list_groups, mock_list_available
    ):
        client = DataQueryFileAPIClient(
            client_id="id", client_secret="secret", out_dir=self.test_dir
        )
        mock_list_groups.return_value = pd.DataFrame({"file-group-id": ["FG1"]})
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

    @patch("macrosynergy.download.dataquery_file_api.logger")
    @patch("macrosynergy.download.dataquery_file_api.utc_now")
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

        client.check_file_availability(
            filename="JPMAQS_GENERIC_RETURNS_20250501.parquet"
        )
        mock_get.assert_called_with(
            "/group/file/availability",
            {"file-group-id": "JPMAQS_GENERIC_RETURNS", "file-datetime": "20250501"},
        )

        # Invalid cases
        with self.assertRaises(ValueError):
            client.check_file_availability()

        # a filename with no "_" cannot be split into a group id and a datetime
        with self.assertRaises(ValueError):
            client.check_file_availability(filename="file.parquet")

        with self.assertRaises(ValueError):
            client.check_file_availability(
                filename="f.pq", file_group_id="FG", file_datetime="20230101"
            )

    @suppress_logging
    @patch("macrosynergy.download.dataquery_file_api.Path")
    @patch("macrosynergy.download.dataquery_file_api.DataQueryFileAPIOauth")
    def test_download_file_no_overwrite(self, mock_oauth, mock_path):
        client = DataQueryFileAPIClient(
            client_id="id", client_secret="secret", out_dir=self.test_dir
        )
        mock_final_path = MagicMock()
        mock_path.return_value.__truediv__.return_value.__truediv__.return_value = (
            mock_final_path
        )
        mock_final_path.exists.return_value = True
        result = client.download_file(
            filename="TEST_FULL_20230101.parquet", overwrite=False
        )
        mock_final_path.unlink.assert_not_called()
        self.assertEqual(result, str(mock_final_path))

    @suppress_logging
    @patch("macrosynergy.download.dataquery_file_api._delete_jpmaqs_file")
    @patch("macrosynergy.download.dataquery_file_api.SegmentedFileDownloader")
    @patch("macrosynergy.download.dataquery_file_api.Path")
    @patch("macrosynergy.download.dataquery_file_api.DataQueryFileAPIOauth")
    def test_download_file_overwrite(
        self, mock_oauth, mock_path, mock_segmented_downloader, mock_delete
    ):
        client = DataQueryFileAPIClient(
            client_id="id", client_secret="secret", out_dir=self.test_dir
        )
        mock_final_path = MagicMock()
        mock_path.return_value.__truediv__.return_value.__truediv__.return_value = (
            mock_final_path
        )
        mock_final_path.exists.return_value = True

        client.download_file(filename="TEST_FULL_20230101.parquet", overwrite=True)
        # deletion of the existing file goes through the central JPMaQS-only deleter
        mock_delete.assert_called_once_with(mock_final_path)

    @suppress_logging
    @patch(
        "macrosynergy.download.dataquery_file_api.request_wrapper_stream_bytes_to_disk"
    )
    @patch("macrosynergy.download.dataquery_file_api.SegmentedFileDownloader")
    @patch("macrosynergy.download.dataquery_file_api.Path")
    @patch("macrosynergy.download.dataquery_file_api.DataQueryFileAPIOauth")
    def test_download_file_small_file_logic(
        self, mock_oauth, mock_path, mock_segmented_downloader, mock_request_wrapper
    ):
        client = DataQueryFileAPIClient(
            client_id="id", client_secret="secret", out_dir=self.test_dir
        )
        mock_file_path = MagicMock()
        mock_file_path.exists.return_value = False
        mock_path.return_value.__truediv__.return_value.__truediv__.return_value = (
            mock_file_path
        )

        client.download_file(filename="TEST_DELTA_20230101.parquet")
        mock_request_wrapper.assert_called_once()
        mock_segmented_downloader.assert_not_called()

        mock_request_wrapper.reset_mock()
        mock_segmented_downloader.reset_mock()

        client.download_file(filename="TEST_METADATA_20230101.parquet")
        mock_request_wrapper.assert_called_once()
        mock_segmented_downloader.assert_not_called()

    @patch("macrosynergy.download.dataquery_file_api.DataQueryFileAPIOauth")
    def test_download_file_invalid_filename_format(self, mock_oauth):
        client = DataQueryFileAPIClient(
            client_id="id", client_secret="secret", out_dir=self.test_dir
        )
        with self.assertRaisesRegex(ValueError, "Invalid filename format"):
            client.download_file(filename="invalidformat.parquet")

    @patch(
        "macrosynergy.download.dataquery_file_api.request_wrapper_stream_bytes_to_disk"
    )
    @patch("macrosynergy.download.dataquery_file_api.Path")
    @patch("macrosynergy.download.dataquery_file_api.DataQueryFileAPIOauth")
    def test_download_file_writes_raw_file_and_returns_path(
        self, mock_oauth, mock_path, mock_downloader
    ):
        client = DataQueryFileAPIClient(
            client_id="id", client_secret="secret", out_dir=self.test_dir
        )
        mock_file_path = MagicMock()
        mock_file_path.exists.return_value = False
        mock_file_path.suffix = ".parquet"
        mock_path.return_value.__truediv__.return_value.__truediv__.return_value = (
            mock_file_path
        )

        result = client.download_file(filename="TEST_DATA_20230101.parquet")

        # the raw file is streamed straight to its final path (no temp/converted path)
        # and that same path is returned unchanged
        mock_downloader.assert_called_once()
        self.assertEqual(mock_downloader.call_args[1]["filename"], str(mock_file_path))
        self.assertEqual(result, str(mock_file_path))

    @patch("macrosynergy.download.dataquery_file_api.cf.as_completed")
    @patch("macrosynergy.download.dataquery_file_api.cf.ThreadPoolExecutor")
    @patch("macrosynergy.download.dataquery_file_api.DataQueryFileAPIOauth")
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

    @patch("macrosynergy.download.dataquery_file_api.cf.as_completed", return_value=[])
    @patch("macrosynergy.download.dataquery_file_api.cf.ThreadPoolExecutor")
    @patch("macrosynergy.download.dataquery_file_api.DataQueryFileAPIOauth")
    def test_download_multiple_files_n_jobs_passed_straight_through(
        self, mock_oauth, mock_executor_cls, mock_as_completed
    ):
        client = DataQueryFileAPIClient(
            client_id="id", client_secret="secret", out_dir=self.test_dir
        )
        for n_jobs in (None, 4):
            with self.subTest(n_jobs=n_jobs):
                mock_executor_cls.reset_mock()
                client.download_multiple_files(
                    filenames=["f1.parquet"], n_jobs=n_jobs, show_progress=False
                )
                # ThreadPoolExecutor decides for itself when max_workers is None
                mock_executor_cls.assert_called_once_with(max_workers=n_jobs)

    @suppress_logging
    @patch("macrosynergy.download.dataquery_file_api.cf.as_completed")
    @patch("macrosynergy.download.dataquery_file_api.cf.ThreadPoolExecutor")
    @patch("macrosynergy.download.dataquery_file_api.DataQueryFileAPIOauth")
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

    @patch("macrosynergy.download.dataquery_file_api.cf.as_completed")
    @patch("macrosynergy.download.dataquery_file_api.cf.ThreadPoolExecutor")
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

    @patch.object(DataQueryFileAPIClient, "download_file")
    @patch.object(DataQueryFileAPIClient, "list_available_files")
    def test_download_catalog_file(self, mock_list_files, mock_download):
        client = DataQueryFileAPIClient(
            client_id="id", client_secret="secret", out_dir=self.test_dir
        )
        mock_list_files.return_value = pd.DataFrame(
            {
                "file-name": ["CATALOG_20230102.parquet"],
                "file-datetime": pd.to_datetime(["2023-01-02"]),
                "last-modified": pd.to_datetime(["2023-01-02"]),
            }
        )
        fake_path_str = os.path.join(
            self.test_dir, "jpmaqs-download", "CATALOG_20230102.parquet"
        )
        mock_download.return_value = fake_path_str

        # The catalog is downloaded as-is; nothing is read/converted/mutated on disk.
        path = client.download_catalog_file(overwrite=True)
        self.assertEqual(path, fake_path_str)
        mock_download.assert_called_once_with(
            filename="CATALOG_20230102.parquet",
            overwrite=True,
            timeout=300.0,
        )

        # error case
        mock_list_files.return_value = pd.DataFrame()
        with self.assertRaises(DownloadError):
            client.download_catalog_file()

    @patch("macrosynergy.download.dataquery_file_api.logger")
    @patch("macrosynergy.download.dataquery_file_api.pd.read_parquet")
    @patch.object(DataQueryFileAPIClient, "download_catalog_file")
    def test_get_datasets_for_indicators(
        self, mock_download_catalog, mock_read_parquet, mock_logger
    ):
        client = DataQueryFileAPIClient(
            client_id="id", client_secret="secret", out_dir=self.test_dir
        )
        theme, expected_dataset = next(iter(JPMAQS_DATASET_THEME_MAPPING.items()))
        mock_download_catalog.return_value = "catalog.parquet"
        mock_read_parquet.return_value = pd.DataFrame(
            {
                "Ticker": ["USD_GROWTH", "JPY_INFL", "EUR_RATE"],
                "Theme": [theme, "Some unknown theme", theme],
            }
        )

        datasets = client.get_datasets_for_indicators(
            tickers=["USD_GROWTH", "JPY_INFL"]
        )
        # Catalog is downloaded as-is; the Dataset column is derived in-memory.
        mock_download_catalog.assert_called_once_with()
        mock_read_parquet.assert_called_once_with("catalog.parquet")
        # an unmapped theme is no file group, so it is left out and warned about
        self.assertEqual(datasets, [expected_dataset])
        self.assertIn("Some unknown theme", str(mock_logger.warning.call_args))

    def test_abbreviate_names_how_many_were_left_out(self):
        from macrosynergy.download.dataquery_file_api import _abbreviate_tickers_list

        self.assertEqual(_abbreviate_tickers_list(["A", "B"]), "['A', 'B']")
        ten = [f"T{i}" for i in range(10)]
        self.assertEqual(
            _abbreviate_tickers_list(ten), str(ten)
        )  # exactly at the limit, untouched
        self.assertIn(
            "(+15 more)", _abbreviate_tickers_list([f"T{i}" for i in range(25)])
        )

    @patch("macrosynergy.download.dataquery_file_api.lazy_load_from_parquets")
    @patch.object(DataQueryFileAPIClient, "download_catalog_file")
    @patch.object(DataQueryFileAPIClient, "download_latest_files")
    @patch.object(DataQueryFileAPIClient, "get_datasets_for_indicators")
    @patch("macrosynergy.download.dataquery_file_api.DataQueryFileAPIOauth")
    def test_suppress_warnings_covers_both_channels_and_never_leaks(
        self, mock_oauth, mock_get_datasets, mock_files, mock_catalog, mock_load
    ):
        client = DataQueryFileAPIClient(
            client_id="id", client_secret="secret", out_dir=self.test_dir
        )
        mock_catalog.return_value = "catalog.parquet"
        # one warning on each channel: `logger` and `warnings`
        mock_get_datasets.side_effect = lambda **kw: (
            dq_file_api.logger.warning("a logger warning"),
            ["JPMAQS_A"],
        )[1]

        def loader(**kwargs):
            warnings.warn("a warnings.warn warning")
            return "df"

        records = []

        class Collect(logging.Handler):
            def emit(self, record):
                records.append(record.getMessage())

        handler = Collect()
        dq_file_api.logger.addHandler(handler)
        self.addCleanup(dq_file_api.logger.removeHandler, handler)
        dq_file_api.logger.setLevel(logging.WARNING)

        for suppress, raises, expected in [
            (False, False, 1),  # both channels emit
            (True, False, 0),  # both silenced
            (True, True, 0),  # both silenced, and restored despite the exception
        ]:
            with self.subTest(suppress=suppress, raises=raises):
                records.clear()
                filters_before = len(warnings.filters)
                mock_load.side_effect = (
                    (lambda **kw: (_ for _ in ()).throw(ValueError("boom")))
                    if raises
                    else loader
                )
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    try:
                        client.download(
                            cids=["USD"], xcats=["INFL"], suppress_warnings=suppress
                        )
                    except ValueError:
                        self.assertTrue(raises)

                self.assertEqual(len(records), expected)
                self.assertEqual(len(caught), expected if not raises else 0)
                # the leak this replaced left a global "ignore everything" filter behind
                self.assertEqual(len(warnings.filters), filters_before)
                self.assertEqual(dq_file_api.logger.level, logging.WARNING)

    @patch("macrosynergy.download.dataquery_file_api.lazy_load_from_parquets")
    @patch.object(DataQueryFileAPIClient, "download_catalog_file")
    @patch.object(DataQueryFileAPIClient, "download_latest_files")
    @patch.object(DataQueryFileAPIClient, "get_datasets_for_indicators")
    @patch("macrosynergy.download.dataquery_file_api.DataQueryFileAPIOauth")
    def test_download_datasets_narrows_the_derived_set(
        self, mock_oauth, mock_get_datasets, mock_snapshot, mock_catalog, mock_load
    ):
        client = DataQueryFileAPIClient(
            client_id="id", client_secret="secret", out_dir=self.test_dir
        )
        mock_get_datasets.return_value = ["JPMAQS_A", "JPMAQS_B"]
        mock_catalog.return_value = "catalog.parquet"

        client.download(cids=["USD"], xcats=["INFL"], datasets=["JPMAQS_B"])
        # both the download and the load are narrowed to the requested dataset
        self.assertEqual(mock_snapshot.call_args.kwargs["file_group_ids"], ["JPMAQS_B"])
        self.assertEqual(mock_load.call_args.kwargs["datasets"], ["JPMAQS_B"])

        # a dataset holding none of the requested indicators is an error, not an
        # empty download
        with self.assertRaises(ValueError) as ctx:
            client.download(cids=["USD"], xcats=["INFL"], datasets=["JPMAQS_Z"])
        self.assertIn("holds none of the requested", str(ctx.exception))

    @patch("macrosynergy.download.dataquery_file_api.logger")
    @patch("macrosynergy.download.dataquery_file_api.pd.read_parquet")
    @patch.object(DataQueryFileAPIClient, "download_catalog_file")
    def test_get_datasets_for_indicators_names_tickers_outside_the_catalog(
        self, mock_download_catalog, mock_read_parquet, mock_logger
    ):
        client = DataQueryFileAPIClient(
            client_id="id", client_secret="secret", out_dir=self.test_dir
        )
        theme, dataset = next(iter(JPMAQS_DATASET_THEME_MAPPING.items()))
        mock_download_catalog.return_value = "catalog.parquet"
        mock_read_parquet.return_value = pd.DataFrame(
            {"Ticker": ["USD_GROWTH"], "Theme": [theme]}
        )

        # some in the universe, some not: warned about, the rest still resolves
        datasets = client.get_datasets_for_indicators(
            tickers=["USD_GROWTH", "ZZZ_MYSTERY"]
        )
        self.assertEqual(datasets, [dataset])
        self.assertIn("ZZZ_MYSTERY", str(mock_logger.warning.call_args))

        # none in the universe: say so up front instead of failing later on the load
        with self.assertRaises(ValueError) as ctx:
            client.get_datasets_for_indicators(tickers=["ZZZ_MYSTERY"])
        self.assertIn("None of the requested tickers", str(ctx.exception))
        self.assertIn("ZZZ_MYSTERY", str(ctx.exception))

    @patch("macrosynergy.download.dataquery_file_api.pd.read_parquet")
    @patch.object(DataQueryFileAPIClient, "download_catalog_file")
    def test_get_datasets_for_indicators_as_dict(
        self, mock_download_catalog, mock_read_parquet
    ):
        client = DataQueryFileAPIClient(
            client_id="id", client_secret="secret", out_dir=self.test_dir
        )
        themes = list(JPMAQS_DATASET_THEME_MAPPING.items())
        (theme_a, dataset_a), (theme_b, dataset_b) = themes[0], themes[1]
        mock_download_catalog.return_value = "catalog.parquet"
        mock_read_parquet.return_value = pd.DataFrame(
            {
                "Ticker": ["USD_GROWTH", "EUR_GROWTH", "JPY_XR"],
                "Theme": [theme_a, theme_a, theme_b],
            }
        )

        result = client.get_datasets_for_indicators(
            tickers=["USD_GROWTH", "EUR_GROWTH", "JPY_XR"], as_dict=True
        )
        self.assertEqual(
            result,
            {dataset_a: ["EUR_GROWTH", "USD_GROWTH"], dataset_b: ["JPY_XR"]},
        )

    @patch("macrosynergy.download.dataquery_file_api.pd.read_parquet")
    @patch.object(DataQueryFileAPIClient, "download_catalog_file")
    def test_get_datasets_for_indicators_as_dict_is_case_insensitive(
        self, mock_download_catalog, mock_read_parquet
    ):
        client = DataQueryFileAPIClient(
            client_id="id", client_secret="secret", out_dir=self.test_dir
        )
        theme, dataset = next(iter(JPMAQS_DATASET_THEME_MAPPING.items()))
        mock_download_catalog.return_value = "catalog.parquet"
        mock_read_parquet.return_value = pd.DataFrame(
            {"Ticker": ["USD_GROWTH"], "Theme": [theme]}
        )

        result = client.get_datasets_for_indicators(
            tickers=["usd_growth"], as_dict=True
        )
        # matching is case-insensitive, but the catalog's casing is returned
        self.assertEqual(result, {dataset: ["USD_GROWTH"]})

    @patch("macrosynergy.download.dataquery_file_api.logger")
    @patch.object(DataQueryFileAPIClient, "download_multiple_files")
    @patch.object(DataQueryFileAPIClient, "filter_available_files_by_datetime")
    def test_download_files(self, mock_filter_files, mock_download_multi, mock_logger):
        class_dir = os.path.join(self.test_dir, "class", "dir")
        client = DataQueryFileAPIClient(
            client_id="id", client_secret="secret", out_dir=class_dir
        )
        mock_filter_files.return_value = pd.DataFrame(
            {
                "file-name": [
                    "C_DELTA_20250201T110456.parquet",
                    "A_METADATA_20250201T110000.parquet",
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
        client.download_files(since_datetime="20250201", show_progress=False)

        expected_order = [
            "A_full_20250101.parquet",
            "A_full_20250201.parquet",
            "B_full_20250201.parquet",
            "C_DELTA_20250201T110456.parquet",
            "A_METADATA_20250201T110000.parquet",
        ]

        mock_download_multi.assert_called_once_with(
            filenames=expected_order,
            overwrite=False,
            chunk_size=None,
            timeout=300.0,
            show_progress=False,
        )

    @patch.object(DataQueryFileAPIClient, "download_multiple_files")
    @patch.object(DataQueryFileAPIClient, "filter_available_files_by_datetime")
    def test_download_files_filter_args(self, mock_filter_files, mock_download_multi):
        client = DataQueryFileAPIClient(
            client_id="id", client_secret="secret", out_dir=self.test_dir
        )
        mock_filter_files.return_value = pd.DataFrame(
            {
                "file-name": ["f1.parquet"],
                "file-datetime": ["20230101T000000"],
            }
        )
        client.download_files(since_datetime="20230101", show_progress=False)

        mock_filter_files.assert_called_once_with(
            since_datetime="20230101",
            to_datetime=None,
            include_full_snapshots=True,
            include_delta=True,
            include_metadata=True,
        )
        self.assertEqual(mock_download_multi.call_args[1]["filenames"], ["f1.parquet"])

    @patch("macrosynergy.download.dataquery_file_api.logger")
    @patch.object(DataQueryFileAPIClient, "download_multiple_files")
    @patch.object(DataQueryFileAPIClient, "filter_available_files_by_datetime")
    def test_download_files_no_new_files(
        self, mock_filter_files, mock_download_multi, mock_logger
    ):
        client = DataQueryFileAPIClient(
            client_id="id", client_secret="secret", out_dir=self.test_dir
        )
        mock_filter_files.return_value = pd.DataFrame(columns=["file-name"])
        client.download_files(since_datetime="20230102T000000", show_progress=False)
        mock_download_multi.assert_not_called()
        mock_logger.info.assert_any_call("No new files to download.")

    @patch.object(DataQueryFileAPIClient, "download_multiple_files")
    @patch.object(DataQueryFileAPIClient, "filter_available_files_by_datetime")
    def test_download_files_file_group_ids(
        self, mock_filter_files, mock_download_multi
    ):
        client = DataQueryFileAPIClient(
            client_id="id", client_secret="secret", out_dir=self.test_dir
        )
        mock_filter_files.return_value = pd.DataFrame(
            {
                "file-group-id": ["FG1", "FG2", "FG1"],
                "file-name": ["f1", "f2", "f3"],
                "file-datetime": ["20230101", "20230101", "20230101"],
            }
        )
        client.download_files(
            since_datetime="20230101",
            file_group_ids=["FG1"],
            show_progress=False,
        )
        called_args = mock_download_multi.call_args[1]
        self.assertCountEqual(called_args["filenames"], ["f1", "f3"])

        with self.assertRaises(ValueError):
            client.download_files(
                since_datetime="20230101", file_group_ids="not-a-list"
            )


class TestDownloadedFilesSchema(unittest.TestCase):
    """
    `list_downloaded_files` must report one schema whether or not anything is on disk -
    callers index it by name, and `_load_metadata_jsons` does so without an .empty guard.
    """

    COLS = [
        "file-name",
        "file-datetime",
        "dataset",
        "file-type",
        "file-timestamp",
        "path",
    ]

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)

    def test_schema_is_stable_when_the_save_dir_is_absent(self):
        client = _make_client(self.temp_dir.name)
        self.assertEqual(list(client.list_downloaded_files().columns), self.COLS)

    def test_schema_is_stable_when_the_save_dir_is_empty(self):
        client = _make_client(self.temp_dir.name)
        Path(client._get_save_dir()).mkdir(parents=True, exist_ok=True)
        self.assertEqual(list(client.list_downloaded_files().columns), self.COLS)

    def test_schema_is_stable_when_files_are_present(self):
        client = _make_client(self.temp_dir.name)
        save_dir = Path(client._get_save_dir())
        save_dir.mkdir(parents=True, exist_ok=True)
        (save_dir / "JPMAQS_MACROECONOMIC_TRENDS_20240102.parquet").write_bytes(b"x")
        df = client.list_downloaded_files()
        self.assertEqual(list(df.columns), self.COLS)
        self.assertEqual(df.iloc[0]["file-type"], "parquet")
        self.assertEqual(df.iloc[0]["dataset"], "JPMAQS_MACROECONOMIC_TRENDS")

    @suppress_logging
    def test_notification_loaders_are_graceful_on_an_empty_output_dir(self):
        client = _make_client(self.temp_dir.name)
        # no downloaded files at all: report nothing found rather than KeyError
        self.assertEqual(
            client._load_metadata_jsons(date="2024-01-02", skip_download=True), {}
        )
        Path(client._get_save_dir()).mkdir(parents=True, exist_ok=True)
        self.assertEqual(
            client._load_metadata_jsons(date="2024-01-02", skip_download=True), {}
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
    @patch.object(DataQueryFileAPIClient, "download_files")
    def test_load_metadata_jsons_filters_normalizes_and_canonicalizes_titles(
        self, mock_download_files, mock_list_downloaded_files
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
                    "file-timestamp": date + pd.Timedelta(6, unit="h"),
                    "path": str(p_missing),
                },
                {
                    "file-name": p_changed.name,
                    "dataset": "JPMAQS_METADATA_NOTIFICATIONS",
                    "file-timestamp": date
                    + pd.Timedelta(6, unit="h")
                    + pd.Timedelta(1, unit="m"),
                    "path": str(p_changed),
                },
                {
                    "file-name": p_addl.name,
                    "dataset": "JPMAQS_METADATA_NOTIFICATIONS",
                    "file-timestamp": date
                    + pd.Timedelta(7, unit="h")
                    + pd.Timedelta(24, unit="m"),
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

        mock_download_files.assert_not_called()

    @patch("macrosynergy.download.dataquery_file_api.logger")
    @patch("macrosynergy.download.dataquery_file_api.utc_now")
    @patch.object(DataQueryFileAPIClient, "list_downloaded_files")
    @patch.object(DataQueryFileAPIClient, "download_files")
    def test_load_metadata_jsons_future_date_warns_and_downloads(
        self,
        mock_download_files,
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
        mock_download_files.assert_not_called()
        mock_list_downloaded_files.assert_not_called()

    @patch("macrosynergy.download.dataquery_file_api.logger")
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

    @patch("macrosynergy.download.dataquery_file_api.logger")
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

    @patch("requests.head")
    def test_each_client_instance_probes_independently(self, mock_head):
        """Each new DataQueryFileAPIClient instance probes the URL fresh."""

        def _side_effect(url, **kwargs):
            if url == DQ_FILE_API_BASE_URL:
                raise requests.exceptions.ConnectionError("unreachable")
            return MagicMock(status_code=200)

        mock_head.side_effect = _side_effect

        with patch.dict(os.environ, {"DQ_CLIENT_ID": "x", "DQ_CLIENT_SECRET": "y"}):
            with self.assertWarns(UserWarning):
                client1 = DataQueryFileAPIClient()

            probe_count_after_first = (
                mock_head.call_count
            )  # 2 (primary fail + fallback ok)

            with self.assertWarns(UserWarning):
                client2 = DataQueryFileAPIClient()

            # Second instance must probe again (no global cache)
            self.assertEqual(mock_head.call_count, probe_count_after_first * 2)

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

    @patch("macrosynergy.download.dataquery_file_api.logger")
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

    @patch("macrosynergy.download.dataquery_file_api.logger")
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


class TestSnapDateColumn(unittest.TestCase):
    """Edge cases for `_add_snap_date_column` (06:00 UTC snapshot-day boundary)."""

    def setUp(self):
        self.client = _make_client()

    def _snap(self, filename):
        df = self.client._add_snap_date_column(pd.DataFrame({"file-name": [filename]}))
        return df["snap-date"].iloc[0]

    def test_full_snapshot_date_only(self):
        self.assertEqual(
            self._snap("JPMAQS_GENERIC_RETURNS_20260728.parquet"), "20260728"
        )

    def test_delta_at_0600_belongs_to_same_day(self):
        self.assertEqual(
            self._snap("JPMAQS_X_DELTA_20260728T060000.parquet"), "20260728"
        )

    def test_just_before_0600_belongs_to_previous_day(self):
        self.assertEqual(
            self._snap("JPMAQS_X_DELTA_20260728T055959.parquet"), "20260727"
        )

    def test_midnight_belongs_to_previous_day(self):
        self.assertEqual(
            self._snap("JPMAQS_X_DELTA_20260728T000000.parquet"), "20260727"
        )

    def test_end_of_day_delta_same_day(self):
        self.assertEqual(
            self._snap("JPMAQS_X_DELTA_20260131T235959.parquet"), "20260131"
        )

    def test_month_rollback_before_0600(self):
        self.assertEqual(
            self._snap("JPMAQS_X_DELTA_20260301T030000.parquet"), "20260228"
        )

    def test_year_rollback_before_0600(self):
        self.assertEqual(
            self._snap("JPMAQS_X_DELTA_20260101T030000.parquet"), "20251231"
        )

    def test_notification_json_parses(self):
        self.assertEqual(
            self._snap("JPMAQS_METADATA_NOTIFICATIONS_20260728T060735.json"), "20260728"
        )

    def test_mixed_batch_all_bucketed_together(self):
        df = pd.DataFrame(
            {
                "file-name": [
                    "JPMAQS_A_20260728.parquet",
                    "JPMAQS_A_DELTA_20260728T060000.parquet",
                    "JPMAQS_METADATA_CATALOG_20260728.parquet",
                ]
            }
        )
        out = self.client._add_snap_date_column(df)
        self.assertEqual(list(out["snap-date"]), ["20260728", "20260728", "20260728"])

    def test_empty_frame_ok(self):
        out = self.client._add_snap_date_column(pd.DataFrame({"file-name": []}))
        self.assertEqual(len(out), 0)
        self.assertIn("snap-date", out.columns)

    @suppress_logging
    def test_raises_on_malformed_name(self):
        with self.assertRaisesRegex(ValueError, "Incorrectly named"):
            self.client._add_snap_date_column(
                pd.DataFrame({"file-name": ["JPMAQS_BROKEN.parquet"]})
            )

    @suppress_logging
    def test_raise_names_the_offending_file(self):
        df = pd.DataFrame(
            {
                "file-name": [
                    "JPMAQS_GENERIC_RETURNS_20260728.parquet",
                    "JPMAQS_BROKEN.parquet",
                ]
            }
        )
        with self.assertRaises(ValueError) as cm:
            self.client._add_snap_date_column(df)
        self.assertIn("JPMAQS_BROKEN.parquet", str(cm.exception))


class TestLatestCompleteSnapshotDate(unittest.TestCase):
    """Edge cases for `_latest_complete_snapshot_date` (H1 completeness gate)."""

    def setUp(self):
        self.client = _make_client()
        self.themes = list(JPMAQS_DATASET_THEME_MAPPING.values())

    def _full_rows(self, snap, themes=None):
        themes = self.themes if themes is None else themes
        return [
            {"file-name": f"{t}_{snap}.parquet", "file-group-id": t} for t in themes
        ]

    def _df(self, rows):
        return self.client._add_snap_date_column(pd.DataFrame(rows))

    def test_all_complete_returns_latest(self):
        df = self._df(self._full_rows("20260728") + self._full_rows("20260729"))
        self.assertEqual(self.client._latest_complete_snapshot_date(df), "20260729")

    def test_newer_incomplete_falls_back_to_complete(self):
        rows = self._full_rows("20260728") + self._full_rows(
            "20260729", themes=self.themes[:1]
        )
        self.assertEqual(
            self.client._latest_complete_snapshot_date(self._df(rows)), "20260728"
        )

    def test_none_complete_returns_none(self):
        rows = self._full_rows("20260729", themes=self.themes[:3])
        self.assertIsNone(self.client._latest_complete_snapshot_date(self._df(rows)))

    def test_empty_returns_none(self):
        df = self._df(
            {
                "file-name": pd.Series([], dtype="object"),
                "file-group-id": pd.Series([], dtype="object"),
            }
        )
        self.assertIsNone(self.client._latest_complete_snapshot_date(df))

    def test_extra_full_group_still_complete(self):
        rows = self._full_rows("20260728") + [
            {
                "file-name": "JPMAQS_EXTRA_20260728.parquet",
                "file-group-id": "JPMAQS_EXTRA",
            }
        ]
        self.assertEqual(
            self.client._latest_complete_snapshot_date(self._df(rows)), "20260728"
        )

    def test_deltas_and_metadata_do_not_count_as_full(self):
        rows = [
            {
                "file-name": f"{t}_DELTA_20260728T060000.parquet",
                "file-group-id": f"{t}_DELTA",
            }
            for t in self.themes
        ]
        rows.append(
            {
                "file-name": "JPMAQS_METADATA_CATALOG_20260728.parquet",
                "file-group-id": "JPMAQS_METADATA_CATALOG",
            }
        )
        self.assertIsNone(self.client._latest_complete_snapshot_date(self._df(rows)))


class TestSortFileForDownloadOrder(unittest.TestCase):
    """Edge cases for `_sort_file_for_download_order` (snapshot < delta < metadata)."""

    def setUp(self):
        self.client = _make_client()

    def test_priority_snapshot_delta_metadata(self):
        df = pd.DataFrame(
            {
                "file-name": [
                    "JPMAQS_A_METADATA_20260728.json",
                    "JPMAQS_A_DELTA_20260728T060000.parquet",
                    "JPMAQS_A_20260728.parquet",
                ],
                "file-datetime": ["20260728", "20260728T060000", "20260728"],
            }
        )
        order = self.client._sort_file_for_download_order(df)["file-name"].tolist()
        self.assertEqual(
            order,
            [
                "JPMAQS_A_20260728.parquet",
                "JPMAQS_A_DELTA_20260728T060000.parquet",
                "JPMAQS_A_METADATA_20260728.json",
            ],
        )

    def test_lowercase_tokens_not_treated_as_delta_or_metadata(self):
        df = pd.DataFrame(
            {
                "file-name": ["JPMAQS_x_delta_20260728.parquet"],
                "file-datetime": ["20260728"],
            }
        )
        priority = self.client._sort_file_for_download_order(df)["download-priority"]
        self.assertEqual(priority.iloc[0], 1)

    def test_within_category_sorted_by_datetime_then_name(self):
        df = pd.DataFrame(
            {
                "file-name": [
                    "JPMAQS_B_20260728.parquet",
                    "JPMAQS_A_20260728.parquet",
                    "JPMAQS_A_20260701.parquet",
                ],
                "file-datetime": ["20260728", "20260728", "20260701"],
            }
        )
        order = self.client._sort_file_for_download_order(df)["file-name"].tolist()
        self.assertEqual(
            order,
            [
                "JPMAQS_A_20260701.parquet",
                "JPMAQS_A_20260728.parquet",
                "JPMAQS_B_20260728.parquet",
            ],
        )


class TestCleanupOldFiles(unittest.TestCase):
    """Edge cases for `cleanup_old_files` (retention, dry-run, empty-dir, folders)."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.client = _make_client(self.temp_dir.name)
        self.save = Path(self.client._get_save_dir())

    def _make(self, date_folder, filename):
        p = self.save / date_folder / filename
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")
        return p

    def test_keep_zero_deletes_older_keeps_latest(self):
        old = self._make("2026-07-27", "JPMAQS_GENERIC_RETURNS_20260727.parquet")
        new = self._make("2026-07-28", "JPMAQS_GENERIC_RETURNS_20260728.parquet")
        self.client.cleanup_old_files(keep_n_days_old_files=0, to_datetime="20260728")
        self.assertFalse(old.exists())
        self.assertTrue(new.exists())

    def test_keep_one_keeps_previous_day(self):
        d26 = self._make("2026-07-26", "JPMAQS_GENERIC_RETURNS_20260726.parquet")
        d27 = self._make("2026-07-27", "JPMAQS_GENERIC_RETURNS_20260727.parquet")
        d28 = self._make("2026-07-28", "JPMAQS_GENERIC_RETURNS_20260728.parquet")
        self.client.cleanup_old_files(keep_n_days_old_files=1, to_datetime="20260728")
        self.assertFalse(d26.exists())
        self.assertTrue(d27.exists())
        self.assertTrue(d28.exists())

    def test_snap_date_equal_cutoff_is_kept(self):
        f = self._make("2026-07-28", "JPMAQS_GENERIC_RETURNS_20260728.parquet")
        self.client.cleanup_old_files(keep_n_days_old_files=0, to_datetime="20260728")
        self.assertTrue(f.exists())

    def test_none_and_negative_keep_are_noops(self):
        f = self._make("2026-07-27", "JPMAQS_GENERIC_RETURNS_20260727.parquet")
        self.client.cleanup_old_files(
            keep_n_days_old_files=None, to_datetime="20260728"
        )
        self.assertTrue(f.exists())
        self.client.cleanup_old_files(keep_n_days_old_files=-1, to_datetime="20260728")
        self.assertTrue(f.exists())

    def test_empty_directory_is_noop(self):
        self.client.cleanup_old_files(keep_n_days_old_files=0)  # must not raise

    def test_emptied_date_folder_removed_but_others_kept(self):
        self._make("2026-07-27", "JPMAQS_GENERIC_RETURNS_20260727.parquet")
        self._make("2026-07-28", "JPMAQS_GENERIC_RETURNS_20260728.parquet")
        self.client.cleanup_old_files(keep_n_days_old_files=0, to_datetime="20260728")
        self.assertFalse((self.save / "2026-07-27").exists())
        self.assertTrue((self.save / "2026-07-28").exists())

    def test_dry_run_returns_candidates_and_deletes_nothing(self):
        old = self._make("2026-07-27", "JPMAQS_GENERIC_RETURNS_20260727.parquet")
        self._make("2026-07-28", "JPMAQS_GENERIC_RETURNS_20260728.parquet")
        result = self.client.cleanup_old_files(
            keep_n_days_old_files=0, to_datetime="20260728", dry_run=True
        )
        self.assertEqual(
            list(result["file-name"]), ["JPMAQS_GENERIC_RETURNS_20260727.parquet"]
        )
        self.assertTrue(old.exists())

    @patch(
        "macrosynergy.download.dataquery_file_api.utc_now",
        return_value=pd.Timestamp("2026-07-28T02:00:00Z"),
    )
    def test_bare_call_reports_instead_of_deleting(self, _now):
        # with no anchor the wall clock would destroy a snapshot that has simply not been
        # replaced yet, so the call warns and deletes nothing
        old = self._make("2026-07-27", "JPMAQS_GENERIC_RETURNS_20260727.parquet")
        with self.assertWarns(UserWarning) as cm:
            would_delete = self.client.cleanup_old_files(keep_n_days_old_files=0)
        self.assertTrue(old.exists())
        self.assertIn("nothing was deleted", str(cm.warning))
        self.assertIn("JPMAQS_GENERIC_RETURNS_20260727.parquet", str(cm.warning))
        self.assertEqual(
            list(would_delete["file-name"]), ["JPMAQS_GENERIC_RETURNS_20260727.parquet"]
        )


class TestDownloadMultipleFilesReturn(unittest.TestCase):
    """The return/retry contract of `download_multiple_files`."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.client = _make_client(self.temp_dir.name)

    def test_empty_input_returns_empty_list(self):
        self.assertEqual(
            self.client.download_multiple_files(filenames=[], show_progress=False), []
        )

    @suppress_logging
    @patch("macrosynergy.download.dataquery_file_api.time.sleep")
    def test_all_success_returns_sorted_completed(self, _sleep):
        with patch.object(
            self.client,
            "download_file",
            side_effect=lambda filename, **kw: filename,
        ):
            with patch.object(self.client, "delete_corrupt_files", return_value=[]):
                res = self.client.download_multiple_files(
                    filenames=[
                        "JPMAQS_B_20260728.parquet",
                        "JPMAQS_A_20260728.parquet",
                    ],
                    show_progress=False,
                )
        self.assertEqual(
            res, ["JPMAQS_A_20260728.parquet", "JPMAQS_B_20260728.parquet"]
        )

    @suppress_logging
    @patch("macrosynergy.download.dataquery_file_api.time.sleep")
    def test_retry_accumulates_first_pass_and_retry_successes(self, _sleep):
        state = {"failed_once": False}

        def flaky(filename, **kw):
            if filename == "JPMAQS_B_20260728.parquet" and not state["failed_once"]:
                state["failed_once"] = True
                raise Exception("transient")
            return filename

        with patch.object(self.client, "download_file", side_effect=flaky):
            with patch.object(self.client, "delete_corrupt_files", return_value=[]):
                res = self.client.download_multiple_files(
                    filenames=[
                        "JPMAQS_A_20260728.parquet",
                        "JPMAQS_B_20260728.parquet",
                    ],
                    max_retries=2,
                    show_progress=False,
                )
        self.assertEqual(
            res, ["JPMAQS_A_20260728.parquet", "JPMAQS_B_20260728.parquet"]
        )

    @suppress_logging
    @patch("macrosynergy.download.dataquery_file_api.time.sleep")
    def test_corrupt_first_pass_recovered_on_retry(self, _sleep):
        passes = {"n": 0}

        def corrupt(files):
            passes["n"] += 1
            return ["JPMAQS_B_20260728.parquet"] if passes["n"] == 1 else []

        with patch.object(
            self.client,
            "download_file",
            side_effect=lambda filename, **kw: filename,
        ):
            with patch.object(self.client, "delete_corrupt_files", side_effect=corrupt):
                res = self.client.download_multiple_files(
                    filenames=[
                        "JPMAQS_A_20260728.parquet",
                        "JPMAQS_B_20260728.parquet",
                    ],
                    max_retries=1,
                    show_progress=False,
                )
        self.assertEqual(
            res, ["JPMAQS_A_20260728.parquet", "JPMAQS_B_20260728.parquet"]
        )

    @suppress_logging
    @patch("macrosynergy.download.dataquery_file_api.time.sleep")
    def test_persistent_failure_raises(self, _sleep):
        with patch.object(self.client, "download_file", side_effect=Exception("boom")):
            with patch.object(self.client, "delete_corrupt_files", return_value=[]):
                for max_retries in (0, -1):
                    with self.subTest(max_retries=max_retries):
                        with self.assertRaises(DownloadError):
                            self.client.download_multiple_files(
                                filenames=["JPMAQS_A_20260728.parquet"],
                                max_retries=max_retries,
                                show_progress=False,
                            )

    @suppress_logging
    @patch("macrosynergy.download.dataquery_file_api.time.sleep")
    def test_overwrite_is_forwarded_to_the_retry(self, _sleep):
        seen = []

        def flaky(filename, overwrite=False, **kw):
            seen.append(overwrite)
            if len(seen) == 1:
                raise Exception("transient")
            return filename

        with patch.object(self.client, "download_file", side_effect=flaky):
            with patch.object(self.client, "delete_corrupt_files", return_value=[]):
                self.client.download_multiple_files(
                    filenames=["JPMAQS_A_20260728.parquet"],
                    overwrite=True,
                    max_retries=1,
                    show_progress=False,
                )
        self.assertEqual(seen, [True, True])


class TestClientDeleteCorruptFiles(unittest.TestCase):
    """`delete_corrupt_files` accepts file names as well as file paths."""

    NAME = "JPMAQS_MACROECONOMIC_TRENDS_20240102.parquet"

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.client = _make_client(self.temp_dir.name)
        self.save_dir = Path(self.client._get_save_dir())

    def _write_corrupt(self, sub_dir, name=NAME):
        path = self.save_dir / sub_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"not a parquet file")
        return path.resolve()

    @suppress_logging
    def test_a_path_only_touches_that_file(self):
        target = self._write_corrupt("2024-01-02")
        other = self._write_corrupt("2024-01-03")

        self.assertEqual(
            self.client.delete_corrupt_files(files=[str(target)]), [str(target)]
        )
        self.assertFalse(target.exists())
        self.assertTrue(other.exists())

    @suppress_logging
    def test_a_name_touches_every_copy_of_that_name(self):
        first = self._write_corrupt("2024-01-02")
        second = self._write_corrupt("moved")

        self.assertEqual(
            self.client.delete_corrupt_files(files=[self.NAME]),
            sorted([str(first), str(second)]),
        )
        self.assertFalse(first.exists())
        self.assertFalse(second.exists())


class TestDownloadLatestSnapshot(unittest.TestCase):
    """Orchestration edge cases for `download_latest_files`."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.client = _make_client(self.temp_dir.name)
        self.themes = list(JPMAQS_DATASET_THEME_MAPPING.values())

    def _available(self, extra_rows=None):
        rows = [
            {
                "file-name": f"{t}_20260728.parquet",
                "file-group-id": t,
                "file-datetime": "20260728",
            }
            for t in self.themes
        ]
        rows.append(
            {
                "file-name": "JPMAQS_GENERIC_RETURNS_DELTA_20260728T060000.parquet",
                "file-group-id": "JPMAQS_GENERIC_RETURNS_DELTA",
                "file-datetime": "20260728T060000",
            }
        )
        rows.append(
            {
                "file-name": "JPMAQS_METADATA_CATALOG_20260728.parquet",
                "file-group-id": "JPMAQS_METADATA_CATALOG",
                "file-datetime": "20260728",
            }
        )
        if extra_rows:
            rows.extend(extra_rows)
        return pd.DataFrame(rows)

    def _run(self, available, downloaded, overwrite=False, file_group_ids=None):
        empty = pd.DataFrame(columns=["file-name"])
        # using , to seperate patch.object() does not work in python3.7
        with patch.object(
            self.client,
            "filter_available_files_by_datetime",
            return_value=available,
        ):
            with patch.object(
                self.client,
                "list_downloaded_files",
                return_value=downloaded if downloaded is not None else empty,
            ):
                with patch.object(
                    self.client,
                    "download_multiple_files",
                    side_effect=lambda filenames, **kw: filenames,
                ) as mdl:
                    with patch.object(self.client, "cleanup_old_files") as mclean:
                        res = self.client.download_latest_files(
                            overwrite=overwrite,
                            file_group_ids=file_group_ids,
                            show_progress=False,
                        )
                        return res, mdl, mclean

    def test_downloads_latest_complete_and_cleans_anchored_to_it(self):
        avail = self._available(
            extra_rows=[
                {
                    "file-name": f"{self.themes[0]}_20260729.parquet",
                    "file-group-id": self.themes[0],
                    "file-datetime": "20260729",
                }
            ]
        )
        res, mdl, mclean = self._run(avail, None)
        downloaded = mdl.call_args[1]["filenames"]
        self.assertEqual(len(downloaded), 9)
        self.assertNotIn(f"{self.themes[0]}_20260729.parquet", downloaded)
        self.assertEqual(mclean.call_args[1]["to_datetime"], "20260728")
        self.assertEqual(res, downloaded)

    @suppress_logging
    def test_no_complete_snapshot_skips_download_and_cleanup(self):
        avail = pd.DataFrame(
            [
                {
                    "file-name": f"{self.themes[0]}_20260729.parquet",
                    "file-group-id": self.themes[0],
                    "file-datetime": "20260729",
                }
            ]
        )
        res, mdl, mclean = self._run(avail, None)
        self.assertEqual(res, [])
        mdl.assert_not_called()
        mclean.assert_not_called()

    def test_dedup_skips_already_downloaded(self):
        already = pd.DataFrame(
            {"file-name": ["JPMAQS_METADATA_CATALOG_20260728.parquet"]}
        )
        _res, mdl, _clean = self._run(self._available(), already)
        downloaded = mdl.call_args[1]["filenames"]
        self.assertNotIn("JPMAQS_METADATA_CATALOG_20260728.parquet", downloaded)
        self.assertEqual(len(downloaded), 8)

    def test_overwrite_includes_already_downloaded(self):
        already = pd.DataFrame(
            {"file-name": ["JPMAQS_METADATA_CATALOG_20260728.parquet"]}
        )
        _res, mdl, _clean = self._run(self._available(), already, overwrite=True)
        downloaded = mdl.call_args[1]["filenames"]
        self.assertIn("JPMAQS_METADATA_CATALOG_20260728.parquet", downloaded)
        self.assertEqual(len(downloaded), 9)

    def test_file_group_ids_narrows_after_completeness_check(self):
        # completeness is judged over ALL themes, then narrowed to the requested subset
        # (this is the path `download()` uses; filtering before the check would find
        # no complete snapshot and download nothing).
        subset = self.themes[:2]
        _res, mdl, mclean = self._run(self._available(), None, file_group_ids=subset)
        downloaded = mdl.call_args[1]["filenames"]
        # only what was asked for: a restricted selection does not pull the catalog
        self.assertEqual(
            sorted(downloaded), sorted(f"{t}_20260728.parquet" for t in subset)
        )
        self.assertEqual(mclean.call_args[1]["to_datetime"], "20260728")

    def test_no_file_group_ids_includes_the_catalog(self):
        _res, mdl, _mclean = self._run(self._available(), None, file_group_ids=None)
        self.assertIn(
            "JPMAQS_METADATA_CATALOG_20260728.parquet", mdl.call_args[1]["filenames"]
        )

    def test_cleanup_protects_every_file_the_snapshot_needs(self):
        res, mdl, mclean = self._run(self._available(), None)
        protected = mclean.call_args[1]["protect_files"]
        # the whole snapshot, not just what this call happened to fetch
        self.assertEqual(sorted(protected), sorted(mdl.call_args[1]["filenames"]))
        self.assertEqual(res, mdl.call_args[1]["filenames"])

    def test_cleanup_runs_when_the_snapshot_is_already_on_disk(self):
        # nothing to download means the snapshot is complete, so old files can still go
        already = self._available()
        res, mdl, mclean = self._run(self._available(), already)
        self.assertEqual(res, [])
        mdl.assert_not_called()
        mclean.assert_called_once()
        self.assertEqual(mclean.call_args[1]["to_datetime"], "20260728")
        # and the files already on disk are protected, even though nothing was fetched
        self.assertEqual(
            sorted(mclean.call_args[1]["protect_files"]),
            sorted(already["file-name"]),
        )

    def test_empty_file_group_ids_downloads_nothing(self):
        # `download()` passes [] when no requested ticker matches any dataset; that must
        # download nothing and skip cleanup (not fall through to the whole snapshot).
        res, mdl, mclean = self._run(self._available(), None, file_group_ids=[])
        self.assertEqual(res, [])
        mdl.assert_not_called()
        mclean.assert_not_called()

    def test_unmatched_file_group_ids_downloads_nothing(self):
        res, mdl, mclean = self._run(
            self._available(), None, file_group_ids=["JPMAQS_DOES_NOT_EXIST"]
        )
        self.assertEqual(res, [])
        mdl.assert_not_called()
        mclean.assert_not_called()


class TestDownloadLatestSnapshotPrune(unittest.TestCase):
    """
    `download_latest_files` end to end against a real directory, with only the network
    calls stubbed. The prune, the file listing and the snap-date maths all run for real.
    """

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.client = _make_client(self.temp_dir.name)
        self.save_dir = Path(self.client._get_save_dir())
        self.themes = list(JPMAQS_DATASET_THEME_MAPPING.values())

    def snapshot_names(self, date):
        return [f"{theme}_{date}.parquet" for theme in self.themes]

    def put_on_disk(self, date):
        """Write a complete snapshot for `date`, as `download_file` lays it out."""
        folder = self.save_dir / f"{date[:4]}-{date[4:6]}-{date[6:]}"
        folder.mkdir(parents=True, exist_ok=True)
        for name in self.snapshot_names(date):
            (folder / name).write_bytes(b"x")

    def available(self, dates):
        return pd.DataFrame(
            [
                {"file-name": name, "file-group-id": theme, "file-datetime": date}
                for date in dates
                for theme, name in zip(self.themes, self.snapshot_names(date))
            ]
        )

    def on_disk(self):
        return sorted(p.name for p in self.save_dir.rglob("*.parquet"))

    def run_snapshot(self, dates, keep_n_days_old_files=0):
        # the files are already written to disk by the test; downloading is a no-op
        with patch.object(
            self.client,
            "filter_available_files_by_datetime",
            return_value=self.available(dates),
        ):
            with patch.object(
                self.client,
                "download_multiple_files",
                side_effect=lambda filenames, **kw: filenames,
            ) as mock_download:
                result = self.client.download_latest_files(
                    keep_n_days_old_files=keep_n_days_old_files, show_progress=False
                )
                return result, mock_download

    @suppress_logging
    def test_old_snapshot_is_pruned_and_the_latest_survives(self):
        self.put_on_disk("20260727")
        self.put_on_disk("20260728")
        result, mock_download = self.run_snapshot(["20260727", "20260728"])

        self.assertEqual(result, [])  # nothing needed fetching
        mock_download.assert_not_called()
        # the prune still ran, and kept exactly the latest snapshot
        self.assertEqual(self.on_disk(), sorted(self.snapshot_names("20260728")))

    @suppress_logging
    def test_prune_keeps_the_previous_day_when_asked(self):
        self.put_on_disk("20260727")
        self.put_on_disk("20260728")
        self.run_snapshot(["20260727", "20260728"], keep_n_days_old_files=1)
        self.assertEqual(
            self.on_disk(),
            sorted(self.snapshot_names("20260727") + self.snapshot_names("20260728")),
        )

    @suppress_logging
    def test_nothing_is_pruned_when_the_download_fails(self):
        self.put_on_disk("20260727")  # only the old snapshot is present
        with patch.object(
            self.client,
            "filter_available_files_by_datetime",
            return_value=self.available(["20260727", "20260728"]),
        ):
            with patch.object(
                self.client,
                "download_multiple_files",
                side_effect=DownloadError("boom"),
            ):
                with self.assertRaises(DownloadError):
                    self.client.download_latest_files(
                        keep_n_days_old_files=0, show_progress=False
                    )
        # the replacement never arrived, so the old snapshot must still be there
        self.assertEqual(self.on_disk(), sorted(self.snapshot_names("20260727")))

    @suppress_logging
    def test_a_partially_present_latest_snapshot_is_completed_before_pruning(self):
        self.put_on_disk("20260727")
        missing = self.snapshot_names("20260728")[0]
        folder = self.save_dir / "2026-07-28"
        folder.mkdir(parents=True, exist_ok=True)
        for name in self.snapshot_names("20260728")[1:]:
            (folder / name).write_bytes(b"x")

        result, mock_download = self.run_snapshot(["20260727", "20260728"])

        # only the missing file is fetched, and the rest of the snapshot is protected
        self.assertEqual(result, [missing])
        self.assertEqual(mock_download.call_args[1]["filenames"], [missing])
        survivors = set(self.on_disk())
        self.assertEqual(survivors, set(self.snapshot_names("20260728")[1:]))


class TestSnapshotRetentionWindow(unittest.TestCase):
    """
    Retention is counted in published snapshot dates, so weekend and holiday gaps do not
    consume the allowance. Fri 2026-08-14, Sat 15th and Sun 16th are all publish dates
    here; the 15th and 16th are non-business days.
    """

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.client = _make_client(self.temp_dir.name)
        self.save_dir = Path(self.client._get_save_dir())
        self.themes = list(JPMAQS_DATASET_THEME_MAPPING.values())

    def names(self, date):
        return [f"{theme}_{date}.parquet" for theme in self.themes]

    def put_on_disk(self, date):
        folder = self.save_dir / f"{date[:4]}-{date[4:6]}-{date[6:]}"
        folder.mkdir(parents=True, exist_ok=True)
        for name in self.names(date):
            (folder / name).write_bytes(b"x")

    def available(self, dates):
        return pd.DataFrame(
            [
                {"file-name": name, "file-group-id": theme, "file-datetime": date}
                for date in dates
                for theme, name in zip(self.themes, self.names(date))
            ]
        )

    def on_disk_dates(self):
        return sorted(
            {p.name.rsplit("_", 1)[-1][:8] for p in self.save_dir.rglob("*.parquet")}
        )

    def run_snapshot(self, upstream, keep):
        with patch.object(
            self.client,
            "filter_available_files_by_datetime",
            return_value=self.available(upstream),
        ):
            with patch.object(
                self.client,
                "download_multiple_files",
                side_effect=lambda filenames, **kw: filenames,
            ):
                return self.client.download_latest_files(
                    keep_n_days_old_files=keep, show_progress=False
                )

    @suppress_logging
    def test_weekend_snapshot_becomes_the_latest(self):
        # a Sunday publication is the latest, business day or not
        self.put_on_disk("20260814")
        self.put_on_disk("20260816")
        self.run_snapshot(["20260814", "20260816"], keep=0)
        self.assertEqual(self.on_disk_dates(), ["20260816"])

    @suppress_logging
    def test_keep_one_steps_back_a_publication_not_a_day(self):
        # Sat 15th is the publication before Sun 16th, so keep=1 spans 15th-16th and the
        # Friday goes, even though the Friday is only two calendar days back
        for date in ("20260814", "20260815", "20260816"):
            self.put_on_disk(date)
        self.run_snapshot(["20260814", "20260815", "20260816"], keep=1)
        self.assertEqual(self.on_disk_dates(), ["20260815", "20260816"])

    @suppress_logging
    def test_keep_two_spans_the_whole_weekend(self):
        for date in ("20260814", "20260815", "20260816"):
            self.put_on_disk(date)
        self.run_snapshot(["20260814", "20260815", "20260816"], keep=2)
        self.assertEqual(self.on_disk_dates(), ["20260814", "20260815", "20260816"])

    def test_warns_when_the_retained_history_is_not_on_disk(self):
        # the user asked to keep one publication beyond the latest, but never downloaded
        # Saturday's, so the Friday they do hold falls outside the window and is deleted
        self.put_on_disk("20260814")
        self.put_on_disk("20260816")
        logging.disable(logging.CRITICAL)
        try:
            with self.assertWarns(UserWarning) as cm:
                self.run_snapshot(["20260814", "20260815", "20260816"], keep=1)
        finally:
            logging.disable(logging.NOTSET)
        message = str(cm.warning)
        self.assertIn("20260815", message)  # the date that is missing locally
        self.assertIn("20260816", message)  # what is actually held
        self.assertEqual(self.on_disk_dates(), ["20260816"])


class TestDownloadFilesReturn(unittest.TestCase):
    """`download_files` returns the downloaded file list."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.client = _make_client(self.temp_dir.name)

    @suppress_logging
    def test_returns_downloaded_list(self):
        avail = pd.DataFrame(
            {"file-name": ["JPMAQS_A_20260201.parquet"], "file-datetime": ["20260201"]}
        )
        with patch.object(
            self.client, "filter_available_files_by_datetime", return_value=avail
        ):
            with patch.object(
                self.client,
                "list_downloaded_files",
                return_value=pd.DataFrame(columns=["file-name"]),
            ):
                with patch.object(
                    self.client,
                    "download_multiple_files",
                    return_value=["JPMAQS_A_20260201.parquet"],
                ):
                    result = self.client.download_files(
                        since_datetime="20260201", show_progress=False
                    )
        self.assertEqual(result, ["JPMAQS_A_20260201.parquet"])

    @suppress_logging
    def test_returns_empty_when_no_new_files(self):
        with patch.object(
            self.client,
            "filter_available_files_by_datetime",
            return_value=pd.DataFrame(columns=["file-name"]),
        ):
            with patch.object(
                self.client,
                "list_downloaded_files",
                return_value=pd.DataFrame(columns=["file-name"]),
            ):
                result = self.client.download_files(
                    since_datetime="20260201", show_progress=False
                )
        self.assertEqual(result, [])


class TestDownloadFullSnapshotDeprecated(unittest.TestCase):
    """`download_full_snapshot` is a deprecated shim over `download_files`."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.client = _make_client(self.temp_dir.name)

    def test_emits_futurewarning_and_delegates(self):
        with patch.object(
            self.client, "download_files", return_value=["JPMAQS_A_20260201.parquet"]
        ) as mock_download_files:
            with self.assertWarns(FutureWarning) as cm:
                result = self.client.download_full_snapshot(
                    since_datetime="20260201",
                    include_delta=False,
                    show_progress=False,
                )

        self.assertIn("deprecated", str(cm.warning))
        self.assertIn("download_files", str(cm.warning))
        # delegates verbatim and returns the delegate's result
        mock_download_files.assert_called_once_with(
            since_datetime="20260201",
            to_datetime=None,
            overwrite=False,
            chunk_size=None,
            timeout=300.0,
            include_full_snapshots=True,
            include_delta=False,
            include_metadata=True,
            file_group_ids=None,
            show_progress=False,
        )
        self.assertEqual(result, ["JPMAQS_A_20260201.parquet"])


class TestDownloadForwardsLoadOptions(unittest.TestCase):
    """
    `download` resolves the datasets for the requested tickers, downloads the latest
    snapshot and the catalog, then hands the load options to `lazy_load_from_parquets`.
    """

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.client = _make_client(self.temp_dir.name)
        self.datasets = ["JPMAQS_MACROECONOMIC_TRENDS"]

    def _download(self, **kwargs):
        with patch.object(
            self.client, "get_datasets_for_indicators", return_value=self.datasets
        ):
            with patch.object(
                self.client, "download_latest_files", return_value=[]
            ) as mock_snapshot:
                with patch.object(
                    self.client,
                    "download_catalog_file",
                    return_value=str(Path(self.temp_dir.name) / "catalog.parquet"),
                ):
                    with patch(
                        "macrosynergy.download.dataquery_file_api.lazy_load_from_parquets",
                        return_value="LOADED",
                    ) as mock_load:
                        result = self.client.download(tickers=["USD_INFL"], **kwargs)
        return result, mock_load.call_args.kwargs, mock_snapshot.call_args.kwargs

    def test_include_source_file_is_forwarded(self):
        result, load_kwargs, _ = self._download(include_source_file=True)
        self.assertEqual(result, "LOADED")
        self.assertTrue(load_kwargs["include_source_file"])

    def test_include_source_file_defaults_to_false(self):
        _, load_kwargs, _ = self._download()
        self.assertFalse(load_kwargs["include_source_file"])

    def test_load_options_and_datasets_are_forwarded(self):
        _, load_kwargs, _ = self._download(
            metrics=["value"],
            start_date="2020-01-01",
            end_date="2024-01-01",
            dataframe_type="polars",
            include_delta_files=True,
        )
        self.assertEqual(load_kwargs["metrics"], ["value"])
        self.assertEqual(load_kwargs["start_date"], "2020-01-01")
        self.assertEqual(load_kwargs["end_date"], "2024-01-01")
        self.assertEqual(load_kwargs["dataframe_type"], "polars")
        self.assertTrue(load_kwargs["include_delta_files"])
        self.assertEqual(load_kwargs["datasets"], self.datasets)
        # the catalog path is resolved to a Path and passed through
        self.assertEqual(Path(load_kwargs["catalog_path"]).name, "catalog.parquet")

    def test_include_delta_files_defaults_to_true(self):
        _, load_kwargs, _ = self._download()
        self.assertTrue(load_kwargs["include_delta_files"])

    def test_snapshot_download_is_scoped_to_the_resolved_datasets(self):
        _, _, snapshot_kwargs = self._download(overwrite=True, show_progress=False)
        self.assertEqual(snapshot_kwargs["file_group_ids"], self.datasets)
        self.assertTrue(snapshot_kwargs["overwrite"])
        self.assertFalse(snapshot_kwargs["show_progress"])
        # 0 means "keep only the latest snapshot" - the default prunes older files
        self.assertEqual(snapshot_kwargs["keep_n_days_old_files"], 0)


if __name__ == "__main__":
    unittest.main()
