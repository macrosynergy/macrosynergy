import unittest
import pandas as pd
from unittest.mock import patch, MagicMock
import functools
import logging
from macrosynergy.compat import PD_2_0_OR_LATER
from macrosynergy.download.dataquery_file_api import (
    validate_dq_timestamp,
    get_client_id_secret,
    DataQueryFileAPIClient,
    DownloadError,
    InvalidResponseError,
    DQ_FILE_API_SCOPE,
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
    @patch("macrosynergy.download.dataquery_file_api.DataQueryFileAPIOauth")
    @patch(
        "macrosynergy.download.dataquery_file_api.get_client_id_secret",
        return_value=(None, None),
    )
    def test_init_no_credentials_raises_error(self, mock_get_client, mock_oauth):
        with self.assertRaisesRegex(
            ValueError, "Client ID and Client Secret must be provided"
        ):
            DataQueryFileAPIClient()

    @patch("macrosynergy.download.dataquery_file_api.DataQueryFileAPIOauth")
    def test_init_with_args(self, mock_oauth_constructor):
        client = DataQueryFileAPIClient(
            client_id="arg_id", client_secret="arg_secret", out_dir="./test/dir"
        )
        self.assertEqual(client.client_id, "arg_id")
        self.assertEqual(client.out_dir, "./test/dir")
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
        self.assertEqual(client.out_dir, "./jpmaqs-download")
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
            with DataQueryFileAPIClient(client_id="id", client_secret="secret"):
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

        client = DataQueryFileAPIClient(client_id="id", client_secret="secret")
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
        client = DataQueryFileAPIClient(client_id="id", client_secret="secret")
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
        client = DataQueryFileAPIClient(client_id="id", client_secret="secret")
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
        client = DataQueryFileAPIClient(client_id="id", client_secret="secret")
        with self.assertRaises(ValueError):
            client.list_group_files(
                include_full_snapshots=False,
                include_delta=False,
                include_metadata=False,
            )

    @patch.object(DataQueryFileAPIClient, "_get")
    def test_list_group_files_cache(self, mock_get):
        client = DataQueryFileAPIClient(client_id="id", client_secret="secret")
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
        client = DataQueryFileAPIClient(client_id="id", client_secret="secret")
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
        client = DataQueryFileAPIClient(client_id="id", client_secret="secret")
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
        client = DataQueryFileAPIClient(client_id="id", client_secret="secret")
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
        client = DataQueryFileAPIClient(client_id="id", client_secret="secret")
        mock_list_groups.return_value = pd.DataFrame({"file-group-id": ["FG1"]})
        mock_list_available.return_value = pd.DataFrame(
            {
                "file-datetime": pd.to_datetime(["20230101T120000"], utc=True),
                "last-modified": pd.to_datetime(["20230101T120000"], utc=True),
            }
        )
        df = client.list_available_files_for_all_file_groups()
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(df["file-datetime"]))

    @patch.object(DataQueryFileAPIClient, "list_available_files")
    @patch.object(DataQueryFileAPIClient, "list_group_files")
    def test_list_available_files_for_all_missing_column_error(
        self, mock_list_groups, mock_list_available
    ):
        client = DataQueryFileAPIClient(client_id="id", client_secret="secret")
        mock_list_groups.return_value = pd.DataFrame({"file-group-id": ["FG1"]})
        mock_list_available.side_effect = InvalidResponseError(
            'Missing "last-modified" in response'
        )
        with self.assertRaisesRegex(InvalidResponseError, 'Missing "last-modified"'):
            # mssing 'last-modified'
            client.list_available_files_for_all_file_groups()

    @patch.object(DataQueryFileAPIClient, "list_available_files_for_all_file_groups")
    def test_filter_available_files_by_datetime(self, mock_list_all_files):
        client = DataQueryFileAPIClient(client_id="id", client_secret="secret")
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
    @patch("pandas.Timestamp.utcnow")
    @patch.object(DataQueryFileAPIClient, "list_available_files_for_all_file_groups")
    def test_filter_available_files_by_datetime_defaults_and_swap(
        self, mock_list_all_files, mock_now, mock_logger
    ):
        client = DataQueryFileAPIClient(client_id="id", client_secret="secret")
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
        client = DataQueryFileAPIClient(client_id="id", client_secret="secret")

        # Valid cases
        client.check_file_availability(file_group_id="FG", file_datetime="20230101")
        mock_get.assert_called_with(
            "/group/file/availability",
            {"file-group-id": "FG", "file-datetime": "20230101"},
        )

        client.check_file_availability(filename="file.parquet")
        mock_get.assert_called_with(
            "/group/file/availability", {"file-group-id": None, "file-datetime": None}
        )

        # Invalid cases
        with self.assertRaises(ValueError):
            client.check_file_availability()

        with self.assertRaises(ValueError):
            client.check_file_availability(
                filename="f.pq", file_group_id="FG", file_datetime="20230101"
            )

    @suppress_logging
    @patch("macrosynergy.download.dataquery_file_api.Path")
    @patch("macrosynergy.download.dataquery_file_api.DataQueryFileAPIOauth")
    def test_download_file_no_overwrite(self, mock_oauth, mock_path):
        client = DataQueryFileAPIClient(client_id="id", client_secret="secret")
        mock_final_path = MagicMock()
        mock_path.return_value.__truediv__.return_value = mock_final_path
        mock_final_path.exists.return_value = True
        result = client.download_file(
            filename="TEST_FULL_20230101.parquet", overwrite=False
        )
        mock_final_path.unlink.assert_not_called()
        self.assertEqual(result, str(mock_final_path))

    @suppress_logging
    @patch("macrosynergy.download.dataquery_file_api.SegmentedFileDownloader")
    @patch("macrosynergy.download.dataquery_file_api.Path")
    @patch("macrosynergy.download.dataquery_file_api.DataQueryFileAPIOauth")
    def test_download_file_overwrite(
        self, mock_oauth, mock_path, mock_segmented_downloader
    ):
        client = DataQueryFileAPIClient(client_id="id", client_secret="secret")
        mock_final_path = MagicMock()
        mock_path.return_value.__truediv__.return_value = mock_final_path
        mock_final_path.exists.return_value = True

        client.download_file(filename="TEST_FULL_20230101.parquet", overwrite=True)
        mock_final_path.unlink.assert_called_once()

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
        client = DataQueryFileAPIClient(client_id="id", client_secret="secret")
        mock_file_path = MagicMock()
        mock_file_path.exists.return_value = False
        mock_path.return_value.__truediv__.return_value = mock_file_path

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
        client = DataQueryFileAPIClient(client_id="id", client_secret="secret")
        with self.assertRaisesRegex(ValueError, "Invalid filename format"):
            client.download_file(filename="invalidformat.parquet")

    @patch(
        "macrosynergy.download.dataquery_file_api.convert_ticker_based_parquet_file_to_qdf"
    )
    @patch(
        "macrosynergy.download.dataquery_file_api.request_wrapper_stream_bytes_to_disk"
    )
    @patch("macrosynergy.download.dataquery_file_api.Path")
    @patch("macrosynergy.download.dataquery_file_api.DataQueryFileAPIOauth")
    def test_download_file_catalog_no_conversion(
        self, mock_oauth, mock_path, mock_downloader, mock_convert
    ):
        client = DataQueryFileAPIClient(client_id="id", client_secret="secret")
        mock_file_path = MagicMock()
        mock_file_path.exists.return_value = False
        mock_file_path.suffix = ".parquet"
        mock_path.return_value.__truediv__.return_value = mock_file_path

        client.download_file(
            filename="JPMAQS_METADATA_CATALOG_20230101.parquet", qdf=True
        )
        mock_downloader.assert_called_once()
        mock_convert.assert_not_called()

    @patch(
        "macrosynergy.download.dataquery_file_api.convert_ticker_based_parquet_file_to_qdf"
    )
    @patch("macrosynergy.download.dataquery_file_api.SegmentedFileDownloader")
    @patch("macrosynergy.download.dataquery_file_api.Path")
    @patch("macrosynergy.download.dataquery_file_api.DataQueryFileAPIOauth")
    def test_download_file_qdf_conversion(
        self, mock_oauth, mock_path, mock_downloader, mock_convert
    ):
        client = DataQueryFileAPIClient(client_id="id", client_secret="secret")
        mock_file_path = MagicMock()
        mock_file_path.exists.return_value = False
        mock_file_path.suffix = ".parquet"
        mock_file_path.__str__.return_value = "mock_dir/TEST_DATA_20230101.parquet"
        mock_path.return_value.__truediv__.return_value = mock_file_path

        client.download_file(
            filename="TEST_DATA_20230101.parquet",
            qdf=True,
            as_csv=True,
            keep_raw_data=True,
        )

        mock_convert.assert_called_once_with(
            filename=str(mock_file_path),
            as_csv=True,
            qdf=True,
            keep_raw_data=True,
        )

    @patch("macrosynergy.download.dataquery_file_api.cf.as_completed")
    @patch("macrosynergy.download.dataquery_file_api.cf.ThreadPoolExecutor")
    @patch("macrosynergy.download.dataquery_file_api.DataQueryFileAPIOauth")
    def test_download_multiple_files_success(
        self, mock_oauth, mock_executor_cls, mock_as_completed
    ):
        client = DataQueryFileAPIClient(client_id="id", client_secret="secret")
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
    @patch("macrosynergy.download.dataquery_file_api.cf.as_completed")
    @patch("macrosynergy.download.dataquery_file_api.cf.ThreadPoolExecutor")
    @patch("macrosynergy.download.dataquery_file_api.DataQueryFileAPIOauth")
    def test_download_multiple_files_retry(
        self, mock_oauth, mock_executor_cls, mock_as_completed
    ):
        client = DataQueryFileAPIClient(client_id="id", client_secret="secret")
        future_success, future_fail = MagicMock(), MagicMock()
        future_fail.result.side_effect = Exception("Download failed!")
        mock_executor = mock_executor_cls.return_value.__enter__.return_value
        mock_executor.submit.side_effect = (
            lambda fn, *args, **kwargs: future_success
            if kwargs.get("filename") == "f1.parquet"
            else future_fail
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
        client = DataQueryFileAPIClient(client_id="id", client_secret="secret")
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
        client = DataQueryFileAPIClient(client_id="id", client_secret="secret")
        mock_list_files.return_value = pd.DataFrame(
            {
                "file-name": ["CATALOG_20230102.parquet", "CATALOG_20230101.parquet"],
                "file-datetime": pd.to_datetime(["2023-01-02", "2023-01-01"]),
                "last-modified": pd.to_datetime(["2023-01-02", "2023-01-01"]),
            }
        )

        client.download_catalog_file(out_dir="./cat", overwrite=True)
        mock_download.assert_called_once_with(
            filename="CATALOG_20230102.parquet",
            out_dir="./cat",
            overwrite=True,
            timeout=300.0,
        )

        mock_list_files.return_value = pd.DataFrame()
        with self.assertRaises(DownloadError):
            client.download_catalog_file()

    @patch("macrosynergy.download.dataquery_file_api.logger")
    @patch.object(DataQueryFileAPIClient, "download_multiple_files")
    @patch.object(DataQueryFileAPIClient, "filter_available_files_by_datetime")
    def test_download_full_snapshot(
        self, mock_filter_files, mock_download_multi, mock_logger
    ):
        client = DataQueryFileAPIClient(
            client_id="id", client_secret="secret", out_dir="./class/dir"
        )
        mock_filter_files.return_value = pd.DataFrame(
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
            "A_full_20250101.parquet",
            "A_full_20250201.parquet",
            "B_full_20250201.parquet",
            "C_delta_20250201T110456.parquet",
            "A_metadata_20250201T110000.parquet",
        ]

        mock_download_multi.assert_called_once_with(
            filenames=expected_order,
            out_dir="./class/dir",
            overwrite=False,
            qdf=True,
            as_csv=False,
            keep_raw_data=False,
            chunk_size=None,
            timeout=300.0,
            show_progress=False,
        )

        mock_download_multi.reset_mock()
        client.download_full_snapshot(
            since_datetime="20250201", show_progress=False, out_dir="./method/dir", qdf=False, as_csv=True
        )
        mock_download_multi.assert_called_once_with(
            filenames=expected_order,
            out_dir="./method/dir",
            overwrite=False,
            qdf=False,
            as_csv=True,
            keep_raw_data=False,
            chunk_size=None,
            timeout=300.0,
            show_progress=False,
        )

    @patch.object(DataQueryFileAPIClient, "download_multiple_files")
    @patch.object(DataQueryFileAPIClient, "filter_available_files_by_datetime")
    def test_download_full_snapshot_with_file_datetime(
        self, mock_filter_files, mock_download_multi
    ):
        client = DataQueryFileAPIClient(client_id="id", client_secret="secret")
        mock_filter_files.return_value = pd.DataFrame(
            {
                "file-name": ["f1.parquet"],
                "file-datetime": ["20230101T000000"],
            }
        )
        client.download_full_snapshot(since_datetime="20230101", show_progress=False)

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
    def test_download_full_snapshot_no_new_files(
        self, mock_filter_files, mock_download_multi, mock_logger
    ):
        client = DataQueryFileAPIClient(client_id="id", client_secret="secret")
        mock_filter_files.return_value = pd.DataFrame(columns=["file-name"])
        client.download_full_snapshot(
            since_datetime="20230102T000000", show_progress=False
        )
        mock_download_multi.assert_not_called()
        mock_logger.info.assert_any_call("No new files to download.")

    @patch.object(DataQueryFileAPIClient, "download_multiple_files")
    @patch.object(DataQueryFileAPIClient, "filter_available_files_by_datetime")
    def test_download_full_snapshot_file_group_ids(
        self, mock_filter_files, mock_download_multi
    ):
        client = DataQueryFileAPIClient(client_id="id", client_secret="secret")
        mock_filter_files.return_value = pd.DataFrame(
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

    @patch.object(DataQueryFileAPIClient, "_get")
    def test_cache_decorator_caching_and_buster(self, mock_get):
        DataQueryFileAPIClient._list_available_files.cache_clear()
        call_counter = {"count": 0}

        def fake_get(endpoint, params):
            call_counter["count"] += 1
            return {
                "available-files": [
                    {
                        "file-datetime": f"20230101T10000{call_counter['count']}",
                        "last-modified": f"20230101T10000{call_counter['count']}",
                        "is-available": True,
                    }
                ]
            }

        mock_get.side_effect = fake_get
        client = DataQueryFileAPIClient(client_id="id", client_secret="secret")
        file_group_id = f"test_id_{id(self)}"
        df1 = client.list_available_files(file_group_id=file_group_id, no_cache=False)
        self.assertEqual(call_counter["count"], 1)
        df2 = client.list_available_files(file_group_id=file_group_id, no_cache=False)
        self.assertEqual(call_counter["count"], 1)
        self.assertEqual(df1.to_dict(), df2.to_dict())

        df3 = client.list_available_files(file_group_id=file_group_id, no_cache=True)
        self.assertEqual(call_counter["count"], 2)
        self.assertNotEqual(df1.to_dict(), df3.to_dict())

        DataQueryFileAPIClient._list_available_files.cache_clear()
        df4 = client.list_available_files(file_group_id=file_group_id, no_cache=False)
        self.assertEqual(call_counter["count"], 3)
        self.assertEqual(df4.to_dict(), df4.to_dict())

    @patch.object(DataQueryFileAPIClient, "_get")
    def test_cache_decorator_ttl_expiry(self, mock_get):
        DataQueryFileAPIClient._list_available_files.cache_clear()
        call_counter = {"count": 0}

        def fake_get(endpoint, params):
            call_counter["count"] += 1
            return {
                "available-files": [
                    {
                        "file-datetime": f"20230101T10000{call_counter['count']}",
                        "last-modified": f"20230101T10000{call_counter['count']}",
                        "is-available": True,
                    }
                ]
            }

        mock_get.side_effect = fake_get
        client = DataQueryFileAPIClient(client_id="id", client_secret="secret")
        file_group_id = f"test_id_{id(self)}_ttl"
        import macrosynergy.download.fusion_interface as fusion_interface

        orig_time = fusion_interface.time.time
        try:
            fusion_interface.time.time = lambda: 0
            df1 = client.list_available_files(
                file_group_id=file_group_id, no_cache=False
            )
            self.assertEqual(call_counter["count"], 1)

            fusion_interface.time.time = lambda: 0
            df2 = client.list_available_files(
                file_group_id=file_group_id, no_cache=False
            )
            self.assertEqual(call_counter["count"], 1)
            self.assertEqual(df1.to_dict(), df2.to_dict())

            fusion_interface.time.time = lambda: 61

            DataQueryFileAPIClient._list_available_files.cache_clear()
            df3 = client.list_available_files(
                file_group_id=file_group_id, no_cache=False
            )
            self.assertEqual(call_counter["count"], 2)
            self.assertNotEqual(df1.to_dict(), df3.to_dict())
        finally:
            fusion_interface.time.time = orig_time


if __name__ == "__main__":
    unittest.main()
