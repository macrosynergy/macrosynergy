import unittest
import pandas as pd
from unittest.mock import patch, MagicMock, call

from macrosynergy.download.dataquery_file_api import (
    validate_dq_timestamp,
    get_client_id_secret,
    DataQueryFileAPIClient,
    DownloadError,
    DQ_FILE_API_SCOPE,
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
        client = DataQueryFileAPIClient(client_id="arg_id", client_secret="arg_secret")
        self.assertEqual(client.client_id, "arg_id")
        mock_oauth_constructor.assert_called_once_with(
            client_id="arg_id",
            client_secret="arg_secret",
            resource=DQ_FILE_API_SCOPE,
            verify=True,
        )

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
        mock_get.assert_has_calls(
            [call("/groups", {}), call("/groups/search", {"keywords": "search_term"})]
        )

    @patch("pandas.Timestamp.now")
    @patch.object(DataQueryFileAPIClient, "_get")
    def test_list_available_files(self, mock_get, mock_now):
        client = DataQueryFileAPIClient(client_id="id", client_secret="secret")
        mock_now.return_value = pd.Timestamp("2023-01-02")
        mock_get.return_value = {
            "available-files": [
                {"file-datetime": "20230101T100000", "is-available": True}
            ]
        }

        client.list_available_files(file_group_id="test_id")
        self.assertEqual(mock_get.call_args[0][1]["end-date"], "20230102")

    @patch.object(DataQueryFileAPIClient, "list_available_files")
    @patch.object(DataQueryFileAPIClient, "list_group_files")
    def test_list_available_files_for_all_with_conversion(
        self, mock_list_groups, mock_list_available
    ):
        client = DataQueryFileAPIClient(client_id="id", client_secret="secret")
        mock_list_groups.return_value = pd.DataFrame({"file-group-id": ["FG1"]})
        mock_list_available.return_value = pd.DataFrame(
            {"file-datetime": ["20230101T120000"], "last-modified": ["20230101T120000"]}
        )

        df = client.list_available_files_for_all_file_groups()
        self.assertEqual(len(df), 1)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(df["file-datetime"]))

    @patch.object(DataQueryFileAPIClient, "list_available_files")
    @patch.object(DataQueryFileAPIClient, "list_group_files")
    def test_list_available_files_for_all_without_conversion(
        self, mock_list_groups, mock_list_available
    ):
        client = DataQueryFileAPIClient(client_id="id", client_secret="secret")
        mock_list_groups.return_value = pd.DataFrame({"file-group-id": ["FG1"]})
        mock_list_available.return_value = pd.DataFrame(
            {"file-datetime": ["20230101"], "last-modified": ["20230101"]}
        )

        df = client.list_available_files_for_all_file_groups(
            convert_metadata_timestamps=False
        )
        self.assertTrue(pd.api.types.is_object_dtype(df["file-datetime"]))

    @patch.object(DataQueryFileAPIClient, "_get")
    def test_check_file_availability(self, mock_get):
        client = DataQueryFileAPIClient(client_id="id", client_secret="secret")

        client.check_file_availability(file_group_id="FG", file_datetime="20230101")
        mock_get.assert_called_with(
            "/group/file/availability",
            {"file-group-id": "FG", "file-datetime": "20230101"},
        )

        with self.assertRaises(ValueError):
            client.check_file_availability()

        client.check_file_availability(filename="f", file_group_id="fg")

    @patch("macrosynergy.download.dataquery_file_api.DataQueryFileAPIOauth")
    def test_download_parquet_file_bad_filename_raises_error(self, mock_oauth):
        client = DataQueryFileAPIClient(client_id="id", client_secret="secret")
        with self.assertRaisesRegex(
            ValueError, "Invalid filename format: invalid.parquet"
        ):
            client.download_parquet_file(filename="invalid.parquet")

    @patch("macrosynergy.download.dataquery_file_api.cf.as_completed")
    @patch("macrosynergy.download.dataquery_file_api.cf.ThreadPoolExecutor")
    @patch("macrosynergy.download.dataquery_file_api.DataQueryFileAPIOauth")
    def test_download_multiple_parquet_files_retry(
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
            "download_multiple_parquet_files",
            wraps=client.download_multiple_parquet_files,
        ) as mock_check:
            with self.assertRaises(DownloadError):
                client.download_multiple_parquet_files(
                    filenames=["f1.parquet", "f2.parquet"],
                    max_retries=1,
                    show_progress=False,
                )
            self.assertEqual(mock_check.call_count, 2)
            self.assertEqual(
                mock_check.call_args_list[1].kwargs["filenames"], ["f2.parquet"]
            )

    @patch("macrosynergy.download.dataquery_file_api.logger")
    @patch.object(DataQueryFileAPIClient, "download_multiple_parquet_files")
    @patch.object(DataQueryFileAPIClient, "list_available_files_for_all_file_groups")
    def test_download_full_snapshot(
        self, mock_list_all, mock_download_multi, mock_logger
    ):
        client = DataQueryFileAPIClient(client_id="id", client_secret="secret")
        mock_list_all.return_value = pd.DataFrame(
            {
                "is-available": [True, True, True],
                "file-datetime": pd.to_datetime(
                    ["2023-01-02", "2023-01-03", "2023-01-04"]
                ),
                "last-modified": pd.to_datetime(
                    [
                        "2023-01-02T12:00:00Z",
                        "2023-01-03T12:00:00Z",
                        "2023-01-05T12:00:00Z",
                    ]
                ),
                "file-name": ["f2.parquet", "f3.parquet", "f4.parquet"],
            }
        )

        client.download_full_snapshot(
            since_datetime="20230103T000000", show_progress=False
        )

        # filter_date = 2023-01-03, filter_ts = 2023-01-03T00Z
        # f2: file-datetime < filter_date -> FAIL
        # f3: last-modified >= filter_ts, file-datetime >= filter_date -> PASS
        # f4: last-modified >= filter_ts, file-datetime >= filter_date -> PASS
        mock_download_multi.assert_called_once()
        self.assertEqual(
            mock_download_multi.call_args.kwargs["filenames"],
            ["f3.parquet", "f4.parquet"],
        )


if __name__ == "__main__":
    unittest.main()
