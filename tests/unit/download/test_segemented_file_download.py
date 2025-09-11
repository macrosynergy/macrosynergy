import unittest
from unittest.mock import patch, MagicMock, mock_open, call
from pathlib import Path
import logging
import requests
import shutil  # noqa
import uuid  # noqa
import concurrent.futures  # noqa

from macrosynergy.download.dataquery_file_api import SegmentedFileDownloader


class TestSegmentedFileDownloaderInitAndLifecycle(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        logging.disable(logging.CRITICAL)

    @classmethod
    def tearDownClass(cls):
        logging.disable(logging.NOTSET)

    def setUp(self):
        self.filename = "./test_output/group1_20230101.parquet"
        self.url = "some.sort.of/url"
        self.headers = {"Authorization": "Bearer sometoken"}
        self.params = {"file-group-id": "group1", "file-datetime": "20230101"}
        self.base_args = {
            "filename": self.filename,
            "url": self.url,
            "headers": self.headers,
            "params": self.params,
        }

    @patch("pathlib.Path.mkdir")
    def test_init_success(self, mock_mkdir):
        downloader = SegmentedFileDownloader(**self.base_args)

        self.assertEqual(downloader.filename, Path(self.filename))
        self.assertEqual(downloader.url, self.url)
        self.assertEqual(downloader.headers, self.headers)
        self.assertEqual(downloader.params, self.params)
        self.assertEqual(downloader.file_id, "group1_20230101")
        self.assertEqual(downloader.out_dir, Path("./test_output"))
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
        self.assertTrue(
            downloader.temp_dir.name.startswith("_tmp_group1_20230101.parquet_")
        )

    def test_init_missing_params_raises_error(self):
        with self.assertRaises(ValueError):
            SegmentedFileDownloader(
                filename=self.filename, url=self.url, headers={}, params={}
            )

    @patch("macrosynergy.download.dataquery_file_api.SegmentedFileDownloader.download")
    @patch("pathlib.Path.mkdir", MagicMock())
    def test_init_with_start_download(self, mock_download):
        SegmentedFileDownloader(**self.base_args, start_download=True)
        mock_download.assert_called_once()

    @patch("macrosynergy.download.dataquery_file_api.SegmentedFileDownloader.cleanup")
    @patch("macrosynergy.download.dataquery_file_api.SegmentedFileDownloader.download")
    @patch("pathlib.Path.mkdir", MagicMock())
    def test_init_start_download_exception_cleanup(self, mock_download, mock_cleanup):
        mock_download.side_effect = Exception("Download failed")
        with self.assertRaises(Exception):
            SegmentedFileDownloader(**self.base_args, start_download=True)
        mock_download.assert_called_once()
        mock_cleanup.assert_called_once()

    @patch("macrosynergy.download.dataquery_file_api.SegmentedFileDownloader.cleanup")
    @patch("pathlib.Path.mkdir", MagicMock())
    def test_context_manager_lifecycle(self, mock_cleanup):
        with SegmentedFileDownloader(**self.base_args) as downloader:
            self.assertIsInstance(downloader, SegmentedFileDownloader)
        mock_cleanup.assert_called_once()

        mock_cleanup.reset_mock()
        with self.assertRaises(ValueError):
            with SegmentedFileDownloader(**self.base_args):
                raise ValueError("Test error")
        mock_cleanup.assert_called_once()

    @patch("shutil.rmtree")
    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.mkdir", MagicMock())
    def test_cleanup(self, mock_exists, mock_rmtree):
        downloader = SegmentedFileDownloader(**self.base_args)

        # case where temp_dir exists
        mock_exists.return_value = True
        downloader.cleanup()
        mock_rmtree.assert_called_once_with(downloader.temp_dir)

        # case where temp_dir does not exist
        mock_rmtree.reset_mock()
        mock_exists.return_value = False
        downloader.cleanup()
        mock_rmtree.assert_not_called()


@patch("macrosynergy.download.dataquery_file_api._wait_for_api_call", MagicMock())
class TestSegmentedFileDownloaderNetworking(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        logging.disable(logging.CRITICAL)

    @classmethod
    def tearDownClass(cls):
        logging.disable(logging.NOTSET)

    def setUp(self):
        with patch("pathlib.Path.mkdir", MagicMock()):
            self.downloader = SegmentedFileDownloader(
                filename="./output/g1_dt1.parquet",
                url="https://fake.url/data",
                headers={"Auth": "token"},
                params={"file-group-id": "g1", "file-datetime": "dt1"},
            )
        self.downloader.temp_dir = Path("/fake/temp/dir")

    @patch("requests.head")
    def test_get_file_size_success(self, mock_head):
        mock_response = MagicMock(headers={"Content-Length": "50000"})
        mock_head.return_value = mock_response

        size = self.downloader._get_file_size()

        self.assertEqual(size, 50000)
        mock_head.assert_called_once_with(
            self.downloader.url,
            params=self.downloader.params,
            headers=self.downloader.headers,
            proxies=self.downloader.proxies,
            verify=self.downloader.verify_ssl,
        )
        mock_response.raise_for_status.assert_called_once()

    @patch("requests.head")
    def test_get_file_size_missing_header_raises_error(self, mock_head):
        mock_response = MagicMock(headers={})
        mock_head.return_value = mock_response
        with self.assertRaisesRegex(ValueError, "Invalid or missing Content-Length"):
            self.downloader._get_file_size()

    @patch("builtins.open", new_callable=mock_open)
    @patch("requests.get")
    def test_download_chunk_retry_success(self, mock_get, mock_file):
        mock_response = MagicMock()
        mock_response.iter_content.return_value = [b"part1", b"part2"]
        mock_get.return_value.__enter__.return_value = mock_response

        self.downloader._download_chunk_retry(
            part_num=0, start_byte=0, end_byte=100, retries=1
        )

        mock_get.assert_called_once()
        self.assertIn("bytes=0-100", mock_get.call_args.kwargs["headers"]["Range"])
        mock_file.assert_called_once_with(self.downloader.temp_dir / "part_0", "wb")
        mock_file().write.assert_has_calls([call(b"part1"), call(b"part2")])
        mock_response.raise_for_status.assert_called_once()

    @patch("requests.get")
    def test_download_chunk_fails_after_retries(self, mock_get):
        exception = requests.exceptions.Timeout("Timed out")
        mock_get.side_effect = exception

        with self.assertRaises(requests.exceptions.Timeout) as cm:
            self.downloader._download_chunk_retry(
                part_num=0, start_byte=0, end_byte=100, retries=2
            )

        self.assertIs(cm.exception, exception)
        # 3 tries= 1 initial + 2 retries
        self.assertEqual(mock_get.call_count, 3)

    @patch("requests.get")
    def test_download_chunk_4xx_error_no_retry(self, mock_get):
        mock_response = MagicMock(status_code=403)
        exception = requests.exceptions.HTTPError(response=mock_response)
        mock_get.return_value.__enter__.side_effect = exception

        with self.assertRaises(requests.exceptions.HTTPError) as cm:
            self.downloader._download_chunk_retry(
                part_num=0, start_byte=0, end_byte=100, retries=3
            )

        self.assertIs(cm.exception, exception)
        self.assertEqual(mock_get.call_count, 1)


@patch("shutil.rmtree")
@patch("pathlib.Path.mkdir")
class TestSegmentedFileDownloaderOrchestration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        logging.disable(logging.CRITICAL)

    @classmethod
    def tearDownClass(cls):
        logging.disable(logging.NOTSET)

    def setUp(self):
        self.downloader = SegmentedFileDownloader(
            # using a random filename - as the user is allowed to do
            filename="./output/file.dat",
            url="some.sort.of/url",
            headers={"Auth": "token"},
            params={"file-group-id": "g1", "file-datetime": "dt1"},
            max_file_retries=2,
        )

    @patch("pathlib.Path.exists", return_value=True)
    @patch("builtins.open", new_callable=mock_open)
    @patch("shutil.copyfileobj")
    @patch("pathlib.Path.stat")
    def test_assemble_parts(
        self, mock_stat, mock_copy, mock_open, mock_exists, mock_mkdir, mock_rmtree
    ):
        mock_stat.return_value.st_size = 999
        final_path = Path("/final/output.dat")
        num_parts = 3

        self.downloader._assemble_parts(final_path, num_parts)

        expected_open_calls = [call(final_path, "wb")] + [
            call(self.downloader.temp_dir / f"part_{i}", "rb") for i in range(num_parts)
        ]
        mock_open.assert_has_calls(expected_open_calls, any_order=True)
        self.assertEqual(mock_copy.call_count, num_parts)
        mock_rmtree.assert_called_once_with(self.downloader.temp_dir)

    @patch(
        "macrosynergy.download.dataquery_file_api.SegmentedFileDownloader._download_chunk"
    )
    @patch("concurrent.futures.as_completed", return_value=[])
    @patch("concurrent.futures.ThreadPoolExecutor")
    def test_download_chunks_concurrently(
        self,
        mock_executor_cls,
        mock_as_completed,
        mock_download_chunk,
        mock_mkdir,
        mock_rmtree,
    ):
        mock_executor = MagicMock()
        mock_executor_cls.return_value.__enter__.return_value = mock_executor

        chunks = range(0, 250, 100)
        self.downloader._download_chunks_concurrently(chunks, 250)
        self.assertEqual(mock_executor.submit.call_count, 3)

    @patch(
        "macrosynergy.download.dataquery_file_api.SegmentedFileDownloader._assemble_parts"
    )
    @patch(
        "macrosynergy.download.dataquery_file_api.SegmentedFileDownloader._download_chunks_concurrently"
    )
    @patch(
        "macrosynergy.download.dataquery_file_api.SegmentedFileDownloader._get_file_size"
    )
    @patch("pathlib.Path.exists", return_value=False)
    def test_download_main_success(
        self,
        mock_exists,
        mock_get_size,
        mock_download_chunks,
        mock_assemble,
        mock_mkdir,
        mock_rmtree,
    ):
        mock_get_size.return_value = 1000

        with patch.object(Path, "resolve", return_value=Path(self.downloader.filename)):
            self.downloader.download()

        mock_mkdir.assert_called_once_with(exist_ok=True, parents=True)
        mock_rmtree.assert_not_called()
        mock_get_size.assert_called_once()
        mock_download_chunks.assert_called_once()
        mock_assemble.assert_called_once()

    @patch("time.sleep", MagicMock())
    @patch(
        "macrosynergy.download.dataquery_file_api.SegmentedFileDownloader._get_file_size"
    )
    @patch("pathlib.Path.exists", return_value=True)
    def test_download_main_retry_and_succeed(
        self, mock_exists, mock_get_size, mock_mkdir, mock_rmtree
    ):
        mock_get_size.side_effect = [requests.exceptions.ConnectionError, 1024]

        with patch(
            "macrosynergy.download.dataquery_file_api.SegmentedFileDownloader._download_chunks_concurrently"
        ), patch("builtins.open", mock_open()), patch("shutil.copyfileobj"), patch(
            "pathlib.Path.stat"
        ) as mock_stat:
            mock_stat.return_value.st_size = 1024 * 5

            self.downloader.download()

        self.assertEqual(mock_get_size.call_count, 2)
        # Attempt 1 (fail): rmtree at start(1) + rmtree in cleanup(1) = 2
        # Attempt 2 (success): rmtree at start(1) + rmtree in assemble's cleanup(1) = 2
        # Total = 4
        self.assertEqual(mock_rmtree.call_count, 4)

    @patch("time.sleep", MagicMock())
    @patch(
        "macrosynergy.download.dataquery_file_api.SegmentedFileDownloader._get_file_size"
    )
    @patch("pathlib.Path.exists", return_value=True)
    def test_download_main_fails_after_retries(
        self, mock_exists, mock_get_size, mock_mkdir, mock_rmtree
    ):
        mock_get_size.side_effect = requests.exceptions.Timeout("Timed out")

        with self.assertRaises(requests.exceptions.Timeout):
            self.downloader.download()

        # 1 initial + 2 retries
        self.assertEqual(mock_get_size.call_count, 3)
        # rmtree called at start + in cleanup for each of the 3 failed attempts
        self.assertEqual(mock_rmtree.call_count, 6)


if __name__ == "__main__":
    unittest.main()
