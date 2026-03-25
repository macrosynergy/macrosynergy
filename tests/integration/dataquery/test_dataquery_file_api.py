import unittest
import os
import tempfile
import warnings

from macrosynergy.download.dataquery_file_api import (
    DataQueryFileAPIClient,
    DQ_FILE_API_FALLBACK_BASE_URL,
)


class TestDataQueryFileAPIClient(unittest.TestCase):
    def setUp(self):
        self.client = DataQueryFileAPIClient(
            client_id=os.getenv("DQ_CLIENT_ID"),
            client_secret=os.getenv("DQ_CLIENT_SECRET"),
        )

    def test_download_delta_file(self):
        """Download a single delta file and verify it lands on disk."""
        available = self.client.filter_available_files_by_datetime(
            include_full_snapshots=False,
            include_delta=True,
            include_metadata=False,
        )
        self.assertFalse(available.empty, "No delta files available for today.")

        filename = available["file-name"].iloc[0]
        with tempfile.TemporaryDirectory() as tempdir:
            path = self.client.download_file(filename=filename, out_dir=tempdir)
            self.assertTrue(
                os.path.isfile(path),
                f"Downloaded delta file not found at {path}",
            )
            self.assertGreater(os.path.getsize(path), 0)

    def test_download_snapshot_file(self):
        """Download a single full-snapshot file (not all) and verify it lands on disk."""
        available = self.client.filter_available_files_by_datetime(
            include_full_snapshots=True,
            include_delta=False,
            include_metadata=False,
        )
        self.assertFalse(available.empty, "No snapshot files available for today.")

        filename = available["file-name"].iloc[0]
        with tempfile.TemporaryDirectory() as tempdir:
            path = self.client.download_file(filename=filename, out_dir=tempdir)
            self.assertTrue(
                os.path.isfile(path),
                f"Downloaded snapshot file not found at {path}",
            )
            self.assertGreater(os.path.getsize(path), 0)

    def test_fallback_url(self):
        """Pass an unreachable primary URL and verify the client falls back."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            client = DataQueryFileAPIClient(
                client_id=os.getenv("DQ_CLIENT_ID"),
                client_secret=os.getenv("DQ_CLIENT_SECRET"),
                base_url="https://example.com",
            )

        # The resolved URL must be the known fallback, not example.com
        self.assertEqual(
            client.base_url,
            DQ_FILE_API_FALLBACK_BASE_URL.rstrip("/"),
        )

        # A UserWarning about the fallback should have been emitted
        fallback_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
        self.assertTrue(
            len(fallback_warnings) > 0,
            "Expected a UserWarning about fallback URL activation.",
        )

        # Verify the client is functional by listing groups
        groups = client.list_groups()
        self.assertFalse(groups.empty)


if __name__ == "__main__":
    unittest.main()
