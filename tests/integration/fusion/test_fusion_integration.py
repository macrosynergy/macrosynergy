import unittest
import os
import pandas as pd
from macrosynergy.download.fusion_interface import JPMaQSFusionClient, FusionOAuth
from macrosynergy.management.types import QuantamentalDataFrame
import tempfile
import glob


class TestFusionIntegration(unittest.TestCase):
    def setUp(self):
        self.oauth_handler = FusionOAuth(
            client_id=os.getenv("FUSION_CLIENT_ID"),
            client_secret=os.getenv("FUSION_CLIENT_SECRET"),
        )
        self.jpmaqs_client = JPMaQSFusionClient(oauth_handler=self.oauth_handler)

    def test_dataframe_download(self):
        cids = ["USD", "GBP", "EUR", "JPY", "CHF", "AUD", "CAD"]
        xcats = ["FXXR_NSA", "EQXR_NSA"]  # , "EQCRY_NSA"]
        tickers = ["USD_EQXR_NSA", "GBP_EQXR_NSA"]

        expected_tickers = [f"{c}_{x}" for c in cids for x in xcats] + tickers
        expected_tickers = sorted(set(expected_tickers))
        expected_tickers.remove("USD_FXXR_NSA")

        df = self.jpmaqs_client.download(
            cids=cids,
            xcats=xcats,
            tickers=tickers,
            start_date="2025-07-17",
        )
        self.assertIsNotNone(df)
        self.assertIsInstance(df, QuantamentalDataFrame)

        found_tickers = QuantamentalDataFrame(df).list_tickers()
        self.assertEqual(set(found_tickers), set(expected_tickers))

    def test_full_snapshot_download(self):
        with tempfile.TemporaryDirectory() as tempdir:
            self.jpmaqs_client.download_latest_full_snapshot(
                folder=tempdir,
                keep_raw_data=False,
            )
            expected_datasets = self.jpmaqs_client.list_datasets()

            folders = [
                f
                for f in os.listdir(os.path.abspath(tempdir))
                if os.path.isdir(os.path.join(tempdir, f))
            ]
            self.assertEqual(len(folders), 1)
            date = pd.Timestamp.now().strftime("%Y-%m-%d")
            expected_folder_name_start = f"jpmaqs-download-{date}"
            self.assertTrue(folders[0].startswith(expected_folder_name_start))
            check_path = os.path.join(tempdir, folders[0])

            found_files = [
                os.path.basename(f).split(".")[0].split("-")[0]
                for f in glob.glob(os.path.join(check_path, "*.parquet"))
                if not f.endswith("jpmaqs-metadata-catalog.parquet")
            ]
            expected_file_base_names = set(expected_datasets["identifier"])

            self.assertEqual(set(expected_file_base_names), set(found_files))

    def test_latest_delta_download(self):
        with tempfile.TemporaryDirectory() as tempdir:
            self.jpmaqs_client.download_latest_delta_distribution(
                folder=tempdir, keep_raw_data=False, qdf=True, as_csv=True
            )
            expected_datasets = self.jpmaqs_client.list_datasets(
                include_full_datasets=False, include_delta_datasets=True
            )

            folders = [
                f
                for f in os.listdir(os.path.abspath(tempdir))
                if os.path.isdir(os.path.join(tempdir, f))
            ]

            self.assertEqual(len(folders), 1)
            date = pd.Timestamp.now().strftime("%Y-%m-%d")
            expected_folder_name_start = f"jpmaqs-download-{date}"
            self.assertTrue(folders[0].startswith(expected_folder_name_start))
            check_path = os.path.join(tempdir, folders[0])
            # Check for CSV files
            found_files = [
                os.path.basename(f).split(".")[0].split("-")[0]
                for f in glob.glob(os.path.join(check_path, "*.csv"))
                if not f.endswith("jpmaqs-metadata-catalog.csv")
            ]
            expected_file_base_names = set(expected_datasets["identifier"])
            self.assertEqual(set(expected_file_base_names), set(found_files))


if __name__ == "__main__":
    unittest.main()
