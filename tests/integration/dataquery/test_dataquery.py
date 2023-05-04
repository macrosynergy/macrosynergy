import unittest
import os
import pandas as pd
import datetime
from typing import List, Dict, Union, Optional
from macrosynergy.download import JPMaQSDownload
from macrosynergy.download.exceptions import (
    AuthenticationError,
    InvalidResponseError,
    InvalidDataframeError,
)
from macrosynergy.download.dataquery import DataQueryInterface


class TestDataQueryOAuth(unittest.TestCase):
    def test_authentication_error(self):
        with JPMaQSDownload(
            oauth=True,
            client_id="WRONG_CLIENT_ID",
            client_secret="NOT_A_SECRET",
            check_connection=False,
        ) as jpmaqs:
            with self.assertRaises(AuthenticationError):
                jpmaqs.check_connection()

    def test_connection(self):
        with JPMaQSDownload(
            oauth=True,
            client_id=os.getenv("DQ_CLIENT_ID"),
            client_secret=os.getenv("DQ_CLIENT_SECRET"),
            check_connection=False,
        ) as jpmaqs:
            self.assertTrue(
                jpmaqs.check_connection(),
                msg="Authentication error - unable to access DataQuery:",
            )

    def test_download_jpmaqs_data(self):
        with JPMaQSDownload(
            oauth=True,
            client_id=os.getenv("DQ_CLIENT_ID"),
            client_secret=os.getenv("DQ_CLIENT_SECRET"),
            check_connection=False,
        ) as jpmaqs:
            data = jpmaqs.download(
                tickers=["EUR_FXXR_NSA"],
                start_date=(
                    datetime.date.today() - datetime.timedelta(days=30)
                ).isoformat(),
            )

        self.assertIsInstance(data, pd.DataFrame)

        self.assertFalse(data.empty)

        self.assertGreater(data.shape[0], 0)

    def test_download_jpmaqs_data_big(self):
        # This test is to check that the download works for a large number of tickers.
        # This is to specifically test the multi-threading functionality.
        cids: List[str] = [
            "AUD",
            "CAD",
            "CHF",
            "EUR",
            "GBP",
            "NZD",
            "USD",
        ]

        xcats: List[str] = [
            "EQXR_NSA",
            "FXXR_NSA",
            "FXXR_VT10",
            "FXTARGETED_NSA",
            "FXUNTRADABLE_NSA",
        ]

        metrics: List[str] = ["all"]
        start_date: str = "2020-01-01"
        end_date: str = "2020-02-01"

        data: pd.DataFrame
        with JPMaQSDownload(
            oauth=True,
            client_id=os.getenv("DQ_CLIENT_ID"),
            client_secret=os.getenv("DQ_CLIENT_SECRET"),
            check_connection=True,
        ) as jpmaqs:
            data = jpmaqs.download(
                cids=cids,
                xcats=xcats,
                metrics=metrics,
                start_date=start_date,
                end_date=end_date,
            )

        self.assertIsInstance(data, pd.DataFrame)

        self.assertFalse(data.empty)

        self.assertGreater(data.shape[0], 0)

    def test_download_jpmaqs_invalid_df(self):
        bad_cids: List[str] = [chr(i) for i in range(65, 91)]
        bad_xcats: List[str] = bad_cids[::2][::-1]

        # with self.assertRaises(InvalidDataframeError):
        with JPMaQSDownload(
            client_id=os.getenv("DQ_CLIENT_ID"),
            client_secret=os.getenv("DQ_CLIENT_SECRET"),
        ) as jpmaqs:
            with self.assertRaises(InvalidDataframeError):
                df: pd.DataFrame = jpmaqs.download(
                    cids=bad_cids,
                    xcats=bad_xcats,
                    metrics=["all"],
                    start_date="2020-01-01",
                    end_date="2020-02-01",
                )

    def test_download_jpmaqs_catalogue(self):
        cids: List[str] = [
            "AUD",
            "CAD",
            "CHF",
            "EUR",
            "GBP",
            "NZD",
            "USD",
        ]

        xcats: List[str] = [
            "EQXR_NSA",
            "FXXR_NSA",
            "FXXR_VT10",
            "FXTARGETED_NSA",
            "FXUNTRADABLE_NSA",
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        ]
        valid_metrics: List[str] = ["value", "grading", "eop_lag", "mop_lag"]
        metrics: List[str] = valid_metrics.copy()
        start_date: str = "2020-01-01"
        end_date: str = "2020-02-01"

        # Testing whether the catalogue is being downloaded
        with JPMaQSDownload(
            client_id=os.getenv("DQ_CLIENT_ID"),
            client_secret=os.getenv("DQ_CLIENT_SECRET"),
        ) as jpmaqs:
            catalogue: List[str] = jpmaqs.get_catalogue()
            self.assertGreater(len(catalogue), 5000)

            # now test if without the catalogue, some expressions are unavailable
            df: pd.DataFrame = jpmaqs.download(
                cids=cids,
                xcats=xcats,
                metrics=metrics,
                start_date=start_date,
                end_date=end_date,
                get_catalogue=False,
            )

            self.assertIsInstance(df, pd.DataFrame)

            self.assertTrue(len(jpmaqs.unavailable_expressions) > 0)

        # Testing whether the catalogue is filtering
        with JPMaQSDownload(
            client_id=os.getenv("DQ_CLIENT_ID"),
            client_secret=os.getenv("DQ_CLIENT_SECRET"),
        ) as jpmaqs:
            df: pd.DataFrame = jpmaqs.download(
                cids=cids,
                xcats=xcats,
                metrics=metrics,
                start_date=start_date,
                end_date=end_date,
                get_catalogue=True,
            )

            self.assertIsInstance(df, pd.DataFrame)

            self.assertEqual(len(jpmaqs.unavailable_expressions), 0)

        # Lastly test with a very small set of tickers to see if the filtering is successful
        cids: List[str] = [
            "AUD",
            "CAD",
        ]
        xcats: List[str] = ["EQXR_NSA", "FXXR_NSA", "foo-bar"]

        with JPMaQSDownload(
            client_id=os.getenv("DQ_CLIENT_ID"),
            client_secret=os.getenv("DQ_CLIENT_SECRET"),
        ) as jpmaqs:
            df: pd.DataFrame = jpmaqs.download(
                cids=cids,
                xcats=xcats,
                metrics=metrics,
                start_date=start_date,
                end_date=end_date,
                get_catalogue=False,
            )

            # assert that the unavailable expressions are not empty
            self.assertTrue(len(jpmaqs.unavailable_expressions) > 0)
            bad_exprs: List[str] = jpmaqs.construct_expressions(
                cids=cids, xcats=[xcats[-1]], metrics=metrics
            )
            self.assertEqual(set(jpmaqs.unavailable_expressions), set(bad_exprs))


if __name__ == "__main__":
    unittest.main()
