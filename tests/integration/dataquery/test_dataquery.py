import unittest
from unittest.mock import patch, Mock
import os
import pandas as pd
import datetime
from typing import List, Dict, Union, Optional, Any, Tuple
import requests
from macrosynergy.download import JPMaQSDownload
from macrosynergy.download.dataquery import (
    DataQueryInterface,
    request_wrapper,
    validate_response,
)
from macrosynergy.download.dataquery import (
    OAUTH_BASE_URL,
    OAUTH_TOKEN_URL,
    HEARTBEAT_ENDPOINT,
    TIMESERIES_ENDPOINT,
)
from macrosynergy.download.exceptions import (
    AuthenticationError,
    HeartbeatError,
    InvalidResponseError,
    DownloadError,
    InvalidDataframeError,
)
from macrosynergy.management.utils import Config


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

        with DataQueryInterface(
            oauth=True,
            config=Config(
                client_id="WRONG_CLIENT_ID",
                client_secret="NOT_A_SECRET",
            ),
        ) as dq:
            with self.assertRaises(AuthenticationError):
                dq.check_connection()

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

        with DataQueryInterface(
            oauth=True,
            config=Config(
                client_id=os.getenv("DQ_CLIENT_ID"),
                client_secret=os.getenv("DQ_CLIENT_SECRET"),
            ),
        ) as dq:
            self.assertTrue(
                dq.check_connection(),
                msg="Authentication error - unable to access DataQuery:",
            )

    def test_download_jpmaqs_data(self):
        data: pd.DataFrame
        with JPMaQSDownload(
            oauth=True,
            client_id=os.getenv("DQ_CLIENT_ID"),
            client_secret=os.getenv("DQ_CLIENT_SECRET"),
            check_connection=False,
        ) as jpmaqs:
            data: pd.DataFrame = jpmaqs.download(
                tickers=["EUR_FXXR_NSA"],
                start_date=(
                    datetime.date.today() - datetime.timedelta(days=30)
                ).isoformat(),
            )

        self.assertIsInstance(data, pd.DataFrame)

        self.assertFalse(data.empty)

        self.assertGreater(data.shape[0], 0)

        test_expr: str = "DB(JPMAQS,EUR_FXXR_NSA,value)"
        with DataQueryInterface(
            oauth=True,
            config=Config(
                client_id=os.getenv("DQ_CLIENT_ID"),
                client_secret=os.getenv("DQ_CLIENT_SECRET"),
            ),
        ) as dq:
            data: List[str] = dq.download_data(
                expressions=[test_expr],
            )

        self.assertIsInstance(data, list)
        self.assertEqual(len(data), 1)
        self.assertIsInstance(data[0], dict)
        _data: Dict[str, Any] = data[0]
        self.assertEqual(len(_data.keys()), 5)
        for key in ["item", "group", "attributes", "instrument-id", "instrument-name"]:
            self.assertIn(key, _data.keys())

        self.assertIsInstance(_data["attributes"], list)
        self.assertEqual(_data["attributes"][0]["expression"], test_expr)
        self.assertIsInstance(_data["attributes"][0]["time-series"], list)
        self.assertGreater(len(_data["attributes"][0]["time-series"]), 0)

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


if __name__ == "__main__":
    unittest.main()
