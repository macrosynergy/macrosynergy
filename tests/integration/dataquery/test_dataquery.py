import unittest
import os
import pandas as pd
import datetime
from macrosynergy.download import JPMaQSDownload
from macrosynergy.download.dataquery import DataQueryInterface, AuthenticationError


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
        dq = DataQueryInterface(
            oauth=True,
            client_id=os.getenv("DQ_CLIENT_ID"),
            client_secret=os.getenv("DQ_CLIENT_SECRET"),
        )
        self.assertTrue(
            dq.check_connection(),
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


if __name__ == "__main__":
    unittest.main()
