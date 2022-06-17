import unittest
import os
import pandas as pd
import datetime
from macrosynergy.dataquery.api import Interface


class TestDataQueryOAuth(unittest.TestCase):
    def test_connection(self):
        with Interface(
            oauth=True,
            client_id=os.getenv("DQ_CLIENT_ID"),
            client_secret=os.getenv("DQ_CLIENT_SECRET")
        ) as dq:
            self.assertTrue(dq.check_connection())

    def test_authentication_error(self):
        with Interface(
            oauth=True,
            client_id="WRONG_CLIENT_ID",
            client_secret="NOT_A_SECRET"
        ) as dq:
            with self.assertRaises(RuntimeError, msg="Authentication error - unable to access DataQuery:"):
                self.assertFalse(dq.check_connection())

    def test_download_jpmaqs_data(self):
        with Interface(
            oauth=True,
            client_id=os.getenv("DQ_CLIENT_ID"),
            client_secret=os.getenv("DQ_CLIENT_SECRET")
        ) as dq:
            data = dq.download(
                tickers="EUR_FXXR_NSA",
                start_date=(datetime.date.today() - datetime.timedelta(days=10)).isoformat()
            )
        self.assertIsInstance(data, pd.DataFrame)

        self.assertFalse(data.empty)

        self.assertGreater(data.shape[0], 0)


if __name__ == '__main__':
    unittest.main()
