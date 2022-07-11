import unittest
import os
import pandas as pd
import datetime
from macrosynergy.dataquery.api import Interface


class TestDataQueryOAuth(unittest.TestCase):

    def test_authentication_error(self):
        with Interface(
            oauth=True,
            client_id="WRONG_CLIENT_ID",
            client_secret="NOT_A_SECRET"
        ) as dq:
            with self.assertRaises(RuntimeError):
                self.assertFalse(dq.check_connection())


if __name__ == "__main__":
    unittest.main()