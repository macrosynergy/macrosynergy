import unittest
import os
from macrosynergy.dataquery.api import Interface


class TestDataQuery(unittest.TestCase):
    def test_connection(self):
        with Interface(
            oauth=True,
            client_id=os.getenv("DQ_CLIENT_ID"),
            client_secret=os.getenv("DQ_CLIENT_SECRET")
        ) as dq:
            self.assertTrue(dq.check_connection())


if __name__ == '__main__':
    unittest.main()
