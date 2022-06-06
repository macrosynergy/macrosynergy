import unittest
import os
from macrosynergy.dataquery.api import Interface


class TestDataQuery(unittest.TestCase):
    def test_connection(self):
        with Interface(
            oauth=True,
            client_id=os.getenv("dq_client_id"),
            client_secret=os.getenv("dq_client_secret")
        ) as dq:
            self.assertTrue(dq.check_connection())


if __name__ == '__main__':
    unittest.main()
