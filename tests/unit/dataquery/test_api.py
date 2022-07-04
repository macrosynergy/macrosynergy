import unittest
from unittest import mock
from macrosynergy.dataquery.api import Interface


class TestDataQueryInterface(unittest.TestCase):
    def test_check_connection(self):
        with mock.patch(
            "macrosynergy.dataquery.auth.OAuth.get_dq_api_result",
            return_value={"info": {"code": 200}}  # TODO check exact response from DataQuery Swagger docs
        ) as p_request:
            with Interface(client_id="client1", client_secret="123", oauth=True) as dq:
                self.assertTrue(dq.check_connection())
                p_request.assert_called_with(url=dq.access.base_url + "/services/heartbeat")

            p_request.assert_called_once()

    def test_check_connection_fail(self):
        with mock.patch(
            "macrosynergy.dataquery.auth.OAuth.get_dq_api_result",
            # TODO check exact response from DataQuery Swagger docs
            return_value={
                "info": {"code": 400, "message": "TODO check message from DQ", "description": "description here."}
            }
        ) as p_request:
            with Interface(client_id="client1", client_secret="123", oauth=True) as dq:
                self.assertFalse(dq.check_connection())
                p_request.assert_called_with(url=dq.access.base_url + "/services/heartbeat")

            p_request.assert_called_once()


if __name__ == '__main__':
    unittest.main()
