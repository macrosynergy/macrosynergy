import unittest
from unittest.mock import patch, MagicMock
import json, requests
import datetime
import pandas as pd

from macrosynergy.download.fusion_interface import (
    request_wrapper as fusion_request_wrapper,
    FusionOAuth,
    SimpleFusionAPIClient,
    JPMaQSFusionClient,
)


class TestRequestWrapper(unittest.TestCase):
    URL = "https://example.com/api"
    HDRS = {"Authorization": "Bearer test"}

    def _make_response(
        self,
        *,
        status=200,
        content=b"",
        text="",
        json_data=None,
        raise_exc=None,
    ):
        """Return a MagicMock imitating `requests.Response`."""
        r = MagicMock()
        r.status_code = status
        r.content = content
        r.text = text or content.decode(errors="ignore")
        r.url = self.URL
        r.request = MagicMock(method="GET")

        # raise_for_status
        if raise_exc:
            r.raise_for_status.side_effect = raise_exc
        else:
            r.raise_for_status.return_value = None

        # json()
        if json_data is not None:
            r.json.return_value = json_data
        else:
            r.json.side_effect = json.JSONDecodeError("Expecting value", r.text, 0)

        return r

    def _call(self, response, **kwargs):
        with patch(
            "macrosynergy.download.fusion_interface.wait_for_api_call",
            return_value=True,
        ), patch("requests.request", return_value=response):
            return fusion_request_wrapper("GET", self.URL, headers=self.HDRS, **kwargs)

    def assertRaisesMessage(self, exc_type, msg, func, *args, **kwargs):
        with self.assertRaises(exc_type) as cm:
            func(*args, **kwargs)
        self.assertIn(msg, str(cm.exception))

    def test_http_error(self):
        resp = self._make_response(
            status=400,
            text="Bad Request",
            raise_exc=requests.exceptions.HTTPError("HTTP Error"),
        )
        self.assertRaisesMessage(Exception, "API HTTP error", self._call, resp)

    def test_request_exception(self):
        resp = self._make_response(status=500, text="Server Error")
        with patch(
            "macrosynergy.download.fusion_interface.wait_for_api_call",
            return_value=True,
        ), patch(
            "requests.request",
            side_effect=requests.exceptions.RequestException(
                "Request failed", response=resp
            ),
        ):
            self.assertRaisesMessage(
                Exception,
                "API request failed",
                fusion_request_wrapper,
                "GET",
                self.URL,
                headers=self.HDRS,
            )

    def test_json_decode_error(self):
        resp = self._make_response(content=b"notjson")
        self.assertRaisesMessage(Exception, "decode JSON", self._call, resp)

    def test_status_204_returns_none(self):
        resp = self._make_response(status=204, content=b"")
        self.assertIsNone(self._call(resp))

    def test_as_bytes(self):
        resp = self._make_response(content=b"bytesdata")
        self.assertEqual(self._call(resp, as_bytes=True), b"bytesdata")

    def test_as_text(self):
        resp = self._make_response(content=b"sometext")
        self.assertEqual(self._call(resp, as_text=True), "sometext")

    def test_as_json(self):
        resp = self._make_response(content=b'{"foo": "bar"}', json_data={"foo": "bar"})
        self.assertEqual(self._call(resp, as_json=True), {"foo": "bar"})


class TestFusionOAuth(unittest.TestCase):
    def setUp(self):
        self.creds = {
            "client_id": "abc",
            "client_secret": "def",
            "resource": "resource",
            "application_name": "fusion",
            "root_url": "https://root",
            "auth_url": "https://auth",
            "proxies": None,
        }
        self.token_response = {
            "access_token": "tok123",
            "expires_in": 3600,
        }

    @patch("requests.post")
    def test_retrieve_token_success(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = self.token_response
        mock_resp.raise_for_status.return_value = None
        mock_post.return_value = mock_resp

        oauth = FusionOAuth(**self.creds)
        oauth._retrieve_token()
        self.assertIsNotNone(oauth._stored_token)
        self.assertEqual(oauth._stored_token["access_token"], "tok123")
        self.assertEqual(oauth._stored_token["expires_in"], 3600)

    @patch("requests.post")
    def test_retrieve_token_failure(self, mock_post):
        mock_post.side_effect = requests.exceptions.RequestException("fail")
        oauth = FusionOAuth(**self.creds)
        with self.assertRaises(Exception) as cm:
            oauth._retrieve_token()
        self.assertIn("Error retrieving token", str(cm.exception))

    def test_is_valid_token_false_when_none(self):
        oauth = FusionOAuth(**self.creds)
        oauth._stored_token = None
        self.assertFalse(oauth._is_valid_token())

    def test_is_valid_token_true_when_not_expired(self):
        oauth = FusionOAuth(**self.creds)
        oauth._stored_token = {
            "created_at": datetime.datetime.now() - datetime.timedelta(seconds=10),
            "expires_in": 100,
            "access_token": "tok",
        }
        self.assertTrue(oauth._is_valid_token())

    def test_is_valid_token_false_when_expired(self):
        oauth = FusionOAuth(**self.creds)
        oauth._stored_token = {
            "created_at": datetime.datetime.now() - datetime.timedelta(seconds=200),
            "expires_in": 100,
            "access_token": "tok",
        }
        self.assertFalse(oauth._is_valid_token())

    @patch.object(FusionOAuth, "_retrieve_token")
    def test_get_token_calls_retrieve_if_invalid(self, mock_retrieve):
        oauth = FusionOAuth(**self.creds)
        oauth._stored_token = None
        mock_retrieve.side_effect = lambda: setattr(
            oauth,
            "_stored_token",
            {
                "created_at": datetime.datetime.now(),
                "expires_in": 100,
                "access_token": "tok",
            },
        )
        token = oauth._get_token()
        self.assertEqual(token, "tok")
        self.assertTrue(mock_retrieve.called)

    @patch.object(FusionOAuth, "_get_token", return_value="tok")
    def test_get_auth_returns_headers(self, mock_get_token):
        oauth = FusionOAuth(**self.creds)
        headers = oauth.get_auth()
        self.assertIn("Authorization", headers)
        self.assertTrue(headers["Authorization"].startswith("Bearer "))
        self.assertIn("User-Agent", headers)


class TestSimpleFusionAPIClient(unittest.TestCase):
    ENDPOINTS = [
        ("get_common_catalog", {}),
        ("get_products", {}),
        ("get_product_details", {"product_id": "JPMAQS"}),
        (
            "get_dataset",
            {
                "catalog": "common",
                "dataset": "JPMAQS_METADATA_CATALOG",
            },
        ),
        (
            "get_dataset_series",
            {
                "catalog": "common",
                "dataset": "JPMAQS_METADATA_CATALOG",
            },
        ),
        (
            "get_dataset_seriesmember",
            {
                "catalog": "common",
                "dataset": "JPMAQS_METADATA_CATALOG",
                "seriesmember": "latest",
            },
        ),
        (
            "get_seriesmember_distributions",
            {
                "catalog": "common",
                "dataset": "JPMAQS_METADATA_CATALOG",
                "seriesmember": "latest",
            },
        ),
        (
            "get_seriesmember_distribution_details",
            {
                "catalog": "common",
                "dataset": "JPMAQS_METADATA_CATALOG",
                "seriesmember": "latest",
                "distribution": "parquet",
            },
        ),
    ]

    def setUp(self):
        self.oauth = MagicMock(spec=FusionOAuth)
        self.oauth.get_auth.return_value = {"Authorization": "Bearer test"}
        self.client = SimpleFusionAPIClient(
            self.oauth, base_url="https://example.com/api"
        )

    @patch("macrosynergy.download.fusion_interface.request_wrapper")
    def test_endpoints_return_expected_payload(self, mock_request):
        """
        Smoke-test every high-level helper delegates to request_wrapper
        and returns its payload unchanged.
        """
        mock_request.return_value = {"resources": []}

        for method_name, kwargs in self.ENDPOINTS:
            with self.subTest(endpoint=method_name):
                result = getattr(self.client, method_name)(**kwargs)
                mock_request.assert_called_once()
                self.assertEqual(result, {"resources": []})
                mock_request.reset_mock()


class TestJPMaQSFusionClient(unittest.TestCase):
    def setUp(self):
        self.oauth = MagicMock(spec=FusionOAuth)
        self.simple_client = MagicMock(spec=SimpleFusionAPIClient)
        patcher = patch(
            "macrosynergy.download.fusion_interface.SimpleFusionAPIClient",
            return_value=self.simple_client,
        )
        self.addCleanup(patcher.stop)
        self.mock_simple_client_ctor = patcher.start()
        self.client = JPMaQSFusionClient(self.oauth)

    def test_list_datasets_smoke(self):
        self.simple_client.get_product_details.return_value = {
            "resources": [
                {
                    "@id": "id1",
                    "identifier": "ds1",
                    "title": "t1",
                    "description": "desc",
                    "isRestricted": False,
                },
                {
                    "@id": "id2",
                    "identifier": "JPMAQS_METADATA_CATALOG",
                    "title": "t2",
                    "description": "desc2",
                    "isRestricted": True,
                },
            ]
        }
        # should filter out JPMAQS_METADATA_CATALOG -- not really a dataset
        df = self.client.list_datasets()
        self.assertIn("identifier", df.columns)
        self.assertTrue((df["identifier"] != "JPMAQS_METADATA_CATALOG").all())

    def test_get_metadata_catalog_smoke(self):
        fake_bytes = b"parquetbytes"
        self.simple_client.get_seriesmember_distribution_details.return_value = (
            fake_bytes
        )
        with patch("pandas.read_parquet", return_value="DF") as mock_read_parquet:
            result = self.client.get_metadata_catalog()
            mock_read_parquet.assert_called_once()
            self.assertEqual(result, "DF")

    def test_get_dataset_available_series_smoke(self):
        self.simple_client.get_dataset_series.return_value = {
            "resources": [
                {
                    "@id": "id1",
                    "identifier": "ser1",
                    "createdDate": "2020-01-01",
                    "fromDate": "2020-01-01",
                    "toDate": "2020-12-31",
                }
            ]
        }
        df = self.client.get_dataset_available_series("SOME_DATASET")
        self.assertIn("identifier", df.columns)
        self.assertIn("@id", df.columns)

    def test_get_seriesmember_distributions_smoke(self):
        self.simple_client.get_seriesmember_distributions.return_value = {
            "resources": [
                {"@id": "id1", "identifier": "dist1", "title": "Distribution 1"}
            ]
        }
        df = self.client.get_seriesmember_distributions("ds", "sm")
        self.assertIn("identifier", df.columns)
        self.assertIn("@id", df.columns)

    def test_download_series_member_distribution_smoke(self):
        fake_bytes = b"parquetbytes"
        self.simple_client.get_seriesmember_distribution_details.return_value = (
            fake_bytes
        )
        with patch("pandas.read_parquet", return_value="DF") as mock_read_parquet:
            result = self.client.download_series_member_distribution("ds", "sm")
            mock_read_parquet.assert_called_once()
            self.assertEqual(result, "DF")

    def test_download_latest_distribution_smoke(self):
        self.client.get_dataset_available_series = MagicMock(
            return_value=pd.DataFrame({"identifier": ["20230101", "20230102"]})
        )
        self.client.download_series_member_distribution = MagicMock(
            return_value=pd.DataFrame({"ticker": ["A_B"]})
        )
        with patch(
            "macrosynergy.download.fusion_interface.convert_ticker_based_parquet_to_qdf",
            return_value="QDF",
        ) as mock_convert:
            result = self.client.download_latest_distribution("ds")
            mock_convert.assert_called_once()
            self.assertEqual(result, "QDF")


if __name__ == "__main__":
    unittest.main()
