import unittest
from unittest.mock import patch, MagicMock
import json
import datetime
import io

import tempfile
import os
import warnings
import time

import pandas as pd
import requests

from macrosynergy.download.fusion_interface import cache_decorator
from macrosynergy.compat import PD_2_0_OR_LATER
from macrosynergy.management.simulate import make_test_df
from macrosynergy.management.utils.df_utils import is_categorical_qdf
from macrosynergy.management.types import QuantamentalDataFrame
import pyarrow as pa

from macrosynergy.download import fusion_interface as fusion_interface_module

from macrosynergy.download.fusion_interface import (
    request_wrapper as fusion_request_wrapper,
    convert_ticker_based_pandas_df_to_qdf,
    get_resources_df,
    FusionOAuth,
    SimpleFusionAPIClient,
    JPMaQSFusionClient,
    read_parquet_from_bytes_to_pandas_dataframe,
    request_wrapper_stream_bytes_to_disk,
    NoContentError,
    convert_ticker_based_parquet_file_to_qdf,
    read_parquet_from_bytes_to_pyarrow_table,
    coerce_real_date,
    filter_parquet_table_as_qdf,
    convert_ticker_based_pyarrow_table_to_qdf,
    _wait_for_api_call,
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
            "macrosynergy.download.fusion_interface._wait_for_api_call",
            return_value=True,
        ):
            with patch("requests.request", return_value=response):
                return fusion_request_wrapper(
                    "GET", self.URL, headers=self.HDRS, **kwargs
                )

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
            "macrosynergy.download.fusion_interface._wait_for_api_call",
            return_value=True,
        ):
            with patch(
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
        with self.assertRaises(NoContentError):
            resp = self._make_response(status=204, content=b"")
            self._call(resp)

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
        oauth.retrieve_token()
        self.assertIsNotNone(oauth._stored_token)
        self.assertEqual(oauth._stored_token["access_token"], "tok123")
        self.assertEqual(oauth._stored_token["expires_in"], 3600)

    @patch("requests.post")
    def test_retrieve_token_failure(self, mock_post):
        mock_post.side_effect = requests.exceptions.RequestException("fail")
        oauth = FusionOAuth(**self.creds)
        with self.assertRaises(Exception) as cm:
            oauth.retrieve_token()
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

    @patch.object(FusionOAuth, "retrieve_token")
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

    def test_get_seriesmember_distribution_details_to_disk_calls_stream_bytes(self):
        with patch(
            "macrosynergy.download.fusion_interface.request_wrapper_stream_bytes_to_disk"
        ) as mock_stream:
            # Arrange
            filename = "out.parquet"
            catalog = "common"
            dataset = "MY_DS"
            seriesmember = "latest"
            extra_kwargs = {"timeout": 30, "verify": False}

            # Act
            self.client.get_seriesmember_distribution_details_to_disk(
                filename=filename,
                catalog=catalog,
                dataset=dataset,
                seriesmember=seriesmember,
                **extra_kwargs,
            )

            # Assert
            expected_endpoint = (
                f"catalogs/{catalog}"
                f"/datasets/{dataset}"
                f"/datasetseries/{seriesmember}"
                f"/distributions/parquet"
            )
            expected_url = f"{self.client.base_url}/{expected_endpoint}"
            mock_stream.assert_called_once_with(
                filename=filename,
                headers=self.oauth.get_auth.return_value,
                url=expected_url,
                method="GET",
                **extra_kwargs,
            )


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

    def test_list_datasets(self):
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

    def test_get_metadata_catalog(self):
        fake_bytes = b"parquetbytes"
        self.simple_client.get_seriesmember_distribution_details.return_value = (
            fake_bytes
        )
        with patch("pandas.read_parquet", return_value="DF") as mock_read_parquet:
            result = self.client.get_metadata_catalog()
            mock_read_parquet.assert_called_once()
            self.assertEqual(result, "DF")

    def test_get_dataset_available_series(self):
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

    def test_get_seriesmember_distributions(self):
        self.simple_client.get_seriesmember_distributions.return_value = {
            "resources": [
                {"@id": "id1", "identifier": "dist1", "title": "Distribution 1"}
            ]
        }
        df = self.client.get_seriesmember_distributions("ds", "sm")
        self.assertIn("identifier", df.columns)
        self.assertIn("@id", df.columns)

    def test_download_series_member_distribution(self):
        fake_bytes = b"parquetbytes"
        self.simple_client.get_seriesmember_distribution_details.return_value = (
            fake_bytes
        )
        with patch("pandas.read_parquet", return_value="DF") as mock_read_parquet:
            result = self.client.download_series_member_distribution("ds", "sm")
            mock_read_parquet.assert_called_once()
            self.assertEqual(result, "DF")

    def test_download_latest_distribution(self):
        self.client.get_dataset_available_series = MagicMock(
            return_value=pd.DataFrame({"identifier": ["20230101", "20230102"]})
        )
        self.client.download_series_member_distribution = MagicMock(
            return_value=pd.DataFrame({"ticker": ["A_B"]})
        )
        with patch(
            "macrosynergy.download.fusion_interface.convert_ticker_based_pandas_df_to_qdf",
            return_value="QDF",
        ) as mock_convert:
            result = self.client.download_latest_distribution("ds")
            mock_convert.assert_called_once()
            self.assertEqual(result, "QDF")


class TestJPMaQSFusionClientTickers(unittest.TestCase):
    def setUp(self):
        self.oauth = MagicMock(spec=FusionOAuth)
        self.client = JPMaQSFusionClient(self.oauth)

    def test_list_tickers_returns_sorted_list(self):
        # Mock get_metadata_catalog to return a DataFrame with 'Ticker' column
        df = pd.DataFrame({"Ticker": ["BBB", "aaa", "CCC"]})
        self.client.get_metadata_catalog = MagicMock(return_value=df)
        result = self.client.list_tickers()
        self.assertEqual(result, sorted(["BBB", "aaa", "CCC"]))

    def test_list_tickers_missing_column_raises(self):
        df = pd.DataFrame({"NotTicker": [1, 2, 3]})
        self.client.get_metadata_catalog = MagicMock(return_value=df)
        with self.assertRaises(ValueError) as cm:
            self.client.list_tickers()
        self.assertIn("'Ticker' column not found", str(cm.exception))

    def test_get_ticker_metadata_found(self):
        df = pd.DataFrame({"Ticker": ["AAA", "BBB", "CCC"], "Meta": [1, 2, 3]})
        self.client.get_metadata_catalog = MagicMock(return_value=df)
        result = self.client.get_ticker_metadata("bbb")
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["Ticker"], "BBB")
        self.assertEqual(result.iloc[0]["Meta"], 2)

    def test_get_ticker_metadata_not_found(self):
        df = pd.DataFrame({"Ticker": ["AAA", "BBB"], "Meta": [1, 2]})
        self.client.get_metadata_catalog = MagicMock(return_value=df)
        with self.assertRaises(ValueError) as cm:
            self.client.get_ticker_metadata("ZZZ")
        self.assertIn("No metadata found for ticker", str(cm.exception))


class TestGetResourcesDf(unittest.TestCase):
    def setUp(self):
        self.resources = [
            {"@id": "id1", "identifier": "foo", "title": "Foo", "extra": 1},
            {"@id": "id2", "identifier": "bar", "title": "Bar", "extra": 2},
        ]
        self.response_dict = {"resources": self.resources}

    def test_missing_resources_key_raises(self):
        with self.assertRaises(ValueError) as cm:
            get_resources_df({}, resources_key="resources")
        self.assertIn("Field 'resources' not found", str(cm.exception))

    def test_missing_at_id_raises(self):
        bad = {"resources": [{"identifier": "foo"}]}
        with self.assertRaises(ValueError) as cm:
            get_resources_df(bad)
        self.assertIn("Column '@id' not found", str(cm.exception))

    def test_keep_fields(self):
        df = get_resources_df(self.response_dict, keep_fields=["@id", "identifier"])
        self.assertListEqual(list(df.columns), ["@id", "identifier"])

    def test_custom_sort_columns_true(self):
        df = get_resources_df(self.response_dict, custom_sort_columns=True)
        # Should start with @id, identifier, title
        self.assertEqual(list(df.columns)[:3], ["@id", "identifier", "title"])

    def test_custom_sort_columns_false(self):
        df = get_resources_df(self.response_dict, custom_sort_columns=False)
        # Should preserve original order
        self.assertListEqual(list(df.columns), list(self.resources[0].keys()))

    def test_missing_title_column(self):
        # Remove 'title' from one row, should still work
        resources = [
            {"@id": "id1", "identifier": "foo", "extra": 1},
            {"@id": "id2", "identifier": "bar", "extra": 2},
        ]
        response_dict = {"resources": resources}
        df = get_resources_df(response_dict, custom_sort_columns=True)
        self.assertIn("@id", df.columns)
        self.assertIn("identifier", df.columns)
        self.assertNotIn("title", df.columns)


class TestWaitSimple(unittest.TestCase):
    def test_repeated_calls_delay(self):
        fusion_interface_module.FUSION_API_DELAY = 0.5
        fusion_interface_module.LAST_API_CALL = None

        calls = 5
        start = time.time()
        for _ in range(calls):
            _wait_for_api_call()
        elapsed = time.time() - start

        expected = (calls - 1) * fusion_interface_module.FUSION_API_DELAY
        self.assertGreaterEqual(
            elapsed,
            expected,
            f"Elapsed {elapsed:.2f}s should be >= expected {expected:.2f}s",
        )
        self.assertLess(
            elapsed - expected,
            0.1,
            f"Test overhead too large: extra {elapsed - expected:.2f}s",
        )


class TestUtilityFunctions(unittest.TestCase):
    def setUp(self):
        qdf = make_test_df(start="2010-01-01", end="2011-02-01")
        self.expected_qdf = qdf.copy()

        qdf["ticker"] = qdf["cid"] + "_" + qdf["xcat"]
        qdf = qdf.drop(columns=["cid", "xcat"])
        self.qdf = qdf

    def test_convert_ticker_based_pandas_df_to_qdf_empty(self):
        result = convert_ticker_based_pandas_df_to_qdf(self.qdf, categorical=False)
        pd.testing.assert_frame_equal(result, self.expected_qdf)
        self.assertFalse(is_categorical_qdf(result))

    def test_convert_ticker_based_pandas_df_to_qdf_categorical(self):
        result = convert_ticker_based_pandas_df_to_qdf(self.qdf, categorical=True)
        self.expected_qdf = QuantamentalDataFrame(self.expected_qdf, categorical=True)
        pd.testing.assert_frame_equal(result, self.expected_qdf)
        self.assertTrue(is_categorical_qdf(result))

    def test_read_parquet_from_bytes(self):
        df = self.qdf.copy()
        buf = io.BytesIO()
        df.to_parquet(buf, index=False)
        buf.seek(0)
        bytes_data = buf.read()
        result = read_parquet_from_bytes_to_pandas_dataframe(bytes_data)
        pd.testing.assert_frame_equal(result, df)


class TestParquetArrowFunctions(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            {
                "ticker": ["AAA_BBB", "CCC_DDD"],
                "real_date": ["2023-02-01", "2023-02-02"],
                "value": [1.0, 2.0],
                "grading": [0, 1],
                "eop_lag": [0, 0],
                "mop_lag": [0, 0],
                "last_updated": ["2023-02-01", "2023-02-02"],
            }
        )
        self.table = pa.Table.from_pandas(self.df)
        self.bad_table = pa.Table.from_pandas(self.df.drop(columns=["ticker"]))

    def test_convert_ticker_based_pyarrow_table_to_qdf(self):
        with patch(
            "macrosynergy.download.fusion_interface.convert_ticker_based_pyarrow_table_to_qdf",
            return_value=self.table,
        ):
            qdf_table = convert_ticker_based_pyarrow_table_to_qdf(self.table)
            qdf_table = QuantamentalDataFrame(qdf_table.to_pandas())
            expc: pd.DataFrame = self.table.to_pandas()
            expc[["cid", "xcat"]] = expc["ticker"].str.split("_", expand=True, n=1)
            expc = expc.drop(columns=["ticker"])
            expc = QuantamentalDataFrame(expc)

            if PD_2_0_OR_LATER:
                self.assertTrue((qdf_table == expc).all().all())
            else:
                self.assertTrue(
                    pd.DataFrame(qdf_table).eq(pd.DataFrame(expc)).all().all()
                )

    def test_convert_ticker_based_pyarrow_table_to_qdf_error(self):
        with self.assertRaises(KeyError):
            convert_ticker_based_pyarrow_table_to_qdf(self.bad_table)

    def test_read_parquet_from_bytes_to_pyarrow_table(self):
        buf = io.BytesIO()
        self.df.to_parquet(buf, index=False)
        buf.seek(0)
        bytes_data = buf.read()
        table = read_parquet_from_bytes_to_pyarrow_table(bytes_data)
        pd.testing.assert_frame_equal(table.to_pandas(), self.df)

    def test_coerce_real_date(self):
        table = self.table
        coerced = coerce_real_date(table)
        # Should be date32 type
        self.assertEqual(coerced.column("real_date").type, pa.date32())
        # Should match the original dates
        dates = pd.to_datetime(self.df["real_date"]).values
        coerced_dates = pd.to_datetime(coerced.to_pandas()["real_date"]).values
        self.assertTrue((dates == coerced_dates).all())

    def test_filter_parquet_table_as_qdf(self):
        with patch(
            "macrosynergy.download.fusion_interface.convert_ticker_based_parquet_file_to_qdf"
        ):
            tickers = ["AAA_BBB"]
            filtered = filter_parquet_table_as_qdf(
                self.table,
                tickers=tickers,
                start_date="2023-02-01",
                end_date="2023-02-03",
                qdf=True,
            )
            df_filtered = filtered.to_pandas()
            self.assertEqual(len(df_filtered), 1)


    def test_filter_parquet_table_as_qdf_error(self):
        with self.assertRaises(TypeError):
            filter_parquet_table_as_qdf(table=123)

        with self.assertRaises(KeyError):
            filter_parquet_table_as_qdf(self.bad_table, tickers=["AAA_BBB"])


class TestFusionInterfaceEdgeCases(unittest.TestCase):
    def setUp(self):
        self.creds = {
            "client_id": "abc",
            "client_secret": "def",
            "resource": "resource",
            "application_name": "fusion",
            "root_url": "https://root",
            "auth_url": "https://auth",
            "proxies": {"http": "proxy"},
        }

    def test_fusionoauth_from_credentials_json_missing_keys(self):
        creds = {"client_id": "a"}  # missing client_secret
        with tempfile.NamedTemporaryFile("w", delete=False) as f:
            json.dump(creds, f)
            fname = f.name
        try:
            with self.assertRaises(TypeError):
                FusionOAuth.from_credentials_json(fname)
        finally:
            os.remove(fname)

    def test_fusionoauth_from_credentials_invalid_dict(self):
        with self.assertRaises(TypeError):
            FusionOAuth.from_credentials({"client_id": "a"})

    def test_fusionoauth_init_proxies(self):
        oauth = FusionOAuth(**self.creds)
        self.assertEqual(oauth.proxies, {"http": "proxy"})

    def test_cache_decorator_expiry_and_manual_clear(self):
        calls = []

        @cache_decorator(ttl=1)
        def f(x):
            calls.append(x)
            return x

        f(1)
        f(1)
        self.assertEqual(len(calls), 1)
        f.cache_clear()
        f(1)
        self.assertEqual(len(calls), 2)

        time.sleep(1.1)
        f(1)
        self.assertEqual(len(calls), 3)

    def test_cache_decorator_maxsize(self):
        @cache_decorator(ttl=10, maxsize=2)
        def f(x):
            return x

        self.assertEqual(f(1), 1)
        self.assertEqual(f(2), 2)
        self.assertEqual(f(3), 3)

    def test_request_wrapper_invalid_method(self):
        with self.assertRaises(ValueError):
            fusion_request_wrapper("PATCH", "url")

    def test_request_wrapper_type_error(self):
        with self.assertRaises(TypeError):
            fusion_request_wrapper(123, "url")

    def test_request_wrapper_multiple_as_flags(self):
        # Patch requests.request to avoid real HTTP call
        with patch("requests.request") as mock_req:
            with patch(
                "macrosynergy.download.fusion_interface._wait_for_api_call",
                return_value=True,
            ):
                mock_req.return_value = MagicMock(
                    status_code=200,
                    content=b"{}",
                    raise_for_status=lambda: None,
                    json=lambda: {},
                )
                with self.assertRaises(ValueError):
                    fusion_request_wrapper(
                        "GET", "http://example.com", as_bytes=True, as_text=True
                    )

    @patch(
        "macrosynergy.download.fusion_interface._wait_for_api_call", return_value=True
    )
    @patch("requests.request")
    def test_request_wrapper_empty_content(self, mock_req, _):
        resp = MagicMock()
        resp.status_code = 200
        resp.content = b""
        resp.raise_for_status.return_value = None
        mock_req.return_value = resp
        with self.assertRaises(NoContentError):
            fusion_request_wrapper("GET", "url")

    @patch(
        "macrosynergy.download.fusion_interface._wait_for_api_call", return_value=True
    )
    @patch("requests.request")
    def test_request_wrapper_json_decode_error_with_raw_response(self, mock_req, _):
        resp = MagicMock()
        resp.status_code = 200
        resp.content = b"notjson"
        resp.raise_for_status.return_value = None
        resp.json.side_effect = json.JSONDecodeError("Expecting value", "", 0)
        resp.text = "notjson"
        mock_req.return_value = resp
        with self.assertRaises(Exception) as cm:
            fusion_request_wrapper("GET", "url")
        self.assertIn("decode JSON", str(cm.exception))
        self.assertIn("Response text", str(cm.exception))

    def test_simplefusionapiclient_type_error(self):
        with self.assertRaises(TypeError):
            SimpleFusionAPIClient(oauth_handler="not_oauth")

    def test_simplefusionapiclient_proxy_warning(self):
        oauth = MagicMock(spec=FusionOAuth)
        oauth.proxies = {"http": "foo"}
        with warnings.catch_warnings(record=True) as w:
            SimpleFusionAPIClient(oauth, proxies={"http": "bar"})
            self.assertTrue(
                any("Proxies defined for OAuth handler" in str(x.message) for x in w)
            )

    def test_get_resources_df_keep_fields_missing(self):
        d = {"resources": [{"@id": "id1", "identifier": "foo"}]}
        with self.assertRaises(KeyError):
            get_resources_df(d, keep_fields=["@id", "missing"])  # missing field

    def test_get_resources_df_custom_sort_columns_missing_column(self):
        d = {"resources": [{"@id": "id1", "identifier": "foo"}]}
        # Should not raise AssertionError, but should work (title is optional)
        df = get_resources_df(d, custom_sort_columns=True)
        self.assertIn("@id", df.columns)
        self.assertIn("identifier", df.columns)
        self.assertNotIn("title", df.columns)

    def test_convert_ticker_based_pandas_df_to_qdf_missing_ticker(self):
        df = pd.DataFrame({"foo": [1, 2]})
        with self.assertRaises(KeyError):
            convert_ticker_based_pandas_df_to_qdf(df)

    def test_convert_ticker_based_pandas_df_to_qdf_malformed_ticker(self):
        df = pd.DataFrame({"ticker": ["A"]})
        with self.assertRaises(ValueError):
            convert_ticker_based_pandas_df_to_qdf(df)

    def test_read_parquet_from_bytes_keyboardinterrupt(self):
        with patch("pandas.read_parquet", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                read_parquet_from_bytes_to_pandas_dataframe(b"bytes")

    def test_read_parquet_from_bytes_invalid(self):
        with patch("pandas.read_parquet", side_effect=Exception("fail")):
            with self.assertRaises(ValueError) as cm:
                read_parquet_from_bytes_to_pandas_dataframe(b"bytes")
            self.assertIn("Failed to read Parquet".lower(), str(cm.exception).lower())

    def test_jpmaqsclient_list_datasets_all_explorer(self):
        # All datasets are explorer datasets
        with patch(
            "macrosynergy.download.fusion_interface.SimpleFusionAPIClient"
        ) as mock_client:
            inst = mock_client.return_value
            inst.get_product_details.return_value = {
                "resources": [
                    {
                        "@id": "id1",
                        "identifier": "JPMAQS_EXPLORER_1",
                        "title": "t",
                        "description": "d",
                        "isRestricted": False,
                    },
                    {
                        "@id": "id2",
                        "identifier": "JPMAQS_EXPLORER_2",
                        "title": "t2",
                        "description": "d2",
                        "isRestricted": False,
                    },
                ]
            }
            with warnings.catch_warnings(record=True) as w:
                client = JPMaQSFusionClient(FusionOAuth(**self.creds))
                df = client.list_datasets(include_explorer_datasets=False)
                self.assertTrue(
                    any("include_explorer_datasets" in str(x.message) for x in w)
                )
                self.assertEqual(len(df), 0)

    def test_jpmaqsclient_list_datasets_all_non_full(self):
        with patch(
            "macrosynergy.download.fusion_interface.SimpleFusionAPIClient"
        ) as mock_client:
            inst = mock_client.return_value
            inst.get_product_details.return_value = {
                "resources": [
                    {
                        "@id": "id1",
                        "identifier": "JPMAQS_DELTA_ABC",
                        "title": "t",
                        "description": "d",
                        "isRestricted": False,
                    },
                    {
                        "@id": "id2",
                        "identifier": "JPMAQS_EXPLORER_DEF",
                        "title": "t2",
                        "description": "d2",
                        "isRestricted": False,
                    },
                    {
                        "@id": "id3",
                        "identifier": "JPMAQS_XYZ",
                        "title": "t3",
                        "description": "d3",
                        "isRestricted": False,
                    },
                ]
            }
            client = JPMaQSFusionClient(FusionOAuth(**self.creds))
            df = client.list_datasets(
                include_full_datasets=False, include_delta_datasets=True
            )
            self.assertEqual(len(df), 1)
            self.assertTrue((df["identifier"].isin(["JPMAQS_DELTA_ABC"])).all())

    def test_jpmaqsclient_list_datasets_error(self):
        client = JPMaQSFusionClient(FusionOAuth(**self.creds))
        try:
            client.list_datasets(include_full_datasets=False)
        except Exception as e:
            expc_msg = "At least one of `include_catalog`, `include_full_datasets`, `include_explorer_datasets`, or `include_delta_datasets` must be True."
            self.assertIn(expc_msg, str(e))
            self.assertIsInstance(e, ValueError)

    def test_jpmaqsclient_download_latest_distribution_empty_series(self):
        with patch(
            "macrosynergy.download.fusion_interface.SimpleFusionAPIClient"
        ) as mock_client:
            inst = mock_client.return_value
            inst.get_dataset_series.return_value = {"resources": []}
            client = JPMaQSFusionClient(FusionOAuth(**self.creds))
            with self.assertRaises(KeyError):
                client.download_latest_distribution("ds")


class TestRequestWrapperStreamBytesToDisk(unittest.TestCase):
    @patch(
        "macrosynergy.download.fusion_interface._wait_for_api_call", return_value=True
    )
    @patch("requests.request")
    def test_stream_bytes_to_disk_writes_file(self, mock_request, _):
        # Prepare mock response with iter_content
        mock_resp = MagicMock()
        mock_resp.__enter__.return_value = mock_resp
        mock_resp.__exit__.return_value = False
        mock_resp.raise_for_status.return_value = None
        # Simulate two chunks
        mock_resp.iter_content.return_value = [b"abc", b"def"]
        mock_request.return_value = mock_resp

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "testfile.bin")
            request_wrapper_stream_bytes_to_disk(
                filename=file_path,
                url="http://example.com/file",
                method="GET",
                headers={"Authorization": "Bearer test"},
            )
            # Check file exists and content is correct
            self.assertTrue(os.path.exists(file_path))
            with open(file_path, "rb") as f:
                content = f.read()
            self.assertEqual(content, b"abcdef")

    @patch(
        "macrosynergy.download.fusion_interface._wait_for_api_call", return_value=True
    )
    def test_stream_bytes_to_disk_invalid_method(self, _):
        with self.assertRaises(ValueError):
            request_wrapper_stream_bytes_to_disk(
                filename="dummy",
                url="http://example.com/file",
                method="POST",
            )

        with self.assertRaises(TypeError):
            request_wrapper_stream_bytes_to_disk(
                filename="dummy",
                url="http://example.com/file",
                method=123,
            )

    @patch(
        "macrosynergy.download.fusion_interface._wait_for_api_call", return_value=True
    )
    @patch("requests.request")
    def test_stream_bytes_to_disk_creates_directory(self, mock_request, _):
        mock_resp = MagicMock()
        mock_resp.__enter__.return_value = mock_resp
        mock_resp.__exit__.return_value = False
        mock_resp.raise_for_status.return_value = None
        mock_resp.iter_content.return_value = [b"xyz"]
        mock_request.return_value = mock_resp

        with tempfile.TemporaryDirectory() as tmpdir:
            subdir = os.path.join(tmpdir, "subdir")
            file_path = os.path.join(subdir, "file.bin")
            self.assertFalse(os.path.exists(subdir))
            request_wrapper_stream_bytes_to_disk(
                filename=file_path,
                url="http://example.com/file",
                method="GET",
            )
            self.assertTrue(os.path.exists(file_path))
            with open(file_path, "rb") as f:
                self.assertEqual(f.read(), b"xyz")

    @patch(
        "macrosynergy.download.fusion_interface._wait_for_api_call", return_value=True
    )
    @patch("requests.request")
    def test_stream_bytes_to_disk_writes_many_chunks(self, mock_request, _):
        mock_resp = MagicMock()
        mock_resp.__enter__.return_value = mock_resp
        mock_resp.__exit__.return_value = False
        mock_resp.raise_for_status.return_value = None
        # 10 bytes per chunk, values 1..20
        chunks = [bytes([i]) * 10 for i in range(1, 21)]
        mock_resp.iter_content.return_value = chunks
        mock_request.return_value = mock_resp

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "testfile_many_chunks.bin")
            request_wrapper_stream_bytes_to_disk(
                filename=file_path,
                url="http://example.com/file",
                method="GET",
                headers={"Authorization": "Bearer test"},
            )
            self.assertTrue(os.path.exists(file_path))
            with open(file_path, "rb") as f:
                content = f.read()
            expected_content = b"".join(chunks)
            self.assertEqual(content, expected_content)

    def test_convert_ticker_based_parquet_file_to_qdf_creates_qdf_file_and_csv(self):
        df = pd.DataFrame(
            {
                "ticker": ["AAA_BBB", "CCC_DDD"],
                "real_date": [20230101, 20230102],
                "value": [1.0, 2.0],
                "grading": [0, 1],
                "eop_lag": [0, 0],
                "mop_lag": [0, 0],
                "last_updated": [20230101, 20230102],
            }
        )

        for as_csv in [False, True]:
            for keep_raw_data in [False, True]:
                with tempfile.TemporaryDirectory() as tmpdir:
                    parquet_path = os.path.join(tmpdir, "test.parquet")
                    df.to_parquet(parquet_path, index=False)
                    convert_ticker_based_parquet_file_to_qdf(
                        parquet_path,
                        qdf=True,
                        as_csv=as_csv,
                        keep_raw_data=keep_raw_data,
                    )

                    base, ext = os.path.splitext(parquet_path)
                    if as_csv:
                        expected_path = (
                            os.path.join(tmpdir, "test_qdf.csv")
                            if keep_raw_data
                            else base + ".csv"
                        )
                        readfunc = pd.read_csv
                    else:
                        expected_path = (
                            os.path.join(tmpdir, "test_qdf.parquet")
                            if keep_raw_data
                            else parquet_path
                        )
                        readfunc = pd.read_parquet

                    self.assertTrue(os.path.exists(expected_path))
                    qdf = readfunc(expected_path)
                    try:
                        QuantamentalDataFrame(qdf)
                    except Exception as e:
                        self.fail(f"Failed to load QDF file: {e}")

                    tkrs = qdf["cid"] + "_" + qdf["xcat"]
                    expected_cols = (set(df.columns) - {"ticker"}) | {"cid", "xcat"}
                    self.assertEqual(set(qdf.columns), expected_cols)
                    self.assertEqual(set(tkrs), set(df["ticker"]))

                    # Check raw file existence/deletion
                    if keep_raw_data:
                        self.assertTrue(os.path.exists(parquet_path))
                    else:
                        # If as_csv and not keep_raw_data, parquet should be deleted
                        if as_csv:
                            self.assertFalse(os.path.exists(parquet_path))
                        else:
                            self.assertTrue(os.path.exists(parquet_path))

    def test_convert_ticker_based_parquet_file_to_qdf_do_nothing_case(self):
        df = pd.DataFrame(
            {
                "ticker": ["AAA_BBB", "CCC_DDD"],
                "real_date": [20230101, 20230102],
                "value": [1.0, 2.0],
                "grading": [0, 1],
                "eop_lag": [0, 0],
                "mop_lag": [0, 0],
                "last_updated": [20230101, 20230102],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            parquet_path = os.path.join(tmpdir, "test.parquet")
            df.to_parquet(parquet_path, index=False)
            convert_ticker_based_parquet_file_to_qdf(parquet_path, qdf=False)
            self.assertTrue(os.path.exists(parquet_path))
            # check that this is the same file
            qdf = pd.read_parquet(parquet_path)
            pd.testing.assert_frame_equal(qdf, df)

    def test_convert_ticker_based_parquet_file_to_qdf_as_csv_not_qdf(self):
        df = pd.DataFrame(
            {
                "ticker": ["AAA_BBB", "CCC_DDD"],
                "real_date": [20230101, 20230102],
                "value": [1.01, 2.02],
                "grading": [0.1, 1.1],
                "eop_lag": [0.2, 0.3],
                "mop_lag": [0.4, 0.5],
                "last_updated": [20230101, 20230102],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            parquet_path = os.path.join(tmpdir, "test.parquet")
            df.to_parquet(parquet_path, index=False)
            convert_ticker_based_parquet_file_to_qdf(
                parquet_path, qdf=False, as_csv=True, keep_raw_data=False
            )
            expected_file = os.path.join(tmpdir, "test.csv")
            self.assertTrue(os.path.exists(expected_file))
            # check that this is the same file
            qdf = pd.read_csv(expected_file)
            pd.testing.assert_frame_equal(qdf, df)

    def test_convert_ticker_based_parquet_file_to_qdf_error(self):
        with self.assertRaises(FileNotFoundError):
            convert_ticker_based_parquet_file_to_qdf("non_existant_file.parquet")


if __name__ == "__main__":
    unittest.main()
