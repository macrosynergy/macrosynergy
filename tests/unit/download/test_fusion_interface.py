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
from pathlib import Path

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

        if raise_exc:
            r.raise_for_status.side_effect = raise_exc
        else:
            r.raise_for_status.return_value = None

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
            filename = "out.parquet"
            catalog = "common"
            dataset = "MY_DS"
            seriesmember = "latest"
            extra_kwargs = {"timeout": 30, "verify": False}

            self.client.get_seriesmember_distribution_details_to_disk(
                filename=filename,
                catalog=catalog,
                dataset=dataset,
                seriesmember=seriesmember,
                **extra_kwargs,
            )

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
        # should filter out metadata and notifs datasets -- not really datasets
        df = self.client.list_datasets()
        self.assertIn("identifier", df.columns)
        ds = [self.client._catalog_dataset, self.client._notifications_dataset]
        self.assertTrue(df[df["identifier"].isin(ds)].empty)

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
        self.assertEqual(list(df.columns)[:3], ["@id", "identifier", "title"])

    def test_custom_sort_columns_false(self):
        df = get_resources_df(self.response_dict, custom_sort_columns=False)
        self.assertListEqual(list(df.columns), list(self.resources[0].keys()))

    def test_missing_title_column(self):
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

    def test_read_parquet_from_bytes_to_pyarrow_table_error(self):
        with self.assertRaises(ValueError) as cm:
            read_parquet_from_bytes_to_pyarrow_table(b"not a parquet file")
        self.assertIn("Failed to read Parquet", str(cm.exception))

        with patch("pyarrow.parquet.read_table", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                read_parquet_from_bytes_to_pyarrow_table(b"bytes")

    def test_coerce_real_date(self):
        table = self.table
        coerced = coerce_real_date(table)
        self.assertEqual(coerced.column("real_date").type, pa.date32())
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
                # the dates are switched on purpose to test if they are correctly swapped
                start_date="2023-02-03",
                end_date="2023-02-01",
                qdf=True,
            )
            df_filtered = filtered.to_pandas()
            self.assertEqual(len(df_filtered), 1)

        filtered = filter_parquet_table_as_qdf(self.table)
        filtered = filtered.to_pandas()
        expc = self.table.to_pandas()
        expc["real_date"] = pd.to_datetime(expc["real_date"])
        filtered["real_date"] = pd.to_datetime(filtered["real_date"])
        pd.testing.assert_frame_equal(filtered, expc)

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
            expc_msg = (
                "At least one of `include_catalog`, `include_notifications`, "
                "`include_full_datasets`, `include_explorer_datasets`, or "
                "`include_delta_datasets` must be True."
            )
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
        mock_resp = MagicMock()
        mock_resp.__enter__.return_value = mock_resp
        mock_resp.__exit__.return_value = False
        mock_resp.raise_for_status.return_value = None
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

                    if keep_raw_data:
                        self.assertTrue(os.path.exists(parquet_path))
                    else:
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
            qdf = pd.read_csv(expected_file)
            pd.testing.assert_frame_equal(qdf, df)

    def test_convert_ticker_based_parquet_file_to_qdf_error(self):
        with self.assertRaises(FileNotFoundError):
            convert_ticker_based_parquet_file_to_qdf("non_existant_file.parquet")


class TestJPMaQSFusionClientDownloadSeriesMemberDistributionToDisk(unittest.TestCase):
    def setUp(self):
        self.oauth = MagicMock(spec=FusionOAuth)
        self.simple_client = MagicMock(spec=SimpleFusionAPIClient)
        patcher = patch(
            "macrosynergy.download.fusion_interface.SimpleFusionAPIClient",
            return_value=self.simple_client,
        )
        self.addCleanup(patcher.stop)
        patcher.start()
        self.client = JPMaQSFusionClient(self.oauth)
        self.save_dir = "./tmp"

    @patch(
        "macrosynergy.download.fusion_interface.convert_ticker_based_parquet_file_to_qdf"
    )
    @patch("os.makedirs")
    @patch("os.path.exists")
    def test_successful_download_and_conversion(
        self, mock_exists, mock_makedirs, mock_convert
    ):
        mock_exists.return_value = True
        self.simple_client.get_seriesmember_distribution_details_to_disk.return_value = None

        with patch("builtins.print") as mock_print:
            self.client.download_series_member_distribution_to_disk(
                save_directory=self.save_dir,
                dataset="DS",
                seriesmember="SM",
                distribution="parquet",
                qdf=True,
                as_csv=True,
                keep_raw_data=False,
            )
            mock_makedirs.assert_called_once_with(self.save_dir, exist_ok=True)
            mock_exists.assert_called()
            mock_convert.assert_called_once()
            self.assertTrue(
                any(
                    "Successfully downloaded" in str(c[0][0])
                    for c in mock_print.call_args_list
                )
            )
            self.assertTrue(
                any(
                    "Successfully converted" in str(c[0][0])
                    for c in mock_print.call_args_list
                )
            )

    @patch(
        "macrosynergy.download.fusion_interface.convert_ticker_based_parquet_file_to_qdf"
    )
    @patch("os.makedirs")
    @patch("os.path.exists")
    def test_file_not_found_raises(self, mock_exists, mock_makedirs, mock_convert):
        mock_exists.return_value = False
        self.simple_client.get_seriesmember_distribution_details_to_disk.return_value = None
        with self.assertRaises(FileNotFoundError) as cm:
            self.client.download_series_member_distribution_to_disk(
                save_directory=self.save_dir,
                dataset="DS",
                seriesmember="SM",
            )
        self.assertIn(
            "Failed to download series member distribution", str(cm.exception)
        )
        mock_makedirs.assert_called_once()
        mock_convert.assert_not_called()

    @patch(
        "macrosynergy.download.fusion_interface.convert_ticker_based_parquet_file_to_qdf"
    )
    @patch("os.makedirs")
    @patch("os.path.exists")
    def test_no_qdf_prints_only_download(
        self, mock_exists, mock_makedirs, mock_convert
    ):
        mock_exists.return_value = True
        self.simple_client.get_seriesmember_distribution_details_to_disk.return_value = None
        with patch("builtins.print") as mock_print:
            self.client.download_series_member_distribution_to_disk(
                save_directory=self.save_dir,
                dataset="DS",
                seriesmember="SM",
                qdf=False,
            )
            prints = [str(c[0][0]) for c in mock_print.call_args_list]
            self.assertTrue(any("Successfully downloaded" in p for p in prints))
            self.assertFalse(any("Successfully converted" in p for p in prints))

    @patch(
        "macrosynergy.download.fusion_interface.convert_ticker_based_parquet_file_to_qdf"
    )
    @patch("os.makedirs")
    @patch("os.path.exists")
    def test_conversion_message_varies_with_as_csv(
        self, mock_exists, mock_makedirs, mock_convert
    ):
        mock_exists.return_value = True
        self.simple_client.get_seriesmember_distribution_details_to_disk.return_value = None
        with patch("builtins.print") as mock_print:
            self.client.download_series_member_distribution_to_disk(
                save_directory=self.save_dir,
                dataset="DS",
                seriesmember="SM",
                qdf=True,
                as_csv=False,
            )
            prints = [str(c[0][0]) for c in mock_print.call_args_list]
            self.assertTrue(
                any("Successfully converted" in p and "CSV" not in p for p in prints)
            )

        with patch("builtins.print") as mock_print:
            self.client.download_series_member_distribution_to_disk(
                save_directory=self.save_dir,
                dataset="DS",
                seriesmember="SM",
                qdf=True,
                as_csv=True,
            )
            prints = [str(c[0][0]) for c in mock_print.call_args_list]
            self.assertTrue(
                any("Successfully converted" in p and "CSV" in p for p in prints)
            )

    @patch.object(
        JPMaQSFusionClient,
        "get_latest_seriesmember_identifier",
        return_value="LATEST_SERIES",
    )
    @patch.object(JPMaQSFusionClient, "download_series_member_distribution_to_disk")
    def test_download_latest_distribution_to_disk(
        self, mock_download_to_disk, mock_get_latest
    ):
        client = self.client
        save_dir = self.save_dir
        dataset = "DS"
        client.download_latest_distribution_to_disk(
            save_directory=save_dir,
            dataset=dataset,
            distribution="parquet",
            qdf=True,
            as_csv=True,
            keep_raw_data=False,
        )
        mock_get_latest.assert_called_once_with(dataset=dataset)
        mock_download_to_disk.assert_called_once_with(
            save_directory=save_dir,
            dataset=dataset,
            seriesmember="LATEST_SERIES",
            distribution="parquet",
            qdf=True,
            as_csv=True,
            keep_raw_data=False,
        )


class TestJPMaQSFusionClientDownloadAndFilterSeriesMemberDistribution(
    unittest.TestCase
):
    def setUp(self):
        self.oauth = MagicMock(spec=FusionOAuth)
        self.simple_client = MagicMock(spec=SimpleFusionAPIClient)
        patcher = patch(
            "macrosynergy.download.fusion_interface.SimpleFusionAPIClient",
            return_value=self.simple_client,
        )
        self.addCleanup(patcher.stop)
        patcher.start()
        self.client = JPMaQSFusionClient(self.oauth)
        self.client._catalog = "CATALOG"

    @patch("macrosynergy.download.fusion_interface.filter_parquet_table_as_qdf")
    @patch(
        "macrosynergy.download.fusion_interface.read_parquet_from_bytes_to_pyarrow_table"
    )
    def test_download_and_filter_calls_all_steps(self, mock_read_parquet, mock_filter):
        fake_bytes = b"bytes"
        fake_table = MagicMock()
        filtered_table = MagicMock()
        filtered_df = MagicMock()
        filtered_table.to_pandas.return_value = filtered_df
        self.simple_client.get_seriesmember_distribution_details.return_value = (
            fake_bytes
        )
        mock_read_parquet.return_value = fake_table
        mock_filter.return_value = filtered_table

        result = self.client.download_and_filter_series_member_distribution(
            dataset="DS",
            seriesmember="SM",
            tickers=["A_B"],
            start_date="2020-01-01",
            end_date="2020-12-31",
            qdf=True,
        )

        self.simple_client.get_seriesmember_distribution_details.assert_called_once_with(
            catalog="CATALOG",
            dataset="DS",
            seriesmember="SM",
            distribution="parquet",
            as_bytes=True,
        )
        mock_read_parquet.assert_called_once_with(fake_bytes)
        mock_filter.assert_called_once_with(
            table=fake_table,
            tickers=["A_B"],
            start_date="2020-01-01",
            end_date="2020-12-31",
            qdf=True,
        )
        self.assertIs(result, filtered_df)

    @patch("macrosynergy.download.fusion_interface.filter_parquet_table_as_qdf")
    @patch(
        "macrosynergy.download.fusion_interface.read_parquet_from_bytes_to_pyarrow_table"
    )
    def test_download_and_filter_defaults(self, mock_read_parquet, mock_filter):
        fake_bytes = b"bytes"
        fake_table = MagicMock()
        filtered_table = MagicMock()
        filtered_df = MagicMock()
        filtered_table.to_pandas.return_value = filtered_df
        self.simple_client.get_seriesmember_distribution_details.return_value = (
            fake_bytes
        )
        mock_read_parquet.return_value = fake_table
        mock_filter.return_value = filtered_table

        result = self.client.download_and_filter_series_member_distribution(
            dataset="DS", seriesmember="SM"
        )
        mock_filter.assert_called_once()
        self.assertIs(result, filtered_df)

    @patch(
        "macrosynergy.download.fusion_interface.filter_parquet_table_as_qdf",
        side_effect=Exception("fail"),
    )
    @patch(
        "macrosynergy.download.fusion_interface.read_parquet_from_bytes_to_pyarrow_table"
    )
    def test_download_and_filter_raises(self, mock_read_parquet, mock_filter):
        fake_bytes = b"bytes"
        fake_table = MagicMock()
        self.simple_client.get_seriesmember_distribution_details.return_value = (
            fake_bytes
        )
        mock_read_parquet.return_value = fake_table
        with self.assertRaises(Exception) as cm:
            self.client.download_and_filter_series_member_distribution(
                dataset="DS", seriesmember="SM"
            )
        self.assertIn("fail", str(cm.exception))


class TestJPMaQSFusionClientDownloadMultipleDistributionsToDisk(unittest.TestCase):
    def setUp(self):
        self.patcher_makedirs = patch("os.makedirs")
        self.mock_makedirs = self.patcher_makedirs.start()

        self.patcher_timestamp = patch("pandas.Timestamp")
        self.mock_timestamp_class = self.patcher_timestamp.start()

        self.patcher_sleep = patch("time.sleep")
        self.mock_sleep = self.patcher_sleep.start()

        self.patcher_to_csv = patch.object(pd.DataFrame, "to_csv")
        self.mock_to_csv = self.patcher_to_csv.start()

        self.patcher_to_parquet = patch.object(pd.DataFrame, "to_parquet")
        self.mock_to_parquet = self.patcher_to_parquet.start()

        self.patcher_simple_client = patch(
            "macrosynergy.download.fusion_interface.SimpleFusionAPIClient"
        )
        self.mock_simple_client_cls = self.patcher_simple_client.start()
        self.mock_simple_client = MagicMock()
        self.mock_simple_client_cls.return_value = self.mock_simple_client

        fixed_ts = MagicMock()
        fixed_ts.strftime.return_value = "2025-07-23_12-00-00"
        self.mock_timestamp_class.utcnow.return_value = fixed_ts

        dummy_oauth = MagicMock()
        self.client = JPMaQSFusionClient(oauth_handler=dummy_oauth)

    def tearDown(self):
        self.patcher_makedirs.stop()
        self.patcher_timestamp.stop()
        self.patcher_sleep.stop()
        self.patcher_to_csv.stop()
        self.patcher_to_parquet.stop()
        self.patcher_simple_client.stop()

    def test_successful_download_csv(self):
        fake_catalog = MagicMock(spec=pd.DataFrame)
        self.client.get_metadata_catalog = MagicMock(return_value=fake_catalog)
        ds_df = pd.DataFrame({"identifier": ["ds1", "ds2"]})
        self.client.list_datasets = MagicMock(return_value=ds_df)
        self.client.download_latest_distribution_to_disk = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdirname:
            tmp_path = Path(tmpdirname)
            result = self.client._download_multiple_distributions_to_disk(
                folder=str(tmp_path),
                qdf=True,
                include_catalog=True,
                include_full_datasets=False,
                include_explorer_datasets=True,
                include_delta_datasets=False,
                as_csv=True,
                keep_raw_data=True,
                datasets_list=["ds1"],
            )

            expected_folder = tmp_path / "jpmaqs-download-2025-07-23_12-00-00"

            self.client.get_metadata_catalog.assert_called_once()
            fake_catalog.to_csv.assert_called_once()
            fake_catalog.to_parquet.assert_not_called()
            self.client.list_datasets.assert_called_once_with(
                include_catalog=True,
                include_full_datasets=False,
                include_explorer_datasets=True,
                include_delta_datasets=False,
            )
            self.assertEqual(
                self.client.download_latest_distribution_to_disk.call_count,
                2,
                "Expected download_latest_distribution_to_disk to be called twice",
            )
            expected_calls = [
                unittest.mock.call(
                    save_directory=expected_folder,
                    dataset="ds1",
                    qdf=True,
                    as_csv=True,
                    keep_raw_data=True,
                ),
                unittest.mock.call(
                    save_directory=expected_folder,
                    dataset="ds2",
                    qdf=True,
                    as_csv=True,
                    keep_raw_data=True,
                ),
            ]
            self.client.download_latest_distribution_to_disk.assert_has_calls(
                expected_calls, any_order=True
            )
            self.assertIs(result, fake_catalog)
            self.assertTrue(self.mock_sleep.called)
            self.mock_makedirs.assert_called_once_with(expected_folder, exist_ok=True)

    def test_failure_records_populated_on_exception(self):
        fake_catalog = MagicMock(spec=pd.DataFrame)
        self.client.get_metadata_catalog = MagicMock(return_value=fake_catalog)
        ds_df = pd.DataFrame({"identifier": ["ds1", "ds2"]})
        self.client.list_datasets = MagicMock(return_value=ds_df)

        def download_side_effect(save_directory, dataset, qdf, as_csv, keep_raw_data):
            if dataset == "ds2":
                raise Exception("network error")
            return None

        self.client.download_latest_distribution_to_disk = MagicMock(
            side_effect=download_side_effect
        )

        with tempfile.TemporaryDirectory() as tmpdirname:
            tmp_path = Path(tmpdirname)
            result = self.client._download_multiple_distributions_to_disk(
                folder=str(tmp_path),
                qdf=False,
                include_catalog=False,
                include_full_datasets=False,
                include_explorer_datasets=False,
                include_delta_datasets=False,
                as_csv=False,
                keep_raw_data=False,
                datasets_list=None,
            )

            self.assertEqual(len(self.client.failure_messages), 1)
            self.assertIn(
                "Failed to download dataset ds2: network error",
                self.client.failure_messages[0],
            )
            self.assertIs(result, fake_catalog)

    def test_folder_none_uses_cwd(self):
        fake_catalog = MagicMock(spec=pd.DataFrame)
        self.client.get_metadata_catalog = MagicMock(return_value=fake_catalog)
        ds_df = pd.DataFrame({"identifier": ["ds1"]})
        self.client.list_datasets = MagicMock(return_value=ds_df)
        self.client.download_latest_distribution_to_disk = MagicMock()
        with tempfile.TemporaryDirectory() as tmpdirname:
            tmp_path = Path(tmpdirname)
            with patch.object(Path, "cwd", return_value=tmp_path):
                result = self.client._download_multiple_distributions_to_disk(
                    folder=None,
                    datasets_list=["ds1"],
                )

            expected_folder = tmp_path / "jpmaqs-download-2025-07-23_12-00-00"
            self.mock_makedirs.assert_called_once_with(expected_folder, exist_ok=True)
            self.client.download_latest_distribution_to_disk.assert_called_once_with(
                save_directory=expected_folder,
                dataset="ds1",
                qdf=False,
                as_csv=False,
                keep_raw_data=False,
            )
            self.assertIs(result, fake_catalog)

    def test_datasets_list_no_match_raises(self):
        fake_catalog = MagicMock(spec=pd.DataFrame)
        self.client.get_metadata_catalog = MagicMock(return_value=fake_catalog)
        ds_df = pd.DataFrame({"identifier": ["alpha", "beta"]})
        self.client.list_datasets = MagicMock(return_value=ds_df)
        with tempfile.TemporaryDirectory() as tmpdirname:
            tmp_path = Path(tmpdirname)
            with self.assertRaises(ValueError) as cm:
                self.client._download_multiple_distributions_to_disk(
                    folder=str(tmp_path),
                    datasets_list=["gamma"],
                )
            msg = str(cm.exception)
            self.assertIn(
                "No datasets found in the provided `datasets_list`. Available datasets: alpha, beta",
                msg,
            )

    def test_download_latest_delta_distribution_calls_download_multiple(self):
        self.client._download_multiple_distributions_to_disk = MagicMock(
            return_value=pd.DataFrame()
        )
        result = self.client.download_latest_delta_distribution(
            folder="test_folder",
            qdf=True,
            as_csv=False,
            keep_raw_data=True,
            extra_param=123,
        )
        self.client._download_multiple_distributions_to_disk.assert_called_once_with(
            folder="test_folder",
            qdf=True,
            include_catalog=False,
            include_full_datasets=False,
            include_explorer_datasets=False,
            include_delta_datasets=True,
            as_csv=False,
            keep_raw_data=True,
        )
        self.assertIsInstance(result, pd.DataFrame)

    def test_download_latest_full_snapshot_calls_download_multiple(self):
        self.client._download_multiple_distributions_to_disk = MagicMock(
            return_value=pd.DataFrame()
        )
        result = self.client.download_latest_full_snapshot(
            folder="snapshot_folder",
            qdf=False,
            include_catalog=True,
            include_explorer_datasets=True,
            include_delta_datasets=True,
            as_csv=True,
            keep_raw_data=False,
            datasets_list=["dsA", "dsB"],
            another_arg="value",
        )
        self.client._download_multiple_distributions_to_disk.assert_called_once_with(
            folder="snapshot_folder",
            qdf=False,
            include_catalog=True,
            include_full_datasets=True,
            include_explorer_datasets=True,
            include_delta_datasets=True,
            as_csv=True,
            keep_raw_data=False,
            datasets_list=["dsA", "dsB"],
        )
        self.assertIsInstance(result, pd.DataFrame)


class TestJPMaQSFusionClientDownload(unittest.TestCase):
    def setUp(self):
        self.patcher_simple = patch(
            "macrosynergy.download.fusion_interface.SimpleFusionAPIClient"
        )
        self.mock_simple_cls = self.patcher_simple.start()
        self.mock_simple = MagicMock()
        self.mock_simple_cls.return_value = self.mock_simple

        dummy_oauth = FusionOAuth(client_id="dummy", client_secret="dummy")
        self.client = JPMaQSFusionClient(oauth_handler=dummy_oauth)

        self.catalog = pd.DataFrame({"Ticker": ["AAA", "BBB"], "Theme": ["one", "two"]})
        self.client.get_metadata_catalog = MagicMock(return_value=self.catalog)

        self.client.get_latest_seriesmember_identifier = MagicMock(
            side_effect=lambda dataset, **kw: f"sm_{dataset}"
        )

        def filter_side_effect(
            dataset, seriesmember, tickers, start_date, end_date, qdf, **kw
        ):
            df = pd.DataFrame({"value": [1, 2]})
            df["dataset"] = dataset
            return df

        self.client.download_and_filter_series_member_distribution = MagicMock(
            side_effect=filter_side_effect
        )

    def tearDown(self):
        self.patcher_simple.stop()

    def test_error_on_cids_xcats_xor(self):
        with self.assertRaises(ValueError):
            self.client.download(folder=None, cids=["X"], xcats=None)
        with self.assertRaises(ValueError):
            self.client.download(folder=None, cids=None, xcats=["Y"])

    def test_error_on_no_tickers(self):
        with self.assertRaises(ValueError):
            self.client.download(folder=None, tickers=None, cids=None, xcats=None)

    @patch("pathlib.Path.cwd", return_value=Path("./tmp"))
    def test_save_to_folder_calls_full_snapshot(self, mock_cwd):
        self.client.download_latest_full_snapshot = MagicMock(
            return_value=pd.DataFrame()
        )
        df = self.client.download(
            folder="myfolder",
            tickers=["AAA"],
            qdf=False,
            as_csv=True,
            keep_raw_data=False,
        )
        self.client.download_latest_full_snapshot.assert_called_once_with(
            folder=Path("myfolder"),
            qdf=False,
            include_catalog=True,
            include_full_datasets=True,
            as_csv=True,
            keep_raw_data=False,
            datasets_list=["JPMAQS_ONE"],
        )
        self.assertIsInstance(df, pd.DataFrame)

    @patch.object(pd.Timestamp, "utcnow")
    def test_no_results_raises(self, mock_utcnow):
        mock_utcnow.return_value = pd.Timestamp("2025-07-23")
        self.client.download_and_filter_series_member_distribution = MagicMock(
            return_value=pd.DataFrame()
        )
        with self.assertRaises(ValueError):
            with warnings.catch_warnings(record=True):
                self.client.download(tickers=["ABCDEFGH"])

    @patch("time.sleep", lambda *args, **kwargs: None)
    def test_download_threadpool_success(self):
        df = self.client.download(
            folder=None,
            tickers=["AAA", "BBB"],
            start_date="2025-01-01",
            end_date="2025-06-01",
            qdf=False,
            as_csv=False,
        )
        self.assertEqual(
            self.client.download_and_filter_series_member_distribution.call_count, 2
        )
        self.assertEqual(len(df), 4)
        unique_datasets = set(df["dataset"].tolist())
        expected = {"JPMAQS_ONE", "JPMAQS_TWO"}
        self.assertEqual(unique_datasets, expected)

    @patch("time.sleep", lambda *args, **kwargs: None)
    def test_download_with_cids_xcats(self):
        custom_catalog = pd.DataFrame({"Ticker": ["X_Y"], "Theme": ["special theme"]})
        self.client.get_metadata_catalog.return_value = custom_catalog

        df = self.client.download(
            folder=None,
            cids=["X"],
            xcats=["Y"],
            start_date="2025-02-01",
            end_date="2025-02-15",
        )
        self.assertEqual(
            self.client.download_and_filter_series_member_distribution.call_count, 1
        )
        self.assertTrue((df["dataset"] == "JPMAQS_SPECIAL_THEME").all())

    @patch("time.sleep", lambda *args, **kwargs: None)
    def test_start_end_dates_swapped(self):
        captured = []

        def capture_side_effect(
            dataset, seriesmember, tickers, start_date, end_date, qdf, **kw
        ):
            captured.append((start_date, end_date))
            return pd.DataFrame({"value": [1], "dataset": [dataset]})

        self.client.download_and_filter_series_member_distribution.side_effect = (
            capture_side_effect
        )

        self.client.download(
            folder=None, tickers=["AAA"], start_date="2025-06-01", end_date="2025-01-01"
        )
        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0], ("2025-01-01", "2025-06-01"))

    @patch("time.sleep", lambda *args, **kwargs: None)
    def test_warning_for_non_existing_tickers(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            df = self.client.download(
                folder=None,
                tickers=["AAA", "ZZZ"],
                start_date="2025-03-01",
                end_date="2025-03-10",
            )
            self.assertEqual(len(w), 1)
            self.assertIn(
                "There are 1 tickers that do not exist in the metadata catalog",
                str(w[0].message),
            )
        self.assertTrue((df["dataset"] == "JPMAQS_ONE").all())
        self.assertEqual(len(df), 2)


if __name__ == "__main__":
    unittest.main()
