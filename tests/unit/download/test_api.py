from unittest import mock
from random import random
import unittest
import numpy as np
import pandas as pd
import os
import datetime
import base64
import io
import copy

from typing import List, Dict, Union, Optional, Any, Tuple
import requests

from macrosynergy.download import dataquery
from macrosynergy.download import JPMaQSDownload
from macrosynergy.download.dataquery import (
    DataQueryInterface,
    OAuth,
    CertAuth,
    request_wrapper,
    validate_response,
    validate_download_args,
)
from macrosynergy.download.dataquery import (
    OAUTH_BASE_URL,
    OAUTH_TOKEN_URL,
    HEARTBEAT_ENDPOINT,
    TIMESERIES_ENDPOINT,
    API_DELAY_PARAM,
    HL_RETRY_COUNT,
    CERT_BASE_URL,
)
from macrosynergy.download.exceptions import (
    AuthenticationError,
    HeartbeatError,
    InvalidResponseError,
    DownloadError,
    InvalidDataframeError,
)
from macrosynergy.management.utils import Config


class TestCertAuth(unittest.TestCase):
    def mock_isfile(self, path: str) -> bool:
        good_paths: List[str] = ["path/key.key", "path/crt.crt"]
        if path in good_paths:
            return True

    def good_args(self) -> Dict[str, str]:
        return {
            "username": "user",
            "password": "pass",
            "crt": "path/crt.crt",
            "key": "path/key.key",
        }

    def test_init(self):
        try:
            with mock.patch(
                "os.path.isfile", side_effect=lambda x: self.mock_isfile(x)
            ):
                certauth: CertAuth = CertAuth(**self.good_args())

                expctd_auth: str = base64.b64encode(
                    bytes(
                        f"{self.good_args()['username']}:{self.good_args()['password']}",
                        "utf-8",
                    )
                ).decode("ascii")
                self.assertEqual(certauth.auth, expctd_auth)
                self.assertEqual(certauth.crt, self.good_args()["crt"])
                self.assertEqual(certauth.key, self.good_args()["key"])

        except Exception as e:
            self.fail(f"Unexpected exception raised: {e}")

        with mock.patch("os.path.isfile", side_effect=lambda x: self.mock_isfile(x)):
            for key in self.good_args().keys():
                bad_args: Dict[str, str] = self.good_args().copy()
                bad_args[key] = 1
                with self.assertRaises(TypeError):
                    CertAuth(**bad_args)

        with mock.patch("os.path.isfile", side_effect=lambda x: self.mock_isfile(x)):
            for key in ["crt", "key"]:
                bad_args: Dict[str, str] = self.good_args().copy()
                bad_args[key] = "path/invalid_path"
                with self.assertRaises(FileNotFoundError):
                    CertAuth(**bad_args)

    def test_get_auth(self):
        with mock.patch("os.path.isfile", side_effect=lambda x: self.mock_isfile(x)):
            certauth: CertAuth = CertAuth(**self.good_args())

            expctd_auth: str = base64.b64encode(
                bytes(
                    f"{self.good_args()['username']}"
                    f":{self.good_args()['password']}",
                    "utf-8",
                )
            ).decode("ascii")
            self.assertEqual(certauth.auth, expctd_auth)
            self.assertEqual(certauth.crt, self.good_args()["crt"])
            self.assertEqual(certauth.key, self.good_args()["key"])

            authx: Dict[str, Dict[str, Any]] = certauth.get_auth()
            self.assertEqual(authx["headers"]["Authorization"], f"Basic {expctd_auth}")
            self.assertEqual(
                authx["cert"], (self.good_args()["crt"], self.good_args()["key"])
            )

    def test_with_dqinterface(self):
        with mock.patch("os.path.isfile", side_effect=lambda x: self.mock_isfile(x)):
            cfg: Config = Config(
                username="user", password="pass", crt="path/crt.crt", key="path/key.key"
            )
            dq_interface: DataQueryInterface = DataQueryInterface(
                config=cfg, oauth=False
            )

            # assert that dq_interface.auth is CertAuth type
            self.assertIsInstance(dq_interface.auth, CertAuth)
            # check that the base_url is cert_base url
            self.assertEqual(dq_interface.base_url, CERT_BASE_URL)


class TestOAuth(unittest.TestCase):
    def test_init(self):
        jpmaqs: JPMaQSDownload = JPMaQSDownload(
            oauth=True,
            client_id="test-id",
            client_secret="SECRET",
            check_connection=False,
        )
        self.assertEqual(jpmaqs.dq_interface.base_url, dataquery.OAUTH_BASE_URL)

    def test_invalid_init_args(self):
        good_args: Dict[str, str] = {
            "client_id": "test-id",
            "client_secret": "SECRET",
            "token_url": "https://token.url",
            "dq_resource_id": "test-resource-id",
        }

        for key in good_args.keys():
            bad_args: Dict[str, str] = good_args.copy()
            bad_args[key] = 1
            with self.assertRaises(TypeError):
                OAuth(**bad_args)

        try:
            oath_obj: OAuth = OAuth(**good_args)

            self.assertIsInstance(oath_obj.token_data, dict)
            expcted_token_data: Dict[str, str] = {
                "grant_type": "client_credentials",
                "client_id": "test-id",
                "client_secret": "SECRET",
                "aud": "test-resource-id",
            }
            self.assertEqual(oath_obj.token_data, expcted_token_data)
        except Exception as e:
            self.fail(f"Unexpected exception raised: {e}")

    def test_valid_token(self):
        oauth = dataquery.OAuth(client_id="test-id", client_secret="SECRET")
        self.assertFalse(oauth._valid_token())


class TestDataQueryInterface(unittest.TestCase):
    @staticmethod
    def jpmaqs_value(elem: str):
        """
        Used to produce a value or grade for the associated ticker. If the metric is
        grade, the function will return 1.0 and if value, the function returns a random
        number between (0, 1).

        :param <str> elem: ticker.
        """
        ticker_split = elem.split(",")
        if ticker_split[-1][:-1] == "grading":
            value = 1.0
        else:
            value = random()
        return value

    def request_wrapper(
        self, dq_expressions: List[str], start_date: str, end_date: str
    ):
        """
        Contrived request method to replicate output from DataQuery. Will replicate the
        form of a JPMaQS expression from DataQuery which will subsequently be used to
        test methods held in the api.Interface() Class.
        """
        aggregator = []
        for i, elem in enumerate(dq_expressions):
            elem_dict = {
                "item": (i + 1),
                "group": None,
                "attributes": [
                    {
                        "expression": elem,
                        "label": None,
                        "attribute-id": None,
                        "attribute-name": None,
                        "time-series": [
                            [d.strftime("%Y%m%d"), self.jpmaqs_value(elem)]
                            for d in pd.bdate_range(start_date, end_date)
                        ],
                    },
                ],
                "instrument-id": None,
                "instrument-name": None,
            }
            aggregator.append(elem_dict)

        return aggregator

    def test_init(self):
        with self.assertRaises(ValueError):
            DataQueryInterface(config=1)

    @mock.patch(
        "macrosynergy.download.dataquery.OAuth._get_token",
        return_value=("SOME_TEST_TOKEN"),
    )
    @mock.patch(
        "macrosynergy.download.dataquery.request_wrapper",
        return_value=({"info": {"code": 200, "message": "Service Available."}}),
    )
    def test_check_connection(self, mock_p_request, mock_p_get_token):
        # If the connection to DataQuery is working, the response code will invariably be
        # 200. Therefore, use the Interface Object's method to check DataQuery
        # connections.

        with JPMaQSDownload(
            client_id="client1", client_secret="123", oauth=True, check_connection=False
        ) as jpmaqs:
            self.assertTrue(jpmaqs.check_connection())

        mock_p_request.assert_called_once()
        mock_p_get_token.assert_called_once()

    @mock.patch(
        "macrosynergy.download.dataquery.OAuth._get_token",
        return_value=("SOME_TEST_TOKEN"),
    )
    @mock.patch(
        "macrosynergy.download.dataquery.request_wrapper",
        return_value=({"info": {"code": 200, "message": "Service Available."}}),
    )
    def test_check_connection_on_init(self, mock_p_request, mock_p_get_token):
        # If the connection to DataQuery is working, the response code will invariably be
        # 200. Therefore, use the Interface Object's method to check DataQuery
        # connections.

        with JPMaQSDownload(
            client_id="client1",
            client_secret="123",
            oauth=True,
        ) as jpmaqs:
            pass

        mock_p_request.assert_called_once()
        mock_p_get_token.assert_called_once()

    @mock.patch(
        "macrosynergy.download.dataquery.OAuth._get_token",
        return_value=("SOME_TEST_TOKEN"),
    )
    @mock.patch(
        "macrosynergy.download.dataquery.request_wrapper",
        return_value=(
            {"info": {"code": 400}},
            False,
            {
                "headers": "{'Content-Type': 'application/json'}",
                "status_code": 400,
                "text": "{'error': 'invalid_request', 'error_description': 'The request is somehow corrupt.'}",
                "url": "https://api-developer.jpmorgan.com/research/dataquery-authe/api/v2/ **SOMETHING**",
            },
        ),
    )
    def test_check_connection_fail(self, mock_p_fail, mock_p_get_token):
        # Opposite of above method: if the connection to DataQuery fails, the error code
        # will be 400.

        with JPMaQSDownload(
            client_id="client1", client_secret="123", oauth=True, check_connection=False
        ) as jpmaqs_download:
            # Method returns a Boolean. In this instance, the method should return False
            # (unable to connect).
            self.assertFalse(jpmaqs_download.check_connection())
        mock_p_fail.assert_called_once()
        mock_p_get_token.assert_called_once()

    def test_oauth_condition(self):
        # Accessing DataQuery can be achieved via two methods: OAuth or Certificates /
        # Keys. To handle for the idiosyncrasies of the two access methods, split the
        # methods across individual Classes. The usage of each Class is controlled by the
        # parameter "oauth".
        # First check is that the DataQuery instance is using an OAuth Object if the
        # parameter "oauth" is set to to True.
        jpmaqs_download = JPMaQSDownload(
            oauth=True, client_id="client1", client_secret="123", check_connection=False
        )

        self.assertIsInstance(
            jpmaqs_download.dq_interface, dataquery.DataQueryInterface
        )
        self.assertIsInstance(jpmaqs_download.dq_interface.auth, dataquery.OAuth)

    def test_certauth_condition(self):
        # Second check is that the DataQuery instance is using an CertAuth Object if the
        # parameter "oauth" is set to to False. The DataQuery Class's default is to use
        # certificate / keys.

        # Given the certificate and key will not point to valid directories, the expected
        # behaviour is for an OSError to be thrown.
        with self.assertRaises(FileNotFoundError):
            with JPMaQSDownload(
                username="user1",
                password="123",
                crt="/api_macrosynergy_com.crt",
                key="/api_macrosynergy_com.key",
                oauth=False,
                check_connection=False,
            ) as downloader:
                pass

    def test_timeseries_to_df(self):
        cids_dmca = ["AUD", "CAD", "CHF", "EUR", "GBP", "JPY"]
        xcats = ["EQXR_NSA", "FXXR_NSA"]

        tickers = [cid + "_" + xcat for xcat in xcats for cid in cids_dmca]

        jpmaqs_download = JPMaQSDownload(
            oauth=True,
            client_id="client_id",
            client_secret="client_secret",
            check_connection=False,
        )

        # First replicate the api.Interface()._request() method using the associated
        # JPMaQS expression.
        expression = jpmaqs_download.construct_expressions(
            metrics=["value", "grading"], tickers=tickers
        )
        start_date: str = "2000-01-01"
        end_date: str = "2020-01-01"

        timeseries_output = self.request_wrapper(
            dq_expressions=expression, start_date=start_date, end_date=end_date
        )

        expressions_found: List[str] = [
            ts["attributes"][0]["expression"] for ts in timeseries_output
        ]

        out_df: pd.DataFrame = jpmaqs_download.time_series_to_df(
            dicts_list=timeseries_output,
            expected_expressions=expressions_found,
            start_date=start_date,
            end_date=end_date,
        )

        # Check that the output is a Pandas DataFrame
        self.assertIsInstance(out_df, pd.DataFrame)

        # Check that the output has the correct number of rows and columns
        # len(tickers)*len(pd.bdate_range(start_date, end_date)) = expected number of rows
        # expected cols = [["real_date", "cid", "xcat", "value", "grading"]] = 5
        self.assertEqual(
            out_df.shape, (len(tickers) * len(pd.bdate_range(start_date, end_date)), 5)
        )

        # Check that the output has the correct columns
        self.assertEqual(
            set(out_df.columns.tolist()),
            set(["real_date", "cid", "xcat", "value", "grading"]),
        )

    def test_construct_expressions(self):
        jpmaqs_download = JPMaQSDownload(
            oauth=True,
            client_id="client_id",
            client_secret="client_secret",
            check_connection=False,
        )

        cids = ["AUD", "CAD", "CHF", "EUR", "GBP", "JPY"]
        xcats = ["EQXR_NSA", "FXXR_NSA"]

        tickers = [cid + "_" + xcat for xcat in xcats for cid in cids]

        metrics = ["value", "grading"]

        set_a = jpmaqs_download.construct_expressions(metrics=metrics, tickers=tickers)

        set_b = jpmaqs_download.construct_expressions(
            metrics=metrics, cids=cids, xcats=xcats
        )

        self.assertEqual(set(set_a), set(set_b))

    def test_deconstruct_expressions(self):
        jpmaqs_download = JPMaQSDownload(
            oauth=True,
            client_id="client_id",
            client_secret="client_secret",
            check_connection=False,
        )

        cids = ["AUD", "CAD", "CHF", "EUR", "GBP", "JPY"]
        xcats = ["EQXR_NSA", "FXXR_NSA"]
        tickers = [cid + "_" + xcat for xcat in xcats for cid in cids]
        metrics = ["value", "grading"]
        tkms = [f"{ticker}_{metric}" for ticker in tickers for metric in metrics]
        expressions = jpmaqs_download.construct_expressions(
            metrics=["value", "grading"], tickers=tickers
        )
        deconstructed_expressions = jpmaqs_download.deconstruct_expression(
            expression=expressions
        )
        dtkms = ["_".join(d) for d in deconstructed_expressions]

        self.assertEqual(set(tkms), set(dtkms))

        for tkm, expression in zip(tkms, expressions):
            self.assertEqual(
                tkm,
                "_".join(jpmaqs_download.deconstruct_expression(expression=expression)),
            )

        for expr in [1, [1, 2]]:
            # type error
            with self.assertRaises(TypeError):
                jpmaqs_download.deconstruct_expression(expression=expr)

        with self.assertRaises(ValueError):
            jpmaqs_download.deconstruct_expression(expression=[])

    def test_get_catalogue(self):
        dq: DataQueryInterface = DataQueryInterface(
            oauth=True,
            config=Config(client_id="client_id", client_secret="client_secret"),
        )
        # assert raises NotImplementedError
        with self.assertRaises(NotImplementedError):
            dq.get_catalogue()

        with self.assertRaises(NotImplementedError):
            dq.filter_exprs_from_catalogue(expressions=["expression1", "expression2"])

    def test_dq_fetch(self):
        cfg = Config(
            client_id=os.getenv("DQ_CLIENT_ID"),
            client_secret=os.getenv("DQ_CLIENT_SECRET"),
        )
        dq: DataQueryInterface = DataQueryInterface(oauth=True, config=cfg)

        self.assertTrue(
            dq.check_connection()
        )  # doing this so oath token is ready for _fetch method
        invl_responses: List[Any] = [
            None,
            {},
            {"attributes": []},
            {"attributes": [{"expression": "expression1"}]},
        ]

        for invl_response in invl_responses:
            with mock.patch(
                "macrosynergy.download.dataquery.request_wrapper",
                return_value=invl_response,
            ):
                with self.assertRaises(InvalidResponseError):
                    dq._fetch(
                        url=OAUTH_BASE_URL + TIMESERIES_ENDPOINT,
                        params={"expr": "expression1"},
                    )

    def test_download(self):
        good_args: Dict[str, Any] = {
            "expressions": ["expression1", "expression2"],
            "params": {"start_date": "2000-01-01", "end_date": "2020-01-01"},
            "url": OAUTH_BASE_URL + TIMESERIES_ENDPOINT,
            "tracking_id": str,
            "delay_param": 0.25,
            "retry_counter": 0,
        }

        bad_args: Dict[str, Any] = good_args.copy()
        bad_args["retry_counter"] = 10

        with mock.patch("sys.stdout", new=io.StringIO()) as mock_std:
            with mock.patch(
                "macrosynergy.download.dataquery.request_wrapper",
                return_value={"attributes": []},
            ):
                with self.assertRaises(DownloadError):
                    DataQueryInterface(
                        Config(client_id="client_id", client_secret="client_secret"),
                        oauth=True,
                    )._download(**bad_args)
            err_string_1: str = (
                f"Retrying failed downloads. Retry count: {bad_args['retry_counter']}"
            )
            self.assertIn(err_string_1, mock_std.getvalue())

    def test_dq_download_args(self):
        good_args: Dict[str, Any] = {
            "expressions": ["DB(JPMAQS,EUR_FXXR_NSA,value)"],
            "start_date": "2020-01-01",
            "end_date": "2020-02-01",
            "show_progress": True,
            "endpoint": HEARTBEAT_ENDPOINT,
            "calender": "CAL_ALLDAYS",
            "frequency": "FREQ_DAY",
            "conversion": "CONV_LASTBUS_ABS",
            "nan_treatment": "NA_NOTHING",
            "reference_data": "NO_REFERENCE_DATA",
            "retry_counter": 0,
            "delay_param": API_DELAY_PARAM,
        }
        self.assertTrue(validate_download_args(**good_args))

        # rplace expressions with None. should raise value error
        bad_args: Dict[str, Any] = good_args.copy()
        bad_args["expressions"] = None
        with self.assertRaises(ValueError):
            validate_download_args(**bad_args)

        # replace expressions with list of ints. should raise type error
        bad_args: Dict[str, Any] = good_args.copy()
        bad_args["expressions"] = [1, 2, 3]
        with self.assertRaises(TypeError):
            validate_download_args(**bad_args)

        for key in good_args.keys():
            bad_value: Union[int, str] = 1
            if key == "retry_counter":
                bad_value = "1"
            bad_args: Dict[str, Any] = good_args.copy()
            bad_args[key] = bad_value
            with self.assertRaises(TypeError):
                validate_download_args(**bad_args)

        for delay_param in [0.1, -1.0]:
            bad_args: Dict[str, Any] = good_args.copy()
            bad_args["delay_param"] = delay_param
            with self.assertRaises(ValueError):
                validate_download_args(**bad_args)

        for date_arg in ["start_date", "end_date"]:
            bad_args: Dict[str, Any] = good_args.copy()
            bad_args[date_arg] = "1-Jan-2023"
            with self.assertRaises(ValueError):
                validate_download_args(**bad_args)


class TestDataQueryDownloads(unittest.TestCase):
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

        test_expr: str = JPMaQSDownload.construct_expressions(
            cids=cids, xcats=xcats, metrics=["value", "grading"]
        )
        with DataQueryInterface(
            oauth=True,
            config=Config(
                client_id=os.getenv("DQ_CLIENT_ID"),
                client_secret=os.getenv("DQ_CLIENT_SECRET"),
            ),
        ) as dq:
            data: List[Dict[str, Any]] = dq.download_data(
                expressions=test_expr,
                start_date=start_date,
                end_date=end_date,
            )

        self.assertIsInstance(data, list)
        self.assertGreater(len(data), 0)
        for _data in data:
            self.assertIsInstance(_data, dict)


##############################################


class TestRequestWrapper(unittest.TestCase):
    def mock_response(
        self,
        url: str,
        status_code: int = 200,
        headers: Dict[str, str] = None,
        text: str = None,
        content: bytes = None,
    ) -> requests.Response:
        mock_resp: requests.Response = requests.Response()
        mock_resp.status_code = status_code
        mock_resp.headers = headers or {}
        mock_resp._content = content
        mock_resp._text = text
        mock_resp.request = requests.Request("GET", url)
        return mock_resp

    def test_validate_response(self):
        # mock a response with 401. assert raises authentication error
        with self.assertRaises(AuthenticationError):
            validate_response(self.mock_response(url=OAUTH_TOKEN_URL, status_code=401))

        # mock with a 403, and use url=oauth+heartbeat. assert raises heartbeat error
        with self.assertRaises(HeartbeatError):
            validate_response(
                self.mock_response(
                    url=OAUTH_BASE_URL + HEARTBEAT_ENDPOINT, status_code=403
                )
            )

        # mock with a 403, and use url=oauth_base_url. assert raises invalid response error
        with self.assertRaises(InvalidResponseError):
            validate_response(self.mock_response(url=OAUTH_BASE_URL, status_code=403))

        # oauth_bas_url+timeseires and empty content, assert raises invalid response error
        with self.assertRaises(InvalidResponseError):
            validate_response(
                self.mock_response(
                    url=OAUTH_BASE_URL + TIMESERIES_ENDPOINT,
                    status_code=200,
                    content=b"",
                )
            )

        # with 200 , and empty content, assert raises invalid response error
        with self.assertRaises(InvalidResponseError):
            validate_response(
                self.mock_response(url=OAUTH_TOKEN_URL, status_code=200, content=b"")
            )

        # with non-json content, assert raises invalid response error
        with self.assertRaises(InvalidResponseError):
            validate_response(
                self.mock_response(
                    url=OAUTH_TOKEN_URL, status_code=200, content=b"not json"
                )
            )

    def test_request_wrapper(self):
        with self.assertRaises(ValueError):
            request_wrapper(method="pop", url=OAUTH_TOKEN_URL)

        def mock_auth_error(*args, **kwargs) -> requests.Response:
            return self.mock_response(url=OAUTH_TOKEN_URL, status_code=401)

        with mock.patch("requests.Session.send", side_effect=mock_auth_error):
            with self.assertRaises(AuthenticationError):
                request_wrapper(
                    method="get",
                    url=OAUTH_TOKEN_URL,
                )

        def mock_heartbeat_error(*args, **kwargs) -> requests.Response:
            # mock a response with 403. assert raises heartbeat error
            return self.mock_response(
                url=OAUTH_BASE_URL + HEARTBEAT_ENDPOINT, status_code=403
            )

        with mock.patch("requests.Session.send", side_effect=mock_heartbeat_error):
            with self.assertRaises(HeartbeatError):
                request_wrapper(
                    method="get",
                    url=OAUTH_BASE_URL + HEARTBEAT_ENDPOINT,
                )

        def mock_known_errors(*args, **kwargs) -> requests.Response:
            # mock a response with 400. assert raises invalid response error
            raise ConnectionResetError

        with mock.patch("requests.Session.send", side_effect=mock_known_errors):
            with self.assertRaises(DownloadError):
                request_wrapper(
                    method="get",
                    url=OAUTH_BASE_URL + HEARTBEAT_ENDPOINT,
                )

        def mock_unknown_errors(*args, **kwargs) -> requests.Response:
            # raise some unrelated error -
            # using InvalidDataframeError as it does not interact with this scope
            raise InvalidDataframeError

        with mock.patch("requests.Session.send", side_effect=mock_unknown_errors):
            with self.assertRaises(InvalidDataframeError):
                request_wrapper(
                    method="get",
                    url=OAUTH_BASE_URL + HEARTBEAT_ENDPOINT,
                )

        def mock_keyboard_interrupt(*args, **kwargs) -> requests.Response:
            raise KeyboardInterrupt

        with mock.patch("requests.Session.send", side_effect=mock_keyboard_interrupt):
            with self.assertRaises(KeyboardInterrupt):
                request_wrapper(
                    method="get",
                    url=OAUTH_BASE_URL + HEARTBEAT_ENDPOINT,
                )


##############################################


class MockDataQueryInterface(DataQueryInterface):
    @staticmethod
    def jpmaqs_value(elem: str):
        """
        Used to produce a value or grade for the associated ticker. If the metric is
        grade, the function will return 1.0 and if value, the function returns a random
        number between (0, 1).

        :param <str> elem: ticker.
        """
        ticker_split = elem.split(",")
        if ticker_split[-1][:-1] == "grading":
            value = 1.0
        else:
            value = random()
        return value

    def request_wrapper(
        self, dq_expressions: List[str], start_date: str, end_date: str
    ):
        """
        Contrived request method to replicate output from DataQuery. Will replicate the
        form of a JPMaQS expression from DataQuery which will subsequently be used to
        test methods held in the api.Interface() Class.
        """
        aggregator = []
        for i, elem in enumerate(dq_expressions):
            elem_dict = {
                "item": (i + 1),
                "group": None,
                "attributes": [
                    {
                        "expression": elem,
                        "label": None,
                        "attribute-id": None,
                        "attribute-name": None,
                        "time-series": [
                            [d.strftime("%Y%m%d"), self.jpmaqs_value(elem)]
                            for d in pd.bdate_range(start_date, end_date)
                        ],
                    },
                ],
                "instrument-id": None,
                "instrument-name": None,
            }
            aggregator.append(elem_dict)

        return aggregator

    def __init__(self, *args, **kwargs):
        if "config" in kwargs:
            self.config = kwargs["config"]
        else:
            self.config = Config(
                client_id="test_clid",
                client_secret="test_clsc",
                crt="test_crt",
                key="test_key",
                username="test_user",
                password="test_pass",
            )

        self.mask_expressions = []
        super().__init__(config=self.config)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def check_connection(self, *args, **kwargs) -> bool:
        return True

    def download_data(
        self, expressions: List[str], start_date: str, end_date: str, **kwargs
    ) -> pd.DataFrame:
        ts: List[dict] = self.request_wrapper(expressions, start_date, end_date)
        if self.mask_expressions:
            for d in ts:
                if d["attributes"][0]["expression"] in self.mask_expressions:
                    d["attributes"][0]["time-series"] = None
                    d["attributes"][0][
                        "message"
                    ] = f"MASKED - {d['attributes'][0]['expression']}"

        return ts

    def _gen_attributes(
        self,
        msg_errors: List[str] = None,
        mask_expressions: List[str] = None,
        msg_warnings: List[str] = None,
        unavailable_expressions: List[str] = None,
        egress_data: List[Dict[str, Any]] = None,
    ):
        self.msg_errors: List[str] = (
            ["DEFAULT ERROR MESSAGE"] if msg_errors is None else msg_errors
        )
        self.msg_warnings: List[str] = (
            ["DEFAULT WARNING MESSAGE"] if msg_warnings is None else msg_warnings
        )
        self.unavailable_expressions: List[str] = (
            ["Some_Expression"]
            if unavailable_expressions is None
            else unavailable_expressions
        )
        self.egress_data: List[Dict[str, Any]] = {
            "tracking-id-123": {
                "upload_size": 200,
                "download_size": 2000,
                "url": OAUTH_BASE_URL + TIMESERIES_ENDPOINT,
                "time_taken": 10,
            }
        }

        self.mask_expressions: List[str] = (
            ["Some_Expression_X"] if mask_expressions is None else mask_expressions
        )


class TestJPMaQSDownload(unittest.TestCase):
    def test_init(self):
        good_args: Dict[str, Any] = {
            "oauth": True,
            "client_id": "client_id",
            "client_secret": "client_secret",
            "check_connection": False,
            "proxy": {"http": "proxy.com"},
            "suppress_warning": True,
            "debug": False,
            "print_debug_data": False,
            "dq_download_kwargs": {"test": "test"},
        }

        try:
            jpmaqs: JPMaQSDownload = JPMaQSDownload(**good_args)
            self.assertEqual(
                set(jpmaqs.valid_metrics),
                set(["value", "grading", "eop_lag", "mop_lag"]),
            )
            for varx in [
                jpmaqs.msg_errors,
                jpmaqs.msg_warnings,
                jpmaqs.unavailable_expressions,
            ]:
                self.assertEqual(varx, [])

            self.assertEqual(jpmaqs.downloaded_data, {})
        except Exception as e:
            self.fail("Unexpected exception raised: {}".format(e))

        for argx in good_args:
            with self.assertRaises(TypeError):
                bad_args = good_args.copy()
                bad_args[argx] = -1  # 1 would evaluate to True for bools
                JPMaQSDownload(**bad_args)

        with self.assertRaises(TypeError):
            bad_args = good_args.copy()
            bad_args["oauth"] = "test"
            JPMaQSDownload(**bad_args)

        with self.assertRaises(TypeError):
            good_args["credentials_config"] = 1
            JPMaQSDownload(**good_args)

    def test_download_arg_validation(self):
        good_args: Dict[str, Any] = {
            "tickers": ["EUR_FXXR_NSA", "USD_FXXR_NSA"],
            "cids": ["GBP", "EUR"],
            "xcats": ["FXXR_NSA", "EQXR_NSA"],
            "metrics": ["value", "grading"],
            "start_date": "2019-01-01",
            "end_date": "2019-01-31",
            "expressions": [
                "DB(JPMAQS,AUD_FXXR_NSA,value)",
                "DB(JPMAQS,CAD_FXXR_NSA,value)",
            ],
            "show_progress": True,
            "as_dataframe": True,
            "report_time_taken": True,
            "report_egress": True,
        }
        bad_args: Dict[str, Any] = {}
        jpmaqs: JPMaQSDownload = JPMaQSDownload(
            client_id="client_id",
            client_secret="client_secret",
            check_connection=False,
        )

        try:
            if not jpmaqs.validate_download_args(**good_args):
                self.fail("Unexpected validation failure")
        except Exception as e:
            self.fail("Unexpected exception raised: {}".format(e))

        for argx in good_args:
            with self.assertRaises(TypeError):
                bad_args = good_args.copy()
                bad_args[argx] = -1
                jpmaqs.validate_download_args(**bad_args)

        # value error for tickers==cids==xcats==expressions == None
        with self.assertRaises(ValueError):
            bad_args = good_args.copy()
            bad_args["tickers"] = None
            bad_args["cids"] = None
            bad_args["xcats"] = None
            bad_args["expressions"] = None
            jpmaqs.validate_download_args(**bad_args)

        for lvarx in ["tickers", "cids", "xcats", "expressions"]:
            with self.assertRaises(ValueError):
                bad_args = good_args.copy()
                bad_args[lvarx] = []
                jpmaqs.validate_download_args(**bad_args)

            with self.assertRaises(TypeError):
                bad_args = good_args.copy()
                bad_args[lvarx] = [1]
                jpmaqs.validate_download_args(**bad_args)

        # test cases for metrics arg
        bad_args = good_args.copy()
        bad_args["metrics"] = None
        with self.assertRaises(ValueError):
            jpmaqs.validate_download_args(**bad_args)

        bad_args = good_args.copy()
        bad_args["metrics"] = ["Metallica"]
        with self.assertRaises(ValueError):
            jpmaqs.validate_download_args(**bad_args)

        # cid AND xcat cases
        with self.assertRaises(ValueError):
            bad_args = good_args.copy()
            bad_args["xcats"] = None
            jpmaqs.validate_download_args(**bad_args)

        with self.assertRaises(ValueError):
            bad_args = good_args.copy()
            bad_args["cids"] = None
            jpmaqs.validate_download_args(**bad_args)

        for date_args in ["start_date", "end_date"]:
            bad_args = good_args.copy()
            bad_args[date_args] = "Metallica"
            with self.assertRaises(ValueError):
                jpmaqs.validate_download_args(**bad_args)

            bad_args[date_args] = "1900-01-01"
            with self.assertWarns(UserWarning):
                jpmaqs.validate_download_args(**bad_args)

            # pd.Timestamp extreme cases
            bad_args[date_args] = "1200-01-01"
            with self.assertRaises(ValueError):
                jpmaqs.validate_download_args(**bad_args)

    def test_download_func(self):
        good_args: Dict[str, Any] = {
            "tickers": ["EUR_FXXR_NSA", "USD_FXXR_NSA"],
            "cids": ["GBP", "EUR"],
            "xcats": ["FXXR_NSA", "EQXR_NSA"],
            "metrics": ["value", "grading", "eop_lag", "mop_lag"],
            "start_date": "2019-01-01",
            "end_date": "2019-01-31",
            "expressions": [
                "DB(JPMAQS,AUD_FXXR_NSA,value)",
                "DB(JPMAQS,CAD_FXXR_NSA,value)",
            ],
            "show_progress": True,
            "as_dataframe": True,
            "report_time_taken": True,
            "report_egress": True,
        }

        jpmaqs: JPMaQSDownload = JPMaQSDownload(
            client_id="client_id",
            client_secret="client_secret",
            check_connection=False,
        )

        config: Config = Config(
            client_id="client_id",
            client_secret="client_secret",
        )

        mock_dq_interface: MockDataQueryInterface = MockDataQueryInterface(
            config=config
        )
        un_avail_exprs: List[str] = [
            "DB(JPMAQS,USD_FXXR_NSA,value)",
            "DB(JPMAQS,USD_FXXR_NSA,grading)",
            "DB(JPMAQS,USD_FXXR_NSA,eop_lag)",
            "DB(JPMAQS,USD_FXXR_NSA,mop_lag)",
        ]
        mock_dq_interface._gen_attributes(
            unavailable_expressions=un_avail_exprs, mask_expressions=un_avail_exprs
        )

        # mock dq interface
        jpmaqs.dq_interface = mock_dq_interface

        try:
            test_df: pd.DataFrame = jpmaqs.download(**good_args)

            tickers: pd.Series = test_df["cid"] + "_" + test_df["xcat"]

            self.assertFalse("USD_FXXR_NSA" in tickers.values)
            for cid in ["GBP", "EUR"]:
                for xcat in ["FXXR_NSA", "EQXR_NSA"]:
                    self.assertTrue(f"{cid}_{xcat}" in tickers.values)

            self.assertTrue(len(jpmaqs.msg_errors) > 0)
            self.assertTrue(len(jpmaqs.msg_warnings) > 0)
            for unav in set(un_avail_exprs):
                self.assertTrue(
                    sum(
                        [
                            unav in msg_unav
                            for msg_unav in set(jpmaqs.unavailable_expr_messages)
                        ]
                    )
                    == 1
                )

        except Exception as e:
            self.fail("Unexpected exception raised: {}".format(e))

        # now test with fail condition where no expressions are available
        test_exprs: List[str] = JPMaQSDownload.construct_expressions(
            cids=["GBP", "EUR", "CAD"],
            xcats=["FXXR_NSA", "EQXR_NSA"],
            metrics=[
                "value",
                "grading",
            ],
        )
        mock_dq_interface._gen_attributes(
            unavailable_expressions=test_exprs, mask_expressions=test_exprs
        )

        with self.assertRaises(InvalidDataframeError):
            jpmaqs.dq_interface = mock_dq_interface
            bad_args: Dict[str, Any] = good_args.copy()
            bad_args["expressions"] = test_exprs
            jpmaqs.download(**bad_args)

        with self.assertRaises(AssertionError):
            # the assertion checks whether the download/DQInterface is "mismatched"
            jpmaqs.download(**good_args)

        jpmaqs: JPMaQSDownload = JPMaQSDownload(
            client_id="client_id",
            client_secret="client_secret",
            check_connection=False,
        )

        config: Config = Config(
            client_id="client_id",
            client_secret="client_secret",
        )

        mock_dq_interface: MockDataQueryInterface = MockDataQueryInterface(
            config=config
        )
        un_avail_exprs: List[str] = [
            "DB(JPMAQS,USD_FXXR_NSA,value)",
            "DB(JPMAQS,USD_FXXR_NSA,grading)",
            "DB(JPMAQS,USD_FXXR_NSA,eop_lag)",
            "DB(JPMAQS,USD_FXXR_NSA,mop_lag)",
        ]
        mock_dq_interface._gen_attributes(
            unavailable_expressions=un_avail_exprs, mask_expressions=un_avail_exprs
        )
        jpmaqs.dq_interface = mock_dq_interface
        with self.assertWarns(UserWarning):
            bad_args = good_args.copy()
            bad_args["start_date"] = "2019-01-31"
            bad_args["end_date"] = "2019-01-01"
            jpmaqs.download(**bad_args)

    def test_validate_downloaded_df(self):
        pass


if __name__ == "__main__":
    unittest.main()
