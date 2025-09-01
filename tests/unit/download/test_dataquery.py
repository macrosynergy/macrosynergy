from unittest import mock
import unittest
import datetime
import base64
import warnings
import logging
from typing import List, Dict, Union, Any
import requests
import numpy as np
import itertools

from macrosynergy.download.jpmaqs import JPMaQSDownload, construct_expressions
from macrosynergy.download.dataquery import (
    DataQueryInterface,
    DataQueryOAuth,
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
    CERT_BASE_URL,
    CATALOGUE_ENDPOINT,
)
from macrosynergy.download.exceptions import (
    AuthenticationError,
    HeartbeatError,
    InvalidResponseError,
    DownloadError,
    InvalidDataframeError,
    NoContentError,
)


from .mock_helpers import mock_jpmaqs_value, mock_request_wrapper, random_string


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
        user_id: str = f"User_{random_string()}"
        with self.assertRaises(AuthenticationError):
            validate_response(
                self.mock_response(url=OAUTH_TOKEN_URL, status_code=401),
                user_id=user_id,
            )

        # mock with a 403, and use url=oauth+heartbeat. assert raises heartbeat error
        with self.assertRaises(HeartbeatError):
            validate_response(
                self.mock_response(
                    url=OAUTH_BASE_URL + HEARTBEAT_ENDPOINT, status_code=403
                ),
                user_id=user_id,
            )

        # mock with a 403, and use url=oauth_base_url. assert raises invalid response error
        with self.assertRaises(InvalidResponseError):
            validate_response(
                self.mock_response(url=OAUTH_BASE_URL, status_code=403), user_id=user_id
            )

        # oauth_bas_url+timeseires and empty content, assert raises invalid response error
        with self.assertRaises(InvalidResponseError):
            validate_response(
                self.mock_response(
                    url=OAUTH_BASE_URL + TIMESERIES_ENDPOINT,
                    status_code=200,
                    content=b"",
                ),
                user_id=user_id,
            )

        # with 200 , and empty content, assert raises invalid response error
        with self.assertRaises(InvalidResponseError):
            validate_response(
                self.mock_response(url=OAUTH_TOKEN_URL, status_code=200, content=b""),
                user_id=user_id,
            )

        # with non-json content, assert raises invalid response error
        with self.assertRaises(InvalidResponseError):
            validate_response(
                self.mock_response(
                    url=OAUTH_TOKEN_URL, status_code=200, content=b"not json"
                ),
                user_id=user_id,
            )

        # mock a response.Response with 200, and json content. assert returns response
        response: requests.Response = self.mock_response(
            url=OAUTH_TOKEN_URL,
            status_code=200,
            content=b'{"access_token": "SOME_TOKEN", "expires_in": 3600}',
        )
        self.assertEqual(validate_response(response, user_id=user_id), response.json())

        # mock the call to response.json() to return None. assert raises invalid response error
        with mock.patch("requests.Response.json", return_value=None):
            with self.assertRaises(InvalidResponseError):
                validate_response(response, user_id=user_id)

    def test_request_wrapper(self):
        warnings.filterwarnings("ignore", category=UserWarning, module="logger")
        curr_logger_level: int = logging.getLogger().getEffectiveLevel()
        logging.getLogger().setLevel(logging.ERROR)
        user_id: str = f"User_{random_string()}"

        with self.assertRaises(ValueError):
            request_wrapper(method="pop", url=OAUTH_TOKEN_URL, user_id=user_id)

        def mock_auth_error(*args, **kwargs) -> requests.Response:
            return self.mock_response(url=OAUTH_TOKEN_URL, status_code=401)

        with mock.patch("requests.Session.send", side_effect=mock_auth_error):
            with self.assertRaises(AuthenticationError):
                request_wrapper(
                    method="get",
                    url=OAUTH_TOKEN_URL,
                    user_id=user_id,
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
                    tracking_id=random_string(),
                )

        warnings.resetwarnings()
        logging.getLogger().setLevel(curr_logger_level)


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
            cfg: dict = dict(
                username="user", password="pass", crt="path/crt.crt", key="path/key.key"
            )
            dq_interface: DataQueryInterface = DataQueryInterface(**cfg, oauth=False)

            # assert that dq_interface.auth is CertAuth type
            self.assertIsInstance(dq_interface.auth, CertAuth)
            # check that the base_url is cert_base url
            self.assertEqual(dq_interface.base_url, CERT_BASE_URL)


##############################################


class TestOAuth(unittest.TestCase):
    def test_init(self):
        jpmaqs: JPMaQSDownload = JPMaQSDownload(
            oauth=True,
            client_id="test-id",
            client_secret="SECRET",
            check_connection=False,
        )
        self.assertEqual(jpmaqs.base_url, OAUTH_BASE_URL)

        with self.assertRaises(TypeError):
            DataQueryOAuth(client_id="test-id", client_secret="SECRET", proxy="proxy")

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
                DataQueryOAuth(**bad_args)

        try:
            oath_obj: DataQueryOAuth = DataQueryOAuth(**good_args)

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
        oauth = DataQueryOAuth(client_id="test-id", client_secret="SECRET")
        self.assertFalse(oauth._is_valid_token())

    def test_get_token(self):
        oauth = DataQueryOAuth(client_id="test-id", client_secret="SECRET")

        token_data: Dict[str, str] = {
            "access_token": "SOME_TOKEN",
            "expires_in": 3600,
        }

        with mock.patch(
            "macrosynergy.download.jpm_oauth.JPMorganOAuth.retrieve_token",
            return_value=token_data,
        ):
            with mock.patch(
                "macrosynergy.download.dataquery.DataQueryOAuth._is_valid_token",
                return_value=False,
            ):
                oauth._stored_token = token_data
                self.assertEqual(oauth._get_token(), token_data["access_token"])


##############################################


class TestDataQueryInterface(unittest.TestCase):
    def setUp(self) -> None:
        self.cids: List[str] = ["GBP", "EUR", "CAD"]
        self.xcats: List[str] = ["FXXR_NSA", "EQXR_NSA"]
        self.metrics: List[str] = ["value", "grading", "eop_lag", "mop_lag"]
        self.start_date = "2000-01-01"
        self.end_date = "2000-02-01"
        self.tickers: List[str] = [
            cid + "_" + xcat for xcat in self.xcats for cid in self.cids
        ]
        self.expressions: List[str] = construct_expressions(
            cids=self.cids, xcats=self.xcats, metrics=self.metrics
        )
        self.dq = DataQueryInterface(
            client_id=random_string(),
            client_secret=random_string(),
            oauth=True,
            check_connection=False,
        )

    @staticmethod
    def jpmaqs_value(elem: str) -> float:
        """
        Use the mock jpmaqs_value to return a mock numerical jpmaqs value.
        """
        return mock_jpmaqs_value(elem=elem)

    def request_wrapper(
        self, dq_expressions: List[str], start_date: str, end_date: str
    ) -> List[Dict[str, Any]]:
        """
        Use the mock request_wrapper to return a mock response.
        """
        return mock_request_wrapper(
            dq_expressions=dq_expressions, start_date=start_date, end_date=end_date
        )

    def test_init(self):
        def mock_isfile(path: str) -> bool:
            return path in ["path/key.key", "path/crt.crt"]

        with self.assertRaises(TypeError):
            DataQueryInterface(client_id=1, client_secret="SECRET")

        with self.assertRaises(ValueError):
            with self.assertWarns(UserWarning):
                DataQueryInterface(
                    client_id=None,
                    client_secret=None,
                    oauth=True,
                )
        with mock.patch("os.path.isfile", side_effect=lambda x: mock_isfile(x)):
            with self.assertWarns(UserWarning):
                DataQueryInterface(
                    client_id=None,
                    client_secret=None,
                    check_connection=False,
                    oauth=True,
                    username="user",
                    password="pass",
                    crt="path/crt.crt",
                    key="path/key.key",
                )

    @mock.patch(
        "macrosynergy.download.dataquery.DataQueryOAuth._get_token",
        return_value=("SOME_TEST_TOKEN"),
    )
    @mock.patch(
        "macrosynergy.download.dataquery.request_wrapper",
        return_value=({"info": {"code": 200, "message": "Service Available."}}),
    )
    def test_check_connection(
        self, mock_p_request: mock.MagicMock, mock_p_get_token: mock.MagicMock
    ):
        # If the connection to DataQuery is working, the response code will invariably be
        # 200. Therefore, use the Interface Object's method to check DataQuery
        # connections.

        def _test(verbose: bool):
            with self.dq as dq:
                self.assertTrue(
                    dq.check_connection(
                        verbose=verbose,
                    )
                )

            mock_p_request.assert_called_once()
            mock_p_get_token.assert_called_once()
            # reset the mocks
            mock_p_request.reset_mock()
            mock_p_get_token.reset_mock()
            return True

        for verbose in [True, False]:
            self.assertTrue(_test(verbose))

        with mock.patch(
            "macrosynergy.download.dataquery.request_wrapper",
            return_value=None,
        ):
            with self.dq as dq:
                with self.assertRaises(ConnectionError):
                    dq.check_connection(raise_error=True)

    @mock.patch(
        "macrosynergy.download.dataquery.DataQueryOAuth._get_token",
        return_value=random_string(),
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
    def test_check_connection_fail(
        self, mock_p_fail: mock.MagicMock, mock_p_get_token: mock.MagicMock
    ):
        # Opposite of above method: if the connection to DataQuery fails, the error code
        # will be 400.
        curr_logger_level: int = logging.getLogger().getEffectiveLevel()
        logging.getLogger().setLevel(logging.ERROR)
        with self.dq as dq:
            # Method returns a BoolFean. In this instance, the method should return False
            # (unable to connect).
            self.assertFalse(dq.check_connection())
        mock_p_fail.assert_called_once()
        mock_p_get_token.assert_called_once()
        logging.getLogger().setLevel(curr_logger_level)

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

        # check that jpmaqs_download is a superclass of dataquery interface
        self.assertIsInstance(jpmaqs_download, DataQueryInterface)

        self.assertIsInstance(jpmaqs_download.auth, DataQueryOAuth)

    def test_certauth_condition(self):
        # Second check is that the DataQuery instance is using an CertAuth Object if the
        # parameter "oauth" is set to to False. The DataQuery Class's default is to use
        # certificate / keys.

        # Given the certificate and key will not point to valid directories, the expected
        # behaviour is for an OSError to be thrown.
        with self.assertRaises(FileNotFoundError):
            DataQueryInterface(
                username="user1",
                password="123",
                crt="/api_macrosynergy_com.crt",
                key="/api_macrosynergy_com.key",
                oauth=False,
                check_connection=False,
            )

    def test_dq_fetch(self):
        invl_responses: List[Any] = [
            None,
            {},
            {"attributes": []},
            {"attributes": [{"expression": "expression1"}]},
        ]

        self.dq.auth._stored_token = {
            "created_at": datetime.datetime.now(datetime.timezone.utc),
            "access_token": random_string(),
            "expires_in": 3600,
        }

        for invl_response in invl_responses:
            with mock.patch(
                "macrosynergy.download.dataquery.request_wrapper",
                return_value=invl_response,
            ):
                with self.assertRaises(InvalidResponseError):
                    self.dq._fetch(
                        url=OAUTH_BASE_URL + TIMESERIES_ENDPOINT,
                        params={"expr": "expression1"},
                    )

        invl_response: Dict[str, Any] = {
            "info": {"code": "204", "message": "No Content"}
        }
        with mock.patch(
            "macrosynergy.download.dataquery.request_wrapper",
            return_value=invl_response,
        ):
            with self.assertRaises(NoContentError):
                self.dq._fetch(
                    url=OAUTH_BASE_URL + CATALOGUE_ENDPOINT,
                    params={"group": "group-name"},
                )

    def test_chain_download_outputs(self):
        exprs = construct_expressions(
            cids=["AUD", "CAD"], xcats=["EQXR_NSA"], metrics=["value"]
        )
        responses = mock_request_wrapper(
            dq_expressions=exprs,
            start_date="2000-01-01",
            end_date="2001-01-01",
        )

        for i in range(len(responses)):
            responses[i] = [responses[i]]

        lA = self.dq._chain_download_outputs(download_outputs=responses)
        lB = itertools.chain.from_iterable(responses)
        lA = str(sorted([str(x) for x in lA]))
        lB = str(sorted([str(x) for x in lB]))
        self.assertEqual(lA, lB)

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
            "batch_size": 20,
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

        # test all against an int (except retry_counter, batch_size)
        for key in good_args.keys():
            bad_value: Union[int, str] = 1
            if key in ["retry_counter", "batch_size"]:
                bad_value = "1"
            bad_args: Dict[str, Any] = good_args.copy()
            bad_args[key] = bad_value
            with self.assertRaises(TypeError):
                validate_download_args(**bad_args)

        # test delay_param
        for delay_param in [-0.1, -1.0]:
            bad_args: Dict[str, Any] = good_args.copy()
            bad_args["delay_param"] = delay_param
            with self.assertRaises(ValueError):
                validate_download_args(**bad_args)

        for delay_param in [0.0, 0.1, 0.15]:
            bad_args: Dict[str, Any] = good_args.copy()
            bad_args["delay_param"] = delay_param
            with warnings.catch_warnings(record=True) as w:
                validate_download_args(**bad_args)
                self.assertEqual(len(w), 1)
                self.assertTrue(issubclass(w[-1].category, RuntimeWarning))

        for date_arg in ["start_date", "end_date"]:
            bad_args: Dict[str, Any] = good_args.copy()
            bad_args[date_arg] = "1-Jan-2023"
            with self.assertRaises(ValueError):
                validate_download_args(**bad_args)

        # if the batchsize if negative or non int, should raise type error
        for batch_size in [0.1, "1", None, [], {}]:
            bad_args: Dict[str, Any] = good_args.copy()
            bad_args["batch_size"] = batch_size
            with self.assertRaises(TypeError):
                validate_download_args(**bad_args)

        for batch_size in [-1, 0]:
            bad_args: Dict[str, Any] = good_args.copy()
            bad_args["batch_size"] = batch_size
            with self.assertRaises(ValueError):
                validate_download_args(**bad_args)

        # test that no warnings are raised if 1<bath_size<20
        for batch_size in [1, 19, 20]:
            good_args: Dict[str, Any] = good_args.copy()
            good_args["batch_size"] = batch_size
            # assert no warnings raised
            with warnings.catch_warnings(record=True) as w:
                validate_download_args(**good_args)
                self.assertEqual(len(w), 0)

        # test that >20 batch_size raises warning
        for batch_size in [21, 100]:
            bad_args: Dict[str, Any] = good_args.copy()
            bad_args["batch_size"] = batch_size
            with self.assertWarns(RuntimeWarning):
                validate_download_args(**bad_args)

    def test_get_unavailable_expressions(self):
        cids = ["AUD", "CAD", "CHF", "EUR"]
        xcats = ["EQXR_NSA", "FXXR_NSA"]
        metrics = ["value", "grading", "eop_lag", "mop_lag"]

        expression = construct_expressions(
            metrics=metrics,
            cids=cids,
            xcats=xcats,
        )

        # slect 10 random expressions
        unavailable_expressions = list(np.random.choice(expression, 10))
        expression = list(set(expression) - set(unavailable_expressions))

        dicts_list = self.request_wrapper(
            dq_expressions=expression,
            start_date="2000-01-01",
            end_date="2001-01-01",
        )

        mexprs = self.dq._get_unavailable_expressions(
            expected_exprs=expression + unavailable_expressions,
            dicts_list=dicts_list,
        )
        self.assertEqual(set(mexprs), set(unavailable_expressions))

    def test_concurrent_loop(self):
        params_dict: Dict = {
            "format": "JSON",
            "start-date": "1990-01-01",
            "end-date": "2020-01-01",
        }

        def _rw(url, params, tracking_id, **kwargs):
            return self.request_wrapper(
                dq_expressions=params["expressions"],
                start_date=params["start-date"],
                end_date=params["end-date"],
            )

        expr_batches = [
            self.expressions[i : i + 20] for i in range(0, len(self.expressions), 20)
        ]
        show_progress = False
        delay_param = 0.25

        arguments = dict(
            url=random_string(),
            params=params_dict,
            tracking_id=random_string(),
            delay_param=delay_param,
            expr_batches=expr_batches,
            show_progress=show_progress,
        )

        with mock.patch(
            "macrosynergy.download.dataquery.DataQueryInterface._fetch_timeseries",
            side_effect=Exception,
        ):
            results = self.dq._concurrent_loop(
                **arguments,
            )

            self.assertEqual(len(results), len(expr_batches))

        ## test complete failure
        arguments["expr_batches"] += [None for _ in range(10)]
        with mock.patch(
            "macrosynergy.download.dataquery.DataQueryInterface._fetch_timeseries",
            side_effect=_rw,
        ):
            with self.assertRaises(DownloadError):
                results = self.dq._concurrent_loop(
                    **arguments,
                )

        arguments["expr_batches"][0] += ["KEYBOARD_INTERRUPT"]
        with mock.patch(
            "macrosynergy.download.dataquery.DataQueryInterface._fetch_timeseries",
            side_effect=KeyboardInterrupt,
        ):
            with self.assertRaises(KeyboardInterrupt):
                results = self.dq._concurrent_loop(
                    **arguments,
                )

    def test_fetch(self):
        def _rw(url: str = "", params: Dict = {}, tracking_id: str = "", **kwargs):
            if len(params) > 0:
                d = {
                    "instruments": mock_request_wrapper(
                        dq_expressions=params["expressions"],
                        start_date=params["start-date"],
                        end_date=params["end-date"],
                    ),
                    "links": [
                        {"self": "SELF"},
                        {"next": "NEXT"},
                    ],
                }
            else:
                d = {
                    "instruments": mock_request_wrapper(
                        dq_expressions=["DB(JPMAQS,A_B_C,value)"],
                        start_date="2020-01-01",
                        end_date="2020-02-01",
                    ),
                    "links": [
                        {"self": "SELF"},
                        {"next": None},
                    ],
                }
            return d

        params_dict = {
            "expressions": self.expressions,
            "start-date": "2020-01-01",
            "end-date": "2020-02-01",
        }

        with mock.patch(
            "macrosynergy.download.dataquery.DataQueryOAuth.get_auth",
            return_value={"headers": "headers", "cert": "cert"},
        ):
            with mock.patch(
                "macrosynergy.download.dataquery.request_wrapper",
                side_effect=_rw,
            ):
                fetch_list = self.dq._fetch(url=random_string(), params=params_dict)
                self.assertIsInstance(fetch_list, list)
                for f_item in fetch_list:
                    self.assertIsInstance(f_item, dict)
                    self.assertTrue(
                        f_item["attributes"][0]["expression"]
                        in self.expressions + ["DB(JPMAQS,A_B_C,value)"]
                    )

    def test_fetch_timeseries(self):
        def _mock_fetch(*args, **kwargs):
            return [str(i) for i in range(1, 11)]

        with mock.patch(
            "macrosynergy.download.dataquery.DataQueryInterface._fetch",
            side_effect=_mock_fetch,
        ):
            self.assertEqual(
                self.dq._fetch_timeseries(
                    url=random_string(),
                    params={},
                    tracking_id=random_string(),
                ),
                [str(i) for i in range(1, 11)],
            )

    def test_get_catalogue(self):
        def _mock_fetch(*args, **kwargs):
            return [
                {"item": i, "instrument-id": f"ID_{i}", "instrument-name": f"NAME_{i}"}
                for i in range(1, 11)
            ]

        def _bad_mock_fetch(*args, **kwargs):
            return [
                {"item": i, "instrument-id": f"ID_{i}", "instrument-name": f"NAME_{i}"}
                for i in range(1, 11)
            ] + [{"item": 10, "instrument-id": "ID_10", "instrument-name": "NAME_10"}]

        with mock.patch(
            "macrosynergy.download.dataquery.DataQueryInterface._fetch",
            side_effect=_mock_fetch,
        ):
            cat = self.dq.get_catalogue()
            self.assertIsInstance(cat, list)

            self.assertEqual(set(cat), set([f"NAME_{i}" for i in range(1, 11)]))

        with mock.patch(
            "macrosynergy.download.dataquery.DataQueryInterface._fetch",
            side_effect=_bad_mock_fetch,
        ):
            with self.assertRaises(ValueError):
                cat = self.dq.get_catalogue()

        with mock.patch(
            "macrosynergy.download.dataquery.DataQueryInterface._fetch",
            side_effect=Exception,
        ):
            with self.assertRaises(Exception):
                cat = self.dq.get_catalogue()

    def test_download(self):
        def _mock_concurrent_loop(*args, **kwargs):
            if "expr_batches" in kwargs:
                if len(kwargs["expr_batches"]) > 1:
                    m = mock_request_wrapper(
                        dq_expressions=itertools.chain.from_iterable(
                            kwargs["expr_batches"][:-1]
                        ),
                        start_date=self.start_date,
                        end_date=self.end_date,
                    )
                    m = [[u] for u in m]
                    return m, [kwargs["expr_batches"][-1]]
                else:
                    m = mock_request_wrapper(
                        dq_expressions=kwargs["expr_batches"][0],
                        start_date=self.start_date,
                        end_date=self.end_date,
                    )
                    return [[u] for u in m], []

        def _bad_mock_concurrent_loop(*args, **kwargs):
            return [], [kwargs["expr_batches"]]

        good_args = dict(
            expressions=self.expressions,
            params={},
            url=random_string(),
            tracking_id=random_string(),
            delay_param=0.25,
        )

        with mock.patch(
            "macrosynergy.download.dataquery.DataQueryInterface._concurrent_loop",
            side_effect=_mock_concurrent_loop,
        ):
            result = self.dq._download(**good_args)
            self.assertIsInstance(result, list)
            self.assertEqual(len(result), len(self.expressions))

        with mock.patch(
            "macrosynergy.download.dataquery.DataQueryInterface._concurrent_loop",
            side_effect=_bad_mock_concurrent_loop,
        ):
            with self.assertRaises(DownloadError):
                result = self.dq._download(**good_args)

    def test_download_data(self):
        good_args: Dict[str, Any] = dict(
            expressions=self.expressions,
            start_date="2020-01-01",
            end_date=None,
            show_progress=False,
        )

        def _mock_fetch(*args, **kwargs):
            return mock_request_wrapper(
                dq_expressions=self.expressions,
                start_date=self.start_date,
                end_date=datetime.datetime.today().strftime("%Y-%m-%d"),
            )

        with mock.patch(
            "macrosynergy.download.dataquery.DataQueryInterface.check_connection",
            return_value=True,
        ):
            with mock.patch(
                "macrosynergy.download.dataquery.DataQueryInterface._fetch_timeseries",
                side_effect=_mock_fetch,
            ):
                with DataQueryInterface(
                    client_id=random_string(),
                    client_secret=random_string(),
                    oauth=True,
                ) as dq:
                    result = dq.download_data(**good_args)
                    self.assertIsInstance(result, list)

                with self.assertWarns(UserWarning):
                    bad_args = good_args.copy()
                    bad_args["start_date"] = self.end_date
                    bad_args["end_date"] = self.start_date
                    result = dq.download_data(**bad_args)

        with mock.patch(
            "macrosynergy.download.dataquery.DataQueryInterface.check_connection",
            return_value=False,
        ):
            with mock.patch(
                "macrosynergy.download.dataquery.DataQueryOAuth.get_auth",
                return_value={"user_id": random_string()},
            ):
                with self.assertRaises(ConnectionError):
                    with DataQueryInterface(
                        client_id=random_string(),
                        client_secret=random_string(),
                        oauth=True,
                    ) as dq:
                        result = dq.download_data(**good_args)


if __name__ == "__main__":
    unittest.main()
