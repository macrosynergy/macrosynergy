from macrosynergy.download import dataquery
from macrosynergy.download import JPMaQSDownload
from typing import List
from unittest import mock
from random import random
import unittest
import numpy as np
import pandas as pd


class TestCertAuth(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, True)
        # TODO : Write tests for the CertAuth class


class TestOAuth(unittest.TestCase):
    def test_init(self):
        oauth: dataquery.OAuth = dataquery.OAuth(
            client_id="test-id", client_secret="SECRET"
        )

        self.assertEqual(dataquery.OAUTH_BASE_URL, oauth.base_url)
        self.assertEqual("test-id", oauth.client_id)
        self.assertEqual("SECRET", oauth.client_secret)

    # def test_invalid_args_passed(self):
    def test_invalid_init_args(self):
        # test invalid client_id
        with self.assertRaises(TypeError):
            dataquery.OAuth(client_id=123, client_secret="SECRET")

        # test invalid client_secret
        with self.assertRaises(TypeError):
            dataquery.OAuth(client_id="test-id", client_secret=123)

        # test invalid base_url
        with self.assertRaises(TypeError):
            dataquery.OAuth(client_id="test-id", client_secret="SECRET", base_url=None)

        # test invalid token_url
        with self.assertRaises(TypeError):
            dataquery.OAuth(client_id="test-id", client_secret="SECRET", token_url=None)

        # test invalid dq_resource_id
        with self.assertRaises(TypeError):
            dataquery.OAuth(
                client_id="test-id", client_secret="SECRET", dq_resource_id=None
            )

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

    @mock.patch(
        "macrosynergy.download.dataquery.OAuth._request",
        return_value=({"info": {"code": 200, "message": "Service Available."}}),
    )
    def test_check_connection(self, mock_p_request):
        # If the connection to DataQuery is working, the response code will invariably be
        # 200. Therefore, use the Interface Object's method to check DataQuery
        # connections.

        with JPMaQSDownload(
            client_id="client1", client_secret="123", oauth=True, check_connection=False
        ) as jpmaqs:
            self.assertTrue(jpmaqs.check_connection())

        mock_p_request.assert_called_once()

    @mock.patch(
        "macrosynergy.download.dataquery.OAuth._request",
        return_value=({"info": {"code": 200, "message": "Service Available."}}),
    )
    def test_check_connection_on_init(self, mock_p_request):
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

    @mock.patch(
        "macrosynergy.download.dataquery.OAuth._request",
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
    def test_check_connection_fail(self, mock_p_fail):
        # Opposite of above method: if the connection to DataQuery fails, the error code
        # will be 400.

        with JPMaQSDownload(
            client_id="client1", client_secret="123", oauth=True, check_connection=False
        ) as jpmaqs_download:
            # Method returns a Boolean. In this instance, the method should return False
            # (unable to connect).
            self.assertFalse(jpmaqs_download.check_connection())
        mock_p_fail.assert_called_once()

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
        self.assertIsInstance(
            jpmaqs_download.dq_interface.auth, dataquery.OAuth
        )

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


if __name__ == "__main__":
    unittest.main()
