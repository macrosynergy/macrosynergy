from unittest import mock
import unittest
import warnings
import pandas as pd
import itertools
import os
from typing import List, Dict, Any
from macrosynergy.download.jpmaqs import (
    JPMaQSDownload,
    construct_expressions,
    deconstruct_expression,
    get_expression_from_qdf,
    get_expression_from_wide_df,
    concat_column_dfs,
    timeseries_to_qdf,
    timeseries_to_column,
    concat_single_metric_qdfs,
    validate_downloaded_df,
    check_attributes_in_sync,
    DEFAULT_CLIENT_ID_ENV_VAR,
    DEFAULT_CLIENT_SECRET_ENV_VAR,
)

from macrosynergy.download.exceptions import InvalidDataframeError
from macrosynergy.management.types import QuantamentalDataFrame
from .mock_helpers import (
    mock_jpmaqs_value,
    mock_request_wrapper,
    random_string,
    MockDataQueryInterface,
)


class TestJPMaQSDownload(unittest.TestCase):
    def setUp(self) -> None:
        self.cids: List[str] = ["GBP", "EUR", "CAD"]
        self.xcats: List[str] = ["FXXR_NSA", "EQXR_NSA"]
        self.metrics: List[str] = ["value", "grading", "eop_lag", "mop_lag"]
        self.start_date = "2019-01-01"
        self.end_date = "2019-01-31"
        self.tickers: List[str] = [
            cid + "_" + xcat for xcat in self.xcats for cid in self.cids
        ]
        self.expressions: List[str] = construct_expressions(
            cids=self.cids, xcats=self.xcats, metrics=self.metrics
        )

        self.jpmaqs_download = JPMaQSDownload(
            client_id="client_id",
            client_secret="client_secret",
            check_connection=False,
        )

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

        with self.assertRaises(ValueError):
            bad_args = good_args.copy()
            for vx in [
                "client_id",
                "client_secret",
                "crt",
                "key",
                "username",
                "password",
            ]:
                bad_args[vx] = None

            JPMaQSDownload(**bad_args)
            # the above line will NOT fail for systems with the env vars set
            # therefore if the below is reached, the the env vars must be there.
            if bool(os.getenv(DEFAULT_CLIENT_ID_ENV_VAR)) and bool(
                os.getenv(DEFAULT_CLIENT_SECRET_ENV_VAR)
            ):
                raise ValueError("Env. variables are set; raising error to pass test")

    @mock.patch(
        "macrosynergy.download.dataquery.OAuth._get_token",
        return_value=("SOME_TEST_TOKEN"),
    )
    @mock.patch(
        "macrosynergy.download.dataquery.request_wrapper",
        return_value=({"info": {"code": 200, "message": "Service Available."}}),
    )
    def test_check_connection_on_init(
        self, mock_p_request: mock.MagicMock, mock_p_get_token: mock.MagicMock
    ):
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

    def test_get_unavailable_expressions(self):
        dicts_list = mock_request_wrapper(
            dq_expressions=self.expressions,
            start_date="2019-01-01",
            end_date="2019-01-31",
        )
        expected_expressions = self.expressions
        missing_exprs = []
        for i in range(3):
            missing_exprs.append(dicts_list[i]["attributes"][0]["expression"])
            dicts_list[i]["attributes"][0]["time-series"] = None

        self.assertEqual(
            set(
                self.jpmaqs_download._get_unavailable_expressions(
                    expected_exprs=expected_expressions,
                    dicts_list=dicts_list,
                )
            ),
            set(missing_exprs),
        )

        ## check raises error when both dicts_list and df_list are provided
        with self.assertRaises(AssertionError):
            self.jpmaqs_download._get_unavailable_expressions(
                expected_exprs=expected_expressions,
                dicts_list=dicts_list,
                downloaded_df=dicts_list,
            )

        self.assertEqual(
            set(expected_expressions),
            set(
                self.jpmaqs_download._get_unavailable_expressions(
                    expected_exprs=expected_expressions,
                    downloaded_df=pd.DataFrame(),
                ),
            ),
        )

        qdf = concat_single_metric_qdfs(list(map(timeseries_to_qdf, dicts_list)))
        wdf = concat_column_dfs(list(map(timeseries_to_column, dicts_list)))

        self.assertEqual(
            set(missing_exprs),
            set(
                self.jpmaqs_download._get_unavailable_expressions(
                    expected_exprs=expected_expressions,
                    downloaded_df=qdf,
                ),
            ),
        )

        self.assertEqual(
            set(missing_exprs),
            set(
                self.jpmaqs_download._get_unavailable_expressions(
                    expected_exprs=expected_expressions,
                    downloaded_df=wdf,
                ),
            ),
        )

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
            "get_catalogue": True,
            "dataframe_format": "qdf",
        }
        bad_args: Dict[str, Any] = {}

        try:
            if not self.jpmaqs_download.validate_download_args(**good_args):
                self.fail("Unexpected validation failure")
        except Exception as e:
            self.fail("Unexpected exception raised: {}".format(e))

        for argx in good_args:
            with self.assertRaises(TypeError):
                bad_args = good_args.copy()
                bad_args[argx] = -1
                self.jpmaqs_download.validate_download_args(**bad_args)

        # value error for tickers==cids==xcats==expressions == None
        with self.assertRaises(ValueError):
            bad_args = good_args.copy()
            bad_args["tickers"] = None
            bad_args["cids"] = None
            bad_args["xcats"] = None
            bad_args["expressions"] = None
            self.jpmaqs_download.validate_download_args(**bad_args)

        for lvarx in ["tickers", "cids", "xcats", "expressions"]:
            with self.assertRaises(ValueError):
                bad_args = good_args.copy()
                bad_args[lvarx] = []
                self.jpmaqs_download.validate_download_args(**bad_args)

            with self.assertRaises(TypeError):
                bad_args = good_args.copy()
                bad_args[lvarx] = [1]
                self.jpmaqs_download.validate_download_args(**bad_args)

        # test cases for metrics arg
        bad_args = good_args.copy()
        bad_args["metrics"] = None
        with self.assertRaises(ValueError):
            self.jpmaqs_download.validate_download_args(**bad_args)

        bad_args = good_args.copy()
        bad_args["metrics"] = ["Metallica"]
        with self.assertRaises(ValueError):
            self.jpmaqs_download.validate_download_args(**bad_args)

        # cid AND xcat cases
        with self.assertRaises(ValueError):
            bad_args = good_args.copy()
            bad_args["xcats"] = None
            self.jpmaqs_download.validate_download_args(**bad_args)

        with self.assertRaises(ValueError):
            bad_args = good_args.copy()
            bad_args["cids"] = None
            self.jpmaqs_download.validate_download_args(**bad_args)

        for date_args in ["start_date", "end_date"]:
            bad_args = good_args.copy()
            bad_args[date_args] = "Metallica"
            with self.assertRaises(ValueError):
                self.jpmaqs_download.validate_download_args(**bad_args)

            bad_args[date_args] = "1900-01-01"
            with self.assertWarns(UserWarning):
                self.jpmaqs_download.validate_download_args(**bad_args)

            # pd.Timestamp extreme cases
            bad_args[date_args] = "1200-01-01"
            with self.assertRaises(ValueError):
                self.jpmaqs_download.validate_download_args(**bad_args)

        # test cases for dataframe_format
        bad_args = good_args.copy()
        bad_args["dataframe_format"] = "Metallica"
        with self.assertRaises(ValueError):
            self.jpmaqs_download.validate_download_args(**bad_args)

    def test_filter_expressions_from_catalogue(self):

        catalogue = self.expressions.copy()

        with mock.patch(
            "macrosynergy.download.dataquery.DataQueryInterface.get_catalogue",
            return_value=self.tickers.copy(),
        ):
            self.assertEqual(
                set(self.jpmaqs_download.filter_expressions_from_catalogue(catalogue)),
                set(catalogue),
            )

            ctlg = catalogue + [random_string() for _ in range(3)]
            self.assertEqual(
                set(self.jpmaqs_download.filter_expressions_from_catalogue(ctlg)),
                set(catalogue),
            )

    def test_chain_download_outputs(self):

        with self.assertRaises(TypeError):
            self.jpmaqs_download._chain_download_outputs("")

        self.assertEqual([], self.jpmaqs_download._chain_download_outputs([]))

        ts_list = mock_request_wrapper(
            dq_expressions=self.expressions,
            start_date="2019-01-01",
            end_date="2019-01-31",
        )

        ts_list_list = [ts_list[:10], ts_list[10:]]
        ## Test for timeseries list

        self.assertEqual(
            list(itertools.chain.from_iterable(ts_list_list)),
            self.jpmaqs_download._chain_download_outputs(ts_list_list),
        )

        ## Test for QDF list

        qdf_list_list = [
            list(map(timeseries_to_qdf, _ts_list)) for _ts_list in ts_list_list
        ]

        _dfA = concat_single_metric_qdfs(
            list(itertools.chain.from_iterable(qdf_list_list.copy()))
        )  #
        _dfB: QuantamentalDataFrame = self.jpmaqs_download._chain_download_outputs(
            qdf_list_list
        )
        _dfB["cid"] = _dfB["cid"].astype("string")
        _dfB["xcat"] = _dfB["xcat"].astype("string")

        idx_cols = QuantamentalDataFrame.IndexCols
        _dfA = _dfA.sort_values(by=idx_cols).reset_index(drop=True)
        _dfB = _dfB.sort_values(by=idx_cols).reset_index(drop=True)
        self.assertTrue(_dfA.eq(_dfB).all().all())

        ## Test for column list

        cdf_list = list(map(timeseries_to_column, ts_list))
        _dfA = concat_column_dfs(df_list=cdf_list.copy())
        _dfB = self.jpmaqs_download._chain_download_outputs(cdf_list)
        self.assertTrue(_dfA.equals(_dfB))
        ts_LLL = [ts_list_list, ts_list_list]

        with self.assertRaises(NotImplementedError):
            self.jpmaqs_download._chain_download_outputs(ts_LLL)

    def test_fetch_timeseries(self):
        static_ts_list = mock_request_wrapper(
            dq_expressions=self.expressions,
            start_date="2019-01-01",
            end_date="2019-01-31",
        )

        good_args = dict(
            url="",
            params={},
            tracking_id="",
        )

        # test for "good" conditions
        with mock.patch(
            "macrosynergy.download.dataquery.DataQueryInterface._fetch",
            return_value=static_ts_list.copy(),
        ):
            ts_list = self.jpmaqs_download._fetch_timeseries(
                as_dataframe=False,
                **good_args,
            )
            self.assertIsInstance(ts_list, list)
            _test_list = sorted([str(t) for t in static_ts_list])
            self.assertEqual(
                set(sorted([str(t) for t in ts_list])),
                set(_test_list),
            )

            qdf_list = self.jpmaqs_download._fetch_timeseries(
                as_dataframe=True,
                dataframe_format="qdf",
                **good_args,
            )

            self.assertIsInstance(qdf_list, list)
            self.assertEqual(len(qdf_list), len(ts_list))
            for qdf in qdf_list:
                self.assertIsInstance(qdf, QuantamentalDataFrame)

            wdf = self.jpmaqs_download._fetch_timeseries(
                as_dataframe=True,
                dataframe_format="wide",
                **good_args,
            )

            self.assertIsInstance(wdf, pd.DataFrame)
            self.assertEqual(wdf.index.name, "real_date")
            self.assertEqual(set(wdf.columns), set(self.expressions))

        bad_ts_list = static_ts_list.copy()
        bad_ts_list[0]["attributes"][0]["time-series"] = None
        bad_ts_list[1]["attributes"][0]["time-series"] = None
        bad_ts_list[0]["attributes"][0]["message"] = "No data available"
        missing_expr = [bad_ts_list[i]["attributes"][0]["expression"] for i in range(2)]

        def bad_ts_list_func(*args, **kwargs):
            # ts_list gets modified in place by _fetch_timeseries
            return bad_ts_list.copy()

        with mock.patch(
            "macrosynergy.download.dataquery.DataQueryInterface._fetch",
            side_effect=bad_ts_list_func,
        ):
            ts_list = self.jpmaqs_download._fetch_timeseries(
                as_dataframe=False,
                **good_args,
            )
            self.assertIsInstance(ts_list, list)
            self.assertEqual(len(ts_list), len(bad_ts_list) - 2)
            self.assertEqual(
                set([str(t) for t in ts_list]),
                set([str(t) for t in bad_ts_list[2:]]),
            )

            qdf_list = self.jpmaqs_download._fetch_timeseries(
                as_dataframe=True,
                dataframe_format="qdf",
                **good_args,
            )

            self.assertIsInstance(qdf_list, list)
            self.assertEqual(len(qdf_list), len(bad_ts_list) - 2)
            for qdf in qdf_list:
                self.assertIsInstance(qdf, QuantamentalDataFrame)

            wdf = self.jpmaqs_download._fetch_timeseries(
                as_dataframe=True,
                dataframe_format="wide",
                **good_args,
            )

            self.assertIsInstance(wdf, pd.DataFrame)
            self.assertEqual(wdf.index.name, "real_date")
            self.assertEqual(
                set(wdf.columns), set(self.expressions) - set(missing_expr)
            )

            self.assertEqual(
                set(missing_expr), set(self.jpmaqs_download.unavailable_expressions)
            )

    def test_download_func(self):
        def _mock_fetch(
            url: str = "", params: Dict = {}, tracking_id: str = "", **kwargs
        ):
            if len(params) > 0:
                return mock_request_wrapper(
                    dq_expressions=params["expressions"],
                    start_date=params["start-date"],
                    end_date=params["end-date"],
                )
            else:
                return []

        warnings.simplefilter("always")
        good_args = dict(
            # tickers=None,
            cids=self.cids,
            xcats=self.xcats,
            metrics=["all"],
            end_date=None,
            start_date=self.start_date,
            # expressions=None,
            get_catalogue=True,
            show_progress=False,
            debug=False,
            suppress_warning=False,
            as_dataframe=True,
            dataframe_format="qdf",
            report_time_taken=True,
        )
        with mock.patch(
            "macrosynergy.download.dataquery.DataQueryInterface.get_catalogue",
            return_value=self.tickers,
        ):
            with mock.patch(
                "macrosynergy.download.dataquery.DataQueryInterface._fetch",
                side_effect=_mock_fetch,
            ):
                self.jpmaqs_download.download(**good_args)

                # swap dates and see if it raises a warning
                bad_args = good_args.copy()
                bad_args["start_date"] = self.end_date
                bad_args["end_date"] = self.start_date
                with self.assertWarns(UserWarning):
                    self.jpmaqs_download.download(**bad_args)

                # patch validate_download_args to return False
                with mock.patch(
                    "macrosynergy.download.jpmaqs.JPMaQSDownload.validate_download_args",
                    return_value=False,
                ):
                    with self.assertRaises(ValueError):
                        self.jpmaqs_download.download(**good_args)

                # mock validate_downloaded_df to return False
                with mock.patch(
                    "macrosynergy.download.jpmaqs.validate_downloaded_df",
                    return_value=False,
                ):
                    with self.assertRaises(InvalidDataframeError):
                        self.jpmaqs_download.download(**good_args)

                with self.assertWarns(UserWarning):
                    with JPMaQSDownload(
                        client_id="client1",
                        client_secret="123",
                        oauth=True,
                        check_connection=False,
                    ) as jpmaqs:
                        jpmaqs.msg_errors = ["Some error"]
                        jpmaqs.download(**good_args)

        warnings.resetwarnings()


class TestFunctions(unittest.TestCase):
    def setUp(self) -> None:
        self.cids: List[str] = ["GBP", "EUR", "CAD"]
        self.xcats: List[str] = ["FXXR_NSA", "EQXR_NSA"]
        self.metrics: List[str] = ["value", "grading", "eop_lag", "mop_lag"]
        self.expressions: List[str] = construct_expressions(
            cids=self.cids, xcats=self.xcats, metrics=self.metrics
        )

    def test_construct_expressions(self):
        cids = ["AUD", "CAD", "CHF", "EUR", "GBP", "JPY"]
        xcats = ["EQXR_NSA", "FXXR_NSA"]
        tickers = [cid + "_" + xcat for xcat in xcats for cid in cids]
        metrics = ["value", "grading"]

        set_a = construct_expressions(metrics=metrics, tickers=tickers)
        set_b = construct_expressions(metrics=metrics, cids=cids, xcats=xcats)
        self.assertEqual(set(set_a), set(set_b))

        exprs = set(
            [
                f"DB(JPMAQS,{ticker},{metric})"
                for ticker in tickers
                for metric in metrics
            ]
        )
        self.assertEqual(set(set_a), exprs)

    def test_deconstruct_expressions(self):
        cids = ["AUD", "CAD", "CHF", "EUR", "GBP", "JPY"]
        xcats = ["EQXR_NSA", "FXXR_NSA"]
        tickers = [cid + "_" + xcat for xcat in xcats for cid in cids]
        metrics = ["value", "grading"]
        tkms = [f"{ticker}_{metric}" for ticker in tickers for metric in metrics]
        expressions = construct_expressions(
            metrics=["value", "grading"], tickers=tickers
        )
        deconstructed_expressions = deconstruct_expression(expression=expressions)
        dtkms = ["_".join(d) for d in deconstructed_expressions]

        self.assertEqual(set(tkms), set(dtkms))

        for tkm, expression in zip(tkms, expressions):
            self.assertEqual(
                tkm,
                "_".join(deconstruct_expression(expression=expression)),
            )

        for expr in [1, [1, 2]]:
            # type error
            with self.assertRaises(TypeError):
                deconstruct_expression(expression=expr)

        with self.assertRaises(ValueError):
            deconstruct_expression(expression=[])

        # now give it a bad expression. it should warn and return [expression, expression, expression]
        with self.assertWarns(UserWarning):
            self.assertEqual(
                deconstruct_expression(expression="bad_expression"),
                ["bad_expression", "bad_expression", "value"],
            )

        with self.assertWarns(UserWarning):
            deconstruct_expression(expression="badexpression,metric")

    def test_get_expression_from_qdf(self):
        dicts_list = mock_request_wrapper(
            dq_expressions=[self.expressions[0]],
            start_date="2019-01-01",
            end_date="2019-01-31",
        )
        qdf: pd.DataFrame = timeseries_to_qdf(dicts_list[0])
        expr: str = get_expression_from_qdf(qdf)
        self.assertEqual(expr, [self.expressions[0]])

        qdfs = list(map(timeseries_to_qdf, dicts_list))
        exprs = get_expression_from_qdf(qdfs)
        self.assertIsInstance(exprs, list)
        for expr, _qdf in zip(exprs, qdfs):
            self.assertEqual(expr, get_expression_from_qdf(_qdf)[0])

    def test_get_expression_from_wide_df(self):
        exprs = self.expressions
        wdf = concat_column_dfs(
            [
                timeseries_to_column(
                    mock_request_wrapper(
                        dq_expressions=[expr],
                        start_date="2019-01-01",
                        end_date="2019-01-31",
                    )[0]
                )
                for expr in exprs
            ]
        )
        self.assertEqual(set(get_expression_from_wide_df(wdf)), set(exprs))

        wList = [wdf, wdf]
        self.assertEqual(set(get_expression_from_wide_df(wList)), set(exprs))

    def test_timeseries_to_qdf(self):
        with self.assertRaises(TypeError):
            timeseries_to_qdf("")

        dicts_list = mock_request_wrapper(
            dq_expressions=self.expressions,
            start_date="2019-01-01",
            end_date="2019-01-31",
        )

        def _test_qdf(_qdf: pd.DataFrame, expr: str):
            self.assertIsInstance(_qdf, QuantamentalDataFrame)
            cid = _qdf["cid"].unique()
            xcat = _qdf["xcat"].unique()
            metrics = list(set(_qdf.columns) - set(QuantamentalDataFrame.IndexCols))
            self.assertEqual(len(cid), 1)
            self.assertEqual(len(xcat), 1)
            self.assertEqual(len(metrics), 1)
            self.assertEqual(expr, f"DB(JPMAQS,{cid[0]}_{xcat[0]},{metrics[0]})")

        for dictx in dicts_list:
            expression = dictx["attributes"][0]["expression"]
            _test_qdf(timeseries_to_qdf(dictx), expression)

        for mts in [None, [(None, None)]]:
            dicts_list[0]["attributes"][0]["time-series"] = mts
            self.assertIsNone(timeseries_to_qdf(dicts_list[0]))

    def test_timeseries_to_column(self):

        with self.assertRaises(TypeError):
            timeseries_to_column("")

        with self.assertRaises(ValueError):
            timeseries_to_column({}, errors=True)

        dicts_list = mock_request_wrapper(
            dq_expressions=self.expressions,
            start_date="2019-01-01",
            end_date="2019-01-31",
        )

        ts_values = [None, [(None, None)]]

        for ts in ts_values:
            bad_ts = mock_request_wrapper(
                self.expressions[0], "2019-01-01", "2019-01-31"
            )[0]
            bad_ts["attributes"][0]["time-series"] = ts
            with self.assertRaises(ValueError):
                timeseries_to_column(bad_ts, errors="raise")

            self.assertIsNone(timeseries_to_column(bad_ts, errors="ignore"))

        def _test_column(_column: pd.DataFrame, expr: str):
            self.assertIsInstance(_column, pd.DataFrame)
            self.assertEqual(_column.index.name, "real_date")
            self.assertEqual(len(_column.columns), 1)
            self.assertEqual(_column.columns[0], expr)

        for dictx in dicts_list:
            expression = dictx["attributes"][0]["expression"]
            _test_column(timeseries_to_column(dictx), expression)

    def test_concat_column_dfs(self):
        dicts_list = mock_request_wrapper(
            dq_expressions=self.expressions,
            start_date="2019-01-01",
            end_date="2019-01-31",
        )
        ts_cols = [timeseries_to_column(dicts) for dicts in dicts_list]
        wdf = concat_column_dfs(ts_cols)

        self.assertIsInstance(wdf, pd.DataFrame)
        self.assertEqual(set(wdf.columns), set(self.expressions))

        # test that there is an index called "real_date"
        self.assertEqual(wdf.index.name, "real_date")

        # test errors
        with self.assertRaises(TypeError):
            concat_column_dfs({})

        with self.assertRaises(ValueError):
            concat_column_dfs(ts_cols, errors=True)

        bad_ts_cols = list(map(timeseries_to_column, dicts_list))
        bad_ts_cols[0] = None

        with self.assertRaises(TypeError):
            concat_column_dfs(bad_ts_cols, errors="raise")

        _exprs = [ts["attributes"][0]["expression"] for ts in dicts_list if ts]
        _exprs.pop(0)
        wdf = concat_column_dfs(bad_ts_cols, errors="ignore")
        self.assertEqual(set(wdf.columns), set(_exprs))

    def test_validate_downloaded_df(self):
        start_date = "1989-12-01"
        end_date = "1995-01-31"
        # generate for all expressions
        dicts_list = mock_request_wrapper(
            dq_expressions=self.expressions,
            start_date=start_date,
            end_date="1994-01-31",
        )

        missing_exprs = [d["attributes"][0]["expression"] for d in dicts_list[:3]]
        for i in range(3):
            dicts_list[i]["attributes"][0]["time-series"] = None
        data_df = concat_single_metric_qdfs(list(map(timeseries_to_qdf, dicts_list)))

        found_expressions = get_expression_from_qdf(data_df)

        with self.assertRaises(InvalidDataframeError):
            validate_downloaded_df(
                data_df="",
                expected_expressions=self.expressions,
                found_expressions=found_expressions,
                start_date=start_date,
                end_date=end_date,
            )

        self.assertFalse(
            validate_downloaded_df(
                data_df=pd.DataFrame(),
                expected_expressions=self.expressions,
                found_expressions=found_expressions,
                start_date=start_date,
                end_date=end_date,
            )
        )

        validate_downloaded_df(
            data_df=data_df,
            expected_expressions=self.expressions,
            found_expressions=found_expressions,
            start_date=start_date,
            end_date=end_date,
        )

        with self.assertRaises(InvalidDataframeError):
            validate_downloaded_df(
                data_df=data_df,
                expected_expressions=missing_exprs,
                found_expressions=found_expressions + [random_string()],
                start_date=start_date,
                end_date=end_date,
            )

    def test_check_attributes_in_sync(self):
        ts_list = [{"attributes": [{"expression": "DB(JPMAQS,CID_TEST,value)", "time-series": [["20240125", 0.1], ["20240126", -0.1], ["20240127", None], ["20240128", None], ["20240129", 0.8], ["20240130", -0.01], ["20240131", 1.5], ["20240201", 1.0], ["20240202", 1.0], ["20240203", None], ["20240204", None], ["20240205", 1.0]]}]}, {"attributes": [{"expression": "DB(JPMAQS,CID_TEST,grading)", "time-series": [["20240125", 1.0], ["20240126", 1.0], ["20240127", None], ["20240128", None], ["20240129", 1.0], ["20240130", 1.0], ["20240131", 1.0], ["20240201", 1.0], ["20240202", 1.0], ["20240203", None], ["20240204", None], ["20240205", 1.0]]}]}, {"attributes": [{"expression": "DB(JPMAQS,CID_TEST,mop_lag)", "time-series": [["20240125", 0.0], ["20240126", 0.0], ["20240127", None], ["20240128", None], ["20240129", 0.0], ["20240130", 0.0], ["20240131", 0.0], ["20240201", 0.0], ["20240202", 0.0], ["20240203", None], ["20240204", None], ["20240205", 0.0]]}]}, {"attributes": [{"expression": "DB(JPMAQS,CID_TEST,eop_lag)", "time-series": [["20240125", 0.0], ["20240126", 0.0], ["20240127", None], ["20240128", None], ["20240129", 0.0], ["20240130", 0.0], ["20240131", 0.0], ["20240201", 0.0], ["20240202", 0.0], ["20240203", None], ["20240204", None], ["20240205", 0.0]]}]}]
        self.assertTrue(check_attributes_in_sync(ts_list))

        out_of_sync_ts_list = [{"attributes": [{"expression": "DB(JPMAQS,CID_TEST,value)", "time-series": [["20240125", 0.1], ["20240126", -0.1], ["20240127", None], ["20240128", None], ["20240129", 0.8], ["20240130", -0.01], ["20240131", 1.5], ["20240201", 1.0], ["20240202", 1.0], ["20240203", None], ["20240204", None], ["20240205", None]]}]}, {"attributes": [{"expression": "DB(JPMAQS,CID_TEST,grading)", "time-series": [["20240125", 1.0], ["20240126", 1.0], ["20240127", None], ["20240128", None], ["20240129", 1.0], ["20240130", 1.0], ["20240131", 1.0], ["20240201", 1.0], ["20240202", 1.0], ["20240203", None], ["20240204", None], ["20240205", 1.0]]}]}, {"attributes": [{"expression": "DB(JPMAQS,CID_TEST,mop_lag)", "time-series": [["20240125", 0.0], ["20240126", 0.0], ["20240127", None], ["20240128", None], ["20240129", 0.0], ["20240130", 0.0], ["20240131", 0.0], ["20240201", 0.0], ["20240202", 0.0], ["20240203", None], ["20240204", None], ["20240205", 0.0]]}]}, {"attributes": [{"expression": "DB(JPMAQS,CID_TEST,eop_lag)", "time-series": [["20240125", 0.0], ["20240126", 0.0], ["20240127", None], ["20240128", None], ["20240129", 0.0], ["20240130", 0.0], ["20240131", 0.0], ["20240201", 0.0], ["20240202", 0.0], ["20240203", None], ["20240204", None], ["20240205", 0.0]]}]}]
        self.assertFalse(check_attributes_in_sync(out_of_sync_ts_list))


if __name__ == "__main__":
    unittest.main()
