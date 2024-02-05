import unittest
import warnings
import pandas as pd

from typing import List, Dict, Any
from macrosynergy.download import JPMaQSDownload

from macrosynergy.download.exceptions import InvalidDataframeError
from .mock_helpers import (
    mock_jpmaqs_value,
    mock_request_wrapper,
    random_string,
    MockDataQueryInterface,
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
        warnings.simplefilter("always")

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
        }

        jpmaqs: JPMaQSDownload = JPMaQSDownload(
            client_id="client_id",
            client_secret="client_secret",
            check_connection=False,
        )

        config: dict = dict(
            client_id="client_id",
            client_secret="client_secret",
        )

        mock_dq_interface: MockDataQueryInterface = MockDataQueryInterface(**config)
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

        jpmaqs: JPMaQSDownload = JPMaQSDownload(
            client_id="client_id",
            client_secret="client_secret",
            check_connection=False,
        )

        config: dict = dict(
            client_id="client_id",
            client_secret="client_secret",
        )

        mock_dq_interface: MockDataQueryInterface = MockDataQueryInterface(**config)
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

    warnings.resetwarnings()

    def test_validate_downloaded_df(self):
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
            "get_catalogue": False,
        }

        jpmaqs: JPMaQSDownload = JPMaQSDownload(
            client_id="client_id",
            client_secret="client_secret",
            check_connection=False,
        )

        mock_dq_interface: MockDataQueryInterface = MockDataQueryInterface(
            client_id="client_id", client_secret="client_secret"
        )
        un_avail_exprs: List[str] = [
            "DB(JPMAQS,USD_FXXR_NSA,value)",
            "DB(JPMAQS,USD_FXXR_NSA,grading)",
            "DB(JPMAQS,USD_FXXR_NSA,eop_lag)",
            "DB(JPMAQS,USD_FXXR_NSA,mop_lag)",
        ]
        dupl_exprs: List[str] = [
            "DB(JPMAQS,EUR_EQXR_NSA,value)",
        ]
        mock_dq_interface._gen_attributes(
            unavailable_expressions=un_avail_exprs,
            mask_expressions=un_avail_exprs,
            duplicate_entries=dupl_exprs,
        )
        jpmaqs.dq_interface = mock_dq_interface
        # with self.assertRaises(InvalidDataframeError):
        #     jpmaqs.download(**good_args)

        mock_dq_interface._gen_attributes(
            unavailable_expressions=un_avail_exprs,
            mask_expressions=un_avail_exprs,
            duplicate_entries=[],
        )
        jpmaqs.dq_interface = mock_dq_interface

        with self.assertRaises(InvalidDataframeError):
            jpmaqs.download(expressions=un_avail_exprs)


if __name__ == "__main__":
    unittest.main()
