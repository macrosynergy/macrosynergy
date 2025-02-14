import datetime
import unittest

import numpy as np
import pandas as pd
from parameterized import parameterized

from macrosynergy.management.simulate import make_qdf
from macrosynergy.panel import extend_history


class TestExtendHistory(unittest.TestCase):

    @classmethod
    def setUpClass(self):

        xcats = ["INFL1", "INFL2", "INFL3"]
        cids = ["GBP", "CAD"]
        end_date = pd.Timestamp.now()

        date_ranges = {
            "INFL1": pd.date_range(start="2015-01-01", end=end_date, freq="D"),
            "INFL2": pd.date_range(start="2010-01-01", end=end_date, freq="D"),
            "INFL3": pd.date_range(start="2005-01-01", end=end_date, freq="D"),
        }

        # Create the DataFrame
        data = []
        for xcat, dates in date_ranges.items():
            for cid in cids:
                for date in dates:
                    value = int(xcat[-1])
                    data.append(
                        {"real_date": date, "xcat": xcat, "cid": cid, "value": value}
                    )

        self.df = pd.DataFrame(data)

    def setUp(self):
        self.valid_args = {
            "df": self.df,
            "new_xcat": "TEST",
            "cids": ["GBP", "CAD"],
            "hierarchy": ["INFL1", "INFL2"],
            "backfill": True,
            "start": "1995-01-01",
        }

    def test_valid_args(self):
        try:
            extend_history(**self.valid_args)
        except Exception as e:
            self.fail(f"extend_history raised {e} unexpectedly")

    def test_invalid_args(self):

        type_error_args = {
            "df": 1,
            "new_xcat": 1,
            "cids": 1,
            "hierarchy": 1,
            "backfill": 1,
            "start": 1,
        }

        for key, value in type_error_args.items():
            with self.assertRaises(TypeError):
                invalid_args = self.valid_args.copy()
                invalid_args[key] = value
                extend_history(**invalid_args)

    def test_backfill_start_args(self):
        with self.assertRaises(ValueError):
            invalid_args = self.valid_args.copy()
            invalid_args["backfill"] = True
            invalid_args["start"] = None
            extend_history(**invalid_args)

        try:
            valid_args = self.valid_args.copy()
            valid_args["backfill"] = True
            valid_args["start"] = "1995-01-01"
            extend_history(**valid_args)
        except Exception as e:
            self.fail(f"extend_history raised {e} unexpectedly")

        try:
            valid_args = self.valid_args.copy()
            valid_args["backfill"] = False
            valid_args["start"] = None
            extend_history(**valid_args)
        except Exception as e:
            self.fail(f"extend_history raised {e} unexpectedly")

    def test_invalid_cids(self):
        with self.assertRaises(TypeError):
            invalid_args = self.valid_args.copy()
            invalid_args["cids"] = [1, 2, 3]
            extend_history(**invalid_args)

        with self.assertRaises(ValueError):
            invalid_args = self.valid_args.copy()
            invalid_args["cids"] = ["bad_cid"]
            extend_history(**invalid_args)

    def test_extend_history_one_xcat(self):

        df = self.df
        new_xcat = "NEW_XCAT"
        cids = ["GBP", "CAD"]
        hierarchy = ["INFL1"]
        backfill = True
        start = "1995-01-02"

        result_df = extend_history(df, new_xcat, cids, hierarchy, backfill, start)

        self._check_result_df(result_df, new_xcat, cids, start)

        for cid in cids:
            cid_df = df[(df["cid"] == cid) & (df["xcat"] == hierarchy[0])]
            result_cid_df = result_df[
                (result_df["cid"] == cid) & (result_df["xcat"] == hierarchy[0])
            ]
            backfill_value = cid_df.loc[cid_df["real_date"].idxmin(), "value"]
            self.assertTrue(
                (
                    result_cid_df.loc[
                        result_cid_df["real_date"] < cid_df["real_date"].min(), "value"
                    ]
                    == backfill_value
                ).all(),
                f"Backfill values for cid {cid} are not consistent.",
            )

    @parameterized.expand(
        [
            (["INFL1", "INFL2", "INFL3"],),
            (["INFL1", "INVALID", "INFL3"],),
        ]
    )
    def test_extend_history_multiple_xcats(self, hierarchy):

        df = self.df
        new_xcat = "NEW_XCAT"
        cids = ["GBP", "CAD"]
        backfill = True
        start = "1995-01-02"

        result_df = extend_history(df, new_xcat, cids, hierarchy, backfill, start)

        self._check_result_df(result_df, new_xcat, cids, start)

        for cid in cids:
            cid_df3 = df[(df["cid"] == cid) & (df["xcat"] == hierarchy[2])]
            result_cid_df = result_df[(result_df["cid"] == cid)]

            backfill_value = cid_df3.loc[cid_df3["real_date"].idxmin(), "value"]
            self.assertTrue(
                (
                    result_cid_df.loc[
                        result_cid_df["real_date"] < cid_df3["real_date"].min(), "value"
                    ]
                    == backfill_value
                ).all(),
                f"Backfill values for cid {cid} are not consistent.",
            )

    def test_extend_history_no_backfill(self):

        df = self.df
        new_xcat = "NEW_XCAT"
        cids = ["GBP", "CAD"]
        hierarchy = ["INFL1", "INFL2", "INFL3"]
        backfill = False
        start = "1995-01-02"

        result_df = extend_history(df, new_xcat, cids, hierarchy, backfill, start)

        self._check_result_df(result_df, new_xcat, cids, start=None)

        self.assertTrue(
            set(result_df["real_date"].unique()) == set(df["real_date"].unique()),
            "The real_date column is not consistent with the original DataFrame.",
        )

        for cid in cids:
            cid_df1 = df[(df["cid"] == cid) & (df["xcat"] == hierarchy[0])]
            cid_df2 = df[(df["cid"] == cid) & (df["xcat"] == hierarchy[1])]
            cid_df3 = df[(df["cid"] == cid) & (df["xcat"] == hierarchy[2])]
            result_cid_df = result_df[(result_df["cid"] == cid)]

            backfill_value2 = cid_df2.loc[cid_df2["real_date"].idxmin(), "value"]
            backfill_value = cid_df3.loc[cid_df3["real_date"].idxmin(), "value"]
            self.assertTrue(
                (
                    result_cid_df.loc[
                        (
                            (result_cid_df["real_date"] < cid_df1["real_date"].min())
                            & (result_cid_df["real_date"] > cid_df3["real_date"].max())
                        ),
                        "value",
                    ]
                    == backfill_value2
                ).all(),
                f"Backfill values for cid {cid} are not consistent.",
            )
            self.assertTrue(
                (
                    result_cid_df.loc[
                        result_cid_df["real_date"] < cid_df3["real_date"].min(), "value"
                    ]
                    == backfill_value
                ).all(),
                f"Backfill values for cid {cid} are not consistent.",
            )

    def test_extend_history_missing_cid(self):

        df = self.df
        new_xcat = "NEW_XCAT"
        missing_cid = "AUD"
        cids = ["GBP", "CAD", missing_cid]
        hierarchy = ["INFL1", "INFL2", "INFL3"]
        backfill = True
        start = "1995-01-02"

        result_df = extend_history(df, new_xcat, cids, hierarchy, backfill, start)

        self._check_result_df(result_df, new_xcat, ["GBP", "CAD"], start)

        self.assertTrue(
            "AUD" not in result_df["cid"].unique(),
            "Missing cids are present in the result.",
        )

    def test_extend_history_no_cids(self):

        df = self.df
        new_xcat = "NEW_XCAT"
        cids = None
        hierarchy = ["INFL1", "INFL2", "INFL3"]
        backfill = True
        start = "1995-01-02"

        result_df = extend_history(df, new_xcat, cids, hierarchy, backfill, start)

        self._check_result_df(result_df, new_xcat, ["GBP", "CAD"], start)

    def _check_result_df(self, result_df, new_xcat, cids, start):
        expected_columns = {"real_date", "xcat", "cid", "value"}
        self.assertTrue(
            set(result_df.columns) == expected_columns,
            "Result DataFrame columns do not match expected columns.",
        )

        self.assertTrue(
            set(result_df["xcat"].unique()) == {new_xcat},
            "xcat column does not contain the expected value.",
        )

        self.assertTrue(
            set(result_df["cid"].unique()) == set(cids),
            "cid column does not contain the expected cids.",
        )

        if start is not None:
            min_date = pd.to_datetime(start)
            self.assertTrue(
                result_df["real_date"].min() == min_date,
                "The history is not extended to the specified start date.",
            )
