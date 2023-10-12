import unittest
import numpy as np
import pandas as pd

from typing import Any, List
from macrosynergy.management.types import Numeric, NoneType, QuantamentalDataFrame
from macrosynergy.management.simulate_quantamental_data import make_test_df


class TestTypes(unittest.TestCase):
    def test_numeric_type(self):
        true_cases: List[Any] = [1, 1.0, np.int64(1), np.float64(1)]

        for case in true_cases:
            self.assertTrue(isinstance(case, Numeric), msg=f"Failed for {case}")

        false_cases: List[Any] = [
            "1",
            None,
            object(),
            pd.DataFrame(),
            pd.Series(dtype=int),
        ]

        for case in false_cases:
            self.assertFalse(isinstance(case, Numeric), msg=f"Failed for {case}")

    def test_none_type(self):
        self.assertTrue(isinstance(None, NoneType))

        false_cases: List[Any] = [
            1,
            1.0,
            np.int64(1),
            np.float64(1),
            "1",
            object(),
            pd.DataFrame(),
            pd.Series(dtype=int),
        ]

        for case in false_cases:
            self.assertFalse(isinstance(case, NoneType), msg=f"Failed for {case}")

    def test_quantamental_dataframe_type(self):
        test_df: pd.DataFrame = make_test_df()
        self.assertTrue(isinstance(test_df, QuantamentalDataFrame))
        # rename xcat to xkat
        df: pd.DataFrame = test_df.copy()
        df.columns = df.columns.str.replace("xcat", "xkat")
        self.assertFalse(isinstance(df, QuantamentalDataFrame))

        # rename cid to xid
        df: pd.DataFrame = test_df.copy()
        df.columns = df.columns.str.replace("cid", "xid")
        self.assertFalse(isinstance(df, QuantamentalDataFrame))

        # change one date to a string
        df: pd.DataFrame = test_df.copy()
        self.assertTrue(isinstance(df, QuantamentalDataFrame))

        # change one date to a string -- remember the caveats for pd arrays
        df: pd.DataFrame = test_df.copy()
        nseries: List[pd.Timestamp] = df["real_date"].tolist()
        nseries[0] = "2020-01-01"
        df["real_date"] = pd.Series(nseries, dtype=object).copy()
        self.assertFalse(isinstance(df, QuantamentalDataFrame))


if __name__ == "__main__":
    unittest.main()
