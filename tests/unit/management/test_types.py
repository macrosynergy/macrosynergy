import unittest
import numpy as np
import pandas as pd

from typing import Any, List
from macrosynergy.management.types import (
    NoneType,
    QuantamentalDataFrame,
    SubscriptableMeta,
    ArgValidationMeta,
)
from macrosynergy.management.simulate import make_test_df


class TestTypes(unittest.TestCase):

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

        # Check if subclassing works as expected
        df: pd.DataFrame = test_df.copy()

        dfx: QuantamentalDataFrame = QuantamentalDataFrame(df)

        self.assertTrue(isinstance(dfx, QuantamentalDataFrame))
        self.assertTrue(isinstance(dfx, pd.DataFrame))

        df_Q = (
            QuantamentalDataFrame(df)
            .sort_values(["cid", "xcat", "real_date"])
            .reset_index(drop=True)
        )
        df_S = df.sort_values(["cid", "xcat", "real_date"]).reset_index(drop=True)
        self.assertTrue((df_S == df_Q).all().all())

        # test with categorical=False
        df_Q = (
            QuantamentalDataFrame(df, categorical=False)
            .sort_values(["cid", "xcat", "real_date"])
            .reset_index(drop=True)
        )
        df_S = df.sort_values(["cid", "xcat", "real_date"]).reset_index(drop=True)
        self.assertTrue(df_Q.equals(df_S))


class TestSubscriptableMeta(unittest.TestCase):
    class Sample(metaclass=SubscriptableMeta):
        @staticmethod
        def foo():
            return "bar"

        @staticmethod
        def hello():
            return "world"

        @classmethod
        def class_method(cls):
            return "class method"

        def instance_method(self):
            return "instance method"

    def test_valid_method_subscription(self):
        self.assertEqual(self.Sample["foo"](), "bar")
        self.assertEqual(self.Sample["hello"](), "world")
        self.assertEqual(self.Sample["class_method"](), "class method")

    def test_invalid_method_subscription(self):
        with self.assertRaises(KeyError):
            self.Sample["non_existent_method"]

    def test_instance_method_subscription(self):
        self.Sample["instance_method"]


class TestArgValidationMeta(unittest.TestCase):
    class Sample(metaclass=ArgValidationMeta):
        def greet(self, name: str):
            return f"Hello, {name}!"

        def add(self, a: int, b: int) -> int:
            return a + b

    def setUp(self):
        self.sample = self.Sample()

    def test_method_with_arg_validation(self):
        self.assertEqual(self.sample.greet("Alice"), "Hello, Alice!")
        with self.assertRaises(TypeError):
            self.sample.greet(99)

    def test_add_method_with_arg_validation(self):
        self.assertEqual(self.sample.add(3, 4), 7)
        with self.assertRaises(TypeError):
            self.sample.add(3, "4")


if __name__ == "__main__":
    unittest.main()
