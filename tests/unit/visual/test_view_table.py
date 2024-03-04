import unittest
import pandas as pd
from typing import List, Dict, Any
from macrosynergy.management.simulate import make_test_df
import macrosynergy.visuals as msv
import matplotlib
import matplotlib.pyplot as plt
from unittest.mock import patch


class TestAll(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        # Prevents plots from being displayed during tests.
        self.mpl_backend: str = matplotlib.get_backend()
        matplotlib.use("Agg")
        self.mock_show = patch("matplotlib.pyplot.show").start()

        data = {
            "A": [1, 2, 3, 4],
            "B": [5, 6, 7, 8],
            "C": [9, 10, 11, 12],
            "D": [13, 14, 15, 16],
        }
        row_labels = ["Row1", "Row2", "Row3", "Row4"]

        self.df = pd.DataFrame(data, index=row_labels)

    @classmethod
    def tearDownClass(self) -> None:
        plt.close("all")
        patch.stopall()
        matplotlib.use(self.mpl_backend)

    def setUp(self):
        self.valid_args: Dict[str, Any] = {
            "df": self.df,
            "title": "test",
            "figsize": (10, 10),
            "min_color": 0.0,
            "max_color": 1.0,
            "xlabel": "X",
            "ylabel": "Y",
            "xticklabels": None,
            "yticklabels": None,
            "annot": True,
        }

    def tearDown(self) -> None:
        plt.close("all")

    def test_view_table_no_error(self):
        try:
            msv.view_table(**self.valid_args)
        except Exception as e:
            self.fail(f"view_table raised {e} unexpectedly")

        self.valid_args["title"] = None
        self.valid_args["figsize"] = None
        self.valid_args["min_color"] = None
        self.valid_args["max_color"] = None
        self.valid_args["xlabel"] = None
        self.valid_args["ylabel"] = None
        self.valid_args["annot"] = False
        try:
            msv.view_table(**self.valid_args)
        except Exception as e:
            self.fail(f"view_table raised {e} unexpectedly")

    def test_view_table_invalid_df(self):
        data = {
            "A": ["A", "B", "C", "D"],
        }
        row_labels = ["Row1", "Row2", "Row3", "Row4"]

        df = pd.DataFrame(data, index=row_labels)
        invalid_args: Dict[str, Any] = self.valid_args.copy()

        invalid_args["df"] = df
        with self.assertRaises(ValueError):
            msv.view_table(**invalid_args)

        invalid_args["df"] = pd.DataFrame()
        with self.assertRaises(ValueError):
            msv.view_table(**invalid_args)

        invalid_args["df"] = "INVALID_TYPE"
        with self.assertRaises(TypeError):
            msv.view_table(**invalid_args)

    def test_view_table_valid_tick_labels(self):
        self.valid_args["xticklabels"] = ["A", "B", "C", "D"]
        self.valid_args["yticklabels"] = ["Row1", "Row2", "Row3", "Row4"]

        try:
            msv.view_table(**self.valid_args)
        except Exception as e:
            self.fail(f"view_table raised {e} unexpectedly")

    def test_view_table_invalid_tick_labels(self):
        args = self.valid_args.copy()
        args["xticklabels"] = ["A", "B", "C", "D", "EXTRA_COL"]
        args["yticklabels"] = None
        with self.assertRaises(ValueError):
            msv.view_table(**args)

        args["xticklabels"] = None
        args["yticklabels"] = ["Row1", "Row2", "Row3", "Row4", "EXTRA_ROW"]
        with self.assertRaises(ValueError):
            msv.view_table(**args)


if __name__ == "__main__":
    unittest.main()
