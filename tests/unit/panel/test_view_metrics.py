import unittest
import pandas as pd
from typing import List, Dict, Any
from macrosynergy.management.simulate_quantamental_data import make_test_df
from macrosynergy.panel import view_metrics
import matplotlib


class TestAll(unittest.TestCase):
    def dataframe_construction(self):
        self.cids: List[str] = ["AUD", "CAD", "GBP", "NZD"]
        self.xcats: List[str] = ["XR", "CRY", "INFL"]
        self.metrics: List[str] = ["value", "grading", "eop_lag", "mop_lag"]
        idx_cols: List[str] = ["cid", "xcat", "real_date"]
        self.start: str = "2010-01-01"
        self.end: str = "2020-12-31"

        df: pd.DataFrame = make_test_df(self.cids, self.xcats, self.start, self.end)
        for mtr in self.metrics:
            df: pd.DataFrame = df.merge(
                make_test_df(self.cids, self.xcats, self.start, self.end).rename(
                    columns={"value": mtr}
                ),
                on=idx_cols,
            )

        self.df: pd.DataFrame = df

    def test_view_metrics(self):
        self.dataframe_construction()

        mpl_backend: str = matplotlib.get_backend()
        matplotlib.use("Agg")

        good_args: Dict[str, Any] = {
            "df": self.df,
            "xcat": "XR",
            "cids": self.cids,
            "metric": "eop_lag",
            "start": self.start,
            "end": self.end,
            "freq": "M",
            "agg": "mean",
            "title": "Test plot",
            "figsize": (10, None),
        }
        # test simple good args
        try:
            view_metrics(**good_args)
        except Exception as e:
            self.fail(f"view_metrics raised {e} unexpectedly")

        # test type errors
        for arg in good_args:
            bad_args: Dict[str, Any] = good_args.copy()
            bad_args[arg] = 1
            with self.assertRaises(TypeError):
                view_metrics(**bad_args)

        bad_args: Dict[str, Any] = good_args.copy()

        # test bad xcat
        bad_args["xcat"] = "bad_xcat"
        with self.assertRaises(ValueError):
            try:
                view_metrics(**bad_args)
            except Exception as e:
                self.assertIsInstance(e, ValueError)
                raise ValueError(e)

        # test bad metric
        bad_args = good_args.copy()
        bad_args["metric"] = "bad_metric"
        with self.assertRaises(ValueError):
            view_metrics(**bad_args)

        # test with empty df
        bad_args = good_args.copy()
        bad_cases = [pd.DataFrame(), self.df.rename(columns={"eop_lag": "bad_metric"})]

        bad_args["df"] = pd.DataFrame()

        for case in bad_cases:
            bad_args["df"] = case
            with self.assertRaises(ValueError):
                view_metrics(**bad_args)

        # test cids with empty list and list of int
        bad_args = good_args.copy()
        bad_cases = [
            [1, 2, 3],
        ]
        for case in bad_cases:
            bad_args["cids"] = case
            with self.assertRaises(TypeError):
                view_metrics(**bad_args)

        # test bad start and end
        bad_args = good_args.copy()
        for arg in ["start", "end"]:
            bad_args[arg] = "bad_date"
            with self.assertRaises(ValueError):
                view_metrics(**bad_args)

        # test freq
        bad_args = good_args.copy()
        bad_args["freq"] = "z"
        with self.assertRaises(ValueError):
            view_metrics(**bad_args)

        bad_args["freq"] = "m"
        try:
            view_metrics(**bad_args)
        except Exception as e:
            self.fail(f"view_metrics raised {e} unexpectedly")

        # test agg
        bad_args = good_args.copy()
        bad_args["agg"] = "bad_agg"
        with self.assertRaises(ValueError):
            view_metrics(**bad_args)

        bad_args["agg"] = "MEAN"
        try:
            view_metrics(**bad_args)
        except Exception as e:
            self.fail(f"view_metrics raised {e} unexpectedly")

        # test if the figsize is a tuple
        bad_args = good_args.copy()
        bad_cases: List[Any] = [
            ["apple", 10],
            (10, "apple"),
            [None, 10],
        ]
        for case in bad_cases:
            bad_args["figsize"] = case
            with self.assertRaises(ValueError):
                view_metrics(**bad_args)

        # pass a list of 3 ints as figsize. should raise a type error
        bad_args["figsize"] = [10, 10, 10]
        with self.assertRaises(TypeError):
            view_metrics(**bad_args)

        # give it one where the first one is int, second one is none. should work
        bad_args["figsize"] = [10, None]
        try:
            view_metrics(**bad_args)
        except Exception as e:
            self.fail(f"view_metrics raised {e} unexpectedly")

        rp_args: List[Dict[str, Any]] = ["cids", "start", "end", "title"]
        for rp_arg in rp_args:
            bad_args = good_args.copy()
            bad_args[rp_arg] = None
            try:
                view_metrics(**bad_args)
            except Exception as e:
                self.fail(f"view_metrics raised {e} unexpectedly")

        matplotlib.use(mpl_backend)


if __name__ == "__main__":
    unittest.main()