import unittest
import pandas as pd
from typing import List, Tuple, Optional, Dict
from tests.simulate import make_qdf
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
        pass


if __name__ == "__main__":
    unittest.main()
