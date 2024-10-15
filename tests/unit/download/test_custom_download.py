import unittest
from unittest.mock import patch
from macrosynergy.download import custom_download
import pandas as pd


def download_func(expressions, startDate=None, endDate=None):
    data = {}
    for expr in expressions:
        ticker, metric = expr.split(",")[1], expr.split(",")[2].strip(")")
        if metric == "value":
            data[expr] = [100, 200, 300]
        elif metric == "grading":
            data[expr] = [1, 2, 3]

    date_index = pd.date_range(start="2020-01-01", periods=3)
    return pd.DataFrame(data, index=date_index)


class TestCustomDownload(unittest.TestCase):
    def test_custom_download_basic(self):
        # Inputs to the function
        tickers = ["CAD_XR", "EUR_XR"]
        metrics = ["value", "grading"]
        start_date = "2020-01-01"
        end_date = "2020-01-03"

        result_df = custom_download(
            tickers, download_func, metrics, start_date, end_date
        )

        self.assertTrue("grading" in result_df.columns)
        self.assertTrue("value" in result_df.columns)


if __name__ == "__main__":
    unittest.main()
