import unittest
import random
import numpy as np
import pandas as pd
import os
from typing import List
from macrosynergy.management.simulate import VintageData


class Test_All(unittest.TestCase):
    def setUp(self):
        self.vins_m = VintageData(
            "USD_INDX_SA",
            cutoff="2019-06-30",
            release_lags=[3, 20, 25],
            number_firsts=12,
            shortest=12,
            sd_ar=5,
            trend_ar=20,
            seasonal=10,
            added_dates=6,
        )

    def test_grade1(self):
        grade1: pd.DataFrame = self.vins_m.make_grade1()
        expected_columns: List[str] = [
            "cross_section",
            "category_code",
            "adjustment",
            "transformation",
            "release_date",
            "observation_date",
            "value",
            "grading",
        ]
        self.assertEqual(list(grade1.columns), expected_columns)

        # check that grade1[cross_section] is a single value - USD
        unique_cross_section = grade1["cross_section"].unique().tolist()
        self.assertEqual(len(unique_cross_section), 1)
        self.assertEqual(unique_cross_section[0], "USD")

        unique_category_code = grade1["category_code"].unique().tolist()
        self.assertEqual(len(unique_category_code), 1)
        self.assertEqual(unique_category_code[0], "INDX")

        unique_adjustment = grade1["adjustment"].unique().tolist()
        self.assertEqual(len(unique_adjustment), 1)
        self.assertEqual(unique_adjustment[0], "SA")

        unique_transformation = grade1["transformation"].unique().tolist()
        self.assertEqual(len(unique_transformation), 1)
        self.assertEqual(unique_transformation[0], None)

        min_release_date = grade1["release_date"].min().strftime("%Y-%m-%d")
        max_release_date = grade1["release_date"].max().strftime("%Y-%m-%d")
        self.assertEqual(min_release_date, "2018-07-03")
        self.assertEqual(max_release_date, "2019-06-26")

        # grading should be 1
        unique_grading = grade1["grading"].unique().tolist()
        self.assertEqual(len(unique_grading), 1)
        self.assertAlmostEqual(unique_grading[0], 1.0)

    def test_grade2(self):
        dfm2: pd.DataFrame = self.vins_m.make_grade2()

        # earlist release date - 2018-01-01, latest release date - 2019-06-25
        min_release_date = dfm2["release_date"].min().strftime("%Y-%m-%d")
        max_release_date = dfm2["release_date"].max().strftime("%Y-%m-%d")
        self.assertEqual(min_release_date, "2018-01-01")
        self.assertEqual(max_release_date, "2019-06-25")

        # check that grade2[cross_section] is a single value - USD
        unique_cross_section = dfm2["cross_section"].unique().tolist()
        self.assertEqual(len(unique_cross_section), 1)
        self.assertEqual(unique_cross_section[0], "USD")

        unique_category_code = dfm2["category_code"].unique().tolist()
        self.assertEqual(len(unique_category_code), 1)
        self.assertEqual(unique_category_code[0], "INDX")

        unique_adjustment = dfm2["adjustment"].unique().tolist()
        self.assertEqual(len(unique_adjustment), 1)
        self.assertEqual(unique_adjustment[0], "SA")

        unique_transformation = dfm2["transformation"].unique().tolist()
        self.assertEqual(len(unique_transformation), 1)
        self.assertEqual(unique_transformation[0], None)

        # ticker should be a single value - USD_INDX_SA
        unique_ticker = dfm2["ticker"].unique().tolist()
        self.assertEqual(len(unique_ticker), 1)
        self.assertEqual(unique_ticker[0], "USD_INDX_SA")

    def test_make_graded(self):
        graded: pd.DataFrame = self.vins_m.make_graded(
            grading=[3, 2.1, 1], upgrades=[12, 24]
        )

        min_release_date = graded["release_date"].min().strftime("%Y-%m-%d")
        max_release_date = graded["release_date"].max().strftime("%Y-%m-%d")

        self.assertEqual(min_release_date, "2018-07-03")
        self.assertEqual(max_release_date, "2019-06-26")

        # check that grade2[cross_section] is a single value - USD
        unique_cross_section = graded["cross_section"].unique().tolist()
        self.assertEqual(len(unique_cross_section), 1)
        self.assertEqual(unique_cross_section[0], "USD")

        unique_category_code = graded["category_code"].unique().tolist()
        self.assertEqual(len(unique_category_code), 1)
        self.assertEqual(unique_category_code[0], "INDX")

        unique_adjustment = graded["adjustment"].unique().tolist()
        self.assertEqual(len(unique_adjustment), 1)
        self.assertEqual(unique_adjustment[0], "SA")

        unique_transformation = graded["transformation"].unique().tolist()
        self.assertEqual(len(unique_transformation), 1)
        self.assertEqual(unique_transformation[0], None)

    def test_misc(self):
        # make grade1 with freq = W
        vins_m = VintageData(
            "USD_INDX_SA",
            cutoff="2019-06-30",
            release_lags=[3, 20, 25],
            number_firsts=12,
            shortest=12,
            sd_ar=5,
            trend_ar=20,
            seasonal=10,
            added_dates=6,
            freq="W",
        ).make_grade1()

        with self.assertRaises(ValueError):
            vins_m = VintageData(
                "USD_INDX_SA",
                cutoff="sdfm",
                release_lags=[3, 20, 25],
                number_firsts=12,
                shortest=12,
                sd_ar=5,
                trend_ar=20,
                seasonal=10,
                added_dates=6,
                freq="M",
            ).make_grade1()


if __name__ == "__main__":
    unittest.main()
