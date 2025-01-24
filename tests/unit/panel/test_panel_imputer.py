from typing import Dict, List
import unittest
import numpy as np
import pandas as pd
import pytest
import warnings

from macrosynergy.management.types.qdf.classes import QuantamentalDataFrame
from tests.simulate import make_qdf
from macrosynergy.panel.panel_imputer import (
    MeanPanelImputer,
    MedianPanelImputer,
    BasePanelImputer,
)


class TestAll(unittest.TestCase):
    def dataframe_generator(self, date="2002-01-01"):
        self.cids: List[str] = ["AUD", "CAD", "GBP", "NZD", "USD"]
        self.xcats: List[str] = ["XR", "CRY", "GROWTH", "INFL"]
        self.cids: List[str] = ["AUD", "CAD", "GBP", "NZD", "USD"]
        self.xcats: List[str] = ["XR", "CRY", "GROWTH", "INFL"]
        self.all_cids: List[str] = [
            "AUD",
            "BRL",
            "CAD",
            "EUR",
            "GBP",
            "NZD",
            "USD",
            "ZAR",
        ]

        df_cids = pd.DataFrame(
            index=self.cids, columns=["earliest", "latest", "mean_add", "sd_mult"]
        )

        df_cids.loc["AUD"] = ["2000-01-01", "2020-12-31", 0.1, 1]
        df_cids.loc["CAD"] = ["2001-01-01", "2020-11-30", 0, 1]
        df_cids.loc["GBP"] = ["2002-01-01", "2020-11-30", 0, 2]
        df_cids.loc["NZD"] = ["2002-01-01", "2020-09-30", -0.1, 2]
        df_cids.loc["USD"] = [date, "2020-10-30", 0.2, 2]

        df_xcats = pd.DataFrame(
            index=self.xcats,
            columns=[
                "earliest",
                "latest",
                "mean_add",
                "sd_mult",
                "ar_coef",
                "back_coef",
            ],
        )

        df_xcats.loc["XR"] = ["2010-01-01", "2020-12-31", 0.1, 1, 0, 0.3]
        df_xcats.loc["CRY"] = ["2011-01-01", "2020-12-31", 1, 2, 0.95, 1]
        df_xcats.loc["GROWTH"] = ["2011-01-01", "2020-10-30", 1, 2, 0.9, 1]
        df_xcats.loc["INFL"] = ["2011-01-01", "2020-10-30", 1, 2, 0.8, 0.5]

        dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)
        self.dfd: pd.DataFrame = dfd

        black = {
            "AUD": ["2021-01-01", "2022-12-31"],
            "GBP": ["2021-01-01", "2100-01-01"],
        }

        self.blacklist: Dict[str, List[str]] = black
        self.start: str = "2012-01-01"
        self.end: str = "2020-01-01"

    def test_panel_extension_arg_types(self):
        self.dataframe_generator()

        # Test if the BaseImputerPanel class raises a TypeError when the df argument is not
        # a pandas DataFrame
        with self.assertRaises(TypeError):
            BasePanelImputer(
                df=1,
                xcats=self.xcats,
                cids=self.all_cids,
                start=self.start,
                end=self.end,
            )

        # Test if the BaseImputerPanel class raises a ValueError when the df argument is not
        # a quantamental DataFrame
        with self.assertRaises(ValueError):
            BasePanelImputer(
                df=pd.DataFrame(columns=["A", "B", "C"]),
                xcats=self.xcats,
                cids=self.all_cids,
                start=self.start,
                end=self.end,
            )

        # Test if the BaseImputerPanel class raises a TypeError when the xcats argument is
        # not a list
        with self.assertRaises(TypeError):
            BasePanelImputer(
                df=self.dfd,
                xcats="XR",
                cids=self.all_cids,
                start=self.start,
                end=self.end,
            )

        # Test if the BaseImputerPanel class raises a TypeError when the cids argument is
        # not a list
        with self.assertRaises(TypeError):
            BasePanelImputer(
                df=self.dfd,
                xcats=self.xcats,
                cids="AUD",
                start=self.start,
                end=self.end,
            )

        # Test if the BaseImputerPanel class raises a TypeError when the start argument is
        # not a string
        with self.assertRaises(TypeError):
            BasePanelImputer(
                df=self.dfd,
                xcats=self.xcats,
                cids=self.all_cids,
                start=1,
                end=self.end,
            )

        # Test if the BaseImputerPanel class raises a ValueError when the start argument is not
        # a valid date of format YYYY-MM-DD
        with self.assertRaises(ValueError):
            BasePanelImputer(
                df=self.dfd,
                xcats=self.xcats,
                cids=self.all_cids,
                start="01-01-2000",
                end=self.end,
            )

        # Test if the BaseImputerPanel class raises a TypeError when the end argument is
        # not a string
        with self.assertRaises(TypeError):
            BasePanelImputer(
                df=self.dfd,
                xcats=self.xcats,
                cids=self.all_cids,
                start=self.start,
                end=1,
            )

        # Test if the BaseImputerPanel class raises a ValueError when the end argument is
        # not a valid date of format YYYY-MM-DD
        with self.assertRaises(ValueError):
            BasePanelImputer(
                df=self.dfd,
                xcats=self.xcats,
                cids=self.all_cids,
                start=self.start,
                end="01-01-2000",
            )

        # Test if the BaseImputerPanel class raises a TypeError when the min_cids argument
        # is not an integer
        with self.assertRaises(TypeError):
            BasePanelImputer(
                df=self.dfd,
                xcats=self.xcats,
                cids=self.all_cids,
                start=self.start,
                end=self.end,
                min_cids="1",
            )

        # Test if the BaseImputerPanel class raises a TypeError when the min_cids argument
        # is less than 0
        with self.assertRaises(TypeError):
            BasePanelImputer(
                df=self.dfd,
                xcats=self.xcats,
                cids=self.all_cids,
                start=self.start,
                end=self.end,
                min_cids=-1,
            )

        # Test if the BaseImputerPanel class raises a TypeError when the postfix argument
        # is not a string
        with self.assertRaises(TypeError):
            BasePanelImputer(
                df=self.dfd,
                xcats=self.xcats,
                cids=self.all_cids,
                start=self.start,
                end=self.end,
                postfix=1,
            )

        # Test if min_cids is greater than the number of unique cids in the df argument
        with self.assertRaises(ValueError):
            BasePanelImputer(
                df=self.dfd,
                xcats=self.xcats,
                cids=self.all_cids,
                start=self.start,
                end=self.end,
                min_cids=6,
            )

    def test_mean_imputed_values(self):
        self.dataframe_generator()

        cid_value_dict = {
            "AUD": 0.5,
            "CAD": -0.6,
            "GBP": 123.7,
            "NZD": 1230.8,
            "USD": -0.19,
        }

        expected_mean = np.mean(list(cid_value_dict.values()))

        for cid in self.cids:
            self.dfd.loc[
                (self.dfd["real_date"] == "2015-01-02")
                & (self.dfd["cid"] == cid)
                & (self.dfd["xcat"] == "XR"),
                "value",
            ] = cid_value_dict[cid]

        panel = MeanPanelImputer(
            df=self.dfd,
            xcats=self.xcats,
            cids=self.all_cids,
            start=self.start,
            end=self.end,
            min_cids=0,
            postfix="",
        )

        filled_df = panel.impute()

        imputed_value = filled_df[
            (filled_df["real_date"] == "2015-01-02")
            & (filled_df["xcat"] == "XR")
            & (filled_df["cid"] == "ZAR")
        ]["value"].values[0]

        self.assertTrue(np.isclose(imputed_value, expected_mean, rtol=1e-5))

    def test_median_imputed_values(self):
        self.dataframe_generator()

        cid_value_dict = {
            "AUD": 0.5,
            "CAD": -0.6,
            "GBP": 123.7,
            "NZD": 1230.8,
            "USD": -0.19,
        }

        expected_median = np.median(list(cid_value_dict.values()))

        for cid in self.cids:
            self.dfd.loc[
                (self.dfd["real_date"] == "2015-01-02")
                & (self.dfd["cid"] == cid)
                & (self.dfd["xcat"] == "XR"),
                "value",
            ] = cid_value_dict[cid]

        panel = MedianPanelImputer(
            df=self.dfd,
            xcats=self.xcats,
            cids=self.all_cids,
            start=self.start,
            end=self.end,
            min_cids=0,
            postfix="",
        )

        filled_df = panel.impute()

        imputed_value = filled_df[
            (filled_df["real_date"] == "2015-01-02")
            & (filled_df["xcat"] == "XR")
            & (filled_df["cid"] == "ZAR")
        ]["value"].values[0]

        self.assertTrue(np.isclose(imputed_value, expected_median, rtol=1e-5))

    def test_impute(self):
        self.dataframe_generator()

        original_filtered_df = self.dfd[
            (self.dfd["real_date"] >= self.start) & (self.dfd["real_date"] <= self.end)
        ].reset_index(drop=True)

        # Select 100 different random values between 0 and original_filtered_df.shape[0] - 1
        random_indices = np.random.choice(
            original_filtered_df.index, 100, replace=False
        )
        original_filtered_df = original_filtered_df.drop(
            original_filtered_df.index[random_indices]
        )

        panel = MeanPanelImputer(
            df=original_filtered_df,
            xcats=self.xcats,
            cids=self.all_cids,
            start=self.start,
            end=self.end,
            min_cids=0,
            postfix="",
        )

        filled_df = panel.impute()

        # Check that the impute method is a QuantamentalDataFrame
        self.assertIsInstance(filled_df, QuantamentalDataFrame)

        # Check that the filled_df does not edit any of the data that is already present
        # and is not nan
        merged_df = pd.merge(
            original_filtered_df,
            filled_df,
            how="inner",
            on=["cid", "xcat", "real_date"],
        )

        self.assertTrue(merged_df.shape[0] == original_filtered_df.shape[0])

        self.assertTrue(merged_df["value_x"].equals(merged_df["value_y"]))

        # Check that the impute contains no nans (since min_cids = 0)
        self.assertFalse(filled_df.isnull().values.any())

    def test_impute_on_cid_that_doesnt_exist(self):
        self.dataframe_generator()

        panel = MeanPanelImputer(
            df=self.dfd,
            xcats=self.xcats,
            cids=["BRL"],
            start=self.start,
            end=self.end,
            min_cids=0,
            postfix="",
        )

        filled_df = panel.impute()

        # Check that values are imputeted for cids that don't exist in the original df
        self.assertTrue("BRL" in filled_df["cid"].unique())

        # Check that for each business day from start to end, there is a value for each
        # xcat for the cid that doesn't exist in the original df
        for date in pd.date_range(start=self.start, end=self.end, freq="B"):
            for xcat in self.xcats:
                self.assertTrue(
                    filled_df[
                        (filled_df["cid"] == "BRL")
                        & (filled_df["xcat"] == xcat)
                        & (filled_df["real_date"] == date)
                    ].shape[0]
                    == 1
                )

    def test_impute_on_cid_with_gaps(self):
        self.dataframe_generator()

        self.dfd = self.dfd[
            (self.dfd["real_date"] >= self.start) & (self.dfd["real_date"] <= self.end)
        ].reset_index(drop=True)

        # Select 100 different random values between 0 and original_filtered_df.shape[0] - 1
        random_indices = np.random.choice(self.dfd.index, 100, replace=False)
        self.dfd = self.dfd.drop(self.dfd.index[random_indices])

        panel = MeanPanelImputer(
            df=self.dfd,
            xcats=self.xcats,
            cids=self.cids,
            start=self.start,
            end=self.end,
            min_cids=0,
            postfix="",
        )

        filled_df = panel.impute()

        for date in pd.date_range(start=self.start, end=self.end, freq="B"):
            for xcat in self.xcats:
                for cid in self.cids:
                    self.assertTrue(
                        filled_df[
                            (filled_df["cid"] == cid)
                            & (filled_df["xcat"] == xcat)
                            & (filled_df["real_date"] == date)
                        ].shape[0]
                        == 1
                    )

    def test_min_cids(self):
        self.dataframe_generator()

        # Remove values from the dataframe so that there are less than min_cids for a
        # specific date

        # Now only 4 cids have values for the date "2015-01-02" and xcat "XR" but min_cids
        # is set to 5
        self.dfd.loc[
            (self.dfd["real_date"] == "2015-01-02")
            & (self.dfd["cid"] == "USD")
            & (self.dfd["xcat"] == "XR"),
            "value",
        ] = np.nan

        panel = MeanPanelImputer(
            df=self.dfd,
            xcats=self.xcats,
            cids=self.all_cids,
            start=self.start,
            end=self.end,
            min_cids=5,
            postfix="",
        )

        filled_df = panel.impute()

        # Check that the filled_df contains no imputed values where there are less observations
        # than the min_cids
        self.assertTrue(
            filled_df[
                (filled_df["real_date"] == "2015-01-02") & (filled_df["xcat"] == "XR")
            ]["cid"].nunique()
            == 4
        )

    def test_no_imputed_values(self):
        self.dataframe_generator()

        self.dfd = self.dfd[self.dfd["cid"].isin(self.cids[:4])]

        # Add values outside of start and end for cid[4]
        new_row = pd.DataFrame(
            {
                "cid": ["USD"],
                "xcat": ["XR"],
                "real_date": ["2024-01-02"],
                "value": [0.5],
            }
        )
        self.dfd = pd.concat([self.dfd, new_row], ignore_index=True)

        # Check a warning is thrown to say that no imputation was performed

        panel = MeanPanelImputer(
            df=self.dfd,
            xcats=self.xcats,
            cids=self.all_cids,
            start=self.start,
            end=self.end,
            min_cids=5,
            postfix="",
        )

        with pytest.warns(
            UserWarning,
            match="No imputation was performed. Consider changing the impute_method or min_cids.",
        ):
            filled_df = panel.impute()

        trimmed_self_dfd = self.dfd[
            (self.dfd["real_date"] >= self.start) & (self.dfd["real_date"] <= self.end)
        ]
        # Check that the filled_df contains no imputed values
        self.assertTrue(filled_df.shape == trimmed_self_dfd.shape)

    def test_return_blacklist(self):
        self.dataframe_generator()

        panel = MeanPanelImputer(
            df=self.dfd,
            xcats=self.xcats,
            cids=self.cids,
            start="2010-01-01",
            end="2020-12-31",
            min_cids=1,
        )

        expected_XR_blacklist = {
            "CAD": (pd.Timestamp("2020-12-01"), pd.Timestamp("2020-12-31")),
            "GBP": (pd.Timestamp("2020-12-01"), pd.Timestamp("2020-12-31")),
            "NZD": (pd.Timestamp("2020-10-01"), pd.Timestamp("2020-12-31")),
            "USD": (pd.Timestamp("2020-11-02"), pd.Timestamp("2020-12-31")),
        }

        with pytest.warns(
            RuntimeWarning,
            match="No imputation was performed. The blacklist is empty.",
        ):
            _ = panel.return_blacklist("XR")

        _ = panel.impute()

        blacklist = panel.return_blacklist("XR")

        # Check that the blacklist is a dictionary

        self.assertTrue(isinstance(blacklist, dict))

        # Check that the blacklist contains the correct cids
        self.assertTrue(set(blacklist.keys()) == set(expected_XR_blacklist.keys()))
        # Check that the blacklist contains the correct dates
        self.assertTrue(
            all(
                [
                    blacklist[cid] == expected_XR_blacklist[cid]
                    for cid in expected_XR_blacklist.keys()
                ]
            )
        )

    def test_postfix(self):
        self.dataframe_generator()

        panel = MeanPanelImputer(
            df=self.dfd,
            xcats=self.xcats,
            cids=self.cids,
            start="2010-01-01",
            end="2020-12-31",
            min_cids=1,
            postfix="_new",
        )

        filled_df = panel.impute()

        postfix_xcats = [xcat + "_new" for xcat in self.xcats]

        self.assertTrue(set(filled_df["xcat"].unique()) == set(postfix_xcats))
