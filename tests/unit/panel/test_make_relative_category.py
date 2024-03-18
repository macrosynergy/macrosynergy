import unittest
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Set

from tests.simulate import make_qdf
from macrosynergy.panel.make_relative_category import (
    make_relative_category,
    _prepare_category_basket,
)
from macrosynergy.management.utils import reduce_df
from random import randint, choice
import warnings


class TestAll(unittest.TestCase):

    def setUp(self) -> None:
        self.cids: List[str] = ["AUD", "CAD", "GBP", "NZD"]
        self.xcats: List[str] = ["XR", "CRY", "GROWTH", "INFL"]

        df_cids: pd.DataFrame = pd.DataFrame(
            index=self.cids, columns=["earliest", "latest", "mean_add", "sd_mult"]
        )

        df_cids.loc["AUD"] = ["2000-01-01", "2020-12-31", 0.1, 1]
        df_cids.loc["CAD"] = ["2001-01-01", "2020-11-30", 0, 1]
        df_cids.loc["GBP"] = ["2002-01-01", "2020-11-30", 0, 2]
        df_cids.loc["NZD"] = ["2002-01-01", "2020-09-30", -0.1, 2]

        df_xcats: pd.DataFrame = pd.DataFrame(
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

        df_xcats.loc["XR"] = ["2000-01-01", "2020-12-31", 0.1, 1, 0, 0.3]
        df_xcats.loc["CRY"] = ["2000-01-01", "2020-12-31", 1, 2, 0.95, 1]
        df_xcats.loc["GROWTH"] = ["2001-01-01", "2020-10-30", 1, 2, 0.9, 1]
        df_xcats.loc["INFL"] = ["2001-01-01", "2020-10-30", 1, 2, 0.8, 0.5]

        self.dfd: pd.DataFrame = make_qdf(df_cids, df_xcats, back_ar=0.75)

        black = {
            "AUD": ["2000-01-01", "2003-12-31"],
            "GBP": ["2018-01-01", "2100-01-01"],
        }

        self.blacklist: Dict[str, List[str]] = black

    def tearDown(self) -> None:
        return super().tearDown()

    def test_relative_value_dimensionality(self):
        dfd: pd.DataFrame = self.dfd.copy()

        with self.assertRaises(ValueError):
            # Validate the assertion on the parameter "rel_meth".
            dfd_1: pd.DataFrame = make_relative_category(
                df=dfd,
                xcats=["GROWTH", "INFL"],
                cids=None,
                blacklist=None,
                rel_meth="subtraction",
                rel_xcats=None,
                postfix="RV",
            )

        with self.assertRaises(TypeError):
            # xcats and rel_xcats must be same length.
            dfd_2: pd.DataFrame = make_relative_category(
                df=dfd,
                xcats=("XR", "GROWTH"),
                cids=None,
                blacklist=None,
                basket=["AUD", "CAD", "GBP"],
                rel_meth="subtract",
                rel_xcats=["XRvB3", "GROWTHvB3", "INFLvB3"],
            )

        # Basket is not a subset of xcats.
        with self.assertRaises(ValueError):
            dfd_3: pd.DataFrame = make_relative_category(
                df=dfd,
                xcats=["XR", "GROWTH"],
                cids=["GBP"],
                blacklist=None,
                basket=["XR", "GROWTH", "INVALID"],
                rel_meth="divide",
                rel_xcats=["XRvB3", "GROWTHvB3"],
            )

    def test_prepare_category_basket(self):
        dfd: pd.DataFrame = self.dfd

        xcats: List[str] = ["XR", "GROWTH"]
        cids: List[str] = ["AUD", "NZD"]
        start: str = "2000-01-01"
        end: str = "2020-12-31"
        dfx: pd.DataFrame = reduce_df(
            df=dfd,
            xcats=xcats,
            cids=cids,
            start=start,
            end=end,
            blacklist=None,
            out_all=False,
        )

        # Set the basket to all available categories.
        basket: List[str] = xcats
        dfb: pd.DataFrame
        xcats_used: List[str]
        with warnings.catch_warnings(record=True) as w:
            dfb, xcats_used = _prepare_category_basket(
                df=dfx, cid="AUD", basket=basket, xcats_avl=xcats, complete_set=False
            )
        self.assertEqual(sorted(xcats_used), sorted(xcats))
        self.assertEqual(sorted(list(set(dfb["xcat"]))), sorted(xcats))

        with warnings.catch_warnings(record=True) as w:
            dfb, xcats_used = _prepare_category_basket(
                df=dfx,
                cid="AUD",
                basket=basket + ["MISSING"],
                xcats_avl=xcats,
                complete_set=True,
            )
        self.assertTrue(len(xcats_used) == 0)
        self.assertTrue(dfb.empty)

    def test_relative_value_logic(self):
        dfd: pd.DataFrame = self.dfd

        basket_xcat: List[str] = ["XR"]
        with warnings.catch_warnings(record=True) as w:
            dfd_2: pd.DataFrame = make_relative_category(
                df=dfd,
                xcats=self.xcats,
                cids=self.cids,
                blacklist=None,
                basket=basket_xcat,
                rel_meth="subtract",
                rel_xcats=None,
                postfix="RV",
            )

        basket_df: pd.DataFrame = dfd_2[dfd_2["xcat"] == basket_xcat[0]]
        values: np.ndarray = basket_df["value"].to_numpy()
        self.assertTrue(np.isclose(np.sum(values), 0.0, rtol=0.001))

        basket_xcat: List[str] = ["XR"]
        with warnings.catch_warnings(record=True) as w:
            dfd_2: pd.DataFrame = make_relative_category(
                df=dfd,
                xcats=self.xcats,
                cids=self.cids,
                blacklist=None,
                basket=basket_xcat,
                rel_meth="divide",
                rel_xcats=None,
                postfix="RV",
            )

        basket_df: pd.DataFrame = dfd_2[dfd_2["xcat"] == basket_xcat[0] + "RV"]
        values: np.ndarray = basket_df["value"].to_numpy().mean()
        np.testing.assert_allclose(values, 1.0, rtol=0.001)
