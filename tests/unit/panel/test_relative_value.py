import unittest
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Set

from tests.simulate import make_qdf
from macrosynergy.panel.make_relative_value import make_relative_value, _prepare_basket
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
            dfd_1: pd.DataFrame = make_relative_value(
                df=dfd,
                xcats=["GROWTH", "INFL"],
                cids=None,
                blacklist=None,
                rel_meth="subtraction",
                rel_xcats=None,
                postfix="RV",
            )

        with self.assertRaises(TypeError):
            # Validate the assertion on "xcats" parameter.
            dfd_2: pd.DataFrame = make_relative_value(
                df=dfd,
                xcats=("XR", "GROWTH"),
                cids=None,
                blacklist=None,
                basket=["AUD", "CAD", "GBP"],
                rel_meth="subtract",
                rel_xcats=["XRvB3", "GROWTHvB3", "INFLvB3"],
            )

        with self.assertRaises(ValueError):
            # Validate the clause constructed around the basket parameter. The
            # cross-sections included in the "basket" must either be complete, inclusive
            # of all defined cross-sections, or a valid subset dependent on the cross-
            # sections passed into the function.
            dfd_3: pd.DataFrame = make_relative_value(
                df=dfd,
                xcats=["XR", "GROWTH"],
                cids=["GBP"],
                blacklist=None,
                basket=["AUD", "CAD", "INVALID"],
                rel_meth="divide",
                rel_xcats=["XRvB3", "GROWTHvB3"],
            )

        # The first aspect of the code to validate is if the DataFrame is reduced to a
        # single cross-section, and the basket is naturally limited to a single cross
        # section as well, the notion of computing the relative value is not appropriate.
        # Therefore, the code should throw a runtime error given it is being used
        # incorrectly.
        with self.assertRaises(RuntimeError) as context:
            make_relative_value(
                df=dfd,
                xcats=["GROWTH", "INFL"],
                cids=["AUD"],
                blacklist=None,
                basket=["AUD"],
                rel_meth="subtract",
                rel_xcats=None,
                postfix="RV",
            )
            run_error: str = (
                "Computing the relative value on a single cross-section using a "
                "basket consisting exclusively of the aforementioned "
                "cross-section is an incorrect usage of the function."
            )

            self.assertTrue(run_error in context.exception)

        # First part of the logic to validate is the stacking mechanism, and subsequent
        # dimensions of the returned DataFrame. Once the reduction is accounted for, the
        # dimensions should reflect the returned input.
        xcats: List[str] = self.xcats[:-2]
        cids: List[str] = self.cids
        start: str = "2001-01-01"
        end: str = "2020-11-30"
        dfx: pd.DataFrame = reduce_df(
            df=self.dfd,
            xcats=xcats,
            cids=cids,
            start=start,
            end=end,
            blacklist=None,
            out_all=False,
        )
        # To confirm the above statement, the parameter "basket" must be equated to None
        # to prevent any further reduction.
        # Further, for the dimensions of the input DataFrame to match the output
        # DataFrame, each date the DataFrame is defined over must have greater than one
        # cross-section available for each index (date) otherwise rows with only a single
        # realised value will be removed.
        # Therefore, to achieve this, each date having at least two realised values,
        # set both the start date & end date parameters to the second earliest and
        # latest date respectively (of the defined cross-sections' realised series). Both
        # the categories are defined over the same time-period, so the cross-sections
        # will delimit the dimensions.
        with warnings.catch_warnings(record=True) as w:
            dfd_rl: pd.DataFrame = make_relative_value(
                df=self.dfd,
                xcats=xcats,
                cids=cids,
                start=start,
                end=end,
                blacklist=None,
                basket=None,
                rel_meth="subtract",
                rel_xcats=None,
                postfix="RV",
            )
            self.assertEqual(dfx.shape, dfd_rl.shape)

        # Test the proposal that any dates with only a single realised value will be
        # truncated from the DataFrame given understanding the relative value of a single
        # realised return is meaningless.
        # The difference between the dimensions of the input DataFrame and the returned
        # DataFrame should correspond to the number of indices with only a single value.

        xcats: List[str] = self.xcats[0]
        cids: List[str] = self.cids
        # Ensures for a period of time only a single cross-section is defined.
        start: str = "2000-01-01"
        end: str = "2020-09-30"
        dfx: pd.DataFrame = reduce_df(
            df=self.dfd,
            xcats=[xcats],
            cids=cids,
            start=start,
            end=end,
            blacklist=None,
            out_all=False,
        )
        input_rows: int = dfx.shape[0]
        dfw: pd.DataFrame = dfx.pivot(index="real_date", columns="cid", values="value")

        data: np.ndarray = dfw.to_numpy()
        data = data.astype(dtype=np.float64)
        active_cross: np.ndarray = np.sum(~np.isnan(data), axis=1)

        single_value: np.ndarray = np.where(active_cross == 1)[0]
        no_single_values: int = single_value.size

        # Apply the function to understand if the logic above holds.
        with warnings.catch_warnings(record=True) as w:
            dfd_rl: pd.DataFrame = make_relative_value(
                df=self.dfd,
                xcats=xcats,
                cids=cids,
                start=start,
                end=end,
                blacklist=None,
                basket=None,
                rel_meth="divide",
                rel_xcats=None,
                postfix="RV",
            )
            output_rows: int = dfd_rl.shape[0]

            self.assertTrue(output_rows == (input_rows - no_single_values))

        # Test "complete_cross" parameter.

        # Construct a DataFrame containing two categories but one of the categories is
        # defined over fewer cross-sections. To be precise, the cross-sections present
        # for the aforementioned category will be a subset of the cross-sections
        # available for the secondary category. Further, the basket will be set to the
        # union of cross-sections.
        dfd: pd.DataFrame = self.dfd
        xcats = ["XR", "CRY"]
        cids: List[str] = ["AUD", "CAD", "GBP", "NZD"]
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

        # On the reduced DataFrame, remove a single cross-section from one of the
        # categories.
        filt1: pd.Series = ~((dfx["cid"] == "AUD") & (dfx["xcat"] == "XR"))
        dfdx: pd.DataFrame = dfx[filt1]

        # Pass in the filtered DataFrame, and test whether the correct print statement
        # appears in the console.
        with warnings.catch_warnings(record=True) as w:
            dfd_rl: pd.DataFrame = make_relative_value(
                df=dfdx,
                xcats=xcats,
                cids=cids,
                start=start,
                end=end,
                blacklist=None,
                basket=None,
                complete_cross=False,
                rel_meth="subtract",
                rel_xcats=None,
                postfix="RV",
            )
            wlist = [w for w in w if issubclass(w.category, UserWarning)]
            # Assert a UserWarning is raised.
            self.assertTrue(len(wlist) == 1)
            self.assertTrue(issubclass(wlist[-1].category, UserWarning))
            warning_message: str = str(wlist[-1].message)

            printed_cids: str = set(eval(warning_message[-23:-1]))
            test: str = set(["CAD", "GBP", "NZD"])
            self.assertEqual(printed_cids, test)

        # If the "complete_cross" parameter is set to True, the corresponding category
        # defined over an incomplete set of cross-sections, relative to the basket, will
        # be removed from the output DataFrame.
        rel_xcats: List[str] = ["XR_RelValue", "CRY_RelVal"]
        with warnings.catch_warnings(record=True) as w:
            dfd_rl: pd.DataFrame = make_relative_value(
                df=dfdx,
                xcats=xcats,
                cids=cids,
                start=start,
                end=end,
                blacklist=None,
                basket=self.cids,
                complete_cross=True,
                rel_meth="subtract",
                rel_xcats=rel_xcats,
                postfix=None,
            )

        # Assert the DataFrame only contains a single category: the category with a
        # complete set of cross-sections relative to the basket ("CRY_RelVal").
        rlt_value_xcats: Set = set(dfd_rl["xcat"])
        no_rlt_value_xcats: int = len(rlt_value_xcats)

        self.assertTrue(no_rlt_value_xcats == 1)

        rl_xcat: str = next(iter(rlt_value_xcats))
        self.assertTrue(rl_xcat == "CRY_RelVal")

        # try with xcats=xcats, it should NOT raise a warning
        sel_cids: List[str] = ["CAD", "GBP", "NZD"]
        with warnings.catch_warnings(record=True) as w:
            dfd_rl: pd.DataFrame = make_relative_value(
                df=dfdx,
                xcats=xcats,
                cids=sel_cids,
                start=start,
                end=end,
            )
            wlist = [w for w in w if issubclass(w.category, UserWarning)]
            # Assert a UserWarning is not raised.
            self.assertTrue(len(wlist) == 0)

        with self.assertRaises(ValueError):
            dfd_1 = make_relative_value(
                df=dfdx,
                xcats=xcats,
                cids=cids,
                start=start,
                end=end,
                blacklist=None,
                basket=self.cids,
                rel_meth="subtract",
                rel_xcats=rel_xcats,
                rel_reference="INVALID",
                postfix=None,
            )

    def test_prepare_basket(self):
        # Explicitly test _prepare_basket() method.
        dfd: pd.DataFrame = self.dfd

        # Set the cids parameter to a reduced subset (a particuliar category is missing
        # requested cross-sections).
        cids: List[str] = ["AUD", "NZD"]
        start: str = "2000-01-01"
        end: str = "2020-12-31"
        dfx: pd.DataFrame = reduce_df(
            df=dfd,
            xcats=["XR"],
            cids=cids,
            start=start,
            end=end,
            blacklist=None,
            out_all=False,
        )

        # Set the basket to all available cross-sections. Larger request.
        basket: List[str] = self.cids
        dfb: pd.DataFrame
        cids_used: List[str]
        with warnings.catch_warnings(record=True) as w:
            dfb, cids_used = _prepare_basket(
                df=dfx, xcat="XR", basket=basket, cids_avl=cids, complete_cross=False
            )
        self.assertTrue(sorted(cids_used) == cids)
        self.assertTrue(sorted(list(set(dfb["cid"]))) == cids)

        # If complete_cross parameter is set to True and the respective category is not
        # defined over all cross-sections defined in the basket, the function should
        # return an empty list and an empty DataFrame.
        with warnings.catch_warnings(record=True) as w:
            dfb, cids_used = _prepare_basket(
                df=dfx, xcat="XR", basket=basket, cids_avl=cids, complete_cross=True
            )
        self.assertTrue(len(cids_used) == 0)
        self.assertTrue(dfb.empty)

    def test_relative_value_logic(self):
        dfd: pd.DataFrame = self.dfd

        # Aim to test the application of the actual relative_value method: subtract or
        # divide.
        # If the basket contains a single cross-section, the relative value benchmark is
        # simply the realised return of the respective cross-section. Therefore, the
        # cross-section chosen will consequently have a zero value for each output if the
        # logic is correct.
        basket_cid: List[str] = ["AUD"]
        with warnings.catch_warnings(record=True) as w:
            dfd_2: pd.DataFrame = make_relative_value(
                df=dfd,
                xcats=["INFL"],
                cids=self.cids,
                blacklist=None,
                basket=basket_cid,
                rel_meth="subtract",
                rel_xcats=None,
                postfix="RV",
            )

        basket_df: pd.DataFrame = dfd_2[dfd_2["cid"] == basket_cid[0]]
        values: np.ndarray = basket_df["value"].to_numpy()
        self.assertTrue(np.isclose(np.sum(values), 0.0, rtol=0.001))

        # Test the logic of the function if there are multiple cross-sections defined in
        # basket. First, test the relative value using subtraction and secondly test
        # relative value using division.

        # Incorporate three cross-sections for the basket.
        basket_cid: List[str] = ["AUD", "CAD", "GBP"]
        xcats: str = choice(self.xcats)
        start: str = "2001-01-01"
        end: str = "2020-10-30"
        dfx: pd.DataFrame = reduce_df(
            df=dfd,
            xcats=[xcats],
            cids=self.cids,
            start=start,
            end=end,
            blacklist=None,
            out_all=False,
        )

        dfd_3: pd.DataFrame = make_relative_value(
            df=dfx,
            xcats=xcats,
            cids=self.cids,
            blacklist=self.blacklist,
            basket=basket_cid,
            rel_meth="subtract",
            rel_xcats=None,
            postfix="RV",
        )
        # Isolate an arbitrarily chosen date and test the logic
        dfw: pd.DataFrame = dfx.pivot(index="real_date", columns="cid", values="value")
        index: pd.Index = dfw.index
        no_rows: int = index.size
        range_: Tuple[int, int] = (int(no_rows * 0.25), int(no_rows * 0.75))

        random_index: int = randint(*range_)
        random_index: int = 1567
        date: str = index[random_index]
        assert date in list(index)

        random_row: pd.Series = dfw.iloc[random_index, :]
        random_row_dict: Dict[str, float] = random_row.to_dict()
        values: List[float] = [v for k, v in random_row_dict.items() if k in basket_cid]
        manual_mean: float = sum(values) / len(values)

        computed_values: np.ndarray = (random_row - manual_mean).to_numpy()

        dfd_3_pivot: pd.DataFrame = dfd_3.pivot(
            index="real_date", columns="cid", values="value"
        )
        output_index: pd.Index = dfd_3_pivot.index
        index_val: np.ndarray = np.where(output_index == date)[0]

        function_output: np.ndarray = (dfd_3_pivot.iloc[index_val, :]).to_numpy()

        function_output: np.ndarray = function_output[0]
        self.assertTrue(np.allclose(computed_values, function_output))

        # Test the division.
        # Computing make_relative_value() on a single category that has been chosen
        # randomly.
        dfd_4: pd.DataFrame = make_relative_value(
            df=dfx,
            xcats=xcats,
            cids=self.cids,
            blacklist=self.blacklist,
            basket=basket_cid,
            rel_meth="divide",
            rel_xcats=None,
            postfix="RV",
        )

        # Divide each cross-section's realised return by the mean of the basket.
        computed_values: np.ndarray = (random_row / manual_mean).to_numpy()

        dfd_4_pivot: pd.DataFrame = dfd_4.pivot(
            index="real_date", columns="cid", values="value"
        )
        output_index: pd.Series = dfd_4_pivot.index
        index_val: np.ndarray = np.where(output_index == date)[0]

        function_output: np.ndarray = (dfd_4_pivot.iloc[index_val, :]).to_numpy()

        self.assertTrue(np.allclose(computed_values, function_output[0]))

        # Running where cids and basket are disjoint sets. This running without error is
        # a test in itself.
        dfd_5: pd.DataFrame = make_relative_value(
            df=self.dfd,
            xcats=self.xcats[0],
            cids=["AUD", "CAD"],
            blacklist=self.blacklist,
            basket=["GBP"],
            rel_meth="divide",
            rel_xcats=None,
            postfix="RV",
        )

        # Check that the correct tickers are in the output.
        out_basket_tickers: Set = set(dfd_5["cid"] + dfd_5["xcat"])
        expct_basket_tickers: Set = set(
            [f"{cd}{self.xcats[0]}RV" for cd in ["AUD", "CAD", "GBP"]]
        )
        self.assertEqual(out_basket_tickers, expct_basket_tickers)

if __name__ == "__main__":
    unittest.main()
