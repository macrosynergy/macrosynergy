import unittest
import numpy as np
import pandas as pd
from typing import *

from macrosynergy.panel.linear_composite import *
from macrosynergy.management.shape_dfs import reduce_df


class TestAll(unittest.TestCase):
    def generate_test_dfs(
        self, cids: List[str], xcats: List[str], values: List[int] = None,
        missing_indices: List[int] = None
    ) -> pd.DataFrame:
        dates = pd.date_range("2000-01-01", "2000-01-03")
        total_entries = len(cids) * len(xcats) * len(dates)
        randomints = list(np.arange(total_entries) - total_entries // 2)
        lx = [
            [cid, xcat, date, randomints.pop()]
            for cid in cids
            for xcat in xcats
            for date in dates
        ]
        dfst = pd.DataFrame(lx, columns=["cid", "xcat", "real_date", "value"])

        dfst.loc[missing_indices, "value"] = np.NaN

        return dfst
    
    def test_linear_composite_noNans(self):
        cids = ["AUD", "CAD", "GBP"]
        xcats = ["XR", "CRY", "INFL"]
        dfta = self.generate_test_dfs(cids, xcats)

        weights = [1, 2, 3]
        signs = [1, -1, 1]  

        

    def test_linear_composite_A(self):
        cids = ["AUD", "CAD", "GBP"]
        xcats = ["XR", "CRY", "INFL"]
        missing_idx = [9, 18, 19, 20, 23, 25, 26]
        dfta = self.generate_test_dfs(cids, xcats, missing_indices=missing_idx)

        weights = [1, 2, 3]
        signs = [-1, 1, 1]


        outdf = linear_composite(
            df=dfta,
            xcats=xcats,
            cids=cids,
            weights=weights,
            signs=signs,
            complete_xcats=False,
        )

        excpected_vals_cFalse = [
            4.666666666666667,
            4.000000000000002,
            3.3333333333333344,
            -0.8000000000000002,
            -2.0000000000000004,
            -2.6666666666666674,
            -9.8,
            -9.0,
            np.NaN,
        ]
        
        # test case for complete_xcats=False
        self.assertTrue(
            np.allclose(
                outdf["value"].values, excpected_vals_cFalse, rtol=1e-5, equal_nan=True
            )
        )
        
        self.assertTrue(np.allclose(outdf["value"].values, excpected_vals_cFalse, rtol=1e-5, equal_nan=False))

        excpected_vals_cTrue = [
            4.666666666666668,
            4.000000000000002,
            3.3333333333333344,
            np.NaN,
            -2.0000000000000004,
            -2.6666666666666674,
            np.NaN,
            np.NaN,
            np.NaN,
        ]
               
        outdf = linear_composite(
            df=dfta,
            xcats=xcats,
            cids=cids,
            weights=weights,
            signs=signs,
            complete_xcats=True,
        )
        
        # test case for complete_xcats=False
        self.assertTrue(
            np.allclose(
                outdf["value"].values, excpected_vals_cTrue, rtol=1e-5, equal_nan=True
            )
        )

    def test_linear_composite_B(self):
        cids = ["AUD", "CAD", "GBP", "USD"]
        xcats = ["XR", "CRY", "INFL", "RIR"]
        dfst = self.generate_test_dfs(cids, xcats)

        weights = [0, 1, 5, 10]
        signs = [-1, 1, 1, -1]
        
        outdf = linear_composite(
            df=dfst, xcats=xcats, cids=cids, weights=weights, signs=signs,
            complete_xcats=False,
        )


if __name__ == "__main__":
    unittest.main()
