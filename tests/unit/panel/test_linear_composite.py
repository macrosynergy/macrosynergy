import unittest
import numpy as np
import pandas as pd
from typing import *

from macrosynergy.panel.linear_composite import *


class TestAll(unittest.TestCase):
    def generate_test_dfs(
        self, cids: List[str], xcats: List[str], values: List[int] = None,
        dates: pd.DatetimeIndex = None, missing_indices: List[int] = None, 
    ) -> pd.DataFrame:
        if values is None:
            total_entries = len(cids) * len(xcats) * len(dates)
            values = list(np.arange(total_entries) - total_entries // 2)
        elif isinstance(values, (int, float)):
            values = [values for _ in range(len(cids) * len(xcats) * len(dates))]
        lx = [
            [cid, xcat, date, values.pop()]
            for cid in cids
            for xcat in xcats
            for date in dates
        ]
        dfst = pd.DataFrame(lx, columns=["cid", "xcat", "real_date", "value"])
        if missing_indices is not None:
            dfst.loc[missing_indices, "value"] = np.NaN

        return dfst
    
    def test_linear_composite_noNans(self):
        cids = ["AUD", "CAD", "GBP"]
        xcats = ["XR", "CRY", "INFL"]
        dates = pd.date_range("2023-01-01", "2023-01-30")
        
        df_test = self.generate_test_dfs(cids, xcats, values=1, dates=dates)
        df_test.loc[df_test.index % 2 == 0, "value"] = 0

        weights = [1, 2, 1]
        signs = [-1, 1, 1] 

        outdf = linear_composite(
            df=df_test, xcats=xcats, cids=cids, weights=weights, signs=signs,
            complete_xcats=False, new_xcat="testCase"
        )

        self.assertTrue(isinstance(outdf, pd.DataFrame))
        # since this is artificial data, all dates have values (no weekends, no holidays)
        self.assertEqual(len(outdf), len(cids) * len(dates)) 

        # since all xcats are complete, the output should be the same for complete_xcats=True and =False
        self.assertTrue(outdf.equals(
            linear_composite(
                df=df_test, xcats=xcats, cids=cids, weights=weights, signs=signs,
                complete_xcats=True, new_xcat="testCase"
            )))


        # check if df[xcats].unq is just 'testCase'
        self.assertEqual(len(outdf["xcat"].unique()), 1)
        self.assertTrue("testCase" in outdf["xcat"].unique())
        

        
        # in this test case, the output is [0, 0.5, 0, 0.5 ...](len(dates) * len(cids) * len(xcats))
        # these test values clearly show the correct behaviour for the weights and the signs functionalities
        vals = np.array(outdf['value'])
        self.assertTrue(np.all((vals * 2).astype(bool)[1::2]))
        self.assertFalse(np.any((vals * 2).astype(bool)[::2]))


    def test_linear_composite_with_nans(self):
        cids = ["AUD", "CAD", "GBP"]
        xcats = ["XR", "CRY", "INFL", "RIR"]
        dates = pd.date_range("2023-01-01", "2023-01-30")
        
        df_test = self.generate_test_dfs(cids, xcats, values=1, dates=dates)

        nans_offset = 15
        df_test.loc[df_test.index % nans_offset == 0, "value"] = np.NaN

        weights = [1, 1, 5, 1]
        signs = [-1, 1, 1, -1]

        outdf = linear_composite(
            df=df_test, xcats=xcats, cids=cids, weights=weights, signs=signs,
            complete_xcats=False, new_xcat="testCase"
        )
        
        # output = [nan, 0.5 ... (x 14)... nan, 0.5 ... ] (len(dates) * len(cids))
        # IMPORTANT NOTE:
        # in this case, it "just so happens" that the nans are in the same place for all cids
        # a new test case would require a different set of rules
        self.assertTrue(np.all(np.isnan(np.array(outdf['value'])[::nans_offset])))
        # sum of the non nans should be 0.5 * len(cids) * len(dates) // nans_offset
        self.assertEqual(
            np.sum(np.array(outdf['value'])[1::nans_offset]), 
            0.5 * len(cids) * len(dates) // nans_offset
        )
        
        # now test the same with complete_xcats=True
        outdf = linear_composite(
            df=df_test, xcats=xcats, cids=cids, weights=weights, signs=signs,
            complete_xcats=True, new_xcat="testCase"
        )
        
        # check if the nans are in some particular places
        # again, this is a very specific test case
        self.assertTrue(np.all(np.isnan(np.array(outdf['value'])[::nans_offset])))
        self.assertTrue(np.all(outdf.loc[np.isnan(outdf['value']), :].index.values \
                               == np.arange(6) * nans_offset))
        
        return True        


    def test_linear_composite_hc(self):
        # hard coded test
        cids = ["AUD", "CAD", "GBP"]
        xcats = ["XR", "CRY", "INFL"]
        missing_idx = [9, 18, 19, 20, 23, 25, 26]
        dates = pd.date_range("2000-01-01", "2000-01-03")
        df_test = self.generate_test_dfs(cids, xcats, dates=dates, missing_indices=missing_idx)

        weights = [1, 2, 3]
        signs = [-1, 1, 1]


        # test case for complete_xcats=False → excpected_vals_cFalse
        outdf = linear_composite(
            df=df_test,
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
        
        self.assertTrue(
            np.allclose(
                outdf["value"].values, excpected_vals_cFalse, rtol=1e-5, equal_nan=True
            )
        )

        # test case for complete_xcats=False → excpected_vals_cTrue
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
            df=df_test,
            xcats=xcats,
            cids=cids,
            weights=weights,
            signs=signs,
            complete_xcats=True,
        )
        
        # test case for complete_xcats=True → excpected_vals_cTrue
        self.assertTrue(
            np.allclose(
                outdf["value"].values, excpected_vals_cTrue, rtol=1e-5, equal_nan=True
            )
        )



if __name__ == "__main__":
    unittest.main()
