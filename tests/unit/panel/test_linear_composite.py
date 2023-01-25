import unittest
import numpy as np
import pandas as pd
from typing import *

from macrosynergy.panel.linear_composite import *

def rle(arr):
    # take array and return as run length encoded array as [[v1, c1], [v2, c2], ...]
    oarr = []
    for k in arr:
        if (oarr== k) and (oarr[-1][0]== k) :
            oarr[-1][1] += 1
        else:
            oarr.append([k, 1])
    return oarr

def un_rle(arr):
    # take run length encoded array as [[v1, c1], [v2, c2], ...] and return as array
    oarr = []
    for k in arr:
        oarr += [k[0]] * k[1]
    return oarr


class TestAll(unittest.TestCase):
    def generate_test_dfs(
        self,
        cids: List[str],
        xcats: List[str],
        values: List[int] = None,
        dates: pd.DatetimeIndex = None,
        missing_indices: List[int] = None,
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
            df=df_test,
            xcats=xcats,
            cids=cids,
            weights=weights,
            signs=signs,
            complete_xcats=False,
            new_xcat="testCase",
        )

        self.assertTrue(isinstance(outdf, pd.DataFrame))
        # since this is artificial data, all dates have values (no weekends, no holidays)
        self.assertEqual(len(outdf), len(cids) * len(dates))

        # since all xcats are complete, the output should be the same for complete_xcats=True and =False
        self.assertTrue(
            outdf.equals(
                linear_composite(
                    df=df_test,
                    xcats=xcats,
                    cids=cids,
                    weights=weights,
                    signs=signs,
                    complete_xcats=True,
                    new_xcat="testCase",
                )
            )
        )

        # check if df[xcats].unq is just 'testCase'
        self.assertEqual(len(outdf["xcat"].unique()), 1)
        self.assertTrue("testCase" in outdf["xcat"].unique())

        # in this test case, the output is [0, 0.5, 0, 0.5 ...](len(dates) * len(cids) * len(xcats))
        # these test values clearly show the correct behaviour for the weights and the signs functionalities
        vals = np.array(outdf["value"])
        self.assertTrue(np.all((vals * 2).astype(bool)[1::2]))
        self.assertFalse(np.any((vals * 2).astype(bool)[::2]))

    def test_linear_composite_with_nans(self):
        def fib(n, ns={0: 0, 1: 1}):
            if n not in ns:
                ns[n] = fib(n - 1) + fib(n - 2)
            return ns[n]

        cids = ["AUD", "CAD"]
        xcats = ["XR", "CRY", "INFL", "RIR", "RER"]
        dates = pd.date_range("2023-01-01", "2023-01-30")

        df_test = self.generate_test_dfs(cids, xcats, values=2, dates=dates)

        ctr, missing_idx = 2, []
        while fib(ctr) < len(df_test):
            missing_idx.append(fib(ctr))
            ctr = ctr + 1

        df_test.loc[missing_idx, "value"] = np.NaN

        weights = [1, 1, 5, 1, 1]
        signs = [-1, 1, 1, -1, 1]

        outdf = linear_composite(
            df=df_test,
            xcats=xcats,
            cids=cids,
            weights=weights,
            signs=signs,
            complete_xcats=False,
            new_xcat="testCase",
        )

        expected_results = np.array(
        un_rle(
            [[10/9, 1], 
            [3/2, 3],
            [1.0, 1],
            [3/2, 1],
            [10/9, 2], 
            [3/2, 1],
            [10/9, 4], 
            [3/2, 1],
            [10/9, 7], 
            [3/2, 1],
            [10/9, 2], 
            [1.0, 2],
            [10/9, 3], 
            [0.0, 1],
            [10/9, 23],
            [0.0, 1],
            [10/9, 6]]
        ))
        self.assertTrue(np.allclose(outdf["value"], expected_results))           
        

    def test_linear_composite_hc(self):
        # hard coded test
        cids = ["AUD", "CAD", "GBP"]
        xcats = ["XR", "CRY", "INFL"]
        missing_idx = [9, 18, 19, 20, 23, 25, 26]
        dates = pd.date_range("2000-01-01", "2000-01-03")
        df_test = self.generate_test_dfs(
            cids, xcats, dates=dates, missing_indices=missing_idx
        )

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
