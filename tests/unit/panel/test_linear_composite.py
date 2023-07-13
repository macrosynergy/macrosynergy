import unittest
import numpy as np
import pandas as pd
from typing import List



from macrosynergy.panel.linear_composite import linear_composite
from macrosynergy.management.simulate_quantamental_data import make_test_df, make_qdf


class TestAll(unittest.TestCase):
    def dataframe_generator(self) -> pd.DataFrame:
        self.cids: List[str] = ["AUD", "CAD", "GBP"]
        self.xcats: List[str] = ["CRY", "XR", "INFL"]

        df_cids = pd.DataFrame(
            index=self.cids, columns=["earliest", "latest", "mean_add", "sd_mult"]
        )
        df_cids.loc["AUD", :] = ["2010-01-01", "2020-12-31", 0.5, 2]
        df_cids.loc["CAD", :] = ["2011-01-01", "2020-11-30", 0, 1]
        df_cids.loc["GBP", :] = ["2012-01-01", "2020-11-30", -0.2, 0.5]

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

        df_xcats.loc["CRY", :] = ["2011-01-01", "2020-10-30", 1, 2, 0.9, 0.5]
        df_xcats.loc["XR", :] = ["2010-01-01", "2020-12-31", 0, 1, 0, 0.3]
        df_xcats.loc["INFL", :] = ["2012-01-01", "2020-11-30", 0.5, 1, 0.5, 0.5]

        dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)
        self.dfd: pd.DataFrame = dfd

    def test_linear_composite_xcat_agg_mode(self):
        all_cids: List[str] = ["AUD", "CAD", "GBP"]
        all_xcats: List[str] = ["XR", "CRY", "INFL"]
        start: str = "2000-01-01"
        end: str = "2001-01-01"

        ## Test Case 1a - Testing non-normalized weights
        dfd = make_test_df(
            cids=all_cids, xcats=all_xcats, start_date=start, end_date=end
        )
        # set all values to 1
        dfd["value"] = 1

        weights: List[str] = [1, 2, 4]
        # Don't use 1,2,3 as 3*2 is 6. the only way to get 7 is to use 1,2,4

        rdf: pd.DataFrame = linear_composite(
            df=dfd,
            xcats=all_xcats,
            weights=weights,
            normalize_weights=False,
        )
        # all return values should be 7
        assert sum(weights) == 7
        self.assertTrue(np.all(rdf["value"].values == sum(weights)))

        ## Test Case 1b - Testing default normalization behavior
        rdf: pd.DataFrame = linear_composite(
            df=dfd,
            xcats=all_xcats,
            weights=weights,
        )
        # all return values should be 1
        self.assertTrue(np.all(rdf["value"].values == 1))

        # Test Case 1c - Testing default equal weights
        dfd["value"] = 2
        rdf: pd.DataFrame = linear_composite(
            df=dfd,
            xcats=all_xcats,
            new_xcat="TEST-1C",
        )
        # all return values should be 2
        self.assertTrue(np.all(rdf["value"].values == 2))

        # no cids, xcats, or dates should be missing
        self.assertTrue(set(rdf["cid"].values) == set(all_cids))
        self.assertTrue(rdf["xcat"].unique().tolist() == ["TEST-1C"])
        self.assertTrue(set(rdf["real_date"].values) == set(dfd["real_date"].values))

        ## Test Case 2a
        _end: str = "2000-02-01"
        dfd = make_test_df(
            cids=all_cids, xcats=all_xcats, start_date=start, end_date=_end
        )
        dfd["value"] = 1

        # make 2000-01-17 for all CAD and AUD values a nan
        dfd.loc[
            (dfd["real_date"] == "2000-01-17")
            & ((dfd["cid"] == "CAD") | (dfd["cid"] == "AUD")),
            "value",
        ] = np.nan

        _cids: List[str] = ["AUD", "CAD"]
        rdf: pd.DataFrame = linear_composite(
            df=dfd,
            cids=_cids,
            xcats=all_xcats,
        )
        # 2000-01-17 should be nan
        self.assertTrue(rdf[rdf["real_date"] == "2000-01-17"]["value"].isna().all())



if __name__ == "__main__":
    unittest.main()
