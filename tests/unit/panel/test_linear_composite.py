import unittest
import numpy as np
import pandas as pd
from typing import List, Tuple, Union, Dict, Any

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

    def test_linear_composite_args(self):
        """
        This section is meant to test the arguments of the linear_composite function.
        It simply runs the function with a set of arguments and checks for errors.
        No return values are checked.
        """

        self.dataframe_generator()
        rdf: pd.DataFrame = linear_composite(
            df=self.dfd,
            cids=self.cids[0],
            xcats=None,
        )
        # any return data implies success
        self.assertTrue(not rdf.empty)

        # base case
        base_args: Dict[str, Any] = {
            "df": self.dfd,
            "xcats": self.xcats,
        }
        # type error cases
        type_error_cases: List[Dict[str, Any]] = [
            {"df": 1},
            {"signs": 1},
            {"xcats": 1},
            {"cids": 1},
            {"weights": pd.DataFrame()},
            {"weights": [1, 2, "bar"]},
        ]
        for case in type_error_cases:
            argsx: Dict[str, Any] = base_args.copy()
            argsx.update(case)
            with self.assertRaises(TypeError):
                rdf: pd.DataFrame = linear_composite(**argsx)

        # value error cases
        value_error_cases: List[Dict[str, Any]] = [
            {"df": pd.DataFrame()},
            {"df": self.dfd.assign(value=np.NaN)},
            {"start": 1},
            {"end": 1},
            {"xcats": ["foo"]},
            {"cids": ["bar"]},
            {"xcats": self.xcats[0], "weights": "foo"},
            {"weights": [1] * 100},
            {"signs": [1] * 100},
            {"weights": [1] * (len(self.xcats) - 1) + [0]},
            {"signs": [1] * (len(self.xcats) - 1) + [0]},
        ]
        for case in value_error_cases:
            argsx: Dict[str, Any] = base_args.copy()
            argsx.update(case)
            with self.assertRaises(ValueError):
                rdf: pd.DataFrame = linear_composite(**argsx)

        # check that passings signs as random values works
        alt_signs: List[float] = [
            (rnd - 0.5) * 2 for rnd in np.random.random(len(self.xcats))
        ]

        with self.assertWarns(UserWarning):
            rdf: pd.DataFrame = linear_composite(
                df=self.dfd,
                xcats=self.xcats,
                signs=alt_signs,
            )

    def test_linear_composite_xcat_agg_mode(self):
        self.dataframe_generator()

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

        ## Test Case 2a & b - Testing nan logic
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

        for iter_bool in [True, False]:
            rdf: pd.DataFrame = linear_composite(
                df=dfd,
                cids=_cids,
                xcats=all_xcats,
                complete_xcats=iter_bool,
            )
            # 2000-01-17 should be nan on both settings of complete_xcats (True and False)
            self.assertTrue(rdf[rdf["real_date"] == "2000-01-17"]["value"].isna().all())

        ## Test Case 3a - Testing weights with nan values
        _xcats: List[str] = ["XR", "CRY", "INFL"]
        weights: List[str] = [1, 2, 4]
        _end: str = "2000-02-01"
        dfd = make_test_df(
            cids=all_cids, xcats=all_xcats, start_date=start, end_date=_end
        )
        dfd["value"] = 1

        # make 2000-01-17 for CAD and AUD XR values a nan
        dfd.loc[
            (dfd["real_date"] == "2000-01-17")
            & ((dfd["cid"] == "CAD") | (dfd["cid"] == "AUD"))
            & ((dfd["xcat"] == "XR") | (dfd["xcat"] == "CRY")),
            "value",
        ] = np.nan

        # also blank out GBP INFL for 2023-01-18
        dfd.loc[
            (dfd["real_date"] == "2000-01-18")
            & (dfd["cid"] == "GBP")
            & (dfd["xcat"] == "INFL"),
            "value",
        ] = np.nan

        rdf: pd.DataFrame = linear_composite(
            df=dfd,
            xcats=_xcats,
            weights=weights,
        )

        # all values should be 1
        self.assertTrue(np.all(rdf["value"].values == 1))

        # testing with normalize_weights=False
        rdf: pd.DataFrame = linear_composite(
            df=dfd,
            xcats=_xcats,
            weights=weights,
            normalize_weights=False,
        )

        test_cases: List[Dict[str, Union[str, float]]] = [
            {
                "cid": "CAD",
                "real_date": "2000-01-17",
                "expc_value": 7.0,
                "outl_value": 4.0,
            },
            {
                "cid": "AUD",
                "real_date": "2000-01-17",
                "expc_value": 7.0,
                "outl_value": 4.0,
            },
            {
                "cid": "GBP",
                "real_date": "2000-01-18",
                "expc_value": 7.0,
                "outl_value": 3.0,
            },
        ]

        for casex in test_cases:
            rdf_casex: pd.DataFrame = rdf[(rdf["cid"] == casex["cid"])].copy()
            # there should only be two values for each cid
            self.assertTrue(
                set(rdf_casex["value"].unique().tolist())
                == {casex["expc_value"], casex["outl_value"]}
            )
            # on the "outlier" date, the value should be 3
            self.assertTrue(
                rdf_casex[rdf_casex["value"] == casex["outl_value"]][
                    "real_date"
                ].values[0]
                == pd.to_datetime(casex["real_date"])
            )
            # all other dates should be expc_value
            self.assertTrue(
                np.all(
                    rdf_casex[rdf["real_date"] != pd.to_datetime(casex["real_date"])][
                        "value"
                    ]
                    == casex["expc_value"]
                )
            )

            # Test Case 3b - Testing weights with nan values

            _xcats: List[str] = ["XR", "CRY"]
            signs: List[str] = [1, -1]

            dfd: pd.DataFrame = make_test_df(
                cids=all_cids, xcats=_xcats, start_date=start, end_date=end
            )
            dfd["value"] = 1

            adf: pd.DataFrame = linear_composite(
                df=dfd,
                xcats=_xcats,
                signs=signs,
                normalize_weights=True,
            )

            bdf: pd.DataFrame = linear_composite(
                df=dfd,
                xcats=_xcats,
                signs=signs,
                normalize_weights=False,
            )
            # both should be the same
            self.assertTrue(np.all(adf["value"].values == bdf["value"].values))
            # all values should be 0
            self.assertTrue(np.all(adf["value"].values == 0))

            # small tweak to the test df
            dfd: pd.DataFrame = make_test_df(
                cids=all_cids,
                xcats=_xcats,
                start_date=start,
                end_date=end,
                prefer="linear",
            )
            adf: pd.DataFrame = linear_composite(
                df=dfd,
                xcats=_xcats,
                signs=signs,
            )
            # all values should be 0
            self.assertTrue(np.all(adf["value"].values == 0))


if __name__ == "__main__":
    unittest.main()
