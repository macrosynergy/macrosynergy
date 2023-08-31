import unittest
import numpy as np
import pandas as pd
from typing import List, Tuple, Union, Dict, Any

from macrosynergy.panel.linear_composite import linear_composite
from macrosynergy.management.simulate_quantamental_data import make_test_df


class TestAll(unittest.TestCase):
    def generate_test_dfs(self) -> pd.DataFrame:
        self.cids: List[str] = ["AUD", "CAD", "GBP"]
        self.xcats: List[str] = ["CRY", "XR", "INFL"]

        dfd: pd.DataFrame = make_test_df(
            cids=self.cids,
            xcats=self.xcats,
            start_date="2000-01-01",
            end_date="2020-12-31",
        )
        self.dfd: pd.DataFrame = dfd

    def test_linear_composite_args(self):
        """
        This section is meant to test the arguments of the linear_composite function.
        It simply runs the function with a set of arguments and checks for errors.
        No return values are checked.
        """

        self.generate_test_dfs()
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
            {"cids": [self.cids[0], "foo"]},
            {"xcats": self.xcats[0], "weights": "foo"},
            {"weights": "foo"},
            {"weights": [1] * 100},
            {"signs": [1] * 100},
            {"weights": [1] * (len(self.xcats) - 1) + [0]},
            {"signs": [1] * (len(self.xcats) - 1) + [0]},
        ]
        for case in value_error_cases:
            argsx: Dict[str, Any] = base_args.copy()
            argsx.update(case)
            with self.assertRaises(ValueError, msg=f"Failed on case: {case}"):
                rdf: pd.DataFrame = linear_composite(**argsx)

        bad_df: pd.DataFrame = self.dfd.copy()
        # remove any entries for AUD XR
        bad_df = bad_df[
            (bad_df["cid"] != self.cids[0]) | (bad_df["xcat"] != self.xcats[1])
        ].reset_index(drop=True)

        with self.assertWarns(UserWarning):
            argsx: Dict[str, Any] = base_args.copy()
            argsx.update({"df": bad_df})
            rdf: pd.DataFrame = linear_composite(**argsx)

        # Case with agg_cid, with only one xcat (no weights), with one cid missing the xcat
        _test_df: pd.DataFrame = make_test_df(
            cids=self.cids,
            xcats=self.xcats,
            start_date="2000-01-01",
            end_date="2000-02-01",
        )

        _test_df = _test_df[
            ~((_test_df["cid"] == "AUD") & (_test_df["xcat"] == "XR"))
        ].reset_index(drop=True)

        with self.assertRaises(ValueError):
            rdf: pd.DataFrame = linear_composite(
                df=_test_df, xcats="XR", cids=self.cids
            )

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

        _test_df: pd.DataFrame = make_test_df(
            cids=self.cids,
            xcats=self.xcats,
            start_date="2000-01-01",
            end_date="2000-02-01",
        )

        _test_df = _test_df[
            ~((_test_df["cid"] == "AUD") & (_test_df["xcat"] == "XR"))
        ].reset_index(drop=True)

        with self.assertWarns(UserWarning):
            rdf: pd.DataFrame = linear_composite(
                df=_test_df, xcats="INFL", weights="XR", cids=self.cids
            )

    def test_linear_composite_xcat_agg_mode(self):
        """
        Meant to test the "xcat_agg" mode of the linear_composite function.
        (i.e. engaging `_linear_composite_xcat_agg()`)
        """

        self.generate_test_dfs()

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
                    rdf_casex[
                        rdf_casex["real_date"] != pd.to_datetime(casex["real_date"])
                    ]["value"]
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
                style="linear",
            )
            adf: pd.DataFrame = linear_composite(
                df=dfd,
                xcats=_xcats,
                signs=signs,
            )
            # all values should be 0
            self.assertTrue(np.all(adf["value"].values == 0))

    def test_linear_composite_cid_agg_mode(self):
        """
        Meant to test the "cid_agg" mode of the linear_composite function.
        (i.e. engaging `_linear_composite_cid_agg()`)
        """
        self.generate_test_dfs()

        all_cids: List[str] = ["AUD", "CAD", "GBP"]
        all_xcats: List[str] = ["XR", "CRY", "INFL"]
        start: str = "2000-01-01"
        end: str = "2001-01-01"

        # Test Case 1a - Testing basic functionality
        dfd: pd.DataFrame = make_test_df(
            cids=all_cids, xcats=all_xcats, start_date=start, end_date=end
        )

        dfd["value"] = 1

        _xcat: str = all_xcats[0]
        new_cid_name: str = "TEST-CIDX"
        rdf: pd.DataFrame = linear_composite(
            df=dfd,
            cids=all_cids,
            xcats=_xcat,
            new_cid=new_cid_name,
        )

        # all values should be 1
        self.assertTrue(np.all(rdf["value"].values == 1))
        self.assertTrue(np.all(rdf["cid"].unique().tolist() == [new_cid_name]))
        self.assertTrue(np.all(rdf["xcat"].unique().tolist() == [_xcat]))

        _cids: List[str] = ["AUD", "CAD"]
        signs: List[str] = [1, -1]

        rdf: pd.DataFrame = linear_composite(
            df=dfd,
            cids=_cids,
            xcats=_xcat,
            signs=signs,
        )

        # all values should be 0
        self.assertTrue(np.all(rdf["value"].values == 0))

        # Test Case 1b - Testing weights and signs w/ & w/o normalization

        _cids: List[str] = ["AUD", "CAD"]
        _weights: List[str] = [1, 2]
        _signs: List[str] = [1, -1]

        rdf: pd.DataFrame = linear_composite(
            df=dfd,
            cids=_cids,
            xcats=_xcat,
            weights=_weights,
            signs=_signs,
            normalize_weights=False,
        )

        # all values should be -1 (1*1 + -1*2)
        self.assertTrue(np.all(rdf["value"].values == -1))

        rdf: pd.DataFrame = linear_composite(
            df=dfd,
            cids=_cids,
            xcats=_xcat,
            weights=_weights,
            signs=_signs,
        )
        # all should be -1/3 -- sum of weights is 3, so 1/3 - 2/3 = -1/3
        self.assertTrue(np.all(rdf["value"].values == -1 / 3))

        # Test Case 2a - Testing nan logic

        _cids: List[str] = all_cids
        _xcat: str = "XR"
        _weights: List[str] = [1, 2, 4]

        dfd: pd.DataFrame = make_test_df(
            cids=all_cids, xcats=all_xcats, start_date=start, end_date=end
        )
        dfd["value"] = 1
        for n_w, r_v in zip([True, False], [1, 7]):
            rdf: pd.DataFrame = linear_composite(
                df=dfd,
                cids=_cids,
                xcats=_xcat,
                weights=_weights,
                normalize_weights=n_w,
            )
            self.assertTrue(np.all(rdf["value"].values == r_v))

        # Test Case 2b - Testing nan logic

        # make 2000-01-17 for all CAD & AUD-XR values a nan
        _cids: List[str] = ["AUD", "CAD", "GBP"]
        _xcat: str = "XR"
        _weights: List[str] = [1, 2, 4]
        _end: str = "2000-02-01"
        dfd: pd.DataFrame = make_test_df(
            cids=all_cids, xcats=all_xcats, start_date=start, end_date=_end
        )
        dfd["value"] = 1
        dfd.loc[
            (dfd["real_date"] == "2000-01-17")
            & ((dfd["cid"] == "CAD") | (dfd["cid"] == "AUD"))
            & (dfd["xcat"] == "XR"),
            "value",
        ] = np.nan

        # make 2000-01-18 for all GBP-XR values a nan
        dfd.loc[
            (dfd["real_date"] == "2000-01-18")
            & (dfd["cid"] == "GBP")
            & (dfd["xcat"] == "XR"),
            "value",
        ] = np.nan

        # for 2023-01-19, all entires are nan
        dfd.loc[(dfd["real_date"] == "2000-01-19"), "value"] = np.nan

        rdf: pd.DataFrame = linear_composite(
            df=dfd, cids=_cids, xcats=_xcat, complete_cids=True, weights=_weights
        )
        # chekc that all the dates in the input are there in the output, cast to pd.Timestamp
        self.assertTrue(
            set(rdf["real_date"].unique().tolist())
            == set(pd.to_datetime(dfd["real_date"].unique().tolist()))
        )
        interesting_dates: List[str] = ["2000-01-17", "2000-01-18", "2000-01-19"]
        # with complete_cids=True, 2000-01-17 to 2000-01-19 should be nan
        self.assertTrue(
            rdf[rdf["real_date"].isin(interesting_dates)]["value"].isna().all()
        )

        # for all other dates, the value should be 1
        self.assertTrue(
            np.all(rdf[~rdf["real_date"].isin(interesting_dates)]["value"].values == 1)
        )

        rdf: pd.DataFrame = linear_composite(
            df=dfd,
            cids=_cids,
            xcats=_xcat,
            complete_cids=False,
            weights=_weights,
            normalize_weights=False,
        )

        self.assertTrue(
            rdf[rdf["real_date"].isin(["2000-01-19"])]["value"].isna().all()
        )

        self.assertTrue(
            np.all(
                rdf[~rdf["real_date"].isin(interesting_dates)]["value"].values
                == sum(_weights)
            )
        )

        # value on 2000-01-17 should be 4:
        self.assertTrue(
            rdf[rdf["real_date"].isin(["2000-01-17"])]["value"].values[0] == 4
        )
        # value on 2000-01-18 should be 3
        self.assertTrue(
            rdf[rdf["real_date"].isin(["2000-01-18"])]["value"].values[0] == 3
        )

        ## Testing category-weights
        _cids: List[str] = ["AUD", "CAD", "GBP"]
        _xcat: str = "XR"
        _weights: str = "INFL"
        _end: str = "2000-02-01"
        dfd: pd.DataFrame = make_test_df(
            cids=all_cids, xcats=all_xcats, start_date=start, end_date=_end
        )
        dfd["value"] = 1

        dfd.loc[
            (dfd["real_date"] == "2000-01-17") & (dfd["xcat"] == "INFL"),
            "value",
        ] = 10

        rdf: pd.DataFrame = linear_composite(
            df=dfd,
            cids=_cids,
            xcats=_xcat,
            weights=_weights,
            normalize_weights=False,
        )

        # value on 2000-01-17 should be 30:
        self.assertTrue(
            rdf[rdf["real_date"].isin(["2000-01-17"])]["value"].values[0] == 30
        )
        # all else should be 3
        self.assertTrue(
            np.all(rdf[~rdf["real_date"].isin(["2000-01-17"])]["value"].values == 3)
        )
        self.assertTrue(rdf["cid"].unique().tolist() == ["GLB"])
        self.assertTrue(rdf["xcat"].unique().tolist() == ["XR"])

        # with normalized wieghts, all values should be 1

        rdf: pd.DataFrame = linear_composite(
            df=dfd,
            cids=_cids,
            xcats=_xcat,
            weights=_weights,
        )

        self.assertTrue(np.all(rdf["value"].values == 1))
        self.assertTrue(rdf["cid"].unique().tolist() == ["GLB"])
        self.assertTrue(rdf["xcat"].unique().tolist() == ["XR"])

        # Test Case 3a - Testing signs with nan values
        _cids: List[str] = ["AUD", "CAD", "GBP"]
        _xcat: str = "XR"
        _weights: str = "INFL"
        _end: str = "2000-02-01"
        dfd: pd.DataFrame = make_test_df(
            cids=all_cids, xcats=all_xcats, start_date=start, end_date=_end
        )
        dfd["value"] = 1

        # set AUD INFL on 2000-01-17 to nan
        # set GBP XR on 2000-01-17 to nan
        dfd.loc[
            (
                (
                    (dfd["real_date"] == "2000-01-17")
                    & (dfd["xcat"] == "INFL")
                    & (dfd["cid"] == "AUD")
                )
                | (
                    (dfd["real_date"] == "2000-01-17")
                    & (dfd["xcat"] == "XR")
                    & (dfd["cid"] == "GBP")
                )
            ),
            "value",
        ] = np.nan

        rdf: pd.DataFrame = linear_composite(
            df=dfd,
            cids=_cids,
            xcats=_xcat,
            weights=_weights,
            normalize_weights=False,
        )
        # on 2023-01-17, the value should be 1, all else should be 3
        self.assertTrue(
            np.all(rdf[rdf["real_date"] == "2000-01-17"]["value"].values == 1)
        )
        self.assertTrue(
            np.all(rdf[rdf["real_date"] != "2000-01-17"]["value"].values == 3)
        )

        # Test Case 3b - Testing signs with nan values
        _cids: List[str] = [
            "AUD",
            "CAD",
        ]
        signs: List[str] = [1, -1]
        _xcat: str = "XR"
        _weights: str = "INFL"
        _end: str = "2000-02-01"
        dfd: pd.DataFrame = make_test_df(
            cids=all_cids, xcats=all_xcats, start_date=start, end_date=_end
        )
        # for each, mutiply the weight by the sign
        dfd["value"] = 1

        # set AUD INFL on 2000-01-17 to nan, set complete_cids=True
        dfd.loc[
            (dfd["real_date"] == "2000-01-17")
            & (dfd["xcat"] == "INFL")
            & (dfd["cid"] == "AUD"),
            "value",
        ] = np.nan

        rdf: pd.DataFrame = linear_composite(
            df=dfd,
            cids=_cids,
            xcats=_xcat,
            weights=_weights,
            signs=signs,
            complete_cids=True,
        )

        # sum of the value col should be 0, with only 1 nan value on 2000-01-17
        self.assertTrue(np.all(rdf.groupby("real_date")["value"].sum().values == 0))
        self.assertTrue(
            np.all(rdf[rdf["real_date"] == "2000-01-17"]["value"].isna().values)
        )
        self.assertTrue(
            np.all(rdf[rdf["real_date"] != "2000-01-17"]["value"].values == 0)
        )

        rdf: pd.DataFrame = linear_composite(
            df=dfd,
            cids=_cids,
            xcats=_xcat,
            weights=_weights,
            signs=signs,
            complete_cids=False,
        )

        # sum of the value col should be -1, with -1 on 2000-01-17 and the rest 0
        self.assertTrue(sum(rdf["value"].values) == -1)
        self.assertTrue(
            np.all(rdf[rdf["real_date"] == "2000-01-17"]["value"].values == -1)
        )
        self.assertTrue(
            np.all(rdf[rdf["real_date"] != "2000-01-17"]["value"].values == 0)
        )


if __name__ == "__main__":
    unittest.main()
