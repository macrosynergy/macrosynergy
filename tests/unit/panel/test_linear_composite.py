import unittest
import numpy as np
import pandas as pd
import warnings
from typing import List, Union, Dict, Any

from macrosynergy.panel.linear_composite import (
    linear_composite,
    _missing_cids_xcats_str,
)
from macrosynergy.management.simulate import make_test_df


class TestAll(unittest.TestCase):
    def setUp(self) -> pd.DataFrame:
        self.cids: List[str] = ["AUD", "CAD", "GBP"]
        self.xcats: List[str] = ["CRY", "XR", "INFL"]

        dfd: pd.DataFrame = make_test_df(
            cids=self.cids,
            xcats=self.xcats,
            start="2000-01-01",
            end="2020-12-31",
        )
        self.dfd: pd.DataFrame = dfd

    def test_linear_composite_args(self):
        """
        This section is meant to test the arguments of the linear_composite function.
        It simply runs the function with a set of arguments and checks for errors.
        No return values are checked.
        """

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
            {"df": pd.DataFrame()},
            {"df": self.dfd.assign(value=np.NaN)},
            {"normalize_weights": "foo"},
            {"complete_cids": "foo"},
            {"complete_xcats": "foo"},
            {"new_cid": 1},
            {"new_xcat": 1},
            {"blacklist": 1},
        ]
        for case in type_error_cases:
            argsx: Dict[str, Any] = base_args.copy()
            argsx.update(case)
            with self.assertRaises(TypeError):
                rdf: pd.DataFrame = linear_composite(**argsx)

        # value error cases
        value_error_cases: List[Dict[str, Any]] = [
            {"start": 1},
            {"end": 1},
            {"xcats": ["foo"]},
            {"cids": ["bar"]},
            {"cids": [self.cids[0], "foo"], "complete_cids": True},
            {"xcats": [self.xcats[0], "foo"], "complete_xcats": True},
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
            start="2000-01-01",
            end="2000-02-01",
        )

        _test_df = _test_df[
            ~((_test_df["cid"] == "AUD") & (_test_df["xcat"] == "XR"))
        ].reset_index(drop=True)

        with self.assertRaises(ValueError):
            rdf: pd.DataFrame = linear_composite(
                df=_test_df, xcats="XR", cids=self.cids, complete_cids=True
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
            start="2000-01-01",
            end="2000-02-01",
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

        all_cids: List[str] = ["AUD", "CAD", "GBP"]
        all_xcats: List[str] = ["XR", "CRY", "INFL"]
        start: str = "2000-01-01"
        end: str = "2001-01-01"

        ## Test Case 1a - Testing non-normalized weights
        dfd = make_test_df(cids=all_cids, xcats=all_xcats, start=start, end=end)
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
        dfd = make_test_df(cids=all_cids, xcats=all_xcats, start=start, end=_end)
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
        dfd = make_test_df(cids=all_cids, xcats=all_xcats, start=start, end=_end)
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
                cids=all_cids, xcats=_xcats, start=start, end=end
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
                start=start,
                end=end,
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

        all_cids: List[str] = ["AUD", "CAD", "GBP"]
        all_xcats: List[str] = ["XR", "CRY", "INFL"]
        start: str = "2000-01-01"
        end: str = "2001-01-01"

        # Test Case 1a - Testing basic functionality
        dfd: pd.DataFrame = make_test_df(
            cids=all_cids, xcats=all_xcats, start=start, end=end
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
            cids=all_cids, xcats=all_xcats, start=start, end=end
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
            cids=all_cids, xcats=all_xcats, start=start, end=_end
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

        # check that all the dates in the input are there in the output, cast to pd.Timestamp
        self.assertTrue(
            set(rdf["real_date"].unique().tolist())
            == set(dfd["real_date"].unique().tolist())
        )
        interesting_dates: List[pd.Timestamp] = [
            pd.to_datetime(x) for x in ["2000-01-17", "2000-01-18", "2000-01-19"]
        ]
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
            rdf[rdf["real_date"].isin([pd.Timestamp("2000-01-19")])]["value"]
            .isna()
            .all()
        )

        self.assertTrue(
            np.all(
                rdf[~rdf["real_date"].isin(interesting_dates)]["value"].values
                == sum(_weights)
            )
        )

        # value on 2000-01-17 should be 4:
        self.assertTrue(
            rdf[rdf["real_date"].isin([pd.Timestamp("2000-01-17")])]["value"].values[0]
            == 4
        )
        # value on 2000-01-18 should be 3
        self.assertTrue(
            rdf[rdf["real_date"].isin([pd.Timestamp("2000-01-18")])]["value"].values[0]
            == 3
        )

        ## Testing category-weights
        _cids: List[str] = ["AUD", "CAD", "GBP"]
        _xcat: str = "XR"
        _weights: str = "INFL"
        _end: str = "2000-02-01"
        dfd: pd.DataFrame = make_test_df(
            cids=all_cids, xcats=all_xcats, start=start, end=_end
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
            rdf[rdf["real_date"].isin([pd.Timestamp("2000-01-17")])]["value"].values[0]
            == 30
        )
        # all else should be 3
        self.assertTrue(
            np.all(
                rdf[~rdf["real_date"].isin([pd.Timestamp("2000-01-17")])][
                    "value"
                ].values
                == 3
            )
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
            cids=all_cids, xcats=all_xcats, start=start, end=_end
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
            cids=all_cids, xcats=all_xcats, start=start, end=_end
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

    def test_linear_composite_err_msg(self):

        cids = ["AUD", "CAD", "GBP"]
        xcats = ["XR", "CRY", "INFL"]

        df = make_test_df(cids=cids, xcats=xcats)

        # Test Case 1 - test if a missing xcat is included in the error message
        df_test = df[~((df["cid"] == "AUD") & (df["xcat"] == "XR"))].reset_index(
            drop=True
        )

        err_str = _missing_cids_xcats_str(
            df=df_test,
            cids=cids,
            xcats=xcats,
        )

        expc_start = "The following `cids` are missing for the respective `xcats`:"
        self.assertTrue(err_str.startswith(expc_start))
        self.assertTrue(err_str.endswith("XR:  ['AUD']"))

        # test with complete xcat missing
        df_test = df[~(df["xcat"] == "XR")].reset_index(drop=True)

        err_str = _missing_cids_xcats_str(
            df=df_test,
            cids=cids,
            xcats=xcats,
        )

        expc_start = f"Missing xcats: {['XR']}"
        self.assertTrue(err_str.startswith(expc_start))

        expc_end = "XR:  ['AUD', 'CAD', 'GBP']"
        self.assertTrue(err_str.split("\n")[-1] == expc_end)

    def test_complete_cids(self):
        cids = ["AUD", "CAD"]
        xcats = ["XR", "CRY", "INFL"]

        df = make_test_df(cids=cids, xcats=xcats)

        cids = ["GBP", "CAD", "AUD"]

        # If complete_cids is True, then error should be thrown since not all cids are in
        # the df
        with self.assertRaises(ValueError):
            lc_df = linear_composite(df=df, cids=cids, xcats=xcats, complete_cids=True)

        lc_df = linear_composite(
            df=df,
            cids=cids,
            xcats="XR",
            weights=[1, 7, 9],
            signs=[1, -1, 1],
            complete_cids=False,
        )
        lc_values = lc_df["value"].values

        aud_values = (
            df[(df["xcat"] == "XR") & (df["cid"] == "AUD")]["value"]
            .reset_index(drop=True)
            .values
        )
        cad_values = (
            df[(df["xcat"] == "XR") & (df["cid"] == "CAD")]["value"]
            .reset_index(drop=True)
            .values
        )
        expected_values = (9 * aud_values - 7 * cad_values) / 16

        self.assertTrue(np.allclose(lc_values, expected_values))

        # Test when weight is a category

        # Initially ensure that an error is thrown if a cid doesn't exist for the weight category
        df_1 = df[~((df["xcat"] == "INFL") & (df["cid"] == "AUD"))].reset_index(
            drop=True
        )

        with self.assertRaises(ValueError):
            lc_df = linear_composite(df=df_1, cids=cids, xcats=xcats, weights="INFL")

        lc_df = linear_composite(
            df=df,
            cids=cids,
            xcats="XR",
            weights="INFL",
            signs=[1, -1, 1],
            complete_cids=False,
        )
        lc_values = lc_df["value"].values

        aud_values = (
            df[(df["xcat"] == "XR") & (df["cid"] == "AUD")]["value"]
            .reset_index(drop=True)
            .values
        )
        cad_values = (
            df[(df["xcat"] == "XR") & (df["cid"] == "CAD")]["value"]
            .reset_index(drop=True)
            .values
        )
        aud_infl_values = (
            df[(df["xcat"] == "INFL") & (df["cid"] == "AUD")]["value"]
            .reset_index(drop=True)
            .values
        )
        cad_infl_values = (
            df[(df["xcat"] == "INFL") & (df["cid"] == "CAD")]["value"]
            .reset_index(drop=True)
            .values
        )

        expected_values = aud_values * aud_infl_values - cad_values * cad_infl_values

        weight_magnitude = np.abs(aud_infl_values) + np.abs(cad_infl_values)
        # Avoid division by zero, since the weights are both zero the magnitude does not matter in this case
        weight_magnitude[weight_magnitude == 0] = 1

        expected_values = expected_values / weight_magnitude

        self.assertTrue(np.allclose(lc_values, expected_values))

    def test_complete_xcats(self):
        cids = ["GBP", "AUD", "CAD"]
        xcats = ["XR", "CRY"]

        df = make_test_df(cids=cids, xcats=xcats)

        xcats = ["XR", "CRY", "INFL"]

        # If complete_xcats is True, then error should be thrown since not all xcats are
        # in the df

        with self.assertRaises(ValueError):
            lc_df = linear_composite(df=df, cids=cids, xcats=xcats, complete_xcats=True)

        lc_df = linear_composite(
            df=df,
            cids=cids,
            xcats=xcats,
            weights=[4, 3, 9],
            signs=[1, -1, 1],
            complete_xcats=False,
        )
        lc_values = lc_df["value"].values

        xr_values = df[(df["xcat"] == "XR")]["value"].reset_index(drop=True).values
        cry_values = df[(df["xcat"] == "CRY")]["value"].reset_index(drop=True).values
        expected_values = (4 * xr_values - 3 * cry_values) / 7

        self.assertTrue(np.allclose(lc_values, expected_values))

    def test_args(self):
        cids = ["GBP", "AUD", "CAD"]
        xcats = ["XR", "CRY"]

        df = make_test_df(cids=cids, xcats=xcats)

        with self.assertRaises(ValueError) as e_handler:
            # contains non-existent xcat
            bad_xcats = ["XR", "CRY", "INFL", "GDP"]
            _ = linear_composite(df=df, cids=cids, xcats=bad_xcats, complete_xcats=True)
            self.assertTrue(
                "Not all `xcats` are available in `df`." in str(e_handler.exception)
            )

        with self.assertRaises(ValueError) as e_handler:
            # contains non-existent weight category
            _ = linear_composite(df=df, cids=cids, xcats=xcats[0], weights="INFL")
            # 'When using a category-string as `weights` it must be present in `df`'
            # check the error message
            self.assertTrue(
                "When using a category-string as `weights` it must be present in `df`"
                in str(e_handler.exception)
            )

        with warnings.catch_warnings(record=True) as w:
            # extra xcat in xcat agg mode should raise a warning
            bad_xcats = ["XR", "CRY", "INFL", "GDP"]
            _ = linear_composite(
                df=df, cids=cids, xcats=bad_xcats, complete_xcats=False
            )
            self.assertTrue(len(w) == 1)

        with self.assertRaises(ValueError) as e_handler:
            bad_cids = ["GBP", "AUD", "CAD", "GDP"]
            _ = linear_composite(
                df=df, cids=bad_cids, xcats=xcats[0], complete_cids=True
            )
            err_str = "ValueError: Not all `cids` are available in `df`"
            self.assertTrue(err_str in str(e_handler.exception))

        with warnings.catch_warnings(record=True) as w:
            bad_cids = ["GBP", "AUD", "CAD", "USD"]
            _ = linear_composite(
                df=df,
                cids=bad_cids,
                xcats=xcats[0],
                complete_cids=False,
            )
            err_str = "Not all `cids` are available in `df`: ['USD'] The calculation will be performed with the available cids"
            self.assertTrue(len(w) == 1)
            self.assertTrue(err_str in str(w[0].message))


if __name__ == "__main__":
    unittest.main()
