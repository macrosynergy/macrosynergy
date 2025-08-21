import unittest
import numpy as np
import pandas as pd
import warnings
from typing import List, Union, Dict, Any

from macrosynergy.panel.cross_asset_effects import cross_asset_effects
from macrosynergy.management.simulate import make_test_df


class TestAll(unittest.TestCase):
    def setUp(self) -> pd.DataFrame:
        self.cids: List[str] = ["AUD", "CAD", "GBP"]
        self.xcats: List[str] = ["X1", "X2", "X3", "V1", "V2", "V3"]

        dfd: pd.DataFrame = make_test_df(
            cids=self.cids,
            xcats=self.xcats,
            start="2000-01-01",
            end="2020-12-31",
        )
        self.dfd: pd.DataFrame = dfd

    def test_cross_asset_effects(self):
        """

        """
        base_args = {
            "df": self.dfd,
            "cids": self.cids,
            "effect_name": "TESTEFFECT",
        }
        assertions_inputs = [
            # (signal_xcats, weights_xcats, signal_signs)
            (
                {"asset1": "X1", "asset2": "X2"}, {"asset1": "V1"}, None,
            ),
            (
                {"asset1": "X1", "asset2": "X2"}, {"asset1": "V1", "asset3": "V2"}, None,
            ),
            (
                {"asset1": "X1", "asset2": "X2"}, {"asset1": "V1", "asset2": "V2"}, {"asset1": 1},
            ),
            (
                {"asset1": "X1", "asset2": "X2"}, {"asset1": "V1", "asset2": "V2"}, {"asset1": 1, "asset3": 1},
            ),
        ]
        for (signal_xcats, weights_xcats, signal_signs) in assertions_inputs:
            argsx: Dict[str, Any] = base_args.copy()
            argsx.update(
                dict(
                    zip(
                        ["signal_xcats", "weights_xcats", "signal_signs"],
                        [signal_xcats, weights_xcats, signal_signs]
                    )

                )
            )
            with self.assertRaises(AssertionError):
                rdf: pd.DataFrame = cross_asset_effects(**argsx)

        rdf: pd.DataFrame = cross_asset_effects(
            signal_xcats={"asset1": "X1", "asset2": "X2", "asset3": "X3"},
            weights_xcats={"asset1": "V1", "asset2": "V2", "asset3": "V3"},
            signal_signs={"asset1": 1, "asset2": 1, "asset3": 1},
            **base_args.copy()
        )
        # any return data implies success
        self.assertTrue(not rdf.empty)

        rdf_neg: pd.DataFrame = cross_asset_effects(
            signal_xcats={"asset1": "X1", "asset2": "X2", "asset3": "X3"},
            weights_xcats={"asset1": "V1", "asset2": "V2", "asset3": "V3"},
            signal_signs={"asset1": -1, "asset2": -1, "asset3": -1},
            **base_args.copy()
        )

        # base case
        compare_df = pd.merge(
            rdf, rdf_neg,
            on=["cid", "xcat", "real_date"],
            how="outer",
            suffixes=["", "_neg"]
        )

        mask = compare_df["xcat"] == "TESTEFFECT"
        self.assertTrue(
            (
                    (
                            compare_df.loc[mask, "value"].values + compare_df.loc[mask, "value_neg"]
                    ) < 1e-6
            ).all()
        )


if __name__ == "__main__":
    unittest.main()
