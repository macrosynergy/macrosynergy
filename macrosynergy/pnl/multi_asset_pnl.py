"""
Multi Asset PnLs combine multiple "Naive" PnLs with limited signal options and disregarding transaction costs.
"""

import warnings
from itertools import product
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

from macrosynergy.management.simulate import make_qdf
from macrosynergy.management.utils import reduce_df, update_df
from macrosynergy.panel.make_zn_scores import make_zn_scores
from macrosynergy.signal import SignalReturnRelations
from macrosynergy.pnl import NaivePnL


class MultiAssetPNL:

    def __init__(self, pnls: dict, names: dict = None, weights: dict = None):
        self.pnls = pnls
        self.names = names
        self._validate_pnls()

        self.multi_asset_pnl = None

    def combine_pnls(self, weights: dict = None) -> pd.DataFrame:
        """
        Combine the PnLs in the list with the given weights.
        """
        # Normalize weights
        # weights = np.array(weights) / np.array(weights).sum()

        multiasset_df = []
        for asset_name, asset_pnl in self.pnls.items():
            asset_pnl_name = asset_pnl.pnl_names[0]
            asset_data = asset_pnl.df
            asset_signal = asset_pnl.pnl_params[asset_pnl_name].signal

            single_asset_df = asset_data.loc[
                (asset_data["xcat"] == asset_pnl_name) & (asset_data["cid"] == "ALL")
            ].assign(asset=asset_name)

            multiasset_df.append(single_asset_df)

        multiasset_df = pd.concat(multiasset_df, axis=0, ignore_index=True)

        raw_pnls = multiasset_df.set_index(["real_date", "asset"])["value"].unstack()

        # Default weights for each strategy
        start_weights = pd.DataFrame(
            {
                asset_name: weights[asset_name]
                for asset_name in raw_pnls.columns
            },
            index=raw_pnls.index,
        )

        # Daily change in portfolio weights due to previous returns since the last rebalancing
        weights_change = (
            (1 + raw_pnls / 100).groupby(pd.Grouper(freq="M")).cumprod()
        )  # in decimals, not percentage, gross amount
        weights_change = (
            weights_change.groupby(pd.Grouper(freq="M"))
            .shift(periods=1)
            .fillna(value=1)
        )

        # Dynamic weights
        final_weights = start_weights * weights_change
        final_weights = final_weights.div(final_weights.sum(axis=1), axis=0)

        # final calculation
        multiasset_rets = (final_weights * raw_pnls).sum(axis=1)
        multiasset_rets = pd.concat(
            [raw_pnls, multiasset_rets], axis=1, ignore_index=False
        ).rename(columns={0: "NaivePnLCombo"})

        out = {
            "pnls": multiasset_rets,
            "weights": final_weights,
        }

        return out

    def _validate_pnls(self):

        for name, pnl in self.pnls.items():
            if not isinstance(pnl, NaivePnL):
                raise ValueError("All elements in the list must be NaivePnL objects.")
            if len(pnl.pnl_names) > 1:
                raise ValueError("The NaivePnL object must have only one signal.")
            if name not in self.names:
                raise ValueError(
                    "The name of the NaivePnL object must be in the names list."
                )

        if len(self.pnls) != len(self.names):
            raise ValueError("The number of PnLs and names must be the same.")
        return True


if __name__ == "__main__":
    cids = ["AUD", "CAD", "GBP", "NZD", "USD", "EUR"]
    xcats = ["EQXR_NSA", "FXXR", "GROWTH", "INFL", "DUXR"]

    cols_1 = ["earliest", "latest", "mean_add", "sd_mult"]
    df_cids = pd.DataFrame(index=cids, columns=cols_1)
    df_cids.loc["AUD", :] = ["2008-01-03", "2020-12-31", 0.5, 2]
    df_cids.loc["CAD", :] = ["2010-01-03", "2020-11-30", 0, 1]
    df_cids.loc["GBP", :] = ["2012-01-03", "2020-11-30", -0.2, 0.5]
    df_cids.loc["NZD"] = ["2002-01-03", "2020-09-30", -0.1, 2]
    df_cids.loc["USD"] = ["2015-01-03", "2020-12-31", 0.2, 2]
    df_cids.loc["EUR"] = ["2008-01-03", "2020-12-31", 0.1, 2]

    cols_2 = cols_1 + ["ar_coef", "back_coef"]

    df_xcats = pd.DataFrame(index=xcats, columns=cols_2)
    df_xcats.loc["EQXR_NSA"] = ["2000-01-03", "2020-12-31", 0.1, 1, 0, 0.3]
    df_xcats.loc["FXXR"] = ["2000-01-01", "2020-10-30", 1, 2, 0.95, 1]
    df_xcats.loc["GROWTH"] = ["2010-01-03", "2020-10-30", 1, 2, 0.9, 1]
    df_xcats.loc["INFL"] = ["2001-01-01", "2020-10-30", 1, 2, 0.8, 0.5]
    df_xcats.loc["DUXR"] = ["2000-01-01", "2020-12-31", 0.1, 0.5, 0, 0.1]

    black = {"AUD": ["2006-01-01", "2015-12-31"], "GBP": ["2022-01-01", "2100-01-01"]}
    dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)

    # Instantiate a new instance to test the long-only functionality.
    # Benchmarks are used to calculate correlation against PnL series.
    pnl_eq = NaivePnL(
        dfd,
        ret="EQXR_NSA",
        sigs=["GROWTH"],
        cids=cids,
        start="2000-01-01",
        blacklist=black,
        bms=["EUR_EQXR_NSA", "USD_EQXR_NSA"],
    )

    pnl_eq.make_pnl(
        sig="GROWTH",
        sig_op="zn_score_pan",
        sig_neg=False,
        sig_add=0.5,
        rebal_freq="monthly",
        vol_scale=5,
        rebal_slip=1,
        min_obs=250,
        thresh=2,
    )

    # pnl_eq.make_long_pnl(vol_scale=10, label="Long")

    pnl_fx = NaivePnL(
        dfd,
        ret="FXXR",
        sigs=["INFL"],
        cids=cids,
        start="2000-01-01",
        blacklist=black,
        bms=["EUR_EQXR_NSA", "USD_EQXR_NSA"],
    )

    pnl_fx.make_pnl(
        sig="INFL",
        sig_op="zn_score_pan",
        sig_neg=True,
        sig_add=0.5,
        rebal_freq="monthly",
        vol_scale=5,
        rebal_slip=1,
        min_obs=250,
        thresh=2,
    )

    # pnl_fx.make_long_pnl(vol_scale=10, label="Long")
    print(pnl_fx.pnl_names)
    # df_eval = pnl_fx.evaluate_pnls(
    #     pnl_cats=["PNL_GROWTH_NEG", "PNL_INFL_NEG"], start="2015-01-01", end="2020-12-31"
    # )
    pnl_dict = {"FX": pnl_fx, "EQ": pnl_eq}
    mapnl = MultiAssetPNL(pnl_dict, names={"FX": "FX", "EQ": "EQ"})
    multiasset_analysis = mapnl.combine_pnls(weights={"FX": 1, "EQ": 1})
    multiasset_analysis["pnls"].cumsum().plot()
    plt.show()
    pass
