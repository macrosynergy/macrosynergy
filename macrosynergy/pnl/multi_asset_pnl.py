"""
Multi Asset PnLs combine multiple "Naive" PnLs with limited signal options and disregarding transaction costs.
"""

from functools import reduce
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


class MultiAssetPnL:

    def __init__(self):
        # self.pnls = pnls
        # self.pnl_xcats = pnl_xcats
        # self._validate_pnls()

        self.pnls_df = pd.DataFrame()
        self.single_asset_pnls = {}
        self._multi_asset_xcats = []

    def combine_pnls(
        self, pnls: dict[str, NaivePnL], weights: dict = None, combined_xcat: str = None
    ) -> pd.DataFrame:
        """
        Combine the PnLs in the list with the given weights.
        """
        # Todo: check if pnls already in self.single_asset_pnls
        self.single_asset_pnls = {**self.single_asset_pnls, **pnls}
        self._validate_pnls()
        # Default combined_xcat
        combined_xcat = (
            "_".join(self.single_asset_pnls.values())
            if combined_xcat is None
            else combined_xcat
        )
        # Default weights
        if weights is None:
            weights = {pnl_name: 1 for pnl_name in pnls.keys()}
        weights = self._normalize_weights(weights)

        multiasset_df = []
        # for asset_name, asset_pnl in self.pnls.items():
        #     asset_pnl_xcat = self.pnl_xcats[asset_name]
        #     single_asset_df = asset_pnl.pnl_df([asset_pnl_xcat]).assign(
        #         asset=asset_name
        #     )
        #     multiasset_df.append(single_asset_df)
        for pnl_xcat, pnl in pnls.items():
            single_asset_df = pnl.pnl_df([pnl_xcat]).assign(asset=pnl.ret)
            multiasset_df.append(single_asset_df)

        multiasset_df = pd.concat(multiasset_df, axis=0, ignore_index=True)

        raw_pnls = multiasset_df.set_index(["real_date", "xcat"])["value"].unstack()

        # Default weights for each strategy
        start_weights = pd.DataFrame(
            {asset_name: weights[asset_name] for asset_name in raw_pnls.columns},
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
        ).rename(columns={0: combined_xcat})

        multi_asset_pnl = multiasset_rets.reset_index().melt(
            id_vars=["real_date"], var_name="xcat", value_name="value"
        )
        multi_asset_pnl = multi_asset_pnl.sort_values(by=["xcat", "real_date"])

        self.pnls_df = pd.concat(
            [self.pnls_df, multi_asset_pnl], axis=0, ignore_index=True
        )
        self._multi_asset_xcats.append(combined_xcat)

    def plot_pnls(self, pnl_xcats: List[str] = None):
        """
        Plots the PnLs in
        """
        if self.pnls_df is None:
            raise ValueError("combine_pnls() must be run first.")
        if pnl_xcats is None:
            pnl_xcats = list(self.single_asset_pnls.keys()) + self._multi_asset_xcats

        pnl_df = self.pnls_df[self.pnls_df["xcat"].isin(pnl_xcats)]
        pnl_df["cumulative pnl"] = pnl_df.groupby("xcat")["value"].cumsum()

        plt.figure(figsize=(10, 6))
        sns.lineplot(data=pnl_df, x="real_date", y="cumulative pnl", hue=("xcat"))
        plt.title("Line Plot of DataFrame Columns")
        plt.xlabel("Index")
        plt.ylabel("% risk capital, no compounding")
        plt.legend(title="Assets")
        plt.show()

    def evaluate_pnls(self, pnl_xcats: List[str] = None) -> pd.DataFrame:
        """
        Evaluate the combined PnLs.
        """
        if pnl_xcats is None:
            pnl_xcats = self._multi_asset_xcats + list(self.single_asset_pnls.keys())
        pnl_evals = []
        for pnl_xcat in pnl_xcats:
            if pnl_xcat in self._multi_asset_xcats:
                eval_df = self._evaluate_combined_pnls(pnl_xcat)
            else:
                eval_df = self.single_asset_pnls[pnl_xcat].evaluate_pnls([pnl_xcat])
            pnl_evals.append(eval_df)

        return pd.concat(pnl_evals, axis=1, ignore_index=False, sort=False)

    def _evaluate_combined_pnls(self, pnl_xcat: str) -> pd.DataFrame:
        """
        Evaluate the combined PnLs.

        """
        stats = [
            "Return %",
            "St. Dev. %",
            "Sharpe Ratio",
            "Sortino Ratio",
            "Max 21-Day Draw %",
            "Max 6-Month Draw %",
            "Peak to Trough Draw %",
            "Top 5% Monthly PnL Share",
            "Traded Months",
        ]
        pnl_df = self.get_pnls([pnl_xcat])
        dfw = pnl_df.pivot(index="real_date", columns="xcat", values="value")
        df = pd.DataFrame(columns=dfw.columns, index=stats)

        df.iloc[0, :] = dfw.mean(axis=0) * 261
        df.iloc[1, :] = dfw.std(axis=0) * np.sqrt(261)
        df.iloc[2, :] = df.iloc[0, :] / df.iloc[1, :]
        dsd = dfw.apply(lambda x: np.sqrt(np.sum(x[x < 0] ** 2) / len(x))) * np.sqrt(
            261
        )
        df.iloc[3, :] = df.iloc[0, :] / dsd
        df.iloc[4, :] = dfw.rolling(21).sum().min()
        df.iloc[5, :] = dfw.rolling(6 * 21).sum().min()

        cum_pnl = dfw.cumsum()
        high_watermark = cum_pnl.cummax()
        drawdown = high_watermark - cum_pnl

        df.iloc[6, :] = -drawdown.max()

        monthly_pnl = dfw.resample("M").sum()
        total_pnl = monthly_pnl.sum(axis=0)
        top_5_percent_cutoff = int(np.ceil(len(monthly_pnl) * 0.05))
        top_months = pd.DataFrame(columns=monthly_pnl.columns)
        for column in monthly_pnl.columns:
            top_months[column] = (
                monthly_pnl[column]
                .nlargest(top_5_percent_cutoff)
                .reset_index(drop=True)
            )

        df.iloc[7, :] = top_months.sum() / total_pnl

        df.iloc[8, :] = dfw.resample("M").sum().count()

        return df

    def get_pnls(self, pnl_xcats: list[str] = None) -> pd.DataFrame:
        """
        Returns the combined PnLs.
        """
        if self.pnls_df is None:
            raise ValueError(
                "The PnLs have not been combined yet. combine_pnls() must be run first."
            )
        if pnl_xcats is None:
            return self.pnls_df

        return self.pnls_df[self.pnls_df["xcat"].isin(pnl_xcats)]

    def _validate_pnls(self):

        for name, pnl in self.single_asset_pnls.items():
            if not isinstance(pnl, NaivePnL):
                raise ValueError("All elements in the list must be NaivePnL objects.")
            if name not in pnl.pnl_names:
                raise ValueError("The pnl_xcat must be in the NaivePnL object.")
        return True

    def _normalize_weights(self, weights: dict) -> dict:
        """
        Normalize the weights to sum up to 1.
        """
        weights_sum = sum(weights.values())
        return {k: v / weights_sum for k, v in weights.items()}

    def get_pnl_xcats(self):
        return self.pnl_xcats


if __name__ == "__main__":

    np.random.seed(0)

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
        # bms=["EUR_EQXR_NSA", "USD_EQXR_NSA"],
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
        pnl_name="PNL_EQ",
    )

    pnl_eq.make_long_pnl(vol_scale=10, label="Long_EQ")

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
        pnl_name="PNL_FX",
    )

    pnl_fx.make_long_pnl(vol_scale=10, label="Long_FX")

    # pnl_fx.make_long_pnl(vol_scale=10, label="Long")
    print(pnl_eq.pnl_names)

    mapnl = MultiAssetPnL()
    mapnl.combine_pnls(
        pnls={"PNL_FX": pnl_fx, "PNL_EQ": pnl_eq},
        weights={"PNL_FX": 1, "PNL_EQ": 1},
        combined_xcat="FX_EQ",
    )
    # mapnl.combine_pnls(
    #     pnls={"Long_FX": pnl_fx, "Long_EQ": pnl_eq},
    #     weights={"Long_FX": 1, "Long_EQ": 1},
    #     combined_xcat="Long_FX_EQ",
    # )
    mapnl.plot_pnls()
    # print(mapnl.evaluate_pnls(['FX_EQ', 'Long_FX_EQ']))

    # multiasset_analysis = mapnl.combine_pnls(pnls={pnl_fx: "LONG_PNL", pnl_eq: "LONG_PNL"}, weights={pnl_fx: 1, pnl_eq: 1}, combined_name="LONG_FX_EQ")

    # # # multiasset_analysis["pnls"].cumsum().plot()
    # # plt.show()
    # # pass

    # print(pnl_fx.df)
