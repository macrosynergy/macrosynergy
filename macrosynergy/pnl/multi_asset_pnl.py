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


class MultiAssetPnL:

    def __init__(self, pnls: dict, pnl_xcats: dict = None):
        self.pnls = pnls
        self.pnl_xcats = pnl_xcats
        self._validate_pnls()

        self.multi_asset_pnl = None

    def combine_pnls(
        self, weights: dict = None, combined_name: str = None
    ) -> pd.DataFrame:
        """
        Combine the PnLs in the list with the given weights.
        """
        combined_name = (
            "_".join(self.pnls.keys()) if combined_name is None else combined_name
        )
        # default weights
        if weights is None:
            weights = {pnl_name: 1 for pnl_name in self.pnls.keys()}
        weights = self._normalize_weights(weights)

        multiasset_df = []
        for asset_name, asset_pnl in self.pnls.items():
            asset_pnl_xcat = self.pnl_xcats[asset_name]
            single_asset_df = asset_pnl.pnl_df([asset_pnl_xcat]).assign(
                asset=asset_name
            )
            multiasset_df.append(single_asset_df)

        multiasset_df = pd.concat(multiasset_df, axis=0, ignore_index=True)

        raw_pnls = multiasset_df.set_index(["real_date", "asset"])["value"].unstack()

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
        ).rename(columns={0: combined_name})

        self.multi_asset_pnl = multiasset_rets.reset_index().melt(
            id_vars=["real_date"], var_name="xcat", value_name="value"
        )
        self.multi_asset_pnl = self.multi_asset_pnl.sort_values(
            by=["xcat", "real_date"]
        )

        # return out

    def plot_pnls(self, weights: dict = None):
        """
        Plots the PnLs in
        """
        if self.multi_asset_pnl is None:
            self.combine_pnls(weights=weights)
        self.multi_asset_pnl["cumulative pnl"] = self.multi_asset_pnl.groupby("xcat")[
            "value"
        ].cumsum()

        plt.figure(figsize=(10, 6))
        sns.lineplot(
            data=self.multi_asset_pnl, x="real_date", y="cumulative pnl", hue=("xcat")
        )
        plt.title("Line Plot of DataFrame Columns")
        plt.xlabel("Index")
        plt.ylabel("% risk capital, no compounding")
        plt.legend(title="Assets")
        plt.show()

    def evaluate_pnls(self):
        """
        Evaluate the combined PnLs.
        """
        pnl_evals = []
        for name, pnl in self.pnls.items():
            pnl_xcat = self.pnl_xcats[name]
            eval_df = pnl.evaluate_pnls([pnl_xcat])
            # eval_df["asset"] = name
            pnl_evals.append(eval_df)

        return pd.concat(pnl_evals, axis=1, ignore_index=False)

    def _evaluate_combined_pnl(
        self,
        start: str = None,
        end: str = None,
    ):
        """
        Table of key PnL statistics.

        :param <List[str]> pnl_cats: list of PnL categories that should be plotted.
        :param <List[str]> pnl_cids: list of cross-sections to be plotted; default is
            'ALL' (global PnL).
            Note: one can only have multiple PnL categories or multiple cross-sections,
            not both.
        :param <str> start: earliest date in ISO format. Default is None and earliest
            date in df is used.
        :param <str> end: latest date in ISO format. Default is None and latest date
            in df is used.
        :param <dict[str, str]> label_dict: dictionary with keys as pnl_cats and values
            as new labels for the PnLs.

        :return <pd.DataFrame>: standardized DataFrame with key PnL performance
            statistics.
        """

        dfx = reduce_df(
            self.df, pnl_cats, pnl_cids, start, end, self.black, out_all=False
        )

        groups = "xcat" if len(pnl_cids) == 1 else "cid"
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

        # If benchmark tickers have been passed into the Class and if the tickers are
        # present in self.dfd.
        list_for_dfbm = []

        if self.bm_bool and bool(self._bm_dict):
            list_for_dfbm = list(self._bm_dict.keys())
            for bm in list_for_dfbm:
                stats.insert(len(stats) - 1, f"{bm} correl")

        dfw = dfx.pivot(index="real_date", columns=groups, values="value")
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

        if len(list_for_dfbm) > 0:
            bm_df = pd.concat(list(self._bm_dict.values()), axis=1)
            for i, bm in enumerate(list_for_dfbm):
                index = dfw.index.intersection(bm_df.index)
                correlation = dfw.loc[index].corrwith(
                    bm_df.loc[index].iloc[:, i], axis=0, method="pearson", drop=True
                )
                df.iloc[8 + i, :] = correlation

        df.iloc[8 + len(list_for_dfbm), :] = dfw.resample("M").sum().count()

        if label_dict is not None:
            if not isinstance(label_dict, dict):
                raise TypeError("label_dict must be a dictionary.")
            if not all([isinstance(k, str) for k in label_dict.keys()]):
                raise TypeError("Keys in label_dict must be strings.")
            if not all([isinstance(v, str) for v in label_dict.values()]):
                raise TypeError("Values in label_dict must be strings.")
            if len(label_dict) != len(df.columns):
                raise ValueError(
                    "label_dict must have the same number of keys as columns in the "
                    "DataFrame."
                )
            df.rename(index=label_dict, inplace=True)
            df = df[label_dict.values()]

        return df

    def get_pnls(self) -> pd.DataFrame:
        """
        Returns the combined PnLs.
        """
        if self.multi_asset_pnl is None:
            raise ValueError("The PnLs have not been combined yet.")
        return self.multi_asset_pnl

    def _validate_pnls(self):

        for name, pnl in self.pnls.items():
            if not isinstance(pnl, NaivePnL):
                raise ValueError("All elements in the list must be NaivePnL objects.")
            if name not in self.pnl_xcats:
                raise ValueError(
                    "The name of the NaivePnL object must be in the pnl_xcats dictionary."
                )

        if len(self.pnls) != len(self.pnl_xcats):
            raise ValueError("The number of PnLs and pnl_xcats must be the same.")
        return True

    def _normalize_weights(self, weights: dict) -> dict:
        """
        Normalize the weights to sum up to 1.
        """
        weights_sum = sum(weights.values())
        return {k: v / weights_sum for k, v in weights.items()}


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
        pnl_name="PNL_GROWTH",
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
    pnl_xcat = {"FX": "PNL_INFL_NEG", "EQ": "PNL_GROWTH"}
    mapnl = MultiAssetPnL(pnl_dict, pnl_xcats=pnl_xcat)
    multiasset_analysis = mapnl.combine_pnls(weights={"FX": 1, "EQ": 1})
    # mapnl.plot_pnls()
    # # multiasset_analysis["pnls"].cumsum().plot()
    # plt.show()
    # pass

    print(mapnl.evaluate_pnls())
