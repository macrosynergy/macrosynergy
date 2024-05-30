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
        self.pnls_df = pd.DataFrame()
        self.single_asset_pnls = {}
        self.multi_asset_xcats = []
        self.assets = set()

    def add_pnl(self, pnl: NaivePnL, asset: str, pnl_xcats: List[str]):
        """
        Add a NaivePnL object.

        :param pnl: NaivePnL object.
        :param asset: Asset name.
        :param pnl_xcats: List of PnLs to add from the NaivePnL object.
        """
        self._validate_pnl(pnl, pnl_xcats, asset)
        if not isinstance(asset, str):
            raise ValueError("The asset must be a string.")
        self.assets.add(asset)

        pnl_df = pnl.pnl_df(pnl_xcats)
        pnl_df.loc[:, "xcat"] = f"{asset}::" + pnl_df["xcat"]
        self.pnls_df = pd.concat([self.pnls_df, pnl_df], axis=0, ignore_index=True)

        for xcat in pnl_df.xcat.unique():
            self.single_asset_pnls[xcat] = pnl

    def combine_pnls(
        self,
        pnl_xcats: List[str],
        weights: Optional[dict[str, float]] = None,
        multi_asset_xcat: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Combine PnLs with optional weighting.

        :param pnl_xcats:
            List of PnLs to combine. Must be in the format 'asset::xcat' and
            added using `add_pnl()`.
        :param weights:
            Weights for each PnL, by default None. Must be in the format
            {'asset::xcat': weight}.
        :param multi_asset_xcat:
            Name of the combined PnL, by default None.
        """
        self._check_pnls_added(min_pnls=2)
        for pnl_xcat in pnl_xcats:
            if "::" not in pnl_xcat:
                raise ValueError(f"{pnl_xcat} must be in the format 'asset::xcat'.")
            if pnl_xcat not in self.single_asset_pnls:
                raise ValueError(f"{pnl_xcat} has not been added with add_pnl() yet.")
        # Default multi_asset_xcat
        assets = [x.split("::")[0] for x in pnl_xcats]
        multi_asset_xcat = (
            f"{'_'.join(assets)}_PNL" if multi_asset_xcat is None else multi_asset_xcat
        )
        # Default weights
        if weights is None:
            weights = {pnl_name: 1 for pnl_name in pnl_xcats}
        weights = self._normalize_weights(weights)

        multiasset_df = []
        for pnl_xcat in pnl_xcats:
            single_asset_df = self.pnls_df[self.pnls_df["xcat"] == pnl_xcat].assign(
                asset=pnl_xcat
            )
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

        multiasset_rets.name = multi_asset_xcat

        multi_asset_pnl = multiasset_rets.reset_index().melt(
            id_vars=["real_date"], var_name="xcat", value_name="value"
        )
        multi_asset_pnl = multi_asset_pnl.sort_values(by=["xcat", "real_date"])
        multi_asset_pnl["cid"] = "ALL"
        self.pnls_df = pd.concat(
            [self.pnls_df, multi_asset_pnl], axis=0, ignore_index=True
        )
        self.multi_asset_xcats.append(multi_asset_xcat)

    def plot_pnls(self, pnl_xcats: List[str] = None, title: str = None):
        """
        Creates a plot of PnLs

        :param pnl_xcats: List of PnLs to plot. If None, all PnLs are plotted.
            Must be in the format 'asset::xcat', or 'xcat' for multi-asset PnLs.
        """
        self._check_pnls_added()

        if pnl_xcats is None:
            pnl_df = self.pnls_df
        else:
            pnl_df = self.pnls_df[self.pnls_df["xcat"].isin(pnl_xcats)].copy()
        pnl_df.loc[:, "cumulative pnl"] = pnl_df.groupby("xcat")["value"].cumsum()

        sns.lineplot(data=pnl_df, x="real_date", y="cumulative pnl", hue=("xcat"))
        plt.title(title)
        plt.xlabel(None)
        plt.ylabel("% risk capital, no compounding")
        plt.legend(title="PnLs")
        plt.show()

    def evaluate_pnls(self, pnl_xcats: List[str] = None) -> pd.DataFrame:
        """
        Evaluate single and multi-asset PnLs.

        :param pnl_xcats: List of PnLs to evaluate. If None, all PnLs are evaluated.
            Must be in the format 'asset::xcat', or 'xcat' for multi-asset PnLs.
        """
        self._check_pnls_added()
        if pnl_xcats is None:
            pnl_xcats = self.multi_asset_xcats + list(self.single_asset_pnls.keys())
        pnl_evals = []
        for pnl_xcat in pnl_xcats:
            if pnl_xcat in self.multi_asset_xcats:
                eval_df = self._evaluate_multi_asset_pnl(pnl_xcat)
            else:
                pnl = self.single_asset_pnls[pnl_xcat]
                eval_df = pnl.evaluate_pnls([pnl_xcat.split("::")[1]])
                eval_df.columns = [pnl_xcat]
            pnl_evals.append(eval_df)

        return pd.concat(pnl_evals, axis=1, ignore_index=False, sort=False)

    def _evaluate_multi_asset_pnl(self, pnl_xcat: str) -> pd.DataFrame:
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

    def _normalize_weights(self, weights: dict) -> dict:
        """
        Normalize the weights to sum up to 1.
        """
        weights_sum = sum(weights.values())
        return {k: v / weights_sum for k, v in weights.items()}

    def _validate_pnl(self, pnl: NaivePnL, pnl_xcats: List[str], asset: str):
        if not isinstance(pnl, NaivePnL):
            raise ValueError("The pnl must be a NaivePnL object.")
        if not set(pnl_xcats).issubset(pnl.pnl_names):
            raise ValueError("The pnl_xcats must be in the NaivePnL object.")
        if not all(isinstance(x, str) for x in pnl_xcats):
            raise ValueError("All elements in the list must be strings.")
        for xcat in pnl_xcats:
            asset_xcat = f"{asset}::{xcat}"
            if asset_xcat in self.single_asset_pnls:
                raise ValueError(f"{asset_xcat} has already been added.")
        return True

    def _validate_pnls(self):

        for name, pnl in self.single_asset_pnls.items():
            if not isinstance(pnl, NaivePnL):
                raise ValueError("All elements in the list must be NaivePnL objects.")
            if name not in pnl.pnl_names:
                raise ValueError("The pnl_xcat must be in the NaivePnL object.")
        return True

    def pnl_xcats(self):
        return self.multi_asset_xcats + list(self.single_asset_pnls.keys())

    def _check_pnls_added(self, min_pnls: int = 1):
        if len(self.pnls_df) < min_pnls:
            raise ValueError(f"At least {min_pnls} PnL must be added with add_pnl() first.")


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

    pnl_eq.make_long_pnl(vol_scale=10, label="LONG")

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

    pnl_fx.make_long_pnl(vol_scale=10, label="LONG")

    # pnl_fx.make_long_pnl(vol_scale=10, label="Long")
    print(pnl_eq.pnl_names)

    mapnl = MultiAssetPnL()

    mapnl.add_pnl(pnl_fx, "FX", ["PNL_FX", "LONG"])
    mapnl.add_pnl(pnl_eq, "EQ", ["PNL_EQ", "LONG"])

    mapnl.combine_pnls(
        ["FX::LONG", "EQ::LONG"],
        weights={"FX::LONG": 9, "EQ::LONG": 1},
        # multi_asset_xcat="EQ_FX_LONG",
    )

    mapnl.plot_pnls()
    print(mapnl.evaluate_pnls())

    # multiasset_analysis = mapnl.combine_pnls(pnls={pnl_fx: "LONG_PNL", pnl_eq: "LONG_PNL"}, weights={pnl_fx: 1, pnl_eq: 1}, combined_name="LONG_FX_EQ")

    # # # multiasset_analysis["pnls"].cumsum().plot()
    # plt.show()
    # # pass

    # print(pnl_fx.df)
