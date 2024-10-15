"""
Multi PnLs combine multiple "Naive" PnLs with limited signal options and disregarding transaction costs.
"""

from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from macrosynergy.management.simulate import make_qdf
from macrosynergy.management.utils import update_df, _map_to_business_day_frequency
from macrosynergy.pnl import NaivePnL


class MultiPnL:

    def __init__(self):
        self.pnls_df = pd.DataFrame(columns=["real_date", "xcat", "value", "cid"])
        self.single_return_pnls = {}
        self.composite_pnl_xcats = []
        self.xcat_to_ret = {}

    def add_pnl(self, pnl: NaivePnL, pnl_xcats: List[str]):
        """
        Add a NaivePnL object.

        :param <NaivePnL> pnl: NaivePnL object.
        :param <List[str]> pnl_xcats: List of PnLs to add from the NaivePnL object.
        """
        self._validate_pnl(pnl, pnl_xcats)

        pnl_df = pnl.pnl_df(pnl_xcats)
        pnl_df.loc[:, "xcat"] = pnl_df["xcat"] + "/" + pnl.ret
        # self.pnls_df = pd.concat([self.pnls_df, pnl_df], axis=0, ignore_index=True)
        self.pnls_df = update_df(self.pnls_df, pnl_df)
        for xcat in pnl_df.xcat.unique():
            self.single_return_pnls[xcat] = pnl

        for xcat in pnl_xcats:
            if xcat not in self.xcat_to_ret:
                self.xcat_to_ret[xcat] = {pnl.ret}
            else:
                self.xcat_to_ret[xcat].add(pnl.ret)
        pass

    def combine_pnls(
        self,
        pnl_xcats: List[str],
        composite_pnl_xcat: str,
        weights: Optional[Dict[str, float]] = None,
    ) -> pd.DataFrame:
        """
        Combine PnLs with optional weighting.

        :param <List[str]> pnl_xcats: List of PnLs to combine. Must be in the format
            'xcat/return' and added using `add_pnl()`.
        :param <str> composite_pnl_xcat: xcat for the combined PnL.
        :param <Optional[Dict[str, float]]> weights: Weights for each PnL, by default None.
            Must be in the format {'xcat': weight} or {'xcat/return': weight}.
        """
        self._check_pnls_added(min_pnls=2)
        for i, pnl_xcat in enumerate(pnl_xcats):
            pnl_xcats[i] = self._infer_return_by_xcat(pnl_xcat)

        # Default weights
        if weights is None:
            weights = {pnl_name: 1 for pnl_name in pnl_xcats}
        else:
            weights = {self._infer_return_by_xcat(k): v for k, v in weights.items()}
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
        mfreq = _map_to_business_day_frequency("M")
        weights_change = (
            (1 + raw_pnls / 100).groupby(pd.Grouper(freq=mfreq)).cumprod()
        )  # in decimals, not percentage, gross amount
        weights_change = (
            weights_change.groupby(pd.Grouper(freq=mfreq))
            .shift(periods=1)
            .fillna(value=1)
        )

        # Dynamic weights
        final_weights = start_weights * weights_change
        final_weights = final_weights.div(final_weights.sum(axis=1), axis=0)

        # final calculation
        multiasset_rets = (final_weights * raw_pnls).sum(axis=1)

        multiasset_rets.name = composite_pnl_xcat

        multi_asset_pnl = multiasset_rets.reset_index().melt(
            id_vars=["real_date"], var_name="xcat", value_name="value"
        )
        multi_asset_pnl = multi_asset_pnl.sort_values(by=["xcat", "real_date"])
        multi_asset_pnl["cid"] = "ALL"

        self.pnls_df = update_df(self.pnls_df, multi_asset_pnl).sort_values(
            by=["xcat", "real_date"]
        )
        self.composite_pnl_xcats.append(composite_pnl_xcat)

    def plot_pnls(
        self,
        pnl_xcats: List[str] = None,
        title: str = None,
        title_fontsize: int = 14,
        xcat_labels: Union[List[str], dict] = None,
    ):
        """
        Creates a plot of PnLs

        :param <List[str]> pnl_xcats: List of PnLs to plot. If None, all PnLs are plotted.
            Must be in the format 'xcat', or 'xcat/return_xcat'.
        :param <str> title: Title of the plot.
        """
        self._check_pnls_added()

        if pnl_xcats is None:
            pnl_df = self.pnls_df
            pnl_xcats = self.pnl_xcats()
        else:
            for i, pnl_xcat in enumerate(pnl_xcats):
                pnl_xcats[i] = self._infer_return_by_xcat(pnl_xcat)
            pnl_df = self.pnls_df[self.pnls_df["xcat"].isin(pnl_xcats)].copy()

        if xcat_labels is not None:

            xcat_labels = self._check_xcat_labels(pnl_xcats, xcat_labels)
            pnl_df["xcat"] = pnl_df["xcat"].map(xcat_labels)

        pnl_df.loc[:, "cumulative pnl"] = pnl_df.groupby("xcat")["value"].cumsum()

        sns.lineplot(data=pnl_df, x="real_date", y="cumulative pnl", hue=("xcat"))
        plt.title(title, fontsize=title_fontsize)
        plt.xlabel(None)
        plt.ylabel("% risk capital, no compounding")
        plt.legend(title="PnL Category(s)")
        plt.show()
        pnl_df.drop(columns="cumulative pnl", inplace=True)

    def evaluate_pnls(self, pnl_xcats: List[str] = None) -> pd.DataFrame:
        """
        Evaluate individual and composite PnLs.

        :param <List[str]> pnl_xcats: List of PnLs to evaluate. If None, all PnLs are evaluated.
            Must be in the format 'xcat', or 'xcat/return_xcat'.
        """
        self._check_pnls_added()
        if pnl_xcats is None:
            pnl_xcats = self.pnl_xcats()
        else:
            for i, pnl_xcat in enumerate(pnl_xcats):
                pnl_xcats[i] = self._infer_return_by_xcat(pnl_xcat)
        pnl_evals = []
        for pnl_xcat in pnl_xcats:
            if pnl_xcat in self.composite_pnl_xcats:
                eval_df = self._evaluate_composite_pnl(pnl_xcat)
            else:
                pnl = self.single_return_pnls[pnl_xcat]
                eval_df = pnl.evaluate_pnls([pnl_xcat.split("/")[0]])
                eval_df.columns = [pnl_xcat]
            pnl_evals.append(eval_df)

        return pd.concat(pnl_evals, axis=1, ignore_index=False, sort=False)

    def _evaluate_composite_pnl(self, pnl_xcat: str) -> pd.DataFrame:
        """
        Evaluate the combined PnLs in a manner similar to NaivePnL's `evaluate_pnls()`.

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

    def get_pnls(self, pnl_xcats: List[str] = None) -> pd.DataFrame:
        """
        Returns a DataFrame with PnLs.

        :param <List[str]> pnl_xcats: List of PnLs to return. If None, all PnLs are returned.
            Must be in the format 'xcat', or 'xcat/return_xcat'.
        """
        if self.pnls_df is None:
            raise ValueError("The PnLs have been added. Use add_pnl() first.")

        if pnl_xcats is None:
            return self.pnls_df

        else:
            for i, pnl_xcat in enumerate(pnl_xcats):
                pnl_xcats[i] = self._infer_return_by_xcat(pnl_xcat)

            return self.pnls_df[self.pnls_df["xcat"].isin(pnl_xcats)]

    def _normalize_weights(self, weights: dict) -> dict:
        """
        Normalize the weights to sum up to 1.
        """
        weights_sum = sum(weights.values())
        return {k: v / weights_sum for k, v in weights.items()}

    def _validate_pnl(self, pnl: NaivePnL, pnl_xcats: List[str]):
        """
        Validate the PnL and PnL categories.
        """
        if not isinstance(pnl, NaivePnL):
            raise ValueError("The pnl must be a NaivePnL object.")
        if not set(pnl_xcats).issubset(pnl.pnl_names):
            raise ValueError("The pnl_xcats must be in the NaivePnL object.")
        if not all(isinstance(x, str) for x in pnl_xcats):
            raise ValueError("All elements in the list must be strings.")
        return True

    def pnl_xcats(self):
        """
        Return all PnL categories.
        """
        return self.pnls_df["xcat"].unique().tolist()

    def return_xcats(self):
        """
        Return all return categories associated with PnLs.
        """
        return list(set(self.xcat_to_ret.values()))

    def _check_pnls_added(self, min_pnls: int = 1):
        """
        Check if at least `min_pnls` PnLs have been added.
        """
        if len(self.pnl_xcats()) < min_pnls:
            raise ValueError(
                f"At least {min_pnls} PnL must be added with add_pnl() first."
            )

    def _infer_return_by_xcat(self, pnl_xcat):
        """
        Infer the return category from the xcat if not provided.

        Throws an error is there are multiple return categories for the xcat.
        """
        if pnl_xcat in self.composite_pnl_xcats:
            return pnl_xcat

        if "/" not in pnl_xcat:
            if pnl_xcat not in self.xcat_to_ret:
                raise ValueError(f"{pnl_xcat} has not been added with add_pnl() yet.")
            if len(self.xcat_to_ret[pnl_xcat]) > 1:
                raise ValueError(
                    f"{pnl_xcat} corresponds to multiple return categories: {self.xcat_to_ret[pnl_xcat]}. "
                    "Must append return to xcat in the format 'xcat/return'."
                )
            else:
                return f"{pnl_xcat}/{list(self.xcat_to_ret[pnl_xcat])[0]}"
        else:
            if pnl_xcat not in self.pnl_xcats():
                raise ValueError(f"{pnl_xcat} has not been added with add_pnl() yet.")
            else:
                return pnl_xcat

    def _check_xcat_labels(self, pnl_xcats, xcat_labels):
        if isinstance(xcat_labels, dict):
            xcat_labels = {
                self._infer_return_by_xcat(k): v for k, v in xcat_labels.items()
            }
        elif isinstance(xcat_labels, list):
            if len(pnl_xcats) != len(xcat_labels):
                raise ValueError(
                    "If using a list, the number of labels must match the number of PnLs."
                )
            xcat_labels = dict(zip(pnl_xcats, xcat_labels))
        else:
            raise ValueError("xcat_labels must be a list or a dictionary.")
        return xcat_labels


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
    df_xcats.loc["GROWTH"] = ["2000-01-03", "2020-10-30", 1, 2, 0.9, 1]
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

    mapnl = MultiPnL()

    mapnl.add_pnl(pnl_fx, ["PNL_FX", "LONG"])
    mapnl.add_pnl(pnl_eq, ["PNL_EQ", "LONG"])

    mapnl.combine_pnls(
        # ["PNL_FX", "PNL_EQ"],
        ["PNL_EQ", "PNL_FX"],
        # weights={"PNL_FX": 1, "LONG": 1},
        composite_pnl_xcat="EQ_FX_LONG",
    )

    mapnl.plot_pnls(["PNL_FX", "PNL_EQ"], xcat_labels=["z", "FX"], title="PnLs")
    # print(mapnl.get_pnls(["PNL_FX"]))
    # print(mapnl.evaluate_pnls(["PNL_EQ"]))
