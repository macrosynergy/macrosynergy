from typing import Dict, List, Optional, Tuple
import warnings

from matplotlib import cm, pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import seaborn as sns

from macrosynergy.management.utils.df_utils import reduce_df, update_df
from macrosynergy.panel import linear_composite, make_zn_scores


class ScoreVisualisers:
    def __init__(
        self,
        df: pd.DataFrame,
        cids: List[str] = None,
        xcats: List[str] = None,
        xcat_labels: Dict[str, str] = None,
        xcat_comp: str = "Composite",
        weights: Dict[str, str] = None,
        blacklist: Dict[str, str] = None,
        sequential: bool = True,
        iis: bool = True,
        neutral: str = "mean",
        pan_weight: float = 1,
        thresh: float = 3,
        min_obs: int = 261,
        est_freq: str = "m",
        postfix: str = "_ZN",
        normalize_weights: bool = True,
        signs: Optional[List[float]] = None,
        complete_xcats: bool = False,
    ):
        self._validate_params(cids, xcats, xcat_labels, xcat_comp)

        self.cids = cids if cids else list(df["cid"].unique())
        self.xcat_labels = xcat_labels
        self.xcat_comp = xcat_comp + postfix
        self.weights = weights
        self.postfix = postfix

        self.df = self._create_df(
            df, xcats, blacklist, sequential, iis, neutral, pan_weight, thresh, min_obs, est_freq, postfix
        )
        self.old_xcats = [xcat_comp] + xcats
        self.xcats = self.df["xcat"].unique().tolist()

        composite_df = linear_composite(
            self.df,
            xcats=self.xcats,
            cids=self.cids,
            weights=self.weights,
            normalize_weights=normalize_weights,
            signs=signs,
            blacklist=blacklist,
            complete_xcats=complete_xcats,
            new_xcat=self.xcat_comp,
        )

        self.df = update_df(self.df, composite_df)
        self.xcats = self.df["xcat"].unique().tolist()

    def _validate_params(self, cids, xcats, xcat_labels, xcat_comp):
        if cids and (not isinstance(cids, list) or not all(isinstance(cid, str) for cid in cids)):
            raise TypeError("cids must be a list of strings")

        if xcats and (not isinstance(xcats, list) or not all(isinstance(xcat, str) for xcat in xcats)):
            raise TypeError("xcats must be a list of strings")

        if xcat_labels:
            if not isinstance(xcat_labels, dict):
                raise TypeError("xcat_labels must be a dictionary of strings")
            if len(xcat_labels) != len(xcats):
                raise ValueError("xcat_labels must be the same length as xcats")

        if not isinstance(xcat_comp, str):
            raise TypeError("xcat_comp must be a string")

    def _create_df(
        self, df, xcats, blacklist, sequential, iis, neutral, pan_weight, thresh, min_obs, est_freq, postfix
    ):
        result_df = None
        for xcat in xcats:
            dfzm = make_zn_scores(
                df, xcat=xcat, sequential=sequential, cids=self.cids,
                blacklist=blacklist, iis=iis, neutral=neutral, pan_weight=pan_weight,
                thresh=thresh, min_obs=min_obs, est_freq=est_freq, postfix=postfix
            )
            result_df = update_df(result_df, dfzm) if result_df is not None else dfzm
        return result_df

    def _plot_heatmap(
        self,
        df: pd.DataFrame,
        title: str,
        title_fontsize: int = 20,
        annot: bool = True,
        xticks=None,
        figsize=(20, 10),
        round_decimals: int = 2,
        cmap: str = None,
        cmap_range: Tuple[float, float] = None,
        horizontal_divider: bool = False,
        vertical_divider: bool = False,
    ):
        fig, ax = plt.subplots(figsize=figsize)

        cmap = cmap or "coolwarm_r"
        vmax = np.nanmax(np.abs(df.values))
        vmin = -vmax
        cmap_range = cmap_range or (vmin, vmax)

        sns.heatmap(
            df,
            cmap=cmap,
            annot=annot,
            xticklabels="auto",
            yticklabels="auto",
            fmt=f".{round_decimals}f",
            ax=ax,
            vmin=cmap_range[0],
            vmax=cmap_range[1],
        )

        ax.set_title(title, fontsize=title_fontsize)

        if horizontal_divider:
            ax.hlines([1], *ax.get_xlim(), linewidth=2, color="black")
        if vertical_divider:
            ax.vlines([1], *ax.get_ylim(), linewidth=2, color="black")

        plt.xticks(**(xticks or {"rotation": 45, "ha": "right"}))
        plt.tight_layout()
        plt.show()

    def _apply_postfix(self, items: List[str]) -> List[str]:
        return [item + self.postfix if not item.endswith(self.postfix) else item for item in items]

    def _strip_postfix(self, items: List[str]) -> List[str]:
        return [item[:-len(self.postfix)] if item.endswith(self.postfix) else item for item in items]

    def view_snapshot(
        self,
        cids: List[str] = None,
        xcats: List[str] = None,
        transpose: bool = False,
        date: str = None,
        annot: bool = True,
        title: str = None,
        title_fontsize: int = 20,
        figsize: tuple = (20, 10),
        xcat_labels: Dict[str, str] = None,
        xticks: dict = None,
        round_decimals: int = 2,
        cmap: str = None,
        cmap_range: Tuple[float, float] = None,
    ):
        cids = cids or self.cids
        xcats = xcats or self.xcats
        xcats = self._apply_postfix(xcats)

        date = pd.to_datetime(date) if date else self.df["real_date"].max() - pd.tseries.offsets.BDay(1)

        df = self.df[(self.df["xcat"].isin(xcats)) & (self.df["cid"].isin(cids)) & (self.df["real_date"] == date)]
        dfw = df.pivot(index="cid", columns="xcat", values="value")

        composite_zscore = self.xcat_comp
        if composite_zscore in xcats:
            dfw = dfw[[composite_zscore] + [xcat for xcat in dfw.columns if xcat != composite_zscore]]

        if xcat_labels:
            if set(self._apply_postfix(list(xcat_labels.keys()))) == set(dfw.columns):
                dfw.columns = [xcat_labels.get(self._strip_postfix([xcat])[0], xcat_labels.get(self._apply_postfix([xcat])[0], xcat)) for xcat in dfw.columns]

        if transpose:
            dfw = dfw.transpose()

        title = title or f"Snapshot for {date.strftime('%Y-%m-%d')}"

        horizontal_divider = transpose and composite_zscore in xcats
        vertical_divider = not transpose and composite_zscore in xcats

        self._plot_heatmap(
            dfw, title=title, annot=annot, xticks=xticks, figsize=figsize,
            title_fontsize=title_fontsize, round_decimals=round_decimals,
            cmap=cmap, cmap_range=cmap_range, horizontal_divider=horizontal_divider,
            vertical_divider=vertical_divider
        )

    def view_score_evolution(
        self,
        cids: List[str],
        xcat: str,
        freq: str,
        include_latest_period: bool = True,
        include_latest_day: bool = True,
        start: str = None,
        transpose: bool = False,
        annot: bool = True,
        title: str = None,
        title_fontsize: int = 20,
        xticks: dict = None,
        figsize: tuple = (20, 10),
        cmap: str = None,
        cmap_range: Tuple[float, float] = None,
        round_decimals: int = 2,
    ):
        cids = cids or self.cids
        xcat = xcat if xcat.endswith(self.postfix) else xcat + self.postfix

        if freq not in ["Q", "A", "BA"]:
            raise ValueError("freq must be 'Q', 'A', or 'BA'")

        freq = "2AS" if freq == "BA" else freq
        df = self.df[self.df["xcat"] == xcat]
        df = df[df["cid"].isin(cids)]
        df = df if start is None else df[df["real_date"] >= start]

        dfw = df.pivot(index="real_date", columns="cid", values="value")
        dfw_resampled = dfw.resample(freq, origin="start").mean()

        if not include_latest_period:
            dfw_resampled = dfw_resampled.iloc[:-1]

        if include_latest_day:
            latest_day = dfw.ffill().iloc[-1]
            dfw_resampled.loc[df["real_date"].max()] = latest_day
            if freq == "Q":
                dfw_resampled.index = list(dfw_resampled.index.to_period("Q").strftime("%YQ%q")[:-1]) + ["Latest"]
            else:
                dfw_resampled.index = list(dfw_resampled.index.strftime("%Y")[:-1]) + ["Latest"]
        else:
            if freq == "Q":
                dfw_resampled.index = list(dfw_resampled.index.to_period("Q").strftime("%YQ%q"))
            else:
                dfw_resampled.index = list(dfw_resampled.index.strftime("%Y"))

        dfw_resampled = dfw_resampled.transpose()

        if transpose:
            dfw_resampled = dfw_resampled.transpose()

        title = title or f"Evolution for {xcat}"

        self._plot_heatmap(
            dfw_resampled, title=title, annot=annot, xticks=xticks,
            figsize=figsize, title_fontsize=title_fontsize,
            round_decimals=round_decimals, cmap=cmap, cmap_range=cmap_range
        )

    def view_cid_evolution(
        self,
        cid: str,
        xcats: List[str],
        freq: str,
        include_latest_period: bool = True,
        include_latest_day: bool = True,
        start: str = None,
        transpose: bool = False,
        annot: bool = True,
        title: str = None,
        title_fontsize: int = 20,
        figsize: tuple = (20, 10),
        xticks: dict = None,
        xcat_labels: Dict[str, str] = None,
        cmap: str = None,
        cmap_range: Tuple[float, float] = None,
        round_decimals: int = 2,
    ):
        if not isinstance(cid, str):
            raise TypeError("cid must be a string")

        if freq not in ["Q", "A", "BA"]:
            raise ValueError("freq must be 'Q', 'A', or 'BA'")

        freq = "2AS" if freq == "BA" else freq

        xcats = self._apply_postfix(xcats)

        df = self.df[self.df["cid"] == cid]
        df = df if start is None else df[df["real_date"] >= start]
        df = df[df["xcat"].isin(xcats)]

        dfw = df.pivot(index="real_date", columns="xcat", values="value")
        dfw_resampled = dfw.resample(freq).mean()

        if not include_latest_period:
            dfw_resampled = dfw_resampled.iloc[:-1]

        if include_latest_day:
            latest_day = dfw.ffill().iloc[-1]
            dfw_resampled.loc[df["real_date"].max()] = latest_day
            if freq == "Q":
                dfw_resampled.index = list(dfw_resampled.index.to_period("Q").strftime("%YQ%q")[:-1]) + ["Latest"]
            else:
                dfw_resampled.index = list(dfw_resampled.index.strftime("%Y")[:-1]) + ["Latest"]
        else:
            if freq == "Q":
                dfw_resampled.index = list(dfw_resampled.index.to_period("Q").strftime("%YQ%q"))
            else:
                dfw_resampled.index = list(dfw_resampled.index.strftime("%Y"))

        if xcat_labels:
            if set(self._apply_postfix(list(xcat_labels.keys()))) == set(dfw_resampled.columns):
                dfw_resampled.columns = [xcat_labels.get(self._strip_postfix([xcat])[0], xcat_labels.get(self._apply_postfix([xcat])[0], xcat)) for xcat in dfw_resampled.columns]

        dfw_resampled = dfw_resampled.transpose()

        if transpose:
            dfw_resampled = dfw_resampled.transpose()

        title = title or f"Evolution for {cid}"

        horizontal_divider = not transpose and self.xcat_comp in xcats
        vertical_divider = transpose and self.xcat_comp in xcats

        self._plot_heatmap(
            dfw_resampled, title=title, annot=annot, xticks=xticks,
            figsize=figsize, title_fontsize=title_fontsize,
            round_decimals=round_decimals, cmap=cmap, cmap_range=cmap_range,
            horizontal_divider=horizontal_divider, vertical_divider=vertical_divider
        )

    def view_3d_surface(self, xcat: str, cids: List[str] = None):
        if cids is None:
            cids = self.cids

        df = self.df[
            (self.df["xcat"] == xcat + self.postfix) & (self.df["cid"].isin(cids))
        ]
        df["cid_num"] = df["cid"].astype("category").cat.codes

        # Pivot the DataFrame to create a 2D matrix for Z values
        pivot_table = df.pivot(index="cid_num", columns="real_date", values="value")

        X = mdates.date2num(pivot_table.columns)
        Y = pivot_table.index
        X, Y = np.meshgrid(X, Y)
        Z = pivot_table.values

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection="3d")
        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm_r)

        ax.set_xlabel("Date")
        ax.set_zlabel("Value")

        cid_labels = df[["cid", "cid_num"]].drop_duplicates().sort_values("cid_num")
        ax.set_yticks(cid_labels["cid_num"])
        ax.set_yticklabels(cid_labels["cid"], rotation=90, ha="right")

        ax.xaxis_date()
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        fig.autofmt_xdate()

        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

        plt.show()


if __name__ == "__main__":
    cids_dmea = ["FRF", "DEM", "ITL", "ESP", "EUR"]
    cids_dmxe = ["CHF", "GBP", "JPY", "SEK", "USD"]
    cids_dm = cids_dmea + cids_dmxe
    cids_g10 = ["AUD", "DEM", "FRF", "ESP", "ITL", "JPY", "NZD", "GBP", "USD"]
    cids_latm = ["BRL", "CLP", "COP", "MXN", "PEN"]  # Latam sovereigns
    cids_emea = [
        "CZK",
        "HUF",
        "ILS",
        "PLN",
        "RON",
        "ZAR",
        "TRY",
    ]  # EMEA sovereigns
    cids_emas = [
        "CNY",
        "IDR",
        "KRW",
        "MYR",
        "PHP",
        "THB",
    ]  # EM Asia sovereigns
    cids_ea = ["DEM", "FRF", "ESP", "ITL"]  # major Euro currencies before EUR
    cids_em = cids_emea + cids_latm + cids_emas
    cids = cids_dm + cids_em
    main = [
        "GGIEDGDP_NSA",
        # Currency reserve expansion as % of GDP
        "NIIPGDP_NSA",
        # Monetary base expansion as % of GDP
        "CABGDPRATIO_NSA_12MMA",
        # Intervention-driven liquidity expansion as % of GDP, diff over 3 months
        "GGOBGDPRATIO_NSA",
        # Intervention-driven liquidity expansion as % of GDP, diff over 6 months
    ]

    rets = []

    # rets = [
    #     "DU05YXR_NSA",
    #     "DU05YXR_VT10",
    #     "EQXR_NSA",
    #     "EQXR_VT10",
    #     "FXTARGETED_NSA",
    #     "FXUNTRADABLE_NSA",
    #     "FXXR_VT10",
    # ]

    xcats = main + rets

    tickers = [cid + "_" + xcat for cid in cids for xcat in xcats]

    start_date = "1990-01-01"
    end_date = "2023-07-01"

    import os
    from macrosynergy.download import JPMaQSDownload

    # Retrieve credentials
    client_id: str = os.getenv("DQ_CLIENT_ID")
    client_secret: str = os.getenv("DQ_CLIENT_SECRET")

    with JPMaQSDownload(client_id=client_id, client_secret=client_secret) as dq:
        df = dq.download(
            tickers=tickers,
            start_date=start_date,
            suppress_warning=True,
            metrics=["all"],
            show_progress=True,
        )

    sv = ScoreVisualisers(df, cids=cids, xcats=xcats)

    sv.view_snapshot(
        cids=cids,
        transpose=True,
        figsize=(14, 12),
    )
    # sv.view_snapshot(cids=cids, xcats=xcats, transpose=True)
    sv.view_cid_evolution(
        cid="USD",
        xcats=xcats + ["Composite"],
        xcat_labels={"GGIEDGDP_NSA_ZN": "Currency reserve expansion as % of GDP", "Composite_ZN": "Composite", "NIIPGDP_NSA_ZN": "Monetary base expansion as % of GDP", "CABGDPRATIO_NSA_12MMA_ZN": "Intervention-driven liquidity expansion as % of GDP, diff over 3 months", "GGOBGDPRATIO_NSA_ZN": "Intervention-driven liquidity expansion as % of GDP, diff over 6 months"},
        freq="A",
        transpose=False,
    )
    # sv.view_cid_evolution(cid="USD", xcats=xcats, freq="A", transpose=True)
    sv.view_score_evolution(
        xcat="GGIEDGDP_NSA",
        cids=cids,
        freq="BA",
        transpose=False,
        start="2010-01-01",
        title="AHKSJDA",
    )
    # sv.view_score_evolution(xcat="CRESFXGDP_NSA_D1M1ML6", cids=cids, freq="A", transpose=True, start="2010-01-01")

    sv.view_3d_surface("GGIEDGDP_NSA")
