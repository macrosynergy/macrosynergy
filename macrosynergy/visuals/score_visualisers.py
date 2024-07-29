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
    """
    Class to visualize the scores and linear composite of specified categories and 
    cross-sections.
    
    Parameters
    :param <pd.DataFrame> df: A DataFrame with the following columns:
        'cid', 'xcat', 'real_date', and at least one metric from -
        'value', 'grading', 'eop_lag', or 'mop_lag'.
    :param <List[str]> cids: A list of cids to select from the DataFrame.
        If None, all cids are selected.
    :param <List[str]> xcats: A list of xcats to select from the DataFrame.
        If None, all xcats are selected.
    :param <Dict[str, str]> xcat_labels: A dictionary mapping xcats to their labels.
    :param <str> xcat_comp: The name of the composite xcat.
    :param <List[float]> weights: A list of weights for the linear composite.
    :param <Dict[str, str]> blacklist: A dictionary mapping xcats to their blacklists.
    :param <bool> sequential: If True, calculate ZN scores sequentially.
    :param <bool> iis: If True, calculate ZN scores for individual indicators.
    :param <str> neutral: The method to calculate the neutral score.
    :param <float> pan_weight: The weight for the panel score.
    :param <float> thresh: The threshold for the ZN score.
    :param <int> min_obs: The minimum number of observations required to calculate ZN 
        scores.
    :param <str> est_freq: The frequency of the ZN score.
    :param <str> postfix: The postfix to append to the xcats.
    :param <bool> normalize_weights: If True, normalize the weights.
    :param <List[float]> signs: A list of signs for the linear composite.
    :param <bool> complete_xcats: If True, all xcats must have data for the composite to 
        have data.
    :param <bool> no_zn_scores: If True, do not calculate ZN scores.
    :param <bool> rescore_composite: If True, rescore the composite ZN scores.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        cids: List[str] = None,
        xcats: List[str] = None,
        xcat_labels: Dict[str, str] = None,
        xcat_comp: str = "Composite",
        weights: List[float] = None,
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
        no_zn_scores: bool = False,
        rescore_composite: bool = False,
    ):
        self._validate_params(cids, xcats, xcat_comp)

        self.cids = cids if cids else list(df["cid"].unique())
        self.weights = weights
        self.postfix = postfix
        if no_zn_scores:
            self.postfix = ""
        self.xcat_comp = xcat_comp + self.postfix

        self.df = self._create_df(
            df,
            xcats,
            blacklist,
            sequential,
            iis,
            neutral,
            pan_weight,
            thresh,
            min_obs,
            est_freq,
            postfix,
            no_zn_scores,
        )
        self.old_xcats = [xcat_comp] + xcats
        self.xcats = self._apply_postfix(xcats)
        for xcat in self.xcats:
            if xcat not in self.df["xcat"].unique():
                self.xcats.remove(xcat)
                warnings.warn(f"{xcat} not in the DataFrame")


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

        if rescore_composite:
            composite_df = make_zn_scores(
                composite_df,
                xcat=self.xcat_comp,
                sequential=sequential,
                cids=self.cids,
                blacklist=blacklist,
                iis=iis,
                neutral=neutral,
                pan_weight=pan_weight,
                thresh=thresh,
                min_obs=min_obs,
                est_freq=est_freq,
                postfix="",
            )

        self.df = update_df(self.df, composite_df)
        self.xcats = [self.xcat_comp] + self.xcats
        self.xcat_labels = xcat_labels

    def _validate_params(self, cids, xcats, xcat_comp):
        if cids and (
            not isinstance(cids, list) or not all(isinstance(cid, str) for cid in cids)
        ):
            raise TypeError("cids must be a list of strings")

        if xcats and (
            not isinstance(xcats, list)
            or not all(isinstance(xcat, str) for xcat in xcats)
        ):
            raise TypeError("xcats must be a list of strings")

        if not isinstance(xcat_comp, str):
            raise TypeError("xcat_comp must be a string")

    def _create_df(
        self,
        df,
        xcats,
        blacklist,
        sequential,
        iis,
        neutral,
        pan_weight,
        thresh,
        min_obs,
        est_freq,
        postfix,
        no_zn_scores
    ):
        if no_zn_scores:
            return reduce_df(df, xcats=xcats, cids=self.cids)
        
        result_df = None
        for xcat in xcats:
            dfzm = make_zn_scores(
                df,
                xcat=xcat,
                sequential=sequential,
                cids=self.cids,
                blacklist=blacklist,
                iis=iis,
                neutral=neutral,
                pan_weight=pan_weight,
                thresh=thresh,
                min_obs=min_obs,
                est_freq=est_freq,
                postfix=postfix,
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
        return [
            item + self.postfix if not item.endswith(self.postfix) else item
            for item in items
        ]

    def _strip_postfix(self, items: List[str]) -> List[str]:
        return [
            item[: -len(self.postfix)] if item.endswith(self.postfix) else item
            for item in items
        ]

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
        """
        View a snapshot of the scores for the specified cids and xcats at the specified
        date.

        Parameters
        :param <List[str]> cids: A list of cids to select from the DataFrame.
            If None, all cids are selected.
        :param <List[str]> xcats: A list of xcats to select from the DataFrame.
            If None, all xcats are selected.
        :param <bool> transpose: If True, transpose the snapshot so cids are on the x-axis
            and xcats are on the y-axis.
        :param <str> date: ISO-8601 formatted date. The date of the snapshot. If None, the
            latest date in the DataFrame is selected.
        :param <bool> annot: If True, annotate the heatmap.
        :param <str> title: The title of the heatmap.
        :param <int> title_fontsize: The fontsize of the title.
        :param <tuple> figsize: The size of the figure.
        :param <dict> xcat_labels: A dictionary mapping xcats to their labels.
        :param <dict> xticks: A dictionary of arguments to label the x axis.
        :param <int> round_decimals: The number of decimals to round the scores to.
        :param <str> cmap: The colormap of the heatmap.
        :param <tuple> cmap_range: The range of the colormap.
        """
        cids = cids or self.cids
        xcats = xcats or self.xcats
        xcats = self._apply_postfix(xcats)
        xcat_labels = xcat_labels or self.xcat_labels

        if date:
            date = pd.to_datetime(date)
        else:
            if self.df["real_date"].max() == pd.Timestamp.today():
                date = pd.Timestamp.today() - pd.tseries.offsets.BDay(1)
            else:
                date = self.df["real_date"].max()

        df = self.df[
            (self.df["xcat"].isin(xcats))
            & (self.df["cid"].isin(cids))
            & (self.df["real_date"] == date)
        ]
        dfw = df.pivot(index="cid", columns="xcat", values="value")
        dfw = dfw.reindex(cids)
        dfw.columns.name = None
        dfw.index.name = None

        composite_zscore = self.xcat_comp
        if composite_zscore in xcats:
            dfw = dfw[
                [composite_zscore]
                + [xcat for xcat in xcats if xcat != composite_zscore]
            ]
        else:
            dfw = dfw[xcats]

        if xcat_labels:
            if set(self._apply_postfix(list(xcat_labels.keys()))) >= set(dfw.columns):
                dfw.columns = [
                    xcat_labels.get(
                        self._strip_postfix([xcat])[0],
                        xcat_labels.get(self._apply_postfix([xcat])[0], xcat),
                    )
                    for xcat in dfw.columns
                ]

        if transpose:
            dfw = dfw.transpose()

        title = title or f"Snapshot for {date.strftime('%Y-%m-%d')}"

        horizontal_divider = transpose and composite_zscore in xcats
        vertical_divider = not transpose and composite_zscore in xcats

        self._plot_heatmap(
            dfw,
            title=title,
            annot=annot,
            xticks=xticks,
            figsize=figsize,
            title_fontsize=title_fontsize,
            round_decimals=round_decimals,
            cmap=cmap,
            cmap_range=cmap_range,
            horizontal_divider=horizontal_divider,
            vertical_divider=vertical_divider,
        )

    def view_score_evolution(
        self,
        xcat: str,
        freq: str = 'A',
        cids: List[str] = None,
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
        """
        View the evolution of the scores for the specified xcat and cids.

        Parameters
        :param <str> xcat: The category to view the evolution of.
        :param <str> freq: The frequency of the evolution. Must be 'Q', 'A', or 'BA'.
        :param <List[str]> cids: A list of cids to select from the DataFrame.
            If None, all cids are selected.
        :param <bool> include_latest_period: If True, include the latest period in the
            evolution.
        :param <bool> include_latest_day: If True, include the latest day in the 
            evolution.
        :param <str> start: ISO-8601 formatted date. Select data from this date onwards.
            If None, all dates are selected.
        :param <bool> transpose: If True, transpose the evolution so cids are on the 
            x-axis and dates are on the y-axis.
        :param <bool> annot: If True, annotate the heatmap.
        :param <str> title: The title of the heatmap.
        :param <int> title_fontsize: The fontsize of the title.
        :param <dict> xticks: A dictionary of arguments to label the x axis.
        :param <tuple> figsize: The size of the figure.
        :param <int> round_decimals: The number of decimals to round the scores to.
        :param <str> cmap: The colormap of the heatmap.
        :param <tuple> cmap_range: The range of the colormap.
        """
        cids = cids or self.cids
        xcat = xcat if xcat.endswith(self.postfix) else xcat + self.postfix

        if freq not in ["Q", "A", "BA"]:
            raise ValueError("freq must be 'Q', 'A', or 'BA'")

        freq = "2AS" if freq == "BA" else freq
        df = self.df[self.df["xcat"] == xcat]
        df = df[df["cid"].isin(cids)]
        df = df if start is None else df[df["real_date"] >= start]

        dfw = df.pivot(index="real_date", columns="cid", values="value")
        dfw.columns.name = None
        dfw.index.name = None
        dfw_resampled = dfw.resample(freq, origin="start").mean()

        if not include_latest_period:
            dfw_resampled = dfw_resampled.iloc[:-1]

        if include_latest_day:
            latest_day = dfw.ffill().iloc[-1]
            if self.df["real_date"].max() == pd.Timestamp.today():
                dfw_resampled.loc[self.df["real_date"].max() - pd.tseries.offsets.BDay(1)] = latest_day
            else:
                dfw_resampled.loc[self.df["real_date"].max()] = latest_day
            if freq == "Q":
                dfw_resampled.index = list(
                    dfw_resampled.index.to_period("Q").strftime("%YQ%q")[:-1]
                ) + ["Latest"]
            else:
                dfw_resampled.index = list(dfw_resampled.index.strftime("%Y")[:-1]) + [
                    "Latest"
                ]
        else:
            if freq == "Q":
                dfw_resampled.index = list(
                    dfw_resampled.index.to_period("Q").strftime("%YQ%q")
                )
            else:
                dfw_resampled.index = list(dfw_resampled.index.strftime("%Y"))

        dfw_resampled = dfw_resampled.transpose()
        dfw_resampled = dfw_resampled.reindex(cids)

        if transpose:
            dfw_resampled = dfw_resampled.transpose()

        title = title or f"Evolution for {xcat}"

        self._plot_heatmap(
            dfw_resampled,
            title=title,
            annot=annot,
            xticks=xticks,
            figsize=figsize,
            title_fontsize=title_fontsize,
            round_decimals=round_decimals,
            cmap=cmap,
            cmap_range=cmap_range,
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
        """
        View the evolution of the scores for the specified cid and xcats.

        Parameters
        :param <str> cid: The cross-section to view the evolution of.
        :param <List[str]> xcats: A list of xcats to select from the DataFrame.
            If None, all xcats are selected.
        :param <str> freq: The frequency of the evolution. Must be 'Q', 'A', or 'BA'.
        :param <bool> include_latest_period: If True, include the latest period in the
            evolution.
        :param <bool> include_latest_day: If True, include the latest day in the
            evolution.
        :param <str> start: ISO-8601 formatted date. Select data from this date onwards.
            If None, all dates are selected.
        :param <bool> transpose: If True, transpose the evolution so xcats are on the
            x-axis and dates are on the y-axis.
        :param <bool> annot: If True, annotate the heatmap.
        :param <str> title: The title of the heatmap.
        :param <int> title_fontsize: The fontsize of the title.
        :param <dict> xticks: A dictionary of arguments to label the x axis.
        :param <tuple> figsize: The size of the figure.
        :param <dict> xcat_labels: A dictionary mapping xcats to their labels.
        :param <int> round_decimals: The number of decimals to round the scores to.
        :param <str> cmap: The colormap of the heatmap.
        :param <tuple> cmap_range: The range of the colormap.
        """
        if not isinstance(cid, str):
            raise TypeError("cid must be a string")

        if freq not in ["Q", "A", "BA"]:
            raise ValueError("freq must be 'Q', 'A', or 'BA'")

        freq = "2AS" if freq == "BA" else freq

        xcat_labels = xcat_labels or self.xcat_labels
        xcats = self._apply_postfix(xcats)

        df = self.df[self.df["cid"] == cid]
        df = df if start is None else df[df["real_date"] >= start]
        df = df[df["xcat"].isin(xcats)]

        dfw = df.pivot(index="real_date", columns="xcat", values="value")
        dfw.columns.name = None
        dfw.index.name = None
        dfw_resampled = dfw.resample(freq).mean()

        if not include_latest_period:
            dfw_resampled = dfw_resampled.iloc[:-1]

        if include_latest_day:
            latest_day = dfw.ffill().iloc[-1]
            if self.df["real_date"].max() == pd.Timestamp.today():
                dfw_resampled.loc[self.df["real_date"].max() - pd.tseries.offsets.BDay(1)] = latest_day
            else:
                dfw_resampled.loc[self.df["real_date"].max()] = latest_day
            if freq == "Q":
                dfw_resampled.index = list(
                    dfw_resampled.index.to_period("Q").strftime("%YQ%q")[:-1]
                ) + ["Latest"]
            else:
                dfw_resampled.index = list(dfw_resampled.index.strftime("%Y")[:-1]) + [
                    "Latest"
                ]
        else:
            if freq == "Q":
                dfw_resampled.index = list(
                    dfw_resampled.index.to_period("Q").strftime("%YQ%q")
                )
            else:
                dfw_resampled.index = list(dfw_resampled.index.strftime("%Y"))

        composite_zscore = self.xcat_comp
        if composite_zscore in xcats:
            dfw_resampled = dfw_resampled[
                [composite_zscore]
                + [xcat for xcat in xcats if xcat != composite_zscore]
            ]
        else:
            dfw_resampled = dfw_resampled[xcats]

        if xcat_labels:
            if set(self._apply_postfix(list(xcat_labels.keys()))) >= set(
                dfw_resampled.columns
            ):
                dfw_resampled.columns = [
                    xcat_labels.get(
                        self._strip_postfix([xcat])[0],
                        xcat_labels.get(self._apply_postfix([xcat])[0], xcat),
                    )
                    for xcat in dfw_resampled.columns
                ]

        dfw_resampled = dfw_resampled.transpose()

        if transpose:
            dfw_resampled = dfw_resampled.transpose()

        title = title or f"Evolution for {cid}"

        horizontal_divider = not transpose and self.xcat_comp in xcats
        vertical_divider = transpose and self.xcat_comp in xcats

        self._plot_heatmap(
            dfw_resampled,
            title=title,
            annot=annot,
            xticks=xticks,
            figsize=figsize,
            title_fontsize=title_fontsize,
            round_decimals=round_decimals,
            cmap=cmap,
            cmap_range=cmap_range,
            horizontal_divider=horizontal_divider,
            vertical_divider=vertical_divider,
        )


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

    sv = ScoreVisualisers(df, cids=cids, xcats=xcats, xcat_labels={
            "GGIEDGDP_NSA_ZN": "Currency reserve expansion as % of GDP",
            "Composite_ZN": "Composite",
            "NIIPGDP_NSA_ZN": "Monetary base expansion as % of GDP",
            "CABGDPRATIO_NSA_12MMA_ZN": "Intervention-driven liquidity expansion as % of GDP, diff over 3 months",
            "GGOBGDPRATIO_NSA_ZN": "Intervention-driven liquidity expansion as % of GDP, diff over 6 months",
        }, rescore_composite=True, weights=[1, 1, 1, 10])

    sv.view_snapshot(
        cids=cids,
        xcats=xcats,
        transpose=True,
        figsize=(14, 12),
    )
    sv.view_cid_evolution(
        cid="USD",
        xcats=xcats + ["Composite"],
        freq="A",
        transpose=False,
    )
    sv.view_score_evolution(
        xcat="GGIEDGDP_NSA",
        cids=cids,
        freq="BA",
        transpose=False,
        start="2010-01-01",
        title="AHKSJDA",
    )
