from typing import Dict, List, Optional

from matplotlib import cm, pyplot as plt
import matplotlib.dates as mdates

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay

import seaborn as sns

from macrosynergy.management.utils.df_utils import reduce_df, update_df
from macrosynergy.panel import linear_composite, make_zn_scores


class ScoreVisualisers(object):
    """
    Class for displaying heatmaps of normalized quantamental categories, including a weighted composite

    :param <pd.DataFrame> df: A DataFrame with the following columns:
        'cid', 'xcat', 'real_date', and at least one metric from -
        'value', 'grading', 'eop_lag', or 'mop_lag'.
    :param <List[str]> cids: A list of cids to select from the DataFrame.
        If None, all cids are selected.
    :param <List[str]> xcats: A list of xcats to select from the DataFrame.
        These should be normalized indicators.
    :param <Dict[str, str]> xcat_labels: Optional list of custom labels for xcats
        Default is None.
    :param <str> xcat_comp: Label for composite score, if one is to be calculated and used. Default is “Composite”/
        Calculations are done based for the linear_composite function
    :param <Dict[str, str]> weights: A list of weights as large as the xcats list for calculating the composite.
        Default is equal weights.
        [we need passthrough arguments for linear_composite and make_zn_scores]
    :param bool sequential: If True, the function will calculate the composite score sequentially.
    """

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
        thresh: float = None,
        min_obs: int = 261,
        est_freq: str = "m",
        postfix: str = "_ZN",
        normalize_weights: bool = True,
        signs: Optional[List[float]] = None,
        complete_xcats: bool = False,
    ):
        if cids is None:
            cids = list(df["cid"].unique())
        elif not isinstance(cids, list) or not all(
            isinstance(cid, str) for cid in cids
        ):
            raise TypeError("cids must be a list of strings")

        if xcats is None:
            xcats = list(df["xcat"].unique())
        elif not isinstance(xcats, list) or not all(
            isinstance(xcat, str) for xcat in xcats
        ):
            raise TypeError("xcats must be a list of strings")

        if xcat_labels is not None:
            if not isinstance(xcat_labels, list):
                raise TypeError("xcat_labels must be a dictionary of strings")
            if len(xcat_labels) != len(xcats):
                raise ValueError("xcat_labels must be the same length as xcats")

        if not isinstance(xcat_comp, str):
            raise TypeError("xcat_comp must be a string")

        self.cids = cids
        self.xcats = xcats
        self.xcat_labels = xcat_labels
        self.xcat_comp = xcat_comp + postfix
        self.weights = weights
        self.df = None

        for xcat in self.xcats:
            dfzm = make_zn_scores(
                df,
                xcat=xcat,
                sequential=sequential,
                cids=cids,
                blacklist=blacklist,
                iis=iis,
                neutral=neutral,
                pan_weight=pan_weight,
                thresh=thresh,
                min_obs=min_obs,
                est_freq=est_freq,
                postfix=postfix,
            )
            if self.df is None:
                self.df = dfzm
            else:
                self.df = update_df(self.df, dfzm)

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
        self.postfix = postfix

    def _plot_heatmap(self, df: pd.DataFrame, title: str, annot: bool = True, xticks=None, figsize=(20, 10)):
        fig, ax = plt.subplots(figsize=figsize)
        
        sns.heatmap(
            df,
            cmap="coolwarm",
            annot=annot,
            xticklabels="auto",
            yticklabels="auto",
            ax=ax,
        )

        ax.set_title(title, fontsize=14)

        if xticks is None:
            xticks = {"rotation": 45, "ha": "right"}
        plt.xticks(**xticks)

        plt.tight_layout()
        plt.show()

    def view_snapshot(
        self,
        cids: List[str] = None,
        xcats: List[str] = None,
        transpose: bool = False,
        date: str = None,
        annot: bool = True,
        title: str = None,
        xticks: dict = None,
        figsize: tuple = (20, 10)
    ):
        """
        Display a multiple scores for multiple countries for the latest available or any previous date

        :param <List[str]> cids: A list of cids whose values are displayed. Default is all in the class
        :param <List[str]> xcats: A list of xcats to be displayed in the given order. Default is all in the class, including the composite, with the latter being the first row (or column).
        :param <bool> transpose: If False (default) rows are cids and columns are xcats. If True rows are xcats and columns are cids.
        :param <str> date: ISO-8601 formatted date string giving the date (or nearest previous if not available). Default is latest business day in the dataframe -1 business day,
        """
        if cids is None:
            cids = self.cids
        elif not isinstance(cids, list) or not all(
            isinstance(cid, str) for cid in cids
        ):
            raise TypeError("cids must be a list of strings")

        if xcats is None:
            xcats = self.xcats
        elif not isinstance(xcats, list) or not all(
            isinstance(xcat, str) for xcat in xcats
        ):
            raise TypeError("xcats must be a list of strings")
        else:
            xcats = [xcat + self.postfix for xcat in xcats]

        if not isinstance(transpose, bool):
            raise TypeError("transpose must be a boolean")

        if date is not None:
            if not isinstance(date, str):
                raise TypeError("start must be a string")
            date = pd.to_datetime(date)

        df = self.df[self.df["xcat"].isin(xcats)]

        df = df[df["cid"].isin(cids)]

        if date is None:
            max_date = df["real_date"].max()
            date = max_date - BDay(1)

        df = df[df["real_date"] == date]

        dfw = df.pivot(index="cid", columns="xcat", values="value")

        # If xcats contains the composite, it is moved to the first column
        composite_zscore = self.xcat_comp
        if composite_zscore in xcats:
            dfw = dfw[
                [composite_zscore]
                + [xcat for xcat in dfw.columns if xcat != composite_zscore]
            ]

        if transpose:
            dfw = dfw.transpose()

        if title is None:
            title = f"Snapshot for {date.strftime('%Y-%m-%d')}"

        self._plot_heatmap(dfw, title=title, annot=annot, xticks=xticks, figsize=figsize)

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
        xticks: dict = None,
        figsize: tuple = (20, 10)
    ):
        """
        :param <List[str]> cids: A list of cids whose values are displayed. Default is all in the class
        :param <str> xcat: Single xcat to be displayed. Default is xcat_comp.
        :param<str> freq: frequency to which values are aggregated, i.e. averaged. Default is annual (A). The alternative is quarterly (Q) or bi-annnual (6M)
        :param <bool> include_latest_period: include the latest period average as defined by freq, even if it is not complete. Default is True.
        :param <bool> include_latest_day: include the latest working day date as defined by freq, even if it is not complete. Default is True.
        :param <str> start: ISO-8601 formatted date string. Select data from
            this date onwards. If None, all dates are selected.
        :param <bool> transpose: If False (default) rows are time periods and columns are cids. If True rows are cids and columns are time periods.
        """

        if cids is None:
            cids = self.cids
        elif not isinstance(cids, list) or not all(
            isinstance(cid, str) for cid in cids
        ):
            raise TypeError("cids must be a list of strings")

        if freq not in ["Q", "A", "BA"]:
            raise ValueError("freq must be 'Q', 'A', or 'BA'")

        if freq == "BA":
            freq = "6MS"

        if not isinstance(xcat, str):
            raise TypeError("xcat must be a string")

        if not isinstance(transpose, bool):
            raise TypeError("transpose must be a boolean")

        if start is not None:
            if not isinstance(start, str):
                raise TypeError("start must be a string")

        df = self.df[self.df["xcat"] == xcat + self.postfix].drop(columns=["xcat"])

        df = df[df["cid"].isin(cids)]

        if start is None:
            start = df["real_date"].max()
        else:
            df = df[df["real_date"] >= start]

        dfw = df.pivot(index="real_date", columns="cid", values="value")

        dfw_resampled = dfw.resample(freq, origin="start").mean()
        if not include_latest_period:
            dfw_resampled = dfw_resampled.iloc[:-1]

        if include_latest_day:
            latest_day = dfw.loc[df["real_date"].max()]
            dfw_resampled.loc[df["real_date"].max()] = latest_day
            dfw_resampled.index = list(
                dfw_resampled.index.strftime("%Y-%m-%d")[:-1]
            ) + ["Latest Day"]
        else:
            dfw_resampled.index = list(dfw_resampled.index.strftime("%Y-%m-%d"))

        dfw_resampled = dfw_resampled.transpose()

        if transpose:
            dfw_resampled = dfw_resampled.transpose()

        if title is None:
            title = f"Score Evolution for {xcat}"

        self._plot_heatmap(dfw_resampled, title=title, annot=annot, xticks=xticks, figsize=figsize)

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
        xticks: dict = None,
        figsize: tuple = (20, 10)
    ):
        """
        :param <str> cid: Single cid to be displayed
        :param <List[str]> xcats: A list of xcats to be displayed in the given order. Default is all in the class, including the composite, with the latter being the first row (or column).
        :param<str> freq: frequency to which values are aggregated, i.e. averaged. Default is annual (A). The alternative is quarterly (Q) or bi-annnual (BA)
        :param <bool> include_latest_period: include the latest period average as defined by freq, even if it is not complete. Default is True.
        :param <bool> include_latest_day: include the latest working day date as defined by freq, even if it is not complete. Default is True.
        :param <str> start: ISO-8601 formatted date string. Select data from
            this date onwards. If None, all dates are selected.
        :param <bool> transpose: If False (default) rows are xcats and columns are time periods. If True rows are time periods and columns are xcats.
        """

        if not isinstance(cid, str):
            raise TypeError("cid must be a string")

        if freq not in ["Q", "A", "BA"]:
            raise ValueError("freq must be 'Q', 'A', or 'BA'")

        if freq == "BA":
            freq = "6MS"

        if xcats is None:
            xcats = self.xcats
        elif not isinstance(xcats, list) or not all(
            isinstance(xcat, str) for xcat in xcats
        ):
            raise TypeError("xcats must be a list of strings")
        else:
            xcats = [xcat + self.postfix for xcat in xcats]

        if not isinstance(transpose, bool):
            raise TypeError("transpose must be a boolean")

        if start is not None and not isinstance(start, str):
            raise TypeError("start must be a string")

        df = self.df[self.df["cid"] == cid].drop(columns=["cid"])

        if start is not None:
            df = df[df["real_date"] >= start]

        df = df[df["xcat"].isin(xcats)]

        dfw = df.pivot(index="real_date", columns="xcat", values="value")

        dfw_resampled = dfw.resample(freq).mean()
        if not include_latest_period:
            dfw_resampled = dfw_resampled.iloc[:-1]

        if include_latest_day:
            latest_day = dfw.loc[df["real_date"].max()]
            dfw_resampled.loc[df["real_date"].max()] = latest_day
            dfw_resampled.index = list(
                dfw_resampled.index.strftime("%Y-%m-%d")[:-1]
            ) + ["Latest Day"]
        else:
            dfw_resampled.index = list(dfw_resampled.index.strftime("%Y-%m-%d"))

        dfw_resampled = dfw_resampled.transpose()

        if transpose:
            dfw_resampled = dfw_resampled.transpose()

        if title is None:
            title = f"CID Evolution for {cid}"

        self._plot_heatmap(dfw_resampled, title=title, annot=annot, xticks=xticks, figsize=figsize)

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
        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm)

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

    sv.view_snapshot(cids=cids, transpose=True, figsize=(14,12))
    # sv.view_snapshot(cids=cids, xcats=xcats, transpose=True)
    sv.view_cid_evolution(cid="USD", xcats=xcats, freq="A", transpose=False)
    # sv.view_cid_evolution(cid="USD", xcats=xcats, freq="A", transpose=True)
    sv.view_score_evolution(
        xcat="GGIEDGDP_NSA",
        cids=cids,
        freq="BA",
        transpose=False,
        start="2010-01-01",
    )
    # sv.view_score_evolution(xcat="CRESFXGDP_NSA_D1M1ML6", cids=cids, freq="A", transpose=True, start="2010-01-01")

    sv.view_3d_surface("GGIEDGDP_NSA")
