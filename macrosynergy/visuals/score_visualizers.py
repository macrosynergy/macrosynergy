from typing import Dict, List

from matplotlib import pyplot as plt
import pandas as pd

import seaborn as sns

from macrosynergy.management.utils.df_utils import reduce_df, update_df
from macrosynergy.panel import linear_composite, make_zn_scores


class ScoreVisualizers(object):
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
        self.xcat_comp = xcat_comp
        self.weights = weights

        composite_df = linear_composite(
            df,
            xcats=self.xcats,
            cids=self.cids,
            weights=self.weights,
            new_xcat=self.xcat_comp,
            complete_xcats=False
        )  # Look at adding signs and blacklisting

        self.df = make_zn_scores(
            composite_df,
            xcat=xcat_comp,
            sequential=True,
            cids=cids,
            blacklist=blacklist,
            iis=True,
            neutral="mean",
            pan_weight=0.75,
            min_obs=261,
            est_freq="m",
            postfix="_ZN",
        )

        composite_df = None # Clear memory

        for xcat in self.xcats:
            dfzm = make_zn_scores(
                df,
                xcat=xcat,
                sequential=True,
                cids=cids,
                blacklist=blacklist,
                iis=True,
                neutral="mean",
                pan_weight=0.75,
                min_obs=261,
                est_freq="m",
                postfix="_ZN",
            )
            self.df = update_df(self.df, dfzm)

        self.xcats = [self.xcat_comp] + self.xcats

    def _plot_heatmap(self, df: pd.DataFrame, title: str, annot: bool = False):
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(df, cmap="coolwarm", annot=annot, xticklabels=True, yticklabels=True, ax=ax)

        ax.set_yticklabels(ax.get_yticklabels())
        ax.set_xticklabels(ax.get_xticklabels())
        ax.set_title(title, fontsize=14)

        plt.tight_layout()
        plt.show()


    def view_snapshot(
        self,
        cids: List[str],
        xcats: List[str],
        transpose: bool = False,
        start: str = None,
    ):
        """
        Display a multiple scores for multiple countries for the latest available or any previous date

        :param <List[str]> cids: A list of cids whose values are displayed. Default is all in the class
        :param <List[str]> xcats: A list of xcats to be displayed in the given order. Default is all in the class, including the composite, with the latter being the first row (or column).
        :param <bool> transpose: If False (default) rows are cids and columns are xcats. If True rows are xcats and columns are cids.
        :param <str> start: ISO-8601 formatted date string giving the date (or nearest previous if not available). Default is latest day in the dataframe,
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

        if not isinstance(transpose, bool):
            raise TypeError("transpose must be a boolean")

        if start is not None:
            if not isinstance(start, str):
                raise TypeError("start must be a string")

        if start is None:
            start = self.df["real_date"].max()

        df = self.df[self.df["real_date"] == start]

        # Filter dataframe to only contain xcats that end in _ZN

        df = df[df["xcat"].str.endswith("_ZN")]

        df = df[df["cid"].isin(cids)]

        dfw = df.pivot(index="cid", columns="xcat", values="value")

        if transpose:
            dfw = dfw.transpose()

        self._plot_heatmap(dfw, f"Snapshot for {start.strftime('%Y-%m-%d')}")

    def view_score_evolution(
        self,
        cids: List[str],
        xcat: str,
        freq: str,
        include_latest_period: bool = True,
        include_latest_day: bool = True,
        start: str = None,
        transpose: bool = False,
    ):
        """
        :param <List[str]> cids: A list of cids whose values are displayed. Default is all in the class
        :param <str> xcat: Single xcat to be displayed. Default is xcat_comp.
        :param<str> freq: frequency to which values are aggregated, i.e. averaged. Default is annual (A). The alternative is quarterly (Q) or bi-annnual (BA)
        :param <bool> include_latest_period: include the latest period average as defined by freq, even if it is not complete. Default is True.
        :param <bool> include_latest_day: include the latest working day date as defined by freq, even if it is not complete. Default is True.
        :param <str> start: ISO-8601 formatted date string. Select data from
            this date onwards. If None, all dates are selected.
        :param <bool> transpose: If False (default) rows are cids and columns are time periods. If True rows are time periods and columns are cids.
        """

        if cids is None:
            cids = self.cids
        elif not isinstance(cids, list) or not all(
            isinstance(cid, str) for cid in cids
        ):
            raise TypeError("cids must be a list of strings")

        if not isinstance(xcat, str):
            raise TypeError("xcat must be a string")

        if not isinstance(transpose, bool):
            raise TypeError("transpose must be a boolean")

        if start is not None:
            if not isinstance(start, str):
                raise TypeError("start must be a string")
            
        df = self.df[self.df["xcat"] == xcat + "_ZN"].drop(columns=["xcat"])

        df = df[df["cid"].isin(cids)]

        if start is None:
            start = df["real_date"].max()
        else:
            df = df[df["real_date"] >= start]

        dfw = df.pivot(index="real_date", columns="cid", values="value")

        dfw_resampled = dfw.resample(freq).mean()
        if not include_latest_period:
            dfw_resampled = dfw_resampled.iloc[:-1]

        if include_latest_day:
            latest_day = dfw.loc[df["real_date"].max()]
            dfw_resampled.loc[df["real_date"].max()] = latest_day
            dfw_resampled.index = list(dfw_resampled.index.strftime("%Y-%m-%d")[:-1]) + ["Latest Day"]
        else:
            dfw_resampled.index = list(dfw_resampled.index.strftime("%Y-%m-%d"))

        if transpose:
            dfw_resampled = dfw_resampled.transpose()

        self._plot_heatmap(dfw_resampled, f"Score Evolution for {xcat}")

    def view_cid_evolution(
        self,
        cid: str,
        xcats: List[str],
        freq: str,
        include_latest_period: bool = True,
        include_latest_day: bool = True,
        start: str = None,
        transpose: bool = False,
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

        if xcats is None:
            xcats = self.xcats
        elif not isinstance(xcats, list) or not all(isinstance(xcat, str) for xcat in xcats):
            raise TypeError("xcats must be a list of strings")

        if not isinstance(transpose, bool):
            raise TypeError("transpose must be a boolean")

        if start is not None and not isinstance(start, str):
            raise TypeError("start must be a string")

        df = self.df[self.df["cid"] == cid].drop(columns=["cid"])

        if start is not None:
            df = df[df["real_date"] >= start]

        df = df[df["xcat"].str.endswith("_ZN")]

        dfw = df.pivot(index="real_date", columns="xcat", values="value")

        dfw_resampled = dfw.resample(freq).mean()
        if not include_latest_period:
            dfw_resampled = dfw_resampled.iloc[:-1]

        if include_latest_day:
            latest_day = dfw.loc[df["real_date"].max()]
            dfw_resampled.loc[df["real_date"].max()] = latest_day
            dfw_resampled.index = list(dfw_resampled.index.strftime("%Y-%m-%d")[:-1]) + ["Latest Day"]
        else:
            dfw_resampled.index = list(dfw_resampled.index.strftime("%Y-%m-%d"))

        if transpose:
            dfw_resampled = dfw_resampled.transpose()

        self._plot_heatmap(dfw_resampled, f"CID Evolution for {cid}")


if __name__ == "__main__":
    cids_dm = ["AUD", "CAD", "CHF", "EUR", "GBP", "JPY", "NOK", "NZD", "SEK", "USD"]
    cids_em = [
        "CLP",
        "CNY",
        "COP",
        "CZK",
        "HKD",
        "HUF",
        "IDR",
        "ILS",
        "INR",
        "KRW",
        "MXN",
        "PLN",
        "RON",
        "RUB",
        "SGD",
        "THB",
        "TRY",
        "TWD",
        "ZAR",
    ]
    cids = cids_dm + cids_em
    main = [
        "CRESFXGDP_NSA_D1M1ML6",  # Currency reserve expansion as % of GDP
        "MBASEGDP_SA_D1M1ML6",  # Monetary base expansion as % of GDP
        "INTLIQGDP_NSA_D1M1ML3",  # Intervention-driven liquidity expansion as % of GDP, diff over 3 months
        "INTLIQGDP_NSA_D1M1ML6",  # Intervention-driven liquidity expansion as % of GDP, diff over 6 months
    ]

    rets = [
        "DU05YXR_NSA",
        "DU05YXR_VT10",
        "EQXR_NSA",
        "EQXR_VT10",
        "FXTARGETED_NSA",
        "FXUNTRADABLE_NSA",
        "FXXR_VT10",
    ]

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

    sv = ScoreVisualizers(df, cids=cids, xcats=xcats)

    xcats = ["DU05YXR_NSA", "DU05YXR_VT10", "EQXR_NSA"]
    # cids = ["AUD", "CAD", "GBP", "USD"]

    sv.view_snapshot(cids=cids, xcats=xcats, transpose=False)
    sv.view_snapshot(cids=cids, xcats=xcats, transpose=True)
    sv.view_cid_evolution(cid="USD", xcats=xcats, freq="A", transpose=False)
    sv.view_cid_evolution(cid="USD", xcats=xcats, freq="A", transpose=True)
    sv.view_score_evolution(xcat="CRESFXGDP_NSA_D1M1ML6", cids=cids, freq="A", transpose=False, start="2010-01-01")
    sv.view_score_evolution(xcat="CRESFXGDP_NSA_D1M1ML6", cids=cids, freq="A", transpose=True, start="2010-01-01")
