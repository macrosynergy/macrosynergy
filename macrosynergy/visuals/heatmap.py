"""
A subclass inheriting from `macrosynergy.visuals.plotter.Plotter`,
designed to plot time series data as a heatmap.
"""

import pandas as pd
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from macrosynergy.visuals.plotter import Plotter
from macrosynergy.visuals.common import Numeric, NoneType

from macrosynergy.management.simulate_quantamental_data import make_test_df
from macrosynergy.management.shape_dfs import reduce_df


class Heatmap(Plotter):
    """
    Class for plotting time series data as a heatmap.
    Inherits from `macrosynergy.visuals.plotter.Plotter`.

    Parameters
    :param <pd.DataFrame> df: A DataFrame with the following columns:
        'cid', 'xcat', 'real_date', and at least one metric from -
        'value', 'grading', 'eop_lag', or 'mop_lag'.
    :param <List[str]> cids: A list of cids to select from the DataFrame.
        If None, all cids are selected.
    :param <List[str]> xcats: A list of xcats to select from the DataFrame.
        If None, all xcats are selected.
    :param <List[str]> metrics: A list of metrics to select from the DataFrame.
        If None, all metrics are selected.
    :param <str> start_date: ISO-8601 formatted date. Select data from
        this date onwards. If None, all dates are selected.
    :param <str> end_date: ISO-8601 formatted date. Select data up to
        and including this date. If None, all dates are selected.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        cids: Optional[List[str]] = None,
        xcats: Optional[List[str]] = None,
        metrics: Optional[List[str]] = None,
        intersect: bool = False,
        tickers: Optional[List[str]] = None,
        blacklist: Optional[Dict[str, List[str]]] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        *args,
        **kwargs,
    ):
        super().__init__(
            df=df,
            cids=cids,
            xcats=xcats,
            metrics=metrics,
            intersect=intersect,
            tickers=tickers,
            blacklist=blacklist,
            start=start,
            end=end,
            *args,
            **kwargs,
        )

    def plot(
        self,
        freq: str = "M",
        agg: str = "mean",
        metric: str = "eop_lag",
        xcat: str = "USD",
        title: Optional[str] = None,
        figsize: Tuple[Numeric, Numeric] = (12, 8),
        on_axis: Optional[plt.Axes] = None,
        # args, kwargs
        *args,
        **kwargs,
    ):
        if on_axis:
            fig: plt.Figure = on_axis.get_figure()
            ax: plt.Axes = on_axis
        else:
            fig: plt.Figure
            ax: plt.Axes
            fig, ax = plt.subplots(figsize=figsize, layout="constrained")

        filtered_df: pd.DataFrame = reduce_df(
            df=self.df, cids=self.cids, xcats=self.xcats, start=self.start, end=self.end
        )

        # group by cid, then set the index to real_date, then resample to the required freq and agg
        filtered_df: pd.DataFrame = (
            filtered_df.groupby(["cid", "xcat"])
            .apply(
                lambda x: x.set_index("real_date")
                .resample(freq)
                .agg(agg, numeric_only=True)
            )
            .reset_index()
        )

        filtered_df["real_date"]: pd.Series = filtered_df["real_date"].dt.strftime(
            "%Y-%m-%d"
        )

        pivoted_df: pd.DataFrame = filtered_df.pivot_table(
            index="cid", columns="real_date", values=metric
        )

        max_mes: float = max(1, filtered_df[metric].max())
        min_mes: float = min(0, filtered_df[metric].min())

        print(pivoted_df.loc["USD"])
        im = ax.imshow(
            pivoted_df.to_numpy(),
            cmap="Reds",
            vmin=min_mes,
            vmax=max_mes,
            aspect="auto",
            **kwargs
        )

        real_dates = pivoted_df.columns.to_list()

        ax.set_xticks(np.arange(len(real_dates)), labels=real_dates)
        ax.set_yticks(np.arange(len([1])), labels=[xcat])
        
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
        ax.tick_params(which='major', length=4, width=1, direction='out')
        plt.xticks(rotation=90)
        plt.grid(False)


        if title is None:
            title = f"Visualising {metric} for {xcat} from {self.start} to {self.end}"
            ax.set_title(title)
            ax.set_xlabel("Date")
            ax.set_ylabel("Cross Sections")

        plt.show()


if __name__ == "__main__":
    test_cids: List[str] = [
        "USD",
    ]  # "EUR", "GBP"]
    test_xcats: List[str] = ["FX", "IR"]
    dfE: pd.DataFrame = make_test_df(
        cids=test_cids, xcats=test_xcats, style="sharp-hill"
    )

    dfM: pd.DataFrame = make_test_df(
        cids=test_cids, xcats=test_xcats, style="four-bit-sine"
    )

    dfG: pd.DataFrame = make_test_df(cids=test_cids, xcats=test_xcats, style="sine")

    dfE.rename(columns={"value": "eop_lag"}, inplace=True)
    dfM.rename(columns={"value": "mop_lag"}, inplace=True)
    dfG.rename(columns={"value": "grading"}, inplace=True)
    mergeon = ["cid", "xcat", "real_date"]
    dfx: pd.DataFrame = pd.merge(pd.merge(dfE, dfM, on=mergeon), dfG, on=mergeon)

    # view_metrics(
    #     df=dfx,
    #     xcat="FX",
    # )
    # view_metrics(
    #     df=dfx,
    #     xcat="IR",
    #     metric="mop_lag",
    # )
    # view_metrics(
    #     df=dfx,
    #     xcat="IR",
    #     metric="grading",
    # )

    heatmap = Heatmap(df=dfx, xcats=["FX"])
    heatmap.plot(metric="eop_lag")


