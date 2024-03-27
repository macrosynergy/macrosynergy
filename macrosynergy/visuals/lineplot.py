"""
A subclass inheriting from `macrosynergy.visuals.plotter.Plotter`,
designed to plot time series data on a line plot.
"""

import pandas as pd
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt

from macrosynergy.visuals.plotter import Plotter
from macrosynergy.management.types import Numeric, NoneType

from macrosynergy.management.simulate import make_test_df

from statsmodels.tsa.seasonal import seasonal_decompose
from ipywidgets import Checkbox, VBox, interactive_output, Output


class LinePlot(Plotter):
    """
    Class for plotting time series data on a line plot.
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
        # df/plot args
        metric: str = "value",
        compare_series: Optional[str] = None,
        # Plotting specific arguments
        # fig args
        figsize: Tuple[Numeric, Numeric] = (12, 8),
        aspect: Numeric = 1.618,
        height: Numeric = 0.8,
        # plot args
        grid: bool = True,
        x_axis_label: Optional[str] = None,
        y_axis_label: Optional[str] = None,
        axis_fontsize: int = 12,
        # title args
        title: Optional[str] = None,
        title_fontsize: int = 16,
        title_xadjust: Numeric = 0.5,
        title_yadjust: Numeric = 1.05,
        # legend args
        legend: bool = True,
        legend_labels: Optional[List[str]] = None,
        legend_title: Optional[str] = None,
        legend_loc: Optional[str] = "best",
        legend_fontsize: int = 14,
        legend_ncol: int = 1,
        legend_bbox_to_anchor: Optional[Tuple[Numeric, Numeric]] = None,
        legend_frame: bool = True,
        # return args
        show: bool = True,
        save_to_file: Optional[str] = None,
        dpi: int = 300,
        return_figure: bool = False,
        on_axis: Optional[plt.Axes] = None,
        decompose: bool = False,
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
            fig, ax = plt.subplots(figsize=figsize)

        dfx: pd.DataFrame = self.df.copy()

        if title is not None:
            title_newline_adjust: float = 0.2
            if title_yadjust is not None and title_yadjust != 1.0:
                title_newline_adjust = abs(title_yadjust - 1.0)
            title_height: float = title.count("\n") * title_newline_adjust
            # increase the title_yadjust by the title_height
            title_yadjust = title_yadjust + title_height

        ax.set_title(
            title,
            fontsize=title_fontsize,
            x=title_xadjust,
            y=title_yadjust,
        )

        if compare_series:
            _cid, _xcat = compare_series.split("_", 1)
            if _cid not in dfx["cid"].unique() or _xcat not in dfx["xcat"].unique():
                raise ValueError(
                    f"Series `{compare_series}` not in DataFrame - used as "
                    "`compare_series`."
                )

            comp_df = (
                dfx.loc[
                    (dfx["cid"] == _cid) & (dfx["xcat"] == _xcat), ["real_date", metric]
                ]
                .copy()
                .reset_index(drop=True)
            )

        valid_tickers = dfx[["cid", "xcat"]].drop_duplicates().values.tolist()

        for xcat in self.xcats:
            for cid in self.cids:
                # Get the unique cid values in dfx for xcat and check if cid is in it
                if [cid, xcat] in valid_tickers:
                    _df = dfx.loc[(dfx["cid"] == cid) & (dfx["xcat"] == xcat), :].copy()
                    _df = _df.sort_values(by="real_date", ascending=True).reset_index(
                        drop=True
                    )
                    ax.plot(_df["real_date"], _df[metric], label=f"{cid}_{xcat}")

        # if there is a compare_series, plot it on the same axis, using a red dashed line
        if compare_series:
            ax.plot(
                comp_df["real_date"],
                comp_df[metric],
                color="red",
                linestyle="--",
                label=compare_series,
            )

        if grid:
            ax.grid(axis="both", linestyle="--", alpha=0.5)

        if x_axis_label:
            ax.set_xlabel(x_axis_label, fontsize=axis_fontsize)

        if y_axis_label:
            ax.set_ylabel(y_axis_label, fontsize=axis_fontsize)

        if decompose:
            time_series_df = dfx.set_index('real_date')[['value']]
            decomposition = seasonal_decompose(time_series_df, model="additive", period=261)
            trend_check = Checkbox(description="Trend", value=False)
            seasonal_check = Checkbox(description="Seasonality", value=False)
            resid_check = Checkbox(description="Random Walk", value=False)

            plot_output = Output()

            def update_plot(trend, seasonal, resid):
                with plot_output:
                    plot_output.clear_output(wait=True)  # Clear the previous plot
                    # Plot initial time series
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot(time_series_df.index, time_series_df['value'], label='Original')
                    if trend:
                        ax.plot(time_series_df.index, decomposition.trend, label="Trend", color="red")
                    if seasonal:
                        ax.plot(time_series_df.index, decomposition.seasonal, label="Seasonal", color="green")
                    if resid:
                        ax.plot(time_series_df.index, decomposition.resid, label="Residual", color="yellow")
                    ax.legend()
                    plt.show()

            ui = VBox([trend_check, seasonal_check, resid_check])
            out = interactive_output(update_plot, {'trend': trend_check, 'seasonal': seasonal_check, 'resid': resid_check})
            return ui, out, plot_output

        # if there is a legend, add it
        if legend:
            ax.legend(
                labels=legend_labels if legend_labels else None,
                title=legend_title,
                loc=legend_loc,
                fontsize=legend_fontsize,
                ncol=legend_ncol,
                bbox_to_anchor=legend_bbox_to_anchor,
                frameon=legend_frame,
            )

        plt.tight_layout()
        title: str = title if title else f"LinePlot: Viewing `{metric}`"

        if save_to_file:
            plt.savefig(
                save_to_file,
                dpi=dpi,
                bbox_inches="tight",
            )

        if return_figure:
            return fig

        if show:
            plt.show()
            return


if __name__ == "__main__":
    from macrosynergy.management.simulate import make_test_df
    from macrosynergy.download import JPMaQSDownload
    import os

    cids: List[str] = [
        "USD",
        "EUR",
        "GBP",
        "AUD",
        "CAD",
        "JPY",
        "CHF",
        "NZD",
        "SEK",
        "INR",
    ]
    # Quantamental categories of interest

    xcats = [
        "NIR_NSA",
        "RIR_NSA",
        "DU05YXR_NSA",
        "DU05YXR_VT10",
        "FXXR_NSA",
        "EQXR_NSA",
        "DU05YXR_NSA",
    ]  # market links

    cids: List[str] = ["GBP"]
    xcats: List[str] = ["RIR_NSA"]#, "RIR_NSA"]#, "FXXR_NSA", "EQXR_NSA"]

    client_id: str = os.getenv("DQ_CLIENT_ID")
    client_secret: str = os.getenv("DQ_CLIENT_SECRET")

    with JPMaQSDownload(client_id=client_id, client_secret=client_secret) as jpmaqs:
        df: pd.DataFrame = jpmaqs.download(
            cids=cids,
            xcats=xcats,
            start_date="1990-01-01",
        )

    from random import SystemRandom

    random = SystemRandom()

    # random.seed(42)

    # for cidx, xcatx in df[["cid", "xcat"]].drop_duplicates().values.tolist():
    #     # if random() > 0.5 multiply by random.random()*10
    #     _bools = (df["cid"] == cidx) & (df["xcat"] == xcatx)
    #     r = max(random.random(), 0.1)
    #     df.loc[_bools, "value"] = df.loc[_bools, "value"] * r

    # FacetPlot(df).lineplot()
    import time

    print("From same object:")
    timer_start: float = time.time()

    LinePlot(df, cids=cids, xcats=xcats).plot(
        title=(
            "Test Title with a very long title to see how it looks, \n and a new "
            "line - why not?"
        ),
        legend_fontsize=8,
    )

    # facet_size=(5, 4),
    print(f"Time taken: {time.time() - timer_start}")
