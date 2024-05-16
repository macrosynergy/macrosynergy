"""
A subclass inheriting from `macrosynergy.visuals.plotter.Plotter`,
designed to plot time series data on a line plot.
"""

import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
import matplotlib.pyplot as plt
from macrosynergy.management.utils import is_valid_iso_date
from macrosynergy.visuals.plotter import Plotter
from numbers import Number
from macrosynergy.management.simulate import make_test_df


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
        figsize: Tuple[Number, Number] = (12, 8),
        aspect: Number = 1.618,
        height: Number = 0.8,
        # plot args
        grid: bool = True,
        x_axis_label: Optional[str] = None,
        y_axis_label: Optional[str] = None,
        axis_fontsize: int = 12,
        ax_hline: Optional[Union[Number, List[Number]]] = None,
        ax_vline: Optional[Union[str, List[str]]] = None,
        # title args
        title: Optional[str] = None,
        title_fontsize: int = 16,
        title_xadjust: Number = 0.5,
        title_yadjust: Number = 1.05,
        # legend args
        legend: bool = True,
        legend_labels: Optional[List[str]] = None,
        legend_title: Optional[str] = None,
        legend_loc: Optional[str] = "best",
        legend_fontsize: int = 14,
        legend_ncol: int = 1,
        legend_bbox_to_anchor: Optional[Tuple[Number, Number]] = None,
        legend_frame: bool = True,
        # return args
        show: bool = True,
        save_to_file: Optional[str] = None,
        dpi: int = 300,
        return_figure: bool = False,
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
            fig, ax = plt.subplots(figsize=figsize)

        dfx: pd.DataFrame = self.df.copy()

        if ax_vline is not None:
            # check it's all valid iso-8601 dates
            if isinstance(ax_vline, str):
                ax_vline = [ax_vline]
            for vline in ax_vline:
                if not is_valid_iso_date(vline):
                    raise ValueError(
                        f"Invalid date format for `ax_vline`: {vline}. "
                        "Please use ISO-8601 formatted dates."
                    )

        if ax_hline is not None:
            if isinstance(ax_hline, Number):
                ax_hline = [ax_hline]
            for hline in ax_hline:
                if not isinstance(hline, Number):
                    raise ValueError(
                        f"Invalid value for `ax_hline`: {hline}. "
                        "Please use a number."
                    )

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

        if ax_vline is not None:
            for vline in ax_vline:
                ax.axvline(x=pd.to_datetime(vline), color="black", linestyle="--")

        if ax_hline is not None:
            for hline in ax_hline:
                ax.axhline(y=hline, color="black", linestyle="--")

        if grid:
            ax.grid(axis="both", linestyle="--", alpha=0.5)

        if x_axis_label:
            ax.set_xlabel(x_axis_label, fontsize=axis_fontsize)

        if y_axis_label:
            ax.set_ylabel(y_axis_label, fontsize=axis_fontsize)

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

    cids = ["USD", "EUR", "GBP", "JPY"]
    xcats = ["RIR_NSA", "FXXR_NSA", "EQXR_NSA"]
    df = make_test_df(cids=cids, xcats=xcats)

    import time

    timer_start: float = time.time()

    LinePlot(df).plot(
        title=(
            "Test Title with a very long title to see how it looks, \n and a new "
            "line - why not?"
        ),
        legend_fontsize=8,
        compare_series="USD_RIR_NSA",
        ax_hline=[45, 55],
        ax_vline=["2015-4-01", "2015-09-01"],
    )

    # facet_size=(5, 4),
    print(f"Time taken: {time.time() - timer_start}")
