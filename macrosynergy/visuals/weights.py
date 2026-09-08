"""Stacked area chart of portfolio allocation weights over time."""

from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from macrosynergy.management.types import QuantamentalDataFrame
from macrosynergy.management.utils import reduce_df


def view_weights(
    df: Union[pd.DataFrame, QuantamentalDataFrame],
    xcat: str = None,
    cids: List[str] = None,
    start: str = None,
    end: str = None,
    blacklist: dict = None,
    freq: str = None,
    title: str = "Allocation weights over time",
    title_fontsize: int = 14,
    cid_labels: Dict[str, str] = None,
    xlabel: str = "",
    ylabel: str = "weight",
    label_fontsize: int = 12,
    tick_fontsize: int = None,
    cmap: str = "tab20",
    figsize: Tuple[float, float] = (12, 6),
    legend: bool = True,
    legend_fontsize: int = 9,
    return_fig: bool = False,
) -> Optional[plt.Figure]:
    """
    Plot allocation weights over time as a stacked area chart.

    The stack shows the composition of the portfolio at each date, so the tilts between
    cross-sections and the share held in any residual sleeve, such as cash, are visible
    at full resolution. Where the weights of a date sum to one, the stack reaches one.

    Weights may be passed either as a wide dataframe, indexed by date with one column
    per cross-section, or as a `QuantamentalDataFrame` together with the `xcat` holding
    the weights.

    Parameters
    ----------
    df : ~pandas.DataFrame or QuantamentalDataFrame
        weights to be plotted. Either a wide dataframe, indexed by date with one column
        per cross-section, or a standardised quantamental dataframe with the columns
        'cid', 'xcat', 'real_date' and 'value'.
    xcat : str
        category holding the weights. Required when `df` is a quantamental dataframe,
        and ignored when `df` is already wide.
    cids : List[str]
        cross-sections to be plotted, in the order given, which is also the order they
        are stacked in. Default is None and all available cross-sections are used, in
        the order of the columns of a wide `df` or sorted for a quantamental one.
    start : str
        earliest date in ISO format. Default is None and the earliest date of the
        weights is used.
    end : str
        latest date in ISO format. Default is None and the latest date of the weights is
        used.
    blacklist : dict
        cross-sections with date ranges to be excluded. Applies only when `df` is a
        quantamental dataframe.
    freq : str
        frequency the weights are down-sampled to before plotting, using the last
        observation of each period: 'D' (daily), 'W' (weekly), 'M' (monthly), 'Q'
        (quarterly) or 'A' (annual). Default is None and the weights are plotted at the
        frequency they are given in.
    title : str
        allows entering text for a custom chart header.
    title_fontsize : int
        font size of the title. Default is 14.
    cid_labels : Dict[str, str]
        custom labels for the cross-sections, keyed by cross-section, used in the
        legend. Default is None and the cross-section names are used.
    xlabel : str
        label for the x-axis. Default is an empty string.
    ylabel : str
        label for the y-axis. Default is 'weight'.
    label_fontsize : int
        font size for the axis labels. Default is 12.
    tick_fontsize : int
        font size for the axis ticks. Default is None, which uses the matplotlib
        default.
    cmap : str
        name of the colormap used across the stack. Default is 'tab20', which is
        qualitative and therefore suited to a stack of many sleeves.
    figsize : (float, float)
        width and height in inches. Default is (12, 6).
    legend : bool
        if True (default) a legend is drawn to the right of the axes.
    legend_fontsize : int
        font size of the legend. Default is 9.
    return_fig : bool
        if True the figure is returned rather than displayed. Default is False.

    Returns
    -------
    Optional[matplotlib.figure.Figure]
        the figure, if `return_fig` is True.


    .. note::
        Weights that change sign break the reading of a stacked area chart, since the
        bands no longer add up along the vertical axis, and are rejected. A wholly
        negative panel plots as a stack below the axis.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"'df' must be a DataFrame - received {type(df)}.")
    if df.empty:
        raise ValueError("'df' must not be empty.")

    dfw = _to_wide_weights(
        df=df,
        xcat=xcat,
        cids=cids,
        start=start,
        end=end,
        blacklist=blacklist,
    )

    if freq is not None:
        dfw = _downsample_weights(dfw, freq)

    if dfw.empty:
        raise ValueError("No weight data available for the requested period.")

    if cid_labels is None:
        cid_labels = {}
    elif not isinstance(cid_labels, dict):
        raise TypeError(
            "'cid_labels' should be a dictionary with keys as the cross-sections and "
            "values as the custom labels."
        )

    fig = _plot_weight_area(
        dfw=dfw,
        cid_labels=cid_labels,
        title=title,
        title_fontsize=title_fontsize,
        xlabel=xlabel,
        ylabel=ylabel,
        label_fontsize=label_fontsize,
        tick_fontsize=tick_fontsize,
        cmap=cmap,
        figsize=figsize,
        legend=legend,
        legend_fontsize=legend_fontsize,
    )

    if return_fig:
        return fig
    plt.show()


def _plot_weight_area(
    dfw: pd.DataFrame,
    cid_labels: Dict[str, str],
    title: str,
    title_fontsize: int,
    xlabel: str,
    ylabel: str,
    label_fontsize: int,
    tick_fontsize: int,
    cmap: str,
    figsize: Tuple[float, float],
    legend: bool,
    legend_fontsize: int,
) -> plt.Figure:
    """
    Draw the stacked area chart of a wide weights frame.
    """
    values = dfw.to_numpy(dtype=float)
    if np.nanmin(values) < 0 and np.nanmax(values) > 0:
        mixed = [
            cid
            for cid in dfw.columns
            if (dfw[cid] < 0).any() and (dfw[cid] > 0).any()
        ]
        raise ValueError(
            "A stacked area chart cannot represent weights that change sign, but "
            f"{mixed if mixed else 'the weights'} contain both positive and negative "
            "values. Plot long and short exposures separately, or use "
            "`NaivePnL.plot_pnl_attribution` for a signed view."
        )

    dfp = dfw.rename(columns=lambda cid: cid_labels.get(cid, cid))

    fig, ax = plt.subplots(figsize=figsize)
    dfp.plot.area(ax=ax, linewidth=0, colormap=cmap)

    ax.set_title(title, fontsize=title_fontsize)
    ax.set_ylabel(ylabel if ylabel is not None else "", fontsize=label_fontsize)
    ax.set_xlabel(xlabel, fontsize=label_fontsize)
    if tick_fontsize is not None:
        ax.tick_params(axis="both", labelsize=tick_fontsize)

    if legend:
        ax.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=legend_fontsize)
    elif ax.get_legend() is not None:
        ax.get_legend().remove()

    plt.tight_layout()
    return fig


def _to_wide_weights(
    df: pd.DataFrame,
    xcat: str = None,
    cids: List[str] = None,
    start: str = None,
    end: str = None,
    blacklist: dict = None,
) -> pd.DataFrame:
    """
    Coerce weights given either wide or as a quantamental dataframe into a wide frame,
    indexed by date with one column per cross-section in the order requested.
    """
    is_qdf = set(["cid", "xcat", "real_date", "value"]).issubset(df.columns)

    if is_qdf:
        if xcat is None:
            raise ValueError(
                "'xcat' must be given when 'df' is a quantamental dataframe, to "
                "identify the category holding the weights."
            )
        dfr = reduce_df(
            df, xcats=[xcat], cids=cids, start=start, end=end, blacklist=blacklist
        )
        if dfr.empty:
            raise ValueError(
                f"No data available for '{xcat}' over the requested cross-sections and "
                "period."
            )
        dfw = dfr.pivot(index="real_date", columns="cid", values="value")
    else:
        if xcat is not None:
            raise ValueError(
                "'xcat' is only applicable when 'df' is a quantamental dataframe; the "
                "columns of a wide dataframe are already the cross-sections."
            )
        dfw = df.copy()
        if not isinstance(dfw.index, pd.DatetimeIndex):
            dfw.index = pd.to_datetime(dfw.index)

    dfw.columns = pd.Index([str(c) for c in dfw.columns], name="cid")

    if cids is not None:
        missing = [cid for cid in cids if cid not in dfw.columns]
        if missing:
            raise ValueError(
                f"No weights available for {missing}. Available cross-sections are "
                f"{list(dfw.columns)}."
            )
        dfw = dfw.reindex(columns=list(cids))

    if not is_qdf:
        dfw = dfw.truncate(before=start, after=end)

    return dfw.dropna(how="all")


def _downsample_weights(dfw: pd.DataFrame, freq: str) -> pd.DataFrame:
    """
    Down-sample a wide weights frame to `freq`, taking the last observation of each
    period so that the weights shown are ones actually held.
    """
    freq_map = {"D": None, "W": "week", "M": "month", "Q": "quarter", "A": "year"}
    if not isinstance(freq, str) or freq.upper() not in freq_map:
        raise ValueError(f"'freq' must be one of {list(freq_map)} - received {freq}.")
    freq = freq.upper()

    if freq == "D":
        return dfw

    index = dfw.index
    if freq == "W":
        keys = [index.isocalendar().year.to_numpy(), index.isocalendar().week.to_numpy()]
    elif freq == "M":
        keys = [index.year, index.month]
    elif freq == "Q":
        keys = [index.year, index.quarter]
    else:
        keys = [index.year]

    dates = index.to_series().groupby(keys).max()
    return dfw.loc[dates.values]
