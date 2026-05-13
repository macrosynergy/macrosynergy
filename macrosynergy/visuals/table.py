"""
Table and availability visualisation utilities.

Functions
---------
view_table
    Render a numeric DataFrame as an annotated heatmap table.
view_availability
    Render a binary (0/1) wide DataFrame as a colour-coded availability heatmap,
    with columns sorted by recency, frequency, and name.

Design note — proposed integration with ``view_panel_dates``
------------------------------------------------------------
``view_availability`` is a natural complement to ``view_panel_dates`` in
``view_panel_dates.py``.  Both functions render a time × series heatmap, but
they target different data shapes:

* ``view_panel_dates`` — accepts a QDF (long-format or wide with
  date/lag-count values) and uses a continuous ``YlOrBr`` colormap with
  annotations to show missing days or start years.
* ``view_availability`` — expects a pre-pivoted wide DataFrame with binary
  0/1 values and uses a two-tone white/blue colormap to show simple presence
  or absence.

Suggested change to ``view_panel_dates``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Add a binary-detection branch at the top of ``view_panel_dates`` that
delegates to ``view_availability`` when all non-null values are 0 or 1::

    if set(df.stack().dropna().unique()).issubset({0, 1, 0.0, 1.0}):
        return view_availability(
            df,
            title=header or "Variable Availability",
            fig_kw={"figsize": size} if size else None,
            heatmap_kw=heatmap_kw,
            xticklabel_kw=xticklabel_kw,
            yticklabel_kw=yticklabel_kw,
            title_kw={"fontsize": title_fontsize} if title_fontsize else None,
            return_fig=return_fig,
        )

This would also require:

1. Adding ``heatmap_kw``, ``xticklabel_kw``, ``yticklabel_kw``, and
   ``title_kw`` dict parameters to ``view_panel_dates`` (mirroring the
   pattern used here) so both code paths benefit from kwargs flexibility.
2. Keeping ``title_fontsize`` as a named parameter in ``view_panel_dates``
   for backwards compatibility; it should feed into ``title_kw`` rather
   than replace it.
3. Hardcoded ``sns.heatmap`` call in the continuous path of
   ``view_panel_dates`` should be updated to use a ``_heatmap_kw`` dict
   merge, consistent with the pattern used in ``view_availability``.

The change is non-breaking: callers that pass lag-count or start-year data
to ``view_panel_dates`` are unaffected, and ``view_availability`` remains
directly callable for users who already have a binary wide DataFrame.

Exports
-------
``view_availability`` should be added to ``visuals/__init__.py`` once
the integration decision is finalised.
"""

import pandas as pd
from typing import List, Optional, Tuple
import seaborn as sns
import matplotlib.pyplot as plt


def view_table(
    df: pd.DataFrame,
    title: Optional[str] = None,
    title_fontsize: Optional[int] = 16,
    figsize: Optional[Tuple[float, float]] = (14, 4),
    min_color: float = -1,
    max_color: float = 1,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    xticklabels: Optional[List[str]] = None,
    yticklabels: Optional[List[str]] = None,
    annot: bool = True,
    fmt: str = ".2f",
    return_fig: bool = False,
) -> Optional[plt.Figure]:
    """
    Display a numeric DataFrame as an annotated colour-coded heatmap table.

    Parameters
    ----------
    df : pd.DataFrame
        Numeric DataFrame to display.
    title : str, optional
        Title displayed above the heatmap.
    title_fontsize : int, optional
        Font size of the title. Default is 16.
    figsize : Tuple[float, float], optional
        Width and height of the figure in inches.
    min_color : float
        Data value mapped to the bottom of the colormap. Default is -1.
    max_color : float
        Data value mapped to the top of the colormap. Default is 1.
    xlabel : str, optional
        Label for the x-axis.
    ylabel : str, optional
        Label for the y-axis.
    xticklabels : List[str], optional
        Tick labels for the columns. Defaults to the DataFrame column names.
    yticklabels : List[str], optional
        Tick labels for the rows. Defaults to the DataFrame index values.
    annot : bool
        Whether to annotate each cell with its numeric value. Default is True.
    fmt : str
        Format string for cell annotations, for example ".2f". Default is ".2f".
    return_fig : bool
        If True, return the Matplotlib figure instead of displaying it.

    Returns
    -------
    plt.Figure or None
        The figure object when "return_fig" is True, otherwise None.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Table must be a DataFrame")

    if df.empty:
        raise ValueError("Table must not be empty")

    try:
        df = df.astype(float)
    except ValueError:
        raise ValueError("Table must be numeric")

    if xticklabels is None:
        xticklabels = df.columns.to_list()
    elif len(xticklabels) != len(df.columns):
        raise ValueError("Number of xticklabels must match number of columns")

    if yticklabels is None:
        yticklabels = df.index.to_list()
    elif len(yticklabels) != len(df.index):
        raise ValueError("Number of yticklabels must match number of rows")

    fig, ax = plt.subplots(figsize=figsize)
    sns.set(style="ticks")
    sns.heatmap(
        df,
        cmap="vlag_r",
        vmin=min_color,
        vmax=max_color,
        square=False,
        linewidths=0.5,
        cbar_kws={"shrink": 0.5},
        annot=annot,
        fmt=fmt,
        xticklabels=xticklabels,
        yticklabels=yticklabels,
    )

    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.set_title(title, fontsize=title_fontsize)
    plt.tight_layout()

    if return_fig:
        return fig
    else:
        plt.show()


def view_availability(
    df: pd.DataFrame,
    title: str = "Variable Availability",
    n_ticks: int = 10,
    fig_kw: dict = None,
    heatmap_kw: dict = None,
    xticklabel_kw: dict = None,
    yticklabel_kw: dict = None,
    title_kw: dict = None,
    return_fig: bool = False,
) -> Optional[plt.Figure]:
    """
    Plot a binary availability heatmap with columns sorted by recency.

    Columns are ordered descending by last available date, then by total count
    of available observations, then alphabetically.

    Parameters
    ----------
    df : pd.DataFrame
        Wide DataFrame with a DatetimeIndex, string column names, and binary
        (0/1) values indicating availability.
    title : str
        Title displayed above the heatmap. Default is "Variable Availability".
    n_ticks : int
        Number of date labels shown on the x-axis. Default is 10.
    fig_kw : dict, optional
        Keyword arguments forwarded to "plt.subplots", for example "figsize".
    heatmap_kw : dict, optional
        Keyword arguments forwarded to "sns.heatmap", for example "cmap" or
        "linewidths".
    xticklabel_kw : dict, optional
        Keyword arguments forwarded to "ax.set_xticklabels", for example
        "rotation", "fontsize", or "ha".
    yticklabel_kw : dict, optional
        Keyword arguments forwarded to "ax.set_yticklabels", for example
        "rotation" or "fontsize".
    title_kw : dict, optional
        Keyword arguments forwarded to "ax.set_title", for example "fontsize",
        "pad", or "loc".
    return_fig : bool
        If True, return the Matplotlib figure instead of displaying it.

    Returns
    -------
    plt.Figure or None
        The figure object when "return_fig" is True, otherwise None.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"`df` must be a pandas DataFrame, got {type(df).__name__}.")
    if df.empty:
        raise ValueError("`df` must not be empty.")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(
            f"`df` must have a DatetimeIndex, got {type(df.index).__name__}."
        )
    unique_vals = set(df.stack().dropna().unique())
    if not unique_vals.issubset({0, 1, 0.0, 1.0}):
        raise ValueError(
            "`df` must contain only binary (0/1) values; "
            f"found unexpected values: {unique_vals - {0, 1, 0.0, 1.0}}."
        )
    if not isinstance(n_ticks, int) or n_ticks < 1:
        raise ValueError(f"`n_ticks` must be a positive integer, got {n_ticks!r}.")
    for name, val in [
        ("fig_kw", fig_kw),
        ("heatmap_kw", heatmap_kw),
        ("xticklabel_kw", xticklabel_kw),
        ("yticklabel_kw", yticklabel_kw),
        ("title_kw", title_kw),
    ]:
        if val is not None and not isinstance(val, dict):
            raise TypeError(
                f"`{name}` must be a dict or None, got {type(val).__name__}."
            )

    # Sort columns: descending by last date traded, then by count traded, then alphabetically.
    last_date = df.apply(lambda s: s[s > 0].index.max() if s.any() else pd.NaT)
    count_available = df.sum()
    sort_keys = pd.DataFrame(
        {"last_date": last_date, "count": count_available, "name": df.columns}
    )
    sort_keys = sort_keys.sort_values(
        ["last_date", "count", "name"],
        ascending=[False, False, False],
    )
    df = df[sort_keys.index]

    _fig_kw = {"figsize": (14, max(3, len(df.columns) * 0.4))}
    if fig_kw:
        _fig_kw.update(fig_kw)

    fig, ax = plt.subplots(**_fig_kw)

    _heatmap_kw = {
        "cmap": ["white", "#1f77b4"],
        "vmin": 0,
        "vmax": 1,
        "linewidths": 0,
        "cbar": False,
        "xticklabels": False,
    }
    if heatmap_kw:
        _heatmap_kw.update(heatmap_kw)

    sns.heatmap(df.T, ax=ax, **_heatmap_kw)

    tick_positions = [int(i) for i in range(0, len(df), max(1, len(df) // n_ticks))]
    ax.set_xticks(tick_positions)

    _xticklabel_kw = {"rotation": 45, "ha": "right", "fontsize": 9}
    if xticklabel_kw:
        _xticklabel_kw.update(xticklabel_kw)
    ax.set_xticklabels(
        [df.index[i].strftime("%Y-%m") for i in tick_positions],
        **_xticklabel_kw,
    )

    _yticklabel_kw = {"rotation": 0, "fontsize": 9}
    if yticklabel_kw:
        _yticklabel_kw.update(yticklabel_kw)
    ax.set_yticklabels(ax.get_yticklabels(), **_yticklabel_kw)

    ax.set_xlabel("")
    ax.set_ylabel("")

    _title_kw = {"fontsize": 12, "pad": 10}
    if title_kw:
        _title_kw.update(title_kw)
    ax.set_title(title, **_title_kw)

    plt.tight_layout()

    if return_fig:
        return fig
    else:
        plt.show()


if __name__ == "__main__":
    data = {
        "Col1": [1, 2, 3, 4],
        "Col2": [5, 6, 7, 8],
        "Col3": [9, 10, 11, 12],
        "Col4": [13, 14, 15, 16],
    }
    row_labels = ["Row1", "Row2", "Row3", "Row4"]

    df = pd.DataFrame(data, index=row_labels)

    view_table(df, title="Table")
