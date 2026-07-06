import pandas as pd
from typing import Optional
import seaborn as sns
import matplotlib.pyplot as plt

from macrosynergy.management.types import QuantamentalDataFrame
from macrosynergy.management.utils import qdf_to_ticker_df


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
    Plot a binary availability heatmap from a Quantamental DataFrame.

    Tickers (one per cid/xcat pair) are ordered descending by last available
    date, then by total count of available observations, then alphabetically.

    Parameters
    ----------
    df : pd.DataFrame
        Standardised QuantamentalDataFrame with columns "real_date", "cid",
        "xcat", and "value". The "value" column must contain only binary
        (0/1) entries indicating availability.
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
    if not isinstance(df, QuantamentalDataFrame):
        raise TypeError(
            "`df` must be a standardised QuantamentalDataFrame with columns "
            "'real_date', 'cid', 'xcat', and a value column."
        )
    if df.empty:
        raise ValueError("`df` must not be empty.")
    unique_vals = set(df["value"].dropna().unique())
    if not unique_vals.issubset({0, 1, 0.0, 1.0}):
        raise ValueError(
            "`df['value']` must contain only binary (0/1) values; "
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

    df = qdf_to_ticker_df(df, value_column="value").sort_index()
    df.columns = [c.rstrip("_") for c in df.columns]

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
