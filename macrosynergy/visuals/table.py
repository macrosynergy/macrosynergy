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
):
    """
    Displays a DataFrame representing a table as a heatmap.

    Parameters
    ----------
    df : ~pandas.DataFrame
        table to be displayed.
    title : str, optional
        string of chart title; defaults depend on type of range plot.
    title_fontsize : int, optional
        font size of chart header. Default is 16.
    figsize : Tuple[float, float], optional
        Tuple (w, h) of width and height of plot.
    min_color : float
        minimum value of colorbar. Default is -1.
    max_color : float
        maximum value of colorbar. Default is 1.
    xlabel : str, optional
        string of x-axis label. Default is None.
    ylabel : str, optional
        string of y-axis label. Default is None.
    xticklabels : List[str], optional
        list of strings to label x-axis ticks. Default is None.
    yticklabels : List[str], optional
        list of strings to label y-axis ticks. Default is None.
    annot : bool
        whether to annotate heatmap with values.
    fmt : str
        string format for annotations. Default is '.2f'.
    return_fig : bool
        If True, return the Matplotlib figure object instead of displaying.
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


def plot_availability(
    df: pd.DataFrame,
    title: str = "Variable Availability",
    fig_kw: dict = None,
    heatmap_kw: dict = None,
) -> None:
    """
    Plot a binary availability heatmap.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with DatetimeIndex, string column names, and 0/1 values.
    title : str
        Plot title.
    fig_kw : dict, optional
        Keyword arguments forwarded to ``plt.subplots`` (e.g. ``figsize``).
    heatmap_kw : dict, optional
        Keyword arguments forwarded to ``sns.heatmap`` (e.g. ``cmap``, ``linewidths``).
    """
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

    n_ticks = 10
    tick_positions = [int(i) for i in range(0, len(df), max(1, len(df) // n_ticks))]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(
        [df.index[i].strftime("%Y-%m") for i in tick_positions],
        rotation=45,
        ha="right",
        fontsize=9,
    )

    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(title, fontsize=12, pad=10)

    plt.tight_layout()
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
