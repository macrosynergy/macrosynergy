import numpy as np
import pandas as pd
from typing import List, Optional, Tuple, Union
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
    annot: Union[bool, np.ndarray, pd.DataFrame] = True,
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
    annot : bool or array-like of str
        if a bool, controls whether the numeric values of ``df`` are annotated.
        If a DataFrame or 2D array of strings is supplied, those strings are
        rendered as cell annotations verbatim (``fmt`` is ignored).
    fmt : str
        string format for annotations. Default is '.2f'. Ignored when ``annot``
        is array-like.
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

    annot_fmt = fmt
    if isinstance(annot, (pd.DataFrame, np.ndarray)):
        annot_arr = annot.values if isinstance(annot, pd.DataFrame) else annot
        if annot_arr.shape != df.shape:
            raise ValueError(
                "annot array shape must match the DataFrame shape "
                f"{df.shape}, got {annot_arr.shape}."
            )
        annot = annot_arr
        annot_fmt = ""

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
        fmt=annot_fmt,
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
