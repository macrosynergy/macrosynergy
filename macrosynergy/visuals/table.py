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
) -> None:
    """
    Displays a DataFrame representing a table as a heatmap.

    Parameters
    ----------
    table : ~pandas.DataFrame
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

    ax: plt.Axes
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
