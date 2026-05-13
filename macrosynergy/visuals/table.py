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
