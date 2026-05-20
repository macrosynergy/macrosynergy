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
    highlight_mask: Optional[Union[np.ndarray, pd.DataFrame]] = None,
    footnote: Optional[str] = None,
    footnote_fontsize: int = 10,
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
    highlight_mask : array-like of bool, optional
        DataFrame or 2D array of the same shape as ``df``. Cells where the
        mask is True have their annotation text rendered in black and bold.
        Has no effect when ``annot`` is False.
    footnote : str, optional
        Free-text caption rendered below the heatmap, useful for noting the
        statistical test, the panel scope, or how to read the annotations.
        Multi-line strings are supported. Default is None (no footnote).
    footnote_fontsize : int, optional
        Font size for the footnote text. Default is 10.
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

    if highlight_mask is not None and not (annot is False):
        if isinstance(highlight_mask, pd.DataFrame):
            mask_arr = highlight_mask.values
        else:
            mask_arr = np.asarray(highlight_mask)
        if mask_arr.shape != df.shape:
            raise ValueError(
                "highlight_mask shape must match the DataFrame shape "
                f"{df.shape}, got {mask_arr.shape}."
            )
        # seaborn lays out ax.texts in row-major order matching df.values.ravel()
        for txt, hi in zip(ax.texts, mask_arr.ravel()):
            if bool(hi):
                txt.set_color("black")
                txt.set_weight("bold")

    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.set_title(title, fontsize=title_fontsize)
    plt.tight_layout()

    if footnote:
        n_lines = footnote.count("\n") + 1
        # Reserve bottom margin for the heatmap + tick labels + optional xlabel + footnote.
        # Figure-relative units: tick labels ~0.06, xlabel ~0.04 if present, then footnote.
        xlabel_pad = 0.04 if ax.get_xlabel() else 0.0
        footnote_block = 0.04 * n_lines
        fig.subplots_adjust(bottom=0.10 + xlabel_pad + footnote_block)
        fig.text(
            0.5,
            0.02,
            footnote,
            ha="center",
            va="bottom",
            fontsize=footnote_fontsize,
            style="italic",
            wrap=True,
        )

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
