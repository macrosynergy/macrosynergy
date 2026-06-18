from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def transaction_cost_heatmap(
    df: pd.DataFrame,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    tcost_name: str = "TCOST",
    figsize: Tuple[float, float] = (10, 6),
    exclude_cids: Tuple[str, ...] = ("GLB",),
    label_dict: Dict[str, str] = None,
) -> plt.Axes:
    """
    Plot a heatmap of summed transaction costs by cross-section and category.

    Transaction-cost categories are selected by matching the suffix of the
    xcat column, summed per (cid, xcat), and arranged into a grid with one
    row per category and one column per cross-section.

    Parameters
    ----------
    df: pd.DataFrame
        Transaction cost data in long format. Must contain cid, xcat,
        and value columns.
    title: str
        Title of the heatmap. Defaults to an empty string.
    xlabel: str
        Label for the x-axis. Defaults to an empty string.
    ylabel: str
        Label for the y-axis. Defaults to an empty string.
    tcost_name: str
        Suffix identifying transaction cost categories in xcat. Only
        categories whose name ends with this string are included. Defaults to
        "TCOST".
    figsize: Tuple[float, float]
        Size of the figure. Defaults to (10, 6).
    exclude_cids: Tuple[str, ...]
        Cross-sections to exclude from the heatmap. Defaults to ("GLB",).
    label_dict: Dict[str, str]
        Optional mapping used to rename categories for display. Defaults to
        None, in which case the original category names are used.

    Returns
    -------
    plt.Axes
        The axes containing the heatmap.
    """
    mask = df["xcat"].str.endswith(tcost_name) & ~df["cid"].isin(exclude_cids)
    data = (
        df.loc[mask]
        .groupby(["cid", "xcat"], as_index=False)["value"]
        .sum()
        .pivot(index="xcat", columns="cid", values="value")
    )

    if label_dict:
        data = data.rename(label_dict)

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(data, cmap="rocket_r", annot=True, fmt=".2f", ax=ax)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    return ax


def sensitivity_plot(
    x_values: np.ndarray,
    y_values: np.ndarray,
    labels: List[str],
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    figsize: Tuple[float, float] = (10, 6),
    ax: plt.Axes = None,
) -> plt.Axes:
    """
    Plot one line per series in a sensitivity analysis.

    Each row of y_values is drawn as a separate line against the shared
    x_values, labelled by the corresponding entry in labels.

    Parameters
    ----------
    x_values: np.ndarray
        Values for the x-axis, shared across all series. For example, a range
        of volatility targets.
    y_values: np.ndarray
        Array of shape (n, len(x_values)) holding the sensitivity-analysis
        results, where n is the number of series to plot.
    labels: List[str]
        Labels for the plotted series, one per row of y_values.
    title: str
        Title of the plot. Defaults to an empty string.
    xlabel: str
        Label for the x-axis. Defaults to an empty string.
    ylabel: str
        Label for the y-axis. Defaults to an empty string.
    figsize: Tuple[float, float]
        Size of the figure, used only when ax is not
        provided. Defaults to (10, 6).
    ax: plt.Axes
        Optional existing axes to draw on. Defaults to None, in which case a
        new figure and axes are created.

    Returns
    -------
    plt.Axes
        The axes containing the line plot.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    for i, label in enumerate(labels):
        sns.lineplot(x=x_values, y=y_values[i], label=label, ax=ax)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    return ax
