from numbers import Number
from typing import Dict, List, Tuple, Optional, Any, Union

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
from matplotlib import pyplot as plt

from macrosynergy.management import reduce_df

FREQ_TO_DAYS_MAP = {"D": 1, "W": 5, "M": 21, "Q": 63}


def transaction_cost_heatmap(
    df: pd.DataFrame,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    tcost_name: str = "TCOST",
    figsize: Tuple[float, float] = (10, 6),
    exclude_cids: Tuple[str, ...] = ("GLB",),
    label_dict: Dict[str, str] = None,
    title_fontsize: int = 14,
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
    title_fontsize: int
        Font size of the title. Defaults to 14.

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

    ax.set_title(title, fontsize=title_fontsize)
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
    title_fontsize: int = 14,
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
    title_fontsize: int
        Font size of the title. Defaults to 14.

    Returns
    -------
    plt.Axes
        The axes containing the line plot.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    for i, label in enumerate(labels):
        sns.lineplot(x=x_values, y=y_values[i], label=label, ax=ax)

    ax.set_title(title, fontsize=title_fontsize)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    return ax


def covariance_estimates_scatterplot(
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    configs: List[dict],
    title: str = "Bias vs Variance",
    xlabel: str = "Bias",
    ylabel: str = "Variance",
    title_fontsize: int = 14,
    figsize: Tuple[float, float] = (10, 6),
) -> None:

    # define point colours, size, and shape
    z = {"D": 1, "W": 5, "M": 21}
    styles, hues, effective_lbacks = [], [], []
    for config in configs:
        style = config["lback_meth"].upper()
        hue = "-".join(config["est_freqs"])

        est_freqs = config["est_freqs"]
        if style == "XMA":
            half_life = np.array(config["half_life"])
            lam = 2 ** (-1 / half_life)
            eff_lback = (1 + lam) / (1 - lam)
        else:
            eff_lback = np.array(config["lback_periods"])

        if len(est_freqs) > 1:
            weights = np.array(config["est_weights"], dtype=np.float32)
            weights /= np.sum(weights)
        else:
            weights = np.ones(len(est_freqs), dtype=np.float32)

        eff_lback = np.average(
            [z[freq] * per for freq, per in zip(est_freqs, eff_lback)],
            weights=weights,
        ).round()

        styles.append(style)
        hues.append(hue)
        effective_lbacks.append(eff_lback)

    # create a dataframe and plot
    plot_df = pd.DataFrame(
        {
            "x_vals": x_vals,
            "y_vals": y_vals,
            "Method": styles,
            "Freq": hues,
            "Effective lookback": effective_lbacks,
        }
    )

    with sns.axes_style("whitegrid"), sns.plotting_context("notebook"):
        fig, ax = plt.subplots(figsize=figsize)

        sns.scatterplot(
            data=plot_df,
            x="x_vals",
            y="y_vals",
            hue="Freq",
            style="Method",
            size="Effective lookback",
            sizes=(20, 250),
            alpha=0.8,
            edgecolor="white",
            linewidth=0.6,
            palette="deep",
            ax=ax,
        )

        ax.set_title(title, fontsize=title_fontsize)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        sns.move_legend(
            ax,
            "upper left",
            bbox_to_anchor=(1.02, 1),
            borderaxespad=0,
            frameon=True,
            title=None,
        )

        sns.despine()
        plt.tight_layout()
        plt.show()


def notional_positions_scatterplot(
    pos_dfs: List[pd.DataFrame],
    sig_df: pd.DataFrame,
    df_labels: List[str],
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    title_fontsize: int = 14,
    sharex: bool = True,
    sharey: bool = False,
    n_cols: int = 3,
    figsize: Tuple[float, float] = (15, 5),
    point_size: float = 5,
) -> Tuple[plt.Figure, Any]:
    with sns.axes_style("whitegrid"), sns.plotting_context("notebook"):
        fig, axes = plt.subplots(
            nrows=1 + (len(pos_dfs) // (n_cols + 1)),
            ncols=n_cols,
            figsize=figsize,
            sharex=sharex,
            sharey=sharey,
        )
        axes = np.atleast_2d(axes)

        piv_sig = sig_df.pivot(index="real_date", columns="cid", values="value")
        x_vals = piv_sig.abs().sum(axis=1)  # signals

        for i in range(len(pos_dfs)):
            piv_pos = pos_dfs[i].pivot(index="real_date", columns="cid", values="value")
            y_vals = piv_pos.abs().sum(axis=1) # positions

            ax = axes[i // n_cols, i % n_cols]
            sns.scatterplot(
                x=x_vals,
                y=y_vals,
                ax=ax,
                s=point_size,
            )

            ax.set_title(df_labels[i], fontsize=10, fontweight="bold")
            ax.set_xlabel("")
            ax.set_ylabel("")

        if title:
            fig.suptitle(title, fontsize=title_fontsize)

        if ylabel:
            fig.supylabel(ylabel, fontsize=11)
        if xlabel:
            fig.supxlabel(xlabel, fontsize=11)

        fig.tight_layout()

    return fig, axes


def proxy_pnl_plot(
    pnl_df: pd.DataFrame,
    portfolio_names: Optional[List[str]] = None,
    portfolio_labels: Optional[List[str]] = None,
    background_vals: Optional[pd.Series] = None,
    aum: Optional[Number] = None,
    y_label: str = "",
    x_label: str = "",
    title: str = "",
    legend_title: str = "Portfolio",
    title_fontsize: int = 20,
    legend_fontsize: int = 10,
    label_fontsize: int = 12,
    tick_fontsize: int = 12,
    cumsum: bool = True,
    line_width: int = 1,
    figsize: Tuple[float, float] = (12, 7),
) -> Tuple[plt.Figure, Any]:
    # checks
    for arg, type, val in [
        ("pnl_df", pd.DataFrame, pnl_df),
    ]:
        if not isinstance(val, type):
            raise TypeError()

    # reduce pnl_df to cids/xcats of interest
    pnl_df = reduce_df(pnl_df, cids=portfolio_names)
    if pnl_df.empty:
        raise ValueError()

    # aggregate and put on desired scale
    pnl_df["value"] = pnl_df.groupby("cid")["value"].cumsum() if cumsum else pnl_df
    if aum is not None:
        pnl_df["value"] = 100 * pnl_df["value"] / aum

    sns.set_theme(
        style="whitegrid",
        palette="colorblind",
        rc={"figure.figsize": figsize}
    )

    # lineplot
    fig, ax = plt.subplots()
    sns.lineplot(
        data=pnl_df,
        x="real_date",
        y="value",
        hue="cid",
        estimator=None,
        lw=line_width,
        ax=ax,
    )
    plt.title(title, fontsize=title_fontsize)
    plt.legend(
        labels=portfolio_labels,
        title=legend_title,
        title_fontsize=legend_fontsize,
        fontsize=legend_fontsize,
    )
    plt.xlabel(x_label, fontsize=label_fontsize)
    plt.ylabel(y_label, fontsize=label_fontsize)
    ax.tick_params(axis="both", labelsize=tick_fontsize)
    plt.axhline(y=0, color="black", linestyle="--", lw=1)

    # optionally shade the background
    if background_vals is not None:
        cmap = plt.get_cmap("viridis")
        norm = mpl.colors.Normalize(vmin=background_vals.min(), vmax=background_vals.max())

        # Shade each interval between dates
        for i in range(background_vals.shape[0] - 1):
            start = background_vals.index[i]
            end = background_vals.index[i + 1]
            value = background_vals[i]

            ax.axvspan(
                start,
                end,
                color=cmap(norm(value)),
                alpha=0.2,
                zorder=0
            )

        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])

        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label("Signal strength")

    plt.show()

    return fig, ax

