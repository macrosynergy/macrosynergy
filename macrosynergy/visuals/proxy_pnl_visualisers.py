from numbers import Number
from typing import Dict, List, Tuple, Optional, Any, Union, Sequence

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
from matplotlib import pyplot as plt

from macrosynergy.management import reduce_df
from macrosynergy.visuals import timelines

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


# def proxy_pnl_plot(
#     pnl_df: pd.DataFrame,
#     portfolio_names: Optional[List[str]] = None,
#     portfolio_labels: Optional[List[str]] = None,
#     background_vals: Optional[pd.Series] = None,
#     aum: Optional[Number] = None,
#     y_label: str = "",
#     x_label: str = "",
#     title: str = "",
#     legend_title: str = "Portfolio",
#     title_fontsize: int = 20,
#     legend_fontsize: int = 10,
#     label_fontsize: int = 12,
#     tick_fontsize: int = 12,
#     cumsum: bool = True,
#     line_width: int = 1,
#     figsize: Tuple[float, float] = (12, 7),
# ) -> Tuple[plt.Figure, Any]:
#     # checks
#     for arg, type, val in [
#         ("pnl_df", pd.DataFrame, pnl_df),
#     ]:
#         if not isinstance(val, type):
#             raise TypeError()
#
#     # reduce pnl_df to cids/xcats of interest
#     pnl_df = reduce_df(pnl_df, cids=portfolio_names)
#     if pnl_df.empty:
#         raise ValueError()
#
#     # aggregate and put on desired scale
#     pnl_df["value"] = pnl_df.groupby("cid")["value"].cumsum() if cumsum else pnl_df
#     if aum is not None:
#         pnl_df["value"] = 100 * pnl_df["value"] / aum
#
#     sns.set_theme(
#         style="whitegrid",
#         palette="colorblind",
#         rc={"figure.figsize": figsize}
#     )
#
#     # lineplot
#     fig, ax = plt.subplots()
#     sns.lineplot(
#         data=pnl_df,
#         x="real_date",
#         y="value",
#         hue="cid",
#         estimator=None,
#         lw=line_width,
#         ax=ax,
#     )
#     plt.title(title, fontsize=title_fontsize)
#     plt.legend(
#         labels=portfolio_labels,
#         title=legend_title,
#         title_fontsize=legend_fontsize,
#         fontsize=legend_fontsize,
#     )
#     plt.xlabel(x_label, fontsize=label_fontsize)
#     plt.ylabel(y_label, fontsize=label_fontsize)
#     ax.tick_params(axis="both", labelsize=tick_fontsize)
#     plt.axhline(y=0, color="black", linestyle="--", lw=1)
#
#     # optionally shade the background
#     if background_vals is not None:
#         cmap = plt.get_cmap("viridis")
#         norm = mpl.colors.Normalize(vmin=background_vals.min(), vmax=background_vals.max())
#
#         # Shade each interval between dates
#         for i in range(background_vals.shape[0] - 1):
#             start = background_vals.index[i]
#             end = background_vals.index[i + 1]
#             value = background_vals[i]
#
#             ax.axvspan(
#                 start,
#                 end,
#                 color=cmap(norm(value)),
#                 alpha=0.2,
#                 zorder=0
#             )
#
#         sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
#         sm.set_array([])
#
#         cbar = fig.colorbar(sm, ax=ax)
#         cbar.set_label("Signal strength")
#
#     plt.show()
#
#     return fig, ax

def compare_proxy_pnls(
    pnl_dfs: Union[pd.DataFrame, List[pd.DataFrame]],
    pnle_dfs: Union[pd.DataFrame, List[pd.DataFrame]],
    portfolio_names: Union[str, List[str]],
    pnl_names: Optional[List[str]] = None,
    title: str = "Proxy PnL Comparison",
    title_fontsize: int = 22,
    pnl_incl_costs_name="PNL",
    include_exclude_cost_labels=["Incl. Costs", "Excl. Costs"],
    cumsum: bool = True,
    return_fig: bool = False,
    **kwargs,
) -> plt.Figure:
    if isinstance(pnl_dfs, pd.DataFrame):
        pnl_dfs = [pnl_dfs]
    if isinstance(pnle_dfs, pd.DataFrame):
        pnle_dfs = [pnle_dfs]

    assert len(pnl_dfs) == len(pnle_dfs) == len(portfolio_names)

    pnlcount = len(pnl_dfs)

    pnl_names = portfolio_names if pnl_names is None else pnl_names

    comp_dfs = []
    for i in range(pnlcount):
        pnl_df = reduce_df(
            pd.concat([pnl_dfs[i], pnle_dfs[i]], axis=0, ignore_index=True),
            cids=[portfolio_names[i]],
        )

        pnl_xcats_found = []
        for pnlcatname in [pnl_incl_costs_name, pnl_incl_costs_name + "e"]:
            pnl_xcat = (
                pnl_df["xcat"][pnl_df["xcat"].str.endswith(pnlcatname)]
                .unique()
                .tolist()
            )
            if len(pnl_xcat) != 1:
                raise ValueError(
                    f"Expected exactly one xcat ending with {pnlcatname}, "
                    f"found {len(pnl_xcat)}: {pnl_xcat}"
                )
            pnl_xcats_found.append(pnl_xcat[0])

        if len(pnl_xcats_found) != 2:
            raise ValueError(
                f"Expected exactly two xcats for PnL (including and excluding costs), "
                f"found {len(pnl_xcats_found)}: {pnl_xcats_found}"
            )
        pnlname, pnle_name = sorted(pnl_xcats_found)
        rename_map = dict(zip([pnlname, pnle_name], include_exclude_cost_labels))
        pnl_df["xcat"] = pnl_df["xcat"].replace(rename_map)
        pnl_df["cid"] = pnl_names[i]
        comp_dfs.append(pnl_df)

    fig = timelines(
        pd.concat(comp_dfs, axis=0),
        title=title,
        cumsum=cumsum,
        return_fig=True,
        cid_labels=pnl_names,
        ax_hline=0.0,
        title_fontsize=title_fontsize,
        **kwargs,
    )
    if return_fig:
        return fig
    plt.show()


def implied_leverage_plot(
    npos_dfs: Union[pd.DataFrame, List[pd.DataFrame]],
    labels: Union[str, List[str]],
    aum: Number,
    figsize: Tuple[float, float] = (13, 6),
    alpha: float = 0.8,
    title: str = "Implied leverage",
    title_fontsize: int = 14,
    xlabel: str = "",
    ylabel: str = "Leverage",
    label_fontsize: int = 10,
    baseline: bool = False,
):
    if isinstance(npos_dfs, pd.DataFrame):
        npos_dfs = [npos_dfs]
    if isinstance(labels, str):
        labels = [labels]

    _, ax = plt.subplots(figsize=figsize)

    for label, npos_df in zip(labels, npos_dfs):
        total_pos = npos_df["value"].abs().groupby(npos_df["real_date"]).sum()

        implied_leverage = total_pos / aum

        sns.lineplot(implied_leverage, ax=ax, alpha=alpha, label=label)

    ax.set_title(title, fontsize=title_fontsize)
    ax.set_xlabel(xlabel, fontsize=label_fontsize)
    ax.set_ylabel(ylabel, fontsize=label_fontsize)

    if baseline:
        ax.axhline(y=1, color="red", linestyle="--", linewidth=1.5)



def _prepare_pnl_df(
    pnl_df: pd.DataFrame,
    portfolio_names: Optional[List[str]] = None,
    aum: Optional[Number] = None,
    cumsum: bool = True,
) -> pd.DataFrame:
    """
    Filter a long-format PnL frame to the portfolios of interest, cumulate it
    and put it on the desired scale.
    """
    df = reduce_df(pnl_df, cids=portfolio_names)
    df = df.copy().sort_values(["cid", "real_date"])

    if cumsum:
        df["value"] = df.groupby("cid")["value"].cumsum()

    if aum is not None:
        df["value"] = 100 * df["value"] / aum

    return df


def _plot_pnl_panel(
    ax: plt.Axes,
    pnl_df: pd.DataFrame,
    hue_order: List[str],
    panel_title: str = "",
    x_label: str = "",
    y_label: str = "",
    line_width: int = 1,
    label_fontsize: int = 12,
    tick_fontsize: int = 12,
    panel_title_fontsize: int = 14,
    background_vals: Optional[pd.Series] = None,
    cmap: Optional[mpl.colors.Colormap] = None,
    norm: Optional[mpl.colors.Normalize] = None,
) -> Tuple[List[Any], List[str]]:
    """
    Draw a single cumulative-PnL panel on `ax` and return the legend
    handles/labels so the caller can draw one legend for the whole figure.

    `hue_order`, `cmap` and `norm` are passed in rather than derived here so
    that colours and background shading are identical across panels.
    """
    sns.lineplot(
        data=pnl_df,
        x="real_date",
        y="value",
        hue="cid",
        hue_order=hue_order,
        estimator=None,
        lw=line_width,
        ax=ax,
    )

    ax.axhline(y=0, color="black", linestyle="--", lw=1)
    ax.set_title(panel_title, fontsize=panel_title_fontsize)
    ax.set_xlabel(x_label, fontsize=label_fontsize)
    ax.set_ylabel(y_label, fontsize=label_fontsize)
    ax.tick_params(axis="both", labelsize=tick_fontsize)

    # shade the background by signal strength
    if background_vals is not None:
        vals = background_vals.sort_index()
        for i in range(vals.shape[0] - 1):
            ax.axvspan(
                vals.index[i],
                vals.index[i + 1],
                color=cmap(norm(vals.iloc[i])),
                alpha=0.2,
                zorder=0,
            )

    # hand the legend back to the caller and drop the per-axes one
    handles, labels = ax.get_legend_handles_labels()
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()

    return handles, labels


def proxy_pnl_plot(
    pnl_df: pd.DataFrame,
    pnle_df: Optional[pd.DataFrame] = None,
    portfolio_names: Optional[List[str]] = None,
    portfolio_labels: Optional[List[str]] = None,
    background_vals: Optional[pd.Series] = None,
    aum: Optional[Number] = None,
    y_label: str = "",
    x_label: str = "",
    title: str = "",
    panel_titles: Sequence[str] = ("Incl. costs", "Excl. costs"),
    legend_title: str = "Portfolio",
    title_fontsize: int = 20,
    panel_title_fontsize: int = 14,
    legend_fontsize: int = 10,
    label_fontsize: int = 12,
    tick_fontsize: int = 12,
    cumsum: bool = True,
    line_width: int = 1,
    figsize: Optional[Tuple[float, float]] = None,
    share_y: bool = True,
    show: bool = True,
) -> Tuple[plt.Figure, Any]:
    """
    Plot cumulative proxy PnL, optionally as two panels side by side.
    """
    frames = [
        _prepare_pnl_df(pnl_df, portfolio_names, aum, cumsum)
    ]
    if pnle_df is not None:
        frames.append(
            _prepare_pnl_df(pnle_df, portfolio_names, aum, cumsum)
        )

    n_panels = len(frames)

    # one colour per portfolio, identical across panels
    if portfolio_names is not None:
        hue_order = list(portfolio_names)
    else:
        hue_order = sorted(set().union(*(set(f["cid"].unique()) for f in frames)))

    if portfolio_labels is not None and len(portfolio_labels) != len(hue_order):
        raise ValueError(
            f"`portfolio_labels` has {len(portfolio_labels)} entries but there "
            f"are {len(hue_order)} portfolios to plot."
        )

    if figsize is None:
        figsize = (12, 7) if n_panels == 1 else (16, 7)

    cmap = norm = None
    if background_vals is not None:
        cmap = plt.get_cmap("viridis")
        norm = mpl.colors.Normalize(
            vmin=background_vals.min(), vmax=background_vals.max()
        )

    sns.set_theme(style="whitegrid", palette="colorblind")

    fig, axes = plt.subplots(
        nrows=1,
        ncols=n_panels,
        figsize=figsize,
        sharex=True,
        sharey=share_y,
        squeeze=False,
        constrained_layout=True,
    )
    axes = axes.flatten()

    handles: List[Any] = []
    labels: List[str] = []
    for i, (ax, frame) in enumerate(zip(axes, frames)):
        panel_title = ""
        if n_panels > 1 and panel_titles is not None and i < len(panel_titles):
            panel_title = panel_titles[i]

        handles, labels = _plot_pnl_panel(
            ax=ax,
            pnl_df=frame,
            hue_order=hue_order,
            panel_title=panel_title,
            x_label=x_label,
            # only the leftmost panel needs the y-label when the axis is shared
            y_label=y_label if (i == 0 or not share_y) else "",
            line_width=line_width,
            label_fontsize=label_fontsize,
            tick_fontsize=tick_fontsize,
            panel_title_fontsize=panel_title_fontsize,
            background_vals=background_vals,
            cmap=cmap,
            norm=norm,
        )

    # one legend and one colourbar for the whole figure
    if portfolio_labels is not None:
        labels = list(portfolio_labels)
    axes[0].legend(
        handles=handles,
        labels=labels,
        title=legend_title,
        title_fontsize=legend_fontsize,
        fontsize=legend_fontsize,
    )

    if background_vals is not None:
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=list(axes))
        cbar.set_label("Signal strength")

    if title:
        fig.suptitle(title, fontsize=title_fontsize)

    if show:
        plt.show()

    return fig, (axes[0] if n_panels == 1 else axes)