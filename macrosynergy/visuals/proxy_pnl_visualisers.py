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

        for text in ax.get_legend().get_texts():
            if text.get_text() in {"Freq", "Method", "Effective lookback"}:
                text.set_fontweight("bold")

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

        piv_sig = sig_df.pivot(
            index="real_date", columns=["cid", "xcat"], values="value"
        )
        x_vals = piv_sig.abs().sum(axis=1)  # signals

        for i in range(len(pos_dfs)):
            piv_pos = pos_dfs[i].pivot(
                index="real_date", columns=["cid", "xcat"], values="value"
            )
            y_vals = piv_pos.abs().sum(axis=1)  # positions

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


def compare_proxy_pnls(
    pnl_dfs: List[pd.DataFrame],
    pnle_dfs: List[pd.DataFrame],
    portfolio_names: List[str],
    pnl_names: Optional[List[str]] = None,
    title: str = "PnL with and without costs",
    ylabel: str = "USD mn",
    title_fontsize: int = 18,
    incl_costs_label: str = "Incl. Costs",
    excl_costs_label: str = "Excl. Costs",
    sharey: bool = False,
    ncols: Optional[int] = None,
    cumsum: bool = True,
    line_width: float = 1,
    figsize: Tuple[float, float] = (12, 6),
):
    assert len(pnl_dfs) == len(pnle_dfs) == len(portfolio_names)

    if pnl_names is None:
        pnl_names = portfolio_names

    assert len(pnl_names) == len(portfolio_names)

    pnlcount = len(pnl_dfs)

    if ncols is None:
        ncols = min(3, pnlcount)

    nrows = (pnlcount + ncols - 1) // ncols

    sns.set_theme(style="whitegrid", palette="colorblind")

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
        squeeze=False,
        sharex=False,
        sharey=sharey,
    )

    axes_flat = axes.ravel()

    legend_handles = None
    legend_labels = None

    for i, (pnl_df, pnle_df, portfolio_name, pnl_name) in enumerate(
        zip(pnl_dfs, pnle_dfs, portfolio_names, pnl_names)
    ):
        ax = axes_flat[i]

        pnl = pnl_df.copy()
        pnle = pnle_df.copy()

        pnl["cost_type"] = incl_costs_label
        pnle["cost_type"] = excl_costs_label

        data = pd.concat([pnl, pnle], ignore_index=True)
        data = reduce_df(data, cids=[portfolio_name])
        data = data.sort_values(["cid", "cost_type", "real_date"])

        if cumsum:
            data["plot_value"] = data.groupby(["cid", "cost_type"])["value"].cumsum()
        else:
            data["plot_value"] = data["value"]

        sns.lineplot(
            data=data,
            x="real_date",
            y="plot_value",
            hue="cost_type",
            estimator=None,
            lw=line_width,
            ax=ax,
        )

        ax.set_title(pnl_name)
        ax.set_xlabel("")

        # Only show y-axis label on leftmost subplot in each row
        if i % ncols == 0:
            ax.set_ylabel(ylabel)
        else:
            ax.set_ylabel("")

        ax.axhline(
            y=0,
            color="black",
            linestyle="--",
            lw=1,
        )

        # Capture legend entries once
        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()

        # Remove subplot-level legend
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()

    # Remove empty subplots
    for ax in axes_flat[pnlcount:]:
        ax.remove()

    fig.suptitle(
        title,
        fontsize=title_fontsize,
    )

    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=2,
        frameon=False,
    )

    return fig, axes


def implied_leverage_plot(
    npos_dfs: Union[pd.DataFrame, List[pd.DataFrame]],
    labels: Union[str, List[str]],
    aum: Number,
    figsize: Tuple[float, float] = (13, 6),
    alpha: float = 0.9,
    linewidth: float = 2.0,
    title: str = "Implied leverage",
    title_fontsize: int = 14,
    xlabel: str = "",
    ylabel: str = "Leverage",
    label_fontsize: int = 11,
    baseline: bool = False,
    drop_leading_zeros: bool = True,
):
    if isinstance(npos_dfs, pd.DataFrame):
        npos_dfs = [npos_dfs]

    if isinstance(labels, str):
        labels = [labels]

    fig, ax = plt.subplots(figsize=figsize)

    for label, npos_df in zip(labels, npos_dfs):
        total_pos = (
            npos_df["value"].abs().groupby(npos_df["real_date"]).sum().sort_index()
        )

        implied_leverage = total_pos / aum

        if drop_leading_zeros:
            non_zero = implied_leverage.ne(0)

            if non_zero.any():
                implied_leverage = implied_leverage.loc[non_zero.idxmax() :]

        ax.plot(
            implied_leverage.index,
            implied_leverage.values,
            label=label,
            alpha=alpha,
            linewidth=linewidth,
        )

    if baseline:
        ax.axhline(
            y=1,
            linestyle="--",
            linewidth=1.25,
            color="0.4",
            alpha=0.8,
            label="1x leverage",
        )

    ax.set_title(
        title,
        fontsize=title_fontsize,
        pad=12,
    )
    ax.set_xlabel(xlabel, fontsize=label_fontsize)
    ax.set_ylabel(ylabel, fontsize=label_fontsize)

    # Cleaner chart appearance
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.grid(
        axis="y",
        linestyle="--",
        linewidth=0.7,
        alpha=0.3,
    )
    ax.grid(axis="x", visible=False)

    ax.tick_params(
        axis="both",
        labelsize=label_fontsize - 1,
        length=0,
    )

    ax.legend(
        frameon=False,
        fontsize=label_fontsize,
        loc="best",
    )

    fig.tight_layout()

    return fig, ax


def realized_vol_plot(
    pnl_dfs: Union[pd.DataFrame, List[pd.DataFrame]],
    labels: Union[str, List[str]],
    portfolio_names: Union[str, List[str]],
    aum: Number,
    lback: int = 252,
    annualization_factor: int = 252,
    vol_target: Optional[float] = None,
    figsize: Tuple[float, float] = (13, 5),
    alpha: float = 0.9,
    linewidth: float = 1.5,
    title: str = "Rolling realized volatility of strategy PnL",
    title_fontsize: int = 14,
    xlabel: str = "",
    ylabel: str = "Annualized volatility, % of AUM",
    label_fontsize: int = 11,
    show_mean: bool = True,
):
    if isinstance(pnl_dfs, pd.DataFrame):
        pnl_dfs = [pnl_dfs]

    if isinstance(labels, str):
        labels = [labels]

    if isinstance(portfolio_names, str):
        portfolio_names = [portfolio_names]

    fig, ax = plt.subplots(figsize=figsize)

    for label, pnl_df in zip(labels, pnl_dfs):
        pnl_df = reduce_df(pnl_df, cids=portfolio_names)
        pnl = pnl_df.set_index("real_date")["value"].sort_index()

        realized_vol = (100 * pnl / aum).rolling(lback).std() * np.sqrt(
            annualization_factor
        )

        if show_mean:
            plot_label = f"{label} (mean {realized_vol.mean():.1f}%)"
        else:
            plot_label = label

        ax.plot(
            realized_vol.index,
            realized_vol.values,
            label=plot_label,
            linewidth=linewidth,
            alpha=alpha,
        )

    if vol_target is not None:
        ax.axhline(
            vol_target,
            linestyle="--",
            linewidth=1.25,
            color="0.3",
            alpha=0.9,
            label=f"{vol_target}% target",
        )

    ax.set_title(
        title,
        fontsize=title_fontsize,
        pad=12,
    )
    ax.set_xlabel(xlabel, fontsize=label_fontsize)
    ax.set_ylabel(ylabel, fontsize=label_fontsize)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.grid(
        axis="y",
        linestyle="--",
        linewidth=0.7,
        alpha=0.3,
    )
    ax.grid(axis="x", visible=False)

    ax.tick_params(
        axis="both",
        labelsize=label_fontsize - 1,
        length=0,
    )

    ax.legend(
        frameon=False,
        fontsize=label_fontsize,
        title=None,
    )

    fig.tight_layout()

    return fig, ax


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
    frames = [_prepare_pnl_df(pnl_df, portfolio_names, aum, cumsum)]
    if pnle_df is not None:
        frames.append(_prepare_pnl_df(pnle_df, portfolio_names, aum, cumsum))

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


def vol_target_scaling_factor_plot(
    vol_df: pd.DataFrame,
    vol_target: int,
    vol_xcat: str,
    linewidth: float = 1,
    x_label: str = "",
    y_label_pvol: str = "Annualized volatility (%)",
    y_label_scale: str = "Scale factor",
    title: str = "",
    title_fontsize: int = 15,
    figsize: Tuple[float, float] = (13, 6),
):
    df = reduce_df(vol_df, xcats=[vol_xcat]).sort_values(by="real_date")

    x_vals = df["real_date"].values
    y_vals_pvol = df["value"].values
    y_vals_scale = vol_target / y_vals_pvol

    fig, ax = plt.subplots(1, 2, figsize=figsize)

    # Portfolio volatility
    sns.lineplot(
        x=x_vals,
        y=y_vals_pvol,
        ax=ax[0],
        linewidth=linewidth,
        alpha=0.8,
        label=r"$\sqrt{s_{t}^{\top} \Sigma s_{t}}$",
    )
    ax[0].set_title("Portfolio volatility prior to volatility targeting")
    ax[0].set_xlabel(x_label)
    ax[0].set_ylabel(y_label_pvol)
    ax[0].legend()
    ax[0].grid(alpha=0.3)

    mean_vol = np.nanmean(y_vals_pvol)
    median_vol = np.nanmedian(y_vals_pvol)

    ax[0].plot([], [], " ", label=f"Mean: {mean_vol:.1f}")
    ax[0].plot([], [], " ", label=f"Median: {median_vol:.1f}")
    ax[0].legend()

    # Volatility scaling factor
    sns.lineplot(
        x=x_vals,
        y=y_vals_scale,
        ax=ax[1],
        linewidth=linewidth,
        alpha=0.8,
        label=rf"$\frac{{{vol_target}}}{{\sqrt{{s_{{t}}^{{\top}} \Sigma s_{{t}}}}}}$",
    )
    ax[1].set_title("Scaling factor needed to achieve volatility target")
    ax[1].set_xlabel(x_label)
    ax[1].set_ylabel(y_label_scale)
    ax[1].legend()
    ax[1].grid(alpha=0.3)

    mean_scale = np.nanmean(y_vals_scale)
    median_scale = np.nanmedian(y_vals_scale)

    ax[1].plot([], [], " ", label=f"Mean: {mean_scale:.1f}")
    ax[1].plot([], [], " ", label=f"Median: {median_scale:.1f}")
    ax[1].legend()

    fig.suptitle(title, fontsize=title_fontsize)

    fig.autofmt_xdate()
    fig.tight_layout()

    return fig, ax


def scaling_factor_error_impact_plot(
    bias: np.ndarray,
    var: np.ndarray,
    costs: np.ndarray,
    pnl_vols: np.ndarray,
    title: str = "Impact of scaling factor estimator bias and variance on cost and PnL vol",
    title_fontsize: int = 15,
    subtitles: List[str] = None,
    xlabel: str = "Bias of the scaling factor",
    ylabel: str = "Standard deviation of the scaling factor",
    figsize: Tuple[float, float] = (16, 5),
    point_size: int = 170,
):
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    if subtitles is None:
        subtitles = ["Transaction costs (USDmn)", "Realized PnL volatility"]

    metrics = [(costs, subtitles[0]), (pnl_vols, subtitles[1])]

    for i, (ax, (values, colorbar_label)) in enumerate(zip(axes, metrics)):
        points = ax.scatter(
            bias,
            var,
            c=values,
            cmap="rocket_r",
            s=point_size,
            edgecolor="white",
            linewidth=0.8,
        )

        fig.colorbar(
            points,
            ax=ax,
            label=colorbar_label,
        )

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel if i == 0 else "")
        ax.set_title(colorbar_label)
        ax.grid(alpha=0.25)

    fig.suptitle(title, fontsize=title_fontsize)

    fig.tight_layout()
    plt.show()


def plot_metrics_before_and_after_costs(
    metric_df: pd.DataFrame,
    metrics: List[str],
    rules: List[str],
    signal_labels: List[str],
    title: str = "",
    title_fontsize: int = 15,
    figsize: Tuple[float, float] = (15, 5.5),
):
    dfp = metric_df.pivot_table(
        index=["metric", "rule_label", "signal_label"],
        columns="costs",
        values="value",
    )

    fig, axes = plt.subplots(
        1,
        len(metrics),
        figsize=figsize,
        squeeze=False,
    )
    axes = axes.ravel()

    xpos = np.arange(len(rules))
    width = 0.8 / len(signal_labels)
    colors = sns.color_palette("colorblind", len(signal_labels))

    for ax, metric in zip(axes, metrics):
        for j, signal_label in enumerate(signal_labels):
            offset = (j - (len(signal_labels) - 1) / 2) * width

            keys = [(metric, rule, signal_label) for rule in rules]

            gross = [dfp.loc[key, "Gross"] for key in keys]
            net = [dfp.loc[key, "Net"] for key in keys]

            ax.bar(
                xpos + offset,
                gross,
                width,
                facecolor="none",
                edgecolor=colors[j],
                linewidth=1.2,
                linestyle="--",
            )

            ax.bar(
                xpos + offset,
                net,
                width,
                color=colors[j],
                label=signal_label if metric == metrics[0] else None,
            )

        ax.set_title(metric, fontsize=12)

        ax.set_xticks(xpos)
        ax.set_xticklabels(
            rules,
            fontsize=9,
        )

        ax.axhline(
            0,
            color="black",
            linewidth=0.8,
        )

        ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.3)
        ax.grid(axis="x", visible=False)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()

    handles += [
        plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor="grey",
        ),
        plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor="none",
            edgecolor="grey",
            linestyle="--",
        ),
    ]
    labels += [
        "Net of costs",
        "Gross",
    ]

    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(signal_labels) + 2,
        frameon=False,
        bbox_to_anchor=(0.5, -0.03),
    )

    if title:
        fig.suptitle(title, fontsize=title_fontsize)

    fig.tight_layout()

    return fig, axes


def plot_costs_by_type(
    cost_dfs: List[pd.DataFrame],
    labels: List[str],
    rollcost_suffix: str = "TCOST_ROLLCOST",
    bidoffer_suffix: str = "TCOST_BIDOFFER",
    title: str = "",
    title_fontsize: int = 16,
    figsize: Tuple[float, float] = (11, 5.5),
    label_rotation: int = 90,
    sort_by_label: bool = False,
):
    if sort_by_label:
        sorted_idx = np.argsort(labels)
        labels = [labels[i] for i in sorted_idx]
        cost_dfs = [cost_dfs[i] for i in sorted_idx]

    # prepare data
    rows = []
    for label, cost_df in zip(labels, cost_dfs):
        bidoffer_cost = cost_df.loc[
            cost_df["xcat"].str.endswith(bidoffer_suffix), "value"
        ].sum()

        roll_cost = cost_df.loc[
            cost_df["xcat"].str.endswith(rollcost_suffix), "value"
        ].sum()

        rows.append(
            {
                "label": label,
                "Bid-offer": bidoffer_cost,
                "Roll": roll_cost,
            }
        )

    data = pd.DataFrame(rows).set_index("label")

    # plot data
    fig, ax = plt.subplots(figsize=figsize)
    data.plot(
        kind="bar",
        stacked=True,
        ax=ax,
        width=0.65,
        color=["#4C78A8", "#F2A541"],
        edgecolor="white",
        linewidth=0.7,
    )

    ax.set_title(title, fontsize=title_fontsize, pad=18)

    # format axes
    ax.set_xlabel("")
    ax.set_ylabel("Cumulative cost (USD mn)", fontsize=11)

    ax.tick_params(
        axis="x",
        labelsize=9,
        rotation=0,
        length=0,
        pad=8,
        labelrotation=label_rotation,
    )
    ax.tick_params(axis="y", labelsize=9)

    # format grid
    ax.set_axisbelow(True)
    ax.grid(
        axis="y",
        linestyle="--",
        linewidth=0.7,
        alpha=0.25,
    )
    ax.grid(axis="x", visible=False)

    # remove spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_alpha(0.7)
    ax.spines["bottom"].set_alpha(0.7)

    # format legend
    ax.legend(
        title=None,
        frameon=False,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.03),
    )

    # add total above each stacked bar
    totals = data.sum(axis=1)

    for i, total in enumerate(totals):
        ax.annotate(
            f"{total:,.1f}",
            xy=(i, total),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="semibold",
        )

    # Leave a little room for labels
    ax.margins(y=0.12)

    fig.tight_layout()

    return fig, ax
