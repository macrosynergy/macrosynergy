from numbers import Number
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter

from macrosynergy.management import reduce_df

FREQ_TO_DAYS_MAP = {"D": 1, "W": 5, "M": 21, "Q": 63}


def _ordered_levels(
    values: List[Any],
    order: Optional[List[Any]],
    argname: str,
) -> List[Any]:
    """
    Return the unique entries of values, either in order of first appearance or
    in the order requested. A requested order may be a subset, in which case the
    remaining levels are dropped.

    Parameters
    ----------
    values : List[Any]
        Level of each entry.
    order : Optional[List[Any]]
        Requested order of the levels. Defaults to None, in which case the order
        of first appearance is used.
    argname : str
        Name of the argument that supplied order, used in the error message.

    Returns
    -------
    List[Any]
        The ordered levels.
    """
    levels = list(dict.fromkeys(values))

    if order is None:
        return levels

    unknown = [level for level in order if level not in levels]
    if unknown:
        raise ValueError(f"`{argname}` holds entries not present in `evals`: {unknown}")

    return list(order)


def _resolve_cost_columns(
    df: pd.DataFrame,
    key: Any,
    net_col: Optional[str] = None,
    gross_col: Optional[str] = None,
    need_gross: bool = True,
    pnl_name: str = "PNL",
) -> Tuple[str, Optional[str]]:
    """
    Identify the net and gross columns of a single evaluate_pnl output.

    The columns of an evaluate_pnl output are ordered alphabetically by the
    underlying category, so the two series cannot be identified by position.
    Each of them is resolved in turn by

    1. the explicit net_col or gross_col argument,
    2. the single column ending in pnl_name for the net and in pnl_name with an
       "e" appended for the gross, which is the convention proxy_pnl_calc
       writes,
    3. the single column whose name contains "net" or "incl" for the net and
       "gross" or "excl" for the gross, for outputs whose columns have been
       renamed through the label_dict of evaluate_pnl.

    Parameters
    ----------
    df : pd.DataFrame
        Output of evaluate_pnl, with metrics as its index and one column per
        PnL series.
    key : Any
        Key of the data frame in evals, used in the error messages.
    net_col : Optional[str]
        Name of the column holding the PnL net of costs. Defaults to None, in
        which case it is resolved as described above.
    gross_col : Optional[str]
        Name of the column holding the PnL gross of costs. Defaults to None, in
        which case it is resolved as described above.
    need_gross : bool
        Whether the gross column is required. Defaults to True. When False the
        gross column is not looked for, so that an output holding only a PnL
        net of costs is accepted.
    pnl_name : str
        Name given to the PnL by proxy_pnl_calc, which ends the category of the
        PnL net of costs and, with an "e" appended, that of the PnL gross of
        costs. Defaults to "PNL".

    Returns
    -------
    Tuple[str, Optional[str]]
        Names of the net and the gross column, the latter None when need_gross
        is False.
    """
    columns = list(df.columns)
    resolved = {}

    roles = [("net", net_col, pnl_name, ("net", "incl"))]
    if need_gross:
        roles.append(("gross", gross_col, pnl_name + "e", ("gross", "excl")))

    for role, explicit, suffix, tokens in roles:
        if explicit is not None:
            if explicit not in columns:
                raise KeyError(
                    f"Column '{explicit}' is not in the `evals` output at {key!r}, "
                    f"which holds {columns}."
                )
            resolved[role] = explicit
            continue

        # the categories written by proxy_pnl_calc end in the name of the PnL
        matches = [col for col in columns if str(col).endswith(suffix)]

        # an output renamed through label_dict carries the display labels
        if len(matches) != 1:
            matches = [
                col for col in columns if any(t in str(col).lower() for t in tokens)
            ]

        if len(matches) != 1:
            quoted = " or ".join(f"'{token}'" for token in tokens)
            raise ValueError(
                f"Could not identify the {role} column of the `evals` output at "
                f"{key!r} from its columns {columns}: no single column ends in "
                f"'{suffix}' or holds {quoted}. Pass `{role}_col` explicitly."
            )

        resolved[role] = matches[0]

    if need_gross and resolved["net"] == resolved["gross"]:
        raise ValueError(
            f"The net and gross columns of the `evals` output at {key!r} both "
            f"resolved to '{resolved['net']}'."
        )

    return resolved["net"], resolved.get("gross")


def _cost_columns(
    evals: Dict[Any, pd.DataFrame],
    net_col: Optional[str] = None,
    gross_col: Optional[str] = None,
    need_gross: bool = True,
    pnl_name: str = "PNL",
) -> Dict[Any, Tuple[str, Optional[str]]]:
    """
    Identify the net and gross columns of every evaluate_pnl output in evals.

    Each output is resolved on its own, since configurations may differ in the
    portfolio and strategy names that make up their categories. The resolution
    order is documented in _resolve_cost_columns.

    Parameters
    ----------
    evals : Dict[Any, pd.DataFrame]
        Outputs of evaluate_pnl, keyed as the calling plot function requires.
    net_col : Optional[str]
        Name of the column holding the PnL net of costs, applied to every
        output. Defaults to None, in which case it is resolved per output.
    gross_col : Optional[str]
        Name of the column holding the PnL gross of costs, applied to every
        output. Defaults to None, in which case it is resolved per output.
    need_gross : bool
        Whether the gross column is required. Defaults to True.
    pnl_name : str
        Name given to the PnL by proxy_pnl_calc. Defaults to "PNL".

    Returns
    -------
    Dict[Any, Tuple[str, Optional[str]]]
        The net and gross column of each output, keyed as evals.
    """
    return {
        key: _resolve_cost_columns(
            df, key, net_col, gross_col, need_gross=need_gross, pnl_name=pnl_name
        )
        for key, df in evals.items()
    }


def _sensitivity_frame(
    evals: Dict[Tuple[Any, ...], pd.DataFrame],
    group_order: Optional[List[Any]],
    series_order: Optional[List[Any]],
) -> Tuple[Dict[Tuple[Any, Any], List[Tuple[float, Any]]], List[Any], List[Any], bool]:
    """
    Group the keys of a sweep by the line they belong to.

    The keys of evals are read as (group, series, x) triples or, where no group
    level is present, as (series, x) pairs. They are collected into one entry
    per line, holding the x-value and the key of every point on that line,
    sorted by x.

    Parameters
    ----------
    evals : Dict[Tuple[Any, ...], pd.DataFrame]
        Outputs of evaluate_pnl, keyed by (group, series, x) or (series, x).
    group_order : Optional[List[Any]]
        Requested order of the groups, or None for the order of first
        appearance.
    series_order : Optional[List[Any]]
        Requested order of the series, or None for the order of first
        appearance.

    Returns
    -------
    Tuple[Dict[Tuple[Any, Any], List[Tuple[float, Any]]], List[Any], List[Any], bool]
        The points of each (group, series) line, the ordered groups, the
        ordered series, and whether the keys carry a group level.
    """
    keys = list(evals)

    arities = {len(key) if isinstance(key, tuple) else 0 for key in keys}

    if arities - {2, 3}:
        raise ValueError(
            "The keys of `evals` must be (group, series, x) triples or "
            f"(series, x) pairs, got lengths {sorted(arities)}."
        )

    if len(arities) > 1:
        raise ValueError("The keys of `evals` must all be of the same length.")

    grouped = arities == {3}

    key_parts = {}
    for key in keys:
        group, series, x = key if grouped else (None, *key)

        try:
            x = float(x)
        except (TypeError, ValueError):
            raise TypeError(
                "The last element of every key of `evals` is the x-value of "
                f"that point and must be numeric, got {x!r}."
            )

        key_parts[key] = (group, series, x)

    groups = _ordered_levels(
        [key_parts[key][0] for key in keys],
        group_order if grouped else None,
        "group_order",
    )
    series = _ordered_levels(
        [key_parts[key][1] for key in keys], series_order, "series_order"
    )

    lines = {(group, s): [] for group in groups for s in series}

    for key, (group, s, x) in key_parts.items():
        if (group, s) in lines:  # a subset order drops the remaining levels
            lines[(group, s)].append((x, key))

    for points in lines.values():
        points.sort(key=lambda point: point[0])

    return lines, groups, series, grouped


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
    df : pd.DataFrame
        Transaction cost data in long format. Must contain cid, xcat,
        and value columns.
    title : str
        Title of the heatmap. Defaults to an empty string.
    xlabel : str
        Label for the x-axis. Defaults to an empty string.
    ylabel : str
        Label for the y-axis. Defaults to an empty string.
    tcost_name : str
        Suffix identifying transaction cost categories in xcat. Only
        categories whose name ends with this string are included. Defaults to
        "TCOST".
    figsize : Tuple[float, float]
        Size of the figure. Defaults to (10, 6).
    exclude_cids : Tuple[str, ...]
        Cross-sections to exclude from the heatmap. Defaults to ("GLB",).
    label_dict : Dict[str, str]
        Optional mapping used to rename categories for display. Defaults to
        None, in which case the original category names are used.
    title_fontsize : int
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
    x_values : np.ndarray
        Values for the x-axis, shared across all series. For example, a range
        of volatility targets.
    y_values : np.ndarray
        Array of shape (n, len(x_values)) holding the sensitivity-analysis
        results, where n is the number of series to plot.
    labels : List[str]
        Labels for the plotted series, one per row of y_values.
    title : str
        Title of the plot. Defaults to an empty string.
    xlabel : str
        Label for the x-axis. Defaults to an empty string.
    ylabel : str
        Label for the y-axis. Defaults to an empty string.
    figsize : Tuple[float, float]
        Size of the figure, used only when ax is not
        provided. Defaults to (10, 6).
    ax : plt.Axes
        Optional existing axes to draw on. Defaults to None, in which case a
        new figure and axes are created.
    title_fontsize : int
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
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    title_fontsize: int = 14,
    figsize: Tuple[float, float] = (10, 6),
) -> None:
    """
    Scatter-plot outcomes of alternative covariance estimation configurations.
    Each point corresponds to one entry in configs. The marker shape encodes
    the lookback method, the colour the estimation frequencies, and the marker
    size the effective lookback expressed in business days.

    Notes
    -----
    For exponential moving averages the effective lookback is derived from the
    half-life as (1 + lam) / (1 - lam), with lam = 2 ** (-1 / half_life). Where
    several estimation frequencies are combined, the effective lookback is
    their weighted average.

    Parameters
    ----------
    x_vals : np.ndarray
        Values for the x-axis, one per configuration.
    y_vals : np.ndarray
        Values for the y-axis, one per configuration.
    configs : List[dict]
        Covariance estimation configurations, one per plotted point
    title : str
        Title of the plot
    xlabel : str
        Label for the x-axis
    ylabel : str
        Label for the y-axis
    title_fontsize : int
        Font size of the title. Defaults to 14.
    figsize : Tuple[float, float]
        Size of the figure. Defaults to (10, 6).
    """
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
    """
    Scatterplot showing aggregate notional positions against aggregate
    signal strength

    Parameters
    ----------
    pos_dfs : List[pd.DataFrame]
        Notional position data in long format, one data frame per subplot.
    sig_df : pd.DataFrame
        Signal data in long format, shared across all subplots
    df_labels : List[str]
        Subplot titles, one per entry in pos_dfs
    title : str
        Overall figure title
    xlabel : str
        Shared label for the x-axis
    ylabel : str
        Shared label for the y-axis
    title_fontsize : int
        Font size of the figure title. Defaults to 14.
    sharex : bool
        Whether the subplots share the x-axis. Defaults to True.
    sharey : bool
        Whether the subplots share the y-axis. Defaults to False.
    n_cols : int
        Number of subplot columns. Defaults to 3.
    figsize : Tuple[float, float]
        Size of the figure. Defaults to (15, 5).
    point_size : float
        Size of the scatter points. Defaults to 5.

    Returns
    -------
    Tuple[plt.Figure, Any]
        The figure and the two-dimensional array of axes.
    """
    with sns.axes_style("whitegrid"), sns.plotting_context("notebook"):
        fig, axes = plt.subplots(
            nrows=(len(pos_dfs) + n_cols - 1) // n_cols,
            ncols=n_cols,
            figsize=figsize,
            sharex=sharex,
            sharey=sharey,
            squeeze=False,
        )

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
    """
    Compare proxy PnLs before and after transaction costs. Each portfolio is drawn
    in its own subplot, with one line for the PnL including costs and one for the
    PnL excluding costs.

    Parameters
    ----------
    pnl_dfs : List[pd.DataFrame]
        PnL data including transaction costs, in long format, one data frame
        per portfolio
    pnle_dfs : List[pd.DataFrame]
        PnL data excluding transaction costs, in the same format and order as
        pnl_dfs.
    portfolio_names : List[str]
        Cross section identifiers of the portfolios, used to select the rows
        of each PnL data frame.
    pnl_names : Optional[List[str]]
        Subplot titles, one per portfolio
    title : str
        Overall figure title.
    ylabel : str
        Label for the y-axis, shown on the leftmost subplot of each row.
    title_fontsize : int
        Font size of the figure title. Defaults to 18.
    incl_costs_label : str
        Legend label for the PnL including costs. Defaults to "Incl. Costs".
    excl_costs_label : str
        Legend label for the PnL excluding costs. Defaults to "Excl. Costs".
    sharey : bool
        Whether the subplots share the y-axis. Defaults to False.
    ncols : Optional[int]
        Number of subplot columns. Defaults to None, in which case at most
        three columns are used.
    cumsum : bool
        Whether to plot cumulative rather than period PnL. Defaults to True.
    line_width : float
        Width of the plotted lines. Defaults to 1.
    figsize : Tuple[float, float]
        Size of the figure. Defaults to (12, 6).

    Returns
    -------
    Tuple[plt.Figure, np.ndarray]
        The figure and the two-dimensional array of axes.
    """
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
    """
    Plot the leverage implied by notional positions over time. For each
    date the absolute notional positions are summed across all
    contracts and divided by the assets under management, giving the gross
    exposure as a multiple of AUM. One line is drawn per position dataframe.

    Parameters
    ----------
    npos_dfs : Union[pd.DataFrame, List[pd.DataFrame]]
        Notional position data in long format, either a single data frame or
        one per line to plot
    labels : Union[str, List[str]]
        Legend labels, one per entry in npos_dfs.
    aum : Number
        Assets under management, in the same units as the notional positions.
    figsize : Tuple[float, float]
        Size of the figure. Defaults to (13, 6).
    alpha : float
        Opacity of the plotted lines. Defaults to 0.9.
    linewidth : float
        Width of the plotted lines. Defaults to 2.0.
    title : str
        Title of the plot. Defaults to "Implied leverage".
    title_fontsize : int
        Font size of the title. Defaults to 14.
    xlabel : str
        Label for the x-axis
    ylabel : str
        Label for the y-axis. Defaults to "Leverage".
    label_fontsize : int
        Font size of the axis labels and the legend. Defaults to 11.
    baseline : bool
        Whether to draw a horizontal reference line at a leverage of one.
        Defaults to False.
    drop_leading_zeros : bool
        Whether to start each line at its first non-zero leverage, dropping
        the period before any position is held. Defaults to True.

    Returns
    -------
    Tuple[plt.Figure, plt.Axes]
        The figure and the axes containing the line plot.
    """
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
    ylabel: str = "Annualized volatility (%)",
    label_fontsize: int = 11,
    show_mean: bool = True,
):
    """
    Plot the rolling realized volatility of strategy PnL as a share of AUM.
    PnL is expressed in percent of assets under management, and its rolling
    standard deviation is annualized with the square root of the annualization
    factor. One line is drawn per PnL data frame.

    Parameters
    ----------
    pnl_dfs : Union[pd.DataFrame, List[pd.DataFrame]]
        PnL data in long format, either a single data frame or one per line to
        plot
    labels : Union[str, List[str]]
        Legend labels, one per entry in pnl_dfs.
    portfolio_names : Union[str, List[str]]
        Cross-section identifiers of the portfolios to include, applied to
        every PnL data frame.
    aum : Number
        Assets under management
    lback : int
        Number of observations in the rolling volatility window. Defaults to
        252.
    annualization_factor : int
        Number of observations per year used to annualize the volatility.
        Defaults to 252.
    vol_target : Optional[float]
        Volatility target, drawn as a horizontal reference line. Defaults to None,
        in which case no line is drawn.
    figsize : Tuple[float, float]
        Size of the figure. Defaults to (13, 5).
    alpha : float
        Opacity of the plotted lines. Defaults to 0.9.
    linewidth : float
        Width of the plotted lines. Defaults to 1.5.
    title : str
        Title of the plot. Defaults to "Rolling realized volatility of
        strategy PnL".
    title_fontsize : int
        Font size of the title. Defaults to 14.
    xlabel : str
        Label for the x-axis
    ylabel : str
        Label for the y-axis.
    label_fontsize : int
        Font size of the axis labels and the legend. Defaults to 11.
    show_mean : bool
        Whether to append each series' mean realized volatility to its legend
        label. Defaults to True.

    Returns
    -------
    Tuple[plt.Figure, plt.Axes]
        The figure and the axes containing the line plot.
    """
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
    """
    Plot portfolio volatility and the scaling factor implied by a vol target.

    The left panel shows the portfolio volatility prior to volatility
    targeting, the right panel the factor by which positions must be scaled to
    meet the target, that is the target divided by the portfolio volatility.
    The mean and median of each series are added to the panel legends.

    Parameters
    ----------
    vol_df : pd.DataFrame
        Portfolio volatility data in long format
    vol_target : int
        Annualized volatility target
    vol_xcat : str
        Category of the portfolio volatility series to plot.
    linewidth : float
        Width of the plotted lines. Defaults to 1.
    x_label : str
        Label for the x-axis of both panels
    y_label_pvol : str
        Label for the y-axis of the volatility panel
    y_label_scale : str
        Label for the y-axis of the scaling factor panel
    title : str
        Overall figure title
    title_fontsize : int
        Font size of the figure title. Defaults to 15.
    figsize : Tuple[float, float]
        Size of the figure. Defaults to (13, 6).

    Returns
    -------
    Tuple[plt.Figure, np.ndarray]
        The figure and the array of the two axes.
    """
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
) -> None:
    """
    Plot the effect of scaling factor estimation error on costs and PnL vol.
    Both panels scatter the bias of the scaling factor estimator against its
    standard deviation, with one point per simulated estimator. Points are
    coloured by transaction costs in the left panel and by realized PnL
    volatility in the right panel.

    Parameters
    ----------
    bias : np.ndarray
        Bias of the scaling factor estimator, one value per point.
    var : np.ndarray
        Standard deviation of the scaling factor estimator, one value per
        point.
    costs : np.ndarray
        Transaction costs used to colour the left panel, one value per point.
    pnl_vols : np.ndarray
        Realized PnL volatility used to colour the right panel, one value per
        point.
    title : str
        Overall figure title. Defaults to "Impact of scaling factor estimator
        bias and variance on cost and PnL vol".
    title_fontsize : int
        Font size of the figure title. Defaults to 15.
    subtitles : List[str]
        Panel titles, also used as the colour bar labels. Defaults to None, in
        which case ["Transaction costs (USDmn)", "Realized PnL volatility"] is
        used.
    xlabel : str
        Label for the x-axis of both panels. Defaults to "Bias of the scaling
        factor".
    ylabel : str
        Label for the y-axis of the left panel. Defaults to "Standard
        deviation of the scaling factor".
    figsize : Tuple[float, float]
        Size of the figure. Defaults to (16, 5).
    point_size : int
        Size of the scatter points. Defaults to 170.
    """
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
    evals: Dict[Union[str, Tuple[str, str]], pd.DataFrame],
    metrics: Optional[Union[str, List[str]]] = None,
    net_col: Optional[str] = None,
    gross_col: Optional[str] = None,
    pnl_name: str = "PNL",
    show_gross: bool = True,
    group_order: Optional[List[str]] = None,
    series_order: Optional[List[str]] = None,
    title: str = "",
    title_fontsize: int = 15,
    net_label: str = "Net of costs",
    gross_label: str = "Gross",
    ncols: Optional[int] = None,
    figsize: Optional[Tuple[float, float]] = None,
    label_fontsize: int = 9,
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Plot performance metrics gross and net of transaction costs. Each metric is
    drawn in its own panel as a grouped bar chart. The gross value is drawn as a
    dashed outline and the net value as a filled bar, so that the cost impact is
    the gap between the two.

    The input is a collection of `evaluate_pnl` outputs, one per configuration,
    each holding the metrics as its index and the net and gross PnL as two of
    its columns. The keys of that collection supply the remaining dimensions of
    the chart. Scalar keys give one bar per configuration, labelled by its key.
    Two-element tuple keys are read as (group, series) and give one group of
    bars per group and one bar per series within each group, for instance one
    group per position rule and one bar per signal.

    Parameters
    ----------
    evals : Dict[Union[str, Tuple[str, str]], pd.DataFrame]
        Outputs of `evaluate_pnl`, keyed either by a single label or by a
        (group, series) tuple. All keys must be of the same kind. Each data
        frame must hold a PnL net of costs and a PnL gross of costs column,
        which is the case when `evaluate_pnl` is called with `df_pnle`. A
        gross column is not needed when `show_gross` is False.
    metrics : Optional[Union[str, List[str]]]
        Metrics to plot, one panel per metric, given as index entries of the
        `evaluate_pnl` outputs. Defaults to None, in which case every metric
        shared by all outputs is plotted.
    net_col : Optional[str]
        Name of the column holding the PnL net of costs, applied to every
        output. Defaults to None, in which case the column ending in `pnl_name`
        is used, or failing that the one whose name contains "net" or "incl".
    gross_col : Optional[str]
        Name of the column holding the PnL gross of costs, applied to every
        output. Defaults to None, in which case the column ending in `pnl_name`
        with an "e" appended is used, or failing that the one whose name
        contains "gross" or "excl". Ignored when `show_gross` is False.
    pnl_name : str
        Name given to the PnL by `proxy_pnl_calc`, which ends the category of
        the PnL net of costs and, with an "e" appended, that of the PnL gross of
        costs. Defaults to "PNL". Used only where `net_col` or `gross_col` is
        not given.
    show_gross : bool
        Whether to draw the gross value of each bar as a dashed outline behind
        the net value. Defaults to True. When False the cost-state entries are
        dropped from the legend.
    group_order : Optional[List[str]]
        Order of the bar groups along the x-axis. Defaults to None, in which
        case the order of first appearance in evals is used.
    series_order : Optional[List[str]]
        Order of the bars within each group, used only with tuple keys.
        Defaults to None, in which case the order of first appearance in evals
        is used.
    title : str
        Overall figure title. Defaults to an empty string.
    title_fontsize : int
        Font size of the figure title. Defaults to 15.
    net_label : str
        Legend label of the filled bars. Defaults to "Net of costs". Unused
        when `show_gross` is False.
    gross_label : str
        Legend label of the dashed outlines. Defaults to "Gross". Unused when
        `show_gross` is False.
    ncols : Optional[int]
        Number of panel columns. Defaults to None, in which case at most four
        columns are used.
    figsize : Optional[Tuple[float, float]]
        Size of the figure. Defaults to None, in which case it is scaled to the
        number of panel rows and columns.
    label_fontsize : int
        Font size of the x-axis tick labels. Defaults to 9.

    Returns
    -------
    Tuple[plt.Figure, np.ndarray]
        The figure and the two-dimensional array of axes.
    """
    if not isinstance(evals, dict) or not evals:
        raise ValueError("`evals` must be a non-empty dict of `evaluate_pnl` outputs.")

    for key, df in evals.items():
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"The entry of `evals` at {key!r} must be a pd.DataFrame.")

    keys = list(evals)
    tuple_keys = [isinstance(key, tuple) for key in keys]

    if any(tuple_keys) and not all(tuple_keys):
        raise ValueError(
            "The keys of `evals` must either all be (group, series) tuples or "
            "all be single labels."
        )

    two_level = all(tuple_keys)

    if two_level and any(len(key) != 2 for key in keys):
        raise ValueError("Tuple keys of `evals` must be (group, series) pairs.")

    key_pairs = {key: (key if two_level else (key, None)) for key in keys}

    groups = _ordered_levels(
        [key_pairs[key][0] for key in keys], group_order, "group_order"
    )
    series = _ordered_levels(
        [key_pairs[key][1] for key in keys], series_order, "series_order"
    )

    frames = list(evals.values())

    if metrics is None:
        shared = set(frames[0].index)
        for frame in frames[1:]:
            shared &= set(frame.index)

        metrics = [metric for metric in frames[0].index if metric in shared]

        if not metrics:
            raise ValueError(
                "The `evals` outputs hold no metric in common; pass `metrics` "
                "explicitly."
            )
    else:
        if isinstance(metrics, str):
            metrics = [metrics]

        unknown = [
            metric
            for metric in metrics
            if not any(metric in frame.index for frame in frames)
        ]
        if unknown:
            raise ValueError(f"Metrics not found in any `evals` output: {unknown}")

    cost_cols = _cost_columns(
        evals, net_col, gross_col, need_gross=show_gross, pnl_name=pnl_name
    )

    # Panel layout
    if ncols is None:
        ncols = min(4, len(metrics))

    nrows = (len(metrics) + ncols - 1) // ncols

    if figsize is None:
        figsize = (5.0 * ncols, 5.5 * nrows)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=figsize,
        squeeze=False,
    )
    axes_flat = axes.ravel()

    xpos = np.arange(len(groups))
    width = 0.8 / len(series)
    colors = sns.color_palette("colorblind", len(series))

    lookup = {pair: key for key, pair in key_pairs.items()}

    for ax, metric in zip(axes_flat, metrics):
        for j, series_label in enumerate(series):
            offset = (j - (len(series) - 1) / 2) * width

            gross, net = [], []
            for group in groups:
                key = lookup.get((group, series_label))
                df = evals[key] if key is not None else None

                # missing combinations and metrics leave a gap rather than raise
                if df is None or metric not in df.index:
                    gross.append(np.nan)
                    net.append(np.nan)
                    continue

                net_name, gross_name = cost_cols[key]
                net.append(df.at[metric, net_name])
                gross.append(df.at[metric, gross_name] if show_gross else np.nan)

            if show_gross:
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
                label=(
                    series_label
                    if series_label is not None and metric == metrics[0]
                    else None
                ),
            )

        ax.set_title(metric, fontsize=12)

        ax.set_xticks(xpos)
        ax.set_xticklabels([str(group) for group in groups], fontsize=label_fontsize)

        ax.axhline(0, color="black", linewidth=0.8)

        ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.3)
        ax.grid(axis="x", visible=False)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Remove empty panels
    for ax in axes_flat[len(metrics) :]:
        ax.remove()

    handles, labels = axes_flat[0].get_legend_handles_labels()

    if show_gross:
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
        labels += [net_label, gross_label]

    if labels:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=len(labels),
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
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot cumulative transaction costs split into bid-offer and roll costs.

    Parameters
    ----------
    cost_dfs : List[pd.DataFrame]
        Transaction cost data in long format, one data frame per bar. Each
        must contain xcat and value columns.
    labels : List[str]
        Bar labels, one per entry in cost_dfs.
    rollcost_suffix : str
        Suffix identifying roll cost categories in xcat. Defaults to
        "TCOST_ROLLCOST".
    bidoffer_suffix : str
        Suffix identifying bid-offer cost categories in xcat. Defaults to
        "TCOST_BIDOFFER".
    title : str
        Title of the plot. Defaults to an empty string.
    title_fontsize : int
        Font size of the title. Defaults to 16.
    figsize : Tuple[float, float]
        Size of the figure. Defaults to (11, 5.5).
    label_rotation : int
        Rotation of the x-axis tick labels, in degrees. Defaults to 90.
    sort_by_label : bool
        Whether to sort the bars alphabetically by label. Defaults to False.

    Returns
    -------
    Tuple[plt.Figure, plt.Axes]
        The figure and the axes containing the bar chart.
    """
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

        rows.append({"label": label, "Bid-offer": bidoffer_cost, "Roll": roll_cost})

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


def metric_tradeoff_plot(
    x_values: Union[np.ndarray, List[Number]],
    y_values: Union[np.ndarray, List[Number], List[List[Number]]],
    labels: Optional[Union[str, List[str]]] = None,
    point_labels: Optional[List[str]] = None,
    annotate_series: int = 0,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    title_fontsize: int = 14,
    marker: str = "o",
    linewidth: float = 2.0,
    annotation_offset: Tuple[float, float] = (5, -10),
    annotation_fontsize: int = 9,
    sort_by_x: bool = True,
    figsize: Tuple[float, float] = (10, 6),
    ax: plt.Axes = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot one or more metrics against a shared set of x-axis values.

    Each series in y_values is drawn as a marked line against x_values, so that
    the trade-off between the quantity on the x-axis and the metrics on the
    y-axis becomes visible.

    Parameters
    ----------
    x_values : Union[np.ndarray, List[Number]]
        Values for the x-axis, shared across all series
    y_values : Union[np.ndarray, List[Number], List[List[Number]]]
        Metric values to plot. Either a single series of the same length as
        x_values, or an array of shape (n, len(x_values)) holding n series.
    labels : Optional[Union[str, List[str]]]
        Legend labels, one per series in y_values. Defaults to None, in which
        case no legend is drawn.
    point_labels : Optional[List[str]]
        Annotations for the individual points, one per entry in x_values
    annotate_series : int
        Index of the series whose points carry the annotations. Defaults to 0.
    title : str
        Title of the plot. Defaults to an empty string.
    xlabel : str
        Label for the x-axis. Defaults to an empty string.
    ylabel : str
        Label for the y-axis. Defaults to an empty string.
    title_fontsize : int
        Font size of the title. Defaults to 14.
    marker : str
        Marker style used for the plotted points. Defaults to "o".
    linewidth : float
        Width of the plotted lines. Defaults to 2.0.
    annotation_offset : Tuple[float, float]
        Offset of the annotations from their points, in points. Defaults to
        (5, -10).
    annotation_fontsize : int
        Font size of the annotations. Defaults to 9.
    sort_by_x : bool
        Whether to sort the points by their x-value before plotting.
        Defaults to True.
    figsize : Tuple[float, float]
        Size of the figure, used only when ax is not provided. Defaults to
        (10, 6).
    ax : plt.Axes
        Optional existing axes to draw on. Defaults to None, in which case a
        new figure and axes are created.

    Returns
    -------
    Tuple[plt.Figure, plt.Axes]
        The figure and the axes containing the line plot.
    """
    x_vals = np.asarray(x_values)
    y_vals = np.atleast_2d(np.asarray(y_values, dtype=float))

    if y_vals.shape[1] != x_vals.size:
        raise ValueError(
            "Each series in `y_values` must have the same length as `x_values`, "
            f"got {y_vals.shape[1]} and {x_vals.size}."
        )

    if labels is None:
        labels = [None] * len(y_vals)
    elif isinstance(labels, str):
        labels = [labels]

    if len(labels) != len(y_vals):
        raise ValueError(
            "`labels` must hold one label per series in `y_values`, "
            f"got {len(labels)} and {len(y_vals)}."
        )

    if point_labels is not None and len(point_labels) != x_vals.size:
        raise ValueError(
            "`point_labels` must hold one label per entry in `x_values`, "
            f"got {len(point_labels)} and {x_vals.size}."
        )

    if sort_by_x:
        order = np.argsort(x_vals)
        x_vals = x_vals[order]
        y_vals = y_vals[:, order]

        if point_labels is not None:
            point_labels = [point_labels[i] for i in order]

    new_figure = ax is None
    if new_figure:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    for label, y_series in zip(labels, y_vals):
        ax.plot(
            x_vals,
            y_series,
            marker=marker,
            linewidth=linewidth,
            label=label,
        )

    if point_labels is not None:
        for x, y, point_label in zip(x_vals, y_vals[annotate_series], point_labels):
            ax.annotate(
                str(point_label),
                (x, y),
                xytext=annotation_offset,
                textcoords="offset points",
                fontsize=annotation_fontsize,
            )

    ax.set_title(title, fontsize=title_fontsize)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.25)

    if any(label is not None for label in labels):
        ax.legend(frameon=False)

    if new_figure:
        fig.tight_layout()

    return fig, ax


def metric_scatterplot(
    data: pd.DataFrame,
    x: str,
    y: str,
    hue: Optional[str] = None,
    style: Optional[str] = None,
    title: str = "",
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    trendline: bool = True,
    trend_degree: int = 1,
    trend_label: Optional[str] = None,
    trend_margin: float = 0.05,
    title_fontsize: int = 15,
    label_fontsize: float = 10.5,
    tick_fontsize: float = 9,
    legend_fontsize: float = 9,
    legend_outside: bool = True,
    point_size: float = 110,
    alpha: float = 0.85,
    figsize: Tuple[float, float] = (9.5, 6),
    ax: plt.Axes = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Scatter one metric against another across configurations, with an optional
    fitted trend line.

    Each row of data is drawn as a single point, so that the relationship
    between the two metrics becomes visible. For example the annualized
    turnover of each configuration against the drag that transaction costs
    impose on its Sharpe ratio. Two further columns may be encoded as the
    colour and the marker shape of the points, for instance the positioning
    rule and the signal behind each configuration.

    Parameters
    ----------
    data : pd.DataFrame
        One row per configuration, holding at least the columns named by x and
        y.
    x : str
        Column of data plotted on the x-axis.
    y : str
        Column of data plotted on the y-axis.
    hue : Optional[str]
        Column of data encoded as the colour of the points. Defaults to None,
        in which case a single colour is used.
    style : Optional[str]
        Column of data encoded as the marker shape of the points. Defaults to
        None, in which case a single marker is used.
    title : str
        Title of the plot. Defaults to an empty string.
    xlabel : Optional[str]
        Label for the x-axis. Defaults to None, in which case the name of the x
        column is used.
    ylabel : Optional[str]
        Label for the y-axis. Defaults to None, in which case the name of the y
        column is used.
    trendline : bool
        Whether to fit and draw a trend line through the points. Defaults to
        True. The line is drawn only where the points span a range of x-values
        and at least trend_degree + 1 of them are numeric.
    trend_degree : int
        Degree of the polynomial fitted for the trend line. Defaults to 1, that
        is a straight line.
    trend_label : Optional[str]
        Legend label of the trend line. Defaults to None, in which case the
        line is drawn without a legend entry.
    trend_margin : float
        Fraction of the observed range of x-values by which the trend line
        extends beyond the outermost points at either end. Defaults to 0.05.
    title_fontsize : int
        Font size of the title. Defaults to 15.
    label_fontsize : float
        Font size of the axis labels. Defaults to 10.5.
    tick_fontsize : float
        Font size of the axis tick labels. Defaults to 9.
    legend_fontsize : float
        Font size of the legend. Defaults to 9.
    legend_outside : bool
        Whether to place the legend to the right of the axes rather than inside
        them. Defaults to True.
    point_size : float
        Size of the scatter points. Defaults to 110.
    alpha : float
        Opacity of the scatter points. Defaults to 0.85.
    figsize : Tuple[float, float]
        Size of the figure, used only when ax is not provided. Defaults to
        (9.5, 6).
    ax : plt.Axes
        Optional existing axes to draw on. Defaults to None, in which case a
        new figure and axes are created.

    Returns
    -------
    Tuple[plt.Figure, plt.Axes]
        The figure and the axes containing the scatter plot.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("`data` must be a pd.DataFrame.")

    for argname, col in (("x", x), ("y", y), ("hue", hue), ("style", style)):
        if col is not None and col not in data.columns:
            raise KeyError(
                f"`{argname}` column '{col}' is not in `data`, which holds "
                f"{list(data.columns)}."
            )

    new_figure = ax is None
    if new_figure:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    sns.scatterplot(
        data=data,
        x=x,
        y=y,
        hue=hue,
        style=style,
        s=point_size,
        alpha=alpha,
        edgecolor="white",
        linewidth=0.8,
        ax=ax,
    )

    if trendline:
        fit_data = data[[x, y]].apply(pd.to_numeric, errors="coerce").dropna()
        x_min, x_max = fit_data[x].min(), fit_data[x].max()
        span = x_max - x_min

        # an exact fit needs one point per coefficient, a slope a range of x
        if len(fit_data) > trend_degree and span > 0:
            coeffs = np.polyfit(fit_data[x], fit_data[y], trend_degree)

            pad = trend_margin * span
            xs = np.linspace(x_min - pad, x_max + pad, 100)

            ax.plot(
                xs,
                np.polyval(coeffs, xs),
                color="0.4",
                linestyle="--",
                linewidth=1.2,
                alpha=0.8,
                zorder=1,
                label=trend_label,
            )

    ax.set_title(title, fontsize=title_fontsize, pad=14)

    ax.set_xlabel(
        xlabel if xlabel is not None else x,
        fontsize=label_fontsize,
        labelpad=10,
    )
    ax.set_ylabel(
        ylabel if ylabel is not None else y,
        fontsize=label_fontsize,
        labelpad=10,
    )

    ax.tick_params(axis="both", labelsize=tick_fontsize)

    ax.grid(
        axis="both",
        linestyle="--",
        linewidth=0.6,
        alpha=0.3,
    )
    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_alpha(0.7)
    ax.spines["bottom"].set_alpha(0.7)

    handles, _ = ax.get_legend_handles_labels()
    if handles:
        if legend_outside:
            ax.legend(
                title=None,
                bbox_to_anchor=(1.02, 1),
                loc="upper left",
                borderaxespad=0,
                frameon=False,
                fontsize=legend_fontsize,
            )
        else:
            ax.legend(
                title=None,
                loc="best",
                frameon=False,
                fontsize=legend_fontsize,
            )

    if new_figure:
        fig.tight_layout()

    return fig, ax


def plot_metric_sensitivity(
    evals: Dict[Tuple[Any, ...], pd.DataFrame],
    metrics: Optional[Union[str, List[str]]] = None,
    net_col: Optional[str] = None,
    gross_col: Optional[str] = None,
    pnl_name: str = "PNL",
    show_gross: Union[bool, Sequence[str]] = True,
    metric_labels: Optional[Dict[str, str]] = None,
    ylabels: Optional[Dict[str, str]] = None,
    group_order: Optional[List[Any]] = None,
    series_order: Optional[List[Any]] = None,
    title: str = "",
    xlabel: str = "",
    net_label: str = "Net of costs",
    gross_label: str = "Gross",
    title_fontsize: int = 15,
    label_fontsize: float = 11,
    net_linewidth: float = 2.0,
    gross_linewidth: float = 1.3,
    gross_alpha: float = 0.7,
    zero_line: bool = True,
    thousands_separator: bool = True,
    ncols: Optional[int] = None,
    figsize: Optional[Tuple[float, float]] = None,
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Plot performance metrics against the parameter swept in a sensitivity
    analysis, gross and net of transaction costs.

    Each line traces one configuration as a positioning parameter is varied,
    for instance the Sharpe ratio of a signal as the dollar per signal, the
    leverage or the volatility target is raised. The value net of costs is drawn
    as a solid line and the value gross of costs as a dashed line in the same
    colour, so that the cost drag is the gap between the two.

    The input is a collection of `evaluate_pnl` outputs, one per point of the
    sweep, each holding the metrics as its index and the net and gross PnL as
    two of its columns. The keys of that collection supply the remaining
    dimensions of the chart. Two-element (series, x) keys give one line per
    series and one panel per metric. Three-element (group, series, x) keys add a
    panel dimension, giving one panel per group when a single metric is plotted
    and a grid of one row per metric and one column per group otherwise. The
    x-value is the position of that point along the x-axis.

    Parameters
    ----------
    evals : Dict[Tuple[Any, ...], pd.DataFrame]
        Outputs of `evaluate_pnl`, keyed either by a (group, series, x) triple
        or by a (series, x) pair, where x is the numeric position of that point
        along the x-axis. All keys must be of the same kind. Each data frame
        must hold a PnL net of costs column, and a PnL gross of costs column
        wherever a gross line is drawn, which is the case when `evaluate_pnl` is
        called with `df_pnle`.
    metrics : Optional[Union[str, List[str]]]
        Metrics to plot, given as index entries of the `evaluate_pnl` outputs.
        Defaults to None, in which case every metric shared by all outputs is
        plotted.
    net_col : Optional[str]
        Name of the column holding the PnL net of costs, applied to every
        output. Defaults to None, in which case the column ending in `pnl_name`
        is used, or failing that the one whose name contains "net" or "incl".
    gross_col : Optional[str]
        Name of the column holding the PnL gross of costs, applied to every
        output. Defaults to None, in which case the column ending in `pnl_name`
        with an "e" appended is used, or failing that the one whose name
        contains "gross" or "excl". Ignored where no gross line is drawn.
    pnl_name : str
        Name given to the PnL by `proxy_pnl_calc`, which ends the category of
        the PnL net of costs and, with an "e" appended, that of the PnL gross of
        costs. Defaults to "PNL". Used only where `net_col` or `gross_col` is
        not given.
    show_gross : Union[bool, Sequence[str]]
        Metrics for which the gross value is drawn as a dashed line behind the
        net value. Defaults to True, that is every plotted metric. False draws
        the net values alone. A sequence of metric names restricts the gross
        line to those metrics, which is needed for a metric such as the
        transaction cost, whose gross value is zero by construction.
    metric_labels : Optional[Dict[str, str]]
        Mapping from metric to the label used for it, as the panel title with
        two-element keys and as the y-axis label with three-element keys, where
        the panel title carries the group. Defaults to None, in which case the
        metric names are used.
    ylabels : Optional[Dict[str, str]]
        Mapping from metric to the y-axis label used for it. Defaults to None,
        in which case the label from `metric_labels` is used, or failing that
        the metric name.
    group_order : Optional[List[Any]]
        Order of the panel groups, used only with three-element keys. Defaults
        to None, in which case the order of first appearance in `evals` is used.
        A subset may be given, in which case the remaining groups are dropped.
    series_order : Optional[List[Any]]
        Order of the lines within each panel. Defaults to None, in which case
        the order of first appearance in `evals` is used. A subset may be given,
        in which case the remaining series are dropped.
    title : str
        Overall figure title. Defaults to an empty string, in which case no
        title is drawn.
    xlabel : str
        Label for the x-axis, drawn on the bottom panel of each column.
    net_label : str
        Legend label of the solid lines. Defaults to "Net of costs". Unused
        where no gross line is drawn.
    gross_label : str
        Legend label of the dashed lines. Defaults to "Gross". Unused where no
        gross line is drawn.
    title_fontsize : int
        Font size of the figure title. Defaults to 15.
    label_fontsize : float
        Font size of the axis labels. Defaults to 11.
    net_linewidth : float
        Width of the solid lines. Defaults to 2.0.
    gross_linewidth : float
        Width of the dashed lines. Defaults to 1.3.
    gross_alpha : float
        Opacity of the dashed lines. Defaults to 0.7.
    zero_line : bool
        Whether to mark the zero level on panels whose values straddle it.
        Defaults to True.
    thousands_separator : bool
        Whether to write the x-axis tick labels with thousands separators.
        Defaults to True.
    ncols : Optional[int]
        Number of panel columns. Defaults to None, in which case one column per
        group is used with three-element keys and at most three columns
        otherwise.
    figsize : Optional[Tuple[float, float]]
        Size of the figure. Defaults to None, in which case it is scaled to the
        number of panel rows and columns.

    Returns
    -------
    Tuple[plt.Figure, np.ndarray]
        The figure and the two-dimensional array of axes.
    """
    if not isinstance(evals, dict) or not evals:
        raise ValueError("`evals` must be a non-empty dict of `evaluate_pnl` outputs.")

    for key, df in evals.items():
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"The entry of `evals` at {key!r} must be a pd.DataFrame.")

    lines, groups, series, grouped = _sensitivity_frame(
        evals, group_order, series_order
    )

    frames = list(evals.values())

    if metrics is None:
        shared = set(frames[0].index)
        for frame in frames[1:]:
            shared &= set(frame.index)

        metrics = [metric for metric in frames[0].index if metric in shared]

        if not metrics:
            raise ValueError(
                "The `evals` outputs hold no metric in common; pass `metrics` "
                "explicitly."
            )
    else:
        if isinstance(metrics, str):
            metrics = [metrics]

        unknown = [
            metric
            for metric in metrics
            if not any(metric in frame.index for frame in frames)
        ]
        if unknown:
            raise ValueError(f"Metrics not found in any `evals` output: {unknown}")

    if isinstance(show_gross, bool):
        gross_metrics = set(metrics) if show_gross else set()
    else:
        gross_metrics = set(show_gross)

        unknown = [metric for metric in gross_metrics if metric not in metrics]
        if unknown:
            raise ValueError(
                f"`show_gross` holds metrics that are not plotted: {unknown}"
            )

    cost_cols = _cost_columns(
        evals, net_col, gross_col, need_gross=bool(gross_metrics), pnl_name=pnl_name
    )

    metric_labels = metric_labels or {}
    ylabels = ylabels or {}

    # Panel layout: one panel per metric, or a metric by group grid
    panels = [(metric, group) for metric in metrics for group in groups]

    if ncols is None:
        ncols = len(groups) if grouped else min(3, len(panels))

    nrows = (len(panels) + ncols - 1) // ncols

    if figsize is None:
        figsize = (5.5 * ncols, 4.6 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes_flat = axes.ravel()

    colors = sns.color_palette("colorblind", len(series))

    drew_gross = False

    for i, (metric, group) in enumerate(panels):
        ax = axes_flat[i]
        panel_values = []

        for color, series_label in zip(colors, series):
            points = lines[(group, series_label)]

            x_vals, net_vals, gross_vals = [], [], []
            for x, key in points:
                df = evals[key]

                # a point whose output lacks the metric leaves a gap
                if metric not in df.index:
                    continue

                net_name, gross_name = cost_cols[key]

                x_vals.append(x)
                net_vals.append(df.at[metric, net_name])
                gross_vals.append(
                    df.at[metric, gross_name] if metric in gross_metrics else np.nan
                )

            if metric in gross_metrics:
                ax.plot(
                    x_vals,
                    gross_vals,
                    color=color,
                    linestyle="--",
                    linewidth=gross_linewidth,
                    alpha=gross_alpha,
                )
                drew_gross = True

            ax.plot(
                x_vals,
                net_vals,
                color=color,
                linewidth=net_linewidth,
            )

            panel_values += [
                value for value in net_vals + gross_vals if pd.notna(value)
            ]

        # With a group per column the group is constant down a column, so only
        # the top row needs a title
        if not grouped:
            ax.set_title(metric_labels.get(metric, metric), fontsize=12)
        elif i < ncols:
            ax.set_title(str(group), fontsize=12)

        # With a group per column the metric varies down the rows, so the
        # y-label carries it and only the leftmost panel of each row needs one
        if not grouped or i % ncols == 0:
            ax.set_ylabel(
                ylabels.get(metric, metric_labels.get(metric, metric)),
                fontsize=label_fontsize,
            )

        # The bottom panel of each column carries the x-label
        if xlabel and i + ncols >= len(panels):
            ax.set_xlabel(xlabel, fontsize=label_fontsize)

        if zero_line and panel_values and min(panel_values) < 0 < max(panel_values):
            ax.axhline(0, color="0.4", linewidth=0.8, alpha=0.5, zorder=0)

        if thousands_separator:
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f}"))

        ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.3)
        ax.grid(axis="x", visible=False)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_alpha(0.7)
        ax.spines["bottom"].set_alpha(0.7)

        ax.tick_params(axis="both", labelsize=label_fontsize - 1, length=0)

    # Remove empty panels
    for ax in axes_flat[len(panels) :]:
        ax.remove()

    handles = [
        plt.Line2D([], [], color=color, linewidth=net_linewidth, label=str(label))
        for color, label in zip(colors, series)
    ]

    if drew_gross:
        handles += [
            plt.Line2D([], [], color="0.35", linewidth=net_linewidth, label=net_label),
            plt.Line2D(
                [],
                [],
                color="0.35",
                linewidth=gross_linewidth,
                linestyle="--",
                label=gross_label,
            ),
        ]

    fig.legend(
        handles=handles,
        labels=[handle.get_label() for handle in handles],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.03),
        ncol=len(handles),
        frameon=False,
        fontsize=label_fontsize - 1,
        handlelength=2.5,
        columnspacing=1.7,
    )

    if title:
        fig.suptitle(title, fontsize=title_fontsize)

    fig.tight_layout()

    return fig, axes
