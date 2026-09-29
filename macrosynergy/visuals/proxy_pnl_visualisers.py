from numbers import Number
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.dates as mdates
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
    Identify the net and gross columns of a single evaluate_pnl output. Each
    of them is resolved in turn by

    1. the explicit net_col or gross_col argument,
    2. the single column ending in pnl_name for the net and in pnl_name with an
       "e" appended for the gross, which is the convention proxy_pnl_calc
       writes,
    3. the single column whose name contains "net" or "incl" for the net and
       "gross" or "excl" for the gross
    """
    columns = list(df.columns)
    resolved = {}

    roles = [("net", net_col, pnl_name, ("net", "incl"))]
    if need_gross:
        roles.append(("gross", gross_col, pnl_name + "e", ("gross", "excl")))

    for role, explicit, suffix, tokens in roles:
        if explicit is not None:
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
    """
    return {
        key: _resolve_cost_columns(
            df=df,
            key=key,
            net_col=net_col,
            gross_col=gross_col,
            need_gross=need_gross,
            pnl_name=pnl_name,
        )
        for key, df in evals.items()
    }


def _sensitivity_frame(
    evals: Dict[Tuple[Any, ...], pd.DataFrame],
    group_order: Optional[List[Any]],
    series_order: Optional[List[Any]],
) -> Tuple[Dict[Tuple[Any, Any], List[Tuple[float, Any]]], List[Any], List[Any], bool]:
    """
    Group the keys of a sweep by the line they belong to. The keys of evals are
    read as (group, series, x) triples or, where no group level is present, as
    (series, x) pairs. They are collected into one entry per line, holding the
    x-value and the key of every point on that line,.

    Parameters
    ----------
    evals : Dict[Tuple[Any, ...], pd.DataFrame]
        Outputs of evaluate_pnl, keyed by (group, series, x) or (series, x).
    group_order : Optional[List[Any]]
        Requested order of the groups, or None for the order of first appearance.
    series_order : Optional[List[Any]]
        Requested order of the series, or None for the order of first appearance.

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
        key_parts[key] = (group, series, float(x))

    groups = _ordered_levels(
        values=[key_parts[key][0] for key in keys],
        order=group_order if grouped else None,
        argname="group_order",
    )
    series = _ordered_levels(
        values=[key_parts[key][1] for key in keys],
        order=series_order,
        argname="series_order",
    )

    lines = {(group, s): [] for group in groups for s in series}
    for key, (group, s, x) in key_parts.items():
        if (group, s) in lines:
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


def compare_proxy_pnls(
    pnl_dfs: List[pd.DataFrame],
    pnle_dfs: List[pd.DataFrame],
    portfolio_names: List[str],
    pnl_names: Optional[List[str]] = None,
    title: str = "PnL with and without costs",
    aum: Optional[Union[Number, Sequence[Number]]] = None,
    ylabel: Optional[str] = None,
    title_fontsize: int = 18,
    incl_costs_label: str = "Incl. Costs",
    excl_costs_label: str = "Excl. Costs",
    sharey: bool = False,
    ncols: Optional[int] = None,
    cumsum: bool = True,
    line_width: float = 1,
    height: float = 3.4,
    aspect: float = 1.06,
    label_fontsize: int = 13,
    subtitle_fontsize: int = 14,
    max_xticks: int = 6,
    start: Optional[str] = None,
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
    aum : Optional[Union[Number, Sequence[Number]]]
        Assets under management, in the same units as the PnL values. When given,
        each PnL is divided by it and plotted as a percentage of risk capital.
    ylabel : Optional[str]
        Label for the y-axis. Defaults to None, in which case it follows the
        units plotted: "% of risk capital" when `aum` is given and "USD mn"
        when it is not.
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
        Whether to plot cumulative sum. Defaults to True.
    line_width : float
        Width of the plotted lines. Defaults to 1.
    height : float
        Height of a single subplot, in inches. Defaults to 3.4. The figure is
        this tall times the number of subplot rows.
    aspect : float
        Width-height ratio of a single subplot. Defaults to 1.06. The figure is
        `aspect` times `height` wide times the number of subplot columns.
    label_fontsize : int
        Font size of the axis labels and the legend. Defaults to 13.
    subtitle_fontsize : int
        Font size of the subplot titles. Defaults to 14.
    max_xticks : int
        Upper bound on the number of date ticks per subplot. Defaults to 6.
    start : Optional[str]
        Earliest date to plot, as an ISO date string.

    Returns
    -------
    Tuple[plt.Figure, np.ndarray]
        The figure and the two-dimensional array of axes.
    """
    if aum is None:
        aums = [None] * len(pnl_dfs)
    else:
        if isinstance(aum, (list, tuple, np.ndarray, pd.Series)):
            aums = list(aum)
        else:
            aums = [aum] * len(pnl_dfs)

        if len(aums) != len(pnl_dfs):
            raise ValueError(
                "`aum` must be a single number or hold one number per portfolio, "
                f"got {len(aums)} for {len(pnl_dfs)} portfolios."
            )

        for value in aums:
            if not isinstance(value, Number) or isinstance(value, bool):
                raise TypeError("`aum` must hold numbers.")
            if value <= 0:
                raise ValueError("`aum` must be positive.")

    if ylabel is None:
        ylabel = "% of risk capital" if aum is not None else "USD mn"

    if pnl_names is None and len(pnl_dfs) == 1:
        pnl_names = [""]
    elif pnl_names is None:
        pnl_names = portfolio_names

    assert len(pnl_names) == len(portfolio_names)

    pnlcount = len(pnl_dfs)

    if ncols is None:
        ncols = min(3, pnlcount)

    nrows = (pnlcount + ncols - 1) // ncols
    figsize = (aspect * height * ncols, height * nrows)

    with sns.axes_style("whitegrid"), sns.plotting_context("notebook"):
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

        for i, (pnl_df, pnle_df, portfolio_name, pnl_name, portfolio_aum) in enumerate(
            zip(pnl_dfs, pnle_dfs, portfolio_names, pnl_names, aums)
        ):
            ax = axes_flat[i]

            pnl = pnl_df.copy()
            pnle = pnle_df.copy()

            pnl["cost_type"] = incl_costs_label
            pnle["cost_type"] = excl_costs_label

            data = pd.concat([pnl, pnle], ignore_index=True)
            data = reduce_df(data, cids=[portfolio_name], start=start)
            data = data.sort_values(["cid", "cost_type", "real_date"])

            if portfolio_aum is not None:
                data["plot_value"] = 100 * data["value"] / portfolio_aum
            else:
                data["plot_value"] = data["value"]

            if cumsum:
                data["plot_value"] = data.groupby(["cid", "cost_type"])[
                    "plot_value"
                ].cumsum()

            sns.lineplot(
                data=data,
                x="real_date",
                y="plot_value",
                hue="cost_type",
                palette="colorblind",
                estimator=None,
                lw=line_width,
                ax=ax,
            )

            ax.set_title(pnl_name, fontsize=subtitle_fontsize)
            ax.set_xlabel("")

            # only show y-axis label on leftmost subplot in each row
            if i % ncols == 0:
                ax.set_ylabel(ylabel, fontsize=label_fontsize)
            else:
                ax.set_ylabel("")

            # above the grid, below the data
            ax.axhline(0, color="0.4", linewidth=1.0, alpha=0.8, zorder=1)

            # a concise date axis
            locator = mdates.AutoDateLocator(maxticks=max_xticks)
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

            ax.tick_params(axis="both", labelsize=label_fontsize - 1, length=0)

            ax.set_axisbelow(True)
            ax.grid(axis="both", linestyle="--", linewidth=0.7, alpha=0.7)

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            if legend_handles is None:
                legend_handles, legend_labels = ax.get_legend_handles_labels()

            # remove subplot-level legend
            legend = ax.get_legend()
            if legend is not None:
                legend.remove()

        # remove empty subplots
        for ax in axes_flat[pnlcount:]:
            ax.remove()

        fig.suptitle(title, fontsize=title_fontsize)

        fig.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.0),
            ncol=2,
            frameon=False,
            fontsize=label_fontsize,
        )

        legend_in = label_fontsize / 72 + 0.22
        fig.tight_layout(rect=(0, legend_in / figsize[1], 1, 1))

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
    with sns.axes_style("whitegrid"), sns.plotting_context("notebook"):
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
            alpha=0.7,
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


def plot_metrics_before_and_after_costs(
    evals: Dict[Union[str, Tuple[str, str]], pd.DataFrame],
    metrics: Optional[Union[str, List[str]]] = None,
    metric_labels: Optional[Dict[str, str]] = None,
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
    height: float = 5.5,
    aspect: float = 0.91,
    label_fontsize: int = 9,
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Plot performance metrics gross and net of transaction costs. Each metric is
    drawn in its own panel as a grouped bar chart. The gross value is drawn as a
    dashed outline and the net value as a filled bar, so that the cost impact is
    the gap between the two.

    Parameters
    ----------
    evals : Dict[Union[str, Tuple[str, str]], pd.DataFrame]
        Outputs of evaluate_pnl, keyed either by a single label or by a
        (group, series) tuple.
    metrics : Optional[Union[str, List[str]]]
        Metrics to plot, one panel per metric. Defaults to None, in which case
        every metric shared by all outputs is plotted.
    metric_labels : Optional[Dict[str, str]]
        Mapping from metric to the panel title used for display.
    net_col : Optional[str]
        Name of the column holding the PnL net of costs. Defaults to None, in
        which case the column ending in `pnl_name` is used, or failing that
        the one whose name contains "net" or "incl".
    gross_col : Optional[str]
        Name of the column holding the PnL gross of costs. Defaults to None, in which
        case the column ending in `pnl_name` with an "e" appended is used, or failing
        that the one whose name contains "gross" or "excl".
    pnl_name : str
        Name given to the PnL by `proxy_pnl_calc`.  Defaults to "PNL"
    show_gross : bool
        Whether to draw the gross value of each bar as a dashed outline behind
        the net value. Defaults to True.
    group_order : Optional[List[str]]
        Order of the bar groups along the x-axis.
    series_order : Optional[List[str]]
        Order of the bars within each group, used only with tuple keys.
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
    height : float
        Height of a single panel, in inches. Defaults to 5.5. The figure is this
        tall times the number of panel rows.
    aspect : float
        Width-height ratio of a single panel. Defaults to 0.91. The figure is
        `aspect` times `height` wide times the number of panel columns.
    label_fontsize : int
        Font size of the x-axis tick labels. Defaults to 9.

    Returns
    -------
    Tuple[plt.Figure, np.ndarray]
        The figure and array of axes.
    """
    if not isinstance(evals, dict) or not evals:
        raise ValueError("evals must be a dict of evaluate_pnl outputs")

    for key, df in evals.items():
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"The entry of evals at {key!r} must be a pd.DataFrame")

    if metric_labels is None:
        metric_labels = {}
    elif not isinstance(metric_labels, dict):
        raise TypeError("metric_labels must be a dict of metric to label.")
    elif not all(isinstance(label, str) for label in metric_labels.values()):
        raise TypeError("The values of metric_labels must be strings.")

    keys = list(evals)
    tuple_keys = [isinstance(key, tuple) for key in keys]

    if any(tuple_keys) and not all(tuple_keys):
        raise ValueError(
            "The keys of evals must either all be (group, series) tuples or "
            "all be single labels."
        )

    two_level = all(tuple_keys)
    if two_level and any(len(key) != 2 for key in keys):
        raise ValueError("Tuple keys of evals must be (group, series) pairs")

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
            raise ValueError("The evals outputs hold no metric in common")
    else:
        if isinstance(metrics, str):
            metrics = [metrics]

        unknown = [
            metric
            for metric in metrics
            if not any(metric in frame.index for frame in frames)
        ]
        if unknown:
            raise ValueError(f"Metrics not found in any evals output: {unknown}")

    cost_cols = _cost_columns(
        evals=evals,
        net_col=net_col,
        gross_col=gross_col,
        need_gross=show_gross,
        pnl_name=pnl_name,
    )

    with sns.axes_style("whitegrid"), sns.plotting_context("notebook"):
        # Panel layout
        if ncols is None:
            ncols = min(4, len(metrics))

        nrows = (len(metrics) + ncols - 1) // ncols
        figsize = (aspect * height * ncols, height * nrows)

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

                    # missing combinations and metrics leave a gap
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

            ax.set_title(metric_labels.get(metric, metric), fontsize=12)

            ax.set_xticks(xpos)
            ax.set_xticklabels(
                [str(group) for group in groups], fontsize=label_fontsize
            )

            ax.axhline(0, color="black", linewidth=0.8)

            ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.7)
            ax.grid(axis="x", visible=False)

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        # remove empty panels
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
    height: float = 4.6,
    aspect: float = 1.2,
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Plot performance metrics against the parameter swept in a sensitivity
    analysis, gross and net of transaction costs.

    The input is a collection of `evaluate_pnl` outputs, one per point of the
    sweep. The keys of that collection supply the remaining dimensions of the
    chart. Two-element (series, x) keys give one line per series and one panel
    per metric. Three-element (group, series, x) keys add a panel dimension,
    giving one panel per group when a single metric is plotted and a grid of one
    row per metric and one column per group otherwise. The x-value is the position
    of that point along the x-axis.

    Parameters
    ----------
    evals : Dict[Tuple[Any, ...], pd.DataFrame]
        Outputs of `evaluate_pnl`, keyed either by (group, series, x) or by a
        (series, x) pair, where x is the numeric position of that point along
        the x-axis.
    metrics : Optional[Union[str, List[str]]]
        Metrics to plot. Defaults to None, in which case every metric shared
        by all outputs is plotted.
    net_col : Optional[str]
        Name of the column holding the PnL net of costs, applied to every
        output. Defaults to None, in which case the column ending in `pnl_name`
        is used, or failing that the one whose name contains "net" or "incl".
    gross_col : Optional[str]
        Name of the column holding the PnL gross of costs, applied to every
        output. Defaults to None, in which case the column ending in `pnl_name`
        with an "e" appended is used, or failing that the one whose name
        contains "gross" or "excl".
    pnl_name : str
        Name given to the PnL by `proxy_pnl_calc` Defaults to "PNL".
    show_gross : Union[bool, Sequence[str]]
        Metrics for which the gross value is drawn as a dashed line behind the
        net value. Defaults to True, that is every plotted metric. False draws
        the net values alone. A sequence of metric names restricts the gross
        line to those metrics.
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
        to None.
    series_order : Optional[List[Any]]
        Order of the lines within each panel. Defaults to None.
    title : str
        Overall figure title. Defaults to an empty string.
    xlabel : str
        Label for the x-axis, drawn on the bottom panel of each column.
    net_label : str
        Legend label of the solid lines. Defaults to "Net of costs".
    gross_label : str
        Legend label of the dashed lines. Defaults to "Gross".
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
        Whether to mark the zero level on panels. Defaults to True.
    thousands_separator : bool
        Whether to write the x-axis tick labels with thousands separators.
        Defaults to True.
    ncols : Optional[int]
        Number of panel columns. Defaults to None, in which cas at most 3 columns
        are used.
    height : float
        Height of a single panel, in inches. Defaults to 4.6. The figure is this
        tall times the number of panel rows.
    aspect : float
        Width-height ratio of a single panel. Defaults to 1.2. The figure is
        `aspect` times `height` wide times the number of panel columns.

    Returns
    -------
    Tuple[plt.Figure, np.ndarray]
        The figure and the two-dimensional array of axes.
    """
    if not isinstance(evals, dict) or not evals:
        raise ValueError("evals must be a non-empty dict of evaluate_pnl outputs")

    for key, df in evals.items():
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"The entry of evals at {key!r} must be a pd.DataFrame")

    with sns.axes_style("whitegrid"), sns.plotting_context("notebook"):
        lines, groups, series, grouped = _sensitivity_frame(
            evals=evals, group_order=group_order, series_order=series_order
        )

        frames = list(evals.values())

        if metrics is None:
            # find the metrics that are shared
            shared = set(frames[0].index)
            for frame in frames[1:]:
                shared &= set(frame.index)

            metrics = [metric for metric in frames[0].index if metric in shared]

            if not metrics:
                raise ValueError("The evals outputs hold no metric in common")
        else:
            if isinstance(metrics, str):
                metrics = [metrics]

            unknown = [
                metric
                for metric in metrics
                if not any(metric in frame.index for frame in frames)
            ]
            if unknown:
                raise ValueError(f"Metrics not found in any evals output: {unknown}")

        if isinstance(show_gross, bool):
            gross_metrics = set(metrics) if show_gross else set()
        else:
            gross_metrics = set(show_gross)

            unknown = [metric for metric in gross_metrics if metric not in metrics]
            if unknown:
                raise ValueError(
                    f"show_gross holds metrics that are not plotted: {unknown}"
                )

        cost_cols = _cost_columns(
            evals=evals,
            net_col=net_col,
            gross_col=gross_col,
            need_gross=bool(gross_metrics),
            pnl_name=pnl_name,
        )

        metric_labels = metric_labels or {}
        ylabels = ylabels or {}

        # Panel layout: one panel per metric, or a metric by group grid
        panels = [(metric, group) for metric in metrics for group in groups]

        if ncols is None:
            ncols = len(groups) if grouped else min(3, len(panels))

        nrows = (len(panels) + ncols - 1) // ncols
        figsize = (aspect * height * ncols, height * nrows)

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
                ax.axhline(0, color="0.4", linewidth=1.0, alpha=0.9, zorder=0)

            if thousands_separator:
                ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f}"))

            ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.7)
            ax.grid(axis="x", visible=False)

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_alpha(0.85)
            ax.spines["bottom"].set_alpha(0.85)

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
                plt.Line2D(
                    [], [], color="0.35", linewidth=net_linewidth, label=net_label
                ),
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
