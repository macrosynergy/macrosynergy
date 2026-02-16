"""
Module for plotting comparative return performance metrics across cross-sections,
categories, or tickers.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Optional, Union

from macrosynergy.management.types import QuantamentalDataFrame
from macrosynergy.management.utils import reduce_df, _map_to_business_day_frequency


def view_performance(
    df: pd.DataFrame,
    xcats: Optional[List[str]] = None,
    cids: Optional[List[str]] = None,
    tickers: Optional[List[str]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    val: str = "value",
    bms: Optional[str] = None,
    metrics: Optional[List[str]] = None,
    title: Optional[str] = None,
    title_fontsize: int = 16,
    ylab: Optional[str] = None,
    size: Tuple[float] = (14, 8),
    labels: Optional[Union[List[str], dict]] = None,
    legend_loc: str = "upper center",
    legend_bbox_to_anchor: Optional[Tuple[float]] = None,
    return_metrics: bool = False,
    return_fig: bool = False,
):
    """
    Plots comparative return performance metrics in a grouped bar chart.

    Creates a bar chart showing performance metrics (annualized returns, standard
    deviation, Sharpe ratio, Sortino ratio, and optionally benchmark correlation)
    across cross-sections, categories, or specific tickers.
    

    Parameters
    ----------
    df : ~pandas.DataFrame
        Standardized DataFrame with the necessary columns: 'cid', 'xcat', 'real_date'
        and at least one column with values of interest (typically returns).
    xcats : List[str], optional
        Extended categories to compare. Either xcats or cids or tickers must be
        specified, but not combinations.
    cids : List[str], optional
        Cross-sections to compare. Either xcats or cids or tickers must be specified,
        but not combinations.
    tickers : List[str], optional
        Specific tickers to compare (format: "CID_XCAT"). Either xcats or cids or
        tickers must be specified, but not combinations.
    start : str, optional
        Earliest date in ISO format. Default is earliest date in df.
    end : str, optional
        Latest date in ISO format. Default is latest date in df.
    val : str
        Name of column that contains the values (returns). Default is 'value'.
    bms : str, optional
        Benchmark ticker (format: "CID_XCAT") for correlation calculation. If None,
        benchmark correlation is not shown.
    metrics : List[str], optional
        List of metrics to display. Available options: "Return %", "St. Dev. %",
        "Sharpe Ratio", "Sortino Ratio", and "{bms} correl" (if bms provided).
        If None, all available metrics are shown. Default is None.
    title : str, optional
        Chart title. If None, a default title is generated.
    title_fontsize : int
        Font size of the title. Default is 16.
    ylab : str, optional
        Y-axis label. Default is no label.
    size : Tuple[float]
        Tuple of width and height of graph. Default is (14, 8).
    labels : Union[List[str], dict], optional
        Custom labels for the compared items. If dict, maps from cid/xcat/ticker to
        label. If list, must match the order of items being compared.
    legend_loc : str
        Location of legend; passed to matplotlib.pyplot.legend(). Default is 'upper
        center'.
    legend_bbox_to_anchor : Tuple[float], optional
        Passed to matplotlib.pyplot.legend(). Default is None, which positions the
        legend below the plot.
    return_metrics : bool
        If True, return the metrics DataFrame instead of plotting. Default is False.
    return_fig : bool
        If True, return the Matplotlib figure object instead of displaying. Default
        is False.

    Returns
    -------
    pd.DataFrame or matplotlib.figure.Figure or None
        If return_metrics=True, returns DataFrame with performance metrics.
        If return_fig=True, returns the figure object.
        Otherwise displays the plot and returns None.

    Notes
    -----
    Performance metrics calculated:
    - Return %: Annualized return (mean * 261)
    - St. Dev. %: Annualized standard deviation (std * sqrt(261))
    - Sharpe Ratio: Annualized return / annualized standard deviation
    - Sortino Ratio: Annualized return / downside deviation
    - Benchmark correlation: Correlation with benchmark return series (if bms provided)
    """

    df = QuantamentalDataFrame(df)

    # Validate input: exactly one of xcats, cids, or tickers should enable comparison
    n_params = sum([
        xcats is not None and len(xcats) > 1,
        cids is not None and len(cids) > 1,
        tickers is not None,
    ])

    if n_params == 0:
        raise ValueError(
            "Must specify multiple xcats, multiple cids, or tickers for comparison."
        )
    if n_params > 1:
        raise ValueError(
            "Can only compare across one dimension: specify either multiple xcats OR "
            "multiple cids OR tickers, not combinations."
        )

    # Determine comparison mode and items
    if tickers is not None:
        comparison_mode = "tickers"
        comparison_items = tickers
        # Parse tickers into cids and xcats
        parsed_tickers = [ticker.split("_", 1) for ticker in tickers]
        cids_from_tickers = [t[0] for t in parsed_tickers]
        xcats_from_tickers = [t[1] for t in parsed_tickers]

        # Validate tickers exist
        for cid, xcat in parsed_tickers:
            filt = (df["cid"] == cid) & (df["xcat"] == xcat)
            if df[filt].empty:
                raise ValueError(f"Ticker {cid}_{xcat} not found in DataFrame.")
    elif xcats is not None and len(xcats) > 1:
        comparison_mode = "xcats"
        comparison_items = xcats
        aggregate_cids = False
        if cids is None or len(cids) == 0:
            # Get all unique cids from the dataframe and aggregate
            cids = df["cid"].unique().tolist()
            aggregate_cids = True
        elif len(cids) == 1 and cids[0] == "ALL":
            # Get all unique cids from the dataframe and aggregate
            cids = df["cid"].unique().tolist()
            aggregate_cids = True
        elif len(cids) > 1:
            raise ValueError(
                "When comparing across xcats, can only specify a single cid or 'ALL'."
            )
    else:  # Multiple cids
        comparison_mode = "cids"
        comparison_items = cids
        if xcats is None or len(xcats) == 0:
            raise ValueError("Must specify xcat when comparing across cids.")
        elif len(xcats) > 1:
            raise ValueError(
                "When comparing across cids, can only specify a single xcat."
            )

    # Reduce dataframe based on comparison mode
    if comparison_mode == "tickers":
        # For tickers, we need to handle each ticker separately
        dfs = []
        for cid, xcat in zip(cids_from_tickers, xcats_from_tickers):
            df_reduced = reduce_df(
                df, xcats=[xcat], cids=[cid], start=start, end=end, out_all=False
            )
            df_reduced["ticker"] = f"{cid}_{xcat}"
            dfs.append(df_reduced)
        dfx = pd.concat(dfs, ignore_index=True)
        group_by = "ticker"
    else:
        dfx = reduce_df(df, xcats, cids, start, end, out_all=False)
        group_by = "xcat" if comparison_mode == "xcats" else "cid"

        # If comparing across xcats and need to aggregate cids, do it now
        if comparison_mode == "xcats" and aggregate_cids:
            # Calculate equal-weighted average across all cids for each xcat/date
            dfx = dfx.groupby(["real_date", "xcat"], observed=True)[val].mean().reset_index()
            dfx["cid"] = "ALL"  # Mark as aggregated

    # Check if dataframe is empty after filtering
    if dfx.empty:
        available_xcats = df["xcat"].unique().tolist()
        available_cids = df["cid"].unique().tolist()
        error_msg = (
            f"No data found after filtering. "
            f"Available xcats: {available_xcats}, "
            f"Available cids: {available_cids}. "
        )
        if tickers is not None:
            error_msg += f"Requested tickers: {tickers}. "
        elif xcats is not None:
            error_msg += f"Requested xcats: {xcats}. "
        if cids is not None:
            error_msg += f"Requested cids: {cids}. "
        raise ValueError(error_msg)

    # Calculate metrics
    metrics_df = _calculate_performance_metrics(
        dfx,
        group_by=group_by,
        val=val,
        bms=bms,
        df_full=df,
        start=start,
        end=end
    )

    # Filter metrics if specified
    if metrics is not None:
        # Validate requested metrics
        available_metrics = metrics_df.index.tolist()
        invalid_metrics = [m for m in metrics if m not in available_metrics]
        if invalid_metrics:
            raise ValueError(
                f"Invalid metrics: {invalid_metrics}. "
                f"Available metrics: {available_metrics}"
            )
        # Filter to requested metrics
        metrics_df = metrics_df.loc[metrics, :]

    # Apply custom labels if provided
    if labels is not None:
        if isinstance(labels, dict):
            metrics_df.rename(columns=labels, inplace=True)
        elif isinstance(labels, list):
            if len(labels) != len(metrics_df.columns):
                raise ValueError(
                    f"Number of labels ({len(labels)}) must match number of items "
                    f"being compared ({len(metrics_df.columns)})."
                )
            metrics_df.columns = labels
        else:
            raise TypeError("labels must be a list or dict.")

    if return_metrics:
        return metrics_df

    # Create visualization
    # Get date range for title (safe from NaT since we checked dfx is not empty)
    start_date = start if start is not None else dfx["real_date"].min().strftime("%Y-%m-%d")
    end_date = end if end is not None else dfx["real_date"].max().strftime("%Y-%m-%d")

    fig = _plot_performance_bars(
        metrics_df,
        title=title,
        title_fontsize=title_fontsize,
        ylab=ylab,
        size=size,
        legend_loc=legend_loc,
        legend_bbox_to_anchor=legend_bbox_to_anchor,
        start=start_date,
        end=end_date,
    )

    if return_fig:
        return fig
    else:
        plt.show()


def _calculate_performance_metrics(
    df: pd.DataFrame,
    group_by: str,
    val: str = "value",
    bms: Optional[str] = None,
    df_full: Optional[pd.DataFrame] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    Calculate performance metrics for return series.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with return data.
    group_by : str
        Column name to group by ('cid', 'xcat', or 'ticker').
    val : str
        Column name containing values.
    bms : str, optional
        Benchmark ticker for correlation calculation.
    df_full : pd.DataFrame, optional
        Full dataframe (needed if bms is provided).
    start : str, optional
        Start date for benchmark data.
    end : str, optional
        End date for benchmark data.

    Returns
    -------
    pd.DataFrame
        DataFrame with metrics as index and groups as columns.
    """

    # Pivot data for calculations
    dfw = df.pivot(index="real_date", columns=group_by, values=val)

    # Initialize metrics list
    metrics = [
        "Return %",
        "St. Dev. %",
        "Sharpe Ratio",
        "Sortino Ratio",
    ]

    # Add benchmark correlation if provided
    if bms is not None:
        metrics.append(f"{bms} correl")

    # Create results dataframe
    results = pd.DataFrame(index=metrics, columns=dfw.columns)

    # Calculate annualized return (261 trading days per year)
    results.loc["Return %", :] = dfw.mean(axis=0) * 261

    # Calculate annualized standard deviation
    results.loc["St. Dev. %", :] = dfw.std(axis=0) * np.sqrt(261)

    # Calculate Sharpe ratio
    results.loc["Sharpe Ratio", :] = (
        results.loc["Return %", :] / results.loc["St. Dev. %", :]
    )

    # Calculate Sortino ratio (using downside deviation)
    dsd = dfw.apply(
        lambda x: np.sqrt(np.sum(x[x < 0] ** 2) / len(x))
    ) * np.sqrt(261)
    results.loc["Sortino Ratio", :] = results.loc["Return %", :] / dsd

    # Calculate benchmark correlation if provided
    if bms is not None:
        if df_full is None:
            raise ValueError("df_full must be provided when bms is specified.")

        # Parse benchmark ticker
        bms_parts = bms.split("_", 1)
        if len(bms_parts) != 2:
            raise ValueError(f"Benchmark ticker '{bms}' must be in format 'CID_XCAT'.")

        bms_cid, bms_xcat = bms_parts

        # Get benchmark data
        df_bms = reduce_df(
            df_full, xcats=[bms_xcat], cids=[bms_cid], start=start, end=end, out_all=False
        )

        if df_bms.empty:
            raise ValueError(f"Benchmark ticker '{bms}' not found in DataFrame.")

        # Pivot benchmark data
        bms_series = df_bms.set_index("real_date")[val]

        # Calculate correlations
        correlations = {}
        for col in dfw.columns:
            # Find common dates
            common_idx = dfw.index.intersection(bms_series.index)
            if len(common_idx) > 0:
                corr = dfw.loc[common_idx, col].corr(bms_series.loc[common_idx])
                correlations[col] = corr
            else:
                correlations[col] = np.nan

        results.loc[f"{bms} correl", :] = pd.Series(correlations)

    return results


def _plot_performance_bars(
    metrics_df: pd.DataFrame,
    title: Optional[str] = None,
    title_fontsize: int = 16,
    ylab: Optional[str] = None,
    size: Tuple[float] = (14, 8),
    legend_loc: str = "upper center",
    legend_bbox_to_anchor: Optional[Tuple[float]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> plt.Figure:
    """
    Create grouped bar chart for performance metrics.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        DataFrame with metrics as index and items as columns.
    title : str, optional
        Chart title.
    title_fontsize : int
        Font size of the title.
    size : Tuple[float]
        Figure size.
    start : str, optional
        Start date for title.
    end : str, optional
        End date for title.

    Returns
    -------
    matplotlib.figure.Figure
        The created figure.
    """

    # Set style to match view_ranges
    sns.set_theme(style="darkgrid")

    # Reshape data for seaborn: need long format with columns [item, metric, value]
    # Transpose so items are in index and metrics are columns
    df_transposed = metrics_df.T
    df_transposed.index.name = 'item'

    # Convert to long format for seaborn
    df_long = df_transposed.reset_index()
    df_long = pd.melt(
        df_long,
        id_vars=['item'],
        var_name='metric',
        value_name='value'
    )

    # Create figure
    fig, ax = plt.subplots(figsize=size)

    # Create grouped bar plot with items on x-axis and metrics as hue
    # Use Paired palette to match view_ranges
    sns.barplot(
        data=df_long,
        x='item',
        y='value',
        hue='metric',
        ax=ax,
        palette='Paired',
    )

    # Rotate x-axis labels to prevent overlap
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    # Customize plot to match view_ranges style
    ax.set_xlabel("")
    if ylab is None:
        ylab = ""
    ax.set_ylabel(ylab)

    if title is None:
        if start is not None and end is not None:
            title = f"Performance metrics from {start} to {end}"
        else:
            title = "Performance metrics"

    ax.set_title(title, fontdict={"fontsize": title_fontsize})

    # Match view_ranges styling
    ax.xaxis.grid(True)
    ax.axhline(0, ls="--", linewidth=1, color="black")

    # Position legend to match view_ranges (below the plot)
    if legend_bbox_to_anchor is None:
        n_metrics = len(metrics_df.index)
        legend_bbox_to_anchor = (0.5, -0.15 - 0.05 * max(0, (n_metrics - 2)))

    ax.legend(loc=legend_loc, bbox_to_anchor=legend_bbox_to_anchor, ncol=3)

    plt.tight_layout()

    return fig


if __name__ == "__main__":
    from macrosynergy.management.simulate import make_qdf

    # Create sample data
    cids = ["AUD", "CAD", "GBP", "USD"]
    xcats = ["FXXR_NSA", "EQXR_NSA"]

    df_cids = pd.DataFrame(
        index=cids, columns=["earliest", "latest", "mean_add", "sd_mult"]
    )
    df_cids.loc["AUD"] = ["2010-01-01", "2020-12-31", 0.5, 1.0]
    df_cids.loc["CAD"] = ["2010-01-01", "2020-12-31", 0.3, 1.2]
    df_cids.loc["GBP"] = ["2010-01-01", "2020-12-31", 0.4, 1.1]
    df_cids.loc["USD"] = ["2010-01-01", "2020-12-31", 0.2, 0.9]

    df_xcats = pd.DataFrame(
        index=xcats,
        columns=["earliest", "latest", "mean_add", "sd_mult", "ar_coef", "back_coef"],
    )
    df_xcats.loc["FXXR_NSA"] = ["2010-01-01", "2020-12-31", 0.1, 1, 0, 0.3]
    df_xcats.loc["EQXR_NSA"] = ["2010-01-01", "2020-12-31", 0.3, 1.5, 0, 0.3]

    dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)

    # Test 1: Compare across cids for single xcat
    print("Test 1: Compare across cids")
    view_performance(dfd, xcats=["FXXR_NSA"], cids=["AUD", "CAD", "GBP", "USD"])

    # Test 2: Compare across xcats for single cid
    print("\nTest 2: Compare across xcats")
    view_performance(dfd, xcats=["FXXR_NSA", "EQXR_NSA"], cids=["ALL"])

    # Test 3: Compare specific tickers with benchmark
    print("\nTest 3: Compare tickers with benchmark")
    view_performance(
        dfd,
        tickers=["AUD_FXXR_NSA", "GBP_FXXR_NSA", "USD_FXXR_NSA", "USD_EQXR_NSA"],
        bms="USD_EQXR_NSA"
    )
    
    # Test 4: Filter metrics
    view_performance(
        dfd,
        tickers=["AUD_FXXR_NSA", "GBP_FXXR_NSA", "USD_FXXR_NSA", "USD_EQXR_NSA"],
        bms="USD_EQXR_NSA",
        metrics=["USD_EQXR_NSA correl"]
    )
