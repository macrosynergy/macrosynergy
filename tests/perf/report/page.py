"""
HTML report for a benchmark results file.

Usage: python -m tests.perf.report.page perf-results.json [--baseline baseline.json]
"""

from __future__ import annotations

import argparse
import html
import io
import json
import math
import re
import urllib.parse
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from matplotlib import rc_context
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, NullFormatter

from tests.perf.report.results import (
    BOOTSTRAP_SEED,
    NOISE_THRESHOLD,
    SIGNIFICANCE_LEVEL,
    BenchmarkMeasurement,
    BenchmarkRunResults,
    bootstrap_mean_changes,
)

PAGE_TITLE = "Macrosynergy Package performance benchmarks"
# Each verdict is shown with a symbol and its label, never by colour alone.
VERDICT_SYMBOLS = {
    "regressed": "▲",
    "improved": "▼",
    "within noise": "≈",
    "no change": "=",
    "workload changed": "!",
}

# Charts are drawn in these light-mode colours, then each is swapped for a CSS variable
# so the page can restyle them for dark mode.
CHART_COLOR_VARIABLES = {
    "#0b0b0b": "--ink",
    "#52514e": "--ink-secondary",
    "#898781": "--ink-muted",
    "#e1e0d9": "--gridline",
    "#c3c2b7": "--axis",
    "#2a78d6": "--series-1",
    "#eb6834": "--series-2",
    "#1baf7a": "--series-3",
    "#006300": "--good",
}
# Only three slots stay colour-blind distinguishable when every pair can sit side by side.
SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a")
CHART_STYLE = {
    "font.size": 9,
    "svg.fonttype": "none",
    "text.color": "#0b0b0b",
    "axes.edgecolor": "#c3c2b7",
    "axes.labelcolor": "#52514e",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#e1e0d9",
    "grid.linewidth": 0.8,
    "xtick.color": "#898781",
    "ytick.color": "#898781",
    "xtick.labelcolor": "#52514e",
    "ytick.labelcolor": "#52514e",
    "legend.frameon": False,
}
# Status colours and marker shapes, so a verdict never rests on colour alone.
VERDICT_STYLES = {
    "regressed": ("#d03b3b", "^"),
    "improved": ("#006300", "v"),
}
NEUTRAL_VERDICT_STYLE = ("#898781", "o")
TIME_UNITS = ((1.0, "s"), (1e-3, "ms"), (1e-6, "µs"), (1e-9, "ns"))


def _time_unit(seconds: float) -> Tuple[float, str]:
    """
    The largest unit in which a duration reads as at least one.

    Parameters
    ----------
    seconds : float
        A duration, in seconds.

    Returns
    -------
    Tuple[float, str]
        The unit's size in seconds and its symbol.
    """
    for scale, unit in TIME_UNITS:
        if seconds >= scale:
            return scale, unit
    return TIME_UNITS[-1]


def _format_seconds(seconds: Optional[float]) -> str:
    """
    A duration in its natural unit, to three significant figures.

    Parameters
    ----------
    seconds : Optional[float]
        A duration, in seconds, or None.

    Returns
    -------
    str
        The formatted duration, or an en dash when there is none.
    """
    if seconds is None or (isinstance(seconds, float) and math.isnan(seconds)):
        return "–"
    scale, unit = _time_unit(seconds)
    return f"{seconds / scale:.3g} {unit}"


def _format_interval(row: pd.Series, statistic: str) -> str:
    """
    A statistic followed by its bootstrap bounds.

    Parameters
    ----------
    row : pd.Series
        One row of `measurement_table`.
    statistic : str
        The column holding the statistic, whose bounds sit in `_low` and `_high`.

    Returns
    -------
    str
        The value, then the bounds in brackets when the run had enough rounds for them.
    """
    value = _format_seconds(row[statistic])
    low, high = row[f"{statistic}_low"], row[f"{statistic}_high"]
    if pd.isna(low) or pd.isna(high):
        return value
    return f"{value} <span class=bounds>[{_format_seconds(low)}, {_format_seconds(high)}]</span>"


def _count(number: int, noun: str) -> str:
    """
    A count with its noun, pluralised when it is not one.

    Parameters
    ----------
    number : int
        How many.
    noun : str
        The singular noun.

    Returns
    -------
    str
        Such as "1 case" or "3 cases".
    """
    return f"{number} {noun}" if number == 1 else f"{number} {noun}s"


def _anchor(identifier: str) -> str:
    """
    A measurement identifier made safe for use as an HTML id.

    Parameters
    ----------
    identifier : str
        The measurement's identifier.

    Returns
    -------
    str
        The identifier with every run of other characters replaced by a hyphen.
    """
    return re.sub(r"[^A-Za-z0-9]+", "-", identifier).strip("-")


def _svg(figure: Figure) -> str:
    """
    A figure as inline SVG whose colours follow the page's CSS variables.

    Parameters
    ----------
    figure : Figure
        The drawn figure.

    Returns
    -------
    str
        The `<svg>` element, without the XML prologue.
    """
    buffer = io.StringIO()
    figure.savefig(
        buffer,
        format="svg",
        bbox_inches="tight",
        transparent=True,
        metadata={"Date": None},
    )
    svg = buffer.getvalue()
    svg = svg[svg.index("<svg") :]
    for color, variable in CHART_COLOR_VARIABLES.items():
        svg = svg.replace(color, f"var({variable})")
    return svg


def _format_rate(value: Optional[float]) -> str:
    """
    A per-second count in thousands, millions or billions.

    Parameters
    ----------
    value : Optional[float]
        Observations per second, or None.

    Returns
    -------
    str
        Such as "897k/s", or an en dash when there is none.
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "–"
    for scale, suffix in ((1e9, "G"), (1e6, "M"), (1e3, "k")):
        if abs(value) >= scale:
            return f"{value / scale:.3g}{suffix}/s"
    return f"{value:.3g}/s"


def _format_bytes(value: Optional[float]) -> str:
    """
    A byte count in binary units.

    Parameters
    ----------
    value : Optional[float]
        Bytes, or None.

    Returns
    -------
    str
        Such as "13.4 MiB", or an en dash when there is none.
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "–"
    for scale, unit in ((2**30, "GiB"), (2**20, "MiB"), (2**10, "KiB")):
        if value >= scale:
            return f"{value / scale:.3g} {unit}"
    return f"{value:.0f} B"


# Column plotted, axis label, tick formatter and whether the axis is logarithmic.
METRIC_CHARTS = {
    "time": ("median_seconds", "median time per call", _format_seconds, True),
    "throughput": (
        "observations_per_second",
        "observations per second",
        _format_rate,
        False,
    ),
    "memory": ("peak_memory_bytes", "peak memory", _format_bytes, True),
}


def _value_axis(axis: Any, format_value: Callable[[float], str], log: bool) -> None:
    """
    Label an axis's ticks with a formatter, once the data is drawn.

    Parameters
    ----------
    axis : Any
        The matplotlib y axis.
    format_value : Callable[[float], str]
        Formats one tick value.
    log : bool
        Whether the axis is logarithmic. One spanning under a decade is made linear, and
        a linear axis starts at zero.

    Returns
    -------
    None
    """
    axis.axes.autoscale_view()
    low, high = sorted(axis.get_view_interval())
    if log and high < 10 * low:
        # Under a decade a log axis has one major label or none; linear reads better.
        axis.axes.set_yscale("linear")
        log = False
    formatter = FuncFormatter(lambda value, _: format_value(value))
    axis.set_major_formatter(formatter)
    axis.set_minor_formatter(NullFormatter())
    if not log:
        axis.axes.set_ylim(bottom=0)


def _tier_axis(axes: Any, rows: pd.DataFrame) -> None:
    """
    Put one tick per tier on a logarithmic observation-count x axis, named by tier.

    Parameters
    ----------
    axes : Any
        The matplotlib axes.
    rows : pd.DataFrame
        `measurement_table` rows, read for tiers and their observation counts.

    Returns
    -------
    None
    """
    tiers = rows.drop_duplicates("tier").sort_values("observation_count")
    axes.set_xscale("log")
    axes.set_xticks(tiers["observation_count"], tiers["tier"])
    axes.set_xticks([], minor=True)
    axes.margins(x=0.08)


@rc_context(CHART_STYLE)
def metric_chart(
    rows: pd.DataFrame, baseline_rows: Optional[pd.DataFrame], metric: str
) -> str:
    """
    One statistic against tier for one benchmark, one line per variant.

    Parameters
    ----------
    rows : pd.DataFrame
        The `measurement_table` rows of a single benchmark.
    baseline_rows : Optional[pd.DataFrame]
        The baseline run's rows for the same benchmark, drawn dashed, or None.
    metric : str
        A key of `METRIC_CHARTS`.

    Returns
    -------
    str
        The chart as inline SVG. Time carries the median's bootstrap bounds as error
        bars.
    """
    column, label, format_value, log = METRIC_CHARTS[metric]
    figure = Figure(figsize=(4.6, 2.5))
    axes = figure.subplots()
    variants = sorted(rows["variant"].unique())
    for variant, color in zip(variants, SERIES_COLORS):
        series = rows[rows["variant"] == variant].sort_values("observation_count")
        values = series[column].astype(float).to_numpy()
        errors = None
        if metric == "time":
            errors = [
                values - series["median_seconds_low"].fillna(series[column]).to_numpy(),
                series["median_seconds_high"].fillna(series[column]).to_numpy()
                - values,
            ]
        axes.errorbar(
            series["observation_count"].to_numpy(),
            values,
            yerr=errors,
            color=color,
            linewidth=2,
            marker="o",
            markersize=5,
            label=variant or "current",
        )
        if baseline_rows is not None and not baseline_rows.empty:
            previous = baseline_rows[baseline_rows["variant"] == variant].sort_values(
                "observation_count"
            )
            axes.plot(
                previous["observation_count"].to_numpy(),
                previous[column].astype(float).to_numpy(),
                color=color,
                linewidth=1.5,
                linestyle="--",
                marker="o",
                markersize=4,
                markerfacecolor="none",
            )
    if log:
        axes.set_yscale("log")
    _value_axis(axes.yaxis, format_value, log)
    _tier_axis(axes, rows)
    axes.set_ylabel(label)
    handles, labels = axes.get_legend_handles_labels()
    if baseline_rows is not None and not baseline_rows.empty:
        handles.append(Line2D([], [], color="#898781", linestyle="--", linewidth=1.5))
        labels.append("baseline")
    if len(handles) > 1:
        axes.legend(handles, labels, fontsize=7.5)
    return _svg(figure)


@rc_context(CHART_STYLE)
def change_by_tier_chart(rows: pd.DataFrame, changes: pd.DataFrame) -> str:
    """
    Change in mean against tier for one benchmark, with bounds and the noise band.

    Parameters
    ----------
    rows : pd.DataFrame
        The `measurement_table` rows of a single benchmark.
    changes : pd.DataFrame
        `compare_against` rows indexed by `measurement_id`.

    Returns
    -------
    str
        The chart as inline SVG; each marker's colour and shape give its verdict.
    """
    figure = Figure(figsize=(4.6, 2.5))
    axes = figure.subplots()
    axes.axhspan(
        -100 * NOISE_THRESHOLD, 100 * NOISE_THRESHOLD, color="#e1e0d9", linewidth=0
    )
    axes.axhline(0, color="#898781", linewidth=1)
    variants = sorted(rows["variant"].unique())
    tiers = list(rows.sort_values("observation_count")["tier"].drop_duplicates())
    for offset_index, (variant, color) in enumerate(zip(variants, SERIES_COLORS)):
        offset = (offset_index - (len(variants) - 1) / 2) * 0.12
        series = rows[rows["variant"] == variant]
        for _, row in series.iterrows():
            if row["measurement_id"] not in changes.index:
                continue
            change = changes.loc[row["measurement_id"]]
            if pd.isna(change["relative_change"]):
                continue
            marker_color, marker = VERDICT_STYLES.get(change["verdict"], (color, "o"))
            axes.errorbar(
                [tiers.index(row["tier"]) + offset],
                [100 * change["relative_change"]],
                yerr=[
                    [100 * (change["relative_change"] - change["relative_change_low"])],
                    [
                        100
                        * (change["relative_change_high"] - change["relative_change"])
                    ],
                ],
                color=marker_color,
                ecolor=color,
                marker=marker,
                markersize=7,
                linewidth=2,
                label=variant or None,
            )
    axes.set_xticks(range(len(tiers)), tiers)
    axes.set_xlim(-0.5, len(tiers) - 0.5)
    axes.grid(axis="x", visible=False)
    axes.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:+.0f}%"))
    axes.set_ylabel("change in mean")
    if len(variants) > 1:
        handles = [
            Line2D([], [], color=color, linewidth=2)
            for color in SERIES_COLORS[: len(variants)]
        ]
        axes.legend(handles, variants, fontsize=7.5)
    return _svg(figure)


def _change_panel(
    axes: Any,
    measurement: BenchmarkMeasurement,
    baseline: BenchmarkMeasurement,
    change: pd.Series,
) -> None:
    """
    Draw the bootstrap distribution of one case's change in mean, as Criterion.rs does.

    Parameters
    ----------
    axes : Any
        The matplotlib axes to draw on.
    measurement : BenchmarkMeasurement
        The case in the current run.
    baseline : BenchmarkMeasurement
        The same case in the baseline run.
    change : pd.Series
        The case's `compare_against` row, read for its change, bounds and verdict.

    Returns
    -------
    None
    """
    changes = 100.0 * bootstrap_mean_changes(
        measurement.round_seconds,
        baseline.round_seconds,
        np.random.default_rng(BOOTSTRAP_SEED),
    )
    color, _ = VERDICT_STYLES.get(change["verdict"], NEUTRAL_VERDICT_STYLE)
    axes.axvspan(
        -100 * NOISE_THRESHOLD, 100 * NOISE_THRESHOLD, color="#e1e0d9", linewidth=0
    )
    axes.hist(changes, bins=50, color=color, alpha=0.7)
    axes.axvspan(
        100 * change["relative_change_low"],
        100 * change["relative_change_high"],
        color=color,
        alpha=0.15,
        linewidth=0,
    )
    axes.axvline(0, color="#898781", linewidth=1)
    axes.axvline(100 * change["relative_change"], color=color, linewidth=2)
    axes.set_xlabel("change in mean (%)")
    axes.set_yticks([])
    axes.set_title(
        f"{VERDICT_SYMBOLS[change['verdict']]} {change['verdict']}",
        color=color,
        fontsize=9,
    )


@rc_context(CHART_STYLE)
def samples_chart(
    measurement: BenchmarkMeasurement,
    row: pd.Series,
    baseline: Optional[BenchmarkMeasurement] = None,
    change: Optional[pd.Series] = None,
) -> str:
    """
    The distribution of one case's round timings, and the timings in round order.

    Parameters
    ----------
    measurement : BenchmarkMeasurement
        The case, read for its round timings.
    row : pd.Series
        The case's `measurement_table` row, read for the median and its bounds.
    baseline : Optional[BenchmarkMeasurement]
        The same case in the baseline run, drawn as an outline over the histogram and as
        a dashed median line, or None.
    change : Optional[pd.Series]
        The case's `compare_against` row. When it carries a change, a third panel shows
        the bootstrap distribution of that change against the noise band.

    Returns
    -------
    str
        A histogram beside a round-order scatter, as inline SVG. The ink line is the
        median and the grey band its bootstrap bounds.
    """
    scale, unit = _time_unit(row["median_seconds"])
    samples = np.asarray(measurement.round_seconds) / scale
    median = row["median_seconds"] / scale
    low, high = row["median_seconds_low"], row["median_seconds_high"]

    compared = change is not None and not pd.isna(change["relative_change"])
    if compared:
        figure = Figure(figsize=(11.6, 2.4))
        histogram, in_order, moved = figure.subplots(
            1, 3, gridspec_kw={"width_ratios": [3, 2, 2.4]}
        )
        _change_panel(moved, measurement, baseline, change)
    else:
        figure = Figure(figsize=(8.8, 2.4))
        histogram, in_order = figure.subplots(
            1, 2, gridspec_kw={"width_ratios": [3, 2]}
        )
    previous = None if baseline is None else np.asarray(baseline.round_seconds) / scale
    combined = samples if previous is None else np.concatenate([samples, previous])
    # Rounds past Tukey's far fence would squash the histogram; the round-order plot
    # still shows them.
    first_quartile, third_quartile = np.percentile(combined, [25, 75])
    fence = third_quartile + 3 * (third_quartile - first_quartile)
    off_scale = int(np.sum(combined > fence))
    bins = np.histogram_bin_edges(combined[combined <= fence], bins="auto")
    histogram.hist(
        samples, bins=bins, color=SERIES_COLORS[0], rwidth=0.9, label="current"
    )
    if previous is not None:
        histogram.hist(
            previous,
            bins=bins,
            histtype="step",
            color=SERIES_COLORS[1],
            linewidth=2,
            label="baseline",
        )
        in_order.plot(
            np.arange(1, previous.size + 1),
            previous,
            linestyle="none",
            marker="o",
            markersize=3,
            markerfacecolor="none",
            color=SERIES_COLORS[1],
            alpha=0.6,
        )
        for axes, draw_line in (
            (histogram, histogram.axvline),
            (in_order, in_order.axhline),
        ):
            draw_line(
                np.median(previous),
                color=SERIES_COLORS[1],
                linewidth=1.5,
                linestyle="--",
            )
        histogram.legend(fontsize=8)
    histogram.set_xlabel(
        f"time per call ({unit})"
        + (f", {_count(off_scale, 'slow round')} off scale" if off_scale else "")
    )
    histogram.set_ylabel("rounds")
    in_order.plot(
        np.arange(1, samples.size + 1),
        samples,
        linestyle="none",
        marker="o",
        markersize=3,
        color=SERIES_COLORS[0],
        alpha=0.6,
    )
    in_order.set_xlabel("round")
    in_order.set_ylabel(unit)
    for axes, draw_line, draw_band in (
        (histogram, histogram.axvline, histogram.axvspan),
        (in_order, in_order.axhline, in_order.axhspan),
    ):
        if not (pd.isna(low) or pd.isna(high)):
            draw_band(
                low / scale, high / scale, color="#898781", alpha=0.25, linewidth=0
            )
        draw_line(median, color="#0b0b0b", linewidth=1.5)
    return _svg(figure)


def version_label(machine: Dict[str, Any]) -> str:
    """
    The measured code as release, build kind and commit, such as `v1.8.1.dev+f0ca6970`.

    The commit comes from git at run time rather than from the version string, whose
    embedded hash is fixed when the package is installed.

    Parameters
    ----------
    machine : Dict[str, Any]
        The `machine_info.macrosynergy` block of a run.

    Returns
    -------
    str
        `v` and the release, `.dev` for a development build, `+` and the first eight
        digits of the commit, and `.dirty` when the package had uncommitted changes.
    """
    version = machine.get("package_version") or (
        machine.get("library_versions") or {}
    ).get("macrosynergy", "")
    release = re.match(r"\d+(\.\d+)*", version)
    label = f"v{release.group(0)}" if release else "unknown version"
    if "dev" in version:
        label += ".dev"
    if machine.get("commit_sha"):
        label += f"+{machine['commit_sha'][:8]}"
    if machine.get("package_diff_hash"):
        label += ".dirty"
    return label


def _format_timestamp(value: Any) -> str:
    """
    A pytest-benchmark timestamp as a day and minute.

    Parameters
    ----------
    value : Any
        The payload's ISO-8601 `datetime`.

    Returns
    -------
    str
        Such as "27 Oct 2026, 13:59 UTC", or the value as given when it does not parse.
    """
    try:
        moment = datetime.fromisoformat(str(value))
    except ValueError:
        return str(value)
    zone = moment.tzname() or ""
    return f"{moment:%d %b %Y, %H:%M} {'UTC' if zone in ('UTC', 'UTC+00:00') else zone}"


def _split_target(measurement: BenchmarkMeasurement) -> Tuple[str, str]:
    """
    The module and qualified name of the callable a case measured.

    Parameters
    ----------
    measurement : BenchmarkMeasurement
        Any case of the benchmark.

    Returns
    -------
    Tuple[str, str]
        The module, empty when the run did not record a target, and the qualified name,
        falling back to the benchmark name.
    """
    if not measurement.target:
        return "", measurement.benchmark_name
    module, _, qualname = measurement.target.partition(":")
    return module, qualname


def _verdict(change: Optional[pd.Series]) -> str:
    """
    A case's verdict as symbol and change, coloured by the page.

    Parameters
    ----------
    change : Optional[pd.Series]
        The case's `compare_against` row, or None when it was not compared.

    Returns
    -------
    str
        HTML for the verdict, with the interval and p-value as a hover title; an en dash
        when the case was not compared.
    """
    if change is None:
        return "<span class=muted>–</span>"
    verdict = change["verdict"]
    if pd.isna(change["relative_change"]):
        return (
            f"<span class='verdict {verdict.replace(' ', '-')}'>"
            f"{VERDICT_SYMBOLS[verdict]} {verdict}</span>"
        )
    title = (
        f"{verdict}: 95% interval {change['relative_change_low']:+.1%} to "
        f"{change['relative_change_high']:+.1%}, p = {change['p_value']:.3f}"
    )
    return (
        f"<span class='verdict {verdict.replace(' ', '-')}' title='{html.escape(title)}'>"
        f"{VERDICT_SYMBOLS[verdict]} {change['relative_change']:+.1%}</span>"
    )


def _run_facts(run: BenchmarkRunResults) -> Dict[str, str]:
    """
    One run's provenance, as the rows of the run details table.

    Parameters
    ----------
    run : BenchmarkRunResults
        The run to describe.

    Returns
    -------
    Dict[str, str]
        Row label to plain-text value, in display order.
    """
    machine = run.machine or {}
    versions = machine.get("library_versions") or {}
    tiers = sorted(
        {m.tier: m.panel_size["observation_count"] for m in run.measurements}.items(),
        key=lambda item: item[1],
    )
    facts = {
        "Recorded": _format_timestamp(run.payload.get("datetime")),
        "Code": version_label(machine),
        "Committed": _format_timestamp(machine.get("commit_time") or "–"),
        "Uncommitted package changes": machine.get("package_diff_hash") or "none",
        "CPU": f"{machine.get('cpu_model')}, {machine.get('usable_cpu_count')} usable",
        "Memory": f"{machine.get('total_ram_gb')} GiB",
        "Operating system": str(machine.get("operating_system")),
        "Python": str(machine.get("python_version")),
    }
    facts.update(
        {name: versions.get(name, "–") for name in ("numpy", "pandas", "scipy")}
    )
    facts["BLAS"] = (
        f"{machine.get('blas_name')}, {machine.get('blas_thread_count')} threads"
    )
    facts["Tiers"] = ", ".join(tier for tier, _ in tiers)
    facts["Cases"] = str(len(run.measurements))
    facts["Hardware id"] = machine.get("hardware_id") or "–"
    facts["Runtime id"] = machine.get("software_id") or "–"
    return facts


def _statistics_table(
    row: pd.Series,
    stats: Dict[str, Any],
    previous_row: Optional[pd.Series],
    previous_stats: Optional[Dict[str, Any]],
    change: Optional[pd.Series],
) -> str:
    """
    One case's statistics, beside the baseline's when there is one.

    Parameters
    ----------
    row : pd.Series
        The case's `measurement_table` row.
    stats : Dict[str, Any]
        The case's pytest-benchmark `stats` block.
    previous_row : Optional[pd.Series]
        The baseline's row for the case, or None.
    previous_stats : Optional[Dict[str, Any]]
        The baseline's `stats` block for the case, or None.
    change : Optional[pd.Series]
        The case's `compare_against` row, or None.

    Returns
    -------
    str
        HTML for the table, ending with the change and its test when compared.
    """

    def cells(source: pd.Series, block: Dict[str, Any]) -> List[str]:
        return [
            _format_interval(source, "mean_seconds"),
            _format_interval(source, "median_seconds"),
            _format_interval(source, "stddev_seconds"),
            _format_interval(source, "median_absolute_deviation_seconds"),
            f"{block.get('rounds')} × {block.get('iterations', 1)}",
            str(block.get("iqr_outliers", "–")),
            _format_bytes(source["peak_memory_bytes"]),
            _format_rate(source["observations_per_second"]),
        ]

    labels = [
        "Mean",
        "Median",
        "Std. dev.",
        "MAD",
        "Rounds × iterations",
        "Outliers (1.5 IQR)",
        "Peak memory",
        "Throughput",
    ]
    current = cells(row, stats)
    if previous_row is None:
        body = "".join(
            f"<tr><th scope=row>{label}</th><td class=n>{value}</td></tr>"
            for label, value in zip(labels, current)
        )
        return f"<table class=stats><tbody>{body}</tbody></table>"
    before = cells(previous_row, previous_stats or {})
    body = "".join(
        f"<tr><th scope=row>{label}</th><td class=n>{old}</td><td class=n>{new}</td></tr>"
        for label, old, new in zip(labels, before, current)
    )
    if change is not None and not pd.isna(change["relative_change"]):
        body += (
            f"<tr class=change-row><th scope=row>Change in mean</th><td colspan=2 class=n>"
            f"{_verdict(change)} <span class=bounds>[{change['relative_change_low']:+.1%}, "
            f"{change['relative_change_high']:+.1%}], p = {change['p_value']:.3f}, "
            f"{html.escape(change['verdict'])}</span></td></tr>"
        )
    return (
        "<table class=stats><thead><tr><th></th><th class=n>Baseline</th>"
        f"<th class=n>Current</th></tr></thead><tbody>{body}</tbody></table>"
    )


def _call_table(measurement: BenchmarkMeasurement, row: pd.Series) -> str:
    """
    What one case called, with what, on what data.

    Parameters
    ----------
    measurement : BenchmarkMeasurement
        The case.
    row : pd.Series
        The case's `measurement_table` row.

    Returns
    -------
    str
        HTML for the table: each argument, any setup call, the data's shape, the timing
        mode and the workload id.
    """
    size = measurement.panel_size
    rows = [
        (f"<code>{html.escape(name)}</code>", html.escape(value))
        for name, value in measurement.call.items()
    ]
    for prepared, arguments in measurement.setup.items():
        rows.append(
            (
                f"<code>{html.escape(prepared)}</code>",
                "<br>".join(
                    f"<code>{html.escape(name)}</code> {html.escape(value)}"
                    for name, value in arguments.items()
                ),
            )
        )
    if size.get("cid_count"):
        shape = f"{size['cid_count']} cids × {size.get('xcat_count')} xcats"
    else:
        shape = f"{size.get('ticker_count')} tickers"
    rows += [
        (
            "Data",
            f"{size.get('df_format', '–')} format, {shape}, {size.get('date_count')} dates",
        ),
        ("Observations", f"{row['observation_count']:,}"),
        ("Timing", html.escape(row["timing_mode"])),
        ("Workload id", f"<code>{row['workload_id']}</code>"),
    ]
    body = "".join(
        f"<tr><th scope=row>{label}</th><td>{value}</td></tr>" for label, value in rows
    )
    return f"<table class=call><tbody>{body}</tbody></table>"


def _method_name(row: pd.Series) -> str:
    """
    The qualified name of the callable a case measured, for display.

    Parameters
    ----------
    row : pd.Series
        A `measurement_table` row.

    Returns
    -------
    str
        The target's qualified name, or the benchmark name for runs that recorded none.
    """
    return (row["target"] or "").partition(":")[2] or row["benchmark_name"]


def _open_case(identifier: str, text: str) -> str:
    """
    A button opening a case's detail dialog.

    Parameters
    ----------
    identifier : str
        The case's `measurement_id`.
    text : str
        HTML shown on the button.

    Returns
    -------
    str
        HTML for the button.
    """
    return (
        f"<button type=button class=open data-dialog='case-{_anchor(identifier)}'>"
        f"{text}</button>"
    )


def _changes_section(table: pd.DataFrame, changes: pd.DataFrame) -> str:
    """
    Regressed and improved cases, each in its own table, worst first.

    Parameters
    ----------
    table : pd.DataFrame
        The current run's `measurement_table`.
    changes : pd.DataFrame
        `compare_against` rows indexed by `measurement_id`.

    Returns
    -------
    str
        HTML for the section, with the decision rule and any cases left uncompared.
    """
    joined = table.set_index("measurement_id").join(changes, how="inner")

    def changes_table(verdict: str, ascending: bool, empty: str) -> str:
        cases = joined[joined["verdict"] == verdict].sort_values(
            "relative_change", ascending=ascending
        )
        if cases.empty:
            return f"<p class=empty>{empty}</p>"
        rows = "".join(
            f"<tr><td>{_open_case(identifier, f'<code>{html.escape(_method_name(row))}</code>')}"
            f"</td><td>{html.escape(row['variant']) or '–'}</td><td>{html.escape(row['tier'])}</td>"
            f"<td class=n>{_format_seconds(row['baseline_mean_seconds'])}</td>"
            f"<td class=n>{_format_seconds(row['current_mean_seconds'])}</td>"
            f"<td class=n>{_verdict(row)}</td>"
            f"<td class='n muted'>{row['relative_change_low']:+.1%} to "
            f"{row['relative_change_high']:+.1%}</td>"
            f"<td class='n muted'>{row['p_value']:.3f}</td></tr>"
            for identifier, row in cases.iterrows()
        )
        return (
            "<div class=scroll><table><thead><tr><th>Method</th><th>Variant</th>"
            "<th>Tier</th><th class=n>Baseline</th><th class=n>Current</th>"
            "<th class=n>Change</th><th class=n>95% interval</th><th class=n>p</th>"
            f"</tr></thead><tbody>{rows}</tbody></table></div>"
        )

    counts = joined["verdict"].value_counts()
    notes = []
    if counts.get("workload changed", 0):
        notes.append(
            f"{_count(counts['workload changed'], 'case')} changed workload and "
            "could not be compared."
        )
    if len(table) > len(joined):
        notes.append(f"{_count(len(table) - len(joined), 'case')} had no baseline.")
    return (
        "<section class=stack>"
        f"<div class=tile><h2>Regressions <span class=count>{counts.get('regressed', 0)}"
        "</span></h2>"
        + changes_table("regressed", False, "No case is slower than the baseline.")
        + "</div>"
        f"<div class=tile><h2>Improvements <span class=count>{counts.get('improved', 0)}"
        "</span></h2>"
        + changes_table("improved", True, "No case is faster than the baseline.")
        + "</div>"
        f"<p class=note>A case counts only when p &lt; {SIGNIFICANCE_LEVEL} and its whole 95% "
        f"interval for the change in mean lies beyond ±{NOISE_THRESHOLD:.0%}. "
        + " ".join(notes)
        + "</p></section>"
    )


def _method_tile(
    name: str, rows: pd.DataFrame, context: Dict[str, Any], compared: bool
) -> str:
    """
    One benchmarked method: a chart per chart type, and its cases as variant by tier.

    Parameters
    ----------
    name : str
        The benchmark name.
    rows : pd.DataFrame
        The current run's `measurement_table` rows for the benchmark.
    context : Dict[str, Any]
        Lookups shared across the page: `changes` and, with a baseline,
        `previous_table`.
    compared : bool
        Whether there is a baseline.

    Returns
    -------
    str
        HTML for the tile. Every chart is present; the page's switcher shows one type.
    """
    changes = context["changes"]
    baseline_rows = None
    if compared:
        previous_table = context["previous_table"]
        baseline_rows = previous_table[previous_table["benchmark_name"] == name]
    charts = "".join(
        f"<div class='chart chart-{metric}'>{metric_chart(rows, baseline_rows, metric)}</div>"
        for metric in METRIC_CHARTS
    )
    if compared:
        charts += f"<div class='chart chart-change'>{change_by_tier_chart(rows, changes)}</div>"

    tiers = list(rows.sort_values("observation_count")["tier"].drop_duplicates())
    # "dtype=categorical" reads as a "dtype" column holding "categorical".
    parameters = [
        part.partition("=")[0] for part in rows["variant"].iloc[0].split(",") if part
    ]
    body = []
    for variant, cases in rows.groupby("variant", sort=True):
        by_tier = cases.set_index("tier")
        cells = []
        for tier in tiers:
            if tier not in by_tier.index:
                cells.append("<td class='n muted'>–</td>")
                continue
            row = by_tier.loc[tier]
            change = (
                changes.loc[row["measurement_id"]]
                if row["measurement_id"] in changes.index
                else None
            )
            delta = f"<br>{_verdict(change)}" if change is not None else ""
            cells.append(
                f"<td class=n>{_open_case(row['measurement_id'], _format_seconds(row['mean_seconds']))}"
                f"{delta}</td>"
            )
        values = ", ".join(
            part.partition("=")[2] for part in variant.split(",") if part
        )
        body.append(
            f"<tr><th scope=row>{html.escape(values) or '–'}</th>{''.join(cells)}</tr>"
        )
    first = rows.iloc[0]
    module = (first["target"] or "").partition(":")[0]
    return (
        f"<article class=tile id='{_anchor('method-' + name)}'>"
        f"<h3><code title='{html.escape(module)}'>{html.escape(_method_name(first))}</code></h3>"
        f"<p class=label>{html.escape(first['label'] or '')}</p>{charts}"
        "<div class=scroll><table class=matrix><thead><tr>"
        f"<th>{html.escape(', '.join(parameters)) or 'Variant'}</th>"
        + "".join(f"<th class=n>{html.escape(tier)}</th>" for tier in tiers)
        + f"</tr></thead><tbody>{''.join(body)}</tbody></table></div></article>"
    )


def _rankings_section(table: pd.DataFrame, count: int = 5) -> str:
    """
    The fastest and slowest cases by throughput, each at its largest measured tier.

    Ranking by observations per second lets cases of different sizes be compared.

    Parameters
    ----------
    table : pd.DataFrame
        The current run's `measurement_table`.
    count : int
        How many cases each table lists.

    Returns
    -------
    str
        HTML for the two tables, side by side.
    """
    largest = (
        table.sort_values("observation_count")
        .groupby(["benchmark_name", "variant"], sort=False)
        .tail(1)
    )
    ranked = largest.dropna(subset=["observations_per_second"]).sort_values(
        "observations_per_second", ascending=False
    )

    def ranking(cases: pd.DataFrame) -> str:
        rows = "".join(
            f"<tr><td>{_open_case(row['measurement_id'], f'<code>{html.escape(_method_name(row))}</code>')}"
            f"</td><td>{html.escape(row['variant']) or '–'}</td><td>{html.escape(row['tier'])}</td>"
            f"<td class=n>{_format_seconds(row['mean_seconds'])}</td>"
            f"<td class=n>{_format_rate(row['observations_per_second'])}</td></tr>"
            for _, row in cases.iterrows()
        )
        return (
            "<div class=scroll><table><thead><tr><th>Method</th><th>Variant</th>"
            "<th>Tier</th><th class=n>Mean</th><th class=n>Throughput</th></tr></thead>"
            f"<tbody>{rows}</tbody></table></div>"
        )

    return (
        "<section class=pair>"
        f"<div class=tile><h2>Fastest</h2>{ranking(ranked.head(count))}</div>"
        f"<div class=tile><h2>Slowest</h2>{ranking(ranked.tail(count).iloc[::-1])}</div>"
        "<p class=note>Ranked by observations per second at each method and variant's "
        "largest measured tier.</p></section>"
    )


def _environment_section(
    results: BenchmarkRunResults, baseline: Optional[BenchmarkRunResults]
) -> str:
    """
    The runs' provenance, the baseline beside the current run when there is one.

    Parameters
    ----------
    results : BenchmarkRunResults
        The current run.
    baseline : Optional[BenchmarkRunResults]
        The run compared against, or None.

    Returns
    -------
    str
        HTML for the tile. Current values that differ from the baseline are bold.
    """
    current = _run_facts(results)
    previous = _run_facts(baseline) if baseline is not None else None
    rows = []
    for label, value in current.items():
        if previous is None:
            rows.append(
                f"<tr><th scope=row>{label}</th><td>{html.escape(value)}</td></tr>"
            )
            continue
        before = previous.get(label, "–")
        differs = " class=differs" if before != value and label != "Recorded" else ""
        rows.append(
            f"<tr><th scope=row>{label}</th><td>{html.escape(before)}</td>"
            f"<td{differs}>{html.escape(value)}</td></tr>"
        )
    head = (
        ""
        if previous is None
        else ("<thead><tr><th></th><th>Baseline</th><th>Current</th></tr></thead>")
    )
    note = (
        ""
        if previous is None
        else "<p class=note>Bold values differ from the baseline.</p>"
    )
    return (
        f"<section class=tile><h2>Environment</h2>{note}<div class=scroll>"
        f"<table class=environment>{head}<tbody>{''.join(rows)}</tbody></table></div></section>"
    )


def _case_dialog(identifier: str, row: pd.Series, context: Dict[str, Any]) -> str:
    """
    Everything about one case, in a dialog opened from any table that lists it.

    Parameters
    ----------
    identifier : str
        The case's `measurement_id`.
    row : pd.Series
        The case's `measurement_table` row.
    context : Dict[str, Any]
        Lookups shared across the page: `measurements`, `stats`, `changes`, and with a
        baseline `previous`, `previous_stats` and `previous_table`.

    Returns
    -------
    str
        HTML for the dialog: timing charts, statistics beside the baseline's, and the
        call and data the case measured.
    """
    measurement = context["measurements"][identifier]
    changes = context["changes"]
    change = changes.loc[identifier] if identifier in changes.index else None
    previous_table = context.get("previous_table")
    previous_row = None
    if previous_table is not None and identifier in previous_table.index:
        previous_row = previous_table.loc[identifier]
    module, qualname = _split_target(measurement)
    title = " · ".join(part for part in (qualname, row["tier"], row["variant"]) if part)
    return (
        f"<dialog id='case-{_anchor(identifier)}' aria-label='{html.escape(title)}'>"
        "<form method=dialog class=dialog-head>"
        f"<div><h2><code>{html.escape(title)}</code></h2>"
        + (f"<p class=module>{html.escape(module)}</p>" if module else "")
        + f"<p class=label>{html.escape(measurement.label or '')}</p></div>"
        "<button class=close aria-label=Close>Close</button></form>"
        + samples_chart(
            measurement, row, context.get("previous", {}).get(identifier), change
        )
        + "<div class=pair>"
        + _statistics_table(
            row,
            context["stats"][identifier],
            previous_row,
            context.get("previous_stats", {}).get(identifier),
            change,
        )
        + _call_table(measurement, row)
        + "</div></dialog>"
    )


def render_page(
    results: BenchmarkRunResults, baseline: Optional[BenchmarkRunResults] = None
) -> str:
    """
    One report for a run, compared against a baseline when one is given.

    Parameters
    ----------
    results : BenchmarkRunResults
        The run to render.
    baseline : Optional[BenchmarkRunResults]
        An earlier run on the same hardware to compare against, or None.

    Returns
    -------
    str
        A self-contained HTML document: regressions and improvements first when there
        is a baseline, then one tile per method with switchable charts and a variant by
        tier table, the fastest and slowest cases, and the environment. Each case's
        detail opens in a dialog.

    Raises
    ------
    ValueError
        If the two runs record different hardware.
    """
    machine = results.machine or {}
    table = results.measurement_table()
    tier_rank = {
        tier: rank
        for rank, tier in enumerate(
            dict.fromkeys(table.sort_values("observation_count")["tier"])
        )
    }
    table = (
        table.assign(tier_rank=table["tier"].map(tier_rank))
        .sort_values(["group", "benchmark_name", "variant", "tier_rank"])
        .drop(columns="tier_rank")
    )
    compared = baseline is not None
    context: Dict[str, Any] = {
        "measurements": results.by_identifier(),
        "stats": {
            m.identifier: entry["stats"]
            for m, entry in zip(results.measurements, results.payload["benchmarks"])
        },
        "changes": pd.DataFrame(columns=["verdict"]),
    }
    csv_rows = table
    label = version_label(machine)
    facts = [("Version", f"<code>{html.escape(label)}</code>")]
    changes_section = ""
    if compared:
        comparison = results.compare_against(baseline)
        context["changes"] = comparison.set_index("measurement_id")
        context["previous"] = baseline.by_identifier()
        context["previous_stats"] = {
            m.identifier: entry["stats"]
            for m, entry in zip(baseline.measurements, baseline.payload["benchmarks"])
        }
        context["previous_table"] = baseline.measurement_table().set_index(
            "measurement_id", drop=False
        )
        csv_rows = table.merge(
            comparison.drop(columns="current_mean_seconds"),
            on="measurement_id",
            how="left",
        )
        facts.append(
            (
                "Baseline",
                f"<code>{html.escape(version_label(baseline.machine or {}))}</code>",
            )
        )
        changes_section = _changes_section(table, context["changes"])
    facts += [
        ("Recorded", html.escape(_format_timestamp(results.payload.get("datetime")))),
        ("Cases", str(len(table))),
        ("Tiers", html.escape(", ".join(tier_rank))),
    ]

    switches = [("time", "Time"), ("throughput", "Throughput"), ("memory", "Memory")]
    if compared:
        switches.append(("change", "Change"))
    switcher = "".join(
        f"<input type=radio name=chart id=show-{key} value={key}{' checked' if index == 0 else ''}>"
        f"<label for=show-{key}>{text}</label>"
        for index, (key, text) in enumerate(switches)
    )
    tiles = "".join(
        _method_tile(name, rows, context, compared)
        for name, rows in table.groupby("benchmark_name", sort=False)
    )
    dialogs = "".join(
        _case_dialog(row["measurement_id"], row, context) for _, row in table.iterrows()
    )
    csv_link = (
        f"<a class=button download='macrosynergy-benchmarks-{html.escape(label)}.csv' "
        "href='data:text/csv;charset=utf-8,"
        f"{urllib.parse.quote(csv_rows.to_csv(index=False))}'>Download results as CSV</a>"
    )
    return PAGE_TEMPLATE.format(
        title=html.escape(f"{PAGE_TITLE} {label}"),
        heading=html.escape(PAGE_TITLE),
        facts="".join(
            f"<div><dt>{term}</dt><dd>{value}</dd></div>" for term, value in facts
        ),
        csv_link=csv_link,
        changes=changes_section,
        switcher=switcher,
        tiles=tiles,
        rankings=_rankings_section(table),
        environment=_environment_section(results, baseline),
        dialogs=dialogs,
        chart_rules="\n".join(
            f"body:has(#show-{key}:checked) .chart-{key} {{ display: block; }}"
            for key, _ in switches
        ),
    )


PAGE_TEMPLATE = """<!doctype html>
<html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
:root {{
  color-scheme: light;
  --page: #f5f6f8; --surface: #ffffff; --border: #e1e4e8; --ink: #1f2328;
  --ink-secondary: #59636e; --ink-muted: #818b98; --gridline: #eaedf0; --axis: #c8ced5;
  --series-1: #2a78d6; --series-2: #eb6834; --series-3: #1baf7a;
  --good: #1a7f37; --critical: #cf222e; --control: #f0f2f4;
  --sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  --mono: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
}}
@media (prefers-color-scheme: dark) {{ :root {{
  color-scheme: dark;
  --page: #0d1117; --surface: #151b23; --border: #2a313c; --ink: #e6edf3;
  --ink-secondary: #9da7b3; --ink-muted: #7d8590; --gridline: #232a33; --axis: #3d444d;
  --series-1: #3987e5; --series-2: #d95926; --series-3: #199e70;
  --good: #3fb950; --critical: #f85149; --control: #212830;
}} }}
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: var(--page); color: var(--ink); font: 14px/1.5 var(--sans); }}
.page {{ max-width: 1240px; margin: 0 auto; padding: 28px 16px 56px; }}
code {{ font-family: var(--mono); font-size: 0.92em; }}
header.top {{ display: flex; flex-wrap: wrap; gap: 12px 24px; align-items: flex-end;
  justify-content: space-between; margin-bottom: 20px; }}
h1 {{ font-size: 22px; font-weight: 600; margin: 0 0 8px; }}
dl.facts {{ display: flex; flex-wrap: wrap; gap: 4px 24px; margin: 0; }}
dl.facts div {{ display: flex; gap: 6px; }}
dl.facts dt {{ color: var(--ink-secondary); }}
dl.facts dd {{ margin: 0; }}
.button {{ display: inline-block; padding: 6px 14px; border: 1px solid var(--border);
  border-radius: 6px; background: var(--surface); color: var(--ink); text-decoration: none;
  font-weight: 500; }}
.button:hover {{ background: var(--control); }}
h2 {{ font-size: 15px; font-weight: 600; margin: 0 0 10px; }}
h3 {{ font-size: 14px; font-weight: 600; margin: 0; }}
.count {{ display: inline-block; min-width: 22px; padding: 0 7px; margin-left: 4px;
  border-radius: 10px; background: var(--control); color: var(--ink-secondary);
  font-size: 12px; text-align: center; }}
.tile {{ background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
  padding: 16px; min-width: 0; }}
.pair {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
  gap: 16px; margin-bottom: 24px; align-items: start; }}
.pair > .note {{ grid-column: 1 / -1; margin: -6px 0 0; }}
.stack {{ display: grid; gap: 16px; margin-bottom: 24px; }}
.stack > .note {{ margin: -6px 0 0; }}
.note, .empty, .label, .module {{ color: var(--ink-secondary); }}
.note {{ font-size: 12px; }}
.empty {{ margin: 0; }}
.label {{ margin: 2px 0 8px; font-size: 13px; min-height: 2.6em; }}
.label:empty {{ min-height: 0; margin: 0 0 8px; }}
.module {{ margin: 0; font-family: var(--mono); font-size: 12px; }}
.toolbar {{ display: flex; align-items: center; justify-content: space-between;
  margin: 8px 0 12px; }}
.switcher {{ display: inline-flex; border: 1px solid var(--border); border-radius: 6px;
  overflow: hidden; background: var(--surface); }}
.switcher input {{ position: absolute; opacity: 0; pointer-events: none; }}
.switcher label {{ padding: 5px 14px; cursor: pointer; color: var(--ink-secondary);
  border-left: 1px solid var(--border); }}
.switcher label:first-of-type {{ border-left: none; }}
.switcher input:checked + label {{ background: var(--control); color: var(--ink); font-weight: 600; }}
.switcher input:focus-visible + label {{ outline: 2px solid var(--series-1); outline-offset: -2px; }}
.tiles {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(360px, 1fr));
  gap: 16px; margin-bottom: 24px; }}
.chart {{ display: none; margin: 0 -4px 8px; }}
{chart_rules}
svg {{ display: block; max-width: 100%; height: auto; }}
svg text {{ font-family: var(--sans) !important; }}
.scroll {{ overflow-x: auto; }}
table {{ width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; }}
th, td {{ padding: 5px 8px; border-bottom: 1px solid var(--border); text-align: left;
  vertical-align: top; }}
th:first-child, td:first-child {{ padding-left: 0; }}
th:last-child, td:last-child {{ padding-right: 0; }}
thead th {{ color: var(--ink-secondary); font-weight: 500; font-size: 12px; }}
tbody th {{ font-weight: 400; color: var(--ink-secondary); }}
tbody tr:last-child > * {{ border-bottom: none; }}
.n {{ text-align: right; white-space: nowrap; }}
.muted {{ color: var(--ink-muted); }}
.bounds {{ color: var(--ink-muted); font-size: 12px; }}
.matrix {{ font-size: 13px; }}
.matrix th, .matrix td {{ padding: 4px 5px; line-height: 1.35; }}
.matrix .verdict {{ font-size: 11px; }}
.open {{ font: inherit; color: var(--series-1); background: none; border: none; padding: 0;
  cursor: pointer; text-align: inherit; }}
.open:hover {{ text-decoration: underline; }}
.open:focus-visible, .close:focus-visible {{ outline: 2px solid var(--series-1); outline-offset: 2px; }}
.verdict {{ font-size: 12px; white-space: nowrap; }}
.verdict.regressed {{ color: var(--critical); font-weight: 600; }}
.verdict.improved {{ color: var(--good); font-weight: 600; }}
.verdict.within-noise, .verdict.no-change, .verdict.workload-changed {{ color: var(--ink-muted); }}
.environment td {{ min-width: 16ch; }}
.environment td.differs {{ font-weight: 600; }}
dialog {{ width: min(1180px, calc(100vw - 32px)); max-height: calc(100vh - 32px);
  padding: 20px; border: 1px solid var(--border); border-radius: 10px;
  background: var(--surface); color: var(--ink); }}
dialog::backdrop {{ background: rgba(15, 20, 25, 0.45); }}
.dialog-head {{ display: flex; justify-content: space-between; gap: 16px; margin-bottom: 8px; }}
.dialog-head h2 {{ margin: 0; }}
.close {{ align-self: flex-start; font: inherit; padding: 4px 12px; border: 1px solid var(--border);
  border-radius: 6px; background: var(--surface); color: var(--ink); cursor: pointer; }}
dialog .pair {{ margin: 12px 0 0; }}
.change-row th, .change-row td {{ border-bottom: none; }}
.call td {{ overflow-wrap: anywhere; }}
@media (max-width: 520px) {{ .pair {{ grid-template-columns: 1fr; }} }}
</style></head><body><div class=page>
<header class=top>
<div><h1>{heading}</h1><dl class=facts>{facts}</dl></div>
{csv_link}
</header>
<main>
{changes}
<div class=toolbar><h2 id=methods>Methods</h2>
<div class=switcher role=radiogroup aria-label="Chart">{switcher}</div></div>
<section class=tiles aria-labelledby=methods>{tiles}</section>
{rankings}
{environment}
</main>
{dialogs}
</div>
<script>
document.addEventListener("click", (event) => {{
  const opener = event.target.closest("[data-dialog]");
  if (opener) {{
    document.getElementById(opener.dataset.dialog).showModal();
  }} else if (event.target.tagName === "DIALOG") {{
    event.target.close();
  }}
}});
</script>
</body></html>
"""


def main(argv: Optional[Sequence[str]] = None) -> int:
    """
    Enrich one results file in place and write its page beside it, optionally
    comparing against a baseline run.

    Parameters
    ----------
    argv : Optional[Sequence[str]]
        Command-line arguments, or None to read `sys.argv`.

    Returns
    -------
    int
        Zero, the process exit status.

    Raises
    ------
    ValueError
        If an entry's `extra_info` is incomplete, two measurements share an identifier,
        or the baseline ran on different hardware.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("path", type=Path, help="JSON written by --benchmark-json")
    parser.add_argument(
        "--baseline", type=Path, help="an earlier run's JSON to compare against"
    )
    arguments = parser.parse_args(argv)

    results = BenchmarkRunResults.load(arguments.path)
    baseline = (
        BenchmarkRunResults.load(arguments.baseline) if arguments.baseline else None
    )
    page = render_page(results, baseline)
    arguments.path.write_text(json.dumps(results.with_derived_statistics(), indent=2))
    arguments.path.with_suffix(".html").write_text(page, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
