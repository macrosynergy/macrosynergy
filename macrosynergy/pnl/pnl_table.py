"""Presentation-ready HTML rendering of an ``evaluate_pnls()`` statistics
table -- grouped sections, tabular-number alignment, no computation. Kept
separate from ``NaivePnL`` so the stats/rendering concerns stay decoupled;
``NaivePnL.evaluate_pnls_pretty`` is the thin wrapper that calls
``evaluate_pnls`` and hands its output to ``pnl_table_html``.
"""

from typing import Dict, List, Sequence, Tuple, Union

import pandas as pd

DEFAULT_METRIC_GROUPS: Dict[str, List[str]] = {
    "Performance": ["Return %", "Sharpe Ratio", "Sortino Ratio"],
    "Risk & drawdown": [
        "St. Dev. %",
        "Peak to Trough Draw %",
        "Top 5% Monthly PnL Share",
        "Benchmark correlation",
        "Max Draw Recovery (months)",
    ],
    "Robustness": [
        "Sharpe Stability Ratio",
        "Prob. Sharpe Ratio > 0.25",
        "Prob. Sharpe Ratio > 0.5",
        "Prob. Sharpe Ratio > 0.75",
        "Traded Months",
    ],
}

# INK matches the macrosynergy package's default LinePlot/view_timeseries
# color: Plotter.__init__ calls sns.set_theme(style="darkgrid",
# palette="colorblind"), and seaborn's "colorblind" palette's first swatch is
# #0173b2 -- the blue every unstyled macrosynergy time series chart uses.
_INK, _INK2, _MUTED = "#0174b2c6", "#52514e", "#898781"
_BAND, _ZEBRA = "#f4f3ef", "#faf9f6"
_NUM_FIELD_W = "7ch"  # 4ch integer slot + 3ch fraction slot

# matplotlib never sets a custom typeface in this package's plotting code
# (only ever *sizes*, never *family*), so every chart inherits matplotlib's
# bundled default, DejaVu Sans. Matched here, with a system-font fallback for
# renderers (e.g. browsers) that don't have DejaVu Sans registered.
_TEXT_FONT = "'DejaVu Sans', system-ui, -apple-system, 'Segoe UI', sans-serif"
_MONO_FONT = "'DejaVu Sans Mono', ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"


def _fmt(metric: str, v: float, whole_number_metrics: Sequence[str]) -> str:
    if v != v:  # NaN
        return ""
    return f"{v:,.0f}" if metric in whole_number_metrics else f"{v:,.2f}"


def _decimal_aligned(s: str) -> str:
    # Decimal values: split "int.frac" so the "." lands in a fixed column
    # across every row, the way a LaTeX `&`-aligned tabular column would.
    # Whole numbers (no ".") have no fraction slot to sit in, so instead
    # they're right-aligned across the *whole* field -- landing their last
    # digit on the same column as the last fractional digit of the decimal
    # rows, rather than at the (nonexistent) decimal point.
    int_part, dot, frac_part = s.partition(".")
    if not dot:
        return (
            f'<span style="display:inline-block;width:{_NUM_FIELD_W};'
            f'text-align:right;">{int_part}</span>'
        )
    return (
        f'<span style="display:inline-grid;grid-template-columns:4ch 3ch;'
        f'width:{_NUM_FIELD_W};text-align:left;">'
        f'<span style="text-align:right;">{int_part}</span>'
        f'<span style="text-align:left;">.{frac_part}</span>'
        f"</span>"
    )


def pnl_table_html(
    tbr: pd.DataFrame,
    groups: List[Tuple[str, List[str]]],
    headlines: List[str],
    bench: Union[str, List[str]],
    headline_labels: List[str],
    bench_label: Union[str, List[str]],
    whole_number_metrics: Sequence[str] = (),
    title: str = "Naive PnL statistics",
    subtitle: str = "",
) -> str:
    """Render a grouped ledger-style HTML table from an already-computed
    ``evaluate_pnls()`` DataFrame. Every strategy column (however many there
    are) shares one identical color and weight -- they're all "the compared
    time series" -- and every benchmark column (however many there are) is
    visually distinct (muted), regardless of how many strategy columns
    there are.

    Parameters
    ----------
    tbr : pd.DataFrame
        Output of ``evaluate_pnls()`` (or a renamed copy of it), indexed by
        metric name, columns are PnL categories.
    groups : list of (section title, [metric names])
        Row layout and order. Every metric referenced must be a row in
        ``tbr``.
    headlines : list of str
        ``tbr`` columns to show as the (non-benchmark) strategy columns, in
        display order.
    bench : str or list of str
        ``tbr`` column(s) to show as the benchmark (muted) column(s), in
        display order.
    headline_labels, bench_label : list of str, str or list of str
        Column header labels, matching ``headlines`` 1:1, and ``bench``
        1:1 (a bare str is only valid when ``bench`` is a single column).
    whole_number_metrics : sequence of str, default ()
        Metric names formatted with no decimal places (e.g. "Traded
        Months") instead of two.
    title, subtitle : str
        Header text shown above the table.

    Returns
    -------
    str
        A self-contained HTML string.
    """
    benches = bench if isinstance(bench, list) else [bench]
    bench_labels = bench_label if isinstance(bench_label, list) else [bench_label]
    cols = [*headlines, *benches]
    col_colors = [_INK2] * len(headlines) + [_MUTED] * len(benches)

    def num_cell(metric, col, color):
        v = tbr.loc[metric, col]
        aligned = _decimal_aligned(_fmt(metric, v, whole_number_metrics))
        return (
            f'<td style="padding:8px 20px;text-align:center;font-size:14.5px;'
            f"font-family:{_MONO_FONT};"
            f'font-variant-numeric:tabular-nums;color:{color};font-weight:400;">{aligned}</td>'
        )

    rows = []
    row_i = 0
    n_cols = len(cols) + 1
    for group, metrics in groups:
        rows.append(
            f'<tr><td colspan="{n_cols}" style="padding:9px 20px;font-size:11px;letter-spacing:.08em;'
            f'text-transform:uppercase;color:#fff;font-weight:700;background:{_INK};text-align:left;">{group}</td></tr>'
        )
        for m in metrics:
            bg = _ZEBRA if row_i % 2 == 0 else "#ffffff"
            cells = "".join(
                num_cell(m, col, color) for col, color in zip(cols, col_colors)
            )
            rows.append(
                f'<tr style="background:{bg};"><td style="padding:8px 20px;text-align:left;'
                f'font-size:13.5px;color:{_INK2};">{m}</td>{cells}</tr>'
            )
            row_i += 1

    header_labels = [*headline_labels, *bench_labels]
    header_cells = "".join(
        f'<th style="text-align:center;padding:8px 20px;font-size:11.5px;letter-spacing:.04em;'
        f'text-transform:uppercase;color:{_INK};font-weight:700;border-bottom:1px solid {_INK};">{lbl}</th>'
        for lbl in header_labels
    )

    return f"""<div style="font-family:{_TEXT_FONT};
    max-width:640px;background:#fff;border:1px solid {_INK};border-radius:2px;overflow:hidden;">
  <div style="padding:16px 20px 2px 20px;font-size:16px;font-weight:700;color:{_INK};">
    {title}</div>
  <div style="padding:0 20px 12px 20px;font-size:12px;color:{_MUTED};">
    {subtitle}</div>
  <table style="border-collapse:collapse;width:100%;border-top:2px solid {_INK};">
    <thead><tr style="background:{_BAND};">
      <th style="text-align:left;padding:8px 20px;font-size:11.5px;letter-spacing:.04em;
          text-transform:uppercase;color:{_MUTED};border-bottom:1px solid {_INK};"></th>
      {header_cells}
    </tr></thead>
    <tbody>{"".join(rows)}</tbody>
  </table>
</div>"""
