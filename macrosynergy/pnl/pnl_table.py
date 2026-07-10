"""Presentation-ready HTML rendering of an ``evaluate_pnls()`` statistics
table -- grouped sections, tabular-number alignment, no computation. Kept
separate from ``NaivePnL`` so the stats/rendering concerns stay decoupled;
``NaivePnL.evaluate_pnls_pretty`` is the thin wrapper that calls
``evaluate_pnls`` and hands its output to ``pnl_table_html``.
"""

from typing import Dict, List, Optional, Sequence, Tuple, Union

import pandas as pd

DEFAULT_METRIC_GROUPS: Dict[str, List[str]] = {
    "Performance": ["Return %", "Sharpe Ratio", "Sortino Ratio"],
    "Risk & drawdown": [
        "St. Dev. %",
        "Peak to Trough Draw %",
        "Max Draw Recovery (months)",
        "Benchmark correlation",
    ],
    "Robustness": [
        "Sharpe Stability Ratio",
        "Prob. Sharpe Ratio > 0.25",
        "Prob. Sharpe Ratio > 0.5",
        "Prob. Sharpe Ratio > 0.75",
        "Top 5% Monthly PnL Share",
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

# Every visual rule lives here, addressed by class, so a caller-supplied
# `custom_css` block -- injected in a second <style> tag right after this one
# -- can override any of it: same selector, later in source order, wins.
_DEFAULT_CSS = f"""
.pnl-wrap {{ font-family:{_TEXT_FONT}; width:fit-content; min-width:640px; max-width:100%;
  background:#fff; border:1px solid {_INK}; border-radius:2px; overflow:hidden; }}
.pnl-title {{ padding:16px 20px 2px 20px; font-size:16px; font-weight:700; color:{_INK}; }}
.pnl-subtitle {{ padding:0 20px 12px 20px; font-size:12px; color:{_MUTED}; }}
.pnl-table {{ border-collapse:collapse; width:100%; border-top:2px solid {_INK}; }}
.pnl-thead-row {{ background:{_BAND}; }}
.pnl-th-corner {{ text-align:left; padding:8px 20px; font-size:11.5px; letter-spacing:.04em;
  text-transform:uppercase; color:{_MUTED}; border-bottom:1px solid {_INK}; white-space:nowrap; }}
.pnl-th {{ text-align:center; padding:8px 20px; font-size:11.5px; letter-spacing:.04em;
  text-transform:uppercase; color:{_INK}; font-weight:700; border-bottom:1px solid {_INK};
  white-space:nowrap; }}
.pnl-group-cell {{ padding:9px 20px; font-size:11px; letter-spacing:.08em;
  text-transform:uppercase; color:#fff; font-weight:700; background:{_INK}; text-align:left; }}
.pnl-row-even {{ background:{_ZEBRA}; }}
.pnl-row-odd {{ background:#ffffff; }}
.pnl-label {{ padding:8px 20px; text-align:left; font-size:13.5px; color:{_INK2};
  white-space:nowrap; }}
.pnl-value {{ padding:8px 20px; text-align:center; font-size:14.5px; font-family:{_MONO_FONT};
  font-variant-numeric:tabular-nums; font-weight:400; white-space:nowrap; }}
.pnl-value-strategy {{ color:{_INK2}; }}
.pnl-value-bench {{ color:{_MUTED}; }}
.pnl-num {{ display:inline-block; width:{_NUM_FIELD_W}; text-align:right; }}
.pnl-num-split {{ display:inline-grid; grid-template-columns:4ch 3ch; width:{_NUM_FIELD_W};
  text-align:left; }}
.pnl-num-int {{ text-align:right; }}
.pnl-num-frac {{ text-align:left; }}
.pnl-footnote {{ padding:8px 20px 14px 20px; font-size:11px; color:{_MUTED}; }}
""".strip()


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
        return f'<span class="pnl-num">{int_part}</span>'
    return (
        f'<span class="pnl-num-split">'
        f'<span class="pnl-num-int">{int_part}</span>'
        f'<span class="pnl-num-frac">.{frac_part}</span>'
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
    plus_prefix: Dict[str, Sequence[str]] = None,
    title: str = "Naive PnL statistics",
    subtitle: str = "",
    custom_css: Optional[str] = None,
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
    plus_prefix : dict of {row name: [column names]}, optional
        Cells to prefix with "+" (e.g. an unrecovered drawdown still in
        progress). A footnote explaining the "+" is appended below the
        table whenever this marks at least one cell.
    title, subtitle : str
        Header text shown above the table.
    custom_css : str, optional
        Raw CSS injected in a second ``<style>`` block, right after the
        table's own default stylesheet -- same-specificity selectors win by
        source order, so rules targeting the classes below override the
        defaults without needing ``!important``:

        - ``.pnl-wrap`` -- outer container (font, border, background).
          Sizes to fit its content (``width:fit-content``) between a
          640px minimum and its embedding page's width as a maximum, so
          extra columns or long labels widen the box rather than being
          clipped or forcing a scrollbar -- keeping the whole table
          selectable in one paste, e.g. into Word.
        - ``.pnl-title``, ``.pnl-subtitle`` -- header text above the table
        - ``.pnl-table`` -- the ``<table>`` element itself
        - ``.pnl-thead-row``, ``.pnl-th-corner``, ``.pnl-th`` -- header row
          background and column-header cells
        - ``.pnl-group-cell`` -- section title bars (e.g. "Risk & drawdown")
        - ``.pnl-row-even``, ``.pnl-row-odd`` -- zebra-striped data rows
        - ``.pnl-label`` -- the metric-name cell of a data row
        - ``.pnl-value``, ``.pnl-value-strategy``, ``.pnl-value-bench`` --
          numeric cells; the latter two color strategy vs. benchmark
          columns differently
        - ``.pnl-num``, ``.pnl-num-split``, ``.pnl-num-int``,
          ``.pnl-num-frac`` -- decimal-point alignment of the numbers
          themselves
        - ``.pnl-footnote`` -- the "+" explanation footnote, if shown

    Returns
    -------
    str
        A self-contained HTML string.
    """
    benches = bench if isinstance(bench, list) else [bench]
    bench_labels = bench_label if isinstance(bench_label, list) else [bench_label]
    cols = [*headlines, *benches]
    col_classes = ["pnl-value-strategy"] * len(headlines) + ["pnl-value-bench"] * len(
        benches
    )
    plus_prefix = plus_prefix or {}
    used_plus = False

    def num_cell(metric, col, value_class):
        nonlocal used_plus
        v = tbr.loc[metric, col]
        fmt = _fmt(metric, v, whole_number_metrics)
        if fmt and col in plus_prefix.get(metric, ()):
            used_plus = True
            fmt = "+" + fmt
        aligned = _decimal_aligned(fmt)
        return f'<td class="pnl-value {value_class}">{aligned}</td>'

    rows = []
    row_i = 0
    n_cols = len(cols) + 1
    for group, metrics in groups:
        rows.append(
            f'<tr><td colspan="{n_cols}" class="pnl-group-cell">{group}</td></tr>'
        )
        for m in metrics:
            row_class = "pnl-row-even" if row_i % 2 == 0 else "pnl-row-odd"
            cells = "".join(
                num_cell(m, col, value_class)
                for col, value_class in zip(cols, col_classes)
            )
            rows.append(
                f'<tr class="{row_class}"><td class="pnl-label">{m}</td>{cells}</tr>'
            )
            row_i += 1

    header_labels = [*headline_labels, *bench_labels]
    header_cells = "".join(
        f'<th class="pnl-th">{lbl}</th>' for lbl in header_labels
    )

    footnote = (
        '<div class="pnl-footnote">'
        "+ drawdown had not yet recovered as of the last observation</div>"
        if used_plus
        else ""
    )

    custom_style = f"<style>{custom_css}</style>" if custom_css else ""

    return f"""<style>{_DEFAULT_CSS}</style>
{custom_style}
<div class="pnl-wrap">
  <div class="pnl-title">{title}</div>
  <div class="pnl-subtitle">{subtitle}</div>
  <table class="pnl-table">
    <thead><tr class="pnl-thead-row">
      <th class="pnl-th-corner"></th>
      {header_cells}
    </tr></thead>
    <tbody>{"".join(rows)}</tbody>
  </table>
  {footnote}
</div>"""
