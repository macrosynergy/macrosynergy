"""Max Draw Recovery (months): time from the peak preceding a series' worst
drawdown to its full recovery.
"""

import numpy as np
import pandas as pd


def max_drawdown_recovery_months(cum_pnl: pd.Series, traded_months: float) -> float:
    """
    Trading days from the peak preceding the worst drawdown to its full
    recovery, expressed in months of 21 trading days -- the same convention
    used for "Max 21-Day Draw %" elsewhere in ``evaluate_pnls()``.

    Parameters
    ----------
    cum_pnl : pd.Series
        Cumulative PnL series (e.g. ``dfw[col].cumsum()``). NaNs are dropped.
    traded_months : float
        Total traded months for this series (e.g. the "Traded Months" row of
        ``evaluate_pnls()``), used as the fallback value below.

    Returns
    -------
    float
        Recovery time in months. 0 if the series never had a drawdown at
        all. ``traded_months`` (rather than an open-ended NaN) if the worst
        drawdown hasn't recovered by the end of the sample -- i.e. it's
        taken at least the whole traded history to (not yet) recover. NaN
        if ``cum_pnl`` has no non-NaN observations.
    """
    s = cum_pnl.dropna()
    if s.empty:
        return float("nan")

    high_watermark = s.cummax()
    drawdown = high_watermark - s
    trough_pos = int(np.argmax(drawdown.values))
    if drawdown.iloc[trough_pos] == 0:
        return 0.0

    peak_level = high_watermark.iloc[trough_pos]
    peak_pos = int(np.where(s.values[: trough_pos + 1] == peak_level)[0][-1])

    recovered = np.where(s.values[trough_pos:] >= peak_level)[0]
    if len(recovered) == 0:
        return traded_months
    recovery_pos = trough_pos + int(recovered[0])

    return (recovery_pos - peak_pos) / 21
