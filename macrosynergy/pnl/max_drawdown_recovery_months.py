"""Max Draw Recovery (months): time from the peak preceding a series' worst
drawdown to its full recovery.
"""

import numpy as np
import pandas as pd


def max_drawdown_recovery_months(cum_pnl: pd.Series, return_ongoing: bool = False):
    """
    Trading days from the peak preceding the worst drawdown to its full
    recovery, expressed in months of 21 trading days -- the same convention
    used for "Max 21-Day Draw %" elsewhere in ``evaluate_pnls()``.

    Parameters
    ----------
    cum_pnl : pd.Series
        Cumulative PnL series (e.g. ``dfw[col].cumsum()``). NaNs are dropped.
    return_ongoing : bool, default False
        If True, return a ``(months, ongoing)`` tuple instead of just
        ``months``, where ``ongoing`` is True when the worst drawdown
        hadn't recovered by the end of the sample -- i.e. ``months``
        reflects an open, still-running drawdown rather than a completed
        recovery.

    Returns
    -------
    float, or (float, bool) if ``return_ongoing``
        Recovery time in months. 0 if the series never had a drawdown at
        all. If the worst drawdown hasn't recovered by the end of the
        sample, the elapsed time from its preceding peak to the last
        observation -- i.e. how long that drawdown has been running so
        far, not the length of the whole trading history. NaN if
        ``cum_pnl`` has no non-NaN observations.
    """
    s = cum_pnl.dropna()
    if s.empty:
        months, ongoing = float("nan"), False
    else:
        high_watermark = s.cummax()
        drawdown = high_watermark - s
        trough_pos = int(np.argmax(drawdown.values))
        if drawdown.iloc[trough_pos] == 0:
            months, ongoing = 0.0, False
        else:
            peak_level = high_watermark.iloc[trough_pos]
            peak_pos = int(np.where(s.values[: trough_pos + 1] == peak_level)[0][-1])

            recovered = np.where(s.values[trough_pos:] >= peak_level)[0]
            if len(recovered) == 0:
                months, ongoing = (len(s) - 1 - peak_pos) / 21, True
            else:
                recovery_pos = trough_pos + int(recovered[0])
                months, ongoing = (recovery_pos - peak_pos) / 21, False

    return (months, ongoing) if return_ongoing else months
