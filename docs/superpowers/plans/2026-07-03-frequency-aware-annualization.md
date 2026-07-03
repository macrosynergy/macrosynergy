# Frequency-Aware Annualization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the static, ticker-name-derived √(1/freq) annualization with a per-observation, time-varying weight inferred from actual release cadence (`eop` spacing), robust to intra-series frequency structural breaks (e.g. Australia CPI Q→M).

**Architecture:** A pure `infer_release_frequency(eop, window, freqs)` helper classifies each observation's release frequency from the rolling-median gap between distinct `eop` dates, snapped to the nearest standard frequency. A QDF transform `annualize_by_release_frequency` calls it per `(cid, xcat)` group and multiplies each value by `√(1 / ANNUALIZATION_FACTORS[freq])`.

**Tech Stack:** Python, pandas, numpy, macrosynergy (`ANNUALIZATION_FACTORS`, `reduce_df`, `QuantamentalDataFrame`), pytest.

## Global Constraints

- Repo: `macrosynergy`. Branch: `feature/infra` (base `develop`). Spec: `docs/superpowers/specs/2026-07-03-frequency-aware-annualization-design.md`.
- Estimator: **rolling-median eop gap → snap to nearest standard frequency** (window default 3, `min_periods=1`). Accepted trade-off: genuine breaks lock in ~1–2 releases after the true break.
- Reuse `ANNUALIZATION_FACTORS` from `macrosynergy.management.constants` — **no hard-coded 4/12**.
- Weight per observation: `sqrt(1 / ANNUALIZATION_FACTORS[freq])`.
- `eop` sourcing: **require an `eop` column** (ISC-sourced). If absent, raise a clear `ValueError` — do NOT reconstruct from `eop_lag`.
- Snap by nearest reference period-length in days, compared in **log space** (ratio-symmetric boundaries). Reference days = `365.25 / ANNUALIZATION_FACTORS[freq]`.
- Supported frequency set default: `("D", "W", "M", "Q", "A")`.
- Output `value`-only QDF; annualized categories get a `postfix` (default `"A"`); preserve categorical dtype.

## File Structure

- Create: `macrosynergy/management/utils/frequency.py` — pure `infer_release_frequency` helper (no QDF/ISC dependency).
- Modify: `macrosynergy/management/utils/__init__.py` — export `infer_release_frequency`.
- Create: `macrosynergy/panel/annualize_by_release_frequency.py` — the QDF transform.
- Modify: `macrosynergy/panel/__init__.py` — export `annualize_by_release_frequency`.
- Create: `tests/unit/management/test_frequency.py` — helper tests.
- Create: `tests/unit/panel/test_annualize_by_release_frequency.py` — transform tests.

---

### Task 1: `infer_release_frequency` helper

**Files:**
- Create: `macrosynergy/management/utils/frequency.py`
- Modify: `macrosynergy/management/utils/__init__.py`
- Test: `tests/unit/management/test_frequency.py`

**Interfaces:**
- Consumes: `ANNUALIZATION_FACTORS` (`macrosynergy.management.constants`).
- Produces: `infer_release_frequency(eop, window=3, freqs=("D","W","M","Q","A")) -> pd.Series` — input is a per-observation `eop` datetime Series (index preserved); output is a per-observation frequency-label Series aligned to the input index.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/management/test_frequency.py
import pandas as pd
import pytest

from macrosynergy.management.utils import infer_release_frequency


def _eop_series(dates):
    d = pd.to_datetime(dates)
    return pd.Series(d, index=range(len(d)))


def test_pure_monthly_and_quarterly():
    monthly = _eop_series(pd.date_range("2020-01-31", periods=12, freq="ME"))
    quarterly = _eop_series(pd.date_range("2020-03-31", periods=8, freq="QE"))
    assert (infer_release_frequency(monthly) == "M").all()
    assert (infer_release_frequency(quarterly) == "Q").all()
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/unit/management/test_frequency.py::test_pure_monthly_and_quarterly -v`
Expected: FAIL with `ImportError: cannot import name 'infer_release_frequency'`.

- [ ] **Step 3: Write minimal implementation**

```python
# macrosynergy/management/utils/frequency.py
"""
Infer per-observation release frequency from the spacing of end-of-period (eop) dates.
"""
from typing import Sequence, Tuple

import numpy as np
import pandas as pd

from macrosynergy.management.constants import ANNUALIZATION_FACTORS


def _reference_days(freqs: Sequence[str]) -> dict:
    # Calendar days per period, derived from periods-per-year. e.g. Q -> 365.25/4.
    return {f: 365.25 / ANNUALIZATION_FACTORS[f] for f in freqs}


def infer_release_frequency(
    eop: pd.Series,
    window: int = 3,
    freqs: Tuple[str, ...] = ("D", "W", "M", "Q", "A"),
) -> pd.Series:
    """
    Classify the release frequency of each observation from its local ``eop`` cadence.

    The gap (in days) between consecutive *distinct* eop dates is smoothed with a rolling
    median (``window``, ``min_periods=1``) and snapped to the nearest supported frequency
    by log-distance to the reference period length (``365.25 / ANNUALIZATION_FACTORS``).
    Observations sharing an eop (revisions) inherit that period's frequency.

    Parameters
    ----------
    eop : pd.Series
        per-observation end-of-period dates (datetime); the index is preserved.
    window : int
        rolling-median window over distinct-eop gaps. Default 3.
    freqs : Tuple[str, ...]
        candidate frequency labels. Default ("D", "W", "M", "Q", "A").

    Returns
    -------
    pd.Series
        per-observation frequency labels, aligned to the input index.
    """
    eop = pd.to_datetime(eop)
    ref = _reference_days(freqs)
    log_ref = {f: np.log(d) for f, d in ref.items()}

    # Distinct, sorted eop periods and their smoothed gaps (in days).
    distinct = pd.Series(sorted(pd.unique(eop.dropna())))
    if len(distinct) == 0:
        return pd.Series(index=eop.index, dtype=object)
    gaps = distinct.diff().dt.days
    # Seed the first gap with the first observed gap (min_periods=1 covers the rest).
    gaps.iloc[0] = gaps.dropna().iloc[0] if gaps.dropna().size else np.nan
    smoothed = gaps.rolling(window=window, min_periods=1).median()

    def _snap(g):
        if pd.isna(g) or g <= 0:
            return freqs[0]
        lg = np.log(g)
        return min(freqs, key=lambda f: abs(lg - log_ref[f]))

    period_freq = {d: _snap(g) for d, g in zip(distinct, smoothed)}
    return eop.map(period_freq)
```

Then modify `macrosynergy/management/utils/__init__.py`: add `from .frequency import infer_release_frequency` and include it in `__all__`.

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/unit/management/test_frequency.py::test_pure_monthly_and_quarterly -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add macrosynergy/management/utils/frequency.py macrosynergy/management/utils/__init__.py tests/unit/management/test_frequency.py
git commit -m "feat(utils): add infer_release_frequency (eop-cadence classifier)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Structural break + jitter robustness of the classifier

**Files:**
- Test: `tests/unit/management/test_frequency.py`

**Interfaces:**
- Consumes: `infer_release_frequency` (Task 1).

- [ ] **Step 1: Write the failing tests**

```python
def test_quarterly_to_monthly_break():
    # AUD-CPI-like: 8 quarterly eops then 12 monthly eops in one series.
    q = pd.date_range("2015-03-31", periods=8, freq="QE")
    m = pd.date_range(q[-1] + pd.offsets.MonthEnd(1), periods=12, freq="ME")
    s = _eop_series(list(q) + list(m))
    labels = infer_release_frequency(s, window=3)

    # Early observations are quarterly.
    assert (labels.iloc[:6] == "Q").all()
    # Late observations are monthly (allow ~1-2 releases of lag at the break).
    assert (labels.iloc[-4:] == "M").all()


def test_one_off_irregular_gap_does_not_flip():
    # Monthly cadence with a single delayed print (a ~2-month gap once).
    base = list(pd.date_range("2020-01-31", periods=5, freq="ME"))
    base += [base[-1] + pd.offsets.MonthEnd(2)]           # one skipped month
    base += list(pd.date_range(base[-1] + pd.offsets.MonthEnd(1), periods=5, freq="ME"))
    labels = infer_release_frequency(_eop_series(base), window=3)
    # Rolling median absorbs the single 2-month gap -> stays monthly throughout.
    assert (labels == "M").all()


def test_revisions_share_eop_frequency():
    # Two observations (revisions) with identical eop inherit the same frequency.
    dates = list(pd.date_range("2020-01-31", periods=6, freq="ME"))
    eop = _eop_series(dates + [dates[-1]])                 # a revision of the last eop
    labels = infer_release_frequency(eop)
    assert labels.iloc[-1] == labels.iloc[-2] == "M"
```

- [ ] **Step 2: Run to verify**

Run: `pytest tests/unit/management/test_frequency.py -k "break or irregular or revisions" -v`
Expected: PASS. If `test_quarterly_to_monthly_break` shows more than ~2 releases of lag, that indicates the `window` smoothing is too wide for the assertion — verify the median lag and adjust the assertion's slice (`iloc[-4:]`) to match documented behaviour, not the implementation.

- [ ] **Step 3: (no new implementation)**

- [ ] **Step 4: Run the full helper suite**

Run: `pytest tests/unit/management/test_frequency.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/unit/management/test_frequency.py
git commit -m "test(utils): break-detection and jitter robustness for infer_release_frequency

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Snap-boundary correctness

**Files:**
- Test: `tests/unit/management/test_frequency.py`

**Interfaces:**
- Consumes: `infer_release_frequency` (Task 1).

- [ ] **Step 1: Write the failing tests**

```python
def test_snap_boundaries_weekly_monthly_quarterly():
    # Weekly-ish (7d) -> W; ~30d -> M; ~91d -> Q.
    weekly = _eop_series(pd.date_range("2020-01-03", periods=10, freq="W-FRI"))
    assert (infer_release_frequency(weekly) == "W").all()

    # A ~45-day cadence sits between M (30.4) and Q (91.3); in log space it is nearer M.
    mid = _eop_series(pd.to_datetime(
        ["2020-01-31", "2020-03-16", "2020-04-30", "2020-06-14"]
    ))
    assert (infer_release_frequency(mid) == "M").all()
```

- [ ] **Step 2: Run to verify**

Run: `pytest tests/unit/management/test_frequency.py -k "snap_boundaries" -v`
Expected: PASS. If the ~45-day case snaps to Q, recompute `log(45)` vs `log(30.4)` and `log(91.3)`; the nearer is M — fix the test dates if the intended midpoint was misplaced, not the snap logic.

- [ ] **Step 3: (no new implementation)**

- [ ] **Step 4: Run the full helper suite**

Run: `pytest tests/unit/management/test_frequency.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/unit/management/test_frequency.py
git commit -m "test(utils): snap-to-nearest boundary behaviour for infer_release_frequency

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: `annualize_by_release_frequency` transform

**Files:**
- Create: `macrosynergy/panel/annualize_by_release_frequency.py`
- Modify: `macrosynergy/panel/__init__.py`
- Test: `tests/unit/panel/test_annualize_by_release_frequency.py`

**Interfaces:**
- Consumes: `infer_release_frequency` (Task 1), `ANNUALIZATION_FACTORS`, `reduce_df`, `QuantamentalDataFrame`.
- Produces: `annualize_by_release_frequency(df, xcats=None, cids=None, eop_col="eop", window=3, freqs=("D","W","M","Q","A"), postfix="A") -> QuantamentalDataFrame`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/panel/test_annualize_by_release_frequency.py
import numpy as np
import pandas as pd
import pytest

from macrosynergy.panel import annualize_by_release_frequency


def _qdf_with_eop(cid, xcat, real_dates, eops, values):
    return pd.DataFrame({
        "cid": cid, "xcat": xcat,
        "real_date": pd.to_datetime(real_dates),
        "eop": pd.to_datetime(eops),
        "value": values,
    })


def test_pure_monthly_weight_matches_static():
    dates = pd.date_range("2020-01-31", periods=12, freq="ME")
    df = _qdf_with_eop("AUD", "CPIH", dates, dates, [1.0] * 12)
    out = annualize_by_release_frequency(df, postfix="A")
    assert set(out["xcat"].unique()) == {"CPIHA"}
    # Monthly -> value * sqrt(1/12).
    assert np.allclose(out["value"].to_numpy(), 1.0 * np.sqrt(1 / 12))


def test_missing_eop_raises():
    df = pd.DataFrame({
        "cid": ["AUD"], "xcat": ["CPIH"],
        "real_date": pd.to_datetime(["2020-01-31"]), "value": [1.0],
    })
    with pytest.raises(ValueError):
        annualize_by_release_frequency(df)
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/unit/panel/test_annualize_by_release_frequency.py::test_pure_monthly_weight_matches_static -v`
Expected: FAIL with `ImportError: cannot import name 'annualize_by_release_frequency'`.

- [ ] **Step 3: Write minimal implementation**

```python
# macrosynergy/panel/annualize_by_release_frequency.py
"""
Annualize quantamental values by a time-varying weight inferred from release cadence.
"""
from typing import List, Tuple

import numpy as np
import pandas as pd

from macrosynergy.management.constants import ANNUALIZATION_FACTORS
from macrosynergy.management.utils import infer_release_frequency
from macrosynergy.management.types import QuantamentalDataFrame


def annualize_by_release_frequency(
    df: pd.DataFrame,
    xcats: List[str] = None,
    cids: List[str] = None,
    eop_col: str = "eop",
    window: int = 3,
    freqs: Tuple[str, ...] = ("D", "W", "M", "Q", "A"),
    postfix: str = "A",
) -> QuantamentalDataFrame:
    """
    Multiply each value by ``sqrt(1 / ANNUALIZATION_FACTORS[freq])`` where ``freq`` is the
    contemporaneous release frequency inferred per observation from ``eop`` cadence.

    The weight is time-varying: a series whose cadence changes (e.g. quarterly -> monthly)
    is weighted quarterly before the break and monthly after it.

    Parameters
    ----------
    df : ~pandas.DataFrame
        QuantamentalDataFrame with columns 'cid', 'xcat', 'real_date', 'value' and an
        end-of-period column (``eop_col``). Emit it via
        ``InformationStateChanges.to_qdf(metrics=["eop", ...])``.
    xcats, cids : List[str]
        categories / cross-sections to transform. Default is all in ``df``.
    eop_col : str
        name of the end-of-period date column. Default "eop".
    window : int
        rolling-median window passed to ``infer_release_frequency``. Default 3.
    freqs : Tuple[str, ...]
        candidate frequency labels. Default ("D", "W", "M", "Q", "A").
    postfix : str
        suffix appended to each output category. Default "A".

    Returns
    -------
    ~pandas.DataFrame
        standardized QuantamentalDataFrame with columns 'cid', 'xcat', 'real_date',
        'value'; categories renamed ``{xcat}{postfix}``.
    """
    cols = ["cid", "xcat", "real_date", "value"]
    if eop_col not in df.columns:
        raise ValueError(
            f"`{eop_col}` column required. Emit it via "
            f"InformationStateChanges.to_qdf(metrics=['{eop_col}', ...])."
        )

    _as_categorical = QuantamentalDataFrame(df[cols]).InitializedAsCategorical

    # reduce_df strips non-standard columns, so subset on a plain frame to keep eop_col.
    work = df[cols + [eop_col]].copy()
    work["cid"] = work["cid"].astype(str)
    work["xcat"] = work["xcat"].astype(str)
    if xcats is not None:
        work = work[work["xcat"].isin(xcats)]
    if cids is not None:
        work = work[work["cid"].isin(cids)]

    weights = {v: np.sqrt(1 / ANNUALIZATION_FACTORS[v]) for v in freqs}

    frames = []
    for (cid, xcat), g in work.sort_values("real_date").groupby(["cid", "xcat"]):
        g = g.copy()
        freq = infer_release_frequency(g[eop_col], window=window, freqs=freqs)
        g["value"] = g["value"].to_numpy() * freq.map(weights).to_numpy()
        g["xcat"] = f"{xcat}{postfix}"
        frames.append(g[cols])

    df_out = pd.concat(frames, axis=0, ignore_index=True) if frames else pd.DataFrame(columns=cols)
    return QuantamentalDataFrame.from_long_df(df_out, categorical=_as_categorical)
```

> Implementation note: `reduce_df` drops non-standard columns, so subsetting is done directly on a plain frame (`work`) to keep `eop_col` attached, rather than routing through `reduce_df`.

Then modify `macrosynergy/panel/__init__.py`: add `from .annualize_by_release_frequency import annualize_by_release_frequency` and include it in `__all__`.

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/unit/panel/test_annualize_by_release_frequency.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add macrosynergy/panel/annualize_by_release_frequency.py macrosynergy/panel/__init__.py tests/unit/panel/test_annualize_by_release_frequency.py
git commit -m "feat(panel): add annualize_by_release_frequency (time-varying freq annualization)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: Break transition + quarterly weight + no magic numbers

**Files:**
- Test: `tests/unit/panel/test_annualize_by_release_frequency.py`

**Interfaces:**
- Consumes: `annualize_by_release_frequency` (Task 4).

- [ ] **Step 1: Write the failing tests**

```python
from macrosynergy.management.constants import ANNUALIZATION_FACTORS


def test_pure_quarterly_weight():
    dates = pd.date_range("2015-03-31", periods=8, freq="QE")
    df = _qdf_with_eop("AUD", "CPIH", dates, dates, [1.0] * 8)
    out = annualize_by_release_frequency(df)
    # Quarterly -> value * sqrt(1/4) = 0.5.
    assert np.allclose(out["value"].to_numpy(), 0.5)


def test_break_transitions_weight():
    q = pd.date_range("2015-03-31", periods=8, freq="QE")
    m = pd.date_range(q[-1] + pd.offsets.MonthEnd(1), periods=12, freq="ME")
    dates = list(q) + list(m)
    df = _qdf_with_eop("AUD", "CPIH", dates, dates, [1.0] * len(dates))
    out = annualize_by_release_frequency(df).sort_values("real_date")
    vals = out["value"].to_numpy()
    # Early (quarterly) weight 0.5; late (monthly) weight sqrt(1/12).
    assert np.isclose(vals[0], 0.5)
    assert np.isclose(vals[-1], np.sqrt(1 / 12))


def test_uses_annualization_factors_constant():
    # Guards against re-hard-coding 4/12: quarterly weight must equal the constant.
    dates = pd.date_range("2015-03-31", periods=8, freq="QE")
    df = _qdf_with_eop("AUD", "CPIH", dates, dates, [1.0] * 8)
    out = annualize_by_release_frequency(df)
    assert np.allclose(out["value"].to_numpy(), np.sqrt(1 / ANNUALIZATION_FACTORS["Q"]))
```

- [ ] **Step 2: Run to verify**

Run: `pytest tests/unit/panel/test_annualize_by_release_frequency.py -k "quarterly or break or annualization_factors" -v`
Expected: PASS.

- [ ] **Step 3: (no new implementation)**

- [ ] **Step 4: Run the full transform suite**

Run: `pytest tests/unit/panel/test_annualize_by_release_frequency.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/unit/panel/test_annualize_by_release_frequency.py
git commit -m "test(panel): break transition, quarterly weight, ANNUALIZATION_FACTORS reuse

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 6: Wire into the surprises notebook (separate academy PR)

**Files:**
- Modify: `academy/drafts/surprises/Cyclical strength composite.ipynb` — the "Normalize surprises" and "Annualize normalized surprises" cells.

> Lands in the **academy** repo on `feature/eco-surp`, after this feature is importable. Kept here for traceability.

- [ ] **Step 1:** Keep `eop` through normalization so it is available for cadence inference:

```python
dfa = isc_arma.to_qdf(value_column="zscore", postfix="N", thresh=3, metrics=["eop"]) \
              .dropna(subset=["value"])
dfx = msm.update_df(dfx, dfa[["real_date", "cid", "xcat", "value", "eop"]])
```

- [ ] **Step 2:** Replace the static annualization cell (`dict_freq` + hard-coded √(1/4)/√(1/12)) with:

```python
dfa = msp.annualize_by_release_frequency(
    dfx, xcats=[xc + "N" for xc in surprises], cids=cids, postfix="A",
)
dfx = msm.update_df(dfx, dfa)
```

- [ ] **Step 3:** Confirm the AUD CPI series now annualizes quarterly pre-break and monthly post-break; drop the now-unused `dict_freq` cell if nothing else consumes it.

- [ ] **Step 4:** Run the notebook end-to-end (`/run-nb`) and commit on `feature/eco-surp` (academy repo).

## Self-Review Notes
- Spec coverage: helper (Task 1), break + jitter + revisions (Task 2), snap boundaries (Task 3), transform + missing-eop error (Task 4), weight values + ANNUALIZATION_FACTORS reuse + break transition (Task 5), notebook wiring (Task 6). All spec sections mapped.
- Type consistency: `infer_release_frequency(eop, window, freqs) -> Series[str]` used identically in Tasks 1–5; `annualize_by_release_frequency(...)` signature stable across Tasks 4–6.
- The one implementation subtlety flagged inline in Task 4: `reduce_df` strips `eop`, so subsetting is done directly on a plain frame to keep `eop_col` attached (no `reduce_df` call needed here).
