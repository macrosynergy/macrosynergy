# panel_ewm_sum Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a fast, vectorised exponential moving-**sum** panel transform (`panel_ewm_sum`) that replaces the slow ~68-formula `panel_calculator` EWM loop in the surprises pipeline.

**Architecture:** One pivot of the selected panels to a ticker-columned wide frame → reindex to a dense business-day grid → zero-fill interior gaps → one `.ewm(halflife).sum()` sweep per half-life → stack back to a QuantamentalDataFrame. No per-category Python loop, no in-loop `concat`.

**Tech Stack:** Python, pandas, macrosynergy (`QuantamentalDataFrame`, `reduce_df`, `make_test_df`), pytest.

## Global Constraints

- Repo: `macrosynergy`. Branch: `feature/performance` (base `develop`). Spec: `docs/superpowers/specs/2026-07-03-panel-ewm-sum-design.md`.
- Grid is **business days only** (`freq="B"`). No `freq` parameter.
- **Zero-fill** interior gaps; region **before each series' first real observation stays NaN** (`mask_leading=True` default).
- Output is **value-only**: columns exactly `["cid", "xcat", "real_date", "value"]`. No `eop_lag`/`grading`.
- Uses pandas `.ewm(halflife).sum()` semantics (NOT `management.utils.math.ewm_sum`, which is a different formula — cross-reference it in the docstring).
- QuantamentalDataFrame in → out; preserve categorical dtype via `InitializedAsCategorical`.
- Output xcat naming: `{xcat}_{h}DXMS` by default (e.g. `_5DXMS`), or `{xcat}_{postfix}`.

## File Structure

- Create: `macrosynergy/panel/panel_ewm_sum.py` — the transform (one public function, one responsibility).
- Modify: `macrosynergy/panel/__init__.py` — add import + `__all__` entry.
- Create: `tests/unit/panel/test_panel_ewm_sum.py` — unit tests (mirrors `test_historic_vol.py`).

---

### Task 1: Core `panel_ewm_sum` — scalar half-life, dense grid, zero-fill, leading mask, export

**Files:**
- Create: `macrosynergy/panel/panel_ewm_sum.py`
- Modify: `macrosynergy/panel/__init__.py`
- Test: `tests/unit/panel/test_panel_ewm_sum.py`

**Interfaces:**
- Consumes: `reduce_df` (`macrosynergy.management.utils`), `QuantamentalDataFrame` (`macrosynergy.management.types`), `make_test_df` (`macrosynergy.management.simulate`).
- Produces: `panel_ewm_sum(df, xcats=None, cids=None, halflife=5, fillna=0.0, mask_leading=True, start=None, end=None, blacklist=None, postfix=None) -> QuantamentalDataFrame`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/panel/test_panel_ewm_sum.py
import numpy as np
import pandas as pd
import pytest

from macrosynergy.management.simulate import make_test_df
from macrosynergy.panel import panel_ewm_sum


def test_basic_ewm_sum_and_naming():
    # Dense daily (business-day) panel, no gaps.
    df = make_test_df(
        cids=["AUD", "CAD"], xcats=["GROWTH", "INFL"],
        start="2020-01-01", end="2020-06-30",
    )
    out = panel_ewm_sum(df, halflife=5)

    # New categories are named with the _{h}DXMS suffix.
    assert set(out["xcat"].unique()) == {
        "GROWTH_5DXMS", "INFL_5DXMS",
    }
    # Value-only standard columns, in order.
    assert list(out.columns) == ["cid", "xcat", "real_date", "value"]

    # Matches a hand-built reference for one series (already dense -> reindex is identity).
    ref = (
        df[(df["cid"] == "AUD") & (df["xcat"] == "GROWTH")]
        .set_index("real_date")["value"]
        .ewm(halflife=5).sum()
    )
    got = (
        out[(out["cid"] == "AUD") & (out["xcat"] == "GROWTH_5DXMS")]
        .set_index("real_date")["value"]
    )
    pd.testing.assert_series_equal(
        got.astype(float), ref.astype(float),
        check_names=False, check_freq=False,
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/panel/test_panel_ewm_sum.py::test_basic_ewm_sum_and_naming -v`
Expected: FAIL with `ImportError: cannot import name 'panel_ewm_sum'`.

- [ ] **Step 3: Write minimal implementation**

```python
# macrosynergy/panel/panel_ewm_sum.py
"""
Fast exponential moving sum of quantamental panels on a business-day grid.
"""
from typing import List, Optional, Union

import numpy as np
import pandas as pd

from macrosynergy.management.utils import reduce_df
from macrosynergy.management.types import QuantamentalDataFrame


def panel_ewm_sum(
    df: pd.DataFrame,
    xcats: List[str] = None,
    cids: List[str] = None,
    halflife: Union[int, float, List[Union[int, float]]] = 5,
    fillna: float = 0.0,
    mask_leading: bool = True,
    start: str = None,
    end: str = None,
    blacklist: dict = None,
    postfix: Optional[Union[str, List[str]]] = None,
) -> QuantamentalDataFrame:
    """
    Exponentially weighted moving sum of one or more category panels, computed on a
    dense business-day grid.

    Unlike :func:`macrosynergy.management.utils.math.ewm_sum` (which returns
    ``ewm().mean()`` scaled by cumulative weights), this uses the pandas
    ``ewm(halflife).sum()`` definition directly. Sparse inputs are reindexed to a
    business-day grid and zero-filled between observations, so ``halflife`` is measured
    in business days.

    Parameters
    ----------
    df : ~pandas.DataFrame
        standardized QuantamentalDataFrame with columns 'cid', 'xcat', 'real_date',
        'value'.
    xcats : List[str]
        categories to transform. Default is all categories in ``df``.
    cids : List[str]
        cross-sections to transform. Default is all cross-sections in ``df``.
    halflife : int | float | List
        EWM half-life in business days. A list produces one output category per value.
    fillna : float
        value used for interior gaps after reindexing to the business-day grid.
        Default 0.0 (a business day with no release contributes zero to the moving sum).
    mask_leading : bool
        if True (default) output before each series' first real observation is NaN.
    start, end : str
        date bounds (ISO). Default None uses the range in ``df``.
    blacklist : dict
        cross-sections with date ranges to exclude.
    postfix : str | List[str]
        output category suffix. Default None -> ``f"{h}DXMS"`` per half-life. A single
        string is allowed only for a scalar ``halflife``; a list must match its length.

    Returns
    -------
    ~pandas.DataFrame
        standardized QuantamentalDataFrame with columns 'cid', 'xcat', 'real_date',
        'value'; new categories named ``{xcat}_{h}DXMS`` (or ``{xcat}_{postfix}``).
    """
    cols = ["cid", "xcat", "real_date", "value"]
    assert set(cols).issubset(set(df.columns)), f"df must contain columns: {cols}."

    qdf = QuantamentalDataFrame(df[cols])
    _as_categorical = qdf.InitializedAsCategorical

    hls = [halflife] if isinstance(halflife, (int, float)) else list(halflife)
    assert all(isinstance(h, (int, float)) and h > 0 for h in hls), (
        "halflife must be a positive number or a list of positive numbers."
    )
    if postfix is None:
        postfixes = [f"{h}DXMS" for h in hls]
    elif isinstance(postfix, str):
        assert len(hls) == 1, "A string postfix requires a scalar halflife."
        postfixes = [postfix]
    else:
        assert len(postfix) == len(hls), "postfix list must match halflife length."
        postfixes = list(postfix)

    dfr = reduce_df(
        qdf, xcats=xcats, cids=cids, start=start, end=end, blacklist=blacklist
    )
    if dfr.empty:
        return QuantamentalDataFrame.from_long_df(
            pd.DataFrame(columns=cols), categorical=_as_categorical
        )

    dfr = dfr.assign(
        ticker=dfr["cid"].astype(str) + "_" + dfr["xcat"].astype(str)
    )
    p = dfr.pivot(index="real_date", columns="ticker", values="value")
    first_valid = {c: p[c].first_valid_index() for c in p.columns}

    grid = pd.date_range(p.index.min(), p.index.max(), freq="B")
    p = p.reindex(grid)
    p.index.name = "real_date"
    p = p.fillna(fillna)

    frames = []
    for h, pf in zip(hls, postfixes):
        out = p.ewm(halflife=h).sum()
        if mask_leading:
            for c in out.columns:
                out.loc[out.index < first_valid[c], c] = np.nan
        out.columns = [f"{c}_{pf}" for c in out.columns]
        tmp = out.stack().to_frame("value").reset_index()
        tmp.columns = ["real_date", "ticker", "value"]
        tmp[["cid", "xcat"]] = tmp["ticker"].str.split("_", n=1, expand=True)
        frames.append(tmp[cols])

    df_out = pd.concat(frames, axis=0, ignore_index=True)
    return QuantamentalDataFrame.from_long_df(df_out, categorical=_as_categorical)
```

Then modify `macrosynergy/panel/__init__.py`: add `from .panel_ewm_sum import panel_ewm_sum` next to the other imports, and add `"panel_ewm_sum",` to the `__all__` list.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/panel/test_panel_ewm_sum.py::test_basic_ewm_sum_and_naming -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add macrosynergy/panel/panel_ewm_sum.py macrosynergy/panel/__init__.py tests/unit/panel/test_panel_ewm_sum.py
git commit -m "feat(panel): add panel_ewm_sum (fast EWM moving sum on B-day grid)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Zero-fill decay + leading-NaN semantics

**Files:**
- Test: `tests/unit/panel/test_panel_ewm_sum.py`

**Interfaces:**
- Consumes: `panel_ewm_sum` from Task 1.

- [ ] **Step 1: Write the failing tests**

```python
def _sparse_qdf():
    # AUD_GROWTH observed only on 2020-01-01 (value 10) and 2020-01-10 (value 0),
    # nothing in between -> tests interior zero-fill decay and leading NaN.
    rows = [
        ("AUD", "GROWTH", pd.Timestamp("2020-01-01"), 10.0),
        ("AUD", "GROWTH", pd.Timestamp("2020-01-10"), 0.0),
    ]
    return pd.DataFrame(rows, columns=["cid", "xcat", "real_date", "value"])


def test_leading_region_is_nan_then_present():
    df = _sparse_qdf()
    out = panel_ewm_sum(df, halflife=5, mask_leading=True)
    # First business day of output equals the first observation date, not earlier.
    assert out["real_date"].min() == pd.Timestamp("2020-01-01")
    # mask_leading=False still cannot precede the grid start (== first obs here).
    out2 = panel_ewm_sum(df, halflife=5, mask_leading=False)
    assert out2["real_date"].min() == pd.Timestamp("2020-01-01")


def test_zero_fill_decays_between_releases():
    df = _sparse_qdf()
    out = panel_ewm_sum(df, halflife=5).set_index("real_date")["value"]
    # On 2020-01-01 the sum is the first value itself.
    assert out.loc["2020-01-01"] == pytest.approx(10.0)
    # Business days 02..09 have zero input, so the sum decays geometrically.
    alpha_hl = 0.5 ** (1 / 5)
    bdays = pd.date_range("2020-01-01", "2020-01-09", freq="B")
    expected_09 = 10.0 * alpha_hl ** (len(bdays) - 1)
    assert out.loc["2020-01-09"] == pytest.approx(expected_09, rel=1e-9)
```

- [ ] **Step 2: Run to verify they fail (or pass) — establish behaviour**

Run: `pytest tests/unit/panel/test_panel_ewm_sum.py -k "leading or zero_fill" -v`
Expected: PASS with the Task 1 implementation (these lock in existing behaviour). If `test_zero_fill_decays_between_releases` fails, verify the `.ewm(halflife).sum()` decay factor `0.5**(1/hl)` against the actual pandas adjust=True weighting and correct the expected value — do NOT change the implementation.

- [ ] **Step 3: (no new implementation — behaviour already provided by Task 1)**

- [ ] **Step 4: Run the full test file**

Run: `pytest tests/unit/panel/test_panel_ewm_sum.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/unit/panel/test_panel_ewm_sum.py
git commit -m "test(panel): lock zero-fill decay and leading-NaN semantics of panel_ewm_sum

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Multi-half-life + `postfix` override

**Files:**
- Test: `tests/unit/panel/test_panel_ewm_sum.py`

**Interfaces:**
- Consumes: `panel_ewm_sum` from Task 1.

- [ ] **Step 1: Write the failing tests**

```python
def test_multiple_halflives():
    df = make_test_df(cids=["AUD"], xcats=["GROWTH"], start="2020-01-01", end="2020-03-31")
    out = panel_ewm_sum(df, halflife=[3, 5])
    assert set(out["xcat"].unique()) == {"GROWTH_3DXMS", "GROWTH_5DXMS"}


def test_postfix_override_scalar():
    df = make_test_df(cids=["AUD"], xcats=["GROWTH"], start="2020-01-01", end="2020-03-31")
    out = panel_ewm_sum(df, halflife=5, postfix="EWMSUM")
    assert set(out["xcat"].unique()) == {"GROWTH_EWMSUM"}


def test_postfix_string_with_list_halflife_raises():
    df = make_test_df(cids=["AUD"], xcats=["GROWTH"], start="2020-01-01", end="2020-03-31")
    with pytest.raises(AssertionError):
        panel_ewm_sum(df, halflife=[3, 5], postfix="EWMSUM")
```

- [ ] **Step 2: Run to verify they pass**

Run: `pytest tests/unit/panel/test_panel_ewm_sum.py -k "halflives or postfix" -v`
Expected: PASS (behaviour provided by Task 1's list/postfix handling).

- [ ] **Step 3: (no new implementation)**

- [ ] **Step 4: Run the full test file**

Run: `pytest tests/unit/panel/test_panel_ewm_sum.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/unit/panel/test_panel_ewm_sum.py
git commit -m "test(panel): cover multi-halflife and postfix override in panel_ewm_sum

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: Equivalence lock vs `panel_calculator`; sparse divergence

**Files:**
- Test: `tests/unit/panel/test_panel_ewm_sum.py`

**Interfaces:**
- Consumes: `panel_ewm_sum` (Task 1), `panel_calculator` (`macrosynergy.panel`).

- [ ] **Step 1: Write the failing tests**

```python
from macrosynergy.panel import panel_calculator


def test_matches_panel_calculator_on_dense_daily_panel():
    # Already-dense daily-B panel: reindex is identity, fillna a no-op in the interior,
    # so panel_ewm_sum must equal the panel_calculator EWM-sum on the shared region.
    cids = ["AUD", "CAD"]
    df = make_test_df(cids=cids, xcats=["GROWTH"], start="2020-01-01", end="2020-06-30")

    fast = panel_ewm_sum(df, halflife=5)
    ref = panel_calculator(
        df, calcs=["GROWTH_5DXMS = GROWTH.ewm(halflife=5).sum()"], cids=cids
    )

    fast_i = fast.set_index(["cid", "xcat", "real_date"])["value"].sort_index()
    ref_i = ref.set_index(["cid", "xcat", "real_date"])["value"].sort_index()
    # Compare on the intersection of indices (both start at first valid).
    common = fast_i.index.intersection(ref_i.index)
    assert len(common) > 0
    pd.testing.assert_series_equal(
        fast_i.loc[common].astype(float),
        ref_i.loc[common].astype(float),
        check_names=False,
    )


def test_diverges_from_per_event_calc_on_sparse_panel():
    # Sparse panel: panel_calculator decays per release event; panel_ewm_sum decays per
    # business day. They must differ, and panel_ewm_sum must match a dense-grid reference.
    rows = [
        ("AUD", "GROWTH", pd.Timestamp("2020-01-01"), 5.0),
        ("AUD", "GROWTH", pd.Timestamp("2020-02-03"), 5.0),
        ("AUD", "GROWTH", pd.Timestamp("2020-03-02"), 5.0),
    ]
    df = pd.DataFrame(rows, columns=["cid", "xcat", "real_date", "value"])

    fast = panel_ewm_sum(df, halflife=5).set_index("real_date")["value"]
    per_event = panel_calculator(
        df, calcs=["GROWTH_5DXMS = GROWTH.ewm(halflife=5).sum()"], cids=["AUD"]
    ).set_index("real_date")["value"]

    # On the last release date the two definitions disagree.
    last = pd.Timestamp("2020-03-02")
    assert fast.loc[last] != pytest.approx(per_event.loc[last])
```

- [ ] **Step 2: Run to verify**

Run: `pytest tests/unit/panel/test_panel_ewm_sum.py -k "panel_calculator or per_event" -v`
Expected: PASS. If `test_matches_panel_calculator_on_dense_daily_panel` fails on dtype/index edges, align by comparing only the common index (already done) and casting to float — do not weaken the equality.

- [ ] **Step 3: (no new implementation)**

- [ ] **Step 4: Run the full test file**

Run: `pytest tests/unit/panel/test_panel_ewm_sum.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/unit/panel/test_panel_ewm_sum.py
git commit -m "test(panel): equivalence to panel_calculator on dense input, divergence on sparse

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: Subsetting, blacklist, categorical round-trip, degenerate inputs

**Files:**
- Test: `tests/unit/panel/test_panel_ewm_sum.py`

**Interfaces:**
- Consumes: `panel_ewm_sum` (Task 1).

- [ ] **Step 1: Write the failing tests**

```python
def test_cids_and_xcats_subsetting():
    df = make_test_df(
        cids=["AUD", "CAD", "GBP"], xcats=["GROWTH", "INFL"],
        start="2020-01-01", end="2020-03-31",
    )
    out = panel_ewm_sum(df, xcats=["GROWTH"], cids=["AUD", "CAD"], halflife=5)
    assert set(out["xcat"].unique()) == {"GROWTH_5DXMS"}
    assert set(out["cid"].unique()) == {"AUD", "CAD"}


def test_blacklist_excludes_range():
    df = make_test_df(cids=["AUD"], xcats=["GROWTH"], start="2020-01-01", end="2020-06-30")
    black = {"AUD": ["2020-03-01", "2020-04-30"]}
    out = panel_ewm_sum(df, halflife=5, blacklist=black)
    masked = out[(out["cid"] == "AUD") &
                 (out["real_date"] >= "2020-03-01") &
                 (out["real_date"] <= "2020-04-30")]
    assert masked.empty


def test_categorical_round_trip():
    from macrosynergy.management.types import QuantamentalDataFrame
    df = make_test_df(cids=["AUD"], xcats=["GROWTH"], start="2020-01-01", end="2020-03-31")
    qdf = QuantamentalDataFrame(df)  # categorical cid/xcat
    out = panel_ewm_sum(qdf, halflife=5)
    assert isinstance(out["cid"].dtype, pd.CategoricalDtype)
    assert isinstance(out["xcat"].dtype, pd.CategoricalDtype)


def test_single_cid_single_xcat():
    df = make_test_df(cids=["AUD"], xcats=["GROWTH"], start="2020-01-01", end="2020-02-28")
    out = panel_ewm_sum(df, halflife=3)
    assert set(out["xcat"].unique()) == {"GROWTH_3DXMS"}
    assert not out.empty
```

- [ ] **Step 2: Run to verify**

Run: `pytest tests/unit/panel/test_panel_ewm_sum.py -k "subsetting or blacklist or categorical or single" -v`
Expected: PASS. If `test_categorical_round_trip` fails, verify `QuantamentalDataFrame.from_long_df(..., categorical=_as_categorical)` is fed `_as_categorical=True` — the input QDF's `InitializedAsCategorical` must propagate.

- [ ] **Step 3: (no new implementation expected; fix only if a test exposes a real gap)**

- [ ] **Step 4: Run the full suite for the module**

Run: `pytest tests/unit/panel/test_panel_ewm_sum.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/unit/panel/test_panel_ewm_sum.py
git commit -m "test(panel): subsetting, blacklist, categorical round-trip, degenerate inputs

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 6: Wire into the surprises notebooks (separate academy PR)

**Files:**
- Modify: `academy/drafts/surprises/Cyclical strength composite.ipynb` — "Exponential moving sum" cell.
- Modify: `academy/notebooks/Strategies/Commodity strategies/Economic surprises and commodity futures returns.ipynb` — `_3DXMS` cell.

> This task lands in the **academy** repo on `feature/eco-surp`, only after `panel_ewm_sum` is released/importable. Kept here for traceability; execute it as its own academy change.

- [ ] **Step 1:** Replace the surprises `panel_calculator` EWM loop with:

```python
# Exponential moving sum of normalized surprises (vectorised, business-day grid)
dfa = msp.panel_ewm_sum(dfx, xcats=armanas, cids=cids, halflife=5)
dfx = msm.update_df(dfx, dfa)
ms_cats = [f"{c}_5DXMS" for c in armanas]
```

- [ ] **Step 2:** Replace the commodity `_3DXMS` block with:

```python
dfa = msp.panel_ewm_sum(dfx, xcats=comp_surprises, cids=cids, halflife=3)
dfx = msm.update_df(dfx, dfa)
```

- [ ] **Step 3:** Run each notebook end-to-end (`/run-nb <path>`); confirm downstream cells that consume `_5DXMS`/`_3DXMS` categories still resolve.

- [ ] **Step 4:** Commit on `feature/eco-surp` (academy repo).

## Self-Review Notes
- Spec coverage: core algorithm (Task 1), zero-fill + leading NaN (Task 2), multi-halflife/postfix (Task 3), equivalence + sparse divergence (Task 4), subsetting/blacklist/categorical/degenerate (Task 5), notebook wiring (Task 6). All spec sections mapped.
- The decay-factor assertion in Task 2 is the one number to verify against pandas `adjust=True` behaviour during implementation; adjust the *expected value*, never the implementation, if it disagrees.
