# Design — `panel_ewm_sum`

- **Date:** 2026-07-03
- **Repo:** `macrosynergy`
- **Target branch:** `feature/performance` (base `develop`)
- **Status:** proposed, awaiting review
- **Origin:** surprises pipeline (`academy/drafts/surprises/Cyclical strength composite.ipynb`,
  "Exponential moving sum of normalized surprises" cell) and the identical `_3DXMS` block in
  `academy/notebooks/Strategies/Commodity strategies/Economic surprises and commodity futures returns.ipynb`.

## Problem

The pipeline computes an exponential moving **sum** of ~68 normalized-annualized surprise panels
(`*_ARMASNA`) with a single `panel_calculator` call driven by a list of ~68 formula strings:

```python
calcs = [f"{xc}_{ht}DXMS = {xc}.ewm(halflife={ht}).sum()" for xc in armanas]
dfa = msp.panel_calculator(dfx, calcs=calcs, cids=cids)
```

This is slow for what is, mathematically, a single vectorised operation. Reading
`macrosynergy/panel/panel_calculator.py`, one such call, over the *entire* `dfx` (all categories,
not just the target panels), does:

- `QuantamentalDataFrame(df[cols])` + `add_ticker_column()` — categorical conversion and a
  ticker-string concat over the whole frame (once).
- **Per xcat (×68):** a boolean filter, a `pivot`, an `eval()` of the formula string, a `pd.melt`,
  a `QuantamentalDataFrame.from_long_df` conversion — and the per-category results are glued with
  `pd.concat` *inside the loop*, which is O(n²) in the number of categories.
- Regex / char-by-char formula parsing (`_check_calcs`, `xcat_isolator`) per formula.

None of the per-category Python machinery is necessary: EWM-summing every column of a matrix is one
call. The reference `_3DXMS` block in the commodity notebook already does the fast shape by hand
(one pivot → one `p.ewm(hl).sum()` → one stack). We promote that pattern to a tested, reusable
package function.

### The correctness fork (decided)

`pandas` `.ewm(halflife=h).sum()` with the defaults `times=None, ignore_na=False, adjust=True`
decays by **row position**, not by calendar distance. The `*_ARMASNA` panels in `dfx` are **sparse**:
`InformationStateChanges.to_qdf(...)` densifies onto a business-day grid, but the notebook then
`.dropna(subset=["value"])`, so each surprise exists only on its release/revision dates.
Consequently, `panel_calculator`, which pivots **per xcat**, decays over *release events* — a
`halflife=5` is 5 releases, not 5 days — and each category sees a different (its own) row index.

**Decision:** the intended meaning of `5DXMS` is a **≈5 business-day** half-life. `panel_ewm_sum`
therefore reindexes to a dense business-day grid before the EWM. This deliberately does **not**
reproduce the current cell's numbers on sparse input — it fixes what is effectively a latent
event-count bug. On an already-dense daily panel the two agree (locked by a test, below).

## Goals / non-goals

**Goals**
- One vectorised EWM-sum over a panel; seconds, not minutes, for the surprises use case.
- Dense business-day-grid semantics (`halflife` in days).
- QuantamentalDataFrame in → QuantamentalDataFrame out, categorical dtype round-tripped.
- Drop-in for both the surprises `_5DXMS` cell and the commodity `_3DXMS` block.

**Non-goals (YAGNI)**
- No bundled normalization / annualization / weighting. The function EWM-sums *whatever panel it is
  given*. "Raw vs modified" output is achieved by calling it on the raw `*_ARMAS` panel or on the
  normalized-annualized `*_ARMASNA` panel — not via a flag.
- Not a general `panel_calculator` replacement; it does exactly one operation.
- Does not reconcile with the pre-existing private `management.utils.math.ewm_sum`, which uses a
  *different* definition (`ewm().mean() × cumulative-weights`). We use pandas `.ewm().sum()` to match
  the pipeline's current definition, and cross-reference `math.ewm_sum` in the docstring so the two
  are not confused.

## Public interface

Module: `macrosynergy/panel/panel_ewm_sum.py`; exported as `panel_ewm_sum` from
`macrosynergy.panel`. Shape mirrors `historic_vol` (halflife-based temporal panel transform).

```python
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
    ...
```

- `xcats`, `cids` — panels to transform (default: all in `df`).
- `halflife` — scalar or list; the EWM half-life in **business days**. A list produces one output
  category per value.
- The grid is **fixed to business days** (`"B"`), matching the standard Quantamental Indicator
  daily frequency. Not a parameter — see resolved open question below.
- `fillna` — value for interior gaps after reindexing to the grid. Default `0.0` (see semantics).
- `mask_leading` — if `True` (default), output before each series' first real observation is `NaN`
  (we do not emit values built purely from pre-history fill).
- `start`, `end`, `blacklist` — standard reduce/exclusion semantics, as in `panel_calculator`.
- `postfix` — output suffix. Default `None` → auto `f"_{h}DXMS"` per half-life. If provided:
  a `str` is allowed only for a scalar `halflife`; a list must match `len(halflife)`.

Returns a standard QDF with columns `["cid", "xcat", "real_date", "value"]`, new xcats named
`{xcat}_{h}DXMS` (or `{xcat}_{postfix}`).

## Algorithm

```
1.  assert required columns; df = QuantamentalDataFrame(df[cols]);
    _as_categorical = df.InitializedAsCategorical
2.  dfr = reduce_df(df, xcats, cids, start, end, blacklist, intersect=False)
3.  p = dfr.pivot(index="real_date", columns=["cid", "xcat"], values="value")   # one pivot
4.  first_valid = {col: p[col].first_valid_index() for col in p.columns}        # pre-fill
5.  grid = pd.date_range(p.index.min(), p.index.max(), freq="B")
    p = p.reindex(grid)
    p = p.fillna(fillna)                                                        # interior → 0.0
6.  for h in as_list(halflife):
        out = p.ewm(halflife=h).sum()
        if mask_leading:
            for col in out.columns:
                out.loc[out.index < first_valid[col], col] = np.nan
        out.columns = MultiIndex[(cid, f"{xcat}_{suffix(h)}")]
        long = out.stack(["cid", "xcat"]).rename("value").reset_index()
        long = long.dropna(subset=["value"])          # drop leading NaNs
        collect long
7.  df_out = concat(collected); return QuantamentalDataFrame(df_out, categorical=_as_categorical)
```

Notes:
- On the dense grid the interior has no NaN, so `ignore_na` is irrelevant there; the semantics are
  fully determined by the grid + zero-fill.
- Only `len(halflife)` EWM sweeps and stacks — no per-category Python loop, no in-loop `concat`.

### Semantics of `fillna=0.0`

A surprise is an event: on non-release business days no new surprise arrives, so its contribution to
the moving sum is `0` and the accumulated stock decays via the EWM weights. Forward-filling would
re-inject the same surprise every day and the sum would diverge — wrong for a *sum*. Interior gaps
are therefore zero-filled; the region *before* a series' first observation is left `NaN` (via
`mask_leading`) so we never emit spurious pre-history values.

## Testing

New file `tests/unit/panel/test_panel_ewm_sum.py` (mirrors `test_historic_vol.py`):

1. **Equivalence lock.** On an *already dense, daily-`B`* panel with no gaps, `panel_ewm_sum`
   (which on such input reindexes to itself and zero-fills nothing in the interior) matches a
   reference `panel_calculator` EWM-sum on the interior region. Proves densification is the *only*
   intended behavioural difference.
2. **Densification changes sparse input.** On a sparse panel, `panel_ewm_sum` ≠ the per-event
   `panel_calculator` result, and equals a hand-built dense-grid reference.
3. **Zero-fill vs decay.** A single spike followed by empty days decays by the correct EWM factor
   per business day.
4. **Leading NaN.** No output before each column's first real observation; `mask_leading=False`
   emits from grid start.
5. **Multi-halflife naming.** `halflife=[3, 5]` yields `_3DXMS` and `_5DXMS`; `postfix` overrides.
6. **`cids`/`xcats`/`blacklist`/`start`/`end`** subsetting behaves like `panel_calculator`.
7. **Categorical round-trip.** `InitializedAsCategorical` preserved in the output.
8. **Degenerate inputs.** single cid, single xcat, all-NaN column, empty reduction.

## Delivery checklist
- `panel_ewm_sum.py` + numpydoc docstring (cross-referencing `math.ewm_sum`).
- Export in `macrosynergy/panel/__init__.py` (`import` + `__all__`).
- `tests/unit/panel/test_panel_ewm_sum.py`.
- Update the two notebooks to call it (separate academy PR, on `feature/eco-surp`).

## Resolved open questions
- **Grid frequency — RESOLVED: business days only.** No `freq` parameter; the grid is always `"B"`,
  matching the standard Quantamental Indicator daily frequency.
- **Preserve `eop_lag`/`grading` metrics — RESOLVED: no, value-only** (follow `panel_calculator`).
  The output is a dense derived series (a decayed sum defined on *every* business day), so per-release
  metrics like `eop_lag`/`grading` do not map onto it cleanly — carrying them would be more tricky
  than useful. Keep the code clean and emit `["cid", "xcat", "real_date", "value"]`.
