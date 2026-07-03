# Design — frequency-aware annualization (release-cadence inference)

- **Date:** 2026-07-03
- **Repo:** `macrosynergy`
- **Target branch:** `feature/infra` (new, base `develop`)
- **Status:** proposed, awaiting review
- **Origin:** surprises pipeline (`academy/drafts/surprises/Cyclical strength composite.ipynb`,
  "Annualize normalized surprises in dependence on release frequency" cell).

## Problem

The pipeline annualizes each normalized surprise (`*_ARMASN`) by a √-of-frequency factor so that
monthly and quarterly series contribute comparable annualized variance to the downstream moving-sum
composite. Today:

```python
# dict_freq built earlier from the ticker NAME:
ac["freq"] = ac["xcat"].str.split("_").str[2].str[2:3]   # 'M' / 'Q' from the transform code
dom = ac.groupby("ci")["fp"].max()                       # ONE dominant freq per (cid, indicator)
...
filt_q = dfa["ci"].isin(dict_freq["Q"])
dfa.loc[filt_q,  "value"] *= np.sqrt(1 / 4)              # hard-coded
dfa.loc[~filt_q, "value"] *= np.sqrt(1 / 12)             # hard-coded
```

Two defects:

1. **Frequency is guessed from the ticker name, not the data,** and collapsed to a single dominant
   label for the *entire history* of each `(cid, indicator)`.
2. **It cannot represent a structural break in release cadence.** An indicator whose *same*
   quantamental time series switches frequency — e.g. **Australia CPI going quarterly → monthly** —
   gets one label for all time, so either its pre-break (quarterly) or its post-break (monthly)
   observations are annualized with the wrong factor.

The √(1/4) and √(1/12) are also magic numbers rather than `√(1 / ANNUALIZATION_FACTORS[freq])`.

## Goal

A **per-observation, time-varying** annualization weight `√(1 / apy_t)`, where `apy_t` is the
*contemporaneous* annualization factor inferred from the actual release cadence (`eop` spacing), so
the weight flips at a genuine Q→M break and generalises to any supported frequency. Reuse
`ANNUALIZATION_FACTORS` and `FREQUENCY_MAP` from `macrosynergy/management/constants.py` — no
hard-coded numbers.

**Non-goals:** does not change normalization; does not touch the moving-sum step (that is the
separate `panel_ewm_sum` spec); does not attempt to *correct* or resample the underlying data — only
to weight it.

## Cadence estimator (decided: rolling-median gap, then snap)

Per ticker, using its sequence of `eop` dates (revisions sharing an `eop` collapse to one period):

```
distinct_eops = sorted unique eop dates
gap_i (days)  = distinct_eops[i] - distinct_eops[i-1]
gap_i         = rolling_median(gaps, window=k, min_periods=1)          # robustness
freq_i        = snap(gap_i -> nearest of {D, W, M, Q, A})              # by period-in-days
apy_i         = ANNUALIZATION_FACTORS[freq_i]                          # 252/52/12/4/1
weight_i      = sqrt(1 / apy_i)
```

- **Rolling median (window `k`, default 3)** absorbs one-off irregular releases (a delayed or early
  print) without flipping the weight. Trade-off (accepted): a genuine break locks in after ~1–2
  releases rather than instantly. `min_periods=1` seeds the early observations.
- **Snap-to-nearest** uses reference period-lengths in days — `D≈1, W≈7, M≈30.4, Q≈91.3, A≈365` —
  and classifies by nearest in **log** space (so the W/M and M/Q boundaries are ratio-symmetric).
  Supported frequency set is configurable; default `{D, W, M, Q, A}`.
- Each observation (including revisions) inherits the frequency of the `eop` period it belongs to.

For a pure single-frequency series this reproduces today's static factor exactly (monthly →
`√(1/12)`, quarterly → `√(1/4) = 0.5`). For AUD-CPI-style input the weight transitions from `0.5`
to `√(1/12) ≈ 0.2887` shortly after the cadence change.

## Components & home

Two pieces, cleanly separated:

1. **`infer_release_frequency(eop, window=3, freqs=("D","W","M","Q","A")) -> pd.Series`** — pure,
   reusable, side-effect-free. Input: a per-observation `eop` datetime series (indexed by
   `real_date` or positional). Output: a per-observation frequency-label series. Home:
   `macrosynergy/management/utils/` (near `sparse.py` / `math.py`; both already reason about `eop`).
   This is independently unit-testable and has no QDF or ISC dependency.

2. **The annualization applier** — computes `weight = √(1 / ANNUALIZATION_FACTORS[freq])` per
   observation and multiplies `value`, appending a postfix (default `"A"`).

**Recommended home for (2): ISC-sourced `eop`.** `InformationStateChanges` already owns `eop`
per observation (computed from `eop_lag` in `from_qdf`; each ticker frame carries an `eop` column),
and the pipeline already builds `isc_arma`. Cadence must come from `eop`, which the notebook
currently *drops* immediately before this step (`dfa[["real_date","cid","xcat","value"]]`). So the
natural flow keeps `eop` from `to_qdf(metrics=["eop", ...])` and annualizes from it.

Concretely, the public entry is a QDF transform that consumes an `eop` column:

```python
def annualize_by_release_frequency(
    df: pd.DataFrame,          # QDF carrying value + eop (as emitted by ISC.to_qdf)
    xcats: List[str] = None,
    cids: List[str] = None,
    eop_col: str = "eop",
    window: int = 3,
    freqs: Tuple[str, ...] = ("D", "W", "M", "Q", "A"),
    postfix: str = "A",
) -> QuantamentalDataFrame:
    ...
```

with `infer_release_frequency` called per `(cid, xcat)` group. An optional thin
`InformationStateChanges` convenience wrapper may follow, but the standalone helper + QDF transform
is the testable core.

## Resolved — `eop` sourcing

**RESOLVED: (a) require an `eop` column (ISC-sourced).** The transform expects `eop` present in the
QDF (as emitted by `InformationStateChanges.to_qdf(metrics=["eop", ...])`); if it is absent, raise a
clear error rather than reconstructing. This is exact and avoids a fragile `eop_lag → eop`
holiday/business-day reconstruction convention. The notebook change is simply to *keep* `eop` through
`to_qdf` instead of dropping it before this step.

## Testing

`infer_release_frequency` (in `tests/unit/management/test_utils.py` or a new
`test_frequency.py`):
1. Pure monthly `eop` → all `"M"`; pure quarterly → all `"Q"`.
2. **AUD-CPI Q→M break** → labels transition Q → M within ~1–2 observations after the cadence change
   (documents the rolling-median lag).
3. **One-off irregular gap** (single delayed release) does **not** flip the label (robustness).
4. Snap boundaries: gaps near the W/M and M/Q thresholds classify to the correct nearest frequency.
5. Revisions (repeated `eop`) inherit the period's frequency; `window`/`min_periods` seeding at the
   series start behaves.

Annualization applier (`tests/unit/...`):
6. Pure monthly → `value × √(1/12)`; pure quarterly → `value × 0.5` (matches today's static output).
7. AUD-CPI series → weight transitions `0.5 → √(1/12)` at the break.
8. `ANNUALIZATION_FACTORS` reused (no literal 4/12 in the implementation).
9. Categorical round-trip; `postfix` applied.

## Delivery checklist
- `infer_release_frequency` in `management/utils` + numpydoc.
- `annualize_by_release_frequency` (home per resolved open question) + export.
- Tests above.
- Update the surprises notebook to keep `eop` through `to_qdf` and call the new transform
  (separate academy PR, on `feature/eco-surp`).

## Relationship to `panel_ewm_sum`
Independent feature, separate branch (`feature/infra` vs `feature/performance`). In the pipeline the
order is: normalize (ISC) → **annualize (this)** → moving-sum (`panel_ewm_sum`). Neither depends on
the other's implementation.
