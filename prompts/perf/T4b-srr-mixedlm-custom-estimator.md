# T4b · `perf/srr-mixedlm-custom-estimator` — builder brief  (METHODOLOGY CHANGE — NOT output-identical)

> Item: **T4b** · slug: **`srr-mixedlm-custom-estimator`** · depends-on: **none** · **supersedes Q5** (`perf/srr-parallel-mixedlm`)
> Branch/worktree: `perf/srr-mixedlm-custom-estimator` · Base: `feature/performance`
> **Separate workstream — NOT byte-identical.** This item deliberately does NOT reproduce statsmodels'
> p-values byte-for-byte. Its acceptance gate is *statistical agreement within a stated tolerance*,
> validated across synthetic + real panels via an **objective agreement report** — NOT a golden-file diff.
> This is the crucial difference from every other perf/ item and from Q5.

---

## Settled decisions (READ FIRST — resolves the prior open questions)

Empirical evidence was gathered on 2026-07-01 (closed-form estimator vs statsmodels on realistic
panel shapes) and the methodology calls were made by the owner (LSimonsen). Do not re-open these:

1. **Tolerance reference = statsmodels FULL-PRECISION `re.pvalues[1]`, NOT the 3-dp `summary()` string.**
   Pass criterion: `abs(p_custom_fullprec - p_statsmodels_fullprec) <= 1e-3` for every non-nan segment.
2. **1e-8 is NOT a target — it is physically unachievable against statsmodels, and this was proven:**
   - The custom closed-form profile-ML/GLS estimator matches statsmodels' **slope `beta1`** and
     **residual variance `scale`** to ~**1e-17** (machine precision) at the same variance ratio θ.
   - The *entire* p-value gap (typ. ~1e-5, worst ~**2.6e-4** on a small 50×4 panel) comes from
     statsmodels' reported **fixed-effect standard error** differing by ~**2.4e-4 relative** from the
     exact GLS `sqrt([(X'V⁻¹X)⁻¹]₁₁)`. This gap is **irreducible**: tightening statsmodels' optimizer
     `gtol` from 1e-2 → 1e-10 leaves its p-value unchanged (flat at 0.1785651 on the probe case). So
     it is a fixed property of statsmodels' SE formula, not a convergence artifact.
   - Therefore agreement *with statsmodels* is floored at ~3e-4; 1e-3 is comfortably achievable (and
     tighter on the large real panels). Do not chase sub-1e-3 agreement with statsmodels — it does not exist.
3. **NO human methodology sign-off gate.** The objective agreement report (tolerance + zero
   decision-flips + identical `nan` set, below) IS the acceptance evidence. This item is a normal,
   run-perf-queue-claimable Q-item — no external approval blocks merge.
4. **ML (`reml=False`) is intentional** — line 961 explicitly overrides the statsmodels default; the
   estimator uses the ML profile objective, not REML.
5. **`tau²→0` boundary → finite OLS-limit p-value, NOT nan** (statsmodels returns finite there; match it).
6. **3-dp rounding is preserved in the RETURNED value** (see spec below) to keep output shape identical;
   the *validation comparison* is done at full precision. Because the custom value can differ ~1e-4 from
   statsmodels, the 3-dp rounded result can occasionally flip at a rounding boundary (found 0.17846 vs
   0.17857 → `0.178` vs `0.179`); such flips are within the accepted 1e-3 tolerance and are **reported**,
   not hard-failed — the binding gate is the decision-flip count at the 0.9 threshold.

---

## Goal

Reduce the **per-fit** cost of the `MixedLM` panel test in `SignalReturnRelations` by replacing
statsmodels' general iterative REML/ML optimizer with a **specialized estimator** for the exact
random-effects structure actually used (single scalar random intercept grouped by date; one fixed
slope + intercept). Q5 tried to *parallelize* the identical fits and failed: on Windows + OpenBLAS,
threads gave no speedup (BLAS thread contention) and parallelism does not reduce the per-fit cost
anyway. T4b attacks the per-fit cost directly.

**Opportunity.** The SRR MixedLM panel test is ~**749s / ~25%** of a key notebook's total runtime
across cells 31/33/35/39/45/50; the control cell with `ms_panel_test=False` costs only **7.3s**,
confirming the panel test is essentially the entire SRR cost. Each cell runs up to ~35 fully
independent `(signal × return)` fits. The cost per fit is dominated by: (a) the general iterative
optimizer (L-BFGS by default, plus any fallback retries), and (b) two full `summary()` builds per fit
used only to parse a single p-value string. A closed-form / direct-REML-GLS profile likelihood for
this one-variance-component structure removes the iteration and the summary parsing entirely.

---

## Exact spec characterization  (the most valuable part — a builder MUST be able to reproduce the statistic from this alone)

All references are to `macrosynergy/signal/signal_return_relations.py` as of this brief.

### Call path
- `single_statistic_table(...)` (**~1495**) loops over `(ret, sig, freq, agg_sig)` tuples (**1754**),
  calls `manipulate_df` (**1767**), then `calculate_single_stat(stat, ret, sig, type)` (**1772**), and
  for the p-value column `calculate_single_stat(pval_stat, ...)` (**1776**).
- `calculate_single_stat` (**1067**): for `type == "panel"` it sets `css = ["Panel"]`, slices the full
  panel via `__slice_df__` (**1121**, returns the whole `df` for `cs == "Panel"`), drops NA rows
  (**1122**), and for `stat == "map_pval" and self.ms_panel_test` calls `self.map_pval(ret_vals, sig_vals)`
  (**1164–1165**), where `ret_vals, sig_vals = df_segment[ret], df_segment[sig]` (**1130**). For panel
  type it returns `list_of_results[0]` (**1177**). (The same `map_pval` is also reached from
  `__table_stats__` at **924** for `summary_table`.)
- `map_pval(self, ret_vals, sig_vals) -> float` (**931**) does the fit.

### DataFrame / index shape passed in
- `manipulate_df` (**725**) builds `self.df` via `categories_df(...)` (**772**) → a **MultiIndex
  `(cid, real_date)`** wide frame with one column per xcat. So `ret_vals`/`sig_vals` are `pd.Series`
  indexed by `(cid, real_date)`, with index level names `["cid", "real_date"]`.

### The MixedLM specification (map_pval, lines 948–973)
- **Guard (948–955):** if `"cid"` is not an index level, or `ret_vals.index.get_level_values("cid").nunique() <= 1`,
  warn `"P-value could not be calculated, since there wasn't enough datapoints."` and return `np.nan`.
- **Design matrices:**
  - `X = sm.add_constant(ret_vals)` (**956**) → 2-column exog: column 0 = `const` (all ones),
    column 1 = the **return** series. **The fixed-effect regressor is the RETURN.** `add_constant`
    prepends `const`, so column order is `[const, <ret>]`.
  - `y = sig_vals.copy()` (**957**) → **the dependent variable is the SIGNAL.**
  - Model in words: **`signal ~ 1 + return`** (fixed intercept + fixed slope on the return).
- **Random-effects structure / grouping:**
  - `groups = ret_vals.index.get_level_values("real_date")` (**958**) → **groups = TIME (`real_date`),
    NOT cross-section (`cid`).** Every observation on the same date is one group.
  - `mlm = sm.MixedLM(y, X, groups=groups)` (**959**) — arrays form (NOT the formula/`vc_formula`/
    `re_formula` API). With no `exog_re` and no `re_formula`, statsmodels defaults to a **single scalar
    random intercept per group** (one variance component + residual variance). **No random slope, no
    additional variance components.** This is the simplest possible mixed model: random intercept by
    date, one fixed covariate.
- **Estimation method:**
  - `re = mlm.fit(reml=False)` (**961**) → **ML, not REML** (statsmodels `MixedLM` defaults to
    `reml=True`; this call explicitly overrides to **`reml=False`**). Optimizer is statsmodels' default
    cascade for `MixedLM.fit` (L-BFGS-B first, with internal fallbacks); convergence warnings are
    emitted by statsmodels and are otherwise not handled here beyond the `LinAlgError` catch below.
- **Error / degenerate handling:**
  - `except np.linalg.LinAlgError` (**962**): warn `"Singular matrix encountered, so p-value could not
    be calculated."` and return `np.nan`.
  - After fit, `if re.summary().tables[1].iloc[1, 3] == ""` (**967**): warn `"P-value could not be
    calculated, since there wasn't enough datapoints."` and return `np.nan`. (statsmodels renders an
    empty p-value string when the coefficient's standard error is undefined.)
- **Statistic extracted (EXACT):**
  - `pval_string = re.summary().tables[1].iloc[1, 3]` (**972**); `return float(pval_string)` (**973**).
  - `summary().tables[1]` is the coefficient table with rows `[const, <ret>, Group Var]` and column 3 =
    `"P>|z|"`. Row index **1** is the **return slope**. So **the returned statistic is the two-sided
    p-value of the fixed-effect slope on the RETURN**, i.e. `re.pvalues[<ret>]` (equivalently
    `re.pvalues[1]`).
  - **Rounding:** the p-value is read from `summary()`, which formats `P>|z|` to **3 decimal places**;
    `float("0.###")` therefore yields a value **rounded to 3 dp** (and exactly `0.000` when statsmodels
    prints `0.000`). Any replacement must reproduce this **3-dp rounding** of the two-sided normal-tail
    p-value, `2 * (1 - Φ(|z|))`, where `z = beta_ret / se(beta_ret)`.

### Summary of what a custom estimator must reproduce
Given `y = signal`, `X = [1, return]`, groups = `real_date`, ML (not REML), single scalar random
intercept: estimate `beta_ret` and its standard error, form `z = beta_ret / se`, compute the two-sided
p-value `2*(1 - Φ(|z|))`, **round to 3 dp**, return as `float`. Return `np.nan` on the same degenerate
conditions (≤1 cid; singular/undefined SE). Note the guard is on **cid** count, but the RE grouping is
**date** — reproduce both exactly.

---

## Files

- **Modify:** `macrosynergy/signal/signal_return_relations.py` — specifically `map_pval` (**931**), and
  optionally one private helper (e.g. `_reml_gls_pval` / `_mixedlm_slope_pval`) and a small design-matrix
  cache keyed by segment. **No other functions should change behaviour.**
- **Do NOT change the public API** of `SignalReturnRelations`, `single_statistic_table`, or `map_pval`.
  In particular `map_pval` must keep the signature `(self, ret_vals, sig_vals)` — the tripwire
  `TestMapPvalDirect.test_map_pval_signature_unchanged` pins `["self", "ret_vals", "sig_vals"]`. Keep the
  same return type (`float` / `np.nan`) and the same warning messages/paths.
- Do **not** add a required new dependency. `scipy`/`numpy`/`statsmodels` are already present; prefer a
  hand-rolled REML/ML profile-likelihood in numpy/scipy over pulling in a new package. (Per the global
  policy, if any dependency addition is contemplated, research it via the Sonatype MCP tools first — but
  the intent here is **no new dependency**.)

---

## Design

### Specialized estimator for the identified RE structure
The model is the canonical **one-way random-intercept panel** grouped by date `g`:

    signal_it = beta0 + beta1 * return_it + u_g + eps_it,   u_g ~ N(0, tau^2),  eps_it ~ N(0, sigma^2)

with a single scalar variance component `tau^2` and residual `sigma^2`. This has a **one-dimensional
profile likelihood** in `theta = tau^2 / sigma^2` (or equivalently the intraclass ratio). Approach:

1. Group observations by date. For a scalar random intercept, the marginal covariance is block-diagonal
   with blocks `V_g = sigma^2 (I_{n_g} + theta * J_{n_g})` (J = ones matrix). Its inverse and log-det
   have **closed forms** (Sherman–Morrison): no dense inversion, `O(sum n_g)` per likelihood evaluation.
2. Profile out `sigma^2` and the fixed effects `beta` (GLS given `theta`), leaving a **scalar** objective
   in `theta`. Solve with a 1-D bounded optimizer (`scipy.optimize.minimize_scalar`, bounded on
   `theta >= 0`) — far cheaper and more robust than statsmodels' multi-parameter L-BFGS cascade + retries.
   **Use ML (`reml=False`) to match line 961**, not REML, so the profiled objective must be the ML
   objective (do not switch to REML "because it's the statsmodels default" — this code overrides to ML).
3. From the GLS solution at the optimal `theta`, recover `beta1` and `Cov(beta) = (X' V^{-1} X)^{-1}`;
   `se(beta1) = sqrt` of its `[1,1]` entry; `z = beta1/se`; `p = 2*(1 - Φ(|z|))`; round to 3 dp; return.
4. Reproduce degenerate handling: ≤1 cid → nan (same message); singular `X' V^{-1} X` or undefined
   `se` → nan (same messages as the `LinAlgError` / empty-string branches today).

This eliminates the iterative multi-parameter optimizer, all optimizer retries, and **both** `summary()`
builds.

### Cheaper fallbacks / complementary wins (lower risk, can layer)
- **Kill the double `summary()`.** `map_pval` builds `re.summary()` **twice** (**967** and **972**) just to
  read one cell. Even if the estimator stays statsmodels for now, read `re.pvalues[1]` (rounded to 3 dp)
  and the SE-undefined condition directly from the fitted result — one object, no string parsing. This is
  a cheap serial win and is a safe first commit.
- **Reuse/cache design matrices across segments.** Within a `single_statistic_table` call the same
  `(ret, sig)` panel feeds both the primary stat and the p-value column; the `add_constant` /
  grouping-array construction can be built once per segment and reused. Keep the cache keyed strictly to
  the segment identity so results never bleed across `(ret, sig, freq, agg_sig)` tuples.
- **Faster REML/ML backend as a stepping stone.** A tuned single-parameter profile solver (above) is the
  target; if it proves risky, a bounded 1-D solve that still calls into statsmodels' log-likelihood is an
  intermediate.

### Explicitly NOT acceptable
- **Do NOT "speed up" by capping or skipping the optimizer fallback cascade** (e.g. forcing a single
  optimizer, lowering `maxiter`, or loosening `tol`). That changes the p-values statsmodels would have
  produced and is a silent methodology regression, not a perf win. Any change to the numeric result must
  go through the validation + sign-off gate below — it is never a free optimization.
- Do NOT parallelize (that was Q5; superseded). Do NOT change the guard semantics (cid-based) or warning
  text.

---

## VALIDATION PLAN  (replaces byte-identical parity — this is the gate that matters)

Because the custom estimator will **not** reproduce statsmodels' p-values exactly, the parity mechanism
is statistical agreement, proven with a harness (objective tolerance + zero decision-flips + identical
`nan` set). No human sign-off gate (Settled decision 3).

1. **Build a validation harness** (a script or `-m perf`/opt-in test, kept out of the default suite so it
   doesn't gate CI on statsmodels timing) that, for a set of **representative panels**:
   - Constructs the identical `(y = signal, X = [1, return], groups = real_date)` inputs `map_pval` would
     see. **Synthetic coverage** = `tests/perf/data.srr_panel` swept across shapes (#dates, #cids,
     #signals×#returns), including small-N and near-singular cases — always runnable in CI. **Real
     coverage** = the cyclical-strength notebook panels (rates / equity / FX / vGLB, up to 30 DM/EM cids,
     weekly; `academy/drafts/surprises/Cyclical strength composite.ipynb`, cells 57/59/61/65/71); these
     need JPMaQS data (msydevelopers) — capture a stored panel fixture when data access is available and
     add it to the harness. If data access is unavailable at build time, ship on synthetic coverage and
     note the real-panel fixture as a follow-up (the tolerance holds tighter on the large real panels).
   - Computes **p_statsmodels at FULL PRECISION** (`np.asarray(re.pvalues)[1]` from the current
     `mlm.fit(reml=False)` path — NOT the 3-dp `summary()` string) and **p_custom at full precision** for
     every `(signal, return)` segment, across a range of panel sizes / #dates / #cids.
   - Emits a report: per-segment `(p_statsmodels, p_custom, abs_diff)`, max/mean abs diff, the count of
     **3-dp rounded mismatches** (expected only at rounding boundaries — reported, not fatal), and the
     count of segments where the **significance decision** at the SRR threshold (raw p < 0.1, i.e.
     `1 - p > 0.9`, per `significance_threshold=0.9`) **flips**.
2. **Stated numeric tolerance (pass criterion, all three BINDING):**
   - **`abs(p_custom - p_statsmodels) <= 1e-3`** at full precision for every non-nan segment (the ~3e-4
     statsmodels-SE floor sits well inside this — see Settled decision 2), **and**
   - **zero significance-decision flips** at the 0.9 probability-of-significance threshold across the
     validation panels, **and**
   - `nan` is returned for exactly the same segments as the statsmodels path (identical degenerate set).
   - Segments where statsmodels itself failed to converge / returned `nan` are excluded from the diff but
     must be reported (the custom path must also return `nan` there — see risks).
   - 3-dp rounding-boundary flips (Settled decision 6) are reported but do NOT fail the gate.
   - If the 1e-3 tolerance or zero-flip criterion cannot be met on any panel, do **not** ship; report back.

---

## GATE  (verify before hand-back; `--no-cov -n0`, never `-p no:cov`)

> **No `tests/perf/golden/` file applies** — T4b is a methodology change, so there is no byte-identical
> golden to diff. The unit suite + the validation harness + human sign-off are the parity substitute.

1. **Behaviour/API tests stay GREEN** (these still hold — signature, float-in-`[0,1]`, and the
   `map_pval`-needs-`ms_panel_test` guard):
   ```bash
   pytest tests/unit/signal/test_signal_return_relations.py -k "MapPvalDirect" -v --no-cov -n0   # 2 passed
   pytest tests/unit/signal/test_signal_return_relations.py --no-cov -n0                          # full SRR suite green
   ```
   Note: `TestMapPvalDirect.test_map_pval_returns_float_in_unit_interval` only asserts `p in [0,1]`; it
   does **not** pin the exact value, so it will still pass under the new estimator — but exact-value
   agreement is the validation harness's job, not this test's.
2. **Measurable per-fit win (benchmark):**
   ```bash
   pytest tests/perf/test_perf_srr_mixedlm.py -m perf --benchmark-only -n0 --no-cov \
     --benchmark-json=<scratch>/after.json
   python tests/perf/record.py <baseline-json> <scratch>/after.json
   ```
   **FIX THE BENCHMARK FIRST.** As written, `test_perf_srr_mixedlm.py` benchmarks
   `single_statistic_table(stat="accuracy")`, which **never calls `map_pval`** — this is exactly the
   dead-path measurement that let Q5's `[2-3]` case pass without exercising the fit. The benchmark MUST
   drive the MixedLM path: use `single_statistic_table(stat="map_pval", type="panel")` (the panel MAP
   path, as the notebook's `_srr_scalar` does). Re-baseline against the current statsmodels path on that
   same invocation, then demonstrate a clear per-fit speedup on both `[1-1]` and `[2-3]` and **report the
   measurement**. The win is per-fit cost reduction (not parallelism), so even single-fit `[1-1]` improves.
3. **Validation harness passes** its tolerance + zero-decision-flip + identical-`nan`-set criteria (above);
   the agreement report is attached to the PR. **No human sign-off gate** (Settled decision 3).
4. **macrosynergy suite:**
   ```bash
   pytest tests/unit/signal --no-cov -n0
   ```
   (Manager runs the full suite as the merge gate.)
5. **Hygiene:** `git status` shows only `signal_return_relations.py` (and, if added, the validation harness
   file) changed; no `tests/perf/golden/*` modified; no scratch files committed.

---

## Acceptance criteria

- [ ] `map_pval` computes the two-sided p-value of the **return slope** in `signal ~ 1 + return`, random
  intercept grouped by **`real_date`**, **ML (`reml=False`)**, via a specialized estimator — no general
  iterative multi-parameter optimizer, no optimizer-retry cascade, no `summary()` string parsing.
- [ ] Result rounded to **3 dp** and returned as `float`; `np.nan` returned on the same degenerate
  conditions (≤1 cid; singular / undefined SE) with the same warning messages.
- [ ] Public API of `SignalReturnRelations` / `single_statistic_table` / `map_pval` unchanged;
  `TestMapPvalDirect` (incl. signature tripwire) passes — 2 passed.
- [ ] Validation harness shows `abs(p_custom - p_statsmodels) <= 1e-3` at **full precision** (vs
  `re.pvalues[1]`, not the 3-dp string) on every non-nan segment, **zero** significance-decision flips at
  the 0.9 threshold, and an identical `nan` set — across synthetic panels (and real panels if data
  access is available); any 3-dp boundary flips are reported, not fatal.
- [ ] Objective agreement report attached to the PR; **no human sign-off gate** required.
- [ ] Benchmark drives the `map_pval` path (`stat="map_pval", type="panel"`, NOT `stat="accuracy"`) and
  shows a per-fit speedup on `[1-1]` and `[2-3]`; full `tests/unit/signal` suite green.
- [ ] `summary()` is built at most once (ideally not at all — read from the fitted result / closed form).

---

## Notes / risks

- **Methodology change, not an optimization.** This is the one perf/ item whose output is intentionally
  not byte-identical. Treat the tolerance + zero-decision-flip + identical-`nan`-set criteria as hard
  gates; if any fails, report back rather than ship. Do not let it be reviewed under the standard
  golden-file rubric. (No human sign-off gate — Settled decision 3.)
- **Numerical stability.** The profile likelihood in `theta = tau^2/sigma^2` must be evaluated with the
  closed-form (Sherman–Morrison) inverse/log-det, not dense inversion, to stay both fast and stable.
  Guard `theta` at the boundary `0` (no random effect) — statsmodels returns a finite p-value there, so
  the estimator must too, not `nan`.
- **Edge cases the statsmodels path handles today — the custom estimator MUST match:**
  - **Few observations / few dates:** when the SE is undefined statsmodels prints `""` and `map_pval`
    returns `nan` (**967**). Reproduce the same `nan` (do not emit a spurious finite p-value).
  - **Singular design / collinear return-vs-const:** statsmodels raises `LinAlgError` → `nan` (**962**).
    The custom GLS must detect a singular `X' V^{-1} X` and return `nan` with the same warning.
  - **≤1 cross-section:** guard at **948–955** returns `nan` before any fit. Preserve exactly (note this
    guard is on **cid** even though grouping is by **date**).
  - **Non-convergence in statsmodels (CRITICAL for the harness):** statsmodels frequently emits
    `ConvergenceWarning` (optimizer cascade lbfgs→cg retries) yet still returns a **finite, non-optimal**
    p-value — it does NOT nan out. Measured on the synthetic `tests/perf/data.srr_panel` `[2-3]` tier:
    the `[1-1]` single fit converges cleanly (0 warnings), but `[2-3]` throws **7** convergence warnings
    across its 6 fits (near the `tau²→0` boundary, where the monthly date-grouping variance is ~0). The
    custom estimator finds the TRUE MLE, so on these non-converged segments it will legitimately DIVERGE
    from statsmodels' non-optimal value by **more than 1e-3** — that is statsmodels being wrong, not the
    estimator. **The agreement harness MUST detect statsmodels non-convergence** (capture
    `ConvergenceWarning` and/or inspect `re.converged` / `mle_retvals`) and **exclude those segments from
    the ≤1e-3 tolerance check** (report them separately). Compute the tolerance/decision-flip gate only on
    segments where statsmodels genuinely converged. Well-conditioned agreement panels (genuine date-level
    random effects — e.g. the real notebook panels, or synthetic data with a real per-date intercept
    injected) are the meaningful agreement testbed; the raw `srr_panel` is fine for the benchmark (timing)
    but a poor agreement testbed because statsmodels itself doesn't converge on it.
- **Boundary variance (`tau^2 -> 0`).** At the estimated boundary the z-statistic and p-value are still
  well-defined (it reduces to OLS with a robust/aggregated SE); ensure the profile solver returns the OLS
  limit rather than degenerating.
- **Grouping subtlety.** `groups` is **time**, not cross-section — a natural mistake is to group by `cid`.
  The variance component is across dates. Any estimator or reviewer must verify this against line 958.
- **Supersedes Q5.** The Q5 parallelization brief (`prompts/perf/Q5-t4-srr-parallel-mixedlm.md`) is
  retired by this item; do not pursue thread/process fan-out.
