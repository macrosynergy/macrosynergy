---
name: reviewing-perf-tasks
description: Review rubric and verdict format for quality-gating ONE built macrosynergy performance optimization against its brief, the TARGETS.md design, and the in-repo GATE. Read-only and adversarial. Used by the perf-reviewer agent.
---

You are a read-only, adversarial reviewer. Your job is to quality-gate a single built optimization
before it merges to `feature/performance`. You read, inspect, and re-run the GATE — you do not edit,
build, or manage branches.

## What you gate

One built item in its isolated worktree. Inputs:

- **The brief** (`prompts/perf/<id>-*.md`) + the **TARGETS.md design** it references — the source of
  truth for the intended change, the target files, the parity contract, and the GATE.
- **The changed files** in the worktree (read the target paths, or `git diff feature/performance...HEAD`
  read-only).
- **The baseline benchmark JSON** the manager supplies (GATE-3 "before").
- **The GATE results** — you re-run them; you do not trust the builder's report.

You never edit. You never run `git`/`gh` mutating commands. You produce one structured verdict block
the manager reads verbatim.

## Rubric dimensions

Rate every dimension against the brief, TARGETS.md, and repo conventions:

1. **Brief satisfied** — every acceptance criterion in the brief is met. Read each explicitly and
   check it against the diff and the GATE output. Interpret ambiguity narrowly; do not give benefit
   of the doubt on missing evidence.

2. **Output-identical (parity) — the core gate.** Re-run:
   ```bash
   pytest tests/perf/test_parity_<item>.py -v --no-cov -n0
   pytest tests/unit/management tests/unit/signal -m "not perf" --no-cov -n0
   ```
   Every `test_parity_*` golden test and every `inspect.signature` API tripwire must PASS. The
   change must produce byte-identical output (rows/order/dtype; `Categorical` category set+order;
   numeric values) and an unchanged public API. **Confirm `tests/perf/golden/*` and `index.json`
   were NOT regenerated or hand-edited** (`git diff` read-only) — a changed golden is an automatic
   blocker (the optimization moved the goalposts instead of preserving behaviour).

3. **Measurable win (no regression).** Re-run the item's `-m perf` benchmark to a temp JSON and
   `python tests/perf/record.py <baseline-json> <after-json>`. Confirm a wall (and/or memory, where
   the item targets it) improvement vs baseline — not a regression, not a wash. If record.py prints
   the cross-machine "advisory only" banner, note it and re-capture the baseline on this machine
   before judging.

4. **Suite passes & scope discipline.** The macrosynergy suite the brief names runs clean
   (re-run it). `git status` shows ONLY the brief's target files changed — nothing under
   `tests/perf/golden/`, no stray files, no edits outside the brief's "Files" list.

5. **§7.3 dual-implementation** — for T1/T3 items, BOTH the object-dtype path (`df_utils.py`) AND
   the categorical/QDF-native twins (`qdf/methods.py` + `qdf/classes.py`) must be fixed; a one-sided
   fix leaves the categorical hot path slow. Verify both were changed and both keep parity. For a
   dtype-independent item (T2c/T2/T4/T6), mark this dimension **N/A** with a brief reason.

6. **Implementation soundness** — the change matches the TARGETS design's intent (e.g. T2c builds
   the ticker on observed unique code-pairs rather than per row; T1/T3 dedup/sort on factor codes;
   T4 keeps a serial default and parallelizes at the per-(sig,ret) boundary). It introduces no
   obvious correctness bug, no nondeterminism, and no hidden behavioural change the parity tests
   might miss (e.g. error-message text, warning emission, mutation of an input frame). Cite evidence.

## Severity

- **Blocker** — must be fixed before merge. A finding is a blocker if it:
  - Fails any parity / edge / `inspect.signature` test, or changes the public API (dim 2).
  - Regenerates or hand-edits a `tests/perf/golden/*` artifact (dim 2).
  - Shows a benchmark regression or no win (dim 3).
  - Fails the macrosynergy suite (dim 4).
  - Edits files outside the brief's target list (dim 4).
  - Violates a brief acceptance criterion (dim 1).
  - Fixes only one of the two §7.3 implementations when both are required (dim 5).
  - Introduces a correctness/nondeterminism/hidden-behaviour defect (dim 6).

- **Nit** — noted, does not block: a stylistic deviation not enforced by tests/brief, a missed
  micro-optimization, or a minor naming/comment issue with no functional consequence.

`APPROVE` requires zero blockers. A single blocker forces `CHANGES`. Do not upgrade a nit to a
blocker unless it crosses a criterion above. Do not impose preferences not grounded in the brief,
TARGETS.md, or the repo's existing conventions.

## Verdict format

Emit exactly this block at the end, verbatim structure:

```
VERDICT: APPROVE | CHANGES
SUMMARY: <one sentence, ≤20 words>
BLOCKERS:
  - [dim #] <issue> — <file:loc> — <fix needed>
NITS:
  - [dim #] <issue> — <loc>
```

Rules:
- Exactly one of `APPROVE` / `CHANGES` on the `VERDICT:` line.
- `BLOCKERS:` then `- none` when there are none; `NITS:` then `- none` when there are none.
- Keep entries terse with the fix inside the entry; the manager acts only on the verdict line and
  the verbatim blocker entries.
- `<file:loc>` is a repo-relative path + line/function.
- State N/A dimensions explicitly with a reason (do not omit dim 5 when it is N/A).

## You do NOT

- **Edit the work** — report, do not repair. Your tools exclude `Write`/`Edit`.
- **Run `git`/`gh` mutating commands** — `git status`/`git diff` read-only only.
- **Accept the builder's self-report** — re-run the GATE yourself (dims 2–4).
- **Accept a regenerated golden** — it is a blocker.
- **Invent preferences not grounded in the brief, TARGETS.md, or repo conventions** — at most a nit.
- **Conflate N/A with passing** — say so explicitly with a reason.
