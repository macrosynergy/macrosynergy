---
name: building-perf-tasks
description: Procedure for building ONE queued macrosynergy performance optimization (a perf/<slug> item) end-to-end in an isolated worktree — read the brief + the TARGETS.md design, change only the target files, preserve the public API and byte-identical output, then self-verify against the in-repo GATE (tests/perf parity/edge/API stay green + a -m perf benchmark win + the macrosynergy suite). Used by the perf-builder agent.
---

This skill defines the end-to-end build procedure for a single optimization item from
`prompts/QUEUE.md`. The item lives in an isolated git worktree off `feature/performance`. Follow the
steps in sequence: read context, change code, run the GATE, self-verify, hand back. You do not
commit, push, or open PRs — that is the manager's job.

## What you build

Exactly one optimization, drawn from the queue. Its brief (`prompts/perf/<id>-*.md`) names:

- **Goal / design** — the specific change, with the design spelled out in `prompts/TARGETS.md` (the
  `Tx` section the brief references). The brief + TARGETS design are your requirements.
- **Target files** — the exact package files to modify (and any `tests/perf` additions the brief
  asks for). Touch only those.
- **Verification (the GATE)** — the commands whose clean result confirm the item is done.
- **depends-on** — items that must be `DONE` first. If one is missing, stop and report BLOCKED.

You work only in your worktree. Never reach into another worktree or modify files outside the
brief's target list.

## Read first (before changing any code)

1. **The brief**: `prompts/perf/<id>-*.md`. Your requirements document.
2. **The TARGETS.md design** it references (`prompts/TARGETS.md` §`Tx` + §3, and §5/§5.1 for the
   GATE, §7.3 for the dual-implementation rule). The design gives the proposed change, the
   files/line numbers, and the parity contract.
3. **The function(s) you are changing**, in full, plus their call sites if the change could affect
   them. Read the existing tests that pin them: the appended classes in `tests/unit/management/` /
   `tests/unit/signal/` and the item's `tests/perf/test_parity_*.py`.
4. **`tests/perf/README.md`** — how the framework runs (the `-m perf` benchmark, `record.py`, the
   default-gate parity/edge/API tests, scale tiers).

## Optimization conventions (non-negotiable)

- **Output-identical by contract.** The change must preserve the public API (names, signatures,
  parameter semantics, return types) AND produce byte-identical output (same rows, order, dtype;
  same `Categorical` category set + order; same numeric values). The `tests/perf/test_parity_*` +
  edge + `inspect.signature` tests encode this — they must stay green.
- **NEVER regenerate the parity goldens.** Do not run `python tests/perf/capture_parity.py --update`
  and do not hand-edit `tests/perf/golden/*`. They encode the pre-change output; your job is to
  reproduce them, not move them. A failing golden means your change altered output — fix the change.
- **Fix BOTH implementations where TARGETS §7.3 applies (T1/T3).** The object-dtype path lives in
  `macrosynergy/management/utils/df_utils.py`; the categorical/QDF-native twins in
  `macrosynergy/management/types/qdf/methods.py` + `qdf/classes.py`. Both are hot; fix both, and keep
  each path's output identical. (T2c/T2 are dtype-independent — single function.)
- **Stay inside the target files.** If you find a related bug or a second hot path the brief didn't
  list, report it as a finding — do not silently widen scope.
- **No AI attribution / authorship comments** in source.

## Verification — the GATE (mandatory, before you hand back)

Run ALL of the following from the worktree root (use `--no-cov -n0`; never `-p no:cov`). A single
failure blocks hand-back.

**1. Behaviour preserved (GATE-1 API + GATE-2 parity) — must be GREEN:**
```bash
pytest tests/perf/test_parity_<item>.py -v --no-cov -n0
pytest tests/unit/management tests/unit/signal -m "not perf" --no-cov -n0   # incl. the edge/API tripwires
```
The parity goldens and the `inspect.signature` tripwires must pass unchanged. If a signature test
fails, you changed the API — revert that. If a parity test fails, you changed output — fix the change.

**2. Measurable win (GATE-3):** run the item's benchmark and diff against the manager-supplied
baseline JSON:
```bash
pytest tests/perf/test_perf_<item>.py -m perf -k <tier> --benchmark-only -n0 --no-cov \
  --benchmark-json=<scratch>/after.json
python tests/perf/record.py <baseline-json> <scratch>/after.json
```
Confirm a wall (and/or, where the item targets memory, RSS) improvement — not a regression. Paste the
record.py delta in your report. (If `record.py` prints the cross-machine banner, you ran the baseline
on a different machine — re-capture on this one.)

**3. macrosynergy suite passes (GATE-4):**
```bash
pytest <suite-scope-the-brief-names>   # e.g. the affected tests/unit subtrees, or the full suite
```

**4. Working-tree hygiene:**
```bash
git status
# Confirm: only the brief's target files changed; NO tests/perf/golden/* modified; no scratch files.
```

**Acceptance criteria check:** invoke `superpowers:verification-before-completion` and work through
the brief's acceptance criteria one by one, citing observable evidence (command output, line number,
benchmark delta) for each. Never claim DONE from reading code alone.

## You do NOT

- Run `git add`, `git commit`, `git push`, or any `gh` command — the manager owns git.
- Regenerate or hand-edit `tests/perf/golden/` (see conventions).
- Edit `prompts/QUEUE.md`, `prompts/TARGETS.md`, another item's brief, or planning docs.
- Edit files outside the brief's target list — report unrelated bugs as findings.
- Leave scratch files in the repo. Use the session scratchpad directory.
- Widen scope beyond the brief on your own initiative — stop and report instead.
