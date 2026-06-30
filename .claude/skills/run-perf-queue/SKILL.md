---
name: run-perf-queue
description: Run the macrosynergy performance optimization queue — drain prompts/QUEUE.md of perf/<slug> items in dependency order, spawning a worktree-isolated perf-builder and an adversarial perf-reviewer per item, integrating approved work serially via per-item PRs into feature/performance. The in-repo tests/perf framework (TARGETS §5.1) is the GATE. Invoke as "run-perf-queue" (drives the next claimable item) or "run-perf-queue Q1" to target a specific item; optional "N=<k>" sets parallelism (default N=1).
---

## Role

You are the manager. You own all git/gh and the queue; the agents own the build and the review. You
drain `prompts/QUEUE.md` respecting `depends-on` order, building items in **manager-provisioned
worktrees off `feature/performance`**, gating each on the in-repo `tests/perf` framework, and
integrating serially.

Do not build or review anything yourself. Do not touch git on behalf of an agent. The `perf-builder`
agent changes package code; the `perf-reviewer` agent reads and re-runs the GATE; you alone run
`git`, `gh`, queue edits, and the baseline benchmark capture.

## Default parallelism: N=1 (serial)

Unlike the research/parse pipelines, these items optimize **shared, broad-blast-radius package code**
(`df_utils.py`, `qdf/methods.py`, `qdf/classes.py`, `signal_return_relations.py`) and several items
touch the same files (T1/T3 both, the §7.3 twins). **Default N=1 — strictly sequential.** Only raise
`N` if the user explicitly passes `N=<k>` AND the selected items touch disjoint files. When in doubt,
stay serial.

## Resolve the target

- `run-perf-queue` → drive the next claimable item (lowest Q-number that is claimable).
- `run-perf-queue Q<k>` → target that specific item (must be claimable).
- Optional `N=<k>` after the argument raises parallelism (see above; default 1).

Read `prompts/QUEUE.md`. If there are no claimable items, stop and print:

```
INFO: no claimable items in prompts/QUEUE.md. Nothing to do.
```

## Dependency-aware selection

An item (Q-row) is **claimable** when both hold:

1. Its status is `TODO` (not `IN PROGRESS` / `IN REVIEW` / `DONE` / `BLOCKED`).
2. Every item in its "depends on" note is already `DONE` — e.g. Q6 (T5) and Q8 (T7) depend on Q4
   (T3); Q1–Q5 and Q7 have no dependency.

Select the claimable item(s) per the parallelism rule. For each selected item, set its status to
`IN PROGRESS` in `QUEUE.md` and save before spawning agents. Record: the item ID (e.g. `Q1`), its
slug (e.g. `qdf-ticker-series-vectorize` → branch `perf/<slug>`), the brief path
(`prompts/perf/<id>-*.md`), and the target files + affected `tests/perf` module(s) from the brief.

> If the item has no detailed brief at `prompts/perf/<id>-*.md`, stop and ask the human to author one
> (the terse QUEUE row + the TARGETS design are the inputs; the brief is the builder's contract).

## Phase 0 — Capture the pre-change benchmark baseline (you only)

GATE-3 (measurable win) is a before/after on the **same machine**. Before dispatching the builder,
capture the item's benchmark on the current `feature/performance` HEAD from the **main checkout**:

```bash
pytest <item-perf-module> -m perf -k <tier> --benchmark-only -n0 --no-cov \
  --benchmark-json=.claude/perf-baselines/<id>-baseline.json
```

(`<item-perf-module>` and `<tier>` come from the brief, e.g. `tests/perf/test_perf_qdf_ticker_series.py`
+ `small`.) This baseline JSON path is passed to both the builder and the reviewer. `tests/perf/record.py`
guards comparability via the environment fingerprint.

## Phase 1 — Build (manager-provisioned worktree)

> **CRITICAL — do NOT use the Agent tool's `isolation: "worktree"`.** It forks from the repo's
> default branch (`origin/HEAD`), not `feature/performance`, and can silently discard work. The
> manager provisions worktrees itself.

Pre-create one worktree per claimed item, off the base branch — the worktree branch doubles as the
integration branch `perf/<slug>` (matches the TARGETS sub-branch naming, so Phase 4 needs no copy):

```bash
git worktree add -b perf/<slug> .claude/worktrees/perf-<slug> feature/performance
```

Then spawn one `perf-builder` agent **without** `isolation` (plain, non-isolated), instructing it to
work exclusively inside that worktree. Issue all builder calls in a single message if N>1. Pass each:

```
Brief: prompts/perf/<id>-*.md
Baseline benchmark JSON (GATE-3 "before"): .claude/perf-baselines/<id>-baseline.json
Work ONLY inside this worktree (cd into it first; touch no other path):
  .claude/worktrees/perf-<slug>
```

The builder applies the design, runs the GATE, and hands back a terse report (changed paths; one-line
summary + which implementation(s) it touched; parity/API caveats; the GATE result — parity/edge/API
test counts, the record.py before/after delta, the suite summary). If a builder reports `BLOCKED`,
mark the row `BLOCKED` in `QUEUE.md`, do not advance it, and continue with any remaining items.

## Phase 2 — Review

For each item that completed Phase 1 without `BLOCKED`, spawn one `perf-reviewer` agent. Pass it the
brief path, the worktree path, and the baseline benchmark JSON path. The reviewer is adversarial and
read-only; it re-runs the GATE and returns:

```
VERDICT: APPROVE | CHANGES
SUMMARY: <one sentence>
BLOCKERS:
  - [dim #] <issue> — <file:loc> — <fix needed>
NITS:
  - [dim #] <issue> — <loc>
```

`APPROVE` requires zero blockers. Set the row to `IN REVIEW` while reviewing.

## Phase 3 — Revise (≤3 rounds)

For each `CHANGES`:

1. Re-spawn `perf-builder` (non-isolated) at the **same worktree** (do not create a new one); pass
   the verbatim `BLOCKERS` list as the revision prompt alongside the brief + baseline JSON.
2. Re-spawn `perf-reviewer` at the same worktree, re-running the full GATE (a fix can regress parity).
3. Repeat until `APPROVE` or 3 rounds are exhausted. After 3 rounds with no `APPROVE`, mark the row
   `BLOCKED` with the unresolved blockers in its notes, leave the worktree in place, and carry on.

## Phase 4 — Integrate (serial, you only)

Process approved items one at a time. The worktree branch `perf/<slug>` already holds the change.

1. **Confirm base is current.** Fetch `feature/performance`; if it advanced, rebase the worktree
   branch onto it (`git -C .claude/worktrees/perf-<slug> rebase origin/feature/performance`).
2. **Stage only the brief's target files by explicit path — never `git add -A`.** Run git from inside
   the worktree (`git -C .claude/worktrees/perf-<slug> add <path1> <path2> ...`). Verify with
   `git status` that only those files are staged and that **no `tests/perf/golden/*` artifact** is
   staged (the optimization must reproduce existing goldens, not change them).
3. **Commit** with a conventional prefix + the Co-Authored-By trailer:
   ```bash
   git commit -m "$(cat <<'EOF'
   perf(<area>): <one-line summary> [<id>]

   Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
   EOF
   )"
   ```
4. **Push and open a PR targeting `feature/performance` — never `develop` or `main`:**
   ```bash
   git push -u origin perf/<slug>
   gh pr create --base feature/performance --head perf/<slug> \
     --title "perf(<area>): <summary> [<id>]" \
     --body "$(cat <<'EOF'
   ## Item
   <id> · <slug> (TARGETS <Tx>)

   ## Changed files
   <list from builder report>

   ## Reviewer verdict
   APPROVE — zero blockers. Rounds: <n>.

   ## GATE
   Parity/edge/API: <counts>. Benchmark before/after (record.py): <delta>. Suite: <summary>.

   🤖 Generated with [Claude Code](https://claude.com/claude-code)
   EOF
   )"
   ```
5. **Merge on clean APPROVE:** `gh pr merge <pr-number> --squash --delete-branch`.
6. **Re-run the GATE on the integrated tree** (on `feature/performance` after merge): the default
   `pytest` gate over `tests/perf` + affected `tests/unit` (parity must still hold) and the
   macrosynergy suite. If it fails post-merge, do not proceed; diagnose/fix before continuing.
7. **Record the win + prune.** Fill the item's before/after row in `QUEUE.md` (from record.py), set
   the row to `DONE`, and prune the brief. Commit on `feature/performance`:
   ```bash
   git checkout feature/performance
   git add prompts/QUEUE.md && git commit -m "chore(perf): record <id> result; mark DONE"
   git push
   ```
8. **Remove the worktree** (branch already deleted by `--delete-branch`):
   ```bash
   git worktree remove .claude/worktrees/perf-<slug> --force && git worktree prune
   ```

## Output-identity guard

The whole point of these items is **byte-identical output**. After each integration, confirm the
parity goldens still match on the merged tree (`pytest tests/perf/test_parity_*.py -m "not perf"
--no-cov -n0`) and that `tests/perf/golden/` was not modified. If any parity test fails or a golden
changed, stop integration and flag the item — the optimization changed behaviour and is not mergeable.

## Report

After draining (or reaching the end of claimable items), emit:

```
=== run-perf-queue ===
Drove: <id1>, <id2>, ...
Per-item:
  <id1>: PR #<n> — MERGED — <n> round(s) — win: <wall/mem delta> — paths: <files>
  <id2>: BLOCKED — 3 rounds — unresolved: <blockers>
Queue state:
  Done this run: <count> | Blocked: <count> | Remaining TODO: <count>
```

All merged work stays on `feature/performance`. No `develop`/`main` promotion — a human decides that.

## Guardrails

- **Only `feature/performance` as PR base.** Never `--base develop`/`--base main`. A wrong-base PR is
  closed immediately.
- **Output-identity is sacred.** Never accept (or instruct) a regenerated `tests/perf/golden/*`. The
  goldens are the pre-optimization contract; the change reproduces them.
- **Never force-push.**
- **Never merge with open blockers.** 3 rounds without APPROVE → the PR stays open/draft, item BLOCKED.
- **Never `git add -A` while worktrees are unintegrated.** Stage the item's target files by explicit
  path, after verifying `git status`.
- **Agents never touch git or the queue.** `perf-builder`/`perf-reviewer` are structurally prevented
  from git/gh and from editing `QUEUE.md`. Do not ask them to, and do not accept a builder report
  claiming it committed.
- **Default serial (N=1).** Raise parallelism only on explicit `N=<k>` AND disjoint target files.
- **Clean up worktrees.** Remove every provisioned worktree by run end (`git worktree remove --force`
  + `git worktree prune`).
- **Do not widen integration scope.** Stage only files in the brief's "Files" section, confirmed in
  the builder's changed-paths report.
