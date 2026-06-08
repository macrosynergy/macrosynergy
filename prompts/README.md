# `prompts/` — implementation plans for agent-driven work

This folder holds **self-contained implementation plans** written for AI coding agents (and humans) to execute task-by-task. Each plan is a TDD-structured spec: exact files, exact code, exact test commands, and frequent commits. One file per feature, named `YYYY-MM-DD-<feature-slug>.md`.

## Current plans

| Plan | Feature | Branch |
| --- | --- | --- |
| [`2026-06-08-sharpe-ssr-single-statistic-table.md`](./2026-06-08-sharpe-ssr-single-statistic-table.md) | Sharpe ratio & Sharpe Stability Ratio (SSR) statistics for `SignalReturnRelations.single_statistic_table` | `feature/srr-sharpe-ssr` |

## How to implement a plan

### 1. Environment

```bash
cd ~/repos/macrosynergy            # or your clone path
git checkout feature/srr-sharpe-ssr
python -m venv .venv && source .venv/bin/activate   # if you don't already have an env
pip install -e ".[test]"           # editable install + test extras (see pyproject.toml)
```

### 2. Read the plan top-to-bottom first

Every plan opens with **Goal / Architecture / Tech Stack**, then a **"Context the implementer must read first"** section listing the exact source locations (with line numbers) you must read before writing code. Read those files in the repo at their cited lines — the plan quotes them, but the code is the source of truth and may have shifted. Then read the **Design decisions** and **File structure** sections so you understand what is in and out of scope.

### 3. Execute task-by-task (TDD, red → green → commit)

Each task is a sequence of bite-sized steps. Follow them literally:

1. **Write the failing test** (copy the test from the step).
2. **Run it and confirm it fails** with the stated message.
3. **Write the minimal implementation** (copy the code from the step).
4. **Run the test and confirm it passes.**
5. **Commit** with the conventional-commit message given in the step.

Run a single test exactly as the plan specifies, e.g.:

```bash
pytest tests/unit/signal/test_signal_return_relations.py::TestAll::test_freq_to_rebal_freq -v
```

Do **not** skip the "run and confirm it fails" step — a test that passes before you implement anything is testing the wrong thing.

### 4. Recommended driver: subagent-driven development

For agentic execution, use the **`superpowers:subagent-driven-development`** skill: dispatch one fresh subagent per task, review the diff between tasks, then proceed. For inline execution in a single session, use **`superpowers:executing-plans`** (batch with checkpoints). Both are referenced in the plan header.

### 5. Final verification before opening a PR

Run the full relevant suites (the plan's last task lists these):

```bash
pytest tests/unit/signal/test_signal_return_relations.py -rEf --verbose   # feature + regressions
pytest tests/unit/pnl/ -rEf --verbose                                     # reused modules unchanged
pytest tests/unit/signal/ -rEf --verbose                                  # broader signal suite
```

Then walk the plan's **Self-review checklist** and the **Acceptance criteria** snippet — the latter is the real-world use case the feature must satisfy.

## Conventions

- **Branch per plan.** Never implement a plan directly on `develop`.
- **Conventional commits.** `feat(...)`, `fix(...)`, `docs(...)`, etc. PR titles/descriptions feed the auto-generated release notes (`docs/release_notes.py`) — there is **no** hand-maintained changelog to edit.
- **No silent scope creep.** If a plan's cited line numbers or code no longer match the repo, stop and reconcile (the code wins); note the drift in the PR.
- **Open decisions** flagged inside a plan should be surfaced in the PR description for the reviewer, even when a sensible default was chosen.
