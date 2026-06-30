"""Diff two pytest-benchmark JSON files into a QUEUE-ready markdown table, with env guard."""

from __future__ import annotations

import json
import sys
from typing import Optional

from tests.perf.env import comparable


def load_benchmark(path: str) -> dict:
    with open(path) as fh:
        return json.load(fh)


def extract_env(bench_json: dict) -> dict:
    return bench_json.get("machine_info", {}).get("macrosynergy_env", {})


def _index_by_name(bench_json: dict) -> dict:
    return {b["name"]: b["stats"]["mean"] for b in bench_json.get("benchmarks", [])}


def render_report(baseline: dict, branch: dict) -> str:
    env_a, env_b = extract_env(baseline), extract_env(branch)
    is_comparable = comparable(env_a, env_b)
    lines = []
    if not is_comparable:
        lines.append("> ⚠ **cross-machine — advisory only** "
                     f"(baseline `{env_a.get('cpu_brand')}` vs branch `{env_b.get('cpu_brand')}`)")
        lines.append("")
    base_idx, br_idx = _index_by_name(baseline), _index_by_name(branch)
    if is_comparable:
        lines.append("| benchmark | baseline (s) | branch (s) | change |")
        lines.append("|---|--:|--:|--:|")
    else:
        lines.append("| benchmark | baseline (s) | branch (s) |")
        lines.append("|---|--:|--:|")
    for name in sorted(set(base_idx) | set(br_idx)):
        b0, b1 = base_idx.get(name), br_idx.get(name)
        b0s = f"{b0:.4f}" if b0 is not None else "—"
        b1s = f"{b1:.4f}" if b1 is not None else "—"
        if is_comparable and b0 and b1:
            pct = (1 - b1 / b0) * 100
            lines.append(f"| {name} | {b0s} | {b1s} | {pct:.0f}% |")
        elif is_comparable:
            lines.append(f"| {name} | {b0s} | {b1s} | — |")
        else:
            lines.append(f"| {name} | {b0s} | {b1s} |")
    return "\n".join(lines)


def main(argv=None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if len(argv) != 2:
        print("usage: python tests/perf/record.py <baseline.json> <branch.json>")
        return 2
    print(render_report(load_benchmark(argv[0]), load_benchmark(argv[1])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
