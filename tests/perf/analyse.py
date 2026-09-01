"""Add derived statistics to a pytest-benchmark JSON payload, in place by default."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from tests.perf.results import BenchmarkRunResults


def main(argv: Optional[Sequence[str]] = None) -> int:
    """
    Enrich one results file with the statistics the suite derives.

    Parameters
    ----------
    argv : Optional[Sequence[str]]
        Command-line arguments, or None to read `sys.argv`.

    Returns
    -------
    int
        Zero, the process exit status.

    Raises
    ------
    ValueError
        If an entry's `extra_info` is incomplete or two measurements share an
        identifier.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="pytest-benchmark JSON to enrich")
    parser.add_argument("-o", "--out", type=Path, help="write here instead of in place")
    parser.add_argument(
        "--drop-round-timings", action="store_true", help="omit the per-round samples"
    )
    arguments = parser.parse_args(argv)

    results = BenchmarkRunResults.load(arguments.path)
    payload = results.with_derived_statistics(
        drop_round_timings=arguments.drop_round_timings
    )
    (arguments.out or arguments.path).write_text(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
