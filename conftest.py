"""Root conftest. Declares the perf suite's command-line options, and nothing else.

Why this file exists at all: pytest honours `pytest_addoption` only in *initial*
conftest files, meaning the rootdir conftest plus conftests in directories named on the
command line. `testpaths` is not set, so defining `--perf-tiers` only in
`tests/perf/conftest.py` would make it work for `pytest tests/perf ...` but fail with
"unrecognized arguments" for `pytest --perf-tiers=small` with no path argument.

Deliberately inert. No fixtures, no collection hooks, no imports of project or test
modules, so it adds nothing to the default gate and cannot affect collection. The option
defaults to `None`, and the perf suite falls back to `MACROSYN_PERF_TIERS` and then to
its own default, so behaviour with the flag absent is identical to having no root
conftest at all.
"""


def pytest_addoption(parser):
    parser.addoption(
        "--perf-tiers",
        action="store",
        default=None,
        metavar="TIERS",
        help=(
            "Comma-separated data-scale tiers for the performance suite, from "
            "tiny,small,medium,large. Defaults to $MACROSYN_PERF_TIERS, then to the "
            "suite default. Affects only tests under tests/perf."
        ),
    )
