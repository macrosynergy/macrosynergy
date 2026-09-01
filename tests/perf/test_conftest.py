"""
Tests for the perf-suite pytest wiring. The subprocess cases are `perf`-marked because
each pays about twelve seconds importing the package.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pytest
from parameterized import parameterized

from tests.perf.panel_sizes import PANEL_SIZES

PERF_DIR = Path(__file__).resolve().parent
REPO_ROOT = PERF_DIR.parents[1]

PROBE = (
    "import pytest\n"
    "@pytest.mark.perf\n"
    "@pytest.mark.perf_group('probe')\n"
    "def test_probe(benchmark):\n"
    "    benchmark(lambda: sum(range(10)))\n"
)


def run_pytest(*args: str, env: dict = None) -> subprocess.CompletedProcess:
    """
    Run pytest in a subprocess from the repository root.

    Parameters
    ----------
    *args : str
        Arguments to pass to pytest.
    env : dict
        Environment variables to add to the inherited environment.

    Returns
    -------
    subprocess.CompletedProcess
        The finished process, with stdout and stderr captured as text.
    """
    return subprocess.run(
        [sys.executable, "-m", "pytest", *args],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=600,
        env={**os.environ, **(env or {})},
    )


# These two are pytest-native functions because pytest cannot inject fixtures into
# `TestCase` methods. The rest of this module is `TestCase`, like the wider suite.


def test_machine_profile_fixture(perf_machine_profile):
    assert perf_machine_profile.fingerprint
    assert perf_machine_profile.usable_cpu_count


def test_selected_tiers_match_the_invocation(pytestconfig):
    spec = pytestconfig.getoption("--perf-tiers", default=None) or os.environ.get(
        "MACROSYN_PERF_TIERS"
    )
    expected = (
        tuple(part.strip() for part in spec.split(",")) if spec else ("small", "medium")
    )
    assert PANEL_SIZES.selected_tiers == expected


class TestConftestSource(unittest.TestCase):
    def test_conftest_declares_no_module_level_benchmark_hook(self):
        """A hookspec is unregistered when the plugin is absent, which aborts collection."""
        tree = ast.parse((PERF_DIR / "conftest.py").read_text())
        offenders = [
            node.name
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name.startswith("pytest_benchmark")
        ]
        self.assertEqual(
            offenders,
            [],
            f"{offenders} must live on a plugin object registered in pytest_configure",
        )

    def test_root_conftest_is_inert(self):
        source = (REPO_ROOT / "conftest.py").read_text()
        self.assertIn("pytest_addoption", source)
        for forbidden in (
            "import ",
            "@pytest.fixture",
            "pytest_configure",
            "pytest_collection",
        ):
            self.assertNotIn(forbidden, source)


@pytest.mark.perf
class TestSubprocessWiring(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.tmp_path = Path(directory.name)

    def write_probe(self, name: str, source: str = PROBE) -> Path:
        """
        Write a throwaway test module inside tests/perf, where the perf conftest applies.

        Parameters
        ----------
        name : str
            Suffix distinguishing this module from other probes.
        source : str
            Module source to write.

        Returns
        -------
        Path
            The written module, removed again when the test finishes.
        """
        path = PERF_DIR / f"test_zz_probe_{name}.py"
        path.write_text(source)
        self.addCleanup(path.unlink, True)
        return path

    def test_extra_info_is_complete_and_matches_the_dataframe(self):
        module = self.write_probe(
            "stamp",
            "import pytest\n"
            "from tests.perf.panel_sizes import PANEL_SIZES\n"
            "@pytest.mark.perf\n"
            "@pytest.mark.perf_group('wiring')\n"
            "@pytest.mark.parametrize('panel_size', PANEL_SIZES.qdf_sizes('tiny'), ids=str)\n"
            "@pytest.mark.parametrize('dtype', ['object'], ids=str)\n"
            "def test_stamped(benchmark, panel_size, dtype):\n"
            "    df = panel_size.as_qdf()\n"
            "    benchmark(lambda: None)\n"
            "    benchmark.extra_info['rows_seen'] = len(df)\n",
        )
        out = self.tmp_path / "b.json"
        result = run_pytest(
            str(module), "-m", "perf", "-n0", "--no-cov", "-q",
            "--perf-tiers=tiny", f"--benchmark-json={out}",
        )
        self.assertTrue(out.exists(), result.stdout[-2500:])
        info = json.loads(out.read_text())["benchmarks"][0]["extra_info"]
        self.assertEqual(info["group"], "wiring")
        self.assertEqual(info["benchmark_name"], "stamped")
        self.assertEqual(info["variant"], {"dtype": "object"})
        self.assertEqual(info["timing_mode"], "calibrated")
        self.assertEqual(info["panel_size"]["df_format"], "long")
        self.assertEqual(info["panel_size"]["row_count"], info["rows_seen"])
        self.assertEqual(info["panel_size"]["observation_count"], info["rows_seen"])

    def test_host_is_stamped_into_machine_info(self):
        module = self.write_probe("machine")
        out = self.tmp_path / "b.json"
        run_pytest(
            str(module), "-m", "perf", "-n0", "--no-cov", "-q",
            f"--benchmark-json={out}",
        )
        profile = json.loads(out.read_text())["machine_info"]["macrosynergy"]
        self.assertTrue(profile["fingerprint"])
        self.assertTrue(profile["usable_cpu_count"])

    def test_host_is_stamped_when_the_conftest_loads_lazily(self):
        self.write_probe("lazy", PROBE.replace("test_probe", "test_probe_lazy"))
        out = self.tmp_path / "b.json"
        run_pytest(
            "tests", "-m", "perf", "-n0", "--no-cov", "-q",
            "--ignore=tests/unit/learning", "-k", "test_probe_lazy",
            f"--benchmark-json={out}",
        )
        machine = json.loads(out.read_text())["machine_info"]["macrosynergy"]
        self.assertTrue(machine["fingerprint"])

    def test_missing_group_fails_loudly(self):
        module = self.write_probe(
            "nogroup", PROBE.replace("@pytest.mark.perf_group('probe')\n", "")
        )
        result = run_pytest(str(module), "-m", "perf", "-n0", "--no-cov", "-q")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("perf_group", result.stdout + result.stderr)

    def test_collection_survives_the_plugin_being_disabled(self):
        result = run_pytest("-p", "no:benchmark", "-n0", "--no-cov", "-q", "--co")
        combined = result.stdout + result.stderr
        self.assertNotIn("INTERNALERROR", combined)
        self.assertNotIn("unknown hook", combined)

    def test_perf_tiers_is_accepted_without_a_path_argument(self):
        result = run_pytest("--perf-tiers=small", "-n0", "--no-cov", "-q", "--co")
        self.assertNotIn("unrecognized arguments", result.stdout + result.stderr)

    @parameterized.expand(["option", "env"])
    def test_unknown_tier_is_rejected_by_name(self, via):
        args = ["tests/perf/test_panel_sizes.py", "-n0", "--no-cov", "-q"]
        env = None
        if via == "option":
            args.append("--perf-tiers=nonsense")
        else:
            env = {"MACROSYN_PERF_TIERS": "nonsense"}
        result = run_pytest(*args, env=env)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("nonsense", result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
