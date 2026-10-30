"""
Tests for `tests.perf.harness` and `tests/perf/conftest.py`.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest
from parameterized import parameterized

from tests.perf.harness.machine import REPOSITORY_ROOT, MachineProfile
from tests.perf.harness.measurement import (
    PeakMemoryTracker,
    describe_argument,
    describe_call,
)
from tests.perf.harness.panel_sizes import (
    PANEL_SIZES,
    TARGET_OBSERVATION_COUNTS,
    PanelSize,
    clear_df_cache,
)

# -------------------------------------------------------------------------- panel sizes


OBSERVATION_TOLERANCE = 0.10
CHEAP_TIERS = ["tiny", "small"]
EXPENSIVE_TIERS = ["medium", "large"]


def a_panel_size(**overrides) -> PanelSize:
    """
    A small `PanelSize` for tests that do not care about the exact numbers.

    Parameters
    ----------
    **overrides
        Fields to replace on the default size.

    Returns
    -------
    PanelSize
        The default size with `overrides` applied.
    """
    defaults = dict(
        tier="test",
        date_count=10,
        min_rounds=1,
        max_seconds=1.0,
        cid_count=2,
        xcat_count=3,
    )
    return PanelSize(**{**defaults, **overrides})


class TestPanelSizeValidation(unittest.TestCase):
    def test_rejects_both_ticker_forms(self):
        with self.assertRaisesRegex(ValueError, "not both forms and not neither"):
            a_panel_size(tickers=("AAA_XCAT000",))

    def test_rejects_neither_ticker_form(self):
        with self.assertRaisesRegex(ValueError, "not both forms and not neither"):
            PanelSize(tier="test", date_count=10, min_rounds=1, max_seconds=1.0)

    @parameterized.expand(
        [
            ("missing_xcat_count", {"cid_count": 2, "xcat_count": None}),
            ("missing_cid_count", {"cid_count": None, "xcat_count": 3}),
        ]
    )
    def test_rejects_one_half_of_the_parts_form(self, _name, overrides):
        with self.assertRaisesRegex(ValueError, "both `cid_count` and `xcat_count`"):
            PanelSize(
                tier="test", date_count=10, min_rounds=1, max_seconds=1.0, **overrides
            )

    def test_rejects_duplicate_tickers(self):
        with self.assertRaisesRegex(ValueError, "duplicates"):
            a_panel_size(cid_count=None, xcat_count=None, tickers=("AAA_X", "AAA_X"))

    def test_rejects_non_positive_date_count(self):
        with self.assertRaisesRegex(ValueError, "must be positive"):
            a_panel_size(date_count=0)


class TestPanelSizeCounting(unittest.TestCase):
    def test_ticker_count_from_parts(self):
        self.assertEqual(a_panel_size(cid_count=2, xcat_count=3).ticker_count, 6)

    def test_ticker_count_and_observations_from_explicit_tickers(self):
        size = a_panel_size(cid_count=None, xcat_count=None, tickers=("AAA_X", "BBB_Y"))
        self.assertEqual(size.ticker_count, 2)
        self.assertEqual(size.observation_count, 20)

    def test_shape_differs_between_formats(self):
        size = a_panel_size()
        self.assertEqual(size.qdf_shape, (60, 4))
        self.assertEqual(size.ticker_df_shape, (10, 6))
        self.assertEqual(size.shape, size.qdf_shape)
        self.assertEqual(replace(size, is_ticker_df=True).shape, size.ticker_df_shape)

    def test_df_format_names_the_selected_format(self):
        self.assertEqual(a_panel_size().df_format, "long")
        self.assertEqual(replace(a_panel_size(), is_ticker_df=True).df_format, "wide")

    def test_panel_size_is_usable_as_a_cache_key(self):
        self.assertEqual(len({a_panel_size(), a_panel_size()}), 1)

    def test_describe_reports_the_selected_format(self):
        described = replace(a_panel_size(), is_ticker_df=True).describe()
        self.assertEqual(described["df_format"], "wide")
        self.assertEqual((described["row_count"], described["column_count"]), (10, 6))
        self.assertEqual(described["metrics"], ["value"])


class TestPanelSizeCatalog(unittest.TestCase):
    def setUp(self) -> None:
        previous = PANEL_SIZES.selected_tiers
        self.addCleanup(setattr, PANEL_SIZES, "selected_tiers", previous)

    def test_tier_names_lists_every_registered_tier(self):
        self.assertEqual(PANEL_SIZES.tier_names, ("tiny", "small", "medium", "large"))

    @parameterized.expand(list(TARGET_OBSERVATION_COUNTS))
    def test_each_tier_lands_on_its_observation_target(self, tier):
        target = TARGET_OBSERVATION_COUNTS[tier]
        actual = PANEL_SIZES[tier].observation_count
        self.assertLessEqual(abs(actual - target) / target, OBSERVATION_TOLERANCE)

    def test_parameters_carry_the_measurement_budget(self):
        parameter = PANEL_SIZES.qdf_sizes("small")[0]
        marks = {mark.name: mark.kwargs for mark in parameter.marks}
        self.assertEqual(marks["benchmark"], {"min_rounds": 15, "max_time": 3.0})

    def test_ticker_df_sizes_are_marked_as_such(self):
        self.assertTrue(PANEL_SIZES.ticker_df_sizes("small")[0].values[0].is_ticker_df)
        self.assertFalse(PANEL_SIZES.qdf_sizes("small")[0].values[0].is_ticker_df)

    def test_a_tier_limit_outside_the_selection_yields_one_explained_skip(self):
        PANEL_SIZES.select_tiers("medium")
        parameters = PANEL_SIZES.qdf_sizes("tiny")
        self.assertEqual(len(parameters), 1)
        reason = parameters[0].marks[0].kwargs["reason"]
        self.assertIn("tiny", reason)
        self.assertIn("medium", reason)

    def test_select_tiers_names_an_unknown_tier(self):
        with self.assertRaisesRegex(ValueError, "enormous"):
            PANEL_SIZES.select_tiers("enormous")

    def test_select_tiers_falls_back_to_the_default(self):
        PANEL_SIZES.select_tiers("tiny")
        PANEL_SIZES.select_tiers(None)
        self.assertEqual(PANEL_SIZES.selected_tiers, ("small", "medium"))

    def test_select_tiers_accepts_a_comma_separated_list(self):
        PANEL_SIZES.select_tiers("tiny, large")
        self.assertEqual(PANEL_SIZES.selected_tiers, ("tiny", "large"))


def check_qdf_matches_its_size(case: unittest.TestCase, tier: str) -> None:
    """
    Assert the long DataFrame for a tier has the counts that tier derives.

    Parameters
    ----------
    case : unittest.TestCase
        The running test, used for its assertion methods.
    tier : str
        Name of the tier to build.

    Returns
    -------
    None
    """
    size = PANEL_SIZES[tier]
    df = size.as_qdf()
    case.assertEqual(df.shape, size.qdf_shape)
    case.assertEqual(len(df), size.observation_count)
    case.assertEqual(df["real_date"].nunique(), size.date_count)
    tickers = df["cid"].astype(str) + "_" + df["xcat"].astype(str)
    case.assertEqual(tickers.nunique(), size.ticker_count)


def check_ticker_df_matches_its_size(case: unittest.TestCase, tier: str) -> None:
    """
    Assert the wide DataFrame for a tier has the counts that tier derives.

    Parameters
    ----------
    case : unittest.TestCase
        The running test, used for its assertion methods.
    tier : str
        Name of the tier to build.

    Returns
    -------
    None
    """
    size = replace(PANEL_SIZES[tier], is_ticker_df=True)
    df = size.as_ticker_df()
    case.assertEqual(df.shape, size.ticker_df_shape)
    case.assertEqual(df.index.nunique(), size.date_count)
    case.assertEqual(df.shape[0] * df.shape[1], size.observation_count)


class TestDataFrameConstruction(unittest.TestCase):
    @parameterized.expand(CHEAP_TIERS)
    def test_qdf_matches_its_size(self, tier):
        check_qdf_matches_its_size(self, tier)

    @parameterized.expand(CHEAP_TIERS)
    def test_ticker_df_matches_its_size(self, tier):
        check_ticker_df_matches_its_size(self, tier)

    def test_built_frames_are_shared_and_copies_are_not(self):
        size = PANEL_SIZES["tiny"]
        self.assertIs(size.as_qdf(), size.as_qdf())
        copy = size.as_qdf_copy()
        self.assertIsNot(copy, size.as_qdf())
        copy.loc[0, "value"] = -999.0
        self.assertNotEqual(size.as_qdf().loc[0, "value"], -999.0)

    def test_clearing_the_cache_rebuilds(self):
        size = PANEL_SIZES["tiny"]
        first = size.as_qdf()
        clear_df_cache()
        self.assertIsNot(size.as_qdf(), first)


@pytest.mark.perf
class TestDataFrameConstructionAtScale(unittest.TestCase):
    """
    The same construction checks at the tiers whose frames are expensive to build.

    The mark sits on the class rather than on each method: pytest marks do not survive
    `parameterized.expand`, which replaces the decorated function with generated ones.
    """

    @parameterized.expand(EXPENSIVE_TIERS)
    def test_qdf_matches_its_size(self, tier):
        check_qdf_matches_its_size(self, tier)

    @parameterized.expand(EXPENSIVE_TIERS)
    def test_ticker_df_matches_its_size(self, tier):
        check_ticker_df_matches_its_size(self, tier)


# -------------------------------------------------------------------------- measurement


class TestPeakMemoryTracker(unittest.TestCase):
    def test_reports_a_peak(self):
        with PeakMemoryTracker() as tracker:
            _ = [0.0] * 1_000_000
        self.assertGreater(tracker.peak_bytes, 1_000_000)

    def test_an_exception_inside_the_block_propagates(self):
        with self.assertRaises(ZeroDivisionError):
            with PeakMemoryTracker():
                1 / 0


class TestDescribeArgument(unittest.TestCase):
    def test_a_dataframe_is_described_by_class_and_shape(self):
        df = pd.DataFrame({"cid": ["AUD", "CAD"], "value": [1.0, 2.0]})
        self.assertEqual(describe_argument(df), "DataFrame 2×2")

    def test_a_categorical_column_is_noted(self):
        df = pd.DataFrame({"cid": pd.Categorical(["AUD", "CAD"])})
        self.assertEqual(describe_argument(df), "DataFrame 2×1, categorical")

    def test_a_long_list_shows_its_length_and_first_three_items(self):
        self.assertEqual(
            describe_argument(["AUD", "CAD", "CHF", "EUR"]), "list[4]: AUD, CAD, CHF, …"
        )

    def test_anything_else_is_its_repr(self):
        self.assertEqual(describe_argument("panel"), "'panel'")


class TestDescribeCall(unittest.TestCase):
    def test_arguments_are_keyed_by_parameter_in_signature_order(self):
        def target(df, xcats=None, cids=None, blacklist=None):
            return df

        described = describe_call(target, (["A"],), {"cids": ["B"], "xcats": ["C"]})
        self.assertEqual(list(described), ["df", "xcats", "cids"])
        self.assertEqual(described["cids"], "list[1]: B")


# ---------------------------------------------------------------------- machine profile


class TestMachineProfile(unittest.TestCase):
    def setUp(self) -> None:
        self.profile = MachineProfile.capture()

    def test_identifiers_are_stable_across_calls(self):
        again = MachineProfile.capture()
        self.assertEqual(again.hardware_id, self.profile.hardware_id)
        self.assertEqual(again.software_id, self.profile.software_id)

    def test_hardware_change_moves_only_the_hardware_id(self):
        more_cpus = replace(
            self.profile, usable_cpu_count=(self.profile.usable_cpu_count or 1) + 1
        )
        self.assertNotEqual(more_cpus.hardware_id, self.profile.hardware_id)
        self.assertEqual(more_cpus.software_id, self.profile.software_id)

    def test_library_change_moves_only_the_software_id(self):
        upgraded = replace(self.profile, library_versions=(("pandas", "0.0.0-test"),))
        self.assertNotEqual(upgraded.software_id, self.profile.software_id)
        self.assertEqual(upgraded.hardware_id, self.profile.hardware_id)

    def test_code_change_moves_neither_identifier(self):
        edited = replace(
            self.profile,
            package_version="0.0.0-test",
            commit_sha="0" * 40,
            package_diff_hash="deadbeef",
        )
        self.assertEqual(edited.hardware_id, self.profile.hardware_id)
        self.assertEqual(edited.software_id, self.profile.software_id)

    def test_describe_reports_every_field(self):
        described = self.profile.describe()
        self.assertEqual(
            sorted(described),
            [
                "blas_name",
                "blas_thread_count",
                "commit_sha",
                "commit_time",
                "cpu_model",
                "hardware_id",
                "library_versions",
                "operating_system",
                "package_diff_hash",
                "package_version",
                "python_version",
                "software_id",
                "total_ram_gb",
                "usable_cpu_count",
            ],
        )
        self.assertIsInstance(described["library_versions"], dict)

    def test_capture_finds_the_host_it_runs_on(self):
        self.assertTrue(self.profile.usable_cpu_count)
        self.assertTrue(self.profile.python_version)
        self.assertTrue(self.profile.package_version)

    @unittest.skipUnless(
        (REPOSITORY_ROOT / ".git").exists(),
        "the commit is only known in a git checkout",
    )
    def test_capture_records_the_full_commit(self):
        self.assertEqual(len(self.profile.commit_sha), 40)


# ---------------------------------------------------------------------- conftest wiring


PERF_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PERF_DIR.parents[1]

PROBE = (
    "import pytest\n"
    "@pytest.mark.perf\n"
    "@pytest.mark.perf_group('probe')\n"
    "def test_probe(benchmark):\n"
    "    '''Sum a short range.'''\n"
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
    assert perf_machine_profile.hardware_id
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


# Marked `perf`: each case starts a pytest subprocess that imports the package.
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
            "from tests.perf.harness.panel_sizes import PANEL_SIZES\n"
            "@pytest.mark.perf\n"
            "@pytest.mark.perf_group('wiring')\n"
            "@pytest.mark.parametrize('panel_size', PANEL_SIZES.qdf_sizes('tiny'), ids=str)\n"
            "@pytest.mark.parametrize('dtype', ['object'], ids=str)\n"
            "def test_stamped(benchmark, panel_size, dtype):\n"
            "    '''Do nothing, measured.'''\n"
            "    df = panel_size.as_qdf()\n"
            "    benchmark(lambda: None)\n"
            "    benchmark.extra_info['rows_seen'] = len(df)\n",
        )
        out = self.tmp_path / "b.json"
        result = run_pytest(
            str(module),
            "-m",
            "perf",
            "-n0",
            "--no-cov",
            "-q",
            "--perf-tiers=tiny",
            f"--benchmark-json={out}",
        )
        self.assertTrue(out.exists(), result.stdout[-2500:])
        info = json.loads(out.read_text())["benchmarks"][0]["extra_info"]
        self.assertEqual(info["group"], "wiring")
        self.assertEqual(info["benchmark_name"], "stamped")
        self.assertEqual(info["variant"], {"dtype": "object"})
        self.assertEqual(info["timing_mode"], "calibrated")
        self.assertEqual(info["label"], "Do nothing, measured.")
        self.assertEqual(info["panel_size"]["df_format"], "long")
        self.assertEqual(info["panel_size"]["row_count"], info["rows_seen"])
        self.assertEqual(info["panel_size"]["observation_count"], info["rows_seen"])

    def test_host_is_stamped_into_machine_info(self):
        module = self.write_probe("machine")
        out = self.tmp_path / "b.json"
        run_pytest(
            str(module),
            "-m",
            "perf",
            "-n0",
            "--no-cov",
            "-q",
            f"--benchmark-json={out}",
        )
        profile = json.loads(out.read_text())["machine_info"]["macrosynergy"]
        self.assertTrue(profile["hardware_id"])
        self.assertTrue(profile["software_id"])
        self.assertTrue(profile["usable_cpu_count"])

    def test_host_is_stamped_when_the_conftest_loads_lazily(self):
        self.write_probe("lazy", PROBE.replace("test_probe", "test_probe_lazy"))
        out = self.tmp_path / "b.json"
        run_pytest(
            "tests",
            "-m",
            "perf",
            "-n0",
            "--no-cov",
            "-q",
            "--ignore=tests/unit/learning",
            "-k",
            "test_probe_lazy",
            f"--benchmark-json={out}",
        )
        machine = json.loads(out.read_text())["machine_info"]["macrosynergy"]
        self.assertTrue(machine["hardware_id"])

    def test_missing_group_fails_loudly(self):
        module = self.write_probe(
            "nogroup", PROBE.replace("@pytest.mark.perf_group('probe')\n", "")
        )
        result = run_pytest(str(module), "-m", "perf", "-n0", "--no-cov", "-q")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("perf_group", result.stdout + result.stderr)

    def test_missing_docstring_fails_loudly(self):
        module = self.write_probe(
            "nolabel", PROBE.replace("    '''Sum a short range.'''\n", "")
        )
        result = run_pytest(str(module), "-m", "perf", "-n0", "--no-cov", "-q")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("add a docstring", result.stdout + result.stderr)

    def test_collection_survives_the_plugin_being_disabled(self):
        result = run_pytest("-p", "no:benchmark", "-n0", "--no-cov", "-q", "--co")
        combined = result.stdout + result.stderr
        self.assertNotIn("INTERNALERROR", combined)
        self.assertNotIn("unknown hook", combined)

    @parameterized.expand(["option", "env"])
    def test_unknown_tier_is_rejected_by_name(self, via):
        args = ["tests/perf/benchmarks", "-n0", "--no-cov", "-q"]
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
