"""Tests for the objects in machine.py."""

from __future__ import annotations

import unittest
from dataclasses import replace

from tests.perf.machine import MachineProfile, PeakMemoryTracker


class TestMachineProfile(unittest.TestCase):
    def setUp(self) -> None:
        self.profile = MachineProfile.capture()

    def test_fingerprint_is_stable_across_calls(self):
        self.assertEqual(
            MachineProfile.capture().fingerprint,
            MachineProfile.capture().fingerprint,
        )

    def test_fingerprint_changes_with_usable_cpu_count(self):
        more_cpus = replace(
            self.profile, usable_cpu_count=(self.profile.usable_cpu_count or 1) + 1
        )
        self.assertNotEqual(more_cpus.fingerprint, self.profile.fingerprint)

    def test_describe_reports_every_field(self):
        described = self.profile.describe()
        self.assertEqual(
            sorted(described),
            [
                "blas_name",
                "blas_thread_count",
                "cpu_model",
                "git_commit",
                "library_versions",
                "operating_system",
                "python_version",
                "total_ram_gb",
                "usable_cpu_count",
            ],
        )
        self.assertIsInstance(described["library_versions"], dict)

    def test_capture_finds_the_host_it_runs_on(self):
        self.assertTrue(self.profile.usable_cpu_count)
        self.assertTrue(self.profile.python_version)
        self.assertIn("macrosynergy", dict(self.profile.library_versions))


class TestPeakMemoryTracker(unittest.TestCase):
    def test_reports_a_peak(self):
        with PeakMemoryTracker() as tracker:
            _ = [0.0] * 1_000_000
        self.assertGreater(tracker.peak_bytes, 1_000_000)

    def test_an_exception_inside_the_block_propagates(self):
        with self.assertRaises(ZeroDivisionError):
            with PeakMemoryTracker():
                1 / 0


if __name__ == "__main__":
    unittest.main()
