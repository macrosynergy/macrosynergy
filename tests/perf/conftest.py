"""Perf-suite fixtures and the pytest-benchmark machine-info hook."""

import pytest

from tests.perf.env import environment_fingerprint


@pytest.fixture(scope="session")
def perf_env():
    return environment_fingerprint()


def pytest_benchmark_update_machine_info(config, machine_info):
    # Stamp our richer fingerprint into pytest-benchmark's JSON for comparability.
    machine_info["macrosynergy_env"] = environment_fingerprint()
