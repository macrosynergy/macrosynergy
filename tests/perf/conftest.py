"""
Perf-suite wiring: tier selection, machine capture, and benchmark metadata recording.
`benchmark.extra_info` is the only path from a test to the results file, so nothing
downstream has to recover a measurement's identity by parsing a test name.
"""

from __future__ import annotations

import os
from typing import Optional

import pytest

from tests.perf import panel_sizes
from tests.perf.machine import MachineProfile
from tests.perf.panel_sizes import PANEL_SIZES

TIERS_ENVIRONMENT_VARIABLE = "MACROSYN_PERF_TIERS"

_machine_profile: Optional[MachineProfile] = None


def machine_profile() -> MachineProfile:
    """
    The captured machine, built on first use rather than at startup.

    Capture costs a couple of seconds, and the repository's `addopts` sets `-n auto`, so
    capturing at configure time would charge every worker for a fact most sessions never
    record.

    Returns
    -------
    MachineProfile
        The profile of this host, the same object on every call.
    """
    global _machine_profile
    if _machine_profile is None:
        _machine_profile = MachineProfile.capture()
    return _machine_profile


class _MachineProfileStamper:
    """
    Writes the machine profile into pytest-benchmark's machine_info block.

    Registered as a plugin object rather than declared as a module-level hook, because a
    module-level `pytest_benchmark_*` function in a conftest makes pluggy abort
    collection of the whole repository whenever that plugin is absent.
    """

    def pytest_benchmark_update_machine_info(self, config, machine_info):
        """
        Add the profile and its fingerprint under one key of `machine_info`.

        Parameters
        ----------
        config : pytest.Config
            The active pytest configuration, unused.
        machine_info : dict
            The block pytest-benchmark writes into the results file.

        Returns
        -------
        None
        """
        profile = machine_profile()
        machine_info["macrosynergy"] = {
            **profile.describe(),
            "fingerprint": profile.fingerprint,
        }


def pytest_configure(config):
    """
    Resolve the measured tiers and register the machine stamper.

    Tiers come from `--perf-tiers`, then `MACROSYN_PERF_TIERS`, then the catalogue's own
    default. This runs before collection, which is when the catalogue's size methods are
    evaluated inside `@parametrize`.

    Parameters
    ----------
    config : pytest.Config
        The active pytest configuration.

    Returns
    -------
    None

    Raises
    ------
    pytest.UsageError
        If the requested tiers name a tier the catalogue does not hold.
    """
    requested = config.getoption("--perf-tiers", default=None) or os.environ.get(
        TIERS_ENVIRONMENT_VARIABLE
    )
    try:
        PANEL_SIZES.select_tiers(requested)
    except ValueError as error:
        raise pytest.UsageError(str(error)) from error
    if config.pluginmanager.hasplugin("benchmark"):
        config.pluginmanager.register(
            _MachineProfileStamper(), "macrosynergy_machine_profile"
        )


def pytest_sessionfinish(session, exitstatus):
    """
    Release every DataFrame the run built.

    Parameters
    ----------
    session : pytest.Session
        The finished session, unused.
    exitstatus : int
        The session's exit status, unused.

    Returns
    -------
    None
    """
    panel_sizes.clear_df_cache()


@pytest.fixture(scope="session")
def perf_machine_profile() -> MachineProfile:
    """
    The machine this session runs on.

    Returns
    -------
    MachineProfile
        The captured profile, shared across the session.
    """
    return machine_profile()


@pytest.fixture(autouse=True)
def _record_benchmark_metadata(request):
    """
    Record group, name, variant, timing mode and panel size on `benchmark.extra_info`.

    The size block is written after the test body, so it describes a DataFrame that was
    actually built.

    Parameters
    ----------
    request : pytest.FixtureRequest
        The requesting test, read for its parameters and markers.

    Yields
    ------
    None
        Control returns to the test between the two recording steps.
    """
    if "benchmark" not in request.fixturenames:
        yield
        return

    benchmark = request.getfixturevalue("benchmark")
    callspec = getattr(request.node, "callspec", None)
    parameters = dict(callspec.params) if callspec is not None else {}
    panel_size = parameters.pop("panel_size", None)
    marker = request.node.get_closest_marker("perf_group")
    group = str(marker.args[0]) if marker is not None and marker.args else None

    if group is None and request.node.get_closest_marker("perf") is not None:
        pytest.fail(
            f"{request.node.nodeid}: add @pytest.mark.perf_group(...) beside @pytest.mark.perf."
        )

    if group is not None:
        benchmark.extra_info["group"] = group
    benchmark.extra_info["benchmark_name"] = request.node.function.__name__.removeprefix("test_")
    benchmark.extra_info["variant"] = {
        name: str(value) for name, value in sorted(parameters.items())
    }
    benchmark.extra_info.setdefault("timing_mode", "calibrated")

    yield

    if panel_size is not None:
        benchmark.extra_info["panel_size"] = panel_size.describe()
