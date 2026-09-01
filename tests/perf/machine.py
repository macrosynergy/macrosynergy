"""
What a benchmark run happened on. `MachineProfile` records the host in the fields that
move a timing and hashes them into a fingerprint, so two runs can be told apart without
per-field negotiation, and `PeakMemoryTracker` reports peak allocation over a block.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import tracemalloc
from dataclasses import dataclass
from functools import partial
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np


def _value_or_none(read: Callable[[], Any]) -> Any:
    """
    The result of a collector, or None when it fails.

    Every field of a `MachineProfile` degrades to None rather than failing a run, so a
    host without `/proc` or without git costs one field instead of the measurement.

    Parameters
    ----------
    read : Callable[[], Any]
        Zero-argument callable that reads one fact about the host.

    Returns
    -------
    Any
        Whatever `read` returned, or None if it raised.
    """
    try:
        return read()
    except Exception:
        return None


def _cpu_model_name() -> Optional[str]:
    """
    The CPU model, from `/proc/cpuinfo` where available.

    Returns
    -------
    Optional[str]
        The model name, falling back to `platform.processor`, then to None.
    """

    def from_proc_cpuinfo():
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
        return None

    return _value_or_none(from_proc_cpuinfo) or platform.processor() or None


def _usable_cpu_count() -> Optional[int]:
    """
    The number of CPUs this process may run on.

    This is the affinity count rather than the count the host has, because BLAS sizes
    its thread pool from what the process may use and that is what enters a timing.

    Returns
    -------
    Optional[int]
        Usable CPU count, falling back to `os.cpu_count`, then to None.
    """
    return _value_or_none(lambda: len(os.sched_getaffinity(0))) or os.cpu_count()


def _total_ram_gb() -> Optional[float]:
    """
    Total physical memory, in gibibytes.

    Returns
    -------
    Optional[float]
        Total memory rounded to one decimal place, or None if psutil cannot report it.
    """

    def total_bytes():
        import psutil

        return psutil.virtual_memory().total

    total = _value_or_none(total_bytes)
    return round(total / 2**30, 1) if total else None


def _library_versions() -> Dict[str, str]:
    """
    Installed versions of the libraries a timing depends on.

    Returns
    -------
    Dict[str, str]
        Version string per package, omitting any package that is not installed.
    """
    versions = {}
    for package in ("numpy", "pandas", "macrosynergy"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            pass
    return versions


def _blas_name() -> Optional[str]:
    """
    The BLAS implementation numpy was built against.

    Returns
    -------
    Optional[str]
        The build's BLAS name, or None if numpy does not report one.
    """
    return _value_or_none(
        lambda: np.show_config(mode="dicts")["Build Dependencies"]["blas"]["name"]
    )


def _blas_thread_count() -> Optional[int]:
    """
    The number of threads BLAS will use.

    Returns
    -------
    Optional[int]
        The first thread count set in the environment, otherwise the usable CPU count.
    """
    for variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
        configured = os.environ.get(variable)
        if configured:
            return _value_or_none(partial(int, configured))
    return _usable_cpu_count()


def _git_commit() -> Optional[str]:
    """
    The short commit hash of the checked-out tree.

    Returns
    -------
    Optional[str]
        The abbreviated hash, or None outside a git checkout.
    """

    def short_sha():
        completed = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return completed.stdout.strip() or None

    return _value_or_none(short_sha)


@dataclass(frozen=True)
class MachineProfile:
    """
    The machine a run happened on, in the fields that move a timing.

    Parameters
    ----------
    cpu_model : Optional[str]
        The CPU model name.
    usable_cpu_count : Optional[int]
        Number of CPUs the process may run on.
    total_ram_gb : Optional[float]
        Total physical memory, in gibibytes.
    operating_system : Optional[str]
        System name and release.
    python_version : str
        Interpreter version.
    library_versions : Tuple[Tuple[str, str], ...]
        Sorted package-version pairs, as a tuple so the profile stays hashable.
    blas_name : Optional[str]
        The BLAS implementation numpy was built against.
    blas_thread_count : Optional[int]
        Number of threads BLAS will use.
    git_commit : Optional[str]
        Short commit hash of the checked-out tree.
    """

    cpu_model: Optional[str]
    usable_cpu_count: Optional[int]
    total_ram_gb: Optional[float]
    operating_system: Optional[str]
    python_version: str
    library_versions: Tuple[Tuple[str, str], ...]
    blas_name: Optional[str]
    blas_thread_count: Optional[int]
    git_commit: Optional[str]

    @classmethod
    def capture(cls) -> "MachineProfile":
        """
        Read every field from the current host.

        Returns
        -------
        MachineProfile
            The profile of the machine this call runs on.
        """
        return cls(
            cpu_model=_cpu_model_name(),
            usable_cpu_count=_usable_cpu_count(),
            total_ram_gb=_total_ram_gb(),
            operating_system=f"{platform.system()} {platform.release()}",
            python_version=platform.python_version(),
            library_versions=tuple(sorted(_library_versions().items())),
            blas_name=_blas_name(),
            blas_thread_count=_blas_thread_count(),
            git_commit=_git_commit(),
        )

    @property
    def fingerprint(self) -> str:
        """
        Short hash of every field; two runs compare only when these match.

        Returns
        -------
        str
            The first eight hex digits of the SHA-256 of `describe`.
        """
        blob = json.dumps(self.describe(), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:8]

    def describe(self) -> Dict[str, Any]:
        """
        The machine block recorded with a run.

        Returns
        -------
        Dict[str, Any]
            Every field, in the plain types pytest-benchmark's `machine_info` channel
            can serialise.
        """
        return {
            "cpu_model": self.cpu_model,
            "usable_cpu_count": self.usable_cpu_count,
            "total_ram_gb": self.total_ram_gb,
            "operating_system": self.operating_system,
            "python_version": self.python_version,
            "library_versions": dict(self.library_versions),
            "blas_name": self.blas_name,
            "blas_thread_count": self.blas_thread_count,
            "git_commit": self.git_commit,
        }


class PeakMemoryTracker:
    """
    Context manager reporting peak allocation in bytes over the block it wraps.

    Always used outside a measured region, because tracemalloc's overhead would distort
    whatever it wrapped.
    """

    def __init__(self) -> None:
        self.peak_bytes: Optional[int] = None

    def __enter__(self) -> "PeakMemoryTracker":
        tracemalloc.start()
        return self

    def __exit__(self, *exception: Any) -> bool:
        # Returning False lets an exception raised inside the block propagate.
        self.peak_bytes = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        return False
