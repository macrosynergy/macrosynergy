"""
Hardware and software profile of the machine a benchmark run executes on.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import platform
import subprocess
from dataclasses import dataclass
from functools import partial
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_DIRECTORY = "macrosynergy"
HARDWARE_FIELDS = ("cpu_model", "usable_cpu_count", "total_ram_gb")
SOFTWARE_FIELDS = (
    "operating_system",
    "python_version",
    "library_versions",
    "blas_name",
    "blas_thread_count",
)


def _short_hash(value: Any) -> str:
    """
    Short, stable hash of a JSON-serialisable value.

    Parameters
    ----------
    value : Any
        The value to hash; dictionary key order does not matter.

    Returns
    -------
    str
        The first eight hex digits of the SHA-256 of the sorted JSON.
    """
    blob = json.dumps(value, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:8]


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


def _total_ram_gb() -> Optional[int]:
    """
    Total physical memory, in gibibytes.

    Returns
    -------
    Optional[int]
        Total memory rounded to whole gibibytes, so machines of the same size agree, or
        None if psutil cannot report it.
    """

    def total_bytes():
        import psutil

        return psutil.virtual_memory().total

    total = _value_or_none(total_bytes)
    return round(total / 2**30) if total else None


def _library_versions() -> Dict[str, str]:
    """
    Installed versions of the libraries a timing depends on.

    Returns
    -------
    Dict[str, str]
        Version string per package, omitting any package that is not installed. The
        package under measurement is recorded separately, as `package_version`.
    """
    versions = {}
    for package in ("numpy", "pandas", "scipy"):
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


def _git(*arguments: str) -> bytes:
    """
    Run one git command at the repository root.

    Parameters
    ----------
    *arguments : str
        Arguments after `git`.

    Returns
    -------
    bytes
        The command's standard output.
    """
    return subprocess.run(
        ["git", *arguments],
        capture_output=True,
        timeout=5,
        check=True,
        cwd=REPOSITORY_ROOT,
    ).stdout


def _commit_sha_and_time() -> Tuple[Optional[str], Optional[str]]:
    """
    The checked-out commit and when it was committed.

    Returns
    -------
    Tuple[Optional[str], Optional[str]]
        The full hash and the ISO-8601 commit time, or two Nones outside a git checkout.
    """
    shown = _value_or_none(lambda: _git("show", "-s", "--format=%H %cI", "HEAD"))
    return tuple(shown.decode().split()) if shown else (None, None)


def _package_diff_hash() -> Optional[str]:
    """
    Hash of the uncommitted changes to the package, untracked files included.

    Only the package is hashed, because it is the code being measured; an edit to the
    suite itself leaves this None.

    Returns
    -------
    Optional[str]
        A short hash of the changes, or None when the package matches the commit or
        git is unavailable.
    """

    def changes():
        tracked = _git("diff", "HEAD", "--binary", "--", PACKAGE_DIRECTORY)
        untracked = (
            _git("ls-files", "--others", "--exclude-standard", "--", PACKAGE_DIRECTORY)
            .decode()
            .split()
        )
        if not tracked and not untracked:
            return None
        digest = hashlib.sha256(tracked)
        for path in sorted(untracked):
            digest.update(path.encode() + (REPOSITORY_ROOT / path).read_bytes())
        return digest.hexdigest()[:8]

    return _value_or_none(changes)


@dataclass(frozen=True)
class MachineProfile:
    """
    The machine and code a run happened on, in the fields that move a timing.

    Parameters
    ----------
    cpu_model : Optional[str]
        The CPU model name.
    usable_cpu_count : Optional[int]
        Number of CPUs the process may run on.
    total_ram_gb : Optional[int]
        Total physical memory, in gibibytes.
    operating_system : Optional[str]
        System name and release.
    python_version : str
        Interpreter version.
    library_versions : Tuple[Tuple[str, str], ...]
        Sorted package-version pairs of the runtime libraries, as a tuple so the profile
        stays hashable.
    blas_name : Optional[str]
        The BLAS implementation numpy was built against.
    blas_thread_count : Optional[int]
        Number of threads BLAS will use.
    package_version : Optional[str]
        `macrosynergy.__version__`, read from the source tree rather than from installed
        metadata, which CI removes.
    commit_sha : Optional[str]
        Full hash of the checked-out commit.
    commit_time : Optional[str]
        ISO-8601 time of that commit, which orders runs along history.
    package_diff_hash : Optional[str]
        Hash of uncommitted package changes, None when the package matches the commit.
    """

    cpu_model: Optional[str]
    usable_cpu_count: Optional[int]
    total_ram_gb: Optional[int]
    operating_system: Optional[str]
    python_version: str
    library_versions: Tuple[Tuple[str, str], ...]
    blas_name: Optional[str]
    blas_thread_count: Optional[int]
    package_version: Optional[str]
    commit_sha: Optional[str]
    commit_time: Optional[str]
    package_diff_hash: Optional[str]

    @classmethod
    def capture(cls) -> "MachineProfile":
        """
        Read every field from the current host.

        Returns
        -------
        MachineProfile
            The profile of the machine this call runs on.
        """
        commit_sha, commit_time = _commit_sha_and_time()
        return cls(
            cpu_model=_cpu_model_name(),
            usable_cpu_count=_usable_cpu_count(),
            total_ram_gb=_total_ram_gb(),
            operating_system=f"{platform.system()} {platform.release()}",
            python_version=platform.python_version(),
            library_versions=tuple(sorted(_library_versions().items())),
            blas_name=_blas_name(),
            blas_thread_count=_blas_thread_count(),
            package_version=_value_or_none(
                lambda: importlib.import_module("macrosynergy").__version__
            ),
            commit_sha=commit_sha,
            commit_time=commit_time,
            package_diff_hash=_package_diff_hash(),
        )

    @property
    def hardware_id(self) -> str:
        """
        Hash of the hardware; two runs are comparable only when these match.

        Returns
        -------
        str
            Short hash of the CPU model, usable CPU count and total memory.
        """
        return self.describe()["hardware_id"]

    @property
    def software_id(self) -> str:
        """
        Hash of the runtime the package ran on, excluding the package itself.

        Returns
        -------
        str
            Short hash of the operating system, Python, BLAS and library versions.
        """
        return self.describe()["software_id"]

    def describe(self) -> Dict[str, Any]:
        """
        The machine block recorded with a run.

        Returns
        -------
        Dict[str, Any]
            Every field plus `hardware_id` and `software_id`, in the plain types
            pytest-benchmark's `machine_info` channel can serialise.
        """
        described = {
            "cpu_model": self.cpu_model,
            "usable_cpu_count": self.usable_cpu_count,
            "total_ram_gb": self.total_ram_gb,
            "operating_system": self.operating_system,
            "python_version": self.python_version,
            "library_versions": dict(self.library_versions),
            "blas_name": self.blas_name,
            "blas_thread_count": self.blas_thread_count,
            "package_version": self.package_version,
            "commit_sha": self.commit_sha,
            "commit_time": self.commit_time,
            "package_diff_hash": self.package_diff_hash,
        }
        described["hardware_id"] = _short_hash(
            {field: described[field] for field in HARDWARE_FIELDS}
        )
        described["software_id"] = _short_hash(
            {field: described[field] for field in SOFTWARE_FIELDS}
        )
        return described
