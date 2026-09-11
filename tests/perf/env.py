"""Environment fingerprint so perf results are never silently compared across machines."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import subprocess
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version as _pkg_version
from typing import Optional


def _cpu_brand() -> str:
    # py-cpuinfo ships with pytest-benchmark; use it when available for a real brand string.
    try:
        import cpuinfo  # type: ignore

        info = cpuinfo.get_cpu_info()
        return info.get("brand_raw") or info.get("brand") or platform.processor() or "unknown"
    except Exception:
        return platform.processor() or platform.machine() or "unknown"


def _cpu_count_physical() -> Optional[int]:
    try:
        import psutil  # type: ignore

        return psutil.cpu_count(logical=False)
    except Exception:
        return None


def _ram_total_gib() -> Optional[float]:
    try:
        import psutil  # type: ignore

        return round(psutil.virtual_memory().total / (1024 ** 3), 2)
    except Exception:
        return None


def _git_sha() -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode != 0:
            return None
        return out.stdout.strip() or None
    except Exception:
        return None


def _ci_label() -> Optional[str]:
    if os.environ.get("GITHUB_ACTIONS") == "true":
        return f"github-actions:{os.environ.get('RUNNER_NAME', 'unknown')}"
    if os.environ.get("CI"):
        return "ci:unknown"
    return None


def _lib_versions() -> dict:
    out = {}
    for pkg in ["numpy", "pandas", "statsmodels", "scipy", "pyarrow", "macrosynergy"]:
        try:
            out[pkg] = _pkg_version(pkg)
        except PackageNotFoundError:
            out[pkg] = "not-installed"
    return out


def environment_fingerprint() -> dict:
    """Capture CPU/chip, RAM, OS, Python, library versions, git SHA, and CI context."""
    return {
        "cpu_brand": _cpu_brand(),
        "cpu_arch": platform.machine(),
        "cpu_count_logical": os.cpu_count() or 0,
        "cpu_count_physical": _cpu_count_physical(),
        "ram_total_gib": _ram_total_gib(),
        "os_system": platform.system(),
        "os_release": platform.release(),
        "python_version": platform.python_version(),
        "lib_versions": _lib_versions(),
        "git_sha": _git_sha(),
        "ci": _ci_label(),
        "hostname": socket.gethostname(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


_IDENTITY_KEYS = ("cpu_brand", "cpu_count_logical", "cpu_arch", "os_system")


def fingerprint_hash(fp: dict) -> str:
    """8-char hash of the hardware/OS identity subset (excludes timestamp, lib versions, git)."""
    identity = {k: fp.get(k) for k in _IDENTITY_KEYS}
    blob = json.dumps(identity, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:8]


def comparable(fp_a: dict, fp_b: dict) -> bool:
    """True iff two fingerprints describe the same hardware/OS (benchmarks are comparable)."""
    return all(fp_a.get(k) == fp_b.get(k) for k in _IDENTITY_KEYS)
