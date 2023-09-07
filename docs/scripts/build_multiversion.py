import os
import sys
import shutil
import subprocess
import warnings
import glob
import git
import tempfile
from typing import List, Dict, Optional

# the intention of this script is to build the documentation for all
# branches of a project, and then copy the results into a single
# directory.  This allows the documentation for multiple versions of
# a project to be hosted on the same site.

TARGET_BRANCHES: List[str] = ["origin/main", "origin/test", "origin/develop"]
REPO_URL: str = "https://github.com/macrosynergy/macrosynergy/"


def clone_repo(repo_url: str, target_dir: str) -> None:
    """Clone a git repo into a target directory."""
    try:
        git.Repo.clone_from(repo_url, target_dir)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to clone repo {repo_url} into {target_dir}"
        ) from exc
    

def checkout_branch(repo_dir: str, branch: str) -> None:
    """Checkout a branch of a git repo."""
    try:
        repo: git.Repo = git.Repo(repo_dir)
        repo.git.checkout(branch)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to checkout branch {branch} of repo {repo_dir}"
        ) from exc
    
    

