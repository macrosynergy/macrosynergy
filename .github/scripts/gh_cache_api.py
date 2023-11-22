import argparse
import logging
import os
from typing import Any, Dict, List, Optional

from gh_api_request import api_request

REPO_OWNER: str = "macrosynergy"
REPO_NAME: str = "macrosynergy"
REPO_URL: str = f"github.com/{REPO_OWNER}/{REPO_NAME}"

API_URL = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/actions/caches"


def get_caches_list(ref: str) -> List[str]:
    params: Dict[str, Any] = {"per_page": 100, "ref": ref}
    resp_dict: Dict[str, Any] = api_request(url=API_URL, params=params, method="GET")

    caches_data: List[Dict[str, Any]] = resp_dict["actions_caches"]
    assert len(caches_data) == int(resp_dict["total_count"])

    # sort by last_accessed_at
    caches_data.sort(key=lambda x: x["last_accessed_at"], reverse=True)

    return [cache["cache_key"] for cache in caches_data]


def clear_cache(ref: str) -> None:
    params = {"ref": ref}
    result: dict = api_request(url=API_URL, params=params, method="DELETE")
    return bool(result)


def main(ref: str) -> None:
    caches_list: List[str] = get_caches_list()
    if cache_key not in caches_list:
        logging.warning(f"Cache with key {cache_key} not found")
        raise RuntimeError(f"Cache with key {cache_key} not found")

    if not clear_cache(cache_key=cache_key, ref=ref):
        logging.warning(f"Cache with key {cache_key} not cleared")
        raise RuntimeError(f"Cache with key {cache_key} not cleared")
    logging.info(f"Cache with key {cache_key} cleared")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clears a branch cache")
    parser.add_argument(
        "--branch",
        "-b",
        type=str,
        required=False,
        help="The branch to clear the cache for",
        default=None,
    )

    parser.add_argument(
        "--ref",
        "-r",
        type=str,
        required=False,
        help="The ref to clear the cache for",
        default=None,
    )

    args = parser.parse_args()

    cache_key: str = args.cache_key

    branch: Optional[str] = args.branch
    ref: Optional[str] = args.ref

    # both shouldn't be set or unset
    if bool(branch) == bool(ref):
        raise ValueError("Only one of `-b` or `-r` should be set")

    ref: str = ref or branch

    main(ref=ref)
