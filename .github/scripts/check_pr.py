import os
import sys
from time import sleep
from typing import Any, Dict, List, Optional

import requests
from packaging import version

REPO_OWNER: str = "macrosynergy"
REPO_NAME: str = "macrosynergy"
REPO_URL: str = f"github.com/{REPO_OWNER}/{REPO_NAME}"

OAUTH_TOKEN: Optional[str] = os.getenv("GH_TOKEN", None)


ACCEPTED_TITLE_PATTERNS: List[str] = [
    "Feature:",
    "Bugfix:",
    "Refactor:",
    "Docs:",
    "Documentation:",
    "Chore:",
    "CI/CD:",
    "Misc:",
    "Misc.:",
    "Miscellaneous:",
    "Suggestion:",
    "Suggestions:",
]


def api_request(
    url: str,
    headers: Dict[str, str] = {"Accept": "application/vnd.github.v3+json"},
    params: Dict[str, Any] = {},
) -> Any:
    """
    Make a request to the GitHub API.

    :param <str> url: The URL to make the request to.
    :param <Dict[str, str]> headers: The headers to send with the request.
    :param <Dict[str, Any]> params: The parameters to send with the request.
    :return <Any>: The response from the API.
    """
    retries = 0
    max_retries = 5
    while retries < max_retries:
        try:
            # Add the OAuth token to the headers if it exists
            if OAUTH_TOKEN:
                headers["Authorization"] = f"token {OAUTH_TOKEN}"

            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()  # Raise exception for failed requests
            return response.json()
        except Exception as exc:
            print(f"Request failed: {exc}")
            retries += 1
            sleep(1)
    print(f"Request failed after {max_retries} retries. Exiting.")
    sys.exit(1)


def check_title(
    title: str,
) -> bool:
    title_check: bool = False
    for pattern in ACCEPTED_TITLE_PATTERNS:
        tx: str = title.strip()
        if tx.startswith(pattern):
            title_check = True
            break

    if not title_check:
        eMsg: str = (
            f"PR title '{title}' does not match any "
            f"of the accepted patterns: {ACCEPTED_TITLE_PATTERNS}"
        )
        raise ValueError(eMsg)

    return title_check


def _check_merge_w_version(
    body: str,
) -> bool:
    assert bool(body)
    MIV_STR: str = "MERGE-IN-VERSION-v"
    MAV_STR: str = "MERGE-AFTER-VERSION-v"

    try:
        from setup import VERSION
    except ImportError:
        eMsg: str = "Could not import VERSION from setup.py."
        # print cwd and os.listdir()
        print(f"cwd: {os.getcwd()}")
        print(f"ls: {os.listdir()}")
        raise ImportError(eMsg)

    results: List[bool] = [True, True]

    for pattern in [MIV_STR, MAV_STR]:
        if pattern in body:
            fidx: int = body.find(pattern)
            sidx: int = body.find(" ", fidx)
            if sidx == -1:
                sidx = len(body)
            # get all the chars between fidx and sidx
            mversion: str = body[fidx + len(pattern) : sidx].strip()
            try:
                version.parse(mversion)
            except ValueError:
                eMsg: str = f"'{mversion}' is not a valid version number."
                raise ValueError(eMsg)

            if pattern == MIV_STR:
                results[0] = version.parse(mversion) == version.parse(VERSION)
                if not results[0]:
                    return False
            else:
                results[1] = version.parse(mversion) <= version.parse(VERSION)
                if not results[1]:
                    return False

    return all(results)


def _check_do_not_merge(
    body: str,
) -> bool:
    assert bool(body)
    result: bool = "DO-NOT-MERGE" in body
    return result


def _check_merge_after(
    body: str,
) -> bool:
    assert bool(body)
    MA_STR: str = "MERGE-AFTER-#"

    if MA_STR not in body:
        return True

    fidx: int = body.find(MA_STR)
    sidx: int = body.find(" ", fidx)
    if sidx == -1:
        sidx = len(body)
    # get all the chars between fidx and sidx
    merge_after_pr: str = body[fidx + len(MA_STR) : sidx].strip()
    try:
        merge_after_pr = int(merge_after_pr)
    except ValueError:
        eMsg: str = f"PR number '{merge_after_pr}' is not an integer."
        raise ValueError(eMsg)

    # get the PR details
    pr_info: Dict[str, Any] = get_pr_details(pr_number=merge_after_pr)
    # if closed return true
    mergable: bool = pr_info["state"] == "closed"
    return mergable


def check_pr_directives(
    body: str,
    state: str,
) -> bool:
    if body is None:
        return True

    body = body.strip().replace(" ", "-").upper()

    do_not_merge: bool = _check_do_not_merge(body=body)

    merge_after: bool = _check_merge_after(body=body)

    merge_w_version: bool = _check_merge_w_version(body=body)

    results: List[bool] = [do_not_merge, merge_after, merge_w_version]

    return all(results)


def test_pr_info(
    pr_info: Dict[str, Any],
) -> bool:
    assert set(["number", "title", "state", "body"]).issubset(pr_info.keys())

    if not check_title(title=pr_info["title"]):
        raise ValueError("PR title does not match any of the accepted patterns.")

    if not check_pr_directives(body=pr_info["body"], state=pr_info["state"]):
        raise ValueError(
            "PR number is not mergable due to merge restrictions"
            " specified in the PR body."
        )

    return True


def get_pr_details(
    pr_number: int,
):
    URL: str = (
        f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/pulls/{pr_number}"
    )

    pr_details: Dict[str, Any] = api_request(url=URL)

    pr_info: Dict[str, Any] = {
        "number": pr_details["number"],
        "title": pr_details["title"],
        "state": pr_details["state"],
        "body": pr_details["body"],
    }

    return pr_info


def main(pr_number: int):
    pr_info: Dict[str, Any] = get_pr_details(pr_number=pr_number)
    if not test_pr_info(pr_info=pr_info):
        raise ValueError("PR is not mergable.")

    print("PR is mergable.")

    return True


if __name__ == "__main__":
    pr_number: int = int(sys.argv[1])
    main(pr_number=pr_number)
