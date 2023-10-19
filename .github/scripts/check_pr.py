import os
import sys
from time import sleep
from typing import Any, Dict, List, Optional, Callable, Tuple

import requests
from packaging import version

REPO_OWNER: str = "macrosynergy"
REPO_NAME: str = "macrosynergy"
REPO_URL: str = f"github.com/{REPO_OWNER}/{REPO_NAME}"

sys.path.append(os.getcwd())

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
    """
    Checks if the PR title matches any of the accepted patterns.

    :param <str> title: The PR title to check.
    :return <bool>: True if the title matches any of the accepted patterns.
    :raises <ValueError>: If the title does not match any of the accepted patterns.
    """
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


def _get_pattern_idx(
    body: str,
    pattern: str,
    numeric: bool = False,
) -> Tuple[int, int]:
    """
    Get the start and end index of a pattern in a string.
    :param <str> body: The string to search.
    :param <str> pattern: The pattern to search for.
    :return <Tuple[int, int]>: The start and end index of the pattern.
    :raises <AssertionError>: If the body or pattern is empty.
    """
    NUM_CHARS: List[str] = list(map(str, range(10))) + ["."]
    assert bool(body)
    assert bool(pattern)
    assert pattern in body
    fidx: int = body.find(pattern)
    sidx: int = body.find("-", fidx)  # all spaces have been replaced with dashes
    if sidx == -1 and not numeric:
        return (fidx, len(body))

    if numeric:
        # set fidx to the end of the pattern
        fidx = fidx + len(pattern)
        # get the first non-numeric char after the pattern
        for idx, char in enumerate(body[fidx:]):
            if char not in NUM_CHARS:
                break
        # set sidx to the first non-numeric char after the pattern
        sidx = fidx + idx + 1

    return (fidx, sidx)


def _tvp(vx: str) -> version.Version:
    """
    Try-Version-Parse
    :param <str> vx: The version string to parse.
    :return <version.Version>: The parsed version.
    :raises <ValueError>: If the version string is not valid.
    """
    try:
        return version.parse(vx)
    except ValueError:
        eMsg: str = f"'{vx}' is not a valid version number."
        raise ValueError(eMsg)


def _check_merge_w_version(
    body: str,
) -> bool:
    """
    Checks if the PR body contains a merge directive and if so, whether
    the merge directive is valid.
    :param <str> body: The PR body to check.
    :return <bool>: True if the PR body does not contain a merge directive
        or if the merge directive is valid.
    :raises <ValueError>: If the PR body contains a merge directive that is
        not valid.
    """
    assert bool(body)
    MIV_STR: str = "MERGE-IN-VERSION-V"  # "IN" this version
    MAV_STR: str = "MERGE-AFTER-VERSION-V"  # "AFTER" this version

    if not any([MIV_STR in body, MAV_STR in body]):
        return True

    try:
        from setup import VERSION

        _tvp(VERSION)
    except ImportError:
        eMsg: str = "Could not import VERSION from setup.py."
        raise ImportError(eMsg)

    results: List[bool] = [True, True]

    p_methods: Dict[str, Callable] = {
        MIV_STR: lambda x: _tvp(x) == _tvp(VERSION),
        MAV_STR: lambda x: _tvp(x) <= _tvp(VERSION),
    }

    for px, pattern in enumerate(p_methods):
        if pattern not in body:
            continue
        fidx, sidx = _get_pattern_idx(body=body, pattern=pattern, numeric=True)
        mversion: str = body[fidx:sidx].strip()
        results[px] = p_methods[pattern](mversion)

        if not results[px]:
            break

    return all(results)


def _check_do_not_merge(
    body: str,
) -> bool:
    assert bool(body)
    result: bool = not ("DO-NOT-MERGE" in body)
    return result


def _check_merge_after(
    body: str,
) -> bool:
    assert bool(body)
    MA_STR: str = "MERGE-AFTER-#"

    if MA_STR not in body:
        return True

    fidx, sidx = _get_pattern_idx(body=body, pattern=MA_STR, numeric=True)
    # get all the chars between fidx and sidx
    merge_after_pr: str = body[fidx:sidx].strip()
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

    results: List[bool] = [
        _check_do_not_merge(body=body),
        _check_merge_after(body=body),
        _check_merge_w_version(body=body),
    ]

    return all(results)


def test_pr_info(
    pr_info: Dict[str, Any],
) -> bool:
    assert set(["number", "title", "state", "body"]).issubset(pr_info.keys())

    if not check_title(title=pr_info["title"]):
        raise ValueError("PR title does not match any of the accepted patterns.")

    if not check_pr_directives(body=pr_info["body"], state=pr_info["state"]):
        raise ValueError(
            f"PR #{pr_info['number']} is not mergable due to merge restrictions"
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
