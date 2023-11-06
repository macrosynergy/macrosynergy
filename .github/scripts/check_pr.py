import os
import sys
from typing import Any, Dict, List, Optional, Callable, Tuple

from packaging import version

from gh_api_request import api_request

REPO_OWNER: str = "macrosynergy"
ORGANIZATION: str = "macrosynergy"
REPO_NAME: str = "macrosynergy"
REPO_URL: str = f"github.com/{REPO_OWNER}/{REPO_NAME}"

sys.path.append(os.getcwd())

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
        sidx = fidx + idx

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


def find_previous_at(body: str, idx: int) -> int:
    i = idx

    while i >= 0:
        if body[i] == "@":
            return i + 1
        i -= 1
    raise ValueError("NO @ FOUND")


def _check_required_reviewers(
    body: str,
    pr_info: Dict[str, Any],
) -> bool:
    assert bool(body)
    RR_STR: str = "-MUST-REVIEW"

    if RR_STR not in body:
        return True

    last_idx = body.find(RR_STR)
    first_idx = find_previous_at(body, last_idx)

    required_reviewer: str = body[first_idx:last_idx].strip()

    # check from the PR details if the user has approved the PR
    # check the reviews
    pr_reviews: Dict[str, Any] = get_pr_reviews(pr_number=pr_info["number"])
    # check if the user has approved the PR
    approved: bool = any(
        [str(rx).upper() == required_reviewer.upper() for rx in pr_reviews["APPROVED"]]
    )

    return approved


def check_pr_directives(
    pr_info: Dict[str, Any],
) -> bool:
    body: str = pr_info["body"]

    if body is None:
        return True

    body = body.strip().replace(" ", "-").upper()
    body = f" {body} "

    results: List[bool] = [
        _check_do_not_merge(body=body),
        _check_merge_after(body=body),
        _check_merge_w_version(body=body),
        _check_required_reviewers(body=body, pr_info=pr_info),
    ]

    return all(results)


def test_pr_info(
    pr_info: Dict[str, Any],
) -> bool:
    assert set(["number", "title", "state", "body"]).issubset(pr_info.keys())
    err_msg: str = ""
    if not check_title(title=pr_info["title"]):
        err_msg += f"PR #{pr_info['number']} is not mergable due to invalid title.\n"

    if not check_pr_directives(pr_info=pr_info):
        err_msg += (
            f"PR #{pr_info['number']} is not mergable due to merge restrictions"
            " specified in the PR body."
        )

    if err_msg:
        raise ValueError(err_msg.strip())

    return True


def get_pr_reviews(
    pr_number: int,
):
    URL: str = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/pulls/{pr_number}/reviews"

    pr_reviews: Dict[str, Any] = api_request(url=URL)
    results: Dict[str, List[str]] = {
        "APPROVED": [],
        "CHANGES_REQUESTED": [],
        "COMMENTED": [],
        "DISMISSED": [],
    }

    for item in pr_reviews:
        review_name: str = item["user"]["login"]
        review_state: str = item["state"]
        if review_state not in results:
            results[review_state] = []
        results[review_state].append(review_name)

    return results


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
