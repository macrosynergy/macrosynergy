import os
import argparse
import requests
from packaging import version
from typing import Dict, List

REPO_OWNER: str = "macrosynergy"
REPO_NAME: str = "macrosynergy"
REPO_URL: str = f"github.com/{REPO_OWNER}/{REPO_NAME}"
RELEASES_API_URL = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/releases"
OUTPUT_DIR = "./docs/source"
PATH_TO_RELEASE_MD: str = os.path.join(OUTPUT_DIR, "release_notes.md")
LINE_SEPARATOR: str = "\n\n____________________\n\n"


def format_gh_username(strx: str) -> str:
    findstr = "by @"
    lx = strx.rfind("by @")
    rx = strx.find(" ", lx + len(findstr))
    uname = strx[lx + len(findstr) : rx]
    ustr = f"by [@{uname}](https://github.com/{uname})"
    return strx.replace(f"by @{uname}", ustr)


def format_new_contributors(strx: str) -> str:
    findstr = "* @"
    lx = strx.rfind(findstr)
    rx = strx.find(" ", lx + len(findstr))
    uname = strx[lx + len(findstr) : rx]
    ustr = f"* [@{uname}](https://github.com/{uname})"
    return strx.replace(f"* @{uname}", ustr)


def _get_prnum_from_prlink(strx: str, as_int: bool = True) -> str:
    findstr = "in https://github.com/macrosynergy/macrosynergy/pull/"
    lx = strx.rfind(findstr)
    prnum = strx[lx + len(findstr) :]
    try:
        int(prnum)
    except ValueError:
        raise ValueError(f"Could not parse PR number from {strx}")
    return int(prnum) if as_int else prnum


def format_pr_link(strx: str) -> str:
    findstr = "in https://github.com/macrosynergy/macrosynergy/pull/"
    lx = strx.rfind(findstr)
    prnum = strx[lx + len(findstr) :]
    prcom = f"in [PR #{prnum}]({findstr.split(' ')[1]}{prnum})"
    rstr = strx[:lx] + prcom
    return rstr


def format_chg_log(strx: str) -> str:
    findstr = (
        "**Full Changelog**: https://github.com/macrosynergy/macrosynergy/compare/"
    )
    comp_str = strx.replace(findstr, "").replace("...", "←")
    comp_str = f"**Full Changelog**: [{comp_str}]({strx.split(' ')[-1]})"
    return comp_str


def clean_features_bugfixes(release_text: str) -> str:
    """
    Get a list of all features and bugfixes in the release,
    and format them as markdown lists.
    """
    ACCEPTED_PREFIXES = ["* Feature: ", "* Bugfix: ", "* Hotfix: "]
    return_str: List[str] = []
    features: List[str] = []
    fixes: List[str] = []
    for line in release_text.splitlines():
        if not line.startswith("* "):
            continue
        if not any(line.startswith(prefix) for prefix in ACCEPTED_PREFIXES):
            continue
        if line.startswith("* Feature: "):
            line = line[0] + line[10:]
            features.append(line)
        elif line.startswith("* Bugfix: ") or line.startswith("* Hotfix: "):
            line = line[0] + line[9:]
            fixes.append(line)

    # sort each list by pr number
    try:
        features.sort(key=lambda x: _get_prnum_from_prlink(x))
        fixes.sort(key=lambda x: _get_prnum_from_prlink(x))
    except Exception as exc:
        raise ValueError(
            "Could not parse release notes, please update manually."
        ) from exc
    if (len(features) + len(fixes)) == 0:
        return "#" + release_text

    if features:
        return_str += ["### New Features"]
        return_str += features
    if fixes:
        return_str += ["### Bugfixes"]
        return_str += fixes
    return_str += ["\n"]
    return_str += [release_text.splitlines()[-1]]

    return "\n".join(return_str)


def process_individual_release(release_dict: Dict) -> str:
    release_text: str = release_dict["body"]
    lines = []
    release_text = clean_features_bugfixes(release_text)
    for line in release_text.splitlines():
        linex = line
        if linex.startswith("* "):
            linex = format_gh_username(linex)
            linex = format_pr_link(linex)
        if linex.startswith("**Full Changelog**: https://"):
            linex = format_chg_log(linex)
        if "## New Contributors" in linex:
            linex = "#" + linex
        if line.startswith("* @"):
            linex = format_new_contributors(linex)
        lines.append(linex)
    release_text: str = "\n".join(lines)
    md = (
        f"## Release {release_dict['name']}\n\n"
        f"{release_text.strip()}"  # add one level of header
        f"\n\n[View complete release notes on GitHub]({release_dict['html_url']})"
        + LINE_SEPARATOR
    )

    return md


def _gh_api_call(url: str) -> List[Dict]:
    response: requests.Response = requests.get(url)
    response.raise_for_status()
    # in this case, response.json() is a list of dicts.
    rlist: List[Dict] = response.json()
    if hasattr(response, "links"):
        if "next" in response.links.keys():
            next_url = response.links["next"]["url"]
            next_page = _gh_api_call(next_url)
            rlist.extend(next_page)
    return rlist


def fetch_release_notes(
    output_path: str = PATH_TO_RELEASE_MD,
):
    """
    Fetches release notes from GitHub API and writes them to `output_path`.
    """
    releases_list: List[Dict] = _gh_api_call(RELEASES_API_URL)
    assert isinstance(releases_list, list)
    assert all(isinstance(x, dict) for x in releases_list)
    releases_list.sort(
        key=lambda x: version.parse(str(x["name"]).split(" ")[-1]), reverse=True
    )
    release_mds: List[str] = []
    for release in releases_list:
        release_mds.append(process_individual_release(release))

    release_md: str = "(release_notes)=\n# Release Notes\n\n" + "\n\n".join(release_mds)

    if os.path.exists(output_path):
        os.remove(output_path)
    with open(output_path, "w", encoding="utf8") as file:
        file.write(release_md)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--output-path",
        default=PATH_TO_RELEASE_MD,
        help="File path to write release notes to.",
    )
    args = parser.parse_args()
    fetch_release_notes(output_path=args.output_path)


if __name__ == "__main__":
    main()
