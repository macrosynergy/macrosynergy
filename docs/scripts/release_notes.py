from typing import List, Optional, Tuple, Dict, Any, Callable

import git
import os
from tqdm import tqdm
import re
import sys

from datetime import datetime
import requests
from time import sleep
from functools import lru_cache

REPO_OWNER: str = "macrosynergy"
REPO_NAME: str = "macrosynergy"
REPO_URL: str = f"github.com/{REPO_OWNER}/{REPO_NAME}"


OAUTH_TOKEN: Optional[str] = os.getenv("GH_TOKEN", None)

from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from build_md import DocstringMethods


@lru_cache(maxsize=None)
def git_repo(repo_path: str = ".", fetch: bool = True) -> git.Repo:
    """
    Returns a git.Repo object from a path to a local git repository.
    Mainly purpose is to cache the repo object.

    :param <str> repo_path: Path to the local git repository.
    :param <bool> fetch: Whether to fetch the remote branches.
    :return <git.Repo>: A git.Repo object.
    """
    repo: git.Repo = git.Repo(repo_path)
    # fetch origin and all branch names and tags
    if fetch:
        for remote in repo.remotes:
            remote.fetch()
            remote.pull()

    return repo


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


def get_pr_reviews(
    pr_number: int, owner: str = REPO_OWNER, repo: str = REPO_NAME
) -> str:
    # https://api.github.com/repos/macrosynergy/macrosynergy/pulls/1061/reviews
    url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}/reviews"
    data = api_request(url)

    results: Dict[str, List[str]] = {
        "APPROVED": [],
        "CHANGES_REQUESTED": [],
        "COMMENTED": [],
        "DISMISSED": [],
    }

    for item in data:
        review_name: str = item["user"]["login"]
        review_state: str = item["state"]
        if review_state not in results:
            results[review_state] = []
        results[review_state].append(review_name)

    # remove any empty lists
    results = {k: v for k, v in results.items() if v}

    return results


def get_pr_general_info(
    pr_number: int, owner: str = REPO_OWNER, repo: str = REPO_NAME
) -> str:
    """
    Get the title of a pull request from GitHub.

    :param <str> owner: GitHub repository owner (username or organization name).
    :param <str> repo: GitHub repository name.
    :param <int or str> pr_number: Pull request number.
    :return <str>: Title of the pull request.
    """
    url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}"

    data = api_request(url)

    return {
        "title": data["title"],
        "author": data["user"]["login"],
        "head": data["head"],
        # head ref contains ref, sha, user, repo
    }


def get_pr_commit_authors(branch: str, repo_path: str = ".") -> List[str]:
    """
    Get the list of authors who made commits to the pull request.

    :param <str> branch: The name of the branch.
    :param <str> repo_dir: The path to the local git repository.
    """
    repo: git.Repo = git_repo(repo_path)

    branches: List[str] = [branchx.name for branchx in repo.branches]
    branch_index: int = branches.index(branch)

    # Iterate through the commits in the branch
    pr_commits = [
        commit.hexsha for commit in repo.heads[branch_index].commit.iter_items()
    ]

    authors: List[str] = [
        commit.author.name for commit in pr_commits if commit.author.name != "GitHub"
    ]

    authors = list(set(authors))
    return authors


def get_pr_info(pr_number: int, owner: str = REPO_OWNER, repo: str = REPO_NAME) -> str:
    """
    Returns a dict :
    title : str = title of the PR
    author : str = author of the PR
    url : str = url of the PR
    contributors : list = list of contributors to the PR
    reviews : list = list of reviews to the PR
    """

    args_dict = dict(pr_number=pr_number, owner=owner, repo=repo)

    url: str = f"https://{REPO_URL}/pull/{pr_number}"

    results: Dict[str, Any] = {}
    for name, func in [
        ("title_and_author", get_pr_general_info),
        ("reviews", get_pr_reviews),
    ]:
        try:
            results[name] = func(**args_dict)
        except Exception as exc:
            print(f"Exception in {name}: {exc}")

    results["contributors"]: List[str] = get_pr_commit_authors(
        branch=results["title_and_author"]["head"]["ref"]
    )

    result = dict(
        title=results["title_and_author"]["title"],
        author=results["title_and_author"]["author"],
        contributors=results["contributors"],
        reviews=results["reviews"],
        url=url,
    )

    return result


def get_diff_info(
    repo_path: str,
    source_branch: str,
    base_branch: str,
    show_progress: bool = True,
) -> Dict[str, List[str]]:
    """
    Get the names (numbers) of pull requests and their authors that are different between
        the input and output branches.

    :param <str> repo_path: Path to the local git repository.
    :param <str> source_branch: The name of the first branch.
    :param <str> base_branch: The name of the second branch.
    :return <Dict[str, List[str]]>: A dictionary with the PR numbers as keys and the
        authors as values.
    """
    repo: git.Repo = git_repo(repo_path)

    # Get the commits that are different between the two branches
    diff_commits: List[git.Commit] = list(
        repo.iter_commits(f"{base_branch}..{source_branch}")
    )

    pr_infos: Dict[str, Any] = {}

    # Regex pattern to extract PR number from commit message
    pr_pattern: re.Pattern = re.compile(r"Merge pull request #(\d+)")

    tqdm_comment: str = f"Generating release notes [{source_branch} ← {base_branch}]"

    def _getprinfo(commit: git.Commit) -> Dict[str, Any]:
        match: Optional[re.Match] = pr_pattern.search(commit.message)
        if match:
            pr_number = match.group(1)
            return get_pr_info(pr_number=int(pr_number))
        else:
            return {}

    for commit in tqdm(
        diff_commits,
        desc=tqdm_comment,
        total=len(diff_commits),
        disable=not show_progress,
    ):
        pr_info: Dict[str, Any] = _getprinfo(commit)
        if pr_info:
            pr_number: str = str(pr_info["number"])
            pr_infos[pr_number] = pr_info

    # # use a thread pool to get the PR info
    # with ThreadPoolExecutor(max_workers=10) as executor:
    #     futures: List[Future] = [
    #         executor.submit(_getprinfo, commit) for commit in diff_commits
    #     ]
    #     for future in tqdm(
    #         as_completed(futures),
    #         desc=tqdm_comment,
    #         total=len(diff_commits),
    #         disable=not show_progress,
    #     ):
    #         pr_info: Dict[str, Any] = future.result()
    #         pr_number: str = str(pr_info["number"])
    #         pr_infos[pr_number] = pr_info

    return pr_infos


def markdown_from_pr_attributes(
    pr_attrs: Dict[str, Any],
    dev=False,
) -> str:
    """
    Converts release info from dict to markdown.
    :param pr_attrs: A dictionary with the PR attributes.
    :return: A markdown string.
    """
    assert isinstance(pr_attrs, dict), "pr_attrs must be a dictionary"
    ATTRS: List[str] = ["title", "author", "url", "contributors", "reviews"]
    user_md_link = lambda x: f"[{x}](https://github.com/{x})"
    users_md_links = lambda x: map(user_md_link, x)

    review_states = {
        "APPROVED": "✅",
        "CHANGES_REQUESTED": "❌",
        "COMMENTED": "💬",
        "DISMISSED": "👋",
    }

    pr_number: int = pr_attrs["number"]
    title: str = pr_attrs["title"]
    url: str = pr_attrs["url"]
    author: str = pr_attrs["author"]
    contributors: List[str] = pr_attrs["contributors"]
    reviews: Dict[str, List[str]] = pr_attrs["reviews"]

    if not dev:
        # if the title starts with "Chore:", return ""
        if title.strip().split(":")[0].lower() == "chore":
            return ""

    md: str = f"- [**#{pr_number}** {title}]({url}) by {user_md_link(author)}\n"
    md += f"  - Contributors: {', '.join(users_md_links(contributors))}\n"
    if dev:
        for state, reviewers in reviews.items():
            md += (
                f"  - {review_states[state]} {state}: "
                f"{', '.join(users_md_links(reviewers))}\n"
            )

    return md


def generate_markdown(
    json_dict: Dict[str, Any], title: str = "Release Notes", dev: bool = False
) -> str:
    """
    Converts release info from dict to markdown
    """
    # order items by PR number
    json_dict = dict(sorted(json_dict.items(), key=lambda item: int(item[0])))
    # insert pr number as a key as well
    for pr_number, pr_info in json_dict.items():
        pr_info["number"]: int = int(pr_number)

    md = f"# {title}\n\n"
    for pr_number, pr_info in json_dict.items():
        md += markdown_from_pr_attributes(pr_attrs=pr_info, dev=dev)

    return md


def generate_notes(
    repo_path: str,
    source_branch: str,
    base_branch: str,
    dev: bool = False,
) -> str:
    repo: git.Repo = git_repo(repo_path)
    # fetch origin and all branch names and tags
    for remote in repo.remotes:
        remote.fetch()

    if source_branch == "":
        # check if the repo is in a detached head state
        if repo.head.is_detached:
            source_branch = (
                repo.head.log()[-1]
                .message.replace("checkout: moving from", "")
                .split()[0]
            )
        else:
            source_branch: str = repo.active_branch.name

    tags = sorted(repo.tags, key=lambda t: t.commit.committed_datetime)
    # if the base_branch is not specified, use the latest tag
    if base_branch == "":
        base_branch = tags[-1].name
        print(f"Warning: base_branch not specified. Using latest tag: {base_branch}")

    branch_tag_commits: List[str] = [tagx.name for tagx in repo.tags] + [
        branchx.name for branchx in repo.branches
    ]

    for branch, name in [
        (source_branch, "source_branch"),
        (base_branch, "base_branch"),
    ]:
        # if there is no origin/ prefix, add it and warn
        if branch not in branch_tag_commits:
            if not str(branch).startswith("origin/"):
                branch = f"origin/{branch}"
                print(f"Warning: {name} {branch} has no origin/ prefix. Adding it.")
                if branch not in branch_tag_commits:
                    raise ValueError(
                        f"{name} {branch} is not a valid branch or tag",
                        f"Valid branches and tags are: {branch_tag_commits}",
                    )

    result = get_diff_info(
        repo_path=repo_path, source_branch=source_branch, base_branch=base_branch
    )

    name: str = f"Changes: {source_branch} ← {base_branch}"

    md: str = generate_markdown(json_dict=result, title=name, dev=dev)
    md += "---\n\n"

    md = DocstringMethods.markdown_format(md)

    return md


def main(
    repo_path: str,
    source_branch: str,
    base_branch: str,
    output_path: str,
    dev: bool = False,
    all_versions: bool = False,
) -> None:
    md: str = generate_notes(
        repo_path=repo_path,
        source_branch=source_branch,
        base_branch=base_branch,
        dev=dev,
    )

    # if all_versions is True, generate release notes for all versions
    if all_versions:
        repo: git.Repo = git_repo(repo_path)
        tags = sorted(repo.tags, key=lambda t: t.commit.committed_datetime)
        md_list: List[str] = []
        for tag in tags[:-30]:
            # add notes for each tag→tag+1
            tagT: str = tags[tags.index(tag) + 1].name
            md_list += generate_notes(
                repo_path=repo_path,
                source_branch=tagT,
                base_branch=tag.name,
                dev=dev,
            )
        # reverse the list so that the latest version is first
        md_list += md
        md_list.reverse()
        md = "\n\n".join(md_list)

    md = DocstringMethods.markdown_format(md)

    # create the dirs to output_path if they don't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf8") as f:
        f.write(md)

    # print where the file was saved
    print(f"Release notes saved to {os.path.abspath(output_path)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate release notes from PRs")
    parser.add_argument(
        "--repo_path", "-p", type=str, default=".", help="path to the repo"
    )
    parser.add_argument(
        "--source_branch", "-s", type=str, default="", help="source branch (feature)"
    )
    parser.add_argument(
        "--base_branch",
        "-t",
        type=str,
        default="",
        help="base branch (main/master))",
    )

    parser.add_argument(
        "--output_path",
        "-o",
        type=str,
        default="docs/build/release_notes.md",
        help="output file path",
    )

    parser.add_argument(
        "--dev",
        "-d",
        action="store_true",
        help="include dev info (reviews, etc)",
    )

    parser.add_argument(
        "--all",
        "-a",
        action="store_true",
        help="Generate release notes for all releases",
    )

    args = parser.parse_args()

    main(
        repo_path=args.repo_path,
        source_branch=args.source_branch,
        base_branch=args.base_branch,
        output_path=args.output_path,
        dev=args.dev,
        # all_versions=args.all,
        all_versions=True,
    )

    # python docs/scripts/release_notes.py --repo_path . --source_branch "" --target_branch main --output_path release_notes.md
