from typing import List, Optional, Tuple, Dict, Any

import git
import os
from tqdm import tqdm
import re
import sys

from datetime import datetime
import requests
from time import sleep

REPO_OWNER: str = "macrosynergy"
REPO_NAME: str = "macrosynergy"
REPO_URL: str = f"github.com/{REPO_OWNER}/{REPO_NAME}"

OAUTH_TOKEN: Optional[str] = os.getenv("GH_TOKEN", None)

from concurrent.futures import ThreadPoolExecutor, as_completed


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


def get_pr_title_and_author(
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
    }


def get_pr_commit_authors(
    pr_number: int, owner: str = REPO_OWNER, repo: str = REPO_NAME
) -> List[str]:
    """
    Get the list of authors who made commits to the pull request.

    :param pr_number: Pull request number.
    :param owner: GitHub repository owner.
    :param repo: GitHub repository name.
    :return: List of commit authors' usernames.
    """
    url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}/commits"
    data = api_request(url)

    # Extract the list of unique commit authors
    authors = list(
        set([commit["author"]["login"] for commit in data if commit["author"]])
    )

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

    jobs_dict: Dict[str, Any] = {
        "title_and_author": get_pr_title_and_author,
        "reviews": get_pr_reviews,
        "contributors": get_pr_commit_authors,
    }

    with ThreadPoolExecutor(max_workers=3) as executor:
        # use such that the name of the job is returned as well - tuple of (name, future)
        jobs = {
            executor.submit(func, **args_dict): name for name, func in jobs_dict.items()
        }

        results = {}
        for future in as_completed(jobs):
            name = jobs[future]
            try:
                results[name] = future.result()
            except Exception as exc:
                print(f"Exception in {name}: {exc}")
    result = dict(
        title=results["title_and_author"]["title"],
        author=results["title_and_author"]["author"],
        contributors=results["contributors"],
        reviews=results["reviews"],
        url=url,
    )

    return result


def get_diff_prs_and_authors(
    repo_path: str, source_branch: str, base_branch: str
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
    repo: git.Repo = git.Repo(repo_path)

    # Get the commits that are different between the two branches
    diff_commits: List[git.Commit] = list(
        repo.iter_commits(f"{base_branch}..{source_branch}")
    )

    pr_infos: Dict[str, Any] = {}

    # Regex pattern to extract PR number from commit message
    pr_pattern: re.Pattern = re.compile(r"Merge pull request #(\d+)")

    for commit in tqdm(diff_commits):
        match: Optional[re.Match] = pr_pattern.search(commit.message)
        if match:
            pr_number = match.group(1)
            pr_infos[pr_number]: Dict[str, Dict[str, str]] = get_pr_info(
                pr_number=int(pr_number)
            )

    return pr_infos


def json_to_md(json_dict: Dict[str, Any], title: str = "Release Notes") -> str:
    """
    Converts release info from dict to markdown
    """
    # order items by PR number
    json_dict = dict(sorted(json_dict.items(), key=lambda item: int(item[0])))
    user_md_link = lambda x: f"[{x}](https://github.com/{x})"
    users_md_links = lambda x: map(user_md_link, x)
    # review states and icons
    review_states = {
        "APPROVED": "✅",
        "CHANGES_REQUESTED": "❌",
        "COMMENTED": "💬",
        "DISMISSED": "👋",
    }

    md = f"# {title}\n\n"
    for pr_number, pr_info in json_dict.items():
        title: str = pr_info["title"]
        author: str = pr_info["author"]
        url: str = pr_info["url"]
        contributors: List[str] = pr_info["contributors"]
        reviews: Dict[str, List[str]] = pr_info["reviews"]

        md += f"- [**#{pr_number}** {title}]({url}) by {user_md_link(author)}\n"
        md += f"  - Contributors: {', '.join(users_md_links(contributors))}\n"
        for state, reviewers in reviews.items():
            md += f"  - {review_states[state]} {state}: {', '.join(users_md_links(reviewers))}\n"

    return md


def main(
    repo_path: str, source_branch: str, base_branch: str, output_path: str
) -> None:
    repo: git.Repo = git.Repo(repo_path)
    # fetch origin and all branch names and tags
    repo.remotes.origin.fetch()

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

    result = get_diff_prs_and_authors(
        repo_path=repo_path, source_branch=source_branch, base_branch=base_branch
    )

    name: str = f"Changes: {source_branch} → {base_branch}"

    md: str = json_to_md(result, title=name)

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
        default="main",
        help="base branch (main/master))",
    )

    parser.add_argument(
        "--output_path",
        "-o",
        type=str,
        default="docs/build/release_notes.md",
        help="output file path",
    )

    args = parser.parse_args()

    main(
        repo_path=args.repo_path,
        source_branch=args.source_branch,
        base_branch=args.base_branch,
        output_path=args.output_path,
    )

    # python docs/scripts/release_notes.py --repo_path . --source_branch "" --target_branch main --output_path release_notes.md
