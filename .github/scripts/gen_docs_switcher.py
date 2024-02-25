import json
import requests
import os

REPO: str = "macrosynergy/macrosynergy"
STABLE_BRANCH: str = "main"
TEST_BRANCH: str = "test"
LATEST_BRANCH: str = "develop"
DOCS_URL = "https://macrosynergy.readthedocs.io"


def setuppy_url(repo: str = REPO, branch: str = STABLE_BRANCH) -> str:
    return f"https://raw.githubusercontent.com/{repo}/{branch}/setup.py"


def get_latest_commit_url(repo: str = REPO, branch: str = STABLE_BRANCH) -> str:
    return f"https://api.github.com/repos/{repo}/commits/{branch}"


def _gh_request(
    repo: str = REPO,
    branch: str = STABLE_BRANCH,
    funcx=setuppy_url,
    rtype="text",
):
    assert rtype in ["text", "json"]
    try:
        url = funcx(repo, branch)
        r = requests.get(url)
        r.raise_for_status()
        if rtype == "text":
            return r.text
        return r.json()
    except Exception as exc:
        try:
            return _gh_request(repo, branch.replace("-", "/", 1), funcx)
        except Exception as exc:
            raise exc


def getpyfile(repo: str = REPO, branch: str = STABLE_BRANCH) -> str:
    return _gh_request(repo, branch, setuppy_url, "text")


def get_latest_commit_sha(repo: str = REPO, branch: str = STABLE_BRANCH) -> str:
    js = _gh_request(repo, branch, get_latest_commit_url, "json")
    try:
        return js["sha"]
    except Exception as exc:
        err_info = f"Type of js: {type(js)}\njs: {js} \n"
        err_info += f"Content of js: {js}\n"
        err_info += f"Error: {exc}"
        raise ValueError(err_info)


def get_version_from_py(repo: str = REPO, branch: str = STABLE_BRANCH) -> str:
    def find_line(line: str, breakline=None):
        for i, iterline in enumerate(pyfile.split("\n")):
            if iterline.strip().startswith(line):
                return iterline
            if breakline is not None and iterline.strip().startswith(breakline):
                return None
        return None

    pyfile = getpyfile(repo, branch)
    VERSION_LINES = ["MAJOR", "MINOR", "MICRO"]
    breakline = "# The first version not in the `Programming Language :: Python :: ...` classifiers above"

    _proc = lambda x: eval(str(x).split("=")[-1].strip())

    isreleased = find_line("ISRELEASED", breakline)

    vStr = ".".join(
        [str(_proc(find_line(f"{vx} = ", breakline))) for vx in VERSION_LINES]
    )
    if not _proc(isreleased):
        vStr += "dev0+" + get_latest_commit_sha(repo, branch)[:7]

    return vStr


def switch_alias(
    branch: str,
):
    return {
        "stable": STABLE_BRANCH,
        "latest": LATEST_BRANCH,
        STABLE_BRANCH: "stable",
        LATEST_BRANCH: "latest",
    }.get(branch, branch)


def generate_json(
    docs_site_url: str,
    repo: str = REPO,
    branches=[STABLE_BRANCH, TEST_BRANCH, LATEST_BRANCH],
) -> str:

    def _json(name, version, url):
        return {"name": name, "version": version, "url": url}

    data = []
    for branch in branches:
        branch = switch_alias(branch)
        version = get_version_from_py(repo, branch)
        branch = switch_alias(branch)
        branch = branch.replace("/", "-")
        url = f"{docs_site_url}/{branch}"
        data.append(_json(branch, version, url))

    return json.dumps(data, indent=4)


def parse_aws_ls_output_file(file: str) -> dict:
    if not os.path.exists(file):
        raise FileNotFoundError(f"File not found: {file}")
    with open(file, "r", encoding="utf8") as f:
        data = f.readlines()

    get_file = lambda x: str(x).split(" ")[-1].strip().split("/")[0] if "/" in x else ""
    data = [x for x in set(map(get_file, data)) if x.strip() != ""]
    return data


# print(','.join([x for x in set(map(lambda x: str(x).split(' ')[-1].strip().split('/')[0] if '/' in x else '', open(r'C:\Users\PalashTyagi\Downloads\folders.txt').readlines())) if x.strip() != '']))


def parse_branches_file(file: str) -> list:
    with open(file, "r", encoding="utf8") as f:
        return (",".join([x.strip() for x in f.readlines() if x.strip() != ""])).split(
            ","
        )


def main(
    docs_site_url: str,
    repo: str = REPO,
    branches=[STABLE_BRANCH, TEST_BRANCH, LATEST_BRANCH],
    outfile: str = "versions.json",
):
    for ix, bx in enumerate(branches):
        if bx == STABLE_BRANCH:
            branches[ix] = "stable"
        if bx == LATEST_BRANCH:
            branches[ix] = "latest"

    branches = list(set(branches))

    data = generate_json(docs_site_url, repo, branches)
    with open(outfile, "w", encoding="utf8") as file:
        file.write(data)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--docs-site-url",
        default=DOCS_URL,
        help="URL to the documentation site.",
    )
    parser.add_argument(
        "-r",
        "--repo",
        default=REPO,
        help="GitHub repository in the format `username/repo`.",
    )
    parser.add_argument(
        "-b",
        "--branches",
        default=",".join([STABLE_BRANCH, TEST_BRANCH, LATEST_BRANCH]),
        help="Comma-separated list of branches to generate versions for.",
    )
    parser.add_argument(
        "-bf",
        "--branches-file",
        default=None,
        help="File path to read the branches from.",
    )
    parser.add_argument(
        "-o",
        "--outfile",
        default="versionswitcher.json",
        help="File path to write the JSON to.",
    )
    args = parser.parse_args()

    if args.branches_file is not None:
        branches = parse_branches_file(args.branches_file)
    else:
        branches = str(args.branches).split(",")

    docs_site_url = args.docs_site_url
    repo = args.repo
    outfile = args.outfile
    main(docs_site_url, repo, branches, outfile)
