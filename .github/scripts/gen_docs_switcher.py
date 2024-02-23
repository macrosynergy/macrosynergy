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


def get_version_from_py(repo: str = REPO, branch: str = STABLE_BRANCH) -> str:
    if not branch in [STABLE_BRANCH, TEST_BRANCH, LATEST_BRANCH]:
        return setuppy_url(repo, LATEST_BRANCH)

    def find_line(line: str, breakline=None):
        for i, iterline in enumerate(pyfile.split("\n")):
            if iterline.strip().startswith(line):
                return iterline
            if breakline is not None and iterline.strip().startswith(breakline):
                return None
        return None

    url = setuppy_url(repo, branch)
    r = requests.get(url)
    r.raise_for_status()
    pyfile = r.text
    VERSION_LINES = ["MAJOR", "MINOR", "MICRO"]
    breakline = "# The first version not in the `Programming Language :: Python :: ...` classifiers above"

    _proc = lambda x: eval(str(x).split("=")[-1].strip())

    isreleased = find_line("ISRELEASED", breakline)

    vStr = ".".join(
        [str(_proc(find_line(f"{vx} = ", breakline))) for vx in VERSION_LINES]
    )
    if not _proc(isreleased):
        vStr += "rc" if branch == TEST_BRANCH else "dev"

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
        "-o",
        "--outfile",
        default="versionswitcher.json",
        help="File path to write the JSON to.",
    )
    args = parser.parse_args()
    docs_site_url = args.docs_site_url
    repo = args.repo
    branches = str(args.branches).split(",")
    outfile = args.outfile
    main(docs_site_url, repo, branches, outfile)
