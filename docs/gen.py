import os
import shutil
import argparse
import requests
from packaging import version
from typing import Dict, List

REPO_OWNER: str = "macrosynergy"
ORGANIZATION: str = "macrosynergy"
REPO_NAME: str = "macrosynergy"
REPO_URL: str = f"github.com/{REPO_OWNER}/{REPO_NAME}"
RELEASES_API_URL = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/releases"
PATH_TO_DOCS: str = "./docs/source/gen_rsts/release_notes.md"
LINE_SEPARATOR: str = "\n\n____________________\n\n"


def process_individual_release(release_dict: Dict) -> str:
    def format_gh_username(strx: str) -> str:
        findstr = "by @"
        lx = strx.rfind("by @")
        rx = strx.find(" ", lx+len(findstr))
        uname = strx[lx + len(findstr) : rx]
        ustr = f"by [@{uname}](https://github.com/{uname})"
        return strx.replace(f"by @{uname}", ustr)

    def format_pr_link(strx: str) -> str:
        findstr = "in https://github.com/macrosynergy/macrosynergy/pull/"
        lx = strx.rfind(findstr)
        prnum = strx[lx + len(findstr) :]
        prcom = f"in [PR #{prnum}]({findstr.split(' ')[1]}{prnum})"
        rstr = strx[:lx] + prcom
        return rstr

    release_text: str = release_dict["body"]
    lines = []    
    for line in release_text.splitlines():
        linex = line
        if linex.startswith("* "):
            linex = format_gh_username(linex)
            linex = format_pr_link(linex)
        lines.append(linex)
    release_text: str = "\n".join(lines)
    md = (
        f"## Release {release_dict['name']}\n\n"
        f"#{release_text.strip()}"  # add one level of header
        f"\n\n[View on GitHub]({release_dict['html_url']})" + LINE_SEPARATOR
    )
    return md


def fetch_release_notes(
    release_api_url: str = RELEASES_API_URL,
    output_path: str = PATH_TO_DOCS,
):
    response: requests.Response = requests.get(release_api_url)
    if response.status_code == 200:
        releases_json: Dict = response.json()
    else:
        raise Exception(f"Request failed: {response.status_code}")

    # sort them by version number using packaging.version, and release['name'] is the version number
    releases_list = list(releases_json)
    releases_list.sort(key=lambda x: version.parse(x["name"]), reverse=True)
    release_mds: List[str] = []
    for release in releases_list:
        release_mds.append(process_individual_release(release))

    release_md: str = "# Release Notes\n\n" + "\n\n".join(release_mds)

    # if it exists, delete it
    if os.path.exists(output_path):
        os.remove(output_path)
    with open(
        output_path,
        "w",
        encoding="utf8",
    ) as file:
        # file.write()
        file.write(release_md)


def generate_rsts(output_dir: str, package: str = "macrosynergy"):
    """
    Calls `sphinx-apidoc` to generate rst files for all modules in `package`.
    """
    rst_gen = f"sphinx-apidoc -o {output_dir} -fMeT {package}"
    os.system(rst_gen)


def copy_readme(output_dir: str, readme: str):
    """
    Copies README.md to `output_dir`.
    """
    shutil.copy(readme, output_dir)


def make_docs(docs_dir: str = "./docs", show: bool = False):
    """
    Calls `make html` in `docs_dir` (makefile from sphinx-quickstart).
    """
    makescript = "make" + (".bat" if os.name == "nt" else "")
    makehtml = makescript + " html"
    makeclean = makescript + " clean"
    current_dir: str = os.getcwd()
    os.chdir(docs_dir)
    os.system(makeclean)
    os.system(makehtml)
    os.chdir(current_dir)
    print(f"Documentation generated successfully.")
    print(
        "Paste the following path into your browser to view:\n\t\t "
        f"file:///{os.path.abspath('docs/build/html/index.html')}"
    )

    if show:
        os.system(f"start docs/build/html/index.html")


def main():
    """
    Complete build sequence.
    """
    parser = argparse.ArgumentParser(description="Generate documentation.")
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show documentation in browser after generation.",
        default=False,
    )
    # build
    parser.add_argument(
        "--build",
        action="store_true",
        help="Build documentation.",
        default=False,
    )

    args = parser.parse_args()

    SHOW = args.show
    BUILD = args.build
    OUTPUT_DIR = "./docs/source/gen_rsts"
    README = "./README.md"

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    generate_rsts(output_dir=OUTPUT_DIR)

    fetch_release_notes()
    # copy_readme(output_dir=OUTPUT_DIR, readme=README)

    if BUILD:
        make_docs(show=SHOW)


if __name__ == "__main__":
    main()
