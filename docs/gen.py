import os
import glob
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
OUTPUT_DIR = "./docs/source/gen_rsts"
STATIC_RSTS_DIR = "./docs/source/static_rsts"
PATH_TO_DOCS: str = os.path.join(OUTPUT_DIR, "release_notes.md")
LINE_SEPARATOR: str = "\n\n____________________\n\n"


def process_individual_release(release_dict: Dict) -> str:
    def format_gh_username(strx: str) -> str:
        findstr = "by @"
        lx = strx.rfind("by @")
        rx = strx.find(" ", lx + len(findstr))
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

    def format_chg_log(strx: str) -> str:
        findstr = (
            "**Full Changelog**: https://github.com/macrosynergy/macrosynergy/compare/"
        )
        comp_str = strx.replace(findstr, "").replace("...", "←")
        comp_str = f"**Full Changelog**: [{comp_str}]({strx.split(' ')[-1]})"
        return comp_str

    release_text: str = release_dict["body"]
    lines = []
    for line in release_text.splitlines():
        linex = line
        if linex.startswith("* "):
            linex = format_gh_username(linex)
            linex = format_pr_link(linex)
        if linex.startswith("**Full Changelog**: https://"):
            linex = format_chg_log(linex)
        lines.append(linex)
    release_text: str = "\n".join(lines)
    md = (
        f"## Release {release_dict['name']}\n\n"
        f"#{release_text.strip()}"  # add one level of header
        f"\n\n[View on GitHub]({release_dict['html_url']})" + LINE_SEPARATOR
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
    release_api_url: str = RELEASES_API_URL,
    output_path: str = PATH_TO_DOCS,
):
    """
    Fetches release notes from GitHub API and writes them to `output_path`.
    """
    releases_list: List[Dict] = _gh_api_call(release_api_url)
    assert isinstance(releases_list, list)
    assert all(isinstance(x, dict) for x in releases_list)
    releases_list.sort(
        key=lambda x: version.parse(str(x["name"]).split(" ")[-1]), reverse=True
    )
    release_mds: List[str] = []
    for release in releases_list:
        release_mds.append(process_individual_release(release))

    release_md: str = "# Release Notes\n\n" + "\n\n".join(release_mds)

    if os.path.exists(output_path):
        os.remove(output_path)
    with open(output_path, "w", encoding="utf8") as file:
        file.write(release_md)


def remove_file_spec(
    gen_dir: str = OUTPUT_DIR,
    static_dir: str = "./docs/source/static_rsts",
):
    static_rsts_basenames = list(
        map(os.path.basename, glob.glob(os.path.join(static_dir, "*.rst")))
    )
    for fname in glob.glob(os.path.join(gen_dir, "*.rst")):
        if os.path.basename(fname) in static_rsts_basenames:
            os.remove(fname)

    for fname in glob.glob(os.path.join(gen_dir,"*.rst")):
        # open the file
        with open(fname, "r", encoding="utf8") as file:
            data = file.readlines()
            data[0] = data[0].split(" ")[0] + "\n"

        with open(fname, "w", encoding="utf8") as file:
            file.writelines(data)


def generate_rsts(output_dir: str, package: str = "macrosynergy"):
    """
    Calls `sphinx-apidoc` to generate rst files for all modules in `package`.
    """
    rst_gen = f"sphinx-apidoc -o {output_dir} -fMeT {package}"
    os.system(rst_gen)
    remove_file_spec(gen_dir=output_dir, static_dir=STATIC_RSTS_DIR)


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
        default=True,
    )

    args = parser.parse_args()

    SHOW = args.show
    BUILD = args.build
    README = "./README.md"

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    generate_rsts(output_dir=OUTPUT_DIR)

    fetch_release_notes()
    # copy_readme(output_dir=OUTPUT_DIR, readme=README)

    if BUILD:
        make_docs(show=SHOW)


if __name__ == "__main__":
    main()
