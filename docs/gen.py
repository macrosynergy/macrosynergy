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
OUTPUT_DIR = "./docs/source"
STATIC_RSTS_DIR = "./docs/source"
PATH_TO_DOCS: str = os.path.join(OUTPUT_DIR, "release_notes.md")
LINE_SEPARATOR: str = "\n\n____________________\n\n"


def remove_file_spec(gen_dir: str = OUTPUT_DIR, perm_files: List[str] = []):
    """
    Removes '... module'/'... package' from the first line of each rst file.
    Specifically ignores the files in `perm_files`.
    """
    for fname in glob.glob(os.path.join(gen_dir, "*.rst")):
        if os.path.abspath(fname) in perm_files:
            continue

        # open the file
        with open(fname, "r", encoding="utf8") as file:
            data = file.readlines()
            data[0] = data[0].split(" ")[0] + "\n"

        with open(fname, "w", encoding="utf8") as file:
            file.writelines(data)


def copy_subpackage_readmes(
    output_dir: str = OUTPUT_DIR, package: str = "macrosynergy"
):
    """
    Copies the README.md files from each subpackage to the docs/source/ folder.
    Renames them to <subpackage_name>.README.md to avoid conflicts.
    Also removes the header from each README.md file.
    """
    subpackage_readmes: List[str] = [
        # look at all levels recursively
        filex
        for filex in glob.glob(f"{package}/**/README.md", recursive=True)
        # if basename is README.md
        if os.path.basename(filex) == "README.md"
    ]
    new_names: List[str] = [
        ".".join(px.split(os.sep)[-2:]) for px in subpackage_readmes
    ]
    new_names = [os.path.join(output_dir, px) for px in new_names]

    for old, new in zip(subpackage_readmes, new_names):
        # if the file already exists, remove it
        if os.path.exists(new):
            os.remove(new)
        shutil.copyfile(old, new)
        with open(new, "r", encoding="utf8") as file:
            data = file.readlines()
            while data[0].strip() == "":
                data.pop(0)
            data.pop(0)

        with open(new, "w", encoding="utf8") as file:
            file.writelines(data)


def generate_rsts(
    output_dir: str, package: str = "macrosynergy", perm_files: List[str] = []
):
    """
    Calls `sphinx-apidoc` to generate rst files for all modules in `package`.
    """
    # make a temporary directory to store the generated rst files
    # move all files in output_dir to a temporary directory
    temp_folder = os.path.join(output_dir, "_temp")
    os.makedirs(temp_folder, exist_ok=True)
    for fname in glob.glob(os.path.join(output_dir, "*")):
        if os.path.isfile(fname):
            shutil.move(fname, temp_folder)

    rst_gen = f"sphinx-apidoc -o {output_dir} -fMeT {package}"
    os.system(rst_gen)

    # move all files from the temporary directory to output_dir, overwriting the generated rst files
    for fname in glob.glob(os.path.join(temp_folder, "*")):
        if os.path.isfile(fname):
            if os.path.exists(os.path.join(output_dir, os.path.basename(fname))):
                os.remove(os.path.join(output_dir, os.path.basename(fname)))
            shutil.move(fname, output_dir)

    # remove the temporary directory
    shutil.rmtree(temp_folder)

    remove_file_spec(gen_dir=output_dir, perm_files=perm_files)


def make_docs(
    docs_dir: str = "./docs",
    show: bool = False,
):
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


def fetch_release_notes(release_notes_file: str):
    os.system(f"python docs/release_notes.py --output-path {release_notes_file}")


def driver(
    build: bool = False,
    show: bool = False,
    prod: bool = False,
    output_dir: str = OUTPUT_DIR,
    docs_dir: str = "./docs",
):
    """
    Driver function.
    :param <bool> build: Whether to build the documentation.
    :param <bool> show: Whether to show the documentation in browser after generation.
    :param <bool> prod: Whether to build the documentation for production.
    """
    # get a list of all files in the output directory
    perm_files: List[str] = [
        (filex) for filex in glob.glob(f"{docs_dir}/**/*", recursive=True)
    ]
    if not build:
        return
    os.makedirs(output_dir, exist_ok=True)

    # copy all rsts

    generate_rsts(output_dir=output_dir, perm_files=perm_files)
    release_notes_file = os.path.join(output_dir, "release_notes.md")
    fetch_release_notes(release_notes_file=release_notes_file)
    copy_subpackage_readmes(output_dir=output_dir)
    make_docs(docs_dir=docs_dir, show=show)

    if prod:
        return

    # get a list of all files in docs/build/
    build_files: List[str] = [
        (filex) for filex in glob.glob(f"{docs_dir}/build/**/*", recursive=True)
    ]
    keep_files = perm_files + build_files
    for filex in glob.glob(f"{docs_dir}/**/*", recursive=True):
        if (filex) not in keep_files and os.path.isfile(filex):
            os.remove(filex)


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

    parser.add_argument(
        "--prod",
        action="store_true",
        help="Build documentation for production.",
        default=False,
    )

    args = parser.parse_args()

    SHOW = args.show
    BUILD = args.build
    PROD = args.prod
    DOCS_DIR = "./docs"

    driver(build=BUILD, show=SHOW, prod=PROD, docs_dir=DOCS_DIR)


if __name__ == "__main__":
    main()
