import os
import glob
import shutil
import argparse
from typing import List, Tuple
import os.path as OPx

# All paths are relative to the root of the repository

TEMP_DIR = "./docs/docs.build"
TEMP_RST_DIR = "./docs/_temp_rst"
RST_OUTPUT_DIR = "./docs/source"
RELEASE_NOTES_MD = "release_notes.md"
SITE_OUTPUT_DIR = "./docs/build/"
PACKAGE_ROOT_DIR = "./"
PACKAGE_NAME = "macrosynergy"
PACKAGE_DOCS_DIR = "./docs"


def copy_subpackage_readmes(
    rst_output_dir: str = RST_OUTPUT_DIR,
    package: str = "macrosynergy",
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
        if OPx.basename(filex) == "README.md"
    ]
    new_names: List[str] = [
        ".".join(px.split(os.sep)[-2:]) for px in subpackage_readmes
    ]
    new_names = [OPx.join(rst_output_dir, px) for px in new_names]

    for old, new in zip(subpackage_readmes, new_names):
        # if the file already exists, remove it
        if OPx.exists(new):
            os.remove(new)
        shutil.copyfile(old, new)
        with open(new, "r", encoding="utf8") as file:
            data = file.readlines()
            while data[0].strip() == "":
                data.pop(0)
            data.pop(0)

        with open(new, "w", encoding="utf8") as file:
            file.writelines(data)


def remove_file_spec(
    rst_output_dir: str = RST_OUTPUT_DIR, permanent_files: List[str] = []
):
    """
    Removes '... module'/'... package' from the first line of each rst file.
    Specifically ignores the files in `perm_files`.
    """
    for fname in glob.glob(OPx.join(rst_output_dir, "*.rst")):
        if OPx.normpath(fname) in permanent_files:
            continue

        # open the file
        with open(fname, "r", encoding="utf8") as file:
            data = file.readlines()
            data[0] = data[0].split(" ")[0] + "\n"

        with open(fname, "w", encoding="utf8") as file:
            file.writelines(data)


def fetch_release_notes(release_notes_file: str):
    os.system(f'python ./docs/release_notes.py -o "{release_notes_file}"')


def format_docstrings(package_dir: str):
    os.system(f'python ./docs/format_docstrings.py -d "{package_dir}"')


def generate_rst_files(
    rst_output_dir: str = RST_OUTPUT_DIR,
    temp_rst_dir: str = TEMP_RST_DIR,
    package_name: str = PACKAGE_NAME,
    permanent_files: List[str] = [],
) -> bool:
    """
    Calls `sphinx-apidoc` to generate RST files for all modules in the package.
    Also makes sure none of the existing RST files are overwritten.
    """
    # get a list of all RST files in the rst_output_dir recursively
    rst_files = glob.glob(OPx.join(rst_output_dir, "**/*"), recursive=True)
    temp_rst_files: List[Tuple[str, str]] = []  # (src, dst)
    os.makedirs(OPx.normpath(OPx.join(os.getcwd(), temp_rst_dir)), exist_ok=True)

    # move all RST files to the temporary directory
    for ir, rst_file in enumerate(rst_files):
        if not OPx.isfile(rst_file):
            continue
        # copy files, and add to temp_rst_files keeping the relative paths intact
        dst = OPx.join(temp_rst_dir, OPx.relpath(rst_file, rst_output_dir))
        temp_rst_files.append((rst_file, dst))
        os.makedirs(OPx.dirname(dst), exist_ok=True)
        shutil.copyfile(rst_file, dst)

    rst_gen_cmd = f"sphinx-apidoc -o {rst_output_dir} -fMeT {package_name}"
    os.system(rst_gen_cmd)

    # copy all RST files from the temporary directory to the rst_output_dir
    for src, dst in temp_rst_files:
        expc_dst_file_path = OPx.join(
            OPx.normpath(OPx.join(os.getcwd(), rst_output_dir)),
            OPx.relpath(dst, OPx.join(os.getcwd(), temp_rst_dir)),
        )
        expc_src_file_path = OPx.join(
            OPx.normpath(OPx.join(os.getcwd(), temp_rst_dir)),
            OPx.relpath(src, OPx.join(os.getcwd(), rst_output_dir)),
        )
        if OPx.exists(expc_src_file_path) and OPx.exists(expc_dst_file_path):
            os.remove(expc_dst_file_path)
        os.makedirs(OPx.dirname(src), exist_ok=True)
        shutil.copyfile(src=dst, dst=src)

    # remove the temporary directory
    shutil.rmtree(temp_rst_dir)

    # get the list of "permanent" files from temp_rst_files
    permanent_files += [OPx.normpath(x[0]) for x in temp_rst_files]

    # remove '... module'/'... package' from the first line of each rst file
    remove_file_spec(rst_output_dir=rst_output_dir, permanent_files=permanent_files)

    # Delete the package.rst file
    os.remove(OPx.join(rst_output_dir, f"{package_name}.rst"))


def make_docs(
    docs_dir: str = "./docs",
) -> str:
    """
    Calls `make html` to generate the HTML files.
    :param <str> docs_dir: Path to the docs directory.
    :return <str>: directory where the HTML files are generated.
    """
    makescript = "make" + (".bat" if os.name == "nt" else "")
    makehtml = makescript + " html"
    makeclean = makescript + " clean"
    curdir = os.getcwd()
    os.chdir(OPx.normpath(OPx.join(curdir, docs_dir)))
    os.system(makeclean)
    os.system(makehtml)
    os.chdir(curdir)
    print("Documentation generated successfully.")

    return f"{docs_dir}/build"


def driver(
    package_dir: str = PACKAGE_ROOT_DIR,
    temp_dir: str = TEMP_DIR,
    temp_rst_dir: str = TEMP_RST_DIR,
    rst_output_dir: str = RST_OUTPUT_DIR,
    site_output_dir: str = SITE_OUTPUT_DIR,
    release_notes_md: str = RELEASE_NOTES_MD,
    package_name: str = PACKAGE_NAME,
    show_site: bool = False,
    clean: bool = False,
) -> bool:
    """Driver function for generating documentation.

    :param <str> package_dir: Path to the package directory.
    :param <str> temp_dir: Path to the temporary directory where the documentation will be
        generated.
    :param <str> rst_output_dir: Path to the directory where the reStructuredText files
        will be generated.
    :param <str> site_output_dir: Path to the directory where the HTML files will be
        generated.
    :param <str> package_name: Name of the package.
    :return <bool> success: Whether the documentation was generated successfully.
    """

    # create the temporary directories
    if OPx.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)

    if OPx.exists(temp_rst_dir):
        shutil.rmtree(temp_rst_dir)
    os.makedirs(temp_rst_dir, exist_ok=True)

    # from the root_dir, copy package name and docs into the temporary directory, make sure to allow the directory to exist/merge
    # cannot use shutil.copytree as it fails in a recursive setup

    files: List[str] = glob.glob(
        OPx.join(package_dir, package_name, "**/*"),
        recursive=True,
    )
    files += glob.glob(
        OPx.join(package_dir, PACKAGE_DOCS_DIR, "**/*"),
        recursive=True,
    )
    files = list(map(OPx.normpath, files))
    for ix, filex in enumerate(files):
        if OPx.isdir(filex):
            continue
        # if the extension is pyc or pyo or pyi, skip
        if any([filex.endswith(ext) for ext in [".pyc", ".pyo", ".pyi"]]):
            continue
        # copy file to relative path
        rel_path = OPx.relpath(filex, package_dir)
        dst_path = OPx.join(temp_dir, rel_path)
        os.makedirs(OPx.dirname(dst_path), exist_ok=True)
        shutil.copyfile(filex, dst_path)

    # change to the temporary directory
    starting_dir = os.getcwd()
    os.chdir(temp_dir)

    # generate the reStructuredText files
    generate_rst_files(
        rst_output_dir=rst_output_dir,
        temp_rst_dir=temp_rst_dir,
        package_name=package_name,
    )

    # generate release notes
    rnl_abs_path = OPx.abspath(OPx.normpath(OPx.join(rst_output_dir, release_notes_md)))
    fetch_release_notes(release_notes_file=rnl_abs_path)
    # format_docstrings(package_dir=f"./{package_name}")  # format the docstrings
    copy_subpackage_readmes(rst_output_dir=rst_output_dir, package=package_name)
    make_docs(docs_dir=PACKAGE_DOCS_DIR)

    temp_site_dir = OPx.normpath(OPx.join(temp_dir, PACKAGE_DOCS_DIR, "build"))

    os.chdir(starting_dir)

    if OPx.exists(site_output_dir):
        shutil.rmtree(site_output_dir)
    shutil.copytree(src=temp_site_dir, dst=site_output_dir)

    # remove _temp_rst directory
    shutil.rmtree(temp_rst_dir)

    if clean:
        shutil.rmtree(temp_dir)

    abssiteout = OPx.normpath(OPx.abspath(site_output_dir)).replace("\\", "/")
    indexfile = f"{abssiteout}/html/index.html"

    print("View the documentation at: ")
    print("\t\t", f"file://{indexfile}")
    if show_site:
        os.system(f"start {indexfile}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate documentation for the package."
    )
    parser.add_argument(
        "-c",
        "--clean",
        action="store_true",
        help="Clean the documentation build.",
        default=False,
    )

    parser.add_argument(
        "-s",
        "--show",
        action="store_true",
        help="Show the documentation in the browser.",
    )
    args = parser.parse_args()
    driver(show_site=args.show, clean=args.clean)