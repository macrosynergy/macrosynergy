import os
import glob
import shutil
import argparse
from typing import List, Tuple

# All paths are relative to the root of the repository

TEMP_DIR = "./docs/docs.build"
TEMP_RST_DIR = "./docs/_temp_rst"
RST_OUTPUT_DIR = "./docs/source"
RELEASE_NOTES_MD = "release_notes.md"
SITE_OUTPUT_DIR = "./docs/build/"
PACKAGE_ROOT_DIR = "./"
PACKAGE_NAME = "macrosynergy"


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
        if os.path.basename(filex) == "README.md"
    ]
    new_names: List[str] = [
        ".".join(px.split(os.sep)[-2:]) for px in subpackage_readmes
    ]
    new_names = [os.path.join(rst_output_dir, px) for px in new_names]

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


def remove_file_spec(rst_output_dir: str = RST_OUTPUT_DIR, perm_files: List[str] = []):
    """
    Removes '... module'/'... package' from the first line of each rst file.
    Specifically ignores the files in `perm_files`.
    """
    for fname in glob.glob(os.path.join(rst_output_dir, "*.rst")):
        if os.path.abspath(fname) in perm_files:
            continue

        # open the file
        with open(fname, "r", encoding="utf8") as file:
            data = file.readlines()
            data[0] = data[0].split(" ")[0] + "\n"

        with open(fname, "w", encoding="utf8") as file:
            file.writelines(data)


def fetch_release_notes(release_notes_file: str):
    os.system(f'python ./docs/release_notes.py -o "{release_notes_file}"')


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
    rst_files = glob.glob(os.path.join(rst_output_dir, "**/*"), recursive=True)
    temp_rst_files: List[Tuple[str, str]] = []  # (src, dst)
    os.makedirs(temp_rst_dir, exist_ok=True)

    # move all RST files to the temporary directory
    for ir, rst_file in enumerate(rst_files):
        if os.path.isfile(rst_file):
            shutil.move(rst_file, temp_rst_dir)
            temp_rst_files.append((rst_file, os.path.join(temp_rst_dir, rst_file)))

    rst_gen_cmd = f"sphinx-apidoc -o {temp_rst_dir} -fMeT {package_name}"
    os.system(rst_gen_cmd)

    # copy all RST files from the temporary directory to the rst_output_dir
    for src, dst in temp_rst_files:
        # if there is already a file with the same basename in the rst_output_dir, delete it
        if os.path.isfile(os.path.join(rst_output_dir, os.path.basename(src))):
            os.remove(os.path.join(rst_output_dir, os.path.basename(src)))
        shutil.move(src=dst, dst=src)  # yes, src and dst are reversed here

    # remove the temporary directory
    shutil.rmtree(temp_rst_dir)

    # get the list of "permanent" files from temp_rst_files
    permanent_files += [x[0] for x in temp_rst_files]

    # remove '... module'/'... package' from the first line of each rst file
    remove_file_spec(rst_output_dir=rst_output_dir, permanent_files=permanent_files)


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
    os.chdir(docs_dir)
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
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)

    if os.path.exists(temp_rst_dir):
        shutil.rmtree(temp_rst_dir)
    os.makedirs(temp_rst_dir, exist_ok=True)

    shutil.copytree(src=package_dir, dst=temp_dir)

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
    fetch_release_notes(release_notes_file=os.path.join(temp_dir, release_notes_md))
    copy_subpackage_readmes(rst_output_dir=rst_output_dir, package=package_name)
    make_docs(docs_dir=temp_dir)

    temp_site_dir = os.path.join(temp_dir, "build")

    os.chdir(starting_dir)

    if os.path.exists(site_output_dir):
        shutil.rmtree(site_output_dir)
    shutil.copytree(src=temp_site_dir, dst=site_output_dir)
