import ast
import os
from typing import Any, Dict, List, Optional, Tuple, Union
import jupyter_book
import warnings
import subprocess
import shutil
import argparse
import glob
import mdformat
import yaml

SOURCE_DIR = "./docs.old/build/"
BUILD_DIR = "./docs.old/build/"
OUTPUT_DIR = "./docs.old/build/html/"

BUILD_CONFIG = "./docs.old/assets/jpb-config.yml"
SITE_WIDE_CSS = "./docs.old/assets/custom_css.css"


def get_config() -> Dict[str, Any]:
    """Get the configuration for the build."""
    with open(BUILD_CONFIG, "r") as f:
        config = yaml.safe_load(f)
    return config


def copy_source_files(source_dir: str, destination_dir: str) -> bool:
    # if destination directory does not exist, create it
    if os.path.exists(destination_dir):
        shutil.rmtree(destination_dir)

    shutil.copytree(source_dir, destination_dir)
    return True


def create_jpb_config(destination_dir: str) -> bool:
    # EXTRACT "config" from jpb-config.yml
    build_config: Dict[str, Any] = get_config()
    config: Dict[str, Any] = build_config["config"]
    # place the config in only folder in the destination directory
    folder: str = [
        os.path.join(destination_dir, folder)
        for folder in os.listdir(destination_dir)
        if os.path.isdir(os.path.join(destination_dir, folder))
    ][0]
    with open(os.path.join(destination_dir, "_config.yml"), "w") as f:
        yaml.dump(config, f)
    return True


def create_dummy_readmes(destination_dir: str) -> bool:
    # if the current folder does not have a readme.md, create a dummy readme.md

    if "README.md" not in os.listdir(destination_dir):
        with open(
            os.path.normpath(os.path.join(destination_dir, "README.md")), "w"
        ) as f:
            f.write("# " + os.path.basename(destination_dir))

    # get all the folders in the destination directory
    folders: List[str] = [
        os.path.join(destination_dir, folder)
        for folder in os.listdir(destination_dir)
        if os.path.isdir(os.path.join(destination_dir, folder))
    ]

    # recursively go to each folder
    result: bool = True
    for folder in folders:
        result = result and create_dummy_readmes(folder)

    return result


def _get_toc_from_folder(folder: str, toc_path: str) -> str:
    # subprocess.run(["jupyter-book", "toc", "from-project", folder, "-f", "jb-book"] )  - pipe the output to a file toc_path
    toc = subprocess.run(
        ["jupyter-book", "toc", "from-project", folder, "-f", "jb-book"],
        stdout=subprocess.PIPE,
    ).stdout.decode("utf-8")

    with open(toc_path, "w") as f:
        f.write(toc)
    return toc_path


def create_toc(destination_dir: str) -> bool:
    build_config: Dict[str, Any] = get_config()

    toc: str = _get_toc_from_folder(
        destination_dir,
        os.path.join(
            # os.path.dirname
            (destination_dir),
            "_toc.yml",
        ),
    )
    return bool(toc)


def copy_css_file(destination_dir: str) -> bool:
    static_dir: str = os.path.join(destination_dir, "_static")
    os.makedirs(static_dir, exist_ok=True)
    shutil.copy(SITE_WIDE_CSS, static_dir)

    return True


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(
        description="Build the Jupyter Book from Markdown files."
    )
    parser.add_argument(
        "--source_dir",
        type=str,
        default=SOURCE_DIR,
        help="The directory containing the Markdown files.",
    )
    parser.add_argument(
        "--build_dir",
        type=str,
        default=BUILD_DIR,
        help="The directory containing the Jupyter Notebook files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=OUTPUT_DIR,
        help="The directory containing the HTML files.",
    )
    args = parser.parse_args()

    # copy the source files to the build directory
    # copy_source_files(args.source_dir, args.build_dir)
    # create the jpb config file
    create_jpb_config(args.build_dir)
    # create dummy readmes
    # get the list of folders in the build directory
    folder: str = [
        os.path.join(args.build_dir, folder)
        for folder in os.listdir(args.build_dir)
        if os.path.isdir(os.path.join(args.build_dir, folder))
    ][0]
    # create_dummy_readmes(destination_dir=folder)
    # create the toc file
    create_toc(destination_dir=args.build_dir)
    # copy the css file
    copy_css_file(destination_dir=args.build_dir)
