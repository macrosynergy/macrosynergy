import ast
import os
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings
import shutil
import argparse
import glob
import mdformat
import yaml

SOURCE_DIR = "./docs/build/md/"
BUILD_DIR = "./docs/build/nb/"
OUTPUT_DIR = "./docs/build/html/"

BUILD_CONFIG = "./docs/static/jpb-config.yml"


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
    # place the config in destination as "_config.yml"
    with open(destination_dir + "_config.yml", "w") as f:
        yaml.dump(config, f, indent=4)

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


def create_toc(destination_dir: str) -> bool:
    # EXTRACT "toc" from jpb-config.yml
    build_config: Dict[str, Any] = get_config()
    toc: Dict[str, Any] = build_config["toc"]
    # get all the files in the destination directory as relative paths (without the destination_dir). use glob to get recursive files
    files: List[str] = glob.glob(destination_dir + "**/*.md", recursive=True)
    # remove the destination_dir from the file paths using os.path.relpath
    files = [os.path.relpath(file, destination_dir) for file in files]
    # remove package readme.md
    files = [file for file in files if file != "README.md"]
    # add a list of dicts of the form {"file": file - extension, "title": os.path.basename(file) - extension} to the toc

    target_dir = [
        os.path.join(destination_dir, folder)
        for folder in os.listdir(destination_dir)
        if os.path.isdir(os.path.join(destination_dir, folder))
    ]

    toc["chapters"]: List[Dict[str, Any]] = []
    for folder in [f for f in os.listdir(destination_dir) if os.path.isdir(f)]:
        ch_dict: Dict[str, Any] = {
            "file": folder.replace("\\", "/"),
            "sections": [
                {
                    "file": os.path.join(folder, f).replace("\\", "/")
                    for f in os.listdir(os.path.join(destination_dir, folder))
                }
            ],
        }

    toc["chapters"] = [
        {
            "file": str(os.path.splitext(file)[0]).replace(
                "\\", "/"
            ),  # remove the extension
            "title": str(os.path.splitext(file)[0]).replace(os.sep, "."),
        }
        for file in files
    ]
    # place the toc in destination as "_toc.yml"
    with open(destination_dir + "_toc.yml", "w") as f:
        yaml.dump(toc, f, indent=4)

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
    copy_source_files(args.source_dir, args.build_dir)
    # create the jpb config file
    create_jpb_config(args.build_dir)
    # create dummy readmes
    # get the list of folders in the build directory
    folder: str = [
        os.path.join(args.build_dir, folder)
        for folder in os.listdir(args.build_dir)
        if os.path.isdir(os.path.join(args.build_dir, folder))
    ][0]
    create_dummy_readmes(destination_dir=folder)
    # create the toc file
    create_toc(args.build_dir)
