try:
    import jupytext
except ImportError:
    raise ImportError(
        "`jupytext` is not installed. " "Please install jupytext to build the examples."
    )
import os
import shutil
import subprocess
import glob
from typing import List, Dict, Any, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

INPUT_DIR = "./docs/examples"
OUTPUT_DIR = "./docs/examples.build"
JB_TOC = os.path.join(INPUT_DIR, "_toc.yml")
JB_CONFIG = os.path.join(INPUT_DIR, "_config.yml")
JB_ENTRYPOINT = os.path.join(INPUT_DIR, "README.md")


def get_files(
    directory: str, extension: str = "*.py", recursive: bool = True
) -> List[str]:
    """Get all files with a given extension in a directory.
    :param <str> directory: The directory to search.
    :param <str> extension: The extension to search for.
    :param <bool> recursive: Whether to search recursively.
    :return <list[str]>: A list of file paths.
    """
    path = os.path.join(directory, extension)
    if recursive:
        path = os.path.join(directory, "**", extension)

    return glob.glob(path, recursive=recursive)


def convert_to_notebook(
    file_path: str,
    output_dir: str,
    input_dir: str,
    output_extension: str = ".ipynb",
    markdown: bool = False,
) -> str:
    """Convert a file to a notebook.
    :param <str> file_path: The file path.
    :param <str> output_dir: The output directory.
    :param <str> output_extension: The output extension.

    :return <str>: The output path.
    """
    # set the output_path such that the relative path is preserved
    output_path = (
        os.path.splitext(
            os.path.join(output_dir, os.path.relpath(file_path, input_dir))
        )[0]
        + output_extension
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    outputfmt = "md" if markdown else "notebook"

    subprocess.run(
        [
            "jupytext",
            "--to",
            outputfmt,
            "-o",
            output_path,
            file_path,
        ],
        check=True,
    )
    return output_path


def apply_black(output_dir: str) -> None:
    """Apply black to all notebooks in a directory.
    :param <str> output_dir: The output directory.
    """
    # apply black to the whole directory
    subprocess.run(["black", output_dir], check=True)


def create_jupyter_book(
    output_dir: str,
    jb_toc: str = JB_TOC,
    jb_config: str = JB_CONFIG,
    jb_entrypoint: str = JB_ENTRYPOINT,
    build: bool = True,
):
    """Create a jupyter book from a directory of notebooks.
    :param <str> output_dir: The output directory.
    :param <str> jb_toc: The jupyter book toc file.
    """
    # copy the toc to the output directory
    shutil.copy(jb_toc, output_dir)

    # copy the config to the output directory
    shutil.copy(jb_config, output_dir)

    # copy the entrypoint to the output directory
    shutil.copy(jb_entrypoint, output_dir)
    # rename the entrypoint to index.md
    # rpath = os.path.relpath(jb_entrypoint, INPUT_DIR)
    # os.rename(
    #     os.path.join(output_dir, rpath),
    #     os.path.join(output_dir, "index.md"),
    # )

    # create the jupyter book
    if build:
        subprocess.run(["jupyter-book", "build", output_dir], check=True)


def change_heading(
    py_file: str,
):
    # look for the first line, it should be a triple quote
    with open(py_file, "r") as f:
        lines = f.readlines()

    # find the first line that is not empty
    for i, line in enumerate(lines):
        if line.strip():
            break

    if '"""' not in line:
        raise ValueError(
            f"The first line should be a triple quote. File: {py_file}, line: {i}"
        )

    # split the first line by /
    fname = lines[i].replace('"""', "").strip().split("/")[-1]

    # in line, replace example/ with Example for:
    lines[i] = "\n".join(
        [
            "# %% [markdown]",
            f"# # Examples for: `{fname}`",
            "# %%",
        ]
    )

    lines[i] = lines[i].replace('"""', "")
    # join and write
    with open(py_file, "w") as f:
        f.write("".join(lines))


def convert_all_files(
    input_dir: str,
    output_dir: str,
    output_extension: str = None,
    markdown: bool = False,
    build: bool = False,
) -> None:
    """Convert all files in a directory to notebooks.
    :param <str> input_dir: The input directory.
    :param <str> output_dir: The output directory.
    :param <str> output_format: The output format.
    :param <str> output_extension: The output extension.
    """
    if not markdown and output_extension is None:
        output_extension = ".ipynb"
    elif markdown and output_extension is None:
        output_extension = ".md"

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    shutil.copytree(input_dir, output_dir, dirs_exist_ok=True)

    for file_path in get_files(output_dir):
        change_heading(file_path)

    print("Converting files to notebooks...")
    # for file_path in get_files(output_dir):
    #     convert_to_notebook(file_path=file_path, output_dir=output_dir,
    #       input_dir=output_dir, output_extension=output_extension)

    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(
                convert_to_notebook,
                file_path=file_path,
                output_dir=output_dir,
                input_dir=output_dir,
                output_extension=output_extension,
                markdown=markdown,
            )
            for file_path in get_files(output_dir)
        ]
        for future in as_completed(futures):
            future.result()

    # remvoe the py files
    for file_path in get_files(output_dir):
        if file_path.endswith(".py"):
            os.remove(file_path)

    print("Applying black formatting...")
    apply_black(output_dir)

    print("Notebooks built successfully.")

    print("Creating jupyter book...")
    create_jupyter_book(output_dir=output_dir, build=build)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Build the examples for the documentation."
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        type=str,
        default=INPUT_DIR,
        help="The input directory.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=OUTPUT_DIR,
        help="The output directory.",
    )
    parser.add_argument(
        "-m",
        "--markdown",
        action="store_true",
        help="Whether to convert to markdown.",
        default=False,
    )
    parser.add_argument(
        "-b",
        "--build",
        action="store_true",
        help="Whether to build the jupyter book.",
        default=False,
    )

    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    markdown = args.markdown
    build = args.build

    JB_TOC = os.path.join(input_dir, "_toc.yml")
    JB_CONFIG = os.path.join(input_dir, "_config.yml")
    JB_ENTRYPOINT = os.path.join(input_dir, "README.md")

    convert_all_files(
        input_dir=input_dir,
        output_dir=output_dir,
        markdown=markdown,
        build=build,
    )
