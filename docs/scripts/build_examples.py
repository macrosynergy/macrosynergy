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

INPUT_DIR = "./docs/examples"
OUTPUT_DIR = "./docs/examples.build"


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
) -> None:
    """Convert a file to a notebook.
    :param <str> file_path: The file path.
    :param <str> output_dir: The output directory.
    :param <str> output_extension: The output extension.
    """
    # set the output_path such that the relative path is preserved
    output_path = os.path.join(output_dir, os.path.relpath(file_path, input_dir))
    output_path = os.path.splitext(output_path)[0] + output_extension
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    subprocess.run(
        [
            "jupytext",
            "--to",
            "notebook",
            "-o",
            output_path,
            file_path,
        ],
        check=True,
    )


def convert_all_files(
    input_dir: str,
    output_dir: str,
    output_extension: str = ".ipynb",
) -> None:
    """Convert all files in a directory to notebooks.
    :param <str> input_dir: The input directory.
    :param <str> output_dir: The output directory.
    :param <str> output_format: The output format.
    :param <str> output_extension: The output extension.
    """
    os.makedirs(output_dir, exist_ok=True)

    for file_path in get_files(input_dir):
        convert_to_notebook(
            file_path=file_path,
            output_dir=output_dir,
            input_dir=input_dir,
            output_extension=output_extension,
        )


if __name__ == "__main__":
    convert_all_files(INPUT_DIR, OUTPUT_DIR)
