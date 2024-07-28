import argparse
import glob
import re
from typing import List, Tuple, Dict, Any


def format_docstring(docstr: str) -> str:
    param_pattern = re.compile(r":param\s+<(\w+)>\s+(\w+):\s+(.*)")
    return_pattern = re.compile(r":return\s+<(\w+)>:\s+(.*)")
    raises_pattern = re.compile(r":raises\s+<(\w+)>:\s+(.*)")

    lines = docstr.split("\n")
    formatted_lines = []

    # Process :param lines
    params = []
    for line in lines:
        param_match = param_pattern.match(line)
        if param_match:
            param_type, param_name, param_desc = param_match.groups()
            params.append((param_name, param_type, param_desc))
        else:
            formatted_lines.append(line)

    if params:
        formatted_lines.append("Parameters")
        formatted_lines.append("----------")
        for name, type_, desc in params:
            formatted_lines.append(f"{name} : {type_}")
            formatted_lines.append(f"    {desc}")
        formatted_lines.append("")

    # Process :return line
    for line in lines:
        return_match = return_pattern.match(line)
        if return_match:
            return_type, return_desc = return_match.groups()
            formatted_lines.append("Returns")
            formatted_lines.append("-------")
            formatted_lines.append(f"{return_type}")
            formatted_lines.append(f"    {return_desc}")
            formatted_lines.append("")
        else:
            formatted_lines.append(line)

    # Process :raises line
    for line in lines:
        raises_match = raises_pattern.match(line)
        if raises_match:
            raises_type, raises_desc = raises_match.groups()
            formatted_lines.append("Raises")
            formatted_lines.append("------")
            formatted_lines.append(f"{raises_type}")
            formatted_lines.append(f"    {raises_desc}")
            formatted_lines.append("")
        else:
            formatted_lines.append(line)

    # Remove any duplicate lines
    final_lines = []
    for line in formatted_lines:
        if line not in final_lines:
            final_lines.append(line)

    return "\n".join(final_lines)


# def format_docstring(docstr: str):
def find_docstrings(file_lines: List[str]) -> List[Tuple[int, int]]:
    """
    Finds the start and end indices of all docstrings in the given file.
    :param <str> filestr: The content of the file.
    :return <List[Tuple[int, int]>: A list of tuples containing the start and end indices of all docstrings.
    """
    results: List[Tuple[int, int]] = []

    curr_start = None
    for i, line in enumerate(file_lines):
        if (line.strip() in ['"""', "'''"]) and curr_start is None:
            curr_start = i
        elif curr_start is not None and line.strip() in ['"""', "'''"]:
            results.append((curr_start, i))
            curr_start = None

    return results


def format_python_file(file_path: str):
    """
    Formats the docstrings in the given python file.
    :param <str> file_path: The path to the python file.
    """
    # open the file
    with open(file_path, "r", encoding="utf8") as file:
        data_lines = file.readlines()

    # find all docstrings
    docstrings_coords = find_docstrings(data_lines)
    for start, end in docstrings_coords:
        data_lines[start : end + 1] = format_docstring(data_lines[start : end + 1])

    with open(file_path, "w", encoding="utf8") as file:
        file.writelines(data_lines)


def format_python_files(root_dir: str):
    """
    Formats the docstrings in all python files in the given directory.
    :param <str> root_dir: The root directory to search for python files.
    """
    for file in glob.glob(f"{root_dir}/**/*.py", recursive=True):
        format_python_file(file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Format docstrings in python files.")
    parser.add_argument(
        "-d",
        "--dir",
        type=str,
        help="The root directory to search for python files.",
        required=True,
    )
    args = parser.parse_args()

    format_python_files(args.dir)
