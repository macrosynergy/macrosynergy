import argparse
import glob
import re
import os
from typing import List, Tuple, Dict, Any

sphinx_directives = {
    "param": "Parameters",
    "return": "Returns",
    "raises": "Raises",
    "rtype": "Returns",
}


def is_directive(line: str) -> bool:
    return any(
        line.startswith(f":{directive}") for directive in sphinx_directives.keys()
    )


def format_param_docstr(docstr: str) -> str:
    # the docstring is in the format ":param <TYPE> name: description"
    type_start = docstr.find("<") + 1
    type_end = docstr.find(">", type_start)
    type_str = docstr[type_start:type_end]

    name_start = type_end + 1
    name_end = docstr.find(":", name_start)
    name_str = docstr[name_start:name_end]


def format_return_docstr(docstr: str) -> str: ...


def format_raises_docstr(docstr: str) -> str: ...


def format_rtype_docstr(docstr: str) -> str: ...


def format_directive(docstr: str) -> str:
    patterns = ["param", "return", "raises", "rtype"]
    fdict = {f: eval(f"format_{f}_docstr") for f in patterns}
    assert is_directive(docstr), "The given docstring is not a directive."

    for pattern in patterns:
        if docstr.startswith(f":{pattern}"):
            return fdict[pattern](docstr)


def format_docstring(data_lines: List[str]) -> str:
    directive_lines: List[bool] = list(map(is_directive, data_lines))

    formatted_groups: List[str] = []
    curr_group = []
    for i, line in enumerate(data_lines):
        if (directive_lines[i] is True) and bool(curr_group):
            formatted_groups.append("\n".join(curr_group))
            curr_group = []
        curr_group.append(line)
    if bool(curr_group):
        formatted_groups.append("\n".join(curr_group))

    for group in formatted_groups:
        if is_directive(group):
            format_directive(group)

    formatted_groups

    return "\n".join(formatted_groups)


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

    # find the last occurance of os.sep in the file path
    last_sep = file_path.rfind(".")
    file_path = file_path[:last_sep] + "_fmt" + file_path[last_sep:]
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
        # required=True,
        default="macrosynergy",
    )
    args = parser.parse_args()

    format_python_files(args.dir)
