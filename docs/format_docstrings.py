import argparse
import glob


def format_docstring(docstr: str):
    """
    Formats the docstring to remove parameter types and return types from the docstring.
    :param <str> docstr: The docstring to format.
    :return <str>: The formatted docstring.
    """
    # only works with param; not with return, raises, etc.
    if not docstr.strip().startswith(":param"):
        return docstr

    first_colon: int = docstr.find(":")
    second_colon: str = docstr.find(":", first_colon)

    # look for first "<"
    start = docstr.find("<") - 1
    # find the next first instance of ">"
    end = docstr.find(">", start)

    # remove the substring from the docstring
    docstr = docstr[:start] + docstr[end + 1 :]
    return docstr


def format_python_file(file_path: str):
    """
    Formats the docstrings in the given python file.
    :param <str> file_path: The path to the python file.
    """
    # open the file
    with open(file_path, "r", encoding="utf8") as file:
        data = file.readlines()

    with open(file_path, "w", encoding="utf8") as file:
        for line in data:
            file.write(format_docstring(line))


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