import ast
from typing import List, Dict, Any
import glob
import json


def read_python_file(file_path: str) -> str:
    with open(file_path, "r") as file:
        return file.read()


class DocstringVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.docstrings_info: List[Dict[str, Any]] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._check_docstring(node)
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._check_docstring(node)
        self.generic_visit(node)

    def visit_Module(self, node: ast.Module) -> None:
        self._check_docstring(node)
        self.generic_visit(node)

    def _check_docstring(self, node: ast.AST) -> None:
        docstring = ast.get_docstring(node)
        if docstring:
            # Retrieve the starting line number
            start_line = node.body[0].lineno
            # Calculate the ending line number
            end_line = start_line + len(docstring.split("\n")) - 1
            self.docstrings_info.append(
                {
                    "type": type(node).__name__,
                    "name": getattr(node, "name", "Module"),
                    "start_line": start_line,
                    "end_line": end_line,
                    "docstring": docstring,
                }
            )


def get_docstring_info(file_path: str) -> List[Dict[str, Any]]:
    with open(file_path, "r") as file:
        source = file.read()

    tree = ast.parse(source)
    visitor = DocstringVisitor()
    visitor.visit(tree)
    return visitor.docstrings_info


class DSParser:
    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        self.source = None
        self.docstrings_info: List[Dict[str, Any]] = []

    def read_python_file(self) -> None:
        self.source = read_python_file(self.file_path)


def format_python_file(file_path: str):
    """
    Formats the docstrings in the given python file.
    :param <str> file_path: The path to the python file.
    """
    # open the file
    with open(file_path, "r", encoding="utf8") as file:
        data_lines = file.readlines()

    # find all docstrings
    docstrings_info = get_docstring_info(file_path)
    for docstring_info in docstrings_info:
        print(json.dumps(docstring_info, indent=4))

    # find the last occurance of os.sep in the file path
    last_sep = file_path.rfind(".")
    # file_path = file_path[:last_sep] + "_fmt" + file_path[last_sep:]
    with open(file_path, "w", encoding="utf8") as file:
        file.writelines(data_lines)


def format_python_files(root_dir: str):
    """
    Formats the docstrings in all python files in the given directory.
    :param <str> root_dir: The root directory to search for python files.
    """
    for file in glob.glob(f"{root_dir}/**/*.py", recursive=True):
        format_python_file(file)


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Format docstrings in python files.")
#     parser.add_argument(
#         "-d",
#         "--dir",
#         type=str,
#         help="The root directory to search for python files.",
#         # required=True,
#         default="macrosynergy",
#     )
#     args = parser.parse_args()

#     format_python_files(args.dir)

if __name__ == "__main__":
    file_path = "macrosynergy/download/dataquery.py"
    format_python_file(file_path)
