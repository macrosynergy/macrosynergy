import ast
from typing import List, Dict, Any, Tuple, Optional
import glob
import json


def read_python_file(file_path: str) -> str:
    with open(file_path, "r") as file:
        return file.read()


MAX_LINE_LENGTH = 88


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
            end_line = start_line + len(docstring.split("\n")) + 1
            # NOTE: this may break if the docstring is at the very end of the file with no newline
            # however, this case does not seem to exist in the current codebase
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


SPHINX_DIRECTIVES: Dict[str, str] = {
    "param": "Parameters",
    "raises": "Raises",
    "return": "Returns",
    "rtype": "Returns",
}
DIRECTIVES = [f":{directive}" for directive in SPHINX_DIRECTIVES]


def is_directive_line(line: str) -> bool:
    return line.strip().startswith(tuple(DIRECTIVES))


def get_indentation(line: str) -> int:
    return len(line) - len(line.lstrip())


def split_docstring(docstring: str) -> Dict[str, List[str]]:
    # First, split the docstring by lines
    lines = docstring.split("\n")
    # Apply is_directive_line to each line
    directive_lines = list(map(is_directive_line, lines))

    # Initialize variables to hold the description and sections
    sections = []
    current_section = []

    for i, line in enumerate(lines):
        if directive_lines[i]:
            # This line is a directive
            if current_section:
                # Append the current section to sections before starting a new section
                sections.append(" ".join(current_section).strip())
                current_section = []
            # Start the new section with the directive line
            current_section.append(line.strip())
        else:
            # This line is not a directive, add to the current section
            current_section.append(line.strip())

    # Append the last section if it exists
    if current_section:
        sections.append(" ".join(current_section).strip())

    return dict(
        description=sections[0],
        directives=sections[1:],
    )


MAX_LINE_LENGTH = 88


def format_param_directive(directive: str) -> str:
    # e.g.: ":param <str> name: The name of the person."
    try:
        assert directive.startswith(":param")
        # split by < and >
        typex = directive.split("<", 1)[1].split(">", 1)[0].strip()
        name, description = directive.split(">", 1)[1].split(":", 1)
        name, description = name.strip(), description.strip()
        idt = chr(32) * 4
        return f"{name} : ({typex})\n{idt}{description}"
    except Exception as e:
        print(directive)
        print(e)
        return directive


def format_raises_directive(directive: str) -> str:
    # e.g.: ":raises <Exception>: description"
    assert directive.startswith(":raises")
    # split by < and >
    typex = directive.split("<", 1)[1].split(">", 1)[0].strip()
    description = directive.split(">", 1)[1].split(":", 1)[1].strip()
    idt = chr(32) * 4
    return f"{typex}\n{idt}{description}"


def format_return_directive(directive: str) -> str:
    # e.g.: ":return <type>: description"
    assert directive.startswith(":return") or directive.startswith(":rtype")
    # split by < and >
    typex = directive.split("<", 1)[1].split(">", 1)[0].strip()
    description = directive.split(">", 1)[1].split(":", 1)[1].strip()
    idt = chr(32) * 4
    return f"{typex}\n{idt}{description}"


def format_docstring_section(section: Dict[str, Any]) -> Dict[str, Any]:
    dir_func_map = {
        "param": format_param_directive,
        "raises": format_raises_directive,
        "return": format_return_directive,
        "rtype": format_return_directive,
    }
    # first group the sections by directive
    directive_sections = {}
    # if there are any param directives, they should be first
    for directive in ["param", "raises", "return", "rtype"]:
        directive_sections[directive] = [
            dirx
            for dirx in section["directives"]
            if str(dirx).startswith(f":{directive}")
        ]

    # then format each directive section
    formatted_sections = {}
    for directive, directives in directive_sections.items():
        if directives:
            formatted_sections[SPHINX_DIRECTIVES[directive]] = [
                dir_func_map[directive](dirx) for dirx in directives
            ]

    # Put the description at the top
    output = ""
    # dscr = roll_description(section["description"])
    dscr = str(section["description"])
    if dscr:
        output += dscr + "\n\n"

    # idt = chr(32) * 4
    for dirtype, dirtextlist in formatted_sections.items():
        full_name = f"{dirtype}\n{'-' * len(dirtype)}"
        output += f"{full_name}\n"
        output += "\n\n".join(dirtextlist) + "\n\n"

    return output


class DSParser:
    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        self.source = None
        self.docstrings_info: List[Dict[str, Any]] = []
        self.docstrings_sections: List[Dict[str, Any]] = []
        self.read_python_file()

    def read_python_file(self) -> None:
        self.source = read_python_file(self.file_path)
        self.docstrings_info = get_docstring_info(self.file_path)
        for ds_info in self.docstrings_info:
            docstring = ds_info["docstring"]
            ds_info["docstrings_sections"] = split_docstring(docstring)
            ds_info["formatted_sections"] = format_docstring_section(
                ds_info["docstrings_sections"]
            )
            # ds_info["indentation"] = ds_info["docstrings_sections"]["indentation"]
        self.docstrings_info

    def write_formatted_file(self, file_path: Optional[str] = None) -> None:
        source_lines = self.source.split("\n")
        assert 0 == 1, "Docstrings formatted correctly. Need to fix indentation logic."
        for ds_info in self.docstrings_info:
            start_line = ds_info["start_line"]
            end_line = ds_info["end_line"]
            idt = chr(32) * (ds_info["indentation"])
            source_lines[start_line - 1] = (
                f'{idt}"""\n' + ds_info["formatted_sections"] + f'\n{idt}"""'
            )
            # source_lines[end_line - 1] =
            for i in range(start_line, end_line):
                source_lines[i] = ""

        if file_path is None:
            file_path = self.file_path
        with open(file_path, "w") as file:
            file.write("\n".join(source_lines))


def format_python_file(file_path: str):
    """
    Formats the docstrings in the given python file.
    :param <str> file_path: The path to the python file.
    """
    last_dot = file_path.rfind(".")
    out_path = file_path[:last_dot] + "_fmt" + file_path[last_dot:]
    DSParser(file_path).write_formatted_file(file_path=out_path)


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
