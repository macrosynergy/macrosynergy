import ast
from typing import List, Dict, Any, Tuple, Optional
import glob
import textwrap
import argparse
import os
import colorama
import black
from pathlib import Path

colorama.init(autoreset=True)


def read_python_file(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


MAX_LINE_LENGTH = 88

TEMPFILE_SUFFIX = "_fmttemp"


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
            # get the indentation of the docstring by checking start_line-1
            indentation = node.body[0].col_offset

            self.docstrings_info.append(
                {
                    "type": type(node).__name__,
                    "name": getattr(node, "name", "Module"),
                    "start_line": start_line,
                    "end_line": end_line,
                    "docstring": docstring,
                    "indentation": indentation,
                }
            )


def get_docstring_info(source_str: str) -> List[Dict[str, Any]]:
    tree = ast.parse(source_str)
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
    assert directive.startswith(":param")
    # split by < and >
    typex = directive.split("<", 1)[1].split(">", 1)[0].strip()
    name, description = directive.split(">", 1)[1].split(":", 1)
    name, description = name.strip(), description.strip()
    idt = chr(32) * 4
    return f"{name} : {typex}\n{idt}{description}"


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


def format_docstring_section(
    section: Dict[str, Any], idt_level: int = 4
) -> Dict[str, Any]:
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
        # output += dscr + "\n\n"
        dscr = text_wrap(dscr, idt=idt_level)
        output += dscr + "\n\n"

    # idt = chr(32) * 4
    for dirtype, dirtextlist in formatted_sections.items():

        new_ds = f"{dirtype}\n{'-' * len(dirtype)}\n"
        new_ds = text_wrap(new_ds, idt=idt_level) + "\n"

        for ix, dirtext in enumerate(dirtextlist):
            dtext = text_wrap(dirtext, idt=idt_level)
            if len(dtext.splitlines()) > 2:
                split_dtext = dtext.splitlines()
                apt_idt = chr(32) * get_indentation(dtext.splitlines()[1])

                dtext = "\n".join(
                    [split_dtext[0]] + [apt_idt + x.lstrip() for x in split_dtext[1:]]
                )

            if ix < len(dirtextlist) - 1:
                dtext += "\n"

            new_ds += dtext
        output += new_ds + "\n\n"

    output = output.rstrip()
    return output


def text_wrap(text: str, max_line_length: int = MAX_LINE_LENGTH, idt: int = 4) -> str:
    indent = " " * idt
    paragraphs = text.splitlines()
    wrapped_paragraphs = [
        textwrap.fill(
            paragraph,
            width=max_line_length,
            initial_indent=indent,
            subsequent_indent=indent,
        )
        for paragraph in paragraphs
    ]
    wrapped_text = "\n".join(wrapped_paragraphs)
    return wrapped_text


class DSParser:
    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        self.source = None
        self.docstrings_info: List[Dict[str, Any]] = []
        self.docstrings_sections: List[Dict[str, Any]] = []
        self.read_python_file()

    def read_python_file(self) -> None:
        self.source = read_python_file(self.file_path)
        self.docstrings_info = get_docstring_info(source_str=self.source)
        for ds_info in self.docstrings_info:
            docstring = ds_info["docstring"]
            ds_info["docstrings_sections"] = split_docstring(docstring)
            ds_info["formatted_sections"] = format_docstring_section(
                section=ds_info["docstrings_sections"], idt_level=ds_info["indentation"]
            )
            # ds_info["indentation"] = ds_info["docstrings_sections"]["indentation"]
        self.docstrings_info

    def format_content(self) -> None:
        source_lines = self.source.split("\n")
        for ds_info in self.docstrings_info:
            start_line = ds_info["start_line"] - 1
            end_line = ds_info["end_line"] - 1
            indentation: str = ds_info["indentation"] * " "
            formatted_sections: str = ds_info["formatted_sections"]

            ref_lines = source_lines[start_line:end_line]
            # if the start and end line are the same, then the docstring is on one line
            if len(ref_lines) > 2:
                while not (
                    source_lines[start_line:end_line][-1]
                    .strip()
                    .endswith(tuple(['"""', "'''"]))
                ):
                    end_line += 1
            else:
                print(f"Docstring on one line: {self.file_path}")
                continue

            for ix, line in enumerate(source_lines[start_line:end_line]):
                source_lines[start_line + ix] = None

            # at the start line, write the new docstring
            out_dstring = f'{indentation}"""\n{formatted_sections}\n{indentation}"""\n'
            source_lines[start_line] = out_dstring

        # remove the None lines
        rmlines = lambda lines: [line for line in lines if line is not None]
        source_lines = rmlines(source_lines)

        return source_lines

    def write_formatted_file(self, file_path: Optional[str] = None) -> None:
        source_lines = self.format_content()
        if file_path is None:
            file_path = self.file_path
        with open(file_path, "w", encoding="utf-8") as file:
            file.write("\n".join(source_lines))


def check_valid_python_content(content: str) -> bool:
    try:
        ast.parse(content)
        return True
    except SyntaxError:
        return False


def format_file_with_black(filename):
    file_path = Path(filename)

    # Read the original content of the file
    original_content = file_path.read_text()

    # Format the content using black
    try:
        formatted_content = black.format_str(original_content, mode=black.Mode())
    except black.NothingChanged:
        # If nothing changed, skip saving
        print(f"No changes needed for {filename}")
        return

    # Check if there are changes
    if original_content != formatted_content:
        # Save the formatted content back to the file
        file_path.write_text(formatted_content)
        print(f"Formatted and saved {filename}")
    else:
        print(f"No changes detected in {filename}")


def format_python_file(file_path: str, applyfmt=False):
    """
    Formats the docstrings in the given python file.
    :param <str> file_path: The path to the python file.
    """
    last_dot = file_path.rfind(".")
    out_path = file_path
    out_path = file_path[:last_dot] + TEMPFILE_SUFFIX + file_path[last_dot:]
    DSParser(file_path).write_formatted_file(file_path=out_path)
    # read the file content
    with open(out_path, "r") as file:
        content = file.read()

    if not check_valid_python_content(content):
        print(f"{colorama.Fore.RED}Error in formatting docstrings in {file_path}")
    else:
        # write the formatted content back to the file
        if applyfmt:
            format_file_with_black(out_path)
            with open(file_path, "w") as file:
                with open(out_path, "r") as out_file:
                    file.write(out_file.read())

    # delete the formatted file
    os.remove(out_path)


def format_python_files(root_dir: str = "./macrosynergy", applyfmt=False):
    """
    Formats the docstrings in all python files in the given directory.
    :param <str> root_dir: The root directory to search for python files.
    """
    all_files = glob.glob(f"{root_dir}/**/*.py", recursive=True)
    learning_files = []
    if os.path.exists(f"{root_dir}/learning"):
        learning_files = glob.glob(f"{root_dir}/learning/**/*.py", recursive=True)

    learning_files = list(map(os.path.normpath, learning_files))
    all_files = list(map(os.path.normpath, all_files))

    all_files = [file for file in all_files if file not in learning_files]
    for file in all_files:
        if (os.path.basename(file) in ["__init__.py", "compat.py"]) or (
            os.path.basename(file).split(".")[0].endswith(TEMPFILE_SUFFIX)
        ):
            continue

        print(f"Formatting docstrings in {file}")

        format_python_file(file, applyfmt=applyfmt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Format docstrings in python files.")
    parser.add_argument(
        "-d",
        "--dir",
        type=str,
        help="The root directory to search for python files.",
        # required=True,
        default="./macrosynergy",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run the doctests in this file.",
        # default=True,
    )

    args = parser.parse_args()

    if args.test:
        file_path = "macrosynergy/download/dataquery.py"
        format_python_file(file_path, applyfmt=True)
    else:
        format_python_files(args.dir, applyfmt=True)
