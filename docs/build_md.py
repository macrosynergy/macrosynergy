import ast
import os
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings
import shutil
import argparse
import mdformat

SOURCE_DIR = "macrosynergy"
OUTPUT_DIR = "docs/build/md/"


class DocstringMethods:
    def __init__(self, *args, **kwargs):
        pass

    @staticmethod
    def markdown_format(docstring: str) -> str:
        """
        Formats a docstring to be written to markdown.
        """
        if not docstring:
            return ""

        options: Dict[str, Any] = {
            "number": True,
            "code_blocks": True,
            "tables": True,
            "footnotes": True,
            "strikethrough": True,
            "autolinks": True,
            "wrap": 90,
            "bullet": "-",
        }

        return mdformat.text(docstring, options=options)

    @staticmethod
    def format_parameters(docstring: str) -> str:
        """
        Formats the parameters section of a docstring.
        """
        if not docstring:
            return ""

        lines = docstring.split("\n")
        formatted_lines = []

        for il, line in enumerate(lines):
            try:
                kws: List[str] = ["param", "raises", "return", "yield"]
                kw: str = [f":{kw}" for kw in kws if line.startswith(f":{kw}")]
                if kw:
                    kw = kw[0]
                    # get the index of the first colon after the keyword
                    colon_index: int = line.index(":", len(kw))
                    # insert a '`' before the colon
                    line = "`" + line[:colon_index] + "`" + line[colon_index:] + ":"
                formatted_lines.append(line)
            except Exception as exc:
                e_str: str = f"Parsing error on line {il}: {line}, {exc}"
                raise Exception(e_str) from exc

        return DocstringMethods.markdown_format("\n".join(formatted_lines))


def extract_docstrings(source: str) -> Dict[str, Union[str, Dict[str, Any]]]:
    """
    Extracts docstrings from the given Python source code and returns a structured dictionary.
    """
    tree = ast.parse(source)
    structure = {"<module>": ast.get_docstring(tree), "classes": {}, "functions": {}}

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            class_methods = {}
            for child in node.body:
                if isinstance(
                    child, (ast.FunctionDef, ast.AsyncFunctionDef)
                ) and ast.get_docstring(child):
                    class_methods[child.name] = ast.get_docstring(child)

            structure["classes"][node.name] = {
                "doc": ast.get_docstring(node),
                "methods": class_methods,
            }
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Ensure it's not nested inside classes (top-level function)
            if not any(isinstance(parent, ast.ClassDef) for parent in ast.walk(node)):
                structure["functions"][node.name] = ast.get_docstring(node)

    return structure


def process_file(filepath: str, output_directory: str) -> bool:
    """
    Processes a single file to extract docstrings and write to markdown.
    """
    with open(filepath, "r", encoding="utf8") as f:
        source = f.read()

    structure = extract_docstrings(source=source)

    if structure:
        relative_path = os.path.relpath(filepath).replace(".py", ".md")
        output_path = os.path.join(output_directory, relative_path)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        output_str: str = ""
        output_str += f"# {relative_path}\n\n{structure['<module>']}\n\n"
        for class_name, class_info in structure["classes"].items():
            output_str += f"## `{class_name}`\n\n{class_info['doc']}\n\n"
            for method_name, method_doc in class_info["methods"].items():
                output_str += f"### `{class_name}.{method_name}`\n\n{method_doc}\n\n"

        for function_name, function_doc in structure["functions"].items():
            output_str += f"## `{function_name}`\n\n{function_doc}\n\n"

        output_str = DocstringMethods.format_parameters(docstring=output_str)

        with open(output_path, "w") as f:
            f.write(output_str)

    return bool(structure["classes"] or structure["functions"] or structure["<module>"])


def process_directory(
    input_directory: str, output_directory: str, skip_files: Optional[List[str]] = None
):
    """
    Processes an entire directory, extracting docstrings from each Python file.
    """

    for root, _, files in os.walk(input_directory):
        for file in files:
            if skip_files and file in skip_files:
                continue
            if file.endswith(".py"):
                if not process_file(
                    filepath=os.path.join(root, file), output_directory=output_directory
                ):
                    warnings.warn(
                        "Could not process "
                        f"{os.path.abspath(os.path.join(root, file))}.",
                        RuntimeWarning,
                    )
            if file.endswith(".md"):
                # copy the file to the output directory
                # create a var outputdir which is the relative path of the file in the input directory concatenated with the output directory
                outputdir: str = os.path.join(
                    output_directory,
                    os.path.relpath(root, os.path.dirname(input_directory)),
                )
                shutil.copy(
                    src=os.path.join(root, file),
                    dst=os.path.join(outputdir, file),
                )


def driver(readme: str, input_directory: str, output_directory: str):
    """
    Driver function for the script.
    """

    if os.path.exists(output_directory):
        shutil.rmtree(output_directory)
    os.makedirs(output_directory)

    shutil.copy(src=readme, dst=output_directory)

    process_directory(
        input_directory=input_directory,
        output_directory=output_directory,
        skip_files=["__init__.py", "version.py"],
    )
    dirs_found: List[str] = [
        os.path.normpath(os.path.join(output_directory, d + "/"))
        for d in os.listdir(output_directory)
        if os.path.isdir(d)
    ]
    if dirs_found:
        # assert tjhere is only one directory
        assert len(dirs_found) == 1
        # move the readme into the first directory
        shutil.move(os.path.join(output_directory, readme), dirs_found[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build markdown docs from source code")

    parser.add_argument(
        "-s",
        "--source",
        type=str,
        default=SOURCE_DIR,
        help="Path to the Python package directory",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=OUTPUT_DIR,
        help="Path to the output directory",
    )

    parser.add_argument(
        "--readme",
        type=str,
        default="README.md",
        help="Path to the README file",
    )

    args = parser.parse_args()

    if not os.path.exists(args.source):
        raise ValueError(f"Input directory {args.source} does not exist.")

    source_dir = os.path.abspath(os.path.normpath(os.path.expanduser(args.source)))
    output_dir = os.path.abspath(os.path.normpath(os.path.expanduser(args.output)))

    driver(input_directory=source_dir, output_directory=output_dir, readme=args.readme)
