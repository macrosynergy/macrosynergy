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
            # 'bullet': '-',
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
            except Exception as exc:
                e_str: str = f"Parsing error on line {il}: {line}, {exc}"
                raise Exception(e_str) from exc

        return DocstringMethods.markdown_format("\n".join(formatted_lines))


def extract_docstrings(source: str) -> Dict[str, str]:
    """
    Extracts docstrings from the given Python source code.
    """
    tree = ast.parse(source)
    docstrings = {}

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if ast.get_docstring(node):
                docstrings[node.name] = ast.get_docstring(node)
        elif isinstance(node, ast.Module):
            doc = ast.get_docstring(node)
            if doc:
                docstrings["<module>"] = doc

    return docstrings


def process_file(filepath: str, output_directory: str) -> bool:
    """
    Processes a single file to extract docstrings and write to markdown.
    """
    with open(filepath, "r", encoding="utf8") as f:
        source = f.read()

    docstrings = extract_docstrings(source=source)

    if docstrings:
        relative_path = os.path.relpath(filepath).replace(".py", ".md")
        output_path = os.path.join(output_directory, relative_path)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w") as f:
            for name, doc in docstrings.items():
                try:
                    docx: str = DocstringMethods.format_parameters(doc)
                except Exception as exc:
                    e_str: str = f"Error processing {name} in {filepath}"
                    raise Exception(e_str) from exc

                f.write(f"## {name}\n\n")
                f.write(f"{docx}\n\n")

    return bool(docstrings)


def process_directory(input_directory: str, output_directory: str):
    """
    Processes an entire directory, extracting docstrings from each Python file.
    """
    for root, _, files in os.walk(input_directory):
        for file in files:
            if file.endswith(".py"):
                if not process_file(
                    filepath=os.path.join(root, file), output_directory=output_directory
                ):
                    warnings.warn(f"Could not process {file}.", RuntimeWarning)


def driver(readme: str, input_directory: str, output_directory: str):
    """
    Driver function for the script.
    """
    # move the README to the output directory

    shutil.copy(src=readme, dst=output_directory)

    process_directory(
        input_directory=input_directory, output_directory=output_directory
    )


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

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    driver(input_directory=source_dir, output_directory=output_dir, readme=args.readme)
