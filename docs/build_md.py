import ast
import os
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings
import shutil
import argparse
import glob
import mdformat
from functools import wraps

SOURCE_DIR = "macrosynergy"
OUTPUT_DIR = "docs/build/"


def try_except(func):
    """
    Decorator to wrap a function in a try/except block.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except Exception as exc:
            # use glob to print the entire directory structure from .
            print(glob.glob("./*", recursive=True))
            raise exc(f"Error processing {args[0]}: {exc}.")

    return wrapper


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
                    line = "\n`" + line[:colon_index] + "`" + line[colon_index:]
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

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            class_methods = {}
            for child in node.body:
                if isinstance(
                    child, (ast.FunctionDef, ast.AsyncFunctionDef)
                ) and ast.get_docstring(child):
                    class_methods[child.name] = {
                        "doc": ast.get_docstring(child),
                        "parameters": child.args.args,
                    }
            structure["classes"][node.name] = {
                "doc": ast.get_docstring(node),
                "methods": class_methods,
            }
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            structure["functions"][node.name] = {
                "doc": ast.get_docstring(node),
                "parameters": node.args.args,
            }

    return structure


def process_file(filepath: str, output_directory: str) -> bool:
    """
    Processes a single file to extract docstrings and write to markdown.
    """
    with open(filepath, "r", encoding="utf8") as f:
        source = f.read()

    structure = extract_docstrings(source=source)

    LINE_SEPARATOR = "----------------\n\n"

    if structure:
        relative_path = os.path.relpath(filepath).replace(".py", ".md")
        output_path = os.path.join(output_directory, relative_path)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        output_str: str = ""
        mname = relative_path.replace(".md", "").replace("\\", ".").replace("/", ".")
        output_str += f"# **`{mname}`**\n\n{structure['<module>']}\n\n"
        output_str += LINE_SEPARATOR
        if structure["classes"]:
            output_str += "## **Classes**\n\n"
            output_str += LINE_SEPARATOR

        for class_name, class_info in structure["classes"].items():
            # add line separators
            output_str += f"## **`{class_name}`**\n\n{class_info['doc']}\n\n"
            output_str += LINE_SEPARATOR

            for method_name, method_info in class_info["methods"].items():
                params = ", ".join([arg.arg for arg in method_info["parameters"]])
                output_str += f"### `{class_name}.{method_name}()`\n\n"
                output_str += f"**`{class_name}.{method_name}({params})`**\n\n"
                output_str += f" {method_info['doc']}\n\n"
                output_str += LINE_SEPARATOR

        if structure["functions"]:
            output_str += "## **Functions**\n\n"
            output_str += LINE_SEPARATOR

        for function_name, function_info in structure["functions"].items():
            params = ", ".join([arg.arg for arg in function_info["parameters"]])
            output_str += f"## `{function_name}()`\n\n"
            output_str += f"**`{function_name}({params})`**\n\n"
            output_str += f"{function_info['doc']}\n\n"
            output_str += LINE_SEPARATOR

        # remove the last line separator
        output_str = "\n".join(output_str.split("\n")[:-2])

        output_str = DocstringMethods.format_parameters(docstring=output_str)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(output_str)

    return bool(structure["classes"] or structure["functions"] or structure["<module>"])


def create_subpackage_readmes(package_dir: str, root_package_dir: str) -> bool:
    """
    Creates a README.md with a contents hyper-linked table of contents,
    listing each module/sub-package in a directory.
    """
    # if normpath(abspath(package_dir)) == normpath(abspath(root_package_dir)):
    # return True
    px = lambda x: os.path.normpath(os.path.abspath(os.path.expanduser(x)))
    if px(package_dir) == px(root_package_dir):
        return True

    # get all .py and .md files using glob
    files: List[str] = glob.glob(os.path.join(package_dir, "**/*.py"), recursive=True)
    files += glob.glob(os.path.join(package_dir, "**/*.md"), recursive=True)

    # get the relative path of each file
    rel_paths: List[str] = [
        os.path.relpath(file, root_package_dir).replace("\\", "/") for file in files
    ]
    # filter out __init__.py, pycache, and version.py
    filter_words: List[str] = ["__init__.py", "version.py", "__pycache__"]
    files: List[str] = [
        file for file in files if not any(word in file for word in filter_words)
    ]
    # create a list of the subpackages
    subpackages: List[str] = [
        file for file in files if os.path.isdir(os.path.join(package_dir, file))
    ]
    # create a list of the modules
    modules: List[str] = [
        file for file in files if os.path.isfile(os.path.join(package_dir, file))
    ]

    # create a readme.md with the subpackage name - relpath.replace("\\", "/").replace("/", ".").replace(".py", "")
    output_str: str = ""

    rpath: str = os.path.relpath(package_dir, root_package_dir)
    rpath = "`" + rpath + "`"
    output_str += "# " + rpath.replace("\\", "/").replace("/", ".").replace(".py", "")
    output_str += "\n\n"
    output_str += "## Contents\n\n"
    if subpackages:
        output_str += "### Subpackages\n\n"
        for subpackage in sorted(subpackages):
            output_str += f"- [{subpackage}](./{subpackage})\n"
    output_str += "\n"
    if modules:
        # remove README.md from modules
        modules = [
            module for module in modules if os.path.basename(module) != "README.md"
        ]
        output_str += "### Modules\n\n"
        for module in sorted(modules):
            mrpath: str = os.path.relpath(
                os.path.join(package_dir, module), root_package_dir
            ).replace("\\", "/")
            mrname: str = mrpath.split(".")[0].replace("/", ".")
            output_str += f"- [{mrname}](./{os.path.basename(mrpath)})\n"

    if "README.md" not in os.listdir(package_dir):
        with open(os.path.join(package_dir, "README.md"), "w") as f:
            output_str = DocstringMethods.markdown_format(docstring=output_str)
            f.write(output_str)

    else:
        output_str = "\n".join(output_str.split("\n")[1:])
        with open(os.path.join(package_dir, "README.md"), "r", encoding="utf8") as f:
            lines: List[str] = f.readlines()
            for il, line in enumerate(lines):
                if line.strip():
                    break
            # insert the output_str after the first line
            lines.insert(il + 1, output_str)
            # write the lines back to the file
        with open(os.path.join(package_dir, "README.md"), "w", encoding="utf8") as f:
            output_str = "\n".join(lines)
            output_str = DocstringMethods.markdown_format(docstring=output_str)
            f.write(output_str)

    return True


@try_except
def process_directory(
    input_directory: str,
    output_directory: str,
    skip_files: Optional[List[str]] = None,
    readme: str = "README.md",
):
    """
    Processes an entire package directory, extracting docstrings from each Python file.
    """

    for root, _, files in os.walk(input_directory):
        # if the folder is pycache, continue
        if os.path.basename(root) == "__pycache__":
            continue

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
                    os.path.relpath(
                        os.path.abspath(root), os.path.dirname(input_directory)
                    ),
                )
                # if __pycache
                shutil.copy(
                    src=os.path.join(root, os.path.basename(file)),
                    dst=os.path.join(outputdir, os.path.basename(file)),
                )

    # move the package readme to the package directory
    # get the name of the package dir by checling the only dir in the output directory
    package_dir: List[str] = [
        os.path.join(output_directory, folder)
        for folder in os.listdir(output_directory)
        if os.path.isdir(os.path.join(output_directory, folder))
    ]
    # assert there is only one package dir
    assert len(package_dir) == 1
    shutil.move(
        src=os.path.join(output_directory, readme),
        dst=os.path.join(package_dir[0], readme),
    )

    # first get all the subdirectories at any level
    subdirectories: List[str] = glob.glob(
        os.path.join(output_directory, "**/"), recursive=True
    )
    # filter by isdir
    subdirectories: List[str] = [
        os.path.normpath(d) for d in subdirectories if os.path.isdir(d)
    ]

    subdirectories.remove(os.path.normpath(output_directory))
    # remove the root package dir
    subdirectories = sorted(subdirectories)
    subdirectories.remove(os.path.normpath(package_dir[0]))
    # go to each dir, and see if there is a readme.md
    for dirx in subdirectories:
        if os.path.basename(dirx) == "__pycache__":
            continue

        if not create_subpackage_readmes(
            package_dir=dirx, root_package_dir=output_directory
        ):
            warnings.warn(
                "Could not create README.md for "
                f"{os.path.abspath(os.path.join(root, dirx))}.",
                RuntimeWarning,
            )
        else:
            shutil.move(
                os.path.join(dirx, "README.md"),
                os.path.join(dirx, "index.md"),
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
        readme=readme,
        skip_files=["__init__.py", "version.py"],
    )
    dirs_found: List[str] = [
        os.path.normpath(os.path.join(output_directory, d + "/"))
        for d in os.listdir(output_directory)
        if os.path.isdir(d)
    ]
    # if dirs_found:
    #     # assert tjhere is only one directory
    #     assert len(dirs_found) == 1
    #     # move the readme into the first directory
    #     shutil.move(os.path.join(output_directory, readme), dirs_found[0])


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
    try:
        driver(
            input_directory=source_dir, output_directory=output_dir, readme=args.readme
        )
    except Exception as exc:
        # print the ls -R output
        print(os.system(f"ls -R {source_dir}"))
        raise exc
