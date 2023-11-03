import ast
import os
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings
import shutil
import argparse
import glob
import mdformat
from functools import wraps
import fnmatch

SOURCE_DIR = "macrosynergy"
STATIC_SOURCE_DIR = "docs/static/"
OUTPUT_DIR = "docs/build/"


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

        # remove any docstring directives
        # look for lines where line.strip() starts with "::docs::" and ends with "::"
        lines: List[str] = docstring.split("\n")
        for il, line in enumerate(lines):
            if line.strip().startswith("::docs::") and line.strip().endswith("::"):
                lines[il] = "\n"

        docstring = "\n".join(lines)

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
        formatted_lines: List[str] = []

        for il, line in enumerate(lines):
            try:
                kws: List[str] = ["param", "raises", "return", "yield"]
                kw: str = [f":{kw}" for kw in kws if line.startswith(f":{kw}")]
                if kw:
                    kw = kw[0]
                    # get the index of the first colon after the keyword
                    if ":" in line[len(kw) + 1 :]:
                        colon_index: int = line.index(":", len(kw))
                    else:
                        colon_index: int = len(line)
                        line += ":"
                    # insert a '`' before the colon
                    line = "\n`" + line[:colon_index] + "`" + line[colon_index:]
                formatted_lines.append(line)
            except Exception as exc:
                e_str: str = f"Parsing error on line {il}: {line}, {exc}"
                raise Exception(e_str) from exc

        return DocstringMethods.markdown_format("\n".join(formatted_lines))

    @staticmethod
    def sort_docstrings(
        module_docstrings: Dict[str, Union[str, Dict[str, Any]]],
        directives: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        # sort the module_docstrings['classes'] and module_docstrings['functions'] by key
        if module_docstrings["classes"]:
            module_docstrings["classes"] = dict(
                sorted(module_docstrings["classes"].items(), key=lambda x: x[0])
            )

        if module_docstrings["functions"]:
            module_docstrings["functions"] = dict(
                sorted(module_docstrings["functions"].items(), key=lambda x: x[0])
            )

        if module_docstrings["functions"]:
            underscore_functions: Dict[str, Any] = {
                k: v
                for k, v in module_docstrings["functions"].items()
                if k.startswith("_")
            }
            # sort the underscore_functions by key
            underscore_functions = dict(
                sorted(underscore_functions.items(), key=lambda x: x[0])
            )
            # remove the underscore_functions from module_docstrings["functions"], and add them back at the end
            module_docstrings["functions"] = {
                k: v
                for k, v in module_docstrings["functions"].items()
                if not k in underscore_functions
            }
            for k, v in underscore_functions.items():
                module_docstrings["functions"][k] = v

        output: Dict[str, Union[dict, str]] = {
            "classes": {},
            "functions": {},
            "<module>": "",
        }

        sorted_first: Dict[str, Union[dict, str]] = output.copy()
        sorted_last: Dict[str, Union[dict, str]] = output.copy()

        for doctype in ["classes", "functions"]:
            if doctype in module_docstrings and module_docstrings[doctype]:
                for name, doc in module_docstrings[doctype].items():
                    if name in directives and directives[name]:
                        # if there is a sort_first directive then move the docstring to the top
                        if "sort_first" in directives[name]:
                            sorted_first[doctype][name] = doc
                        elif "sort_last" in directives[name]:
                            sorted_last[doctype][name] = doc

            # reverse the order of the sorted_last dict
            sorted_last[doctype]: Dict[str, Any] = {
                k: sorted_last[doctype][k]
                for k in list(sorted_last[doctype].keys())[::-1]
            }

            output[doctype] = {
                **sorted_first[doctype],  # if sorted_first[doctype] else {},
                **module_docstrings[doctype],  # if module_docstrings[doctype] else {},
                **sorted_last[doctype],  # if sorted_last[doctype] else {},
            }

        output["<module>"] = module_docstrings["<module>"]

        return output

    @staticmethod
    def get_directives(docstring: str) -> Dict[str, Dict[str, Any]]:
        """
        Returns the docs-directives and the docs-flags from the docstring.
        Looks for lines where line.strip() starts with "::docs::" and ends with "::".
        """
        # look for lines where line.strip() starts with "::docs::" and ends with "::"
        lines: List[str] = docstring.split("\n")
        directives: Dict[str, Dict[str, str]] = {}
        for line in lines:
            if line.strip().startswith("::docs::") and line.strip().endswith("::"):
                directive: List[str] = line.strip().replace("::docs::", "").split("::")
                # remove any empty strings
                directive = [d.strip() for d in directive if d.strip()]
                if len(directive) != 2:
                    raise ValueError(f"Invalid directive: {line}")

                if directive[0] not in directives:
                    directives[directive[0]] = {}

                directives[directive[0]][directive[1]] = True

        return directives


def extract_docstrings(source: str) -> Dict[str, Union[str, Dict[str, Any]]]:
    """
    Extracts docstrings from the given Python source code and returns a structured dictionary.
    """
    tree = ast.parse(source)

    directives: Optional[dict] = None
    if tree is not None:
        directives: Dict[str, Dict[str, Any]] = DocstringMethods.get_directives(
            docstring=source
        )
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

    # pass the dict to the sort_docstrings method
    structure = DocstringMethods.sort_docstrings(
        module_docstrings=structure, directives=directives
    )

    return structure


def process_file(filepath: str, output_directory: str) -> bool:
    """
    Processes a single file to extract docstrings and write to markdown.
    """
    with open(filepath, "r", encoding="utf8") as f:
        source = f.read()

    try:
        structure = extract_docstrings(source=source)
    except Exception as exc:
        warnings.warn(
            f"Could not extract docstrings from {os.path.abspath(filepath)}: {exc}",
            RuntimeWarning,
        )
        return False

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

    subpackages: List[str] = [
        os.path.relpath(file, root_package_dir).replace("\\", "/")
        for file in glob.glob(os.path.join(package_dir, "**/"), recursive=True)
        if os.path.isdir(file)
        and (
            os.path.normpath(os.path.abspath(file))
            != os.path.normpath(os.path.abspath(package_dir))
        )
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
            sreadme: str = os.path.join(
                os.path.relpath(subpackage, package_dir), "index.md"
            ).replace("\\", "/")
            spname: str = subpackage.replace("\\", "/").replace("/", ".")
            output_str += f"- [{spname}](./{sreadme})\n"
    output_str += "\n"
    if modules:
        # remove README.md from modules
        modules = [
            module for module in modules if os.path.basename(module) != "README.md"
        ]
        output_str += "### Modules\n\n"
        for module in sorted(modules):
            if not os.path.dirname(module) == package_dir:
                continue  # filtering for modules within subpackages within subpackages

            mrpath: str = os.path.relpath(
                os.path.join(package_dir, module), root_package_dir
            ).replace("\\", "/")
            mrname: str = mrpath.split(".")[0].replace("/", ".")
            mrpath = os.path.relpath(mrpath, package_dir).replace("\\", "/")
            output_str += f"- [{mrname}](./{mrpath})\n"

    if "README.md" not in os.listdir(package_dir):
        output_str = DocstringMethods.markdown_format(docstring=output_str)
        with open(os.path.join(package_dir, "README.md"), "w") as f:
            f.write(output_str)

    else:
        output_str = "\n".join(output_str.split("\n")[1:])
        with open(os.path.join(package_dir, "README.md"), "r", encoding="utf8") as f:
            lines: List[str] = f.readlines()
            for il, line in enumerate(lines):
                if line.strip():
                    break
            lines.insert(il + 1, output_str)

        output_str = "".join(lines)
        output_str = DocstringMethods.markdown_format(docstring=output_str)
        with open(os.path.join(package_dir, "README.md"), "w", encoding="utf8") as f:
            f.write(output_str)

    return True


def process_directory(
    input_directory: str,
    output_directory: str,
    skip_files: Optional[List[str]] = None,
    readme: str = "README.md",
):
    """
    Processes an entire package directory, extracting docstrings from each Python file.
    """

    python_files: List[str] = glob.glob(
        os.path.join(input_directory, "**/*.py"), recursive=True
    )
    md_files: List[str] = glob.glob(
        os.path.join(input_directory, "**/*.md"), recursive=True
    )
    # remove the ignore files
    if skip_files:
        # use fnmatch to match the patterns
        _filt = lambda file, patterns: not (
            any(fnmatch.fnmatch(file, pattern) for pattern in patterns)
            or (os.path.basename(file) in patterns)
            or any(pattern in os.path.basename(file) for pattern in patterns)
            or any(
                fnmatch.fnmatch(os.path.basename(file), pattern) for pattern in patterns
            )
        )

        python_files = [file for file in python_files if _filt(file, skip_files)]
        md_files = [file for file in md_files if _filt(file, skip_files)]
    # create the output directory
    os.makedirs(output_directory, exist_ok=True)

    # create output paths by attaching the relative path of each file to the output directory
    python_output_paths: List[str] = [
        os.path.join(
            output_directory,
            os.path.relpath(file, os.path.dirname(input_directory)).replace("\\", "/"),
        )
        for file in python_files
    ]

    md_output_paths: List[str] = [
        os.path.join(
            output_directory,
            os.path.relpath(file, os.path.dirname(input_directory)).replace("\\", "/"),
        )
        for file in md_files
    ]

    for file, output_path in zip(python_files, python_output_paths):
        if not process_file(filepath=file, output_directory=output_directory):
            warnings.warn(
                "Could not process " f"{os.path.abspath(file)}.",
                RuntimeWarning,
            )

    for file, output_path in zip(md_files, md_output_paths):
        # just copy the file
        shutil.copy(src=file, dst=output_path)

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
        dst=os.path.join(package_dir[0], "index.md"),
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
        # if fnmath to match the patterns then continue
        if skip_files:
            if any(fnmatch.fnmatch(dirx, pattern) for pattern in skip_files):
                continue

        if not create_subpackage_readmes(
            package_dir=dirx, root_package_dir=output_directory
        ):
            warnings.warn(
                "Could not create README.md for "
                f"{os.path.abspath(os.path.join(output_directory, dirx))}.",
                RuntimeWarning,
            )
        else:
            shutil.move(
                os.path.join(dirx, "README.md"),
                os.path.join(dirx, "index.md"),
            )


def modify_readme(readme: str) -> bool:
    # if the first element of the readme is an image then remove it
    lines: List[str]
    with open(readme, "r", encoding="utf8") as f:
        lines: List[str] = f.readlines()
        if lines[0].startswith("!"):
            lines: List[str] = lines[1:]

    # look for a line containing: "# Macrosynergy Quant Research"
    for il, line in enumerate(lines):
        if line.strip().startswith("# Macrosynergy Quant Research"):
            lines[il] = "# `</>` Package Documentation\n\n"
            break

    with open(readme, "w", encoding="utf8") as f:
        out_lines: str = "".join(lines)
        f.write(out_lines)
    return True


def copy_static_md(source_dir: str, destination_dir: str) -> bool:
    """
    Copies the static markdown files from the source directory to the destination directory.
    """
    # get all the markdown files in the source directory recursively as absolute paths
    md_files: List[str] = [
        os.path.abspath(file)
        for file in glob.glob(os.path.join(source_dir, "**/*.md"), recursive=True)
    ]

    # copy each file to the destination directory
    for file in md_files:
        # make sure the path exists
        dst: str = os.path.join(destination_dir, os.path.relpath(file, source_dir))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(src=file, dst=dst)

    return True


def driver(readme: str, input_directory: str, output_directory: str, static_dir: str):
    """
    Driver function for the script.
    """

    if os.path.exists(output_directory):
        shutil.rmtree(output_directory)
    os.makedirs(output_directory)

    shutil.copy(src=readme, dst=output_directory)
    modify_readme(readme=os.path.join(output_directory, readme))

    process_directory(
        input_directory=input_directory,
        output_directory=output_directory,
        readme=readme,
        skip_files=["__init__.py", "version.py", "*/__pycache__/*", "*.pyc"],
    )
    dirs_found: List[str] = [
        os.path.normpath(os.path.join(output_directory, d + "/"))
        for d in os.listdir(output_directory)
        if os.path.isdir(d)
    ]
    # assert there is only one package dir
    assert len(dirs_found) == 1, "Found more than one package directory."

    # copy the static markdown files
    copy_static_md(source_dir=static_dir, destination_dir=output_directory)


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

    driver(
        input_directory=source_dir,
        output_directory=output_dir,
        readme=args.readme,
        static_dir=STATIC_SOURCE_DIR,
    )
