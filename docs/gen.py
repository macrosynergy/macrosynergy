import os
import shutil
import argparse


def generate_rsts(output_dir: str, package: str = "macrosynergy"):
    """
    Calls `sphinx-apidoc` to generate rst files for all modules in `package`.
    """
    rst_gen = f"sphinx-apidoc -o {output_dir} -fMeT {package}"
    os.system(rst_gen)


def copy_readme(output_dir: str, readme: str):
    """
    Copies README.md to `output_dir`.
    """
    shutil.copy(readme, output_dir)


def make_docs(docs_dir: str = "./docs", show: bool = False):
    """
    Calls `make html` in `docs_dir` (makefile from sphinx-quickstart).
    """
    makescript = "make" + (".bat" if os.name == "nt" else "")
    makehtml = makescript + " html"
    makeclean = makescript + " clean"
    current_dir: str = os.getcwd()
    os.chdir(docs_dir)
    os.system(makeclean)
    os.system(makehtml)
    os.chdir(current_dir)
    print(f"Documentation generated successfully.")
    print(
        "Paste the following path into your browser to view:\n\t\t "
        f"file:///{os.path.abspath('docs/build/html/index.html')}"
    )

    if show:
        os.system(f"start docs/build/html/index.html")


def main():
    """
    Complete build sequence.
    """
    parser = argparse.ArgumentParser(description="Generate documentation.")
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show documentation in browser after generation.",
        default=False,
    )
    # build
    parser.add_argument(
        "--build",
        action="store_true",
        help="Build documentation.",
        default=False,
    )

    args = parser.parse_args()

    SHOW = args.show
    BUILD = args.build
    OUTPUT_DIR = "./docs/source/gen_rsts"
    README = "./README.md"

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    generate_rsts(output_dir=OUTPUT_DIR)

    copy_readme(output_dir=OUTPUT_DIR, readme=README)

    if BUILD:
        make_docs(show=SHOW)


if __name__ == "__main__":
    main()
