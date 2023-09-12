import os
import shutil
import subprocess
import argparse

SOURCE_DIR = "./docs/build/"
RTD_OUTPUT_DIR = "./docs/"
CLEAN_OUTPUT_DIR = "./docs/docs.build/"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Post build script for documentation")
    parser.add_argument(
        "--rtd", action="store_true", help="Build for Read the Docs", default=True
    )
    parser.add_argument(
        "--clean", action="store_true", help="Build for clean", default=False
    )

    args = parser.parse_args()
    rtd_mode: bool = args.rtd
    clean_mode: bool = args.clean
    rtd_mode = not clean_mode

    outpit_dir: str = RTD_OUTPUT_DIR if rtd_mode else CLEAN_OUTPUT_DIR

    # if output/_build exists, remove it
    if os.path.exists(os.path.join(outpit_dir, "_build")):
        shutil.rmtree(os.path.join(outpit_dir, "_build"))

    shutil.copytree(SOURCE_DIR, outpit_dir, dirs_exist_ok=True)

    print(
        "Documentation is available at: \n\n\t\t",
        os.path.normpath(
            (os.path.abspath(os.path.join(outpit_dir, "_build/html/index.html")))
        ),
        "\n\n",
    )
