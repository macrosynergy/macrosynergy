import os
import shutil
import argparse

parser = argparse.ArgumentParser(description="Generate documentation.")
parser.add_argument(
    "--show",
    action="store_true",
    help="Show documentation in browser after generation.",
    default=False,
)

OUTPUT_DIR = "./docs/source/gen_rsts"
README = "./README.md"
args = parser.parse_args()

SHOW = args.show

#####################

os.makedirs(OUTPUT_DIR, exist_ok=True)

# make command for windows or linux
makescript = "make" + (".bat" if os.name == "nt" else "")
makehtml = makescript + " html"
makeclean = makescript + " clean"


# command to generate rst files
rst_gen = f"sphinx-apidoc -o {OUTPUT_DIR} -fMeT macrosynergy"

os.system(rst_gen)

# copy readme to rst
shutil.copy(README, OUTPUT_DIR)

# get current directory
current_dir = os.getcwd()

# change directory to docs
os.chdir("./docs")

# clean docs
os.system(makeclean)
# make html
os.system(makehtml)

# change directory back to original
os.chdir(current_dir)

# print success message with path to index.html

print(f"Documentation generated successfully.")
print(
    "Paste the following path into your browser to view:\n"
    f"\t\t file:///{os.path.abspath('docs/build/html/index.html')}"
)

if SHOW:
    os.system(f"start docs/build/html/index.html")
