import os
import shutil
import subprocess

SOURCE_DIR = "./docs/build/"
OUTPUT_DIR = "./docs/"

# look in the source dir, there will be only one directory

for dir in os.listdir(SOURCE_DIR):
    if os.path.isdir(os.path.join(SOURCE_DIR, dir)):
        SOURCE_DIR = os.path.join(SOURCE_DIR, dir)
        break

# in source dir there is _build/ dir, copy it to output dir
# and remove source dir

# if output/_build exists, remove it
if os.path.exists(os.path.join(OUTPUT_DIR, "_build")):
    shutil.rmtree(os.path.join(OUTPUT_DIR, "_build"))

shutil.copytree(os.path.join(SOURCE_DIR, "_build"), os.path.join(OUTPUT_DIR, "_build"))
# shutil.rmtree(SOURCE_DIR)
# os.rmdir(os.path.dirname(SOURCE_DIR))

# print the abs path of  os.path.join(OUTPUT_DIR, "_build")
print(
    "Documentation is available at: \n\n\t\t",
    os.path.normpath(
        (os.path.abspath(os.path.join(OUTPUT_DIR, "_build/html/index.html")))
    ),
    "\n\n",
)
