import os

OUTPUT_DIR = "./docs/source/rsts"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# make command for windows or linux
makecommand = "make" + (".bat" if os.name == "nt" else "") + " html"

# command to generate rst files
rst_gen = f"sphinx-apidoc -o {OUTPUT_DIR} -fMeT macrosynergy"

os.system(rst_gen)  # generate rst files
current_dir = os.getcwd()  # get current directory
os.chdir("docs")  # change directory to docs
os.system(makecommand)  # make html
os.chdir(current_dir)  # change directory back to original
