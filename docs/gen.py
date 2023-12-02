# sphinx-apidoc -o docs/source/rsts -fMeT macrosynergy
# cd docs
# make.bat html

import os


makecom = "make" + ".bat" if os.name == "nt" else ""

rst_gen = "sphinx-apidoc -o docs/source/rsts -fMeT macrosynergy"

makecommand = f"{makecom} html"

os.system(rst_gen)
os.chdir("docs")
os.system(makecommand)
