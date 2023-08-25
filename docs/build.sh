python docs/build_md.py
python docs/build_jpb.py

cd ./docs/build/
jupyter-book build macrosynergy
jupyter-book config sphinx macrosynergy

cd ../../..