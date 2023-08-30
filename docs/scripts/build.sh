python docs/scripts/build_md.py
python docs/scripts/build_jpb.py

cd ./docs/build/
jupyter-book build macrosynergy
jupyter-book config sphinx macrosynergy

cd ../..

python docs/scripts/post_build.py

