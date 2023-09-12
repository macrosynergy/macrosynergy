#!/bin/bash

flag=$1

if [ "$flag" == "--clean" ]; then
    flag="--clean"
else
    flag=""
fi

python.exe ./docs/scripts/build_md.py
python.exe ./docs/scripts/build_jpb.py
jupyter-book build ./docs/build/

# python.exe ./docs/scripts/post_build.py

if [ "$flag" == "--clean" ]; then
    python.exe ./docs/scripts/post_build.py --clean
else
    python.exe ./docs/scripts/post_build.py
fi

# jupyter-book config sphinx macrosynergy
# ./scripts/build.sh
