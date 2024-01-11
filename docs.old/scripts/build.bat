python.exe .\docs.old\scripts\build_md.py
python.exe .\docs.old\scripts\build_jpb.py
jupyter-book build .\docs.old\build\
python.exe .\docs.old\scripts\post_build.py

@REM jupyter-book config sphinx macrosynergy
@REM .\scripts\build.bat