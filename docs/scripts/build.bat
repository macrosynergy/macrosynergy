python.exe .\docs\scripts\build_md.py
python.exe .\docs\scripts\build_jpb.py
jupyter-book build .\docs\build\
python.exe .\docs\scripts\post_build.py

@REM jupyter-book config sphinx macrosynergy
@REM .\scripts\build.bat