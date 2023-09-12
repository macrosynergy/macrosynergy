@REM  if there is a flag for --clean, pass it to post_build.py

set "flag=%1"
if "%flag%"=="--clean" (
    set "flag=--clean"
) else (
    set "flag="
)


python.exe .\docs\scripts\build_md.py
python.exe .\docs\scripts\build_jpb.py
jupyter-book build .\docs\build\
@REM python.exe .\docs\scripts\post_build.py

if "%flag%"=="--clean" (
    python.exe .\docs\scripts\post_build.py --clean
) else (
    python.exe .\docs\scripts\post_build.py
)


@REM jupyter-book config sphinx macrosynergy
@REM .\scripts\build.bat