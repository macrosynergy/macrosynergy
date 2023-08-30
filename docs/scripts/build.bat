python.exe .\docs\build_md.py
python.exe .\docs\build_jpb.py
cd .\docs\build\
jupyter-book build macrosynergy
jupyter-book config sphinx macrosynergy
cd ..\..

python.exe .\docs\post_build.py

@REM .\scripts\build.bat