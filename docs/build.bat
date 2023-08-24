python.exe .\docs\build_md.py
python.exe .\docs\build_jpb.py
cd .\docs\build\
jupyter-book build macrosynergy

cd ..\..\..
@REM .\scripts\build.bat