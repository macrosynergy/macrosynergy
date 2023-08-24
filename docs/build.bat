python.exe .\docs\build_md.py
python.exe .\docs\build_jpb.py
cd .\docs\build\nb
jupyter-book build .

cd ..\..\..
@REM .\scripts\build.bat