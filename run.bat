@echo off
REM Moon Generator GUI Launcher
REM Simple batch file to run the program without typing long commands

cd /d "%~dp0"
call .venv\Scripts\activate.bat
python -m moon_gen
pause
