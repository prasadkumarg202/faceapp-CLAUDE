@echo off
rem Launches the Deep Face Net GUI.
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Python environment not found at .venv - run setup first.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" run.py %*
if errorlevel 1 pause
