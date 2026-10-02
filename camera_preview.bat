@echo off
rem Opens a live camera preview with face detection. Usage: camera_preview.bat [camera_index]
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Python environment not found at .venv - run setup first.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" camera_preview.py %*
if errorlevel 1 pause
