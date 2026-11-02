@echo off
setlocal
if not exist "%~dp0.venv\Scripts\pythonw.exe" (
  echo BadmintonAI needs its local Python environment first. See README.md installation instructions.
  pause
  exit /b 1
)
start "BadmintonAI" "%~dp0.venv\Scripts\pythonw.exe" "%~dp0academy_app.py"
